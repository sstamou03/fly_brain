"""Event-driven leaky integrate-and-fire simulator for the whole male CNS.

Neuron/synapse model and every constant are taken from Shiu et al., Nature 2024
(github.com/philshiu/Drosophila_brain_model, model.py). What differs is only the engine:
all trials run as one batch in torch, and synaptic input is delivered event-driven
(only the columns of neurons that actually spiked are touched), so no C++ toolchain is needed.

Per 0.1 ms step, in Brian2's order:
  1. integrate v, g of non-refractory neurons (exact solution of the linear ODEs)
  2. threshold -> spikes
  3. deliver spikes emitted t_dly ago (g += w) and Poisson stimulus kicks (v += w_syn*f_poi)
  4. reset spikers (v = v_rst, g = 0) and start their refractory period
"""
import math
import time

import numpy as np
import torch

PARAMS = {
    "dt": 0.1,       # ms
    "t_run": 1000.0,  # ms
    "n_run": 30,     # trials
    "v_0": -52.0,    # mV resting
    "v_rst": -52.0,  # mV reset
    "v_th": -45.0,   # mV threshold
    "t_mbr": 20.0,   # ms membrane time constant
    "tau": 5.0,      # ms synaptic time constant
    "t_rfc": 2.2,    # ms refractory period
    "t_dly": 1.8,    # ms synaptic delay
    "w_syn": 0.275,  # mV per synapse
    "f_poi": 250,    # Poisson kick = w_syn * f_poi (always drives a spike)
}

# Shiu's w_syn was tuned on FlyWire (~50M synapses); the male CNS has 124M. calibrate.py sweeps a
# scale on it: bitter circuits ignite between 0.50 and 0.55 (spikes/s x6), sugar -> MN9 is silent
# below 0.45. 0.5 is the largest scale that keeps bitter bounded.
W_SYN_MALE_CNS = PARAMS["w_syn"] * 0.5


class Brain:
    """Connectome as CSC: column j lists the post-synaptic partners of neuron j."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.body_ids = z["body_ids"]
        self.col_ptr = torch.from_numpy(z["col_ptr"].astype(np.int64))
        self.post = torch.from_numpy(z["post"].astype(np.int64))
        self.weight = torch.from_numpy(z["weight"].astype(np.float32))  # signed synapse counts
        self.n = len(self.body_ids)
        self.index = {int(b): i for i, b in enumerate(self.body_ids)}

    def idx(self, body_ids):
        return np.array([self.index[int(b)] for b in body_ids if int(b) in self.index], dtype=np.int64)


@torch.no_grad()
def simulate(brain, stim_idx, stim_hz, readout_idx=(), params=None, n_run=None, t_run=None,
             bin_ms=10.0, seed=0, silence_idx=(), progress=True):
    """Run n_run trials in one batch.

    stim_idx/stim_hz: neurons driven by Poisson input and their rates (Hz).
    readout_idx: neurons whose spike trains are kept in time bins.
    Returns dict(rate=(N,) mean Hz over trials, readout=(n_run, R, n_bins) spike counts).
    """
    p = dict(PARAMS, **(params or {}))
    B = n_run or p["n_run"]
    T = t_run or p["t_run"]
    N, dt = brain.n, p["dt"]
    steps = int(round(T / dt))
    delay = int(round(p["t_dly"] / dt))
    refr_steps = int(round(p["t_rfc"] / dt))
    gen = torch.Generator().manual_seed(seed)

    # exact update of u = v - v_0 and g over one step
    eg = math.exp(-dt / p["tau"])
    em = math.exp(-dt / p["t_mbr"])
    kg = p["tau"] / (p["tau"] - p["t_mbr"]) * (eg - em)
    u_th = p["v_th"] - p["v_0"]
    u_rst = p["v_rst"] - p["v_0"]
    kick = p["w_syn"] * p["f_poi"]

    u = torch.zeros(B * N)
    g = torch.zeros(B * N)
    refr_until = torch.zeros(B * N, dtype=torch.int32)
    counts = torch.zeros(B * N, dtype=torch.int32)

    silent = torch.zeros(N, dtype=torch.bool)
    if len(silence_idx):
        silent[torch.as_tensor(silence_idx)] = True
    silent = silent.repeat(B)

    stim_idx = torch.as_tensor(np.asarray(stim_idx, dtype=np.int64))
    stim_p = torch.as_tensor(np.asarray(stim_hz, dtype=np.float32)) * (dt / 1000.0)
    stim_flat = (torch.arange(B)[:, None] * N + stim_idx[None, :]).reshape(-1)
    stim_p = stim_p.repeat(B)

    readout_idx = np.asarray(readout_idx, dtype=np.int64)
    R, n_bins = len(readout_idx), int(math.ceil(T / bin_ms))
    readout_pos = torch.full((N,), -1, dtype=torch.int64)
    readout_pos[torch.as_tensor(readout_idx)] = torch.arange(R)
    readout = torch.zeros(B * R * n_bins, dtype=torch.int32)
    steps_per_bin = int(round(bin_ms / dt))

    queue = [torch.empty(0, dtype=torch.int64)] * delay  # ring buffer of spike indices
    w_syn = p["w_syn"]
    t0 = time.time()

    for t in range(steps):
        active = refr_until <= t
        u_new = u * em + g * kg
        torch.where(active, u_new, u, out=u)
        torch.where(active, g * eg, g, out=g)

        spk = torch.nonzero((u > u_th) & active & ~silent).squeeze(1)

        # deliver spikes emitted `delay` steps ago
        arriving = queue[t % delay]
        if arriving.numel():
            trial, pre = arriving // N, arriving % N
            starts = brain.col_ptr[pre]
            lens = brain.col_ptr[pre + 1] - starts
            total = int(lens.sum())
            if total:
                rep = torch.repeat_interleave
                pos = rep(starts - (torch.cumsum(lens, 0) - lens), lens) + torch.arange(total)
                g.index_add_(0, rep(trial, lens) * N + brain.post[pos], brain.weight[pos] * w_syn)

        # Poisson stimulus
        fire = torch.rand(stim_flat.numel(), generator=gen) < stim_p
        if fire.any():
            u.index_add_(0, stim_flat[fire], torch.full((int(fire.sum()),), kick))

        # reset + bookkeeping
        if spk.numel():
            u[spk] = u_rst
            g[spk] = 0.0
            refr_until[spk] = t + refr_steps
            counts[spk] += 1
            if R:
                rp = readout_pos[spk % N]
                keep = rp >= 0
                if keep.any():
                    flat = ((spk[keep] // N) * R + rp[keep]) * n_bins + t // steps_per_bin
                    readout.index_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
        queue[t % delay] = spk

        if progress and (t + 1) % (steps // 10) == 0:
            print(f"  {100 * (t + 1) // steps:3d}%  {time.time() - t0:6.1f}s  spikes/step={spk.numel()}", flush=True)

    rate = counts.view(B, N).float().mean(0) / (T / 1000.0)
    return {"rate": rate.numpy(), "readout": readout.view(B, R, n_bins).numpy(), "bin_ms": bin_ms,
            "n_run": B, "t_run": T, "seconds": time.time() - t0}
