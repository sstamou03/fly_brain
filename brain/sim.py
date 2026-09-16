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
# scale on it with 1 s runs (bitter ignition builds up slowly, 300 ms runs hide it):
#   0.30-0.48 stable for sugar, bitter and sugar+bitter (late/early activity ~x1.0)
#   0.50      bitter x3.5, sugar+bitter x10 -> runaway
# Real drinks drive more taste neurons than the calibration stimuli, so 0.45 keeps a margin. At 0.45
# every pilot drink stays stable and orange juice drives MN9_L to 86 Hz (water, espresso: 0).
# FLYCNS (another male-CNS port of Shiu's model) needed 0.3 for stability, where feeding fails.
W_SYN_MALE_CNS = PARAMS["w_syn"] * 0.45


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
             bin_ms=10.0, seed=0, silence_idx=(), progress=True, stim_seg_ms=None):
    """Run n_run trials in one batch.

    stim_idx/stim_hz: neurons driven by Poisson input and their rates (Hz). stim_hz may be 2-D
    (n_stim, n_segments) with stim_seg_ms set: the rates then change every stim_seg_ms.
    readout_idx: neurons whose spike trains are kept in time bins.
    Returns dict(rate=(N,) mean Hz over trials, readout=(n_run, R, n_bins) spike counts,
                 pop_hz=(n_bins,) spikes/s of all non-stimulated neurons, per trial,
                 readout_v=(n_run, R, n_bins) mean membrane potential in mV, sampled after integration).
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
    rates = np.asarray(stim_hz, dtype=np.float32)
    segmented = rates.ndim == 2
    if segmented:
        seg_rates = torch.as_tensor(rates) * (dt / 1000.0)  # (n_stim, n_segments)
        steps_per_seg = int(round(stim_seg_ms / dt))
        stim_p = seg_rates[:, 0].repeat(B)
    else:
        stim_p = torch.as_tensor(rates * (dt / 1000.0)).repeat(B)
    stim_flat = (torch.arange(B)[:, None] * N + stim_idx[None, :]).reshape(-1)
    is_stim = torch.zeros(B * N, dtype=torch.bool)
    is_stim[stim_flat] = True

    readout_idx = np.asarray(readout_idx, dtype=np.int64)
    R, n_bins = len(readout_idx), int(math.ceil(T / bin_ms))
    readout_pos = torch.full((N,), -1, dtype=torch.int64)
    readout_pos[torch.as_tensor(readout_idx)] = torch.arange(R)
    readout = torch.zeros(B * R * n_bins, dtype=torch.int32)
    readout_flat = (torch.arange(B)[:, None] * N + torch.as_tensor(readout_idx)[None, :]).reshape(-1)
    readout_u = torch.zeros(B * R, n_bins, dtype=torch.float64)  # summed membrane potential per bin
    pop = np.zeros(n_bins, dtype=np.int64)  # spikes of non-stimulated neurons per bin, all trials
    steps_per_bin = int(round(bin_ms / dt))

    queue = [torch.empty(0, dtype=torch.int64)] * delay  # ring buffer of spike indices
    w_syn = p["w_syn"]
    t0 = time.time()

    for t in range(steps):
        active = refr_until <= t
        u_new = u * em + g * kg
        torch.where(active, u_new, u, out=u)
        torch.where(active, g * eg, g, out=g)
        if R:
            readout_u[:, t // steps_per_bin] += u[readout_flat]

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
        if segmented and t % steps_per_seg == 0:
            stim_p = seg_rates[:, min(t // steps_per_seg, seg_rates.shape[1] - 1)].repeat(B)
        fire = torch.rand(stim_flat.numel(), generator=gen) < stim_p
        if fire.any():
            u.index_add_(0, stim_flat[fire], torch.full((int(fire.sum()),), kick))

        # reset + bookkeeping
        if spk.numel():
            u[spk] = u_rst
            g[spk] = 0.0
            refr_until[spk] = t + refr_steps
            counts[spk] += 1
            pop[t // steps_per_bin] += int((~is_stim[spk]).sum())
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
            "pop_hz": pop / (B * bin_ms / 1000.0),
            "readout_v": (readout_u / steps_per_bin + p["v_0"]).view(B, R, n_bins).float().numpy(),
            "n_run": B, "t_run": T, "seconds": time.time() - t0}
