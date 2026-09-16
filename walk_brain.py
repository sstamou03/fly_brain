"""Let the brain decide how the fly wanders on the counter -> app/public/data/walk.json.

The fly gets slowly changing, random sensations of the bar (ASSUMPTION: this is invented input, not
recorded): every SEG_MS a new random handful of visual projection neurons ("what she sees", already
processed by the optic lobes: LC, LPLC, LoVP...) and ascending neurons ("what her legs and body feel")
fires, biased to the left or right side. Raw photoreceptors and sensory neurons were tried first:
the optic lobes lit up but the walking neurons stayed silent, too many synapses away. The whole brain runs on that, and we read the descending neurons
that drive walking in real flies:
  turning  DNa01, DNa02 (steering; more activity on one side = turn to that side)
  forward  DNp09, DNg100 (= BDN2) and DNg97 (= oDN1), the forward-walking DNs of Sapkal et al. 2024,
           minus MDN (backward walking, "moonwalker")
Their activity becomes the app's `forward` (0..1) and `turn` (-1..1, positive = left) per bin.

usage: python walk_brain.py [--seconds 12] [--threads 4]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sim import W_SYN_MALE_CNS, Brain, simulate

ROOT = Path(__file__).parent
OUT = ROOT / "app" / "public" / "data" / "walk.json"
SEG_MS = 400.0
BIN_MS = 50.0
INPUT_SUPERCLASSES = ["visual_projection", "ascending_neuron"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=0.04, help="share of sensory neurons active per segment")
    ap.add_argument("--hz", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)

    brain = Brain(ROOT / "brain.npz")
    meta = pd.read_parquet(ROOT / "brain_meta.parquet")
    sensory = meta[meta["superclass"].isin(INPUT_SUPERCLASSES)]
    side = sensory["somaSide"].fillna("").to_numpy()
    idx = sensory["idx"].to_numpy()
    t_ms = args.seconds * 1000.0
    n_seg = int(np.ceil(t_ms / SEG_MS))

    rates = np.zeros((len(idx), n_seg), dtype=np.float32)
    bias = np.zeros(n_seg)
    for s in range(n_seg):
        bias[s] = rng.uniform(-1, 1)  # -1 = everything on the right, +1 = on the left
        p = np.full(len(idx), args.fraction)
        p[side == "L"] *= 1 + bias[s]
        p[side == "R"] *= 1 - bias[s]
        on = rng.random(len(idx)) < p
        rates[on, s] = args.hz * rng.uniform(0.5, 1.5, on.sum())

    by_instance = meta.groupby("instance")["idx"].apply(list).to_dict()
    groups = {
        "steer_L": by_instance.get("DNa01(VES006)_L", []) + by_instance.get("DNa02_L", []),
        "steer_R": by_instance.get("DNa01(VES006)_R", []) + by_instance.get("DNa02_R", []),
        "forward": sum((by_instance.get(k, []) for k in ["DNp09_L", "DNp09_R", "DNg100_L", "DNg100_R", "DNg97_L", "DNg97_R"]), []),
        "backward": by_instance.get("MDN_L", []) + by_instance.get("MDN_R", []),
    }
    readout = [i for g in groups.values() for i in g]
    print({k: len(v) for k, v in groups.items()}, f"sensory neurons {len(idx):,}, {n_seg} segments", flush=True)

    r = simulate(brain, idx, rates, readout_idx=readout, n_run=1, t_run=t_ms, bin_ms=BIN_MS,
                 params={"w_syn": W_SYN_MALE_CNS}, stim_seg_ms=SEG_MS, progress=True)
    hz = r["readout"][0] / (BIN_MS / 1000.0)  # (R, bins)
    pos = {n: i for i, n in enumerate(readout)}
    mean = {k: hz[[pos[i] for i in v]].mean(0) if v else np.zeros(hz.shape[1]) for k, v in groups.items()}

    smooth = lambda x: np.convolve(np.pad(x, 3, mode="edge"), np.ones(7) / 7, mode="valid")
    steer = smooth(mean["steer_L"] - mean["steer_R"])
    # remove the constant left/right offset (reconstruction asymmetry, like MN9_R's) so she doesn't circle;
    # the turns the brain makes around that baseline are kept
    steer -= np.median(steer)
    drive = smooth(mean["forward"] - 0.5 * mean["backward"])
    turn_scale = max(np.percentile(np.abs(steer), 90), 1.0)
    drive_scale = max(np.percentile(drive, 90), 1.0)
    turn = np.tanh(steer / turn_scale)
    forward = np.clip(drive / drive_scale, 0, 1)

    summary = {k: round(float(v.mean()), 2) for k, v in mean.items()}
    print("mean DN firing (Hz):", summary, flush=True)
    print(f"forward>0 in {np.mean(forward > 0.05):.0%} of bins, |turn|>0.2 in {np.mean(np.abs(turn) > 0.2):.0%}", flush=True)
    print(f"brain spikes/s (non-stimulated): {r['pop_hz'].mean():,.0f}", flush=True)
    steering = max(summary["steer_L"], summary["steer_R"])
    if steering < 0.5 or summary["forward"] < 0.5:
        raise SystemExit(f"walking neurons too quiet (steering {steering} Hz, forward {summary['forward']} Hz): "
                         "not writing walk.json, the app keeps its fallback wander")

    OUT.write_text(json.dumps({
        "binMs": BIN_MS, "forward": forward.round(3).tolist(), "turn": turn.round(3).tolist(),
        "source": "male-cns v1.0 descending neurons DNa01/DNa02 (steering), DNp09 (forward), MDN (backward); "
                  "random sensory input (assumption)",
        "dn_mean_hz": summary, "seconds": args.seconds,
    }), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
