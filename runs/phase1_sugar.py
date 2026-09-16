"""Phase 1: does the brain behave like a fly? Reproduce the Shiu et al. 2024 core result.

Expected: labellar sugar GRNs drive MN9 (proboscis extension), bitter GRNs alone do not,
and adding bitter on top of sugar suppresses MN9.

usage: python phase1_sugar.py            full run (30 trials x 1000 ms per condition)
       python phase1_sugar.py --quick    speed check (4 trials x 100 ms)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sim import Brain, simulate

ROOT = Path(__file__).parent
torch.set_num_threads(max(1, torch.get_num_threads()))


def main():
    quick = "--quick" in sys.argv
    n_run, t_run = (4, 100.0) if quick else (30, 1000.0)

    brain = Brain(ROOT / "brain.npz")
    meta = pd.read_parquet(ROOT / "brain_meta.parquet")
    by_type = meta.groupby("type")["idx"].apply(np.array).to_dict()
    pick = lambda types: np.concatenate([by_type[t] for t in types])

    sugar = pick(["LB3b", "LB3c"])
    bitter = pick(["LB1a", "LB1b", "LB1c", "LB1d"])
    mn9 = by_type["MN9"]
    print(f"{brain.n:,} neurons, {len(brain.post):,} edges | sugar GRNs {len(sugar)}, bitter GRNs {len(bitter)}, MN9 {len(mn9)}")

    conditions = {
        "nothing": ([], []),
        "sugar 200Hz": (sugar, [200.0] * len(sugar)),
        "bitter 150Hz": (bitter, [150.0] * len(bitter)),
        "sugar 200Hz + bitter 150Hz": (np.concatenate([sugar, bitter]), [200.0] * len(sugar) + [150.0] * len(bitter)),
    }
    results = {}
    for name, (idx, hz) in conditions.items():
        print(f"\n== {name}  ({n_run} trials x {t_run:.0f} ms)")
        r = simulate(brain, idx, hz, readout_idx=mn9, n_run=n_run, t_run=t_run)
        mn9_hz = r["rate"][mn9]
        n_active = int((r["rate"] > 0).sum())
        print(f"   MN9 L/R = {mn9_hz.round(1).tolist()} Hz   active neurons = {n_active:,}   {r['seconds']:.1f}s")
        results[name] = {"mn9_hz": mn9_hz.tolist(), "active_neurons": n_active, "seconds": r["seconds"]}

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / ("phase1_quick.json" if quick else "phase1.json")).write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
