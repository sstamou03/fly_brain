"""How much bitter does the fly tolerate in a sweet drink?

Sweet is fixed (default 0.80, like a piña colada); the bitter channel is stepped up. Every other
channel is 0, all organs touch the drink, w_syn = W_SYN_MALE_CNS. The point where MN9_L drops to 0
tells which drinks can score at all, since every alcoholic/acidic drink carries some bitter.

usage: python -m runs.bitter_tolerance [sweet] [--trials 6]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from brain.sim import W_SYN_MALE_CNS, Brain, simulate
from brain.taste import CHANNELS, neuron_input, taste_matrix

ROOT = Path(__file__).resolve().parents[1]
BITTER_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweet", type=float, nargs="?", default=0.80)
    ap.add_argument("--trials", type=int, default=6)
    args = ap.parse_args()

    brain = Brain(ROOT / "brain.npz")
    meta = pd.read_parquet(ROOT / "brain_meta.parquet")
    mn9_l = int(meta.set_index("instance")["idx"]["MN9_L"])
    grn_idx, M, _ = taste_matrix(meta)
    rows = []
    for bitter in BITTER_LEVELS:
        t = np.zeros(len(CHANNELS))
        t[CHANNELS.index("sweet")] = args.sweet
        t[CHANNELS.index("bitter")] = bitter
        u = neuron_input(t, M)
        on = u >= 1.0
        r = simulate(brain, grn_idx[on], u[on], readout_idx=[mn9_l], n_run=args.trials, t_run=1000.0,
                     bin_ms=50.0, params={"w_syn": W_SYN_MALE_CNS}, progress=False)
        per_trial = r["readout"][:, 0].sum(1)
        rest = np.ones(brain.n, bool)
        rest[grn_idx[on]] = False
        pop = r["pop_hz"]
        rows.append({"sweet": args.sweet, "bitter": bitter, "mn9_l_hz": float(per_trial.mean()),
                     "mn9_sem": float(per_trial.std(ddof=1) / np.sqrt(args.trials)),
                     "over_10hz": int((r["rate"][rest] > 10).sum()),
                     "growth": float(pop[-4:].mean() / max(pop[2:6].mean(), 1.0))})
        x = rows[-1]
        print(f"sweet {args.sweet:.2f}  bitter {bitter:.2f}  MN9_L {x['mn9_l_hz']:6.1f} ± {x['mn9_sem']:4.1f} Hz  "
              f">10Hz {x['over_10hz']:5,}  growth x{x['growth']:.1f}  {r['seconds']:.0f}s", flush=True)

    (ROOT / "results" / f"bitter_tolerance_sweet{args.sweet}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
