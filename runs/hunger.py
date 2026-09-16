"""How hungry must the fly get before it drinks each cocktail?

A fed fly rejects every cocktail (bitter from alcohol, acid and bitters vetoes feeding). Hungry flies
accept bitter-laced sugar, so hunger h in [0, 1] scales the bitter channel by (1 - h). For each drink,
bisection finds the lowest hunger at which the feeding motor neuron (MN9_L) fires >= FIRE_HZ.
The fly's choice is the drink it accepts at the lowest hunger; ties go to the stronger response.

usage: python -m runs.hunger [--menu cocktails] [--trials 4] [--steps 4]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from brain.sim import W_SYN_MALE_CNS, Brain, simulate
from brain.taste import CHANNELS, MENUS, neuron_input, taste_matrix, taste_vector

ROOT = Path(__file__).resolve().parents[1]
FIRE_HZ = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--menu", default="cocktails", choices=sorted(MENUS))
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()

    brain = Brain(ROOT / "brain.npz")
    meta = pd.read_parquet(ROOT / "brain_meta.parquet")
    mn9_l = int(meta.set_index("instance")["idx"]["MN9_L"])
    grn_idx, M, _ = taste_matrix(meta)
    bitter = CHANNELS.index("bitter")

    def mn9_at(t, h):
        t = t.copy()
        t[bitter] *= 1.0 - h
        u = neuron_input(t, M)
        on = u >= 1.0
        r = simulate(brain, grn_idx[on], u[on], readout_idx=[mn9_l], n_run=args.trials, t_run=1000.0,
                     params={"w_syn": W_SYN_MALE_CNS}, progress=False)
        return float(r["readout"][:, 0].sum(1).mean())

    rows = []
    for name, x in MENUS[args.menu].items():
        t = taste_vector(x)
        tried = {1.0: mn9_at(t, 1.0)}
        print(f"== {name}  bitter {t[bitter]:.2f}  starving (h=1): {tried[1.0]:.1f} Hz", flush=True)
        if tried[1.0] < FIRE_HZ:
            rows.append({"drink": name, "hunger": None, "hz_at_threshold": tried[1.0], "tried": tried})
            continue
        lo, hi = 0.0, 1.0
        for _ in range(args.steps):
            mid = (lo + hi) / 2
            tried[mid] = mn9_at(t, mid)
            print(f"   h={mid:.3f}  {tried[mid]:.1f} Hz", flush=True)
            lo, hi = (lo, mid) if tried[mid] >= FIRE_HZ else (mid, hi)
        rows.append({"drink": name, "hunger": hi, "hz_at_threshold": tried[hi],
                     "tried": {f"{k:.4f}": v for k, v in sorted(tried.items())}})
        print(f"   -> drinks it from hunger {hi:.3f}", flush=True)

    rows.sort(key=lambda r: (r["hunger"] is None, r["hunger"] or 0, -r["hz_at_threshold"]))
    print("\nchoice order:", " > ".join(f"{r['drink']} ({'never' if r['hunger'] is None else round(r['hunger'], 3)})" for r in rows))
    (ROOT / "results" / f"hunger_{args.menu}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
