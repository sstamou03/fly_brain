"""Phase 3: the Fly Bar. Serve every drink to the brain and rank by MN9 (proboscis extension).

drink vector -> taste vector -> taste-neuron input -> whole-brain simulation -> MN9_L firing.
The drink touches legs, labellum and pharynx. All drinks share one random seed, so differences
come from the drink and not from noise.

usage: python bar.py            30 trials x 1000 ms per drink
       python bar.py --quick    4 trials x 100 ms
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sim import Brain, simulate
from taste import CHANNELS, DRINKS, INGREDIENTS, neuron_input, taste_matrix, taste_vector

ROOT = Path(__file__).parent


def main():
    quick = "--quick" in sys.argv
    n_run, t_run = (4, 100.0) if quick else (30, 1000.0)
    out = ROOT / "results"
    (out / "bar_rates").mkdir(parents=True, exist_ok=True)

    brain = Brain(ROOT / "brain.npz")
    meta = pd.read_parquet(ROOT / "brain_meta.parquet")
    # score on MN9_L: MN9_R is flagged "RT Hard to trace" and has 556 input synapses vs 6,012
    by_instance = meta.set_index("instance")["idx"]
    mn9 = np.array([by_instance["MN9_L"], by_instance["MN9_R"]])
    grn_idx, M, organ = taste_matrix(meta)
    print(f"taste neurons touching the drink: {len(grn_idx)} "
          f"({', '.join(f'{o} {int((organ == o).sum())}' for o in np.unique(organ))})")

    rows = []
    for name, x in DRINKS.items():
        t = taste_vector(x)
        u = neuron_input(t, M)
        on = u >= 1.0  # ignore channels that are only numerically non-zero
        print(f"\n== {name}  taste = [" + " ".join(f"{c} {v:.2f}" for c, v in zip(CHANNELS, t)) + "]")
        r = simulate(brain, grn_idx[on], u[on], readout_idx=mn9, n_run=n_run, t_run=t_run, progress=False)
        per_trial = r["readout"][:, 0].sum(1) / (t_run / 1000.0)  # MN9_L
        emoji, drink = name.split(" ", 1)
        slug = re.sub(r"\W+", "_", drink).strip("_")
        np.save(out / "bar_rates" / f"{slug}.npy", r["rate"].astype(np.float16))
        rows.append({
            "drink": drink, "emoji": emoji, "slug": slug,
            "mn9_hz": float(per_trial.mean()), "mn9_sem": float(per_trial.std(ddof=1) / np.sqrt(n_run)),
            "mn9_lr": r["rate"][mn9].round(2).tolist(),
            "active_neurons": int((r["rate"] > 0).sum()), "stimulated_neurons": int(on.sum()),
            "drink_vector": dict(zip(INGREDIENTS, x.tolist())),
            "taste_vector": dict(zip(CHANNELS, t.round(3).tolist())), "seconds": r["seconds"],
        })
        print(f"   MN9 {per_trial.mean():6.1f} ± {rows[-1]['mn9_sem']:.1f} Hz   active {rows[-1]['active_neurons']:,}   {r['seconds']:.0f}s")

    rows.sort(key=lambda row: -row["mn9_hz"])
    top = max(rows[0]["mn9_hz"], 1e-9)
    print("\n🪰  THE FLY BAR — ranked by proboscis extension drive (MN9)\n")
    for i, row in enumerate(rows, 1):
        bar = "█" * int(round(30 * row["mn9_hz"] / top))
        print(f"{i:2d}. {row['emoji']} {row['drink']:<18} {row['mn9_hz']:6.1f} ± {row['mn9_sem']:4.1f} Hz  {bar}")
    run = {"n_run": n_run, "t_run_ms": t_run, "dataset": "male-cns v1.0"}
    (out / ("bar_quick.json" if quick else "bar.json")).write_text(
        json.dumps({"run": run, "ranking": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
