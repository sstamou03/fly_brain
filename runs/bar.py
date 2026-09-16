"""Phase 3: the Fly Bar. Serve every drink to the brain and rank by MN9 (proboscis extension).

drink vector -> taste vector -> taste-neuron input -> whole-brain simulation -> MN9_L firing.
The drink touches legs, labellum and pharynx. All drinks share one random seed, so differences
come from the drink and not from noise. Synaptic weight: W_SYN_MALE_CNS (see calibrate.py).

usage: python -m runs.bar                              30 trials x 1000 ms, all drinks
       python -m runs.bar --trials 10 --only Κόλα,Espresso,Τσίπουρο,Νερό    pilot on a subset
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from brain.sim import PARAMS, W_SYN_MALE_CNS, Brain, simulate
from brain.taste import CHANNELS, INGREDIENTS, MENUS, neuron_input, taste_matrix, taste_vector

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--ms", type=float, default=1000.0)
    ap.add_argument("--menu", default="classic", choices=sorted(MENUS))
    ap.add_argument("--only", default="", help="comma-separated substrings of drink names")
    ap.add_argument("--scale", type=float, default=None, help="w_syn scale (default: W_SYN_MALE_CNS)")
    args = ap.parse_args()
    w_syn = PARAMS["w_syn"] * args.scale if args.scale else W_SYN_MALE_CNS
    bin_ms = 50.0
    early, late = slice(2, 6), slice(int((args.ms - 200) / bin_ms), None)  # 100-300 ms vs last 200 ms
    only = [s for s in args.only.split(",") if s]
    drinks = {k: v for k, v in MENUS[args.menu].items() if not only or any(s in k for s in only)}
    n_run, t_run = args.trials, args.ms

    out = ROOT / "results"
    (out / "bar_rates").mkdir(parents=True, exist_ok=True)
    brain = Brain(ROOT / "brain.npz")
    meta = pd.read_parquet(ROOT / "brain_meta.parquet")
    # score on MN9_L: MN9_R is flagged "RT Hard to trace" and has 556 input synapses vs 6,012
    by_instance = meta.set_index("instance")["idx"]
    mn9 = np.array([by_instance["MN9_L"], by_instance["MN9_R"]])
    grn_idx, M, organ = taste_matrix(meta)
    print(f"taste neurons touching the drink: {len(grn_idx)} "
          f"({', '.join(f'{o} {int((organ == o).sum())}' for o in np.unique(organ))})   "
          f"w_syn {w_syn:.4f} mV   {n_run} trials x {t_run:.0f} ms")

    rows = []
    for name, x in drinks.items():
        t = taste_vector(x)
        u = neuron_input(t, M)
        on = u >= 1.0  # ignore channels that are only numerically non-zero
        print(f"\n== {name}  taste = [" + " ".join(f"{c} {v:.2f}" for c, v in zip(CHANNELS, t)) + "]", flush=True)
        r = simulate(brain, grn_idx[on], u[on], readout_idx=mn9, n_run=n_run, t_run=t_run, bin_ms=bin_ms,
                     params={"w_syn": w_syn}, progress=False)
        pop = r["pop_hz"]
        per_trial = r["readout"][:, 0].sum(1) / (t_run / 1000.0)  # MN9_L
        rest = np.ones(brain.n, bool)
        rest[grn_idx[on]] = False
        emoji, drink = name.split(" ", 1)
        slug = re.sub(r"\W+", "_", drink).strip("_")
        np.save(out / "bar_rates" / f"{slug}.npy", r["rate"].astype(np.float16))
        rows.append({
            "drink": drink, "emoji": emoji, "slug": slug,
            "mn9_hz": float(per_trial.mean()), "mn9_sem": float(per_trial.std(ddof=1) / np.sqrt(n_run)),
            "mn9_lr": r["rate"][mn9].round(2).tolist(),
            "neurons_over_10hz": int((r["rate"][rest] > 10).sum()),
            "neurons_over_100hz": int((r["rate"][rest] > 100).sum()),
            "spikes_per_s": float(r["rate"][rest].sum()), "stimulated_neurons": int(on.sum()),
            "growth": float(pop[late].mean() / max(pop[early].mean(), 1.0)), "pop_hz": pop.round(0).tolist(),
            "drink_vector": dict(zip(INGREDIENTS, x.tolist())),
            "taste_vector": dict(zip(CHANNELS, t.round(3).tolist())), "seconds": r["seconds"],
        })
        x_ = rows[-1]
        print(f"   MN9 {x_['mn9_hz']:6.1f} ± {x_['mn9_sem']:.1f} Hz   >10Hz {x_['neurons_over_10hz']:,}   "
              f">100Hz {x_['neurons_over_100hz']:,}   spikes/s {x_['spikes_per_s']:,.0f}   growth x{x_['growth']:.1f}   "
              f"{r['seconds']:.0f}s", flush=True)

    rows.sort(key=lambda row: -row["mn9_hz"])
    top = max(rows[0]["mn9_hz"], 1e-9)
    print("\n🪰  THE FLY BAR — ranked by proboscis extension drive (MN9)\n")
    for i, row in enumerate(rows, 1):
        bar = "█" * int(round(30 * row["mn9_hz"] / top))
        print(f"{i:2d}. {row['emoji']} {row['drink']:<18} {row['mn9_hz']:6.1f} ± {row['mn9_sem']:4.1f} Hz  {bar}")
    run = {"n_run": n_run, "t_run_ms": t_run, "w_syn_mV": w_syn, "dataset": "male-cns v1.0",
           "menu": args.menu, "only": only}
    tag = ("" if args.menu == "classic" else f"_{args.menu}") + (f"_scale{args.scale}" if args.scale else "")
    (out / (f"bar_pilot{tag}.json" if only else f"bar{tag}.json")).write_text(
        json.dumps({"run": run, "ranking": rows}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
