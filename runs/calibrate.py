"""Calibrate w_syn for the male CNS.

Shiu et al. tuned w_syn = 0.275 mV on FlyWire (~50M synapses). The male CNS has 124M synapses
on a similar neuron count, so the same per-synapse weight drives the network into runaway.

Sweep a scale on w_syn and keep the largest one that still satisfies the Phase-1 behaviour:
  sugar -> MN9_L fires,  bitter alone -> MN9_L silent,  sugar + bitter -> suppressed,
with activity staying bounded for the WHOLE run. Bitter ignition builds up slowly: a first sweep
with 300 ms runs picked 0.5, but 1 s drinks at 0.5 still ignited. So runs are 1 s, and `growth`
compares population activity in the last 200 ms with 100-300 ms (≈1 = steady, >>1 = igniting).

usage: python -m runs.calibrate 0.3 0.4 0.45 0.5 --ms 1000 --trials 4
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from brain.sim import PARAMS, Brain, simulate

ROOT = Path(__file__).resolve().parents[1]
BIN_MS = 50.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scales", type=float, nargs="+")
    ap.add_argument("--ms", type=float, default=1000.0)
    ap.add_argument("--trials", type=int, default=4)
    args = ap.parse_args()

    brain = Brain(ROOT / "brain.npz")
    meta = pd.read_parquet(ROOT / "brain_meta.parquet")
    by_type = meta.groupby("type")["idx"].apply(np.array).to_dict()
    mn9_l = int(meta.set_index("instance")["idx"]["MN9_L"])
    sugar = np.concatenate([by_type["LB3b"], by_type["LB3c"]])
    bitter = np.concatenate([by_type[t] for t in ["LB1a", "LB1b", "LB1c", "LB1d"]])
    rest = np.ones(brain.n, bool)
    rest[sugar] = rest[bitter] = False
    conditions = {
        "sugar": (sugar, [200.0] * len(sugar)),
        "bitter": (bitter, [150.0] * len(bitter)),
        "sugar+bitter": (np.concatenate([sugar, bitter]), [200.0] * len(sugar) + [150.0] * len(bitter)),
    }
    early = slice(int(100 / BIN_MS), int(300 / BIN_MS))
    late = slice(int((args.ms - 200) / BIN_MS), None)

    rows = []
    for scale in args.scales:
        for name, (idx, hz) in conditions.items():
            r = simulate(brain, idx, hz, readout_idx=[mn9_l], n_run=args.trials, t_run=args.ms, bin_ms=BIN_MS,
                         params={"w_syn": PARAMS["w_syn"] * scale}, progress=False)
            pop = r["pop_hz"]
            rate = r["rate"][rest]
            rows.append({
                "scale": scale, "condition": name, "mn9_l_hz": float(r["rate"][mn9_l]),
                "over_10hz": int((rate > 10).sum()), "over_100hz": int((rate > 100).sum()),
                "pop_early": float(pop[early].mean()), "pop_late": float(pop[late].mean()),
                "growth": float(pop[late].mean() / max(pop[early].mean(), 1.0)),
                "pop_hz": pop.round(0).tolist(), "seconds": r["seconds"],
            })
            x = rows[-1]
            print(f"scale {scale:4.2f}  {name:13s} MN9_L {x['mn9_l_hz']:6.1f} Hz  >10Hz {x['over_10hz']:6,}  "
                  f">100Hz {x['over_100hz']:5,}  spikes/s early {x['pop_early']:9,.0f} late {x['pop_late']:9,.0f}  "
                  f"growth x{x['growth']:.1f}  {r['seconds']:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    print("\n" + df.pivot(index="scale", columns="condition", values=["mn9_l_hz", "over_10hz", "growth"]).round(1).to_string())
    tag = "_".join(map(str, args.scales))
    (ROOT / "results" / f"calibration_{int(args.ms)}ms_{tag}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
