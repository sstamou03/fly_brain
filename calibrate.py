"""Calibrate w_syn for the male CNS.

Shiu et al. tuned w_syn = 0.275 mV on FlyWire (~50M synapses). The male CNS has 124M synapses
on a similar neuron count, so the same per-synapse weight drives the network into runaway
(thousands of unrelated neurons near the ~300 Hz refractory ceiling).

Sweep a scale on w_syn and keep the largest one that still satisfies the Phase-1 behaviour:
  sugar -> MN9_L fires,  bitter alone -> MN9_L silent,  sugar + bitter -> suppressed,
with activity staying bounded (no crowd of saturated neurons).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sim import PARAMS, Brain, simulate

ROOT = Path(__file__).parent
SCALES = [1.0, 0.7, 0.5, 0.4, 0.3, 0.2]
N_RUN, T_RUN = 8, 300.0


def main():
    brain = Brain(ROOT / "brain.npz")
    meta = pd.read_parquet(ROOT / "brain_meta.parquet")
    by_type = meta.groupby("type")["idx"].apply(np.array).to_dict()
    mn9_l = int(meta.set_index("instance")["idx"]["MN9_L"])
    sugar = np.concatenate([by_type["LB3b"], by_type["LB3c"]])
    bitter = np.concatenate([by_type[t] for t in ["LB1a", "LB1b", "LB1c", "LB1d"]])
    stimulated = set(sugar) | set(bitter)
    not_stim = np.array([i not in stimulated for i in range(brain.n)])
    conditions = {
        "sugar": (sugar, [200.0] * len(sugar)),
        "bitter": (bitter, [150.0] * len(bitter)),
        "sugar+bitter": (np.concatenate([sugar, bitter]), [200.0] * len(sugar) + [150.0] * len(bitter)),
    }

    scales = [float(s) for s in sys.argv[1:]] or SCALES  # e.g. python calibrate.py 0.45 0.55 0.6
    rows = []
    for scale in scales:
        for name, (idx, hz) in conditions.items():
            r = simulate(brain, idx, hz, readout_idx=[mn9_l], n_run=N_RUN, t_run=T_RUN,
                         params={"w_syn": PARAMS["w_syn"] * scale}, progress=False)
            rate = r["rate"][not_stim]
            rows.append({
                "scale": scale, "w_syn_mV": round(PARAMS["w_syn"] * scale, 4), "condition": name,
                "mn9_l_hz": float(r["rate"][mn9_l]), "active": int((rate > 0).sum()),
                "over_100hz": int((rate > 100).sum()), "spikes_per_s": float(rate.sum()), "seconds": r["seconds"],
            })
            print(f"scale {scale:4.2f}  {name:13s} MN9_L {rows[-1]['mn9_l_hz']:6.1f} Hz  active {rows[-1]['active']:6,}  "
                  f">100Hz {rows[-1]['over_100hz']:5,}  spikes/s {rows[-1]['spikes_per_s']:10,.0f}  {r['seconds']:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    print("\n" + df.pivot(index="scale", columns="condition", values=["mn9_l_hz", "active", "over_100hz"]).round(1).to_string())
    (ROOT / "results").mkdir(exist_ok=True)
    name = "calibration.json" if scales == SCALES else f"calibration_{'_'.join(map(str, scales))}.json"
    (ROOT / "results" / name).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
