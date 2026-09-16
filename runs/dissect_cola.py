"""Why does cola ignite the brain and leave MN9 silent? Take its taste apart.

Pilot at W_SYN_MALE_CNS (10 x 1 s): water was quiet, but cola / espresso / tsipouro drove
5-7k neurons above 10 Hz and MN9_L stayed at 0. Each condition removes one ingredient of that:
which organs are touched, and which taste channels are on.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sim import W_SYN_MALE_CNS, Brain, simulate
from taste import CHANNELS, DRINKS, neuron_input, taste_matrix, taste_vector

ROOT = Path(__file__).parent
N_RUN, T_RUN = 4, 500.0


def main():
    brain = Brain(ROOT / "brain.npz")
    meta = pd.read_parquet(ROOT / "brain_meta.parquet")
    mn9_l = int(meta.set_index("instance")["idx"]["MN9_L"])
    cola = taste_vector(DRINKS["🥤 Κόλα"])
    ch = {c: i for i, c in enumerate(CHANNELS)}

    def only(*names):
        t = np.zeros_like(cola)
        for n in names:
            t[ch[n]] = cola[ch[n]]
        return t

    no_bitter = cola.copy()
    no_bitter[ch["bitter"]] = 0
    conditions = [
        ("cola, all organs", cola, ("legs", "labellum", "pharynx")),
        ("cola, mouth only (labellum+pharynx)", cola, ("labellum", "pharynx")),
        ("cola, legs only", cola, ("legs",)),
        ("cola without bitter, all organs", no_bitter, ("legs", "labellum", "pharynx")),
        ("sweet only, all organs", only("sweet"), ("legs", "labellum", "pharynx")),
        ("sweet only, mouth only", only("sweet"), ("labellum", "pharynx")),
        ("bitter only, mouth only", only("bitter"), ("labellum", "pharynx")),
        ("bitter only, legs only", only("bitter"), ("legs",)),
    ]

    rows = []
    for name, t, organs in conditions:
        grn_idx, M, _ = taste_matrix(meta, organs)
        u = neuron_input(t, M)
        on = u >= 1.0
        r = simulate(brain, grn_idx[on], u[on], readout_idx=[mn9_l], n_run=N_RUN, t_run=T_RUN,
                     params={"w_syn": W_SYN_MALE_CNS}, progress=False)
        rest = np.ones(brain.n, bool)
        rest[grn_idx[on]] = False
        rate = r["rate"][rest]
        rows.append({"condition": name, "stimulated": int(on.sum()), "mn9_l_hz": float(r["rate"][mn9_l]),
                     "over_10hz": int((rate > 10).sum()), "over_100hz": int((rate > 100).sum()),
                     "spikes_per_s": float(rate.sum()), "seconds": r["seconds"]})
        x = rows[-1]
        print(f"{name:38s} stim {x['stimulated']:3d}  MN9_L {x['mn9_l_hz']:6.1f} Hz  >10Hz {x['over_10hz']:6,}  "
              f">100Hz {x['over_100hz']:5,}  spikes/s {x['spikes_per_s']:9,.0f}  {r['seconds']:.0f}s", flush=True)

    (ROOT / "results" / "dissect_cola.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
