"""Where does the activity go? Short sugar run + static checks for runaway excitation."""
from pathlib import Path

import numpy as np
import pandas as pd

from sim import PARAMS, Brain, simulate

ROOT = Path(__file__).parent
pd.set_option("display.width", 250, "display.max_columns", 30)


def main():
    brain = Brain(ROOT / "brain.npz")
    meta = pd.read_parquet(ROOT / "brain_meta.parquet")

    # static: autapses (a neuron exciting itself can self-sustain)
    pre = np.repeat(np.arange(brain.n), np.diff(brain.col_ptr.numpy()))
    post = brain.post.numpy()
    w = brain.weight.numpy()
    auto = pre == post
    print(f"autapses: {auto.sum():,} edges, {(w[auto] > 0).sum():,} excitatory, "
          f"max {w[auto].max() if auto.any() else 0:.0f} synapses")
    single_spike_mv = lambda syn: syn * PARAMS["w_syn"] * 0.158  # peak PSP factor for tau=5, t_mbr=20
    strong = w * PARAMS["w_syn"] * 0.158 > PARAMS["v_th"] - PARAMS["v_0"]
    print(f"edges where ONE spike alone crosses threshold (>{(PARAMS['v_th'] - PARAMS['v_0']) / (PARAMS['w_syn'] * 0.158):.0f} syn): "
          f"{strong.sum():,}   of which autapses: {(strong & auto).sum():,}")

    by_type = meta.groupby("type")["idx"].apply(np.array).to_dict()
    sugar = np.concatenate([by_type["LB3b"], by_type["LB3c"]])
    r = simulate(brain, sugar, [200.0] * len(sugar), n_run=8, t_run=300.0, progress=False)
    meta["rate"] = r["rate"]
    meta["autapse_syn"] = 0.0
    meta.loc[pre[auto], "autapse_syn"] = w[auto]
    act = meta[(meta["rate"] > 0) & ~meta["idx"].isin(sugar)]
    print(f"\nactive (non-stimulated) neurons: {len(act):,}   total spikes/s: {act['rate'].sum():,.0f}")
    print("rate distribution:", {f">{k}Hz": int((act['rate'] > k).sum()) for k in (1, 10, 50, 100, 200)})

    g = act.groupby(act["superclass"]).agg(n=("rate", "size"), mean_hz=("rate", "mean"), spikes=("rate", "sum"))
    g["share"] = g["spikes"] / g["spikes"].sum()
    print("\n" + g.sort_values("spikes", ascending=False).round(2).to_string())

    cols = ["idx", "type", "instance", "superclass", "nt", "in_synapses", "out_synapses", "autapse_syn", "rate"]
    print("\ntop 30 by rate:\n" + act.sort_values("rate", ascending=False)[cols].head(30).to_string(index=False))
    hi = act[act["rate"] > 100]
    print(f"\n>100 Hz neurons: {len(hi)}, with excitatory autapse: {(hi['autapse_syn'] > 0).sum()}, "
          f"nt: {hi['nt'].value_counts().to_dict()}")
    meta.to_parquet(ROOT / "results" / "diag_sugar.parquet", index=False)


if __name__ == "__main__":
    main()
