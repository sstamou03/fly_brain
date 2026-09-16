"""Record the brain's response to each drink for the Fly Bar UI -> ui/data.json.

For every drink the whole brain is simulated (W_SYN_MALE_CNS, all taste organs touching the drink),
and the UI gets a replay of what actually happened, binned in time:
  - taste channels: mean firing of the real taste neurons in each channel
  - the circuit: the strongest relays downstream of the taste neurons and upstream of MN9_L
    (chosen from the wiring, not from activity), with their real connections
  - MN9_L: firing rate and membrane potential

Preference score: MN9_L firing rate first; drinks that never make MN9_L fire are ordered by how
close its membrane potential got to threshold. So the fly always ends up somewhere.

usage: python -m export.export_ui --menu cocktails [--extra "πορτοκάλι,Νερό"] [--trials 6]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from brain.sim import PARAMS, W_SYN_MALE_CNS, Brain, simulate
from brain.taste import CHANNELS, INGREDIENTS, MENUS, neuron_input, taste_matrix, taste_vector

ROOT = Path(__file__).resolve().parents[1]
BIN_MS = 25.0
K_DOWN, K_UP, MAX_EDGES = 40, 40, 400


def circuit(brain, grn_idx, mn9_l):
    """Pick relay neurons from the wiring: most synapses from all taste neurons / onto MN9_L."""
    pre = np.repeat(np.arange(brain.n), np.diff(brain.col_ptr.numpy()))
    post = brain.post.numpy()
    w = brain.weight.numpy()
    is_grn = np.zeros(brain.n, bool)
    is_grn[grn_idx] = True

    m = is_grn[pre] & ~is_grn[post]
    down = np.bincount(post[m], weights=np.abs(w[m]), minlength=brain.n)
    down[mn9_l] = 0
    m = post == mn9_l
    up = np.bincount(pre[m], weights=np.abs(w[m]), minlength=brain.n)
    up[is_grn] = 0
    up[mn9_l] = 0
    top_down = list(np.argsort(-down)[:K_DOWN])
    top_up = [i for i in np.argsort(-up) if i not in set(top_down)][:K_UP]
    return np.array(top_down), np.array(top_up), (pre, post, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--menu", default="cocktails", choices=sorted(MENUS))
    ap.add_argument("--extra", default="", help="comma-separated substrings of classic drinks to add")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--ms", type=float, default=1000.0)
    ap.add_argument("--only", default="", help="comma-separated substrings; smoke test on a subset")
    ap.add_argument("--out", default="data.json")
    ap.add_argument("--hunger", type=float, default=0.0, help="scales the bitter channel by (1 - hunger), see hunger.py")
    args = ap.parse_args()
    drinks = dict(MENUS[args.menu])
    for s in filter(None, args.extra.split(",")):
        drinks.update({k: v for k, v in MENUS["classic"].items() if s in k})
    only = [s for s in args.only.split(",") if s]
    drinks = {k: v for k, v in drinks.items() if not only or any(s in k for s in only)}

    brain = Brain(ROOT / "brain.npz")
    meta = pd.read_parquet(ROOT / "brain_meta.parquet")
    mn9_l = int(meta.set_index("instance")["idx"]["MN9_L"])
    grn_idx, M, organ = taste_matrix(meta)
    down, up, (pre, post, w) = circuit(brain, grn_idx, mn9_l)
    relays = np.concatenate([down, up])
    readout = np.concatenate([grn_idx, relays, [mn9_l]])
    pos = {int(n): i for i, n in enumerate(readout)}

    # nodes: one per taste channel (group of real neurons), each relay neuron, MN9_L
    info = meta.set_index("idx")
    nodes = [{"id": f"ch:{c}", "kind": "taste", "label": c, "n_neurons": int(M[:, j].sum())}
             for j, c in enumerate(CHANNELS)]
    for n in relays:
        r = info.loc[int(n)]
        nodes.append({"id": f"n:{int(n)}", "kind": "relay_in" if n in set(down) else "relay_out",
                      "label": r["instance"] if isinstance(r["instance"], str) else str(int(r["bodyId"])),
                      "type": r["type"] if isinstance(r["type"], str) else None, "nt": r["nt"],
                      "body_id": int(r["bodyId"]), "superclass": r["superclass"]})
    nodes.append({"id": f"n:{mn9_l}", "kind": "motor", "label": "MN9_L", "type": "MN9", "nt": info.loc[mn9_l, "nt"],
                  "body_id": int(info.loc[mn9_l, "bodyId"]), "superclass": "cb_motor"})

    # edges between displayed nodes, taste channels aggregate their neurons' synapses
    node_of = {int(n): f"n:{int(n)}" for n in np.concatenate([relays, [mn9_l]])}
    shown = np.zeros(brain.n, bool)
    shown[list(node_of)] = True
    grn_channels = {int(g): [CHANNELS[j] for j in np.nonzero(M[i])[0]] for i, g in enumerate(grn_idx)}
    is_grn = np.zeros(brain.n, bool)
    is_grn[grn_idx] = True
    m = (shown[pre] | is_grn[pre]) & shown[post] & (pre != post)
    agg = {}
    for a, b, ww in zip(pre[m], post[m], w[m]):
        sources = [f"ch:{c}" for c in grn_channels[int(a)]] if a in grn_channels else [node_of[int(a)]]
        for s in sources:
            agg[(s, node_of[int(b)])] = agg.get((s, node_of[int(b)]), 0.0) + float(ww)
    edges = sorted(({"source": s, "target": t, "synapses": round(v)} for (s, t), v in agg.items()),
                   key=lambda e: -abs(e["synapses"]))[:MAX_EDGES]

    out_drinks = []
    (ROOT / "ui" / "rates").mkdir(parents=True, exist_ok=True)
    for name, x in drinks.items():
        t = taste_vector(x)
        bitter_fed = float(t[CHANNELS.index("bitter")])
        t[CHANNELS.index("bitter")] *= 1.0 - args.hunger
        u = neuron_input(t, M)
        on = u >= 1.0
        print(f"== {name}", flush=True)
        r = simulate(brain, grn_idx[on], u[on], readout_idx=readout, n_run=args.trials, t_run=args.ms,
                     bin_ms=BIN_MS, params={"w_syn": W_SYN_MALE_CNS}, progress=False)
        hz = r["readout"].mean(0) / (BIN_MS / 1000.0)  # (R, bins)
        activity = {}
        for j, c in enumerate(CHANNELS):
            members = [pos[int(g)] for g in grn_idx[M[:, j] > 0]]
            activity[f"ch:{c}"] = hz[members].mean(0).round(1).tolist()
        for n in np.concatenate([relays, [mn9_l]]):
            activity[node_of[int(n)]] = hz[pos[int(n)]].round(1).tolist()
        per_trial = r["readout"][:, pos[mn9_l]].sum(1) / (args.ms / 1000.0)
        v = r["readout_v"][:, pos[mn9_l]].mean(0)
        pop = r["pop_hz"]
        emoji, drink = name.split(" ", 1)
        slug = "".join(ch if ch.isalnum() else "_" for ch in drink).strip("_")
        np.save(ROOT / "ui" / "rates" / f"{slug}.npy", r["rate"].astype(np.float16))
        out_drinks.append({
            "name": drink, "emoji": emoji, "slug": slug, "bitter_fed": round(bitter_fed, 3),
            "drink_vector": dict(zip(INGREDIENTS, x.tolist())), "taste_vector": dict(zip(CHANNELS, t.round(3).tolist())),
            "mn9_hz": float(per_trial.mean()), "mn9_sem": float(per_trial.std(ddof=1) / np.sqrt(args.trials)),
            "mn9_v_mean": float(v.mean()), "mn9_v": v.round(2).tolist(),
            "growth": float(pop[-8:].mean() / max(pop[4:12].mean(), 1.0)), "pop_hz": pop.round(0).tolist(),
            "activity": activity, "seconds": r["seconds"],
        })
        d = out_drinks[-1]
        print(f"   MN9_L {d['mn9_hz']:6.1f} Hz   mean V {d['mn9_v_mean']:.2f} mV   growth x{d['growth']:.1f}   "
              f"{r['seconds']:.0f}s", flush=True)

    ranked = sorted(out_drinks, key=lambda d: (d["mn9_hz"], d["mn9_v_mean"]), reverse=True)
    for i, d in enumerate(ranked, 1):
        d["rank"] = i
    (ROOT / "ui").mkdir(exist_ok=True)
    data = {"bin_ms": BIN_MS, "t_run_ms": args.ms, "n_trials": args.trials, "w_syn_mV": W_SYN_MALE_CNS,
            "hunger": args.hunger,
            "v_rest": PARAMS["v_0"], "v_th": PARAMS["v_th"], "dataset": "male-cns v1.0",
            "nodes": nodes, "edges": edges, "drinks": out_drinks}
    (ROOT / "ui" / args.out).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print("\nranking:", " > ".join(f"{d['emoji']} {d['name']}" for d in ranked))


if __name__ == "__main__":
    main()
