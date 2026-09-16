"""Build the simulator's connectome (brain.npz + brain_meta.parquet) from the male-CNS v1.0 files.

Neurons: every body with a `superclass` annotation (166,700 = the published neuron count).
Edges: all neuron->neuron rows of the minconf-0.5 weight table, weight = synapse count.
Sign (Shiu et al. 2024 rule): GABA / glutamate inhibitory, everything else excitatory.
Histamine, absent from FlyWire's predictions, is also inhibitory (chloride channels in flies).
Transmitter label: consensus_nt, falling back to the cell-type prediction, then the body's own.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ANN = DATA / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NT = DATA / "body-neurotransmitters-male-cns-v1.0.feather"
W = DATA / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
INHIBITORY = {"gaba", "glutamate", "histamine"}


def usable(s):
    return s.notna() & (s != "unclear")


def main():
    t0 = time.time()
    ann = pd.read_feather(ANN)
    neu = ann[ann["superclass"].notna()].sort_values("bodyId").reset_index(drop=True)
    body_ids = neu["bodyId"].to_numpy(np.int64)
    n = len(body_ids)
    print(f"neurons: {n:,}")

    nt = pd.read_feather(NT, columns=["body", "consensus_nt", "celltype_predicted_nt", "predicted_nt"])
    nt = nt.drop_duplicates("body").set_index("body").reindex(body_ids)
    label = nt["consensus_nt"].where(usable(nt["consensus_nt"]), nt["celltype_predicted_nt"])
    label = label.where(usable(label), nt["predicted_nt"]).where(lambda s: usable(s), "unknown")
    sign = np.where(label.isin(INHIBITORY), -1.0, 1.0).astype(np.float32)
    print("transmitter labels:\n" + label.value_counts().to_string())

    reader = ipc.open_file(W)
    pre_l, post_l, w_l, rows = [], [], [], 0
    for b in range(reader.num_record_batches):
        batch = reader.get_batch(b)
        pre = batch.column("body_pre").to_numpy()
        post = batch.column("body_post").to_numpy()
        ip = np.minimum(np.searchsorted(body_ids, pre), n - 1)
        iq = np.minimum(np.searchsorted(body_ids, post), n - 1)
        ok = (body_ids[ip] == pre) & (body_ids[iq] == post)
        pre_l.append(ip[ok].astype(np.int32))
        post_l.append(iq[ok].astype(np.int32))
        w_l.append(batch.column("weight").to_numpy()[ok].astype(np.int32))
        rows += batch.num_rows
        if b % 500 == 0:
            print(f"  batch {b}/{reader.num_record_batches}  {time.time() - t0:.0f}s", flush=True)

    pre = np.concatenate(pre_l)
    post = np.concatenate(post_l)
    w = np.concatenate(w_l)
    order = np.argsort(pre, kind="stable")
    pre, post, w = pre[order], post[order], w[order]
    col_ptr = np.concatenate([[0], np.cumsum(np.bincount(pre, minlength=n))]).astype(np.int64)
    weight = w.astype(np.float32) * sign[pre]

    print(f"weight rows read: {rows:,}   neuron->neuron edges kept: {len(w):,}   synapses: {int(w.sum()):,}")
    print(f"inhibitory neurons: {(sign < 0).mean():.1%}   inhibitory synapses: {w[sign[pre] < 0].sum() / w.sum():.1%}")
    np.savez(ROOT / "brain.npz", body_ids=body_ids, col_ptr=col_ptr, post=post, weight=weight)

    keep = ["bodyId", "type", "instance", "superclass", "class", "subclass", "somaSide", "receptorType", "status"]
    meta = neu[keep].copy()
    meta.insert(0, "idx", np.arange(n))
    meta["nt"] = label.to_numpy()
    meta["sign"] = sign
    meta["out_synapses"] = np.bincount(pre, weights=w, minlength=n).astype(np.int64)
    meta["in_synapses"] = np.bincount(post, weights=w, minlength=n).astype(np.int64)
    meta.to_parquet(ROOT / "brain_meta.parquet", index=False)
    print(f"saved brain.npz + brain_meta.parquet in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
