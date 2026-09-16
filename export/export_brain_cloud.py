"""Export the male-CNS neuron positions as a 3D point cloud for the web app.

Every neuron with a soma (or to-soma) location in the v1.0 annotations: 140,638 of 166,700.
Coordinates are centered and scaled to [-1, 1] on the longest axis. Files go to web_data/:
  brain_xyz.bin    float32 x,y,z per point
  brain_idx.bin    int32 simulator index per point (to colour points by simulated activity)
  brain_group.bin  uint8 region group per point
  brain_cloud.json counts, group labels, source bounding box in voxels
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "web_data"
GROUPS = ["central brain", "optic lobes", "nerve cord", "sensory", "motor & long-range"]


def group_of(superclass):
    s = superclass or ""
    if "sensory" in s:
        return 3
    if s.startswith(("ol_", "visual_")):
        return 1
    if any(k in s for k in ("motor", "descending", "ascending", "efferent", "endocrine")):
        return 4
    if s.startswith("vnc_"):
        return 2
    return 0


def main():
    OUT.mkdir(exist_ok=True)
    ann = pd.read_feather(ROOT / "data" / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
                          columns=["bodyId", "superclass", "somaLocation", "tosomaLocation"])
    meta = pd.read_parquet(ROOT / "brain_meta.parquet", columns=["idx", "bodyId"])
    df = meta.merge(ann, on="bodyId", how="left")
    ok3 = lambda v: v is not None and not (isinstance(v, float)) and len(v) == 3
    loc = [s if ok3(s) else (t if ok3(t) else None) for s, t in zip(df["somaLocation"], df["tosomaLocation"])]
    keep = np.array([v is not None for v in loc])
    xyz = np.stack([loc[i] for i in np.nonzero(keep)[0]]).astype(np.float64)
    lo, hi = xyz.min(0), xyz.max(0)
    xyz_n = ((xyz - (lo + hi) / 2) / ((hi - lo).max() / 2)).astype(np.float32)

    idx = df["idx"].to_numpy()[keep].astype(np.int32)
    groups = np.array([group_of(s) for s in df["superclass"].to_numpy()[keep]], dtype=np.uint8)
    xyz_n.tofile(OUT / "brain_xyz.bin")
    idx.tofile(OUT / "brain_idx.bin")
    groups.tofile(OUT / "brain_group.bin")
    info = {"count": int(keep.sum()), "neurons_total": int(len(df)), "groups": GROUPS,
            "group_counts": np.bincount(groups, minlength=len(GROUPS)).tolist(),
            "bbox_voxels": {"min": lo.tolist(), "max": hi.tolist()}, "voxel_nm": 8}
    (OUT / "brain_cloud.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
