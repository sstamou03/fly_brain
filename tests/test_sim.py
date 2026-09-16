"""Sanity checks of the simulator on toy 3-neuron networks.

0 -> 1 with 100 synapses (excitatory), 0 -> 2 with 100 synapses (inhibitory, 2 also driven).
One spike through 100 synapses peaks at ~4.4 mV (threshold is 7 mV above rest), so neuron 1
fires only by temporal summation, which 100 Hz input provides (mean drive ~13.75 mV).
"""
import tempfile
from pathlib import Path

import numpy as np

from brain.sim import Brain, simulate


def toy_brain(edges, n):
    pre = np.array([e[0] for e in edges])
    order = np.argsort(pre, kind="stable")
    post = np.array([e[1] for e in edges])[order]
    w = np.array([e[2] for e in edges], dtype=np.float32)[order]
    col_ptr = np.concatenate([[0], np.cumsum(np.bincount(pre[order], minlength=n))])
    path = Path(tempfile.mkdtemp()) / "toy.npz"
    np.savez(path, body_ids=np.arange(n), col_ptr=col_ptr, post=post, weight=w)
    return Brain(path)


def main():
    b = toy_brain([(0, 1, 100.0), (0, 2, -100.0)], 3)

    r = simulate(b, [0], [100.0], n_run=50, t_run=1000.0, progress=False)
    print(f"driven neuron at 100 Hz input  -> {r['rate'][0]:.1f} Hz (expect ~100, a bit less from refractoriness)")
    assert 90 < r["rate"][0] <= 101
    print(f"excitatory partner (100 syn)    -> {r['rate'][1]:.1f} Hz (expect > 0)")
    assert r["rate"][1] > 0
    assert r["rate"][2] == 0

    alone = simulate(b, [2], [50.0], n_run=50, t_run=1000.0, progress=False)["rate"][2]
    inhib = simulate(b, [0, 2], [150.0, 50.0], n_run=50, t_run=1000.0, progress=False)["rate"][2]
    print(f"Poisson-driven neuron 2: alone {alone:.1f} Hz, with inhibition {inhib:.1f} Hz (kicks are suprathreshold, so ~equal)")

    silent = simulate(b, [0], [100.0], n_run=10, t_run=500.0, silence_idx=[1], progress=False)["rate"]
    assert silent[1] == 0
    print("silencing works; all checks passed")


if __name__ == "__main__":
    main()
