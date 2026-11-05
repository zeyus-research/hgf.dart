"""Shared setup for the pyhgf repros: float64 JAX on the CPU, the USD/CHF
example series, and access to the fixture generator's spec builders.

Run every script from ``tool/parity`` (its uv environment has pyhgf and
rshgf built from ../pyhgf):

    cd tool/parity && uv run python ../../docs/upstream/pyhgf/<script>.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from pyhgf import load_data

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tool" / "parity"))


def usdchf() -> np.ndarray:
    return np.asarray(load_data("continuous"), dtype=np.float64)


def fn_name(fn) -> str:
    while hasattr(fn, "func"):
        fn = fn.func
    return getattr(fn, "__name__", repr(fn))


def steps(net) -> list[tuple[int, str]]:
    """The JAX update steps as (node, function name)."""
    if net.update_sequence is None:
        net.create_belief_propagation_fn()
    return [(int(i), fn_name(f)) for i, f in net.update_sequence.update_steps]


def first_nonfinite(trajectories, nodes) -> int | None:
    """First trial where any node's ``mean`` is non-finite."""
    first = None
    for i in nodes:
        bad = np.flatnonzero(~np.isfinite(np.asarray(trajectories[i]["mean"])))
        if bad.size and (first is None or bad[0] < first):
            first = int(bad[0])
    return first
