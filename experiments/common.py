"""Shared setup for the experiment scripts."""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

# Problem: u0 = C + A sin(x) on [0, 2 pi), shock at t_s = 1/A, x_s = pi + C t.
C, A = 0.5, 1.0
T_SHOCK = 1.0 / A
U_MAX = C + A


def u0(x):
    return C + A * np.sin(x)


def out(name):
    return os.path.join(RESULTS, name)
