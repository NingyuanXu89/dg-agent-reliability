"""Shared setup for the driver scripts."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

# Test problem used throughout: u0 = C + A sin(x) on [0, 2 pi), t_b = 1 / A.
C = 0.5
A = 1.0
TVB_M = 1.0   # ~ max|u0''|; keeps smooth extrema unlimited (M = 0 is TVD minmod)
