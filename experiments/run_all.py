"""Regenerate every result in results/: convergence study, diagnostics and animation."""

import time

import animate
import convergence

if __name__ == "__main__":
    t0 = time.perf_counter()
    convergence.main()
    animate.main()
    print(f"\nAll experiments finished in {time.perf_counter() - t0:.1f} s; outputs in results/")
