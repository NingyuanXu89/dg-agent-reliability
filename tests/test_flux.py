import numpy as np

from burgers_dg.flux import godunov, rusanov


def test_consistency():
    u = np.linspace(-2, 2, 9)
    assert np.allclose(godunov(u, u), 0.5 * u * u)
    assert np.allclose(rusanov(u, u), 0.5 * u * u)


def test_godunov_cases():
    # right-moving shock (uL > uR > 0): upwind left state
    assert godunov(np.array(2.0), np.array(1.0)) == 2.0
    # left-moving shock (0 > uL > uR): upwind right state
    assert godunov(np.array(-1.0), np.array(-2.0)) == 2.0
    # rarefaction entirely right-moving: left state
    assert godunov(np.array(1.0), np.array(2.0)) == 0.5
    # transonic rarefaction: sonic point flux f(0) = 0
    assert godunov(np.array(-1.0), np.array(2.0)) == 0.0
    # stationary shock uL = -uR > 0: f(uL)
    assert godunov(np.array(1.5), np.array(-1.5)) == 1.125


def test_godunov_is_monotone():
    a = np.linspace(-2, 2, 41)
    UL, UR = np.meshgrid(a, a, indexing="ij")
    F = godunov(UL, UR)
    assert np.all(np.diff(F, axis=0) >= -1e-15)   # nondecreasing in uL
    assert np.all(np.diff(F, axis=1) <= 1e-15)    # nonincreasing in uR
