"""Conservative Legendre DG, LLF flux, SSPRK3, and a TVB limiter."""
from dataclasses import dataclass
import time
import numpy as np
from numpy.polynomial import legendre as leg

LENGTH = 2 * np.pi


@dataclass(frozen=True)
class Config:
    cells: int = 128
    degree: int = 2
    final_time: float = 1.5
    cfl: float = 0.15
    limiter: bool = True
    tvb_m: float = 1.0
    dt_scale: float = 1.0
    time_accuracy: bool = False
    origin: float = 0.0

    def __post_init__(self):
        if isinstance(self.cells, bool) or not isinstance(self.cells, int) or self.cells < 3:
            raise ValueError("cells must be an integer >= 3")
        if isinstance(self.degree, bool) or not isinstance(self.degree, int) or self.degree not in (1, 2, 3):
            raise ValueError("degree must be 1, 2, or 3")
        for key in ("final_time", "cfl", "tvb_m", "dt_scale", "origin"):
            if not np.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite")
        if self.final_time < 0 or self.tvb_m < 0:
            raise ValueError("final_time and tvb_m must be nonnegative")
        if not 0 < self.cfl <= 0.3 or not 0 < self.dt_scale <= 1:
            raise ValueError("require 0 < cfl <= 0.3 and 0 < dt_scale <= 1")


@dataclass
class Result:
    config: Config
    times: np.ndarray
    coefficients: np.ndarray
    diagnostics: list
    steps: int
    runtime: float


def initial_wave(x):
    return 1 + np.sin(x)


def numerical_flux(left, right):
    """A single conservative local Lax--Friedrichs interface flux."""
    return 0.25 * (left**2 + right**2) - 0.5 * np.maximum(
        np.abs(left), np.abs(right)) * (right - left)


def minmod(a, b, c):
    same = (np.sign(a) == np.sign(b)) & (np.sign(a) == np.sign(c))
    return np.where(same, np.sign(a) * np.minimum(np.abs(a),
                    np.minimum(np.abs(b), np.abs(c))), 0.0)


class DG:
    def __init__(self, config):
        self.config = config
        self.p = config.degree
        self.h = LENGTH / config.cells
        self.edges = config.origin + np.arange(config.cells + 1) * self.h
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.nodes, self.weights = leg.leggauss(2 * self.p + 3)
        self.basis = leg.legvander(self.nodes, self.p)
        self.parity = (-1.0)**np.arange(self.p + 1)
        self.mass = self.h / (2 * np.arange(self.p + 1) + 1)
        derivative = np.zeros_like(self.basis)
        for j in range(1, self.p + 1):
            c = np.zeros(j + 1)
            c[j] = 1
            derivative[:, j] = leg.legval(self.nodes, leg.legder(c))
        self.volume = self.weights[:, None] * derivative

    def project(self, function):
        x = self.centers[:, None] + self.h / 2 * self.nodes
        values = np.asarray(function(x), dtype=float)
        values = np.broadcast_to(values, x.shape)
        if not np.all(np.isfinite(values)):
            raise ValueError("initial data must be finite")
        return (values @ (self.weights[:, None] * self.basis)) * (
            (2 * np.arange(self.p + 1) + 1) / 2)

    def traces(self, coefficients):
        return coefficients @ self.parity, coefficients.sum(axis=1)

    def rhs(self, coefficients):
        """M_j dU_j/dt = integral f(u) P'_j - f_R + (-1)^j f_L."""
        values = coefficients @ self.basis.T
        _, right = self.traces(coefficients)
        left, _ = self.traces(coefficients)
        face = numerical_flux(right, np.roll(left, -1))
        residual = 0.5 * values**2 @ self.volume
        residual -= face[:, None]
        residual += np.roll(face, 1)[:, None] * self.parity
        return residual / self.mass

    def evaluate(self, coefficients, x, cells=None):
        x = np.asarray(x)
        if cells is None:
            local = (x - self.config.origin) % LENGTH
            cells = np.minimum((local / self.h).astype(int), self.config.cells - 1)
            xi = 2 * (local / self.h - cells) - 1
        else:
            xi = 2 * (x - self.centers[cells]) / self.h
        return np.sum(coefficients[cells] * leg.legvander(xi, self.p), axis=-1)

    def bounds(self, c):
        """Exact extrema of each degree <= 3 polynomial, including endpoints."""
        left, right = self.traces(c)
        lo, hi = np.minimum(left, right), np.maximum(left, right)
        roots = []
        if self.p == 2:
            roots.append(np.divide(-c[:, 1], 3*c[:, 2],
                         out=np.full(len(c), np.nan), where=c[:, 2] != 0))
        if self.p == 3:
            a, b, d = 7.5*c[:, 3], 3*c[:, 2], c[:, 1]-1.5*c[:, 3]
            linear = np.abs(a) <= 1e-14 * np.maximum(1, np.abs(b)+np.abs(d))
            roots.append(np.divide(-d, b, out=np.full(len(c), np.nan),
                                   where=linear & (b != 0)))
            disc = b*b-4*a*d
            q = -0.5*(b + np.copysign(np.sqrt(np.maximum(disc, 0)), b))
            valid = (~linear) & (disc >= 0)
            roots.append(np.divide(q, a, out=np.full(len(c), np.nan), where=valid))
            roots.append(np.divide(d, q, out=np.full(len(c), np.nan),
                                   where=valid & (q != 0)))
        for xi in roots:
            valid = np.isfinite(xi) & (np.abs(xi) <= 1)
            safe = np.where(valid, xi, 0)
            val = np.sum(c * leg.legvander(safe, self.p), axis=1)
            lo = np.where(valid, np.minimum(lo, val), lo)
            hi = np.where(valid, np.maximum(hi, val), hi)
        return lo, hi

    def limit(self, c):
        """TVB endpoint detector; slope-limit flagged cells without changing means."""
        out = c.copy()
        mean = c[:, 0]
        dl, dr = mean-np.roll(mean, 1), np.roll(mean, -1)-mean
        left, right = self.traces(c)
        a, b = right-mean, mean-left
        threshold = self.config.tvb_m * self.h**2
        def tvb(z):
            return np.where(np.abs(z) <= threshold, z, minmod(z, dl, dr))
        tolerance = 1e-12 * np.maximum(1, np.max(np.abs(c), axis=1))
        flagged = (np.abs(tvb(a)-a) > tolerance) | (np.abs(tvb(b)-b) > tolerance)
        out[flagged, 1] = minmod(c[:, 1], dl, dr)[flagged]
        out[flagged, 2:] = 0
        return out, int(np.count_nonzero(flagged))

    def step(self, c, dt):
        count = 0
        def stage(v):
            nonlocal count
            if not np.all(np.isfinite(v)):
                raise FloatingPointError("nonfinite DG state; reduce CFL or enable limiting")
            if self.config.limiter:
                v, n = self.limit(v)
                count += n
            return v
        u1 = stage(c + dt*self.rhs(c))
        u2 = stage(0.75*c + 0.25*(u1 + dt*self.rhs(u1)))
        u3 = stage(c/3 + (2/3)*(u2 + dt*self.rhs(u2)))
        return u3, count

    def summary(self, c):
        lo, hi = self.bounds(c)
        left, right = self.traces(c)
        return dict(mass=float(self.h*np.sum(c[:, 0])),
                    entropy=float(0.5*np.sum(c*c*self.mass)),
                    minimum=float(lo.min()), maximum=float(hi.max()),
                    max_interface_jump=float(np.max(np.abs(right-np.roll(left, -1)))))


def simulate(config, initial=initial_wave, times=None):
    start = time.perf_counter()
    dg = DG(config)
    c = dg.project(initial)
    if times is None:
        times = np.linspace(0, config.final_time, 151) if config.final_time else np.array([0.])
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times))
            or times[0] != 0 or times[-1] != config.final_time
            or np.any(np.diff(times) <= 0)):
        raise ValueError("times must increase strictly from 0 to final_time")
    initial_mass = dg.summary(c)["mass"]
    snapshots, diagnostics = [], []
    t, steps, cumulative, interval, interval_steps = 0., 0, 0, 0, 0
    for target in times:
        while t < target:
            lo, hi = dg.bounds(c)
            speed = max(np.max(np.abs(lo)), np.max(np.abs(hi)))
            if speed == 0:
                dt = target-t
            else:
                dt = config.cfl*dg.h / ((2*dg.p+1)*speed) * config.dt_scale
                if config.time_accuracy:
                    dt *= min(1., dg.h**max(0., (dg.p+1)/3-1))
            dt = min(dt, target-t)
            if dt <= 0 or t+dt == t:
                raise FloatingPointError(f"time step underflow at t={t:g}")
            try:
                c, flagged = dg.step(c, dt)
            except FloatingPointError as exc:
                raise FloatingPointError(f"{exc}; t={t:g}, dt={dt:g}, step={steps}") from exc
            t = target if dt == target-t else t+dt
            steps += 1
            interval_steps += 1
            interval += flagged
            cumulative += flagged
        row = dg.summary(c)
        row.update(time=float(target), mass_drift=row["mass"]-initial_mass,
                   limiter_fraction=interval/(3*config.cells*interval_steps) if interval_steps else 0.,
                   limiter_activations=cumulative, steps=steps)
        diagnostics.append(row)
        snapshots.append(c.copy())
        interval, interval_steps = 0, 0
    return Result(config, times, np.stack(snapshots), diagnostics, steps, time.perf_counter()-start)
