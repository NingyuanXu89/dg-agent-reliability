"""Smooth and post-shock refinement studies with temporal-error checks."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from .core import Config, DG, simulate
from .diagnostics import errors, coefficient_distance, save_result, write_csv
from .reference import shock_position


def time_checked(config, norm="l2"):
    """Accept the finer of two runs once their difference is < 10% of error."""
    times = np.array([0., config.final_time])
    coarse = simulate(config, times=times)
    for attempt in range(7):
        fine_config = replace(coarse.config, dt_scale=coarse.config.dt_scale/2)
        fine = simulate(fine_config, times=times)
        dg = DG(fine_config)
        err = errors(dg, fine.coefficients[-1], config.final_time)
        change = coefficient_distance(dg, coarse.coefficients[-1], fine.coefficients[-1],
                                      config.final_time, norm)
        ratio = change/max(err[norm], 1e-30)
        if ratio <= 0.1:
            return fine, err, change, ratio
        coarse = fine
    raise RuntimeError(f"temporal check failed for N={config.cells}, p={config.degree}: ratio={ratio:g}")


def convergence(directory, cells=(16, 32, 64, 128), degrees=(1, 2, 3)):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in degrees:
        previous = None
        for n in cells:
            cfg = Config(cells=int(n), degree=int(p), final_time=0.5, cfl=0.05,
                         limiter=False, time_accuracy=True)
            result, err, change, ratio = time_checked(cfg)
            row = dict(degree=p, cells=n, h=2*np.pi/n, **err,
                       l1_rate="", l2_rate="", linf_rate="",
                       temporal_l2_difference=change, temporal_error_ratio=ratio,
                       dt_scale=result.config.dt_scale, steps=result.steps,
                       mass_drift=result.diagnostics[-1]["mass_drift"])
            if previous is not None:
                for key, rate in (("l1", "l1_rate"), ("l2", "l2_rate"), ("linf_sampled", "linf_rate")):
                    row[rate] = float(np.log(previous[key]/row[key])/np.log(n/previous["cells"]))
            rows.append(row)
            previous = row
            save_result(result, directory/f"smooth_p{p}_n{n}")
            print(f"smooth p={p} N={n}: L2={err['l2']:.3e}, rate={row['l2_rate']}, time ratio={ratio:.2e}", flush=True)
    write_csv(directory/"smooth_convergence.csv", rows)
    return rows


def postshock(directory, cells=(64, 128, 256)):
    directory = Path(directory)
    rows = []
    for aligned in (False, True):
        previous = None
        for n in cells:
            h = 2*np.pi/n
            origin = shock_position(1.5) % h if aligned else 0.
            cfg = Config(cells=int(n), degree=2, final_time=1.5, origin=origin)
            result, err, change, ratio = time_checked(cfg, norm="l1")
            row = dict(aligned=aligned, cells=n, degree=2, origin=origin, **err,
                       temporal_l1_difference=change, temporal_error_ratio=ratio,
                       dt_scale=result.config.dt_scale,
                       mass_drift=result.diagnostics[-1]["mass_drift"],
                       minimum=result.diagnostics[-1]["minimum"],
                       maximum=result.diagnostics[-1]["maximum"],
                       entropy=result.diagnostics[-1]["entropy"])
            if previous is not None and row["l1"] >= previous["l1"]:
                raise RuntimeError("post-shock L1 error did not decrease with refinement")
            previous = row
            rows.append(row)
            save_result(result, directory/f"shock_{'aligned' if aligned else 'unaligned'}_n{n}")
            print(f"shock aligned={aligned} N={n}: L1={err['l1']:.3e}, time ratio={ratio:.2e}", flush=True)
    write_csv(directory/"postshock_convergence.csv", rows)
    return rows


def acceptance(smooth, shock, directory):
    checks = {}
    for p in (1, 2, 3):
        group = [r for r in smooth if r["degree"] == p]
        checks[f"p{p}_errors_decrease"] = all(b["l2"] < a["l2"] for a, b in zip(group[:-1], group[1:]))
        checks[f"p{p}_finest_rate"] = abs(group[-1]["l2_rate"]-(p+1)) <= 0.4
    checks["mass_drift_below_1e-10"] = all(abs(r["mass_drift"]) < 1e-10 for r in smooth+shock)
    checks["time_error_below_10_percent"] = all(r["temporal_error_ratio"] <= .1 for r in smooth+shock)
    checks["postshock_finite"] = all(np.isfinite(r["l1"]) for r in shock)
    checks["postshock_bounds"] = all(r["minimum"] > -.05 and r["maximum"] < 2.05 for r in shock)
    checks["postshock_l1_decreases"] = all(all(b["l1"] < a["l1"] for a,b in zip(g[:-1],g[1:]))
        for g in ([r for r in shock if r["aligned"] == k] for k in (False, True)))
    report = dict(passed=all(checks.values()), checks=checks,
                  entropy_note="Quadratic entropy is monitored; no fully discrete entropy inequality is asserted.")
    (Path(directory)/"acceptance.json").write_text(json.dumps(report, indent=2)+"\n")
    lines = ["# Numerical validation report", "", "## Smooth solution, t = 0.5", "",
             "| Degree | Finest grid | L2 error | Observed L2 order | Expected order |",
             "|---|---:|---:|---:|---:|"]
    for p in (1, 2, 3):
        r = [row for row in smooth if row["degree"] == p][-1]
        lines.append(f"| {p} | {r['cells']} | {r['l2']:.6e} | {r['l2_rate']:.4f} | {p+1} |")
    lines += ["", "The half-step comparisons control temporal contamination. Dotted lines in the smooth convergence plot indicate the expected p+1 slopes.",
              "", "## Shock solution, t = 1.5", "",
              "| Final alignment | Cells | L1 error | Half-step difference / error |",
              "|---|---:|---:|---:|"]
    for r in shock:
        lines.append(f"| {'Face' if r['aligned'] else 'Inside cell'} | {r['cells']} | {r['l1']:.6e} | {r['temporal_error_ratio']:.3e} |")
    drift = max(abs(r["mass_drift"]) for r in smooth + shock)
    lines += ["", f"Largest final mass drift across refinement studies: {drift:.3e}.",
              "", "The shock loses smooth-solution high-order convergence. Integrated L1 error decreases in both mesh families; sampled Linf error is not expected to converge uniformly at a discontinuity.",
              "", f"All acceptance checks passed: {report['passed']}.", "",
              "Quadratic entropy and overshoot are measured; a fully discrete entropy inequality is not asserted."]
    (Path(directory)/"report.md").write_text("\n".join(lines)+"\n")
    if not report["passed"]:
        raise RuntimeError(f"acceptance failures: {[k for k,v in checks.items() if not v]}")
    return report
