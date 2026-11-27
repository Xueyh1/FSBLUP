"""Use identical baseline caches to isolate byte counts and estimator-unit changes."""

import argparse, csv, subprocess, time, itertools
import numpy as np
from pathlib import Path


def read(p):
    with p.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def save(p, records):
    with p.open("w") as f:
        w = csv.DictWriter(f, records[0], delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(records)


def invoke(exe, opts, out):
    cmd = [exe, "fit", *opts, "--log-level", "off", "--out", out]
    t = time.monotonic()
    subprocess.run(list(map(str, cmd)), check=True, capture_output=True)
    return time.monotonic() - t


def compare(pred, ref):
    assert [(r["id"], r["role"]) for r in pred] == [(r["id"], r["role"]) for r in ref]
    return max(
        abs(float(r[k]) - float(s[k])) for r, s in zip(pred, ref) for k in ("u", "yhat")
    )


def measure_operations(a):
    metrics = []
    for n in [256, 4096]:
        for mode in ["A", "B"]:
            d = a.before / "reproduce_coverage" / f"{mode}_{n}"
            for alpha in [0, 0.2] if mode == "B" else [0]:
                common = [
                    "--cache",
                    d / "cache",
                    "--pheno",
                    d / "pheno",
                    "--trait",
                    "trait",
                    "--alpha",
                    alpha,
                    "--beta",
                    ".4",
                    "--varcomp",
                    "fixed",
                    "--lambda",
                    "1",
                    "--memory-mb",
                    "512",
                ]
                dense = None
                if n == 256:
                    dest = a.out / f"dense-{mode}-{alpha}"
                    invoke(
                        a.new,
                        common + ["--cache-backend", "dense", "--solver", "direct"],
                        dest,
                    )
                    dense = read(dest / "predictions.tsv")
                for rep in range(4):
                    for version, exe in (
                        [("old", a.old), ("new", a.new)]
                        if rep % 2 == 0
                        else [("new", a.new), ("old", a.old)]
                    ):
                        dest = a.out / f"{n}-{mode}-{alpha}-{version}-{rep}"
                        wall = invoke(
                            exe,
                            common
                            + [
                                "--cache-backend",
                                "operator",
                                "--cache-storage",
                                "stream",
                            ],
                            dest,
                        )
                        pred = read(dest / "predictions.tsv")
                        err = compare(pred, dense) if dense else None
                        assert err is None or err < 1e-8, err
                        usage = read(dest / "cache_usage.tsv")[0]
                        fit = read(dest / "pcg_summary.tsv")[0]
                        rt = read(dest / "runtime.tsv")[0]
                        if rep:
                            metrics.append(
                                dict(
                                    version=version,
                                    n=n,
                                    train=8,
                                    targets=12,
                                    mode=mode,
                                    alpha=alpha,
                                    rep=rep,
                                    wall_s=wall,
                                    rss_KiB=rt["peak_rss_kib"],
                                    rhs=fit["rhs"],
                                    matvec=fit["matvec"],
                                    feature_B=usage["feature_bytes"],
                                    verified_B=usage["verified_cache_bytes"],
                                    calibration_B=usage.get("calibration_bytes", "NA"),
                                    preload_B=usage.get("training_preload_bytes", "NA"),
                                    training_B=usage.get("training_bytes", "NA"),
                                    cross_B=usage.get("cross_bytes", "NA"),
                                    dense_error=err,
                                )
                            )
                assert (
                    compare(
                        read(
                            a.out.joinpath(
                                f"{n}-{mode}-{alpha}-old-3", "predictions.tsv"
                            )
                        ),
                        read(
                            a.out.joinpath(
                                f"{n}-{mode}-{alpha}-new-3", "predictions.tsv"
                            )
                        ),
                    )
                    < 1e-8
                )
    save(a.out / "operations.tsv", metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("old", "new", "before", "out"):
        p.add_argument(k, type=Path)
    p.add_argument("--units-only", action="store_true")
    a = p.parse_args()
    a.out.mkdir()
    if not a.units_only:
        measure_operations(a)
    # All combinations share a cache, an external fold table and a separate scoring column.
    d = a.before / "reproduce_operators" / "stream"
    # Make an informative independent score column by direct out-of-fold
    # GLS on a known interior weight. Normalize raw features over all IDs,
    # matching the population used in prepare.
    snp = np.loadtxt(d / "snp", skiprows=1, usecols=range(1, 4097))
    omics = np.loadtxt(d / "omics", skiprows=1, usecols=(1, 2))
    freq = snp.mean(axis=0) / 2
    w = snp - 2 * freq
    z = (omics - omics.mean(axis=0)) / omics.std(axis=0, ddof=1)
    records = read(d / "pheno")
    ref = np.array([i for i, r in enumerate(records) if r["trait"] != "NA"])
    y = np.array([float(records[i]["trait"]) for i in ref])
    k = (
        (1 - 0.4375 - 0.1875 + 1e-6) * np.eye(len(ref))
        + 0.1875 * (w[ref] @ w[ref].T) / np.sum(2 * freq * (1 - freq))
        + 0.4375 * z[ref] @ z[ref].T / z.shape[1]
    )
    labels = {r["id"]: r["fold"] for r in read(d / "folds")}
    groups = np.array([labels[records[i]["ID"]] for i in ref])
    targets = {}
    for fold in set(groups):
        tr, va = np.flatnonzero(groups != fold), np.flatnonzero(groups == fold)
        h = np.linalg.solve(
            k[np.ix_(tr, tr)] + np.eye(len(tr)),
            np.stack((y[tr], np.ones(len(tr))), axis=1),
        )
        estimate = k[np.ix_(va, tr)] @ (
            h[:, 0] - h[:, 1] * h[:, 0].sum() / h[:, 1].sum()
        )
        targets.update(zip(ref[va], estimate))
    cases = []
    for backend, solver, parallel in itertools.product(
        ("dense", "operator"), ("pcg", "direct"), ("blas", "weights")
    ):
        baseline = None
        for scale in [1, 0.001, 1000]:
            tag = f"{backend}-{solver}-{parallel}-{scale}"
            pheno = a.out / (tag + ".tsv")
            rec = read(d / "pheno")
            for i, r in enumerate(rec):
                if r["trait"] != "NA":
                    y = float(r["trait"])
                    r["target"] = scale * targets[i]
                    r["trait"] = scale * y
                else:
                    r["target"] = "NA"
            save(pheno, rec)
            dest = a.out / tag
            opts = [
                "--cache",
                d / "cache",
                "--cache-backend",
                backend,
                "--solver",
                solver,
                "--pheno",
                pheno,
                "--trait",
                "trait",
                "--metric",
                "rmse",
                "--score-col",
                "target",
                "--strict-simplex",
                "--varcomp",
                "fixed",
                "--lambda",
                "1",
                "--parallel",
                parallel,
                "--threads",
                2,
                "--folds-file",
                d / "folds",
            ]
            invoke(a.new, opts, dest)
            rows = read(dest / "search.tsv")
            t = read(dest / "tuning.tsv")[0]
            if baseline is None:
                baseline = rows
                initial = t
            assert [(r["alpha"], r["beta"], r["stage"]) for r in rows] == [
                (r["alpha"], r["beta"], r["stage"]) for r in baseline
            ]
            err = max(
                abs(float(r["mean"]) / scale - float(s["mean"]))
                for r, s in zip(rows, baseline)
            )
            assert err < 1e-8 and all(
                t[k] == initial[k] for k in ("stop", "selected_alpha", "selected_beta")
            )
            cases.append(
                dict(
                    backend=backend,
                    solver=solver,
                    parallel=parallel,
                    scale=scale,
                    points=len(rows),
                    point=t["selected_alpha"] + "," + t["selected_beta"],
                    max_unit_error=err,
                    stop=t["stop"],
                )
            )
    save(a.out / "scale-grid.tsv", cases)
    d = a.before / "replay_cv_units"
    results = []
    for policy, scale in itertools.product(("legacy", "relative"), (1, 0.001, 1000)):
        rec = read(d / "pheno_1.tsv")
        for r in rec:
            for k in ("trait", "target"):
                r[k] = float(r[k]) * scale
        pheno = a.out / f"units-{scale}"
        save(pheno, rec)
        opts = [
            *[v for c in "AGO" for v in ("--" + c, d / (c + ".tsv"))],
            "--pheno",
            pheno,
            "--trait",
            "trait",
            "--metric",
            "rmse",
            "--score-col",
            "target",
            "--folds-file",
            d / "folds.tsv",
            "--strict-simplex",
            "--varcomp",
            "fixed",
            "--lambda",
            1,
            "--score-policy",
            policy,
        ]
        dest = a.out / f"review-{policy}-{scale}"
        invoke(a.new, opts, dest)
        t = read(dest / "tuning.tsv")[0]
        results.append(
            dict(
                policy=policy,
                scale=scale,
                alpha=t["selected_alpha"],
                beta=t["selected_beta"],
                n=t["candidates_visited"],
                rmse=t["selection_score"],
            )
        )
    save(a.out / "legacy.tsv", results)


if __name__ == "__main__":
    main()
