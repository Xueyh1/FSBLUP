"""Run in a Slurm allocation with OPENBLAS_NUM_THREADS=1; use a new work directory."""

import argparse
import csv
import subprocess
import time
from pathlib import Path
import numpy as np


def save(path, header, rows):
    with path.open("w") as stream:
        w = csv.writer(stream, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def generate(d, n, m, train, targets, nested=False):
    d.mkdir()
    rng = np.random.default_rng(174)
    ids = ["i" + str(j) for j in range(n)]
    parents = []
    for j in range(n):
        if nested and j >= 4:
            parents.append([ids[j], ids[j - 3], ids[j - 4]])
        else:
            parents.append([ids[j], "0", "0"])
    save(d / "ped", ["ID", "Sire", "Dam"], parents)
    for name, width, skip in [
        ("snp", m, n // 3 if nested else 0),
        ("omics", m if n == 160 else 2, 2 * n // 3 if nested else 0),
    ]:
        data = (
            rng.integers(0, 3, (n - skip, width))
            if name == "snp"
            else rng.normal(size=(n - skip, width))
        )
        save(
            d / name,
            ["ID", *["v" + str(i) for i in range(width)]],
            ([s, *r] for s, r in zip(ids[skip:], data)),
        )
    chosen = np.linspace(0, n - 1, train + targets, dtype=int)
    y = rng.normal(size=train)
    if n == 160:
        y = np.random.default_rng(274).normal(size=(train, 5000))[:, 170]
    if nested:
        y += np.linspace(-2, 2, train)
    save(
        d / "pheno",
        ["ID", "trait"],
        ((ids[j], y[i] if i < train else "NA") for i, j in enumerate(chosen)),
    )
    save(
        d / "folds",
        ["repeat", "fold", "id"],
        ((1, i % 2 + 1, ids[j]) for i, j in enumerate(chosen[:train])),
    )


def prepare(exe, d, cache=None):
    subprocess.run(
        [
            exe,
            "prepare",
            *[v for k in ["ped", "snp", "omics"] for v in ["--" + k, d / k]],
            "--snp-format",
            "2bit",
            "--out",
            cache if cache is not None else d / "cache",
            "--log-level",
            "off",
        ],
        check=True,
    )


def run(exe, d, label, opts, report, cache=None):
    out = d / label
    cache = cache if cache is not None else d / "cache"
    args = [
        exe,
        "fit",
        "--cache",
        cache,
        "--pheno",
        d / "pheno",
        "--trait",
        "trait",
        "--varcomp",
        "fixed",
        "--lambda",
        "1",
        "--log-level",
        "off",
        *opts,
        "--out",
        out,
    ]
    tick = time.monotonic()
    subprocess.run(
        ["/usr/bin/time", "-f", "%M", "-o", d / (label + ".rss"), *args], check=True
    )
    elapsed = time.monotonic() - tick
    usage = rows(out / "cache_usage.tsv")[0]
    audits = rows(out / "pcg_summary.tsv")
    if (out / "cv_pcg_summary.tsv").exists():
        audits += rows(out / "cv_pcg_summary.tsv")
    model = rows(out / "model.tsv")[0]
    model.update(rows(out / "fusion.tsv")[0])
    tune = rows(out / "tuning.tsv")[0] if (out / "tuning.tsv").exists() else {}
    shape = (cache / "G.shape").read_text().split()
    data = [
        d.name,
        label,
        model["reference"],
        model["candidate"],
        shape[1],
        usage["backend"],
        tune.get("requested_domain", "historical" if tune else "fixed"),
        1,
        opts[opts.index("--memory-mb") + 1],
        (d / (label + ".rss")).read_text().strip(),
        usage["feature_bytes"],
        usage["verified_cache_bytes"],
        sum(int(a["rhs"]) for a in audits),
        sum(int(a["matvec"]) for a in audits),
        elapsed,
        model["alpha"],
        model["beta"],
        tune.get("search_coverage", "NA"),
    ]
    report.writerow(data)
    return rows(out / "predictions.tsv"), model


def dense_cross_reference(exe, d, pred):
    genotypes = np.loadtxt(d / "snp", skiprows=1, usecols=range(1, 129))
    p = genotypes.mean(axis=0) / 2
    selected = [int(r["id"][1:]) for r in pred]
    features = genotypes[selected] - 2 * p
    k = features @ features.T / np.sum(2 * p * (1 - p))
    ids = [r["id"] for r in pred]
    save(d / "K.tsv", ["ID", *ids], ([key, *r] for key, r in zip(ids, k)))
    dest = d / "dense"
    cmd = [
        exe,
        "fit",
        *[v for layer in "AGO" for v in ("--" + layer, d / "K.tsv")],
        "--pheno",
        d / "pheno",
        "--trait",
        "trait",
        "--alpha",
        "0",
        "--beta",
        "1",
        "--varcomp",
        "fixed",
        "--lambda",
        "1",
        "--solver",
        "direct",
        "--out",
        dest,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    oracle = rows(dest / "predictions.tsv")
    assert ids == [r["id"] for r in oracle]
    errors = [
        max(abs(float(r[key]) - float(q[key])) for r, q in zip(pred, oracle))
        for key in ("u", "yhat")
    ]
    assert max(errors) < 1e-10, errors
    save(d / "cross-errors.tsv", ["max_abs_u", "max_abs_yhat"], [errors])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exe", type=Path)
    p.add_argument("dir", type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--cases", default="domain,stats,cross,connected,topology")
    a = p.parse_args()
    a.dir.mkdir()
    with (a.dir / "report.tsv").open("w", buffering=1) as f:
        report = csv.writer(f, delimiter="\t", lineterminator="\n")
        report.writerow(
            "case label train candidates retained_snps backend domain threads memory_MiB RSS_KiB feature_B verified_B RHS matvec wall_s alpha beta coverage".split()
        )
        if "topology" in a.cases.split(",") and a.baseline:
            d = a.dir / "topology"
            generate(d, 1024, 8, 16, 8)
            # Repeated ancestry in a connected pedigree makes the cost of
            # recomputing Mendelian diagonals visible without a dense fit.
            ids = ["i" + str(i) for i in range(1024)]
            save(
                d / "ped",
                ["ID", "Sire", "Dam"],
                (
                    (key, ids[j - 3] if j >= 4 else "0", ids[j - 4] if j >= 4 else "0")
                    for j, key in enumerate(ids)
                ),
            )
            for fmt, writer in [("v2", a.baseline), ("v3", a.exe)]:
                prepare(writer, d, d / fmt)
            opts = [
                "--alpha",
                "0",
                "--beta",
                "1",
                "--cache-backend",
                "operator",
                "--memory-mb",
                "128",
            ]
            for i in range(3):
                first, _ = run(a.exe, d, f"v2-{i}", opts, report, d / "v2")
                second, _ = run(a.exe, d, f"v3-{i}", opts, report, d / "v3")
                assert first == second
        for case, n, m, train, target in [
            ("domain", 160, 96, 24, 4),
            ("stats", 10, 1, 8, 2),
            ("cross", 4096, 128, 8, 4),
            ("connected", 144, 96, 96, 48),
        ]:
            if case not in a.cases.split(","):
                continue
            d = a.dir / case
            generate(d, n, m, train, target, case == "connected")
            prepare(a.baseline or a.exe, d)
            variants = [(d, case)]
            if case == "stats":
                big = a.dir / "filtered"
                generate(big, n, m, train, target)
                lines = (big / "snp").read_text().splitlines()
                with (big / "snp").open("w") as seq:
                    print(
                        lines[0],
                        *["drop" + str(i) for i in range(199999)],
                        sep="\t",
                        file=seq,
                    )
                    for row in lines[1:]:
                        seq.write(row + "\tNA" * 199999 + "\n")
                prepare(a.baseline or a.exe, big)
                variants.append((big, "filtered"))
            for dataset, name in variants:
                for label, exe in ([("old", a.baseline)] if a.baseline else []) + [
                    ("new", a.exe)
                ]:
                    if case == "domain":
                        scores = []
                        for cap in ("200", "4"):
                            _, model = run(
                                exe,
                                d,
                                label + cap,
                                [
                                    "--folds-file",
                                    d / "folds",
                                    "--metric",
                                    "rmse",
                                    "--memory-mb",
                                    cap,
                                ],
                                report,
                            )
                            scores.append((model["alpha"], model["beta"]))
                        if label == "new":
                            run(
                                exe,
                                d,
                                label + "partial",
                                [
                                    "--folds-file",
                                    d / "folds",
                                    "--metric",
                                    "rmse",
                                    "--memory-mb",
                                    "0.8",
                                ],
                                report,
                            )
                            assert (
                                rows(d / (label + "partial") / "tuning.tsv")[0][
                                    "search_coverage"
                                ]
                                == "partial"
                            )
                            assert any(
                                r["candidate_status"] == "uncertified"
                                for r in rows(d / (label + "partial") / "search.tsv")
                            )
                            dense = rows(d / "new200" / "search.tsv")
                            operator = rows(d / "new4" / "search.tsv")
                            assert len(dense) == len(operator)
                            for x, y in zip(dense, operator):
                                for key in (
                                    "alpha",
                                    "beta",
                                    "stage",
                                    "valid",
                                    "full_status",
                                ):
                                    assert x[key] == y[key], (key, x, y)
                                if x["valid"] == "1":
                                    for key in ("r1f1", "r1f2", "mean"):
                                        assert (
                                            abs(float(x[key]) - float(y[key])) < 1e-10
                                        )
                            assert scores[0] == scores[1] == ("-0.125", "-0.125"), (
                                scores
                            )
                    else:
                        alpha, beta = (
                            (".2", ".5") if case == "connected" else ("0", "1")
                        )
                        pred, _ = run(
                            exe,
                            dataset,
                            label,
                            [
                                "--alpha",
                                alpha,
                                "--beta",
                                beta,
                                "--memory-mb",
                                "256",
                                "--cache-backend",
                                "operator",
                                "--pd-policy",
                                "nearPD",
                            ],
                            report,
                        )
                        if label == "new" and case == "cross":
                            dense_cross_reference(exe, dataset, pred)
                            usage = rows(dataset / label / "cache_usage.tsv")[0]
                            assert int(usage["feature_bytes"]) <= 12 * 32
                        if label == "new" and a.baseline:
                            old = rows(dataset / "old" / "predictions.tsv")
                            assert [r["id"] for r in pred] == [r["id"] for r in old]
                            assert (
                                max(
                                    abs(float(x[k]) - float(y[k]))
                                    for x, y in zip(old, pred)
                                    for k in ("u", "yhat")
                                )
                                < 1e-8
                            )
                        if case == "stats" and name == "filtered":
                            assert (
                                dataset / label / "predictions.tsv"
                            ).read_bytes() == (
                                d / label / "predictions.tsv"
                            ).read_bytes()


if __name__ == "__main__":
    main()
