"""Summarize raw 0.4.2 replay trees. A failed fit retains its status in the table."""

import argparse
import csv
import itertools
import gzip
from pathlib import Path
import statistics


def read(p):
    with p.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write(p, rs):
    with p.open("w") as f:
        w = csv.DictWriter(f, rs[0], delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rs)


def align(left, right):
    l = read(left / "predictions.tsv")
    r = read(right / "predictions.tsv")
    assert [(x["id"], x["role"]) for x in l] == [(x["id"], x["role"]) for x in r]
    return {
        f"max_d{k}": max(abs(float(x[k]) - float(y[k])) for x, y in zip(l, r))
        for k in ("u", "yhat")
    }


def candidate_ranks(folder):
    rows = [r for r in read(folder / "search.tsv") if r["valid"] == "1"]
    tuning = read(folder / "tuning.tsv")[0]
    tol = float(tuning["comparison_tie"]) * float(tuning["score_scale"])
    ordered = []
    while rows:
        best = 0
        for j in range(1, len(rows)):
            if float(rows[j]["mean"]) < float(rows[best]["mean"]) - tol:
                best = j
        r = rows.pop(best)
        ordered.append(
            dict(
                rank=len(ordered) + 1,
                alpha=r["alpha"],
                beta=r["beta"],
                rmse=r["mean"],
                selected=int(r["full_status"] == "selected"),
            )
        )
    return ordered


def id_deltas(writer, case, left, right):
    if (
        not (left / "predictions.tsv").exists()
        or not (right / "predictions.tsv").exists()
    ):
        return
    a, b = (read(p / "predictions.tsv") for p in (left, right))

    def ranks(rows):
        candidates = [r for r in rows if r["role"] == "candidate"]
        return {
            r["id"]: j
            for j, r in enumerate(sorted(candidates, key=lambda r: -float(r["u"])), 1)
        }

    rank_a, rank_b = ranks(a), ranks(b)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        key = x["id"]
        assert (key, x["role"]) == (y["id"], y["role"])
        writer.writerow(
            [
                case,
                left.name,
                right.name,
                key,
                x["role"],
                float(y["u"]) - float(x["u"]),
                float(y["yhat"]) - float(x["yhat"]),
                rank_a.get(key, "NA"),
                rank_b.get(key, "NA"),
            ]
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--wide-reference", type=Path)
    a = p.parse_args()
    d = a.root
    out = a.out
    out.mkdir(exist_ok=True, parents=True)
    if (d / "ops" / "operations.tsv").exists():
        rows = read(d / "ops" / "operations.tsv")
        keys = ("version", "mode", "n", "alpha")
        key = lambda r: tuple(r[k] for k in keys)
        summary = []
        for _, group in itertools.groupby(sorted(rows, key=key), key):
            group = list(group)
            row = group[0].copy()
            row.pop("rep")
            for k in ("wall_s", "rss_KiB"):
                v = [float(g[k]) for g in group]
                row[k] = statistics.median(v)
                row[k + "_MAD"] = statistics.median(abs(value - row[k]) for value in v)
            summary.append(row)
        write(out / "B-medians.tsv", summary)
    profiles = []
    for name in ("old", "new"):
        massif = d / ("massif-C-" + name)
        if massif.exists():
            snapshots = []
            for line in massif.read_text().splitlines():
                if line.startswith("snapshot="):
                    snapshots.append({})
                if line.startswith("mem_heap"):
                    k, v = line.split("=")
                    snapshots[-1][k] = int(v)
            profiles.append(
                dict(
                    binary=name,
                    peak_heap_with_overhead_B=max(
                        v["mem_heap_B"] + v["mem_heap_extra_B"] for v in snapshots
                    ),
                )
            )
    if profiles:
        write(out / "C-heap.tsv", profiles)
    for filename, table in [
        ("summary", "probe-budgets"),
        ("comparisons", "probe-ranking"),
    ]:
        rs = []
        for ceiling in (8, 16, 32, 64):
            folder = d / f"panel-{ceiling}"
            f = folder / (filename + ".tsv")
            if f.is_file():
                for row in read(f):
                    if filename == "comparisons":
                        ranked = [
                            [
                                (r["alpha"], r["beta"])
                                for r in candidate_ranks(folder / row[side])
                            ]
                            for side in ("left", "right")
                        ]
                        common = set(ranked[0]) & set(ranked[1])
                        l, r = [[k for k in side if k in common] for side in ranked]
                        row["max_common_weight_rank_delta"] = max(
                            (abs(i - r.index(k)) for i, k in enumerate(l)), default=None
                        )
                    rs.append(dict(max_probes=ceiling, **row))
        if rs:
            write(out / (table + ".tsv"), rs)
    ranking = []
    with gzip.open(d / "ID-deltas.tsv.gz", "wt") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "case",
                "left",
                "right",
                "id",
                "role",
                "u_right_minus_left",
                "yhat_right_minus_left",
                "rank_left",
                "rank_right",
            ]
        )
        for ceiling in (8, 16, 32, 64):
            folder = d / f"panel-{ceiling}"
            if not (folder / "comparisons.tsv").is_file():
                continue
            for row in read(folder / "comparisons.tsv"):
                id_deltas(
                    writer, str(ceiling), folder / row["left"], folder / row["right"]
                )
            for path in sorted(folder.glob("seed*-retry*")):
                t = read(path / "tuning.tsv")[0]
                fits = read(path / "cv_pcg_summary.tsv")
                assert len(fits) == int(t["fold_attempts"])
                successes = sum(
                    not fit["status"].startswith(fit["context"] + ":") for fit in fits
                )
                assert successes == int(t["folds_completed"]), (path, successes, t)
                ranking.extend(
                    dict(max_probes=ceiling, case=path.name, **r)
                    for r in candidate_ranks(path)
                )
        if a.wide_reference:
            id_deltas(writer, "wide", a.wide_reference, d / "wide-final")
    if ranking:
        write(out / "candidate-ranks.tsv", ranking)
    timing = d / "measured"
    if (timing / "medians.tsv").is_file():
        write(out / "thread-medians.tsv", read(timing / "medians.tsv"))
        checks = []
        for case, nthreads in itertools.product(("small", "medium"), (1, 2, 4, 8)):
            left, right = [
                timing / f"{case}-{version}-{nthreads}-1" for version in ("old", "new")
            ]
            x, y = [read(root / "search.tsv") for root in (left, right)]
            assert [(r["alpha"], r["beta"], r["stage"]) for r in x] == [
                (r["alpha"], r["beta"], r["stage"]) for r in y
            ]
            delta = max(
                abs(float(i["mean"]) - float(j["mean"]))
                for i, j in zip(x, y)
                if i["valid"] == "1"
            )
            diffs = align(left, right)
            assert max(delta, *diffs.values()) < 1e-8
            checks.append(
                dict(
                    case=case,
                    threads=nthreads,
                    points=len(x),
                    max_mean_delta=delta,
                    **diffs,
                )
            )
        write(out / "thread-equivalence.tsv", checks)
    if a.wide_reference:
        rows = []
        for version, root in [("0.4.1", a.wide_reference), ("0.4.2", d / "wide-final")]:
            info = (
                read(root / "model.tsv")[0]
                if (root / "model.tsv").is_file()
                else {
                    "reference": 120000,
                    "candidate": 30000,
                    "status": (root / "failure.txt").read_text().strip(),
                }
            )
            rt = read(root / "runtime.tsv")[0]
            s = read(root / "search.tsv")
            t = read(root / "tuning.tsv")[0]
            u = read(root / "cache_usage.tsv")[0]
            allfits = read(root / "cv_pcg_summary.tsv")
            if (root / "pcg_summary.tsv").is_file():
                allfits += read(root / "pcg_summary.tsv")
            rows.append(
                dict(
                    version=version,
                    fit_status=info["status"],
                    train=info["reference"],
                    candidate=info["candidate"],
                    SNP=50000,
                    seed=42,
                    folds=2,
                    visited=len(s),
                    scored=sum(r["valid"] == "1" for r in s),
                    completed=sum(
                        int(
                            r.get(
                                "folds_completed",
                                sum(r[k] != "nan" for k in ("r1f1", "r1f2")),
                            )
                        )
                        for r in s
                    ),
                    alpha=t["selected_alpha"],
                    beta=t["selected_beta"],
                    rmse=t["selection_score"],
                    spend=t.get("retry_reserved", 0),
                    wall_s=rt["wall_seconds"],
                    rss_KiB=rt["peak_rss_kib"],
                    rhs=sum(int(r["rhs"]) for r in allfits),
                    matvec=sum(int(r["matvec"]) for r in allfits),
                    probes=sum(int(r["probes"]) for r in allfits),
                    feature_B=u["feature_bytes"],
                    verified_B=u["verified_cache_bytes"],
                    calibration_B=u.get("calibration_bytes", "NA"),
                    preload_B=u.get("training_preload_bytes", "NA"),
                    training_B=u.get("training_bytes", "NA"),
                    cross_B=u.get("cross_bytes", "NA"),
                )
            )
        write(out / "wide.tsv", rows)
        if (d / "wide-final" / "model.tsv").is_file():
            write(
                out / "wide-predictions.tsv",
                [align(a.wide_reference, d / "wide-final")],
            )


if __name__ == "__main__":
    main()
