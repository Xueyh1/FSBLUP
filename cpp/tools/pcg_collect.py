"""Join scale job records to the per-fit diagnostics, including failed jobs."""

import argparse
import csv
from pathlib import Path


def read(path):
    if not path.is_file():
        return []
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def collect(path):
    for row in read(path):
        n = int(row["N_A"])
        mode = "A" if n == int(row["N_G"]) else "C"
        cohort = "large" if int(row["n_train"]) == n - (n + 4) // 5 else "small"
        folder = path.parent / f"{mode}-{n}-{row['markers']}-{row['omics']}"
        dest = folder / (
            "cache"
            if row["method"] == "prepare"
            else f"{path.stem}-{cohort}-{row['method']}-{row['threads']}"
        )
        result = dict(experiment=path.stem, mode=mode, **row)
        if row["status"] == "0":
            rt = read(dest / "runtime.tsv")[0]
            result.update(blas_threads=rt["threads"], workers=rt["weight_workers"])
            if row["method"] != "prepare":
                info = read(dest / "pcg_summary.tsv")[0]
                cv = read(dest / "cv_pcg_summary.tsv")
                kernel = read(dest / "cache_usage.tsv")[0]
                model = read(dest / "model.tsv")[0]
                weights = read(dest / "fusion.tsv")[0]
                iterations = {}
                for tr in read(dest / "cv_reml_trace.tsv") + read(
                    dest / "reml_trace.tsv"
                ):
                    iterations[tr["context"]] = max(
                        iterations.get(tr["context"], 0), int(tr["iteration"])
                    )
                records = cv + [info]
                result.update(
                    estimator=row["method"].removeprefix("cv-"),
                    solver=info["solver"],
                    status_final=model["status"],
                    Vu=model["Vu"],
                    Ve=model["Ve"],
                    ratio=model["lambda"],
                    alpha=weights["alpha"],
                    beta=weights["beta"],
                    eligible=sum(r["valid"] == "1" for r in read(dest / "search.tsv"))
                    if cv
                    else 1,
                    folds=len(read(dest / "fold_counts.tsv")),
                    min_fold_train=min((int(x["n_train"]) for x in cv), default="NA"),
                    max_fold_train=max((int(x["n_train"]) for x in cv), default="NA"),
                    last_recorded_rounds=sum(iterations.values()),
                    probes=info.get("probes", ""),
                    trace="hutchinson"
                    if info.get("probes", "0") != "0"
                    else ("none" if row["method"] == "fixed" else "exact"),
                    max_residual_over_tol=max(
                        float(r["max_true_residual_over_tolerance"]) for r in records
                    ),
                    verified_bytes=kernel.get("verified_cache_bytes", ""),
                    preconditioner=info.get("preconditioner", ""),
                    read_calls=kernel.get("read_calls", ""),
                    read_s=kernel.get("read_seconds", ""),
                    decode_s=kernel.get("decode_seconds", ""),
                    estimate_s=info.get("estimation_s", ""),
                    gls_s=info.get("gls_s", ""),
                    predict_s=info.get("predict_s", ""),
                    cross_rhs=sum(int(r.get("cross_rhs", 0)) for r in records)
                    if "cross_rhs" in info
                    else "",
                )
        yield result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("logs", nargs="+", type=Path)
    a = p.parse_args()
    allrows = [r for path in a.logs for r in collect(path)]
    columns = dict.fromkeys(k for r in allrows for k in r)
    with a.output.open("w") as f:
        w = csv.DictWriter(
            f, fieldnames=list(columns), delimiter="\t", lineterminator="\n"
        )
        w.writeheader()
        w.writerows(allrows)


if __name__ == "__main__":
    main()
