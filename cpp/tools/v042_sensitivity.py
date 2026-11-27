"""Measure retry costs, complete candidate rankings and ID-aligned predictions."""

import argparse
import csv
import subprocess
import time
from pathlib import Path


def read(path):
    if not path.is_file():
        return []
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def save(path, records):
    with path.open("w") as f:
        w = csv.DictWriter(f, records[0], delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(records)


def ranks(rows, key, value, reverse):
    return {
        key(row): i for i, row in enumerate(sorted(rows, key=value, reverse=reverse), 1)
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("exe", "inputs", "out"):
        p.add_argument(k, type=Path)
    p.add_argument("--mc-tol", default=".10")
    p.add_argument("--max-probes", default="8")
    p.add_argument("--trace-rank", default="8")
    a = p.parse_args()
    a.out.mkdir()
    summaries = []
    comparisons = []
    for seed in (42, 19):
        for budget in (0, 256):
            dest = a.out / f"seed{seed}-retry{budget}"
            opts = [
                "fit",
                "--cache",
                a.inputs / "cache",
                "--cache-backend",
                "operator",
                "--cache-storage",
                "auto",
                "--pheno",
                a.inputs / "small.tsv",
                "--trait",
                "trait",
                "--folds-file",
                a.inputs / "small-folds.tsv",
                "--metric",
                "rmse",
                "--strict-simplex",
                "--parallel",
                "weights",
                "--threads",
                2,
                "--memory-mb",
                512,
                "--varcomp",
                "ai-reml",
                "--trace",
                "hutchinson",
                "--probes",
                4,
                "--max-probes",
                a.max_probes,
                "--retry-probes",
                budget,
                "--mc-tol",
                a.mc_tol,
                "--probe-seed",
                seed,
                "--probe-solver",
                "lanczos",
                "--trace-rank",
                a.trace_rank,
                "--slq-depth",
                32,
                "--out",
                dest,
                "--log-level",
                "error",
            ]
            start = time.monotonic()
            r = subprocess.run(
                list(map(str, [a.exe, *opts])), capture_output=True, text=True
            )
            elapsed = time.monotonic() - start
            t = read(dest / "tuning.tsv")[0]
            s = read(dest / "search.tsv")
            pred = read(dest / "predictions.tsv")
            fits = read(dest / "cv_pcg_summary.tsv") + read(dest / "pcg_summary.tsv")
            usage = (read(dest / "cache_usage.tsv") or [{}])[0]
            rt = read(dest / "runtime.tsv")[0]
            summaries.append(
                dict(
                    seed=seed,
                    budget=budget,
                    exit=r.returncode,
                    status=r.stderr.strip(),
                    scored=t["candidates_scored"],
                    visited=t["candidates_visited"],
                    completed=t["folds_completed"],
                    spent=t["retry_reserved"],
                    exhausted=t["probe_exhausted"],
                    alpha=t["selected_alpha"],
                    beta=t["selected_beta"],
                    rhs=sum(int(i["rhs"]) for i in fits),
                    matvec=sum(int(i["matvec"]) for i in fits),
                    probes=sum(int(i["probes"]) for i in fits),
                    wall_s=elapsed,
                    rss_KiB=rt["peak_rss_kib"],
                    feature_B=usage.get("feature_bytes", "NA"),
                    verified_B=usage.get("verified_cache_bytes", "NA"),
                )
            )
            save(a.out / "summary.tsv", summaries)
    names = [(42, 0), (42, 256), (19, 0), (19, 256)]
    for l, r in [(0, 1), (2, 3), (0, 2), (1, 3)]:
        left, right = [a.out / f"seed{names[i][0]}-retry{names[i][1]}" for i in (l, r)]

        def scores(d):
            return {
                (i["alpha"], i["beta"]): float(i["mean"])
                for i in read(d / "search.tsv")
                if i["valid"] == "1"
            }

        x, y = scores(left), scores(right)
        keys = x.keys() & y.keys()
        p, q = read(left / "predictions.tsv"), read(right / "predictions.tsv")
        row = dict(
            left=left.name,
            right=right.name,
            common_scored=len(keys),
            max_common_weight_rank_delta=max(
                (
                    abs(i - sorted(keys, key=y.get).index(k))
                    for i, k in enumerate(sorted(keys, key=x.get))
                ),
                default=None,
            ),
            max_du=None,
            max_dyhat=None,
            max_candidate_rank_delta=None,
        )
        if p and q:
            assert [v["id"] for v in p] == [v["id"] for v in q]
            for k in ("u", "yhat"):
                row["max_d" + k] = max(
                    abs(float(a[k]) - float(b[k])) for a, b in zip(p, q)
                )
            rl, rr = [
                ranks(
                    [i for i in z if i["role"] == "candidate"],
                    lambda a: a["id"],
                    lambda a: float(a["u"]),
                    True,
                )
                for z in (p, q)
            ]
            row["max_candidate_rank_delta"] = max(abs(rl[k] - rr[k]) for k in rl)
        comparisons.append(row)
    save(a.out / "comparisons.tsv", comparisons)


if __name__ == "__main__":
    main()
