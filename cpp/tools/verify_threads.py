"""Validate official raw/fixed paths and frozen nested fold scores at each CPU count."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from nearpd_bench import physical_cpus, rows, save


def matrix(path, delim):
    # Keep string IDs including leading zeros on both axes.
    table = pd.read_csv(path, sep=delim, dtype={0: str}, float_precision="round_trip")
    return table.set_index(table.columns[0])


def max_error(left, right):
    assert set(left.index) == set(right.index) and set(left.columns) == set(
        right.columns
    )
    return float(
        np.max(
            np.abs(left.loc[right.index, right.columns].to_numpy() - right.to_numpy())
        )
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "official", "reference", "output"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    nested = Path(__file__).resolve().parents[1] / "tests/nested"
    official = args.official
    ref = args.reference
    fold_ref = rows(nested / "scores.tsv")
    cpus = physical_cpus()
    assert len(cpus) >= 8
    summaries = []
    for n in (1, 2, 4, 8):
        with tempfile.TemporaryDirectory(dir=args.output) as td:
            tmp = Path(td)

            def run(cmd, *options):
                flags = [
                    "taskset",
                    "-c",
                    ",".join(map(str, cpus[:n])),
                    args.binary.resolve(),
                    cmd,
                    "--threads",
                    n,
                    *options,
                ]
                subprocess.run(
                    list(map(str, flags)),
                    check=True,
                    env={**os.environ, "OPENBLAS_NUM_THREADS": str(n)},
                )

            run(
                "build",
                "--ped",
                official / "test_pedi.csv",
                "--snp",
                official / "test_snp.csv",
                "--omics",
                official / "test_omic.csv",
                "--omics-type",
                "fpkm",
                "--out",
                tmp / "raw",
            )
            rec = dict(threads=n)
            for layer in "AGO":
                error = max_error(
                    matrix(tmp / "raw" / (layer + ".tsv"), "\t"),
                    matrix(official / ("test_mat" + layer + ".csv"), ","),
                )
                assert error < 1e-11, (n, layer, error)
                rec["max_abs_" + layer] = error
            names = [
                s
                for k in "AGO"
                for s in ("--" + k, official / ("test_mat" + k + ".csv"))
            ]
            run(
                "fit",
                *names,
                "--pheno",
                ref / "masked.tsv",
                "--id",
                "id",
                "--trait",
                "age",
                "--alpha",
                ".2",
                "--beta",
                ".5",
                "--pd-policy",
                "nearPD",
                "--write-k",
                "--out",
                tmp / "fixed",
            )
            result = subprocess.check_output(
                [
                    sys.executable,
                    Path(__file__).with_name("nearpd_compare.py"),
                    ref,
                    tmp / "fixed",
                ],
                env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
            )
            (args.output / f"official-{n}.json").write_bytes(result)
            pd_stat = rows(tmp / "fixed/pd.tsv")[0]
            assert (
                pd_stat["iterations"] == "2"
                and abs(float(pd_stat["relative_normF"]) - 0.0855089025421) < 1e-11
            )
            cand = [
                r
                for r in rows(tmp / "fixed/predictions.tsv")
                if r["role"] == "candidate"
            ]
            ref_ids = {r["id"] for r in cand}

            def rank_order(rr):
                return [
                    r["id"] for r in sorted(rr, key=lambda r: (-float(r["u"]), r["id"]))
                ]

            assert rank_order(cand) == rank_order(
                [r for r in rows(ref / "r_predictions.tsv") if r["id"] in ref_ids]
            )
            rec["candidate_rank_matches"] = len(cand)
            opts = [s for k in "AGO" for s in ("--" + k, nested / (k + ".tsv"))]
            for mode in ("blas", "weights"):
                dest = tmp / mode
                run(
                    "fit",
                    *opts,
                    "--pheno",
                    nested / "pheno.tsv",
                    "--id",
                    "ID",
                    "--trait",
                    "trait",
                    "--folds-file",
                    nested / "folds.tsv",
                    "--metric",
                    "rmse",
                    "--strict-simplex",
                    "--pd-policy",
                    "nearPD",
                    "--parallel",
                    mode,
                    "--write-k",
                    "--out",
                    dest,
                )
                trace = rows(dest / "search.tsv")
                assert len(trace) == len(fold_ref) and all(
                    (r["alpha"], r["beta"]) == (s["alpha"], s["beta"])
                    for r, s in zip(trace, fold_ref)
                )
                rec[mode + "_max_fold_delta"] = max(
                    abs(float(r[c]) - float(s[c]))
                    for r, s in zip(trace, fold_ref)
                    for c in ("r1f1", "r1f2", "r1f3", "mean")
                )
                assert rec[mode + "_max_fold_delta"] < 1e-6
                winner = rows(dest / "fusion.tsv")[0]
                assert (winner["alpha"], winner["beta"]) == ("0.75", "0.25")
                assert (
                    max_error(
                        matrix(dest / "K.tsv", "\t"),
                        matrix(nested / "selected_K.tsv", "\t"),
                    )
                    < 1e-11
                )
            summaries.append(rec)
    save(args.output / "threads.tsv", summaries)


if __name__ == "__main__":
    main()
