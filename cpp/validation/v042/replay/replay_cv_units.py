"""Replay RMSE unit sensitivity on an analytic 32-individual, 4-fold example."""
import argparse
import csv
import os
from pathlib import Path
import subprocess

import numpy as np


def write_table(path, header, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def read_table(path):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    binary, out = args.binary.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    n, m = 32, 64
    ids = [f"i{i:03}" for i in range(n)]
    labels = np.arange(n) % 4
    rng = np.random.default_rng(178)
    x, z = rng.normal(size=(2, n, m))
    kernels = {"A": np.eye(n), "G": x @ x.T / m, "O": z @ z.T / m}
    for name, matrix in kernels.items():
        write_table(out / (name + ".tsv"), ["ID", *ids],
                    ([key, *row] for key, row in zip(ids, matrix)))
    write_table(out / "folds.tsv", ["repeat", "fold", "id"],
                ((1, fold + 1, key) for fold, key in zip(labels, ids)))
    alpha, beta = 0.4375, 0.1875
    k = ((1 - alpha - beta + 1e-6) * kernels["A"]
         + beta * kernels["G"] + alpha * kernels["O"])
    y = rng.normal(size=n)
    target = np.empty_like(y)
    for fold in range(4):
        t = np.flatnonzero(labels != fold)
        v = np.flatnonzero(labels == fold)
        h = k[np.ix_(t, t)] + np.eye(t.size)
        q = np.linalg.solve(h, np.column_stack((y[t], np.ones(t.size))))
        r = q[:, 0] - q[:, 1] * (q[:, 0].sum() / q[:, 1].sum())
        target[v] = k[np.ix_(v, t)] @ r

    results = []
    for scale in [1, 0.001]:
        pheno = out / f"pheno_{scale}.tsv"
        write_table(pheno, ["ID", "trait", "target"], zip(ids, scale*y, scale*target))
        for solver in ["pcg", "direct"]:
            dest = out / f"{solver}_{scale}"
            cmd = [binary, "fit"]
            for name in kernels:
                cmd.extend(["--" + name, out / (name + ".tsv")])
            cmd.extend(["--pheno", pheno, "--trait", "trait", "--score-col", "target",
                        "--metric", "rmse", "--folds-file", out / "folds.tsv",
                        "--strict-simplex", "--solver", solver, "--varcomp", "fixed",
                        "--lambda", "1", "--log-level", "error", "--out", dest])
            subprocess.run(cmd, env=dict(os.environ, OPENBLAS_NUM_THREADS="1"), check=True)
            summary = read_table(dest / "tuning.tsv")[0]
            results.append([solver, scale, len(read_table(dest / "search.tsv")),
                            summary["selected_alpha"], summary["selected_beta"],
                            summary["selection_score"], summary["search_coverage"]])
    path = out / "observed.tsv"
    write_table(path, ["solver", "trait_scale", "candidates", "alpha", "beta", "rmse", "coverage"], results)
    print(path.read_text(), end="")

if __name__ == "__main__":
    main()
