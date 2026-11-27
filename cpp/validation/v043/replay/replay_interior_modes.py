"""Two interior REML modes and a reversal of candidate breeding-value ranks.
Run: python replay_interior_modes.py /absolute/path/to/fsblup NEW_OUTPUT_DIR
Dependencies: Python 3, NumPy, SciPy. Creates explicit TSV inputs and summaries.
"""
import argparse
import csv
from pathlib import Path
import subprocess

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import brentq

DIAGONAL = np.array([1e-4, 1e-3, .01, .1, 1., 10., 100., 1000.])
PHENOTYPE = np.array([.15860451755560323, .2830747764750611, .29163276057487875,
                      3.453537174827784, -1.3742288680750976, -7.226413775202968,
                      -1.224989746234572, 5.63878316007931])


def inputs(folder):
    train = len(DIAGONAL)
    names = [f"T{i+1}" for i in range(train)] + ["C1", "C2"]
    cov = np.diag(np.r_[DIAGONAL, 1., 1.])
    for i, j in [(train, 3), (train + 1, 7)]:
        cov[i, j] = cov[j, i] = .5 * np.sqrt(cov[j, j])
    raw = cov - 1e-6 * np.eye(len(cov))   # the public fusion CLI adds this back
    np.linalg.cholesky(raw)
    for name, header, data in [
        ("K.tsv", ["id", *names], [[a, *(format(x, ".17g") for x in row)] for a, row in zip(names, raw)]),
        ("y.tsv", ["id", "y"], zip(names, [*(format(x, ".17g") for x in PHENOTYPE), "NA", "NA"])),
    ]:
        with (folder / name).open("w", newline="") as file:
            writer = csv.writer(file, delimiter="\t", lineterminator="\n")
            writer.writerow(header)
            writer.writerows(data)
    return cov


def oracle(cov):
    n = len(PHENOTYPE)
    contrast = null_space(np.ones((1, n)))
    k, vec = np.linalg.eigh(contrast.T @ cov[:n, :n] @ contrast)
    energies = (vec.T @ contrast.T @ PHENOTYPE) ** 2
    df = len(energies)

    def score(t):
        ratio = np.exp(t)
        den = k + ratio
        return .5 * ratio * (df * np.sum(energies / den**2) / np.sum(energies / den) - np.sum(1/den))

    grid = np.linspace(-18, 22, 4001)
    ss = [score(t) for t in grid]
    peaks = []
    for t0, t1, s0, s1 in zip(grid, grid[1:], ss, ss[1:]):
        if s0 > 0 and s1 <= 0:
            ratio = np.exp(brentq(score, t0, t1))
            g = np.mean(energies / (k + ratio))
            peaks.append(("interior", g, ratio*g))
    peaks.extend([("genetic-zero", 0., np.mean(energies)), ("residual-zero", np.mean(energies/k), 0.)])
    for status, g, e in peaks:
        d = g*k + e
        lik = -.5 * np.sum(np.log(d) + energies/d)
        den = g*DIAGONAL + e
        mu = np.sum(PHENOTYPE/den) / np.sum(1/den)
        u = g * cov[n:, :n] @ ((PHENOTYPE - mu)/den)
        yield ["scipy-profile", status, g, e, lik, *u, "", ""]


def public_fits(exe, folder):
    options = [("direct-profile", ["--varcomp", "profile-reml", "--solver", "direct"])]
    for method in ["ai-reml", "em-reml"]:
        for trace, start in [("exact", []), ("hutchinson", []), ("hutchinson", ["--init-vu", ".024", "--init-ve", "10.6"])]:
            name = f"{method}-{trace}" + ("-alternate" if start else "")
            options.append((name, ["--varcomp", method, "--trace", trace, *start]))
    for name, args in options:
        output = folder / name
        command = [str(exe), "fit", *[v for layer in "AGO" for v in ["--"+layer, str(folder / "K.tsv")]],
                   "--pheno", str(folder / "y.tsv"), "--id", "id", "--trait", "y", "--alpha", "0",
                   "--beta", "0", "--out", str(output), *args]
        proc = subprocess.run(command, capture_output=True, text=True)
        (folder / (name + ".stderr")).write_text(proc.stderr)
        if proc.returncode:
            yield [name, "failed", "", "", "", "", "", "", proc.returncode]
            continue
        def rows(filename):
            with (output / filename).open() as file:
                return list(csv.DictReader(file, delimiter="\t"))
        model = rows("model.tsv")[0]
        candidates = {r["id"]:r["u"] for r in rows("predictions.tsv") if r["role"] == "candidate"}
        yield [name, model["status"], model["Vu"], model["Ve"], model["contrast_loglik"],
               candidates["C1"], candidates["C2"], rows("pcg_summary.tsv")[0]["logdet_se"], proc.returncode]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fsblup", type=Path)
    p.add_argument("new_folder", type=Path)
    args = p.parse_args()
    folder = args.new_folder.resolve()
    folder.mkdir(parents=True)
    cov = inputs(folder)
    with (folder / "summary.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["method", "status", "Vu", "Ve", "contrast_loglik", "u_C1", "u_C2", "logdet_SE", "exit"])
        w.writerows(oracle(cov))
        w.writerows(public_fits(args.fsblup.resolve(), folder))
    print((folder / "summary.tsv").read_text(), end="")


if __name__ == "__main__":
    main()
