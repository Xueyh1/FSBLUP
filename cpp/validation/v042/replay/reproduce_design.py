"""Compare two full-rank bases for the same fixed-effect design space."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from scipy.linalg import hadamard


def write_table(path, header, data, formats):
    np.savetxt(path, data, header="\t".join(header), delimiter="\t",
               comments="", fmt=formats)


def run_fit(executable, kernel, phenotype, out, solver, varcomp):
    args = [executable, "fit", "--A", kernel, "--G", kernel, "--O", kernel,
            "--pheno", phenotype, "--id", "id", "--trait", "y", "--alpha", "0",
            "--beta", "0", "--out", out, "--solver", solver, "--varcomp", varcomp,
            "--covar", "x,w"]
    if varcomp == "fixed":
        args.extend(["--lambda", ".5"])
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        return [result.returncode, result.stderr.strip().splitlines()[-1], "", "", ""]
    with (out / "model.tsv").open() as file:
        model = next(csv.DictReader(file, delimiter="\t"))
    return [result.returncode, model["status"], model["Vu"], model["Ve"],
            model["contrast_loglik"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="fsblup-design-"))
    print(f"raw files: {root}", file=sys.stderr)
    ids = np.arange(1, 9)
    q = hadamard(8)
    y = q[:, 3:] @ np.sqrt([3, 5, 7, 9, 11]) / np.sqrt(8)
    k = q @ np.diag([2, 2, 2, 1, 2, 3, 4, 5]) @ q.T / 8 - 1e-6 * np.eye(8)
    kernel = root / "K.tsv"
    write_table(kernel, ["id", *map(str, ids)], np.column_stack((ids, k)),
                ["%d"] + ["%.17g"] * 8)
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    writer.writerow(["basis", "cond_X", "solver", "varcomp", "exit", "status",
                     "Vu", "Ve", "contrast_loglik"])
    for basis, last in [("orthogonal", q[:, 2]), ("near-collinear", q[:, 1] + 1e-7 * q[:, 2])]:
        cond = np.linalg.cond(np.column_stack((np.ones(8), q[:, 1], last)))
        phenotype = root / f"{basis}.tsv"
        write_table(phenotype, ["id", "y", "x", "w"], np.column_stack((ids, y, q[:, 1], last)),
                    ["%d", "%.17g", "%.17g", "%.17g"])
        for solver, varcomp in [("direct", "profile-reml"), ("pcg", "ai-reml"), ("pcg", "fixed")]:
            out = root / f"{basis}-{solver}-{varcomp}"
            result = run_fit(args.binary.resolve(), kernel, phenotype, out, solver, varcomp)
            writer.writerow([basis, cond, solver, varcomp, *result])


if __name__ == "__main__":
    main()
