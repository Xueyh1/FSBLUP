"""Fit a full-sib-style constant-off-diagonal relationship matrix with an intercept.
Usage: python replay_identifiability.py /path/to/fsblup NEW_OUTPUT_DIRECTORY
The REML contrasts see only (.5+1e-6)*Vu + Ve.
"""
import argparse
import csv
from pathlib import Path
import subprocess

import numpy as np


def read_first(path):
    with path.open() as stream:
        return next(csv.DictReader(stream, delimiter="\t"))


def run(exe, root):
    root.mkdir(parents=True)
    ids = [f"animal{i}" for i in range(8)]
    y = np.sin(np.arange(8.)**2)
    raw = .5*np.eye(8) + .5*np.ones((8,8))
    for filename, columns, rows in [
        ("K.tsv", ["id", *ids], [[i,*r] for i, r in zip(ids, raw)]),
        ("y.tsv", ["id", "y"], zip(ids, y))]:
        with (root / filename).open("w", newline="") as outf:
            writer = csv.writer(outf, delimiter="\t", lineterminator="\n")
            writer.writerow(columns)
            writer.writerows(rows)
    with (root / "summary.tsv").open("w", newline="") as outf:
        writer = csv.writer(outf, delimiter="\t", lineterminator="\n")
        writer.writerow(["method", "trace", "exit", "status", "updates", "Vu", "Ve"])
        for method in ["ai-reml", "em-reml"]:
            for trace in ["exact", "hutchinson"]:
                output = root / (method+"-"+trace)
                cmd = [str(exe), "fit", *[v for layer in "AGO" for v in ["--"+layer, str(root/"K.tsv")]],
                       "--pheno", str(root/"y.tsv"), "--id", "id", "--trait", "y", "--alpha", "0", "--beta", "0",
                       "--varcomp", method, "--trace", trace, "--out", str(output)]
                p = subprocess.run(cmd, capture_output=True, text=True)
                (root / (output.name + ".stderr")).write_text(p.stderr)
                summary = read_first(output/"pcg_summary.tsv")
                if p.returncode:
                    writer.writerow([method, trace, p.returncode, p.stderr.strip().splitlines()[-1], summary["updates"], "", ""])
                else:
                    model = read_first(output/"model.tsv")
                    writer.writerow([method, trace, 0, model["status"], summary["updates"], model["Vu"], model["Ve"]])
    print((root / "summary.tsv").read_text(), end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fsblup", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    run(args.fsblup.resolve(), args.out.resolve())
