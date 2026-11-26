"""Exact/profile/EM multi-start boundary comparison; the CLI adds one jitter."""

import argparse
import csv
import subprocess
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("binary", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    d = a.out
    d.mkdir(parents=True, exist_ok=False)
    q = (
        np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], float)
        / 2
    )
    diag = np.array([2, 1, 2, 4])
    k = q @ np.diag(diag) @ q.T
    ids = ["01", "02", "03", "04"]
    with (d / "k.tsv").open("w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ID", *ids])
        w.writerows([key, *v] for key, v in zip(ids, k))
    values = []
    for case, z in [
        ("interior", np.sqrt(2 * (diag[1:] + 1e-6) + 1)),
        ("near-zero-g", np.sqrt(0.001 * (diag[1:] + 1e-6) + 1)),
        ("zero-g", [1, 0, 0]),
        ("zero-e", [0, 0, 1]),
    ]:
        y = q[:, 1:] @ z
        with (d / "pheno.tsv").open("w") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["ID", "y"])
            w.writerows(zip(ids, y))
        for method in ("ai-reml", "em-reml", "profile-reml"):
            for g0, e0 in (
                [(0.1, 0.5), (5.0, 0.1)] if method != "profile-reml" else [(None, None)]
            ):
                dest = d / f"{case}-{method}-{g0}"
                cmd = [
                    a.binary.resolve(),
                    "fit",
                    "--varcomp",
                    method,
                    "--solver",
                    "direct",
                    "--trait",
                    "y",
                    "--pheno",
                    d / "pheno.tsv",
                    "--alpha",
                    "0",
                    "--beta",
                    "0",
                    "--log-level",
                    "off",
                    "--out",
                    dest,
                ]
                for layer in "AGO":
                    cmd += ["--" + layer, d / "k.tsv"]
                if g0 is not None:
                    cmd += ["--init-vu", str(g0), "--init-ve", str(e0)]
                r = subprocess.run(cmd, capture_output=True, text=True)
                row = dict(
                    case=case,
                    method=method,
                    g0=g0,
                    e0=e0,
                    status=r.stderr.strip(),
                    g=None,
                    e=None,
                    ratio=None,
                    objective=None,
                )
                if not r.returncode:
                    with (dest / "model.tsv").open() as f:
                        m = next(csv.DictReader(f, delimiter="\t"))
                    row.update(
                        status=m["status"],
                        g=m["Vu"],
                        e=m["Ve"],
                        ratio=m["lambda"],
                        objective=m["profile_objective"],
                    )
                elif "iteration-limit" not in r.stderr:
                    raise RuntimeError(r.stderr)
                values.append(row)
    with (d / "summary.tsv").open("w") as f:
        w = csv.DictWriter(f, values[0], delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(values)


if __name__ == "__main__":
    main()
