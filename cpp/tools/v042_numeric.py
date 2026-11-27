"""Compare the two binaries with the review fixture and an independent error-contrast profile."""

import argparse, csv, subprocess
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize_scalar


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("old", "new", "out"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    a.out.mkdir()
    d = Path(__file__).resolve().parents[1] / "tests/face"
    yrows = read(d / "y.tsv")
    ids = [r["id"] for r in yrows]
    y = np.array([float(r["y"]) for r in yrows])
    k = (
        np.array([[float(r[i]) for i in ids] for r in read(d / "K.tsv")])
        + np.eye(len(ids)) * 1e-6
    )
    q = null_space(np.ones((1, len(ids))))
    eig, basis = np.linalg.eigh(q.T @ k @ q)
    z = (basis.T @ q.T @ y) ** 2

    def profile(logratio):
        ratio = np.exp(logratio)
        denom = eig + ratio
        g = np.mean(z / denom)
        return -0.5 * (len(z) * (1 + np.log(g)) + np.log(denom).sum()), g, ratio * g

    grid = np.linspace(-24, 24, 2049)
    values = np.array([profile(t)[0] for t in grid])
    face0 = z.mean()
    face1 = np.mean(z / eig)
    candidates = [
        (-0.5 * (len(z) * (1 + np.log(face0))), 0, face0),
        (-0.5 * np.sum(np.log(face1 * eig) + z / (face1 * eig)), face1, 0),
    ]
    for j in (
        np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])) + 1
    ):
        peak = minimize_scalar(
            lambda t: -profile(t)[0],
            bounds=grid[[j - 1, j + 1]],
            method="bounded",
            options={"xatol": 1e-13},
        )
        candidates.append(profile(peak.x))
    lik, vu, ve = max(candidates)
    h = np.linalg.solve(
        vu * k + ve * np.eye(len(y)), np.column_stack((y, np.ones(len(y))))
    )
    b = h[:, 0].sum() / h[:, 1].sum()
    u = vu * k @ (h[:, 0] - b * h[:, 1])
    records = []
    for version, exe in [("old", a.old), ("new", a.new)]:
        for method in ("ai-reml", "em-reml"):
            for name, trace, opts in [
                ("default", "hutchinson", []),
                (
                    "near-start",
                    "exact",
                    ["--init-vu", "1e-5", "--init-ve", "6.325402076"],
                ),
                ("interior-start", "hutchinson", ["--init-vu", "1", "--init-ve", ".8"]),
                (
                    "controlled",
                    "hutchinson",
                    [
                        "--slq-depth",
                        "8",
                        "--probes",
                        "32",
                        "--max-probes",
                        "32",
                        "--probe-seed",
                        "17",
                    ],
                ),
                (
                    "lanczos",
                    "hutchinson",
                    [
                        "--probe-solver",
                        "lanczos",
                        "--slq-depth",
                        "8",
                        "--probe-seed",
                        "19",
                    ],
                ),
            ]:
                out = a.out / (version + method + name)
                args = [
                    exe,
                    "fit",
                    *[v for c in "AGO" for v in ("--" + c, d / "K.tsv")],
                    "--pheno",
                    d / "y.tsv",
                    "--trait",
                    "y",
                    "--id",
                    "id",
                    "--alpha",
                    0,
                    "--beta",
                    0,
                    "--varcomp",
                    method,
                    "--trace",
                    trace,
                    *opts,
                    "--out",
                    out,
                ]
                res = subprocess.run(
                    list(map(str, args)), capture_output=True, text=True
                )
                (a.out / (out.name + ".log")).write_text(res.stderr)
                audit = read(out / "pcg_summary.tsv")[0]
                steps = read(out / "reml_trace.tsv")
                challenges = [s for s in steps if s["status"] == "face-challenger"]
                row = dict(
                    version=version,
                    method=method,
                    case=name,
                    exit=res.returncode,
                    status=audit["status"],
                    rhs=audit["rhs"],
                    matvec=audit["matvec"],
                    probes=audit["probes"],
                    challengers=len(challenges),
                    challenger_matvec=sum(int(r["face_matvec"]) for r in challenges),
                    dVu=None,
                    dVe=None,
                    dLik=None,
                    max_du=None,
                    max_dyhat=None,
                )
                if not res.returncode:
                    model = read(out / "model.tsv")[0]
                    pred = read(out / "predictions.tsv")
                    assert [r["id"] for r in pred] == ids
                    row.update(
                        dVu=abs(float(model["Vu"]) - vu),
                        dVe=abs(float(model["Ve"]) - ve),
                        dLik=abs(float(model["contrast_loglik"]) - lik),
                        max_du=max(abs(float(r["u"]) - v) for r, v in zip(pred, u)),
                        max_dyhat=max(
                            abs(float(r["yhat"]) - v - b) for r, v in zip(pred, u)
                        ),
                    )
                records.append(row)
                if version == "new":
                    assert (
                        not res.returncode
                        and row["dLik"] < 1e-7
                        and max(row[k] for k in ("dVu", "dVe", "max_du", "max_dyhat"))
                        < 1e-5
                    ), row
    with (a.out / "comparison.tsv").open("w") as f:
        w = csv.DictWriter(f, records[0], delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(records)


if __name__ == "__main__":
    main()
