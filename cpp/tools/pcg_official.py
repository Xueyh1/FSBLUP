"""Official mode C 178/20 verification, with a separate fixed-ratio GLS oracle."""

import os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

exe, official, ref, out = map(Path, sys.argv[1:])
out.mkdir(exist_ok=False, parents=True)
ids = (ref / "ids.txt").read_text().splitlines()
K = np.fromfile(ref / "nearPD.bin", dtype="<f8").reshape(
    (len(ids), len(ids)), order="F"
)
y = pd.read_csv(ref / "masked.tsv", sep="\t").set_index("id").age
idx = [ids.index(i) for i in y.index]
k = K[np.ix_(idx, idx)]
train = y.notna().to_numpy()
ratio = 4.39429236354486
v = k[train][:, train] + ratio * np.eye(train.sum())
h, w = np.linalg.solve(v, np.stack([y[train], np.ones(train.sum())], axis=1)).T
beta = h.sum() / w.sum()
u = k[:, train] @ (h - w * beta)
r = pd.read_csv(ref / "r_predictions.tsv", sep="\t").set_index("id")
rv = pd.read_csv(ref / "r_model.tsv", sep="\t").iloc[0]
results = []
for solver, method in [
    ("pcg", "fixed"),
    ("direct", "fixed"),
    ("pcg", "ai-reml"),
    ("pcg", "em-reml"),
    ("direct", "profile-reml"),
]:
    d = out / (solver + "-" + method)
    args = [
        exe.resolve(),
        "fit",
        "--pheno",
        ref / "masked.tsv",
        "--trait",
        "age",
        "--id",
        "id",
        "--solver",
        solver,
        "--varcomp",
        method,
        "--pd-policy",
        "nearPD",
        "--alpha",
        ".2",
        "--beta",
        ".5",
        "--threads",
        "1",
        "--log-level",
        "off",
        "--out",
        d,
    ]
    for name in "AGO":
        args += ["--" + name, official / f"test_mat{name}.csv"]
    if method == "fixed":
        args += ["--lambda", str(ratio)]
    if method == "ai-reml":
        args += ["--write-k"]
    subprocess.run(args, check=True, env=dict(os.environ, OPENBLAS_NUM_THREADS="1"))
    m = pd.read_csv(d / "model.tsv", sep="\t").iloc[0]
    pred = pd.read_csv(d / "predictions.tsv", sep="\t").set_index("id")
    assert list(pred.index) == list(y.index)
    if method == "fixed":
        eu = np.abs(u - pred.u)
        ey = np.abs(u + beta - pred.yhat)
    else:
        eu = np.abs(r.u - pred.u)
        ey = np.abs(r.yhat - pred.yhat)
    info = pd.read_csv(d / "pcg_summary.tsv", sep="\t").iloc[0]
    results.append(
        dict(
            method=method,
            solver=solver,
            Vu=m.Vu,
            Ve=m.Ve,
            ratio=m["lambda"],
            max_du=eu.max(),
            max_dyhat=ey.max(),
            candidate_du=eu[~train].max(),
            candidate_dyhat=ey[~train].max(),
            rhs=info.rhs,
            matvec=info.matvec,
            true_residual_over_tol=info.max_true_residual_over_tolerance,
        )
    )
    assert max(eu.max(), ey.max()) < (1e-7 if method == "fixed" else 1e-4)
    assert abs(m["lambda"] - rv["lambda"]) < 1e-4
    assert max(abs(m.Vu / rv.Vu - 1), abs(m.Ve / rv.Ve - 1)) < 1e-4
    if method == "ai-reml":
        f = (
            pd.read_csv(
                d / "K.tsv", sep="\t", index_col=0, float_precision="round_trip"
            )
            .loc[ids, ids]
            .to_numpy()
        )
        assert (
            np.max(np.abs(f - K)) < 1e-11
            and np.linalg.norm(f - K) / np.linalg.norm(K) < 1e-12
        )
pd.DataFrame(results).to_csv(out / "comparison.tsv", sep="\t", index=False)
