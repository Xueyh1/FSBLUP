"""Predeclared interior Monte Carlo checks; seeds fix the input before fitting."""

import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("binary", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument(
    "probe_solver", nargs="?", choices=("pcg", "lanczos"), default="lanczos"
)
parser.add_argument(
    "--family", choices=("heterogeneous", "genotypes"), default="heterogeneous"
)
parser.add_argument("--max-probes", type=int, default=4096)
parser.add_argument("--trace-rank", type=int, default=0)
args = parser.parse_args()
binary = args.binary.resolve()
out = args.output
out.mkdir(parents=True, exist_ok=False)
records = []
for n in (32, 96):
    for seed in (7, 31, 73):
        rng = np.random.default_rng(seed)
        N = n + 8
        ids = [f"i{i:04}" for i in range(N)]
        K = np.diag(np.linspace(0.15, 4.5, N)) + 0.12 * 0.6 ** np.abs(
            np.subtract.outer(np.arange(N), np.arange(N))
        )
        if args.family == "genotypes":
            Z = rng.binomial(2, 0.4, size=(N, 2 * n)).astype(float)
            Z -= Z.mean(axis=0)
            K = Z @ Z.T / (2 * n * 0.48) + 0.2 * np.eye(N)
        # Random contrast signs with prescribed squared amplitudes yield an interior
        # score zero at g=2,e=1, also varying the fitted effects across datasets.
        Q = np.linalg.qr(np.ones((n, 1)), mode="complete")[0][:, 1:]
        k, U = np.linalg.eigh(Q.T @ (K[:n, :n] + np.eye(n) * 1e-6) @ Q)
        y = Q @ U @ (rng.choice([-1.0, 1.0], n - 1) * np.sqrt(2 * k + 1))
        target = out / f"n{n}-s{seed}"
        target.mkdir()
        pd.DataFrame(K, index=ids, columns=ids).to_csv(
            target / "K", sep="\t", index_label="ID"
        )
        pd.DataFrame({"ID": ids, "y": list(y) + [np.nan] * 8}).to_csv(
            target / "y", sep="\t", index=False, na_rep="NA"
        )
        cmd = [
            str(binary),
            "fit",
            "--pheno",
            str(target / "y"),
            "--trait",
            "y",
            "--alpha",
            "0",
            "--beta",
            "0",
            "--log-level",
            "off",
        ]
        for layer in "AGO":
            cmd += ["--" + layer, str(target / "K")]
        exact = target / "exact"
        subprocess.run(cmd + ["--out", str(exact)], check=True)

        def outputs(d):
            return pd.read_csv(d / "model.tsv", sep="\t").iloc[0], pd.read_csv(
                d / "predictions.tsv", sep="\t"
            ).u.to_numpy()[-8:]

        ref, u = outputs(exact)
        for estimator in ["ai-reml", "em-reml"]:
            for probe_seed in [3, 19, 43]:
                d = target / f"{estimator}-{probe_seed}"
                run = subprocess.run(
                    cmd
                    + [
                        "--varcomp",
                        estimator,
                        "--trace",
                        "hutchinson",
                        "--probe-solver",
                        args.probe_solver,
                        "--trace-rank",
                        str(args.trace_rank),
                        "--probes",
                        "8",
                        "--max-probes",
                        str(args.max_probes),
                        "--slq-depth",
                        "24",
                        "--probe-seed",
                        str(probe_seed),
                        "--mc-tol",
                        ".02",
                        "--out",
                        str(d),
                    ],
                    capture_output=True,
                    text=True,
                )
                row = dict(
                    family=args.family,
                    n=n,
                    seed=seed,
                    probe_seed=probe_seed,
                    method=estimator,
                    reason=run.stderr.strip(),
                    log_ratio_error=np.nan,
                    relative_rmse=np.nan,
                    probes=0,
                )
                if not run.returncode:
                    m, uh = outputs(d)
                    row.update(
                        log_ratio_error=abs(np.log(m["lambda"] / ref["lambda"])),
                        relative_rmse=np.sqrt(np.mean((uh - u) ** 2)) / np.std(u),
                        probes=pd.read_csv(d / "pcg_summary.tsv", sep="\t").probes.iloc[
                            0
                        ],
                    )
                records.append(row)
                pd.DataFrame(records).to_csv(out / "samples.tsv", sep="\t", index=False)
r = pd.DataFrame(records)
accepted = True
for name, g in r.groupby("method"):
    q = g[["log_ratio_error", "relative_rmse"]].quantile(0.95)
    print(
        name,
        "failures",
        g.log_ratio_error.isna().sum(),
        "of",
        len(g),
        "p95",
        q.to_dict(),
    )
    accepted &= (
        not g.log_ratio_error.isna().any() and q.iloc[0] <= 0.05 and q.iloc[1] <= 0.01
    )

assert accepted, "Monte Carlo acceptance gate failed; inspect samples.tsv"
