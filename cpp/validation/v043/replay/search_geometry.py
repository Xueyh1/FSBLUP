"""Exercise an attained, independently scored simplex edge optimum through fsblup fit."""
import argparse
import csv
import os
import subprocess
from pathlib import Path

import numpy as np


def table(path, header, data):
    with path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(data)


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def fold_predictions(A, G, O, y, folds, alpha, beta):
    K = (1 - alpha - beta) * A + beta * G + alpha * O + 1.e-6 * np.eye(len(y))
    u, fit = np.empty_like(y), np.empty_like(y)
    for fold in sorted(set(folds)):
        t, v = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        h = K[np.ix_(t, t)] + np.eye(len(t))
        vi = np.linalg.solve(h, np.column_stack((y[t], np.ones(len(t)))))
        intercept = vi[:, 0].sum() / vi[:, 1].sum()
        u[v] = K[np.ix_(v, t)] @ (vi[:, 0] - vi[:, 1] * intercept)
        fit[v] = u[v] + intercept
    return u, fit


def fixed_weight_cv(exe, d, ids, y, folds, target, solver, label, alpha, beta):
    errors = []
    for fold in sorted(set(folds)):
        valid = folds == fold
        phenotype = d / f"fixed-{label}-{fold}.tsv"
        table(phenotype, ["ID", "Y"], ((key, "NA" if mask else val)
                                      for key, mask, val in zip(ids, valid, y)))
        out = d / f"fixed-{label}-{solver}-{fold}"
        cmd = [exe, "fit", "--A", d/"A.tsv", "--G", d/"G.tsv", "--O", d/"O.tsv",
               "--pheno", phenotype, "--trait", "Y", "--alpha", str(alpha),
               "--beta", str(beta), "--solver", solver, "--varcomp", "fixed",
               "--lambda", "1", "--log-level", "error", "--out", out]
        subprocess.run(cmd, check=True, env=dict(os.environ, OPENBLAS_NUM_THREADS="1"))
        predictions = {row["id"]: float(row["u"]) for row in read(out/"predictions.tsv")
                       if row["role"] == "candidate"}
        observed = np.array([predictions[ids[i]] for i in np.flatnonzero(valid)])
        errors.append(np.sqrt(np.mean((observed - target[valid]) ** 2)))
    return np.mean(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    d = args.out.resolve()
    d.mkdir(parents=True, exist_ok=False)
    n, m = 32, 64
    ids = [f"i{i:03d}" for i in range(n)]
    folds = np.arange(n) % 4
    rng = np.random.default_rng(178)
    x, z = rng.normal(size=(2,n,m))
    A, G, O = np.eye(n), x @ x.T / m, z @ z.T / m
    y = rng.normal(size=n)
    for name, matrix in zip("AGO", [A, G, O]):
        table(d/f"{name}.tsv", ["ID",*ids], ([i,*v] for i,v in zip(ids,matrix)))
    table(d/"folds.tsv", ["repeat","fold","id"], ((1,int(f)+1,i) for f,i in zip(folds,ids)))
    results = []
    for label,alpha,beta in [("edge",.125,0.),("interior",.375,.25)]:
        target,_ = fold_predictions(A,G,O,y,folds,alpha,beta)
        phenotype = d/(label+".tsv")
        table(phenotype,["ID","Y","score"],zip(ids,y,target))
        for solver in ["direct","pcg"]:
            dest = d/(label+"-"+solver)
            args_cli=[args.binary.resolve(),"fit", "--A",d/"A.tsv","--G",d/"G.tsv","--O",d/"O.tsv",
                 "--pheno",phenotype,"--trait","Y","--score-col","score","--metric","rmse","--folds-file",
                 d/"folds.tsv","--strict-simplex","--varcomp","fixed","--lambda","1","--solver",solver,
                 "--log-level","error","--out",dest]
            subprocess.run(args_cli,check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS="1"))
            row = read(dest/"tuning.tsv")[0]
            visited = read(dest/"search.tsv")
            inferred_u,_ = fold_predictions(A,G,O,y,folds,float(row["selected_alpha"]),float(row["selected_beta"]))
            independent = np.mean([np.sqrt(np.mean((inferred_u[folds==f]-target[folds==f])**2)) for f in range(4)])
            hit = any((float(v["alpha"]),float(v["beta"])) == (alpha,beta) for v in visited)
            results.append([label,solver,alpha,beta,row["selected_alpha"],row["selected_beta"],row["selection_score"],
                            independent,len(visited),int(hit),0.,
                            fixed_weight_cv(args.binary.resolve(), d, ids, y, folds, target, solver, label, alpha, beta)])
    table(d/"geometry.tsv",["target","solver","target_alpha","target_beta","selected_alpha","selected_beta",
          "cli_rmse","independent_selected_rmse","visited","target_visited","target_oracle_rmse","target_fixed_cli_cv_rmse"],results)
    print((d/"geometry.tsv").read_text(), end="")


if __name__ == "__main__":
    main()
