"""Usage: python reproduce_reml.py /path/to/fsblup [output_parent]. Requires numpy/scipy."""
import csv
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize_scalar

exe = Path(sys.argv[1]).resolve()
root = Path(tempfile.mkdtemp(prefix="fsblup-reml-", dir=sys.argv[2] if len(sys.argv) > 2 else None))
print("raw files:", root)
diag = np.array([.002, .01, .1, 1., 5., 10., 20., 100.])
y = np.array([.077869525304753107, 1.1183912873377264, 2.1572583184874374,
              .69263970846924439, .99807996539091282, -6.0332965102701319,
              .52182914486587306, .46722856041418481])
n = len(y)
ids = np.arange(1, n + 1)
# All three input kernels are identical. The default final fusion adds 1e-6 I.
np.savetxt(root / "K.tsv", np.column_stack((ids, np.diag(diag - 1e-6))),
           header="id\t" + "\t".join(map(str, ids)), comments="", delimiter="\t",
           fmt=["%d"] + ["%.17g"] * n)
np.savetxt(root / "y.tsv", np.column_stack((ids, y)), header="id\ty", comments="",
           delimiter="\t", fmt=["%d", "%.17g"])

# Independent orthonormal contrast profile. Each interval bracket gets refined,
# and both feasible faces are compared on the same restricted likelihood scale.
Q = null_space(np.ones((1, n)))
k, U = np.linalg.eigh(Q.T @ (diag[:, None] * Q))
energy = (U.T @ Q.T @ y) ** 2
df = n - 1
def candidate(g, e):
    v = g * k + e
    return -.5 * (np.log(v).sum() + (energy / v).sum()), g, e
def profile(logratio):
    ratio = np.exp(logratio)
    g = (energy / (k + ratio)).mean()
    return candidate(g, g * ratio)
grid = np.linspace(-24, 24, 2049)
v = np.array([profile(t)[0] for t in grid])
candidates = [candidate(0., energy.mean()), candidate((energy / k).mean(), 0.)]
for j in np.flatnonzero((v[1:-1] > v[:-2]) & (v[1:-1] > v[2:])) + 1:
    z = minimize_scalar(lambda t: -profile(t)[0], bounds=grid[[j-1, j+1]], method="bounded",
                        options={"xatol": 1e-13})
    candidates.append(profile(z.x))
ell, vu, ve = max(candidates)
print("independent oracle ell=%.12g Vu=%.12g Ve=%.12g" % (ell, vu, ve))

def row(path):
    with path.open() as f:
        return next(csv.DictReader(f, delimiter="\t"))

runs = [("direct-profile", ["--solver", "direct", "--varcomp", "profile-reml"])]
for method in ("ai-reml", "em-reml"):
    for trace in ("exact", "hutchinson"):
        runs.append((method + "-" + trace, ["--varcomp", method, "--trace", trace]))
    runs.append((method + "-hutchinson-interior-start", ["--varcomp", method, "--trace",
                 "hutchinson", "--init-vu", "1", "--init-ve", ".8"]))
    runs.append((method + "-near-boundary", ["--varcomp", method, "--trace", "exact",
                 "--init-vu", ".00001", "--init-ve", "6.325402076"]))
cmd0 = [str(exe), "fit", *[v for c in "AGO" for v in ("--" + c, str(root / "K.tsv"))],
        "--pheno", str(root / "y.tsv"), "--id", "id", "--trait", "y",
        "--alpha", "0", "--beta", "0"]
print("case\texit\tstatus\tVu\tVe\tlogLik\tlogdet_se\tmax_abs_u")
for name, options in runs:
    out = root / name
    p = subprocess.run([*cmd0, *options, "--out", str(out)], text=True, capture_output=True)
    if p.returncode:
        print(name, p.returncode, p.stderr.strip().splitlines()[-1], sep="\t")
        continue
    m = row(out / "model.tsv")
    with (out / "predictions.tsv").open() as f:
        u = [abs(float(r["u"])) for r in csv.DictReader(f, delimiter="\t")]
    print(name, p.returncode, m["status"], m["Vu"], m["Ve"], m["contrast_loglik"],
          row(out / "pcg_summary.tsv").get("logdet_se"), max(u), sep="\t")
