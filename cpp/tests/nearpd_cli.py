"""Check the positive-definiteness policy through fuse, fixed fit, and tuning."""
import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile

JITTER = 1e-6
IDS = [f"{i:03d}" for i in range(1, 9)]


def write(path, header, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def records(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def identity(n):
    return [[float(i == j) for j in range(n)] for i in range(n)]


def save_kernel(path, ids, matrix):
    write(path, ["ID", *ids], ([i, *row] for i, row in zip(ids, matrix)))
    return path


def kernel(path, ids):
    rows = records(path)
    assert [row["ID"] for row in rows] == ids
    return [[float(row[i]) for i in ids] for row in rows]


def near(a, b, tolerance=1e-10):
    assert abs(a - b) <= tolerance, (a, b, tolerance)


def check_matrix(actual, expected):
    for row, target in zip(actual, expected):
        for a, b in zip(row, target):
            near(a, b)


def cholesky_solve(matrix, rhs):
    n = len(rhs)
    lower = [[0.0] * n for _ in rhs]
    for i in range(n):
        for j in range(i + 1):
            value = matrix[i][j] - sum(lower[i][k]*lower[j][k] for k in range(j))
            lower[i][j] = math.sqrt(value) if i == j else value / lower[j][j]
    z = [0.0] * n
    for i in range(n):
        z[i] = (rhs[i] - sum(lower[i][j]*z[j] for j in range(i))) / lower[i][i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (z[i] - sum(lower[j][i]*x[j] for j in range(i+1, n))) / lower[i][i]
    return x


def check_fit(directory, matrix, y):
    train = [i for i, yi in enumerate(y) if yi is not None]
    lam = float(records(directory / "model.tsv")[0]["lambda"])
    v = [[matrix[i][j] + (lam if i == j else 0.) for j in train] for i in train]
    t = cholesky_solve(v, [y[i] for i in train])
    w = cholesky_solve(v, [1.] * len(train))
    intercept = sum(t) / sum(w)
    q = [a-intercept*b for a, b in zip(t, w)]
    predictions = records(directory / "predictions.tsv")
    assert [r["id"] for r in predictions] == IDS
    for i, row in enumerate(predictions):
        assert row["role"] == ("candidate" if y[i] is None else "reference")
        u = sum(matrix[i][j]*v for j, v in zip(train, q))
        near(float(row["u"]), u, 2e-7)
        near(float(row["yhat"]), intercept + u, 2e-7)


def run(binary, root):
    def invoke(command, name, args, error=None):
        target = root / name
        proc = subprocess.run([str(binary), command, "--out", str(target),
                               *map(str, args)], text=True, capture_output=True)
        if error is None:
            assert proc.returncode == 0, (name, proc.stderr)
        else:
            assert proc.returncode != 0 and error.lower() in proc.stderr.lower(), (name, proc.stderr)
        return target

    def layers(a, g=None, o=None):
        return ["--A", a, "--G", g or a, "--O", o or a]

    fixed = ["--alpha", "0", "--beta", "0"]
    repair = ["--pd-policy", "nearPD"]
    raw = identity(len(IDS))
    raw[0][1] = raw[1][0] = 2.
    raw[6][7] = raw[7][6] = .4
    indefinite = save_kernel(root / "indefinite.tsv", IDS, raw)
    inp = layers(indefinite)
    for name, policy in (("default", []), ("strict", ["--pd-policy", "strict"])):
        invoke("fuse", name, inp + fixed + policy, "indefinite")
    invoke("fuse", "unknown", inp + fixed + ["--pd-policy", "typo"], "policy")
    out = invoke("fuse", "repaired", inp + fixed + repair)

    # The first block has shifted eigenvalues 3+epsilon and -1+epsilon.
    # Matrix::nearPD zeroes the latter, applies posd.tol, then restores
    # the diagonal of the PSD projection by a diagonal congruence.
    largest = 3 + JITTER
    floor = largest * 1e-8
    expected = [row.copy() for row in raw]
    for i in range(len(IDS)):
        expected[i][i] += JITTER
    expected[0][0] = expected[1][1] = largest / 2
    expected[0][1] = expected[1][0] = (largest/2) * (largest-floor)/(largest+floor)
    check_matrix(kernel(out / "K_raw.tsv", IDS), raw)
    check_matrix(kernel(out / "K.tsv", IDS), expected)
    audit = records(out / "pd.tsv")
    assert len(audit) == 1 and audit[0]["block"] == "K"
    row = audit[0]
    assert row["policy"].lower() == "nearpd" and row["action"] == "nearPD"
    assert int(row["iterations"]) >= 1 and row["converged"].lower() in ("1", "true")
    near(float(row["jitter"]), JITTER)
    near(float(row["min_eigen_raw"]), -1., 1e-8)
    near(float(row["min_eigen_used"]), expected[0][0]-expected[0][1], 1e-10)
    assert .99 < float(row["normF"]) < 1.01 and float(row["relative_normF"]) > 0

    # Exact PSD, a genuinely SPD matrix and a slight indefinite matrix for
    # which the R diagonal shift suffices require no spectral projection.
    for label, off in (("spd", .4), ("psd", 1.), ("rescued_by_shift", 1.0000005)):
        m = identity(2)
        m[0][1] = m[1][0] = off
        path = save_kernel(root / (label+".tsv"), IDS[:2], m)
        dest = invoke("fuse", label, layers(path) + fixed + repair)
        check_matrix(kernel(dest / "K.tsv", IDS[:2]), [[1+JITTER, off], [off, 1+JITTER]])
        row = records(dest / "pd.tsv")[0]
        assert row["action"] == "jitter", (label, row)
        if label == "rescued_by_shift":
            invoke("fuse", label+"_strict", layers(path) + fixed, "indefinite")

    y = [-3., -.2, 1.7, 4.5, -1.6, 2.9, 3.5, None]
    phen = root / "phen.tsv"
    write(phen, ["ID", "y"], zip(IDS, [v if v is not None else "NA" for v in y]))
    model = ["--pheno", phen, "--id", "ID", "--trait", "y"]
    out = invoke("fit", "fit_fixed", inp + fixed + repair + model + ["--write-k"])
    check_fit(out, expected, y)
    check_matrix(kernel(out / "K.tsv", IDS), expected)

    folds = root / "folds.tsv"
    write(folds, ["repeat", "fold", "id"], ((1, 1+i%2, id_) for i,id_ in enumerate(IDS[:-1])))
    auto = model + ["--folds-file", folds, "--metric", "rmse", "--strict-simplex"]
    # Nested C: all source kernels are SPD. On one pedigree-only and one
    # genotyped-only animal its conditional formulas produce [[1,3.6],[3.6,4]]
    # with negative determinant. This block persists for every weight.
    a = identity(len(IDS))
    a[0][3] = a[3][0] = .9
    g = identity(5)
    g[0][0] = 4.
    nested = layers(save_kernel(root / "C_A.tsv", IDS, a),
                    save_kernel(root / "C_G.tsv", IDS[3:], g),
                    save_kernel(root / "C_O.tsv", IDS[5:], identity(3)))
    invoke("fit", "strict_search", nested + auto, "no eligible grid point")
    out = invoke("fit", "repair_search", nested + auto + repair + ["--write-k"])
    row = records(out / "fusion.tsv")[0]
    assert row["mode"] == "C" and row["pd_policy"].lower() == "nearpd"
    assembled = kernel(out / "K_raw.tsv", IDS)
    near(assembled[0][0], 1.)
    near(assembled[3][3], 4.)
    near(assembled[0][3], 3.6)
    trace = [r for r in records(out / "search.tsv") if r["stage"] == "grid"]
    assert len(trace) == 15
    for row in trace:
        assert row["valid"] == "1" and math.isfinite(float(row["mean"])), row
        assert row["pd_policy"].lower() == "nearpd"
        assert int(row["nearpd_blocks"]) == 1 and float(row["max_relative_normF"]) > 0
        # Matrix 1.7-0 nearPD on reference K + jitter, then direct GLS in R
        # minimizing (n-1)log(RSS/(n-1)) + log|V| + log|(1'V^-1 1)|.
        # The two fold minima give lambda = 5.071765 and 1e-9 respectively.
        near(float(row["r1f1"]), 2.81528092996, 1e-6)
        near(float(row["r1f2"]), 5.95040614548, 1e-6)
    near(float(records(out / "tuning.tsv")[0]["selection_score"]), float(trace[0]["mean"]), 1e-7)
    check_fit(out, kernel(out / "K.tsv", IDS), y)

    # B applies the R policy to the conditioning Gss block; C's pedigree
    # regression is a separate model precondition.
    m = identity(len(IDS))
    ped = save_kernel(root / "A.tsv", IDS, m)
    o = save_kernel(root / "O.tsv", IDS[:2], [[1., .1], [.1, 1.]])
    weights = ["--alpha", 0, "--beta", .5]
    args = layers(ped, indefinite, o) + weights
    invoke("fuse", "B_strict", args, "indefinite")
    out = invoke("fuse", "B_repair", args + repair)
    blocks = {r["block"]:r["action"] for r in records(out / "pd.tsv")}
    assert blocks == {"Gss":"nearPD", "K":"jitter"}, blocks
    m[0][1] = m[1][0] = 1.
    singular = save_kernel(root / "singular_A.tsv", IDS, m)
    g = save_kernel(root / "G.tsv", IDS[:4], identity(4))
    invoke("fuse", "C_singular", layers(singular, g, o) + weights + repair, "conditioning block")
    print("nearPD CLI: analytic matrix, jitter, fixed fit, 15 tuning grid points, B and C blocks passed")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as work:
        run(Path(sys.argv[1]).resolve(), Path(work))
