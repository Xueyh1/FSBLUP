"""Check identical spans, extrapolation and back-transformation of coefficients."""

import csv, math, subprocess, sys, tempfile
from pathlib import Path


def rows(p):
    with p.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    vals = [2, 2, 2, 1, 2, 3, 4, 5]
    q = [[(-1) ** bin(i & j).count("1") for j in range(8)] for i in range(8)]
    k = [[sum(v * a * b for v, a, b in zip(vals, r, s)) / 8 for s in q] for r in q]
    for i, r in enumerate(k):
        r.append(0.1 * r[0])
        r[i] -= 1e-6
    k.append([r[-1] for r in k] + [2])
    with (root / "K").open("w") as f:
        print("id", *range(1, 10), sep="\t", file=f)
        for i, r in enumerate(k, 1):
            print(i, *r, sep="\t", file=f)
    for solver, method in [
        ("direct", "profile-reml"),
        ("direct", "ai-reml"),
        ("pcg", "ai-reml"),
        ("pcg", "fixed"),
    ]:
        for near in (False, True):
            with (root / "y").open("w") as f:
                print("id", "x", "w", "y", sep="\t", file=f)
                for i in range(9):
                    x, z = (q[i][1], q[i][2]) if i < 8 else (2, -3)
                    y = (
                        3
                        + 4 * x
                        + 2 * z
                        + sum(
                            c * math.sqrt(2 * v + 1) for c, v in zip(q[i][3:], vals[3:])
                        )
                        / math.sqrt(8)
                        if i < 8
                        else "NA"
                    )
                    print(i + 1, x, x + 1e-7 * z if near else z, y, sep="\t", file=f)
            dest = root / (solver + method + str(near))
            cmd = [
                sys.argv[1],
                "fit",
                *[v for name in "AGO" for v in ("--" + name, root / "K")],
                "--alpha",
                "0",
                "--beta",
                "0",
                "--pheno",
                root / "y",
                "--trait",
                "y",
                "--id",
                "id",
                "--covar",
                "x,w",
                "--solver",
                solver,
                "--varcomp",
                method,
                "--out",
                dest,
            ]
            if method == "fixed":
                cmd += ["--lambda", ".5"]
            p = subprocess.run(cmd, capture_output=True, text=True)
            assert not p.returncode, p.stderr
            m = rows(dest / "model.tsv")[0]
            assert abs(float(m["Vu"]) - 2) < 1e-5 and abs(float(m["Ve"]) - 1) < 1e-5
            b = [float(r["estimate"]) for r in rows(dest / "fixed.tsv")]
            assert (
                abs(b[0] - 3) < 1e-6 and abs((b[1] + b[2] if near else b[1]) - 4) < 1e-6
            )
            assert abs((1e-7 if near else 1) * b[2] - 2) < 1e-6
            pred = rows(dest / "predictions.tsv")
            if near:
                assert (
                    max(
                        abs(float(x[c]) - float(y[c]))
                        for x, y in zip(base, pred)
                        for c in ("u", "yhat")
                    )
                    < 1e-6
                )
            else:
                base = pred
    # A level represented in the validation set only is absent from its
    # training design, even though the full population matrix is full rank.
    with (root / "factor").open("w") as f, (root / "folds").open("w") as folds:
        print("id", "y", "level", sep="\t", file=f)
        print("repeat", "fold", "id", sep="\t", file=folds)
        for i in range(9):
            print(
                i + 1,
                i * i + 1 if i < 8 else "NA",
                "rare" if i == 0 else "common",
                sep="\t",
                file=f,
            )
            if i < 8:
                print(1, 1 + i % 2, i + 1, sep="\t", file=folds)
    dest = root / "missing-level"
    p = subprocess.run(
        [
            sys.argv[1],
            "fit",
            *[v for c in "AGO" for v in ("--" + c, root / "K")],
            "--factor",
            "level",
            "--pheno",
            root / "factor",
            "--id",
            "id",
            "--trait",
            "y",
            "--folds-file",
            root / "folds",
            "--strict-simplex",
            "--metric",
            "rmse",
            "--out",
            dest,
        ],
        capture_output=True,
        text=True,
    )
    assert p.returncode and all(
        "rank deficient" in r["reason"] for r in rows(dest / "search.tsv")
    )
