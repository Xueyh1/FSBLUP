"""Independent diagonal GLS predictions and the orthonormal-profile reference."""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

exe = Path(sys.argv[1]).resolve()
data = Path(__file__).parent / "face"


def rows(p):
    with p.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


with tempfile.TemporaryDirectory() as d:
    weights = [0.002, 0.01, 0.1, 1, 5, 10, 20, 100]
    truth = rows(data / "y.tsv")
    y = [float(r["y"]) for r in truth]
    g, e = 1.0110784086, 0.8270563024
    vi = [1 / (g * k + e) for k in weights]
    b = sum(a * v for a, v in zip(y, vi)) / sum(vi)
    for method in ["ai-reml", "em-reml"]:
        for i, (trace, opts) in enumerate(
            [
                ("exact", ["--init-vu", "1e-5", "--init-ve", "6.325402076"]),
                ("hutchinson", []),
                ("hutchinson", ["--init-vu", "1e-5", "--init-ve", "6.325402076"]),
                (
                    "hutchinson",
                    [
                        "--probe-solver",
                        "lanczos",
                        "--slq-depth",
                        "8",
                        "--probe-seed",
                        "17",
                        "--probes",
                        "32",
                        "--max-probes",
                        "32",
                    ],
                ),
            ]
        ):
            out = Path(d) / (method + str(i))
            args = [
                exe,
                "fit",
                *[v for k in "AGO" for v in ("--" + k, data / "K.tsv")],
                "--pheno",
                data / "y.tsv",
                "--id",
                "id",
                "--trait",
                "y",
                "--alpha",
                "0",
                "--beta",
                "0",
                "--varcomp",
                method,
                "--trace",
                trace,
                *opts,
                "--out",
                out,
            ]
            p = subprocess.run(args, capture_output=True, text=True)
            assert not p.returncode, p.stderr
            m = rows(out / "model.tsv")[0]
            assert abs(float(m["contrast_loglik"]) + 9.19651386833) < 1e-7, m
            assert abs(float(m["Vu"]) - g) < 1e-5 and abs(float(m["Ve"]) - e) < 1e-5, m
            for j, r in enumerate(rows(out / "predictions.tsv")):
                u = g * weights[j] * vi[j] * (y[j] - b)
                assert r["id"] == truth[j]["id"] and abs(float(r["u"]) - u) < 1e-5
                assert abs(float(r["yhat"]) - b - u) < 1e-5
            if i == 1:
                assert any(
                    r["status"] == "interior-restart"
                    for r in rows(out / "reml_trace.tsv")
                )

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    for case in ("genetic-zero", "residual-floor", "unidentifiable"):
        k = data / "K.tsv"
        if case == "unidentifiable":
            k = d / "identity.tsv"
            with k.open("w") as f:
                print("id", *range(1, 9), sep="\t", file=f)
                for i in range(8):
                    print(
                        i + 1,
                        *[(1 - 1e-6) if i == j else 0 for j in range(8)],
                        sep="\t",
                        file=f,
                    )
        with (d / "y").open("w") as f:
            print("id", "y", sep="\t", file=f)
            for i, z in enumerate(
                (
                    [1, -1, 0, 0, 0, 0, 0, 0]
                    if case == "genetic-zero"
                    else [0, 0, 0, 0, 0, 0, 1, -1]
                )
            ):
                print(i + 1, z, sep="\t", file=f)
        for method in ("ai-reml", "em-reml"):
            out = d / (method + case)
            args = [
                exe,
                "fit",
                *[v for key in "AGO" for v in ("--" + key, k)],
                "--pheno",
                d / "y",
                "--id",
                "id",
                "--trait",
                "y",
                "--alpha",
                "0",
                "--beta",
                "0",
                "--varcomp",
                method,
                "--trace",
                "hutchinson",
                "--out",
                out,
            ]
            if case == "residual-floor":
                args += ["--init-vu", "1", "--init-ve", "1e-12"]
            p = subprocess.run(args, capture_output=True, text=True)
            assert not p.returncode, p.stderr
            assert case in rows(out / "model.tsv")[0]["status"]
