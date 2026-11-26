import csv
import math
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

EXE = Path(sys.argv[1]).resolve()
MATH = Path(sys.argv[2]).resolve()
DATA = Path(__file__).parent / "data"


def rows(p):
    with p.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def invoke(root, name, cmd, opts, fail=""):
    out = root / name
    run = subprocess.run(
        [EXE, cmd, *map(str, opts), "--out", out], capture_output=True, text=True
    )
    if fail:
        assert run.returncode and fail.lower() in run.stderr.lower(), (name, run.stderr)
    else:
        assert run.returncode == 0, (name, run.stderr)
    assert not list(root.glob(name + ".tmp.*"))
    return out


def run(root):
    matrices = [v for k in "AGO" for v in ("--" + k, DATA / (k + ".tsv"))]
    weights = ["--alpha", ".2", "--beta", ".5"]
    model = [
        "--pheno",
        DATA / "pheno.tsv",
        "--trait",
        "trait",
        "--covar",
        "age",
        "--factor",
        "herd",
    ]
    fixed = ["--varcomp", "fixed", "--lambda", ".7"]
    for mode in "ABC":
        sources = []
        for layer, file, skip in [
            ("ped", "ped", 0),
            ("snp", "snp", 8 if mode == "C" else 0),
            ("omics", "omics", 12 if mode != "A" else 0),
        ]:
            p = DATA / (file + ".tsv")
            if skip:
                lines = p.read_text().splitlines()
                p = root / (mode + file)
                p.write_text("\n".join(lines[:1] + lines[1 + skip :]) + "\n")
            sources += ["--" + layer, p]
        cache = invoke(
            root, "cache" + mode, "prepare", [*sources, "--snp-format", "2bit"]
        )
        built = invoke(root, "raw" + mode, "build", sources)
        subprocess.run([MATH, cache, built, mode], check=True)
    expected = rows(DATA / "fixed_lambda.tsv")
    for solver in ["pcg", "direct"]:
        for pre in ["jacobi", "identity", "nystrom"] if solver == "pcg" else ["direct"]:
            opts = ["--solver", solver] + (
                [] if solver == "direct" else ["--preconditioner", pre]
            )
            d = invoke(
                root,
                solver + pre,
                "fit",
                ["--cache", root / "cacheA", *weights, *model, *fixed, *opts],
            )
            pred = rows(d / "predictions.tsv")
            assert [p["id"] for p in pred] == [p["id"] for p in expected]
            assert (
                max(
                    abs(float(p[c]) - float(q[c]))
                    for p, q in zip(pred, expected)
                    for c in ("u", "yhat")
                )
                < 1e-9
            )
    for level in ("off", "info", "debug", "trace"):
        logfile = root / (level + ".log")
        d = invoke(
            root,
            "fit" + level,
            "fit",
            [
                *matrices,
                *weights,
                *model,
                "--log-level",
                level,
                "--log-every",
                1,
                "--log-file",
                logfile,
            ],
        )
        if level != "off":
            assert (d / "predictions.tsv").read_bytes() == (
                root / "fitoff" / "predictions.tsv"
            ).read_bytes()
            assert "stage=complete" in logfile.read_text()
        else:
            assert logfile.stat().st_size == 0
    invoke(
        root,
        "limit",
        "fit",
        [*matrices, *weights, *model, *fixed, "--pcg-maxiter", 1],
        "iteration-limit",
    )
    invoke(
        root,
        "emlimit",
        "fit",
        [*matrices, *weights, *model, "--varcomp", "em-reml", "--reml-maxiter", 1],
        "iteration-limit",
    )
    invoke(
        root, "badlog", "fit", [*matrices, *weights, *model, "--log-file", root], "open"
    )
    invoke(
        root,
        "fulllog",
        "fit",
        [*matrices, *weights, *model, "--log-file", "/dev/full"],
        "log write/flush",
    )
    assert not (root / "fulllog").exists()
    invoke(
        root,
        "log-collision",
        "fit",
        [
            *matrices,
            *model,
            *weights,
            "--log-file",
            root / "log-collision" / "model.tsv",
        ],
        "conflicts",
    )
    with (root / "cacheA" / "G.bin").open("ab") as corrupt:
        corrupt.write(b"x")
    invoke(
        root,
        "badcache",
        "fit",
        ["--cache", root / "cacheA", *model, *weights, *fixed],
        "integrity",
    )
    invoke(
        root,
        "badtrace",
        "fit",
        [*matrices, *model, *weights, *fixed, "--trace", "exact"],
        "applies",
    )
    for method in ["ai-reml", "em-reml"]:
        d = invoke(
            root,
            "fit" + method,
            "fit",
            [*matrices, *model, *weights, "--varcomp", method],
        )
        m = rows(d / "model.tsv")[0]
        assert abs(float(m["Vu"]) - 0.604754055) < 1e-5
        tr = rows(d / "reml_trace.tsv")
        assert len(tr) >= 2
    d = invoke(
        root,
        "folds",
        "fit",
        [
            *matrices,
            *model,
            "--folds-file",
            DATA / "folds.tsv",
            "--metric",
            "rmse",
            "--strict-simplex",
            "--parallel",
            "weights",
            "--threads",
            2,
        ],
    )
    assert len(rows(d / "cv_pcg_summary.tsv")) >= 60


with TemporaryDirectory() as tmp:
    run(Path(tmp))
