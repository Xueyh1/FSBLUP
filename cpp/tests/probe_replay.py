"""Offline sign-probe replay and reference/validation separation."""

import csv, math, subprocess, sys, tempfile
from pathlib import Path

d = Path(__file__).parent / "pcg_data"
binary = Path(sys.argv[1]).resolve()


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write(path, values):
    with path.open("w") as f:
        w = csv.DictWriter(
            f, fieldnames=values[0].keys(), delimiter="\t", lineterminator="\n"
        )
        w.writeheader()
        w.writerows(values)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    values = rows(d / "trait.tsv")
    args = [
        str(binary),
        "fit",
        "--pheno",
        str(d / "trait.tsv"),
        "--id",
        "ID",
        "--trait",
        "y",
        "--trace",
        "hutchinson",
        "--slq-depth",
        "24",
        "--log-level",
        "off",
    ]
    for k in "AGO":
        args += ["--" + k, str(d / "kernel.tsv")]

    def run(name, options, err=""):
        dest = root / name
        r = subprocess.run(
            [*args, *options, "--out", dest], capture_output=True, text=True
        )
        if err:
            assert r.returncode and err in r.stderr, (name, r.stderr)
        else:
            assert not r.returncode, (name, r.stderr)
        return dest

    fixed = ["--alpha", "0", "--beta", "0"]
    run(
        "uncertain",
        fixed + ["--probes", "2", "--max-probes", "2", "--mc-tol", "1e-12"],
        "precision insufficient",
    )
    for backend in ("pcg", "lanczos"):
        out = run(
            backend,
            fixed
            + [
                "--probe-solver",
                backend,
                "--probe-seed",
                "19",
                "--trace-rank",
                "4",
                "--probes",
                "16",
                "--max-probes",
                "4096",
                "--mc-tol",
                ".025",
            ],
        )
        m = rows(out / "model.tsv")[0]
        audit = rows(out / "pcg_summary.tsv")[0]
        assert (
            abs(math.log(float(m["lambda"]) / 0.5)) < 0.05
            and float(audit["max_true_residual_over_tolerance"]) <= 1
        )
        assert (
            float(audit["trace_identity_error"]) < 1e-7 and int(audit["probes"]) >= 16
        )
    panel = run(
        "panel",
        fixed
        + [
            "--probe-solver",
            "lanczos",
            "--probe-seed",
            "19",
            "--trace-rank",
            "4",
            "--probes",
            "16",
            "--max-probes",
            "4096",
            "--mc-tol",
            ".025",
            "--slq-reorth",
            "local",
            "--slq-batch",
            "4",
        ],
    )
    assert (
        abs(
            float(rows(panel / "model.tsv")[0]["lambda"])
            - float(rows(root / "lanczos" / "model.tsv")[0]["lambda"])
        )
        < 1e-5
    )
    m0 = rows(root / "pcg" / "model.tsv")[0]
    m1 = rows(root / "lanczos" / "model.tsv")[0]
    assert abs(float(m0["lambda"]) - float(m1["lambda"])) < 1e-5
    pheno_option = args.index(str(d / "trait.tsv"))
    perm = root / "permuted.tsv"
    write(perm, list(reversed(values)))
    args[pheno_option] = str(perm)
    alt = run(
        "permuted",
        fixed
        + [
            "--probe-solver",
            "lanczos",
            "--probe-seed",
            "19",
            "--trace-rank",
            "4",
            "--probes",
            "16",
            "--max-probes",
            "4096",
            "--mc-tol",
            ".025",
        ],
    )
    orig = {r["id"]: float(r["u"]) for r in rows(root / "lanczos" / "predictions.tsv")}
    assert (
        max(abs(float(r["u"]) - orig[r["id"]]) for r in rows(alt / "predictions.tsv"))
        < 1e-7
    )
    args[pheno_option] = str(d / "trait.tsv")
    for i, r in enumerate(values):
        r["year"] = int(i >= 32)
        if i >= 32:
            r["y"] = str(1 + i / 30)
    phen = root / "phen"
    write(phen, values)
    args[args.index(str(d / "trait.tsv"))] = str(phen)
    cv = [
        "--year",
        "0",
        "--metric",
        "rmse",
        "--strict-simplex",
        "--probe-solver",
        "lanczos",
        "--probes",
        "8",
        "--max-probes",
        "128",
        "--mc-tol",
        ".15",
        "--trace-rank",
        "4",
    ]
    one = run("one", cv)
    many = run("many", cv + ["--parallel", "weights", "--threads", "2"])
    assert (one / "search.tsv").read_bytes() == (many / "search.tsv").read_bytes()
    for r in values[32:]:
        r["y"] = str(100 + float(r["y"]))
    write(phen, values)
    changed = run("truth", cv)
    assert (one / "cv_reml_trace.tsv").read_bytes() == (
        changed / "cv_reml_trace.tsv"
    ).read_bytes()
    assert (
        rows(one / "tuning.tsv")[0]["selection_score"]
        != rows(changed / "tuning.tsv")[0]["selection_score"]
    )
