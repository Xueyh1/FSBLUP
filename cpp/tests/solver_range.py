"""Check REML units and diagnostics for extreme finite phenotype scales."""
import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def records(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_pheno(directory, rows, exponent, offset):
    path = directory / f"trait_{exponent}.tsv"
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        for raw in rows:
            row = raw.copy()
            if row["trait"] != "NA":
                value = round(float(row["trait"]) * 16) / 16 + offset
                row["trait"] = repr(math.ldexp(value, exponent))
            writer.writerow(row)
    return path


def close(value, expected):
    assert math.isclose(value, expected, rel_tol=3e-6, abs_tol=5e-7), (value, expected)


def run(binary, root):
    data = Path(__file__).resolve().parent / "data"
    rows = records(data / "pheno.tsv")
    base = [binary, "fit", "--A", data / "A.tsv", "--G", data / "G.tsv",
            "--O", data / "O.tsv", "--trait", "trait", "--covar", "age",
            "--factor", "herd"]
    base += ["--solver","direct","--varcomp","profile-reml"]
    fixed = ["--alpha", ".2", "--beta", ".5"]
    for exponent, offset in ((0, 0), (500, 11), (-500, 11)):
        pheno = write_pheno(root, rows, exponent, offset)
        out = root / str(exponent)
        subprocess.run(base + fixed + ["--pheno", pheno, "--out", out], check=True)
        model = records(out / "model.tsv")[0]
        pred = records(out / "predictions.tsv")
        effects = records(out / "fixed.tsv")
        if exponent == 0:
            m0, p0, b0 = model, pred, effects
            continue
        assert model["status"] == m0["status"]
        close(float(model["lambda"]), float(m0["lambda"]))
        for key in ("Vu", "Ve"):
            close(math.ldexp(float(model[key]), -2*exponent), float(m0[key]))
        df = int(model["reference"]) - len(effects)
        close(float(model["profile_objective"]) - 2*df*exponent*math.log(2),
              float(m0["profile_objective"]))
        assert len(pred) == len(p0) and len(effects) == len(b0)
        for a, b in zip(pred, p0):
            assert (a["id"], a["role"]) == (b["id"], b["role"])
            close(math.ldexp(float(a["u"]), -exponent), float(b["u"]))
            close(math.ldexp(float(a["yhat"]), -exponent) - offset, float(b["yhat"]))
        for a, b in zip(effects, b0):
            assert a["term"] == b["term"]
            centered = math.ldexp(float(a["estimate"]), -exponent)
            if a["term"] == "(Intercept)":
                centered -= offset
            close(centered, float(b["estimate"]))

    for exponent in (-665, 665):
        pheno = write_pheno(root, rows, exponent, 0)
        out = root / str(exponent)
        failed = subprocess.run(base + fixed + ["--pheno", pheno, "--out", out],
                                capture_output=True, text=True)
        assert failed.returncode and "representable range" in failed.stderr, failed.stderr
        assert not out.exists()

    out = root / "cv"
    failed = subprocess.run(base + ["--pheno", pheno, "--folds-file", data / "folds.tsv",
                                    "--metric", "rmse", "--out", out],
                            capture_output=True, text=True)
    assert failed.returncode and (out / "failure.txt").exists()
    trial_rows = records(out / "search.tsv")
    assert len(trial_rows) == 15
    assert all(r["valid"] == "0" and "representable range" in r["reason"]
               for r in trial_rows), trial_rows


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        run(Path(sys.argv[1]).resolve(), Path(directory))
