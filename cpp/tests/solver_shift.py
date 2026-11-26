"""Check that an intercept absorbs a large, exactly representable trait offset."""
import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def fit(binary, data, work, offset, label):
    rows = read(data / "pheno.tsv")
    for row in rows:
        if row["trait"] != "NA":
            # Multiples of 1/16 are exactly representable at both offsets.
            value = round(float(row["trait"]) * 2) / 16
            row["trait"] = repr(offset + value)
    pheno = work / (label + ".tsv")
    with pheno.open("w") as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    out = work / label
    subprocess.run([binary, "fit", "--A", data / "A.tsv", "--G", data / "G.tsv",
                    "--O", data / "O.tsv", "--pheno", pheno, "--trait", "trait",
                    "--covar", "age", "--factor", "herd", "--alpha", "0.2",
                    "--beta", "0.5", "--out", out], check=True)
    return read(out / "predictions.tsv"), read(out / "model.tsv")[0]


def main(binary, data):
    shift = 1e14
    with tempfile.TemporaryDirectory() as tmp:
        p0, m0 = fit(binary, data, Path(tmp), 0., "original")
        p1, m1 = fit(binary, data, Path(tmp), shift, "shifted")
    assert len(p0) == len(p1) and m0["status"] == m1["status"]
    for key in ("lambda", "Vu", "Ve", "profile_objective"):
        assert math.isclose(float(m0[key]), float(m1[key]), abs_tol=1e-10)
    for old, new in zip(p0, p1):
        assert (old["id"], old["role"]) == (new["id"], new["role"])
        assert math.isclose(float(old["u"]), float(new["u"]), abs_tol=1e-10)
        assert abs((shift + float(old["yhat"])) - float(new["yhat"])) <= math.ulp(shift)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(__file__).resolve().parent / "data")
