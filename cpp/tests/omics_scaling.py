import csv
import math
import subprocess
import sys
import tempfile
from pathlib import Path


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    data = Path(__file__).parent / "data"
    ids = [r["ID"] for r in rows(data / "ped.tsv")]
    base = {}
    for label, values in [
        ("normal", [2 * i for i in range(24)]),
        ("large", [1e200 * i for i in range(24)]),
        ("offset", [10**16 + 2 * (i // 8) for i in range(24)]),
    ]:
        omics = root / label
        omics.write_text(
            "ID\tfeature\n" + "".join(f"{k}\t{v}\n" for k, v in zip(ids, values))
        )
        opts = ["--ped", data / "ped.tsv", "--snp", data / "snp.tsv", "--omics", omics]
        for cmd in ["build", "prepare"]:
            out = root / (cmd + label)
            subprocess.run([sys.argv[1], cmd, *opts, "--out", out], check=True)
            if cmd == "build":
                k = rows(out / "O.tsv")
                if label == "large":
                    assert (
                        max(
                            abs(float(r[i]) - float(s[i]))
                            for r, s in zip(k, base["O"])
                            for i in ids
                        )
                        < 1e-12
                    )
                base["O"] = k
                source = [v for c in "AGO" for v in ("--" + c, out / (c + ".tsv"))]
            else:
                stats = rows(out / "O.stats.tsv")[0]
                assert stats["keep"] == "1" and math.isfinite(float(stats["divisor"]))
                if label == "offset":
                    assert abs(float(stats["divisor"]) - math.sqrt(64 / 23)) < 1e-14
                source = ["--cache", out, "--cache-backend", "operator"]
            dest = root / (label + cmd + "fit")
            subprocess.run(
                [
                    sys.argv[1],
                    "fit",
                    *source,
                    "--pheno",
                    data / "pheno.tsv",
                    "--trait",
                    "trait",
                    "--alpha",
                    ".6",
                    "--beta",
                    ".2",
                    "--varcomp",
                    "fixed",
                    "--lambda",
                    "1",
                    "--out",
                    dest,
                ],
                check=True,
            )
            pred = rows(dest / "predictions.tsv")
            key = "normal" if label == "large" else label
            if key in base:
                assert [r["id"] for r in pred] == [r["id"] for r in base[key]]
                assert (
                    max(
                        abs(float(r[k]) - float(s[k]))
                        for r, s in zip(pred, base[key])
                        for k in ("u", "yhat")
                    )
                    < 1e-10
                )
            base[key] = pred
