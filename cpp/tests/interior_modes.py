import math, csv, subprocess, sys, tempfile
from pathlib import Path


def rows(p):
    with p.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def fit(kernel, pheno, opts, out):
    args = [
        sys.argv[1],
        "fit",
        *[v for c in "AGO" for v in ("--" + c, kernel)],
        "--pheno",
        pheno,
        "--id",
        "id",
        "--trait",
        "y",
        "--alpha",
        "0",
        "--beta",
        "0",
        "--trace",
        "hutchinson",
        *opts,
        "--out",
        out,
    ]
    p = subprocess.run(args, capture_output=True, text=True)
    assert not p.returncode, p.stderr
    return rows(out / "model.tsv")[0]


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    d = Path(__file__).parent / "v043"
    for method in ["ai-reml", "em-reml"]:
        for seed in [17, 42]:
            out = root / (method + str(seed))
            m = fit(
                d / "K.tsv",
                d / "y.tsv",
                ["--varcomp", method, "--probe-seed", str(seed)],
                out,
            )
            assert abs(float(m["contrast_loglik"]) + 12.41283843291) < 1e-7, m
            preds = {r["id"]: float(r["u"]) for r in rows(out / "predictions.tsv")}
            assert (
                abs(preds["C1"] - 0.0014331594) < 1e-5
                and abs(preds["C2"] - 0.067579024) < 1e-5
            )
    ids = ["T" + str(i) for i in range(1, 9)]
    with (root / "K").open("w") as f:
        print("id", *ids, sep="\t", file=f)
        for i, id in enumerate(ids):
            print(id, *[1 if i == j else 0.5 for j in range(8)], sep="\t", file=f)
    (root / "y").write_text(
        "\n".join((d / "y.tsv").read_text().splitlines()[:9]) + "\n"
    )
    for method in ["ai-reml", "em-reml"]:
        out = root / (method + "flat")
        m = fit(root / "K", root / "y", ["--varcomp", method], out)
        assert (
            "unidentifiable" in m["status"]
            and rows(out / "pcg_summary.tsv")[0]["updates"] == "0"
        )
    local = fit(d / "K.tsv", d / "y.tsv", ["--mode-policy", "local"], root / "local")
    assert (
        local["status"] == "approx-interior-local"
        and abs(float(local["contrast_loglik"]) + 13.26684619876) < 1e-6
    )

    for perturb in (0, 0.25):
        with (root / "K").open("w") as f:
            print("id", *ids, sep="\t", file=f)
            for i, id in enumerate(ids):
                print(
                    id,
                    *[
                        0.5 * (i == j)
                        + 0.2
                        + 0.3 * (-1) ** (i + j)
                        + (perturb * i if i == j else 0)
                        for j in range(8)
                    ],
                    sep="\t",
                    file=f,
                )
        with (root / "design").open("w") as f:
            print("id", "y", "x", sep="\t", file=f)
            for i, id in enumerate(ids):
                print(id, math.sin(i * i), (-1) ** i, sep="\t", file=f)
        out = root / ("covar" + str(perturb))
        if not perturb:
            m = fit(root / "K", root / "design", ["--covar", "x"], out)
            assert (
                m["status"].startswith("unidentifiable")
                and rows(out / "pcg_summary.tsv")[0]["updates"] == "0"
            )
        else:
            args = [
                sys.argv[1],
                "fit",
                *[v for c in "AGO" for v in ("--" + c, root / "K")],
                "--pheno",
                root / "design",
                "--id",
                "id",
                "--trait",
                "y",
                "--covar",
                "x",
                "--alpha",
                "0",
                "--beta",
                "0",
                "--trace",
                "hutchinson",
                "--out",
                out,
            ]
            proc = subprocess.run(args, capture_output=True, text=True)
            assert "unidentifiable" not in proc.stderr
            if not proc.returncode:
                assert "unidentifiable" not in rows(out / "model.tsv")[0]["status"]
