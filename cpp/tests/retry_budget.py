import csv, subprocess, sys, tempfile
from pathlib import Path


def rows(p):
    with p.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


d = Path(__file__).parent / "pcg_data"
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    values = rows(d / "trait.tsv")
    for i, v in enumerate(values):
        v["year"] = int(i >= 32)
        if i >= 32:
            v["y"] = 1 + i / 30
    with (root / "y").open("w") as f:
        w = csv.DictWriter(f, values[0], delimiter="\t")
        w.writeheader()
        w.writerows(values)
    for policy, workers in (("legacy", 1), ("legacy", 2), ("rounds", 1), ("rounds", 2)):
        dest = root / (policy + str(workers))
        args = [
            sys.argv[1],
            "fit",
            *[v for c in "AGO" for v in ("--" + c, d / "kernel.tsv")],
            "--pheno",
            root / "y",
            "--trait",
            "y",
            "--year",
            "0",
            "--metric",
            "rmse",
            "--strict-simplex",
            "--trace",
            "hutchinson",
            "--probe-solver",
            "lanczos",
            "--slq-depth",
            "24",
            "--probes",
            "2",
            "--max-probes",
            "2",
            "--mc-tol",
            "1e-12",
            "--retry-policy",
            policy,
            "--retry-probes",
            "12",
            "--parallel",
            "weights",
            "--threads",
            str(workers),
            "--out",
            dest,
        ]
        r = subprocess.run(args, capture_output=True, text=True)
        assert r.returncode and (dest / "failure.txt").exists(), r.stderr
        t = rows(dest / "tuning.tsv")[0]
        assert (
            int(t["probe_exhausted"]) == 15
            and int(t["candidates_scored"]) == 0
            and int(t["retry_reserved"]) == 12
        ), t
        assert int(t["folds_completed"]) == 0
        assert (
            int(t["fold_attempts"]) == (17 if policy == "legacy" else 18)
            and int(t["folds_scored_last_attempt"]) == 0
        )
        s = rows(dest / "search.tsv")
        assert (
            (
                int(s[0]["retry_reserved_probes"]) == 12
                and "8:" in s[0]["probe_attempts"]
            )
            if policy == "legacy"
            else all(r["retry_reserved_probes"] == "4" for r in s[:3])
        )
        if workers == 1:
            first = s
        else:
            assert first == s
