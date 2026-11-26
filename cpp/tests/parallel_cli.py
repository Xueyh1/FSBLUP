"""Determinism, low-memory scheduling, recovery and competing publication."""

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def rows(p):
    with p.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def run(bin, root):
    data = Path(__file__).parent / "nested"
    args = [
        bin,
        "fit",
        "--id",
        "ID",
        "--trait",
        "trait",
        "--metric",
        "rmse",
        "--strict-simplex",
        "--pd-policy",
        "nearPD",
        "--pheno",
        data / "pheno.tsv",
        "--folds-file",
        data / "folds.tsv",
        "--write-k",
    ]
    for k in "AGO":
        args.extend(["--" + k, data / (k + ".tsv")])

    def call(name, *extra):
        return subprocess.run(
            [*args, *extra, "--out", root / name], capture_output=True, text=True
        )

    def succeed(name, *extra):
        result = call(name, *extra)
        assert result.returncode == 0, result.stderr
        return root / name

    ref = succeed("serial")
    for n in (1, 2, 4, 8):
        d = succeed(str(n), "--parallel", "weights", "--threads", str(n))
        stats = rows(d / "runtime.tsv")[0]
        assert (int(stats["threads"]), int(stats["weight_workers"])) == (1, n)
        for file in (
            "K.tsv",
            "K_raw.tsv",
            "model.tsv",
            "search.tsv",
            "predictions.tsv",
            "tuning.tsv",
            "folds.tsv",
        ):
            assert (ref / file).read_bytes() == (d / file).read_bytes(), file
    args.extend(["--parallel", "weights", "--threads", "8"])
    capped = succeed("capped", "--memory-mb", "22")
    assert rows(capped / "worker_memory.tsv")[0]["workers"] == "2"
    failed = call("no-memory", "--memory-mb", "5")
    assert (
        failed.returncode
        and "candidate workspace" in failed.stderr
        and not (root / "no-memory").exists()
    )
    with ThreadPoolExecutor(2) as pool:
        codes = list(pool.map(lambda _: call("same-destination").returncode, range(2)))
    assert sorted(codes) == [0, 1] and not list(root.glob("*.tmp.*"))
    args[args.index("nearPD")] = "strict"
    failed = call("all-failed")
    assert failed.returncode and (root / "all-failed" / "failure.txt").is_file()
    assert all(
        "indefinite" in r["reason"] for r in rows(root / "all-failed" / "search.tsv")
    )


if __name__ == "__main__":
    with TemporaryDirectory() as d:
        run(sys.argv[1], Path(d))
