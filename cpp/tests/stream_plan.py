import csv, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation/v042/replay"))
from reproduce_operators import stream_case, run

# Memcheck may emit large reachable-allocation reports in the raw stderr column.
csv.field_size_limit(sys.maxsize)

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    exe = Path(sys.argv[1]).resolve()
    stream_case(exe, root)

    def rows(p):
        with p.open() as f:
            return list(csv.DictReader(f, delimiter="\t"))

    assert all(r["exit"] == "0" for r in rows(root / "planning.tsv"))
    d = root / "stream"
    for p in (d / "weights_1_32", d / "weights_2_32", d / "weights_1_100"):
        for filename in ["search.tsv", "predictions.tsv", "tuning.tsv"]:
            assert (p / filename).read_bytes() == (
                d / "blas_1_32" / filename
            ).read_bytes()
    code, _, reason = run(
        exe,
        [
            "fit",
            "--cache",
            d / "cache",
            "--pheno",
            d / "pheno",
            "--trait",
            "trait",
            "--metric",
            "rmse",
            "--folds-file",
            d / "folds",
            "--varcomp",
            "fixed",
            "--lambda",
            1,
            "--cache-backend",
            "operator",
            "--cache-storage",
            "stream",
            "--parallel",
            "weights",
            "--memory-mb",
            0.01,
            "--out",
            d / "tiny",
        ],
    )
    assert code and "memory" in reason
