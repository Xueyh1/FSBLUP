"""Cold-hinted versus warm file pages; no system-wide cache eviction."""

import csv, os, shutil, subprocess, sys, time
from pathlib import Path

binary, source, out = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=False)
cache = out / "cache"
shutil.copytree(source / "cache", cache)
stats = []
for rep in range(3):
    for storage in ("stream", "auto"):
        for hint in ("cold-hinted", "warm"):
            if hint == "cold-hinted":
                for p in cache.iterdir():
                    with p.open("rb") as f:
                        os.fsync(f.fileno())
                        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            dst = out / f"{storage}-{hint}-{rep}"
            cmd = [
                binary.resolve(),
                "fit",
                "--cache",
                cache,
                "--cache-backend",
                "operator",
                "--cache-storage",
                storage,
                "--trait",
                "trait",
                "--pheno",
                source / "large.tsv",
                "--alpha",
                ".2",
                "--beta",
                ".5",
                "--varcomp",
                "fixed",
                "--lambda",
                "1.5",
                "--out",
                dst,
                "--log-level",
                "off",
            ]
            tick = time.perf_counter()
            subprocess.run(cmd, check=True)
            elapsed = time.perf_counter() - tick

            def row(p):
                with (dst / p).open() as f:
                    return next(csv.DictReader(f, delimiter="\t"))

            io = row("cache_usage.tsv")
            work = row("pcg_summary.tsv")
            stats.append(
                dict(
                    rep=rep,
                    storage=storage,
                    hint=hint,
                    wall=elapsed,
                    verify_bytes=io["verified_cache_bytes"],
                    feature_bytes=io["feature_bytes"],
                    read_calls=io["read_calls"],
                    read_s=io["read_seconds"],
                    decode_s=io["decode_seconds"],
                    apply_s=work["operator_s"],
                    gls_s=work["gls_s"],
                    predict_s=work["predict_s"],
                )
            )
with (out / "io.tsv").open("w") as f:
    w = csv.DictWriter(f, stats[0], delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(stats)
