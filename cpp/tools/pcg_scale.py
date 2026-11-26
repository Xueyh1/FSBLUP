"""Generate/replay genuine streamed features; directory names include all dimensions.
Run in a Slurm allocation. Timings include actual text preparation, full cache validation,
and predictions for every ID. Byte counts distinguish normalized doubles from genotypes.
"""

import argparse, csv, os, subprocess, time
from pathlib import Path
import numpy as np


def save(path, header, rows):
    with path.open("w") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def generate(d, n, m, h, nested):
    d.mkdir(parents=True)
    rng = np.random.default_rng(274)
    ids = [f"i{i:07}" for i in range(n)]
    last = n - min(72, n // 3) if nested else n
    sire = ["0"] * n
    dam = sire.copy()
    u = rng.normal(size=n)
    for j in range(2, last, 3):
        sire[j] = ids[j - 2]
        dam[j] = ids[j - 1]
        u[j] = (u[j - 2] + u[j - 1]) / 2 + u[j] * np.sqrt(0.5)
    save(d / "ped.tsv", ["ID", "Sire", "Dam"], zip(ids, sire, dam))
    y = u + rng.normal(size=n)
    for typ, cols, start in [
        ("snp", m, last if nested else 0),
        ("omics", h, n - min(36, n // 5) if nested else 0),
    ]:
        effects = rng.normal(size=cols) / np.sqrt(cols)
        with (d / (typ + ".tsv")).open("w") as f:
            f.write("ID\t" + "\t".join("f" + str(i) for i in range(cols)) + "\n")
            for i in range(start, n, 256):
                size = min(256, n - i)
                if typ == "snp":
                    z = rng.binomial(2, 0.4, size=(size, cols)).astype(np.int8)
                    y[i : i + size] += 0.9 * ((z - 0.8) @ effects)
                    z[::119, 0] = -9
                    text = np.empty((size, 2 * cols), dtype=np.uint8)
                    text[:, ::2] = z + 48
                    text[:, 1::2] = 9
                    text[:, -1] = 10
                    f.flush()
                    for j, id_ in enumerate(ids[i : i + size]):
                        payload = text[j].tobytes()
                        if z[j, 0] == -9:
                            payload = b"-9" + payload[1:]
                        f.buffer.write(id_.encode() + b"\t" + payload)
                else:
                    z = rng.normal(size=(size, cols))
                    y[i : i + size] += 0.5 * (z @ effects)
                    for id_, row in zip(ids[i : i + size], z):
                        f.write(id_ + "\t" + "\t".join("%.9g" % v for v in row) + "\n")
    for cohort in ("small", "large"):
        small = set(range(min(128, n // 2)))
        if nested:
            small.update(range(last, n, 2))
        ref = [i for i in range(n) if (i in small if cohort == "small" else i % 5 != 0)]
        selected = set(ref)
        save(
            d / f"{cohort}.tsv",
            ["ID", "trait"],
            ((k, y[i] if i in selected else "NA") for i, k in enumerate(ids)),
        )
        save(
            d / f"{cohort}-folds.tsv",
            ["repeat", "fold", "id"],
            ((1, j % 2 + 1, ids[i]) for j, i in enumerate(ref)),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("binary", type=Path)
    p.add_argument("root", type=Path)
    p.add_argument("--sizes", default="10000,50000,150000")
    p.add_argument("--markers", type=int, default=13)
    p.add_argument("--omics", type=int, default=9)
    p.add_argument("--threads", default="1,2,4,8")
    p.add_argument("--cohorts", default="small,large")
    p.add_argument("--reml-maxiter", type=int)
    p.add_argument("--methods", default="fixed")
    p.add_argument("--modes", default="A,C")
    p.add_argument("--memory-mb", default="2048")
    p.add_argument("--trace", default="exact")
    p.add_argument("--tag", default="run")
    p.add_argument("--storage", default="auto")
    p.add_argument("--trace-rank", default="0")
    p.add_argument("--slq-batch", default="1")
    p.add_argument("--reorth", default="local")
    p.add_argument("--slq-depth", default="40")
    p.add_argument("--probe-solver", default="lanczos")
    p.add_argument("--probes", default="8")
    p.add_argument("--max-probes", default="128")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--prec", default="jacobi")
    p.add_argument("--parallel", default="blas")
    p.add_argument("--snp-format", default="f64")
    a = p.parse_args()
    a.root.mkdir(exist_ok=True, parents=True)
    exe = a.binary.resolve()
    cpus = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        cpus.setdefault(
            tuple((path / f).read_text() for f in ("physical_package_id", "core_id")),
            cpu,
        )
    fields = [
        "N_A",
        "N_G",
        "N_O",
        "markers",
        "omics",
        "n_train",
        "n_outputs",
        "p",
        "method",
        "threads",
        "elapsed_s",
        "status",
        "peak_rss_KiB",
        "rhs",
        "matvec",
        "bytes",
        "weights",
        "scored_fits",
        "backend",
        "reason",
    ]
    with (a.root / (a.tag + ".tsv")).open("x") as log:
        wr = csv.DictWriter(log, fields, delimiter="\t", lineterminator="\n")
        wr.writeheader()
        for mode in a.modes.split(","):
            for n in map(int, a.sizes.split(",")):
                d = a.root / f"{mode}-{n}-{a.markers}-{a.omics}"
                if not d.exists():
                    generate(d, n, a.markers, a.omics, mode == "C")
                cache = d / "cache"

                def run(args, out, nt, method, train):
                    assert len(cpus) >= nt
                    mask = ",".join(map(str, list(cpus.values())[:nt]))
                    cmd = [
                        "taskset",
                        "-c",
                        mask,
                        str(exe),
                        *map(str, args),
                        "--threads",
                        str(nt),
                        "--memory-mb",
                        a.memory_mb,
                        "--log-level",
                        "off",
                        "--out",
                        out,
                    ]
                    begin = time.perf_counter()
                    try:
                        r = subprocess.run(
                            cmd,
                            timeout=a.timeout,
                            capture_output=True,
                            text=True,
                            env={
                                **os.environ,
                                "OPENBLAS_NUM_THREADS": str(
                                    1
                                    if method.startswith("cv-")
                                    and a.parallel == "weights"
                                    else nt
                                ),
                                "OMP_NUM_THREADS": "1",
                            },
                        )
                        code, reason = r.returncode, r.stderr.strip()
                    except subprocess.TimeoutExpired:
                        code, reason = 124, "per-job time budget"
                    row = dict(
                        N_A=n,
                        N_G=n if mode == "A" else min(72, n // 3),
                        N_O=n if mode == "A" else min(36, n // 5),
                        markers=a.markers,
                        omics=a.omics,
                        n_train=train,
                        n_outputs=n,
                        p=1,
                        method=method,
                        threads=nt,
                        elapsed_s=time.perf_counter() - begin,
                        status=code,
                        reason=reason,
                    )

                    def rows(f):
                        return list(csv.DictReader((out / f).open(), delimiter="\t"))

                    if not code:
                        row["peak_rss_KiB"] = rows("runtime.tsv")[0]["peak_rss_kib"]
                        if method != "prepare":
                            details = rows("pcg_summary.tsv")
                            cv = (
                                rows("cv_pcg_summary.tsv")
                                if (out / "cv_pcg_summary.tsv").exists()
                                else []
                            )
                            row.update(
                                rhs=sum(int(v["rhs"]) for v in cv + details),
                                matvec=sum(int(v["matvec"]) for v in cv + details),
                                bytes=rows("cache_usage.tsv")[0]["feature_bytes"],
                                backend=rows("cache_usage.tsv")[0]["backend"],
                                weights=len(rows("search.tsv")) if cv else 1,
                                scored_fits=len(cv) + 1,
                            )
                    wr.writerow(row)
                    log.flush()
                    print(
                        mode, n, method, nt, code, row["elapsed_s"], reason, flush=True
                    )

                if not cache.exists():
                    run(
                        [
                            "prepare",
                            "--ped",
                            d / "ped.tsv",
                            "--snp",
                            d / "snp.tsv",
                            "--omics",
                            d / "omics.tsv",
                            "--snp-format",
                            a.snp_format,
                        ],
                        cache,
                        1,
                        "prepare",
                        0,
                    )
                for cohort in a.cohorts.split(","):
                    with (d / f"{cohort}-folds.tsv").open() as f:
                        train = sum(1 for _ in f) - 1
                    for nt in map(int, a.threads.split(",")):
                        for method in a.methods.split(","):
                            estimator = method.removeprefix("cv-")
                            opts = ["--varcomp", estimator]
                            if estimator == "fixed":
                                opts += ["--lambda", "1.5"]
                            else:
                                opts += ["--trace", a.trace]
                                if a.reml_maxiter:
                                    opts += ["--reml-maxiter", str(a.reml_maxiter)]
                                if a.trace == "hutchinson":
                                    opts += [
                                        "--trace-rank",
                                        a.trace_rank,
                                        "--slq-batch",
                                        a.slq_batch,
                                        "--slq-reorth",
                                        a.reorth,
                                        "--probe-solver",
                                        a.probe_solver,
                                        "--probes",
                                        a.probes,
                                        "--max-probes",
                                        a.max_probes,
                                        "--slq-depth",
                                        a.slq_depth,
                                        "--mc-tol",
                                        ".025",
                                    ]
                            if method.startswith("cv-"):
                                opts += [
                                    "--parallel",
                                    a.parallel,
                                    "--folds-file",
                                    d / f"{cohort}-folds.tsv",
                                    "--metric",
                                    "rmse",
                                    "--strict-simplex",
                                ]
                            else:
                                opts += ["--alpha", ".2", "--beta", ".5"]
                            run(
                                [
                                    "fit",
                                    "--cache",
                                    cache,
                                    "--cache-backend",
                                    "operator",
                                    "--cache-storage",
                                    a.storage,
                                    "--preconditioner",
                                    a.prec,
                                    "--pheno",
                                    d / f"{cohort}.tsv",
                                    "--trait",
                                    "trait",
                                    *opts,
                                ],
                                d / f"{a.tag}-{cohort}-{method}-{nt}",
                                nt,
                                method,
                                train,
                            )


if __name__ == "__main__":
    main()
