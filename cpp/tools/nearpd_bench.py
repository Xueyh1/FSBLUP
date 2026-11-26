"""Slurm CPU-affinity benchmark. Timings include formatting and buffered filesystem writes."""

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def save(path, records):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, records[0].keys(), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(records)


def physical_cpus():
    groups = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = tuple(
            (topology / k).read_text().strip()
            for k in ("physical_package_id", "core_id")
        )
        groups.setdefault(key, cpu)
    return list(groups.values())


def medium(dest, n=192):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import numpy as np

    rng = np.random.default_rng(4127)
    ids = [f"{i:05}" for i in range(n)]
    for name, start in (("A", 0), ("G", n // 3), ("O", 2 * n // 3)):
        z = rng.normal(size=(n - start, 36))
        k = 0.55 * np.eye(len(z)) + 0.45 * (z @ z.T) / 36
        if name == "A":
            k[0, n // 3] = k[n // 3, 0] = 0.7
        if name == "G":
            k[0, 0] = 3
        with (dest / f"{name}.tsv").open("w") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["ID", *ids[start:]])
            w.writerows([i, *row] for i, row in zip(ids[start:], k))
    y = rng.normal(size=n) + np.sin(np.arange(n) * 0.31)
    ref = [i for i in range(n) if i % 13]
    save(
        dest / "pheno.tsv",
        [dict(ID=id_, trait=y[i] if i % 13 else "NA") for i, id_ in enumerate(ids)],
    )
    save(
        dest / "folds.tsv",
        [dict(repeat=1, fold=j % 3 + 1, id=ids[i]) for j, i in enumerate(ref)],
    )


def pedigree_subblocks(official, dest):
    for layer in ("G", "O"):
        with (official / f"test_mat{layer}.csv").open() as f:
            ids = next(csv.reader(f))[1:]
        with (official / "test_matA.csv").open() as f:
            keep = set(ids)
            subset = {r["ID"]: r for r in csv.DictReader(f) if r["ID"] in keep}
        save(
            dest / f"{layer}.tsv",
            [{"ID": i, **{j: subset[i][j] for j in ids}} for i in ids],
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("binary", "official", "masked", "output"):
        p.add_argument(k, type=Path)
    p.add_argument(
        "--scenario",
        choices=["grid", "official", "nested", "medium", "build", "suite"],
        required=True,
    )
    p.add_argument("--nested", type=Path, default=Path("tests/nested"))
    p.add_argument(
        "--threads", default="1", help="comma-separated physical core counts"
    )
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--variant",
        action="append",
        default=[],
        help="additional label=executable, compared in the same rounds",
    )
    p.add_argument("--parallel", choices=["blas", "both", "weights"], default="blas")
    a = p.parse_args()
    out = a.output
    if a.repeats < 1:
        p.error("--repeats must be at least 1")
    cpus = physical_cpus()
    threads = [int(v) for v in a.threads.split(",")]
    if not threads or min(threads) < 1 or max(threads) > len(cpus):
        p.error(f"allocated physical CPUs {cpus}")
    if len(set(threads)) != len(threads) or any("=" not in v for v in a.variant):
        p.error("use unique thread counts and label=executable variants")
    variants = [("current", a.binary)] + [
        (v.split("=", 1)[0], Path(v.split("=", 1)[1])) for v in a.variant
    ]
    labels = [label for label, _ in variants]
    if len(set(labels)) != len(labels) or any(
        not label or "/" in label for label in labels
    ):
        p.error("variant labels must be distinct file names")
    out.mkdir(parents=True, exist_ok=False)
    inp = a.official
    abc = [inp / f"test_mat{k}.csv" for k in "AGO"]
    tasks = {}

    def fit(ks, phen, idcol, trait, flags):
        return [
            "fit",
            "--A",
            ks[0],
            "--G",
            ks[1],
            "--O",
            ks[2],
            "--pheno",
            phen,
            "--id",
            idcol,
            "--trait",
            trait,
            "--pd-policy",
            "nearPD",
            *flags,
        ]

    fixed = ["--alpha", ".2", "--beta", ".5"]
    if a.scenario in ("build", "suite"):
        tasks["build"] = [
            [
                "build",
                "--ped",
                inp / "test_pedi.csv",
                "--snp",
                inp / "test_snp.csv",
                "--omics",
                inp / "test_omic.csv",
                "--omics-type",
                "fpkm",
            ]
        ]
    if a.scenario in ("official", "suite"):
        pedigree_subblocks(inp, out)
        for name, ks, flags in (
            ("shift", [abc[0], out / "G.tsv", out / "O.tsv"], ["--write-k"]),
            ("project", abc, ["--write-k"]),
            ("no_write", abc, []),
        ):
            tasks[name] = [fit(ks, a.masked, "id", "age", fixed + flags)]
    if a.scenario == "grid":
        tasks["grid15"] = [
            fit(abc, a.masked, "id", "age", ["--alpha", i / 4, "--beta", j / 4])
            for i in range(5)
            for j in range(5 - i)
        ]
    for name in ("nested", "medium"):
        if a.scenario in (name, "suite"):
            d = a.nested
            if name == "medium":
                d = out / "medium"
                d.mkdir()
                medium(d)
            tasks[name] = [
                fit(
                    [d / f"{k}.tsv" for k in "AGO"],
                    d / "pheno.tsv",
                    "ID",
                    "trait",
                    [
                        "--folds-file",
                        d / "folds.tsv",
                        "--metric",
                        "rmse",
                        "--strict-simplex",
                        "--write-k",
                    ],
                )
            ]
    configurations = [
        (v, exe, t, k, mode)
        for k in tasks
        for t in threads
        for v, exe in variants
        for mode in (
            ["blas", "weights"]
            if a.parallel == "both" and k in ("nested", "medium")
            else ["weights"]
            if a.parallel == "weights" and k in ("nested", "medium")
            else ["blas"]
        )
    ]
    data = []
    audits = []
    weight_runs = []
    legacy_output = (
        len(threads) == 1
        and len(variants) == 1
        and a.parallel == "blas"
        and a.scenario in ("grid", "official", "nested")
    )
    for rep in range(a.repeats + 1):
        order = (
            configurations[rep % len(configurations) :]
            + configurations[: rep % len(configurations)]
        )
        if rep % 2:
            order = order[::-1]
        for version, exe, t, k, mode in order:
            rec = dict(
                version=version,
                case=k,
                mode=mode,
                requested=t,
                rep=rep,
                cpus=",".join(map(str, cpus[:t])),
            )
            totals = {}
            peak = 0
            size = 0
            for idx, flags in enumerate(tasks[k]):
                dest = out / "current-output"
                cmd = [
                    "taskset",
                    "-c",
                    rec["cpus"],
                    str(exe.resolve()),
                    *map(str, flags),
                    "--out",
                    str(dest),
                    "--threads",
                    str(t),
                ]
                if mode == "weights":
                    cmd += ["--parallel", "weights"]
                tick = time.perf_counter()
                subprocess.run(
                    cmd,
                    check=True,
                    env={
                        **os.environ,
                        "OPENBLAS_NUM_THREADS": str(1 if mode == "weights" else t),
                        "OMP_NUM_THREADS": "1",
                    },
                )
                elapsed = time.perf_counter() - tick
                rt = rows(dest / "runtime.tsv")[0]
                times = {
                    r["stage"]: float(r["seconds"]) for r in rows(dest / "timing.tsv")
                }
                times.update(total=float(rt["wall_seconds"]), process_wall=elapsed)
                for stage, val in times.items():
                    totals[stage] = totals.get(stage, 0) + val
                peak = max(peak, int(rt["peak_rss_kib"]))
                size += sum(f.stat().st_size for f in dest.iterdir())
                rec.update(
                    blas_threads=int(rt["threads"]),
                    workers=int(rt.get("weight_workers", 1)),
                )
                if k in ("project", "no_write", "grid15"):
                    assert rows(dest / "pd.tsv")[-1]["action"] == "nearPD"
                name = k
                if k == "grid15":
                    fusion = rows(dest / "fusion.tsv")[0]
                    name = f"a{round(float(fusion['alpha']) * 4)}b{round(float(fusion['beta']) * 4)}"
                    weight_runs.append(
                        dict(
                            **rec, point=name, rss_KiB=int(rt["peak_rss_kib"]), **times
                        )
                    )
                if rep == 1 and k != "build":
                    fusion = rows(dest / "fusion.tsv")[0]
                    audits.append(
                        dict(
                            version=version,
                            case=name,
                            requested=t,
                            mode=mode,
                            alpha=fusion["alpha"],
                            beta=fusion["beta"],
                            **rows(dest / "pd.tsv")[-1],
                        )
                    )
                if legacy_output:
                    if rep != 1:
                        for fname in ("K.tsv", "K_raw.tsv"):
                            (dest / fname).unlink(missing_ok=True)
                    dest.rename(out / f"{name}.{rep}")
                    continue
                if rep == 1 and idx == 0 and k != "grid15":
                    keep = out / f"{version}-{k}-{mode}-{t}"
                    keep.mkdir()
                    for f in dest.iterdir():
                        if f.name not in (
                            "K.tsv",
                            "K_raw.tsv",
                            "A.tsv",
                            "G.tsv",
                            "O.tsv",
                        ):
                            shutil.copyfile(f, keep / f.name)
                shutil.rmtree(dest)
            rec.update(rss_KiB=peak, output_bytes=size, **totals)
            data.append(rec)
            keys = dict.fromkeys(key for r in data for key in r)
            save(out / "runs.tsv", [{key: r.get(key, 0) for key in keys} for r in data])
    if audits:
        save(out / "audit.tsv", audits)
    if weight_runs:
        save(out / "weight-runs.tsv", weight_runs)
    summary = []
    for version, exe, t, k, mode in configurations:
        rr = [
            r
            for r in data
            if (r["version"], r["requested"], r["case"], r["mode"])
            == (version, t, k, mode)
            and r["rep"]
        ]
        base = {
            key: rr[0][key]
            for key in (
                "version",
                "case",
                "mode",
                "requested",
                "blas_threads",
                "workers",
                "cpus",
            )
        }
        for col in rr[0]:
            if col in base or col == "rep":
                continue
            vals = [r[col] for r in rr]
            base[col] = statistics.median(vals)
            if col == "total":
                base.update(min_s=min(vals), max_s=max(vals))
        summary.append(base)
    for row in summary:
        if legacy_output:
            row.update(
                read_s=row.get("read_kernels", 0) + row.get("read_phenotype", 0),
                search_s=row.get("search", 0),
                fusion_s=row.get("fusion", 0),
                write_k_s=row.get("write_kernels", 0),
                reml_s=row.get("final_reml_blup", 0),
                total_s=row["total"],
            )
        baseline = next(
            (
                r["total"]
                for r in summary
                if r["version"] == row["version"]
                and r["case"] == row["case"]
                and r["mode"] == row["mode"]
                and r["requested"] == 1
            ),
            None,
        )
        row["speedup_vs_one"] = baseline / row["total"] if baseline is not None else ""
    keys = dict.fromkeys(key for r in summary for key in r)
    save(out / "medians.tsv", [{key: r.get(key, 0) for key in keys} for r in summary])


if __name__ == "__main__":
    main()
