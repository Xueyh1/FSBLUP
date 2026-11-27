"""Collect compact summaries from independent, immutable validation runs."""

import argparse, csv, shutil, statistics
from pathlib import Path
from v043_panel import read, save


def copy(src, dst):
    if src.is_file():
        dst.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(src, dst)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    r = a.run
    v = a.output
    v.mkdir(exist_ok=True, parents=True)
    for kind, d in [("before", r / "before"), ("after", r / "release" / "after")]:
        for case, f in [
            ("replay_interior_modes", "summary"),
            ("replay_identifiability", "summary"),
            ("replay_temporal_scale", "observed"),
            ("connected_c", "observed"),
            ("feature_panel", "observed"),
            ("search_geometry", "geometry"),
        ]:
            copy(d / case / (f + ".tsv"), v / kind / (case + ".tsv"))
        copy(d / "exits.txt", v / kind / "exits.txt")
    for d in ("comparison", "connected", "validation", "panel"):
        for f in (r / "release" / d).glob("*.tsv"):
            copy(f, v / (d + "-" + f.name))
    for f in ("summary", "overlap"):
        copy(r / "release" / "medium-1" / (f + ".tsv"), v / ("medium-" + f + ".tsv"))
    for name, ds, key in [
        ("paired", [r / "release" / "comparison" / "paired.tsv"], ("n", "version")),
        (
            "modes",
            [r / "release" / "comparison" / "mode-cost.tsv"],
            ("method", "version"),
        ),
        (
            "medium",
            [r / "release" / f"medium-{i}" / "summary.tsv" for i in (1, 2, 3)],
            ("policy", "seed", "budget"),
        ),
    ]:
        if not all(d.exists() for d in ds):
            continue
        rows = [row for d in ds for row in read(d)]
        groups = {}
        for row in rows:
            groups.setdefault(tuple(row[k] for k in key), []).append(row)
        result = []
        for keys, group in groups.items():
            record = dict(zip(key, keys))
            col = "wall" if "wall" in group[0] else "wall_s"
            times = [float(x[col]) for x in group]
            med = statistics.median(times)
            record.update(
                repeats=len(group),
                wall_median=med,
                wall_MAD=statistics.median(abs(x - med) for x in times),
                rss_median=statistics.median(float(x["rss"]) for x in group),
            )
            for field in [
                "matvec",
                "rhs",
                "scored",
                "used",
                "completed",
                "truth_cor",
                "truth_rmse",
            ]:
                if field in group[0]:
                    record[field] = group[0][field]
            result.append(record)
        save(v / (name + "-medians.tsv"), result)
    records = []
    for batch in ["b1", "b2", "b3", "b4", "b5", "bfinal"]:
        d = r / batch
        row = dict(batch=batch)
        for tool in ("release", "asan", "tsan", "mem"):
            f = d / (tool + "-tests.log")
            row[tool] = (
                next(
                    (
                        line
                        for line in f.read_text().splitlines()
                        if "% tests passed" in line
                    ),
                    "incomplete",
                )
                if f.exists()
                else "pending"
            )
            copy(f, v / "batches" / batch / f.name)
        life = read(d / "lifetime.tsv")
        row.update(
            iterations=len(life) and life[-1]["iteration"],
            fd=life[-1]["fds"] if life else None,
            threads=life[-1]["threads"] if life else None,
        )
        for f in ("lifetime.tsv", "valgrind-lifetime.log", "controls/summary.tsv"):
            copy(d / f, v / "batches" / batch / f)
        records.append(row)
    save(v / "batches.tsv", records)


if __name__ == "__main__":
    main()
