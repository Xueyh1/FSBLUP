"""Check 0.3's frozen CLI, including reference-only fusion and sampled logging cost."""

import csv, statistics, subprocess, sys, time
from pathlib import Path

old, new, data, out = map(Path, sys.argv[1:])
old = old.resolve()
new = new.resolve()
out.mkdir(parents=True, exist_ok=False)


def read(p):
    with p.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write(p, records, fields=None):
    with p.open("w") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields or records[0].keys(),
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(records)


evidence = []
for mode in "ABC":
    d = data / ("nested" if mode == "C" else "data")
    layers = [(d / (("Ob" if mode == "B" and k == "O" else k) + ".tsv")) for k in "AGO"]
    opts = [s for layer, path in zip("AGO", layers) for s in ("--" + layer, path)]
    cv = [
        "--pheno",
        d / "pheno.tsv",
        "--trait",
        "trait",
        "--folds-file",
        d / "folds.tsv",
        "--metric",
        "rmse",
        "--write-k",
        "--strict-simplex",
        "--pd-policy",
        "nearPD",
    ]
    for exe, name, opts2 in [
        (old, "old", []),
        (
            new,
            "new",
            ["--solver", "direct", "--varcomp", "profile-reml", "--log-level", "off"],
        ),
    ]:
        dest = out / (mode + name)
        subprocess.run([exe, "fit", *opts, *cv, *opts2, "--out", dest], check=True)
    ref = out / (mode + "old")
    actual = out / (mode + "new")
    for file in ("folds", "search", "pd", "K", "K_raw", "fixed", "predictions", "ids"):
        assert (ref / (file + ".tsv")).read_bytes() == (
            actual / (file + ".tsv")
        ).read_bytes(), (mode, file)
    m1 = read(ref / "model.tsv")[0]
    m2 = read(actual / "model.tsv")[0]
    assert all(m1[k] == m2[k] for k in m1)
    wanted = [r["ID"] for r in read(d / "pheno.tsv") if r["trait"] != "NA"]
    refopts = []
    for layer, path in zip("AGO", layers):
        records = read(path)
        key = next(iter(records[0]))
        indices = [r[key] for r in records if r[key] in wanted]
        file = out / (mode + layer + ".tsv")
        write(file, [r for r in records if r[key] in indices], [key, *indices])
        refopts += ["--" + layer, file]
    weights = read(ref / "fusion.tsv")[0]
    for exe, name in [(old, "old"), (new, "new")]:
        target = out / (mode + "ref" + name)
        subprocess.run(
            [
                exe,
                "fuse",
                *refopts,
                "--alpha",
                weights["alpha"],
                "--beta",
                weights["beta"],
                "--pd-policy",
                "nearPD",
                "--out",
                target,
            ],
            check=True,
        )
    assert (out / (mode + "refold") / "K.tsv").read_bytes() == (
        out / (mode + "refnew") / "K.tsv"
    ).read_bytes()
    evidence.append(
        dict(
            mode=mode,
            reference_ids=len(wanted),
            fold_rows=len(read(ref / "folds.tsv")),
            trials=len(read(ref / "search.tsv")),
            data_equal="yes",
        )
    )
write(out / "compatibility.tsv", evidence)
d = data / "data"
opts = [s for k in "AGO" for s in ("--" + k, d / (k + ".tsv"))]
stats = []
first = None
for rep in range(6):
    for level in ["off", "info", "debug"][rep % 3 :] + ["off", "info", "debug"][
        : rep % 3
    ]:
        dest = out / f"{level}-{rep}"
        log = out / f"{level}-{rep}.log"
        tick = time.perf_counter()
        subprocess.run(
            [
                new,
                "fit",
                *opts,
                "--pheno",
                d / "pheno.tsv",
                "--trait",
                "trait",
                "--alpha",
                ".2",
                "--beta",
                ".5",
                "--out",
                dest,
                "--log-level",
                level,
                "--log-every",
                "1",
                "--log-file",
                log,
            ],
            stderr=subprocess.DEVNULL,
            check=True,
        )
        elapsed = time.perf_counter() - tick
        bits = (dest / "predictions.tsv").read_bytes()
        if first is None:
            first = bits
        assert bits == first
        stats.append(
            dict(level=level, rep=rep, seconds=elapsed, bytes=log.stat().st_size)
        )
write(out / "logging.tsv", stats)
