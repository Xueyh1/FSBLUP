"""Four simulated generations, complete-cohort C certification and a dense ID oracle."""

import argparse, csv, subprocess, time
from pathlib import Path
import numpy as np
from v043_panel import tab, read, save

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("binary", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
d = a.output
d.mkdir()
rng = np.random.default_rng(890)
n = 1024
ids = [f"p{j:04d}" for j in range(n)]
roots = list(range(n))


def find(i):
    while roots[i] != i:
        roots[i] = roots[roots[i]]
        i = roots[i]
    return i


ped = []
for j, key in enumerate(ids):
    gen = j // 256
    if not gen:
        ped.append((key, 0, 0))
        continue
    sire = (gen - 1) * 256 + int(rng.integers(128))
    dam = (gen - 1) * 256 + 128 + int(rng.integers(128))
    ped.append((key, ids[sire], ids[dam]))
    roots[find(j)] = find(sire)
    roots[find(sire)] = find(dam)
tab(d / "ped", ["ID", "Sire", "Dam"], ped)
g = rng.choice(n, 64, replace=False)
o = g[:24]
cover = {find(j) for j in g}
active = sum(find(j) in cover for j in range(n))
for name, ix, m in [("snp", g, 96), ("omics", o, 8)]:
    vv = (
        rng.binomial(2, 0.4, (len(ix), m))
        if name == "snp"
        else rng.normal(size=(len(ix), m))
    )
    tab(
        d / name,
        ["ID", *[f"v{j}" for j in range(m)]],
        ([ids[i], *row] for i, row in zip(ix, vv)),
    )
t = rng.choice(n, 128, replace=False)
tab(
    d / "y",
    ["ID", "y"],
    ((ids[j], rng.normal() if i < 96 else "NA") for i, j in enumerate(t)),
)
subprocess.run(
    [
        a.binary,
        "prepare",
        "--ped",
        d / "ped",
        "--snp",
        d / "snp",
        "--omics",
        d / "omics",
        "--snp-format",
        "2bit",
        "--out",
        d / "cache",
    ],
    check=True,
)
res = []
preds = {}
for backend, cap in [("operator", 16), ("operator", 320), ("dense", 320)]:
    out = d / (backend + str(cap))
    start = time.monotonic()
    proc = subprocess.run(
        list(
            map(
                str,
                [
                    a.binary,
                    "fit",
                    "--cache",
                    d / "cache",
                    "--cache-backend",
                    backend,
                    "--pheno",
                    d / "y",
                    "--trait",
                    "y",
                    "--alpha",
                    0.2,
                    "--beta",
                    0.4,
                    "--pd-policy",
                    "nearPD",
                    "--varcomp",
                    "fixed",
                    "--lambda",
                    1,
                    "--memory-mb",
                    cap,
                    "--out",
                    out,
                ],
            )
        ),
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - start
    (d / (out.name + ".stderr")).write_text(proc.stderr)
    use = (read(out / "cache_usage.tsv") or [{}])[0]
    res.append(
        dict(
            backend=backend,
            cap=cap,
            exit=proc.returncode,
            active=active,
            reported=use.get("active_component_ids"),
            certificate=use.get("certification_backend"),
            reason=use.get("fallback_reason", proc.stderr.splitlines()[-1]),
            wall=elapsed,
            rss=(read(out / "runtime.tsv") or [{}])[0].get("peak_rss_kib"),
            max_d_prediction=None,
        )
    )
    if not proc.returncode:
        preds[backend] = read(out / "predictions.tsv")
assert (
    res[0]["exit"]
    and not res[1]["exit"]
    and not res[2]["exit"]
    and int(res[1]["reported"]) == active
), res
left, right = preds["operator"], preds["dense"]
assert [r["id"] for r in left] == [r["id"] for r in right]
res[1]["max_d_prediction"] = max(
    abs(float(x[k]) - float(y[k])) for x, y in zip(left, right) for k in ("u", "yhat")
)
assert res[1]["max_d_prediction"] < 1e-8
save(d / "summary.tsv", res)
