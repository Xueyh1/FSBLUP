"""Fixed-grid retry comparison with latent genetic values kept outside fit."""

import csv, itertools, argparse, time, subprocess, gzip
from pathlib import Path
import numpy as np


def tab(p, header, rows):
    with p.open("w") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def read(p):
    if not p.is_file():
        return []
    with p.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def save(p, rs):
    if rs:
        tab(p, list(rs[0]), ([r[k] for k in rs[0]] for r in rs))


def inputs(d, n, m, h):
    d.mkdir()
    rng = np.random.default_rng(743)
    ids = [f"g{i:05}" for i in range(n)]
    x = rng.binomial(2, 0.37, (n, m))
    z = rng.normal(size=(n, h))
    tab(d / "ped", ["ID", "Sire", "Dam"], ((s, 0, 0) for s in ids))
    for name, data in [("snp", x), ("omics", z)]:
        tab(
            d / name,
            ["ID", *[f"f{j}" for j in range(data.shape[1])]],
            ([s, *row] for s, row in zip(ids, data)),
        )
    u = 0.85 * ((x - 0.74) @ rng.normal(size=m)) / np.sqrt(0.4662 * m) + 0.5 * (
        z @ rng.normal(size=h)
    ) / np.sqrt(h)
    y = u + rng.normal(size=n) * 0.8
    tr = min(128, n // 2)
    tab(
        d / "y",
        ["ID", "trait"],
        ((s, y[i] if i < tr else "NA") for i, s in enumerate(ids)),
    )
    tab(
        d / "folds",
        ["repeat", "fold", "id"],
        ((1, 1 + j % 2, s) for j, s in enumerate(ids[:tr])),
    )
    tab(d / "truth", ["ID", "u", "phenotype"], zip(ids, u, y))


def cmp(left, right, truth, name, writer):
    x, y = read(left / "predictions.tsv"), read(right / "predictions.tsv")
    if not x or not y:
        return dict(
            case=name,
            n=0,
            top1=None,
            top5=None,
            top10=None,
            rho=None,
            max_du=None,
            max_dyhat=None,
        )
    assert [(r["id"], r["role"]) for r in x] == [(r["id"], r["role"]) for r in y]
    vals = lambda rs: np.array([float(r["u"]) for r in rs if r["role"] == "candidate"])
    v, w = vals(x), vals(y)
    n = len(v)
    # Stable ties follow the original phenotype order in both lists.
    rank1, rank2 = [np.argsort(-a, kind="stable") for a in (v, w)]
    overlap = []
    for frac in (0.01, 0.05, 0.1):
        k = max(1, int(np.ceil(n * frac)))
        overlap.append(len(set(rank1[:k]) & set(rank2[:k])) / k)
    du = np.array([float(b["u"]) - float(a["u"]) for a, b in zip(x, y)])
    dy = np.array([float(b["yhat"]) - float(a["yhat"]) for a, b in zip(x, y)])
    names = [r["id"] for r in x if r["role"] == "candidate"]
    lr, rr = [
        {names[k]: j + 1 for j, k in enumerate(order)} for order in (rank1, rank2)
    ]
    writer.writerows(
        [name, r["id"], r["role"], a, b, lr.get(r["id"]), rr.get(r["id"])]
        for r, a, b in zip(x, du, dy)
    )
    return dict(
        case=name,
        n=n,
        top1=overlap[0],
        top5=overlap[1],
        top10=overlap[2],
        rho=np.corrcoef(np.argsort(rank1), np.argsort(rank2))[0, 1],
        max_du=np.abs(du).max(),
        max_dyhat=np.abs(dy).max(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exe", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--cap", type=int, default=32)
    p.add_argument("--mc", type=float, default=0.18)
    p.add_argument("--n", type=int, default=768)
    a = p.parse_args()
    d = a.out
    d.mkdir()
    src = d / "inputs"
    inputs(src, a.n, 32, 8)
    exe = a.exe.resolve()
    subprocess.run(
        [
            exe,
            "prepare",
            "--ped",
            src / "ped",
            "--snp",
            src / "snp",
            "--omics",
            src / "omics",
            "--snp-format",
            "2bit",
            "--out",
            src / "cache",
        ],
        check=True,
    )
    truths = {r["ID"]: float(r["u"]) for r in read(src / "truth")}
    stats = []
    comparisons = []
    with gzip.open(d / "delta.tsv.gz", "wt") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(["case", "id", "role", "du", "dyhat", "left_rank", "right_rank"])
        for seed, budget in itertools.product((19, 42), (128, 512)):
            for policy in ("legacy", "rounds"):
                out = d / f"{seed}-{budget}-{policy}"
                cmd = [
                    exe,
                    "fit",
                    "--cache",
                    src / "cache",
                    "--pheno",
                    src / "y",
                    "--trait",
                    "trait",
                    "--folds-file",
                    src / "folds",
                    "--metric",
                    "rmse",
                    "--strict-simplex",
                    "--search-policy",
                    "grid",
                    "--retry-policy",
                    policy,
                    "--trace",
                    "hutchinson",
                    "--probe-solver",
                    "lanczos",
                    "--slq-depth",
                    36,
                    "--trace-rank",
                    16,
                    "--probes",
                    4,
                    "--max-probes",
                    a.cap,
                    "--mc-tol",
                    a.mc,
                    "--retry-probes",
                    budget,
                    "--probe-seed",
                    seed,
                    "--memory-mb",
                    256,
                    "--parallel",
                    "weights",
                    "--threads",
                    2,
                    "--out",
                    out,
                ]
                start = time.monotonic()
                run = subprocess.run(
                    list(map(str, cmd)), capture_output=True, text=True
                )
                (d / (out.name + ".log")).write_text(run.stderr)
                t = read(out / "tuning.tsv")[0]
                r = read(out / "runtime.tsv")[0]
                fits = read(out / "cv_pcg_summary.tsv") + read(out / "pcg_summary.tsv")
                assert (
                    int(t["candidates_visited"]) == 15
                    and int(t["retry_used"]) <= int(t["retry_reserved"]) <= budget
                )
                preds = [
                    v for v in read(out / "predictions.tsv") if v["role"] == "candidate"
                ]
                u = np.array([float(v["u"]) for v in preds])
                v = np.array([truths[r["id"]] for r in preds])
                stats.append(
                    dict(
                        seed=seed,
                        budget=budget,
                        policy=policy,
                        exit=run.returncode,
                        wall=time.monotonic() - start,
                        rss=r["peak_rss_kib"],
                        visited=t["candidates_visited"],
                        scored=t["candidates_scored"],
                        attempted=t["fold_attempts"],
                        completed=t["folds_completed"],
                        reserved=t["retry_reserved"],
                        used=t["retry_used"],
                        discarded=t["discarded_fit_probes"],
                        alpha=t["selected_alpha"],
                        beta=t["selected_beta"],
                        matvec=sum(int(v["matvec"]) for v in fits),
                        rhs=sum(int(v["rhs"]) for v in fits),
                        truth_cor=np.corrcoef(u, v)[0, 1] if preds else None,
                        truth_rmse=np.sqrt(np.mean((u - v) ** 2)) if preds else None,
                    )
                )
                save(d / "summary.tsv", stats)
            comparisons.append(
                cmp(
                    d / f"{seed}-{budget}-legacy",
                    d / f"{seed}-{budget}-rounds",
                    truths,
                    f"{seed}-{budget}:rounds-legacy",
                    wr,
                )
            )
        for policy, budget in itertools.product(("rounds", "legacy"), (128, 512)):
            comparisons.append(
                cmp(
                    d / f"19-{budget}-{policy}",
                    d / f"42-{budget}-{policy}",
                    truths,
                    f"{policy}-{budget}:42-19",
                    wr,
                )
            )
        for seed, policy in itertools.product((19, 42), ("rounds", "legacy")):
            comparisons.append(
                cmp(
                    d / f"{seed}-128-{policy}",
                    d / f"{seed}-512-{policy}",
                    truths,
                    f"{seed}-{policy}:512-128",
                    wr,
                )
            )
    save(d / "overlap.tsv", comparisons)


if __name__ == "__main__":
    main()
