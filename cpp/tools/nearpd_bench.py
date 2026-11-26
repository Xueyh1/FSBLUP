"""Single-thread, warm-cache fsblup timing and PD audits. See VALIDATION.md."""
import argparse
import csv
import os
from pathlib import Path
from statistics import median
import subprocess

def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))

def save(path, records):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, records[0].keys(), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

def pedigree_subblocks(official, dest):
    # Identical coverage and dimensions for a positive-definite control.
    for layer in ("G", "O"):
        with (official/f"test_mat{layer}.csv").open() as f:
            ids = next(csv.reader(f))[1:]
        with (official/"test_matA.csv").open() as f:
            table = csv.DictReader(f)
            keep = set(ids)
            subset = {r["ID"]: r for r in table if r["ID"] in keep}
        save(dest/f"{layer}.tsv", [{"ID": i, **{j: subset[i][j] for j in ids}} for i in ids])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("binary", "official", "masked", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--scenario", choices=["grid","official","nested"], required=True)
    parser.add_argument("--nested", type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    inp = args.official
    abc = [inp/f"test_mat{k}.csv" for k in "AGO"]
    tasks = []
    def case(name, matrices, phen, trait, options):
        idcol = "ID" if name == "nested" else "id"
        flags = ["--A",matrices[0],"--G",matrices[1],"--O",matrices[2],
                 "--pheno",phen,"--id",idcol,"--trait",trait,"--pd-policy","nearPD","--threads","1"]
        tasks.append((name, flags+options))
    if args.scenario == "grid":
        for a in range(5):
            for b in range(5-a):
                case(f"a{a}b{b}",abc,args.masked,"age",["--alpha",a/4,"--beta",b/4])
    elif args.scenario == "official":
        pedigree_subblocks(inp,out)
        weights = ["--alpha",".2","--beta",".5","--write-k"]
        case("shift",[abc[0],out/"G.tsv",out/"O.tsv"],args.masked,"age",weights)
        case("project",abc,args.masked,"age",weights)
    else:
        case("nested",[args.nested/f"{k}.tsv" for k in "AGO"],args.nested/"pheno.tsv","trait",
             ["--folds-file",args.nested/"folds.tsv","--metric","rmse","--strict-simplex","--write-k"])
    audits, times = [], []
    for rep in range(4):
        batch = []
        for name, flags in tasks:
            run = out/f"{name}.{rep}"
            subprocess.run([str(args.binary.resolve()),"fit","--out",str(run),*map(str,flags)],check=True)
            time = {r["stage"]:float(r["seconds"]) for r in rows(run/"timing.tsv")}
            rt = rows(run/"runtime.tsv")[0]
            rec = dict(case=name,rep=rep,read_s=time["read_kernels"]+time["read_phenotype"],
                       search_s=time.get("search",0.),fusion_s=time["fusion"],
                       write_k_s=time.get("write_kernels",0.),reml_s=time["final_reml_blup"],
                       total_s=float(rt["wall_seconds"]),rss_KiB=int(rt["peak_rss_kib"]))
            batch.append(rec)
            pd = rows(run/"pd.tsv")[-1]
            assert pd["policy"]=="nearPD" and pd["block"]=="K"
            if name=="shift":
                assert pd["action"]=="jitter" and float(pd["relative_normF"])==0
            else:
                assert pd["action"]=="nearPD" and pd["converged"]=="1" and float(pd["min_eigen_used"])>0
            if rep==1:
                fusion=rows(run/"fusion.tsv")[0]
                audits.append(dict(case=name,alpha=fusion["alpha"],beta=fusion["beta"],**pd))
            # Retain one dump for the independent numeric checks.
            if rep!=1:
                for filename in ("K.tsv","K_raw.tsv"):
                    (run/filename).unlink(missing_ok=True)
        times.extend(batch)
        if args.scenario=="grid":
            group=dict(case="grid15",rep=rep)
            for col in list(batch[0])[2:]:
                group[col]=(max if col=="rss_KiB" else sum)(r[col] for r in batch)
            times.append(group)
        save(out/"runs.tsv",times)
    save(out/"audit.tsv",audits)
    summary=[]
    for name in dict.fromkeys(r["case"] for r in times):
        repeat=[r for r in times if r["case"]==name and r["rep"]]
        summary.append(dict(case=name,**{col:median(r[col] for r in repeat) for col in list(times[0])[2:]}))
    save(out/"medians.tsv",summary)

if __name__=="__main__":
    main()
