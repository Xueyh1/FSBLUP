"""Memcheck small production paths with expected application exit codes."""
import argparse
import csv
import os
from pathlib import Path
import re
import resource
import signal
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("build",type=Path);p.add_argument("out",type=Path)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
data=Path(__file__).resolve().parents[2]/"tests"; tiny=data/"data"; nested=data/"nested"
def kernels(d):
    return [v for k in "AGO" for v in ("--"+k,d/(k+".tsv"))]
tasks=[
 ("raw",0,["build","--ped",tiny/"ped.tsv","--snp",tiny/"snp.tsv","--omics",tiny/"omics.tsv"]),
 ("spd",0,["fuse",*kernels(tiny),"--alpha",.2,"--beta",.5]),
 ("fixed",0,["fit",*kernels(tiny),"--pheno",tiny/"pheno.tsv","--trait","trait","--alpha",.2,"--beta",.5]),
 ("project",0,["fuse",*kernels(nested),"--alpha",.2,"--beta",.5,"--pd-policy","nearPD"]),
 ("strict",1,["fuse",*kernels(nested),"--alpha",.2,"--beta",.5]),
 ("outer",0,["fit",*kernels(nested),"--pheno",nested/"pheno.tsv","--trait","trait","--folds-file",nested/"folds.tsv","--metric","rmse","--pd-policy","nearPD","--threads",4,"--parallel","weights"]),
 ("short",1,["fuse",*kernels(tiny),"--alpha",.2,"--beta",.5]),
 ("repeat",0,["30"])]
os.environ.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
with (a.out/"summary.tsv").open("w") as summary:
    w=csv.writer(summary,delimiter="\t",lineterminator="\n")
    w.writerow(["case","exit","errors","definitely_lost","indirectly_lost","possibly_lost","still_reachable"])
    for name,expected,flags in tasks:
        log=a.out/(name+".log")
        cmd=["valgrind","--tool=memcheck","--leak-check=full","--show-leak-kinds=all",
             "--errors-for-leak-kinds=definite,indirect","--track-origins=yes","--error-exitcode=99"]
        # Keep the diagnostic stream outside the capped process's regular files.
        if name=="short":
            def setup():
                signal.signal(signal.SIGXFSZ,signal.SIG_IGN)
                resource.setrlimit(resource.RLIMIT_FSIZE,(512,512))
        else: setup=None
        exe="repeat_test" if name=="repeat" else "fsblup"
        args=[*cmd,a.build/exe,*flags]
        if name!="repeat":args+=["--out",a.out/name]
        run=subprocess.run(list(map(str,args)),stderr=subprocess.PIPE,stdout=subprocess.PIPE,preexec_fn=setup)
        text=run.stderr.decode()
        log.write_text(text+run.stdout.decode())
        errors=int(re.search(r"ERROR SUMMARY: ([\d,]+)",text)[1].replace(",",""))
        lost=[]
        for kind in ("definitely lost","indirectly lost","possibly lost","still reachable"):
            match=re.search(kind+r":\s+([\d,]+) bytes",text)
            lost.append(int(match[1].replace(",","")) if match else 0)
        w.writerow([name,run.returncode,errors,*lost]);summary.flush()
        if run.returncode!=expected or errors or any(lost[:2]):raise RuntimeError(f"inspect {log}")
