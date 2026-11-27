import csv,math,subprocess,sys,tempfile
from pathlib import Path
exe=Path(sys.argv[1]).resolve()
def read(p):
    with p.open() as f:return list(csv.DictReader(f,delimiter="\t"))
with tempfile.TemporaryDirectory() as work:
    d=Path(work)
    src=Path(__file__).parent/"data"
    pheno=read(src/"pheno.tsv")
    for i,scale in enumerate((1,.001,1000)):
        p=d/f"pheno{i}"
        with p.open("w") as f:
            print("ID","y","z",sep="\t",file=f)
            for r in pheno:
                y=r["trait"]
                print(r["ID"],scale*float(y) if y!="NA" else y,
                      scale*(float(y)+.13*int(r["ID"])) if y!="NA" else "NA",sep="\t",file=f)
        out=d/str(i)
        cmd=[exe,"fit",*[v for c in "AGO" for v in ("--"+c,src/(c+".tsv"))],
             "--pheno",p,"--trait","y","--score-col","z","--metric","rmse","--folds-file",src/"folds.tsv",
             "--strict-simplex","--varcomp","fixed","--lambda","1","--out",out]
        subprocess.run(cmd,check=True,capture_output=True)
        r=read(out/"search.tsv");t=read(out/"tuning.tsv")[0]
        if i==0: ref=r;winner=t
        assert [(a["alpha"],a["beta"],a["stage"]) for a in r]==[(a["alpha"],a["beta"],a["stage"]) for a in ref]
        assert t["stop"]==winner["stop"] and t["selected_alpha"]==winner["selected_alpha"] and t["selected_beta"]==winner["selected_beta"]
        for x,y in zip(r,ref): assert math.isclose(float(x["mean"])/scale,float(y["mean"]),rel_tol=1e-10)
