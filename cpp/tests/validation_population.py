import csv,subprocess,sys,tempfile
from pathlib import Path
def rows(p):
    with p.open() as f:return list(csv.DictReader(f,delimiter="\t"))
with tempfile.TemporaryDirectory() as work:
    p=Path(work);d=Path(__file__).parent/"data"
    data=[r for r in rows(d/"pheno.tsv") if r["trait"]!="NA"]
    for label in (0,400,"auc"):
        with (p/"y").open("w") as f:
            print("ID","y","year","z",sep="\t",file=f)
            for i,r in enumerate(data):
                test=i%4==0
                z=(1 if test else i%2) if label=="auc" else (float(r["trait"]) if test else label)
                print(r["ID"],r["trait"],int(test),z,sep="\t",file=f)
        out=p/str(label)
        cmd=[sys.argv[1],"fit",*[v for k in "AGO" for v in ("--"+k,d/(k+".tsv"))],"--pheno",p/"y","--year","0",
             "--metric","auc" if label=="auc" else "rmse","--score-col","z","--trait","y","--strict-simplex","--varcomp","fixed","--lambda","1","--out",out]
        run=subprocess.run(cmd,capture_output=True,text=True);t=rows(out/"tuning.tsv")[0]
        if label=="auc":
            assert run.returncode and "no scorable fold" in run.stderr and t["invalid_kernel"]==t["fold_attempts"]=="0" and t["validation_unscorable"]=="1",t
        else:
            assert not run.returncode,run.stderr
            if label==0: base=t;search=rows(out/"search.tsv")
            else: assert base==t and search==rows(out/"search.tsv")
