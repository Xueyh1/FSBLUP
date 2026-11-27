"""Independent diagonal GLS predictions and the orthonormal-profile reference."""
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

exe = Path(sys.argv[1]).resolve()
data = Path(__file__).parent / "face"

def rows(p):
    with p.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))

with tempfile.TemporaryDirectory() as d:
    weights = [.002,.01,.1,1,5,10,20,100]
    truth = rows(data/"y.tsv")
    y = [float(r["y"]) for r in truth]
    g, e = 1.0110784086,.8270563024
    vi = [1/(g*k + e) for k in weights]
    b = sum(a*v for a,v in zip(y,vi))/sum(vi)
    for method in ["ai-reml","em-reml"]:
        for i,(trace,opts) in enumerate([
            ("exact",["--init-vu","1e-5","--init-ve","6.325402076"]),
            ("hutchinson",[]), ("hutchinson",["--init-vu","1e-5","--init-ve","6.325402076"]),
            ("hutchinson",["--probe-solver","lanczos","--slq-depth","8","--probe-seed","17",
                           "--probes","32","--max-probes","32"])]):
            out=Path(d)/(method+str(i))
            args=[exe,"fit",*[v for k in "AGO" for v in ("--"+k,data/"K.tsv")],
                  "--pheno",data/"y.tsv","--id","id","--trait","y","--alpha","0","--beta","0",
                  "--varcomp",method,"--trace",trace,*opts,"--out",out]
            p=subprocess.run(args,capture_output=True,text=True)
            assert not p.returncode,p.stderr
            m=rows(out/"model.tsv")[0]
            assert abs(float(m["contrast_loglik"])+9.19651386833)<1e-7,m
            assert abs(float(m["Vu"])-g)<1e-5 and abs(float(m["Ve"])-e)<1e-5,m
            for j,r in enumerate(rows(out/"predictions.tsv")):
                u=g*weights[j]*vi[j]*(y[j]-b)
                assert r["id"]==truth[j]["id"] and abs(float(r["u"])-u)<1e-5
                assert abs(float(r["yhat"])-b-u)<1e-5
            if i==1:
                assert any(r["status"]=="interior-restart" for r in rows(out/"reml_trace.tsv"))
