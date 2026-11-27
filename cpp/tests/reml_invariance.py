"""Analytic contrast fixtures. The ninth row is an unphenotyped relative."""
import csv
import math
import subprocess
import sys
import tempfile
from pathlib import Path

exe = Path(sys.argv[1]).resolve()
observe = "--observe" in sys.argv

def table(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))

def fixture(root, case, method, shift=0, offset=0, units=1):
    if case == "large-start":
        eig, energy = [2, 1, 2, 4], [0, 3, 5, 9]
    elif case == "two-peaks":
        eig = [2, .0001, .001, .01, .1, 1, 2, 4]
        energy = [0, .09115771, .23225591, .06013410, 10.79779764,
                  6.60499804, 12.73051338, .01842142]
    else:
        eig, energy = [2, 2, 1, 2, 3, 4, 5, 6], [0, 0, 3, 5, 7, 9, 11, 13]
    n = len(eig)
    q = [[(-1)**(bin(i & j).count("1")) / math.sqrt(n) for j in range(n)] for i in range(n)]
    ids = [str(i) for i in range(1,n+2)]
    k = [[sum(l*a*b for l,a,b in zip(eig,r,s)) for s in q] for r in q]
    cross = [v*.1 for v in k[0]]
    for i,r in enumerate(k):
        r.append(cross[i])
    k.append(cross+[2])
    with (root/"K").open("w") as f:
        print("id", *ids, sep="\t", file=f)
        for i,r in enumerate(k):
            r[i] -= 1e-6
            print(ids[i], *r, sep="\t", file=f)
    with (root/"y").open("w") as f:
        print("id\ty\tx", file=f)
        for i,key in enumerate(ids):
            x = (-1)**i if i<n else 3
            y = sum(a*math.sqrt(v) for a,v in zip(q[i],energy))+shift*x if i<n else "NA"
            print(key,y,x*units+offset,sep="\t",file=f)
    out=root/str(len(list(root.iterdir())))
    args=[exe,"fit",*[s for c in "AGO" for s in ("--"+c,root/"K")],
          "--pheno",root/"y","--id","id","--trait","y","--alpha","0","--beta","0",
          "--out",out,"--varcomp",method,"--reml-maxiter","30000","--reml-tol","1e-10"]
    if case=="design":
        args.extend(["--covar","x"])
    if case=="large-start":
        args.extend(["--init-vu","1e12","--init-ve","1e12"])
    proc=subprocess.run(args,capture_output=True,text=True)
    if proc.returncode and observe:
        print(case,method,shift,offset,units,proc.stderr.strip()); return
    assert proc.returncode==0,proc.stderr
    m=table(out/"model.tsv")[0]
    t=table(out/"reml_trace.tsv")[-1]
    a=table(out/"pcg_summary.tsv")[0]
    print(case,method,shift,offset,units,*(a[c] for c in ("initial_Vu","initial_Ve","updates")),
          *(m[c] for c in ("Vu","Ve","lambda","status")),t["contrast_loglik"],sep="\t")
    if observe: return
    if case=="two-peaks":
        assert abs(float(t["contrast_loglik"])+7.41302154)<1e-6
        assert abs(float(m["lambda"])-.00510521)<1e-7
    else:
        for c,v in [("Vu",2),("Ve",1),("lambda",.5)]:
            assert abs(float(m[c])-v)<v*1e-5,(c,m)
        if case=="design":
            assert abs(float(t["contrast_loglik"])+8.90701478)<1e-6
    return table(out/"predictions.tsv"),table(out/"fixed.tsv")

with tempfile.TemporaryDirectory() as temp:
    root=Path(temp)
    for method in ["ai-reml","em-reml"]:
        for case in ["large-start","two-peaks"]:
            fixture(root,case,method)
        base=fixture(root,"design",method)
        for shift,offset,units in [(1e6,0,1),(0,20000,1),(0,0,1e8)]:
            new=fixture(root,"design",method,shift,offset,units)
            if observe: continue
            for i,(p,r) in enumerate(zip(base[0],new[0])):
                x=(-1)**i if i<8 else 3
                assert p["id"]==r["id"] and abs(float(p["u"])-float(r["u"]))<1e-6
                assert abs(float(r["yhat"])-float(p["yhat"])-shift*x)<1e-6
            b=float(base[1][1]["estimate"])
            assert abs(float(new[1][1]["estimate"])-(b+shift)/units)<1e-7
