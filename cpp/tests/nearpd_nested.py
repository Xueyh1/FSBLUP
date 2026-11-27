"""Replay a weight-sensitive, ID-permuted C search against frozen R/GLS scores."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile

def records(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))

def compare(data, run):
    actual = records(run/"search.tsv")
    expected = records(data/"scores.tsv")
    assert len(actual) == len(expected)
    errors = []
    for r, ref in zip(actual, expected):
        for col in ("alpha","beta"):
            assert float(r[col]) == float(ref[col])
        assert r["valid"]=="1" and r["nearpd_blocks"]=="1"
        errors.extend(abs(float(r[c])-float(ref[c])) for c in ("r1f1","r1f2","r1f3","mean"))
    assert max(errors)<1e-6, max(errors)
    best = next(r for r in actual if r["full_status"]=="selected")
    pd = records(run/"pd.tsv")[0]
    assert best["full_nearpd_blocks"]=="1"
    assert float(best["full_max_relative_normF"]) == float(pd["relative_normF"])
    assert all(r["full_max_relative_normF"]=="NA" for r in actual if r!=best)
    def matrix(path):
        rows=records(path)
        return {(r["ID"],j):float(v) for r in rows for j,v in r.items() if j!="ID"}
    k=matrix(run/"K.tsv")
    ref=matrix(data/"selected_K.tsv")
    assert k.keys()==ref.keys()
    err=max(abs(k[ij]-ref[ij]) for ij in k)
    assert err < 1e-11, err
    print("Max absolute fold/mean error",max(errors),"selected full K error",err)

if __name__=="__main__":
    data=Path(__file__).parent/"nested"
    with tempfile.TemporaryDirectory() as tmp:
        run=Path(tmp)/"fit"
        args=[sys.argv[1],"fit","--out",run,"--pd-policy","nearPD","--strict-simplex","--search-policy","legacy","--write-k",
              "--folds-file",data/"folds.tsv","--metric","rmse","--id","ID","--trait","trait","--pheno",data/"pheno.tsv"]
        for layer in "AGO":
            args.extend(["--"+layer,data/(layer+".tsv")])
        subprocess.run(args,check=True)
        compare(data,run)
