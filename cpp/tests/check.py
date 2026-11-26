"""Offline CLI acceptance with named matrix coordinates and independent R fixtures."""
import csv
import math
import subprocess
import sys
import tempfile
from pathlib import Path

BIN = Path(sys.argv[1]).resolve()
DATA = Path(__file__).resolve().parent / "data"


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write(path, rows, fields=None):
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields or rows[0].keys(), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return path


def compare(actual, expected, columns, key, name, atol=1e-8, rtol=1e-7):
    ref = {r[key]: r for r in expected}
    assert {r[key] for r in actual} == ref.keys()
    worst = (-1, "", "")
    for row in actual:
        for c in columns:
            v = float(row[c])
            target = float(ref[row[key]][c])
            diff = abs(v - target)
            assert diff <= atol + rtol * abs(target), (name, row[key], c, v, target)
            worst = max(worst, (diff, row[key], c))
    print(f"{name}: max absolute error {worst[0]:.9g} at {worst[1]}/{worst[2]}")


def matrix(path, name):
    expected = read(DATA / (name + ".tsv"))
    compare(read(path), expected, list(expected[0])[1:], "ID", name)


def run(tmp):
    def cli(cmd, dest, args, failure=""):
        target = tmp / dest
        p = subprocess.run([BIN, cmd, "--out", target, *map(str, args)],
                           text=True, capture_output=True)
        if failure:
            assert p.returncode and failure in p.stderr, (dest, p.stderr)
        else:
            assert p.returncode == 0, (dest, p.stderr)
        return target

    raw = ["--ped", DATA/"ped.tsv", "--snp", DATA/"snp.tsv", "--omics", DATA/"omics.tsv"]
    kernels = cli("build", "kernels", raw)
    for layer in "AGO":
        matrix(kernels/(layer+".tsv"), layer)

    def layers(a="A", g="G", o="O", base=DATA):
        return ["--A", base/(a+".tsv"), "--G", base/(g+".tsv"), "--O", base/(o+".tsv")]

    weights = ["--alpha", ".2", "--beta", ".5"]
    fit = ["--id", "ID", "--trait", "trait", "--pheno", DATA/"pheno.tsv"]
    out = cli("fuse", "fused", layers()+weights)
    for name in ("K", "K_raw"):
        matrix(out/(name+".tsv"), name)
    for a,b in ((0,0),(1,0),(0,1)):
        out = cli("fuse", f"end{a}{b}", layers()+["--alpha",a,"--beta",b])
        matrix(out/"K.tsv", f"end{a}{b}")
    out = cli("fuse", "subset", layers(o="Ob")+weights)
    matrix(out/"K.tsv", "B_K")
    for name,o,a in (("C_expected","C_O",.2),("C_changed","C_perturb",.2),
                       ("C_zero","C_O",0),("C_zero","C_perturb",0)):
        out = cli("fuse", name+o, layers("C_A","C_G",o)+["--alpha",a,"--beta",.5])
        matrix(out/"K_raw.tsv",name)
    c1,c2 = [read(DATA/(s+".tsv")) for s in ("C_expected","C_changed")]
    for i,j in [(0,0),(1,4),(2,4),(4,5)]:
        assert abs(float(c1[i][c1[j]["ID"]])-float(c2[i][c1[j]["ID"]]))>1e-12
    # A perturbation confined to the off-diagonal can have zero projection on
    # one particular row. Compare whole implicated blocks.
    refs = read(DATA/"pheno.tsv")
    ref = read(DATA/"reml.tsv")
    fx = ["--covar","age","--factor","herd"]
    out = cli("fit","fixed",fit+layers(base=kernels)+weights+fx)
    pred = read(out/"predictions.tsv")
    assert [p["id"] for p in pred]==[p["ID"] for p in refs]
    assert [p["id"] for p in pred if p["role"]=="candidate"]==["021","022","023","024"]
    compare(pred,ref,["u","yhat"],"id","REML",1e-5,1e-5)
    params = {r["term"]:float(r["value"]) for r in read(DATA/"parameters.tsv")}
    model = read(out/"model.tsv")[0]
    assert model["status"]=="interior"
    objective=float(read(DATA/"likelihood.tsv")[0]["profile_objective"])
    assert abs(float(model["profile_objective"])-objective)<1e-8
    for k in ("Vu","Ve","lambda"):
        err = abs(float(model[k])-params[k])
        assert err<2e-4,(k,err)
        print(f"{k} error {err:.9g}")
    for row,term in zip(read(out/"fixed.tsv"),["intercept","age","herdB"]):
        assert abs(float(row["estimate"])-params[term])<1e-5
    # Independently permute each matrix on both axes and phenotype order.
    for i,k in enumerate("AGO",1):
        rows = read(DATA/(k+".tsv"));cols = list(rows[0])[1:]
        write(tmp/(k+".tsv"),rows[i:]+rows[:i],["ID"]+cols[-i:]+cols[:-i])
    shuffled=refs[11:]+refs[:11];path=write(tmp/"ph.tsv",shuffled)
    out=cli("fit","permuted",fit[:-1]+[path]+layers(base=tmp)+weights+fx)
    pred=read(out/"predictions.tsv")
    assert [p["id"] for p in pred]==[p["ID"] for p in shuffled]
    compare(pred,ref,["u","yhat"],"id","permuted",1e-5,1e-5)
    ext=["--folds-file",DATA/"folds.tsv","--metric","rmse"]
    out=cli("fit","tuning",fit+layers()+ext)
    trace=read(out/"search.tsv")
    grid=[r for r in trace if r["stage"]=="grid"]
    assert len(grid)==15
    worst=(0,"")
    for r,s in zip(grid,read(DATA/"grid_scores.tsv")):
        assert float(r["alpha"])==float(s["V1"]) and float(r["beta"])==float(s["V2"])
        for j in range(1,5):
            err=abs(float(r[f"r1f{j}"])-float(s[f"V{j+2}"]))
            assert err<1e-4,(r,s,err)
            worst=max(worst,(err,f'{r["alpha"]},{r["beta"]} fold {j}'))
    edge=next(r for r in grid if float(r["alpha"])==.5 and float(r["beta"])==0)
    direct=read(DATA/"boundary.tsv")[0]
    assert abs(float(edge["r1f2"])-float(direct["rmse"]))<1e-8
    print("grid fold RMSE max absolute error",*worst)
    assert min(float(r["mean"]) for r in trace if r["valid"]=="1")==float(read(out/"tuning.tsv")[0]["selection_score"])
    # The optimal reference kernel can become indefinite for a new individual.
    # Full-cohort checks may use its covariances but never its phenotype.
    fallback = tmp / "fallback_input"
    fallback.mkdir()
    ids = [f"{i:03}" for i in range(1, 9)]
    train_y = [-1, 1, -2, 2]
    valid_y = [1.25 * .25 * y / (2 + 1e-6) for y in train_y[:3]]
    ph_rows = [{"ID": id_, "trait": y, "year": 2020 if i < 4 else 2021}
               for i, (id_, y) in enumerate(zip(ids, train_y + valid_y + ["NA"]))]
    ph_file = write(fallback / "pheno.tsv", ph_rows)
    for layer in "AGO":
        rows = []
        for i, id_ in enumerate(ids):
            row = {"ID": id_}
            for j, other in enumerate(ids):
                val = float(i == j)
                if layer == "A" and i == j == 7:
                    val = 10
                if layer != "A" and min(i, j) < 3 and abs(i - j) == 4:
                    val = .25
                row[other] = val
            rows.append(row)
        write(fallback / (layer + ".tsv"), rows)
    args = layers(base=fallback) + ["--trait", "trait", "--pheno", ph_file,
                                    "--year", 2020, "--metric", "rmse"]
    out = cli("fit", "full_fallback", args)
    threaded = cli("fit", "threaded_full_fallback", args + ["--parallel","weights","--threads",4])
    for file in ("search.tsv","predictions.tsv","tuning.tsv"):
        assert (out/file).read_bytes()==(threaded/file).read_bytes()
    trials = read(out / "search.tsv")
    winner, = [r for r in trials if r["reference_best"] == "1"]
    assert winner["full_status"] == "rejected" and "indefinite" in winner["full_reason"]
    assert abs(float(winner["mean"])) < 1e-10
    chosen, = [r for r in trials if r["full_status"] == "selected"]
    summary, = read(out / "tuning.tsv")
    assert float(chosen["mean"]) == float(summary["selection_score"])
    assert float(chosen["mean"]) > float(summary["reference_best_score"])
    assert int(summary["full_rejections"]) == sum(r["full_status"] == "rejected" for r in trials)
    assert all(float(r["mean"]) >= float(chosen["mean"]) - 1e-12 for r in trials
               if r["valid"] == "1" and r["full_status"] != "rejected")
    predicted = read(out / "predictions.tsv")
    assert predicted[-1]["role"] == "candidate" and math.isfinite(float(predicted[-1]["u"]))
    # The same reference-best point needs projection only on the full cohort.
    projected = cli("fit", "full_repaired", args + ["--pd-policy", "nearPD"])
    projected_best, = [r for r in read(projected/"search.tsv") if r["reference_best"]=="1"]
    assert projected_best["full_status"]=="selected" and projected_best["nearpd_blocks"]=="0"
    assert projected_best["full_nearpd_blocks"]=="1" and float(projected_best["full_max_relative_normF"])>0
    # When all reference-eligible candidates fail, publish the complete trace.
    for layer in "AGO":
        path = fallback / (layer + ".tsv")
        rows = read(path)
        rows[-1][ids[-1]] = -2
        write(path, rows)
    failed = cli("fit", "all_full_invalid", args, "no reference-eligible weights")
    assert (failed / "failure.txt").is_file()
    assert all(r["full_status"] == "rejected" for r in read(failed / "search.tsv")
               if r["valid"] == "1")
    random=cli("fit","random",fit+layers()+["--strict-simplex"])
    replay=cli("fit","replay",fit+layers()+["--folds-file",random/"folds.tsv","--strict-simplex"])
    assert (random/"search.tsv").read_bytes()==(replay/"search.tsv").read_bytes()
    temporal=cli("fit","time",fit+layers()+["--year","2021","--metric","auc","--score-col","binary"])
    fc=read(temporal/"fold_counts.tsv")[0]
    assert (fc["training"],fc["validation"])==("12","8")
    cli("fit","Bsearch",fit+layers(o="Ob")+ext)
    cli("fit","conflict",fit+layers()+ext+["--year",2021],"mutually exclusive")
    cli("fit","rank",fit+layers()+weights+["--covar","age,age"],"duplicate")
    cli("fit","rank2",fit+layers()+weights+["--factor","ID"],"rank")
    cli("fit","degenerate",fit+layers(o="Ob")+["--folds-file",DATA/"folds.tsv","--memory-mb",.001],"working set")
    bad=[r.copy() for r in refs];bad[0]["ID"]="999"
    cli("fit","outside",fit[:-1]+[write(tmp/"badph.tsv",bad)]+layers()+weights,"outside support")
    rows=read(DATA/"A.tsv");rows[0]["002"]="10";write(tmp/"bad.tsv",rows)
    cli("fuse","asymmetric",["--A",tmp/"bad.tsv"]+layers()[2:]+weights,"asymmetric")
    rows[1]["001"]="10";write(tmp/"bad.tsv",rows)
    cli("fuse","indefinite",["--A",tmp/"bad.tsv"]+layers()[2:]+["--alpha",0,"--beta",0],"indefinite")
    rows[0]["ID"]=rows[1]["ID"];write(tmp/"bad.tsv",rows)
    cli("fuse","duplicate",["--A",tmp/"bad.tsv"]+layers()[2:]+weights,"duplicate")
    rows=read(DATA/"O.tsv")
    for r in rows:
        r["999"]=r.pop("001")
        if r["ID"]=="001": r["ID"]="999"
    write(tmp/"other.tsv",rows)
    cli("fuse","coverage",layers()[:-1]+[tmp/"other.tsv"]+weights,"O_only=1")
    bad=[r.copy() for r in refs]
    for r in bad: r["trait"]=r["trait"] if int(r["ID"])>=9 else "NA"
    cli("fit","deg",fit[:-1]+[write(tmp/"deg.tsv",bad)]+layers(o="Ob")+ext,"coverage degenerates")
    for r in refs:
        r["binary"]="1"
    cli("fit","undefined",fit[:-1]+[write(tmp/"constant.tsv",refs)]+layers()+["--metric","auc","--score-col","binary"],"single validation class")


if __name__=="__main__":
    with tempfile.TemporaryDirectory() as tmp:
        run(Path(tmp))
