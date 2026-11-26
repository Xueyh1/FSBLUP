"""Check phenotype-unit restoration of fixed and independently estimated models."""
import csv,math,subprocess,sys
from pathlib import Path
binary, data, out=map(Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=False)
def rows(p):
    with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))
source=rows(data/'pheno.tsv');checks=[]
for method in ('fixed','ai-reml','em-reml'):
 for exponent in (0,-500,500,-665,665):
    values=[r.copy() for r in source]
    for r in values:
        if r['trait']!='NA':r['trait']=repr(math.ldexp(round(float(r['trait'])*32)/32,exponent))
    file=out/'input.tsv'
    with file.open('w') as f:
        w=csv.DictWriter(f,values[0].keys(),delimiter='\t');w.writeheader();w.writerows(values)
    d=out/f'{method}-{exponent}';args=[binary.resolve(),'fit','--pheno',file,'--id','ID','--trait','trait','--covar','age','--factor','herd','--varcomp',method,'--alpha','.2','--beta','.5','--out',d,'--log-level','off']
    for k in 'AGO':args+=['--'+k,data/(k+'.tsv')]
    if method=='fixed':args+=['--lambda','.7']
    r=subprocess.run(args,capture_output=True,text=True)
    if abs(exponent)>500:
        assert r.returncode and 'representable range' in r.stderr and not d.exists()
        checks.append([method,exponent,'rejected',None,None])
        continue
    assert not r.returncode,r.stderr
    m=rows(d/'model.tsv')[0];u=rows(d/'predictions.tsv')
    vu=math.ldexp(float(m['Vu']),-2*exponent)
    pred=[math.ldexp(float(x['yhat']),-exponent) for x in u]
    if exponent==0:reference=vu,pred
    du=max(abs(v-w) for v,w in zip(pred,reference[1]));dv=abs(vu-reference[0])
    assert du<1e-8 and dv<1e-8
    checks.append([method,exponent,'ok',dv,du])
with (out/'units.tsv').open('w') as f:
    w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['method','exponent','status','scaled_Vu_error','scaled_yhat_error']);w.writerows(checks)
