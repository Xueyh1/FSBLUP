"""Recheck validation-only RMS against fold memberships through both cached backends."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from v043_panel import inputs,read,tab,save
p=argparse.ArgumentParser(description=__doc__);p.add_argument('exe',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
d=a.out;d.mkdir();src=d/'src';inputs(src,64,24,6);exe=a.exe.resolve()
subprocess.run([exe,'prepare','--ped',src/'ped','--snp',src/'snp','--omics',src/'omics','--snp-format','2bit','--out',src/'cache'],check=True)
data=read(src/'y');rng=np.random.default_rng(742);ref=[i for i,r in enumerate(data) if r['trait']!='NA'];valid=ref[-11:]
report=[]
for split in ['year','ngen','repeated']:
    for backend in ['dense','operator']:
        for par in ['weights','blas']:
            pair=[]
            for tag in [0,400]:
                truth=np.array([float(r['trait']) if r['trait']!='NA' else 0 for r in data])*.27
                if split!='repeated':truth[ref[:-11]]=tag
                path=d/f'y-{split}-{tag}'
                tab(path,['ID','trait','z','year','ngen'],([r['ID'],r['trait'],truth[i],int(i in valid),int(i in valid)] for i,r in enumerate(data)))
                out=d/f'{split}-{backend}-{par}-{tag}'
                opts=['--folds',3,'--repeats',2,'--seed',111] if split=='repeated' else ['--'+split,0]
                cmd=[exe,'fit','--cache',src/'cache','--cache-backend',backend,'--pheno',path,'--trait','trait','--score-col','z','--metric','rmse','--strict-simplex','--parallel',par,'--threads',2,'--varcomp','fixed','--lambda',1,*opts,'--out',out]
                r=subprocess.run(list(map(str,cmd)),capture_output=True,text=True);assert r.returncode==0,r.stderr
                t=read(out/'tuning.tsv')[0];folds=read(out/'folds.tsv')
                mapping={r['ID']:truth[i] for i,r in enumerate(data)}
                groups={}
                for f in folds:
                    if split=='repeated' or f['fold']=='1':
                        groups.setdefault((f['repeat'],f['fold']),[]).append(mapping[f['id']])
                scale=float(np.sqrt(np.mean([np.mean(np.square(z)) for z in groups.values()])))
                assert abs(float(t['score_scale'])-scale)<1e-12,(t,groups)
                pair.append((read(out/'search.tsv'),t))
            assert pair[0]==pair[1]
            report.append(dict(split=split,backend=backend,parallel=par,scale=scale,visited=len(pair[0][0]),alpha=t['selected_alpha'],beta=t['selected_beta']))
save(d/'summary.tsv',report)
