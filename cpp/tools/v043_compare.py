"""Independent grid reference, budget/accuracy replays and paired elapsed time."""
import argparse,csv,time,subprocess,itertools,statistics,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation/v043/replay'))
from search_geometry import fold_predictions
from v043_panel import read,save,inputs

def fit(exe,opts,out):
    cmd=list(map(str,[exe,'fit',*opts,'--out',out]))
    t=time.monotonic();p=subprocess.run(cmd,capture_output=True,text=True);s=time.monotonic()-t
    out.with_suffix('.stderr').write_text(p.stderr)
    return p.returncode,s
def main():
    p=argparse.ArgumentParser();p.add_argument('old',type=Path);p.add_argument('exe',type=Path);p.add_argument('before',type=Path);p.add_argument('after',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();a.out.mkdir()
    errors=[]
    for version,root in [('before',a.before),('after',a.after)]:
        folder=root/'replay_interior_modes';oracle_rows=read(folder/'summary.tsv')
        ref=max((r for r in oracle_rows if r['method']=='scipy-profile'),key=lambda r:float(r['contrast_loglik']))
        g,e=float(ref['Vu']),float(ref['Ve']);kr=read(folder/'K.tsv');ids=[r['id'] for r in kr]
        K=np.array([[float(r[key]) for key in ids] for r in kr])+1e-6*np.eye(len(ids))
        y=np.array([float(r['y']) for r in read(folder/'y.tsv') if r['y']!='NA'])
        h=np.linalg.solve(g*K[:len(y),:len(y)]+e*np.eye(len(y)),np.column_stack((y,np.ones(len(y)))))
        b=h[:,0].sum()/h[:,1].sum();expected=g*K[:,:len(y)]@(h[:,0]-b*h[:,1])
        for row in oracle_rows:
            model=folder/row['method']
            if not model.is_dir() or row['exit']!='0':continue
            audit=read(model/'pcg_summary.tsv')[0];steps=read(model/'reml_trace.tsv')
            for j,pred in enumerate(read(model/'predictions.tsv')):
                assert ids[j]==pred['id']
                du=float(pred['u'])-expected[j];dy=float(pred['yhat'])-b-expected[j]
                errors.append(dict(version=version,method=row['method'],id=ids[j],role=pred['role'],du=du,dyhat=dy,
                    dlik=float(row['contrast_loglik'])-float(ref['contrast_loglik']),rhs=audit['rhs'],matvec=audit['matvec'],probes=audit['probes'],
                    extra_rhs=sum(int(r.get('comparison_rhs',0)) for r in steps),extra_matvec=sum(int(r['face_matvec']) for r in steps),
                    compare_s=audit.get('likelihood_comparison_s','NA')))
                if version=='after':assert abs(du)<1e-5 and abs(dy)<1e-5 and abs(errors[-1]['dlik'])<1e-7
    save(a.out/'per-id.tsv',errors)
    geo=a.after/'search_geometry'
    keys=lambda p:np.array([[float(v) for v in list(r.values())[1:]] for r in read(p)])
    layers=[keys(geo/(k+'.tsv')) for k in 'AGO']; folds=np.array([int(r['fold']) for r in read(geo/'folds.tsv')])
    cases=[]
    for label in ['edge','interior']:
        d=read(geo/(label+'.tsv'));y=np.array([float(r['Y']) for r in d]);target=np.array([float(r['score']) for r in d])
        scored=[]
        for i in range(33):
            for j in range(33-i):
                u,_=fold_predictions(*layers,y,folds,i/32,j/32)
                val=np.mean([np.sqrt(np.mean((u[folds==k]-target[folds==k])**2)) for k in set(folds)])
                scored.append((val,i/32,j/32))
        val,aa,bb=min(scored)
        for solver in ('direct','pcg'):
            for version,base in [('old',a.before),('new',a.after)]:
                out=base/'search_geometry'/(label+'-'+solver);t=read(out/'tuning.tsv')[0];s=read(out/'search.tsv')
                cases.append(dict(label=label,solver=solver,version=version,grid_points=len(scored),grid_alpha=aa,grid_beta=bb,oracle=val,alpha=t['selected_alpha'],beta=t['selected_beta'],rmse=t['selection_score'],visited=len(s),fold_attempts=t['fold_attempts']))
                if version=='new': assert abs(float(t['selection_score'])-val)<1e-8 and (float(t['selected_alpha']),float(t['selected_beta']))==(aa,bb)
    save(a.out/'grid.tsv',cases)
    d=a.before/'feature_panel';opts=['--mode-policy','local','--cache',d/'cache','--cache-backend','operator','--cache-storage','stream','--pheno',d/'pheno','--trait','trait','--covar',','.join('x'+str(i) for i in range(1,60)),'--alpha',0,'--beta',1,'--trace','hutchinson','--probe-batch',1,'--probes',2,'--max-probes',2,'--slq-depth',2,'--pcg-rtol','.01','--reml-tol','.1','--mc-tol','10','--slq-tol',1]
    res=[]
    for cap in (2,12,128):
        out=a.out/f'cap{cap}';rc,elapsed=fit(a.exe,opts+['--memory-mb',cap],out);u=(read(out/'cache_usage.tsv') or [{}])[0];s=(read(out/'pcg_summary.tsv') or [{}])[0];m=(read(out/'model.tsv') or [{}])[0]
        res.append(dict(cap=cap,exit=rc,wall=elapsed,actual_panel=u.get('actual_panel'),rhs=u.get('requested_rhs'),workspace_B=u.get('panel_workspace_bytes'),Vu=m.get('Vu'),Ve=m.get('Ve'),residual=s.get('max_true_residual_over_tolerance')))
    save(a.out/'workspace.tsv',res)
    def aligned(d,e):
        p,q=read(d/'predictions.tsv'),read(e/'predictions.tsv');assert p and [r['id'] for r in p]==[r['id'] for r in q]
        return dict(max_du=max(abs(float(r['u'])-float(s['u'])) for r,s in zip(p,q)),max_dy=max(abs(float(r['yhat'])-float(s['yhat'])) for r,s in zip(p,q)))
    assert res[0]['exit'] and not res[1]['exit'] and not res[2]['exit'],res
    save(a.out/'workspace-pred.tsv',[aligned(a.out/'cap12',a.out/'cap128')])
    d=a.after/'connected_c';cs=[]
    for name,active in [('unrelated',48),('connected',512),('one_disconnected',511)]:
        op=d/(name+'-96');den=a.out/(name+'-dense')
        rc,_=fit(a.exe,['--cache',d/(name+'-cache'),'--cache-backend','dense','--pheno',d/'Y','--trait','y','--alpha','.2','--beta','.4','--varcomp','fixed','--lambda',1,'--pd-policy','nearPD','--memory-mb',128],den)
        assert not rc
        usage=read(op/'cache_usage.tsv')[0];assert int(usage['active_component_ids'])==active
        diff=aligned(op,den);assert max(diff.values())<1e-8
        cs.append(dict(case=name,active=active,backend=usage['certification_backend'],reason=usage['fallback_reason'],**diff))
    save(a.out/'C.tsv',cs)
    timings=[]
    for size in (64,512):
        src=a.out/f'perf{size}';inputs(src,size,96,8)
        subprocess.run([a.old,'prepare','--ped',src/'ped','--snp',src/'snp','--omics',src/'omics','--snp-format','2bit','--out',src/'cache'],check=True)
        for rep in range(4):
            for version,exe in ([('old',a.old),('new',a.exe)] if rep%2==0 else [('new',a.exe),('old',a.old)]):
                out=src/f'{version}-{rep}'
                rc,elapsed=fit(exe,['--cache',src/'cache','--cache-backend','operator','--pheno',src/'y','--trait','trait','--folds-file',src/'folds','--metric','rmse','--strict-simplex','--varcomp','fixed','--lambda',1,'--threads',1,'--memory-mb',64]+(['--search-policy','legacy'] if version=='new' else []),out);assert not rc
                if not rep:continue
                f=read(out/'cv_pcg_summary.tsv')+read(out/'pcg_summary.tsv');t=read(out/'tuning.tsv')[0]
                timings.append(dict(n=size,rep=rep,version=version,wall_s=elapsed,rss=read(out/'runtime.tsv')[0]['peak_rss_kib'],rhs=sum(int(v['rhs']) for v in f),matvec=sum(int(v['matvec']) for v in f),scored=t['candidates_scored'],**{k:r for k,r in read(out/'cache_usage.tsv')[0].items() if k.endswith('bytes')}))
    save(a.out/'paired.tsv',timings)
if __name__=='__main__':main()
