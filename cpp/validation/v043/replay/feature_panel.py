"""Reproduce unplanned feature workspace for a wide fixed-effect panel. Usage: python feature_panel.py EXE NEW_FOLDER"""
import argparse, csv, math, os, subprocess
from pathlib import Path
import numpy as np

def write(path, header, records):
    with path.open("w") as f:
        out=csv.writer(f, delimiter="\t",lineterminator="\n")
        out.writerow(header)
        out.writerows(records)

def first(path):
    with path.open() as f:
        return next(csv.DictReader(f, delimiter="\t"))

def run(cmd):
    return subprocess.run(list(map(str,cmd)),capture_output=True,text=True,env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1"))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary",type=Path); parser.add_argument("output",type=Path)
    args=parser.parse_args(); exe=args.binary.resolve(); out=args.output.resolve()
    out.mkdir()
    n,m,max_p=80,60000,60
    rng=np.random.default_rng(974)
    ids=[f'i{i}' for i in range(n)]
    write(out/'ped',['ID','Sire','Dam'],((i,0,0) for i in ids))
    write(out/'omics',['ID','m'],((i,j%5) for j,i in enumerate(ids)))
    # Stream one sample's dosage at a time, leaving stochastic K nontrivial.
    with (out/'snp').open('w') as f:
        f.write('ID\t'+'\t'.join(f's{i}' for i in range(m))+'\n')
        for i in ids:
            chars=np.frombuffer(b'012',np.uint8)[rng.integers(0,3,size=m)].tobytes()
            f.write(i+'\t'+'\t'.join(chars.decode())+'\n')
    names=[f'x{i}' for i in range(1,max_p)]
    xx=rng.normal(size=(n,max_p-1)); y=rng.normal(size=n)
    write(out/'pheno',['ID','trait',*names],([key,y[i],*xx[i]] for i,key in enumerate(ids)))
    p=run([exe,'prepare','--ped',out/'ped','--snp',out/'snp','--snp-format','2bit','--omics',out/'omics','--out',out/'cache'])
    if p.returncode: raise RuntimeError(p.stderr)
    observations=[]
    for pcols,complete in ((2,False),(max_p,False),(max_p,True)):
        dest=out/(f'completed{pcols}' if complete else f'fit{pcols}')
        # The PCG iteration cap keeps this test short. Both requests pass the
        # workspace checks and enter the same algorithm, including K times Qx.
        settings=['--pcg-rtol','.01','--reml-tol','.1','--mc-tol','10','--slq-tol','1'] if complete else ['--pcg-maxiter',1]
        p=run([exe,'fit','--cache',out/'cache','--cache-backend','operator','--cache-storage','stream','--pheno',out/'pheno','--trait','trait','--covar',','.join(names[:pcols-1]),'--alpha',0,'--beta',1,'--varcomp','ai-reml','--trace','hutchinson','--probe-batch',1,'--probes',2,'--max-probes',2,'--slq-depth',2,*settings,'--memory-mb',12,'--log-level','error','--out',dest])
        runtime=first(dest/'runtime.tsv') if (dest/'runtime.tsv').exists() else {}
        model=first(dest/'model.tsv') if (dest/'model.tsv').exists() else {}
        observations.append(['default_pcg_limit' if complete else 'early_stop',pcols,m,12,len(ids),1,pcols*m*8/2**20,runtime.get('peak_rss_kib',''),p.returncode,model.get('status',''),p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ''])
    write(out/'observed.tsv',['case','fixed_columns','features','algorithm_cap_MiB','train','planned_rhs_panel','coefficient_matrix_MiB','peak_RSS_KiB','exit','model_status','reason'],observations)
    print((out/'observed.tsv').read_text())
if __name__=='__main__': main()
