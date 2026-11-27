"""Compare cache reads with identical full G and the same model rows at two omics coverages."""
import csv
import math
from pathlib import Path
import random
import sys
from reproduce_operators import tsv, prepare, run

bin_path=Path(sys.argv[1]).resolve()
work=Path(sys.argv[2])
work.mkdir()
results=[]
for n in [256,4096]:
    for mode in ['A','B']:
        d=work/f'{mode}_{n}'; d.mkdir()
        rng=random.Random(174)
        ids=[f'i{i}' for i in range(n)]
        tsv(d/'ped',['ID','Sire','Dam'],((key,0,0) for key in ids))
        tsv(d/'snp',['ID',*[f'v{i}' for i in range(128)]],([key,*[rng.randrange(3) for _ in range(128)]] for key in ids))
        tsv(d/'omics',['ID','v1','v2'],((key,i%3,(i//2)%3) for i,key in enumerate(ids if mode=='A' else ids[:4])))
        tsv(d/'pheno',['ID','trait'],((ids[i],math.sin(i) if i<8 else 'NA') for i in range(12)))
        code,_,err=prepare(bin_path,d)
        if code: raise RuntimeError(err)
        out=d/'fit'
        code,time,err=run(bin_path,['fit','--cache',d/'cache','--cache-storage','stream','--cache-backend','operator','--pheno',d/'pheno',
                            '--trait','trait','--alpha',0,'--beta','.4','--varcomp','fixed','--lambda',1,'--threads',1,'--out',out])
        if code:
            raise RuntimeError(err)
        def row(name):
            with (out/name).open() as f:
                return next(csv.DictReader(f,delimiter='\t'))
        usage,pcg=row('cache_usage.tsv'),row('pcg_summary.tsv')
        results.append((mode,n,usage['feature_bytes'],usage['verified_cache_bytes'],pcg['matvec'],round(time,5)))
tsv(work/'coverage.tsv',['mode','n','feature_bytes','verified_cache_bytes','matvec','wall_s'],results)
