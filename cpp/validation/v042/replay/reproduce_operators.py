"""Reproduce common-coverage stream planning and pedigree preparation timings on the audited 0.4.1."""
import argparse
import csv
from pathlib import Path
import random
import subprocess
import time


def tsv(path, header, data):
    with path.open('w', encoding='utf-8', newline='') as out:
        w = csv.writer(out, delimiter='\t', lineterminator='\n')
        w.writerow(header)
        w.writerows(data)


def run(bin_path, args):
    start = time.perf_counter()
    p = subprocess.run([bin_path, *map(str, args)], text=True, capture_output=True)
    return p.returncode, time.perf_counter() - start, p.stderr.strip()


def prepare(bin_path, d):
    return run(bin_path, ['prepare', '--ped', d/'ped', '--snp', d/'snp', '--omics', d/'omics',
                          '--snp-format','2bit','--log-level','off','--out',d/'cache'])


def stream_case(bin_path, work):
    d = work/'stream'; d.mkdir()
    m, n = 4096, 40
    rng = random.Random(174)
    ids = [f'i{i}' for i in range(n)]
    tsv(d/'ped',['ID','Sire','Dam'], ((key,0,0) for key in ids))
    tsv(d/'snp',['ID',*[f'v{i}' for i in range(m)]], ([key,*[rng.randrange(3) for _ in range(m)]] for key in ids))
    tsv(d/'omics',['ID','v0','v1'], ((key,rng.gauss(0,1),rng.gauss(0,1)) for key in ids))
    tsv(d/'pheno',['ID','trait'], ((key,rng.gauss(0,1) if i<32 else 'NA') for i,key in enumerate(ids)))
    tsv(d/'folds',['repeat','fold','id'], ((1,i%2+1,ids[i]) for i in range(32)))
    code, _, err = prepare(bin_path, d)
    if code:
        raise RuntimeError(err)
    result=[]
    for mode, threads, cap in [('blas',1,32),('weights',1,32),('weights',2,32),('weights',1,100)]:
        dest=d/f'{mode}_{threads}_{cap}'
        args=['fit','--cache',d/'cache','--pheno',d/'pheno','--trait','trait','--varcomp','fixed','--lambda',1,
              '--strict-simplex','--cache-backend','operator','--cache-storage','stream','--folds-file',d/'folds',
              '--metric','rmse','--memory-mb',cap,'--parallel',mode,'--threads',threads,'--log-level','off','--out',dest]
        code, elapsed, stderr=run(bin_path, args)
        rss = chosen = ''
        if code==0:
            with (dest/'tuning.tsv').open() as f:
                row=next(csv.DictReader(f,delimiter='\t'))
                chosen=row['selected_alpha']+','+row['selected_beta']
            with (dest/'runtime.tsv').open() as f:
                row=next(csv.DictReader(f,delimiter='\t'))
                rss=row.get('peak_rss_kib','')
        elif 'insufficient for one candidate' in stderr:
            stderr='--memory-mb insufficient for one candidate workspace'
        result.append((mode,threads,cap,code,round(elapsed,5),rss,chosen,stderr))
    tsv(work/'planning.tsv',['parallel','threads','cap_MiB','exit','wall_s','peak_RSS_KiB','alpha_beta','error'],result)


def pedigree_cases(bin_path, work):
    result=[]
    for n in (500,1000,2000,4000):
        for layout in ('shallow','ancestral'):
            d=work/f'{layout}_{n}'; d.mkdir()
            tsv(d/'ped',['ID','Sire','Dam'], ((f'i{i}','0' if i<2 else 'i0','0' if i<2 else ('i1' if layout=='shallow' else f'i{i-1}')) for i in range(n)))
            for name in ('snp','omics'):
                tsv(d/name,['ID','v1','v2'],((f'i{i}',i%3,(i//2)%3) for i in range(n)))
            code, elapsed, stderr = prepare(bin_path,d)
            result.append((layout,n,code,round(elapsed,5),stderr))
    tsv(work/'pedigree.tsv',['pedigree','n','exit','wall_s','error'],result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('binary',type=Path)
    parser.add_argument('new_output',type=Path)
    args=parser.parse_args()
    args.new_output.mkdir()
    stream_case(args.binary.resolve(),args.new_output)
    pedigree_cases(args.binary.resolve(),args.new_output)
