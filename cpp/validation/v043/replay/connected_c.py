"""Measure the change in connected-component certification cost in cached C mode."""
import argparse
import csv
from pathlib import Path
import random
import subprocess


def table(path, columns, values):
    with path.open('w', newline='') as out:
        writer = csv.writer(out, delimiter='\t', lineterminator='\n')
        writer.writerow(columns)
        writer.writerows(values)


def head(path):
    with path.open() as f:
        return next(csv.DictReader(f, delimiter='\t'))


def cmd(args):
    return subprocess.run([str(x) for x in args],text=True, capture_output=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exe', type=Path)
    parser.add_argument('new_dir', type=Path)
    args=parser.parse_args()
    exe=args.exe.resolve()
    folder=args.new_dir.resolve()
    folder.mkdir()
    n=512
    m=64
    ids=[f'i{i:04d}' for i in range(n)]
    rng=random.Random(940)
    table(folder/'S',['ID', *[f's{i}' for i in range(m)]],
          ([id_,*[rng.randrange(3) for i in range(m)]] for id_ in ids[2:50]))
    table(folder/'O',['ID','o1','o2'],
          ([id_,rng.gauss(0,1),rng.gauss(0,1)] for id_ in ids[2:18]))
    table(folder/'Y',['ID','y'],
          ([id_,rng.gauss(0,1) if i<60 else 'NA'] for i,id_ in enumerate(ids[:64])))
    results=[]
    for mode in ('unrelated','connected','one_disconnected'):
        fam=folder/(mode+'.tsv')
        table(fam,['ID','Sire','Dam'],([v,ids[0] if ((mode=='connected' and i>=1) or (mode=='one_disconnected' and i>=2)) else '0','0'] for i,v in enumerate(ids)))
        cache=folder/(mode+'-cache')
        prepar=cmd([exe,'prepare','--ped',fam,'--snp',folder/'S','--snp-format','2bit','--omics',folder/'O','--out',cache])
        if prepar.returncode:
            raise RuntimeError(prepar.stderr)
        for budget in (16,96):
            out=folder/f'{mode}-{budget}'
            fit=cmd([exe,'fit','--cache',cache,'--cache-backend','operator','--cache-storage','stream',
                     '--pheno',folder/'Y','--trait','y','--alpha','.2','--beta','.4',
                     '--varcomp','fixed','--lambda','1','--pd-policy','nearPD','--memory-mb',budget,'--out',out])
            expected_active={'unrelated':48, 'connected':512, 'one_disconnected':511}[mode]
            active=''
            status=''
            if fit.returncode==0:
                active=head(out/'cache_usage.tsv').get('active_component_ids','')
                status=head(out/'model.tsv')['status']
            else:
                status=fit.stderr.strip().splitlines()[-1]
            results.append([mode,n,len(ids[2:50]),len(ids[2:18]),budget,fit.returncode,expected_active,active,status])
    table(folder/'observed.tsv',['pedigree','cohort','genotyped','omics','cap_MiB','exit','pedigree_active','reported_active','result'],results)
    print((folder/'observed.tsv').read_text())
if __name__=='__main__':
    main()
