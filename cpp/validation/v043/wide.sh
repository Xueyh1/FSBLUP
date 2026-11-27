#!/bin/bash
# Submit with sbatch -n1 -c16 --mem=68G. Input layout matches the v042 wide run.
set -eu
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=1
exe=$(realpath "${1:?packaged fsblup}")
d=$(realpath "${2:?wide input directory}")
/usr/bin/time -v "$exe" fit --cache "$d/cache" --pheno "$d/large.tsv" --trait trait \
 --folds-file "$d/large-folds.tsv" --metric rmse --strict-simplex --search-policy legacy \
 --cache-backend operator --cache-storage resident --threads 8 --memory-mb 65536 \
 --varcomp ai-reml --trace hutchinson --slq-reorth local --probe-solver lanczos \
 --slq-depth 18 --slq-batch 4 --probes 4 --max-probes 8 --retry-probes 32 \
 --retry-policy rounds --preconditioner nystrom --log-level debug --log-every 25 --out "${3:?new output directory}"
