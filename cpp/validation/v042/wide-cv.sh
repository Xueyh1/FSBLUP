#!/bin/bash
set -eu
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=1
exe=$(realpath "${1:?fsblup binary}")
d=$(realpath "${2:?directory containing cache, large.tsv and large-folds.tsv}")
out=${3:?new output directory}
/usr/bin/time -v "$exe" fit --cache "$d/cache" --pheno "$d/large.tsv" --trait trait --folds-file "$d/large-folds.tsv" --metric rmse --strict-simplex --cache-backend operator --cache-storage resident --threads 8 --memory-mb 65536 --varcomp ai-reml --trace hutchinson --slq-reorth local --probe-solver lanczos --slq-depth 18 --slq-batch 4 --probes 4 --max-probes 8 --retry-probes 32 --score-policy relative --preconditioner nystrom --log-level debug --log-every 25 --out "$out"
