#!/usr/bin/env bash
# Run in cpp in a Slurm allocation. Use a new output directory.
set -eu
exe=$(realpath "${1:?fsblup executable}")
old=$(realpath "${2:?0.4.0 executable}")
out=$(realpath -m "${3:?new result directory}")
mkdir "$out"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 tests/reml_invariance.py "$old" --observe > "$out/before.tsv"
python3 tests/reml_invariance.py "$exe" > "$out/after.tsv"
python3 tools/v041_bench.py "$exe" "$out/cache" --baseline "$old"
python3 tools/pcg_accuracy.py "$exe" "$out/seeds" lanczos
if [[ $# == 4 ]]; then
  Rscript --vanilla tools/nearpd_official.R "$4" "$out/R"
  python3 tools/pcg_official.py "$exe" "$4" "$out/R" "$out/official"
  python3 tools/nearpd_compare.py "$out/R" "$out/official/pcg-ai-reml" > "$out/official.json"
fi
