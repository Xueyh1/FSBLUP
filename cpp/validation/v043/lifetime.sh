#!/bin/bash
set -eu
root=$(realpath "${1:?completed batch build}")
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$root/release/repeat_test" 1000 >"$root/lifetime.tsv"
bash tools/debug/controls.sh "$root/controls"
valgrind --leak-check=full --errors-for-leak-kinds=definite,indirect --error-exitcode=99 "$root/mem/repeat_test" 8 >"$root/valgrind-lifetime.tsv" 2>"$root/valgrind-lifetime.log"
