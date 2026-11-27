#!/bin/bash
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exe=$(realpath "$1")
out=$(realpath -m "$2")
mkdir "$out"
scripts=$PWD/validation/v042/replay
for case in reproduce_reml reproduce_design replay_cv_units reproduce_operators reproduce_coverage; do
 args=("$exe")
 case "$case" in
 reproduce_design) ;;
 reproduce_reml) args+=("$out");;
 *) args+=("$out/$case");;
 esac
 status=0
 TMPDIR="$out" python3 "$scripts/$case.py" "${args[@]}" >"$out/$case.log" 2>&1 || status=$?
 echo "$case $status" >> "$out/exits.txt"
done
