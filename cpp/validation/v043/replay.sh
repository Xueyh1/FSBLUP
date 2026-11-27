#!/bin/bash
set -eu
exe=$(realpath "$1")
root=$(realpath -m "$2")
mkdir "$root"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for name in replay_interior_modes replay_identifiability replay_temporal_scale search_geometry feature_panel connected_c; do
 status=0
 python3 "validation/v043/replay/$name.py" "$exe" "$root/$name" >"$root/$name.log" 2>&1 || status=$?
 echo "$name $status" >> "$root/exits.txt"
done
