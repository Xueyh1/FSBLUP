#!/bin/bash
# Run in a Slurm allocation from cpp. Snapshot source so later edits do not affect a batch.
set -eu
root=$(realpath -m "$1"); mkdir -p "$root/source"
git archive "$2" cpp | tar -x -C "$root/source"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for spec in release asan tsan mem; do
 case "$spec" in
 release) cfg=Release; cc=g++; flags='-O3 -DNDEBUG';;
 asan) cfg=Debug; cc=clang++; flags='-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer';;
 tsan) cfg=Debug; cc=clang++; flags='-O1 -g -fsanitize=thread -fno-omit-frame-pointer';;
 mem) cfg=Debug; cc=g++; flags='-O1 -g -gdwarf-4';;
 esac
 b="$root/$spec"; cmake -S "$root/source/cpp" -B "$b" -DCMAKE_BUILD_TYPE="$cfg" -DARMADILLO_INCLUDE_DIR="$PWD/.deps/include" -DCMAKE_CXX_COMPILER="$cc" "-DCMAKE_CXX_FLAGS_${cfg^^}=$flags" >"$root/$spec-config.log"
 cmake --build "$b" -j4 >"$root/$spec-build.log" 2>&1
 if [[ $spec == mem ]]; then python3 tools/debug/memcheck_ctest.py "$b" "$root/valgrind" >"$root/$spec-tests.log" 2>&1
 else ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 TSAN_OPTIONS=halt_on_error=1 ctest --test-dir "$b" -j2 --output-on-failure >"$root/$spec-tests.log" 2>&1
 fi
done
