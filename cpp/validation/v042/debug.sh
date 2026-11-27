#!/usr/bin/env bash
# Run from cpp in a Slurm allocation; the result directory must be new.
set -eu
out=$(realpath -m "${1:?new results directory}")
mkdir "$out"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for detector in asan tsan mem; do
  cxx=clang++
  flags='-O1 -g -fno-omit-frame-pointer'
  case "$detector" in
    asan) flags+=' -fsanitize=address,undefined -fno-sanitize-recover=all';;
    tsan) flags+=' -fsanitize=thread';;
    mem) cxx=g++; flags+=' -gdwarf-4';;
  esac
  build="$out/build-$detector"
  cmake -S . -B "$build" -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug \
    -DARMADILLO_INCLUDE_DIR="${ARMADILLO_INCLUDE_DIR:-$PWD/.deps/include}" \
    -DCMAKE_CXX_COMPILER="$cxx" -DCMAKE_CXX_FLAGS_DEBUG="$flags"
  cmake --build "$build" -j4
  if [[ $detector == mem ]]; then
    python3 tools/debug/memcheck_ctest.py "$build" "$out/mem-suite"
    valgrind --leak-check=full --errors-for-leak-kinds=definite,indirect \
      --error-exitcode=99 "$build/repeat_test" 8 2>"$out/mem-lifetime.log"
  else
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
      TSAN_OPTIONS=halt_on_error=1 ctest --test-dir "$build" -j2 --output-on-failure
  fi
done
bash tools/debug/controls.sh "$out/controls"
