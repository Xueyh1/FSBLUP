#!/usr/bin/env bash
set -eu
# Run from cpp. The control programs are independent of the FSBLUP binary.
out=${1:?supply a new directory for the detector logs}
mkdir "$out"
clang++ -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined tools/debug/control.cpp -pthread -o "$out/control-san"
clang++ -O1 -g -fsanitize=thread tools/debug/control.cpp -pthread -o "$out/control-tsan"
g++ -O1 -g -gdwarf-4 tools/debug/control.cpp -pthread -o "$out/control-mem"
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 TSAN_OPTIONS=halt_on_error=1
printf 'detector\tcase\texpected_exit\tactual_exit\n' > "$out/summary.tsv"
for entry in san:clean:0 san:leak:1 tsan:safe:0 tsan:race:66 mem:clean:0 mem:leak:99; do
    IFS=: read -r tool which expected <<< "$entry"
    prefix=()
    if [[ $tool == mem ]]; then
        prefix=(valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --errors-for-leak-kinds=definite,indirect --track-origins=yes --error-exitcode=99)
    fi
    status=0
    "${prefix[@]}" "$out/control-$tool" "$which" >"$out/$tool-$which.log" 2>&1 || status=$?
    printf '%s\t%s\t%s\t%s\n' "$tool" "$which" "$expected" "$status" >> "$out/summary.tsv"
    test "$status" = "$expected"
done
grep -q '79 byte(s) leaked' "$out/san-leak.log"
grep -q 'ThreadSanitizer: data race' "$out/tsan-race.log"
grep -q 'definitely lost: 79 bytes' "$out/mem-leak.log"
