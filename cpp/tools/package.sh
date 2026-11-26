#!/usr/bin/env bash
set -euo pipefail
# Package a trusted local build on CentOS Stream 9 x86_64.
cd "$(dirname "$0")/.."
builddir=${1:-build}
dest=${2:-dist}
name=fsblup-0.3.0-linux-x86_64-glibc2.34
mkdir -p "$dest"
stage=$(mktemp -d "$dest/.package.XXXXXX")
trap 'rm -rf "$stage"' EXIT
root=$stage/$name
mkdir -p "$root"/{lib,licenses,source/cpp}
cmake --install "$builddir" --prefix "$root"
test "$(uname -m)" = x86_64
test "$(getconf GNU_LIBC_VERSION)" = 'glibc 2.34'
ldd "$root/bin/fsblup" > "$stage/linked"
while read -r lib arrow path rest; do
  case "$lib" in
    libopenblas*.so.*|libstdc++.so.*|libgcc_s.so.*|libgfortran.so.*|libquadmath.so.*)
      test "$arrow" = '=>'
      cp -L "$path" "$root/lib/$lib"
      ;;
  esac
done < "$stage/linked"
cp ../LICENSE.md "$root/LICENSE.md"
cp /usr/share/licenses/openblas/LICENSE "$root/licenses/OpenBLAS"
cp /usr/share/licenses/libgcc/COPYING* /usr/share/licenses/libquadmath/COPYING* "$root/licenses/"
cp licenses/Armadillo.txt "$root/licenses/"
cp README.md VALIDATION.md "$root/"
cp -r validation "$root/"
cp -r CMakeLists.txt src tools tests licenses validation README.md VALIDATION.md "$root/source/cpp/"
mkdir -p "$root/source/R"
cp ../R/FSBLUP.R "$root/source/R/"
cp ../LICENSE.md "$root/source/"
cp -r tests/data "$root/example"
(cd "$root" && ldd bin/fsblup | sed -E 's/ \(0x[0-9a-f]+\)//g' | sed "s|$PWD|.|g" > linked-libraries.txt)
tar --exclude=__pycache__ -C "$stage" -czf "$stage/$name.tar.gz" "$name"
mv "$stage/$name.tar.gz" "$dest/$name.tar.gz"
(cd "$dest" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")
