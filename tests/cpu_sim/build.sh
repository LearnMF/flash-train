#!/bin/sh
# usage: build.sh <repo root> <out dir>  -> <out dir>/sim
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$1; out=$2
python3 "$here/prep.py" "$root/src/primitive/hygon/kda" "$out/src"
mkdir -p "$out/gen/flash_train"
sed 's/#cmakedefine01 FTRAIN_PLATFORM_HYGON_HIP/#define FTRAIN_PLATFORM_HYGON_HIP 1/; s/@FTRAIN_HYGON_HIP_ARCHS@/gfx936/' \
    "$root/include/flash_train/control.h.in" > "$out/gen/flash_train/control.h"
sed 's/@PROJECT_VERSION_MAJOR @/0/; s/@PROJECT_VERSION_MINOR @/0/; s/@PROJECT_VERSION_PATCH @/0/; s/@PROJECT_VERSION@/0.0.0/' \
    "$root/include/flash_train/version.h.in" > "$out/gen/flash_train/version.h"
inc="-I$out/gen -I$here/fake -I$root/src/include -I$root/include -I$out/src"
flags="-std=c++20 -O1 -ffp-contract=off -fno-strict-aliasing -w -pthread"
clang++ $flags $inc -c "$out/src/g2_kernels.cpp" -o "$out/kernels.o" &
clang++ $flags $inc -c "$out/src/g2_runner.cpp" -o "$out/runner.o" &
clang++ $flags $inc -c "$here/sim_main.cpp" -o "$out/main.o" &
clang++ $flags $inc -c "$root/src/error.cpp" -o "$out/error.o" &
wait
clang++ -pthread "$out/kernels.o" "$out/runner.o" "$out/main.o" "$out/error.o" -o "$out/sim"
