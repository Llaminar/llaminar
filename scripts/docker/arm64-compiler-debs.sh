#!/bin/sh
# @file arm64-compiler-debs.sh
# @brief Give the ARM64 test-runner the builder's exact compiler packages.
#
# The NEON tile-register proof (V2_Integration_CPUNativeNEONTileRegisters)
# binds the compiler, assembler and headers the Integration build used, and
# the installed test re-checks them in the separate test-runner image. A
# test-runner that apt-installs its compilers gets whatever the archive serves
# at that moment, which drifts from the cached builder toolchain whenever
# Ubuntu publishes an update.
#
# save <dir>    (builder) repacks the installed compiler packages as they
#               are, without touching the archive, so the compiler changes
#               only when the cached toolchain itself is rebuilt.
# install <dir> (test-runner) installs exactly those packages.
set -eu

PACKAGES="build-essential g++-14"

mode=${1:?usage: arm64-compiler-debs.sh save|install <dir>}
dir=${2:?usage: arm64-compiler-debs.sh save|install <dir>}

case "${mode}" in
save)
    apt-get update
    apt-get install -y --no-install-recommends dpkg-repack
    # Installed real packages in the compiler packages' dependency closure.
    closure=$(apt-cache depends --recurse --no-recommends --no-suggests --no-conflicts \
                  --no-breaks --no-replaces --no-enhances ${PACKAGES} \
              | grep -v '^[ <]' | sort -u)
    installed=$(dpkg-query -W -f '${db:Status-Abbrev} ${Package}\n' ${closure} 2>/dev/null \
                | awk '$1 == "ii" { print $2 }')
    mkdir -p "${dir}"
    (cd "${dir}" && dpkg-repack ${installed})
    rm -rf /var/lib/apt/lists/*
    ;;
install)
    dpkg --unpack "${dir}"/*.deb
    # The builder selects GCC 14 through alternatives, so the repacked gcc and
    # g++ packages ship /usr/bin/gcc and /usr/bin/g++ as alternative links.
    # Register them before the packages' configuration scripts use them.
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-14
    dpkg --configure -a
    ;;
*)
    echo "unknown mode: ${mode}" >&2
    exit 1
    ;;
esac
