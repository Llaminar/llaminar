#!/bin/sh
# @file arm64-compiler-debs.sh
# @brief Give the ARM64 builder and test-runner byte-identical compiler packages.
#
# The NEON tile-register proof (V2_Integration_CPUNativeNEONTileRegisters)
# binds the exact compiler, assembler and system headers the Integration build
# compiled with, and the installed test re-checks them in the test-runner.
# Builder and test-runner are different images: the builder inherits a cached
# toolchain context, while the test-runner apt-installs later. Ubuntu security
# updates (binutils, gcc, libc, linux-libc-dev) published in between made the
# two disagree, and the gate failed on a package-age difference, not on code.
#
# save <dir>   (builder, before compiling) upgrades the complete dependency
#              closure of the test-runner's compiler packages to the current
#              archive and saves the exact .debs the builder now compiles with.
# install <dir> (test-runner) installs exactly those .debs and fails if any
#              installed version differs, so both images run the same bytes
#              regardless of later archive updates or BuildKit cache reuse.
set -eu

PACKAGES="build-essential g++-14"

mode=${1:?usage: arm64-compiler-debs.sh save|install <dir>}
dir=${2:?usage: arm64-compiler-debs.sh save|install <dir>}

# Upgrading Ubuntu's g++ package restores /usr/bin/g++ -> g++-13 over the GCC 14
# alternative, which would silently change the compiler; select GCC 14 again
# and require the exact driver chain the proof records.
require_gcc14() {
    if update-alternatives --query gcc >/dev/null 2>&1; then
        update-alternatives --set gcc /usr/bin/gcc-14
    else
        update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 \
            --slave /usr/bin/g++ g++ /usr/bin/g++-14
    fi
    resolved=$(readlink -f /usr/bin/c++)
    if [ "${resolved}" != /usr/bin/aarch64-linux-gnu-g++-14 ]; then
        echo "/usr/bin/c++ resolves to ${resolved}, not GCC 14" >&2
        exit 1
    fi
}

case "${mode}" in
save)
    mkdir -p "${dir}"
    apt-get update
    apt-get install -y --no-install-recommends ${PACKAGES}
    # Every real package the compiler packages can pull in (virtual names are <bracketed>).
    closure=$(apt-cache depends --recurse --no-recommends --no-suggests --no-conflicts \
                  --no-breaks --no-replaces --no-enhances ${PACKAGES} \
              | grep -v '^[ <]' | sort -u)
    installed=$(dpkg-query -W -f '${db:Status-Abbrev} ${Package}\n' ${closure} 2>/dev/null \
                | awk '$1 == "ii" { print $2 }')
    # Old cached versions may no longer be downloadable; bring the closure current first.
    apt-get install -y --no-install-recommends --only-upgrade ${installed}
    pinned=""
    for package in ${installed}; do
        pinned="${pinned} ${package}=$(dpkg-query -W -f '${Version}' "${package}")"
    done
    (cd "${dir}" && apt-get download ${pinned})
    require_gcc14
    rm -rf /var/lib/apt/lists/*
    echo "==> [arm64-compiler-debs] saved $(ls "${dir}" | wc -l) packages to ${dir}"
    ;;
install)
    set -- "${dir}"/*.deb
    [ -e "$1" ] || { echo "no compiler packages in ${dir}" >&2; exit 1; }
    apt-get update
    apt-get install -y --no-install-recommends --allow-downgrades "$@"
    for deb in "$@"; do
        package=$(dpkg-deb -f "${deb}" Package)
        expected=$(dpkg-deb -f "${deb}" Version)
        actual=$(dpkg-query -W -f '${Version}' "${package}")
        if [ "${actual}" != "${expected}" ]; then
            echo "compiler package ${package} is ${actual}, builder compiled with ${expected}" >&2
            exit 1
        fi
    done
    require_gcc14
    rm -rf /var/lib/apt/lists/*
    echo "==> [arm64-compiler-debs] installed the builder's $# compiler packages"
    ;;
*)
    echo "unknown mode: ${mode}" >&2
    exit 1
    ;;
esac
