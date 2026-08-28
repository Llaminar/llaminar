#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# @file setup_production_parity_tmpfs.sh
# @brief Own the persistent, node-local tmpfs used by model parity campaigns.
#
# The devcontainer shares the host IPC namespace, so systemd-logind may remove
# ordinary /dev/shm entries when the host login session ends. This script
# creates a separately mounted tmpfs whose lifetime is the devcontainer mount
# namespace instead. Re-running setup is intentionally idempotent: an existing
# mount with the exact Llaminar source identity is retained without remounting
# or touching its authenticated model cache.

set -euo pipefail

readonly DEFAULT_MOUNT_POINT="/mnt/llaminar-production-parity"
readonly DEFAULT_SIZE="320G"
readonly MOUNT_SOURCE="llaminar-production-parity"

mount_point="${DEFAULT_MOUNT_POINT}"
mount_size="${DEFAULT_SIZE}"
operation="setup"

usage() {
    cat <<EOF
Usage: $0 [--mount-point PATH] [--size SIZE] [--status | --unmount]

Create or inspect the dedicated persistent tmpfs used by production model
parity campaigns. SIZE is a positive integer followed by K, M, G, T, or P.
The default setup is idempotent and never clears an existing cache.

Options:
  --mount-point PATH  Absolute mount point (default: ${DEFAULT_MOUNT_POINT})
  --size SIZE         tmpfs capacity used only when creating it (default: ${DEFAULT_SIZE})
  --status            Inspect the mount without changing it
  --unmount           Explicitly destroy the tmpfs after campaigns have exited
  -h, --help          Show this help

After setup, run the campaign driver with:
  --model-ramdisk-root ${DEFAULT_MOUNT_POINT} \\
  --persistent-model-cache-dir cache
EOF
}

fail() {
    echo "production parity tmpfs: $*" >&2
    exit 1
}

while (($# > 0)); do
    case "$1" in
        --mount-point)
            (($# >= 2)) || fail "--mount-point requires a value"
            mount_point="$2"
            shift 2
            ;;
        --size)
            (($# >= 2)) || fail "--size requires a value"
            mount_size="$2"
            shift 2
            ;;
        --status)
            [[ "${operation}" == "setup" ]] || fail "choose only one operation"
            operation="status"
            shift
            ;;
        --unmount)
            [[ "${operation}" == "setup" ]] || fail "choose only one operation"
            operation="unmount"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown argument: $1"
            ;;
    esac
done

[[ "${mount_point}" == /* ]] || fail "mount point must be absolute: ${mount_point}"
[[ "${mount_point}" != "/" && "${mount_point}" != "/mnt" ]] || \
    fail "refusing broad mount point: ${mount_point}"
[[ "${mount_size}" =~ ^[1-9][0-9]*[KMGTP]$ ]] || \
    fail "invalid tmpfs size '${mount_size}'; expected a positive K/M/G/T/P value"

# Normalize dot components while allowing the not-yet-created final directory.
normalized_mount_point="$(realpath -m -- "${mount_point}")"
[[ "${normalized_mount_point}" == "${mount_point}" ]] || \
    fail "mount point must already be normalized: ${mount_point}"
[[ ! -L "${mount_point}" ]] || fail "mount point must not be a symlink: ${mount_point}"

describe_mount() {
    local filesystem source options
    filesystem="$(findmnt -n -o FSTYPE --target "${mount_point}")"
    source="$(findmnt -n -o SOURCE --target "${mount_point}")"
    options="$(findmnt -n -o OPTIONS --target "${mount_point}")"

    [[ "${filesystem}" == "tmpfs" ]] || \
        fail "${mount_point} is mounted as ${filesystem}, not tmpfs"
    [[ "${source}" == "${MOUNT_SOURCE}" ]] || \
        fail "${mount_point} is owned by '${source}', not '${MOUNT_SOURCE}'"

    echo "production parity tmpfs: ${mount_point} (${options})"
    df -h -- "${mount_point}"
}

if mountpoint -q -- "${mount_point}"; then
    describe_mount
    if [[ "${operation}" == "unmount" ]]; then
        # Unmount is the explicit destructive boundary: tmpfs contents cease to
        # exist. umount will reject active campaign/model mappings as busy.
        sudo umount -- "${mount_point}"
        echo "production parity tmpfs: unmounted ${mount_point}; cached models are gone"
    fi
    exit 0
fi

[[ "${operation}" != "status" ]] || fail "no Llaminar tmpfs is mounted at ${mount_point}"
[[ "${operation}" != "unmount" ]] || fail "nothing is mounted at ${mount_point}"

if [[ -e "${mount_point}" ]]; then
    [[ -d "${mount_point}" ]] || fail "mount point exists and is not a directory"
    [[ -z "$(find "${mount_point}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || \
        fail "refusing to hide existing files below ${mount_point}"
else
    sudo install -d -m 0755 -- "${mount_point}"
fi

readonly caller_uid="$(id -u)"
readonly caller_gid="$(id -g)"
sudo mount -t tmpfs \
    -o "size=${mount_size},mode=0700,uid=${caller_uid},gid=${caller_gid},nosuid,nodev" \
    "${MOUNT_SOURCE}" "${mount_point}"

# The cache is deliberately a child of the mount: the campaign driver can seal
# that child between runs without changing mount-point ownership or permissions.
mkdir -m 0700 -- "${mount_point}/cache"
describe_mount
