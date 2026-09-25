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
# or touching its authenticated model cache. When a privileged host has already
# exported that same named tmpfs from a devcontainer, setup adopts it with one
# bind mount rather than allocating a second ramdisk or copying any GGUF bytes.

set -euo pipefail

readonly DEFAULT_MOUNT_POINT="/mnt/llaminar-production-parity"
readonly DEFAULT_SIZE="320G"
readonly MOUNT_SOURCE="llaminar-production-parity"

mount_point="${DEFAULT_MOUNT_POINT}"
mount_size="${DEFAULT_SIZE}"
operation="setup"
shared_uid=""

usage() {
    cat <<EOF
Usage: $0 [--mount-point PATH] [--size SIZE] [--share-uid UID] [--status | --unmount]

Create or inspect the dedicated persistent tmpfs used by production model
parity campaigns. SIZE is a positive integer followed by K, M, G, T, or P.
The default setup is idempotent and never clears an existing cache.

Options:
  --mount-point PATH  Absolute mount point (default: ${DEFAULT_MOUNT_POINT})
  --size SIZE         tmpfs capacity used only when creating it (default: ${DEFAULT_SIZE})
  --share-uid UID     Grant an ARC runner UID read access without copying cache data
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
        --share-uid)
            (($# >= 2)) || fail "--share-uid requires a value"
            shared_uid="$2"
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
[[ -z "${shared_uid}" || "${shared_uid}" =~ ^[1-9][0-9]*$ ]] || \
    fail "invalid shared UID '${shared_uid}'; expected a positive integer"

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

ensure_cache_directory() {
    # A named Llaminar mount may have been exported before its first campaign.
    # Create only the documented cache child and never change an existing cache.
    if [[ ! -e "${mount_point}/cache" ]]; then
        mkdir -m 0700 -- "${mount_point}/cache"
    fi
    [[ -d "${mount_point}/cache" ]] || fail "${mount_point}/cache is not a directory"
}

existing_owned_mount() {
    # An exported devcontainer tmpfs keeps the same source name in the host
    # namespace. More than one candidate is ambiguous: choosing one by path or
    # recency could bind an unrelated cached model corpus, so fail explicitly.
    local candidate_target candidate_source
    local -a candidates=()
    while read -r candidate_target candidate_source; do
        [[ "${candidate_source}" == "${MOUNT_SOURCE}" ]] || continue
        [[ "${candidate_target}" == "${mount_point}" ]] && continue
        candidates+=("${candidate_target}")
    done < <(findmnt -rn -t tmpfs -o TARGET,SOURCE)
    ((${#candidates[@]} <= 1)) || \
        fail "multiple existing '${MOUNT_SOURCE}' tmpfs mounts exist; refuse ambiguous adoption"
    ((${#candidates[@]} == 1)) || return 1
    printf '%s\n' "${candidates[0]}"
}

share_runner_access() {
    # Local callers do not request ARC ACLs. This is a successful no-op: bare
    # `return` would preserve the false test status and abort setup under -e.
    [[ -n "${shared_uid}" ]] || return 0
    # ARC's non-root runner receives read/traverse ACLs on the same mounted
    # pages. This deliberately does not make that runner the persistent-cache
    # lifecycle owner: staging toggles directory modes and therefore must stay
    # with the one UID that owns the cache lock, manifest and directories.
    # ACLs preserve the owner, source identity, cache bytes and NUMA placement;
    # they make retained, sealed GGUFs reusable without a second ramdisk.
    sudo setfacl -m "u:${shared_uid}:r-x" -- "${mount_point}"
    sudo setfacl -m "d:u:${shared_uid}:r-x" -- "${mount_point}"
    sudo setfacl -R -m "u:${shared_uid}:rX" -- "${mount_point}/cache"
    sudo setfacl -m "d:u:${shared_uid}:r-x" -- "${mount_point}/cache"
}

if mountpoint -q -- "${mount_point}"; then
    describe_mount
    if [[ "${operation}" == "unmount" ]]; then
        # Unmount is the explicit destructive boundary: tmpfs contents cease to
        # exist. umount will reject active campaign/model mappings as busy.
        sudo umount -- "${mount_point}"
        echo "production parity tmpfs: unmounted ${mount_point}; cached models are gone"
    elif [[ "${operation}" == "setup" ]]; then
        ensure_cache_directory
        share_runner_access
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

if existing_mount="$(existing_owned_mount)"; then
    # This is a bind of the one existing named tmpfs. It adds no page cache,
    # filesystem allocation, copy, remap or cleanup behavior.
    sudo mount --bind -- "${existing_mount}" "${mount_point}"
    ensure_cache_directory
    share_runner_access
    describe_mount
    exit 0
fi

readonly caller_uid="$(id -u)"
readonly caller_gid="$(id -g)"
sudo mount -t tmpfs \
    -o "size=${mount_size},mode=0700,uid=${caller_uid},gid=${caller_gid},nosuid,nodev" \
    "${MOUNT_SOURCE}" "${mount_point}"

# The cache is deliberately a child of the mount: the campaign driver can seal
# that child between runs without changing mount-point ownership or permissions.
ensure_cache_directory
share_runner_access
describe_mount
