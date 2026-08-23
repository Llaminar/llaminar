/**
 * @file Sha256.h
 * @brief Streaming SHA-256 helpers for stable content-addressed artifacts.
 *
 * Runtime artifacts that outlive a process must be named from the bytes that
 * produced them, not from a pathname that can later point at another file.
 * This helper computes the SHA-256 digest of a file without materializing the
 * whole file in RAM. Concurrent callers asking for the same immutable file
 * identity share one calculation.
 */

#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Compute the lowercase hexadecimal SHA-256 digest of a file.
     *
     * The implementation reads the file through a bounded streaming buffer.
     * Calls for the same canonical path, device/inode, size, modification time,
     * and change time are coalesced within one process. The identity is checked
     * again after the scan so a concurrent replacement is never memoized.
     *
     * @param path File whose exact contents identify the artifact namespace.
     * @param error Optional diagnostic populated when hashing fails.
     * @return Sixty-four lowercase hexadecimal characters on success.
     */
    std::optional<std::string> sha256FileHex(
        const std::filesystem::path &path,
        std::string *error = nullptr);

    /**
     * @brief Compute a file digest through a trusted cross-process cache.
     *
     * The cache key binds the canonical path, device/inode identity, byte
     * length, modification time, and change time. A process takes an advisory
     * lock for that exact identity, validates the identity again, and either
     * consumes the already-authenticated digest or streams every file byte and
     * atomically publishes the result. Consequently several parity processes
     * can authenticate one immutable GGUF once without accepting a digest for
     * replaced or modified bytes.
     *
     * The caller must provide a private, trusted cache directory. Cache setup,
     * locking, identity instability, and publication failures are reported as
     * hard errors; this explicit mode never falls back to independent scans.
     *
     * @param path File whose exact contents identify the artifact namespace.
     * @param cache_directory Private directory shared by cooperating processes.
     * @param error Optional diagnostic populated when authentication fails.
     * @return Sixty-four lowercase hexadecimal characters on success.
     */
    std::optional<std::string> sha256FileHexShared(
        const std::filesystem::path &path,
        const std::filesystem::path &cache_directory,
        std::string *error = nullptr);

    /**
     * @brief Hash the stable filesystem identity of an ordered artifact set.
     *
     * This deliberately does not read file payload bytes. It serializes each
     * regular file's canonical path, device, inode, size, mtime, and ctime,
     * then SHA-256 hashes that small descriptor. Ordinary replacement or
     * in-place mutation therefore selects a new namespace, while prefix-cache
     * setup remains constant-time even for a multi-hundred-gigabyte model.
     *
     * The ordered set supports split GGUF models: every shard participates in
     * one identity and changing any shard invalidates the complete artifact.
     * This identity is suitable for runtime caches owned by an already-loaded
     * model context; it is not a cryptographic attestation of untrusted bytes.
     *
     * @param paths Ordered regular files comprising one loaded model artifact.
     * @param error Optional diagnostic populated when identity capture fails.
     * @return Sixty-four lowercase hexadecimal characters on success.
     */
    std::optional<std::string> sha256FileSetIdentityHex(
        const std::vector<std::filesystem::path> &paths,
        std::string *error = nullptr);

    /**
     * @brief Compute the lowercase hexadecimal SHA-256 digest of in-memory bytes.
     *
     * This overload is intended for small runtime identities such as an exact
     * benchmark prompt. It hashes every byte, including embedded NULs and final
     * newlines, so diagnostics can prove that two benchmark invocations used
     * precisely the same input rather than merely the same pathname.
     *
     * @param bytes Exact byte range to hash.
     * @param error Optional diagnostic populated when OpenSSL rejects the hash.
     * @return Sixty-four lowercase hexadecimal characters on success.
     */
    std::optional<std::string> sha256BytesHex(
        std::string_view bytes,
        std::string *error = nullptr);
} // namespace llaminar2
