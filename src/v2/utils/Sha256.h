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

namespace llaminar2
{
    /**
     * @brief Compute the lowercase hexadecimal SHA-256 digest of a file.
     *
     * The implementation reads the file through a bounded streaming buffer.
     * Calls for the same canonical path, size, and modification time are
     * coalesced within the process so LocalTP/EP children do not independently
     * rescan one model.
     *
     * @param path File whose exact contents identify the artifact namespace.
     * @param error Optional diagnostic populated when hashing fails.
     * @return Sixty-four lowercase hexadecimal characters on success.
     */
    std::optional<std::string> sha256FileHex(
        const std::filesystem::path &path,
        std::string *error = nullptr);
} // namespace llaminar2
