/**
 * @file ParityReferenceCache.h
 * @brief Explicit, idempotent reference-pack location for container campaigns.
 *
 * The model definition still owns its reference-pack directory and numerical
 * identity. A runner may place those same packs under a persistent mounted
 * root; it must never change their contents or bypass reference authentication.
 * This location policy is test infrastructure, not production inference state.
 */
#pragma once

#include <filesystem>
#include <optional>
#include <stdexcept>

namespace llaminar2::test::parity
{
    /**
     * @brief Resolve a declared pack directory beneath an optional cache root.
     * @param declared Directory supplied by the canonical model definition.
     * @param root Explicit absolute persistent-cache root, or no relocation.
     * @return The same directory policy, relocated at most once.
     * @throws std::invalid_argument If relocation escapes its owned root.
     *
     * Setup may resolve a fixture repeatedly. Already-rooted paths therefore
     * remain unchanged, while foreign absolute paths and parent traversal are
     * rejected instead of silently writing outside the mounted cache.
     */
    inline std::filesystem::path resolveParityReferenceCache(
        const std::filesystem::path &declared,
        const std::optional<std::filesystem::path> &root = std::nullopt)
    {
        if (!root)
            return declared;
        if (!root->is_absolute() || root->lexically_normal() == root->root_path())
            throw std::invalid_argument("parity reference cache requires an absolute non-root directory");
        const auto normalized_root = root->lexically_normal();
        auto relative = declared.is_absolute()
            ? declared.lexically_normal().lexically_relative(normalized_root)
            : declared;
        if (relative.empty() || relative == ".")
            throw std::invalid_argument("parity reference pack must name a child directory");
        for (const auto &component : relative)
            if (component == "..")
                throw std::invalid_argument("parity reference pack escapes its configured cache root");
        return (normalized_root / relative).lexically_normal();
    }
}
