/**
 * @file MoESparseRequestIdentity.h
 * @brief Typed request authority for retained sparse-MoE graph transactions.
 *
 * A process-local graph and a distributed ExpertOverlay graph both need a
 * nonzero transaction generation, but they do not share an authority.  Local
 * execution is authenticated by the orchestrator's request-state session;
 * distributed overlay execution is authenticated by the continuation root's
 * published generation.  Keeping that distinction in the type prevents
 * coordinator presence, backend kind, or an unrelated overlay field from
 * becoming an accidental lifecycle proxy.
 */

#pragma once

#include <cstdint>
#include <optional>

namespace llaminar2
{
    /** Exact owner of one sparse-MoE request generation. */
    enum class MoESparseRequestAuthority : std::uint8_t
    {
        LocalRequestState = 0, ///< Rank-local request-state session.
        ExpertOverlayRoot, ///< Continuation-root publication shared by ranks.
    };

    /**
     * @brief Immutable identity installed before capture or retained replay.
     *
     * Zero is reserved as the wire-level invalid value.  Construction is
     * therefore restricted to the two validating factories below; consumers
     * cannot accidentally create an apparently typed but unusable identity.
     */
    class MoESparseRequestIdentity final
    {
    public:
        /** @return A local-session identity, or nullopt for generation zero. */
        [[nodiscard]] static std::optional<MoESparseRequestIdentity>
        localRequestState(std::uint64_t generation) noexcept
        {
            return make(
                MoESparseRequestAuthority::LocalRequestState,
                generation);
        }

        /** @return A root-published overlay identity, or nullopt for zero. */
        [[nodiscard]] static std::optional<MoESparseRequestIdentity>
        expertOverlayRoot(std::uint64_t generation) noexcept
        {
            return make(
                MoESparseRequestAuthority::ExpertOverlayRoot,
                generation);
        }

        /** @return Exact authority selected for this transaction. */
        [[nodiscard]] MoESparseRequestAuthority authority() const noexcept
        {
            return authority_;
        }

        /** @return Nonzero generation published by that authority. */
        [[nodiscard]] std::uint64_t generation() const noexcept
        {
            return generation_;
        }

    private:
        MoESparseRequestIdentity(
            MoESparseRequestAuthority authority,
            std::uint64_t generation) noexcept
            : authority_(authority), generation_(generation)
        {
        }

        /** Validate the wire invariant shared by both authority kinds. */
        [[nodiscard]] static std::optional<MoESparseRequestIdentity> make(
            MoESparseRequestAuthority authority,
            std::uint64_t generation) noexcept
        {
            if (generation == 0)
                return std::nullopt;
            return MoESparseRequestIdentity(authority, generation);
        }

        MoESparseRequestAuthority authority_ =
            MoESparseRequestAuthority::LocalRequestState;
        std::uint64_t generation_ = 0;
    };
} // namespace llaminar2
