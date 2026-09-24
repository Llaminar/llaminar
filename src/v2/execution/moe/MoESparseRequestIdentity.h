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
 * CPU graph invocation ordinals have a separate host-owned sequence shared by
 * every retained sidecar variant; positions and graph-cache lifetimes cannot
 * identify speculative work that may revisit the same row.
 */

#pragma once

#include <cstdint>
#include <limits>
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
        /** @brief Construct only after a factory validates the generation. */
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

    /**
     * @brief Single host graph owner's non-reusable sparse operation sequence.
     *
     * Homogeneous CPU TP participants execute the same ordered host program;
     * each endpoint issues matching ordinals under the root-published request
     * generation and the transport authenticates their equality. One sequence
     * serves full, chained and correction sidecars, regardless of which cached
     * graph executes them. Request reset does not reset this sequence.
     *
     * This is not a device-state mirror. GPU graphs use their device sequence;
     * heterogeneous graph followers use coordinator-issued transaction IDs.
     * The CPU owner calls issue() only after request admission, and is already
     * single-threaded with respect to its own graph execution.
     */
    class MoESparseHostOperationSequence final
    {
    public:
        /** @brief Start a fresh graph owner before its first operation. */
        MoESparseHostOperationSequence() = default;
        MoESparseHostOperationSequence(const MoESparseHostOperationSequence &) = delete;
        MoESparseHostOperationSequence &operator=(const MoESparseHostOperationSequence &) = delete;

        /** @brief Transfer issuance authority and permanently exhaust the source. */
        MoESparseHostOperationSequence(MoESparseHostOperationSequence &&other) noexcept
            : last_issued_(other.last_issued_)
        {
            other.last_issued_ = std::numeric_limits<std::uint64_t>::max();
        }

        /** @brief Follow graph-owner replacement without cloning its operation IDs. */
        MoESparseHostOperationSequence &operator=(MoESparseHostOperationSequence &&other) noexcept
        {
            if (this != &other)
            {
                last_issued_ = other.last_issued_;
                // A moved-from graph must never issue an identity already owned here.
                other.last_issued_ = std::numeric_limits<std::uint64_t>::max();
            }
            return *this;
        }

        /**
         * @return The next positive operation ID, or nullopt on exhaustion.
         *
         * Do not rewind on a failed execution or rejected speculative row:
         * those are completed or failed operations, never unissued identities.
         */
        [[nodiscard]] std::optional<std::uint64_t> issue() noexcept
        {
            if (last_issued_ == std::numeric_limits<std::uint64_t>::max())
                return std::nullopt;
            return ++last_issued_;
        }

    private:
        std::uint64_t last_issued_ = 0; ///< Zero means no invocation has been issued.
    };
} // namespace llaminar2
