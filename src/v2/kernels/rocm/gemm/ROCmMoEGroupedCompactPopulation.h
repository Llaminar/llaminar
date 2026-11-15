/**
 * @file ROCmMoEGroupedCompactPopulation.h
 * @brief Capture-local ownership of Compact GU's shared fixed-row directory.
 *
 * A complete operation retains one immutable address/geometry contract. Only a
 * successful GU submission creates a move-only receipt that its own operation
 * can consume for down. The host tracks construction, never live device counts
 * or the device's arithmetic choice. Both roles use a fixed sixteen-row map,
 * so SwiGLU can intervene without rebuilding or reinterpreting the directory.
 */
#pragma once
#include "ROCmMoEGroupedPopulationProjection.h"
#include <memory>
#include <optional>
#include <utility>

namespace llaminar2::rocm
{
/** @brief One nonmovable complete-operation construction scope for Compact Auto. */
class MoEGroupedCompactPopulationTransaction final
{
    /** Retained host identity prevents stale-token acceptance after scope-address reuse. */
    struct ConstructionIdentity final {};
public:
    /** Fixed TM16 storage bound, with the live count immediately after payload. */
    static constexpr int kDirectoryCapacity = 512 * 8 / 16 + 8;
    /** Physical words, including the one authoritative live-count publication. */
    static constexpr std::size_t kDirectoryWords = kDirectoryCapacity + 1;

    /** @brief Proof that this scope successfully recorded every GU producer. */
    class GateUpSubmission final
    {
    public:
        /** @param other Receipt whose unique consumption right moves here. */
        GateUpSubmission(GateUpSubmission &&other) noexcept
            : identity_(std::move(other.identity_)),receipt_(std::move(other.receipt_)) {}
        GateUpSubmission(const GateUpSubmission &) = delete;
        GateUpSubmission &operator=(const GateUpSubmission &) = delete;
        GateUpSubmission &operator=(GateUpSubmission &&) = delete;
        /** @return Successful GU launch inventory before down consumes the receipt.
         * @throws std::logic_error For a moved-from or consumed receipt. */
        [[nodiscard]] const MoEGroupedPrefillRoleSubmission &receipt() const
        {
            if (!identity_) throw std::logic_error("Compact GU receipt was already consumed");
            return receipt_;
        }
    private:
        friend class MoEGroupedCompactPopulationTransaction;
        /** @param identity Retained identity of the lexical operation owning all bindings.
         * @param receipt Inventory published only after all GU launches succeed. */
        GateUpSubmission(std::shared_ptr<const ConstructionIdentity> identity,
            MoEGroupedPrefillRoleSubmission receipt) : identity_(std::move(identity)),receipt_(std::move(receipt)) {}
        std::shared_ptr<const ConstructionIdentity> identity_;
        MoEGroupedPrefillRoleSubmission receipt_;
    };

    /** @param key Complete immutable operation, including source/physical identities.
     * @param workspace Canonical shared directory and decision storage.
     * @param counts Authoritative device expert counts. @param offsets Matching grouped offsets.
     * @param device Admitted visible device. @param stream Exact producer/consumer stream.
     * @throws std::invalid_argument For an unrecognized domain or incomplete storage/bindings.
     * @note Native stream/device authentication occurs before the first submission. */
    MoEGroupedCompactPopulationTransaction(MoEGroupedPopulationKey key,
        MoEGroupedPopulationWorkspace workspace,const int *counts,const int *offsets,
        int device,void *stream)
        : key_(key),workspace_(workspace),counts_(counts),offsets_(offsets),device_(device),stream_(stream)
    {
        if (groupedPopulationDomain(key) != MoEGroupedPopulationDomain::Compact)
            throw std::invalid_argument("Compact population requires the complete admitted operation");
        if (!workspace.selection || !workspace.directory || workspace.directory_words < kDirectoryWords)
            throw std::invalid_argument("Compact population directory requires 265 canonical words and a decision");
        if (!counts || !offsets || device < 0 || !stream)
            throw std::invalid_argument("Compact population requires counts, offsets, device and exact stream");
    }
    MoEGroupedCompactPopulationTransaction(const MoEGroupedCompactPopulationTransaction &) = delete;
    MoEGroupedCompactPopulationTransaction &operator=(const MoEGroupedCompactPopulationTransaction &) = delete;
    MoEGroupedCompactPopulationTransaction(MoEGroupedCompactPopulationTransaction &&) = delete;
    MoEGroupedCompactPopulationTransaction &operator=(MoEGroupedCompactPopulationTransaction &&) = delete;

    /** @brief Record one shared directory and one adaptive ordered/native GU consumer.
     * @param codebooks Actual physical GU mask. @param args Complete original projection bindings.
     * @return Unique successful producer receipt, or no receipt after a rejected/failed launch.
     * @throws std::logic_error If this construction scope was already used. */
    [[nodiscard]] std::optional<GateUpSubmission> gateUp(uint32_t codebooks,const MoEGroupedWmmaArguments &args);
    /** @brief Consume this operation's fixed-TM16 directory after ordered SwiGLU.
     * @param produced Unique GU receipt from this lexical scope, consumed exactly once.
     * @param codebooks Actual physical down mask. @param args Down-specific projection packet.
     * @return Successful down inventory, or no receipt after a rejected/failed launch.
     * @throws std::logic_error For a foreign, moved-from or previously consumed receipt. */
    [[nodiscard]] std::optional<MoEGroupedPrefillRoleSubmission> down(
        GateUpSubmission &&produced,uint32_t codebooks,const MoEGroupedWmmaArguments &args);
private:
    /** Only capture construction advances this state; replay has no host-side state. */
    enum class Construction { Fresh, GateUpRecorded, Retired };
    /** @param role Actual role. @param codebooks Physical mask. @param args Original packet.
     * @return Whether all immutable bindings and the source arithmetic contract match this scope. */
    [[nodiscard]] bool accepts(MoEPrefillProjectionRole role,uint32_t codebooks,
        const MoEGroupedWmmaArguments &args) const;
    const MoEGroupedPopulationKey key_;
    const MoEGroupedPopulationWorkspace workspace_;
    const int *const counts_;
    const int *const offsets_;
    const int device_;
    void *const stream_;
    // This tiny lifetime tag is allocated during graph construction only. It
    // never enters a GPU packet or replay, and retains no mutable device state.
    const std::shared_ptr<const ConstructionIdentity> identity_ = std::make_shared<const ConstructionIdentity>();
    Construction construction_ = Construction::Fresh;
    /** The native recording interval is immutable construction metadata, not device state. */
    enum class Recording { Unbound, Eager, Captured };
    Recording recording_ = Recording::Unbound;
    unsigned long long capture_id_ = 0;
};
}
