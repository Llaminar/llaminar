/**
 * @file PipelineMetadataExchange.h
 * @brief Typed, captured publication of terminal-owned pipeline metadata.
 *
 * A homogeneous pipeline uses its native collective. Heterogeneous domains
 * receive through TransferEngine, broadcast inside their own native TP group,
 * then publish to the preceding domain when present. The same local banks feed
 * the model and accepted-state commit: there is no host mirror or staging copy.
 * All addresses and semantic message identities are frozen before capture.
 */
#pragma once
#include "PipelineNativeDomain.h"

#include "execution/compute_stages/IComputeStage.h"
#include "collective/ILocalTPContext.h"
#include "transfer/CapturedTransferChannel.h"
#include "memory/BufferId.h"
#include <array>
#include <span>
#include <variant>

namespace llaminar2
{
/** @brief Immutable wire schema; shape alone never chooses an execution role. */
class PipelineMetadataLayout final
{
public:
    /** @brief Exactly the metadata families consumed by captured pipeline graphs. */
    enum class Kind : uint8_t { Condition = 1, Verifier, CommittedState, NextToken };
    /** @brief Validate a semantic family and its physical row extent.
     * @param kind Terminal-produced family; no sampler/controller payload is allowed.
     * @param rows Physical verifier rows, or one for every scalar family. */
    PipelineMetadataLayout(Kind kind, int rows);
    /** @return Semantic identity independent of device addresses and contents. */
    Kind kind() const noexcept { return kind_; }
    /** @return Number of independent INT32 banks in this family. */
    size_t fieldCount() const noexcept;
    /** @return Number of INT32 elements in one bank; throws for an invalid index. */
    size_t elements(size_t field) const;
    /** @return Exact role/row/field-qualified identity for one channel operation. */
    CapturedTransferMessage message(size_t field) const;
    /** @return Stable semantic label, shared by native and heterogeneous paths. */
    const char *name() const noexcept;
private:
    Kind kind_;
    int rows_;
};

/** @brief Persistent local INT32 region retained by a heterogeneous graph edge. */
class PipelineMetadataBank final
{
public:
    /** @brief Retain an exact subregion of an existing device tensor.
     * @param owner Original tensor storage owner; no allocation is performed.
     * @param arena_id Canonical arena identity used by the executor's dependency ledger.
     * @param offset Byte offset within that owner; must be INT32 aligned.
     * @param elements Positive count of INT32 elements. */
    PipelineMetadataBank(std::shared_ptr<TensorBase> owner, BufferId arena_id, size_t offset, size_t elements);
    /** @brief Retain an existing named workspace allocation with its original PMA claim.
     * @param owner Retained bounded region, not a borrowed raw workspace pointer.
     * @param offset Byte offset in the retained region; must be INT32 aligned.
     * @param elements Positive count of INT32 elements. */
    PipelineMetadataBank(std::shared_ptr<const WorkspaceBufferLease> owner, size_t offset, size_t elements);
    /** @return Exact local device frozen at construction. */
    DeviceId device() const noexcept { return device_; }
    /** @return Exact bank length, not allocation capacity. */
    size_t elements() const noexcept { return elements_; }
    /** @return Canonical arena identity, absent only for an owned workspace region. */
    std::optional<BufferId> arenaId() const noexcept { return arena_id_; }
    /** @return Stable address after revalidating the physical owner and extent. */
    void *data() const;
    /** @brief Bind this original owner through the canonical transfer authority. */
    CapturedTransferBinding bind(TransferEngine &transfer, std::shared_ptr<CapturedTransferChannel> channel,
        CapturedTransferEndpoint endpoint, CapturedTransferMessage message) const;
    /** @brief Prepare native broadcast output on the exact stream, without allocation. */
    void requireOutput(void *stream) const;
    /** @brief Publish native broadcast completion on the same explicit stream. */
    void publishOutput(void *stream) const;
private:
    /** @brief Validate positive, aligned, non-overflowing subregion geometry. */
    void validateGeometry(size_t available) const;
    std::variant<std::shared_ptr<TensorBase>, std::shared_ptr<const WorkspaceBufferLease>> owner_;
    size_t offset_;
    size_t elements_;
    DeviceId device_;
    void *data_;
    std::optional<BufferId> arena_id_;
};

/** @brief One participant-local metadata collective, with no host execution state. */
class PipelineMetadataExchange final : public IComputeStage
{
public:
    /** @brief Native pipeline-wide broadcast; terminal is the last participant. */
    struct Native
    {
        ILocalTPContext &context;
        int participant;
    };
    /**
     * @brief Reverse-domain chain carrying only terminal-authored metadata.
     *
     * Inbound comes from the next layer domain; outbound leads to the previous
     * one. At least one edge is required. Only member zero touches channels;
     * every member of a receiving domain participates in its native broadcast.
     * A null context means a single-device domain, never a host collective.
     */
    struct CapturedDomain
    {
        ILocalTPContext *context = nullptr;
        int participant = 0;
        std::shared_ptr<CapturedTransferChannel> inbound;
        std::shared_ptr<CapturedTransferChannel> outbound;
    };
    /** @brief Retain the existing homogeneous metadata protocol and exact local pointers. */
    PipelineMetadataExchange(DeviceId device, PipelineMetadataLayout layout, Native native,
        std::span<void *const> fields);
    /** @brief Bind a cross-backend domain using original storage owners and native TP.
     * @param device Exact local participant.
     * @param layout Immutable semantic family/extent.
     * @param domain Frozen adjacent transport and native membership.
     * @param banks Exact local regions; length must match the complete schema.
     * @param transfer Canonical movement authority, injectable for device-free tests. */
    PipelineMetadataExchange(DeviceId device, PipelineMetadataLayout layout, CapturedDomain domain,
        std::vector<PipelineMetadataBank> banks, TransferEngine &transfer = TransferEngine::instance());
    /** @brief Record receive, native fanout, and preceding-domain publication in order. */
    bool execute(IDeviceContext *context) override;
    /** @return Explicit collective classification, never hidden in compute. */
    ComputeStageType type() const override { return ComputeStageType::PIPELINE_ACTIVATION_EXCHANGE; }
    /** @return Stable family label, independent of live values. */
    std::string name() const override { return layout_.name(); }
    /** @return True only for the participant's frozen GPU backend. */
    bool supportsBackend(ComputeBackendType backend) const override;
    /** @return Complete graph-native transport with no replay-time host work. */
    bool isGraphCapturable() const override { return true; }
    /** @return Exact logical metadata bytes, not backing allocation capacity. */
    size_t estimatedMemoryBytes() const override;
    /** @return Each region's transfer/publication owns its explicit coherence edge. */
    CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
    /** @return Exact arena reads/writes for normal executor-owned capture dependency planning. */
    StageBufferContract bufferContract() const override;
    /** @return Immutable geometry only; never downloads model or controller state. */
    StageDumpInfo buildDumpInfoImpl() const override;
private:
    /** @brief Original regions and leader-only channel bindings outlive the captured graph. */
    struct DomainBinding
    {
        CapturedDomain domain;
        std::vector<PipelineMetadataBank> banks;
        std::vector<CapturedTransferBinding> inbound;
        std::vector<CapturedTransferBinding> outbound;
        TransferEngine *transfer;
    };
    /** @brief Reject foreign or changed native membership before any native submission. */
    void validateMembership() const;
    PipelineMetadataLayout layout_;
    std::variant<Native, DomainBinding> transport_;
    std::array<LocalTPCollectiveSidebandBuffer, 4> fields_{};
    PipelineNativeDomain membership_;
};
}
