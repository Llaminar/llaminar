/**
 * @file RetainedParentInputAuthority.h
 * @brief Closed proof that an original retained wire child precedes its exact arena consumer.
 *
 * This is capture-time authority, never ready-byte publication or a request state mirror.
 * Only the actual pipeline transport can mint it. The final composed parent retains
 * the source and its physical owners until native graph retirement.
 */
#pragma once
#include "backends/DeviceId.h"
#include "memory/BufferId.h"
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
namespace llaminar2 {
class ComputeGraph;
class IGPUGraphCapture;
class IWorkerGPUContext;
class PipelineForwardGraphEdges;
namespace test { struct PipelinePublicationCaptureReplayAccess; }
/** @brief Immutable, non-forgeable original producer/consumer frontier proof. */
class RetainedParentInputAuthority final {
public:
    /** @brief Reject a substituted graph/device/stream or a stale physical/source identity. */
    void require(const ComputeGraph& graph, DeviceId device, void* stream) const {
        if (&graph != consumer_ || device != device_ || stream != stream_)
            throw std::logic_error("Retained prior-child producer changed its exact consumer graph/device/stream");
        validate_();
    }
    /** @return The only arena owner authored before this exact local child. */
    std::span<const BufferId> inputs() const noexcept { return inputs_; }
    /** @return Original consumer device, never a substituted source endpoint. */
    DeviceId device() const noexcept { return device_; }
    /** @return Exact immutable binding equality across repeated capture/replay calls. */
    bool sameBinding(const RetainedParentInputAuthority& other) const noexcept {
        return original_worker_.get()==other.original_worker_.get() &&
            owner_.get()==other.owner_.get() && producer_==other.producer_ &&
            consumer_==other.consumer_ && stage_==other.stage_ && device_==other.device_ &&
            stream_==other.stream_ && recording_==other.recording_ && workspace_==other.workspace_ &&
            consumer_signature_==other.consumer_signature_;
    }
private:
    friend class PipelineForwardGraphEdges;
    friend struct test::PipelinePublicationCaptureReplayAccess; ///< Device-free recorded-boundary replay only; never runtime issuance.
    /** @brief Mint only after the real transport has authenticated every bank and native recording.
     * The worker is declared before the source owner so native graph destruction
     * cannot outlive its original context, even when this is their last borrower.
     */
    RetainedParentInputAuthority(std::shared_ptr<IWorkerGPUContext> worker, std::shared_ptr<const void> owner,const IGPUGraphCapture* producer,
        const ComputeGraph* consumer,const void* stage,DeviceId device,void* stream,
        uint64_t recording,uint64_t workspace,uint64_t signature,std::function<void()> validate)
        : original_worker_(std::move(worker)),owner_(std::move(owner)),producer_(producer),consumer_(consumer),stage_(stage),device_(device),
          stream_(stream),recording_(recording),workspace_(workspace),consumer_signature_(signature),validate_(std::move(validate)) {}
    std::shared_ptr<IWorkerGPUContext> original_worker_; ///< Destroyed after the exact native source graph.
    std::shared_ptr<const void> owner_;
    const IGPUGraphCapture* producer_;
    const ComputeGraph* consumer_;
    const void* stage_;
    DeviceId device_;
    void* stream_;
    uint64_t recording_,workspace_,consumer_signature_;
    std::function<void()> validate_;
    const std::array<BufferId,1> inputs_{BufferId::STOCHASTIC_BATCH_OUTPUT_META};
};
}
