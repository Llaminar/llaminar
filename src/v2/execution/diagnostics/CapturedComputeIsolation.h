/**
 * @file CapturedComputeIsolation.h
 * @brief Closed diagnostic input/reset ownership for isolated captured computation.
 *
 * This contract neither allocates nor suppresses production communication. The
 * concrete stage adapter derives reads/writes from the real stage contract and
 * retains already-admitted GPU storage. Provenance names an external qualified
 * observation or an explicit synthetic fixture; it is not a byte certificate.
 * Every read is restored from a disjoint immutable resident baseline. Outputs
 * and mutable state cannot alias a baseline or another independent binding.
 */
#pragma once
#include "../../backends/DeviceId.h"
#include "../../memory/StageBufferContract.h"
#include "../compute_stages/IComputeStage.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2::diagnostics
{
/** @brief An isolated component never advances a request or publishes sampled tokens. */
enum class IsolationInputOrigin : std::uint8_t { QualifiedInferenceObservation, SyntheticFixture };
/** @brief Exact mathematical source operation and one participant's admitted physical slice. */
struct IsolationGemmShape
{
    int rows=0, original_columns=0, reduction_columns=0;
    int first_column=0, local_columns=0;
    int participant=0, participants=0;
};
/**
 * @brief Original serial output width whose arithmetic this diagnostic must retain.
 *
 * Zero names ordinary identity-width execution. A positive value is the actual
 * source graph's immutable output-partition-equivalence policy, never a tuning
 * knob inferred from measured speed. Physical local columns may be larger.
 */
struct IsolationOutputPartitionPolicy
{
    int serial_columns=0;
    bool operator==(const IsolationOutputPartitionPolicy &) const = default;
};
/** @brief Reject negative/foreign partition geometry before engine inspection or capture. */
inline void validateIsolationOutputPartitionPolicy(IsolationGemmShape shape,IsolationOutputPartitionPolicy policy)
{
    if(policy.serial_columns<0 || shape.local_columns<=0 || policy.serial_columns>shape.local_columns)
        throw std::invalid_argument("Isolated output partition policy exceeds its actual physical columns");
}
/** @brief Immutable external observation identity; no tensor/count value is mirrored here. */
struct IsolationInputProvenance
{
    IsolationInputOrigin origin=IsolationInputOrigin::QualifiedInferenceObservation;
    std::string observation, producer_stage;
    std::uint64_t generation=0;
    std::shared_ptr<const void> owner;
};
/** @brief One exact already-admitted resident binding and its immutable pre-execution baseline. */
struct IsolationBuffer
{
    BufferId id=BufferId::_COUNT;
    std::uintptr_t address=0, baseline=0;
    std::size_t bytes=0;
    std::shared_ptr<const void> owner, baseline_owner;
};
/**
 * @brief Frozen read closure derived from the actual production StageBufferContract.
 *
 * Inputs and inouts require a baseline. Write-only outputs need no reset for
 * computation, but still participate in alias checks and owner lifetime. This
 * initial contract rejects named workspaces as explicit buffer bindings: their
 * real prepared-stage owner is retained separately by the concrete adapter.
 */
class CapturedComputeIsolationContract final
{
public:
    /**
     * @brief Validate the actual stage read/write closure before any capture or enqueue.
     * @param device Exact GPU owner; CPU/global/service adapters are unsupported here.
     * @param shape Original work and an equal contiguous local output slice.
     * @param provenance Pre-execution observation/fixture and its retained lifetime.
     * @param kind Actual concrete stage type; only the initial plain GEMM adapter is implemented.
     * @param stage Actual production stage's contract, not an independently authored read list.
     * @param bindings Exact resident owners corresponding to every arena read/write.
     * @param partition Original identity/serial-width arithmetic, retained as immutable capture identity.
     * @throws std::invalid_argument For missing owners/reset, unsupported shape or aliasing.
     */
    CapturedComputeIsolationContract(DeviceId device, IsolationGemmShape shape,
        IsolationInputProvenance provenance, ComputeStageType kind, const StageBufferContract &stage,
        std::vector<IsolationBuffer> bindings,IsolationOutputPartitionPolicy partition={})
        : device_(device), shape_(shape), partition_(partition), provenance_(std::move(provenance)), bindings_(std::move(bindings))
    {
        validateIsolationOutputPartitionPolicy(shape,partition_);
        if (kind!=ComputeStageType::GEMM || (!device.is_cuda() && !device.is_rocm()) || device.ordinal<0 ||
            shape.rows<=0 || shape.original_columns<=0 || shape.reduction_columns<=0 ||
            shape.participants<=0 || shape.participant<0 || shape.participant>=shape.participants ||
            shape.original_columns%shape.participants || shape.local_columns!=shape.original_columns/shape.participants ||
            shape.first_column!=shape.participant*shape.local_columns ||
            (provenance_.origin!=IsolationInputOrigin::QualifiedInferenceObservation &&
             provenance_.origin!=IsolationInputOrigin::SyntheticFixture) ||
            provenance_.observation.empty() || provenance_.producer_stage.empty() ||
            !provenance_.generation || !provenance_.owner)
            throw std::invalid_argument("Isolated compute requires exact GPU/source topology and retained input provenance");
        std::set<BufferId> wanted, reads;
        for(const auto &binding:stage.inputs) { wanted.insert(binding.id); reads.insert(binding.id); }
        for(const auto &binding:stage.inouts) { wanted.insert(binding.id); reads.insert(binding.id); }
        for(const auto &binding:stage.outputs) wanted.insert(binding.id);
        if(wanted.empty() || reads.empty() || wanted.contains(BufferId::_COUNT) || bindings_.size()!=wanted.size())
            throw std::invalid_argument("Isolated compute requires complete nonempty production buffer closure");
        std::set<BufferId> seen;
        for(const auto &binding:bindings_)
        {
            if(!wanted.contains(binding.id) || !seen.insert(binding.id).second || !binding.owner ||
                !binding.address || !binding.bytes || binding.bytes>std::numeric_limits<std::uintptr_t>::max()-binding.address)
                throw std::invalid_argument("Isolated compute has missing, duplicate or overflowing resident storage");
            if(reads.contains(binding.id))
            {
                if(!binding.baseline || !binding.baseline_owner ||
                    binding.bytes>std::numeric_limits<std::uintptr_t>::max()-binding.baseline)
                    throw std::invalid_argument("Every isolated read/inout requires an owned pre-execution reset baseline");
                resets_.push_back(binding);
            }
            else if(binding.baseline || binding.baseline_owner)
                throw std::invalid_argument("Write-only isolation output cannot invent an input baseline");
        }
        const auto overlap=[](std::uintptr_t a,std::size_t an,std::uintptr_t b,std::size_t bn)
        { return a<b+bn && b<a+an; };
        for(std::size_t i=0;i<bindings_.size();++i)
        {
            const auto &a=bindings_[i];
            for(std::size_t j=0;j<bindings_.size();++j)
            {
                const auto &b=bindings_[j];
                if((i!=j && overlap(a.address,a.bytes,b.address,b.bytes)) ||
                    (a.baseline && overlap(a.baseline,a.bytes,b.address,b.bytes)))
                    throw std::invalid_argument("Isolation reset/compute banks must have disjoint exact storage");
            }
            for(const auto &b:resets_)
                if(a.baseline && a.id!=b.id && overlap(a.baseline,a.bytes,b.baseline,b.bytes))
                    throw std::invalid_argument("Independent isolation input baselines may not alias");
        }
    }
    /** @return Actual participant owning all captured spans. */
    [[nodiscard]] DeviceId device() const noexcept {return device_;}
    /** @return Exact immutable original/local mathematical work. */
    [[nodiscard]] const IsolationGemmShape &shape() const noexcept {return shape_;}
    /** @return Immutable source arithmetic width; part of this exclusive capture identity. */
    [[nodiscard]] IsolationOutputPartitionPolicy outputPartitionPolicy() const noexcept {return partition_;}
    /** @return External provenance; consumers must retain its actual byte/oracle evidence separately. */
    [[nodiscard]] const IsolationInputProvenance &provenance() const noexcept {return provenance_;}
    /** @return Every exact resident read/write owner; never a second allocation ledger. */
    [[nodiscard]] std::span<const IsolationBuffer> bindings() const noexcept {return bindings_;}
    /** @return Complete pre-execution restore spans; record outside compute timing. */
    [[nodiscard]] std::span<const IsolationBuffer> resets() const noexcept {return resets_;}
private:
    DeviceId device_;
    IsolationGemmShape shape_;
    IsolationOutputPartitionPolicy partition_;
    IsolationInputProvenance provenance_;
    std::vector<IsolationBuffer> bindings_,resets_;
};
} // namespace llaminar2::diagnostics
