/**
 * @file ImmutableLookupStage.h
 * @brief Backend-neutral selected native-row gather at an explicit storage boundary.
 *
 * CPU execution calls the immutable RAM/mmap table directly. Hosted GPU lookup
 * uses its exact retained I/O boundary. Resident GPU lookup reads the admitted
 * original-format device table directly and owns no host protocol or native-row
 * scratch. All modes retain their exact source and prepared operand identities.
 */
#pragma once
#include "Qwen4ExpPrimitiveStage.h"
#include "../../../qwen4exp/ImmutableLookupExecutionBoundary.h"
#include "../../../qwen4exp/Qwen4ExpPreparedStorage.h"
#include "../../../mtp/DeviceGenerationControlBinding.h"
#include "../../../../loaders/ResidentImmutableRowTable.h"
#include <memory>
#include <variant>

namespace llaminar2
{
/** @brief One native immutable lookup operation with participant-local output. */
class ImmutableLookupStage final : public IComputeStage
{
public:
    /** @brief CPU execution directly borrows the model-owned native row table. */
    struct CPUStorage
    {
        std::shared_ptr<HostImmutableRowProvider> table;
        Qwen4ExpIntBinding nativeRows; ///< Exact host-gather scratch, absent from resident GPU mode.
    };
    /** @brief GPU execution retains its explicit I/O boundary and protocol arena. */
    struct GPUStorage
    {
        std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> boundary;
        Qwen4ExpIntBinding state; ///< Exact 64-byte persistent protocol owner.
        Qwen4ExpIntBinding nativeRows; ///< Exact device response scratch for the explicit host service.
    };
    /** @brief Direct device lookup retains the admitted model owner and original controller words. */
    struct ResidentGPUStorage
    {
        std::shared_ptr<const ResidentImmutableRowTable> table;
        DeviceGenerationControlBinding control;
    };
    /**
     * @brief One GPU-owned cache plus its explicit subordinate RAM/SSD
     * boundary.
     *
     * Roots are the exact model-owned allocations registered under canonical
     * arena roles before graph materialization. Their IDs participate in
     * dependency and publication contracts; no stage allocates, aliases or
     * rebinds them.
     */
    struct AdaptiveGPUStorage
    {
        std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> boundary;
        DeviceGenerationControlBinding control;
        std::optional<Qwen4ExpArenaBinding<TensorType::INT8>> cache;
        Qwen4ExpArenaBinding<TensorType::INT8> frame;
        Qwen4ExpIntBinding state;
        /** @brief Complete original helper-issued query prefix, including counted M1/live0. */
        DeviceRowRange originalRows;
        /** @brief Original arena count subview, present exactly for counted queries. */
        std::optional<Qwen4ExpIntBinding> count;
    };
    using Storage = std::variant<CPUStorage, GPUStorage, ResidentGPUStorage, AdaptiveGPUStorage>;
    /** @brief Immutable storage policy, exact extents and typed output views. */
    struct Params
    {
        STAGE_PARAMS_COMMON_FIELDS;
        Qwen4ExpStageRows rows;
        Qwen4ExpIntBinding ids; ///< [M,16] INT64 packed in INT32 arena backing.
        Qwen4ExpFloatBinding output; ///< [M,16,rowElements] decoded values.
        TensorType format; ///< Authenticated GGUF source type.
        int rowElements; ///< Logical values per lookup row.
        std::size_t rowBytes; ///< Native bytes per lookup row.
        Storage storage;
    };
    /** @brief Authenticate storage/device identity and all admitted view capacities. */
    explicit ImmutableLookupStage(Params params);
    /** @brief Execute direct CPU I/O or record the retained GPU protocol and decoder. */
    bool execute(IDeviceContext *context) override;
    /** @brief Seal already-prepared roots and join resident upload completion on the exact capture stream. */
    bool prepareGraphLaunch(IDeviceContext *context, void *stream) override;
    /** @brief Event joins and pointer seals happen once before capture, never on each replay. */
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
    { return GraphLaunchPreparationPolicy::CaptureOnly; }
    /** @brief Include source/tier/control/geometry identity and reject drift in prepared operands. */
    std::uint64_t graphCaptureVariantSignature() const override;
    /** @brief Stable semantic operation shared by all backends. */
    ComputeStageType type() const override { return ComputeStageType::QWEN4EXP_NGRAM_LOOKUP; }
    /** @return The explicit adaptive arm consumes its complete original counted query prefix. */
    bool supportsPaddedPrefillRealLengthContract() const override
    {
        return std::holds_alternative<AdaptiveGPUStorage>(params_.storage);
    }
    /** @brief Native decoding is linear in the exact gathered values. */
    std::size_t estimatedFlops() const override;
    /** @brief Return declared maximum read/write traffic, excluding unused table bytes. */
    std::size_t estimatedMemoryBytes() const override;
    /** @brief Only the explicitly bound participant backend can execute this stage. */
    bool supportsBackend(ComputeBackendType backend) const override;
    /** @brief Publish canonical output and arena dependencies without host device reads. */
    StageDumpInfo buildDumpInfoImpl() const override;
    /** @brief All storage is admitted before stage construction. */
    StageBufferRequirements getBufferRequirements() const override { return {}; }
    /** @brief Declare exact arena identities; table/channel lifetime is retained by Params. */
    StageBufferContract bufferContract() const override;
private:
    /** @brief Resolve a prebound arena view on the exact execution participant. */
    template<typename T,TensorType Type> T *pointer(Qwen4ExpArenaBinding<Type> binding) const;
    /** @brief Resolve an optional live count without reading GPU contents on the host. */
    const std::int32_t *liveCount() const;
    /** @return The explicit host-service scratch, or no scratch for direct resident lookup. */
    std::optional<Qwen4ExpIntBinding> nativeRows() const;
    Params params_;
    std::optional<Qwen4ExpPreparedStorage> prepared_; ///< Immutable arena pointer/runtime seal.
    std::unique_ptr<qwen4exp::IImmutableResidentLookupKernels> resident_kernels_; ///< Setup-owned codebook bundle.
    std::optional<ImmutableNativeDeviceRead> resident_read_; ///< Exact-stream setup event acquire.
};
} // namespace llaminar2
