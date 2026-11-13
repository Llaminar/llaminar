/**
 * @file GEMMArenaReshapeStage.h
 * @brief Retain checked FP32 reshape operands in the existing prepared GEMM stage.
 *
 * The base stage remains the sole arithmetic/workspace/weight authority. This
 * adapter owns only immutable operand metadata and root lifetime. Capture-only
 * preparation seals already-admitted storage; replay does not prepare or rebind.
 */
#pragma once
#include "GEMMStage.h"
#include "tensors/FP32ArenaReshape.h"
#include <stdexcept>
#include <bit>
namespace llaminar2
{
/** @brief Existing prepared projection plus stage-owned whole-storage FP32 shape handles. */
class GEMMArenaReshapeStage final : public GEMMStage
{
public:
    /**
     * @brief Authenticate root BufferIds/pointers and retain operands for the stage lifetime.
     * @param params Existing prepared GEMM geometry and immutable weight identity.
     * @param input Owned arena input shape, with no independent physical allocation.
     * @param output Distinct arena output shape retained until stage retirement.
     * @throws std::invalid_argument For a mismatched device, pointer, BufferId or matrix shape.
     */
    GEMMArenaReshapeStage(Params params, FP32ArenaReshape input, FP32ArenaReshape output)
        : GEMMStage(bind(std::move(params), input, output)), input_(std::move(input)), output_(std::move(output)) {}
    /** @return Capture-only setup; no host preparation runs between native replay iterations. */
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
    { return GraphLaunchPreparationPolicy::CaptureOnly; }
    /**
     * @brief Seal after ordinary arena allocation, without allocating another device buffer.
     * @param context Existing executor participant context.
     * @param stream Exact GPU capture stream, or null for CPU metadata preparation.
     * @return Existing stage preparation result after both storage seals succeed.
     * @throws std::exception For absent, rebound or stale physical storage.
     */
    bool prepareGraphLaunch(IDeviceContext *context, void *stream) override
    {
        if (device().is_gpu() && !stream) throw std::invalid_argument("Reshape GEMM requires an exact capture stream");
        if (!GEMMStage::prepareGraphLaunch(context, stream)) return false;
        input_.sealStorage(); output_.sealStorage(); return true;
    }
    /**
     * @brief Validate sealed GPU operands; CPU execution seals existing host storage once.
     * @param context Existing participant executor context, unchanged for the base GEMM.
     * @return The canonical prepared GEMM result.
     * @throws std::exception If native storage no longer matches the capture seal.
     */
    bool execute(IDeviceContext *context) override
    {
        if (device().is_cpu()) { input_.sealStorage(); output_.sealStorage(); }
        input_.validateStorage(); output_.validateStorage();
        return GEMMStage::execute(context);
    }
    /**
     * @brief Publish stable declaration geometry and validate sealed physical storage on every cache check.
     * @return Nonzero complete stage variant; mutable device count contents are excluded.
     * @throws std::exception For released/rebound storage or a changed backend generation.
     *
     * The executor queries variants before normal preparation, then seals native
     * addresses before recording and snapshots stage variants afterward. Keeping
     * the declaration hash stable avoids a false recapture after valid setup;
     * declarationIdentity validates each physical seal once it exists, so a stale
     * address/runtime fails before any retained launch instead of requesting recapture.
     */
    uint64_t graphCaptureVariantSignature() const override
    {
        return captureVariantForDeclaration(getParams(), input_.declarationIdentity(),
            output_.declarationIdentity(), GEMMStage::graphCaptureVariantSignature());
    }
    /**
     * @brief Compute the same pure capture key used by the stage, without storage preparation.
     * @param params Canonical GEMM bindings, geometry and prepared-weight policy.
     * @param input Exact immutable admitted input declaration.
     * @param output Exact immutable admitted output declaration.
     * @param base_variant Existing GEMM arithmetic/kernel policy identity.
     * @return Versioned signature of every immutable operand/row/launch policy field.
     */
    static uint64_t captureVariantForDeclaration(const Params &params,
        const FP32ArenaReshape::DeclarationIdentity &input,
        const FP32ArenaReshape::DeclarationIdentity &output, uint64_t base_variant)
    {
        uint64_t signature = 0x4650333252530001ULL;
        const auto mix = [&](uint64_t value) {
            signature ^= value + 0x9e3779b97f4a7c15ULL + (signature << 6) + (signature >> 2);
        };
        const auto pointer = [&](const void *value) { mix(reinterpret_cast<uintptr_t>(value)); };
        const auto declaration = [&](const auto &value) {
            pointer(value.root); mix(static_cast<uint64_t>(value.buffer));
            mix(static_cast<uint64_t>(value.device.type)); mix(value.device.ordinal);
            mix(value.bytes); for (size_t dimension : value.shape) mix(dimension);
        };
        declaration(input); declaration(output); mix(base_variant);
        mix(params.m); mix(params.n); mix(params.k);
        mix(std::bit_cast<uint32_t>(params.alpha)); mix(std::bit_cast<uint32_t>(params.beta));
        mix(params.transpose_B); pointer(params.B); pointer(params.bias); pointer(params.bias_tensor);
        mix(params.bias_required); pointer(params.gate_input); mix(params.do_swiglu);
        mix(params.output_range.start); mix(params.output_range.end); mix(params.needs_allreduce);
        mix(params.force_decode_equivalent_verifier_prefill); mix(static_cast<uint64_t>(params.gemm_context));
        mix(static_cast<uint64_t>(params.device_id.type)); mix(params.device_id.ordinal); pointer(params.mpi_ctx);
        pointer(params.prepared_store); mix(params.prepared_ref.has_value());
        if (params.prepared_ref) {
            mix(params.prepared_ref->model_id.value); mix(params.prepared_ref->binding_id);
            mix(static_cast<uint64_t>(params.prepared_ref->kind));
            mix(static_cast<uint64_t>(params.prepared_ref->device.type)); mix(params.prepared_ref->device.ordinal);
        }
        const auto rows = params.verifier_row_range.value_or(DeviceRowRange::fullyActive(params.m)).captureIdentity();
        pointer(rows.count); mix(rows.count_capacity); mix(rows.first_physical_row);
        mix(rows.physical_rows); mix(rows.repeats_per_row); mix(rows.schema_version);
        return signature ? signature : 1;
    }
    /** @return Complete input storage identity to include in the retained graph's capture key. */
    FP32ArenaReshape::CaptureIdentity inputCaptureIdentity() const { return input_.captureIdentity(); }
    /** @return Complete output storage identity to include in the retained graph's capture key. */
    FP32ArenaReshape::CaptureIdentity outputCaptureIdentity() const { return output_.captureIdentity(); }
private:
    /** @brief Reject conflicting caller pointers/IDs and preserve original physical arena contracts. */
    static Params bind(Params params, const FP32ArenaReshape &input, const FP32ArenaReshape &output)
    {
        if ((params.A && params.A != &input.tensor()) || (params.C && params.C != &output.tensor()) ||
            (params.a_buffer_id && *params.a_buffer_id != input.rootBuffer()) ||
            (params.c_buffer_id && *params.c_buffer_id != output.rootBuffer()) ||
            params.device_id != input.tensor().home_device() || params.device_id != output.tensor().home_device() ||
            input.tensor().transferStorageOwner() == output.tensor().transferStorageOwner())
            throw std::invalid_argument("Reshape GEMM requires exact distinct physical root/BufferId/device bindings");
        const auto &a = input.tensor().shape(); const auto &c = output.tensor().shape();
        if (a.size() != 2 || c.size() != 2 || params.m <= 0 || params.n <= 0 || params.k <= 0 ||
            a[0] < static_cast<size_t>(params.m) || a[1] != static_cast<size_t>(params.k) ||
            c[0] < static_cast<size_t>(params.m) || c[1] != static_cast<size_t>(params.n))
            throw std::invalid_argument("Reshape GEMM logical matrix geometry is invalid");
        params.A = &input.tensor(); params.C = &output.tensor();
        params.a_buffer_id = input.rootBuffer(); params.c_buffer_id = output.rootBuffer();
        return params;
    }
    FP32ArenaReshape input_, output_;
};
} // namespace llaminar2
