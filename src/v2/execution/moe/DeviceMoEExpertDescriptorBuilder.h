/**
 * @file DeviceMoEExpertDescriptorBuilder.h
 * @brief Typed export of prepared expert engines into the runtime-table ABI.
 *
 * Prepared weights may be NativeVNNI or contiguous FP16/BF16/FP32. Graph
 * builders, compute stages and migration share one family/geometry validator.
 * Whole experts require all three projections. Projection-partitioned domains
 * publish an explicitly typed movable gate/up pair and keep fixed down slices
 * outside that placement lifecycle. Rejection never partially mutates a bank.
 */

#pragma once

#include "MoERuntimeTable.h"
#include "MoEExpertProjectionOwnership.h"
#include "MoEOverlayPreparedExpertPayload.h"

#include <array>

namespace llaminar2
{
    /**
     * @brief Convert the tensor-facing floating descriptor into the device ABI.
     *
     * @param source Prepared engine's immutable contiguous weight view.
     * @param output Compact device-readable matrix descriptor.
     * @param format Receives the exact floating-point element precision.
     * @return True only when pointer, byte count, precision, and geometry agree.
     */
    inline bool exportDeviceMoEFloatingMatrixDescriptor(
        const ContiguousFloatingPointWeightDescriptor &source,
        DeviceMoEFloatingMatrixDesc &output,
        DeviceMoEWeightFormat &format) noexcept
    {
        if (!source.valid())
            return false;

        switch (source.type)
        {
        case TensorType::FP16:
            format = DeviceMoEWeightFormat::FP16;
            break;
        case TensorType::BF16:
            format = DeviceMoEWeightFormat::BF16;
            break;
        case TensorType::FP32:
            format = DeviceMoEWeightFormat::FP32;
            break;
        default:
            return false;
        }

        output = {
            .data = source.data,
            .n = source.n,
            .k = source.k,
        };
        return output.valid();
    }

    namespace moe_descriptor_detail
    {
    /**
     * @brief Validate an entire movable payload before replacing any descriptor.
     * @param gate Prepared complete gate projection.
     * @param up Prepared complete up projection.
     * @param down Complete down engine, absent only for an explicit GateUp contract.
     * @param projections Immutable projection set selected by the ownership plan.
     * @param d_model Full original model width, never a local down-slice width.
     * @param intermediate Full expert intermediate width.
     * @param output Publication target; placement and allocation capacity survive.
     * @return False without mutation for mixed families, missing engines or bad geometry.
     */
    inline bool exportMovableWeights(
        ITensorGemm *gate,
        ITensorGemm *up,
        ITensorGemm *down,
        DeviceMoEProjectionSet projections,
        int d_model,
        int intermediate,
        DeviceMoEExpertDescriptor &output) noexcept
    {
        const bool complete = projections == DeviceMoEProjectionSet::CompleteExpert;
        if (!deviceMoEProjectionSetValid(projections) || !gate || !up ||
            (complete ? down == nullptr : down != nullptr) ||
            d_model <= 0 || intermediate <= 0)
            return false;

        const std::array<ITensorGemm *, 3> engines{gate, up, down};
        const std::size_t count = complete ? 3u : 2u;
        std::array<DeviceNativeVNNIMatrixDesc, 3> native{};
        std::size_t native_count = 0u;
        for (std::size_t i = 0; i < count; ++i)
            native_count += engines[i]->exportNativeVNNIMatrixDesc(native[i]) ? 1u : 0u;
        if (native_count != 0u)
        {
            if (native_count != count)
                return false;
            for (std::size_t i = 0; i < count; ++i)
            {
                const int n = i == 2 ? d_model : intermediate;
                const int k = i == 2 ? intermediate : d_model;
                if (!native[i].valid() || native[i].n != n || native[i].k != k)
                    return false;
            }

            output.gate = native[0];
            output.up = native[1];
            output.down = native[2];
            output.floating_gate = {};
            output.floating_up = {};
            output.floating_down = {};
            output.weight_format = DeviceMoEWeightFormat::NativeVNNI;
            output.projection_set = projections;
            return true;
        }

        std::array<ContiguousFloatingPointWeightDescriptor, 3> source{};
        std::array<DeviceMoEFloatingMatrixDesc, 3> floating{};
        std::array<DeviceMoEWeightFormat, 3> formats{};
        for (std::size_t i = 0; i < count; ++i)
        {
            const int n = i == 2 ? d_model : intermediate;
            const int k = i == 2 ? intermediate : d_model;
            if (!engines[i]->exportContiguousFloatingPointWeights(source[i]) ||
                !exportDeviceMoEFloatingMatrixDescriptor(source[i], floating[i], formats[i]) ||
                formats[i] != formats[0] || floating[i].n != n || floating[i].k != k)
                return false;
        }

        // Publish only after all required matrices validate. A rejected family
        // leaves the previous usable descriptor and its allocation tags intact.
        output.gate = {};
        output.up = {};
        output.down = {};
        output.floating_gate = floating[0];
        output.floating_up = floating[1];
        output.floating_down = floating[2];
        output.weight_format = formats[0];
        output.projection_set = projections;
        return true;
    }
    } // namespace moe_descriptor_detail

    /**
     * @brief Export a complete expert without weakening whole-FFN readiness.
     * @param gate Prepared gate engine with shape [intermediate, d_model].
     * @param up Prepared up engine with the same shape and arithmetic family.
     * @param down Prepared complete down engine with shape [d_model, intermediate].
     * @param d_model Full model width.
     * @param intermediate Full expert intermediate width.
     * @param output Runtime publication target; unchanged on rejection.
     * @return Whether every required projection validates as one supported family.
     */
    inline bool exportDeviceMoEExpertWeightDescriptors(
        ITensorGemm *gate, ITensorGemm *up, ITensorGemm *down,
        int d_model, int intermediate, DeviceMoEExpertDescriptor &output) noexcept
    {
        return moe_descriptor_detail::exportMovableWeights(gate, up, down,
            DeviceMoEProjectionSet::CompleteExpert, d_model, intermediate, output);
    }

    /**
     * @brief Export the movable pair of a projection-partitioned expert.
     * @param gate Prepared complete gate engine, authenticated by the prepared registry.
     * @param up Prepared complete up engine, authenticated by the same registry.
     * @param ownership Frozen domain layout; a whole-expert layout is rejected.
     * @param output Runtime publication target; no fixed down pointer is accepted.
     * @return Whether the pair validates against the original source geometry.
     */
    inline bool exportDeviceMoEGateUpWeightDescriptors(
        ITensorGemm *gate, ITensorGemm *up,
        const MoEExpertProjectionOwnership &ownership,
        DeviceMoEExpertDescriptor &output) noexcept
    {
        if (ownership.movableProjections() != DeviceMoEProjectionSet::GateUp)
            return false;
        const auto geometry = ownership.geometry();
        return moe_descriptor_detail::exportMovableWeights(gate, up, nullptr,
            ownership.movableProjections(), geometry.model_columns,
            geometry.intermediate_columns, output);
    }

    /**
     * @brief Export an authenticated arrival without inventing a host geometry shadow.
     * @param payload Complete declared family retained by the physical lifetime authority.
     * @param output Publication target; unchanged on rejection.
     * @return Whether all movable projections validate against the prepared gate geometry.
     *
     * Fixed down slices never occur in a gate/up payload. The same validator
     * handles quantized and floating families and checks every declared engine;
     * inspecting the gate here supplies dimensions, not a weaker readiness test.
     */
    inline bool exportDeviceMoEPreparedPayload(
        const MoEOverlayPreparedExpertPayload &payload,
        DeviceMoEExpertDescriptor &output) noexcept
    {
        if (!payload.ready())
            return false;
        DeviceNativeVNNIMatrixDesc native_gate{};
        if (payload.gate()->exportNativeVNNIMatrixDesc(native_gate))
        {
            return native_gate.valid() && moe_descriptor_detail::exportMovableWeights(
                payload.gate().get(), payload.up().get(), payload.down().get(), payload.projections(),
                native_gate.k, native_gate.n, output);
        }
        ContiguousFloatingPointWeightDescriptor floating_gate{};
        return payload.gate()->exportContiguousFloatingPointWeights(floating_gate) &&
               floating_gate.valid() && moe_descriptor_detail::exportMovableWeights(
                   payload.gate().get(), payload.up().get(), payload.down().get(), payload.projections(),
                   floating_gate.k, floating_gate.n, output);
    }

    /**
     * @brief Export a prepared expert while deriving its coherent geometry.
     *
     * Migration destinations already own fully prepared gate/up/down engines,
     * but the device-authored command intentionally carries only logical
     * placement and byte identity.  Requiring a host policy object merely to
     * repeat `d_model` and intermediate width would create a second source of
     * truth.  This overload derives those two dimensions from the immutable
     * engine descriptors, then delegates to the exact geometry validator above.
     *
     * @param gate Prepared gate projection engine.
     * @param up Prepared up projection engine.
     * @param down Prepared down projection engine.
     * @param output Runtime descriptor to populate.
     * @return True only for one complete coherent NativeVNNI or floating family.
     */
    inline bool exportDeviceMoEExpertWeightDescriptors(
        ITensorGemm *gate,
        ITensorGemm *up,
        ITensorGemm *down,
        DeviceMoEExpertDescriptor &output) noexcept
    {
        if (!gate || !up || !down)
            return false;

        DeviceNativeVNNIMatrixDesc native_gate{};
        DeviceNativeVNNIMatrixDesc native_up{};
        DeviceNativeVNNIMatrixDesc native_down{};
        const bool gate_native = gate->exportNativeVNNIMatrixDesc(native_gate);
        const bool up_native = up->exportNativeVNNIMatrixDesc(native_up);
        const bool down_native = down->exportNativeVNNIMatrixDesc(native_down);
        if (gate_native || up_native || down_native)
        {
            if (!gate_native || !up_native || !down_native ||
                !native_gate.valid() || !native_up.valid() ||
                !native_down.valid())
            {
                return false;
            }
            return exportDeviceMoEExpertWeightDescriptors(
                gate,
                up,
                down,
                native_gate.k,
                native_gate.n,
                output);
        }

        ContiguousFloatingPointWeightDescriptor floating_gate{};
        ContiguousFloatingPointWeightDescriptor floating_up{};
        ContiguousFloatingPointWeightDescriptor floating_down{};
        if (!gate->exportContiguousFloatingPointWeights(floating_gate) ||
            !up->exportContiguousFloatingPointWeights(floating_up) ||
            !down->exportContiguousFloatingPointWeights(floating_down) ||
            !floating_gate.valid() || !floating_up.valid() ||
            !floating_down.valid())
        {
            return false;
        }
        return exportDeviceMoEExpertWeightDescriptors(
            gate,
            up,
            down,
            floating_gate.k,
            floating_gate.n,
            output);
    }
} // namespace llaminar2
