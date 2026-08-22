/**
 * @file DeviceMoEExpertDescriptorBuilder.h
 * @brief Typed export of prepared expert engines into the runtime-table ABI.
 *
 * Prepared weights may be NativeVNNI or contiguous FP16/BF16/FP32.  Graph
 * builders and compute stages must classify the complete gate/up/down triple
 * identically, so this header owns the one conversion and geometry check used
 * by both call sites.
 */

#pragma once

#include "MoERuntimeTable.h"

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

    /**
     * @brief Export one complete, uniform routed-expert projection family.
     *
     * The three prepared engines must all export NativeVNNI descriptors or all
     * export the same floating precision.  Mixed families are rejected because
     * one captured grouped kernel and one migration byte contract own the
     * expert.  Existing placement metadata in @p output is preserved; only the
     * six pointer-bearing descriptors and @ref DeviceMoEExpertDescriptor::weight_format
     * are replaced after the complete triple validates.
     *
     * @param gate Prepared gate projection engine.
     * @param up Prepared up projection engine.
     * @param down Prepared down projection engine.
     * @param d_model Hidden width expected by gate/up and produced by down.
     * @param intermediate Expert intermediate width produced by gate/up.
     * @param output Runtime descriptor to populate.
     * @return True when a complete supported family with exact geometry exists.
     */
    inline bool exportDeviceMoEExpertWeightDescriptors(
        ITensorGemm *gate,
        ITensorGemm *up,
        ITensorGemm *down,
        int d_model,
        int intermediate,
        DeviceMoEExpertDescriptor &output) noexcept
    {
        if (!gate || !up || !down || d_model <= 0 || intermediate <= 0)
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
                !native_gate.valid() || !native_up.valid() || !native_down.valid() ||
                native_gate.n != intermediate || native_gate.k != d_model ||
                native_up.n != intermediate || native_up.k != d_model ||
                native_down.n != d_model || native_down.k != intermediate)
            {
                return false;
            }

            output.gate = native_gate;
            output.up = native_up;
            output.down = native_down;
            output.floating_gate = {};
            output.floating_up = {};
            output.floating_down = {};
            output.weight_format = DeviceMoEWeightFormat::NativeVNNI;
            return true;
        }

        std::array<ContiguousFloatingPointWeightDescriptor, 3> source{};
        if (!gate->exportContiguousFloatingPointWeights(source[0]) ||
            !up->exportContiguousFloatingPointWeights(source[1]) ||
            !down->exportContiguousFloatingPointWeights(source[2]))
        {
            return false;
        }

        std::array<DeviceMoEFloatingMatrixDesc, 3> floating{};
        std::array<DeviceMoEWeightFormat, 3> formats{};
        for (std::size_t projection = 0; projection < source.size(); ++projection)
        {
            if (!exportDeviceMoEFloatingMatrixDescriptor(
                    source[projection], floating[projection], formats[projection]))
            {
                return false;
            }
        }
        if (formats[0] != formats[1] || formats[0] != formats[2] ||
            floating[0].n != intermediate || floating[0].k != d_model ||
            floating[1].n != intermediate || floating[1].k != d_model ||
            floating[2].n != d_model || floating[2].k != intermediate)
        {
            return false;
        }

        output.gate = {};
        output.up = {};
        output.down = {};
        output.floating_gate = floating[0];
        output.floating_up = floating[1];
        output.floating_down = floating[2];
        output.weight_format = formats[0];
        return output.weightsReady();
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
