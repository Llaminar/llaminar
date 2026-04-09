/**
 * @file ShortConv1dStage.h
 * @brief Short causal depthwise conv1d stage for GDN layers
 *
 * Applies a causal depthwise convolution (kernel_size=4 typically) followed
 * by SiLU activation on the mixed QKV projection output. Maintains a small
 * conv_state for incremental decode (kernel_size - 1 history frames).
 *
 * Prefill: conv1d over the full sequence, stores tail in conv_state
 * Decode:  conv1d_update using conv_state, outputs single timestep
 *
 * Delegates to ITensorShortConvolution kernel interface for the actual conv
 * computation, enabling device-specific implementations (CPU, CUDA).
 *
 * Reference: HuggingFace Qwen3_5GatedDeltaNet.forward() conv1d path
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"

#include <optional>

namespace llaminar2
{

    class ITensorShortConvolution;

    /**
     * @brief Causal depthwise conv1d + SiLU for GDN QKV preprocessing
     *
     * Computes: output = SiLU(DepthwiseConv1D(input, weight, bias))
     * with causal padding and conv_state management for decode.
     *
     * Delegates to ITensorShortConvolution* kernel for the actual computation.
     */
    class ShortConv1dStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            ITensor *input = nullptr;        ///< Input [seq_len, channels] (modified in-place for decode)
            ITensor *output = nullptr;       ///< Output [seq_len, channels]
            const ITensor *weight = nullptr; ///< Conv weight [channels, kernel_size] (squeezed from [channels, 1, kernel_size])
            const ITensor *bias = nullptr;   ///< Optional conv bias [channels]

            float *conv_state = nullptr; ///< Conv state buffer [channels, kernel_size-1] (from GDNLayerState)
            int seq_len = 0;             ///< Sequence length
            int channels = 0;            ///< Number of channels (= QKV dim)
            int kernel_size = 4;         ///< Convolution kernel width

            /// Kernel implementation (set during graph construction)
            ITensorShortConvolution *kernel = nullptr;

            // Optional BufferIds
            std::optional<BufferId> input_buffer_id;
            std::optional<BufferId> output_buffer_id;
        };

        static_assert(StageParamsRequired<Params>);

        explicit ShortConv1dStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::SHORT_CONV1D; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;

        void updateDynamicParams(int pos_offset, int seq_len) override
        {
            (void)pos_offset; // Conv1d doesn't use position offsets
            params_.seq_len = seq_len;
        }
        bool hasDynamicParams() const override { return true; }

        // Short conv1d operates fully on-device when GPU is active — graph-capturable
        bool isGraphCapturable() const override { return true; }

        const Params &getParams() const { return params_; }

    private:
        Params params_;
    };

} // namespace llaminar2
