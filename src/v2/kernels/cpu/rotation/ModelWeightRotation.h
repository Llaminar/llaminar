/**
 * @file ModelWeightRotation.h
 * @brief Utility to tag model GEMM weights with block-diagonal rotation metadata
 *
 * Sets rotation metadata on GEMM weight tensors so that:
 *   1. CPUNativeVNNIWeightPacker fuses the rotation into one-time weight packing
 *      (dequant → rotate → requant to INT8 → pack into VNNI layout)
 *   2. CPUNativeVNNIGemmKernel rotates activations before Q8_1 quantization
 *
 * The original weight tensors are NEVER modified — they stay in their native
 * quantized format (Q4_0, IQ4_NL, Q8_0, etc.). Rotation is applied only during
 * the VNNI packing step, which already dequantizes blocks as part of its normal flow.
 *
 * Two rotations per model:
 *   - hidden_dim rotation (R_h): applied to QKV, Gate, Up, LM Head weights
 *   - ffn_dim rotation (R_f): applied to Down weight
 *
 * Usage:
 *   auto rotator = ModelWeightRotation::create(d_model, d_ff, block_dim);
 *   weight_mgr->setWeightPreprocessor(rotator->createPreprocessor());
 */

#pragma once

#include "ActivationRotation.h"
#include "loaders/IWeightManager.h"
#include "models/GraphTypes.h"
#include "utils/Logger.h"

#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    class ModelWeightRotation
    {
    public:
        /// Which rotation to apply
        enum RotationDim
        {
            HIDDEN, ///< hidden_dim rotation (QKV, Wo, Gate, Up, LM Head)
            FFN     ///< ffn_dim rotation (Down)
        };

        /**
         * @brief Create a model weight rotator.
         *
         * @param hidden_dim  Model hidden dimension (e.g., 2560)
         * @param ffn_dim     FFN intermediate dimension (e.g., 9216)
         * @param block_dim   Block size for rotation (e.g., 128)
         * @param seed        Random seed for rotation generation (default: 31)
         */
        static std::shared_ptr<ModelWeightRotation> create(
            int hidden_dim, int ffn_dim, int block_dim, uint64_t seed = 31)
        {
            auto rotator = std::make_shared<ModelWeightRotation>();
            rotator->hidden_rotation_ = std::make_shared<ActivationRotation>(
                hidden_dim, block_dim, seed);
            rotator->ffn_rotation_ = std::make_shared<ActivationRotation>(
                ffn_dim, block_dim, seed + 1); // Different seed for FFN
            return rotator;
        }

        /// Get the hidden_dim rotation
        const ActivationRotation *hiddenRotation() const { return hidden_rotation_.get(); }

        /// Get the ffn_dim rotation
        const ActivationRotation *ffnRotation() const { return ffn_rotation_.get(); }

        /**
         * @brief Tag a weight tensor with rotation metadata (no data copy).
         *
         * Sets setActivationRotation() on the tensor so that:
         *   - CPUNativeVNNIWeightPacker fuses rotation into weight packing
         *   - CPUNativeVNNIGemmKernel rotates activations at runtime
         *
         * @param weight_ptr  Pointer to the weight pointer
         * @param dim         Which rotation to use (HIDDEN or FFN)
         * @return true if rotation tag was set, false on error
         */
        bool rotateWeight(TensorBase **weight_ptr, RotationDim dim = HIDDEN)
        {
            if (!weight_ptr || !*weight_ptr)
                return false;

            TensorBase *tensor = *weight_ptr;
            const auto &rot = (dim == HIDDEN) ? hidden_rotation_ : ffn_rotation_;

            if (tensor->shape().size() != 2)
            {
                LOG_WARN("[ModelWeightRotation] Skipping non-2D tensor: "
                         << tensor->debugName());
                return false;
            }

            const int K = static_cast<int>(tensor->shape()[1]);
            if (K != rot->total_dim())
            {
                LOG_DEBUG("[ModelWeightRotation] Skipping rotation: weight K="
                          << K << " vs rotation dim=" << rot->total_dim()
                          << " for " << tensor->debugName());
                return false;
            }

            tensor->setActivationRotation(rot.get());

            LOG_DEBUG("[ModelWeightRotation] Tagged " << tensor->debugName()
                                                      << " [" << tensor->shape()[0] << "×" << K
                                                      << "] with rotation (block_dim="
                                                      << rot->block_dim() << ")");
            return true;
        }

        /**
         * @brief Tag all layer GEMM weights with rotation metadata.
         */
        void rotateLayerWeights(LayerWeights &layer)
        {
            if (layer.wq)
                rotateWeight(&layer.wq, HIDDEN);
            if (layer.wk)
                rotateWeight(&layer.wk, HIDDEN);
            if (layer.wv)
                rotateWeight(&layer.wv, HIDDEN);

            if (layer.attn_qkv)
                rotateWeight(&layer.attn_qkv, HIDDEN);
            if (layer.attn_gate)
                rotateWeight(&layer.attn_gate, HIDDEN);
            if (layer.ssm_alpha)
                rotateWeight(&layer.ssm_alpha, HIDDEN);
            if (layer.ssm_beta)
                rotateWeight(&layer.ssm_beta, HIDDEN);

            if (layer.gate_proj)
                rotateWeight(&layer.gate_proj, HIDDEN);
            if (layer.up_proj)
                rotateWeight(&layer.up_proj, HIDDEN);

            if (layer.down_proj)
                rotateWeight(&layer.down_proj, FFN);
        }

        /**
         * @brief Tag all weights in a ModelWeights structure with rotation metadata.
         */
        void rotateAllWeights(ModelWeights &weights, int n_layers,
                              std::shared_ptr<ModelWeightRotation> self)
        {
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < n_layers; ++i)
            {
                auto layer = weights.get_layer_weights(i);
                rotateLayerWeights(layer);
            }

            if (weights.lm_head)
                rotateWeight(&weights.lm_head, HIDDEN);

            auto end = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            LOG_DEBUG("[ModelWeightRotation] Tagged all weights for "
                     << n_layers << " layers in " << ms << " ms");
        }

        /**
         * @brief Create a WeightPreprocessor for use with IWeightManager::setWeightPreprocessor().
         *
         * Returns a callback that tags each GEMM weight tensor with the
         * appropriate rotation metadata via setActivationRotation(). The
         * original tensor is returned unchanged (no data copy, no format
         * conversion). The VNNI weight packer will fuse rotation into packing.
         *
         * The preprocessor captures shared_ptrs to the rotation matrices,
         * ensuring they outlive any tensor that references them via
         * setActivationRotation().
         */
        WeightPreprocessor createPreprocessor() const
        {
            auto hidden_rot = hidden_rotation_;
            auto ffn_rot = ffn_rotation_;

            return [hidden_rot, ffn_rot](
                       const std::string &name,
                       std::shared_ptr<TensorBase> tensor) -> std::shared_ptr<TensorBase>
            {
                const ActivationRotation *rot = nullptr;

                if (name.find("ffn_down.weight") != std::string::npos)
                {
                    rot = ffn_rot.get();
                }
                else if (name.find("attn_q.weight") != std::string::npos ||
                         name.find("attn_k.weight") != std::string::npos ||
                         name.find("attn_v.weight") != std::string::npos ||
                         name.find("attn_qkv.weight") != std::string::npos ||
                         name.find("attn_gate.weight") != std::string::npos ||
                         name.find("ssm_alpha.weight") != std::string::npos ||
                         name.find("ssm_beta.weight") != std::string::npos ||
                         name.find("ffn_gate.weight") != std::string::npos ||
                         name.find("ffn_up.weight") != std::string::npos ||
                         name == "output.weight")
                {
                    rot = hidden_rot.get();
                }

                if (!rot)
                    return tensor;

                if (!tensor || tensor->shape().size() != 2)
                    return tensor;

                const int K = static_cast<int>(tensor->shape()[1]);
                if (K != rot->total_dim())
                    return tensor;

                // Tag only — no data copy, no format conversion
                tensor->setActivationRotation(rot);

                LOG_DEBUG("[WeightPreprocessor] Tagged " << name
                                                         << " [" << tensor->shape()[0] << "×" << K
                                                         << "] with rotation");

                return tensor;
            };
        }

    private:
        std::shared_ptr<ActivationRotation> hidden_rotation_;
        std::shared_ptr<ActivationRotation> ffn_rotation_;
    };

} // namespace llaminar2
