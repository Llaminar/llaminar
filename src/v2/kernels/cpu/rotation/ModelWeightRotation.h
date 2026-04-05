/**
 * @file ModelWeightRotation.h
 * @brief Utility to apply block-diagonal rotation to model GEMM weights
 *
 * Creates rotated copies of all GEMM weight tensors and sets the rotation
 * metadata on each, so that CPUNativeVNNIGemmKernel can rotate activations
 * at quantization time.
 *
 * Two rotations per model:
 *   - hidden_dim rotation (R_h): applied to QKV, Wo, Gate, Up, LM Head weights
 *   - ffn_dim rotation (R_f): applied to Down weight
 *
 * Usage in graph config builder:
 *   auto rotator = ModelWeightRotation::create(d_model, d_ff, block_dim);
 *   rotator->rotateWeight(&layer.wq);
 *   rotator->rotateWeight(&layer.down_proj, ModelWeightRotation::FFN);
 */

#pragma once

#include "ActivationRotation.h"
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
         * @brief Rotate a weight tensor in-place (replaces the pointer).
         *
         * Creates a rotated Q8_0 copy:
         *   1. Dequantize each weight row to FP32
         *   2. Apply block-diagonal rotation along K-dimension
         *   3. Re-quantize to Q8_0
         *   4. Set rotation metadata on the new tensor
         *   5. Replace the pointer and keep the new tensor alive
         *
         * @param weight_ptr  Pointer to the weight pointer (will be updated)
         * @param dim         Which rotation to use (HIDDEN or FFN)
         * @return true if rotation was applied, false on error
         */
        bool rotateWeight(TensorBase **weight_ptr, RotationDim dim = HIDDEN)
        {
            if (!weight_ptr || !*weight_ptr)
                return false;

            TensorBase *original = *weight_ptr;
            const auto &rot = (dim == HIDDEN) ? hidden_rotation_ : ffn_rotation_;

            // Validate K-dimension matches rotation
            if (original->shape().size() != 2)
            {
                LOG_WARN("[ModelWeightRotation] Skipping non-2D tensor: "
                         << original->debugName());
                return false;
            }

            const int K = static_cast<int>(original->shape()[1]);
            if (K != rot->total_dim())
            {
                LOG_DEBUG("[ModelWeightRotation] Skipping rotation: weight K="
                          << K << " vs rotation dim=" << rot->total_dim()
                          << " for " << original->debugName());
                return false;
            }

            auto start = std::chrono::high_resolution_clock::now();

            // Create rotated copy
            auto rotated = rot->create_rotated_weight_generic(original);
            if (!rotated)
            {
                LOG_ERROR("[ModelWeightRotation] Failed to create rotated copy of "
                          << original->debugName());
                return false;
            }

            // Set rotation metadata on the new tensor
            rotated->setActivationRotation(rot.get());
            rotated->setDebugName(original->debugName() + "_rotated");

            auto end = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            LOG_DEBUG("[ModelWeightRotation] Rotated " << original->debugName()
                      << " [" << original->shape()[0] << "×" << K
                      << "] in " << ms << " ms");

            // Replace pointer and keep the shared_ptr alive
            *weight_ptr = rotated.get();
            owned_tensors_.push_back(std::move(rotated));

            return true;
        }

        /**
         * @brief Apply rotation to a full set of layer GEMM weights.
         *
         * Rotates all GEMM weights whose K-dimension matches a rotation:
         *   - HIDDEN: wq, wk, wv, wo, gate_proj, up_proj (K = hidden_dim)
         *   - HIDDEN: attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_out (GDN, K = hidden_dim)
         *   - FFN:    down_proj (K = ffn_dim)
         *
         * Skips null pointers and non-GEMM weights (norms, biases, conv1d).
         *
         * @param layer  LayerWeights struct (pointers will be updated in-place)
         */
        void rotateLayerWeights(LayerWeights &layer)
        {
            // FA attention projections (K-dim = hidden_dim)
            if (layer.wq) rotateWeight(&layer.wq, HIDDEN);
            if (layer.wk) rotateWeight(&layer.wk, HIDDEN);
            if (layer.wv) rotateWeight(&layer.wv, HIDDEN);
            // Note: wo is NOT rotated — its K-dim is n_heads*head_dim (attention
            // output space), not hidden_dim. The rotation target is the Q8_1
            // quantization of the post-RMSNorm hidden state, not the attention output.

            // GDN projections (K-dim = hidden_dim)
            if (layer.attn_qkv) rotateWeight(&layer.attn_qkv, HIDDEN);
            if (layer.attn_gate) rotateWeight(&layer.attn_gate, HIDDEN);
            if (layer.ssm_alpha) rotateWeight(&layer.ssm_alpha, HIDDEN);
            if (layer.ssm_beta) rotateWeight(&layer.ssm_beta, HIDDEN);
            // Note: ssm_out is NOT rotated — its K-dim is inner_size
            // (recurrence output space), not hidden_dim.

            // FFN Gate/Up projections (K-dim = hidden_dim)
            if (layer.gate_proj) rotateWeight(&layer.gate_proj, HIDDEN);
            if (layer.up_proj) rotateWeight(&layer.up_proj, HIDDEN);

            // FFN Down projection (K-dim = ffn_dim)
            if (layer.down_proj) rotateWeight(&layer.down_proj, FFN);
        }

        /**
         * @brief Apply rotation to all weights in a ModelWeights structure.
         *
         * Pre-rotates every GEMM weight in the model, building a lookup table
         * of original→rotated pointer mappings. Then wraps the layer accessor
         * to return rotated pointers transparently.
         *
         * Also rotates the LM head (global weight).
         *
         * IMPORTANT: The returned ModelWeights holds a shared_ptr to this
         * rotator (via the lambda), keeping all rotated tensors alive.
         *
         * @param weights  ModelWeights struct (lm_head and get_layer_weights will be updated)
         * @param n_layers Number of layers in the model
         * @param self     shared_ptr to this rotator (for lifecycle management)
         */
        void rotateAllWeights(ModelWeights &weights, int n_layers,
                              std::shared_ptr<ModelWeightRotation> self)
        {
            auto start = std::chrono::high_resolution_clock::now();

            // Build per-layer rotated weight caches
            rotated_layers_.resize(n_layers);
            for (int i = 0; i < n_layers; ++i)
            {
                rotated_layers_[i] = weights.get_layer_weights(i);
                rotateLayerWeights(rotated_layers_[i]);
            }

            // Rotate LM head (K-dim = hidden_dim)
            if (weights.lm_head)
                rotateWeight(&weights.lm_head, HIDDEN);

            // Wrap the layer accessor to return pre-rotated weights.
            // Captures shared_ptr to keep rotator (and all rotated tensors) alive.
            weights.get_layer_weights = [self](int layer_idx) -> LayerWeights
            {
                return self->rotated_layers_[layer_idx];
            };

            auto end = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            LOG_INFO("[ModelWeightRotation] Rotated all weights for "
                     << n_layers << " layers in " << ms << " ms");
        }

    private:
        std::shared_ptr<ActivationRotation> hidden_rotation_;
        std::shared_ptr<ActivationRotation> ffn_rotation_;

        /// Keeps rotated weight tensors alive
        std::vector<std::shared_ptr<TensorBase>> owned_tensors_;

        /// Per-layer rotated weight caches (populated by rotateAllWeights)
        std::vector<LayerWeights> rotated_layers_;
    };

} // namespace llaminar2
