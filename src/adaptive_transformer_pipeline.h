// Adaptive Transformer Pipeline for Llaminar
//
// This implementation extends the MPI transformer pipeline with adaptive
// backend selection between OpenBLAS and COSMA based on operation characteristics.
//
// Key features:
// - Automatic backend selection for each matrix operation
// - Optimized prefill processing using COSMA for large sequences
// - Efficient single-token generation using OpenBLAS
// - Performance monitoring and optimization

#pragma once

#include "qwen_pipeline.h" // Provides QwenPipeline definition
#include "adaptive_matmul.h"
#include "logger.h"
#include <memory>
#include <chrono>
#include <cmath>
#include "utils/perf_counters.h"

namespace llaminar
{

    class AdaptiveTransformerPipeline : public QwenPipeline
    {
    private:
        std::unique_ptr<AdaptiveMatMulManager> matmul_manager_;
        bool enable_performance_logging_;

    public:
        AdaptiveTransformerPipeline(const ModelConfig &config, bool enable_logging = true)
            : QwenPipeline(config), enable_performance_logging_(enable_logging)
        {
            matmul_manager_ = std::make_unique<AdaptiveMatMulManager>();

            if (enable_performance_logging_ && getRank() == 0)
            {
                LOG_INFO("Initialized AdaptiveTransformerPipeline with hybrid OpenBLAS/COSMA backend");
            }
        }

        ~AdaptiveTransformerPipeline()
        {
            if (enable_performance_logging_)
            {
                matmul_manager_->printPerformanceSummary();
            }
        }

        // Override attention projection computation
        virtual std::shared_ptr<TensorBase> computeAttentionProjection(
            const std::shared_ptr<TensorBase> &input,
            const std::shared_ptr<TensorBase> &weight,
            bool is_prefill = false)
        {

            const auto &input_shape = input->shape();
            const auto &weight_shape = weight->shape();

            int seq_len = input_shape[0];
            int hidden_dim = input_shape[1];
            int output_dim = weight_shape[1];

            // Create output tensor
            std::vector<int> output_shape = {seq_len, output_dim};
            auto output = TensorFactory::create_simple(output_shape);

            // Use adaptive matrix multiplication
            bool success = matmul_manager_->multiply(
                input->data(), weight->data(), const_cast<float *>(output->data()),
                seq_len, output_dim, hidden_dim,
                false, false, 1.0f, 0.0f, is_prefill);

            if (!success)
            {
                throw std::runtime_error("Adaptive matrix multiplication failed in attention projection");
            }

            return output;
        }

        // Override attention computation with adaptive backend
        virtual std::shared_ptr<TensorBase> computeAttention(
            const std::shared_ptr<TensorBase> &query,
            const std::shared_ptr<TensorBase> &key,
            const std::shared_ptr<TensorBase> &value,
            bool is_prefill = false)
        {

            const auto &q_shape = query->shape();
            int seq_len = q_shape[0];
            int head_dim = q_shape[1];

            // Q * K^T for attention scores
            std::vector<int> scores_shape = {seq_len, seq_len};
            auto attention_scores = TensorFactory::create_simple(scores_shape);

            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            bool success = matmul_manager_->multiply(
                query->data(), key->data(), const_cast<float *>(attention_scores->data()),
                seq_len, seq_len, head_dim,
                false, true, scale, 0.0f, is_prefill // K^T with scaling
            );

            if (!success)
            {
                throw std::runtime_error("Adaptive matrix multiplication failed in attention scores");
            }

            // Apply softmax (distributed implementation)
            applySoftmax(attention_scores, seq_len);

            // Attention * V for output
            std::vector<int> output_shape = {seq_len, head_dim};
            auto attention_output = TensorFactory::create_simple(output_shape);
            success = matmul_manager_->multiply(
                attention_scores->data(), value->data(), const_cast<float *>(attention_output->data()),
                seq_len, head_dim, seq_len,
                false, false, 1.0f, 0.0f, is_prefill);

            if (!success)
            {
                throw std::runtime_error("Adaptive matrix multiplication failed in attention output");
            }

            return attention_output;
        }

        // Override FFN computation with SwiGLU and adaptive backend
        virtual std::shared_ptr<TensorBase> computeFFN(
            const std::shared_ptr<TensorBase> &input,
            const std::shared_ptr<TensorBase> &gate_weight,
            const std::shared_ptr<TensorBase> &up_weight,
            const std::shared_ptr<TensorBase> &down_weight,
            bool is_prefill = false)
        {

            const auto &input_shape = input->shape();
            int seq_len = input_shape[0];
            int hidden_dim = input_shape[1];
            int intermediate_dim = gate_weight->shape()[1];

            // Gate projection for SwiGLU
            std::vector<int> intermediate_shape = {seq_len, intermediate_dim};
            auto gate_proj = TensorFactory::create_simple(intermediate_shape);
            bool success = matmul_manager_->multiply(
                input->data(), gate_weight->data(), const_cast<float *>(gate_proj->data()),
                seq_len, intermediate_dim, hidden_dim,
                false, false, 1.0f, 0.0f, is_prefill);

            if (!success)
            {
                throw std::runtime_error("Adaptive matrix multiplication failed in FFN gate projection");
            }

            // Up projection for SwiGLU
            auto up_proj = TensorFactory::create_simple(intermediate_shape);
            success = matmul_manager_->multiply(
                input->data(), up_weight->data(), const_cast<float *>(up_proj->data()),
                seq_len, intermediate_dim, hidden_dim,
                false, false, 1.0f, 0.0f, is_prefill);

            if (!success)
            {
                throw std::runtime_error("Adaptive matrix multiplication failed in FFN up projection");
            }

            // Apply SwiGLU activation: gate_proj * silu(up_proj)
            applySwiGLU(gate_proj, up_proj);

            // Down projection
            std::vector<int> output_shape = {seq_len, hidden_dim};
            auto output = TensorFactory::create_simple(output_shape);
            success = matmul_manager_->multiply(
                gate_proj->data(), down_weight->data(), const_cast<float *>(output->data()),
                seq_len, hidden_dim, intermediate_dim,
                false, false, 1.0f, 0.0f, is_prefill);

            if (!success)
            {
                throw std::runtime_error("Adaptive matrix multiplication failed in FFN down projection");
            }

            return output;
        }

        // Enhanced token generation with adaptive prefill/inference phases
        virtual std::vector<int> generateTokens(const std::vector<int> &prompt_tokens,
                                                int max_new_tokens)
        {

            auto start_time = std::chrono::high_resolution_clock::now();

            if (enable_performance_logging_ && getRank() == 0)
            {
                LOG_INFO("Starting adaptive token generation: " << prompt_tokens.size()
                                                                << " prompt tokens, max " << max_new_tokens << " new tokens");
            }

            // Phase 1: Prefill - Process entire prompt with COSMA (if beneficial)
            auto prefill_start = std::chrono::high_resolution_clock::now();
            auto prefill_output = forwardPrefill(prompt_tokens, true); // is_prefill=true
            auto prefill_end = std::chrono::high_resolution_clock::now();

            double prefill_time = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();

            if (enable_performance_logging_ && getRank() == 0)
            {
                LOG_INFO("Prefill completed in " << prefill_time << " ms ("
                                                 << (prefill_time / prompt_tokens.size()) << " ms/token)");
            }

            // Phase 2: Autoregressive generation - Single token generation with OpenBLAS
            std::vector<int> generated_tokens;
            auto current_hidden = prefill_output;

            auto generation_start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < max_new_tokens; ++i)
            {
                auto token_start = std::chrono::high_resolution_clock::now();

                // Single token generation - optimized for OpenBLAS
                auto next_hidden = forwardSingleToken(current_hidden, false); // is_prefill=false

                // Sample next token
                int next_token = sampleToken(next_hidden);
                if (next_token == getEOSToken())
                {
                    if (enable_performance_logging_ && getRank() == 0)
                    {
                        LOG_INFO("Generation completed at EOS token after " << (i + 1) << " tokens");
                    }
                    break;
                }

                generated_tokens.push_back(next_token);
                current_hidden = next_hidden;

                auto token_end = std::chrono::high_resolution_clock::now();
                double token_time = std::chrono::duration<double, std::milli>(token_end - token_start).count();

                if (enable_performance_logging_ && getRank() == 0 && (i + 1) % 10 == 0)
                {
                    LOG_DEBUG("Generated " << (i + 1) << " tokens, last token: " << token_time << " ms");
                }
            }

            auto generation_end = std::chrono::high_resolution_clock::now();
            double generation_time = std::chrono::duration<double, std::milli>(generation_end - generation_start).count();
            double total_time = std::chrono::duration<double, std::milli>(generation_end - start_time).count();

            if (enable_performance_logging_ && getRank() == 0)
            {
                LOG_INFO("Generation phase completed in " << generation_time << " ms ("
                                                          << (generation_time / generated_tokens.size()) << " ms/token)");
                LOG_INFO("Total generation time: " << total_time << " ms for "
                                                   << (prompt_tokens.size() + generated_tokens.size()) << " total tokens");
                LOG_INFO("Throughput: " << (1000.0 * generated_tokens.size() / generation_time) << " tokens/sec");
            }

            return generated_tokens;
        }

        // Batch processing with adaptive backend selection
        std::vector<std::vector<int>> generateBatch(
            const std::vector<std::vector<int>> &prompt_batch,
            int max_new_tokens,
            int batch_size = 0)
        {

            if (batch_size == 0)
            {
                batch_size = prompt_batch.size();
            }

            std::vector<std::vector<int>> results;
            results.reserve(batch_size);

            for (size_t i = 0; i < std::min(static_cast<size_t>(batch_size), prompt_batch.size()); ++i)
            {
                // For batch processing, prefill is likely to benefit from COSMA
                auto result = generateTokens(prompt_batch[i], max_new_tokens);
                results.push_back(result);
            }

            return results;
        }

    private:
        // Apply SwiGLU activation in-place
        void applySwiGLU(std::shared_ptr<TensorBase> &gate,
                         const std::shared_ptr<TensorBase> &up)
        {
            float *gate_data = const_cast<float *>(gate->data());
            const float *up_data = up->data();

            size_t total_elements = gate->total_elements();

            // Apply SwiGLU: gate * silu(up)
            for (size_t i = 0; i < total_elements; ++i)
            {
                float x = up_data[i];
                float silu = x / (1.0f + std::exp(-x)); // SiLU activation
                gate_data[i] *= silu;
            }
        }

        // Distributed softmax implementation
        void applySoftmax(std::shared_ptr<TensorBase> &attention_scores, int seq_len)
        {
            float *data = const_cast<float *>(attention_scores->data());

            for (int i = 0; i < seq_len; ++i)
            {
                float *row = data + i * seq_len;

                // Find max for numerical stability
                float max_val = *std::max_element(row, row + seq_len);
                float global_max = max_val;

                // In MPI context, we'd use MPI_Allreduce for global max
                if (getSize() > 1)
                {
                    PerfBarrier(MPI_COMM_WORLD); // Ensure all ranks reach this point
                    PerfAllreduce(&max_val, &global_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
                }

                // Compute exp and sum
                float sum = 0.0f;
                for (int j = 0; j < seq_len; ++j)
                {
                    row[j] = std::exp(row[j] - global_max);
                    sum += row[j];
                }

                float global_sum = sum;
                if (getSize() > 1)
                {
                    PerfBarrier(MPI_COMM_WORLD); // Ensure all ranks complete local sum
                    PerfAllreduce(&sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                }

                // Normalize
                for (int j = 0; j < seq_len; ++j)
                {
                    row[j] /= global_sum;
                }
            }
        }

        // Process prefill phase (entire prompt at once)
        std::shared_ptr<TensorBase> forwardPrefill(const std::vector<int> &tokens, bool is_prefill)
        {
            // This would implement the full forward pass for the prompt
            // For now, return a placeholder
            std::vector<int> output_shape = {static_cast<int>(tokens.size()), getConfig().d_model};
            auto output = TensorFactory::create_simple(output_shape);
            // TODO: Implement full prefill logic
            return output;
        }

        // Process single token inference
        std::shared_ptr<TensorBase> forwardSingleToken(const std::shared_ptr<TensorBase> &input, bool is_prefill)
        {
            // This would implement the forward pass for a single token
            // For now, return a placeholder
            std::vector<int> output_shape = {1, getConfig().vocab_size};
            auto output = TensorFactory::create_simple(output_shape);
            // TODO: Implement single token logic
            return output;
        }

        // Sample next token from logits
        int sampleToken(const std::shared_ptr<TensorBase> &logits)
        {
            // Simple greedy sampling for now
            const float *data = logits->data();
            int vocab_size = logits->shape()[1];

            int best_token = 0;
            float best_score = data[0];

            for (int i = 1; i < vocab_size; ++i)
            {
                if (data[i] > best_score)
                {
                    best_score = data[i];
                    best_token = i;
                }
            }

            return best_token;
        }

        // Get EOS token ID (model-specific)
        int getEOSToken() const
        {
            return 151643; // Qwen EOS token
        }
    };

} // namespace llaminar