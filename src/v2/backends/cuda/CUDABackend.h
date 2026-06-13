/**
 * @file CUDABackend.h
 * @brief CUDA backend public API (no CUDA headers exposed)
 *
 * **Purpose**: Public interface for CUDA backend. Implementation lives in .cu file
 * to avoid exposing cuda_runtime.h to other compilation units.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../IBackend.h"
#include <cstdint>
#include <future>
#include <vector>

namespace llaminar2
{

    /**
     * @class CUDABackend
     * @brief CUDA compute backend implementation
     *
     * **Implementation**: See CUDABackend.cu
     * **Requirements**: NVIDIA GPU with CUDA Toolkit 11.0+
     * **Compilation**: Requires nvcc compiler, -DHAVE_CUDA=ON
     */
    class CUDABackend : public IBackend
    {
    public:
        CUDABackend();
        ~CUDABackend() override;

        // Memory transfer operations (see IBackend documentation)
        bool deviceToHost(void *dst, const void *src, size_t bytes, int device_id, void *stream = nullptr) override;
        bool hostToDevice(void *dst, const void *src, size_t bytes, int device_id, void *stream = nullptr) override;
        bool deviceToDevice(void *dst, const void *src, size_t bytes, int device_id, void *stream = nullptr) override;
        bool synchronize(int device_id) override;
        bool streamSynchronize(int device_id) override;
        bool setDevice(int device_id) override;

        // Async variants of transfer/sync operations
        std::future<bool> deviceToHostAsync(void *dst, const void *src, size_t bytes, int device_id) override;
        std::future<bool> hostToDeviceAsync(void *dst, const void *src, size_t bytes, int device_id) override;
        std::future<bool> synchronizeAsync(int device_id) override;

        // Event operations (fine-grained synchronization)
        void *createEvent(int device_id) override;
        void destroyEvent(void *event, int device_id) override;
        bool recordEvent(void *event, int device_id, void *stream = nullptr) override;
        bool waitForEvent(void *event, int device_id) override;

        // Memory allocation operations
        void *allocate(size_t bytes, int device_id) override;
        void free(void *ptr, int device_id) override;
        bool memset(void *ptr, int value, size_t bytes, int device_id, void *stream = nullptr) override;

        // Async variants of allocation operations
        std::future<void *> allocateAsync(size_t bytes, int device_id) override;
        std::future<void> freeAsync(void *ptr, int device_id) override;
        std::future<bool> memsetAsync(void *ptr, int value, size_t bytes, int device_id) override;

        // Zero-copy mapped memory operations
        void *allocateMapped(size_t bytes, int device_id, void **device_ptr) override;
        void freeMapped(void *host_ptr, int device_id) override;

        // Device query operations
        int deviceCount() const override;
        std::string backendName() const override;
        std::string deviceName(int device_id) const override;
        size_t deviceMemoryTotal(int device_id) const override;
        size_t deviceMemoryFree(int device_id) const override;

        // Host memory pinning for async DMA
        bool pinHostMemory(void *ptr, size_t bytes) override;
        bool unpinHostMemory(void *ptr) override;

        // GPU-side argmax for greedy sampling
        bool argmaxF32(const void *data_device, int n, int device_id,
                       float *out_value, int *out_index, void *stream = nullptr,
                       void *partial_vals = nullptr, void *partial_idxs = nullptr,
                       int partial_capacity = 0) override;
        bool argmaxF32BatchedRows(const void *data_device, int rows, int cols, int device_id,
                                  float *out_values, int *out_indices, void *stream = nullptr,
                                  void *partial_vals = nullptr, void *partial_idxs = nullptr,
                                  int partial_capacity = 0) override;
        bool enqueueArgmaxF32BatchedRowsDevice(
            const void *data_device,
            int rows,
            int cols,
            int device_id,
            void *stream,
            void *out_values_device,
            void *out_indices_device,
            void *partial_vals = nullptr,
            void *partial_idxs = nullptr,
            int partial_capacity = 0) override;

        // GPU-side top-k selection for sampling
        bool topKF32(const void *data_device, int n, int k, int device_id,
                     float *out_values, int *out_indices, void *stream = nullptr) override;
        bool sampleTopKTopPF32(const void *data_device, int n,
                               int top_k, float top_p, float temperature,
                               uint64_t rng_seed, uint64_t rng_offset,
                               int device_id, int *out_token,
                               void *stream = nullptr) override;
        bool enqueueSampleTopKTopPF32Device(const void *data_device, int n,
                                            int top_k, float top_p, float temperature,
                                            uint64_t rng_seed, uint64_t rng_offset,
                                            int device_id, void *stream,
                                            void *out_token_device) override;
        bool enqueueBuildTopKTopPDistributionF32Device(const void *data_device, int n,
                                                       int top_k, float top_p, float temperature,
                                                       int device_id, void *stream,
                                                       void *out_token_ids_device,
                                                       void *out_probs_device,
                                                       void *scratch_values_device = nullptr,
                                                       void *scratch_indices_device = nullptr,
                                                       int scratch_capacity = 0) override;
        bool enqueueBuildTopKTopPDistributionsF32Device(
            const void *data_device,
            int row_count,
            int n,
            int row_stride,
            int top_k,
            float top_p,
            float temperature,
            int device_id,
            void *stream,
            void *out_token_ids_device,
            int out_stride,
            void *out_probs_device,
            void *scratch_values_device = nullptr,
            void *scratch_indices_device = nullptr,
            int scratch_capacity = 0) override;
        bool enqueueBuildTopKTopPProcessedLogitsF32Device(
            const void *data_device,
            int row_count,
            int n,
            int row_stride,
            int top_k,
            float top_p,
            float temperature,
            int device_id,
            void *stream,
            void *out_logits_device,
            int out_row_stride,
            void *scratch_values_device = nullptr,
            void *scratch_indices_device = nullptr,
            int scratch_capacity = 0) override;
        bool enqueueSampleDistributionF32Device(
            const void *token_ids_device,
            const void *probs_device,
            int top_k,
            float threshold,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_probability_device = nullptr) override;
        bool enqueueSampleProcessedLogitsF32Device(
            const void *logits_device,
            int vocab_size,
            int row_stride,
            float threshold,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_probability_device = nullptr) override;
        bool enqueueSampleProcessedLogitsF32DeviceIfSpeculativeBatchNeedsBonus(
            const void *logits_device,
            int vocab_size,
            int row_stride,
            float threshold,
            const void *verify_tokens_device,
            const void *verify_accepted_device,
            int row_count,
            int first_token,
            const void *first_token_device,
            const int *stop_tokens_host,
            int stop_token_count,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_probability_device = nullptr) override;
        bool enqueueSoftmaxAndSampleTemperatureLogitsF32Device(
            const void *logits_device,
            int vocab_size,
            int row_stride,
            float temperature,
            float threshold,
            int device_id,
            void *stream,
            void *out_probabilities_device,
            int out_row_stride,
            void *out_token_device,
            void *out_probability_device = nullptr) override;
        bool enqueueScaleAndSampleTemperatureLogitsF32Device(
            const void *logits_device,
            int vocab_size,
            int row_stride,
            float temperature,
            float threshold,
            int device_id,
            void *stream,
            void *out_logits_device,
            int out_row_stride,
            void *out_token_device,
            void *out_probability_device = nullptr) override;
        bool enqueueSoftmaxProcessedLogitsF32Device(
            const void *logits_device,
            int row_count,
            int vocab_size,
            int row_stride,
            int device_id,
            void *stream,
            void *out_probabilities_device,
            int out_row_stride) override;
        bool enqueueFillInverseExponentialSamplesF32Device(
            void *out_samples_device,
            int row_count,
            int vocab_size,
            int row_stride,
            uint64_t seed,
            int first_logical_position,
            int device_id,
            void *stream) override;
        bool enqueueSpeculativeVerifyDistributionsF32Device(
            const void *target_token_ids_device,
            const void *target_probs_device,
            const void *draft_token_ids_device,
            const void *draft_probs_device,
            int top_k,
            int draft_token,
            uint64_t accept_seed,
            uint64_t accept_offset,
            uint64_t residual_seed,
            uint64_t residual_offset,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr) override;
        bool enqueueSpeculativeVerifyDistributionsF32DeviceThresholds(
            const void *target_token_ids_device,
            const void *target_probs_device,
            const void *draft_token_ids_device,
            const void *draft_probs_device,
            int top_k,
            int draft_token,
            float accept_threshold,
            float residual_threshold,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr) override;
        bool enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatch(
            const void *target_token_ids_device,
            const void *target_probs_device,
            const void *draft_token_ids_device,
            const void *draft_probs_device,
            int top_k,
            int distribution_stride,
            const int *draft_tokens_host,
            const float *accept_thresholds_host,
            const float *residual_thresholds_host,
            int row_count,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr) override;
        bool enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
            const void *target_token_ids_device,
            const void *target_probs_device,
            const void *draft_token_ids_device,
            const void *draft_probs_device,
            int top_k,
            int distribution_stride,
            const void *draft_tokens_device,
            const float *accept_thresholds_host,
            const float *residual_thresholds_host,
            int row_count,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr,
            const void *draft_token_probabilities_device = nullptr) override;
        bool enqueueSpeculativeVerifyProcessedLogitsF32DeviceThresholdsBatchDeviceTokens(
            const void *target_logits_device,
            const void *draft_logits_device,
            int row_count,
            int vocab_size,
            int target_row_stride,
            int draft_row_stride,
            const void *draft_tokens_device,
            const float *accept_thresholds_host,
            const float *residual_thresholds_host,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr,
            const void *draft_token_probabilities_device = nullptr) override;
        bool enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
            const void *target_logits_device,
            const void *draft_probabilities_device,
            int row_count,
            int vocab_size,
            int target_row_stride,
            int draft_row_stride,
            const void *draft_tokens_device,
            const float *accept_thresholds_host,
            uint64_t inverse_sample_seed,
            int inverse_sample_first_logical_position,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr,
            bool no_draft_probabilities = false) override;
        bool enqueueSpeculativeVerifyProcessedTargetDraftLogitsF32DeviceThresholdsBatchDeviceTokens(
            const void *target_logits_device,
            const void *draft_logits_device,
            int row_count,
            int vocab_size,
            int target_row_stride,
            int draft_row_stride,
            const void *draft_tokens_device,
            const float *accept_thresholds_host,
            uint64_t inverse_sample_seed,
            int inverse_sample_first_logical_position,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr,
            const void *draft_token_probabilities_device = nullptr) override;
        bool enqueueSpeculativeVerifyProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
            const void *target_probabilities_device,
            const void *draft_probabilities_device,
            const void *inverse_rejection_samples_device,
            int row_count,
            int vocab_size,
            int target_row_stride,
            int draft_row_stride,
            int inverse_sample_row_stride,
            const void *draft_tokens_device,
            const float *accept_thresholds_host,
            int device_id,
            void *stream,
            void *out_token_device,
            void *out_accepted_device,
            void *out_accept_probability_device = nullptr,
            void *out_accept_threshold_device = nullptr,
            bool no_draft_probabilities = false) override;
        bool enqueueSummarizeSpeculativeVerifyBatch(
            const void *verify_tokens_device,
            const void *verify_accepted_device,
            int row_count,
            int first_token,
            const int *stop_tokens_host,
            int stop_token_count,
            const void *bonus_token_device,
            bool has_bonus_token,
            int device_id,
            void *stream,
            void *out_tokens_device,
            void *out_meta_device) override;
        bool enqueueSummarizeSpeculativeVerifyBatchDeviceFirstToken(
            const void *verify_tokens_device,
            const void *verify_accepted_device,
            int row_count,
            const void *first_token_device,
            const int *stop_tokens_host,
            int stop_token_count,
            const void *bonus_token_device,
            bool has_bonus_token,
            int device_id,
            void *stream,
            void *out_tokens_device,
            void *out_meta_device) override;
        bool enqueueSummarizeGreedySpeculativeVerifyBatch(
            const void *verify_tokens_device,
            const void *draft_tokens_device,
            int compare_row_count,
            int first_token,
            const int *stop_tokens_host,
            int stop_token_count,
            int device_id,
            void *stream,
            void *out_tokens_device,
            void *out_meta_device) override;

        // GPU-side sparse logit penalty application
        bool applyLogitPenaltiesF32(void *logits_device,
                                    const int *token_ids_host,
                                    const float *penalties_host,
                                    int num_penalties, int vocab_size,
                                    int device_id, void *stream = nullptr) override;
        bool enqueueLogitPenaltiesF32Device(void *logits_device,
                                            const void *token_ids_device,
                                            const void *penalties_device,
                                            int num_penalties, int vocab_size,
                                            int device_id, void *stream) override;

        // Capability queries
        bool supportsBF16(int device_id) const override;
        bool supportsFP16(int device_id) const override;
        bool supportsINT8(int device_id) const override;

        // Compute operations
        bool gemmIQ4NL(
            const void *A_device,
            const void *B_device,
            void *C_device,
            int m,
            int n,
            int k,
            int device_id) override;

        // Stream management
        void *createStream(int device_id) override;
        void destroyStream(void *stream, int device_id) override;
        bool synchronizeStream(void *stream, int device_id) override;
        bool streamWaitEvent(void *stream, void *event, int device_id) override;

        // Async H2D without sync (for pipelined loading)
        bool hostToDeviceOnStream(void *dst, const void *src, size_t bytes,
                                  int device_id, void *stream) override;
        bool deviceToHostOnStream(void *dst, const void *src, size_t bytes,
                                  int device_id, void *stream) override;

        // Pinned host memory
        void *allocatePinned(size_t bytes, int device_id) override;
        void freePinned(void *ptr, int device_id) override;

        // Stream-aware memory operations
        bool deviceCopyAsync(void *dst, const void *src, size_t bytes,
                             int device_id, void *stream = nullptr) override;

        // Collective reduction primitives
        bool vectorAddInplace(void *output, const void *input, size_t count,
                              int element_size, int device_id, void *stream = nullptr) override;

        // Backend identity
        DeviceType backendDeviceType() const override { return DeviceType::CUDA; }

    private:
        int device_count_;

        // Per-device argmax result buffers (lazily allocated)
        struct ArgmaxDeviceBuffers
        {
            void *value_ptr = nullptr;
            void *index_ptr = nullptr;
            int allocated_count = 0;
        };
        std::vector<ArgmaxDeviceBuffers> argmax_buffers_;

        // Per-device top-k result buffers (lazily allocated)
        struct TopKDeviceBuffers
        {
            void *values_ptr = nullptr;
            void *indices_ptr = nullptr;
            int allocated_k = 0;
        };
        std::vector<TopKDeviceBuffers> topk_buffers_;

        // Per-device sampled-token result buffers (lazily allocated)
        struct SampleTokenDeviceBuffers
        {
            void *token_ptr = nullptr; // int on device
        };
        std::vector<SampleTokenDeviceBuffers> sample_token_buffers_;

        // Per-device penalty upload buffers (lazily allocated)
        struct PenaltyDeviceBuffers
        {
            void *token_ids_ptr = nullptr;   // int[] on device
            void *penalties_ptr = nullptr;    // float[] on device
            int allocated_count = 0;
        };
        std::vector<PenaltyDeviceBuffers> penalty_buffers_;
    };

} // namespace llaminar2
