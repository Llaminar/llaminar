/**
 * @file LogitsGatherer.h
 * @brief Manages combined logits buffer and D2H gather/copy operations
 *
 * Extracted from RankOrchestrator to isolate logits gathering
 * responsibilities: buffer allocation/pinning, column-parallel D2H gather
 * from TP device runners, and PP stage logits copy.
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "../../../backends/DeviceId.h"

namespace llaminar2
{

    class TensorBase;
    class IBackend;
    class IInferenceRunner;
    struct DeviceId;
    struct LogitsLocalInfo;

    /**
     * @brief Manages the combined logits buffer and gather operations for multi-device inference.
     *
     * In tensor parallel (TP) mode with column-parallel LM head, each device produces a
     * partial logits shard [seq_len, vocab_local]. LogitsGatherer handles:
     * - Allocating the combined [max_tokens, vocab] buffer
     * - Pinning the buffer for zero-copy DMA from GPU
     * - D2H gathering partial logits into the combined buffer (fast decode + general prefill paths)
     * - Copying logits from PP stage runners
     * - Skip-gather control for GPU-side sampling
     */
    class LogitsGatherer
    {
    public:
        /// Backend resolver hook used by unit tests; production callers leave it null.
        using BackendResolver = IBackend *(*)(DeviceId);

        /**
         * @brief Construct a LogitsGatherer with a pre-allocated FP32 buffer.
         *
         * @param vocab_size Full vocabulary size (total across all TP devices)
         * @param max_tokens Maximum tokens (batch_size * max_seq_len)
         * @param backend_resolver Optional backend resolver for deterministic unit tests.
         *                         When null, LogitsGatherer uses the global BackendManager.
         */
        LogitsGatherer(int vocab_size, size_t max_tokens, BackendResolver backend_resolver = nullptr);

        ~LogitsGatherer();

        // Non-copyable
        LogitsGatherer(const LogitsGatherer &) = delete;
        LogitsGatherer &operator=(const LogitsGatherer &) = delete;

        // Movable
        LogitsGatherer(LogitsGatherer &&) noexcept;
        LogitsGatherer &operator=(LogitsGatherer &&) noexcept;

        // =========================================================================
        // Buffer pinning
        // =========================================================================

        /**
         * @brief Pin the combined logits buffer for DMA transfer.
         *
         * Page-locks the buffer memory via the GPU backend, enabling zero-copy
         * DMA without internal staging buffers (~50-100µs savings per D2H).
         * Call once after construction with a GPU device.
         *
         * @param device GPU device to pin for
         */
        void pinForDevice(const DeviceId &device);

        // =========================================================================
        // Gather operations
        // =========================================================================

        /**
         * @brief Gather logits from TP device runners into the combined buffer.
         *
         * Handles three paths:
         * 1. Single device / no column-parallel: memcpy from primary runner
         * 2. Column-parallel + decode (seq_len=1): fast D2H directly to offsets
         * 3. Column-parallel + prefill (seq_len>1): staging buffers + interleave
         *
         * @param runners Device runners to gather from
         * @param seq_len Actual sequence length for this forward pass
         * @param full_vocab_size Full vocabulary size for validation
         * @return true if gather succeeded
         */
        bool gather(const std::vector<std::unique_ptr<IInferenceRunner>> &runners,
                    size_t seq_len, int full_vocab_size);

        /**
         * @brief Gather already-resolved local logits shards into the combined buffer.
         *
         * This is used for logits surfaces that are not the main LOGITS_LOCAL buffer,
         * such as MTP sidecar logits. The copy semantics and CPU/GPU fallbacks are
         * identical to gather().
         */
        bool gatherLocalInfos(const std::vector<LogitsLocalInfo> &device_infos,
                              size_t seq_len,
                              int full_vocab_size);

        /**
         * @brief Copy logits from a PP stage runner into the combined buffer.
         *
         * @param stage_runner The stage runner with logits
         * @param fallback_copy_elements Fallback element count if no prior gather size
         * @param batch_size Batch size for buffer allocation
         * @param max_seq_len Max sequence length for buffer allocation
         */
        void copyFromStage(const IInferenceRunner &stage_runner,
                           size_t fallback_copy_elements,
                           int batch_size, int max_seq_len);

        // =========================================================================
        // Access
        // =========================================================================

        /// Get combined logits data pointer (may be nullptr if not allocated)
        const float *data() const;

        /// Get mutable data pointer for direct writes
        float *mutableData();

        /// Whether the buffer is allocated
        bool isAllocated() const;

        /// Number of elements in the buffer
        size_t bufferNumel() const;

        /// Actual size of the last gather/copy operation
        size_t lastGatheredSize() const { return last_gathered_size_; }

        // =========================================================================
        // Skip-gather control
        // =========================================================================

        void setSkipDecode(bool skip) { skip_decode_ = skip; }
        void setSkipPrefill(bool skip) { skip_prefill_ = skip; }
        bool skipDecode() const { return skip_decode_; }
        bool skipPrefill() const { return skip_prefill_; }

        /**
         * @brief Determine if a gather is needed for the given sequence length.
         *
         * @param seq_len Current sequence length
         * @return true if logits need to be gathered
         */
        bool needsGather(size_t seq_len) const;

    private:
        /// @brief Resolve a backend through the injected test resolver or global BackendManager.
        IBackend *resolveBackend(DeviceId device) const;

        std::unique_ptr<TensorBase> buffer_;
        BackendResolver backend_resolver_ = nullptr; ///< Optional test hook for backend selection.
        bool pinned_ = false;
        DeviceType pinned_device_type_ = DeviceType::CPU; ///< Backend type used for pinning (for correct unpin)
        bool skip_decode_ = false;
        bool skip_prefill_ = false;
        size_t last_gathered_size_ = 0;
    };

} // namespace llaminar2
