/**
 * @file LogitsGatherer.cpp
 * @brief Implementation of combined logits buffer management and D2H gather operations
 * @author David Sanftenberg
 * @date April 2026
 */

#include "LogitsGatherer.h"
#include "../../../backends/BackendManager.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/Logger.h"
#include "IInferenceRunner.h"

#include <cstring>
#include <vector>

namespace llaminar2
{

    LogitsGatherer::LogitsGatherer(int vocab_size, size_t max_tokens, BackendResolver backend_resolver)
        : backend_resolver_(backend_resolver)
    {
        if (vocab_size > 0 && max_tokens > 0)
        {
            buffer_ = std::make_unique<FP32Tensor>(
                std::vector<size_t>{max_tokens, static_cast<size_t>(vocab_size)});
            LOG_DEBUG("LogitsGatherer: Allocated buffer [" << max_tokens << ", " << vocab_size << "]");
        }
    }

    IBackend *LogitsGatherer::resolveBackend(DeviceId device) const
    {
        if (backend_resolver_)
            return backend_resolver_(device);
        return getBackendFor(device);
    }

    LogitsGatherer::~LogitsGatherer()
    {
        if (pinned_ && buffer_)
        {
            // Use the stored device type from pinForDevice() to select the correct
            // backend for unpinning. Previously this probed CUDA first, which caused
            // cudaHostUnregister failures when the buffer was actually pinned via HIP.
            DeviceId probe{pinned_device_type_, 0};
            IBackend *backend = resolveBackend(probe);
            if (backend)
            {
                backend->unpinHostMemory(buffer_->mutable_data());
                LOG_DEBUG("LogitsGatherer: Unpinned buffer via " << probe.toString());
            }
            pinned_ = false;
        }
    }

    LogitsGatherer::LogitsGatherer(LogitsGatherer &&) noexcept = default;
    LogitsGatherer &LogitsGatherer::operator=(LogitsGatherer &&) noexcept = default;

    void LogitsGatherer::pinForDevice(const DeviceId &device)
    {
        if (pinned_ || !buffer_ || !device.is_gpu())
            return;

        IBackend *backend = resolveBackend(device);
        if (!backend)
            return;

        size_t pin_bytes = buffer_->numel() * sizeof(float);
        if (backend->pinHostMemory(buffer_->mutable_data(), pin_bytes))
        {
            pinned_ = true;
            pinned_device_type_ = device.type; // Remember which backend pinned it
            LOG_DEBUG("LogitsGatherer: Pinned buffer (" << (pin_bytes / 1024) << " KB) for "
                                                        << device.toString());
        }
    }

    bool LogitsGatherer::gather(
        const std::vector<std::unique_ptr<IInferenceRunner>> &runners,
        size_t seq_len, int full_vocab_size)
    {
        if (!buffer_ || runners.empty())
            return false;

        // Single device — simple memcpy from primary runner
        if (runners.size() == 1)
        {
            const float *primary_logits = runners[0]->logits();
            if (primary_logits)
            {
                size_t copy_size = seq_len * static_cast<size_t>(full_vocab_size);
                std::memcpy(buffer_->mutable_data(), primary_logits,
                            copy_size * sizeof(float));
                last_gathered_size_ = copy_size;
            }
            return true;
        }

        // Check if column-parallel LM head is enabled
        bool has_column_parallel_lm_head = false;
        for (const auto &runner : runners)
        {
            if (runner && runner->hasLogitsLocal())
            {
                has_column_parallel_lm_head = true;
                break;
            }
        }

        if (!has_column_parallel_lm_head)
        {
            // LM head is replicated — use primary device's full logits
            const float *primary_logits = runners[0]->logits();
            if (primary_logits)
            {
                size_t copy_size = seq_len * static_cast<size_t>(full_vocab_size);
                std::memcpy(buffer_->mutable_data(), primary_logits,
                            copy_size * sizeof(float));
                last_gathered_size_ = copy_size;
            }
            return true;
        }

        // Column-parallel LM head: each device has logits_local [max_seq_len, vocab_local]
        // Gather along the vocab dimension (axis=1), producing [seq_len, vocab_total]

        // Phase 1: Validate all devices and collect metadata
        std::vector<LogitsLocalInfo> device_infos;
        device_infos.reserve(runners.size());

        for (const auto &runner : runners)
        {
            if (!runner)
            {
                LOG_ERROR("LogitsGatherer::gather: null device runner");
                return false;
            }

            auto info = runner->getLogitsLocalInfo();
            if (!info)
            {
                LOG_ERROR("LogitsGatherer::gather: device missing logits_local");
                return false;
            }

            if (info.vocab_local == 0)
            {
                LOG_ERROR("LogitsGatherer::gather: logits_local has zero vocab");
                return false;
            }

            device_infos.push_back(info);
        }

        // Calculate total vocab and validate output buffer
        size_t total_vocab = 0;
        for (const auto &info : device_infos)
            total_vocab += info.vocab_local;

        size_t expected_output_size = seq_len * total_vocab;
        if (buffer_->numel() < expected_output_size)
        {
            LOG_ERROR("LogitsGatherer::gather: output buffer too small. "
                      << "Need " << expected_output_size << ", have " << buffer_->numel());
            return false;
        }

        float *output = buffer_->mutable_data();

        // =================================================================
        // FAST PATH: Decode (seq_len=1) — D2H directly to combined buffer
        // =================================================================
        if (seq_len == 1)
        {
            size_t col_offset = 0;
            for (size_t dev = 0; dev < device_infos.size(); ++dev)
            {
                const auto &info = device_infos[dev];
                float *dst = output + col_offset;
                size_t copy_bytes = info.vocab_local * sizeof(float);

                if (info.gpu_ptr && info.device.has_value())
                {
                    IBackend *backend = resolveBackend(*info.device);
                    if (backend)
                    {
                        backend->deviceToHostFast(dst, info.gpu_ptr, copy_bytes,
                                                  info.device->gpu_ordinal());
                    }
                    else
                    {
                        std::memcpy(dst, info.tensor->data(), copy_bytes);
                    }
                }
                else
                {
                    std::memcpy(dst, info.tensor->data(), copy_bytes);
                }
                col_offset += info.vocab_local;
            }

            last_gathered_size_ = total_vocab;
            LOG_DEBUG("LogitsGatherer::gather: DECODE fast path — "
                      << device_infos.size() << " devices, " << total_vocab << " total vocab");
            return true;
        }

        // =================================================================
        // GENERAL PATH: Prefill (seq_len > 1) — staging buffers + interleave
        // =================================================================
        std::vector<std::vector<float>> staging_buffers(device_infos.size());
        std::vector<const float *> device_data(device_infos.size());

        for (size_t dev = 0; dev < device_infos.size(); ++dev)
        {
            const auto &info = device_infos[dev];

            if (info.gpu_ptr && info.device.has_value())
            {
                IBackend *backend = resolveBackend(*info.device);
                if (backend)
                {
                    size_t copy_bytes = seq_len * info.vocab_local * sizeof(float);
                    staging_buffers[dev].resize(seq_len * info.vocab_local);
                    backend->deviceToHost(staging_buffers[dev].data(), info.gpu_ptr,
                                          copy_bytes, info.device->gpu_ordinal());
                    device_data[dev] = staging_buffers[dev].data();
                }
                else
                {
                    LOG_WARN("LogitsGatherer::gather: no backend for device "
                             << info.device->toString() << ", falling back to full D2H");
                    device_data[dev] = info.tensor->data();
                }
            }
            else
            {
                device_data[dev] = info.tensor->data();
            }
        }

        // Interleave vocab slices into combined output
        for (size_t row = 0; row < seq_len; ++row)
        {
            size_t col_offset = 0;
            for (size_t dev = 0; dev < device_data.size(); ++dev)
            {
                const float *src = device_data[dev] + row * device_infos[dev].vocab_local;
                float *dst = output + row * total_vocab + col_offset;
                std::memcpy(dst, src, device_infos[dev].vocab_local * sizeof(float));
                col_offset += device_infos[dev].vocab_local;
            }
        }

        last_gathered_size_ = expected_output_size;

        LOG_DEBUG("LogitsGatherer::gather: gathered column-parallel logits "
                  << "[" << seq_len << ", " << total_vocab << "] from " << device_data.size() << " devices");

        return true;
    }

    void LogitsGatherer::copyFromStage(
        const IInferenceRunner &stage_runner,
        size_t fallback_copy_elements,
        int batch_size, int max_seq_len)
    {
        const float *stage_logits = stage_runner.logits();
        if (!stage_logits)
        {
            LOG_DEBUG("LogitsGatherer::copyFromStage: stage has no logits (may not have LM head)");
            return;
        }

        int vocab = stage_runner.vocab_size();
        if (vocab <= 0)
        {
            LOG_ERROR("LogitsGatherer::copyFromStage: Invalid vocab_size from stage");
            return;
        }

        // Allocate on demand if needed
        if (!buffer_)
        {
            size_t max_tokens = static_cast<size_t>(batch_size) * static_cast<size_t>(max_seq_len);
            buffer_ = std::make_unique<FP32Tensor>(
                std::vector<size_t>{max_tokens, static_cast<size_t>(vocab)});
            LOG_DEBUG("LogitsGatherer::copyFromStage: Allocated buffer ["
                      << max_tokens << ", " << vocab << "]");
        }

        size_t copy_elements = last_gathered_size_ > 0
                                   ? last_gathered_size_
                                   : (fallback_copy_elements > 0 ? fallback_copy_elements
                                                                 : static_cast<size_t>(vocab));
        std::memcpy(buffer_->mutable_data(), stage_logits, copy_elements * sizeof(float));

        LOG_DEBUG("LogitsGatherer::copyFromStage: Copied " << copy_elements << " elements");
    }

    const float *LogitsGatherer::data() const
    {
        return buffer_ ? buffer_->data() : nullptr;
    }

    float *LogitsGatherer::mutableData()
    {
        return buffer_ ? buffer_->mutable_data() : nullptr;
    }

    bool LogitsGatherer::isAllocated() const
    {
        return buffer_ != nullptr;
    }

    size_t LogitsGatherer::bufferNumel() const
    {
        return buffer_ ? buffer_->numel() : 0;
    }

    bool LogitsGatherer::needsGather(size_t seq_len) const
    {
        if (seq_len == 1)
            return !skip_decode_;
        return !skip_prefill_;
    }

} // namespace llaminar2
