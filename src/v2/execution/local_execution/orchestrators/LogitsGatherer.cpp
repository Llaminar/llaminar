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

#include <algorithm>
#include <cstring>
#include <vector>

namespace llaminar2
{
    namespace
    {
        size_t logitsRowStride(const LogitsLocalInfo &info)
        {
            return info.row_stride > 0 ? info.row_stride : info.vocab_local;
        }
    } // namespace


    LogitsGatherer::LogitsGatherer(int vocab_size, size_t max_tokens, BackendResolver backend_resolver)
        : vocab_size_(vocab_size > 0 ? static_cast<size_t>(vocab_size) : 0),
          backend_resolver_(backend_resolver)
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

        const size_t pin_elements = std::min(buffer_->numel(), vocab_size_);
        if (pin_elements == 0)
            return;

        size_t pin_bytes = pin_elements * sizeof(float);
        if (backend->pinHostMemory(buffer_->mutable_data(), pin_bytes))
        {
            pinned_ = true;
            pinned_device_type_ = device.type; // Remember which backend pinned it
            LOG_DEBUG("LogitsGatherer: Pinned decode row prefix (" << (pin_bytes / 1024)
                                                                    << " KB of "
                                                                    << ((buffer_->numel() * sizeof(float)) / 1024)
                                                                    << " KB buffer) for "
                                                                    << device.toString());
        }
    }

    bool LogitsGatherer::gatherLocalInfos(
        const std::vector<LogitsLocalInfo> &device_infos,
        size_t seq_len,
        int full_vocab_size)
    {
        if (!buffer_ || device_infos.empty())
            return false;

        for (const auto &info : device_infos)
        {
            if (!info)
            {
                LOG_ERROR("LogitsGatherer::gatherLocalInfos: device missing local logits tensor");
                return false;
            }
            if (info.vocab_local == 0)
            {
                LOG_ERROR("LogitsGatherer::gatherLocalInfos: local logits has zero vocab");
                return false;
            }
            if (logitsRowStride(info) < info.vocab_local)
            {
                LOG_ERROR("LogitsGatherer::gatherLocalInfos: local logits row stride "
                          << logitsRowStride(info) << " is smaller than local vocab "
                          << info.vocab_local);
                return false;
            }
        }

        size_t total_vocab = 0;
        for (const auto &info : device_infos)
            total_vocab += info.vocab_local;
        const bool use_explicit_vocab_offsets =
            std::any_of(
                device_infos.begin(),
                device_infos.end(),
                [](const LogitsLocalInfo &info)
                {
                    return info.vocab_offset != 0;
                });

        const bool has_full_vocab = full_vocab_size > 0;
        bool replicated_full_vocab = has_full_vocab && device_infos.size() > 1;
        if (replicated_full_vocab)
        {
            const size_t expected_vocab = static_cast<size_t>(full_vocab_size);
            for (const auto &info : device_infos)
            {
                if (info.vocab_local != expected_vocab)
                {
                    replicated_full_vocab = false;
                    break;
                }
            }
        }

        if (replicated_full_vocab)
        {
            const auto &primary = device_infos.front();
            const size_t copy_elements =
                seq_len * static_cast<size_t>(full_vocab_size);
            const size_t copy_bytes = copy_elements * sizeof(float);
            if (buffer_->numel() < copy_elements)
            {
                LOG_ERROR("LogitsGatherer::gatherLocalInfos: output buffer too small for replicated logits. "
                          << "Need " << copy_elements << ", have " << buffer_->numel());
                return false;
            }

            float *output = buffer_->mutable_data();
            if (primary.gpu_ptr && primary.device.has_value())
            {
                IBackend *backend = resolveBackend(*primary.device);
                if (backend)
                {
                    if (seq_len == 1)
                    {
                        backend->deviceToHostFast(output, primary.gpu_ptr, copy_bytes,
                                                  primary.device->gpu_ordinal());
                    }
                    else
                    {
                        backend->deviceToHost(output, primary.gpu_ptr, copy_bytes,
                                              primary.device->gpu_ordinal());
                    }
                }
                else
                {
                    std::memcpy(output, primary.tensor->data(), copy_bytes);
                }
            }
            else
            {
                std::memcpy(output, primary.tensor->data(), copy_bytes);
            }

            last_gathered_size_ = copy_elements;
            LOG_DEBUG("LogitsGatherer::gatherLocalInfos: gathered replicated full-vocab logits "
                      << "[" << seq_len << ", " << full_vocab_size << "] from primary of "
                      << device_infos.size() << " devices");
            return true;
        }

        if (full_vocab_size > 0 && total_vocab != static_cast<size_t>(full_vocab_size))
        {
            LOG_ERROR("LogitsGatherer::gatherLocalInfos: vocab shard total "
                      << total_vocab << " does not match full vocab " << full_vocab_size);
            return false;
        }
        if (use_explicit_vocab_offsets)
        {
            for (const auto &info : device_infos)
            {
                if (info.vocab_offset + info.vocab_local > total_vocab)
                {
                    LOG_ERROR("LogitsGatherer::gatherLocalInfos: vocab shard offset "
                              << info.vocab_offset << " plus local vocab "
                              << info.vocab_local << " exceeds total vocab "
                              << total_vocab);
                    return false;
                }
            }
        }

        size_t expected_output_size = seq_len * total_vocab;
        if (buffer_->numel() < expected_output_size)
        {
            LOG_ERROR("LogitsGatherer::gatherLocalInfos: output buffer too small. "
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
            for (const auto &info : device_infos)
            {
                const size_t dst_offset =
                    use_explicit_vocab_offsets ? info.vocab_offset : col_offset;
                float *dst = output + dst_offset;
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
            LOG_DEBUG("LogitsGatherer::gatherLocalInfos: DECODE fast path — "
                      << device_infos.size() << " devices, " << total_vocab << " total vocab");
            return true;
        }

        // =================================================================
        // GENERAL PATH: Prefill (seq_len > 1) — staging buffers + interleave
        // =================================================================
        std::vector<std::vector<float>> staging_buffers(device_infos.size());
        std::vector<const float *> device_data(device_infos.size());
        std::vector<size_t> device_strides(device_infos.size(), 0);

        for (size_t dev = 0; dev < device_infos.size(); ++dev)
        {
            const auto &info = device_infos[dev];
            const size_t row_stride = logitsRowStride(info);

            if (info.gpu_ptr && info.device.has_value())
            {
                IBackend *backend = resolveBackend(*info.device);
                if (backend)
                {
                    staging_buffers[dev].resize(seq_len * info.vocab_local);
                    if (row_stride == info.vocab_local)
                    {
                        size_t copy_bytes = seq_len * info.vocab_local * sizeof(float);
                        backend->deviceToHost(staging_buffers[dev].data(), info.gpu_ptr,
                                              copy_bytes, info.device->gpu_ordinal());
                    }
                    else
                    {
                        const auto *src_base = static_cast<const float *>(info.gpu_ptr);
                        for (size_t row = 0; row < seq_len; ++row)
                        {
                            backend->deviceToHost(
                                staging_buffers[dev].data() + row * info.vocab_local,
                                src_base + row * row_stride,
                                info.vocab_local * sizeof(float),
                                info.device->gpu_ordinal());
                        }
                    }
                    device_data[dev] = staging_buffers[dev].data();
                    device_strides[dev] = info.vocab_local;
                }
                else
                {
                    LOG_WARN("LogitsGatherer::gatherLocalInfos: no backend for device "
                             << info.device->toString() << ", falling back to full D2H");
                    device_data[dev] = info.tensor->data();
                    device_strides[dev] = row_stride;
                }
            }
            else
            {
                device_data[dev] = info.tensor->data();
                device_strides[dev] = row_stride;
            }
        }

        // Interleave vocab slices into combined output
        for (size_t row = 0; row < seq_len; ++row)
        {
            size_t col_offset = 0;
            for (size_t dev = 0; dev < device_data.size(); ++dev)
            {
                const float *src = device_data[dev] + row * device_strides[dev];
                const size_t dst_offset =
                    use_explicit_vocab_offsets
                        ? device_infos[dev].vocab_offset
                        : col_offset;
                float *dst = output + row * total_vocab + dst_offset;
                std::memcpy(dst, src, device_infos[dev].vocab_local * sizeof(float));
                col_offset += device_infos[dev].vocab_local;
            }
        }

        last_gathered_size_ = expected_output_size;

        LOG_DEBUG("LogitsGatherer::gatherLocalInfos: gathered column-parallel logits "
                  << "[" << seq_len << ", " << total_vocab << "] from " << device_data.size() << " devices");

        return true;
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

        return gatherLocalInfos(device_infos, seq_len, full_vocab_size);
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
        last_gathered_size_ = copy_elements;

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
