/**
 * @file BufferArena.cpp
 * @brief Implementation of the central buffer management system
 */

#include "BufferArena.h"
#include "CoherenceTracker.h"
#include "config/CollectiveBackendType.h"
#include "execution/debug/BufferRole.h"
#include "tensors/TensorClasses.h"
#include "tensors/TensorFactory.h"
#include "tensors/ITensor.h"
#include "models/qwen/Qwen2BufferSpec.h"
#include "utils/Logger.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace llaminar2
{

    // =========================================================================
    // Internal helpers
    // =========================================================================

    BufferArena::ManagedBuffer &BufferArena::buf(BufferId id)
    {
        auto idx = static_cast<size_t>(id);
        if (idx >= kBufferCount)
        {
            LOG_ERROR("BufferArena::buf() - BufferId out of range: " << bufferIdName(id)
                                                                     << " (index=" << idx << ", max=" << kBufferCount << ")");
        }
        assert(idx < kBufferCount && "BufferId out of range");
        auto &b = buffers_[idx];
        if (!b.registered)
        {
            LOG_ERROR("BufferArena::buf() - Buffer not registered: " << bufferIdName(id)
                                                                     << " (index=" << idx << ")");
        }
        assert(b.registered && "Buffer not registered");
        return b;
    }

    const BufferArena::ManagedBuffer &BufferArena::buf(BufferId id) const
    {
        auto idx = static_cast<size_t>(id);
        if (idx >= kBufferCount)
        {
            LOG_ERROR("BufferArena::buf() - BufferId out of range: " << bufferIdName(id)
                                                                     << " (index=" << idx << ", max=" << kBufferCount << ")");
        }
        assert(idx < kBufferCount && "BufferId out of range");
        auto &b = buffers_[idx];
        if (!b.registered)
        {
            LOG_ERROR("BufferArena::buf() - Buffer not registered: " << bufferIdName(id)
                                                                     << " (index=" << idx << ")");
        }
        assert(b.registered && "Buffer not registered");
        return b;
    }

    ITensor *BufferArena::ManagedBuffer::tensor() const
    {
        if (owned_tensor)
            return owned_tensor.get();
        return external_tensor;
    }

    TensorBase *BufferArena::ManagedBuffer::tensorBase() const
    {
        if (owned_tensor)
            return owned_tensor.get();
        return dynamic_cast<TensorBase *>(external_tensor);
    }

    // =========================================================================
    // Registration
    // =========================================================================

    bool BufferArena::registerBuffer(BufferId id, size_t rows, size_t cols,
                                     const char *dtype, DeviceId device)
    {
        auto idx = static_cast<size_t>(id);
        if (idx >= kBufferCount)
            return false;

        auto &b = buffers_[idx];
        if (b.registered)
        {
            LOG_WARN("BufferArena: buffer " << bufferIdName(id) << " already registered");
            return false;
        }

        b.registered = true;
        b.rows = rows;
        b.cols = cols;
        b.dtype = dtype;
        b.home_device = device;
        b.coherence = {}; // UNINITIALIZED
        return true;
    }

    bool BufferArena::registerExternalBuffer(BufferId id, ITensor *tensor)
    {
        auto idx = static_cast<size_t>(id);
        if (idx >= kBufferCount)
            return false;

        auto &b = buffers_[idx];
        if (b.registered)
        {
            LOG_WARN("BufferArena: buffer " << bufferIdName(id) << " already registered");
            return false;
        }

        b.registered = true;
        b.external_tensor = tensor;
        if (tensor)
        {
            b.rows = tensor->rows();
            b.cols = tensor->cols();
            // External buffers start as HOST-authoritative (weights loaded on CPU)
            b.coherence.authority = CoherenceState::HOST;
        }
        return true;
    }

    bool BufferArena::bindExternalBuffer(BufferId id, ITensor *tensor)
    {
        auto idx = static_cast<size_t>(id);
        if (idx >= kBufferCount || !tensor)
            return false;

        auto &b = buffers_[idx];
        if (b.owned_tensor || (b.registered && !b.external_tensor))
        {
            LOG_WARN("BufferArena: cannot bind external tensor over arena-owned buffer "
                     << bufferIdName(id));
            return false;
        }

        b.registered = true;
        b.external_tensor = tensor;
        b.rows = tensor->rows();
        b.cols = tensor->cols();
        b.dtype = nullptr;
        b.coherence = {};
        return true;
    }

    void BufferArena::registerAlias(BufferId a, BufferId b)
    {
        auto idx_a = static_cast<size_t>(a);
        auto idx_b = static_cast<size_t>(b);
        assert(idx_a < kBufferCount && idx_b < kBufferCount);

        auto &ba = buffers_[idx_a];
        auto &bb = buffers_[idx_b];

        // Assign to same alias group
        if (ba.alias_group >= 0 && bb.alias_group >= 0)
        {
            // Both already in groups — merge by rewriting all of b's group to a's
            int old_group = bb.alias_group;
            int new_group = ba.alias_group;
            for (auto &buf : buffers_)
            {
                if (buf.alias_group == old_group)
                    buf.alias_group = new_group;
            }
        }
        else if (ba.alias_group >= 0)
        {
            bb.alias_group = ba.alias_group;
        }
        else if (bb.alias_group >= 0)
        {
            ba.alias_group = bb.alias_group;
        }
        else
        {
            // Neither in a group — create new
            int g = next_alias_group_++;
            ba.alias_group = g;
            bb.alias_group = g;
        }
    }

    // =========================================================================
    // Tensor creation helper
    // =========================================================================

    /// Map BufferId to buffer name string.
    static const char *bufferIdToBufferName(BufferId id)
    {
        switch (id)
        {
        case BufferId::ATTN_PROJ:
            return "attn_proj";
        case BufferId::FFN_OUTPUT:
            return "ffn_output";
        case BufferId::HIDDEN_STATE:
            return "current_hidden";
        case BufferId::LOGITS:
            return "logits";
        case BufferId::ALL_POSITION_LOGITS:
            return "all_position_logits";
        case BufferId::ALL_POSITION_LOGITS_LOCAL:
            return "all_position_logits_local";
        default:
            return nullptr;
        }
    }

    /// Map buffer name string (from BufferDescriptor/BufferNames) to BufferId.
    /// Returns BufferId::_COUNT if no mapping exists.
    BufferId BufferArena::bufferNameToId(const std::string &name)
    {
        // Layer-level activation buffers
        if (name == "residual")
            return BufferId::RESIDUAL;
        if (name == "normalized")
            return BufferId::NORMALIZED;
        if (name == "Q")
            return BufferId::Q_PROJ;
        if (name == "K")
            return BufferId::K_PROJ;
        if (name == "V")
            return BufferId::V_PROJ;
        if (name == "attn_output")
            return BufferId::ATTN_OUTPUT;
        if (name == "attn_proj")
            return BufferId::ATTN_PROJ;
        if (name == "workspace_scores")
            return BufferId::ATTN_SCORES_WORKSPACE;
        if (name == "workspace_context")
            return BufferId::ATTN_CONTEXT_WORKSPACE;
        if (name == "workspace_mask")
            return BufferId::GEMM_WORKSPACE; // reuse for mask
        if (name == "lm_head_input_row")
            return BufferId::LM_HEAD_INPUT_ROW;
        if (name == "lm_head_input_rows")
            return BufferId::LM_HEAD_INPUT_ROWS;
        if (name == "gate")
            return BufferId::GATE_PROJ;
        if (name == "up")
            return BufferId::UP_PROJ;
        if (name == "ffn_output")
            return BufferId::FFN_OUTPUT;
        if (name == "Q_rope")
            return BufferId::Q_ROPE;
        if (name == "K_rope")
            return BufferId::K_ROPE;
        if (name == "V_dequant")
            return BufferId::V_DEQUANT;
        if (name == "all_position_logits")
            return BufferId::ALL_POSITION_LOGITS;
        if (name == "all_position_logits_local")
            return BufferId::ALL_POSITION_LOGITS_LOCAL;

        // GDN (Gated Delta Network) buffers
        if (name == "gdn_qkv")
            return BufferId::GDN_QKV;
        if (name == "gdn_z")
            return BufferId::GDN_Z;
        if (name == "gdn_alpha")
            return BufferId::GDN_ALPHA;
        if (name == "gdn_beta")
            return BufferId::GDN_BETA;

        // FA (Full Attention) extended buffers
        if (name == "fa_q_raw")
            return BufferId::FA_Q_RAW;
        if (name == "fa_gate")
            return BufferId::FA_GATE;

        // Model-level buffers
        if (name == "current_hidden" || name == "hidden")
            return BufferId::HIDDEN_STATE;
        if (name == "logits")
            return BufferId::LOGITS;
        if (name == "logits_local")
            return BufferId::LOGITS_LOCAL;

        // Prefix cache staging buffers
        if (name == "prefix_k_staging")
            return BufferId::PREFIX_K_STAGING;
        if (name == "prefix_v_staging")
            return BufferId::PREFIX_V_STAGING;
        if (name == "prefix_hybrid_state_staging")
            return BufferId::PREFIX_HYBRID_STATE_STAGING;
        if (name == "prefix_mtp_k_staging")
            return BufferId::PREFIX_MTP_K_STAGING;
        if (name == "prefix_mtp_v_staging")
            return BufferId::PREFIX_MTP_V_STAGING;
        if (name == "prefix_terminal_hidden")
            return BufferId::PREFIX_TERMINAL_HIDDEN;
        if (name == "prefix_terminal_logits")
            return BufferId::PREFIX_TERMINAL_LOGITS;

        // MTP sidecar graph buffers
        if (name == "mtp_embedding")
            return BufferId::MTP_EMBEDDING;
        if (name == "mtp_norm_hidden")
            return BufferId::MTP_NORM_HIDDEN;
        if (name == "mtp_norm_embedding")
            return BufferId::MTP_NORM_EMBEDDING;
        if (name == "mtp_concat")
            return BufferId::MTP_CONCAT;
        if (name == "mtp_projected")
            return BufferId::MTP_PROJECTED;
        if (name == "mtp_hidden")
            return BufferId::MTP_HIDDEN;
        if (name == "mtp_q")
            return BufferId::MTP_Q_PROJ;
        if (name == "mtp_k")
            return BufferId::MTP_K_PROJ;
        if (name == "mtp_v")
            return BufferId::MTP_V_PROJ;
        if (name == "mtp_q_raw")
            return BufferId::MTP_FA_Q_RAW;
        if (name == "mtp_q_gate")
            return BufferId::MTP_FA_GATE;
        if (name == "mtp_q_rope")
            return BufferId::MTP_Q_ROPE;
        if (name == "mtp_k_rope")
            return BufferId::MTP_K_ROPE;
        if (name == "mtp_attn_output")
            return BufferId::MTP_ATTN_OUTPUT;
        if (name == "mtp_attn_proj")
            return BufferId::MTP_ATTN_PROJ;
        if (name == "mtp_gate")
            return BufferId::MTP_GATE_PROJ;
        if (name == "mtp_up")
            return BufferId::MTP_UP_PROJ;
        if (name == "mtp_ffn_output")
            return BufferId::MTP_FFN_OUTPUT;
        if (name == "mtp_logits")
            return BufferId::MTP_LOGITS;

        return BufferId::_COUNT; // sentinel: no mapping
    }

    /// Convert BufferTensorType enum to dtype string for registerBuffer().
    const char *BufferArena::bufferTensorTypeToStr(BufferTensorType type)
    {
        switch (type)
        {
        case BufferTensorType::FP32:
            return "FP32";
        case BufferTensorType::FP16:
            return "FP16";
        case BufferTensorType::BF16:
            return "BF16";
        case BufferTensorType::Q8_1:
            return "Q8_1";
        case BufferTensorType::Q8_0:
            return "Q8_0";
        case BufferTensorType::Q16_1:
            return "Q16_1";
        case BufferTensorType::INT32:
            return "INT32";
        default:
            return "FP32"; // safe default
        }
    }

    std::shared_ptr<TensorBase> BufferArena::createTensorForBuffer(
        const ManagedBuffer &b, BufferId id) const
    {
        std::vector<size_t> shape{b.rows, b.cols};

        // ── Factory-based allocation (NUMA-aware, dtype-aware) ──────────
        if (config_.factory)
        {
            // Mapped memory for GPU FP32 tensors in snapshot/debugging mode
            if (config_.use_mapped_memory && b.home_device.is_gpu() &&
                b.dtype && std::string(b.dtype) == "FP32")
            {
                auto mapped = FP32Tensor::createMapped(shape, b.home_device);
                if (mapped && mapped->isMapped())
                {
                    LOG_DEBUG("[BufferArena] Created mapped FP32 tensor for '"
                              << bufferIdName(id) << "' on " << b.home_device.toString());
                    return mapped;
                }
                LOG_WARN("[BufferArena] Mapped allocation failed for '"
                         << bufferIdName(id) << "', using regular allocation");
            }

            // Dispatch by dtype string
            if (!b.dtype || std::string(b.dtype) == "FP32")
            {
                return config_.factory->createFP32(shape, b.home_device);
            }
            else if (std::string(b.dtype) == "FP16")
            {
                return config_.factory->createFP16(shape);
            }
            else if (std::string(b.dtype) == "BF16")
            {
                return config_.factory->createBF16(shape);
            }
            else if (std::string(b.dtype) == "Q8_1")
            {
                return config_.factory->createQ8_1(shape, b.home_device);
            }
            else if (std::string(b.dtype) == "Q16_1")
            {
                return config_.factory->createQ16_1(shape, b.home_device);
            }
            else if (std::string(b.dtype) == "INT32")
            {
                return config_.factory->createINT32(shape);
            }
            else
            {
                LOG_DEBUG("[BufferArena] Unknown dtype '" << b.dtype
                                                          << "' for " << bufferIdName(id)
                                                          << ", defaulting to FP32");
                return config_.factory->createFP32(shape, b.home_device);
            }
        }

        // ── Fallback: direct FP32 construction (no factory, no NUMA) ────
        return std::make_shared<FP32Tensor>(shape, b.home_device);
    }

    // =========================================================================
    // Allocation
    // =========================================================================

    bool BufferArena::allocate()
    {
        if (allocated_)
        {
            LOG_WARN("BufferArena: already allocated");
            return false;
        }

        stats_.reset();

        for (size_t i = 0; i < kBufferCount; ++i)
        {
            auto &b = buffers_[i];
            if (!b.registered)
                continue;
            if (b.external_tensor)
                continue; // External — already has storage
            if (b.owned_tensor)
                continue; // Should not happen

            auto bid = static_cast<BufferId>(i);
            auto tensor = createTensorForBuffer(b, bid);
            if (!tensor)
            {
                LOG_ERROR("[BufferArena] Failed to allocate buffer '"
                          << bufferIdName(bid) << "'");
                return false;
            }

            size_t bytes = tensor->size_bytes();
            stats_.total_buffers++;
            stats_.total_bytes += bytes;

            b.owned_tensor = tensor;
            b.coherence.authority = CoherenceState::UNINITIALIZED;
        }

        allocated_ = true;

        LOG_DEBUG("[BufferArena] Allocated " << stats_.total_buffers << " buffers, "
                                             << (stats_.total_bytes / 1024.0 / 1024.0) << " MB total");
        return true;
    }

    void BufferArena::logAllocationSummary() const
    {
        if (!allocated_)
            return;

        // Collect registered buffers, sorted by size descending
        struct BufInfo
        {
            const char *name;
            size_t rows;
            size_t cols;
            const char *dtype;
            size_t bytes;
        };
        std::vector<BufInfo> infos;
        infos.reserve(stats_.total_buffers);

        for (size_t i = 0; i < kBufferCount; ++i)
        {
            const auto &b = buffers_[i];
            if (!b.registered)
                continue;
            auto bid = static_cast<BufferId>(i);
            size_t bytes = 0;
            if (b.owned_tensor)
                bytes = b.owned_tensor->size_bytes();
            else if (b.external_tensor)
                bytes = b.external_tensor->size_bytes();
            infos.push_back({bufferIdName(bid), b.rows, b.cols,
                             b.dtype ? b.dtype : "ext", bytes});
        }

        std::sort(infos.begin(), infos.end(),
                  [](const BufInfo &a, const BufInfo &b)
                  { return a.bytes > b.bytes; });

        // Find max name length for alignment
        size_t max_name = 6; // "Buffer" header
        for (const auto &info : infos)
            max_name = std::max(max_name, std::strlen(info.name));

        // Log header + each buffer
        std::ostringstream summary;
        summary << "Activation buffer allocation plan (" << infos.size() << " buffers, "
                << std::fixed << std::setprecision(1)
                << (stats_.total_bytes / (1024.0 * 1024.0)) << " MB total):\n";

        // Header
        summary << "  " << std::left << std::setw(static_cast<int>(max_name + 2)) << "Buffer"
                << std::right << std::setw(14) << "Shape"
                << std::setw(8) << "Dtype"
                << std::setw(12) << "Size (MB)"
                << "\n";
        summary << "  " << std::string(max_name + 2 + 14 + 8 + 12, '-') << "\n";

        // Rows
        for (const auto &info : infos)
        {
            std::string shape_str = std::to_string(info.rows) + "x" + std::to_string(info.cols);
            double mb = info.bytes / (1024.0 * 1024.0);
            summary << "  " << std::left << std::setw(static_cast<int>(max_name + 2)) << info.name
                    << std::right << std::setw(14) << shape_str
                    << std::setw(8) << info.dtype
                    << std::setw(12) << std::fixed << std::setprecision(1) << mb
                    << "\n";
        }

        LOG_DEBUG("[BufferArena] " << summary.str());
    }

    // =========================================================================
    // Runtime coherence
    // =========================================================================

    bool BufferArena::prepareForRead(BufferId id, DeviceId target, void *stream)
    {
        auto &b = buf(id);
        return CoherenceTracker::prepareForRead(b.tensorBase(), b.coherence, target, stream);
    }

    bool BufferArena::prepareForWrite(BufferId id, DeviceId target, void *stream)
    {
        auto &b = buf(id);
        return CoherenceTracker::prepareForWrite(b.tensorBase(), b.coherence, target, stream);
    }

    void BufferArena::markWritten(BufferId id, DeviceId device, void *stream)
    {
        auto &b = buf(id);
        CoherenceTracker::markWrittenWithEvent(b.tensorBase(), b.coherence, device, stream);
    }

    void BufferArena::markWrittenFlagsOnly(BufferId id, DeviceId device)
    {
        auto &b = buf(id);
        CoherenceTracker::markWrittenFlagsOnly(b.tensorBase(), b.coherence, device);
    }

    // =========================================================================
    // Borrow tracking
    // =========================================================================

    void BufferArena::acquireReadBorrow(BufferId id)
    {
#ifndef NDEBUG
        validateBorrowSafe(id, BufferAccess::READ);
#endif
        buf(id).active_read_borrows++;
    }

    void BufferArena::acquireWriteBorrow(BufferId id)
    {
#ifndef NDEBUG
        validateBorrowSafe(id, BufferAccess::WRITE);
#endif
        buf(id).active_write_borrow = true;
    }

    void BufferArena::releaseReadBorrow(BufferId id)
    {
        auto &b = buf(id);
        assert(b.active_read_borrows > 0 && "Releasing read borrow that wasn't acquired");
        b.active_read_borrows--;
    }

    void BufferArena::releaseWriteBorrow(BufferId id)
    {
        auto &b = buf(id);
        assert(b.active_write_borrow && "Releasing write borrow that wasn't acquired");
        b.active_write_borrow = false;
    }

    bool BufferArena::validateNoBorrowsActive() const
    {
        for (size_t i = 0; i < kBufferCount; ++i)
        {
            const auto &b = buffers_[i];
            if (!b.registered)
                continue;
            if (b.active_read_borrows > 0 || b.active_write_borrow)
            {
                LOG_ERROR("BufferArena: buffer " << bufferIdName(static_cast<BufferId>(i))
                                                 << " has active borrows (read=" << b.active_read_borrows
                                                 << " write=" << b.active_write_borrow << ")");
                return false;
            }
        }
        return true;
    }

    void BufferArena::validateBorrowSafe(BufferId id, BufferAccess access) const
    {
        auto idx = static_cast<size_t>(id);
        const auto &b = buffers_[idx];

        if (access == BufferAccess::READ)
        {
            // Read borrows can coexist with other reads, but not with writes
            if (b.active_write_borrow)
            {
                LOG_ERROR("BufferArena: READ borrow on " << bufferIdName(id)
                                                         << " while WRITE borrow is active");
                assert(false && "active write borrow conflicts with read borrow");
            }
            // Check alias group for write borrows
            if (b.alias_group >= 0 && aliasGroupHasWriteBorrow(b.alias_group, id))
            {
                LOG_ERROR("BufferArena: READ borrow on " << bufferIdName(id)
                                                         << " while aliased buffer has WRITE borrow");
                assert(false && "aliased buffer has active write borrow");
            }
        }
        else
        {
            // Write borrows conflict with everything
            if (b.active_read_borrows > 0)
            {
                LOG_ERROR("BufferArena: WRITE borrow on " << bufferIdName(id)
                                                          << " while " << b.active_read_borrows
                                                          << " READ borrows are active");
                assert(false && "active read borrow conflicts with write borrow");
            }
            if (b.active_write_borrow)
            {
                LOG_ERROR("BufferArena: WRITE borrow on " << bufferIdName(id)
                                                          << " while another WRITE borrow is active");
                assert(false && "double write borrow");
            }
            // Check alias group for any borrows
            if (b.alias_group >= 0 && aliasGroupHasAnyBorrow(b.alias_group, id))
            {
                LOG_ERROR("BufferArena: WRITE borrow on " << bufferIdName(id)
                                                          << " while aliased buffer has active borrow");
                assert(false && "aliased buffer has active borrow conflicts with write");
            }
        }
    }

    bool BufferArena::aliasGroupHasWriteBorrow(int group, BufferId exclude) const
    {
        for (size_t i = 0; i < kBufferCount; ++i)
        {
            if (static_cast<BufferId>(i) == exclude)
                continue;
            const auto &b = buffers_[i];
            if (b.registered && b.alias_group == group && b.active_write_borrow)
                return true;
        }
        return false;
    }

    bool BufferArena::aliasGroupHasAnyBorrow(int group, BufferId exclude) const
    {
        for (size_t i = 0; i < kBufferCount; ++i)
        {
            if (static_cast<BufferId>(i) == exclude)
                continue;
            const auto &b = buffers_[i];
            if (b.registered && b.alias_group == group &&
                (b.active_read_borrows > 0 || b.active_write_borrow))
                return true;
        }
        return false;
    }

    // =========================================================================
    // Buffer access
    // =========================================================================

    ITensor *BufferArena::getTensor(BufferId id) const
    {
        if (!isRegistered(id))
            return nullptr;
        return buf(id).tensor();
    }

    std::shared_ptr<TensorBase> BufferArena::getSharedTensor(BufferId id) const
    {
        if (!isRegistered(id))
            return nullptr;
        return buf(id).owned_tensor; // nullptr for external buffers
    }

    void *BufferArena::getDevicePtr(BufferId id, DeviceId target) const
    {
        const auto &b = buf(id);
        auto *t = b.tensorBase();
        if (!t)
            return nullptr;

        if (target.is_gpu())
        {
            return t->gpu_data_ptr();
        }
        else
        {
            // raw_data() is the public const accessor for host memory
            return const_cast<void *>(t->raw_data());
        }
    }

    size_t BufferArena::getRows(BufferId id) const
    {
        return buf(id).rows;
    }

    size_t BufferArena::getCols(BufferId id) const
    {
        return buf(id).cols;
    }

    // =========================================================================
    // Introspection
    // =========================================================================

    CoherenceState BufferArena::getCoherenceState(BufferId id) const
    {
        return buf(id).coherence;
    }

    bool BufferArena::isRegistered(BufferId id) const
    {
        auto idx = static_cast<size_t>(id);
        if (idx >= kBufferCount)
            return false;
        return buffers_[idx].registered;
    }

    size_t BufferArena::registeredCount() const
    {
        size_t count = 0;
        for (const auto &b : buffers_)
        {
            if (b.registered)
                ++count;
        }
        return count;
    }

} // namespace llaminar2
