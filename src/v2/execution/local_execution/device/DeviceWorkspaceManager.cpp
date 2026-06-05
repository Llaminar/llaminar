/**
 * @file DeviceWorkspaceManager.cpp
 * @brief Per-device workspace buffer management implementation
 *
 * (Formerly GpuWorkspaceManager.cpp)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "DeviceWorkspaceManager.h"
#include "../../../backends/BackendManager.h"
#include "../../../backends/IBackend.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

namespace llaminar2
{

    // =========================================================================
    // Construction / Destruction
    // =========================================================================

    DeviceWorkspaceManager::DeviceWorkspaceManager(DeviceId device, size_t budget_bytes)
        : device_(device), budget_bytes_(budget_bytes)
    {
        LOG_DEBUG("[DeviceWorkspaceManager] Created for device " << device_.to_string()
                                                                 << " with budget " << budget_bytes_ << " bytes");
    }

    DeviceWorkspaceManager::~DeviceWorkspaceManager()
    {
        release();
    }

    // =========================================================================
    // Allocation
    // =========================================================================

    bool DeviceWorkspaceManager::allocate(const WorkspaceRequirements &requirements)
    {
        if (allocated_)
        {
            LOG_WARN("[DeviceWorkspaceManager] Already allocated, call release() first");
            return false;
        }

        // Handle empty requirements - still mark as allocated
        if (requirements.buffers.empty())
        {
            LOG_DEBUG("[DeviceWorkspaceManager] Empty requirements, marking as allocated with no buffers");
            PerfStatsCollector::addCounter(
                "memory",
                "workspace_allocate_requests",
                1.0,
                "allocate",
                device_.to_string(),
                {{"result", "empty"},
                 {"budget_bytes", std::to_string(budget_bytes_)},
                 {"buffer_count", "0"}});
            allocated_ = true;
            return true;
        }

        // Phase 1: Calculate total size needed with alignment
        size_t total_size = 0;
        for (const auto &buf : requirements.buffers)
        {
            // Align the current offset
            total_size = alignUp(total_size, buf.alignment);
            total_size += buf.size_bytes;
        }

        // Log all buffer requirements
        LOG_DEBUG("[DeviceWorkspaceManager] Workspace requirements (" << requirements.buffers.size() << " buffers):");
        for (const auto &buf : requirements.buffers)
        {
            LOG_DEBUG("[DeviceWorkspaceManager]   - " << buf.name << ": " << (buf.size_bytes / (1024 * 1024)) << " MB"
                                                      << (buf.required ? " (required)" : " (optional)"));
        }
        LOG_DEBUG("[DeviceWorkspaceManager] Total size needed: " << (total_size / (1024 * 1024)) << " MB, budget: " << (budget_bytes_ / (1024 * 1024)) << " MB");

        // Check budget
        if (total_size > budget_bytes_)
        {
            // Check if any required buffers exceed budget
            for (const auto &buf : requirements.buffers)
            {
                if (buf.required && buf.size_bytes > budget_bytes_)
                {
                    LOG_ERROR("[DeviceWorkspaceManager] Required buffer '" << buf.name
                                                                           << "' (" << buf.size_bytes << " bytes) exceeds budget ("
                                                                           << budget_bytes_ << " bytes)");
                    return false;
                }
            }

            // Recalculate with only buffers that fit
            total_size = 0;
            std::vector<const WorkspaceDescriptor *> fitting_buffers;
            for (const auto &buf : requirements.buffers)
            {
                size_t aligned_offset = alignUp(total_size, buf.alignment);
                size_t end_offset = aligned_offset + buf.size_bytes;

                if (end_offset <= budget_bytes_)
                {
                    fitting_buffers.push_back(&buf);
                    total_size = end_offset;
                }
                else if (buf.required)
                {
                    LOG_ERROR("[DeviceWorkspaceManager] Required buffer '" << buf.name
                                                                           << "' doesn't fit in remaining budget");
                    return false;
                }
                else
                {
                    LOG_DEBUG("[DeviceWorkspaceManager] Skipping optional buffer '" << buf.name
                                                                                    << "' (doesn't fit)");
                }
            }

            // If nothing fits, succeed with zero allocation
            if (fitting_buffers.empty())
            {
                LOG_DEBUG("[DeviceWorkspaceManager] No buffers fit in budget, marking as allocated with no buffers");
                allocated_ = true;
                return true;
            }

            // Allocate what fits
            return allocateBuffers(fitting_buffers, total_size);
        }

        // All buffers fit - allocate them all
        std::vector<const WorkspaceDescriptor *> all_buffers;
        for (const auto &buf : requirements.buffers)
        {
            all_buffers.push_back(&buf);
        }
        return allocateBuffers(all_buffers, total_size);
    }

    bool DeviceWorkspaceManager::allocateBuffers(
        const std::vector<const WorkspaceDescriptor *> &buffers,
        size_t total_size)
    {
        // Get backend for device
        IBackend *backend = getBackendFor(device_);
        if (!backend)
        {
            LOG_ERROR("[DeviceWorkspaceManager] No backend available for device " << device_.to_string());
            return false;
        }

        // Allocate single contiguous block
        int device_ordinal = device_.is_cpu() ? 0 : device_.ordinal;
        block_ = backend->allocate(total_size, device_ordinal);
        if (!block_)
        {
            LOG_ERROR("[DeviceWorkspaceManager] Failed to allocate " << total_size
                                                                     << " bytes on device " << device_.to_string());
            return false;
        }
        block_size_ = total_size;

        LOG_DEBUG("[WORKSPACE_ALLOC] block_ptr=" << block_
                                                 << " bytes=" << total_size
                                                 << " device=" << device_.to_string()
                                                 << " ordinal=" << device_ordinal);
        PerfStatsCollector::addCounter(
            "memory",
            "workspace_block_bytes",
            static_cast<double>(total_size),
            "allocate",
            device_.to_string(),
            {{"budget_bytes", std::to_string(budget_bytes_)},
             {"buffer_count", std::to_string(buffers.size())},
             {"bytes", std::to_string(total_size)}});

        // Suballocate buffers at aligned offsets
        size_t current_offset = 0;
        for (const auto *buf : buffers)
        {
            current_offset = alignUp(current_offset, buf->alignment);

            BufferInfo info;
            info.offset = current_offset;
            info.size = buf->size_bytes;
            buffers_[buf->name] = info;

            void *buf_ptr = static_cast<char *>(block_) + current_offset;
            LOG_DEBUG("[WORKSPACE_SUBALLOC] '" << buf->name << "'"
                                               << " ptr=" << buf_ptr
                                               << " offset=" << current_offset
                                               << " size=" << buf->size_bytes
                                               << " device=" << device_.to_string());
            PerfStatsCollector::addCounter(
                "memory",
                "workspace_suballoc_bytes",
                static_cast<double>(buf->size_bytes),
                "allocate",
                device_.to_string(),
                {{"name", buf->name},
                 {"required", buf->required ? "true" : "false"},
                 {"alignment", std::to_string(buf->alignment)},
                 {"offset_bytes", std::to_string(current_offset)},
                 {"bytes", std::to_string(buf->size_bytes)}});

            current_offset += buf->size_bytes;
        }

        used_bytes_ = current_offset;
        allocated_ = true;

        LOG_DEBUG("[DeviceWorkspaceManager] Allocated " << buffers_.size() << " buffers, "
                                                        << used_bytes_ << "/" << budget_bytes_ << " bytes used");
        return true;
    }

    void DeviceWorkspaceManager::release()
    {
        if (!allocated_)
        {
            return;
        }

        if (block_)
        {
            const size_t release_bytes = block_size_;
            const size_t release_buffer_count = buffers_.size();
            IBackend *backend = getBackendFor(device_);
            if (backend)
            {
                int device_ordinal = device_.is_cpu() ? 0 : device_.ordinal;
                backend->free(block_, device_ordinal);
                LOG_DEBUG("[DeviceWorkspaceManager] Released " << block_size_
                                                               << " bytes on device " << device_.to_string());
                PerfStatsCollector::addCounter(
                    "memory",
                    "workspace_release_bytes",
                    static_cast<double>(release_bytes),
                    "release",
                    device_.to_string(),
                    {{"buffer_count", std::to_string(release_buffer_count)},
                     {"bytes", std::to_string(release_bytes)}});
            }
            block_ = nullptr;
            block_size_ = 0;
        }

        buffers_.clear();
        used_bytes_ = 0;
        allocated_ = false;
    }

    // =========================================================================
    // Buffer Access
    // =========================================================================

    void *DeviceWorkspaceManager::getBuffer(const std::string &name) const
    {
        auto it = buffers_.find(name);
        if (it == buffers_.end())
        {
            return nullptr;
        }

        // Return block_ + offset
        return static_cast<char *>(block_) + it->second.offset;
    }

    size_t DeviceWorkspaceManager::getBufferSize(const std::string &name) const
    {
        auto it = buffers_.find(name);
        if (it == buffers_.end())
        {
            return 0;
        }
        return it->second.size;
    }

    bool DeviceWorkspaceManager::hasBuffer(const std::string &name) const
    {
        return buffers_.find(name) != buffers_.end();
    }

    std::vector<std::string> DeviceWorkspaceManager::bufferNames() const
    {
        std::vector<std::string> names;
        names.reserve(buffers_.size());
        for (const auto &pair : buffers_)
        {
            names.push_back(pair.first);
        }
        return names;
    }

    // =========================================================================
    // Static Helpers
    // =========================================================================

    size_t DeviceWorkspaceManager::alignUp(size_t offset, size_t alignment)
    {
        if (alignment == 0)
        {
            return offset;
        }
        return (offset + alignment - 1) & ~(alignment - 1);
    }

} // namespace llaminar2
