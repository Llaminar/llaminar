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
#include "../../../utils/VramBillOfMaterials.h"

#include <algorithm>
#include <atomic>
#include <cstdint>

namespace llaminar2
{
    namespace
    {
        std::atomic<uint64_t> g_next_workspace_manager_id{1};

        size_t mixPublicationHash(size_t seed, std::uint64_t word) noexcept
        {
            const size_t value = std::hash<std::uint64_t>{}(word);
            return seed ^ (value + size_t{0x9e3779b9U} +
                           (seed << 6U) + (seed >> 2U));
        }
    }

    size_t PersistentWorkspacePublicationKeyHash::operator()(
        const PersistentWorkspacePublicationKey &key) const noexcept
    {
        size_t hash = 0;
        hash = mixPublicationHash(hash, key.word0);
        hash = mixPublicationHash(hash, key.word1);
        hash = mixPublicationHash(hash, key.word2);
        hash = mixPublicationHash(hash, key.word3);
        return hash;
    }

    // =========================================================================
    // Construction / Destruction
    // =========================================================================

    PersistentWorkspaceSlotLease::~PersistentWorkspaceSlotLease()
    {
        auto registry = registry_.lock();
        if (!registry)
            return;

        std::lock_guard<std::mutex> lock(registry->mutex);
        auto domain_it = registry->occupied_slots.find(domain_);
        if (domain_it == registry->occupied_slots.end() ||
            slot_ >= domain_it->second.size())
        {
            return;
        }
        domain_it->second[slot_] = false;
    }

    DeviceWorkspaceManager::DeviceWorkspaceManager(DeviceId device, size_t budget_bytes)
        : device_(device),
          id_(g_next_workspace_manager_id.fetch_add(1, std::memory_order_relaxed)),
          budget_bytes_(budget_bytes)
    {
        LOG_DEBUG("[DeviceWorkspaceManager] Created for device " << device_.to_string()
                                                                 << " id=" << id_
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

    bool DeviceWorkspaceManager::extend(
        const WorkspaceRequirements &requirements)
    {
        if (!allocated_)
        {
            LOG_ERROR("[DeviceWorkspaceManager] extend called before initial allocation on "
                      << device_.to_string());
            return false;
        }

        std::vector<const WorkspaceDescriptor *> additions;
        additions.reserve(requirements.buffers.size());
        for (const auto &buffer : requirements.buffers)
        {
            const auto existing = buffers_.find(buffer.name);
            if (existing == buffers_.end() ||
                existing->second.size < buffer.size_bytes)
            {
                additions.push_back(&buffer);
            }
        }
        if (additions.empty())
            return true;

        size_t extension_size = 0;
        std::vector<const WorkspaceDescriptor *> fitting;
        fitting.reserve(additions.size());
        for (const WorkspaceDescriptor *buffer : additions)
        {
            const size_t aligned_offset =
                alignUp(extension_size, buffer->alignment);
            const bool fits =
                aligned_offset <= budget_bytes_ &&
                buffer->size_bytes <= budget_bytes_ - aligned_offset &&
                used_bytes_ <= budget_bytes_ -
                                   (aligned_offset + buffer->size_bytes);
            if (!fits)
            {
                if (buffer->required)
                {
                    LOG_ERROR("[DeviceWorkspaceManager] Append-only workspace extension for required buffer '"
                              << buffer->name << "' (" << buffer->size_bytes
                              << " bytes) exceeds remaining budget "
                              << remaining() << " on " << device_.to_string());
                    return false;
                }
                LOG_TRACE("[DeviceWorkspaceManager] Skipping optional append-only buffer '"
                          << buffer->name << "' because it exceeds remaining budget");
                continue;
            }

            fitting.push_back(buffer);
            extension_size = aligned_offset + buffer->size_bytes;
        }

        if (fitting.empty())
            return true;
        return allocateExtensionBuffers(fitting, extension_size);
    }

    bool DeviceWorkspaceManager::zeroAll(void *stream)
    {
        if (!allocated_)
        {
            LOG_ERROR("[DeviceWorkspaceManager] zeroAll called before allocation on "
                      << device_.to_string());
            return false;
        }
        if (device_.is_gpu() && !stream)
        {
            LOG_ERROR("[DeviceWorkspaceManager] zeroAll requires an explicit GPU stream on "
                      << device_.to_string());
            return false;
        }

        IBackend *backend = getBackendFor(device_);
        if (!backend)
        {
            LOG_ERROR("[DeviceWorkspaceManager] zeroAll has no backend for "
                      << device_.to_string());
            return false;
        }
        const int device_ordinal = device_.is_cpu() ? 0 : device_.ordinal;
        if (block_ && block_size_ > 0 &&
            !backend->memset(
                block_,
                0,
                block_size_,
                device_ordinal,
                stream))
        {
            return false;
        }
        for (const ExtensionBlock &extension : extension_blocks_)
        {
            if (extension.base && extension.size > 0 &&
                !backend->memset(
                    extension.base,
                    0,
                    extension.size,
                    device_ordinal,
                    stream))
            {
                return false;
            }
        }
        return true;
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

        if (!device_.is_cpu())
        {
            size_t max_alignment = 1;
            for (const auto *buf : buffers)
                max_alignment = std::max(max_alignment, buf->alignment);
            const auto block_addr = reinterpret_cast<std::uintptr_t>(block_);
            if ((block_addr & (max_alignment - 1)) != 0)
            {
                LOG_ERROR("[DeviceWorkspaceManager] Backend allocation for " << device_.to_string()
                                                                              << " returned block " << block_
                                                                              << " not aligned to " << max_alignment
                                                                              << " bytes");
                backend->free(block_, device_ordinal);
                block_ = nullptr;
                block_size_ = 0;
                return false;
            }
        }

        LOG_TRACE("[WORKSPACE_ALLOC] block_ptr=" << block_
                                                 << " bytes=" << total_size
                                                 << " device=" << device_.to_string()
                                                 << " ordinal=" << device_ordinal);
        logVramBomLine(
            "workspace_block",
            "device=" + device_.to_string() +
                " manager_id=" + std::to_string(id_) +
                " buffer_count=" + std::to_string(buffers.size()) +
                " budget_bytes=" + std::to_string(budget_bytes_) +
                " budget_mib=" + vramBomMiB(budget_bytes_) +
                " " + vramBomBytes(total_size));
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
            info.base = block_;
            info.offset = current_offset;
            info.size = buf->size_bytes;
            buffers_[buf->name] = info;

            void *buf_ptr = static_cast<char *>(block_) + current_offset;
            if (!device_.is_cpu())
            {
                const auto addr = reinterpret_cast<std::uintptr_t>(buf_ptr);
                if ((addr & (buf->alignment - 1)) != 0)
                {
                    LOG_ERROR("[DeviceWorkspaceManager] Workspace buffer '" << buf->name
                                                                            << "' on " << device_.to_string()
                                                                            << " is not aligned to " << buf->alignment
                                                                            << " bytes (ptr=" << buf_ptr
                                                                            << ", offset=" << current_offset << ")");
                    backend->free(block_, device_ordinal);
                    block_ = nullptr;
                    block_size_ = 0;
                    buffers_.clear();
                    used_bytes_ = 0;
                    return false;
                }
            }
            LOG_TRACE("[WORKSPACE_SUBALLOC] '" << buf->name << "'"
                                               << " ptr=" << buf_ptr
                                               << " offset=" << current_offset
                                               << " size=" << buf->size_bytes
                                               << " device=" << device_.to_string());
            logVramBomLine(
                "workspace_buffer",
                "device=" + device_.to_string() +
                    " manager_id=" + std::to_string(id_) +
                    " name=" + buf->name +
                    " required=" + (buf->required ? "true" : "false") +
                    " alignment=" + std::to_string(buf->alignment) +
                    " offset_bytes=" + std::to_string(current_offset) +
                    " " + vramBomBytes(buf->size_bytes));
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

        LOG_TRACE("[DeviceWorkspaceManager] Allocated " << buffers_.size() << " buffers, "
                                                        << used_bytes_ << "/" << budget_bytes_ << " bytes used");
        return true;
    }

    bool DeviceWorkspaceManager::allocateExtensionBuffers(
        const std::vector<const WorkspaceDescriptor *> &buffers,
        size_t total_size)
    {
        IBackend *backend = getBackendFor(device_);
        if (!backend)
        {
            LOG_ERROR("[DeviceWorkspaceManager] Cannot extend workspace without backend for "
                      << device_.to_string());
            return false;
        }
        if (total_size == 0 || used_bytes_ > budget_bytes_ - total_size)
        {
            LOG_ERROR("[DeviceWorkspaceManager] Invalid append-only extension size "
                      << total_size << " with " << remaining()
                      << " bytes remaining on " << device_.to_string());
            return false;
        }

        const int device_ordinal = device_.is_cpu() ? 0 : device_.ordinal;
        void *extension_base = backend->allocate(total_size, device_ordinal);
        if (!extension_base)
        {
            LOG_ERROR("[DeviceWorkspaceManager] Failed to allocate append-only workspace extension of "
                      << total_size << " bytes on " << device_.to_string());
            return false;
        }

        size_t max_alignment = 1;
        for (const WorkspaceDescriptor *buffer : buffers)
            max_alignment = std::max(max_alignment, buffer->alignment);
        if (!device_.is_cpu() &&
            (reinterpret_cast<std::uintptr_t>(extension_base) &
             (max_alignment - 1)) != 0)
        {
            LOG_ERROR("[DeviceWorkspaceManager] Append-only workspace extension on "
                      << device_.to_string() << " is not aligned to "
                      << max_alignment << " bytes");
            backend->free(extension_base, device_ordinal);
            return false;
        }

        size_t current_offset = 0;
        for (const WorkspaceDescriptor *buffer : buffers)
        {
            current_offset = alignUp(current_offset, buffer->alignment);
            void *buffer_ptr =
                static_cast<char *>(extension_base) + current_offset;
            if (!device_.is_cpu() &&
                (reinterpret_cast<std::uintptr_t>(buffer_ptr) &
                 (buffer->alignment - 1)) != 0)
            {
                LOG_ERROR("[DeviceWorkspaceManager] Append-only buffer '"
                          << buffer->name << "' on " << device_.to_string()
                          << " is not aligned to " << buffer->alignment
                          << " bytes");
                backend->free(extension_base, device_ordinal);
                return false;
            }

            buffers_[buffer->name] = BufferInfo{
                .base = extension_base,
                .offset = current_offset,
                .size = buffer->size_bytes,
            };
            LOG_TRACE("[WORKSPACE_APPEND_ONLY_SUBALLOC] '" << buffer->name
                                                           << "' ptr=" << buffer_ptr
                                                           << " offset=" << current_offset
                                                           << " size=" << buffer->size_bytes
                                                           << " device=" << device_.to_string());
            PerfStatsCollector::addCounter(
                "memory",
                "workspace_append_only_suballoc_bytes",
                static_cast<double>(buffer->size_bytes),
                "materialize",
                device_.to_string(),
                {{"name", buffer->name},
                 {"required", buffer->required ? "true" : "false"},
                 {"alignment", std::to_string(buffer->alignment)},
                 {"bytes", std::to_string(buffer->size_bytes)}});
            current_offset += buffer->size_bytes;
        }

        extension_blocks_.push_back({extension_base, total_size});
        used_bytes_ += total_size;
        PerfStatsCollector::addCounter(
            "memory",
            "workspace_append_only_extension_bytes",
            static_cast<double>(total_size),
            "materialize",
            device_.to_string(),
            {{"buffer_count", std::to_string(buffers.size())},
             {"bytes", std::to_string(total_size)}});
        return true;
    }

    void DeviceWorkspaceManager::release()
    {
        /*
         * Invalidate the ownership namespace before releasing device memory.
         * Existing leases contain weak references, so their later destruction
         * is harmless and cannot mutate the registry for a future allocation.
         */
        persistent_slot_registry_ =
            std::make_shared<detail::PersistentWorkspaceSlotRegistry>();

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
        if (!extension_blocks_.empty())
        {
            IBackend *backend = getBackendFor(device_);
            if (backend)
            {
                const int device_ordinal =
                    device_.is_cpu() ? 0 : device_.ordinal;
                for (const ExtensionBlock &extension : extension_blocks_)
                {
                    if (extension.base)
                        backend->free(extension.base, device_ordinal);
                }
            }
            extension_blocks_.clear();
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

        return static_cast<char *>(it->second.base) + it->second.offset;
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

    void *DeviceWorkspaceManager::getPersistentSlotBuffer(
        const std::string &name,
        size_t slot_capacity,
        size_t slot,
        size_t payload_bytes) const
    {
        if (name.empty() || slot_capacity == 0 || slot >= slot_capacity ||
            payload_bytes == 0)
        {
            LOG_ERROR("[DeviceWorkspaceManager] Invalid persistent slot address "
                      "request for buffer '" << name << "' (slots="
                      << slot_capacity << ", slot=" << slot
                      << ", payload_bytes=" << payload_bytes << ")");
            return nullptr;
        }

        void *base = getBuffer(name);
        const size_t total_bytes = getBufferSize(name);
        if (!base || total_bytes < slot_capacity)
        {
            LOG_ERROR("[DeviceWorkspaceManager] Persistent slot buffer '"
                      << name << "' is missing or too small (bytes="
                      << total_bytes << ", slots=" << slot_capacity << ")");
            return nullptr;
        }

        const size_t slot_stride = total_bytes / slot_capacity;
        if (payload_bytes > slot_stride)
        {
            LOG_ERROR("[DeviceWorkspaceManager] Persistent slot payload exceeds "
                      "the fixed stride in buffer '" << name << "' (payload="
                      << payload_bytes << ", stride=" << slot_stride
                      << ", slot=" << slot << ")");
            return nullptr;
        }

        const size_t offset = slot * slot_stride;
        if (offset > total_bytes || payload_bytes > total_bytes - offset)
        {
            LOG_ERROR("[DeviceWorkspaceManager] Persistent slot address exceeds "
                      "buffer '" << name << "' (offset=" << offset
                      << ", payload=" << payload_bytes
                      << ", bytes=" << total_bytes << ")");
            return nullptr;
        }

        return static_cast<char *>(base) + offset;
    }

    std::shared_ptr<PersistentWorkspaceSlotLease>
    DeviceWorkspaceManager::acquirePersistentSlot(
        const std::string &domain,
        size_t slot_capacity)
    {
        if (domain.empty() || slot_capacity == 0)
        {
            LOG_ERROR("[DeviceWorkspaceManager] Persistent slot acquisition "
                      "requires a non-empty domain and positive capacity");
            return nullptr;
        }

        auto registry = persistent_slot_registry_;
        std::lock_guard<std::mutex> lock(registry->mutex);
        auto &slots = registry->occupied_slots[domain];
        if (slots.empty())
        {
            slots.resize(slot_capacity, false);
        }
        else if (slots.size() != slot_capacity)
        {
            LOG_ERROR("[DeviceWorkspaceManager] Persistent slot domain '"
                      << domain << "' changed capacity from " << slots.size()
                      << " to " << slot_capacity);
            return nullptr;
        }

        for (size_t slot = 0; slot < slots.size(); ++slot)
        {
            if (slots[slot])
                continue;
            slots[slot] = true;
            return std::shared_ptr<PersistentWorkspaceSlotLease>(
                new PersistentWorkspaceSlotLease(registry, domain, slot));
        }

        LOG_ERROR("[DeviceWorkspaceManager] Persistent slot domain '"
                  << domain << "' exhausted all " << slot_capacity << " slots");
        return nullptr;
    }

    PersistentWorkspacePublicationResult
    DeviceWorkspaceManager::getOrCreatePersistentPublication(
        const std::string &domain,
        const PersistentWorkspacePublicationKey &key,
        size_t slot_capacity,
        const PersistentWorkspacePublicationFactory &factory)
    {
        if (domain.empty() || slot_capacity == 0 || !factory)
        {
            LOG_ERROR("[DeviceWorkspaceManager] Immutable publication requires "
                      "a non-empty domain, positive capacity, and factory");
            return {};
        }

        auto registry = persistent_slot_registry_;
        std::lock_guard<std::mutex> lock(registry->mutex);
        auto &slots = registry->occupied_slots[domain];
        if (slots.empty())
        {
            slots.resize(slot_capacity, false);
        }
        else if (slots.size() != slot_capacity)
        {
            LOG_ERROR("[DeviceWorkspaceManager] Persistent publication domain '"
                      << domain << "' changed capacity from " << slots.size()
                      << " to " << slot_capacity);
            return {};
        }

        auto &publications = registry->immutable_publications[domain];
        const auto existing = publications.find(key);
        if (existing != publications.end())
        {
            if (!existing->second.publication ||
                existing->second.slot >= slots.size() ||
                !slots[existing->second.slot])
            {
                LOG_ERROR("[DeviceWorkspaceManager] Immutable publication registry "
                          "is internally inconsistent for domain '" << domain << "'");
                return {};
            }
            return {
                .publication = existing->second.publication,
                .slot = existing->second.slot,
                .created = false,
            };
        }

        const auto free_slot = std::find(slots.begin(), slots.end(), false);
        if (free_slot == slots.end())
        {
            LOG_ERROR("[DeviceWorkspaceManager] Persistent publication domain '"
                      << domain << "' exhausted all " << slot_capacity << " slots");
            return {};
        }

        const size_t slot =
            static_cast<size_t>(std::distance(slots.begin(), free_slot));
        slots[slot] = true;

        std::shared_ptr<void> publication;
        try
        {
            publication = factory(slot);
        }
        catch (...)
        {
            slots[slot] = false;
            throw;
        }
        if (!publication)
        {
            slots[slot] = false;
            LOG_ERROR("[DeviceWorkspaceManager] Immutable publication factory "
                      "failed for domain '" << domain << "' slot " << slot);
            return {};
        }

        publications.emplace(
            key,
            detail::PersistentWorkspacePublicationRecord{
                .slot = slot,
                .publication = publication,
            });
        return {
            .publication = std::move(publication),
            .slot = slot,
            .created = true,
        };
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
