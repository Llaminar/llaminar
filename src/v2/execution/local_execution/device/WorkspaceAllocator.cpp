/**
 * @file WorkspaceAllocator.cpp
 * @brief Implementation of standalone workspace allocation
 * @author David Sanftenberg
 * @date March 2026
 */

#include "WorkspaceAllocator.h"
#include "../graph/DeviceGraphExecutor.h"
#include "../../../backends/BackendManager.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../compute_stages/IComputeStage.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"
#include <algorithm>
#include <cctype>

namespace llaminar2
{
    namespace
    {
        /// @brief Emit a coarse per-device VRAM checkpoint when LLAMINAR_VRAM_TRACE is enabled.
        void logWorkspaceVramTrace(DeviceId device, const char *label, size_t bytes = 0)
        {
            if (!debugEnv().vram_trace || !device.is_gpu())
                return;

            IBackend *backend = getBackendFor(device);
            if (!backend)
                return;

            const int ordinal = device.gpu_ordinal();
            const size_t free_bytes = backend->deviceMemoryFree(ordinal);
            const size_t total_bytes = backend->deviceMemoryTotal(ordinal);
            const size_t used_bytes = total_bytes > free_bytes ? total_bytes - free_bytes : 0;
            LOG_INFO("[VRAM_TRACE] " << label
                                     << " device=" << device.toString()
                                     << " used_mib=" << (used_bytes / (1024 * 1024))
                                     << " free_mib=" << (free_bytes / (1024 * 1024))
                                     << " total_mib=" << (total_bytes / (1024 * 1024))
                                     << " bytes=" << bytes);
        }
    }

    // =========================================================================
    // Memory Query
    // =========================================================================

    size_t WorkspaceAllocator::queryAvailableMemory(DeviceId device)
    {
        if (!device.is_valid())
        {
            LOG_WARN("[WorkspaceAllocator] Cannot query memory for invalid device");
            return 0;
        }

        IBackend *backend = getBackendFor(device);
        if (!backend)
        {
            LOG_WARN("[WorkspaceAllocator] No backend available for " << device.toString());
            return 0;
        }

        int device_idx = device.is_cpu() ? 0 : device.gpu_ordinal();
        return backend->deviceMemoryFree(device_idx);
    }

    size_t WorkspaceAllocator::computeWorkspaceBudget(DeviceId device,
                                                      const WorkspaceBudgetConfig &config)
    {
        size_t available = queryAvailableMemory(device);
        if (available == 0)
        {
            LOG_DEBUG("[WorkspaceAllocator] No memory available for " << device.toString());
            return 0;
        }

        float fraction = device.is_cpu() ? config.cpu_fraction : config.gpu_fraction;
        size_t budget = static_cast<size_t>(static_cast<double>(available) * fraction);

        if (budget > config.headroom)
        {
            budget -= config.headroom;
        }
        else
        {
            budget = 0;
        }

        budget = std::max(budget, config.min_budget);
        budget = std::min(budget, config.max_budget);

        LOG_DEBUG("[WorkspaceAllocator] " << device.toString()
                                          << " available=" << (available / (1024 * 1024)) << "MB"
                                          << ", budget=" << (budget / (1024 * 1024)) << "MB"
                                          << " (fraction=" << fraction << ", headroom=" << (config.headroom / (1024 * 1024)) << "MB)");

        return budget;
    }

    // =========================================================================
    // Allocation
    // =========================================================================

    size_t WorkspaceAllocator::computeModelAwareBudgetFloor(const WorkspaceSizingHints &hints) const
    {
        const int max_seq_len = std::max(1, hints.max_seq_len);
        const int batch_size = std::max(1, hints.batch_size);
        const int vocab_size = std::max(1, hints.vocab_size);
        const int d_model = std::max(1, hints.d_model);

        // LM head always computes M=1 (last token only, even during prefill),
        // so its workspace is proportional to batch_size, not max_seq_len.
        const size_t lm_mn_buffer_size = static_cast<size_t>(batch_size) * static_cast<size_t>(vocab_size) * sizeof(float);
        const size_t lm_head_workspace = 3 * lm_mn_buffer_size;

        // Per-layer GEMM workspace uses full max_seq_len (prefill processes all tokens)
        const size_t mk_overhead = static_cast<size_t>(max_seq_len) * static_cast<size_t>(d_model) * sizeof(float) * 2;
        const size_t padded_n_buffer = 8ULL * static_cast<size_t>(vocab_size) * sizeof(float);
        const size_t embed_table_temp = static_cast<size_t>(vocab_size) * static_cast<size_t>(d_model) * sizeof(float);
        const size_t base_workspace = lm_head_workspace + mk_overhead + padded_n_buffer + embed_table_temp;
        const size_t safety_margin = base_workspace / 10;
        const size_t min_budget = 768ULL * 1024 * 1024;
        return std::max(min_budget, base_workspace + safety_margin);
    }

    bool WorkspaceAllocator::allocateForGraph(
        const ComputeGraph &graph,
        const WorkspaceSizingHints &hints,
        const std::vector<WorkspaceConsumerRequest> &extra_consumers,
        const WorkspaceBudgetConfig &config)
    {
        struct ConsumerBinding
        {
            IWorkspaceConsumer *consumer = nullptr;
            int m = 4096;
            int n = 0;
            int k = 0;
        };

        std::unordered_map<DeviceId, std::vector<ConsumerBinding>> consumers_by_device;

        const auto execution_order = graph.getExecutionOrder();
        for (const auto &node_name : execution_order)
        {
            const ComputeNode *node = graph.getNode(node_name);
            if (!node || !node->stage)
            {
                continue;
            }

            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(node->stage.get());
            if (!consumer)
            {
                continue;
            }

            DeviceId device = node->device;
            if (!device.is_gpu() && node->stage->device().is_gpu())
                device = node->stage->device();
            if (!device.is_gpu())
            {
                continue;
            }

            std::string lowered_name = node_name;
            std::transform(
                lowered_name.begin(), lowered_name.end(), lowered_name.begin(),
                [](unsigned char c)
                { return static_cast<char>(std::tolower(c)); });

            const bool is_embedding = (lowered_name == "embedding") || (lowered_name.find("embed") != std::string::npos);
            const bool is_attention = (lowered_name.find("attention") != std::string::npos);
            const bool is_lm_head = (lowered_name == "lm_head") || (lowered_name.find("lm_head") != std::string::npos);

            ConsumerBinding binding;
            binding.consumer = consumer;

            if (is_attention)
            {
                binding.m = std::max(1, hints.batch_size);
                binding.n = std::max(0, hints.n_heads);
                binding.k = std::max(0, hints.head_dim);
            }
            else if (is_lm_head)
            {
                binding.m = std::max(1, hints.batch_size);
                binding.n = 0;
                binding.k = 0;
            }
            else if (is_embedding)
            {
                binding.m = std::max(1, hints.max_seq_len);
                binding.n = std::max(1, hints.vocab_size);
                binding.k = std::max(0, hints.d_model);
            }
            else
            {
                binding.m = std::max(1, hints.max_seq_len);
                binding.n = 0;
                binding.k = 0;
            }

            consumers_by_device[device].push_back(binding);
        }

        for (const auto &request : extra_consumers)
        {
            if (!request.consumer || !request.device.is_gpu())
            {
                continue;
            }

            consumers_by_device[request.device].push_back(ConsumerBinding{
                request.consumer,
                std::max(1, request.m),
                request.n,
                request.k,
            });
        }

        if (consumers_by_device.empty())
        {
            LOG_DEBUG("[WorkspaceAllocator] No GPU workspace consumers found in graph");
            return true;
        }

        const size_t model_floor_budget = computeModelAwareBudgetFloor(hints);

        for (const auto &[device, consumers] : consumers_by_device)
        {
            if (!device.is_valid())
            {
                LOG_WARN("[WorkspaceAllocator] Skipping invalid device from graph consumer");
                continue;
            }

            auto existing = device_workspaces_.find(device);
            if (existing != device_workspaces_.end() && existing->second)
            {
                // Check if new consumers need buffers that are absent or larger
                // than the existing workspace allocation. Bucketed prefill can
                // warm a smaller graph before a larger bucket arrives; name-only
                // reuse would bind undersized scratch to the larger graph.
                bool needs_realloc = false;
                for (const auto &consumer_binding : consumers)
                {
                    auto reqs = consumer_binding.consumer->getWorkspaceRequirements(
                        consumer_binding.m,
                        consumer_binding.n,
                        consumer_binding.k);
                    for (const auto &buf : reqs.buffers)
                    {
                        if (!existing->second->hasBuffer(buf.name) ||
                            existing->second->getBufferSize(buf.name) < buf.size_bytes)
                        {
                            needs_realloc = true;
                            break;
                        }
                    }
                    if (needs_realloc)
                        break;
                }

                if (!needs_realloc)
                {
                    for (const auto &consumer_binding : consumers)
                    {
                        consumer_binding.consumer->bindWorkspace(existing->second.get());
                    }
                    continue;
                }

                // Reconstruct existing requirements from current workspace
                WorkspaceRequirements existing_reqs;
                for (const auto &name : existing->second->bufferNames())
                {
                    size_t sz = existing->second->getBufferSize(name);
                    existing_reqs.buffers.push_back({name, sz, 256, true});
                }

                // Release old workspace so we can reallocate with merged requirements
                size_t old_budget = existing->second->budget();
                existing->second->release();
                existing->second.reset();
                device_workspaces_.erase(device);
                device_workspace_budgets_.erase(device);

                // Merge existing + new requirements
                WorkspaceRequirements combined = existing_reqs;
                for (const auto &consumer_binding : consumers)
                {
                    auto reqs = consumer_binding.consumer->getWorkspaceRequirements(
                        consumer_binding.m,
                        consumer_binding.n,
                        consumer_binding.k);
                    combined.merge(reqs);
                }

                size_t budget = std::max(old_budget, model_floor_budget);
                const size_t needed = combined.total_bytes_with_alignment();
                if (needed > budget)
                {
                    const size_t available = queryAvailableMemory(device);
                    const size_t max_expandable = (available > config.headroom)
                                                      ? available - config.headroom
                                                      : 0;
                    if (needed <= max_expandable)
                    {
                        budget = needed;
                    }
                }

                LOG_DEBUG("[WorkspaceAllocator] Reallocating workspace on "
                          << device.toString() << " with "
                          << combined.buffers.size() << " buffers ("
                          << (needed / (1024 * 1024)) << "MB needed, budget="
                          << (budget / (1024 * 1024)) << "MB)");
                logWorkspaceVramTrace(device, "workspace.before_reallocate", needed);

                auto manager = std::make_unique<DeviceWorkspaceManager>(device, budget);
                if (!manager->allocate(combined))
                {
                    LOG_ERROR("[WorkspaceAllocator] Failed to reallocate workspace on "
                              << device.toString()
                              << " (needed=" << needed
                              << ", budget=" << budget << ")");
                    return false;
                }

                for (const auto &consumer_binding : consumers)
                {
                    consumer_binding.consumer->bindWorkspace(manager.get());
                }

                LOG_DEBUG("[WorkspaceAllocator] Reallocated " << (manager->used() / (1024 * 1024))
                                                              << "MB workspace on " << device.toString()
                                                              << " (" << manager->bufferCount() << " buffers)");
                logWorkspaceVramTrace(device, "workspace.after_reallocate", manager->used());

                device_workspace_budgets_[device] = budget;
                bumpDeviceGeneration(device);
                device_workspaces_[device] = std::move(manager);
                continue;
            }

            size_t budget = computeWorkspaceBudget(device, config);
            budget = std::max(budget, model_floor_budget);

            WorkspaceRequirements combined;
            for (const auto &consumer_binding : consumers)
            {
                auto reqs = consumer_binding.consumer->getWorkspaceRequirements(
                    consumer_binding.m,
                    consumer_binding.n,
                    consumer_binding.k);
                combined.merge(reqs);
            }

            if (combined.buffers.empty())
            {
                LOG_DEBUG("[WorkspaceAllocator] No workspace requirements for device "
                          << device.toString());
                continue;
            }

            // If the combined requirements exceed the initial budget, try to
            // expand up to the available device memory (minus headroom).
            // The initial budget uses a conservative max_budget cap that may
            // be too small for models with many per-instance GEMM workspaces.
            const size_t needed = combined.total_bytes_with_alignment();
            if (needed > budget)
            {
                const size_t available = queryAvailableMemory(device);
                const size_t max_expandable = (available > config.headroom)
                                                  ? available - config.headroom
                                                  : 0;
                if (needed <= max_expandable)
                {
                    LOG_DEBUG("[WorkspaceAllocator] Expanding budget on "
                              << device.toString() << " from "
                              << (budget / (1024 * 1024)) << "MB to "
                              << (needed / (1024 * 1024)) << "MB (available="
                              << (available / (1024 * 1024)) << "MB)");
                    budget = needed;
                }
            }

            auto manager = std::make_unique<DeviceWorkspaceManager>(device, budget);
            logWorkspaceVramTrace(device, "workspace.before_allocate", needed);
            if (!manager->allocate(combined))
            {
                LOG_ERROR("[WorkspaceAllocator] Failed to allocate workspace on "
                          << device.toString()
                          << " (needed=" << needed
                          << ", budget=" << budget << ")");
                return false;
            }

            for (const auto &consumer_binding : consumers)
            {
                consumer_binding.consumer->bindWorkspace(manager.get());
            }

            LOG_DEBUG("[WorkspaceAllocator] Allocated " << (manager->used() / (1024 * 1024))
                                                        << "MB workspace on " << device.toString()
                                                        << " (" << manager->bufferCount() << " buffers, model-aware budget)");
            logWorkspaceVramTrace(device, "workspace.after_allocate", manager->used());

            device_workspace_budgets_[device] = budget;
            bumpDeviceGeneration(device);
            device_workspaces_[device] = std::move(manager);
        }

        return true;
    }

    bool WorkspaceAllocator::allocateForStages(const std::vector<IComputeStage *> &stages,
                                               const WorkspaceBudgetConfig &config)
    {
        std::unordered_map<DeviceId, std::vector<IWorkspaceConsumer *>> device_consumers;

        for (auto *stage : stages)
        {
            if (!stage)
            {
                continue;
            }

            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(stage);
            if (consumer)
            {
                DeviceId device = stage->device();
                device_consumers[device].push_back(consumer);
            }
        }

        if (device_consumers.empty())
        {
            LOG_DEBUG("[WorkspaceAllocator] No workspace consumers found in " << stages.size() << " stages");
            return true;
        }

        LOG_DEBUG("[WorkspaceAllocator] Found " << device_consumers.size()
                                                << " devices with workspace consumers");

        for (const auto &[device, consumers] : device_consumers)
        {
            if (!device.is_valid())
            {
                LOG_WARN("[WorkspaceAllocator] Skipping invalid device from stage");
                continue;
            }

            size_t budget = computeWorkspaceBudget(device, config);
            if (budget == 0)
            {
                LOG_WARN("[WorkspaceAllocator] Zero budget for " << device.toString()
                                                                 << ", skipping workspace allocation");
                continue;
            }

            WorkspaceRequirements combined;
            for (auto *consumer : consumers)
            {
                auto reqs = consumer->getWorkspaceRequirements(/*max_m=*/4096);
                combined.merge(reqs);
            }

            if (combined.buffers.empty())
            {
                LOG_DEBUG("[WorkspaceAllocator] No workspace requirements for device "
                          << device.toString());
                continue;
            }

            LOG_DEBUG("[WorkspaceAllocator] Device " << device.toString()
                                                     << ": " << consumers.size() << " consumers, "
                                                     << combined.buffers.size() << " buffers, "
                                                     << combined.total_bytes_with_alignment() << " bytes needed");

            auto manager = std::make_unique<DeviceWorkspaceManager>(device, budget);
            logWorkspaceVramTrace(device, "workspace.before_allocate_legacy", combined.total_bytes_with_alignment());
            if (!manager->allocate(combined))
            {
                LOG_ERROR("[WorkspaceAllocator] Failed to allocate workspace on "
                          << device.toString()
                          << " (needed=" << combined.total_bytes_with_alignment()
                          << ", budget=" << budget << ")");
                return false;
            }

            for (auto *consumer : consumers)
            {
                consumer->bindWorkspace(manager.get());
            }

            LOG_DEBUG("[WorkspaceAllocator] Allocated " << (manager->used() / (1024 * 1024))
                                                        << "MB workspace on " << device.toString()
                                                        << " (" << manager->bufferCount() << " buffers)");
            logWorkspaceVramTrace(device, "workspace.after_allocate_legacy", manager->used());

            device_workspace_budgets_[device] = budget;
            device_workspaces_[device] = std::move(manager);
        }

        return true;
    }

    void WorkspaceAllocator::releaseAll()
    {
        if (!device_workspaces_.empty())
        {
            LOG_DEBUG("[WorkspaceAllocator] Releasing " << device_workspaces_.size()
                                                        << " workspace managers");
        }

        for (const auto &[device, _manager] : device_workspaces_)
        {
            bumpDeviceGeneration(device);
        }

        device_workspaces_.clear();
        device_workspace_budgets_.clear();
    }

    // =========================================================================
    // Access
    // =========================================================================

    DeviceWorkspaceManager *WorkspaceAllocator::getDeviceWorkspace(DeviceId device)
    {
        auto it = device_workspaces_.find(device);
        return (it != device_workspaces_.end()) ? it->second.get() : nullptr;
    }

    uint64_t WorkspaceAllocator::deviceGeneration(DeviceId device) const
    {
        auto it = device_workspace_generations_.find(device);
        return (it != device_workspace_generations_.end()) ? it->second : 0;
    }

    // =========================================================================
    // Metrics
    // =========================================================================

    size_t WorkspaceAllocator::totalAllocated() const
    {
        size_t total = 0;
        for (const auto &[device, mgr] : device_workspaces_)
        {
            total += mgr->used();
        }
        return total;
    }

    size_t WorkspaceAllocator::deviceAllocated(DeviceId device) const
    {
        auto it = device_workspaces_.find(device);
        return (it != device_workspaces_.end()) ? it->second->used() : 0;
    }

    void WorkspaceAllocator::bumpDeviceGeneration(DeviceId device)
    {
        device_workspace_generations_[device] = next_workspace_generation_++;
    }

} // namespace llaminar2
