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
#include "../../../utils/PerfStatsCollector.h"
#include "../../../utils/VramBillOfMaterials.h"
#include <algorithm>
#include <cctype>
#include <limits>
#include <unordered_map>
#include <unordered_set>

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
            LOG_TRACE("[VRAM_TRACE] " << label
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

        LOG_TRACE("[WorkspaceAllocator] " << device.toString()
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
        // Prepared embedding weights live in their own device allocation, so
        // embedding tables do not contribute to transient graph workspace.
        const size_t base_workspace = lm_head_workspace + mk_overhead + padded_n_buffer;
        const size_t safety_margin = base_workspace / 10;
        const size_t min_budget = 768ULL * 1024 * 1024;
        const size_t floor = std::max(min_budget, base_workspace + safety_margin);
        logVramBomLine(
            "workspace_model_floor",
            "max_seq_len=" + std::to_string(max_seq_len) +
                " batch_size=" + std::to_string(batch_size) +
                " vocab_size=" + std::to_string(vocab_size) +
                " d_model=" + std::to_string(d_model) +
                " lm_head_bytes=" + std::to_string(lm_head_workspace) +
                " lm_head_mib=" + vramBomMiB(lm_head_workspace) +
                " mk_overhead_bytes=" + std::to_string(mk_overhead) +
                " mk_overhead_mib=" + vramBomMiB(mk_overhead) +
                " padded_n_bytes=" + std::to_string(padded_n_buffer) +
                " padded_n_mib=" + vramBomMiB(padded_n_buffer) +
                " min_budget_bytes=" + std::to_string(min_budget) +
                " min_budget_mib=" + vramBomMiB(min_budget) +
                " floor_bytes=" + std::to_string(floor) +
                " floor_mib=" + vramBomMiB(floor));
        return floor;
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
            /**
             * @brief True when this consumer's M follows graph prefill rows.
             *
             * Terminal LM-head projection consumes one selected row during
             * ordinary prefill, and attention exposes its own batch/head
             * workspace geometry. Feeding either consumer an artificial prompt
             * M fabricates buffers that no real graph can execute.
             */
            bool scales_with_serial_family_rows = true;
            /**
             * @brief Whether auxiliary graph roles may replace M with their rows.
             *
             * Graph stages use participant rows. Explicit consumers may instead
             * declare a fixed control-plane shape whose M means request count or
             * another non-token dimension.
             */
            WorkspaceConsumerShapePolicy shape_policy =
                WorkspaceConsumerShapePolicy::GraphParticipantRows;
        };

        /**
         * @brief Logical participant whose requirements are being assembled.
         *
         * The role is explicit because row count is not a lifetime contract:
         * a short prompt and grouped MTP verification may have the same M while
         * selecting different production kernels and disjoint scratch arenas.
         */
        enum class SerialWorkspaceParticipantRole : uint8_t
        {
            PrefillGraph,
            DecodeGraph,
            GroupedVerifierGraph,
        };

        auto requirementsForRows =
            [&hints](
                const ConsumerBinding &binding,
                int requested_rows,
                SerialWorkspaceParticipantRole role) -> WorkspaceRequirements
        {
            if (!binding.consumer)
                return {};

            const int effective_rows =
                binding.shape_policy ==
                        WorkspaceConsumerShapePolicy::FixedDeclaredShape
                    ? binding.m
                    : requested_rows;
            WorkspaceRequirements requirements =
                binding.consumer->getWorkspaceRequirements(
                    std::max(1, effective_rows),
                    binding.n,
                    binding.k);

            if (hints.graph_family_policy !=
                WorkspaceGraphFamilyPolicy::ExclusiveLifetime)
            {
                std::erase_if(
                    requirements.buffers,
                    [role](const WorkspaceDescriptor &descriptor)
                    {
                        switch (role)
                        {
                        case SerialWorkspaceParticipantRole::PrefillGraph:
                            return descriptor.regime ==
                                   WorkspaceExecutionRegime::
                                       CompactDecodeOnly;
                        case SerialWorkspaceParticipantRole::DecodeGraph:
                        case SerialWorkspaceParticipantRole::
                            GroupedVerifierGraph:
                            return descriptor.regime ==
                                   WorkspaceExecutionRegime::PrefillOnly;
                        }
                        return false;
                    });
            }

            return requirements;
        };

        auto activeRowsForBinding =
            [&hints](const ConsumerBinding &binding) -> int
        {
            if (hints.graph_family_policy ==
                    WorkspaceGraphFamilyPolicy::
                        SerialDeviceFamilyLargestParticipant &&
                binding.scales_with_serial_family_rows &&
                hints.serial_family_max_rows > binding.m)
            {
                return hints.serial_family_max_rows;
            }
            return binding.m;
        };

        auto requirementsForGraphBinding =
            [&](const ConsumerBinding &binding) -> WorkspaceRequirements
        {
            if (hints.graph_family_policy !=
                WorkspaceGraphFamilyPolicy::ExclusiveLifetime)
            {
                return requirementsForRows(
                    binding,
                    activeRowsForBinding(binding),
                    binding.scales_with_serial_family_rows
                        ? SerialWorkspaceParticipantRole::PrefillGraph
                        : SerialWorkspaceParticipantRole::DecodeGraph);
            }

            /*
             * Exclusive diagnostic users retain the historical union contract:
             * their storage cannot alias another participant, so the same block
             * must cover both the declared shape and one-row decode.
             */
            WorkspaceRequirements combined;
            if (binding.m != 1)
            {
                combined.merge(requirementsForRows(
                    binding,
                    1,
                    SerialWorkspaceParticipantRole::DecodeGraph));
            }
            combined.merge(requirementsForRows(
                binding,
                binding.m,
                SerialWorkspaceParticipantRole::PrefillGraph));
            return combined;
        };

        auto clampDimToInt = [](size_t value) -> int
        {
            if (value > static_cast<size_t>(std::numeric_limits<int>::max()))
            {
                return std::numeric_limits<int>::max();
            }
            return static_cast<int>(value);
        };

        auto applyDeclaredStageShape = [&](const IComputeStage &stage,
                                           ConsumerBinding &binding) -> bool
        {
            const StageBufferRequirements buffer_reqs = stage.getBufferRequirements();
            int declared_m = 0;
            int declared_k = 0;

            for (const auto &buffer : buffer_reqs.buffers)
            {
                if (buffer.shape.size() < 2 || buffer.shape[0] == 0 || buffer.shape.back() == 0)
                {
                    continue;
                }

                const bool activation_like =
                    buffer.role == BufferRole::INPUT ||
                    buffer.role == BufferRole::INOUT ||
                    buffer.role == BufferRole::OUTPUT ||
                    buffer.role == BufferRole::SCRATCH;
                if (!activation_like)
                {
                    continue;
                }

                declared_m = clampDimToInt(buffer.shape[0]);
                if (buffer.role == BufferRole::INPUT || buffer.role == BufferRole::INOUT)
                {
                    declared_k = clampDimToInt(buffer.shape.back());
                }
                break;
            }

            if (declared_m <= 0)
            {
                return false;
            }

            binding.m = std::max(1, declared_m);
            if (declared_k > 0)
            {
                binding.k = declared_k;
            }
            return true;
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
            if ((!device.is_valid() || !device.is_gpu()) &&
                node->stage->device().is_valid())
            {
                device = node->stage->device();
            }
            if (!device.is_gpu() && node->stage->device().is_gpu())
            {
                device = node->stage->device();
            }
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
                binding.scales_with_serial_family_rows = false;
            }
            else if (is_lm_head)
            {
                binding.m = std::max(1, hints.batch_size);
                binding.n = 0;
                binding.k = 0;
                binding.scales_with_serial_family_rows = false;
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
                (void)applyDeclaredStageShape(*node->stage, binding);
            }

            consumers_by_device[device].push_back(binding);
        }

        for (const auto &request : extra_consumers)
        {
            if (!request.consumer ||
                !request.device.is_gpu())
            {
                continue;
            }

            consumers_by_device[request.device].push_back(ConsumerBinding{
                request.consumer,
                std::max(1, request.m),
                request.n,
                request.k,
                true,
                request.shape_policy,
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
                    auto reqs = requirementsForGraphBinding(consumer_binding);
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

                WorkspaceRequirements combined;
                for (const auto &consumer_binding : consumers)
                {
                    combined.merge(requirementsForGraphBinding(consumer_binding));
                }
                const size_t needed = combined.total_bytes_with_alignment();
                if (hints.graph_family_policy !=
                    WorkspaceGraphFamilyPolicy::ExclusiveLifetime)
                {
                    logVramBomLine(
                        "workspace_plan",
                        "phase=serial_alias_bind device=" +
                            device.toString() +
                            " consumers=" +
                            std::to_string(consumers.size()) +
                            " buffers=" +
                            std::to_string(combined.buffers.size()) +
                            " logical_bytes=" + std::to_string(needed) +
                            " logical_mib=" + vramBomMiB(needed) +
                            " primary_block_bytes=" +
                            std::to_string(existing->second->primaryBlockSize()) +
                            " physical_growth_bytes=0");
                    if (!existing->second->bindSerialParticipant(combined))
                    {
                        LOG_ERROR("[WorkspaceAllocator] Serial graph-family participant could not bind into the primary workspace on "
                                  << device.toString());
                        return false;
                    }
                    for (const auto &consumer_binding : consumers)
                    {
                        consumer_binding.consumer->bindWorkspace(
                            existing->second.get());
                    }
                    continue;
                }

                LOG_TRACE("[WorkspaceAllocator] Extending workspace append-only on "
                          << device.toString() << " for "
                          << combined.buffers.size() << " current requirements ("
                          << (needed / (1024 * 1024)) << "MB logical, remaining="
                          << (existing->second->remaining() / (1024 * 1024))
                          << "MB)");
                logVramBomLine(
                    "workspace_plan",
                    "phase=append_only_extend device=" + device.toString() +
                        " consumers=" + std::to_string(consumers.size()) +
                        " buffers=" + std::to_string(combined.buffers.size()) +
                        " needed_bytes=" + std::to_string(needed) +
                        " needed_mib=" + vramBomMiB(needed) +
                        " budget_bytes=" + std::to_string(existing->second->budget()) +
                        " budget_mib=" + vramBomMiB(existing->second->budget()) +
                        " model_floor_bytes=" + std::to_string(model_floor_budget) +
                        " model_floor_mib=" + vramBomMiB(model_floor_budget));
                logWorkspaceVramTrace(device, "workspace.before_append_only_extend", needed);
                if (!existing->second->extend(combined))
                {
                    LOG_ERROR("[WorkspaceAllocator] Failed to extend workspace append-only on "
                              << device.toString() << " (logical_needed="
                              << needed << ", budget="
                              << existing->second->budget() << ", used="
                              << existing->second->used() << ")");
                    return false;
                }

                for (const auto &consumer_binding : consumers)
                {
                    consumer_binding.consumer->bindWorkspace(existing->second.get());
                }

                LOG_TRACE("[WorkspaceAllocator] Append-only workspace now owns "
                          << (existing->second->used() / (1024 * 1024))
                          << "MB on " << device.toString()
                          << " (" << existing->second->bufferCount()
                          << " current names)");
                logWorkspaceVramTrace(
                    device,
                    "workspace.after_append_only_extend",
                    existing->second->used());
                continue;
            }

            size_t budget = computeWorkspaceBudget(device, config);
            if (device.is_gpu())
            {
                budget = std::max(budget, model_floor_budget);
            }

            WorkspaceRequirements combined;
            for (const auto &consumer_binding : consumers)
            {
                combined.merge(requirementsForGraphBinding(consumer_binding));
            }

            WorkspaceRequirements decode_participant;
            WorkspaceRequirements compact_participant;
            if (hints.graph_family_policy !=
                WorkspaceGraphFamilyPolicy::ExclusiveLifetime)
            {
                /*
                 * Build auxiliary layouts independently. Their names are
                 * published before the first capture, but their bytes overlay
                 * the largest participant because device events serialize the
                 * graph roles.
                 */
                for (const auto &consumer_binding : consumers)
                {
                    decode_participant.merge(
                        requirementsForRows(
                            consumer_binding,
                            1,
                            SerialWorkspaceParticipantRole::DecodeGraph));
                }
                if (hints.serial_family_max_compact_rows > 1)
                {
                    for (const auto &consumer_binding : consumers)
                    {
                        compact_participant.merge(
                            requirementsForRows(
                                consumer_binding,
                                hints.serial_family_max_compact_rows,
                                SerialWorkspaceParticipantRole::
                                    GroupedVerifierGraph));
                    }
                }

                /*
                 * A workspace name is part of the captured pointer ABI. If
                 * several serial participants use the same name, the first
                 * published address must have enough capacity for every one of
                 * them. Promote only the first participant that publishes the
                 * name to its family-wide capacity. Later participants retain
                 * their actual live extent, allowing role-exclusive buffers
                 * to reuse the unused tail while the manager still advertises
                 * the larger captured capacity. Names absent from a
                 * participant remain fully role-exclusive.
                 */
                struct FamilyBufferCapacity
                {
                    size_t size_bytes = 0;
                    size_t alignment = 1;
                    bool required = false;
                };

                std::unordered_map<std::string, FamilyBufferCapacity>
                    family_capacities;
                const std::vector<WorkspaceRequirements *>
                    serial_participants{
                        &combined,
                        &decode_participant,
                        &compact_participant};
                for (const WorkspaceRequirements *participant :
                     serial_participants)
                {
                    for (const WorkspaceDescriptor &descriptor :
                         participant->buffers)
                    {
                        FamilyBufferCapacity &capacity =
                            family_capacities[descriptor.name];
                        capacity.size_bytes = std::max(
                            capacity.size_bytes,
                            descriptor.size_bytes);
                        capacity.alignment = std::max(
                            capacity.alignment,
                            descriptor.alignment);
                        capacity.required =
                            capacity.required || descriptor.required;
                    }
                }

                size_t promoted_descriptors = 0;
                size_t promoted_bytes = 0;
                std::unordered_set<std::string> published_names;
                for (WorkspaceRequirements *participant :
                     serial_participants)
                {
                    for (WorkspaceDescriptor &descriptor :
                         participant->buffers)
                    {
                        const bool publishes_name =
                            published_names.insert(
                                descriptor.name).second;
                        if (!publishes_name)
                            continue;

                        const FamilyBufferCapacity &capacity =
                            family_capacities.at(descriptor.name);
                        if (descriptor.size_bytes <
                            capacity.size_bytes)
                        {
                            const size_t delta =
                                capacity.size_bytes -
                                descriptor.size_bytes;
                            promoted_bytes += delta;
                            ++promoted_descriptors;
                            logVramBomLine(
                                "workspace_serial_shared_capacity",
                                "device=" + device.toString() +
                                    " name=" + descriptor.name +
                                    " previous_bytes=" +
                                    std::to_string(
                                        descriptor.size_bytes) +
                                    " family_bytes=" +
                                    std::to_string(
                                        capacity.size_bytes) +
                                    " promoted_bytes=" +
                                    std::to_string(delta));
                        }
                        descriptor.size_bytes =
                            capacity.size_bytes;
                        descriptor.alignment =
                            capacity.alignment;
                        descriptor.required =
                            capacity.required;
                    }
                }
                if (promoted_descriptors > 0)
                {
                    PerfStatsCollector::addCounter(
                        "memory",
                        "workspace_serial_shared_capacity_promotions",
                        static_cast<double>(
                            promoted_descriptors),
                        "materialize",
                        device.to_string(),
                        {{"promoted_bytes",
                          std::to_string(promoted_bytes)}});
                }
            }

            if (combined.buffers.empty() &&
                decode_participant.buffers.empty() &&
                compact_participant.buffers.empty())
            {
                LOG_DEBUG("[WorkspaceAllocator] No workspace requirements for device "
                          << device.toString());
                continue;
            }

            // If the combined requirements exceed the initial budget, try to
            // expand up to the available device memory (minus headroom).
            // The initial budget uses a conservative max_budget cap that may
            // be too small for models with many per-instance GEMM workspaces.
            const size_t active_needed =
                combined.total_bytes_with_alignment();
            const size_t decode_needed =
                decode_participant.total_bytes_with_alignment();
            const size_t compact_needed =
                compact_participant.total_bytes_with_alignment();
            const size_t needed = std::max(
                active_needed,
                std::max(decode_needed, compact_needed));
            if (hints.graph_family_policy ==
                WorkspaceGraphFamilyPolicy::
                    SerialDeviceFamilyLargestParticipant)
            {
                PerfStatsCollector::addCounter(
                    "memory",
                    "workspace_serial_family_largest_participant_bytes",
                    static_cast<double>(needed),
                    "materialize",
                    device.to_string(),
                    {{"current_rows", std::to_string(hints.max_seq_len)},
                     {"largest_rows",
                      std::to_string(hints.serial_family_max_rows)},
                     {"buffer_count",
                      std::to_string(combined.buffers.size())},
                     {"active_logical_bytes",
                      std::to_string(active_needed)},
                     {"decode_logical_bytes",
                      std::to_string(decode_needed)},
                     {"compact_logical_bytes",
                      std::to_string(compact_needed)},
                     {"ordering", "serial_graph_family"}});
                logVramBomLine(
                    "workspace_serial_family_plan",
                    "device=" + device.toString() +
                        " current_rows=" +
                        std::to_string(hints.max_seq_len) +
                        " largest_rows=" +
                        std::to_string(hints.serial_family_max_rows) +
                        " buffers=" +
                        std::to_string(combined.buffers.size()) +
                        " active_logical_bytes=" +
                        std::to_string(active_needed) +
                        " decode_logical_bytes=" +
                        std::to_string(decode_needed) +
                        " compact_logical_bytes=" +
                        std::to_string(compact_needed) +
                        " needed_bytes=" + std::to_string(needed) +
                        " needed_mib=" + vramBomMiB(needed) +
                        " ownership=largest_participant" +
                        " ordering=serial_graph_family");
            }
            if (needed > budget)
            {
                const size_t available = queryAvailableMemory(device);
                const size_t max_expandable = (available > config.headroom)
                                                  ? available - config.headroom
                                                  : 0;
                if (needed <= max_expandable)
                {
                    LOG_TRACE("[WorkspaceAllocator] Expanding budget on "
                              << device.toString() << " from "
                              << (budget / (1024 * 1024)) << "MB to "
                              << (needed / (1024 * 1024)) << "MB (available="
                              << (available / (1024 * 1024)) << "MB)");
                    budget = needed;
                }
            }

            auto manager = std::make_unique<DeviceWorkspaceManager>(device, budget);
            logVramBomLine(
                "workspace_plan",
                "phase=allocate device=" + device.toString() +
                    " consumers=" + std::to_string(consumers.size()) +
                    " buffers=" + std::to_string(combined.buffers.size()) +
                    " needed_bytes=" + std::to_string(needed) +
                    " needed_mib=" + vramBomMiB(needed) +
                    " budget_bytes=" + std::to_string(budget) +
                    " budget_mib=" + vramBomMiB(budget) +
                    " model_floor_bytes=" + std::to_string(model_floor_budget) +
                    " model_floor_mib=" + vramBomMiB(model_floor_budget));
            logWorkspaceVramTrace(device, "workspace.before_allocate", needed);
            if (!manager->allocate(combined, needed))
            {
                LOG_ERROR("[WorkspaceAllocator] Failed to allocate workspace on "
                          << device.toString()
                          << " (needed=" << needed
                          << ", budget=" << budget << ")");
                return false;
            }

            if (!compact_participant.buffers.empty() &&
                !manager->bindSerialParticipant(compact_participant))
            {
                LOG_ERROR("[WorkspaceAllocator] Failed to prebind compact serial participant on "
                          << device.toString());
                return false;
            }
            if (!decode_participant.buffers.empty() &&
                !manager->bindSerialParticipant(decode_participant))
            {
                LOG_ERROR("[WorkspaceAllocator] Failed to prebind decode serial participant on "
                          << device.toString());
                return false;
            }

            for (const auto &consumer_binding : consumers)
            {
                consumer_binding.consumer->bindWorkspace(manager.get());
            }

            LOG_TRACE("[WorkspaceAllocator] Allocated " << (manager->used() / (1024 * 1024))
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

        LOG_TRACE("[WorkspaceAllocator] Found " << device_consumers.size()
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
                combined.merge(consumer->getWorkspaceRequirements(/*max_m=*/4096));
                combined.merge(consumer->getWorkspaceRequirements(/*decode_m=*/1));
            }

            if (combined.buffers.empty())
            {
                LOG_DEBUG("[WorkspaceAllocator] No workspace requirements for device "
                          << device.toString());
                continue;
            }

            LOG_TRACE("[WorkspaceAllocator] Device " << device.toString()
                                                     << ": " << consumers.size() << " consumers, "
                                                     << combined.buffers.size() << " buffers, "
                                                     << combined.total_bytes_with_alignment() << " bytes needed");

            auto manager = std::make_unique<DeviceWorkspaceManager>(device, budget);
            logVramBomLine(
                "workspace_plan",
                "phase=legacy_allocate device=" + device.toString() +
                    " consumers=" + std::to_string(consumers.size()) +
                    " buffers=" + std::to_string(combined.buffers.size()) +
                    " needed_bytes=" + std::to_string(combined.total_bytes_with_alignment()) +
                    " needed_mib=" + vramBomMiB(combined.total_bytes_with_alignment()) +
                    " budget_bytes=" + std::to_string(budget) +
                    " budget_mib=" + vramBomMiB(budget));
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

            LOG_TRACE("[WorkspaceAllocator] Allocated " << (manager->used() / (1024 * 1024))
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
            LOG_TRACE("[WorkspaceAllocator] Releasing " << device_workspaces_.size()
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
