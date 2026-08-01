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
#include <limits>
#include <map>
#include <unordered_map>

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
        for (const std::uint64_t word : key.identity_words)
        {
            hash = mixPublicationHash(hash, word);
        }
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

    bool DeviceWorkspaceManager::allocate(
        const WorkspaceRequirements &requirements,
        size_t minimum_primary_block_bytes)
    {
        if (allocated_)
        {
            LOG_WARN("[DeviceWorkspaceManager] Already allocated, call release() first");
            return false;
        }

        if (minimum_primary_block_bytes > budget_bytes_)
        {
            LOG_ERROR("[DeviceWorkspaceManager] Required primary family capacity "
                      << minimum_primary_block_bytes << " bytes exceeds budget "
                      << budget_bytes_ << " bytes on " << device_.to_string());
            return false;
        }

        // Handle a genuinely empty allocation. A serial family may have no
        // names in its first participant while still reserving physical space
        // for a later participant, so a non-zero minimum continues below.
        if (requirements.buffers.empty() &&
            minimum_primary_block_bytes == 0)
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
        total_size = std::max(total_size, minimum_primary_block_bytes);

        // Log all buffer requirements
        LOG_DEBUG("[DeviceWorkspaceManager] Workspace requirements (" << requirements.buffers.size() << " buffers):");
        for (const auto &buf : requirements.buffers)
        {
            LOG_TRACE("[DeviceWorkspaceManager]   - " << buf.name << ": " << (buf.size_bytes / (1024 * 1024)) << " MB"
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

            // Recalculate with only buffers that fit.
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
            total_size = std::max(
                total_size,
                minimum_primary_block_bytes);

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

    SerialWorkspaceFamilyPlan DeviceWorkspaceManager::planSerialFamily(
        const std::vector<WorkspaceRequirements> &participants)
    {
        SerialWorkspaceFamilyPlan plan;

        /*
         * A sorted map gives every workspace ABI name a deterministic index.
         * The canonical descriptor records the capacity that must be published
         * before capture, independent of which participant happened to be
         * discovered first.
         */
        std::map<std::string, WorkspaceDescriptor> canonical_by_name;
        for (const WorkspaceRequirements &participant : participants)
        {
            for (const WorkspaceDescriptor &descriptor : participant.buffers)
            {
                if (descriptor.name.empty())
                {
                    plan.error =
                        "Serial workspace family contains an empty buffer name";
                    return plan;
                }
                if (descriptor.alignment == 0 ||
                    (descriptor.alignment & (descriptor.alignment - 1)) != 0)
                {
                    plan.error =
                        "Serial workspace buffer '" + descriptor.name +
                        "' has non-power-of-two alignment " +
                        std::to_string(descriptor.alignment);
                    return plan;
                }

                auto [it, inserted] =
                    canonical_by_name.emplace(descriptor.name, descriptor);
                if (inserted)
                    continue;

                WorkspaceDescriptor &canonical = it->second;
                canonical.size_bytes =
                    std::max(canonical.size_bytes, descriptor.size_bytes);
                canonical.alignment =
                    std::max(canonical.alignment, descriptor.alignment);
                canonical.required =
                    canonical.required || descriptor.required;
                if (canonical.regime != descriptor.regime)
                    canonical.regime = WorkspaceExecutionRegime::Any;
            }
        }

        if (canonical_by_name.empty())
            return plan;

        std::vector<WorkspaceDescriptor> descriptors;
        descriptors.reserve(canonical_by_name.size());
        std::unordered_map<std::string, size_t> index_by_name;
        index_by_name.reserve(canonical_by_name.size());
        for (const auto &[name, descriptor] : canonical_by_name)
        {
            index_by_name.emplace(name, descriptors.size());
            descriptors.push_back(descriptor);
        }

        /*
         * Each participant is a clique in the interval-conflict graph, but a
         * shared name may expose a larger family capacity than one participant
         * actually touches. Record directional live extents for each co-resident
         * pair: [A][B] is the largest prefix of A used by any participant that
         * also contains B. This lets a compact-only arena reuse the unused tail
         * of a large prefill buffer without lying about A's published capacity.
         *
         * A dense size matrix is intentional: production families have
         * hundreds, not millions, of names, and the representation remains
         * negligible beside model materialization.
         */
        const size_t count = descriptors.size();
        std::vector<std::vector<size_t>> coexistent_live_extents(
            count,
            std::vector<size_t>(count, size_t{0}));
        for (const WorkspaceRequirements &participant : participants)
        {
            std::unordered_map<size_t, size_t> live_extent_by_index;
            live_extent_by_index.reserve(participant.buffers.size());
            for (const WorkspaceDescriptor &descriptor : participant.buffers)
            {
                const size_t index = index_by_name.at(descriptor.name);
                live_extent_by_index[index] =
                    std::max(
                        live_extent_by_index[index],
                        descriptor.size_bytes);
            }

            std::vector<std::pair<size_t, size_t>> participant_buffers(
                live_extent_by_index.begin(),
                live_extent_by_index.end());
            for (size_t left = 0; left < participant_buffers.size(); ++left)
            {
                for (size_t right = left + 1;
                     right < participant_buffers.size();
                     ++right)
                {
                    const auto [lhs, lhs_extent] =
                        participant_buffers[left];
                    const auto [rhs, rhs_extent] =
                        participant_buffers[right];
                    coexistent_live_extents[lhs][rhs] =
                        std::max(
                            coexistent_live_extents[lhs][rhs],
                            lhs_extent);
                    coexistent_live_extents[rhs][lhs] =
                        std::max(
                            coexistent_live_extents[rhs][lhs],
                            rhs_extent);
                }
            }
        }

        std::vector<size_t> placement_order(count);
        for (size_t index = 0; index < count; ++index)
            placement_order[index] = index;
        std::sort(
            placement_order.begin(),
            placement_order.end(),
            [&](size_t lhs, size_t rhs)
            {
                if (descriptors[lhs].size_bytes !=
                    descriptors[rhs].size_bytes)
                {
                    return descriptors[lhs].size_bytes >
                           descriptors[rhs].size_bytes;
                }
                if (descriptors[lhs].alignment !=
                    descriptors[rhs].alignment)
                {
                    return descriptors[lhs].alignment >
                           descriptors[rhs].alignment;
                }
                return descriptors[lhs].name < descriptors[rhs].name;
            });

        struct ConflictInterval
        {
            size_t begin = 0;
            size_t end = 0;
            size_t current_live_extent = 0;
        };
        std::vector<size_t> offsets(count, 0);
        std::vector<uint8_t> placed(count, uint8_t{0});

        for (const size_t index : placement_order)
        {
            const WorkspaceDescriptor &descriptor = descriptors[index];
            std::vector<ConflictInterval> forbidden;
            forbidden.reserve(count);
            for (size_t other = 0; other < count; ++other)
            {
                const size_t current_live_extent =
                    coexistent_live_extents[index][other];
                const size_t other_live_extent =
                    coexistent_live_extents[other][index];
                if (!placed[other] ||
                    current_live_extent == 0 ||
                    other_live_extent == 0)
                {
                    continue;
                }
                if (other_live_extent >
                    std::numeric_limits<size_t>::max() - offsets[other])
                {
                    plan.error =
                        "Serial workspace interval overflow for '" +
                        descriptors[other].name + "'";
                    return plan;
                }
                forbidden.push_back(ConflictInterval{
                    .begin = offsets[other],
                    .end = offsets[other] + other_live_extent,
                    .current_live_extent = current_live_extent,
                });
            }
            std::sort(
                forbidden.begin(),
                forbidden.end(),
                [](const ConflictInterval &lhs,
                   const ConflictInterval &rhs)
                {
                    if (lhs.begin != rhs.begin)
                        return lhs.begin < rhs.begin;
                    return lhs.end < rhs.end;
                });

            size_t candidate = 0;
            for (const ConflictInterval &interval : forbidden)
            {
                candidate = alignUp(candidate, descriptor.alignment);
                if (candidate <= interval.begin &&
                    interval.current_live_extent <=
                        interval.begin - candidate)
                {
                    continue;
                }
                if (interval.end > candidate)
                    candidate = interval.end;
            }
            candidate = alignUp(candidate, descriptor.alignment);
            if (descriptor.size_bytes >
                std::numeric_limits<size_t>::max() - candidate)
            {
                plan.error =
                    "Serial workspace placement overflow for '" +
                    descriptor.name + "'";
                return plan;
            }

            offsets[index] = candidate;
            placed[index] = uint8_t{1};
            plan.total_bytes = std::max(
                plan.total_bytes,
                candidate + descriptor.size_bytes);
        }

        plan.placements.reserve(count);
        for (size_t index = 0; index < count; ++index)
        {
            plan.placements.push_back(SerialWorkspaceBufferPlacement{
                .descriptor = descriptors[index],
                .offset = offsets[index],
            });
        }

        /*
         * Keep a final independent validation close to the planner. A future
         * heuristic change must fail here instead of publishing overlapping
         * addresses to a captured graph.
         */
        for (size_t lhs = 0; lhs < count; ++lhs)
        {
            for (size_t rhs = lhs + 1; rhs < count; ++rhs)
            {
                const size_t lhs_live_extent =
                    coexistent_live_extents[lhs][rhs];
                const size_t rhs_live_extent =
                    coexistent_live_extents[rhs][lhs];
                if (lhs_live_extent == 0 || rhs_live_extent == 0)
                    continue;
                const size_t lhs_end =
                    offsets[lhs] + lhs_live_extent;
                const size_t rhs_end =
                    offsets[rhs] + rhs_live_extent;
                if (offsets[lhs] < rhs_end && offsets[rhs] < lhs_end)
                {
                    plan.error =
                        "Serial workspace planner overlapped co-resident names '" +
                        descriptors[lhs].name + "' and '" +
                        descriptors[rhs].name + "'";
                    plan.placements.clear();
                    plan.total_bytes = 0;
                    return plan;
                }
            }
        }

        return plan;
    }

    bool DeviceWorkspaceManager::allocateSerialFamily(
        const SerialWorkspaceFamilyPlan &plan)
    {
        if (allocated_)
        {
            LOG_ERROR("[DeviceWorkspaceManager] Cannot allocate a serial family after workspace publication on "
                      << device_.to_string());
            return false;
        }
        if (!plan.valid())
        {
            LOG_ERROR("[DeviceWorkspaceManager] Refusing invalid serial workspace family plan on "
                      << device_.to_string() << ": " << plan.error);
            return false;
        }
        if (plan.total_bytes > budget_bytes_)
        {
            LOG_ERROR("[DeviceWorkspaceManager] Serial workspace family requires "
                      << plan.total_bytes << " bytes but budget is "
                      << budget_bytes_ << " bytes on " << device_.to_string());
            /*
             * A family-sized budget miss is a declaration defect, not a
             * recoverable allocation failure. Attribute the largest canonical
             * names while the complete plan is still available so operators
             * can identify an accidental M/K multiplier or lifetime-policy
             * error without enabling millions of per-stage DEBUG messages.
             */
            std::vector<const SerialWorkspaceBufferPlacement *>
                largest_placements;
            largest_placements.reserve(plan.placements.size());
            for (const auto &placement : plan.placements)
                largest_placements.push_back(&placement);
            std::sort(
                largest_placements.begin(),
                largest_placements.end(),
                [](const SerialWorkspaceBufferPlacement *lhs,
                   const SerialWorkspaceBufferPlacement *rhs)
                {
                    if (lhs->descriptor.size_bytes !=
                        rhs->descriptor.size_bytes)
                    {
                        return lhs->descriptor.size_bytes >
                               rhs->descriptor.size_bytes;
                    }
                    return lhs->descriptor.name <
                           rhs->descriptor.name;
                });
            constexpr size_t kMaximumDiagnosticPlacements = 12;
            const size_t diagnostic_count = std::min(
                kMaximumDiagnosticPlacements,
                largest_placements.size());
            for (size_t index = 0;
                 index < diagnostic_count;
                 ++index)
            {
                const auto &placement =
                    *largest_placements[index];
                LOG_ERROR("[DeviceWorkspaceManager] Oversized serial-family contributor rank="
                          << index
                          << " name="
                          << placement.descriptor.name
                          << " bytes="
                          << placement.descriptor.size_bytes
                          << " offset="
                          << placement.offset
                          << " regime="
                          << static_cast<int>(
                                 placement.descriptor.regime)
                          << " required="
                          << placement.descriptor.required);
            }
            return false;
        }
        if (plan.placements.empty())
        {
            if (plan.total_bytes != 0)
            {
                LOG_ERROR("[DeviceWorkspaceManager] Empty serial workspace family has non-zero physical size on "
                          << device_.to_string());
                return false;
            }
            allocated_ = true;
            return true;
        }

        return allocatePlacedBuffers(plan.placements, plan.total_bytes);
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

        /*
         * Keep descriptor-level evidence for every late graph-family request.
         * A late extension is a lifetime-planning signal: if its producer graph
         * is serialized with an already-resident graph, these names may be
         * candidates for one preplanned largest-participant slot. Recording
         * both the previous and requested capacities makes that analysis
         * possible without a TRACE-sized execution log.
         */
        for (const WorkspaceDescriptor *buffer : fitting)
        {
            const auto existing = buffers_.find(buffer->name);
            const size_t previous_bytes =
                existing == buffers_.end() ? 0 : existing->second.size;
            logVramBomLine(
                "workspace_extension_buffer",
                "device=" + device_.to_string() +
                    " manager_id=" + std::to_string(id_) +
                    " name=" + buffer->name +
                    " previous_bytes=" + std::to_string(previous_bytes) +
                    " requested_bytes=" +
                    std::to_string(buffer->size_bytes) +
                    " required=" +
                    (buffer->required ? "true" : "false"));
        }
        return allocateExtensionBuffers(fitting, extension_size);
    }

    bool DeviceWorkspaceManager::bindSerialParticipant(
        const WorkspaceRequirements &requirements)
    {
        if (!allocated_ || !block_ || block_size_ == 0)
        {
            LOG_ERROR("[DeviceWorkspaceManager] Cannot bind a serial participant before the primary workspace allocation on "
                      << device_.to_string());
            return false;
        }

        struct Interval
        {
            size_t begin = 0;
            size_t end = 0;
            std::string name;
        };
        struct PendingAlias
        {
            const WorkspaceDescriptor *descriptor = nullptr;
            size_t offset = 0;
        };

        std::vector<Interval> occupied;
        std::vector<const WorkspaceDescriptor *> missing;
        occupied.reserve(requirements.buffers.size());
        missing.reserve(requirements.buffers.size());

        for (const WorkspaceDescriptor &descriptor : requirements.buffers)
        {
            const auto existing = buffers_.find(descriptor.name);
            if (existing == buffers_.end())
            {
                missing.push_back(&descriptor);
                continue;
            }

            if (existing->second.size < descriptor.size_bytes)
            {
                LOG_ERROR("[DeviceWorkspaceManager] Serial graph-family name '"
                          << descriptor.name << "' was captured with "
                          << existing->second.size << " bytes but a later participant requires "
                          << descriptor.size_bytes << " bytes on "
                          << device_.to_string()
                          << "; compact/largest-participant preflight is incomplete");
                return false;
            }

            /*
             * An existing name in an old exclusive extension remains a valid
             * fixed address, but it consumes no interval in the primary block.
             * Fresh production families should never reach this case; retaining
             * it makes mixed diagnostic setup deterministic.
             */
            if (existing->second.base != block_)
                continue;

            occupied.push_back(Interval{
                .begin = existing->second.offset,
                .end = existing->second.offset + descriptor.size_bytes,
                .name = descriptor.name,
            });
        }

        auto intervalOrder = [](const Interval &lhs, const Interval &rhs)
        {
            if (lhs.begin != rhs.begin)
                return lhs.begin < rhs.begin;
            if (lhs.end != rhs.end)
                return lhs.end < rhs.end;
            return lhs.name < rhs.name;
        };
        std::sort(occupied.begin(), occupied.end(), intervalOrder);
        for (size_t i = 1; i < occupied.size(); ++i)
        {
            if (occupied[i].begin < occupied[i - 1].end)
            {
                LOG_ERROR("[DeviceWorkspaceManager] Serial graph participant requests previously published aliases '"
                          << occupied[i - 1].name << "' and '" << occupied[i].name
                          << "' whose primary-block lifetimes overlap; the graph-family declaration is invalid");
                return false;
            }
        }

        /*
         * Largest-first placement reduces fragmentation for model-sized state
         * captures. Name ordering makes the resulting addresses reproducible
         * across runs even if consumer discovery order changes.
         */
        std::sort(
            missing.begin(),
            missing.end(),
            [](const WorkspaceDescriptor *lhs,
               const WorkspaceDescriptor *rhs)
            {
                if (lhs->size_bytes != rhs->size_bytes)
                    return lhs->size_bytes > rhs->size_bytes;
                return lhs->name < rhs->name;
            });

        std::vector<PendingAlias> pending;
        pending.reserve(missing.size());
        for (const WorkspaceDescriptor *descriptor : missing)
        {
            size_t candidate = 0;
            bool placed = false;
            std::sort(occupied.begin(), occupied.end(), intervalOrder);
            for (const Interval &interval : occupied)
            {
                candidate = alignUp(candidate, descriptor->alignment);
                if (candidate <= interval.begin &&
                    descriptor->size_bytes <= interval.begin - candidate)
                {
                    placed = true;
                    break;
                }
                candidate = std::max(candidate, interval.end);
            }

            if (!placed)
            {
                candidate = alignUp(candidate, descriptor->alignment);
                placed =
                    candidate <= block_size_ &&
                    descriptor->size_bytes <= block_size_ - candidate;
            }
            if (!placed)
            {
                LOG_ERROR("[DeviceWorkspaceManager] Serial graph participant cannot fit buffer '"
                          << descriptor->name << "' (" << descriptor->size_bytes
                          << " bytes) into the " << block_size_
                          << "-byte primary family workspace on "
                          << device_.to_string()
                          << "; no append-only fallback is permitted");
                return false;
            }

            pending.push_back(PendingAlias{
                .descriptor = descriptor,
                .offset = candidate,
            });
            occupied.push_back(Interval{
                .begin = candidate,
                .end = candidate + descriptor->size_bytes,
                .name = descriptor->name,
            });
        }

        for (const PendingAlias &alias : pending)
        {
            const WorkspaceDescriptor &descriptor = *alias.descriptor;
            buffers_[descriptor.name] = BufferInfo{
                .base = block_,
                .offset = alias.offset,
                .size = descriptor.size_bytes,
            };
            logVramBomLine(
                "workspace_serial_alias",
                "device=" + device_.to_string() +
                    " manager_id=" + std::to_string(id_) +
                    " name=" + descriptor.name +
                    " offset_bytes=" + std::to_string(alias.offset) +
                    " primary_block_bytes=" + std::to_string(block_size_) +
                    " requested_bytes=" +
                    std::to_string(descriptor.size_bytes));
            PerfStatsCollector::addCounter(
                "memory",
                "workspace_serial_alias_bytes",
                static_cast<double>(descriptor.size_bytes),
                "materialize",
                device_.to_string(),
                {{"name", descriptor.name},
                 {"offset_bytes", std::to_string(alias.offset)},
                 {"primary_block_bytes", std::to_string(block_size_)}});
        }

        PerfStatsCollector::addCounter(
            "memory",
            "workspace_serial_participant_bindings",
            1.0,
            "materialize",
            device_.to_string(),
            {{"buffer_count", std::to_string(requirements.buffers.size())},
             {"new_alias_count", std::to_string(pending.size())},
             {"physical_growth_bytes", "0"}});
        return true;
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
        std::vector<SerialWorkspaceBufferPlacement> placements;
        placements.reserve(buffers.size());
        size_t current_offset = 0;
        for (const WorkspaceDescriptor *buffer : buffers)
        {
            current_offset = alignUp(current_offset, buffer->alignment);
            placements.push_back(SerialWorkspaceBufferPlacement{
                .descriptor = *buffer,
                .offset = current_offset,
            });
            current_offset += buffer->size_bytes;
        }
        const bool allocated =
            allocatePlacedBuffers(placements, total_size);
        if (allocated)
        {
            /*
             * The generic allocator may reserve an unmapped tail for legacy
             * callers. Preserve its historical logical-usage accounting;
             * allocateSerialFamily owns and reports its complete physical plan.
             */
            used_bytes_ = current_offset;
        }
        return allocated;
    }

    bool DeviceWorkspaceManager::allocatePlacedBuffers(
        const std::vector<SerialWorkspaceBufferPlacement> &placements,
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
            for (const auto &placement : placements)
            {
                max_alignment =
                    std::max(
                        max_alignment,
                        placement.descriptor.alignment);
            }
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
                " buffer_count=" + std::to_string(placements.size()) +
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
             {"buffer_count", std::to_string(placements.size())},
             {"bytes", std::to_string(total_size)}});

        // Publish every planned name at its stable pre-capture offset.
        for (const auto &placement : placements)
        {
            const WorkspaceDescriptor &buffer = placement.descriptor;
            if (placement.offset > total_size ||
                buffer.size_bytes > total_size - placement.offset)
            {
                LOG_ERROR("[DeviceWorkspaceManager] Workspace placement for '"
                          << buffer.name << "' exceeds the primary block on "
                          << device_.to_string());
                backend->free(block_, device_ordinal);
                block_ = nullptr;
                block_size_ = 0;
                buffers_.clear();
                used_bytes_ = 0;
                return false;
            }

            BufferInfo info;
            info.base = block_;
            info.offset = placement.offset;
            info.size = buffer.size_bytes;
            buffers_[buffer.name] = info;

            void *buf_ptr =
                static_cast<char *>(block_) + placement.offset;
            if (!device_.is_cpu())
            {
                const auto addr = reinterpret_cast<std::uintptr_t>(buf_ptr);
                if ((addr & (buffer.alignment - 1)) != 0)
                {
                    LOG_ERROR("[DeviceWorkspaceManager] Workspace buffer '" << buffer.name
                                                                            << "' on " << device_.to_string()
                                                                            << " is not aligned to " << buffer.alignment
                                                                            << " bytes (ptr=" << buf_ptr
                                                                            << ", offset=" << placement.offset << ")");
                    backend->free(block_, device_ordinal);
                    block_ = nullptr;
                    block_size_ = 0;
                    buffers_.clear();
                    used_bytes_ = 0;
                    return false;
                }
            }
            LOG_TRACE("[WORKSPACE_SUBALLOC] '" << buffer.name << "'"
                                               << " ptr=" << buf_ptr
                                               << " offset=" << placement.offset
                                               << " size=" << buffer.size_bytes
                                               << " device=" << device_.to_string());
            logVramBomLine(
                "workspace_buffer",
                "device=" + device_.to_string() +
                " manager_id=" + std::to_string(id_) +
                " name=" + buffer.name +
                " required=" + (buffer.required ? "true" : "false") +
                " alignment=" + std::to_string(buffer.alignment) +
                " offset_bytes=" + std::to_string(placement.offset) +
                " " + vramBomBytes(buffer.size_bytes));
            PerfStatsCollector::addCounter(
                "memory",
                "workspace_suballoc_bytes",
                static_cast<double>(buffer.size_bytes),
                "allocate",
                device_.to_string(),
                {{"name", buffer.name},
                 {"required", buffer.required ? "true" : "false"},
                 {"alignment", std::to_string(buffer.alignment)},
                 {"offset_bytes", std::to_string(placement.offset)},
                 {"bytes", std::to_string(buffer.size_bytes)}});
        }

        used_bytes_ = total_size;
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

    bool DeviceWorkspaceManager::rewritePersistentPublication(
            const std::string &domain,
            const PersistentWorkspacePublicationKey &new_key,
            const std::shared_ptr<void> &publication,
            size_t slot,
            size_t slot_capacity,
            const PersistentWorkspacePublicationRewrite &rewrite)
    {
        if (domain.empty() || !publication || slot_capacity == 0 ||
            slot >= slot_capacity || !rewrite)
        {
            LOG_ERROR("[DeviceWorkspaceManager] Ordered publication rewrite "
                      "requires a valid domain, publication, slot, capacity, "
                      "and producer");
            return false;
        }

        auto registry = persistent_slot_registry_;
        std::lock_guard<std::mutex> lock(registry->mutex);
        const auto slots_it = registry->occupied_slots.find(domain);
        if (slots_it == registry->occupied_slots.end() ||
            slots_it->second.size() != slot_capacity ||
            slot >= slots_it->second.size() ||
            !slots_it->second[slot])
        {
            LOG_ERROR("[DeviceWorkspaceManager] Ordered publication rewrite "
                      "does not own its declared slot in domain '"
                      << domain << "'");
            return false;
        }

        auto &publications = registry->immutable_publications[domain];
        if (!rewrite())
        {
            return false;
        }
        for (auto it = publications.begin(); it != publications.end();)
        {
            if (it->second.publication == publication)
            {
                if (it->second.slot != slot)
                {
                    LOG_ERROR("[DeviceWorkspaceManager] Ordered publication "
                              "changed physical slot in domain '" << domain
                              << "'");
                    return false;
                }
                it = publications.erase(it);
            }
            else
            {
                ++it;
            }
        }

        const auto canonical = publications.find(new_key);
        if (canonical == publications.end())
        {
            publications.emplace(
                new_key,
                detail::PersistentWorkspacePublicationRecord{
                    .slot = slot,
                    .publication = publication,
                });
        }
        return true;
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
