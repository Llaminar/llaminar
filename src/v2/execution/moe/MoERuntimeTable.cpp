/**
 * @file MoERuntimeTable.cpp
 * @brief Stable graph-facing MoE placement runtime tables.
 */

#include "MoERuntimeTable.h"

#include "DecodeExpertHistogram.h"
#include "DeviceMoEOverlayEpochArena.h"
#include "MoEOverlayEconomyCalibrationPlanner.h"
#include "../../backends/BackendManager.h"
#include "../../utils/Logger.h"
#include "../../utils/PerfStatsCollector.h"
#include "../../utils/VramBillOfMaterials.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace llaminar2
{
    namespace
    {
        bool descriptorReady(const DeviceMoEExpertDescriptor &desc)
        {
            return desc.weightsReady();
        }

        uint32_t portableMoEExpertFlags(uint32_t flags) noexcept
        {
            return flags & ~toMoEExpertFlags(DeviceMoEExpertFlags::TransferSlot);
        }

        uint32_t clearPayloadBearingMoEExpertFlags(uint32_t flags) noexcept
        {
            return flags & ~(toMoEExpertFlags(DeviceMoEExpertFlags::Valid) |
                             toMoEExpertFlags(DeviceMoEExpertFlags::Resident) |
                             toMoEExpertFlags(DeviceMoEExpertFlags::LocalCompute) |
                             toMoEExpertFlags(DeviceMoEExpertFlags::TransferSlot));
        }

        /**
         * @brief Compare portable placement semantics while ignoring request history.
         *
         * Active epochs, histograms, physical payload pointers, and rolling slot
         * identities do not decide where an expert is logically resident or
         * eligible to execute. Keeping this comparison beside the runtime-table
         * ABI prevents prefix restore from manufacturing a placement movement
         * merely because it imported newer counters.
         */
        bool portablePlacementMatchesRuntime(
            const DeviceMoEPortableLayerRuntimeState &portable,
            const DeviceMoELayerRuntime &runtime) noexcept
        {
            if (runtime.active_bank > 1u ||
                portable.expert_count != runtime.expert_count ||
                portable.top_k != runtime.top_k ||
                portable.participant_id != runtime.participant_id ||
                portable.participant_count != runtime.participant_count ||
                portable.experts.size() !=
                    static_cast<size_t>(runtime.expert_count))
            {
                return false;
            }

            const auto &bank = runtime.banks[runtime.active_bank];
            for (uint32_t expert = 0; expert < runtime.expert_count; ++expert)
            {
                const auto &saved = portable.experts[expert];
                const auto &live = bank.experts[expert];
                const int32_t live_logical_expert =
                    live.logical_expert_id >= 0
                        ? live.logical_expert_id
                        : static_cast<int32_t>(expert);
                const uint32_t saved_policy_flags =
                    clearPayloadBearingMoEExpertFlags(
                        portableMoEExpertFlags(saved.flags));
                const uint32_t live_policy_flags =
                    clearPayloadBearingMoEExpertFlags(
                        portableMoEExpertFlags(live.flags));

                if (saved.logical_expert_id != live_logical_expert ||
                    saved.owner_participant != live.owner_participant ||
                    saved.local_compute !=
                        (bank.local_compute_mask[expert] != 0u ? 1u : 0u) ||
                    saved.replica_role != bank.replica_role[expert] ||
                    saved.resident_participant_mask !=
                        bank.resident_participant_mask[expert] ||
                    saved_policy_flags != live_policy_flags)
                {
                    return false;
                }
            }
            return true;
        }

        bool descriptorRequiresReadyPayload(const DeviceMoEExpertDescriptor &desc, uint8_t local_compute)
        {
            return local_compute != 0 ||
                   hasMoEExpertFlag(desc.flags, DeviceMoEExpertFlags::Valid) ||
                   hasMoEExpertFlag(desc.flags, DeviceMoEExpertFlags::Resident) ||
                   hasMoEExpertFlag(desc.flags, DeviceMoEExpertFlags::LocalCompute);
        }

        std::string layerPrefix(int layer_idx)
        {
            return "[MoERuntimeTable] layer " + std::to_string(layer_idx) + ": ";
        }

        IBackend *mirrorBackend(DeviceId device, const std::string &what)
        {
            if (!device.is_gpu())
                throw std::runtime_error(what + ": mirrored MoE runtime tables require a GPU device, got " + device.to_string());
            IBackend *backend = getBackendFor(device);
            if (!backend)
                throw std::runtime_error(what + ": no backend available for " + device.to_string());
            return backend;
        }

        void *allocateMirror(DeviceId device, size_t bytes, const std::string &what)
        {
            IBackend *backend = mirrorBackend(device, what);
            void *ptr = backend->allocate(bytes, device.toKernelDeviceIndex());
            if (!ptr)
                throw std::runtime_error(what + ": backend allocation failed for " + std::to_string(bytes) + " bytes on " + device.to_string());
            return ptr;
        }

        void freeMirror(DeviceId device, void *ptr, const std::string &what) noexcept
        {
            if (!ptr)
                return;
            try
            {
                IBackend *backend = getBackendFor(device);
                if (!backend)
                {
                    LOG_ERROR(what << ": no backend available for " << device.to_string());
                    return;
                }
                backend->free(ptr, device.toKernelDeviceIndex());
            }
            catch (const std::exception &e)
            {
                LOG_ERROR(what << ": backend free failed for " << device.to_string() << ": " << e.what());
            }
            catch (...)
            {
                LOG_ERROR(what << ": backend free failed for " << device.to_string() << ": unknown exception");
            }
        }

        void copyHostToMirror(DeviceId device, void *dst, const void *src,
                              size_t bytes, void *stream, const std::string &what)
        {
            IBackend *backend = mirrorBackend(device, what);
            const int ordinal = device.toKernelDeviceIndex();
            const bool ok = stream
                                ? backend->hostToDeviceOnStream(dst, src, bytes, ordinal, stream)
                                : backend->hostToDevice(dst, src, bytes, ordinal, nullptr);
            if (!ok)
                throw std::runtime_error(what + ": backend H2D copy failed on " + device.to_string());
        }

        /**
         * @brief Enqueue a device-owned runtime-table restore.
         *
         * Both endpoints are immutable/model-lifetime GPU allocations.  This
         * helper deliberately accepts only an explicit producer stream and
         * returns after enqueueing the copy.  Consumers inherit ordering from
         * that stream or from its published event; no host synchronization is
         * part of the request lifecycle.
         */
        void copyMirrorToMirrorAsync(
            DeviceId device,
            void *dst,
            const void *src,
            size_t bytes,
            void *stream,
            const std::string &what)
        {
            if (!stream)
            {
                throw std::invalid_argument(
                    what + ": device-owned runtime reset requires an explicit stream");
            }
            IBackend *backend = mirrorBackend(device, what);
            if (!backend->deviceCopyAsync(
                    dst,
                    src,
                    bytes,
                    device.toKernelDeviceIndex(),
                    stream))
            {
                throw std::runtime_error(
                    what + ": backend D2D copy failed on " + device.to_string());
            }
        }

        void copyMirrorToHost(DeviceId device, void *dst, const void *src,
                              size_t bytes, void *stream, const std::string &what)
        {
            IBackend *backend = mirrorBackend(device, what);
            const bool ok = backend->deviceToHostFast(dst, src, bytes, device.toKernelDeviceIndex(), stream);
            if (!ok)
                throw std::runtime_error(what + ": backend D2H copy failed on " + device.to_string());
        }

        void memsetMirror(DeviceId device, void *dst, int value,
                          size_t bytes, void *stream, const std::string &what)
        {
            IBackend *backend = mirrorBackend(device, what);
            if (!backend->memset(dst, value, bytes, device.toKernelDeviceIndex(), stream))
                throw std::runtime_error(what + ": backend memset failed on " + device.to_string());
        }

        void synchronizeMirror(DeviceId device, void *stream, const std::string &what)
        {
            IBackend *backend = mirrorBackend(device, what);
            const int ordinal = device.toKernelDeviceIndex();
            const bool ok = stream
                                ? backend->synchronizeStream(stream, ordinal)
                                : backend->streamSynchronize(ordinal);
            if (!ok)
                throw std::runtime_error(what + ": backend stream synchronization failed on " + device.to_string());
        }

        void *createMirrorStream(DeviceId device, const std::string &what)
        {
            IBackend *backend = mirrorBackend(device, what);
            void *stream = backend->createStream(device.toKernelDeviceIndex());
            if (!stream)
                throw std::runtime_error(what + ": backend stream creation failed on " + device.to_string());
            return stream;
        }

        void destroyMirrorStream(DeviceId device, void *stream, const std::string &what) noexcept
        {
            if (!stream)
                return;
            try
            {
                IBackend *backend = getBackendFor(device);
                if (!backend)
                {
                    LOG_ERROR(what << ": no backend available for " << device.to_string());
                    return;
                }
                backend->destroyStream(stream, device.toKernelDeviceIndex());
            }
            catch (const std::exception &e)
            {
                LOG_ERROR(what << ": backend stream destroy failed for " << device.to_string() << ": " << e.what());
            }
            catch (...)
            {
                LOG_ERROR(what << ": backend stream destroy failed for " << device.to_string() << ": unknown exception");
            }
        }

        uint32_t checkedRouteCapacity(int token_capacity, int top_k)
        {
            if (token_capacity < 0)
                throw std::invalid_argument("[MoERuntimeTable] prefill token capacity must be non-negative");
            const uint64_t route_capacity = static_cast<uint64_t>(token_capacity) * static_cast<uint64_t>(top_k);
            if (route_capacity > std::numeric_limits<uint32_t>::max())
                throw std::invalid_argument("[MoERuntimeTable] prefill route capacity exceeds uint32_t range");
            return static_cast<uint32_t>(route_capacity);
        }

        void releasePrefillRouteScratchBindings(
            DeviceId device,
            DeviceMoEPrefillRouteScratchBindings &scratch,
            const std::string &label) noexcept
        {
            freeMirror(device, scratch.route_expert_ids, label + " route_expert_ids");
            freeMirror(device, scratch.route_weights, label + " route_weights");
            freeMirror(
                device,
                scratch.route_participant_ids,
                label + " route_participant_ids");
            freeMirror(device, scratch.expert_counts, label + " expert_counts");
            freeMirror(device, scratch.expert_offsets, label + " expert_offsets");
            freeMirror(
                device,
                scratch.grouped_token_ids,
                label + " grouped_token_ids");
            freeMirror(
                device,
                scratch.grouped_route_weights,
                label + " grouped_route_weights");
            freeMirror(device, scratch.llep_split_ends, label + " llep_split_ends");
            freeMirror(
                device,
                scratch.llep_assignment_spans,
                label + " llep_assignment_spans");
            freeMirror(
                device,
                scratch.llep_weight_transfers,
                label + " llep_weight_transfers");
            scratch = {};
        }

        void allocatePrefillRouteScratchBindings(
            DeviceId device,
            int num_experts,
            int top_k,
            int token_capacity,
            DeviceMoEPrefillRouteScratchBindings &allocation,
            const std::string &label)
        {
            if (!device.is_gpu())
                throw std::invalid_argument(label + ": route scratch requires a GPU device");
            if (num_experts <= 0 ||
                num_experts > static_cast<int>(kDeviceMoEMaxExperts))
            {
                throw std::invalid_argument(label + ": num_experts is out of range");
            }
            if (top_k <= 0 || top_k > static_cast<int>(kDeviceMoEMaxTopK))
                throw std::invalid_argument(label + ": top_k is out of range");
            if (token_capacity <= 0)
                throw std::invalid_argument(label + ": token_capacity must be positive");

            const uint32_t route_capacity =
                checkedRouteCapacity(token_capacity, top_k);
            auto allocate = [&](auto **ptr, size_t count, const char *name)
            {
                using Pointer =
                    std::remove_pointer_t<std::remove_pointer_t<decltype(ptr)>>;
                *ptr = static_cast<Pointer *>(
                    allocateMirror(
                        device,
                        count * sizeof(Pointer),
                        label + " allocation failed for " + name));
            };

            try
            {
                const size_t llep_plan_capacity =
                    static_cast<size_t>(num_experts) *
                    static_cast<size_t>(kDeviceMoEMaxParticipants);
                allocate(
                    &allocation.route_expert_ids,
                    route_capacity,
                    "route_expert_ids");
                allocate(
                    &allocation.route_weights,
                    route_capacity,
                    "route_weights");
                allocate(
                    &allocation.route_participant_ids,
                    route_capacity,
                    "route_participant_ids");
                allocate(
                    &allocation.expert_counts,
                    static_cast<size_t>(num_experts),
                    "expert_counts");
                allocate(
                    &allocation.expert_offsets,
                    static_cast<size_t>(num_experts),
                    "expert_offsets");
                allocate(
                    &allocation.grouped_token_ids,
                    route_capacity,
                    "grouped_token_ids");
                allocate(
                    &allocation.grouped_route_weights,
                    route_capacity,
                    "grouped_route_weights");
                allocate(
                    &allocation.llep_split_ends,
                    llep_plan_capacity,
                    "llep_split_ends");
                allocate(
                    &allocation.llep_assignment_spans,
                    llep_plan_capacity,
                    "llep_assignment_spans");
                allocate(
                    &allocation.llep_weight_transfers,
                    llep_plan_capacity,
                    "llep_weight_transfers");

                allocation.token_capacity =
                    static_cast<uint32_t>(token_capacity);
                allocation.route_capacity = route_capacity;
                allocation.expert_capacity =
                    static_cast<uint32_t>(num_experts);
                allocation.llep_plan_capacity =
                    static_cast<uint32_t>(llep_plan_capacity);
            }
            catch (...)
            {
                releasePrefillRouteScratchBindings(device, allocation, label);
                throw;
            }
        }

        size_t prefillRouteScratchBindingBytes(
            const DeviceMoEPrefillRouteScratchBindings &bindings) noexcept
        {
            const size_t route_capacity = bindings.route_capacity;
            const size_t expert_capacity = bindings.expert_capacity;
            const size_t llep_plan_capacity = bindings.llep_plan_capacity;
            return route_capacity *
                       (sizeof(int32_t) + sizeof(float) + sizeof(int32_t) +
                        sizeof(int32_t) + sizeof(float)) +
                   expert_capacity * (sizeof(int32_t) + sizeof(int32_t)) +
                   llep_plan_capacity *
                       (sizeof(int32_t) +
                        sizeof(
                            least_loaded_ep::
                                LeastLoadedExpertAssignmentSpan) +
                        sizeof(
                            least_loaded_ep::
                                LeastLoadedExpertWeightTransfer));
        }

        uint32_t participantBit(uint32_t participant)
        {
            return 1u << participant;
        }

        uint32_t participantMaskLimit(uint32_t participant_count)
        {
            return (1u << participant_count) - 1u;
        }

        uint32_t participantMaskCount(uint32_t mask) noexcept
        {
            uint32_t count = 0;
            while (mask != 0u)
            {
                mask &= (mask - 1u);
                ++count;
            }
            return count;
        }

        uint32_t synthesizedResidentParticipantMask(
            const MoEPlacementUpdate &update,
            uint32_t expert)
        {
            uint32_t mask = 0;
            const auto &desc = update.experts[expert];
            if (desc.owner_participant >= 0 &&
                desc.owner_participant < static_cast<int32_t>(update.participant_count))
            {
                mask |= participantBit(static_cast<uint32_t>(desc.owner_participant));
            }
            if (update.local_compute_mask[expert] != 0u)
                mask |= participantBit(update.participant_id);
            return mask;
        }

        uint32_t residentParticipantMaskForUpdate(
            const MoEPlacementUpdate &update,
            uint32_t expert)
        {
            if (!update.resident_participant_mask.empty())
                return update.resident_participant_mask[expert];
            return synthesizedResidentParticipantMask(update, expert);
        }

        /** @return Exact overlay-wide packet destination for one update entry. */
        int32_t overlayRouteParticipantForUpdate(
            const MoEPlacementUpdate &update,
            uint32_t expert) noexcept
        {
            if (!update.overlay_route_participant.empty())
                return update.overlay_route_participant[expert];
            return update.experts[expert].owner_participant;
        }

        void resetRouterHotCacheCounters(DeviceMoELayerRuntime &state) noexcept
        {
            state.router_hot_cache_eligible_dispatches = 0;
            state.router_hot_cache_used_dispatches = 0;
            state.router_hot_cache_improved_dispatches = 0;
            state.router_hot_cache_default_load_spread_total = 0;
            state.router_hot_cache_actual_load_spread_total = 0;
            state.router_hot_cache_load_spread_improvement_total = 0;
            state.router_hot_cache_active_dispatches = 0;
            state.router_hot_cache_miss_dispatches = 0;
            state.router_hot_cache_selected_expert_slots = 0;
            state.router_hot_cache_replicated_selected_expert_slots = 0;
        }

        /**
         * @brief Immutable-address bindings that survive every request reset.
         *
         * Placement contents and per-request counters are reset separately.
         * Route scratch, deferred ledgers, overlay placement, and external
         * histogram banks are model-topology pointers and must be restored as
         * one typed unit whenever a pristine runtime template is rebuilt.
         */
        struct RuntimePersistentBindings
        {
            int32_t *route_expert_ids = nullptr;
            float *route_weights = nullptr;
            int32_t *route_participant_ids = nullptr;
            int32_t *deferred_verifier_route_expert_ids = nullptr;
            int32_t *deferred_verifier_route_participant_ids = nullptr;
            int32_t *expert_counts = nullptr;
            int32_t *expert_offsets = nullptr;
            int32_t *grouped_token_ids = nullptr;
            float *grouped_route_weights = nullptr;
            float *grouped_gate_scratch = nullptr;
            float *grouped_up_scratch = nullptr;
            float *grouped_output_partials = nullptr;
            void *decode_scratch = nullptr;
            void *reserved_ptrs[3] = {};
            uint64_t reserved_u64[2] = {};
            uint32_t prefill_token_capacity = 0;
            uint32_t prefill_route_capacity = 0;
            uint32_t deferred_verifier_route_capacity = 0;
            DeviceMoERuntimeHistogramBank *runtime_histogram_banks = nullptr;
            const uint32_t *runtime_histogram_active_bank = nullptr;
            const DeviceMoEPlacementBank *overlay_placement_banks = nullptr;
        };

        RuntimePersistentBindings captureRuntimePersistentBindings(
            const DeviceMoELayerRuntime &state) noexcept
        {
            RuntimePersistentBindings scratch;
            scratch.route_expert_ids = state.route_expert_ids;
            scratch.route_weights = state.route_weights;
            scratch.route_participant_ids = state.route_participant_ids;
            scratch.deferred_verifier_route_expert_ids =
                state.deferred_verifier_route_expert_ids;
            scratch.deferred_verifier_route_participant_ids =
                state.deferred_verifier_route_participant_ids;
            scratch.expert_counts = state.expert_counts;
            scratch.expert_offsets = state.expert_offsets;
            scratch.grouped_token_ids = state.grouped_token_ids;
            scratch.grouped_route_weights = state.grouped_route_weights;
            scratch.grouped_gate_scratch = state.grouped_gate_scratch;
            scratch.grouped_up_scratch = state.grouped_up_scratch;
            scratch.grouped_output_partials = state.grouped_output_partials;
            scratch.decode_scratch = state.decode_scratch;
            scratch.reserved_ptrs[0] = state.reserved_ptrs[0];
            scratch.reserved_ptrs[1] = state.reserved_ptrs[1];
            scratch.reserved_ptrs[2] = state.reserved_ptrs[2];
            scratch.reserved_u64[0] = state.reserved_u64[0];
            scratch.reserved_u64[1] = state.reserved_u64[1];
            scratch.prefill_token_capacity = state.prefill_token_capacity;
            scratch.prefill_route_capacity = state.prefill_route_capacity;
            scratch.deferred_verifier_route_capacity =
                state.deferred_verifier_route_capacity;
            scratch.runtime_histogram_banks =
                state.runtime_histogram_banks;
            scratch.runtime_histogram_active_bank =
                state.runtime_histogram_active_bank;
            scratch.overlay_placement_banks =
                state.overlay_placement_banks;
            return scratch;
        }

        void restoreRuntimePersistentBindings(
            DeviceMoELayerRuntime &state,
            const RuntimePersistentBindings &scratch) noexcept
        {
            state.route_expert_ids = scratch.route_expert_ids;
            state.route_weights = scratch.route_weights;
            state.route_participant_ids = scratch.route_participant_ids;
            state.deferred_verifier_route_expert_ids =
                scratch.deferred_verifier_route_expert_ids;
            state.deferred_verifier_route_participant_ids =
                scratch.deferred_verifier_route_participant_ids;
            state.expert_counts = scratch.expert_counts;
            state.expert_offsets = scratch.expert_offsets;
            state.grouped_token_ids = scratch.grouped_token_ids;
            state.grouped_route_weights = scratch.grouped_route_weights;
            state.grouped_gate_scratch = scratch.grouped_gate_scratch;
            state.grouped_up_scratch = scratch.grouped_up_scratch;
            state.grouped_output_partials = scratch.grouped_output_partials;
            state.decode_scratch = scratch.decode_scratch;
            state.reserved_ptrs[0] = scratch.reserved_ptrs[0];
            state.reserved_ptrs[1] = scratch.reserved_ptrs[1];
            state.reserved_ptrs[2] = scratch.reserved_ptrs[2];
            state.reserved_u64[0] = scratch.reserved_u64[0];
            state.reserved_u64[1] = scratch.reserved_u64[1];
            state.reserved_u64[2] = 0;
            state.reserved_u64[3] = 0;
            state.prefill_token_capacity = scratch.prefill_token_capacity;
            state.prefill_route_capacity = scratch.prefill_route_capacity;
            state.deferred_verifier_route_capacity =
                scratch.deferred_verifier_route_capacity;
            state.runtime_histogram_banks =
                scratch.runtime_histogram_banks;
            state.runtime_histogram_active_bank =
                scratch.runtime_histogram_active_bank;
            state.overlay_placement_banks =
                scratch.overlay_placement_banks;
        }

        /** @brief Clear every exact production-phase routing counter. */
        void resetRuntimeHistogramFields(
            DeviceMoELayerRuntime &state,
            int num_experts) noexcept
        {
            std::fill(state.decode_histogram, state.decode_histogram + num_experts, 0ULL);
            std::fill(state.decode_local_histogram, state.decode_local_histogram + num_experts, 0ULL);
            std::fill(state.prefill_histogram, state.prefill_histogram + num_experts, 0ULL);
            std::fill(state.prefill_local_histogram, state.prefill_local_histogram + num_experts, 0ULL);
            std::fill(state.grouped_verifier_histogram,
                      state.grouped_verifier_histogram + num_experts,
                      0ULL);
            std::fill(state.grouped_verifier_local_histogram,
                      state.grouped_verifier_local_histogram + num_experts,
                      0ULL);
        }

        /** @brief Clear request-local demand, diagnostics, and LLEP evidence. */
        void resetPerRequestRuntimeFields(DeviceMoELayerRuntime &state, int num_experts) noexcept
        {
            resetRuntimeHistogramFields(state, num_experts);
            resetRouterHotCacheCounters(state);
            state.reserved_u64[2] = 0;
            state.reserved_u64[3] = 0;
            state.current_batch_llep_movement_observed = 0;
            state.current_batch_llep_non_owner_assignment_observed = 0;
            state.current_batch_llep_transient_bank_active = 0;
        }

        /**
         * @brief Find a payload-ready descriptor in one placement bank.
         *
         * Portable prefix-cache state records a slot only for model-lifetime
         * local payloads. When that static slot id is known, restore must not
         * accept another ready descriptor for the same logical expert. Rolling
         * transfer-slot replicas use `-1` and are rehydrated by the captured
         * device transaction instead of entering this lookup.
         */
        bool findReadyDescriptorForExpertInBank(const DeviceMoEPlacementBank &bank,
                                                uint32_t expert,
                                                int32_t expected_local_slot,
                                                DeviceMoEExpertDescriptor &out) noexcept
        {
            if (expert >= bank.expert_count || expert >= kDeviceMoEMaxExperts)
                return false;
            const auto &candidate = bank.experts[expert];
            if (candidate.logical_expert_id != static_cast<int32_t>(expert) ||
                (expected_local_slot >= 0 && candidate.local_slot != expected_local_slot) ||
                !descriptorReady(candidate))
            {
                return false;
            }
            out = candidate;
            return true;
        }

        bool findReadyDescriptorForExpert(const DeviceMoELayerRuntime &state,
                                          uint32_t expert,
                                          int32_t expected_local_slot,
                                          DeviceMoEExpertDescriptor &out) noexcept
        {
            if (state.active_bank <= 1u &&
                findReadyDescriptorForExpertInBank(
                    state.banks[state.active_bank], expert, expected_local_slot, out))
            {
                return true;
            }
            for (uint32_t bank_idx = 0; bank_idx < 2u; ++bank_idx)
            {
                if (bank_idx == state.active_bank)
                    continue;
                if (findReadyDescriptorForExpertInBank(
                        state.banks[bank_idx], expert, expected_local_slot, out))
                {
                    return true;
                }
            }
            return false;
        }

        /**
         * @brief Validate a local payload descriptor returned during portable restore.
         *
         * The restore path will add placement flags such as LocalCompute after it
         * binds the payload descriptor, so this check intentionally focuses on the
         * pointer-bearing contract: the descriptor must name the requested expert,
         * have complete gate/up/down payloads, and match the stable local slot
         * recorded in the portable blob when that slot is known.
         */
        bool descriptorMatchesPortableLocalClaim(const DeviceMoEExpertDescriptor &desc,
                                                 int expert,
                                                 int32_t expected_local_slot) noexcept
        {
            return desc.logical_expert_id == static_cast<int32_t>(expert) &&
                   (expected_local_slot < 0 || desc.local_slot == expected_local_slot) &&
                   descriptorReady(desc);
        }

        /**
         * @brief Decide whether a portable local expert must be rebound by resolver.
         *
         * A local-compute expert whose owner is another participant represents a
         * graph-owned transfer-slot replica.  Prefix restore must ask the graph
         * side transfer-slot directory for the current descriptor before looking
         * at old runtime banks, because the old banks may still contain descriptor
         * shapes that are "ready" but point into stale device allocations.
         */
        bool portableLocalReplicaRequiresResolver(
            const DeviceMoEPortableLayerRuntimeState &snapshot,
            const DeviceMoEPortableExpertRuntimeState &saved,
            const DeviceMoERuntimeTable::LocalPayloadDescriptorResolver &resolver) noexcept
        {
            return resolver &&
                   saved.local_compute != 0u &&
                   saved.local_slot >= 0 &&
                   saved.owner_participant >= 0 &&
                   saved.owner_participant != static_cast<int32_t>(snapshot.participant_id);
        }

    } // namespace

    void declareFullyReplicatedPlacementTopology(
        MoEPlacementUpdate &update,
        int local_participant,
        int participant_count,
        const std::vector<int> &owner_participants)
    {
        if (participant_count <= 0 ||
            participant_count > static_cast<int>(kDeviceMoEMaxParticipants))
        {
            throw std::invalid_argument(
                "fully replicated MoE placement participant_count must be in [1, " +
                std::to_string(kDeviceMoEMaxParticipants) + "]");
        }
        if (local_participant < 0 || local_participant >= participant_count)
        {
            throw std::invalid_argument(
                "fully replicated MoE placement local participant is outside the domain");
        }
        if (update.expert_count == 0 ||
            update.expert_count > kDeviceMoEMaxExperts ||
            update.experts.size() != static_cast<size_t>(update.expert_count))
        {
            throw std::invalid_argument(
                "fully replicated MoE placement requires one descriptor per logical expert");
        }
        if (owner_participants.size() != static_cast<size_t>(update.expert_count))
        {
            throw std::invalid_argument(
                "fully replicated MoE placement requires one canonical owner per logical expert");
        }

        const uint32_t all_participants_mask =
            (1u << static_cast<uint32_t>(participant_count)) - 1u;
        const uint32_t replicated_flag =
            toMoEExpertFlags(DeviceMoEExpertFlags::Replicated);
        const uint32_t preferred_owner_flag =
            toMoEExpertFlags(DeviceMoEExpertFlags::PreferredOwner);

        update.participant_id = static_cast<uint32_t>(local_participant);
        update.participant_count = static_cast<uint32_t>(participant_count);
        update.local_compute_mask.assign(
            static_cast<size_t>(update.expert_count), 1u);
        update.replica_role.resize(static_cast<size_t>(update.expert_count));
        update.resident_participant_mask.assign(
            static_cast<size_t>(update.expert_count),
            all_participants_mask);

        for (uint32_t expert = 0; expert < update.expert_count; ++expert)
        {
            const int owner = owner_participants[static_cast<size_t>(expert)];
            if (owner < 0 || owner >= participant_count)
            {
                throw std::invalid_argument(
                    "fully replicated MoE placement owner for expert " +
                    std::to_string(expert) + " is outside the domain");
            }

            auto &descriptor = update.experts[static_cast<size_t>(expert)];
            if (descriptor.logical_expert_id != static_cast<int32_t>(expert) ||
                descriptor.local_slot < 0 ||
                !descriptorReady(descriptor) ||
                !hasMoEExpertFlag(descriptor.flags, DeviceMoEExpertFlags::Valid) ||
                !hasMoEExpertFlag(descriptor.flags, DeviceMoEExpertFlags::Resident) ||
                !hasMoEExpertFlag(descriptor.flags, DeviceMoEExpertFlags::LocalCompute))
            {
                throw std::invalid_argument(
                    "fully replicated MoE placement descriptor for expert " +
                    std::to_string(expert) + " is not compute-ready");
            }

            descriptor.owner_participant = owner;
            descriptor.flags &= ~preferred_owner_flag;
            descriptor.flags |= replicated_flag;
            if (owner == local_participant)
                descriptor.flags |= preferred_owner_flag;

            update.replica_role[static_cast<size_t>(expert)] =
                static_cast<uint8_t>(
                    owner == local_participant
                        ? DeviceMoEReplicaRole::Primary
                        : DeviceMoEReplicaRole::Replica);
        }
    }

    DeviceMoESerialRouteScratchArena::DeviceMoESerialRouteScratchArena(
        Config config)
        : device_id_(config.device_id),
          num_experts_(config.num_experts),
          top_k_(config.top_k)
    {
        allocatePrefillRouteScratchBindings(
            device_id_,
            num_experts_,
            top_k_,
            config.token_capacity,
            bindings_,
            "[DeviceMoESerialRouteScratchArena]");

        const size_t bytes = allocationBytes();
        const PerfStatsCollector::Tags tags{
            {"bytes", std::to_string(bytes)},
            {"experts", std::to_string(num_experts_)},
            {"immutable", "true"},
            {"largest_participant", "true"},
            {"ownership", "per_device_serial_graph_domain"},
            {"route_capacity", std::to_string(bindings_.route_capacity)},
            {"token_capacity", std::to_string(bindings_.token_capacity)},
            {"top_k", std::to_string(top_k_)}};
        PerfStatsCollector::addCounter(
            "memory",
            "moe_serial_route_scratch_arena_allocations",
            1.0,
            "model_setup",
            device_id_.toString(),
            tags);
        PerfStatsCollector::addCounter(
            "memory",
            "moe_serial_route_scratch_arena_bytes",
            static_cast<double>(bytes),
            "model_setup",
            device_id_.toString(),
            tags);
        logVramBomLine(
            "moe_serial_route_scratch_arena",
            "device=" + device_id_.toString() +
                " ptr=" + vramBomPointer(bindings_.route_expert_ids) +
                " ownership=per_device_serial_graph_domain"
                " immutable=true largest_participant=true"
                " token_capacity=" +
                std::to_string(bindings_.token_capacity) +
                " route_capacity=" +
                std::to_string(bindings_.route_capacity) +
                " experts=" + std::to_string(num_experts_) +
                " top_k=" + std::to_string(top_k_) +
                " " + vramBomBytes(bytes));
    }

    DeviceMoESerialRouteScratchArena::~DeviceMoESerialRouteScratchArena()
    {
        releasePrefillRouteScratchBindings(
            device_id_,
            bindings_,
            "[DeviceMoESerialRouteScratchArena] free");
    }

    size_t DeviceMoESerialRouteScratchArena::allocationBytes() const noexcept
    {
        return prefillRouteScratchBindingBytes(bindings_);
    }

    DeviceMoERuntimeTable::DeviceMoERuntimeTable(Config config)
        : device_id_(config.device_id),
          num_layers_(config.num_layers),
          num_experts_(config.num_experts),
          top_k_(config.top_k),
          mirror_to_device_(config.mirror_to_device),
          grouped_verifier_histogram_publication_(
              config.grouped_verifier_histogram_publication),
          overlay_service_telemetry_coverage_(
              config.overlay_service_telemetry_coverage),
          overlay_service_telemetry_catalog_(
              std::move(config.overlay_service_telemetry_catalog)),
          prefill_token_capacity_(config.prefill_token_capacity),
          deferred_verifier_token_capacity_(
              config.deferred_verifier_token_capacity),
          serial_route_scratch_arena_(
              std::move(config.serial_route_scratch_arena)),
          overlay_epoch_arena_(std::move(config.overlay_epoch_arena)),
          overlay_epoch_ticket_slot_(config.overlay_epoch_ticket_slot),
          overlay_placement_source_(config.overlay_placement_source)
    {
        if (!device_id_.is_valid())
            throw std::invalid_argument("[MoERuntimeTable] device_id must be valid");
        if (num_layers_ <= 0)
            throw std::invalid_argument("[MoERuntimeTable] num_layers must be positive");
        if (num_experts_ <= 0 || num_experts_ > static_cast<int>(kDeviceMoEMaxExperts))
            throw std::invalid_argument("[MoERuntimeTable] num_experts must be in [1, " +
                                        std::to_string(kDeviceMoEMaxExperts) + "]");
        if (top_k_ <= 0 || top_k_ > static_cast<int>(kDeviceMoEMaxTopK))
            throw std::invalid_argument("[MoERuntimeTable] top_k must be in [1, " +
                                        std::to_string(kDeviceMoEMaxTopK) + "]");
        if (mirror_to_device_ && !device_id_.is_gpu())
            throw std::runtime_error("[MoERuntimeTable] device mirroring requires a GPU device");
        if (grouped_verifier_histogram_publication_ !=
                GroupedVerifierHistogramPublicationMode::Disabled &&
            !mirror_to_device_)
        {
            throw std::invalid_argument(
                "[MoERuntimeTable] grouped-verifier publication requires a mirrored GPU table");
        }
        const bool collects_overlay_service_telemetry =
            overlay_service_telemetry_coverage_ !=
            MoEOverlayServiceTelemetryCoverage::Disabled;
        if (collects_overlay_service_telemetry && !mirror_to_device_)
        {
            throw std::invalid_argument(
                "[MoERuntimeTable] ExpertOverlay service telemetry requires a mirrored GPU table");
        }
        switch (overlay_service_telemetry_coverage_)
        {
        case MoEOverlayServiceTelemetryCoverage::Disabled:
            if (overlay_service_telemetry_catalog_)
            {
                throw std::invalid_argument(
                    "[MoERuntimeTable] disabled service telemetry cannot retain a layer catalog");
            }
            break;
        case MoEOverlayServiceTelemetryCoverage::AllRuntimeLayers:
            if (overlay_service_telemetry_catalog_)
            {
                throw std::invalid_argument(
                    "[MoERuntimeTable] all-layer service telemetry cannot also select catalog representatives");
            }
            break;
        case MoEOverlayServiceTelemetryCoverage::CatalogStratifiedSample:
            if (!overlay_service_telemetry_catalog_ ||
                overlay_service_telemetry_catalog_->layerCount() <
                    static_cast<std::size_t>(num_layers_))
            {
                throw std::invalid_argument(
                    "[MoERuntimeTable] stratified service telemetry requires a catalog covering every runtime layer");
            }
            break;
        }
        if (prefill_token_capacity_ < 0)
            throw std::invalid_argument("[MoERuntimeTable] prefill_token_capacity must be non-negative");
        if (prefill_token_capacity_ > 0 && !mirror_to_device_)
            throw std::runtime_error("[MoERuntimeTable] prefill route scratch requires a mirrored GPU runtime table");
        if (deferred_verifier_token_capacity_ < 0)
            throw std::invalid_argument(
                "[MoERuntimeTable] deferred verifier token capacity must be non-negative");
        if (deferred_verifier_token_capacity_ > 0 && !mirror_to_device_)
        {
            throw std::runtime_error(
                "[MoERuntimeTable] deferred verifier route ledger requires a "
                "mirrored GPU runtime table");
        }
        if (serial_route_scratch_arena_)
        {
            if (!mirror_to_device_)
            {
                throw std::invalid_argument(
                    "[MoERuntimeTable] serial route scratch arena requires a "
                    "mirrored GPU runtime table");
            }
            if (serial_route_scratch_arena_->deviceId() != device_id_ ||
                serial_route_scratch_arena_->expertCount() != num_experts_ ||
                serial_route_scratch_arena_->topK() != top_k_)
            {
                throw std::invalid_argument(
                    "[MoERuntimeTable] serial route scratch arena shape/device "
                    "does not match the runtime table");
            }
            if (prefill_token_capacity_ >
                serial_route_scratch_arena_->tokenCapacity())
            {
                throw std::invalid_argument(
                    "[MoERuntimeTable] requested prefill capacity exceeds the "
                    "immutable serial route scratch arena");
            }
            prefill_token_capacity_ =
                serial_route_scratch_arena_->tokenCapacity();
            PerfStatsCollector::addCounter(
                "memory",
                "moe_serial_route_scratch_runtime_table_bindings",
                1.0,
                "model_setup",
                device_id_.toString(),
                {{"arena_bytes",
                  std::to_string(
                      serial_route_scratch_arena_->allocationBytes())},
                 {"arena_token_capacity",
                  std::to_string(
                      serial_route_scratch_arena_->tokenCapacity())},
                 {"layers", std::to_string(num_layers_)},
                 {"ownership", "per_device_serial_graph_domain"}});
        }
        if (overlay_epoch_arena_)
        {
            if (overlay_epoch_arena_->deviceId() != device_id_)
            {
                throw std::invalid_argument(
                    "[MoERuntimeTable] ExpertOverlay epoch arena device does "
                    "not match the runtime table");
            }
            if (overlay_epoch_ticket_slot_ >=
                overlay_epoch_arena_->requestSlotCapacity())
            {
                throw std::invalid_argument(
                    "[MoERuntimeTable] ExpertOverlay epoch ticket slot is "
                    "outside the immutable arena capacity");
            }
            PerfStatsCollector::addCounter(
                "memory",
                "moe_overlay_epoch_runtime_table_bindings",
                1.0,
                "model_setup",
                device_id_.toString(),
                {{"layers", std::to_string(num_layers_)},
                 {"request_slot",
                  std::to_string(overlay_epoch_ticket_slot_)},
                 {"arena_bytes",
                  std::to_string(overlay_epoch_arena_->allocationBytes())}});
        }
        else if (overlay_epoch_ticket_slot_ != 0u)
        {
            throw std::invalid_argument(
                "[MoERuntimeTable] a nonzero ExpertOverlay ticket slot "
                "requires an epoch arena");
        }
        if (overlay_placement_source_)
        {
            if (grouped_verifier_histogram_publication_ !=
                GroupedVerifierHistogramPublicationMode::Disabled)
            {
                throw std::invalid_argument(
                    "[MoERuntimeTable] an ExpertOverlay child table cannot own the canonical grouped-verifier publication stream");
            }
            if (!overlay_epoch_arena_ || !mirror_to_device_ ||
                !overlay_placement_source_->isMirroredToDevice() ||
                overlay_placement_source_->deviceId() != device_id_ ||
                overlay_placement_source_->layerCount() < num_layers_ ||
                overlay_placement_source_->expertCount() != num_experts_ ||
                overlay_placement_source_->topK() != top_k_ ||
                !overlay_placement_source_->usesOverlayEpochTicket() ||
                overlay_placement_source_->overlayEpochTicket() !=
                    overlayEpochTicket() ||
                overlay_placement_source_->overlayPlacementSource() != nullptr)
            {
                throw std::invalid_argument(
                    "[MoERuntimeTable] overlay placement source must be the "
                    "canonical mirrored main table on the same device, cover "
                    "every target layer, and share the exact epoch ticket");
            }
            if (overlay_service_telemetry_coverage_ !=
                    overlay_placement_source_
                        ->overlayServiceTelemetryCoverage() ||
                overlay_service_telemetry_catalog_.get() !=
                    overlay_placement_source_
                        ->overlayServiceTelemetryCatalog().get())
            {
                throw std::invalid_argument(
                    "[MoERuntimeTable] child and canonical ExpertOverlay tables must share one service telemetry coverage authority");
            }
        }
        (void)checkedRouteCapacity(prefill_token_capacity_, top_k_);
        (void)checkedRouteCapacity(deferred_verifier_token_capacity_, top_k_);

        host_layers_.resize(static_cast<size_t>(num_layers_));
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            auto &state = host_layers_[static_cast<size_t>(layer_idx)];
            resetLayer(state);
            if (overlay_placement_source_)
            {
                state.overlay_placement_banks =
                    overlay_placement_source_->devicePlacementBanks(layer_idx);
            }
        }
        empty_host_layers_ = host_layers_;
        initial_host_layers_ = empty_host_layers_;
        initial_layer_captured_.assign(host_layers_.size(), 0u);
        decode_runtime_publication_required_.assign(host_layers_.size(), 0u);

        if (mirror_to_device_)
        {
            try
            {
                allocateDeviceMirror();
                if (grouped_verifier_histogram_publication_ ==
                    GroupedVerifierHistogramPublicationMode::AcceptedRows)
                {
                    allocateGroupedVerifierHistogramPublicationStream();
                }
                if (collects_overlay_service_telemetry &&
                    !overlay_placement_source_)
                {
                    allocateOverlayServiceTelemetry();
                }
                if (serial_route_scratch_arena_)
                {
                    for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
                    {
                        bindPrefillRouteScratchToLayer(
                            layer_idx,
                            serial_route_scratch_arena_->bindings_);
                    }
                }
                else if (prefill_token_capacity_ > 0)
                {
                    prefill_route_scratch_.resize(host_layers_.size());
                    for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
                        allocatePrefillRouteScratchForLayer(layer_idx, prefill_token_capacity_);
                }
                if (deferred_verifier_token_capacity_ > 0)
                    allocateDeferredVerifierRouteLedger();
                uploadAllLayerStates();
            }
            catch (...)
            {
                releaseRuntimeHistogramResources();
                releaseOverlayServiceTelemetry();
                releaseDeferredVerifierRouteLedger();
                releasePrefillRouteScratch();
                releaseDeviceMirror();
                throw;
            }
        }
    }

    DeviceMoERuntimeTable::DeviceMoERuntimeTable(DeviceId device_id,
                                                 int num_layers,
                                                 int num_experts,
                                                 int top_k,
                                                 bool mirror_to_device)
        : DeviceMoERuntimeTable(Config{.device_id = device_id,
                                       .num_layers = num_layers,
                                       .num_experts = num_experts,
                                       .top_k = top_k,
                                       .mirror_to_device = mirror_to_device})
    {
    }

    DeviceMoERuntimeTable::~DeviceMoERuntimeTable()
    {
        releaseRuntimeHistogramResources();
        releaseOverlayServiceTelemetry();
        releaseDeferredVerifierRouteLedger();
        releasePrefillRouteScratch();
        releaseDeviceMirror();
    }

    DeviceMoELayerRuntime *DeviceMoERuntimeTable::deviceLayerState(int layer_idx)
    {
        validateLayerIndex(layer_idx);
        if (mirror_to_device_)
            return device_layers_ + layer_idx;
        return host_layers_.data() + layer_idx;
    }

    DeviceMoEOverlayServiceTelemetryCell *
    DeviceMoERuntimeTable::deviceOverlayServiceTelemetry() const noexcept
    {
        return overlay_placement_source_
                   ? overlay_placement_source_
                         ->deviceOverlayServiceTelemetry()
                   : device_overlay_service_telemetry_;
    }

    DeviceMoEOverlayServiceTelemetrySample *
    DeviceMoERuntimeTable::deviceOverlayServiceTelemetrySample(
        int layer_idx) const noexcept
    {
        if (layer_idx < 0 || layer_idx >= num_layers_)
            return nullptr;
        if (overlay_placement_source_)
        {
            return overlay_placement_source_
                ->deviceOverlayServiceTelemetrySample(layer_idx);
        }
        return device_overlay_service_samples_
                   ? device_overlay_service_samples_ + layer_idx
                   : nullptr;
    }

    bool DeviceMoERuntimeTable::
        collectsDeviceOverlayServiceTelemetryForLayer(
            int layer_idx) const noexcept
    {
        if (layer_idx < 0 || layer_idx >= num_layers_)
            return false;
        if (overlay_placement_source_)
        {
            return overlay_placement_source_
                ->collectsDeviceOverlayServiceTelemetryForLayer(layer_idx);
        }
        switch (overlay_service_telemetry_coverage_)
        {
        case MoEOverlayServiceTelemetryCoverage::Disabled:
            return false;
        case MoEOverlayServiceTelemetryCoverage::AllRuntimeLayers:
            return true;
        case MoEOverlayServiceTelemetryCoverage::CatalogStratifiedSample:
            return overlay_service_telemetry_catalog_ &&
                   overlay_service_telemetry_catalog_
                       ->isServiceTelemetryLayer(layer_idx);
        }
        return false;
    }

    DeviceMoELayerRuntime &DeviceMoERuntimeTable::hostLayerState(int layer_idx)
    {
        validateLayerIndex(layer_idx);
        return host_layers_[static_cast<size_t>(layer_idx)];
    }

    const DeviceMoELayerRuntime &DeviceMoERuntimeTable::hostLayerState(int layer_idx) const
    {
        validateLayerIndex(layer_idx);
        return host_layers_[static_cast<size_t>(layer_idx)];
    }

    const DeviceMoEOverlayEpochTicket *
    DeviceMoERuntimeTable::overlayEpochTicket() const noexcept
    {
        return overlay_epoch_arena_
                   ? overlay_epoch_arena_->requestTicket(
                         overlay_epoch_ticket_slot_)
                   : nullptr;
    }

    const DeviceMoEOverlayEpochStatus *
    DeviceMoERuntimeTable::overlayEpochStatus() const noexcept
    {
        return overlay_epoch_arena_
                   ? overlay_epoch_arena_->requestStatus(
                         overlay_epoch_ticket_slot_)
                   : nullptr;
    }

    const DeviceMoEPlacementBank *
    DeviceMoERuntimeTable::devicePlacementBanks(int layer_idx) const
    {
        validateLayerIndex(layer_idx);
        const DeviceMoELayerRuntime *layer =
            mirror_to_device_
                ? device_layers_ + layer_idx
                : host_layers_.data() + layer_idx;
        return reinterpret_cast<const DeviceMoEPlacementBank *>(
            reinterpret_cast<const std::byte *>(layer) +
            offsetof(DeviceMoELayerRuntime, banks));
    }

    DeviceMoERuntimeBankPublicationRecipe
    DeviceMoERuntimeTable::preparedInactiveBankPublicationRecipe(
        int layer_idx,
        uint32_t epoch)
    {
        validateLayerIndex(layer_idx);
        if (!mirror_to_device_ || !device_id_.is_gpu() || !device_layers_)
        {
            throw std::logic_error(
                "[MoERuntimeTable] inactive-bank publication recipe requires a mirrored GPU table");
        }

        auto &state = host_layers_[static_cast<size_t>(layer_idx)];
        const uint32_t inactive_bank = 1u - state.active_bank;
        auto &prepared = state.banks[inactive_bank];
        if (epoch == 0u || prepared.epoch != epoch ||
            prepared.expert_count != static_cast<uint32_t>(num_experts_) ||
            epoch <= state.active_epoch)
        {
            throw std::logic_error(
                layerPrefix(layer_idx) +
                "requested GPU publication does not match the prepared inactive bank");
        }

        auto *const device_runtime = device_layers_ + layer_idx;
        DeviceMoERuntimeBankPublicationRecipe recipe{
            .bank = inactive_bank,
            .epoch = epoch,
            .host_bank = &prepared,
            .device_bank = &device_runtime->banks[inactive_bank],
            .device_runtime = device_runtime,
        };
        if (!recipe.valid())
        {
            throw std::logic_error(
                layerPrefix(layer_idx) +
                "constructed an invalid GPU inactive-bank publication recipe");
        }
        return recipe;
    }

    void DeviceMoERuntimeTable::acknowledgeDevicePublishedBank(
        int layer_idx,
        uint32_t epoch,
        uint32_t bank)
    {
        validateLayerIndex(layer_idx);
        if (!mirror_to_device_ || !device_id_.is_gpu() || !device_layers_)
        {
            throw std::logic_error(
                "[MoERuntimeTable] device publication acknowledgement requires a mirrored GPU table");
        }

        const size_t layer = static_cast<size_t>(layer_idx);
        auto &state = host_layers_[layer];
        const uint32_t inactive_bank = 1u - state.active_bank;
        if (bank != inactive_bank || bank >= kDeviceMoEOverlayEpochBankCount ||
            epoch == 0u || epoch <= state.active_epoch ||
            state.banks[bank].epoch != epoch ||
            state.banks[bank].expert_count !=
                static_cast<uint32_t>(num_experts_))
        {
            throw std::logic_error(
                layerPrefix(layer_idx) +
                "device publication acknowledgement disagrees with the prepared inactive bank");
        }

        /*
         * This is deliberately a host-only recipe transition.  Uploading the
         * entire DeviceMoELayerRuntime here would race and overwrite live
         * routing, histogram, and scratch fields owned solely by the GPU.
         */
        state.active_bank = bank;
        state.active_epoch = epoch;
        decode_runtime_publication_required_[layer] = 0u;
    }

    MoEOverlayRoutePlacementDeviceBinding
    DeviceMoERuntimeTable::overlayRoutePlacementBinding(int layer_idx) const
    {
        validateLayerIndex(layer_idx);
        if (!mirror_to_device_ || !device_id_.is_gpu() ||
            !usesOverlayEpochTicket())
        {
            return {};
        }

        /*
         * A sidecar's embedded banks are transient request-local scratch. Its
         * durable placement authority is the canonical main-model table named
         * at construction. Resolving that authority here prevents graph builders
         * from accidentally capturing the child's private bank addresses.
         */
        const DeviceMoERuntimeTable *const placement_authority =
            overlay_placement_source_ ? overlay_placement_source_ : this;
        const DeviceMoEPlacementBank *const banks =
            placement_authority->devicePlacementBanks(layer_idx);
        return {
            .banks = {
                MoEOverlayRoutePlacementBankDeviceView{
                    .route_participants =
                        banks[0].overlay_route_participant,
                    .epoch = &banks[0].epoch,
                },
                MoEOverlayRoutePlacementBankDeviceView{
                    .route_participants =
                        banks[1].overlay_route_participant,
                    .epoch = &banks[1].epoch,
                },
            },
            .ticket = overlayEpochTicket(),
            .status = overlayEpochStatus(),
            .expert_count = static_cast<uint32_t>(num_experts_),
        };
    }

    bool DeviceMoERuntimeTable::decodeRuntimePublicationRequired(int layer_idx) const
    {
        validateLayerIndex(layer_idx);
        return decode_runtime_publication_required_[static_cast<size_t>(layer_idx)] != 0u;
    }

    bool DeviceMoERuntimeTable::hasPrefillRouteScratchCapacity(int layer_idx, int token_count) const
    {
        validateLayerIndex(layer_idx);
        if (token_count <= 0)
            return false;
        const auto &state = host_layers_[static_cast<size_t>(layer_idx)];
        const uint32_t route_count = checkedRouteCapacity(token_count, top_k_);
        return state.prefill_token_capacity >= static_cast<uint32_t>(token_count) &&
               state.prefill_route_capacity >= route_count &&
               state.route_expert_ids &&
               state.route_weights &&
               state.route_participant_ids &&
               state.expert_counts &&
               state.expert_offsets &&
               state.grouped_token_ids &&
               state.grouped_route_weights &&
               state.reserved_ptrs[0] &&
               state.reserved_ptrs[1] &&
               state.reserved_ptrs[2] &&
               state.reserved_u64[0] >=
                   static_cast<uint64_t>(num_experts_) * static_cast<uint64_t>(kDeviceMoEMaxParticipants) &&
               state.reserved_u64[1] >=
                   static_cast<uint64_t>(num_experts_) * static_cast<uint64_t>(kDeviceMoEMaxParticipants);
    }

    bool DeviceMoERuntimeTable::hasDeferredVerifierRouteLedgerCapacity(
        int layer_idx,
        int token_count) const
    {
        validateLayerIndex(layer_idx);
        if (token_count <= 0)
            return false;
        const auto &state = host_layers_[static_cast<size_t>(layer_idx)];
        const uint32_t route_count = checkedRouteCapacity(token_count, top_k_);
        return state.deferred_verifier_route_capacity >= route_count &&
               state.deferred_verifier_route_expert_ids &&
               state.deferred_verifier_route_participant_ids;
    }

    void DeviceMoERuntimeTable::prepareDecodeHistogramProducerStream(void *stream)
    {
        if (mirror_to_device_ && !stream)
        {
            throw std::invalid_argument(
                "[MoERuntimeTable] mirrored decode histogram producer preparation requires an explicit stream");
        }

        std::lock_guard<std::mutex> lock(runtime_histogram_drain_mutex_);
        if (runtime_histogram_producer_lifecycle_ !=
            RuntimeHistogramProducerLifecycle::CollectingProducers)
        {
            throw std::logic_error(
                "[MoERuntimeTable] histogram producer preparation occurred after producer retirement");
        }
        if (mirror_to_device_ && runtime_histogram_drain_enabled_)
        {
            registerRuntimeHistogramProducerStreamLocked(
                stream,
                RuntimeHistogramProducerStream::Ownership::
                    BorrowedExecutionStream);
        }
        decode_histogram_producer_stream_ = stream;
    }

    void DeviceMoERuntimeTable::transitionDecodeHistogramProducerCapture(
        void *stream,
        RuntimeHistogramProducerCaptureTransition transition)
    {
        if (mirror_to_device_ && !stream)
        {
            throw std::invalid_argument(
                "[MoERuntimeTable] mirrored histogram capture transition requires an exact stream");
        }

        std::lock_guard<std::mutex> lock(runtime_histogram_drain_mutex_);
        if (runtime_histogram_producer_lifecycle_ !=
            RuntimeHistogramProducerLifecycle::CollectingProducers)
        {
            throw std::logic_error(
                "[MoERuntimeTable] histogram capture transition occurred after producer retirement");
        }
        if (!mirror_to_device_ || !runtime_histogram_drain_enabled_)
            return;

        const auto producer = std::find_if(
            runtime_histogram_producer_streams_.begin(),
            runtime_histogram_producer_streams_.end(),
            [stream](const RuntimeHistogramProducerStream &candidate)
            { return candidate.stream == stream; });
        if (producer == runtime_histogram_producer_streams_.end())
        {
            throw std::logic_error(
                "[MoERuntimeTable] native capture began on a histogram producer that was not admitted before beginCapture");
        }

        switch (transition)
        {
        case RuntimeHistogramProducerCaptureTransition::Entering:
            if (producer->active_capture_references ==
                std::numeric_limits<uint32_t>::max())
            {
                throw std::overflow_error(
                    "[MoERuntimeTable] histogram producer capture reference count overflowed");
            }
            ++producer->active_capture_references;
            break;
        case RuntimeHistogramProducerCaptureTransition::Completed:
        case RuntimeHistogramProducerCaptureTransition::Aborted:
            if (producer->active_capture_references == 0u)
            {
                throw std::logic_error(
                    "[MoERuntimeTable] histogram producer received a terminal capture edge without Entering");
            }
            --producer->active_capture_references;
            break;
        }
    }

    void DeviceMoERuntimeTable::recordDecodeHistogramProducerStream(void *stream)
    {
        if (mirror_to_device_ && !stream)
        {
            throw std::invalid_argument(
                "[MoERuntimeTable] mirrored decode histogram producer stream must be explicit");
        }

        std::lock_guard<std::mutex> lock(runtime_histogram_drain_mutex_);
        if (runtime_histogram_producer_lifecycle_ !=
            RuntimeHistogramProducerLifecycle::CollectingProducers)
        {
            throw std::logic_error(
                "[MoERuntimeTable] histogram producer publication occurred after producer retirement");
        }
        if (mirror_to_device_ && runtime_histogram_drain_enabled_ &&
            !isRuntimeHistogramProducerStreamRegisteredLocked(stream))
        {
            throw std::logic_error(
                "[MoERuntimeTable] decode histogram producer stream was not prepared before graph/eager execution");
        }
        /* This assignment is deliberately the only operation on the captured
         * path. Backend event allocation and initialization ordering belong to
         * prepareDecodeHistogramProducerStream(), before beginCapture(). */
        decode_histogram_producer_stream_ = stream;
    }

    void DeviceMoERuntimeTable::retireRuntimeHistogramProducerStreams()
    {
        std::lock_guard<std::mutex> lock(runtime_histogram_drain_mutex_);
        retireRuntimeHistogramProducerStreamsLocked();
    }

    void *DeviceMoERuntimeTable::decodeHistogramProducerStream() const
    {
        std::lock_guard<std::mutex> lock(runtime_histogram_drain_mutex_);
        return decode_histogram_producer_stream_;
    }

    void *DeviceMoERuntimeTable::
        groupedVerifierHistogramPublicationStream() const
    {
        std::lock_guard<std::mutex> lock(runtime_histogram_drain_mutex_);
        return grouped_verifier_histogram_publication_stream_;
    }

    void DeviceMoERuntimeTable::enableAsyncDecodeHistogramDrain(
        RuntimeExpertHistogramSourceMask sources)
    {
        if (std::none_of(sources.begin(), sources.end(), [](bool value)
                         { return value; }))
        {
            throw std::invalid_argument(
                "[MoERuntimeTable] asynchronous histogram drain must own at least one production phase");
        }

        const auto grouped_verifier_source = static_cast<std::size_t>(
            moe_runtime_abi::HistogramSource::GroupedVerifier);
        const bool drains_grouped_verifier =
            sources[grouped_verifier_source];
        const bool publishes_grouped_verifier =
            grouped_verifier_histogram_publication_ ==
            GroupedVerifierHistogramPublicationMode::AcceptedRows;
        if (drains_grouped_verifier != publishes_grouped_verifier)
        {
            throw std::invalid_argument(
                "[MoERuntimeTable] asynchronous grouped-verifier drain ownership must match the table's typed publication policy");
        }

        std::lock_guard<std::mutex> lock(runtime_histogram_drain_mutex_);
        if (runtime_histogram_producer_lifecycle_ !=
            RuntimeHistogramProducerLifecycle::CollectingProducers)
        {
            throw std::logic_error(
                "[MoERuntimeTable] asynchronous histogram drain enablement occurred after producer retirement");
        }
        if (runtime_histogram_drain_enabled_)
        {
            if (runtime_histogram_sources_ != sources)
            {
                throw std::logic_error(
                    "[MoERuntimeTable] asynchronous histogram drain source ownership cannot change after setup");
            }
            return;
        }

        runtime_histogram_sources_ = sources;
        try
        {
            if (mirror_to_device_)
            {
                allocateRuntimeHistogramDrainResources();

                /* A setup path may have recorded its graph stream just before
                 * the histogram owner was attached. Install its event edge
                 * before publishing the enabled lifecycle state. */
                if (decode_histogram_producer_stream_)
                {
                    registerRuntimeHistogramProducerStreamLocked(
                        decode_histogram_producer_stream_,
                        RuntimeHistogramProducerStream::Ownership::
                            BorrowedExecutionStream);
                }
            }
            runtime_histogram_drain_enabled_ = true;
        }
        catch (...)
        {
            releaseRuntimeHistogramResources();
            runtime_histogram_sources_ = {};
            throw;
        }
    }

    RuntimeExpertHistogramDrainResult
    DeviceMoERuntimeTable::progressAsyncDecodeHistogramDrain(
        DecodeExpertHistogram &histogram)
    {
        std::lock_guard<std::mutex> lock(runtime_histogram_drain_mutex_);
        if (runtime_histogram_producer_lifecycle_ !=
            RuntimeHistogramProducerLifecycle::CollectingProducers)
        {
            return RuntimeExpertHistogramDrainResult::failed(
                "Runtime histogram drain was polled after producer retirement");
        }
        if (!runtime_histogram_drain_enabled_)
        {
            return RuntimeExpertHistogramDrainResult::failed(
                "Runtime histogram drain was polled before model-setup enablement");
        }

        if (!mirror_to_device_)
        {
            return syncDecodeHistogramToHost(
                       histogram,
                       /*stream=*/nullptr,
                       /*reset_runtime_counts=*/true)
                       ? RuntimeExpertHistogramDrainResult::ready()
                       : RuntimeExpertHistogramDrainResult::failed(
                             "CPU runtime histogram merge failed");
        }

        if (hasActiveRuntimeHistogramProducerCaptureLocked())
        {
            /* Do not even query a previously submitted completion event while
             * the producer stream is in native capture. The graph owner has
             * exclusive backend use of that stream until its terminal edge;
             * maintenance remains a pure host-side Pending observation. */
            return RuntimeExpertHistogramDrainResult::pending();
        }

        IBackend *backend = mirrorBackend(
            device_id_,
            "[MoERuntimeTable] asynchronous runtime histogram drain");
        const int ordinal = device_id_.toKernelDeviceIndex();
        if (runtime_histogram_drain_in_flight_)
        {
            bool ready = false;
            if (!backend->queryEvent(
                    runtime_histogram_drain_complete_event_,
                    ordinal,
                    &ready))
            {
                return RuntimeExpertHistogramDrainResult::failed(
                    "Backend failed to query the exact runtime histogram drain event on " +
                    device_id_.to_string());
            }
            if (!ready)
                return RuntimeExpertHistogramDrainResult::pending();
            if (!mergeRuntimeHistogramSnapshot(histogram))
            {
                return RuntimeExpertHistogramDrainResult::failed(
                    "Completed runtime histogram snapshot did not match the host histogram geometry");
            }

            runtime_histogram_drain_in_flight_ = false;
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "runtime_histogram_async_drains_completed",
                1.0,
                "maintenance",
                device_id_.toString(),
                {{"blocking", "false"},
                 {"bank", std::to_string(
                              runtime_histogram_frozen_bank_host_)}});
            return RuntimeExpertHistogramDrainResult::ready();
        }

        if (runtime_histogram_producer_streams_.empty())
        {
            return RuntimeExpertHistogramDrainResult::failed(
                "Runtime histogram drain began before any exact producer stream was registered");
        }
        runtime_histogram_producer_topology_ =
            RuntimeHistogramProducerTopology::Sealed;
        const uint32_t frozen_bank = runtime_histogram_active_bank_host_;
        const uint32_t next_bank = 1u - frozen_bank;
        const bool admit_next_generation =
            admitsRuntimeExpertHistogramRows(histogram.admissionState());
        const uint32_t next_writer_state =
            moe_runtime_abi::makeHistogramWriterState(
                next_bank,
                admit_next_generation);

        /* Certification rotates into a quarantined bank; ordinary proposal
         * drains rotate into an admitted bank. One maintenance-stream writer
         * publishes the state after joining all previous producer work, then
         * every producer waits on its exact publication event. */
        std::string publication_failure;
        if (!publishRuntimeHistogramWriterStateLocked(
                next_writer_state,
                publication_failure))
        {
            return RuntimeExpertHistogramDrainResult::failed(
                std::move(publication_failure));
        }

        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            auto *source =
                device_runtime_histogram_banks_ +
                static_cast<std::size_t>(layer_idx) * 2u + frozen_bank;
            auto *destination =
                host_runtime_histogram_snapshot_ +
                static_cast<std::size_t>(layer_idx);
            if (!backend->deviceToHostOnStream(
                    destination,
                    source,
                    sizeof(DeviceMoERuntimeHistogramBank),
                    ordinal,
                    runtime_histogram_maintenance_stream_) ||
                !backend->memset(
                    source,
                    0,
                    sizeof(DeviceMoERuntimeHistogramBank),
                    ordinal,
                    runtime_histogram_maintenance_stream_))
            {
                return RuntimeExpertHistogramDrainResult::failed(
                    "Failed to enqueue an asynchronous runtime histogram bank drain/reset");
            }
        }
        if (!backend->recordEvent(
                runtime_histogram_drain_complete_event_,
                ordinal,
                runtime_histogram_maintenance_stream_))
        {
            return RuntimeExpertHistogramDrainResult::failed(
                "Failed to record the runtime histogram drain completion event");
        }

        runtime_histogram_frozen_bank_host_ = frozen_bank;
        runtime_histogram_active_bank_host_ = next_bank;
        runtime_histogram_drain_in_flight_ = true;
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "runtime_histogram_async_drains_started",
            1.0,
            "maintenance",
            device_id_.toString(),
            {{"blocking", "false"},
             {"producer_streams",
              std::to_string(runtime_histogram_producer_streams_.size())},
             {"bank", std::to_string(frozen_bank)}});
        return RuntimeExpertHistogramDrainResult::pending();
    }

    bool DeviceMoERuntimeTable::publishAsyncDecodeHistogramAdmission(
        RuntimeExpertHistogramAdmission admission)
    {
        if (admission !=
            RuntimeExpertHistogramAdmission::OptimizationDemand)
        {
            return false;
        }

        std::lock_guard<std::mutex> lock(runtime_histogram_drain_mutex_);
        if (runtime_histogram_producer_lifecycle_ !=
            RuntimeHistogramProducerLifecycle::CollectingProducers)
        {
            return false;
        }
        if (!runtime_histogram_drain_enabled_)
            return false;
        if (!mirror_to_device_)
            return true;
        if (runtime_histogram_drain_in_flight_ ||
            runtime_histogram_producer_streams_.empty() ||
            !host_runtime_histogram_writer_states_ ||
            hasActiveRuntimeHistogramProducerCaptureLocked())
        {
            return false;
        }

        const uint32_t writer_state =
            moe_runtime_abi::makeHistogramWriterState(
                runtime_histogram_active_bank_host_,
                /*admit_rows=*/true);
        std::string publication_failure;
        if (!publishRuntimeHistogramWriterStateLocked(
                writer_state,
                publication_failure))
        {
            LOG_ERROR(
                "[MoERuntimeTable] runtime histogram demand activation failed on "
                << device_id_.to_string() << ": "
                << publication_failure);
            return false;
        }

        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "runtime_histogram_demand_activations",
            1.0,
            "request_admission",
            device_id_.toString(),
            {{"blocking", "false"},
             {"producer_streams",
              std::to_string(runtime_histogram_producer_streams_.size())},
             {"bank",
              std::to_string(runtime_histogram_active_bank_host_)}});
        return true;
    }

    bool DeviceMoERuntimeTable::syncDecodeHistogramToHost(
        DecodeExpertHistogram &histogram,
        void *stream,
        bool reset_runtime_counts)
    {
        const auto &hist_config = histogram.config();
        if (hist_config.num_layers != num_layers_ ||
            hist_config.num_experts != num_experts_ ||
            hist_config.top_k != top_k_)
        {
            LOG_ERROR("[MoERuntimeTable] decode histogram config mismatch: table layers="
                      << num_layers_ << " experts=" << num_experts_ << " top_k=" << top_k_
                      << " histogram layers=" << hist_config.num_layers
                      << " experts=" << hist_config.num_experts
                      << " top_k=" << hist_config.top_k);
            return false;
        }

        if (mirror_to_device_ && !stream)
        {
            throw std::invalid_argument(
                "[MoERuntimeTable] mirrored decode histogram sync requires an explicit stream");
        }

        const std::size_t entry_count =
            static_cast<size_t>(num_layers_) *
            static_cast<size_t>(num_experts_);
        std::array<std::vector<uint64_t>,
                   kExpertHistogramProductionSourceCount>
            counts;
        std::array<std::vector<uint64_t>,
                   kExpertHistogramProductionSourceCount>
            local_counts;
        for (std::size_t source = 0;
             source < kExpertHistogramProductionSourceCount;
             ++source)
        {
            counts[source].assign(entry_count, 0);
            local_counts[source].assign(entry_count, 0);
        }
        constexpr std::array<ExpertHistogramSource,
                             kExpertHistogramProductionSourceCount>
            sources{
                ExpertHistogramSource::DecodeToken,
                ExpertHistogramSource::PrefillChunk,
                ExpertHistogramSource::GroupedVerifier,
            };
        constexpr std::array<const char *,
                             kExpertHistogramProductionSourceCount>
            source_names{"decode", "prefill", "grouped_verifier"};

        if (!mirror_to_device_)
        {
            for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            {
                const auto &state = host_layers_[static_cast<size_t>(layer_idx)];
                const std::array<const uint64_t *,
                                 kExpertHistogramProductionSourceCount>
                    selected_sources{
                        state.decode_histogram,
                        state.prefill_histogram,
                        state.grouped_verifier_histogram,
                    };
                const std::array<const uint64_t *,
                                 kExpertHistogramProductionSourceCount>
                    local_sources{
                        state.decode_local_histogram,
                        state.prefill_local_histogram,
                        state.grouped_verifier_local_histogram,
                    };
                const std::size_t layer_offset =
                    static_cast<size_t>(layer_idx) *
                    static_cast<size_t>(num_experts_);
                for (std::size_t source = 0;
                     source < kExpertHistogramProductionSourceCount;
                     ++source)
                {
                    std::copy(
                        selected_sources[source],
                        selected_sources[source] + num_experts_,
                        counts[source].data() + layer_offset);
                    std::copy(
                        local_sources[source],
                        local_sources[source] + num_experts_,
                        local_counts[source].data() + layer_offset);
                }
            }
        }
        else
        {
            for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            {
                const auto &state = device_layers_[layer_idx];
                const std::array<const uint64_t *,
                                 kExpertHistogramProductionSourceCount>
                    selected_sources{
                        state.decode_histogram,
                        state.prefill_histogram,
                        state.grouped_verifier_histogram,
                    };
                const std::array<const uint64_t *,
                                 kExpertHistogramProductionSourceCount>
                    local_sources{
                        state.decode_local_histogram,
                        state.prefill_local_histogram,
                        state.grouped_verifier_local_histogram,
                    };
                const std::size_t layer_offset =
                    static_cast<size_t>(layer_idx) *
                    static_cast<size_t>(num_experts_);
                for (std::size_t source = 0;
                     source < kExpertHistogramProductionSourceCount;
                     ++source)
                {
                    copyMirrorToHost(
                        device_id_,
                        counts[source].data() + layer_offset,
                        selected_sources[source],
                        static_cast<size_t>(num_experts_) * sizeof(uint64_t),
                        stream,
                        layerPrefix(layer_idx) + " " + source_names[source] +
                            " selected histogram D2H");
                    copyMirrorToHost(
                        device_id_,
                        local_counts[source].data() + layer_offset,
                        local_sources[source],
                        static_cast<size_t>(num_experts_) * sizeof(uint64_t),
                        stream,
                        layerPrefix(layer_idx) + " " + source_names[source] +
                            " local histogram D2H");
                }
            }
        }

        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            const std::size_t layer_offset =
                static_cast<size_t>(layer_idx) *
                static_cast<size_t>(num_experts_);
            uint64_t selected_slots = 0;
            uint64_t local_slots = 0;
            for (std::size_t source = 0;
                 source < kExpertHistogramProductionSourceCount;
                 ++source)
            {
                const auto *layer_counts =
                    counts[source].data() + layer_offset;
                const auto *layer_local_counts =
                    local_counts[source].data() + layer_offset;
                histogram.mergeLayerCounts(
                    layer_idx,
                    layer_counts,
                    num_experts_,
                    /*count_window_tokens=*/
                        sources[source] !=
                        ExpertHistogramSource::DecodeToken,
                    sources[source]);
                const uint64_t phase_selected_slots = std::accumulate(
                    layer_counts,
                    layer_counts + num_experts_,
                    uint64_t{0});
                const uint64_t phase_local_slots = std::accumulate(
                    layer_local_counts,
                    layer_local_counts + num_experts_,
                    uint64_t{0});
                selected_slots += phase_selected_slots;
                local_slots += phase_local_slots;

                if (PerfStatsCollector::isDomainEnabled("moe_rebalance"))
                {
                    const auto &state =
                        host_layers_[static_cast<size_t>(layer_idx)];
                    const PerfStatsCollector::Tags phase_tags{
                        {"layer", std::to_string(layer_idx)},
                        {"phase", source_names[source]},
                        {"participant", std::to_string(state.participant_id)},
                        {"participants", std::to_string(state.participant_count)},
                        {"active_epoch", std::to_string(state.active_epoch)},
                        {"reset", reset_runtime_counts ? "true" : "false"}};
                    PerfStatsCollector::addCounter(
                        "moe_rebalance",
                        "runtime_phase_selected_slots",
                        static_cast<double>(phase_selected_slots),
                        "rebalance",
                        device_id_.toString(),
                        phase_tags);
                    PerfStatsCollector::addCounter(
                        "moe_rebalance",
                        "runtime_phase_local_compute_slots",
                        static_cast<double>(phase_local_slots),
                        "rebalance",
                        device_id_.toString(),
                        phase_tags);
                }
            }

            if (PerfStatsCollector::isDomainEnabled("moe_rebalance"))
            {
                const auto &state = host_layers_[static_cast<size_t>(layer_idx)];
                const PerfStatsCollector::Tags tags{
                    {"layer", std::to_string(layer_idx)},
                    {"participant", std::to_string(state.participant_id)},
                    {"participants", std::to_string(state.participant_count)},
                    {"active_epoch", std::to_string(state.active_epoch)},
                    {"reset", reset_runtime_counts ? "true" : "false"}};
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "runtime_selected_slots",
                    static_cast<double>(selected_slots),
                    "rebalance",
                    device_id_.toString(),
                    tags);
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "runtime_local_compute_slots",
                    static_cast<double>(local_slots),
                    "rebalance",
                    device_id_.toString(),
                    tags);
            }
        }

        if (!reset_runtime_counts)
            return true;

        for (auto &state : host_layers_)
        {
            resetRuntimeHistogramFields(state, num_experts_);
            resetRouterHotCacheCounters(state);
        }

        if (mirror_to_device_)
        {
            const size_t counters_offset = offsetof(DeviceMoELayerRuntime, router_hot_cache_eligible_dispatches);
            const size_t counters_bytes =
                offsetof(DeviceMoELayerRuntime, route_expert_ids) - counters_offset;
            const size_t histogram_bytes =
                offsetof(DeviceMoELayerRuntime,
                         router_hot_cache_eligible_dispatches) -
                offsetof(DeviceMoELayerRuntime, decode_histogram);
            for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            {
                auto *dst = device_layers_[layer_idx].decode_histogram;
                memsetMirror(device_id_, dst, 0,
                             histogram_bytes,
                             stream,
                             layerPrefix(layer_idx) + "all phase histograms reset");
                auto *counter_dst =
                    reinterpret_cast<std::byte *>(device_layers_ + layer_idx) + counters_offset;
                memsetMirror(device_id_, counter_dst, 0,
                             counters_bytes,
                             stream,
                             layerPrefix(layer_idx) + "router hot-cache counter reset");
            }
            synchronizeMirror(device_id_, stream, "[MoERuntimeTable] decode histogram reset sync");
        }

        return true;
    }

    bool DeviceMoERuntimeTable::captureDecodeHistogramCounts(
        std::vector<uint64_t> &selected_counts,
        std::vector<uint64_t> &local_counts,
        void *stream)
    {
        const size_t entry_count =
            static_cast<size_t>(num_layers_) * static_cast<size_t>(num_experts_);
        selected_counts.assign(entry_count, 0ULL);
        local_counts.assign(entry_count, 0ULL);

        if (!mirror_to_device_)
        {
            for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            {
                const auto &state = host_layers_[static_cast<size_t>(layer_idx)];
                auto *selected_dst =
                    selected_counts.data() + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
                auto *local_dst =
                    local_counts.data() + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
                std::copy(state.decode_histogram,
                          state.decode_histogram + num_experts_,
                          selected_dst);
                std::copy(state.decode_local_histogram,
                          state.decode_local_histogram + num_experts_,
                          local_dst);
            }
            return true;
        }

        if (!stream)
            throw std::invalid_argument(
                "[MoERuntimeTable] mirrored decode histogram capture requires an explicit stream");

        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            const auto *selected_src = device_layers_[layer_idx].decode_histogram;
            auto *selected_dst =
                selected_counts.data() + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
            copyMirrorToHost(device_id_, selected_dst, selected_src,
                             static_cast<size_t>(num_experts_) * sizeof(uint64_t),
                             stream,
                             layerPrefix(layer_idx) + "decode histogram capture");

            const auto *local_src = device_layers_[layer_idx].decode_local_histogram;
            auto *local_dst =
                local_counts.data() + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
            copyMirrorToHost(device_id_, local_dst, local_src,
                             static_cast<size_t>(num_experts_) * sizeof(uint64_t),
                             stream,
                             layerPrefix(layer_idx) + "decode local histogram capture");
        }
        synchronizeMirror(device_id_, stream, "[MoERuntimeTable] decode histogram capture sync");
        return true;
    }

    bool DeviceMoERuntimeTable::restoreDecodeHistogramCounts(
        const uint64_t *selected_counts,
        const uint64_t *local_counts,
        size_t layer_count,
        size_t expert_count,
        void *stream)
    {
        if (!selected_counts || !local_counts)
            return false;
        if (layer_count != static_cast<size_t>(num_layers_) ||
            expert_count != static_cast<size_t>(num_experts_))
        {
            LOG_ERROR("[MoERuntimeTable] decode histogram restore shape mismatch: table layers="
                      << num_layers_ << " experts=" << num_experts_
                      << " blob layers=" << layer_count
                      << " experts=" << expert_count);
            return false;
        }

        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            auto &state = host_layers_[static_cast<size_t>(layer_idx)];
            resetRuntimeHistogramFields(state, num_experts_);
            const auto *selected_src =
                selected_counts + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
            const auto *local_src =
                local_counts + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
            std::copy(selected_src, selected_src + num_experts_, state.decode_histogram);
            std::copy(local_src, local_src + num_experts_, state.decode_local_histogram);
            resetRouterHotCacheCounters(state);
        }

        if (!mirror_to_device_)
            return true;

        if (!stream)
            throw std::invalid_argument(
                "[MoERuntimeTable] mirrored decode histogram restore requires an explicit stream");

        const size_t counters_offset = offsetof(DeviceMoELayerRuntime, router_hot_cache_eligible_dispatches);
        const size_t counters_bytes =
            offsetof(DeviceMoELayerRuntime, route_expert_ids) - counters_offset;
        const size_t histogram_bytes =
            offsetof(DeviceMoELayerRuntime,
                     router_hot_cache_eligible_dispatches) -
            offsetof(DeviceMoELayerRuntime, decode_histogram);
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            memsetMirror(
                device_id_,
                device_layers_[layer_idx].decode_histogram,
                0,
                histogram_bytes,
                stream,
                layerPrefix(layer_idx) +
                    "phase histogram restore reset");
            const auto *selected_src =
                selected_counts + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
            auto *selected_dst = device_layers_[layer_idx].decode_histogram;
            copyHostToMirror(device_id_, selected_dst, selected_src,
                             static_cast<size_t>(num_experts_) * sizeof(uint64_t),
                             stream,
                             layerPrefix(layer_idx) + "decode histogram restore");

            const auto *local_src =
                local_counts + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
            auto *local_dst = device_layers_[layer_idx].decode_local_histogram;
            copyHostToMirror(device_id_, local_dst, local_src,
                             static_cast<size_t>(num_experts_) * sizeof(uint64_t),
                             stream,
                             layerPrefix(layer_idx) + "decode local histogram restore");

            auto *counter_dst =
                reinterpret_cast<std::byte *>(device_layers_ + layer_idx) + counters_offset;
            memsetMirror(device_id_, counter_dst, 0,
                         counters_bytes,
                         stream,
                         layerPrefix(layer_idx) + "router hot-cache counter restore reset");
        }
        return true;
    }

    void DeviceMoERuntimeTable::resetDecodeHistogramCounts(void *stream)
    {
        for (auto &state : host_layers_)
        {
            resetRuntimeHistogramFields(state, num_experts_);
            resetRouterHotCacheCounters(state);
        }

        if (!mirror_to_device_)
            return;

        const size_t counters_offset = offsetof(DeviceMoELayerRuntime, router_hot_cache_eligible_dispatches);
        const size_t counters_bytes =
            offsetof(DeviceMoELayerRuntime, route_expert_ids) - counters_offset;
        const size_t histogram_bytes =
            offsetof(DeviceMoELayerRuntime,
                     router_hot_cache_eligible_dispatches) -
            offsetof(DeviceMoELayerRuntime, decode_histogram);
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            auto *dst = device_layers_[layer_idx].decode_histogram;
            memsetMirror(device_id_, dst, 0,
                         histogram_bytes,
                         stream,
                         layerPrefix(layer_idx) + "all phase histograms reset");
            auto *counter_dst =
                reinterpret_cast<std::byte *>(device_layers_ + layer_idx) + counters_offset;
            memsetMirror(device_id_, counter_dst, 0,
                         counters_bytes,
                         stream,
                         layerPrefix(layer_idx) + "router hot-cache counter reset");
        }
        synchronizeMirror(device_id_, stream, "[MoERuntimeTable] decode histogram reset sync");
    }

    void DeviceMoERuntimeTable::resetDecodeRuntimeState(void *stream)
    {
        std::fill(decode_runtime_publication_required_.begin(),
                  decode_runtime_publication_required_.end(),
                  uint8_t{1});

        if (mirror_to_device_)
        {
            if (!stream)
            {
                throw std::invalid_argument(
                    "[MoERuntimeTable] GPU decode runtime reset requires an explicit stream");
            }
            /*
             * Do not manufacture a "coherent" host mirror here.  The reset
             * transaction is device-owned and the immutable device baseline is
             * its sole source of truth.  Any later diagnostic that genuinely
             * needs a host snapshot must request an explicit, ordered readback
             * instead of accidentally consuming this setup-time vector.
             */
            copyMirrorToMirrorAsync(
                device_id_,
                device_layers_,
                device_empty_layers_,
                host_layers_.size() * sizeof(DeviceMoELayerRuntime),
                stream,
                "[MoERuntimeTable] device-owned empty runtime reset");
            return;
        }

        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            auto &state = host_layers_[static_cast<size_t>(layer_idx)];
            /*
             * Scratch bindings are model-lifetime graph identities, not
             * request state.  Capture them from the authoritative live table
             * before clearing placement.  Looking them up only through the
             * optional allocation registry lost bindings for runtime tables
             * whose scratch was attached before a prefix/MTP graph boundary;
             * the stage-local "warmed" bit then remained true while the device
             * table advertised a zero route capacity.
             */
            const RuntimePersistentBindings scratch =
                captureRuntimePersistentBindings(state);

            resetLayer(state);
            restoreRuntimePersistentBindings(state, scratch);
        }
    }

    bool DeviceMoERuntimeTable::hasInitialRuntimeState() const noexcept
    {
        return std::any_of(initial_layer_captured_.begin(),
                           initial_layer_captured_.end(),
                           [](uint8_t captured)
                           { return captured != 0u; });
    }

    bool DeviceMoERuntimeTable::hasCompleteInitialRuntimeState() const noexcept
    {
        if (initial_layer_captured_.size() != host_layers_.size() ||
            decode_runtime_publication_required_.size() !=
                host_layers_.size() ||
            host_layers_.empty())
        {
            return false;
        }
        for (std::size_t layer = 0u; layer < host_layers_.size(); ++layer)
        {
            const auto &runtime = host_layers_[layer];
            if (initial_layer_captured_[layer] == 0u ||
                decode_runtime_publication_required_[layer] != 0u ||
                runtime.active_bank > 1u || runtime.active_epoch == 0u ||
                runtime.expert_count !=
                    static_cast<std::uint32_t>(num_experts_) ||
                runtime.top_k != static_cast<std::uint32_t>(top_k_))
            {
                return false;
            }
            const auto &bank = runtime.banks[runtime.active_bank];
            if (bank.epoch != runtime.active_epoch ||
                bank.expert_count !=
                    static_cast<std::uint32_t>(num_experts_))
            {
                return false;
            }
        }
        return true;
    }

    void DeviceMoERuntimeTable::sealAndPublishCompleteInitialRuntimeState(
        void *stream)
    {
        if (!mirror_to_device_ || !device_id_.is_gpu() || !stream)
        {
            throw std::invalid_argument(
                "[MoERuntimeTable] complete initial runtime publication "
                "requires a mirrored GPU table and an explicit stream");
        }
        if (initial_runtime_family_lifecycle_ !=
            InitialRuntimeFamilyLifecycle::Collecting)
        {
            throw std::logic_error(
                "[MoERuntimeTable] complete initial runtime family was "
                "already published");
        }
        if (!hasCompleteInitialRuntimeState())
        {
            throw std::logic_error(
                "[MoERuntimeTable] cannot publish an incomplete retained "
                "runtime family");
        }

        /*
         * The first valid bank is only a construction checkpoint. Later graph
         * families may finish persistent scratch binding or replace that bank
         * with a newer setup-only recipe before controller ownership begins.
         * Freezing the first checkpoint made reset publish an older generation
         * than the live recipe that was validated above. Close Collecting by
         * snapshotting the exact final family now; after this transition the
         * device controller, rather than either host vector, owns placement.
         */
        for (std::size_t layer = 0u; layer < host_layers_.size(); ++layer)
        {
            initial_host_layers_[layer] = host_layers_[layer];
            resetPerRequestRuntimeFields(
                initial_host_layers_[layer], num_experts_);
            initial_layer_captured_[layer] = 1u;
        }

        const size_t bytes =
            initial_host_layers_.size() * sizeof(DeviceMoELayerRuntime);
        /*
         * Layer setup can occur while a future graph is being materialized.
         * A valid host recipe therefore does not prove that an unlaunched
         * retained graph has made the recipe device-visible. Publish the
         * complete immutable template once outside capture, then use a D2D
         * edge to install exactly those bytes as the live starting state.
         * This setup handoff is the last host-authored placement publication;
         * Dynamic epochs that follow remain device-owned.
         */
        copyHostToMirror(
            device_id_,
            device_initial_layers_,
            initial_host_layers_.data(),
            bytes,
            stream,
            "[MoERuntimeTable] sealed initial runtime template publication");
        copyMirrorToMirrorAsync(
            device_id_,
            device_layers_,
            device_initial_layers_,
            bytes,
            stream,
            "[MoERuntimeTable] sealed live runtime family publication");
        initial_runtime_family_lifecycle_ =
            InitialRuntimeFamilyLifecycle::Published;
    }

    void DeviceMoERuntimeTable::restoreInitialRuntimeState(void *stream)
    {
        if (mirror_to_device_)
        {
            if (!stream)
            {
                throw std::invalid_argument(
                    "[MoERuntimeTable] GPU initial runtime restore requires an explicit stream");
            }
            /*
             * Keep host setup templates out of the request lifecycle.  They
             * populated device_initial_layers_ during model construction, but
             * they are not a live coherence peer once GPU execution begins.
             */
            copyMirrorToMirrorAsync(
                device_id_,
                device_layers_,
                device_initial_layers_,
                host_layers_.size() * sizeof(DeviceMoELayerRuntime),
                stream,
                "[MoERuntimeTable] device-owned initial runtime restore");
            for (size_t layer_idx = 0; layer_idx < initial_layer_captured_.size(); ++layer_idx)
            {
                decode_runtime_publication_required_[layer_idx] =
                    initial_layer_captured_[layer_idx] == 0u ? 1u : 0u;
            }
            return;
        }

        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            auto &state = host_layers_[static_cast<size_t>(layer_idx)];
            const auto scratch = captureRuntimePersistentBindings(state);
            state = initial_layer_captured_[static_cast<size_t>(layer_idx)] != 0u
                        ? initial_host_layers_[static_cast<size_t>(layer_idx)]
                        : empty_host_layers_[static_cast<size_t>(layer_idx)];
            restoreRuntimePersistentBindings(state, scratch);
            resetPerRequestRuntimeFields(state, num_experts_);
            decode_runtime_publication_required_[static_cast<size_t>(layer_idx)] =
                initial_layer_captured_[static_cast<size_t>(layer_idx)] == 0u ? 1u : 0u;
        }
    }

    void DeviceMoERuntimeTable::syncRuntimeStateToHost(void *stream)
    {
        if (!mirror_to_device_)
            return;

        void *owned_stream = nullptr;
        void *active_stream = stream;
        if (!active_stream)
        {
            owned_stream = createMirrorStream(device_id_,
                                              "[MoERuntimeTable] runtime state D2H stream");
            active_stream = owned_stream;
        }

        try
        {
            for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            {
                copyMirrorToHost(device_id_,
                                 host_layers_.data() + layer_idx,
                                 device_layers_ + layer_idx,
                                 sizeof(DeviceMoELayerRuntime),
                                 active_stream,
                                 layerPrefix(layer_idx) + "runtime table D2H");
            }
            synchronizeMirror(device_id_, active_stream,
                              "[MoERuntimeTable] runtime state D2H sync");

            if (owned_stream)
            {
                destroyMirrorStream(device_id_, owned_stream,
                                    "[MoERuntimeTable] runtime state D2H stream destroy");
                owned_stream = nullptr;
            }
        }
        catch (...)
        {
            if (owned_stream)
                destroyMirrorStream(device_id_, owned_stream,
                                    "[MoERuntimeTable] runtime state D2H stream destroy");
            throw;
        }
    }

    void DeviceMoERuntimeTable::restoreRuntimeStateSnapshot(
        const DeviceMoELayerRuntime *layers,
        size_t layer_count,
        void *stream)
    {
        if (!layers)
            throw std::invalid_argument("[MoERuntimeTable] runtime snapshot restore requires layer data");
        if (layer_count != static_cast<size_t>(num_layers_))
            throw std::invalid_argument("[MoERuntimeTable] runtime snapshot layer count mismatch");

        void *owned_stream = nullptr;
        void *active_stream = stream;
        if (mirror_to_device_ && !active_stream)
        {
            owned_stream = createMirrorStream(device_id_,
                                              "[MoERuntimeTable] runtime snapshot restore stream");
            active_stream = owned_stream;
        }

        try
        {
            for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            {
                auto &state = host_layers_[static_cast<size_t>(layer_idx)];
                const auto scratch = captureRuntimePersistentBindings(state);
                const auto &snapshot = layers[static_cast<size_t>(layer_idx)];
                if (snapshot.expert_count != static_cast<uint32_t>(num_experts_) ||
                    snapshot.top_k != static_cast<uint32_t>(top_k_) ||
                    snapshot.active_bank > 1u)
                {
                    throw std::invalid_argument(
                        layerPrefix(layer_idx) + "runtime snapshot metadata mismatch");
                }

                if (snapshot.active_epoch == 0u)
                {
                    resetPerRequestRuntimeFields(state, num_experts_);
                    if (mirror_to_device_)
                        uploadLayerState(layer_idx, active_stream);
                    continue;
                }

                state = snapshot;
                restoreRuntimePersistentBindings(state, scratch);
                resetPerRequestRuntimeFields(state, num_experts_);
                if (mirror_to_device_)
                    uploadLayerState(layer_idx, active_stream);
                decode_runtime_publication_required_[static_cast<size_t>(layer_idx)] = 0u;
            }

            if (mirror_to_device_)
                synchronizeMirror(device_id_, active_stream,
                                  "[MoERuntimeTable] runtime snapshot restore sync");

            if (owned_stream)
            {
                destroyMirrorStream(device_id_, owned_stream,
                                    "[MoERuntimeTable] runtime snapshot restore stream destroy");
                owned_stream = nullptr;
            }
        }
        catch (...)
        {
            if (owned_stream)
                destroyMirrorStream(device_id_, owned_stream,
                                    "[MoERuntimeTable] runtime snapshot restore stream destroy");
            throw;
        }
    }

    bool DeviceMoERuntimeTable::capturePortableRuntimeState(
        std::vector<DeviceMoEPortableLayerRuntimeState> &layers,
        void *stream)
    {
        layers.clear();
        const DeviceMoELayerRuntime *source_layers = host_layers_.data();
        std::vector<DeviceMoELayerRuntime> mirrored_layers;

        if (mirror_to_device_)
        {
            if (!stream)
                throw std::invalid_argument(
                    "[MoERuntimeTable] mirrored portable runtime capture requires an explicit stream");
            mirrored_layers.resize(static_cast<size_t>(num_layers_));
            copyMirrorToHost(device_id_,
                             mirrored_layers.data(),
                             device_layers_,
                             sizeof(DeviceMoELayerRuntime) *
                                 static_cast<size_t>(num_layers_),
                             stream,
                             "[MoERuntimeTable] portable runtime state capture");
            synchronizeMirror(device_id_, stream,
                              "[MoERuntimeTable] portable runtime state capture sync");
            source_layers = mirrored_layers.data();
        }

        layers.reserve(static_cast<size_t>(num_layers_));
        uint32_t rehydratable_placement_layers = 0;
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            const auto &state = source_layers[static_cast<size_t>(layer_idx)];
            if (state.expert_count != static_cast<uint32_t>(num_experts_) ||
                state.top_k != static_cast<uint32_t>(top_k_) ||
                state.active_bank > 1u)
            {
                LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                     << ": cannot capture malformed portable runtime state"
                                                     << " active_bank=" << state.active_bank
                                                     << " experts=" << state.expert_count
                                                     << " top_k=" << state.top_k);
                layers.clear();
                return false;
            }

            /*
             * Preserve exact logical placement while excluding every
             * pointer-bearing transfer-slot descriptor.
             *
             * A resident mask is domain-wide state: both the immutable owner
             * and the replica destination observe the same bits. Only the
             * destination owns the rolling slot descriptor. Canonicalizing
             * transient layers to immutable placement made a cache hit resume
             * with a different row-to-participant assignment than an uncached
             * split prefill. That changes FP32 reduction association and breaks
             * byte equality even when every route is present.
             *
             * The portable payload now records exact masks and local roles, but
             * portableMoEExpertFlags() still strips TransferSlot. Restore
             * reconstructs those bytes from immutable owner weights in a
             * dedicated captured collective before routing resumes.
             */
            const auto &live_bank = state.banks[state.active_bank];
            const bool local_transient_payload =
                deviceMoELayerUsesTransientLocalPayload(state);
            const bool domain_has_transient_placement =
                live_bank.transient_placement_observed != 0u ||
                local_transient_payload;
            const size_t idx = static_cast<size_t>(layer_idx);
            const DeviceMoELayerRuntime *initial_state = nullptr;
            if (domain_has_transient_placement)
            {
                if (idx >= initial_layer_captured_.size() ||
                    initial_layer_captured_[idx] == 0u)
                {
                    LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                         << ": transient portable capture has no immutable initial placement");
                    layers.clear();
                    return false;
                }
                initial_state = &initial_host_layers_[idx];
                if (initial_state->expert_count != static_cast<uint32_t>(num_experts_) ||
                    initial_state->top_k != static_cast<uint32_t>(top_k_) ||
                    initial_state->active_bank > 1u ||
                    deviceMoELayerUsesTransientLocalPayload(*initial_state))
                {
                    LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                         << ": immutable initial placement is invalid or contains transient payload");
                    layers.clear();
                    return false;
                }
                ++rehydratable_placement_layers;
            }

            const auto &bank = live_bank;
            DeviceMoEPortableLayerRuntimeState captured;
            captured.active_epoch = state.active_epoch;
            captured.expert_count = state.expert_count;
            captured.top_k = state.top_k;
            captured.participant_id = state.participant_id;
            captured.participant_count = state.participant_count;
            captured.experts.resize(static_cast<size_t>(num_experts_));
            captured.selected_histogram.assign(
                state.decode_histogram,
                state.decode_histogram + num_experts_);
            captured.local_histogram.assign(
                state.decode_local_histogram,
                state.decode_local_histogram + num_experts_);
            captured.prefill_selected_histogram.assign(
                state.prefill_histogram,
                state.prefill_histogram + num_experts_);
            captured.prefill_local_histogram.assign(
                state.prefill_local_histogram,
                state.prefill_local_histogram + num_experts_);
            captured.grouped_verifier_selected_histogram.assign(
                state.grouped_verifier_histogram,
                state.grouped_verifier_histogram + num_experts_);
            captured.grouped_verifier_local_histogram.assign(
                state.grouped_verifier_local_histogram,
                state.grouped_verifier_local_histogram + num_experts_);

            for (int expert = 0; expert < num_experts_; ++expert)
            {
                const auto &desc = bank.experts[static_cast<size_t>(expert)];

                auto &dst = captured.experts[static_cast<size_t>(expert)];
                dst.logical_expert_id =
                    desc.logical_expert_id >= 0 ? desc.logical_expert_id : expert;
                dst.owner_participant = desc.owner_participant;
                dst.local_slot = hasMoEExpertFlag(
                                     desc.flags,
                                     DeviceMoEExpertFlags::TransferSlot)
                                     ? -1
                                     : desc.local_slot;
                dst.flags = portableMoEExpertFlags(desc.flags);
                dst.local_compute = bank.local_compute_mask[static_cast<size_t>(expert)] != 0u ? 1u : 0u;
                dst.replica_role = bank.replica_role[static_cast<size_t>(expert)];
                dst.resident_participant_mask =
                    bank.resident_participant_mask[static_cast<size_t>(expert)];
                if (initial_state)
                {
                    const auto &initial_bank =
                        initial_state->banks[initial_state->active_bank];
                    const auto &initial_desc =
                        initial_bank.experts[static_cast<size_t>(expert)];
                    if (dst.owner_participant != initial_desc.owner_participant ||
                        dst.resident_participant_mask !=
                            initial_bank.resident_participant_mask[
                                static_cast<size_t>(expert)])
                    {
                        captured.requires_device_payload_rehydration = 1u;
                    }
                }
            }
            layers.push_back(std::move(captured));
        }

        if (rehydratable_placement_layers != 0u)
        {
            PerfStatsCollector::addCounter(
                "prefix_cache",
                "moe_portable_device_rehydration_layers",
                static_cast<double>(rehydratable_placement_layers),
                "prefix_cache",
                device_id_.toString());
        }
        return true;
    }

    DeviceMoEPortableRuntimeRestoreResult
    DeviceMoERuntimeTable::restorePortableRuntimeState(
        const std::vector<DeviceMoEPortableLayerRuntimeState> &layers,
        void *stream,
        const LocalPayloadDescriptorResolver &local_payload_resolver)
    {
        if (layers.size() != static_cast<size_t>(num_layers_))
        {
            LOG_ERROR("[MoERuntimeTable] portable runtime restore layer count mismatch: table="
                      << num_layers_ << " snapshot=" << layers.size());
            return {};
        }
        if (mirror_to_device_ && !stream)
            throw std::invalid_argument(
                "[MoERuntimeTable] mirrored portable runtime restore requires an explicit stream");

        bool placement_changed = false;
        bool requires_device_payload_rehydration = false;
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            const auto &snapshot = layers[static_cast<size_t>(layer_idx)];
            if (snapshot.expert_count != static_cast<uint32_t>(num_experts_) ||
                snapshot.top_k != static_cast<uint32_t>(top_k_) ||
                snapshot.experts.size() != static_cast<size_t>(num_experts_) ||
                snapshot.selected_histogram.size() != static_cast<size_t>(num_experts_) ||
                snapshot.local_histogram.size() != static_cast<size_t>(num_experts_) ||
                snapshot.prefill_selected_histogram.size() !=
                    static_cast<size_t>(num_experts_) ||
                snapshot.prefill_local_histogram.size() !=
                    static_cast<size_t>(num_experts_) ||
                snapshot.grouped_verifier_selected_histogram.size() !=
                    static_cast<size_t>(num_experts_) ||
                snapshot.grouped_verifier_local_histogram.size() !=
                    static_cast<size_t>(num_experts_))
            {
                LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                     << ": portable runtime restore metadata mismatch");
                return {};
            }

            auto &state = host_layers_[static_cast<size_t>(layer_idx)];
            const size_t layer_offset = static_cast<size_t>(layer_idx);
            const bool has_initial_baseline =
                layer_offset < initial_layer_captured_.size() &&
                initial_layer_captured_[layer_offset] != 0u;
            const DeviceMoELayerRuntime &placement_baseline =
                has_initial_baseline
                    ? initial_host_layers_[layer_offset]
                    : state;
            placement_changed =
                placement_changed ||
                !portablePlacementMatchesRuntime(snapshot, placement_baseline);
            requires_device_payload_rehydration =
                requires_device_payload_rehydration ||
                snapshot.requires_device_payload_rehydration != 0u;
            if (snapshot.requires_device_payload_rehydration != 0u)
            {
                /*
                 * Recreate transient placement from immutable owner payloads.
                 *
                 * The cache blob deliberately contains no device pointers. We
                 * therefore restore the model-lifetime bank now and preload the
                 * logical owner-to-replica edges into the persistent LLEP transfer
                 * array. A one-shot captured forward graph consumes this array
                 * before it assigns any suffix routes. The H2D below is part of
                 * the explicit RAM/disk prefix import boundary; expert payload
                 * bytes themselves never leave the GPU domain.
                 */
                const size_t idx = static_cast<size_t>(layer_idx);
                if (!mirror_to_device_ ||
                    !device_id_.is_gpu() ||
                    idx >= initial_layer_captured_.size() ||
                    initial_layer_captured_[idx] == 0u)
                {
                    LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                         << ": transient portable restore requires a mirrored GPU table with immutable placement");
                    return {};
                }

                const auto scratch = captureRuntimePersistentBindings(state);
                if (!scratch.reserved_ptrs[2] ||
                    scratch.reserved_u64[1] == 0u)
                {
                    LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                         << ": transient portable restore has no persistent LLEP transfer scratch");
                    return {};
                }

                const auto &initial = initial_host_layers_[idx];
                if (initial.active_bank > 1u ||
                    initial.expert_count != static_cast<uint32_t>(num_experts_) ||
                    initial.top_k != static_cast<uint32_t>(top_k_) ||
                    initial.participant_id != snapshot.participant_id ||
                    initial.participant_count != snapshot.participant_count ||
                    initial.participant_count == 0u ||
                    initial.participant_count > kDeviceMoEMaxParticipants)
                {
                    LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                         << ": transient portable restore immutable placement metadata mismatch");
                    return {};
                }

                const auto &initial_bank = initial.banks[initial.active_bank];
                const uint32_t valid_participants =
                    participantMaskLimit(snapshot.participant_count);
                const uint32_t local_bit =
                    participantBit(snapshot.participant_id);
                std::vector<least_loaded_ep::LeastLoadedExpertWeightTransfer>
                    transfers;
                transfers.reserve(static_cast<size_t>(num_experts_));

                for (int expert = 0; expert < num_experts_; ++expert)
                {
                    const auto &saved =
                        snapshot.experts[static_cast<size_t>(expert)];
                    const auto &initial_desc =
                        initial_bank.experts[static_cast<size_t>(expert)];
                    const uint32_t initial_mask =
                        initial_bank.resident_participant_mask[
                            static_cast<size_t>(expert)];
                    const uint32_t desired_mask =
                        saved.resident_participant_mask;

                    if (saved.logical_expert_id != expert ||
                        saved.owner_participant !=
                            initial_desc.owner_participant ||
                        initial_desc.owner_participant < 0 ||
                        initial_desc.owner_participant >=
                            static_cast<int32_t>(snapshot.participant_count) ||
                        (desired_mask & ~valid_participants) != 0u ||
                        (desired_mask & initial_mask) != initial_mask ||
                        (desired_mask &
                         participantBit(static_cast<uint32_t>(
                             initial_desc.owner_participant))) == 0u)
                    {
                        LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                             << ": transient portable restore cannot derive an immutable-owner replica plan for expert "
                                                             << expert);
                        return {};
                    }

                    const bool expected_local_compute =
                        (desired_mask & local_bit) != 0u;
                    if ((saved.local_compute != 0u) !=
                        expected_local_compute)
                    {
                        LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                             << ": transient portable restore local-compute/residency mismatch for expert "
                                                             << expert);
                        return {};
                    }
                    uint32_t arrivals = desired_mask & ~initial_mask;
                    while (arrivals != 0u)
                    {
                        const uint32_t destination =
                            static_cast<uint32_t>(
                                std::countr_zero(arrivals));
                        arrivals &= arrivals - 1u;
                        transfers.push_back(
                            least_loaded_ep::LeastLoadedExpertWeightTransfer{
                                .expert = static_cast<uint32_t>(expert),
                                .source_participant =
                                    static_cast<uint32_t>(
                                        initial_desc.owner_participant),
                                .destination_participant = destination});
                    }
                }

                if (transfers.empty() ||
                    transfers.size() >
                        static_cast<size_t>(scratch.reserved_u64[1]))
                {
                    LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                         << ": transient portable restore produced invalid transfer count "
                                                         << transfers.size()
                                                         << " capacity="
                                                         << scratch.reserved_u64[1]);
                    return {};
                }

                state = initial;
                restoreRuntimePersistentBindings(state, scratch);
                std::copy(snapshot.selected_histogram.begin(),
                          snapshot.selected_histogram.end(),
                          state.decode_histogram);
                std::copy(snapshot.local_histogram.begin(),
                          snapshot.local_histogram.end(),
                          state.decode_local_histogram);
                std::copy(snapshot.prefill_selected_histogram.begin(),
                          snapshot.prefill_selected_histogram.end(),
                          state.prefill_histogram);
                std::copy(snapshot.prefill_local_histogram.begin(),
                          snapshot.prefill_local_histogram.end(),
                          state.prefill_local_histogram);
                std::copy(
                    snapshot.grouped_verifier_selected_histogram.begin(),
                    snapshot.grouped_verifier_selected_histogram.end(),
                    state.grouped_verifier_histogram);
                std::copy(
                    snapshot.grouped_verifier_local_histogram.begin(),
                    snapshot.grouped_verifier_local_histogram.end(),
                    state.grouped_verifier_local_histogram);
                resetRouterHotCacheCounters(state);
                state.reserved_u64[2] = 0u;
                state.reserved_u64[3] =
                    static_cast<uint64_t>(transfers.size());

                copyHostToMirror(
                    device_id_,
                    state.reserved_ptrs[2],
                    transfers.data(),
                    transfers.size() *
                        sizeof(least_loaded_ep::LeastLoadedExpertWeightTransfer),
                    stream,
                    layerPrefix(layer_idx) +
                        "portable restore transfer-plan upload");
                uploadLayerState(layer_idx, stream);
                decode_runtime_publication_required_[idx] = 0u;
                continue;
            }

            MoEPlacementUpdate update;
            update.epoch = std::max<uint32_t>(
                state.active_epoch + 1u,
                snapshot.active_epoch == 0u ? 1u : snapshot.active_epoch);
            if (update.epoch <= state.active_epoch)
                update.epoch = state.active_epoch + 1u;
            update.expert_count = static_cast<uint32_t>(num_experts_);
            update.participant_id = snapshot.participant_id;
            update.participant_count = snapshot.participant_count;
            update.experts.resize(static_cast<size_t>(num_experts_));
            update.local_compute_mask.assign(static_cast<size_t>(num_experts_), 0u);
            update.replica_role.assign(
                static_cast<size_t>(num_experts_),
                static_cast<uint8_t>(DeviceMoEReplicaRole::None));
            update.resident_participant_mask.assign(static_cast<size_t>(num_experts_), 0u);

            for (int expert = 0; expert < num_experts_; ++expert)
            {
                const auto &saved = snapshot.experts[static_cast<size_t>(expert)];
                if (saved.logical_expert_id != static_cast<int32_t>(expert))
                {
                    LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                         << ": portable runtime restore logical expert mismatch"
                                                         << " slot=" << expert
                                                         << " logical=" << saved.logical_expert_id);
                    return {};
                }

                DeviceMoEExpertDescriptor desc;
                desc.logical_expert_id = expert;
                desc.owner_participant = saved.owner_participant;
                desc.local_slot = saved.local_compute ? saved.local_slot : -1;
                desc.flags = clearPayloadBearingMoEExpertFlags(
                    portableMoEExpertFlags(saved.flags));

                if (saved.local_compute != 0u)
                {
                    DeviceMoEExpertDescriptor ready_desc;
                    bool has_ready_descriptor = false;
                    const bool resolver_is_authoritative =
                        portableLocalReplicaRequiresResolver(snapshot,
                                                             saved,
                                                             local_payload_resolver);

                    if (resolver_is_authoritative)
                    {
                        has_ready_descriptor =
                            local_payload_resolver(layer_idx,
                                                   expert,
                                                   saved.local_slot,
                                                   ready_desc);
                        if (!has_ready_descriptor)
                        {
                            LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                                 << ": portable runtime restore requires graph-owned local payload for remote-owned expert "
                                                                 << expert
                                                                 << " slot=" << saved.local_slot
                                                                 << " but the resolver could not bind it");
                            return {};
                        }
                        if (!descriptorMatchesPortableLocalClaim(ready_desc,
                                                                 expert,
                                                                 saved.local_slot))
                        {
                            LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                                 << ": portable runtime restore resolver returned an invalid descriptor for expert "
                                                                 << expert);
                            return {};
                        }
                    }
                    else
                    {
                        has_ready_descriptor =
                            findReadyDescriptorForExpert(state,
                                                         static_cast<uint32_t>(expert),
                                                         saved.local_slot,
                                                         ready_desc);
                        if (!has_ready_descriptor && local_payload_resolver)
                        {
                            has_ready_descriptor =
                                local_payload_resolver(layer_idx,
                                                       expert,
                                                       saved.local_slot,
                                                       ready_desc);
                            if (has_ready_descriptor &&
                                !descriptorMatchesPortableLocalClaim(ready_desc,
                                                                     expert,
                                                                     saved.local_slot))
                            {
                                LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                                     << ": portable runtime restore resolver returned an invalid descriptor for expert "
                                                                     << expert);
                                return {};
                            }
                        }
                    }

                    if (!has_ready_descriptor)
                    {
                        LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                             << ": portable runtime restore requires local payload for expert "
                                                             << expert
                                                             << " but no live descriptor is resident");
                        return {};
                    }
                    desc = ready_desc;
                    desc.logical_expert_id = expert;
                    desc.owner_participant = saved.owner_participant;
                    if (saved.local_slot >= 0)
                        desc.local_slot = saved.local_slot;
                    desc.flags = portableMoEExpertFlags(saved.flags) |
                                 toMoEExpertFlags(DeviceMoEExpertFlags::Valid) |
                                 toMoEExpertFlags(DeviceMoEExpertFlags::Resident) |
                                 toMoEExpertFlags(DeviceMoEExpertFlags::LocalCompute);
                    if (hasMoEExpertFlag(ready_desc.flags,
                                         DeviceMoEExpertFlags::TransferSlot))
                    {
                        desc.flags |= toMoEExpertFlags(DeviceMoEExpertFlags::TransferSlot);
                    }
                }

                update.experts[static_cast<size_t>(expert)] = desc;
                update.local_compute_mask[static_cast<size_t>(expert)] =
                    saved.local_compute != 0u ? 1u : 0u;
                update.replica_role[static_cast<size_t>(expert)] = saved.replica_role;
                update.resident_participant_mask[static_cast<size_t>(expert)] =
                    saved.resident_participant_mask;
            }

            try
            {
                prepareInactiveBank(layer_idx, update);
                flipActiveBank(layer_idx, update.epoch, stream);
            }
            catch (const std::exception &ex)
            {
                LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                     << ": portable runtime restore failed: "
                                                     << ex.what());
                return {};
            }

            auto &restored = host_layers_[static_cast<size_t>(layer_idx)];
            std::copy(snapshot.selected_histogram.begin(),
                      snapshot.selected_histogram.end(),
                      restored.decode_histogram);
            std::copy(snapshot.local_histogram.begin(),
                      snapshot.local_histogram.end(),
                      restored.decode_local_histogram);
            std::copy(snapshot.prefill_selected_histogram.begin(),
                      snapshot.prefill_selected_histogram.end(),
                      restored.prefill_histogram);
            std::copy(snapshot.prefill_local_histogram.begin(),
                      snapshot.prefill_local_histogram.end(),
                      restored.prefill_local_histogram);
            std::copy(
                snapshot.grouped_verifier_selected_histogram.begin(),
                snapshot.grouped_verifier_selected_histogram.end(),
                restored.grouped_verifier_histogram);
            std::copy(
                snapshot.grouped_verifier_local_histogram.begin(),
                snapshot.grouped_verifier_local_histogram.end(),
                restored.grouped_verifier_local_histogram);
            resetRouterHotCacheCounters(restored);
            restored.reserved_u64[2] = 0;
            restored.reserved_u64[3] = 0;
            if (mirror_to_device_)
                uploadLayerState(layer_idx, stream);
        }

        return DeviceMoEPortableRuntimeRestoreResult{
            .restored = true,
            .placement_effect =
                placement_changed
                    ? DeviceMoEPortablePlacementEffect::Changed
                    : DeviceMoEPortablePlacementEffect::Unchanged,
            .requires_device_payload_rehydration =
                requires_device_payload_rehydration,
        };
    }

    void DeviceMoERuntimeTable::ensurePrefillRouteScratchCapacity(int token_capacity, void *stream)
    {
        if (token_capacity <= 0)
            throw std::invalid_argument("[MoERuntimeTable] prefill route scratch token_capacity must be positive");
        if (!mirror_to_device_ || !device_id_.is_gpu())
            throw std::runtime_error("[MoERuntimeTable] prefill route scratch requires a mirrored GPU runtime table");
        (void)checkedRouteCapacity(token_capacity, top_k_);

        if (serial_route_scratch_arena_)
        {
            if (token_capacity >
                serial_route_scratch_arena_->tokenCapacity())
            {
                throw std::logic_error(
                    "[MoERuntimeTable] graph-captured serial route scratch "
                    "capacity exceeded: requested=" +
                    std::to_string(token_capacity) +
                    " planned=" +
                    std::to_string(
                        serial_route_scratch_arena_->tokenCapacity()) +
                    "; captured device addresses are immutable");
            }

            /*
             * Shared graph scratch is fully allocated and bound before any
             * capture starts. A request within the planned capacity is only a
             * contract check: it must not enqueue an upload, synchronize a
             * stream, or mutate host/device runtime records.
             */
            return;
        }

        if (static_cast<int>(prefill_route_scratch_.size()) != num_layers_)
            prefill_route_scratch_.resize(static_cast<size_t>(num_layers_));

        bool changed = false;
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            const auto &allocation = prefill_route_scratch_[static_cast<size_t>(layer_idx)];
            if (!prefillRouteScratchAllocationHasCapacity(allocation, token_capacity))
            {
                allocatePrefillRouteScratchForLayer(layer_idx, token_capacity);
                changed = true;
            }
        }

        if (changed)
        {
            prefill_token_capacity_ = std::max(prefill_token_capacity_, token_capacity);
            for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            {
                uploadLayerState(layer_idx, stream);
                uploadResetTemplatesForLayer(layer_idx, stream);
            }
            synchronizeMirror(device_id_, stream, "[MoERuntimeTable] prefill route scratch upload sync");
        }
    }

    bool DeviceMoERuntimeTable::prepareInactiveBank(int layer_idx, const MoEPlacementUpdate &update)
    {
        validateLayerIndex(layer_idx);

        const size_t layer = static_cast<size_t>(layer_idx);
        if (decode_runtime_publication_required_[layer] != 0u)
        {
            /*
             * A device-only reset deliberately leaves the old host recipe
             * untouched.  Starting a new publication generation must therefore
             * clear that stale epoch before validating the replacement update.
             * Persistent graph scratch is part of model identity and survives
             * the generation boundary.
             */
            auto &staging_state = host_layers_[layer];
            const RuntimePersistentBindings scratch =
                captureRuntimePersistentBindings(staging_state);
            resetLayer(staging_state);
            restoreRuntimePersistentBindings(staging_state, scratch);
        }

        validateUpdate(layer_idx, update);

        auto &state = host_layers_[layer];
        const uint32_t inactive_bank = 1u - state.active_bank;
        auto &bank = state.banks[inactive_bank];
        state.participant_id = update.participant_id;
        state.participant_count = update.participant_count;
        bank = DeviceMoEPlacementBank{};
        bank.epoch = update.epoch;
        bank.expert_count = update.expert_count;
        bank.transient_placement_observed =
            update.transient_placement_observed ? 1u : 0u;

        for (uint32_t expert = 0; expert < update.expert_count; ++expert)
        {
            bank.experts[expert] = update.experts[expert];
            bank.local_compute_mask[expert] = update.local_compute_mask[expert];
            bank.replica_role[expert] = update.replica_role[expert];
            const uint32_t resident_mask = residentParticipantMaskForUpdate(update, expert);
            bank.resident_participant_mask[expert] = resident_mask;
            bank.overlay_route_participant[expert] =
                overlayRouteParticipantForUpdate(update, expert);
            if (participantMaskCount(resident_mask) > 1u)
                ++bank.multi_resident_expert_count;
            if (hasMoEExpertFlag(
                    update.experts[expert].flags,
                    DeviceMoEExpertFlags::TransferSlot))
            {
                bank.transient_placement_observed = 1u;
            }
        }

        return true;
    }

    bool DeviceMoERuntimeTable::flipActiveBank(int layer_idx, uint32_t epoch, void *stream)
    {
        if (mirror_to_device_ && !stream)
        {
            throw std::invalid_argument(
                "[MoERuntimeTable] GPU active-bank publication requires an "
                "explicit non-null producer stream");
        }
        validateLayerIndex(layer_idx);
        auto &state = host_layers_[static_cast<size_t>(layer_idx)];
        const uint32_t inactive_bank = 1u - state.active_bank;
        const auto &prepared_bank = state.banks[inactive_bank];

        if (prepared_bank.epoch == 0 || prepared_bank.expert_count != static_cast<uint32_t>(num_experts_))
            throw std::runtime_error(layerPrefix(layer_idx) + "inactive bank has not been prepared");
        if (prepared_bank.epoch != epoch)
            throw std::invalid_argument(layerPrefix(layer_idx) + "flip epoch does not match prepared inactive bank epoch");
        if (epoch <= state.active_epoch)
            throw std::invalid_argument(layerPrefix(layer_idx) + "flip epoch must increase monotonically");

        state.active_bank = inactive_bank;
        state.active_epoch = epoch;
        captureInitialLayerStateIfNeeded(layer_idx, stream);

        if (mirror_to_device_)
            uploadLayerState(layer_idx, stream);

        decode_runtime_publication_required_[static_cast<size_t>(layer_idx)] = 0u;
        return true;
    }

    void DeviceMoERuntimeTable::validateLayerIndex(int layer_idx) const
    {
        if (layer_idx < 0 || layer_idx >= num_layers_)
            throw std::out_of_range("[MoERuntimeTable] layer index out of range: " + std::to_string(layer_idx));
    }

    void DeviceMoERuntimeTable::validateUpdate(int layer_idx, const MoEPlacementUpdate &update) const
    {
        if (update.epoch == 0)
            throw std::invalid_argument(layerPrefix(layer_idx) + "placement update epoch must be non-zero");
        const auto &state = host_layers_[static_cast<size_t>(layer_idx)];
        if (update.epoch <= state.active_epoch)
            throw std::invalid_argument(layerPrefix(layer_idx) + "placement update epoch must be newer than active epoch");
        if (update.expert_count != static_cast<uint32_t>(num_experts_))
            throw std::invalid_argument(layerPrefix(layer_idx) + "placement update expert_count must match table expert_count");
        if (update.participant_count == 0 || update.participant_count > kDeviceMoEMaxParticipants)
            throw std::invalid_argument(layerPrefix(layer_idx) + "participant_count must be in [1, " +
                                        std::to_string(kDeviceMoEMaxParticipants) + "]");
        if (update.participant_id >= update.participant_count)
            throw std::invalid_argument(layerPrefix(layer_idx) + "participant_id must be less than participant_count");
        if (update.experts.size() != update.expert_count ||
            update.local_compute_mask.size() != update.expert_count ||
            update.replica_role.size() != update.expert_count ||
            (!update.resident_participant_mask.empty() &&
             update.resident_participant_mask.size() != update.expert_count) ||
            (!update.overlay_route_participant.empty() &&
             update.overlay_route_participant.size() != update.expert_count))
        {
            throw std::invalid_argument(layerPrefix(layer_idx) + "placement update vectors must match expert_count");
        }

        const uint32_t valid_participant_mask = participantMaskLimit(update.participant_count);
        for (uint32_t expert = 0; expert < update.expert_count; ++expert)
        {
            const auto &desc = update.experts[expert];
            if (desc.logical_expert_id != -1 && desc.logical_expert_id != static_cast<int32_t>(expert))
                throw std::invalid_argument(layerPrefix(layer_idx) + "descriptor logical_expert_id must match table index");
            if (desc.local_slot < -1)
                throw std::invalid_argument(layerPrefix(layer_idx) + "descriptor local_slot must be -1 or non-negative");
            if (update.local_compute_mask[expert] > 1)
                throw std::invalid_argument(layerPrefix(layer_idx) + "local_compute_mask entries must be 0 or 1");
            if (update.replica_role[expert] > static_cast<uint8_t>(DeviceMoEReplicaRole::PreferredReplica))
                throw std::invalid_argument(layerPrefix(layer_idx) + "replica_role entries must be valid DeviceMoEReplicaRole values");
            if (overlayRouteParticipantForUpdate(update, expert) < -1)
            {
                throw std::invalid_argument(
                    layerPrefix(layer_idx) +
                    "overlay_route_participant entries must be -1 or non-negative");
            }

            const uint32_t resident_mask = residentParticipantMaskForUpdate(update, expert);
            if ((resident_mask & ~valid_participant_mask) != 0u)
                throw std::invalid_argument(layerPrefix(layer_idx) + "resident_participant_mask names a participant outside participant_count");
            if (update.local_compute_mask[expert] != 0u &&
                (resident_mask & participantBit(update.participant_id)) == 0u)
            {
                throw std::invalid_argument(layerPrefix(layer_idx) + "resident_participant_mask must include local participant for local compute");
            }
            if (desc.owner_participant >= 0 &&
                desc.owner_participant < static_cast<int32_t>(update.participant_count) &&
                (resident_mask & participantBit(static_cast<uint32_t>(desc.owner_participant))) == 0u)
            {
                throw std::invalid_argument(layerPrefix(layer_idx) + "resident_participant_mask must include the owner participant");
            }

            if (descriptorRequiresReadyPayload(desc, update.local_compute_mask[expert]))
            {
                if (desc.logical_expert_id != static_cast<int32_t>(expert))
                    throw std::invalid_argument(layerPrefix(layer_idx) + "active descriptor must name its logical expert");
                if (!descriptorReady(desc))
                    throw std::invalid_argument(layerPrefix(layer_idx) + "active descriptor must include ready gate/up/down payload descriptors");
            }
        }
    }

    void DeviceMoERuntimeTable::resetLayer(DeviceMoELayerRuntime &state) const
    {
        state = DeviceMoELayerRuntime{};
        state.expert_count = static_cast<uint32_t>(num_experts_);
        state.top_k = static_cast<uint32_t>(top_k_);
        state.participant_id = 0;
        state.participant_count = 1;
        state.banks[0].expert_count = static_cast<uint32_t>(num_experts_);
        state.banks[1].expert_count = static_cast<uint32_t>(num_experts_);
        /* The ticket address is model topology. Request reset clears routed
         * data but must never unbind the captured residency authority. */
        state.overlay_epoch_ticket = overlayEpochTicket();
        state.overlay_epoch_status = overlayEpochStatus();
    }

    void DeviceMoERuntimeTable::captureInitialLayerStateIfNeeded(
        int layer_idx,
        void *stream)
    {
        const auto idx = static_cast<size_t>(layer_idx);
        if (idx >= initial_layer_captured_.size() ||
            initial_layer_captured_[idx] != 0u)
        {
            return;
        }

        initial_host_layers_[idx] = host_layers_[idx];
        resetPerRequestRuntimeFields(initial_host_layers_[idx], num_experts_);
        initial_layer_captured_[idx] = 1u;
        if (mirror_to_device_)
            uploadResetTemplatesForLayer(layer_idx, stream);
    }

    bool DeviceMoERuntimeTable::prefillRouteScratchAllocationHasCapacity(
        const DeviceMoEPrefillRouteScratchBindings &allocation,
        int token_capacity) const
    {
        const uint32_t route_capacity = checkedRouteCapacity(token_capacity, top_k_);
        return allocation.token_capacity >= static_cast<uint32_t>(token_capacity) &&
               allocation.route_capacity >= route_capacity &&
               allocation.expert_capacity >= static_cast<uint32_t>(num_experts_) &&
               allocation.route_expert_ids &&
               allocation.route_weights &&
               allocation.route_participant_ids &&
               allocation.expert_counts &&
               allocation.expert_offsets &&
               allocation.grouped_token_ids &&
               allocation.grouped_route_weights &&
               allocation.llep_split_ends &&
               allocation.llep_assignment_spans &&
               allocation.llep_weight_transfers &&
               allocation.llep_plan_capacity >=
                   static_cast<uint32_t>(num_experts_) * kDeviceMoEMaxParticipants;
    }

    void DeviceMoERuntimeTable::allocatePrefillRouteScratchForLayer(int layer_idx, int token_capacity)
    {
        validateLayerIndex(layer_idx);
        if (!mirror_to_device_ || !device_id_.is_gpu())
            throw std::runtime_error(layerPrefix(layer_idx) + "prefill route scratch requires a mirrored GPU runtime table");
        if (token_capacity <= 0)
            throw std::invalid_argument(layerPrefix(layer_idx) + "prefill route scratch token capacity must be positive");

        if (static_cast<int>(prefill_route_scratch_.size()) != num_layers_)
            prefill_route_scratch_.resize(static_cast<size_t>(num_layers_));

        auto &allocation = prefill_route_scratch_[static_cast<size_t>(layer_idx)];
        if (prefillRouteScratchAllocationHasCapacity(allocation, token_capacity))
            return;

        releasePrefillRouteScratchBindings(
            device_id_,
            allocation,
            layerPrefix(layer_idx) + "free prefill");
        allocatePrefillRouteScratchBindings(
            device_id_,
            num_experts_,
            top_k_,
            token_capacity,
            allocation,
            layerPrefix(layer_idx) + "prefill route scratch");
        bindPrefillRouteScratchToLayer(layer_idx, allocation);
    }

    void DeviceMoERuntimeTable::bindPrefillRouteScratchToLayer(
        int layer_idx,
        const DeviceMoEPrefillRouteScratchBindings &allocation)
    {
        validateLayerIndex(layer_idx);
        if (!prefillRouteScratchAllocationHasCapacity(
                allocation,
                static_cast<int>(allocation.token_capacity)))
        {
            throw std::invalid_argument(
                layerPrefix(layer_idx) +
                "cannot bind incomplete prefill route scratch");
        }
        auto &state = host_layers_[static_cast<size_t>(layer_idx)];
        state.route_expert_ids = allocation.route_expert_ids;
        state.route_weights = allocation.route_weights;
        state.route_participant_ids = allocation.route_participant_ids;
        state.expert_counts = allocation.expert_counts;
        state.expert_offsets = allocation.expert_offsets;
        state.grouped_token_ids = allocation.grouped_token_ids;
        state.grouped_route_weights = allocation.grouped_route_weights;
        state.reserved_ptrs[0] = allocation.llep_split_ends;
        state.reserved_ptrs[1] = allocation.llep_assignment_spans;
        state.reserved_ptrs[2] = allocation.llep_weight_transfers;
        state.reserved_u64[0] = allocation.llep_plan_capacity;
        state.reserved_u64[1] = allocation.llep_plan_capacity;
        state.reserved_u64[2] = 0;
        state.reserved_u64[3] = 0;
        state.prefill_token_capacity = allocation.token_capacity;
        state.prefill_route_capacity = allocation.route_capacity;

        /*
         * Request-reset templates are model-lifetime device state. Refresh
         * their scratch bindings whenever setup grows the persistent route
         * workspace, while preserving each template's placement semantics.
         * This happens only during setup/replanning, never in inference.
         */
        const auto scratch = captureRuntimePersistentBindings(state);
        auto &empty = empty_host_layers_[static_cast<size_t>(layer_idx)];
        resetLayer(empty);
        restoreRuntimePersistentBindings(empty, scratch);
        if (initial_layer_captured_[static_cast<size_t>(layer_idx)] != 0u)
        {
            restoreRuntimePersistentBindings(
                initial_host_layers_[static_cast<size_t>(layer_idx)],
                scratch);
        }
        else
        {
            initial_host_layers_[static_cast<size_t>(layer_idx)] = empty;
        }
    }

    void DeviceMoERuntimeTable::releasePrefillRouteScratch() noexcept
    {
        if (prefill_route_scratch_.empty())
            return;
        for (auto &allocation : prefill_route_scratch_)
        {
            releasePrefillRouteScratchBindings(
                device_id_,
                allocation,
                "[MoERuntimeTable] free prefill");
        }
        prefill_route_scratch_.clear();
    }

    void DeviceMoERuntimeTable::allocateDeferredVerifierRouteLedger()
    {
        if (!mirror_to_device_ || !device_id_.is_gpu())
        {
            throw std::logic_error(
                "[MoERuntimeTable] deferred verifier ledger allocation requires "
                "a mirrored GPU table");
        }
        if (deferred_verifier_token_capacity_ <= 0)
        {
            throw std::invalid_argument(
                "[MoERuntimeTable] deferred verifier ledger token capacity must "
                "be positive");
        }

        deferred_verifier_route_capacity_ =
            checkedRouteCapacity(deferred_verifier_token_capacity_, top_k_);
        const size_t route_capacity =
            static_cast<size_t>(deferred_verifier_route_capacity_);
        const size_t layer_count = static_cast<size_t>(num_layers_);
        if (route_capacity >
            std::numeric_limits<size_t>::max() / layer_count / sizeof(int32_t))
        {
            throw std::overflow_error(
                "[MoERuntimeTable] deferred verifier route ledger size overflow");
        }
        const size_t entries = route_capacity * layer_count;
        const size_t bytes = entries * sizeof(int32_t);

        try
        {
            deferred_verifier_route_expert_ids_ = static_cast<int32_t *>(
                allocateMirror(
                    device_id_,
                    bytes,
                    "[MoERuntimeTable] deferred verifier expert-route ledger allocation"));
            deferred_verifier_route_participant_ids_ = static_cast<int32_t *>(
                allocateMirror(
                    device_id_,
                    bytes,
                    "[MoERuntimeTable] deferred verifier participant-route ledger allocation"));
            bindDeferredVerifierRouteLedgerToLayers();
        }
        catch (...)
        {
            releaseDeferredVerifierRouteLedger();
            throw;
        }

        PerfStatsCollector::addCounter(
            "memory",
            "moe_deferred_verifier_route_ledger_allocations",
            1.0,
            "model_setup",
            device_id_.toString(),
            {{"bytes", std::to_string(bytes * 2)},
             {"layers", std::to_string(num_layers_)},
             {"route_capacity_per_layer",
              std::to_string(deferred_verifier_route_capacity_)},
             {"ownership", "per_layer_main_verifier"}});
        logVramBomLine(
            "moe_deferred_verifier_route_ledger",
            "device=" + device_id_.toString() +
                " expert_ids_ptr=" +
                vramBomPointer(deferred_verifier_route_expert_ids_) +
                " participant_ids_ptr=" +
                vramBomPointer(deferred_verifier_route_participant_ids_) +
                " ownership=per_layer_main_verifier immutable=true layers=" +
                std::to_string(num_layers_) +
                " route_capacity_per_layer=" +
                std::to_string(deferred_verifier_route_capacity_) +
                " bytes=" + std::to_string(bytes * 2));
    }

    void DeviceMoERuntimeTable::bindDeferredVerifierRouteLedgerToLayers()
    {
        if (!deferred_verifier_route_expert_ids_ ||
            !deferred_verifier_route_participant_ids_ ||
            deferred_verifier_route_capacity_ == 0u)
        {
            throw std::logic_error(
                "[MoERuntimeTable] cannot bind an incomplete deferred verifier ledger");
        }

        const size_t route_capacity =
            static_cast<size_t>(deferred_verifier_route_capacity_);
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            const size_t offset = static_cast<size_t>(layer_idx) * route_capacity;
            auto &state = host_layers_[static_cast<size_t>(layer_idx)];
            state.deferred_verifier_route_expert_ids =
                deferred_verifier_route_expert_ids_ + offset;
            state.deferred_verifier_route_participant_ids =
                deferred_verifier_route_participant_ids_ + offset;
            state.deferred_verifier_route_capacity =
                deferred_verifier_route_capacity_;

            const auto bindings = captureRuntimePersistentBindings(state);
            auto &empty = empty_host_layers_[static_cast<size_t>(layer_idx)];
            resetLayer(empty);
            restoreRuntimePersistentBindings(empty, bindings);
            if (initial_layer_captured_[static_cast<size_t>(layer_idx)] != 0u)
            {
                restoreRuntimePersistentBindings(
                    initial_host_layers_[static_cast<size_t>(layer_idx)],
                    bindings);
            }
            else
            {
                initial_host_layers_[static_cast<size_t>(layer_idx)] = empty;
            }
        }
    }

    void DeviceMoERuntimeTable::releaseDeferredVerifierRouteLedger() noexcept
    {
        freeMirror(
            device_id_,
            deferred_verifier_route_participant_ids_,
            "[MoERuntimeTable] free deferred verifier participant-route ledger");
        freeMirror(
            device_id_,
            deferred_verifier_route_expert_ids_,
            "[MoERuntimeTable] free deferred verifier expert-route ledger");
        deferred_verifier_route_participant_ids_ = nullptr;
        deferred_verifier_route_expert_ids_ = nullptr;
        deferred_verifier_route_capacity_ = 0u;
    }

    void DeviceMoERuntimeTable::allocateDeviceMirror()
    {
        const size_t bytes = host_layers_.size() * sizeof(DeviceMoELayerRuntime);
        try
        {
            device_layers_ = static_cast<DeviceMoELayerRuntime *>(
                allocateMirror(device_id_, bytes, "[MoERuntimeTable] runtime table mirror allocation"));
            device_initial_layers_ = static_cast<DeviceMoELayerRuntime *>(
                allocateMirror(device_id_, bytes, "[MoERuntimeTable] initial runtime template allocation"));
            device_empty_layers_ = static_cast<DeviceMoELayerRuntime *>(
                allocateMirror(device_id_, bytes, "[MoERuntimeTable] empty runtime template allocation"));
        }
        catch (...)
        {
            releaseDeviceMirror();
            throw;
        }
    }

    void DeviceMoERuntimeTable::allocateOverlayServiceTelemetry()
    {
        if (overlay_service_telemetry_coverage_ ==
                MoEOverlayServiceTelemetryCoverage::Disabled ||
            overlay_placement_source_ ||
            device_overlay_service_telemetry_)
        {
            return;
        }
        if (!mirror_to_device_ || !device_id_.is_gpu())
        {
            throw std::logic_error(
                "[MoERuntimeTable] service telemetry allocation requires the canonical mirrored GPU table");
        }

        const std::size_t cell_count =
            deviceMoEOverlayServiceTelemetryCellCount(
                static_cast<std::size_t>(num_layers_));
        const std::size_t bytes =
            cell_count * sizeof(DeviceMoEOverlayServiceTelemetryCell);
        const std::size_t sample_bytes =
            static_cast<std::size_t>(num_layers_) *
            sizeof(DeviceMoEOverlayServiceTelemetrySample);
        void *setup_stream = nullptr;
        try
        {
            device_overlay_service_telemetry_ = static_cast<
                DeviceMoEOverlayServiceTelemetryCell *>(
                    allocateMirror(
                        device_id_,
                        bytes,
                        "[MoERuntimeTable] ExpertOverlay service telemetry allocation"));
            device_overlay_service_samples_ = static_cast<
                DeviceMoEOverlayServiceTelemetrySample *>(
                    allocateMirror(
                        device_id_,
                        sample_bytes,
                        "[MoERuntimeTable] ExpertOverlay service sample allocation"));
            setup_stream = createMirrorStream(
                device_id_,
                "[MoERuntimeTable] ExpertOverlay service telemetry setup");
            memsetMirror(
                device_id_,
                device_overlay_service_telemetry_,
                0,
                bytes,
                setup_stream,
                "[MoERuntimeTable] zero ExpertOverlay service telemetry");
            memsetMirror(
                device_id_,
                device_overlay_service_samples_,
                0,
                sample_bytes,
                setup_stream,
                "[MoERuntimeTable] zero ExpertOverlay service samples");
            /*
             * This one synchronization closes model-setup initialization
             * before any graph can capture the stable accumulator address. It
             * is not a request/inference operation and never orders a live
             * producer. Subsequent timing and snapshot work uses stream/event
             * edges only.
             */
            synchronizeMirror(
                device_id_,
                setup_stream,
                "[MoERuntimeTable] initialize ExpertOverlay service telemetry");
            destroyMirrorStream(
                device_id_,
                setup_stream,
                "[MoERuntimeTable] ExpertOverlay service telemetry setup");
            setup_stream = nullptr;
            PerfStatsCollector::addCounter(
                "memory",
                "moe_overlay_service_telemetry_bytes",
                static_cast<double>(bytes),
                "model_setup",
                device_id_.toString(),
                {{"layers", std::to_string(num_layers_)},
                 {"phases", std::to_string(
                      kDeviceMoEOverlayServicePhaseCount)},
                 {"authority", "device_local"}});
        }
        catch (...)
        {
            destroyMirrorStream(
                device_id_,
                setup_stream,
                "[MoERuntimeTable] failed ExpertOverlay service telemetry setup");
            releaseOverlayServiceTelemetry();
            throw;
        }
    }

    void DeviceMoERuntimeTable::releaseOverlayServiceTelemetry() noexcept
    {
        if (overlay_placement_source_)
        {
            /* The canonical table owns the shared main/MTP allocation. */
            device_overlay_service_telemetry_ = nullptr;
            device_overlay_service_samples_ = nullptr;
            return;
        }
        freeMirror(
            device_id_,
            device_overlay_service_samples_,
            "[MoERuntimeTable] free ExpertOverlay service samples");
        freeMirror(
            device_id_,
            device_overlay_service_telemetry_,
            "[MoERuntimeTable] free ExpertOverlay service telemetry");
        device_overlay_service_telemetry_ = nullptr;
        device_overlay_service_samples_ = nullptr;
    }

    void DeviceMoERuntimeTable::allocateGroupedVerifierHistogramPublicationStream()
    {
        if (!mirror_to_device_ ||
            grouped_verifier_histogram_publication_ !=
                GroupedVerifierHistogramPublicationMode::AcceptedRows ||
            grouped_verifier_histogram_publication_stream_)
        {
            throw std::logic_error(
                "[MoERuntimeTable] grouped-verifier publication stream has an invalid setup lifecycle");
        }

        IBackend *backend = mirrorBackend(
            device_id_,
            "[MoERuntimeTable] grouped-verifier publication setup");
        grouped_verifier_histogram_publication_stream_ =
            backend->createStream(device_id_.toKernelDeviceIndex());
        if (!grouped_verifier_histogram_publication_stream_)
        {
            throw std::runtime_error(
                "[MoERuntimeTable] backend failed to allocate the model-lifetime grouped-verifier publication stream");
        }
    }

    void DeviceMoERuntimeTable::allocateRuntimeHistogramDrainResources()
    {
        if (!mirror_to_device_ ||
            device_runtime_histogram_banks_ ||
            device_runtime_histogram_active_bank_ ||
            host_runtime_histogram_snapshot_ ||
            host_runtime_histogram_writer_states_ ||
            runtime_histogram_maintenance_stream_ ||
            runtime_histogram_initialization_event_ ||
            runtime_histogram_writer_state_published_event_ ||
            runtime_histogram_drain_complete_event_)
        {
            throw std::logic_error(
                "[MoERuntimeTable] asynchronous histogram resources have an invalid setup lifecycle");
        }

        IBackend *backend = mirrorBackend(
            device_id_,
            "[MoERuntimeTable] asynchronous histogram setup");
        const int ordinal = device_id_.toKernelDeviceIndex();
        const std::size_t bank_count =
            static_cast<std::size_t>(num_layers_) * 2u;
        const std::size_t device_bytes =
            bank_count * sizeof(DeviceMoERuntimeHistogramBank);
        const std::size_t snapshot_bytes =
            static_cast<std::size_t>(num_layers_) *
            sizeof(DeviceMoERuntimeHistogramBank);

        try
        {
            device_runtime_histogram_banks_ =
                static_cast<DeviceMoERuntimeHistogramBank *>(
                    allocateMirror(
                        device_id_,
                        device_bytes,
                        "[MoERuntimeTable] asynchronous histogram banks"));
            device_runtime_histogram_active_bank_ =
                static_cast<uint32_t *>(
                    allocateMirror(
                        device_id_,
                        sizeof(uint32_t),
                        "[MoERuntimeTable] asynchronous histogram active bank"));
            host_runtime_histogram_snapshot_ =
                static_cast<DeviceMoERuntimeHistogramBank *>(
                    backend->allocatePinned(snapshot_bytes, ordinal));
            host_runtime_histogram_writer_states_ =
                static_cast<uint32_t *>(
                    backend->allocatePinned(4u * sizeof(uint32_t), ordinal));
            runtime_histogram_maintenance_stream_ =
                backend->createStream(ordinal);
            const auto grouped_verifier_source = static_cast<std::size_t>(
                moe_runtime_abi::HistogramSource::GroupedVerifier);
            runtime_histogram_initialization_event_ =
                backend->createEvent(ordinal);
            runtime_histogram_writer_state_published_event_ =
                backend->createEvent(ordinal);
            runtime_histogram_drain_complete_event_ =
                backend->createEvent(ordinal);
            if (!host_runtime_histogram_snapshot_ ||
                !host_runtime_histogram_writer_states_ ||
                !runtime_histogram_maintenance_stream_ ||
                (runtime_histogram_sources_[grouped_verifier_source] &&
                 !grouped_verifier_histogram_publication_stream_) ||
                !runtime_histogram_initialization_event_ ||
                !runtime_histogram_writer_state_published_event_ ||
                !runtime_histogram_drain_complete_event_)
            {
                throw std::runtime_error(
                    "[MoERuntimeTable] backend failed to allocate persistent asynchronous histogram resources");
            }

            for (uint32_t state = 0u;
                 state <= moe_runtime_abi::kHistogramWriterStateMask;
                 ++state)
            {
                host_runtime_histogram_writer_states_[state] = state;
            }
            runtime_histogram_active_bank_host_ = 0u;
            runtime_histogram_frozen_bank_host_ = 0u;

            if (!backend->memset(
                    device_runtime_histogram_banks_,
                    0,
                    device_bytes,
                    ordinal,
                    runtime_histogram_maintenance_stream_) ||
                !backend->hostToDeviceOnStream(
                    device_runtime_histogram_active_bank_,
                    host_runtime_histogram_writer_states_,
                    sizeof(uint32_t),
                    ordinal,
                    runtime_histogram_maintenance_stream_) ||
                !backend->recordEvent(
                    runtime_histogram_initialization_event_,
                    ordinal,
                    runtime_histogram_maintenance_stream_))
            {
                throw std::runtime_error(
                    "[MoERuntimeTable] failed to initialize persistent asynchronous histogram resources");
            }

            /* Accepted-state publication can acquire a new capture identity
             * after maintenance has started because its verifier bindings are
             * depth/policy specific. Admit its sole table-owned stream now,
             * before the first poll can seal producer topology. The graph
             * owner's typed capture-activity edge then makes every such later
             * materialization mutually exclusive with maintenance submission;
             * replay merely borrows the already-proven stream identity. */
            if (grouped_verifier_histogram_publication_stream_)
            {
                registerRuntimeHistogramProducerStreamLocked(
                    grouped_verifier_histogram_publication_stream_,
                    RuntimeHistogramProducerStream::Ownership::
                        TableOwnedStream);
            }

            /* The event, rather than a host fence, publishes setup completion.
             * Producer registration installs a one-time stream wait before any
             * captured or eager production kernel may write these banks. */

            /* Each layer points at its adjacent two-bank pair. Request-reset
             * templates receive the same model-lifetime addresses, so a D2D
             * reset changes request state without erasing routing evidence. */
            for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            {
                const auto layer_offset =
                    static_cast<std::size_t>(layer_idx) * 2u;
                const auto bind = [&](DeviceMoELayerRuntime &state)
                {
                    state.runtime_histogram_banks =
                        device_runtime_histogram_banks_ + layer_offset;
                    state.runtime_histogram_active_bank =
                        device_runtime_histogram_active_bank_;
                };
                bind(host_layers_[static_cast<std::size_t>(layer_idx)]);
                bind(initial_host_layers_[static_cast<std::size_t>(layer_idx)]);
                bind(empty_host_layers_[static_cast<std::size_t>(layer_idx)]);
            }

            /* Enabling is model-setup only, immediately after construction.
             * Re-uploading the pristine tables publishes the external pointer
             * identity before any graph can capture it. */
            uploadAllLayerStates();
            PerfStatsCollector::addCounter(
                "memory",
                "moe_runtime_async_histogram_bytes",
                static_cast<double>(
                    device_bytes + sizeof(uint32_t) + snapshot_bytes +
                    4u * sizeof(uint32_t)),
                "model_setup",
                device_id_.toString(),
                {{"layers", std::to_string(num_layers_)},
                 {"banks", "2"},
                 {"persistent", "true"}});
        }
        catch (...)
        {
            releaseRuntimeHistogramResources();
            throw;
        }
    }

    void DeviceMoERuntimeTable::registerRuntimeHistogramProducerStreamLocked(
        void *stream,
        RuntimeHistogramProducerStream::Ownership ownership)
    {
        if (!stream || !runtime_histogram_initialization_event_)
        {
            throw std::logic_error(
                "[MoERuntimeTable] runtime histogram producer registration requires an exact stream and a published initialization event");
        }

        const auto existing = std::find_if(
            runtime_histogram_producer_streams_.begin(),
            runtime_histogram_producer_streams_.end(),
            [stream](const RuntimeHistogramProducerStream &producer)
            { return producer.stream == stream; });
        if (existing != runtime_histogram_producer_streams_.end())
            return;
        if (runtime_histogram_producer_topology_ ==
            RuntimeHistogramProducerTopology::Sealed)
        {
            throw std::logic_error(
                "[MoERuntimeTable] a new histogram producer stream appeared after the asynchronous drain topology was sealed");
        }

        IBackend *backend = mirrorBackend(
            device_id_,
            "[MoERuntimeTable] runtime histogram producer registration");
        const int ordinal = device_id_.toKernelDeviceIndex();
        void *arrival_event = backend->createEvent(ordinal);
        void *departure_event = backend->createEvent(ordinal);
        if (!arrival_event || !departure_event)
        {
            if (arrival_event)
                backend->destroyEvent(arrival_event, ordinal);
            if (departure_event)
                backend->destroyEvent(departure_event, ordinal);
            throw std::runtime_error(
                "[MoERuntimeTable] failed to allocate exact two-phase histogram producer events on " +
                device_id_.to_string());
        }
        if (!backend->streamWaitEvent(
                stream,
                runtime_histogram_initialization_event_,
                ordinal))
        {
            backend->destroyEvent(arrival_event, ordinal);
            backend->destroyEvent(departure_event, ordinal);
            throw std::runtime_error(
                "[MoERuntimeTable] failed to order a histogram producer after asynchronous bank initialization on " +
                device_id_.to_string());
        }
        runtime_histogram_producer_streams_.push_back(
            {.stream = stream,
             .flip_arrival_event = arrival_event,
             .flip_departure_event = departure_event,
             .ownership = ownership});
    }

    bool DeviceMoERuntimeTable::isRuntimeHistogramProducerStreamRegisteredLocked(
        void *stream) const noexcept
    {
        return std::any_of(
            runtime_histogram_producer_streams_.begin(),
            runtime_histogram_producer_streams_.end(),
            [stream](const RuntimeHistogramProducerStream &producer)
            { return producer.stream == stream; });
    }

    bool DeviceMoERuntimeTable::
        hasActiveRuntimeHistogramProducerCaptureLocked() const noexcept
    {
        return std::any_of(
            runtime_histogram_producer_streams_.begin(),
            runtime_histogram_producer_streams_.end(),
            [](const RuntimeHistogramProducerStream &producer)
            { return producer.active_capture_references != 0u; });
    }

    bool DeviceMoERuntimeTable::publishRuntimeHistogramWriterStateLocked(
        uint32_t writer_state,
        std::string &failure)
    {
        failure.clear();
        if (!moe_runtime_abi::validHistogramWriterState(writer_state))
        {
            failure =
                "Runtime histogram writer-state publication received an invalid encoded state";
            return false;
        }
        if (hasActiveRuntimeHistogramProducerCaptureLocked())
        {
            failure =
                "Runtime histogram writer-state publication overlapped native producer capture";
            return false;
        }
        if (runtime_histogram_producer_streams_.empty() ||
            !device_runtime_histogram_active_bank_ ||
            !host_runtime_histogram_writer_states_ ||
            !runtime_histogram_maintenance_stream_ ||
            !runtime_histogram_writer_state_published_event_)
        {
            failure =
                "Runtime histogram writer-state publication has incomplete persistent resources";
            return false;
        }

        IBackend *backend = mirrorBackend(
            device_id_,
            "[MoERuntimeTable] runtime histogram writer-state publication");
        const int ordinal = device_id_.toKernelDeviceIndex();

        /* Arrival events close the prior producer generation. They are
         * recorded before the corresponding producer wait below, making the
         * event DAG acyclic even while inference is still draining. */
        for (const auto &producer : runtime_histogram_producer_streams_)
        {
            if (!backend->recordEvent(
                    producer.flip_arrival_event,
                    ordinal,
                    producer.stream))
            {
                failure =
                    "Failed to record a runtime histogram producer arrival";
                return false;
            }
        }
        for (const auto &producer : runtime_histogram_producer_streams_)
        {
            if (!backend->streamWaitEvent(
                    runtime_histogram_maintenance_stream_,
                    producer.flip_arrival_event,
                    ordinal))
            {
                failure =
                    "Maintenance stream failed to join a runtime histogram producer arrival";
                return false;
            }
        }

        /* This is the only live mutation of the shared device scalar. A
         * single writer is required even when several producers would publish
         * identical bytes: concurrent DMA to graph-read state has no defined
         * cross-stream ordering. */
        if (!backend->hostToDeviceOnStream(
                device_runtime_histogram_active_bank_,
                host_runtime_histogram_writer_states_ + writer_state,
                sizeof(uint32_t),
                ordinal,
                runtime_histogram_maintenance_stream_))
        {
            failure =
                "Failed to enqueue the sole runtime histogram writer-state publication";
            return false;
        }
        if (!backend->recordEvent(
                runtime_histogram_writer_state_published_event_,
                ordinal,
                runtime_histogram_maintenance_stream_))
        {
            failure =
                "Failed to record the runtime histogram writer-state publication event";
            return false;
        }
        for (const auto &producer : runtime_histogram_producer_streams_)
        {
            if (!backend->streamWaitEvent(
                    producer.stream,
                    runtime_histogram_writer_state_published_event_,
                    ordinal) ||
                !backend->recordEvent(
                    producer.flip_departure_event,
                    ordinal,
                    producer.stream))
            {
                failure =
                    "A runtime histogram producer failed to consume and acknowledge the writer-state publication";
                return false;
            }
        }

        /* Another host thread may submit a retained graph between any two
         * backend calls above. Such work is necessarily ordered either before
         * the arrival, between arrival and departure, or after departure. By
         * joining the departure phase here, the frozen bank cannot be drained
         * while an interleaved graph still owns its old writer-state snapshot. */
        for (const auto &producer : runtime_histogram_producer_streams_)
        {
            if (!backend->streamWaitEvent(
                    runtime_histogram_maintenance_stream_,
                    producer.flip_departure_event,
                    ordinal))
            {
                failure =
                    "Maintenance stream failed to join a runtime histogram producer departure";
                return false;
            }
        }
        return true;
    }

    void DeviceMoERuntimeTable::
        retireRuntimeHistogramProducerStreamsLocked()
    {
        if (runtime_histogram_producer_lifecycle_ ==
                RuntimeHistogramProducerLifecycle::ResourcesReleased ||
            runtime_histogram_producer_lifecycle_ ==
                RuntimeHistogramProducerLifecycle::ProducersRetired)
        {
            return;
        }

        IBackend *backend = nullptr;
        const int ordinal = device_id_.is_gpu()
                                ? device_id_.toKernelDeviceIndex()
                                : 0;
        if (device_id_.is_gpu() && runtime_histogram_maintenance_stream_)
        {
            backend = mirrorBackend(
                device_id_,
                "[MoERuntimeTable] runtime histogram producer retirement");

            if (hasActiveRuntimeHistogramProducerCaptureLocked())
            {
                throw std::logic_error(
                    "[MoERuntimeTable] histogram producer retirement overlapped native graph capture on " +
                    device_id_.to_string());
            }

            /* No graph submission may begin after this terminal transition.
             * Recording each producer's model-lifetime arrival event closes
             * its exact queue without guessing a replacement stream. The one
             * maintenance fence then covers initialization, any in-flight bank
             * drain, and all producer work admitted before teardown. */
            for (const auto &producer :
                 runtime_histogram_producer_streams_)
            {
                if (!producer.stream || !producer.flip_arrival_event)
                {
                    throw std::logic_error(
                        "[MoERuntimeTable] terminal histogram retirement found an incomplete producer registration on " +
                        device_id_.to_string());
                }
                if (!backend->recordEvent(
                        producer.flip_arrival_event,
                        ordinal,
                        producer.stream) ||
                    !backend->streamWaitEvent(
                        runtime_histogram_maintenance_stream_,
                        producer.flip_arrival_event,
                        ordinal))
                {
                    throw std::runtime_error(
                        "[MoERuntimeTable] failed to join a histogram producer into the terminal maintenance fence on " +
                        device_id_.to_string());
                }
            }

            /* Model teardown may block after request admission has stopped.
             * This is the sole terminal host fence for the entire producer
             * family, never a request, graph, or maintenance-wave edge. */
            if (!backend->synchronizeStream(
                    runtime_histogram_maintenance_stream_, ordinal))
            {
                throw std::runtime_error(
                    "[MoERuntimeTable] failed to drain the terminal histogram maintenance fence on " +
                    device_id_.to_string());
            }
        }

        /* Events remain table-owned until resource destruction. Only the raw
         * borrowed stream identities must disappear before graph contexts do.
         * The table-owned grouped-verifier stream remains live so release can
         * destroy it after graph executables have relinquished it. */
        for (auto &producer : runtime_histogram_producer_streams_)
        {
            if (producer.ownership ==
                RuntimeHistogramProducerStream::Ownership::
                    BorrowedExecutionStream)
            {
                producer.stream = nullptr;
            }
        }
        decode_histogram_producer_stream_ = nullptr;
        runtime_histogram_producer_lifecycle_ =
            RuntimeHistogramProducerLifecycle::ProducersRetired;
    }

    void DeviceMoERuntimeTable::releaseRuntimeHistogramResources() noexcept
    {
        if (runtime_histogram_producer_lifecycle_ ==
            RuntimeHistogramProducerLifecycle::ResourcesReleased)
        {
            return;
        }

        if (runtime_histogram_producer_lifecycle_ ==
            RuntimeHistogramProducerLifecycle::CollectingProducers)
        {
            const bool owns_borrowed_stream = std::any_of(
                runtime_histogram_producer_streams_.begin(),
                runtime_histogram_producer_streams_.end(),
                [](const RuntimeHistogramProducerStream &producer)
                {
                    return producer.ownership ==
                           RuntimeHistogramProducerStream::Ownership::
                               BorrowedExecutionStream;
                });
            if (owns_borrowed_stream)
            {
                LOG_ERROR(
                    "[MoERuntimeTable] borrowed histogram producer streams reached resource destruction before explicit retirement on "
                    << device_id_.to_string());
                std::terminate();
            }

            /* Constructor-failure cleanup can own only table-created streams.
             * Retiring those streams here is safe because their lifetime is a
             * strict subset of this table's lifetime. */
            try
            {
                retireRuntimeHistogramProducerStreamsLocked();
            }
            catch (const std::exception &error)
            {
                LOG_ERROR(
                    "[MoERuntimeTable] table-owned histogram producer retirement failed on "
                    << device_id_.to_string() << ": " << error.what());
                std::terminate();
            }
            catch (...)
            {
                LOG_ERROR(
                    "[MoERuntimeTable] table-owned histogram producer retirement failed on "
                    << device_id_.to_string() << ": unknown exception");
                std::terminate();
            }
        }

        IBackend *backend = nullptr;
        try
        {
            backend = device_id_.is_gpu() ? getBackendFor(device_id_) : nullptr;
            const int ordinal = device_id_.is_gpu()
                                    ? device_id_.toKernelDeviceIndex()
                                    : 0;
            if (backend)
            {
                for (auto &producer : runtime_histogram_producer_streams_)
                {
                    if (producer.flip_arrival_event)
                        backend->destroyEvent(
                            producer.flip_arrival_event, ordinal);
                    if (producer.flip_departure_event)
                        backend->destroyEvent(
                            producer.flip_departure_event, ordinal);
                }
                if (runtime_histogram_drain_complete_event_)
                    backend->destroyEvent(
                        runtime_histogram_drain_complete_event_, ordinal);
                if (runtime_histogram_writer_state_published_event_)
                    backend->destroyEvent(
                        runtime_histogram_writer_state_published_event_,
                        ordinal);
                if (runtime_histogram_initialization_event_)
                    backend->destroyEvent(
                        runtime_histogram_initialization_event_, ordinal);
                if (runtime_histogram_maintenance_stream_)
                    backend->destroyStream(
                        runtime_histogram_maintenance_stream_, ordinal);
                if (grouped_verifier_histogram_publication_stream_)
                    backend->destroyStream(
                        grouped_verifier_histogram_publication_stream_,
                        ordinal);
                if (host_runtime_histogram_snapshot_)
                    backend->freePinned(
                        host_runtime_histogram_snapshot_, ordinal);
                if (host_runtime_histogram_writer_states_)
                    backend->freePinned(
                        host_runtime_histogram_writer_states_, ordinal);
            }
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "[MoERuntimeTable] asynchronous histogram teardown failed on "
                << device_id_.to_string() << ": " << error.what());
        }
        catch (...)
        {
            LOG_ERROR(
                "[MoERuntimeTable] asynchronous histogram teardown failed on "
                << device_id_.to_string() << ": unknown exception");
        }

        runtime_histogram_producer_streams_.clear();
        runtime_histogram_drain_complete_event_ = nullptr;
        runtime_histogram_writer_state_published_event_ = nullptr;
        runtime_histogram_initialization_event_ = nullptr;
        runtime_histogram_maintenance_stream_ = nullptr;
        grouped_verifier_histogram_publication_stream_ = nullptr;
        host_runtime_histogram_snapshot_ = nullptr;
        host_runtime_histogram_writer_states_ = nullptr;
        freeMirror(
            device_id_,
            device_runtime_histogram_active_bank_,
            "[MoERuntimeTable] free asynchronous histogram active bank");
        freeMirror(
            device_id_,
            device_runtime_histogram_banks_,
            "[MoERuntimeTable] free asynchronous histogram banks");
        device_runtime_histogram_active_bank_ = nullptr;
        device_runtime_histogram_banks_ = nullptr;
        runtime_histogram_drain_in_flight_ = false;
        runtime_histogram_producer_topology_ =
            RuntimeHistogramProducerTopology::Sealed;
        runtime_histogram_drain_enabled_ = false;
        runtime_histogram_producer_lifecycle_ =
            RuntimeHistogramProducerLifecycle::ResourcesReleased;
    }

    bool DeviceMoERuntimeTable::mergeRuntimeHistogramSnapshot(
        DecodeExpertHistogram &histogram)
    {
        const auto &hist_config = histogram.config();
        if (!host_runtime_histogram_snapshot_ ||
            hist_config.num_layers != num_layers_ ||
            hist_config.num_experts != num_experts_ ||
            hist_config.top_k != top_k_)
        {
            return false;
        }

        constexpr std::array<ExpertHistogramSource,
                             moe_runtime_abi::kHistogramSourceCount>
            sources{
                ExpertHistogramSource::DecodeToken,
                ExpertHistogramSource::PrefillChunk,
                ExpertHistogramSource::GroupedVerifier,
            };
        constexpr std::array<const char *,
                             moe_runtime_abi::kHistogramSourceCount>
            source_names{"decode", "prefill", "grouped_verifier"};

        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            const auto &bank =
                host_runtime_histogram_snapshot_[
                    static_cast<std::size_t>(layer_idx)];
            for (std::size_t source = 0;
                 source < moe_runtime_abi::kHistogramSourceCount;
                 ++source)
            {
                if (!runtime_histogram_sources_[source])
                    continue;
                histogram.mergeLayerCounts(
                    layer_idx,
                    bank.selected[source],
                    num_experts_,
                    /*count_window_tokens=*/
                        sources[source] !=
                        ExpertHistogramSource::DecodeToken,
                    sources[source]);

                if (PerfStatsCollector::isDomainEnabled("moe_rebalance"))
                {
                    const uint64_t selected_slots = std::accumulate(
                        bank.selected[source],
                        bank.selected[source] + num_experts_,
                        uint64_t{0});
                    const uint64_t local_slots = std::accumulate(
                        bank.local[source],
                        bank.local[source] + num_experts_,
                        uint64_t{0});
                    PerfStatsCollector::addCounter(
                        "moe_rebalance",
                        "runtime_phase_selected_slots",
                        static_cast<double>(selected_slots),
                        "rebalance",
                        device_id_.toString(),
                        {{"layer", std::to_string(layer_idx)},
                         {"phase", source_names[source]},
                         {"async", "true"},
                         {"reset", "true"}});
                    PerfStatsCollector::addCounter(
                        "moe_rebalance",
                        "runtime_phase_local_compute_slots",
                        static_cast<double>(local_slots),
                        "rebalance",
                        device_id_.toString(),
                        {{"layer", std::to_string(layer_idx)},
                         {"phase", source_names[source]},
                         {"async", "true"},
                         {"reset", "true"}});
                }
            }
        }
        return true;
    }

    void DeviceMoERuntimeTable::releaseDeviceMirror() noexcept
    {
        freeMirror(device_id_, device_empty_layers_, "[MoERuntimeTable] free empty runtime template");
        freeMirror(device_id_, device_initial_layers_, "[MoERuntimeTable] free initial runtime template");
        freeMirror(device_id_, device_layers_, "[MoERuntimeTable] free runtime table mirror");
        device_empty_layers_ = nullptr;
        device_initial_layers_ = nullptr;
        device_layers_ = nullptr;
    }

    void DeviceMoERuntimeTable::uploadLayerState(int layer_idx, void *stream)
    {
        if (mirror_to_device_ && !stream)
        {
            throw std::invalid_argument(
                "[MoERuntimeTable] GPU runtime-table upload requires an "
                "explicit non-null producer stream");
        }
        auto *dst = device_layers_ + layer_idx;
        auto *src = host_layers_.data() + layer_idx;
        copyHostToMirror(device_id_, dst, src, sizeof(DeviceMoELayerRuntime), stream,
                         layerPrefix(layer_idx) + "runtime table upload");
    }

    void DeviceMoERuntimeTable::uploadResetTemplatesForLayer(
        int layer_idx,
        void *stream)
    {
        if (mirror_to_device_ && !stream)
        {
            throw std::invalid_argument(
                "[MoERuntimeTable] GPU runtime-template upload requires an "
                "explicit non-null producer stream");
        }
        validateLayerIndex(layer_idx);
        const auto idx = static_cast<size_t>(layer_idx);
        copyHostToMirror(
            device_id_,
            device_initial_layers_ + layer_idx,
            initial_host_layers_.data() + idx,
            sizeof(DeviceMoELayerRuntime),
            stream,
            layerPrefix(layer_idx) + "initial runtime template upload");
        copyHostToMirror(
            device_id_,
            device_empty_layers_ + layer_idx,
            empty_host_layers_.data() + idx,
            sizeof(DeviceMoELayerRuntime),
            stream,
            layerPrefix(layer_idx) + "empty runtime template upload");
    }

    void DeviceMoERuntimeTable::uploadAllLayerStates()
    {
        // Create a dedicated one-shot stream for the bulk initialization upload.
        // We avoid the default stream (nullptr/0) to maintain stream hygiene --
        // all async GPU work must use an explicit stream for correctness and overlap.
        void *init_stream = createMirrorStream(device_id_, "[MoERuntimeTable] runtime table initial upload stream");
        const size_t bytes = host_layers_.size() * sizeof(DeviceMoELayerRuntime);
        try
        {
            copyHostToMirror(device_id_, device_layers_, host_layers_.data(), bytes, init_stream,
                             "[MoERuntimeTable] runtime table initial upload");
            copyHostToMirror(
                device_id_,
                device_initial_layers_,
                initial_host_layers_.data(),
                bytes,
                init_stream,
                "[MoERuntimeTable] initial runtime templates upload");
            copyHostToMirror(
                device_id_,
                device_empty_layers_,
                empty_host_layers_.data(),
                bytes,
                init_stream,
                "[MoERuntimeTable] empty runtime templates upload");
            synchronizeMirror(device_id_, init_stream, "[MoERuntimeTable] runtime table initial upload sync");
            destroyMirrorStream(device_id_, init_stream, "[MoERuntimeTable] runtime table initial upload stream destroy");
        }
        catch (...)
        {
            destroyMirrorStream(device_id_, init_stream, "[MoERuntimeTable] runtime table initial upload stream destroy");
            throw;
        }
    }

} // namespace llaminar2
