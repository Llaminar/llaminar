/**
 * @file GpuExpertSlotPool.cpp
 * @brief Epoch-aware lease accounting for persistent GPU expert slots.
 *
 * Physical device addresses are allocated once by `LoadOrchestrator`. This
 * implementation changes only logical assignment under a mutex; destruction
 * of the last lease alias returns a slot without freeing device storage.
 */

#include "execution/moe/GpuExpertSlotPool.h"
#include "GPUExpertTransfer.h"
#include "loaders/GPUPreparedWeightPoolLayout.h"

#include "../../backends/IBackend.h"
#include "../../loaders/gpu_pipeline/LoadOrchestrator.h"
#include "../../utils/Logger.h"
#include "../../utils/PerfStatsCollector.h"
#include "../../utils/VramBillOfMaterials.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        struct SlotLeaseToken
        {
            int slot_index = -1;
            int layer_idx = -1;
            int expert_id = -1;
            uint64_t residency_epoch = 0;
            bool transfer_slot = false;
            std::shared_ptr<LoadOrchestrator> orchestrator;
            std::weak_ptr<GpuExpertSlotPool> pool;
        };
    } // namespace

    std::optional<WeightVRAMPool::SuperblockCodedMetadata>
    GpuExpertSlotPool::ProjectionSpec::codedDeclaration() const
    {
        if (format.isNativeQuantised())
        {
            const auto *source = native_quantised_formats::forSourceIdentity(
                format.native_quantised.codebook_id, format.native_quantised.is_superblock);
            if (source && cpu::native_quantised::migrationUsesSuperblockCodes(*source) &&
                gpu_superblock_phase < native_quantised_superblock::kBlocksPerSuperblock)
                return WeightVRAMPool::SuperblockCodedMetadata{
                    canonicalDeviceVnniCodebookId(*source), gpu_superblock_phase};
        }
        if (gpu_superblock_phase != 0)
            throw std::invalid_argument("GPU expert slot phase requires a valid coded source root");
        return std::nullopt;
    }

    GpuExpertSlotPool::ProjectionSpec GpuExpertSlotPool::migrationProjectionSpec(
        std::string label, int n, int k, ExpertWeightFormat format, uint8_t phase)
    {
        ProjectionSpec spec{.label = std::move(label), .N = n, .K = k,
                            .format = format, .gpu_superblock_phase = phase};
        if (!format.valid() || n <= 0 || k <= 0 || (format.isNativeQuantised() && k % 32))
            throw std::invalid_argument("Invalid reusable GPU expert projection geometry");
        if (format.isNativeQuantised())
        {
            const auto *source = native_quantised_formats::forSourceIdentity(
                format.native_quantised.codebook_id, format.native_quantised.is_superblock);
            if (!source)
                throw std::invalid_argument("Reusable GPU expert projection source format is not catalogued");
            const auto storage = gpuPreparedWeightPoolAllocation(*source, GPUPreparedWeightPoolLayout::MigrationReusable);
            spec.payload_bytes_per_block = storage.payload_bytes_per_block;
            spec.is_asymmetric = storage.has_mins;
            spec.has_emins = storage.has_emins;
            spec.codebook_id = canonicalDeviceVnniCodebookId(*source);
            spec.scale_plane = nativeQuantisedScalePlaneWidth(spec.codebook_id);
        }
        if (phase >= native_quantised_superblock::kBlocksPerSuperblock || (phase && !spec.codedDeclaration()))
            throw std::invalid_argument("Invalid reusable GPU expert projection phase");
        return spec;
    }

    GpuExpertSlotPool::ProjectionSlot GpuExpertSlotPool::adoptResidentProjection(
        ProjectionSpec family, const GpuExpertPackedDescriptor &resident)
    {
        const auto fail = [&family]() {
            throw std::invalid_argument("Initial GPU expert does not match its reusable slot family: " + family.label);
        };
        if (!family.format.isNativeQuantised() || !resident.valid() ||
            resident.n != family.N || resident.k != family.K || resident.blocks_per_row != uint32_t(family.K / 32) ||
            resident.codebook_id != family.codebook_id ||
            resident.allocation_payload_bytes_per_block != family.payload_bytes_per_block ||
            resident.allocation_has_mins != family.is_asymmetric || resident.allocation_has_emins != family.has_emins)
            fail();
        const auto coded = family.codedDeclaration();
        if (coded)
        {
            namespace sb = native_quantised_superblock;
            const auto bits = resident.blockCount() * sb::superblockCodePlaneBits(family.codebook_id);
            const auto shift = nativeQuantisedLayoutCodeBitShift(resident.metadata_layout);
            const auto planes = sb::planeBytes(family.codebook_id, family.N, resident.blocks_per_row, coded->superblock_phase);
            if (!resident.metadata_layout || nativeQuantisedLayoutPhase(resident.metadata_layout) != coded->superblock_phase ||
                resident.vnni_bytes != resident.allocationPayloadBytes() || resident.mins_bytes != planes.superblock ||
                resident.scales_bytes != (bits ? (bits + shift + 7u) / 8u : 0u) ||
                (!bits && resident.ptrs.d_scales != resident.ptrs.d_vnni))
                fail();
        }
        else if (resident.metadata_layout)
            fail();
        return {.spec = std::move(family),
                .slot = {.d_native_quantised_payload = resident.ptrs.d_vnni,
                         .d_native_quantised_scales = resident.ptrs.d_scales,
                         .d_native_quantised_mins = resident.ptrs.d_mins,
                         .d_native_quantised_emins = resident.ptrs.d_emins,
                         .payload_bytes = resident.allocationPayloadBytes(),
                         .scales_bytes = resident.scales_bytes,
                         .mins_bytes = resident.allocationMinsBytes(),
                         .emins_bytes = resident.allocationEminsBytes(),
                         .metadata_layout = resident.metadata_layout},
                .blocks_per_row = resident.blocks_per_row};
    }

    GpuExpertSlotPool::ProjectionSlot GpuExpertSlotPool::adoptResidentProjection(
        ProjectionSpec family, const ContiguousFloatingPointWeightDescriptor &resident)
    {
        if (!family.format.isFloating() || !resident.valid() || family.N != resident.n || family.K != resident.k ||
            family.format.floatingTensorType() != resident.type)
            throw std::invalid_argument("Initial GPU floating expert does not match its reusable slot family: " + family.label);
        return {.spec = std::move(family),
                .slot = {.d_native_quantised_payload = static_cast<uint8_t *>(const_cast<void *>(resident.data)),
                         .payload_bytes = resident.bytes}};
    }

    std::shared_ptr<GpuExpertSlotPool> GpuExpertSlotPool::create(
        IBackend *backend,
        DeviceId device,
        int device_ordinal,
        int layer_idx,
        int active_capacity,
        std::vector<ProjectionSpec> specs,
        std::shared_ptr<PhysicalMemoryAuthority> memory_authority,
        PhysicalMemoryOwner active_owner,
        int transfer_capacity)
    {
        return createImpl(
            backend,
            device,
            device_ordinal,
            layer_idx,
            active_capacity,
            std::move(specs),
            std::move(memory_authority),
            active_owner,
            transfer_capacity,
            /*explicit_test_allocation=*/false);
    }

    std::shared_ptr<GpuExpertSlotPool> GpuExpertSlotPool::createForTest(
        IBackend *backend,
        DeviceId device,
        int device_ordinal,
        int layer_idx,
        int active_capacity,
        std::vector<ProjectionSpec> specs,
        int transfer_capacity)
    {
        return createImpl(
            backend,
            device,
            device_ordinal,
            layer_idx,
            active_capacity,
            std::move(specs),
            nullptr,
            PhysicalMemoryOwner::RoutedExpertWeights,
            transfer_capacity,
            /*explicit_test_allocation=*/true);
    }

    std::shared_ptr<GpuExpertSlotPool> GpuExpertSlotPool::createImpl(
        IBackend *backend,
        DeviceId device,
        int device_ordinal,
        int layer_idx,
        int active_capacity,
        std::vector<ProjectionSpec> specs,
        std::shared_ptr<PhysicalMemoryAuthority> memory_authority,
        PhysicalMemoryOwner active_owner,
        int transfer_capacity,
        bool explicit_test_allocation)
    {
        if (active_capacity <= 0)
            throw std::invalid_argument("GpuExpertSlotPool active capacity must be positive");
        if (transfer_capacity < 0)
            throw std::invalid_argument("GpuExpertSlotPool transfer capacity cannot be negative");
        if (specs.empty())
            throw std::invalid_argument("GpuExpertSlotPool requires at least one projection spec");
        if (!explicit_test_allocation &&
            (!memory_authority ||
             (active_owner != PhysicalMemoryOwner::RoutedExpertWeights &&
              active_owner != PhysicalMemoryOwner::ExpertShadowSlots)))
        {
            throw std::invalid_argument(
                "GpuExpertSlotPool production allocation requires an authority and a routed-live or shadow owner");
        }

        auto orchestrator = explicit_test_allocation
                                ? std::make_shared<LoadOrchestrator>(
                                      backend,
                                      kTestOnlyUnadmittedGPUAllocation)
                                : std::make_shared<LoadOrchestrator>(
                                      backend,
                                      std::move(memory_authority),
                                      active_owner);
        orchestrator->addDevice(device_ordinal);

        auto plan_slot_family = [&](int count,
                                    const auto &slot_name_for,
                                    PhysicalMemoryOwner owner)
        {
            for (int slot = 0; slot < count; ++slot)
            {
                for (const auto &spec : specs)
                {
                    if (spec.N <= 0 || spec.K <= 0 || !spec.format.valid())
                        throw std::invalid_argument("GpuExpertSlotPool projection spec has invalid shape or format");
                    if (spec.format.isFloating())
                    {
                        const auto elements = static_cast<std::size_t>(spec.N) *
                                              static_cast<std::size_t>(spec.K);
                        const auto element_bytes =
                            spec.format.floatingElementBytes();
                        if (element_bytes == 0 ||
                            elements >
                                std::numeric_limits<std::size_t>::max() /
                                    element_bytes)
                        {
                            throw std::invalid_argument(
                                "GpuExpertSlotPool floating projection size overflows");
                        }
                        orchestrator->planRawWeightForOwner(
                            device_ordinal,
                            slot_name_for(slot, spec.label),
                            spec.N,
                            spec.K,
                            elements * element_bytes,
                            owner);
                    }
                    else
                    {
                        if ((spec.K % 32) != 0 ||
                            spec.payload_bytes_per_block <= 0)
                        {
                            throw std::invalid_argument(
                                "GpuExpertSlotPool NativeQuantised projection spec has invalid block geometry");
                        }
                        const auto *format = native_quantised_formats::forSourceIdentity(
                            spec.format.native_quantised.codebook_id, spec.format.native_quantised.is_superblock);
                        const auto coded = spec.codedDeclaration();
                        if (coded && !gpuPreparedSuperblockCoded(*format, GPUPreparedWeightPoolLayout::MigrationReusable, device))
                            throw std::invalid_argument("GPU expert slot backend does not support coded migration storage");
                        orchestrator->planWeightForOwner(
                            device_ordinal,
                            slot_name_for(slot, spec.label),
                            spec.N,
                            spec.K,
                            coded ? format->payload_bytes : spec.payload_bytes_per_block,
                            spec.is_asymmetric,
                            spec.has_emins,
                            spec.scale_plane,
                            /*raw_gguf_bytes=*/0,
                            owner, {}, coded);
                    }
                }
            }
        };

        plan_slot_family(
            active_capacity,
            [](int slot, const std::string &label)
            {
                return activeSlotName(slot, label);
            },
            active_owner);
        plan_slot_family(
            transfer_capacity,
            [](int slot, const std::string &label)
            {
                return transferSlotName(slot, label);
            },
            PhysicalMemoryOwner::ExpertMigrationStaging);

        orchestrator->allocate(/*pinned_slot_size=*/0, /*num_h2d_streams=*/0);

        auto pool = std::shared_ptr<GpuExpertSlotPool>(new GpuExpertSlotPool(
            backend,
            device,
            device_ordinal,
            layer_idx,
            active_capacity,
            transfer_capacity,
            std::move(specs),
            std::move(orchestrator)));

        logVramBomLine(
            "moe_gpu_expert_slot_pool",
                "backend=" + std::string(backend ? backend->backendName() : "none") +
                    " device=" + device.to_string() +
                    " device_id=" + std::to_string(device_ordinal) +
                    " layer=" + std::to_string(layer_idx) +
                    " active_capacity=" + std::to_string(active_capacity) +
                    " transfer_capacity=" + std::to_string(transfer_capacity) +
                    " total_slots=" + std::to_string(active_capacity + transfer_capacity) +
                    " projections_per_slot=" + std::to_string(pool->specs_.size()) +
                    " planned_bytes=" + std::to_string(pool->orchestrator_->getPool(device_ordinal)->totalPlannedBytes()) +
                    " planned_mib=" + vramBomMiB(pool->orchestrator_->getPool(device_ordinal)->totalPlannedBytes()));

        return pool;
    }

    int GpuExpertSlotPool::recommendedCapacity(int num_experts, size_t arrival_batch_size)
    {
        const int ten_percent = std::max(1, (num_experts + 9) / 10);
        const int churn_headroom = ten_percent + std::max(1, (ten_percent + 1) / 2);
        const int batch = static_cast<int>(std::min<size_t>(
            arrival_batch_size,
            static_cast<size_t>(std::max(0, num_experts))));
        return std::min(std::max(0, num_experts), std::max(churn_headroom, batch));
    }

    int GpuExpertSlotPool::recommendedTransferCapacity(int num_experts, size_t arrival_batch_size)
    {
        if (num_experts <= 0)
            return 0;
        const int batch = static_cast<int>(std::min<size_t>(
            std::max<size_t>(1, arrival_batch_size),
            static_cast<size_t>(num_experts)));
        // Staging can arrive in multiple source-stage groups before a single
        // publish consumes the transfer slots, so reserve the same bounded churn
        // headroom as active slots instead of sizing to only the first subbatch.
        return std::min(
            std::max(0, num_experts),
            std::max(batch, recommendedCapacity(num_experts, arrival_batch_size)));
    }

    GpuExpertSlotPool::GpuExpertSlotPool(IBackend *backend,
                                         DeviceId device,
                                         int device_ordinal,
                                         int layer_idx,
                                         int active_capacity,
                                         int transfer_capacity,
                                         std::vector<ProjectionSpec> specs,
                                         std::shared_ptr<LoadOrchestrator> orchestrator)
        : backend_(backend),
          device_(device),
          device_ordinal_(device_ordinal),
          layer_idx_(layer_idx),
          active_capacity_(active_capacity),
          transfer_capacity_(transfer_capacity),
          specs_(std::move(specs)),
          orchestrator_(std::move(orchestrator)),
          expert_by_slot_(static_cast<size_t>(active_capacity), -1),
          layer_by_slot_(static_cast<size_t>(active_capacity), -1),
          epoch_by_slot_(static_cast<size_t>(active_capacity), 0),
          expert_by_transfer_slot_(static_cast<size_t>(transfer_capacity), -1),
          epoch_by_transfer_slot_(static_cast<size_t>(transfer_capacity), 0)
    {
    }

    size_t GpuExpertSlotPool::SlotIdentityHash::operator()(
        const SlotIdentity &identity) const noexcept
    {
        /* Mix every coordinate because layers reuse one exact-geometry arena. */
        const size_t epoch_hash =
            std::hash<uint64_t>{}(identity.residency_epoch);
        const size_t expert_hash = std::hash<int>{}(identity.expert_id);
        const size_t layer_hash = std::hash<int>{}(identity.layer_idx);
        const size_t expert_epoch = epoch_hash ^
               (expert_hash + static_cast<size_t>(0x9e3779b9u) +
                (epoch_hash << 6u) + (epoch_hash >> 2u));
        return expert_epoch ^
               (layer_hash + static_cast<size_t>(0x9e3779b9u) +
                (expert_epoch << 6u) + (expert_epoch >> 2u));
    }

    std::optional<GpuExpertSlotPool::AcquiredSlot>
    GpuExpertSlotPool::acquire(int expert_id)
    {
        return acquire(expert_id, /*residency_epoch=*/0);
    }

    std::optional<GpuExpertSlotPool::AcquiredSlot>
    GpuExpertSlotPool::acquire(
        int expert_id,
        uint64_t residency_epoch)
    {
        return acquireForLayer(layer_idx_, expert_id, residency_epoch);
    }

    std::optional<GpuExpertSlotPool::AcquiredSlot>
    GpuExpertSlotPool::acquireForLayer(
        int layer_idx,
        int expert_id,
        uint64_t residency_epoch)
    {
        if (layer_idx < 0 || expert_id < 0)
            return std::nullopt;

        const SlotIdentity identity{layer_idx, expert_id, residency_epoch};
        int slot_index = -1;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (slot_by_identity_.count(identity) > 0)
                return std::nullopt;

            for (int i = 0; i < active_capacity_; ++i)
            {
                if (expert_by_slot_[static_cast<size_t>(i)] < 0)
                {
                    slot_index = i;
                    break;
                }
            }
            if (slot_index < 0)
                return std::nullopt;

            expert_by_slot_[static_cast<size_t>(slot_index)] = expert_id;
            layer_by_slot_[static_cast<size_t>(slot_index)] = layer_idx;
            epoch_by_slot_[static_cast<size_t>(slot_index)] = residency_epoch;
            slot_by_identity_[identity] = slot_index;
        }

        auto *pool =
            orchestrator_ ? orchestrator_->getPool(device_ordinal_) : nullptr;
        if (!pool)
        {
            releaseSlot(slot_index, identity);
            return std::nullopt;
        }

        AcquiredSlot acquired;
        acquired.slot_index = slot_index;
        acquired.layer_idx = layer_idx;
        acquired.expert_id = expert_id;
        acquired.residency_epoch = residency_epoch;
        acquired.projections.reserve(specs_.size());
        for (const auto &spec : specs_)
        {
            auto slot = pool->getSlot(activeSlotName(slot_index, spec.label));
            if (!slot)
            {
                releaseSlot(slot_index, identity);
                return std::nullopt;
            }

            ProjectionSlot projection;
            projection.spec = spec;
            projection.slot = *slot;
            projection.blocks_per_row = spec.format.isNativeQuantised()
                                            ? static_cast<uint32_t>(spec.K / 32)
                                            : 0;
            acquired.projections.push_back(std::move(projection));
        }

        auto token = std::make_shared<SlotLeaseToken>(SlotLeaseToken{
            slot_index,
            layer_idx,
            expert_id,
            residency_epoch,
            false,
            orchestrator_,
            weak_from_this()});
        /*
         * Capture the stored pointer before moving `token` into the deleter.
         * Constructor-argument evaluation order is not a lifetime contract;
         * reading token.get() in the other argument can otherwise observe the
         * moved-from shared_ptr and manufacture a false/null lease handle.
         */
        void *const token_pointer = token.get();
        acquired.lifetime = std::shared_ptr<void>(
            token_pointer,
            [token = std::move(token)](void *) mutable
            {
                if (auto pool = token->pool.lock())
                {
                    const SlotIdentity token_identity{
                        token->layer_idx,
                        token->expert_id,
                        token->residency_epoch};
                    if (token->transfer_slot)
                    {
                        pool->releaseTransferSlot(
                            token->slot_index, token_identity);
                    }
                    else
                    {
                        pool->releaseSlot(
                            token->slot_index, token_identity);
                    }
                }
            });

        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "gpu_direct_slot_pool_acquire",
            1.0,
            "rebalance",
            device_.to_string(),
            {{"layer", std::to_string(layer_idx)},
             {"slot", std::to_string(slot_index)},
             {"epoch", std::to_string(residency_epoch)}});
        return acquired;
    }

    std::optional<GpuExpertSlotPool::TransferSlot>
    GpuExpertSlotPool::acquireTransferSlot(int expert_id)
    {
        return acquireTransferSlot(expert_id, /*residency_epoch=*/0);
    }

    std::optional<GpuExpertSlotPool::TransferSlot>
    GpuExpertSlotPool::acquireTransferSlot(
        int expert_id,
        uint64_t residency_epoch)
    {
        if (expert_id < 0)
            return std::nullopt;

        const SlotIdentity identity{layer_idx_, expert_id, residency_epoch};
        int slot_index = -1;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (transfer_slot_by_identity_.count(identity) > 0)
                return std::nullopt;

            for (int i = 0; i < transfer_capacity_; ++i)
            {
                if (expert_by_transfer_slot_[static_cast<size_t>(i)] < 0)
                {
                    slot_index = i;
                    break;
                }
            }
            if (slot_index < 0)
                return std::nullopt;

            expert_by_transfer_slot_[static_cast<size_t>(slot_index)] =
                expert_id;
            epoch_by_transfer_slot_[static_cast<size_t>(slot_index)] =
                residency_epoch;
            transfer_slot_by_identity_[identity] = slot_index;
        }

        auto *pool =
            orchestrator_ ? orchestrator_->getPool(device_ordinal_) : nullptr;
        if (!pool)
        {
            releaseTransferSlot(slot_index, identity);
            return std::nullopt;
        }

        TransferSlot acquired;
        acquired.slot_index = slot_index;
        acquired.expert_id = expert_id;
        acquired.residency_epoch = residency_epoch;
        acquired.projections.reserve(specs_.size());
        for (const auto &spec : specs_)
        {
            auto slot = pool->getSlot(transferSlotName(slot_index, spec.label));
            if (!slot)
            {
                releaseTransferSlot(slot_index, identity);
                return std::nullopt;
            }

            ProjectionSlot projection;
            projection.spec = spec;
            projection.slot = *slot;
            projection.blocks_per_row = spec.format.isNativeQuantised()
                                            ? static_cast<uint32_t>(spec.K / 32)
                                            : 0;
            acquired.projections.push_back(std::move(projection));
        }

        auto token = std::make_shared<SlotLeaseToken>(SlotLeaseToken{
            slot_index,
            layer_idx_,
            expert_id,
            residency_epoch,
            true,
            orchestrator_,
            weak_from_this()});
        /* See acquire(): the stable stored pointer must precede token capture. */
        void *const token_pointer = token.get();
        acquired.lifetime = std::shared_ptr<void>(
            token_pointer,
            [token = std::move(token)](void *) mutable
            {
                if (auto pool = token->pool.lock())
                {
                    const SlotIdentity token_identity{
                        token->layer_idx,
                        token->expert_id,
                        token->residency_epoch};
                    if (token->transfer_slot)
                    {
                        pool->releaseTransferSlot(
                            token->slot_index, token_identity);
                    }
                    else
                    {
                        pool->releaseSlot(
                            token->slot_index, token_identity);
                    }
                }
            });

        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "gpu_direct_transfer_slot_acquire",
            1.0,
            "rebalance",
            device_.to_string(),
            {{"layer", std::to_string(layer_idx_)},
             {"slot", std::to_string(slot_index)},
             {"epoch", std::to_string(residency_epoch)}});
        return acquired;
    }

    size_t GpuExpertSlotPool::activeCapacity() const
    {
        return static_cast<size_t>(std::max(0, active_capacity_));
    }

    size_t GpuExpertSlotPool::transferCapacity() const
    {
        return static_cast<size_t>(std::max(0, transfer_capacity_));
    }

    size_t GpuExpertSlotPool::capacity() const
    {
        return activeCapacity();
    }

    size_t GpuExpertSlotPool::usedSlots() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return slot_by_identity_.size();
    }

    size_t GpuExpertSlotPool::usedTransferSlots() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return transfer_slot_by_identity_.size();
    }

    size_t GpuExpertSlotPool::availableSlots() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t capacity = activeCapacity();
        const size_t used = slot_by_identity_.size();
        return used < capacity ? capacity - used : 0;
    }

    size_t GpuExpertSlotPool::availableTransferSlots() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t capacity = transferCapacity();
        const size_t used = transfer_slot_by_identity_.size();
        return used < capacity ? capacity - used : 0;
    }

    std::optional<int> GpuExpertSlotPool::slotForExpert(int expert_id) const
    {
        return slotForExpert(expert_id, /*residency_epoch=*/0);
    }

    std::optional<int> GpuExpertSlotPool::slotForExpert(
        int expert_id,
        uint64_t residency_epoch) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = slot_by_identity_.find({layer_idx_, expert_id, residency_epoch});
        if (it == slot_by_identity_.end())
            return std::nullopt;
        return it->second;
    }

    std::optional<int> GpuExpertSlotPool::transferSlotForExpert(int expert_id) const
    {
        return transferSlotForExpert(expert_id, /*residency_epoch=*/0);
    }

    std::optional<int> GpuExpertSlotPool::transferSlotForExpert(
        int expert_id,
        uint64_t residency_epoch) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = transfer_slot_by_identity_.find(
            {layer_idx_, expert_id, residency_epoch});
        if (it == transfer_slot_by_identity_.end())
            return std::nullopt;
        return it->second;
    }

    std::string GpuExpertSlotPool::activeSlotName(int slot_index, const std::string &label)
    {
        return "expert_slot_" + std::to_string(slot_index) + "_" + label;
    }

    std::string GpuExpertSlotPool::transferSlotName(int slot_index, const std::string &label)
    {
        return "expert_transfer_slot_" + std::to_string(slot_index) + "_" + label;
    }

    void GpuExpertSlotPool::releaseSlot(
        int slot_index,
        SlotIdentity identity) noexcept
    {
        bool released = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (slot_index < 0 || slot_index >= active_capacity_)
                return;

            auto &assigned = expert_by_slot_[static_cast<size_t>(slot_index)];
            auto &assigned_layer =
                layer_by_slot_[static_cast<size_t>(slot_index)];
            auto &assigned_epoch =
                epoch_by_slot_[static_cast<size_t>(slot_index)];
            if (assigned_layer != identity.layer_idx ||
                assigned != identity.expert_id ||
                assigned_epoch != identity.residency_epoch)
                return;

            assigned = -1;
            assigned_layer = -1;
            assigned_epoch = 0;
            slot_by_identity_.erase(identity);
            released = true;
        }

        if (released)
        {
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "gpu_direct_slot_pool_release",
                1.0,
                "rebalance",
                device_.to_string(),
                {{"layer", std::to_string(identity.layer_idx)},
                 {"slot", std::to_string(slot_index)},
                 {"epoch", std::to_string(identity.residency_epoch)}});
        }
    }

    void GpuExpertSlotPool::releaseTransferSlot(
        int slot_index,
        SlotIdentity identity) noexcept
    {
        bool released = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (slot_index < 0 || slot_index >= transfer_capacity_)
                return;

            auto &assigned =
                expert_by_transfer_slot_[static_cast<size_t>(slot_index)];
            auto &assigned_epoch =
                epoch_by_transfer_slot_[static_cast<size_t>(slot_index)];
            if (assigned != identity.expert_id ||
                assigned_epoch != identity.residency_epoch)
                return;

            assigned = -1;
            assigned_epoch = 0;
            transfer_slot_by_identity_.erase(identity);
            released = true;
        }

        if (released)
        {
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "gpu_direct_transfer_slot_release",
                1.0,
                "rebalance",
                device_.to_string(),
                {{"layer", std::to_string(layer_idx_)},
                 {"slot", std::to_string(slot_index)},
                 {"epoch", std::to_string(identity.residency_epoch)}});
        }
    }

} // namespace llaminar2
