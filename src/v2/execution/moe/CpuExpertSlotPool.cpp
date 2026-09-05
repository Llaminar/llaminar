/**
 * @file CpuExpertSlotPool.cpp
 * @brief Stable CPU engine and NUMA-buffer ownership for overlay arrivals.
 */

#include "CpuExpertSlotPool.h"

#include "ExpertPreparedMemoryGeometry.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include "memory/NUMAAllocator.h"
#include "tensors/TensorClasses.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    using CpuKernel = cpu::native_vnni::CPUNativeVNNIGemmKernel;
    using CpuPacked = cpu::native_vnni::CPUNativeVNNIPackedWeights;
    using CpuEncoding = cpu::native_vnni::CPUNativeVNNIEncoding;

    /** @brief One stable engine plus the final bytes written by a GPU source. */
    struct CpuExpertSlotPool::PreparedProjection
    {
        ProjectionSpec spec;
        std::uint8_t *destination = nullptr;
        std::size_t destination_bytes = 0;
        std::size_t allocation_bytes = 0;
        std::shared_ptr<TensorBase> floating_tensor;
        std::shared_ptr<ITensorGemm> engine;
    };

    /** @brief Persistent physical storage and current RCU logical identity. */
    struct CpuExpertSlotPool::Slot
    {
        std::vector<PreparedProjection> projections;
        int layer_idx = -1;
        int expert_id = -1;
        std::uint64_t residency_epoch = 0;
    };

    namespace
    {
        /** Gate, up, and down are the complete routed-expert projection set. */
        inline constexpr std::size_t kCpuExpertProjectionCount = 3;

        /** @brief Control block shared by all gate/up/down engine aliases. */
        struct CpuSlotLeaseToken
        {
            std::shared_ptr<CpuExpertSlotPool> pool;
            int slot_index = -1;
            int layer_idx = -1;
            int expert_id = -1;
            std::uint64_t residency_epoch = 0;
        };

        /** Floating tensor together with its allocator-owned physical extent. */
        struct FloatingTensorAllocation
        {
            std::shared_ptr<TensorBase> tensor;
            std::size_t allocation_bytes = 0u;
        };

        /** @brief Sum exact CPU slot bytes without allowing integer wraparound. */
        std::size_t checkedByteAdd(
            std::size_t left,
            std::size_t right,
            const char *description)
        {
            if (right > std::numeric_limits<std::size_t>::max() - left)
            {
                throw std::overflow_error(
                    std::string(description) + " byte sum overflows size_t");
            }
            return left + right;
        }

        /** @brief Multiply exact CPU slot bytes without allowing integer wraparound. */
        std::size_t checkedByteMultiply(
            std::size_t left,
            std::size_t right,
            const char *description)
        {
            if (left != 0u &&
                right > std::numeric_limits<std::size_t>::max() / left)
            {
                throw std::overflow_error(
                    std::string(description) + " byte product overflows size_t");
            }
            return left * right;
        }

        /** @brief Validate one projection and return its catalogued source. */
        const NativeVnniFormatInfo *validateProjectionSpec(
            const CpuExpertSlotPool::ProjectionSpec &spec)
        {
            if (spec.N <= 0 || spec.K <= 0 || !spec.format.valid())
            {
                throw std::invalid_argument(
                    "CPU expert slot projection has invalid geometry or format identity");
            }
            if (spec.format.isFloating())
                return nullptr;
            if ((spec.K % 32) != 0)
                throw std::invalid_argument(
                    "CPU NativeVNNI expert slot projection K is not block aligned");
            const NativeVnniFormatInfo *source =
                native_vnni_formats::forSourceIdentity(
                    spec.format.native_vnni.codebook_id,
                    spec.format.native_vnni.is_superblock);
            if (!source)
            {
                throw std::invalid_argument(
                    "CPU expert slot projection source identity is not cataloged");
            }
            return source;
        }

        /** @brief Canonicalize and allocate one projection's final metadata. */
        CpuPacked makePackedMetadata(
            const CpuExpertSlotPool::ProjectionSpec &spec,
            std::size_t allocation_alignment,
            bool requires_dedicated_mapping)
        {
            const NativeVnniFormatInfo *source =
                validateProjectionSpec(spec);
            if (!source)
                throw std::invalid_argument(
                    "CPU NativeVNNI metadata requested for a floating projection");

            CpuPacked packed;
            packed.N = spec.N;
            packed.K = spec.K;
            packed.N_padded = (spec.N + 63) / 64 * 64;
            packed.blocks_per_row = spec.K / 32;
            packed.codebook_id = source->codebook_id;
            packed.is_asymmetric = source->is_asymmetric;
            /* Preserve provenance even when execution expands source blocks. */
            packed.is_superblock = source->is_superblock;
            packed.encoding = cpu::native_vnni::preparedEncodingForCodebook(
                source->codebook_id);
            packed.data_stride = static_cast<int>(
                cpu::native_vnni::preparedDataStride(packed.encoding));
            packed.interleaved_block_stride =
                cpu::native_vnni::preparedInterleavedBlockStride(
                    packed.encoding,
                    packed.is_asymmetric);

            switch (packed.encoding)
            {
            case CpuEncoding::NibbleLUT:
                packed.payload_bytes = 16;
                break;
            case CpuEncoding::ExpandedInt8:
                packed.payload_bytes = 32;
                break;
            case CpuEncoding::Q6KNativeDualScale:
                packed.payload_bytes = 24;
                break;
            }

            const std::uint64_t unit_count =
                static_cast<std::uint64_t>(packed.N_padded / 64) *
                static_cast<std::uint64_t>(packed.blocks_per_row);
            if (unit_count == 0 ||
                unit_count > std::numeric_limits<std::size_t>::max() /
                                 static_cast<std::uint64_t>(
                                     packed.interleaved_block_stride))
            {
                throw std::invalid_argument(
                    "CPU expert slot projection byte capacity overflows size_t");
            }
            const std::size_t destination_bytes =
                static_cast<std::size_t>(unit_count) *
                static_cast<std::size_t>(
                    packed.interleaved_block_stride);
            if (requires_dedicated_mapping)
            {
                packed.native_interleaved =
                    AlignedVector<std::uint8_t>::pageMappedUninitialized(
                        destination_bytes);
            }
            else
            {
                packed.native_interleaved.resize_uninitialized_aligned(
                    destination_bytes, allocation_alignment);
            }
            return packed;
        }

        /** @brief Convert a projection role to a compact diagnostic label. */
        const char *projectionName(
            ExpertTierWeightProjection projection) noexcept
        {
            switch (projection)
            {
            case ExpertTierWeightProjection::Gate:
                return "gate";
            case ExpertTierWeightProjection::Up:
                return "up";
            case ExpertTierWeightProjection::Down:
                return "down";
            }
            return "unknown";
        }

        /** @brief Require exactly one gate, up, and down projection. */
        void validateProjectionSet(
            const std::vector<CpuExpertSlotPool::ProjectionSpec> &specs)
        {
            if (specs.size() != kCpuExpertProjectionCount)
            {
                throw std::invalid_argument(
                    "CPU expert slot pool requires exactly gate/up/down projections");
            }
            std::array<bool, kCpuExpertProjectionCount> seen{};
            for (const auto &spec : specs)
            {
                const auto index = static_cast<std::size_t>(spec.projection);
                if (index >= seen.size() || seen[index])
                {
                    throw std::invalid_argument(
                        "CPU expert slot pool has a duplicate/unknown projection role");
                }
                seen[index] = true;
                (void)validateProjectionSpec(spec);
            }
            if (!std::all_of(seen.begin(), seen.end(), [](bool value)
                             { return value; }))
            {
                throw std::invalid_argument(
                    "CPU expert slot pool projection triplet is incomplete");
            }
        }

        /** @brief Allocate one final row-major floating tensor for a slot. */
        FloatingTensorAllocation makeFloatingTensor(
            const CpuExpertSlotPool::ProjectionSpec &spec,
            bool requires_dedicated_mapping)
        {
            const std::vector<std::size_t> shape{
                static_cast<std::size_t>(spec.N),
                static_cast<std::size_t>(spec.K)};
            const std::size_t elements = checkedByteMultiply(
                static_cast<std::size_t>(spec.N),
                static_cast<std::size_t>(spec.K),
                "CPU floating expert elements");
            const auto type = spec.format.floatingTensorType();
            if (!type)
                throw std::invalid_argument(
                    "CPU floating expert slot lacks a floating tensor type");
            constexpr std::size_t allocation_alignment =
                NUMAAllocator::externalRangeAlignment();
            switch (*type)
            {
            case TensorType::FP16:
            {
                AlignedVector<std::uint16_t> storage;
                if (requires_dedicated_mapping)
                {
                    storage = AlignedVector<std::uint16_t>::
                        pageMappedUninitialized(elements);
                }
                else
                {
                    storage.resize_uninitialized_aligned(
                        elements, allocation_alignment);
                }
                const std::size_t bytes = storage.allocationBytes();
                return {
                    .tensor = std::make_shared<FP16Tensor>(
                        shape, std::move(storage)),
                    .allocation_bytes = bytes,
                };
            }
            case TensorType::BF16:
            {
                AlignedVector<std::uint16_t> storage;
                if (requires_dedicated_mapping)
                {
                    storage = AlignedVector<std::uint16_t>::
                        pageMappedUninitialized(elements);
                }
                else
                {
                    storage.resize_uninitialized_aligned(
                        elements, allocation_alignment);
                }
                const std::size_t bytes = storage.allocationBytes();
                return {
                    .tensor = std::make_shared<BF16Tensor>(
                        shape, std::move(storage)),
                    .allocation_bytes = bytes,
                };
            }
            case TensorType::FP32:
            {
                AlignedVector<float> storage;
                if (requires_dedicated_mapping)
                {
                    storage = AlignedVector<float>::pageMappedUninitialized(
                        elements);
                }
                else
                {
                    storage.resize_uninitialized_aligned(
                        elements, allocation_alignment);
                }
                const std::size_t bytes = storage.allocationBytes();
                return {
                    .tensor = std::make_shared<FP32Tensor>(
                        shape, std::move(storage), DeviceId::cpu()),
                    .allocation_bytes = bytes,
                };
            }
            default:
                throw std::invalid_argument(
                    "CPU floating expert slot received an unsupported precision");
            }
        }
    } // namespace

    CpuExpertSlotPool::MemoryPlacement
    CpuExpertSlotPool::MemoryPlacement::boundNode(int node)
    {
        if (node < 0)
        {
            throw std::invalid_argument(
                "CPU expert slot NUMA node must be non-negative");
        }
        return MemoryPlacement(Scope::BoundNode, node);
    }

    std::shared_ptr<CpuExpertSlotPool> CpuExpertSlotPool::create(
        Config config,
        std::shared_ptr<PhysicalMemoryAuthority> memory_authority,
        PhysicalMemoryOwner owner)
    {
        return createImpl(
            std::move(config),
            std::move(memory_authority),
            owner,
            /*admitted=*/true);
    }

    std::shared_ptr<CpuExpertSlotPool>
    CpuExpertSlotPool::createForTest(Config config)
    {
        return createImpl(
            std::move(config),
            nullptr,
            PhysicalMemoryOwner::Count,
            /*admitted=*/false);
    }

    std::shared_ptr<CpuExpertSlotPool> CpuExpertSlotPool::createImpl(
        Config config,
        std::shared_ptr<PhysicalMemoryAuthority> memory_authority,
        PhysicalMemoryOwner owner,
        bool admitted)
    {
        if (config.participant_id < 0 || config.layer_idx < 0 ||
            config.capacity <= 0)
        {
            throw std::invalid_argument(
                "CPU expert slot pool requires valid endpoint identity and capacity");
        }
        validateProjectionSet(config.projections);
        if (config.perf_device.empty())
            config.perf_device = "CPU";

        std::size_t bytes_per_slot = 0u;
        for (const ProjectionSpec &spec : config.projections)
        {
            bytes_per_slot = checkedByteAdd(
                bytes_per_slot,
                resolveExpertPreparedProjectionMemoryGeometry(
                    spec.N, spec.K, spec.format)
                    .cpu_bytes,
                "CPU expert slot");
        }
        const std::size_t planned_allocation_bytes = checkedByteMultiply(
            bytes_per_slot,
            static_cast<std::size_t>(config.capacity),
            "CPU expert slot pool");
        std::optional<PhysicalMemoryAllocationLease> memory_lease;
        if (admitted)
        {
            if (!memory_authority ||
                (owner != PhysicalMemoryOwner::ExpertShadowSlots &&
                 owner != PhysicalMemoryOwner::RoutedExpertWeights) ||
                !memory_authority->contains(DeviceId::cpu()))
            {
                throw std::invalid_argument(
                    "CPU expert slot pool requires a rank-local CPU memory authority and a live/shadow expert owner");
            }
            memory_lease.emplace(memory_authority->claimNewAllocation(
                DeviceId::cpu(), owner, planned_allocation_bytes));
        }

        std::vector<Slot> slots;
        slots.reserve(static_cast<std::size_t>(config.capacity));
        std::size_t materialized_allocation_bytes = 0u;
        for (int slot_index = 0; slot_index < config.capacity; ++slot_index)
        {
            Slot slot;
            slot.projections.reserve(config.projections.size());
            for (const ProjectionSpec &spec : config.projections)
            {
                const bool requires_dedicated_mapping =
                    config.memory_placement.requiresNodeBinding();
                const std::size_t allocation_alignment =
                    NUMAAllocator::externalRangeAlignment();
                if (spec.format.isFloating())
                {
                    auto allocation = makeFloatingTensor(
                        spec, requires_dedicated_mapping);
                    auto *destination = static_cast<std::uint8_t *>(
                        allocation.tensor->raw_mutable_data());
                    const std::size_t destination_bytes =
                        allocation.tensor->size_bytes();
                    if (!destination || destination_bytes == 0)
                    {
                        throw std::runtime_error(
                            "CPU floating expert slot did not allocate final row-major storage");
                    }
                    if (config.memory_placement.requiresNodeBinding() &&
                        !NUMAAllocator::instance()
                             .prepareExternalReceiveRangeOnNode(
                                 destination,
                                 destination_bytes,
                                 config.memory_placement.node()))
                    {
                        throw std::runtime_error(
                            "CPU floating expert slot could not bind " +
                            std::string(projectionName(spec.projection)) +
                            " bytes to NUMA node " +
                            std::to_string(
                                config.memory_placement.node()));
                    }
                    auto engine = std::make_shared<
                        gemm::FloatingPointGemmKernel>(
                            allocation.tensor.get(),
                            gemm::FloatingPointGemmKernel::NumericalPolicy::
                                GPUAlignedExpert);
                    ContiguousFloatingPointWeightDescriptor descriptor;
                    if (!engine->exportContiguousFloatingPointWeights(
                            descriptor) ||
                        descriptor.data != destination ||
                        descriptor.bytes != destination_bytes)
                    {
                        throw std::runtime_error(
                            "CPU floating expert engine did not retain its preallocated final buffer");
                    }
                    slot.projections.push_back({
                        .spec = spec,
                        .destination = destination,
                        .destination_bytes = destination_bytes,
                        .allocation_bytes = allocation.allocation_bytes,
                        .floating_tensor = std::move(allocation.tensor),
                        .engine = std::move(engine),
                    });
                    materialized_allocation_bytes = checkedByteAdd(
                        materialized_allocation_bytes,
                        allocation.allocation_bytes,
                        "CPU floating expert pool");
                    continue;
                }
                CpuPacked packed = makePackedMetadata(
                    spec,
                    allocation_alignment,
                    requires_dedicated_mapping);
                if (config.memory_placement.requiresNodeBinding() &&
                    !NUMAAllocator::instance().prepareExternalReceiveRangeOnNode(
                        packed.native_interleaved.data(),
                        packed.native_interleaved.size(),
                        config.memory_placement.node()))
                {
                    throw std::runtime_error(
                        "CPU expert slot pool could not bind " +
                        std::string(projectionName(spec.projection)) +
                        " final bytes to NUMA node " +
                        std::to_string(config.memory_placement.node()));
                }

                /* Moving the vector into the kernel preserves its stable address. */
                std::uint8_t *destination = packed.native_interleaved.data();
                const std::size_t destination_bytes =
                    packed.native_interleaved.size();
                const std::size_t allocation_bytes =
                    packed.native_interleaved.allocationBytes();
                auto engine = std::make_shared<CpuKernel>(
                    std::move(packed),
                    CPUProjectionNumericalPolicy::GPUAlignedExpert);
                if (!engine->isValid() ||
                    engine->packedWeights().native_interleaved.data() !=
                        destination)
                {
                    throw std::runtime_error(
                        "CPU expert slot engine did not retain its preallocated final buffer");
                }
                slot.projections.push_back({
                    .spec = spec,
                    .destination = destination,
                    .destination_bytes = destination_bytes,
                    .allocation_bytes = allocation_bytes,
                    .floating_tensor = nullptr,
                    .engine = std::move(engine),
                });
                materialized_allocation_bytes = checkedByteAdd(
                    materialized_allocation_bytes,
                    allocation_bytes,
                    "CPU NativeVNNI expert pool");
            }
            slots.push_back(std::move(slot));
        }

        if (materialized_allocation_bytes != planned_allocation_bytes)
        {
            throw std::logic_error(
                "CPU expert slot pool materialized bytes differ from canonical projection geometry");
        }

        auto pool = std::shared_ptr<CpuExpertSlotPool>(
            new CpuExpertSlotPool(
                std::move(config),
                std::move(slots),
                std::move(memory_lease),
                materialized_allocation_bytes));
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "cpu_inactive_slots_materialized",
            static_cast<double>(pool->capacity()),
            "model_setup",
            pool->config_.perf_device,
            {{"participant", std::to_string(pool->config_.participant_id)},
             {"layer", std::to_string(pool->config_.layer_idx)},
             {"numa_node", std::to_string(pool->config_.memory_placement.node())}});
        return pool;
    }

    CpuExpertSlotPool::CpuExpertSlotPool(
        Config config,
        std::vector<Slot> slots,
        std::optional<PhysicalMemoryAllocationLease> memory_lease,
        std::size_t allocation_bytes)
        : config_(std::move(config)),
          memory_lease_(std::move(memory_lease)),
          slots_(std::move(slots)),
          allocation_bytes_(allocation_bytes)
    {
    }

    std::optional<CpuExpertSlotPool::Lease> CpuExpertSlotPool::acquire(
        int expert_id,
        std::uint64_t residency_epoch)
    {
        return acquireForLayer(
            config_.layer_idx, expert_id, residency_epoch);
    }

    std::optional<CpuExpertSlotPool::Lease>
    CpuExpertSlotPool::acquireForLayer(
        int layer_idx,
        int expert_id,
        std::uint64_t residency_epoch)
    {
        if (layer_idx < 0 || expert_id < 0 || residency_epoch == 0)
            return std::nullopt;

        int selected = -1;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (std::size_t index = 0; index < slots_.size(); ++index)
            {
                const Slot &slot = slots_[index];
                if (slot.layer_idx == layer_idx &&
                    slot.expert_id == expert_id &&
                    slot.residency_epoch == residency_epoch)
                {
                    return std::nullopt;
                }
                if (selected < 0 && slot.expert_id < 0)
                    selected = static_cast<int>(index);
            }
            if (selected < 0)
                return std::nullopt;
            Slot &slot = slots_[static_cast<std::size_t>(selected)];
            slot.layer_idx = layer_idx;
            slot.expert_id = expert_id;
            slot.residency_epoch = residency_epoch;
        }

        auto token = std::shared_ptr<CpuSlotLeaseToken>(
            new CpuSlotLeaseToken{
                .pool = shared_from_this(),
                .slot_index = selected,
                .layer_idx = layer_idx,
                .expert_id = expert_id,
                .residency_epoch = residency_epoch,
            },
            [](CpuSlotLeaseToken *lease) noexcept
            {
                if (lease && lease->pool)
                {
                    lease->pool->release(
                        lease->slot_index,
                        lease->layer_idx,
                        lease->expert_id,
                        lease->residency_epoch);
                }
                delete lease;
            });

        Lease lease;
        lease.slot_index = selected;
        lease.layer_idx = layer_idx;
        lease.expert_id = expert_id;
        lease.residency_epoch = residency_epoch;
        lease.lifetime = token;
        const Slot &slot = slots_[static_cast<std::size_t>(selected)];
        lease.projections.reserve(slot.projections.size());
        for (const PreparedProjection &projection : slot.projections)
        {
            /*
             * Aliasing makes the epoch token, rather than the pool's permanent
             * engine owner, the public lifetime. The slot cannot be recycled
             * until every bank/operation alias for all three projections dies.
             */
            std::shared_ptr<ITensorGemm> engine(
                token,
                static_cast<ITensorGemm *>(projection.engine.get()));
            lease.projections.push_back({
                .projection = projection.spec.projection,
                .destination_bytes = std::span<std::uint8_t>(
                    projection.destination,
                    projection.destination_bytes),
                .engine = std::move(engine),
            });
        }

        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "cpu_inactive_slot_acquisitions",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"participant", std::to_string(config_.participant_id)},
             {"layer", std::to_string(layer_idx)},
             {"slot", std::to_string(selected)},
             {"epoch", std::to_string(residency_epoch)}});
        return lease;
    }

    std::size_t CpuExpertSlotPool::capacity() const noexcept
    {
        return slots_.size();
    }

    std::size_t CpuExpertSlotPool::usedSlots() const noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<std::size_t>(std::count_if(
            slots_.begin(),
            slots_.end(),
            [](const Slot &slot) { return slot.expert_id >= 0; }));
    }

    std::size_t CpuExpertSlotPool::availableSlots() const noexcept
    {
        return capacity() - usedSlots();
    }

    std::optional<int> CpuExpertSlotPool::slotFor(
        int expert_id,
        std::uint64_t residency_epoch) const noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t index = 0; index < slots_.size(); ++index)
        {
            const Slot &slot = slots_[index];
            if (slot.layer_idx == config_.layer_idx &&
                slot.expert_id == expert_id &&
                slot.residency_epoch == residency_epoch)
            {
                return static_cast<int>(index);
            }
        }
        return std::nullopt;
    }

    void CpuExpertSlotPool::release(
        int slot_index,
        int layer_idx,
        int expert_id,
        std::uint64_t residency_epoch) noexcept
    {
        bool released = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (slot_index < 0 ||
                slot_index >= static_cast<int>(slots_.size()))
            {
                return;
            }
            Slot &slot = slots_[static_cast<std::size_t>(slot_index)];
            if (slot.layer_idx != layer_idx ||
                slot.expert_id != expert_id ||
                slot.residency_epoch != residency_epoch)
            {
                return;
            }
            slot.layer_idx = -1;
            slot.expert_id = -1;
            slot.residency_epoch = 0;
            released = true;
        }
        if (!released)
            return;

        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "cpu_inactive_slot_releases",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"participant", std::to_string(config_.participant_id)},
             {"layer", std::to_string(layer_idx)},
             {"slot", std::to_string(slot_index)},
             {"epoch", std::to_string(residency_epoch)}});
    }

} // namespace llaminar2
