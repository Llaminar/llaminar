/**
 * @file CpuExpertSlotPool.cpp
 * @brief Stable CPU engine and NUMA-buffer ownership for overlay arrivals.
 */

#include "CpuExpertSlotPool.h"

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
        std::shared_ptr<TensorBase> floating_tensor;
        std::shared_ptr<ITensorGemm> engine;
    };

    /** @brief Persistent physical storage and current RCU logical identity. */
    struct CpuExpertSlotPool::Slot
    {
        std::vector<PreparedProjection> projections;
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
            int expert_id = -1;
            std::uint64_t residency_epoch = 0;
        };

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
            std::size_t allocation_alignment)
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
            packed.native_interleaved.resize_uninitialized_aligned(
                static_cast<std::size_t>(unit_count) *
                    static_cast<std::size_t>(
                        packed.interleaved_block_stride),
                allocation_alignment);
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
        std::shared_ptr<TensorBase> makeFloatingTensor(
            const CpuExpertSlotPool::ProjectionSpec &spec)
        {
            const std::vector<std::size_t> shape{
                static_cast<std::size_t>(spec.N),
                static_cast<std::size_t>(spec.K)};
            const auto type = spec.format.floatingTensorType();
            if (!type)
                throw std::invalid_argument(
                    "CPU floating expert slot lacks a floating tensor type");
            switch (*type)
            {
            case TensorType::FP16:
                return std::make_shared<FP16Tensor>(shape);
            case TensorType::BF16:
                return std::make_shared<BF16Tensor>(shape);
            case TensorType::FP32:
                return std::make_shared<FP32Tensor>(shape, DeviceId::cpu());
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
        Config config)
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

        std::vector<Slot> slots;
        slots.reserve(static_cast<std::size_t>(config.capacity));
        for (int slot_index = 0; slot_index < config.capacity; ++slot_index)
        {
            Slot slot;
            slot.projections.reserve(config.projections.size());
            for (const ProjectionSpec &spec : config.projections)
            {
                const std::size_t allocation_alignment =
                    config.memory_placement.requiresNodeBinding()
                        ? NUMAAllocator::externalRangeAlignment()
                        : AlignedVector<std::uint8_t>::ALIGNMENT;
                if (spec.format.isFloating())
                {
                    auto tensor = makeFloatingTensor(spec);
                    auto *destination = static_cast<std::uint8_t *>(
                        tensor->raw_mutable_data());
                    const std::size_t destination_bytes =
                        tensor->size_bytes();
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
                        gemm::FloatingPointGemmKernel>(tensor.get());
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
                        .floating_tensor = std::move(tensor),
                        .engine = std::move(engine),
                    });
                    continue;
                }
                CpuPacked packed = makePackedMetadata(
                    spec, allocation_alignment);
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
                auto engine = std::make_shared<CpuKernel>(std::move(packed));
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
                    .floating_tensor = nullptr,
                    .engine = std::move(engine),
                });
            }
            slots.push_back(std::move(slot));
        }

        auto pool = std::shared_ptr<CpuExpertSlotPool>(
            new CpuExpertSlotPool(std::move(config), std::move(slots)));
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
        std::vector<Slot> slots)
        : config_(std::move(config)), slots_(std::move(slots))
    {
    }

    std::optional<CpuExpertSlotPool::Lease> CpuExpertSlotPool::acquire(
        int expert_id,
        std::uint64_t residency_epoch)
    {
        if (expert_id < 0 || residency_epoch == 0)
            return std::nullopt;

        int selected = -1;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (std::size_t index = 0; index < slots_.size(); ++index)
            {
                const Slot &slot = slots_[index];
                if (slot.expert_id == expert_id &&
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
            slot.expert_id = expert_id;
            slot.residency_epoch = residency_epoch;
        }

        auto token = std::shared_ptr<CpuSlotLeaseToken>(
            new CpuSlotLeaseToken{
                .pool = shared_from_this(),
                .slot_index = selected,
                .expert_id = expert_id,
                .residency_epoch = residency_epoch,
            },
            [](CpuSlotLeaseToken *lease) noexcept
            {
                if (lease && lease->pool)
                {
                    lease->pool->release(
                        lease->slot_index,
                        lease->expert_id,
                        lease->residency_epoch);
                }
                delete lease;
            });

        Lease lease;
        lease.slot_index = selected;
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
             {"layer", std::to_string(config_.layer_idx)},
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
            if (slot.expert_id == expert_id &&
                slot.residency_epoch == residency_epoch)
            {
                return static_cast<int>(index);
            }
        }
        return std::nullopt;
    }

    void CpuExpertSlotPool::release(
        int slot_index,
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
            if (slot.expert_id != expert_id ||
                slot.residency_epoch != residency_epoch)
            {
                return;
            }
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
             {"layer", std::to_string(config_.layer_idx)},
             {"slot", std::to_string(slot_index)},
             {"epoch", std::to_string(residency_epoch)}});
    }

} // namespace llaminar2
