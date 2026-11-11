/**
 * @file PooledImmutableReplicatedReaderGeometry.h
 * @brief Exact native result/ACK geometry for one shared replicated PLE transaction.
 *
 * Consumers are not storage owners or CPU service channels. Each foreign
 * consumer receives only its graph-required live FP32 result and returns a
 * fixed protocol ACK before the original pin authority may retire readers.
 * This metadata grants no topology, source or runtime materialization authority.
 */
#pragma once
#include "kernels/qwen4exp/PooledImmutableReplicatedReaderOps.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "loaders/ImmutableRowTableLayoutGeometry.h"
#include <algorithm>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace llaminar2::qwen4exp {

/** @brief Immutable original compute-reader order around one source coordinator. */
class PooledImmutableReplicatedReaderGeometry final {
public:
    /**
     * @brief Check the actual declared reader coordinates without querying hardware.
     * @param coordinator Original source device, which is itself exactly one reader.
     * @param consumers Original Rank/placement order, independent of storage coordinates.
     * @param maximumOriginal Original replicated sixteen-head query envelope, not a summed union.
     * @param rowElements Required FP32 output width per original hash ID.
     * @return Checked result/ACK shape; the runtime still requires original inventory/permits.
     */
private:
    static PooledImmutableReplicatedReaderGeometry checked(
        PhysicalMemoryAllocatorIdentity coordinator,
        std::span<const PhysicalMemoryAllocatorIdentity> consumers,
        std::uint32_t maximumOriginal, std::uint32_t rowElements)
    {
        if (consumers.size() < 2 || consumers.size() > std::numeric_limits<std::uint32_t>::max() ||
            coordinator.world_rank < 0 || !coordinator.device.is_valid() || !coordinator.device.is_gpu() ||
            !replicated_reader::valid({0, static_cast<std::uint32_t>(consumers.size()), maximumOriginal, rowElements}))
            throw std::invalid_argument("Shared PLE readers require complete replicated query/output geometry");
        std::size_t sources = 0;
        for (std::size_t i = 0; i < consumers.size(); ++i) {
            const auto consumer = consumers[i];
            if (consumer.world_rank != coordinator.world_rank || !consumer.device.is_valid() ||
                consumer.device.type != coordinator.device.type ||
                std::find(consumers.begin(), consumers.begin() + i, consumer) != consumers.begin() + i)
                throw std::invalid_argument("Shared PLE readers require unique original homogeneous local consumers");
            sources += consumer == coordinator;
        }
        if (sources != 1)
            throw std::invalid_argument("Shared PLE source must be one exact original compute reader");
        return {coordinator,{consumers.begin(),consumers.end()},maximumOriginal,rowElements};
    }
public:
    /** @brief Declare native homogeneous endpoints before BOM, independently of hardware P2P coverage. */
    static PooledImmutableReplicatedReaderGeometry native(
        PhysicalMemoryAllocatorIdentity coordinator, std::span<const PhysicalMemoryAllocatorIdentity> consumers,
        std::uint32_t maximumOriginal, std::uint32_t rowElements)
    {
        return checked(coordinator, consumers, maximumOriginal, rowElements);
    }
    /** @return True only for original NCCL/RCCL operations; this metadata grants no communicator readiness. */
    bool nativeCollective() const noexcept { return true; }
    /** @return Original source, never the first reader selected by enumeration. */
    PhysicalMemoryAllocatorIdentity coordinator() const noexcept { return coordinator_; }
    /** @return Exact declared reader order; storage-only GPUs are absent. */
    std::span<const PhysicalMemoryAllocatorIdentity> consumers() const noexcept { return consumers_; }
    /** @return Original single replicated query envelope. */
    std::uint32_t maximumOriginal() const noexcept { return maximum_original_; }
    /** @return Exact required FP32 elements for one original ID. */
    std::uint32_t rowElements() const noexcept { return row_elements_; }
    /** @return Maximum admitted result; never the runtime transfer extent. */
    std::size_t resultMaximum() const
    {
        const auto bytes = replicated_reader::outputBytes(
            {0, static_cast<std::uint32_t>(consumers_.size()), maximum_original_, row_elements_}, maximum_original_);
        if (bytes > std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("Shared PLE result envelope exceeds native size_t");
        return static_cast<std::size_t>(bytes);
    }
    /**
     * @brief Contribute exact channel storage to the original PMA before optional quotas.
     * @param host Original host resource identity; its native completion owner prices the operand backing separately.
     * @param devices Complete original GPU observations; no runtime free-memory query occurs.
     * @return Source/reader protocol controls only; native completion/payload roots remain separately owned.
     */
    PhysicalMemoryPlan completeBOM(PhysicalMemoryResource host,
        std::span<const PhysicalMemoryResource> devices) const
    {
        if (host.world_rank != coordinator_.world_rank || host.device != DeviceId::cpu())
            throw std::invalid_argument("Shared PLE reader channels require their original CPU0 allocator");
        const auto resource = [&](PhysicalMemoryAllocatorIdentity allocator) {
            const PhysicalMemoryResource* found = nullptr;
            for (const auto& candidate : devices) {
                if (candidate.world_rank != allocator.world_rank || candidate.device != allocator.device) continue;
                if (found) throw std::invalid_argument("Duplicate original shared PLE reader resource");
                found = &candidate;
            }
            if (!found) throw std::invalid_argument("Missing original shared PLE reader resource");
            return *found;
        };
        const auto source = resource(coordinator_);
        PhysicalMemoryPlanBuilder result;
        for (const auto consumer : consumers_) {
            const auto endpoint = resource(consumer);
            if (consumer == coordinator_) continue;
            result.add(source, PhysicalMemoryOwner::ActivationTransportStaging, sizeof(replicated_reader::Control));
            result.add(endpoint, PhysicalMemoryOwner::ActivationTransportStaging, sizeof(replicated_reader::Control));

        }
        return result.build();
    }
private:
    /** @brief Publish only complete immutable metadata after all coordinates/overflow checks. */
    PooledImmutableReplicatedReaderGeometry(PhysicalMemoryAllocatorIdentity coordinator,
        std::vector<PhysicalMemoryAllocatorIdentity> consumers, std::uint32_t maximumOriginal,
        std::uint32_t rowElements)
        : coordinator_(coordinator), consumers_(std::move(consumers)),
          maximum_original_(maximumOriginal), row_elements_(rowElements) {}
    PhysicalMemoryAllocatorIdentity coordinator_;
    std::vector<PhysicalMemoryAllocatorIdentity> consumers_;
    std::uint32_t maximum_original_, row_elements_;
};
} // namespace llaminar2::qwen4exp
