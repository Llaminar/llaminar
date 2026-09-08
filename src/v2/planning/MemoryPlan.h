/**
 * @file MemoryPlan.h
 * @brief Final and incremental device-memory admission result types.
 *
 * A plan reports the complete post-initialization footprint while separately
 * tracking bytes that are already resident and reflected in the device's live
 * free-memory reading. This distinction lets a fresh graph adopt immutable
 * prepared weights without either hiding them from the BOM or charging their
 * allocation twice.
 */

#pragma once
#include "backends/DeviceId.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <vector>
#include <string>
#include <cstddef>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <iomanip>

namespace llaminar2
{

/**
 * @brief One model geometry bound to one canonical physical-memory BOM.
 *
 * Named accessors are presentation views over @ref bom; they never store an
 * independently mutable subtotal.  This makes MemoryPlanner, ExpertOverlay,
 * and preflight share the same owner arithmetic while retaining useful
 * component-level diagnostics.
 */
class DeviceMemoryPlan final
{
public:
    /** @brief Bind immutable geometry to its complete physical-memory BOM. */
    DeviceMemoryPlan(
        PhysicalMemoryBOM bom,
        int maximum_sequence_length,
        int activation_sequence_length)
        : bom_(std::move(bom)),
          max_seq_len_(maximum_sequence_length),
          activation_seq_len_(activation_sequence_length)
    {
    }

    /** @return Physical device that owns this BOM. */
    [[nodiscard]] DeviceId device() const noexcept
    {
        return bom_.resource().device;
    }

    /** @return Model context capacity used by cache/state estimators. */
    [[nodiscard]] int max_seq_len() const noexcept { return max_seq_len_; }

    /** @return Resident row capacity used by graph/arena estimators. */
    [[nodiscard]] int activation_seq_len() const noexcept
    {
        return activation_seq_len_;
    }

    /** @return Sole immutable per-owner byte authority. */
    [[nodiscard]] const PhysicalMemoryBOM &bom() const noexcept
    {
        return bom_;
    }

    /**
     * @brief Return a new plan with one additional typed physical owner.
     *
     * The original plan remains immutable. This is used when a graph running
     * on one device owns a side allocation in another already-planned physical
     * resource; it never exposes an untyped subtotal setter.
     */
    [[nodiscard]] DeviceMemoryPlan withAdditionalCharge(
        PhysicalMemoryOwner owner,
        size_t planned_bytes,
        size_t already_resident_bytes = 0) const
    {
        PhysicalMemoryBOMBuilder builder(bom_);
        builder.add(owner, planned_bytes, already_resident_bytes);
        return DeviceMemoryPlan(
            builder.build(), max_seq_len_, activation_seq_len_);
    }

    /** @return Primary graph-view model weights. */
    [[nodiscard]] size_t weight_bytes() const
    {
        return bom_.bytes(PhysicalMemoryOwner::PrimaryModelWeights);
    }

    /** @return Concurrent replicated or mirrored model-weight views. */
    [[nodiscard]] size_t additional_weight_bytes() const
    {
        return bom_.bytes(PhysicalMemoryOwner::AdditionalModelWeights);
    }

    /** @return Certified already-resident bytes from both weight owners. */
    [[nodiscard]] size_t retained_weight_bytes() const
    {
        return bom_.alreadyResidentBytes(
                   PhysicalMemoryOwner::PrimaryModelWeights) +
               bom_.alreadyResidentBytes(
                   PhysicalMemoryOwner::AdditionalModelWeights);
    }

    [[nodiscard]] size_t kv_cache_bytes() const
    {
        return bom_.bytes(PhysicalMemoryOwner::KVCache);
    }

    [[nodiscard]] size_t live_recurrent_state_bytes() const
    {
        return bom_.bytes(PhysicalMemoryOwner::RecurrentLiveState);
    }

    [[nodiscard]] size_t checkpoint_state_bytes() const
    {
        return bom_.bytes(PhysicalMemoryOwner::RecurrentCheckpointState);
    }

    /** @return Persistent request-position and sequence-table storage. */
    [[nodiscard]] size_t sequence_metadata_bytes() const
    {
        return bom_.bytes(PhysicalMemoryOwner::SequenceMetadata);
    }

    [[nodiscard]] size_t persistent_state_bytes() const
    {
        return live_recurrent_state_bytes() + checkpoint_state_bytes() +
               sequence_metadata_bytes();
    }

    [[nodiscard]] size_t prefix_cache_staging_bytes() const
    {
        return bom_.bytes(PhysicalMemoryOwner::PrefixArchiveStaging);
    }

    [[nodiscard]] size_t prefix_cache_device_hot_bytes() const
    {
        return bom_.bytes(PhysicalMemoryOwner::PrefixDeviceTier);
    }

    [[nodiscard]] size_t captured_graph_bytes() const
    {
        return bom_.bytes(PhysicalMemoryOwner::NativeGraphExecutable);
    }

    /** @return Device bytes reserved for graph-resident diagnostic checkpoints. */
    [[nodiscard]] size_t graph_snapshot_bytes() const
    {
        return bom_.bytes(PhysicalMemoryOwner::GraphSnapshotArena);
    }

    [[nodiscard]] size_t collective_bytes() const
    {
        return bom_.bytes(PhysicalMemoryOwner::LocalCollective);
    }

    [[nodiscard]] size_t activation_bytes() const
    {
        return bom_.bytes(PhysicalMemoryOwner::ActivationArena);
    }

    [[nodiscard]] size_t workspace_bytes() const
    {
        return bom_.bytes(PhysicalMemoryOwner::ExecutionWorkspace);
    }

    [[nodiscard]] size_t retained_workspace_bytes() const
    {
        return bom_.alreadyResidentBytes(
            PhysicalMemoryOwner::ExecutionWorkspace);
    }

    [[nodiscard]] size_t device_total_bytes() const
    {
        return bom_.resource().total_bytes;
    }

    [[nodiscard]] size_t device_free_bytes() const
    {
        return bom_.resource().admission_available_bytes;
    }

    [[nodiscard]] size_t total_bytes() const
    {
        return bom_.totalBytes();
    }

    /** @return Complete persistent weight footprint across every physical view. */
    size_t total_weight_bytes() const
    {
        return weight_bytes() + additional_weight_bytes();
    }

    /** @return Weight bytes this runner must newly allocate. */
    size_t incremental_weight_bytes() const
    {
        const size_t total = total_weight_bytes();
        return total - std::min(total, retained_weight_bytes());
    }

    /**
     * @return Bytes newly required from the currently free device capacity.
     *
     * Retained weights remain part of total_bytes(), but the device allocator
     * has already removed them from device_free_bytes.
     */
    size_t incremental_bytes() const
    {
        return bom_.incrementalBytes();
    }

    bool fits() const
    {
        return bom_.fits();
    }

    size_t deficit() const
    {
        return bom_.deficitBytes();
    }

    size_t remaining() const
    {
        return bom_.remainingBytes();
    }

    std::string summary() const
    {
        auto mb = [](size_t b) { return static_cast<double>(b) / (1024.0 * 1024.0); };
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(0);
        ss << device().to_string() << ": "
           << "weights=" << mb(weight_bytes()) << " MB, "
           << "additional_weights=" << mb(additional_weight_bytes()) << " MB, "
           << "retained_weights=" << mb(retained_weight_bytes()) << " MB, "
           << "kv_cache=" << mb(kv_cache_bytes()) << " MB, "
           << "state=" << mb(persistent_state_bytes()) << " MB, "
           << "prefix_staging=" << mb(prefix_cache_staging_bytes()) << " MB, "
           << "prefix_hot=" << mb(prefix_cache_device_hot_bytes()) << " MB, "
           << "captured_graphs=" << mb(captured_graph_bytes()) << " MB, "
           << "graph_snapshots=" << mb(graph_snapshot_bytes()) << " MB, "
           << "collective=" << mb(collective_bytes()) << " MB, "
           << "activations=" << mb(activation_bytes()) << " MB, "
           << "workspace=" << mb(workspace_bytes()) << " MB, "
           << "retained_workspace=" << mb(retained_workspace_bytes()) << " MB, "
           << "total=" << mb(total_bytes()) << " MB, "
           << "new=" << mb(incremental_bytes()) << "/" << mb(device_free_bytes()) << " MB"
           << (fits() ? " [OK]" : " [OVER by " + std::to_string(static_cast<int>(mb(deficit()))) + " MB]");
        return ss.str();
    }

private:
    PhysicalMemoryBOM bom_;
    int max_seq_len_ = 0;
    int activation_seq_len_ = 0;
};

class MemoryPlan final
{
public:
    std::vector<DeviceMemoryPlan> devices;
    std::vector<std::string> diagnostics;  // Warnings/errors

    /** @return Whether this plan was sealed and every allocator admits it. */
    [[nodiscard]] bool fits() const
    {
        return physical_plan_ && physical_plan_->fits();
    }

    /** @return Complete topology footprint from the canonical aggregate. */
    [[nodiscard]] size_t total_bytes() const
    {
        return physicalPlan().totalBytes();
    }

    /** @return Sum of independently observed allocator capacity. */
    [[nodiscard]] size_t total_available() const
    {
        return physicalPlan().admissionAvailableBytes();
    }

    /** @return Sole immutable CPU/GPU plan behind the diagnostic views. */
    [[nodiscard]] const PhysicalMemoryPlan &physicalPlan() const
    {
        if (!physical_plan_)
        {
            throw std::logic_error(
                "MemoryPlan has not entered its sealed physical-plan state");
        }
        return *physical_plan_;
    }

    /**
     * @brief Enter the one legal fit-to-allocation transition.
     * @return Immutable topology-wide admission proof.
     * @throws std::invalid_argument when any physical allocator is over budget.
     */
    [[nodiscard]] PhysicalMemoryPlanAdmissionCertificate admit() const
    {
        return PhysicalMemoryPlanAdmissionCertificate(physicalPlan());
    }

    /// Render a libfort-formatted table. Declared here, defined in MemoryPlanner.cpp.
    std::string renderTable() const;

private:
    friend class MemoryPlanner;

    /** @brief Freeze all presentation rows into one canonical physical plan. */
    void sealPhysicalPlan()
    {
        PhysicalMemoryPlanBuilder builder;
        for (const auto &device : devices)
            builder.add(device.bom());
        physical_plan_ = std::make_shared<const PhysicalMemoryPlan>(
            builder.build());
    }

    std::shared_ptr<const PhysicalMemoryPlan> physical_plan_;
};

} // namespace llaminar2
