/**
 * @file OriginalNativePipelineReplayAdmission.h
 * @brief Original MPI ownership and production PMA preparation for model-free
 * pipeline generation replays with authored numerical graph recipes.
 *
 * Each process admits one selected cohort and one graph geometry. The fixture
 * retains its source past all worker, graph, tensor and communicator borrowers.
 */
#pragma once

#include "PipelineGenerationTestSupport.h"
#include "backends/NativeContextPreparation.h"
#include "planning/PlanningObservedResource.h"
#include "planning/RankHardwareOwnership.h"
#include <algorithm>
#include <charconv>
#include <string_view>
#include <cstdlib>

namespace llaminar2::test
{
class OriginalNativePipelineReplayAdmission final
{
public:
    /** @brief Resolve the requested ordered subset of the live, bound original rank.
     * @param family GPU backend selected before MPI startup.
     * @throws std::exception If the original rank or required cohort is absent.
     */
    explicit OriginalNativePipelineReplayAdmission(DeviceType family)
        : mpi(MPIContextFactory::global())
    {
        if (!mpi || mpi->world_size() != 1 || !mpi->clusterInventory())
            throw std::logic_error("Native pipeline replay requires its original bound singleton MPI rank");
        const auto &rank = mpi->clusterInventory()->getRank(mpi->rank());
        for (const auto &gpu : rank.gpus)
            if (gpu.type == family && rankOwnsGPUDevice(rank, gpu))
                devices.emplace_back(family, gpu.local_device_id);
        int width = 2;
        if (const auto *value = std::getenv("LLAMINAR_TEST_PIPELINE_REPLAY_WIDTH"))
        {
            const std::string_view argument(value);
            const auto result = std::from_chars(argument.data(), argument.data() + argument.size(), width);
            if (result.ec != std::errc{} || result.ptr != argument.data() + argument.size() || width < 2)
                throw std::invalid_argument("Native pipeline replay width must be at least two");
        }
        if (devices.size() < static_cast<size_t>(width))
            throw std::invalid_argument("Native pipeline replay width exceeds its actual original NUMA cohort");
        devices.resize(width);
        if (const auto *value = std::getenv("LLAMINAR_TEST_PIPELINE_REPLAY_ORDER"))
        {
            const std::string_view order(value);
            if (order != "forward" && order != "reverse")
                throw std::invalid_argument("Native pipeline replay has an invalid original stage order");
            if (order == "reverse") std::reverse(devices.begin(), devices.end());
        }
    }

    OriginalNativePipelineReplayAdmission(const OriginalNativePipelineReplayAdmission&) = delete;
    OriginalNativePipelineReplayAdmission& operator=(const OriginalNativePipelineReplayAdmission&) = delete;

    /** @brief Retire original workers after all later-declared owners leave scope. */
    ~OriginalNativePipelineReplayAdmission() noexcept
    {
        if (source_)
            try { source_->retireReleasedContexts(); }
            catch (...) { std::terminate(); }
    }

    /** @brief Add the physical observations of one declared stage to its graph BOM.
     * @param input Stage geometry and backend declarations retained by MemoryPlanner.
     * @throws std::exception If the device is foreign to the original inventory.
     */
    void observe(DevicePlanConfig &input) const
    {
        const auto &rank = mpi->clusterInventory()->getRank(mpi->rank());
        const auto resource = planningObservedResource(rank, input.device);
        const auto gpu = std::find_if(rank.gpus.begin(), rank.gpus.end(), [&](const auto &entry) {
            return entry.type == input.device.type && entry.local_device_id == input.device.ordinal;
        });
        if (gpu == rank.gpus.end() || !rankOwnsGPUDevice(rank, *gpu))
            throw std::logic_error("Native pipeline stage lost its original physical owner");
        input.world_rank = rank.rank;
        input.device_total_bytes = resource.total_bytes;
        input.device_free_bytes = resource.admission_available_bytes;
        input.associated_host_memory = planningObservedResource(rank, DeviceId::cpu());
        input.device_compute_units = gpu->compute_units;
        input.pipeline_forward_only_graph_owner = PipelineForwardOnlyGraphOwner::Participant;
    }

    /** @brief Birth selected workers and admit the exact final graph and transport BOM.
     * @param profile Model-free arithmetic and cache recipe.
     * @param inputs Ordered, original per-stage graph declarations.
     * @return The unchanged source PMA used by all subsequent workers and graphs.
     * @throws std::exception If the declarations or native final admission fail.
     */
    std::shared_ptr<PhysicalMemoryAuthority> admit(const ModelMemoryProfile &profile,
        std::vector<DevicePlanConfig> &inputs)
    {
        if (source_ || inputs.size() != devices.size())
            throw std::logic_error("Native pipeline replay requires one complete source declaration");
        const auto &rank = mpi->clusterInventory()->getRank(mpi->rank());
        std::vector<PhysicalMemoryResource> resources{planningObservedResource(rank, DeviceId::cpu())};
        for (size_t index = 0; index < inputs.size(); ++index)
        {
            auto &input = inputs[index];
            if (input.device != devices[index])
                throw std::logic_error("Native pipeline replay reordered the admitted stage declaration");
            observe(input);
            resources.push_back(planningObservedResource(rank, input.device));
        }
        source_ = NativeContextPreparation::create(resources, mpi->rank(), devices);
        for (const auto device : devices) (void)source_->acquire(device);
        source_->freezeForPlanning();
        for (auto &input : inputs) input.native_context_preparation = source_;
        PhysicalMemoryPlanBuilder plan(MemoryPlanner::plan(profile, inputs).physicalPlan());
        const auto table_source = OrdinaryDGOGraph::tableSourceStorage();
        for (const auto device : devices)
        {
            // The fixture profile has no transformer weight catalog. Add the
            // actual source and native embedding bytes used by each stage.
            plan.add(resources.front(), table_source.owner(), table_source.physicalBytes());
            plan.add(planningObservedResource(rank, device), PhysicalMemoryOwner::PrimaryModelWeights,
                OrdinaryDGOGraph::tableDeviceBytes());
        }
        source_->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()));
        return source_->memory();
    }

    const std::shared_ptr<IMPIContext> mpi; ///< Actual original MPI process/rank.
    std::vector<DeviceId> devices; ///< Original NUMA-owned devices in declared stage order.

private:
    std::shared_ptr<NativeContextPreparation> source_; ///< One live original source for the process.
};
}
