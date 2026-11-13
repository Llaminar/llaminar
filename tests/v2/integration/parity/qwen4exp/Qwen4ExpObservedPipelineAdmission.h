/**
 * @file Qwen4ExpObservedPipelineAdmission.h
 * @brief Actual original-rank binding for deferred native QFN pipeline requests.
 *
 * This setup-only fixture reuses production discovery and selected membership.
 * It owns no execution controller or model mirror. Unused ranks record explicit
 * inactivity; selected owners retain their original physical NUMA/UUID facts
 * and complete ordinary source admission before terminal evidence is emitted.
 */
#pragma once
#include "Qwen4ExpSourceAdmission.h"
#include "Qwen4ExpObservedPipelineTerminalEvidence.h"
#include "app/RuntimeInitPhase.h"
#include "backends/ComputeBackend.h"
#include "config/OrchestrationStartupPolicy.h"
#include "planning/ExecutionRankMembership.h"
#include "planning/ObservedPeerGrouping.h"
#include "planning/RankHardwareOwnership.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace llaminar2::test::parity::qwen4exp {
/**
 * @brief Retain one original discovery, selected execution and terminal evidence transaction.
 *
 * The inactive discovery rank holds no model context. The selected communicator
 * and all runner borrowers retire before both ranks publish terminal evidence.
 */
class Qwen4ExpObservedPipelineBinding final {
public:
    /**
     * @brief Bind physical addresses before any source or numerical runtime config.
     * @param requested Canonical unbound case in the original discovery namespace.
     * @throws std::logic_error When discovery cannot bind the complete original cohort.
     */
    explicit Qwen4ExpObservedPipelineBinding(const ModelParityCase& requested)
        :bound_(requested) {
        if(!requested.topology.needsObservedPipelineBinding() ||
           MPI_Comm_rank(MPI_COMM_WORLD,&discovery_rank_)!=MPI_SUCCESS ||
           MPI_Comm_size(MPI_COMM_WORLD,&discovery_size_)!=MPI_SUCCESS ||
           discovery_size_!=requested.topology.mpi_ranks)
            throw std::logic_error("QFN observed pipeline requires its original discovery ranks");
        OrchestrationConfig startup=OrchestrationConfig::defaults();
        startup.model_path=requested.model.model_path;
        startup.mpi_procs=discovery_size_;
        startup.planning_mode=OrchestrationPlanningMode::Automatic;
        startup.automatic_planning.only_backends=std::vector{requested.topology.observed_native_pipeline->backend};
        startup.automatic_planning.only_strategies=std::vector{OrchestrationStrategy::PipelineParallel};
        publishOrchestrationStartupPolicy(startup);
        if(!DeviceManager::instance().hardware())
            RuntimeInitPhase::initializeDiscoveryRuntime(startup,discovery_rank_,discovery_size_);
        discovery_=MPIContextFactory::global();
        if(!discovery_ || discovery_->rank()!=discovery_rank_ || discovery_->world_size()!=discovery_size_)
            throw std::logic_error("QFN observed pipeline lost original discovery ownership");
        inventory_=discovery_->clusterInventory();
        if(!inventory_)throw std::logic_error("QFN observed pipeline has no original inventory");
        bound_.topology=requested.topology.bindObservedPipeline(*inventory_);
        const auto& selection=bound_.topology.execution_rank_selection;
        if(!selection || selection->size()!=1)
            throw std::logic_error("QFN observed pipeline requires one selected original owner");
        const auto& original=inventory_->ranks.at(selection->discoveryRanks().front());
        binding_={{"schema",1},{"cell",requested.testName()},
            {"discovery_rank",discovery_rank_},{"discovery_size",discovery_size_},
            {"selected_discovery_ranks",selection->discoveryRanks()},
            {"requested_native_isa",requested.topology.observed_native_pipeline->native_isa},
            {"stage_sizes",requested.topology.pipeline_stage_sizes},
            {"original_owner",selection->discoveryRanks().front()},{"original_hostname",original.hostname},
            {"original_numa_node",original.cpu.numa_node},{"participants",nlohmann::json::array()}};
        const ObservedPeerGrouping devices(original);
        size_t offset=0;
        for(size_t stage=0;stage<bound_.topology.pipeline_stage_sizes.size();++stage)
            for(int member=0;member<bound_.topology.pipeline_stage_sizes[stage];++member) {
                const auto& address=bound_.topology.participants.at(offset++).address;
                const DeviceId device{address.device_type,address.device_ordinal};
                requireRankLocalGPUOwnership(original,device,"QFN observed original pipeline");
                const auto& gpu=devices.device(device);
                binding_["participants"].push_back({{"stage",stage},{"backend",device.type==DeviceType::CUDA?"cuda":"rocm"},
                    {"ordinal",device.ordinal},{"uuid",gpu.uuid},{"native_isa",gpu.native_isa},
                    {"numa_node",gpu.numa_node},{"pcie_upstream_bdfs",gpu.pcie_upstream_bdfs}});
            }
        auto admission=MPIContextFactory::selectRanks(discovery_,selection->discoveryRanks());
        if(const auto* active=std::get_if<std::shared_ptr<MPIContext>>(&admission))
            execution_=*active;
        binding_["active"]=bool(execution_);
        binding_["execution_rank"]=execution_?nlohmann::json(execution_->rank()):nlohmann::json(nullptr);
    }

    Qwen4ExpObservedPipelineBinding(const Qwen4ExpObservedPipelineBinding&)=delete;
    Qwen4ExpObservedPipelineBinding& operator=(const Qwen4ExpObservedPipelineBinding&)=delete;

    /** @return The original case with concrete participant addresses and rank selection. */
    const ModelParityCase& boundCase() const noexcept {return bound_;}
    /** @return Actual selected MPI owner, absent on the inactive discovery rank. */
    const std::shared_ptr<MPIContext>& execution() const noexcept {return execution_;}
    /** @return Discovery identity retained for terminal rank evidence. */
    int discoveryRank() const noexcept {return discovery_rank_;}

    /** @brief Authenticate physical owners and exact active/inactive membership before setup. */
    void authenticate() const {
        const auto& selection=*bound_.topology.execution_rank_selection;
        const int original_owner=selection.discoveryRanks().front();
        if(!execution_) {
            if(discovery_rank_==original_owner)
                throw std::logic_error("QFN observed pipeline excluded the original owner");
            return;
        }
        if(execution_->rank()!=0 || execution_->world_size()!=1 ||
           !execution_->executionMembership() ||
           execution_->executionMembership()->selection()!=selection ||
           execution_->executionMembership()->executionRank(discovery_rank_)!=0 || discovery_rank_!=original_owner)
            throw std::logic_error("QFN observed pipeline substituted original selected membership");
        const auto& original=inventory_->ranks.at(original_owner);
        const auto& projected=execution_->clusterInventory()->ranks.at(0);
        if(projected.cpu.numa_node!=original.cpu.numa_node || projected.hostname!=original.hostname)
            throw std::logic_error("QFN observed pipeline changed original NUMA or physical host");
        const ObservedPeerGrouping original_devices(original), projected_devices(projected);
        for(const auto& participant:bound_.topology.participants) {
            const DeviceId device{participant.address.device_type,participant.address.device_ordinal};
            const auto& before=original_devices.device(device);
            const auto& after=projected_devices.device(device);
            if(before.uuid!=after.uuid || before.native_isa!=after.native_isa || before.numa_node!=after.numa_node ||
               before.pcie_upstream_bdfs!=after.pcie_upstream_bdfs)
                throw std::logic_error("QFN observed pipeline changed original GPU UUID/ISA/NUMA identity");
        }
    }

    /**
     * @brief Join original discovery after all selected work and runner owners retire.
     * @param locally_succeeded Actual setup/execution and teardown outcome.
     * @return Whether the entire original discovery transaction completed.
     */
    bool complete(bool locally_succeeded) {
        if(binding_.contains("terminal_join_completed"))
            throw std::logic_error("QFN observed pipeline repeated terminal join");
        const std::weak_ptr<MPIContext> retired=execution_;
        execution_.reset();
        const ObservedPipelineRankTerminalOutcome local{
            retired.expired()?ObservedPipelineRetirement::Retired:ObservedPipelineRetirement::Live,
            locally_succeeded?ObservedPipelineWorkOutcome::Succeeded:ObservedPipelineWorkOutcome::Failed};
        std::vector<ObservedPipelineRankTerminalOutcome> completed(discovery_size_);
        discovery_->allgather_bytes(&local,completed.data(),sizeof(local));
        terminal_.emplace(completed);
        binding_["terminal_join_completed"]=true;
        binding_["terminal_owners_retired"]=terminal_->allRetired();
        return terminal_->succeeded();
    }

    /** @return Whether this authenticated original active owner retired before diagnostic export. */
    bool mayExportConfiguration() const noexcept {
        return terminal_ && binding_.at("active").get<bool>() &&
            terminal_->mayExportConfiguration(bound_.topology.execution_rank_selection,discovery_rank_);
    }

    /** @brief Write one receipt under the existing canonical artifact owner after the terminal join. */
    void writeReceipt(const std::filesystem::path& output,const char* success_key,bool succeeded) const {
        ASSERT_TRUE(binding_.contains("terminal_join_completed"));
        ASSERT_TRUE(output.is_absolute());
        ASSERT_FALSE(std::filesystem::is_symlink(output));
        ASSERT_TRUE(std::filesystem::is_directory(output));
        ASSERT_FALSE(std::filesystem::exists(output/"observed-pipeline-binding.json"))
            << "Preserve prior original membership evidence and select a fresh artifact root";
        auto receipt=binding_;
        receipt[success_key]=succeeded;
        std::ofstream file(output/"observed-pipeline-binding.json");
        ASSERT_TRUE(file.is_open());
        file<<receipt.dump()<<'\n';file.flush();ASSERT_TRUE(file.good());
    }
private:
    ModelParityCase bound_;
    int discovery_rank_=-1;
    int discovery_size_=0;
    std::shared_ptr<IMPIContext> discovery_;
    std::shared_ptr<const ClusterInventory> inventory_;
    std::shared_ptr<MPIContext> execution_;
    nlohmann::json binding_;
    std::optional<Qwen4ExpObservedPipelineTerminalEvidence> terminal_;
};

/**
 * @brief Bind original ownership and execute one selected source admission.
 * @param requested Canonical unbound source case.
 * @param work Existing source admission consuming the bound case and selected context.
 */
template<class Work>
void withQwen4ExpObservedPipelineAdmission(const ModelParityCase& requested,Work&& work) {
    Qwen4ExpObservedPipelineBinding binding(requested);
    const bool active=bool(binding.execution());
    std::exception_ptr failure;
    try {
        binding.authenticate();
        if(active)work(binding.boundCase(),binding.execution());
    } catch(...) {failure=std::current_exception();}
    const bool all_completed=binding.complete(!failure&&!::testing::Test::HasFailure());
    const char* artifact_root=std::getenv("LLAMINAR_PRODUCTION_PARITY_ARTIFACT_ROOT");
    const auto base=artifact_root?std::filesystem::path(artifact_root):
        std::filesystem::current_path()/"parity-results"/"qwen-flash-next";
    ASSERT_TRUE(base.is_absolute());
    const auto output=base/"source-admission"/parity::artifactIdentityPath(requested.testName())/
        ("rank-"+std::to_string(binding.discoveryRank()));
    if (!active || !std::filesystem::exists(output)) prepareQwen4ExpAdmissionEvidenceDirectory(output);
    ASSERT_NO_FATAL_FAILURE(binding.writeReceipt(output,"source_work_succeeded",all_completed));
    EXPECT_TRUE(all_completed)<<"Original selected pipeline admission failed before terminal retirement";
    if(failure)std::rethrow_exception(failure);
}
} // namespace llaminar2::test::parity::qwen4exp
