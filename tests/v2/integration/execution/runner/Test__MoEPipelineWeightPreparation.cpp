/**
 * @file Test__MoEPipelineWeightPreparation.cpp
 * @brief Real GGUF preparation through the PP factory and its admitted memory authority.
 *
 * A tiny generated model reaches the production loading, packing and graph
 * initialization paths. Sparse zero payloads test ownership, never model
 * quality. Planning alone cannot detect generic preparation incorrectly
 * charging routed experts to PrimaryModelWeights. Every participant must load
 * only its own PP layers and logical expert slices, including terminal MTP.
 * Initialization must also materialize the captured maintenance family before
 * checking readiness; a prepared weight registry alone is not a serving graph.
 * Prepared-stage inspection retains the actual original world rank: resolver
 * defaults for a single process cannot inspect another rank's local TP cohort.
 * A reusable context must publish completion and exact retirement ownership
 * for every inner TP participant, including the non-primary pipeline stages.
 * Singleton stages retain their exact owner without migration work. Native
 * homogeneous domains retain their actual rank-local captured transfer edges.
 * The EP4+EP4 profile selects exactly eight observed gfx906 UUIDs after original
 * two-rank discovery, then uses production execution-rank admission for their
 * one physical NUMA owner. Static/Dynamic and ordinal/random remain real axes.
 * Request regressions additionally enter real sparse generation after setup,
 * reset and prefix reuse. Every stage must receive the root's request identity;
 * successful weight preparation alone cannot prove that lifecycle publication.
 * Cold one-token responses must leave no hidden stream-binding prerequisite.
 * Exact-width follower families are authenticated by width and may materialize
 * once only; cache reset and prefix restore must reuse that same topology.
 * Shutdown must retire every actually recorded maintenance frame once before
 * the original native cohort disappears; repeated shutdown creates no event.
 * Setup capture evidence comes from successfully instantiated native graph
 * metadata. A materialized, unlaunched graph is not an executed decode phase.
 */
#include "../../../utils/PlanningGGUFFixture.h"
#include "../../../utils/ObservedISAPipelinePlacement.h"
#include "../../parity/qwen35moe/Qwen35MoEParityEconomics.h"
#include "../../../utils/ServingGGUFFixturePayload.h"
#include "PipelineGenerationTestSupport.h"
#include "../../../fixtures/backends/PreparedNativeLookupFixture.h"
#include "../../../fixtures/backends/RetainedParticipantGraphExecutableFixtureAccess.h"
#include "memory/CPUMappedPublicationRoot.h"
#include "memory/CPUTensorStorageDeclaration.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/graph/ReadAheadScheduling.h"
#include "transfer/TransferEngine.h"
#include "backends/BackendManager.h"
#include "config/OrchestrationConfigParser.h"
#include "config/OrchestrationStartupPolicy.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "execution/moe/MoEPipelinePreparedPlan.h"
#include "execution/runner/OrchestrationRunner.h"
#include "execution/runner/ModelContextRetirement.h"
#include "execution/mtp/MTPGraphOwnerPlan.h"
#include "loaders/ModelContext.h"
#include "loaders/PreparedWeightStore.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "utils/PerfStatsCollector.h"
#include "utils/KernelProfiler.h"
#include "planning/RankHardwareOwnership.h"
#include "utils/NUMATopology.h"
#include "planning/ExecutionRankMembership.h"
#include "utils/MPIContext.h"
#include "execution/moe/MoEOverlayCapacityAdmission.h"
#include "execution/moe/MoEOverlayNoPeerResidency.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <charconv>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <cstdio>
#include <exception>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace llaminar2::test
{
namespace
{
    /** @brief Explicit serving policy, independent of the generated test name. */
    enum class MTPLane { Disabled, Fixed, Automatic };

    /** @brief Hardware and expert representation are explicit request policy. */
    enum class PipelineCohort { NativeDomain, ObservedMI50WholeExpert };

    /**
     * @brief Authenticate an actual speculative native-parent launch observation.
     * @param record Existing DGO counter recorded after successful launch and terminal publication.
     * @param participant Exact upstream device whose optimized execution must be proved.
     * @return False for ordinary depth0, foreign devices/backends, hosted or malformed records.
     */
    bool isNativeSpeculativeParentLaunch(const PerfStatRecord &record,DeviceId participant)
    {
        if(!participant.is_gpu() || record.kind!=PerfStatRecord::Kind::Counter ||
            record.domain!="mtp" || record.name!="device_generation_loop_graph_launches" ||
            record.device!=participant.toString())return false;
        const auto backend=record.tags.find("backend"),depth=record.tags.find("draft_depth"),
            execution=record.tags.find("execution");
        if(backend==record.tags.end() || depth==record.tags.end() || execution==record.tags.end() ||
            backend->second!=(participant.is_cuda()?"CUDA":"HIP") ||
            (execution->second!="single_async_native_while_launch" &&
             execution->second!="single_async_native_selector_while_launch"))return false;
        int value=0;const auto *end=depth->second.data()+depth->second.size();
        const auto parsed=std::from_chars(depth->second.data(),end,value);
        return parsed.ec==std::errc{} && parsed.ptr==end && value>0;
    }

    /** @test Ordinary, foreign and hosted work cannot certify a missing speculative native parent. */
    TEST(MoEPipelineGenerationEvidence, NativeSpeculativeLaunchRejectsOrdinaryForeignOrHostedRecords)
    {
        const auto device=DeviceId::cuda(0);
        const PerfStatRecord valid{.domain="mtp",.name="device_generation_loop_graph_launches",.phase="decode",
            .device=device.toString(),.tags={{"backend","CUDA"},{"draft_depth","3"},
                {"execution","single_async_native_while_launch"}},.value=1};
        EXPECT_TRUE(isNativeSpeculativeParentLaunch(valid,device));
        auto selector=valid;selector.tags["execution"]="single_async_native_selector_while_launch";
        EXPECT_TRUE(isNativeSpeculativeParentLaunch(selector,device));
        for(const auto *depth:{"0","-1","3junk","","999999999999999999999999"})
        {auto invalid=valid;invalid.tags["draft_depth"]=depth;EXPECT_FALSE(isNativeSpeculativeParentLaunch(invalid,device));}
        auto invalid=valid;invalid.tags.erase("draft_depth");EXPECT_FALSE(isNativeSpeculativeParentLaunch(invalid,device));
        invalid=valid;invalid.tags.erase("backend");EXPECT_FALSE(isNativeSpeculativeParentLaunch(invalid,device));
        invalid=valid;invalid.tags.erase("execution");EXPECT_FALSE(isNativeSpeculativeParentLaunch(invalid,device));
        invalid=valid;invalid.device=DeviceId::cuda(1).toString();EXPECT_FALSE(isNativeSpeculativeParentLaunch(invalid,device));
        invalid=valid;invalid.tags["backend"]="HIP";EXPECT_FALSE(isNativeSpeculativeParentLaunch(invalid,device));
        invalid=valid;invalid.tags["execution"]="hosted_captured_transactions_with_ticket_only_dispatch";
        EXPECT_FALSE(isNativeSpeculativeParentLaunch(invalid,device));
        invalid=valid;invalid.domain="generation";EXPECT_FALSE(isNativeSpeculativeParentLaunch(invalid,device));
        invalid=valid;invalid.kind=PerfStatRecord::Kind::Timer;EXPECT_FALSE(isNativeSpeculativeParentLaunch(invalid,device));
        EXPECT_FALSE(isNativeSpeculativeParentLaunch(valid,DeviceId::cpu()));
    }

    /** @brief Two disjoint native TP domains and one selected MTP policy. */
    struct PreparationCase
    {
        bool first_cuda;
        bool last_cuda;
        int width;
        MTPLane lane;
        PipelineCohort cohort = PipelineCohort::NativeDomain;
        int fixed_depth = 3;
        RoutedExpertOwnerOrder owner_order = RoutedExpertOwnerOrder::Ordinal;
        MoERebalanceRuntimeMode movement = MoERebalanceRuntimeMode::Dynamic;
        std::vector<int> stage_widths; ///< Empty retains the original equal-width two-stage case.
        /** @return Exact authored stage count, independent of cohort capacity. */
        int stageCount() const { return stage_widths.empty() ? 2 : static_cast<int>(stage_widths.size()); }
        /** @return Exact participant count for one declared stage. */
        int stageWidth(int stage) const { return stage_widths.empty() ? width : stage_widths.at(stage); }
        /** @return Actual declared backend; architecture-selected profiles are homogeneous ROCm. */
        bool cudaStage(int stage) const { return stage == 0 ? first_cuda : last_cuda; }
        /** @return Whether original discovery narrows to the physical MI50 owner. */
        bool selectsMI50Owner() const { return cohort == PipelineCohort::ObservedMI50WholeExpert; }
        /** @return Whether the source publication spans the two discovery ranks. */
        bool distributedDiscovery() const { return first_cuda != last_cuda || selectsMI50Owner(); }
        /** @return Stable CTest/GTest identity for the exact physical topology. */
        std::string name() const
        {
            if (selectsMI50Owner())
            {
                std::string topology;
                for (int stage = 0; stage < stageCount(); ++stage)
                    topology += "MI50_EP" + std::to_string(stageWidth(stage)) + "_";
                return topology +
                    (owner_order == RoutedExpertOwnerOrder::Ordinal ? "Ordinal_" : "Random_") +
                    (movement == MoERebalanceRuntimeMode::Off ? "Static_" : "Dynamic_") +
                    (lane == MTPLane::Disabled ? "DisabledMTP" : lane == MTPLane::Automatic ? "DynamicMTP" :
                        "Fixed" + std::to_string(fixed_depth) + "MTP");
            }
            const auto domain = [this](bool cuda) { return std::string(cuda ? "CUDA" : "ROCm") + std::to_string(width); };
            return domain(first_cuda) + "_" + domain(last_cuda) + "_" +
                (lane == MTPLane::Disabled ? "DisabledMTP" : lane == MTPLane::Fixed ? "FixedMTP" : "DynamicMTP");
        }
    };

    /** @return Only topologies whose native backends were compiled. */
    std::vector<PreparationCase> cases()
    {
        std::vector<PreparationCase> result;
        for (const auto lane : {MTPLane::Disabled, MTPLane::Fixed, MTPLane::Automatic})
        {
#ifdef HAVE_CUDA
            result.push_back({true, true, 1, lane});
            result.push_back({true, true, 2, lane});
#endif
#ifdef HAVE_ROCM
            result.push_back({false, false, 1, lane});
            result.push_back({false, false, 2, lane});
#endif
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            for (const int width : {2, 4})
            {
                result.push_back({false, true, width, lane});
                result.push_back({true, false, width, lane});
            }
#endif
        }
#ifdef HAVE_ROCM
        for (const auto widths : {std::vector<int>{4, 4}, std::vector<int>{1, 2},
                 std::vector<int>{2, 1}, std::vector<int>{1, 2, 4}})
        for (const auto owner_order : {RoutedExpertOwnerOrder::Ordinal, RoutedExpertOwnerOrder::Random})
            for (const auto movement : {MoERebalanceRuntimeMode::Off, MoERebalanceRuntimeMode::Dynamic})
                for (const auto [lane, depth] : {std::pair{MTPLane::Disabled, 0},
                        std::pair{MTPLane::Fixed, 1}, std::pair{MTPLane::Fixed, 3},
                        std::pair{MTPLane::Automatic, 3}})
                    result.push_back({false, false, 4, lane, PipelineCohort::ObservedMI50WholeExpert,
                        depth, owner_order, movement, widths});
#endif
        return result;
    }

    /** @brief Immutable physical endpoints selected from the original discovery publication. */
    struct PreparationPlacement
    {
        std::vector<int> owners; ///< Original world ranks, never backend ordinal heuristics.
        std::vector<std::vector<DeviceId>> devices; ///< Exact NUMA-owned local device namespaces.
    };

    /** @brief Select disjoint UUID-authenticated devices without changing rank affinity or visibility. */
    PreparationPlacement placement(const PreparationCase &test)
    {
        PreparationPlacement result;
        result.owners.resize(test.stageCount());
        result.devices.resize(test.stageCount());
        const auto mpi = MPIContextFactory::global();
        const int required_ranks = test.distributedDiscovery() ? 2 : 1;
        if (!mpi || mpi->world_size() != required_ranks || !mpi->clusterInventory())
            throw std::runtime_error("Pipeline preparation requires its original observed rank geometry");
        if (test.selectsMI50Owner())
        {
            std::vector<std::size_t> sizes;
            for (int stage = 0; stage < test.stageCount(); ++stage)
                sizes.push_back(static_cast<std::size_t>(test.stageWidth(stage)));
            const auto observed = ObservedISAPipelinePlacement::require(*mpi->clusterInventory(), DeviceType::ROCm, "gfx906", sizes);
            return {std::vector<int>(observed.stages.size(), observed.discovery_owner), observed.stages};
        }
        if(test.first_cuda!=test.last_cuda) {
            // The existing distributed case must retain two original native
            // owners even when one socket now owns an incidental other-vendor
            // device. Only the complete joint request selects its participants.
            std::vector<ObservedNativePipelineStage> request;
            for(int stage=0;stage<test.stageCount();++stage)
                request.push_back({test.cudaStage(stage)?DeviceType::CUDA:DeviceType::ROCm,
                    static_cast<std::size_t>(test.stageWidth(stage))});
            auto observed=ObservedCrossVendorPipelinePlacement::require(*mpi->clusterInventory(),request);
            return {std::move(observed.owners),std::move(observed.stages)};
        }
        std::set<std::string> selected;
        for (int stage = 0; stage < test.stageCount(); ++stage)
        {
            const auto family = test.cudaStage(stage) ? DeviceType::CUDA : DeviceType::ROCm;
            bool found = false;
            for (const auto &rank : mpi->clusterInventory()->ranks)
            {
                std::vector<DeviceId> devices;
                std::vector<std::string> identities;
                for (const auto &gpu : rank.gpus)
                    if (gpu.type == family && rankOwnsGPUDevice(rank, gpu) &&
                        !gpu.uuid.empty() && !selected.contains(gpu.uuid) &&
                        std::find(identities.begin(), identities.end(), gpu.uuid) == identities.end())
                    {
                        devices.emplace_back(family, gpu.local_device_id);
                        identities.push_back(gpu.uuid);
                        if (devices.size() == static_cast<size_t>(test.stageWidth(stage))) break;
                    }
                if (devices.size() != static_cast<size_t>(test.stageWidth(stage))) continue;
                result.owners[stage] = rank.rank;
                result.devices[stage] = std::move(devices);
                selected.insert(identities.begin(), identities.end());
                found = true;
                break;
            }
            if (!found) throw std::runtime_error("Required original NUMA-owned pipeline cohort is absent");
        }
        return result;
    }

    /** @brief Admit original discovery membership without changing physical GPU ownership. */
    struct PreparationExecution final
    {
        std::shared_ptr<IMPIContext> mpi;
        PreparationPlacement physical;
        std::optional<ExecutionRankSelection> selection;

        /** @brief Reuse the production selected-rank transaction; inactive ranks execute no GPU work. */
        static PreparationExecution admit(const PreparationCase &test)
        {
            const auto discovery = MPIContextFactory::global();
            auto physical = placement(test);
            if (!test.selectsMI50Owner()) return {discovery, std::move(physical), std::nullopt};
            ExecutionRankSelection selection({physical.owners[0]});
            auto admitted = MPIContextFactory::selectRanks(discovery, selection.discoveryRanks());
            if (std::holds_alternative<InactiveMPIRank>(admitted))
                return {nullptr, std::move(physical), selection};
            auto active = std::get<std::shared_ptr<MPIContext>>(std::move(admitted));
            if (!active->executionMembership() || active->world_size() != 1 || active->rank() != 0 ||
                active->executionMembership()->selection() != selection)
                throw std::logic_error("MI50 pipeline lost its selected original owner");
            const auto original_inventory = discovery->clusterInventory();
            const auto &original_owner = original_inventory->ranks.at(selection.discoveryRanks().front());
            const auto projected_inventory = active->clusterInventory();
            const auto &projected_owner = projected_inventory->ranks.at(active->rank());
            if (discovery->rank() != selection.discoveryRanks().front() ||
                active->executionMembership()->executionRank(discovery->rank()) != active->rank() ||
                projected_owner.cpu.numa_node != original_owner.cpu.numa_node ||
                projected_owner.hostname != original_owner.hostname)
                throw std::logic_error("MI50 pipeline selected rank changed original discovery/NUMA ownership");
            for (const auto &stage : physical.devices)
                for (const auto device : stage)
                {
                    const auto find = [&](const RankInventory &owner) -> const DeviceInfo & {
                        const auto found = std::find_if(owner.gpus.begin(), owner.gpus.end(), [&](const auto &gpu) {
                            return gpu.type == device.type && gpu.local_device_id == device.ordinal; });
                        if (found == owner.gpus.end()) throw std::logic_error("Selected pipeline lost original GPU observation");
                        return *found;
                    };
                    const auto &before = find(original_owner), &after = find(projected_owner);
                    if (before.uuid.empty() || before.uuid != after.uuid || before.numa_node != after.numa_node ||
                        before.native_isa != after.native_isa || before.pcie_upstream_bdfs != after.pcie_upstream_bdfs)
                        throw std::logic_error("Selected pipeline substituted original GPU UUID/NUMA/ISA identity");
                }
            std::fill(physical.owners.begin(), physical.owners.end(), 0); // Compact namespace; physical facts unchanged.
            for (const auto &stage : physical.devices)
                for (const auto device : stage)
                    requireRankLocalGPUOwnership(active->clusterInventory()->ranks.at(0), device,
                        "MI50 original selected pipeline owner");
            return {std::move(active), std::move(physical), selection};
        }
    };

    /** @return The exact original-world two-stage configuration shared by loading and request tests. */
    OrchestrationConfig configuration(const PreparationCase &test, const std::string &model,
        const PreparationExecution &execution)
    {
        const auto &physical = execution.physical;
        std::vector<std::string> arguments{"moe-pipeline-preparation", "-m", model,
            "--mpi-procs", test.distributedDiscovery() ? "2" : "1", "--tp-scope", "rank_local", "--context-length", "512"};
        for (int stage = 0; stage < test.stageCount(); ++stage)
        {
            const bool cuda = test.cudaStage(stage);
            std::string domain = "stage" + std::to_string(stage) + "=";
            for (size_t participant = 0; participant < physical.devices[stage].size(); ++participant)
                domain += (participant ? "," : "") + std::string(cuda ? "cuda:" : "rocm:") +
                    std::to_string(physical.devices[stage][participant].ordinal);
            domain += ";scope=rank_local;backend=" + std::string(cuda ? "nccl" : "rccl") +
                ";owner=" + std::to_string(physical.owners[stage]);
            if (test.selectsMI50Owner())
                domain += ";expert_compute=apportioned;expert_phase=uniform";
            arguments.insert(arguments.end(), {"--define-domain", domain, "--pp-stage",
                std::to_string(stage) + "=stage" + std::to_string(stage) + ":" + std::to_string(stage) + "-" + std::to_string(stage)});
        }
        if (test.lane == MTPLane::Disabled) arguments.push_back("--no-mtp");
        if (test.selectsMI50Owner() && test.lane == MTPLane::Automatic)
            // This fixture authors the complete adaptive range inside retained3;
            // an omitted maximum means the public full supported range (15).
            arguments.insert(arguments.end(), {"--mtp-depth-policy", "dynamic", "--mtp-draft-tokens", "3",
                "--mtp-max-draft-tokens", "3"});
        if (test.lane == MTPLane::Fixed)
            arguments.insert(arguments.end(), {"--mtp-depth-policy", "fixed", "--mtp-draft-tokens", std::to_string(test.fixed_depth)});
        std::vector<char *> argv;
        for (auto &argument : arguments) argv.push_back(argument.data());
        auto config = OrchestrationConfigParser{}.parseArgs(argv.size(), argv.data());
        config.prefill_max_bucket_size = 64;
        if (test.selectsMI50Owner())
        {
            config.planning_mode = OrchestrationPlanningMode::Apply;
            config.execution_rank_selection = execution.selection;
            config.routed_expert_owner_order = test.owner_order;
            config.moe_rebalance = parity::qwen35moe::qwen35MoEDynamicParityEconomics();
            config.moe_rebalance.mode = test.movement;
            // Preserve a complete depth3 predictor+verifier between recurring
            // maintenance boundaries, including the fixed1 retained3 control.
            config.moe_rebalance.device_min_maintenance_period_tokens = 4;
            if (test.lane != MTPLane::Disabled) config.mtp.graph_capacity_draft_tokens = 3;
        }
        return config;
    }

    /**
     * @brief Own one real fixture runner and its explicitly exported final model duty.
     * Inspection aliases declared later must die before this scope, including
     * on fatal assertions. Shutdown publishes Reusable; only the canonical
     * final-owner API may then reset its original prepared native generations.
     */
    class PipelinePreparedFixtureLifetime final
    {
    public:
        /** @param config Actual public pipeline configuration, never a synthetic native-ready runner. */
        explicit PipelinePreparedFixtureLifetime(const OrchestrationConfig& config, bool original_world = false,
            std::shared_ptr<IMPIContext> original = {})
            :runner_(original_world ? std::make_unique<OrchestrationRunner>(
                original ? std::move(original) : MPIContextFactory::global(), config,createExecutionPlanBuilder()) :
                std::make_unique<OrchestrationRunner>(config,createExecutionPlanBuilder())){}
        /** @brief Retire every original owner on normal and exceptional test exits. */
        ~PipelinePreparedFixtureLifetime()noexcept
        {
            try{finish();}catch(...){std::terminate();}
        }
        /** @return Sole original runner, borrowed only before final retirement. */
        OrchestrationRunner& runner()const{return *runner_;}
        /** @brief Retain the actual coordinated source's original worker-loop shutdown duty. */
        void armCommandSource(bool source) { command_source_ = source; }
        /** @brief Stop original worker admission before inspecting terminal local observations. */
        void closeWorkers()
        {
            if (command_source_)
            {
                runner_->shutdownMPIWorkers();
                command_source_ = false;
            }
        }
        /**
         * @brief Retain the actual exported contract rather than abandon its reset duty.
         * @return Original optional contract, without another ModelContext ownership copy.
         */
        const std::optional<ModelContextReuseContract>& exportContract()
        {
            if(!contract_)contract_=runner_->modelContextReuseContract();
            return contract_;
        }
        /**
         * @brief Destroy the runner before consuming the sole original reusable model.
         * @return Actual canonical receipt when a contract was exported, otherwise no model reuse.
         * @throws std::logic_error For re-entry after an unproven retirement or retained inspection owners.
         */
        const std::optional<ModelContextRetirementReceipt>& finish()
        {
            if(phase_==Phase::Retired)return receipt_;
            if(phase_!=Phase::Owned)throw std::logic_error("Pipeline fixture retirement did not prove completion");
            phase_=Phase::Retiring;
            closeWorkers();
            runner_->shutdown();runner_.reset();
            if(contract_){receipt_=retireExclusiveModelContextReuseContract(*contract_);contract_.reset();}
            phase_=Phase::Retired;return receipt_;
        }
    private:
        enum class Phase {Owned,Retiring,Retired};
        std::optional<ModelContextReuseContract> contract_;
        std::unique_ptr<OrchestrationRunner> runner_;
        std::optional<ModelContextRetirementReceipt> receipt_;
        Phase phase_=Phase::Owned;
        bool command_source_ = false;
    };

    /** @brief Complete model initialization, not an admission-only substitute. */
    class MoEPipelineWeightPreparation : public ::testing::TestWithParam<PreparationCase>
    {
    protected:
        /** @brief Require the exact devices selected by original hardware qualification. */
        void SetUp() override
        {
            const auto test = GetParam();
            int cuda_count = 0, rocm_count = 0;
            for (int stage = 0; stage < test.stageCount(); ++stage)
                (test.cudaStage(stage) ? cuda_count : rocm_count) += test.stageWidth(stage);
            for (const auto [cuda, required] : {std::pair{true, cuda_count}, std::pair{false, rocm_count}})
            {
                if (!required) continue;
                auto *backend = cuda ? getCUDABackend() : getROCmBackend();
                ASSERT_NE(backend, nullptr);
                ASSERT_GE(backend->deviceCount(), required);
            }
        }
    };

    /** @brief Exact native stage order followed by a real host compute tail. */
    struct NoPeerHostTailCase
    {
        std::vector<DeviceType> stages;
        std::string name;
    };

    /** @brief Backend-conditional cases with an actual GPU continuation and CPU tail. */
    std::vector<NoPeerHostTailCase> noPeerHostTailCases()
    {
        std::vector<NoPeerHostTailCase> result;
#ifdef HAVE_CUDA
        result.push_back({{DeviceType::CUDA, DeviceType::CPU}, "CUDA1_CPU1"});
#endif
#ifdef HAVE_ROCM
        result.push_back({{DeviceType::ROCm, DeviceType::CPU}, "ROCm1_CPU1"});
#endif
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        result.push_back({{DeviceType::CUDA, DeviceType::ROCm, DeviceType::CPU}, "CUDA1_ROCm1_CPU1"});
        result.push_back({{DeviceType::ROCm, DeviceType::CUDA, DeviceType::CPU}, "ROCm1_CUDA1_CPU1"});
#endif
        return result;
    }

    /**
     * @brief Stop divergent test failure before original ranks enter unmatched retirement.
     * @note The actual MPI communicator remains owned by the existing fixture lifetime.
     */
    struct NoPeerOriginalFailureBoundary final
    {
        std::shared_ptr<IMPIContext> original;
        ~NoPeerOriginalFailureBoundary() noexcept
        {
            if (original && (::testing::Test::HasFatalFailure() || std::uncaught_exceptions()))
            {
                std::fprintf(stderr, "Original mixed no-peer rank %d failed before terminal retirement\n",
                             original->rank());
                MPI_Abort(original->communicator(), 1);
                std::terminate();
            }
        }
    };

    /**
     * @brief Copy a live GPU prefix to the original CPU bank after a delayed native producer.
     * @param family Native family admitted by the canonical original-NUMA bootstrap.
     *
     * The host destination keeps poison beyond the requested prefix. The GPU
     * remains authoritative and its full host mirror remains invalid. Only the
     * original event-completed live copy may contribute D2H bytes.
     */
    void provePendingHostActivationExtent(DeviceType family)
    {
        using Graph = RetainedParticipantGraphExecutableFixtureAccess;
        const auto placement = observedSingletonPipelinePlacement(family);
        const auto source_device = placement.gpu;
        const auto mpi = MPIContextFactory::global();
        initCPUBackend(placement.cpu_numa_node);
        OrdinaryGraphEvidence evidence("transfer");
        const std::array devices{source_device};
        auto original = PreparedNativeLookupFixture::prepare(devices, mpi->rank());
        const CPUTensorStorageDeclaration tensor(64, 128, BufferTensorType::FP32,
            PhysicalMemoryOwner::ExecutionWorkspace, DeviceId::cpu());
        const CPUMappedPublicationRoot gate_root(sizeof(std::uint64_t), DeviceId::cpu(), source_device);
        const auto capacity = tensor.rows() * tensor.columns() * sizeof(float);
        PhysicalMemoryPlanBuilder bom(RetainedParticipantGraphExecutable::generalBOM(
            original->resource(DeviceId::cpu()), original->resource(source_device), 1));
        bom.add(original->resource(DeviceId::cpu()), PhysicalMemoryOwner::ExecutionWorkspace,
            2 * tensor.physicalBytes());
        bom.add(original->resource(source_device), PhysicalMemoryOwner::ExecutionWorkspace, capacity);
        bom.add(original->resource(DeviceId::cpu()), gate_root.owner(), gate_root.physicalBytes());
        original->admit(bom.build());
        {
            auto worker = original->permit(source_device).worker();
            auto *const backend = getBackendFor(source_device);
            ASSERT_NE(backend, nullptr);
            void *const producer = worker->defaultStream();
            auto &transfers = TransferEngine::instance();
            auto gate = transfers.allocateMappedHostRegion(gate_root, original->memory());
            auto *signal = static_cast<std::uint64_t *>(gate->mutableHostData());
            std::atomic_ref<std::uint64_t>(*signal).store(0, std::memory_order_release);
            auto source = tensor.materialize(original->memory());
            auto destination = tensor.materialize(original->memory());
            std::memset(source->raw_mutable_data(), 0x11, capacity);
            TransferEngine::publishHostWrite(source.get());
            worker->submitAndWait([&] {
                TransferEngine::prepareDeviceInput(source.get(), source_device, producer);
                TransferEngine::requireDeviceInput(source.get(), source_device, producer);
                TransferEngine::publishDeviceWrite(source.get(), source_device, producer);
            });
            // Admit the original transfer stream and CPU completion resources
            // before the timeline can block the producer. Lazy native setup
            // must never supply this regression's required event ordering.
            const auto warm = transfers.copyActivation(source.get(), destination.get(), DeviceId::cpu(), capacity);
            ASSERT_TRUE(warm.success) << warm.error;
            // General owner construction belongs to the setup caller. Recording,
            // instantiation and actual invocation use its original worker.
            auto graph = Graph::original(original->permit(source_device), producer);
            const auto await = [&] {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
                while (!Graph::observeTerminal(graph))
                {
                    if (std::chrono::steady_clock::now() >= deadline)
                        throw std::runtime_error("Original hosted activation producer did not retire");
                    std::this_thread::yield();
                }
            };
            struct OriginalProducerDrain final
            {
                std::uint64_t *signal;
                std::function<void()> await;
                ~OriginalProducerDrain() noexcept
                {
                    std::atomic_ref<std::uint64_t>(*signal).store(1, std::memory_order_release);
                    try { await(); } catch (...) { std::terminate(); }
                }
            } drain{signal, await};
            worker->submitAndWait([&] {
                auto &definition = Graph::capture(graph);
                ScopedBackendGraphCapture capture(*worker, definition, "hosted activation live-prefix producer");
                ASSERT_TRUE(capture.begin());
                ASSERT_TRUE(backend->streamWaitTimelineSignal64(producer, gate->deviceAlias(source_device),
                    1, source_device.ordinal));
                ASSERT_TRUE(backend->memset(source->gpu_data_ptr(), 0x3f, capacity, source_device.ordinal, producer));
                capture.finish();
                ASSERT_TRUE(definition.prepareRuntimeContextStorage(original->memory()));
                ASSERT_TRUE(definition.instantiate());
            });
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
            std::uint64_t invocation = 0;
            for (int replay = 0; replay < 4; ++replay)
                for (const std::size_t rows : {8u, 3u, 0u, 1u})
                {
                    SCOPED_TRACE(::testing::Message() << "replay=" << replay << " rows=" << rows);
                    std::memset(destination->raw_mutable_data(), 0xa5, capacity);
                    TransferEngine::publishHostWrite(destination.get());
                    // Finish poisoning before holding the next producer. Every
                    // replay distinguishes this graph's write from old bytes.
                    worker->submitAndWait([&] {
                        ASSERT_TRUE(backend->memset(source->gpu_data_ptr(), 0x11, capacity,
                            source_device.ordinal, producer));
                    });
                    await();
                    std::atomic_ref<std::uint64_t>(*signal).store(0, std::memory_order_release);
                    std::jthread release([signal](std::stop_token stop) {
                        std::mutex mutex;
                        std::unique_lock lock(mutex);
                        std::condition_variable_any wake;
                        wake.wait_for(lock, stop, std::chrono::milliseconds(20), [] { return false; });
                        std::atomic_ref<std::uint64_t>(*signal).store(1, std::memory_order_release);
                    });
                    worker->submitAndWait([&] {
                        ASSERT_TRUE(Graph::submit(graph, ++invocation));
                        TransferEngine::publishDeviceWrite(source.get(), source_device, producer);
                    });
                    const std::size_t bytes = rows * 128 * sizeof(float);
                    const auto before = TransferProfiler::getD2HStats().total_bytes.load();
                    const auto copied = transfers.copyActivation(source.get(), destination.get(), DeviceId::cpu(), bytes);
                    ASSERT_TRUE(copied.success) << copied.error;
                    release.request_stop();
                    release.join();
                    EXPECT_EQ(TransferProfiler::getD2HStats().total_bytes.load() - before, bytes);
                    EXPECT_TRUE(source->isDeviceAuthoritative(source_device));
                    EXPECT_FALSE(source->hostValid());
                    const auto *actual = reinterpret_cast<const unsigned char *>(destination->data());
                    for (std::size_t byte = 0; byte < capacity; ++byte)
                        ASSERT_EQ(actual[byte], byte < bytes ? 0x3f : 0xa5) << byte;
                    // Empty copies read no producer bytes. Retire the original
                    // submitted definition before resetting the same native gate.
                    await();
                }
        }
        original->requirePayloadRetired();
        original->retire();
        original->requireExactRetired();
    }

#ifdef HAVE_CUDA
    /** @test CUDA publishes one live CPU prefix after its original delayed producer. */
    TEST(MoEHostedActivationPublication, PendingHostLiveExtent_CUDA)
    {
        provePendingHostActivationExtent(DeviceType::CUDA);
    }
#endif
#ifdef HAVE_ROCM
    /** @test ROCm uses the same exact event and live CPU prefix across retained replay. */
    TEST(MoEHostedActivationPublication, PendingHostLiveExtent_ROCm)
    {
        provePendingHostActivationExtent(DeviceType::ROCm);
    }
#endif

    /** @brief Actual OR loading/admission and Rank host/native family composition. */
    class MoENoPeerHostTail : public ::testing::TestWithParam<NoPeerHostTailCase> {};

    TEST_P(MoENoPeerHostTail, DynamicOwnershipAndLiveActivationExtents)
    {
        const auto test = GetParam();
        const bool distributed = test.stages.size() > 2;
        const auto mpi = MPIContextFactory::global();
        ASSERT_NE(mpi, nullptr);
        ASSERT_EQ(mpi->world_size(), distributed ? 2 : 1);
        const auto inventory = mpi->clusterInventory();
        ASSERT_NE(inventory, nullptr);
        const auto &local_rank = inventory->ranks.at(mpi->rank());
        const auto affinity = NUMATopology::detectLocalNUMANode();
        ASSERT_TRUE(affinity.detection_succeeded);
        ASSERT_EQ(local_rank.cpu.numa_node, affinity.local_numa_node);
        initCPUBackend(local_rank.cpu.numa_node);

        std::vector<int> stage_owners(test.stages.size(), mpi->rank());
        std::vector<DeviceId> stage_devices;
        std::vector<GlobalDeviceAddress> addresses;
        if (distributed)
        {
            // Reuse the same immutable original inventory selection as mixed
            // prepared-weight and generation tests. Local ordinals stay in
            // their actual original rank namespaces.
            const PreparationCase gpu_case{test.stages[0] == DeviceType::CUDA,
                test.stages[1] == DeviceType::CUDA, 1, MTPLane::Disabled};
            const auto selected = placement(gpu_case);
            ASSERT_NE(selected.owners[0], selected.owners[1]);
            for (std::size_t stage = 0; stage < 2; ++stage)
            {
                stage_owners[stage] = selected.owners[stage];
                stage_devices.push_back(selected.devices[stage].front());
                const auto &owner = inventory->ranks.at(stage_owners[stage]);
                requireRankLocalGPUOwnership(owner, stage_devices.back(), "Original mixed no-peer stage");
                addresses.push_back(GlobalDeviceAddress::fromLocalDeviceId(
                    stage_devices.back(), owner.hostname, owner.cpu.numa_node));
            }
            // The host tail belongs to the second GPU's actual socket. Both
            // discovery ranks therefore execute only their own real stages.
            stage_owners.back() = stage_owners[1];
            const auto &tail = inventory->ranks.at(stage_owners.back());
            stage_devices.push_back(DeviceId::cpu());
            addresses.push_back(GlobalDeviceAddress::cpu(tail.cpu.numa_node, tail.hostname));
        }
        else
        {
            const auto observed = observedSingletonPipelinePlacement(test.stages.front());
            stage_devices = {observed.gpu, DeviceId::cpu()};
            addresses = {GlobalDeviceAddress::fromLocalDeviceId(observed.gpu, local_rank.hostname, local_rank.cpu.numa_node),
                         GlobalDeviceAddress::cpu(observed.cpu_numa_node, local_rank.hostname)};
        }
        SharedPipelinePlanningGGUFFixture fixture(distributed, static_cast<std::uint32_t>(test.stages.size()), true);
        std::vector<std::string> arguments{"no-peer-host-tail", "-m", fixture.path(), "--mpi-procs",
            std::to_string(mpi->world_size()), "--backend", "heterogeneous", "--context-length", "512", "--no-mtp",
            "--activation-precision", "fp32", "--kv-cache-precision", "fp16",
            "--moe-residency-maintenance", "dynamic"};
        for (std::size_t stage = 0; stage < test.stages.size(); ++stage)
        {
            const auto domain = "s" + std::to_string(stage);
            arguments.insert(arguments.end(), {"--define-domain",
                domain + "=" + addresses[stage].toString() +
                    ";scope=rank_local;backend=auto;owner=" + std::to_string(stage_owners[stage]), "--pp-stage",
                std::to_string(stage) + "=" + domain + ":" + std::to_string(stage) + "-" + std::to_string(stage)});
        }
        std::vector<char *> argv;
        for (auto &argument : arguments) argv.push_back(argument.data());
        auto config = OrchestrationConfigParser{}.parseArgs(argv.size(), argv.data());
        config.prefill_max_bucket_size = 64;
        // Separate prefix archival traffic from activation-economy proof.
        // Prefix/reset ownership remains covered by the retained pipeline family.
        config.prefix_cache.enabled = false;
        publishOrchestrationStartupPolicy(config);
        OrdinaryGraphEvidence evidence("generation,forward_graph,moe_overlay_residency,transfer,request_admission");
        PipelinePreparedFixtureLifetime lifetime(config, true, mpi);
        NoPeerOriginalFailureBoundary failure_boundary{distributed ? mpi : nullptr};
        auto &runner = lifetime.runner();
        ASSERT_TRUE(runner.initialize()) << runner.lastError();
        const bool source = mpi->rank() == stage_owners.back();
        if (source) ASSERT_EQ(runner.primaryDeviceId(), DeviceId::cpu());
        EXPECT_EQ(runner.executionPlan().runtime.moe_rebalance.mode, MoERebalanceRuntimeMode::Dynamic);
        const auto &contract = lifetime.exportContract();
        ASSERT_TRUE(contract);
        ASSERT_TRUE(contract->pipeline_prepared_plan);
        const auto &prepared_stages = contract->pipeline_prepared_plan->stages();
        const auto &admitted_stages = contract->pipeline_prepared_plan->admission()->stages();
        const auto owned = static_cast<std::size_t>(std::count(stage_owners.begin(), stage_owners.end(), mpi->rank()));
        ASSERT_EQ(prepared_stages.size(), owned);
        ASSERT_EQ(admitted_stages.size(), owned);
        std::uint64_t owned_stage_mask = 0;
        const auto check_owners = [&]
        {
            std::optional<MoEOptimizationMovementTopology> topology;
            std::optional<MoEOptimizationStatus> aggregate;
            if (!distributed)
            {
                topology = runner.moeOptimizationMovementTopology();
                aggregate = runner.moeOptimizationStatus();
                ASSERT_EQ(topology->authority, MoEOptimizationAuthority::Pipeline);
                EXPECT_EQ(topology->axes, MoEOptimizationMovementAxes::None);
                ASSERT_EQ(topology->stages.entries().size(), test.stages.size());
                ASSERT_EQ(aggregate->authority, MoEOptimizationAuthority::Pipeline);
                ASSERT_EQ(aggregate->stages.entries().size(), test.stages.size());
            }
            for (std::size_t local = 0; local < prepared_stages.size(); ++local)
            {
                const auto &prepared = prepared_stages[local];
                const auto &admitted = admitted_stages[local].topology();
                const int stage = admitted.stageId();
                ASSERT_GE(stage, 0);
                ASSERT_LT(static_cast<std::size_t>(stage), test.stages.size());
                ASSERT_EQ(stage_owners[stage], mpi->rank());
                owned_stage_mask |= std::uint64_t{1} << stage;
                const auto owners = MoEExpertOwnerMap::build(*prepared.placement);
                const auto proof = MoEOverlayNoPeerResidency::seal(*prepared.placement, owners, prepared.metadata);
                ASSERT_TRUE(proof);
                const auto device = stage_devices[stage];
                EXPECT_EQ(proof->device(), device);
                EXPECT_EQ(proof->worldRank(), mpi->rank());
                EXPECT_EQ(admitted.scope().first_layer, stage);
                EXPECT_EQ(admitted.scope().last_layer, stage + 1);
                EXPECT_EQ(admitted.scope().has_lm_head, static_cast<std::size_t>(stage + 1) == test.stages.size());
                EXPECT_EQ(prepared.placement->first_model_layer, stage);
                EXPECT_EQ(MoEOverlayCapacityAdmission::selectMigrationStorage(*prepared.placement,
                    MoERebalanceRuntimeMode::Dynamic), MoEOverlayMigrationStorageKind::NoPeer);
                const auto value = distributed ? runner.localPipelineStageOptimizationStatus(stage) :
                    std::optional<MoEOptimizationStatus>(aggregate->stages.entries()[local].value);
                ASSERT_TRUE(value);
                EXPECT_EQ(value->authority, device.is_gpu() ? MoEOptimizationAuthority::Device : MoEOptimizationAuthority::Host);
                EXPECT_EQ(value->state, MoEOptimizationLifecycleState::MovementDisabled);
                EXPECT_EQ(value->completed_movement.physical_bytes, 0u);
                EXPECT_EQ(value->completed_movement.commands, 0u);
                EXPECT_EQ(value->completed_decision_windows, 0u);
                EXPECT_EQ(contract->physical_memory_authority->plannedBytes(device,
                    PhysicalMemoryOwner::ExpertMigrationStaging), 0u);
                if (topology)
                {
                    const auto &scope = topology->stages.entries()[local].identity;
                    ASSERT_EQ(scope.participants().size(), 1u);
                    EXPECT_EQ(scope.participants().front(), addresses[stage]);
                    EXPECT_EQ(scope.firstLayer(), stage);
                    EXPECT_EQ(scope.mainLastLayer(), stage + 1);
                    EXPECT_EQ(scope.terminal(), static_cast<std::size_t>(stage + 1) == test.stages.size());
                }
            }
        };
        ASSERT_NO_FATAL_FAILURE(check_owners());
        SamplingParams sampling;
        sampling.temperature = 0.0F;
        sampling.seed = 123;
        if (distributed)
        {
            runner.setMPICoordinatedMode(true);
            EXPECT_EQ(runner.coordinatedRootRank(), stage_owners.back());
            lifetime.armCommandSource(source);
        }
        const auto capture_inventory = [] {
            std::map<std::string, std::pair<std::uint64_t, double>> result;
            for (const auto &record : PerfStatsCollector::snapshot())
                if (record.domain == "forward_graph" &&
                    record.name == "full_graph_capture_executable_nodes" &&
                    record.tags.contains("context") && record.tags.at("context") == "main_decode")
                {
                    // Successful native instantiation publishes this metadata
                    // even when setup deliberately never submits the graph.
                    // Keep construction multiplicity as well as node totals:
                    // a recapture with identical nodes must still fail reuse.
                    EXPECT_EQ(record.kind, PerfStatRecord::Kind::Counter);
                    EXPECT_EQ(record.phase, "decode");
                    EXPECT_EQ(record.tags.at("type"), "materialized_unlaunched_executable");
                    EXPECT_GT(record.count, 0u);
                    EXPECT_GT(record.value, 0.0);
                    auto &[captures, nodes] = result[record.device];
                    captures += record.count;
                    nodes += record.value;
                }
            return result;
        };
        // These terminal observations prove participant-local ingress; no
        // execution decision reads them and no device state is mirrored here.
        const auto decode_ingress_bytes = [](DeviceId device, const char *name) {
            double value = 0;
            for (const auto &record : PerfStatsCollector::snapshot())
                if (record.domain == "request_admission" && record.name == name &&
                    record.phase == "decode" && record.device == device.toString())
                    value += record.value;
            return value;
        };
        const auto local_gpu = [&](std::size_t stage)
        {
            return stage_devices[stage].is_gpu() && stage_owners[stage] == mpi->rank();
        };
        const auto original_captures = capture_inventory();
        std::size_t gpu_stages = 0;
        for (std::size_t stage = 0; stage < test.stages.size(); ++stage)
            if (local_gpu(stage)) ++gpu_stages;
        ASSERT_EQ(original_captures.size(), gpu_stages)
            << "Initialization must materialize every actually local native stage";
        for (std::size_t stage = 0; stage < test.stages.size(); ++stage)
            if (local_gpu(stage)) EXPECT_TRUE(original_captures.contains(stage_devices[stage].toString()));

        std::array<std::uint64_t, 6> terminal{};
        TransferProfiler::reset();
        const auto initial_ingress = [&]
        {
            std::map<std::string, std::array<double, 3>> result;
            for (std::size_t stage = 0; stage < test.stages.size(); ++stage)
                if (local_gpu(stage))
                    result[stage_devices[stage].toString()] = {
                        decode_ingress_bytes(stage_devices[stage], "device_input_token_bytes"),
                        decode_ingress_bytes(stage_devices[stage], "device_input_position_bytes"),
                        decode_ingress_bytes(stage_devices[stage], "device_input_padding_bytes")};
            return result;
        }();
        if (!source)
        {
            runner.runMPIWorkerLoop();
            terminal[2] = 1;
        }
        else
        {
            runner.setSamplingParams(sampling);
            runner.setStopTokens({});
            for (const int rows : {65, 3, 1})
            {
                SCOPED_TRACE(rows);
                runner.clearCache();
                std::vector<int32_t> tokens(static_cast<std::size_t>(rows), 11);
                const auto previous_d2h = TransferProfiler::getD2HStats().total_bytes.load();
                ASSERT_TRUE(runner.prefill(tokens)) << runner.lastError();
                // This model has H256. CPU owns the final sampler, so every D2H
                // byte in this diagnostic-free/no-prefix call is a real boundary
                // activation. Each GPU boundary must copy only root live rows,
                // including the final partial64+1 and large-to-small request reset.
                const auto expected = static_cast<std::uint64_t>(gpu_stages) * rows * 256u * sizeof(float);
                EXPECT_EQ(TransferProfiler::getD2HStats().total_bytes.load() - previous_d2h, expected);
                std::vector<std::tuple<DeviceId, double, double, double>> ingress_before;
                for (std::size_t stage = 0; stage < test.stages.size(); ++stage)
                    if (local_gpu(stage))
                    {
                        const auto device = stage_devices[stage];
                        ingress_before.emplace_back(device,
                            decode_ingress_bytes(device, "device_input_token_bytes"),
                            decode_ingress_bytes(device, "device_input_position_bytes"),
                            decode_ingress_bytes(device, "device_input_padding_bytes"));
                    }
                // The CPU-tail serial API emits one token per decodeStep; its
                // budget is an upper bound, not a request to batch scalar calls.
                // Retire each actual sample before admitting the next command,
                // matching generate(). The first call consumes prefill logits;
                // only the second consumes one true history-bearing pipeline row.
                for (const int step : {0, 1})
                {
                    SCOPED_TRACE(step);
                    runner.setDecodeStepTokenBudget(1);
                    const auto result = runner.decodeStep();
                    ASSERT_TRUE(result.success()) << result.error;
                    ASSERT_EQ(result.tokens.size(), 1u);
                    terminal[1] += result.tokens.size();
                    ASSERT_FALSE(result.is_complete);
                    const auto history_bytes = static_cast<std::uint64_t>(step) *
                        gpu_stages * 256u * sizeof(float);
                    EXPECT_EQ(TransferProfiler::getD2HStats().total_bytes.load() - previous_d2h,
                              expected + history_bytes)
                        << "Only the history call transfers one live activation row per GPU boundary";
                    // Every GPU, including a non-embedding middle participant,
                    // receives exactly the live scalar token on the history call.
                    // Its captured KV root authors positions; no position shadow
                    // or padded token capacity crosses host admission.
                    for (const auto &[device, token_bytes, position_bytes, padding_bytes] : ingress_before)
                    {
                        EXPECT_DOUBLE_EQ(decode_ingress_bytes(device, "device_input_token_bytes") - token_bytes,
                                         static_cast<double>(step * sizeof(int32_t)));
                        EXPECT_DOUBLE_EQ(decode_ingress_bytes(device, "device_input_position_bytes"), position_bytes);
                        EXPECT_DOUBLE_EQ(decode_ingress_bytes(device, "device_input_padding_bytes"), padding_bytes);
                    }
                    ASSERT_TRUE(runner.maybeApplyMoERebalance(result.tokens.size())) << runner.lastError();
                }
                for (std::size_t stage = 0; stage < test.stages.size(); ++stage)
                    if (local_gpu(stage)) EXPECT_GT(evidence.completedForwards(stage_devices[stage]), 0.0);
                EXPECT_EQ(capture_inventory(), original_captures) << "Reset must preserve the retained native graph family";
                ASSERT_NO_FATAL_FAILURE(check_owners());
                ++terminal[0];
            }
            runner.clearCache();
            const auto before_empty_d2h = TransferProfiler::getD2HStats().total_bytes.load();
            const auto before_empty_h2d = TransferProfiler::getH2DStats().total_bytes.load();
            EXPECT_FALSE(runner.prefill({})) << "Empty input is rejected before a PP activation transfer";
            EXPECT_EQ(TransferProfiler::getD2HStats().total_bytes.load(), before_empty_d2h);
            EXPECT_EQ(TransferProfiler::getH2DStats().total_bytes.load(), before_empty_h2d);
            lifetime.closeWorkers();
        }
        // Both original ranks have completed their public command lifetime.
        // Join only passive local observations after the real worker loop exits.
        const auto physical_d2h = TransferProfiler::getD2HStats().total_bytes.load();
        EXPECT_EQ(physical_d2h, gpu_stages * (65u + 3u + 1u + 3u) * 256u * sizeof(float));
        EXPECT_EQ(capture_inventory(), original_captures);
        ASSERT_NO_FATAL_FAILURE(check_owners());
        for (std::size_t stage = 0; stage < test.stages.size(); ++stage)
            if (local_gpu(stage))
            {
                const auto device = stage_devices[stage];
                EXPECT_GT(evidence.completedForwards(device), 0.0);
                const auto &initial = initial_ingress.at(device.toString());
                EXPECT_DOUBLE_EQ(decode_ingress_bytes(device, "device_input_token_bytes") - initial[0], 3u * sizeof(int32_t));
                EXPECT_DOUBLE_EQ(decode_ingress_bytes(device, "device_input_position_bytes"), initial[1]);
                EXPECT_DOUBLE_EQ(decode_ingress_bytes(device, "device_input_padding_bytes"), initial[2]);
                if (distributed)
                    for (const auto phase : {ForwardExecutionPhase::Prefill, ForwardExecutionPhase::Decode})
                    {
                        std::vector<std::uint64_t> words;
                        if (phase == ForwardExecutionPhase::Prefill)
                            for (const std::uint64_t rows : {64u, 1u, 3u, 1u})
                                words.insert(words.end(), {rows, rows * 256u * sizeof(float)});
                        else
                            for (int request = 0; request < 3; ++request)
                                words.insert(words.end(), {1u, 256u * sizeof(float)});
                        const auto records = PerfStatsCollector::snapshot({"transfer"});
                        const auto record = std::find_if(records.begin(), records.end(), [&](const auto &value) {
                            return value.name == "hosted_pipeline_activation_bytes" && value.device == device.toString() &&
                                value.phase == (phase == ForwardExecutionPhase::Prefill ? "prefill" : "decode");
                        });
                        ASSERT_NE(record, records.end());
                        EXPECT_EQ(record->tags.at("source_stage"), std::to_string(stage));
                        EXPECT_EQ(record->tags.at("consumer_stage"), std::to_string(stage + 1));
                        EXPECT_EQ(record->tags.at("transport"), stage_owners[stage] == stage_owners[stage + 1] ? "local" : "mpi");
                        EXPECT_TRUE(PerfStatsCollector::matchesCounterSequence(*record, words, 2));
                        EXPECT_DOUBLE_EQ(record->value, (phase == ForwardExecutionPhase::Prefill ? 69u : 3u) * 256u * sizeof(float));
                    }
            }
        runner.shutdown();
        const auto &retirement = lifetime.finish();
        ASSERT_TRUE(retirement.has_value());
        EXPECT_EQ(retirement->retiredDeviceCount(), gpu_stages);
        terminal[3] = owned_stage_mask;
        terminal[4] = retirement->retiredDeviceCount();
        terminal[5] = physical_d2h;
        if (distributed)
        {
            std::array<std::uint64_t, 12> receipts{};
            mpi->allgather_bytes(terminal.data(), receipts.data(), sizeof(terminal));
            EXPECT_EQ(receipts[0] + receipts[6], 3u);
            EXPECT_EQ(receipts[1] + receipts[7], 6u);
            EXPECT_EQ(receipts[2] + receipts[8], 1u);
            EXPECT_EQ(receipts[3] | receipts[9], 7u);
            EXPECT_EQ(receipts[3] & receipts[9], 0u);
            EXPECT_EQ(receipts[4] + receipts[10], 2u);
            EXPECT_EQ(receipts[5] + receipts[11], 2u * 72u * 256u * sizeof(float));
        }

    }

    INSTANTIATE_TEST_SUITE_P(Topologies, MoENoPeerHostTail,
        ::testing::ValuesIn(noPeerHostTailCases()), [](const auto &info) { return info.param.name; });

    TEST_P(MoEPipelineWeightPreparation, NativePreparationContract)
    {
        const auto test = GetParam();
        SharedPipelinePlanningGGUFFixture fixture(test.distributedDiscovery(), test.stageCount(), test.selectsMI50Owner());
        const auto execution = PreparationExecution::admit(test);
        if (!execution.mpi) return; // Original inactive rank only joins fixture terminal retirement.
        auto config = configuration(test, fixture.path(), execution);
        const auto &physical = execution.physical;
        const auto mpi = execution.mpi;
        const bool distributed = mpi->world_size() > 1;
        const auto &devices = physical.devices;
        const int rank = mpi->rank();
        publishOrchestrationStartupPolicy(config);
        OrdinaryGraphEvidence maintenance_evidence("generation,forward_graph,moe_rebalance");
        PipelinePreparedFixtureLifetime lifetime(config, true, mpi);
        auto &runner=lifetime.runner();
        const bool initialized = runner.initialize();
        ASSERT_TRUE(initialized) << runner.lastError();
        ASSERT_TRUE(runner.isInitialized());
        EXPECT_EQ(runner.executionPlan().runtime.mtp.enabled, test.lane != MTPLane::Disabled);
        if (test.lane != MTPLane::Disabled)
            EXPECT_EQ(runner.executionPlan().runtime.mtp.depth_policy.mode,
                test.lane == MTPLane::Fixed ? MTPDepthPolicyMode::Fixed : MTPDepthPolicyMode::Dynamic);
        if (test.first_cuda == test.last_cuda && config.moe_rebalance.mode == MoERebalanceRuntimeMode::Dynamic)
            for (int stage = 0; stage < test.stageCount(); ++stage)
                if (physical.owners[stage] == rank && test.stageWidth(stage) > 1)
                    for (const auto device : devices[stage])
                    {
                        double initializations = 0;
                        for (const auto &record : PerfStatsCollector::snapshot())
                            if (record.domain == "moe_rebalance" &&
                                record.name == "device_controller_first_request_initializations" &&
                                record.device == device.toString())
                            {
                                EXPECT_EQ(record.tags.at("ordering"), "request_reset_event");
                                EXPECT_NE(record.tags.at("workspace_generation"), "0");
                                initializations += record.value;
                            }
                        EXPECT_EQ(initializations, 1.0)
                            << "Every original stage controller must bind its complete workspace exactly once before capture";
                    }
        const auto &contract = lifetime.exportContract();
        ASSERT_TRUE(contract.has_value())
            << "The complete original-rank graph must publish its shared source lifecycle before export";
        EXPECT_EQ(contract->context.get(), runner.modelContextForDiagnostics());
        ASSERT_NE(contract->pipeline_prepared_plan, nullptr);
        ASSERT_NE(contract->physical_memory_authority, nullptr);
        {
        const auto manager = contract->context->concreteWeightManager();
        ASSERT_NE(manager, nullptr);
        EXPECT_TRUE(manager->lifecycleGates().graph_materialization_complete);
        EXPECT_TRUE(manager->lifecycleGates().canReleaseHostData());
        EXPECT_EQ(manager->preparedWeightStore()->physicalMemoryAuthority(), contract->physical_memory_authority);
        const auto &registry = manager->expertGemmRegistry();
        const auto &stages = contract->pipeline_prepared_plan->stages();
        std::vector<int> local_stages;
        for (int stage = 0; stage < test.stageCount(); ++stage)
            if (physical.owners[stage] == rank) local_stages.push_back(stage);
        ASSERT_EQ(stages.size(), local_stages.size());
        const int owned_stage_count = static_cast<int>(local_stages.size());
        std::array<int, 2> owned_stage_counts{};
        if (distributed)
        {
            mpi->allgather_bytes(&owned_stage_count, owned_stage_counts.data(), sizeof(owned_stage_count));
            EXPECT_EQ(owned_stage_counts[0] + owned_stage_counts[1], test.stageCount());
        }
        else EXPECT_EQ(owned_stage_count, test.stageCount());
        for (size_t local = 0; local < stages.size(); ++local)
        {
            const int stage = local_stages[local];
            // The retained stage keeps its original world ownership. Inspect
            // it from this exact MPI rank, retaining the strict local-root guard.
            const auto runtime = resolveMoEExpertOverlayRuntimePlan(stages[local].placement,
                {.current_world_rank = rank});
            ASSERT_NE(runtime, nullptr);
            ASSERT_EQ(runtime->currentWorldRank(), rank);
            if (test.selectsMI50Owner())
            {
                ASSERT_NE(stages[local].placement, nullptr);
                EXPECT_EQ(stages[local].placement->owner_order, test.owner_order);
                EXPECT_EQ(stages[local].placement->residency_policy,
                    defaultRoutedExpertResidencyPolicy(test.movement));
                ASSERT_EQ(stages[local].placement->domains.size(), 1u);
                const auto &domain = stages[local].placement->domains.front();
                EXPECT_EQ(domain.participants.size(), static_cast<size_t>(test.stageWidth(stage)));
                EXPECT_EQ(domain.routed_compute_policy, RoutedExpertComputePolicy::Apportioned);
                EXPECT_EQ(domain.routed_phase_policy, RoutedExpertPhasePolicy::Uniform);
            }
            const auto preparation = MoEExpertOverlayPreparationPlan::build(*runtime, contract->context->concreteLoader());
            std::set<int> prepared_layers;
            std::map<std::tuple<std::string, DeviceId, int>, size_t> expected_entries;
            for (const auto &request : preparation.requests())
            {
                prepared_layers.insert(request.layer);
                if (test.selectsMI50Owner()) EXPECT_FALSE(request.projection_ownership.has_value());
                ASSERT_TRUE(request.participant_world_rank_known);
                ASSERT_EQ(request.participant_world_rank, rank);
                // A participant key and its domain alias retain the same
                // prepared engine. Neither is a second physical allocation.
                expected_entries[{request.domain_name, request.device, request.layer}] += 2;
                EXPECT_NE(registry.getEngineForParticipant(request.domain_name, request.device,
                    request.participant_world_rank, request.participant_index, request.layer,
                    request.expert_id, request.role, request.projection_ownership), nullptr)
                    << request.domain_name << " layer=" << request.layer << " expert=" << request.expert_id;
            }
            std::set<int> expected_layers{stage};
            if (stage + 1 == test.stageCount() && test.lane != MTPLane::Disabled)
                expected_layers.insert(test.stageCount());
            EXPECT_EQ(prepared_layers, expected_layers);
            for (const auto &[key, count] : expected_entries)
            {
                const auto &[domain, device, layer] = key;
                EXPECT_EQ(registry.countEnginesForLayerInDomain(domain, device, layer), count);
            }
            for (const auto device : devices[stage])
            {
                const auto &authority = *contract->physical_memory_authority;
                const auto live = authority.claimedBytes(device, PhysicalMemoryOwner::RoutedExpertWeights,
                    PhysicalMemoryMaterializationKind::NewAllocation);
                EXPECT_GT(live, 0u) << device.toString();
                EXPECT_LE(live, authority.plannedBytes(device, PhysicalMemoryOwner::RoutedExpertWeights));
                for (int layer = 0; layer <= test.stageCount(); ++layer)
                    if (!expected_layers.contains(layer))
                        for (const auto &domain : runtime->domains())
                            EXPECT_EQ(registry.countEnginesForLayerInDomain(domain.name, device, layer), 0u);
                // The overlay is the only routed owner. A legacy generic
                // preparation must not publish another full expert bank.
                EXPECT_EQ(registry.countEnginesForDevice(device), 0u);
            }
        }
        runner.shutdown();
        EXPECT_FALSE(runner.isInitialized());
        ASSERT_NE(contract->reuse_authority, nullptr);
        std::string retention_error;
        const auto retention = contract->reuse_authority->sealedDeviceMemoryRetention(&retention_error);
        ASSERT_TRUE(retention.has_value()) << retention_error;
        std::set<DeviceId> expected_devices, retained_devices;
        for (const int stage : local_stages)
            expected_devices.insert(devices[stage].begin(), devices[stage].end());
        for (const auto &row : *retention)
        {
            EXPECT_TRUE(retained_devices.insert(row.device).second);
            EXPECT_GT(row.prepared_weight_bytes, 0u);
            EXPECT_EQ(row.prepared_weight_bytes, manager->retainedPreparedDeviceBytes(row.device));
        }
        EXPECT_EQ(retained_devices, expected_devices);
        } // Release prepared-manager/registry inspection borrows before the actual final-owner API.
        const auto &retirement=lifetime.finish();
        ASSERT_TRUE(retirement.has_value());
        size_t expected_retired = 0;
        for (int stage = 0; stage < test.stageCount(); ++stage)
            if (physical.owners[stage] == rank) expected_retired += devices[stage].size();
        EXPECT_EQ(retirement->retiredDeviceCount(), expected_retired);
    }

    INSTANTIATE_TEST_SUITE_P(Topologies, MoEPipelineWeightPreparation,
        ::testing::ValuesIn(cases()), [](const auto &info) { return info.param.name(); });

#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
    /** @brief Public sparse request lifecycle on each supported pipeline placement. */
    class MoEPipelineRequestLifecycle : public MoEPipelineWeightPreparation {};

    TEST_P(MoEPipelineRequestLifecycle, FirstDecodeResetAndPrefixReuse)
    {
        const auto test = GetParam();
        SharedPipelinePlanningGGUFFixture fixture(test.distributedDiscovery(), test.stageCount(), test.selectsMI50Owner());
        const auto execution = PreparationExecution::admit(test);
        if (!execution.mpi) return; // Original inactive rank only joins fixture terminal retirement.
        const auto &physical = execution.physical;
        const auto mpi = execution.mpi;
        const bool distributed = mpi->world_size() > 1;
        const bool source = mpi->rank() == physical.owners.back();
        std::vector<DeviceId> upstream_devices;
        for (int stage = 0; stage + 1 < test.stageCount(); ++stage)
            if (physical.owners[stage] == mpi->rank())
                upstream_devices.insert(upstream_devices.end(), physical.devices[stage].begin(), physical.devices[stage].end());
        const bool follower = !upstream_devices.empty();
        auto config = configuration(test, fixture.path(), execution);
        config.prefix_cache.ram_budget_bytes = 64ull * 1024 * 1024;
        config.prefix_cache.disk_budget_bytes = 0;
        publishOrchestrationStartupPolicy(config);
        OrdinaryGraphEvidence evidence(test.selectsMI50Owner()
            ? "generation,forward_graph,mtp,prefix_cache,moe_overlay_residency,moe_rebalance,moe_overlay_epoch"
            : "generation,forward_graph,mtp,prefix_cache");
        OrchestrationRunner runner(mpi, config, createExecutionPlanBuilder());
        ASSERT_TRUE(runner.initialize()) << runner.lastError();

        const auto publications = []
        {
            double count = 0;
            for (const auto &record : PerfStatsCollector::snapshot())
                if (record.domain == "forward_graph" &&
                    record.name == "moe_overlay_collective_request_generation")
                    count += record.value;
            return count;
        };
        // Check the lifecycle boundary before any collective launch. The old
        // single-plan-only publisher returned success while every PP child
        // retained generation zero; its first hosted draft then failed fatally.
        ASSERT_EQ(publications(), 1.0);
        SamplingParams sampling;
        sampling.temperature = 0.8F;
        sampling.top_k = 20;
        sampling.seed = 123;
        runner.setSamplingParams(sampling);
        runner.setStopTokens({});
        // This fixture has one full-attention query program. Resolve the active
        // model-selected policy while retaining the authored physical capacity;
        // the automatic CLI activation sentinel is not the admitted MTP policy.
        const auto admitted_mtp = composeMTPRequestConfig(config.mtp, runner.mtpRequestPolicy());
        using FollowerMaterializationKey = std::tuple<std::string, std::string, std::string, std::string>;
        std::map<FollowerMaterializationKey, double> expected_follower_materializations;
        if (test.lane != MTPLane::Disabled && follower)
        {
            const auto policy = test.first_cuda && test.last_cuda && test.stageCount() == 2 && test.stageWidth(0) == 1 && test.stageWidth(1) == 1
                ? DeviceGenerationExecutionPolicy::NativeConditionalGraph
                : DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions;
            const auto family = resolveMTPVerifierGraphFamily(mtpVerifierExecutionRole(policy));
            const MTPGraphOwnerPlan owners(admitted_mtp, family, MTPGenerationParentRole::PipelineFollower);
            for (const auto participant : upstream_devices)
            {
                const auto device = participant.toString();
                if (policy == DeviceGenerationExecutionPolicy::NativeConditionalGraph)
                    expected_follower_materializations[{device, deviceGenerationExecutionPolicyName(policy), "", ""}] = 1;
                else
                    for (size_t width = 0; width < owners.generationParentWidthCount(); ++width)
                    {
                        const int rows = mtpVerifierGraphFamilyWidthAt(width,
                            owners.verifierRowsPerRequest(), family);
                        expected_follower_materializations[{device, deviceGenerationExecutionPolicyName(policy),
                            "exact_rows", std::to_string(rows)}] = 1;
                    }
            }
        }
        const auto follower_materializations = [](const char *name = "pipeline_mtp_follower_graph_materializations")
        {
            std::map<FollowerMaterializationKey, double> result;
            for (const auto &record : PerfStatsCollector::snapshot())
                if (record.domain == "generation" && record.name == name)
                {
                    EXPECT_EQ(record.kind, PerfStatRecord::Kind::Counter);
                    EXPECT_EQ(record.phase, "graph_setup");
                    const auto tag = [&](const char *name)
                    {
                        const auto value = record.tags.find(name);
                        return value == record.tags.end() ? std::string{} : value->second;
                    };
                    result[{record.device, tag("execution"), tag("family"), tag("physical_rows")}] += record.value;
                }
            return result;
        };
        std::vector<int32_t> prompt(96);
        for (size_t i = 0; i < prompt.size(); ++i)
            prompt[i] = 11 + static_cast<int32_t>(i);
        for (int request = 0; request < 3; ++request)
        {
            SCOPED_TRACE(::testing::Message() << "request=" << request);
            if (request != 0)
            {
                EXPECT_EQ(follower_materializations(), expected_follower_materializations);
                const double before = publications();
                runner.clearCache();
                ASSERT_EQ(publications(), before + 1);
                EXPECT_EQ(follower_materializations(), expected_follower_materializations)
                    << "Reset must publish new data without rebuilding any follower width";
            }
            if (request == 2) prompt.push_back(137);
            ASSERT_TRUE(runner.prefill(prompt)) << runner.lastError();
            for (const int budget : {1, 17, 1})
            {
                SCOPED_TRACE(::testing::Message() << "budget=" << budget);
                runner.setDecodeStepTokenBudget(budget);
                const auto result = runner.decodeStep();
                if (distributed)
                {
                    // Retire both public source/follower calls before checking
                    // their roles. Only the actual vocabulary owner may emit
                    // response tokens; success requires the other stage too.
                    const std::array<int32_t, 4> local{source ? 1 : 0, result.success() ? 1 : 0,
                        static_cast<int32_t>(result.tokens.size()), follower ? 1 : 0};
                    std::array<int32_t, 8> receipts{};
                    mpi->allgather_bytes(local.data(), receipts.data(), sizeof(local));
                    EXPECT_EQ(receipts[0] + receipts[4], 1);
                    EXPECT_EQ(receipts[1], 1);
                    EXPECT_EQ(receipts[5], 1);
                    EXPECT_EQ(receipts[2] + receipts[6], budget);
                    EXPECT_EQ(receipts[3] + receipts[7], 1);
                }
                ASSERT_TRUE(result.success()) << result.error;
                ASSERT_EQ(result.tokens.size(), source ? static_cast<size_t>(budget) : 0u);
                for (const int32_t token : result.tokens)
                    EXPECT_GE(token, 0);
                // The serving frontend retires each completed response into
                // maintenance policy before admitting the next outer step.
                ASSERT_TRUE(runner.maybeApplyMoERebalance(budget)) << runner.lastError();
            }
            // The first budget deliberately remains one: no eager verifier
            // warmup may supply a stream to a later cold exact-width child.
            EXPECT_EQ(follower_materializations(), expected_follower_materializations)
                << "Every admitted follower width must materialize exactly once across reset/prefix reuse";
            if (!source) continue;
            const auto summary = runner.requestRuntimeSummary();
            ASSERT_TRUE(summary.prefix_request.enabled);
            ASSERT_FALSE(summary.prefix_request.bypassed) << summary.prefix_request.bypass_reason;
            if (request != 0)
                EXPECT_GT(summary.prefix_request.matched_tokens, 0);
            EXPECT_EQ(summary.mtp_request.enabled, test.lane != MTPLane::Disabled);
            if (test.lane != MTPLane::Disabled)
            {
                EXPECT_GT(summary.mtp_verifier_runs, 0u);
                EXPECT_GT(summary.mtp_request.draft_steps, 0u);
                EXPECT_FALSE(summary.mtp_request.bypassed) << summary.mtp_request.bypass_reason;
                EXPECT_EQ(summary.mtp_request.depth_policy_mode,
                    test.lane == MTPLane::Fixed ? "fixed" : "dynamic");
            }
        }
        const auto follower_reuses = follower_materializations("pipeline_mtp_follower_graph_reuses");
        const auto records = PerfStatsCollector::snapshot();
        if (test.selectsMI50Owner() && test.movement == MoERebalanceRuntimeMode::Dynamic)
        {
            // Authenticate the actual replay stream's dependency for each
            // upstream EP domain. A wait on a child's setup stream cannot
            // certify a parent that launches on another retained stream.
            for (int stage = 0; stage + 1 < test.stageCount(); ++stage)
            {
                if (physical.owners[stage] != mpi->rank() || test.stageWidth(stage) == 1) continue;
                for (const auto device : physical.devices[stage])
                {
                    double launches = 0, waits = 0;
                    for (const auto &record : records)
                    {
                        if (record.device != device.toString()) continue;
                        if (record.domain == "moe_overlay_epoch" && record.name == "pipeline_follower_captured_epoch_terminals")
                            launches += record.value;
                        if (record.domain == "moe_rebalance" && record.name == "device_maintenance_graph_consumer_event_waits")
                        {
                            const auto consumer = record.tags.find("consumer");
                            if (consumer != record.tags.end() && consumer->second == "pipeline_follower_captured_transaction")
                                waits += record.value;
                        }
                    }
                    EXPECT_GT(launches, 0) << device.toString();
                    EXPECT_EQ(waits, launches) << device.toString();
                }
            }
        }
        if (test.selectsMI50Owner())
        {
            double native_endpoints = 0;
            for (const auto &record : records)
                if (record.domain == "generation" && record.name == "pipeline_native_byte_endpoint_materializations")
                {
                    EXPECT_EQ(record.kind, PerfStatRecord::Kind::Counter);
                    EXPECT_EQ(record.phase, "graph_setup");
                    const auto transport = record.tags.find("transport");
                    const auto mapped = record.tags.find("mapped_payload_bytes");
                    ASSERT_NE(transport, record.tags.end());
                    ASSERT_NE(mapped, record.tags.end());
                    EXPECT_EQ(transport->second, "rccl");
                    EXPECT_EQ(mapped->second, "0");
                    native_endpoints += record.value;
                }
            int expected_endpoints = 0;
            for (int boundary = 0; boundary + 1 < test.stageCount(); ++boundary)
                expected_endpoints += 2 * (test.stageWidth(boundary) + test.stageWidth(boundary + 1));
            EXPECT_EQ(native_endpoints, expected_endpoints)
                << "Every activation/metadata edge must retain its native source and consumer frame exactly once";
        }
        double bridges = 0;
        for (const auto &record : records)
            if (record.domain == "forward_graph" && record.name == "pipeline_scalar_input_transactions")
                bridges += record.value;
        EXPECT_EQ(bridges, 1) << "The one-token suffix must traverse the captured pipeline in every MTP mode";
        if (source && test.lane == MTPLane::Disabled && test.stageCount() == 2 && test.stageWidth(0) == 1 && test.stageWidth(1) == 1 && test.first_cuda == test.last_cuda)
        {
            double native_tail_builds = 0;
            for (const auto &record : records)
                if (record.domain == "generation" &&
                    record.name == "ordinary_native_pipeline_nopeer_tail_materializations")
                    native_tail_builds += record.value;
            EXPECT_GT(native_tail_builds, 0)
                << "The real native PP parent must import the singleton overlay tail's sampler and forward lease";
        }
        if (test.lane != MTPLane::Disabled && follower)
        {
            // Every upstream TP participant must execute its complete admitted
            // native parent or hosted transaction after the first verifier. A
            // terminal-only lease omits required follower verification/commit.
            // Reset/prefix restore changes data, never this graph topology.
            for (const auto participant : upstream_devices)
            {
                const auto device=participant.toString();
                // Only the supported homogeneous singleton CUDA topology owns
                // one native conditional PP parent. HIP and explicit mixed
                // boundaries retain their authenticated hosted transactions.
                const auto policy=test.first_cuda && test.last_cuda && test.stageCount() == 2 && test.stageWidth(0) == 1 && test.stageWidth(1) == 1
                    ? DeviceGenerationExecutionPolicy::NativeConditionalGraph
                    : DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions;
                const auto count = [&](const char *name,const char *domain="generation")
                {
                    double value = 0;
                    for (const auto &record : records)
                        if (record.domain == domain && record.name == name && record.device == device)
                            value += record.value;
                    return value;
                };
                double native_launches=0,materializations_for_policy=0;
                for(const auto &record:records)
                {
                    if(isNativeSpeculativeParentLaunch(record,participant))native_launches+=record.value;
                    if(record.domain=="generation" && record.name=="pipeline_mtp_follower_graph_materializations" &&
                        record.device==device)
                        if(const auto execution=record.tags.find("execution");execution!=record.tags.end() &&
                            execution->second==deviceGenerationExecutionPolicyName(policy))
                            materializations_for_policy+=record.value;
                }
                if(policy==DeviceGenerationExecutionPolicy::NativeConditionalGraph)
                {
                    EXPECT_GT(native_launches,0) << device;
                    EXPECT_EQ(count("pipeline_mtp_follower_transaction_submissions"),0) << device;
                    EXPECT_EQ(count("device_generation_dispatch_tickets_observed","mtp"),0) << device;
                }
                else
                {
                    EXPECT_GT(count("pipeline_mtp_follower_transaction_submissions"),0) << device;
                    EXPECT_EQ(native_launches,0) << device;
                }
                EXPECT_GE(count("pipeline_mtp_follower_graph_reuses"), 2) << device;
                const auto expected_count = std::count_if(expected_follower_materializations.begin(),
                    expected_follower_materializations.end(), [&](const auto &entry)
                    { return std::get<0>(entry.first) == device; });
                EXPECT_EQ(count("pipeline_mtp_follower_graph_materializations"), expected_count) << device;
                EXPECT_EQ(materializations_for_policy, expected_count) << device;
                if (policy == DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions)
                {
                    size_t reused_widths = 0;
                    for (const auto &[key, value] : follower_reuses)
                        if (std::get<0>(key) == device)
                        {
                            ++reused_widths;
                            EXPECT_TRUE(expected_follower_materializations.contains(key))
                                << "Reuse must name an admitted exact-width complete parent on " << device;
                            EXPECT_GE(value, 2)
                                << "Each retained width must reuse across both request resets on " << device
                                << " rows=" << std::get<3>(key);
                        }
                    EXPECT_EQ(reused_widths, expected_count) << device;
                }
            }
        }
        // These are immutable setup/teardown observations from the actual
        // admitted family, including neutral endpoints in inactive stages.
        // Infer neither a participant population nor a controller from width.
        const auto maintenance_events = [](const char *name, const char *phase)
        {
            std::map<std::string, double> by_device;
            for (const auto &record : PerfStatsCollector::snapshot())
                if (record.domain == "generation" && record.name == name)
                {
                    EXPECT_EQ(record.phase, phase);
                    EXPECT_FALSE(record.device.empty());
                    const auto bytes = record.tags.find("frame_bytes");
                    EXPECT_NE(bytes, record.tags.end());
                    if (bytes != record.tags.end()) EXPECT_EQ(bytes->second, "16");
                    EXPECT_GT(record.value, 0);
                    by_device[record.device] += record.value;
                }
            return by_device;
        };
        const auto original_maintenance = maintenance_events("pipeline_maintenance_budget_recordings", "graph_setup");
        runner.shutdown();
        EXPECT_FALSE(runner.isInitialized());
        const auto retired_maintenance = maintenance_events("pipeline_maintenance_budget_retirements", "graph_teardown");
        EXPECT_EQ(retired_maintenance, original_maintenance)
            << "Every actual maintenance recording must release its graph, controller and exact scalar frame";
        runner.shutdown();
        EXPECT_EQ(maintenance_events("pipeline_maintenance_budget_retirements", "graph_teardown"), retired_maintenance)
            << "Repeated shutdown cannot retire an original controller/frame twice";
        if (test.selectsMI50Owner())
        {
            const auto status = runner.moeOptimizationStatus();
            const auto ledger = runner.moeOptimizationMovementLedger();
            ASSERT_EQ(status.authority, MoEOptimizationAuthority::Pipeline);
            ASSERT_EQ(status.stages.entries().size(), static_cast<size_t>(test.stageCount()));
            ASSERT_TRUE(ledger.complete());
            ASSERT_EQ(ledger.stages.entries().size(), static_cast<size_t>(test.stageCount()));
            for (int stage = 0; stage < test.stageCount(); ++stage)
            {
                const auto &value = status.stages.entries()[stage].value;
                const auto &journal = ledger.stages.entries()[stage].value;
                EXPECT_EQ(value.authority, MoEOptimizationAuthority::Device);
                EXPECT_NE(value.state, MoEOptimizationLifecycleState::Failed);
                if (test.movement == MoERebalanceRuntimeMode::Off || test.stageWidth(stage) == 1)
                {
                    EXPECT_EQ(value.completed_decision_windows, 0u);
                    EXPECT_EQ(value.completed_movement.physical_bytes, 0u);
                    EXPECT_TRUE(journal.edges.empty());
                    EXPECT_TRUE(journal.device_publications.empty());
                }
                else
                {
                    EXPECT_GT(value.completed_decision_windows, 0u);
                    ASSERT_TRUE(value.last_decision.has_value());
                    EXPECT_TRUE(value.last_decision->valid());
                }
                for (const auto &edge : journal.edges) EXPECT_TRUE(edge.valid());
                for (const auto &economy : journal.economy) EXPECT_TRUE(economy.valid());
                for (const auto &publication : journal.device_publications) EXPECT_TRUE(publication.valid());
            }
        }
    }

    /** @return Every admitted topology retains request/reset/prefix coverage. */
    std::vector<PreparationCase> requestCases() { return cases(); }

    INSTANTIATE_TEST_SUITE_P(Topologies, MoEPipelineRequestLifecycle,
        ::testing::ValuesIn(requestCases()), [](const auto &info) { return info.param.name(); });
#endif
}
}
