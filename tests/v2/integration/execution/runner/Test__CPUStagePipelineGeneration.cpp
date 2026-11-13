/**
 * @file Test__CPUStagePipelineGeneration.cpp
 * @brief Real original two-socket CPU TP/EP stage followed by its CPU singleton tail.
 *
 * Both original ranks remain genuine CPU-stage participants. The public CPU
 * coordinated source and original MPI worker loop execute nonzero GGUF weights, compact sparse
 * dispatch, original CPU collectives and ordinary host-owned generation. The
 * singleton tail is a declared stage of the same original rank service. No GPU
 * aliases, MPI-self contexts or alternative arithmetic implementations occur.
 * The fixture supplies RuntimeInitPhase's explicit CPU-backend birth from the
 * authenticated original NUMA inventory before any model workspace is bound.
 */
#include "PipelineGenerationTestSupport.h"
#include "backends/BackendManager.h"
#include "config/OrchestrationConfigParser.h"
#include "config/OrchestrationStartupPolicy.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "execution/runner/ModelContextRetirement.h"
#include "utils/NUMATopology.h"
#include "utils/NodeTopology.h"
#include "execution/global/OriginalCPUPipelineConstruction.h"
#include "collective/backends/ShmemSpinBackend.h"
#include "collective/NativeCPUCollectiveProgram.h"
#include "planning/OriginalRankCapacityExchange.h"
#include <set>
namespace llaminar2::test {
namespace {
/** @brief Explicit native backend and exact authored expert arithmetic policy. */
struct CPUStageCase { const char* backend; const char* compute; const char* name; bool asymmetric_capacity=false; };
class CPUStagePipelineGeneration : public ::testing::TestWithParam<CPUStageCase> {};
/** @brief Retire the original command service/model after actual stage/native borrowers, also on assertions. */
class OriginalCPUPipelineFixtureLifetime final {
public:
    explicit OriginalCPUPipelineFixtureLifetime(std::shared_ptr<IMPIContext> mpi,const OrchestrationConfig& config)
        :runner_(std::make_unique<OrchestrationRunner>(std::move(mpi),config,createExecutionPlanBuilder())){}
    ~OriginalCPUPipelineFixtureLifetime() noexcept {
        try {finish();} catch(...) {std::terminate();}
    }
    /** @return Sole original runner, before terminal retirement only. */
    OrchestrationRunner& runner() const {return *runner_;}
    /** @brief Retain only the actual source's duty to release its original worker command receive. */
    void armCommandSource(bool source) {command_source_=source;}
    /** @brief Close worker admission before collective model/native retirement; repeatable. */
    void finish() {
        if(!runner_)return;
        if(command_source_)runner_->shutdownMPIWorkers();
        auto model=runner_->modelContextReuseContract();
        runner_->shutdown();runner_.reset();
        if(model)receipt_=retireExclusiveModelContextReuseContract(*model);
    }
    /** @return Actual final-owner receipt, never a caller-authored readiness flag. */
    const std::optional<ModelContextRetirementReceipt>& receipt() const {return receipt_;}
private:
    std::unique_ptr<OrchestrationRunner> runner_;
    std::optional<ModelContextRetirementReceipt> receipt_;
    bool command_source_=false;
};
/** @test Reset and changing live rows reuse actual original CPU stage service and retire its owners. */
TEST_P(CPUStagePipelineGeneration, OriginalSocketsOrdinaryResetAndLiveRows) {
    const auto mpi=MPIContextFactory::global();
    ASSERT_NE(mpi,nullptr);ASSERT_EQ(mpi->world_size(),2);
    const auto inventory=mpi->clusterInventory();ASSERT_NE(inventory,nullptr);
    ASSERT_EQ(inventory->ranks.size(),2u);
    const auto affinity=NUMATopology::detectLocalNUMANode();
    ASSERT_TRUE(affinity.detection_succeeded);
    ASSERT_EQ(inventory->ranks.at(mpi->rank()).cpu.numa_node,affinity.local_numa_node);
    const auto physical=NodeTopology::detect();
    const auto node=std::find_if(physical.numaNodes().begin(),physical.numaNodes().end(),
        [&](const auto& n){return n.node_id==affinity.local_numa_node;});
    ASSERT_NE(node,physical.numaNodes().end());ASSERT_GE(node->socket_id,0);
    std::array<int,2> sockets{};
    mpi->allgather_bytes(&node->socket_id,sockets.data(),sizeof(int));
    ASSERT_NE(sockets[0],sockets[1]);
    const auto& first=inventory->ranks[0];const auto& second=inventory->ranks[1];
    ASSERT_EQ(first.hostname,second.hostname);ASSERT_NE(first.cpu.numa_node,second.cpu.numa_node);
    const auto a=GlobalDeviceAddress::cpu(first.cpu.numa_node,first.hostname);
    const auto b=GlobalDeviceAddress::cpu(second.cpu.numa_node,second.hostname);
    SharedPipelinePlanningGGUFFixture source(true,2,true);
    const auto policy=GetParam();
    std::vector<std::string> args{"cpu-stage-native","-m",source.path(),"--mpi-procs","2",
        "--context-length","64","--no-mtp","--define-domain",
        "shared="+a.toString()+","+b.toString()+";scope=node_local;ranks=0,1;backend="+policy.backend+
        ";expert_compute="+policy.compute+";expert_phase=uniform",
        "--define-domain","tail="+a.toString()+";scope=single;owner=0;backend=host;expert_compute=apportioned",
        "--pp-stage","0=shared:0-0","--pp-stage","1=tail:1-1"};
    std::vector<char*> argv;for(auto& arg:args)argv.push_back(arg.data());
    auto config=OrchestrationConfigParser{}.parseArgs(argv.size(),argv.data());
    config.prefill_max_bucket_size=8;
    config.prefix_cache.ram_budget_bytes=8u*1024u*1024u;config.prefix_cache.disk_budget_bytes=0;
    config.moe_rebalance.mode=MoERebalanceRuntimeMode::Off;
    if(policy.asymmetric_capacity && mpi->rank()==1)
        config.max_cpu_memory_mb=(NativeCPUCollectiveProgram::domainBytes()+NativeCPUCollectiveProgram::processPoolBytes())/(1024u*1024u);
    publishOrchestrationStartupPolicy(config);
    // This executable drives the runner directly rather than RuntimeInitPhase.
    // Supply its same explicit original NUMA owner; a workspace accessor must
    // never invent an aggregate CPU backend or reinterpret a stage-local rank.
    initCPUBackend(inventory->ranks.at(mpi->rank()).cpu.numa_node);
    ASSERT_NE(getCPUBackend(),nullptr);
    ASSERT_EQ(cpuBackendNUMANode(),affinity.local_numa_node);
    if(policy.asymmetric_capacity) {
        OriginalCPUPipelineFixtureLifetime lifetime(mpi,config);
        auto& runner=lifetime.runner();
        const bool initialized=runner.initialize();
        const std::string error=runner.lastError();
        lifetime.finish();
        const int observed_rejection=!initialized && !lifetime.receipt() &&
            error.find("No complete MoE pipeline graph fits")!=std::string::npos &&
            error.find("Original pipeline fixed capacity")!=std::string::npos;
        std::array<int,2> rejected{};
        mpi->allgather_bytes(&observed_rejection,rejected.data(),sizeof(observed_rejection));
        EXPECT_EQ(rejected,(std::array{1,1}))<<error;
        return;
    }
    // The CPU public protocol owns one root and the existing original MPI
    // worker loop. GPU immutable-ticket source/follower calls are a different
    // lifecycle and must never be impersonated by a CPU test.
    std::array<int32_t,6> terminal{mpi->rank(),0,0,0,0,0};
    {
        OriginalCPUPipelineFixtureLifetime lifetime(mpi,config);
        auto& runner=lifetime.runner();
        ASSERT_TRUE(runner.initialize())<<runner.lastError();
        runner.setMPICoordinatedMode(true);
        const bool source_rank=mpi->rank()==0; // The authored singleton tail owns vocabulary on original rank0.
        lifetime.armCommandSource(source_rank);
        ASSERT_EQ(runner.coordinatedRootRank(),0);
        if(!source_rank) {
            runner.runMPIWorkerLoop();
            terminal[3]=1; // Only the actual loop return can author this terminal observation.
        } else {
            runner.setStopTokens({});
            SamplingParams sampling;sampling.temperature=0.8f;sampling.top_k=20;sampling.seed=123;
            std::vector<int32_t> first_response;
            for(int replay=0;replay<4;++replay) {
                SCOPED_TRACE(replay);
                if(replay)runner.clearCache();
                auto request_sampling = sampling;
                if (replay == 3)
                {
                    // Greedy workers still enter the original token broadcast;
                    // only the vocabulary owner may mutate its history logits.
                    request_sampling.temperature = 0.0f;
                    request_sampling.presence_penalty = 0.2f;
                    request_sampling.frequency_penalty = 0.1f;
                    request_sampling.repetition_penalty = 1.1f;
                }
                runner.setSamplingParams(request_sampling);
                // full -> partial -> same full -> prefix extension, all on one
                // admitted bank and original CPU request/worker command service.
                std::vector<int32_t> prompt=replay==1?std::vector<int32_t>{11}:
                    std::vector<int32_t>{11,12,13,14,15,16,17,18};
                if(replay==3)prompt.push_back(19);
                ASSERT_TRUE(runner.prefill(prompt))<<runner.lastError();
                std::vector<int32_t> response;
                for(const int budget:{1,3,1}) {
                    // CPU ordinary decodeStep is one token, bounded by the
                    // remaining response budget. The source frontend requests
                    // each real step; no grouped GPU result is fabricated.
                    for(int remaining=budget;remaining>0;--remaining) {
                        runner.setDecodeStepTokenBudget(remaining);
                        const auto result=runner.decodeStep();
                        ASSERT_TRUE(result.success())<<result.error;
                        ASSERT_EQ(result.tokens.size(),1u);
                        EXPECT_GE(result.tokens.front(),0);
                        response.push_back(result.tokens.front());
                        ++terminal[2];
                        ASSERT_TRUE(runner.maybeApplyMoERebalance(result.tokens.size()))<<runner.lastError();
                    }
                }
                ASSERT_EQ(response.size(),5u);
                if(replay==0)first_response=response;
                if(replay==2)EXPECT_EQ(response,first_response);
                ++terminal[1];
            }
        }
        // Source sends SHUTDOWN once; workers have already left their original
        // loop. Both ranks then retire original CPU graphs/contexts and host
        // model owners, before exchanging immutable test receipts below.
        lifetime.finish();
        ASSERT_TRUE(lifetime.receipt());
        EXPECT_EQ(lifetime.receipt()->retiredDeviceCount(),0u);
        terminal[4]=1;
        terminal[5]=static_cast<int32_t>(lifetime.receipt()->retiredDeviceCount());
    }
    std::array<int32_t,12> receipts{};
    mpi->allgather_bytes(terminal.data(),receipts.data(),sizeof(terminal));
    EXPECT_EQ(receipts,(std::array<int32_t,12>{0,4,20,0,1,0,1,0,0,1,1,0}));
    RecordProperty("actual_original_cpu_stage_members",2);
    RecordProperty("actual_singleton_tail_members",1);
}
/** @test A structural failure on one original rank prevents both ranks from entering a smaller candidate. */
TEST(OriginalCPUStageCapacity, StructuralFailureNeverSelectsFewerRows)
{
    const auto original = MPIContextFactory::global();
    ASSERT_NE(original, nullptr);
    ASSERT_EQ(original->world_size(), 2);
    constexpr std::array candidates{8, 4, 1};
    const std::vector<int> ladder(candidates.begin(), candidates.end());
    requireIdenticalOverlayGraphRowCandidates(ladder, original);

    for (const bool structural : {false, true}) {
        std::vector<int> attempted;
        std::string failure;
        try {
            for (const int rows : candidates) {
                auto rejected = runOriginalPipelineCapacityTrial(*original, rows,
                    "pipeline-capacity.local-fixed", "Original pipeline fixed capacity", [&] {
                        attempted.push_back(rows);
                        if (rows != candidates.front() || original->rank() != 1)
                            return;
                        if (structural)
                            throw std::invalid_argument("malformed certified contribution");
                        throw PhysicalMemoryCapacityExhausted("certified fixed capacity exhausted");
                    });
                if (!rejected)
                    break;
            }
        } catch (const std::invalid_argument& error) {
            failure = error.what();
        }

        // Record only terminal evidence after every phase has retired.
        const std::array<int, 3> terminal{static_cast<int>(attempted.size()),
            attempted.empty() ? 0 : attempted.back(), failure.empty() ? 0 : 1};
        std::array<int, 6> gathered{};
        original->allgather_bytes(terminal.data(), gathered.data(), sizeof(terminal));
        if (structural) {
            EXPECT_EQ(gathered, (std::array{1, 8, 1, 1, 8, 1}));
            EXPECT_NE(failure.find("Original pipeline fixed capacity"), std::string::npos);
            if (original->rank() == 1)
                EXPECT_NE(failure.find("malformed certified contribution"), std::string::npos);
        } else {
            EXPECT_EQ(gathered, (std::array{2, 4, 0, 2, 4, 0}));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(OriginalCPUStage,CPUStagePipelineGeneration,
    ::testing::Values(CPUStageCase{"upi","apportioned","UPIApportioned"},
        CPUStageCase{"upi","tensor-sharded","UPITensorSharded"},
        CPUStageCase{"mpi","apportioned","MPIApportioned"},
        CPUStageCase{"mpi","tensor-sharded","MPITensorSharded"},
        CPUStageCase{"upi","tensor-sharded","UPITensorShardedAsymmetricCapacity",true}),
    [](const auto& p){return p.param.name;});
}
}

namespace llaminar2::test {
/**
 * @test Intentional real failure AFTER original CPU-domain birth retains the first reason before MPI abort.
 * The dedicated preflight driver selects this negative child and authenticates its nonzero status/retirement.
 */
TEST(CPUStagePipelineConstructionFailure, DISABLED_OriginalParentReasonBeforeAbort)
{
    const auto original=MPIContextFactory::global();
    ASSERT_NE(original,nullptr);ASSERT_EQ(original->world_size(),2);
    const auto inventory=original->clusterInventory();ASSERT_NE(inventory,nullptr);
    ASSERT_EQ(inventory->ranks.size(),2u);
    const auto affinity=NUMATopology::detectLocalNUMANode();ASSERT_TRUE(affinity.detection_succeeded);
    ASSERT_EQ(inventory->ranks.at(original->rank()).cpu.numa_node,affinity.local_numa_node);
    const auto physical=NodeTopology::detect();
    const auto node=std::find_if(physical.numaNodes().begin(),physical.numaNodes().end(),
        [&](const auto& n){return n.node_id==affinity.local_numa_node;});
    ASSERT_NE(node,physical.numaNodes().end());ASSERT_GE(node->socket_id,0);
    std::array<int,2> sockets{};original->allgather_bytes(&node->socket_id,sockets.data(),sizeof(int));
    ASSERT_NE(sockets[0],sockets[1]);
    GlobalPPStageSpec stage;stage.stage_id=0;stage.domain_name="original_cpu_failure";
    stage.first_layer=stage.last_layer=0;stage.has_embedding=stage.has_lm_head=true;
    stage.is_global_tp=true;stage.backend=CollectiveBackendType::UPI;
    for(const auto& owner:inventory->ranks) {
        stage.participating_ranks.push_back(owner.rank);
        stage.per_rank_devices.push_back(GlobalDeviceAddress::cpu(owner.cpu.numa_node,owner.hostname));
    }
    const auto topology=GlobalPPTopology::build({stage},1,2);
    const auto declaration=OriginalCPUPipelineProgram::checked(stage,*inventory);
    const auto& observed=inventory->ranks[original->rank()];
    ASSERT_GT(observed.cpu.memory_bytes,0u);ASSERT_GT(observed.cpu.free_memory_bytes,0u);
    const PhysicalMemoryResource cpu{original->rank(),DeviceId::cpu(),observed.cpu.memory_bytes,observed.cpu.free_memory_bytes};
    PhysicalMemoryPlanBuilder admission;
    declaration.nativeProgram().contributeDomainBOM(admission,cpu,original->rank());
    NativeCPUCollectiveProgram::contributeProcessPoolBOM(admission,cpu);
    if(original->rank()==stage.participating_ranks.front())
        admission.add(cpu,PhysicalMemoryOwner::ActivationTransportStaging,
            ShmemSpinArena::allocationGeometry(static_cast<int>(stage.participating_ranks.size())).mapped_bytes);
    auto memory=std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(admission.build()),original->rank());
    DomainCommunicatorRegistry domains;
    withOriginalCPUPipelineConstruction(domains,*original,[&] {
        domains.initialize(topology,original->communicator(),original->rank(),std::span(&declaration,1),memory);
        if(!domains.hasContextForStage(stage.stage_id))
            throw std::runtime_error("CPU cold negative fixture did not create its original domain");
        {
            const auto native = domains.globalTPContextForStage(stage.stage_id);
            const auto dense = MPIContextFactory::borrowOriginalCPUStage(original, native);
            if (dense->communicator() != native->communicator() ||
                dense->communicator() == original->communicator() ||
                dense->rank() != native->myIndex() || dense->world_size() != native->degree() ||
                dense->clusterInventory()->ranks.at(dense->rank()).cpu.numa_node != observed.cpu.numa_node)
                throw std::runtime_error("CPU cold fixture lost the original dense split view");
        }
        original->barrier(); // Both real domain owners exist before the controlled failure.
        if(original->rank()==stage.participating_ranks.front())
            throw std::runtime_error("CPU_PIPELINE_COLD_FAILURE_ORIGINAL_REASON");
        original->barrier(); // The original parent's abort must release this real peer.
    });
    FAIL()<<"An original native CPU construction failure returned to its caller";
}
}
