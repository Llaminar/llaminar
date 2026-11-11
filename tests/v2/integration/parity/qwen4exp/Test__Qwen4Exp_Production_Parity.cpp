/**
 * @file Test__Qwen4Exp_Production_Parity.cpp
 * @brief Canonical Flash Next RAM/mmap numerical and focused admission cases.
 *
 * The typed expander owns precision, storage, placement, and movement axes.
 * This MPI-aware executable retains the ordinary production runner and shared
 * six-CSV numerical workflow; missing backend/model/reference evidence fails.
 * SnapshotAdmission shares numerical setup, including reference validation and
 * diagnostic arenas. FullSourceAdmission uses a separate gold-free lifecycle
 * with snapshots disabled. Its four-policy MTP product admits real source,
 * sidecar and workspace owners without changing numerical-reference support.
 * Both retain real serving preparation/capture and
 * stop before inference; neither is numerical or performance evidence.
 * The process entry uses the production MPI session owner, so admitted CPU
 * collectives retain their original process pool until native finalization.
 * Terminal evidence follows both MPI and GPU retirement.
 */
#include "app/MPIProcessSession.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include "integration/NativeLifetimeCompletion.h"
#include "utils/MPISessionOwner.h"
#include "Qwen4ExpParityTestBase.h"
#include "Qwen4ExpNgramAdmissionCases.h"
#include "Qwen4ExpSourceAdmission.h"
#include "Qwen4ExpObservedPipelineCases.h"
#include "Qwen4ExpObservedPipelineAdmission.h"
#include "Qwen4ExpSourceAdmissionCases.h"
#include "Qwen4ExpSharedContinuationMTPSourceCases.h"
#include "Qwen4ExpCPUJointParityCases.h"
#include "Qwen4ExpSourceAdmissionCI.h"
#include "Qwen4ExpFixedHeadParityCases.h"
#include "Qwen4ExpGPUAdaptiveParityCases.h"
#include "Qwen4ExpFullROCmTP4ParityCases.h"
#include "Qwen4ExpAllGPUQ6MTPParityCases.h"
#include "Qwen4ExpLocalPairedCases.h"
#include "collective/BackendRouter.h"
#include "backends/GPUDeviceContextPool.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "backends/BackendManager.h"
#include "config/OrchestrationStartupPolicy.h"
#include "loaders/ModelContext.h"
#include "utils/TestOrchestrationHelper.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen4exp;

/**
 * @brief Bind observed QFN cells before the original fixture enters model or MPI execution.
 *
 * Existing fixed-address cells continue through the same base lifecycle.
 * Inactive discovery ranks never create a parity context, runner or GPU graph.
 */
template<class Derived>
class Qwen4ExpObservedPipelineParityFixture
    :public Qwen4ExpConfigDrivenParityTest<Derived>,public ModelParityCaseParameter {
    using Base=Qwen4ExpConfigDrivenParityTest<Derived>;
public:
    /** @return Bound runtime case on observed ranks, otherwise the original parameter. */
    const ModelParityCase& modelParityCase() const {
        return observed_?observed_->boundCase():ModelParityCaseParameter::modelParityCase();
    }

    /** @return Compatibility configuration built only after original inventory binding. */
    const TestConfig& getTestConfig() const {
        if(!observed_)return ModelParityCaseParameter::getTestConfig();
        if(!bound_config_)bound_config_=modelParityCase().toTestConfig();
        return *bound_config_;
    }

protected:
    /** @brief Enter the existing numerical lifecycle only on the selected original owner. */
    void SetUp() override {
        const auto& original=ModelParityCaseParameter::modelParityCase();
        if(original.topology.needsObservedPipelineBinding()) {
            ASSERT_NO_THROW(observed_.emplace(original));
            ASSERT_NO_THROW(observed_->authenticate());
            if(!observed_->execution())return;
            Base::mpi_ctx_=observed_->execution();
        }
        Base::SetUp();
    }

    /** @brief Retire active production resources before the original discovery join. */
    void TearDown() override {
        if(!observed_) {
            Base::TearDown();
            return;
        }
        const bool active=bool(observed_->execution());
        std::exception_ptr failure;
        if(active) {
            try {Base::TearDown();} catch(...) {failure=std::current_exception();}
            Base::mpi_ctx_.reset();
        }
        const bool complete=observed_->complete(!failure&&!Base::HasFailure()&&!Base::IsSkipped());
        const auto output=Base::getResultsDir()/"observed-pipeline"/("rank-"+std::to_string(observed_->discoveryRank()));
        std::filesystem::create_directories(output);
        ASSERT_NO_FATAL_FAILURE(observed_->writeReceipt(output,"parity_work_succeeded",complete));
        if(active && complete) {
            // Reuse the canonical lossless configuration exporter only after
            // actual original binding and the ordinary numerical lifecycle.
            std::ostringstream declaration,configuration;
            PrintTo(ModelParityCaseParameter::modelParityCase(),&declaration);
            PrintTo(observed_->boundCase(),&configuration);
            const auto* test=::testing::UnitTest::GetInstance()->current_test_info();
            ASSERT_TRUE(test);
            const nlohmann::json exported={{"schema",1},
                {"exact_cell",std::string(test->test_suite_name())+"."+test->name()},
                {"declaration",nlohmann::json::parse(declaration.str())},
                {"configuration",nlohmann::json::parse(configuration.str())}};
            std::ofstream file(output/"observed-pipeline-config.json");
            ASSERT_TRUE(file.is_open());
            file<<exported.dump()<<'\n';file.flush();ASSERT_TRUE(file.good());
        }
        EXPECT_TRUE(complete)<<"Original QFN parity work failed before selected retirement";
        if(failure)std::rethrow_exception(failure);
    }

    /** @return Whether this original rank runs the existing GTest body. */
    bool executesQwen4ExpBody() const noexcept {return !observed_ || bool(observed_->execution());}

    /** @return Borrow the same exact selected MPI context when constructing the runner. */
    std::shared_ptr<IMPIContext> freshParityMPIContext() const override {
        return observed_?observed_->execution():Base::freshParityMPIContext();
    }

    /** @return Actual selected entry GPU, retaining its observed device ordinal. */
    DeviceId getDevice() override {
        return observed_?modelParityCase().topology.participants.front().address.toLocalDeviceId():Base::getDevice();
    }

    /** @return The same selected owner used for original local execution. */
    DeviceId getDeviceForRank() override {
        return observed_?getDevice():Base::getDeviceForRank();
    }

private:
    std::optional<Qwen4ExpObservedPipelineBinding> observed_;
    mutable std::optional<TestConfig> bound_config_;
};

/** @brief One canonical production fixture per exact expanded matrix cell. */
class Qwen4ExpMainParityTest
    : public Qwen4ExpConfigDrivenParityTest<Qwen4ExpMainParityTest>, public ModelParityCaseParameter {};

/** @test Every published numerical boundary and actual selected native row path. */
TEST_P(Qwen4ExpMainParityTest, ProductionParity) { runQwen4ExpProductionParityCampaign(); }

/** @brief Complete-history numerical candidates reuse the original campaign with mandatory paired admission. */
class Qwen4ExpLocalHotNumericalTest
    : public Qwen4ExpConfigDrivenParityTest<Qwen4ExpLocalHotNumericalTest>, public ModelParityCaseParameter
{
protected:
    /** @brief Fail before original setup if this new cell has no frozen panel obligation. */
    void SetUp() override
    {
        ASSERT_NO_THROW(qfnRequireLocalHotNumericalBinding(modelParityCase()));
        Qwen4ExpConfigDrivenParityTest<Qwen4ExpLocalHotNumericalTest>::SetUp();
    }

    /**
     * @brief Preserve the driver's exact full-stage history before model admission.
     * Numerical acceptance runs through the complete paired canary panel; this
     * fixture provides its frozen-history stage diagnosis without a legacy prompt.
     */
    void applyModelOverrides() override
    {
        Qwen4ExpConfigDrivenParityTest<Qwen4ExpLocalHotNumericalTest>::applyModelOverrides();
        accuracy::requireQFN(bool(frozen_history_replay_),
            "Local-hot full-stage diagnosis requires the original authenticated frozen history");
    }
};

/** @test The unchanged canonical complete-history production campaign. */
TEST_P(Qwen4ExpLocalHotNumericalTest, ProductionParity)
{
    runQwen4ExpProductionParityCampaign();
}

INSTANTIATE_TEST_SUITE_P(Qwen4ExpLocalHotNumerical, Qwen4ExpLocalHotNumericalTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpLocalHotNumerical", qwen4ExpLocalHotNumericalCases())),
    [](const ::testing::TestParamInfo<ModelParityCase>& info) { return info.param.testName(); });

/** @brief Functional byte replay uses the existing active cases, without another matrix producer. */
class Qwen4ExpMTPExactReplayTest
    : public Qwen4ExpObservedPipelineParityFixture<Qwen4ExpMTPExactReplayTest> {
protected:
    /** @return Native byte proof; existing reference headers size snapshots, without download/scoring. */
    MTPParityProofPurpose mtpParityProofPurpose() const noexcept override
    {
        return MTPParityProofPurpose::NativeExactReplay;
    }
    /**
     * @return The predeclared native-functional history, independent of numerical rows.
     *
     * Keep the original prompt and greedy policy. Thirty-two prefix rows are
     * fixed before execution; a fully rejecting history still fails rather
     * than extending the window or changing its inputs to obtain acceptance.
     */
    int nativeMTPFunctionalHistoryRows() const noexcept override
    {
        return 32;
    }
    /** @brief Reference generation would violate this fixture's existing-header-only contract. */
    bool regeneratePyTorchSnapshots() override
    {
        ADD_FAILURE() << "MTPExactReplay cannot generate reference snapshots";
        return false;
    }
};

/** @test Original native condition/verifier inputs, snapshots and reset/reuse, without HF acceptance. */
TEST_P(Qwen4ExpMTPExactReplayTest, MTPExactReplay) {
    if(executesQwen4ExpBody())runQwen4ExpMTPExactReplay();
}

/** @brief Public ordinary generation on the original source, followed by real reset/reuse. */
class Qwen4ExpOrdinaryGenerationTest
    : public Qwen4ExpObservedPipelineParityFixture<Qwen4ExpOrdinaryGenerationTest> {
    using Base=Qwen4ExpConfigDrivenParityTest<Qwen4ExpOrdinaryGenerationTest>;
    std::optional<Qwen4ExpOrdinaryGenerationWorkload> ordinary_workload_;
protected:
    /** @return The same original source-selected input/sampler after real reset. */
    Qwen4ExpNativeGenerationRequest qwen4ExpLookupReuseRequest() const override
    {
        if(!ordinary_workload_)throw std::logic_error("Ordinary reuse lost its original workload");
        return ordinary_workload_->request(Qwen4ExpOrdinaryRequestPhase::Reuse);
    }
    /** @return Existing-header-only setup; this fixture performs no HF scoring. */
    MTPParityProofPurpose mtpParityProofPurpose() const noexcept override
    { return MTPParityProofPurpose::NativeExactReplay; }
    /** @brief Functional generation cannot manufacture numerical reference rows. */
    bool regeneratePyTorchSnapshots() override
    { ADD_FAILURE()<<"NativeGeneration cannot generate reference snapshots";return false; }
    /** @brief Run two real public requests, with existing graph/source/reader retirement checks. */
    void runOrdinaryGeneration()
    {
        ASSERT_EQ(modelParityCase().mtp,ModelParityMTP::Off);
        ASSERT_NO_FATAL_FAILURE(Base::beginProductionParitySnapshotAdmission());
        ASSERT_NO_FATAL_FAILURE(Base::setupProductionParityPipeline());
        ASSERT_NE(Base::orch_runner_,nullptr);
        ASSERT_FALSE(Base::orch_runner_->config().mtp.enabled);
        const auto tokenizer=Base::orch_runner_->tokenizer();
        ASSERT_TRUE(tokenizer);
        const std::vector<int32_t> checkpoint(Base::config_.token_ids.begin(),Base::config_.token_ids.end());
        ordinary_workload_.emplace(Qwen4ExpOrdinaryGenerationWorkload::fromOriginalCase(modelParityCase(),
            checkpoint,Base::productionParitySamplingParams(),[&](const auto& messages,bool add,const auto& tools,bool thinking){
                return tokenizer->encodeChat(messages,add,tools,thinking);
            }));
        const auto request=ordinary_workload_->request(Qwen4ExpOrdinaryRequestPhase::First);
        ASSERT_LE(request.tokens.size()+static_cast<std::size_t>(request.maximum_tokens),
            static_cast<std::size_t>(Base::orch_runner_->config().max_seq_len));
        if(ordinary_workload_->sustained())Base::orch_runner_->setStopTokens(tokenizer->stop_tokens());
        const auto generation=Base::orch_runner_->generate(request.tokens,request.maximum_tokens,request.sampling);
        ASSERT_TRUE(generation.error.empty())<<generation.error;
        // Completion must satisfy the original declared request horizon. A
        // model stop below it fails; no stop suppression, retry or drain forces
        // an asynchronous candidate to become successful movement evidence.
        ASSERT_GE(generation.tokens.size(),static_cast<std::size_t>(request.minimum_tokens));
        ASSERT_LE(generation.tokens.size(),static_cast<std::size_t>(request.maximum_tokens));
        // This first request may finish while the original background economy
        // profile is still learning. It remains a complete generation/graph
        // observation, but cannot certify the two-request movement obligation.
        ASSERT_NO_FATAL_FAILURE(Base::finishProductionParityEvidence(
            ParityMoEMovementEvidenceBoundary::Intermediate));
        ASSERT_NO_FATAL_FAILURE(Base::assertLookupReuseAfterProductionProof());
        // Mandatory explicit final boundary: use the latest cumulative original
        // snapshot after real reuse, never add request snapshots or wait for work.
        ASSERT_NO_FATAL_FAILURE(Base::finishProductionParityEvidence(
            ParityMoEMovementEvidenceBoundary::Final));
        std::ofstream record(Base::ensureResultsDir()/"ordinary_generation_scope.txt");
        ASSERT_TRUE(record.is_open());
        record<<"public_requests=2\nfirst_response_budget="<<request.maximum_tokens
            <<"\nreuse_response_budget="<<ordinary_workload_->request(Qwen4ExpOrdinaryRequestPhase::Reuse).maximum_tokens
            <<"\nsource_declared_sustained="<<ordinary_workload_->sustained()
            <<"\nmtp=off\nhf_numerical_scoring=0\n";
        record<<"first_request_model_stopped="<<generation.is_complete<<'\n';
        record<<"movement_evidence_scope=final_two_request_cohort\n";
        record.flush();ASSERT_TRUE(record.good());
    }
};
/** @test Actual complete captured ordinary generation and original-owner reuse; no numerical certificate. */
TEST_P(Qwen4ExpOrdinaryGenerationTest, NativeGeneration) {
    if(executesQwen4ExpBody())runOrdinaryGeneration();
}
INSTANTIATE_TEST_SUITE_P(Qwen4ExpOwnerSubsetOrdinaryGenerationCandidate,Qwen4ExpOrdinaryGenerationTest,
    ::testing::ValuesIn(qwen4ExpOwnerSubsetOrdinaryGenerationCases()),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});

// Focused combined C0 requests reuse original ROCm TP2 RAM topology and unchanged production fixtures.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpCombinedLocalHotC0OrdinaryCandidate,Qwen4ExpOrdinaryGenerationTest,
    ::testing::ValuesIn(qwen4ExpCombinedLocalHotC0OrdinaryCases()),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpCombinedLocalHotC0ExactCandidate,Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpCombinedLocalHotC0ExactCases()),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpLocalHotBudgetOrdinaryCandidate,Qwen4ExpOrdinaryGenerationTest,
    ::testing::ValuesIn(qwen4ExpLocalHotBudgetOrdinaryCases()),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpLocalHotBudgetExactCandidate,Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpLocalHotBudgetExactCases()),
    [](const auto& info){return info.param.testName();});
// Six explicit R3 requests extend only C0/equal/unequal to recurrent draft reuse.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpLocalHotRecurrentExactCandidate,Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpLocalHotRecurrentCases()),
    [](const auto& info){return info.param.testName();});

// Bounded CUDA siblings retain the original source, parent and canonical fixtures.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpCUDALocalHotOrdinaryCandidate,Qwen4ExpOrdinaryGenerationTest,
    ::testing::ValuesIn(qwen4ExpCUDALocalHotOrdinaryCases()),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpCUDALocalHotExactCandidate,Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpCUDALocalHotExactCases()),
    [](const auto& info){return info.param.testName();});

// Same original GPU-only Q6 parent; source admission and actual generation are separate gates.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpAllGPUQ6LocalHotOrdinaryCandidate,Qwen4ExpOrdinaryGenerationTest,
    ::testing::ValuesIn(qwen4ExpAllGPUQ6LocalHotOrdinaryCases()),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpAllGPUQ6LocalHotExactCandidate,Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpAllGPUQ6LocalHotExactCases()),
    [](const auto& info){return info.param.testName();});

// Rank-local all-GPU two-tier candidates use the original mapped device
// GraphService. Source admission and native exact proof remain separate jobs.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpAllGPUQ6DynamicOrdinalR3Exact, Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpAllGPUQ6MTPNativeExactCases()),
    [](const ::testing::TestParamInfo<ModelParityCase>& info) { return info.param.testName(); });

// Real physical world2; source admission and native exact arithmetic are disjoint proofs.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpCPUJointWorld2MTPExact, Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpCPUJointWorld2MTPExactCases()),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});


INSTANTIATE_TEST_SUITE_P(Qwen4ExpCPUJointWorld2DynamicMTPExact, Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpCPUJointWorld2DynamicMTPExactCases()),
    [](const ::testing::TestParamInfo<ModelParityCase> &info) { return info.param.testName(); });

INSTANTIATE_TEST_SUITE_P(Qwen4ExpOppositeVendorMTPRetained15Exact, Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpOppositeVendorMTPRetained15Cases()),
    [](const ::testing::TestParamInfo<ModelParityCase> &info) { return info.param.testName(); });



INSTANTIATE_TEST_SUITE_P(Qwen4ExpConsumerLocalTP4StaticRetained15Exact, Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpConsumerLocalTP4NativeExactCases()),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpConsumerLocalTP4StaticRetained15, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpConsumerLocalTP4StaticRetained15",qwen4ExpConsumerLocalTP4ExecutionCases())),
    [](const auto& info){return info.param.testName();});

INSTANTIATE_TEST_SUITE_P(Qwen4ExpConsumerLocalTP2StaticRetained15Exact, Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpConsumerLocalTP2NativeExactCases(ModelParityExpertOverlayPolicy{})),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpConsumerLocalTP2DynamicOrdinalRetained15Exact, Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpConsumerLocalTP2NativeExactCases({.owner_order=RoutedExpertOwnerOrder::Ordinal,.movement=ModelParityExpertMovement::Dynamic})),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpConsumerLocalTP2DynamicRandomRetained15Exact, Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpConsumerLocalTP2NativeExactCases({.owner_order=RoutedExpertOwnerOrder::Random,.movement=ModelParityExpertMovement::Dynamic})),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});

// Shared source readers retain the same numerical cells and original serial
// controls. Native exact replay is separate from source-only admission.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpSharedCoordinatorTP2StaticRetained15Exact, Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpSharedCoordinatorTP2NativeExactCases(ModelParityExpertOverlayPolicy{})),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpSharedCoordinatorTP2DynamicOrdinalRetained15Exact, Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpSharedCoordinatorTP2NativeExactCases({.owner_order=RoutedExpertOwnerOrder::Ordinal,.movement=ModelParityExpertMovement::Dynamic})),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpSharedCoordinatorTP2DynamicRandomRetained15Exact, Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpSharedCoordinatorTP2NativeExactCases({.owner_order=RoutedExpertOwnerOrder::Random,.movement=ModelParityExpertMovement::Dynamic})),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});

// Active native exact proof remains the original fixture. Expert movement
// requirements are retained independently from predictor depth/controller policy.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpOppositeVendorDynamicOrdinalRetained15Exact, Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpOppositeVendorDynamicExpertMTPRetained15Cases(RoutedExpertOwnerOrder::Ordinal)),
    [](const ::testing::TestParamInfo<ModelParityCase> &info) { return info.param.testName(); });
INSTANTIATE_TEST_SUITE_P(Qwen4ExpOppositeVendorDynamicRandomRetained15Exact, Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpOppositeVendorDynamicExpertMTPRetained15Cases(RoutedExpertOwnerOrder::Random)),
    [](const ::testing::TestParamInfo<ModelParityCase> &info) { return info.param.testName(); });

/**
 * @test Admit and materialize the exact production cell before numerical work.
 *
 * initialize() must construct every declared serving family and diagnostic
 * arena under the real physical-memory authority. A dry-run plan cannot prove
 * this boundary. The ordinary fixture teardown retires the captured graphs,
 * lookup service, weights and arenas, including after a failed initialization.
 */
TEST_P(Qwen4ExpMainParityTest, SnapshotAdmission)
{
    ASSERT_NO_FATAL_FAILURE(beginProductionParitySnapshotAdmission());
    ASSERT_TRUE(setupProductionParityPipeline())
        << "Flash Next production admission/materialization failed";
    ASSERT_TRUE(hasOrchestrationRunner());
    ASSERT_TRUE(orch_runner_->isInitialized());
    EXPECT_EQ(orch_runner_->executionPlan().layerCount(), modelParityCase().model.transformer_layers);

    const auto *context = activeModelContextForDiagnostics();
    ASSERT_NE(context, nullptr);
    const auto placement = context->resolvedImmutableRowTablePlacement();
    ASSERT_NE(placement, nullptr);
    ASSERT_TRUE(modelParityCase().ngram_table_policy.has_value());
    const auto &policy = *modelParityCase().ngram_table_policy;
    EXPECT_EQ(orch_runner_->config().ngram_table_policy, policy);
    ASSERT_NE(orch_runner_->executionPlan().runtime.ngram_table_placement, nullptr);
    EXPECT_EQ(*orch_runner_->executionPlan().runtime.ngram_table_placement, *placement);
    placement->requireRequestedPolicy(policy);
    if(placement->isGPUAdaptive())
    {
        ASSERT_NO_FATAL_FAILURE(assertGPUAdaptiveNativeOwners());
    }
    else if(placement->isHostAdaptive())
    {
        ASSERT_TRUE(policy.isHostAdaptive());
        const auto slots=placement->hostAdaptiveLayout().owners().front().cache_rows;
        if(!policy.host_adaptive.maximum_cache_rows)EXPECT_EQ(slots,0u);
        else EXPECT_GT(slots,0u) << "Bounded positive-cache qualification cannot certify an admitted zero-slot response";
        EXPECT_LE(slots,policy.host_adaptive.maximum_cache_rows);
    }
    else EXPECT_TRUE(ngramStorageHonors(policy.storage,placement->storage()));
    EXPECT_GT(placement->extent().rowBytes(), 0u);
    if(placement->isGPUAdaptive())
    {
        // Actual mixed owners were proved above; never request a fake whole-tier view.
    }
    else if(!placement->isHostAdaptive() && placement->storage()==ImmutableRowStorage::VRAM)
    {
        EXPECT_EQ(context->hostImmutableRowProvider(), nullptr);
        std::size_t total_native_bytes = 0;
        for(const auto &allocator:placement->allocators())
        {
            const auto table=context->residentImmutableRowTable(allocator.device);
            ASSERT_NE(table,nullptr);
            EXPECT_EQ(table->device(),allocator.device);
            EXPECT_EQ(table->source().rowBytes(),placement->extent().rowBytes());
            const auto range = placement->isFixedHeadShards()
                ? placement->fixedHeadSourceRangeFor(allocator)
                : ImmutableRowTableSourceRange::whole(placement->extent());
            EXPECT_EQ(table->sourceRange(), range);
            EXPECT_EQ(table->nativeBytes(), range.nativeBytes());
            EXPECT_EQ(context->residentImmutableRowTableBytes(allocator.device),table->nativeBytes());
            total_native_bytes += table->nativeBytes();
        }
        // A distributed source owns every original byte exactly once. Whole
        // resident copies remain explicit replicas and have a different total.
        if (placement->isFixedHeadShards())
        {
            EXPECT_TRUE(policy.isFixedHeadSharded());
            EXPECT_EQ(total_native_bytes, placement->extent().nativeBytes());
            EXPECT_EQ(placement->allocators().size(), placement->consumers().size());
        }
    }
    else
    {
        const auto table=context->hostImmutableRowProvider();
        ASSERT_NE(table,nullptr);
        if(placement->isHostAdaptive())
        {
            EXPECT_EQ(table->mode(),HostImmutableRowMode::AdaptiveRAMSSD);
            EXPECT_EQ(table->maximumLiveRows(),placement->hostAdaptiveMaximumLiveRows());
            const auto owner=table->retainedOwnerObservation();
            EXPECT_EQ(owner.lookupStagingBytes(),CpuAdaptiveImmutableRowCacheGeometry::fromLayout(
                placement->hostAdaptiveLayout(),placement->allocators().front(),placement->hostAdaptiveMaximumLiveRows(),
                placement->hostAdaptivePolicy()).executionBytes());
        }
        else EXPECT_EQ(table->wholeStorage(),placement->storage());
        EXPECT_EQ(table->source().rowBytes(),placement->extent().rowBytes());
        for(const auto &consumer:placement->consumers())
            EXPECT_EQ(context->residentImmutableRowTableBytes(consumer.device),0u);
    }

    // The real runner exports its admitted prepared-weight identity only after
    // model preparation and graph materialization have both completed. Keep
    // its one physical ledger's report instead of estimating from free VRAM.
    const auto prepared = orch_runner_->modelContextReuseContract();
    ASSERT_TRUE(prepared.has_value());
    ASSERT_NE(prepared->physical_memory_authority, nullptr);
    std::ofstream memory(ensureResultsDir() / "admission_memory.txt");
    ASSERT_TRUE(memory.is_open());
    memory << prepared->physical_memory_authority->rankAttestationSummary(true);
    ASSERT_NO_THROW(requireQwen4ExpCPUStorageMaterialization(prepared->physical_memory_authority->rankAttestation(),
        prepared->prepared_routed_weight_plan, placement->allocators()));
    memory.flush();
    ASSERT_TRUE(memory.good());

    // Admission has its own artifact: no numerical CSV or full-suite receipt
    // is synthesized when prefill and decode have deliberately not executed.
    std::ofstream csv(ensureResultsDir() / "admission.csv");
    ASSERT_TRUE(csv.is_open());
    csv << "cell,storage,layers,initialized,numerical_inference_executed\n"
        << modelParityCase().testName() << ','
        << (placement->isGPUAdaptive()?std::string("adaptive_vram_ram_ssd"):
            placement->isHostAdaptive()?std::string(hostImmutableRowModeName(HostImmutableRowMode::AdaptiveRAMSSD)):
            ngramStorageLabel(placement->storage())) << ','
        << orch_runner_->executionPlan().layerCount() << ",1,0\n";
    csv.flush();
    ASSERT_TRUE(csv.good());
}

/** @brief Same canonical numerical framework, without unrelated source/snapshot-only registrations. */
class Qwen4ExpCPUJointFullStageTest
    :public Qwen4ExpConfigDrivenParityTest<Qwen4ExpCPUJointFullStageTest>,public ModelParityCaseParameter {};
/** @test Original complete frozen-history checkpoints under the actual coordinated socket-rank lifecycle. */
TEST_P(Qwen4ExpCPUJointFullStageTest,ProductionParity) {runQwen4ExpProductionParityCampaign();}
INSTANTIATE_TEST_SUITE_P(Qwen4ExpCPUJointWorld2FullStage,Qwen4ExpCPUJointFullStageTest,
    ::testing::ValuesIn(qwen4ExpCPUJointWorld2NumericalCases()),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});

/** @test Actual declared host authority, calibrated movement and epoch publication on original GPU/CPU2. */
INSTANTIATE_TEST_SUITE_P(Qwen4ExpCPUJointWorld2DynamicFullStage,Qwen4ExpCPUJointFullStageTest,
    ::testing::ValuesIn(qwen4ExpCPUJointWorld2DynamicNumericalCases()),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});

INSTANTIATE_TEST_SUITE_P(Qwen4ExpMain, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpMain", qwen4ExpProductionCases())),
    [](const ::testing::TestParamInfo<ModelParityCase> &info) { return info.param.testName(); });

// Canonical discovery sees these same ProductionParity/SnapshotAdmission tests
// with complete request identities; the unchanged main20 suite remains separate.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpHostAdaptiveRAMSSD, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpHostAdaptiveRAMSSD", qwen4ExpHostAdaptiveProductionCases())),
    [](const ::testing::TestParamInfo<ModelParityCase> &info) { return info.param.testName(); });

// Registration preserves every checkpoint and production-path proof. The
// unchanged ordinary/adaptive families retain their own exact identities.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpFixedHeadTP, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpFixedHeadTP", qwen4ExpFixedHeadProductionCases())),
    [](const ::testing::TestParamInfo<ModelParityCase> &info) { return info.param.testName(); });

// Four explicit diagnostic cells use the same main-source discovery and every
// mathematical/prefix/path gate. Explicit CLI selection is lossless; Auto and
// E2E eligibility remain unchanged and no qualification is claimed by declaration.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpGPUAdaptive, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpGPUAdaptive", qwen4ExpGPUAdaptiveProductionCases())),
    [](const ::testing::TestParamInfo<ModelParityCase> &info) { return info.param.testName(); });

// Explicit opposite-vendor STORAGE owners; compute remains one GPU+CPU ExpertOverlay.
// Retained15 is inactive ordinary storage, not active MTP qualification.
// Same canonical runtime/export cells own frozen-history accuracy, real expert
// movement and matched economic proof; registration never implies a passing run.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpConsumerLocalTP2StaticRetained15, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpConsumerLocalTP2StaticRetained15",qwen4ExpConsumerLocalTP2ExecutionCases(ModelParityExpertOverlayPolicy{}))),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpConsumerLocalTP2DynamicOrdinalRetained15, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpConsumerLocalTP2DynamicOrdinalRetained15",qwen4ExpConsumerLocalTP2ExecutionCases({.owner_order=RoutedExpertOwnerOrder::Ordinal,.movement=ModelParityExpertMovement::Dynamic}))),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpConsumerLocalTP2DynamicRandomRetained15, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpConsumerLocalTP2DynamicRandomRetained15",qwen4ExpConsumerLocalTP2ExecutionCases({.owner_order=RoutedExpertOwnerOrder::Random,.movement=ModelParityExpertMovement::Dynamic}))),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});

INSTANTIATE_TEST_SUITE_P(Qwen4ExpSharedCoordinatorTP2StaticRetained15, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpSharedCoordinatorTP2StaticRetained15",qwen4ExpSharedCoordinatorTP2ExecutionCases(ModelParityExpertOverlayPolicy{}))),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpSharedCoordinatorTP2DynamicOrdinalRetained15, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpSharedCoordinatorTP2DynamicOrdinalRetained15",qwen4ExpSharedCoordinatorTP2ExecutionCases({.owner_order=RoutedExpertOwnerOrder::Ordinal,.movement=ModelParityExpertMovement::Dynamic}))),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpSharedCoordinatorTP2DynamicRandomRetained15, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpSharedCoordinatorTP2DynamicRandomRetained15",qwen4ExpSharedCoordinatorTP2ExecutionCases({.owner_order=RoutedExpertOwnerOrder::Random,.movement=ModelParityExpertMovement::Dynamic}))),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});

INSTANTIATE_TEST_SUITE_P(Qwen4ExpOppositeVendorOrdinaryRetained15, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpOppositeVendorOrdinaryRetained15",qwen4ExpOppositeVendorOrdinaryRetained15Cases())),
    [](const ::testing::TestParamInfo<ModelParityCase> &info) { return info.param.testName(); });

// Active policies share the original source/placement and Off serial controls.
// Numerical acceptance still requires frozen-history panels; these declarations
// also provide the original typed runtime export for real generation diagnostics.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpOppositeVendorMTPRetained15, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpOppositeVendorMTPRetained15",qwen4ExpOppositeVendorMTPRetained15Cases())),
    [](const ::testing::TestParamInfo<ModelParityCase> &info) { return info.param.testName(); });

// Separate DynamicOrdinal/Random families preserve the original static16 and
// reuse the same numerical/movement checks and lossless runtime exporter.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpOppositeVendorDynamicOrdinalRetained15, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpOppositeVendorDynamicOrdinalRetained15",qwen4ExpOppositeVendorDynamicExpertRetained15Cases(RoutedExpertOwnerOrder::Ordinal))),
    [](const ::testing::TestParamInfo<ModelParityCase> &info) { return info.param.testName(); });
INSTANTIATE_TEST_SUITE_P(Qwen4ExpOppositeVendorDynamicRandomRetained15, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpOppositeVendorDynamicRandomRetained15",qwen4ExpOppositeVendorDynamicExpertRetained15Cases(RoutedExpertOwnerOrder::Random))),
    [](const ::testing::TestParamInfo<ModelParityCase> &info) { return info.param.testName(); });

// One exact all-routed-experts GPU TP4 request; real admission/numerical gates
// remain mandatory. RAM PLE is independent host storage, not CPU expert offload.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpFullROCmTP4, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpFullROCmTP4", qwen4ExpFullROCmTP4ProductionCases())),
    [](const ::testing::TestParamInfo<ModelParityCase> &info) { return info.param.testName(); });

// Three exact controls preserve the original full-stage/partial-prefix/path
// fixture and optional paired-document attachment. Original A1 remains separate.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpROCm4OverlayPolicy, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpROCm4OverlayPolicy", qwen4ExpROCm4OverlayPolicyProductionCases())),
    [](const ::testing::TestParamInfo<ModelParityCase> &info) { return info.param.testName(); });

/**
 * @brief Plain setup-only fixture for existing canonical source configurations.
 *
 * Unlike Qwen4ExpMainParityTest this inherits no numerical SetUp. Source and
 * physical admission remain real; reference directories and snapshot policies
 * never enter this fixture's lifecycle.
 */
class Qwen4ExpSourceAdmissionTest : public ::testing::Test, public Qwen4ExpSourceAdmissionParameter {};

/** @test Original source preparation, exact lookup owners and snapshot-zero PMA. */
TEST_P(Qwen4ExpSourceAdmissionTest, FullSourceAdmission)
{
    ASSERT_NO_FATAL_FAILURE(admitQwen4ExpSource(modelParityCase()));
}

// Source admission keeps real world2 socket roles; active MTP proof is separate.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpCPUJointWorld2SourceAdmission, Qwen4ExpSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpCPUJointWorld2MTPSourceAdmissionCases())),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase>& info){return info.param.testName();});

INSTANTIATE_TEST_SUITE_P(Qwen4ExpMainSourceAdmission, Qwen4ExpSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpExpandSourceAdmissionMTP(qwen4ExpProductionCases()))),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase> &info) { return info.param.testName(); });

INSTANTIATE_TEST_SUITE_P(Qwen4ExpHostAdaptiveSourceAdmission, Qwen4ExpSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpExpandSourceAdmissionMTP(qwen4ExpHostAdaptiveProductionCases()))),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase> &info) { return info.param.testName(); });

INSTANTIATE_TEST_SUITE_P(Qwen4ExpFixedHeadSourceAdmission, Qwen4ExpSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpFixedHeadSourceAdmissionParameters(qwen4ExpExpandSourceAdmissionMTP(qwen4ExpFixedHeadProductionCases()))),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase> &info) { return info.param.testName(); });

INSTANTIATE_TEST_SUITE_P(Qwen4ExpGPUAdaptiveSourceAdmission, Qwen4ExpSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpExpandSourceAdmissionMTP(qwen4ExpGPUAdaptiveSourceAdmissionSeeds()))),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase> &info) { return info.param.testName(); });

// Concurrent consumer-local channels retain the original four-policy source setup product.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpConsumerLocalTP2AdaptiveSourceAdmission, Qwen4ExpSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpExpandSourceAdmissionMTP(qwen4ExpConsumerLocalTP2AdaptiveSourceAdmissionSeeds()))),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase> &info) { return info.param.testName(); });

INSTANTIATE_TEST_SUITE_P(Qwen4ExpConsumerLocalTP4AdaptiveSourceAdmission, Qwen4ExpSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpExpandSourceAdmissionMTP(qwen4ExpConsumerLocalTP4AdaptiveSourceAdmissionSeeds()))),
    [](const auto& info){return info.param.testName();});

// Shared physical owners retain one real source and an explicit foreign result/ACK reader.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpSharedCoordinatorTP2AdaptiveSourceAdmission, Qwen4ExpSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpExpandSourceAdmissionMTP(qwen4ExpSharedCoordinatorTP2AdaptiveSourceAdmissionSeeds()))),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase> &info) { return info.param.testName(); });

INSTANTIATE_TEST_SUITE_P(Qwen4ExpSharedCoordinatorCUDATP2AdaptiveSourceAdmission, Qwen4ExpSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpExpandSourceAdmissionMTP(qwen4ExpSharedCoordinatorCUDATP2AdaptiveSourceAdmissionSeeds()))),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase> &info) { return info.param.testName(); });

INSTANTIATE_TEST_SUITE_P(Qwen4ExpOppositeVendorOrdinaryRetained15SourceAdmission, Qwen4ExpSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpOppositeVendorOrdinaryRetained15Cases())),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase> &info) { return info.param.testName(); });

INSTANTIATE_TEST_SUITE_P(Qwen4ExpOppositeVendorMTPRetained15SourceAdmission, Qwen4ExpSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpOppositeVendorMTPRetained15Cases())),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase> &info) { return info.param.testName(); });

INSTANTIATE_TEST_SUITE_P(Qwen4ExpFullROCmTP4SourceAdmission, Qwen4ExpSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpExpandSourceAdmissionMTP(qwen4ExpFullROCmTP4ProductionCases()))),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase> &info) { return info.param.testName(); });

INSTANTIATE_TEST_SUITE_P(Qwen4ExpROCm4OverlayPolicySourceAdmission, Qwen4ExpSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpROCm4OverlayPolicySourceAdmissionParameters(qwen4ExpExpandSourceAdmissionMTP(qwen4ExpROCm4OverlayPolicyProductionCases()))),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase> &info) { return info.param.testName(); });

// This explicit bounded family complements the unchanged full15 request;
// setup must still prove its actual physical fit and materialized ownership.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpFixedHeadFourExpertRetained3SourceAdmission, Qwen4ExpSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(
        qwen4ExpCUDAFixedHeadFourExpertRetained3SourceAdmissionCases())),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase> &info) { return info.param.testName(); });

/** @brief Physical-capacity rejection remains separate from unsupported shard policies. */
class Qwen4ExpCapacitySourceAdmissionTest : public ::testing::Test, public Qwen4ExpSourceAdmissionParameter {};

/** @test The explicit public cap rejects the unchanged larger request on every GPU capacity. */
TEST_P(Qwen4ExpCapacitySourceAdmissionTest, RejectCapacitySourceAdmission)
{
    ASSERT_TRUE(GetParam().gpu_memory_cap_mb.has_value());
    ASSERT_TRUE(GetParam().capacity_boundary.has_value());
    ASSERT_NO_FATAL_FAILURE(admitQwen4ExpSourceCapacityRejected(
        modelParityCase(), *GetParam().gpu_memory_cap_mb, *GetParam().capacity_boundary));
}

INSTANTIATE_TEST_SUITE_P(Qwen4ExpFixedHeadCapacitySourceAdmission, Qwen4ExpCapacitySourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpCUDAFixedHeadCapacityRejectionCases())),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase> &info) { return info.param.testName(); });

INSTANTIATE_TEST_SUITE_P(Qwen4ExpROCm4ReplicatedCapacitySourceAdmission, Qwen4ExpCapacitySourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpROCm4ReplicatedCapacityRejectionCases())),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase> &info) { return info.param.testName(); });

// Source-only extensions share this lifecycle without entering the numerical
// families or manufacturing reference packs for new physical compositions.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpExtendedSourceAdmission, Qwen4ExpSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpExpandSourceAdmissionMTP(qwen4ExpExtendedSourceAdmissionCases()))),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase> &info) { return info.param.testName(); });

/** @brief Unqualified composed topologies retain a positive materialization obligation. */
class Qwen4ExpCandidateSourceAdmissionTest : public ::testing::Test, public Qwen4ExpSourceAdmissionParameter {};

/** @test Candidate support requires the exact requested native graph and placement owners. */
TEST_P(Qwen4ExpCandidateSourceAdmissionTest, CandidateSourceAdmission)
{
    ASSERT_NO_FATAL_FAILURE(admitQwen4ExpSource(modelParityCase()));
}

INSTANTIATE_TEST_SUITE_P(Qwen4ExpAllGPUQ6DynamicOrdinalR3SourceAdmission, Qwen4ExpCandidateSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpAllGPUQ6MTPNativeExactCases())),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase>& info) { return info.param.testName(); });

INSTANTIATE_TEST_SUITE_P(Qwen4ExpComposedCandidateSourceAdmission, Qwen4ExpCandidateSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpExpandSourceAdmissionMTP(qwen4ExpComposedCandidateSourceAdmissionCases()))),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase> &info) { return info.param.testName(); });

// A distinct authored split retains full15 and all expert policies; original
// 16/32 candidates remain independent positive admission obligations.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpCUDA2CPU2FifteenLayerSourceAdmission, Qwen4ExpCandidateSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpExpandSourceAdmissionMTP(
        qwen4ExpCUDA2CPU2FifteenLayerSourceAdmissionCases()))),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase> &info) { return info.param.testName(); });

/** @brief Unsupported cross-tier expert slices retain an explicit rejection contract. */
class Qwen4ExpRejectedSourceAdmissionTest : public ::testing::Test, public Qwen4ExpSourceAdmissionParameter {};

/** @test A requested unsupported shard policy fails precisely without entering another mode. */
TEST_P(Qwen4ExpRejectedSourceAdmissionTest, RejectUnsupportedSourceAdmission)
{
    ASSERT_NO_FATAL_FAILURE(admitQwen4ExpSourceRejected(modelParityCase()));
}

INSTANTIATE_TEST_SUITE_P(Qwen4ExpRejectedSourceAdmission, Qwen4ExpRejectedSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpExpandSourceAdmissionMTP(qwen4ExpRejectedSourceAdmissionCases()))),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase> &info) { return info.param.testName(); });

/**
 * @brief Gold-free serving admission with only GoogleTest's empty setup lifecycle.
 *
 * This deliberately does not inherit the numerical fixture: its SetUp validates
 * or generates reference snapshots even when native snapshot capture is disabled.
 * The parameter adapter still owns the exact canonical topology/storage input.
 */
class Qwen4ExpNgramAdmissionTest : public ::testing::Test, public Qwen4ExpSourceAdmissionParameter {};

/**
 * @test Materialize the complete native source under the declared Automatic/VRAM policy.
 *
 * The normal runner owns weights, captured serving families and the table.
 * This setup-only proof admits ordinary serving without diagnostic snapshot
 * banks and writes no numerical or performance receipt. The CPU
 * tail owns layers2..47; ROCm0 owns layer0 and only ROCm1 owns layer1/PLE.
 * Observation aliases expire before the shared final-owner retirement and
 * disposable-prefix cleanup, including assertion/exception failure paths.
 */
TEST_P(Qwen4ExpNgramAdmissionTest, FullSourceAdmission)
{
    int rank = -1;
    int world_size = 0;
    ASSERT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &rank), MPI_SUCCESS);
    ASSERT_EQ(MPI_Comm_size(MPI_COMM_WORLD, &world_size), MPI_SUCCESS);
    ASSERT_EQ(world_size, modelParityCase().topology.mpi_ranks);
    auto config = qwen4ExpNgramAdmissionConfig(
        modelParityCase(), modelParityCase().model.model_path, rank);
    // Preserve the historical inactive probe; active setup uses production
    // verifier and kernel policy from the shared source-admission projection.
    if (modelParityCase().mtp == ModelParityMTP::Off) config.deterministic = true;
    publishOrchestrationStartupPolicy(config);
    initCPUBackend(-1);
    // Reuse the same startup-policy-scoped observation as generic admission;
    // CPU NUMA identity and GPU resolution never rediscover hardware after setup.
    auto &devices = DeviceManager::instance();
    if (!devices.hardware()) devices.initialize(-1, false);
    const auto *hardware = devices.hardware();
    ASSERT_NE(hardware, nullptr);
    ASSERT_NO_THROW(requireQwen4ExpSourceCPUInventory(modelParityCase(), rank, *hardware));
    for (const auto &participant : modelParityCase().topology.participants)
    {
        const auto type = participant.address.device_type;
        if (type == DeviceType::CPU) continue;
        auto *backend = type == DeviceType::CUDA ? getCUDABackend() : getROCmBackend();
        if (!backend || participant.address.device_ordinal >= backend->deviceCount())
            GTEST_SKIP() << "Missing declared admission backend " << participant.address.toShortString();
    }
    ASSERT_TRUE(std::filesystem::is_regular_file(config.model_path)) << config.model_path;
    if (config.shared_predictor)
        ASSERT_TRUE(std::filesystem::is_regular_file(config.shared_predictor->path()))
            << config.shared_predictor->path();
    const char *artifact_root = std::getenv("LLAMINAR_PRODUCTION_PARITY_ARTIFACT_ROOT");
    const auto base = artifact_root ? std::filesystem::path(artifact_root)
        : std::filesystem::current_path() / "parity-results" / "qwen-flash-next";
    ASSERT_TRUE(base.is_absolute()) << "Admission artifact root must be absolute";
    const auto output = base / "source-admission" / artifactIdentityPath(modelParityCase().testName()) / ("rank-" + std::to_string(rank));
    // This cell owns only its generated child. No gold directory is resolved,
    // read, generated or removed, including when no reference pack is installed.
    ASSERT_NO_THROW(prepareQwen4ExpAdmissionEvidenceDirectory(output));
    config.prefix_cache.enabled = true;
    config.prefix_cache.storage_mode = PrefixCacheStorageMode::Tiered;
    config.prefix_cache.block_size = productionParityPrefixRestoreProofBlockSize(
        modelParityCase().model.token_ids.size());
    config.prefix_cache.terminal_state = PrefixCacheTerminalStateMode::Auto;
    // Each existing special case owns one MPI process and one output child.
    // Keep its exact Auto/24GiB or whole-VRAM request; only disposable cache
    // namespace metadata changes to the common explicitly owned child.
    config.prefix_cache.disk_dir = (output / "prefix-cache").string();
    auto authored = config;
    if (config.moe_routed_expert_plan)
        authored.moe_routed_expert_plan = std::make_shared<MoERoutedExpertPlacementPlan>(*config.moe_routed_expert_plan);
    Qwen4ExpSourceAdmissionLifetime lifetime(config, output / "prefix-cache");
    observeQwen4ExpAdmissionThenRetire([&] {
        auto *runner = &lifetime.runner();
        runner->disableSnapshotCapture();
        std::vector<PerfStatRecord> mtp_setup_before;
        ASSERT_NO_THROW(mtp_setup_before = beginQwen4ExpSourceMTPEvidence(modelParityCase()));
        ASSERT_TRUE(runner->initialize()) << runner->lastError();
        ASSERT_TRUE(runner->isInitialized());
        EXPECT_TRUE(runner->getSnapshotKeys().empty());
        ASSERT_NO_FATAL_FAILURE(expectQwen4ExpSourceDomains(
            canonicalQwen4ExpSourceAdmissionConfig(authored, rank), runner->config()));
        ASSERT_NO_THROW(requireQwen4ExpSourceMaintenancePolicy(authored.moe_rebalance, runner->config().moe_rebalance));
        ASSERT_NO_THROW(requireQwen4ExpSourceMaintenancePolicy(authored.moe_rebalance, runner->executionPlan().runtime.moe_rebalance));
        ASSERT_NO_THROW(requireQwen4ExpSourceExecutionPlan(modelParityCase(), rank, authored,
            runner->executionPlan(), *hardware));
        const auto &prepared = lifetime.reuseContract();
        ASSERT_TRUE(prepared.has_value());
        const auto context = prepared->context;
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(runner->modelContextForDiagnostics(), context.get());
        EXPECT_EQ(context->path(), config.model_path);
        EXPECT_EQ(context->totalBlockCount(), 48);
        EXPECT_EQ(context->model().block_count, 48u);
        ASSERT_EQ(context->model().split_paths.size(), 4u);
        const auto source_artifacts = context->artifactPaths();
        ASSERT_NO_FATAL_FAILURE(expectQwen4ExpSourceMTPMaterialization(
            modelParityCase(), rank, authored, *runner, *prepared, output, mtp_setup_before));
        EXPECT_EQ(runner->executionPlan().layerCount(), 48);
        EXPECT_EQ(runner->config().max_seq_len, 4096);
        EXPECT_EQ(runner->config().ngram_table_policy, *modelParityCase().ngram_table_policy);
        const auto placement = context->resolvedImmutableRowTablePlacement();
        ASSERT_NE(placement, nullptr);
        ASSERT_NE(runner->executionPlan().runtime.ngram_table_placement, nullptr);
        EXPECT_EQ(*runner->executionPlan().runtime.ngram_table_placement, *placement);
        EXPECT_EQ(placement->storage(), qwen4ExpNgramAdmissionExpectedStorage(modelParityCase()));
        // Exact source metadata authenticates a full table, rather than a tiny
        // standalone fixture or an admitted subset of the original hash space.
        EXPECT_EQ(placement->extent().rows(), 320001536ull);
        EXPECT_EQ(placement->extent().rowBytes(), 90u);
        EXPECT_EQ(placement->extent().nativeBytes(), 28800138240ull);
        const auto authority = prepared->physical_memory_authority;
        ASSERT_NE(authority, nullptr);
        ASSERT_NE(authority->admission(), nullptr);
        ASSERT_EQ(placement->allocators().size(), 1u);
        const auto allocator = placement->allocators().front();
        const auto *bom = authority->admission()->plan().find(allocator);
        ASSERT_NE(bom, nullptr);
        for (const auto &resource : authority->admission()->plan().resources())
            EXPECT_EQ(resource.bytes(PhysicalMemoryOwner::GraphSnapshotArena), 0u);
        if (placement->storage() == ImmutableRowStorage::VRAM)
        {
            EXPECT_EQ(context->hostImmutableRowProvider(), nullptr);
            EXPECT_EQ(allocator.device, DeviceId::rocm(1));
            ASSERT_EQ(placement->consumers().size(), 1u);
            EXPECT_EQ(placement->consumers().front(), allocator);
            const auto table = context->residentImmutableRowTable(allocator.device);
            ASSERT_NE(table, nullptr);
            EXPECT_EQ(table->device(), allocator.device);
            ASSERT_NO_THROW(table->authenticateSource());
            ASSERT_NO_THROW(requireQwen4ExpSourceTableIdentity(
                context->model(), source_artifacts, table->source()));
            EXPECT_EQ(table->nativeBytes(), placement->extent().nativeBytes());
            EXPECT_EQ(context->residentImmutableRowTableBytes(allocator.device), table->nativeBytes());
            EXPECT_EQ(bom->bytes(PhysicalMemoryOwner::ImmutableTableStorage), table->nativeBytes());
            EXPECT_EQ(authority->claimedBytes(allocator.device, PhysicalMemoryOwner::ImmutableTableStorage,
                      PhysicalMemoryMaterializationKind::NewAllocation), table->nativeBytes());
        }
        else
        {
            EXPECT_TRUE(allocator.device.is_cpu());
            const auto table = context->hostImmutableRowProvider();
            ASSERT_NE(table, nullptr);
            EXPECT_EQ(table->wholeStorage(), ImmutableRowStorage::RAM);
            ASSERT_NO_THROW(table->authenticateSource());
            ASSERT_NO_THROW(requireQwen4ExpSourceTableIdentity(
                context->model(), source_artifacts, table->source()));
            EXPECT_EQ(table->source().tensorBytes(), placement->extent().nativeBytes());
            const auto retained_owner = table->retainedOwnerObservation();
            EXPECT_EQ(retained_owner.authority().get(), authority.get());
            EXPECT_EQ(retained_owner.allocator(), allocator);
            EXPECT_EQ(retained_owner.hostPlacement(), placement->hostPlacement());
            EXPECT_EQ(retained_owner.lookupStagingBytes(), 0u);
            EXPECT_EQ(bom->bytes(PhysicalMemoryOwner::ImmutableTableStorage), retained_owner.tableStorageBytes());
            EXPECT_EQ(authority->claimedBytes(allocator.device, PhysicalMemoryOwner::ImmutableTableStorage,
                      PhysicalMemoryMaterializationKind::NewAllocation), retained_owner.tableStorageBytes());
            for (const auto &consumer : placement->consumers())
                EXPECT_EQ(context->residentImmutableRowTableBytes(consumer.device), 0u);
            if (modelParityCase().topology.isExpertOverlay())
            {
                ASSERT_EQ(config.max_gpu_memory_mb, 24u * 1024u);
                EXPECT_GT(placement->extent().nativeBytes(), *config.max_gpu_memory_mb * 1024ull * 1024ull);
            }
        }
        std::ofstream memory(output / "admission_memory.txt");
        ASSERT_TRUE(memory.is_open());
        memory << authority->rankAttestationSummary(true);
        ASSERT_NO_THROW(requireQwen4ExpCPUStorageMaterialization(authority->rankAttestation(),
            prepared->prepared_routed_weight_plan, placement->allocators()));
        memory.flush();
        ASSERT_TRUE(memory.good());
        std::ofstream csv(output / "admission.csv");
        ASSERT_TRUE(csv.is_open());
        csv << "cell,storage,layers,initialized,numerical_inference_executed,native_table_bytes,snapshots_enabled\n"
            << modelParityCase().testName() << ',' << ngramStorageLabel(placement->storage())
            << ",48,1,0," << placement->extent().nativeBytes() << ",0\n";
        csv.flush();
        ASSERT_TRUE(csv.good());
    }, [&] {
        // Fatal assertions return from the observation callback, after dropping
        // its aliases. They must not bypass final retirement or its receipt.
        ASSERT_NO_THROW(lifetime.finish());
        std::ofstream retirement(output / "retirement.csv");
        ASSERT_TRUE(retirement.is_open());
        retirement << "cell,runner_retired,model_retired,prefix_removed,retired_gpu_generations\n"
            << modelParityCase().testName() << ",1,1,1," << lifetime.retiredDeviceCount() << '\n';
        retirement.flush();
        ASSERT_TRUE(retirement.good());
    });
}

INSTANTIATE_TEST_SUITE_P(Qwen4ExpNgramAdmission, Qwen4ExpNgramAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpExpandSourceAdmissionMTP(qwen4ExpNgramAdmissionCases()))),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase> &info) { return info.param.testName(); });

/** @brief Preserve the canonical MPI lifecycle and retire backend authorities before process exit. */
int main(int argc, char **argv)
{
    if (auto terminal = llaminar2::MPISessionOwner::enterProcess(argc, argv))
        return *terminal;
    const auto completion = llaminar2::test::NativeLifetimeCompletion::fromEnvironment();
    // The original MPI session outlives all test runners and their CPU domain
    // commitments. Its destructor finalizes MPI before the enclosing lexical
    // GPU lifetime returns and permits terminal publication. Bare MPI_Init
    // cannot supply that ownership to admitted CPU expert collectives.
    int result = llaminar2::test::runWithNativeLifetimeCompletion<llaminar2::GPUDeviceProcessLifetime>(
        [&] {
            auto mpi_session = llaminar2::MPIProcessSession::initialize(argc, argv);
            int result = 0;
            constexpr std::string_view inventory_option = "--qfn-source-hardware-inventory=";
            if (argc == 2 && std::string_view(argv[1]).starts_with(inventory_option))
            {
                try { exportQwen4ExpSourceHardware(std::string_view(argv[1]).substr(inventory_option.size())); }
                catch (const std::exception &error) { std::cerr << error.what() << '\n'; result = 2; }
            }
            else
            {
                ::testing::InitGoogleTest(&argc, argv);
                result = RUN_ALL_TESTS();
            }
            GlobalBackendRouter::shutdown();
            llaminar2::GPUDeviceProcessLifetime::retireActive();
            return result;
        },
        [&](int result) { if (completion) completion->publish(result); });
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}

// Whole-source controls use supported original execution without GPU-owner colocation.
// Fresh paired bindings are required before their diagnostic benchmark is eligible.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpHardwareProductionDefaults, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpHardwareProductionDefaults", qwen4ExpHardwareProductionSourceCases())),
    [](const ::testing::TestParamInfo<ModelParityCase>& info) { return info.param.testName(); });
INSTANTIATE_TEST_SUITE_P(Qwen4ExpHardwareProductionDefaultsSourceAdmission, Qwen4ExpSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpHardwareProductionSourceAdmissionCases())),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase>& info) { return info.param.testName(); });

// Experimental matched residency, never a production-default or fit certificate.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpHardwareMatchedCacheEconomy, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpHardwareMatchedCacheEconomy", qwen4ExpHardwareMatchedCacheEconomyCases())),
    [](const ::testing::TestParamInfo<ModelParityCase>& info) { return info.param.testName(); });
// Active MTP source materialization is independent of the unchanged Off economy baseline.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpSharedContinuationMTPSourceAdmission, Qwen4ExpSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpSharedContinuationMTPSourceAdmissionCases())),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase>& info){return info.param.testName();});

INSTANTIATE_TEST_SUITE_P(Qwen4ExpHardwareMatchedCacheEconomySourceAdmission, Qwen4ExpSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpHardwareMatchedCacheEconomyCases())),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase>& info) { return info.param.testName(); });

// Same matched original expert/cache workload, genuine single full-RAM source; fresh bindings required.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpHardwareMatchedCacheRAMBackingEconomy, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpHardwareMatchedCacheRAMBackingEconomy", qwen4ExpHardwareMatchedCacheRAMBackingEconomyCases())),
    [](const ::testing::TestParamInfo<ModelParityCase>& info) { return info.param.testName(); });
INSTANTIATE_TEST_SUITE_P(Qwen4ExpHardwareMatchedCacheRAMBackingEconomySourceAdmission, Qwen4ExpSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpHardwareMatchedCacheRAMBackingEconomyCases())),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase>& info) { return info.param.testName(); });

// Dedicated movement workload retains original histories/tiers and native exact controls.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpConsumerLocalTP4MovementOrdinalRetained15Exact, Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpConsumerLocalTP4MovementNativeExactCases(RoutedExpertOwnerOrder::Ordinal)),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpConsumerLocalTP4MovementRandomRetained15Exact, Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpConsumerLocalTP4MovementNativeExactCases(RoutedExpertOwnerOrder::Random)),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpConsumerLocalTP4MovementOrdinalRetained15, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpConsumerLocalTP4MovementOrdinalRetained15",
        qwen4ExpConsumerLocalTP4MovementCases(RoutedExpertOwnerOrder::Ordinal))),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpConsumerLocalTP4MovementRandomRetained15, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpConsumerLocalTP4MovementRandomRetained15",
        qwen4ExpConsumerLocalTP4MovementCases(RoutedExpertOwnerOrder::Random))),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpConsumerLocalTP4MovementSourceAdmission, Qwen4ExpSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpExpandSourceAdmissionMTP(
        qwen4ExpConsumerLocalTP4MovementSourceAdmissionSeeds()))),
    [](const auto& info){return info.param.testName();});

// Unexecuted Q6 candidates retain distinct setup, numerical and exact-MTP obligations.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpExpertFollowerOwnerSourceAdmissionCandidate, Qwen4ExpCandidateSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpExpandSourceAdmissionMTP(qwen4ExpExpertFollowerOwnerSourceAdmissionSeeds()))),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase>& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpExpertFollowerOwnerStaticOrdinalCandidate, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpExpertFollowerOwnerStaticOrdinalCandidate",qwen4ExpExpertFollowerOwnerExecutionCases(ModelParityExpertOverlayPolicy{}))),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpExpertFollowerOwnerStaticOrdinalCandidateExact, Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpExpertFollowerOwnerNativeExactCases(ModelParityExpertOverlayPolicy{})),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpExpertFollowerOwnerDynamicOrdinalCandidate, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpExpertFollowerOwnerDynamicOrdinalCandidate",qwen4ExpExpertFollowerOwnerExecutionCases(ModelParityExpertOverlayPolicy{.owner_order=RoutedExpertOwnerOrder::Ordinal,.movement=ModelParityExpertMovement::Dynamic}))),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpExpertFollowerOwnerDynamicOrdinalCandidateExact, Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpExpertFollowerOwnerNativeExactCases(ModelParityExpertOverlayPolicy{.owner_order=RoutedExpertOwnerOrder::Ordinal,.movement=ModelParityExpertMovement::Dynamic})),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});

//40 source /40 numerical /30 active exact candidates; declarations are not qualification.
class Qwen4ExpOwnerSubsetSourceAdmissionTest : public ::testing::Test, public Qwen4ExpSourceAdmissionParameter {};
/** @test Real subset source admission must consume actual original cold recreation and retirement. */
TEST_P(Qwen4ExpOwnerSubsetSourceAdmissionTest, CandidateSourceAdmission)
{
    ASSERT_NO_FATAL_FAILURE(admitQwen4ExpSource(modelParityCase(),Qwen4ExpSourceRecreationProof::Required));
}
INSTANTIATE_TEST_SUITE_P(Qwen4ExpOwnerSubsetSourceAdmissionCandidate, Qwen4ExpOwnerSubsetSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpExpandSourceAdmissionMTP(
        qwen4ExpOwnerSubsetSourceAdmissionSeeds(),Qwen4ExpSourceMTPEnvelope::Retained3))),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpCombinedLocalHotC0SourceCandidate,Qwen4ExpOwnerSubsetSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpCombinedLocalHotC0Cases())),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpLocalHotBudgetSourceCandidate,Qwen4ExpOwnerSubsetSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpLocalHotBudgetCases())),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpAllGPUQ6LocalHotSourceCandidate,Qwen4ExpOwnerSubsetSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpAllGPUQ6LocalHotCases())),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpCUDALocalHotSourceCandidate,Qwen4ExpOwnerSubsetSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpCUDALocalHotCases())),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpLocalHotRecurrentSourceCandidate,Qwen4ExpOwnerSubsetSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpLocalHotRecurrentCases())),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpOwnerSubsetStaticOrdinalCandidate, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpOwnerSubsetStaticOrdinalCandidate",
        qwen4ExpOwnerSubsetExecutionCases(ModelParityExpertOverlayPolicy{}))),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpOwnerSubsetDynamicOrdinalCandidate, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpOwnerSubsetDynamicOrdinalCandidate",
        qwen4ExpOwnerSubsetExecutionCases({.owner_order=RoutedExpertOwnerOrder::Ordinal,.movement=ModelParityExpertMovement::Dynamic}))),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpOwnerSubsetStaticOrdinalCandidateExact, Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpOwnerSubsetNativeExactCases(ModelParityExpertOverlayPolicy{})),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpOwnerSubsetDynamicOrdinalCandidateExact, Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpOwnerSubsetNativeExactCases({.owner_order=RoutedExpertOwnerOrder::Ordinal,.movement=ModelParityExpertMovement::Dynamic})),
    [](const auto& info){return info.param.testName();});

namespace llaminar2::test::parity::qwen4exp {
INSTANTIATE_TEST_SUITE_P(Qwen4ExpLocalHotSSDOrdinaryCandidate, Qwen4ExpOrdinaryGenerationTest,
    ::testing::ValuesIn(qwen4ExpLocalHotSSDOrdinaryCases()),
    [](const auto& info) { return info.param.testName(); });
INSTANTIATE_TEST_SUITE_P(Qwen4ExpLocalHotSSDExactCandidate, Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpLocalHotSSDExactCases()),
    [](const auto& info) { return info.param.testName(); });
INSTANTIATE_TEST_SUITE_P(Qwen4ExpLocalHotSSDSourceCandidate, Qwen4ExpOwnerSubsetSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpLocalHotSSDCases())),
    [](const auto& info) { return info.param.testName(); });
}

INSTANTIATE_TEST_SUITE_P(Qwen4ExpLocalHotMeasuredBudgetSourceCandidate, Qwen4ExpOwnerSubsetSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpLocalHotMeasuredBudgetCandidates())),
    [](const auto& info) { return info.param.testName(); });
INSTANTIATE_TEST_SUITE_P(Qwen4ExpLocalHotMeasuredBudgetNumerical, Qwen4ExpLocalHotNumericalTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpLocalHotMeasuredBudgetNumerical",
        qwen4ExpLocalHotMeasuredBudgetCandidates())),
    [](const auto& info) { return info.param.testName(); });

#include "Qwen4ExpLocalHotHeldSourceTest.h"
/** @test Genuine unequal-cohort current LocalRead progress and original source retirement. */
TEST_P(Qwen4ExpLocalHotHeldSourceTest, CapturedLocalReadProgress)
{
    runHeldSourceProof();
}
INSTANTIATE_TEST_SUITE_P(Qwen4ExpLocalHotHeldSourceCandidate, Qwen4ExpLocalHotHeldSourceTest,
    ::testing::ValuesIn(qwen4ExpLocalHotHeldSourceCases()),
    [](const auto& info) { return info.param.testName(); });
INSTANTIATE_TEST_SUITE_P(Qwen4ExpLocalHotHeldSourceAdmissionCandidate, Qwen4ExpOwnerSubsetSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpLocalHotHeldSourceCases())),
    [](const auto& info) { return info.param.testName(); });

INSTANTIATE_TEST_SUITE_P(Qwen4ExpLocalHotRefinedBudgetSourceCandidate, Qwen4ExpOwnerSubsetSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpLocalHotRefinedBudgetCandidates())),
    [](const auto& info) { return info.param.testName(); });
INSTANTIATE_TEST_SUITE_P(Qwen4ExpLocalHotRefinedBudgetNumerical, Qwen4ExpLocalHotNumericalTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpLocalHotRefinedBudgetNumerical",
        qwen4ExpLocalHotRefinedBudgetCandidates())),
    [](const auto& info) { return info.param.testName(); });

/** @test Real-source rank-local TP domains linked by original cross-rank activation ownership. */
INSTANTIATE_TEST_SUITE_P(Qwen4ExpCrossRankPipelineFullStage,Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qwen4ExpCrossRankPipelineOrdinaryCases()),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});
/** @test Public real-GGUF generation followed by original request reset/reuse. */
INSTANTIATE_TEST_SUITE_P(Qwen4ExpCrossRankPipelineOrdinary,Qwen4ExpOrdinaryGenerationTest,
    ::testing::ValuesIn(qwen4ExpCrossRankPipelineOrdinaryCases()),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});
/** @test Fixed1/fixed3/dynamic serial/grouped bytes; real-source evidence is qualified independently. */
INSTANTIATE_TEST_SUITE_P(Qwen4ExpCrossRankPipelineMTPExact,Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpCrossRankPipelineMTPExactCases()),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});

/** @brief Real original-inventory pipeline source request; no numerical setup or fake GPU address. */
class Qwen4ExpObservedPipelineSourceAdmissionTest
    :public ::testing::Test,public Qwen4ExpSourceAdmissionParameter {};
/** @test Original MPI/NUMA/UUID projection followed by full ordinary prepared/captured source admission. */
TEST_P(Qwen4ExpObservedPipelineSourceAdmissionTest,CandidateSourceAdmission) {
    withQwen4ExpObservedPipelineAdmission(modelParityCase(),[](const auto& bound,const auto& execution) {
        admitQwen4ExpSource(bound,Qwen4ExpSourceRecreationProof::FromOriginalRoles,{},execution);
    });
}
INSTANTIATE_TEST_SUITE_P(Qwen4ExpObservedMI50PipelineSourceCandidate,Qwen4ExpObservedPipelineSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpObservedMI50PipelineSourceCases())),
    [](const auto& info){return info.param.testName();});


/** @brief Observed main-model numerical campaign with the original full reference policy. */
class Qwen4ExpObservedPipelineMainParityTest
    :public Qwen4ExpObservedPipelineParityFixture<Qwen4ExpObservedPipelineMainParityTest> {};

/** @test Original numerical, prefix and graph evidence through the selected Q7 pipeline. */
TEST_P(Qwen4ExpObservedPipelineMainParityTest,ProductionParity) {
    if(executesQwen4ExpBody())runQwen4ExpProductionParityCampaign();
}

INSTANTIATE_TEST_SUITE_P(Qwen4ExpObservedMI50FixedVRAMSourceCandidate,Qwen4ExpObservedPipelineSourceAdmissionTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(qwen4ExpObservedMI50FixedVRAMSourceCases())),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpObservedMI50FixedVRAMOrdinary,Qwen4ExpOrdinaryGenerationTest,
    ::testing::ValuesIn(qwen4ExpObservedMI50FixedVRAMCases()),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpObservedMI50FixedVRAMFullStage,Qwen4ExpObservedPipelineMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpObservedMI50FixedVRAMFullStage",
        qwen4ExpObservedMI50FixedVRAMCases())),
    [](const auto& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpObservedMI50FixedVRAMExact,Qwen4ExpMTPExactReplayTest,
    ::testing::ValuesIn(qwen4ExpObservedMI50FixedVRAMExactCases()),
    [](const auto& info){return info.param.testName();});
