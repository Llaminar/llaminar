/**
 * @file Test__Qwen4Exp_PooledCurrentDemand_Candidate.cpp
 * @brief Explicit grouped-current source and native exact candidates on genuine Rank-owned graphs.
 *
 * Source admission observes immutable original captured frontiers and replacement
 * retirement. Separate native exact children reuse the original functional proof;
 * capture observations never stand in for executed bytes or numerical evidence, which still requires the frozen-history panel contract.
 */
#include "Qwen4ExpParityTestBase.h"
#include "Qwen4ExpSourceAdmission.h"
#include "Qwen4ExpSourceAdmissionCI.h"
#include "Qwen4ExpCurrentDemandObservation.h"
#include "utils/DebugEnv.h"
#include <set>
using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen4exp;
namespace {
/** @return Active retained3 cells projected from the original four seeds. */
const std::vector<ModelParityCase>& currentDemandCases(){return qwen4ExpPooledCurrentMTPDemandCases();}
/** @brief Require the discovery-authored schedule before any runner/source construction. */
void requireCandidateSchedule(){
    if(debugEnv().execution.qfn_ngram_lookup_schedule!=ImmutableLookupSchedule::EarlyCurrentHostedCandidate)
        throw std::logic_error("Current-demand candidate requires its exact before-startup schedule environment");
}
/** @brief Validate actual role coverage without deriving runtime authority from a case name. */
void requireRoles(const ModelParityCase& cell,const std::vector<current_demand_observation::Graph>& rows){
    std::set<DeviceId> sources,readers,verifiers;
    const auto expected=qwen4ExpCurrentDemandRole(cell)==Qwen4ExpCurrentDemandRole::PureReader?
        PooledImmutableDemandRole::PureReader:PooledImmutableDemandRole::InlineOwner;
    for(const auto& row:rows){
        if(row.verifier)verifiers.insert(row.device);
        if(row.role==PooledImmutableDemandRole::Source)sources.insert(row.device);
        else {if(row.role!=expected)throw std::logic_error("Actual captured reader role differs from authored owner geometry");readers.insert(row.device);}
    }
    if(sources.size()!=1 || readers.size()!=1)throw std::logic_error("Current-demand candidate lost its genuine TP2 source/reader coverage");
    if(cell.mtp!=ModelParityMTP::Off && verifiers.size()!=2)
        throw std::logic_error("Active current-demand candidate did not prove both actual grouped target participant graphs");
}
}
/** @brief Gold-free original source lifecycle, selected before native startup. */
class Qwen4ExpPooledCurrentSourceTest:public ::testing::Test,public Qwen4ExpSourceAdmissionParameter {
protected:
    /** @brief Authenticate immutable schedule before constructing the original source. */
    void SetUp() override {ASSERT_NO_THROW(requireCandidateSchedule());}
};
/** @test Actual original and replacement graphs preserve role/capture identity and retire old source tokens. */
TEST_P(Qwen4ExpPooledCurrentSourceTest,FullSourceAdmission){
    std::vector<std::weak_ptr<const RetainedGraphParticipantPreparationStream>> old;
    const auto inspect=[&](const IOrchestrationRunner& runner,Qwen4ExpSourceInspectionPhase phase,const std::filesystem::path& output){
        if(runner.executionPlan().runtime.ngram_lookup_schedule!=ImmutableLookupSchedule::EarlyCurrentHostedCandidate)
            throw std::logic_error("Actual runner did not retain its explicitly selected current-demand schedule");
        if(phase==Qwen4ExpSourceInspectionPhase::RetainedReplacement)
            for(const auto& token:old)if(!token.expired())throw std::logic_error("Replacement retained an original retired graph source token");
        const auto rows=OrchestrationRunnerResolvedAdmissionTestAccess::currentDemand(runner);
        requireRoles(modelParityCase(),rows);
        std::ofstream csv(output/(phase==Qwen4ExpSourceInspectionPhase::Original?"current_demand_original.csv":"current_demand_replacement.csv"));
        csv<<"device,role,rows,decode,topology,variant,capture_generation,verifier,execution_proven\n";
        for(const auto& row:rows){
            csv<<row.device.toString()<<','<<static_cast<int>(row.role)<<','<<row.rows<<','<<row.decode<<','<<row.topology<<','<<row.variant<<','<<row.capture_generation<<','<<row.verifier<<",0\n";
            if(phase==Qwen4ExpSourceInspectionPhase::Original && !row.source.expired())old.push_back(row.source);
        }
        if(phase==Qwen4ExpSourceInspectionPhase::Original && old.empty())throw std::logic_error("Original current-demand capture omitted source token witnesses");
        if(!csv.good())throw std::runtime_error("Current-demand immutable admission observation could not be written");
    };
    ASSERT_NO_FATAL_FAILURE(admitQwen4ExpSource(modelParityCase(),
        qwen4ExpCurrentDemandRole(modelParityCase())==Qwen4ExpCurrentDemandRole::InlineOwner
            ?Qwen4ExpSourceRecreationProof::Required:Qwen4ExpSourceRecreationProof::FromOriginalRoles,inspect));
}
/** @brief Original native serial/grouped byte replay; no numerical scorer/reference path is added. */
class Qwen4ExpPooledCurrentExactTest:public Qwen4ExpConfigDrivenParityTest<Qwen4ExpPooledCurrentExactTest>,public ModelParityCaseParameter {
    using Base=Qwen4ExpConfigDrivenParityTest<Qwen4ExpPooledCurrentExactTest>;
protected:
    /** @brief Select the schedule before original setup and source creation. */
    void SetUp() override {ASSERT_NO_THROW(requireCandidateSchedule());Base::SetUp();}
    /** @return Existing functional exact contract, distinct from numerical panels. */
    MTPParityProofPurpose mtpParityProofPurpose()const noexcept override{return MTPParityProofPurpose::NativeExactReplay;}
    /** @return Existing declared native functional history. */
    int nativeMTPFunctionalHistoryRows()const noexcept override{return 32;}
    /** @brief Functional replay may price authenticated headers but cannot generate numerical snapshots. */
    bool regeneratePyTorchSnapshots()override{ADD_FAILURE()<<"Pooled current exact replay cannot generate numerical snapshots";return false;}
};
/** @test Real serial/grouped byte proof plus actual executed original grouped demand graphs. */
TEST_P(Qwen4ExpPooledCurrentExactTest,MTPExactReplay){
    ASSERT_NO_FATAL_FAILURE(runQwen4ExpMTPExactReplay());
    ASSERT_NE(orch_runner_,nullptr);
    ASSERT_EQ(orch_runner_->executionPlan().runtime.ngram_lookup_schedule,ImmutableLookupSchedule::EarlyCurrentHostedCandidate);
    ASSERT_NO_THROW(requireRoles(modelParityCase(),OrchestrationRunnerResolvedAdmissionTestAccess::currentDemand(*orch_runner_,true)));
}
INSTANTIATE_TEST_SUITE_P(Qwen4ExpPooledCurrentDemandSource,Qwen4ExpPooledCurrentSourceTest,
    ::testing::ValuesIn(qwen4ExpSourceAdmissionParameters(currentDemandCases())),
    [](const ::testing::TestParamInfo<Qwen4ExpSourceAdmissionCase>& info){return info.param.testName();});
INSTANTIATE_TEST_SUITE_P(Qwen4ExpPooledCurrentDemandExact,Qwen4ExpPooledCurrentExactTest,
    ::testing::ValuesIn(currentDemandCases()),
    [](const ::testing::TestParamInfo<ModelParityCase>& info){return info.param.testName();});
