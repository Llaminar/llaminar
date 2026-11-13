/**
 * @file Test__Qwen4ExpReleaseGates.cpp
 * @brief Device-free canonical QFN release coverage and export regressions.
 *
 * The six source-owned configurations must export native placement and
 * production-default benchmark intent. OpenCode consumes that same serving
 * manifest; diagnostic siblings cannot add a release claim. No model, accelerator or numerical certificate is
 * manufactured by these inventory checks.
 */
#include "integration/parity/qwen4exp/Qwen4ExpReleaseCases.h"
#include "integration/parity/qwen38/Qwen38ModelParityDefinitions.h"
#include "integration/parity/qwen36/Ornith15ModelParityDefinitions.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>

namespace llaminar2::test::parity
{
/** @test The sole blessed inventory contains two models times three exact hardware shapes. */
TEST(Qwen4ExpReleaseGates, ExactlySixCellsCarryTheReleaseContract)
{
    auto cells=qwen4exp::qwen4ExpReleaseCases();
    for(const auto &definition:qwen38::qwen38DenseMultiDeviceDefinitions())
        for(auto cell:expandModelParityDefinition(definition))
            if(cell.e2e_certification)cells.push_back(std::move(cell));
    ASSERT_EQ(cells.size(),6u);
    std::map<std::string,std::set<std::map<std::string,int>>> shapes;
    for(const auto &cell:cells)
    {
        SCOPED_TRACE(cell.testName());
        ASSERT_TRUE(cell.e2e_certification);
        EXPECT_EQ(cell.inventory_role,ModelParityInventoryRole::Canonical);
        shapes[cell.model.model_path].insert(modelParityAutomaticDeviceCounts(cell));
        std::ostringstream output;PrintTo(cell,&output);
        const auto record=nlohmann::json::parse(output.str());
        EXPECT_EQ(record.at("benchmark").at("policy"),"production_defaults");
        EXPECT_FALSE(record.at("benchmark").at("args").empty());
        if(cell.topology.kind==ModelParityTopologyKind::RankLocalPipelineParallel)
        {
            const auto runtime=cell.makeOrchestrationConfig(cell.model.model_path,0);
            ASSERT_EQ(runtime.domain_definitions.size(),3u);
            EXPECT_EQ(runtime.domain_definitions[0].devices.size(),2u);
            EXPECT_EQ(runtime.domain_definitions[1].devices.size(),4u);
            EXPECT_EQ(runtime.domain_definitions[2].devices.size(),1u);
            // A single CPU participant has no TP collective to select.
            EXPECT_EQ(runtime.domain_definitions[2].backend,CollectiveBackendType::AUTO);
        }
        if(cell.model.model_path==qwen4exp::kMainModelPath)
        {
            EXPECT_EQ(cell.mtp,ModelParityMTP::Off);
            ASSERT_TRUE(cell.ngram_table_policy);
            EXPECT_EQ(cell.ngram_table_policy->storage,ImmutableRowStoragePreference::RAM);
        }
        else EXPECT_EQ(cell.mtp,ModelParityMTP::DynamicDepth);
    }
    const std::set<std::map<std::string,int>> expected={{{"rocm",4}},{{"cuda",2}},{{"cuda",2},{"rocm",4},{"cpu",1}}};
    ASSERT_EQ(shapes.size(),2u);
    for(const auto &[model,observed]:shapes)EXPECT_EQ(observed,expected)<<model;
}

/** @test Broader QFN storage and former model selections retain math without a release claim. */
TEST(Qwen4ExpReleaseGates, DiagnosticMatricesCannotExpandTheBlessedScope)
{
    for(const auto *cells:{&qwen4exp::qwen4ExpProductionCases(),&qwen4exp::qwen4ExpHostAdaptiveProductionCases()})
        for(const auto &cell:*cells)EXPECT_FALSE(cell.e2e_certification);
    auto old=qwen36::qwen36MoECPU2NodeTPParityDefinition();
    for(const auto &definition:qwen36::withOrnith15CertificationModels(std::array{old}))
        for(const auto &cell:expandModelParityDefinition(definition))EXPECT_FALSE(cell.e2e_certification);
}
} // namespace llaminar2::test::parity
