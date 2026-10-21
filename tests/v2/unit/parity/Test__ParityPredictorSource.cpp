/**
 * @file Test__ParityPredictorSource.cpp
 * @brief Device-free explicit predictor propagation and source admission proofs.
 *
 * No weights, runner, core DSO or accelerator is loaded by the standalone proof.
 * The same inline adapter builds the real ModelContextConfig used by the fixture.
 */
#include "integration/parity/ModelParityDefinition.h"
#include "integration/parity/ParityPredictorSource.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <optional>
#include <unistd.h>
namespace llaminar2::test::parity {
namespace {
/** @brief Restore one process environment word even after an assertion throws. */
class Environment final {
 std::string name_; std::optional<std::string> old_;
public:
 /** @brief Save the previous value and install the test-owned setting. */
 Environment(std::string name,const std::optional<std::string>&value):name_(std::move(name)) {
  if(const auto*p=std::getenv(name_.c_str()))old_=p;
  if(value)setenv(name_.c_str(),value->c_str(),1);else unsetenv(name_.c_str());
 }
 /** @brief Retire only this test's process metadata. */
 ~Environment(){if(old_)setenv(name_.c_str(),old_->c_str(),1);else unsetenv(name_.c_str());}
};
/** @brief Create a bounded real filesystem fixture with no GGUF payload parsing. */
class Files final {
public:
 std::filesystem::path root;
 /** @brief Make an isolated directory and remove it at teardown. */
 Files(){std::string p="/tmp/qfn-parity-source-XXXXXX";root=mkdtemp(p.data());}
 /** @brief Remove only marker files owned by this isolated fixture. */
 ~Files(){std::filesystem::remove_all(root);}
 /** @brief Publish one small marker artifact for source-path authority tests. */
 std::string file(const std::string&name){auto p=root/name;std::ofstream(p)<<"fixture";return p.string();}
};
}
TEST(ParityPredictorSource, AbsentSelectionPreservesLegacyIdentityAndLoaderDefaults){
 Environment campaign("LLAMINAR_PRODUCTION_PARITY_PROCESS_CAMPAIGN",std::nullopt);
 Environment ramdisk("LLAMINAR_PRODUCTION_PARITY_MODEL_RAMDISK",std::nullopt);
 EXPECT_FALSE(resolveParityPredictorSource(std::nullopt));
 EXPECT_FALSE(bindParityPredictorSource(std::nullopt,std::nullopt));
 EXPECT_TRUE(parityPredictorIdentity(std::nullopt).empty());
 const auto config=parityModelContextConfig(nullptr,WeightDistributionStrategy::REPLICATED,std::nullopt,0);
 EXPECT_FALSE(config.shared_predictor);
 EXPECT_EQ(config.predictor_materialization,SharedPredictorMaterialization::Inactive);
 EXPECT_EQ(config.strategy,WeightDistributionStrategy::REPLICATED);
}
TEST(ParityPredictorSource, DefinitionAdapterAndContextRetainInactiveSecondSource){
 ModelParityCase cell;
 cell.model.test_id="source";cell.model.model_path="/models/main.gguf";
 cell.model.shared_predictor.emplace("/models/explicit predictor.gguf");
 cell.topology.test_id="CPU";cell.topology.kind=ModelParityTopologyKind::SingleDevice;
 cell.topology.participants={{{GlobalDeviceAddress::cpu(0)},0}};
 const auto adapter=cell.toTestConfig();
 ASSERT_EQ(adapter.shared_predictor,cell.model.shared_predictor);
 EXPECT_FALSE(cell.mtpEnabled());
 const auto inactive=parityModelContextConfig(nullptr,WeightDistributionStrategy::REPLICATED,adapter.shared_predictor,0);
 EXPECT_EQ(inactive.shared_predictor,cell.model.shared_predictor);
 EXPECT_EQ(inactive.predictor_materialization,SharedPredictorMaterialization::Inactive);
 const auto retained=parityModelContextConfig(nullptr,WeightDistributionStrategy::REPLICATED,adapter.shared_predictor,15);
 EXPECT_EQ(retained.predictor_materialization,SharedPredictorMaterialization::Active);
 EXPECT_THROW(parityModelContextConfig(nullptr,WeightDistributionStrategy::REPLICATED,adapter.shared_predictor,-1),std::invalid_argument);
}
TEST(ParityPredictorSource, ActualRuntimeProjectionPreservesExplicitSourceInMTPBothPolicies){
 ModelParityCase cell;cell.model.test_id="source";cell.model.model_path="/models/main.gguf";
 cell.model.shared_predictor.emplace("/models/predictor.gguf");
 cell.topology.test_id="CPU";cell.topology.kind=ModelParityTopologyKind::SingleDevice;
 cell.topology.participants={{{GlobalDeviceAddress::cpu(0)},0}};
 cell.retained_mtp_draft_capacity=15;cell.model.maximum_mtp_draft_depth=15;
 for(const auto mode:{ModelParityMTP::Off,ModelParityMTP::Depth1}) {
  cell.mtp=mode;const auto config=cell.makeOrchestrationConfig("/staged/main.gguf",0);
  EXPECT_EQ(config.model_path,"/staged/main.gguf");
  EXPECT_EQ(config.shared_predictor,cell.model.shared_predictor);
  EXPECT_EQ(config.mtp.enabled,mode!=ModelParityMTP::Off);
  EXPECT_EQ(config.mtp.graph_capacity_draft_tokens,15);
 }
 cell.model.shared_predictor.reset();
 EXPECT_FALSE(cell.makeOrchestrationConfig("/models/main.gguf",0).shared_predictor);
}
TEST(ParityPredictorSource, PublicArgumentsAndDiscoveryCarryTheSameExplicitSource){
 ModelParityCase cell;cell.model.test_id="source";cell.model.model_path="/models/main.gguf";
 cell.model.shared_predictor.emplace("/models/predictor.gguf");
 cell.model.prefix_state=ModelParityPrefixState::AttentionKV;
 cell.topology.test_id="CPU";cell.topology.kind=ModelParityTopologyKind::SingleDevice;
 cell.topology.participants={{{GlobalDeviceAddress::cpu(0)},0}};
 cell.e2e_certification=ModelParityE2EProfile{};
 for(const auto&args:{modelParityServerArguments(cell),modelParityBenchmarkArguments(cell)}) {
  const auto flag=std::find(args.begin(),args.end(),"--mtp-model");ASSERT_NE(flag,args.end());
  ASSERT_NE(std::next(flag),args.end());EXPECT_EQ(*std::next(flag),cell.model.shared_predictor->path());
 }
 std::ostringstream selected;PrintTo(cell,&selected);
 EXPECT_NE(selected.str().find("\"shared_predictor\":\"/models/predictor.gguf\""),std::string::npos);
 cell.model.shared_predictor.reset();
 const auto args=modelParityServerArguments(cell);
 EXPECT_EQ(std::find(args.begin(),args.end(),"--mtp-model"),args.end());
 std::ostringstream absent;PrintTo(cell,&absent);
 EXPECT_EQ(absent.str().find("shared_predictor"),std::string::npos);
}
TEST(ParityPredictorSource, ForeignRunnerSourceRejectedBeforeMaterialization){
 const std::optional<SharedPredictorGGUFSource> selected{SharedPredictorGGUFSource("/models/a.gguf")};
 const std::optional<SharedPredictorGGUFSource> foreign{SharedPredictorGGUFSource("/models/b.gguf")};
 EXPECT_EQ(bindParityPredictorSource(selected,std::nullopt),selected);
 EXPECT_EQ(bindParityPredictorSource(selected,selected),selected);
 EXPECT_THROW(bindParityPredictorSource(selected,foreign),std::invalid_argument);
 EXPECT_THROW(bindParityPredictorSource(std::nullopt,foreign),std::invalid_argument);
 EXPECT_NE(parityPredictorIdentity(selected),parityPredictorIdentity(foreign));
}
TEST(ParityPredictorSource, CampaignRequiresDeclaredPredictorAndRemapsBothSources){
 Files files;const auto main=files.file("main.gguf"),predictor=files.file("predictor.gguf");
 const auto staged=files.root/"staged";std::filesystem::create_directory(staged);
 std::ofstream(staged/"main.gguf")<<"main";std::ofstream(staged/"predictor.gguf")<<"predictor";
 Environment campaign("LLAMINAR_PRODUCTION_PARITY_PROCESS_CAMPAIGN","1");
 Environment ramdisk("LLAMINAR_PRODUCTION_PARITY_MODEL_RAMDISK",staged.string());
 Environment manifest("LLAMINAR_PRODUCTION_PARITY_DECLARED_MODELS",main);
 const std::optional<SharedPredictorGGUFSource> selected{SharedPredictorGGUFSource(predictor)};
 EXPECT_THROW(resolveParityPredictorSource(selected),std::runtime_error);
 setenv("LLAMINAR_PRODUCTION_PARITY_DECLARED_MODELS",(main+"|"+predictor).c_str(),1);
 EXPECT_EQ(productionParityResolvedModelPath(main),(staged/"main.gguf").string());
 const auto resolved=resolveParityPredictorSource(selected);ASSERT_TRUE(resolved);
 EXPECT_EQ(resolved->path(),(staged/"predictor.gguf").string());
 std::filesystem::remove(staged/"predictor.gguf");
 EXPECT_THROW(resolveParityPredictorSource(selected),std::runtime_error);
}
}
