/**
 * @file Test__SharedPredictorConfig.cpp
 * @brief Device-free explicit predictor CLI/YAML/document and no-MTP selection tests.
 *
 * Paths remain declarative strings: parser tests neither discover artifacts nor
 * touch model payloads. Exact source compatibility belongs to loader admission.
 */
#include "config/OrchestrationConfigParser.h"
#include "config/OrchestrationConfigDocument.h"
#include "config/ConfigValidator.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace llaminar2::test
{
/** @test Explicit native sidecar selection is independent of the no-MTP execution request. */
TEST(SharedPredictorConfig, CLISelectsExactPathWithNoMTPWithoutChangingMainPath)
{
    OrchestrationConfigParser parser;
    std::vector<std::string> values{"llaminar2","--model","/models/main.gguf","--mtp-model","/models/dense draft.gguf","--no-mtp"};
    std::vector<char*> argv; for(auto &v:values)argv.push_back(v.data());
    const auto c=parser.parseArgs(argv.size(),argv.data());
    ASSERT_TRUE(c.shared_predictor); EXPECT_EQ(c.shared_predictor->path(),values[4]);
    EXPECT_EQ(c.model_path,values[2]); EXPECT_FALSE(c.mtp.enabled); EXPECT_EQ(c.mtp_activation_policy,MTPActivationPolicy::Disabled);
    EXPECT_EQ(ConfigValidator::createStandard().findRule("shared-predictor-target-required")->check(c),std::nullopt);
    EXPECT_NE(OrchestrationConfigParser::getHelpText().find("--mtp-model"),std::string::npos);
}
/** @test Authored YAML and strict lossless document preserve exact source path and table storage independently. */
TEST(SharedPredictorConfig, YAMLLosslessDocumentRoundTripPreservesSourceAndNgramPolicy)
{
    OrchestrationConfigParser parser;
    const auto c=parser.parseYamlString("model: /models/main.gguf\nmtp:\n  model: /models/dense draft.gguf\n  enabled: false\nngram:\n  storage: mmap\n  mmap_window_bytes: 1048576\n  mmap_window_count: 2\n");
    ASSERT_TRUE(c.shared_predictor); EXPECT_EQ(c.shared_predictor->path(),"/models/dense draft.gguf"); EXPECT_FALSE(c.mtp.enabled);
    const auto wire=serializeOrchestrationConfig(c); const auto copy=deserializeOrchestrationConfig(wire);
    EXPECT_EQ(copy.shared_predictor,c.shared_predictor); EXPECT_EQ(copy.model_path,c.model_path);
    EXPECT_EQ(copy.ngram_table_policy,c.ngram_table_policy); EXPECT_EQ(copy.mtp_activation_policy,c.mtp_activation_policy);
    EXPECT_EQ(serializeOrchestrationConfig(copy),wire);
}
/** @test No implicit predictor path is invented for ordinary main-only input. */
TEST(SharedPredictorConfig, MainOnlyRemainsValidAndEmptyOrUnpairedSelectionFails)
{
    OrchestrationConfigParser parser; const auto c=parser.parseYamlString("model: /models/main.gguf\nmtp:\n  enabled: false\n");
    EXPECT_FALSE(c.shared_predictor); EXPECT_FALSE(deserializeOrchestrationConfig(serializeOrchestrationConfig(c)).shared_predictor);
    EXPECT_THROW(SharedPredictorGGUFSource(""),std::invalid_argument);
    auto unpaired=c; unpaired.model_path.clear(); unpaired.shared_predictor.emplace("/models/draft.gguf");
    const auto validator=ConfigValidator::createStandard();
    const auto rule=validator.findRule("shared-predictor-target-required");
    ASSERT_NE(rule,nullptr); ASSERT_TRUE(rule->check(unpaired));
}
/** @test Changed source field is explicit in the wire ABI; stale/missing/wrong-typed documents fail. */
TEST(SharedPredictorConfig, RejectsStaleSchemaMissingFieldAndInvalidSourceType)
{
    OrchestrationConfig c; c.model_path="/main.gguf"; c.shared_predictor.emplace("/draft.gguf");
    auto j=nlohmann::json::parse(serializeOrchestrationConfig(c));
    auto stale=j; stale["schema_version"]=6; EXPECT_THROW((void)deserializeOrchestrationConfig(stale.dump()),std::invalid_argument);
    auto missing=j; missing["configuration"].erase("shared_predictor"); EXPECT_THROW((void)deserializeOrchestrationConfig(missing.dump()),std::invalid_argument);
    auto number=j; number["configuration"]["shared_predictor"]=7; EXPECT_THROW((void)deserializeOrchestrationConfig(number.dump()),std::invalid_argument);
    auto empty=j; empty["configuration"]["shared_predictor"]=""; EXPECT_THROW((void)deserializeOrchestrationConfig(empty.dump()),std::invalid_argument);
}
} // namespace llaminar2::test
