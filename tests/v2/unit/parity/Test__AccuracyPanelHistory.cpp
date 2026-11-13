/**
 * @file Test__AccuracyPanelHistory.cpp
 * @brief Device-free frozen-history admission and reference binding regressions.
 *
 * Tiny files are metadata fixtures. They prove that the actual shared request
 * adapter rejects foreign cells, partial teachers, changed reference bytes and
 * conflicting replay ownership before any model, device or runner is admitted.
 */
#include "integration/parity/AccuracyPanelHistory.h"
#include "integration/parity/FinalLogitMeanKLContract.h"
#include <gtest/gtest.h>
#include <unistd.h>

namespace llaminar2::test::parity
{
namespace
{
/** @brief Own private metadata and restore every temporary environment value. */
struct PanelMetadata
{
    std::filesystem::path root;
    std::vector<std::pair<std::string, std::optional<std::string>>> saved;
    /** @brief Create only bounded host metadata, never numerical payloads. */
    PanelMetadata()
    {
        char pattern[] = "/tmp/llaminar-panel-unit-XXXXXX";
        const auto *path = ::mkdtemp(pattern);
        if (!path) throw std::runtime_error("Cannot create test metadata");
        root = path;
        std::ofstream(root / "metadata.txt") << "token_ids: 1,2\ndecode_tokens: 3,4,5\n";
        set("LLAMINAR_ACCURACY_PANEL_HISTORY", std::nullopt);
        set("LLAMINAR_ACCURACY_PANEL_SHA256", std::nullopt);
        set("LLAMINAR_PARITY_HISTORY_REPLAY_FILE", std::nullopt);
    }
    /** @brief Restore prior environment and retire this fixture's files. */
    ~PanelMetadata()
    {
        for (auto i = saved.rbegin(); i != saved.rend(); ++i)
            if (i->second) ::setenv(i->first.c_str(), i->second->c_str(), 1);
            else ::unsetenv(i->first.c_str());
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }
    /** @brief Retain the previous value before one scoped environment change. */
    void set(const std::string &key, std::optional<std::string> value)
    {
        const char *old = std::getenv(key.c_str());
        saved.emplace_back(key, old ? std::optional<std::string>(old) : std::nullopt);
        if (value) ::setenv(key.c_str(), value->c_str(), 1);
        else ::unsetenv(key.c_str());
    }
    /** @return One complete metadata-only history under the actual wire contract. */
    nlohmann::json value(const std::string &model = "/models/model.gguf") const
    {
        return {{"schema", "llaminar.accuracy-panel-history.v1"}, {"canonical_cell", "Suite.Cell/0"},
                {"model", model}, {"reference_scope", "full_stages"}, {"reference_arithmetic", "cpu_fp32"},
                {"reference_execution", "incremental_cached"}, {"reference_directory", root.string()},
                {"metadata_sha256", *sha256BytesHex("token_ids: 1,2\ndecode_tokens: 3,4,5\n")},
                {"vocabulary", 6}, {"decode_rows", 2}, {"prompt", "original prompt"},
                {"prompt_ids", {1, 2}}, {"continuation_ids", {3, 4, 5}}};
    }
    /** @brief Publish exact bytes and supply both required handoff fields. */
    void install(const nlohmann::json &value)
    {
        const auto bytes = value.dump();
        std::ofstream(root / "history.json") << bytes;
        set("LLAMINAR_ACCURACY_PANEL_HISTORY", (root / "history.json").string());
        set("LLAMINAR_ACCURACY_PANEL_SHA256", *sha256BytesHex(bytes));
    }
};

/** @brief Minimal request data surface; no topology, stream or model authority. */
struct PanelRequest
{
    std::string prompt, snapshot_dir;
    std::vector<int> token_ids;
    int decode_steps = 0;
    float kl_threshold = .03f;
    std::optional<FinalLogitMeanKLContract> final_logit_mean_kl = FinalLogitMeanKLContract{};
    /** @return Existing model-owned prefill limit, independent of request inputs. */
    float prefillKLThreshold() const { return .03f; }
};

TEST(AccuracyPanelHistory, FrozenInputsRetainOriginalNativeLimitsAndBinding)
{
    PanelMetadata files;
    files.install(files.value());
    auto history = AccuracyPanelHistory::fromDriver("Suite.Cell/0", "/models/model.gguf");
    ASSERT_TRUE(history);
    PanelRequest request;
    history->apply(request);
    EXPECT_EQ(request.token_ids, (std::vector<int>{1, 2}));
    EXPECT_EQ(request.decode_steps, 2);
    EXPECT_FLOAT_EQ(request.final_logit_mean_kl->prefill_maximum, .05f);
    EXPECT_NO_THROW(history->requireTokens({1, 2}, {3, 4, 5}));
    EXPECT_THROW(history->requireTokens({1, 2}, {3, 5, 4}), std::invalid_argument);
    history->publish(files.root, request);
    std::ifstream input(files.root / "accuracy_panel_binding.json");
    const auto binding = nlohmann::json::parse(input);
    EXPECT_EQ(binding.at("prompt_ids"), files.value().at("prompt_ids"));
    EXPECT_FLOAT_EQ(binding.at("native_limits").at("decode_maximum").get<float>(), .1f);
}

TEST(AccuracyPanelHistory, EveryModelUsesTheSameHistoryAuthority)
{
    PanelMetadata files;
    for (const char *model : {"Qwen3.6-MoE.gguf", "Qwen3.8-Dense.gguf", "QFN-first.gguf"})
    {
        files.install(files.value(std::string("/models/") + model));
        EXPECT_TRUE(AccuracyPanelHistory::fromDriver("Suite.Cell/0", std::string("/models/") + model));
        EXPECT_THROW(AccuracyPanelHistory::fromDriver("Other.Cell/0", std::string("/models/") + model), std::invalid_argument);
    }
}

TEST(AccuracyPanelHistory, NativeStateOraclesCannotBecomeTeacherRowsOrInflatePromptCoverage)
{
    PanelMetadata files;
    files.install(files.value());
    const auto history = AccuracyPanelHistory::fromDriver("Suite.Cell/0", "/models/model.gguf");
    ASSERT_TRUE(history);
    EXPECT_TRUE(history->observes(-1, AccuracyObservationPurpose::ReferenceHistory));
    EXPECT_TRUE(history->observes(0, AccuracyObservationPurpose::ReferenceHistory));
    EXPECT_TRUE(history->observes(1, AccuracyObservationPurpose::ReferenceHistory));
    EXPECT_THROW((void)history->observes(2, AccuracyObservationPurpose::ReferenceHistory), std::invalid_argument);
    EXPECT_THROW((void)history->observes(-2, AccuracyObservationPurpose::ReferenceHistory), std::invalid_argument);
    EXPECT_FALSE(history->observes(-2, AccuracyObservationPurpose::NativeStateOracle));
    EXPECT_FALSE(history->observes(0, AccuracyObservationPurpose::NativeStateOracle));
}

TEST(AccuracyPanelHistory, RejectsForeignPolicyMalformedTokensAndIncompleteTeachers)
{
    PanelMetadata files;
    for (int fault = 0; fault < 8; ++fault)
    {
        auto value = files.value();
        switch (fault)
        {
        case 0: value["reference_scope"] = "final_logits_only"; break;
        case 1: value["reference_execution"] = "eager"; break;
        case 2: value["continuation_ids"] = {3, 4}; break;
        case 3: value["prompt_ids"] = {1, false}; break;
        case 4: value["prompt_ids"] = {1, 6}; break;
        case 5: value["vocabulary"] = 4294967296LL; break;
        case 6: value["decode_rows"] = 0; break;
        case 7: value["model"] = "/models/different.gguf"; break;
        }
        files.install(value);
        EXPECT_THROW(AccuracyPanelHistory::fromDriver("Suite.Cell/0", "/models/model.gguf"), std::exception);
    }
}

TEST(AccuracyPanelHistory, ChangedMetadataAndPartialOrMixedHandoffsFailBeforeAdmission)
{
    PanelMetadata files;
    EXPECT_FALSE(AccuracyPanelHistory::fromDriver("Suite.Cell/0", "/models/model.gguf"));
    files.set("LLAMINAR_ACCURACY_PANEL_HISTORY", (files.root / "history.json").string());
    EXPECT_THROW(AccuracyPanelHistory::fromDriver("Suite.Cell/0", "/models/model.gguf"), std::invalid_argument);
    files.install(files.value());
    auto history = AccuracyPanelHistory::fromDriver("Suite.Cell/0", "/models/model.gguf");
    files.set("LLAMINAR_PARITY_HISTORY_REPLAY_FILE", "another-history.json");
    PanelRequest request;
    EXPECT_THROW(history->apply(request), std::invalid_argument);
    std::ofstream(files.root / "metadata.txt") << "changed";
    EXPECT_THROW(AccuracyPanelHistory::fromDriver("Suite.Cell/0", "/models/model.gguf"), std::invalid_argument);
}
} // namespace
} // namespace llaminar2::test::parity
