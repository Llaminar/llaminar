/**
 * @file Test__ParityModelContextStorageIdentity.cpp
 * @brief Model-free regressions for immutable reuse and exact snapshot-header setup.
 *
 * Probes call the actual campaign cache key and setup prompt/header operations.
 * No model is loaded, runner created, or device initialized. Tiny authored
 * metadata distinguishes header capacity from a numerical reference oracle;
 * original declared input tokens retain their own authority.
 */
#include "integration/parity/qwen2/Qwen2ParityTestBase.h"
#include "integration/parity/qwen4exp/Qwen4ExpProductionParityCases.h"
#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include <unistd.h>

namespace llaminar2::test::parity
{
namespace
{
/** @brief Instantiate the production CRTP identity contract without model setup. */
class StorageIdentityProbe final : public qwen2::ConfigDrivenParityTest<StorageIdentityProbe>
{
public:
    /** @brief Bind one valid CPU case while retaining no physical model authority. */
    StorageIdentityProbe()
    {
        case_ = expandModelParityDefinition(qwen4exp::qwen4ExpProductionDefinition(
            qwen4exp::Qwen4ExpParityTopology::CPU)).front();
        config_.model_path = case_.model.model_path;
        view_ = case_.toTestConfig();
    }
    /** @return Compatibility view consumed by the actual canonical cache key. */
    const TestConfig &getTestConfig() const { return view_; }
    /** @return Typed immutable policy owned by this synthetic case. */
    const ModelParityCase &modelParityCase() const { return case_; }
    /** @brief Select the next independently admitted immutable row policy. */
    void policy(ImmutableRowTablePolicy value) { value.validate(); case_.ngram_table_policy = value; }
    /** @return The production model-context cache key, without loading weights. */
    std::string key() const { return productionParityModelContextKey(WeightDistributionStrategy::REPLICATED); }
    /** @brief Exercise the actual derived setup prompt operation without loading a model. */
    void bindPrompt(bool native, const std::filesystem::path &directory, std::vector<int> tokens)
    {
        native_ = native;
        config_.snapshot_dir = directory.string();
        config_.token_ids = std::move(tokens);
        production_parity_campaign_active_ = true;
        bindParityPrefillPrompt();
    }
    /** @brief Authenticate actual exact-setup header metadata without a runner or model load. */
    void prepareGeometry(const std::filesystem::path &directory,
                         const std::filesystem::path &model, std::vector<int> tokens)
    {
        native_ = true;
        config_.model_path = model.string();
        config_.prompt = "declared native prompt";
        config_.snapshot_dir = directory.string();
        config_.token_ids = std::move(tokens);
        config_.decode_steps = 0;
        production_parity_campaign_active_ = true;
        prepareNativeExactSnapshotGeometryMetadata();
    }
    /** @return Prompt retained by the actual setup owner. */
    const std::vector<int> &prompt() const { return config_.token_ids; }
private:
    /** @return The typed proof purpose selected for this setup probe. */
    MTPParityProofPurpose mtpParityProofPurpose() const noexcept override
    { return native_ ? MTPParityProofPurpose::NativeExactReplay : MTPParityProofPurpose::ReferenceNumerics; }
    bool native_ = false;
    /** @brief This probe is never registered as a running fixture. */
    void TestBody() override {}
    ModelParityCase case_;
    TestConfig view_;
};
} // namespace

/** @test RAM/mmap and every bounded-window scalar own distinct immutable cache identities. */
TEST(ParityModelContextStorageIdentity, StorageAndWindowGeometryCannotReuseThePreviousTable)
{
    StorageIdentityProbe probe;
    const ImmutableRowTablePolicy ram{.storage = ImmutableRowStoragePreference::RAM};
    probe.policy(ram);
    const auto ram_key = probe.key();
    auto mapped = ram;
    mapped.storage = ImmutableRowStoragePreference::MemoryMapped;
    probe.policy(mapped);
    const auto mapped_key = probe.key();
    EXPECT_NE(ram_key, mapped_key);
    mapped.window_bytes *= 2;
    probe.policy(mapped);
    const auto wider_key = probe.key();
    EXPECT_NE(mapped_key, wider_key);
    ++mapped.window_count;
    probe.policy(mapped);
    EXPECT_NE(wider_key, probe.key());
    probe.policy(ram);
    EXPECT_EQ(ram_key, probe.key());
}
/** @test Exact setup preserves declared tokens with no pack; numerics still imports/requires one. */
TEST(ParityModelContextStorageIdentity, NativeExactPromptSetupDoesNotReadReferenceMetadata)
{
    const auto directory = std::filesystem::temp_directory_path() /
        ("llaminar-prompt-setup-" + std::to_string(::getpid()));
    std::filesystem::remove_all(directory);
    {
        StorageIdentityProbe exact;
        exact.bindPrompt(true, directory, {7, 101, 900});
        EXPECT_EQ(exact.prompt(), (std::vector<int>{7, 101, 900}));
        ASSERT_NE(std::getenv("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES"), nullptr);
        EXPECT_STREQ(std::getenv("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES"), "3");
    }
    std::filesystem::create_directories(directory);
    { std::ofstream metadata(directory / "metadata.txt"); metadata << "token_ids: 8, 19\n"; }
    {
        StorageIdentityProbe numerical;
        numerical.bindPrompt(false, directory, {7, 101, 900});
        EXPECT_EQ(numerical.prompt(), (std::vector<int>{8, 19}));
        EXPECT_STREQ(std::getenv("LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES"), "2");
    }
    std::filesystem::remove_all(directory);
    ::testing::TestPartResultArray failures;
    {
        ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &failures);
        StorageIdentityProbe numerical;
        numerical.bindPrompt(false, directory, {7, 101, 900});
    }
    ASSERT_EQ(failures.size(), 1);
    EXPECT_TRUE(failures.GetTestPartResult(0).fatally_failed());
    EXPECT_NE(std::string(failures.GetTestPartResult(0).message()).find("reference metadata has no token_ids"), std::string::npos);
}
/** @test Header geometry is an explicit dependency, independent of numerical payloads and input import. */
TEST(ParityModelContextStorageIdentity, NativeExactAuthenticatesHeaderGeometryWithoutNumericalOracle)
{
    const auto directory = std::filesystem::temp_directory_path() /
        ("llaminar-native-header-geometry-" + std::to_string(::getpid()));
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto model = directory / "original.gguf";
    { std::ofstream file(model); file << "recorded metadata source"; }
    const auto writeMetadata = [&](std::string tokens = "7,101,900") {
        std::ofstream metadata(directory / "metadata.txt");
        metadata << "snapshot_version: 4\nreference_identity_version: 1\n"
                 << "reference_engine: pytorch\nreference_device: cpu\nreference_dtype: float32\n"
                 << "model_filename: original.gguf\nmodel_size_bytes: " << std::filesystem::file_size(model)
                 << "\nprompt_sha256: " << *sha256BytesHex("declared native prompt")
                 << "\ntoken_ids: " << tokens << "\ntoken_ids_sha256: " << *sha256BytesHex(tokens)
                 << "\ndecode_steps: 0\n";
    };
    writeMetadata();
    // These files contain only valid headers: numerical loading/scoring would
    // fail. The existing capacity owner intentionally consumes shapes alone.
    for (const auto *key : {"EMBEDDING", "LM_HEAD"})
    {
        const auto path = directory / (std::string(key) + ".npy");
        const std::array<float, 6> unused{};
        cnpy::npy_save<float>(path.string(), unused.data(), {3, 2}, "w");
        std::ifstream file(path, std::ios::binary);
        std::array<unsigned char, 10> prefix{};
        file.read(reinterpret_cast<char *>(prefix.data()), prefix.size());
        ASSERT_EQ(prefix[6], 1u);
        const auto header_end = 10u + prefix[8] + (static_cast<unsigned>(prefix[9]) << 8);
        file.close();
        std::filesystem::resize_file(path, header_end);
    }
    StorageIdentityProbe exact;
    ASSERT_NO_FATAL_FAILURE(exact.prepareGeometry(directory, model, {7, 101, 900}));
    EXPECT_EQ(exact.prompt(), (std::vector<int>{7, 101, 900}));
    const auto capacity = deriveParitySnapshotMemoryCapacity(directory, 3, 16, {});
    ASSERT_TRUE(capacity.valid());
    EXPECT_EQ(capacity.reference_file_count, 2u);
    EXPECT_GT(capacity.capacity.per_accelerator_bytes, 0u);
    // A valid pack for different IDs must never replace the native input IDs.
    writeMetadata("8,19,21");
    ::testing::TestPartResultArray failures;
    {
        ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &failures);
        exact.prepareGeometry(directory, model, {7, 101, 900});
    }
    ASSERT_EQ(failures.size(), 1);
    EXPECT_TRUE(failures.GetTestPartResult(0).fatally_failed());
    EXPECT_EQ(exact.prompt(), (std::vector<int>{7, 101, 900}));
    std::filesystem::remove(directory / "metadata.txt");
    ::testing::TestPartResultArray missing;
    {
        ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &missing);
        exact.prepareGeometry(directory, model, {7, 101, 900});
    }
    ASSERT_EQ(missing.size(), 1);
    EXPECT_TRUE(missing.GetTestPartResult(0).fatally_failed());
    EXPECT_NE(std::string(missing.GetTestPartResult(0).message()).find("snapshot-header geometry"), std::string::npos);
    EXPECT_EQ(exact.prompt(), (std::vector<int>{7, 101, 900}));
    std::filesystem::remove_all(directory);
}
} // namespace llaminar2::test::parity
