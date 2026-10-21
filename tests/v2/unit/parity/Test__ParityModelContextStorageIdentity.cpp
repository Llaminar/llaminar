/**
 * @file Test__ParityModelContextStorageIdentity.cpp
 * @brief Model-free regression for immutable RAM/mmap authority reuse identity.
 *
 * This probe calls the actual canonical campaign cache-key implementation.
 * It never runs fixture setup, creates a runner, opens a GGUF, or initializes
 * a device. A changed native policy must evict the former table authority.
 */
#include "integration/parity/qwen2/Qwen2ParityTestBase.h"
#include "integration/parity/qwen4exp/Qwen4ExpProductionParityCases.h"
#include <gtest/gtest.h>

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
private:
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
} // namespace llaminar2::test::parity
