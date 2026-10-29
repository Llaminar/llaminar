/**
 * @file Test__MTPZeroHiddenBootstrap.cpp
 * @brief Device-free actual QFN CPU bootstrap producer and terminal-hidden read-lease regressions.
 *
 * Original arena FP32 storage and bounded synthetic native projection weights
 * are allocated. The real producer performs
 * zeroing/coherence/publication, and the real sidecar lease type consumes that
 * publication. No model files, accelerator work, copied mailbox or fake sidecar
 * execution is used; broader predictor numerical qualification remains separate.
 * The predictor-hidden role also enters the real branch norm and strict stage
 * verifier, preserving finite/shape checks while admitting the valid zero seed.
 * Its compact M1 rows also reach the real prepared HC down GEMM stage and
 * serial-byte oracle through an exactly admitted CPU workspace.
 * CPU arena roots use an explicit model-free fixture admission authority.
 */
#include "../../../../utils/CPUActivationArenaFixture.h"
#include "backends/BackendManager.h"
#include "kernels/cpu/CPURingKVCache.h"
#include "execution/mtp/MTPStateTransaction.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "memory/BufferArena.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPrimitiveStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/StageVerifier.h"
#include "tensors/TensorVerification.h"
#include "utils/DebugEnv.h"
#include "utils/OpenMPUtils.h"
#include "models/qwen/QwenStandardGraph.h"
#include "models/qwen4exp/Qwen4ExpGraph.h"
#include "models/qwen4exp/Qwen4ExpGraphConfigBuilder.h"
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPredictorRows.h"
#include "../../../../utils/PreparedWeightTestHarness.h"
#include "../../../../utils/CPUProjectionTestWorkspace.h"
#include "transfer/TransferEngine.h"
#include <cstring>
#include "utils/MPIContext.h"
#include "../../../models/qwen4exp/Qwen4ExpMetadataFixture.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <memory>
#include <vector>

namespace llaminar2
{
/** @brief Closed test peer of the actual original mailbox producer; never a production permit. */
struct DeviceGraphOrchestratorMTPZeroHiddenTestAccess
{
    /** @brief Install one real arena-owned host mailbox, with optionally foreign state ownership. */
    static bool install(DeviceGraphOrchestrator &runner, std::size_t rows, std::size_t width, bool foreign = false, const char *dtype = "FP32",
                        TensorFactory *factory = nullptr)
    {
        runner.state_.device_id = DeviceId::cpu();
        runner.arena_ = std::make_unique<BufferArena>(::llaminar2::test::cpuArenaFixtureConfig(factory));
        if (!runner.arena_->registerBuffer(BufferId::PREFIX_TERMINAL_HIDDEN, rows, width, dtype, DeviceId::cpu()) ||
            !runner.arena_->allocate())
            return false;
        runner.state_.prefix_terminal_hidden = foreign
            ? std::make_shared<FP32Tensor>(std::vector<std::size_t>{rows, width})
            : runner.arena_->getSharedTensor(BufferId::PREFIX_TERMINAL_HIDDEN);
        return runner.state_.prefix_terminal_hidden != nullptr;
    }
    /** @brief Exercise the real CPU cache frontier validator without executing a synthetic model. */
    static bool validateCommitBoundary(DeviceGraphOrchestrator &runner, IKVCache &cache,
                                       int position, bool allow_discard)
    {
        runner.state_.device_id = DeviceId::cpu();
        return runner.prepareShiftedMTPKVCommitBoundary(cache, 0, 0,
            mtpShiftedAppendBoundaryRows(position, 0, cache.stateOwnership().shifted_origin),
            allow_discard, nullptr, nullptr, "direct_emit_checkpoint_origin_regression");
    }
    /** @return Original actual state mailbox, never copied row storage. */
    static std::shared_ptr<TensorBase> mailbox(DeviceGraphOrchestrator &runner)
    { return runner.state_.prefix_terminal_hidden; }
    /** @return Actual producer result, including original arena/source checks and host publication. */
    static bool publish(DeviceGraphOrchestrator &runner)
    { return runner.publishCPUVisibleZeroHiddenMTPBootstrap(); }
    /** @return Actual publication authority used by executeMTPDepth0's acquireReadLease. */
    static MTPTerminalHiddenPublication &publication(DeviceGraphOrchestrator &runner)
    { return runner.state_.mtp_terminal_hidden_publication; }
};
}

namespace
{
using namespace llaminar2;
using Access = DeviceGraphOrchestratorMTPZeroHiddenTestAccess;
/** @return Published geometry only; no source file or native model payload is loaded. */
GraphConfig bootstrapConfig()
{
    GraphConfig config;
    Qwen4ExpGraphConfigBuilder::populateFromDirectory(test::qwen4MetadataFixture(false), config);
    config.default_device = DeviceId::cpu();
    config.mtp.enabled = true;
    return config;
}
/** @brief Initialize the ordinary CPU owner once for this device-free test process. */
class MTPZeroHiddenBootstrap : public ::testing::Test
{
protected:
    /** @brief Initialize the aggregate host execution authority without accelerator or model startup. */
    void SetUp() override
    {
        initCPUBackend(-1);
        DeviceManager::instance().initialize(-1);
    }
};

/** @test Direct emission validates actual CPU storage at both model origins and rejects the old QFN +1. */
TEST_F(MTPZeroHiddenBootstrap, DirectEmitCheckpointOriginUsesActualCPUCacheFrontier)
{
    MPIContext mpi(0, 1);
    auto builder = std::make_shared<Qwen4ExpGraph>(bootstrapConfig());
    DeviceGraphOrchestrator runner(builder, nullptr);
    for (const auto origin : {MTPShiftedCacheOrigin::SuccessorPairsOnly,
                              MTPShiftedCacheOrigin::VisibleZeroHiddenTokenZero})
        for (const int rows : {1, 11, 31})
        {
            SCOPED_TRACE(rows);
            auto cache = createCPURingKVCache(ActivationPrecision::FP32, mpi, 1, 1, 64, 1, 128);
            ASSERT_NE(cache, nullptr);
            cache->bindStateOwnership({.role=IKVCache::StateRole::MTPShiftedSidecar,
                                       .mtp_depth=0, .shifted_origin=origin});
            FP32Tensor keys({static_cast<std::size_t>(rows), 128}),
                       values({static_cast<std::size_t>(rows), 128});
            std::fill_n(static_cast<float *>(keys.raw_mutable_data()), keys.numel(), 0.25f);
            std::fill_n(static_cast<float *>(values.raw_mutable_data()), values.numel(), -0.5f);
            ASSERT_TRUE(cache->append(0, 0, &keys, &values, rows));
            ASSERT_EQ(cache->get_cached_tokens(0, 0), rows);
            const int position = mtpShiftedAppendPosition(rows, origin);
            EXPECT_TRUE(Access::validateCommitBoundary(runner, *cache, position, true));
            EXPECT_EQ(cache->get_cached_tokens(0, 0), rows);
            // The real validator must continue to reject absent state even
            // when speculative suffix discard is allowed. No count repair.
            EXPECT_FALSE(Access::validateCommitBoundary(runner, *cache, position + 1, true));
            EXPECT_EQ(cache->get_cached_tokens(0, 0), rows);
            EXPECT_FALSE(Access::validateCommitBoundary(runner, *cache, position - 1, false));
            EXPECT_EQ(cache->get_cached_tokens(0, 0), rows);
            EXPECT_TRUE(Access::validateCommitBoundary(runner, *cache, position - 1, true));
            EXPECT_EQ(cache->get_cached_tokens(0, 0), rows - 1);
        }
}

/** @test The real 4H producer makes the actual sidecar read lease current, without touching spare rows. */
TEST_F(MTPZeroHiddenBootstrap, PublishesOriginalWideFirstRowAndCurrentReadLease)
{
    const auto config = bootstrapConfig();
    ASSERT_EQ(config.residualWidth(), 10240);
    auto builder = std::make_shared<Qwen4ExpGraph>(config);
    DeviceGraphOrchestrator runner(builder, nullptr);
    ASSERT_TRUE(Access::install(runner, 3, 10240));
    const auto owner = Access::mailbox(runner);
    ASSERT_EQ(owner->native_type(), TensorType::FP32);
    auto *data = static_cast<float *>(owner->raw_mutable_data());
    std::fill_n(data, owner->numel(), 7.25f);
    auto &publication = Access::publication(runner);
    EXPECT_FALSE(publication.acquireReadLease());
    ASSERT_TRUE(Access::publish(runner));
    EXPECT_EQ(Access::mailbox(runner).get(), owner.get());
    // Host publication retains HOST_ONLY when no GPU allocation exists;
    // HOST_AUTHORITATIVE specifically requires both physical copies.
    EXPECT_EQ(owner->gpu_data_ptr(), nullptr);
    EXPECT_TRUE(owner->hostValid());
    EXPECT_EQ(owner->coherenceState(), TensorCoherenceState::HOST_ONLY);
    const auto lease = publication.acquireReadLease();
    ASSERT_TRUE(lease);
    EXPECT_EQ(lease->source, MTPTerminalHiddenPublication::Source::VisibleZeroHiddenBootstrap);
    EXPECT_STREQ(MTPTerminalHiddenPublication::sourceName(lease->source), "visible_zero_hidden_bootstrap");
    EXPECT_TRUE(publication.stillOwns(*lease));
    EXPECT_TRUE(std::all_of(data, data + 10240, [](float x) { return x == 0.0f && !std::signbit(x); }));
    EXPECT_TRUE(std::all_of(data + 10240, data + owner->numel(), [](float x) { return x == 7.25f; }));
    // These are the unchanged actual consumer lease operations, not an emulated
    // state flag; executeMTPDepth0 uses them before and after sidecar submission.
    publication.invalidate();
    EXPECT_FALSE(publication.stillOwns(*lease));
    EXPECT_FALSE(publication.acquireReadLease());
    ASSERT_TRUE(Access::publish(runner));
    const auto next = publication.acquireReadLease();
    ASSERT_TRUE(next);
    EXPECT_GT(next->generation, lease->generation);
    EXPECT_FALSE(publication.stillOwns(*lease));
}

/** @test A prior restore lease is invalidated by the real zero producer before a new reader can use it. */
TEST_F(MTPZeroHiddenBootstrap, BootstrapReplacesPriorLeaseWithoutReplacingBacking)
{
    auto builder = std::make_shared<Qwen4ExpGraph>(bootstrapConfig());
    DeviceGraphOrchestrator runner(builder, nullptr);
    ASSERT_TRUE(Access::install(runner, 2, 10240));
    auto &publication = Access::publication(runner);
    publication.publishPrefixRestore();
    const auto prior = publication.acquireReadLease();
    ASSERT_TRUE(prior);
    const auto owner = Access::mailbox(runner);
    ASSERT_TRUE(Access::publish(runner));
    const auto current = publication.acquireReadLease();
    ASSERT_TRUE(current);
    EXPECT_FALSE(publication.stillOwns(*prior));
    EXPECT_TRUE(publication.stillOwns(*current));
    EXPECT_GT(current->generation, prior->generation);
    EXPECT_EQ(owner.get(), Access::mailbox(runner).get());
}

/** @test Truncated4H or foreign ownership rejects before bytes or the previous lease change. */
TEST_F(MTPZeroHiddenBootstrap, RejectsBadGeometryAndForeignOriginalOwnerBeforeMutation)
{
    for (const bool foreign : {false, true})
    {
        auto builder = std::make_shared<Qwen4ExpGraph>(bootstrapConfig());
        DeviceGraphOrchestrator runner(builder, nullptr);
        ASSERT_TRUE(Access::install(runner, 2, foreign ? 10240 : 2560, foreign));
        const auto owner = Access::mailbox(runner);
        auto *data = static_cast<float *>(owner->raw_mutable_data());
        std::fill_n(data, owner->numel(), 3.5f);
        auto &publication = Access::publication(runner);
        publication.publishPrefixRestore();
        const auto prior = publication.acquireReadLease();
        ASSERT_TRUE(prior);
        EXPECT_FALSE(Access::publish(runner));
        EXPECT_TRUE(publication.stillOwns(*prior));
        EXPECT_TRUE(std::all_of(data, data + owner->numel(), [](float x) { return x == 3.5f; }));
    }
}

/** @test An ordinary successor-pair model cannot mint a QFN bootstrap publication. */
TEST_F(MTPZeroHiddenBootstrap, RejectsForeignSourcePolicyBeforeMutation)
{
    auto config = bootstrapConfig();
    config.flash_next.reset();
    auto builder = std::make_shared<QwenStandardGraph>(config, nullptr);
    DeviceGraphOrchestrator runner(builder, nullptr);
    ASSERT_TRUE(Access::install(runner, 2, 10240));
    const auto owner = Access::mailbox(runner);
    auto *data = static_cast<float *>(owner->raw_mutable_data());
    std::fill_n(data, owner->numel(), 2.25f);
    EXPECT_THROW((void)Access::publish(runner), std::invalid_argument);
    EXPECT_FALSE(Access::publication(runner).acquireReadLease());
    EXPECT_TRUE(std::all_of(data, data + owner->numel(), [](float x) { return x == 2.25f; }));
}
/** @test A same-byte-width integer arena root cannot claim a FP32 bootstrap source. */
TEST_F(MTPZeroHiddenBootstrap, RejectsNonFP32MailboxBeforeMutation)
{
    // The canonical factory owns dtype dispatch. Keep its rank/NUMA authority
    // alive until after the actual arena/runner; the no-factory arena creates
    // FP32 regardless of the requested dtype and cannot prove this rejection.
    MPIContext mpi(0, 1);
    TensorFactory factory(mpi);
    auto builder = std::make_shared<Qwen4ExpGraph>(bootstrapConfig());
    DeviceGraphOrchestrator runner(builder, nullptr);
    ASSERT_TRUE(Access::install(runner, 2, 10240, false, "INT32", &factory));
    const auto owner = Access::mailbox(runner);
    ASSERT_EQ(owner->native_type(), TensorType::INT32);
    ASSERT_EQ(owner->transferStorageOwner(), owner.get());
    auto *data = static_cast<std::int32_t *>(owner->raw_mutable_data());
    std::fill_n(data, owner->numel(), 73);
    EXPECT_THROW((void)Access::publish(runner), std::invalid_argument);
    EXPECT_FALSE(Access::publication(runner).acquireReadLease());
    EXPECT_TRUE(std::all_of(data, data + owner->numel(), [](std::int32_t x) { return x == 73; }));
}

/** @test Supplement real producer/consumer tests with its exact initial-prefill call edge. */
TEST_F(MTPZeroHiddenBootstrap, VisibleTokenZeroSidecarUsesTheActualPublicationProducer)
{
    std::ifstream input("src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    ASSERT_TRUE(input.good());
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const auto source = buffer.str();
    const auto begin = source.find("bool DeviceGraphOrchestrator::populateCPUMTPShiftedCacheFromPrefill(");
    ASSERT_NE(begin, std::string::npos);
    const auto origin = source.find("if(origin==MTPShiftedCacheOrigin::VisibleZeroHiddenTokenZero)", begin);
    ASSERT_NE(origin, std::string::npos);
    const auto empty = source.find("if(cached!=0)", origin);
    const auto producer = source.find("if(!publishCPUVisibleZeroHiddenMTPBootstrap())return false;", origin);
    const auto consumer = source.find("if(!executeMTPDepth0(tokens[0],state_.prefix_terminal_hidden.get(),0,", origin);
    ASSERT_NE(empty, std::string::npos);
    ASSERT_NE(producer, std::string::npos);
    ASSERT_NE(consumer, std::string::npos);
    EXPECT_LT(empty, producer);
    EXPECT_LT(producer, consumer);
}


#if LLAMINAR_ASSERTIONS_ACTIVE
/** @brief Restore process validation configuration after exercising the actual strict verifier. */
class ScopedMTPNormValidation final
{
public:
    /** @brief Enable every original finite/zero check, without emitting diagnostic payload files. */
    ScopedMTPNormValidation()
    {
        for (std::size_t i = 0; i < names_.size(); ++i)
        {
            if (const char *value = std::getenv(names_[i])) saved_[i] = std::string(value);
            setenv(names_[i], i == 4 ? "0" : "1", 1);
        }
        mutableDebugEnv().reload();
    }
    /** @brief Restore copied environment values and the canonical parsed configuration. */
    ~ScopedMTPNormValidation()
    {
        for (std::size_t i = 0; i < names_.size(); ++i)
            if (saved_[i]) setenv(names_[i], saved_[i]->c_str(), 1);
            else unsetenv(names_[i]);
        mutableDebugEnv().reload();
    }
private:
    const std::array<const char *, 5> names_{"LLAMINAR_VALIDATE_BUFFERS", "LLAMINAR_VALIDATE_INPUTS",
        "LLAMINAR_FAIL_ON_ZERO", "LLAMINAR_FAIL_ON_NAN", "LLAMINAR_DUMP_ON_FAILURE"};
    std::array<std::optional<std::string>, 5> saved_{};
};

/** @test Actual published bootstrap zero survives the same typed norm and strict verifier used by the predictor. */
TEST_F(MTPZeroHiddenBootstrap, PredictorNormPreservesPublishedZeroThroughStrictVerifier)
{
    ScopedMTPNormValidation validation;
    llaminar::v2::ThreadCountGuard team(1);
    CPUDeviceContext context(DeviceId::cpu(), 1);
    DeviceGraphOrchestrator runner(std::make_shared<Qwen4ExpGraph>(bootstrapConfig()), nullptr);
    ASSERT_TRUE(Access::install(runner, 2, 10240));
    const auto owner = Access::mailbox(runner);
    auto *input = static_cast<float *>(owner->raw_mutable_data());
    std::fill_n(input, owner->numel(), 7.25f);
    ASSERT_FALSE(Access::publication(runner).acquireReadLease());
    ASSERT_TRUE(Access::publish(runner));
    const auto lease = Access::publication(runner).acquireReadLease();
    ASSERT_TRUE(lease);
    ASSERT_EQ(lease->source, MTPTerminalHiddenPublication::Source::VisibleZeroHiddenBootstrap);
    ASSERT_TRUE(Access::publication(runner).stillOwns(*lease));
    FP32Tensor gain({10240}, DeviceId::cpu()), output({1, 10240}, DeviceId::cpu());
    std::fill_n(gain.mutable_data(), gain.numel(), 1.0f);
    std::fill_n(output.mutable_data(), output.numel(), -37.0f);
    using Stage = Qwen4ExpPrimitiveStage;
    const auto operation = Stage::BranchRMSNorm::predictorHidden(2560, 1e-6f,
        {*owner, BufferId::PREFIX_TERMINAL_HIDDEN}, Qwen4ExpImmutableWeightBinding(gain),
        {output, BufferId::MTP_NORM_HIDDEN}, Stage::DiagnosticInputPublication::MTPTerminalHidden);
    ComputeNode node("arbitrary_name_no_bootstrap_allowlist", std::make_unique<Stage>(Stage::Params{
        .rows={1,{}}, .operation=operation}), DeviceId::cpu());
    ASSERT_TRUE(node.stage->allowsZeroOutput());
    EXPECT_NO_THROW(verifyStageEntry(node, -1));
    ASSERT_TRUE(node.stage->execute(&context));
    EXPECT_NO_THROW(verifyStageExit(node, -1));
    EXPECT_TRUE(std::all_of(output.data(), output.data() + output.numel(), [](float x) { return x == 0.0f; }));
    EXPECT_TRUE(std::all_of(input, input + 10240, [](float x) { return x == 0.0f; }));
    EXPECT_TRUE(std::all_of(input + 10240, input + owner->numel(), [](float x) { return x == 7.25f; }));
    EXPECT_TRUE(Access::publication(runner).stillOwns(*lease));
    Access::publication(runner).invalidate();
    EXPECT_FALSE(Access::publication(runner).stillOwns(*lease));
    EXPECT_FALSE(Access::publication(runner).acquireReadLease());

    // The diagnostic snapshot alone is not zero-output permission. Ordinary
    // branch normalization retains the existing uninitialized-output heuristic.
    auto ordinary = operation;
    ordinary.input_role = Stage::BranchNormInputRole::Ordinary;
    ComputeNode strict("mtp0_norm_hidden", std::make_unique<Stage>(Stage::Params{
        .rows={1,{}}, .operation=ordinary}), DeviceId::cpu());
    EXPECT_FALSE(strict.stage->allowsZeroOutput());
    EXPECT_NE(strict.stage->graphCaptureVariantSignature(), node.stage->graphCaptureVariantSignature());
    EXPECT_THROW(verifyStageExit(strict, -1), verification::VerificationFailure);
    ASSERT_TRUE(strict.stage->execute(&context));
    EXPECT_THROW(verifyStageExit(strict, -1), verification::VerificationFailure);
}

/** @test A predictor-hidden role permits zero, never NaN/Inf, absent storage, bad roots or incompatible geometry. */
TEST_F(MTPZeroHiddenBootstrap, PredictorNormZeroContractRetainsFiniteAndOwnerChecks)
{
    ScopedMTPNormValidation validation;
    llaminar::v2::ThreadCountGuard team(1);
    CPUDeviceContext context(DeviceId::cpu(), 1);
    DeviceGraphOrchestrator runner(std::make_shared<Qwen4ExpGraph>(bootstrapConfig()), nullptr);
    ASSERT_TRUE(Access::install(runner, 1, 10240));
    ASSERT_TRUE(Access::publish(runner));
    const auto owner = Access::mailbox(runner);
    FP32Tensor gain({10240}, DeviceId::cpu()), output({1,10240}, DeviceId::cpu());
    std::fill_n(gain.mutable_data(), gain.numel(), 1.0f);
    using Stage = Qwen4ExpPrimitiveStage;
    auto operation = Stage::BranchRMSNorm::predictorHidden(2560, 1e-6f,
        {*owner, BufferId::PREFIX_TERMINAL_HIDDEN}, Qwen4ExpImmutableWeightBinding(gain),
        {output, BufferId::MTP_NORM_HIDDEN});
    ComputeNode node("zero_preserving_norm", std::make_unique<Stage>(Stage::Params{
        .rows={1,{}},.operation=operation}), DeviceId::cpu());
    ASSERT_TRUE(node.stage->execute(&context));
    EXPECT_NO_THROW(verifyStageExit(node,-1));
    for (const float bad : {std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()})
    {
        output.mutable_data()[0] = bad;
        EXPECT_THROW(verifyStageExit(node,-1), verification::VerificationFailure);
        output.mutable_data()[0] = 0.0f;
        static_cast<float *>(owner->raw_mutable_data())[0] = bad;
        EXPECT_THROW(verifyStageEntry(node,-1), verification::VerificationFailure);
        static_cast<float *>(owner->raw_mutable_data())[0] = 0.0f;
    }
    gain.mutable_data()[0] = std::numeric_limits<float>::quiet_NaN();
    ASSERT_TRUE(node.stage->execute(&context));
    EXPECT_THROW(verifyStageExit(node,-1), verification::VerificationFailure);
    gain.mutable_data()[0] = 1.0f;
    auto invalid = operation;
    invalid.input.tensor = nullptr;
    EXPECT_THROW((void)Stage(Stage::Params{.rows={1,{}},.operation=invalid}), std::invalid_argument);
    invalid = operation;
    invalid.input.buffer = BufferId::NORMALIZED;
    EXPECT_THROW((void)Stage(Stage::Params{.rows={1,{}},.operation=invalid}), std::invalid_argument);
    invalid = operation;
    invalid.input_role = static_cast<Stage::BranchNormInputRole>(255);
    EXPECT_THROW((void)Stage(Stage::Params{.rows={1,{}},.operation=invalid}), std::invalid_argument);
    FP32Tensor wrong({1,10239}, DeviceId::cpu());
    invalid = operation;
    invalid.output.tensor = &wrong;
    EXPECT_THROW((void)Stage(Stage::Params{.rows={1,{}},.operation=invalid}), std::invalid_argument);
    invalid = operation;
    invalid.epsilon = 0.0f;
    EXPECT_THROW((void)Stage(Stage::Params{.rows={1,{}},.operation=invalid}), std::invalid_argument);
}
#endif // LLAMINAR_ASSERTIONS_ACTIVE


/** @test The actual current wide bootstrap feeds a prepared HC down stage at the original compact verifier M1. */
TEST_F(MTPZeroHiddenBootstrap, PredictorM1PreparedHCDownConsumesCurrentPublishedNorm)
{
    llaminar::v2::ThreadCountGuard team(1);
    CPUDeviceContext context(DeviceId::cpu(), 1);
    DeviceGraphOrchestrator runner(std::make_shared<Qwen4ExpGraph>(bootstrapConfig()), nullptr);
    ASSERT_TRUE(Access::install(runner, 2, 10240));
    const auto owner = Access::mailbox(runner);
    auto *input = static_cast<float *>(owner->raw_mutable_data());
    std::fill_n(input, owner->numel(), 7.25f);
    ASSERT_TRUE(Access::publish(runner));
    auto &publication = Access::publication(runner);
    const auto lease = publication.acquireReadLease();
    ASSERT_TRUE(lease);
    ASSERT_TRUE(publication.stillOwns(*lease));

    std::vector<int> count{1};
    const auto rows = Qwen4ExpPredictorRows::admit(1, 1, DeviceId::cpu(), &count, nullptr);
    ASSERT_NO_THROW(rows.requireCompactCPUQuery());
    ASSERT_EQ(rows.tokens().countOwner(), count.data());
    FP32Tensor gain({10240}), normalized({1, 10240}), down({1, 320}), serial({1, 320});
    std::fill_n(gain.mutable_data(), gain.numel(), 1.0f);
    using Stage = Qwen4ExpPrimitiveStage;
    Stage norm(Stage::Params{.rows=rows.primitives(), .operation=Stage::BranchRMSNorm::predictorHidden(
        2560, 1e-6f, {*owner, BufferId::PREFIX_TERMINAL_HIDDEN}, Qwen4ExpImmutableWeightBinding(gain),
        {normalized, BufferId::MTP_NORM_HIDDEN}, Stage::DiagnosticInputPublication::MTPTerminalHidden)});
    ASSERT_TRUE(norm.execute(&context));

    // Real original native source and the production prepared-store factory;
    // no registered fake engine or scalar projection stands in for this stage.
    std::vector<uint8_t> bytes(size_t{320} * 10240 / 32 * sizeof(Q8_0Block));
    auto *blocks = reinterpret_cast<Q8_0Block *>(bytes.data());
    for (size_t b = 0; b < bytes.size() / sizeof(Q8_0Block); ++b)
    {
        blocks[b].d = fp32_to_fp16(1.0f / 128.0f);
        for (int j = 0; j < 32; ++j)
            blocks[b].qs[j] = static_cast<int8_t>(static_cast<int>((b + j) % 17) - 8);
    }
    Q8_0Tensor weights({320, 10240}, bytes);
    auto prepared = test::makePreparedGemmFixture(&weights, DeviceId::cpu(), "blk.48.hc_attn_down.weight");
    GEMMStage::Params params;
    params.device_id = DeviceId::cpu(); params.A = &normalized; params.B = &weights; params.C = &down;
    params.m = 1; params.n = 320; params.k = 10240;
    params.force_decode_equivalent_verifier_prefill = true;
    params.verifier_row_range = rows.tokens();
    params.prepared_ref = prepared.ref; params.prepared_store = prepared.store.get();
    GEMMStage grouped(params);
    ASSERT_TRUE(grouped.validatePreparedWeights(nullptr));
    test::CPUStageTestWorkspace workspace(grouped);
    std::fill_n(down.mutable_data(), down.numel(), -37.0f);
    ASSERT_TRUE(grouped.execute(&context));
    auto *engine = prepared.store->gemmKernel(prepared.ref);
    ASSERT_NE(engine, nullptr);
    test::CPUProjectionTestWorkspace serial_workspace(1, 10240, test::cpuProjectionTestRequirements(1, {engine}));
    ASSERT_TRUE(engine->multiply_tensor(&normalized, &serial, 1, 320, 10240,
        true, 1.0f, 0.0f, nullptr, nullptr, -1, serial_workspace.get()));
    EXPECT_EQ(std::memcmp(down.data(), serial.data(), down.numel() * sizeof(float)), 0);
    EXPECT_TRUE(std::all_of(down.data(), down.data() + down.numel(), [](float x) { return x == 0.0f; }));
    EXPECT_TRUE(std::all_of(input + 10240, input + owner->numel(), [](float x) { return x == 7.25f; }));
    EXPECT_TRUE(publication.stillOwns(*lease));

    // A subsequent host publication on the same original backing exercises
    // nonzero projection bytes too. This tests the publication API, not a
    // substitute prefix-cache restore implementation.
    publication.invalidate();
    for (size_t j = 0; j < 10240; ++j)
        input[j] = static_cast<float>(static_cast<int>(j % 29) - 14) / 32.0f;
    TransferEngine::publishHostWrite(owner);
    publication.publishPrefixRestore();
    const auto restored = publication.acquireReadLease();
    ASSERT_TRUE(restored);
    EXPECT_FALSE(publication.stillOwns(*lease));
    ASSERT_TRUE(publication.stillOwns(*restored));
    ASSERT_TRUE(norm.execute(&context));
    ASSERT_TRUE(grouped.execute(&context));
    ASSERT_TRUE(engine->multiply_tensor(&normalized, &serial, 1, 320, 10240,
        true, 1.0f, 0.0f, nullptr, nullptr, -1, serial_workspace.get()));
    EXPECT_EQ(std::memcmp(down.data(), serial.data(), down.numel() * sizeof(float)), 0);
    EXPECT_TRUE(std::all_of(down.data(), down.data() + down.numel(), [](float x) { return std::isfinite(x); }));
    EXPECT_TRUE(std::any_of(down.data(), down.data() + down.numel(), [](float x) { return x != 0.0f; }));
    EXPECT_TRUE(std::all_of(input + 10240, input + owner->numel(), [](float x) { return x == 7.25f; }));

    // CPU predictor rows remain compact: host live0 is rejected at the actual
    // producer descriptor boundary, rather than treating a padded M1 as work.
    const std::vector<float> before(down.data(), down.data() + down.numel());
    count[0] = 0;
    EXPECT_THROW(rows.requireCompactCPUQuery(), std::invalid_argument);
    EXPECT_EQ(std::memcmp(down.data(), before.data(), before.size() * sizeof(float)), 0);
    publication.invalidate();
    EXPECT_FALSE(publication.stillOwns(*lease));
    EXPECT_FALSE(publication.acquireReadLease());
}

}
