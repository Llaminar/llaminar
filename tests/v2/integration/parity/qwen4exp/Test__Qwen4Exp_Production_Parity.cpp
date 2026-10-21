/**
 * @file Test__Qwen4Exp_Production_Parity.cpp
 * @brief Canonical Flash Next RAM/mmap numerical and focused admission cases.
 *
 * The typed expander owns precision, storage, placement, and movement axes.
 * This MPI-aware executable retains the ordinary production runner and shared
 * six-CSV numerical workflow; missing backend/model/reference evidence fails.
 * SnapshotAdmission shares numerical setup, including reference validation and
 * diagnostic arenas. FullSourceAdmission uses a separate gold-free lifecycle
 * with snapshots disabled. Both retain real serving preparation/capture and
 * stop before inference; neither is numerical or performance evidence.
 */
#include "Qwen4ExpParityTestBase.h"
#include "Qwen4ExpNgramAdmissionCases.h"
#include "Qwen4ExpFixedHeadParityCases.h"
#include "Qwen4ExpGPUAdaptiveParityCases.h"
#include "Qwen4ExpFullROCmTP4ParityCases.h"
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

/** @brief One canonical production fixture per exact expanded matrix cell. */
class Qwen4ExpMainParityTest
    : public Qwen4ExpConfigDrivenParityTest<Qwen4ExpMainParityTest>, public ModelParityCaseParameter {};

/** @test Every published numerical boundary and actual selected native row path. */
TEST_P(Qwen4ExpMainParityTest, ProductionParity) { runQwen4ExpProductionParityCampaign(); }

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
    beginProductionParityEvidence();
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

// One exact all-routed-experts GPU TP4 request; real admission/numerical gates
// remain mandatory. RAM PLE is independent host storage, not CPU expert offload.
INSTANTIATE_TEST_SUITE_P(Qwen4ExpFullROCmTP4, Qwen4ExpMainParityTest,
    ::testing::ValuesIn(qfnLocalPairedFamily("Qwen4ExpFullROCmTP4", qwen4ExpFullROCmTP4ProductionCases())),
    [](const ::testing::TestParamInfo<ModelParityCase> &info) { return info.param.testName(); });

/**
 * @brief Gold-free serving admission with only GoogleTest's empty setup lifecycle.
 *
 * This deliberately does not inherit the numerical fixture: its SetUp validates
 * or generates reference snapshots even when native snapshot capture is disabled.
 * The parameter adapter still owns the exact canonical topology/storage input.
 */
class Qwen4ExpNgramAdmissionTest : public ::testing::Test, public ModelParityCaseParameter {};

/**
 * @test Materialize the complete native source under the declared Automatic/VRAM policy.
 *
 * The normal runner owns weights, captured serving families and the table.
 * This setup-only proof admits ordinary serving without diagnostic snapshot
 * banks and writes no numerical or performance receipt. The CPU
 * tail owns layers2..47; ROCm0 owns layer0 and only ROCm1 owns layer1/PLE.
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
    config.deterministic = true;
    publishOrchestrationStartupPolicy(config);
    initCPUBackend(-1);
    for (const auto &participant : modelParityCase().topology.participants)
    {
        const auto type = participant.address.device_type;
        if (type == DeviceType::CPU) continue;
        auto *backend = type == DeviceType::CUDA ? getCUDABackend() : getROCmBackend();
        if (!backend || participant.address.device_ordinal >= backend->deviceCount())
            GTEST_SKIP() << "Missing declared admission backend " << participant.address.toShortString();
    }
    ASSERT_TRUE(std::filesystem::is_regular_file(config.model_path)) << config.model_path;
    ASSERT_FALSE(config.shared_predictor);
    const char *artifact_root = std::getenv("LLAMINAR_PRODUCTION_PARITY_ARTIFACT_ROOT");
    const auto base = artifact_root ? std::filesystem::path(artifact_root)
        : std::filesystem::current_path() / "parity-results" / "qwen-flash-next" / "ngram-admission";
    ASSERT_TRUE(base.is_absolute()) << "Admission artifact root must be absolute";
    const auto output = base / modelParityCase().testName();
    // This cell owns only its generated child. No gold directory is resolved,
    // read, generated or removed, including when no reference pack is installed.
    std::error_code artifact_error;
    std::filesystem::remove_all(output, artifact_error);
    ASSERT_FALSE(artifact_error) << artifact_error.message();
    ASSERT_TRUE(std::filesystem::create_directories(output, artifact_error));
    ASSERT_FALSE(artifact_error) << artifact_error.message();
    config.prefix_cache.enabled = true;
    config.prefix_cache.storage_mode = PrefixCacheStorageMode::Tiered;
    config.prefix_cache.block_size = productionParityPrefixRestoreProofBlockSize(
        modelParityCase().model.token_ids.size());
    config.prefix_cache.terminal_state = PrefixCacheTerminalStateMode::Auto;
    config.prefix_cache.disk_dir = (output / ("prefix-cache-rank-" + std::to_string(rank))).string();
    auto runner = test::TestOrchestrationHelper::create(config);
    ASSERT_NE(runner, nullptr);
    runner->disableSnapshotCapture();
    ASSERT_TRUE(runner->initialize()) << runner->lastError();
    ASSERT_TRUE(runner->isInitialized());
    EXPECT_TRUE(runner->getSnapshotKeys().empty());
    const auto prepared = runner->modelContextReuseContract();
    ASSERT_TRUE(prepared.has_value());
    const auto context = prepared->context;
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(runner->modelContextForDiagnostics(), context.get());
    EXPECT_EQ(context->path(), config.model_path);
    EXPECT_EQ(runner->executionPlan().layerCount(), 48);
    EXPECT_FALSE(runner->config().mtp.enabled);
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
    memory.flush();
    ASSERT_TRUE(memory.good());
    std::ofstream csv(output / "admission.csv");
    ASSERT_TRUE(csv.is_open());
    csv << "cell,storage,layers,initialized,numerical_inference_executed,native_table_bytes,diagnostic_snapshots\n"
        << modelParityCase().testName() << ',' << ngramStorageLabel(placement->storage())
        << ",48,1,0," << placement->extent().nativeBytes() << ",0\n";
    csv.flush();
    ASSERT_TRUE(csv.good());
    runner->shutdown();
    EXPECT_FALSE(runner->isInitialized());
}

INSTANTIATE_TEST_SUITE_P(Qwen4ExpNgramAdmission, Qwen4ExpNgramAdmissionTest,
    ::testing::ValuesIn(qwen4ExpNgramAdmissionCases()),
    [](const ::testing::TestParamInfo<ModelParityCase> &info) { return info.param.testName(); });

/** @brief Preserve the canonical MPI lifecycle and retire backend authorities before process exit. */
int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) MPI_Abort(MPI_COMM_WORLD, 1);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();
    MPI_Finalize();
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
