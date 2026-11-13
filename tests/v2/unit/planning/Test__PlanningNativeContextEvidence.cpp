/** @file Test__PlanningNativeContextEvidence.cpp
 * @brief Device-free native birth role/ownership replay, with no runtime or
 * model payload materialization.
 */
#include "../../utils/CPUExecutionTestGeometry.h"
#include "../../utils/PlanningGGUFFixture.h"
#include "../../utils/PlanningHostServiceFixture.h"
#include "backends/PreparedGPUWorkerThread.h"
#include "config/OrchestrationConfigDocument.h"
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include "mocks/MockWorkerGPUContext.h"
#include "planning/AutomaticOrchestrationCandidates.h"
#include "planning/ExecutionRankMembership.h"
#include "planning/MemoryPlanner.h"
#include "planning/OrchestrationCandidateAdmission.h"
#include "planning/PlanningModelMetadata.h"
#include "planning/PlanningNativeContextEvidence.h"
#include <array>
#include <gtest/gtest.h>
#include <sstream>
#include <type_traits>
namespace llaminar2::test {
namespace {
/** @return Complete recorded family with real original pthread geometry; never
 * production observations. */
PlanningNativeContextDeclaration recorded(DeviceId device) {
  auto program = GPUWorkerStackProgram::originalDefaults();
  return {device,
          "recorded-device",
          program.stackBytes(),
          program.guardBytes(),
          program.pageBytes(),
          64,
          64};
}
} // namespace
/** @test The same key inputs price opaque commitments and original stack pages
 * on every backend. */
TEST(PlanningNativeContextEvidence, InstalledFamilyOwnersAndOmissionReject) {
  const PhysicalMemoryResource host{0, DeviceId::cpu(), 16ull << 30,
                                    16ull << 30};
  for (const auto device : {DeviceId::cuda(2), DeviceId::rocm(4)}) {
    const auto k = recorded(device);
    const std::array keys{k};
    const std::array gpu{
        PhysicalMemoryResource{0, device, 16ull << 30, 16ull << 30}};
    const auto automatic = planningNativeContextBOM(keys, host, gpu);
    const auto contract = NativeContextOpaqueCatalog::planningBudget(
        k.device.type, NativeContextBirthPopulation::AdditionalContext);
    EXPECT_EQ(automatic.find({0, device})
                  ->bytes(PhysicalMemoryOwner::NativeExecutionContext),
              contract.deviceUpperBytes());
    EXPECT_EQ(automatic.find({0, DeviceId::cpu()})
                  ->bytes(PhysicalMemoryOwner::NativeExecutionContext),
              contract.hostUpperBytes());
    EXPECT_EQ(automatic.find({0, DeviceId::cpu()})
                  ->bytes(PhysicalMemoryOwner::LocalCollective),
              GPUWorkerStackProgram::planningAllocationBytes(
                  k.stack_bytes, k.guard_bytes, k.page_bytes, k.object_bytes,
                  k.object_alignment));
    EXPECT_THROW(planningNativeContextBOM({}, host, gpu),
                 std::invalid_argument);
    const std::array duplicate{k, k};
    EXPECT_THROW(planningNativeContextBOM(duplicate, host, gpu),
                 std::invalid_argument);
  }
  EXPECT_EQ(planningNativeContextBOM({}, host, {}).totalBytes(), 0u);
}
/** @test Original rank1 remains the authority when selected execution
 * membership compacts it to0. */
TEST(PlanningNativeContextEvidence,
     OriginalRankRemapNeverBorrowsRootOrRequiresGPUForCPUOnly) {
  ClusterInventory inventory;
  inventory.world_size = 2;
  for (int i = 0; i < 2; ++i) {
    RankInventory r;
    r.rank = i;
    r.node_id = i;
    r.local_rank = 0;
    r.hostname = "host" + std::to_string(i);
    r.cpu.memory_bytes = r.cpu.free_memory_bytes = 16ull << 30;
    r.cpu.compute_units = r.cpu_cores = r.cpu_worker_threads = 1;
    r.numa_nodes = 1;
    inventory.ranks.push_back(r);
  }
  inventory.buildNodeAggregations();
  ExecutionRankMembership membership(inventory, {1});
  PlanningNativeContextBudgets budgets{{0, {}},
                                       {1, {recorded(DeviceId::rocm(4))}}};
  ASSERT_EQ(projectPlanningNativeContextBudget(budgets, membership, 0).size(),
            1u);
  EXPECT_EQ(
      projectPlanningNativeContextBudget(budgets, membership, 0).front().device,
      DeviceId::rocm(4));
  budgets.erase(1);
  EXPECT_THROW(projectPlanningNativeContextBudget(budgets, membership, 0),
               std::invalid_argument);
  budgets.emplace(1, std::vector<PlanningNativeContextDeclaration>{});
  EXPECT_TRUE(
      projectPlanningNativeContextBudget(budgets, membership, 0).empty());
}
/** @test Planning limits neither authorize a native birth nor invent an
 * unsupported class. */
TEST(PlanningNativeContextEvidence,
     FamilyBudgetCannotAuthorizeBirthOrInventUnsupportedScope) {
  static_assert(!std::is_convertible_v<NativeContextFamilyBudget,
                                       NativeContextOpaqueContract>);
  EXPECT_THROW(
      NativeContextOpaqueCatalog::planningBudget(
          DeviceType::CPU, NativeContextBirthPopulation::AdditionalContext),
      std::runtime_error);
  EXPECT_THROW(NativeContextOpaqueCatalog::planningBudget(
                   DeviceType::ROCm,
                   NativeContextBirthPopulation::InitialBackendContext),
               std::runtime_error);
}

/** @test Real source mandatory BOM and planning declaration contribute
 * identical original owners. */
TEST(PlanningNativeContextEvidence,
     ActualSourceMandatoryBOMMatchesPlanningProjection) {
  using Access = NativeContextPreparationReplayAccess;
  class Worker final : public llaminar2::testing::MockWorkerGPUContext {
  public:
    Worker(int ordinal, std::shared_ptr<PreparedGPUWorkerStack>)
        : MockWorkerGPUContext(ordinal) {}
  };
  const auto device = DeviceId::cuda(87);
  const std::array resources{
      PhysicalMemoryResource{0, DeviceId::cpu(), 4ull << 30, 4ull << 30},
      PhysicalMemoryResource{0, device, 4ull << 30, 4ull << 30}};
  const std::array devices{device};
  Access::seed(device, [](std::shared_ptr<PreparedGPUWorkerStack> root) {
    return Access::construct<Worker>(std::move(root), 87);
  });
  auto native = Access::source<Worker>(
      devices, resources, NativePhysicalLifetime::ContextGeneration, true);
  struct Retirement {
    DeviceId device;
    std::shared_ptr<NativeContextPreparation> source;
    ~Retirement() {
      auto scope =
          GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(
              device);
      Access::physicalReset(scope);
      if (source->memory()->phase() == PhysicalMemoryAuthorityPhase::Preparing)
        source->retireUnadmitted();
    }
  } retirement{device, native};
  {
    auto use = native->acquire(device);
  }
  native->freezeForPlanning();
  const auto key = Access::birthKey(native, device);
  const auto program = GPUWorkerStackProgram::originalDefaults();
  PlanningNativeContextDeclaration declaration{device,
                                               key->physical_uuid,
                                               key->worker_stack_bytes,
                                               key->worker_guard_bytes,
                                               program.pageBytes(),
                                               key->context_object_bytes,
                                               key->context_object_alignment};
  PlanningGGUFFixture file;
  PlanningModelSource model(file.path());
  DevicePlanConfig cfg;
  cfg.world_rank = 0;
  cfg.device = device;
  cfg.device_total_bytes = cfg.device_free_bytes = 4ull << 30;
  cfg.device_compute_units = 80;
  cfg.max_seq_len = 64;
  cfg.activation_seq_len = 1;
  cfg.first_layer = 0;
  cfg.last_layer = model.metadata().memoryProfile().n_layers - 1;
  cfg.associated_host_memory = resources[0];
  cfg.native_context_preparation = native;
  const auto actual =
      MemoryPlanner::plan(model.metadata().memoryProfile(), {cfg})
          .physicalPlan();
  cfg.native_context_preparation.reset();
  cfg.planning_native_context_birth = declaration;
  const auto planned =
      MemoryPlanner::plan(model.metadata().memoryProfile(), {cfg})
          .physicalPlan();
  EXPECT_FALSE(actual.requiredFootprintMismatch(planned));
  EXPECT_FALSE(planned.requiredFootprintMismatch(actual));
  const auto original = native->memory();
  auto certificate =
      std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(actual);
  native->admitFinal(certificate);
  EXPECT_EQ(native->memory(), original);
  EXPECT_EQ(original->admission(), certificate);
}

/** @test Real candidate compilation prices selected native owners before
 * ordinary/overlay/pipeline admission. */
TEST(PlanningNativeContextEvidence,
     ActualCandidateAdmissionProjectsOriginalRankAcrossAllStrategies) {
  PlanningGGUFFixture file(true, true, GGUFTensorType::F16, 256, std::nullopt,
                           8, PlanningGGUFFixture::Geometry::Directory, 4);
  PlanningModelSource model(file.path());
  ClusterInventory inventory;
  inventory.world_size = 2;
  for (int rank = 0; rank < 2; ++rank) {
    RankInventory r;
    r.rank = rank;
    r.node_id = rank;
    r.local_rank = 0;
    r.hostname = "host" + std::to_string(rank);
    r.cpu.numa_node = rank;
    r.cpu.memory_bytes = r.cpu.free_memory_bytes = 128ull << 30;
    r.cpu.compute_units = r.cpu_cores = r.cpu_worker_threads = 8;
    r.cpu_execution = kSyntheticCPUExecutionGeometry;
    r.numa_nodes = 1;
    for (int ordinal = 2; ordinal < 4; ++ordinal)
      r.gpus.push_back({.type = DeviceType::ROCm,
                        .local_device_id = ordinal,
                        .memory_bytes = 128ull << 30,
                        .free_memory_bytes = 120ull << 30,
                        .compute_units = 64,
                        .uuid = "rank" + std::to_string(rank) + "-gpu" +
                                std::to_string(ordinal),
                        .numa_node = rank});
    inventory.ranks.push_back(std::move(r));
  }
  inventory.buildNodeAggregations();
  for (const auto strategy : {OrchestrationStrategy::SingleDevice,
                              OrchestrationStrategy::ExpertOverlay,
                              OrchestrationStrategy::PipelineParallel}) {
    OrchestrationConfig request;
    request.model_path = file.path();
    request.max_seq_len = 64;
    request.prefix_cache.enabled = false;
    request.moe_rebalance.mode = MoERebalanceRuntimeMode::Off;
    request.mtp.enabled = true;
    request.mtp.graph_capacity_draft_tokens = 15;
    request.mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
    request.mtp.depth_policy.max_depth = 15;
    // Two same-rank GPU domains have no executable overlay boundary. The
    // overlay fixture explicitly admits a same-rank CPU expert domain.
    request.automatic_planning.only_backends =
        strategy == OrchestrationStrategy::ExpertOverlay
            ? std::vector{DeviceType::ROCm, DeviceType::CPU}
            : std::vector{DeviceType::ROCm};
    request.automatic_planning.only_strategies = std::vector{strategy};
    std::optional<AutomaticOrchestrationCandidate> selected;
    std::ostringstream emitted;
    visitAutomaticOrchestrationCandidates(
        request, model.metadata(), inventory, [&](auto candidate) {
          emitted << " [";
          for (const auto rank : candidate.membership.discoveryRanks())
            emitted << rank << ',';
          emitted << ']';
          if (candidate.membership.discoveryRanks() == std::vector<int>{1})
            selected = std::move(candidate);
        });
    ASSERT_TRUE(selected.has_value())
        << "strategy=" << static_cast<int>(strategy)
        << " emitted memberships=" << emitted.str();
    OrchestrationCandidateMemoryPolicy policy{
        .prefill_bucket_rows = {32, 64},
        .minimum_prefill_sequence_rows = 1,
        .maximum_cached_prefill_buckets = 8,
        .retained_host_services = planningHostServiceBudgets(2)};
    for (const auto &gpu : inventory.ranks[1].gpus) {
      auto declaration = recorded({gpu.type, gpu.local_device_id});
      declaration.physical_uuid = gpu.uuid;
      policy.native_context_budgets[1].push_back(declaration);
    }
    const auto without = AdmittedOrchestrationCandidate::admit(
        *selected, model,
        OrchestrationCandidateMemoryPolicy{.prefill_bucket_rows = {32, 64},
                                           .minimum_prefill_sequence_rows = 1,
                                           .maximum_cached_prefill_buckets = 8,
                                           .retained_host_services =
                                               planningHostServiceBudgets(2)});
    policy.require_native_context_budgets = true;
    const auto with =
        AdmittedOrchestrationCandidate::admit(*selected, model, policy);
    auto restored = *selected;
    restored.config = deserializeOrchestrationConfig(
        serializeOrchestrationConfig(with.config()));
    const auto roundtrip =
        AdmittedOrchestrationCandidate::admit(restored, model, policy);
    EXPECT_FALSE(with.physicalAdmission().plan().requiredFootprintMismatch(
        roundtrip.physicalAdmission().plan()));
    EXPECT_FALSE(roundtrip.physicalAdmission().plan().requiredFootprintMismatch(
        with.physicalAdmission().plan()));
    size_t gpu_count = 0;
    for (const auto &row : with.physicalAdmission().plan().resources())
      if (row.resource().device.is_gpu()) {
        ++gpu_count;
        EXPECT_EQ(row.resource().world_rank, 0);
        EXPECT_EQ(row.bytes(PhysicalMemoryOwner::NativeExecutionContext),
                  NativeContextOpaqueCatalog::planningBudget(
                      row.resource().device.type,
                      NativeContextBirthPopulation::AdditionalContext)
                      .deviceUpperBytes());
        const auto *old =
            without.physicalAdmission().plan().find({0, row.resource().device});
        ASSERT_NE(old, nullptr);
        EXPECT_EQ(old->bytes(PhysicalMemoryOwner::NativeExecutionContext), 0u);
      }
    EXPECT_GT(gpu_count, 0u);
    policy.native_context_budgets[1].clear();
    EXPECT_THROW(
        (void)AdmittedOrchestrationCandidate::admit(*selected, model, policy),
        std::invalid_argument);
  }
}

} // namespace llaminar2::test
