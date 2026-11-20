/**
 * @file Test__MaximumContextStateAllocation.cpp
 * @brief Real maximum-context cache/workspace allocation across MTP policies.
 *
 * The geometry is the GGUF metadata of Qwen3.8-27B and Qwen3.6-35B-A3B,
 * including their single full-attention predictor. No model file is required:
 * this is a production-factory physical allocation test, not a certificate
 * that a model's weights plus all graphs fit on one device. The latter remains
 * a real-model admission/HTTP obligation. In particular, large logical sampler
 * positions alone must never be mistaken for this physical-memory proof.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "kernels/HybridKVCacheConfig.h"
#include "kernels/IKVCache.h"
#include "kernels/KernelFactory.h"
#include "kernels/attention/AttentionWorkspaceContract.h"
#include "mocks/MockMPIContext.h"
#include "planning/MemoryPlanner.h"
#include "../../fixtures/planning/MaximumContextStateGeometry.h"
#include "planning/PersistentStateMemoryEstimator.h"
#include "planning/PhysicalMemoryAuthority.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <string>
#include <tuple>
#include <vector>

namespace {
using namespace llaminar2;
using namespace llaminar::v2::kernels;

using namespace llaminar2::test::maximum_context;

/** @brief Same factory/authority contract on CPU, CUDA and ROCm. */
class MaximumContextStateAllocation
    : public ::testing::TestWithParam<std::tuple<std::string, Case>> {};

/**
 * @brief Admit every MTP policy and materialize its full-context state
 * geometry.
 *
 * Main and shifted cache topology depends on model, head sharding and whether a
 * sidecar exists, not on draft depth. Allocate each immutable topology once
 * under the largest policy's certificate, then exercise every policy's admitted
 * workspace geometry against that same authority. This is the production
 * ownership rule for mutually exclusive depth graphs, and avoids clearing
 * terabytes of identical CPU slabs merely to test different query-row counts.
 */
TEST_P(MaximumContextStateAllocation, MaterializesEveryAdmittedPolicy) {
  const auto& [family, item] = GetParam();
  const DeviceId device = family == "CPU"    ? DeviceId::cpu()
                          : family == "CUDA" ? DeviceId::cuda(0)
                                             : DeviceId::rocm(0);
  if (device.is_cpu() && !hasCPUBackend())
    initCPUBackend(-1);
  IBackend *backend = getBackendFor(device);
  ASSERT_NE(backend, nullptr);
  const test::MockMPIContext local_mpi;
  const auto policies = maximumContextPolicies();
  const bool sidecar = item.sidecar;
  const auto profile = maximumContextGeometry(item.moe);
  std::vector<int> materialized(policies.size(), 0);
  std::string rejected;
  for (const int shards : shardCounts) {
    SCOPED_TRACE(::testing::Message()
                 << family << " geometry=" << item.name << " shards=" << shards
                 << " sidecar=" << sidecar);
    const auto &envelope = sidecar ? policies.at(1) : policies.front();
    const auto cfg = configuration(profile, device, shards, envelope,
                                   backend->deviceMemoryTotal(0), backend->deviceMemoryFree(0));
    const auto plan = MemoryPlanner::plan(profile, {cfg});
    if (!plan.physicalPlan().fits()) {
      // Rejection is a separate admission assertion, never a
      // reduced-context retry or a silent allocation skip.
      EXPECT_THROW((void)plan.admit(), PhysicalMemoryCapacityExhausted);
      rejected += plan.renderTable();
      continue;
    }
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(
            plan.physicalPlan()),
        0);

    HybridKVCacheConfig hybrid;
    hybrid.n_heads = profile.n_heads;
    hybrid.local_n_heads = profile.n_heads / shards;
    hybrid.gdn_conv_kernel_size = profile.gdn_conv_kernel_size;
    hybrid.gdn_state_size = profile.gdn_state_size;
    hybrid.gdn_inner_size = profile.gdn_inner_size;
    hybrid.gdn_group_count = profile.gdn_group_count;
    hybrid.gdn_time_step_rank = profile.gdn_time_step_rank;
    for (int layer = 0; layer < profile.n_layers - profile.mtp_layer_count;
         ++layer)
      hybrid.layer_types.push_back(
          PersistentStateMemoryEstimator::isFullAttentionLayer(profile,
                                                               layer)
              ? "full_attention"
              : "gdn");

    KVCacheConfig primary;
    primary.device = device;
    primary.precision = ActivationPrecision::FP16;
    primary.num_layers = profile.n_layers - profile.mtp_layer_count;
    primary.max_seq_len = profile.max_seq_len;
    primary.n_kv_heads = profile.n_kv_heads;
    primary.local_n_kv_heads = cfg.local_kv_heads;
    primary.head_dim = profile.head_dim;
    primary.mpi_ctx = &local_mpi;
    primary.hybrid_config = &hybrid;
    primary.physical_memory_authority = authority;
    KVCacheConfig shifted = primary;
    shifted.num_layers = profile.mtp_layer_count;
    shifted.hybrid_config = nullptr;
    const auto cache_bytes =
        primary.estimateBytes() + (sidecar ? shifted.estimateBytes() : 0u);
    EXPECT_EQ(authority->plannedBytes(device, PhysicalMemoryOwner::KVCache),
              cache_bytes);

    {
      auto main_cache = KernelFactory::createKVCache(primary);
      ASSERT_NE(main_cache, nullptr);
      EXPECT_EQ(main_cache->max_seq_len(), profile.max_seq_len);
      EXPECT_TRUE(main_cache->hasPhysicalMemoryLease());
      auto mtp_cache =
          sidecar ? KernelFactory::createKVCache(shifted) : nullptr;
      if (sidecar) {
        ASSERT_NE(mtp_cache, nullptr);
        EXPECT_EQ(mtp_cache->max_seq_len(), profile.max_seq_len);
        EXPECT_TRUE(mtp_cache->hasPhysicalMemoryLease());
      }
      EXPECT_EQ(authority->remainingAdmittedNewAllocationBytes(
                    device, PhysicalMemoryOwner::KVCache),
                0u);
      EXPECT_EQ(authority->remainingAdmittedNewAllocationBytes(
                    device, PhysicalMemoryOwner::RecurrentLiveState),
                0u);

      for (size_t index = 0; index < policies.size(); ++index) {
        const auto &mtp = policies[index];
        if (mtp.enabled != sidecar)
          continue;
        const MTPGraphOwnerPlan owners(mtp, MTPVerifierGraphFamily::BoundedBuckets, MTPGenerationParentRole::Controller);
        const int query_rows = sidecar ? owners.flattenedTargetRows() : 1;
        SCOPED_TRACE(::testing::Message()
                     << "depth=" << owners.draftDepth() << " policy="
                     << static_cast<int>(mtp.depth_policy.mode)
                     << " query_rows=" << query_rows);
        auto policy_cfg = cfg;
        policy_cfg.mtp_target_query_rows = query_rows;
        const auto policy_plan = MemoryPlanner::plan(profile, {policy_cfg});
        ASSERT_TRUE(policy_plan.physicalPlan().fits())
            << policy_plan.renderTable();
        const auto *policy_bom = policy_plan.physicalPlan().find(
            {.world_rank = 0, .device = device});
        ASSERT_NE(policy_bom, nullptr);
        EXPECT_EQ(policy_bom->bytes(PhysicalMemoryOwner::KVCache),
                  cache_bytes);

        // Each real query geometry is independently allocated from
        // the envelope. Depth graphs are mutually exclusive and
        // can share the same largest admitted workspace region.
        const auto requirements =
            device.is_cpu()
                ? attention_workspace::cpuParallelRequirements(
                      {.compact_query_rows = query_rows,
                       .local_query_heads = profile.n_heads / shards,
                       .head_dim = profile.head_dim,
                       .worker_count = 1})
                : attention_workspace::requirements(
                      {.compact_query_rows = query_rows,
                       .request_count = 1,
                       .local_query_heads = profile.n_heads / shards,
                       .local_kv_heads = cfg.local_kv_heads,
                       .head_dim = profile.head_dim,
                       .context_rows = profile.max_seq_len,
                       .include_device_params = true,
                       .include_fp32_kv_conversion = false});
        DeviceWorkspaceManager workspace(
            device,
            authority->plannedBytes(
                device, PhysicalMemoryOwner::ExecutionWorkspace),
            authority);
        ASSERT_TRUE(workspace.allocate(requirements));
        ++materialized[index];
      }
      if (device.is_gpu()) {
        // Construction publishes asynchronously. Only this final
        // test observation/retirement boundary waits on its stream.
        auto &worker = GPUDeviceContextPool::instance().getContext(device);
        ASSERT_TRUE(
            worker.synchronizeStreamChecked(worker.defaultStream()));
      }
    }
    EXPECT_EQ(authority->remainingAdmittedNewAllocationBytes(
                  device, PhysicalMemoryOwner::KVCache),
              cache_bytes);
    EXPECT_EQ(authority->remainingAdmittedNewAllocationBytes(
                  device, PhysicalMemoryOwner::RecurrentLiveState),
              authority->plannedBytes(
                  device, PhysicalMemoryOwner::RecurrentLiveState));
  }
  for (size_t index = 0; index < policies.size(); ++index) {
    if (policies[index].enabled != sidecar) continue;
    ASSERT_GT(materialized[index], 0)
        << "A full-context layout must physically cover every policy: " << item.name << " policy_index=" << index << rejected;
  }
}

INSTANTIATE_TEST_SUITE_P(Backends, MaximumContextStateAllocation,
                         ::testing::Combine(::testing::Values("CPU", "CUDA", "ROCm"),
                                            ::testing::ValuesIn(cases)),
                         [](const auto &info) {
                           return std::get<0>(info.param) + std::string(std::get<1>(info.param).name);
                         });
} // namespace
