/**
 * @file MaximumContextStateGeometry.h
 * @brief Original weight-free Qwen metadata and storage plans shared by preflight admission and allocation.
 *
 * The two shard layouts represent different local KV widths on a single device.
 * Every case must physically materialize at least one complete 262144-token layout.
 * Both qualification and native execution price identical production planner inputs.
 */
#pragma once
#include "execution/mtp/MTPGraphOwnerPlan.h"
#include "planning/MemoryPlanner.h"
#include "planning/PersistentStateMemoryEstimator.h"
#include <algorithm>
#include <array>
#include <limits>
#include <string_view>
#include <vector>

namespace llaminar2::test::maximum_context {

/** @brief Weight-free metadata fixtures; every size is a GGUF architecture
 * fact. */
inline ModelMemoryProfile maximumContextGeometry(bool moe) {
  ModelMemoryProfile p;
  p.architecture = moe ? "qwen35moe" : "qwen35";
  p.n_layers = moe ? 41 : 65;
  p.mtp_layer_count = 1;
  p.max_seq_len = 262144;
  p.d_model = moe ? 2048 : 5120;
  p.d_ff = moe ? 512 : 17408;
  p.n_heads = moe ? 16 : 24;
  p.n_kv_heads = moe ? 2 : 4;
  p.head_dim = 256;
  p.vocab_size = 248320;
  p.full_attention_interval = 4;
  p.gdn_conv_kernel_size = 4;
  p.gdn_state_size = 128;
  p.gdn_inner_size = moe ? 4096 : 6144;
  p.gdn_group_count = 16;
  p.gdn_time_step_rank = moe ? 32 : 48;
  p.expert_count = moe ? 256 : 0;
  p.expert_used_count = moe ? 8 : 0;
  p.expert_feed_forward_length = moe ? 512 : 0;
  p.expert_shared_feed_forward_length = moe ? 512 : 0;
  // The appended predictor is FA, not the next element in the main model's
  // periodic GDN/FA schedule. Preserve that tensor-directory distinction.
  p.tensors.push_back(
      {.name = "blk." + std::to_string(p.n_layers - 1) + ".attn_q.weight",
       .layer_index = p.n_layers - 1});
  return p;
}

/** @return Off, dynamic 1..15, then every fixed depth 2..15, without omissions.
 */
inline std::vector<MTPRuntimeConfig> maximumContextPolicies() {
  std::vector<MTPRuntimeConfig> policies(1);
  MTPRuntimeConfig dynamic;
  dynamic.enabled = true;
  dynamic.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
  dynamic.depth_policy.min_depth = 1;
  dynamic.depth_policy.max_depth = 15;
  policies.push_back(dynamic);
  for (int depth = 2; depth <= 15; ++depth) {
    MTPRuntimeConfig fixed;
    fixed.enabled = true;
    fixed.draft_tokens = depth;
    fixed.depth_policy.mode = MTPDepthPolicyMode::Fixed;
    policies.push_back(fixed);
  }
  return policies;
}

/** @brief Independent physical coverage for each model's main-only and MTP cache layouts. */
struct Case {
    std::string_view name;
    bool moe;
    bool sidecar;
};
inline constexpr std::array<Case, 4> cases{{
    {"DenseOff", false, false}, {"DenseMTP", false, true},
    {"MoEOff", true, false}, {"MoEMTP", true, true}}};
inline constexpr std::array<int, 2> shardCounts{1, 2};

/** @brief Build the original per-device profile at the real allocator capacity and selected local KV width. */
inline DevicePlanConfig configuration(const ModelMemoryProfile& profile, DeviceId device, int shards,
                                      const MTPRuntimeConfig& envelope, size_t total, size_t available) {
    const MTPGraphOwnerPlan envelope_owners(envelope, MTPVerifierGraphFamily::BoundedBuckets, MTPGenerationParentRole::Controller);
    DevicePlanConfig cfg;
    cfg.world_rank = 0;
    cfg.device = device;
    cfg.device_total_bytes = total;
    cfg.device_free_bytes = available;
    cfg.device_compute_units = 1;
    cfg.max_seq_len = profile.max_seq_len;
    cfg.activation_seq_len = 512;
    cfg.mtp_enabled = envelope.enabled;
    cfg.mtp_target_query_rows =
        envelope.enabled ? envelope_owners.flattenedTargetRows() : 1;
    cfg.total_shards = shards;
    cfg.local_kv_heads = profile.n_kv_heads / shards;
    cfg.kv_precision = "fp16";
    return cfg;
}

/** @brief Price the smallest admissible full-context layout using the production physical BOM. */
inline size_t minimumIncrementalBytes(Case item, DeviceId device, size_t total) {
    const auto profile = maximumContextGeometry(item.moe);
    const auto policies = maximumContextPolicies();
    const auto& envelope = policies.at(item.sidecar ? 1 : 0);
    size_t minimum = std::numeric_limits<size_t>::max();
    for (int shards : shardCounts) {
        const auto cfg = configuration(profile, device, shards, envelope, total, total);
        const auto plan = MemoryPlanner::plan(profile, {cfg});
        minimum = std::min(minimum, plan.physicalPlan().incrementalBytes());
    }
    return minimum;
}
}
