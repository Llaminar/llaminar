#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "execution/moe/MoERuntimeTable.h"
#include "execution/prefix_cache/PrefixCacheFingerprint.h"

using namespace llaminar2;

namespace
{
    PrefixFingerprintMaterial makeMaterial()
    {
        PrefixFingerprintMaterial material;
        material.model = {
            {"architecture", "qwen2"},
            {"gguf_metadata", "model-meta-a"},
            {"tensor_directory", "tensor-dir-a"},
            {"vocab_size", "151936"},
            {"lm_head", "tied"},
        };
        material.tokenizer = {
            {"model", "bpe"},
            {"added_tokens", "42"},
            {"chat_template", "chatml"},
        };
        material.runtime = {
            {"activation_precision", "fp32"},
            {"kv_precision", "q16_1"},
            {"kv_layout", "kv_pos_head_dim"},
            {"rope_on_read", "false"},
            {"fused_attention", "jit"},
        };
        material.topology = {
            {"tp_degree", "1"},
            {"pp_degree", "1"},
            {"rank", "0"},
            {"device", "cpu:0"},
            {"kv_head_start", "0"},
            {"local_kv_heads", "8"},
        };
        material.hybrid = {
            {"layer_types", "fa,fa"},
            {"gdn_layers", "0"},
        };
        material.moe = {
            {"expert_count", "0"},
            {"placement_epoch", "0"},
        };
        material.mtp = {
            {"enabled", "false"},
            {"depth", "0"},
        };
        return material;
    }

    DeviceNativeVNNIMatrixDesc fakeMatrixDesc(int seed)
    {
        const auto base = static_cast<uintptr_t>(0x10000 + seed * 0x100);
        DeviceNativeVNNIMatrixDesc desc;
        desc.payload = reinterpret_cast<const uint8_t *>(base);
        desc.scales = reinterpret_cast<const void *>(base + 0x40);
        desc.n = 16;
        desc.k = 32;
        desc.blocks_per_row = 2;
        return desc;
    }

    MoEPlacementUpdate makePlacementUpdate(
        uint32_t epoch,
        const std::vector<int> &owners,
        const std::vector<uint8_t> &local_compute,
        const std::vector<uint8_t> &replica_roles)
    {
        MoEPlacementUpdate update;
        update.epoch = epoch;
        update.expert_count = static_cast<uint32_t>(owners.size());
        update.experts.resize(owners.size());
        update.local_compute_mask = local_compute;
        update.replica_role = replica_roles;

        for (size_t expert = 0; expert < owners.size(); ++expert)
        {
            DeviceMoEExpertDescriptor desc;
            desc.gate = fakeMatrixDesc(static_cast<int>(expert) * 3 + 0);
            desc.up = fakeMatrixDesc(static_cast<int>(expert) * 3 + 1);
            desc.down = fakeMatrixDesc(static_cast<int>(expert) * 3 + 2);
            desc.logical_expert_id = static_cast<int32_t>(expert);
            desc.owner_participant = owners[expert];
            desc.local_slot = static_cast<int32_t>(expert);
            desc.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                                          DeviceMoEExpertFlags::Resident |
                                          DeviceMoEExpertFlags::LocalCompute);
            update.experts[expert] = desc;
        }
        return update;
    }
} // namespace

TEST(Test__PrefixCacheFingerprint, DeterministicAndFieldOrderIndependent)
{
    PrefixFingerprintMaterial material = makeMaterial();
    PrefixFingerprintMaterial reordered = material;
    std::reverse(reordered.model.begin(), reordered.model.end());
    std::reverse(reordered.runtime.begin(), reordered.runtime.end());

    const auto first = buildPrefixCacheFingerprint(
        material, /*model_is_moe=*/false, PrefixCacheMoEPolicy::PlacementFingerprint);
    const auto second = buildPrefixCacheFingerprint(
        material, /*model_is_moe=*/false, PrefixCacheMoEPolicy::PlacementFingerprint);
    const auto third = buildPrefixCacheFingerprint(
        reordered, /*model_is_moe=*/false, PrefixCacheMoEPolicy::PlacementFingerprint);

    EXPECT_FALSE(first.bypass);
    EXPECT_NE(first.key, 0u);
    EXPECT_EQ(first.key, second.key);
    EXPECT_EQ(first.key, third.key);
}

TEST(Test__PrefixCacheFingerprint, ChangingEachNamedPartChangesFinalKey)
{
    const PrefixFingerprintMaterial baseline = makeMaterial();
    const uint64_t baseline_key = buildPrefixCacheFingerprint(
                                      baseline, /*model_is_moe=*/false, PrefixCacheMoEPolicy::PlacementFingerprint)
                                      .key;

    auto expect_changed = [&](auto mutator)
    {
        PrefixFingerprintMaterial changed = baseline;
        mutator(changed);
        const uint64_t changed_key = buildPrefixCacheFingerprint(
                                         changed, /*model_is_moe=*/false, PrefixCacheMoEPolicy::PlacementFingerprint)
                                         .key;
        EXPECT_NE(changed_key, baseline_key);
    };

    expect_changed([](PrefixFingerprintMaterial &m) { m.model.push_back({"tensor_shape", "changed"}); });
    expect_changed([](PrefixFingerprintMaterial &m) { m.tokenizer.push_back({"added_token_hash", "changed"}); });
    expect_changed([](PrefixFingerprintMaterial &m) { m.runtime.push_back({"partial_rope_factor", "0.5"}); });
    expect_changed([](PrefixFingerprintMaterial &m) { m.topology.push_back({"vocab_shard", "1/2"}); });
    expect_changed([](PrefixFingerprintMaterial &m) { m.hybrid.push_back({"gdn_state_size", "128"}); });
    expect_changed([](PrefixFingerprintMaterial &m) { m.moe.push_back({"active_bank", "1"}); });
    expect_changed([](PrefixFingerprintMaterial &m) { m.mtp.push_back({"draft_tokens", "2"}); });
}

TEST(Test__PrefixCacheFingerprint, MoEDisabledPolicyBypassesOnlyMoEModels)
{
    const PrefixFingerprintMaterial material = makeMaterial();

    const auto moe_result = buildPrefixCacheFingerprint(
        material, /*model_is_moe=*/true, PrefixCacheMoEPolicy::Disabled);
    EXPECT_TRUE(moe_result.bypass);
    EXPECT_EQ(moe_result.key, 0u);
    EXPECT_FALSE(moe_result.bypass_reason.empty());

    const auto dense_result = buildPrefixCacheFingerprint(
        material, /*model_is_moe=*/false, PrefixCacheMoEPolicy::Disabled);
    EXPECT_FALSE(dense_result.bypass);
    EXPECT_NE(dense_result.key, 0u);
}

TEST(Test__PrefixCacheFingerprint, MoEPlacementEpochChangesKeyMaterial)
{
    PrefixFingerprintMaterial stable = makeMaterial();
    stable.moe = {
        {"expert_count", "256"},
        {"top_k", "8"},
        {"placement_epoch", "7"},
        {"active_bank", "0"},
        {"local_compute_mask", "0xff"},
    };

    PrefixFingerprintMaterial rebalanced = stable;
    rebalanced.moe = {
        {"expert_count", "256"},
        {"top_k", "8"},
        {"placement_epoch", "8"},
        {"active_bank", "1"},
        {"local_compute_mask", "0x0f"},
    };

    const uint64_t stable_key = buildPrefixCacheFingerprint(
                                    stable, /*model_is_moe=*/true, PrefixCacheMoEPolicy::PlacementFingerprint)
                                    .key;
    const uint64_t rebalanced_key = buildPrefixCacheFingerprint(
                                        rebalanced, /*model_is_moe=*/true, PrefixCacheMoEPolicy::PlacementFingerprint)
                                        .key;

    EXPECT_NE(stable_key, rebalanced_key);
}

TEST(Test__PrefixCacheFingerprint, MoERuntimePlacementChangesKeyButHistogramDoesNot)
{
    MoERuntimeTable table(DeviceId::cpu(), /*num_layers=*/1, /*num_experts=*/3, /*top_k=*/2);
    auto first_update = makePlacementUpdate(
        /*epoch=*/1,
        /*owners=*/{0, 0, 1},
        /*local_compute=*/{1, 1, 0},
        /*replica_roles=*/{
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Replica),
        });
    ASSERT_TRUE(table.prepareInactiveBank(/*layer_idx=*/0, first_update));
    ASSERT_TRUE(table.flipActiveBank(/*layer_idx=*/0, /*epoch=*/1, /*stream=*/nullptr));

    std::vector<PrefixFingerprintField> first_fields;
    appendMoEPlacementFingerprintFields(first_fields, table, /*layer_count=*/1, "runtime");
    const uint64_t first_hash = hashPrefixFingerprintFields("moe", first_fields);

    table.hostLayerState(0).decode_histogram[0] = 123;
    table.hostLayerState(0).decode_histogram[2] = 456;

    std::vector<PrefixFingerprintField> histogram_fields;
    appendMoEPlacementFingerprintFields(histogram_fields, table, /*layer_count=*/1, "runtime");
    const uint64_t histogram_hash = hashPrefixFingerprintFields("moe", histogram_fields);
    EXPECT_EQ(histogram_hash, first_hash)
        << "Decode histogram counters are telemetry and must not enter prefix payload keys";

    auto second_update = makePlacementUpdate(
        /*epoch=*/2,
        /*owners=*/{1, 0, 1},
        /*local_compute=*/{0, 1, 1},
        /*replica_roles=*/{
            static_cast<uint8_t>(DeviceMoEReplicaRole::Replica),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
            static_cast<uint8_t>(DeviceMoEReplicaRole::Primary),
        });
    ASSERT_TRUE(table.prepareInactiveBank(/*layer_idx=*/0, second_update));
    ASSERT_TRUE(table.flipActiveBank(/*layer_idx=*/0, /*epoch=*/2, /*stream=*/nullptr));

    std::vector<PrefixFingerprintField> rebalanced_fields;
    appendMoEPlacementFingerprintFields(rebalanced_fields, table, /*layer_count=*/1, "runtime");
    const uint64_t rebalanced_hash = hashPrefixFingerprintFields("moe", rebalanced_fields);
    EXPECT_NE(rebalanced_hash, first_hash);
}
