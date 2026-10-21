/**
 * @file Test__PrefixCacheFingerprint.cpp
 * @brief Device-free prefix identity and placement-publication regressions.
 *
 * Exercise stable field hashing, policy-dependent epoch invalidation, and
 * immutable provenance without executing models or accessing accelerators.
 * Numerical generation changes reject incompatible recurrent and terminal
 * state using metadata alone, including the finite Q8 scale publication rule.
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <set>
#include <vector>

#include "execution/moe/MoERuntimeTable.h"
#include "execution/prefix_cache/PrefixCacheFingerprint.h"

using namespace llaminar2;

/** @brief QFN activation correction cannot restore state produced by the ordinary path. */
TEST(Test__PrefixCacheFingerprint, FlashNextHCUpActivationSeparatesRestarts)
{
    const std::vector<PrefixFingerprintField> base{{"kv_cache_precision", "fp16"}};
    std::set<uint64_t> keys{hashPrefixFingerprintFields("runtime", base)};
    for (const auto policy : {QuantizedActivationOperandPrecision::SingleQ8,
                              QuantizedActivationOperandPrecision::TwoTermQ8})
    {
        auto fields = base;
        appendQwen4ExpArithmeticFingerprintFields(fields, policy);
        const auto key = hashPrefixFingerprintFields("runtime", fields);
        EXPECT_TRUE(keys.insert(key).second);
        auto restarted = base;
        appendQwen4ExpArithmeticFingerprintFields(restarted, policy);
        std::reverse(restarted.begin(), restarted.end());
        EXPECT_EQ(key, hashPrefixFingerprintFields("runtime", restarted));
    }
    EXPECT_EQ(keys.size(), 3u);
}

/** @brief Reject conflicting or invalid arithmetic before mutating compatibility material. */
TEST(Test__PrefixCacheFingerprint, FlashNextHCUpActivationRejectsDuplicateAuthority)
{
    for (const auto name : {"qfn_hc_up_activation", "qfn_hc_up_activation_revision"})
    {
        std::vector<PrefixFingerprintField> fields{{name, "previous"}};
        EXPECT_THROW(appendQwen4ExpArithmeticFingerprintFields(fields,
            QuantizedActivationOperandPrecision::TwoTermQ8), std::invalid_argument);
        ASSERT_EQ(fields.size(), 1u);
        EXPECT_EQ(fields.front().value, "previous");
    }
    std::vector<PrefixFingerprintField> fields;
    EXPECT_THROW(appendQwen4ExpArithmeticFingerprintFields(fields,
        static_cast<QuantizedActivationOperandPrecision>(255)), std::invalid_argument);
    EXPECT_TRUE(fields.empty());
}

/** @brief Every supported arithmetic pair has a restart-stable metadata identity. */
TEST(Test__PrefixCacheFingerprint, ProjectionArithmeticSeparatesEveryPrecisionPair)
{
    std::set<uint64_t> keys;
    const std::vector<PrefixFingerprintField> historical{{"kv_cache_precision", "fp16"}};
    const auto old_key = hashPrefixFingerprintFields("runtime", historical);
    for (auto gdn : {GDNProjectionArithmetic::Native, GDNProjectionArithmetic::FP32,
                    GDNProjectionArithmetic::FP16, GDNProjectionArithmetic::FP16AlphaBeta, GDNProjectionArithmetic::FP16MoEAlphaBeta})
        for (auto router : {MoERouterArithmetic::Native, MoERouterArithmetic::FP16, MoERouterArithmetic::NativeHadamard32})
        for (auto prefill : {CUDARouterPrefillArithmetic::FP32, CUDARouterPrefillArithmetic::Q8})
        {
            auto fields = historical;
            appendProjectionArithmeticFingerprintFields(fields, gdn, router, prefill);
            const auto key = hashPrefixFingerprintFields("runtime", fields);
            EXPECT_NE(key, old_key);
            EXPECT_TRUE(keys.insert(key).second);
            auto restarted = historical;
            appendProjectionArithmeticFingerprintFields(restarted, gdn, router, prefill);
            std::reverse(restarted.begin(), restarted.end());
            EXPECT_EQ(key, hashPrefixFingerprintFields("runtime", restarted));
        }
    EXPECT_EQ(keys.size(), 30u);
}

/** @brief Callers cannot relabel a persisted entry with conflicting arithmetic metadata. */
TEST(Test__PrefixCacheFingerprint, ProjectionArithmeticRejectsDuplicateAuthority)
{
    for (const auto name : {"projection_arithmetic_revision", "gdn_projection_arithmetic",
                            "moe_router_arithmetic", "cuda_router_prefill_arithmetic"})
    {
        std::vector<PrefixFingerprintField> fields{{name, "previous"}};
        EXPECT_THROW(appendProjectionArithmeticFingerprintFields(fields,
            GDNProjectionArithmetic::FP16, MoERouterArithmetic::FP16), std::invalid_argument);
        EXPECT_EQ(fields.size(), 1u);
    }
    std::vector<PrefixFingerprintField> fields;
    EXPECT_THROW(appendProjectionArithmeticFingerprintFields(fields,
        static_cast<GDNProjectionArithmetic>(255), MoERouterArithmetic::Native), std::invalid_argument);
    EXPECT_THROW(appendProjectionArithmeticFingerprintFields(fields,
        GDNProjectionArithmetic::Native, static_cast<MoERouterArithmetic>(255)), std::invalid_argument);
    EXPECT_TRUE(fields.empty());
}

/** @brief Persisted unsliced-vocabulary archives must miss after this fix is installed. */
TEST(Test__PrefixCacheFingerprint, LegacyUnslicedTerminalArchivesAreIncompatible)
{
    const PrefixFingerprintParts parts{.model = 1, .tokenizer = 2, .runtime = 3,
        .topology = 4, .hybrid = 5, .moe = 6, .mtp = 7};
    // Frozen output from the original v2 serializer, not the current builder.
    constexpr uint64_t old_unsliced_archive_key = 0x212b6fe0db14eb16ULL;
    EXPECT_NE(combinePrefixFingerprintParts(parts), old_unsliced_archive_key);
}

/**
 * @brief Reject restart state produced before FP16 tensor-tile prefill.
 *
 * The golden key is the frozen v5 metadata serializer for these seven parts.
 * KV, recurrent GDN state and terminal logits share this arithmetic identity;
 * invalidation requires no payload hashing or device access.
 */
TEST(Test__PrefixCacheFingerprint, LegacyFP32PrefillArithmeticCannotRestore)
{
    const PrefixFingerprintParts parts{.model = 1, .tokenizer = 2, .runtime = 3,
        .topology = 4, .hybrid = 5, .moe = 6, .mtp = 7};
    constexpr uint64_t legacy_fp32_prefill_key = 0x32047ed96a3e1be1ULL;
    EXPECT_NE(combinePrefixFingerprintParts(parts), legacy_fp32_prefill_key);
}

/**
 * @brief Larger-bucket FP32 archives cannot alias the expanded FP16 dispatch.
 *
 * V6 used the tensor arithmetic only at M512. The identical model/topology
 * metadata could therefore name an M1024 archive produced by the old FP32
 * body. The frozen V6 key must miss before reading any KV/GDN/logit payload.
 */
TEST(Test__PrefixCacheFingerprint, LegacyM512OnlyTensorArithmeticCannotRestore)
{
    const PrefixFingerprintParts parts{.model = 1, .tokenizer = 2, .runtime = 3,
        .topology = 4, .hybrid = 5, .moe = 6, .mtp = 7};
    constexpr uint64_t legacy_m512_only_tensor_key = 0x121dc518e20a501aULL;
    EXPECT_NE(combinePrefixFingerprintParts(parts), legacy_m512_only_tensor_key);
}

/** V7 large-M query bodies used FP32 despite admitting a native tensor mapping. */
TEST(Test__PrefixCacheFingerprint, LegacyLargeGridFP32ArithmeticCannotRestore)
{
    const PrefixFingerprintParts parts{.model = 1, .tokenizer = 2, .runtime = 3,
        .topology = 4, .hybrid = 5, .moe = 6, .mtp = 7};
    constexpr uint64_t legacy_large_grid_key = 0x82d08c6e05dd0887ULL;
    EXPECT_NE(combinePrefixFingerprintParts(parts), legacy_large_grid_key);
}

/** The restored V3 arithmetic namespace must miss after installing CUDA FP16 prefill. */
TEST(Test__PrefixCacheFingerprint, LegacyRestoredFP32ArithmeticCannotRestore)
{
    const PrefixFingerprintParts parts{.model = 1, .tokenizer = 2, .runtime = 3,
        .topology = 4, .hybrid = 5, .moe = 6, .mtp = 7};
    constexpr uint64_t legacy_restored_fp32_key = 0xe660dbd5cb4ab743ULL;
    EXPECT_NE(combinePrefixFingerprintParts(parts), legacy_restored_fp32_key);
}

/** The unvalidated checkpoint namespace cannot authenticate this scoped recovery. */
TEST(Test__PrefixCacheFingerprint, LegacyUnvalidatedCheckpointArithmeticCannotRestore)
{
    const PrefixFingerprintParts parts{.model = 1, .tokenizer = 2, .runtime = 3,
        .topology = 4, .hybrid = 5, .moe = 6, .mtp = 7};
    constexpr uint64_t legacy_checkpoint_key = 0xd8c9af0cd92e2c30ULL;
    EXPECT_NE(combinePrefixFingerprintParts(parts), legacy_checkpoint_key);
}

/** The current arithmetic generation retains the scoped attention recovery fence. */
TEST(Test__PrefixCacheFingerprint, ScopedAttentionRecoveryArithmeticIsRepeatable)
{
    const PrefixFingerprintParts parts{.model = 1, .tokenizer = 2, .runtime = 3,
        .topology = 4, .hybrid = 5, .moe = 6, .mtp = 7};
    constexpr uint64_t recovery_key = 0x5cc5df15e248ed72ULL;
    EXPECT_EQ(combinePrefixFingerprintParts(parts), recovery_key);
    EXPECT_EQ(combinePrefixFingerprintParts(parts), combinePrefixFingerprintParts(parts));
}


/**
 * @brief Fence lossy-embedding archives while preserving current restart identity.
 *
 * Frozen V9 and current golden metadata keys independently identify the arithmetic
 * generations for the same seven named parts. Reconstructing the parts models
 * a new process reading the same authenticated metadata: no cache payload, model
 * weight or device value is hashed or restored by this device-free regression.
 */
TEST(Test__PrefixCacheFingerprint, NativeEmbeddingRejectsV9AndRestartsStably)
{
    constexpr uint64_t old_lossy_embedding_key = 0x1e2eb47a6b0c05b5ULL;
    constexpr uint64_t native_source_embedding_key = 0x5cc5df15e248ed72ULL;
    const PrefixFingerprintParts original_process{.model = 1, .tokenizer = 2, .runtime = 3,
        .topology = 4, .hybrid = 5, .moe = 6, .mtp = 7};
    const uint64_t persisted_current_key = combinePrefixFingerprintParts(original_process);
    EXPECT_NE(persisted_current_key, old_lossy_embedding_key);
    ASSERT_EQ(persisted_current_key, native_source_embedding_key);
    // Fresh metadata objects must accept the same generation after every
    // restart; invalidation changes arithmetic identity, never request epochs.
    for (int restart = 0; restart < 8; ++restart)
    {
        const PrefixFingerprintParts restarted_process{.model = 1, .tokenizer = 2, .runtime = 3,
            .topology = 4, .hybrid = 5, .moe = 6, .mtp = 7};
        EXPECT_EQ(combinePrefixFingerprintParts(restarted_process), persisted_current_key);
        EXPECT_NE(combinePrefixFingerprintParts(restarted_process), old_lossy_embedding_key);
    }
}

/**
 * @brief Reject old GDN norm-clamp state while accepting same-equation restarts.
 *
 * Frozen V10 and current metadata keys use identical model/topology parts, isolating
 * the arithmetic generation. No model or prefix payload bytes are examined.
 */
TEST(Test__PrefixCacheFingerprint, GDNAdditiveEpsilonRejectsV10AndRestartsStably)
{
    constexpr uint64_t old_gdn_norm_clamp_key = 0xab525267a6098d45ULL;
    constexpr uint64_t additive_epsilon_key = 0x5cc5df15e248ed72ULL;
    for (int restart = 0; restart < 8; ++restart)
    {
        const PrefixFingerprintParts parts{.model = 1, .tokenizer = 2, .runtime = 3,
            .topology = 4, .hybrid = 5, .moe = 6, .mtp = 7};
        const uint64_t key = combinePrefixFingerprintParts(parts);
        EXPECT_NE(key, old_gdn_norm_clamp_key);
        EXPECT_EQ(key, additive_epsilon_key);
    }
}

/**
 * @brief Reject invalid-scale arithmetic while accepting identical restarts.
 *
 * V11 could quantize nonzero activations through a zero or infinite half scale.
 * The V12 metadata identity fences all resulting KV, recurrent and terminal
 * state without reading or hashing model weights or cached payloads.
 */
TEST(Test__PrefixCacheFingerprint, FiniteQ8ScaleRejectsV11AndRestartsStably)
{
    constexpr uint64_t old_unbounded_q8_scale_key = 0xfc4266874f559540ULL;
    constexpr uint64_t finite_q8_scale_key = 0x5cc5df15e248ed72ULL;
    for (int restart = 0; restart < 8; ++restart)
    {
        const PrefixFingerprintParts parts{.model = 1, .tokenizer = 2, .runtime = 3,
            .topology = 4, .hybrid = 5, .moe = 6, .mtp = 7};
        const uint64_t key = combinePrefixFingerprintParts(parts);
        EXPECT_NE(key, old_unbounded_q8_scale_key);
        EXPECT_EQ(key, finite_q8_scale_key);
    }
}

/**
 * @brief Reject previous shared-gate/residual arithmetic and retain stable restarts.
 *
 * V12 could flush CUDA shared-gate products/sums or contract a full fused gate
 * differently from its separate product/add path. ROCm residual scalar tails
 * could also omit elements. Frozen old and current metadata keys isolate the
 * arithmetic generation without reading model weights or prefix-cache blobs.
 */
TEST(Test__PrefixCacheFingerprint, ResidualArithmeticRejectsV12AndRestartsStably)
{
    constexpr uint64_t old_residual_arithmetic_key = 0xda62a3544f706d9bULL;
    constexpr uint64_t current_residual_arithmetic_key = 0x5cc5df15e248ed72ULL;
    for (int restart = 0; restart < 8; ++restart)
    {
        const PrefixFingerprintParts parts{.model = 1, .tokenizer = 2, .runtime = 3,
            .topology = 4, .hybrid = 5, .moe = 6, .mtp = 7};
        const uint64_t value = combinePrefixFingerprintParts(parts);
        EXPECT_NE(value, old_residual_arithmetic_key);
        EXPECT_EQ(value, current_residual_arithmetic_key);
    }
}

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

/** @brief Persistent pre-fix cache arithmetic must not alias current execution. */
TEST(Test__PrefixCacheFingerprint, LegacyChunkMeanArithmeticCannotRestore)
{
    const PrefixFingerprintParts parts{
        .model = 11, .tokenizer = 22, .runtime = 33, .topology = 44,
        .hybrid = 55, .moe = 66, .mtp = 77};
    // Golden digest of these exact parts under the old prefix-cache-v1
    // FNV wire identity. Those archives may contain chunk-mean AQ8 bases.
    constexpr uint64_t legacy_key = 544840603072160241ull;
    EXPECT_NE(combinePrefixFingerprintParts(parts), legacy_key);
}

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

/** @brief Key invalidation and returned provenance share exactly one epoch. */
TEST(Test__PrefixCacheFingerprint, EpochAndKeyComeFromOneObservation)
{
    const auto material = makeMaterial();
    const auto admitted = buildPrefixCacheFingerprint(
        material, true, PrefixCacheMoEPolicy::InvalidateOnRebalance, 7);
    const auto published = buildPrefixCacheFingerprint(
        material, true, PrefixCacheMoEPolicy::InvalidateOnRebalance, 8);
    EXPECT_EQ(admitted.placement_epoch, 7u);
    EXPECT_EQ(published.placement_epoch, 8u);
    EXPECT_NE(admitted.key, published.key);

    const auto compatible_before = buildPrefixCacheFingerprint(
        material, true, PrefixCacheMoEPolicy::PlacementFingerprint, 7);
    const auto compatible_after = buildPrefixCacheFingerprint(
        material, true, PrefixCacheMoEPolicy::PlacementFingerprint, 8);
    EXPECT_EQ(compatible_before.key, compatible_after.key);
    EXPECT_EQ(compatible_before.placement_epoch, 7u);
    EXPECT_EQ(compatible_after.placement_epoch, 8u);
}

/** @brief Callers cannot supply a second contradictory movement observation. */
TEST(Test__PrefixCacheFingerprint, RejectsParallelMovementEpochMaterial)
{
    auto material = makeMaterial();
    material.moe.push_back({"runtime_movement_epoch", "7"});
    EXPECT_THROW(buildPrefixCacheFingerprint(
        material, true, PrefixCacheMoEPolicy::InvalidateOnRebalance, 8),
        std::invalid_argument);
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

/**
 * @brief Old rounded-Q6 state misses while identical source-factor restarts hit.
 *
 * Only the fixed metadata tuple is examined. A version change must never
 * trigger a model/prefix payload hash or silently accept incompatible KV state.
 */
TEST(Test__PrefixCacheFingerprint, OriginalQ6ScaleRejectsV13AndRestartsStably)
{
    constexpr uint64_t old_rounded_q6_key = 0xe8f93d2b09cca08eULL;
    constexpr uint64_t source_factor_key = 0x5cc5df15e248ed72ULL;
    for (int restart = 0; restart < 8; ++restart)
    {
        const PrefixFingerprintParts parts{.model = 1, .tokenizer = 2, .runtime = 3,
            .topology = 4, .hybrid = 5, .moe = 6, .mtp = 7};
        EXPECT_EQ(combinePrefixFingerprintParts(parts), source_factor_key);
        EXPECT_NE(combinePrefixFingerprintParts(parts), old_rounded_q6_key);
    }
}

/** @brief Old derived-half IQ2_S state misses while current metadata restarts stably. */
TEST(Test__PrefixCacheFingerprint, OriginalIQ2SScaleRejectsV14AndRestartsStably)
{
    constexpr uint64_t old_rounded_iq2s_key = 0x6d9a77d4f2a0d599ULL;
    constexpr uint64_t source_factor_key = 0x5cc5df15e248ed72ULL;
    for (int restart = 0; restart < 8; ++restart)
    {
        const PrefixFingerprintParts parts{.model = 1, .tokenizer = 2, .runtime = 3,
            .topology = 4, .hybrid = 5, .moe = 6, .mtp = 7};
        EXPECT_EQ(combinePrefixFingerprintParts(parts), source_factor_key);
        EXPECT_NE(combinePrefixFingerprintParts(parts), old_rounded_iq2s_key);
    }
}

/** @brief IQ4_XS factor preservation fences V15 while retaining exact restart identity. */
TEST(Test__PrefixCacheFingerprint, OriginalIQ4XSScaleRejectsV15AndRestartsStably)
{
    constexpr uint64_t old_derived_half_key = 0x7850c4d9fdfa2764ULL;
    constexpr uint64_t source_tuple_key = 0x5cc5df15e248ed72ULL;
    for (int restart = 0; restart < 8; ++restart)
    {
        const PrefixFingerprintParts parts{.model = 1, .tokenizer = 2, .runtime = 3,
            .topology = 4, .hybrid = 5, .moe = 6, .mtp = 7};
        EXPECT_EQ(combinePrefixFingerprintParts(parts), source_tuple_key);
        EXPECT_NE(combinePrefixFingerprintParts(parts), old_derived_half_key);
    }
}

/** @brief IQ3 original factors fence V16 state before payload access. */
TEST(Test__PrefixCacheFingerprint, OriginalIQ3SScaleRejectsV16AndRestartsStably)
{
    constexpr uint64_t old_derived_half_key=0x8573b84bbc243a1fULL;
    constexpr uint64_t source_tuple_key=0x5cc5df15e248ed72ULL;
    for(int restart=0;restart<8;++restart) {
        const PrefixFingerprintParts parts{.model=1,.tokenizer=2,.runtime=3,
            .topology=4,.hybrid=5,.moe=6,.mtp=7};
        EXPECT_EQ(combinePrefixFingerprintParts(parts),source_tuple_key);
        EXPECT_NE(combinePrefixFingerprintParts(parts),old_derived_half_key);
    }
}

/** @test Independent HC/GDN choices cannot restore each other's prefix state; defaults retain old bytes. */
TEST(Test__PrefixCacheFingerprint, FlashNextGDNOutputActivationSeparatesRestarts)
{
    const auto single = QuantizedActivationOperandPrecision::SingleQ8;
    const auto two = QuantizedActivationOperandPrecision::TwoTermQ8;
    std::set<uint64_t> keys;
    for (const auto hc : {single, two})
    {
        std::vector<PrefixFingerprintField> historical;
        appendQwen4ExpArithmeticFingerprintFields(historical, hc);
        for (const auto gdn : {single, two})
        {
            std::vector<PrefixFingerprintField> fields;
            appendQwen4ExpArithmeticFingerprintFields(fields, hc, gdn);
            const auto key = hashPrefixFingerprintFields("runtime", fields);
            EXPECT_TRUE(keys.insert(key).second);
            if (gdn == single) EXPECT_EQ(key, hashPrefixFingerprintFields("runtime", historical));
            std::vector<PrefixFingerprintField> replay;
            appendQwen4ExpArithmeticFingerprintFields(replay, hc, gdn);
            EXPECT_EQ(key, hashPrefixFingerprintFields("runtime", replay));
        }
    }
    EXPECT_EQ(keys.size(), 4u);
    for (const auto name : {"qfn_gdn_output_activation", "qfn_gdn_output_activation_revision"})
    {
        std::vector<PrefixFingerprintField> fields{{name, "original"}};
        EXPECT_THROW(appendQwen4ExpArithmeticFingerprintFields(fields, single, two), std::invalid_argument);
        ASSERT_EQ(fields.size(), 1u);
        EXPECT_EQ(fields.front().value, "original");
    }
    std::vector<PrefixFingerprintField> fields;
    EXPECT_THROW(appendQwen4ExpArithmeticFingerprintFields(fields, single,
        static_cast<QuantizedActivationOperandPrecision>(255)), std::invalid_argument);
    EXPECT_TRUE(fields.empty());
}

/** @test Independent FA correction separates all eight arithmetic identities without changing old default keys. */
TEST(Test__PrefixCacheFingerprint, FlashNextFAOutputActivationSeparatesRestarts)
{
    constexpr auto single=QuantizedActivationOperandPrecision::SingleQ8;
    constexpr auto two=QuantizedActivationOperandPrecision::TwoTermQ8;
    std::set<uint64_t> keys;
    for(const auto hc:{single,two})for(const auto gdn:{single,two})
    {
        std::vector<PrefixFingerprintField> historical;
        appendQwen4ExpArithmeticFingerprintFields(historical,hc,gdn);
        for(const auto fa:{single,two})
        {
            std::vector<PrefixFingerprintField> fields,restart;
            appendQwen4ExpArithmeticFingerprintFields(fields,hc,gdn,fa);
            const auto key=hashPrefixFingerprintFields("runtime",fields);
            EXPECT_TRUE(keys.insert(key).second);
            appendQwen4ExpArithmeticFingerprintFields(restart,hc,gdn,fa);
            EXPECT_EQ(key,hashPrefixFingerprintFields("runtime",restart));
            if(fa==single){EXPECT_EQ(key,hashPrefixFingerprintFields("runtime",historical));}
        }
    }
    EXPECT_EQ(keys.size(),8u);
    for(const auto name:{"qfn_fa_output_activation","qfn_fa_output_activation_revision"})
    {
        std::vector<PrefixFingerprintField> fields{{name,"foreign"}};
        EXPECT_THROW(appendQwen4ExpArithmeticFingerprintFields(fields,single,single,two),std::invalid_argument);
        ASSERT_EQ(fields.size(),1u);
    }
    std::vector<PrefixFingerprintField> fields;
    EXPECT_THROW(appendQwen4ExpArithmeticFingerprintFields(fields,single,single,
        static_cast<QuantizedActivationOperandPrecision>(255)),std::invalid_argument);
    EXPECT_TRUE(fields.empty());
}

/** @test All sixteen independent QFN arithmetic identities retain default compatibility and distinct restart keys. */
TEST(Test__PrefixCacheFingerprint, FlashNextGDNQKVActivationSeparatesRestarts)
{
    constexpr auto single = QuantizedActivationOperandPrecision::SingleQ8;
    constexpr auto two = QuantizedActivationOperandPrecision::TwoTermQ8;
    std::set<uint64_t> keys;
    for (const auto hc : {single, two})
    for (const auto gdn : {single, two})
    for (const auto fa : {single, two})
    {
        std::vector<PrefixFingerprintField> historical;
        appendQwen4ExpArithmeticFingerprintFields(historical, hc, gdn, fa);
        for (const auto qkv : {single, two})
        {
            std::vector<PrefixFingerprintField> fields, restart;
            appendQwen4ExpArithmeticFingerprintFields(fields, hc, gdn, fa, qkv);
            const auto key = hashPrefixFingerprintFields("runtime", fields);
            EXPECT_TRUE(keys.insert(key).second);
            appendQwen4ExpArithmeticFingerprintFields(restart, hc, gdn, fa, qkv);
            EXPECT_EQ(key, hashPrefixFingerprintFields("runtime", restart));
            if (qkv == single)
            {
                EXPECT_EQ(key, hashPrefixFingerprintFields("runtime", historical));
            }
        }
    }
    EXPECT_EQ(keys.size(), 16u);
    for (const auto name : {"qfn_gdn_qkv_activation", "qfn_gdn_qkv_activation_revision"})
    {
        std::vector<PrefixFingerprintField> fields{{name, "foreign"}};
        EXPECT_THROW(appendQwen4ExpArithmeticFingerprintFields(fields, single, single, single, two), std::invalid_argument);
        ASSERT_EQ(fields.size(), 1u);
    }
    std::vector<PrefixFingerprintField> fields;
    EXPECT_THROW(appendQwen4ExpArithmeticFingerprintFields(fields, single, single, single,
        static_cast<QuantizedActivationOperandPrecision>(255)), std::invalid_argument);
    EXPECT_TRUE(fields.empty());
}

/** @brief Persistent state uses the concrete source policy, preserving explicit native isolation. */
TEST(Test__PrefixCacheFingerprint, QFNRouterDefaultResolvesBeforeIdentity)
{
    const auto identity = [](MoERouterArithmetic requested, std::string_view architecture) {
        std::vector<PrefixFingerprintField> fields;
        appendProjectionArithmeticFingerprintFields(fields, kDefaultGDNProjectionArithmetic,
            resolveMoERouterArithmetic(requested, architecture));
        return hashPrefixFingerprintFields("runtime", fields);
    };
    const auto qfn = identity(MoERouterArithmetic::ModelDefault, "qwen4exp");
    EXPECT_EQ(qfn, identity(MoERouterArithmetic::FP16, "qwen4exp"));
    EXPECT_NE(qfn, identity(MoERouterArithmetic::Native, "qwen4exp"));
    EXPECT_EQ(identity(MoERouterArithmetic::ModelDefault, "qwen35moe"),
              identity(MoERouterArithmetic::Native, "qwen35moe"));
    std::vector<PrefixFingerprintField> unresolved;
    EXPECT_THROW(appendProjectionArithmeticFingerprintFields(unresolved,
        kDefaultGDNProjectionArithmetic, MoERouterArithmetic::ModelDefault), std::invalid_argument);
    EXPECT_TRUE(unresolved.empty());
}
