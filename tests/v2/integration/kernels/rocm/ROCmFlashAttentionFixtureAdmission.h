/**
 * @file ROCmFlashAttentionFixtureAdmission.h
 * @brief Production rank-plan admission for the attention fixture's real source graphs.
 *
 * This fixture declares its actual graph interval, terminal role and selected
 * diagnostic outputs. The production rank-input builder and MemoryPlanner own
 * every storage estimate; the certificate and retained PMA alone admit/claim
 * allocations. No mock override, anonymous reserve or alternate live ledger is
 * permitted. The same pure admission function supports device-free replay.
 */
#pragma once
#include "planning/RankMemoryPlanInputs.h"
#include "planning/ActivationBufferSizing.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "snapshots/SnapshotStorageLayout.h"
#include "snapshots/SnapshotCapture.h"
#include "execution/compute_stages/stages/FusedQKVGEMMStage.h"
#include "execution/compute_stages/stages/RoPEStage.h"
#include "tensors/Tensors.h"
#include <algorithm>
#include <limits>
#include <optional>

namespace llaminar2::test::rocm_attention_fixture
{
    /** @return Exact semantic selection shared by admission and source capture. */
    inline std::vector<std::string> snapshotFilter()
    {
        return {"layer3_Q_ROPE", "layer3_K_ROPE", "layer3_V_PROJECTION"};
    }

    /**
     * @brief Price the actual stage descriptors selected by the source's semantic filter.
     * @param profile Exact source model geometry, with its original layer count.
     * @param rows Actual admitted maximum activation rows of the GPU graph.
     * @return Canonical layouts of distinct selected temporal producer outputs.
     *
     * Empty FP32 schema tensors carry only native type: no backing, allocation,
     * backend admission or execution is attempted. Logical shape comes from the
     * actual stage params, exactly as getDumpInfoSnapshot reports it. In
     * particular, stage-wide QKV aliases currently select Q/K as well as V;
     * semantic key count is not a physical descriptor count. Projection Q/K and
     * later in-place RoPE Q/K are distinct snapshots even though they alias live
     * tensor storage. Production owns descriptor emission, selection and layout.
     */
    inline std::vector<SnapshotStorageLayout> snapshotLayouts(const ModelMemoryProfile &profile, int rows)
    {
        if (rows <= 0 || profile.n_heads <= 0 || profile.n_kv_heads <= 0 || profile.head_dim <= 0)
            throw std::invalid_argument("Attention source snapshots require positive native geometry");
        const auto query = snapshotPopulationProduct(profile.n_heads, profile.head_dim);
        const auto kv = snapshotPopulationProduct(profile.n_kv_heads, profile.head_dim);
        if (query > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            kv > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("Attention snapshot stage columns exceed native parameter range");
        FP32Tensor q({0, 0}), k({0, 0}), v({0, 0});
        FusedQKVGEMMStage projection({.m = rows, .k = profile.d_model,
            .output_q = &q, .n_q = static_cast<int>(query),
            .output_k = &k, .n_k = static_cast<int>(kv),
            .output_v = &v, .n_v = static_cast<int>(kv)});
        RoPEStage rope({.Q = &q, .K = &k, .n_heads = profile.n_heads,
            .n_kv_heads = profile.n_kv_heads, .head_dim = profile.head_dim, .seq_len = rows});
        const auto filter = snapshotFilter();
        std::vector<SnapshotStorageLayout> layouts;
        const auto append = [&](const char *stage_name, const IComputeStage &stage) {
            const auto dump = stage.getDumpInfoSnapshot();
            for (const auto &output : dump.outputs)
            {
                StageDumpInfo view;
                view.outputs.push_back(output);
                const auto keys = SnapshotCapture::possibleKeysForStage(stage_name, view);
                if (std::none_of(keys.begin(), keys.end(), [&](const auto &key) {
                    return std::find(filter.begin(), filter.end(), key) != filter.end();
                })) continue;
                if (!std::holds_alternative<std::monostate>(output.semantic_extent))
                    throw std::logic_error("Attention source requires admission for newly authored snapshot geometry");
                layouts.push_back(snapshotStorageLayout({output.byte_size, std::nullopt}));
            }
        };
        append("layer3_qkv_proj", projection);
        append("layer3_rope", rope);
        return layouts;
    }

    /** @return Descriptor-derived wide/scalar payloads; production resolves serving owners. */
    inline GraphSnapshotMemoryCapacity snapshotCapacity(const ModelMemoryProfile &profile, int rows)
    {
        const auto payload = [&](int row_count) {
            std::size_t end = 0;
            for (const auto &layout : snapshotLayouts(profile, row_count))
                end = placeSnapshotStorageSlot(end, layout).end_bytes;
            return end;
        };
        return {.main_phase_payload = MainSnapshotPhasePayload{
            .wide_main_bytes = payload(rows),
            .scalar_main_bytes = payload(1)}};
    }

    /**
     * @brief Admit the exact source graph through the ordinary production rank BOM.
     * @param profile Original model metadata/native tensor inventory, never a fabricated layer count.
     * @param plan Actual fragment interval, terminal ownership and runtime configuration.
     * @param inventory Actual rank observation; replay supplies explicit synthetic observations.
     * @param upload Exact production upload-ring geometry, absent for CPU graphs.
     * @param prefill_buckets Original configured capture ladder, absent for CPU graphs.
     * @return Sole rank-bound authority installed before any factory materialization.
     */
    inline std::shared_ptr<PhysicalMemoryAuthority> admit(
        const ModelMemoryProfile &profile, const RankExecutionPlan &plan,
        const RankInventory &inventory, std::optional<GPUWeightLoadMemoryGeometry> upload = {},
        std::optional<std::vector<int>> prefill_buckets = {})
    {
        const int last = plan.last_layer < 0 ? profile.mainLayerCount() - 1 : plan.last_layer;
        if (plan.has_lm_head != (last == profile.mainLayerCount() - 1))
            throw std::invalid_argument("Attention source terminal ownership must match the original model boundary");
        const auto device = plan.primary_device.toLocalDeviceId();
        const auto snapshots = device.is_gpu()
            ? std::optional(snapshotCapacity(profile, resolveActivationBufferSeqLen(plan.runtime.max_seq_len, device)))
            : std::nullopt;
        const auto inputs = buildRankMemoryPlanInputs({
            .model = profile, .plan = plan, .inventory = inventory,
            .weight_load_geometry = upload, .snapshot_capacity = snapshots,
            .captured_prefill_buckets = std::move(prefill_buckets)});
        const auto memory = MemoryPlanner::plan(profile, inputs);
        return std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(memory.admit()), plan.rank);
    }
}
