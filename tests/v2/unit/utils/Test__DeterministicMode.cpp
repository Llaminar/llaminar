#include <gtest/gtest.h>

#include "kernels/cuda/moe/CUDAMoEBatchInvariantPolicy.h"
#include "kernels/cuda/moe/CUDAMoERouterPrefillPolicy.h"
#include "kernels/cuda/gemm/CUDAMoEGroupedPrefillKernels.h"
#include "utils/DebugEnv.h"

#include <array>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

using namespace llaminar2;

namespace
{
    class ScopedEnv
    {
    public:
        explicit ScopedEnv(std::initializer_list<std::pair<const char *, const char *>> values)
        {
            for (const auto &entry : values)
            {
                Entry saved;
                saved.name = entry.first;
                if (const char *old = std::getenv(entry.first))
                {
                    saved.had_old = true;
                    saved.old_value = old;
                }
                saved_.push_back(std::move(saved));
                setenv(entry.first, entry.second, 1);
            }
            mutableDebugEnv().reload();
        }

        ~ScopedEnv()
        {
            for (auto it = saved_.rbegin(); it != saved_.rend(); ++it)
            {
                if (it->had_old)
                    setenv(it->name.c_str(), it->old_value.c_str(), 1);
                else
                    unsetenv(it->name.c_str());
            }
            mutableDebugEnv().reload();
        }

        ScopedEnv(const ScopedEnv &) = delete;
        ScopedEnv &operator=(const ScopedEnv &) = delete;

    private:
        struct Entry
        {
            std::string name;
            bool had_old = false;
            std::string old_value;
        };

        std::vector<Entry> saved_;
    };
}

/**
 * @brief Lock in CUDA MoE arithmetic while retaining geometry controls.
 *
 * Empty environment values exercise the same invalid-override branch as an
 * unset variable while keeping the process-global test environment reversible.
 * The normal runtime must retain the geometry whose stochastic production
 * token stream is byte-stable; a faster candidate is promoted only after that
 * stronger gate passes.
 */
TEST(Test__DeterministicMode, CudaMoEArithmeticPolicyIsFixedAndGeometryDefaultsAreExplicit)
{
    ScopedEnv env({
        {"LLAMINAR_CUDA_MOE_GATEUP_ORDERED_KPART_TILE_N", ""},
        {"LLAMINAR_CUDA_MOE_DOWN_ORDERED_KPART_TILE_N", ""},
    });

    EXPECT_EQ(CUDAMoEBatchInvariantPolicy::gate_up_k_partitions, 16);
    EXPECT_EQ(CUDAMoEBatchInvariantPolicy::down_k_partitions, 16);
    EXPECT_EQ(debugEnv().gemm.cuda_moe_gateup_ordered_kpart_tile_n, 128);
    EXPECT_FALSE(
        debugEnv().gemm.cuda_moe_gateup_ordered_kpart_tile_n_override_active);
    EXPECT_EQ(debugEnv().gemm.cuda_moe_down_ordered_kpart_tile_n, 128);
}

/**
 * @brief Accept only warp-aligned ordered split-K scatter geometries.
 *
 * The launch width is selected before graph capture and becomes part of the
 * executable topology. Invalid values must resolve to the proven production
 * geometry instead of creating a partial warp or an oversized CUDA block.
 */
TEST(Test__DeterministicMode, CudaMoEOrderedKPartTileRequiresWarpAlignedGeometry)
{
    {
        ScopedEnv env({
            {"LLAMINAR_CUDA_MOE_GATEUP_ORDERED_KPART_TILE_N", "192"},
            {"LLAMINAR_CUDA_MOE_DOWN_ORDERED_KPART_TILE_N", "224"},
        });
        EXPECT_EQ(debugEnv().gemm.cuda_moe_gateup_ordered_kpart_tile_n, 192);
        EXPECT_TRUE(
            debugEnv().gemm.cuda_moe_gateup_ordered_kpart_tile_n_override_active);
        EXPECT_EQ(debugEnv().gemm.cuda_moe_down_ordered_kpart_tile_n, 224);
    }
    {
        ScopedEnv env({
            {"LLAMINAR_CUDA_MOE_GATEUP_ORDERED_KPART_TILE_N", "190"},
            {"LLAMINAR_CUDA_MOE_DOWN_ORDERED_KPART_TILE_N", "33"},
        });
        EXPECT_EQ(debugEnv().gemm.cuda_moe_gateup_ordered_kpart_tile_n, 128);
        EXPECT_FALSE(
            debugEnv().gemm.cuda_moe_gateup_ordered_kpart_tile_n_override_active);
        EXPECT_EQ(debugEnv().gemm.cuda_moe_down_ordered_kpart_tile_n, 128);
    }
}

/**
 * @brief Assign exactly one direct-down warp to every valid router slot.
 *
 * Extra warps are idle and strictly dominated; fewer warps serialize routes.
 * The typed arithmetic policy therefore derives geometry from top-k instead
 * of accepting a mutable process-wide override.
 */
TEST(Test__DeterministicMode, CudaMoEDownDirectWarpGeometryMatchesTopK)
{
    EXPECT_EQ(CUDAMoEBatchInvariantPolicy::directDownWarps(1), 1);
    EXPECT_EQ(CUDAMoEBatchInvariantPolicy::directDownWarps(4), 4);
    EXPECT_EQ(CUDAMoEBatchInvariantPolicy::directDownWarps(8), 8);
    EXPECT_EQ(CUDAMoEBatchInvariantPolicy::directDownWarps(16), 16);
}

/**
 * @brief Prove CUDA router-prefill dispatch is total and geometry-specific.
 *
 * Known Qwen3.6 35B graph buckets must resolve to their measured exact overlay.
 * Positive unseen M values and unrelated model geometries must still resolve to
 * a complete compiled specification rather than failing dispatch.
 */
TEST(Test__DeterministicMode, CudaMoERouterPrefillGeometryIsMTotal)
{
    using Geometry = CUDAMoERouterPrefillGeometry;
    struct ExpectedSelection
    {
        int seq_len;
        Geometry geometry;
    };
    constexpr std::array expected{
        ExpectedSelection{1, Geometry::Tile16x16},
        ExpectedSelection{64, Geometry::Tile16x16},
        ExpectedSelection{128, Geometry::Tile16x16},
        ExpectedSelection{129, Geometry::Tile32x32},
        ExpectedSelection{256, Geometry::Tile32x32},
        ExpectedSelection{257, Geometry::Tile24x32},
        ExpectedSelection{384, Geometry::Tile24x32},
        ExpectedSelection{385, Geometry::Tile32x32},
        ExpectedSelection{640, Geometry::Tile32x32},
        ExpectedSelection{641, Geometry::Tile32x24},
        ExpectedSelection{704, Geometry::Tile32x24},
        ExpectedSelection{705, Geometry::Tile32x32},
        ExpectedSelection{768, Geometry::Tile32x32},
        ExpectedSelection{769, Geometry::Tile32x64},
        ExpectedSelection{1536, Geometry::Tile32x64},
        ExpectedSelection{1537, Geometry::Tile64x64},
        ExpectedSelection{4096, Geometry::Tile64x64},
        ExpectedSelection{1 << 20, Geometry::Tile64x64},
    };

    for (const ExpectedSelection &selection : expected)
    {
        const Geometry actual = selectCUDAMoERouterPrefillGeometry(
            selection.seq_len,
            /*d_model=*/2048,
            /*num_experts=*/256);
        EXPECT_EQ(actual, selection.geometry)
            << "M=" << selection.seq_len;
        const auto spec = cudaMoERouterPrefillGeometrySpec(actual);
        EXPECT_GT(spec.tile_rows, 0);
        EXPECT_GT(spec.tile_experts, 0);
        EXPECT_EQ(spec.threads_per_block, 256);
    }

    for (const int seq_len : {1, 64, 512, 4096, 1 << 20})
    {
        EXPECT_EQ(
            selectCUDAMoERouterPrefillGeometry(
                seq_len,
                /*d_model=*/4096,
                /*num_experts=*/128),
            Geometry::Tile64x64)
            << "generic geometry must remain total at M=" << seq_len;
    }
}

/**
 * @brief Lock the measured Qwen grouped-IMMA boundary and generic totality.
 *
 * The selector executes at graph-capture time and must always return a compiled
 * geometry. Every real Qwen3.6 UD-IQ3_S routed tuple, plus the separately
 * measured IQ2_S/IQ3_S tuple, pins its contiguous 32-to-64-column regime.
 * Unseen formats and geometries retain the paired schedule and conservative
 * 32-column geometry without leaving an M dispatch hole.
 */
TEST(Test__DeterministicMode, CudaMoEGroupedImmaPolicyIsMTotal)
{
    using llaminar2::cuda::moe::GroupedImmaColumns;
    using llaminar2::cuda::moe::GroupedImmaGateUpSchedule;
    using llaminar2::cuda::moe::selectGroupedImmaLaunchPolicy;

    struct ExactRegime
    {
        int gate_up_codebook;
        int down_codebook;
        int last_columns32_rows;
        bool columns32_for_all_rows;
    };
    constexpr std::array<ExactRegime, 4> exact_regimes{{
        {13, 4, 768, false},
        {13, 8, 600, false},
        {11, 8, 0, true},
        {13, 11, 768, false},
    }};
    constexpr std::array<int, 8> exact_rows{
        1, 64, 600, 601, 768, 769, 4096, 1 << 20};
    for (const auto &regime : exact_regimes)
    {
        for (const int rows : exact_rows)
        {
            const auto policy = selectGroupedImmaLaunchPolicy(
                regime.gate_up_codebook,
                regime.down_codebook,
                rows,
                /*hidden_size=*/2048,
                /*expert_width=*/512,
                /*expert_count=*/256,
                /*top_k=*/8);
            EXPECT_EQ(
                policy.gate_up_columns,
                regime.columns32_for_all_rows ||
                        rows <= regime.last_columns32_rows
                    ? GroupedImmaColumns::Columns32
                    : GroupedImmaColumns::Columns64)
                << "gate/up codebook=" << regime.gate_up_codebook
                << " down codebook=" << regime.down_codebook
                << " M=" << rows;
            EXPECT_EQ(policy.down_columns, GroupedImmaColumns::Columns32);
            EXPECT_EQ(
                policy.gate_up_schedule,
                GroupedImmaGateUpSchedule::PairedProjections);
            EXPECT_TRUE(policy.exact_overlay);
        }
    }

    for (const int rows : {1, 17, 769, 4096, 1 << 20})
    {
        const auto generic = selectGroupedImmaLaunchPolicy(
            /*gateup_codebook=*/-1,
            /*down_codebook=*/-1,
            rows,
            /*hidden_size=*/3584,
            /*expert_width=*/1024,
            /*expert_count=*/128,
            /*top_k=*/4);
        EXPECT_EQ(generic.gate_up_columns, GroupedImmaColumns::Columns32);
        EXPECT_EQ(generic.down_columns, GroupedImmaColumns::Columns32);
        EXPECT_EQ(
            generic.gate_up_schedule,
            GroupedImmaGateUpSchedule::PairedProjections);
        EXPECT_FALSE(generic.exact_overlay);
    }
}

TEST(Test__DeterministicMode, DebugEnvDisablesNondeterministicRoutes)
{
    ScopedEnv env({
        {"LLAMINAR_DETERMINISTIC", "1"},
        {"LLAMINAR_CUDA_CONCURRENT_PREFILL", "1"},
        {"LLAMINAR_CUDA_CONCURRENT_DECODE", "1"},
        {"LLAMINAR_CUDA_MOE_ROUTER_Q8", "1"},
        {"LLAMINAR_CUDA_MOE_REUSE_ROUTER_Q8_HIDDEN", "1"},
        {"LLAMINAR_ROCM_CONCURRENT_PREFILL", "1"},
        {"LLAMINAR_ROCM_CONCURRENT_DECODE", "1"},
        {"LLAMINAR_ROCM_GDN_CONCURRENT_DECODE", "1"},
        {"LLAMINAR_ROCM_MOE_ROUTER_Q8", "1"},
        {"LLAMINAR_ROCM_MOE_ROUTER_FP16", "1"},
        {"LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "1"},
        {"LLAMINAR_ROCM_MOE_ROUTER_WAVE_TOPK", "1"},
    });

    const auto &env_snapshot = debugEnv();
    EXPECT_TRUE(env_snapshot.gemm.deterministic);

    EXPECT_FALSE(env_snapshot.gemm.cuda_concurrent_prefill);
    EXPECT_FALSE(env_snapshot.gemm.cuda_concurrent_decode);
    EXPECT_FALSE(env_snapshot.gemm.cuda_moe_router_q8);
    EXPECT_FALSE(env_snapshot.gemm.cuda_moe_reuse_router_q8_hidden);

    EXPECT_FALSE(env_snapshot.rocm.concurrent_prefill);
    EXPECT_FALSE(env_snapshot.rocm.concurrent_decode);
    EXPECT_FALSE(env_snapshot.rocm.gdn_concurrent_decode);
    EXPECT_FALSE(env_snapshot.rocm.moe_router_q8);
    EXPECT_FALSE(env_snapshot.rocm.moe_router_fp16);
    EXPECT_FALSE(env_snapshot.rocm.moe_router_kpart_decode);
    EXPECT_FALSE(env_snapshot.rocm.moe_router_wave_topk);
}


TEST(Test__DeterministicMode, ConcurrentRoutesReturnToDefaultsWhenDeterminismIsCleared)
{
    {
        ScopedEnv env({
            {"LLAMINAR_DETERMINISTIC", "1"},
            {"LLAMINAR_CUDA_CONCURRENT_PREFILL", "1"},
            {"LLAMINAR_CUDA_CONCURRENT_DECODE", "1"},
            {"LLAMINAR_CUDA_MOE_ROUTER_Q8", "1"},
            {"LLAMINAR_CUDA_MOE_REUSE_ROUTER_Q8_HIDDEN", "1"},
            {"LLAMINAR_ROCM_CONCURRENT_PREFILL", "1"},
            {"LLAMINAR_ROCM_CONCURRENT_DECODE", "1"},
        });
        EXPECT_FALSE(debugEnv().gemm.cuda_concurrent_prefill);
        EXPECT_FALSE(debugEnv().gemm.cuda_concurrent_decode);
        EXPECT_FALSE(debugEnv().gemm.cuda_moe_router_q8);
        EXPECT_FALSE(debugEnv().gemm.cuda_moe_reuse_router_q8_hidden);
        EXPECT_FALSE(debugEnv().rocm.concurrent_prefill);
        EXPECT_FALSE(debugEnv().rocm.concurrent_decode);
        EXPECT_FALSE(debugEnv().rocm.gdn_concurrent_decode);
    }

    EXPECT_FALSE(debugEnv().gemm.deterministic);
    EXPECT_TRUE(debugEnv().gemm.cuda_concurrent_prefill);
    EXPECT_TRUE(debugEnv().gemm.cuda_concurrent_decode);
    EXPECT_TRUE(debugEnv().gemm.cuda_moe_router_q8);
    EXPECT_TRUE(debugEnv().gemm.cuda_moe_reuse_router_q8_hidden);
    EXPECT_TRUE(debugEnv().rocm.concurrent_prefill);
    EXPECT_FALSE(debugEnv().rocm.concurrent_decode);
    EXPECT_TRUE(debugEnv().rocm.gdn_concurrent_decode);
}
