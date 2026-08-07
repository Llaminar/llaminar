#include <gtest/gtest.h>

#include "kernels/cuda/moe/CUDAMoEBatchInvariantPolicy.h"
#include "utils/DebugEnv.h"

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
