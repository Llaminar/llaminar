#include <gtest/gtest.h>

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
 * @brief Lock in the production-model-proven CUDA MoE partition geometry.
 *
 * Empty environment values exercise the same invalid-override branch as an
 * unset variable while keeping the process-global test environment reversible.
 * The normal runtime must retain the geometry whose stochastic production
 * token stream is byte-stable; a faster candidate is promoted only after that
 * stronger gate passes.
 */
TEST(Test__DeterministicMode, CudaMoEOrderedKPartDefaultsUseProductionProvenGeometry)
{
    ScopedEnv env({
        {"LLAMINAR_CUDA_MOE_GATEUP_KPARTS", ""},
        {"LLAMINAR_CUDA_MOE_DOWN_KPARTS", ""},
        {"LLAMINAR_CUDA_MOE_ORDERED_KPART_TILE_N", ""},
        {"LLAMINAR_CUDA_MOE_DOWN_DIRECT_WARPS", ""},
    });

    EXPECT_EQ(debugEnv().gemm.cuda_moe_gateup_kparts, 16);
    EXPECT_EQ(debugEnv().gemm.cuda_moe_down_kparts, 16);
    EXPECT_EQ(debugEnv().gemm.cuda_moe_ordered_kpart_tile_n, 128);
    EXPECT_EQ(debugEnv().gemm.cuda_moe_down_direct_warps, 9);
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
        ScopedEnv env({{"LLAMINAR_CUDA_MOE_ORDERED_KPART_TILE_N", "192"}});
        EXPECT_EQ(debugEnv().gemm.cuda_moe_ordered_kpart_tile_n, 192);
    }
    {
        ScopedEnv env({{"LLAMINAR_CUDA_MOE_ORDERED_KPART_TILE_N", "190"}});
        EXPECT_EQ(debugEnv().gemm.cuda_moe_ordered_kpart_tile_n, 128);
    }
}

/**
 * @brief Bound scratch-free grouped-down launches to legal whole-warp blocks.
 *
 * The kernel derives route ownership from the captured block width. Keeping
 * this override between eight and sixteen warps guarantees enough workers for
 * the production top-k range without exceeding CUDA's 512-thread block size.
 */
TEST(Test__DeterministicMode, CudaMoEDownDirectWarpGeometryIsBounded)
{
    {
        ScopedEnv env({{"LLAMINAR_CUDA_MOE_DOWN_DIRECT_WARPS", "11"}});
        EXPECT_EQ(debugEnv().gemm.cuda_moe_down_direct_warps, 11);
    }
    {
        ScopedEnv env({{"LLAMINAR_CUDA_MOE_DOWN_DIRECT_WARPS", "7"}});
        EXPECT_EQ(debugEnv().gemm.cuda_moe_down_direct_warps, 9);
    }
}

TEST(Test__DeterministicMode, DebugEnvDisablesNondeterministicRoutesAndPreservesOrderedCudaKPart)
{
    ScopedEnv env({
        {"LLAMINAR_DETERMINISTIC", "1"},
        {"LLAMINAR_CUDA_CONCURRENT_PREFILL", "1"},
        {"LLAMINAR_CUDA_CONCURRENT_DECODE", "1"},
        {"LLAMINAR_CUDA_MOE_GATEUP_KPARTS", "8"},
        {"LLAMINAR_CUDA_MOE_DOWN_KPARTS", "4"},
        {"LLAMINAR_CUDA_MOE_ROUTER_Q8", "1"},
        {"LLAMINAR_CUDA_MOE_REUSE_ROUTER_Q8_HIDDEN", "1"},
        {"LLAMINAR_ROCM_NVNNI_ATOMIC_REDUCE", "1"},
        {"LLAMINAR_ROCM_CONCURRENT_PREFILL", "1"},
        {"LLAMINAR_ROCM_CONCURRENT_DECODE", "1"},
        {"LLAMINAR_ROCM_GDN_CONCURRENT_DECODE", "1"},
        {"LLAMINAR_ROCM_MOE_ROUTER_Q8", "1"},
        {"LLAMINAR_ROCM_MOE_ROUTER_FP16", "1"},
        {"LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "1"},
        {"LLAMINAR_ROCM_MOE_ROUTER_WAVE_TOPK", "1"},
        {"LLAMINAR_ROCM_MOE_PARALLEL_DOWN_DECODE", "1"},
        {"LLAMINAR_ROCM_MOE_GATEUP_KPART_DECODE", "1"},
    });

    const auto &env_snapshot = debugEnv();
    EXPECT_TRUE(env_snapshot.gemm.deterministic);

    EXPECT_FALSE(env_snapshot.gemm.cuda_concurrent_prefill);
    EXPECT_FALSE(env_snapshot.gemm.cuda_concurrent_decode);
    EXPECT_EQ(env_snapshot.gemm.cuda_moe_gateup_kparts, 8);
    EXPECT_EQ(env_snapshot.gemm.cuda_moe_down_kparts, 4);
    EXPECT_FALSE(env_snapshot.gemm.cuda_moe_router_q8);
    EXPECT_FALSE(env_snapshot.gemm.cuda_moe_reuse_router_q8_hidden);

    EXPECT_FALSE(env_snapshot.rocm.nvnni_atomic_reduce);
    EXPECT_FALSE(env_snapshot.rocm.concurrent_prefill);
    EXPECT_FALSE(env_snapshot.rocm.concurrent_decode);
    EXPECT_FALSE(env_snapshot.rocm.gdn_concurrent_decode);
    EXPECT_FALSE(env_snapshot.rocm.moe_router_q8);
    EXPECT_FALSE(env_snapshot.rocm.moe_router_fp16);
    EXPECT_FALSE(env_snapshot.rocm.moe_router_kpart_decode);
    EXPECT_FALSE(env_snapshot.rocm.moe_router_wave_topk);
    EXPECT_FALSE(env_snapshot.rocm.moe_parallel_down_decode);
    EXPECT_FALSE(env_snapshot.rocm.moe_gateup_kpart_decode);
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
