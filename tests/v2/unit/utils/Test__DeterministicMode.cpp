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

TEST(Test__DeterministicMode, DebugEnvDisablesNondeterministicCudaAndRocmRoutes)
{
    ScopedEnv env({
        {"LLAMINAR_DETERMINISTIC", "1"},
        {"LLAMINAR_CUDA_CONCURRENT_PREFILL", "1"},
        {"LLAMINAR_CUDA_CONCURRENT_DECODE", "1"},
        {"LLAMINAR_CUDA_MOE_GATEUP_KPART_DECODE", "1"},
        {"LLAMINAR_CUDA_MOE_DOWN_KPART_DECODE", "1"},
        {"LLAMINAR_ROCM_NVNNI_ATOMIC_REDUCE", "1"},
        {"LLAMINAR_ROCM_CONCURRENT_PREFILL", "1"},
        {"LLAMINAR_ROCM_CONCURRENT_DECODE", "1"},
        {"LLAMINAR_ROCM_CONCURRENT_M2_ROWS", "1"},
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
    EXPECT_FALSE(env_snapshot.gemm.cuda_moe_gateup_kpart_decode);
    EXPECT_FALSE(env_snapshot.gemm.cuda_moe_down_kpart_decode);

    EXPECT_FALSE(env_snapshot.rocm.nvnni_atomic_reduce);
    EXPECT_FALSE(env_snapshot.rocm.concurrent_prefill);
    EXPECT_FALSE(env_snapshot.rocm.concurrent_decode);
    EXPECT_FALSE(env_snapshot.rocm.concurrent_m2_rows);
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
            {"LLAMINAR_ROCM_CONCURRENT_PREFILL", "1"},
            {"LLAMINAR_ROCM_CONCURRENT_DECODE", "1"},
        });
        EXPECT_FALSE(debugEnv().gemm.cuda_concurrent_prefill);
        EXPECT_FALSE(debugEnv().gemm.cuda_concurrent_decode);
        EXPECT_FALSE(debugEnv().rocm.concurrent_prefill);
        EXPECT_FALSE(debugEnv().rocm.concurrent_decode);
    }

    EXPECT_FALSE(debugEnv().gemm.deterministic);
    EXPECT_TRUE(debugEnv().gemm.cuda_concurrent_prefill);
    EXPECT_TRUE(debugEnv().gemm.cuda_concurrent_decode);
    EXPECT_TRUE(debugEnv().rocm.concurrent_prefill);
    EXPECT_FALSE(debugEnv().rocm.concurrent_decode);
}
