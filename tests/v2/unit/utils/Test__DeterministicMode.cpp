/**
 * @file Test__DeterministicMode.cpp
 * @brief Device-free regression of immutable arithmetic and launch policies.
 *
 * Policy selection runs before capture. These tests retain measured launch
 * boundaries across unseen row counts and distinguish physical format identity
 * from shared arithmetic schedules without executing a GPU kernel.
 */

#include <gtest/gtest.h>

#include "kernels/cuda/moe/CUDARouterQ8Prefill.h"
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
                if (entry.second)
                    setenv(entry.first, entry.second, 1);
                else
                    unsetenv(entry.first);
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

/** @brief HC-up correction is explicit, strict, and independent of other arithmetic. */
TEST(Test__DeterministicMode, FlashNextHCUpActivationPolicyIsExplicit)
{
    ScopedEnv defaults({{"LLAMINAR_QFN_HC_UP_ACTIVATION", nullptr}});
    EXPECT_EQ(GemmConfig{}.qfn_hc_up_activation, QuantizedActivationOperandPrecision::SingleQ8);
    const auto gdn = debugEnv().gemm.gdn_projection_arithmetic;
    const auto router = debugEnv().gemm.moe_router_arithmetic;
    for (const auto policy : {QuantizedActivationOperandPrecision::SingleQ8,
                              QuantizedActivationOperandPrecision::TwoTermQ8})
    {
        const auto name = quantizedActivationOperandPrecisionName(policy);
        EXPECT_EQ(parseQuantizedActivationOperandPrecision(name), policy);
        ScopedEnv selected({{"LLAMINAR_QFN_HC_UP_ACTIVATION", name.data()}});
        EXPECT_EQ(GemmConfig{}.qfn_hc_up_activation, policy);
        EXPECT_EQ(debugEnv().gemm.qfn_hc_up_activation, policy);
        EXPECT_EQ(debugEnv().gemm.gdn_projection_arithmetic, gdn);
        EXPECT_EQ(debugEnv().gemm.moe_router_arithmetic, router);
    }
    EXPECT_EQ(debugEnv().gemm.qfn_hc_up_activation, QuantizedActivationOperandPrecision::SingleQ8);
    for (const auto invalid : {"", "auto", "native", "fp16", "q8", "two_q8", "TWO_TERM_Q8", "two_term_q8 "})
        EXPECT_THROW(parseQuantizedActivationOperandPrecision(invalid), std::invalid_argument);
    EXPECT_THROW(quantizedActivationOperandPrecisionName(
        static_cast<QuantizedActivationOperandPrecision>(255)), std::invalid_argument);
}

/** @brief The qualified alpha/beta default preserves independent router and wide-projection policy. */
TEST(Test__DeterministicMode, ProjectionArithmeticDefaultsAndExplicitOverrides)
{
    ScopedEnv defaults({{"LLAMINAR_GDN_PROJECTION_ARITHMETIC", nullptr},
                        {"LLAMINAR_MOE_ROUTER_ARITHMETIC", nullptr},
                        {"LLAMINAR_CUDA_MOE_ROUTER_PREFILL", nullptr},
                        {"LLAMINAR_CUDA_MOE_ROUTER_Q8", nullptr}});
    EXPECT_EQ(kDefaultGDNProjectionArithmetic, GDNProjectionArithmetic::FP16MoEAlphaBeta);
    EXPECT_EQ(GemmConfig{}.gdn_projection_arithmetic, GDNProjectionArithmetic::FP16MoEAlphaBeta);
    EXPECT_EQ(debugEnv().gemm.gdn_projection_arithmetic, GDNProjectionArithmetic::FP16MoEAlphaBeta);
    EXPECT_EQ(GemmConfig{}.moe_router_arithmetic, MoERouterArithmetic::ModelDefault);
    EXPECT_EQ(GemmConfig{}.cuda_moe_router_prefill, CUDARouterPrefillArithmetic::FP32);
    EXPECT_TRUE(GemmConfig{}.cuda_moe_router_q8);
    {
        ScopedEnv alpha_beta({{"LLAMINAR_GDN_PROJECTION_ARITHMETIC", "fp16_alpha_beta"}});
        EXPECT_EQ(GemmConfig{}.gdn_projection_arithmetic, GDNProjectionArithmetic::FP16AlphaBeta);
        EXPECT_EQ(GemmConfig{}.moe_router_arithmetic, MoERouterArithmetic::ModelDefault);
    }
    {
        ScopedEnv moe_alpha_beta({{"LLAMINAR_GDN_PROJECTION_ARITHMETIC", "fp16_moe_alpha_beta"}});
        EXPECT_EQ(GemmConfig{}.gdn_projection_arithmetic, GDNProjectionArithmetic::FP16MoEAlphaBeta);
        EXPECT_EQ(debugEnv().gemm.gdn_projection_arithmetic, GDNProjectionArithmetic::FP16MoEAlphaBeta);
    }
    {
        ScopedEnv native({{"LLAMINAR_GDN_PROJECTION_ARITHMETIC", "native"}});
        EXPECT_EQ(GemmConfig{}.gdn_projection_arithmetic, GDNProjectionArithmetic::Native);
        EXPECT_EQ(debugEnv().gemm.gdn_projection_arithmetic, GDNProjectionArithmetic::Native);
        EXPECT_EQ(GemmConfig{}.moe_router_arithmetic, MoERouterArithmetic::ModelDefault);
    }
    EXPECT_EQ(debugEnv().gemm.gdn_projection_arithmetic, GDNProjectionArithmetic::FP16MoEAlphaBeta);
    constexpr cuda::RouterQ8PrefillHardware device{0,82,16};
    EXPECT_EQ(cuda::routerQ8PrefillGeometry(512,2048,256,{}).warps,0);
    for (int rows=1; rows<=8192; ++rows)
        for (int experts : {1,17,32,65,128,257,512,1024})
            for (int width : {32,96,512,2048,3072,8192,12288})
            {
                const auto tile = cuda::routerQ8PrefillGeometry(rows,width,experts,device);
                ASSERT_TRUE(tile.warps==1 || tile.warps==2 || tile.warps==4 || tile.warps==8 || tile.warps==16);
                ASSERT_EQ(tile.warps,(tile.rows/16)*(tile.experts/8)*tile.k_warps);
                ASSERT_TRUE(tile.k_warps==1 || (tile.rows==16 && tile.experts==8 && width/32>=tile.k_warps));
                ASSERT_EQ(tile.constant_width,width<=8192 && (width&(width-1))==0 ? width : 0);
            }
    // The same M selects different parallelism when expert width or device size
    // changes. Transformed matrices with the same aligned tile grid agree.
    EXPECT_EQ(cuda::routerQ8PrefillGeometry(512,2048,256,device).warps,1);
    EXPECT_EQ(cuda::routerQ8PrefillGeometry(512,2048,512,device).warps,2);
    EXPECT_EQ(cuda::routerQ8PrefillGeometry(512,2048,1024,device).warps,4);
    EXPECT_GT(cuda::routerQ8PrefillGeometry(512,2048,64,device).k_warps,1);
    EXPECT_EQ(cuda::routerQ8PrefillGeometry(1024,2048,256,device).warps,
              cuda::routerQ8PrefillGeometry(512,2048,512,device).warps);
    EXPECT_EQ(cuda::routerQ8PrefillGeometry(512,2048,256,{0,41,16}).warps,2);
    EXPECT_EQ(cuda::routerQ8PrefillGeometry(512,2048,256,{0,164,16}).k_warps,2);
    EXPECT_EQ(cuda::routerQ8PrefillGeometry(513,2048,256,device).warps,
              cuda::routerQ8PrefillGeometry(512,2048,256,device).warps);
    EXPECT_EQ(parseCUDARouterPrefillArithmetic("q8"), CUDARouterPrefillArithmetic::Q8);
    for (const auto invalid : {"", "auto", "Q8", "fp16"})
        EXPECT_THROW(parseCUDARouterPrefillArithmetic(invalid), std::invalid_argument);
    {
        ScopedEnv prefill({{"LLAMINAR_CUDA_MOE_ROUTER_PREFILL", "q8"}});
        EXPECT_EQ(debugEnv().gemm.cuda_moe_router_prefill, CUDARouterPrefillArithmetic::Q8);
        EXPECT_TRUE(debugEnv().gemm.cuda_moe_router_q8);
    }
    EXPECT_EQ(parseMoERouterArithmetic("native"), MoERouterArithmetic::Native);
    EXPECT_EQ(parseMoERouterArithmetic("fp16"), MoERouterArithmetic::FP16);
    EXPECT_EQ(parseMoERouterArithmetic("native_hadamard32"), MoERouterArithmetic::NativeHadamard32);
    EXPECT_EQ(moeRouterArithmeticName(MoERouterArithmetic::NativeHadamard32), "native_hadamard32");
    EXPECT_EQ(routerQ8Basis(MoERouterArithmetic::NativeHadamard32), RouterQ8Basis::SignedHadamard32);
    EXPECT_EQ(routerQ8Basis(MoERouterArithmetic::FP16), RouterQ8Basis::Original);
    for (const auto invalid : {"", "auto", "FP16", "fp32", "q8"})
        EXPECT_THROW(parseMoERouterArithmetic(invalid), std::invalid_argument);
    ScopedEnv env({{"LLAMINAR_MOE_ROUTER_ARITHMETIC", "fp16"},
                   {"LLAMINAR_GDN_PROJECTION_ARITHMETIC", "fp16"}});
    EXPECT_EQ(debugEnv().gemm.gdn_projection_arithmetic, GDNProjectionArithmetic::FP16);
    EXPECT_EQ(debugEnv().gemm.moe_router_arithmetic, MoERouterArithmetic::FP16);
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
        ExpectedSelection{65, Geometry::Tile32x16},
        ExpectedSelection{128, Geometry::Tile32x16},
        ExpectedSelection{129, Geometry::Tile32x32},
        ExpectedSelection{256, Geometry::Tile32x32},
        ExpectedSelection{257, Geometry::Tile64x32},
        ExpectedSelection{384, Geometry::Tile64x32},
        ExpectedSelection{385, Geometry::Tile32x32K16},
        ExpectedSelection{640, Geometry::Tile32x32K16},
        ExpectedSelection{641, Geometry::Tile64x32},
        ExpectedSelection{1536, Geometry::Tile64x32},
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
 * Both physical IQ4 identities retain that same measured source4 boundary;
 * adopting source factors must not demote physical4 to an unmeasured regime.
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
    constexpr std::array<ExactRegime, 5> exact_regimes{{
        {13, 4, 768, false},
        {13, llaminar2::kNativeVnniIQ4XSSourceScaleCodebook, 768, false},
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

TEST(Test__DeterministicMode, DebugEnvRetainsArithmeticInvariantCudaDecodeConcurrency)
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
        {"LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE", "1"},
        {"LLAMINAR_ROCM_MOE_ROUTER_WAVE_TOPK", "1"},
        {"LLAMINAR_ROCM_MOE_REUSE_ROUTER_Q8_HIDDEN", "1"},
    });

    const auto &env_snapshot = debugEnv();
    EXPECT_TRUE(env_snapshot.gemm.deterministic);

    EXPECT_FALSE(env_snapshot.gemm.cuda_concurrent_prefill);
    EXPECT_TRUE(env_snapshot.gemm.cuda_concurrent_decode);
    EXPECT_TRUE(env_snapshot.gemm.cuda_moe_router_q8);
    EXPECT_TRUE(env_snapshot.gemm.cuda_moe_reuse_router_q8_hidden);

    EXPECT_FALSE(env_snapshot.rocm.concurrent_prefill);
    EXPECT_FALSE(env_snapshot.rocm.concurrent_decode);
    EXPECT_FALSE(env_snapshot.rocm.gdn_concurrent_decode);
    EXPECT_TRUE(env_snapshot.rocm.moe_router_q8);
    EXPECT_FALSE(env_snapshot.rocm.moe_router_kpart_decode);
    EXPECT_FALSE(env_snapshot.rocm.moe_router_wave_topk);
    EXPECT_TRUE(env_snapshot.rocm.moe_reuse_router_q8_hidden);
}

/**
 * @brief Deterministic Q8 reuse remains explicitly configurable per backend.
 *
 * Q8 publication is now part of the deterministic optimized path, but a
 * caller diagnosing that path can still select standalone quantization. The
 * two controls must move together or reuse would advertise a missing producer.
 */
TEST(Test__DeterministicMode, DeterministicRouterQ8ReuseHonorsExplicitDisable)
{
    ScopedEnv env({
        {"LLAMINAR_DETERMINISTIC", "1"},
        {"LLAMINAR_CUDA_MOE_ROUTER_Q8", "0"},
        {"LLAMINAR_CUDA_MOE_REUSE_ROUTER_Q8_HIDDEN", "0"},
        {"LLAMINAR_ROCM_MOE_ROUTER_Q8", "0"},
        {"LLAMINAR_ROCM_MOE_REUSE_ROUTER_Q8_HIDDEN", "0"},
    });

    EXPECT_FALSE(debugEnv().gemm.cuda_moe_router_q8);
    EXPECT_FALSE(debugEnv().gemm.cuda_moe_reuse_router_q8_hidden);
    EXPECT_FALSE(debugEnv().rocm.moe_router_q8);
    EXPECT_FALSE(debugEnv().rocm.moe_reuse_router_q8_hidden);
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
        EXPECT_TRUE(debugEnv().gemm.cuda_concurrent_decode);
        EXPECT_TRUE(debugEnv().gemm.cuda_moe_router_q8);
        EXPECT_TRUE(debugEnv().gemm.cuda_moe_reuse_router_q8_hidden);
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

/** @test GDN-output correction is explicit and does not change other projection policies. */
TEST(Test__DeterministicMode, FlashNextGDNOutputActivationPolicyIsExplicit)
{
    ScopedEnv defaults({{"LLAMINAR_QFN_GDN_OUTPUT_ACTIVATION", nullptr},
                        {"LLAMINAR_QFN_HC_UP_ACTIVATION", "single_q8"}});
    const auto gdn = debugEnv().gemm.gdn_projection_arithmetic;
    const auto router = debugEnv().gemm.moe_router_arithmetic;
    EXPECT_EQ(GemmConfig{}.qfn_gdn_output_activation, QuantizedActivationOperandPrecision::SingleQ8);
    for (const auto policy : {QuantizedActivationOperandPrecision::SingleQ8,
                              QuantizedActivationOperandPrecision::TwoTermQ8})
    {
        const auto name = quantizedActivationOperandPrecisionName(policy);
        ScopedEnv selected({{"LLAMINAR_QFN_GDN_OUTPUT_ACTIVATION", name.data()}});
        EXPECT_EQ(GemmConfig{}.qfn_gdn_output_activation, policy);
        EXPECT_EQ(debugEnv().gemm.qfn_gdn_output_activation, policy);
        EXPECT_EQ(debugEnv().gemm.qfn_hc_up_activation, QuantizedActivationOperandPrecision::SingleQ8);
        EXPECT_EQ(debugEnv().gemm.gdn_projection_arithmetic, gdn);
        EXPECT_EQ(debugEnv().gemm.moe_router_arithmetic, router);
    }
    for (const auto invalid : {"", "auto", "native", "fp16", "q8", "two_q8", "TWO_TERM_Q8"})
    {
        // Avoid a partially reloaded singleton on failure: instantiate the same
        // strict parsed setup record independently, then restore the environment.
        const char *previous = std::getenv("LLAMINAR_QFN_GDN_OUTPUT_ACTIVATION");
        const std::string saved = previous ? previous : "";
        const bool had_previous = previous != nullptr;
        setenv("LLAMINAR_QFN_GDN_OUTPUT_ACTIVATION", invalid, 1);
        EXPECT_THROW(GemmConfig{}, std::invalid_argument);
        if (had_previous) setenv("LLAMINAR_QFN_GDN_OUTPUT_ACTIVATION", saved.c_str(), 1);
        else unsetenv("LLAMINAR_QFN_GDN_OUTPUT_ACTIVATION");
    }
}

/** @test FA-output correction is explicit and does not change other projection policies. */
TEST(Test__DeterministicMode, FlashNextFAOutputActivationPolicyIsExplicit)
{
    ScopedEnv defaults({{"LLAMINAR_QFN_FA_OUTPUT_ACTIVATION", nullptr},
                        {"LLAMINAR_QFN_HC_UP_ACTIVATION", "single_q8"}});
    const auto gdn = debugEnv().gemm.gdn_projection_arithmetic;
    const auto router = debugEnv().gemm.moe_router_arithmetic;
    EXPECT_EQ(GemmConfig{}.qfn_fa_output_activation, QuantizedActivationOperandPrecision::SingleQ8);
    for (const auto policy : {QuantizedActivationOperandPrecision::SingleQ8,
                              QuantizedActivationOperandPrecision::TwoTermQ8})
    {
        const auto name = quantizedActivationOperandPrecisionName(policy);
        ScopedEnv selected({{"LLAMINAR_QFN_FA_OUTPUT_ACTIVATION", name.data()}});
        EXPECT_EQ(GemmConfig{}.qfn_fa_output_activation, policy);
        EXPECT_EQ(debugEnv().gemm.qfn_fa_output_activation, policy);
        EXPECT_EQ(debugEnv().gemm.qfn_hc_up_activation, QuantizedActivationOperandPrecision::SingleQ8);
        EXPECT_EQ(debugEnv().gemm.gdn_projection_arithmetic, gdn);
        EXPECT_EQ(debugEnv().gemm.moe_router_arithmetic, router);
    }
    for (const auto invalid : {"", "auto", "native", "fp16", "q8", "two_q8", "TWO_TERM_Q8"})
    {
        // Avoid a partially reloaded singleton on failure: instantiate the same
        // strict parsed setup record independently, then restore the environment.
        const char *previous = std::getenv("LLAMINAR_QFN_FA_OUTPUT_ACTIVATION");
        const std::string saved = previous ? previous : "";
        const bool had_previous = previous != nullptr;
        setenv("LLAMINAR_QFN_FA_OUTPUT_ACTIVATION", invalid, 1);
        EXPECT_THROW(GemmConfig{}, std::invalid_argument);
        if (had_previous) setenv("LLAMINAR_QFN_FA_OUTPUT_ACTIVATION", saved.c_str(), 1);
        else unsetenv("LLAMINAR_QFN_FA_OUTPUT_ACTIVATION");
    }
}


/** @test Original-main GDN QKV is an independent strict setup policy, with unchanged default/other arithmetic. */
TEST(Test__DeterministicMode, FlashNextGDNQKVActivationPolicyIsExplicit)
{
    ScopedEnv defaults({{"LLAMINAR_QFN_GDN_QKV_ACTIVATION", nullptr},
                        {"LLAMINAR_QFN_HC_UP_ACTIVATION", "single_q8"}});
    const auto gdn = debugEnv().gemm.gdn_projection_arithmetic;
    const auto router = debugEnv().gemm.moe_router_arithmetic;
    const auto output = debugEnv().gemm.qfn_gdn_output_activation;
    const auto fa = debugEnv().gemm.qfn_fa_output_activation;
    EXPECT_EQ(GemmConfig{}.qfn_gdn_qkv_activation, QuantizedActivationOperandPrecision::SingleQ8);
    for (const auto policy : {QuantizedActivationOperandPrecision::SingleQ8,
                              QuantizedActivationOperandPrecision::TwoTermQ8})
    {
        const auto name = quantizedActivationOperandPrecisionName(policy);
        ScopedEnv selected({{"LLAMINAR_QFN_GDN_QKV_ACTIVATION", name.data()}});
        EXPECT_EQ(GemmConfig{}.qfn_gdn_qkv_activation, policy);
        EXPECT_EQ(debugEnv().gemm.qfn_gdn_qkv_activation, policy);
        EXPECT_EQ(debugEnv().gemm.qfn_hc_up_activation, QuantizedActivationOperandPrecision::SingleQ8);
        EXPECT_EQ(debugEnv().gemm.gdn_projection_arithmetic, gdn);
        EXPECT_EQ(debugEnv().gemm.moe_router_arithmetic, router);
        EXPECT_EQ(debugEnv().gemm.qfn_gdn_output_activation, output);
        EXPECT_EQ(debugEnv().gemm.qfn_fa_output_activation, fa);
    }
    for (const auto invalid : {"", "auto", "native", "fp16", "q8", "two_q8", "TWO_TERM_Q8"})
    {
        // Avoid a partially reloaded singleton on failure: instantiate the same
        // strict parsed setup record independently, then restore the environment.
        const char *previous = std::getenv("LLAMINAR_QFN_GDN_QKV_ACTIVATION");
        const std::string saved = previous ? previous : "";
        const bool had_previous = previous != nullptr;
        setenv("LLAMINAR_QFN_GDN_QKV_ACTIVATION", invalid, 1);
        EXPECT_THROW(GemmConfig{}, std::invalid_argument);
        if (had_previous) setenv("LLAMINAR_QFN_GDN_QKV_ACTIVATION", saved.c_str(), 1);
        else unsetenv("LLAMINAR_QFN_GDN_QKV_ACTIVATION");
    }
}

/** @brief Omitted QFN routing resolves once to half; explicit diagnostics and other schemas retain their policies. */
TEST(Test__DeterministicMode, QFNRouterDefaultAndExplicitOverrides)
{
    ScopedEnv defaults({{"LLAMINAR_MOE_ROUTER_ARITHMETIC", nullptr}});
    const auto omitted = GemmConfig{}.moe_router_arithmetic;
    ASSERT_EQ(omitted, MoERouterArithmetic::ModelDefault);
    EXPECT_EQ(routerQ8Basis(omitted), RouterQ8Basis::Original);
    EXPECT_EQ(routerQ8Basis(MoERouterArithmetic::Native), RouterQ8Basis::Original);
    EXPECT_EQ(routerQ8Basis(MoERouterArithmetic::FP16), RouterQ8Basis::Original);
    EXPECT_EQ(resolveMoERouterArithmetic(omitted, "qwen4exp"), MoERouterArithmetic::FP16);
    for (const auto architecture : {"qwen35", "qwen35moe", "llama", "qwen4", "Qwen4Exp", ""})
    {
        EXPECT_EQ(resolveMoERouterArithmetic(omitted, architecture), MoERouterArithmetic::Native);
    }
    for (const auto value : {"native", "fp16", "native_hadamard32"})
    {
        ScopedEnv explicit_choice({{"LLAMINAR_MOE_ROUTER_ARITHMETIC", value}});
        const auto requested = parseMoERouterArithmetic(value);
        EXPECT_EQ(GemmConfig{}.moe_router_arithmetic, requested);
        EXPECT_EQ(resolveMoERouterArithmetic(requested, "qwen4exp"), requested);
        EXPECT_EQ(resolveMoERouterArithmetic(requested, "qwen35moe"), requested);
    }
    EXPECT_THROW(moeRouterArithmeticName(omitted), std::invalid_argument);
    EXPECT_THROW(resolveMoERouterArithmetic(static_cast<MoERouterArithmetic>(255), "qwen4exp"),
                 std::invalid_argument);
    EXPECT_EQ(kDefaultGDNProjectionArithmetic, GDNProjectionArithmetic::FP16MoEAlphaBeta);
}
