/**
 * @file Perf__ROCmPrefillDispatchComparison.cpp
 * @brief A/B benchmark: legacy CK prefill dispatch vs new native prefill dispatch
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "fort.hpp"

#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "tensors/Tensors.h"
#include "../../../utils/TestTensorFactory.h"
#include "utils/DebugEnv.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;

namespace
{
    struct ShapeCase
    {
        std::string category;
        std::string name;
        int M;
        int N;
        int K;
        int warmup_iters;
        int bench_iters;
    };

    struct PathBenchStats
    {
        double mean_ms = 0.0;
        double min_ms = 0.0;
        double max_ms = 0.0;
        double tflops = 0.0;
        std::vector<float> output;
    };

    class ScopedEnvOverride
    {
    public:
        ScopedEnvOverride(const char *name, const std::string &value)
            : name_(name)
        {
            const char *existing = std::getenv(name_);
            had_original_ = (existing != nullptr);
            if (had_original_)
            {
                original_value_ = existing;
            }

            ::setenv(name_, value.c_str(), 1);
            mutableDebugEnv().reload();
        }

        ~ScopedEnvOverride()
        {
            if (had_original_)
            {
                ::setenv(name_, original_value_.c_str(), 1);
            }
            else
            {
                ::unsetenv(name_);
            }
            mutableDebugEnv().reload();
        }

        ScopedEnvOverride(const ScopedEnvOverride &) = delete;
        ScopedEnvOverride &operator=(const ScopedEnvOverride &) = delete;

    private:
        const char *name_;
        bool had_original_ = false;
        std::string original_value_;
    };

    double cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size() || a.empty())
        {
            return 0.0;
        }

        double dot = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            const double av = static_cast<double>(a[i]);
            const double bv = static_cast<double>(b[i]);
            dot += av * bv;
            norm_a += av * av;
            norm_b += bv * bv;
        }

        if (norm_a <= 0.0 || norm_b <= 0.0)
        {
            return 0.0;
        }

        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-12);
    }

    class ROCmPrefillDispatchComparisonPerf : public ::testing::Test
    {
    protected:
        bool has_rocm_device_ = false;
        std::string device_name_;

        void SetUp() override
        {
#ifdef HAVE_ROCM
            int device_count = 0;
            const hipError_t err = hipGetDeviceCount(&device_count);
            has_rocm_device_ = (err == hipSuccess && device_count > 0);
            if (has_rocm_device_)
            {
                hipDeviceProp_t props{};
                (void)hipGetDeviceProperties(&props, 0);
                device_name_ = std::string(props.name) + " (" + props.gcnArchName + ")";
            }
#endif
        }

        PathBenchStats runPathBenchmark(const ShapeCase &shape, bool use_new_dispatch)
        {
            return runPathBenchmarkVariant(shape, use_new_dispatch, false, 8);
        }

        // Extended variant: control wide-tile V3/V7 and KT selection
        PathBenchStats runPathBenchmarkVariant(const ShapeCase &shape, bool use_new_dispatch,
                                               bool use_v3, int kt,
                                               bool use_v7 = false)
        {
            PathBenchStats stats;

            auto weights = TestTensorFactory::createQ8_0Random({static_cast<size_t>(shape.N), static_cast<size_t>(shape.K)});
            ROCmPackedWeights packed;
            EXPECT_TRUE(packWeightsToROCm(weights.get(), packed));

            ROCmQuantisedGemmKernel kernel(&packed, 0);

            auto workspace = std::make_unique<DeviceWorkspaceManager>(DeviceId::rocm(0), 1024ULL * 1024ULL * 1024ULL);
            const auto reqs = kernel.getWorkspaceRequirements(shape.M, shape.N, shape.K);
            EXPECT_TRUE(workspace->allocate(reqs));
            kernel.bindWorkspace(workspace.get());

            auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(shape.M), static_cast<size_t>(shape.K)});
            auto output = TestTensorFactory::createFP32({static_cast<size_t>(shape.M), static_cast<size_t>(shape.N)});

            // Native prefill is always enabled; use_new_dispatch only affects reporting.
            ScopedEnvOverride grid_kpar("LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR", "0");
            ScopedEnvOverride grid_splits("LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR_SPLITS", "0");
            ScopedEnvOverride grid_kb("LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR_KB", "0");
            ScopedEnvOverride prefill_cpt("LLAMINAR_ROCM_VNNI_PREFILL_CPT", "1");
            ScopedEnvOverride prefill_variant("LLAMINAR_ROCM_VNNI_PREFILL_VARIANT", "-1");
            ScopedEnvOverride prefill_grid_variant("LLAMINAR_ROCM_VNNI_PREFILL_GRID_VARIANT", "-1");
            ScopedEnvOverride wide_v3("LLAMINAR_ROCM_WIDE_TILE_V3", use_v3 ? "1" : "0");
            ScopedEnvOverride wide_v7("LLAMINAR_ROCM_WIDE_TILE_V7", use_v7 ? "1" : "0");
            ScopedEnvOverride wide_kt("LLAMINAR_ROCM_WIDE_TILE_KT", std::to_string(kt));

            for (int i = 0; i < shape.warmup_iters; ++i)
            {
                EXPECT_TRUE(kernel.multiply_tensor(input.get(), output.get(), shape.M, shape.N, shape.K));
            }
            (void)hipDeviceSynchronize();

            std::vector<double> times_ms;
            times_ms.reserve(static_cast<size_t>(shape.bench_iters));

            for (int i = 0; i < shape.bench_iters; ++i)
            {
                const auto t0 = std::chrono::high_resolution_clock::now();
                EXPECT_TRUE(kernel.multiply_tensor(input.get(), output.get(), shape.M, shape.N, shape.K));
                (void)hipDeviceSynchronize();
                const auto t1 = std::chrono::high_resolution_clock::now();
                const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                times_ms.push_back(ms);
            }

            EXPECT_FALSE(times_ms.empty());
            stats.mean_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / static_cast<double>(times_ms.size());
            stats.min_ms = *std::min_element(times_ms.begin(), times_ms.end());
            stats.max_ms = *std::max_element(times_ms.begin(), times_ms.end());

            const double ops = 2.0 * static_cast<double>(shape.M) * static_cast<double>(shape.N) * static_cast<double>(shape.K);
            stats.tflops = (ops / (stats.mean_ms * 1e-3)) / 1e12;

            const size_t out_size = static_cast<size_t>(shape.M) * static_cast<size_t>(shape.N);
            stats.output.assign(output->data(), output->data() + out_size);
            return stats;
        }
    };

    TEST_F(ROCmPrefillDispatchComparisonPerf, LegacyCKVsNewNativeDispatch_ByShapeClass)
    {
        if (!has_rocm_device_)
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        std::cout << "\n[Perf] ROCm prefill dispatch comparison: legacy CK fallback vs new native dispatch\n";
        std::cout << "[Perf] Device: " << device_name_ << "\n";

        const std::vector<ShapeCase> shapes = {
            {"Attention", "Qwen2.5-0.5B_AttnOut", 128, 896, 896, 1, 2},
            {"FFN_Up", "Qwen2.5-0.5B_FFN_Up", 128, 4864, 896, 1, 2},
            {"FFN_Gate", "Qwen2.5-0.5B_FFN_Gate", 128, 4864, 896, 1, 2},
            {"FFN_Down", "Qwen2.5-0.5B_FFN_Down", 128, 896, 4864, 1, 2},
            {"LM_Head", "Qwen2.5-0.5B_LM_Head", 128, 151936, 896, 1, 1},

            {"Attention", "Qwen2.5-3B_AttnOut", 128, 2048, 2048, 1, 2},
            {"FFN_Up", "Qwen2.5-3B_FFN_Up", 128, 11008, 2048, 1, 2},
            {"FFN_Gate", "Qwen2.5-3B_FFN_Gate", 128, 11008, 2048, 1, 2},
            {"FFN_Down", "Qwen2.5-3B_FFN_Down", 128, 2048, 11008, 1, 2},
            {"LM_Head", "Qwen2.5-3B_LM_Head", 128, 151936, 2048, 1, 1},

            {"Attention", "Qwen2.5-7B_AttnOut", 128, 3584, 3584, 1, 2},
            {"FFN_Up", "Qwen2.5-7B_FFN_Up", 128, 18944, 3584, 1, 2},
            {"FFN_Gate", "Qwen2.5-7B_FFN_Gate", 128, 18944, 3584, 1, 2},
            {"FFN_Down", "Qwen2.5-7B_FFN_Down", 128, 3584, 18944, 1, 2},
            {"LM_Head", "Qwen2.5-7B_LM_Head", 128, 151936, 3584, 1, 1},
        };

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header << "Class" << "Shape" << "M" << "N" << "K" << "LegacyCK(ms)" << "NewPath(ms)" << "Speedup" << "Winner" << "Cosine" << fort::endr;
        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::left);
        table.column(2).set_cell_text_align(fort::text_align::right);
        table.column(3).set_cell_text_align(fort::text_align::right);
        table.column(4).set_cell_text_align(fort::text_align::right);
        table.column(5).set_cell_text_align(fort::text_align::right);
        table.column(6).set_cell_text_align(fort::text_align::right);
        table.column(7).set_cell_text_align(fort::text_align::right);
        table.column(8).set_cell_text_align(fort::text_align::center);
        table.column(9).set_cell_text_align(fort::text_align::right);

        std::map<std::string, std::vector<double>> speedups_by_class;

        for (const auto &shape : shapes)
        {
            const auto legacy = runPathBenchmark(shape, false);
            const auto modern = runPathBenchmark(shape, true);

            const double speedup = (modern.mean_ms > 0.0) ? (legacy.mean_ms / modern.mean_ms) : 0.0;
            const double cos = cosineSimilarity(legacy.output, modern.output);

            const std::string winner = (speedup >= 1.0) ? "new" : "legacy";
            speedups_by_class[shape.category].push_back(speedup);

            table << shape.category
                  << shape.name
                  << shape.M
                  << shape.N
                  << shape.K
                  << std::fixed << std::setprecision(3) << legacy.mean_ms
                  << std::fixed << std::setprecision(3) << modern.mean_ms
                  << std::fixed << std::setprecision(3) << speedup
                  << winner
                  << std::fixed << std::setprecision(6) << cos
                  << fort::endr;

            EXPECT_GT(cos, 0.99) << "Output mismatch for shape " << shape.name;
        }

        std::cout << "\n"
                  << table.to_string() << std::endl;

        fort::utf8_table summary;
        summary.set_border_style(FT_DOUBLE2_STYLE);
        summary << fort::header << "Class" << "AvgSpeedup(Legacy/New)" << "Preferred" << fort::endr;
        summary.column(0).set_cell_text_align(fort::text_align::left);
        summary.column(1).set_cell_text_align(fort::text_align::right);
        summary.column(2).set_cell_text_align(fort::text_align::center);

        for (const auto &entry : speedups_by_class)
        {
            const auto &values = entry.second;
            const double avg_speedup = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
            summary << entry.first
                    << std::fixed << std::setprecision(3) << avg_speedup
                    << ((avg_speedup >= 1.0) ? "new" : "legacy")
                    << fort::endr;
        }

        std::cout << "\n"
                  << summary.to_string() << std::endl;
    }

    // =========================================================================
    // Wide-tile variant comparison: V1/KT8 vs V2/KT8 vs V2/KT16
    // Tests all shapes to find per-shape-class optimal variant.
    // =========================================================================
    TEST_F(ROCmPrefillDispatchComparisonPerf, WideTileVariantComparison)
    {
        if (!has_rocm_device_)
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        std::cout << "\n[Perf] Wide-tile variant comparison: V3/KT4,8,16 vs V7/KT4,8,16\n";
        std::cout << "[Perf] Device: " << device_name_ << "\n\n";

        struct VariantConfig
        {
            const char *name;
            bool v3;
            bool v7;
            int kt;
        };

        const std::vector<VariantConfig> variants = {
            {"V3/KT4", true, false, 4},
            {"V3/KT8", true, false, 8},
            {"V3/KT16", true, false, 16},
            {"V7/KT4", false, true, 4},
            {"V7/KT8", false, true, 8},
            {"V7/KT16", false, true, 16},
        };

        const std::vector<ShapeCase> shapes = {
            {"Attention", "Qwen2.5-0.5B_AttnOut", 128, 896, 896, 1, 2},
            {"FFN_Up", "Qwen2.5-0.5B_FFN_Up", 128, 4864, 896, 1, 2},
            {"FFN_Down", "Qwen2.5-0.5B_FFN_Down", 128, 896, 4864, 1, 2},
            {"LM_Head", "Qwen2.5-0.5B_LM_Head", 128, 151936, 896, 1, 1},

            {"Attention", "Qwen2.5-3B_AttnOut", 128, 2048, 2048, 1, 2},
            {"FFN_Up", "Qwen2.5-3B_FFN_Up", 128, 11008, 2048, 1, 2},
            {"FFN_Down", "Qwen2.5-3B_FFN_Down", 128, 2048, 11008, 1, 2},
            {"LM_Head", "Qwen2.5-3B_LM_Head", 128, 151936, 2048, 1, 1},

            {"Attention", "Qwen2.5-7B_AttnOut", 128, 3584, 3584, 1, 2},
            {"FFN_Up", "Qwen2.5-7B_FFN_Up", 128, 18944, 3584, 1, 2},
            {"FFN_Down", "Qwen2.5-7B_FFN_Down", 128, 3584, 18944, 1, 2},
            {"LM_Head", "Qwen2.5-7B_LM_Head", 128, 151936, 3584, 1, 1},
        };

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header << "Shape" << "N" << "K" << "CK(ms)";
        for (const auto &v : variants)
        {
            table << (std::string(v.name) + "(ms)");
        }
        table << "Best" << "vs CK" << fort::endr;
        table.column(0).set_cell_text_align(fort::text_align::left);
        const int num_cols = 3 + static_cast<int>(variants.size()) + 2; // Shape+N+K + variants + Best+vsCK
        for (int c = 1; c < num_cols; ++c)
        {
            table.column(static_cast<unsigned>(c)).set_cell_text_align(fort::text_align::right);
        }

        for (const auto &shape : shapes)
        {
            // CK baseline
            const auto ck = runPathBenchmarkVariant(shape, false, false, 8);

            // Native variants
            std::vector<PathBenchStats> variant_stats;
            for (const auto &v : variants)
            {
                variant_stats.push_back(runPathBenchmarkVariant(shape, true, v.v3, v.kt, v.v7));
            }

            // Find best variant
            int best_idx = 0;
            for (size_t i = 1; i < variant_stats.size(); ++i)
            {
                if (variant_stats[i].mean_ms < variant_stats[static_cast<size_t>(best_idx)].mean_ms)
                {
                    best_idx = static_cast<int>(i);
                }
            }

            const double vs_ck = (variant_stats[static_cast<size_t>(best_idx)].mean_ms > 0.0)
                                     ? ck.mean_ms / variant_stats[static_cast<size_t>(best_idx)].mean_ms
                                     : 0.0;

            table << shape.name
                  << shape.N
                  << shape.K
                  << std::fixed << std::setprecision(3) << ck.mean_ms;
            for (const auto &vs : variant_stats)
            {
                table << std::fixed << std::setprecision(3) << vs.mean_ms;
            }
            table << variants[static_cast<size_t>(best_idx)].name
                  << std::fixed << std::setprecision(3) << vs_ck
                  << fort::endr;

            // Verify all variants produce correct output
            for (size_t i = 0; i < variant_stats.size(); ++i)
            {
                const double cos = cosineSimilarity(ck.output, variant_stats[i].output);
                EXPECT_GT(cos, 0.99) << "Output mismatch for " << shape.name << " variant " << variants[i].name;
            }
        }

        std::cout << table.to_string() << std::endl;
    }
}
