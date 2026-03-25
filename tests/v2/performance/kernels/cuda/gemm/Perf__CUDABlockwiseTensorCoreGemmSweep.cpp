/**
 * @file Perf__CUDABlockwiseTensorCoreGemmSweep.cpp
 * @brief Parameterized GEMV dispatch sweep for CUDA native-vnni tuned kernels.
 */

#include <gtest/gtest.h>

#ifdef HAVE_CUDA

#include <cuda_runtime.h>

#include "kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#include "kernels/KernelFactory.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "backends/DeviceId.h"
#include "../../../../utils/TestTensorFactory.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <tuple>
#include <utility>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::cuda;
using namespace llaminar2::test;
using llaminar::v2::kernels::KernelFactory;

extern "C"
{
    void cudaNativeVNNIGemvSweep_setConfig(
        int kernel_family, int tile_n, int cpt,
        int target_waves, int mkg, int max_kb,
        int force_two_phase);
    void cudaNativeVNNIGemvSweep_clearConfig();
}

namespace
{
    constexpr int kDefaultWarmupRuns = 2;
    constexpr int kDefaultBenchRuns = 5;

    struct FormatSpec
    {
        std::string name;
        std::function<std::unique_ptr<TensorBase>(size_t, size_t)> create;
    };

    const std::vector<FormatSpec> kFormats = {
        {"Q4_0", [](size_t n, size_t k)
         { return TestTensorFactory::createQ4_0Random({n, k}); }},
        {"IQ4_NL", [](size_t n, size_t k)
         { return TestTensorFactory::createIQ4_NLRandom({n, k}); }},
        {"Q4_1", [](size_t n, size_t k)
         { return TestTensorFactory::createQ4_1Random({n, k}); }},
        {"Q5_0", [](size_t n, size_t k)
         { return TestTensorFactory::createQ5_0Random({n, k}); }},
        {"Q5_1", [](size_t n, size_t k)
         { return TestTensorFactory::createQ5_1Random({n, k}); }},
        {"Q6_K", [](size_t n, size_t k)
         { return TestTensorFactory::createQ6_KRandom({n, k}); }},
        {"Q3_K", [](size_t n, size_t k)
         { return TestTensorFactory::createQ3_KRandom({n, k}); }},
        {"Q2_K", [](size_t n, size_t k)
         { return TestTensorFactory::createQ2_KRandom({n, k}); }},
        {"IQ3_S", [](size_t n, size_t k)
         { return TestTensorFactory::createIQ3_SRandom({n, k}); }},
        {"IQ3_XXS", [](size_t n, size_t k)
         { return TestTensorFactory::createIQ3_XXSRandom({n, k}); }},
        {"IQ2_S", [](size_t n, size_t k)
         { return TestTensorFactory::createIQ2_SRandom({n, k}); }},
        {"IQ2_XS", [](size_t n, size_t k)
         { return TestTensorFactory::createIQ2_XSRandom({n, k}); }},
        {"IQ2_XXS", [](size_t n, size_t k)
         { return TestTensorFactory::createIQ2_XXSRandom({n, k}); }},
        {"IQ1_S", [](size_t n, size_t k)
         { return TestTensorFactory::createIQ1_SRandom({n, k}); }},
        {"IQ1_M", [](size_t n, size_t k)
         { return TestTensorFactory::createIQ1_MRandom({n, k}); }},
        {"Q8_0", [](size_t n, size_t k)
         { return TestTensorFactory::createQ8_0Random({n, k}); }},
    };

    struct Shape
    {
        std::string name;
        int n;
        int k;
    };

    const std::vector<Shape> kQwenShapes = {
        {"0.5B_Attn", 896, 896},
        {"0.5B_FFN_Up", 4864, 896},
        {"0.5B_FFN_Down", 896, 4864},
        {"0.5B_LM_Head", 151936, 896},
        {"0.5B_TP2_Attn_QKV", 448, 896},
        {"0.5B_TP2_Attn_Wo", 896, 448},
        {"0.5B_TP2_FFN_Up", 2432, 896},
        {"0.5B_TP2_FFN_Down", 896, 2432},
        {"0.5B_TP2_LM_Head", 75968, 896},
        {"0.5B_TP4_Attn_QKV", 224, 896},
        {"0.5B_TP4_Attn_Wo", 896, 224},
        {"0.5B_TP4_FFN_Up", 1216, 896},
        {"0.5B_TP4_FFN_Down", 896, 1216},
        {"0.5B_TP4_LM_Head", 37984, 896},
        {"3B_Attn", 2048, 2048},
        {"3B_FFN_Up", 11008, 2048},
        {"3B_FFN_Down", 2048, 11008},
        {"3B_LM_Head", 151936, 2048},
        {"3B_TP2_Attn_QKV", 1024, 2048},
        {"3B_TP2_Attn_Wo", 2048, 1024},
        {"3B_TP2_FFN_Up", 5504, 2048},
        {"3B_TP2_FFN_Down", 2048, 5504},
        {"3B_TP2_LM_Head", 75968, 2048},
        {"3B_TP4_Attn_QKV", 512, 2048},
        {"3B_TP4_Attn_Wo", 2048, 512},
        {"3B_TP4_FFN_Up", 2752, 2048},
        {"3B_TP4_FFN_Down", 2048, 2752},
        {"3B_TP4_LM_Head", 37984, 2048},
        {"7B_Attn", 3584, 3584},
        {"7B_FFN_Up", 18944, 3584},
        {"7B_FFN_Down", 3584, 18944},
        {"7B_LM_Head", 152064, 3584},
        {"7B_TP2_Attn_QKV", 1792, 3584},
        {"7B_TP2_Attn_Wo", 3584, 1792},
        {"7B_TP2_FFN_Up", 9472, 3584},
        {"7B_TP2_FFN_Down", 3584, 9472},
        {"7B_TP2_LM_Head", 76032, 3584},
        {"7B_TP4_Attn_QKV", 896, 3584},
        {"7B_TP4_Attn_Wo", 3584, 896},
        {"7B_TP4_FFN_Up", 4736, 3584},
        {"7B_TP4_FFN_Down", 3584, 4736},
        {"7B_TP4_LM_Head", 38016, 3584},
        {"14B_Attn", 5120, 5120},
        {"14B_FFN_Up", 13824, 5120},
        {"14B_FFN_Down", 5120, 13824},
        {"14B_LM_Head", 152064, 5120},
        {"14B_TP2_Attn_QKV", 2560, 5120},
        {"14B_TP2_Attn_Wo", 5120, 2560},
        {"14B_TP2_FFN_Up", 6912, 5120},
        {"14B_TP2_FFN_Down", 5120, 6912},
        {"14B_TP2_LM_Head", 76032, 5120},
        {"14B_TP4_Attn_QKV", 1280, 5120},
        {"14B_TP4_Attn_Wo", 5120, 1280},
        {"14B_TP4_FFN_Up", 3456, 5120},
        {"14B_TP4_FFN_Down", 5120, 3456},
        {"14B_TP4_LM_Head", 38016, 5120},
    };

    enum class SweepFamily
    {
        Wide = 0,
        KPar = 1,
        Direct = 2,
        RowPar = 3,
    };

    struct TuneCandidate
    {
        SweepFamily family = SweepFamily::KPar;
        int tile_n = 0;
        int cpt = 0;
        int target_waves = 0;
        int mkg = 0;
        int max_kb = 0;
        int force_two_phase = 0;
    };

    struct TuneResult
    {
        TuneCandidate candidate;
        double min_us = 0.0;
        double eff_bw = 0.0;
        double pct_peak = 0.0;
    };

    struct SweepConfig
    {
        int warmup_runs = kDefaultWarmupRuns;
        int bench_runs = kDefaultBenchRuns;
        std::set<std::string> format_filters;
        std::set<std::string> shape_filters;
        std::set<std::string> family_filters;
        std::vector<std::pair<int, int>> wide_tiles = {{128, 1}, {128, 2}, {256, 2}, {256, 4}, {512, 4}};
        std::vector<std::pair<int, int>> direct_tiles = {{128, 1}, {64, 1}, {32, 1}};
        std::vector<std::pair<int, int>> kpar_tiles = {{128, 1}, {128, 2}, {256, 2}, {256, 4}, {64, 1}, {64, 2}, {32, 1}};
        std::vector<int> target_waves = {4, 8, 16};
        std::vector<int> mkg_values = {2, 4, 8};
        std::vector<int> max_kb_values = {0, 2, 4, 8};
        std::vector<int> force_two_phase_values = {0, 1, 2};
        int max_cases = std::numeric_limits<int>::max();
        bool smoke = false;
        std::string csv_path = "/tmp/llaminar_cuda_tc_gemv_sweep.csv";
    };

    std::string toLower(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    std::string trim(std::string value)
    {
        const auto begin = value.find_first_not_of(" \t\n\r");
        if (begin == std::string::npos)
            return {};
        const auto end = value.find_last_not_of(" \t\n\r");
        return value.substr(begin, end - begin + 1);
    }

    std::optional<int> getEnvInt(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return std::nullopt;
        return std::atoi(raw);
    }

    bool getEnvFlag(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return false;
        return std::atoi(raw) != 0;
    }

    std::string getEnvString(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return {};
        return trim(raw);
    }

    std::set<std::string> getEnvCsvSet(const char *name)
    {
        std::set<std::string> values;
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return values;

        std::stringstream ss(raw);
        std::string item;
        while (std::getline(ss, item, ','))
        {
            item = toLower(trim(item));
            if (!item.empty())
                values.insert(item);
        }
        return values;
    }

    std::vector<int> getEnvCsvInts(const char *name)
    {
        std::vector<int> values;
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return values;

        std::stringstream ss(raw);
        std::string item;
        while (std::getline(ss, item, ','))
        {
            item = trim(item);
            if (!item.empty())
                values.push_back(std::atoi(item.c_str()));
        }
        return values;
    }

    std::vector<std::pair<int, int>> getEnvTilePairs(
        const char *name,
        const std::vector<std::pair<int, int>> &fallback)
    {
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return fallback;

        std::vector<std::pair<int, int>> values;
        std::stringstream ss(raw);
        std::string item;
        while (std::getline(ss, item, ','))
        {
            item = trim(item);
            if (item.empty())
                continue;
            const auto x = item.find('x');
            if (x == std::string::npos)
                continue;
            const int tile_n = std::atoi(item.substr(0, x).c_str());
            const int cpt = std::atoi(item.substr(x + 1).c_str());
            if (tile_n > 0 && cpt > 0)
                values.emplace_back(tile_n, cpt);
        }
        return values.empty() ? fallback : values;
    }

    bool shouldRunName(const std::set<std::string> &filters, const std::string &name)
    {
        return filters.empty() || filters.count(toLower(name)) > 0;
    }

    std::string familyName(SweepFamily family)
    {
        switch (family)
        {
        case SweepFamily::Wide:
            return "wide";
        case SweepFamily::KPar:
            return "kpar";
        case SweepFamily::Direct:
            return "direct";
        case SweepFamily::RowPar:
            return "rowpar";
        }
        return "unknown";
    }

    double measurePeakBandwidth()
    {
        int device = 0;
        cudaGetDevice(&device);
        int clockKHz = 0, busWidthBits = 0;
        cudaDeviceGetAttribute(&clockKHz, cudaDevAttrMemoryClockRate, device);
        cudaDeviceGetAttribute(&busWidthBits, cudaDevAttrGlobalMemoryBusWidth, device);
        if (clockKHz <= 0 || busWidthBits <= 0)
            return 0.0;
        return 2.0 * static_cast<double>(clockKHz) * 1e3 * static_cast<double>(busWidthBits / 8) / 1e9;
    }

    SweepConfig loadSweepConfig()
    {
        SweepConfig cfg;
        cfg.smoke = getEnvFlag("LLAMINAR_CUDA_TC_SMOKE");

        if (cfg.smoke)
        {
            cfg.warmup_runs = 1;
            cfg.bench_runs = 2;
            cfg.format_filters = {"q4_0"};
            cfg.shape_filters = {"0.5b_attn", "0.5b_lm_head", "3b_attn"};
            cfg.target_waves = {8};
            cfg.mkg_values = {4};
            cfg.max_kb_values = {0, 4};
            cfg.force_two_phase_values = {0, 1, 2};
            cfg.max_cases = 3;
        }

        if (const auto value = getEnvInt("LLAMINAR_CUDA_TC_WARMUP_RUNS"))
            cfg.warmup_runs = std::max(0, *value);
        if (const auto value = getEnvInt("LLAMINAR_CUDA_TC_BENCH_RUNS"))
            cfg.bench_runs = std::max(1, *value);
        if (const auto value = getEnvInt("LLAMINAR_CUDA_TC_MAX_CASES"))
            cfg.max_cases = std::max(1, *value);

        const auto formats = getEnvCsvSet("LLAMINAR_CUDA_TC_FORMATS");
        if (!formats.empty())
            cfg.format_filters = formats;

        const auto shapes = getEnvCsvSet("LLAMINAR_CUDA_TC_SHAPES");
        if (!shapes.empty())
            cfg.shape_filters = shapes;

        const auto families = getEnvCsvSet("LLAMINAR_CUDA_TC_SWEEP_FAMILIES");
        if (!families.empty())
            cfg.family_filters = families;

        cfg.wide_tiles = getEnvTilePairs("LLAMINAR_CUDA_TC_SWEEP_WIDE_TILES", cfg.wide_tiles);
        cfg.direct_tiles = getEnvTilePairs("LLAMINAR_CUDA_TC_SWEEP_DIRECT_TILES", cfg.direct_tiles);
        cfg.kpar_tiles = getEnvTilePairs("LLAMINAR_CUDA_TC_SWEEP_KPAR_TILES", cfg.kpar_tiles);

        const auto target_waves = getEnvCsvInts("LLAMINAR_CUDA_TC_SWEEP_TARGET_WAVES");
        if (!target_waves.empty())
            cfg.target_waves = target_waves;

        const auto mkg_values = getEnvCsvInts("LLAMINAR_CUDA_TC_SWEEP_MKG");
        if (!mkg_values.empty())
            cfg.mkg_values = mkg_values;

        const auto max_kb_values = getEnvCsvInts("LLAMINAR_CUDA_TC_SWEEP_MAX_KB");
        if (!max_kb_values.empty())
            cfg.max_kb_values = max_kb_values;

        const auto force_phase = getEnvCsvInts("LLAMINAR_CUDA_TC_SWEEP_FORCE_PHASES");
        if (!force_phase.empty())
            cfg.force_two_phase_values = force_phase;

        const std::string csv_path = getEnvString("LLAMINAR_CUDA_TC_SWEEP_CSV");
        if (!csv_path.empty())
            cfg.csv_path = csv_path;

        return cfg;
    }

    std::vector<TuneCandidate> buildCandidates(const SweepConfig &cfg)
    {
        std::vector<TuneCandidate> candidates;

        const auto allow_family = [&](const std::string &name)
        {
            return cfg.family_filters.empty() || cfg.family_filters.count(name) > 0;
        };

        if (allow_family("wide"))
        {
            for (const auto &[tile_n, cpt] : cfg.wide_tiles)
            {
                candidates.push_back({SweepFamily::Wide, tile_n, cpt, 0, 0, 0, 0});
            }
        }

        if (allow_family("direct"))
        {
            for (const auto &[tile_n, cpt] : cfg.direct_tiles)
            {
                candidates.push_back({SweepFamily::Direct, tile_n, cpt, 0, 0, 0, 0});
            }
        }

        if (allow_family("rowpar"))
        {
            for (const int nwarps : {2, 4, 8})
            {
                candidates.push_back({SweepFamily::RowPar, nwarps, 1, 0, 0, 0, 0});
            }
        }

        if (allow_family("kpar"))
        {
            for (const auto &[tile_n, cpt] : cfg.kpar_tiles)
            {
                for (const int target_waves : cfg.target_waves)
                {
                    for (const int mkg : cfg.mkg_values)
                    {
                        for (const int max_kb : cfg.max_kb_values)
                        {
                            for (const int force_phase : cfg.force_two_phase_values)
                            {
                                candidates.push_back({SweepFamily::KPar, tile_n, cpt, target_waves, mkg, max_kb, force_phase});
                            }
                        }
                    }
                }
            }
        }

        return candidates;
    }

    std::string shellQuote(const std::string &value)
    {
        std::string quoted = "'";
        for (const char ch : value)
        {
            if (ch == '\'')
                quoted += "'\\''";
            else
                quoted += ch;
        }
        quoted += "'";
        return quoted;
    }

    std::string existingPathOrDefault(std::initializer_list<std::string> candidates)
    {
        for (const auto &candidate : candidates)
        {
            if (!candidate.empty() && std::filesystem::exists(candidate))
                return candidate;
        }
        for (const auto &candidate : candidates)
        {
            if (!candidate.empty())
                return candidate;
        }
        return {};
    }

    std::string loadHeuristicInputCsvPath()
    {
        const std::string heuristic_csv = getEnvString("LLAMINAR_CUDA_TC_HEURISTIC_INPUT_CSV");
        const std::string sweep_csv = getEnvString("LLAMINAR_CUDA_TC_SWEEP_CSV");
        return existingPathOrDefault({
            heuristic_csv,
            sweep_csv,
            "/tmp/llaminar_cuda_tc_gemv_sweep_expanded_20260312.csv",
            "/tmp/llaminar_cuda_tc_gemv_sweep.csv",
        });
    }

    std::string loadHeuristicOutputPath()
    {
        const std::string path = getEnvString("LLAMINAR_CUDA_TC_HEURISTIC_OUTPUT");
        return path.empty() ? "/tmp/llaminar_cuda_tc_gemv_dispatch_heuristic_generated.inc" : path;
    }

    std::string loadHeuristicSummaryPath()
    {
        const std::string path = getEnvString("LLAMINAR_CUDA_TC_HEURISTIC_SUMMARY");
        return path.empty() ? "/tmp/llaminar_cuda_tc_gemv_dispatch_heuristic_summary.txt" : path;
    }

    std::string loadHeuristicScriptPath()
    {
        const std::string path = getEnvString("LLAMINAR_CUDA_TC_HEURISTIC_SCRIPT");
        return path.empty()
                   ? "/workspaces/llaminar/tests/v2/performance/kernels/cuda/gemm/analyze_cuda_tc_gemv_dispatch.py"
                   : path;
    }

    bool fileHasContent(const std::string &path)
    {
        std::ifstream input(path);
        return input.good() && input.peek() != std::ifstream::traits_type::eof();
    }

    struct RunResult
    {
        double min_us = 0.0;
    };

    class GemvSweepPerf : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            int device_count = 0;
            const cudaError_t err = cudaGetDeviceCount(&device_count);
            if (err != cudaSuccess || device_count == 0)
                GTEST_SKIP() << "No CUDA devices available";

            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            device_ = DeviceId::cuda(0);
            peak_bw_GB_s_ = measurePeakBandwidth();
        }

        RunResult runTunedGemv(
            TensorBase *weights,
            int n,
            int k,
            const SweepConfig &cfg,
            const TuneCandidate &candidate)
        {
            CUDAQuantisedGemmKernel::setNativeVNNIEnabled(true);

            cudaNativeVNNIGemvSweep_setConfig(
                static_cast<int>(candidate.family),
                candidate.tile_n,
                candidate.cpt,
                candidate.target_waves,
                candidate.mkg,
                candidate.max_kb,
                candidate.force_two_phase);

            EXPECT_TRUE(weights->ensureOnDevice(device_));
            auto kernel = KernelFactory::createGemm(weights, DeviceType::CUDA);
            if (!kernel)
            {
                cudaNativeVNNIGemvSweep_clearConfig();
                throw std::runtime_error("KernelFactory::createGemm returned null");
            }

            auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
            std::unique_ptr<DeviceWorkspaceManager> workspace;
            if (ws_consumer)
            {
                workspace = std::make_unique<DeviceWorkspaceManager>(device_, 512ull * 1024ull * 1024ull);
                const auto reqs = ws_consumer->getWorkspaceRequirements(1, n, k);
                if (!workspace->allocate(reqs))
                {
                    cudaNativeVNNIGemvSweep_clearConfig();
                    throw std::runtime_error("Failed to allocate CUDA GEMM workspace");
                }
                ws_consumer->bindWorkspace(workspace.get());
            }

            auto h_input = TestTensorFactory::createFP32Random({1u, static_cast<size_t>(k)}, -0.25f, 0.25f, 7);

            // Create tensor wrappers and upload to GPU
            auto A_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{1u, static_cast<size_t>(k)});
            std::memcpy(A_tensor->mutable_data(), h_input->data(), static_cast<size_t>(k) * sizeof(float));
            auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{1u, static_cast<size_t>(n)});

            if (!A_tensor->ensureOnDevice(device_))
            {
                cudaNativeVNNIGemvSweep_clearConfig();
                throw std::runtime_error("ensureOnDevice A failed");
            }
            if (!C_tensor->ensureOnDevice(device_))
            {
                cudaNativeVNNIGemvSweep_clearConfig();
                throw std::runtime_error("ensureOnDevice C failed");
            }

            for (int i = 0; i < cfg.warmup_runs; ++i)
            {
                if (!kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), 1, n, k))
                {
                    // Some candidate/codebook combos are unsupported (e.g. ROWPAR
                    // for Q8_0).  Return a sentinel so the caller can skip them.
                    if (ws_consumer)
                        ws_consumer->unbindWorkspace();
                    cudaNativeVNNIGemvSweep_clearConfig();
                    return RunResult{.min_us = std::numeric_limits<double>::infinity()};
                }
            }

            std::vector<double> times_us;
            times_us.reserve(static_cast<size_t>(cfg.bench_runs));
            for (int i = 0; i < cfg.bench_runs; ++i)
            {
                cudaEvent_t start = nullptr;
                cudaEvent_t stop = nullptr;
                if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess)
                {
                    cudaNativeVNNIGemvSweep_clearConfig();
                    throw std::runtime_error("cudaEventCreate failed");
                }

                cudaEventRecord(start);
                const bool run_ok = kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), 1, n, k);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                float elapsed_ms = 0.0f;
                cudaEventElapsedTime(&elapsed_ms, start, stop);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);

                if (!run_ok)
                {
                    cudaNativeVNNIGemvSweep_clearConfig();
                    throw std::runtime_error("CUDA native payload GEMV bench run failed");
                }

                times_us.push_back(static_cast<double>(elapsed_ms) * 1000.0);
            }

            if (ws_consumer)
                ws_consumer->unbindWorkspace();
            cudaNativeVNNIGemvSweep_clearConfig();

            RunResult result;
            result.min_us = *std::min_element(times_us.begin(), times_us.end());
            return result;
        }

        DeviceId device_ = DeviceId::cpu();
        double peak_bw_GB_s_ = 0.0;
    };

    TEST_F(GemvSweepPerf, Sweep_GemvDispatchCsv)
    {
        const SweepConfig cfg = loadSweepConfig();
        const auto candidates = buildCandidates(cfg);
        ASSERT_FALSE(candidates.empty()) << "No sweep candidates configured.";

        std::FILE *csv = std::fopen(cfg.csv_path.c_str(), "w");
        ASSERT_NE(csv, nullptr) << "Failed to open CSV: " << cfg.csv_path;
        std::fprintf(csv,
                     "format,shape,n,k,weight_bytes,family,tile_n,cpt,target_waves,mkg,max_kb,force_two_phase,min_us,eff_bw_gbs,pct_peak,is_best\n");

        int executed_cases = 0;
        int executed_rows = 0;

        for (const auto &format : kFormats)
        {
            if (!shouldRunName(cfg.format_filters, format.name))
                continue;

            for (const auto &shape : kQwenShapes)
            {
                if (!shouldRunName(cfg.shape_filters, shape.name))
                    continue;
                if ((shape.k % 32) != 0)
                    continue;
                if (executed_cases >= cfg.max_cases)
                    break;

                auto weights = format.create(static_cast<size_t>(shape.n), static_cast<size_t>(shape.k));
                const size_t weight_bytes = weights->size_bytes();
                const size_t total_bytes = weight_bytes + static_cast<size_t>(shape.k) * sizeof(float) + static_cast<size_t>(shape.n) * sizeof(float);

                std::vector<TuneResult> results;
                results.reserve(candidates.size());

                for (const auto &candidate : candidates)
                {
                    const RunResult run = runTunedGemv(weights.get(), shape.n, shape.k, cfg, candidate);
                    if (std::isinf(run.min_us))
                        continue; // Unsupported candidate for this codebook
                    const double eff_bw = static_cast<double>(total_bytes) / (run.min_us * 1e-6) / 1e9;
                    const double pct_peak = (peak_bw_GB_s_ > 0.0) ? (eff_bw / peak_bw_GB_s_ * 100.0) : 0.0;
                    results.push_back({candidate, run.min_us, eff_bw, pct_peak});
                }

                const auto best_it = std::min_element(
                    results.begin(), results.end(),
                    [](const TuneResult &lhs, const TuneResult &rhs)
                    {
                        return lhs.min_us < rhs.min_us;
                    });
                ASSERT_NE(best_it, results.end());

                for (const auto &result : results)
                {
                    const int is_best = (&result == &(*best_it)) ? 1 : 0;
                    std::fprintf(csv,
                                 "%s,%s,%d,%d,%zu,%s,%d,%d,%d,%d,%d,%d,%.3f,%.1f,%.1f,%d\n",
                                 format.name.c_str(), shape.name.c_str(), shape.n, shape.k,
                                 weight_bytes,
                                 familyName(result.candidate.family).c_str(),
                                 result.candidate.tile_n,
                                 result.candidate.cpt,
                                 result.candidate.target_waves,
                                 result.candidate.mkg,
                                 result.candidate.max_kb,
                                 result.candidate.force_two_phase,
                                 result.min_us,
                                 result.eff_bw,
                                 result.pct_peak,
                                 is_best);
                    ++executed_rows;
                }
                std::fflush(csv);

                std::fprintf(stderr,
                             "[CUDABlockwiseTC][SWEEP][BEST] format=%s shape=%s family=%s tile=%dx%d waves=%d mkg=%d force_phase=%d time_us=%.3f eff_bw=%.1fGB/s pct_peak=%.1f%%\n",
                             format.name.c_str(), shape.name.c_str(),
                             familyName(best_it->candidate.family).c_str(),
                             best_it->candidate.tile_n,
                             best_it->candidate.cpt,
                             best_it->candidate.target_waves,
                             best_it->candidate.mkg,
                             best_it->candidate.force_two_phase,
                             best_it->min_us,
                             best_it->eff_bw,
                             best_it->pct_peak);

                ++executed_cases;
            }

            if (executed_cases >= cfg.max_cases)
                break;
        }

        std::fclose(csv);
        ASSERT_GT(executed_cases, 0) << "No sweep cases selected. Check LLAMINAR_CUDA_TC_FORMATS / LLAMINAR_CUDA_TC_SHAPES.";
        ASSERT_GT(executed_rows, 0) << "Sweep produced no CSV rows.";
        std::fprintf(stderr,
                     "[CUDABlockwiseTC][SWEEP] wrote %d rows across %d cases to %s\n",
                     executed_rows, executed_cases, cfg.csv_path.c_str());
    }

    TEST(CUDABlockwiseTensorCoreGemmSweepOffline, GenerateDispatchHeuristicFromExistingCsv)
    {
        const std::string input_csv = loadHeuristicInputCsvPath();
        if (input_csv.empty() || !std::filesystem::exists(input_csv))
        {
            GTEST_SKIP() << "No existing sweep CSV found. Set LLAMINAR_CUDA_TC_HEURISTIC_INPUT_CSV.";
        }

        const std::string script_path = loadHeuristicScriptPath();
        ASSERT_TRUE(std::filesystem::exists(script_path)) << "Heuristic generator script missing: " << script_path;

        const std::string output_path = loadHeuristicOutputPath();
        const std::string summary_path = loadHeuristicSummaryPath();

        std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
        std::filesystem::create_directories(std::filesystem::path(summary_path).parent_path());

        const std::string command =
            "python3 " + shellQuote(script_path) +
            " --input " + shellQuote(input_csv) +
            " --output " + shellQuote(output_path) +
            " --summary " + shellQuote(summary_path) +
            " --min-overall-family-pct 99.0" +
            " --min-overall-exact-pct 99.0" +
            " --min-fallback-family-pct 97.0" +
            " --min-fallback-exact-pct 30.0" +
            " 2>&1";

        FILE *pipe = popen(command.c_str(), "r");
        ASSERT_NE(pipe, nullptr) << "Failed to spawn heuristic generator";

        char buffer[256];
        std::string output;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
            output += buffer;

        const int status = pclose(pipe);
        ASSERT_TRUE(WIFEXITED(status)) << "Heuristic generator terminated abnormally:\n"
                                       << output;
        ASSERT_EQ(WEXITSTATUS(status), 0) << "Heuristic generator failed:\n"
                                          << output;

        ASSERT_TRUE(std::filesystem::exists(output_path)) << "Heuristic output was not created: " << output_path;
        ASSERT_TRUE(std::filesystem::exists(summary_path)) << "Heuristic summary was not created: " << summary_path;
        ASSERT_TRUE(fileHasContent(output_path)) << "Heuristic output is empty: " << output_path;
        ASSERT_TRUE(fileHasContent(summary_path)) << "Heuristic summary is empty: " << summary_path;

        std::fprintf(stderr,
                     "[CUDABlockwiseTC][HEURISTIC] input=%s output=%s summary=%s\n%s",
                     input_csv.c_str(), output_path.c_str(), summary_path.c_str(), output.c_str());
    }
}

#endif