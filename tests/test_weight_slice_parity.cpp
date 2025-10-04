#include <gtest/gtest.h>
#include <filesystem>
#include <string>
#include <vector>
#include <cstdlib>
#include <limits>
#include <mpi.h>
#include <sched.h>
#include <unistd.h>

#include "logger.h"
#include "model_loader.h"
#include "distributed_transformer_pipeline.h"

// Forward declaration (existing in namespace)
namespace llaminar
{
    DistributedTransformerPipeline::ModelWeights loadModelWeights(const std::string &model_path, const DistributedTransformerPipeline::LayerConfig &config);
}

namespace
{
    std::string pick_small_model()
    {
        // Explicit override wins.
        if (const char *env = std::getenv("LLAMINAR_SLICE_PARITY_MODEL"))
            return env;
        namespace fs = std::filesystem;
        fs::path dir{"models"};
        if (!fs::exists(dir))
            return {};

        // Collect candidates and prefer unquantized fp16/f32 first so slicing/validation path is exercised.
        std::vector<fs::directory_entry> fp_candidates;
        std::vector<fs::directory_entry> other;
        for (auto &p : fs::directory_iterator(dir))
        {
            if (!p.is_regular_file())
                continue;
            if (p.path().extension() != ".gguf")
                continue;
            std::string name = p.path().filename().string();
            // Heuristic: treat files containing "fp16" or "f16" or "f32" as float candidates.
            if (name.find("fp16") != std::string::npos || name.find("f16") != std::string::npos || name.find("f32") != std::string::npos)
            {
                fp_candidates.push_back(p);
            }
            else
            {
                other.push_back(p);
            }
        }
        auto pick_smallest = [](const std::vector<fs::directory_entry> &list) -> std::string
        {
            uintmax_t best = (std::numeric_limits<uintmax_t>::max)();
            std::string choice;
            for (auto &p : list)
            {
                uintmax_t sz = 0;
                try
                {
                    sz = fs::file_size(p.path());
                }
                catch (...)
                {
                    continue;
                }
                if (sz < best)
                {
                    best = sz;
                    choice = p.path().string();
                }
            }
            return choice;
        };
        std::string chosen = pick_smallest(fp_candidates);
        if (chosen.empty())
            chosen = pick_smallest(other);
        return chosen;
    }
}

TEST(WeightSliceParity, AttentionProjectionColumns)
{
    int init_flag = 0;
    MPI_Initialized(&init_flag);
    int provided = 0;
    if (!init_flag)
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SINGLE, &provided);
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world < 2)
    {
        GTEST_SKIP() << "Weight slicing parity test requires >=2 MPI ranks (run with mpirun -np 2).";
    }
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Diagnose CPU affinity (likely single-core binding causing single-threaded conversion performance)
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        int aff_ok = sched_getaffinity(0, sizeof(cpuset), &cpuset);
        if (aff_ok == 0)
        {
            long nconf = sysconf(_SC_NPROCESSORS_CONF);
            int cnt = 0;
            std::vector<int> first_list;
            for (int i = 0; i < nconf && i < 4096; ++i)
            {
                if (CPU_ISSET(i, &cpuset))
                {
                    cnt++;
                    if ((int)first_list.size() < 32)
                        first_list.push_back(i);
                }
            }
            if (rank == 0)
            {
                fprintf(stderr, "[SLICE-PARITY-DIAG] rank=%d affinity_count=%d nconf=%ld sample=", rank, cnt, nconf);
                for (size_t i = 0; i < first_list.size(); ++i)
                {
                    fprintf(stderr, "%s%d", i ? "," : "", first_list[i]);
                }
                fprintf(stderr, "\n");
                if (cnt <= 2)
                {
                    fprintf(stderr, "[SLICE-PARITY-DIAG] Low affinity count detected (<=2). Try: mpirun --bind-to none --map-by ppr:1:socket or adjust container cpuset.\n");
                }
            }
        }
        else if (rank == 0)
        {
            fprintf(stderr, "[SLICE-PARITY-DIAG] sched_getaffinity failed (%d)\n", aff_ok);
        }
    }

    Logger::getInstance().setLogLevel(LogLevel::INFO);

    // Force slicing + validation (set before any model/tensor operations to ensure snapshot picks them up)
    setenv("LLAMINAR_FORCE_WEIGHT_SHARDING", "1", 1);
    setenv("LLAMINAR_WEIGHT_SLICE_VALIDATE", "1", 1);
    unsetenv("LLAMINAR_DISABLE_WEIGHT_SHARDING");
    // Ensure we don't accidentally skip due to min-cols gating for smaller test models.
    setenv("LLAMINAR_WEIGHT_SLICE_MIN_COLS", "0", 1);
    // Refresh snapshot in case another test or static initialization accessed debugEnv earlier.
    llaminar::debugEnvRefresh();

    auto model_path = pick_small_model();
    ASSERT_FALSE(model_path.empty()) << "No .gguf model found under models/ for parity test";

    if (rank == 0)
        fprintf(stderr, "[SLICE-PARITY] model=%s ranks=%d\n", model_path.c_str(), world);

    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(model_path)) << "Failed to load model: " << model_path;
    auto config = loader.createLayerConfig();
    auto pipeline = std::make_unique<llaminar::DistributedTransformerPipeline>(config);
    // Avoid double full-model load by using overload that reuses existing ModelLoader
    auto weights = llaminar::loadModelWeights(loader, config);
    (void)weights;
    (void)pipeline;

    // Gather recent log lines and parse validation metrics
    const auto lines = Logger::getInstance().recent_lines();
    int validate_lines = 0;
    double worst_rel_l2 = 0.0;
    double worst_max_abs = 0.0;
    for (const auto &l : lines)
    {
        auto pos = l.find("[WEIGHT_SLICE_VALIDATE]");
        if (pos == std::string::npos)
            continue;
        validate_lines++;
        // Parse rel_l2 and max_abs tokens
        auto find_metric = [&](const char *key) -> double
        {
            auto kpos = l.find(key);
            if (kpos == std::string::npos)
                return -1.0;
            kpos += std::strlen(key);
            size_t end = kpos;
            while (end < l.size() && (std::isdigit((unsigned char)l[end]) || l[end] == 'e' || l[end] == 'E' || l[end] == '+' || l[end] == '-' || l[end] == '.'))
                end++;
            try
            {
                return std::stod(l.substr(kpos, end - kpos));
            }
            catch (...)
            {
                return -2.0;
            }
        };
        double rel_l2 = find_metric("rel_l2=");
        double max_abs = find_metric("max_abs=");
        if (rel_l2 > worst_rel_l2)
            worst_rel_l2 = rel_l2;
        if (max_abs > worst_max_abs)
            worst_max_abs = max_abs;
    }

    // Reduce across ranks to get global worst metrics (just in case logs appear on different ranks)
    double global_worst_rel_l2 = worst_rel_l2;
    double global_worst_max_abs = worst_max_abs;
    MPI_Allreduce(MPI_IN_PLACE, &global_worst_rel_l2, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &global_worst_max_abs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    int global_validate_lines = 0;
    MPI_Allreduce(&validate_lines, &global_validate_lines, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0)
    {
        fprintf(stderr, "[SLICE-PARITY-SUMMARY] validations=%d worst_rel_l2=%.3e worst_max_abs=%.3e\n", global_validate_lines, global_worst_rel_l2, global_worst_max_abs);
    }

    if (global_validate_lines == 0)
    {
        // Diagnostic: search for any slice attempt lines; if absent, likely running only quantized model unsupported yet.
        int slice_lines = 0;
        for (const auto &l : lines)
        {
            if (l.find("[WEIGHT_SLICE]") != std::string::npos)
                slice_lines++;
        }
        if (slice_lines == 0)
        {
            GTEST_SKIP() << "Weight slicing not triggered (no [WEIGHT_SLICE] lines) – model may be quantized only; provide fp16/f32 model or set LLAMINAR_SLICE_PARITY_MODEL.";
        }
        FAIL() << "Slicing triggered ([WEIGHT_SLICE] present) but no validation lines; check validation flag or parity registry.";
    }
    // Exact equality should hold (we reconstruct the original tensor) but allow a tiny FP slack.
    ASSERT_LE(global_worst_rel_l2, 1e-8) << "Relative L2 mismatch too large";
    ASSERT_LE(global_worst_max_abs, 1e-7) << "Absolute max mismatch too large";

    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized && !init_flag)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
    }
}
