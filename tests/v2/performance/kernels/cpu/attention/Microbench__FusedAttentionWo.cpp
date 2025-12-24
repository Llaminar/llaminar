/**
 * @file Microbench__FusedAttentionWo.cpp
 * @brief Standalone microbenchmark for the fused attention(+Wo) JIT kernel.
 *
 * Intended usage:
 *   - Run in Release build under `perf stat` to capture cache stats/misses.
 *   - Keep code path as close as possible to Perf__FusedAttentionWo.cpp.
 */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "kernels/KernelFactory.h"
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include "kernels/cpu/attention/q8_1/jit/JitFusedAttentionWo.h"

using namespace llaminar2;
using namespace llaminar::v2::kernels::jit;
using namespace llaminar::v2::kernels::microkernels;

namespace
{

    struct BenchCase
    {
        int seq_len_q;
        int seq_len_kv;
        int num_heads;
        int num_kv_heads;
        int head_dim;
        int d_model;
        const char *name;
    };

    struct Options
    {
        std::string case_name = "qwen05b_decode_kv128";
        std::string fa = "fa2";    // fa1|fa2
        std::string wo = "q8_1";   // fp32|q8_1|q8_1_vnni_packed
        std::string work = "auto"; // auto|small|large|xl
        std::string tile = "auto"; // auto|4|8
        bool sweep = false;
        int sweep_repeats = 1;
        int warmup_iters = 50;
        int bench_iters = 500;
        uint32_t seed = 42;
    };

    [[noreturn]] void usage(const char *argv0)
    {
        std::cerr << "Usage: " << argv0 << " [--case=NAME] [--fa=fa1|fa2] [--wo=fp32|q8_1|q8_1_vnni_packed]"
                                           " [--work=auto|small|large|xl] [--tile=auto|4|8] [--sweep]"
                                           " [--sweep-repeats=N]"
                                           " [--warmup=N] [--iters=N] [--seed=N]\n";
        std::cerr << "  (wo formats: fp32, q8_1, q8_1_vnni_packed)\n";
        std::cerr << "  Sweep mode prints a table over qwen14b/32b decode KV cases.\n";
        std::cerr << "Cases:\n";
        std::cerr << "  qwen05b_decode_kv128   (seq_q=1, kv=128, head_dim=64,  heads=14/2, d_model=896)\n";
        std::cerr << "  qwen05b_decode_kv512   (seq_q=1, kv=512, head_dim=64,  heads=14/2, d_model=896)\n";
        std::cerr << "  qwen05b_prefill_seq32  (seq_q=32,kv=32,  head_dim=64,  heads=14/2, d_model=896)\n";
        std::cerr << "  qwen05b_prefill_seq128 (seq_q=128,kv=128,head_dim=64,  heads=14/2, d_model=896)\n";
        std::cerr << "  qwen7b_decode_kv128    (seq_q=1, kv=128, head_dim=128, heads=28/4, d_model=3584)\n";
        std::cerr << "  qwen7b_prefill_seq128  (seq_q=128,kv=128,head_dim=128, heads=28/4, d_model=3584)\n";
        std::cerr << "  qwen14b_decode_kv512   (seq_q=1, kv=512, head_dim=128, heads=40/8, d_model=5120)\n";
        std::cerr << "  qwen14b_decode_kv2048  (seq_q=1, kv=2048,head_dim=128, heads=40/8, d_model=5120)\n";
        std::cerr << "  qwen14b_decode_kv4096  (seq_q=1, kv=4096,head_dim=128, heads=40/8, d_model=5120)\n";
        std::cerr << "  qwen14b_decode_kv8192  (seq_q=1, kv=8192,head_dim=128, heads=40/8, d_model=5120)\n";
        std::cerr << "  qwen32b_decode_kv512   (seq_q=1, kv=512, head_dim=128, heads=40/8, d_model=5120)\n";
        std::cerr << "  qwen32b_decode_kv2048  (seq_q=1, kv=2048,head_dim=128, heads=40/8, d_model=5120)\n";
        std::cerr << "  qwen32b_decode_kv4096  (seq_q=1, kv=4096,head_dim=128, heads=40/8, d_model=5120)\n";
        std::cerr << "  qwen32b_decode_kv8192  (seq_q=1, kv=8192,head_dim=128, heads=40/8, d_model=5120)\n";
        std::exit(2);
    }

    bool starts_with(std::string_view s, std::string_view prefix)
    {
        return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
    }

    Options parse_args(int argc, char **argv)
    {
        Options opt;
        for (int i = 1; i < argc; ++i)
        {
            std::string_view a(argv[i]);
            if (a == "--help" || a == "-h")
                usage(argv[0]);

            if (starts_with(a, "--case="))
                opt.case_name = std::string(a.substr(std::strlen("--case=")));
            else if (starts_with(a, "--fa="))
                opt.fa = std::string(a.substr(std::strlen("--fa=")));
            else if (starts_with(a, "--wo="))
                opt.wo = std::string(a.substr(std::strlen("--wo=")));
            else if (starts_with(a, "--work="))
                opt.work = std::string(a.substr(std::strlen("--work=")));
            else if (starts_with(a, "--tile="))
                opt.tile = std::string(a.substr(std::strlen("--tile=")));
            else if (a == "--sweep")
                opt.sweep = true;
            else if (starts_with(a, "--sweep-repeats="))
                opt.sweep_repeats = std::stoi(std::string(a.substr(std::strlen("--sweep-repeats="))));
            else if (starts_with(a, "--warmup="))
                opt.warmup_iters = std::stoi(std::string(a.substr(std::strlen("--warmup="))));
            else if (starts_with(a, "--iters="))
                opt.bench_iters = std::stoi(std::string(a.substr(std::strlen("--iters="))));
            else if (starts_with(a, "--seed="))
                opt.seed = static_cast<uint32_t>(std::stoul(std::string(a.substr(std::strlen("--seed=")))));
            else
                usage(argv[0]);
        }

        if (opt.fa != "fa1" && opt.fa != "fa2")
        {
            std::cerr << "Invalid --fa='" << opt.fa << "'\n";
            usage(argv[0]);
        }
        if (opt.wo != "fp32" && opt.wo != "q8_1" && opt.wo != "q8_1_vnni_packed")
        {
            std::cerr << "Invalid --wo='" << opt.wo << "'\n";
            usage(argv[0]);
        }
        if (opt.work != "auto" && opt.work != "small" && opt.work != "large" && opt.work != "xl")
        {
            std::cerr << "Invalid --work='" << opt.work << "'\n";
            usage(argv[0]);
        }
        if (opt.tile != "auto" && opt.tile != "4" && opt.tile != "8")
        {
            std::cerr << "Invalid --tile='" << opt.tile << "'\n";
            usage(argv[0]);
        }
        if (opt.warmup_iters < 0 || opt.bench_iters <= 0)
        {
            std::cerr << "Invalid --warmup/--iters\n";
            usage(argv[0]);
        }
        if (opt.sweep_repeats <= 0)
        {
            std::cerr << "Invalid --sweep-repeats\n";
            usage(argv[0]);
        }

        return opt;
    }

    WorkSizeClass resolve_work_bucket(const Options &opt, int seq_len_kv)
    {
        if (opt.work == "small")
            return WorkSizeClass::SMALL;
        if (opt.work == "large")
            return WorkSizeClass::LARGE;
        if (opt.work == "xl")
            return WorkSizeClass::XL;

        // auto: mirror production heuristics from FusedAttentionWoKernel.h
        // Sweep winners (Dec 2024):
        //   kv=512:  LARGE
        //   kv=2048: SMALL
        //   kv=4096: LARGE
        //   kv=8192: SMALL
        if (seq_len_kv > 8192)
            return WorkSizeClass::XL;
        if (seq_len_kv > 4096)
            return WorkSizeClass::SMALL; // 4097-8192
        if (seq_len_kv > 2048)
            return WorkSizeClass::LARGE; // 2049-4096
        if (seq_len_kv > 1024)
            return WorkSizeClass::SMALL; // 1025-2048
        return WorkSizeClass::LARGE;     // <= 1024
    }

    JitAttentionConfig::Fa2TileWidth resolve_tile_width(const Options &opt, int seq_len_kv)
    {
        if (opt.tile == "4")
            return JitAttentionConfig::Fa2TileWidth::KV4;
        if (opt.tile == "8")
            return JitAttentionConfig::Fa2TileWidth::KV8;

        // auto: mirror production heuristics from FusedAttentionWoKernel.h
        // Sweep winners: t8 for kv <= 4096, t4 for kv > 4096
        return (seq_len_kv <= 4096)
                   ? JitAttentionConfig::Fa2TileWidth::KV8
                   : JitAttentionConfig::Fa2TileWidth::KV4;
    }

    struct BenchResult
    {
        std::string case_name;
        int kv = 0;
        std::string work;
        int tile = 0;
        double per_iter_ms = 0.0;
        bool is_auto = false;
    };

    BenchResult run_one(const BenchCase &cfg, const Options &opt, const void *wo_ptr,
                        const Q8_1Block *Q, const Q8_1Block *K, const Q8_1Block *V,
                        float *output)
    {
        const float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));

        JitAttentionConfig jit_config;
        jit_config.head_dim = cfg.head_dim;
        jit_config.num_heads = cfg.num_heads;
        jit_config.num_kv_heads = cfg.num_kv_heads;
        jit_config.batch_size = cfg.seq_len_q;
        jit_config.causal = true;
        jit_config.use_fa2_tiling = (opt.fa == "fa2");

        WorkSizeClass work_bucket = resolve_work_bucket(opt, cfg.seq_len_kv);
        jit_config.work_size = work_bucket;
        jit_config.fa2_tile_width = resolve_tile_width(opt, cfg.seq_len_kv);

        if (opt.wo == "fp32")
            jit_config.wo_format = WoFormat::FP32;
        else if (opt.wo == "q8_1")
            jit_config.wo_format = WoFormat::Q8_1;
        else
            jit_config.wo_format = WoFormat::Q8_1_VNNI_PACKED;

        JitFusedAttentionWo kernel(jit_config);

        for (int i = 0; i < opt.warmup_iters; ++i)
        {
            kernel.compute(Q, K, V, wo_ptr, output,
                           cfg.seq_len_q, cfg.seq_len_kv, scale,
                           cfg.seq_len_kv - cfg.seq_len_q);
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < opt.bench_iters; ++i)
        {
            kernel.compute(Q, K, V, wo_ptr, output,
                           cfg.seq_len_q, cfg.seq_len_kv, scale,
                           cfg.seq_len_kv - cfg.seq_len_q);
        }
        auto end = std::chrono::high_resolution_clock::now();

        const double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        const double per_iter_ms = total_ms / static_cast<double>(opt.bench_iters);

        BenchResult r;
        r.case_name = cfg.name;
        r.kv = cfg.seq_len_kv;
        r.work = (work_bucket == WorkSizeClass::SMALL) ? "SMALL" : (work_bucket == WorkSizeClass::LARGE) ? "LARGE"
                                                                                                         : "XL";
        r.tile = (jit_config.fa2_tile_width == JitAttentionConfig::Fa2TileWidth::KV8) ? 8 : 4;
        r.per_iter_ms = per_iter_ms;
        r.is_auto = (opt.work == "auto" && opt.tile == "auto");
        return r;
    }

    const BenchCase &lookup_case(const std::string &name)
    {
        static const BenchCase cases[] = {
            {1, 128, 14, 2, 64, 896, "qwen05b_decode_kv128"},
            {1, 512, 14, 2, 64, 896, "qwen05b_decode_kv512"},
            {32, 32, 14, 2, 64, 896, "qwen05b_prefill_seq32"},
            {128, 128, 14, 2, 64, 896, "qwen05b_prefill_seq128"},
            {1, 128, 28, 4, 128, 3584, "qwen7b_decode_kv128"},
            {128, 128, 28, 4, 128, 3584, "qwen7b_prefill_seq128"},
            // Large-model decode shapes (single-token decode, varying KV length)
            // Qwen 14B/32B attention projection shapes are identical for this microbench:
            // d_model=5120, n_heads=40, n_kv_heads=8, head_dim=128.
            {1, 512, 40, 8, 128, 5120, "qwen14b_decode_kv512"},
            {1, 2048, 40, 8, 128, 5120, "qwen14b_decode_kv2048"},
            {1, 4096, 40, 8, 128, 5120, "qwen14b_decode_kv4096"},
            {1, 8192, 40, 8, 128, 5120, "qwen14b_decode_kv8192"},
            {1, 512, 40, 8, 128, 5120, "qwen32b_decode_kv512"},
            {1, 2048, 40, 8, 128, 5120, "qwen32b_decode_kv2048"},
            {1, 4096, 40, 8, 128, 5120, "qwen32b_decode_kv4096"},
            {1, 8192, 40, 8, 128, 5120, "qwen32b_decode_kv8192"},
        };

        for (const auto &c : cases)
        {
            if (name == c.name)
                return c;
        }

        std::cerr << "Unknown --case='" << name << "'\n";
        usage("microbench");
    }

    void generate_q8_1_blocks(std::vector<Q8_1Block> &blocks, const std::vector<float> &fp32_data)
    {
        const size_t num_elements = fp32_data.size();
        const size_t num_blocks = (num_elements + 31) / 32;
        blocks.resize(num_blocks);

        for (size_t b = 0; b < num_blocks; ++b)
        {
            float max_abs = 0.0f;
            const size_t start = b * 32;
            const size_t end = std::min(start + 32, num_elements);

            for (size_t i = start; i < end; ++i)
                max_abs = std::max(max_abs, std::abs(fp32_data[i]));

            float scale = max_abs / 127.0f;
            if (scale < 1e-10f)
                scale = 1e-10f;

            float inv_scale = (max_abs < 1e-10f) ? 0.0f : (127.0f / max_abs);

            int32_t sum_qs = 0;
            for (size_t i = 0; i < 32; ++i)
            {
                if (start + i < end)
                {
                    float val = fp32_data[start + i];
                    int8_t q = static_cast<int8_t>(std::round(val * inv_scale));
                    q = std::max(int8_t(-127), std::min(int8_t(127), q));
                    blocks[b].qs[i] = q;
                    sum_qs += q;
                }
                else
                {
                    blocks[b].qs[i] = 0;
                }
            }

            blocks[b].d = fp32_to_fp16(scale);
            blocks[b].sum_qs = static_cast<int16_t>(sum_qs);
        }
    }

} // namespace

int main(int argc, char **argv)
{
    const Options opt = parse_args(argc, argv);
    const BenchCase &cfg = lookup_case(opt.case_name);
    if (opt.sweep)
    {
        if (opt.fa != "fa2")
        {
            std::cerr << "Sweep mode requires --fa=fa2\n";
            return 2;
        }

        std::mt19937 base_rng(opt.seed);
        std::uniform_real_distribution<float> dist_qkv(-1.0f, 1.0f);
        std::uniform_real_distribution<float> dist_wo(-0.1f, 0.1f);

        const std::vector<std::string> sweep_cases = {
            "qwen14b_decode_kv512",
            "qwen14b_decode_kv2048",
            "qwen14b_decode_kv4096",
            "qwen14b_decode_kv8192",
            "qwen32b_decode_kv512",
            "qwen32b_decode_kv2048",
            "qwen32b_decode_kv4096",
            "qwen32b_decode_kv8192",
        };

        const std::vector<std::string> work_opts = {"small", "large", "xl"};
        const std::vector<std::string> tile_opts = {"4", "8"};

        Options run_opt = opt;
        run_opt.tile = "auto";
        run_opt.work = "auto";

        std::vector<BenchResult> all_results;
        all_results.reserve(sweep_cases.size() * (work_opts.size() * tile_opts.size() + 1));

        std::cout << "case,kv,work,tile,per_iter_ms,is_auto\n";
        std::cout << std::fixed << std::setprecision(3);

        for (const auto &case_name : sweep_cases)
        {
            const BenchCase &sc = lookup_case(case_name);

            // Keep some lightweight progress output on stderr so long sweeps don't look stuck.
            std::cerr << "[sweep] " << sc.name << "..." << std::endl;

            // Build per-case Q/K/V and Wo once, then sweep knobs on the same data.
            std::mt19937 rng(base_rng());
            rng.seed(static_cast<uint32_t>(opt.seed ^ static_cast<uint32_t>(std::hash<std::string>{}(case_name))));

            const size_t Q_elements = static_cast<size_t>(sc.seq_len_q) * sc.num_heads * sc.head_dim;
            const size_t K_elements = static_cast<size_t>(sc.seq_len_kv) * sc.num_kv_heads * sc.head_dim;
            const size_t V_elements = static_cast<size_t>(sc.seq_len_kv) * sc.num_kv_heads * sc.head_dim;

            std::vector<float> Q_fp32(Q_elements);
            std::vector<float> K_fp32(K_elements);
            std::vector<float> V_fp32(V_elements);
            for (auto &v : Q_fp32)
                v = dist_qkv(rng);
            for (auto &v : K_fp32)
                v = dist_qkv(rng);
            for (auto &v : V_fp32)
                v = dist_qkv(rng);

            std::vector<Q8_1Block> Q_blocks, K_blocks, V_blocks;
            generate_q8_1_blocks(Q_blocks, Q_fp32);
            generate_q8_1_blocks(K_blocks, K_fp32);
            generate_q8_1_blocks(V_blocks, V_fp32);

            const int wo_cols = sc.num_heads * sc.head_dim;
            const size_t Wo_elements = static_cast<size_t>(sc.d_model) * static_cast<size_t>(wo_cols);
            std::vector<float> Wo_fp32(Wo_elements);
            for (auto &v : Wo_fp32)
                v = dist_wo(rng);

            std::shared_ptr<llaminar2::Q8_1Tensor> wo_q8_tensor;
            const llaminar2::gemm_v4::QuantisedPackedWeights *wo_packed = nullptr;
            const void *wo_ptr = nullptr;

            if (opt.wo == "fp32")
            {
                wo_ptr = Wo_fp32.data();
            }
            else if (opt.wo == "q8_1")
            {
                wo_q8_tensor = llaminar2::Q8_1Tensor::quantize_from_fp32(
                    Wo_fp32.data(), {static_cast<size_t>(sc.d_model), static_cast<size_t>(wo_cols)});
                wo_ptr = wo_q8_tensor->q8_1_blocks();
            }
            else
            {
                wo_q8_tensor = llaminar2::Q8_1Tensor::quantize_from_fp32(
                    Wo_fp32.data(), {static_cast<size_t>(sc.d_model), static_cast<size_t>(wo_cols)});
                wo_packed = llaminar::v2::kernels::KernelFactory::ensurePackedWeightsInTensorCache(wo_q8_tensor.get());
                wo_ptr = wo_packed;
            }

            std::vector<float> output(static_cast<size_t>(sc.seq_len_q) * sc.d_model);

            struct SweepConfig
            {
                std::string work;
                std::string tile;
                bool is_auto;
            };

            std::vector<SweepConfig> configs;
            configs.reserve(work_opts.size() * tile_opts.size() + 1);
            for (const auto &w : work_opts)
            {
                for (const auto &t : tile_opts)
                {
                    configs.push_back(SweepConfig{w, t, false});
                }
            }
            configs.push_back(SweepConfig{"auto", "auto", true});

            // Shuffle config order to avoid systematic bias (e.g., AUTO always last).
            std::shuffle(configs.begin(), configs.end(), rng);

            for (const auto &cfg_sel : configs)
            {
                run_opt.work = cfg_sel.work;
                run_opt.tile = cfg_sel.tile;

                double best_ms = std::numeric_limits<double>::infinity();
                BenchResult best_r;
                for (int rep = 0; rep < opt.sweep_repeats; ++rep)
                {
                    const auto r = run_one(sc, run_opt, wo_ptr,
                                           Q_blocks.data(), K_blocks.data(), V_blocks.data(),
                                           output.data());
                    if (r.per_iter_ms < best_ms)
                    {
                        best_ms = r.per_iter_ms;
                        best_r = r;
                    }
                }

                best_r.is_auto = cfg_sel.is_auto;
                std::cout << best_r.case_name << "," << best_r.kv << "," << best_r.work << "," << best_r.tile << "," << best_r.per_iter_ms << "," << (best_r.is_auto ? 1 : 0) << "\n";
                std::cout.flush();
                all_results.push_back(best_r);
            }
        }

        // ============== Summary analysis ==============
        std::cerr << "\n========== SWEEP SUMMARY ==========\n";

        // Group by case
        std::map<std::string, std::vector<BenchResult>> by_case;
        for (const auto &r : all_results)
        {
            by_case[r.case_name].push_back(r);
        }

        std::cerr << "\nAUTO vs BEST_FIXED per case:\n";
        std::cerr << std::fixed << std::setprecision(3);
        for (const auto &[case_name, results] : by_case)
        {
            const BenchResult *auto_r = nullptr;
            const BenchResult *best_fixed = nullptr;
            for (const auto &r : results)
            {
                if (r.is_auto)
                {
                    auto_r = &r;
                }
                else
                {
                    if (!best_fixed || r.per_iter_ms < best_fixed->per_iter_ms)
                    {
                        best_fixed = &r;
                    }
                }
            }
            if (auto_r && best_fixed)
            {
                bool match = (auto_r->work == best_fixed->work && auto_r->tile == best_fixed->tile);
                double delta = auto_r->per_iter_ms - best_fixed->per_iter_ms;
                double pct = (best_fixed->per_iter_ms > 0) ? (delta / best_fixed->per_iter_ms * 100.0) : 0.0;
                std::cerr << "  " << case_name << ": AUTO " << auto_r->work << "/t" << auto_r->tile
                          << " " << auto_r->per_iter_ms << " | BEST " << best_fixed->work << "/t" << best_fixed->tile
                          << " " << best_fixed->per_iter_ms << " | match=" << (match ? "yes" : "NO ")
                          << " | " << std::showpos << delta << "ms (" << pct << "%)" << std::noshowpos << "\n";
            }
        }

        // Best per KV averaged across cases (fixed only)
        std::map<std::tuple<int, std::string, int>, std::vector<double>> agg_fixed;
        for (const auto &r : all_results)
        {
            if (!r.is_auto)
            {
                agg_fixed[{r.kv, r.work, r.tile}].push_back(r.per_iter_ms);
            }
        }

        std::map<int, std::tuple<std::string, int, double>> best_per_kv;
        for (const auto &[key, vals] : agg_fixed)
        {
            auto [kv, work, tile] = key;
            double avg = 0.0;
            for (double v : vals)
                avg += v;
            avg /= static_cast<double>(vals.size());

            auto it = best_per_kv.find(kv);
            if (it == best_per_kv.end() || avg < std::get<2>(it->second))
            {
                best_per_kv[kv] = {work, tile, avg};
            }
        }

        std::cerr << "\nBest per KV (avg across cases, fixed only):\n";
        for (const auto &[kv, best] : best_per_kv)
        {
            auto [work, tile, avg_ms] = best;
            std::cerr << "  kv=" << kv << ": " << work << "/t" << tile << " avg_ms=" << avg_ms << "\n";
        }

        // AUTO per KV averaged
        std::map<std::tuple<int, std::string, int>, std::vector<double>> agg_auto;
        for (const auto &r : all_results)
        {
            if (r.is_auto)
            {
                agg_auto[{r.kv, r.work, r.tile}].push_back(r.per_iter_ms);
            }
        }

        std::cerr << "\nAUTO per KV (avg across cases):\n";
        for (const auto &[key, vals] : agg_auto)
        {
            auto [kv, work, tile] = key;
            double avg = 0.0;
            for (double v : vals)
                avg += v;
            avg /= static_cast<double>(vals.size());
            std::cerr << "  kv=" << kv << ": " << work << "/t" << tile << " avg_ms=" << avg << "\n";
        }

        std::cerr << "===================================\n";
        return 0;
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));

    std::mt19937 rng(opt.seed);
    std::uniform_real_distribution<float> dist_qkv(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_wo(-0.1f, 0.1f);

    const size_t Q_elements = static_cast<size_t>(cfg.seq_len_q) * cfg.num_heads * cfg.head_dim;
    const size_t K_elements = static_cast<size_t>(cfg.seq_len_kv) * cfg.num_kv_heads * cfg.head_dim;
    const size_t V_elements = static_cast<size_t>(cfg.seq_len_kv) * cfg.num_kv_heads * cfg.head_dim;

    std::vector<float> Q_fp32(Q_elements);
    std::vector<float> K_fp32(K_elements);
    std::vector<float> V_fp32(V_elements);
    for (auto &v : Q_fp32)
        v = dist_qkv(rng);
    for (auto &v : K_fp32)
        v = dist_qkv(rng);
    for (auto &v : V_fp32)
        v = dist_qkv(rng);

    std::vector<Q8_1Block> Q_blocks, K_blocks, V_blocks;
    generate_q8_1_blocks(Q_blocks, Q_fp32);
    generate_q8_1_blocks(K_blocks, K_fp32);
    generate_q8_1_blocks(V_blocks, V_fp32);

    // Wo layout: [d_model, num_heads * head_dim]
    const int wo_cols = cfg.num_heads * cfg.head_dim;
    const size_t Wo_elements = static_cast<size_t>(cfg.d_model) * static_cast<size_t>(wo_cols);

    std::vector<float> Wo_fp32(Wo_elements);
    for (auto &v : Wo_fp32)
        v = dist_wo(rng);

    // Wo in this kernel is always [d_model, d_model] for the modeled shapes (d_model == num_heads*head_dim).
    // For Q8_1 and packed-VNNI modes, use the canonical Q8_1Tensor quantizer + packer so layout matches
    // production paths.
    std::shared_ptr<llaminar2::Q8_1Tensor> wo_q8_tensor;
    const llaminar2::gemm_v4::QuantisedPackedWeights *wo_packed = nullptr;
    const void *wo_ptr = nullptr;

    if (opt.wo == "fp32")
    {
        wo_ptr = Wo_fp32.data();
    }
    else if (opt.wo == "q8_1")
    {
        wo_q8_tensor = llaminar2::Q8_1Tensor::quantize_from_fp32(Wo_fp32.data(), {static_cast<size_t>(cfg.d_model), static_cast<size_t>(wo_cols)});
        wo_ptr = wo_q8_tensor->q8_1_blocks();
    }
    else
    {
        wo_q8_tensor = llaminar2::Q8_1Tensor::quantize_from_fp32(Wo_fp32.data(), {static_cast<size_t>(cfg.d_model), static_cast<size_t>(wo_cols)});
        wo_packed = llaminar::v2::kernels::KernelFactory::ensurePackedWeightsInTensorCache(wo_q8_tensor.get());
        wo_ptr = wo_packed;
    }

    std::vector<float> output(static_cast<size_t>(cfg.seq_len_q) * cfg.d_model);

    BenchResult r = run_one(cfg, opt, wo_ptr,
                            Q_blocks.data(), K_blocks.data(), V_blocks.data(),
                            output.data());

    std::cout << "Case: " << cfg.name << "\n";
    std::cout << "  seq_q=" << cfg.seq_len_q << ", kv=" << cfg.seq_len_kv
              << ", heads=" << cfg.num_heads << "/" << cfg.num_kv_heads
              << ", head_dim=" << cfg.head_dim << ", d_model=" << cfg.d_model << "\n";
    std::cout << "  fa=" << opt.fa << ", wo=" << opt.wo
              << ", work=" << r.work << ", tile=" << r.tile << "\n";
    std::cout << "  warmup_iters=" << opt.warmup_iters << ", iters=" << opt.bench_iters << "\n";
    std::cout << std::fixed << std::setprecision(3)
              << "  per_iter_ms=" << r.per_iter_ms << "\n";

    volatile float sink = output[0];
    (void)sink;
    return 0;
}
