/**
 * @file Perf__ROCmFlashAttentionPrefill.cpp
 * @brief Performance benchmarks for ROCm Flash Attention prefill kernel
 *
 * Measures prefill attention latency across configurations that mirror
 * tensor-parallel slicing scenarios (TP=1, TP=2, TP=4).
 *
 * **Tuning Vectors**:
 * - KV type: FP32, FP16, Q8_1
 * - n_heads: TP-sliced head counts
 * - seq_len: 128, 256, 512, 1024 (typical prefill lengths)
 *
 * **Key insight**: Unlike decode (split-K), the prefill kernel has NO KV splitting.
 * Grid = (n_heads, num_q_tiles, batch). Occupancy comes from q-tiles × heads.
 * At TP=2 with Qwen-7B (14 heads), shorter seq_len may under-saturate CUs.
 *
 * @date June 2026
 */

#include <gtest/gtest.h>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <cstdio>
#include <cstdlib>

#include "fort.hpp"

// Extern C wrappers from ROCmFlashAttentionKernels.hip
extern "C"
{
    int hipFlashAttn_prefill_fa2(
        const float *Q, const float *K, const float *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const void *device_params,
        const float *mask,
        void *stream);

    int hipFlashAttn_prefill_fa2_fp16(
        const float *Q, const void *K, const void *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const void *device_params,
        const float *mask,
        void *stream);

    int hipFlashAttn_prefill_fa2_q8_1(
        const float *Q, const void *K, const void *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const void *device_params,
        const float *mask,
        void *stream);
}

namespace
{

    // ============================================================================
    // Constants
    // ============================================================================

    constexpr int WARMUP_ITERS = 10;
    constexpr int BENCH_ITERS = 50;

    // Q8_1Block layout: 36 bytes per 32 elements
    constexpr int Q8_1_BLOCK_SIZE = 32;
    constexpr int Q8_1_BLOCK_BYTES = 36;

    // ============================================================================
    // Model configurations
    // ============================================================================

    struct ModelConfig
    {
        const char *name;
        int n_heads;
        int n_kv_heads;
        int head_dim;
    };

    static constexpr ModelConfig kQwen05B = {"Qwen2.5-0.5B", 14, 2, 64};
    static constexpr ModelConfig kQwen3B = {"Qwen2.5-3B", 16, 2, 128};
    static constexpr ModelConfig kQwen7B = {"Qwen2.5-7B", 28, 4, 128};
    static constexpr ModelConfig kQwen14B = {"Qwen2.5-14B", 40, 8, 128};
    static constexpr ModelConfig kQwen32B = {"Qwen2.5-32B", 40, 8, 128};

    static constexpr ModelConfig kAllModels[] = {
        kQwen05B, kQwen3B, kQwen7B, kQwen14B, kQwen32B};

    // ============================================================================
    // KV type enumeration
    // ============================================================================

    enum class KVType
    {
        FP32,
        FP16,
        Q8_1
    };

    const char *kvTypeName(KVType t)
    {
        switch (t)
        {
        case KVType::FP32:
            return "FP32";
        case KVType::FP16:
            return "FP16";
        case KVType::Q8_1:
            return "Q8_1";
        }
        return "?";
    }

    // ============================================================================
    // TP shape derivation
    // ============================================================================

    struct TPConfig
    {
        const char *label;
        int tp_degree;
        int n_heads;
        int n_kv_heads;
        int head_dim;
    };

    static std::vector<TPConfig> getTPConfigs(const ModelConfig &model, const std::vector<int> &tp_degrees)
    {
        std::vector<TPConfig> configs;
        for (int tp : tp_degrees)
        {
            int local_heads = model.n_heads / tp;
            int local_kv_heads = std::max(1, model.n_kv_heads / tp);
            if (local_heads < 1)
                continue;

            char label[64];
            snprintf(label, sizeof(label), "TP=%d (%dh/%dkv)", tp, local_heads, local_kv_heads);
            configs.push_back({strdup(label), tp, local_heads, local_kv_heads, model.head_dim});
        }
        return configs;
    }

    // ============================================================================
    // Benchmark result
    // ============================================================================

    struct PrefillResult
    {
        double median_us;
        double min_us;
        double max_us;
        bool success;
    };

    // ============================================================================
    // Test fixture
    // ============================================================================

    class ROCmFlashAttentionPrefillPerf : public ::testing::Test
    {
    protected:
        int device_id_ = 0;
        int num_cus_ = 0;
        std::string device_name_;
        bool has_device_ = false;

        void SetUp() override
        {
#ifdef HAVE_ROCM
            int count = 0;
            hipError_t err = hipGetDeviceCount(&count);
            has_device_ = (err == hipSuccess && count > 0);
            if (has_device_)
            {
                (void)hipSetDevice(device_id_);
                hipDeviceProp_t props;
                (void)hipGetDeviceProperties(&props, device_id_);
                device_name_ = std::string(props.name) + " (" + props.gcnArchName + ")";
                num_cus_ = props.multiProcessorCount;
            }
#endif
        }

        // =========================================================================
        // Core benchmark: runs flash prefill with specified KV type
        // =========================================================================
        PrefillResult benchmarkPrefill(
            int n_heads, int n_kv_heads, int head_dim,
            int seq_len, int kv_len,
            KVType kv_type)
        {
            PrefillResult result{};
#ifndef HAVE_ROCM
            return result;
#else
            const int batch_size = 1;
            const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
            const size_t out_size = q_size;

            // KV size depends on type
            const size_t kv_elements = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;
            size_t k_bytes = 0, v_bytes = 0;
            switch (kv_type)
            {
            case KVType::FP32:
                k_bytes = kv_elements * sizeof(float);
                v_bytes = kv_elements * sizeof(float);
                break;
            case KVType::FP16:
                k_bytes = kv_elements * sizeof(uint16_t);
                v_bytes = kv_elements * sizeof(uint16_t);
                break;
            case KVType::Q8_1:
            {
                // Q8_1: 36 bytes per 32 elements
                size_t blocks_per_row = (head_dim + Q8_1_BLOCK_SIZE - 1) / Q8_1_BLOCK_SIZE;
                size_t total_blocks = static_cast<size_t>(kv_len) * n_kv_heads * blocks_per_row;
                k_bytes = total_blocks * Q8_1_BLOCK_BYTES;
                v_bytes = total_blocks * Q8_1_BLOCK_BYTES;
                break;
            }
            }

            // Allocate device memory
            float *d_Q = nullptr, *d_O = nullptr;
            void *d_K = nullptr, *d_V = nullptr;

            (void)hipMalloc(&d_Q, q_size * sizeof(float));
            (void)hipMalloc(&d_K, k_bytes);
            (void)hipMalloc(&d_V, v_bytes);
            (void)hipMalloc(&d_O, out_size * sizeof(float));

            // Initialize Q with random data
            {
                std::vector<float> h_Q(q_size);
                std::mt19937 rng(42);
                std::normal_distribution<float> dist(0.0f, 0.1f);
                for (auto &v : h_Q)
                    v = dist(rng);
                (void)hipMemcpy(d_Q, h_Q.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
            }

            // Initialize KV with random data (appropriate format)
            if (kv_type == KVType::FP32)
            {
                std::vector<float> h_KV(kv_elements);
                std::mt19937 rng(123);
                std::normal_distribution<float> dist(0.0f, 0.1f);
                for (auto &v : h_KV)
                    v = dist(rng);
                (void)hipMemcpy(d_K, h_KV.data(), k_bytes, hipMemcpyHostToDevice);
                for (auto &v : h_KV)
                    v = dist(rng);
                (void)hipMemcpy(d_V, h_KV.data(), v_bytes, hipMemcpyHostToDevice);
            }
            else if (kv_type == KVType::FP16)
            {
                // Fill with random uint16_t (FP16 bit patterns)
                std::vector<uint16_t> h_KV(kv_elements);
                std::mt19937 rng(123);
                // Generate small FP16 values: exponent=14 (bias-1), mantissa random
                // This gives values in [-1, 1] range approximately
                for (auto &v : h_KV)
                {
                    int sign = rng() & 1;
                    int exp = 14; // bias=15, so this is 2^(14-15) = 0.5 range
                    int mantissa = rng() & 0x3FF;
                    v = static_cast<uint16_t>((sign << 15) | (exp << 10) | mantissa);
                }
                (void)hipMemcpy(d_K, h_KV.data(), k_bytes, hipMemcpyHostToDevice);
                for (auto &v : h_KV)
                {
                    int sign = rng() & 1;
                    int exp = 14;
                    int mantissa = rng() & 0x3FF;
                    v = static_cast<uint16_t>((sign << 15) | (exp << 10) | mantissa);
                }
                (void)hipMemcpy(d_V, h_KV.data(), v_bytes, hipMemcpyHostToDevice);
            }
            else
            {
                // Q8_1: fill with random block data
                std::vector<uint8_t> h_KV(k_bytes);
                std::mt19937 rng(123);
                for (auto &v : h_KV)
                    v = static_cast<uint8_t>(rng() & 0xFF);
                (void)hipMemcpy(d_K, h_KV.data(), k_bytes, hipMemcpyHostToDevice);
                h_KV.resize(v_bytes);
                for (auto &v : h_KV)
                    v = static_cast<uint8_t>(rng() & 0xFF);
                (void)hipMemcpy(d_V, h_KV.data(), v_bytes, hipMemcpyHostToDevice);
            }
            (void)hipDeviceSynchronize();

            // Lambda to dispatch based on KV type
            auto launch = [&]() -> int
            {
                switch (kv_type)
                {
                case KVType::FP32:
                    return hipFlashAttn_prefill_fa2(
                        d_Q, static_cast<float *>(d_K), static_cast<float *>(d_V), d_O,
                        batch_size, seq_len, kv_len,
                        n_heads, n_kv_heads, head_dim,
                        /*causal=*/true, /*window_size=*/-1, /*position_offset=*/0,
                        nullptr, nullptr, nullptr);
                case KVType::FP16:
                    return hipFlashAttn_prefill_fa2_fp16(
                        d_Q, d_K, d_V, d_O,
                        batch_size, seq_len, kv_len,
                        n_heads, n_kv_heads, head_dim,
                        /*causal=*/true, /*window_size=*/-1, /*position_offset=*/0,
                        nullptr, nullptr, nullptr);
                case KVType::Q8_1:
                    return hipFlashAttn_prefill_fa2_q8_1(
                        d_Q, d_K, d_V, d_O,
                        batch_size, seq_len, kv_len,
                        n_heads, n_kv_heads, head_dim,
                        /*causal=*/true, /*window_size=*/-1, /*position_offset=*/0,
                        nullptr, nullptr, nullptr);
                }
                return -1;
            };

            // Warmup
            for (int i = 0; i < WARMUP_ITERS; ++i)
            {
                int rc = launch();
                if (rc != 0)
                {
                    result.success = false;
                    goto cleanup;
                }
            }
            (void)hipDeviceSynchronize();

            // Benchmark with HIP events
            {
                hipEvent_t ev_start, ev_stop;
                (void)hipEventCreate(&ev_start);
                (void)hipEventCreate(&ev_stop);

                std::vector<double> times_us;
                times_us.reserve(BENCH_ITERS);

                for (int i = 0; i < BENCH_ITERS; ++i)
                {
                    (void)hipDeviceSynchronize();
                    (void)hipEventRecord(ev_start, nullptr);

                    int rc = launch();
                    if (rc != 0)
                    {
                        result.success = false;
                        (void)hipEventDestroy(ev_start);
                        (void)hipEventDestroy(ev_stop);
                        goto cleanup;
                    }

                    (void)hipEventRecord(ev_stop, nullptr);
                    (void)hipEventSynchronize(ev_stop);

                    float ms = 0.0f;
                    (void)hipEventElapsedTime(&ms, ev_start, ev_stop);
                    times_us.push_back(static_cast<double>(ms) * 1000.0);
                }

                (void)hipEventDestroy(ev_start);
                (void)hipEventDestroy(ev_stop);

                std::sort(times_us.begin(), times_us.end());
                result.min_us = times_us.front();
                result.max_us = times_us.back();
                result.median_us = times_us[times_us.size() / 2];
                result.success = true;
            }

        cleanup:
            (void)hipFree(d_Q);
            (void)hipFree(d_K);
            (void)hipFree(d_V);
            (void)hipFree(d_O);

            return result;
#endif
        }

        // =========================================================================
        // KV type comparison for a given model
        // =========================================================================
        void runKVTypeSweep(
            const ModelConfig &model,
            const std::vector<int> &seq_lengths,
            const std::vector<int> &tp_degrees,
            const std::vector<KVType> &kv_types)
        {
#ifndef HAVE_ROCM
            GTEST_SKIP() << "No ROCm support";
#else
            if (!has_device_)
                GTEST_SKIP() << "No ROCm device";

            // Skip Q8_1 and FP16 for head_dim < 64
            std::vector<KVType> valid_types;
            for (auto t : kv_types)
            {
                if (t == KVType::FP16 && model.head_dim < 64)
                    continue;
                if (t == KVType::Q8_1 && (model.head_dim < 64 || model.head_dim % 32 != 0))
                    continue;
                valid_types.push_back(t);
            }

            auto tp_configs = getTPConfigs(model, tp_degrees);

            for (const auto &tc : tp_configs)
            {
                // [kv_idx][seq_idx]
                std::vector<std::vector<PrefillResult>> grid(valid_types.size());
                for (size_t ki = 0; ki < valid_types.size(); ++ki)
                {
                    grid[ki].resize(seq_lengths.size());
                    for (size_t si = 0; si < seq_lengths.size(); ++si)
                    {
                        int seq = seq_lengths[si];
                        auto r = benchmarkPrefill(
                            tc.n_heads, tc.n_kv_heads, tc.head_dim,
                            seq, seq, valid_types[ki]);
                        ASSERT_TRUE(r.success)
                            << model.name << " " << tc.label
                            << " kv=" << kvTypeName(valid_types[ki])
                            << " seq=" << seq;
                        grid[ki][si] = r;
                    }
                }

                // Render comparison table
                fort::utf8_table table;
                table.set_border_style(FT_DOUBLE2_STYLE);

                table << fort::header << "KV Type";
                for (int s : seq_lengths)
                {
                    char colhdr[32];
                    snprintf(colhdr, sizeof(colhdr), "seq=%d", s);
                    table << colhdr;
                }
                table << fort::endr;

                table.column(0).set_cell_text_align(fort::text_align::left);
                for (size_t i = 1; i <= seq_lengths.size(); ++i)
                    table.column(i).set_cell_text_align(fort::text_align::right);

                for (size_t ki = 0; ki < valid_types.size(); ++ki)
                {
                    table << kvTypeName(valid_types[ki]);
                    for (size_t si = 0; si < seq_lengths.size(); ++si)
                    {
                        char cell[32];
                        snprintf(cell, sizeof(cell), "%.1f", grid[ki][si].median_us);
                        table << cell;
                    }
                    table << fort::endr;
                }

                // Speedup vs FP32 row(s)
                if (valid_types.size() > 1)
                {
                    table << fort::separator;
                    for (size_t ki = 1; ki < valid_types.size(); ++ki)
                    {
                        char label[32];
                        snprintf(label, sizeof(label), "%s vs FP32", kvTypeName(valid_types[ki]));
                        table << label;
                        for (size_t si = 0; si < seq_lengths.size(); ++si)
                        {
                            double speedup = grid[0][si].median_us / grid[ki][si].median_us;
                            char cell[32];
                            snprintf(cell, sizeof(cell), "%.2f×", speedup);
                            table << cell;
                        }
                        table << fort::endr;
                    }
                }

                fprintf(stderr,
                        "\n%s %s — KV Type Comparison (μs)\n"
                        "Device: %s (%d CUs)\n%s\n",
                        model.name, tc.label,
                        device_name_.c_str(), num_cus_,
                        table.to_string().c_str());
            }
#endif
        }

        // =========================================================================
        // Table 3: TP scaling efficiency
        // =========================================================================
        void runTPScalingSweep(
            const ModelConfig &model,
            const std::vector<int> &seq_lengths,
            const std::vector<int> &tp_degrees,
            KVType kv_type)
        {
#ifndef HAVE_ROCM
            GTEST_SKIP() << "No ROCm support";
#else
            if (!has_device_)
                GTEST_SKIP() << "No ROCm device";

            auto tp_configs = getTPConfigs(model, tp_degrees);

            // [tp_idx][seq_idx]
            std::vector<std::vector<PrefillResult>> grid(tp_configs.size());
            for (size_t ti = 0; ti < tp_configs.size(); ++ti)
            {
                grid[ti].resize(seq_lengths.size());
                for (size_t si = 0; si < seq_lengths.size(); ++si)
                {
                    int seq = seq_lengths[si];
                    auto r = benchmarkPrefill(
                        tp_configs[ti].n_heads, tp_configs[ti].n_kv_heads,
                        tp_configs[ti].head_dim,
                        seq, seq, kv_type);
                    ASSERT_TRUE(r.success)
                        << model.name << " " << tp_configs[ti].label << " seq=" << seq;
                    grid[ti][si] = r;
                }
            }

            // Latency table
            {
                fort::utf8_table table;
                table.set_border_style(FT_DOUBLE2_STYLE);

                table << fort::header << "Config" << "Heads";
                for (int s : seq_lengths)
                {
                    char colhdr[32];
                    snprintf(colhdr, sizeof(colhdr), "seq=%d", s);
                    table << colhdr;
                }
                table << fort::endr;

                table.column(0).set_cell_text_align(fort::text_align::left);
                table.column(1).set_cell_text_align(fort::text_align::right);
                for (size_t i = 2; i <= seq_lengths.size() + 1; ++i)
                    table.column(i).set_cell_text_align(fort::text_align::right);

                for (size_t ti = 0; ti < tp_configs.size(); ++ti)
                {
                    table << tp_configs[ti].label << tp_configs[ti].n_heads;
                    for (size_t si = 0; si < seq_lengths.size(); ++si)
                    {
                        char cell[32];
                        snprintf(cell, sizeof(cell), "%.1f", grid[ti][si].median_us);
                        table << cell;
                    }
                    table << fort::endr;
                }

                fprintf(stderr,
                        "\n%s KV=%s — Prefill Latency (μs)\n"
                        "Device: %s (%d CUs)\n%s\n",
                        model.name, kvTypeName(kv_type),
                        device_name_.c_str(), num_cus_,
                        table.to_string().c_str());
            }

            // TP scaling efficiency table
            if (tp_configs.size() > 1)
            {
                fort::utf8_table table;
                table.set_border_style(FT_DOUBLE2_STYLE);

                table << fort::header << "Config" << "Heads";
                for (int s : seq_lengths)
                {
                    char colhdr[32];
                    snprintf(colhdr, sizeof(colhdr), "seq=%d", s);
                    table << colhdr;
                }
                table << fort::endr;

                table.column(0).set_cell_text_align(fort::text_align::left);
                table.column(1).set_cell_text_align(fort::text_align::right);
                for (size_t i = 2; i <= seq_lengths.size() + 1; ++i)
                    table.column(i).set_cell_text_align(fort::text_align::right);

                // TP=1 baseline
                table << tp_configs[0].label << tp_configs[0].n_heads;
                for (size_t si = 0; si < seq_lengths.size(); ++si)
                    table << "baseline";
                table << fort::endr;

                // TP>1 scaling efficiency
                for (size_t ti = 1; ti < tp_configs.size(); ++ti)
                {
                    table << tp_configs[ti].label << tp_configs[ti].n_heads;
                    for (size_t si = 0; si < seq_lengths.size(); ++si)
                    {
                        double base_us = grid[0][si].median_us;
                        double tp_us = grid[ti][si].median_us;
                        double ideal_us = base_us / static_cast<double>(tp_configs[ti].tp_degree);
                        double efficiency = (ideal_us / tp_us) * 100.0;

                        char cell[32];
                        snprintf(cell, sizeof(cell), "%.0f%%", efficiency);
                        table << cell;
                    }
                    table << fort::endr;
                }

                fprintf(stderr,
                        "\n%s KV=%s — TP Scaling Efficiency\n%s\n",
                        model.name, kvTypeName(kv_type),
                        table.to_string().c_str());
            }
#endif
        }
    };

    // ============================================================================
    // Test Cases
    // ============================================================================

    // ---------------------------------------------------------------------------
    // KV type comparison for 7B reference
    // ---------------------------------------------------------------------------
    TEST_F(ROCmFlashAttentionPrefillPerf, Qwen7B_KVComparison)
    {
        runKVTypeSweep(
            kQwen7B,
            /*seq_lengths=*/{128, 256, 512, 1024},
            /*tp_degrees=*/{1, 2, 4},
            {KVType::FP32, KVType::FP16, KVType::Q8_1});
    }

    // ---------------------------------------------------------------------------
    // TP scaling efficiency across all model sizes (FP32)
    // Shows how head count affects TP scaling for the attention kernel
    // ---------------------------------------------------------------------------
    TEST_F(ROCmFlashAttentionPrefillPerf, AllModels_TPScaling_FP32)
    {
        for (const auto &model : kAllModels)
        {
            runTPScalingSweep(
                model,
                /*seq_lengths=*/{128, 256, 512, 1024},
                /*tp_degrees=*/{1, 2, 4},
                KVType::FP32);
        }
    }

} // anonymous namespace
