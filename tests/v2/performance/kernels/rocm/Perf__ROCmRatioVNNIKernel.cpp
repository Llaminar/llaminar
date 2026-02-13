#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <array>
#include <string>
#include <functional>

#include "fort.hpp"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

extern "C"
{
    bool rocmGemv_int8_int8_int32_vnni(
        const int8_t *d_A_int8,
        const int8_t *d_B_int8_vnni,
        int32_t *d_C_int32,
        int N, int K,
        int device_id, void *stream);

    bool rocmGemv_ratio_vnni_int8_int32(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const int8_t *d_ratio,
        int32_t *d_C_int32,
        int N, int K,
        uint8_t bitwidth,
        uint8_t codebook_id,
        uint8_t has_min,
        uint8_t block_size,
        uint16_t payload_bytes,
        int device_id, void *stream);

    void rocmGemv_ratio_vnni_set_tuning_overrides(
        int linear_tn,
        int linear_kb,
        int iq4_tn,
        int iq4_kb);

    void rocmGemv_ratio_vnni_reset_tuning_overrides();
}

namespace
{
    constexpr uint8_t CODEBOOK_LINEAR = 0;
    constexpr uint8_t CODEBOOK_IQ4 = 4;

    constexpr int8_t kIq4LutI8[16] = {
        -127, -104, -83, -65,
        -49, -35, -22, -10,
        1, 13, 25, 38,
        53, 69, 89, 113};

    inline void decodeQ4Group(
        const uint8_t *payload16,
        int group,
        int8_t ratio,
        uint8_t codebook,
        int8_t out[4])
    {
        const int byte_base = (group & 3) * 4;
        const bool high = (group >= 4);
        for (int i = 0; i < 4; ++i)
        {
            const uint8_t q = payload16[byte_base + i];
            const uint8_t idx = high ? static_cast<uint8_t>(q >> 4) : static_cast<uint8_t>(q & 0x0F);
            const int decoded = (codebook == CODEBOOK_IQ4) ? static_cast<int>(kIq4LutI8[idx]) : (static_cast<int>(idx) - 8);
            out[i] = static_cast<int8_t>((static_cast<int>(ratio) * decoded + 64) >> 7);
        }
    }

    struct Shape
    {
        std::string name;
        int N;
        int K;
    };

    struct Timings
    {
        double mean_ms = 0.0;
        double min_ms = 0.0;
        double max_ms = 0.0;
    };

    class ROCmRatioVNNIPerfTest : public ::testing::Test
    {
    protected:
        int device_id_ = 0;

        void SetUp() override
        {
#ifdef HAVE_ROCM
            int count = 0;
            if (hipGetDeviceCount(&count) != hipSuccess || count <= 0)
            {
                GTEST_SKIP() << "No ROCm device";
            }
            hipSetDevice(device_id_);
#else
            GTEST_SKIP() << "ROCm not enabled";
#endif
        }

#ifdef HAVE_ROCM
        static Timings summarize(const std::vector<float> &samples)
        {
            Timings t;
            if (samples.empty())
                return t;
            std::vector<float> sorted = samples;
            std::sort(sorted.begin(), sorted.end());
            t.min_ms = sorted.front();
            t.max_ms = sorted.back();
            t.mean_ms = std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(sorted.size());
            return t;
        }

        static Timings benchKernel(
            int warmup,
            int runs,
            const std::function<bool(void)> &launch)
        {
            std::vector<float> samples;
            samples.reserve(static_cast<size_t>(runs));

            hipEvent_t start_ev = nullptr;
            hipEvent_t stop_ev = nullptr;
            hipEventCreate(&start_ev);
            hipEventCreate(&stop_ev);

            for (int i = 0; i < warmup; ++i)
            {
                if (!launch())
                {
                    hipEventDestroy(start_ev);
                    hipEventDestroy(stop_ev);
                    return {};
                }
                hipDeviceSynchronize();
            }

            for (int i = 0; i < runs; ++i)
            {
                hipEventRecord(start_ev);
                if (!launch())
                {
                    hipEventDestroy(start_ev);
                    hipEventDestroy(stop_ev);
                    return {};
                }
                hipEventRecord(stop_ev);
                hipEventSynchronize(stop_ev);
                float ms = 0.0f;
                hipEventElapsedTime(&ms, start_ev, stop_ev);
                samples.push_back(ms);
            }

            hipEventDestroy(start_ev);
            hipEventDestroy(stop_ev);
            return summarize(samples);
        }
#endif
    };

    TEST_F(ROCmRatioVNNIPerfTest, Phase1Q4AndIQ4SpeedupVsInt8VNNI)
    {
#ifdef HAVE_ROCM
        const std::vector<Shape> shapes = {
            {"Q/Wo 3584x3584", 3584, 3584},
            {"FFN Down 3584x18944", 3584, 18944},
            {"FFN Gate 18944x3584", 18944, 3584},
        };

        const std::vector<std::pair<std::string, uint8_t>> formats = {
            {"Q4_0", CODEBOOK_LINEAR},
            {"IQ4_NL", CODEBOOK_IQ4},
        };

        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> qdist(0, 255);
        std::uniform_int_distribution<int> rdist(-127, 127);
        std::uniform_int_distribution<int> adist(-127, 127);

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header << "Format" << "Shape" << "INT8 ms" << "Ratio ms" << "Speedup" << fort::endr;

        double speedup_sum = 0.0;
        int speedup_count = 0;
        double q4_speedup_sum = 0.0;
        int q4_speedup_count = 0;
        double iq4_speedup_sum = 0.0;
        int iq4_speedup_count = 0;

        for (const auto &fmt : formats)
        {
            for (const auto &shape : shapes)
            {
                const int N = shape.N;
                const int K = shape.K;
                const int blocks = K / 32;
                const int kgroups = K / 4;

                std::vector<int8_t> h_A(static_cast<size_t>(K));
                for (auto &v : h_A)
                    v = static_cast<int8_t>(adist(rng));

                std::vector<uint8_t> h_payload(static_cast<size_t>(blocks) * N * 16);
                std::vector<int8_t> h_ratio(static_cast<size_t>(blocks) * N);
                for (auto &v : h_payload)
                    v = static_cast<uint8_t>(qdist(rng));
                for (auto &v : h_ratio)
                {
                    int r = rdist(rng);
                    if (r == 0)
                        r = 1;
                    v = static_cast<int8_t>(r);
                }

                std::vector<int8_t> h_int8_vnni(static_cast<size_t>(kgroups) * N * 4);
                for (int n = 0; n < N; ++n)
                {
                    for (int b = 0; b < blocks; ++b)
                    {
                        const size_t linear = static_cast<size_t>(b) * N + static_cast<size_t>(n);
                        const uint8_t *p16 = h_payload.data() + linear * 16;
                        const int8_t r = h_ratio[linear];
                        for (int g = 0; g < 8; ++g)
                        {
                            int8_t w[4];
                            decodeQ4Group(p16, g, r, fmt.second, w);
                            const int kg = b * 8 + g;
                            const size_t dst = (static_cast<size_t>(kg) * N + static_cast<size_t>(n)) * 4;
                            h_int8_vnni[dst + 0] = w[0];
                            h_int8_vnni[dst + 1] = w[1];
                            h_int8_vnni[dst + 2] = w[2];
                            h_int8_vnni[dst + 3] = w[3];
                        }
                    }
                }

                int8_t *d_A = nullptr;
                int8_t *d_B_vnni = nullptr;
                uint8_t *d_payload = nullptr;
                int8_t *d_ratio = nullptr;
                int32_t *d_C = nullptr;

                ASSERT_EQ(hipMalloc(&d_A, static_cast<size_t>(K) * sizeof(int8_t)), hipSuccess);
                ASSERT_EQ(hipMalloc(&d_B_vnni, h_int8_vnni.size() * sizeof(int8_t)), hipSuccess);
                ASSERT_EQ(hipMalloc(&d_payload, h_payload.size() * sizeof(uint8_t)), hipSuccess);
                ASSERT_EQ(hipMalloc(&d_ratio, h_ratio.size() * sizeof(int8_t)), hipSuccess);
                ASSERT_EQ(hipMalloc(&d_C, static_cast<size_t>(N) * sizeof(int32_t)), hipSuccess);

                ASSERT_EQ(hipMemcpy(d_A, h_A.data(), static_cast<size_t>(K) * sizeof(int8_t), hipMemcpyHostToDevice), hipSuccess);
                ASSERT_EQ(hipMemcpy(d_B_vnni, h_int8_vnni.data(), h_int8_vnni.size() * sizeof(int8_t), hipMemcpyHostToDevice), hipSuccess);
                ASSERT_EQ(hipMemcpy(d_payload, h_payload.data(), h_payload.size() * sizeof(uint8_t), hipMemcpyHostToDevice), hipSuccess);
                ASSERT_EQ(hipMemcpy(d_ratio, h_ratio.data(), h_ratio.size() * sizeof(int8_t), hipMemcpyHostToDevice), hipSuccess);

                const auto int8_timing = benchKernel(5, 30, [&]()
                                                     { return rocmGemv_int8_int8_int32_vnni(d_A, d_B_vnni, d_C, N, K, device_id_, nullptr); });

                const auto ratio_timing = benchKernel(5, 30, [&]()
                                                      { return rocmGemv_ratio_vnni_int8_int32(
                                                            d_A,
                                                            d_payload,
                                                            d_ratio,
                                                            d_C,
                                                            N, K,
                                                            4,
                                                            fmt.second,
                                                            0,
                                                            32,
                                                            16,
                                                            device_id_,
                                                            nullptr); });

                ASSERT_GT(int8_timing.mean_ms, 0.0);
                ASSERT_GT(ratio_timing.mean_ms, 0.0);

                const double speedup = int8_timing.mean_ms / std::max(1e-9, ratio_timing.mean_ms);
                speedup_sum += speedup;
                speedup_count++;
                if (fmt.second == CODEBOOK_LINEAR)
                {
                    q4_speedup_sum += speedup;
                    q4_speedup_count++;
                }
                else if (fmt.second == CODEBOOK_IQ4)
                {
                    iq4_speedup_sum += speedup;
                    iq4_speedup_count++;
                }

                table << fmt.first
                      << shape.name
                      << int8_timing.mean_ms
                      << ratio_timing.mean_ms
                      << speedup
                      << fort::endr;

                hipFree(d_A);
                hipFree(d_B_vnni);
                hipFree(d_payload);
                hipFree(d_ratio);
                hipFree(d_C);
            }
        }

        std::cout << "\n"
                  << table.to_string() << std::endl;

        const double avg_speedup = speedup_sum / static_cast<double>(std::max(1, speedup_count));
        const double q4_avg_speedup = q4_speedup_sum / static_cast<double>(std::max(1, q4_speedup_count));
        const double iq4_avg_speedup = iq4_speedup_sum / static_cast<double>(std::max(1, iq4_speedup_count));
        std::cout << "Average ratio/int8 speedup: " << avg_speedup << "x" << std::endl;
        std::cout << "Q4_0 average speedup: " << q4_avg_speedup << "x" << std::endl;
        std::cout << "IQ4_NL average speedup: " << iq4_avg_speedup << "x" << std::endl;

        EXPECT_GE(q4_avg_speedup, 1.55);
        EXPECT_GE(iq4_avg_speedup, 1.38);
#endif
    }

    TEST_F(ROCmRatioVNNIPerfTest, QWoAutoSweepTnKb)
    {
#ifdef HAVE_ROCM
        const char *run_sweep = std::getenv("LLAMINAR_RUN_QWO_AUTOSWEEP");
        if (!run_sweep || std::string(run_sweep) != "1")
        {
            GTEST_SKIP() << "Set LLAMINAR_RUN_QWO_AUTOSWEEP=1 to run Q/Wo autosweep";
        }

        const int N = 3584;
        const int K = 3584;
        const int blocks = K / 32;
        const int kgroups = K / 4;

        const std::array<int, 2> tn_candidates = {128, 256};
        const std::array<int, 5> kb_candidates = {8, 10, 12, 14, 16};

        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> qdist(0, 255);
        std::uniform_int_distribution<int> rdist(-127, 127);
        std::uniform_int_distribution<int> adist(-127, 127);

        std::vector<int8_t> h_A(static_cast<size_t>(K));
        for (auto &v : h_A)
            v = static_cast<int8_t>(adist(rng));

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header << "Format" << "TN" << "KB" << "INT8 ms" << "Ratio ms" << "Speedup" << fort::endr;

        const std::vector<std::pair<std::string, uint8_t>> formats = {
            {"Q4_0", CODEBOOK_LINEAR},
            {"IQ4_NL", CODEBOOK_IQ4},
        };

        for (const auto &fmt : formats)
        {
            std::vector<uint8_t> h_payload(static_cast<size_t>(blocks) * N * 16);
            std::vector<int8_t> h_ratio(static_cast<size_t>(blocks) * N);
            for (auto &v : h_payload)
                v = static_cast<uint8_t>(qdist(rng));
            for (auto &v : h_ratio)
            {
                int r = rdist(rng);
                if (r == 0)
                    r = 1;
                v = static_cast<int8_t>(r);
            }

            std::vector<int8_t> h_int8_vnni(static_cast<size_t>(kgroups) * N * 4);
            for (int n = 0; n < N; ++n)
            {
                for (int b = 0; b < blocks; ++b)
                {
                    const size_t linear = static_cast<size_t>(b) * N + static_cast<size_t>(n);
                    const uint8_t *p16 = h_payload.data() + linear * 16;
                    const int8_t r = h_ratio[linear];
                    for (int g = 0; g < 8; ++g)
                    {
                        int8_t w[4];
                        decodeQ4Group(p16, g, r, fmt.second, w);
                        const int kg = b * 8 + g;
                        const size_t dst = (static_cast<size_t>(kg) * N + static_cast<size_t>(n)) * 4;
                        h_int8_vnni[dst + 0] = w[0];
                        h_int8_vnni[dst + 1] = w[1];
                        h_int8_vnni[dst + 2] = w[2];
                        h_int8_vnni[dst + 3] = w[3];
                    }
                }
            }

            int8_t *d_A = nullptr;
            int8_t *d_B_vnni = nullptr;
            uint8_t *d_payload = nullptr;
            int8_t *d_ratio = nullptr;
            int32_t *d_C = nullptr;

            ASSERT_EQ(hipMalloc(&d_A, static_cast<size_t>(K) * sizeof(int8_t)), hipSuccess);
            ASSERT_EQ(hipMalloc(&d_B_vnni, h_int8_vnni.size() * sizeof(int8_t)), hipSuccess);
            ASSERT_EQ(hipMalloc(&d_payload, h_payload.size() * sizeof(uint8_t)), hipSuccess);
            ASSERT_EQ(hipMalloc(&d_ratio, h_ratio.size() * sizeof(int8_t)), hipSuccess);
            ASSERT_EQ(hipMalloc(&d_C, static_cast<size_t>(N) * sizeof(int32_t)), hipSuccess);

            ASSERT_EQ(hipMemcpy(d_A, h_A.data(), static_cast<size_t>(K) * sizeof(int8_t), hipMemcpyHostToDevice), hipSuccess);
            ASSERT_EQ(hipMemcpy(d_B_vnni, h_int8_vnni.data(), h_int8_vnni.size() * sizeof(int8_t), hipMemcpyHostToDevice), hipSuccess);
            ASSERT_EQ(hipMemcpy(d_payload, h_payload.data(), h_payload.size() * sizeof(uint8_t), hipMemcpyHostToDevice), hipSuccess);
            ASSERT_EQ(hipMemcpy(d_ratio, h_ratio.data(), h_ratio.size() * sizeof(int8_t), hipMemcpyHostToDevice), hipSuccess);

            rocmGemv_ratio_vnni_reset_tuning_overrides();
            const auto int8_timing = benchKernel(5, 30, [&]()
                                                 { return rocmGemv_int8_int8_int32_vnni(d_A, d_B_vnni, d_C, N, K, device_id_, nullptr); });
            ASSERT_GT(int8_timing.mean_ms, 0.0);

            double best_speedup = -1.0;
            int best_tn = -1;
            int best_kb = -1;

            for (const int tn : tn_candidates)
            {
                for (const int kb : kb_candidates)
                {
                    if (fmt.second == CODEBOOK_LINEAR)
                    {
                        rocmGemv_ratio_vnni_set_tuning_overrides(tn, kb, -1, -1);
                    }
                    else
                    {
                        rocmGemv_ratio_vnni_set_tuning_overrides(-1, -1, tn, kb);
                    }

                    const auto ratio_timing = benchKernel(3, 15, [&]()
                                                          { return rocmGemv_ratio_vnni_int8_int32(
                                                                d_A,
                                                                d_payload,
                                                                d_ratio,
                                                                d_C,
                                                                N, K,
                                                                4,
                                                                fmt.second,
                                                                0,
                                                                32,
                                                                16,
                                                                device_id_,
                                                                nullptr); });

                    ASSERT_GT(ratio_timing.mean_ms, 0.0);
                    const double speedup = int8_timing.mean_ms / std::max(1e-9, ratio_timing.mean_ms);

                    table << fmt.first
                          << tn
                          << kb
                          << int8_timing.mean_ms
                          << ratio_timing.mean_ms
                          << speedup
                          << fort::endr;

                    if (speedup > best_speedup)
                    {
                        best_speedup = speedup;
                        best_tn = tn;
                        best_kb = kb;
                    }
                }
            }

            std::cout << "[AUTOSWEEP] " << fmt.first << " best_tn=" << best_tn
                      << " best_kb=" << best_kb << " best_speedup=" << best_speedup << "x" << std::endl;

            rocmGemv_ratio_vnni_reset_tuning_overrides();
            hipFree(d_A);
            hipFree(d_B_vnni);
            hipFree(d_payload);
            hipFree(d_ratio);
            hipFree(d_C);
        }

        std::cout << "\n[Q/Wo autosweep]" << std::endl;
        std::cout << table.to_string() << std::endl;
#endif
    }
}
