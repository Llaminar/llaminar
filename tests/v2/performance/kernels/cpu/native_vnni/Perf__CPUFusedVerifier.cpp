/**
 * @file Perf__CPUFusedVerifier.cpp
 * @brief Isolate CPU fused-projection scheduling from standalone kernel cost.
 *
 * Both paths use independent prepared weights and the same selected policy;
 * Auto is the default, with explicit row-reuse candidates for diagnosis only.
 * Serial M=1 rows are a diagnostic oracle only. Timing alternates path order
 * and publishes raw samples after complete byte checks. This is a standalone
 * economy diagnostic, not part of production preflight or a policy installer.
 */

#include "../../../../utils/NativeVNNITestPartialStorage.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <omp.h>

#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include "utils/QuantizedVerifierFormats.h"

namespace
{
    using namespace llaminar2;
    using namespace llaminar2::cpu::native_vnni;
    using namespace llaminar2::test;

    /** Read a positive diagnostic geometry without changing production policy. */
    int dimension(const char *name, int default_value)
    {
        const char *text = std::getenv(name);
        if (!text)
            return default_value;
        size_t consumed = 0;
        const int value = std::stoi(text, &consumed);
        if (value <= 0 || consumed != std::strlen(text))
            throw std::invalid_argument(std::string("invalid diagnostic dimension: ") + name);
        return value;
    }

    /** Select a row-reuse experiment using production policy names, never an installed override. */
    VerifierRowsPolicy diagnosticPolicy()
    {
        const char *requested = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_POLICY");
        if (!requested)
            return VerifierRowsPolicy::Auto;
        for (const auto policy : {VerifierRowsPolicy::Auto,
                 VerifierRowsPolicy::Pairwise, VerifierRowsPolicy::WideRows})
            if (std::strcmp(requested, verifierRowsPolicyName(policy)) == 0)
                return policy;
        throw std::invalid_argument("unsupported fused verifier diagnostic policy");
    }

    /** One format from the shared complete source-format registry. */
    class CPUFusedVerifier : public ::testing::TestWithParam<QuantizedVerifierFormatCase> {};

    /** Compare a real two-projection bundle with the same two standalone calls. */
    TEST_P(CPUFusedVerifier, BundleVersusStandalone)
    {
        const auto &format_a = GetParam();
        const auto *format_b = &format_a;
        if (const char *requested = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_FORMAT_B"))
        {
            const auto &formats = quantizedVerifierFormats();
            const auto found = std::find_if(formats.begin(), formats.end(),
                [&](const auto &format) { return std::strcmp(format.label, requested) == 0; });
            ASSERT_NE(found, formats.end());
            format_b = &*found;
        }
        const int m = dimension("LLAMINAR_CPU_FUSED_VERIFIER_M", 16);
        const int n_a = dimension("LLAMINAR_CPU_FUSED_VERIFIER_N_A", 17408);
        const int n_b = dimension("LLAMINAR_CPU_FUSED_VERIFIER_N_B", 17408);
        const int k = dimension("LLAMINAR_CPU_FUSED_VERIFIER_K", 5120);
        const auto policy = diagnosticPolicy();
        ASSERT_GT(m, 1);
        ASSERT_EQ(k % 256, 0);

        auto source_a = format_a.create({static_cast<size_t>(n_a), static_cast<size_t>(k)}, 4242);
        auto source_b = format_b->create({static_cast<size_t>(n_b), static_cast<size_t>(k)}, 4243);
        CPUNativeVNNIGemmKernel kernel_a(source_a.get()), kernel_b(source_b.get());
        ASSERT_TRUE(kernel_a.isValid());
        ASSERT_TRUE(kernel_b.isValid());
        const std::array<const CPUNativeVNNIPackedWeights *, 2> weights = {
            &kernel_a.packedWeights(), &kernel_b.packedWeights()};
        const std::array<int, 2> columns = {n_a, n_b};
        std::array<NativeVNNITestPartialStorage, 2> partial_storage = {
            NativeVNNITestPartialStorage(*weights[0], m), NativeVNNITestPartialStorage(*weights[1], m)};
        std::mt19937 random(4242);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        std::vector<float> input(static_cast<size_t>(m) * k);
        for (auto &value : input)
            value = distribution(random);
        std::vector<Q8_1Block> quantized(static_cast<size_t>(m) * k / 32);
        quantize_activations_to_q8_1(input.data(), quantized.data(), m, k, k / 32);
        std::array<std::vector<float>, 2> fused, separate, serial;
        std::array<FusedVerifierRowsDesc, 2> descriptors;
        for (int projection = 0; projection < 2; ++projection)
        {
            const size_t size = static_cast<size_t>(m) * columns[projection];
            fused[projection].resize(size);
            separate[projection].resize(size);
            serial[projection].resize(size);
            descriptors[projection] = {.packed = weights[projection],
                .output = fused[projection].data(), .bias = nullptr,
                .N = columns[projection], .ldc = columns[projection],
                .verifier_schedule = policy};
            for (int row = 0; row < m; ++row)
                gemv_native_vnni_preq(*weights[projection], quantized.data() + row * (k / 32),
                    serial[projection].data() + row * columns[projection], partial_storage[projection].span());
        }
        llaminar2::test::NativeVNNITestPartialStorage fused_storage(descriptors.data(), 2, m);
        const auto run = [&](bool bundle)
        {
            if (bundle)
            {
                if (!gemm_native_vnni_fused_verifier_rows_preq(quantized.data(),
                        descriptors.data(), fused_storage.span(), 2, m, k / 32))
                    throw std::runtime_error("production fused verifier rejected its descriptors");
            }
            else
                for (int projection = 0; projection < 2; ++projection)
                    gemm_native_vnni_preq_decode_equivalent_rows(*weights[projection],
                        quantized.data(), separate[projection].data(), partial_storage[projection].span(), m, columns[projection],
                        ISAPath::AUTO, policy);
        };
        run(false);
        run(true);
        // Optional diagnostic counters describe the executed routes, not a
        // second reconstruction of the production dispatch decision.
        for (const auto &record : PerfStatsCollector::snapshot({"kernel"}))
        {
            if (record.name.find("launch") == std::string::npos)
                continue;
            std::cout << "route," << record.name;
            for (const auto &[key, value] : record.tags)
                std::cout << ',' << key << '=' << value;
            std::cout << '\n';
        }
        for (int projection = 0; projection < 2; ++projection)
        {
            const size_t bytes = serial[projection].size() * sizeof(float);
            ASSERT_EQ(std::memcmp(serial[projection].data(), fused[projection].data(), bytes), 0);
            ASSERT_EQ(std::memcmp(serial[projection].data(), separate[projection].data(), bytes), 0);
        }

        // A profiler may isolate one path after setup. Paired timing remains
        // the default and is never replaced by profiler-contaminated evidence.
        const char *profile_path = std::getenv("LLAMINAR_CPU_FUSED_VERIFIER_PROFILE_PATH");
        if (profile_path)
        {
            const std::string path(profile_path);
            ASSERT_TRUE(path == "bundle" || path == "separate");
            const int iterations = dimension("LLAMINAR_CPU_FUSED_VERIFIER_ITERS", 1000);
            for (int iteration = 0; iteration < iterations; ++iteration)
                run(path == "bundle");
            return;
        }
        std::cout << "format_a,format_b,m,n_a,n_b,k,threads,policy,sample,first,bundle_us,separate_us\n";
        // Several calls per sample amortize cold thread-team wakeup variance.
        constexpr int repetitions = 16;
        for (int sample = -2; sample < 12; ++sample)
        {
            std::array<double, 2> times{};
            for (int order = 0; order < 2; ++order)
            {
                const bool bundle = ((sample + 2 + order) % 2) == 0;
                const auto start = std::chrono::steady_clock::now();
                for (int repeat = 0; repeat < repetitions; ++repeat)
                    run(bundle);
                times[bundle ? 0 : 1] = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - start).count() / repetitions;
            }
            if (sample >= 0)
                std::cout << format_a.label << ',' << format_b->label << ',' << m << ','
                          << n_a << ',' << n_b << ',' << k << ',' << omp_get_max_threads() << ','
                          << verifierRowsPolicyName(policy) << ','
                          << sample << ',' << (sample % 2 ? "separate" : "bundle") << ','
                          << times[0] << ',' << times[1] << '\n';
        }
    }

    INSTANTIATE_TEST_SUITE_P(AllFormats, CPUFusedVerifier,
        ::testing::ValuesIn(quantizedVerifierFormats()),
        [](const ::testing::TestParamInfo<QuantizedVerifierFormatCase> &info)
        { return std::string(info.param.label); });

    /** Narrow real-model projections retain FP32 activations for every weight dtype. */
    class CPUFloatingVerifier : public ::testing::TestWithParam<TensorType> {};

    /** Time the production narrow grouped entry point after exact M=1 validation. */
    TEST_P(CPUFloatingVerifier, NarrowGroupedRows)
    {
        const int m = dimension("LLAMINAR_CPU_FUSED_VERIFIER_M", 16);
        const int n = dimension("LLAMINAR_CPU_FUSED_VERIFIER_N_A", 48);
        const int k = dimension("LLAMINAR_CPU_FUSED_VERIFIER_K", 5120);
        std::mt19937 random(18448);
        std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
        std::vector<float> input(static_cast<size_t>(m) * k);
        std::vector<float> weights(static_cast<size_t>(n) * k);
        for (float &value : input)
            value = distribution(random);
        for (float &value : weights)
            value = distribution(random);
        std::vector<uint16_t> stored(weights.size());
        const TensorType dtype = GetParam();
        const char *label = dtype == TensorType::FP32 ? "FP32"
            : dtype == TensorType::FP16 ? "FP16" : "BF16";
        for (size_t i = 0; i < weights.size(); ++i)
            stored[i] = dtype == TensorType::BF16
                ? simd::fp32_to_bf16(weights[i]) : fp32_to_fp16(weights[i]);
        std::vector<float> grouped(static_cast<size_t>(m) * n);
        std::vector<float> serial(grouped.size());
        const auto run = [&](const float *a, float *c, int rows)
        {
            switch (dtype)
            {
            case TensorType::FP32:
                return gemm::run_fp32_skinny_matmul(a, weights.data(), c, rows, n, k, true);
            case TensorType::FP16:
                return gemm::run_fp32xfp16_skinny_matmul(a, stored.data(), c, rows, n, k, true);
            case TensorType::BF16:
                return gemm::run_fp32xbf16_skinny_matmul(a, stored.data(), c, rows, n, k, true);
            default:
                throw std::invalid_argument("unsupported floating verifier benchmark dtype");
            }
        };
        for (int row = 0; row < m; ++row)
            ASSERT_TRUE(run(input.data() + row * k, serial.data() + row * n, 1));
        ASSERT_TRUE(run(input.data(), grouped.data(), m));
        ASSERT_EQ(std::memcmp(grouped.data(), serial.data(), serial.size() * sizeof(float)), 0);
        std::cout << "floating_weight,m,n,k,threads,sample,grouped_us\n";
        constexpr int repetitions = 64;
        for (int sample = -2; sample < 12; ++sample)
        {
            const auto start = std::chrono::steady_clock::now();
            bool ok = true;
            for (int repeat = 0; repeat < repetitions; ++repeat)
                ok = run(input.data(), grouped.data(), m) && ok;
            const double elapsed = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - start).count() / repetitions;
            ASSERT_TRUE(ok);
            if (sample >= 0)
                std::cout << label << ',' << m << ',' << n << ',' << k << ','
                          << omp_get_max_threads() << ',' << sample << ',' << elapsed << '\n';
        }
    }

    INSTANTIATE_TEST_SUITE_P(FloatingFormats, CPUFloatingVerifier,
        ::testing::Values(TensorType::FP32, TensorType::FP16, TensorType::BF16),
        [](const ::testing::TestParamInfo<TensorType> &info)
        { return info.param == TensorType::FP32 ? "FP32"
            : info.param == TensorType::FP16 ? "FP16" : "BF16"; });
}
