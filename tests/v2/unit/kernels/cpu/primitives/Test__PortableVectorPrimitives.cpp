/**
 * @file Test__PortableVectorPrimitives.cpp
 * @brief Device-free tests of the production vector primitives on the native ISA.
 *
 * Links the production translation unit directly, without accelerator-bearing
 * core/MPI dependencies. Serial and grouped calls must agree byte-for-byte at
 * every vector/tail boundary; independent double-precision sums prevent two
 * identically broken paths from certifying one another. Explicit failures stay
 * active in Release builds, unlike C assert(). This executable is test-only.
 */
#include "kernels/cpu/primitives/VectorPrimitives.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace
{
    using namespace llaminar2::primitives;
    constexpr float sentinel = -12345.25f;

    /** @brief Fail the test in every build type, with the violated contract. */
    void require(bool condition, const char *message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    /** @brief Compare raw FP32 results, including the sign of zero. */
    bool sameBits(float left, float right)
    {
        return std::bit_cast<std::uint32_t>(left) == std::bit_cast<std::uint32_t>(right);
    }

    /** @brief Generate repeatable, nonuniform finite values without global RNG state. */
    float sample(std::uint32_t &state)
    {
        state = state * 1664525u + 1013904223u;
        const int mantissa = static_cast<int>(state >> 16) - 32768;
        return std::ldexp(static_cast<float>(mantissa), static_cast<int>((state >> 8) % 17) - 20);
    }

    /**
     * @brief Validate one dot result independently of SIMD reduction order.
     * @param weights First n-element input.
     * @param row Second n-element input.
     * @param n Live length, including zero.
     * @param value Observed production result.
     */
    void checkDot(const float *weights, const float *row, int n, float value)
    {
        double expected = 0.0, absolute_sum = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double product = static_cast<double>(weights[i]) * row[i];
            expected += product;
            absolute_sum += std::abs(product);
        }
        // The longest possible serial accumulation is an upper error bound;
        // SIMD paths use shorter chains. Cancellation is measured against the
        // absolute product sum, not an unstable relative-to-result tolerance.
        const double bound = std::numeric_limits<float>::epsilon() *
            (n + 2.0) * absolute_sum;
        require(std::isfinite(value) && std::abs(value - expected) <= bound,
                "dot differs from independent FP64 oracle");
    }

    /** @brief Sweep serial/grouped arithmetic, unaligned rows, strides and untouched padding. */
    void testGroupedDots()
    {
        std::uint32_t state = 42;
        std::vector<int> lengths;
        for (int n = 0; n <= 1025; ++n)
            lengths.push_back(n);
        lengths.insert(lengths.end(), {1536, 2047, 2048, 2049, 4096, 4097});
        for (const int n : lengths)
            for (const std::size_t offset : {0u, 1u, 3u})
            {
                const std::size_t stride = n + 7;
                std::vector<float> weight_storage(n + offset);
                std::vector<float> row_storage(4 * stride + offset);
                for (float &value : weight_storage)
                    value = sample(state);
                for (float &value : row_storage)
                    value = sample(state);
                // Avoid pointer arithmetic on a possibly null empty vector.
                const float *weights = n == 0 ? nullptr : weight_storage.data() + offset;
                const float *rows = row_storage.data() + offset;
                for (const std::size_t output_stride : {1u, 3u, 7u})
                {
                    std::array<float, 32> output;
                    output.fill(sentinel);
                    vec_dot_four_rows(weights, rows, n, stride, output.data() + 1, output_stride);
                    for (std::size_t row = 0; row < 4; ++row)
                    {
                        const float serial = vec_dot(weights, rows + row * stride, n);
                        require(sameBits(serial, output[1 + row * output_stride]),
                                "grouped dot is not serial-row byte equivalent");
                        checkDot(weights, rows + row * stride, n, serial);
                    }
                    for (std::size_t i = 0; i < output.size(); ++i)
                    {
                        const bool live = i >= 1 && (i - 1) % output_stride == 0 &&
                            (i - 1) / output_stride < 4;
                        if (!live)
                            require(sameBits(output[i], sentinel), "grouped dot overwrote output padding");
                    }
                }
            }

        // Cancellation and repeated reuse exercise values not represented by
        // the random sweep and detect accidentally retained accumulator state.
        std::array<float, 257> weights;
        std::array<float, 4 * 257> rows;
        for (std::size_t i = 0; i < weights.size(); ++i)
        {
            weights[i] = i % 2 == 0 ? 4096.0f : -4096.0f;
            for (std::size_t row = 0; row < 4; ++row)
                rows[row * weights.size() + i] = 1.0f + (row + i % 3) * 0.00000011920928955078125f;
        }
        for (int repeat = 0; repeat < 20; ++repeat)
        {
            std::array<float, 4> output;
            output.fill(sentinel);
            vec_dot_four_rows(weights.data(), rows.data(), weights.size(), weights.size(), output.data(), 1);
            for (std::size_t row = 0; row < 4; ++row)
                require(sameBits(output[row], vec_dot(weights.data(), rows.data() + row * weights.size(), weights.size())),
                        "cancellation/reuse changed grouped dot arithmetic");
        }
    }

    /** @brief Check elementwise operations, exact aliasing, vector tails and write bounds. */
    void testElementwise()
    {
        std::uint32_t state = 123;
        for (int n = 0; n <= 513; ++n)
            for (const std::size_t offset : {0u, 1u, 3u})
            {
                std::vector<float> a(n + offset + 1), b(a.size());
                for (float &value : a)
                    value = sample(state);
                for (float &value : b)
                    value = sample(state);
                std::vector<float> output(a.size(), sentinel);
                vec_add(output.data() + offset, a.data() + offset, b.data() + offset, n);
                for (int i = 0; i < n; ++i)
                    require(sameBits(output[offset + i], a[offset + i] + b[offset + i]), "vector add mismatch");
                require(output.back() == sentinel, "vector add overwrote its tail");
                for (std::size_t i = 0; i < offset; ++i)
                    require(output[i] == sentinel, "vector add overwrote its prefix");

                for (const bool alias_second : {false, true})
                {
                    auto left = a, right = b;
                    float *destination = (alias_second ? right.data() : left.data()) + offset;
                    vec_add(destination, left.data() + offset, right.data() + offset, n);
                    for (int i = 0; i < n; ++i)
                        require(sameBits(destination[i], output[offset + i]), "vector add exact-alias mismatch");
                }

                auto scaled = a;
                vec_scale(scaled.data() + offset, -0.7139f, n);
                for (int i = 0; i < n; ++i)
                    require(sameBits(scaled[offset + i], a[offset + i] * -0.7139f), "vector scale mismatch");
                require(sameBits(scaled.back(), a.back()), "vector scale overwrote its tail");

                for (const bool alias : {false, true})
                {
                    auto accumulated = b;
                    const float *source = alias ? accumulated.data() + offset : a.data() + offset;
                    vec_axpy(accumulated.data() + offset, source, 0.7139f, n);
                    for (int i = 0; i < n; ++i)
                    {
                        const float input = alias ? b[offset + i] : a[offset + i];
                        const float expected = std::fma(0.7139f, input, b[offset + i]);
#if defined(__aarch64__) && defined(__ARM_NEON)
                        require(sameBits(accumulated[offset + i], expected), "NEON AXPY lost explicit FMA arithmetic");
#else
                        require(std::abs(accumulated[offset + i] - expected) <=
                                    std::numeric_limits<float>::epsilon() * std::max(1.0f, std::abs(expected)),
                                "AXPY mismatch");
#endif
                    }
                    require(sameBits(accumulated.back(), b.back()), "AXPY overwrote its tail");
                }
            }
    }

    /** @brief Zero-length operations must not dereference null input or elementwise output. */
    void testEmptyAndSpecialValues()
    {
        require(sameBits(vec_dot(nullptr, nullptr, 0), 0.0f), "empty dot must return positive zero");
        vec_axpy(nullptr, nullptr, 1.0f, 0);
        vec_scale(nullptr, 2.0f, 0);
        vec_add(nullptr, nullptr, nullptr, 0);
        std::array<float, 4> empty_output;
        empty_output.fill(sentinel);
        vec_dot_four_rows(nullptr, nullptr, 0, 0, empty_output.data(), 1);
        for (const float value : empty_output)
            require(sameBits(value, 0.0f), "empty grouped dot must write four positive zeros");

        std::array<float, 9> values{0.0f, -0.0f, 1.0f, -1.0f,
            std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::denorm_min(), -0.0f};
        vec_scale(values.data(), 1.0f, values.size());
        require(sameBits(values[0], 0.0f) && sameBits(values[1], -0.0f) && sameBits(values[8], -0.0f),
                "scale failed to preserve signed zeros");
        require(std::isinf(values[4]) && values[4] > 0 && std::isinf(values[5]) && values[5] < 0,
                "scale failed to preserve infinities");
        require(std::isnan(values[6]), "scale failed to preserve NaN classification");
        require(sameBits(values[7], std::numeric_limits<float>::denorm_min()), "scale flushed a subnormal");
    }
} // namespace

/** @brief Run all contracts, or only the focused grouped-dot preflight regression. */
int main(int argc, char **argv)
{
    try
    {
        require(argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--grouped"), "invalid test argument");
        testGroupedDots();
        if (argc == 1)
        {
            testElementwise();
            testEmptyAndSpecialValues();
        }
        std::puts("PASS: production vector primitives, grouped serial-row byte equivalence");
        return 0;
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
