/**
 * @file Test__CUDATrainerVerification.cpp
 * @brief Adversarial CUDA tests for device-resident trainer byte certificates.
 *
 * Exact dispatch overlays are installable only when every output bit matches
 * the selected oracle. This suite protects the small diagnostic primitive
 * used by the CUDA sweep harnesses, including IEEE-754 values that ordinary
 * floating-point equality would incorrectly normalize.
 */

#include "../../performance/kernels/native_vnni_dispatch/GPUTrainerVerification.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
    /** Construct one FP32 value without allowing the compiler to canonicalize it. */
    float fp32FromBits(uint32_t bits)
    {
        float value = 0.0f;
        static_assert(sizeof(value) == sizeof(bits));
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    /** Convert a CUDA setup failure into a constructor-safe fatal exception. */
    void requireCuda(cudaError_t error, const char *operation)
    {
        if (error != cudaSuccess)
        {
            throw std::runtime_error(
                std::string(operation) + ": " + cudaGetErrorString(error));
        }
    }

    /** Own one explicit CUDA stream and all tiny device fixtures for the test. */
    class CudaCertificateFixture final
    {
    public:
        CudaCertificateFixture()
        {
            requireCuda(cudaSetDevice(0), "cudaSetDevice");
            requireCuda(
                cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags");
            requireCuda(cudaMalloc(&actual_, kBytes), "cudaMalloc(actual)");
            requireCuda(cudaMalloc(&expected_, kBytes), "cudaMalloc(expected)");
            requireCuda(
                cudaMalloc(&mismatch_count_, sizeof(uint64_t)),
                "cudaMalloc(mismatch_count)");
            requireCuda(
                cudaMalloc(&first_mismatch_, sizeof(uint64_t)),
                "cudaMalloc(first_mismatch)");
        }

        ~CudaCertificateFixture()
        {
            if (stream_)
                (void)cudaStreamSynchronize(stream_);
            if (first_mismatch_)
                (void)cudaFree(first_mismatch_);
            if (mismatch_count_)
                (void)cudaFree(mismatch_count_);
            if (expected_)
                (void)cudaFree(expected_);
            if (actual_)
                (void)cudaFree(actual_);
            if (stream_)
                (void)cudaStreamDestroy(stream_);
        }

        CudaCertificateFixture(const CudaCertificateFixture &) = delete;
        CudaCertificateFixture &operator=(const CudaCertificateFixture &) = delete;

        /** Upload both operands on the exact comparison stream. */
        void upload(
            const std::array<float, 8> &actual,
            const std::array<float, 8> &expected)
        {
            ASSERT_EQ(
                cudaMemcpyAsync(
                    actual_, actual.data(), kBytes, cudaMemcpyHostToDevice,
                    stream_),
                cudaSuccess);
            ASSERT_EQ(
                cudaMemcpyAsync(
                    expected_, expected.data(), kBytes, cudaMemcpyHostToDevice,
                    stream_),
                cudaSuccess);
        }

        /** Compare all words and materialize only the two terminal counters. */
        std::array<uint64_t, 2> compare()
        {
            EXPECT_TRUE(llaminar2::test::enqueueCudaFP32ByteComparison(
                actual_, expected_, kElements, mismatch_count_,
                first_mismatch_, stream_));
            std::array<uint64_t, 2> result{};
            EXPECT_EQ(
                cudaMemcpyAsync(
                    &result[0], mismatch_count_, sizeof(result[0]),
                    cudaMemcpyDeviceToHost, stream_),
                cudaSuccess);
            EXPECT_EQ(
                cudaMemcpyAsync(
                    &result[1], first_mismatch_, sizeof(result[1]),
                    cudaMemcpyDeviceToHost, stream_),
                cudaSuccess);
            EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
            return result;
        }

        /** @return Raw candidate allocation for guard testing. */
        const float *actual() const { return actual_; }

        /** @return Raw oracle allocation for guard testing. */
        const float *expected() const { return expected_; }

        /** @return Persistent mismatch counter for guard testing. */
        uint64_t *mismatchCount() const { return mismatch_count_; }

        /** @return Persistent first-index counter for guard testing. */
        uint64_t *firstMismatch() const { return first_mismatch_; }

    private:
        static constexpr size_t kElements = 8;
        static constexpr size_t kBytes = kElements * sizeof(float);
        cudaStream_t stream_ = nullptr;
        float *actual_ = nullptr;
        float *expected_ = nullptr;
        uint64_t *mismatch_count_ = nullptr;
        uint64_t *first_mismatch_ = nullptr;
    };
}

/** Prove exact IEEE bits, full-span coverage, and null-stream intolerance. */
TEST(Test__CUDATrainerVerification,
     FullBufferCertificatePreservesIEEEBitsAndRejectsNullStream)
{
    CudaCertificateFixture fixture;
    const std::array<float, 8> expected = {
        fp32FromBits(0x00000000),
        fp32FromBits(0x80000000),
        fp32FromBits(0x7fc00001),
        fp32FromBits(0x3f800000),
        fp32FromBits(0xbf800000),
        fp32FromBits(0x00800000),
        fp32FromBits(0x7f7fffff),
        fp32FromBits(0xff800000),
    };

    fixture.upload(expected, expected);
    EXPECT_EQ(
        fixture.compare(),
        (std::array<uint64_t, 2>{
            0, std::numeric_limits<uint64_t>::max()}));

    auto actual = expected;
    actual[1] = fp32FromBits(0x00000000);
    actual[2] = fp32FromBits(0x7fc00002);
    actual[7] = fp32FromBits(0x7f800000);
    fixture.upload(actual, expected);
    EXPECT_EQ(fixture.compare(), (std::array<uint64_t, 2>{3, 7}));

    EXPECT_FALSE(llaminar2::test::enqueueCudaFP32ByteComparison(
        fixture.actual(), fixture.expected(), expected.size(),
        fixture.mismatchCount(), fixture.firstMismatch(), nullptr));
}
