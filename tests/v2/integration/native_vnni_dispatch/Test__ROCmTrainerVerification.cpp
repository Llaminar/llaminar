/**
 * @file Test__ROCmTrainerVerification.cpp
 * @brief Adversarial ROCm tests for device-resident trainer byte certificates.
 */

#include "../../performance/kernels/native_vnni_dispatch/GPUTrainerVerification.h"

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

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

    /** Convert a HIP setup failure into a constructor-safe fatal exception. */
    void requireROCm(hipError_t error, const char *operation)
    {
        if (error != hipSuccess)
        {
            throw std::runtime_error(
                std::string(operation) + ": " + hipGetErrorString(error));
        }
    }

    /** Own one explicit HIP stream and all tiny device fixtures for the test. */
    class ROCmCertificateFixture final
    {
    public:
        ROCmCertificateFixture()
        {
            requireROCm(hipSetDevice(0), "hipSetDevice");
            requireROCm(
                hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
                "hipStreamCreateWithFlags");
            requireROCm(hipMalloc(&actual_, kBytes), "hipMalloc(actual)");
            requireROCm(hipMalloc(&expected_, kBytes), "hipMalloc(expected)");
            requireROCm(
                hipMalloc(&mismatch_count_, sizeof(uint64_t)),
                "hipMalloc(mismatch_count)");
            requireROCm(
                hipMalloc(&first_mismatch_, sizeof(uint64_t)),
                "hipMalloc(first_mismatch)");
        }

        ~ROCmCertificateFixture()
        {
            if (stream_)
                (void)hipStreamSynchronize(stream_);
            if (first_mismatch_)
                (void)hipFree(first_mismatch_);
            if (mismatch_count_)
                (void)hipFree(mismatch_count_);
            if (expected_)
                (void)hipFree(expected_);
            if (actual_)
                (void)hipFree(actual_);
            if (stream_)
                (void)hipStreamDestroy(stream_);
        }

        ROCmCertificateFixture(const ROCmCertificateFixture &) = delete;
        ROCmCertificateFixture &operator=(const ROCmCertificateFixture &) = delete;

        /** Upload both operands on the exact comparison stream. */
        void upload(
            const std::array<float, 8> &actual,
            const std::array<float, 8> &expected)
        {
            ASSERT_EQ(
                hipMemcpyAsync(
                    actual_, actual.data(), kBytes, hipMemcpyHostToDevice,
                    stream_),
                hipSuccess);
            ASSERT_EQ(
                hipMemcpyAsync(
                    expected_, expected.data(), kBytes, hipMemcpyHostToDevice,
                    stream_),
                hipSuccess);
        }

        /** Compare all words and materialize only the two terminal counters. */
        std::array<uint64_t, 2> compare()
        {
            EXPECT_TRUE(llaminar2::test::enqueueROCmFP32ByteComparison(
                actual_, expected_, kElements, mismatch_count_,
                first_mismatch_, stream_));
            std::array<uint64_t, 2> result{};
            EXPECT_EQ(
                hipMemcpyAsync(
                    &result[0], mismatch_count_, sizeof(result[0]),
                    hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            EXPECT_EQ(
                hipMemcpyAsync(
                    &result[1], first_mismatch_, sizeof(result[1]),
                    hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            EXPECT_EQ(hipStreamSynchronize(stream_), hipSuccess);
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
        hipStream_t stream_ = nullptr;
        float *actual_ = nullptr;
        float *expected_ = nullptr;
        uint64_t *mismatch_count_ = nullptr;
        uint64_t *first_mismatch_ = nullptr;
    };
}

/** Prove exact IEEE bits, full-span coverage, and null-stream intolerance. */
TEST(Test__ROCmTrainerVerification,
     FullBufferCertificatePreservesIEEEBitsAndRejectsNullStream)
{
    ROCmCertificateFixture fixture;
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

    EXPECT_FALSE(llaminar2::test::enqueueROCmFP32ByteComparison(
        fixture.actual(), fixture.expected(), expected.size(),
        fixture.mismatchCount(), fixture.firstMismatch(), nullptr));
}
