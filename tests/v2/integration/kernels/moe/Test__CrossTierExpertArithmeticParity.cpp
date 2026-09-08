/**
 * @file Test__CrossTierExpertArithmeticParity.cpp
 * @brief CPU/CUDA/ROCm GPU-aligned expert arithmetic parity integration tests.
 *
 * ExpertOverlay may execute one logical expert on different physical backends
 * in adjacent movement epochs. These focused tests keep the expensive common
 * all-codebook proof in a small translation unit so arithmetic regressions can
 * be iterated without rebuilding either backend's monolithic MoE test binary.
 */

#include "NativeVNNIExpertTransferParityTest.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace
{
    /** @brief RAII owner for one explicit non-default backend test stream. */
    class ScopedBackendStream final
    {
    public:
        /**
         * @brief Create a stream on the exact backend endpoint.
         * @param device CUDA or ROCm device whose backend owns the stream.
         */
        explicit ScopedBackendStream(llaminar2::DeviceId device)
            : backend_(llaminar2::getBackendFor(device)),
              ordinal_(device.ordinal)
        {
            if (!backend_ || !device.is_gpu() || ordinal_ < 0 ||
                ordinal_ >= backend_->deviceCount())
                throw std::runtime_error(
                    "Cross-tier arithmetic parity received an unavailable GPU endpoint");
            stream_ = backend_->createStream(ordinal_);
            if (!stream_)
                throw std::runtime_error(
                    "Cross-tier arithmetic parity could not create an explicit stream");
        }

        /** @brief Release the stream after the helper's terminal event drain. */
        ~ScopedBackendStream()
        {
            if (stream_)
                backend_->destroyStream(stream_, ordinal_);
        }

        ScopedBackendStream(const ScopedBackendStream &) = delete;
        ScopedBackendStream &operator=(const ScopedBackendStream &) = delete;

        /** @return The exact non-default stream used by every GPU operation. */
        [[nodiscard]] void *get() const noexcept { return stream_; }

    private:
        llaminar2::IBackend *backend_ = nullptr;
        int ordinal_ = -1;
        void *stream_ = nullptr;
    };
} // namespace

#ifdef HAVE_CUDA
/** @test Prove all quantized GPU-aligned experts are CPU/CUDA byte-identical. */
TEST(
    Test__CrossTierExpertArithmeticParity,
    CPUAndCUDAAllQuantizedExpertFormatsAreByteExactAcrossM)
{
    const auto device = llaminar2::DeviceId::cuda(0);
    auto *backend = llaminar2::getBackendFor(device);
    if (!backend || backend->deviceCount() <= 0)
        GTEST_SKIP() << "No CUDA device available";
    ScopedBackendStream stream(device);
    llaminar2::test::runCPUToGPUAllFormatExpertArithmeticParity(
        "CUDA",
        device,
        stream.get());
}
#endif

#ifdef HAVE_ROCM
/** @test Prove all quantized GPU-aligned experts are CPU/ROCm byte-identical. */
TEST(
    Test__CrossTierExpertArithmeticParity,
    CPUAndROCmAllQuantizedExpertFormatsAreByteExactAcrossM)
{
    const auto device = llaminar2::DeviceId::rocm(0);
    auto *backend = llaminar2::getBackendFor(device);
    if (!backend || backend->deviceCount() <= 0)
        GTEST_SKIP() << "No ROCm device available";
    ScopedBackendStream stream(device);
    llaminar2::test::runCPUToGPUAllFormatExpertArithmeticParity(
        "ROCm",
        device,
        stream.get());
}
#endif
