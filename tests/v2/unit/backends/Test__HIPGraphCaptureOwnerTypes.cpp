/**
 * @file Test__HIPGraphCaptureOwnerTypes.cpp
 * @brief Compile the production HIP capture owner contract without a GPU runtime link.
 *
 * Actual graph constructor declarations reject a foreign vendor or generic worker
 * owner even when its ordinal matches. No runtime object is instantiated and the
 * executable links only GTest; selected SDK headers contribute types at build time.
 */
#include "backends/rocm/HIPGraphCapture.h"
#include <gtest/gtest.h>
#include <type_traits>

/**
 * @test Owner-vendor identity is a compile-time contract, independent of devices.
 *
 * An equal ordinal (including zero) cannot make a foreign vendor acceptable:
 * integer values do not participate in constructor overload type acceptance.
 * No context is constructed and no native runtime is queried by this replay.
 */
namespace llaminar2 { class NvidiaDeviceContext; }
static_assert(std::is_constructible_v<llaminar2::HIPGraphCapture, hipStream_t, int, llaminar2::AMDDeviceContext &>);
static_assert(!std::is_constructible_v<llaminar2::HIPGraphCapture, hipStream_t, int, llaminar2::NvidiaDeviceContext &>);
static_assert(!std::is_constructible_v<llaminar2::HIPGraphCapture, hipStream_t, int, llaminar2::IWorkerGPUContext &>);
TEST(HIPGraphCaptureOwnerTypes, RejectsForeignVendorAtConstruction)
{
    EXPECT_TRUE((std::is_constructible_v<llaminar2::HIPGraphCapture, hipStream_t, int, llaminar2::AMDDeviceContext &>));
    EXPECT_FALSE((std::is_constructible_v<llaminar2::HIPGraphCapture, hipStream_t, int, llaminar2::NvidiaDeviceContext &>));
    EXPECT_FALSE((std::is_constructible_v<llaminar2::HIPGraphCapture, hipStream_t, int, llaminar2::IWorkerGPUContext &>));
}
