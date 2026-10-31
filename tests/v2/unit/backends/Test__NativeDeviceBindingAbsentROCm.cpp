/**
 * @file Test__NativeDeviceBindingAbsentROCm.cpp
 * @brief Replay the actual production unsupported-build branch without linking HIP.
 *
 * This dedicated translation unit compiles the original implementation with its
 * backend explicitly absent. Other targets compile their own native copy; no
 * global source property or serving compiler flag is changed by this fixture.
 */
#ifdef HAVE_ROCM
#undef HAVE_ROCM
#endif
#include "backends/rocm/HipDeviceGuard.cpp"
#include <gtest/gtest.h>

/** @test A binary without ROCm must not report successful device selection. */
TEST(NativeDeviceBinding, AbsentROCmRejectsBothGuardEntryPoints)
{
    EXPECT_THROW((void)llaminar2::HipDeviceGuard::setDevice(0), std::logic_error);
    EXPECT_THROW((void)llaminar2::HipDeviceGuard::forceSetDevice(0), std::logic_error);
}
