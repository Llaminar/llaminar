/**
 * @file HiddenQ8NativeGraphIdentity.h
 * @brief Test-only opaque CUDA handles keep native SDKs isolated in the original replay owner.
 */
#pragma once
#include "backends/IGPUGraphCapture.h"
#include "backends/DeviceId.h"
#include <array>
#include <cstdint>
namespace llaminar2::test {
/** @brief Read actual retained CUDA handles without native queries or submitting work. */
std::array<std::uintptr_t,2> hiddenQ8NativeGraphIdentityCUDA(const IGPUGraphCapture&,DeviceId);
}
