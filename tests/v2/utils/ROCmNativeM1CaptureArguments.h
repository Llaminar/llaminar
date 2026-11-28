/**
 * @file ROCmNativeM1CaptureArguments.h
 * @brief Read retained HIP argument bytes only at signature-checked indices and types.
 */
#pragma once
#include "kernels/rocm/gemm/ROCmNativeM1CaptureABI.h"
#include <array>
#include <bit>
#include <cstring>
#include <stdexcept>

namespace llaminar2::test {
/**
 * @brief Recover one immutable captured argument without default construction.
 * @tparam Signature Launch-site-authenticated native function type.
 * @tparam Index Argument index, checked at compile time before any pointer access.
 * @param arguments Borrowed HIP argument table retained by the native graph.
 * @return Exact typed argument bytes.
 * @throws std::runtime_error If the table or selected storage is absent.
 */
template<class Signature, std::size_t Index>
auto nativeM1CaptureArgument(void* const* arguments)
{
    using ABI = rocm::NativeM1CaptureABI<Signature>;
    static_assert(Index < ABI::arity, "Captured argument index exceeds native signature");
    using T = typename ABI::template Argument<Index>;
    static_assert(std::is_trivially_copyable_v<T>);
    if (!arguments || !arguments[Index])
        throw std::runtime_error("Missing captured native M1 argument");
    std::array<std::byte, sizeof(T)> bytes{};
    std::memcpy(bytes.data(), arguments[Index], bytes.size());
    return std::bit_cast<T>(bytes);
}
} // namespace llaminar2::test
