/**
 * @file HIPGraphArgumentGeometryABI.h
 * @brief Narrow host/device packet and original kernel-symbol ABI for the native regression.
 *
 * No MPI, planning, SIMD or backend headers belong in the HIP device pass.
 * Native graph constructors copy this trivial user argument; hidden ABI fields
 * remain separately authenticated by named compiler metadata.
 */
#pragma once
#include <cstdint>
#include <type_traits>
namespace llaminar2::hip_argument_geometry_test {
inline constexpr unsigned Nodes = 97;
inline constexpr unsigned Threads = 128;
inline constexpr unsigned Variants = 4;
inline constexpr unsigned Replays = 3;
/** @brief User packet with explicit first/last words and original output owner. */
struct Packet {
    std::uint32_t *output;
    std::uint32_t slot;
    std::uint32_t salt;
    std::uint32_t words[1020];
};
static_assert(sizeof(Packet) == 4096 && std::is_trivially_copyable_v<Packet>);
/**
 * @brief Return the actual original HIP kernel symbol without executing a device API.
 * @return Native function address for hipKernelNodeParams::func, never a host wrapper.
 */
void *hipGraphArgumentGeometryKernel();
} // namespace llaminar2::hip_argument_geometry_test
