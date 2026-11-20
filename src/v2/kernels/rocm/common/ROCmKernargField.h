/**
 * @file ROCmKernargField.h
 * @brief Phase-local scalar reads from the original immutable HIP kernel argument.
 *
 * The kernel must take Envelope as its first by-value argument. No mirror,
 * allocation, device publication or new ownership is introduced: HIP's retained
 * kernel node remains the sole owner of these exact argument bytes. Volatile
 * reads keep later-phase fields out of earlier decoder register lifetimes.
 */
#pragma once
#include <cstddef>
#include <type_traits>

namespace llaminar2::rocm {
/**
 * @brief Read one typed scalar field at its consuming phase.
 * @tparam Envelope Actual standard-layout first by-value kernel argument type.
 * @tparam Value Exact field type, restricted to scalar, enum or pointer values.
 * @tparam Offset Field offset expressed by offsetof on the actual argument type.
 * @return The identical immutable captured field value.
 * @pre Envelope is the first kernel parameter; no leading hidden explicit args.
 * @details Address-space4 is the native kernarg segment. Natural field alignment
 * and bounds are checked statically. Only the scalar load is volatile; pointed-to
 * tensors retain their existing access and synchronization contracts.
 */
template<class Envelope,class Value,std::size_t Offset>
__device__ __forceinline__ Value phaseKernargField()
{
    static_assert(std::is_standard_layout_v<Envelope> && std::is_trivially_copyable_v<Envelope>);
    static_assert(std::is_scalar_v<Value> && std::is_trivially_copyable_v<Value>);
    static_assert(Offset<=sizeof(Envelope) && sizeof(Value)<=sizeof(Envelope)-Offset);
    static_assert(Offset%alignof(Value)==0 && alignof(Envelope)>=alignof(Value));
#if defined(__HIP_DEVICE_COMPILE__)
    typedef unsigned char ConstantByte __attribute__((opencl_constant));
    typedef Value ConstantValue __attribute__((opencl_constant));
    // Clang gives the builtin a language-qualified constant pointer. Its
    // explicit address-space conversion preserves the same native constant
    // segment while adding byte addressing; no generic pointer is introduced.
    const auto* bytes=(const ConstantByte*)__builtin_amdgcn_kernarg_segment_ptr();
    return *reinterpret_cast<const volatile ConstantValue*>(bytes+Offset);
#else
    // The host pass parses device definitions but never executes this body.
    return Value{};
#endif
}
}
