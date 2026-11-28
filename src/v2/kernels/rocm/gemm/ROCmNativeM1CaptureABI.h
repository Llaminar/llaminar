/**
 * @file ROCmNativeM1CaptureABI.h
 * @brief Typed native M1 producer/reducer signatures for retained argument inspection.
 *
 * Launch-site static assertions bind these signatures to the actual HIP kernels.
 * Metadata is one value argument, not three independent pointer arguments.
 */
#pragma once
#include "kernels/common/ProjectionRowWork.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedBlockMetadata.h"
#include <tuple>
#include <type_traits>

namespace llaminar2::rocm {
/** @brief Actual flat M1 producer function type, including the metadata value. */
using NativeM1ProducerSignature = void(const std::int8_t*, const std::uint8_t*,
    rocm_native_metadata::MetadataPlanes, float*, float*, const float*, const float*,
    int, int, int, ProjectionRowWork);
/** @brief Actual ordered M1 reducer function type. */
using NativeM1ReducerSignature = void(const float*, float*, int, int, ProjectionRowWork);
/** @brief Argument types and arity of an authenticated native function signature. */
template<class Signature> struct NativeM1CaptureABI;
template<class... Args> struct NativeM1CaptureABI<void(Args...)> {
    using Arguments = std::tuple<Args...>;
    static constexpr std::size_t arity = sizeof...(Args);
    template<std::size_t Index> using Argument = std::tuple_element_t<Index, Arguments>;
};
static_assert(NativeM1CaptureABI<NativeM1ProducerSignature>::arity == 11);
static_assert(std::is_same_v<NativeM1CaptureABI<NativeM1ProducerSignature>::Argument<9>, int>);
static_assert(NativeM1CaptureABI<NativeM1ReducerSignature>::arity == 5);
} // namespace llaminar2::rocm
