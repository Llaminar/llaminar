/**
 * @file FullNativeHCFFNOutputDemand.h
 * @brief Immutable optional materializations of one closed HC/router/FFN region.
 *
 * This byte is a cold graph policy, not live execution state or storage
 * authority. Required internal operands, residual/injection state and combined
 * output are outside this mask. Existing owners seal it before native capture;
 * reset never changes it. A selected observation is unioned with actual outside
 * data readers rather than inferred from a nullable output pointer.
 */
#pragma once
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_FULL_HC_FFN_DEMAND_HD __host__ __device__
#else
#define LLAMINAR_FULL_HC_FFN_DEMAND_HD
#endif

namespace llaminar2 {
/** @brief Optional original tensor roles; enum ordinals are the sealed mask bits. */
enum class FullNativeHCFFNOutputRole : std::uint8_t {
    HCActivated=0, ///< Completed /4-SiLU FP32 down value, when externally observed.
    HCUp=1, ///< Completed original four-branch up/read logits.
    RouteContributions=2, ///< Individually weighted original routed FP32 rows.
    Routed=3, ///< Original ordered route fold, before the shared addition.
    Shared=4, ///< Weighted shared result, before the routed addition.
    SharedGate=5, ///< Original shared gate projection, before SwiGLU.
    SharedUp=6, ///< Original shared up projection, before SwiGLU.
    RouterProbabilities=7 ///< Original probability observation, not routing state.
};

/**
 * @brief One immutable eight-bit publication policy, copied directly in kernargs.
 * @details Construction is explicit. all() preserves the observed/public surface;
 * none() requests only mandatory internal/region outputs. A role bit is added
 * cold by with() or set union. Neither operation admits a graph or allocation.
 */
class FullNativeHCFFNOutputDemand {
public:
    /** @return Every optional original materialization, for an observed region. */
    LLAMINAR_FULL_HC_FFN_DEMAND_HD static constexpr FullNativeHCFFNOutputDemand all() noexcept {return FullNativeHCFFNOutputDemand{0xffu};}
    /** @return No optional mirror; mandatory outputs and computations are unchanged. */
    LLAMINAR_FULL_HC_FFN_DEMAND_HD static constexpr FullNativeHCFFNOutputDemand none() noexcept {return FullNativeHCFFNOutputDemand{0u};}
    /** @brief Query one declared role. @param role Original role. @return Whether its mirror is required. */
    LLAMINAR_FULL_HC_FFN_DEMAND_HD constexpr bool contains(FullNativeHCFFNOutputRole role) const noexcept {
        const auto ordinal=static_cast<unsigned>(role);
        if(ordinal>=8u)__builtin_trap();
        return (bits_&(std::uint8_t{1u}<<ordinal))!=0;
    }
    /** @return Exact policy byte for existing graph/native capture identity. */
    LLAMINAR_FULL_HC_FFN_DEMAND_HD constexpr std::uint8_t bits() const noexcept {return bits_;}
    /** @brief Add a role to a new value. @param role Valid original role. @return New union; this value is unchanged.
     * @details Invalid enum values trap, and cannot form a constexpr policy.
     * The host resolver rejects them with a contextual diagnostic before this leaf.
     */
    LLAMINAR_FULL_HC_FFN_DEMAND_HD constexpr FullNativeHCFFNOutputDemand with(FullNativeHCFFNOutputRole role) const noexcept {
        const auto ordinal=static_cast<unsigned>(role);
        if(ordinal>=8u)__builtin_trap();
        return FullNativeHCFFNOutputDemand{static_cast<std::uint8_t>(bits_|(std::uint8_t{1u}<<ordinal))};
    }
    /** @brief Validate a cold role declaration. @param role Candidate enum. @return True only for an original role. */
    LLAMINAR_FULL_HC_FFN_DEMAND_HD static constexpr bool validRole(FullNativeHCFFNOutputRole role) noexcept {return static_cast<unsigned>(role)<8u;}
    /** @brief Combine true outside readers and selected observations. @param other Other cold policy. @return Immutable union. */
    LLAMINAR_FULL_HC_FFN_DEMAND_HD constexpr FullNativeHCFFNOutputDemand operator|(FullNativeHCFFNOutputDemand other) const noexcept {
        return FullNativeHCFFNOutputDemand{static_cast<std::uint8_t>(bits_|other.bits_)};
    }
    /** @brief Intersect declared roles without exposing an arbitrary raw-bit constructor.
     * @param other Allowed role set. @return Immutable intersection.
     */
    LLAMINAR_FULL_HC_FFN_DEMAND_HD constexpr FullNativeHCFFNOutputDemand operator&(FullNativeHCFFNOutputDemand other) const noexcept {
        return FullNativeHCFFNOutputDemand{static_cast<std::uint8_t>(bits_&other.bits_)};
    }
    /** @brief Remove one declared role. @param role Valid role. @return New policy; invalid enums trap. */
    LLAMINAR_FULL_HC_FFN_DEMAND_HD constexpr FullNativeHCFFNOutputDemand without(FullNativeHCFFNOutputRole role) const noexcept {
        const auto ordinal=static_cast<unsigned>(role);
        if(ordinal>=8u)__builtin_trap();
        return FullNativeHCFFNOutputDemand{static_cast<std::uint8_t>(bits_&~(std::uint8_t{1u}<<ordinal))};
    }
    /** @brief Compare complete optional materialization identity. @param other Other policy. @return Exact byte equality. */
    LLAMINAR_FULL_HC_FFN_DEMAND_HD constexpr bool operator==(const FullNativeHCFFNOutputDemand& other) const noexcept {return bits_==other.bits_;}
private:
    /** @brief Build only from the complete original bit domain. @param bits Valid eight-role mask. */
    LLAMINAR_FULL_HC_FFN_DEMAND_HD explicit constexpr FullNativeHCFFNOutputDemand(std::uint8_t bits) noexcept:bits_(bits) {}
    std::uint8_t bits_; ///< Sole immutable encoded policy; no graph/device pointers.
};
static_assert(sizeof(FullNativeHCFFNOutputDemand)==1&&std::is_trivially_copyable_v<FullNativeHCFFNOutputDemand>);
} // namespace llaminar2
#undef LLAMINAR_FULL_HC_FFN_DEMAND_HD
