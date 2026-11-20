/**
 * @file NativeQuantisedDescriptorValidation.h
 * @brief Shared physical-plane validity for original prepared expert publication.
 *
 * This owns no descriptor, allocation or epoch. Source-family compatibility is
 * separately authenticated by the original fragment declaration. Raw numerical
 * callers retain their existing zero-result predicate; publication additionally
 * requires positive, complete K32 geometry before exposing a ready bank row.
 */
#pragma once
#include "DeviceNativeQuantisedMatrixDesc.h"
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_DESCRIPTOR_HD __host__ __device__
#else
#define LLAMINAR_DESCRIPTOR_HD
#endif
namespace llaminar2::rocm::router_native { template<class> class AcquiredRuntimeDecodeBank; }
namespace llaminar2::rocm::full_native_ffn { class PreparedFullNativeSharedSource; }
namespace llaminar2::rocm::grouped_ffn { class PreparedGroupedSharedRole; }
namespace llaminar2 {
/** @brief Semantic projection role; never a caller-provided descriptor offset. */
enum class PublishedNativeProjectionRole : unsigned char { Gate, Up, Down };
/**
 * @brief Borrowed projection certified by the existing producer and acquired bank.
 * @details No descriptor bytes or generations are copied. Only the existing
 * acquired-bank authority or sealed prepared shared source may issue this view.
 * A null view denotes an explicitly nonlocal/non-native route, not a fallback.
 */
class PublishedNativeProjection final {
    template<class> friend class rocm::router_native::AcquiredRuntimeDecodeBank;
    friend class rocm::full_native_ffn::PreparedFullNativeSharedSource;
    friend class rocm::grouped_ffn::PreparedGroupedSharedRole;
    const DeviceNativeQuantisedMatrixDesc* descriptor_;
    /** @brief Issue only inside an existing prepared/acquired authority. */
    LLAMINAR_DESCRIPTOR_HD explicit constexpr PublishedNativeProjection(
        const DeviceNativeQuantisedMatrixDesc* descriptor) noexcept:descriptor_(descriptor){}
public:
    /** @return Whether this route has a local published native projection. */
    LLAMINAR_DESCRIPTOR_HD explicit constexpr operator bool() const noexcept {return descriptor_!=nullptr;}
    /** @return SAME original descriptor, immutable for the issued view's lifetime. @pre View is nonempty. */
    LLAMINAR_DESCRIPTOR_HD constexpr const DeviceNativeQuantisedMatrixDesc& descriptor() const noexcept {return *descriptor_;}
};
/** @brief Canonical physical decoder metadata shared by producer and numeric guards. */
struct NativeQuantisedDescriptorValidation final {
    /** @return Whether physical code names an implemented native decoder. */
    LLAMINAR_DESCRIPTOR_HD static constexpr bool supported(unsigned code) noexcept {
        switch(code) {
        case 0: case 4: case 5: case 6: case 7: case 8: case 9: case 10:
        case 11: case 12: case 13: case 14: case 15: case 16: case 17:
        case 19: case 23: case 24: case 25: return true;
        default: return false;
        }
    }
    /** @return Whether the physical encoding carries a secondary coefficient plane. */
    LLAMINAR_DESCRIPTOR_HD static constexpr bool secondary(unsigned code) noexcept {
        switch(code) {
        case 5: case 7: case 8: case 9: case 10: case 11: case 13:
        case 14: case 16: case 17: case 23: case 24: return true;
        default: return false;
        }
    }
    /**
     * @brief Original numeric predicate, including physical24/physical4 compatibility.
     * @param d Original prepared descriptor. @param n Expected outputs.
     * @param k Expected reduction width. @param decoder Actual compiled decoder.
     * @return Exactly the original invariant-zero acceptance predicate.
     */
    template<class Descriptor>
    LLAMINAR_DESCRIPTOR_HD static constexpr bool accepts(
        const Descriptor& d,unsigned n,unsigned k,unsigned decoder) noexcept {
        return d.payload && d.scales &&
            static_cast<unsigned>(d.n)==n && static_cast<unsigned>(d.k)==k &&
            d.blocks_per_row==k/32 &&
            (d.codebook_id==decoder || (decoder==24 && d.codebook_id==4)) &&
            (!secondary(decoder) || d.codebook_id==4 || d.mins) &&
            (decoder!=10 || d.emins);
    }
    /**
     * @brief Strong ready-publication postcondition; never infers allocation capacity.
     * @param d SAME descriptor exported by its prepared allocation owner.
     * @return Whether positive K32 geometry and all physical planes are complete.
     */
    template<class Descriptor>
    LLAMINAR_DESCRIPTOR_HD static constexpr bool publishable(
        const Descriptor& d) noexcept {
        return supported(d.codebook_id) && d.n>0 && d.k>0 && d.k%32==0 &&
            accepts(d,static_cast<unsigned>(d.n),static_cast<unsigned>(d.k),d.codebook_id);
    }
};
}
#undef LLAMINAR_DESCRIPTOR_HD
