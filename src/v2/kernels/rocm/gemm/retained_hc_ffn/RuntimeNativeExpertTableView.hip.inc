/**
 * @file RuntimeNativeExpertTableView.hip.inc
 * @brief Strided native role views over the existing acquired expert bank.
 *
 * Private diagnostic adapter. The placement bank, its original
 * producer validation and epoch lease remain the sole authority. No matrix,
 * payload, projection certificate, generation or readiness cache is copied.
 * These views change only the expert-record address calculation before the
 * unchanged native reader. Eligibility is resolved once per selected entry,
 * outside all numerical K/P loops, using the original acquired-bank method.
 */
#pragma once
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "kernels/common/DeviceNativeQuantisedMatrixDesc.h"
#include "kernels/rocm/moe/ROCmOriginalRouterPhase.hip.inc"

namespace llaminar2::rocm::fast_native {
template<class Acquired> class AcquiredNativeExpertTables;
/** @brief Select the validation supported by the actual original bank producer. */
enum class NativeRouteMembership : unsigned char { Checked, ValidatedOriginalFragment };
/**
 * @brief Original role member address; a layout helper, not a readiness certificate.
 * @tparam Role Original semantic role. @tparam Expert Actual original bank record type.
 * @param expert Exact original record. @return Address of that record's own role.
 * @details Host/device compilation can prove this mapping against the real host
 * and device ABI types without a raw byte-offset or a mirrored descriptor layout.
 */
template<PublishedNativeProjectionRole Role,class Expert>
__host__ __device__ constexpr const DeviceNativeQuantisedMatrixDesc* roleAddress(const Expert& expert) noexcept {
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(expert.gate)>,DeviceNativeQuantisedMatrixDesc>);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(expert.up)>,DeviceNativeQuantisedMatrixDesc>);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(expert.down)>,DeviceNativeQuantisedMatrixDesc>);
    if constexpr(Role==PublishedNativeProjectionRole::Gate)return &expert.gate;
    else if constexpr(Role==PublishedNativeProjectionRole::Up)return &expert.up;
    else {static_assert(Role==PublishedNativeProjectionRole::Down);return &expert.down;}
}
/**
 * @brief Address-only immutable role-table view issued by the acquired bank.
 * @tparam Expert Actual original expert ABI type; sizeof(Expert) owns the stride.
 * @tparam Role Original compile-time gate, up or down member.
 * @details This is not a second native table or a PublishedNativeProjection.
 * A numerical job must first pass the same owner's local/native eligibility.
 * Original selected IDs remain global expert IDs. Bounds reject before address
 * formation; inactive or remote jobs never enter their numerical scope.
 */
template<class Expert,PublishedNativeProjectionRole Role>
class NativeExpertRoleTable;

/**
 * @brief One native selected job derived from the immutable published LDS row.
 * @details The exact selector ID and bit were validated together before the
 * original CTA publication join. An ineligible route yields an empty job.
 */
class PublishedNativeRoute final {
    template<class,PublishedNativeProjectionRole> friend class NativeExpertRoleTable;
    int expert_;
    /** @brief Seal a selected expert or the empty job sentinel. */
    __device__ explicit PublishedNativeRoute(int expert):expert_(expert){}
public:
    /** @brief Borrow the corresponding ID only when the published row owns it.
     * @pre These are the exact paired ID row and membership mask produced before the existing join.
     * @param ids Original retained LDS selector IDs.
     * @param mask Native membership of that exact row.
     * @param route Semantic selected slot, wave-uniform for the consuming job.
     * @return Empty job for an unavailable slot or the original validated expert ID.
     */
    __device__ __forceinline__ static PublishedNativeRoute borrow(
        const int* ids,std::uint32_t mask,unsigned route) {
        if(route>=router_native::kDeviceMoEMaxTopK||((mask>>route)&1u)==0)
            return PublishedNativeRoute(-1);
        return PublishedNativeRoute(__builtin_amdgcn_readfirstlane(ids[route]));
    }
    /**
     * @brief Borrow the original dispatch ID published through the completed GU edge.
     * @param local_ids Joined LDS copy of the original CTA0 internal local dispatch IDs.
     * @param route Original admitted slot, shared by the active wave.
     * @return Original local expert or the publisher's exact -1 sentinel.
     * @pre The selected slot was validated by the original publisher and its local
     * ID was copied after the original grid and local Q8 publication joins.
     */
    __device__ __forceinline__ static PublishedNativeRoute borrowJoinedLocalDispatch(
        const int* local_ids,unsigned route) {
        return PublishedNativeRoute(__builtin_amdgcn_readfirstlane(local_ids[route]));
    }
    /** @return Whether the original placement owns the native numerical job. */
    __device__ __forceinline__ explicit operator bool() const {return expert_>=0;}
};
template<class Expert,PublishedNativeProjectionRole Role>
class NativeExpertRoleTable final {
    template<class> friend class AcquiredNativeExpertTables;
    const Expert* records_;
    unsigned experts_;
    /** @brief Issue only from the original acquired bank. */
    __device__ NativeExpertRoleTable(const Expert* records,unsigned experts):records_(records),experts_(experts){}
public:
    /**
     * @brief Preserve the fast body's original table-plus-expert syntax.
     * @param expert Same original wave-uniform selected expert ID.
     * @return Original role pointer, accepted by the unchanged native reader.
     * @pre The owning phase has published local/native eligibility for this job.
     */
    __device__ __forceinline__ const DeviceNativeQuantisedMatrixDesc* operator+(int expert) const {
        if(!records_||expert<0||unsigned(expert)>=experts_)__builtin_trap();
        return roleAddress<Role>(records_[expert]);
    }
    /**
     * @brief Preserve the original gate/up reference syntax.
     * @param expert Original selected expert ID. @return SAME original role record.
     * @pre Every active lane owns that same eligible original expert.
     */
    __device__ __forceinline__ const DeviceNativeQuantisedMatrixDesc& operator[](int expert) const {
        return *(operator+(expert));
    }
    /** @brief Use the already range-validated original expert in a published eligible job.
     * @param job Corresponding selected job from the same acquired bank and CTA.
     * @pre The caller admitted this nonempty job before entering role arithmetic.
     * @return Address of the original role in the same immutable bank.
     */
    __device__ __forceinline__ const DeviceNativeQuantisedMatrixDesc* operator+(PublishedNativeRoute job) const {
        return roleAddress<Role>(records_[job.expert_]);
    }
    /** @brief Return the exact role record for the admitted job.
     * @param job Same eligible route as for address formation.
     * @return The original descriptor consumed by the retained native reader.
     */
    __device__ __forceinline__ const DeviceNativeQuantisedMatrixDesc& operator[](PublishedNativeRoute job) const {
        return *(operator+(job));
    }
};
/**
 * @brief Bind the original issued epoch once, without a new storage authority.
 * @tparam Acquired Existing router_native::AcquiredRuntimeDecodeBank type.
 * @details The original bank lease remains immutable through kernel retirement.
 * Binding occurs at a phase boundary, not inside any numerical K/P loop. No host
 * payload pointer is embedded in capture and no bank generation is inferred.
 */
template<class Acquired>
class AcquiredNativeExpertTables final {
    Acquired acquired_;
    unsigned extent_;
    using Bank=std::remove_cvref_t<decltype(static_cast<const Acquired*>(nullptr)->bank())>;
    using Expert=std::remove_cvref_t<decltype(static_cast<const Bank*>(nullptr)->experts[0])>;
public:
    /**
     * @brief Borrow the existing issued token and original captured expert extent.
     * @param acquired SAME originally authenticated epoch token.
     * @param expected_experts Original source geometry, unchanged by migration.
     */
    __device__ explicit AcquiredNativeExpertTables(Acquired acquired,unsigned expected_experts):acquired_(acquired),extent_(expected_experts) {
        if(!expected_experts||expected_experts>router_native::kDeviceMoEMaxExperts||
           acquired_.bank().expert_count!=expected_experts||
           acquired_.runtime()->expert_count!=expected_experts)__builtin_trap();
    }
private:
    struct ProvenFragmentExtent {};
    /** @brief Bind a token relayed after a validated early CTA issue. */
    __device__ AcquiredNativeExpertTables(Acquired acquired,unsigned extent,ProvenFragmentExtent)
        :acquired_(acquired),extent_(extent) {}
public:
    /** @brief Retain the acquire and extent already proved by the fragment owner.
     * @tparam Experts Original bound expert geometry.
     * @param fragment Typed original fragment view.
     */
    template<unsigned Experts>
    __device__ __forceinline__ static AcquiredNativeExpertTables fromBoundFragment(
        router_native::ValidatedOriginalFragmentBank<typename Acquired::Runtime,Experts> fragment) {
        return {fragment.original(),Experts,ProvenFragmentExtent{}};
    }
    /**
     * @brief Verify the original format after that exact slot passed local membership.
     * @param expert Original eligible fragment ID from the same bank.
     * @pre The caller admits only IDs validated by the original bound fragment view.
     * @return True after the required exact native format check.
     */
    __device__ __forceinline__ bool requireOriginalNativeLocalFormat(int expert) const {
        if(acquired_.bank().experts[expert].weight_format!=DeviceMoEWeightFormat::NativeQuantised)
            __builtin_trap();
        return true;
    }
    /**
     * @brief Resolve original local/native membership once per selected entry.
     * @tparam Membership Route validation supported by the original producer.
     * @param expert Original global selected ID. @param active Original live-row decision.
     * @return Whether this entry may enter the fast numerical body.
     * @details The default keeps existing projection issuance and membership
     * checks. The explicit original-fragment policy consumes producer-validated
     * locality, then requires the original row format supported by this tuple.
     * Neither path loads any role descriptor plane here.
     */
    template<NativeRouteMembership Membership=NativeRouteMembership::Checked>
    __device__ __forceinline__ bool eligible(int expert,bool active) const {
        if(!active)return false;
        if constexpr(Membership==NativeRouteMembership::ValidatedOriginalFragment) {
            const auto row=acquired_.publishedOriginalFragmentMembership(expert);
            if(row.disposition!=MoEIntermediateShardRowDisposition::Local)
                return false;
            // This private numerical tuple admits only original native rows.
            // A published local row of a different format is a binding error.
            return requireOriginalNativeLocalFormat(expert);
        } else {
            static_assert(Membership==NativeRouteMembership::Checked);
            return bool(acquired_.localNativeProjectionOrInactive(expert,PublishedNativeProjectionRole::Gate));
        }
    }
    /** @return Original gate member view, with compile-time record stride/offset. */
    __device__ __forceinline__ auto gate() const {
        return NativeExpertRoleTable<Expert,PublishedNativeProjectionRole::Gate>(
            acquired_.bank().experts,extent_);
    }
    /** @return Original up member view, without copying it. */
    __device__ __forceinline__ auto up() const {
        return NativeExpertRoleTable<Expert,PublishedNativeProjectionRole::Up>(
            acquired_.bank().experts,extent_);
    }
    /** @return Original down member view, without copying it. */
    __device__ __forceinline__ auto down() const {
        return NativeExpertRoleTable<Expert,PublishedNativeProjectionRole::Down>(
            acquired_.bank().experts,extent_);
    }
};
/**
 * @brief Publish one row's local/native selected-slot mask before its existing join.
 * @tparam Membership Original checked or producer-validated locality.
 * @tparam Tables Same acquired expert-table binder.
 * @param tables SAME issued owner as the table views.
 * @param ids Original selected ID row; it is never rewritten or remapped.
 * @param routes Original positive topK, at most16.
 * @param active Original live row. @param masks Explicit caller-owned eligibility words.
 * @param row Original logical row index in the caller's retained capacity.
 * @pre All1024 physical lanes call; IDs are ready from the actual selector.
 * @post One wave0 leader authors the mask. The caller MUST retain its existing
 * post-selection CTA join before any job consumes this mask. No new join is added.
 * Its actual physical LDS extent must be admitted; a T1-dead source sorting
 * array is not evidence of free physical storage after compiler elimination.
 */
template<NativeRouteMembership Membership=NativeRouteMembership::Checked,class Tables>
__device__ __forceinline__ void publishNativeRouteEligibility(const Tables& tables,
    const int* ids,unsigned routes,bool active,std::uint32_t* masks,unsigned row) {
    if(!routes||routes>router_native::kDeviceMoEMaxTopK)__builtin_trap();
    if(threadIdx.x<64) {
        const unsigned lane=unsigned(threadIdx.x);
        bool eligible=false;
        if(lane<routes)eligible=tables.template eligible<Membership>(ids[lane],active);
        const auto mask=__ballot(eligible);
        if(lane==0)masks[row]=static_cast<std::uint32_t>(mask);
    }
}
/**
 * @brief Gate a whole expert job, outside its original arithmetic loops.
 * @param mask Published row mask. @param route Original selected slot.
 * @return Whether that selected slot owns native local numerical work.
 */
__host__ __device__ constexpr bool nativeRouteEligible(std::uint32_t mask,unsigned route) noexcept {
    return route<router_native::kDeviceMoEMaxTopK&&((mask>>route)&1u)!=0;
}
} // namespace llaminar2::rocm::fast_native
