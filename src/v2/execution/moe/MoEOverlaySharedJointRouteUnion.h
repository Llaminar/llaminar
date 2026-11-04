/**
 * @file MoEOverlaySharedJointRouteUnion.h
 * @brief Immutable joint-expert union for one original shared matrix group.
 *
 * This group projection never represents a publisher's own expert ownership.
 * Actual channel targets remain in original mapped controls; this value only
 * answers whether some original group member requires a joint expert row.
 */
#pragma once
#include "MoEOverlayStaticJointRouteMembership.h"
#include <span>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_SHARED_UNION_HD __host__ __device__
#else
#define LLAMINAR_SHARED_UNION_HD
#endif
namespace llaminar2 {
/** @brief Cold original channel target and its existing source projection. */
struct MoEOverlaySharedDispatchTarget {
 std::int32_t participant=-1;
 MoEOverlayStaticJointRouteMembership membership{};
};
/** @brief By-value group union; no allocation, row owner, or mutable routing state. */
class MoEOverlaySharedJointRouteUnion final {
public:
 /** @brief An ordinary-only group needs no immutable joint epoch. */
 constexpr MoEOverlaySharedJointRouteUnion() noexcept=default;
 /** @brief Validate original group projections and merge only their joint expert sets. */
 static MoEOverlaySharedJointRouteUnion fromOriginalGroup(std::uint32_t experts,
  std::int32_t layer,std::span<const MoEOverlaySharedDispatchTarget> members) {
  if(!experts || experts>moe_runtime_abi::kMaxExperts || layer<0 || members.empty())
   throw std::invalid_argument("Shared dispatch union requires an original group and expert axis");
  MoEOverlaySharedJointRouteUnion result;
  for(std::size_t i=0;i<members.size();++i) {
   const auto& m=members[i];
   if(m.participant<0 || !m.membership.validFor(experts,m.participant,layer))
    throw std::invalid_argument("Shared dispatch union has foreign source membership");
   for(std::size_t prior=0;prior<i;++prior)if(members[prior].participant==m.participant)
    throw std::invalid_argument("Shared dispatch union repeats an original channel target");
   if(!m.membership.enabled())continue;
   if(result.epoch_ && result.epoch_!=m.membership.initialEpoch())
    throw std::invalid_argument("Shared dispatch union changed the original Static epoch");
   result.epoch_=m.membership.initialEpoch();result.experts_=experts;result.layer_=layer;
   for(std::uint32_t e=0;e<experts;++e)if(m.membership.contains(e))result.bits_[e/32]|=std::uint32_t{1}<<(e%32);
  }
  return result;
 }
 /** @return Whether this group retains the exact original expert axis/layer/epoch. */
 LLAMINAR_SHARED_UNION_HD constexpr bool validFor(std::uint32_t experts,std::int32_t layer,std::uint64_t epoch)const noexcept {
  return !epoch_ || (experts==experts_ && layer==layer_ && epoch==epoch_);
 }
 /** @return Whether some original consumer needs this joint expert's row. */
 LLAMINAR_SHARED_UNION_HD constexpr bool contains(std::int32_t expert)const noexcept {
  return expert>=0 && static_cast<std::uint32_t>(expert)<experts_ && (bits_[expert/32]&(std::uint32_t{1}<<(expert%32)));
 }
 /** @return False for malformed scalar ownership or a competing whole owner. */
 LLAMINAR_SHARED_UNION_HD constexpr bool accepts(std::int32_t expert,std::int32_t scalar)const noexcept {
  return scalar>=-1 && (!contains(expert) || scalar==-1);
 }
private:
 std::uint32_t experts_=0;
 std::int32_t layer_=-1;
 std::uint64_t epoch_=0;
 std::uint32_t bits_[(moe_runtime_abi::kMaxExperts+31u)/32u]={};
};
static_assert(std::is_trivially_copyable_v<MoEOverlaySharedJointRouteUnion>);
}
#undef LLAMINAR_SHARED_UNION_HD
