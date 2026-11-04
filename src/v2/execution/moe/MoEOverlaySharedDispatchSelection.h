/**
 * @file MoEOverlaySharedDispatchSelection.h
 * @brief Stateless original-row selection for one authenticated shared channel group.
 *
 * The original router and immutable source cohort select useful rows. This
 * helper owns no union mask, placement state, counters or storage. Metadata
 * validates the complete prefix before the mapped row writer executes.
 */
#pragma once
#include "MoEOverlayActivationPacketABI.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_SHARED_ROW_HD __host__ __device__
#else
#define LLAMINAR_SHARED_ROW_HD
#endif
namespace llaminar2 {
/** @brief Complete row decision; invalid inputs never authorize a write. */
enum class MoEOverlaySharedRowDecision : std::uint8_t {Invalid,Unneeded,Needed};
/**
 * @brief Resolve one original row against every original consumer, without mutation.
 * @param launch Immutable source operand and original group descriptor.
 * @param routes Acquired original placement bank, not an independent owner table.
 * @param epoch Original acquired Static epoch.
 * @param row Original physical row owned by one copy CTA.
 * @return Needed exactly once if any consumer selects an original route slot.
 */
LLAMINAR_SHARED_ROW_HD inline MoEOverlaySharedRowDecision selectMoEOverlaySharedRow(
 const MoEOverlayActivationDispatchPackLaunch& launch,const std::int32_t* routes,
 std::uint64_t epoch,std::int32_t row) noexcept {
 if(!routes || !launch.route_expert_ids_fp32 || !launch.route_weights || row<0 ||
    ((launch.shared_row_controls==nullptr)!=(launch.shared_row_target_count==0u)) || epoch==0)
  return MoEOverlaySharedRowDecision::Invalid;
 const auto active=launch.active_row_count_device?*launch.active_row_count_device:launch.physical_rows;
 if(active<0 || active>launch.physical_rows)return MoEOverlaySharedRowDecision::Invalid;
 if(row>=active)return MoEOverlaySharedRowDecision::Unneeded;
 bool needed=false;
 for(int slot=0;slot<launch.packet.top_k;++slot) {
  const auto coordinate=static_cast<std::size_t>(row)*launch.packet.top_k+slot;
  const float value=launch.route_expert_ids_fp32[coordinate],weight=launch.route_weights[coordinate];
  if(!(value>=0.0f && value<launch.placement.expert_count) ||
     !(weight>=-0x1.fffffep+127F && weight<=0x1.fffffep+127F))
   return MoEOverlaySharedRowDecision::Invalid;
  const auto expert=static_cast<std::int32_t>(value);
  if(static_cast<float>(expert)!=value)return MoEOverlaySharedRowDecision::Invalid;
  if(launch.shared_row_target_count==0u) {
   std::int32_t selected=-1;
   if(!launch.static_joint_membership.validFor(launch.placement.expert_count,launch.target_participant_id,launch.model_layer_index) ||
      !launch.static_joint_membership.acceptsEpoch(epoch) || !launch.static_joint_membership.resolve(expert,routes[expert],&selected))
    return MoEOverlaySharedRowDecision::Invalid;
   needed=needed || selected==launch.target_participant_id;
  }
  if(launch.shared_row_target_count!=0u) {
   std::int32_t selected=-1;
   if(!launch.shared_joint_union.validFor(launch.placement.expert_count,launch.model_layer_index,epoch) ||
      !launch.shared_joint_union.accepts(expert,routes[expert]))
    return MoEOverlaySharedRowDecision::Invalid;
   needed=needed || launch.shared_joint_union.contains(expert);
   for(std::uint32_t member=0;member<launch.shared_row_target_count;++member)
    needed=needed || routes[expert]==launch.shared_row_controls[member].channel.target_participant_id;
  }
 }
 return needed?MoEOverlaySharedRowDecision::Needed:MoEOverlaySharedRowDecision::Unneeded;
}
/** @brief Write one useful element after complete-prefix validation and row selection. */
LLAMINAR_SHARED_ROW_HD inline void copyMoEOverlaySharedRowElement(
 const MoEOverlayActivationDispatchPackLaunch& launch,std::int32_t row,std::int32_t column) noexcept {
 const auto index=static_cast<std::size_t>(row)*launch.packet.d_model+column;
 launch.packet.hidden_rows_fp32[index]=launch.hidden_rows_fp32[index];
}

}
#undef LLAMINAR_SHARED_ROW_HD
