/**
 * @file LocalPPTransferSelection.h
 * @brief Exact initial PP destination shared by transfer execution and admission.
 *
 * A domain receives through its first same-vendor member when available, then
 * broadcasts through its own admitted TP authority. Router copies own only this
 * initial boundary edge, never every member of the destination domain.
 */
#pragma once
#include "backends/DeviceId.h"
#include <span>
#include <limits>
#include <stdexcept>
namespace llaminar2 {
/**
 * @brief Select the existing PP receiver without copying its immutable topology.
 * @param source Actual representative source endpoint.
 * @param count Original destination-domain population.
 * @param destination Read-only coordinate accessor into that same original domain.
 * @return First same-vendor GPU coordinate, otherwise the original leader0.
 * @throws std::invalid_argument For a missing or unrepresentable domain.
 */
template<class Destination>
inline int localPPInitialDestination(DeviceId source,std::size_t count,Destination&& destination) {
    if(!count || count>static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("PP transfer requires its original representable destination domain");
    if(source.is_gpu())for(std::size_t index=0;index<count;++index)
        if(destination(index).type==source.type)return static_cast<int>(index);
    return 0;
}
/**
 * @brief Apply the same receiver policy to a cold admitted endpoint span.
 * @param source Actual representative source endpoint.
 * @param destinations Original ordered destination-domain endpoints.
 * @return Existing initial destination coordinate, without allocation or discovery.
 * @throws std::invalid_argument For a missing or unrepresentable domain.
 */
inline int localPPInitialDestination(DeviceId source,std::span<const DeviceId> destinations) {
    return localPPInitialDestination(source,destinations.size(),[&](std::size_t index){return destinations[index];});
}
/**
 * @brief Select the existing TP source using the destination representative.
 * @param destination Actual representative destination endpoint.
 * @param count Original source TP population, including a singleton TP authority.
 * @param source Read-only accessor into that same original source domain.
 * @return First same-vendor GPU source, otherwise its original leader0.
 * @throws std::invalid_argument For a missing or unrepresentable source domain.
 */
template<class Source>
inline int localPPInitialSource(DeviceId destination,std::size_t count,Source&& source) {
    return localPPInitialDestination(destination,count,source);
}
/** @brief Exact directed boundary coordinates, before destination TP publication. */
struct LocalPPBoundarySelection { int source=0,destination=0; };
/**
 * @brief Resolve both endpoints with the existing source-TP dispatch priority.
 * @param source_is_tp Whether the actual source stage owns a LocalTP context.
 * @param source_count Exact source domain width.
 * @param source Indexed read-only source endpoints.
 * @param destination_is_tp Whether the actual destination owns a LocalTP context.
 * @param destination_count Exact destination domain width.
 * @param destination Indexed read-only destination endpoints.
 * @return Source TP chooses against destination leader; otherwise destination TP
 * chooses against source leader. This order is also the runtime dispatch order.
 * @throws std::invalid_argument For missing or unrepresentable domain geometry.
 */
template<class Source,class Destination>
inline LocalPPBoundarySelection localPPBoundarySelection(bool source_is_tp,std::size_t source_count,Source&& source,
    bool destination_is_tp,std::size_t destination_count,Destination&& destination) {
    // Validate both original populations without allocating a temporary roster.
    (void)localPPInitialDestination(DeviceId::cpu(),source_count,source);
    (void)localPPInitialDestination(DeviceId::cpu(),destination_count,destination);
    if(source_is_tp)return {localPPInitialSource(destination(0),source_count,source),0};
    if(destination_is_tp)return {0,localPPInitialDestination(source(0),destination_count,destination)};
    return {};
}

}
