/**
 * @file CPUExpertDomainSocketMembership.h
 * @brief Validate recorded original rank/NUMA/physical-socket endpoints for native tests.
 *
 * This predicate owns no topology or admission state. Native fixtures provide
 * observed NodeTopology socket identities; recorded Unit inputs are not hardware
 * evidence. One original rank per distinct physical socket is mandatory.
 */
#pragma once
#include <span>
namespace llaminar2::test {
/** @brief Actual local endpoint projected from original inventory and observed affinity. */
struct CPUExpertSocketEndpoint {int rank; int numa_node; int socket;};
/**
 * @brief Reject duplicate ranks/sockets/NUMA or incomplete native launch membership.
 * @param endpoints Original rank-local observations, in arbitrary arrival order.
 * @param ranks Exact requested original MPI world population.
 * @return True only for complete distinct-socket membership within supported 1–8 geometry.
 */
inline bool validCPUExpertSocketMembership(std::span<const CPUExpertSocketEndpoint> endpoints,int ranks) noexcept {
    if(ranks<1 || ranks>8 || endpoints.size()!=static_cast<unsigned>(ranks))return false;
    for(unsigned i=0;i<endpoints.size();++i) {
        const auto& item=endpoints[i];
        if(item.rank<0 || item.rank>=ranks || item.numa_node<0 || item.socket<0)return false;
        for(unsigned j=0;j<i;++j)
            if(endpoints[j].rank==item.rank || endpoints[j].numa_node==item.numa_node || endpoints[j].socket==item.socket)return false;
    }
    return true;
}
}
