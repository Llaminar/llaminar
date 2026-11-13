/**
 * @file PlanningPCIeCommunicationCost.cpp
 * @brief Exact cut membership and observed PCIe line-rate lower bounds without a transport guess.
 *
 * Only endpoints naming the same exact bottleneck BDF can attest its effective
 * negotiated rate. Missing/conflicting observations are unknown. A transfer
 * between two descendants never consumes their shared upstream link. Opposite
 * directions are full duplex; taking a maximum avoids inventing serialization.
 */
#include "PlanningPCIeCommunicationCost.h"
#include "RankHardwareOwnership.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace llaminar2 {
namespace {
/** @brief Add live bytes without saturating or silently discarding traffic. */
size_t add(size_t a,size_t b) {
    if(b>std::numeric_limits<size_t>::max()-a)throw std::overflow_error("PCIe cut live byte sum overflow");
    return a+b;
}
/** @return Known protocol encoding efficiency for the observed negotiated GT/s, or unknown. */
std::optional<double> encoding(double speed) {
    if(speed==2.5 || speed==5)return .8;
    if(speed==8 || speed==16 || speed==32)return 128.0/130.0;
    return {};
}
}
/** @brief Immutable original endpoint and cut observations used by every demand. */
struct PlanningPCIeCommunicationCost::Impl {
    RankInventory inventory;
    struct Rate {int lanes=0;double speed=0;bool initialized=false,consistent=true;};
    std::map<std::string,Rate> rates;
    explicit Impl(const RankInventory &original):inventory(original) {
        if(inventory.rank<0)throw std::invalid_argument("PCIe cut observer has invalid discovery rank");
        std::set<std::pair<DeviceType,int>> endpoints;
        for(const auto &gpu:inventory.gpus) {
            if(!endpoints.emplace(gpu.type,gpu.local_device_id).second)
                throw std::invalid_argument("PCIe cut observer has ambiguous original GPU ordinal");
            if(!rankOwnsGPUDevice(inventory,gpu) || gpu.pcie_bottleneck_bdf.empty())continue;
            auto &rate=rates[gpu.pcie_bottleneck_bdf];
            if(rate.initialized && (rate.lanes!=gpu.pcie_width || rate.speed!=gpu.pcie_speed_gts))rate.consistent=false;
            rate.initialized=true;rate.lanes=gpu.pcie_width;rate.speed=gpu.pcie_speed_gts;
            if(gpu.pcie_width<=0 || !std::isfinite(gpu.pcie_speed_gts) || !encoding(gpu.pcie_speed_gts) ||
                (gpu.pci_bus_address!=gpu.pcie_bottleneck_bdf &&
                 std::find(gpu.pcie_upstream_bdfs.begin(),gpu.pcie_upstream_bdfs.end(),gpu.pcie_bottleneck_bdf)==gpu.pcie_upstream_bdfs.end()))
                rate.consistent=false;
        }
    }
    /** @return Exact original NUMA-owned GPU; discovery visibility never grants execution. */
    const DeviceInfo &gpu(DeviceId id) const {
        if(!id.is_gpu())throw std::invalid_argument("PCIe cut demands require GPU endpoints");
        const auto found=std::find_if(inventory.gpus.begin(),inventory.gpus.end(),[&](const auto &g){return g.type==id.type && g.local_device_id==id.ordinal;});
        if(found==inventory.gpus.end())throw std::invalid_argument("PCIe cut demand has foreign GPU ordinal");
        requireRankLocalGPUOwnership(inventory,*found,"PCIe cut demand");return *found;
    }
    /** @return Original path completeness needed to qualify descendant absence. */
    bool complete(const DeviceInfo &g) const {
        return !g.pci_bus_address.empty() && !g.pcie_bottleneck_bdf.empty() &&
            (!g.pcie_upstream_bdfs.empty() || g.pcie_bottleneck_bdf==g.pci_bus_address);
    }
    /** @return Exact known descendant membership, including an endpoint's own link cut. */
    bool inside(const DeviceInfo &g,const std::string &bdf) const {
        return g.pci_bus_address==bdf || std::find(g.pcie_upstream_bdfs.begin(),g.pcie_upstream_bdfs.end(),bdf)!=g.pcie_upstream_bdfs.end();
    }
    /** @brief Validate arbitrary participant counts without guessing hardware topology. */
    std::vector<const DeviceInfo*> validate(std::span<const DeviceId> devices) const {
        std::vector<const DeviceInfo*> endpoints;std::set<std::pair<DeviceType,int>> seen;
        for(const auto id:devices) {
            if(!devices.empty() && id.type!=devices.front().type)throw std::invalid_argument("PCIe cut demand mixes GPU backends");
            if(!seen.emplace(id.type,id.ordinal).second)throw std::invalid_argument("PCIe cut demand repeats a GPU endpoint");
            endpoints.push_back(&gpu(id));
        }
        return endpoints;
    }
    /** @brief Evaluate only live crossing candidate cuts; unresolved observations remain explicit. */
    template<class Demand> PlanningPCIeCutBound evaluate(const std::vector<const DeviceInfo*> &devices,Demand demand) const {
        PlanningPCIeCutBound result;
        for(const auto *g:devices)if(!complete(*g))result.topology_complete=false;
        for(const auto &[bdf,rate]:rates) {
            const auto [outbound,inbound]=demand(bdf);
            if(!outbound && !inbound)continue;
            PlanningPCIeCutObservation cut{bdf,outbound,inbound,{ },0,{}};
            if(result.topology_complete && rate.consistent && rate.initialized) {
                const auto factor=encoding(rate.speed);
                const double capacity=rate.speed*1e9*rate.lanes*factor.value_or(0)/8;
                if(std::isfinite(capacity) && capacity>0)cut.capacity_bytes_per_second=capacity;
            }
            if(cut.capacity_bytes_per_second) {
                cut.seconds=static_cast<double>(std::max(outbound,inbound))/ *cut.capacity_bytes_per_second;
                cut.qualification="observed negotiated PCIe encoding wire-capacity floor; not measured payload bandwidth or collective latency";
                ++result.known_cuts;result.seconds=std::max(result.seconds,cut.seconds);
            } else {++result.unknown_cuts;cut.qualification="unqualified PCIe cut: missing/conflicting path, width, speed or supported encoding";}
            result.evidence.push_back(bdf+": "+cut.qualification);result.cuts.push_back(std::move(cut));
        }
        if(!result.topology_complete)result.evidence.push_back("incomplete original endpoint ancestry/bottleneck identity; no complete cut-coverage claim");
        return result;
    }
};
PlanningPCIeCommunicationCost::PlanningPCIeCommunicationCost(const RankInventory &inventory):impl_(std::make_shared<const Impl>(inventory)){}
PlanningPCIeCutBound PlanningPCIeCommunicationCost::allreduce(std::span<const DeviceId> devices,size_t bytes) const {
    if(devices.empty() || !bytes)throw std::invalid_argument("PCIe allreduce requires participants and positive live payload");
    const auto endpoints=impl_->validate(devices);
    if(devices.size()==1)return {};
    return impl_->evaluate(endpoints,[&](const std::string &bdf) {
        const auto count=std::count_if(endpoints.begin(),endpoints.end(),[&](const auto *g){return impl_->inside(*g,bdf);});
        const bool crossing=count>0 && static_cast<size_t>(count)<endpoints.size();
        return std::pair{crossing?bytes:0u,crossing?bytes:0u};
    });
}
PlanningPCIeCutBound PlanningPCIeCommunicationCost::allgather(std::span<const DeviceId> devices,
    std::span<const size_t> owner_bytes) const {
    if(devices.empty() || owner_bytes.size()!=devices.size())
        throw std::invalid_argument("PCIe allgather requires one live fragment per original owner");
    const auto endpoints=impl_->validate(devices);
    size_t total=0;for(const auto bytes:owner_bytes)total=add(total,bytes);
    if(!total)throw std::invalid_argument("PCIe allgather requires a positive total live payload");
    if(devices.size()==1)return {};
    return impl_->evaluate(endpoints,[&](const std::string &bdf) {
        size_t inside_count=0,outbound=0,inbound=0;
        for(size_t index=0;index<endpoints.size();++index) {
            if(impl_->inside(*endpoints[index],bdf)) {++inside_count;outbound=add(outbound,owner_bytes[index]);}
            else inbound=add(inbound,owner_bytes[index]);
        }
        if(!inside_count || inside_count==endpoints.size())return std::pair<size_t,size_t>{0,0};
        return std::pair{outbound,inbound};
    });
}
PlanningPCIeCutBound PlanningPCIeCommunicationCost::transfers(std::span<const PlanningPCIeDirectedTransfer> edges) const {
    std::vector<DeviceId> ids;
    for(const auto &edge:edges) {
        if(!edge.live_bytes || edge.source==edge.destination)throw std::invalid_argument("PCIe transfer requires distinct endpoints and positive live bytes");
        for(const auto id:{edge.source,edge.destination})if(std::find(ids.begin(),ids.end(),id)==ids.end())ids.push_back(id);
    }
    const auto endpoints=impl_->validate(ids);
    return impl_->evaluate(endpoints,[&](const std::string &bdf) {
        size_t outbound=0,inbound=0;
        for(const auto &edge:edges) {
            const bool source=impl_->inside(impl_->gpu(edge.source),bdf),destination=impl_->inside(impl_->gpu(edge.destination),bdf);
            if(source==destination)continue;
            if(source)outbound=add(outbound,edge.live_bytes);else inbound=add(inbound,edge.live_bytes);
        }
        return std::pair{outbound,inbound};
    });
}
}
