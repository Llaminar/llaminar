/**
 * @file MoEOverlayActivationBatchResources.h
 * @brief One declared Batch storage/native-resource plan for production and native replay.
 *
 * Immutable descriptor roots retain their own per-capture bytes. Internal
 * default-record/wait events and named lane streams are context-owned and
 * priced once at peak simultaneous lane geometry, not per layer or retained
 * shape. A separate cold publication event is reused only after its actual
 * TransferEngine copy completion. Every backing is issued by the original PMA.
 */
#pragma once
#include "MoEOverlayActivationChannelPlan.h"
#include "MoEOverlayNodeLocalRankBatchTransport.h"
#include "backends/NativeOrderingEventGroup.h"
#include "backends/NativeAuxiliaryStreamOwnership.h"
#include "backends/GPUCaptureDependencyRoster.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "memory/CPUTensorStorageDeclaration.h"
#include <limits>
#include <array>
#include <span>
namespace llaminar2 {
/** @brief Original private workspace plus optional immutable CPU upload root. */
class MoEOverlayActivationBatchStorage final {
public:
    /** @brief Exact retained allocation/role; there is no missing-authority arm. */
    struct Config {
        DeviceId device=DeviceId::invalid();
        MoEOverlayGraphStorageType type=MoEOverlayGraphStorageType::Int32;
        std::vector<std::size_t> shape;
        bool immutable_input=false;
        std::string identity;
        std::shared_ptr<PhysicalMemoryAuthority> physical_memory_authority;
        std::shared_ptr<GPUCaptureDependencyRoster> context_resources;
    };
    /** @brief Materialize exact schema-owned CPU roots and original stable GPU lease cold. */
    explicit MoEOverlayActivationBatchStorage(Config);
    /** @brief Retire upload root and workspace after all graph users drain. */
    ~MoEOverlayActivationBatchStorage();
    /** @return Single-region layout shared by quote and materialization. */
    static WorkspaceRequirements workspaceRequirements(std::size_t bytes);
    /** @return Exact CPU upload declaration shared by quote and materialization. */
    static CPUTensorStorageDeclaration uploadDeclaration(std::size_t bytes);
    /** @brief Publish once through TransferEngine; later calls require byte-identical metadata. */
    bool publishImmutableBytes(const void*,std::size_t,void*);
    /** @return Fully completed immutable bytes on the exact nondefault consumer stream. */
    bool requireInput(void*)const;
    /** @return Original dedicated graph-private output on the exact nondefault producer stream. */
    bool requireOutput(void*)const;
    /** @return Capture-stable original device address. */
    void* deviceData()const noexcept{return lease_?lease_->data():nullptr;}
    /** @return Logical payload bytes, excluding allocation alignment. */
    std::size_t sizeBytes()const noexcept{return bytes_;}
    /** @return Whether original device storage exists. */
    bool allocated()const noexcept{return lease_!=nullptr;}
    /** @return Whether immutable initialization completed before capture. */
    bool published()const noexcept{return published_;}
private:
    Config config_;std::size_t bytes_=0;
    std::unique_ptr<DeviceWorkspaceManager> workspace_;
    std::shared_ptr<const WorkspaceBufferLease> lease_;
    std::unique_ptr<TensorBase> upload_;
    bool published_=false;
};
/** @brief Pure exact declarations, never a second live ledger or a native creator. */
class MoEOverlayActivationBatchResources final {
public:
    /** @return Same exact retained CPU grant root used by every CPU/GPU endpoint. */
    static CPUTensorStorageDeclaration grantUploadDeclaration(){
        static_assert(sizeof(MoEOverlayActivationDeviceEpochGrant)%sizeof(std::int32_t)==0);
        return MoEOverlayActivationBatchStorage::uploadDeclaration(sizeof(MoEOverlayActivationDeviceEpochGrant));
    }
    /** @return Same allocator-rounded GPU grant slab used by the actual storage owner. */
    static std::size_t grantGPUBytes(){
        return MoEOverlayActivationBatchStorage::workspaceRequirements(sizeof(MoEOverlayActivationDeviceEpochGrant)).total_bytes_with_alignment();
    }
    /** @return Existing one-event cold publication allowance, once per actual GPU context. */
    static PhysicalMemoryPlan coldPublicationBOM(PhysicalMemoryResource host,PhysicalMemoryResource gpu){
        return NativeOrderingEventGroup::bom(NativeOrderingEventProgram::captureSubmissionRetirement(),host,gpu);
    }
    /** @return Actual source lane streams and peak internal capture events, once across variants/layers. */
    static PhysicalMemoryPlan sharedLaneStreamsBOM(const MoEOverlayNodeLocalActivationChannelPlan& channel,
        PhysicalMemoryResource host,PhysicalMemoryResource gpu){
        const std::span channels{&channel,std::size_t{1}};
        return sharedLaneStreamsBOM(channels,host,gpu);
    }
    /** @return Combined simultaneous original source geometry across all rank-pair channels. */
    static PhysicalMemoryPlan sharedLaneStreamsBOM(std::span<const MoEOverlayNodeLocalActivationChannelPlan> channels,
        PhysicalMemoryResource host,PhysicalMemoryResource gpu){
        std::size_t lanes=0,groups=0;
        for(const auto& channel:channels){
            if(channel.source_world_rank!=gpu.world_rank || channel.source_device!=gpu.device || channel.target_lanes.empty())
                throw std::invalid_argument("Batch source declaration changed original channel owner/geometry");
            if(channel.target_lanes.size()>std::numeric_limits<std::size_t>::max()-lanes)
                throw std::overflow_error("Batch source lane population overflow");
            lanes+=channel.target_lanes.size();++groups;
        }
        if(!lanes)throw std::invalid_argument("Batch source declaration has no actual lanes");
        PhysicalMemoryPlanBuilder result;
        const auto add=[&](const PhysicalMemoryPlan& p){for(const auto& row:p.resources())result.add(row);};
        // One external completed upload event, one internal fork, lane grants,
        // and actual physical shared dispatch groups. Event leases are released
        // after recording, while handles remain original context-owned.
        for(std::size_t i=0;i<2+lanes+groups;++i)add(coldPublicationBOM(host,gpu));
        for(std::size_t i=0;i<lanes;++i)add(NativeAuxiliaryStreamOwnership::bom(host,gpu));
        return result.build();
    }
    /** @return Exact serial cache-slot peak across its compact and wide descriptor layouts.
     * @details A named replaceable prefill identity owns only one active layout.
     * Row1 has two descriptor arrays for a lane batch; wide rows have one. The
     * physical per-owner maximum comes from those actual shared declarations,
     * never maximum_rows times a per-row allocation or an independent reserve.
     */
    static PhysicalMemoryPlan variableSourceBOM(std::span<const MoEOverlayNodeLocalActivationChannelPlan> channels,
        int maximum_rows,std::uint32_t ordinal,PhysicalMemoryResource host,PhysicalMemoryResource gpu){
        if(maximum_rows<=0)throw std::invalid_argument("Variable Batch source requires its positive admitted row envelope");
        const std::array layouts{bom(channels,1,ordinal,host,gpu),bom(channels,maximum_rows,ordinal,host,gpu)};
        PhysicalMemoryPlanBuilder result;result.addMutuallyExclusive(layouts);return result.build();
    }
    /** @return Exact descriptor and validity roots for ONE actual retained Batch state. */
    static PhysicalMemoryPlan bom(const MoEOverlayNodeLocalActivationChannelPlan& channel,int rows,
        std::uint32_t ordinal,PhysicalMemoryResource host,PhysicalMemoryResource gpu){
        return bom(std::span{&channel,std::size_t{1}},rows,ordinal,host,gpu);
    }
    /** @return One combined descriptor root over the actual source lane union. */
    static PhysicalMemoryPlan bom(std::span<const MoEOverlayNodeLocalActivationChannelPlan> channels,int rows,
        std::uint32_t ordinal,PhysicalMemoryResource host,PhysicalMemoryResource gpu){
        (void)ordinal;
        if(rows<=0 || !host.valid() || !gpu.valid() || host.world_rank!=gpu.world_rank ||
           !gpu.device.is_gpu())
            throw std::invalid_argument("Batch storage declaration changed original role/rows/resources");
        std::size_t lanes=0;
        for(const auto& channel:channels){
            if(channel.source_world_rank!=gpu.world_rank || channel.source_device!=gpu.device ||
               channel.target_lanes.empty())throw std::invalid_argument("Batch storage declaration changed original channel source");
            if(channel.target_lanes.size()>std::numeric_limits<std::size_t>::max()-lanes)
                throw std::overflow_error("Batch descriptor lane union overflow");
            lanes+=channel.target_lanes.size();
        }
        if(!lanes)throw std::invalid_argument("Batch storage declaration has no actual lanes");
        if(rows==1 && lanes==1)return PhysicalMemoryPlanBuilder{}.build(); // Existing scalar direct stage has no Batch roots.
        PhysicalMemoryPlanBuilder result;
        const auto region=[&](std::size_t element_bytes,bool immutable){
            if(lanes>std::numeric_limits<std::size_t>::max()/element_bytes)throw std::overflow_error("Batch descriptor bytes overflow");
            const auto bytes=lanes*element_bytes;
            result.add(gpu,PhysicalMemoryOwner::ActivationTransportStaging,
                MoEOverlayActivationBatchStorage::workspaceRequirements(bytes).total_bytes_with_alignment());
            if(immutable)result.add(host,PhysicalMemoryOwner::ActivationTransportStaging,
                MoEOverlayActivationBatchStorage::uploadDeclaration(bytes).physicalBytes());
        };
        if(rows>1)region(sizeof(MoEOverlayActivationReturnConsumeLaunch),true);
        else {region(sizeof(MoEOverlayActivationSingleRowDispatchPackLaunch),true);region(sizeof(MoEOverlayActivationSingleRowReturnConsumeLaunch),true);}
        region(sizeof(std::int32_t),false);
        return result.build();
    }
};
}
