/**
 * @file ProjectionRowWorkBinding.cpp
 * @brief Authenticate original query-work owners and publish immutable address metadata.
 *
 * Static tensor identities and offsets select a layout before admission. Native
 * preparation then retains exactly those original allocations. Only arbitrary
 * owners need a table; its named SerialGraphFamily region is charged and owned
 * by the ordinary workspace authority and cannot alias transaction scratch.
 */
#include "ProjectionRowWorkBinding.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/TensorClasses.h"
#include "transfer/TransferEngine.h"
#include <limits>
#include <sstream>
#include <stdexcept>

namespace llaminar2 {
ProjectionRowWorkBinding::ProjectionRowWorkBinding(ProjectionRowWorkDeclaration declaration,
    DeviceId device,int physicalRows,std::optional<DeviceRowRange> prefix,
    std::optional<DeviceRequestRowRanges> requests)
    :declaration_(std::move(declaration)),device_(device),physicalRows_(physicalRows),prefix_(prefix),requests_(requests)
{
    const auto &sources=declaration_.sources;
    if(!device_.is_valid()||(!device_.is_cpu()&&!device_.is_gpu())||sources.empty()||
       declaration_.rowsPerRequest<=0||declaration_.sourceCapacity<declaration_.rowsPerRequest||
       declaration_.firstRequiredLogicalRow<0||physicalRows_<=0||
       sources.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()/declaration_.rowsPerRequest)||
       static_cast<int>(sources.size())*declaration_.rowsPerRequest!=physicalRows_||
       (prefix_&&requests_)||(prefix_&&prefix_->physicalRows()!=physicalRows_)||
       (requests_&&(requests_->physicalRows()!=physicalRows_||requests_->requests()!=static_cast<int>(sources.size())||
                   requests_->rowsPerRequest()!=declaration_.rowsPerRequest)))
        throw std::invalid_argument("Projection work requires exact original request geometry and one inherited count source");
    // Declaration geometry is complete and unexpanded. Reject an invalid
    // inherited source before any workspace publication or source lease exists.
    if(prefix_) {
        const auto identity=prefix_->captureIdentity();
        if(identity.first_physical_row!=0||identity.repeats_per_row!=1||
           (identity.count&&(sources.size()!=1||identity.count_capacity!=declaration_.rowsPerRequest||
            reinterpret_cast<std::uintptr_t>(identity.count)%alignof(std::int32_t))))
            throw std::invalid_argument("Projection work inherited prefix does not match original request coordinates");
    }
    if(requests_&&requests_->lengthOwner()&&
       reinterpret_cast<std::uintptr_t>(requests_->lengthOwner())%alignof(std::int32_t))
        throw std::invalid_argument("Projection work inherited request count is not aligned INT32 storage");
    std::ostringstream name;
    name<<"projection_original_rows_"<<device_.to_string();
    for(const auto &source:sources) {
        if(!source.tensor||source.tensor->native_type()!=TensorType::INT32||
           static_cast<std::size_t>(source.buffer)>=static_cast<std::size_t>(BufferId::_COUNT)||
           source.frontierByteOffset%sizeof(std::int32_t)||source.liveByteOffset%sizeof(std::int32_t)||
           source.frontierByteOffset==source.liveByteOffset||
           source.frontierByteOffset>source.tensor->size_bytes()||sizeof(std::int32_t)>source.tensor->size_bytes()-source.frontierByteOffset||
           source.liveByteOffset>source.tensor->size_bytes()||sizeof(std::int32_t)>source.tensor->size_bytes()-source.liveByteOffset)
            throw std::invalid_argument("Projection work requires exact original INT32 frontier/live tensor subviews");
        sourceBytes_.push_back(source.tensor->size_bytes());
        // Names preserve the complete immutable owner tuple. Different live
        // declarations cannot collide and accidentally share a persistent slot.
        name<<'_'<<reinterpret_cast<std::uintptr_t>(source.tensor)<<'_'<<source.frontierByteOffset<<'_'<<source.liveByteOffset;
    }
    tableName_=name.str();
    strideWords_=1;
    if(sources.size()>1) {
        const auto &first=sources.front();
        const auto &second=sources[1];
        if(second.tensor!=first.tensor||second.frontierByteOffset<=first.frontierByteOffset||
           second.liveByteOffset<=first.liveByteOffset)strideWords_=0;
        else {
            const auto stride=second.frontierByteOffset-first.frontierByteOffset;
            if(stride!=second.liveByteOffset-first.liveByteOffset||
               stride/sizeof(std::int32_t)>static_cast<std::size_t>(std::numeric_limits<int>::max()))strideWords_=0;
            else {
                strideWords_=static_cast<int>(stride/sizeof(std::int32_t));
                for(std::size_t r=1;r<sources.size();++r)
                    if(sources[r].tensor!=first.tensor||
                       r>(std::numeric_limits<std::size_t>::max()-first.frontierByteOffset)/stride||
                       r>(std::numeric_limits<std::size_t>::max()-first.liveByteOffset)/stride||
                       sources[r].frontierByteOffset!=first.frontierByteOffset+r*stride||
                       sources[r].liveByteOffset!=first.liveByteOffset+r*stride)strideWords_=0;
            }
        }
    }
}

void ProjectionRowWorkBinding::appendWorkspaceRequirements(WorkspaceRequirements &requirements) const
{
    if(strideWords_)return;
    requirements.buffers.emplace_back(tableName_,declaration_.sources.size()*sizeof(ProjectionRequestPublication),
        256,true,WorkspaceExecutionRegime::CompactDecodeOnly,WorkspaceContentLifetime::SerialGraphFamily);
}

const std::int32_t *ProjectionRowWorkBinding::nativeBase(const ProjectionRowSource &source) const
{
    const auto *base=static_cast<const std::int32_t *>(device_.is_cpu()?source.tensor->raw_data():source.tensor->gpu_data_ptr());
    if(!base||reinterpret_cast<std::uintptr_t>(base)%alignof(std::int32_t)||
       (device_.is_gpu()&&source.tensor->current_device()!=std::optional{device_}))
        throw std::logic_error("Projection work original metadata was not allocation-prebound on its participant");
    return base;
}

void ProjectionRowWorkBinding::prepare(DeviceWorkspaceManager *workspace,void *stream)
{
    if(prepared()) {requirePrepared(workspace,stream);return;}
    if(isGraphCaptureActive()||(device_.is_gpu()?reinterpret_cast<std::uintptr_t>(stream)<=2:stream!=nullptr))
        throw std::logic_error("Projection row work must prepare on its exact participant outside capture");
    std::vector<const std::int32_t *> bases;
    std::vector<std::shared_ptr<const CapturedDeviceStorageLease>> leases;
    std::vector<ProjectionRequestPublication> entries;
    for(std::size_t r=0;r<declaration_.sources.size();++r) {
        const auto &source=declaration_.sources[r];
        if(source.tensor->size_bytes()!=sourceBytes_[r])
            throw std::logic_error("Projection row source capacity changed after declaration");
        const auto *base=nativeBase(source);
        bases.push_back(base);
        entries.push_back({base+source.frontierByteOffset/sizeof(std::int32_t),base+source.liveByteOffset/sizeof(std::int32_t)});
        if(device_.is_gpu())leases.push_back(TransferEngine{}.retainCapturedDeviceStorage(*source.tensor,device_,sourceBytes_[r]));
    }
    const int requests=static_cast<int>(entries.size());
    std::shared_ptr<const ProjectionRowPublicationTable> table;
    std::optional<ProjectionRowWork> result;
    if(strideWords_)
        result=ProjectionRowWork::causalStrided(requests,declaration_.rowsPerRequest,
            entries.front().frontier,entries.front().live,strideWords_,declaration_.sourceCapacity,declaration_.firstRequiredLogicalRow);
    else {
        if(!workspace||workspace->device()!=device_||!workspace->id()||!workspace->isAllocated())
            throw std::logic_error("Projection row table requires its admitted original participant workspace");
        PersistentWorkspacePublicationKey key{.word0=workspace->id(),.word1=1,.word2=static_cast<std::uint64_t>(requests),.word3=sizeof(ProjectionRequestPublication)};
        for(const auto &entry:entries) {
            key.identity_words.push_back(reinterpret_cast<std::uintptr_t>(entry.frontier));
            key.identity_words.push_back(reinterpret_cast<std::uintptr_t>(entry.live));
        }
        const auto published=workspace->getOrCreatePersistentPublication(tableName_,key,1,[&](std::size_t slot)->std::shared_ptr<void> {
            if(slot!=0)throw std::logic_error("Projection table publication acquired a foreign exclusive slot");
            const auto bytes=entries.size()*sizeof(ProjectionRequestPublication);
            const auto *native=workspace->getPersistentSlotBuffer(tableName_,1,slot,bytes);
            auto lease=workspace->retainBuffer(tableName_,0,bytes);
            if(!native||native!=lease->data())throw std::logic_error("Projection table lost its exact named workspace extent");
            TransferEngine{}.initializeWorkspaceMetadata(*lease,std::as_bytes(std::span{entries}),stream);
            return std::shared_ptr<ProjectionRowPublicationTable>(new ProjectionRowPublicationTable(std::move(lease),workspace->id(),entries));
        });
        if(!published)throw std::logic_error("Projection row work could not publish its exact immutable address table");
        table=std::static_pointer_cast<ProjectionRowPublicationTable>(published.publication);
        result=ProjectionRowWork::causalIndirect(requests,declaration_.rowsPerRequest,table->data(),
            declaration_.sourceCapacity,declaration_.firstRequiredLogicalRow,table->generation());
    }
    if(prefix_)result=result->withInheritedPrefix(*prefix_);
    if(requests_)result=result->withInheritedRequests(*requests_);
    bases_=std::move(bases);sourceLeases_=std::move(leases);table_=std::move(table);
    workspaceId_=table_?workspace->id():0;stream_=stream;work_=result;
    requirePrepared(workspace,stream);
}

void ProjectionRowWorkBinding::requirePrepared(DeviceWorkspaceManager *workspace,void *stream) const
{
    if(!work_||stream!=stream_||bases_.size()!=declaration_.sources.size())
        throw std::logic_error("Projection row work lost its exact prepared launch identity");
    for(std::size_t r=0;r<declaration_.sources.size();++r) {
        const auto &source=declaration_.sources[r];
        if(source.tensor->size_bytes()!=sourceBytes_[r]||nativeBase(source)!=bases_[r])
            throw std::logic_error("Projection row source storage changed after preparation");
        if(device_.is_gpu())TransferEngine{}.requireCapturedDeviceStorage(*sourceLeases_.at(r),*source.tensor,device_,sourceBytes_[r]);
    }
    if(table_&&(!workspace||workspace->id()!=workspaceId_||workspace->device()!=device_||
       workspace->getPersistentSlotBuffer(tableName_,1,0,table_->storage().sizeBytes())!=table_->data()||
       table_->generation()!=workspaceId_))
        throw std::logic_error("Projection row table storage changed after immutable publication");
}

void ProjectionRowWorkBinding::requirePublishedInputs() const
{
    if(!work_)throw std::logic_error("Projection work cannot observe inputs before address preparation");
    if(device_.is_cpu())for(const auto &source:declaration_.sources)
        if(!source.tensor->hostValid())
            throw std::logic_error("CPU projection work original publication lacks host read authority");
}

const ProjectionRowWork &ProjectionRowWorkBinding::work() const
{
    if(!work_)throw std::logic_error("Projection row work has not been prepared");
    return *work_;
}

std::uint64_t ProjectionRowWorkBinding::signature() const noexcept
{
    std::uint64_t value=0x50524f4a574f524bULL;
    const auto mix=[&](std::uint64_t word){value^=word+0x9e3779b97f4a7c15ULL+(value<<6)+(value>>2);};
    mix(static_cast<std::uint64_t>(device_.type));mix(device_.ordinal);mix(physicalRows_);
    mix(declaration_.rowsPerRequest);mix(declaration_.sourceCapacity);mix(declaration_.firstRequiredLogicalRow);
    for(std::size_t r=0;r<declaration_.sources.size();++r) {
        const auto &source=declaration_.sources[r];
        mix(reinterpret_cast<std::uintptr_t>(source.tensor));mix(static_cast<std::uint64_t>(source.buffer));
        mix(source.frontierByteOffset);mix(source.liveByteOffset);mix(sourceBytes_[r]);
    }
    return value?value:1;
}
}
