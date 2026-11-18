/**
 * @file OriginalM1HiddenQ8Publication.h
 * @brief Cold physical seals for one original fully-active M1 hidden-Q8 publication.
 *
 * The router's existing PMA workspace owns codes/scales. This metadata retains
 * their original allocation leases and exact source/stream identity; it owns
 * no activation mirror, allocation or mutable device execution state. A consumer
 * keeps its independent workspace and cannot publish or invalidate the bank.
 */
#pragma once
#include "backends/ExplicitGPUStream.h"
#include "backends/BackendManager.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "execution/local_execution/device/WorkspaceBufferLease.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2 {
/** @brief Existing payload provenance remains distinct for capture recording and ordered eager use. */
enum class OriginalM1HiddenQ8Use : std::uint8_t { OrderedEager, CaptureRecording };
/** @brief Cold reader lifecycle; a hard scratch reset cannot authenticate an old retained graph. */
enum class OriginalM1HiddenQ8ReadLifecycle : std::uint8_t { Cold, Prepared, Invalidated };
/** @brief Exact existing native workspace bank, never a caller-supplied pointer/name convention. */
enum class OriginalM1HiddenQ8Bank : std::uint8_t { RouterROCm, OriginalCUDA, ConsumerScratch };
/** @brief Retain the exact original names and physical bytes without a second allocation authority. */
class OriginalM1HiddenQ8WorkspaceSeal final {
    struct Region {std::string name;void* pointer;std::size_t bytes;std::shared_ptr<const WorkspaceBufferLease> lease;};
    DeviceWorkspaceManager* owner_;std::uint64_t id_;std::array<Region,2> regions_;
public:
    /** @brief Seal both original prepared code/scale names outside capture; no device state is sampled. */
    OriginalM1HiddenQ8WorkspaceSeal(DeviceWorkspaceManager& owner,OriginalM1HiddenQ8Bank bank)
        :owner_(&owner),id_(owner.id()) {
        if(isGraphCaptureActive()||!owner.isAllocated())throw std::logic_error("M1 hidden Q8 needs cold allocated workspace");
        if(bank!=OriginalM1HiddenQ8Bank::RouterROCm&&bank!=OriginalM1HiddenQ8Bank::OriginalCUDA&&bank!=OriginalM1HiddenQ8Bank::ConsumerScratch)
            throw std::invalid_argument("M1 hidden Q8 has no original bank alternative");
        const bool router=bank==OriginalM1HiddenQ8Bank::RouterROCm;
        const std::array<const char*,2> names{router?MoEWorkspaceBuffers::ROCM_ROUTER_Q8_HIDDEN:MoEWorkspaceBuffers::DECODE_HIDDEN_INT8,
            router?MoEWorkspaceBuffers::ROCM_ROUTER_Q8_SCALES:MoEWorkspaceBuffers::DECODE_HIDDEN_SCALES};
        for(std::size_t i=0;i<regions_.size();++i) {
            auto* pointer=owner.getBuffer(names[i]);const auto bytes=owner.getBufferSize(names[i]);
            if(!pointer||!bytes)throw std::logic_error("M1 hidden Q8 original named bank was not prepared");
            regions_[i]={names[i],pointer,bytes,owner.retainBuffer(names[i],0,bytes)};
        }
    }
    /** @brief Validate the same manager namespace, names, addresses and extents with no allocation registry scan. */
    void validate(DeviceWorkspaceManager* owner) const {
        if(owner!=owner_||!owner||!owner->isAllocated()||owner->id()!=id_)
            throw std::logic_error("M1 hidden Q8 workspace owner changed");
        for(const auto& r:regions_)if(owner->getBuffer(r.name)!=r.pointer||owner->getBufferSize(r.name)!=r.bytes||r.lease->data()!=r.pointer)
            throw std::logic_error("M1 hidden Q8 original named workspace binding changed");
    }
    /** @return Original code storage; only the router is allowed to write it. */
    int8_t* codes() const{return static_cast<int8_t*>(regions_[0].pointer);}
    /** @return Original FP32 scale storage paired with codes. */
    float* scales() const{return static_cast<float*>(regions_[1].pointer);}
    /** @return Whether the original banks cover one exact block32 hidden row. */
    bool covers(int width) const{return width>0&&width%32==0&&regions_[0].bytes>=std::size_t(width)&&regions_[1].bytes>=std::size_t(width/32)*sizeof(float);}
    /** @return Exact original participant owning the physical region. */
    DeviceId device() const{return owner_->device();}
    /** @return Manager identity retained by the original graph owner. */
    DeviceWorkspaceManager* owner() const{return owner_;}
};

/** @brief Immutable checked source and original producer bank for one ordinary fully-active Decode row. */
class OriginalM1HiddenQ8Binding final {
    DeviceId device_;const TensorBase* source_;const TensorBase* storage_owner_;
    const void* address_;std::size_t bytes_;std::vector<std::size_t> shape_;int width_;void* stream_;
    IBackend* backend_;std::uint64_t runtime_;
    std::shared_ptr<const CapturedDeviceStorageLease> source_lease_;
    OriginalM1HiddenQ8WorkspaceSeal producer_;
public:
    /**
     * @brief Bind existing original storage outside capture after ordinary router preparation.
     * @param device Exact producer participant.
     * @param source Original canonical FP32 row authority, already resident.
     * @param width Exact original hidden width, a positive multiple of32.
     * @param stream Exact original non-default producer/consumer stream.
     * @param workspace Original PMA-owned router workspace.
     * @param bank Backend-specific original code/scale bank.
     */
    OriginalM1HiddenQ8Binding(DeviceId device,const TensorBase& source,int width,void* stream,
        DeviceWorkspaceManager& workspace,OriginalM1HiddenQ8Bank bank)
        :device_(device),source_(&source),storage_owner_(source.transferStorageOwner()),address_(source.gpu_data_ptr()),
         bytes_(source.size_bytes()),shape_(source.shape()),width_(width),stream_(ExplicitGPUStream(stream).get()),
         backend_(getBackendFor(device)),runtime_(backend_?backend_->deviceRuntimeGeneration(device.ordinal):0),
         producer_(workspace,bank) {
        if(reinterpret_cast<std::uintptr_t>(stream_)<=2u||!device.is_gpu()||workspace.device()!=device||source.native_type()!=TensorType::FP32||!runtime_||
           source.current_device()!=device||!address_||shape_.size()!=2||shape_[1]!=std::size_t(width)||
           shape_[0]<1||!producer_.covers(width)||
           (device.is_rocm()&&bank!=OriginalM1HiddenQ8Bank::RouterROCm)||
           (device.is_cuda()&&bank!=OriginalM1HiddenQ8Bank::OriginalCUDA))
            throw std::invalid_argument("M1 hidden Q8 requires the exact original prepared fully-active FP32 row/bank");
        const auto overlaps=[](const void* a,std::size_t an,const void* b,std::size_t bn){
            const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);return x<y+bn&&y<x+an;
        };
        if(overlaps(address_,std::size_t(width)*sizeof(float),producer_.codes(),width)||
           overlaps(address_,std::size_t(width)*sizeof(float),producer_.scales(),std::size_t(width/32)*sizeof(float))||
           overlaps(producer_.codes(),width,producer_.scales(),std::size_t(width/32)*sizeof(float)))
            throw std::invalid_argument("M1 hidden Q8 source/code/scale roles overlap");
        source_lease_=TransferEngine{}.retainCapturedDeviceStorage(source,device,bytes_);
        validate(source,stream,&workspace);
    }
    /** @brief Reject source owner/address/shape, stream, runtime or original workspace drift before replay. */
    void validate(const TensorBase& source,void* stream,DeviceWorkspaceManager* producerWorkspace) const {
        if(&source!=source_||source.transferStorageOwner()!=storage_owner_||source.native_type()!=TensorType::FP32||
           source.gpu_data_ptr()!=address_||source.size_bytes()!=bytes_||source.shape()!=shape_||stream!=stream_||
           getBackendFor(device_)!=backend_||!backend_||backend_->deviceRuntimeGeneration(device_.ordinal)!=runtime_)
            throw std::logic_error("M1 hidden Q8 source/stream/runtime identity changed");
        TransferEngine{}.requireCapturedDeviceStorage(*source_lease_,source,device_,bytes_);producer_.validate(producerWorkspace);
    }
    /** @brief Check the original seal without interpreting or reading any mutable tensor bytes. */
    void validate() const{validate(*source_,stream_,producer_.owner());}
    /** @return Original row source address for payload provenance checks. */
    const float* source() const{return static_cast<const float*>(address_);}
    /** @return Existing code bank, mutable only at its sole original producer boundary. */
    int8_t* codes() const{return producer_.codes();}
    /** @return Existing scale bank, mutable only at its sole original producer boundary. */
    float* scales() const{return producer_.scales();}
    /** @return Frozen original block32 row width. */
    int width() const{return width_;}
    /** @return Exact single participant. */
    DeviceId device() const{return device_;}
    /** @return Original explicit producer/consumer stream. */
    void* stream() const{return stream_;}
    /** @return Original retained workspace owner. */
    DeviceWorkspaceManager* workspace() const{return producer_.owner();}
};

/** @brief Consumer-only checked read view, keeping independent shared-expert workspace ownership. */
class OriginalM1HiddenQ8ReadView final {
    std::shared_ptr<const OriginalM1HiddenQ8Binding> producer_;
    OriginalM1HiddenQ8WorkspaceSeal consumer_;
public:
    /** @brief Retain the prepared original producer and independent consumer banks outside capture. */
    OriginalM1HiddenQ8ReadView(std::shared_ptr<const OriginalM1HiddenQ8Binding> producer,
        const TensorBase& source,void* stream,DeviceWorkspaceManager& consumer)
        :producer_(std::move(producer)),consumer_(consumer,OriginalM1HiddenQ8Bank::ConsumerScratch) {
        if(!producer_)throw std::invalid_argument("M1 hidden Q8 requires its original producer binding");
        validate(source,stream,&consumer);
        // Code/scales are read-only. Every other named consumer bank can be
        // written by its original pipeline and therefore must be disjoint.
        const auto overlaps=[](const void* a,std::size_t an,const void* b,std::size_t bn){
            const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
            return x<y+bn&&y<x+an;
        };
        for(const auto& name:consumer.bufferNames()) {
            const auto* pointer=consumer.getBuffer(name);const auto bytes=consumer.getBufferSize(name);
            const bool codeName=name==MoEWorkspaceBuffers::DECODE_HIDDEN_INT8||name==MoEWorkspaceBuffers::ROCM_ROUTER_Q8_HIDDEN;
            const bool scaleName=name==MoEWorkspaceBuffers::DECODE_HIDDEN_SCALES||name==MoEWorkspaceBuffers::ROCM_ROUTER_Q8_SCALES;
            if((codeName&&pointer==producer_->codes()&&bytes>=std::size_t(producer_->width()))||
               (scaleName&&pointer==producer_->scales()&&bytes>=std::size_t(producer_->width()/32)*sizeof(float)))continue;
            if(overlaps(pointer,bytes,producer_->source(),std::size_t(producer_->width())*sizeof(float))||
               overlaps(pointer,bytes,producer_->codes(),producer_->width())||
               overlaps(pointer,bytes,producer_->scales(),std::size_t(producer_->width()/32)*sizeof(float)))
                throw std::invalid_argument("M1 hidden Q8 immutable bank aliases consumer writable scratch");
        }
    }
    /** @brief Validate cold read identity on capture or replay; no backend query, allocation or registry lookup. */
    void validate(const TensorBase& source,void* stream,DeviceWorkspaceManager* consumer) const {
        producer_->validate(source,stream,producer_->workspace());consumer_.validate(consumer);
        if(consumer_.device()!=producer_->device())throw std::logic_error("M1 hidden Q8 consumer changed participant");
    }
    /** @brief Reject absent, foreign or merely recorded payload when eager execution would consume it. */
    void requirePayload(const float* source,const int8_t* codes,const float* scales,int rows,
        OriginalM1HiddenQ8Use published,OriginalM1HiddenQ8Use consumer) const {
        if(source!=producer_->source()||codes!=producer_->codes()||scales!=producer_->scales()||rows!=1||
           (published==OriginalM1HiddenQ8Use::CaptureRecording&&consumer!=OriginalM1HiddenQ8Use::CaptureRecording))
            throw std::logic_error("M1 hidden Q8 required original payload/provenance is absent");
    }
    /** @return Read-only original codes; caller cannot request another publication or quantization. */
    const int8_t* codes() const{return producer_->codes();}
    /** @return Read-only original FP32 scales. */
    const float* scales() const{return producer_->scales();}
};
}
