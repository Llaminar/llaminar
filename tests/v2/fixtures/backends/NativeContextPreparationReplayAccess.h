/**
 * @file NativeContextPreparationReplayAccess.h
 * @brief One shared recorded-fixture seam for actual source/pool/PMA replay.
 *
 * Synthetic certificates are confined to this device-free fixture. No public
 * permit constructor, production catalog row or native admission bypass exists.
 */
#pragma once
#include "backends/NativeContextPreparation.h"
#include "backends/GPUDeviceContextPool.h"
#include <array>
#include <algorithm>
#include <mutex>
#include <unordered_map>
namespace llaminar2::test {
/** @brief Device-free recorded original-source access; installed family allowances and native diagnostics stay separate. */
struct NativeContextPreparationReplayAccess {
    /** @brief Invoke the actual released-source retirement body with a device-free native backend authority. */
    static void retireReleased(const std::shared_ptr<NativeContextPreparation>& source,const TransferEngine& transfers){source->retireReleasedContexts(transfers);}

    /** @return Actual source-owned explicit stack/object program, without creating a native worker. */
    static GPUWorkerStackProgram objectProgram(std::size_t bytes,std::size_t alignment){return GPUWorkerStackProgram::originalDefaults().withContextObject(bytes,alignment);}

    template<class Worker>
    static std::shared_ptr<NativeContextPreparation> source(DeviceId device,std::size_t gpuCapacity=64u<<20,NativePhysicalLifetime lifetime=NativePhysicalLifetime::ContextGeneration){
        const std::array observed{PhysicalMemoryResource{0,DeviceId::cpu(),128u<<20,128u<<20},
                                 PhysicalMemoryResource{0,device,gpuCapacity,gpuCapacity}};
        return source<Worker>(device,observed,lifetime);
    }
    /** @brief Use original recorded allocator facts, without replacing them after resource birth. */
    template<class Worker>
    static std::shared_ptr<NativeContextPreparation> source(DeviceId device,std::span<const PhysicalMemoryResource> observed,NativePhysicalLifetime lifetime=NativePhysicalLifetime::ContextGeneration){
        const std::array devices{device};return source<Worker>(std::span<const DeviceId>{devices},observed,lifetime);
    }
    /** @brief Declare multiple recorded original workers in one canonical source without creating foreign authorities. */
    template<class Worker>
    static std::shared_ptr<NativeContextPreparation> source(std::span<const DeviceId> devices,std::span<const PhysicalMemoryResource> observed,NativePhysicalLifetime lifetime=NativePhysicalLifetime::ContextGeneration){
        if(devices.empty())throw std::invalid_argument("Recorded source population is empty");
        for(std::size_t i=0;i<devices.size();++i)if(!devices[i].is_gpu() || std::find(devices.begin(),devices.begin()+i,devices[i])!=devices.begin()+i)
            throw std::invalid_argument("Recorded source population has repeated or invalid workers");
        auto result=std::shared_ptr<NativeContextPreparation>(new NativeContextPreparation(observed,0));
        for(const auto device:devices){
        auto program=GPUWorkerStackProgram::originalDefaults();
        NativeContextBirthKey key;key.device=device;key.physical_uuid="recorded-device";key.architecture="recorded-architecture";
        key.runtime_version="recorded-runtime";key.driver_version="recorded-driver";
        key.source_sha256=key.backend_sha256=key.runtime_sha256=key.driver_sha256=std::string(64,'a');
        key.visibility_identity="recorded-selected-population";key.pthread_defaults_identity=program.identity();
        key.visible_device_count=1;key.compute_units=80;key.native_wave_width=device.is_cuda()?32:64;key.runtime_policy_sha256=std::string(64,'a');key.hsa_sha256=device.is_rocm()?std::string(64,'b'):std::string{};key.worker_stack_bytes=program.stackBytes();key.worker_guard_bytes=program.guardBytes();
        key.context_object_bytes=sizeof(Worker);key.context_object_alignment=alignof(Worker);
        key.primary_streams=key.primary_blas_handles=key.secondary_blas_handles=1;key.dependency_events=32;
        NativeContextOpaqueContract contract(key,4096,8192,std::string(64,'b'),lifetime);
        result->births_.push_back(std::shared_ptr<PreparedGPUContextBirth>(new PreparedGPUContextBirth(result->source_,contract,program)));
        }
        return result;
    }
    /** @brief Observe the original birth-owned key/control block as an independent replay oracle. */
    static std::shared_ptr<const NativeContextBirthKey> birthKey(const std::shared_ptr<NativeContextPreparation>& source,DeviceId device){
        for(const auto& birth:source->births_)if(birth->device()==device)return {birth,&birth->contract_.key()};
        throw std::logic_error("Recorded source omitted original birth key");
    }
    /** @brief Replay the actual first ordinary lookup phase before an admitted factory gap. */
    static void ordinaryBirthPrecheck(DeviceId device){GPUDeviceContextPool::instance().requireOrdinaryContextBirthEligible(device);}
    /** @brief Replay the actual second locked creator phase after the recorded factory gap. */
    static std::shared_ptr<IWorkerGPUContext> ordinaryBirthAfterFactoryAdmission(DeviceId device){
        auto& pool=GPUDeviceContextPool::instance();return device.is_cuda()?pool.retainNvidiaContextAfterFactoryAdmission(device.gpu_ordinal()):pool.retainAMDContextAfterFactoryAdmission(device.gpu_ordinal());
    }
    /** @brief Replay a mismatched canonical worker control block against the original capability. */
    static std::shared_ptr<const NativeContextBirthKey> birthKeyWithSubstitutedWorker(
        const GPUContextProvenance& original,std::shared_ptr<IWorkerGPUContext> replacement){
        return withRecordedPoolWorker(original.device(),std::move(replacement),[&]{return original.originalBirthKey();});
    }
    /** @brief A foreign published ordinary object cannot acquire the original source's key. */
    static std::shared_ptr<const NativeContextBirthKey> birthKeyFromForeignPublishedWorker(
        DeviceId device,std::shared_ptr<IWorkerGPUContext> replacement){
        return withRecordedPoolWorker(device,std::move(replacement),[&]{return GPUDeviceContextPool::instance().retainExistingContextProvenance(device).originalBirthKey();});
    }
    /** @brief Replay actual complete original worker creation then planning closure for a recorded multiowner source. */
    template<class Worker>
    static std::shared_ptr<NativeContextPreparation> preparedSource(std::span<const DeviceId> devices,std::span<const PhysicalMemoryResource> observed){
        auto result=source<Worker>(devices,observed);
        for(const auto device:devices){auto permit=result->acquire(device);}
        result->freezeForPlanning();return result;
    }
    /** @brief Place the recorded original typed worker in the ACTUAL preclaimed object extent. */
    template<class Worker>
    static std::shared_ptr<IWorkerGPUContext> construct(std::shared_ptr<PreparedGPUWorkerStack> root,int ordinal){
        auto* object=new(root->contextObjectStorage(sizeof(Worker),alignof(Worker))) Worker(ordinal,root);
        return std::shared_ptr<IWorkerGPUContext>(object,[root=std::move(root)](IWorkerGPUContext* base){std::destroy_at(static_cast<Worker*>(base));});
    }
    /** @brief Publish recorded extra families in the SAME original state before final admission. */
    static std::shared_ptr<NativeStoragePmaBinding> admit(const std::shared_ptr<NativeContextPreparation>& source,const PhysicalMemoryPlan& extra={}){
        auto snapshot=source->closeForPlanning();PhysicalMemoryPlanBuilder plan(snapshot.retainedPreparationBOM());
        for(const auto& bom:extra.resources())plan.add(bom);
        return source->admit(std::move(snapshot),std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()));
    }
    /** @brief Seed only the recorded constructor body; source birth/pool ownership remains actual production code. */
    static void seed(DeviceId device,std::function<std::shared_ptr<IWorkerGPUContext>(std::shared_ptr<PreparedGPUWorkerStack>)> factory){
        if(!device.is_gpu() || !factory)throw std::invalid_argument("Recorded original worker factory needs one exact device");
        auto& once=device.is_cuda()?cudaRegistration_:hipRegistration_;
        std::call_once(once,[type=device.type]{
        GPUDeviceContextPool::instance().registerPreparedFactory(type,[type](int ordinal,std::shared_ptr<PreparedGPUWorkerStack> stack){
            std::function<std::shared_ptr<IWorkerGPUContext>(std::shared_ptr<PreparedGPUWorkerStack>)> body;
            {std::lock_guard lock(seedMutex_);const auto key=DeviceId(type,ordinal).toString();auto found=seeds_.find(key);
             if(found==seeds_.end())throw std::logic_error("Recorded worker constructor body omitted");
             body=std::move(found->second);seeds_.erase(found);}
            return body(std::move(stack));
        });
        });
        std::lock_guard lock(seedMutex_);
        if(!seeds_.emplace(device.toString(),std::move(factory)).second)throw std::logic_error("Recorded worker constructor repeated");
    }
    /** @brief Replay a foreign attempt to replace the already-registered original constructor. */
    static void repeatRegistration(DeviceType type){
        GPUDeviceContextPool::instance().registerPreparedFactory(type,[](int,std::shared_ptr<PreparedGPUWorkerStack>){return std::shared_ptr<IWorkerGPUContext>{};});
    }
    /** @brief Replay actual non-loading ELF closure authentication without a GPU runtime. */
    static std::string loadedClosure(const void* symbol){return PreparedGPUContextFactory::libraryClosureIdentity(symbol);}
    /** @brief Reject source cancellation while actual attempted process backing remains. */
    static void cancel(const std::shared_ptr<NativeContextPreparation>& source){source->source_->cancel();}
    /** @brief Validate exact preclaimed token against an alternate authority through actual production checks. */
    static void requireProcessRetention(const std::shared_ptr<NativeContextPreparation>& source,const std::shared_ptr<PhysicalMemoryAuthority>& memory){
        const auto& birth=source->births_.front();birth->opaque_->requireOriginalProcessRetention(memory,birth->contract_);
    }
    static void physicalReset(ExclusiveGPUDeviceGenerationRetirement& scope){scope.retireNativeObligationsAfterValidatedReset();}
private:
    /** @brief Replay a recorded publication mismatch, restoring original pool ownership on every exit. */
    template<class Body> static std::shared_ptr<const NativeContextBirthKey> withRecordedPoolWorker(DeviceId device,std::shared_ptr<IWorkerGPUContext> replacement,Body body){
        auto& pool=GPUDeviceContextPool::instance();std::shared_ptr<IWorkerGPUContext> actual;
        {std::lock_guard lock(pool.mutex_);auto& contexts=device.is_cuda()?pool.nvidia_contexts_:pool.amd_contexts_;
         actual=std::exchange(contexts.at(device.gpu_ordinal()),std::move(replacement));}
        const auto restore=[&]{std::lock_guard lock(pool.mutex_);auto& contexts=device.is_cuda()?pool.nvidia_contexts_:pool.amd_contexts_;contexts.at(device.gpu_ordinal())=std::move(actual);};
        try{auto result=body();restore();return result;}catch(...){restore();throw;}
    }
    inline static std::once_flag cudaRegistration_,hipRegistration_;
    inline static std::mutex seedMutex_;
    inline static std::unordered_map<std::string,std::function<std::shared_ptr<IWorkerGPUContext>(std::shared_ptr<PreparedGPUWorkerStack>)>> seeds_;
};
}
