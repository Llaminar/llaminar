/**
 * @file GPUDeviceContextPool.cpp
 * @brief Implementation of GPUDeviceContextPool singleton
 *
 * This file does NOT include any CUDA/HIP headers directly. Instead, it uses
 * factory functions registered by cuda_backend and rocm_backend libraries.
 * This keeps GPUDeviceContextPool in llaminar2_core without GPU dependencies.
 * Typed and named lookups share the same registered-factory authority. Device-
 * free tests can register explicit contexts; absent factories still fail hard.
 * Retained context owners close lookup-to-submission races without holding the
 * pool mutex while another thread owns the native capture interval. Original
 * prepared-key aliases authenticate that same generation and retain metadata
 * without repeating provider ELF hashing or extending native worker lifetime.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "GPUDeviceContextPool.h"
#include "NativeContextPreparation.h"
#include "RetainedParticipantGraphExecutable.h"
#include "../utils/Logger.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        /** @brief Validate retirement identity before any lookup or lifecycle mutation.
         * @param device Authored native endpoint, never a CPU or negative ordinal.
         * @return The validated nonnegative native ordinal.
         * @throws std::invalid_argument if the endpoint cannot name a GPU.
         */
        int retirementDeviceOrdinal(DeviceId device)
        {
            if (!device.is_gpu())
                throw std::invalid_argument(
                    "[GPUDeviceContextPool] Exclusive generation retirement requires an exact nonnegative GPU device");
            return device.gpu_ordinal();
        }
    }

    GPUContextProvenance GPUDeviceContextPool::retainExistingContextProvenance(DeviceId device)
    {
        if(!device.is_gpu())throw std::invalid_argument("Existing pool provenance requires an exact native device");
        std::lock_guard lock(mutex_);const auto ordinal=device.gpu_ordinal();
        const auto& contexts=device.is_cuda()?nvidia_contexts_:amd_contexts_;
        const auto& generations=device.is_cuda()?nvidia_generations_:amd_generations_;
        const auto& retiring=device.is_cuda()?retiring_nvidia_contexts_:retiring_amd_contexts_;
        const auto worker=contexts.find(ordinal);const auto generation=generations.find(ordinal);
        if(retiring.contains(ordinal) || worker==contexts.end() || !worker->second || generation==generations.end() || !generation->second)
            throw std::logic_error("Exact original pool worker/backend/device/generation is not live");
        return {device,generation->second,worker->second};
    }
    std::shared_ptr<const NativeContextBirthKey> GPUContextProvenance::originalBirthKey()const
    {
        requireWorker(worker_);
        return GPUDeviceContextPool::instance().originalBirthKey(*this);
    }
    std::shared_ptr<const NativeContextBirthKey> GPUDeviceContextPool::originalBirthKey(const GPUContextProvenance& original)
    {
        original.requireWorker(original.worker());
        const auto device=original.device();const auto ordinal=device.gpu_ordinal();
        std::lock_guard lock(mutex_);
        const auto& contexts=device.is_cuda()?nvidia_contexts_:amd_contexts_;
        const auto& generations=device.is_cuda()?nvidia_generations_:amd_generations_;
        const auto& retiring=device.is_cuda()?retiring_nvidia_contexts_:retiring_amd_contexts_;
        const auto worker=contexts.find(ordinal);const auto generation=generations.find(ordinal);
        const auto birth=prepared_births_.find(device.toString());
        if(retiring.contains(ordinal) || worker==contexts.end() || generation==generations.end() ||
           !generation->second || generation->second!=original.generation() ||
           birth==prepared_births_.end() || !birth->second)
            throw std::logic_error("Original birth key requires a live exact prepared worker generation; ordinary or stale context unsupported");
        original.requireWorker(worker->second);
        if(!birth->second->stack_)
            throw std::logic_error("Original birth key has no live original source-owned worker object");
        birth->second->stack_->requireContextObjectPointer(worker->second.get());
        const auto& key=birth->second->contract_.key();
        if(key.device!=device)
            throw std::logic_error("Original birth key changed its exact prepared physical endpoint");
        // Aliasing retains only the existing birth/source metadata. No worker,
        // copied key, provider identity cache or resource-use permit is created.
        return {birth->second,&key};
    }
    bool GPUDeviceContextPool::ownsPreparedBirth(const std::shared_ptr<PreparedGPUContextBirth>& birth)const{
        if(!birth)return false;std::lock_guard lock(mutex_);
        const auto found=prepared_births_.find(birth->device().toString());
        return found!=prepared_births_.end() && found->second==birth;
    }
    PreparedGPUContextUsePermit GPUDeviceContextPool::retainExistingPreparedContext(DeviceId device,
        const std::shared_ptr<PhysicalMemoryAuthority>& memory)
    {
        if(!device.is_gpu() || !memory)throw std::invalid_argument("Exact existing source borrow requires native device and original PMA");
        std::lock_guard lock(mutex_);
        const auto ordinal=device.gpu_ordinal();
        const auto& contexts=device.is_cuda()?nvidia_contexts_:amd_contexts_;
        const auto& generations=device.is_cuda()?nvidia_generations_:amd_generations_;
        const auto& retiring=device.is_cuda()?retiring_nvidia_contexts_:retiring_amd_contexts_;
        const auto birth=prepared_births_.find(device.toString());
        const auto worker=contexts.find(ordinal);const auto generation=generations.find(ordinal);
        if(retiring.contains(ordinal) || birth==prepared_births_.end() || worker==contexts.end() || !worker->second ||
           generation==generations.end() || !generation->second)
            throw std::logic_error("No live original prepared source worker for exact backend/device/generation");
        const auto& original=birth->second->memory();
        if(original.get()!=memory.get() || original.owner_before(memory) || memory.owner_before(original))
            throw std::logic_error("Existing source worker borrow has a foreign PMA control block");
        return {birth->second,GPUContextProvenance(device,generation->second,worker->second)};
    }
    bool GPUDeviceContextPool::hasPreparedNativeProgramStack(DeviceId device,
        const std::shared_ptr<PhysicalMemoryAuthority>& memory) const
    {
        std::lock_guard lock(mutex_);
        const auto found=prepared_births_.find(device.toString());
        if(found==prepared_births_.end())return false;
        const auto& original=found->second->memory();
        if(!memory || original.get()!=memory.get() || original.owner_before(memory) || memory.owner_before(original))
            throw std::logic_error("Native program stack has a foreign original source PMA control block");
        if(!device.is_cuda())return false;
        if(retiring_nvidia_contexts_.contains(device.gpu_ordinal()) ||
           !nvidia_contexts_.contains(device.gpu_ordinal()) || !nvidia_contexts_.at(device.gpu_ordinal()))
            throw std::logic_error("Native program stack has no live original source CUDA worker");
        return true;
    }
    void GPUDeviceContextPool::requireOrdinaryContextStackGrowth(DeviceId device,
        const std::shared_ptr<PhysicalMemoryAuthority>& memory) const
    {
        std::lock_guard lock(mutex_);
        const auto found=prepared_births_.find(device.toString());
        if(found==prepared_births_.end())return;
        if(found->second->memory()!=memory)throw std::logic_error("Native stack growth has a foreign original source PMA");
        throw std::runtime_error("Source-prepared CUDA population requires a separately qualified original-PMA native-program stack-growth contract; late measured context ledger forbidden");
    }
    void GPUDeviceContextPool::registerPreparedFactory(DeviceType type,
        std::function<std::shared_ptr<IWorkerGPUContext>(int,std::shared_ptr<PreparedGPUWorkerStack>)> creator)
    {
        if(!creator || (type!=DeviceType::CUDA && type!=DeviceType::ROCm))
            throw std::invalid_argument("Prepared source creator needs one exact native backend");
        std::lock_guard lock(mutex_);
        auto& installed=type==DeviceType::CUDA?prepared_cuda_:prepared_rocm_;
        if(installed)throw std::logic_error("Original prepared source creator registration cannot be replaced or repeated");
        installed=std::move(creator);
    }
    PreparedGPUContextUsePermit GPUDeviceContextPool::retainPreparedContext(std::shared_ptr<PreparedGPUContextBirth> birth)
    {
        if(!birth)throw std::invalid_argument("Missing original prepared source birth");
        bool needsCreator;
        {std::lock_guard lock(mutex_);needsCreator=!(birth->device().is_cuda()?prepared_cuda_:prepared_rocm_);}
#ifdef HAVE_CUDA
        if(needsCreator && birth->device().is_cuda())ensureNvidiaPreparedFactoryRegistered();
#endif
#ifdef HAVE_ROCM
        if(needsCreator && birth->device().is_rocm())ensureAMDPreparedFactoryRegistered();
#endif
        std::lock_guard lock(mutex_);
        const auto device=birth->device();const auto ordinal=device.gpu_ordinal();
        auto& contexts=device.is_cuda()?nvidia_contexts_:amd_contexts_;
        auto& generations=device.is_cuda()?nvidia_generations_:amd_generations_;
        const auto& retiring=device.is_cuda()?retiring_nvidia_contexts_:retiring_amd_contexts_;
        if(retiring.contains(ordinal))throw std::logic_error("Prepared source birth raced original generation retirement");
        const auto prior=prepared_births_.find(device.toString());
        if(prior!=prepared_births_.end()){
            if(prior->second!=birth)throw std::logic_error("Source cannot borrow a foreign original context authority");
            const auto context=contexts.find(ordinal);
            if(context==contexts.end())throw std::logic_error("Failed original source birth requires physical runtime retirement");
            return {birth,GPUContextProvenance(device,generations.at(ordinal),context->second)};
        }
        if(contexts.contains(ordinal))throw std::logic_error("Unowned existing context cannot be adopted from a late memory observation");
        auto& creator=device.is_cuda()?prepared_cuda_:prepared_rocm_;
        if(!creator)throw std::runtime_error("Original prepared source backend creator is unavailable");
        if(next_generation_==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("Original context generation exhausted");
        // Allocate every pool metadata slot before native creation; failures here remain prebirth.
        const auto generation=next_generation_++;
        prepared_births_.emplace(device.toString(),birth);
        try { generations.emplace(ordinal,generation);contexts.emplace(ordinal,nullptr); }
        catch(...){generations.erase(ordinal);contexts.erase(ordinal);prepared_births_.erase(device.toString());throw;}
        try {birth->begin();}
        catch(...){contexts.erase(ordinal);generations.erase(ordinal);prepared_births_.erase(device.toString());throw;}
        try {
            auto worker=creator(ordinal,birth->stack_);
            if(!worker || !worker->isInitialized() || worker->deviceOrdinal()!=ordinal || !worker->defaultStream() ||
               reinterpret_cast<std::uintptr_t>(worker->defaultStream())<=2)
                throw std::runtime_error("Original prepared source creator returned a foreign/incomplete worker or primary stream");
            birth->stack_->requireContextObjectPointer(worker.get());
            contexts.at(ordinal)=std::move(worker);
            auto& owner=contexts.at(ordinal);owner->bindProjectionContextOwner(owner);
            birth->created();
            return {birth,GPUContextProvenance(device,generation,owner)};
        } catch(...) {
            // Original constructor rollback retires handles, never authorizes opaque pool release.
            contexts.erase(ordinal);throw;
        }
    }
    std::shared_ptr<RetainedParticipantGraphExecutable> GPUDeviceContextPool::borrowPreparedCalibration(
        const PreparedGPUContextUsePermit& original,void* stream){
        return retainPreparedCalibrationSlot(original,stream)->acquire(original,stream);
    }
    std::shared_ptr<PreparedServiceCalibrationSlot> GPUDeviceContextPool::retainPreparedCalibrationSlot(
        const PreparedGPUContextUsePermit& original,void* stream){
        if(!original.device().is_gpu() || !original.worker() || original.worker()->ownsCurrentThread() ||
            !stream || stream!=original.worker()->defaultStream())
            throw std::invalid_argument("Original calibration requires its off-worker source stream");
        std::shared_ptr<PreparedServiceCalibrationSlot> slot;
        {
            std::lock_guard lock(mutex_);
            const auto device=original.device();const auto ordinal=device.gpu_ordinal();
            const auto& contexts=device.is_cuda()?nvidia_contexts_:amd_contexts_;
            const auto& generations=device.is_cuda()?nvidia_generations_:amd_generations_;
            const auto& retiring=device.is_cuda()?retiring_nvidia_contexts_:retiring_amd_contexts_;
            const auto found=prepared_births_.find(device.toString());
            const auto worker=contexts.find(ordinal);const auto generation=generations.find(ordinal);
            if(found==prepared_births_.end() || found->second!=original.birth_ ||
                found->second.owner_before(original.birth_) || original.birth_.owner_before(found->second) || worker==contexts.end() ||
                generation==generations.end() || generation->second!=original.generation() || retiring.contains(ordinal) ||
                found->second->calibration_closed_)
                throw std::logic_error("Original calibration source is stale, foreign or retiring");
            original.requireWorker(worker->second);
            auto& holder=found->second->calibration_;
            // Only exact CPU metadata is materialized under the pool mutex.
            // The existing graph factory/worker callbacks run strictly off-lock.
            if(!holder)holder=PreparedServiceCalibrationSlot::create(original.memory(),device);
            slot=holder;
        }
        return slot;
    }
    void GPUDeviceContextPool::retirePreparedCalibration(const std::shared_ptr<PreparedGPUContextBirth>& birth){
        if(!birth)throw std::invalid_argument("Calibration retirement requires its original selected birth");
        std::shared_ptr<PreparedServiceCalibrationSlot> slot;
        {
            std::lock_guard lock(mutex_);
            const auto found=prepared_births_.find(birth->device().toString());
            if(found==prepared_births_.end())return;
            if(found->second!=birth)throw std::logic_error("Calibration retirement substituted its original source");
            birth->calibration_closed_=true;slot=birth->calibration_;
            if(slot){
                slot->closeAcquisition();
                // Birth plus this retirement-local reference are the complete
                // source-owned population. A pending off-lock lookup must return
                // before native handles or its known CPU root can retire.
                if(slot.use_count()!=2)
                    throw std::logic_error("Original prepared calibration retirement has an in-flight holder borrower");
            }
        }
        if(slot)slot->retire(); // Never hold the pool lock across original worker teardown.
        {
            std::lock_guard lock(mutex_);
            if(birth->calibration_!=slot)throw std::logic_error("Original calibration owner changed during final retirement");
            birth->calibration_.reset();
        }
        slot.reset(); // Known CPU holder retires before the existing zero-allocation tickets.
    }
    void GPUDeviceContextPool::confirmPreparedPhysicalRetirement(DeviceId device)
    {
        std::lock_guard lock(mutex_);
        const auto& retiring=device.is_cuda()?retiring_nvidia_contexts_:retiring_amd_contexts_;
        if(!retiring.contains(device.gpu_ordinal()))throw std::logic_error("Physical reset proof lacks original generation exclusion");
        const auto found=prepared_births_.find(device.toString());
        if(found!=prepared_births_.end()){
            found->second->physicallyRetired();prepared_births_.erase(found);
            (device.is_cuda()?nvidia_generations_:amd_generations_).erase(device.gpu_ordinal());
        }
    }
    NativeContextRetirementSlot GPUContextProvenance::reserveNativeRetirementSlot(
        const std::shared_ptr<PhysicalMemoryAuthority>& memory,const std::string& certificate)const
    {
        requireWorker(worker_);
        return GPUDeviceContextPool::instance().reserveNativeRetirementSlot(*this,memory,certificate);
    }
    NativeContextRetirementSlot GPUDeviceContextPool::reserveNativeRetirementSlot(const GPUContextProvenance& original,
        const std::shared_ptr<PhysicalMemoryAuthority>& memory,const std::string& certificate)
    {
        original.requireWorker(original.worker());
        const auto device=original.device();const auto ordinal=device.gpu_ordinal();
        std::lock_guard lock(mutex_);
        const auto& contexts=device.is_cuda()?nvidia_contexts_:amd_contexts_;
        const auto& generations=device.is_cuda()?nvidia_generations_:amd_generations_;
        const auto& retiring=device.is_cuda()?retiring_nvidia_contexts_:retiring_amd_contexts_;
        auto& owners=device.is_cuda()?nvidia_native_retirements_:amd_native_retirements_;
        const auto worker=contexts.find(ordinal);const auto generation=generations.find(ordinal);
        if(retiring.contains(ordinal) || worker==contexts.end() || generation==generations.end() || generation->second!=original.generation())
            throw std::logic_error("Native retirement slot lost original canonical generation");
        original.requireWorker(worker->second);
        auto owner=owners.find(ordinal);
        if(owner==owners.end())owner=owners.emplace(ordinal,std::shared_ptr<NativeContextRetirementOwner>(
            new NativeContextRetirementOwner(device,generation->second))).first;
        try{return owner->second->reserve(memory,certificate);}
        catch(...){if(owner->second->empty())owners.erase(owner);throw;}
    }
    void GPUDeviceContextPool::discardEmptyNativeRetirementOwner(DeviceId device,std::uint64_t generation,
        const std::shared_ptr<NativeContextRetirementOwner>& owner)noexcept
    {
        std::lock_guard lock(mutex_);auto& owners=device.is_cuda()?nvidia_native_retirements_:amd_native_retirements_;
        const auto found=owners.find(device.gpu_ordinal());
        if(found!=owners.end() && found->second==owner && owner->generation_==generation && owner->empty())owners.erase(found);
    }

    // =============================================================================
    // Singleton Instance
    // =============================================================================

    GPUDeviceContextPool &GPUDeviceContextPool::instance()
    {
        // Meyers singleton - thread-safe in C++11 and later
        static GPUDeviceContextPool pool;
        return pool;
    }

    // =============================================================================
    // Destructor
    // =============================================================================

    GPUDeviceContextPool::~GPUDeviceContextPool()
    {
        shutdown();
    }

    // =============================================================================
    // Factory Registration
    // =============================================================================

    void GPUDeviceContextPool::registerNvidiaFactory(GPUContextFactory factory, int device_count)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        nvidia_factory_ = std::move(factory);
        cuda_device_count_ = device_count;

        LOG_DEBUG("[GPUDeviceContextPool] Registered NVIDIA factory with "
                  << device_count << " devices available");
    }

    void GPUDeviceContextPool::registerAMDFactory(GPUContextFactory factory, int device_count)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        amd_factory_ = std::move(factory);
        rocm_device_count_ = device_count;

        LOG_DEBUG("[GPUDeviceContextPool] Registered AMD factory with "
                  << device_count << " devices available");
    }

    // =============================================================================
    // Context Access
    // =============================================================================

    void GPUDeviceContextPool::requireOrdinaryContextBirthEligible(DeviceId device)
    {
        if(!device.is_gpu())throw std::invalid_argument("Ordinary worker precheck requires an exact native device");
        std::lock_guard lock(mutex_);
        const auto& contexts=device.is_cuda()?nvidia_contexts_:amd_contexts_;
        if(prepared_births_.contains(device.toString()) && !contexts.contains(device.gpu_ordinal()))
            throw std::logic_error("Original source birth is armed or failed; no ordinary creator may bypass it");
    }

    IWorkerGPUContext &GPUDeviceContextPool::getNvidiaContext(int device_ordinal)
    { return *retainNvidiaContext(device_ordinal); }

    std::shared_ptr<IWorkerGPUContext> GPUDeviceContextPool::retainNvidiaContext(int device_ordinal)
    {
        requireOrdinaryContextBirthEligible(DeviceId::cuda(device_ordinal));
#ifdef HAVE_CUDA
        // Auto-register factory if not yet registered (same pattern as getContext(DeviceId))
        if (!hasNvidiaSupport())
        {
            ensureNvidiaFactoryRegistered();
        }
#endif

        return retainNvidiaContextAfterFactoryAdmission(device_ordinal);
    }

    std::shared_ptr<IWorkerGPUContext> GPUDeviceContextPool::retainNvidiaContextAfterFactoryAdmission(int device_ordinal)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if(prepared_births_.contains(DeviceId::cuda(device_ordinal).toString()) &&
           !nvidia_contexts_.contains(device_ordinal))
            throw std::logic_error("Original source birth won ordinary factory admission; no ordinary creator may bypass it");


        if (!nvidia_factory_)
        {
            throw std::runtime_error(
                "[GPUDeviceContextPool] NVIDIA factory not registered. "
                "Ensure cuda_backend is linked and initialized.");
        }

        if (cuda_device_count_ == 0)
        {
            throw std::runtime_error("[GPUDeviceContextPool] No CUDA devices available");
        }

        if (device_ordinal < 0 || device_ordinal >= cuda_device_count_)
        {
            throw std::runtime_error("[GPUDeviceContextPool] Invalid CUDA device ordinal " + std::to_string(device_ordinal) + " (valid range: 0-" + std::to_string(cuda_device_count_ - 1) + ")");
        }

        if (retiring_nvidia_contexts_.contains(device_ordinal))
        {
            throw std::logic_error(
                "[GPUDeviceContextPool] CUDA context acquisition raced exclusive generation retirement for device " +
                std::to_string(device_ordinal));
        }

        // Check if context already exists
        auto it = nvidia_contexts_.find(device_ordinal);
        if (it != nvidia_contexts_.end())
        {
            return it->second;
        }

        // Create new context via factory (lazy initialization)
        LOG_DEBUG("[GPUDeviceContextPool] Creating NvidiaDeviceContext for device " << device_ordinal);

        auto context = nvidia_factory_(device_ordinal);
        nvidia_contexts_[device_ordinal] = std::move(context);
        if (next_generation_ == std::numeric_limits<std::uint64_t>::max())
        {
            nvidia_contexts_.erase(device_ordinal);
            throw std::overflow_error(
                "[GPUDeviceContextPool] Context generation identity exhausted");
        }
        nvidia_generations_[device_ordinal] = next_generation_++;

        auto owner = nvidia_contexts_.at(device_ordinal);
        owner->bindProjectionContextOwner(owner);
        return owner;
    }

    IWorkerGPUContext &GPUDeviceContextPool::getAMDContext(int device_ordinal)
    { return *retainAMDContext(device_ordinal); }

    std::shared_ptr<IWorkerGPUContext> GPUDeviceContextPool::retainAMDContext(int device_ordinal)
    {
        requireOrdinaryContextBirthEligible(DeviceId::rocm(device_ordinal));
#ifdef HAVE_ROCM
        // Auto-register factory if not yet registered (same pattern as getContext(DeviceId))
        if (!hasAMDSupport())
        {
            ensureAMDFactoryRegistered();
        }
#endif

        return retainAMDContextAfterFactoryAdmission(device_ordinal);
    }

    std::shared_ptr<IWorkerGPUContext> GPUDeviceContextPool::retainAMDContextAfterFactoryAdmission(int device_ordinal)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if(prepared_births_.contains(DeviceId::rocm(device_ordinal).toString()) &&
           !amd_contexts_.contains(device_ordinal))
            throw std::logic_error("Original source birth won ordinary factory admission; no ordinary creator may bypass it");


        if (!amd_factory_)
        {
            throw std::runtime_error(
                "[GPUDeviceContextPool] AMD factory not registered. "
                "Ensure rocm_backend is linked and initialized.");
        }

        if (rocm_device_count_ == 0)
        {
            throw std::runtime_error("[GPUDeviceContextPool] No ROCm devices available");
        }

        if (device_ordinal < 0 || device_ordinal >= rocm_device_count_)
        {
            throw std::runtime_error("[GPUDeviceContextPool] Invalid ROCm device ordinal " + std::to_string(device_ordinal) + " (valid range: 0-" + std::to_string(rocm_device_count_ - 1) + ")");
        }

        if (retiring_amd_contexts_.contains(device_ordinal))
        {
            throw std::logic_error(
                "[GPUDeviceContextPool] ROCm context acquisition raced exclusive generation retirement for device " +
                std::to_string(device_ordinal));
        }

        // Check if context already exists
        auto it = amd_contexts_.find(device_ordinal);
        if (it != amd_contexts_.end())
        {
            return it->second;
        }

        // Create new context via factory (lazy initialization)
        LOG_DEBUG("[GPUDeviceContextPool] Creating AMDDeviceContext for device " << device_ordinal);

        auto context = amd_factory_(device_ordinal);
        amd_contexts_[device_ordinal] = std::move(context);
        if (next_generation_ == std::numeric_limits<std::uint64_t>::max())
        {
            amd_contexts_.erase(device_ordinal);
            throw std::overflow_error(
                "[GPUDeviceContextPool] Context generation identity exhausted");
        }
        amd_generations_[device_ordinal] = next_generation_++;

        auto owner = amd_contexts_.at(device_ordinal);
        owner->bindProjectionContextOwner(owner);
        return owner;
    }

    std::shared_ptr<IWorkerGPUContext> GPUDeviceContextPool::retainContext(const DeviceId &device)
    {
        if (device.is_cuda()) return retainNvidiaContext(device.gpu_ordinal());
        if (device.is_rocm()) return retainAMDContext(device.gpu_ordinal());
        throw std::invalid_argument("Retained worker context requires an exact GPU device");
    }

    std::shared_ptr<IWorkerGPUContext> GPUDeviceContextPool::retainExistingContext(const DeviceId &device)
    {
        if (!device.is_gpu())
            throw std::invalid_argument("Existing worker context requires an exact GPU device");
        std::lock_guard<std::mutex> lock(mutex_);
        const auto &contexts = device.is_cuda() ? nvidia_contexts_ : amd_contexts_;
        const auto &retiring = device.is_cuda() ? retiring_nvidia_contexts_ : retiring_amd_contexts_;
        const auto it = contexts.find(device.ordinal);
        if (retiring.contains(device.ordinal) || it == contexts.end() || !it->second ||
            !it->second->isInitialized())
            throw std::logic_error("Original GPU worker is absent, uninitialized or retiring");
        return it->second;
    }

    IWorkerGPUContext &GPUDeviceContextPool::getContext(const std::string &device_type, int device_ordinal)
    {
        // Normalize device type string to lowercase for comparison
        std::string type_lower = device_type;
        std::transform(type_lower.begin(), type_lower.end(), type_lower.begin(),
                       [](unsigned char c)
                       { return std::tolower(c); });

        if (type_lower == "cuda" || type_lower == "nvidia")
        {
            return getNvidiaContext(device_ordinal);
        }
        else if (type_lower == "rocm" || type_lower == "hip" || type_lower == "amd")
        {
            return getAMDContext(device_ordinal);
        }
        else
        {
            throw std::invalid_argument("[GPUDeviceContextPool] Unknown device type: '" + device_type + "' (expected: cuda, nvidia, rocm, hip, amd)");
        }
    }

    IWorkerGPUContext &GPUDeviceContextPool::getContext(const DeviceId &device)
    {
        if (!device.is_gpu())
        {
            throw std::invalid_argument("[GPUDeviceContextPool] Device is not a GPU: '" + device.to_string() + "'");
        }

        if (device.is_cuda())
        {
            return getNvidiaContext(device.cuda_ordinal());
        }

        if (device.is_rocm())
        {
            return getAMDContext(device.rocm_ordinal());
        }

        throw std::invalid_argument("[GPUDeviceContextPool] Unsupported GPU device: '" + device.to_string() + "'");
    }

    // =============================================================================
    // Availability Queries
    // =============================================================================

    bool GPUDeviceContextPool::hasNvidiaSupport() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return nvidia_factory_ && cuda_device_count_ > 0;
    }

    bool GPUDeviceContextPool::hasAMDSupport() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return amd_factory_ && rocm_device_count_ > 0;
    }

    int GPUDeviceContextPool::nvidiaDeviceCount() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return cuda_device_count_;
    }

    int GPUDeviceContextPool::amdDeviceCount() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return rocm_device_count_;
    }

    // =============================================================================
    // Lifecycle Management
    // =============================================================================

    void GPUDeviceContextPool::shutdown()
    {
        decltype(nvidia_contexts_) retired_nvidia_contexts;
        decltype(amd_contexts_) retired_amd_contexts;
        size_t nvidia_count = 0u;
        size_t amd_count = 0u;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!retiring_nvidia_contexts_.empty() ||
                !retiring_amd_contexts_.empty())
            {
                throw std::logic_error(
                    "[GPUDeviceContextPool] Shutdown cannot overlap an exclusive runtime-generation retirement");
            }

            if(!prepared_births_.empty())throw std::logic_error("Source-prepared native contexts require original physical reset before shutdown");

            for(const auto& [ordinal,owner]:nvidia_native_retirements_)
                if(!owner->empty())throw std::logic_error("CUDA native resource obligations require original physical reset before shutdown");
            for(const auto& [ordinal,owner]:amd_native_retirements_)
                if(!owner->empty())throw std::logic_error("ROCm native resource obligations require original physical reset before shutdown");
            for (const auto &[ordinal, context] : nvidia_contexts_)
                if (context.use_count() != 1) throw std::logic_error("Cannot shut down a retained CUDA context");
            for (const auto &[ordinal, context] : amd_contexts_)
                if (context.use_count() != 1) throw std::logic_error("Cannot shut down a retained ROCm context");
            // Validate the complete set before closing any generation acquisition.
            for (auto &[ordinal, context] : nvidia_contexts_)
                context->assertProjectionResourcesRetirable();
            for (auto &[ordinal, context] : amd_contexts_)
                context->assertProjectionResourcesRetirable();
            for (auto &[ordinal, context] : nvidia_contexts_)
                context->beginProjectionResourcesRetirement();
            for (auto &[ordinal, context] : amd_contexts_)
                context->beginProjectionResourcesRetirement();
            nvidia_count = nvidia_contexts_.size();
            amd_count = amd_contexts_.size();
            retired_nvidia_contexts.swap(nvidia_contexts_);
            retired_amd_contexts.swap(amd_contexts_);
            nvidia_generations_.clear();
            amd_generations_.clear();
        }

        /* Worker teardown joins threads and enters CUDA/HIP to destroy exact
         * streams, events, and library handles. Keep those operations outside
         * the pool mutex so unrelated device acquisition cannot deadlock on a
         * backend destructor that re-enters infrastructure. */
        retired_nvidia_contexts.clear();
        retired_amd_contexts.clear();

        LOG_DEBUG("[GPUDeviceContextPool] Shutdown cleared " << nvidia_count
                                                             << " NVIDIA and " << amd_count
                                                             << " AMD contexts");
    }

    GPUDeviceContextGenerationRetirementReceipt
    GPUDeviceContextPool::retireExclusiveGeneration(DeviceId device)
    {
        const int ordinal = retirementDeviceOrdinal(device);
        {
            std::lock_guard lock(mutex_);
            if(prepared_births_.contains(device.toString()))
                throw std::logic_error("Prepared source generation requires the original complete physical runtime-reset transaction");
        }
        {
            std::lock_guard lock(mutex_);
            const auto& owners=device.is_cuda()?nvidia_native_retirements_:amd_native_retirements_;
            const auto owner=owners.find(ordinal);
            if(owner!=owners.end() && !owner->second->empty())
                throw std::logic_error("Worker-only retirement cannot release original native resource obligations");
        }
        auto scope = beginExclusiveGenerationRetirement(device);
        return scope.receipt();
    }

    ExclusiveGPUDeviceGenerationRetirement
    GPUDeviceContextPool::beginExclusiveGenerationRetirement(DeviceId device)
    {
        const int ordinal = retirementDeviceOrdinal(device);

        std::shared_ptr<IWorkerGPUContext> retired_context;
        std::shared_ptr<NativeContextRetirementOwner> native_owner;
        std::uint64_t retired_generation = 0u;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto &contexts = device.is_cuda() ? nvidia_contexts_ : amd_contexts_;
            auto &generations =
                device.is_cuda() ? nvidia_generations_ : amd_generations_;
            auto &retiring = device.is_cuda()
                                 ? retiring_nvidia_contexts_
                                 : retiring_amd_contexts_;

            if (retiring.contains(ordinal))
            {
                throw std::logic_error(
                    "[GPUDeviceContextPool] Device context generation is already retiring for " +
                    device.toString());
            }

            /* Install exclusion even when no worker was materialized. The
             * backend runtime generation still exists and must not race a lazy
             * context acquisition during reset. */
            retiring.insert(ordinal);

            const auto context_it = contexts.find(ordinal);
            if (context_it == contexts.end())
            {
                return ExclusiveGPUDeviceGenerationRetirement(
                    this,
                    GPUDeviceContextGenerationRetirementReceipt{
                        .device = device,
                        .retired_generation = generations.contains(ordinal)?generations.at(ordinal):0u,
                    });
            }

            const auto generation_it = generations.find(ordinal);
            if (generation_it == generations.end() ||
                generation_it->second == 0u)
            {
                retiring.erase(ordinal);
                throw std::logic_error(
                    "[GPUDeviceContextPool] Live context has no generation identity for " +
                    device.toString());
            }

            if (context_it->second.use_count() != 1)
            {
                retiring.erase(ordinal);
                throw std::logic_error("Cannot retire a GPU context with a retained capture or submission owner");
            }
            auto& native_owners=device.is_cuda()?nvidia_native_retirements_:amd_native_retirements_;
            const auto original_native_owner=native_owners.find(ordinal);
            if(original_native_owner!=native_owners.end())
            {
                try { original_native_owner->second->requireHandlesRetired(); }
                catch(...) {retiring.erase(ordinal);throw;}
                native_owner=original_native_owner->second;
            }
            // Check before removing authority: a live capture must leave the exact
            // context discoverable and its generation intact on rejection.
            try { context_it->second->beginProjectionResourcesRetirement(); }
            catch (...) { retiring.erase(ordinal); throw; }
            retired_generation = generation_it->second;
            if(original_native_owner!=native_owners.end())native_owners.erase(original_native_owner);
            retired_context = std::move(context_it->second);
            contexts.erase(context_it);
            generations.erase(generation_it);
        }

        /*
         * Context destruction joins the worker and destroys every exact stream,
         * event, and library handle. Do it outside mutex_: cleanup may enter a
         * backend runtime and must not serialize unrelated device acquisition.
         */
        try
        {
            retired_context.reset();
        }
        catch (...)
        {
            finishExclusiveGenerationRetirement(device);
            throw;
        }

        LOG_INFO(
            "[GPUDeviceContextPool] Retired exclusive context generation device="
            << device.toString() << " generation=" << retired_generation);
        return ExclusiveGPUDeviceGenerationRetirement(
            this,
            GPUDeviceContextGenerationRetirementReceipt{
                .device = device,
                .retired_generation = retired_generation,
            },std::move(native_owner));
    }

    void GPUDeviceContextPool::finishExclusiveGenerationRetirement(
        DeviceId device) noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto &retiring = device.is_cuda()
                             ? retiring_nvidia_contexts_
                             : retiring_amd_contexts_;
        if (retiring.erase(device.gpu_ordinal()) != 1u)
        {
            LOG_ERROR(
                "[GPUDeviceContextPool] Exclusive retirement lost its marker for "
                << device.toString());
            std::terminate();
        }
    }

    ExclusiveGPUDeviceGenerationRetirement::~ExclusiveGPUDeviceGenerationRetirement()
    {
        release();
    }

    ExclusiveGPUDeviceGenerationRetirement::ExclusiveGPUDeviceGenerationRetirement(
        ExclusiveGPUDeviceGenerationRetirement &&other) noexcept
        : pool_(other.pool_), receipt_(other.receipt_),native_owner_(std::move(other.native_owner_))
    {
        other.pool_ = nullptr;
        other.receipt_ = {};
    }

    ExclusiveGPUDeviceGenerationRetirement &
    ExclusiveGPUDeviceGenerationRetirement::operator=(
        ExclusiveGPUDeviceGenerationRetirement &&other) noexcept
    {
        if (this == &other)
            return *this;
        release();
        pool_ = other.pool_;
        receipt_ = other.receipt_;
        native_owner_=std::move(other.native_owner_);
        other.pool_ = nullptr;
        other.receipt_ = {};
        return *this;
    }

    void ExclusiveGPUDeviceGenerationRetirement::retireNativeObligationsAfterValidatedReset()noexcept
    {
        if(!pool_)std::terminate();
        if(native_owner_)
        {
            if(native_owner_->device_!=receipt_.device || native_owner_->generation_!=receipt_.retired_generation)std::terminate();
            native_owner_->physicallyRetired();
        }
        try {pool_->confirmPreparedPhysicalRetirement(receipt_.device);}
        catch(...) {std::terminate();}
    }

    void ExclusiveGPUDeviceGenerationRetirement::release() noexcept
    {
        if (!pool_)
            return;
        if(native_owner_ && !native_owner_->discharged())std::terminate();
        native_owner_.reset();
        pool_->finishExclusiveGenerationRetirement(receipt_.device);
        pool_ = nullptr;
        receipt_ = {};
    }

} // namespace llaminar2
