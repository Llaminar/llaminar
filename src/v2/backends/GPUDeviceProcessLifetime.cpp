/**
 * @file GPUDeviceProcessLifetime.cpp
 * @brief Retire canonical GPU contexts before main returns, not from exit order.
 *
 * This control authority holds no memory quantities or second context registry.
 * It may retain original PMA token leaves for certified process-runtime pools;
 * lexical shutdown closes use but cannot claim their physical bytes reclaimed.
 * It closes native admission before asking GPUDeviceContextPool to release its
 * exact existing generations. Late vendor/device runtime initialization cannot
 * move that lexical retirement behind runtime process-exit callbacks.
 */
#include "GPUDeviceProcessLifetime.h"
#include "GPUDeviceContextPool.h"
#include "execution/local_execution/graph/RetainedHostServiceProgram.h"
#include "NativeContextRetirement.h"
#include "collective/NativeStoragePreparationPmaBinding.h"
#include "NativeContextOpaqueContract.h"
#include <vector>
#include <algorithm>
#include "../collective/BackendRouter.h"
#include "../kernels/KernelFactory.h"
#include "../utils/Logger.h"
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>
#include <limits>
#include <stdexcept>
#include <thread>

namespace llaminar2
{
    namespace
    {
        /** @brief One process's native admission state, unrelated to byte accounting. */
        enum class AdmissionPhase { Unowned, Active, Retiring, Retired };
        /** @brief Serialize lifecycle changes while native retirement runs outside this lock. */
        struct ProcessAuthority
        {
            std::mutex mutex;
            std::condition_variable complete;
            AdmissionPhase phase = AdmissionPhase::Unowned;
            const GPUDeviceProcessLifetime *owner = nullptr;
            std::size_t native_contexts = 0; ///< Lifecycle borrowers, never memory quantities.
            std::thread::id retiring_thread{};
            std::vector<std::shared_ptr<RetainedHostServiceProcessObligation>> host_roots;
            // Original token owners only: no byte counters or duplicate reservations.
            std::vector<std::shared_ptr<NativeContextRetirementSlotState>> process_roots;
        };
        /** @return The process-local lifecycle authority, with no GPU initialization. */
        ProcessAuthority &authority()
        {
            // Opaque native backing can survive lexical retirement and arbitrary
            // vendor static teardown order. Preserve original obligations until OS
            // terminal reclamation; never forge a PMA-zero receipt at main return.
            static auto* state = new ProcessAuthority;
            return *state;
        }
    }

    namespace {
        /** @brief Exact authority/control-block comparison shared by CPU and GPU process roots. */
        void requireSameAuthority(const std::shared_ptr<PhysicalMemoryAuthority>& first,const std::shared_ptr<PhysicalMemoryAuthority>& second){
            if(!first || !second || first.get()!=second.get() || first.owner_before(second) || second.owner_before(first))
                throw std::logic_error("Original CPU process root prohibits independent source authority adoption");
        }
    }
    void GPUDeviceProcessLifetime::requireOriginalHostServiceAuthority(const std::shared_ptr<PhysicalMemoryAuthority>& memory){
        auto& state=authority();std::lock_guard lock(state.mutex);
        if(state.phase!=AdmissionPhase::Active||!memory||memory->phase()!=PhysicalMemoryAuthorityPhase::Admitted)
            throw std::logic_error("CPU service requires original admitted process authority");
        for(const auto& root:state.process_roots)NativeContextRetirementOwner::requireOriginalAuthority(root,memory);
        for(const auto& root:state.host_roots)requireSameAuthority(root->memory_,memory);
    }
    void GPUDeviceProcessLifetime::retainHostServiceRoot(const std::shared_ptr<RetainedHostServiceProcessObligation>& root){
        auto& state=authority();std::lock_guard lock(state.mutex);
        if(state.phase!=AdmissionPhase::Active || !root || !root->memory_ || root->memory_->phase()!=PhysicalMemoryAuthorityPhase::Admitted)
            throw std::logic_error("CPU service process obligation requires original admitted CPU0 authority");
        if(const auto* token=std::get_if<PhysicalMemoryOwnerReservation>(&root->original_)){
            if(!token->valid() || token->owner()!=PhysicalMemoryOwner::RetainedHostServiceRuntime ||
               token->allocatorIdentity()!=PhysicalMemoryAllocatorIdentity{root->memory_->worldRank(),DeviceId::cpu()})
                throw std::logic_error("CPU service process obligation lost its original CPU0 reservation");
        }
        for(const auto& prior:state.process_roots)NativeContextRetirementOwner::requireOriginalAuthority(prior,root->memory_);
        for(const auto& prior:state.host_roots)requireSameAuthority(prior->memory_,root->memory_);
        if(root->active_)throw std::logic_error("CPU service creator obligation already active");
        state.host_roots.push_back(root); // Allocation precedes pthread_create; failed publication releases the unattempted token.
        root->active_=true;
    }
    void GPUDeviceProcessLifetime::completeHostServiceUse(const std::shared_ptr<RetainedHostServiceProcessObligation>& root)noexcept{
        auto& state=authority();std::lock_guard lock(state.mutex);
        const auto found=std::find(state.host_roots.begin(),state.host_roots.end(),root);
        if(found==state.host_roots.end())return; // No native creator was attempted; ordinary original-token rollback remains valid.
        if(state.phase!=AdmissionPhase::Active||!root->active_)std::terminate();
        root->active_=false;
    }

    void GPUDeviceProcessLifetime::requireOriginalProcessAuthority(DeviceId device,const std::shared_ptr<PhysicalMemoryAuthority>& memory)
    {
        auto& state=authority();std::lock_guard lock(state.mutex);
        if(state.phase!=AdmissionPhase::Active || !device.is_gpu() || !memory)throw std::logic_error("Original source admission requires live process authority");
        for(const auto& root:state.process_roots)NativeContextRetirementOwner::requireOriginalProcessAuthority(root,device,memory);
        for(const auto& root:state.host_roots)requireSameAuthority(root->memory_,memory);
    }
    void GPUDeviceProcessLifetime::retainPreparingOpaqueRoot(
        const std::shared_ptr<NativeStoragePreparationPmaBinding>& source,
        const std::shared_ptr<PhysicalMemoryOpaqueRuntimeCommitment>& opaque,const NativeContextOpaqueContract& contract)
    {
        if(!source || !opaque || contract.lifetime()!=NativePhysicalLifetime::ProcessRuntime)
            throw std::invalid_argument("Preparing process root requires its original source, opaque commitment and closed physical lifetime");
        opaque->requireOriginalProcessRetention(source->memory(),contract);
        auto root=NativeContextRetirementOwner::preparingProcessRoot(source,opaque,contract);
        auto& state=authority();std::lock_guard lock(state.mutex);
        if(state.phase!=AdmissionPhase::Active)throw std::logic_error("Preparing root cannot reopen process admission");
        for(const auto& prior:state.process_roots)NativeContextRetirementOwner::requireUniqueProcessRoot(prior,root);
        for(const auto& host:state.host_roots)NativeContextRetirementOwner::requireOriginalAuthority(root,host->memory_);
        // Allocate before arming even the PMA lifecycle. A failed allocation or
        // stale original commitment leaves no published process owner/native call.
        state.process_roots.reserve(state.process_roots.size()+1);
        opaque->beginNative();
        state.process_roots.push_back(std::move(root)); // Pre-reserved noexcept shared_ptr move.
    }
    void GPUDeviceProcessLifetime::retainProcessRoot(const std::shared_ptr<NativeContextRetirementSlotState>& root)
    {
        auto& state=authority();std::lock_guard lock(state.mutex);
        if(state.phase!=AdmissionPhase::Active || !root)throw std::logic_error("Process root requires original live entry authority");
        for(const auto& prior:state.process_roots)NativeContextRetirementOwner::requireUniqueProcessRoot(prior,root);
        for(const auto& host:state.host_roots)NativeContextRetirementOwner::requireOriginalAuthority(root,host->memory_);
        state.process_roots.push_back(root); // Before native creation; allocation may roll back safely.
        NativeContextRetirementOwner::startProcessUse(root);
    }
    std::shared_ptr<NativeContextRetirementSlotState> GPUDeviceProcessLifetime::borrowProcessRoot(
        const std::shared_ptr<PhysicalMemoryAuthority>& memory,DeviceId device,const std::string& certificate,const std::string& population)
    {
        auto& state=authority();std::lock_guard lock(state.mutex);
        if(state.phase!=AdmissionPhase::Active)throw std::logic_error("Process population use is closed");
        for(const auto& root:state.process_roots)
            if(NativeContextRetirementOwner::matchesProcessRoot(root,memory,device,certificate,population))return root;
        return {};
    }
    void GPUDeviceProcessLifetime::finishProcessUse(const std::shared_ptr<NativeContextRetirementSlotState>& root)noexcept
    {
        auto& state=authority();std::lock_guard lock(state.mutex);
        NativeContextRetirementOwner::finishProcessUse(root);
    }

    GPUDeviceProcessLifetime::GPUDeviceProcessLifetime()
    {
        // The pool is core-only. Materialize it before admission without touching
        // a vendor factory, SDK, stream, driver, or physical inventory.
        (void)GPUDeviceContextPool::instance();
        auto &state = authority();
        std::lock_guard lock(state.mutex);
        if (state.phase != AdmissionPhase::Unowned)
            throw std::logic_error("GPU process lifetime requires one entry owner and cannot reopen retired admission");
        state.owner = this;
        state.phase = AdmissionPhase::Active;
    }

    GPUDeviceProcessLifetime::NativeContextLease::~NativeContextLease() noexcept
    {
        if (active_) GPUDeviceProcessLifetime::releaseNativeContext();
    }

    GPUDeviceProcessLifetime::NativeContextLease GPUDeviceProcessLifetime::acquireNativeContext()
    {
        auto &state = authority();
        std::lock_guard lock(state.mutex);
        if (state.phase != AdmissionPhase::Active)
            throw std::logic_error("Native GPU context admission requires a live process-entry owner");
        if (state.native_contexts == std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("Native GPU context lifetime count exhausted");
        ++state.native_contexts;
        return NativeContextLease(true);
    }

    void GPUDeviceProcessLifetime::releaseNativeContext() noexcept
    {
        auto &state = authority();
        std::lock_guard lock(state.mutex);
        if (state.native_contexts == 0 || state.phase == AdmissionPhase::Unowned ||
            state.phase == AdmissionPhase::Retired)
        {
            LOG_ERROR("[GPUDeviceProcessLifetime] Native context released outside its exact live process lifetime");
            std::terminate();
        }
        --state.native_contexts;
    }

    GPUDeviceProcessLifetime::~GPUDeviceProcessLifetime() noexcept { retire(); }
    void GPUDeviceProcessLifetime::retire() noexcept { retireOwner(this); }
    void GPUDeviceProcessLifetime::retireActive() noexcept { retireOwner(nullptr); }

    void GPUDeviceProcessLifetime::requireActive()
    {
        auto &state = authority();
        std::lock_guard lock(state.mutex);
        if (state.phase != AdmissionPhase::Active)
            throw std::logic_error("Native GPU admission requires an active GPUDeviceProcessLifetime in the process entry frame");
    }

    void GPUDeviceProcessLifetime::retireOwner(const GPUDeviceProcessLifetime *expected) noexcept
    {
        auto &state = authority();
        std::unique_lock lock(state.mutex);
        if (state.phase == AdmissionPhase::Unowned || state.phase == AdmissionPhase::Retired)
            return;
        if (expected && state.owner != expected)
        {
            LOG_ERROR("[GPUDeviceProcessLifetime] Retirement requested by a foreign process owner");
            std::terminate();
        }
        if (state.phase == AdmissionPhase::Retiring)
        {
            if (state.retiring_thread == std::this_thread::get_id())
            {
                LOG_ERROR("[GPUDeviceProcessLifetime] Recursive terminal retirement cannot prove completion");
                std::terminate();
            }
            state.complete.wait(lock, [&] { return state.phase == AdmissionPhase::Retired; });
            return;
        }
        if(std::any_of(state.host_roots.begin(),state.host_roots.end(),[](const auto& root){return root->active_;}))std::terminate(); // Join/unmap every CPU service before lexical process retirement.
        state.phase = AdmissionPhase::Retiring;
        state.retiring_thread = std::this_thread::get_id();
        lock.unlock();
        try
        {
            // No admission-state lock is held while worker joins/native calls run.
            // Pool shutdown itself rejects retained graph/submission owners before
            // removing a generation, and native release failures remain fatal.
            GlobalBackendRouter::shutdown();
            llaminar::v2::kernels::KernelFactory::clearCache();
            GPUDeviceContextPool::instance().shutdown();
        }
        catch (const std::exception &error)
        {
            LOG_ERROR("[GPUDeviceProcessLifetime] Terminal GPU retirement failed: " << error.what());
            std::terminate();
        }
        catch (...)
        {
            LOG_ERROR("[GPUDeviceProcessLifetime] Terminal GPU retirement failed with an unknown exception");
            std::terminate();
        }
        lock.lock();
        if (state.native_contexts != 0)
        {
            LOG_ERROR("[GPUDeviceProcessLifetime] Terminal retirement still has "
                << state.native_contexts << " direct or incompletely constructed native context owners");
            std::terminate();
        }
        state.owner = nullptr;
        state.retiring_thread = {};
        state.phase = AdmissionPhase::Retired;
        lock.unlock();
        state.complete.notify_all();
    }
}
