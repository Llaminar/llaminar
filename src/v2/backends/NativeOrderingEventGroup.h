/**
 * @file NativeOrderingEventGroup.h
 * @brief Original-PMA commitment for a closed existing event program; owns no event creators.
 */
#pragma once
#include "NativeOrderingEventProgram.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <memory>
namespace llaminar2 {
class GPUContextProvenance;
class NativeOrderingEventBudgetProjection;
class ReadAheadNativeParentTerminal;
class OriginalSlotCaptureTerminals;
class DeviceGraphExecutor;
namespace qwen4exp {class ImmutableLookupSubmissionRetirement;namespace detail {class PooledNativeRootStorage;}}
namespace test {struct NativeOrderingEventGroupReplayAccess;}
/** @brief Group token lifetime brackets existing root/tensor/boundary event handles. */
class NativeOrderingEventGroup final {
public:
    /** @return Closed reviewed group bound on original CPU/GPU observations before optional quota. */
    static PhysicalMemoryPlan bom(NativeOrderingEventProgram,PhysicalMemoryResource host,PhysicalMemoryResource gpu);
    /** @return Validated closed C0-to-positive group growth, rejecting backward or changed program roles. */
    static PhysicalMemoryPlan growthBOM(NativeOrderingEventProgram floor,NativeOrderingEventProgram actual,PhysicalMemoryResource host,PhysicalMemoryResource gpu);
    /** @return Original same-PMA tokens, allocated before the first existing native creator. */
    static std::unique_ptr<NativeOrderingEventGroup> claim(std::shared_ptr<PhysicalMemoryAuthority>,const GPUContextProvenance&,NativeOrderingEventProgram);
    /** @brief Reject attempted handles not explicitly completed; retain original physical-lifetime tokens. */
    ~NativeOrderingEventGroup()noexcept;
    NativeOrderingEventGroup(const NativeOrderingEventGroup&)=delete;
    NativeOrderingEventGroup& operator=(const NativeOrderingEventGroup&)=delete;
private:
    friend class qwen4exp::detail::PooledNativeRootStorage;
    friend class qwen4exp::ImmutableLookupSubmissionRetirement;
    friend class ReadAheadNativeParentTerminal;
    friend class OriginalSlotCaptureTerminals;
    // The original cache owns its singleton native creator/destructor; this
    // friendship only brackets that existing handle with its admitted group.
    friend class DeviceGraphExecutor;
    friend struct test::NativeOrderingEventGroupReplayAccess;
    /** @brief Mark possible backing retention BEFORE the first original event create attempt. */
    void beginNativeAttempt();
    /** @brief Call only after all original setup/tensor/boundary event destroy operations completed. */
    void completeNativeHandles()noexcept;
    static std::unique_ptr<NativeOrderingEventGroup> claimWithProjection(std::shared_ptr<PhysicalMemoryAuthority>,const GPUContextProvenance&,NativeOrderingEventProgram,const NativeOrderingEventBudgetProjection&);
    void requireOriginal(const std::shared_ptr<PhysicalMemoryAuthority>&,const GPUContextProvenance&,NativeOrderingEventProgram)const;
    struct State;
    explicit NativeOrderingEventGroup(std::unique_ptr<State>);
    std::unique_ptr<State> state_;
};
}
