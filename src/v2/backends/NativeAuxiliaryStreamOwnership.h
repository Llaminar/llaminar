/**
 * @file NativeAuxiliaryStreamOwnership.h
 * @brief Original-PMA commitment bracketing one native stream and its immutable scheduling class.
 *
 * This object creates no stream and keeps no second handle registry. The exact
 * native owner starts its commitment before creation and completes it only
 * after destruction. Named context streams delegate the same slot to their
 * original backend record; no worker reference cycle is introduced.
 */
#pragma once
#include "planning/PhysicalMemoryAuthority.h"
#include <memory>
#include <cstdint>
#include <string>
namespace llaminar2 {
class PreparedGPUContextUsePermit;
enum class GPUAuxiliaryStreamSchedulingClass : std::uint8_t;
class IWorkerGPUContext;
class NvidiaDeviceContext;
class AMDDeviceContext;
class MoEOverlayNodeLocalRankBatchTransport;
namespace test {struct NativeAuxiliaryStreamOwnershipReplayAccess;}
/** @brief One closed stream-only resource program, not a stream factory. */
class NativeAuxiliaryStreamOwnership final {
public:
    /** @return Installed conservative one-stream declaration on the exact resources. */
    static PhysicalMemoryPlan bom(PhysicalMemoryResource host,PhysicalMemoryResource gpu);
    /** @brief Reject a native attempt that never completed actual stream destruction. */
    ~NativeAuxiliaryStreamOwnership()noexcept;
    NativeAuxiliaryStreamOwnership(const NativeAuxiliaryStreamOwnership&)=delete;
    NativeAuxiliaryStreamOwnership& operator=(const NativeAuxiliaryStreamOwnership&)=delete;
private:
    friend class NvidiaDeviceContext;
    friend class AMDDeviceContext;
    friend class MoEOverlayNodeLocalRankBatchTransport;
    friend struct test::NativeAuxiliaryStreamOwnershipReplayAccess;
    /** @return Original source commitment before any native creator is invoked. */
    static std::unique_ptr<NativeAuxiliaryStreamOwnership> claim(const PreparedGPUContextUsePermit&,const std::string&);
    /** @brief Claim the same one-stream bound with an immutable native scheduling identity.
     * @param original Exact source-prepared worker and PMA.
     * @param name Immutable context-owned stream name.
     * @param scheduling Exact native priority class, not a second physical allocation family.
     * @return Original retirement obligation before any native creator.
     * @throws std::exception For unknown scheduling, foreign source or insufficient original admission.
     */
    static std::unique_ptr<NativeAuxiliaryStreamOwnership> claim(const PreparedGPUContextUsePermit& original,
        const std::string& name,GPUAuxiliaryStreamSchedulingClass scheduling);
    /** @brief Authenticate repeated named-stream lookup against the same original PMA and worker. */
    void requireOriginal(const PreparedGPUContextUsePermit&,const IWorkerGPUContext*,const std::string&)const;
    /** @brief Reject a changed source, name or scheduling class before native creation.
     * @param original Same prepared worker/PMA permit.
     * @param worker Exact retained native owner.
     * @param name Immutable stream name.
     * @param scheduling Original native scheduling class.
     * @throws std::logic_error For any foreign or retired identity.
     */
    void requireOriginal(const PreparedGPUContextUsePermit& original,const IWorkerGPUContext* worker,
        const std::string& name,GPUAuxiliaryStreamSchedulingClass scheduling)const;
    /** @brief Only the original native context shutdown may retire admitted stream handles. */
    enum class RetirementScope { PublicReset, OriginalContextShutdown };
    /** @brief Validate a context reset before any named stream/event mutation.
     * @param scope Actual original context phase, never a caller-owned handle count.
     * @throws std::logic_error For public reset, unknown scope or an unowned/retired record.
     * This validates only; actual native destruction must precede completion.
     */
    void requireRetirement(RetirementScope scope)const;
    /** @brief Transfer preclaimed tokens before the first actual stream-creation attempt. */
    void beginNativeAttempt();
    /** @brief The existing backend named-stream record now owns this exact live handle. */
    void delegateToContext()noexcept;
    /** @brief Call only after the exact original native stream was destroyed (or never attempted). */
    void completeNativeHandles()noexcept;
    struct State;
    explicit NativeAuxiliaryStreamOwnership(std::unique_ptr<State>);
    std::unique_ptr<State> state_;
};
}
