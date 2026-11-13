/**
 * @file CPUGroupedMoEStorageBacking.h
 * @brief Actual owned payload of one frozen CPU grouped-MoE workspace domain.
 *
 * The existing workspace manager owns this concrete bundle; stages may retain
 * it without retaining a manager pointer. Every backing owns its original PMA
 * claim. Kernels retain only the shared router vector, while their publication
 * metadata remains graph-local. No separate budget or live ledger exists here.
 */
#pragma once
#include "CPUGroupedMoEStorageLayout.h"
#include <memory>
namespace llaminar2
{
class FP32Tensor;
/** @brief Complete immutable-topology storage shared only by serial graph participants. */
class CPUGroupedMoEStorageBacking final
{
public:
    /**
     * @brief Materialize all eight final roots under their declared PMA owner.
     * @param layout Canonical complete source/allocator/row envelope.
     * @param authority Mandatory existing rank-bound allocation authority.
     * @throws std::invalid_argument For a partial layout or missing authority.
     */
    CPUGroupedMoEStorageBacking(const CPUGroupedMoEStorageLayout &layout,
        std::shared_ptr<PhysicalMemoryAuthority> authority);
    CPUGroupedMoEStorageBacking(const CPUGroupedMoEStorageBacking &)=delete;
    CPUGroupedMoEStorageBacking &operator=(const CPUGroupedMoEStorageBacking &)=delete;
    /** @return Exact frozen source and physical allocation declaration. */
    [[nodiscard]] const CPUGroupedMoEStorageLayout &layout() const noexcept {return layout_;}
    /** @return Actual allocator bytes in all live native backings. */
    [[nodiscard]] size_t physicalBytes() const;
    /** @return Whether any tensor/router child survives outside this bundle. */
    [[nodiscard]] bool hasExternalOwners() const noexcept;
    /** @brief Clear scratch data without allocation or topology changes. */
    void zero();
    std::shared_ptr<FP32Tensor> batch,gate,up,output; ///< Actual owned native tensors.
    AlignedVector<float> route_outputs; ///< Saved rows or decode FP32 SwiGLU, according to role.
    AlignedVector<Q8_1Block> router_q8,swiglu_q8; ///< Ordered route payloads.
    std::shared_ptr<AlignedVector<Q8_1Block>> kernel_router_q8; ///< Actual router publication backing, metadata separate.
private:
    const CPUGroupedMoEStorageLayout layout_; ///< One complete physical identity, never a live ledger.
};
}
