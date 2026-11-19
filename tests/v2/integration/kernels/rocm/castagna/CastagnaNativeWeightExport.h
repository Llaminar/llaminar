/**
 * @file CastagnaNativeWeightExport.h
 * @brief Diagnostic B arm: original Castagna full kernel with canonical prepared weight readers.
 * The original A export stays unchanged. B retains fork arithmetic/scheduling except
 * unavoidable Q4_K precomposed-coefficient precision and multiplication association.
 * No GPU weight allocation or repack is owned by this interface.
 */
#pragma once
#include "CastagnaHcPersistExport.h"
#include "kernels/common/DeviceNativeQuantisedMatrixDesc.h"
#include <span>
namespace qfn_castagna_native {
using Matrix=llaminar2::DeviceNativeQuantisedMatrixDesc;
/** @brief Original prepared metadata passed directly, without packed-byte copies. */
struct NativeWeights {
    Matrix hc_down,hc_up,shared_gate,shared_up,shared_down;
    const Matrix* expert_gate=nullptr;
    const Matrix* expert_up=nullptr;
    const Matrix* expert_down=nullptr;
};
/** @brief Same original external inputs/output roles with prepared matrix ownership. */
struct Request {
    qfn_castagna_reference::Request original;
    NativeWeights native;
};
/** @brief Complete cold B identity and actual function resource admission. */
struct Admission {
    const void* function=nullptr;
    int device=-1;
    unsigned grid=0,threads=0,active_blocks_per_cu=0;
    hipFuncAttributes resources{};
    Request request;
};
}
/** @return Actual B full-kernel function, separate from unchanged A. */
extern "C" const void* qfn_castagna_native_weight_kernel() noexcept;
/** @brief Validate every original prepared source metadata record cold.
 * @param request Complete B binding. @param gate Host metadata matching the existing uploaded512-table.
 * @param up Same up metadata. @param down Same down metadata. @param device Exact original GPU.
 * @param admission Receives exact B function/resources/semantic request snapshot.
 * @return HIP status; no format conversion, allocation, fallback or launch.
 */
extern "C" hipError_t qfn_castagna_native_weight_prepare(const qfn_castagna_native::Request* request,
    std::span<const qfn_castagna_native::Matrix> gate,std::span<const qfn_castagna_native::Matrix> up,
    std::span<const qfn_castagna_native::Matrix> down,int device,qfn_castagna_native::Admission* admission) noexcept;
/** @brief Submit B on the exact stream after immutable binding admission.
 * @param request Same retained request. @param admission Cold B authority. @param stream Nondefault stream.
 * @return Immediate HIP launch status. Original global state must have one exclusive owner perGPU.
 */
extern "C" hipError_t qfn_castagna_native_weight_launch(const qfn_castagna_native::Request* request,
    const qfn_castagna_native::Admission* admission,hipStream_t stream) noexcept;
/** @brief Observe B's separate original module-global storage, cold only. */
extern "C" hipError_t qfn_castagna_native_weight_globals(qfn_castagna_reference::GlobalRegions* regions) noexcept;
/** @brief Untimed original topK helper observer over B's retained raw logits. */
extern "C" hipError_t qfn_castagna_native_weight_observe_selection(const qfn_castagna_native::Admission* admission,
    int* ids,float* weights,hipStream_t stream) noexcept;

/** @return Exact separate untimed B observer entry for captured identity inspection. */
extern "C" const void* qfn_castagna_native_weight_selection_observer_kernel() noexcept;
