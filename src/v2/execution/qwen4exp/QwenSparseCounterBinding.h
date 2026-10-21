/**
 * @file QwenSparseCounterBinding.h
 * @brief Setup-only borrow of the original admitted device-generation control row.
 *
 * This capability adds no tensor coherence writer: independent QSA branches use
 * integer atomics, while the existing complete generation parent and terminal
 * copy own reset, ordering and observation. Mutable counter values are excluded
 * from capture identity; original arena, owner, device and exact row are sealed.
 */
#pragma once
#include "execution/mtp/DeviceGenerationControlBinding.h"
#include "kernels/qwen4exp/QwenSparseTerminalCounters.h"
#include <optional>
#include <stdexcept>
namespace llaminar2 {
/** @brief Immutable native control-row capability, constructible only from its canonical arena role. */
class QwenSparseCounterBinding final {
public:
    /** @brief Authenticate already materialized canonical storage without allocating or reading words.
     * @param arena Existing execution-owner admitted arena.
     * @param device Exact GPU participant; CPU reports after its actual host operations.
     * @param request Original request row, independent of graph/layer ordinal.
     * @param localLayerBound Admitted local decoder layer bound, including non-sparse layers conservatively.
     * @param verifierRowBound Admitted maximum verifier rows, not a mutable live count.
     * @return Borrow valid through the existing generation transaction/terminal lifetime.
     */
    static QwenSparseCounterBinding admit(BufferArena& arena,DeviceId device,int request,
        int localLayerBound,int verifierRowBound) {
        if(!qwen4exp::validQwenSparseCounterLifetimeBound(localLayerBound,verifierRowBound))
            throw std::invalid_argument("QSA terminal counter lifetime bound exceeds UINT64");
        return QwenSparseCounterBinding(DeviceGenerationControlBinding::admit(arena,device,request));
    }
    /** @brief Reauthenticate storage identity before capture/submit; no live scalar is read. */
    void requireFor(DeviceId device)const {
        control_.requireFor(device);
    }
    /** @brief Authenticate the stage's explicit canonical request slot before adding nodes or recording writes. */
    void requireRequest(DeviceId device,int request)const {
        control_.requireRequest(device,request);
    }
    /** @brief Resolve the original request row without a transfer or mutable host shadow. */
    qwen4exp::QwenSparseWorkCounterRow native()const noexcept {
        return {control_.row(),sampling_math::kDeviceGenerationControlCount};
    }
    /** @brief Complete immutable metadata for graph-cache identity. */
    std::uint64_t captureIdentity()const noexcept {
        return control_.captureIdentity();
    }
private:
    /** @brief Retain the shared canonical controller capability, without another storage authority. */
    explicit QwenSparseCounterBinding(DeviceGenerationControlBinding control):control_(control){}
    DeviceGenerationControlBinding control_;
};
} // namespace llaminar2
