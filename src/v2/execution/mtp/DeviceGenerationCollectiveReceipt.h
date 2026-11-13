/**
 * @file DeviceGenerationCollectiveReceipt.h
 * @brief Typed native collective observation in the existing canonical request row.
 *
 * No caller supplies a raw address or word offset. Existing controller ownership,
 * arena/runtime identity and request reset remain authoritative. The native
 * counted collective accumulates actual outgoing ring-send bytes; no transfer/readback,
 * scheduling decision, allocation or second statistics ledger is introduced.
 */
#pragma once
#include "DeviceGenerationControlBinding.h"
#include "kernels/qwen4exp/QwenFixedHeadTerminalCounters.h"
#include "kernels/qwen4exp/QwenFixedHeadPipelineCounters.h"
#include <limits>
namespace llaminar2 {
/** @brief Closed native observation roles, separate from count/control authority. */
enum class DeviceGenerationCollectiveReceiptKind {
    FixedHeadHeader, FixedHeadPayload, FixedHeadPipelineRequest, FixedHeadPipelineResponse, FixedHeadPipelineGather
};
/** @brief Immutable capability for exactly one original aligned observation pair. */
class DeviceGenerationCollectiveReceipt final {
public:
    /** @brief Bind a source-authored original frame and native contribution role before graph construction.
     * @throws std::invalid_argument For malformed geometry or an unbounded request lifetime.
     */
    static DeviceGenerationCollectiveReceipt fixedHead(DeviceGenerationControlBinding control,
        DeviceGenerationCollectiveReceiptKind kind,int rows,int heads,std::size_t rowBytes) {
        if((kind!=DeviceGenerationCollectiveReceiptKind::FixedHeadHeader && kind!=DeviceGenerationCollectiveReceiptKind::FixedHeadPayload) ||
            !qwen4exp::validQwenFixedHeadCounterLifetime(rows,heads,rowBytes) ||
            rowBytes>std::numeric_limits<std::size_t>::max()/static_cast<std::size_t>(heads)/static_cast<std::size_t>(rows))
            throw std::invalid_argument("Native receipt requires bounded original fixed-head request geometry");
        return DeviceGenerationCollectiveReceipt(control,kind,rows,heads,rowBytes);
    }
    /** @brief Bind the original request row to the exact native addressed or gather operation. */
    static DeviceGenerationCollectiveReceipt fixedHeadPipeline(DeviceGenerationControlBinding control,
        DeviceGenerationCollectiveReceiptKind kind, int rows, int heads, std::size_t rowBytes) {
        const auto calls = 3 * (std::uint64_t{INT_MAX} + 1);
        // Each logical token contributes at most all model heads to one owner
        // and one actual send to each other possible original head owner.
        constexpr std::uint64_t max_heads = 16;
        constexpr auto max_sends = max_heads - 1;
        if ((kind != DeviceGenerationCollectiveReceiptKind::FixedHeadPipelineRequest &&
             kind != DeviceGenerationCollectiveReceiptKind::FixedHeadPipelineResponse &&
             kind != DeviceGenerationCollectiveReceiptKind::FixedHeadPipelineGather) || rows <= 0 || heads <= 0 ||
            static_cast<std::uint64_t>(heads) > max_heads || !rowBytes || rowBytes > UINT64_MAX / max_heads / max_sends ||
            static_cast<std::uint64_t>(rows) > UINT64_MAX / calls / (max_heads * max_sends * rowBytes) ||
            rowBytes > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(heads) / static_cast<std::size_t>(rows))
            throw std::invalid_argument("Native fixed pipeline receipt requires bounded original row and head geometry");
        return DeviceGenerationCollectiveReceipt(control, kind, rows, heads, rowBytes);
    }
    /** @return Whether this receipt owns one addressed endpoint and its native completion count. */
    bool isFixedHeadPipelineEndpoint() const noexcept {
        return kind_ == DeviceGenerationCollectiveReceiptKind::FixedHeadPipelineRequest ||
            kind_ == DeviceGenerationCollectiveReceiptKind::FixedHeadPipelineResponse;
    }
    /** @return Original aligned endpoint completion counter, written only after native validation. */
    NativeCounter64 pipelineEndpointCompletion(DeviceId device) const {
        if (!isFixedHeadPipelineEndpoint()) throw std::invalid_argument("Gather receipt cannot count addressed endpoints");
        requireFor(device);
        return qwen4exp::FixedHeadPipelineCounterRow{control_.row()}.metric(
            qwen4exp::FixedHeadPipelineTerminalMetric::CompletedReceipts);
    }
    /** @return Whether this receipt owns actual fixed pipeline PLE gather sends. */
    bool isFixedHeadPipelineGather() const noexcept {
        return kind_ == DeviceGenerationCollectiveReceiptKind::FixedHeadPipelineGather;
    }
    /** @brief Reauthenticate original controller before preparation, enqueue or identity reuse. */
    void requireFor(DeviceId device)const {control_.requireFor(device);}
    /** @return Source-owned exact maximum send bank, never the mutable communication size. */
    std::size_t bankBytes()const noexcept {
        return kind_==DeviceGenerationCollectiveReceiptKind::FixedHeadHeader?8:
            static_cast<std::size_t>(rows_)*static_cast<std::size_t>(heads_)*row_bytes_;
    }
    /** @return Original controller root, declared as preallocated InOut by the native writer. */
    TensorBase *owner() const noexcept {return control_.owner();}
    /** @return Sole controller arena role; never a caller-selected alias. */
    static constexpr BufferId bufferId() noexcept {return BufferId::MTP_GENERATION_CONTROL;}
    /** @return Original aligned native pair; no mutable value is read. */
    unsigned long long *native(DeviceId device)const {
        requireFor(device);
        using Kind = DeviceGenerationCollectiveReceiptKind;
        using Metric = qwen4exp::FixedHeadPipelineTerminalMetric;
        switch (kind_) {
        case Kind::FixedHeadHeader:
            return reinterpret_cast<unsigned long long *>(control_.row() + static_cast<int>(qwen4exp::QwenFixedHeadCounter::HeaderNativeSendBytes));
        case Kind::FixedHeadPayload:
            return reinterpret_cast<unsigned long long *>(control_.row() + static_cast<int>(qwen4exp::QwenFixedHeadCounter::PayloadNativeSendBytes));
        case Kind::FixedHeadPipelineRequest:
            return reinterpret_cast<unsigned long long *>(qwen4exp::FixedHeadPipelineCounterRow{control_.row()}.metric(Metric::RequestBytes).words);
        case Kind::FixedHeadPipelineResponse:
            return reinterpret_cast<unsigned long long *>(qwen4exp::FixedHeadPipelineCounterRow{control_.row()}.metric(Metric::ResponseBytes).words);
        case Kind::FixedHeadPipelineGather:
            return reinterpret_cast<unsigned long long *>(qwen4exp::FixedHeadPipelineCounterRow{control_.row()}.metric(Metric::GatherBytes).words);
        }
        throw std::logic_error("Native collective receipt lost its original role");
    }
    /** @return Complete borrowed storage, role and original geometry identity. */
    std::uint64_t captureIdentity()const noexcept {
        auto value=control_.captureIdentity();
        for(const auto word:{static_cast<std::uint64_t>(kind_),static_cast<std::uint64_t>(rows_),
            static_cast<std::uint64_t>(heads_),static_cast<std::uint64_t>(row_bytes_)}) {value^=word;value*=1099511628211ULL;}
        return value?value:1;
    }
private:
    /** @brief Retain original immutable borrow; no byte/value ownership changes. */
    DeviceGenerationCollectiveReceipt(DeviceGenerationControlBinding control,DeviceGenerationCollectiveReceiptKind kind,
        int rows,int heads,std::size_t bytes):control_(control),kind_(kind),rows_(rows),heads_(heads),row_bytes_(bytes){}
    DeviceGenerationControlBinding control_;
    DeviceGenerationCollectiveReceiptKind kind_;
    int rows_,heads_;std::size_t row_bytes_;
};
} // namespace llaminar2
