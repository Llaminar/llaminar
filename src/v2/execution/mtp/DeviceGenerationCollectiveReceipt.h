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
#include <limits>
namespace llaminar2 {
/** @brief Closed native observation roles, separate from count/control authority. */
enum class DeviceGenerationCollectiveReceiptKind { FixedHeadHeader,FixedHeadPayload };
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
        const auto metric=kind_==DeviceGenerationCollectiveReceiptKind::FixedHeadHeader?
            qwen4exp::QwenFixedHeadCounter::HeaderNativeSendBytes:qwen4exp::QwenFixedHeadCounter::PayloadNativeSendBytes;
        return reinterpret_cast<unsigned long long *>(control_.row()+static_cast<int>(metric));
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
