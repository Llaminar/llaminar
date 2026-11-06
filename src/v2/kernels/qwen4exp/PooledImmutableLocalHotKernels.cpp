/**
 * @file PooledImmutableLocalHotKernels.cpp
 * @brief Checked original-owner and exact-stream bridge to stateless CUDA/HIP local cache code.
 *
 * Initialization alone prepares native codebook tables and physical setup.
 * Captured phases consume only sealed original producer spans and prebound
 * roots. The native parent owns Probe producer dependencies; neither output readiness
 * nor demand consumption is inferred from a response or host-side value.
 */
#include "PooledImmutableLocalHotKernels.h"
#include "PooledImmutableLocalHotOperands.h"
#include "PooledImmutableLocalHotPreparedStorage.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <stdexcept>

namespace llaminar2::qwen4exp {
namespace {
/** @brief Reject default and missing streams before invoking any native adapter. */
void exactStream(void* stream)
{
    if (!stream || reinterpret_cast<std::uintptr_t>(stream) <= 2)
        throw std::invalid_argument("Local hot kernels require their exact non-default stream");
}
}
void IPooledImmutableLocalHotKernels::initialize(PooledImmutableLocalHotPreparedStorage& storage, void* stream) const
{
    exactStream(stream);
    if (isGraphCaptureActive() || backend() != storage.geometry().allocator().device.type || !codeIdentity())
        throw std::invalid_argument("Local hot initialization requires its original backend outside capture");
    const auto tables = prepareTables(storage.geometry().allocator().device, storage.plan().nativeFormat());
    storage.initializeNative(backend(), tables, stream,
        [&](const local_hot::NativeRoots& roots) { initializeNative(roots, stream); });
}

void IPooledImmutableLocalHotKernels::enqueue(PooledImmutableLocalHotAction action,
    PooledImmutableLocalHotPreparedStorage& storage, const PooledImmutableLocalHotOperands& operands,
    void* stream) const
{
    exactStream(stream);
    if (backend() != storage.geometry().allocator().device.type || !codeIdentity())
        throw std::invalid_argument("Local hot phase changed its original native backend/code");
    storage.requireInitialized();
    operands.require(action, storage);
    switch (action) {
    case PooledImmutableLocalHotAction::Probe:
        launchProbe(operands.nativeQuery(storage), stream);
        break;
    case PooledImmutableLocalHotAction::LocalRead:
        launchRead(operands.nativeRead(storage), stream);
        break;
    case PooledImmutableLocalHotAction::AcceptResult:
        launchResult(operands.nativeResult(storage), stream);
        break;
    case PooledImmutableLocalHotAction::ResetRequest:
        launchRequestReset(operands.nativeRequestReset(storage), stream);
        break;
    case PooledImmutableLocalHotAction::ConsumeDemand:
    case PooledImmutableLocalHotAction::CopyFills:
    case PooledImmutableLocalHotAction::CompleteFills:
    case PooledImmutableLocalHotAction::Adopt:
    case PooledImmutableLocalHotAction::Discard:
    case PooledImmutableLocalHotAction::Cancel:
    case PooledImmutableLocalHotAction::Retire:
        launchFrame(action, operands.nativeFrame(storage), stream);
        break;
    default:
        throw std::invalid_argument("Unknown local hot native action");
    }
}
void IPooledImmutableLocalHotKernels::packTerminal(PooledImmutableLocalHotPreparedStorage& storage,
    const DeviceGenerationControlBinding& control,int requests,const PooledNativeTransferSpan& output,void* stream)const
{
    exactStream(stream);storage.requireInitialized();const auto device=storage.geometry().allocator().device;
    control.requireRequest(device,0);output.require(device,alignof(local_hot::TerminalObservation));
    const auto bytes=local_hot::terminalObservationBytes(requests,sampling_math::kDeviceGenerationControlCount);
    if(isGraphCaptureActive() || backend()!=device.type || !codeIdentity() || output.bytes()!=bytes ||
        std::size_t(requests)>control.owner()->shape()[0])
        throw std::invalid_argument("Local terminal pack changed its original complete terminal storage/geometry");
    const auto roots=storage.roots();
    for(const auto& root:{roots.cache,roots.frame,roots.native})
        if(root)output.requireDisjoint(PooledNativeTransferSpan(root,0,root->size_bytes()));
    const auto* row=control.row();auto* destination=static_cast<std::byte*>(output.pointer(device,alignof(local_hot::TerminalObservation)));
    const auto inputBegin=reinterpret_cast<std::uintptr_t>(row),outputBegin=reinterpret_cast<std::uintptr_t>(destination);
    const auto inputBytes=std::size_t(requests)*sampling_math::kDeviceGenerationControlCount*sizeof(int);
    if(outputBegin<inputBegin+inputBytes && inputBegin<outputBegin+bytes)
        throw std::invalid_argument("Local terminal output cannot alias its original mutable controller");
    local_hot::NativeTerminal terminal;static_cast<local_hot::NativeRoots&>(terminal)=storage.nativeRoots();
    terminal.control=row;terminal.output=destination;terminal.requests=requests;
    terminal.control_words=sampling_math::kDeviceGenerationControlCount;terminal.output_bytes=bytes;launchTerminal(terminal,stream);
}
} // namespace llaminar2::qwen4exp
