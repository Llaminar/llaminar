/**
 * @file PooledImmutableLocalHotTerminal.h
 * @brief Immutable whole-request projection of actual lifetime local-cache work.
 *
 * Counters remain device-owned and survive request reset. Only completed
 * terminal snapshots are compared on the host; their delta decides no GPU
 * execution or cache state. L1 original rows and compact L2 rows retain distinct
 * units. The packed output is separate from the unchanged controller ABI.
 */
#pragma once
#include "PooledImmutableLocalHotState.h"
#include "kernels/common/DeviceGenerationControlLayout.h"
#include <array>
#include <limits>
#include <mutex>
#include <stdexcept>
namespace llaminar2::qwen4exp::local_hot {
/** @brief Fifteen completed producer totals followed by explicit physical terminal validity. */
enum class TerminalObservationStatus : std::uint64_t { Complete,Unretired,Failed };
/** @brief No model, sampler, policy, ordinal or admission state is mirrored by this record. */
struct alignas(64) TerminalObservation {
    std::uint64_t values[15]{};
    TerminalObservationStatus status=TerminalObservationStatus::Complete;
};
static_assert(sizeof(TerminalObservation)==128);
/** @return Actual completed producer snapshot; an unretired/poisoned frame cannot appear as zero successful work. */
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
inline TerminalObservation snapshotTerminalCounters(const Counters& c,const FrameIdentity& identity,std::uint32_t heldPins,adaptive_lookup::Status frameStatus)
{
    TerminalObservation out{{c.read_original_ids,c.local_hit_original_ids,c.all_hit_wait_avoided_original_ids,
        c.consumed_original_ids,c.future_on_time_native_rows,c.future_late_native_rows,c.future_useful_native_rows,
        c.future_wasted_native_rows,c.promotions,c.evictions,c.request_key_payload_bytes,c.native_result_payload_bytes,
        c.lookup_control_header_bytes,c.acknowledgement_bytes,c.request_admission_bytes},TerminalObservationStatus::Complete};
    if(identity.disposition!=Disposition::Idle || heldPins)out.status=TerminalObservationStatus::Unretired;
    if(identity.terminal_error!=adaptive_lookup::Status::Success || c.reserved ||
        (frameStatus!=adaptive_lookup::Status::Success && frameStatus!=adaptive_lookup::Status::Cancelled))
        out.status=TerminalObservationStatus::Failed;
    return out;
}
/** @brief One actual receipt owner adds its validated live extent, failing before overflow mutation. */
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
inline bool accumulateCompletedWireCounter(std::uint64_t& total,std::uint64_t completed)
{
    if(completed>UINT64_MAX-total)return false;
    total+=completed;return true;
}
/** @return Exact metadata copy bytes for live control rows and one participant aggregate observation. */
inline std::size_t terminalObservationBytes(int requests,int controlWords)
{
    if(requests<=0 || controlWords!=sampling_math::kDeviceGenerationControlCount ||
       std::size_t(requests)>(std::numeric_limits<std::size_t>::max()-sizeof(TerminalObservation))/std::size_t(controlWords*sizeof(int)))
        throw std::invalid_argument("Local terminal observation requires complete canonical control geometry");
    const auto bytes=std::size_t(requests)*std::size_t(controlWords)*sizeof(int);
    return bytes+sizeof(TerminalObservation);
}
/** @return Completed snapshots' checked delta; failed/regressed input leaves the output untouched. */
inline bool terminalObservationDelta(const TerminalObservation& current,const TerminalObservation& previous,TerminalObservation& out)
{
    if(current.status!=TerminalObservationStatus::Complete || previous.status!=TerminalObservationStatus::Complete)return false;
    for(const auto* sample:{&current,&previous})
        if(sample->values[1]>sample->values[0] || sample->values[2]>sample->values[1] || sample->values[3]>sample->values[0])return false;
    TerminalObservation delta;
    for(unsigned i=0;i<15;++i){if(current.values[i]<previous.values[i])return false;delta.values[i]=current.values[i]-previous.values[i];}
    if(delta.values[1]>delta.values[0] || delta.values[2]>delta.values[1] || delta.values[3]>delta.values[0])return false;
    out=delta;return true;
}
/** @brief Original topology assigns receive-only physical traffic to exactly one participant. */
enum class TerminalReceiveRole { Coordinator, Reader };
/** @return Whether completed wire totals use the original native extents and fixed metadata categories. */
inline bool validTerminalWireDelta(const TerminalObservation& delta,TerminalReceiveRole role,std::size_t nativeRowBytes)
{
    if(!nativeRowBytes)return false;
    const auto keys=delta.values[10],rows=delta.values[11],headers=delta.values[12],ack=delta.values[13],admission=delta.values[14];
    if(role==TerminalReceiveRole::Coordinator)
        return !rows && !admission && keys%sizeof(std::uint64_t)==0 &&
            headers%sizeof(local_hot_transport::RequestHeader)==0 && ack%sizeof(local_hot_transport::ResultHeader)==0 &&
            headers/sizeof(local_hot_transport::RequestHeader)==ack/sizeof(local_hot_transport::ResultHeader);
    if(role==TerminalReceiveRole::Reader)
        return !keys && !ack && rows%nativeRowBytes==0 && headers%sizeof(local_hot_transport::ResultHeader)==0 &&
            admission%sizeof(local_hot_transport::SourceAdmissionHeader)==0;
    return false;
}
/** @brief Actual immutable completed-snapshot projection, never a GPU execution-state owner. */
struct TerminalPublication {TerminalObservation delta;bool since_storage_initialization=false;};
/**
 * @brief Storage-lifetime observation baseline survives replacement of its runner/consumer.
 * @details This mutex protects completed HOST observations only. No enqueue,
 * device read, wait or callback runs under it; it cannot block GPU progress.
 * Collector reset does not rewind lifetime totals or republish an old window.
 */
class TerminalObservationAccumulator final {
public:
    /** @brief Bind original cold receive role/native stride, which records cannot relabel. */
    TerminalObservationAccumulator(TerminalReceiveRole role,std::size_t rowBytes):role_(role),row_bytes_(rowBytes)
    {
        if(!rowBytes || (role!=TerminalReceiveRole::Coordinator && role!=TerminalReceiveRole::Reader))
            throw std::invalid_argument("Terminal observations require original receive role/native row extent");
    }
    /** @return Checked delta; stale/malformed input cannot mutate the last completed observation. */
    TerminalPublication observe(const TerminalObservation& current)
    {
        std::scoped_lock lock(mutex_);TerminalObservation delta;
        if(!terminalObservationDelta(current,previous_,delta) || !validTerminalWireDelta(delta,role_,row_bytes_))
            throw std::logic_error("Local terminal observation regressed, failed retirement or changed actual native wire extents");
        const bool initial=!seen_;previous_=current;seen_=true;return {delta,initial};
    }
private:
    TerminalReceiveRole role_;
    std::size_t row_bytes_;
    std::mutex mutex_;
    TerminalObservation previous_{};
    bool seen_=false;
};
/** @brief Stable field names match the native producer order, never inferred from a controller alias. */
inline constexpr std::array<const char*,15> TerminalMetricNames{
    "local_read_original_ids","local_hit_original_ids","local_all_hit_wait_avoided_original_ids","local_consumed_original_ids",
    "local_future_on_time_native_rows","local_future_late_native_rows","local_future_useful_native_rows","local_future_wasted_native_rows",
    "local_promotions","local_evictions","local_request_key_payload_bytes","local_native_result_payload_bytes",
    "local_lookup_control_header_bytes","local_acknowledgement_bytes","local_request_admission_bytes"};
} // namespace llaminar2::qwen4exp::local_hot
