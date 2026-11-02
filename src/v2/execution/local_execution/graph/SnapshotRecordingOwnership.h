/**
 * @file SnapshotRecordingOwnership.h
 * @brief Exactly one owner records immutable diagnostic copies at an ordering boundary.
 *
 * A producer completing a native event must include its snapshot copy before
 * that event. Manual/collective controller boundaries retain their original
 * owner; selecting a side lane never transfers ownership of unrelated stages.
 */
#pragma once
#include <cstdint>
#include <stdexcept>
namespace llaminar2 {
/** @brief Sole recorder of an invocation's point-in-time copy. */
enum class StageSnapshotRecordingAuthority : std::uint8_t {
    StageExecutor, GraphCaptureController
};
/** @brief Actual invocation boundary, independent of the whole graph's lanes. */
enum class SnapshotPublicationBoundary : std::uint8_t {
    ControllerStage, NativeEventClosedProducer
};
/** @brief Resolve the sole copy owner before stage execution.
 * @param original Existing invocation owner, preserved outside event-closed producers.
 * @param boundary Actual producer closure whose completion makes native readers safe.
 * @return Stage owner for event closure; original owner for manual/controller closure.
 */
constexpr StageSnapshotRecordingAuthority snapshotRecordingAuthorityForBoundary(
    StageSnapshotRecordingAuthority original,SnapshotPublicationBoundary boundary)
{
    if(original!=StageSnapshotRecordingAuthority::StageExecutor && original!=StageSnapshotRecordingAuthority::GraphCaptureController)
        throw std::invalid_argument("Unknown snapshot recording authority");
    switch(boundary){
    case SnapshotPublicationBoundary::ControllerStage:return original;
    case SnapshotPublicationBoundary::NativeEventClosedProducer:return StageSnapshotRecordingAuthority::StageExecutor;
    }
    throw std::invalid_argument("Unknown snapshot publication boundary");
}
} // namespace llaminar2
