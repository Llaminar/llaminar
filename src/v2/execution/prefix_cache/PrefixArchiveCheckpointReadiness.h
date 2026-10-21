/**
 * @file PrefixArchiveCheckpointReadiness.h
 * @brief Bind one exact completion owner for diagnostic auxiliary archive handles.
 *
 * Recurrent/terminal snapshots borrow a preadmitted checkpoint event. An
 * auxiliary-only main or shifted snapshot instead prepares its event through
 * the existing archive authority before copies begin. Both readiness APIs
 * retain the same native event; no second allocator or completion state exists.
 * Ordinary retained logical checkpoints do not use this diagnostic helper.
 */
#pragma once
#include "PrefixStorageBackend.h"
namespace llaminar2
{
/**
 * @brief Prepare an auxiliary full archive before its first possible submission.
 * @param handle Diagnostic main or shifted handle with admitted auxiliary storage.
 * @param device Exact producer participant.
 * @param stream Exact non-null producer stream on GPU.
 * @param backend Exact backend owning the event and stream.
 * @param prepare_archive Existing archive authority, called only without a slot event.
 * @return Whether both handle APIs now retain the same prepared event, or no GPU auxiliary work exists.
 */
template<class PrepareArchive>
[[nodiscard]] bool preparePrefixArchiveCheckpointReadiness(
    PrefixBlockHandle &handle,
    DeviceId device,
    void *stream,
    IBackend &backend,
    PrepareArchive &&prepare_archive)
{
    if (!device.is_gpu() || handle.layout.modelAuxiliaryBytes() == 0)
        return true;
    if (handle.live_checkpoint_ready_event)
    {
        return handle.payload_readiness && handle.payload_readiness->prepare(
            handle.live_checkpoint_ready_event, device, stream, backend);
    }
    if (!prepare_archive() || !handle.payload_readiness ||
        !handle.payload_readiness->prepared() ||
        handle.payload_readiness->producerDevice() != device ||
        handle.payload_readiness->producerStream() != stream)
        return false;
    handle.live_checkpoint_ready_event = std::shared_ptr<void>(
        handle.payload_readiness, handle.payload_readiness->event());
    return true;
}
} // namespace llaminar2
