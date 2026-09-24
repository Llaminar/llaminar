/**
 * @file BackgroundTransferProgressBinding.h
 * @brief Explicit native-stream or shared-service authority for a transfer lane.
 *
 * Progress policy is selected during topology construction, never after a copy
 * stalls. CUDA inference-bound clients use the shared graph service; native
 * stream execution remains the explicit contract for standalone transfer work
 * and backends whose native queues independently progress during inference.
 * A failed/missing graph service cannot silently select native submission.
 */
#pragma once

#include <memory>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    class MappedTransferProgressEpoch;

    /** Immutable setup choice retained by every operation on a physical lane. */
    class BackgroundTransferProgressBinding final
    {
    public:
        BackgroundTransferProgressBinding() = delete;

        /** Select exact native-stream completion before admitting any work. */
        [[nodiscard]] static BackgroundTransferProgressBinding nativeStream()
        { return BackgroundTransferProgressBinding(std::shared_ptr<MappedTransferProgressEpoch>{}); }

        /** Retain the sole graph progress authority; null is an invalid policy. */
        [[nodiscard]] static BackgroundTransferProgressBinding graphService(
            std::shared_ptr<MappedTransferProgressEpoch> epoch)
        {
            if (!epoch) throw std::invalid_argument("Graph transfer progress requires its shared authority");
            return BackgroundTransferProgressBinding(std::move(epoch));
        }

        /** @return Bound graph authority, or empty for explicitly native work. */
        [[nodiscard]] const std::shared_ptr<MappedTransferProgressEpoch> &epoch() const noexcept
        { return epoch_; }

    private:
        /** Store one validated immutable authority choice. */
        explicit BackgroundTransferProgressBinding(std::shared_ptr<MappedTransferProgressEpoch> epoch)
            : epoch_(std::move(epoch)) {}
        std::shared_ptr<MappedTransferProgressEpoch> epoch_;
    };
} // namespace llaminar2
