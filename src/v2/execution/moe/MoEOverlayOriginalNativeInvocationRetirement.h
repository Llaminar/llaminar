/**
 * @file MoEOverlayOriginalNativeInvocationRetirement.h
 * @brief Aggregate native observation permit for clearing one original slot's metadata.
 *
 * Only the original participant runner may issue this value, after observing
 * ALL original native receipts in ordinal order at the existing source-owned
 * aggregate boundary. The permit does not infer physical success from logical
 * Submitted or packet Complete and cannot be constructed by graph stages.
 */
#pragma once
#include "MoEOverlayInferenceOriginalSlot.h"
namespace llaminar2 {
class MoEOverlayParticipantGraphRunner;
/** @brief Original slot identity coupled to its runner's completed aggregate proof. */
class MoEOverlayOriginalNativeInvocationRetirement final {
    friend class MoEOverlayParticipantGraphRunner;
public:
    /** @return Same checked original slot whose physical aggregate was observed. */
    const MoEOverlayInferenceOriginalSlot& originalSlot()const noexcept{return original_;}
private:
    /** @brief Issued only inside the real runner after all original physical receipts. */
    explicit MoEOverlayOriginalNativeInvocationRetirement(MoEOverlayInferenceOriginalSlot original)
        :original_(std::move(original)){}
    MoEOverlayInferenceOriginalSlot original_;
};
} // namespace llaminar2
