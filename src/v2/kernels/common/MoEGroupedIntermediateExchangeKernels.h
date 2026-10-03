/**
 * @file MoEGroupedIntermediateExchangeKernels.h
 * @brief Explicit-stream GPU launches for the lossless expert phase boundary.
 *
 * These low-level bridges consume persistent arena/TransferEngine bindings.
 * They perform no allocation, synchronization, transport or format conversion.
 * The graph collective owns ordering between pack and consume; source and
 * destination buffers must have distinct arena ownership until completion.
 */
#pragma once

#include "execution/moe/MoEGroupedIntermediateExchangeABI.h"

namespace llaminar2
{
#ifdef HAVE_CUDA
    namespace cuda
    {
        /** @brief Pack existing activation bytes; reject invalid/default streams. */
        bool packGroupedIntermediate(const MoEGroupedIntermediatePackLaunch &launch, void *stream);
        /** @brief Consume the native collective's completed immutable packet bank. */
        bool consumeGroupedIntermediate(const MoEGroupedIntermediateConsumeLaunch &launch, void *stream);
        /** @brief Pack live rows and their route IDs; publish a GPU byte count, without zero padding. */
        bool packCompactIntermediate(const MoECompactIntermediatePackLaunch &launch, void *stream);
        /** @brief Import one acquired compact packet; unused capacity is never read. */
        bool consumeCompactIntermediate(const MoECompactIntermediateConsumeLaunch &launch, void *stream);
    }
#endif
#ifdef HAVE_ROCM
    namespace rocm
    {
        /** @brief Pack existing activation bytes; reject invalid/default streams. */
        bool packGroupedIntermediate(const MoEGroupedIntermediatePackLaunch &launch, void *stream);
        /** @brief Consume the native collective's completed immutable packet bank. */
        bool consumeGroupedIntermediate(const MoEGroupedIntermediateConsumeLaunch &launch, void *stream);
        /** @brief Symmetric HIP compact pack with the same representation and count contract. */
        bool packCompactIntermediate(const MoECompactIntermediatePackLaunch &launch, void *stream);
        /** @brief Symmetric HIP import of only the release-published live byte range. */
        bool consumeCompactIntermediate(const MoECompactIntermediateConsumeLaunch &launch, void *stream);
    }
#endif
} // namespace llaminar2
