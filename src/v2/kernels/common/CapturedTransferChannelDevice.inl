/**
 * @file CapturedTransferChannelDevice.inl
 * @brief Shared bounded CUDA/HIP fused kernels for retained byte lanes.
 *
 * Included by backend bridges which supply system-scope mapped atomics. One
 * lane handles protocol metadata while all lanes copy payload. Small messages
 * use one block; large messages use a separate acquire followed by a parallel
 * copy whose last completed block publishes. No block waits for another block
 * to become resident. CTA barriers and an acquire/release completion chain
 * carry all writers to one system-release publisher. Exact streams and private
 * phase claims protect replay
 * without host observation, allocation or additional physical cursor bytes.
 */

/** @return Fresh peer descriptor word after its release epoch has been acquired. */
__device__ __forceinline__ std::uint64_t capturedTransferPeerWord(const std::uint64_t *word)
{
#if defined(__CUDA_ARCH__)
    return __ldcv(reinterpret_cast<const unsigned long long *>(word));
#elif defined(__HIP_DEVICE_COMPILE__)
    return __builtin_nontemporal_load(word);
#else
    return *word;
#endif
}

/**
 * @return A fixed-rate device wall timer, independent of shader DVFS.
 * A stalled peer must fail at the declared deadline even while this GPU is
 * power-throttled; converting the maximum shader frequency would stretch it.
 */
__device__ __forceinline__ std::uint64_t capturedTransferWallClock()
{
#if defined(__CUDA_ARCH__)
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
#elif defined(__HIP_DEVICE_COMPILE__)
    return static_cast<std::uint64_t>(wall_clock64());
#else
    return 0;
#endif
}

/** @brief Publish a terminal peer-visible abort before trapping the failed device work. */
__device__ __forceinline__ void capturedTransferFail(
    CapturedTransferChannelDeviceBinding binding, CapturedTransferStatus status)
{
    binding.cursor->phase = CapturedTransferPhase::Failed;
    auto *publication = binding.role == CapturedTransferEndpoint::Producer
        ? &binding.control->producer : &binding.control->consumer;
    mappedSystemRelease64(&publication->epoch, kCapturedTransferAbortEpoch);
    printf("captured_transfer_channel_failure status=%u role=%u nonce=%llu key=%llu completed=%llu\n",
        static_cast<unsigned>(status), static_cast<unsigned>(binding.role),
        static_cast<unsigned long long>(binding.expected.nonce),
        static_cast<unsigned long long>(binding.message.key),
        static_cast<unsigned long long>(binding.cursor->completed_epoch));
#if defined(__CUDA_ARCH__)
    asm volatile("trap;");
#elif defined(__HIP_DEVICE_COMPILE__)
    __builtin_trap();
#endif
}

/**
 * @brief Acquire one next-generation lease and reject overlapping endpoint execution.
 * @param binding Complete graph-stable alias, message and timeout identity.
 *
 * Atomic phase ownership covers the whole acquire/copy/publication interval.
 * Peer waits use system-scope loads; a stale GPU cache line is never sufficient
 * proof. Only the consumer loads the descriptor, and only after acquiring its
 * exact new epoch. The timeout is per no-progress boundary, not per request.
 */
__device__ __forceinline__ void capturedTransferAcquire(CapturedTransferChannelDeviceBinding binding)
{
    auto *phase = reinterpret_cast<unsigned int *>(&binding.cursor->phase);
    if (atomicCAS(phase, static_cast<unsigned int>(CapturedTransferPhase::Idle),
            static_cast<unsigned int>(CapturedTransferPhase::Acquired)) !=
        static_cast<unsigned int>(CapturedTransferPhase::Idle))
    {
        capturedTransferFail(binding, CapturedTransferStatus::InvalidPhase);
        return;
    }
    auto cursor = *binding.cursor;
    // The atomic claim already excludes other invocations. The pure decision
    // sees its own pre-acquisition state, not a second live controller.
    cursor.phase = CapturedTransferPhase::Idle;
    const auto actual = binding.control->identity;
    const auto *peer = binding.role == CapturedTransferEndpoint::Producer
        ? &binding.control->consumer : &binding.control->producer;
    // The graph orders the count writer before acquire. Snapshot once: a
    // producer waiting for the previous acknowledgement must not change the
    // extent of the transaction it is trying to publish.
    const std::uint64_t producer_bytes = binding.extent_source == CapturedTransferExtentSource::ProducerDevice
        ? *binding.device_extent : 0;
    const auto started = capturedTransferWallClock();
    for (;;)
    {
        CapturedTransferPublication observed;
        observed.epoch = mappedSystemAcquire64(&peer->epoch);
        if (binding.role == CapturedTransferEndpoint::Consumer &&
            observed.epoch != kCapturedTransferAbortEpoch && observed.epoch == cursor.completed_epoch + 1)
        {
            observed.message.key = capturedTransferPeerWord(&peer->message.key);
            observed.message.bytes = capturedTransferPeerWord(&peer->message.bytes);
        }
        const auto decision = CapturedTransferChannelProtocol::acquire(
            binding.role, binding.expected, actual, cursor, observed, binding.message,
            binding.extent_source, producer_bytes);
        if (decision.ready())
        {
            binding.cursor->acquired_message = cursor.acquired_message;
            // Only this acquired invocation can initialize the completion
            // cohort. All copy blocks follow this write through the stream
            // dependency or the single-block barrier.
            binding.cursor->completed_payload_blocks = 0;
            if (binding.extent_source == CapturedTransferExtentSource::ProducerPublication)
                *binding.device_extent = cursor.acquired_message.bytes;
            return;
        }
        if (decision.status != CapturedTransferStatus::WaitForPeer)
        {
            capturedTransferFail(binding, decision.status);
            return;
        }
        if (capturedTransferWallClock() - started >= binding.timeout_ticks)
        {
            capturedTransferFail(binding, CapturedTransferStatus::TimedOut);
            return;
        }
#if defined(__CUDA_ARCH__)
        __nanosleep(64u);
#elif defined(__HIP_DEVICE_COMPILE__)
        __builtin_amdgcn_s_sleep(1u);
#endif
    }
}

/**
 * @brief Release the exact message after acquiring every payload writer.
 * @param binding Same immutable endpoint/message as the acquire operation.
 *
 * Publishing is a distinct transient phase so a duplicate release cannot
 * race the successful publisher. Descriptor stores precede the system-release
 * epoch; the local Idle store is last so the next replay cannot acquire a
 * half-published cursor. No other endpoint's publication is ever overwritten.
 */
__device__ __forceinline__ void capturedTransferPublish(CapturedTransferChannelDeviceBinding binding)
{
    auto *phase = reinterpret_cast<unsigned int *>(&binding.cursor->phase);
    if (atomicCAS(phase, static_cast<unsigned int>(CapturedTransferPhase::Acquired),
            static_cast<unsigned int>(CapturedTransferPhase::Publishing)) !=
        static_cast<unsigned int>(CapturedTransferPhase::Acquired))
    {
        capturedTransferFail(binding, CapturedTransferStatus::InvalidPhase);
        return;
    }
    auto cursor = *binding.cursor;
    cursor.phase = CapturedTransferPhase::Acquired;
    if (cursor.acquired_message.key != binding.message.key ||
        (binding.extent_source == CapturedTransferExtentSource::FixedMessage
            ? cursor.acquired_message.bytes != binding.message.bytes : cursor.acquired_message.bytes > binding.message.bytes))
    {
        capturedTransferFail(binding, CapturedTransferStatus::MessageMismatch);
        return;
    }
    auto *owned = binding.role == CapturedTransferEndpoint::Producer
        ? &binding.control->producer : &binding.control->consumer;
    CapturedTransferPublication publication{.epoch = mappedSystemAcquire64(&owned->epoch)};
    const auto decision = CapturedTransferChannelProtocol::publish(cursor, cursor.completed_epoch + 1, publication);
    if (!decision.ready())
    {
        capturedTransferFail(binding, decision.status);
        return;
    }
    owned->message = publication.message;
    mappedSystemRelease64(&owned->epoch, publication.epoch);
    binding.cursor->completed_epoch = cursor.completed_epoch;
    binding.cursor->acquired_message = {};
    __threadfence();
    atomicExch(phase, static_cast<unsigned int>(CapturedTransferPhase::Idle));
}

/**
 * @brief Copy the acquired extent and let the last completed block publish it.
 * @tparam VectorAligned Whether both immutable addresses are sixteen-byte aligned.
 * @param binding Acquired private cursor and complete message identity.
 * @param destination Stable mapped or device destination.
 * @param source Stable device or mapped source.
 *
 * A CTA barrier orders every writer before its block's release/acquire RMW.
 * The RMW modification order chains all blocks to the last one, which performs
 * the existing system-release publication. That transitive happens-before
 * chain covers the payload without a system fence in every lane. A relaxed
 * atomic counter would not suffice. CUDA's PTX causality-order contract and
 * HIP's workgroup barrier/agent acquire-release semantics provide these edges.
 *
 * This is a completion counter, not a grid barrier: finished blocks retire
 * immediately, leaving resources for unscheduled blocks. The consumer follows
 * the same rule before acknowledging, so a producer cannot overwrite a slot
 * while any consumer lane still reads it. Empty extents still release once.
 */
template<bool VectorAligned>
__device__ __forceinline__ void capturedTransferCopyAndPublish(
    CapturedTransferChannelDeviceBinding binding, void *destination, const void *source)
{
    const auto bytes = binding.cursor->acquired_message.bytes;
    if (binding.cursor->phase != CapturedTransferPhase::Acquired || bytes > binding.message.bytes ||
        binding.cursor->acquired_message.key != binding.message.key)
    {
        if (threadIdx.x == 0) capturedTransferFail(binding, CapturedTransferStatus::MessageMismatch);
        return;
    }
    if constexpr (VectorAligned)
        mappedHostCopyVectorBody(static_cast<uint4 *>(destination), static_cast<const uint4 *>(source), bytes);
    else
        mappedHostCopyByteBody(static_cast<std::uint8_t *>(destination), static_cast<const std::uint8_t *>(source), bytes);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        // The single-block case needs no counter. In the multi-block case,
        // acquire/release tickets both identify the last *completed* copy
        // block and acquire all previous blocks' payload accesses. The final
        // peer-visible release belongs to that one publisher, not each lane.
        const auto completed = gridDim.x == 1 ? 0u
            : capturedTransferJoinBlock(&binding.cursor->completed_payload_blocks);
        if (completed >= gridDim.x)
            capturedTransferFail(binding, CapturedTransferStatus::InvalidPhase);
        else if (completed + 1u == gridDim.x)
            capturedTransferPublish(binding);
    }
}

/** @brief One-wave protocol acquire preceding a large parallel copy.
 * @param binding Complete immutable message and private cursor. */
__global__ void capturedTransferAcquireKernel(CapturedTransferChannelDeviceBinding binding)
{
    if (threadIdx.x == 0) capturedTransferAcquire(binding);
}

/** @brief Large-message parallel copy with completion-owned publication.
 * @tparam VectorAligned Capture-frozen source/destination alignment.
 * @param binding Stream-ordered acquired contract.
 * @param destination Stable destination alias.
 * @param source Stable source alias. */
template<bool VectorAligned>
__global__ __launch_bounds__(256) void capturedTransferCopyPublishKernel(
    CapturedTransferChannelDeviceBinding binding, void *destination, const void *source)
{
    capturedTransferCopyAndPublish<VectorAligned>(binding, destination, source);
}

/** @brief Complete bounded message in one cooperative block, without a host boundary.
 * @tparam VectorAligned Capture-frozen source/destination alignment.
 * @param binding Complete protocol identity and extent authority.
 * @param destination Stable destination alias.
 * @param source Stable source alias. */
template<bool VectorAligned>
__global__ __launch_bounds__(256) void capturedTransferFusedKernel(
    CapturedTransferChannelDeviceBinding binding, void *destination, const void *source)
{
    if (threadIdx.x == 0) capturedTransferAcquire(binding);
    __syncthreads();
    capturedTransferCopyAndPublish<VectorAligned>(binding, destination, source);
}
