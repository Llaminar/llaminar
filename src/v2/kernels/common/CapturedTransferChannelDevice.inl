/**
 * @file CapturedTransferChannelDevice.inl
 * @brief Shared bounded CUDA/HIP acquire/release kernels for retained byte lanes.
 *
 * Included by backend bridges which supply system-scope mapped atomics. One
 * lane handles only protocol metadata; the intervening payload copy uses the
 * existing parallel mapped-copy kernel. All three nodes execute on the same
 * exact stream, without host observation or per-replay allocation. A private
 * phase claim rejects overlapping use of an endpoint by two graph instances.
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
__global__ void capturedTransferAcquireKernel(CapturedTransferChannelDeviceBinding binding)
{
    if (threadIdx.x != 0) return;
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
            binding.role, binding.expected, actual, cursor, observed, binding.message);
        if (decision.ready())
        {
            binding.cursor->acquired_message = cursor.acquired_message;
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
 * @brief Release the exact message only after the preceding payload kernel completes.
 * @param binding Same immutable endpoint/message as the acquire node.
 *
 * Publishing is a distinct transient phase so a duplicate release cannot
 * race the successful publisher. Descriptor stores precede the system-release
 * epoch; the local Idle store is last so the next replay cannot acquire a
 * half-published cursor. No other endpoint's publication is ever overwritten.
 */
__global__ void capturedTransferPublishKernel(CapturedTransferChannelDeviceBinding binding)
{
    if (threadIdx.x != 0) return;
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
    if (cursor.acquired_message.key != binding.message.key || cursor.acquired_message.bytes != binding.message.bytes)
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
