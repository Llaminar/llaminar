/**
 * @file CountedChannelTestKernels.inl
 * @brief Shared arithmetic-free CUDA/HIP sequence writer used only by tests.
 *
 * The fill node reads one unchanged sequence on every thread; a separate
 * downstream advance node changes it only after the transfer has completed.
 * This makes count changes genuinely device-driven without a grid barrier.
 */
namespace
{
    /** @brief Fill bounded payload bytes and the producer's authoritative extent.
     * @param payload Stable device payload capacity.
     * @param count Eight-byte device scalar, written by one lane.
     * @param sequence Device sequence unchanged until the later advance node.
     * @param capacity Immutable capacity of the fixture's retained graph. */
    __global__ __launch_bounds__(128) void prepareCountedPayload(
        std::uint8_t *payload, std::uint64_t *count, const std::uint64_t *sequence, std::size_t capacity)
    {
        const auto epoch = *sequence;
        const auto lane = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (lane == 0) *count = llaminar2::counted_channel_test::extent(epoch, capacity);
        for (auto i = lane; i < capacity; i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
            payload[i] = llaminar2::counted_channel_test::byte(epoch, i);
    }

    /** @brief One captured terminal update for the next replay's input generation. */
    __global__ void advanceCountedSequence(std::uint64_t *sequence)
    { if (threadIdx.x == 0) ++*sequence; }
}
