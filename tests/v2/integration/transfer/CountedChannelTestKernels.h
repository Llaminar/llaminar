/**
 * @file CountedChannelTestKernels.h
 * @brief Device-authored changing payloads for retained-transfer regression tests.
 *
 * One GPU sequence selects empty, full, odd and shrinking extents. The host
 * computes an expected answer only after replay; it never supplies the count,
 * changes a graph parameter, or selects a different captured transaction.
 */
#pragma once
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_CHANNEL_TEST_HD __host__ __device__
#else
#define LLAMINAR_CHANNEL_TEST_HD
#endif

namespace llaminar2::counted_channel_test
{
    /** @return Adversarial live extent; the fixture supplies capacity >= 257. */
    LLAMINAR_CHANNEL_TEST_HD inline std::uint64_t extent(std::uint64_t sequence, std::uint64_t capacity)
    {
        switch (sequence % 8)
        {
        case 0: return capacity;
        case 1: return 0;
        case 2: return 1;
        case 3: return capacity / 2 + 3;
        case 4: return 16;
        case 5: return capacity - 1;
        case 6: return 0;
        default: return 257;
        }
    }

    /** @return Position/sequence-sensitive raw bytes, independent of arithmetic precision. */
    LLAMINAR_CHANNEL_TEST_HD inline std::uint8_t byte(std::uint64_t sequence, std::size_t position)
    {
        auto value = static_cast<std::uint32_t>(position * 0x9e3779b9u + sequence * 0xb7e15162u);
        value ^= value >> 16;
        value *= 0x85ebca6bu;
        return static_cast<std::uint8_t>(value ^ (value >> 13));
    }

#ifdef HAVE_CUDA
    /** @brief Resolve fixture functions before recording, with no lazy module work in replay. */
    bool prepareKernelsCUDA();
    /** @brief Fill a persistent payload and publish its next count on the exact stream. */
    bool prepareCUDA(void *payload, std::uint64_t *count, const std::uint64_t *sequence, std::size_t capacity, void *stream);
    /** @brief Advance only after the complete captured round trip has consumed this sequence. */
    bool advanceCUDA(std::uint64_t *sequence, void *stream);
#endif
#ifdef HAVE_ROCM
    /** @brief Symmetric HIP function preparation outside capture. */
    bool prepareKernelsROCm();
    /** @brief Symmetric HIP payload/count writer, without host-selected live state. */
    bool prepareROCm(void *payload, std::uint64_t *count, const std::uint64_t *sequence, std::size_t capacity, void *stream);
    /** @brief Advance the HIP producer after its same-stream round trip. */
    bool advanceROCm(std::uint64_t *sequence, void *stream);
#endif
}
#undef LLAMINAR_CHANNEL_TEST_HD
