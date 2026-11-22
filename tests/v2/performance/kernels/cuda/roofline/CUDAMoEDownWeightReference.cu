/**
 * @file CUDAMoEDownWeightReference.cu
 * @brief Weight-only CUDA reference using original prepared MoE pointers, grid and route order.
 */
#include "CUDAMoEDownWeightReference.h"
#include "kernels/common/DeviceNativeQuantisedMatrixDesc.h"
#include "tensors/NativeQuantisedFormatInfo.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace llaminar2::test
{
    namespace
    {
        /**
         * @brief Read a single physical payload and consume every byte in the checksum.
         *
         * Naturally aligned packets use the same sized memory transactions as the
         * source layout permits. The scalar oracle reconstructs the same unsigned
         * words byte by byte and is checked before the vector path is timed.
         * @tparam Bytes Exact physical payload extent of a K32 packet.
         * @tparam Scalar Use the independent byte-wise checksum implementation.
         * @param payload Original address within the prepared weight allocation.
         * @return Modular sum of every payload byte at its original word offset.
         * @pre The prepared packet plane is aligned and spans groups of four columns.
         */
        template <int Bytes, bool Scalar>
        __device__ __forceinline__ uint32_t payloadChecksum(const uint8_t *payload)
        {
            uint32_t sum = 0;
            if constexpr (!Scalar && Bytes % 16 == 0)
            {
                const auto *vectors = reinterpret_cast<const uint4 *>(payload);
#pragma unroll
                for (int i = 0; i < Bytes / 16; ++i)
                {
                    const uint4 words = vectors[i];
                    sum += (words.x + words.y) + (words.z + words.w);
                }
            }
            else if constexpr (!Scalar && Bytes % 8 == 0)
            {
                const auto *vectors = reinterpret_cast<const uint2 *>(payload);
#pragma unroll
                for (int i = 0; i < Bytes / 8; ++i)
                {
                    const uint2 words = vectors[i];
                    sum += words.x + words.y;
                }
            }
            else if constexpr (!Scalar && Bytes % 4 == 0)
            {
                const auto *words = reinterpret_cast<const uint32_t *>(payload);
#pragma unroll
                for (int i = 0; i < Bytes / 4; ++i)
                    sum += words[i];
            }
            else if constexpr (!Scalar && Bytes % 2 == 0)
            {
                const auto *halves = reinterpret_cast<const uint16_t *>(payload);
#pragma unroll
                for (int i = 0; i < Bytes / 2; ++i)
                    sum += static_cast<uint32_t>(halves[i]) << (16 * (i & 1));
            }
            else if constexpr (!Scalar && Bytes % 4 == 1)
            {
                // For 9- and 13-byte packets, reassemble the neighboring
                // aligned words and consume only this packet's bytes. The
                // first and last loads stay inside the prepared plane.
                const unsigned offset = reinterpret_cast<uintptr_t>(payload) & 3u;
                const auto *words = reinterpret_cast<const uint32_t *>(payload - offset);
                const unsigned shift = offset * 8;
#pragma unroll
                for (int i = 0; i < Bytes / 4; ++i)
                    sum += __funnelshift_r(words[i], words[i + 1], shift);
                sum += (words[Bytes / 4] >> shift) & 0xffu;
            }
            else
            {
#pragma unroll
                for (int i = 0; i < Bytes; ++i)
                    sum += static_cast<uint32_t>(payload[i]) << (8 * (i & 3));
            }
            return sum;
        }

        /**
         * @brief Resolve the physical packet and coefficient layout before compilation.
         * @tparam CodebookId Physical execution identifier of the prepared weight.
         * @return The native plane layout shared by the fixture and its reader.
         */
        template <uint8_t CodebookId>
        consteval NativeQuantisedMigrationStableDeviceFormat weightLayout()
        {
            if constexpr (CodebookId == kNativeQuantisedExpandedInt8MinCodebook)
                return {CodebookId, 32, true, false};
            else
            {
                for (const auto &source : native_quantised_formats::kAllSourceFormats)
                {
                    const auto layout = canonicalDeviceVnniFormat(*source.metadata);
                    if (layout.codebook_id == CodebookId)
                        return layout;
                }
                return {};
            }
        }

        /**
         * @brief Read the same selected output column and K32 planes as the Down leaf.
         *
         * The arguments are the actual captured production arguments. Activation
         * and route-weight reads and dot arithmetic are omitted. The publication
         * address and original-slot traversal are preserved.
         * @tparam CodebookId Exact physical decoder of the fixture's weights.
         * @tparam Scalar Use the independent byte-wise checksum implementation.
         * @param activation_codes Captured argument. The reader leaves it unread.
         * @param activation_scales Captured argument. The reader leaves it unread.
         * @param descriptors Original prepared Down descriptor table.
         * @param original_to_grouped Active-route mapping from the captured plan.
         * @param original_experts Original route packet in submitted slot order.
         * @param grouped_weights Captured argument. The reader leaves it unread.
         * @param output Captured publication buffer used for checksums.
         * @param first_slot First original route in the selected launch.
         * @param N Output columns in each prepared packet plane.
         * @param K Reduction width in individual elements.
         * @param num_experts Exact descriptor table length.
         */
        template <uint8_t CodebookId, bool Scalar>
        __global__ void moeVerifierWeightPacketRead(
            const int8_t * /*activation_codes*/, const float * /*activation_scales*/,
            const DeviceNativeQuantisedMatrixDesc *__restrict__ descriptors,
            const int *__restrict__ original_to_grouped,
            const int *__restrict__ original_experts,
            const float * /*grouped_weights*/,
            float *__restrict__ output,
            int first_slot, int N, int K, int num_experts)
        {
            constexpr auto layout = weightLayout<CodebookId>();
            static_assert(layout.payload_bytes_per_block > 0);
            const int n = blockIdx.x * blockDim.x + threadIdx.x;
            if (n >= N)
                return;
            const int slot = first_slot + blockIdx.y;
            uint32_t sum = 0;
            const int expert = original_experts[slot];
            if (original_to_grouped[slot] >= 0 && expert >= 0 && expert < num_experts)
            {
                const auto &desc = descriptors[expert];
                const auto *weights = desc.payload;
                const auto *scales = static_cast<const uint16_t *>(desc.scales);
                const auto *secondary = static_cast<const uint16_t *>(desc.mins);
                const auto *extra = static_cast<const uint32_t *>(desc.emins);
#pragma unroll 1
                for (int packet = 0; packet < K / 32; ++packet)
                {
                    const auto index = static_cast<size_t>(packet) * N + n;
                    sum += payloadChecksum<layout.payload_bytes_per_block, Scalar>(
                        weights + index * layout.payload_bytes_per_block);
                    sum += scales[index];
                    if constexpr (layout.is_asymmetric)
                        sum += secondary[index];
                    if constexpr (layout.has_emins)
                        sum += extra[index];
                }
            }
            output[static_cast<size_t>(slot) * N + n] = __uint_as_float(sum);
        }

        /**
         * @brief Choose the checksum implementation before recording the native node.
         * @tparam Codebook Exact physical format used by the production fixture.
         * @param access Requested packet access pattern.
         * @return CUDA entry with the original Down argument ABI.
         */
        template <uint8_t Codebook>
        const void *weightPacketReader(CUDAWeightPacketAccess access) noexcept
        {
            return access == CUDAWeightPacketAccess::ByteOracle
                ? reinterpret_cast<const void *>(moeVerifierWeightPacketRead<Codebook, true>)
                : reinterpret_cast<const void *>(moeVerifierWeightPacketRead<Codebook, false>);
        }
    }

    const void *cudaMoEDownWeightReferenceEntry(uint8_t codebook, CUDAWeightPacketAccess access) noexcept
    {
        switch (codebook)
        {
        case 0:  return weightPacketReader<0>(access);
        case 4:  return weightPacketReader<4>(access);
        case 5:  return weightPacketReader<5>(access);
        case 6:  return weightPacketReader<6>(access);
        case 7:  return weightPacketReader<7>(access);
        case 8:  return weightPacketReader<8>(access);
        case 9:  return weightPacketReader<9>(access);
        case 10: return weightPacketReader<10>(access);
        case 11: return weightPacketReader<11>(access);
        case 12: return weightPacketReader<12>(access);
        case 13: return weightPacketReader<13>(access);
        case 14: return weightPacketReader<14>(access);
        case 15: return weightPacketReader<15>(access);
        case 16: return weightPacketReader<16>(access);
        case 17: return weightPacketReader<17>(access);
        case 19: return weightPacketReader<19>(access);
        case kNativeQuantisedExpandedInt8MinCodebook:
            return weightPacketReader<kNativeQuantisedExpandedInt8MinCodebook>(access);
        case kNativeQuantisedIQ4XSSourceScaleCodebook:
            return weightPacketReader<kNativeQuantisedIQ4XSSourceScaleCodebook>(access);
        default: return nullptr;
        }
    }
} // namespace llaminar2::test
