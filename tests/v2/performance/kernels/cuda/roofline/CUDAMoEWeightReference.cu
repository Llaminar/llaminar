/**
 * @file CUDAMoEWeightReference.cu
 * @brief Weight-only CUDA reference using original prepared MoE pointers, grid and route order.
 */
#include "CUDAMoEWeightReference.h"
#include "kernels/common/DeviceNativeQuantisedMatrixDesc.h"
#include "kernels/common/MoEGroupedSourceRows.h"
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
         * @brief Accumulate one prepared packet and its physical coefficients.
         * @tparam CodebookId Exact physical format of the prepared descriptor.
         * @tparam Scalar Use the independent byte-wise checksum implementation.
         * @param desc Original descriptor owning the weight planes.
         * @param index Packet index in the original [K32][column] layout.
         * @return Modular checksum contribution from every original plane.
         */
        template <uint8_t CodebookId, bool Scalar>
        __device__ __forceinline__ uint32_t weightPacketChecksum(
            const DeviceNativeQuantisedMatrixDesc &desc, size_t index)
        {
            constexpr auto layout = weightLayout<CodebookId>();
            static_assert(layout.payload_bytes_per_block > 0);
            uint32_t sum = payloadChecksum<layout.payload_bytes_per_block, Scalar>(
                desc.payload + index * layout.payload_bytes_per_block);
            sum += static_cast<const uint16_t *>(desc.scales)[index];
            if constexpr (layout.is_asymmetric)
                sum += static_cast<const uint16_t *>(desc.mins)[index];
            if constexpr (layout.has_emins)
                sum += static_cast<const uint32_t *>(desc.emins)[index];
            return sum;
        }

        /**
         * @brief Traverse one complete Down column in its original packet order.
         * @tparam CodebookId Exact physical format of the prepared descriptor.
         * @tparam Scalar Use the independent byte-wise checksum implementation.
         * @param desc Original prepared weight descriptor.
         * @param n Output column.
         * @param N Prepared column pitch.
         * @param K Reduction width in activation elements.
         * @return Checksum of the complete projection column.
         */
        template <uint8_t CodebookId, bool Scalar>
        __device__ __forceinline__ uint32_t downColumnChecksum(
            const DeviceNativeQuantisedMatrixDesc &desc, int n, int N, int K)
        {
            uint32_t sum = 0;
#pragma unroll 1
            for (int packet = 0; packet < K / 32; ++packet)
                sum += weightPacketChecksum<CodebookId, Scalar>(
                    desc, static_cast<size_t>(packet) * N + n);
            return sum;
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
            const int n = blockIdx.x * blockDim.x + threadIdx.x;
            if (n >= N)
                return;
            const int slot = first_slot + blockIdx.y;
            uint32_t sum = 0;
            const int expert = original_experts[slot];
            if (original_to_grouped[slot] >= 0 && expert >= 0 && expert < num_experts)
                sum = downColumnChecksum<CodebookId, Scalar>(descriptors[expert], n, N, K);
            output[static_cast<size_t>(slot) * N + n] = __uint_as_float(sum);
        }

        /**
         * @brief Read the original direct decode Down packets, with one result per route column.
         * @tparam CodebookId Exact physical decoder.
         * @tparam Scalar Use byte-wise checksum loads.
         * @param activation_codes Captured argument. The reader leaves it unread.
         * @param activation_scales Captured argument. The reader leaves it unread.
         * @param descriptors Original Down descriptor table.
         * @param expert_ids Original decode route IDs.
         * @param weights Captured argument. The reader leaves it unread.
         * @param output Original route publication buffer used for checksums.
         * @param N Prepared column pitch.
         * @param K Reduction width in activation elements.
         * @param num_experts Exact descriptor table length.
         */
        template <uint8_t CodebookId, bool Scalar>
        __global__ void moeDecodeDownWeightPacketRead(
            const int8_t * /*activation_codes*/, const float * /*activation_scales*/,
            const DeviceNativeQuantisedMatrixDesc *__restrict__ descriptors,
            const int *__restrict__ expert_ids, const float * /*weights*/,
            float *__restrict__ output, int N, int K, int num_experts)
        {
            const int n = blockIdx.x * blockDim.x + threadIdx.x;
            if (n >= N)
                return;
            const int slot = blockIdx.y;
            const int expert = expert_ids[slot];
            const uint32_t sum = expert >= 0 && expert < num_experts
                ? downColumnChecksum<CodebookId, Scalar>(descriptors[expert], n, N, K)
                : 0u;
            output[static_cast<size_t>(slot) * N + n] = __uint_as_float(sum);
        }

        /**
         * @brief Read both Gate and Up planes for one admitted route partition.
         * @tparam CodebookId Exact shared physical decoder of the weight tables.
         * @tparam Scalar Use byte-wise checksum loads.
         * @param gate_descs Original Gate descriptors.
         * @param up_descs Original Up descriptors.
         * @param expert Selected local expert descriptor index.
         * @param n Live output column.
         * @param N Prepared column pitch.
         * @param K Reduction width in activation elements.
         * @param partition Current split-K partition.
         * @param k_partitions Number of ordered partitions.
         * @param gate Checksum for the Gate partial store.
         * @param up Checksum for the Up partial store.
         */
        template <uint8_t CodebookId, bool Scalar>
        __device__ __forceinline__ void gateUpWeightChecksums(
            const DeviceNativeQuantisedMatrixDesc *gate_descs,
            const DeviceNativeQuantisedMatrixDesc *up_descs, int expert, int n, int N, int K,
            int partition, int k_partitions, uint32_t &gate, uint32_t &up)
        {
            const auto &gate_desc = gate_descs[expert];
            const auto &up_desc = up_descs[expert];
            const int packets_per_row = K / 32;
            const int packets_per_part = (packets_per_row + k_partitions - 1) / k_partitions;
            const int end = min((partition + 1) * packets_per_part, packets_per_row);
#pragma unroll 1
            for (int packet = partition * packets_per_part; packet < end; ++packet)
            {
                const auto index = static_cast<size_t>(packet) * N + n;
                gate += weightPacketChecksum<CodebookId, Scalar>(gate_desc, index);
                up += weightPacketChecksum<CodebookId, Scalar>(up_desc, index);
            }
        }

        /**
         * @brief Read both Gate and Up planes in the original verifier partition grid.
         * @tparam CodebookId Exact shared physical decoder of the fixture projections.
         * @tparam Scalar Use byte-wise checksum loads.
         * @param activation_codes Captured argument. The reader leaves it unread.
         * @param activation_scales Captured argument. The reader leaves it unread.
         * @param source_rows Captured argument. The reader leaves it unread.
         * @param gate_descs Original Gate descriptors.
         * @param up_descs Original Up descriptors.
         * @param original_to_grouped Captured active-route map.
         * @param original_experts Original route packet in submitted slot order.
         * @param gate_partials Captured Gate checksums destination.
         * @param up_partials Captured Up checksums destination.
         * @param first_slot First original route in this tile.
         * @param tile_routes Number of route slots in this tile.
         * @param N Prepared column pitch.
         * @param K Reduction width in activation elements.
         * @param num_experts Exact descriptor table length.
         * @param k_partitions Original ordered partition count.
         */
        template <uint8_t CodebookId, bool Scalar>
        __global__ void moeVerifierGateUpWeightPacketRead(
            const int8_t * /*activation_codes*/, const float * /*activation_scales*/,
            MoEGroupedSourceRows /*source_rows*/,
            const DeviceNativeQuantisedMatrixDesc *__restrict__ gate_descs,
            const DeviceNativeQuantisedMatrixDesc *__restrict__ up_descs,
            const int *__restrict__ original_to_grouped, const int *__restrict__ original_experts,
            float *__restrict__ gate_partials, float *__restrict__ up_partials,
            int first_slot, int tile_routes, int N, int K, int num_experts, int k_partitions)
        {
            const int n = blockIdx.x * blockDim.x + threadIdx.x;
            const int local_route = blockIdx.y;
            const int partition = blockIdx.z;
            if (n >= N || local_route >= tile_routes || partition >= k_partitions)
                return;
            const int slot = first_slot + local_route;
            const int expert = original_experts[slot];
            if (original_to_grouped[slot] < 0 || expert < 0 || expert >= num_experts)
                return;
            uint32_t gate = 0;
            uint32_t up = 0;
            gateUpWeightChecksums<CodebookId, Scalar>(
                gate_descs, up_descs, expert, n, N, K, partition, k_partitions, gate, up);
            const auto offset = (static_cast<size_t>(local_route) * k_partitions + partition) * N + n;
            gate_partials[offset] = __uint_as_float(gate);
            up_partials[offset] = __uint_as_float(up);
        }

        /**
         * @brief Read both Gate and Up planes in the original decode partition grid.
         * @tparam CodebookId Exact shared physical decoder of the two fixture projections.
         * @tparam Scalar Use byte-wise checksum loads.
         * @param activation_codes Captured argument. The reader leaves it unread.
         * @param activation_scales Captured argument. The reader leaves it unread.
         * @param gate_descs Original Gate descriptors.
         * @param up_descs Original Up descriptors.
         * @param expert_ids Original decode route IDs.
         * @param gate_partials Original slot-major Gate output used for checksums.
         * @param up_partials Original slot-major Up output used for checksums.
         * @param num_active Captured number of original route slots.
         * @param N Prepared column pitch.
         * @param K Reduction width in activation elements.
         * @param num_experts Exact descriptor table length.
         * @param k_partitions Original ordered partition count.
         */
        template <uint8_t CodebookId, bool Scalar>
        __global__ void moeDecodeGateUpWeightPacketRead(
            const int8_t * /*activation_codes*/, const float * /*activation_scales*/,
            const DeviceNativeQuantisedMatrixDesc *__restrict__ gate_descs,
            const DeviceNativeQuantisedMatrixDesc *__restrict__ up_descs,
            const int *__restrict__ expert_ids, float *__restrict__ gate_partials,
            float *__restrict__ up_partials, int num_active, int N, int K,
            int num_experts, int k_partitions)
        {
            const int n = blockIdx.x * blockDim.x + threadIdx.x;
            const int partition = blockIdx.y;
            const int slot = blockIdx.z;
            if (n >= N || partition >= k_partitions || slot >= num_active)
                return;
            const int expert = expert_ids[slot];
            uint32_t gate = 0;
            uint32_t up = 0;
            if (expert >= 0 && expert < num_experts)
                gateUpWeightChecksums<CodebookId, Scalar>(
                    gate_descs, up_descs, expert, n, N, K, partition, k_partitions, gate, up);
            const auto offset = (static_cast<size_t>(slot) * k_partitions + partition) * N + n;
            gate_partials[offset] = __uint_as_float(gate);
            up_partials[offset] = __uint_as_float(up);
        }

        /**
         * @brief Choose the checksum implementation before recording the native node.
         * @tparam Codebook Exact physical format used by the production fixture.
         * @param phase Original launch and argument layout.
         * @param access Requested packet access pattern.
         * @return CUDA entry with the original projection argument ABI.
         */
        template <uint8_t Codebook>
        const void *weightPacketReader(CUDAMoEWeightPhase phase, CUDAWeightPacketAccess access) noexcept
        {
            const bool scalar = access == CUDAWeightPacketAccess::ByteOracle;
            switch (phase)
            {
            case CUDAMoEWeightPhase::VerifierDown:
                return scalar ? reinterpret_cast<const void *>(moeVerifierWeightPacketRead<Codebook, true>)
                              : reinterpret_cast<const void *>(moeVerifierWeightPacketRead<Codebook, false>);
            case CUDAMoEWeightPhase::VerifierGateUp:
                return scalar ? reinterpret_cast<const void *>(moeVerifierGateUpWeightPacketRead<Codebook, true>)
                              : reinterpret_cast<const void *>(moeVerifierGateUpWeightPacketRead<Codebook, false>);
            case CUDAMoEWeightPhase::DecodeGateUp:
                return scalar ? reinterpret_cast<const void *>(moeDecodeGateUpWeightPacketRead<Codebook, true>)
                              : reinterpret_cast<const void *>(moeDecodeGateUpWeightPacketRead<Codebook, false>);
            case CUDAMoEWeightPhase::DecodeDown:
                return scalar ? reinterpret_cast<const void *>(moeDecodeDownWeightPacketRead<Codebook, true>)
                              : reinterpret_cast<const void *>(moeDecodeDownWeightPacketRead<Codebook, false>);
            }
            return nullptr;
        }
    }

    const void *cudaMoEWeightReferenceEntry(
        CUDAMoEWeightPhase phase, uint8_t codebook, CUDAWeightPacketAccess access) noexcept
    {
        switch (codebook)
        {
        case 0:  return weightPacketReader<0>(phase, access);
        case 4:  return weightPacketReader<4>(phase, access);
        case 5:  return weightPacketReader<5>(phase, access);
        case 6:  return weightPacketReader<6>(phase, access);
        case 7:  return weightPacketReader<7>(phase, access);
        case 8:  return weightPacketReader<8>(phase, access);
        case 9:  return weightPacketReader<9>(phase, access);
        case 10: return weightPacketReader<10>(phase, access);
        case 11: return weightPacketReader<11>(phase, access);
        case 12: return weightPacketReader<12>(phase, access);
        case 13: return weightPacketReader<13>(phase, access);
        case 14: return weightPacketReader<14>(phase, access);
        case 15: return weightPacketReader<15>(phase, access);
        case 16: return weightPacketReader<16>(phase, access);
        case 17: return weightPacketReader<17>(phase, access);
        case 19: return weightPacketReader<19>(phase, access);
        case kNativeQuantisedExpandedInt8MinCodebook:
            return weightPacketReader<kNativeQuantisedExpandedInt8MinCodebook>(phase, access);
        case kNativeQuantisedIQ4XSSourceScaleCodebook:
            return weightPacketReader<kNativeQuantisedIQ4XSSourceScaleCodebook>(phase, access);
        default: return nullptr;
        }
    }
} // namespace llaminar2::test
