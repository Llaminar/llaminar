/**
 * @file ExpertTierWeightWireTestOracle.h
 * @brief Compact native wire bytes from independently prepared CPU oracle roots.
 *
 * Native integration tests still compare against the production CPU packer's
 * exact bytes, while transport tails now exclude inactive stored N columns.
 * This helper never decodes, requantizes or reads an original model payload.
 */
#pragma once
#include "execution/moe/ExpertTierWeightDeviceLayout.h"
#include "kernels/cpu/gemm/CPUNativeVNNIWeightPacker.h"
#include <span>
#include <stdexcept>
#include <vector>
namespace llaminar2::test {
/**
 * @brief Gather one exact wire prefix from a genuine CPU reference root.
 * @param layout Original endpoint/intersection layout.
 * @param root Independently prepared complete local CPU oracle allocation.
 * @param first Intersection-relative first unit.
 * @param count Exact live unit count.
 * @return Native payload and metadata bytes for those actual columns only.
 */
inline std::vector<uint8_t> expertTierWireOracle(const ExpertTierWeightDeviceLayout& layout,
    std::span<const uint8_t> root,uint32_t first,uint32_t count)
{
    std::vector<uint8_t> result(layout.chunkBytes(count,first));
    if(!gatherExpertTierCPUWeightChunk(layout,root,first,count,result))
        throw std::invalid_argument("CPU reference root cannot form exact native transfer chunk");
    return result;
}
/**
 * @brief Identify an expanded-INT8 root byte by original physical column coordinates.
 * @param root Original independently packed CPU reference and its physical strides.
 * @param offset Byte in the complete root allocation.
 * @return Original N coordinate, including allocation-only padded columns.
 * @throws std::invalid_argument If this is not the documented INT8/16-bit-plane layout.
 *
 * This oracle does not inspect transfer writes, gather/scatter output, or zero
 * values. Eight four-byte weight groups precede 64-column 16-bit metadata
 * planes; all use the SAME original N column. It deliberately rejects other
 * encodings instead of inferring their live extent from implementation writes.
 */
inline size_t expertTierExpandedCPUByteRow(
    const cpu::native_vnni::CPUNativeVNNIPackedWeights &root, size_t offset)
{
    constexpr size_t data_bytes = 64u * 32u;
    const size_t stride = root.interleaved_block_stride;
    if (!root.usesExpandedInt8() || root.N <= 0 || root.K <= 0 || root.K % 32 ||
        stride != data_bytes + 256u + (root.is_asymmetric ? 128u : 0u) ||
        root.blocks_per_row != root.K / 32 || offset >= root.native_interleaved.size())
        throw std::invalid_argument("Expanded CPU byte oracle requires its original INT8 root");
    const size_t unit = offset / stride, within = offset % stride;
    const size_t column = within < data_bytes
        ? (within % (64u * 4u)) / 4u : ((within - data_bytes) % 128u) / 2u;
    return (unit / root.blocks_per_row) * 64u + column;
}
}
