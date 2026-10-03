/**
 * @file ROCmVNNIPrefillLaunch.h
 * @brief Capture-time observation of the exact ROCm quantized prefill producer.
 *
 * Native low-bit GEMM and the blockwise INT8 GEMM used by Q8 weights have
 * different tuning controls. One thread-local observation records the actual
 * physical launch, not the requested override or a reconstructed policy.
 * Diagnostics query compiler resources explicitly, outside measured replay;
 * production graph execution owns no host shadow or additional device bytes.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

namespace llaminar2::rocm
{
/** @brief Physically distinct producers; tile sizes alone are not identity. */
enum class VNNIPrefillProducer : std::uint8_t
{
    NativeCooperative,
    NativeStreaming,
    Int8BlockwiseV3,
    Int8BlockwiseV7,
};

/** @brief Stable diagnostic spelling used in authenticated trainer evidence. */
[[nodiscard]] constexpr const char *vnniPrefillProducerName(
    VNNIPrefillProducer producer) noexcept
{
    switch (producer)
    {
    case VNNIPrefillProducer::NativeCooperative: return "native_cooperative";
    case VNNIPrefillProducer::NativeStreaming: return "native_streaming";
    case VNNIPrefillProducer::Int8BlockwiseV3: return "int8_blockwise_v3";
    case VNNIPrefillProducer::Int8BlockwiseV7: return "int8_blockwise_v7";
    }
    return "invalid";
}

/** @brief Immutable launch identity, including dynamically allocated LDS. */
struct VNNIPrefillLaunch
{
    VNNIPrefillProducer producer;
    std::uint8_t execution_codebook;
    int n_tile;
    int m_tile;
    int min_blocks;
    int unroll;
    bool full_tiles;
    const void *function;
    int block_threads;
    std::size_t dynamic_shared_memory_bytes;
};

/** @brief Compiler and occupancy evidence for one exact physical launch. */
struct VNNIPrefillLaunchResources
{
    int registers_per_thread = 0;
    std::size_t local_memory_bytes_per_thread = 0;
    std::size_t static_shared_memory_bytes = 0;
    int max_threads_per_block = 0;
    int max_active_blocks_per_sm = 0;
};

/**
 * @brief Publish one producer on the calling construction thread.
 * @param launch Complete physical identity captured beside the kernel launch.
 * No HIP resource queries, allocations, or synchronization occur here.
 */
void publishVNNIPrefillLaunch(const VNNIPrefillLaunch &launch);

/** @brief Begin a diagnostic observation, rejecting a stale prior producer. */
void clearVNNIPrefillLaunch() noexcept;

/** @brief Read the calling thread's producer, absent since reset until launch. */
[[nodiscard]] std::optional<VNNIPrefillLaunch> lastVNNIPrefillLaunch() noexcept;

/**
 * @brief Inspect the published symbol, using its actual block and LDS geometry.
 * @param launch Snapshot returned by lastVNNIPrefillLaunch().
 * @param resources Receives compiler allocation and theoretical residency.
 * @return False when HIP cannot authenticate complete resource evidence.
 * Call only during untimed diagnostics, never inside graph recording/replay.
 */
[[nodiscard]] bool inspectVNNIPrefillLaunch(
    const VNNIPrefillLaunch &launch, VNNIPrefillLaunchResources &resources);
} // namespace llaminar2::rocm
