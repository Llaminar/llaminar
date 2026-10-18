/**
 * @file CUDATensorAttentionPrimitives.cuh
 * @brief SM80+ exact QK and asynchronous-copy primitives with aligned, bank-rotated key rows.
 *
 * Q and K retain native 256-half rows in the tensor-owned shared plane; XOR permutes eight-half vectors within each 128-byte region.
 * Explicit PTX matrix fragments avoid dependence on WMMA's opaque register ABI.
 * Each accumulator receives the same ascending sixteen-element MMA sequence.
 * Publication remains the caller's existing CTA barrier and phase ownership.
 */
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace llaminar2::cuda::attention_primitives {

/** @brief Return the half-element column for an aligned eight-half key vector.
 * @param row Logical key row in the sixteen-key phase-owned tile.
 * @param column Logical head dimension, including its within-vector offset.
 * @return The XOR-permuted column; no vector crosses a 128-byte region.
 */
__device__ __forceinline__ int key_column(int row, int column) {
    return column ^ ((row & 7) << 3);
}

/** @brief Load four 8x8 half matrices using each lane's explicit row address.
 * @param address Sixteen-byte aligned shared address owned by this lane.
 * @param r0 First matrix fragment word.
 * @param r1 Second matrix fragment word.
 * @param r2 Third matrix fragment word.
 * @param r3 Fourth matrix fragment word.
 */
__device__ __forceinline__ void load_four(const half *address,
    uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3) {
    const uint32_t shared = static_cast<uint32_t>(__cvta_generic_to_shared(address));
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(shared));
}

/** @brief Advance one 16x8 FP32 accumulator by the next sixteen half operands.
 * @param a0 First A fragment word.
 * @param a1 Second A fragment word.
 * @param a2 Third A fragment word.
 * @param a3 Fourth A fragment word.
 * @param b0 First B fragment word.
 * @param b1 Second B fragment word.
 * @param c0 First owned FP32 accumulator.
 * @param c1 Second owned FP32 accumulator.
 * @param c2 Third owned FP32 accumulator.
 * @param c3 Fourth owned FP32 accumulator.
 */
__device__ __forceinline__ void advance(uint32_t a0, uint32_t a1,
    uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1,
    float &c0, float &c1, float &c2, float &c3) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

} // namespace llaminar2::cuda::attention_primitives

namespace llaminar2::cuda::attention_primitives {
/**
 * @brief Stage one aligned 128-bit key vector with explicit tail zero filling.
 * @param shared Destination in the caller-owned exclusive staging slot.
 * @param global Aligned native key vector; no dereference occurs for empty tails.
 * @param live_bytes Either sixteen live bytes or zero for a masked shared row.
 * The instruction and cache policy match the Apache-2.0 NInfer 3090 structure.
 * Llaminar owns the stream, stage lifetime and publication frontier.
 */
__device__ __forceinline__ void copy128ZeroFilled(void *shared, const void *global, int live_bytes) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        : : "r"(address), "l"(global), "r"(live_bytes));
}
/** @brief Commit exactly the copies issued into the next caller-owned stage. */
__device__ __forceinline__ void commitCopies() { asm volatile("cp.async.commit_group;\n"); }
/** @brief Join all committed copies before the CTA publishes the current stage. */
__device__ __forceinline__ void waitAllCopies() { asm volatile("cp.async.wait_group 0;\n"); }
} // namespace llaminar2::cuda::attention_primitives
