/**
 * @file ROCmGemvKernel.hip
 * @brief GEMV kernel suite for decode (M=1) and prefill (M>1) on MI50/MI60 (gfx906)
 *
 * This file is the root of a three-file split:
 *
 *   ROCmGemvKernel_INT8_VNNI.hip
 *       All ratio-VNNI (INT8×INT8) kernels and their C dispatch wrappers:
 *       - Basic decode:         rocmGemv_int8_int8_int32_vnni
 *       - Blockwise scaled:     rocmGemv_int8_int8_fp32_vnni_blockwise_scaled
 *       - Fused FP32→INT8:      rocmGemv_fused_fp32_int8_vnni
 *       - Scatter/reduce:       rocmGemv_fused_scatter_fp32_int8_vnni
 *       - Self-reduce:          rocmGemv_fused_scatter_selfreduce_fp32_int8_vnni
 *       - Blockwise scatter:    rocmGemv_int8_scatter_vnni_blockwise
 *       - Batched scatter:      rocmGemv_int8_scatter_batched_vnni
 *       - VNNI repack:          rocmGemv_repackVNNI_to_rowmajor
 *       - Prefill GEMM:         rocmGemm_int8_int8_int32_vnni_prefill
 *       - Prefill grid-kpar:    rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar
 *       - Tuning overrides:     rocmGemv_int8_vnni_{set,reset}_{,wide_}tuning_overrides
 *
 *   ROCmGemvKernel_native_VNNI.hip
 *       Native-VNNI kernels that decode 4-bit nibbles losslessly:
 *       - Q4_0/IQ4_NL decode:   rocmGemv_native_vnni_fp32
 *       (Future native-VNNI kernels for other bit widths go here.)
 *
 *   ROCmGemvKernel.hip (this file)
 *       Kept as the canonical entry point and documentation index.
 *       All kernel implementations and dispatch functions live in the
 *       two files above; this file compiles to an empty translation unit.
 *
 * ## Design
 *
 * All GEMV kernels target gfx906 (MI50/MI60, 60 CUs, wavefront64) and
 * use the v_dot4_i32_i8 (VNNI) instruction for INT8 dot products.
 * Decode (M=1) is memory-bandwidth-bound: reading full model weights per
 * token at ~480-700 GB/s effective HBM2 bandwidth.
 *
 * Ratio-VNNI (INT8): weights pre-packed as [K/4][N][4] with per-column
 * FP32 scales. Activation quantization happens either ahead of time
 * (scatter variants) or fused in shared memory (fused variants).
 *
 * Native-VNNI (4-bit): weights stored as nibble payloads with FP16
 * per-block scales. Decode to INT8 happens in-register with no
 * precision loss (Q4_0: q-8; IQ4_NL: LUT).
 *
 * @see ROCmGemvKernel_INT8_VNNI.hip
 * @see ROCmGemvKernel_native_VNNI.hip
 * @see ROCmQuantisedGemmKernel.cpp  (C++ dispatch that calls these kernels)
 *
 * @author Llaminar V2
 * @date February 2026
 */
