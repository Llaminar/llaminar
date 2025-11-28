/**
 * @file Ops.h
 * @brief Convenience header for all pipeline operations
 *
 * Include this single header to get access to all reusable operations:
 * - RMSNormOp: RMS normalization (pre-attention, pre-FFN, final)
 * - GemmOp: Matrix multiplication (projections, LM head)
 * - SwiGLUOp: SwiGLU activation (FFN)
 * - RoPEOp: Rotary position embeddings (attention)
 * - ResidualOp: Residual connections (post-attention, post-FFN)
 *
 * Usage in pipeline:
 * @code
 * #include "ops/Ops.h"
 *
 * class Qwen2Pipeline : public PipelineBase {
 *     // Operation instances (stateless, can be shared)
 *     RMSNormOp rmsnorm_;
 *     GemmOp gemm_;
 *     SwiGLUOp swiglu_;
 *     RoPEOp rope_;
 *     ResidualOp residual_;
 *
 *     bool attention_block(...) {
 *         TRY_OP(rmsnorm_(input, norm_weight, normalized, seq_len, d_model,
 *                         1e-6f, "layer0_ATTN_NORM", mpi, device));
 *         TRY_OP(gemm_(normalized, layer.wq.get(), Q, seq_len, q_dim, k_dim,
 *                      "layer0_Q_PROJ", mpi, device));
 *         // ... etc
 *         return true;
 *     }
 * };
 * @endcode
 *
 * @author David Sanftenberg
 */

#pragma once

#include "Op.h"
#include "RMSNormOp.h"
#include "GemmOp.h"
#include "SwiGLUOp.h"
#include "RoPEOp.h"
#include "ResidualOp.h"
