/**
 * @file JitOnlineSoftmax.h
 * @brief JIT Microkernel μK2: Online softmax state management
 * @author David Sanftenberg
 * @date December 2025
 *
 * JIT-generated online softmax for streaming attention computation.
 * Supports both single-score updates (FA1) and batched 4x tile updates (FA2).
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * ALGORITHM (from "Online normalizer calculation for softmax")
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * State: (max, sum_exp)
 *
 * FA1 Single Update:
 *   if score > max:
 *     correction = exp(max - score)
 *     sum_exp *= correction
 *     max = score
 *   weight = exp(score - max)
 *   sum_exp += weight
 *   return weight
 *
 * FA2 Batched Tile Update (4 scores at once):
 *   tile_max = max(score0, score1, score2, score3)
 *   if tile_max > state_max:
 *     correction = exp(state_max - tile_max)
 *     state_sum *= correction
 *     context *= correction  // Rescale accumulated context!
 *     state_max = tile_max
 *
 *   For each score in [score0..score3]:
 *     weight[i] = exp(score[i] - state_max)
 *     state_sum += weight[i]
 *
 *   return weights[0..3]
 *
 * This enables single-pass attention without materializing the full NxN score matrix.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * TYPED REGISTER ALLOCATION (via RegisterAllocation.h)
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * StateZone (zmm16-19): Online softmax state
 *   - StateMax (zmm16): Running max for numerical stability
 *   - StateSum (zmm17): Running sum of exp weights
 *   - StateWeight (zmm18): Current attention weight (FA1 single mode)
 *   - StateCorr (zmm19): Correction factor for context rescaling
 *
 * ScoreZone (xmm20-23): FA2 tile scores (aliased to low Scratch)
 *   - Score0 (xmm20): Q·K[kv+0] - aliases Scratch0
 *   - Score1 (xmm21): Q·K[kv+1] - aliases Scratch1
 *   - Score2 (xmm22): Q·K[kv+2] - aliases Scratch2
 *   - Score3 (xmm23): Q·K[kv+3] - aliases Scratch3
 *
 * CRITICAL: Score/Scratch aliasing lifecycle:
 *   1. Dot product emits scores into Score0-3 (xmm20-23)
 *   2. emit_tile_state_update reads scores from Score0-3
 *   3. emit_tile_state_update overwrites Score0-3 with weights via Scratch0-3
 *   4. V accumulation reads weights from Scratch0-3 (zmm20-23)
 *
 * ScratchZone (zmm20-25): Temporary registers
 *   - Scratch0-3 (zmm20-23): ALIAS ScoreZone! Weights after scores consumed
 *   - Scratch4 (zmm24): Safe for tile_max - does NOT clobber scores
 *   - Scratch5 (zmm25): Safe for additional scratch
 */

#pragma once

#include "JitMicrokernelBase.h"
#include "JitFastExp.h"
#include "../../../jit/RegisterAllocation.h"
#include "../../../jit/RegisterEnforcement.h"

// Import typed register aliases and C++20 concepts for convenience
using namespace llaminar2::jit;

namespace llaminar::v2::kernels::jit
{
    // ============================================================================
    // OnlineSoftmax Register Manifest (compile-time documentation)
    // ============================================================================
    // Documents the registers borrowed by emit_tile_state_update().
    // Note: StateZone registers (zmm16-19) are accessed but not "borrowed" since
    // they're owned by the parent kernel and persist across all operations.

    using OnlineSoftmaxTileManifest = KernelRegisterManifest<
        Score0, Score1, Score2, Score3, // Input scores (xmm20-23)
        Scratch4                        // tile_max storage (zmm24)
        >;
    static_assert(!OnlineSoftmaxTileManifest::has_conflicts(),
                  "OnlineSoftmax tile manifest has conflicts!");

    /**
     * @brief JIT code emitter for online softmax update
     *
     * Emits code for a single online softmax update step.
     * Maintains running (max, sum) state across iterations.
     */
    class JitOnlineSoftmaxEmitter
    {
    public:
        JitOnlineSoftmaxEmitter() = default;

        /**
         * @brief Emit code to initialize softmax state
         *
         * Sets:
         * - zmm_max() = -infinity
         * - zmm_sum() = 0
         *
         * Prerequisites:
         * - zmm_neg_inf() initialized to -infinity
         *
         * @param gen Code generator
         */
        void emit_init_state(JitMicrokernelBase &gen)
        {
            using namespace Xbyak;

            gen.debug_emit("emit_init_softmax_state");

            // max = -inf
            gen.vmovaps(gen.state_max().zmm(), gen.const_neg_inf().zmm());

            // sum = 0
            gen.vxorps(gen.state_sum().zmm(), gen.state_sum().zmm(), gen.state_sum().zmm());
        }

        /**
         * @brief Emit online softmax update: update state with new score, return weight
         *
         * This is the core online softmax operation:
         *   if score > max:
         *     correction = exp(max - score)
         *     sum *= correction
         *     [caller rescales context by correction]
         *     max = score
         *   weight = exp(score - max)
         *   sum += weight
         *
         * Prerequisites:
         * - zmm_max(), zmm_sum() contain current state
         * - zmm_log2e(), zmm_exp_min() initialized for fast_exp
         * - zmm_neg_inf() initialized
         *
         * Output:
         * - zmm_weight() contains weight = exp(score - max) (broadcast)
         * - zmm_corr() contains correction factor (valid only if rescale_needed)
         * - zmm_max(), zmm_sum() updated
         *
         * @param gen Code generator
         * @param score_xmm XMM containing scalar score (element 0)
         * @param label_prefix Unique prefix for jump labels
         * @param rescale_callback Label to jump to for context rescaling (optional)
         */
        void emit_update(
            JitMicrokernelBase &gen,
            const Xbyak::Xmm &score_xmm,
            const std::string &label_prefix)
        {
            using namespace Xbyak;

            gen.debug_emit("emit_softmax_update");

            // Use scratch5 for score - scratch(0-2) are clobbered by emit_fast_exp
            // scratch0-2 = zmm20-22, scratch5 = zmm25
            Zmm zmm_score = gen.scratch5().zmm();

            // Broadcast score to all lanes (needed for comparison and subtraction)
            gen.vbroadcastss(zmm_score, score_xmm);

            // Initialize state_corr to 1.0 (no rescale needed by default)
            // This will be overwritten if score > max
            gen.load_constant_f32(gen.state_corr().zmm(), 1.0f);

            // Compare score with current max
            gen.vcomiss(score_xmm, Xmm(gen.state_max().zmm().getIdx()));

            std::string label_score_le_max = label_prefix + "_score_le_max";
            gen.jbe(label_score_le_max.c_str(), Xbyak::CodeGenerator::T_NEAR);

            // score > max: need to rescale
            gen.debug_emit("  score > max branch");
            {
                // correction = exp(max - score)
                gen.vsubps(gen.state_corr().zmm(), gen.state_max().zmm(), zmm_score);
                exp_emitter_.emit_fast_exp(gen, gen.state_corr().zmm(), gen.state_corr().zmm());

                // sum *= correction
                gen.vmulps(gen.state_sum().zmm(), gen.state_sum().zmm(), gen.state_corr().zmm());

                // max = score
                gen.vmovaps(gen.state_max().zmm(), zmm_score);

                // state_corr() now contains exp(old_max - new_max) < 1.0
                // Caller should rescale context accumulators by state_corr()
            }

            gen.L(label_score_le_max.c_str());

            // weight = exp(score - max)
            gen.vsubps(gen.state_weight().zmm(), zmm_score, gen.state_max().zmm());
            exp_emitter_.emit_fast_exp(gen, gen.state_weight().zmm(), gen.state_weight().zmm());

            // sum += weight
            gen.vaddps(gen.state_sum().zmm(), gen.state_sum().zmm(), gen.state_weight().zmm());
        }

        /**
         * @brief Emit code to compute final normalization factor
         *
         * Computes 1/sum for final context normalization.
         *
         * @param gen Code generator
         * @param dst_zmm ZMM to receive 1/sum (broadcast)
         */
        void emit_finalize(
            JitMicrokernelBase &gen,
            const Xbyak::Zmm &dst_zmm)
        {
            using namespace Xbyak;

            gen.debug_emit("emit_softmax_finalize");

            // inv_sum = 1.0 / sum
            // Use precise division (not approximate vrcp14ps)
            gen.vdivps(dst_zmm, gen.const_one().zmm(), gen.state_sum().zmm());
        }

        /**
         * @brief FA2 Tile Max Reduction: Find maximum across N scores
         *
         * For FA2-style tile processing, we compute all scores in a tile first,
         * then find the tile maximum in one pass. This is used to:
         *   1. Determine if we need to rescale existing context (tile_max > running_max)
         *   2. Compute all weights relative to the tile_max
         *
         * Input: scores in Score0-3 (xmm20-23) from ScoreZone
         * Output: tile_max broadcast in dst_zmm (must be Scratch4 or Scratch5!)
         *
         * CRITICAL: dst_zmm must NOT be Scratch0-3 (zmm20-23) as those alias
         * the Score registers we're reading from!
         *
         * @param gen Code generator
         * @param dst_zmm Output ZMM with max broadcast to all lanes (use zmm24/25)
         * @param score_xmm0..3 Input XMM registers containing scalar scores
         * @param num_scores Number of scores (1-4)
         */
        void emit_tile_max_reduction_4(
            JitMicrokernelBase &gen,
            const Xbyak::Zmm &dst_zmm,
            const Xbyak::Xmm &score_xmm0,
            const Xbyak::Xmm &score_xmm1,
            const Xbyak::Xmm &score_xmm2,
            const Xbyak::Xmm &score_xmm3,
            int num_scores = 4)
        {
            using namespace Xbyak;

            gen.debug_emit("emit_tile_max_reduction_4 (" + std::to_string(num_scores) + " scores)");

            // Verify we're not clobbering scores (runtime check for safety)
            // In typed version below, this is a compile-time check
            assert(dst_zmm.getIdx() >= 24 && "dst_zmm must be zmm24 or zmm25 to avoid clobbering scores!");

            // Start with first score
            gen.vmovss(Xmm(dst_zmm.getIdx()), score_xmm0);

            // Max with remaining scores
            if (num_scores >= 2)
            {
                gen.vmaxss(Xmm(dst_zmm.getIdx()), Xmm(dst_zmm.getIdx()), score_xmm1);
            }
            if (num_scores >= 3)
            {
                gen.vmaxss(Xmm(dst_zmm.getIdx()), Xmm(dst_zmm.getIdx()), score_xmm2);
            }
            if (num_scores >= 4)
            {
                gen.vmaxss(Xmm(dst_zmm.getIdx()), Xmm(dst_zmm.getIdx()), score_xmm3);
            }

            // Broadcast result to all lanes
            gen.vbroadcastss(dst_zmm, Xmm(dst_zmm.getIdx()));
        }

        /**
         * @brief FA2 Tile Max Reduction with compile-time safe registers
         *
         * Type-safe variant using typed registers from RegisterAllocation.h.
         * The template constraint enforces that output is Scratch4 or Scratch5,
         * preventing the bug where tile_max output clobbers score registers.
         *
         * Example usage:
         *   emit_tile_max_reduction_4_typed(gen, gen.scratch4(),
         *                                   gen.score0(), gen.score1(),
         *                                   gen.score2(), gen.score3());
         *
         * @tparam OutputReg Must satisfy require_safe_scratch_for_fa2
         * @param gen Code generator
         * @param output Typed register for output (compile-time verified as zmm24/25)
         * @param score0..3 Typed score registers
         * @param num_scores Number of scores (1-4)
         */
        template <typename OutputReg,
                  require_safe_scratch_for_fa2<OutputReg> = true>
        void emit_tile_max_reduction_4_typed(
            JitMicrokernelBase &gen,
            const OutputReg &output,
            const Score0 &s0,
            const Score1 &s1,
            const Score2 &s2,
            const Score3 &s3,
            int num_scores = 4)
        {
            emit_tile_max_reduction_4(gen, output.zmm(),
                                      s0.xmm(), s1.xmm(), s2.xmm(), s3.xmm(),
                                      num_scores);
        }

        /**
         * @brief FA2 Batched Softmax State Update
         *
         * Updates softmax state for an entire tile of scores at once.
         * This is more efficient than per-score updates because:
         *   1. Context rescaling happens at most once per tile (not per score)
         *   2. Sum accumulation is batched
         *
         * Algorithm:
         *   if tile_max > running_max:
         *     correction = exp(running_max - tile_max)
         *     sum *= correction
         *     [caller rescales context]
         *     running_max = tile_max
         *   for each score in tile:
         *     weight[i] = exp(score[i] - running_max)
         *     sum += weight[i]
         *
         * Uses typed state registers:
         *   - StateMax (zmm16): Running maximum
         *   - StateSum (zmm17): Running sum
         *   - StateCorr (zmm19): Correction factor
         *
         * REGISTER GUARDS:
         * - emit_fast_exp borrows Input0 (zmm8), Input6 (zmm14), Input7 (zmm15)
         * - tile_max_zmm is caller-owned (should be Scratch4 = zmm24)
         * - StateMax/StateSum/StateCorr are state zone (zmm16-19)
         *
         * @param gen Code generator
         * @param tile_max_zmm ZMM with tile max broadcast (should be Scratch4/5)
         * @param label_prefix Unique prefix for jump labels
         */
        void emit_tile_state_update(
            JitMicrokernelBase &gen,
            const Xbyak::Zmm &tile_max_zmm,
            const std::string &label_prefix)
        {
            using namespace Xbyak;

            gen.debug_emit("emit_tile_state_update");

            // Use typed state accessors for clarity
            Zmm zmm_max = gen.state_max().zmm();
            Zmm zmm_sum = gen.state_sum().zmm();
            Zmm zmm_corr = gen.state_corr().zmm();

            // Initialize correction to 1.0 (no rescale needed by default)
            gen.load_constant_f32(zmm_corr, 1.0f);

            // Compare tile_max with running_max
            gen.vcomiss(Xmm(tile_max_zmm.getIdx()), Xmm(zmm_max.getIdx()));

            std::string label_no_rescale = label_prefix + "_no_rescale";
            gen.jbe(label_no_rescale.c_str(), Xbyak::CodeGenerator::T_NEAR);

            // tile_max > running_max: need to rescale
            gen.debug_emit("  tile_max > running_max branch");
            {
                // CRITICAL FIX: Save tile_max to running_max BEFORE calling emit_fast_exp!
                // emit_fast_exp uses Scratch4 (zmm24) and Scratch5 (zmm25) internally,
                // which clobbers tile_max_zmm if it's in zmm24/zmm25.
                // By copying tile_max to zmm_max first, we preserve the value.

                // Step 1: Compute (running_max - tile_max) into zmm_corr
                //         This uses the OLD running_max before we overwrite it
                gen.vsubps(zmm_corr, zmm_max, tile_max_zmm);

                // Step 2: Save tile_max to running_max NOW (before exp clobbers it)
                gen.vmovaps(zmm_max, tile_max_zmm);

                // Step 3: exp(old_max - new_max) = correction factor
                //         NOTE: emit_fast_exp clobbers zmm24/zmm25, but tile_max
                //         is now safely stored in zmm_max (zmm16)
                exp_emitter_.emit_fast_exp(gen, zmm_corr, zmm_corr);

                // Step 4: sum *= correction
                gen.vmulps(zmm_sum, zmm_sum, zmm_corr);

                // zmm_corr now contains correction factor for context rescaling
            }

            gen.L(label_no_rescale.c_str());
        }

        /**
         * @brief Compute weight for a single score and accumulate sum
         *
         * After tile_state_update, compute weight = exp(score - running_max)
         * and accumulate into sum.
         *
         * Uses typed state registers:
         *   - StateMax (zmm16): Current running maximum
         *   - StateSum (zmm17): Running sum (updated)
         *
         * REGISTER GUARDS:
         * - emit_fast_exp internally borrows Input0 (zmm8), Input6 (zmm14), Input7 (zmm15)
         * - The caller's weight_zmm is written but not borrowed here (caller owns it)
         *
         * @param gen Code generator
         * @param score_xmm Input scalar score (typically Score0-3)
         * @param weight_zmm Output weight (broadcast to all lanes)
         */
        void emit_compute_weight_and_accumulate(
            JitMicrokernelBase &gen,
            const Xbyak::Xmm &score_xmm,
            const Xbyak::Zmm &weight_zmm)
        {
            using namespace Xbyak;

            gen.debug_emit("emit_compute_weight_and_accumulate");

            // Use typed state accessors
            Zmm zmm_max = gen.state_max().zmm();
            Zmm zmm_sum = gen.state_sum().zmm();

            // weight = exp(score - running_max)
            gen.vbroadcastss(weight_zmm, score_xmm);
            gen.vsubps(weight_zmm, weight_zmm, zmm_max);

            // emit_fast_exp uses Input0 (zmm8), Input6 (zmm14), Input7 (zmm15)
            // Those guards are managed inside emit_fast_exp/emit_exp2_poly
            exp_emitter_.emit_fast_exp(gen, weight_zmm, weight_zmm);

            // sum += weight
            gen.vaddps(zmm_sum, zmm_sum, weight_zmm);
        }

        /**
         * @brief Typed variant of emit_compute_weight_and_accumulate
         *
         * @tparam ScoreReg Typed score register (Score0-3)
         * @param gen Code generator
         * @param score Typed score register
         * @param weight_zmm Output weight ZMM
         */
        template <typename ScoreReg>
        void emit_compute_weight_and_accumulate_typed(
            JitMicrokernelBase &gen,
            const ScoreReg &score,
            const Xbyak::Zmm &weight_zmm)
        {
            static_assert(is_zone_v<ScoreReg, ScoreZone>,
                          "score must be from ScoreZone (Score0-3)");
            emit_compute_weight_and_accumulate(gen, score.xmm(), weight_zmm);
        }

    private:
        JitFastExpEmitter exp_emitter_;
    };

    /**
     * @brief Standalone JIT kernel for online softmax (for testing)
     *
     * Processes an array of scores and outputs softmax weights.
     *
     * Function signature:
     *   void kernel(const float* scores, float* weights, int n)
     */
    class JitOnlineSoftmaxKernel : public JitMicrokernelBase
    {
    public:
        using kernel_func_t = void (*)(const float *scores, float *weights, int n);

        explicit JitOnlineSoftmaxKernel(bool debug = false)
            : JitMicrokernelBase(8 * 1024, debug)
        {
            generate();
        }

        kernel_func_t get_kernel()
        {
            return getCode<kernel_func_t>();
        }

    private:
        JitOnlineSoftmaxEmitter softmax_emitter_;

        void generate()
        {
            using namespace Xbyak;

            debug_emit("JitOnlineSoftmaxKernel::generate()");

            // Function: void kernel(const float* scores, float* weights, int n)
            // Args: rdi = scores, rsi = weights, edx = n
            const Reg64 &reg_scores = rdi;
            const Reg64 &reg_weights = rsi;
            const Reg64 &reg_n = rdx;
            const Reg64 &reg_i = rcx;

            // Save callee-saved (none needed for this simple kernel)

            // Initialize constants
            debug_emit("  Init constants");
            emit_broadcast_fp32_const(const_neg_inf().zmm(), -std::numeric_limits<float>::infinity(), rax);
            emit_broadcast_fp32_const(const_one().zmm(), 1.0f, rax);
            emit_broadcast_fp32_const(const_log2e().zmm(), 1.4426950408889634f, rax);
            emit_broadcast_fp32_const(const_exp_min().zmm(), -87.0f, rax);

            // Allocate stack for temporary weight storage
            // We need to store weights during first pass, then normalize
            // Stack: [weights_tmp: n * 4 bytes]
            // For simplicity, cap at 1024 elements
            const int max_n = 1024;
            sub(rsp, max_n * 4 + 64); // +64 for alignment

            // Initialize softmax state
            softmax_emitter_.emit_init_state(*this);

            // Pass 1: Compute online softmax weights (unnormalized)
            debug_emit("  Pass 1: Compute weights");
            xor_(reg_i, reg_i);

            Label loop1_start, loop1_end;
            L(loop1_start);
            cmp(reg_i, reg_n);
            jge(loop1_end, T_NEAR);

            // Load score
            vmovss(xmm0, ptr[reg_scores + reg_i * 4]);

            // Update softmax state (unique label per iteration not needed, use index)
            // For loop, we need unique labels - use a counter approach
            // Actually, the label is inside emit_update, so we pass a unique prefix
            // Use reg_i value for uniqueness is tricky in JIT...
            // Simpler: use a single label scheme with je/jne flags
            // For this test kernel, inline the update logic

            // Inline softmax update (without label issues)
            // This duplicates emit_update but avoids label collision in loop
            {
                Zmm zmm_score = scratch0().zmm();
                vbroadcastss(zmm_score, xmm0);

                // Compare score with max
                vcomiss(xmm0, Xmm(state_max().zmm().getIdx()));

                Label score_le_max;
                jbe(score_le_max, T_NEAR);

                // score > max: rescale
                vsubps(state_corr().zmm(), state_max().zmm(), zmm_score);
                JitFastExpEmitter().emit_fast_exp(*this, state_corr().zmm(), state_corr().zmm());
                vmulps(state_sum().zmm(), state_sum().zmm(), state_corr().zmm());
                vmovaps(state_max().zmm(), zmm_score);

                L(score_le_max);

                // weight = exp(score - max)
                vsubps(state_weight().zmm(), zmm_score, state_max().zmm());
                JitFastExpEmitter().emit_fast_exp(*this, state_weight().zmm(), state_weight().zmm());

                // sum += weight
                vaddps(state_sum().zmm(), state_sum().zmm(), state_weight().zmm());
            }

            // Store unnormalized weight to stack
            vmovss(ptr[rsp + reg_i * 4], Xmm(state_weight().zmm().getIdx()));

            inc(reg_i);
            jmp(loop1_start, T_NEAR);

            L(loop1_end);

            // Compute 1/sum
            debug_emit("  Compute 1/sum");
            Zmm zmm_inv_sum = scratch3().zmm();
            softmax_emitter_.emit_finalize(*this, zmm_inv_sum);

            // Pass 2: Normalize weights
            debug_emit("  Pass 2: Normalize");
            xor_(reg_i, reg_i);

            Label loop2_start, loop2_end;
            L(loop2_start);
            cmp(reg_i, reg_n);
            jge(loop2_end, T_NEAR);

            // Load unnormalized weight from stack
            vmovss(xmm0, ptr[rsp + reg_i * 4]);

            // Normalize
            vmulss(xmm0, xmm0, Xmm(zmm_inv_sum.getIdx()));

            // Store to output
            vmovss(ptr[reg_weights + reg_i * 4], xmm0);

            inc(reg_i);
            jmp(loop2_start, T_NEAR);

            L(loop2_end);

            // Restore stack
            add(rsp, max_n * 4 + 64);
            ret();
        }
    };

} // namespace llaminar::v2::kernels::jit
