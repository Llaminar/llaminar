/**
 * @file CPUNativeNEONPackedPrefill.h
 * @brief Panel-major ARM prefill over the canonical file-density publication.
 *
 * Each column strip has one worker owner. The worker decodes one increasing-K
 * panel into its private invocation slab and consumes it across every active
 * row tile before reusing that slot. We never publish an expanded weight bank.
 * Native FP32 state rolls directly through the destination between panels;
 * an ordered partition spanning panels keeps its unfinished sum separate from
 * the already folded destination. Ordered (GPU-aligned) partitions that fit a
 * panel are packed whole and folded in registers, publishing the running
 * total once per panel. No output tile is staged and then copied.
 *
 * One 64-column chunk is the unit of work. A dense projection workshares its
 * own chunks; a pooled multi-projection launch (the batched MoE expert route)
 * interleaves the chunk tasks of several admitted bindings in one workshare,
 * every binding using the same uniform slot stride of one shared slab. An
 * optional bias row is added after each row's complete K result, with the
 * same FP32 operation and operand order as the register tile route.
 */
#pragma once

#include "CPUNativeNEONPanelWorkspace.h"
#include "CPUNativeNEONProjectionBatch.h"

namespace llaminar2::cpu::native_quantised
{
#if defined(__GNUC__) && !defined(__clang__)
#define LLAMINAR_NEON_PREFILL_EXACT __attribute__((optimize("fp-contract=off", "no-associative-math")))
#else
#define LLAMINAR_NEON_PREFILL_EXACT
#endif

    /** @brief Synchronous borrowed prefill program; the batch and slab outlive execute(). */
    class CPUNativeNEONPackedPrefill final
    {
        using Tile = CPUNativeNEONProjectionTile<1>;
        using Dot = multi_scale::NEONByteDot;
        using Policy = CPUProjectionNumericalPolicy;
        static constexpr int dot_floats = 128 / sizeof(float);
        static constexpr int block_floats = neon_panel::block_bytes / sizeof(float);

    public:
        /**
         * @brief Check the stage-owned worker bank before any quantizer or dot writes.
         * @param batch Admitted original-coordinate projection; an attached bias is
         *        added after each row's complete K result.
         * @param scratch Persistent slab, disjoint from all invocation inputs and outputs.
         * @param workers Budget for the active or prospective OpenMP team.
         * @return Binding that uses only the admitted slab prefix.
         * @throws std::invalid_argument For ineligible geometry, a short or misaligned
         *         slab, or a slab aliasing any invocation publication.
         */
        static CPUNativeNEONPackedPrefill bind(const CPUNativeNEONProjectionBatch &batch,
            std::span<float> scratch, int workers)
        {
            const auto &w = batch.weights_;
            const auto slot = neon_panel::workerFloats(rowsOf(batch), w.K, w.numerical_policy);
            admit(batch, scratch, workers, slot);
            return {batch, scratch.data(), slot, workers};
        }

        /**
         * @brief Admit one member of a pooled multi-projection launch before any team write.
         * @param batch Admitted original-coordinate member projection.
         * @param scratch The launch's single slab; worker w owns [w*slot_floats, (w+1)*slot_floats).
         * @param workers Admitted slot count; the executing team must not exceed it.
         * @param slot_floats Launch-uniform slot stride, at least this member's own worker demand,
         *        so that concurrently running members of one launch never share a slot.
         * @return Binding whose chunk tasks touch only the calling worker's slot.
         * @throws std::invalid_argument For ineligible geometry, a short stride or slab,
         *         misalignment, or a slab aliasing this member's publications.
         */
        static CPUNativeNEONPackedPrefill bindPooled(const CPUNativeNEONProjectionBatch &batch,
            std::span<float> scratch, int workers, std::size_t slot_floats)
        {
            admit(batch, scratch, workers, slot_floats);
            return {batch, scratch.data(), slot_floats, workers};
        }

        /** @brief Execute the checked packed panel binding with its admitted native dot family. */
        template<Dot Instruction> void execute() const
        {
            if (!batch_.grid_.beginWorkshare()) return;
            requireTeam(llaminar::v2::inLexicalWorkshare() ? omp_get_num_threads() : omp_get_max_threads());
            if constexpr (Instruction != Dot::ASIMD)
                ARMCPUCapabilities::native().require(ARMCPUFeature::DotProd);
            if constexpr (Instruction == Dot::I8MM)
                ARMCPUCapabilities::native().require(ARMCPUFeature::IntegerMatMul);
            const int chunks = batch_.weights_.N_padded / 64;
            auto work = [&] {
                ARMFloatingPointProfile::require("Native NEON packed prefill");
#pragma omp for schedule(dynamic, 1)
                for (int chunk = 0; chunk < chunks; ++chunk)
                    runAdmittedTask<Instruction>(chunk);
            };
            OMP_WORKSHARE_REGION(work);
        }

        /**
         * @brief Reject a team wider than the admitted slot count, before any team write.
         * @param team Width of the active or prospective team.
         * @throws std::invalid_argument When a worker id could index past the slab.
         */
        void requireTeam(int team) const
        {
            if (team > workers_)
                throw std::invalid_argument("Native NEON prefill team exceeds its admitted projection slab");
        }

        /** @return Independent 64-column chunk tasks; zero when no original row is active. */
        [[nodiscard]] std::int64_t taskCount() const noexcept
        { return batch_.grid_.hasAny() ? batch_.weights_.N_padded / 64 : 0; }

        /**
         * @brief Execute one chunk task of a pooled launch on the calling worker's slot.
         * @tparam Instruction Native dot family whose CPU features the launcher admitted.
         * @param chunk Index in [0, taskCount()); no synchronization, no shared writes.
         * @details Each task streams every four-column strip of its chunk through
         * the worker's slot and writes only that chunk of every active row, so
         * pooled tasks of different members and chunks never conflict. FPCR is
         * per thread, so every task admits its worker's floating-point profile.
         */
        template<Dot Instruction> void runTask(std::int64_t chunk) const noexcept
        {
            ARMFloatingPointProfile::require("Native NEON packed prefill");
            runAdmittedTask<Instruction>(static_cast<int>(chunk));
        }

    private:
        /** @brief Retain the original spans and the size of each worker-exclusive slot. */
        CPUNativeNEONPackedPrefill(const CPUNativeNEONProjectionBatch &batch, float *scratch,
            std::size_t worker_floats, int workers) noexcept
            : batch_(batch), scratch_(scratch), worker_floats_(worker_floats), workers_(workers) {}

        /** @return Physical row capacity of an admitted binding. */
        static int rowsOf(const CPUNativeNEONProjectionBatch &batch) noexcept
        { return static_cast<int>(batch.activation_.size() / batch.weights_.blocks_per_row); }

        /**
         * @brief Shared admission of dense and pooled bindings.
         * @param batch Admitted original-coordinate projection.
         * @param scratch Slab whose first workers*slot_floats floats this binding may write.
         * @param workers Positive admitted slot count.
         * @param slot_floats Slot stride, at least this binding's own worker demand.
         * @throws std::invalid_argument For any ineligible, short, misaligned or aliasing input.
         */
        static void admit(const CPUNativeNEONProjectionBatch &batch, std::span<float> scratch, int workers,
            std::size_t slot_floats)
        {
            const auto &w = batch.weights_;
            const int rows = rowsOf(batch);
            if (workers <= 0 || !neon_panel::supports(w.encoding, w.codebook_id, w.is_asymmetric) ||
                rows < neon_panel::min_rows || slot_floats < neon_panel::workerFloats(rows, w.K, w.numerical_policy) ||
                slot_floats % neon_panel::line_floats)
                throw std::invalid_argument("Native NEON prefill requires eligible retained panel geometry");
            if (slot_floats > std::numeric_limits<std::size_t>::max() / sizeof(float) / std::size_t(workers))
                throw std::overflow_error("Native NEON packed panel workspace exceeds addressable storage");
            const auto used = std::size_t(workers) * slot_floats;
            if (scratch.size() < used || reinterpret_cast<std::uintptr_t>(scratch.data()) % 64)
                throw std::invalid_argument("Native NEON prefill requires its aligned per-worker projection slab");
            const storage_proof::Bytes writes{scratch.data(), used * sizeof(float)};
            storage_proof::requireDisjoint(writes, {batch.activation_.data(), batch.activation_.size_bytes()});
            storage_proof::requireDisjoint(writes, {batch.output_.data(), batch.output_.size_bytes()});
            storage_proof::requireDisjoint(writes, {w.bytes().data(), w.bytes().size()});
            if (!batch.bias_.empty()) storage_proof::requireDisjoint(writes, {batch.bias_.data(), batch.bias_.size_bytes()});
            storage_proof::requireOriginalWorkReadOnly(writes, batch.grid_.publication());
        }

        /** @brief Resolve the source program and arithmetic policy for one chunk on this worker's slot. */
        template<Dot Instruction> void runAdmittedTask(int chunk) const noexcept
        {
            auto *panel = scratch_ + std::size_t(omp_get_thread_num()) * worker_floats_;
            const auto &w = batch_.weights_;
            if (w.usesCompactMultiScale())
            {
                if (w.codebook_id == 11) return dispatch<Instruction, 11>(chunk, panel);
                return dispatch<Instruction, 7>(chunk, panel);
            }
            dispatch<Instruction, 0>(chunk, panel);
        }

        /**
         * @brief Decode one 32-value K block for four columns directly into its slab slot.
         * @tparam C IQ3_S (11), Q5_K (7), or Q5_0 (0).
         * @tparam Ops Native consumer's lane layout; SMMLA operands are zipped once here.
         * @param unit First byte of the file-density block.
         * @param c First column inside the prepared chunk.
         * @param x Existing worker slab; weight bytes use byte-wise access and the scale planes are FP32.
         */
        template<int C, typename Ops> [[gnu::always_inline]] static void packBytes(
            const std::uint8_t *unit, int c, float *x) noexcept
        {
            auto *bytes = reinterpret_cast<std::int8_t *>(x);
            const auto put = [&]<int G>(int8x16_t a, int8x16_t b) __attribute__((always_inline)) {
                if constexpr (multi_scale::usesRowPairs<Ops>)
                {
                    const auto l = vreinterpretq_s32_s8(a), r = vreinterpretq_s32_s8(b);
                    a = vreinterpretq_s8_s32(vzip1q_s32(l, r));
                    b = vreinterpretq_s8_s32(vzip2q_s32(l, r));
                }
                std::memcpy(bytes + 16 * G, &a, sizeof(a));
                std::memcpy(bytes + 16 * (G + 1), &b, sizeof(b));
            };
            if constexpr (C == 11)
            {
                using I = typename Ops::I;
                const auto meta = multi_scale::payloadWord<11, Ops>(unit, 2, c);
                const uint8x16_t gather{255, 0, 255, 4, 255, 8, 255, 12, 255, 0, 255, 4, 255, 8, 255, 12};
                const int16x8_t align{0, 0, 0, 0, -1, -1, -1, -1};
                const auto high = vshlq_u16(vreinterpretq_u16_u8(vqtbl1q_u8(std::bit_cast<uint8x16_t>(meta), gather)), align);
                Tile::forEachGroup<4>([&]<int Pair>() __attribute__((always_inline)) {
                    const auto qs = multi_scale::payloadWord<11, Ops>(unit, Pair / 2, c);
                    const auto indices = Tile::iq3PairIndices<Pair, Ops>(qs, high);
                    const auto signs = [&]() __attribute__((always_inline)) {
                        if constexpr (Pair < 3) return Ops::template repeatByte<1 + Pair>(meta);
                        else
                        {
                            const auto tail = vreinterpretq_u8_u32(vld1q_dup_u32(reinterpret_cast<const std::uint32_t *>(
                                unit + preparedCompactPayloadWords(11) * 256 + c)));
                            return std::bit_cast<I>(vqtbl1q_u8(tail,
                                uint8x16_t{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}));
                        }
                    }();
                    put.template operator()<2 * Pair>(Tile::iq3PairGroup<0, Ops>(indices, signs),
                        Tile::iq3PairGroup<1, Ops>(indices, signs));
                });
            }
            else if constexpr (C == 7)
            {
                Tile::forEachGroup<2>([&]<int Pair>() __attribute__((always_inline)) {
                    constexpr int g = 2 * Pair;
                    const auto a = vld1q_u8(unit + g * 256 + c * 4);
                    const auto b = vld1q_u8(unit + (g + 1) * 256 + c * 4);
                    const auto *plane = unit + 1024 + c / 2 + 32 * g;
                    put.template operator()<g>(
                        vreinterpretq_s8_u8(Tile::q5KMerge(vandq_u8(a, vdupq_n_u8(15)), plane)),
                        vreinterpretq_s8_u8(Tile::q5KMerge(vandq_u8(b, vdupq_n_u8(15)), plane + 32)));
                    put.template operator()<g + 4>(
                        vreinterpretq_s8_u8(Tile::q5KMerge(vshrq_n_u8(a, 4), plane + 128)),
                        vreinterpretq_s8_u8(Tile::q5KMerge(vshrq_n_u8(b, 4), plane + 160)));
                });
            }
            else
            {
                Tile::forEachGroup<2>([&]<int Pair>() __attribute__((always_inline)) {
                    constexpr int g = 2 * Pair;
                    const auto a = Tile::ordinaryQ5Groups<true>(unit, c, g);
                    const auto b = Tile::ordinaryQ5Groups<true>(unit, c, g + 1);
                    put.template operator()<g>(a[0], b[0]);
                    put.template operator()<g + 4>(a[1], b[1]);
                });
            }
        }

        /** @brief Stream one physical K panel; widen each original superblock header once. */
        template<int C, typename Ops> [[gnu::noinline]] static void pack(const CPUNativeQuantisedWeightView &w,
            int chunk, int c, int first, int end, float *buffer) noexcept
        {
            const auto geometry = w.geometry();
            const auto *root = w.interleavedBase() + std::size_t(chunk) * geometry.rowBytes();
            if constexpr (C != 0)
            {
                for (int block = first; block < end; )
                {
                    const int last = std::min(end, static_cast<int>(geometry.superblockEnd(block)));
                    const auto *unit = root + geometry.blockOffset(0, block);
                    const auto base = Tile::codedBase<C, Ops>(root + geometry.headerOffset(0, block), c);
                    constexpr auto stride = preparedInterleavedBlockStride(CPUNativeQuantisedEncoding::CompactMultiScale, C == 7, C);
                    for (; block < last; ++block, unit += stride, buffer += block_floats)
                    {
                        packBytes<C, Ops>(unit, c, buffer);
                        const auto factors = Tile::codedFactors<C, Ops>(unit, base, c);
                        vst1q_f32(buffer + dot_floats, std::bit_cast<float32x4_t>(factors.primary));
                        if constexpr (C == 7)
                            vst1q_f32(buffer + dot_floats + 4, std::bit_cast<float32x4_t>(factors.secondary));
                    }
                }
            }
            else
            {
                constexpr auto stride = preparedInterleavedBlockStride(CPUNativeQuantisedEncoding::Q5NativeSingleScale,
                    false, native_quantised_formats::Q5_0.codebook_id);
                const auto *unit = root + std::size_t(first) * stride;
                for (int block = first; block < end; ++block, unit += stride, buffer += block_floats)
                {
                    packBytes<C, Ops>(unit, c, buffer);
                    vst1q_f32(buffer + dot_floats, std::bit_cast<float32x4_t>(Ops::half(
                        reinterpret_cast<const std::uint16_t *>(unit + Q5NativePayload::bytes) + c)));
                }
            }
        }

        /** @brief Read only live N lanes of the prior running result; padding stays untouched. */
        static float32x4_t loadOutput(const float *p, int n) noexcept
        {
            if (n == 4) return vld1q_f32(p);
            auto value = vld1q_lane_f32(p, vdupq_n_f32(0), 0);
            if (n > 1) value = vld1q_lane_f32(p + 1, value, 1);
            if (n > 2) value = vld1q_lane_f32(p + 2, value, 2);
            return value;
        }

        /** @brief Publish the running result straight to its original N coordinates. */
        static void storeOutput(float *p, float32x4_t value, int n) noexcept
        {
            if (n == 4) { vst1q_f32(p, value); return; }
            vst1q_lane_f32(p, value, 0);
            if (n > 1) vst1q_lane_f32(p + 1, value, 1);
            if (n > 2) vst1q_lane_f32(p + 2, value, 2);
        }

        /** @brief Take one packed native dot operand without treating its INT8 bytes as floats. */
        [[gnu::always_inline]] static int8x16_t dotOperand(const float *block, int group) noexcept
        {
            int8x16_t result;
            std::memcpy(&result, block + 4 * group, sizeof(result));
            return result;
        }

        /** @brief Apply the original single-block FP32 contribution to one output row. */
        template<Policy P, bool Minimum>
        [[gnu::always_inline]] static float32x4_t update(float32x4_t sum, int32x4_t dot, const float *block,
            const Q8_1Block &activation) noexcept
        {
            const float scale = nativeQuantisedFP16ScaleToFP32(activation.d);
            const auto w = vld1q_f32(block + dot_floats);
            if constexpr (Minimum)
                return accumulateCorrectedSingleScaleBlockNEON(sum, dot, w, activation.sum_qs,
                    vld1q_f32(block + dot_floats + 4), scale, P);
            else
                return accumulateSingleScaleBlockNEON(sum, dot, w, scale, P);
        }

        /**
         * @brief Consume one reusable panel for a row or pair, preserving its canonical K tree.
         * @param packed Packed INT8 groups and widened FP32 factors, in ascending logical K order.
         * @param count Blocks in the current panel (zero for a canonical empty partition).
         * @param a,a1 First block of each live activation row for the current panel.
         * @param out,out1 Destination strips in original row order; only nc lanes are writable.
         * @param nc Actual live N lanes in [1,4].
         * @param carry Worker-private unfinished ordered partition, present only when split.
         * @param start,finish Whether the current panel starts/ends its canonical arithmetic interval.
         * @param fold Whether an ordered preceding partition is already resident in the destination.
         */
        template<int Rows, typename Ops, Policy P, bool Minimum>
        [[gnu::noinline]] LLAMINAR_NEON_PREFILL_EXACT static void compute(const float *packed, int count,
            const Q8_1Block *a, const Q8_1Block *a1, float *out, float *out1, int nc, float *carry,
            bool start, bool finish, bool fold) noexcept
        {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
            constexpr bool ordered = P == Policy::GPUAlignedExpert;
            auto v = start ? vdupq_n_f32(0) : loadOutput(ordered ? carry : out, ordered ? 4 : nc);
            auto w = vdupq_n_f32(0);
            if constexpr (Rows == 2)
                if (!start) w = loadOutput(ordered ? carry + 4 : out1, ordered ? 4 : nc);
            for (int b = 0; b < count; ++b, packed += block_floats)
            {
                int32x4_t x, y;
                blockDots<Rows, Ops>(packed, a + b, a1 + (Rows == 2 ? b : 0), x, y);
                v = update<P, Minimum>(v, x, packed, a[b]);
                if constexpr (Rows == 2) w = update<P, Minimum>(w, y, packed, a1[b]);
            }
            if constexpr (ordered)
            {
                if (!finish)
                {
                    vst1q_f32(carry, v);
                    if constexpr (Rows == 2) vst1q_f32(carry + 4, w);
                    return;
                }
                if (fold)
                {
                    v = vaddq_f32(loadOutput(out, nc), v);
                    if constexpr (Rows == 2) w = vaddq_f32(loadOutput(out1, nc), w);
                }
            }
            storeOutput(out, v, nc);
            if constexpr (Rows == 2) storeOutput(out1, w, nc);
        }

        /**
         * @brief Consume one panel of whole canonical ordered partitions, folding them in registers.
         * @param packed Packed INT8 groups and widened FP32 factors of blocks [first, first+count).
         * @param first Absolute first block of the panel, a multiple of @p per_partition.
         * @param count Blocks in the panel, whole partitions except the final ragged one.
         * @param per_partition Canonical blocks per ordered partition (at most one panel deep).
         * @param empties Empty trailing canonical partitions to fold after the last block (final panel only).
         * @param a,a1 First block of each live activation row for the current panel.
         * @param out,out1 Destination strips in original row order; only nc lanes are writable.
         * @param nc Actual live N lanes in [1,4].
         * @details The GPU-aligned tree forms every partition from +0 and folds
         * it as total = total + partial, partition 0 seeding the total, then
         * folds +0 once per empty trailing partition. compute() executes that
         * tree one partition per call through the destination; this executes
         * the identical operations with the same operands in the same order
         * while the total stays in a register, publishing it once per panel.
         * A later panel resumes from the published FP32 total, which a store
         * and reload return bit-exactly.
         */
        template<int Rows, typename Ops, bool Minimum>
        [[gnu::noinline]] LLAMINAR_NEON_PREFILL_EXACT static void computeOrdered(const float *packed, int first,
            int count, int per_partition, int empties, const Q8_1Block *a, const Q8_1Block *a1,
            float *out, float *out1, int nc) noexcept
        {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
            constexpr Policy P = Policy::GPUAlignedExpert;
            auto t = first ? loadOutput(out, nc) : vdupq_n_f32(0);
            auto u = vdupq_n_f32(0);
            if constexpr (Rows == 2)
                if (first) u = loadOutput(out1, nc);
            for (int b = 0; b < count; )
            {
                const int begin = b, end = std::min(count, b + per_partition);
                auto v = vdupq_n_f32(0), w = vdupq_n_f32(0);
                for (; b < end; ++b, packed += block_floats)
                {
                    int32x4_t x, y;
                    blockDots<Rows, Ops>(packed, a + b, a1 + (Rows == 2 ? b : 0), x, y);
                    v = update<P, Minimum>(v, x, packed, a[b]);
                    if constexpr (Rows == 2) w = update<P, Minimum>(w, y, packed, a1[b]);
                }
                // Partition 0 seeds the total; every later one folds into it.
                if (first + begin == 0)
                {
                    t = v;
                    if constexpr (Rows == 2) u = w;
                }
                else
                {
                    t = vaddq_f32(t, v);
                    if constexpr (Rows == 2) u = vaddq_f32(u, w);
                }
            }
            for (int e = 0; e < empties; ++e)
            {
                t = vaddq_f32(t, vdupq_n_f32(0));
                if constexpr (Rows == 2) u = vaddq_f32(u, vdupq_n_f32(0));
            }
            storeOutput(out, t, nc);
            if constexpr (Rows == 2) storeOutput(out1, u, nc);
        }

        /**
         * @brief Exact INT32 dot of one packed block with one row or a row pair.
         * @param packed Packed INT8 groups of the block (SMMLA-zipped for row-pair families).
         * @param a0 First row's activation block.
         * @param a1 Second row's activation block; read only for a row pair.
         * @param x,y Receive the first/second row's four column dots.
         */
        template<int Rows, typename Ops>
        [[gnu::always_inline]] static void blockDots(const float *packed, const Q8_1Block *a0, const Q8_1Block *a1,
            int32x4_t &x, int32x4_t &y) noexcept
        {
            using I = typename Ops::I;
            x = vdupq_n_s32(0); y = vdupq_n_s32(0);
#if defined(__ARM_FEATURE_MATMUL_INT8)
            if constexpr (multi_scale::usesRowPairs<Ops>)
            {
                auto lo = vdupq_n_s32(0), hi = vdupq_n_s32(0);
                Tile::forEachGroup<4>([&]<int G>() __attribute__((always_inline)) {
                    const auto q = [&]() __attribute__((always_inline)) {
                        if constexpr (Rows == 2)
                            return std::bit_cast<int8x16_t>(Ops::activationPair(a0->qs + 8 * G, a1->qs + 8 * G));
                        else
                            return vcombine_s8(vld1_s8(a0->qs + 8 * G), vdup_n_s8(0));
                    }();
                    lo = vmmlaq_s32(lo, q, dotOperand(packed, 2 * G));
                    hi = vmmlaq_s32(hi, q, dotOperand(packed, 2 * G + 1));
                });
                splitRowPairI8MM(lo, hi, x, y);
            }
            else
#endif
            {
                const auto row0 = Ops::activationBlock(a0->qs);
                typename Ops::ActivationBlock row1{};
                if constexpr (Rows == 2) row1 = Ops::activationBlock(a1->qs);
                I d0{}, d1{};
                Tile::forEachGroup<8>([&]<int G>() __attribute__((always_inline)) {
                    const auto weight = std::bit_cast<I>(dotOperand(packed, G));
                    d0 = Ops::template dotLane<G>(d0, weight, row0);
                    if constexpr (Rows == 2) d1 = Ops::template dotLane<G>(d1, weight, row1);
                });
                x = std::bit_cast<int32x4_t>(d0);
                y = std::bit_cast<int32x4_t>(d1);
            }
        }

        /**
         * @brief Add the retained bias to one completed chunk of every active row.
         * @param chunk Physical 64-column chunk whose complete K results are final.
         * @details The register tile route adds bias after a row's full K tree:
         * out = out + bias per lane. This is the same FP32 operation with the
         * same operand order, so attaching a bias never changes the route's bytes.
         */
        LLAMINAR_NEON_PREFILL_EXACT void addBias(int chunk) const noexcept
        {
            const auto &w = batch_.weights_;
            const int columns = std::min(64, w.N - 64 * chunk);
            const float *bias = batch_.bias_.data() + std::size_t(chunk) * 64;
            const int pairs = batch_.grid_.tiles(2);
            for (int pair = 0; pair < pairs; ++pair)
            {
                const auto rows = batch_.grid_.tileSpan(pair, 2);
                for (int row = rows.first; row < rows.end; ++row)
                {
                    float *out = batch_.output_.data() + std::size_t(row) * batch_.stride_ + std::size_t(chunk) * 64;
                    int column = 0;
                    for (; column + 4 <= columns; column += 4)
                        vst1q_f32(out + column, vaddq_f32(vld1q_f32(out + column), vld1q_f32(bias + column)));
                    for (; column < columns; ++column) out[column] += bias[column];
                }
            }
        }

        /**
         * @brief Stream private K panels of one 64-column chunk across all useful row pairs.
         * @tparam Instruction Admitted native dot family.
         * @tparam C IQ3_S (11), Q5_K (7), or Q5_0 (0).
         * @tparam P Immutable projection arithmetic identity.
         * @param chunk Physical output chunk owned by this task.
         * @param panel The calling worker's slot: one weight panel, then its ordered carry.
         * @details Each four-column strip decodes one K panel once and reuses it
         * for every row pair, so native payload decode is paid once per weight
         * tile rather than once per row tile. Rows keep their own increasing-K
         * (or canonical ordered-partition) tree; the strip/panel order changes no byte.
         */
        template<Dot Instruction, int C, Policy P> void runChunk(int chunk, float *panel) const noexcept
        {
            using Ops = multi_scale::NEONMultiScaleOps<Instruction>;
            const auto &w = batch_.weights_;
            const auto *a = batch_.activation_.data();
            auto *out = batch_.output_.data();
            const int stride = batch_.stride_;
            const int pairs = batch_.grid_.tiles(2);
            const int partitions = Tile::partitionsFor(w);
            const int per_partition = (w.blocks_per_row + partitions - 1) / partitions;
            auto *carry = neon_panel::needsCarry(w.K, P) ? panel + neon_panel::panel_floats : nullptr;
            for (int c = 0; c < std::min(64, w.N - 64 * chunk); c += 4)
            {
                const int n = 64 * chunk + c, nc = std::min(4, w.N - n);
                if constexpr (P == Policy::GPUAlignedExpert)
                {
                    if (per_partition <= neon_panel::depth)
                    {
                        // Whole canonical partitions per panel: one decode of
                        // up to sixteen blocks, then one in-register ordered
                        // fold per row pair instead of one call (and one
                        // destination round trip) per partition. Only trees
                        // whose partition outgrows a panel (K > 8192) take
                        // the carry program below.
                        const int span = per_partition * (neon_panel::depth / per_partition);
                        const int filled = (w.blocks_per_row + per_partition - 1) / per_partition;
                        for (int b = 0; b < w.blocks_per_row; )
                        {
                            const int last = std::min(w.blocks_per_row, b + span);
                            pack<C, Ops>(w, chunk, c, b, last, panel);
                            const int empties = last == w.blocks_per_row ? partitions - filled : 0;
                            for (int pair = 0; pair < pairs; ++pair)
                            {
                                const auto rows = batch_.grid_.tileSpan(pair, 2);
                                if (!rows.size()) continue;
                                const auto r = std::size_t(rows.first);
                                const auto *q = a + r * w.blocks_per_row + b;
                                auto *o = out + r * stride + n;
                                if (rows.size() == 2)
                                    computeOrdered<2, Ops, C == 7>(panel, b, last - b, per_partition, empties,
                                        q, q + w.blocks_per_row, o, o + stride, nc);
                                else
                                    computeOrdered<1, Ops, C == 7>(panel, b, last - b, per_partition, empties,
                                        q, nullptr, o, nullptr, nc);
                            }
                            b = last;
                        }
                        continue;
                    }
                }
                for (int p = 0; p < partitions; ++p)
                {
                    // The legacy fold includes empty trailing partitions. They
                    // contribute +0 (including FP sign/payload behavior).
                    const int first = std::min(p * per_partition, w.blocks_per_row);
                    const int end = std::min(first + per_partition, w.blocks_per_row);
                    for (int b = first;;)
                    {
                        const int last = b + std::min(neon_panel::depth, end - b);
                        if (last > b) pack<C, Ops>(w, chunk, c, b, last, panel);
                        for (int pair = 0; pair < pairs; ++pair)
                        {
                            const auto rows = batch_.grid_.tileSpan(pair, 2);
                            if (!rows.size()) continue;
                            const auto r = std::size_t(rows.first);
                            const auto *q = a + r * w.blocks_per_row + b;
                            auto *o = out + r * stride + n;
                            auto *s = carry ? carry + r * 4 : nullptr;
                            if (rows.size() == 2)
                                compute<2, Ops, P, C == 7>(panel, last - b, q, q + w.blocks_per_row, o, o + stride, nc,
                                    s, b == first, last == end, p != 0);
                            else
                                compute<1, Ops, P, C == 7>(panel, last - b, q, nullptr, o, nullptr, nc,
                                    s, b == first, last == end, p != 0);
                        }
                        if (last == end) break;
                        b = last;
                    }
                }
            }
            if (!batch_.bias_.empty()) addBias(chunk);
        }

        /** @brief Resolve the two different arithmetic programs for one chunk task. */
        template<Dot Instruction, int C> void dispatch(int chunk, float *panel) const noexcept
        {
            if (batch_.weights_.numerical_policy == Policy::GPUAlignedExpert)
                return runChunk<Instruction, C, Policy::GPUAlignedExpert>(chunk, panel);
            runChunk<Instruction, C, Policy::BackendNative>(chunk, panel);
        }

        const CPUNativeNEONProjectionBatch &batch_; ///< Synchronous original-coordinate binding.
        float *scratch_; ///< Stage-owned slab; disjoint per-worker slots.
        std::size_t worker_floats_; ///< 64-byte-aligned size of each slot.
        int workers_; ///< Admitted slots, indexed by active-team worker id.
    };
#undef LLAMINAR_NEON_PREFILL_EXACT
}
