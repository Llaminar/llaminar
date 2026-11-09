/**
 * @file CPUNativeNEONProjectionBatch.h
 * @brief Original-coordinate native ARM quantized projection worksharing.
 *
 * One synchronous binding borrows immutable prepared weights, persistent Q8
 * rows, optional bias and exclusive output storage. Independent row/column
 * tiles share native decoding across two/four rows without changing any row's
 * K arithmetic. Existing request publications determine useful original spans;
 * they are neither compacted nor copied into another execution authority.
 * This layer owns no allocation, quantization, weight preparation or publication.
 */
#pragma once
#include "CPUNativeNEONPreparedProjection.h"
#include "CPUProjectionRowGrid.h"
#include "CPUProjectionStorageProof.h"
#include "utils/OpenMPUtils.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace llaminar2::cpu::native_quantised
{
    /** @brief Explicit native weight-reuse candidates, independent of arithmetic policy. */
    enum class NEONProjectionRowTile { Pair, Four };

    /** @brief Checked synchronous worksharing invocation; owners outlive execute(). */
    class CPUNativeNEONProjectionBatch final
    {
    public:
        /**
         * @brief Check one complete original-coordinate invocation before worker writes.
         * @param weights Immutable canonical prepared publication, retained by its owner.
         * @param activation Exactly rows complete Q8 K rows in original physical order.
         * @param output Exact span through the last live output, including inter-row strides.
         * @param rows Positive physical M; never a compacted useful row count.
         * @param output_stride Physical output row stride in floats, at least N.
         * @param row_tile Explicit native shared-weight row candidate.
         * @param work Optional original request publication, immutable during execution.
         * @param bias Empty or exactly N immutable FP32 bias values.
         * @param n_block_chunks Positive adjacent physical 64-column chunks per task.
         * @return Borrowed checked binding, with no buffer or live-word copies.
         * @throws std::invalid_argument For geometry, lifetime-envelope or alias conflicts.
         * @throws std::overflow_error For unrepresentable native span extents.
         */
        [[nodiscard]] static CPUNativeNEONProjectionBatch bind(CPUNativeQuantisedWeightView weights,
            std::span<const Q8_1Block> activation, std::span<float> output,
            int rows, int output_stride, NEONProjectionRowTile row_tile,
            const ProjectionRowWork *work = nullptr, std::span<const float> bias = {}, int n_block_chunks = 1)
        {
            const CPUProjectionRowGrid grid(rows, work);
            const int width = [&] {
                switch (row_tile)
                {
                case NEONProjectionRowTile::Pair: return 2;
                case NEONProjectionRowTile::Four: return 4;
                }
                throw std::invalid_argument("Native NEON projection needs a known row tile");
            }();
            if (n_block_chunks <= 0 || output_stride < weights.N ||
                !activation.data() || !output.data() ||
                activation.size() != extent(rows, weights.blocks_per_row, 0) ||
                output.size() != extent(rows - 1, output_stride, weights.N) ||
                (!bias.empty() && (!bias.data() || bias.size() != std::size_t(weights.N))))
                throw std::invalid_argument("Native NEON batch requires exact physical input/output and bias spans");
            if (activation.size() > std::numeric_limits<std::size_t>::max() / sizeof(Q8_1Block) ||
                output.size() > std::numeric_limits<std::size_t>::max() / sizeof(float))
                throw std::overflow_error("Native NEON batch byte extent is unrepresentable");
            if (overlaps(output.data(), output.size_bytes(), activation.data(), activation.size_bytes()) ||
                overlaps(output.data(), output.size_bytes(), weights.bytes().data(), weights.bytes().size()) ||
                overlaps(output.data(), output.size_bytes(), bias.data(), bias.size_bytes()))
                throw std::invalid_argument("Native NEON batch output aliases a retained read-only publication");
            storage_proof::requireOriginalWorkReadOnly({output.data(), output.size_bytes()}, work);
            // Admit auto-generated code as well as the explicitly selected dot
            // family before spawning a team. A narrower diagnostic route does
            // not narrow a DotProd compiler envelope.
            ARMCPUCapabilities::native().requireCompiledEnvelope();
            return CPUNativeNEONProjectionBatch(weights, activation, output, output_stride, width, grid, bias, n_block_chunks);
        }

        /**
         * @brief Execute independent original-coordinate tiles on the current/new team.
         * @tparam Instruction Explicit ASIMD, or separately compiled/admitted DotProd or I8MM.
         * @details Every member of an existing OpenMP team enters the workshare,
         * following OMP_WORKSHARE_REGION. Fixed per-row K trees make ownership
         * independent of positive team size. Inactive rows and output padding
         * remain untouched. Metadata corruption during a tile is fatal, not
         * rollback; only successful completion may be published by the owner.
         */
        template <multi_scale::NEONByteDot Instruction = multi_scale::NEONByteDot::ASIMD>
        void execute() const
        {
            if constexpr (Instruction == multi_scale::NEONByteDot::DotProd || Instruction == multi_scale::NEONByteDot::I8MM)
                ARMCPUCapabilities::native().require(ARMCPUFeature::DotProd);
            if constexpr (Instruction == multi_scale::NEONByteDot::I8MM)
                ARMCPUCapabilities::native().require(ARMCPUFeature::IntegerMatMul);
            if (!grid_.beginWorkshare()) return;
            const std::int64_t tasks = taskCount();
            // Tasks are independent output tiles whose full-K arithmetic does
            // not depend on the executing worker, so dynamic claiming keeps
            // every byte while letting faster cores of a heterogeneous host
            // (GB10: X925 + A725) take more tiles than an equal static split.
            auto workshare = [&] {
#pragma omp for schedule(dynamic, 1)
                for (std::int64_t task = 0; task < tasks; ++task)
                    runTask<Instruction>(task);
            };
            OMP_WORKSHARE_REGION(workshare);
        }

        /**
         * @brief Number of independent (row tile, chunk group) tasks.
         * @return Zero when no original row is active; otherwise the exact count.
         */
        [[nodiscard]] std::int64_t taskCount() const
        {
            if (!grid_.hasAny()) return 0;
            return std::int64_t(grid_.tiles(width_)) * chunkGroups();
        }

        /**
         * @brief Execute one independent task; no synchronization, no shared writes.
         * @details Lets a caller pool the tasks of several independent bindings
         * into one dynamic workshare (one barrier instead of one per member).
         * Each task writes only its own output tile with the same full-K
         * arithmetic, so pooling changes no byte.
         * @param task Index in [0, taskCount()).
         */
        template <multi_scale::NEONByteDot Instruction = multi_scale::NEONByteDot::ASIMD>
        void runTask(std::int64_t task) const
        {
            const int chunks = weights_.N_padded / 64;
            const auto n_blocks = chunkGroups();
            const auto span = grid_.tileSpan(static_cast<int>(task / n_blocks), width_);
            const auto first = task % n_blocks * n_block_chunks_;
            const auto end = std::min<std::int64_t>(chunks, first + n_block_chunks_);
            for (auto current = first; current < end; ++current)
            {
                const int chunk = static_cast<int>(current);
                switch (span.size())
                {
                case 0: break;
                case 1: tile<1, Instruction>(span.first, chunk); break;
                case 2: tile<2, Instruction>(span.first, chunk); break;
                case 3: tile<3, Instruction>(span.first, chunk); break;
                case 4: tile<4, Instruction>(span.first, chunk); break;
                default: throw std::logic_error("Native NEON task exceeds its bound shared-row tile");
                }
            }
        }

    private:
        /** @return Ceil(physical 64-column chunks / chunks per task). */
        [[nodiscard]] std::int64_t chunkGroups() const noexcept
        {
            const int chunks = weights_.N_padded / 64;
            return (std::int64_t(chunks) + n_block_chunks_ - 1) / n_block_chunks_;
        }

        /** @brief Seal only immutable coordinates and borrowed publication spans.
         * @param weights Checked resident prepared stream.
         * @param activation Original physical Q8 bank.
         * @param output Exclusive physical output bank.
         * @param stride Output row stride in floats.
         * @param width Exact selected native reuse width.
         * @param grid Existing original-coordinate row authority.
         * @param bias Empty or complete immutable bias row.
         * @param n_block_chunks Positive adjacent chunks owned by each task.
         */
        CPUNativeNEONProjectionBatch(CPUNativeQuantisedWeightView weights,
            std::span<const Q8_1Block> activation, std::span<float> output,
            int stride, int width, CPUProjectionRowGrid grid, std::span<const float> bias, int n_block_chunks)
            : weights_(weights), activation_(activation), output_(output), stride_(stride),
              width_(width), grid_(grid), bias_(bias), n_block_chunks_(n_block_chunks) {}

        /** @brief Form a checked element extent without overflowing signed geometry.
         * @param rows Nonnegative preceding row count.
         * @param stride Positive physical elements per preceding row.
         * @param tail Nonnegative final live extent.
         * @return rows*stride+tail in representable native element units.
         * @throws std::invalid_argument For negative/empty geometry.
         * @throws std::overflow_error For an unrepresentable extent.
         */
        static std::size_t extent(int rows, int stride, int tail)
        {
            if (rows < 0 || stride <= 0 || tail < 0)
                throw std::invalid_argument("Native NEON batch extent needs nonnegative rows and positive stride");
            if (std::size_t(rows) > (std::numeric_limits<std::size_t>::max() - std::size_t(tail)) / std::size_t(stride))
                throw std::overflow_error("Native NEON batch extent exceeds native address space");
            return std::size_t(rows) * std::size_t(stride) + std::size_t(tail);
        }

        /** @brief Reject cross-tile aliases using whole invocation envelopes.
         * @param a First byte root.
         * @param a_bytes First bounded byte extent.
         * @param b Second byte root.
         * @param b_bytes Second bounded byte extent.
         * @return Whether nonempty storage ranges overlap, without overflowing ends.
         */
        static bool overlaps(const void *a, std::size_t a_bytes, const void *b, std::size_t b_bytes) noexcept
        {
            if (!a_bytes || !b_bytes) return false;
            const auto first = reinterpret_cast<std::uintptr_t>(a), second = reinterpret_cast<std::uintptr_t>(b);
            return first <= second ? second - first < a_bytes : first - second < b_bytes;
        }

        /** @brief Dispatch one useful bounded row tile with original physical addresses.
         * @tparam Rows Useful original contiguous rows in [1,4].
         * @tparam Instruction Authenticated native dot instruction family.
         * @param first Original slice-relative first row; never compacted.
         * @param chunk Physical 64-column output chunk.
         */
        template <std::size_t Rows, multi_scale::NEONByteDot Instruction>
        void tile(int first, int chunk) const
        {
            const int columns = std::min(64, weights_.N - chunk * 64);
            std::array<std::span<const Q8_1Block>, Rows> input;
            std::array<std::span<float>, Rows> output;
            for (std::size_t row = 0; row < Rows; ++row)
            {
                input[row] = activation_.subspan((std::size_t(first) + row) * weights_.blocks_per_row, weights_.blocks_per_row);
                output[row] = output_.subspan((std::size_t(first) + row) * stride_ + chunk * 64, columns);
            }
            CPUNativeNEONProjectionTile<Rows>::bind(weights_, input, output, chunk).template execute<Instruction>();
            if (bias_.empty()) return;
            for (std::size_t row = 0; row < Rows; ++row)
            {
                int column = 0;
                for (; column + 4 <= columns; column += 4)
                    vst1q_f32(output[row].data() + column, vaddq_f32(vld1q_f32(output[row].data() + column),
                        vld1q_f32(bias_.data() + chunk * 64 + column)));
                for (; column < columns; ++column)
                    output[row][column] += bias_[std::size_t(chunk) * 64 + column];
            }
        }

        CPUNativeQuantisedWeightView weights_; ///< Borrowed immutable source/encoding/arithmetic identity.
        std::span<const Q8_1Block> activation_; ///< Persistent physical-M input bank.
        std::span<float> output_; ///< Exclusive live output envelope; padding stays unauthored.
        int stride_; ///< Original output row stride, never changed by useful work.
        int width_; ///< One explicit physical weight-reuse candidate.
        CPUProjectionRowGrid grid_; ///< Borrowed original publications, not copied live values.
        std::span<const float> bias_; ///< Optional retained immutable epilogue input.
        int n_block_chunks_; ///< Explicit physical N ownership only; never a K arithmetic choice.
    };
} // namespace llaminar2::cpu::native_quantised
