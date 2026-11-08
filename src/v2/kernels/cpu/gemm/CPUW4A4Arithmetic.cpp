/**
 * @file CPUW4A4Arithmetic.cpp
 * @brief Exact native CPU A4 preparation and weight-sharing W4A4/H projections.
 *
 * Source policy and all scratch are admitted before invocation. Independent
 * transform columns may run in parallel, but each Paley output retains ascending
 * source-row additions. A4 group extrema are published before the whole-token
 * range is chosen. The original range survives separately from its rounded unit;
 * reconstructing it would change ceil decisions. Integer group shifts precede
 * all output reduction, making token tiles and OpenMP scheduling byte invariant.
 */
#include "CPUW4A4Arithmetic.h"

#include "utils/Assertions.h"
#include "utils/OpenMPUtils.h"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <limits>
#include <source_location>

namespace llaminar2::w4a4
{
    namespace
    {
        /**
         * @brief Keep admission and publication invariants active in Release too.
         * @param condition Required ownership, extent or arithmetic invariant.
         * @param message Precise diagnostic without success-path string construction.
         * @param location Original caller retained by the shared assertion authority.
         */
        void require(bool condition, const char *message,
                     std::source_location location = std::source_location::current())
        {
            if (!condition)
                throwAssertionError(message, location.file_name(), static_cast<int>(location.line()),
                                    location.function_name());
        }

        /**
         * @brief Checked element product shared by declaration and invocation validation.
         * @param rows Number of live rows.
         * @param columns Elements per row.
         * @param element_bytes Physical width of one typed element.
         * @return Element count whose byte extent is representable.
         */
        std::size_t product(std::size_t rows, std::size_t columns, std::size_t element_bytes)
        {
            require(element_bytes && columns <= std::numeric_limits<std::size_t>::max() / element_bytes,
                       "W4A4 workspace row extent overflows size_t");
            const auto row_bytes = columns * element_bytes;
            require(!row_bytes || rows <= std::numeric_limits<std::size_t>::max() / row_bytes,
                       "W4A4 workspace extent overflows size_t");
            return rows * columns;
        }

        /**
         * @brief Exact live strided matrix extent, excluding final-row padding.
         * @param rows Number of live rows, including the empty-work case.
         * @param width Complete logical row width.
         * @param stride Physical leading dimension in floats.
         * @return Required FP32 element extent with overflow rejected.
         */
        std::size_t liveExtent(int rows, std::size_t width, std::size_t stride)
        {
            require(rows >= 0 && stride >= width, "W4A4 invocation has invalid rows or leading dimension");
            if (!rows) return 0;
            const auto prefix = product(static_cast<std::size_t>(rows - 1), stride, sizeof(float));
            require(prefix <= std::numeric_limits<std::size_t>::max() / sizeof(float) - width,
                       "W4A4 strided extent overflows size_t");
            return prefix + width;
        }

        /** @brief A checked byte extent used only to reject overlapping borrowed buffers. */
        struct ByteRange
        {
            std::uintptr_t begin; ///< Native pointer identity; never used as an allocation ledger.
            std::size_t bytes;    ///< Exact live prefix, not arena capacity.
        };

        /** @return A non-owning byte range for one typed, already checked live span. */
        template<class T>
        ByteRange bytes(std::span<T> values)
        {
            return {reinterpret_cast<std::uintptr_t>(values.data()), values.size_bytes()};
        }

        /**
         * @brief Reject aliasing without overflowing either end-address calculation.
         * @param left First complete live byte interval.
         * @param right Second complete live byte interval.
         */
        void requireDisjoint(ByteRange left, ByteRange right)
        {
            if (!left.bytes || !right.bytes) return;
            const bool overlaps = left.begin <= right.begin
                ? right.begin - left.begin < left.bytes : left.begin - right.begin < right.bytes;
            require(!overlaps, "W4A4 invocation requires disjoint live input, scratch and output storage");
        }

        /** @brief Reject an arithmetic environment incompatible with recorded FP32 rounding. */
        void requireRounding()
        {
            require(std::fegetround() == FE_TONEAREST, "W4A4 arithmetic requires round-to-nearest-even");
        }

        /**
         * @brief Compute several token outputs while decoding each weight nibble only once.
         * @tparam Rows Live token tile width; no inactive lane contributes work.
         * @param weights Complete admitted source row.
         * @param input First activation row in the tile, followed by complete packed rows.
         * @param blocks Native blocks per complete row.
         * @param output First owned FP32 output; subsequent tokens use the supplied stride.
         * @param stride Output leading dimension in floats.
         * @param row_indices Original offsets of the live rows within this four-row tile.
         *
         * The SIMD reduction contains bounded integers only. Every group is
         * shifted independently before accumulation; no floating dot or row
         * replay is substituted when the tile has a partial tail.
         */
        template<int Rows>
        void projectTile(const SourceBlock *weights, const ActivationBlock *input,
                         int blocks, float *output, std::size_t stride,
                         const std::array<int, 4> &row_indices)
        {
            std::int32_t totals[Rows]{};
            for (int b = 0; b < blocks; ++b)
                for (unsigned group = 0; group < kGroupsPerBlock; ++group)
                {
                    std::int32_t dots[Rows]{};
                    std::int32_t weight_sum = 0;
                    const auto *packed_weight = weights[b].codes + group * (kGroupElements / 2);
#pragma omp simd reduction(+ : weight_sum, dots[:Rows])
                    for (int pair = 0; pair < kGroupElements / 2; ++pair)
                    {
                        const int low = signedNibble(packed_weight[pair]);
                        const int high = signedNibble(packed_weight[pair] >> 4);
                        weight_sum += low + high;
                        for (int row = 0; row < Rows; ++row)
                        {
                            const auto packed_input = input[static_cast<std::size_t>(row_indices[row]) * blocks + b]
                                .codes[group * (kGroupElements / 2) + pair];
                            dots[row] += low * signedNibble(packed_input)
                                + high * signedNibble(packed_input >> 4);
                        }
                    }
                    const auto weight_multiplier = weightMultiplier(weights[b].descriptor, group);
                    for (int row = 0; row < Rows; ++row)
                    {
                        const auto &activation = input[static_cast<std::size_t>(row_indices[row]) * blocks + b];
                        const int zero = (activation.zero_points >> (4 * group)) & 15;
                        const int multiplier = (activation.multipliers >> (8 * group)) & 255;
                        totals[row] += shiftedGroupTerm(dots[row], weight_sum, zero, multiplier, weight_multiplier);
                    }
                }
            for (int row = 0; row < Rows; ++row)
            {
                // Preserve each FP32 rounding point, including the conversion
                // of the complete signed integer sum before multiplication.
                float value = static_cast<float>(totals[row]) * static_cast<float>(1 << kTermShift);
                value = value * input[static_cast<std::size_t>(row_indices[row]) * blocks].unit;
                output[static_cast<std::size_t>(row_indices[row]) * stride] = value * weights[0].unit;
            }
        }
    }

    CPUWorkspaceCounts cpuWorkspaceCounts(Transform transform, int columns, int tokens)
    {
        require(tokens >= 0, "W4A4 workspace requires a nonnegative live token count");
        require(transform == Transform::Identity || transform == Transform::PaleySylvester,
                "W4A4 workspace requires its source-declared online transform");
        require(columns > 0 && MatrixContract::sourceExtent(1, columns).has_value(),
                "W4A4 workspace requires a valid complete original reduction width");
        const auto rows = static_cast<std::size_t>(tokens);
        const auto transformed = transform == Transform::PaleySylvester
            ? product(rows, columns, sizeof(float)) : 0;
        return {
            .blocks = product(rows, columns / kBlockElements, sizeof(ActivationBlock)),
            .ranges = product(rows, columns / kGroupElements, sizeof(ActivationRange)),
            .row_ranges = product(rows, 1, sizeof(float)),
            .butterfly_floats = transformed,
            .transformed_floats = transformed,
            .active_rows = rows,
        };
    }

    CPUWorkspaceCounts cpuWorkspaceCounts(const MatrixContract &contract, int tokens)
    {
        return cpuWorkspaceCounts(contract.transform(), contract.columns(), tokens);
    }

    CPUActivations prepareCPUActivations(const MatrixContract &contract, std::span<const float> input,
                                         int tokens, std::size_t input_stride, CPUActivationBuffers workspace,
                                         std::optional<ProjectionRowWork> row_work)
    {
        requireRounding();
        const auto counts = cpuWorkspaceCounts(contract, tokens);
        if (row_work)
        {
            require(row_work->physicalRows() == tokens,
                "W4A4 work selection disagrees with original physical rows");
            for (int request = 0; request < row_work->requests(); ++request)
                static_cast<void>(row_work->resolveHost(request));
        }
        const auto input_extent = liveExtent(tokens, contract.columns(), input_stride);
        require(input.size() >= input_extent, "W4A4 activation input is incomplete");
        require(workspace.blocks.size() >= counts.blocks && workspace.ranges.size() >= counts.ranges
            && workspace.row_ranges.size() >= counts.row_ranges
            && workspace.butterfly.size() >= counts.butterfly_floats
            && workspace.transformed.size() >= counts.transformed_floats
            && (!row_work || workspace.active_rows.size() >= counts.active_rows),
            "W4A4 activation preparation requires its complete admitted workspace");
        input = input.first(input_extent);
        workspace.blocks = workspace.blocks.first(counts.blocks);
        workspace.ranges = workspace.ranges.first(counts.ranges);
        workspace.row_ranges = workspace.row_ranges.first(counts.row_ranges);
        workspace.butterfly = workspace.butterfly.first(counts.butterfly_floats);
        workspace.transformed = workspace.transformed.first(counts.transformed_floats);
        workspace.active_rows = row_work ? workspace.active_rows.first(counts.active_rows) : std::span<std::uint8_t>{};
        const std::array ranges{bytes(input), bytes(workspace.blocks), bytes(workspace.ranges),
            bytes(workspace.row_ranges), bytes(workspace.butterfly), bytes(workspace.transformed),
            bytes(workspace.active_rows)};
        for (std::size_t i = 0; i < ranges.size(); ++i)
            for (std::size_t j = i + 1; j < ranges.size(); ++j) requireDisjoint(ranges[i], ranges[j]);

        const int width = contract.columns(), groups = contract.groupsPerRow(), blocks = contract.blocksPerRow();
        const bool transformed = contract.transform() == Transform::PaleySylvester;
        // The bounded residue set is immutable arithmetic metadata, not a
        // materialized weight or a per-token heap workspace.
        std::array<bool, 128> residues{};
        if (transformed && contract.hadamard()->paley > 1)
        {
            const int prime = contract.hadamard()->paley - 1;
            for (int i = 1; i < prime; ++i) residues[(i * i) % prime] = true;
        }
        const auto active = [&](int row) { return !row_work || workspace.active_rows[row] != 0; };
        auto work = [&]() {
            requireRounding();
            if (row_work)
            {
                // This is validity of the produced A4 bank, not another model
                // live-count authority. A later controller write cannot admit
                // a row whose operand was never produced by this invocation.
#pragma omp for schedule(static)
                for (int row = 0; row < tokens; ++row)
                    workspace.active_rows[row] = static_cast<std::uint8_t>(
                        row_work->resolveHost(row_work->requestForRow(row)).contains(row));
            }
            if (transformed)
            {
                const auto geometry = *contract.hadamard();
                const int parts = width / geometry.length;
                const int p = geometry.paley, m = geometry.sylvester;
#pragma omp for collapse(3) schedule(static)
                for (int row = 0; row < tokens; ++row)
                    for (int part = 0; part < parts; ++part)
                        for (int chunk = 0; chunk < p; ++chunk)
                        {
                            if (!active(row)) continue;
                            const auto offset = static_cast<std::size_t>(part) * geometry.length + chunk * m;
                            auto *destination = workspace.butterfly.data() + static_cast<std::size_t>(row) * width + offset;
                            std::copy_n(input.data() + static_cast<std::size_t>(row) * input_stride + offset, m, destination);
                            for (int stride = 1; stride < m; stride *= 2)
                                for (int begin = 0; begin < m; begin += 2 * stride)
                                {
#pragma omp simd
                                    for (int j = begin; j < begin + stride; ++j)
                                    {
                                        const float low = destination[j], high = destination[j + stride];
                                        destination[j] = low + high;
                                        destination[j + stride] = low - high;
                                    }
                                }
                        }
                // The preceding producer barrier publishes every Sylvester row.
                const float scale = 1.0f / std::sqrt(static_cast<float>(geometry.length));
#pragma omp for collapse(3) schedule(static)
                for (int row = 0; row < tokens; ++row)
                    for (int part = 0; part < parts; ++part)
                        for (int column = 0; column < p; ++column)
                        {
                            if (!active(row)) continue;
                            const auto offset = static_cast<std::size_t>(row) * width + part * geometry.length;
                            const auto *source = workspace.butterfly.data() + offset;
                            auto *destination = workspace.transformed.data() + offset + column * m;
                            if (p == 1)
                            {
                                // Skipping the Paley sum preserves signed zero
                                // in the native pure-Sylvester transform.
#pragma omp simd
                                for (int j = 0; j < m; ++j) destination[j] = source[j] * scale;
                                continue;
                            }
                            std::fill_n(destination, m, 0.0f);
                            for (int i = 0; i < p; ++i)
                            {
                                int difference = column - i;
                                if (difference < 0) difference += p - 1;
                                const bool positive = i == column || i == 0
                                    || (column != 0 && residues[difference]);
                                if (positive)
                                {
#pragma omp simd
                                    for (int j = 0; j < m; ++j) destination[j] += source[i * m + j];
                                }
                                else
                                {
#pragma omp simd
                                    for (int j = 0; j < m; ++j) destination[j] -= source[i * m + j];
                                }
                            }
#pragma omp simd
                            for (int j = 0; j < m; ++j) destination[j] *= scale;
                        }
            }
            const auto *values = transformed ? workspace.transformed.data() : input.data();
            const auto stride = transformed ? static_cast<std::size_t>(width) : input_stride;
#pragma omp for collapse(2) schedule(static)
            for (int row = 0; row < tokens; ++row)
                for (int group = 0; group < groups; ++group)
                {
                    if (!active(row)) continue;
                    float lower = 0.0f, upper = 0.0f;
                    const auto *source = values + static_cast<std::size_t>(row) * stride + group * kGroupElements;
#pragma omp simd reduction(min : lower) reduction(max : upper)
                    for (int i = 0; i < kGroupElements; ++i)
                    {
                        lower = std::min(lower, source[i]);
                        upper = std::max(upper, source[i]);
                    }
                    workspace.ranges[static_cast<std::size_t>(row) * groups + group] =
                        {lower * contract.activationClip(), upper * contract.activationClip()};
                }
#pragma omp for schedule(static)
            for (int row = 0; row < tokens; ++row)
            {
                if (!active(row)) continue;
                float widest = 0.0f;
                for (int group = 0; group < groups; ++group)
                {
                    const auto range = workspace.ranges[static_cast<std::size_t>(row) * groups + group];
                    widest = std::max(widest, range.upper - range.lower);
                }
                if (widest == 0.0f) widest = 2.0f;
                const float unit = widest / static_cast<float>(15 * kActivationMultiplierMax);
                require(std::isfinite(unit) && unit > 0.0f, "W4A4 token unit exceeds its FP32 arithmetic range");
                workspace.row_ranges[row] = widest;
                for (int block = 0; block < blocks; ++block)
                    workspace.blocks[static_cast<std::size_t>(row) * blocks + block].unit = unit;
            }
            // A packed block is one writer's unit, avoiding metadata races
            // between its four groups while retaining parallel M=1 preparation.
#pragma omp for collapse(2) schedule(static)
            for (int row = 0; row < tokens; ++row)
                for (int block = 0; block < blocks; ++block)
                {
                    if (!active(row)) continue;
                    auto &output = workspace.blocks[static_cast<std::size_t>(row) * blocks + block];
                    output.zero_points = kActivationGroupTag;
                    output.multipliers = 0;
                    output.rotation_length = contract.rotationLength();
                    for (int group = 0; group < kGroupsPerBlock; ++group)
                    {
                        const int row_group = block * kGroupsPerBlock + group;
                        const auto range = workspace.ranges[static_cast<std::size_t>(row) * groups + row_group];
                        const float multiplier = std::clamp(std::ceil(
                            ((range.upper - range.lower) * static_cast<float>(kActivationMultiplierMax))
                                / workspace.row_ranges[row]), 1.0f, static_cast<float>(kActivationMultiplierMax));
                        const float inverse = 1.0f / (multiplier * output.unit);
                        require(std::isfinite(inverse), "W4A4 group inverse exceeds its FP32 arithmetic range");
                        const float zero = std::clamp(std::rint(-range.lower * inverse), 0.0f, 15.0f);
                        require(zero >= 0.0f && zero <= 15.0f, "W4A4 zero point is not representable");
                        output.zero_points |= static_cast<std::uint32_t>(zero) << (4 * group);
                        output.multipliers |= static_cast<std::uint32_t>(multiplier) << (8 * group);
                        const auto *source = values + static_cast<std::size_t>(row) * stride + row_group * kGroupElements;
                        for (int pair = 0; pair < kGroupElements / 2; ++pair)
                        {
                            const float low = std::clamp(std::rint(source[2 * pair] * inverse) + zero, 0.0f, 15.0f);
                            const float high = std::clamp(std::rint(source[2 * pair + 1] * inverse) + zero, 0.0f, 15.0f);
                            // This guards the float-to-int conversion itself,
                            // including NaN, without adding a separate input scan.
                            require(low >= 0.0f && low <= 15.0f && high >= 0.0f && high <= 15.0f,
                                       "W4A4 activation code is not representable");
                            output.codes[group * (kGroupElements / 2) + pair] = static_cast<std::uint8_t>(
                                ((static_cast<int>(low) - 8) & 15) | (((static_cast<int>(high) - 8) & 15) << 4));
                        }
                    }
                }
        };
        OMP_WORKSHARE_REGION(work);
        return CPUActivations(contract, tokens, workspace.blocks, workspace.active_rows);
    }

    void projectCPU(const SourceView &weights, const CPUActivations &activations,
                    std::span<float> output, std::size_t output_stride)
    {
        requireRounding();
        const auto &contract = weights.contract();
        require(activations.matches(contract), "W4A4 projection received a different A4 arithmetic policy");
        const auto extent = liveExtent(activations.rows(), contract.rows(), output_stride);
        require(output.size() >= extent, "W4A4 projection output is incomplete");
        output = output.first(extent);
        requireDisjoint(bytes(output), bytes(activations.blocks()));
        requireDisjoint(bytes(output), bytes(activations.active_rows_));
        requireDisjoint(bytes(output), bytes(weights.blocks()));
        requireDisjoint(bytes(activations.blocks()), bytes(weights.blocks()));
        const int blocks = contract.blocksPerRow(), columns = contract.rows();
        const int tiles = activations.rows() / 4 + (activations.rows() % 4 != 0);
        auto work = [&]() {
            requireRounding();
#pragma omp for collapse(2) schedule(static)
            for (int tile = 0; tile < tiles; ++tile)
                for (int column = 0; column < columns; ++column)
                {
                    const int first = tile * 4;
                    std::array<int, 4> row_indices{};
                    int live = 0;
                    for (int row = 0; row < std::min(4, activations.rows() - first); ++row)
                        if (activations.active(first + row)) row_indices[live++] = row;
                    if (!live) continue;
                    const auto *source = weights.blocks().data() + static_cast<std::size_t>(column) * blocks;
                    const auto *input = activations.blocks().data() + static_cast<std::size_t>(first) * blocks;
                    auto *destination = output.data() + static_cast<std::size_t>(first) * output_stride + column;
                    switch (live)
                    {
                    case 1: projectTile<1>(source, input, blocks, destination, output_stride, row_indices); break;
                    case 2: projectTile<2>(source, input, blocks, destination, output_stride, row_indices); break;
                    case 3: projectTile<3>(source, input, blocks, destination, output_stride, row_indices); break;
                    case 4: projectTile<4>(source, input, blocks, destination, output_stride, row_indices); break;
                    default: require(false, "W4A4 projection selected an invalid live tile");
                    }
                }
        };
        OMP_WORKSHARE_REGION(work);
    }
} // namespace llaminar2::w4a4
