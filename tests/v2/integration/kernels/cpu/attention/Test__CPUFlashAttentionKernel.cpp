/**
 * @file Test__CPUFlashAttentionKernel.cpp
 * @brief Integration tests for CPUFlashAttentionKernelT mixed-precision tensor path
 */

#include <gtest/gtest.h>

#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "kernels/cpu/attention/CPUFlashAttentionLaunchPolicy.h"
#include "kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "tensors/FP16Utils.h"
#include "tensors/SIMDHelpers.h"
#include "tensors/Tensors.h"
#include "tensors/TQ4Tensor.h"
#include "tensors/TQ8Tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <omp.h>

using namespace llaminar2;

namespace
{
    static std::vector<float> makeRandom(size_t count, uint32_t seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> v(count);
        for (size_t i = 0; i < count; ++i)
        {
            v[i] = dist(rng);
        }
        return v;
    }

    /**
     * @brief Place oldest-to-newest logical rows into a wrapped physical ring.
     *
     * Non-live rows receive large deterministic poison values. A kernel that
     * accidentally treats the persistent tensor as compact, omits the ring
     * modulo, or uses logical length as the physical head stride will therefore
     * disagree with the independently packed compact-cache invocation.
     */
    static std::vector<float> placeLogicalRowsInPhysicalRing(
        const std::vector<float> &logical,
        int logical_rows,
        int physical_capacity,
        std::size_t row_columns,
        int logical_origin,
        float poison)
    {
        EXPECT_EQ(
            logical.size(),
            static_cast<std::size_t>(logical_rows) * row_columns);
        EXPECT_GT(physical_capacity, logical_rows);
        EXPECT_GE(logical_origin, 0);
        EXPECT_LT(logical_origin, physical_capacity);

        std::vector<float> physical(
            static_cast<std::size_t>(physical_capacity) * row_columns);
        for (std::size_t element = 0; element < physical.size(); ++element)
        {
            physical[element] =
                poison + static_cast<float>(element % 29U) * 0.125f;
        }

        for (int logical_row = 0; logical_row < logical_rows; ++logical_row)
        {
            const int physical_row =
                (logical_origin + logical_row) % physical_capacity;
            std::copy_n(
                logical.data() +
                    static_cast<std::size_t>(logical_row) * row_columns,
                row_columns,
                physical.data() +
                    static_cast<std::size_t>(physical_row) * row_columns);
        }
        return physical;
    }

    static float cosineSimilarity(const float *a, const float *b, size_t count)
    {
        double dot = 0.0;
        double na = 0.0;
        double nb = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            const double av = static_cast<double>(a[i]);
            const double bv = static_cast<double>(b[i]);
            dot += av * bv;
            na += av * av;
            nb += bv * bv;
        }
        if (na < 1e-20 || nb < 1e-20)
        {
            return 0.0f;
        }
        return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
    }

    static float maxAbsDiff(const float *a, const float *b, size_t count)
    {
        float m = 0.0f;
        for (size_t i = 0; i < count; ++i)
        {
            m = std::max(m, std::abs(a[i] - b[i]));
        }
        return m;
    }

    /** Restore the process OpenMP thread preference after a focused test. */
    class ScopedOpenMPThreadCount
    {
    public:
        /** @brief Select a deterministic worker team for physical-mode tests. */
        explicit ScopedOpenMPThreadCount(int threads)
            : previous_(omp_get_max_threads())
        {
            omp_set_num_threads(threads);
        }

        /** @brief Restore the preference observed before this test. */
        ~ScopedOpenMPThreadCount()
        {
            omp_set_num_threads(previous_);
        }

        ScopedOpenMPThreadCount(const ScopedOpenMPThreadCount &) = delete;
        ScopedOpenMPThreadCount &operator=(
            const ScopedOpenMPThreadCount &) = delete;

    private:
        int previous_ = 1; ///< OpenMP worker preference restored at teardown.
    };

    /**
     * @brief Prove physical cache tiling cannot alter native attention bytes.
     *
     * The first query-owned launch at the smallest compiled tile establishes the
     * optimized production-path byte oracle. Every other compiled tile is then
     * exercised through query ownership, K/V-context ownership, and the explicit
     * grouped-verifier entry point. This is not a scalar or row-replay oracle:
     * every side invokes the same vectorized native-format kernel while varying
     * only physical cache blocking and work ownership.
     *
     * A non-power-of-two sliding window is particularly important here. It makes
     * the first visible K/V row unaligned with physical tile boundaries and
     * therefore catches callback vector groups that are accidentally restarted
     * at candidate-dependent offsets.
     *
     * @param query FP32 query rows consumed by every invocation.
     * @param key Native-format key cache.
     * @param value Native-format value cache.
     * @param query_rows Positive decode/grouped row count.
     * @param kv_len Positive logical K/V history length.
     * @param n_heads Local query-head count.
     * @param n_kv_heads Local K/V-head count.
     * @param head_dim Elements in one attention head.
     * @param window_size Positive non-aligned sliding-window width.
     * @param format Diagnostic storage-format name.
     */
    static void expectPhysicalTilesByteExact(
        const FP32Tensor &query,
        const ITensor &key,
        const ITensor &value,
        int query_rows,
        int kv_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int window_size,
        const std::string &format)
    {
        ASSERT_GT(query_rows, 1);
        ASSERT_GT(window_size, 0);
        const std::size_t output_elements =
            static_cast<std::size_t>(query_rows) * n_heads * head_dim;
        const std::size_t output_bytes = output_elements * sizeof(float);

        CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;
        const WorkspaceRequirements requirements =
            kernel.getWorkspaceRequirements(query_rows, n_heads, head_dim);
        DeviceWorkspaceManager workspace(
            DeviceId::cpu(),
            requirements.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(workspace.allocate(requirements)) << format;
        kernel.bindWorkspace(&workspace);

        std::vector<std::uint8_t> reference;
        const auto authenticate = [&](const FP32Tensor &output,
                                      int tile,
                                      std::string_view route)
        {
            const auto *bytes = reinterpret_cast<const std::uint8_t *>(
                output.data());
            if (reference.empty())
            {
                reference.assign(bytes, bytes + output_bytes);
                return;
            }
            for (std::size_t byte = 0; byte < output_bytes; ++byte)
            {
                ASSERT_EQ(bytes[byte], reference[byte])
                    << format << " tile=" << tile << " route=" << route
                    << " first mismatch at output byte " << byte;
            }
        };

        for (const auto axis : {
                 attention::AttentionPrefillParallelAxis::QuerySequence,
                 attention::AttentionPrefillParallelAxis::KeyValueContext})
        {
            for (const int tile : cpu::fa2_policy::kCompiledKVTiles)
            {
                kernel.configureLaunchPolicy({.explicit_kv_tile = tile});
                FP32Tensor output(
                    {static_cast<std::size_t>(query_rows),
                     static_cast<std::size_t>(n_heads) * head_dim});
                ASSERT_TRUE(kernel.compute_tensor(
                    &query,
                    &key,
                    &value,
                    &output,
                    /*batch_size=*/1,
                    query_rows,
                    kv_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    /*causal=*/true,
                    window_size,
                    /*workspace_scores=*/nullptr,
                    /*workspace_mask=*/nullptr,
                    /*mpi_ctx=*/nullptr,
                    /*device_idx=*/-1,
                    /*head_start=*/0,
                    /*local_n_heads=*/-1,
                    /*local_n_kv_heads=*/-1,
                    /*gqa_n_rep=*/0,
                    {.prefill_parallel_axis = axis}))
                    << format << " tile=" << tile;
                authenticate(
                    output,
                    tile,
                    attention::attentionPrefillParallelAxisName(axis));
            }
        }

        for (const int tile : cpu::fa2_policy::kCompiledKVTiles)
        {
            kernel.configureLaunchPolicy({.explicit_kv_tile = tile});
            FP32Tensor output(
                {static_cast<std::size_t>(query_rows),
                 static_cast<std::size_t>(n_heads) * head_dim});
            ASSERT_TRUE(kernel.compute_verifier_rows_decode_equivalent(
                &query,
                &key,
                &value,
                &output,
                query_rows,
                kv_len,
                n_heads,
                n_kv_heads,
                head_dim,
                /*causal=*/true,
                window_size))
                << format << " grouped tile=" << tile;
            authenticate(output, tile, "grouped_verifier");
        }
    }

    /**
     * @brief Prove byte equality between CPU query and K/V-context ownership.
     *
     * Both calls use the public production tensor API. The context call binds
     * the same setup-owned workspace that an AttentionComputeStage binds before
     * execution; no private helper or test-only arithmetic is substituted.
     */
    static void expectPhysicalModesByteExact(
        const FP32Tensor &query,
        const ITensor &key,
        const ITensor &value,
        int query_rows,
        int kv_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        const std::string &format,
        const attention::AttentionKVLogicalView &kv_logical_view = {},
        int window_size = -1)
    {
        const std::size_t output_elements =
            static_cast<std::size_t>(query_rows) * n_heads * head_dim;
        FP32Tensor query_output(
            {static_cast<std::size_t>(query_rows),
             static_cast<std::size_t>(n_heads) * head_dim});
        FP32Tensor context_output(
            {static_cast<std::size_t>(query_rows),
             static_cast<std::size_t>(n_heads) * head_dim});
        CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;

        const auto run = [&](FP32Tensor &destination,
                             attention::AttentionPrefillParallelAxis axis)
        {
            return kernel.compute_tensor(
                &query,
                &key,
                &value,
                &destination,
                /*batch_size=*/1,
                query_rows,
                kv_len,
                n_heads,
                n_kv_heads,
                head_dim,
                /*causal=*/true,
                window_size,
                /*workspace_scores=*/nullptr,
                /*workspace_mask=*/nullptr,
                /*mpi_ctx=*/nullptr,
                /*device_idx=*/-1,
                /*head_start=*/0,
                /*local_n_heads=*/-1,
                /*local_n_kv_heads=*/-1,
                /*gqa_n_rep=*/0,
                {
                    .prefill_parallel_axis = axis,
                },
                kv_logical_view);
        };

        ASSERT_TRUE(run(
            query_output,
            attention::AttentionPrefillParallelAxis::QuerySequence))
            << format;

        const WorkspaceRequirements requirements =
            kernel.getWorkspaceRequirements(query_rows, n_heads, head_dim);
        DeviceWorkspaceManager workspace(
            DeviceId::cpu(),
            requirements.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(workspace.allocate(requirements)) << format;
        kernel.bindWorkspace(&workspace);
        ASSERT_TRUE(run(
            context_output,
            attention::AttentionPrefillParallelAxis::KeyValueContext))
            << format;

        const auto *query_bytes = reinterpret_cast<const std::uint8_t *>(
            query_output.data());
        const auto *context_bytes = reinterpret_cast<const std::uint8_t *>(
            context_output.data());
        const std::size_t output_bytes = output_elements * sizeof(float);
        for (std::size_t byte = 0; byte < output_bytes; ++byte)
        {
            ASSERT_EQ(context_bytes[byte], query_bytes[byte])
                << format << " first mismatch at output byte " << byte;
        }
    }

    /**
     * @brief Prove every native implementation excludes rows older than the window.
     *
     * `older_key` and `older_value` differ from their baseline counterparts only
     * before the oldest row visible to any query in the group. Both declared
     * physical ownership modes and the grouped verifier entry point must therefore
     * produce identical bytes. This catches a dropped window argument even when
     * two schedulers happen to share the same addressing bug.
     */
    static void expectSlidingWindowIgnoresOlderRows(
        const FP32Tensor &query,
        const ITensor &baseline_key,
        const ITensor &baseline_value,
        const ITensor &older_key,
        const ITensor &older_value,
        int query_rows,
        int kv_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int window_size,
        const std::string &format)
    {
        ASSERT_GT(window_size, 0);
        const std::size_t output_elements =
            static_cast<std::size_t>(query_rows) * n_heads * head_dim;
        CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;
        const WorkspaceRequirements requirements =
            kernel.getWorkspaceRequirements(query_rows, n_heads, head_dim);
        DeviceWorkspaceManager workspace(
            DeviceId::cpu(),
            requirements.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(workspace.allocate(requirements)) << format;
        kernel.bindWorkspace(&workspace);

        const auto run = [&](const ITensor &key,
                             const ITensor &value,
                             FP32Tensor &destination,
                             attention::AttentionPrefillParallelAxis axis)
        {
            return kernel.compute_tensor(
                &query,
                &key,
                &value,
                &destination,
                /*batch_size=*/1,
                query_rows,
                kv_len,
                n_heads,
                n_kv_heads,
                head_dim,
                /*causal=*/true,
                window_size,
                /*workspace_scores=*/nullptr,
                /*workspace_mask=*/nullptr,
                /*mpi_ctx=*/nullptr,
                /*device_idx=*/-1,
                /*head_start=*/0,
                /*local_n_heads=*/-1,
                /*local_n_kv_heads=*/-1,
                /*gqa_n_rep=*/0,
                {.prefill_parallel_axis = axis});
        };

        for (const auto axis : {
                 attention::AttentionPrefillParallelAxis::QuerySequence,
                 attention::AttentionPrefillParallelAxis::KeyValueContext})
        {
            FP32Tensor baseline_output(
                {static_cast<std::size_t>(query_rows),
                 static_cast<std::size_t>(n_heads) * head_dim});
            FP32Tensor older_output(
                {static_cast<std::size_t>(query_rows),
                 static_cast<std::size_t>(n_heads) * head_dim});
            ASSERT_TRUE(run(
                baseline_key, baseline_value, baseline_output, axis))
                << format;
            ASSERT_TRUE(run(older_key, older_value, older_output, axis))
                << format;
            ASSERT_EQ(
                std::memcmp(
                    baseline_output.data(),
                    older_output.data(),
                    output_elements * sizeof(float)),
                0)
                << format << " admitted rows older than window=" << window_size
                << " axis=" << attention::attentionPrefillParallelAxisName(axis);
        }

        if (query_rows <= 1)
            return;

        FP32Tensor baseline_grouped(
            {static_cast<std::size_t>(query_rows),
             static_cast<std::size_t>(n_heads) * head_dim});
        FP32Tensor older_grouped(
            {static_cast<std::size_t>(query_rows),
             static_cast<std::size_t>(n_heads) * head_dim});
        const auto run_grouped = [&](const ITensor &key,
                                     const ITensor &value,
                                     FP32Tensor &destination)
        {
            return kernel.compute_verifier_rows_decode_equivalent(
                &query,
                &key,
                &value,
                &destination,
                query_rows,
                kv_len,
                n_heads,
                n_kv_heads,
                head_dim,
                /*causal=*/true,
                window_size);
        };
        ASSERT_TRUE(run_grouped(
            baseline_key, baseline_value, baseline_grouped))
            << format;
        ASSERT_TRUE(run_grouped(older_key, older_value, older_grouped))
            << format;
        ASSERT_EQ(
            std::memcmp(
                baseline_grouped.data(),
                older_grouped.data(),
                output_elements * sizeof(float)),
            0)
            << format << " grouped verifier admitted rows older than window="
            << window_size;
    }

    /**
     * @brief Compare direct persistent-ring attention with compact native K/V.
     *
     * Each side is a public production-kernel invocation using the same native
     * storage format. The comparison is repeated for both physical ownership
     * policies, and grouped rows additionally use the dedicated verifier API.
     * This proves addressing without introducing dequantized or scalar oracles.
     */
    static void expectLogicalViewMatchesCompact(
        const FP32Tensor &query,
        const ITensor &compact_key,
        const ITensor &compact_value,
        const ITensor &physical_key,
        const ITensor &physical_value,
        int query_rows,
        int kv_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        const attention::AttentionKVLogicalView &kv_logical_view,
        const std::string &format,
        int window_size = -1)
    {
        const std::size_t output_elements =
            static_cast<std::size_t>(query_rows) * n_heads * head_dim;
        CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;
        const WorkspaceRequirements requirements =
            kernel.getWorkspaceRequirements(query_rows, n_heads, head_dim);
        DeviceWorkspaceManager workspace(
            DeviceId::cpu(),
            requirements.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(workspace.allocate(requirements)) << format;
        kernel.bindWorkspace(&workspace);

        const auto run = [&kernel,
                          &query,
                          query_rows,
                          kv_len,
                          n_heads,
                          n_kv_heads,
                          head_dim,
                          window_size](
                             const ITensor &key,
                             const ITensor &value,
                             FP32Tensor &destination,
                             attention::AttentionPrefillParallelAxis axis,
                             const attention::AttentionKVLogicalView &view)
        {
            return kernel.compute_tensor(
                &query,
                &key,
                &value,
                &destination,
                /*batch_size=*/1,
                query_rows,
                kv_len,
                n_heads,
                n_kv_heads,
                head_dim,
                /*causal=*/true,
                window_size,
                /*workspace_scores=*/nullptr,
                /*workspace_mask=*/nullptr,
                /*mpi_ctx=*/nullptr,
                /*device_idx=*/-1,
                /*head_start=*/0,
                /*local_n_heads=*/-1,
                /*local_n_kv_heads=*/-1,
                /*gqa_n_rep=*/0,
                {.prefill_parallel_axis = axis},
                view);
        };

        for (const auto axis : {
                 attention::AttentionPrefillParallelAxis::QuerySequence,
                 attention::AttentionPrefillParallelAxis::KeyValueContext})
        {
            FP32Tensor compact_output(
                {static_cast<std::size_t>(query_rows),
                 static_cast<std::size_t>(n_heads) * head_dim});
            FP32Tensor physical_output(
                {static_cast<std::size_t>(query_rows),
                 static_cast<std::size_t>(n_heads) * head_dim});
            ASSERT_TRUE(run(
                compact_key,
                compact_value,
                compact_output,
                axis,
                {}))
                << format;
            ASSERT_TRUE(run(
                physical_key,
                physical_value,
                physical_output,
                axis,
                kv_logical_view))
                << format;
            ASSERT_EQ(
                std::memcmp(
                    physical_output.data(),
                    compact_output.data(),
                    output_elements * sizeof(float)),
                0)
                << format << " axis="
                << attention::attentionPrefillParallelAxisName(axis);
        }

        if (query_rows <= 1)
            return;

        FP32Tensor compact_grouped(
            {static_cast<std::size_t>(query_rows),
             static_cast<std::size_t>(n_heads) * head_dim});
        FP32Tensor physical_grouped(
            {static_cast<std::size_t>(query_rows),
             static_cast<std::size_t>(n_heads) * head_dim});
        ASSERT_TRUE(kernel.compute_verifier_rows_decode_equivalent(
            &query,
            &compact_key,
            &compact_value,
            &compact_grouped,
            query_rows,
            kv_len,
            n_heads,
            n_kv_heads,
            head_dim,
            /*causal=*/true,
            window_size))
            << format;
        ASSERT_TRUE(kernel.compute_verifier_rows_decode_equivalent(
            &query,
            &physical_key,
            &physical_value,
            &physical_grouped,
            query_rows,
            kv_len,
            n_heads,
            n_kv_heads,
            head_dim,
            /*causal=*/true,
            window_size,
            /*mpi_ctx=*/nullptr,
            /*device_idx=*/-1,
            /*head_start=*/0,
            /*gqa_n_rep=*/0,
            kv_logical_view,
            {
                .prefill_parallel_axis =
                    attention::AttentionPrefillParallelAxis::GeometrySelected,
            }))
            << format;
        ASSERT_EQ(
            std::memcmp(
                physical_grouped.data(),
                compact_grouped.data(),
                output_elements * sizeof(float)),
            0)
            << format << " grouped ring/compact mismatch";
    }

    /**
     * @brief Prove one native grouped-verifier launch equals serial decode.
     *
     * Row `r` in a verifier group sees the prefix plus rows `[0, r]`. The
     * oracle therefore invokes the same public production kernel at M=1 with
     * K/V lengths `prefix + r + 1`; it does not call a scalar reference or
     * disable the optimized cache-format implementation.
     */
    static void expectGroupedVerifierMatchesSerialDecode(
        const FP32Tensor &query,
        const ITensor &key,
        const ITensor &value,
        int query_rows,
        int kv_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        const std::string &format,
        const attention::AttentionKVLogicalView &kv_logical_view = {},
        int window_size = -1)
    {
        ASSERT_GT(query_rows, 1);
        const std::size_t query_columns =
            static_cast<std::size_t>(n_heads) * head_dim;
        FP32Tensor grouped_output(
            {static_cast<std::size_t>(query_rows), query_columns});
        FP32Tensor serial_output(
            {static_cast<std::size_t>(query_rows), query_columns});
        CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;
        const WorkspaceRequirements requirements =
            kernel.getWorkspaceRequirements(query_rows, n_heads, head_dim);
        DeviceWorkspaceManager workspace(
            DeviceId::cpu(),
            requirements.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(workspace.allocate(requirements));
        kernel.bindWorkspace(&workspace);

        ASSERT_TRUE(kernel.compute_verifier_rows_decode_equivalent(
            &query,
            &key,
            &value,
            &grouped_output,
            query_rows,
            kv_len,
            n_heads,
            n_kv_heads,
            head_dim,
            /*causal=*/true,
            window_size,
            /*mpi_ctx=*/nullptr,
            /*device_idx=*/-1,
            /*head_start=*/0,
            /*gqa_n_rep=*/0,
            kv_logical_view))
            << format;

        const int prefix_rows = kv_len - query_rows;
        ASSERT_GE(prefix_rows, 1);
        for (int row = 0; row < query_rows; ++row)
        {
            FP32Tensor row_query({1, query_columns});
            FP32Tensor row_output({1, query_columns});
            std::copy_n(
                query.data() + static_cast<std::size_t>(row) * query_columns,
                query_columns,
                row_query.mutable_data());
            ASSERT_TRUE(kernel.compute_tensor(
                &row_query,
                &key,
                &value,
                &row_output,
                /*batch_size=*/1,
                /*seq_len=*/1,
                prefix_rows + row + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                /*causal=*/true,
                window_size,
                /*workspace_scores=*/nullptr,
                /*workspace_mask=*/nullptr,
                /*mpi_ctx=*/nullptr,
                /*device_idx=*/-1,
                /*head_start=*/0,
                /*local_n_heads=*/-1,
                /*local_n_kv_heads=*/-1,
                /*gqa_n_rep=*/0,
                {
                    .prefill_parallel_axis =
                        attention::AttentionPrefillParallelAxis::GeometrySelected,
                },
                kv_logical_view))
                << format << " serial row=" << row;
            std::copy_n(
                row_output.data(),
                query_columns,
                serial_output.mutable_data() +
                    static_cast<std::size_t>(row) * query_columns);
        }

        const auto *grouped_bytes = reinterpret_cast<const std::uint8_t *>(
            grouped_output.data());
        const auto *serial_bytes = reinterpret_cast<const std::uint8_t *>(
            serial_output.data());
        const std::size_t output_bytes =
            static_cast<std::size_t>(query_rows) * query_columns *
            sizeof(float);
        for (std::size_t byte = 0; byte < output_bytes; ++byte)
        {
            const std::size_t element = byte / sizeof(float);
            std::uint32_t grouped_bits = 0;
            std::uint32_t serial_bits = 0;
            std::memcpy(
                &grouped_bits,
                grouped_output.data() + element,
                sizeof(grouped_bits));
            std::memcpy(
                &serial_bits,
                serial_output.data() + element,
                sizeof(serial_bits));
            ASSERT_EQ(grouped_bytes[byte], serial_bytes[byte])
                << format << " first grouped/serial mismatch at output byte "
                << byte << " element=" << element
                << " grouped=" << grouped_output.data()[element]
                << " serial=" << serial_output.data()[element]
                << " grouped_bits=0x" << std::hex << grouped_bits
                << " serial_bits=0x" << serial_bits;
        }
    }

    /**
     * @brief Prove one causal prefill launch is byte-identical to serial decode.
     *
     * A causal prefill row at position `r` sees exactly K/V rows `[0, r]`.
     * The oracle invokes the same native-format production kernel at M=1 and
     * logical K/V length `r + 1`; it never dequantizes the cache or substitutes
     * scalar attention. This authenticates batch invariance across K/V tile and
     * canonical context-partition boundaries.
     */
    static void expectCausalPrefillMatchesSerialDecode(
        const FP32Tensor &query,
        const ITensor &key,
        const ITensor &value,
        int sequence_rows,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        const std::string &format)
    {
        ASSERT_GT(sequence_rows, 0);
        const std::size_t query_columns =
            static_cast<std::size_t>(n_heads) * head_dim;
        FP32Tensor prefill_output(
            {static_cast<std::size_t>(sequence_rows), query_columns});
        FP32Tensor serial_output(
            {static_cast<std::size_t>(sequence_rows), query_columns});

        CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;
        const WorkspaceRequirements requirements =
            kernel.getWorkspaceRequirements(
                sequence_rows, n_heads, head_dim);
        DeviceWorkspaceManager workspace(
            DeviceId::cpu(),
            requirements.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(workspace.allocate(requirements)) << format;
        kernel.bindWorkspace(&workspace);

        const attention::AttentionExecutionPolicy execution_policy{
            .prefill_parallel_axis =
                attention::AttentionPrefillParallelAxis::QuerySequence,
        };
        ASSERT_TRUE(kernel.compute_tensor(
            &query,
            &key,
            &value,
            &prefill_output,
            /*batch_size=*/1,
            sequence_rows,
            sequence_rows,
            n_heads,
            n_kv_heads,
            head_dim,
            /*causal=*/true,
            /*window_size=*/-1,
            /*workspace_scores=*/nullptr,
            /*workspace_mask=*/nullptr,
            /*mpi_ctx=*/nullptr,
            /*device_idx=*/-1,
            /*head_start=*/0,
            /*local_n_heads=*/-1,
            /*local_n_kv_heads=*/-1,
            /*gqa_n_rep=*/0,
            execution_policy))
            << format;

        for (int row = 0; row < sequence_rows; ++row)
        {
            FP32Tensor row_query({1, query_columns});
            FP32Tensor row_output({1, query_columns});
            std::copy_n(
                query.data() + static_cast<std::size_t>(row) * query_columns,
                query_columns,
                row_query.mutable_data());
            ASSERT_TRUE(kernel.compute_tensor(
                &row_query,
                &key,
                &value,
                &row_output,
                /*batch_size=*/1,
                /*seq_len=*/1,
                /*kv_len=*/row + 1,
                n_heads,
                n_kv_heads,
                head_dim,
                /*causal=*/true,
                /*window_size=*/-1,
                /*workspace_scores=*/nullptr,
                /*workspace_mask=*/nullptr,
                /*mpi_ctx=*/nullptr,
                /*device_idx=*/-1,
                /*head_start=*/0,
                /*local_n_heads=*/-1,
                /*local_n_kv_heads=*/-1,
                /*gqa_n_rep=*/0,
                execution_policy))
                << format << " serial row=" << row;
            std::copy_n(
                row_output.data(),
                query_columns,
                serial_output.mutable_data() +
                    static_cast<std::size_t>(row) * query_columns);
        }

        const std::size_t output_bytes =
            static_cast<std::size_t>(sequence_rows) * query_columns *
            sizeof(float);
        ASSERT_EQ(
            std::memcmp(
                prefill_output.data(), serial_output.data(), output_bytes),
            0)
            << format << " causal prefill differs from serial decode";
    }

    /**
     * @brief Prove grouped independent requests equal native serial requests.
     *
     * The grouped invocation receives unequal logical lengths and one explicit
     * ring descriptor per request. The oracle invokes `compute_tensor()` once
     * for each request using the same native tensors and physical policy. Both
     * sides therefore execute optimized production arithmetic; byte equality
     * detects descriptor, dispatch, or grouped-horizon drift.
     */
    static void expectNativeRequestBatchMatchesSerial(
        const FP32Tensor &query,
        const std::vector<std::shared_ptr<ITensor>> &keys,
        const std::vector<std::shared_ptr<ITensor>> &values,
        const std::vector<int> &kv_lens,
        const std::vector<attention::AttentionKVLogicalView> &views,
        int query_rows,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        const std::string &format)
    {
        ASSERT_EQ(keys.size(), values.size());
        ASSERT_EQ(keys.size(), kv_lens.size());
        ASSERT_EQ(keys.size(), views.size());
        ASSERT_GT(keys.size(), 1U);

        const int request_count = static_cast<int>(keys.size());
        const std::size_t query_columns =
            static_cast<std::size_t>(n_heads) * head_dim;
        const std::size_t request_elements =
            static_cast<std::size_t>(query_rows) * query_columns;
        const std::size_t total_elements =
            static_cast<std::size_t>(request_count) * request_elements;
        FP32Tensor grouped_output(
            {static_cast<std::size_t>(request_count * query_rows),
             query_columns});
        FP32Tensor serial_output(
            {static_cast<std::size_t>(request_count * query_rows),
             query_columns});

        std::vector<const ITensor *> key_views(keys.size());
        std::vector<const ITensor *> value_views(values.size());
        for (std::size_t request = 0; request < keys.size(); ++request)
        {
            ASSERT_NE(keys[request], nullptr);
            ASSERT_NE(values[request], nullptr);
            key_views[request] = keys[request].get();
            value_views[request] = values[request].get();
        }

        CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;
        const WorkspaceRequirements requirements =
            kernel.getWorkspaceRequirements(
                request_count * query_rows,
                n_heads,
                head_dim);
        DeviceWorkspaceManager workspace(
            DeviceId::cpu(),
            requirements.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(workspace.allocate(requirements)) << format;
        kernel.bindWorkspace(&workspace);

        const attention::AttentionExecutionPolicy execution_policy{
            .prefill_parallel_axis =
                attention::AttentionPrefillParallelAxis::KeyValueContext,
        };
        ASSERT_TRUE(kernel.compute_request_batch_decode_equivalent(
            &query,
            key_views.data(),
            value_views.data(),
            kv_lens.data(),
            &grouped_output,
            request_count,
            query_rows,
            n_heads,
            n_kv_heads,
            head_dim,
            /*causal=*/true,
            /*window_size=*/-1,
            /*mpi_ctx=*/nullptr,
            /*device_idx=*/-1,
            /*head_start=*/0,
            /*gqa_n_rep=*/0,
            execution_policy,
            views.data()))
            << format;

        for (int request = 0; request < request_count; ++request)
        {
            FP32Tensor request_query(
                {static_cast<std::size_t>(query_rows), query_columns});
            FP32Tensor request_output(
                {static_cast<std::size_t>(query_rows), query_columns});
            std::copy_n(
                query.data() +
                    static_cast<std::size_t>(request) * request_elements,
                request_elements,
                request_query.mutable_data());
            ASSERT_TRUE(kernel.compute_tensor(
                &request_query,
                keys[static_cast<std::size_t>(request)].get(),
                values[static_cast<std::size_t>(request)].get(),
                &request_output,
                /*batch_size=*/1,
                query_rows,
                kv_lens[static_cast<std::size_t>(request)],
                n_heads,
                n_kv_heads,
                head_dim,
                /*causal=*/true,
                /*window_size=*/-1,
                /*workspace_scores=*/nullptr,
                /*workspace_mask=*/nullptr,
                /*mpi_ctx=*/nullptr,
                /*device_idx=*/-1,
                /*head_start=*/0,
                /*local_n_heads=*/-1,
                /*local_n_kv_heads=*/-1,
                /*gqa_n_rep=*/0,
                execution_policy,
                views[static_cast<std::size_t>(request)]))
                << format << " request=" << request;
            std::copy_n(
                request_output.data(),
                request_elements,
                serial_output.mutable_data() +
                    static_cast<std::size_t>(request) * request_elements);
        }

        ASSERT_EQ(
            std::memcmp(
                grouped_output.data(),
                serial_output.data(),
                total_elements * sizeof(float)),
            0)
            << format << " grouped request batch differs from native serial requests";
    }
}

TEST(Test__CPUFlashAttentionKernel, ComputeTensor_Prefill_FP32Q_Q81KV_MatchesReference)
{
    constexpr int batch_size = 1;
    constexpr int seq_len = 8;
    constexpr int kv_len = 8;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr bool causal = true;

    const size_t q_cols = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t out_count = static_cast<size_t>(seq_len) * q_cols;

    auto q_data = makeRandom(static_cast<size_t>(seq_len) * q_cols, 1001);
    auto k_data = makeRandom(static_cast<size_t>(kv_len) * kv_cols, 1002);
    auto v_data = makeRandom(static_cast<size_t>(kv_len) * kv_cols, 1003);

    FP32Tensor q_tensor({static_cast<size_t>(seq_len), q_cols});
    FP32Tensor out_tensor({static_cast<size_t>(seq_len), q_cols});
    FP32Tensor ref_tensor({static_cast<size_t>(seq_len), q_cols});

    std::copy(q_data.begin(), q_data.end(), q_tensor.mutable_data());

    auto k_q81 = Q8_1Tensor::quantize_from_fp32(k_data.data(), {static_cast<size_t>(kv_len), kv_cols});
    auto v_q81 = Q8_1Tensor::quantize_from_fp32(v_data.data(), {static_cast<size_t>(kv_len), kv_cols});
    ASSERT_NE(k_q81, nullptr);
    ASSERT_NE(v_q81, nullptr);

    CPUFlashAttentionKernelT<ActivationPrecision::FP32> flash_kernel;
    ASSERT_TRUE(flash_kernel.compute_tensor(
        &q_tensor,
        k_q81.get(),
        v_q81.get(),
        &out_tensor,
        batch_size,
        seq_len,
        kv_len,
        n_heads,
        n_kv_heads,
        head_dim,
        causal));

    CPUFlashAttentionKernelT<ActivationPrecision::FP32> ref_kernel;
    const float *k_deq = k_q81->fp32_data();
    const float *v_deq = v_q81->fp32_data();
    ASSERT_NE(k_deq, nullptr);
    ASSERT_NE(v_deq, nullptr);
    ASSERT_TRUE(ref_kernel.compute(
        q_tensor.data(),
        k_deq,
        v_deq,
        ref_tensor.mutable_data(),
        seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        causal));

    const float cos = cosineSimilarity(out_tensor.data(), ref_tensor.data(), out_count);
    const float max_diff = maxAbsDiff(out_tensor.data(), ref_tensor.data(), out_count);

    EXPECT_GE(cos, 0.999f);
    EXPECT_LE(max_diff, 1e-3f);
}

TEST(Test__CPUFlashAttentionKernel, ComputeTensor_Decode_FP32Q_Q81KV_MatchesReference)
{
    constexpr int batch_size = 1;
    constexpr int seq_len = 1;
    constexpr int kv_len = 64;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr bool causal = true;

    const size_t q_cols = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_cols = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t out_count = static_cast<size_t>(seq_len) * q_cols;

    auto q_data = makeRandom(static_cast<size_t>(seq_len) * q_cols, 2001);
    auto k_data = makeRandom(static_cast<size_t>(kv_len) * kv_cols, 2002);
    auto v_data = makeRandom(static_cast<size_t>(kv_len) * kv_cols, 2003);

    FP32Tensor q_tensor({static_cast<size_t>(seq_len), q_cols});
    FP32Tensor out_tensor({static_cast<size_t>(seq_len), q_cols});
    FP32Tensor ref_tensor({static_cast<size_t>(seq_len), q_cols});

    std::copy(q_data.begin(), q_data.end(), q_tensor.mutable_data());

    auto k_q81 = Q8_1Tensor::quantize_from_fp32(k_data.data(), {static_cast<size_t>(kv_len), kv_cols});
    auto v_q81 = Q8_1Tensor::quantize_from_fp32(v_data.data(), {static_cast<size_t>(kv_len), kv_cols});
    ASSERT_NE(k_q81, nullptr);
    ASSERT_NE(v_q81, nullptr);

    CPUFlashAttentionKernelT<ActivationPrecision::FP32> flash_kernel;
    ASSERT_TRUE(flash_kernel.compute_tensor(
        &q_tensor,
        k_q81.get(),
        v_q81.get(),
        &out_tensor,
        batch_size,
        seq_len,
        kv_len,
        n_heads,
        n_kv_heads,
        head_dim,
        causal));

    CPUFlashAttentionKernelT<ActivationPrecision::FP32> ref_kernel;
    const float *k_deq = k_q81->fp32_data();
    const float *v_deq = v_q81->fp32_data();
    ASSERT_NE(k_deq, nullptr);
    ASSERT_NE(v_deq, nullptr);
    ASSERT_TRUE(ref_kernel.compute_decode(
        q_tensor.data(),
        k_deq,
        v_deq,
        ref_tensor.mutable_data(),
        seq_len,
        kv_len,
        n_heads,
        n_kv_heads,
        head_dim,
        causal,
        kv_len - seq_len));

    const float cos = cosineSimilarity(out_tensor.data(), ref_tensor.data(), out_count);
    const float max_diff = maxAbsDiff(out_tensor.data(), ref_tensor.data(), out_count);

    EXPECT_GE(cos, 0.999f);
    EXPECT_LE(max_diff, 1e-3f);
}

TEST(Test__CPUFlashAttentionKernel,
     PhysicalModesAndGroupedVerifierAreByteExactForEveryNonQ16NativeKVFormat)
{
    ScopedOpenMPThreadCount thread_count(/*threads=*/8);

    constexpr int kv_len = 1024;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr std::size_t query_columns =
        static_cast<std::size_t>(n_heads) * head_dim;
    constexpr std::size_t kv_columns =
        static_cast<std::size_t>(n_kv_heads) * head_dim;

    const std::vector<float> key_data =
        makeRandom(static_cast<std::size_t>(kv_len) * kv_columns, 77102);
    const std::vector<float> value_data =
        makeRandom(static_cast<std::size_t>(kv_len) * kv_columns, 77103);

    /*
     * Preserve every row that can be visible to M<=15 with a 17-row window,
     * while replacing the older prefix with hostile values. Window semantics
     * are then proven independently from physical-mode and serial/grouped
     * equivalence below.
     */
    constexpr int semantic_window_size = 17;
    constexpr int maximum_verifier_rows = 15;
    constexpr int oldest_potentially_visible_row =
        kv_len - maximum_verifier_rows - semantic_window_size + 1;
    static_assert(oldest_potentially_visible_row > 0);
    std::vector<float> older_key_data = key_data;
    std::vector<float> older_value_data = value_data;
    for (int row = 0; row < oldest_potentially_visible_row; ++row)
    {
        for (std::size_t column = 0; column < kv_columns; ++column)
        {
            const std::size_t element =
                static_cast<std::size_t>(row) * kv_columns + column;
            older_key_data[element] =
                3.0f + static_cast<float>((row + column) % 17U) * 0.125f;
            older_value_data[element] =
                -4.0f - static_cast<float>((row + column) % 19U) * 0.125f;
        }
    }

    FP32Tensor key_fp32({static_cast<std::size_t>(kv_len), kv_columns});
    FP32Tensor value_fp32({static_cast<std::size_t>(kv_len), kv_columns});
    FP32Tensor older_key_fp32(
        {static_cast<std::size_t>(kv_len), kv_columns});
    FP32Tensor older_value_fp32(
        {static_cast<std::size_t>(kv_len), kv_columns});
    std::copy(key_data.begin(), key_data.end(), key_fp32.mutable_data());
    std::copy(value_data.begin(), value_data.end(), value_fp32.mutable_data());
    std::copy(
        older_key_data.begin(), older_key_data.end(),
        older_key_fp32.mutable_data());
    std::copy(
        older_value_data.begin(), older_value_data.end(),
        older_value_fp32.mutable_data());

    std::vector<std::uint16_t> key_fp16_data(key_data.size());
    std::vector<std::uint16_t> value_fp16_data(value_data.size());
    std::transform(
        key_data.begin(),
        key_data.end(),
        key_fp16_data.begin(),
        [](float value)
        { return fp32_to_fp16(value); });
    std::transform(
        value_data.begin(),
        value_data.end(),
        value_fp16_data.begin(),
        [](float value)
        { return fp32_to_fp16(value); });
    FP16Tensor key_fp16(
        {static_cast<std::size_t>(kv_len), kv_columns},
        key_fp16_data);
    FP16Tensor value_fp16(
        {static_cast<std::size_t>(kv_len), kv_columns},
        value_fp16_data);
    std::vector<std::uint16_t> older_key_fp16_data(older_key_data.size());
    std::vector<std::uint16_t> older_value_fp16_data(older_value_data.size());
    std::transform(
        older_key_data.begin(), older_key_data.end(),
        older_key_fp16_data.begin(),
        [](float value)
        { return fp32_to_fp16(value); });
    std::transform(
        older_value_data.begin(), older_value_data.end(),
        older_value_fp16_data.begin(),
        [](float value)
        { return fp32_to_fp16(value); });
    FP16Tensor older_key_fp16(
        {static_cast<std::size_t>(kv_len), kv_columns},
        older_key_fp16_data);
    FP16Tensor older_value_fp16(
        {static_cast<std::size_t>(kv_len), kv_columns},
        older_value_fp16_data);

    std::vector<std::uint16_t> key_bf16_data(key_data.size());
    std::vector<std::uint16_t> value_bf16_data(value_data.size());
    std::transform(
        key_data.begin(),
        key_data.end(),
        key_bf16_data.begin(),
        [](float value)
        { return simd::fp32_to_bf16(value); });
    std::transform(
        value_data.begin(),
        value_data.end(),
        value_bf16_data.begin(),
        [](float value)
        { return simd::fp32_to_bf16(value); });
    BF16Tensor key_bf16(
        {static_cast<std::size_t>(kv_len), kv_columns},
        key_bf16_data);
    BF16Tensor value_bf16(
        {static_cast<std::size_t>(kv_len), kv_columns},
        value_bf16_data);
    std::vector<std::uint16_t> older_key_bf16_data(older_key_data.size());
    std::vector<std::uint16_t> older_value_bf16_data(older_value_data.size());
    std::transform(
        older_key_data.begin(), older_key_data.end(),
        older_key_bf16_data.begin(),
        [](float value)
        { return simd::fp32_to_bf16(value); });
    std::transform(
        older_value_data.begin(), older_value_data.end(),
        older_value_bf16_data.begin(),
        [](float value)
        { return simd::fp32_to_bf16(value); });
    BF16Tensor older_key_bf16(
        {static_cast<std::size_t>(kv_len), kv_columns},
        older_key_bf16_data);
    BF16Tensor older_value_bf16(
        {static_cast<std::size_t>(kv_len), kv_columns},
        older_value_bf16_data);

    const auto key_q8 = Q8_1Tensor::quantize_from_fp32(
        key_data.data(),
        {static_cast<std::size_t>(kv_len), kv_columns});
    const auto value_q8 = Q8_1Tensor::quantize_from_fp32(
        value_data.data(),
        {static_cast<std::size_t>(kv_len), kv_columns});
    const auto older_key_q8 = Q8_1Tensor::quantize_from_fp32(
        older_key_data.data(),
        {static_cast<std::size_t>(kv_len), kv_columns});
    const auto older_value_q8 = Q8_1Tensor::quantize_from_fp32(
        older_value_data.data(),
        {static_cast<std::size_t>(kv_len), kv_columns});
    ASSERT_NE(key_q8, nullptr);
    ASSERT_NE(value_q8, nullptr);
    ASSERT_NE(older_key_q8, nullptr);
    ASSERT_NE(older_value_q8, nullptr);

    TurboQuantContext turboquant_context(
        head_dim,
        /*rotation_seed=*/42,
        /*projection_seed=*/42);
    const auto key_tq4 = TQ4Tensor::quantize_from_fp32(
        key_data.data(),
        {static_cast<std::size_t>(kv_len), kv_columns},
        head_dim,
        turboquant_context);
    const auto key_tq8 = TQ8Tensor::quantize_from_fp32(
        key_data.data(),
        {static_cast<std::size_t>(kv_len), kv_columns},
        head_dim,
        turboquant_context);
    const auto value_tq4 = TQ4Tensor::quantize_from_fp32(
        value_data.data(),
        {static_cast<std::size_t>(kv_len), kv_columns},
        head_dim,
        turboquant_context);
    const auto value_tq8 = TQ8Tensor::quantize_from_fp32(
        value_data.data(),
        {static_cast<std::size_t>(kv_len), kv_columns},
        head_dim,
        turboquant_context);
    const auto older_key_tq4 = TQ4Tensor::quantize_from_fp32(
        older_key_data.data(),
        {static_cast<std::size_t>(kv_len), kv_columns},
        head_dim,
        turboquant_context);
    const auto older_key_tq8 = TQ8Tensor::quantize_from_fp32(
        older_key_data.data(),
        {static_cast<std::size_t>(kv_len), kv_columns},
        head_dim,
        turboquant_context);
    const auto older_value_tq4 = TQ4Tensor::quantize_from_fp32(
        older_value_data.data(),
        {static_cast<std::size_t>(kv_len), kv_columns},
        head_dim,
        turboquant_context);
    const auto older_value_tq8 = TQ8Tensor::quantize_from_fp32(
        older_value_data.data(),
        {static_cast<std::size_t>(kv_len), kv_columns},
        head_dim,
        turboquant_context);
    ASSERT_NE(key_tq4, nullptr);
    ASSERT_NE(key_tq8, nullptr);
    ASSERT_NE(value_tq4, nullptr);
    ASSERT_NE(value_tq8, nullptr);
    ASSERT_NE(older_key_tq4, nullptr);
    ASSERT_NE(older_key_tq8, nullptr);
    ASSERT_NE(older_value_tq4, nullptr);
    ASSERT_NE(older_value_tq8, nullptr);

    /*
     * M=1 authenticates serial decode; 2/4/8 cover ordinary grouped verifier
     * depths; 15 covers the maximum production speculative depth. Every case
     * invokes the public tensor API twice and therefore proves the physical
     * scheduler without substituting a test oracle for either implementation.
     */
    for (const int query_rows : {1, 2, 4, 8, 15})
    {
        const auto query_plan = cpu::fa2_policy::selectCPUFA2ParallelPlan({
            .batch_size = 1,
            .query_rows = query_rows,
            .local_query_heads = n_heads,
            .kv_rows = kv_len,
            .physical_workers = 8,
            .requested_axis =
                attention::AttentionPrefillParallelAxis::QuerySequence,
        });
        const auto context_plan = cpu::fa2_policy::selectCPUFA2ParallelPlan({
            .batch_size = 1,
            .query_rows = query_rows,
            .local_query_heads = n_heads,
            .kv_rows = kv_len,
            .physical_workers = 8,
            .requested_axis =
                attention::AttentionPrefillParallelAxis::KeyValueContext,
        });
        ASSERT_TRUE(query_plan.valid);
        ASSERT_TRUE(context_plan.usesContextParallelism());
        ASSERT_GT(context_plan.context_partitions, 1);
        ASSERT_EQ(
            context_plan.arithmetic_partitions,
            query_plan.arithmetic_partitions)
            << "Both physical modes must reduce the same canonical summaries";

        const std::vector<float> query_data = makeRandom(
            static_cast<std::size_t>(query_rows) * query_columns,
            77101U + static_cast<std::uint32_t>(query_rows));
        FP32Tensor query(
            {static_cast<std::size_t>(query_rows), query_columns});
        std::copy(
            query_data.begin(), query_data.end(), query.mutable_data());
        for (const int window_size : {-1, 17})
        {
            const std::string suffix =
                " M=" + std::to_string(query_rows) +
                " window=" + std::to_string(window_size);

            const auto physical_modes = [&](const ITensor &key,
                                            const ITensor &value,
                                            const char *format)
            {
                expectPhysicalModesByteExact(
                    query, key, value, query_rows, kv_len,
                    n_heads, n_kv_heads, head_dim,
                    std::string(format) + suffix,
                    /*kv_logical_view=*/{},
                    window_size);
            };
            physical_modes(key_fp32, value_fp32, "FP32");
            physical_modes(key_fp16, value_fp16, "FP16");
            physical_modes(key_bf16, value_bf16, "BF16");
            physical_modes(*key_q8, *value_q8, "Q8_1");
            physical_modes(*key_tq4, *value_tq4, "TQ4-K/TQ4-V");
            physical_modes(*key_tq4, *value_tq8, "TQ4-K/TQ8-V");
            physical_modes(*key_tq8, *value_tq4, "TQ8-K/TQ4-V");
            physical_modes(*key_tq8, *value_tq8, "TQ8-K/TQ8-V");

            if (query_rows == maximum_verifier_rows &&
                window_size == semantic_window_size)
            {
                const auto physical_tiles = [&](const ITensor &key,
                                                const ITensor &value,
                                                const char *format)
                {
                    expectPhysicalTilesByteExact(
                        query,
                        key,
                        value,
                        query_rows,
                        kv_len,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        window_size,
                        std::string(format) + suffix);
                };
                physical_tiles(key_fp32, value_fp32, "FP32");
                physical_tiles(key_fp16, value_fp16, "FP16");
                physical_tiles(key_bf16, value_bf16, "BF16");
                physical_tiles(*key_q8, *value_q8, "Q8_1");
                physical_tiles(*key_tq4, *value_tq4, "TQ4-K/TQ4-V");
                physical_tiles(*key_tq4, *value_tq8, "TQ4-K/TQ8-V");
                physical_tiles(*key_tq8, *value_tq4, "TQ8-K/TQ4-V");
                physical_tiles(*key_tq8, *value_tq8, "TQ8-K/TQ8-V");
            }

            if (window_size == semantic_window_size)
            {
                const auto window_semantics = [&](
                                                  const ITensor &key,
                                                  const ITensor &value,
                                                  const ITensor &older_key,
                                                  const ITensor &older_value,
                                                  const char *format)
                {
                    expectSlidingWindowIgnoresOlderRows(
                        query,
                        key,
                        value,
                        older_key,
                        older_value,
                        query_rows,
                        kv_len,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        window_size,
                        std::string(format) + suffix);
                };
                window_semantics(
                    key_fp32, value_fp32,
                    older_key_fp32, older_value_fp32,
                    "FP32");
                window_semantics(
                    key_fp16, value_fp16,
                    older_key_fp16, older_value_fp16,
                    "FP16");
                window_semantics(
                    key_bf16, value_bf16,
                    older_key_bf16, older_value_bf16,
                    "BF16");
                window_semantics(
                    *key_q8, *value_q8,
                    *older_key_q8, *older_value_q8,
                    "Q8_1");
                window_semantics(
                    *key_tq4, *value_tq4,
                    *older_key_tq4, *older_value_tq4,
                    "TQ4-K/TQ4-V");
                window_semantics(
                    *key_tq4, *value_tq8,
                    *older_key_tq4, *older_value_tq8,
                    "TQ4-K/TQ8-V");
                window_semantics(
                    *key_tq8, *value_tq4,
                    *older_key_tq8, *older_value_tq4,
                    "TQ8-K/TQ4-V");
                window_semantics(
                    *key_tq8, *value_tq8,
                    *older_key_tq8, *older_value_tq8,
                    "TQ8-K/TQ8-V");
            }

            if (query_rows <= 1)
                continue;

            const auto grouped = [&](const ITensor &key,
                                     const ITensor &value,
                                     const char *format)
            {
                expectGroupedVerifierMatchesSerialDecode(
                    query, key, value, query_rows, kv_len,
                    n_heads, n_kv_heads, head_dim,
                    std::string(format) + suffix,
                    /*kv_logical_view=*/{},
                    window_size);
            };
            grouped(key_fp32, value_fp32, "FP32");
            grouped(key_fp16, value_fp16, "FP16");
            grouped(key_bf16, value_bf16, "BF16");
            grouped(*key_q8, *value_q8, "Q8_1");
            grouped(*key_tq4, *value_tq4, "TQ4-K/TQ4-V");
            grouped(*key_tq4, *value_tq8, "TQ4-K/TQ8-V");
            grouped(*key_tq8, *value_tq4, "TQ8-K/TQ4-V");
            grouped(*key_tq8, *value_tq8, "TQ8-K/TQ8-V");
        }
    }

    /*
     * Prefill boundaries deliberately extend beyond the MTP range. Values on
     * both sides of 32, 128, and the canonical 256-row context span catch
     * launch-tile and summary-boundary changes while keeping the integration
     * gate compact enough for routine execution.
     */
    for (const int sequence_rows : {
             1, 2, 15, 31, 32, 33, 127, 128, 129, 255, 256, 257})
    {
        const std::vector<float> query_data = makeRandom(
            static_cast<std::size_t>(sequence_rows) * query_columns,
            77200U + static_cast<std::uint32_t>(sequence_rows));
        FP32Tensor query(
            {static_cast<std::size_t>(sequence_rows), query_columns});
        std::copy(
            query_data.begin(), query_data.end(), query.mutable_data());
        const std::string suffix =
            " prefill_M=" + std::to_string(sequence_rows);

        const auto prefill = [&](const ITensor &key,
                                 const ITensor &value,
                                 const char *format)
        {
            expectCausalPrefillMatchesSerialDecode(
                query,
                key,
                value,
                sequence_rows,
                n_heads,
                n_kv_heads,
                head_dim,
                std::string(format) + suffix);
        };
        prefill(key_fp32, value_fp32, "FP32");
        prefill(key_fp16, value_fp16, "FP16");
        prefill(key_bf16, value_bf16, "BF16");
        prefill(*key_q8, *value_q8, "Q8_1");
        prefill(*key_tq4, *value_tq4, "TQ4-K/TQ4-V");
        prefill(*key_tq4, *value_tq8, "TQ4-K/TQ8-V");
        prefill(*key_tq8, *value_tq4, "TQ8-K/TQ4-V");
        prefill(*key_tq8, *value_tq8, "TQ8-K/TQ8-V");
    }
}

TEST(Test__CPUFlashAttentionKernel,
     WrappedNativeKVRingMatchesCompactForEveryNonQ16FormatAndM)
{
    ScopedOpenMPThreadCount thread_count(/*threads=*/8);

    constexpr int kv_len = 37;
    constexpr int physical_capacity = 64;
    constexpr int logical_origin = 51;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr std::size_t query_columns =
        static_cast<std::size_t>(n_heads) * head_dim;
    constexpr std::size_t kv_columns =
        static_cast<std::size_t>(n_kv_heads) * head_dim;
    constexpr attention::AttentionKVLogicalView wrapped_view{
        .logical_row_origin = logical_origin,
        .physical_row_capacity = physical_capacity,
    };
    static_assert(wrapped_view.validFor(kv_len));
    static_assert(
        wrapped_view.physicalRow(kv_len - 1) < logical_origin,
        "The regression geometry must cross the physical ring boundary");

    const std::vector<float> compact_key_data = makeRandom(
        static_cast<std::size_t>(kv_len) * kv_columns,
        78102);
    const std::vector<float> compact_value_data = makeRandom(
        static_cast<std::size_t>(kv_len) * kv_columns,
        78103);
    const std::vector<float> physical_key_data =
        placeLogicalRowsInPhysicalRing(
            compact_key_data,
            kv_len,
            physical_capacity,
            kv_columns,
            logical_origin,
            /*poison=*/91.0f);
    const std::vector<float> physical_value_data =
        placeLogicalRowsInPhysicalRing(
            compact_value_data,
            kv_len,
            physical_capacity,
            kv_columns,
            logical_origin,
            /*poison=*/-113.0f);

    FP32Tensor compact_key_fp32(
        {static_cast<std::size_t>(kv_len), kv_columns});
    FP32Tensor compact_value_fp32(
        {static_cast<std::size_t>(kv_len), kv_columns});
    FP32Tensor physical_key_fp32(
        {static_cast<std::size_t>(physical_capacity), kv_columns});
    FP32Tensor physical_value_fp32(
        {static_cast<std::size_t>(physical_capacity), kv_columns});
    std::copy(
        compact_key_data.begin(),
        compact_key_data.end(),
        compact_key_fp32.mutable_data());
    std::copy(
        compact_value_data.begin(),
        compact_value_data.end(),
        compact_value_fp32.mutable_data());
    std::copy(
        physical_key_data.begin(),
        physical_key_data.end(),
        physical_key_fp32.mutable_data());
    std::copy(
        physical_value_data.begin(),
        physical_value_data.end(),
        physical_value_fp32.mutable_data());

    const auto to_fp16 = [](const std::vector<float> &source)
    {
        std::vector<std::uint16_t> converted(source.size());
        std::transform(
            source.begin(), source.end(), converted.begin(),
            [](float value)
            { return fp32_to_fp16(value); });
        return converted;
    };
    const auto to_bf16 = [](const std::vector<float> &source)
    {
        std::vector<std::uint16_t> converted(source.size());
        std::transform(
            source.begin(), source.end(), converted.begin(),
            [](float value)
            { return simd::fp32_to_bf16(value); });
        return converted;
    };

    FP16Tensor compact_key_fp16(
        {static_cast<std::size_t>(kv_len), kv_columns},
        to_fp16(compact_key_data));
    FP16Tensor compact_value_fp16(
        {static_cast<std::size_t>(kv_len), kv_columns},
        to_fp16(compact_value_data));
    FP16Tensor physical_key_fp16(
        {static_cast<std::size_t>(physical_capacity), kv_columns},
        to_fp16(physical_key_data));
    FP16Tensor physical_value_fp16(
        {static_cast<std::size_t>(physical_capacity), kv_columns},
        to_fp16(physical_value_data));

    BF16Tensor compact_key_bf16(
        {static_cast<std::size_t>(kv_len), kv_columns},
        to_bf16(compact_key_data));
    BF16Tensor compact_value_bf16(
        {static_cast<std::size_t>(kv_len), kv_columns},
        to_bf16(compact_value_data));
    BF16Tensor physical_key_bf16(
        {static_cast<std::size_t>(physical_capacity), kv_columns},
        to_bf16(physical_key_data));
    BF16Tensor physical_value_bf16(
        {static_cast<std::size_t>(physical_capacity), kv_columns},
        to_bf16(physical_value_data));

    const auto compact_key_q8 = Q8_1Tensor::quantize_from_fp32(
        compact_key_data.data(),
        {static_cast<std::size_t>(kv_len), kv_columns});
    const auto compact_value_q8 = Q8_1Tensor::quantize_from_fp32(
        compact_value_data.data(),
        {static_cast<std::size_t>(kv_len), kv_columns});
    const auto physical_key_q8 = Q8_1Tensor::quantize_from_fp32(
        physical_key_data.data(),
        {static_cast<std::size_t>(physical_capacity), kv_columns});
    const auto physical_value_q8 = Q8_1Tensor::quantize_from_fp32(
        physical_value_data.data(),
        {static_cast<std::size_t>(physical_capacity), kv_columns});
    ASSERT_NE(compact_key_q8, nullptr);
    ASSERT_NE(compact_value_q8, nullptr);
    ASSERT_NE(physical_key_q8, nullptr);
    ASSERT_NE(physical_value_q8, nullptr);

    TurboQuantContext turboquant_context(
        head_dim,
        /*rotation_seed=*/43,
        /*projection_seed=*/43);
    const auto quantize_tq4 = [&](
                                  const std::vector<float> &source,
                                  int rows)
    {
        return TQ4Tensor::quantize_from_fp32(
            source.data(),
            {static_cast<std::size_t>(rows), kv_columns},
            head_dim,
            turboquant_context);
    };
    const auto quantize_tq8 = [&](
                                  const std::vector<float> &source,
                                  int rows)
    {
        return TQ8Tensor::quantize_from_fp32(
            source.data(),
            {static_cast<std::size_t>(rows), kv_columns},
            head_dim,
            turboquant_context);
    };
    const auto compact_key_tq4 = quantize_tq4(compact_key_data, kv_len);
    const auto compact_key_tq8 = quantize_tq8(compact_key_data, kv_len);
    const auto compact_value_tq4 = quantize_tq4(compact_value_data, kv_len);
    const auto compact_value_tq8 = quantize_tq8(compact_value_data, kv_len);
    const auto physical_key_tq4 =
        quantize_tq4(physical_key_data, physical_capacity);
    const auto physical_key_tq8 =
        quantize_tq8(physical_key_data, physical_capacity);
    const auto physical_value_tq4 =
        quantize_tq4(physical_value_data, physical_capacity);
    const auto physical_value_tq8 =
        quantize_tq8(physical_value_data, physical_capacity);
    ASSERT_NE(compact_key_tq4, nullptr);
    ASSERT_NE(compact_key_tq8, nullptr);
    ASSERT_NE(compact_value_tq4, nullptr);
    ASSERT_NE(compact_value_tq8, nullptr);
    ASSERT_NE(physical_key_tq4, nullptr);
    ASSERT_NE(physical_key_tq8, nullptr);
    ASSERT_NE(physical_value_tq4, nullptr);
    ASSERT_NE(physical_value_tq8, nullptr);

    for (const int query_rows : {1, 2, 4, 8, 15})
    {
        const std::vector<float> query_data = makeRandom(
            static_cast<std::size_t>(query_rows) * query_columns,
            78101U + static_cast<std::uint32_t>(query_rows));
        FP32Tensor query(
            {static_cast<std::size_t>(query_rows), query_columns});
        std::copy(
            query_data.begin(), query_data.end(), query.mutable_data());

        const auto exercise = [&](
                                  const char *format,
                                  const ITensor &compact_key,
                                  const ITensor &compact_value,
                                  const ITensor &physical_key,
                                  const ITensor &physical_value)
        {
            const std::string label =
                std::string(format) + " M=" + std::to_string(query_rows);
            expectLogicalViewMatchesCompact(
                query,
                compact_key,
                compact_value,
                physical_key,
                physical_value,
                query_rows,
                kv_len,
                n_heads,
                n_kv_heads,
                head_dim,
                wrapped_view,
                label);
            if (query_rows > 1)
            {
                expectGroupedVerifierMatchesSerialDecode(
                    query,
                    physical_key,
                    physical_value,
                    query_rows,
                    kv_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    label,
                    wrapped_view);
            }
        };

        exercise(
            "FP32",
            compact_key_fp32,
            compact_value_fp32,
            physical_key_fp32,
            physical_value_fp32);
        exercise(
            "FP16",
            compact_key_fp16,
            compact_value_fp16,
            physical_key_fp16,
            physical_value_fp16);
        exercise(
            "BF16",
            compact_key_bf16,
            compact_value_bf16,
            physical_key_bf16,
            physical_value_bf16);
        exercise(
            "Q8_1",
            *compact_key_q8,
            *compact_value_q8,
            *physical_key_q8,
            *physical_value_q8);
        exercise(
            "TQ4-K/TQ4-V",
            *compact_key_tq4,
            *compact_value_tq4,
            *physical_key_tq4,
            *physical_value_tq4);
        exercise(
            "TQ4-K/TQ8-V",
            *compact_key_tq4,
            *compact_value_tq8,
            *physical_key_tq4,
            *physical_value_tq8);
        exercise(
            "TQ8-K/TQ4-V",
            *compact_key_tq8,
            *compact_value_tq4,
            *physical_key_tq8,
            *physical_value_tq4);
        exercise(
            "TQ8-K/TQ8-V",
            *compact_key_tq8,
            *compact_value_tq8,
            *physical_key_tq8,
            *physical_value_tq8);
    }
}

TEST(Test__CPUFlashAttentionKernel,
     NativeRequestBatchWithUnequalWrappedHistoriesIsByteExactForEveryNonQ16Format)
{
    ScopedOpenMPThreadCount thread_count(/*threads=*/8);

    constexpr int request_count = 3;
    constexpr int physical_capacity = 64;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr std::size_t query_columns =
        static_cast<std::size_t>(n_heads) * head_dim;
    constexpr std::size_t kv_columns =
        static_cast<std::size_t>(n_kv_heads) * head_dim;
    const std::vector<int> kv_lens = {37, 43, 51};
    const std::vector<int> origins = {51, 57, 49};
    std::vector<attention::AttentionKVLogicalView> views(request_count);
    std::vector<std::vector<float>> physical_key_data(request_count);
    std::vector<std::vector<float>> physical_value_data(request_count);

    for (int request = 0; request < request_count; ++request)
    {
        const int kv_len = kv_lens[static_cast<std::size_t>(request)];
        const int origin = origins[static_cast<std::size_t>(request)];
        views[static_cast<std::size_t>(request)] = {
            .logical_row_origin = origin,
            .physical_row_capacity = physical_capacity,
        };
        ASSERT_TRUE(views[static_cast<std::size_t>(request)].validFor(kv_len));
        ASSERT_LT(
            views[static_cast<std::size_t>(request)].physicalRow(kv_len - 1),
            origin);

        const std::vector<float> logical_key = makeRandom(
            static_cast<std::size_t>(kv_len) * kv_columns,
            79100U + static_cast<std::uint32_t>(request) * 11U);
        const std::vector<float> logical_value = makeRandom(
            static_cast<std::size_t>(kv_len) * kv_columns,
            79200U + static_cast<std::uint32_t>(request) * 13U);
        physical_key_data[static_cast<std::size_t>(request)] =
            placeLogicalRowsInPhysicalRing(
                logical_key,
                kv_len,
                physical_capacity,
                kv_columns,
                origin,
                83.0f + static_cast<float>(request));
        physical_value_data[static_cast<std::size_t>(request)] =
            placeLogicalRowsInPhysicalRing(
                logical_value,
                kv_len,
                physical_capacity,
                kv_columns,
                origin,
                -107.0f - static_cast<float>(request));
    }

    struct NativeRequestBatch
    {
        const char *format = nullptr;
        std::vector<std::shared_ptr<ITensor>> keys;
        std::vector<std::shared_ptr<ITensor>> values;
    };
    std::vector<NativeRequestBatch> batches = {
        {.format = "FP32"},
        {.format = "FP16"},
        {.format = "BF16"},
        {.format = "Q8_1"},
        {.format = "TQ4-K/TQ4-V"},
        {.format = "TQ4-K/TQ8-V"},
        {.format = "TQ8-K/TQ4-V"},
        {.format = "TQ8-K/TQ8-V"},
    };
    for (auto &batch : batches)
    {
        batch.keys.reserve(request_count);
        batch.values.reserve(request_count);
    }

    const auto make_fp32 = [&](const std::vector<float> &source)
    {
        auto tensor = std::make_shared<FP32Tensor>(
            std::vector<std::size_t>{
                static_cast<std::size_t>(physical_capacity), kv_columns});
        std::copy(source.begin(), source.end(), tensor->mutable_data());
        return tensor;
    };
    const auto make_fp16 = [&](const std::vector<float> &source)
    {
        std::vector<std::uint16_t> converted(source.size());
        std::transform(
            source.begin(), source.end(), converted.begin(),
            [](float value)
            { return fp32_to_fp16(value); });
        return std::make_shared<FP16Tensor>(
            std::vector<std::size_t>{
                static_cast<std::size_t>(physical_capacity), kv_columns},
            converted);
    };
    const auto make_bf16 = [&](const std::vector<float> &source)
    {
        std::vector<std::uint16_t> converted(source.size());
        std::transform(
            source.begin(), source.end(), converted.begin(),
            [](float value)
            { return simd::fp32_to_bf16(value); });
        return std::make_shared<BF16Tensor>(
            std::vector<std::size_t>{
                static_cast<std::size_t>(physical_capacity), kv_columns},
            converted);
    };

    TurboQuantContext turboquant_context(
        head_dim,
        /*rotation_seed=*/44,
        /*projection_seed=*/44);
    for (int request = 0; request < request_count; ++request)
    {
        const auto &key = physical_key_data[static_cast<std::size_t>(request)];
        const auto &value =
            physical_value_data[static_cast<std::size_t>(request)];
        const std::vector<std::size_t> shape = {
            static_cast<std::size_t>(physical_capacity), kv_columns};

        batches[0].keys.push_back(make_fp32(key));
        batches[0].values.push_back(make_fp32(value));
        batches[1].keys.push_back(make_fp16(key));
        batches[1].values.push_back(make_fp16(value));
        batches[2].keys.push_back(make_bf16(key));
        batches[2].values.push_back(make_bf16(value));

        const auto key_q8 = Q8_1Tensor::quantize_from_fp32(
            key.data(), shape);
        const auto value_q8 = Q8_1Tensor::quantize_from_fp32(
            value.data(), shape);
        ASSERT_NE(key_q8, nullptr);
        ASSERT_NE(value_q8, nullptr);
        batches[3].keys.push_back(key_q8);
        batches[3].values.push_back(value_q8);

        const auto key_tq4 = TQ4Tensor::quantize_from_fp32(
            key.data(), shape, head_dim, turboquant_context);
        const auto key_tq8 = TQ8Tensor::quantize_from_fp32(
            key.data(), shape, head_dim, turboquant_context);
        const auto value_tq4 = TQ4Tensor::quantize_from_fp32(
            value.data(), shape, head_dim, turboquant_context);
        const auto value_tq8 = TQ8Tensor::quantize_from_fp32(
            value.data(), shape, head_dim, turboquant_context);
        ASSERT_NE(key_tq4, nullptr);
        ASSERT_NE(key_tq8, nullptr);
        ASSERT_NE(value_tq4, nullptr);
        ASSERT_NE(value_tq8, nullptr);
        batches[4].keys.push_back(key_tq4);
        batches[4].values.push_back(value_tq4);
        batches[5].keys.push_back(key_tq4);
        batches[5].values.push_back(value_tq8);
        batches[6].keys.push_back(key_tq8);
        batches[6].values.push_back(value_tq4);
        batches[7].keys.push_back(key_tq8);
        batches[7].values.push_back(value_tq8);
    }

    for (const int query_rows : {1, 4, 15})
    {
        FP32Tensor query(
            {static_cast<std::size_t>(request_count * query_rows),
             query_columns});
        const std::vector<float> query_data = makeRandom(
            static_cast<std::size_t>(request_count * query_rows) *
                query_columns,
            79300U + static_cast<std::uint32_t>(query_rows));
        std::copy(
            query_data.begin(), query_data.end(), query.mutable_data());

        for (const auto &batch : batches)
        {
            expectNativeRequestBatchMatchesSerial(
                query,
                batch.keys,
                batch.values,
                kv_lens,
                views,
                query_rows,
                n_heads,
                n_kv_heads,
                head_dim,
                std::string(batch.format) +
                    " M=" + std::to_string(query_rows));
        }
    }
}
