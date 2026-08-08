/**
 * @file AttentionExecutionPolicy.h
 * @brief Declarative physical execution policy for attention graph capture.
 *
 * Model graphs use these types to state which logical attention axis may be
 * partitioned by a backend.  They deliberately contain no CUDA, HIP, or CPU
 * launch geometry: each backend translates the declared policy and immutable
 * tensor geometry into one concrete capture plan.  The selected plan is fixed
 * for the lifetime of the captured graph, while live sequence lengths remain
 * device-owned inputs to that graph.
 */

#pragma once

#include <cstdint>

namespace llaminar2::attention
{
    /**
     * @brief Physical axis used to expose parallel work during prefill.
     *
     * `QuerySequence` assigns complete query-row tiles to independent workers;
     * every worker scans the visible K/V context. `KeyValueContext` partitions
     * the K/V span into canonical contiguous summaries and merges those
     * summaries in a fixed ascending order. `GeometrySelected` asks the backend
     * to choose between those two byte-equivalent implementations exactly once
     * from immutable capture geometry. It is not a runtime autotuner and may
     * not inspect a host mirror of the live sequence length.
     */
    enum class AttentionPrefillParallelAxis : std::uint8_t
    {
        QuerySequence,
        KeyValueContext,
        GeometrySelected,
    };

    /**
     * @brief Model-declared attention execution policy carried into a stage.
     *
     * Additional backend-independent axes belong here as typed fields. Avoid
     * boolean launch switches in model graph files: the graph should describe
     * policy, while attention implementations own concrete block geometry,
     * workspace layout, and graph-node construction.
     */
    struct AttentionExecutionPolicy
    {
        AttentionPrefillParallelAxis prefill_parallel_axis =
            AttentionPrefillParallelAxis::QuerySequence;
    };

    /**
     * @brief Immutable geometry available before a GPU attention-param producer.
     *
     * A backend may need to create graph control state before the device kernel
     * that publishes live attention geometry. Keeping these fields in one typed
     * value lets the stage declare that capture boundary without exposing CUDA
     * or HIP handles through the common attention interface. `query_rows` is the
     * physical captured bucket width, while the producer may still consume a
     * separate device-owned active-row count during replay.
     *
     * Zero-initialization denotes a call that is not preparing a prefill graph.
     * GPU backends must reject a partially populated value instead of guessing
     * missing geometry or changing physical modes.
     */
    struct AttentionPrefillCaptureGeometry
    {
        int batch_size = 0;        ///< Captured request count.
        int query_rows = 0;        ///< Physical captured query bucket width.
        int local_query_heads = 0; ///< Query heads owned by this participant.
        int head_dim = 0;          ///< Elements in one query/key/value head.
        int kv_capacity = 0;       ///< Stable physical K/V row capacity.
        AttentionExecutionPolicy execution_policy{}; ///< Declarative graph policy.

        /** @return True when every required immutable dimension is positive. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return batch_size > 0 && query_rows > 0 &&
                   local_query_heads > 0 && head_dim > 0 && kv_capacity > 0;
        }

        /** @return True when no capture geometry was supplied by the caller. */
        [[nodiscard]] constexpr bool empty() const noexcept
        {
            return batch_size == 0 && query_rows == 0 &&
                   local_query_heads == 0 && head_dim == 0 && kv_capacity == 0;
        }
    };

    /**
     * @brief Return a stable diagnostic name for a prefill parallel axis.
     *
     * @param axis Axis selected or requested by the attention graph.
     * @return Process-lifetime string literal suitable for logs and PerfStats.
     */
    [[nodiscard]] inline constexpr const char *attentionPrefillParallelAxisName(
        AttentionPrefillParallelAxis axis)
    {
        switch (axis)
        {
        case AttentionPrefillParallelAxis::QuerySequence:
            return "query_sequence";
        case AttentionPrefillParallelAxis::KeyValueContext:
            return "key_value_context";
        case AttentionPrefillParallelAxis::GeometrySelected:
            return "geometry_selected";
        }
        return "invalid";
    }

} // namespace llaminar2::attention
