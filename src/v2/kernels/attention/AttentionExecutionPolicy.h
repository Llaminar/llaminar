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
     * @brief Physical representation of keys published into the KV cache.
     *
     * `PostRotary` pays the positional transform once before native cache
     * quantization and lets every later attention read consume the persistent
     * tensor directly. `PreRotaryDeviceTransform` is reserved for GPU caches
     * whose captured gather/attention path applies RoPE entirely on device.
     * CPU kernels must use `PostRotary`: converting or repeatedly transforming
     * native cache rows would defeat direct Q8/Q16/TurboQuant attention.
     */
    enum class AttentionKeyCacheEncoding : std::uint8_t
    {
        PostRotary,
        PreRotaryDeviceTransform,
    };

    /**
     * @brief Declarative key-cache publication and consumption policy.
     *
     * RoPE parameters are carried with the encoding decision so graph builders,
     * append stages, and attention readers cannot disagree about whether cached
     * key bytes are already positional. They are meaningful only for
     * `PreRotaryDeviceTransform` but remain initialized for complete graph
     * identity and diagnostics.
     */
    struct AttentionKeyCachePolicy
    {
        AttentionKeyCacheEncoding encoding =
            AttentionKeyCacheEncoding::PostRotary;
        float rope_theta = 10000.0f;
        float partial_rotary_factor = 1.0f;

        /** @return True when the consuming device graph must transform K. */
        [[nodiscard]] constexpr bool transformsOnRead() const noexcept
        {
            return encoding ==
                   AttentionKeyCacheEncoding::PreRotaryDeviceTransform;
        }
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
        AttentionKeyCachePolicy key_cache{};
    };

    /**
     * @brief Logical-to-physical row mapping for a native KV tensor view.
     *
     * Attention always reasons about K/V rows in oldest-to-newest logical
     * order. A ring cache may expose its persistent backing tensor directly,
     * in which case logical row zero begins at `logical_row_origin` and wraps
     * at `physical_row_capacity`. Keeping that mapping in the kernel contract
     * removes any need to unroll or dequantize the cache into a shadow tensor.
     *
     * The all-zero value denotes an ordinary contiguous tensor. A circular
     * descriptor is valid only when its origin is inside a positive capacity
     * and the requested logical span fits in that capacity. Invalid geometry
     * is a hard kernel error; implementations must not reinterpret it as a
     * contiguous view.
     */
    struct AttentionKVLogicalView
    {
        int logical_row_origin = 0;  ///< Physical row containing logical row zero.
        int physical_row_capacity = 0; ///< Ring modulus; zero means contiguous.

        /** @return True when this descriptor names an ordinary contiguous view. */
        [[nodiscard]] constexpr bool isContiguous() const noexcept
        {
            return physical_row_capacity == 0;
        }

        /**
         * @brief Validate the descriptor against a requested logical row count.
         * @param logical_rows Number of oldest-to-newest rows attention will read.
         */
        [[nodiscard]] constexpr bool validFor(int logical_rows) const noexcept
        {
            if (logical_rows < 0)
                return false;
            if (isContiguous())
                return logical_row_origin == 0;
            return logical_row_origin >= 0 &&
                   logical_row_origin < physical_row_capacity &&
                   logical_rows <= physical_row_capacity;
        }

        /**
         * @brief Translate one validated logical row into its physical tensor row.
         * @param logical_row Oldest-to-newest row index in `[0, logical_rows)`.
         */
        [[nodiscard]] constexpr int physicalRow(int logical_row) const noexcept
        {
            return isContiguous()
                       ? logical_row
                       : (logical_row_origin + logical_row) %
                             physical_row_capacity;
        }
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

    /** @return Stable diagnostic name for a key-cache encoding policy. */
    [[nodiscard]] inline constexpr const char *attentionKeyCacheEncodingName(
        AttentionKeyCacheEncoding encoding)
    {
        switch (encoding)
        {
        case AttentionKeyCacheEncoding::PostRotary:
            return "post_rotary";
        case AttentionKeyCacheEncoding::PreRotaryDeviceTransform:
            return "pre_rotary_device_transform";
        }
        return "invalid";
    }

} // namespace llaminar2::attention
