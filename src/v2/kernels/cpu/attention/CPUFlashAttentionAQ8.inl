/**
 * @file CPUFlashAttentionAQ8.inl
 * @brief Native anchored-key operand adapter for the canonical CPU FA2 scheduler.
 *
 * Included in CPUFlashAttentionKernelT. Only one key head is reconstructed into
 * bounded worker scratch; no context-length FP32 cache exists. Value accumulation
 * reuses the existing native Q8/TQ primitives and fixed-order summary reduction.
 */

/**
 * @brief Execute attention on AQ8 keys and unchanged native compressed values.
 * @param Q FP32 query rows, grouped identically for serial and verifier calls.
 * @param K Immutable request-local keys, including their physical ring geometry.
 * @param V Matching physical value slots; TQ values carry their layer context.
 * @param output Caller-owned FP32 attention result.
 * @param seq_len Number of live query rows.
 * @param kv_len Visible logical key/value length.
 * @param n_heads Local query-head count.
 * @param n_kv_heads Local key/value-head count.
 * @param head_dim Complete codec head dimension.
 * @param causal Apply the usual causal mask.
 * @param window_size Optional sliding attention window.
 * @param position_offset Absolute position of the first query row in the view.
 * @param head_start Query-head sharding offset.
 * @param gqa_n_rep Explicit global query/KV grouping, or zero for local grouping.
 * @param execution_policy Existing typed attention schedule; arithmetic is fixed.
 * @param view Logical-to-physical ring mapping for this request.
 * @return False before execution if any typed operand contract is invalid.
 */
bool compute_aq8kv(
    const float *Q, const AttentionKeyQ8Tensor *K, const ITensor *V, float *output,
    int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
    bool causal, int window_size, int position_offset, int head_start, int gqa_n_rep,
    const attention::AttentionExecutionPolicy &execution_policy,
    const attention::AttentionKVLogicalView &view)
{
    if (!Q || !K || !V || !output || seq_len <= 0 || kv_len <= 0 ||
        n_heads <= 0 || n_kv_heads <= 0 || K->heads() != static_cast<size_t>(n_kv_heads) ||
        K->head_dim() != head_dim || !view.validFor(kv_len) ||
        static_cast<size_t>(kv_len) > K->positions() ||
        (gqa_n_rep <= 0 && n_heads % n_kv_heads != 0))
        return false;
    if (!view.isContiguous() && static_cast<size_t>(view.physical_row_capacity) != K->positions())
        return false;

    const auto launch = [&]<int D, typename ValueTensor>(const ValueTensor *values)
    {
        if (!values || values->shape() != K->shape()) return false;
        const TurboQuantContext *context = nullptr;
        if constexpr (!std::is_same_v<ValueTensor, Q8_1Tensor>)
        {
            context = values->turboquant_context();
            if (!context || values->head_dim() != D) return false;
        }
        KERNEL_PROFILE_SCOPE(KernelType::ATTENTION);
        constexpr bool q8_values = std::is_same_v<ValueTensor, Q8_1Tensor>;
        const size_t value_bytes = [&]()
        {
            if constexpr (q8_values) return (D / Q8_1Block::BLOCK_SIZE) * sizeof(Q8_1Block);
            else return values->block_bytes();
        }();
        const auto *vraw = static_cast<const uint8_t *>(values->raw_data());
        const auto *blocks = reinterpret_cast<const AttentionKeyQ8Block<D> *>(K->blocks());
        const auto isa = cpu::attention_key_q8::resolveISA(cpu::attention_key_q8::ISA::Automatic);
        const int replicas = gqa_n_rep > 0 ? gqa_n_rep : n_heads / n_kv_heads;
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));
        constexpr auto pair = q8_values ? cpu::fa2_policy::CPUFA2KVStoragePair::AQ8_Q8_1
            : (std::is_same_v<ValueTensor, TQ4Tensor> ? cpu::fa2_policy::CPUFA2KVStoragePair::AQ8_TQ4
                                                   : cpu::fa2_policy::CPUFA2KVStoragePair::AQ8_TQ8);
        const int tile = detail::selectCPUFlashKVTile(pair, D, kv_len,
            sizeof(AttentionKeyQ8Block<D>), value_bytes, launch_policy_.explicit_kv_tile);
        const auto head_for_row = [&](int row)
        {
            return (gqa_n_rep > 0 ? head_start + row % n_heads : row % n_heads) / replicas;
        };
        // Validate the complete shard before workers dereference head-local data.
        if (head_for_row(0) < 0 || head_for_row(n_heads - 1) >= n_kv_heads) return false;

        struct NativeRow
        {
            const float *query = nullptr; ///< One immutable query; never rotated for AQ8 keys.
            const float *anchor = nullptr; ///< Request basis, shared read-only across partitions.
            int head = 0; ///< Explicit local KV head, independent of tensor layout.
        };
        const auto prepare = [&](int row, NativeRow &scratch)
        {
            scratch.query = Q + static_cast<size_t>(row) * D;
            scratch.head = head_for_row(row);
            scratch.anchor = K->anchor().data() + static_cast<size_t>(scratch.head) * D;
        };
        const auto visible = [&](int row, int begin, int end)
        {
            const int position = position_offset + row / n_heads;
            return CPUFA2VisibleTile{
                .begin = std::clamp(window_size > 0 ? position - window_size + 1 : begin, begin, end),
                .end = std::clamp(causal ? position + 1 : end, begin, end)};
        };
        const auto score = [&](int, const NativeRow &scratch, int begin,
                               CPUFA2VisibleTile span, float *scores, float &maximum)
        {
            // Bounded codec scratch, not an accumulator spill or a key shadow.
            alignas(64) float key[D];
            for (int row = span.begin; row < span.end; ++row)
            {
                const size_t index = K->block_index(view.physicalRow(row), scratch.head);
                cpu::attention_key_q8::dequantize<D>(blocks[index],
                    std::span<const float, D>(scratch.anchor, D), std::span<float, D>(key, D), isa);
                const float value = dot_fp32(scratch.query, key, D) * scale;
                scores[row - begin] = value;
                maximum = std::max(maximum, value);
            }
        };
        const auto accumulate = [&](int, const NativeRow &scratch, int begin,
                                    CPUFA2VisibleTile span, const float *scores,
                                    float maximum, float *partial, float &normalizer)
        {
            for (int row = span.begin; row < span.end; ++row)
            {
                const float probability = std::exp(scores[row - begin] - maximum);
                normalizer += probability;
                const auto *value = vraw + K->block_index(view.physicalRow(row), scratch.head) * value_bytes;
                if constexpr (q8_values)
                {
                    const auto *native = reinterpret_cast<const Q8_1Block *>(value);
                    for (int block = 0; block < D / Q8_1Block::BLOCK_SIZE; ++block)
                        accum_weighted_v_q8_1_block(partial + block * Q8_1Block::BLOCK_SIZE,
                                                   native + block, probability);
                }
                else if constexpr (std::is_same_v<ValueTensor, TQ4Tensor>)
                    tq4_accum_weighted(partial, value, probability, D);
                else
                    tq8_accum_weighted(partial, value, probability, D);
            }
        };
        const auto finalize = [&](int row, float *merged, float normalizer)
        {
            if (normalizer > 0.0f)
                scale_vec(merged, (q8_values ? 1.0f : scale) / normalizer, D, false);
            else
                std::fill(merged, merged + D, 0.0f);
            float *destination = output + static_cast<size_t>(row) * D;
            if constexpr (q8_values)
                std::memcpy(destination, merged, D * sizeof(float));
            else
                apply_rotation_transpose(context->for_layer(head_for_row(row)).rotation(), merged, destination);
        };
        return executePartitionedAttention<NativeRow>(seq_len, n_heads, kv_len, D, tile,
            causal, execution_policy, prepare, visible, score, accumulate, finalize);
    };
    const auto dimensions = [&]<typename ValueTensor>(const ValueTensor *values)
    {
        switch (head_dim)
        {
        case 64: return launch.template operator()<64>(values);
        case 128: return launch.template operator()<128>(values);
        case 256: return launch.template operator()<256>(values);
        default: return false;
        }
    };
    switch (V->native_type())
    {
    case TensorType::Q8_1: return dimensions(dynamic_cast<const Q8_1Tensor *>(V));
    case TensorType::TQ4: return dimensions(dynamic_cast<const TQ4Tensor *>(V));
    case TensorType::TQ8: return dimensions(dynamic_cast<const TQ8Tensor *>(V));
    default: return false;
    }
}
