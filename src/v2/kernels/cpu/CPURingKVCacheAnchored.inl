/**
 * @file CPURingKVCacheAnchored.inl
 * @brief Native CPU AQ8 append: immutable request basis and one cursor commit.
 *
 * Included inside namespace llaminar2 by CPURingKVCache.cpp. Values retain
 * their selected Q8/TQ format. Each worker writes disjoint persistent slots;
 * no projection shadow or temporary tensor is allocated. Empty-entry state is
 * the authority for basis initialization, including reset and prefix restore.
 */

template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
template <int D>
bool CPURingKVCache<KPrecision, VPrecision>::append_anchored(
    int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v,
    int rows)
    requires (KPrecision == ActivationPrecision::AQ8)
{
    if (!new_k || !new_v || new_k->native_type() != TensorType::FP32 ||
        new_v->native_type() != TensorType::FP32)
        throw std::invalid_argument("CPU compressed append requires original FP32 K/V projections");

    // Resolve each source independently: grouped RoPE can expose head-major K
    // while V remains position-major. Bucket padding is never a live KV row.
    const auto source_head_major = [&](const TensorBase &tensor)
    {
        const auto &shape = tensor.shape();
        if (shape.size() == 2 && shape[1] == static_cast<size_t>(kv_dim_) &&
            shape[0] >= static_cast<size_t>(rows))
            return false;
        if (shape.size() == 2 && shape[1] == D &&
            shape[0] == static_cast<size_t>(rows) * local_n_kv_heads_)
            return true;
        throw std::invalid_argument("CPU compressed append source geometry mismatch");
    };
    const bool k_head_major = source_head_major(*new_k);
    const bool v_head_major = source_head_major(*new_v);
    const float *keys = static_cast<const float *>(new_k->raw_data());
    const float *values = static_cast<const float *>(new_v->raw_data());
    if (!keys || !values)
        throw std::invalid_argument("CPU compressed append requires host-resident projections");
    const auto source = [&](const float *base, bool head_major, int row, int head)
    {
        return base + (head_major ? static_cast<size_t>(head) * rows + row
                                 : static_cast<size_t>(row) * local_n_kv_heads_ + head) * D;
    };

    auto &entry = entries_[layer][seq_idx];
    const KVRingAppendPlan plan(max_seq_len_, entry.head, entry.size, rows);
    const auto isa = cpu::attention_key_q8::resolveISA(cpu::attention_key_q8::ISA::Automatic);
    auto anchor = entry.K->mutable_anchor();
    auto *key_blocks = reinterpret_cast<AttentionKeyQ8Block<D> *>(entry.K->mutable_blocks());
    auto *value_bytes = static_cast<uint8_t *>(entry.V->raw_mutable_data());
    const size_t value_head_bytes = detail::CPUKVCacheTensor<VPrecision>::head_bytes(entry.V.get(), kv_dim_, D);

    if (entry.size == 0)
    {
        // The first serial token defines the request basis for every execution
        // phase. A batch mean depends on capture bucket/prefix boundaries and
        // would change cached bytes for the same sequence when chunks change.
        // Keep the first input basis even if an oversized append evicts its row.
        for (int head = 0; head < local_n_kv_heads_; ++head)
            for (int coordinate = 0; coordinate < D; ++coordinate)
            {
                const float first = source(keys, k_head_major, 0, head)[coordinate];
                if (!std::isfinite(first))
                    throw std::domain_error("CPU AQ8 basis requires finite key projections");
                anchor[static_cast<size_t>(head) * D + coordinate] = first;
            }
    }

    std::atomic<bool> invalid{false};
    auto encode_rows = [&]()
    {
#pragma omp for schedule(static)
        for (int work = 0; work < plan.retainedRows() * local_n_kv_heads_; ++work)
        {
            const int row = plan.firstRetainedSourceRow() + work / local_n_kv_heads_;
            const int head = work % local_n_kv_heads_;
            const size_t block_index = entry.K->block_index(plan.destination(row), head);
            const float *key = source(keys, k_head_major, row, head);
            const float *value = source(values, v_head_major, row, head);
            try
            {
                cpu::attention_key_q8::quantize<D>(std::span<const float, D>(key, D),
                    std::span<const float, D>(anchor.data() + static_cast<size_t>(head) * D, D),
                    key_blocks[block_index], isa);
                auto *destination = value_bytes + block_index * value_head_bytes;
                if constexpr (VPrecision == ActivationPrecision::Q8_1)
                {
                    auto *blocks = reinterpret_cast<Q8_1Block *>(destination);
                    for (int block = 0; block < D / Q8_1Block::BLOCK_SIZE; ++block)
                        simd::quantize_single_block(value + block * Q8_1Block::BLOCK_SIZE, blocks[block]);
                }
                else
                {
                    alignas(64) float scratch0[D], scratch1[D];
                    const auto &context = *entry.value_head_contexts[head];
                    if constexpr (VPrecision == ActivationPrecision::TQ4)
                        turboquant_quantize_tq4<D>(value, context,
                            *reinterpret_cast<TQ4Block<D> *>(destination), scratch0, scratch1);
                    else
                        turboquant_quantize_tq8<D>(value, context,
                            *reinterpret_cast<TQ8Block<D> *>(destination), scratch0, scratch1);
                }
            }
            catch (...)
            {
                // Exceptions cannot cross an OpenMP boundary. Abort the caller
                // after joining every producer, never publish a partial cursor.
                invalid.store(true, std::memory_order_relaxed);
            }
        }
    };
    OMP_WORKSHARE_REGION(encode_rows);
    if (invalid.load(std::memory_order_relaxed))
        throw std::domain_error("CPU compressed cache append failed; inference must halt");
    entry.head = plan.headAfter();
    entry.size = plan.sizeAfter();
    return true;
}
