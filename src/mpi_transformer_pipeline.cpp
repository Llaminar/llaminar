// Deprecated translation unit. All implementation moved to distributed_transformer_pipeline.cpp
// This file intentionally causes a compile error if directly built to prevent divergence.
#error "mpi_transformer_pipeline.cpp deprecated: use distributed_transformer_pipeline.cpp"
struct DiffSummary
{
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double rel_l2 = 0.0;
    size_t worst_index = 0;
    float value_a = 0.0f;
    float value_b = 0.0f;
};

DiffSummary computeDiffSummary(const float *a, const float *b, size_t size)
{
    DiffSummary summary;
    if (!a || !b || size == 0)
    {
        return summary;
    }
    double max_abs = 0.0;
    size_t worst = 0;
    float worst_a = 0.0f;
    float worst_b = 0.0f;
    double sum_abs = 0.0;
    long double sum_sq = 0.0L;
    long double denom_sq = 0.0L;
    for (size_t i = 0; i < size; ++i)
    {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        double abs_diff = std::fabs(diff);
        sum_abs += abs_diff;
        sum_sq += diff * diff;
        double base = static_cast<double>(b[i]);
        denom_sq += base * base;
        if (abs_diff > max_abs)
        {
            max_abs = abs_diff;
            worst = i;
            worst_a = a[i];
            worst_b = b[i];
        }
    }
    summary.max_abs = max_abs;
    summary.mean_abs = sum_abs / static_cast<double>(size);
    summary.rel_l2 = std::sqrt(static_cast<double>(sum_sq)) / (std::sqrt(static_cast<double>(denom_sq)) + 1e-30);
    summary.worst_index = worst;
    summary.value_a = worst_a;
    summary.value_b = worst_b;
    return summary;
}

std::vector<std::tuple<size_t, float, float, double>> collectTopDiffSamples(const float *a,
                                                                            const float *b,
                                                                            size_t size,
                                                                            size_t top_n)
{
    std::vector<std::tuple<size_t, float, float, double>> samples;
    if (!a || !b || size == 0 || top_n == 0)
    {
        return samples;
    }
    samples.reserve(std::min(size, top_n));
    for (size_t i = 0; i < size; ++i)
    {
        double diff = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (diff <= 0.0)
        {
            continue;
        }
        if (samples.size() < top_n)
        {
            samples.emplace_back(i, a[i], b[i], diff);
        }
        else
        {
            auto min_it = std::min_element(samples.begin(), samples.end(),
                                           [](const auto &lhs, const auto &rhs)
                                           { return std::get<3>(lhs) < std::get<3>(rhs); });
            if (diff > std::get<3>(*min_it))
            {
                *min_it = std::make_tuple(i, a[i], b[i], diff);
            }
        }
    }
    std::sort(samples.begin(), samples.end(),
              [](const auto &lhs, const auto &rhs)
              { return std::get<3>(lhs) > std::get<3>(rhs); });
    return samples;
}

std::string trimWhitespace(const std::string &input)
{
    const auto begin = input.find_first_not_of(" \t\n\r");
    if (begin == std::string::npos)
    {
        return {};
    }
    const auto end = input.find_last_not_of(" \t\n\r");
    return input.substr(begin, end - begin + 1);
}

std::vector<int> parseIndexList(const char *env_value)
{
    std::vector<int> indices;
    if (!env_value || *env_value == '\0')
    {
        return indices;
    }

    std::stringstream ss(env_value);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        auto trimmed = trimWhitespace(token);
        if (trimmed.empty())
        {
            continue;
        }
        try
        {
            int value = std::stoi(trimmed);
            if (value >= 0)
            {
                indices.push_back(value);
            }
        }
        catch (const std::exception &)
        {
            // Ignore malformed entries silently; diagnostics will indicate actual selections later.
        }
    }
    return indices;
}

// Legacy PrefillFFNTraceConfig and related kTrace* env constants removed;
// tracing now driven entirely by debugEnv().ffn_shard_trace snapshot.

void logWorstDiffRowPreview(const char *tag,
                            const std::string &stage,
                            const float *actual,
                            const float *baseline,
                            size_t count,
                            int cols,
                            size_t worst_index,
                            size_t preview_cols = 16)
{
    if (!actual || !baseline || count == 0 || cols <= 0)
    {
        return;
    }
    size_t cols_sz = static_cast<size_t>(cols);
    if (cols_sz == 0)
    {
        return;
    }
    size_t rows = count / cols_sz;
    if (rows == 0)
    {
        return;
    }
    size_t row = worst_index / cols_sz;
    size_t col = worst_index % cols_sz;
    if (row >= rows)
    {
        return;
    }

    size_t offset = row * cols_sz;
    size_t preview = std::min(preview_cols, cols_sz);

    std::ostringstream actual_stream;
    std::ostringstream baseline_stream;
    std::ostringstream delta_stream;
    actual_stream << "[";
    baseline_stream << "[";
    delta_stream << "[";
    for (size_t i = 0; i < preview; ++i)
    {
        size_t idx = offset + i;
        float a_val = actual[idx];
        float b_val = baseline[idx];
        float d_val = a_val - b_val;
        if (i > 0)
        {
            actual_stream << ", ";
            baseline_stream << ", ";
            delta_stream << ", ";
        }
        actual_stream << a_val;
        baseline_stream << b_val;
        delta_stream << d_val;
    }
    if (cols_sz > preview)
    {
        actual_stream << ", ...";
        baseline_stream << ", ...";
        delta_stream << ", ...";
    }
    actual_stream << "]";
    baseline_stream << "]";
    delta_stream << "]";

    LOG_WARN(tag << " " << stage << " worst_row=" << row
                 << " worst_col=" << col
                 << " actual=" << actual_stream.str()
                 << " baseline=" << baseline_stream.str()
                 << " delta=" << delta_stream.str());
}

std::vector<int> resolveFFNTraceIndices(const std::vector<int> &requested, int upper_bound, int limit)
{
    std::vector<int> indices;
    if (upper_bound <= 0)
    {
        return indices;
    }

    auto clamp_push = [&](int value)
    {
        if (value >= 0 && value < upper_bound)
        {
            indices.push_back(value);
        }
    };

    if (!requested.empty())
    {
        indices.reserve(std::min<int>(requested.size(), std::max(limit, 1)));
        std::unordered_set<int> seen;
        for (int value : requested)
        {
            if (value >= 0 && value < upper_bound && seen.insert(value).second)
            {
                indices.push_back(value);
            }
        }
    }
    else
    {
        int fill_count = limit > 0 ? std::min(upper_bound, limit) : std::min(upper_bound, 1);
        indices.reserve(fill_count);
        for (int i = 0; i < fill_count; ++i)
        {
            clamp_push(i);
        }
    }

    if (limit > 0 && static_cast<int>(indices.size()) > limit)
    {
        indices.resize(limit);
    }

    return indices;
}

bool isFFNShardTracingEnabledFor(const std::string &label)
{
    const auto &cfg = llaminar::debugEnv().ffn_shard_trace;
    if (!cfg.enabled)
        return false;
    if (cfg.match_all)
        return true;
    if (cfg.shards_spec.empty())
        return true; // implicit enable
    // simple comma list semantics already stored as original spec; parse on demand for label membership
    if (cfg.match_all)
        return true;
    std::stringstream ss(cfg.shards_spec);
    std::string tok;
    while (std::getline(ss, tok, ','))
    {
        if (trimWhitespace(tok) == label)
            return true;
    }
    return false;
}

std::string formatDiffSamples(const std::vector<std::tuple<size_t, float, float, double>> &samples,
                              int cols)
{
    if (samples.empty())
    {
        return std::string("<none>");
    }
    std::ostringstream oss;
    for (size_t i = 0; i < samples.size(); ++i)
    {
        size_t idx = std::get<0>(samples[i]);
        oss << "[idx=" << idx;
        if (cols > 0)
        {
            int row = static_cast<int>(idx / static_cast<size_t>(cols));
            int col = static_cast<int>(idx % static_cast<size_t>(cols));
            oss << " r=" << row << " c=" << col;
        }
        oss << " cosma=" << std::get<1>(samples[i])
            << " ref=" << std::get<2>(samples[i])
            << " diff=" << std::get<3>(samples[i]) << "]";
        if (i + 1 < samples.size())
        {
            oss << ' ';
        }
    }
    return oss.str();
}

// Baseline capture / compare now sourced from debugEnv().baseline

void enforce_matrix_layout(std::shared_ptr<llaminar::TensorBase> &tensor,
                           int expected_rows,
                           int expected_cols,
                           const std::string &label)
{
    if (!tensor)
    {
        throw std::runtime_error(label + " tensor is null");
    }

    // If either expected dimension is zero (unknown from metadata), accept any 2D shape.
    // This enables late inference of feed-forward dimension without throwing.
    if (expected_rows == 0 || expected_cols == 0)
    {
        const auto &shape_probe = tensor->shape();
        if (shape_probe.size() == 2)
        {
            LOG_WARN(label << ": layout check bypassed (expected dimension 0). Observed shape="
                           << "[" << shape_probe[0] << ", " << shape_probe[1] << "]");
            return; // Accept as-is.
        }
    }

    const auto &shape = tensor->shape();
    if (shape.size() != 2)
    {
        std::ostringstream oss;
        oss << label << " has " << shape.size() << " dimensions, expected 2";
        LOG_ERROR(oss.str());
        throw std::runtime_error(oss.str());
    }

    if (shape[0] == expected_rows && shape[1] == expected_cols)
    {
        return; // Already in desired layout
    }

    if (shape[0] == expected_cols && shape[1] == expected_rows)
    {
        const float *src = tensor->data();
        std::vector<float> transposed(static_cast<size_t>(expected_rows) * expected_cols, 0.0f);
        for (int r = 0; r < expected_rows; ++r)
        {
            for (int c = 0; c < expected_cols; ++c)
            {
                transposed[static_cast<size_t>(r) * expected_cols + c] =
                    src[static_cast<size_t>(c) * expected_rows + r];
            }
        }

        tensor = std::make_shared<llaminar::SimpleTensor>(std::vector<int>{expected_rows, expected_cols}, transposed);
        LOG_INFO(label << " loaded as [" << shape[0] << ", " << shape[1]
                       << "]; transposed to [" << expected_rows << ", " << expected_cols << "] for pipeline layout alignment");
        return;
    }

    std::ostringstream oss;
    oss << label << " has unsupported shape [" << shape[0] << ", " << shape[1]
        << "], expected either [" << expected_rows << ", " << expected_cols
        << "] or its transpose";
    LOG_ERROR(oss.str());
    throw std::runtime_error(oss.str());
}

class PrefillBaselineRegistry
{
public:
    static PrefillBaselineRegistry &instance()
    {
        static PrefillBaselineRegistry inst;
        return inst;
    }

    void clear()
    {
        std::lock_guard<std::mutex> guard(mutex_);
        storage_.clear();
    }

    void store(const std::string &name, const float *data, size_t count)
    {
        if (!data || count == 0)
        {
            return;
        }
        std::lock_guard<std::mutex> guard(mutex_);
        storage_[name] = std::vector<float>(data, data + count);
    }
    bool ensure(const std::string &name, const float *data, size_t count)
    {
        if (!data || count == 0)
        {
            return false;
        }
        std::lock_guard<std::mutex> guard(mutex_);
        auto it = storage_.find(name);
        if (it == storage_.end() || it->second.size() != count)
        {
            storage_[name] = std::vector<float>(data, data + count);
            return true;
        }
        return false;
    }

    bool fetch(const std::string &name, std::vector<float> &out) const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        auto it = storage_.find(name);
        if (it == storage_.end())
        {
            return false;
        }
        out = it->second;
        return true;
    }

private:
    PrefillBaselineRegistry() = default;
    PrefillBaselineRegistry(const PrefillBaselineRegistry &) = delete;
    PrefillBaselineRegistry &operator=(const PrefillBaselineRegistry &) = delete;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::vector<float>> storage_;
};

void handle_prefill_stage_snapshot(int rank,
                                   const std::string &name,
                                   const float *data,
                                   size_t count,
                                   int cols,
                                   double warn_threshold,
                                   bool capture_enabled,
                                   bool compare_enabled)
{
    if ((!capture_enabled && !compare_enabled) || !data || count == 0)
    {
        return;
    }

    constexpr const char *tag = "[PrefillBaseline]";
    auto &registry = PrefillBaselineRegistry::instance();

    if (capture_enabled && rank == 0)
    {
        bool stored = registry.ensure(name, data, count);
        if (stored)
        {
            LOG_DEBUG(tag << " captured stage '" << name << "' elements=" << count);
        }
        else
        {
            LOG_TRACE(tag << " baseline for stage '" << name << "' already present; skipping overwrite");
        }
    }

    if (compare_enabled && rank == 0)
    {
        std::vector<float> baseline;
        if (!registry.fetch(name, baseline))
        {
            LOG_WARN(tag << " missing baseline for stage '" << name << "'");
            return;
        }
        if (baseline.size() != count)
        {
            LOG_WARN(tag << " size mismatch for stage '" << name << "' baseline=" << baseline.size()
                         << " current=" << count);
            return;
        }

        auto diff = computeDiffSummary(data, baseline.data(), count);
        auto samples = collectTopDiffSamples(data, baseline.data(), count, 8);
        if (diff.rel_l2 > warn_threshold)
        {
            LOG_WARN(tag << " " << name << " diff rel_l2=" << diff.rel_l2
                         << " max_abs=" << diff.max_abs
                         << " mean_abs=" << diff.mean_abs
                         << " worst_index=" << diff.worst_index
                         << " cosma=" << diff.value_a
                         << " baseline=" << diff.value_b);
            logWorstDiffRowPreview(tag,
                                   name,
                                   data,
                                   baseline.data(),
                                   count,
                                   cols,
                                   diff.worst_index,
                                   16);
        }
        else
        {
            LOG_INFO(tag << " " << name << " diff rel_l2=" << diff.rel_l2
                         << " max_abs=" << diff.max_abs
                         << " mean_abs=" << diff.mean_abs
                         << " worst_index=" << diff.worst_index
                         << " cosma=" << diff.value_a
                         << " baseline=" << diff.value_b);
        }
        LOG_INFO(tag << " " << name << " diff samples: " << formatDiffSamples(samples, cols));
    }
}

bool findFirstNonFinite(const float *data, size_t size, size_t &index, float &value)
{
    if (!data)
    {
        return false;
    }
    for (size_t i = 0; i < size; ++i)
    {
        if (!std::isfinite(data[i]))
        {
            index = i;
            value = data[i];
            return true;
        }
    }
    return false;
}
} // namespace

namespace llaminar
{
    // Factory registration for AbstractPipeline usage (architecture key: "qwen")
    static std::once_flag qwen_register_flag;
    void registerQwenPipeline()
    {
        std::call_once(qwen_register_flag, []()
                       { PipelineFactory::instance().registerCreator("qwen", [](const TransformerLayerConfig &cfg) -> std::unique_ptr<AbstractPipeline>
                                                                     {
                auto impl = std::make_unique<DistributedTransformerPipeline>(cfg);
                return std::unique_ptr<AbstractPipeline>(impl.release()); }); });
    }

    // Static member definition
    // Legacy static definitions disabled to avoid ODR duplication with unified implementation in
    // distributed_transformer_pipeline.cpp. If this file is linked, these would create separate
    // storage causing replay pipeline writes to go to a different vector instance.
    // std::atomic<size_t> DistributedTransformerPipeline::small_seq_fast_path_calls_{0};
    // std::vector<float> DistributedTransformerPipeline::last_pre_lm_hidden_;
    // std::vector<DistributedTransformerPipeline::LayerActivationStat> DistributedTransformerPipeline::last_layer_stats_;
    // std::vector<DistributedTransformerPipeline::LayerTokenDiffRow> DistributedTransformerPipeline::last_layer_token_rows_;

    DistributedTransformerPipeline::DistributedTransformerPipeline(const LayerConfig &config)
        : PipelineBase(), config_(config), use_kv_cache_(true), n_past_(0),
          total_embedding_time_(0.0), total_attention_time_(0.0),
          total_linear_time_(0.0), total_norm_time_(0.0), total_activation_time_(0.0), total_communication_time_(0.0)
    {
        // Initialize all kernels for the pipeline
        initializeKernels();

        // Initialize KV cache if enabled
        kv_cache_dynamic_init_ = debugEnv().kv_cache.dynamic_init; // snapshot per-instance
        if (getRank() == 0)
        {
            LOG_INFO("[KVCacheInitMode] constructing pipeline this=" << (const void *)this
                                                                     << " dynamic_init=" << (kv_cache_dynamic_init_ ? "on" : "off"));
        }
        if (use_kv_cache_)
        {
            int initial_capacity = config_.max_seq_len;
            if (kv_cache_dynamic_init_)
            {
                // Defer exact sizing until first execute() call where we know prefill length.
                // Allocate minimal (1) placeholder so vectors are non-empty; will be resized on first execute.
                initial_capacity = 1;
            }
            initializeKVCache(initial_capacity);
            if (getRank() == 0)
            {
                LOG_INFO("[KVCacheInitMode] post-initial-capacity this=" << (const void *)this
                                                                         << " capacity_tokens=" << kv_cache_state_.capacity_tokens);
            }
        }

        LOG_INFO("DistributedTransformerPipeline initialized on rank " << getRank() << "/" << getSize()
                                                                       << " with " << config_.n_layers << " layers, " << config_.n_head << " heads");
    }

    // Explicit out-of-line destructor to anchor vtable emission
    DistributedTransformerPipeline::~DistributedTransformerPipeline() = default;

    void DistributedTransformerPipeline::initializeKernels()
    {
        // Register Embedding kernel (supports sharded or full embedding table)
        {
            auto embedding_kernel = std::make_unique<MPIEmbeddingKernel>(config_.vocab_size, config_.d_model);
            if (!registerKernel("embedding", std::move(embedding_kernel)))
            {
                throw std::runtime_error("Failed to register Embedding kernel");
            }
        }

        // Register RMS normalization kernel
        auto rmsnorm_kernel = std::make_unique<MPIRMSNormKernel>(MPIRMSNormKernel::DistributionStrategy::SEQUENCE_WISE);
        rmsnorm_kernel->setEpsilon(config_.eps);
        if (!registerKernel("rmsnorm", std::move(rmsnorm_kernel)))
        {
            throw std::runtime_error("Failed to register RMSNorm kernel");
        }

        // Register attention kernel
        auto attention_kernel = std::make_unique<MPIAttentionKernel>(
            config_.n_head, config_.n_head_kv, config_.head_dim, config_.rope_freq_base);
        if (!registerKernel("attention", std::move(attention_kernel)))
        {
            throw std::runtime_error("Failed to register Attention kernel");
        }

        // Register linear transformation kernel
        auto linear_kernel = std::make_unique<MPILinearKernel>();
        if (!registerKernel("linear", std::move(linear_kernel)))
        {
            throw std::runtime_error("Failed to register Linear kernel");
        }

        // Register SwiGLU activation kernel
        auto swiglu_kernel = std::make_unique<MPISwiGLUKernel>(MPISwiGLUKernel::DistributionStrategy::SEQUENCE_WISE);
        if (!registerKernel("swiglu", std::move(swiglu_kernel)))
        {
            throw std::runtime_error("Failed to register SwiGLU kernel");
        }

        // Register RoPE kernel
        auto rope_kernel = std::make_unique<MPIRoPEKernel>(
            config_.max_seq_len, config_.head_dim, config_.rope_freq_base,
            MPIRoPEKernel::DistributionStrategy::SEQUENCE_WISE);
        if (!registerKernel("rope", std::move(rope_kernel)))
        {
            throw std::runtime_error("Failed to register RoPE kernel");
        }

        // Register residual connection kernel
        auto residual_kernel = std::make_unique<MPIResidualKernel>(MPIResidualKernel::DistributionStrategy::SEQUENCE_WISE);
        if (!registerKernel("residual", std::move(residual_kernel)))
        {
            throw std::runtime_error("Failed to register Residual kernel");
        }

        // Embedding previously handled directly; now executed via registered kernel for consistency

        LOG_DEBUG("DistributedTransformerPipeline: Registered " << getKernelNames().size() << " kernels on rank " << getRank());
    }

    void DistributedTransformerPipeline::traceFFNShardDiagnostics(const std::string &label,
                                                                  const float *data,
                                                                  int seq_len,
                                                                  int feature_dim)
    {
        const auto &cfg = debugEnv().ffn_shard_trace;
        if (!cfg.enabled || !data || seq_len <= 0 || feature_dim <= 0)
        {
            return;
        }
        if (!isFFNShardTracingEnabledFor(label))
        {
            return;
        }
        const int sample_limit = cfg.limit > 0 ? cfg.limit : 1;

        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        const size_t total_elements = static_cast<size_t>(seq_len) * static_cast<size_t>(feature_dim);
        auto stats = computeBufferStats(data, total_elements);

        const bool baseline_enabled = llaminar::debugEnv().baseline.compare;
        std::vector<float> baseline_buffer;
        const float *baseline_ptr = nullptr;
        BufferStats baseline_stats{};

        if (rank == 0 && baseline_enabled)
        {
            auto &registry = PrefillBaselineRegistry::instance();
            if (registry.fetch(label, baseline_buffer) && baseline_buffer.size() == total_elements)
            {
                baseline_ptr = baseline_buffer.data();
                baseline_stats = computeBufferStats(baseline_ptr, total_elements);
            }
            else
            {
                LOG_DEBUG("[PrefillFFNTrace] baseline unavailable for shard '" << label << "'");
            }
        }

        std::ostringstream header;
        header << "[PrefillFFNTrace] rank=" << rank
               << " shard=" << label
               << " shape=(" << seq_len << "," << feature_dim << ")"
               << " min=" << stats.min
               << " max=" << stats.max
               << " mean=" << stats.mean
               << " rms=" << stats.rms
               << " stddev=" << stats.stddev;
        if (baseline_ptr)
        {
            header << " baseline_min=" << baseline_stats.min
                   << " baseline_max=" << baseline_stats.max
                   << " baseline_mean=" << baseline_stats.mean
                   << " baseline_rms=" << baseline_stats.rms
                   << " baseline_stddev=" << baseline_stats.stddev;
        }
        LOG_INFO(header.str());

        auto row_indices = resolveFFNTraceIndices(cfg.rows, seq_len, sample_limit);
        auto col_indices = resolveFFNTraceIndices(cfg.cols, feature_dim, sample_limit);

        if (row_indices.empty() || col_indices.empty())
        {
            return;
        }

        for (int row : row_indices)
        {
            const float *row_ptr = data + static_cast<size_t>(row) * feature_dim;
            std::ostringstream line;
            line << "[PrefillFFNTrace] rank=" << rank
                 << " shard=" << label
                 << " row=" << row
                 << " cols=";
            for (size_t idx = 0; idx < col_indices.size(); ++idx)
            {
                int col = col_indices[idx];
                line << "(" << col << ":" << row_ptr[col];
                if (baseline_ptr)
                {
                    const float *baseline_row = baseline_ptr + static_cast<size_t>(row) * feature_dim;
                    const float baseline_val = baseline_row[col];
                    const float delta = row_ptr[col] - baseline_val;
                    line << "|baseline=" << baseline_val
                         << "|delta=" << delta;
                }
                line << ")";
                if (idx + 1 < col_indices.size())
                {
                    line << ' ';
                }
            }
            LOG_INFO(line.str());
        }
    }

    void logFFNRowPreviewIfEnabled(const std::string &label,
                                   const std::shared_ptr<TensorBase> &tensor,
                                   size_t default_preview_cols = 8)
    {
        if (!tensor)
        {
            return;
        }

        const auto &cfg = debugEnv().ffn_shard_trace;
        if (!cfg.enabled || !isFFNShardTracingEnabledFor(label))
        {
            return;
        }

        const auto &shape = tensor->shape();
        if (shape.size() < 2)
        {
            return;
        }

        const int total_rows = shape[0];
        const int total_cols = shape[1];
        if (total_rows <= 0 || total_cols <= 0)
        {
            return;
        }

        const int sample_limit = cfg.limit > 0 ? cfg.limit : 1;
        auto rows = resolveFFNTraceIndices(cfg.rows, total_rows, sample_limit);
        if (rows.empty())
        {
            return;
        }

        size_t preview_cols = default_preview_cols;
        if (!cfg.cols.empty())
        {
            preview_cols = std::min<size_t>(cfg.cols.size(), static_cast<size_t>(total_cols));
        }
        preview_cols = std::max<size_t>(1, std::min(preview_cols, static_cast<size_t>(total_cols)));

        logTensorRowPreview(tensor, label, rows, preview_cols, "PrefillFFNPreview");
    }

    bool DistributedTransformerPipeline::execute(const std::vector<int> &token_ids,
                                                 const ModelWeights &weights,
                                                 std::shared_ptr<TensorBase> &output)
    {
        PERF_SCOPED_TIMER("DistributedTransformerPipeline::execute");
        start_time_ = std::chrono::high_resolution_clock::now();

        if (!validate(weights))
        {
            LOG_ERROR("DistributedTransformerPipeline: Weight validation failed");
            return false;
        }

        int seq_len = token_ids.size();
        const bool capture_baseline = llaminar::debugEnv().baseline.capture;
        const bool compare_baseline = llaminar::debugEnv().baseline.compare;
        if (capture_baseline && getRank() == 0)
        {
            PrefillBaselineRegistry::instance().clear();
            LOG_DEBUG("[PrefillBaseline] cleared registry at execute start");
        }
        if (seq_len <= 0 || seq_len > config_.max_seq_len)
        {
            LOG_ERROR("DistributedTransformerPipeline: Invalid sequence length " << seq_len);
            return false;
        }

        // Dynamic KV cache initial sizing if requested (placeholder capacity expanded to prefill length)
        if (use_kv_cache_ && kv_cache_dynamic_init_)
        {
            if (kv_cache_state_.capacity_tokens < seq_len)
            {
                if (getRank() == 0)
                {
                    LOG_INFO("[KVCacheInit] dynamic prefill resize trigger old_cap=" << kv_cache_state_.capacity_tokens
                                                                                     << " seq_len=" << seq_len);
                }
                initializeKVCache(std::min(seq_len, config_.max_seq_len));
            }
        }
        // Replicated small-sequence fast path (avoid distributed partition producing zero-length shards).
        // Trigger when global sequence length is smaller than number of ranks.
        int world_size = getSize();
        if (seq_len < world_size)
        {
            small_seq_fast_path_calls_.fetch_add(1, std::memory_order_relaxed);
            LOG_DEBUG("[SmallSeqFastPath] Activate (seq_len=" << seq_len << ", world=" << world_size
                                                              << ") rank=" << getRank());
            // Local naive implementation performed entirely on rank 0, then broadcast.
            if (getRank() == 0)
            {
                LOG_TRACE("[SmallSeqFastPath] Rank 0 begin local forward");
                // Allocate / resize output if needed
                if (!output || output->shape().size() != 2 || output->shape()[0] != seq_len || output->shape()[1] != config_.vocab_size)
                {
                    output = createLocalTensor({seq_len, config_.vocab_size});
                }

                auto embed_shape = weights.token_embedding->shape();
                const float *embedding_data = weights.token_embedding->data();
                // Temporary buffers (row-major)
                std::vector<float> hidden(seq_len * config_.d_model, 0.f);
                std::vector<float> tmp(seq_len * config_.d_model, 0.f);
                auto rmsnorm = [&](std::vector<float> &mat, const float *wn)
                {
                    kernels::RMSNormExecOptions opts; // default heuristics
                    kernels::rmsnorm_row_major_fused(
                        mat.data(),
                        wn,
                        mat.data(),
                        (size_t)seq_len,
                        (size_t)config_.d_model,
                        config_.eps,
                        kernels::GammaMode::REPLICATED,
                        0,
                        opts);
                };
                auto matmul = [&](const std::vector<float> &A, const float *B, int k, int n, std::vector<float> &C)
                {
                    // A: (seq_len x k), B: (k x n), C: (seq_len x n)
                    C.assign(seq_len * n, 0.f);
                    for (int i = 0; i < seq_len; ++i)
                    {
                        const float *a_row = &A[i * k];
                        for (int kk = 0; kk < k; ++kk)
                        {
                            float aval = a_row[kk];
                            const float *b_col_base = &B[kk * n];
                            float *c_row = &C[i * n];
                            for (int j = 0; j < n; ++j)
                                c_row[j] += aval * b_col_base[j];
                        }
                    }
                };
                auto elementwise_add = [&](std::vector<float> &A, const std::vector<float> &B)
                {
                    for (size_t i = 0; i < A.size(); ++i)
                        A[i] += B[i];
                };
                auto sigmoid = [](float x)
                { return 1.f / (1.f + std::exp(-x)); };
                auto swiglu = [&](const std::vector<float> &up, const std::vector<float> &gate, std::vector<float> &out, int dim)
                {
                    out.resize(up.size());
                    for (size_t i = 0; i < up.size(); ++i)
                        out[i] = up[i] * sigmoid(gate[i]);
                };

                // Embedding lookup
                for (int t = 0; t < seq_len; ++t)
                {
                    int tok = token_ids[t];
                    const float *src = &embedding_data[tok * config_.d_model];
                    std::memcpy(&hidden[t * config_.d_model], src, sizeof(float) * config_.d_model);
                }

                // Iterate layers (simplified attention: use average of Q and V as residual proxy)
                for (int layer = 0; layer < config_.n_layers; ++layer)
                {
                    // Attention norm
                    rmsnorm(hidden, weights.attn_norm_weight[layer]->data());
                    // Q,K,V projections (we only compute Q & V for simplified context)
                    std::vector<float> Q, V;
                    matmul(hidden, weights.wq[layer]->data(), config_.d_model, config_.n_head * config_.head_dim, tmp);
                    Q = tmp; // (seq_len x proj)
                    matmul(hidden, weights.wv[layer]->data(), config_.d_model, config_.n_head_kv * config_.head_dim, tmp);
                    V = tmp;
                    // Simplified attention: context = average over sequence of V added to Q slice to keep magnitudes reasonable
                    std::vector<float> context = Q;
                    int ctx_dim = config_.n_head * config_.head_dim;
                    std::vector<float> v_mean(ctx_dim, 0.f);
                    for (int i = 0; i < seq_len; ++i)
                    {
                        const float *vrow = &V[i * ctx_dim];
                        for (int j = 0; j < ctx_dim; ++j)
                            v_mean[j] += vrow[j];
                    }
                    for (int j = 0; j < ctx_dim; ++j)
                        v_mean[j] /= std::max(1, seq_len);
                    for (int i = 0; i < seq_len; ++i)
                    {
                        float *crow = &context[i * ctx_dim];
                        for (int j = 0; j < ctx_dim; ++j)
                            crow[j] = 0.5f * (crow[j] + v_mean[j]);
                    }
                    // Output projection Wo (ctx_dim -> d_model)
                    matmul(context, weights.wo[layer]->data(), ctx_dim, config_.d_model, tmp); // tmp now d_model sized rows
                    elementwise_add(tmp, hidden);                                              // residual
                    hidden = tmp;

                    // FFN norm
                    rmsnorm(hidden, weights.ffn_norm_weight[layer]->data());
                    // Gate & Up
                    std::vector<float> gate, up, swiglu_out;
                    matmul(hidden, weights.w_gate[layer]->data(), config_.d_model, config_.d_ff, gate);
                    matmul(hidden, weights.w_up[layer]->data(), config_.d_model, config_.d_ff, up);
                    swiglu(up, gate, swiglu_out, config_.d_ff);
                    // Down projection
                    matmul(swiglu_out, weights.w_down[layer]->data(), config_.d_ff, config_.d_model, tmp);
                    elementwise_add(tmp, hidden); // residual
                    hidden = tmp;
                }

                // Final norm
                rmsnorm(hidden, weights.output_norm_weight->data());
                // LM head
                std::vector<float> logits;
                matmul(hidden, weights.lm_head->data(), config_.d_model, config_.vocab_size, logits);
                // Copy into output tensor
                float *out_data = const_cast<float *>(output->data());
                std::memcpy(out_data, logits.data(), sizeof(float) * logits.size());

                // Debug: log first few logits before broadcast
                int preview = std::min(5, static_cast<int>(logits.size()));
                std::ostringstream oss;
                oss << "Small-seq fast path (pre-broadcast) first " << preview << " logits: ";
                for (int i = 0; i < preview; ++i)
                    oss << logits[i] << (i + 1 < preview ? ' ' : '\0');
                LOG_DEBUG(oss.str());
                LOG_TRACE("[SmallSeqFastPath] Rank 0 finishing local forward");
            }

            // Broadcast output to all ranks so tests that read it on rank 0 or others are consistent
            // First ensure output tensor allocated on non-root ranks
            if (getRank() != 0)
            {
                if (!output || output->shape().size() != 2 || output->shape()[0] != seq_len || output->shape()[1] != config_.vocab_size)
                {
                    output = createLocalTensor({seq_len, config_.vocab_size});
                }
            }
            // Broadcast data
            int count = seq_len * config_.vocab_size;
            LOG_TRACE("[SmallSeqFastPath] Rank " << getRank() << " entering broadcast of " << count << " floats");
            checkMPIError(MPI_Bcast(const_cast<float *>(output->data()), count, MPI_FLOAT, 0, getComm()), "MPI_Bcast small-seq output");
            LOG_TRACE("[SmallSeqFastPath] Rank " << getRank() << " broadcast complete");

            // Post-broadcast verification log
            {
                int preview = std::min(5, count);
                const float *data = output->data();
                std::ostringstream oss;
                oss << "Small-seq fast path (post-broadcast) rank " << getRank() << " first " << preview << " logits: ";
                for (int i = 0; i < preview; ++i)
                    oss << data[i] << (i + 1 < preview ? ' ' : '\0');
                LOG_DEBUG(oss.str());
            }

            // Bookkeeping timing
            // Finish timing (not aggregated into per-stage stats beyond existing counters)
            LOG_DEBUG("[SmallSeqFastPath] Complete rank=" << getRank());
            if (compare_baseline && getRank() == 0)
            {
                PrefillBaselineRegistry::instance().clear();
                LOG_DEBUG("[PrefillBaseline] cleared registry after comparison run (small seq path)");
            }
            if (use_kv_cache_)
            {
                kv_cache_state_.used_tokens = seq_len;
                n_past_ = seq_len;
            }
            return true;
        }

        // Create intermediate tensors for layer-by-layer processing
        auto intermediate_tensors = createIntermediateTensors(seq_len);
        auto current_input = intermediate_tensors[0];
        auto layer_output = intermediate_tensors[1];

        // 1. Embedding lookup (rank 0 computes, broadcasts to all)
        auto embedding_start = std::chrono::high_resolution_clock::now();
        if (!executeEmbedding(token_ids, weights.token_embedding, current_input))
        {
            LOG_ERROR("DistributedTransformerPipeline: Embedding execution failed");
            return false;
        }
        auto embedding_end = std::chrono::high_resolution_clock::now();
        total_embedding_time_ += std::chrono::duration<double, std::milli>(embedding_end - embedding_start).count();

        // Optional embedding parity spot-check: compare computed embedding rows against direct table lookup
        if (rank_ == 0 && debugEnv().embedding_diag.parity)
        {
            try
            {
                const auto &emb_shape = weights.token_embedding->shape();
                if (emb_shape.size() == 2)
                {
                    const int vocab = static_cast<int>(emb_shape[0]);
                    const int d_model = static_cast<int>(emb_shape[1]);
                    const float *table = weights.token_embedding->data();
                    const float *produced = current_input->data();
                    int produced_seq = seq_len; // current_input shape already (seq_len, d_model)
                    // Determine rows to check
                    std::vector<int> rows = debugEnv().layer_capture.tokens;
                    if (rows.empty())
                        rows.push_back(0);
                    struct DiffSummary
                    {
                        double rel_l2 = 0, max_abs = 0, mean_abs = 0;
                        size_t worst_index = 0;
                        float a = 0, b = 0;
                    };
                    auto compute_summary = [&](const float *a, const float *b, int n)
                    {
                        double sum_sq=0, ref_sq=0, sum_abs=0; double max_abs=0; size_t worst=0; float wa=0, wb=0;
                        for(int i=0;i<n;++i){ double diff = static_cast<double>(a[i])-b[i]; double ad = std::abs(diff); sum_sq += diff*diff; ref_sq += static_cast<double>(b[i])*b[i]; sum_abs += ad; if (ad>max_abs){ max_abs=ad; worst=i; wa=a[i]; wb=b[i]; } }
                        DiffSummary ds; ds.rel_l2 = ref_sq>0 ? std::sqrt(sum_sq)/std::sqrt(ref_sq) : 0.0; ds.max_abs=max_abs; ds.mean_abs = n>0 ? sum_abs/n : 0; ds.worst_index=worst; ds.a=wa; ds.b=wb; return ds; };
                    for (int r : rows)
                    {
                        if (r < 0 || r >= produced_seq)
                            continue;
                        int tok_id = (r < (int)token_ids.size() ? token_ids[r] : -1);
                        if (tok_id < 0 || tok_id >= vocab)
                            continue;
                        const float *ref_row = table + (size_t)tok_id * d_model;
                        const float *prod_row = produced + (size_t)r * d_model;
                        auto ds = compute_summary(prod_row, ref_row, d_model);
                        if (ds.max_abs > 1e-3 || ds.rel_l2 > 1e-5)
                        {
                            LOG_WARN("[EmbeddingParity] row=" << r << " token=" << tok_id << " rel_l2=" << ds.rel_l2 << " max_abs=" << ds.max_abs << " mean_abs=" << ds.mean_abs << " worst_col=" << ds.worst_index << " produced=" << ds.a << " ref=" << ds.b);
                        }
                        else
                        {
                            LOG_INFO("[EmbeddingParity] row=" << r << " token=" << tok_id << " rel_l2=" << ds.rel_l2 << " max_abs=" << ds.max_abs << " mean_abs=" << ds.mean_abs);
                        }
                    }
                }
                else
                {
                    LOG_WARN("[EmbeddingParity] embedding weight shape rank !=2; skipping");
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN("[EmbeddingParity] parity check failed: " << e.what());
            }
        }

        // 2. Execute transformer layers sequentially
        for (int layer_idx = 0; layer_idx < config_.n_layers; ++layer_idx)
        {
            if (!executeTransformerLayer(layer_idx, current_input, weights, layer_output))
            {
                LOG_ERROR("DistributedTransformerPipeline: Layer " << layer_idx << " execution failed");
                return false;
            }

            // Swap tensors for next layer
            std::swap(current_input, layer_output);
        }

        // 3. Final output projection
        if (!executeOutputProjection(current_input, weights, output))
        {
            LOG_ERROR("DistributedTransformerPipeline: Output projection failed");
            return false;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(end_time - start_time_).count();

        if (getRank() == 0)
        {
            LOG_INFO("DistributedTransformerPipeline completed: " << seq_len << " tokens, "
                                                                  << config_.n_layers << " layers in " << std::fixed << std::setprecision(2)
                                                                  << total_time << "ms");
            LOG_DEBUG("Breakdown - Embedding: " << total_embedding_time_ << "ms, "
                                                << "Attention: " << total_attention_time_ << "ms, "
                                                << "Linear: " << total_linear_time_ << "ms, "
                                                << "Norm: " << total_norm_time_ << "ms");
        }

        if (compare_baseline && getRank() == 0)
        {
            PrefillBaselineRegistry::instance().clear();
            LOG_DEBUG("[PrefillBaseline] cleared registry after comparison run");
        }
        // Sanity: ensure prefill sequence length fits within allocated KV cache capacity.
        // Rare edge: some test harness env mutation ordering caused dynamic_init path to miss the early
        // resize (capacity still placeholder=1 here). Provide a last-chance re-init if dynamic_init enabled
        // and no tokens have been written yet (used_tokens==0) to avoid a spurious failure.
        if (use_kv_cache_ && seq_len > kv_cache_state_.capacity_tokens)
        {
            if (kv_cache_dynamic_init_ && kv_cache_state_.used_tokens == 0)
            {
                if (getRank() == 0)
                    LOG_WARN("[KVCache] late dynamic resize fallback triggering (cap=" << kv_cache_state_.capacity_tokens
                                                                                       << " -> " << seq_len << ")");
                initializeKVCache(std::min(seq_len, config_.max_seq_len));
            }
            if (seq_len > kv_cache_state_.capacity_tokens)
            {
                LOG_ERROR("[KVCache] seq_len=" << seq_len << " exceeds capacity_tokens=" << kv_cache_state_.capacity_tokens
                                               << " after prefill; this indicates initialization logic failure");
                return false;
            }
        }
        // KV cache bookkeeping (standard prefill path). The small-seq fast path and
        // some early-return ablation paths already update these counters, but the
        // main path previously forgot to set them, leaving used_tokens at 0 which
        // breaks incremental decode + growth logic expectations.
        if (use_kv_cache_)
        {
            n_past_ = seq_len;
            kv_cache_state_.used_tokens = seq_len;
            if (getRank() == 0)
                LOG_INFO("[KVCacheSummary] post_prefill used_tokens=" << kv_cache_state_.used_tokens
                                                                      << " capacity=" << kv_cache_state_.capacity_tokens
                                                                      << " growth_events=" << kv_cache_state_.growth_events);
        }
        return true;
    }

    bool DistributedTransformerPipeline::executeEmbedding(const std::vector<int> &token_ids,
                                                          const std::shared_ptr<TensorBase> &embedding_weight,
                                                          std::shared_ptr<TensorBase> &embedded_output)
    {
        int seq_len = token_ids.size();

        // Ensure all ranks have embedded_output tensor allocated with correct size
        // This is crucial for MPI_Bcast to work properly
        if (!embedded_output || embedded_output->size() != seq_len * config_.d_model)
        {
            // Reallocate if size mismatch
            embedded_output = createBroadcastTensor({seq_len, config_.d_model});
        }

        // Rank 0 performs embedding lookup using registered kernel
        if (getRank() == 0)
        {
            // Create token_ids tensor
            auto token_ids_tensor = createLocalTensor({seq_len});
            for (int i = 0; i < seq_len; ++i)
            {
                token_ids_tensor->data()[i] = static_cast<float>(token_ids[i]);
            }

            std::vector<std::shared_ptr<TensorBase>> inputs = {token_ids_tensor, embedding_weight};
            std::vector<std::shared_ptr<TensorBase>> outputs = {embedded_output};

            // === EMBEDDING STAGE INSTRUMENTATION ===
            ASSERT_TENSOR_VALID(token_ids_tensor, "Embedding token_ids");
            ASSERT_TENSOR_VALID(embedding_weight, "Embedding weight");
            TensorLogger::logTensorStats(token_ids_tensor, "token_ids", "EMBEDDING_INPUT");
            TensorLogger::logTensorStats(embedding_weight, "embedding_weight", "EMBEDDING_INPUT");

            // Execute using registered embedding kernel
            if (!executeKernel("embedding", inputs, outputs))
            {
                LOG_ERROR("Embedding kernel execution failed");
                return false;
            }

            // === POST-EMBEDDING VALIDATION ===
            ASSERT_TENSOR_NOT_NAN(embedded_output, "Embedding output");
            TensorLogger::logTensorStats(embedded_output, "embedded_output", "EMBEDDING_OUTPUT");
        }

        // Broadcast embedded sequence to all ranks using PipelineBase utility
        if (!broadcastTensor(embedded_output, 0))
        {
            LOG_ERROR("Embedding broadcast failed");
            return false;
        }

        if (!debugEnv().embedding.trace_rows_spec.empty())
        {
            auto rows = parseRowSpecification(debugEnv().embedding.trace_rows_spec.c_str(), static_cast<size_t>(embedded_output->shape()[0]));
            if (!rows.empty())
            {
                logTensorRowPreview(embedded_output, "embedding_output", rows, 8, "EMBEDDING_TRACE");
            }
        }

        const bool capture_baseline = debugEnv().baseline.capture;
        const bool compare_baseline = debugEnv().baseline.compare;
        if (capture_baseline || compare_baseline)
        {
            handle_prefill_stage_snapshot(getRank(),
                                          "embedding_output",
                                          embedded_output->data(),
                                          static_cast<size_t>(embedded_output->size()),
                                          config_.d_model,
                                          5e-4,
                                          capture_baseline,
                                          compare_baseline);
        }

        LOG_DEBUG("Embedding completed: " << seq_len << " tokens -> "
                                          << seq_len << "x" << config_.d_model << " on rank " << getRank());

        return true;
    }

    bool DistributedTransformerPipeline::executeTransformerLayer(int layer_idx,
                                                                 std::shared_ptr<TensorBase> &input,
                                                                 const ModelWeights &weights,
                                                                 std::shared_ptr<TensorBase> &output)
    {
        PERF_SCOPED_TIMER("DistributedTransformerPipeline::executeTransformerLayer");
        int seq_len = input->shape()[0];

        // Early one-shot diagnostic for layer token diff capture flags (before any potential early returns)
        {
            static bool printed_early_token_diff = false;
            if (!printed_early_token_diff && getRank() == 0)
            {
                printed_early_token_diff = true;
                const char *raw_env = std::getenv("LLAMINAR_PIPELINE_LAYER_TOKEN_DIFF");
                const char *raw_env_verbose = std::getenv("LLAMINAR_PIPELINE_LAYER_TOKEN_DIFF_VERBOSE");
                LOG_INFO(std::string("[LayerTokenDiffEarly] snapshot.layer_token_diff=") + (debugEnv().pipeline.layer_token_diff ? "1" : "0") +
                         " snapshot.layer_token_diff_verbose=" + (debugEnv().pipeline.layer_token_diff_verbose ? "1" : "0") +
                         " raw_env=" + (raw_env ? raw_env : "<unset>") +
                         " raw_env_verbose=" + (raw_env_verbose ? raw_env_verbose : "<unset>"));
            }
        }

        // Create temporary tensors for layer computation
        auto attn_norm_out = createLocalTensor({seq_len, config_.d_model});
        auto attn_out = createLocalTensor({seq_len, config_.d_model});
        auto ffn_norm_out = createLocalTensor({seq_len, config_.d_model});
        auto ffn_out = createLocalTensor({seq_len, config_.d_model});
        auto residual_tmp = createLocalTensor({seq_len, config_.d_model});

        // 1. Attention path: RMSNorm -> Attention -> Residual (with optional ablation & activation capture)
        struct LayerAblationRuntime
        {
            bool initialized = false;
            std::vector<int> rows;
        };
        static LayerAblationRuntime labl_rt;
        const auto &abl = debugEnv().ablation;
        const auto &cap = debugEnv().layer_capture;
        if (!labl_rt.initialized)
        {
            if (cap.capture)
            {
                labl_rt.rows = cap.tokens;
                if (labl_rt.rows.empty())
                    labl_rt.rows.push_back(0);
            }
            labl_rt.initialized = true;
            if (getRank() == 0)
            {
                LOG_INFO("[AblationConfig] attention=" << (abl.ablate_attention ? "ON" : "OFF")
                                                       << " ffn=" << (abl.ablate_ffn ? "ON" : "OFF")
                                                       << " capture=" << (cap.capture ? "ON" : "OFF"));
            }
        }
        auto capture_rows = [&](const std::string &tag, const std::shared_ptr<TensorBase> &tensor)
        {
            if (!cap.capture || getRank() != 0 || !tensor || tensor->shape().size() != 2)
                return;
            try
            {
                std::filesystem::create_directories("layer_activations");
                auto sh = tensor->shape();
                int rows = sh[0];
                int cols = sh[1];
                for (int r : labl_rt.rows)
                {
                    if (r < 0 || r >= rows)
                        continue;
                    const float *row_ptr = tensor->data() + (size_t)r * cols;
                    std::ostringstream fn;
                    fn << "layer_activations/layer_" << layer_idx << "_" << tag << "_row" << r << ".txt";
                    std::ofstream ofs(fn.str(), std::ios::out | std::ios::trunc);
                    if (!ofs.good())
                        continue;
                    for (int c = 0; c < cols; ++c)
                    {
                        ofs << row_ptr[c];
                        if (c + 1 < cols)
                            ofs << ' ';
                    }
                    ofs << '\n';
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN("Activation capture failed (" << tag << "): " << e.what());
            }
        };
        DEBUG_ASSERT(input, "Input tensor null before layer " + std::to_string(layer_idx));
        ASSERT_TENSOR_NOT_NAN(input, "Input tensor has NaN before layer " + std::to_string(layer_idx));
        TensorLogger::logTensorStats(input, "layer_" + std::to_string(layer_idx) + "_input");

        const int hidden_size = config_.d_model;
        const int d_ff = config_.d_ff;

        // ---------------- RMSNorm Forensics (env gated) ------------------
        struct RMSForensicsEnv
        {
            bool enabled = false;       // Master enable
            std::vector<int> layers;    // Layer indices to instrument (empty => all)
            std::vector<int> rows;      // Row indices to preview (empty => none)
            double warn_rel_l2 = 1e-5;  // Warning threshold
            bool trace_vectors = false; // Dump selected row vectors
            bool diff_only = false;     // Skip stats if true (just rel_l2)
            bool once_parsed = false;   // Internal sentinel
        };
        static RMSForensicsEnv rms_env; // static retains parse across layers

        auto parse_index_spec = [](const char *spec) -> std::vector<int>
        {
            std::vector<int> out;
            if (!spec || !*spec)
                return out;
            std::string s(spec);
            size_t pos = 0;
            while (pos < s.size())
            {
                size_t comma = s.find(',', pos);
                std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                auto trim = [](std::string &x)
                {
                    size_t a = x.find_first_not_of(" \t");
                    size_t b = x.find_last_not_of(" \t");
                    if (a == std::string::npos)
                    {
                        x.clear();
                        return;
                    }
                    x = x.substr(a, b - a + 1);
                };
                trim(tok);
                if (!tok.empty())
                {
                    size_t dash = tok.find('-');
                    if (dash != std::string::npos)
                    {
                        std::string a = tok.substr(0, dash);
                        trim(a);
                        std::string b = tok.substr(dash + 1);
                        trim(b);
                        try
                        {
                            int ia = std::stoi(a);
                            int ib = std::stoi(b);
                            if (ia <= ib)
                            {
                                for (int v = ia; v <= ib; ++v)
                                    out.push_back(v);
                            }
                        }
                        catch (...)
                        {
                        }
                    }
                    else
                    {
                        try
                        {
                            out.push_back(std::stoi(tok));
                        }
                        catch (...)
                        {
                        }
                    }
                }
                if (comma == std::string::npos)
                    break;
                else
                    pos = comma + 1;
            }
            std::sort(out.begin(), out.end());
            out.erase(std::unique(out.begin(), out.end()), out.end());
            return out;
        };

        if (!rms_env.once_parsed)
        {
            const auto &rf = debugEnv().rms_forensics;
            rms_env.enabled = rf.enabled;
            rms_env.layers = rf.layers;
            rms_env.rows = rf.rows;
            rms_env.warn_rel_l2 = rf.warn_rel_l2;
            rms_env.trace_vectors = rf.trace_vectors;
            rms_env.diff_only = rf.diff_only;
            rms_env.once_parsed = true;
            if (getRank() == 0 && rms_env.enabled)
            {
                LOG_INFO("[RMSForensics] enabled warn_rel_l2=" << rms_env.warn_rel_l2
                                                               << " layers_spec=" << (rf.layers_spec.empty() ? "<all>" : rf.layers_spec)
                                                               << " rows_spec=" << (rf.rows_spec.empty() ? "<none>" : rf.rows_spec)
                                                               << " trace_vectors=" << rms_env.trace_vectors
                                                               << " diff_only=" << rms_env.diff_only);
            }
        }

        auto layer_selected = [&](int li) -> bool
        {
            if (!rms_env.enabled)
                return false;
            if (rms_env.layers.empty())
                return true;
            return std::binary_search(rms_env.layers.begin(), rms_env.layers.end(), li);
        };

        struct RMSStats
        {
            double min = 0, max = 0, mean = 0, sumsq = 0;
        };
        auto compute_stats = [&](const float *data, size_t elements, int cols) -> RMSStats
        {
            RMSStats st;
            if (!data || elements == 0)
                return st;
            double mn = std::numeric_limits<double>::infinity();
            double mx = -std::numeric_limits<double>::infinity();
            long double sum = 0.0L;
            long double sumsq = 0.0L;
            for (size_t i = 0; i < elements; ++i)
            {
                double v = data[i];
                if (v < mn)
                    mn = v;
                if (v > mx)
                    mx = v;
                sum += v;
                sumsq += v * v;
            }
            st.min = mn == std::numeric_limits<double>::infinity() ? 0.0 : mn;
            st.max = mx == -std::numeric_limits<double>::infinity() ? 0.0 : mx;
            st.mean = static_cast<double>(sum / (long double)std::max<size_t>(1, elements));
            st.sumsq = static_cast<double>(sumsq);
            return st;
        };

        auto rel_l2_vs = [&](const float *a, const float *b, size_t n, double &max_abs, size_t &worst_index) -> double
        {
            if (!a || !b || n == 0)
            {
                max_abs = 0;
                worst_index = 0;
                return 0.0;
            }
            long double dsq = 0, rsq = 0;
            max_abs = 0;
            worst_index = 0;
            for (size_t i = 0; i < n; ++i)
            {
                double diff = (double)a[i] - (double)b[i];
                double ref = (double)b[i];
                double ad = std::fabs(diff);
                if (ad > max_abs)
                {
                    max_abs = ad;
                    worst_index = i;
                }
                dsq += diff * diff;
                rsq += ref * ref;
            }
            return rsq > 0 ? std::sqrt((double)dsq) / std::sqrt((double)rsq) : 0.0;
        };

        auto format_row_preview = [&](const char *tag, const float *pre, const float *post, const float *ref, int cols, int row)
        {
            if (!pre || !post)
                return;
            if (row < 0 || row >= seq_len)
                return;
            if (!rms_env.trace_vectors)
                return;
            if (getRank() != 0)
                return;
            size_t base = (size_t)row * cols;
            size_t show = std::min(cols, 16); // limit preview
            std::ostringstream oss;
            oss << "[RMSForensics][Row] layer=" << layer_idx << " tag=" << tag << " row=" << row;
            oss << " pre=";
            for (size_t c = 0; c < show; ++c)
                oss << pre[base + c] << (c + 1 < show ? "," : "");
            oss << " post=";
            for (size_t c = 0; c < show; ++c)
                oss << post[base + c] << (c + 1 < show ? "," : "");
            if (ref)
            {
                oss << " ref=";
                for (size_t c = 0; c < show; ++c)
                    oss << ref[base + c] << (c + 1 < show ? "," : "");
            }
            LOG_INFO(oss.str());
        };
        // --------------- End RMSNorm Forensics setup ----------------------

        const auto &pd = debugEnv().prefill_debug;
        const bool trace_io = pd.trace_io;
        const bool debug_compare = pd.debug_compare;
        const bool debug_attention = pd.debug_attention;
        const bool debug_output = pd.debug_output;
        const bool run_reference = !(abl.ablate_attention || abl.ablate_ffn) && (trace_io || debug_compare || debug_attention || debug_output);

        const bool capture_baseline = debugEnv().baseline.capture;
        const bool compare_baseline = debugEnv().baseline.compare;
        const int baseline_rank = getRank();
        auto baseline_snapshot = [&](const std::string &name,
                                     const float *data,
                                     size_t count,
                                     int cols,
                                     double warn_threshold)
        {
            handle_prefill_stage_snapshot(baseline_rank, name, data, count, cols, warn_threshold, capture_baseline, compare_baseline);
        };

        const char *diag_tag = "[PrefillFFN]";
        if (trace_io)
        {
            diag_tag = "[PrefillDiag]";
        }
        else if (debug_attention)
        {
            diag_tag = "[PrefillAttention]";
        }
        else if (debug_output)
        {
            diag_tag = "[PrefillOutput]";
        }
        else if (debug_compare)
        {
            diag_tag = "[PrefillCompare]";
        }

        auto log_stage_diff = [&](const std::string &name,
                                  const float *actual,
                                  const float *reference,
                                  size_t count,
                                  int cols,
                                  double warn_threshold)
        {
            if (!run_reference || getRank() != 0 || !actual || !reference || count == 0)
            {
                return;
            }
            auto diff = computeDiffSummary(actual, reference, count);
            auto samples = collectTopDiffSamples(actual, reference, count, 8);
            if (diff.rel_l2 > warn_threshold)
            {
                LOG_WARN(diag_tag << " " << name << " diff rel_l2=" << diff.rel_l2
                                  << " max_abs=" << diff.max_abs
                                  << " mean_abs=" << diff.mean_abs
                                  << " worst_index=" << diff.worst_index
                                  << " cosma=" << diff.value_a
                                  << " ref=" << diff.value_b);
            }
            else
            {
                LOG_INFO(diag_tag << " " << name << " diff rel_l2=" << diff.rel_l2
                                  << " max_abs=" << diff.max_abs
                                  << " mean_abs=" << diff.mean_abs
                                  << " worst_index=" << diff.worst_index
                                  << " cosma=" << diff.value_a
                                  << " ref=" << diff.value_b);
            }
            LOG_INFO(diag_tag << " " << name << " diff samples: " << formatDiffSamples(samples, cols));
        };

        std::vector<float> residual_ref_storage;
        std::vector<float> ffn_norm_ref_storage;
        std::vector<float> gate_ref_storage;
        std::vector<float> up_ref_storage;
        std::vector<float> swiglu_ref_storage;
        std::vector<float> ffn_down_ref_storage;
        std::vector<float> final_ref_storage;

        PrefillAttentionTiming cosma_timing;
        if (!abl.ablate_attention)
        {
            BackendContext bctx;
            bctx.is_prefill = is_prefill_stage_; // explicit stage flag set by adapter / external caller
            bctx.seq_len = seq_len;
            bctx.d_model = config_.d_model;
            bctx.n_layers = config_.n_layers;
            bctx.world = getSize();
            BackendDecision backend_dec = selectAttentionBackend(bctx);
            if (backend_dec.use_cosma())
            {
                if (!executePrefillAttentionCosma(layer_idx, input, weights, attn_norm_out, attn_out, cosma_timing))
                {
                    LOG_ERROR("Layer " << layer_idx << " COSMA prefill attention path failed");
                    return false;
                }
                total_norm_time_ += cosma_timing.norm_ms;
                total_attention_time_ += cosma_timing.attention_ms;
                total_linear_time_ += cosma_timing.linear_ms;
                LOG_DEBUG("[backend] attention layer=" << layer_idx << " backend=COSMA stage=" << (is_prefill_stage_ ? "prefill" : "decode") << " reason=" << backend_dec.reason);
            }
            else
            {
                LOG_DEBUG("[backend] attention layer=" << layer_idx << " backend=OpenBLAS stage=" << (is_prefill_stage_ ? "prefill" : "decode") << " reason=" << backend_dec.reason);
                auto norm_start = std::chrono::high_resolution_clock::now();

                // Attention pre-norm using registered RMSNorm kernel
                std::vector<std::shared_ptr<TensorBase>> norm_inputs = {input, weights.attn_norm_weight[layer_idx]};
                std::vector<std::shared_ptr<TensorBase>> norm_outputs = {attn_norm_out};

                // Pre-attention RMS forensics (non-COSMA path only for now)
                std::vector<float> attn_ref_storage; // reference normalized row-major
                RMSStats attn_pre_stats;
                RMSStats attn_post_stats;
                double attn_rel_l2 = 0.0;
                double attn_max_abs = 0;
                size_t attn_worst_idx = 0;
                if (layer_selected(layer_idx) && getRank() == 0)
                {
                    size_t elems = (size_t)seq_len * hidden_size;
                    attn_pre_stats = compute_stats(input->data(), elems, hidden_size);
                }

                if (!executeKernel("rmsnorm", norm_inputs, norm_outputs))
                {
                    LOG_ERROR("Layer " << layer_idx << " attention norm failed");
                    return false;
                }

                auto norm_end = std::chrono::high_resolution_clock::now();
                total_norm_time_ += std::chrono::duration<double, std::milli>(norm_end - norm_start).count();

                if (layer_selected(layer_idx) && getRank() == 0)
                {
                    size_t elems = (size_t)seq_len * hidden_size;
                    attn_post_stats = compute_stats(attn_norm_out->data(), elems, hidden_size);
                    // Build reference normalization if gamma available
                    if (weights.attn_norm_weight[layer_idx] && weights.attn_norm_weight[layer_idx]->data())
                    {
                        const float *gamma = weights.attn_norm_weight[layer_idx]->data();
                        attn_ref_storage.resize(elems);
                        for (int r = 0; r < seq_len; ++r)
                        {
                            const float *row_in = input->data() + (size_t)r * hidden_size;
                            double sum = 0.0;
                            for (int c = 0; c < hidden_size; ++c)
                            {
                                double v = row_in[c];
                                sum += v * v;
                            }
                            double inv = 1.0 / std::sqrt(sum / std::max(1, hidden_size) + config_.eps);
                            float *row_out = attn_ref_storage.data() + (size_t)r * hidden_size;
                            for (int c = 0; c < hidden_size; ++c)
                            {
                                row_out[c] = (float)(row_in[c] * inv * gamma[c]);
                            }
                        }
                        attn_rel_l2 = rel_l2_vs(attn_norm_out->data(), attn_ref_storage.data(), elems, attn_max_abs, attn_worst_idx);
                    }
                    if (!rms_env.diff_only)
                    {
                        double pre_rms = std::sqrt(attn_pre_stats.sumsq / std::max<size_t>(1, (size_t)seq_len * hidden_size));
                        double post_rms = std::sqrt(attn_post_stats.sumsq / std::max<size_t>(1, (size_t)seq_len * hidden_size));
                        size_t worst_row = hidden_size ? attn_worst_idx / (size_t)hidden_size : 0;
                        size_t worst_col = hidden_size ? attn_worst_idx % (size_t)hidden_size : 0;
                        std::ostringstream oss;
                        oss.setf(std::ios::fixed);
                        oss.precision(6);
                        oss << "[RMSSummary] layer=" << layer_idx << " kind=attn"
                            << " rows=" << seq_len << " cols=" << hidden_size
                            << " pre_min=" << attn_pre_stats.min << " pre_max=" << attn_pre_stats.max
                            << " pre_mean=" << attn_pre_stats.mean << " pre_rms=" << pre_rms
                            << " post_min=" << attn_post_stats.min << " post_max=" << attn_post_stats.max
                            << " post_mean=" << attn_post_stats.mean << " post_rms=" << post_rms;
                        if (!attn_ref_storage.empty())
                        {
                            oss << " rel_l2_ref=" << attn_rel_l2 << " max_abs_ref=" << attn_max_abs
                                << " worst_row=" << worst_row << " worst_col=" << worst_col;
                        }
                        if (!attn_ref_storage.empty() && attn_rel_l2 > rms_env.warn_rel_l2)
                        {
                            LOG_WARN(oss.str());
                        }
                        else
                        {
                            LOG_INFO(oss.str());
                        }
                        for (int row : rms_env.rows)
                        {
                            format_row_preview("attn", input->data(), attn_norm_out->data(), attn_ref_storage.empty() ? nullptr : attn_ref_storage.data(), hidden_size, row);
                        }
                    }
                    else if (!attn_ref_storage.empty())
                    {
                        LOG_INFO("[RMSSummary] layer=" << layer_idx << " kind=attn rel_l2_ref=" << attn_rel_l2 << " max_abs_ref=" << attn_max_abs);
                    }
                }

                // Multi-head attention with MPI distribution using registered attention kernel
                auto attn_start = std::chrono::high_resolution_clock::now();

                // Set sequence position in attention kernel
                auto attention_kernel = dynamic_cast<MPIAttentionKernel *>(getKernel("attention"));
                if (attention_kernel)
                {
                    attention_kernel->setSequencePosition(n_past_);
                }

                std::vector<std::shared_ptr<TensorBase>> attn_inputs = {
                    attn_norm_out,                                                                                            // Input sequence
                    weights.wq[layer_idx],                                                                                    // Wq
                    weights.wk[layer_idx],                                                                                    // Wk
                    weights.wv[layer_idx],                                                                                    // Wv
                    weights.wo[layer_idx],                                                                                    // Wo
                    use_kv_cache_ ? k_cache_[layer_idx] : createLocalTensor({seq_len, config_.n_head_kv * config_.head_dim}), // K cache (or temp)
                    use_kv_cache_ ? v_cache_[layer_idx] : createLocalTensor({seq_len, config_.n_head_kv * config_.head_dim})  // V cache (or temp)
                };
                std::vector<std::shared_ptr<TensorBase>> attn_outputs = {attn_out};

                if (!executeKernel("attention", attn_inputs, attn_outputs))
                {
                    LOG_ERROR("Layer " << layer_idx << " attention failed");
                    return false;
                }

                auto attn_end = std::chrono::high_resolution_clock::now();
                total_attention_time_ += std::chrono::duration<double, std::milli>(attn_end - attn_start).count();
            }

            ASSERT_TENSOR_NOT_NAN(attn_norm_out, "Attention norm output has NaN in layer " + std::to_string(layer_idx));
            TensorLogger::logTensorStats(attn_norm_out, "layer_" + std::to_string(layer_idx) + "_attn_norm_out");
            baseline_snapshot("layer_" + std::to_string(layer_idx) + "_attn_norm_out",
                              attn_norm_out->data(),
                              static_cast<size_t>(seq_len) * hidden_size,
                              hidden_size,
                              1e-5);
            capture_rows("attn_norm_out", attn_norm_out);

            ASSERT_TENSOR_NOT_NAN(attn_out, "Attention output has NaN in layer " + std::to_string(layer_idx));
            TensorLogger::logTensorStats(attn_out, "layer_" + std::to_string(layer_idx) + "_attn_out");
            baseline_snapshot("layer_" + std::to_string(layer_idx) + "_attn_out",
                              attn_out->data(),
                              static_cast<size_t>(seq_len) * hidden_size,
                              hidden_size,
                              1e-3);
            capture_rows("attn_out", attn_out);

            std::vector<std::shared_ptr<TensorBase>> residual_inputs = {input, attn_out};
            std::vector<std::shared_ptr<TensorBase>> residual_outputs = {residual_tmp};
            if (!executeKernel("residual", residual_inputs, residual_outputs))
            {
                LOG_ERROR("Layer " << layer_idx << " attention residual failed");
                return false;
            }
            ASSERT_TENSOR_NOT_NAN(residual_tmp, "Attention residual has NaN in layer " + std::to_string(layer_idx));
            TensorLogger::logTensorStats(residual_tmp, "layer_" + std::to_string(layer_idx) + "_attn_residual");
            baseline_snapshot("layer_" + std::to_string(layer_idx) + "_attn_residual",
                              residual_tmp->data(),
                              static_cast<size_t>(seq_len) * hidden_size,
                              hidden_size,
                              1e-6);
            logFFNRowPreviewIfEnabled("layer_" + std::to_string(layer_idx) + "_attn_residual", residual_tmp);
            capture_rows("attn_residual", residual_tmp);
            if (run_reference && getRank() == 0)
            {
                const float *input_ptr = input->data();
                const float *attn_ptr = attn_out->data();
                size_t elems = (size_t)seq_len * hidden_size;
                residual_ref_storage.resize(elems);
                for (size_t idx = 0; idx < elems; ++idx)
                    residual_ref_storage[idx] = input_ptr[idx] + attn_ptr[idx];
                std::string label = "layer_" + std::to_string(layer_idx) + "_attn_residual";
                log_stage_diff(label, residual_tmp->data(), residual_ref_storage.data(), elems, hidden_size, 1e-6);
            }
        }
        else
        {
            // Attention ablated: copy input directly to residual_tmp and attn_out; zero norm tensor
            size_t bytes = (size_t)seq_len * hidden_size * sizeof(float);
            std::memcpy(residual_tmp->data(), input->data(), bytes);
            std::memcpy(attn_out->data(), input->data(), bytes);
            std::memset(attn_norm_out->data(), 0, bytes);
            if (getRank() == 0)
                LOG_WARN("[Ablation] Layer " << layer_idx << " attention skipped (LLAMINAR_ABLATE_ATTENTION)");
            capture_rows("attn_residual", residual_tmp);
        }

        // 2. Feed-forward path (optional ablation)
        if (abl.ablate_ffn)
        {
            size_t bytes = (size_t)seq_len * hidden_size * sizeof(float);
            std::memcpy(output->data(), residual_tmp->data(), bytes);
            if (getRank() == 0)
                LOG_WARN("[Ablation] Layer " << layer_idx << " FFN skipped (LLAMINAR_ABLATE_FFN)");
            capture_rows("layer_output", output);
            LOG_DEBUG("Layer " << layer_idx << " completed (FFN ablated) on rank " << getRank());
            if (use_kv_cache_)
            {
                kv_cache_state_.used_tokens = seq_len;
                n_past_ = seq_len;
            }
            return true;
        }
        auto norm_start = std::chrono::high_resolution_clock::now();

        // FFN pre-norm using registered RMSNorm kernel
        std::vector<std::shared_ptr<TensorBase>> norm_inputs = {residual_tmp, weights.ffn_norm_weight[layer_idx]};
        std::vector<std::shared_ptr<TensorBase>> norm_outputs = {ffn_norm_out};

        std::vector<float> ffn_ref_storage;
        RMSStats ffn_pre_stats;
        RMSStats ffn_post_stats;
        double ffn_rel_l2 = 0.0;
        double ffn_max_abs = 0;
        size_t ffn_worst_idx = 0;
        if (layer_selected(layer_idx) && getRank() == 0)
        {
            size_t elems = (size_t)seq_len * hidden_size;
            ffn_pre_stats = compute_stats(residual_tmp->data(), elems, hidden_size);
        }

        if (!executeKernel("rmsnorm", norm_inputs, norm_outputs))
        {
            LOG_ERROR("Layer " << layer_idx << " FFN norm failed");
            return false;
        }

        ASSERT_TENSOR_NOT_NAN(ffn_norm_out, "FFN norm output has NaN in layer " + std::to_string(layer_idx));
        TensorLogger::logTensorStats(ffn_norm_out, "layer_" + std::to_string(layer_idx) + "_ffn_norm_out");
        baseline_snapshot("layer_" + std::to_string(layer_idx) + "_ffn_norm_out",
                          ffn_norm_out->data(),
                          static_cast<size_t>(seq_len) * hidden_size,
                          hidden_size,
                          1e-5);
        logFFNRowPreviewIfEnabled("layer_" + std::to_string(layer_idx) + "_ffn_norm_out", ffn_norm_out);
        capture_rows("ffn_norm_out", ffn_norm_out);

        if (layer_selected(layer_idx) && getRank() == 0)
        {
            size_t elems = (size_t)seq_len * hidden_size;
            ffn_post_stats = compute_stats(ffn_norm_out->data(), elems, hidden_size);
            if (weights.ffn_norm_weight[layer_idx] && weights.ffn_norm_weight[layer_idx]->data())
            {
                const float *gamma = weights.ffn_norm_weight[layer_idx]->data();
                ffn_ref_storage.resize(elems);
                for (int r = 0; r < seq_len; ++r)
                {
                    const float *row_in = residual_tmp->data() + (size_t)r * hidden_size;
                    double sum = 0.0;
                    for (int c = 0; c < hidden_size; ++c)
                    {
                        double v = row_in[c];
                        sum += v * v;
                    }
                    double inv = 1.0 / std::sqrt(sum / std::max(1, hidden_size) + config_.eps);
                    float *row_out = ffn_ref_storage.data() + (size_t)r * hidden_size;
                    for (int c = 0; c < hidden_size; ++c)
                    {
                        row_out[c] = (float)(row_in[c] * inv * gamma[c]);
                    }
                }
                ffn_rel_l2 = rel_l2_vs(ffn_norm_out->data(), ffn_ref_storage.data(), elems, ffn_max_abs, ffn_worst_idx);
            }
            if (!rms_env.diff_only)
            {
                double pre_rms = std::sqrt(ffn_pre_stats.sumsq / std::max<size_t>(1, (size_t)seq_len * hidden_size));
                double post_rms = std::sqrt(ffn_post_stats.sumsq / std::max<size_t>(1, (size_t)seq_len * hidden_size));
                size_t worst_row = hidden_size ? ffn_worst_idx / (size_t)hidden_size : 0;
                size_t worst_col = hidden_size ? ffn_worst_idx % (size_t)hidden_size : 0;
                std::ostringstream oss;
                oss.setf(std::ios::fixed);
                oss.precision(6);
                oss << "[RMSSummary] layer=" << layer_idx << " kind=ffn"
                    << " rows=" << seq_len << " cols=" << hidden_size
                    << " pre_min=" << ffn_pre_stats.min << " pre_max=" << ffn_pre_stats.max
                    << " pre_mean=" << ffn_pre_stats.mean << " pre_rms=" << pre_rms
                    << " post_min=" << ffn_post_stats.min << " post_max=" << ffn_post_stats.max
                    << " post_mean=" << ffn_post_stats.mean << " post_rms=" << post_rms;
                if (!ffn_ref_storage.empty())
                {
                    oss << " rel_l2_ref=" << ffn_rel_l2 << " max_abs_ref=" << ffn_max_abs
                        << " worst_row=" << worst_row << " worst_col=" << worst_col;
                }
                if (!ffn_ref_storage.empty() && ffn_rel_l2 > rms_env.warn_rel_l2)
                {
                    LOG_WARN(oss.str());
                }
                else
                {
                    LOG_INFO(oss.str());
                }
                for (int row : rms_env.rows)
                {
                    format_row_preview("ffn", residual_tmp->data(), ffn_norm_out->data(), ffn_ref_storage.empty() ? nullptr : ffn_ref_storage.data(), hidden_size, row);
                }
            }
            else if (!ffn_ref_storage.empty())
            {
                LOG_INFO("[RMSSummary] layer=" << layer_idx << " kind=ffn rel_l2_ref=" << ffn_rel_l2 << " max_abs_ref=" << ffn_max_abs);
            }
        }

        if (run_reference && getRank() == 0)
        {
            const float *gamma_ptr = weights.ffn_norm_weight[layer_idx] ? weights.ffn_norm_weight[layer_idx]->data() : nullptr;
            if (gamma_ptr)
            {
                size_t elems = static_cast<size_t>(seq_len) * hidden_size;
                ffn_norm_ref_storage.resize(elems);
                for (int r = 0; r < seq_len; ++r)
                {
                    const float *row = residual_ref_storage.data() + static_cast<size_t>(r) * hidden_size;
                    double sum = 0.0;
                    for (int c = 0; c < hidden_size; ++c)
                    {
                        double v = static_cast<double>(row[c]);
                        sum += v * v;
                    }
                    double inv = 1.0 / std::sqrt(sum / std::max(1, hidden_size) + config_.eps);
                    float *dst = ffn_norm_ref_storage.data() + static_cast<size_t>(r) * hidden_size;
                    for (int c = 0; c < hidden_size; ++c)
                    {
                        dst[c] = static_cast<float>(row[c] * inv * gamma_ptr[c]);
                    }
                }
                std::string label = "layer_" + std::to_string(layer_idx) + "_ffn_norm";
                log_stage_diff(label, ffn_norm_out->data(), ffn_norm_ref_storage.data(), elems, hidden_size, 1e-5);
            }
        }

        auto norm_end = std::chrono::high_resolution_clock::now();
        total_norm_time_ += std::chrono::duration<double, std::milli>(norm_end - norm_start).count();

        // FFN computation using MPI linear kernels and SwiGLU activation
        auto linear_start = std::chrono::high_resolution_clock::now();

        // Gate and Up projections (parallel)
        auto gate_out = createLocalTensor({seq_len, config_.d_ff});
        auto up_out = createLocalTensor({seq_len, config_.d_ff});

        // Gate projection with adaptive backend (COSMA for large prefill else linear kernel / OpenBLAS)
        {
            BackendContext bctx;
            bctx.is_prefill = is_prefill_stage_;
            bctx.seq_len = seq_len;
            bctx.d_model = config_.d_model;
            bctx.n_layers = config_.n_layers;
            bctx.world = getSize();
            auto dec = selectAttentionBackend(bctx); // Reuse attention heuristic for FFN for now
            if (dec.use_cosma())
            {
                LOG_DEBUG("[backend] ffn_gate layer=" << layer_idx << " backend=COSMA stage=" << (is_prefill_stage_ ? "prefill" : "decode") << " reason=" << dec.reason);
                if (!adaptiveMatMul(ffn_norm_out->data(),
                                    weights.w_gate[layer_idx]->data(),
                                    gate_out->data(),
                                    seq_len,     /*m*/
                                    d_ff,        /*n*/
                                    hidden_size, /*k*/
                                    is_prefill_stage_,
                                    /*distributed_partition*/ false,
                                    /*transpose_A*/ false,
                                    /*transpose_B*/ false,
                                    /*alpha*/ 1.0f,
                                    /*beta*/ 0.0f))
                {
                    LOG_ERROR("Layer " << layer_idx << " gate projection COSMA path failed");
                    return false;
                }
            }
            else
            {
                LOG_DEBUG("[backend] ffn_gate layer=" << layer_idx << " backend=OpenBLAS(linear-kernel) stage=" << (is_prefill_stage_ ? "prefill" : "decode") << " reason=" << dec.reason);
                std::vector<std::shared_ptr<TensorBase>> gate_inputs = {ffn_norm_out, weights.w_gate[layer_idx]};
                std::vector<std::shared_ptr<TensorBase>> gate_outputs = {gate_out};
                if (!executeKernel("linear", gate_inputs, gate_outputs))
                {
                    LOG_ERROR("Layer " << layer_idx << " gate projection failed");
                    return false;
                }
            }
        }

        ASSERT_TENSOR_NOT_NAN(gate_out, "FFN gate output has NaN in layer " + std::to_string(layer_idx));
        TensorLogger::logTensorStats(gate_out, "layer_" + std::to_string(layer_idx) + "_ffn_gate");
        baseline_snapshot("layer_" + std::to_string(layer_idx) + "_ffn_gate",
                          gate_out->data(),
                          static_cast<size_t>(seq_len) * d_ff,
                          d_ff,
                          1e-3);
        traceFFNShardDiagnostics("layer_" + std::to_string(layer_idx) + "_ffn_gate",
                                 gate_out->data(),
                                 seq_len,
                                 d_ff);
        logFFNRowPreviewIfEnabled("layer_" + std::to_string(layer_idx) + "_ffn_gate", gate_out);
        capture_rows("ffn_gate", gate_out);

        // Up projection adaptive path
        {
            BackendContext bctx;
            bctx.is_prefill = is_prefill_stage_;
            bctx.seq_len = seq_len;
            bctx.d_model = config_.d_model;
            bctx.n_layers = config_.n_layers;
            bctx.world = getSize();
            auto dec = selectAttentionBackend(bctx);
            if (dec.use_cosma())
            {
                LOG_DEBUG("[backend] ffn_up layer=" << layer_idx << " backend=COSMA stage=" << (is_prefill_stage_ ? "prefill" : "decode") << " reason=" << dec.reason);
                if (!adaptiveMatMul(ffn_norm_out->data(),
                                    weights.w_up[layer_idx]->data(),
                                    up_out->data(),
                                    seq_len,
                                    d_ff,
                                    hidden_size,
                                    is_prefill_stage_,
                                    false,
                                    false,
                                    false,
                                    1.0f,
                                    0.0f))
                {
                    LOG_ERROR("Layer " << layer_idx << " up projection COSMA path failed");
                    return false;
                }
            }
            else
            {
                LOG_DEBUG("[backend] ffn_up layer=" << layer_idx << " backend=OpenBLAS(linear-kernel) stage=" << (is_prefill_stage_ ? "prefill" : "decode") << " reason=" << dec.reason);
                std::vector<std::shared_ptr<TensorBase>> up_inputs = {ffn_norm_out, weights.w_up[layer_idx]};
                std::vector<std::shared_ptr<TensorBase>> up_outputs = {up_out};
                if (!executeKernel("linear", up_inputs, up_outputs))
                {
                    LOG_ERROR("Layer " << layer_idx << " up projection failed");
                    return false;
                }
            }
        }

        ASSERT_TENSOR_NOT_NAN(up_out, "FFN up output has NaN in layer " + std::to_string(layer_idx));
        TensorLogger::logTensorStats(up_out, "layer_" + std::to_string(layer_idx) + "_ffn_up");
        baseline_snapshot("layer_" + std::to_string(layer_idx) + "_ffn_up",
                          up_out->data(),
                          static_cast<size_t>(seq_len) * d_ff,
                          d_ff,
                          1e-3);
        traceFFNShardDiagnostics("layer_" + std::to_string(layer_idx) + "_ffn_up",
                                 up_out->data(),
                                 seq_len,
                                 d_ff);
        logFFNRowPreviewIfEnabled("layer_" + std::to_string(layer_idx) + "_ffn_up", up_out);
        capture_rows("ffn_up", up_out);

        if (run_reference && getRank() == 0)
        {
            size_t elems = static_cast<size_t>(seq_len) * d_ff;
            gate_ref_storage.resize(elems);
            up_ref_storage.resize(elems);
            if (!ffn_norm_ref_storage.empty())
            {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            seq_len, d_ff, hidden_size,
                            1.0f,
                            ffn_norm_ref_storage.data(), hidden_size,
                            weights.w_gate[layer_idx]->data(), d_ff,
                            0.0f,
                            gate_ref_storage.data(), d_ff);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            seq_len, d_ff, hidden_size,
                            1.0f,
                            ffn_norm_ref_storage.data(), hidden_size,
                            weights.w_up[layer_idx]->data(), d_ff,
                            0.0f,
                            up_ref_storage.data(), d_ff);
                std::string gate_label = "layer_" + std::to_string(layer_idx) + "_ffn_gate";
                log_stage_diff(gate_label, gate_out->data(), gate_ref_storage.data(), elems, d_ff, 1e-3);
                std::string up_label = "layer_" + std::to_string(layer_idx) + "_ffn_up";
                log_stage_diff(up_label, up_out->data(), up_ref_storage.data(), elems, d_ff, 1e-3);
            }
        }

        auto linear_mid = std::chrono::high_resolution_clock::now();

        // SwiGLU activation using registered SwiGLU kernel
        auto activation_start = std::chrono::high_resolution_clock::now();
        auto swiglu_out = createLocalTensor({seq_len, config_.d_ff});

        std::vector<std::shared_ptr<TensorBase>> swiglu_inputs = {gate_out, up_out};
        std::vector<std::shared_ptr<TensorBase>> swiglu_outputs = {swiglu_out};

        if (!executeKernel("swiglu", swiglu_inputs, swiglu_outputs))
        {
            LOG_ERROR("Layer " << layer_idx << " SwiGLU activation failed");
            return false;
        }

        ASSERT_TENSOR_NOT_NAN(swiglu_out, "FFN SwiGLU output has NaN in layer " + std::to_string(layer_idx));
        TensorLogger::logTensorStats(swiglu_out, "layer_" + std::to_string(layer_idx) + "_ffn_swiglu");
        baseline_snapshot("layer_" + std::to_string(layer_idx) + "_ffn_swiglu",
                          swiglu_out->data(),
                          static_cast<size_t>(seq_len) * d_ff,
                          d_ff,
                          1e-3);
        traceFFNShardDiagnostics("layer_" + std::to_string(layer_idx) + "_ffn_swiglu",
                                 swiglu_out->data(),
                                 seq_len,
                                 d_ff);
        logFFNRowPreviewIfEnabled("layer_" + std::to_string(layer_idx) + "_ffn_swiglu", swiglu_out);
        capture_rows("ffn_swiglu", swiglu_out);

        auto activation_end = std::chrono::high_resolution_clock::now();
        total_activation_time_ += std::chrono::duration<double, std::milli>(activation_end - activation_start).count();

        if (run_reference && getRank() == 0)
        {
            size_t elems = static_cast<size_t>(seq_len) * d_ff;
            swiglu_ref_storage.resize(elems);
            for (size_t idx = 0; idx < elems; ++idx)
            {
                float up_val = up_ref_storage[idx];
                float silu_val;
                if (up_val > 20.0f)
                {
                    silu_val = up_val;
                }
                else if (up_val < -20.0f)
                {
                    silu_val = 0.0f;
                }
                else
                {
                    silu_val = up_val / (1.0f + std::exp(-up_val));
                }
                swiglu_ref_storage[idx] = gate_ref_storage[idx] * silu_val;
            }
            std::string swiglu_label = "layer_" + std::to_string(layer_idx) + "_ffn_swiglu";
            log_stage_diff(swiglu_label, swiglu_out->data(), swiglu_ref_storage.data(), elems, d_ff, 1e-3);
        }

        // Down projection adaptive path
        {
            BackendContext bctx;
            bctx.is_prefill = is_prefill_stage_;
            bctx.seq_len = seq_len;
            bctx.d_model = config_.d_model;
            bctx.n_layers = config_.n_layers;
            bctx.world = getSize();
            auto dec = selectAttentionBackend(bctx);
            if (dec.use_cosma())
            {
                LOG_DEBUG("[backend] ffn_down layer=" << layer_idx << " backend=COSMA stage=" << (is_prefill_stage_ ? "prefill" : "decode") << " reason=" << dec.reason);
                if (!adaptiveMatMul(swiglu_out->data(),
                                    weights.w_down[layer_idx]->data(),
                                    ffn_out->data(),
                                    seq_len,
                                    hidden_size, /* n (output) */
                                    d_ff,        /* k */
                                    is_prefill_stage_,
                                    false,
                                    false,
                                    false,
                                    1.0f,
                                    0.0f))
                {
                    LOG_ERROR("Layer " << layer_idx << " down projection COSMA path failed");
                    return false;
                }
            }
            else
            {
                LOG_DEBUG("[backend] ffn_down layer=" << layer_idx << " backend=OpenBLAS(linear-kernel) stage=" << (is_prefill_stage_ ? "prefill" : "decode") << " reason=" << dec.reason);
                std::vector<std::shared_ptr<TensorBase>> down_inputs = {swiglu_out, weights.w_down[layer_idx]};
                std::vector<std::shared_ptr<TensorBase>> down_outputs = {ffn_out};
                if (!executeKernel("linear", down_inputs, down_outputs))
                {
                    LOG_ERROR("Layer " << layer_idx << " down projection failed");
                    return false;
                }
            }
        }

        ASSERT_TENSOR_NOT_NAN(ffn_out, "FFN output has NaN in layer " + std::to_string(layer_idx));
        TensorLogger::logTensorStats(ffn_out, "layer_" + std::to_string(layer_idx) + "_ffn_out");
        baseline_snapshot("layer_" + std::to_string(layer_idx) + "_ffn_out",
                          ffn_out->data(),
                          static_cast<size_t>(seq_len) * hidden_size,
                          hidden_size,
                          1e-3);
        traceFFNShardDiagnostics("layer_" + std::to_string(layer_idx) + "_ffn_out",
                                 ffn_out->data(),
                                 seq_len,
                                 hidden_size);
        logFFNRowPreviewIfEnabled("layer_" + std::to_string(layer_idx) + "_ffn_out", ffn_out);
        capture_rows("ffn_out", ffn_out);

        if (run_reference && getRank() == 0)
        {
            size_t elems = static_cast<size_t>(seq_len) * hidden_size;
            ffn_down_ref_storage.resize(elems);
            if (!swiglu_ref_storage.empty())
            {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            seq_len, hidden_size, d_ff,
                            1.0f,
                            swiglu_ref_storage.data(), d_ff,
                            weights.w_down[layer_idx]->data(), hidden_size,
                            0.0f,
                            ffn_down_ref_storage.data(), hidden_size);
                std::string down_label = "layer_" + std::to_string(layer_idx) + "_ffn_down";
                log_stage_diff(down_label, ffn_out->data(), ffn_down_ref_storage.data(), elems, hidden_size, 1e-3);
            }
        }

        auto linear_end = std::chrono::high_resolution_clock::now();
        total_linear_time_ += std::chrono::duration<double, std::milli>(linear_end - linear_start).count();

        // Final residual connection using registered residual kernel
        std::vector<std::shared_ptr<TensorBase>> final_residual_inputs = {residual_tmp, ffn_out};
        std::vector<std::shared_ptr<TensorBase>> final_residual_outputs = {output};

        if (!executeKernel("residual", final_residual_inputs, final_residual_outputs))
        {
            LOG_ERROR("Layer " << layer_idx << " FFN residual failed");
            return false;
        }

        ASSERT_TENSOR_NOT_NAN(output, "Layer output has NaN in layer " + std::to_string(layer_idx));
        TensorLogger::logTensorStats(output, "layer_" + std::to_string(layer_idx) + "_output");
        capture_rows("layer_output", output);
        baseline_snapshot("layer_" + std::to_string(layer_idx) + "_output",
                          output->data(),
                          static_cast<size_t>(seq_len) * hidden_size,
                          hidden_size,
                          1e-6);

        if (run_reference && getRank() == 0 && !residual_ref_storage.empty() && !ffn_down_ref_storage.empty())
        {
            size_t elems = static_cast<size_t>(seq_len) * hidden_size;
            final_ref_storage.resize(elems);
            for (size_t idx = 0; idx < elems; ++idx)
            {
                float residual = residual_ref_storage[idx];
                float ffn_val = ffn_down_ref_storage[idx];
                final_ref_storage[idx] = residual + ffn_val;
            }
            std::string final_label = "layer_" + std::to_string(layer_idx) + "_output_residual";
            log_stage_diff(final_label, output->data(), final_ref_storage.data(), elems, hidden_size, 1e-6);
        }

        LOG_DEBUG("Layer " << layer_idx << " completed on rank " << getRank());
        // Layerwise activation stats (post-layer output) if enabled
        if (debugEnv().pipeline.layerwise_stats && getRank() == 0 && output && output->data())
        {
            int actual_seq = output->shape().size() > 0 ? output->shape()[0] : 0;
            int hidden = output->shape().size() > 1 ? output->shape()[1] : 0;
            size_t total = (size_t)actual_seq * hidden;
            double sum = 0, sumsq = 0, max_abs = 0;
            const float *p = output->data();
            for (size_t i = 0; i < total; ++i)
            {
                double v = p[i];
                sum += v;
                sumsq += v * v;
                double a = fabs(v);
                if (a > max_abs)
                    max_abs = a;
            }
            double mean = total ? sum / total : 0.0;
            double rms = total ? std::sqrt(sumsq / total) : 0.0;
            LayerActivationStat stat{rms, max_abs, mean, layer_idx};
            last_layer_stats_.push_back(stat);
        }
        // Capture last token row for diff diagnostics (only rank 0 to reduce memory)
        // Diagnostics: one-time snapshot of flag state to understand missing capture
        {
            static bool printed_flag_diag = false;
            if (!printed_flag_diag && getRank() == 0)
            {
                printed_flag_diag = true;
                const char *raw_env = std::getenv("LLAMINAR_PIPELINE_LAYER_TOKEN_DIFF");
                const char *raw_env_verbose = std::getenv("LLAMINAR_PIPELINE_LAYER_TOKEN_DIFF_VERBOSE");
                LOG_INFO(std::string("[LayerTokenDiffDiag] snapshot.layer_token_diff=") + (debugEnv().pipeline.layer_token_diff ? "1" : "0") +
                         " snapshot.layer_token_diff_verbose=" + (debugEnv().pipeline.layer_token_diff_verbose ? "1" : "0") +
                         " raw_env=" + (raw_env ? raw_env : "<unset>") +
                         " raw_env_verbose=" + (raw_env_verbose ? raw_env_verbose : "<unset>") +
                         " n_rows_current=" + std::to_string(last_layer_token_rows_.size()));
            }
        }
        if (debugEnv().pipeline.layer_token_diff && getRank() == 0 && output && output->data())
        {
            int rows = output->shape().size() > 0 ? output->shape()[0] : 0;
            int hidden = output->shape().size() > 1 ? output->shape()[1] : 0;
            if (rows > 0 && hidden > 0)
            {
                LayerTokenDiffRow row;
                row.layer = layer_idx;
                row.seq_len = rows;
                row.incremental = in_incremental_pass_;
                row.pipeline = this;
                row.values.assign(output->data() + (size_t)(rows - 1) * hidden, output->data() + (size_t)rows * hidden);
                last_layer_token_rows_.push_back(std::move(row));
                if (debugEnv().pipeline.layer_token_diff_verbose)
                {
                    LOG_INFO("[LayerTokenCapture] pipe=" << this << " incr=" << (in_incremental_pass_ ? 1 : 0) << " layer=" << layer_idx << " seq_len=" << rows << " hidden=" << hidden << " total_rows=" << last_layer_token_rows_.size());
                }
            }
        }
        return true;
    }

    bool DistributedTransformerPipeline::shouldUseCosmaPrefill(int seq_len) const
    {
        // DEPRECATED: Retained temporarily for any external callers; internal logic now
        // uses BackendSelector with explicit stage context. Mirrors former heuristic.
        if (seq_len <= 0)
            return false;
        if (n_past_ != 0)
            return false;
        if (getSize() <= 1)
            return false;
        auto &prefill_mgr = CosmaPrefillManager::instance();
        return prefill_mgr.enabled_for(seq_len);
    }

    std::shared_ptr<TensorBase> DistributedTransformerPipeline::embedSingleToken(int token_id,
                                                                                 const std::shared_ptr<TensorBase> &embedding_weight)
    {
        if (!embedding_weight || embedding_weight->shape().size() != 2)
        {
            LOG_ERROR("embedSingleToken: invalid embedding weight");
            return nullptr;
        }
        int vocab = embedding_weight->shape()[0];
        int dim = embedding_weight->shape()[1];
        if (token_id < 0 || token_id >= vocab)
        {
            LOG_ERROR("embedSingleToken: token id out of range: " << token_id << "/" << vocab);
            return nullptr;
        }
        auto out = createLocalTensor({1, dim});
        const float *src = embedding_weight->data() + (size_t)token_id * dim;
        float *dst = out->data();
        std::memcpy(dst, src, sizeof(float) * dim);
        if (debugEnv().embedding.trace && getRank() == 0)
        {
            int show = std::min(dim, debugEnv().embedding.trace_dims);
            std::ostringstream oss;
            oss << "[EmbedTrace] token=" << token_id << " dims=" << dim << " preview=";
            for (int i = 0; i < show; ++i)
            {
                oss << dst[i];
                if (i + 1 < show)
                    oss << ",";
            }
            LOG_INFO(oss.str());
        }
        return out;
    }

    bool DistributedTransformerPipeline::decodeToken(int token_id,
                                                     const ModelWeights &weights,
                                                     std::shared_ptr<TensorBase> &output_logits)
    {
        const auto &env = debugEnv();
        if (env.pipeline.disable_incremental_decode)
        {
            if (getRank() == 0)
                LOG_DEBUG("[decode] incremental disabled via env; fallback to replay");
            return false; // signal caller to fallback
        }
        if (!use_kv_cache_)
        {
            if (getRank() == 0)
                LOG_DEBUG("[decode] KV cache disabled; fallback to replay");
            return false;
        }
        // Basic KV cache readiness guard (capacity check) — if caches were never initialized or too small, fallback
        if (k_cache_.empty() || v_cache_.empty())
        {
            if (getRank() == 0)
                LOG_DEBUG("[decode] KV cache not initialized; fallback to replay");
            return false;
        }
        if ((int)k_cache_.size() != config_.n_layers || (int)v_cache_.size() != config_.n_layers)
        {
            LOG_WARN("[decode] KV cache layer count mismatch (" << k_cache_.size() << "/" << v_cache_.size() << " vs " << config_.n_layers << ") fallback to replay");
            return false;
        }
        if (!weights.token_embedding)
        {
            LOG_ERROR("decodeToken: missing token embedding");
            return false;
        }
        // Embed single token (1 x d_model)
        auto current = embedSingleToken(token_id, weights.token_embedding);
        if (!current)
            return false;

        if (env.pipeline.incr_trace && getRank() == 0)
        {
            LOG_INFO("[IncrTrace] start token=" << token_id
                                                << " n_past=" << n_past_
                                                << " use_kv=" << (use_kv_cache_ ? 1 : 0)
                                                << " cache_init=" << (!k_cache_.empty() ? 1 : 0));
        }

        // Capacity check / growth before populating new K/V slice
        if (use_kv_cache_)
        {
            int required = n_past_ + 1; // token to be written at index n_past_
            if (!ensureKVCapacity(required))
            {
                if (getRank() == 0)
                    LOG_WARN("[decode] ensureKVCapacity failed (required=" << required << ") fallback to replay");
                return false;
            }
        }

        // Stage & sequence context
        setStageDecode();
        // n_past_ holds number of previously processed tokens (prefill length + prior decodes)
        int position = n_past_;
        setSequencePosition(position);

        // Explicit minimal intermediate allocation for seq_len=1 incremental path.
        // We allocate only what's actually required by executeTransformerLayer + output projection.
        const int seq_len = 1;
        // Current tensor already represents the layer input; we'll create a reusable output buffer.
        auto layer_output = createLocalTensor({seq_len, config_.d_model});
        if (!layer_output)
        {
            LOG_ERROR("decodeToken: failed to allocate layer_output tensor");
            return false;
        }
        // Final logits buffer (1 x vocab)
        auto logits = createLocalTensor({seq_len, config_.vocab_size});
        if (!logits)
        {
            LOG_ERROR("decodeToken: failed to allocate logits tensor");
            return false;
        }

        for (int layer = 0; layer < config_.n_layers; ++layer)
        {
            if (!executeTransformerLayer(layer, current, weights, layer_output))
            {
                LOG_ERROR("decodeToken: layer " << layer << " failed");
                return false;
            }
            current = layer_output; // reuse buffer across layers

            if (env.pipeline.incr_cache_trace && use_kv_cache_ && getRank() == 0)
            {
                // Log brief K/V slice preview for this layer at the just-written position
                int pos = n_past_;
                if (layer < (int)k_cache_.size() && k_cache_[layer] && v_cache_[layer])
                {
                    auto &K = k_cache_[layer];
                    auto &V = v_cache_[layer];
                    int kv_heads = config_.n_head_kv;
                    int head_dim = config_.head_dim;
                    size_t stride = (size_t)config_.max_seq_len * head_dim; // layout assumption (layer-major, head-contiguous) – adjust if different
                    if (pos < config_.max_seq_len)
                    {
                        // Sample first 4 floats of first KV head at this position
                        size_t base = (size_t)pos * head_dim; // simplified: single-head contiguous assumption
                        float k_preview[4] = {0}, v_preview[4] = {0};
                        for (int i = 0; i < 4 && i < head_dim; ++i)
                        {
                            k_preview[i] = K->data()[base + i];
                            v_preview[i] = V->data()[base + i];
                        }
                        LOG_INFO("[IncrCacheTrace] layer=" << layer << " pos=" << pos
                                                           << " k0=[" << k_preview[0] << "," << k_preview[1] << "," << k_preview[2] << "," << k_preview[3]
                                                           << "] v0=[" << v_preview[0] << "," << v_preview[1] << "," << v_preview[2] << "," << v_preview[3] << "]");
                    }
                }
            }
        }

        // Final output projection (norm + lm head) into logits
        if (!executeOutputProjection(current, weights, logits))
        {
            LOG_ERROR("decodeToken: output projection failed");
            return false;
        }

        if (env.pipeline.incr_hidden_trace && getRank() == 0)
        {
            // Dump last hidden row prior to LM head for post-mortem comparison
            int d = config_.d_model;
            int dump = std::min(d, 16); // first 16 dims preview
            std::ostringstream oss;
            oss << "[IncrHiddenTrace] pos=" << n_past_ << " dims=" << d << " preview=";
            const float *row = current->data(); // seq_len=1
            for (int i = 0; i < dump; ++i)
            {
                oss << row[i];
                if (i + 1 < dump)
                    oss << ',';
            }
            LOG_INFO(oss.str());
        }

        output_logits = logits;
        n_past_ += 1; // advance position
        if (use_kv_cache_)
        {
            kv_cache_state_.used_tokens = n_past_;
        }
        if (getRank() == 0)
        {
            if (env.pipeline.incr_trace)
            {
                LOG_INFO("[IncrTrace] done token=" << token_id << " pos=" << position << " n_past(new)=" << n_past_ << " logits_shape=1x" << config_.vocab_size);
            }
            else
            {
                LOG_DEBUG("[decode] incremental token=" << token_id << " pos=" << position);
            }
        }
        return true;
    }

    bool DistributedTransformerPipeline::incrementalDecodeToken(int token_id,
                                                                const ModelWeights &weights,
                                                                std::shared_ptr<TensorBase> &output_logits)
    {
        // Tag diagnostic context so layer capture knows this is incremental
        struct Scope
        {
            DistributedTransformerPipeline *p;
            ~Scope()
            {
                if (p)
                    p->in_incremental_pass_ = false;
            }
        };
        in_incremental_pass_ = true;
        Scope scope{this};
        return decodeToken(token_id, weights, output_logits);
    }

    bool DistributedTransformerPipeline::executePrefillAttentionCosma(int layer_idx,
                                                                      std::shared_ptr<TensorBase> &input,
                                                                      const ModelWeights &weights,
                                                                      std::shared_ptr<TensorBase> &attn_norm_out,
                                                                      std::shared_ptr<TensorBase> &attn_out,
                                                                      PrefillAttentionTiming &timing)
    {
        auto fused_start = std::chrono::high_resolution_clock::now();
        CosmaPrefillManager &manager = CosmaPrefillManager::instance();
        const int seq_len = input->shape()[0];
        const int hidden_size = config_.d_model;
        const int head_dim = config_.head_dim;
        const int n_heads = config_.n_head;
        const int n_kv_heads = config_.n_head_kv;
        const int total_head_dim = n_heads * head_dim;
        const int kv_head_dim = n_kv_heads * head_dim;

        auto make_desc = [&](const std::shared_ptr<TensorBase> &tensor, const std::string &id) -> WeightDescriptor
        {
            const auto &shape = tensor->shape();
            WeightDescriptor desc{id,
                                  shape.size() > 0 ? shape[0] : 0,
                                  shape.size() > 1 ? shape[1] : 0,
                                  static_cast<int64_t>(shape.size() > 1 ? shape[1] : 0),
                                  1,
                                  0,
                                  tensor->data(),
                                  0};
            return desc;
        };

        WeightDescriptor wq_desc = make_desc(weights.wq[layer_idx], "layer" + std::to_string(layer_idx) + "_wq");
        WeightDescriptor wk_desc = make_desc(weights.wk[layer_idx], "layer" + std::to_string(layer_idx) + "_wk");
        WeightDescriptor wv_desc = make_desc(weights.wv[layer_idx], "layer" + std::to_string(layer_idx) + "_wv");

        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        auto fused = manager.fused_rmsnorm_qkv(input->data(),
                                               weights.attn_norm_weight[layer_idx]->data(),
                                               wq_desc,
                                               wk_desc,
                                               wv_desc,
                                               seq_len,
                                               hidden_size,
                                               config_.eps,
                                               scale,
                                               false);
        auto fused_end = std::chrono::high_resolution_clock::now();

        if ((!fused.normalized.mat && !fused.normalized.host_owned) ||
            (!fused.q.mat && !fused.q.host_owned) ||
            (!fused.k.mat && !fused.k.host_owned) ||
            (!fused.v.mat && !fused.v.host_owned))
        {
            LOG_ERROR("CosmaPrefillManager returned incomplete fused result for layer " << layer_idx
                                                                                        << " normalized=(mat=" << static_cast<bool>(fused.normalized.mat)
                                                                                        << ", host=" << static_cast<bool>(fused.normalized.host_owned)
                                                                                        << ") q=(mat=" << static_cast<bool>(fused.q.mat)
                                                                                        << ", host=" << static_cast<bool>(fused.q.host_owned)
                                                                                        << ") k=(mat=" << static_cast<bool>(fused.k.mat)
                                                                                        << ", host=" << static_cast<bool>(fused.k.host_owned)
                                                                                        << ") v=(mat=" << static_cast<bool>(fused.v.mat)
                                                                                        << ", host=" << static_cast<bool>(fused.v.host_owned)
                                                                                        << ")");
            return false;
        }

        std::vector<float> norm_buf(static_cast<size_t>(seq_len) * hidden_size, 0.f);
        manager.to_row_major(fused.normalized, norm_buf.data());
        std::memcpy(attn_norm_out->data(), norm_buf.data(), norm_buf.size() * sizeof(float));

        std::vector<float> q_buf(static_cast<size_t>(seq_len) * total_head_dim, 0.f);
        std::vector<float> k_buf(static_cast<size_t>(seq_len) * kv_head_dim, 0.f);
        std::vector<float> v_buf(static_cast<size_t>(seq_len) * kv_head_dim, 0.f);
        manager.to_row_major(fused.q, q_buf.data(), true);
        manager.to_row_major(fused.k, k_buf.data(), true);
        manager.to_row_major(fused.v, v_buf.data(), true);

        timing.norm_ms = std::chrono::duration<double, std::milli>(fused_end - fused_start).count();

        std::vector<float> context_concat(static_cast<size_t>(seq_len) * total_head_dim, 0.f);
        std::vector<float> q_head(static_cast<size_t>(seq_len) * head_dim, 0.f);
        std::vector<float> k_head(static_cast<size_t>(seq_len) * head_dim, 0.f);
        std::vector<float> v_head(static_cast<size_t>(seq_len) * head_dim, 0.f);
        std::vector<float> scores(static_cast<size_t>(seq_len) * seq_len, 0.f);
        std::vector<float> context_head(static_cast<size_t>(seq_len) * head_dim, 0.f);
        const auto &pd_exec = debugEnv().prefill_debug;
        const auto &abl_exec = debugEnv().ablation;
        const bool trace_io = pd_exec.trace_io;
        const bool debug_compare = pd_exec.debug_compare;
        const bool debug_attention = pd_exec.debug_attention;
        const bool debug_output = pd_exec.debug_output;
        const bool run_reference = !(abl_exec.ablate_attention || abl_exec.ablate_ffn) && (trace_io || debug_compare || debug_attention || debug_output);
        const char *diag_tag = "[PrefillDebug]";
        if (trace_io)
        {
            diag_tag = "[PrefillDiag]";
        }
        else if (debug_attention)
        {
            diag_tag = "[PrefillAttention]";
        }
        else if (debug_output)
        {
            diag_tag = "[PrefillOutput]";
        }
        else if (debug_compare)
        {
            diag_tag = "[PrefillCompare]";
        }

        auto log_buffer = [&](const std::string &name, const float *data, size_t size, int rows, int cols)
        {
            if (!trace_io || rank_ != 0 || !data || size == 0)
            {
                return;
            }
            auto stats = computeBufferStats(data, size);
            LOG_INFO("[PrefillDiag] " << name << " stats rows=" << rows
                                      << " cols=" << cols
                                      << " min=" << stats.min
                                      << " max=" << stats.max
                                      << " mean=" << stats.mean
                                      << " stddev=" << stats.stddev
                                      << " rms=" << stats.rms
                                      << " l2=" << stats.l2);
            size_t bad_idx = 0;
            float bad_val = 0.f;
            if (findFirstNonFinite(data, size, bad_idx, bad_val))
            {
                if (cols > 0)
                {
                    int row = static_cast<int>(bad_idx / static_cast<size_t>(cols));
                    int col = static_cast<int>(bad_idx % static_cast<size_t>(cols));
                    LOG_WARN("[PrefillDiag] " << name << " non-finite value at idx=" << bad_idx
                                              << " (row=" << row << " col=" << col << ") value=" << bad_val);
                }
                else
                {
                    LOG_WARN("[PrefillDiag] " << name << " non-finite value at idx=" << bad_idx
                                              << " value=" << bad_val);
                }
            }
        };

        auto log_diff_summary = [&](const std::string &name,
                                    const std::vector<float> &buf,
                                    const std::vector<float> &ref,
                                    int cols,
                                    double warn_threshold)
        {
            if (!run_reference || rank_ != 0 || buf.size() != ref.size() || buf.empty())
            {
                return;
            }
            auto diff = computeDiffSummary(buf.data(), ref.data(), ref.size());
            auto samples = collectTopDiffSamples(buf.data(), ref.data(), ref.size(), 5);
            if (diff.max_abs > warn_threshold)
            {
                LOG_WARN(diag_tag << " " << name << " diff rel_l2=" << diff.rel_l2
                                  << " max_abs=" << diff.max_abs
                                  << " mean_abs=" << diff.mean_abs
                                  << " worst_index=" << diff.worst_index
                                  << " cosma=" << diff.value_a
                                  << " ref=" << diff.value_b);
            }
            else
            {
                LOG_INFO(diag_tag << " " << name << " diff rel_l2=" << diff.rel_l2
                                  << " max_abs=" << diff.max_abs
                                  << " mean_abs=" << diff.mean_abs
                                  << " worst_index=" << diff.worst_index
                                  << " cosma=" << diff.value_a
                                  << " ref=" << diff.value_b);
            }
            LOG_INFO(diag_tag << " " << name << " diff samples: " << formatDiffSamples(samples, cols));
        };

        const float *layer_input_ptr = input ? input->data() : nullptr;
        // Optional capture of raw layer input (pre-attention RMSNorm) when activation capture enabled
        int layer_input_seq_len = 0;
        int layer_input_hidden = 0;
        if (input)
        {
            auto ish = input->shape();
            if (ish.size() == 2)
            {
                layer_input_seq_len = static_cast<int>(ish[0]);
                layer_input_hidden = static_cast<int>(ish[1]);
            }
        }
        if (layer_idx == 0 && layer_input_ptr && debugEnv().layer_capture.capture && getRank() == 0)
        {
            try
            {
                std::filesystem::create_directories("layer_activations");
                if (layer_input_seq_len > 0 && layer_input_hidden > 0)
                {
                    int seq_len = layer_input_seq_len;
                    int hidden_size = layer_input_hidden;
                    const auto &tokens = debugEnv().layer_capture.tokens;
                    std::vector<int> rows = tokens.empty() ? std::vector<int>{0} : tokens;
                    for (int r : rows)
                    {
                        if (r < 0 || r >= seq_len)
                            continue;
                        const float *row = layer_input_ptr + (size_t)r * hidden_size;
                        std::ostringstream fn;
                        fn << "layer_activations/embedding_out_row" << r << ".txt";
                        std::ofstream ofs(fn.str(), std::ios::out | std::ios::trunc);
                        if (!ofs.good())
                            continue;
                        for (int c = 0; c < hidden_size; ++c)
                        {
                            ofs << row[c];
                            if (c + 1 < hidden_size)
                                ofs << ' ';
                        }
                        ofs << '\n';
                    }
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN("Embedding activation capture failed: " << e.what());
            }
        }
        size_t norm_elems = 0;
        if (layer_input_seq_len > 0 && layer_input_hidden > 0)
            norm_elems = static_cast<size_t>(layer_input_seq_len) * layer_input_hidden;
        if (trace_io && layer_input_ptr)
        {
            // At this point full seq_len/hidden_size may not yet be assigned; use layer_input_* captured above
            log_buffer("layer_input", layer_input_ptr, norm_elems, layer_input_seq_len, layer_input_hidden);
        }
        if (trace_io)
        {
            // norm_buf and q/k/v buffers rely on seq_len/hidden_size which are defined later; defer if zero
            if (seq_len > 0 && hidden_size > 0)
            {
                log_buffer("rmsnorm_output", norm_buf.data(), norm_buf.size(), seq_len, hidden_size);
            }
            if (seq_len > 0)
            {
                log_buffer("Q_buffer", q_buf.data(), q_buf.size(), seq_len, total_head_dim);
                log_buffer("K_buffer", k_buf.data(), k_buf.size(), seq_len, kv_head_dim);
                log_buffer("V_buffer", v_buf.data(), v_buf.size(), seq_len, kv_head_dim);
            }
        }

        std::vector<float> norm_ref;
        if (run_reference && layer_input_ptr)
        {
            norm_ref.assign(norm_buf.size(), 0.f);
            const float *gamma_ptr = weights.attn_norm_weight[layer_idx]->data();
            for (int r = 0; r < seq_len; ++r)
            {
                const float *row = layer_input_ptr + static_cast<size_t>(r) * hidden_size;
                double sum = 0.0;
                for (int c = 0; c < hidden_size; ++c)
                {
                    double v = static_cast<double>(row[c]);
                    sum += v * v;
                }
                double inv = 1.0 / std::sqrt(sum / std::max(1, hidden_size) + config_.eps);
                float *dst_row = norm_ref.data() + static_cast<size_t>(r) * hidden_size;
                for (int c = 0; c < hidden_size; ++c)
                {
                    dst_row[c] = static_cast<float>(row[c] * inv * gamma_ptr[c]);
                }
            }
            log_diff_summary("RMSNorm", norm_buf, norm_ref, hidden_size, 1e-5);
        }

        std::vector<float> q_ref;
        std::vector<float> k_ref;
        std::vector<float> v_ref;
        if (run_reference)
        {
            q_ref.assign(static_cast<size_t>(seq_len) * total_head_dim, 0.f);
            k_ref.assign(static_cast<size_t>(seq_len) * kv_head_dim, 0.f);
            v_ref.assign(static_cast<size_t>(seq_len) * kv_head_dim, 0.f);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        seq_len, total_head_dim, hidden_size,
                        1.0f,
                        norm_buf.data(), hidden_size,
                        weights.wq[layer_idx]->data(), total_head_dim,
                        0.0f,
                        q_ref.data(), total_head_dim);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        seq_len, kv_head_dim, hidden_size,
                        1.0f,
                        norm_buf.data(), hidden_size,
                        weights.wk[layer_idx]->data(), kv_head_dim,
                        0.0f,
                        k_ref.data(), kv_head_dim);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        seq_len, kv_head_dim, hidden_size,
                        1.0f,
                        norm_buf.data(), hidden_size,
                        weights.wv[layer_idx]->data(), kv_head_dim,
                        0.0f,
                        v_ref.data(), kv_head_dim);
            log_diff_summary("Q", q_buf, q_ref, total_head_dim, 1e-3);
            log_diff_summary("K", k_buf, k_ref, kv_head_dim, 1e-3);
            log_diff_summary("V", v_buf, v_ref, kv_head_dim, 1e-3);
        }

        constexpr float theta_base = 10000.0f;

        std::vector<float> context_ref_concat;
        std::vector<float> q_head_ref;
        std::vector<float> k_head_ref;
        std::vector<float> v_head_ref;
        std::vector<float> context_head_ref;
        std::vector<float> scores_head_ref;
        if (run_reference)
        {
            context_ref_concat.assign(static_cast<size_t>(seq_len) * total_head_dim, 0.f);
            q_head_ref.resize(static_cast<size_t>(seq_len) * head_dim);
            k_head_ref.resize(static_cast<size_t>(seq_len) * head_dim);
            v_head_ref.resize(static_cast<size_t>(seq_len) * head_dim);
            context_head_ref.resize(static_cast<size_t>(seq_len) * head_dim);
            scores_head_ref.resize(static_cast<size_t>(seq_len) * seq_len);
        }

        auto attention_start = std::chrono::high_resolution_clock::now();
        for (int head = 0; head < n_heads; ++head)
        {
            const int kv_head = head % n_kv_heads;
            for (int row = 0; row < seq_len; ++row)
            {
                const float *q_src = q_buf.data() + static_cast<size_t>(row) * total_head_dim + head * head_dim;
                const float *k_src = k_buf.data() + static_cast<size_t>(row) * kv_head_dim + kv_head * head_dim;
                const float *v_src = v_buf.data() + static_cast<size_t>(row) * kv_head_dim + kv_head * head_dim;
                std::memcpy(q_head.data() + static_cast<size_t>(row) * head_dim, q_src, head_dim * sizeof(float));
                std::memcpy(k_head.data() + static_cast<size_t>(row) * head_dim, k_src, head_dim * sizeof(float));
                std::memcpy(v_head.data() + static_cast<size_t>(row) * head_dim, v_src, head_dim * sizeof(float));
                if (run_reference)
                {
                    const float *q_ref_src = q_ref.data() + static_cast<size_t>(row) * total_head_dim + head * head_dim;
                    const float *k_ref_src = k_ref.data() + static_cast<size_t>(row) * kv_head_dim + kv_head * head_dim;
                    const float *v_ref_src = v_ref.data() + static_cast<size_t>(row) * kv_head_dim + kv_head * head_dim;
                    std::memcpy(q_head_ref.data() + static_cast<size_t>(row) * head_dim, q_ref_src, head_dim * sizeof(float));
                    std::memcpy(k_head_ref.data() + static_cast<size_t>(row) * head_dim, k_ref_src, head_dim * sizeof(float));
                    std::memcpy(v_head_ref.data() + static_cast<size_t>(row) * head_dim, v_ref_src, head_dim * sizeof(float));
                }
            }

            for (int row = 0; row < seq_len; ++row)
            {
                float *q_row = q_head.data() + static_cast<size_t>(row) * head_dim;
                float *k_row = k_head.data() + static_cast<size_t>(row) * head_dim;
                float *q_ref_row = run_reference ? (q_head_ref.data() + static_cast<size_t>(row) * head_dim) : nullptr;
                float *k_ref_row = run_reference ? (k_head_ref.data() + static_cast<size_t>(row) * head_dim) : nullptr;
                const float position = static_cast<float>(n_past_ + row);
                for (int dim_pair = 0; dim_pair < head_dim / 2; ++dim_pair)
                {
                    float theta = 1.0f / std::pow(theta_base, (2.0f * static_cast<float>(dim_pair)) / static_cast<float>(head_dim));
                    float cos_theta = std::cos(position * theta);
                    float sin_theta = std::sin(position * theta);

                    float q0 = q_row[2 * dim_pair];
                    float q1 = q_row[2 * dim_pair + 1];
                    q_row[2 * dim_pair] = q0 * cos_theta - q1 * sin_theta;
                    q_row[2 * dim_pair + 1] = q0 * sin_theta + q1 * cos_theta;

                    float k0 = k_row[2 * dim_pair];
                    float k1 = k_row[2 * dim_pair + 1];
                    k_row[2 * dim_pair] = k0 * cos_theta - k1 * sin_theta;
                    k_row[2 * dim_pair + 1] = k0 * sin_theta + k1 * cos_theta;

                    if (run_reference)
                    {
                        float rk0 = k_ref_row[2 * dim_pair];
                        float rk1 = k_ref_row[2 * dim_pair + 1];
                        k_ref_row[2 * dim_pair] = rk0 * cos_theta - rk1 * sin_theta;
                        k_ref_row[2 * dim_pair + 1] = rk0 * sin_theta + rk1 * cos_theta;

                        float rq0 = q_ref_row[2 * dim_pair];
                        float rq1 = q_ref_row[2 * dim_pair + 1];
                        q_ref_row[2 * dim_pair] = rq0 * cos_theta - rq1 * sin_theta;
                        q_ref_row[2 * dim_pair + 1] = rq0 * sin_theta + rq1 * cos_theta;
                    }
                }
            }

            if (run_reference)
            {
                std::fill(scores_head_ref.begin(), scores_head_ref.end(), 0.0f);
                std::fill(context_head_ref.begin(), context_head_ref.end(), 0.0f);
            }

            for (int row = 0; row < seq_len; ++row)
            {
                const float *q_row = q_head.data() + static_cast<size_t>(row) * head_dim;
                float *score_row = scores.data() + static_cast<size_t>(row) * seq_len;
                const float *q_ref_row = run_reference ? (q_head_ref.data() + static_cast<size_t>(row) * head_dim) : nullptr;
                float *score_ref_row = run_reference ? (scores_head_ref.data() + static_cast<size_t>(row) * seq_len) : nullptr;

                float row_max = -std::numeric_limits<float>::infinity();
                float row_max_ref = -std::numeric_limits<float>::infinity();
                for (int col = 0; col <= row; ++col)
                {
                    const float *k_row = k_head.data() + static_cast<size_t>(col) * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        dot += q_row[d] * k_row[d];
                    }
                    float scaled = dot * scale;
                    score_row[col] = scaled;
                    if (scaled > row_max)
                    {
                        row_max = scaled;
                    }

                    if (run_reference)
                    {
                        const float *k_ref_row = k_head_ref.data() + static_cast<size_t>(col) * head_dim;
                        float dot_ref = 0.0f;
                        for (int d = 0; d < head_dim; ++d)
                        {
                            dot_ref += q_ref_row[d] * k_ref_row[d];
                        }
                        float scaled_ref = dot_ref * scale;
                        score_ref_row[col] = scaled_ref;
                        if (scaled_ref > row_max_ref)
                        {
                            row_max_ref = scaled_ref;
                        }
                    }
                }
                for (int col = row + 1; col < seq_len; ++col)
                {
                    score_row[col] = -std::numeric_limits<float>::infinity();
                    if (run_reference)
                    {
                        score_ref_row[col] = -std::numeric_limits<float>::infinity();
                    }
                }

                double denom = 0.0;
                double denom_ref = 0.0;
                for (int col = 0; col <= row; ++col)
                {
                    float val = std::exp(score_row[col] - row_max);
                    score_row[col] = val;
                    denom += val;

                    if (run_reference)
                    {
                        float ref_val = std::exp(score_ref_row[col] - row_max_ref);
                        score_ref_row[col] = ref_val;
                        denom_ref += ref_val;
                    }
                }
                float inv = denom > 0.0 ? static_cast<float>(1.0 / denom) : 1.0f;
                for (int col = 0; col <= row; ++col)
                {
                    score_row[col] *= inv;
                }
                for (int col = row + 1; col < seq_len; ++col)
                {
                    score_row[col] = 0.0f;
                }

                if (run_reference)
                {
                    float inv_ref = denom_ref > 0.0 ? static_cast<float>(1.0 / denom_ref) : 1.0f;
                    for (int col = 0; col <= row; ++col)
                    {
                        score_ref_row[col] *= inv_ref;
                    }
                    for (int col = row + 1; col < seq_len; ++col)
                    {
                        score_ref_row[col] = 0.0f;
                    }
                }

                float *context_row = context_head.data() + static_cast<size_t>(row) * head_dim;
                std::fill(context_row, context_row + head_dim, 0.0f);
                float *context_ref_row = run_reference ? (context_head_ref.data() + static_cast<size_t>(row) * head_dim) : nullptr;
                if (run_reference)
                {
                    std::fill(context_ref_row, context_ref_row + head_dim, 0.0f);
                }
                for (int col = 0; col <= row; ++col)
                {
                    const float *v_row = v_head.data() + static_cast<size_t>(col) * head_dim;
                    float weight = score_row[col];
                    for (int d = 0; d < head_dim; ++d)
                    {
                        context_row[d] += weight * v_row[d];
                    }

                    if (run_reference)
                    {
                        const float *v_ref_row = v_head_ref.data() + static_cast<size_t>(col) * head_dim;
                        float weight_ref = score_ref_row[col];
                        for (int d = 0; d < head_dim; ++d)
                        {
                            context_ref_row[d] += weight_ref * v_ref_row[d];
                        }
                    }
                }
            }

            for (int row = 0; row < seq_len; ++row)
            {
                float *dst = context_concat.data() + static_cast<size_t>(row) * total_head_dim + head * head_dim;
                const float *src = context_head.data() + static_cast<size_t>(row) * head_dim;
                std::memcpy(dst, src, head_dim * sizeof(float));
                if (run_reference)
                {
                    float *dst_ref = context_ref_concat.data() + static_cast<size_t>(row) * total_head_dim + head * head_dim;
                    const float *src_ref = context_head_ref.data() + static_cast<size_t>(row) * head_dim;
                    std::memcpy(dst_ref, src_ref, head_dim * sizeof(float));
                }
            }

            if (run_reference && rank_ == 0)
            {
                std::string head_context_label = "attention_context_head_" + std::to_string(head);
                log_diff_summary(head_context_label, context_head, context_head_ref, head_dim, 1e-3);

                std::string head_scores_label = "attention_scores_head_" + std::to_string(head);
                log_diff_summary(head_scores_label, scores, scores_head_ref, seq_len, 1e-3);
            }
        }

        auto attention_end = std::chrono::high_resolution_clock::now();
        if (trace_io)
        {
            log_buffer("attention_scores", scores.data(), scores.size(), seq_len, seq_len);
            log_buffer("context_concat", context_concat.data(), context_concat.size(), seq_len, total_head_dim);
            if (run_reference)
            {
                log_buffer("attention_scores_ref", scores_head_ref.data(), scores_head_ref.size(), seq_len, seq_len);
                log_buffer("context_concat_ref", context_ref_concat.data(), context_ref_concat.size(), seq_len, total_head_dim);
            }
        }
        if (run_reference && rank_ == 0)
        {
            log_diff_summary("attention_context_all_heads", context_concat, context_ref_concat, total_head_dim, 1e-3);
        }
        timing.attention_ms = std::chrono::duration<double, std::milli>(attention_end - attention_start).count();

        auto proj_start = std::chrono::high_resolution_clock::now();
        {
            // Adaptive backend for attention output projection (context * Wo)
            // Use COSMA only for large prefill (same heuristic reused via adaptiveMatMul).
            bool ok = adaptiveMatMul(context_concat.data(),
                                     weights.wo[layer_idx]->data(),
                                     attn_out->data(),
                                     seq_len,        // m
                                     hidden_size,    // n
                                     total_head_dim, // k
                                     /*is_prefill*/ true,
                                     /*distributed_partition*/ false,
                                     /*transpose_A*/ false,
                                     /*transpose_B*/ false,
                                     /*alpha*/ 1.0f,
                                     /*beta*/ 0.0f);
            if (!ok)
            {
                LOG_ERROR("Attention output projection adaptiveMatMul failed; aborting layer " << layer_idx);
                return false;
            }
        }
        auto proj_end = std::chrono::high_resolution_clock::now();
        timing.linear_ms = std::chrono::duration<double, std::milli>(proj_end - proj_start).count();

        if (run_reference || trace_io)
        {
            std::vector<float> attn_out_vec(static_cast<size_t>(seq_len) * hidden_size, 0.f);
            std::memcpy(attn_out_vec.data(), attn_out->data(), attn_out_vec.size() * sizeof(float));
            log_buffer("attention_output", attn_out_vec.data(), attn_out_vec.size(), seq_len, hidden_size);
            if (run_reference)
            {
                std::vector<float> attn_ref(attn_out_vec.size(), 0.f);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            seq_len, hidden_size, total_head_dim,
                            1.0f,
                            context_concat.data(), total_head_dim,
                            weights.wo[layer_idx]->data(), hidden_size,
                            0.0f,
                            attn_ref.data(), hidden_size);
                log_diff_summary("attention_output", attn_out_vec, attn_ref, hidden_size, 1e-3);
            }
        }

        return true;
    }

    bool DistributedTransformerPipeline::executeOutputProjection(std::shared_ptr<TensorBase> &input,
                                                                 const ModelWeights &weights,
                                                                 std::shared_ptr<TensorBase> &output)
    {
        DEBUG_ASSERT(input, "Input tensor null before final normalization");
        int seq_len = input->shape()[0];
        const int hidden_size = config_.d_model;
        const int vocab_size = config_.vocab_size;

        const auto &pd_out = debugEnv().prefill_debug;
        const auto &abl_out = debugEnv().ablation;
        const bool trace_io = pd_out.trace_io;
        const bool debug_compare = pd_out.debug_compare;
        const bool debug_attention = pd_out.debug_attention;
        const bool debug_output = pd_out.debug_output;
        const bool run_reference = !(abl_out.ablate_attention || abl_out.ablate_ffn) && (trace_io || debug_compare || debug_attention || debug_output);

        const bool capture_baseline = debugEnv().baseline.capture;
        const bool compare_baseline = debugEnv().baseline.compare;
        const int baseline_rank = getRank();
        auto baseline_snapshot = [&](const std::string &name,
                                     const float *data,
                                     size_t count,
                                     int cols,
                                     double warn_threshold)
        {
            handle_prefill_stage_snapshot(baseline_rank, name, data, count, cols, warn_threshold, capture_baseline, compare_baseline);
        };

        const char *diag_tag = "[PrefillOutput]";
        if (trace_io)
        {
            diag_tag = "[PrefillDiag]";
        }
        else if (debug_output)
        {
            diag_tag = "[PrefillOutput]";
        }
        else if (debug_attention)
        {
            diag_tag = "[PrefillAttention]";
        }
        else if (debug_compare)
        {
            diag_tag = "[PrefillCompare]";
        }

        auto log_buffer = [&](const std::string &name, const float *data, size_t size, int rows, int cols)
        {
            if (!trace_io || rank_ != 0 || !data || size == 0)
            {
                return;
            }
            auto stats = computeBufferStats(data, size);
            LOG_INFO("[PrefillDiag] " << name << " stats rows=" << rows
                                      << " cols=" << cols
                                      << " min=" << stats.min
                                      << " max=" << stats.max
                                      << " mean=" << stats.mean
                                      << " stddev=" << stats.stddev
                                      << " rms=" << stats.rms
                                      << " l2=" << stats.l2);
            size_t bad_idx = 0;
            float bad_val = 0.f;
            if (findFirstNonFinite(data, size, bad_idx, bad_val))
            {
                if (cols > 0)
                {
                    int row = static_cast<int>(bad_idx / static_cast<size_t>(cols));
                    int col = static_cast<int>(bad_idx % static_cast<size_t>(cols));
                    LOG_WARN("[PrefillDiag] " << name << " non-finite value at idx=" << bad_idx
                                              << " (row=" << row << " col=" << col << ") value=" << bad_val);
                }
                else
                {
                    LOG_WARN("[PrefillDiag] " << name << " non-finite value at idx=" << bad_idx
                                              << " value=" << bad_val);
                }
            }
        };

        auto log_diff_summary = [&](const std::string &name,
                                    const std::vector<float> &buf,
                                    const std::vector<float> &ref,
                                    int cols,
                                    float warn_threshold)
        {
            if (!run_reference || rank_ != 0 || buf.size() != ref.size() || buf.empty())
            {
                return;
            }
            auto diff = computeDiffSummary(buf.data(), ref.data(), ref.size());
            auto samples = collectTopDiffSamples(buf.data(), ref.data(), ref.size(), 8);
            if (diff.rel_l2 > warn_threshold)
            {
                LOG_WARN(diag_tag << " " << name << " diff rel_l2=" << diff.rel_l2
                                  << " max_abs=" << diff.max_abs
                                  << " mean_abs=" << diff.mean_abs
                                  << " worst_index=" << diff.worst_index
                                  << " cosma=" << diff.value_a
                                  << " ref=" << diff.value_b);
            }
            else
            {
                LOG_INFO(diag_tag << " " << name << " diff rel_l2=" << diff.rel_l2
                                  << " max_abs=" << diff.max_abs
                                  << " mean_abs=" << diff.mean_abs
                                  << " worst_index=" << diff.worst_index
                                  << " cosma=" << diff.value_a
                                  << " ref=" << diff.value_b);
            }
            LOG_INFO(diag_tag << " " << name << " diff samples: " << formatDiffSamples(samples, cols));
        };

        auto log_preview = [&](const std::string &label, const float *ptr, size_t count)
        {
            if (!(debug_output || trace_io) || rank_ != 0 || !ptr || count == 0)
            {
                return;
            }
            size_t preview = std::min<size_t>(5, count);
            std::ostringstream oss;
            oss << diag_tag << " " << label << " first " << preview << " values:";
            for (size_t i = 0; i < preview; ++i)
            {
                oss << ' ' << ptr[i];
            }
            LOG_DEBUG(oss.str());
        };

        LOG_INFO("Starting output projection for seq_len=" << seq_len);

        // Snapshot the hidden state before final normalization
        baseline_snapshot("final_hidden", input->data(), static_cast<size_t>(seq_len) * hidden_size, hidden_size, 1e-6);

        bool bypass_final_norm = debugEnv().output_norm.bypass;
        if (bypass_final_norm && rank_ == 0)
        {
            LOG_WARN("[FinalNorm] Bypassing final output RMSNorm due to LLAMINAR_BYPASS_OUTPUT_NORM=1");
        }

        std::shared_ptr<TensorBase> norm_out;
        if (bypass_final_norm)
        {
            norm_out = input; // Reuse input directly
        }
        else
        {
            norm_out = createLocalTensor({seq_len, hidden_size});
            ASSERT_TENSOR_NOT_NAN(input, "Input tensor has NaN before final normalization");
            TensorLogger::logTensorStats(input, "final_norm_input");
            log_preview("final_norm_input", input->data(), static_cast<size_t>(seq_len) * hidden_size);

            std::vector<std::shared_ptr<TensorBase>> norm_inputs = {input, weights.output_norm_weight};
            std::vector<std::shared_ptr<TensorBase>> norm_outputs = {norm_out};

            LOG_INFO("Executing final RMSNorm kernel...");
            if (!executeKernel("rmsnorm", norm_inputs, norm_outputs))
            {
                LOG_ERROR("Output normalization failed");
                return false;
            }
        }

        ASSERT_TENSOR_NOT_NAN(norm_out, "Final norm output has NaN - THIS IS THE SOURCE!");
        TensorLogger::logTensorStats(norm_out, bypass_final_norm ? "final_norm_output_bypassed" : "final_norm_output");
        baseline_snapshot("final_hidden_normed",
                          norm_out->data(),
                          static_cast<size_t>(seq_len) * hidden_size,
                          hidden_size,
                          1e-5);
        log_preview("final_norm_output", norm_out->data(), static_cast<size_t>(seq_len) * hidden_size);
        log_buffer("final_norm_output", norm_out->data(), static_cast<size_t>(seq_len) * hidden_size, seq_len, hidden_size);

        std::vector<float> norm_ref;
        std::vector<float> norm_cosma_copy;
        const float *input_data = input ? input->data() : nullptr;
        if (run_reference && input_data)
        {
            norm_cosma_copy.resize(static_cast<size_t>(seq_len) * hidden_size);
            std::memcpy(norm_cosma_copy.data(), norm_out->data(), norm_cosma_copy.size() * sizeof(float));

            norm_ref.assign(norm_cosma_copy.size(), 0.f);
            const float *gamma_ptr = weights.output_norm_weight->data();
            for (int r = 0; r < seq_len; ++r)
            {
                const float *row = input_data + static_cast<size_t>(r) * hidden_size;
                double sum = 0.0;
                for (int c = 0; c < hidden_size; ++c)
                {
                    double v = static_cast<double>(row[c]);
                    sum += v * v;
                }
                double inv = 1.0 / std::sqrt(sum / std::max(1, hidden_size) + config_.eps);
                float *dst_row = norm_ref.data() + static_cast<size_t>(r) * hidden_size;
                for (int c = 0; c < hidden_size; ++c)
                {
                    dst_row[c] = static_cast<float>(row[c] * inv * gamma_ptr[c]);
                }
            }
            log_diff_summary("final_norm_output", norm_cosma_copy, norm_ref, hidden_size, 1e-5f);
        }

        // Capture pre-LM head hidden (post final norm) if requested
        if (debugEnv().pipeline.capture_pre_lm && getRank() == 0 && norm_out && norm_out->data())
        {
            size_t elems = (size_t)seq_len * hidden_size;
            last_pre_lm_hidden_.assign(norm_out->data(), norm_out->data() + elems);
        }
        LOG_INFO("Preparing LM head projection with vocab_size=" << vocab_size);
        LOG_DEBUG(diag_tag << " LM head weight shape: [" << weights.lm_head->shape()[0] << ", " << weights.lm_head->shape()[1] << "]");

        output = createLocalTensor({seq_len, vocab_size});

        // Instrumentation: snapshot last norm_out row prior to LM head to detect any mutation
        // or mismatch with captured last_pre_lm_hidden_ used by external tests.
        std::vector<float> pre_linear_last_row_snapshot; // empty unless instrumentation enabled
        int last_token_row = seq_len > 0 ? (seq_len - 1) : 0;
        bool enable_pre_lm_row_diff = debugEnv().pipeline.capture_pre_lm && debugEnv().pipeline.pre_lm_row_diff;
        if (enable_pre_lm_row_diff && getRank() == 0 && norm_out && norm_out->data() && hidden_size > 0 && seq_len > 0)
        {
            const float *row_ptr = norm_out->data() + static_cast<size_t>(last_token_row) * hidden_size;
            pre_linear_last_row_snapshot.assign(row_ptr, row_ptr + hidden_size);
            // Also compute immediate diff vs the copy we stored in last_pre_lm_hidden_ (captured full buffer)
            if (last_pre_lm_hidden_.size() >= static_cast<size_t>(seq_len) * hidden_size)
            {
                const float *captured_row = last_pre_lm_hidden_.data() + static_cast<size_t>(last_token_row) * hidden_size;
                double sum_sq = 0.0, ref_sq = 0.0, max_abs = 0.0;
                for (int i = 0; i < hidden_size; ++i)
                {
                    double diff = static_cast<double>(captured_row[i]) - static_cast<double>(row_ptr[i]);
                    sum_sq += diff * diff;
                    ref_sq += static_cast<double>(row_ptr[i]) * static_cast<double>(row_ptr[i]);
                    double a = std::fabs(diff);
                    if (a > max_abs)
                        max_abs = a;
                }
                double rel_l2 = (ref_sq > 0.0) ? std::sqrt(sum_sq / ref_sq) : 0.0;
                LOG_INFO("[PreLMRowDiff] stage=pre_linear type=capture_vs_live rel_l2=" << rel_l2 << " max_abs=" << max_abs << " row=" << last_token_row << " hidden_size=" << hidden_size);
            }
            else
            {
                LOG_INFO("[PreLMRowDiff] stage=pre_linear capture buffer too small for comparison");
            }
        }

        std::vector<std::shared_ptr<TensorBase>> lm_inputs = {norm_out, weights.lm_head};
        std::vector<std::shared_ptr<TensorBase>> lm_outputs = {output};

        LOG_INFO("Executing LM head linear kernel...");
        if (!executeKernel("linear", lm_inputs, lm_outputs))
        {
            LOG_ERROR("LM head projection failed");
            return false;
        }

        // Post-linear comparison: check whether norm_out last row mutated during the linear kernel
        if (enable_pre_lm_row_diff && !pre_linear_last_row_snapshot.empty() && getRank() == 0 && norm_out && norm_out->data())
        {
            const float *post_row_ptr = norm_out->data() + static_cast<size_t>(last_token_row) * hidden_size;
            double sum_sq = 0.0, ref_sq = 0.0, max_abs = 0.0;
            for (int i = 0; i < hidden_size; ++i)
            {
                double diff = static_cast<double>(post_row_ptr[i]) - static_cast<double>(pre_linear_last_row_snapshot[i]);
                sum_sq += diff * diff;
                ref_sq += static_cast<double>(pre_linear_last_row_snapshot[i]) * static_cast<double>(pre_linear_last_row_snapshot[i]);
                double a = std::fabs(diff);
                if (a > max_abs)
                    max_abs = a;
            }
            double rel_l2 = (ref_sq > 0.0) ? std::sqrt(sum_sq / ref_sq) : 0.0;
            LOG_INFO("[PreLMRowDiff] stage=post_linear type=live_mutation rel_l2=" << rel_l2 << " max_abs=" << max_abs << " row=" << last_token_row << " hidden_size=" << hidden_size);

            // Also compare the captured buffer again vs post-linear live row
            if (last_pre_lm_hidden_.size() >= static_cast<size_t>(seq_len) * hidden_size)
            {
                const float *captured_row = last_pre_lm_hidden_.data() + static_cast<size_t>(last_token_row) * hidden_size;
                double sum_sq2 = 0.0, ref_sq2 = 0.0, max_abs2 = 0.0;
                for (int i = 0; i < hidden_size; ++i)
                {
                    double diff = static_cast<double>(captured_row[i]) - static_cast<double>(post_row_ptr[i]);
                    sum_sq2 += diff * diff;
                    ref_sq2 += static_cast<double>(post_row_ptr[i]) * static_cast<double>(post_row_ptr[i]);
                    double a = std::fabs(diff);
                    if (a > max_abs2)
                        max_abs2 = a;
                }
                double rel_l22 = (ref_sq2 > 0.0) ? std::sqrt(sum_sq2 / ref_sq2) : 0.0;
                LOG_INFO("[PreLMRowDiff] stage=post_linear type=capture_vs_live rel_l2=" << rel_l22 << " max_abs=" << max_abs2 << " row=" << last_token_row << " hidden_size=" << hidden_size);
            }
        }

        ASSERT_TENSOR_NOT_NAN(output, "LM head output has NaN - final pipeline output contaminated!");
        TensorLogger::logTensorStats(output, "lm_head_output");
        baseline_snapshot("final_logits_pre_broadcast",
                          output->data(),
                          static_cast<size_t>(seq_len) * vocab_size,
                          vocab_size,
                          1e-3);
        log_preview("lm_head_output", output->data(), static_cast<size_t>(seq_len) * vocab_size);
        log_buffer("lm_head_output", output->data(), static_cast<size_t>(seq_len) * vocab_size, seq_len, vocab_size);

        if (run_reference)
        {
            std::vector<float> logits_cosma(static_cast<size_t>(seq_len) * vocab_size, 0.f);
            std::memcpy(logits_cosma.data(), output->data(), logits_cosma.size() * sizeof(float));

            std::vector<float> logits_ref(logits_cosma.size(), 0.f);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        seq_len, vocab_size, hidden_size,
                        1.0f,
                        norm_out->data(), hidden_size,
                        weights.lm_head->data(), vocab_size,
                        0.0f,
                        logits_ref.data(), vocab_size);
            log_diff_summary("lm_head_output", logits_cosma, logits_ref, vocab_size, 1e-3f);

            if (!norm_ref.empty())
            {
                std::vector<float> logits_ref_manual(logits_cosma.size(), 0.f);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            seq_len, vocab_size, hidden_size,
                            1.0f,
                            norm_ref.data(), hidden_size,
                            weights.lm_head->data(), vocab_size,
                            0.0f,
                            logits_ref_manual.data(), vocab_size);
                log_diff_summary("lm_head_ref_norm_out_vs_manual_norm", logits_ref, logits_ref_manual, vocab_size, 1e-6f);
            }
        }

        // Manual dot check instrumentation: LLAMINAR_LOGIT_DOT_CHECK
        // Format: "vocab_index[,vocab_index2,...][:row=token_row]" where token_row defaults to last token (seq_len-1)
        // Example: export LLAMINAR_LOGIT_DOT_CHECK="0,1,42:row=5"
        if (debugEnv().logit.dot_check)
        {
            std::string spec(debugEnv().logit.dot_check_spec);
            int target_row = seq_len - 1;
            auto row_pos = spec.find(":row=");
            if (row_pos != std::string::npos)
            {
                try
                {
                    target_row = std::stoi(spec.substr(row_pos + 5));
                }
                catch (...)
                {
                    LOG_WARN("[DotCheck] Failed to parse row specifier, defaulting to last row");
                }
                spec = spec.substr(0, row_pos);
            }
            if (target_row < 0 || target_row >= seq_len)
            {
                LOG_WARN("[DotCheck] target_row out of range; skipping dot check");
            }
            else
            {
                // Parse vocab indices
                std::vector<int> vocab_indices;
                std::stringstream ss(spec);
                std::string item;
                while (std::getline(ss, item, ','))
                {
                    if (item.empty())
                        continue;
                    try
                    {
                        int vidx = std::stoi(item);
                        if (vidx >= 0 && vidx < vocab_size)
                            vocab_indices.push_back(vidx);
                        else
                            LOG_WARN("[DotCheck] vocab index out of range: " << vidx);
                    }
                    catch (...)
                    {
                        LOG_WARN("[DotCheck] failed to parse vocab index: " << item);
                    }
                }
                if (vocab_indices.empty())
                {
                    LOG_WARN("[DotCheck] No valid vocab indices parsed; skipping");
                }
                else if (!weights.lm_head || !weights.lm_head->data())
                {
                    LOG_WARN("[DotCheck] LM head weights unavailable; skipping");
                }
                else
                {
                    const bool dump_vectors = debugEnv().logit.dot_dump;
                    const bool try_pre_norm = debugEnv().logit.dot_prenorm;
                    const float *row_hidden_normed = norm_out->data() + static_cast<size_t>(target_row) * hidden_size;
                    const float *row_hidden_prenorm = nullptr;
                    if (try_pre_norm && input) // 'input' here is pre-norm buffer feeding RMSNorm
                    {
                        row_hidden_prenorm = input->data() + static_cast<size_t>(target_row) * hidden_size;
                    }
                    const float *W = weights.lm_head->data(); // assumed layout [hidden_size x vocab_size] row-major

                    // Precompute column norms and hidden row norm for diagnostics.
                    auto compute_l2 = [](const float *ptr, int len) -> double
                    {
                        double s = 0.0; for (int i = 0; i < len; ++i) { double v = ptr[i]; s += v * v; } return std::sqrt(s); };
                    double hidden_l2 = compute_l2(row_hidden_normed, hidden_size);
                    double hidden_pre_l2 = row_hidden_prenorm ? compute_l2(row_hidden_prenorm, hidden_size) : 0.0;

                    // Direct projection (assumed orientation)
                    struct DotSample
                    {
                        int v;
                        float pipeline_val;
                        float proj_normed;
                        float proj_prenorm; // 0 if not available
                        float col_l2;
                        float diff_normed;
                        float rel_normed;
                        float diff_prenorm;
                        float rel_prenorm;
                    };
                    std::vector<DotSample> samples;
                    samples.reserve(vocab_indices.size());

                    double best_scale_num = 0.0; // numerator for scale inference (sum pipe*proj)
                    double best_scale_den = 0.0; // denominator (sum proj^2)

                    for (int v : vocab_indices)
                    {
                        double col_l2_acc = 0.0;
                        double acc = 0.0;
                        double acc_pre = 0.0;
                        for (int h = 0; h < hidden_size; ++h)
                        {
                            double w = static_cast<double>(W[static_cast<size_t>(h) * vocab_size + v]);
                            double hv = static_cast<double>(row_hidden_normed[h]);
                            acc += hv * w;
                            col_l2_acc += w * w;
                            if (row_hidden_prenorm)
                                acc_pre += static_cast<double>(row_hidden_prenorm[h]) * w;
                        }
                        float pipeline_val = output->data()[static_cast<size_t>(target_row) * vocab_size + v];
                        float proj = static_cast<float>(acc);
                        float diff = proj - pipeline_val;
                        float rel = (std::fabs(pipeline_val) > 1e-8f) ? diff / pipeline_val : 0.0f;
                        float proj_pre = row_hidden_prenorm ? static_cast<float>(acc_pre) : 0.0f;
                        float diff_pre = row_hidden_prenorm ? (proj_pre - pipeline_val) : 0.0f;
                        float rel_pre = (row_hidden_prenorm && std::fabs(pipeline_val) > 1e-8f) ? diff_pre / pipeline_val : 0.0f;
                        float col_l2 = static_cast<float>(std::sqrt(col_l2_acc));

                        samples.push_back({v, pipeline_val, proj, proj_pre, col_l2, diff, rel, diff_pre, rel_pre});
                        best_scale_num += static_cast<double>(pipeline_val) * static_cast<double>(proj);
                        best_scale_den += static_cast<double>(proj) * static_cast<double>(proj);
                    }

                    double inferred_scale = 0.0;
                    bool have_scale = best_scale_den > 0.0;
                    if (have_scale)
                        inferred_scale = best_scale_num / best_scale_den;

                    // Log primary line with raw projections
                    {
                        std::ostringstream oss;
                        oss << "[DotCheck] row=" << target_row << " hidden_l2=" << hidden_l2;
                        if (row_hidden_prenorm)
                            oss << " pre_hidden_l2=" << hidden_pre_l2;
                        for (auto &s : samples)
                        {
                            oss << " | v=" << s.v
                                << " proj=" << s.proj_normed
                                << " pipe=" << s.pipeline_val
                                << " diff=" << s.diff_normed
                                << " rel=" << s.rel_normed
                                << " col_l2=" << s.col_l2;
                            if (row_hidden_prenorm)
                            {
                                oss << " pre_proj=" << s.proj_prenorm
                                    << " pre_diff=" << s.diff_prenorm
                                    << " pre_rel=" << s.rel_prenorm;
                            }
                        }
                        if (have_scale)
                            oss << " | inferred_scale=" << inferred_scale;
                        LOG_INFO(oss.str());
                    }

                    // Optional scaled residuals (always emitted now as per request)
                    if (have_scale)
                    {
                        double rms_residual = 0.0;
                        double rms_residual_pre = 0.0;
                        std::ostringstream oss_scaled;
                        oss_scaled << "[DotCheckScaled] row=" << target_row << " scale=" << inferred_scale;
                        for (auto &s : samples)
                        {
                            float scaled_proj = s.proj_normed * static_cast<float>(inferred_scale);
                            float scaled_diff = scaled_proj - s.pipeline_val;
                            float scaled_rel = (std::fabs(s.pipeline_val) > 1e-8f) ? scaled_diff / s.pipeline_val : 0.0f;
                            rms_residual += static_cast<double>(scaled_diff) * static_cast<double>(scaled_diff);
                            oss_scaled << " | v=" << s.v
                                       << " scaled_proj=" << scaled_proj
                                       << " scaled_diff=" << scaled_diff
                                       << " scaled_rel=" << scaled_rel;
                            if (row_hidden_prenorm)
                            {
                                float scaled_pre = s.proj_prenorm * static_cast<float>(inferred_scale);
                                float scaled_pre_diff = scaled_pre - s.pipeline_val;
                                float scaled_pre_rel = (std::fabs(s.pipeline_val) > 1e-8f) ? scaled_pre_diff / s.pipeline_val : 0.0f;
                                rms_residual_pre += static_cast<double>(scaled_pre_diff) * static_cast<double>(scaled_pre_diff);
                                oss_scaled << " pre_scaled_proj=" << scaled_pre
                                           << " pre_scaled_diff=" << scaled_pre_diff
                                           << " pre_scaled_rel=" << scaled_pre_rel;
                            }
                        }
                        if (!samples.empty())
                        {
                            rms_residual = std::sqrt(rms_residual / samples.size());
                            if (row_hidden_prenorm)
                                rms_residual_pre = std::sqrt(rms_residual_pre / samples.size());
                            oss_scaled << " | rms_scaled_residual=" << rms_residual;
                            if (row_hidden_prenorm)
                                oss_scaled << " pre_rms_scaled_residual=" << rms_residual_pre;
                        }
                        LOG_INFO(oss_scaled.str());
                    }

                    // Orientation sanity test (transpose hypothesis)
                    std::ostringstream oss_t;
                    oss_t << "[DotCheckTransposeHypothesis] row=" << target_row;
                    for (int v : vocab_indices)
                    {
                        double acc_t = 0.0;
                        double acc_t_pre = 0.0;
                        for (int h = 0; h < hidden_size; ++h)
                        {
                            acc_t += static_cast<double>(row_hidden_normed[h]) * static_cast<double>(W[static_cast<size_t>(v) * hidden_size + h]);
                            if (row_hidden_prenorm)
                            {
                                acc_t_pre += static_cast<double>(row_hidden_prenorm[h]) * static_cast<double>(W[static_cast<size_t>(v) * hidden_size + h]);
                            }
                        }
                        float pipeline_val = output->data()[static_cast<size_t>(target_row) * vocab_size + v];
                        float projected_t = static_cast<float>(acc_t);
                        float diff_t = projected_t - pipeline_val;
                        float rel_t = (std::fabs(pipeline_val) > 1e-8f) ? diff_t / pipeline_val : 0.0f;
                        oss_t << " | v=" << v << " trans_proj=" << projected_t << " pipe=" << pipeline_val << " diff_t=" << diff_t << " rel_t=" << rel_t;
                        if (row_hidden_prenorm)
                        {
                            float projected_t_pre = static_cast<float>(acc_t_pre);
                            float diff_t_pre = projected_t_pre - pipeline_val;
                            float rel_t_pre = (std::fabs(pipeline_val) > 1e-8f) ? diff_t_pre / pipeline_val : 0.0f;
                            oss_t << " trans_pre_proj=" << projected_t_pre << " trans_pre_diff=" << diff_t_pre << " trans_pre_rel=" << rel_t_pre;
                        }
                    }
                    LOG_INFO(oss_t.str());

                    if (dump_vectors && !vocab_indices.empty())
                    {
                        int v0 = vocab_indices.front();
                        std::ostringstream hv_oss;
                        hv_oss << "[DotCheckDumpHiddenRow] row=" << target_row;
                        int dump_elems = std::min(hidden_size, 32);
                        for (int h = 0; h < dump_elems; ++h)
                            hv_oss << ' ' << row_hidden_normed[h];
                        LOG_INFO(hv_oss.str());
                        if (row_hidden_prenorm)
                        {
                            std::ostringstream pre_oss;
                            pre_oss << "[DotCheckDumpPreNormRow] row=" << target_row;
                            for (int h = 0; h < dump_elems; ++h)
                                pre_oss << ' ' << row_hidden_prenorm[h];
                            LOG_INFO(pre_oss.str());
                        }
                        std::ostringstream col_oss;
                        col_oss << "[DotCheckDumpWeightCol] v=" << v0;
                        for (int h = 0; h < dump_elems; ++h)
                            col_oss << ' ' << W[static_cast<size_t>(h) * vocab_size + v0];
                        LOG_INFO(col_oss.str());
                        std::ostringstream col_t_oss;
                        col_t_oss << "[DotCheckDumpWeightRowAsColHypothesis] v=" << v0;
                        for (int h = 0; h < dump_elems; ++h)
                            col_t_oss << ' ' << W[static_cast<size_t>(v0) * hidden_size + h];
                        LOG_INFO(col_t_oss.str());
                    }
                }
            }
        }

        LOG_DEBUG("Output projection completed: " << seq_len << "x" << hidden_size
                                                  << " -> " << seq_len << "x" << vocab_size);
        return true;
    }

    bool DistributedTransformerPipeline::validate(const ModelWeights &weights) const
    {
        // Validate embedding weights
        if (!weights.token_embedding)
        {
            LOG_ERROR("Token embedding is null");
            return false;
        }

        auto shape = weights.token_embedding->shape();
        LOG_DEBUG("Token embedding shape: [" << shape[0] << ", " << shape[1] << "]");
        LOG_DEBUG("Expected vocab_size: " << config_.vocab_size << ", d_model: " << config_.d_model);

        if (shape.size() != 2)
        {
            LOG_ERROR("Token embedding has " << shape.size() << " dimensions, expected 2");
            return false;
        }

        // Accept either layout:
        //  1) [vocab_size, d_model]  (typical embedding table layout produced by loader)
        //  2) [d_model, vocab_size]  (projection / transposed layout some kernels expect)
        // We only hard-fail if neither orientation matches expected dimensions.
        bool vocab_first = (shape[0] == config_.vocab_size && shape[1] == config_.d_model);
        bool model_first = (shape[0] == config_.d_model && shape[1] == config_.vocab_size);

        if (!vocab_first && !model_first)
        {
            LOG_ERROR("Token embedding shape incompatible with config. Got [" << shape[0] << ", " << shape[1]
                                                                              << "], expected either [vocab_size=" << config_.vocab_size << ", d_model=" << config_.d_model
                                                                              << "] or its transpose.");
            return false;
        }

        if (vocab_first)
        {
            LOG_DEBUG("Token embedding recognized as [vocab_size, d_model] layout (standard embedding table).");
        }
        else // model_first alternative layout
        {
            // Previously this emitted a warning implying a potential orientation issue.
            // Embedding orientation parity tests now confirm both layouts are semantically valid.
            // We downgrade to INFO (once) unless the user explicitly re-enables the warning.
            static bool logged_once = false;
            if (!logged_once)
            {
                if (debugEnv().embedding_warn.transpose_warn)
                {
                    LOG_WARN("Token embedding recognized as [d_model, vocab_size] (transpose) layout; treating as projection weight (warning forced by LLAMINAR_EMBEDDING_TRANSPOSE_WARN=1).");
                }
                else
                {
                    LOG_INFO("Token embedding recognized as [d_model, vocab_size] layout (accepted alternative; warning suppressed by default).");
                }
                logged_once = true;
            }
        }

        // Validate layer weights
        if (weights.attn_norm_weight.size() != config_.n_layers ||
            weights.wq.size() != config_.n_layers ||
            weights.wk.size() != config_.n_layers ||
            weights.wv.size() != config_.n_layers ||
            weights.wo.size() != config_.n_layers ||
            weights.ffn_norm_weight.size() != config_.n_layers ||
            weights.w_gate.size() != config_.n_layers ||
            weights.w_up.size() != config_.n_layers ||
            weights.w_down.size() != config_.n_layers)
        {
            LOG_ERROR("Inconsistent layer weight count");
            return false;
        }

        // Validate attention weight shapes for each layer
        int total_head_dim = config_.n_head * config_.head_dim;
        int kv_head_dim = config_.n_head_kv * config_.head_dim;

        for (int i = 0; i < config_.n_layers; ++i)
        {
            // Attention norm weights
            if (!weights.attn_norm_weight[i] ||
                weights.attn_norm_weight[i]->shape().size() != 1 ||
                weights.attn_norm_weight[i]->shape()[0] != config_.d_model)
            {
                LOG_ERROR("Invalid attention norm weight shape at layer " << i);
                return false;
            }

            // Q, K, V weight matrices
            if (!weights.wq[i] || weights.wq[i]->shape().size() != 2 ||
                weights.wq[i]->shape()[0] != config_.d_model ||
                weights.wq[i]->shape()[1] != total_head_dim)
            {
                LOG_ERROR("Invalid query weight shape at layer " << i);
                return false;
            }

            if (!weights.wk[i] || weights.wk[i]->shape().size() != 2 ||
                weights.wk[i]->shape()[0] != config_.d_model ||
                weights.wk[i]->shape()[1] != kv_head_dim)
            {
                LOG_ERROR("Invalid key weight shape at layer " << i);
                return false;
            }

            if (!weights.wv[i] || weights.wv[i]->shape().size() != 2 ||
                weights.wv[i]->shape()[0] != config_.d_model ||
                weights.wv[i]->shape()[1] != kv_head_dim)
            {
                LOG_ERROR("Invalid value weight shape at layer " << i);
                return false;
            }

            if (!weights.wo[i] || weights.wo[i]->shape().size() != 2 ||
                weights.wo[i]->shape()[0] != total_head_dim ||
                weights.wo[i]->shape()[1] != config_.d_model)
            {
                LOG_ERROR("Invalid output weight shape at layer " << i);
                return false;
            }

            // FFN weights
            if (!weights.ffn_norm_weight[i] ||
                weights.ffn_norm_weight[i]->shape().size() != 1 ||
                weights.ffn_norm_weight[i]->shape()[0] != config_.d_model)
            {
                LOG_ERROR("Invalid FFN norm weight shape at layer " << i);
                return false;
            }

            if (!weights.w_gate[i] || weights.w_gate[i]->shape().size() != 2 ||
                weights.w_gate[i]->shape()[0] != config_.d_model ||
                weights.w_gate[i]->shape()[1] != config_.d_ff)
            {
                LOG_ERROR("Invalid gate weight shape at layer " << i);
                return false;
            }

            if (!weights.w_up[i] || weights.w_up[i]->shape().size() != 2 ||
                weights.w_up[i]->shape()[0] != config_.d_model ||
                weights.w_up[i]->shape()[1] != config_.d_ff)
            {
                LOG_ERROR("Invalid up weight shape at layer " << i);
                return false;
            }

            if (!weights.w_down[i] || weights.w_down[i]->shape().size() != 2 ||
                weights.w_down[i]->shape()[0] != config_.d_ff ||
                weights.w_down[i]->shape()[1] != config_.d_model)
            {
                LOG_ERROR("Invalid down weight shape at layer " << i);
                return false;
            }
        }

        // Validate output weights
        if (!weights.output_norm_weight ||
            weights.output_norm_weight->shape().size() != 1 ||
            weights.output_norm_weight->shape()[0] != config_.d_model)
        {
            LOG_ERROR("Invalid output norm weight shape");
            return false;
        }

        if (!weights.lm_head || weights.lm_head->shape().size() != 2 ||
            weights.lm_head->shape()[0] != config_.d_model ||
            weights.lm_head->shape()[1] != config_.vocab_size)
        {
            LOG_ERROR("Invalid LM head weight shape");
            return false;
        }

        return true;
    }

    // Implement abstract interface from PipelineBase (not used in main execution path)
    bool DistributedTransformerPipeline::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                                 std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        LOG_ERROR("DistributedTransformerPipeline::execute(vector) not supported; use execute(token_ids, weights, output) overload");
        return false;
    }

    bool DistributedTransformerPipeline::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                                  const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        // Minimal shape checks; pipeline primary API differs.
        if (inputs.empty() || outputs.empty())
            return false;
        return true;
    }

    std::vector<std::shared_ptr<TensorBase>> DistributedTransformerPipeline::createIntermediateTensors(int seq_len)
    {
        std::vector<std::shared_ptr<TensorBase>> tensors;

        // Create tensors for intermediate computation
        // Use broadcast-compatible tensors since these will be shared between ranks
        tensors.push_back(createBroadcastTensor({seq_len, config_.d_model})); // Layer input (broadcast from embedding)
        tensors.push_back(createLocalTensor({seq_len, config_.d_model}));     // Layer output

        return tensors;
    }

    // ================= AbstractPipeline interface implementation =================
    bool DistributedTransformerPipeline::prefill(const std::vector<int> &tokens,
                                                 const IModelWeights &weights_iface,
                                                 StageContext &ctx)
    {
        setStagePrefill();
        current_tokens_ = tokens;
        const auto *w = dynamic_cast<const QwenModelWeights *>(&weights_iface);
        if (!w)
        {
            LOG_ERROR("prefill: Invalid weights type for Qwen pipeline");
            return false;
        }
        auto output = TensorFactory::create_simple({(int)tokens.size(), config_.vocab_size});
        if (!execute(tokens, w->inner, output))
            return false;
        last_logits_ = output;
        kv_snapshot_.capacity_tokens = getKVCacheCapacity();
        kv_snapshot_.used_tokens = getKVCacheUsed();
        kv_snapshot_.growth_events = getKVCacheGrowthEvents();
        ctx.stage = InferenceStage::Prefill;
        ctx.seq_len = (int)tokens.size();
        ctx.generated = 0;
        ctx.kv_capacity = kv_snapshot_.capacity_tokens;
        ctx.kv_used = kv_snapshot_.used_tokens;
        return true;
    }

    bool DistributedTransformerPipeline::decode(int next_token,
                                                const IModelWeights &weights_iface,
                                                StageContext &ctx)
    {
        const auto *w = dynamic_cast<const QwenModelWeights *>(&weights_iface);
        if (!w)
        {
            LOG_ERROR("decode: Invalid weights type for Qwen pipeline");
            return false;
        }
        setStageDecode();
        std::shared_ptr<TensorBase> one_logits;
        bool used_incremental = incrementalDecodeToken(next_token, w->inner, one_logits);
        if (!used_incremental)
        {
            current_tokens_.push_back(next_token);
            auto replay = TensorFactory::create_simple({(int)current_tokens_.size(), config_.vocab_size});
            if (!execute(current_tokens_, w->inner, replay))
                return false;
            last_logits_ = replay;
        }
        else
        {
            current_tokens_.push_back(next_token);
            if (!last_logits_)
            {
                last_logits_ = one_logits;
            }
            else
            {
                int prev_rows = last_logits_->shape()[0];
                int vocab = last_logits_->shape()[1];
                auto combined = TensorFactory::create_simple({prev_rows + 1, vocab});
                float *dst = const_cast<float *>(combined->data());
                std::memcpy(dst, last_logits_->data(), sizeof(float) * prev_rows * vocab);
                std::memcpy(dst + prev_rows * vocab, one_logits->data(), sizeof(float) * vocab);
                last_logits_ = combined;
            }
        }
        kv_snapshot_.capacity_tokens = getKVCacheCapacity();
        kv_snapshot_.used_tokens = getKVCacheUsed();
        kv_snapshot_.growth_events = getKVCacheGrowthEvents();
        ctx.stage = InferenceStage::Decode;
        ctx.seq_len = (int)current_tokens_.size();
        ctx.generated += 1;
        ctx.kv_capacity = kv_snapshot_.capacity_tokens;
        ctx.kv_used = kv_snapshot_.used_tokens;
        return true;
    }

    bool DistributedTransformerPipeline::logits(std::shared_ptr<TensorBase> &out_logits)
    {
        out_logits = last_logits_;
        return (bool)last_logits_;
    }

    const KVCacheState *DistributedTransformerPipeline::kvCacheState() const
    {
        return &kv_snapshot_;
    }

    bool DistributedTransformerPipeline::ensureKVCapacity(int required_tokens)
    {
        return ensureKVCapacityPublic(required_tokens);
    }

    void DistributedTransformerPipeline::initializeKVCache(int seq_len)
    {
        if (getRank() == 0)
            LOG_INFO("[KVCacheInit] this=" << (const void *)this << " resizing to seq_len=" << seq_len);
        k_cache_.clear();
        v_cache_.clear();

        for (int i = 0; i < config_.n_layers; ++i)
        {
            int kv_dim = config_.n_head_kv * config_.head_dim;
            k_cache_.push_back(createLocalTensor({seq_len, kv_dim}));
            v_cache_.push_back(createLocalTensor({seq_len, kv_dim}));

            // Initialize cache to zeros
            k_cache_[i]->zero();
            v_cache_[i]->zero();
        }
        kv_cache_state_.capacity_tokens = seq_len;
        if (kv_cache_state_.used_tokens > kv_cache_state_.capacity_tokens)
            kv_cache_state_.used_tokens = kv_cache_state_.capacity_tokens; // clamp safety
        if (getRank() == 0)
            LOG_INFO("[KVCacheInit] completed capacity=" << seq_len << " this=" << (const void *)this);
    }

    bool DistributedTransformerPipeline::ensureKVCapacityInternal(int required_tokens)
    {
        if (!use_kv_cache_)
            return false;
        if (required_tokens <= kv_cache_state_.capacity_tokens)
            return true; // already sufficient
        int max_allowed = config_.max_seq_len;
        if (required_tokens > max_allowed)
        {
            if (getRank() == 0)
                LOG_ERROR("[KVCache] required_tokens=" << required_tokens << " exceeds max_seq_len=" << max_allowed);
            return false;
        }
        // Growth policy
        int growth_factor = std::max(1, std::min(16, debugEnv().kv_cache.growth_factor));
        int new_capacity = kv_cache_state_.capacity_tokens > 0 ? kv_cache_state_.capacity_tokens : 1;
        while (new_capacity < required_tokens)
        {
            long long cand = (long long)new_capacity * growth_factor;
            if (cand > max_allowed)
            {
                new_capacity = max_allowed;
                break;
            }
            new_capacity = (int)cand;
        }
        if (new_capacity > max_allowed)
            new_capacity = max_allowed;
        // Allocate new caches and copy existing data
        if (getRank() == 0)
            LOG_INFO("[KVCacheGrowth] old_cap=" << kv_cache_state_.capacity_tokens << " new_cap=" << new_capacity
                                                << " required=" << required_tokens << " factor=" << growth_factor);
        std::vector<std::shared_ptr<TensorBase>> new_k;
        std::vector<std::shared_ptr<TensorBase>> new_v;
        new_k.reserve(config_.n_layers);
        new_v.reserve(config_.n_layers);
        for (int i = 0; i < config_.n_layers; ++i)
        {
            int kv_dim = config_.n_head_kv * config_.head_dim;
            auto nk = createLocalTensor({new_capacity, kv_dim});
            auto nv = createLocalTensor({new_capacity, kv_dim});
            // zero initialize new region
            nk->zero();
            nv->zero();
            // copy old rows if any
            if (i < (int)k_cache_.size() && k_cache_[i] && v_cache_[i])
            {
                int rows_copy = std::min(kv_cache_state_.used_tokens, kv_cache_state_.capacity_tokens);
                if (rows_copy > 0)
                {
                    size_t bytes = (size_t)rows_copy * kv_dim * sizeof(float);
                    std::memcpy(nk->data(), k_cache_[i]->data(), bytes);
                    std::memcpy(nv->data(), v_cache_[i]->data(), bytes);
                }
            }
            new_k.push_back(nk);
            new_v.push_back(nv);
        }
        k_cache_.swap(new_k);
        v_cache_.swap(new_v);
        kv_cache_state_.capacity_tokens = new_capacity;
        kv_cache_state_.growth_events += 1;
        if (getRank() == 0)
            LOG_INFO("[KVCacheSummary] growth_event=" << kv_cache_state_.growth_events
                                                      << " used_tokens=" << kv_cache_state_.used_tokens
                                                      << " capacity=" << kv_cache_state_.capacity_tokens);
        return true;
    }

    // Legacy createDistributedTransformerPipeline moved to distributed_transformer_pipeline.cpp
    // (stub kept to avoid accidental use if this TU is ever re-enabled)
    std::unique_ptr<DistributedTransformerPipeline> createDistributedTransformerPipeline(
        const DistributedTransformerPipeline::LayerConfig &config)
    {
        LOG_ERROR("Legacy mpi_transformer_pipeline.cpp factory invoked unexpectedly");
        return std::make_unique<DistributedTransformerPipeline>(config);
    }

    // Forward declaration of unified helper (defined below)
    static DistributedTransformerPipeline::ModelWeights loadModelWeightsExistingLoaderImpl(
        ModelLoader &loader,
        const DistributedTransformerPipeline::LayerConfig &config);

    // Internal helper extracted from the second overload (unified rich diagnostics)
    static DistributedTransformerPipeline::ModelWeights loadModelWeightsExistingLoaderImpl(
        ModelLoader &loader,
        const DistributedTransformerPipeline::LayerConfig &config)
    {
        DistributedTransformerPipeline::ModelWeights weights;
        LOG_INFO("Loading model weights using existing ModelLoader");
        // Token embedding load + diagnostics
        weights.token_embedding = loader.loadTensor("token_embd.weight");
        if (!weights.token_embedding)
            throw std::runtime_error("Failed to load token embedding");
        if (debugEnv().sharding.shard_load_diag)
        {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0)
                LOG_INFO("[RoleTag] token_embd.weight -> Embedding (vocab_size x d_model)");
        }
        const auto emb_shape = weights.token_embedding->shape();
        LOG_INFO("Token embedding shape: [" << emb_shape[0] << ", " << emb_shape[1] << "] expected vocab=" << config.vocab_size << " d_model=" << config.d_model);
        {
            const float *ptr = weights.token_embedding->data();
            size_t total = (size_t)emb_shape[0] * emb_shape[1];
            size_t sample = std::min<size_t>(total, 1000);
            size_t nan_ct = 0, inf_ct = 0;
            for (size_t i = 0; i < sample; ++i)
            {
                float v = ptr[i];
                if (std::isnan(v))
                    ++nan_ct;
                if (std::isinf(v))
                    ++inf_ct;
            }
            LOG_INFO("Token embedding validation (first " << sample << "): " << nan_ct << " NaN, " << inf_ct << " inf");
            int first_n = std::min<int>(10, (int)total);
            if (first_n > 0)
            {
                std::ostringstream oss;
                oss << "First 10 token embedding values:";
                for (int i = 0; i < first_n; ++i)
                    oss << " [" << i << "]=" << ptr[i];
                LOG_DEBUG(oss.str());
            }
        }
        // Output norm
        weights.output_norm_weight = loader.loadTensor("output_norm.weight");
        if (!weights.output_norm_weight)
            throw std::runtime_error("Failed to load output norm weights");
        const float *out_gamma = weights.output_norm_weight->data();
        int gamma_n = weights.output_norm_weight->size();
        LOG_INFO("Loaded output_norm.weight size=" << gamma_n);
        for (int j = 0; j < std::min(gamma_n, 10); ++j)
            if (std::isnan(out_gamma[j]))
                throw std::runtime_error("NaN in output_norm.weight");
        {
            auto checksum = [](const float *p, int n)
            { const uint64_t OFF=1469598103934665603ull,PR=1099511628211ull; uint64_t h=OFF; const uint8_t *b=reinterpret_cast<const uint8_t*>(p); size_t L=(size_t)n*sizeof(float); for(size_t i=0;i<L;++i){ h^=b[i]; h*=PR;} return h; };
            uint64_t cs = checksum(out_gamma, gamma_n);
            double gmin = std::numeric_limits<double>::infinity(), gmax = -gmin;
            long double gsum = 0.0L;
            for (int i = 0; i < gamma_n; ++i)
            {
                double v = out_gamma[i];
                gmin = std::min(gmin, v);
                gmax = std::max(gmax, v);
                gsum += v;
            }
            double gmean = (double)(gsum / (long double)gamma_n);
            std::ostringstream oss;
            oss << "[OutputNormIntegrity] size=" << gamma_n << " checksum=0x" << std::hex << cs << std::dec << " min=" << gmin << " max=" << gmax << " mean=" << gmean << " first32=";
            int dump = std::min(32, gamma_n);
            for (int i = 0; i < dump; ++i)
            {
                oss << out_gamma[i];
                if (i + 1 < dump)
                    oss << ',';
            }
            LOG_INFO(oss.str());
            if (debugEnv().rmsnorm.gamma_checksum && (gmax > 4.0 || gmin < -1.0 || gmean > 2.0))
                LOG_WARN("[OutputNormIntegrity] gamma statistics outside expected range (Qwen2.5 typical ~[0,2])");
        }
        if (gamma_n != config.d_model)
            LOG_ERROR("output_norm.weight size " << gamma_n << " != d_model=" << config.d_model);
        bool override_applied = false;
        if (debugEnv().output_norm.force_unit || debugEnv().output_norm.force_unit_all)
        {
            for (int i = 0; i < gamma_n; ++i)
                const_cast<float *>(out_gamma)[i] = 1.0f;
            override_applied = true;
            LOG_WARN("[OutputNormIntegrity] Forcing output_norm.weight to all 1.0");
        }
        else if (debugEnv().output_norm.clamp)
        {
            float lo = 0.f, hi = 4.f;
            int clamp_ct = 0;
            for (int i = 0; i < gamma_n; ++i)
            {
                float v = out_gamma[i];
                float nv = std::clamp(v, lo, hi);
                if (nv != v)
                    ++clamp_ct;
                const_cast<float *>(out_gamma)[i] = nv;
            }
            override_applied = true;
            LOG_WARN("[OutputNormIntegrity] Clamped gamma due to LLAMINAR_OUTPUT_NORM_CLAMP=1");
        }
        if (override_applied)
        {
            double gmin = 1e9, gmax = -1e9;
            long double gsum = 0.0L;
            for (int i = 0; i < gamma_n; ++i)
            {
                double v = out_gamma[i];
                gmin = std::min(gmin, v);
                gmax = std::max(gmax, v);
                gsum += v;
            }
            double gmean = (double)(gsum / (long double)gamma_n);
            LOG_INFO("[OutputNormIntegrity] Post-override stats: min=" << gmin << " max=" << gmax << " mean=" << gmean);
        }
        // LM head
        std::shared_ptr<TensorBase> lm_head;
        try
        {
            lm_head = loader.loadTensor("output.weight");
        }
        catch (const std::exception &e)
        {
            LOG_WARN("Exception while loading output.weight: " << e.what());
        }
        if (lm_head)
        {
            try
            {
                bool raw = debugEnv().lm_head.raw_orientation;
                auto orig = lm_head->shape();
                if (!raw)
                {
                    enforce_matrix_layout(lm_head, config.d_model, config.vocab_size, "output.weight");
                }
                else
                {
                    LOG_WARN("[LMHeadOrientation] Skipping layout enforcement for output.weight; raw shape [" << orig[0] << "," << orig[1] << "]");
                }
                weights.lm_head = lm_head;
                LOG_INFO("Loaded lm_head from output.weight tensor (final shape " << weights.lm_head->shape()[0] << "x" << weights.lm_head->shape()[1] << ")");
                if (debugEnv().lm_head.cosine_diag)
                {
                    const float *wptr = weights.lm_head->data();
                    const auto &sh = weights.lm_head->shape();
                    if (sh.size() == 2)
                    {
                        int R = sh[0];
                        int C = sh[1];
                        auto cosine = [](const float *a, const float *b, int n)
                        {
                            long double dot = 0, na = 0, nb = 0;
                            for (int i = 0; i < n; ++i)
                            {
                                long double va = a[i], vb = b[i];
                                dot += va * vb;
                                na += va * va;
                                nb += vb * vb;
                            }
                            if (na <= 0 || nb <= 0)
                                return 0.0;
                            return (double)(dot / (std::sqrt((double)na) * std::sqrt((double)nb)));
                        };
                        int samples = std::min(8, C - 1);
                        if (samples > 0)
                        {
                            std::vector<double> sims;
                            sims.reserve(samples);
                            for (int j = 0; j < samples; ++j)
                            {
                                const float *colA = wptr + j;
                                const float *colB = wptr + (j + 1);
                                std::vector<float> bufA(R), bufB(R);
                                for (int r = 0; r < R; ++r)
                                {
                                    bufA[r] = colA[r * C];
                                    bufB[r] = colB[r * C];
                                }
                                sims.push_back(cosine(bufA.data(), bufB.data(), R));
                            }
                            double smin = 1e9, smax = -1e9;
                            long double ssum = 0;
                            for (double v : sims)
                            {
                                smin = std::min(smin, v);
                                smax = std::max(smax, v);
                                ssum += v;
                            }
                            double smean = sims.empty() ? 0.0 : (double)(ssum / sims.size());
                            std::ostringstream oss;
                            oss << "[LMHeadCosineDiag] orientation=" << (raw ? "raw" : "enforced")
                                << " shape=" << R << "x" << C
                                << " adj_column_cosine min=" << smin << " max=" << smax << " mean=" << smean
                                << " samples=" << sims.size();
                            LOG_INFO(oss.str());
                        }
                    }
                }
                // Additional row/col norm diag
                if (weights.lm_head && weights.lm_head->data())
                {
                    const auto &sh2 = weights.lm_head->shape();
                    if (sh2.size() == 2)
                    {
                        int rows = sh2[0], cols = sh2[1];
                        const float *p = weights.lm_head->data();
                        double row0 = 0, col0 = 0;
                        for (int c = 0; c < cols; ++c)
                        {
                            double v = p[c];
                            row0 += v * v;
                        }
                        for (int r = 0; r < rows; ++r)
                        {
                            double v = p[r * cols];
                            col0 += v * v;
                        }
                        LOG_DEBUG("[LMHeadDiag] shape=" << rows << "x" << cols << " row0_l2=" << std::sqrt(row0) << " col0_l2=" << std::sqrt(col0));
                    }
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN("output.weight shape unexpected (" << e.what() << "); falling back to tied embeddings");
                weights.lm_head = weights.token_embedding;
            }
        }
        else
        {
            LOG_WARN("output.weight tensor unavailable; falling back to tied token embeddings for LM head");
            weights.lm_head = weights.token_embedding;
        }
        // Per-layer weights
        LOG_INFO("Starting per-layer weight loading for " << config.n_layers << " layers...");
        for (int layer = 0; layer < config.n_layers; ++layer)
        {
            std::string prefix = "blk." + std::to_string(layer) + ".";
            auto attn_norm = loader.loadTensor(prefix + "attn_norm.weight");
            if (!attn_norm)
                throw std::runtime_error("Failed to load attention norm weight");
            const float *attn_ptr = attn_norm->data();
            int attn_sz = attn_norm->size();
            for (int j = 0; j < attn_sz; ++j)
                if (std::isnan(attn_ptr[j]))
                    throw std::runtime_error("NaN in attn_norm.weight");
            weights.attn_norm_weight.push_back(attn_norm);
            if (debugEnv().output_norm.force_unit_all)
            {
                float *m = const_cast<float *>(attn_ptr);
                for (int j = 0; j < attn_sz; ++j)
                    m[j] = 1.0f;
                LOG_WARN("[RMSNormGlobalOverride] Forced attn_norm.weight layer " << layer);
            }
            auto wq = loader.loadTensor(prefix + "attn_q.weight");
            auto wk = loader.loadTensor(prefix + "attn_k.weight");
            auto wv = loader.loadTensor(prefix + "attn_v.weight");
            auto wo = loader.loadTensor(prefix + "attn_output.weight");
            if (!wq || !wk || !wv || !wo)
                throw std::runtime_error("Failed to load attention weights");
            weights.wq.push_back(wq);
            weights.wk.push_back(wk);
            weights.wv.push_back(wv);
            weights.wo.push_back(wo);
            auto ffn_norm = loader.loadTensor(prefix + "ffn_norm.weight");
            if (!ffn_norm)
                throw std::runtime_error("Failed to load FFN norm weight");
            const float *ffn_ptr = ffn_norm->data();
            int ffn_sz = ffn_norm->size();
            for (int j = 0; j < ffn_sz; ++j)
                if (std::isnan(ffn_ptr[j]))
                    throw std::runtime_error("NaN in ffn_norm.weight");
            weights.ffn_norm_weight.push_back(ffn_norm);
            if (debugEnv().output_norm.force_unit_all)
            {
                float *m = const_cast<float *>(ffn_ptr);
                for (int j = 0; j < ffn_sz; ++j)
                    m[j] = 1.0f;
                LOG_WARN("[RMSNormGlobalOverride] Forced ffn_norm.weight layer " << layer);
            }
            auto w_gate = loader.loadTensor(prefix + "ffn_gate.weight");
            auto w_up = loader.loadTensor(prefix + "ffn_up.weight");
            auto w_down = loader.loadTensor(prefix + "ffn_down.weight");
            if (!w_gate || !w_up || !w_down)
                throw std::runtime_error("Failed to load FFN weights");
            int effective_d_ff = config.d_ff;
            if (effective_d_ff == 0 && w_gate && w_gate->shape().size() == 2)
            {
                auto sh = w_gate->shape();
                if (sh[0] == config.d_model)
                {
                    effective_d_ff = sh[1];
                    LOG_WARN("[FFN_DIM_INFER] Inferred d_ff=" << effective_d_ff << " from ffn_gate.weight");
                }
                else if (sh[1] == config.d_model)
                {
                    effective_d_ff = sh[0];
                    LOG_WARN("[FFN_DIM_INFER] Inferred d_ff=" << effective_d_ff << " from transposed ffn_gate.weight");
                }
            }
            if (effective_d_ff > 0)
            {
                enforce_matrix_layout(w_gate, config.d_model, effective_d_ff, prefix + "ffn_gate.weight");
                enforce_matrix_layout(w_up, config.d_model, effective_d_ff, prefix + "ffn_up.weight");
                enforce_matrix_layout(w_down, effective_d_ff, config.d_model, prefix + "ffn_down.weight");
                LOG_DEBUG("[FFN_DIM] Using d_ff=" << effective_d_ff << " layer=" << layer);
            }
            else
            {
                LOG_WARN("[FFN_DIM_INFER] d_ff unresolved (0); skipping layout enforcement layer=" << layer);
            }
            weights.w_gate.push_back(w_gate);
            weights.w_up.push_back(w_up);
            weights.w_down.push_back(w_down);
        }
        LOG_INFO("Successfully loaded all model weights: " << config.n_layers << " layers");
        if (debugEnv().sharding.shard_load_diag)
        {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0)
            {
                LOG_INFO("-- Sharded Tensor Registry (post model load) --");
                size_t total_local_bytes = 0, total_local_elems = 0;
                bool any = false;
                size_t hidden_local = 0, heads_local = 0;
                size_t hidden_global = 0, heads_global = 0;
                TensorFactory::for_each_sharded([&](const ShardSpec &spec, const std::vector<int> &local)
                                                { any=true; size_t elems=1; for(int d:local) elems*=(size_t)d; total_local_elems+=elems; total_local_bytes+=elems*sizeof(float); if(spec.axis==ShardSpec::Axis::Hidden){ hidden_local+=elems; hidden_global=spec.global_dim;} else if(spec.axis==ShardSpec::Axis::Heads){ heads_local+=elems; heads_global=spec.global_dim;} });
                double mb = total_local_bytes / (1024.0 * 1024.0);
                if (!any)
                    LOG_INFO("(no sharded tensors registered)");
                LOG_INFO("Aggregate shard footprint: local_bytes=" << total_local_bytes << " (" << mb << " MB) local_elems=" << total_local_elems);
                if (hidden_local)
                    LOG_INFO("Hidden-axis shards: local_elems=" << hidden_local << " global_dim=" << hidden_global);
                if (heads_local)
                    LOG_INFO("Heads-axis shards: local_elems=" << heads_local << " global_dim=" << heads_global);
                size_t est_global = 0;
                TensorFactory::for_each_sharded([&](const ShardSpec &spec, const std::vector<int> &local)
                                                { size_t elems=1; for(int d:local) elems*=(size_t)d; if(spec.local_dim>0){ double expand=(double)spec.global_dim/(double)spec.local_dim; est_global += (size_t)(elems*expand + 0.5);} else est_global += elems; });
                double est_mb = (est_global * sizeof(float)) / (1024.0 * 1024.0);
                LOG_INFO("Estimated global (logical) elements represented by shards: " << est_global << " (~" << est_mb << " MB as fp32)");
                LOG_INFO("-- End Sharded Tensor Registry --");
            }
        }
        return weights;
    }

    // Legacy loadModelWeights now lives in distributed_transformer_pipeline.cpp; keep shim wrappers
    DistributedTransformerPipeline::ModelWeights loadModelWeights(
        const std::string &model_path,
        const DistributedTransformerPipeline::LayerConfig &config)
    {
        LOG_ERROR("Legacy mpi_transformer_pipeline.cpp loadModelWeights(path) invoked unexpectedly");
        ModelLoader loader;
        if (!loader.loadModel(model_path))
            throw std::runtime_error("Failed legacy load");
        return loadModelWeightsExistingLoaderImpl(loader, config);
    }
    DistributedTransformerPipeline::ModelWeights loadModelWeights(
        ModelLoader &loader,
        const DistributedTransformerPipeline::LayerConfig &config)
    {
        LOG_ERROR("Legacy mpi_transformer_pipeline.cpp loadModelWeights(loader) invoked unexpectedly");
        return loadModelWeightsExistingLoaderImpl(loader, config);
    }

} // namespace llaminar