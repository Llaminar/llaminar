/**
 * @file GPUImmutableFixedHeadPipelineKernels.inl
 * @brief Shared CUDA/HIP compact fixed-range service and PLE consumer math.
 *
 * The original Hash authors every row ID. An addressed owner validates its
 * own range before loading bytes. Each receiving PLE consumer decodes exact
 * native contributions into the original token and global-head order.
 * Included in the same backend namespace as the standalone fixed-head math.
 */

/** @brief Compact capture-time head vocabulary projection with direct parameter accesses. */
struct PipelineHeadBounds {
    std::uint64_t begin[kNgramHeads], size[kNgramHeads];
};

/** @return Existing device count or the declared exact scalar/full invocation. */
__device__ __forceinline__ int pipelineRows(FixedHeadPipelineRows rows)
{
    return rows.live ? *rows.live : rows.exact;
}

/** @brief Keep the original graph controller's first error and revoke health. */
__device__ __forceinline__ void failPipeline(FixedHeadPipelineError control)
{
    atomicCAS(control.error, 0, failure);
    atomicExch(control.healthy, 0);
}

/** @brief Check exact head vocabulary before any storage address is evaluated. */
__device__ __forceinline__ bool validPipelineId(const PipelineHeadBounds &bounds, int head, std::int64_t id)
{
    bool owned = false;
#pragma unroll
    for (int h = 0; h < kNgramHeads; ++h)
        if (head == h)
            owned = id >= 0 && static_cast<std::uint64_t>(id) >= bounds.begin[h] &&
                static_cast<std::uint64_t>(id) - bounds.begin[h] < bounds.size[h];
    return owned;
}

/** @brief Prove the current graph width before permitting native request payload. */
__global__ __launch_bounds__(1) void preparePipelineConsumer(Frame args)
{
    *args.transport = 0;
    const auto live = originalRows(args);
    if (live != args.capacity) fail(args, failure);
    if (!*args.error) *args.transport = live;
}

/** @brief Count one real range invocation and any committed original native rows. */
__device__ __forceinline__ void observePipelineRange(FixedHeadPipelineError control, int rows,
    int heads, std::size_t row_bytes)
{
    using Metric = FixedHeadPipelineTerminalMetric;
    const auto ids = static_cast<std::uint64_t>(rows) * heads;
    if (!control.counters.add(Metric::Calls, 1) || !control.counters.add(Metric::LiveIds, ids) ||
        !control.counters.add(Metric::NativeBytes, ids * row_bytes)) failPipeline(control);
}

/** @brief Publish the single cohort header after every completed addressed response. */
__global__ __launch_bounds__(1) void sealPipelineConsumer(Frame args, FixedHeadPipelineError control,
    int heads, std::size_t row_bytes)
{
    const auto committed = !*control.error && *args.transport == args.capacity ? args.capacity : 0;
    observePipelineRange(control, committed, heads, row_bytes);
    *args.transport = 1;
    args.header->originalLiveRows = originalRows(args);
    args.header->firstError = *control.error;
}

/** @brief Retire cohort status once before any native data gather. */
__global__ __launch_bounds__(1) void validatePipelineConsensus(Frame args, FixedHeadPipelineError control)
{
    validateHeaderConsensus(args);
    if (*control.error && !control.counters.add(FixedHeadPipelineTerminalMetric::Failures, 1))
        failPipeline(control);
}

/** @brief Verify the producer receipt after the reply has drained on this exact stream. */
__global__ __launch_bounds__(1) void validatePipelineResponse(FixedHeadPipelineRows request,
    const std::int32_t *received, FixedHeadPipelineError control)
{
    if (*received != pipelineRows(request)) failPipeline(control);
}

/** @brief Send only the actual original row IDs needed by this range owner. */
__global__ __launch_bounds__(256,2) void projectPipelineIds(FixedHeadPipelineRequest args)
{
    const auto rows = pipelineRows(args.rows);
    if (rows != args.rows.exact) return;
    const auto count = static_cast<std::uint64_t>(rows) * args.heads.count;
    for (auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
        args.compact[index] = args.original[index / args.heads.count * kNgramHeads +
            args.heads.first + index % args.heads.count];
}

/** @brief Copy only validated original rows in their native format and original byte order. */
__global__ __launch_bounds__(256,2) void packPipelineRange(FixedHeadPipelinePack args, FixedHeadPipelineHeads owned,
    const std::byte *source, std::uint64_t first_row, std::size_t row_bytes, PipelineHeadBounds bounds)
{
    const auto rows = pipelineRows(args.rows);
    if (*args.control.error) return;
    if (rows != args.rows.exact) {
        if (!rows) return; // The producer revoked this request before publishing its IDs.
        if (!blockIdx.x && !threadIdx.x) failPipeline(args.control);
        return;
    }
    const auto token = blockIdx.x / owned.count;
    const auto head = blockIdx.x % owned.count;
    const auto id = args.ids[static_cast<std::uint64_t>(token) * args.id_stride + args.id_offset + head];
    if (!validPipelineId(bounds, owned.first + static_cast<int>(head), id)) {
        if (!threadIdx.x) failPipeline(args.control);
        return;
    }
    const auto *native = source + (static_cast<std::uint64_t>(id) - first_row) * row_bytes;
    auto *output = args.destination +
        (static_cast<std::uint64_t>(token) * args.destination_heads + args.destination_offset + head) * row_bytes;
    for (auto byte = static_cast<std::size_t>(threadIdx.x); byte < row_bytes; byte += blockDim.x)
        output[byte] = native[byte];
}

/** @brief Preserve peer progress after a range error by returning an empty completed response. */
__global__ __launch_bounds__(1) void sealPipelineResponse(std::int32_t *rows, FixedHeadPipelineError control,
    int heads, std::size_t row_bytes)
{
    if (*control.error) *rows = 0;
    observePipelineRange(control, *rows, heads, row_bytes);
    if (*control.error) {
        *rows = 0;
        if (!control.counters.add(FixedHeadPipelineTerminalMetric::Failures, 1)) failPipeline(control);
    }
}

/** @brief Merge only useful native rows from the exact completed response. */
__global__ __launch_bounds__(256,2) void mergePipelineRange(FixedHeadPipelineMerge args, std::size_t row_bytes)
{
    const auto rows = pipelineRows(args.rows);
    if (*args.control.error || rows != args.rows.exact) return;
    const auto row = static_cast<std::uint64_t>(blockIdx.x);
    auto *output = args.destination +
        (row / args.heads * args.destination_heads + args.destination_offset + row % args.heads) * row_bytes;
    const auto *native = args.compact + row * row_bytes;
    for (auto byte = static_cast<std::size_t>(threadIdx.x); byte < row_bytes; byte += blockDim.x)
        output[byte] = native[byte];
}

/** @brief Publish exact useful native prefix for an original variable-width PLE contributor. */
__global__ __launch_bounds__(1) void publishPipelineExtent(FixedHeadPipelineRows rows, std::size_t stride,
    std::uint64_t *extent)
{
    const auto live = pipelineRows(rows);
    *extent = live == rows.exact ? static_cast<std::uint64_t>(live) * stride : 0;
}

/** @brief Decode each global head from its actual original contributor, without capacity padding. */
template<TensorType Format>
__global__ __launch_bounds__(128,2) void decodePipelinePackets(FixedHeadPipelineDecode args,
    std::size_t row_bytes, int width, NativeEmbeddingIQTables tables)
{
    const int rows = pipelineRows(args.rows);
    if (*args.control.error || rows != args.rows.exact) return;
    // Every block proves the same small immutable participant roster before
    // reading any received bytes, including zero rows. Constant accesses keep
    // the parameter array out of private addressable storage on both backends.
#pragma unroll
    for (int p = 0; p < kNgramHeads; ++p) {
        if (p < args.participants && args.packets[p].receipt_bytes &&
            *args.packets[p].receipt_bytes != static_cast<std::uint64_t>(rows) * args.packets[p].heads.count * row_bytes) {
            if (!threadIdx.x) failPipeline(args.control);
            return;
        }
    }
    {
        const auto row = static_cast<std::uint64_t>(blockIdx.x);
        const auto token = row / kNgramHeads;
        const int head = row % kNgramHeads;
        const std::byte *native = nullptr;
#pragma unroll
        for (int p = 0; p < kNgramHeads; ++p) {
            const auto packet = args.packets[p];
            if (p < args.participants && head >= packet.heads.first && head - packet.heads.first < packet.heads.count)
                native = packet.data + (token * packet.heads.count + head - packet.heads.first) * row_bytes;
        }
        if (!native) {
            if (!threadIdx.x) failPipeline(args.control);
            return;
        }
        for (unsigned column = threadIdx.x; column < static_cast<unsigned>(width); column += blockDim.x)
            args.output[row * width + column] = lookup_detail::decode<Format>(native, column, tables);
    }
}

/** @brief Immutable per-original-range backend dispatch, reused by every retained forward. */
template<TensorType Format>
class PipelineKernels final : public IImmutableFixedHeadPipelineKernels {
public:
    PipelineKernels(ImmutableResidentTableRangeView table, const NgramHashPolicy &policy,
        FixedHeadPipelineHeads own, NativeEmbeddingIQTables tables)
        : table_(table), own_(own), tables_(tables)
    {
        for (int head = 0; head < kNgramHeads; ++head) {
            bounds_.begin[head] = policy.offsets[head];
            bounds_.size[head] = policy.moduli[head];
        }
    }

    void prepareConsumer(FixedHeadPipelineConsensus args, void *stream) const override
    {
        preparePipelineConsumer<<<1,1,0,exactStream(stream)>>>(consensus(args));
        check();
    }

    void sealConsumer(FixedHeadPipelineConsensus args, void *stream) const override
    {
        sealPipelineConsumer<<<1,1,0,exactStream(stream)>>>(consensus(args), args.control, own_.count, table_.rowBytes());
        check();
    }

    void validateConsensus(FixedHeadPipelineConsensus args, void *stream) const override
    {
        validatePipelineConsensus<<<1,1,0,exactStream(stream)>>>(consensus(args), args.control);
        check();
    }

    void projectRequest(FixedHeadPipelineRequest args, void *stream) const override
    {
        const auto native = exactStream(stream);
        if (!args.rows.exact) return;
        projectPipelineIds<<<idGrid(args.rows.exact, args.heads.count),256,0,native>>>(args);
        check();
    }

    void pack(FixedHeadPipelinePack args, void *stream) const override
    {
        const auto native = exactStream(stream);
        if (!args.rows.exact) return;
        packPipelineRange<<<rowGrid(args.rows.exact, own_.count),256,0,native>>>(args, own_,
            table_.nativeData(), table_.globalRowBegin(), table_.rowBytes(), bounds_);
        check();
    }

    void sealResponse(std::int32_t *rows, FixedHeadPipelineError control, void *stream) const override
    {
        sealPipelineResponse<<<1,1,0,exactStream(stream)>>>(rows, control, own_.count, table_.rowBytes());
        check();
    }

    void validateResponse(FixedHeadPipelineRows request, const std::int32_t *received,
        FixedHeadPipelineError control, void *stream) const override
    {
        validatePipelineResponse<<<1,1,0,exactStream(stream)>>>(request, received, control);
        check();
    }

    void merge(FixedHeadPipelineMerge args, void *stream) const override
    {
        const auto native = exactStream(stream);
        if (!args.rows.exact) return;
        mergePipelineRange<<<rowGrid(args.rows.exact, args.heads),256,0,native>>>(args, table_.rowBytes());
        check();
    }

    void publishExtent(FixedHeadPipelineRows rows, int heads, std::uint64_t *extent, void *stream) const override
    {
        publishPipelineExtent<<<1,1,0,exactStream(stream)>>>(rows, table_.rowBytes() * heads, extent);
        check();
    }

    void decode(FixedHeadPipelineDecode args, void *stream) const override
    {
        const auto native = exactStream(stream);
        if (!args.rows.exact) return;
        decodePipelinePackets<Format><<<rowGrid(args.rows.exact, kNgramHeads),128,0,native>>>(
            args, table_.rowBytes(), table_.rowElements(), tables_);
        check();
    }

private:
    /** @return Exact non-default native recording stream. */
    static QFN_HEAD_STREAM exactStream(void *stream)
    {
        if (reinterpret_cast<std::uintptr_t>(stream) <= 2)
            throw std::invalid_argument("Fixed pipeline lookup requires its exact native stream");
        return static_cast<QFN_HEAD_STREAM>(stream);
    }

    /** @return Small existing fixed-head header transition over the actual consumer roster. */
    static Frame consensus(FixedHeadPipelineConsensus args)
    {
        Frame result{};
        result.capacity = args.rows.exact;
        result.originalCount = args.rows.live;
        result.participants = args.participants;
        result.header = args.local;
        result.headers = args.gathered;
        result.transport = args.transport;
        result.healthy = args.control.healthy;
        result.error = args.control.error;
        return result;
    }

    /** @return One workgroup per exact active row of the retained head interval. */
    static unsigned rowGrid(int rows, int heads)
    {
        const auto grid = static_cast<std::uint64_t>(rows) * heads;
        if (rows <= 0 || heads <= 0 || grid > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Fixed pipeline lookup requires a representable positive exact launch");
        return static_cast<unsigned>(grid);
    }
    /** @return Exact 256-lane coverage of the compact INT64 coordinate prefix. */
    static unsigned idGrid(int rows, int heads)
    { return (rowGrid(rows, heads) + 255u) / 256u; }

    const ImmutableResidentTableRangeView table_;
    const FixedHeadPipelineHeads own_;
    const NativeEmbeddingIQTables tables_;
    PipelineHeadBounds bounds_{};
};
