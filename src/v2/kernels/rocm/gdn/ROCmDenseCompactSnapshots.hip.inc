/**
 * @file ROCmDenseCompactSnapshots.hip.inc
 * @brief Explicit compact correction capture over the unchanged ROCm dense recurrence.
 *
 * The original four-part dense body owns products, reduction order and terminal state.
 * DenseDestination retains its original terminal stores. Explicit retained-only
 * capture removes that unaccepted endpoint at compile time while preserving
 * every immutable prefix. Retention adds disjoint actual operand/result stores. Accepted
 * publication reads immutable checkpoints and at most seven elementwise exact
 * corrections. Neither entrypoint is a serving selector or a model-row replay.
 */
#pragma once
#include "kernels/gdn/GDNDenseCompactSnapshots.h"
namespace
{
/** @brief Execute the original dense body with an explicit compact store policy.
 * @tparam DK Supported original key width.
 * @tparam Axes Immutable persistent/checkpoint matrix axes.
 * @param p Complete borrowed capture identity and explicit required Dense terminal destination.
 * @details Every CTA validates every immutable live count before any store.
 * The existing completion edge, not the metadata header, publishes all CTAs.
 */
template<int DK, llaminar2::GDNRecurrenceStateLayout Axes>
__global__ __launch_bounds__(128, 2)
void rocm_gdn_dense_compact_capture_kernel(llaminar2::gdn::GDNDenseCompactCaptureInput p)
{
    const auto s = p.sources;
    if (threadIdx.x == 0)
    {
        if (p.terminal.kind() != llaminar2::gdn::GDNDenseCompactTerminalKind::DenseDestination ||
            !p.terminal.destination()) __builtin_trap();
        for (int request = 0; request < s.rows.requests(); ++request)
            (void)s.rows.activeRows(request);
    }
    __syncthreads();
    rocm_gdn_recurrent_step_body<DK, GDNRecurrentInputLayout::MergedPostConvolution,
        Axes, ROCmGDNDenseCompactCapture<true>>(
        s.merged, nullptr, nullptr, s.alpha, s.beta_raw, s.A_log, s.dt_bias,
        s.output, p.transaction.initial_state, p.terminal.destination(),
        s.rows.requests(), s.rows.rowsPerRequest(), s.value_heads, s.value_width,
        s.use_qk_l2norm, s.rows.lengthOwner(), 0, nullptr, 0, 0,
        s.key_heads, s.global_value_offset, ROCmGDNDenseCompactCapture<true>{p.transaction, 0,
            unsigned(s.rows.activeRows(blockIdx.x / s.value_heads))});
}

/** @brief Execute original dense forward with complete retained prefixes and no terminal endpoint.
 * @tparam DK Supported original key width.
 * @tparam Axes Immutable persistent/checkpoint matrix axes.
 * @param p Complete immutable retention identity and explicit RetainedPrefixesOnly contract.
 * @details Every CTA validates every immutable live count before any store.
 * The existing completion edge, not the metadata header, publishes all CTAs.
 * A compile-time sink removes terminal pointer arithmetic/stores; cp0, all
 * prefix payloads and repeated/revised acceptance remain identical.
 */
template<int DK, llaminar2::GDNRecurrenceStateLayout Axes>
__global__ __launch_bounds__(128, 2)
void rocm_gdn_retained_prefixes_capture_kernel(llaminar2::gdn::GDNDenseCompactCaptureInput p)
{
    const auto s = p.sources;
    if (threadIdx.x == 0)
    {
        if (p.terminal.kind() != llaminar2::gdn::GDNDenseCompactTerminalKind::RetainedPrefixesOnly ||
            p.terminal.destination() != nullptr) __builtin_trap();
        for (int request = 0; request < s.rows.requests(); ++request)
            (void)s.rows.activeRows(request);
    }
    __syncthreads();
    rocm_gdn_recurrent_step_body<DK, GDNRecurrentInputLayout::MergedPostConvolution,
        Axes, ROCmGDNDenseCompactCapture<false>>(
        s.merged, nullptr, nullptr, s.alpha, s.beta_raw, s.A_log, s.dt_bias,
        s.output, p.transaction.initial_state, nullptr,
        s.rows.requests(), s.rows.rowsPerRequest(), s.value_heads, s.value_width,
        s.use_qk_l2norm, s.rows.lengthOwner(), 0, nullptr, 0, 0,
        s.key_heads, s.global_value_offset, ROCmGDNDenseCompactCapture<false>{p.transaction, 0,
            unsigned(s.rows.activeRows(blockIdx.x / s.value_heads))});
}

/** @brief Read full-width retained protocol without enum narrowing or payload conversion. */
__device__ llaminar2::gdn::GDNDenseCompactPublication rocm_dense_compact_publication(const float *head)
{
    using G = llaminar2::gdn::GDNDenseCompactGeometry;
    const float *w = head;
    llaminar2::gdn::GDNDenseCompactPublication publication{__float_as_uint(w[G::format_word]), __float_as_uint(w[G::published_rows_word]), __float_as_uint(w[G::planned_rows_word]),
        std::uint64_t(__float_as_uint(w[G::generation_low_word])) | (std::uint64_t(__float_as_uint(w[G::generation_high_word])) << 32),
        __float_as_uint(w[G::layout_word]), __float_as_uint(w[G::arithmetic_word])};
    // DFC1 padding is never payload. Read the extra identity only for DFS1;
    // complete-plan validation below rejects every unknown format.
    if (publication.format == std::uint32_t(llaminar2::gdn::GDNDenseCompactEncoding::SplitRetentionV1))
    {
        publication.retention_mode = __float_as_uint(w[G::retention_mode_word]);
        publication.retention_program = __float_as_uint(w[G::retention_program_word]);
        publication.compact_min_live_rows = __float_as_uint(w[G::compact_min_live_rows_word]);
    }
    return publication;
}

/** @brief Match retained gfx906 separate decay and delta-first fused update.
 * @param state Prior original dense element.
 * @param decay Actual retained rounded gate value.
 * @param delta Actual retained original forward correction.
 * @param key Actual retained normalized K element.
 * @return One exact original update without dot/query replay.
 * @details Original DK64 v_fma and DK128 v_fmac use delta,K,decayed ordering.
 * Explicit fmaf preserves fusion while contract(off) prevents decay contraction.
 */
__device__ __forceinline__ float rocm_dense_compact_apply(float state, float decay, float delta, float key)
{
#pragma clang fp contract(off)
    const float decayed = state * decay;
    return fmaf(delta, key, decayed);
}

/** @brief Fixed native scheduling width; not a runtime policy or second capacity ledger. */
constexpr std::size_t kROCmDenseCompactPublisherTileElements = 4096;

/** @brief Restore accepted dense bytes from immutable retained corrections only.
 * @param p Ordered captured transaction and one accepted count per request.
 * @details Before any destination write, each CTA's lanes partition all request/
 * head protocols and count ranges, including the last head. Every CTA sees the
 * same immutable plan. No barrier crosses CTAs and no status/seal is invented.
 * Independent grid.y tiles own disjoint matrix elements after the unchanged
 * all-plan validation. Each element follows the original rounded decay and fused update
 * operand order; no dot, query contraction, normalization or model row runs.
 */
__global__ __launch_bounds__(256)
void rocm_gdn_dense_compact_accepted_prefix_kernel(llaminar2::gdn::GDNDenseCompactAcceptedPrefixInput p)
{
    const auto t = p.transaction;
    const auto g = t.geometry;
    // The checked geometry bounds requests*heads. Use size_t for the
    // increment so a final owner near INT_MAX cannot overflow a signed index.
    // Every CTA covers every immutable owner; the same barrier retires ALL
    // validating lanes before any destination byte can be written.
    const auto owners = std::size_t(g.requests()) * std::size_t(g.heads());
    for (std::size_t owner = threadIdx.x; owner < owners; owner += blockDim.x)
    {
        const int request = int(owner / std::size_t(g.heads()));
        std::size_t offset = 0;
        if (!p.selection.valueOffset(request, offset)) __builtin_trap();
        const int observed = p.selection.data()[offset];
        const float *request_source = t.storage +
            std::size_t(request) * g.heads() * g.headStrideFloats();
        const auto request_publication = rocm_dense_compact_publication(request_source);
        const float *source = t.storage + owner * g.headStrideFloats();
        const auto publication = rocm_dense_compact_publication(source);
        llaminar2::gdn::GDNDenseAcceptedSelectionResult action;
        if (publication.published_rows != request_publication.published_rows ||
            publication.planned_rows != request_publication.planned_rows ||
            !g.validPublication(publication, t.generation, t.layout, t.arithmetic) ||
            !p.selection.resolveObserved(request, int(publication.published_rows), observed, action)) __builtin_trap();
    }
    __syncthreads();
    const int request = blockIdx.x / g.heads();
    const float *source = t.storage + std::size_t(blockIdx.x) * g.headStrideFloats();
    // Complete immutable plan authentication precedes this CTA-uniform skip.
    // Negative row / terminal-length0 preserve the destination; explicit Counts0
    // remains Restore(cp0), including after an earlier target mutation.
    std::size_t offset = 0;
    llaminar2::gdn::GDNDenseAcceptedSelectionResult action;
    if (!p.selection.valueOffset(request, offset) ||
        !p.selection.resolveObserved(request, int(__float_as_uint(source[g.published_rows_word])),
            p.selection.data()[offset], action)) __builtin_trap();
    if (action.action() == llaminar2::gdn::GDNDenseAcceptedSelectionAction::NoOp) return;
    llaminar2::gdn::GDNDenseCompactSelection selection{};
    if (!g.selectForLiveRows(int(__float_as_uint(source[g.published_rows_word])),
            int(action.acceptedCount()), selection)) __builtin_trap();
    const auto checkpoint = selection.checkpoint;
    const auto first = selection.first_record;
    const auto last = first + selection.update_count;
    const float *base = source + g.checkpointOffsetFloats() +
        std::size_t(checkpoint) * g.checkpointStrideFloats();
    float *destination = p.destination_state + std::size_t(blockIdx.x) * g.matrixFloats();
    // A NoOp returned above before either destination can be touched. Each
    // request-zero head/tile owns the same disjoint interval in its mirror.
    float *mirror = request == 0 ? p.request_zero_mirror.destination() : nullptr;
    if (mirror) mirror += std::size_t(blockIdx.x) * g.matrixFloats();
    llaminar2::gdn::GDNDenseCompactMatrixTileSpan tile{};
    if (!g.matrixTileFor(blockIdx.y, kROCmDenseCompactPublisherTileElements, tile)) __builtin_trap();
    const auto tile_end = tile.first_element + tile.live_elements;
    for (std::size_t index = tile.first_element + threadIdx.x; index < tile_end; index += blockDim.x)
    {
        const int key = t.layout == llaminar2::GDNRecurrenceStateLayout::KeyMajorValueContiguous
            ? int(index / g.valueWidth()) : int(index % g.keyWidth());
        const int value = t.layout == llaminar2::GDNRecurrenceStateLayout::KeyMajorValueContiguous
            ? int(index % g.valueWidth()) : int(index / g.keyWidth());
        float state = base[index];
        for (unsigned row = first; row < last; ++row)
        {
            const float *record = source + g.recordsOffsetFloats() +
                std::size_t(row) * g.recordStrideFloats();
            state = rocm_dense_compact_apply(state, record[g.recordDecayOffsetFloats()],
                record[g.recordCorrectionOffsetFloats() + value], record[g.recordKeyOffsetFloats() + key]);
        }
        destination[index] = state;
        if (mirror) mirror[index] = state;
    }
}
} // namespace

namespace llaminar2::gdn
{
/** @brief Authenticate an exact idle ROCm stream at the cold owner boundary.
 * @param device Canonical kernel state owner ordinal.
 * @param stream Explicit stream before any graph recording begins.
 * @return Stream is nondefault, belongs to device, and is not capturing; no work is enqueued. */
bool prepareROCmGDNRetainedCaptureStream(int device, void *stream)
{
    if (device < 0 || reinterpret_cast<std::uintptr_t>(stream) <= 2) return false;
    const auto native = static_cast<hipStream_t>(stream);
    int observed = -1;
    hipStreamCaptureStatus status{};
    return hipStreamGetDevice(native, &observed) == hipSuccess && observed == device &&
        hipStreamIsCapturing(native, &status) == hipSuccess && status == hipStreamCaptureStatusNone;
}

/** @brief Admit and enqueue one original-arithmetic capture on the exact caller stream. */
bool launchROCmDenseCompactRecurrence(const GDNDenseCompactCaptureInput &p, void *stream)
{
    if (reinterpret_cast<std::uintptr_t>(stream) <= 2 || !validROCmDenseCompactCaptureStorage(p))
        return false;
    const auto &g = p.transaction.geometry;
    const dim3 grid(g.requests() * g.heads(), 1 + (g.valueWidth() - 1) / kGdnRecurrentColumnsPerBlock);
    const int tail = g.keyWidth() / kGdnRecurrentRowSplit > 22
        ? g.keyWidth() / kGdnRecurrentRowSplit - 22 : 0;
    const std::size_t shared_bytes = std::size_t(2 * g.keyWidth() + 4 +
        2 * kGdnRecurrentThreads + tail * kGdnRecurrentThreads) * sizeof(float);
    const auto native_stream = static_cast<hipStream_t>(stream);
#define LLAMINAR_DENSE_COMPACT_CAPTURE(DK, AXES) \
    do { \
        switch (p.terminal.kind()) { \
        case GDNDenseCompactTerminalKind::DenseDestination: \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(rocm_gdn_dense_compact_capture_kernel<DK, AXES>), \
                grid, dim3(kGdnRecurrentThreads), shared_bytes, native_stream, p); \
            break; \
        case GDNDenseCompactTerminalKind::RetainedPrefixesOnly: \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(rocm_gdn_retained_prefixes_capture_kernel<DK, AXES>), \
                grid, dim3(kGdnRecurrentThreads), shared_bytes, native_stream, p); \
            break; \
        default: return false; \
        } \
    } while (false)
    if (p.transaction.layout == GDNRecurrenceStateLayout::KeyMajorValueContiguous)
    {
        if (g.keyWidth() == 64)
            LLAMINAR_DENSE_COMPACT_CAPTURE(64, GDNRecurrenceStateLayout::KeyMajorValueContiguous);
        else
            LLAMINAR_DENSE_COMPACT_CAPTURE(128, GDNRecurrenceStateLayout::KeyMajorValueContiguous);
    }
    else
    {
        if (g.keyWidth() == 64)
            LLAMINAR_DENSE_COMPACT_CAPTURE(64, GDNRecurrenceStateLayout::ValueMajorKeyContiguous);
        else
            LLAMINAR_DENSE_COMPACT_CAPTURE(128, GDNRecurrenceStateLayout::ValueMajorKeyContiguous);
    }
#undef LLAMINAR_DENSE_COMPACT_CAPTURE
    return hipGetLastError() == hipSuccess;
}

/** @brief Enqueue exact accepted reconstruction; native completion remains caller-owned. */
bool launchROCmDenseCompactAcceptedPrefix(const GDNDenseCompactAcceptedPrefixInput &p, void *stream)
{
    if (reinterpret_cast<std::uintptr_t>(stream) <= 2 || !validROCmDenseCompactAcceptedPrefixStorage(p))
        return false;
    const auto &g = p.transaction.geometry;
    const auto tiles = g.matrixTileCount(kROCmDenseCompactPublisherTileElements);
    if (!tiles || tiles > 65535) return false;
    hipLaunchKernelGGL(rocm_gdn_dense_compact_accepted_prefix_kernel,
        dim3(g.requests() * g.heads(), static_cast<unsigned>(tiles)), dim3(256), 0,
        static_cast<hipStream_t>(stream), p);
    return hipGetLastError() == hipSuccess;
}
} // namespace llaminar2::gdn
