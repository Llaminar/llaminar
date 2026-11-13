/**
 * @file CUDADenseCompactSnapshots.cuh
 * @brief Explicit compact correction capture over the unchanged CUDA dense recurrence.
 *
 * The original dense body owns products, reduction order and terminal state.
 * DenseDestination retains its original terminal stores. Explicit retained-only
 * capture removes that unaccepted endpoint at compile time while preserving
 * every immutable prefix. Retention adds disjoint actual operand/result stores. Accepted
 * publication reads immutable checkpoints and at most seven elementwise exact
 * corrections. Neither entrypoint is a serving selector or a model-row replay.
 */
#pragma once
#include "kernels/gdn/GDNDenseCompactSnapshots.h"
#include "kernels/gdn/GDNCUDADenseCapturePlan.h"
#include "backends/ComputeBackend.h"
#include "backends/HardwareInventory.h"
#include "backends/DeviceUUID.h"
namespace
{
/** @brief Immutable publication work partition; allocation and arithmetic stay unchanged. */
constexpr std::size_t kCudaDenseCompactPublicationTileElements = 4096;

/** @brief Execute the original dense body with an explicit compact store policy.
 * @tparam DK Supported original key width.
 * @tparam Axes Immutable persistent/checkpoint matrix axes.
 * @param p Complete borrowed capture identity and explicit required Dense terminal destination.
 * @details Every CTA validates every immutable live count before any store.
 * The existing completion edge, not the metadata header, publishes all CTAs.
 * This partition-store specialization uses the min1 register budget on all
 * admitted geometries; it does not request two-block residency.
 */
template<int DK, llaminar2::GDNRecurrenceStateLayout Axes>
__global__ __launch_bounds__(256, 1)
void cuda_gdn_dense_compact_capture_kernel(llaminar2::gdn::GDNDenseCompactCaptureInput p)
{
    const auto s = p.sources;
    if (threadIdx.x == 0)
    {
        if (p.terminal.kind() != llaminar2::gdn::GDNDenseCompactTerminalKind::DenseDestination ||
            !p.terminal.destination()) asm volatile("trap;");
        for (int request = 0; request < s.rows.requests(); ++request)
            (void)s.rows.activeRows(request);
    }
    __syncthreads();
    cuda_gdn_recurrent_step_body<DK, true, CUDAGDNInputLayout::MergedPostConvolution,
        Axes, CUDAGDNDenseCompactCapture<true, false>>(
        s.merged, nullptr, nullptr, s.alpha, s.beta_raw, s.A_log, s.dt_bias,
        s.output, p.transaction.initial_state, p.terminal.destination(),
        s.rows.requests(), s.rows.rowsPerRequest(), s.value_heads, s.value_width,
        s.use_qk_l2norm, s.rows.lengthOwner(), 0, nullptr, 0, 0,
        s.key_heads, s.global_value_offset, CUDAGDNDenseCompactCapture<true, false>{p.transaction,
            s.rows.activeRows(blockIdx.x / s.value_heads)});
}

/** @brief Execute original dense forward with complete retained prefixes and no terminal endpoint.
 * @tparam DK Supported original key width.
 * @tparam Axes Immutable persistent/checkpoint matrix axes.
 * @param p Complete immutable retention identity and explicit RetainedPrefixesOnly contract.
 * @details Every CTA validates every immutable live count before any store.
 * The existing completion edge, not the metadata header, publishes all CTAs.
 * A compile-time sink removes terminal pointer arithmetic/stores; cp0, all
 * prefix payloads and repeated/revised acceptance remain identical.
 * The partition-store min1 specialization does not request two-block residency.
 */
template<int DK, llaminar2::GDNRecurrenceStateLayout Axes>
__global__ __launch_bounds__(256, 1)
void cuda_gdn_retained_prefixes_capture_kernel(llaminar2::gdn::GDNDenseCompactCaptureInput p)
{
    const auto s = p.sources;
    if (threadIdx.x == 0)
    {
        if (p.terminal.kind() != llaminar2::gdn::GDNDenseCompactTerminalKind::RetainedPrefixesOnly ||
            p.terminal.destination() != nullptr) asm volatile("trap;");
        for (int request = 0; request < s.rows.requests(); ++request)
            (void)s.rows.activeRows(request);
    }
    __syncthreads();
    cuda_gdn_recurrent_step_body<DK, true, CUDAGDNInputLayout::MergedPostConvolution,
        Axes, CUDAGDNDenseCompactCapture<false, false>>(
        s.merged, nullptr, nullptr, s.alpha, s.beta_raw, s.A_log, s.dt_bias,
        s.output, p.transaction.initial_state, nullptr,
        s.rows.requests(), s.rows.rowsPerRequest(), s.value_heads, s.value_width,
        s.use_qk_l2norm, s.rows.lengthOwner(), 0, nullptr, 0, 0,
        s.key_heads, s.global_value_offset, CUDAGDNDenseCompactCapture<false, false>{p.transaction,
            s.rows.activeRows(blockIdx.x / s.value_heads)});
}

/** @brief Execute the grid-dense compiled word-store/min2 program on its exact cold plan.
 * @tparam DK Supported original key width.
 * @tparam Axes Immutable persistent/checkpoint matrix axes.
 * @param p Complete borrowed capture identity and explicit required Dense terminal destination.
 * @details Every CTA validates every immutable live count before any store.
 * The existing completion edge, not the metadata header, publishes all CTAs.
 * Only DK128 KeyMajor is admitted. The min2 compiler bound targets two-block
 * residency; actual resource allocation and achieved occupancy require evidence.
 */
template<int DK, llaminar2::GDNRecurrenceStateLayout Axes>
__global__ __launch_bounds__(256, 2)
void cuda_gdn_dense_compact_packed_capture_kernel(llaminar2::gdn::GDNDenseCompactCaptureInput p)
{
    static_assert(DK == 128 && Axes == llaminar2::GDNRecurrenceStateLayout::KeyMajorValueContiguous,
        "Packed checkpoint scheduling is qualified only for DK128 KeyMajor");
    const auto s = p.sources;
    if (threadIdx.x == 0)
    {
        if (p.terminal.kind() != llaminar2::gdn::GDNDenseCompactTerminalKind::DenseDestination ||
            !p.terminal.destination()) asm volatile("trap;");
        for (int request = 0; request < s.rows.requests(); ++request)
            (void)s.rows.activeRows(request);
    }
    __syncthreads();
    cuda_gdn_recurrent_step_body<DK, true, CUDAGDNInputLayout::MergedPostConvolution,
        Axes, CUDAGDNDenseCompactCapture<true, true>>(
        s.merged, nullptr, nullptr, s.alpha, s.beta_raw, s.A_log, s.dt_bias,
        s.output, p.transaction.initial_state, p.terminal.destination(),
        s.rows.requests(), s.rows.rowsPerRequest(), s.value_heads, s.value_width,
        s.use_qk_l2norm, s.rows.lengthOwner(), 0, nullptr, 0, 0,
        s.key_heads, s.global_value_offset, CUDAGDNDenseCompactCapture<true, true>{p.transaction,
            s.rows.activeRows(blockIdx.x / s.value_heads)});
}

/** @brief Execute retained-only grid-dense word-store/min2 capture without a terminal endpoint.
 * @tparam DK Supported original key width.
 * @tparam Axes Immutable persistent/checkpoint matrix axes.
 * @param p Complete immutable retention identity and explicit RetainedPrefixesOnly contract.
 * @details Every CTA validates every immutable live count before any store.
 * The existing completion edge, not the metadata header, publishes all CTAs.
 * A compile-time sink removes terminal pointer arithmetic/stores; cp0, all
 * prefix payloads and repeated/revised acceptance remain identical.
 * Only DK128 KeyMajor is admitted; min2 is a compiler residency target, not
 * a claim of achieved runtime occupancy.
 */
template<int DK, llaminar2::GDNRecurrenceStateLayout Axes>
__global__ __launch_bounds__(256, 2)
void cuda_gdn_retained_prefixes_packed_capture_kernel(llaminar2::gdn::GDNDenseCompactCaptureInput p)
{
    static_assert(DK == 128 && Axes == llaminar2::GDNRecurrenceStateLayout::KeyMajorValueContiguous,
        "Packed checkpoint scheduling is qualified only for DK128 KeyMajor");
    const auto s = p.sources;
    if (threadIdx.x == 0)
    {
        if (p.terminal.kind() != llaminar2::gdn::GDNDenseCompactTerminalKind::RetainedPrefixesOnly ||
            p.terminal.destination() != nullptr) asm volatile("trap;");
        for (int request = 0; request < s.rows.requests(); ++request)
            (void)s.rows.activeRows(request);
    }
    __syncthreads();
    cuda_gdn_recurrent_step_body<DK, true, CUDAGDNInputLayout::MergedPostConvolution,
        Axes, CUDAGDNDenseCompactCapture<false, true>>(
        s.merged, nullptr, nullptr, s.alpha, s.beta_raw, s.A_log, s.dt_bias,
        s.output, p.transaction.initial_state, nullptr,
        s.rows.requests(), s.rows.rowsPerRequest(), s.value_heads, s.value_width,
        s.use_qk_l2norm, s.rows.lengthOwner(), 0, nullptr, 0, 0,
        s.key_heads, s.global_value_offset, CUDAGDNDenseCompactCapture<false, true>{p.transaction,
            s.rows.activeRows(blockIdx.x / s.value_heads)});
}

/** @brief Read full-width retained protocol without enum narrowing or payload conversion. */
__device__ llaminar2::gdn::GDNDenseCompactPublication cuda_dense_compact_publication(const float *head)
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

/** @brief Restore accepted dense bytes from immutable retained corrections only.
 * @param p Ordered captured transaction and one accepted count per request.
 * @details Before any destination write, each CTA's lanes partition all request/
 * head protocols and count ranges, including the last head. Every CTA sees the
 * same immutable plan. No barrier crosses CTAs and no status/seal is invented.
 * Each element then follows the original rounded decay followed by fused update
 * operand order; no dot, query contraction, normalization or model row runs.
 * grid.y gives each CTA one exclusive bounded matrix interval. Every tile still
 * authenticates the complete immutable transaction before touching destination bytes.
 */
__global__ __launch_bounds__(256)
void cuda_gdn_dense_compact_accepted_prefix_kernel(llaminar2::gdn::GDNDenseCompactAcceptedPrefixInput p)
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
        if (!p.selection.valueOffset(request, offset)) asm volatile("trap;");
        const int observed = p.selection.data()[offset];
        const float *request_source = t.storage +
            std::size_t(request) * g.heads() * g.headStrideFloats();
        const auto request_publication = cuda_dense_compact_publication(request_source);
        const float *source = t.storage + owner * g.headStrideFloats();
        const auto publication = cuda_dense_compact_publication(source);
        llaminar2::gdn::GDNDenseAcceptedSelectionResult action;
        if (publication.published_rows != request_publication.published_rows ||
            publication.planned_rows != request_publication.planned_rows ||
            !g.validPublication(publication, t.generation, t.layout, t.arithmetic) ||
            !p.selection.resolveObserved(request, int(publication.published_rows), observed, action)) asm volatile("trap;");
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
            p.selection.data()[offset], action)) asm volatile("trap;");
    if (action.action() == llaminar2::gdn::GDNDenseAcceptedSelectionAction::NoOp) return;
    llaminar2::gdn::GDNDenseCompactSelection selection{};
    if (!g.selectForLiveRows(int(__float_as_uint(source[g.published_rows_word])),
            int(action.acceptedCount()), selection)) asm volatile("trap;");
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
    if (!g.matrixTileFor(blockIdx.y, kCudaDenseCompactPublicationTileElements, tile))
        asm volatile("trap;");
    const std::size_t end = tile.first_element + tile.live_elements;
    // A packed endpoint preserves each element's record order, while sharing
    // the axis-invariant operand and address calculation across four words.
    // Public pointers are only float-aligned: retain the scalar implementation
    // for unaligned endpoints and widths whose packs cross an axis boundary.
    const bool key_major = t.layout == llaminar2::GDNRecurrenceStateLayout::KeyMajorValueContiguous;
    const bool packed_axis = (key_major ? g.valueWidth() : g.keyWidth()) % 4 == 0;
    const bool packed_endpoints =
        (reinterpret_cast<std::uintptr_t>(base) % alignof(float4)) == 0 &&
        (reinterpret_cast<std::uintptr_t>(destination) % alignof(float4)) == 0;
    if (packed_axis && packed_endpoints)
    {
        for (std::size_t index = tile.first_element + 4 * threadIdx.x; index < end;
            index += 4 * blockDim.x)
        {
            // Matrix and tile extents are multiples of four in this branch.
            const int key = key_major ? int(index / g.valueWidth()) : int(index % g.keyWidth());
            const int value = key_major ? int(index % g.valueWidth()) : int(index / g.keyWidth());
            float4 state = *reinterpret_cast<const float4 *>(base + index);
            const float *record = source + g.recordsOffsetFloats() +
                std::size_t(first) * g.recordStrideFloats();
            for (unsigned row = first; row < last; ++row, record += g.recordStrideFloats())
            {
                const float decay = record[g.recordDecayOffsetFloats()];
                if (key_major)
                {
                    const float key_word = record[g.recordKeyOffsetFloats() + key];
                    const float *delta = record + g.recordCorrectionOffsetFloats() + value;
                    state.x = __fmaf_rn(key_word, delta[0], __fmul_rn(state.x, decay));
                    state.y = __fmaf_rn(key_word, delta[1], __fmul_rn(state.y, decay));
                    state.z = __fmaf_rn(key_word, delta[2], __fmul_rn(state.z, decay));
                    state.w = __fmaf_rn(key_word, delta[3], __fmul_rn(state.w, decay));
                }
                else
                {
                    const float delta = record[g.recordCorrectionOffsetFloats() + value];
                    const float *key_words = record + g.recordKeyOffsetFloats() + key;
                    state.x = __fmaf_rn(key_words[0], delta, __fmul_rn(state.x, decay));
                    state.y = __fmaf_rn(key_words[1], delta, __fmul_rn(state.y, decay));
                    state.z = __fmaf_rn(key_words[2], delta, __fmul_rn(state.z, decay));
                    state.w = __fmaf_rn(key_words[3], delta, __fmul_rn(state.w, decay));
                }
            }
            *reinterpret_cast<float4 *>(destination + index) = state;
            if (mirror)
            {
                // Mirror alignment never changes the already-computed arithmetic.
                if (reinterpret_cast<std::uintptr_t>(mirror) % alignof(float4) == 0)
                    *reinterpret_cast<float4 *>(mirror + index) = state;
                else
                {
                    mirror[index] = state.x;
                    mirror[index + 1] = state.y;
                    mirror[index + 2] = state.z;
                    mirror[index + 3] = state.w;
                }
            }
        }
        return;
    }
    for (std::size_t index = tile.first_element + threadIdx.x; index < end;
        index += blockDim.x)
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
            const float decayed = __fmul_rn(state, record[g.recordDecayOffsetFloats()]);
            state = __fmaf_rn(record[g.recordKeyOffsetFloats() + key],
                record[g.recordCorrectionOffsetFloats() + value], decayed);
        }
        destination[index] = state;
        if (mirror) mirror[index] = state;
    }
}
} // namespace

namespace llaminar2::gdn
{
/** @brief Record one original-arithmetic capture using a prepared query-free scheduling binding.
 * @param p Unchanged numerical tuple and complete retained/terminal contract.
 * @param stream Exact stream authenticated by the existing cold owner.
 * @param plan Immutable physical-grid/native-SM scheduling authority.
 * @return Whether tuple admission and selected native enqueue succeeded; no device query occurs here. */
bool launchCUDADenseCompactRecurrence(const GDNDenseCompactCaptureInput &p, void *stream,
    const CUDADenseRetentionCapturePlan &plan)
{
    if (reinterpret_cast<std::uintptr_t>(stream) <= 2 || !validCUDADenseCompactCaptureStorage(p) || !plan.matches(p, stream))
        return false;
    const auto &g = p.transaction.geometry;
    const dim3 grid(g.requests() * g.heads(), 1 + (g.valueWidth() - 1) / kGdnRecurrentColumnsPerBlock);
    const std::size_t shared_bytes = std::size_t(2 * g.keyWidth() + 2 * kGdnRecurrentThreads) * sizeof(float);
    const auto native_stream = static_cast<cudaStream_t>(stream);
    if (plan.schedule() == CUDADenseRetentionSchedule::CheckpointWordTwoCTA)
    {
        if (g.keyWidth() != 128 || p.transaction.layout != GDNRecurrenceStateLayout::KeyMajorValueContiguous)
            return false;
        switch (p.terminal.kind())
        {
        case GDNDenseCompactTerminalKind::DenseDestination:
            cuda_gdn_dense_compact_packed_capture_kernel<128, GDNRecurrenceStateLayout::KeyMajorValueContiguous>
                <<<grid, kGdnRecurrentThreads, shared_bytes, native_stream>>>(p);
            break;
        case GDNDenseCompactTerminalKind::RetainedPrefixesOnly:
            cuda_gdn_retained_prefixes_packed_capture_kernel<128, GDNRecurrenceStateLayout::KeyMajorValueContiguous>
                <<<grid, kGdnRecurrentThreads, shared_bytes, native_stream>>>(p);
            break;
        default: return false;
        }
        return cudaGetLastError() == cudaSuccess;
    }
    if (plan.schedule() != CUDADenseRetentionSchedule::PartitionOneCTA) return false;
#define LLAMINAR_DENSE_COMPACT_CAPTURE(DK, AXES) \
    do { \
        switch (p.terminal.kind()) { \
        case GDNDenseCompactTerminalKind::DenseDestination: \
            cuda_gdn_dense_compact_capture_kernel<DK, AXES><<<grid, kGdnRecurrentThreads, shared_bytes, native_stream>>>(p); \
            break; \
        case GDNDenseCompactTerminalKind::RetainedPrefixesOnly: \
            cuda_gdn_retained_prefixes_capture_kernel<DK, AXES><<<grid, kGdnRecurrentThreads, shared_bytes, native_stream>>>(p); \
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
    return cudaGetLastError() == cudaSuccess;
}

/** @brief Enqueue exact accepted reconstruction; native completion remains caller-owned. */
bool launchCUDADenseCompactAcceptedPrefix(const GDNDenseCompactAcceptedPrefixInput &p, void *stream)
{
    if (reinterpret_cast<std::uintptr_t>(stream) <= 2 || !validCUDADenseCompactAcceptedPrefixStorage(p))
        return false;
    const auto &g = p.transaction.geometry;
    const auto tiles = g.matrixTileCount(kCudaDenseCompactPublicationTileElements);
    if (!tiles || tiles > 65535) return false;
    const dim3 grid(g.requests() * g.heads(), static_cast<unsigned>(tiles));
    cuda_gdn_dense_compact_accepted_prefix_kernel<<<grid, 256, 0,
        static_cast<cudaStream_t>(stream)>>>(p);
    return cudaGetLastError() == cudaSuccess;
}
} // namespace llaminar2::gdn

namespace llaminar2::gdn
{
namespace
{
/** @brief Authenticate native stream ownership before capture; never called from a launch.
 * @param device Exact native CUDA ordinal.
 * @param stream Exact explicit nondefault stream.
 * @throws std::runtime_error For unknown owner, invalid/default stream or active capture. */
void requireColdDenseCaptureStream(int device, void *stream)
{
    if (device < 0 || reinterpret_cast<std::uintptr_t>(stream) <= 2)
        throw std::runtime_error("CUDA Dense cold scheduling requires an exact device and explicit stream");
    auto native = static_cast<cudaStream_t>(stream);
    cudaStreamCaptureStatus status{};
    int owner = -1;
    if (cudaStreamIsCapturing(native, &status) != cudaSuccess || status != cudaStreamCaptureStatusNone ||
        cudaStreamGetDevice(native, &owner) != cudaSuccess || owner != device)
        throw std::runtime_error("CUDA Dense scheduling stream/device mismatch or capture already active");
}
}

CUDADenseRetentionCapturePlan::CUDADenseRetentionCapturePlan(const GDNDenseCompactGeometry &geometry,
    GDNRecurrenceStateLayout axes, GDNDenseCompactTerminalKind terminal,
    const llaminar2::ComputeDevice &observed, void *stream)
    : geometry_(geometry), axes_(axes), terminal_(terminal), device_(observed.device_id),
      multiprocessors_(observed.compute_units), compute_capability_(observed.compute_capability),
      uuid_(observed.uuid), stream_(stream), ctas_(captureCTAs(geometry)),
      schedule_(chooseSchedule(geometry, axes, observed.compute_units))
{
    if (observed.type != ComputeBackendType::GPU_CUDA || device_ < 0 || compute_capability_ <= 0 ||
        uuid_.size() != 32 || !std::all_of(uuid_.begin(), uuid_.end(),
            [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }) ||
        (terminal != GDNDenseCompactTerminalKind::DenseDestination &&
            terminal != GDNDenseCompactTerminalKind::RetainedPrefixesOnly))
        throw std::runtime_error("CUDA Dense scheduling requires a complete native hardware/output identity");
}

CUDADenseRetentionCapturePlan prepareCUDADenseRetentionCapturePlan(
    const GDNDenseCompactGeometry &geometry, GDNRecurrenceStateLayout axes,
    GDNDenseCompactTerminalKind terminal, int device, void *stream)
{
    requireColdDenseCaptureStream(device, stream);
    const auto *inventory = DeviceManager::instance().hardware();
    if (!inventory) throw std::runtime_error("CUDA Dense scheduling requires initialized canonical hardware inventory");
    const ComputeDevice *found = nullptr;
    for (const auto &candidate : inventory->cuda_devices)
        if (candidate.type == ComputeBackendType::GPU_CUDA && candidate.device_id == device)
        {
            if (found) throw std::runtime_error("CUDA Dense scheduling inventory duplicates native device");
            found = &candidate;
        }
    if (!found) throw std::runtime_error("CUDA Dense scheduling device is absent from canonical hardware inventory");
    return {geometry, axes, terminal, *found, stream};
}

CUDADenseRetentionCapturePlan prepareObservedCUDADenseRetentionCapturePlan(
    const GDNDenseCompactGeometry &geometry, GDNRecurrenceStateLayout axes,
    GDNDenseCompactTerminalKind terminal, const llaminar2::ComputeDevice &observed, void *stream)
{
    requireColdDenseCaptureStream(observed.device_id, stream);
    cudaDeviceProp actual{};
    if (cudaGetDeviceProperties(&actual, observed.device_id) != cudaSuccess ||
        observed.type != ComputeBackendType::GPU_CUDA || observed.compute_units != actual.multiProcessorCount ||
        observed.compute_capability != actual.major * 10 + actual.minor ||
        observed.uuid != formatDeviceUUID(actual.uuid.bytes))
        throw std::runtime_error("CUDA Dense fixture hardware observation does not match native stream device");
    return {geometry, axes, terminal, observed, stream};
}
}
