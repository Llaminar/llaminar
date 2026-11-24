/**
 * @file ROCmFixedKProjection.hip
 * @brief ROCm fixed-K explicit-operand prefill projections (gfx1100 WMMA, scalar FMA).
 *
 * Implements ROCmFixedKProjectionLaunch.h. All kernels evaluate
 * `kernels/common/FixedKProjectionContract.h`, so a prompt row's bytes do not
 * depend on its prefill chunk, bucket or row offset (#74):
 *
 * - gfx1100 half: eight K partitions of 64-element blocks; each partition is
 *   an ascending chain of v_wmma_f32_16x16x16_f16 units from +0 (natural K
 *   order inside each 16-wide unit), folded left to right in FP32.
 * - Scalar half (gfx906 and every non-WMMA target) and strict FP32: the shared
 *   body in FixedKScalarProjectionBody.cuh, one IEEE FMA per element.
 *
 * The WMMA body exists only in gfx1100 code objects; other targets compile a
 * trapping stub that the host never launches (fixedKHalfOperandForDevice()).
 * Shared tiles are double buffered through registers; global reads are
 * 16-byte vectors when K is a multiple of eight and element gathers otherwise.
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "kernels/rocm/gemm/ROCmFixedKProjectionLaunch.h"
#include "kernels/common/FixedKScalarProjectionBody.cuh"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"

#include <array>
#include <stdexcept>

namespace llaminar2::rocm
{
namespace
{
namespace contract = ::llaminar2::fixed_k_projection;
namespace scalar = ::llaminar2::fixed_k_scalar;
using HalfGroup = FixedHalfProjectionGroup;
using FP32Group = FixedStrictFP32ProjectionGroup;

/** @brief Whether a fixed-K kernel family evaluates all partitions or publishes one. */
enum class WmmaMode { Tiled, Partial };

constexpr int kBlock = contract::kHalfMMABlockWidth;

/** @brief Byte offset of a 16-byte chunk of a staged 128-byte row, XOR-swizzled by row. */
__device__ __forceinline__ int wmmaChunkOffset(int row, int chunk)
{
    return row * kBlock * 2 + ((chunk ^ (row & 7)) * 16);
}

/**
 * @brief gfx1100 WMMA fixed-K half projection.
 * @tparam BM CTA rows. @tparam BN CTA compact columns. @tparam WM/WN Waves along M/N.
 * @tparam Mode Tiled register fold or Partial publication (partition = grid.z).
 * @tparam Vector K % 8 == 0: 16-byte global reads; otherwise element gathers.
 * @param g Borrowed group. @param m Physical rows. @param k Reduction width.
 * @param partials Partial mode only: [kPartitions][m][totalColumns] chains.
 *
 * Wave64 lane l supplies row/column l % 16 of each 16x16x16 unit (all four
 * lane groups duplicate the operand, as the w64 instruction requires) and
 * receives D[4e + l / 16][l % 16] in result element e.
 */
template<int BM, int BN, int WM, int WN, WmmaMode Mode, bool Vector>
__global__ __launch_bounds__(WM * WN * 64) void wmmaProjectionKernel(
    const HalfGroup g, int m, int k, float *__restrict__ partials)
{
#if defined(__gfx1100__)
    using Half16 = _Float16 __attribute__((ext_vector_type(16)));
    using Float4 = float __attribute__((ext_vector_type(4)));
    struct Span { uint4 lo, hi; };
    constexpr int THREADS = WM * WN * 64;
    constexpr int TM = BM / WM / 16, TN = BN / WN / 16;
    constexpr int CHUNKS = kBlock / 8;
    static_assert(TM >= 1 && TN >= 1 && (BM * CHUNKS) % THREADS == 0 && (BN * CHUNKS) % THREADS == 0);
    constexpr int A_SLOTS = BM * CHUNKS / THREADS, B_SLOTS = BN * CHUNKS / THREADS;
    constexpr int STAGE_BYTES = (BM + BN) * kBlock * 2;
    __shared__ __attribute__((aligned(16))) unsigned char staged[2 * STAGE_BYTES];

    const int total = g.totalColumns();
    const int wave = threadIdx.x / 64, lane = threadIdx.x % 64;
    const int wm = wave / WN, wn = wave % WN;
    // Tiled mode: banded 1-D raster (16 row tiles per band) for L2 reuse.
    int row_tile = blockIdx.x, col_tile = blockIdx.y;
    if constexpr (Mode == WmmaMode::Tiled)
    {
        const int row_tiles = (m + BM - 1) / BM, col_tiles = (total + BN - 1) / BN;
        const int linear = blockIdx.x;
        const int band = linear / (16 * col_tiles), band_rows = min(row_tiles - band * 16, 16);
        const int within = linear - band * 16 * col_tiles;
        row_tile = band * 16 + within % band_rows;
        col_tile = within / band_rows;
    }
    const int m0 = row_tile * BM, c0 = col_tile * BN;
    const int blocks = contract::halfMMABlocks(k);
    const int first = Mode == WmmaMode::Partial ? contract::partitionBegin(blockIdx.z, blocks) : 0;
    const int walk = (Mode == WmmaMode::Partial ? contract::partitionBegin(blockIdx.z + 1, blocks) : blocks) - first;

    const uint16_t *a_src[A_SLOTS]; int a_off[A_SLOTS], a_k[A_SLOTS]; bool a_ok[A_SLOTS];
    const uint16_t *b_src[B_SLOTS]; int b_off[B_SLOTS], b_k[B_SLOTS]; bool b_ok[B_SLOTS];
#pragma unroll
    for (int i = 0; i < A_SLOTS; ++i)
    {
        const int c = threadIdx.x + i * THREADS, row = c / CHUNKS, chunk = c % CHUNKS;
        a_ok[i] = m0 + row < m;
        a_k[i] = chunk * 8;
        a_src[i] = g.input() + static_cast<size_t>(a_ok[i] ? m0 + row : 0) * k + chunk * 8;
        a_off[i] = wmmaChunkOffset(row, chunk);
    }
#pragma unroll
    for (int i = 0; i < B_SLOTS; ++i)
    {
        const int c = threadIdx.x + i * THREADS, row = c / CHUNKS, chunk = c % CHUNKS;
        b_ok[i] = c0 + row < total;
        int local = 0;
        const int p = b_ok[i] ? scalar::projectionOf(g, c0 + row, local) : 0;
        b_k[i] = chunk * 8;
        b_src[i] = g.weight(p) + static_cast<size_t>(local) * k + chunk * 8;
        b_off[i] = BM * kBlock * 2 + wmmaChunkOffset(row, chunk);
    }
    // Rows past the matrix and elements past K stage as zero.
    const auto read = [&](const uint16_t *src, int origin, bool ok) {
        if constexpr (Vector)
            return ok && origin < k ? *reinterpret_cast<const uint4 *>(src) : make_uint4(0, 0, 0, 0);
        else
        {
            unsigned w[4];
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                const unsigned lo = ok && origin + 2 * e < k ? src[2 * e] : 0u;
                const unsigned hi = ok && origin + 2 * e + 1 < k ? src[2 * e + 1] : 0u;
                w[e] = lo | (hi << 16);
            }
            return make_uint4(w[0], w[1], w[2], w[3]);
        }
    };
    uint4 a_stage[A_SLOTS], b_stage[B_SLOTS];
    const auto fetch = [&](int block) {
        const int k0 = block * kBlock;
#pragma unroll
        for (int i = 0; i < A_SLOTS; ++i) a_stage[i] = read(a_src[i] + k0, k0 + a_k[i], a_ok[i]);
#pragma unroll
        for (int i = 0; i < B_SLOTS; ++i) b_stage[i] = read(b_src[i] + k0, k0 + b_k[i], b_ok[i]);
    };
    const auto publish = [&](int buffer) {
        unsigned char *base = staged + buffer * STAGE_BYTES;
#pragma unroll
        for (int i = 0; i < A_SLOTS; ++i) *reinterpret_cast<uint4 *>(base + a_off[i]) = a_stage[i];
#pragma unroll
        for (int i = 0; i < B_SLOTS; ++i) *reinterpret_cast<uint4 *>(base + b_off[i]) = b_stage[i];
    };

    Float4 q[TM][TN], r[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) { q[i][j] = Float4{}; r[i][j] = Float4{}; }
    int partition = 0;
    int boundary = contract::partitionBegin(1, blocks);
    const auto flush = [&](int done) {
        while (partition < contract::kPartitions && boundary == done)
        {
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                {
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        r[i][j][e] = partition ? __fadd_rn(r[i][j][e], q[i][j][e]) : q[i][j][e];
                    q[i][j] = Float4{};
                }
            ++partition;
            boundary = contract::partitionBegin(partition + 1, blocks);
        }
    };
    if constexpr (Mode == WmmaMode::Tiled) flush(0);

    if (walk > 0)
    {
        fetch(first);
        publish(0);
    }
    __syncthreads();
    const int row_lane = lane % 16;
    for (int step = 0; step < walk; ++step)
    {
        const bool more = step + 1 < walk;
        if (more) fetch(first + step + 1);
        const unsigned char *base = staged + (step & 1) * STAGE_BYTES;
#pragma unroll
        for (int u = 0; u < 4; ++u)
        {
            Half16 a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
            {
                const int row = wm * (BM / WM) + i * 16 + row_lane;
                Span span{*reinterpret_cast<const uint4 *>(base + wmmaChunkOffset(row, 2 * u)),
                          *reinterpret_cast<const uint4 *>(base + wmmaChunkOffset(row, 2 * u + 1))};
                a[i] = __builtin_bit_cast(Half16, span);
            }
#pragma unroll
            for (int j = 0; j < TN; ++j)
            {
                const int row = wn * (BN / WN) + j * 16 + row_lane;
                Span span{*reinterpret_cast<const uint4 *>(base + BM * kBlock * 2 + wmmaChunkOffset(row, 2 * u)),
                          *reinterpret_cast<const uint4 *>(base + BM * kBlock * 2 + wmmaChunkOffset(row, 2 * u + 1))};
                b[j] = __builtin_bit_cast(Half16, span);
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    q[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w64(a[i], b[j], q[i][j]);
        }
        if constexpr (Mode == WmmaMode::Tiled) flush(first + step + 1);
        if (more)
        {
            // Every wave finished reading the other buffer one barrier ago.
            publish((step + 1) & 1);
            __syncthreads();
        }
    }

    float *out = Mode == WmmaMode::Partial ? partials + static_cast<size_t>(blockIdx.z) * m * total : nullptr;
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                const int row = m0 + wm * (BM / WM) + i * 16 + lane / 16 + 4 * e;
                const int col = c0 + wn * (BN / WN) + j * 16 + row_lane;
                if (row >= m || col >= total) continue;
                if constexpr (Mode == WmmaMode::Partial)
                    out[static_cast<size_t>(row) * total + col] = q[i][j][e];
                else
                {
                    int local = 0;
                    const int p = scalar::projectionOf(g, col, local);
                    g.output(p)[static_cast<size_t>(row) * g.columns(p) + local] = r[i][j][e];
                }
            }
#else
    (void)g; (void)m; (void)k; (void)partials;
    __builtin_trap();
#endif
}

/** @brief Accept a capture-time launch; outside capture, surface a submission error. */
bool completeLaunch(hipStream_t stream)
{
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    if (llaminar2::isGraphCaptureActive() ||
        (hipStreamIsCapturing(stream, &status) == hipSuccess && status != hipStreamCaptureStatusNone))
        return true;
    return hipGetLastError() == hipSuccess;
}

/** @brief Launch one WMMA instantiation and, for partitioned mode, its fold. */
template<int BM, int BN, int WM, int WN, WmmaMode Mode>
void launchWmma(const HalfGroup &group, int m, int k, hipStream_t stream, float *partials)
{
    const unsigned row_tiles = (m + BM - 1) / BM, col_tiles = (group.totalColumns() + BN - 1) / BN;
    const bool partial = Mode == WmmaMode::Partial;
    const dim3 grid = partial ? dim3(row_tiles, col_tiles, contract::kPartitions) : dim3(row_tiles * col_tiles, 1, 1);
    if (k % contract::kRequiredKAlignment == 0)
        hipLaunchKernelGGL((wmmaProjectionKernel<BM, BN, WM, WN, Mode, true>), grid, dim3(WM * WN * 64), 0, stream,
                           group, m, k, partials);
    else
        hipLaunchKernelGGL((wmmaProjectionKernel<BM, BN, WM, WN, Mode, false>), grid, dim3(WM * WN * 64), 0, stream,
                           group, m, k, partials);
    if (partial)
    {
        const size_t outputs = static_cast<size_t>(m) * group.totalColumns();
        hipLaunchKernelGGL((scalar::foldScalarPartialsKernel<uint16_t>), dim3((outputs + 255) / 256), dim3(256), 0,
                           stream, group, m, partials);
    }
}

/** @brief Launch one scalar instantiation and, for partitioned mode, its fold. */
template<class Operand, int BM, int BN, int TM, int TN, int BK, scalar::Mode Mode>
void launchScalar(const FixedRaggedProjectionGroup<Operand> &group, int m, int k, hipStream_t stream, float *partials)
{
    const unsigned row_tiles = (m + BM - 1) / BM, col_tiles = (group.totalColumns() + BN - 1) / BN;
    const bool partial = Mode == scalar::Mode::Partial;
    const dim3 grid(row_tiles, col_tiles, partial ? contract::kPartitions : 1);
    const dim3 block((BM / TM) * (BN / TN));
    if (!partial && scalar::scalarPartitionsAligned<BK>(k))
        hipLaunchKernelGGL((scalar::scalarProjectionKernel<Operand, BM, BN, TM, TN, BK, Mode, true>), grid, block, 0,
                           stream, group, m, k, partials);
    else
        hipLaunchKernelGGL((scalar::scalarProjectionKernel<Operand, BM, BN, TM, TN, BK, Mode, false>), grid, block, 0,
                           stream, group, m, k, partials);
    if (partial)
    {
        const size_t outputs = static_cast<size_t>(m) * group.totalColumns();
        hipLaunchKernelGGL((scalar::foldScalarPartialsKernel<Operand>), dim3((outputs + 255) / 256), dim3(256), 0,
                           stream, group, m, partials);
    }
}

/** @brief Shared validation for every public launch. */
void validate(int m, int k, void *stream, FixedKSchedule schedule, int columns, float *partials, std::size_t bytes)
{
    if (!stream)
        throw std::invalid_argument("ROCm fixed-K projection requires an exact non-null stream");
    if (m <= 0 || k <= 0)
        throw std::invalid_argument("ROCm fixed-K projection requires positive rows and K");
    if (fixedKScheduleNeedsPartials(schedule) &&
        (!partials || bytes < contract::partialBytes(m, columns)))
        throw std::invalid_argument("Partitioned ROCm fixed-K projection lacks its admitted partial workspace");
}

constexpr std::array kWmmaSchedules{FixedKSchedule::WmmaPartitioned64x64, FixedKSchedule::WmmaTiled64x64,
                                    FixedKSchedule::WmmaTiled128x64};
constexpr std::array kScalarSchedules{FixedKSchedule::ScalarPartitioned64x64, FixedKSchedule::ScalarTiled64x64,
                                      FixedKSchedule::ScalarTiled128x64};
} // namespace

std::string_view fixedKScheduleName(FixedKSchedule schedule) noexcept
{
    switch (schedule)
    {
    case FixedKSchedule::WmmaPartitioned64x64: return "wmma_partitioned_64x64";
    case FixedKSchedule::WmmaTiled64x64: return "wmma_tiled_64x64";
    case FixedKSchedule::WmmaTiled128x64: return "wmma_tiled_128x64";
    case FixedKSchedule::ScalarPartitioned64x64: return "scalar_partitioned_64x64";
    case FixedKSchedule::ScalarTiled64x64: return "scalar_tiled_64x64";
    case FixedKSchedule::ScalarTiled128x64: return "scalar_tiled_128x64";
    }
    return "invalid";
}

std::span<const FixedKSchedule> fixedKSchedules(FixedKOperand operand) noexcept
{
    return operand == FixedKOperand::HalfWmma ? std::span<const FixedKSchedule>(kWmmaSchedules)
                                              : std::span<const FixedKSchedule>(kScalarSchedules);
}

FixedKOperand fixedKHalfOperandForDevice(int device_id)
{
    const auto &hardware = rocmHardwareMetadata().at(device_id);
    const bool wmma = hardware.isa_target == "gfx1100" || hardware.isa_target.starts_with("gfx1100:");
    return wmma ? FixedKOperand::HalfWmma : FixedKOperand::HalfScalar;
}

bool launchFixedKHalfProjection(const FixedHalfProjectionGroup &group, int m, int k, int device_id,
    void *stream, FixedKSchedule schedule, float *partials, std::size_t partial_bytes)
{
    validate(m, k, stream, schedule, group.totalColumns(), partials, partial_bytes);
    const bool wmma_schedule = schedule == FixedKSchedule::WmmaPartitioned64x64 ||
        schedule == FixedKSchedule::WmmaTiled64x64 || schedule == FixedKSchedule::WmmaTiled128x64;
    if (wmma_schedule != (fixedKHalfOperandForDevice(device_id) == FixedKOperand::HalfWmma))
        throw std::invalid_argument("ROCm fixed-K half schedule does not match the device's half arithmetic");
    if (hipSetDevice(device_id) != hipSuccess)
        return false;
    const auto native = static_cast<hipStream_t>(stream);
    switch (schedule)
    {
    case FixedKSchedule::WmmaPartitioned64x64: launchWmma<64, 64, 2, 2, WmmaMode::Partial>(group, m, k, native, partials); break;
    case FixedKSchedule::WmmaTiled64x64: launchWmma<64, 64, 2, 2, WmmaMode::Tiled>(group, m, k, native, nullptr); break;
    case FixedKSchedule::WmmaTiled128x64: launchWmma<128, 64, 4, 2, WmmaMode::Tiled>(group, m, k, native, nullptr); break;
    case FixedKSchedule::ScalarPartitioned64x64:
        launchScalar<uint16_t, 64, 64, 4, 4, 16, scalar::Mode::Partial>(group, m, k, native, partials); break;
    case FixedKSchedule::ScalarTiled64x64:
        launchScalar<uint16_t, 64, 64, 4, 4, 32, scalar::Mode::Tiled>(group, m, k, native, nullptr); break;
    case FixedKSchedule::ScalarTiled128x64:
        launchScalar<uint16_t, 128, 64, 8, 4, 32, scalar::Mode::Tiled>(group, m, k, native, nullptr); break;
    default: throw std::invalid_argument("Unknown ROCm fixed-K schedule");
    }
    return completeLaunch(native);
}

bool launchFixedKFP32Projection(const FixedStrictFP32ProjectionGroup &group, int m, int k, int device_id,
    void *stream, FixedKSchedule schedule, float *partials, std::size_t partial_bytes)
{
    validate(m, k, stream, schedule, group.totalColumns(), partials, partial_bytes);
    if (hipSetDevice(device_id) != hipSuccess)
        return false;
    const auto native = static_cast<hipStream_t>(stream);
    switch (schedule)
    {
    case FixedKSchedule::ScalarPartitioned64x64:
        launchScalar<float, 64, 64, 4, 4, 16, scalar::Mode::Partial>(group, m, k, native, partials); break;
    case FixedKSchedule::ScalarTiled64x64:
        launchScalar<float, 64, 64, 4, 4, 32, scalar::Mode::Tiled>(group, m, k, native, nullptr); break;
    case FixedKSchedule::ScalarTiled128x64:
        launchScalar<float, 128, 64, 8, 4, 32, scalar::Mode::Tiled>(group, m, k, native, nullptr); break;
    default: throw std::invalid_argument("Strict-FP32 ROCm fixed-K projections use scalar schedules only");
    }
    return completeLaunch(native);
}
} // namespace llaminar2::rocm
