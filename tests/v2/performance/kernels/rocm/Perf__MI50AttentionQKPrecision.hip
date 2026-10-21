/**
 * @file Perf__MI50AttentionQKPrecision.hip
 * @brief Matched gfx906 FP16-fdot2 and Q16-sdot2 causal-prefill QK economics.
 *
 * This diagnostic compares the scoring operation, not complete FlashAttention
 * or model inference. Both modes use the same GQA mapping, causal mask, LDS
 * tiles, four accumulator chains, launch geometry and FP32 score publication.
 * Q16 keys retain a native Q16_1 block per head; query codes obey the existing
 * CPU int32 range proof. FP64 equations on the original and encoded operands
 * distinguish representation error from incorrect dot products.
 *
 * Initialization owns every allocation and transfer. Retained HIP graphs time
 * prepared QK, input preparation, and preparation plus QK separately. The last
 * scope encodes all queries and only newly appended keys, exposing quantization
 * cost without re-encoding an existing prefix. AB/BA samples and native timing
 * events keep host submission and profiler overhead out of comparative timing.
 * A separate one-dispatch mode admits exactly one mode/shape for ISA/profiling.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "backends/rocm/ROCmRuntimeStartup.h"
#include "kernels/cpu/attention/CPUQ16AttentionMath.h"
#include "tensors/BlockStructures.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace mi50_qk_bench
{
constexpr int kThreads = 256;
constexpr int kQueryTile = 16;
constexpr int kKeyTile = 64;
constexpr int kGuardElements = 64;
constexpr float kGuardValue = 123456.0f;
// Evaluate the CPU policy at host compile time. Its fatal exception branch
// must not become a device call even though these two reductions are valid.
constexpr int kQueryLimit64 = llaminar2::cpu::q16AttentionQueryLimit(64);
constexpr int kQueryLimit128 = llaminar2::cpu::q16AttentionQueryLimit(128);

/** @brief Arithmetic/storage choice; neither mode changes production dispatch. */
enum class Precision { FP16, Q16 };
/** @brief Explicit captured timing boundary. */
enum class Scope { PreparedQK, Preparation, Pipeline };

/** @brief Immutable participant-local causal prefill geometry. */
struct Shape
{
    int queries = 512; ///< Actual query rows, including a partial final tile.
    int keys = 512; ///< Live history, including any previously encoded prefix.
    int heads = 14; ///< Local query heads.
    int kv_heads = 2; ///< Local key heads; query heads must divide by this count.
    int head_dim = 64; ///< One Q16 physical block per head, either 64 or 128.
};

/** @brief Command-line controls are local to this diagnostic binary. */
struct Options
{
    Shape shape;
    int device = 0;
    int samples = 25;
    int warmups = 4;
    int batch = 8;
    bool exact_shape = false;
    bool single_mode = false;
    Precision mode = Precision::FP16;
    bool profile = false;
    Scope profile_scope = Scope::PreparedQK;
};

/**
 * @brief Surface a native error rather than accepting a fast failed launch.
 * @param result Native HIP status.
 * @param operation Operation whose ownership/order contract failed.
 */
void checked(hipError_t result, const char *operation)
{
    if (result != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(result));
}

/** @brief Stable machine-readable mode name. @param mode Selected precision. */
const char *name(Precision mode) { return mode == Precision::FP16 ? "fp16_fdot2" : "q16_sdot2"; }
/** @brief Stable timing-scope name. @param scope Captured operation boundary. */
const char *name(Scope scope)
{
    switch (scope)
    {
    case Scope::PreparedQK: return "prepared_qk";
    case Scope::Preparation: return "prepare";
    case Scope::Pipeline: return "prepare_qk";
    }
    throw std::logic_error("Unknown MI50 timing scope");
}

/**
 * @brief Parse a whole decimal argument without silently accepting a suffix.
 * @param text Argument value.
 * @param minimum Smallest admitted value.
 * @return Validated integer.
 */
int number(std::string_view text, int minimum = 1)
{
    int value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value < minimum)
        throw std::invalid_argument("Invalid MI50 benchmark integer: " + std::string(text));
    return value;
}

/**
 * @brief Reject a topology that either arithmetic mode cannot implement.
 * @param shape Requested live geometry.
 */
void validate(const Shape &shape)
{
    if (shape.queries <= 0 || shape.keys < shape.queries || shape.heads <= 0 ||
        shape.kv_heads <= 0 || shape.heads % shape.kv_heads != 0 ||
        (shape.head_dim != 64 && shape.head_dim != 128))
        throw std::invalid_argument("MI50 QK requires 0 < queries <= keys, divisible GQA heads, and head_dim 64 or 128");
    const std::uint64_t rows = std::uint64_t(shape.queries) * shape.heads;
    const std::uint64_t key_rows = std::uint64_t(shape.keys) * shape.kv_heads;
    if (rows > std::uint64_t(std::numeric_limits<int>::max()) ||
        rows + key_rows > std::uint64_t(std::numeric_limits<int>::max()) ||
        shape.heads > 65535 || (shape.queries + 15LL) / 16 > 65535)
        throw std::invalid_argument("MI50 QK shape exceeds the native row/grid extent");
    if (rows > (std::numeric_limits<std::size_t>::max() / sizeof(float) - 2 * kGuardElements) / shape.keys)
        throw std::invalid_argument("MI50 QK score storage exceeds the addressable extent");
}

/**
 * @brief Parse explicit shape, timing and isolated-profile selectors.
 * @param argc Native argument count.
 * @param argv Native arguments; unknown options fail before HIP admission.
 * @return Complete benchmark options.
 */
Options parse(int argc, char **argv)
{
    Options options;
    for (int index = 1; index < argc; ++index)
    {
        std::string_view argument(argv[index]);
        if (argument == "--help")
        {
            std::cout << "MI50 causal-prefill QK precision benchmark (CSV on stdout)\n"
                "  --mode=both|fp16|q16 --device=N\n"
                "  --queries=N --keys=N --heads=N --kv-heads=N --head-dim=64|128\n"
                "  --samples=N --warmups=N --batch=N\n"
                "  --profile=prepared_qk|prepare|prepare_qk (requires one mode)\n"
                "Without a shape selector, run the built-in short/long/partial/GQA sweep.\n";
            std::exit(0);
        }
        const auto equal = argument.find('=');
        if (equal == std::string_view::npos)
            throw std::invalid_argument("Expected --option=value: " + std::string(argument));
        const auto key = argument.substr(0, equal);
        const auto value = argument.substr(equal + 1);
        if (key == "--mode")
        {
            if (value != "both" && value != "fp16" && value != "q16")
                throw std::invalid_argument("--mode must be both, fp16 or q16");
            options.single_mode = value != "both";
            options.mode = value == "q16" ? Precision::Q16 : Precision::FP16;
        }
        else if (key == "--device") options.device = number(value, 0);
        else if (key == "--samples") options.samples = number(value);
        else if (key == "--warmups") options.warmups = number(value, 0);
        else if (key == "--batch") options.batch = number(value);
        else if (key == "--profile")
        {
            options.profile = true;
            if (value == "prepared_qk") options.profile_scope = Scope::PreparedQK;
            else if (value == "prepare") options.profile_scope = Scope::Preparation;
            else if (value == "prepare_qk") options.profile_scope = Scope::Pipeline;
            else throw std::invalid_argument("Unknown --profile scope");
        }
        else
        {
            options.exact_shape = true;
            if (key == "--queries") options.shape.queries = number(value);
            else if (key == "--keys") options.shape.keys = number(value);
            else if (key == "--heads") options.shape.heads = number(value);
            else if (key == "--kv-heads") options.shape.kv_heads = number(value);
            else if (key == "--head-dim") options.shape.head_dim = number(value);
            else throw std::invalid_argument("Unknown MI50 option: " + std::string(key));
        }
    }
    validate(options.shape);
    if (options.profile && !options.single_mode)
        throw std::invalid_argument("--profile requires --mode=fp16 or --mode=q16");
    return options;
}

/** @brief Packed half row; pair alignment matches the integer candidate. */
template<int Dim> struct alignas(4) HalfRow { __half qs[Dim]; };
template<Precision Mode, int Dim>
using Row = std::conditional_t<Mode == Precision::FP16, HalfRow<Dim>,
    std::conditional_t<Dim == 64, llaminar2::Q16_1Block_64, llaminar2::Q16_1Block_128>>;
using HalfPair = _Float16 __attribute__((ext_vector_type(2)));
using Int16Pair = std::int16_t __attribute__((ext_vector_type(2)));

/**
 * @brief Encode one wave-owned row without allocation or host observation.
 * @tparam Mode Native operand representation.
 * @tparam Dim Head/block width.
 * @param source FP32 row in an admitted persistent source buffer.
 * @param destination Exclusive output row.
 * @param query Distinguish the int32-safe query range from full-range keys.
 */
template<Precision Mode, int Dim>
__device__ __forceinline__ void encodeRow(const float *source, Row<Mode, Dim> &destination, bool query)
{
    const int lane = threadIdx.x & 63;
    if constexpr (Mode == Precision::FP16)
    {
        for (int col = lane; col < Dim; col += 64)
            destination.qs[col] = __float2half_rn(source[col]);
    }
    else
    {
        float maximum = 0.0f;
        for (int col = lane; col < Dim; col += 64)
            maximum = fmaxf(maximum, fabsf(source[col]));
        for (int offset = 32; offset > 0; offset /= 2)
            maximum = fmaxf(maximum, __shfl_xor(maximum, offset, 64));
        constexpr int query_limit = Dim == 64 ? kQueryLimit64 : kQueryLimit128;
        const int limit = query ? query_limit : 32767;
        const float scale = maximum / float(limit);
        // Preserve the CPU native-key tiny-scale contract. Queries use the
        // separate ephemeral encoder; neither range is inferred from capacity.
        const bool live = query ? maximum > 1e-12f : scale > 1e-10f;
        const float reciprocal = live ? 1.0f / scale : 0.0f;
        int sum = 0;
        for (int col = lane; col < Dim; col += 64)
        {
            const float scaled = source[col] * reciprocal;
            const float rounded = query ? nearbyintf(scaled) : roundf(scaled);
            const int code = live ? int(fmaxf(-float(limit), fminf(float(limit), rounded))) : 0;
            destination.qs[col] = std::int16_t(code);
            sum += code;
        }
        for (int offset = 32; offset > 0; offset /= 2)
            sum += __shfl_xor(sum, offset, 64);
        if (lane == 0)
        {
            destination.d = query && !live ? 0.0f : scale;
            destination.sum_qs = sum;
        }
    }
}

/**
 * @brief Prepare all live queries and only the newly appended key suffix.
 * @tparam Mode Operand representation.
 * @tparam Dim Block width.
 * @param query FP32 queries [row][query head][dimension].
 * @param key FP32 keys [position][KV head][dimension].
 * @param prepared_query Persistent query rows.
 * @param prepared_key Persistent key rows, preserving the encoded prefix.
 * @param query_rows Flattened query/head rows.
 * @param key_begin First new flattened key/head row.
 * @param new_key_rows Live new flattened key/head rows.
 */
template<Precision Mode, int Dim>
__global__ __launch_bounds__(kThreads, 2) void mi50_qk_prepare(
    const float *query, const float *key, Row<Mode, Dim> *prepared_query,
    Row<Mode, Dim> *prepared_key, int query_rows, int key_begin, int new_key_rows)
{
    const int row = int(blockIdx.x) * 4 + int(threadIdx.x) / 64;
    if (row < query_rows)
        encodeRow<Mode, Dim>(query + std::size_t(row) * Dim, prepared_query[row], true);
    else if (row - query_rows < new_key_rows)
    {
        const int key_row = key_begin + row - query_rows;
        encodeRow<Mode, Dim>(key + std::size_t(key_row) * Dim, prepared_key[key_row], false);
    }
}

/**
 * @brief One native pair dot, with no emulation on an unsupported target.
 * @tparam Mode Exact requested arithmetic.
 * @tparam Dim Row/block width.
 * @tparam Acc Float or int32 accumulation, selected with the representation.
 * @param query Padded LDS row of encoded pairs.
 * @param key Padded LDS row of encoded pairs.
 * @param pair Pair index within each head.
 * @param accumulator Ordered accumulator for this complete head reduction.
 * @return Updated native accumulator.
 */
template<Precision Mode, int Dim, class Acc>
__device__ __forceinline__ Acc pairDot(const std::uint32_t *query,
    const std::uint32_t *key, int pair, Acc accumulator)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx906__)
    if constexpr (Mode == Precision::FP16)
        return __builtin_amdgcn_fdot2(
            *reinterpret_cast<const HalfPair *>(query + pair),
            *reinterpret_cast<const HalfPair *>(key + pair), accumulator, false);
    else
        return __builtin_amdgcn_sdot2(
            *reinterpret_cast<const Int16Pair *>(query + pair),
            *reinterpret_cast<const Int16Pair *>(key + pair), accumulator, false);
#else
    __builtin_trap();
    return accumulator;
#endif
}

/**
 * @brief Matched tiled QK scoring; four chains hide the native dot latency.
 * @tparam Mode FP16 fdot2 or overflow-proven Q16 sdot2.
 * @tparam Dim Head width.
 * @param query Prepared token-major queries.
 * @param key Prepared position-major GQA keys.
 * @param scores Exclusive FP32 [head][query][key] score publication.
 * @param shape Live geometry; unused allocation capacity is never an extent.
 */
template<Precision Mode, int Dim>
__global__ __launch_bounds__(kThreads, 2) void mi50_qk_score(
    const Row<Mode, Dim> *query, const Row<Mode, Dim> *key, float *scores, Shape shape)
{
    // An odd dword row stride gives both representations the same bank
    // distribution. Native FP16 rows have a power-of-two stride, whereas Q16
    // metadata changes that stride; staging native structs would bias this
    // instruction comparison through unequal LDS bank conflicts.
    __shared__ std::uint32_t query_tile[kQueryTile][Dim / 2 + 1];
    __shared__ std::uint32_t key_tile[kKeyTile][Dim / 2 + 1];
    __shared__ float query_scale[kQueryTile], key_scale[kKeyTile];
    const int head = int(blockIdx.z);
    const int kv_head = head / (shape.heads / shape.kv_heads);
    const int q_begin = int(blockIdx.y) * kQueryTile;
    const int k_begin = int(blockIdx.x) * kKeyTile;
    const int local_k = int(threadIdx.x) & 63;
    const int local_q = int(threadIdx.x) / 64;
    const int absolute_begin = shape.keys - shape.queries;

    // Fully masked tiles publish the exact mask without loading operands.
    if (k_begin > absolute_begin + q_begin + kQueryTile - 1)
    {
        for (int row = local_q; row < kQueryTile; row += 4)
            if (q_begin + row < shape.queries && k_begin + local_k < shape.keys)
                scores[(std::size_t(head) * shape.queries + q_begin + row) * shape.keys + k_begin + local_k] = -INFINITY;
        return;
    }

    // Identical pair-coalesced LDS staging in both modes. Only Q16 scales add
    // metadata; sum_qs is native storage metadata and is not a dot operand.
    for (int index = int(threadIdx.x); index < kQueryTile * (Dim / 2); index += kThreads)
    {
        const int row = index / (Dim / 2), pair = index % (Dim / 2);
        const bool live = q_begin + row < shape.queries;
        const auto *source = live ? query + std::size_t(q_begin + row) * shape.heads + head : query;
        query_tile[row][pair] = live
            ? *reinterpret_cast<const std::uint32_t *>(source->qs + pair * 2) : 0;
    }
    for (int index = int(threadIdx.x); index < kKeyTile * (Dim / 2); index += kThreads)
    {
        const int row = index / (Dim / 2), pair = index % (Dim / 2);
        const bool live = k_begin + row < shape.keys;
        const auto *source = live ? key + std::size_t(k_begin + row) * shape.kv_heads + kv_head : key;
        key_tile[row][pair] = live
            ? *reinterpret_cast<const std::uint32_t *>(source->qs + pair * 2) : 0;
    }
    if constexpr (Mode == Precision::Q16)
    {
        if (int(threadIdx.x) < kQueryTile)
            query_scale[threadIdx.x] = q_begin + int(threadIdx.x) < shape.queries
                ? query[std::size_t(q_begin + int(threadIdx.x)) * shape.heads + head].d : 0;
        if (int(threadIdx.x) < kKeyTile)
            key_scale[threadIdx.x] = k_begin + int(threadIdx.x) < shape.keys
                ? key[std::size_t(k_begin + int(threadIdx.x)) * shape.kv_heads + kv_head].d : 0;
    }
    __syncthreads();

    using Acc = std::conditional_t<Mode == Precision::FP16, float, std::int32_t>;
    Acc dot0 = 0, dot1 = 0, dot2 = 0, dot3 = 0;
#pragma clang loop unroll(disable)
    for (int pair = 0; pair < Dim / 2; ++pair)
    {
        dot0 = pairDot<Mode, Dim>(query_tile[local_q], key_tile[local_k], pair, dot0);
        dot1 = pairDot<Mode, Dim>(query_tile[local_q + 4], key_tile[local_k], pair, dot1);
        dot2 = pairDot<Mode, Dim>(query_tile[local_q + 8], key_tile[local_k], pair, dot2);
        dot3 = pairDot<Mode, Dim>(query_tile[local_q + 12], key_tile[local_k], pair, dot3);
    }
    const auto publish = [&](int row, Acc dot)
    {
        if (q_begin + row >= shape.queries || k_begin + local_k >= shape.keys) return;
        float result = float(dot) / sqrtf(float(Dim));
        if constexpr (Mode == Precision::Q16)
            result *= query_scale[row] * key_scale[local_k];
        if (k_begin + local_k > absolute_begin + q_begin + row) result = -INFINITY;
        scores[(std::size_t(head) * shape.queries + q_begin + row) * shape.keys + k_begin + local_k] = result;
    };
    publish(local_q, dot0); publish(local_q + 4, dot1);
    publish(local_q + 8, dot2); publish(local_q + 12, dot3);
}

/**
 * @brief Persistent test-owned native allocation outside every captured scope.
 * @tparam T Element representation.
 *
 * This standalone hardware probe has no production admission/planning ledger.
 * Each backing has one RAII owner; capacities describe immutable test inputs,
 * never a competing inference allocation decision or an anonymous reserve.
 */
template<class T> class Storage final
{
public:
    /** @brief Allocate exactly the declared probe extent. @param count Elements. */
    explicit Storage(std::size_t count) { checked(hipMalloc(&data_, count * sizeof(T)), "allocate probe storage"); }
    /** @brief Called after the owning execution scope has joined its consumers. */
    ~Storage() { if (data_) (void)hipFree(data_); }
    Storage(const Storage &) = delete;
    Storage &operator=(const Storage &) = delete;
    /** @return The stable pointer embedded in every retained graph. */
    T *data() const { return data_; }
private:
    T *data_ = nullptr;
};

/** @brief Explicit stream, observation events and immutable native graphs. */
class Execution final
{
public:
    /** @brief Create nondefault execution and explicit timing events. */
    Execution()
    {
        checked(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create explicit stream");
        if (!stream) throw std::runtime_error("HIP returned a null benchmark stream");
        checked(hipEventCreate(&start_), "create timing start");
        checked(hipEventCreate(&stop_), "create timing stop");
    }
    /** @brief Observation/retirement joins work before graphs or buffers expire. */
    ~Execution()
    {
        retireGraphs();
        if (start_) (void)hipEventDestroy(start_);
        if (stop_) (void)hipEventDestroy(stop_);
        if (stream) (void)hipStreamDestroy(stream);
    }
    Execution(const Execution &) = delete;
    Execution &operator=(const Execution &) = delete;

    /**
     * @brief Capture a fixed complete timing transaction once.
     * @tparam Launch Callable that only launches already-bound operations.
     * @param launch Captured operation.
     * @param batch Number of repetitions embedded in the retained graph.
     * @return Executable retained by this execution owner.
     */
    template<class Launch> hipGraphExec_t capture(Launch launch, int batch)
    {
        checked(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "begin probe capture");
        for (int repeat = 0; repeat < batch; ++repeat) launch();
        hipGraph_t graph = nullptr;
        checked(hipStreamEndCapture(stream, &graph), "end probe capture");
        graphs_.push_back(graph);
        hipGraphExec_t executable = nullptr;
        checked(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate probe graph");
        executables_.push_back(executable);
        return executable;
    }

    /**
     * @brief Native event time for one retained replay, normalized by its batch.
     * @param graph Complete captured transaction.
     * @param batch Number of embedded repetitions.
     * @return Microseconds per scoring/preparation operation.
     */
    double sample(hipGraphExec_t graph, int batch)
    {
        checked(hipEventRecord(start_, stream), "record timing start");
        checked(hipGraphLaunch(graph, stream), "launch probe graph");
        checked(hipEventRecord(stop_, stream), "record timing stop");
        checked(hipEventSynchronize(stop_), "observe terminal timing event");
        float milliseconds = 0;
        checked(hipEventElapsedTime(&milliseconds, start_, stop_), "read GPU elapsed time");
        if (!(milliseconds > 0 && std::isfinite(milliseconds)))
            throw std::runtime_error("Invalid native probe duration");
        return double(milliseconds) * 1000 / batch;
    }
    /** @brief Finish intentional setup/result I/O on this exact stream. */
    void observe() { checked(hipStreamSynchronize(stream), "observe probe I/O"); }

    /**
     * @brief Join and retire captured consumers before a candidate frees storage.
     *
     * This terminal test boundary also covers exception unwinding. Sequential
     * candidates share this graph family; no executable survives its backing.
     */
    void retireGraphs() noexcept
    {
        if (stream) (void)hipStreamSynchronize(stream);
        for (auto graph : executables_) (void)hipGraphExecDestroy(graph);
        for (auto graph : graphs_) (void)hipGraphDestroy(graph);
        executables_.clear();
        graphs_.clear();
    }

    hipStream_t stream = nullptr; ///< Sole producer/consumer stream for this probe.
private:
    hipEvent_t start_ = nullptr, stop_ = nullptr;
    std::vector<hipGraph_t> graphs_;
    std::vector<hipGraphExec_t> executables_;
};

/** @brief Same deterministic FP32 operands for both representations. */
struct Inputs
{
    std::vector<float> query, key;
    /** @brief Include ordinary, high-range and signed-max rows. @param shape Geometry. */
    explicit Inputs(const Shape &shape)
        : query(std::size_t(shape.queries) * shape.heads * shape.head_dim),
          key(std::size_t(shape.keys) * shape.kv_heads * shape.head_dim)
    {
        std::mt19937 generator(1729);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        for (auto &value : query) value = distribution(generator);
        for (auto &value : key) value = distribution(generator);
        // Constant extreme-code rows make an incorrect int32 range bound fail
        // even when ordinary random dots happen to cancel.
        for (int col = 0; col < shape.head_dim; ++col)
        {
            query[col] = 1.0f;
            key[col] = 1.0f;
            if (shape.keys > 1)
                key[std::size_t(shape.kv_heads) * shape.head_dim + col] = -1.0f;
        }
        // An outlier is legitimate live data. Preserve it in the comparison
        // so an accuracy result cannot silently assume uniform block ranges.
        if (query.size() > std::size_t(shape.head_dim) * 2) query[shape.head_dim * 2] = 130;
    }
};

/** @brief Numerical evidence gathered outside every timed/captured scope. */
struct Error
{
    double relative_l2 = 0;
    double maximum_absolute = 0;
    double encoded_equation_error = 0;
    int observed_scores = 0;
};

/**
 * @brief Independently encode initial rows for profiler-only device setup.
 * @tparam Mode Operand representation.
 * @tparam Dim Head width.
 * @param input Original FP32 row-major values.
 * @param queries Whether rows use the query reduction bound.
 * @return Prepared rows, uploaded only before timing/profiling.
 */
template<Precision Mode, int Dim>
std::vector<Row<Mode, Dim>> hostEncode(const std::vector<float> &input, bool queries)
{
    std::vector<Row<Mode, Dim>> rows(input.size() / Dim);
    for (std::size_t row = 0; row < rows.size(); ++row)
    {
        if constexpr (Mode == Precision::FP16)
            for (int col = 0; col < Dim; ++col) rows[row].qs[col] = __float2half_rn(input[row * Dim + col]);
        else
        {
            float maximum = 0;
            for (int col = 0; col < Dim; ++col) maximum = std::max(maximum, std::abs(input[row * Dim + col]));
            const int limit = queries ? llaminar2::cpu::q16AttentionQueryLimit(Dim) : 32767;
            const float scale = maximum / limit;
            const bool live = queries ? maximum > 1e-12f : scale > 1e-10f;
            const float reciprocal = live ? 1.0f / scale : 0;
            rows[row].d = queries && !live ? 0 : scale;
            for (int col = 0; col < Dim; ++col)
            {
                const float value = input[row * Dim + col] * reciprocal;
                const float rounded = queries ? std::nearbyint(value) : std::round(value);
                const int code = live ? int(std::clamp(rounded, -float(limit), float(limit))) : 0;
                rows[row].qs[col] = std::int16_t(code);
                rows[row].sum_qs += code;
            }
        }
    }
    return rows;
}

/** @brief Persistent representation, output guards and graph family for one mode. */
template<Precision Mode, int Dim> class Candidate final
{
public:
    /**
     * @brief Bind immutable storage; setup uploads are outside graph replay.
     * @param shape Live geometry.
     * @param inputs Common FP32 input cohort.
     * @param query Original device queries, shared by sequential candidates.
     * @param key Original device keys, shared by sequential candidates.
     * @param execution Exact ordered stream and graph owner.
     */
    Candidate(Shape shape, const Inputs &inputs, const float *query, const float *key, Execution &execution)
        : shape_(shape), source_query_(query), source_key_(key), execution_(execution),
          queries_(std::size_t(shape.queries) * shape.heads), keys_(std::size_t(shape.keys) * shape.kv_heads),
          output_(std::size_t(shape.queries) * shape.keys * shape.heads + 2 * kGuardElements)
    {
        const auto query_rows = hostEncode<Mode, Dim>(inputs.query, true);
        const auto key_rows = hostEncode<Mode, Dim>(inputs.key, false);
        checked(hipMemcpyAsync(queries_.data(), query_rows.data(), query_rows.size() * sizeof(Row<Mode, Dim>), hipMemcpyHostToDevice, execution_.stream), "upload prepared queries");
        checked(hipMemcpyAsync(keys_.data(), key_rows.data(), key_rows.size() * sizeof(Row<Mode, Dim>), hipMemcpyHostToDevice, execution_.stream), "upload prepared keys");
        std::array<float, kGuardElements> guards;
        guards.fill(kGuardValue);
        checked(hipMemcpyAsync(output_.data(), guards.data(), sizeof(guards), hipMemcpyHostToDevice, execution_.stream), "upload leading guards");
        checked(hipMemcpyAsync(scores() + scoreCount(), guards.data(), sizeof(guards), hipMemcpyHostToDevice, execution_.stream), "upload trailing guards");
        checked(hipMemsetAsync(scores(), 0x7f, scoreCount() * sizeof(float), execution_.stream), "initialize score extent");
        execution_.observe();
    }

    /** @brief Retire all bound graphs before the candidate's backing expires. */
    ~Candidate() { execution_.retireGraphs(); }
    Candidate(const Candidate &) = delete;
    Candidate &operator=(const Candidate &) = delete;

    /**
     * @brief Launch one allocation-free operation on the bound nondefault stream.
     * @param scope Exact operation boundary.
     */
    void launch(Scope scope)
    {
        if (!execution_.stream) throw std::logic_error("MI50 candidate has no exact stream");
        if (scope != Scope::PreparedQK)
        {
            const int query_rows = shape_.queries * shape_.heads;
            const int new_key_rows = shape_.queries * shape_.kv_heads;
            const int key_begin = (shape_.keys - shape_.queries) * shape_.kv_heads;
            hipLaunchKernelGGL((mi50_qk_prepare<Mode, Dim>), dim3((query_rows + new_key_rows - 1) / 4 + 1), dim3(kThreads), 0, execution_.stream,
                source_query_, source_key_, queries_.data(), keys_.data(), query_rows, key_begin, new_key_rows);
            checked(hipGetLastError(), "launch live Q/K preparation");
        }
        if (scope != Scope::Preparation)
        {
            hipLaunchKernelGGL((mi50_qk_score<Mode, Dim>), dim3((shape_.keys - 1) / kKeyTile + 1,
                (shape_.queries - 1) / kQueryTile + 1, shape_.heads), dim3(kThreads), 0, execution_.stream,
                queries_.data(), keys_.data(), scores(), shape_);
            checked(hipGetLastError(), "launch matched QK scoring");
        }
    }

    /**
     * @brief Verify guards, mask edges, GQA and independently encoded equations.
     * @param original Original common FP32 operands.
     * @return Original-input error and encoded-equation closure.
     */
    Error verify(const Inputs &original)
    {
        std::vector<Row<Mode, Dim>> query_rows(std::size_t(shape_.queries) * shape_.heads);
        std::vector<Row<Mode, Dim>> key_rows(std::size_t(shape_.keys) * shape_.kv_heads);
        std::array<float, kGuardElements> before{}, after{};
        checked(hipMemcpyAsync(query_rows.data(), queries_.data(), query_rows.size() * sizeof(Row<Mode, Dim>), hipMemcpyDeviceToHost, execution_.stream), "observe encoded queries");
        checked(hipMemcpyAsync(key_rows.data(), keys_.data(), key_rows.size() * sizeof(Row<Mode, Dim>), hipMemcpyDeviceToHost, execution_.stream), "observe encoded keys");
        checked(hipMemcpyAsync(before.data(), output_.data(), sizeof(before), hipMemcpyDeviceToHost, execution_.stream), "observe leading guards");
        checked(hipMemcpyAsync(after.data(), scores() + scoreCount(), sizeof(after), hipMemcpyDeviceToHost, execution_.stream), "observe trailing guards");
        struct Point { int head, query, key; float score = 0; };
        std::vector<Point> points;
        for (int head = 0; head < shape_.heads; ++head)
            for (int query : {0, std::min(15, shape_.queries - 1), std::min(16, shape_.queries - 1), shape_.queries / 2, shape_.queries - 1})
            {
                const int end = shape_.keys - shape_.queries + query;
                for (int key : {0, std::min(1, end), std::min(63, end), std::min(64, end), end / 2, end, std::min(end + 1, shape_.keys - 1), shape_.keys - 1})
                    points.push_back({head, query, key});
            }
        for (auto &point : points)
        {
            const auto offset = (std::size_t(point.head) * shape_.queries + point.query) * shape_.keys + point.key;
            checked(hipMemcpyAsync(&point.score, scores() + offset, sizeof(float), hipMemcpyDeviceToHost, execution_.stream), "observe representative score");
        }
        execution_.observe();
        const auto check_encoding = [&](const auto &rows, const auto &source, bool queries)
        {
            const auto expected = hostEncode<Mode, Dim>(source, queries);
            for (std::size_t row = 0; row < rows.size(); ++row)
            {
                if constexpr (Mode == Precision::FP16)
                {
                    if (std::memcmp(rows[row].qs, expected[row].qs, sizeof(rows[row].qs)) != 0)
                        throw std::runtime_error("FP16 encoding differs from independent round-to-nearest conversion");
                }
                else
                {
                    const int limit = queries ? llaminar2::cpu::q16AttentionQueryLimit(Dim) : 32767;
                    if (std::abs(rows[row].d - expected[row].d) > 1e-6f * std::abs(expected[row].d))
                        throw std::runtime_error("Q16 block scale differs from independent maximum/range equation");
                    int sum = 0;
                    for (int col = 0; col < Dim; ++col)
                    {
                        const int code = rows[row].qs[col];
                        // Host/device reciprocal rounding can move a midpoint
                        // by one code; range and complete metadata stay exact.
                        if (std::abs(code) > limit || std::abs(code - int(expected[row].qs[col])) > 1)
                            throw std::runtime_error("Q16 encoding violates its range or independent rounding equation");
                        sum += code;
                    }
                    if (sum != rows[row].sum_qs)
                        throw std::runtime_error("Q16 sum_qs metadata does not equal the encoded live row");
                }
            }
        };
        check_encoding(query_rows, original.query, true);
        check_encoding(key_rows, original.key, false);
        for (int guard = 0; guard < kGuardElements; ++guard)
            if (before[guard] != kGuardValue || after[guard] != kGuardValue)
                throw std::runtime_error("QK scoring wrote outside its declared live extent");
        Error result;
        double squared_error = 0, squared_reference = 0;
        for (const auto &point : points)
        {
            if (point.key > shape_.keys - shape_.queries + point.query)
            {
                if (point.score != -std::numeric_limits<float>::infinity())
                    throw std::runtime_error("Incorrect causal mask publication");
                continue;
            }
            const int kv_head = point.head / (shape_.heads / shape_.kv_heads);
            const std::size_t qrow = std::size_t(point.query) * shape_.heads + point.head;
            const std::size_t krow = std::size_t(point.key) * shape_.kv_heads + kv_head;
            double reference = 0, encoded = 0;
            std::int64_t integer_dot = 0;
            for (int col = 0; col < Dim; ++col)
            {
                reference += double(original.query[qrow * Dim + col]) * original.key[krow * Dim + col];
                if constexpr (Mode == Precision::FP16)
                    encoded += double(__half2float(query_rows[qrow].qs[col])) * __half2float(key_rows[krow].qs[col]);
                else
                    integer_dot += std::int64_t(query_rows[qrow].qs[col]) * key_rows[krow].qs[col];
            }
            if constexpr (Mode == Precision::Q16)
            {
                if (integer_dot < std::numeric_limits<std::int32_t>::min() || integer_dot > std::numeric_limits<std::int32_t>::max())
                    throw std::runtime_error("Q16 complete score reduction exceeds int32");
                encoded = double(integer_dot) * query_rows[qrow].d * key_rows[krow].d;
            }
            reference /= std::sqrt(double(Dim));
            encoded /= std::sqrt(double(Dim));
            const double equation_error = std::abs(double(point.score) - encoded);
            if (!std::isfinite(point.score) || equation_error > 4e-6 * (1 + std::abs(encoded)))
                throw std::runtime_error(std::string(name(Mode)) + " does not close its independently decoded FP64 score equation");
            const double error = double(point.score) - reference;
            squared_error += error * error; squared_reference += reference * reference;
            result.maximum_absolute = std::max(result.maximum_absolute, std::abs(error));
            result.encoded_equation_error = std::max(result.encoded_equation_error, equation_error);
            ++result.observed_scores;
        }
        result.relative_l2 = std::sqrt(squared_error / std::max(squared_reference, 1e-30));
        return result;
    }

private:
    /** @return Logical score count, excluding output guards. */
    std::size_t scoreCount() const { return std::size_t(shape_.queries) * shape_.keys * shape_.heads; }
    /** @return First score; guards never enter kernel addressing or useful work. */
    float *scores() const { return output_.data() + kGuardElements; }
    Shape shape_;
    const float *source_query_, *source_key_;
    Execution &execution_;
    Storage<Row<Mode, Dim>> queries_, keys_;
    Storage<float> output_;
};

/**
 * @brief Quantile from a sorted measurement cohort.
 * @param sorted Samples already sorted by duration.
 * @param fraction Requested order statistic.
 * @return Duration in microseconds.
 */
double quantile(const std::vector<double> &sorted, double fraction)
{ return sorted[std::size_t((sorted.size() - 1) * fraction)]; }

/**
 * @brief Publish scope-specific throughput without attributing preparation FLOPs.
 * @param shape Logical geometry.
 * @param mode Measured representation.
 * @param scope Complete timed boundary.
 * @param samples Unprofiled native-event measurements.
 * @param error Outside-timing numerical result.
 * @param batch Repetitions per retained graph.
 */
void report(Shape shape, Precision mode, Scope scope, std::vector<double> samples, const Error &error, int batch)
{
    std::sort(samples.begin(), samples.end());
    const double median = quantile(samples, .5);
    const double live_scores = double(shape.heads) *
        (double(shape.queries) * (shape.keys - shape.queries) + double(shape.queries) * (shape.queries + 1) / 2);
    const double tops = scope == Scope::Preparation ? 0 : 2 * live_scores * shape.head_dim / (median * 1e6);
    std::cout << "gfx906," << name(mode) << ',' << name(scope) << ',' << shape.queries << ',' << shape.keys << ','
        << shape.heads << ',' << shape.kv_heads << ',' << shape.head_dim << ',' << kQueryTile << ',' << kKeyTile << ',' << kThreads << ','
        << batch << ',' << samples.size() << ',' << median << ',' << quantile(samples, .1) << ',' << quantile(samples, .9) << ','
        << tops << ',' << error.relative_l2 << ',' << error.maximum_absolute << ',' << error.encoded_equation_error << ',' << error.observed_scores << ','
        << (mode == Precision::Q16 ? llaminar2::cpu::q16AttentionQueryLimit(shape.head_dim) : 0) << '\n';
}

/**
 * @brief Run paired scopes or one exact profiling candidate at a head width.
 * @tparam Dim Supported physical head width.
 * @param shape Immutable live geometry.
 * @param options Timing/profiler contract.
 */
template<int Dim> void run(Shape shape, const Options &options)
{
    const Inputs inputs(shape);
    Storage<float> query(inputs.query.size()), key(inputs.key.size());
    // Execution expires before original inputs. Each candidate explicitly
    // retires its graph consumers before its encoded buffers expire.
    Execution execution;
    checked(hipMemcpyAsync(query.data(), inputs.query.data(), inputs.query.size() * sizeof(float), hipMemcpyHostToDevice, execution.stream), "upload common FP32 queries");
    checked(hipMemcpyAsync(key.data(), inputs.key.data(), inputs.key.size() * sizeof(float), hipMemcpyHostToDevice, execution.stream), "upload common FP32 keys");
    execution.observe();

    const auto single = [&]<Precision Mode>()
    {
        Candidate<Mode, Dim> candidate(shape, inputs, query.data(), key.data(), execution);
        if (options.profile)
        {
            const auto graph = execution.capture([&] { candidate.launch(options.profile_scope); }, 1);
            checked(hipGraphLaunch(graph, execution.stream), "launch isolated profiling transaction");
            execution.observe();
            std::cerr << "Profiled one " << name(Mode) << ' ' << name(options.profile_scope) << " transaction\n";
            return;
        }
        for (const Scope scope : {Scope::PreparedQK, Scope::Preparation, Scope::Pipeline})
        {
            const auto graph = execution.capture([&] { candidate.launch(scope); }, options.batch);
            for (int repeat = 0; repeat < options.warmups; ++repeat) (void)execution.sample(graph, options.batch);
            std::vector<double> samples;
            for (int sample = 0; sample < options.samples; ++sample) samples.push_back(execution.sample(graph, options.batch));
            // Preparation has no scoring node. Observe one scoring launch only
            // outside the timing cohort before checking its prepared operands.
            candidate.launch(Scope::PreparedQK);
            const auto error = candidate.verify(inputs);
            report(shape, Mode, scope, std::move(samples), error, options.batch);
        }
    };
    if (options.single_mode)
    {
        if (options.mode == Precision::FP16) single.template operator()<Precision::FP16>();
        else single.template operator()<Precision::Q16>();
    }
    else
    {
        Candidate<Precision::FP16, Dim> fp16(shape, inputs, query.data(), key.data(), execution);
        Candidate<Precision::Q16, Dim> q16(shape, inputs, query.data(), key.data(), execution);
        for (const Scope scope : {Scope::PreparedQK, Scope::Preparation, Scope::Pipeline})
        {
            const auto half_graph = execution.capture([&] { fp16.launch(scope); }, options.batch);
            const auto integer_graph = execution.capture([&] { q16.launch(scope); }, options.batch);
            for (int repeat = 0; repeat < options.warmups; ++repeat)
            {
                (void)execution.sample(half_graph, options.batch);
                (void)execution.sample(integer_graph, options.batch);
            }
            std::vector<double> half_samples, integer_samples;
            for (int sample = 0; sample < options.samples; ++sample)
            {
                // Alternation balances temperature/clock drift instead of
                // measuring a whole FP16 cohort before a whole Q16 cohort.
                if (sample % 2 == 0)
                {
                    half_samples.push_back(execution.sample(half_graph, options.batch));
                    integer_samples.push_back(execution.sample(integer_graph, options.batch));
                }
                else
                {
                    integer_samples.push_back(execution.sample(integer_graph, options.batch));
                    half_samples.push_back(execution.sample(half_graph, options.batch));
                }
            }
            fp16.launch(Scope::PreparedQK); q16.launch(Scope::PreparedQK);
            const auto half_error = fp16.verify(inputs), integer_error = q16.verify(inputs);
            report(shape, Precision::FP16, scope, std::move(half_samples), half_error, options.batch);
            report(shape, Precision::Q16, scope, std::move(integer_samples), integer_error, options.batch);
        }
    }
    execution.observe();
}
} // namespace mi50_qk_bench

/**
 * @brief Admit an actual gfx906 device and run the declared diagnostic cohort.
 * @param argc Command-line argument count.
 * @param argv Shape/mode/timing selectors.
 * @return Zero only after native launches and numerical checks succeed.
 */
int main(int argc, char **argv)
{
    using namespace mi50_qk_bench;
    try
    {
        const auto options = parse(argc, argv);
        llaminar2::requireROCmRuntimeStartup();
        int count = 0;
        checked(hipGetDeviceCount(&count), "discover MI50 devices");
        if (options.device >= count) throw std::invalid_argument("Requested MI50 device is unavailable");
        checked(hipSetDevice(options.device), "bind exact MI50 device");
        hipDeviceProp_t properties{};
        checked(hipGetDeviceProperties(&properties, options.device), "read MI50 identity");
        if (std::string_view(properties.gcnArchName).substr(0, 6) != "gfx906")
            throw std::invalid_argument("This benchmark requires native gfx906 fdot2 and sdot2; no other arithmetic mode is substituted");
        std::cerr << "device=" << options.device << " name=" << properties.name << " architecture=" << properties.gcnArchName
            << " CUs=" << properties.multiProcessorCount << " scope=causal_QK_only\n";
        std::cout << std::setprecision(10);
        if (!options.profile)
            std::cout << "architecture,mode,scope,queries,keys,heads,kv_heads,head_dim,tile_q,tile_k,threads,graph_batch,samples,median_us,p10_us,p90_us,useful_qk_tops,sampled_relative_l2,sampled_max_abs_error,encoded_equation_max_abs_error,checked_scores,query_code_limit\n";
        const std::vector<Shape> sweep = options.exact_shape || options.profile ? std::vector<Shape>{options.shape} : std::vector<Shape>{
            {17, 65, 7, 1, 64}, {128, 128, 14, 2, 64}, {425, 425, 14, 2, 64},
            {512, 512, 14, 2, 64}, {2048, 2048, 14, 2, 64}, {128, 8192, 7, 1, 64},
            {17, 65, 8, 1, 128}, {128, 128, 16, 2, 128}, {425, 425, 16, 2, 128},
            {512, 512, 16, 2, 128}, {2048, 2048, 16, 2, 128}, {128, 8192, 8, 1, 128}};
        for (const auto shape : sweep)
        {
            validate(shape);
            if (shape.head_dim == 64) run<64>(shape, options);
            else run<128>(shape, options);
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "MI50 QK benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
