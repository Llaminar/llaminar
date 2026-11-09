/**
 * @file CPUAttentionInvocationAdmission.h
 * @brief One ISA-neutral admission of a CPU attention invocation's banks and geometry.
 *
 * The flash-attention row programs address query, output and K/V rows with
 * plain pointer arithmetic, on every ISA. Their safety follows from facts
 * about the whole invocation, not about one row group: every addressed row
 * lies inside a bank of sufficient extent, every key and query row has the
 * same head width, the head mapping stays inside the declared K/V heads, and
 * the output bank is disjoint from every input bank. This header proves those
 * facts once, before any worker team exists, so the row programs themselves
 * carry no per-call validation (and cannot throw inside a team).
 *
 * Banks are synchronous borrows; their owners retain storage for the call.
 * Tensor entry points admit their actual allocation extents; raw-pointer entry
 * points admit the extents their geometry declares. Core-free.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace llaminar2::cpu
{
/** @brief One borrowed byte range of an attention invocation. */
struct CPUAttentionBank
{
    const void *data = nullptr; ///< First byte; must be non-null when the invocation reads/writes it.
    std::size_t bytes = 0;      ///< Live extent available to the invocation.
};

/** @brief Complete declared geometry of one invocation, before admission. */
struct CPUAttentionInvocationRequest
{
    CPUAttentionBank query;  ///< FP32 query rows.
    CPUAttentionBank output; ///< FP32 output rows, written by the invocation.
    CPUAttentionBank key;    ///< Physical K rows in their native encoding.
    CPUAttentionBank value;  ///< Physical V rows in their native encoding.
    int query_rows = 0;           ///< Query positions.
    int kv_physical_rows = 0;     ///< Physical K/V rows any logical row can map to.
    int query_heads = 0;          ///< Local query heads per query row.
    int kv_heads = 0;             ///< K/V heads per physical row.
    int head_dim = 0;             ///< Elements per query/key/value head.
    int head_start = 0;           ///< First global query head when sharded with gqa_n_rep.
    int gqa_n_rep = 0;            ///< Explicit query heads per K/V head; zero derives it.
    std::size_t key_row_bytes = 0;   ///< Bytes of one physical K row (all K/V heads).
    std::size_t value_row_bytes = 0; ///< Bytes of one physical V row (all K/V heads).
};

/**
 * @brief Proof that an invocation's row arithmetic stays inside admitted banks.
 *
 * Only admit() constructs it. Holding one means: dimensions are positive and
 * representable; head_dim is at most kMaxHeadDim; every query head maps to a
 * K/V head below kv_heads; query/output banks hold query_rows rows of
 * query_heads*head_dim FP32 values; K/V banks hold kv_physical_rows rows; no
 * bank is null; and the output bank overlaps neither the query nor the K/V banks.
 */
class CPUAttentionInvocationAdmission final
{
public:
    /// Largest admitted head width; it also bounds one integer Q16 reduction.
    static constexpr int kMaxHeadDim = 65535;

    /**
     * @brief Admit an invocation before any worker team is formed.
     * @param request Declared banks and geometry.
     * @param error Receives the first violated rule.
     * @return The proof, or empty with @p error set.
     */
    [[nodiscard]] static std::optional<CPUAttentionInvocationAdmission> admit(
        const CPUAttentionInvocationRequest &request, std::string *error = nullptr)
    {
        const auto reject = [&](const char *reason) -> std::optional<CPUAttentionInvocationAdmission> {
            if (error) *error = reason;
            return std::nullopt;
        };
        const auto &r = request;
        if (r.query_rows <= 0 || r.kv_physical_rows <= 0 || r.query_heads <= 0 || r.kv_heads <= 0 ||
            r.head_dim <= 0 || r.head_start < 0 || r.gqa_n_rep < 0 || r.key_row_bytes == 0 || r.value_row_bytes == 0)
            return reject("CPU attention invocation requires positive geometry");
        if (r.head_dim > kMaxHeadDim)
            return reject("CPU attention head width exceeds the admitted maximum");
        if (r.gqa_n_rep == 0 && r.query_heads % r.kv_heads != 0)
            return reject("CPU attention query heads must divide evenly over K/V heads");
        const std::int64_t heads_per_kv = r.gqa_n_rep > 0 ? r.gqa_n_rep : r.query_heads / r.kv_heads;
        const std::int64_t last_global_head = (r.gqa_n_rep > 0 ? std::int64_t(r.head_start) : 0) + r.query_heads - 1;
        if (last_global_head / heads_per_kv >= r.kv_heads)
            return reject("CPU attention query heads map beyond the declared K/V heads");
        const auto product = [](std::size_t a, std::size_t b, std::size_t &out) {
            if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
            out = a * b;
            return true;
        };
        std::size_t row_values = 0, query_bytes = 0, key_bytes = 0, value_bytes = 0;
        if (!product(std::size_t(r.query_heads), std::size_t(r.head_dim), row_values) ||
            row_values > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
            !product(row_values * sizeof(float), std::size_t(r.query_rows), query_bytes) ||
            !product(r.key_row_bytes, std::size_t(r.kv_physical_rows), key_bytes) ||
            !product(r.value_row_bytes, std::size_t(r.kv_physical_rows), value_bytes))
            return reject("CPU attention invocation extent is unrepresentable");
        const auto covers = [](const CPUAttentionBank &bank, std::size_t bytes) {
            return bank.data && bank.bytes >= bytes &&
                reinterpret_cast<std::uintptr_t>(bank.data) <= std::numeric_limits<std::uintptr_t>::max() - bank.bytes;
        };
        if (!covers(r.query, query_bytes) || !covers(r.output, query_bytes))
            return reject("CPU attention query/output banks do not cover their declared rows");
        if (!covers(r.key, key_bytes) || !covers(r.value, value_bytes))
            return reject("CPU attention K/V banks do not cover their physical rows");
        const CPUAttentionBank written{r.output.data, query_bytes};
        if (overlaps(written, {r.query.data, query_bytes}) || overlaps(written, {r.key.data, key_bytes}) ||
            overlaps(written, {r.value.data, value_bytes}))
            return reject("CPU attention output overlaps a query or K/V bank");
        return CPUAttentionInvocationAdmission(r.head_dim);
    }

    /** @return Admitted width of every query, key and value head row. */
    [[nodiscard]] int headWidth() const noexcept { return head_dim_; }

    /**
     * @brief Test overlap of two bounded, non-wrapping byte ranges.
     * @return Whether both are nonempty and share a byte.
     */
    [[nodiscard]] static bool overlaps(const CPUAttentionBank &a, const CPUAttentionBank &b) noexcept
    {
        if (!a.bytes || !b.bytes) return false;
        const auto first = reinterpret_cast<std::uintptr_t>(a.data), second = reinterpret_cast<std::uintptr_t>(b.data);
        return first <= second ? second - first < a.bytes : first - second < b.bytes;
    }

private:
    explicit CPUAttentionInvocationAdmission(int head_dim) noexcept : head_dim_(head_dim) {}

    int head_dim_; ///< Admitted head width.
};
} // namespace llaminar2::cpu
