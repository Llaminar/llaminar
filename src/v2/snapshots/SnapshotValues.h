/**
 * @file SnapshotValues.h
 * @brief Diagnostic scalar payloads that retain exact signed 64-bit identities.
 *
 * One immutable publication owns either FP32 numerical values or INT64 identity
 * values. Legacy float views reject integer payloads; no identifier is rounded
 * through float or double. This type has no role in inference execution state.
 */
#pragma once
#include <cstdint>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace llaminar2
{
/** @brief Typed non-owning diagnostic publication; lifetime belongs to its snapshot handle. */
using SnapshotValueView = std::variant<std::span<const float>, std::span<const std::int64_t>>;

/** @brief One owned diagnostic payload with checked legacy floating access. */
class SnapshotValues
{
public:
    /** @brief Create an empty floating publication. */
    SnapshotValues() = default;
    /** @brief Adopt floating observations without copying. */
    SnapshotValues(std::vector<float> values) : values_(std::move(values)) {}
    /** @brief Adopt exact integer observations without copying or conversion. */
    SnapshotValues(std::vector<std::int64_t> values) : values_(std::move(values)) {}
    /** @return Whether this payload contains exact integer identities. */
    [[nodiscard]] bool isInteger() const noexcept { return std::holds_alternative<std::vector<std::int64_t>>(values_); }
    /** @return Logical scalar count, independent of scalar width. */
    [[nodiscard]] std::size_t size() const noexcept { return std::visit([](const auto &v) { return v.size(); }, values_); }
    /** @return Whether the publication has no scalars. */
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    /** @return Floating payload; integer access is an explicit diagnostic error. */
    [[nodiscard]] const std::vector<float> &floats() const
    {
        if (isInteger()) throw std::logic_error("integer snapshot requires typed access; floating conversion is forbidden");
        return std::get<std::vector<float>>(values_);
    }
    /** @return Exact signed integer payload; rejects numerical floating publications. */
    [[nodiscard]] const std::vector<std::int64_t> &integers() const
    {
        if (!isInteger()) throw std::logic_error("floating snapshot cannot be interpreted as integer identities");
        return std::get<std::vector<std::int64_t>>(values_);
    }
    /** @return Typed immutable view backed by this publication. */
    [[nodiscard]] SnapshotValueView view() const
    {
        if (isInteger()) return std::span<const std::int64_t>(integers());
        return std::span<const float>(floats());
    }
    /** @return Checked legacy floating pointer. */
    [[nodiscard]] const float *data() const { return floats().data(); }
    /** @return Checked legacy floating scalar. */
    [[nodiscard]] float operator[](std::size_t index) const { return floats()[index]; }
    /** @return Checked legacy floating begin iterator. */
    [[nodiscard]] auto begin() const { return floats().begin(); }
    /** @return Checked legacy floating end iterator. */
    [[nodiscard]] auto end() const { return floats().end(); }
    /** @brief Append live scalars while preserving the publication's scalar type. */
    void append(const SnapshotValues &piece, std::size_t count)
    {
        if (isInteger() != piece.isInteger() || count > piece.size())
            throw std::logic_error("snapshot sequence has mixed scalar types or invalid live extent");
        std::visit([&](auto &destination) {
            using Vector = std::decay_t<decltype(destination)>;
            const auto &source = std::get<Vector>(piece.values_);
            destination.insert(destination.end(), source.begin(), source.begin() + count);
        }, values_);
    }
    /** @brief Preserve existing floating test comparisons without accepting integer reinterpretation. */
    friend bool operator==(const SnapshotValues &left, const std::vector<float> &right) { return left.floats() == right; }
private:
    std::variant<std::vector<float>, std::vector<std::int64_t>> values_;
};
} // namespace llaminar2
