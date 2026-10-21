/**
 * @file ImmutableRowTableExtent.h
 * @brief Original native extent shared by whole-tier and mixed-layout metadata.
 *
 * This is unchanged loader-authenticated geometry, not a file identity or an
 * allocation permit. The existing materializer retains the original source/file
 * authority; layout validation neither reads nor hashes quantized payloads.
 */
#pragma once
#include <cstddef>
#include <cstdint>
namespace llaminar2 {
enum class GGUFTensorType : std::uint32_t;
/** @brief Checked geometry of the already validated original native source. */
class ImmutableRowTableExtent final
{
public:
    /**
     * @brief Bind native metadata without reading or hashing the table payload.
     * @param format Original loader-authenticated GGUF format, never a repack type.
     * @param rows Original source rows.
     * @param row_bytes Exact complete-native-block bytes per row from the source authority.
     * @return Checked positive extent; the materializer also authenticates its actual source.
     * @throws std::invalid_argument For an empty extent.
     * @throws std::overflow_error If row/byte products cannot be represented.
     */
    static ImmutableRowTableExtent fromNative(GGUFTensorType format, std::uint64_t rows, std::size_t row_bytes);
    /** @return Original loader-authenticated format identity. */
    GGUFTensorType format() const noexcept { return format_; }
    /** @return Original immutable source row count. */
    std::uint64_t rows() const noexcept { return rows_; }
    /** @return Exact original bytes per complete native row. */
    std::size_t rowBytes() const noexcept { return row_bytes_; }
    /** @return Exact original tensor bytes, independent of physical storage kind. */
    std::size_t nativeBytes() const noexcept { return native_bytes_; }
    bool operator==(const ImmutableRowTableExtent &) const = default;
private:
    /** @brief Only checked source-geometry construction may mint an extent. */
    ImmutableRowTableExtent(GGUFTensorType format, std::uint64_t rows, std::size_t row_bytes, std::size_t bytes)
        : format_(format), rows_(rows), row_bytes_(row_bytes), native_bytes_(bytes) {}
    GGUFTensorType format_;
    std::uint64_t rows_;
    std::size_t row_bytes_, native_bytes_;
};
}
