/**
 * @file W4A4ReplayTransport.h
 * @brief Bounded transport of immutable W4A4 observations to production candidates.
 *
 * Python authenticates the pinned recording and owns expected bytes. This
 * transport carries only recorded inputs and explicit candidate configuration;
 * no expectation is computed here. Entry IDs distinguish arithmetic/team
 * execution from the actual single/fused tensor interfaces, so substituting a
 * lower-level executable cannot masquerade as an engine replay certificate.
 */
#pragma once
#include "tensors/W4A4MatrixContract.h"
#include <array>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace llaminar2::w4a4::test_transport
{
    inline constexpr std::uint32_t kMagic = 0x52344157;
    inline constexpr std::size_t kMaximumBytes = 1u << 22;

    /** @brief Explicitly named candidate path, not an inferred executable capability. */
    enum class Entry : std::uint32_t { Arithmetic = 0, ExistingTeam = 1, Tensor = 2, FusedTensor = 3 };

    /** @brief Complete validated recorded input; it owns the original bytes. */
    struct Record
    {
        std::array<std::uint32_t, 7> header;
        std::vector<SourceBlock> weights;
        std::vector<float> input;
        /** @return Complete output columns. */
        int n() const { return static_cast<int>(header[3]); }
        /** @return Complete reduction columns. */
        int k() const { return static_cast<int>(header[4]); }
        /** @return Original recorded input rows. */
        int m() const { return static_cast<int>(header[5]); }
        /** @return Explicit candidate path, authenticated again by the particular driver. */
        Entry entry() const { return static_cast<Entry>(header[6]); }
        /** @return Source-declared arithmetic transform. */
        Transform transform() const { return header[2] == 44 ? Transform::Identity : Transform::PaleySylvester; }
    };

    /** @brief Require a complete bounded payload.
     * @param stream Owned binary input transport.
     * @param destination Previously admitted fixture storage.
     * @param bytes Exact payload extent.
     * @throws std::runtime_error For incomplete or unbounded data.
     */
    inline void readExact(std::istream &stream, void *destination, std::size_t bytes)
    {
        if (bytes > kMaximumBytes) throw std::runtime_error("Replay input exceeds the bounded fixture contract");
        if (bytes && !stream.read(static_cast<char *>(destination), static_cast<std::streamsize>(bytes)))
            throw std::runtime_error("Replay input is incomplete");
    }

    /** @brief Return observed bytes, never generated expectations.
     * @param stream Owned binary result transport.
     * @param source Actual completed production state.
     * @param bytes Exact observed extent.
     * @throws std::runtime_error For unbounded or failed publication.
     */
    inline void writeExact(std::ostream &stream, const void *source, std::size_t bytes)
    {
        if (bytes > kMaximumBytes) throw std::runtime_error("Replay output exceeds the bounded fixture contract");
        if (bytes && !stream.write(static_cast<const char *>(source), static_cast<std::streamsize>(bytes)))
            throw std::runtime_error("Replay result publication failed");
    }

    /** @brief Admit a tiny fixture vector before allocation.
     * @tparam T Recorded element type.
     * @param elements Exact bounded element count.
     * @return Owned fixture-only storage; production kernels borrow their own admitted arenas.
     * @throws std::runtime_error For an excessive extent.
     */
    template<class T>
    std::vector<T> storage(std::size_t elements)
    {
        if (elements > kMaximumBytes / sizeof(T)) throw std::runtime_error("Replay geometry exceeds the fixture bound");
        return std::vector<T>(elements);
    }

    /** @brief Decode complete recorded source/input bytes with no trailing observations.
     * @param path Owned temporary input transport path.
     * @return Bounded original record; the driver separately admits source semantics.
     * @throws std::runtime_error For incompatible version, shape, path or incomplete payload.
     */
    inline Record read(const char *path)
    {
        std::ifstream stream(path, std::ios::binary);
        Record record;
        readExact(stream, record.header.data(), sizeof(record.header));
        const auto [magic, version, type, n, k, m, entry] = record.header;
        if (magic != kMagic || version != kArithmeticVersion || (type != 44 && type != 45)
            || !n || !k || !m || n > std::numeric_limits<int>::max() || k > std::numeric_limits<int>::max()
            || m > std::numeric_limits<int>::max() || entry > static_cast<unsigned>(Entry::FusedTensor))
            throw std::runtime_error("Incompatible replay policy or geometry");
        const auto extent = MatrixContract::sourceExtent(n, k);
        if (!extent) throw std::runtime_error(extent.error().message());
        record.weights = storage<SourceBlock>(*extent / sizeof(SourceBlock));
        if (m > kMaximumBytes / sizeof(float) / k)
            throw std::runtime_error("Replay input geometry exceeds the fixture bound");
        record.input = storage<float>(static_cast<std::size_t>(m) * k);
        readExact(stream, record.weights.data(), record.weights.size() * sizeof(SourceBlock));
        readExact(stream, record.input.data(), record.input.size() * sizeof(float));
        if (stream.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Trailing replay input bytes");
        return record;
    }
} // namespace llaminar2::w4a4::test_transport
