/**
 * @file RoundTripContract.h
 * @brief Device-free geometry and exact-data oracle for the two-host transport experiment.
 *
 * This opt-in component experiment is not a production transport or admission
 * ledger. Every invocation names one allocation policy explicitly; eligibility
 * failure never selects another policy. Values are small exact integers in FP32
 * so a stale epoch, partial transfer or incorrect reduction cannot pass through
 * a floating tolerance. The two-rank fixture does not constrain engine topology.
 */
#pragma once
#include <algorithm>
#include <charconv>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace llaminar2::mpi_roundtrip {
/** @brief Independently selected physical allocation, never an automatic fallback. */
enum class Allocation { Mapped, WriteCombined, Registered, Device };
/** @brief Reject an unknown candidate before allocating or entering a collective. */
inline Allocation parseAllocation(std::string_view text) {
    if (text == "mapped") return Allocation::Mapped;
    if (text == "write-combined") return Allocation::WriteCombined;
    if (text == "registered") return Allocation::Registered;
    if (text == "device") return Allocation::Device;
    throw std::invalid_argument("candidate must be mapped, write-combined, registered or device");
}
/** @brief Explicit data-plane choice, independent of backing allocation. */
enum class DataPlane { MPI, VerbsOrdinary, VerbsDmaBuf };
/** @brief One requested allocation/data-plane pair; no substitution on admission failure. */
struct RunPolicy { Allocation allocation; DataPlane plane; };
/** @brief Parse the experiment's closed candidate names into orthogonal policies. */
inline RunPolicy parseRunPolicy(std::string_view text) {
    if (text == "mapped-verbs") return {Allocation::Mapped, DataPlane::VerbsOrdinary};
    if (text == "write-combined-verbs") return {Allocation::WriteCombined, DataPlane::VerbsOrdinary};
    if (text == "mapped-dmabuf") return {Allocation::Mapped, DataPlane::VerbsDmaBuf};
    if (text == "write-combined-dmabuf") return {Allocation::WriteCombined, DataPlane::VerbsDmaBuf};
    try { return {parseAllocation(text), DataPlane::MPI}; }
    catch (const std::invalid_argument&) {
        throw std::invalid_argument("candidate must be mapped, write-combined, registered, device, mapped-verbs, write-combined-verbs, mapped-dmabuf or write-combined-dmabuf");
    }
}
/** @brief Independent registration mechanism, never a priority/fallback list. */
enum class VerbsRegistration { Ordinary, DmaBuf };
/** @brief Validate rdma-core's MR metadata: DMA-BUF addr is FD offset, not submitted IOVA. */
inline void validateNativeMr(VerbsRegistration registration, std::uintptr_t reportedAddress,
                             std::size_t reportedLength, std::uintptr_t originalAddress,
                             std::size_t allocationBytes, std::uint64_t fdOffset = 0) {
    const auto expected = registration == VerbsRegistration::DmaBuf ? fdOffset : originalAddress;
    if (reportedAddress != expected || reportedLength != allocationBytes)
        throw std::runtime_error("native MR metadata differs from its allocation/FD-range contract");
}
/** @brief Preserve native failure codes without allocating for successful hot-path checks. */
inline void checkNativeStatus(int status, std::string_view operation) {
    if (status) throw std::runtime_error(std::string(operation) + ": native status " + std::to_string(status));
}
/** @brief Retain native receive completion even when it precedes previous-send retirement. */
class TransportReceiptFrontier {
    unsigned completed_ = 0;
public:
    /** @brief Accept only the next original epoch; duplicates and gaps are fatal. */
    void observe(unsigned epoch) {
        if (epoch != completed_ + 1) throw std::runtime_error("stale, duplicate or skipped receive epoch");
        completed_ = epoch;
    }
    /** @return Whether this epoch's remote write has an authenticated native completion. */
    bool ready(unsigned epoch) const { return completed_ >= epoch; }
};
/** @brief Parse positive decimal input without accepting signs, suffixes or overflow. */
inline std::size_t positive(std::string_view text) {
    std::size_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !value)
        throw std::invalid_argument("expected a positive decimal integer");
    return value;
}
/** @brief Exact live message geometry; guard storage is never transmitted. */
struct Geometry {
    static constexpr std::size_t guardWords = 64;
    std::size_t words;
    /** @brief Admit a nonempty FP32 message representable by this MPI count API. */
    explicit Geometry(std::size_t bytes) : words(bytes / sizeof(float)) {
        if (bytes == 0 || bytes % sizeof(float) || words > INT_MAX)
            throw std::invalid_argument("payload requires positive whole FP32 words within MPI count range");
    }
    /** @return Native storage extent including independent prefix and suffix guards. */
    std::size_t allocationBytes() const { return (words + 2 * guardWords) * sizeof(float); }
    /** @return Page-rounded backing for matched native-MR and DMA-BUF experiments. */
    std::size_t alignedAllocationBytes(std::size_t alignment) const {
        if (!alignment || (alignment & (alignment - 1))) throw std::invalid_argument("invalid native allocation alignment");
        const auto bytes = allocationBytes();
        if (bytes > SIZE_MAX - (alignment - 1)) throw std::invalid_argument("aligned allocation overflow");
        return (bytes + alignment - 1) / alignment * alignment;
    }
    /** @return A useful cooperative grid, limited by actual resident hardware workers. */
    unsigned blocks(unsigned threads, unsigned residentWorkers) const {
        if (!threads || !residentWorkers) throw std::invalid_argument("missing native worker geometry");
        return static_cast<unsigned>(std::min<std::size_t>((words + threads - 1) / threads, residentWorkers));
    }
};
/** @brief One disjoint, nonempty FP32 slice; registration capacity is not its extent. */
struct LiveRailSlice { std::size_t offsetWords, words; };
/** @brief Sealed balanced striping across only rails with actual live work. */
class LiveRailPlan {
    std::size_t words_, rails_;
public:
    /** @brief Admit positive requested membership, bounding active rails by payload words. */
    LiveRailPlan(Geometry geometry, std::size_t requested) : words_(geometry.words), rails_(std::min(words_, requested)) {
        if (!requested) throw std::invalid_argument("missing explicit rail membership");
    }
    /** @return Number of rails with a nonempty live slice. */
    std::size_t activeRails() const { return rails_; }
    /** @return One exact slice, rejecting indices outside admitted membership. */
    LiveRailSlice slice(std::size_t index) const {
        if (index >= rails_) throw std::invalid_argument("rail outside live membership");
        const auto quotient = words_ / rails_, remainder = words_ % rails_;
        return {index * quotient + std::min(index, remainder), quotient + (index < remainder)};
    }
};
#if defined(__CUDACC__)
#define ROUNDTRIP_HD __host__ __device__
#else
#define ROUNDTRIP_HD
#endif
/** @return Source value with both row identity and transaction generation. */
ROUNDTRIP_HD inline float sourceValue(std::size_t i, unsigned long long epoch) {
    return static_cast<float>((i * 3 + epoch * 17) % 2048);
}
/** @return The second rank's exactly representable local contribution. */
ROUNDTRIP_HD inline float peerValue(std::size_t i, unsigned long long epoch) {
    return static_cast<float>((i * 5 + epoch * 13) % 1024 + 1);
}
/** @return Canonical two-source FP32 sum, identical on both physical ranks. */
ROUNDTRIP_HD inline float reducedValue(std::size_t i, unsigned long long epoch) {
    return sourceValue(i, epoch) + peerValue(i, epoch);
}
#undef ROUNDTRIP_HD
} // namespace llaminar2::mpi_roundtrip
