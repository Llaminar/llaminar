/**
 * @file CPUGDNStateLayout.h
 * @brief Sole physical layout and arithmetic identity of CPU GDN epoch state.
 *
 * A head owns a rounded checkpoint, its position within an eight-update epoch,
 * and prepared raw updates. Acceptance, request banks and local rollback copy
 * the complete physical image. Durable prefix records serialize the original
 * phase/checkpoint/live history and restore inactive history to canonical zero. Only an eighth accepted update advances the checkpoint;
 * materializing a partial epoch would change subsequent output bytes.
 * The layout is shared by runtime storage, workspace and physical admission.
 * Extent admission accepts only the two named representations; unknown stored
 * IDs must never be interpreted as a different physical state format.
 */
#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <span>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace llaminar2
{
    /** @brief Distinguish device matrix state from CPU checkpoint/history bytes. */
    enum class GDNStateStorage { DeviceDense, CPUFixedEpoch };

    /** @brief Closed host wire formats; local images are never portable prefix records. */
    enum class HybridPrefixHostEncoding : std::uint32_t
    {
        None = 0, ///< Unspecified/legacy; valid only when no CPU GDN host section exists.
        CPUFixedEpochLocalImage = 1, ///< Internal complete physical rollback image.
        CPUFixedEpochPackedV2 = 2 ///< Phase/checkpoint/live-history durable serialization.
    };

    /** @brief Bounded CPU durable-section header; no payload checksum or live accounting. */
    struct CPUGDNPrefixSectionHeader
    {
        std::uint64_t magic;
        std::uint32_t version;
        std::int32_t first_layer;
        std::uint32_t total_layers;
        std::uint32_t gdn_layers;
        std::uint64_t bytes;
    };
    /** @brief Immutable layer-order/geometry declaration preceding packed heads and convolution. */
    struct CPUGDNPrefixLayerHeader
    {
        std::int32_t global_layer;
        std::uint32_t heads;
        std::uint32_t key_width;
        std::uint32_t value_width;
        std::uint64_t recurrence_bytes;
        std::uint64_t conv_bytes;
    };
    static_assert(sizeof(CPUGDNPrefixSectionHeader) == 32 && sizeof(CPUGDNPrefixLayerHeader) == 32);

    /** @brief Validated geometry for complete CPU recurrence state. */
    class CPUGDNStateLayout final
    {
    public:
        static constexpr int kPeriod = 8; ///< Fixed arithmetic boundary, never a tuning dispatch.
        static constexpr std::string_view kArithmetic = "fixed-epoch8-backward-fma-v1";
        static constexpr std::uint64_t kPrefixMagic = 0x324b4341504e4447ULL; ///< GDN packed V2 identity.
        static constexpr std::uint32_t kPrefixVersion = 2; ///< CPU-host section version only.

        /**
         * @brief Resolve one CPU state bank before allocating or accessing it.
         * @param heads Positive value-head count.
         * @param dk Key width in the CPU kernel's admitted range 1..512.
         * @param dv Value width in the CPU kernel's admitted range 1..512.
         * @throws std::invalid_argument for unsupported geometry.
         * @throws std::overflow_error if the complete bank exceeds the INT ABI.
         */
        constexpr CPUGDNStateLayout(int heads, int dk, int dv) : heads_(heads), dk_(dk), dv_(dv)
        {
            if (heads <= 0 || dk <= 0 || dv <= 0 || dk > 512 || dv > 512)
                throw std::invalid_argument("CPU GDN epoch state requires positive heads and widths 1..512");
            matrix_ = dk * dv;
            record_ = dk + dv + 2;
            head_ = 1 + matrix_ + kPeriod * record_;
            if (heads > std::numeric_limits<int>::max() / head_)
                throw std::overflow_error("CPU GDN epoch state exceeds INT element ABI");
        }

        /** @return Complete bank extent, including phase and inactive history. */
        [[nodiscard]] constexpr int floats() const noexcept { return heads_ * head_; }
        /** @return Complete per-head physical stride. */
        [[nodiscard]] constexpr int headFloats() const noexcept { return head_; }
        /** @return Dense checkpoint extent per head. */
        [[nodiscard]] int matrixFloats() const noexcept { return matrix_; }
        /** @return Prepared key/value/decay/beta extent of one raw update. */
        [[nodiscard]] int recordFloats() const noexcept { return record_; }
        /** @return Per-head key width. */
        [[nodiscard]] int keyWidth() const noexcept { return dk_; }
        /** @return Per-head value width. */
        [[nodiscard]] int valueWidth() const noexcept { return dv_; }
        /** @return Mutable raw record; callers use indices 0..7 only. */
        [[nodiscard]] float *record(float *head, int index) const noexcept
        { return head + 1 + matrix_ + index * record_; }
        /** @return Immutable raw record; callers use indices 0..7 only. */
        [[nodiscard]] const float *record(const float *head, int index) const noexcept
        { return head + 1 + matrix_ + index * record_; }
        /** @return Whether all live phases are valid before any output is mutated. */
        [[nodiscard]] bool valid(const float *state) const noexcept
        {
            if (!state) return false;
            for (int h = 0; h < heads_; ++h)
            {
                const float phase = state[static_cast<std::size_t>(h) * head_];
                // Bit admission remains valid under the core's fast-math flags:
                // reject NaNs/infinities/sign bits before any float-to-int cast.
                if (std::bit_cast<std::uint32_t>(phase) > 0x40e00000u ||
                    phase != static_cast<float>(static_cast<int>(phase)))
                    return false;
            }
            return true;
        }

        /** @return Number of logical value heads, independent of any serialized phase. */
        [[nodiscard]] constexpr int heads() const noexcept { return heads_; }

        /**
         * @brief Decode only a canonical FP32 integer phase word, without unsafe NaN casts.
         * @param word Original phase bits from a bounded source or live host owner.
         * @return Phase0..7, or -1 for any other representation (including negative zero).
         */
        [[nodiscard]] static int prefixPhase(std::uint32_t word) noexcept
        {
            for (int phase = 0; phase < kPeriod; ++phase)
                if (word == std::bit_cast<std::uint32_t>(static_cast<float>(phase))) return phase;
            return -1;
        }

        /**
         * @brief Size one contiguous phase/checkpoint/live-record prefix, excluding inactive slots.
         * @param phase Valid original epoch phase0..7, never inferred from token count.
         * @return Exact serialized bytes of this head.
         * @throws std::invalid_argument For an invalid phase.
         */
        [[nodiscard]] size_t packedHeadBytes(int phase) const
        {
            if (phase < 0 || phase >= kPeriod) throw std::invalid_argument("Invalid CPU GDN prefix phase");
            return (size_t{1} + static_cast<size_t>(matrix_) + static_cast<size_t>(phase) * record_) * sizeof(float);
        }

        /** @return Phase-independent maximum packed recurrence extent; not a live transfer count. */
        [[nodiscard]] size_t packedPrefixCapacityBytes() const
        { return static_cast<size_t>(heads_) * packedHeadBytes(kPeriod - 1); }

        /**
         * @brief Project the exact packed extent from the original immutable host phases.
         * @param state Complete state bank whose capacity was checked by its owner.
         * @return Exact extent including every head's original phase word and checkpoint.
         * @throws std::invalid_argument For invalid live phase metadata.
         */
        [[nodiscard]] size_t packedPrefixBytes(const float *state) const
        {
            if (!valid(state)) throw std::invalid_argument("Invalid CPU GDN prefix source phase");
            size_t bytes = 0;
            for (int head = 0; head < heads_; ++head)
                bytes += packedHeadBytes(prefixPhase(std::bit_cast<std::uint32_t>(
                    state[static_cast<size_t>(head) * head_])));
            return bytes;
        }

        /**
         * @brief Validate an exact bounded recurrence section without mutation or matrix materialization.
         * @param bytes Complete packed head sequence; trailing/incomplete bytes are rejected.
         * @return True only for all heads and valid phases, with no inactive record capacity.
         */
        [[nodiscard]] bool validPackedPrefix(std::span<const std::byte> bytes) const noexcept
        {
            size_t cursor = 0;
            for (int head = 0; head < heads_; ++head)
            {
                if (bytes.size() - cursor < sizeof(std::uint32_t)) return false;
                std::uint32_t word;
                std::memcpy(&word, bytes.data() + cursor, sizeof(word));
                const int phase = prefixPhase(word);
                if (phase < 0) return false;
                const size_t count = packedHeadBytes(phase);
                if (count > bytes.size() - cursor) return false;
                cursor += count;
            }
            return cursor == bytes.size();
        }

    private:
        int heads_, dk_, dv_; ///< Validated immutable logical geometry.
        int matrix_, record_, head_; ///< Derived physical extents, never live accounting.
    };

    /**
     * @brief Resolve backend-specific state extent for allocation and publication.
     * @param storage Closed CPU history or device dense representation identity.
     * @param heads Positive value-head count.
     * @param dk Positive key width.
     * @param dv Positive value width.
     * @return Complete FP32 element extent accepted by the selected kernel ABI.
     * @throws std::invalid_argument For unknown storage or unsupported geometry.
     * @throws std::overflow_error For a CPU epoch bank exceeding its INT ABI.
     */
    [[nodiscard]] inline int gdnRecurrenceStateFloats(GDNStateStorage storage, int heads, int dk, int dv)
    {
        if (storage == GDNStateStorage::CPUFixedEpoch)
            return CPUGDNStateLayout(heads, dk, dv).floats();
        if (storage != GDNStateStorage::DeviceDense)
            throw std::invalid_argument("Unknown GDN recurrence state storage");
        if (heads <= 0 || dk <= 0 || dv <= 0 ||
            static_cast<std::size_t>(heads) > static_cast<std::size_t>(std::numeric_limits<int>::max()) / dk / dv)
            throw std::invalid_argument("Device GDN state geometry exceeds positive INT extent");
        return heads * dk * dv;
    }
}
