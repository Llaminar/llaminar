/**
 * @file CPUGDNStateLayout.h
 * @brief Sole physical layout and arithmetic identity of CPU GDN epoch state.
 *
 * A head owns a rounded checkpoint, its position within an eight-update epoch,
 * and prepared raw updates. Acceptance, request banks and durable snapshots copy
 * this complete value. Only an eighth accepted update advances the checkpoint;
 * materializing a partial epoch would change subsequent output bytes.
 * The layout is shared by runtime storage, workspace and physical admission.
 */
#pragma once

#include <bit>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace llaminar2
{
    /** @brief Distinguish device matrix state from CPU checkpoint/history bytes. */
    enum class GDNStateStorage { DeviceDense, CPUFixedEpoch };

    /** @brief Validated geometry for complete CPU recurrence state. */
    class CPUGDNStateLayout final
    {
    public:
        static constexpr int kPeriod = 8; ///< Fixed arithmetic boundary, never a tuning dispatch.
        static constexpr std::string_view kArithmetic = "fixed-epoch8-backward-fma-v1";

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

    private:
        int heads_, dk_, dv_; ///< Validated immutable logical geometry.
        int matrix_, record_, head_; ///< Derived physical extents, never live accounting.
    };

    /**
     * @brief Resolve backend-specific state extent for allocation and publication.
     * @param storage Explicit CPU history or device dense representation.
     * @param heads Positive value-head count.
     * @param dk Positive key width.
     * @param dv Positive value width.
     * @return Complete FP32 element extent accepted by the kernel ABI.
     */
    [[nodiscard]] inline int gdnRecurrenceStateFloats(GDNStateStorage storage, int heads, int dk, int dv)
    {
        if (storage == GDNStateStorage::CPUFixedEpoch)
            return CPUGDNStateLayout(heads, dk, dv).floats();
        if (heads <= 0 || dk <= 0 || dv <= 0 ||
            static_cast<std::size_t>(heads) > static_cast<std::size_t>(std::numeric_limits<int>::max()) / dk / dv)
            throw std::invalid_argument("Device GDN state geometry exceeds positive INT extent");
        return heads * dk * dv;
    }
}
