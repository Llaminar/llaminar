/**
 * @file CUDAMoENativeQuantisedDecodeHelpers.cuh
 * @brief Shared CUDA NativeQuantised MoE block decode/dot helpers for grouped prefill and decode shards.
 */
#pragma once
#include "CUDAMoEKernelsCommon.cuh"

namespace
{
    /**
     * @brief Publish one original SwiGLU K32 packet directly from warp registers.
     * @param gate Original rounded per-lane gate projection.
     * @param up Original rounded per-lane up projection.
     * @param codes Destination for the complete packet.
     * @param scale_out Destination for the packet's FP32 scale word.
     * @param lane Lane owning one element of the complete packet.
     */
    __device__ __forceinline__ void publish_decode_swiglu_q8(
        float gate, float up, int8_t *codes, float *scale_out, int lane)
    {
        const float value = llaminar2::device_swiglu_contract::swigluValue(gate, up);
        float maximum = llaminar2::device_q8_activation_contract::maximumAbsoluteValue(value, 0.0f);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
        {
            maximum = llaminar2::device_q8_activation_contract::maximumAbsoluteValue(
                maximum, __shfl_xor_sync(0xffffffffu, maximum, mask));
        }
        const float scale = llaminar2::device_q8_activation_contract::scale(maximum);
        if (lane == 0)
            *scale_out = scale;
        codes[lane] = static_cast<int8_t>(
            llaminar2::device_q8_activation_contract::quantize(value, scale));
    }


    /** @brief Rounding and scheduling of one NativeQuantised expert dot. */
    enum class NativeMoEDotOrder : uint8_t
    {
        PartitionPartial,   ///< The caller publishes a split-K partial.
        DecodeK32,          ///< Direct serial decode publishes ordered K32 terms.
        VerifierK32,        ///< Direct grouped verification publishes the same terms.
        VerifierWideGrid,  ///< Identical verifier order with the wide IQ grid readers.
    };

    constexpr int kWideVerifierGateN = 256;
    constexpr int kWideVerifierGateK = 2560;
    constexpr int kWideVerifierGatePackets = kWideVerifierGateK / 32 /
        llaminar2::CUDAMoEBatchInvariantPolicy::gate_up_k_partitions;
    constexpr int kWideVerifierDownN = 2560;
    constexpr int kWideVerifierDownK = 256;
    constexpr int kShortVerifierDownN = 512;
    constexpr int kShortVerifierDownK = 256;
    constexpr int kVerifierDownTileColumns = 64;
    constexpr int kWideVerifierDownTileColumns = 256;
    static_assert(kWideVerifierGatePackets *
        llaminar2::CUDAMoEBatchInvariantPolicy::gate_up_k_partitions * 32 == kWideVerifierGateK);
    static_assert(kWideVerifierDownN % 4 == 0);

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 860
    /** @brief Read-only IQ1 grid with eight two-bit symbols stored per entry. */
    struct CompactMoEIQ1Grid
    {
        uint16_t words[sizeof(llaminar2::iq1s_ternary_grid) / sizeof(uint16_t)];
    };

    /**
     * @brief Materialize the canonical ternary grid in the device image.
     * @return The exact signed IQ1 grid encoded as 0, 1, and 3.
     */
    consteval CompactMoEIQ1Grid make_compact_moe_iq1_grid()
    {
        CompactMoEIQ1Grid grid{};
        for (size_t index = 0; index < sizeof(grid.words) / sizeof(uint16_t); ++index)
            grid.words[index] = llaminar2::iq1s_ternary_grid[index];
        return grid;
    }

    /**
     * @brief Check the compact representation against every canonical grid entry.
     * @return Whether all symbols expand to the exact original signed bytes.
     */
    consteval bool compact_moe_iq1_grid_is_canonical()
    {
        constexpr auto grid = make_compact_moe_iq1_grid();
        for (size_t index = 0; index < sizeof(grid.words) / sizeof(uint16_t); ++index)
        {
            const uint16_t word = grid.words[index];
            uint64_t expanded = 0;
            for (int lane = 0; lane < 8; ++lane)
            {
                const uint64_t symbol = (word >> (2 * lane)) & 3u;
                if (symbol == 2u)
                    return false;
                expanded |= (symbol == 3 ? 0xffu : symbol) << (8 * lane);
            }
            if (expanded != llaminar2::iq1s_grid[index])
                return false;
        }
        return true;
    }

    static_assert(compact_moe_iq1_grid_is_canonical());
    __device__ const CompactMoEIQ1Grid compact_moe_iq1_grid = make_compact_moe_iq1_grid();
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 860
    /** @brief Eight nibble selectors for positive IQ2 magnitudes in each grid entry. */
    template <size_t Entries>
    struct MoEIQ2SelectorGrid
    {
        uint32_t words[Entries];
    };

    /**
     * @brief Check that the canonical IQ2 grid can use a three-byte magnitude table.
     * @tparam Source Canonical unsigned-magnitude grid.
     * @return Whether every source lane is one of the three supported magnitudes.
     */
    template <const auto &Source>
    consteval bool moe_iq2_grid_uses_known_magnitudes()
    {
        for (const uint64_t word : Source)
            for (unsigned lane = 0; lane < 8; ++lane)
            {
                const uint32_t magnitude = (word >> (8 * lane)) & 0xffu;
                if (magnitude != 8 && magnitude != 25 && magnitude != 43)
                    return false;
            }
        return true;
    }

    /**
     * @brief Encode each canonical IQ2 entry as one packed byte-permutation selector.
     * @tparam Source Canonical unsigned-magnitude grid.
     * @return One four-bit selector per original signed-byte position.
     */
    template <const auto &Source>
    consteval auto make_moe_iq2_selector_grid()
    {
        static_assert(moe_iq2_grid_uses_known_magnitudes<Source>());
        constexpr size_t kEntries = sizeof(Source) / sizeof(Source[0]);
        MoEIQ2SelectorGrid<kEntries> grid{};
        for (size_t index = 0; index < kEntries; ++index)
            for (unsigned lane = 0; lane < 8; ++lane)
            {
                const uint32_t magnitude = (Source[index] >> (8 * lane)) & 0xffu;
                const uint32_t selector = magnitude == 8 ? 0u : (magnitude == 25 ? 1u : 2u);
                grid.words[index] |= selector << (4 * lane);
            }
        return grid;
    }

    __device__ const auto moe_iq2_s_selectors = make_moe_iq2_selector_grid<llaminar2::iq2s_grid>();
    __device__ const auto moe_iq2_xs_selectors = make_moe_iq2_selector_grid<llaminar2::iq2xs_grid>();
    __device__ const auto moe_iq2_xxs_selectors = make_moe_iq2_selector_grid<llaminar2::iq2xxs_grid>();

    /** @brief Aligned register image of the index and sign bytes of an odd-width packet. */
    template <int IndexWords>
    struct MoEOddGridPacket
    {
        static_assert(IndexWords == 1 || IndexWords == 2);
        uint32_t indices[IndexWords];
        uint32_t high_indices;
        uint32_t signs;
    };

    /**
     * @brief Gather one 9- or 13-byte IQ grid packet without byte loads.
     * @tparam IndexWords One index word for IQ2 or two for IQ3.
     * @tparam ClampedSigns Gather the sign word with a single clamped funnel shift.
     * @param payload Original packet in its [K32][column] weight plane.
     * @return Low index words, high index byte, and four sign bytes.
     * @pre Aligned plane base and a captured column count divisible by four
     *      keep all aligned word loads within the physical packet plane.
     */
    template <int IndexWords, bool ClampedSigns = false>
    __device__ __forceinline__ MoEOddGridPacket<IndexWords> load_moe_odd_grid_packet(const uint8_t *payload)
    {
        const unsigned offset = reinterpret_cast<uintptr_t>(payload) & 3u;
        const auto *words = reinterpret_cast<const uint32_t *>(payload - offset);
        const unsigned shift = 8 * offset;
        uint32_t head = words[0];
        MoEOddGridPacket<IndexWords> packet{};
#pragma unroll
        for (int index = 0; index < IndexWords; ++index)
        {
            const uint32_t next = words[index + 1];
            packet.indices[index] = __funnelshift_r(head, next, shift);
            head = next;
        }
        const uint32_t last = words[IndexWords + 1];
        if constexpr (ClampedSigns)
        {
            // The aligned head always holds the high-index byte. For offset
            // three the sign word is the entire last word, so shift 32 must
            // clamp to that word instead of wrapping to the head.
            packet.high_indices = (head >> shift) & 0xffu;
            packet.signs = __funnelshift_rc(head, last, shift + 8);
        }
        else
        {
            const uint32_t high_and_signs = __funnelshift_r(head, last, shift);
            packet.high_indices = high_and_signs & 0xffu;
            packet.signs = (high_and_signs >> 8) | ((last >> shift) << 24);
        }
        return packet;
    }
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 860
    /**
     * @brief Apply one IQ3 sign per byte using a packed mask and carry-free addition.
     * @param grid Four nonzero positive int8 magnitudes from a canonical grid.
     * @param nibble Four original sign bits.
     * @return Four signed magnitudes in their original byte order.
     */
    __host__ __device__ __forceinline__ constexpr uint32_t moe_iq3_apply_signs(
        uint32_t grid, uint32_t nibble)
    {
        const uint32_t low_bits = ((nibble & 0x0fu) * 0x00204081u) & 0x01010101u;
        // Nonzero magnitudes ensure negating a selected byte cannot carry.
        return (grid ^ (low_bits * 0xffu)) + low_bits;
    }

    /**
     * @brief Compare the packed IQ3 program against every scalar byte and sign combination.
     * @tparam Grid Original prepared codebook lookup values.
     * @return True when every byte remains a valid signed operand with no cross-byte carry.
     */
    template <const auto &Grid>
    consteval bool moe_iq3_signs_are_canonical()
    {
        for (const uint32_t word : Grid)
            for (unsigned signs = 0; signs < 16; ++signs)
            {
                uint32_t expected = 0;
                for (unsigned lane = 0; lane < 4; ++lane)
                {
                    const unsigned shift = lane * 8;
                    const auto magnitude = static_cast<uint8_t>(word >> shift);
                    if (magnitude == 0 || magnitude > 127)
                        return false;
                    const auto signed_byte = static_cast<uint8_t>(
                        signs & (1u << lane) ? -static_cast<int>(magnitude) : magnitude);
                    expected |= static_cast<uint32_t>(signed_byte) << shift;
                }
                if (moe_iq3_apply_signs(word, signs) != expected)
                    return false;
            }
        return true;
    }

    static_assert(moe_iq3_signs_are_canonical<llaminar2::iq3s_grid>());
    static_assert(moe_iq3_signs_are_canonical<llaminar2::iq3xxs_grid>());
#endif

    /** @brief Register schedule for extracting and expanding the IQ2 signs. */
    enum class NativeMoEIQ2SignSchedule : uint8_t
    {
        RegisterShifts,
        IndependentProducts,
        ClampedFunnelAndIndependentProducts,
    };

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 860
    /**
     * @brief Deposit eight source sign bits directly into the PRMT sign selectors.
     * @param signs Original byte of IQ2 signs.
     * @return Bit two of each nibble set from its corresponding source bit.
     */
    __host__ __device__ __forceinline__ constexpr uint32_t moe_iq2_independent_sign_selectors(uint32_t signs)
    {
        // Even and odd source positions issue independently. Carries between
        // off-diagonal terms land outside the selected bit positions.
        const uint32_t even = ((signs & 0x55u) * 0x104104u) & 0x04040404u;
        const uint32_t odd = (((signs >> 1) & 0x55u) * 0x1041040u) & 0x40404040u;
        return even | odd;
    }

    /** @brief Verify all original IQ2 sign bytes against the scalar selector layout. */
    consteval bool moe_iq2_independent_sign_selectors_are_canonical()
    {
        for (uint32_t signs = 0; signs < 256; ++signs)
        {
            uint32_t expected = 0;
            for (unsigned lane = 0; lane < 8; ++lane)
                expected |= ((signs >> lane) & 1u) << (4 * lane + 2);
            if (moe_iq2_independent_sign_selectors(signs) != expected)
                return false;
        }
        return true;
    }
    static_assert(moe_iq2_independent_sign_selectors_are_canonical());
#endif

    /**
     * @brief Form the eight packed signed-byte operands of an MoE K32 dot.
     * @tparam CodebookId Physical payload layout admitted for the route.
     * @tparam Order Dot schedule that owns this packet.
     * @tparam SharedIQ3 Read the canonical IQ3 table prepared by this Down CTA.
     * @tparam SignSchedule IQ2's prepared-selector arithmetic schedule.
     * @param payload Address of one prepared packet. Q2_K and Q6_K packets
     *        and IQ2_XXS start on eight-byte boundaries. Q3_K, Q5_0 and
     *        IQ3_XXS start on four-byte boundaries.
     *        IQ1 packets start on two-byte boundaries.
     * @param groups Decoded operands in the original DP4A order.
     * @param iq3_table Shared canonical IQ3 words when the CTA owns their copy.
     */
    template <uint8_t CodebookId, NativeMoEDotOrder Order, bool SharedIQ3 = false,
              NativeMoEIQ2SignSchedule SignSchedule = NativeMoEIQ2SignSchedule::RegisterShifts>
    __device__ __forceinline__ void decode_moe_native_quantised_packet(
        const uint8_t *payload, int32_t (&groups)[8], const uint32_t *iq3_table = nullptr)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860
        const auto negative_bytes = [](uint32_t bits, uint32_t mask) {
            const uint32_t low_bits = ((bits & 0x0fu) * 0x00204081u) & 0x01010101u;
            // Each selected byte starts at one, so scaling never carries
            // into another lane. The mask supplies the sign-extension bits.
            return low_bits * mask;
        };
        using Traits = llaminar2::cuda_native_quantised::CodebookTraits<CodebookId>;
        static_assert(!SharedIQ3 || (Order == NativeMoEDotOrder::VerifierWideGrid && Traits::is_iq3_grid));
        if constexpr (Order == NativeMoEDotOrder::VerifierWideGrid && Traits::is_iq1_grid)
        {
            const auto *words = reinterpret_cast<const uint16_t *>(payload);
            const uint32_t low_indices = uint32_t{words[0]} | (uint32_t{words[1]} << 16);
            const uint32_t high_indices = words[2];
            const auto expand = [](uint32_t packed) {
                // Each two-bit symbol selects one signed byte in 0xff000100.
                const uint32_t byte = packed & 0xffu;
                const uint32_t pairs = (byte | (byte << 4)) & 0x0f0fu;
                const uint32_t selectors = (pairs | (pairs << 2)) & 0x3333u;
                return static_cast<int32_t>(__byte_perm(0xff000100u, 0u, selectors));
            };
            // IQ1_M interleaves one delta sign bit with each three-bit index.
            constexpr int kHighIndexStride = Traits::is_iq1_m ? 4 : 3;
#pragma unroll
            for (int group = 0; group < 4; ++group)
            {
                const uint32_t index = ((low_indices >> (8 * group)) & 0xffu) |
                                       (((high_indices >> (kHighIndexStride * group)) & 7u) << 8);
                const uint32_t ternary = __ldg(compact_moe_iq1_grid.words + index);
                groups[group * 2] = expand(ternary);
                groups[group * 2 + 1] = expand(ternary >> 8);
            }
        }
        else if constexpr (Order == NativeMoEDotOrder::VerifierWideGrid && Traits::is_iq2_grid)
        {
            constexpr bool kLowOnly = CodebookId == native_formats::IQ2_XXS.codebook_id;
            constexpr unsigned kHighBits = CodebookId == native_formats::IQ2_S.codebook_id ? 2u : 1u;
            MoEOddGridPacket<1> packet{};
            if constexpr (kLowOnly)
            {
                const uint2 words = *reinterpret_cast<const uint2 *>(payload);
                packet.indices[0] = words.x;
                packet.signs = words.y;
            }
            else
                packet = load_moe_odd_grid_packet<1,
                    SignSchedule == NativeMoEIQ2SignSchedule::ClampedFunnelAndIndependentProducts>(payload);

            // Select +8,+25,+43 or -8,-25,-43 directly into each signed byte.
            constexpr uint32_t kPositive = 0x002b1908u;
            constexpr uint32_t kNegative = 0x00d5e7f8u;
#pragma unroll
            for (int group = 0; group < 4; ++group)
            {
                uint32_t index = (packet.indices[0] >> (8 * group)) & 0xffu;
                if constexpr (!kLowOnly)
                    index |= ((packet.high_indices >> (kHighBits * group)) & ((1u << kHighBits) - 1u)) << 8;

                uint32_t magnitudes = 0;
                if constexpr (CodebookId == native_formats::IQ2_S.codebook_id)
                    magnitudes = __ldg(moe_iq2_s_selectors.words + index);
                else if constexpr (CodebookId == native_formats::IQ2_XS.codebook_id)
                    magnitudes = __ldg(moe_iq2_xs_selectors.words + index);
                else
                    magnitudes = __ldg(moe_iq2_xxs_selectors.words + index);

                const uint32_t signs = (packet.signs >> (8 * group)) & 0xffu;
                const uint32_t sign_selectors = [&] {
                    if constexpr (SignSchedule != NativeMoEIQ2SignSchedule::RegisterShifts)
                        return moe_iq2_independent_sign_selectors(signs);
                    else
                    {
                        const uint32_t halves = (signs | (signs << 12)) & 0x000f000fu;
                        const uint32_t pairs = (halves | (halves << 6)) & 0x03030303u;
                        const uint32_t lanes = (pairs | (pairs << 3)) & 0x11111111u;
                        return lanes << 2;
                    }
                }();
                const uint32_t selectors = magnitudes | sign_selectors;
                groups[2 * group] = static_cast<int32_t>(__byte_perm(kPositive, kNegative, selectors));
                groups[2 * group + 1] = static_cast<int32_t>(__byte_perm(kPositive, kNegative, selectors >> 16));
            }
        }
        else if constexpr (Order == NativeMoEDotOrder::VerifierWideGrid &&
                           CodebookId == native_formats::IQ3_S.codebook_id)
        {
            const auto packet = load_moe_odd_grid_packet<2>(payload);
#pragma unroll
            for (int group = 0; group < 8; ++group)
            {
                const uint32_t low = (packet.indices[group / 4] >> (8 * (group & 3))) & 0xffu;
                const uint32_t high = (packet.high_indices >> group) & 1u;
                const uint32_t index = low | (high << 8);
                const uint32_t grid = SharedIQ3 ? iq3_table[index]
                    : llaminar2::cuda_native_quantised::iq3_grid_lookup<CodebookId>(index);
                groups[group] = static_cast<int32_t>(
                    moe_iq3_apply_signs(
                        grid, (packet.signs >> (4 * group)) & 0xfu));
            }
        }
        else if constexpr (Order == NativeMoEDotOrder::VerifierWideGrid &&
                           CodebookId == native_formats::IQ3_XXS.codebook_id)
        {
            // Twelve-byte packets have three aligned words. The first two
            // contain the indices and the last contains eight sign nibbles.
            const auto *words = reinterpret_cast<const uint32_t *>(payload);
            const uint32_t first_indices = words[0];
            const uint32_t last_indices = words[1];
            const uint32_t signs = words[2];
#pragma unroll
            for (int group = 0; group < 8; ++group)
            {
                const uint32_t indices = group < 4 ? first_indices : last_indices;
                const uint32_t index = (indices >> (8 * (group & 3))) & 0xffu;
                const uint32_t grid = SharedIQ3 ? iq3_table[index]
                    : __ldg(llaminar2::cuda_native_quantised::d_iq3xxs_grid + index);
                groups[group] = static_cast<int32_t>(
                    moe_iq3_apply_signs(
                        grid, (signs >> (4 * group)) & 0x0fu));
            }
        }
        else if constexpr (Order == NativeMoEDotOrder::VerifierK32 &&
                           CodebookId == native_formats::IQ2_XXS.codebook_id)
        {
            // Each half of the aligned packet supplies four independent bytes.
            const uint2 packet = *reinterpret_cast<const uint2 *>(payload);
#pragma unroll
            for (int group = 0; group < 4; ++group)
            {
                const uint32_t index = (packet.x >> (8 * group)) & 0xffu;
                const uint32_t signs = packet.y >> (8 * group);
                const uint64_t grid = __ldg(
                    llaminar2::cuda_native_quantised::d_iq2xxs_grid + index);
                groups[group * 2] = static_cast<int32_t>(
                    llaminar2::cuda_native_quantised::iq_apply_signs_4(
                        static_cast<uint32_t>(grid), signs & 0x0fu));
                groups[group * 2 + 1] = static_cast<int32_t>(
                    llaminar2::cuda_native_quantised::iq_apply_signs_4(
                        static_cast<uint32_t>(grid >> 32), (signs >> 4) & 0x0fu));
            }
        }
        else if constexpr (Order == NativeMoEDotOrder::VerifierK32 &&
                           CodebookId == native_formats::Q2_K.codebook_id)
        {
            const uint2 weights = *reinterpret_cast<const uint2 *>(payload);
#pragma unroll
            for (int group = 0; group < 4; ++group)
            {
                groups[group] = static_cast<int32_t>((weights.x >> (2 * group)) & 0x03030303u);
                groups[group + 4] = static_cast<int32_t>((weights.y >> (2 * group)) & 0x03030303u);
            }
        }
        else if constexpr (Order == NativeMoEDotOrder::VerifierK32 &&
                           CodebookId == native_formats::Q3_K.codebook_id)
        {
            const auto *words = reinterpret_cast<const uint32_t *>(payload);
            // The missing high bit makes a centered Q3 weight negative.
            const uint32_t negative = ~words[2];
#pragma unroll
            for (int half = 0; half < 2; ++half)
            {
                const uint32_t word = words[half];
#pragma unroll
                for (int part = 0; part < 4; ++part)
                {
                    const int group = half * 4 + part;
                    const auto low = (word >> (2 * part)) & 0x03030303u;
                    groups[group] = static_cast<int32_t>(
                        low | negative_bytes(negative >> (4 * group), 0xfcu));
                }
            }
        }
        else if constexpr (Order == NativeMoEDotOrder::VerifierK32 &&
                           CodebookId == native_formats::Q5_0.codebook_id)
        {
            const auto *words = reinterpret_cast<const uint32_t *>(payload);
            const uint32_t negative = ~words[4];
#pragma unroll
            for (int group = 0; group < 4; ++group)
            {
                const uint32_t raw = words[group];
                groups[group] = static_cast<int32_t>((raw & 0x0f0f0f0fu) |
                    negative_bytes(negative >> (4 * group), 0xf0u));
                groups[group + 4] = static_cast<int32_t>(((raw >> 4) & 0x0f0f0f0fu) |
                    negative_bytes(negative >> (4 * (group + 4)), 0xf0u));
            }
        }
        else if constexpr (Order == NativeMoEDotOrder::VerifierK32 &&
                           CodebookId == native_formats::Q6_K.codebook_id)
        {
            const auto *words = reinterpret_cast<const uint2 *>(payload);
            const uint2 first = words[0];
            const uint2 second = words[1];
            const uint2 high = words[2];
            const uint32_t low_words[4] = {first.x, first.y, second.x, second.y};
            const auto expand_high_pairs = [](uint32_t bits) {
                return ((bits & 0x03u) << 4) | ((bits & 0x0cu) << 10) |
                       ((bits & 0x30u) << 16) | ((bits & 0xc0u) << 22);
            };
            const auto center = [](uint32_t packed) {
                constexpr uint32_t kByteSignMask = 0x80808080u;
                constexpr uint32_t kByteMidpoint = 0x20202020u;
                // Add a sign-bit guard in every byte so subtracting 32
                // from each lane cannot borrow from its neighbor.
                return static_cast<int32_t>(
                    ((packed ^ kByteSignMask) - kByteMidpoint) ^ kByteSignMask);
            };
#pragma unroll
            for (int group = 0; group < 4; ++group)
            {
                const auto lo = low_words[group] & 0x0f0f0f0fu;
                const auto hi = (low_words[group] >> 4) & 0x0f0f0f0fu;
                groups[group] = center(lo | expand_high_pairs(high.x >> (8 * group)));
                groups[group + 4] = center(hi | expand_high_pairs(high.y >> (8 * group)));
            }
        }
        else
#endif
        {
            llaminar2::cuda_native_quantised::decode_groups_vec<CodebookId>(payload, groups);
        }
    }

    /** @brief Canonical IQ1_M delta numerator for each half and sign pair. */
    struct alignas(16) MoEIQ1MPreparedCorrection
    {
        float halves[2][4]; ///< The two weight-independent pairs of K8 delta terms.
    };

    /**
     * @brief Compute one 32-wide native-quantised block with the serial GEMV FP32 order.
     *
     * MTP verifier rows are allowed to execute as grouped MoE work, but each row
     * must publish exactly the same FP32 bits as the public one-row decode GEMV.
     * The canonical CUDA NativeQuantised verifier kernels make every multiply/add
     * boundary explicit with round-to-nearest intrinsics so nvcc cannot choose a
     * different contraction inside a larger grouped kernel body.  The MoE
     * grouped prefill kernels need the same per-block contract; otherwise a
     * layer-0 ULP drift is amplified by recurrent GDN/short-conv state several
     * layers later. This adapter computes only exact integer terms and then
     * delegates publication to the shared CUDA/cross-backend contract.
     *
     * @tparam CodebookId Canonical NativeQuantised source-policy identifier.
     * @tparam Order Split-K partial, serial decode, or grouped verifier publication.
     * @param a_vals Eight packed activation groups for one 32-value K block.
     * @param packed_groups Eight decoded signed-weight groups.
     * @param payload Prepared codebook payload for IQ1_M sign extraction.
     * @param scale_base Prepared primary binary16 scale plane.
     * @param min_base Secondary scale or minimum plane when required by the compiled codebook.
     * @param emin_base Q2_K paired-minimum plane when required by the compiled codebook.
     * @param linear Block-major prepared-metadata index.
     * @param scale_a FP32 scale for the quantized activation block.
     * @param physical_codebook Actual operand identity within the IQ4 payload family.
     * @tparam SharedIQ1MCorrection Reuse the original signed Q8 delta terms prepared by the CTA.
     * @tparam SplitIQ1SDot Accumulate the exact IQ1_S dot and activation sums in independent halves.
     * @param iq1m_correction Exact delta numerators for each sign pair in each half.
     * @return Canonically rounded contribution, not yet row-accumulated.
     * @pre The caller has admitted the descriptor and all of its required
     *      coefficient planes before entering the K32 traversal.
     */
    template <uint8_t CodebookId,
              NativeMoEDotOrder Order = NativeMoEDotOrder::PartitionPartial,
              bool SharedIQ1MCorrection = false,
              bool SplitIQ1SDot = false>
    __device__ __forceinline__ float moe_native_quantised_block_contribution_rn(
        const int32_t *__restrict__ a_vals,
        const int32_t *__restrict__ packed_groups,
        const uint8_t *__restrict__ payload,
        const uint16_t *__restrict__ scale_base,
        const uint16_t *__restrict__ min_base,
        const uint32_t *__restrict__ emin_base,
        size_t linear,
        float scale_a,
        uint8_t physical_codebook = CodebookId, const MoEIQ1MPreparedCorrection *iq1m_correction = nullptr)
    {
        using Traits = llaminar2::cuda_native_quantised::CodebookTraits<CodebookId>;

        static_assert(!SharedIQ1MCorrection || (Traits::is_iq1_m && Order == NativeMoEDotOrder::VerifierWideGrid));
        static_assert(!SplitIQ1SDot || (CodebookId == native_formats::IQ1_S.codebook_id &&
                                      Order == NativeMoEDotOrder::VerifierWideGrid));
        const auto sum_quantised_word = [](int32_t packed) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860
            if constexpr ((Order == NativeMoEDotOrder::VerifierWideGrid && Traits::is_iq1_grid) ||
                          (Order == NativeMoEDotOrder::VerifierK32 &&
                           (CodebookId == native_formats::Q4_1.codebook_id ||
                            CodebookId == native_formats::Q2_K.codebook_id)))
            {
                // The signed dot against four ones is the exact byte sum.
                return __dp4a(packed, 0x01010101, 0);
            }
            else
#endif
                return llaminar2::cuda_native_quantised::sum_packed_i8(packed);
        };

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860
        if constexpr (SplitIQ1SDot)
        {
            constexpr int32_t ones = 0x01010101;
            int dot_lo = 0;
            int dot_hi = 0;
            int sum_lo = 0;
            int sum_hi = 0;
#pragma unroll
            for (int group = 0; group < 4; ++group)
            {
                const int32_t lo = a_vals[group];
                const int32_t hi = a_vals[group + 4];
                dot_lo = __dp4a(lo, packed_groups[group], dot_lo);
                dot_hi = __dp4a(hi, packed_groups[group + 4], dot_hi);
                sum_lo = __dp4a(lo, ones, sum_lo);
                sum_hi = __dp4a(hi, ones, sum_hi);
            }
            // Both integer halves fit in INT32. The ordered FP32 scale and
            // minimum correction still receive the original combined terms.
            return llaminar2::cuda_native_quantised::
                native_quantised_block_contribution_from_reduced_terms_rn<CodebookId>(
                    dot_lo + dot_hi, 0, scale_a, scale_base[linear], min_base[linear],
                    0u, sum_lo + sum_hi, sum_lo, sum_hi, 0u, 0, 0, 0, 0);
        }
        else if constexpr (Order != NativeMoEDotOrder::PartitionPartial &&
                      CodebookId == native_formats::Q5_1.codebook_id)
        {
            constexpr int32_t kSignedByteOnes = 0x01010101;
            int dot_lo = 0;
            int dot_hi = 0;
            int sum_lo = 0;
            int sum_hi = 0;
#pragma unroll
            for (int group = 0; group < 4; ++group)
            {
                const int32_t lo = a_vals[group];
                const int32_t hi = a_vals[group + 4];
                dot_lo = __dp4a(lo, packed_groups[group], dot_lo);
                dot_hi = __dp4a(hi, packed_groups[group + 4], dot_hi);
                sum_lo = __dp4a(lo, kSignedByteOnes, sum_lo);
                sum_hi = __dp4a(hi, kSignedByteOnes, sum_hi);
            }

            // Each half fits in int32. Join the exact integer terms before
            // applying the original asymmetric scale and rounding contract.
            return llaminar2::cuda_native_quantised::
                native_quantised_block_contribution_from_reduced_terms_rn<CodebookId>(
                    dot_lo + dot_hi, 0, scale_a, scale_base[linear],
                    min_base[linear], 0u, sum_lo + sum_hi, 0, 0, 0u, 0, 0, 0, 0);
        }
        else
#endif
        // IQ4 pairs share exact LUT integers, but retain independent factors.
        if constexpr (Traits::is_iq4_lut)
        {
            int dot = 0;
#pragma unroll
            for (int group = 0; group < 8; ++group)
                dot = __dp4a(a_vals[group], packed_groups[group], dot);
            const float scale = physical_codebook == llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook
                ? llaminar2::cuda_native_quantised::native_quantised_single_scale_rn<llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook>(
                      scale_base[linear], min_base[linear])
                : llaminar2::cuda_native_quantised::fp16_bits_to_float(scale_base[linear]);
            return llaminar2::device_native_quantised_contract::singleScaleBlock(dot, scale, scale_a);
        }
        else if constexpr (Traits::is_dual_scale)
        {
            int dot_lo = 0;
            int dot_hi = 0;
            int sum_lo = 0;
            int sum_hi = 0;
#pragma unroll
            for (int group = 0; group < 4; ++group)
            {
                dot_lo = __dp4a(a_vals[group], packed_groups[group], dot_lo);
                dot_hi = __dp4a(a_vals[group + 4], packed_groups[group + 4], dot_hi);
                sum_lo += sum_quantised_word(a_vals[group]);
                sum_hi += sum_quantised_word(a_vals[group + 4]);
            }

            if constexpr (SharedIQ1MCorrection)
            {
                using namespace llaminar2::device_native_quantised_contract;
                const auto scales = llaminar2::cuda_native_quantised::
                    native_quantised_dual_scales_rn<CodebookId>(scale_base[linear], min_base[linear]);
                const uint16_t signs = reinterpret_cast<const uint16_t *>(payload)[2];
                const int low_pair = ((signs >> 3) & 1) | ((signs >> 6) & 2);
                const int high_pair = ((signs >> 11) & 1) | ((signs >> 14) & 2);
                const float low_delta = __fmul_rn(iq1m_correction->halves[0][low_pair], scales.low);
                const float high_delta = __fmul_rn(iq1m_correction->halves[1][high_pair], scales.high);
                const float correction = __fmul_rn(scale_a, __fadd_rn(low_delta, high_delta));
                return accumulate(
                    dualScaleBlock(dot_lo, dot_hi, scales.low, scales.high, scale_a),
                    correction);
            }

            uint16_t iq1m_qh = 0u;
            int subgroup_sum0 = 0;
            int subgroup_sum1 = 0;
            int subgroup_sum2 = 0;
            int subgroup_sum3 = 0;
            if constexpr (Traits::is_iq1_m)
            {
                iq1m_qh = static_cast<uint16_t>(payload[4]) |
                          (static_cast<uint16_t>(payload[5]) << 8u);
                subgroup_sum0 = sum_quantised_word(a_vals[0]) + sum_quantised_word(a_vals[1]);
                subgroup_sum1 = sum_quantised_word(a_vals[2]) + sum_quantised_word(a_vals[3]);
                subgroup_sum2 = sum_quantised_word(a_vals[4]) + sum_quantised_word(a_vals[5]);
                subgroup_sum3 = sum_quantised_word(a_vals[6]) + sum_quantised_word(a_vals[7]);
            }

            return llaminar2::cuda_native_quantised::
                native_quantised_block_contribution_from_reduced_terms_rn<
                    CodebookId>(
                    dot_lo,
                    dot_hi,
                    scale_a,
                    scale_base[linear],
                    min_base[linear],
                    Traits::is_dual_scale_asym ? emin_base[linear] : 0u,
                    sum_lo + sum_hi,
                    sum_lo,
                    sum_hi,
                    iq1m_qh,
                    subgroup_sum0,
                    subgroup_sum1,
                    subgroup_sum2,
                    subgroup_sum3);
        }
        else
        {
            int dot = 0;
            int sum_a = 0;
#pragma unroll
            for (int group = 0; group < 8; ++group)
            {
                dot = __dp4a(a_vals[group], packed_groups[group], dot);
                sum_a += sum_quantised_word(a_vals[group]);
            }

            return llaminar2::cuda_native_quantised::
                native_quantised_block_contribution_from_reduced_terms_rn<
                    CodebookId>(
                    dot,
                    0,
                    scale_a,
                    scale_base[linear],
                    Traits::requires_secondary_metadata ? min_base[linear] : 0u,
                    0u,
                    sum_a,
                    0,
                    0,
                    0u,
                    0,
                    0,
                    0,
                    0);
        }
    }

    template <uint8_t CodebookId, int TileM>
    __device__ __forceinline__ void accumulate_prefill_dot_block(
        const int32_t (&packed_groups)[8],
        const uint16_t *__restrict__ scale_base,
        const uint16_t *__restrict__ min_base,
        const uint32_t *__restrict__ emin_base,
        const uint8_t *__restrict__ payload,
        size_t linear,
        const int32_t *__restrict__ a4_base,
        int a4_stride_i32,
        const float *__restrict__ scale_a_base,
        int scale_a_stride,
        int tokens_in_group,
        float (&acc)[TileM],
        uint8_t physical_codebook = CodebookId)
    {
#pragma unroll
        for (int m = 0; m < TileM; ++m)
        {
            if (m >= tokens_in_group)
                break;

            // Per-token activation row: a4_base/scale_a_base point at a staged shared-memory
            // tile (contiguous, stride = 8 int32 / 1 scale). The 8 int32 are read as 2x128-bit
            // vector loads to cut the LDS instruction count 4x (relieves the MIO pipe).
            // Requires a4_base + m*stride to be 16-byte aligned (s_a is __align__(16), stride=8).
            const int4 *a4v = reinterpret_cast<const int4 *>(a4_base + static_cast<size_t>(m) * a4_stride_i32);
            const int4 av0 = a4v[0];
            const int4 av1 = a4v[1];
            const int32_t a4[8] = {av0.x, av0.y, av0.z, av0.w, av1.x, av1.y, av1.z, av1.w};
            const float scale_a = scale_a_base[static_cast<size_t>(m) * scale_a_stride];

            const float contribution =
                moe_native_quantised_block_contribution_rn<CodebookId>(
                    a4,
                    packed_groups,
                    payload,
                    scale_base,
                    min_base,
                    emin_base,
                    linear,
                    scale_a, physical_codebook);
            acc[m] = __fadd_rn(acc[m], contribution);
        }
    }

    template <uint8_t CodebookId, int kTileM, int kTileN>
    __global__ void grouped_native_quantised_gate_up_prefill_kernel(
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        llaminar2::MoEGroupedSourceRows source_rows,
        const DeviceNativeQuantisedMatrixDesc *__restrict__ gate_descs,
        const DeviceNativeQuantisedMatrixDesc *__restrict__ up_descs,
        const int *__restrict__ expert_counts,
        const int *__restrict__ expert_offsets,
        const int *__restrict__ active_expert_ids,
        int active_expert_slots,
        float *__restrict__ gate_output,
        float *__restrict__ up_output,
        int N,
        int K)
    {
        // kTileN columns per block (one per thread); kTileM tokens processed per block.
        // Larger kTileM amortizes the expensive IQ-codebook weight decode (done once per
        // (block_idx, n)) over more tokens and reduces redundant cross-group re-decode.
        const int n = blockIdx.x * kTileN + threadIdx.x;
        const int token_group = blockIdx.y;
        const int expert_id = (active_expert_slots > 0)
                                  ? active_expert_ids[blockIdx.z]
                                  : static_cast<int>(blockIdx.z);
        if (expert_id < 0)
            return;

        const int count = expert_counts[expert_id];
        const int first_token = token_group * kTileM;
        if (first_token >= count)
            return; // Whole-block uniform exit (token_group/expert from blockIdx) — barrier-safe.

        const int tokens_in_group = min(kTileM, count - first_token);
        const int first_slot = expert_offsets[expert_id] + first_token;
        const int blocks_per_row = K / 32;

        // Per-thread output column may be out of range when N is not a multiple of kTileN.
        // We must NOT early-return such threads: they still participate in the cooperative
        // shared-memory staging + __syncthreads below. Guard only the weight work / writes.
        const bool active = (n < N);

        const DeviceNativeQuantisedMatrixDesc gate_desc = gate_descs[expert_id];
        const DeviceNativeQuantisedMatrixDesc up_desc = up_descs[expert_id];
        if (!native_quantised_pair_shape_ok<CodebookId>(gate_desc, up_desc, N, K))
            return;

        const uint8_t *gate_payload_base = gate_desc.payload;
        const uint16_t *gate_scale_base = static_cast<const uint16_t *>(gate_desc.scales);
        const uint16_t *gate_min_base = static_cast<const uint16_t *>(gate_desc.mins);
        const uint32_t *gate_emin_base = static_cast<const uint32_t *>(gate_desc.emins);
        const uint8_t *up_payload_base = up_desc.payload;
        const uint16_t *up_scale_base = static_cast<const uint16_t *>(up_desc.scales);
        const uint16_t *up_min_base = static_cast<const uint16_t *>(up_desc.mins);
        const uint32_t *up_emin_base = static_cast<const uint32_t *>(up_desc.emins);

        // Staged activation tile for the current K-block: kTileM tokens × 32 int8 (= 8 int32)
        // plus one blockwise scale per token. Staging once per block (cooperatively) removes
        // the ~kTileN-way redundant L1 loads that all N-threads previously issued.
        // 16-byte aligned so the per-token 8×int32 row can be read as 2× int4 (128-bit) loads.
        __shared__ __align__(16) int32_t s_a[kTileM * 8];
        __shared__ float s_scale[kTileM];

        // Resolve each lane's source once, not once per K block. The activation
        // and scale staging roles differ, but both borrow the same token row.
        static_assert(kTileM * 8 <= kTileN);
        const int a_source = threadIdx.x < tokens_in_group * 8
            ? source_rows.sourceRow(first_slot + (threadIdx.x >> 3)) : 0;
        const int scale_source = threadIdx.x < tokens_in_group
            ? source_rows.sourceRow(first_slot + threadIdx.x) : 0;

        float gate_acc[kTileM] = {};
        float up_acc[kTileM] = {};

        // Weight decode is hoisted above the activation-staging barrier: for IQ3 codebooks
        // the decode is a data-dependent payload-byte load followed by a grid-table lookup
        // (a long-scoreboard latency chain). Issuing it before the cooperative smem staging
        // + __syncthreads lets the load latency overlap the staging loads and barrier wait,
        // with no extra register double-buffer (which would cost occupancy).
        constexpr int kPayloadBytes =
            llaminar2::cuda_native_quantised::CodebookTraits<CodebookId>::payload_bytes;

#pragma unroll 1
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx)
        {
            const size_t linear = static_cast<size_t>(block_idx) * N + static_cast<size_t>(n);

            // Decode this block's weights first so the global-load + grid-lookup latency
            // overlaps the activation staging + barrier below.
            int32_t gate_groups[8];
            int32_t up_groups[8];
            const uint8_t *gate_payload = gate_payload_base + linear * kPayloadBytes;
            const uint8_t *up_payload = up_payload_base + linear * kPayloadBytes;
            if (active)
            {
                llaminar2::cuda_native_quantised::decode_groups_vec<CodebookId>(gate_payload, gate_groups);
                llaminar2::cuda_native_quantised::decode_groups_vec<CodebookId>(up_payload, up_groups);
            }

            // Cooperatively load the activation tile for this K-block into shared memory.
            const int a_count = tokens_in_group * 8;
            for (int idx = threadIdx.x; idx < a_count; idx += kTileN)
            {
                const int g = idx & 7;
                s_a[idx] = reinterpret_cast<const int32_t *>(
                    A_int8 + static_cast<size_t>(a_source) * K + block_idx * 32)[g];
            }
            for (int m = threadIdx.x; m < tokens_in_group; m += kTileN)
            {
                s_scale[m] = scales_A_blockwise[static_cast<size_t>(scale_source) * blocks_per_row + block_idx];
            }
            __syncthreads();

            if (active)
            {
                accumulate_prefill_dot_block<CodebookId, kTileM>(
                    gate_groups, gate_scale_base, gate_min_base, gate_emin_base, gate_payload,
                    linear, s_a, /*a4_stride_i32=*/8, s_scale, /*scale_a_stride=*/1,
                    tokens_in_group, gate_acc, gate_desc.codebook_id);
                accumulate_prefill_dot_block<CodebookId, kTileM>(
                    up_groups, up_scale_base, up_min_base, up_emin_base, up_payload,
                    linear, s_a, /*a4_stride_i32=*/8, s_scale, /*scale_a_stride=*/1,
                    tokens_in_group, up_acc, up_desc.codebook_id);
            }

            // Barrier before the next iteration overwrites the staged tile.
            __syncthreads();
        }

        if (active)
        {
#pragma unroll
            for (int m = 0; m < kTileM; ++m)
            {
                if (m >= tokens_in_group)
                    break;
                const int slot = first_slot + m;
                gate_output[static_cast<size_t>(slot) * N + n] = gate_acc[m];
                up_output[static_cast<size_t>(slot) * N + n] = up_acc[m];
            }
        }
    }


    template <uint8_t CodebookId,
              NativeMoEDotOrder Order = NativeMoEDotOrder::PartitionPartial,
              int FixedBlocks = 0, int Unroll = FixedBlocks, bool SharedIQ3 = false,
              bool SharedIQ1MCorrection = false>
    __device__ __forceinline__ float native_quantised_dot_admitted_range(
        const DeviceNativeQuantisedMatrixDesc &desc,
        int n,
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        int N,
        int b_start,
        int b_end,
        const uint32_t *iq3_table = nullptr,
        const MoEIQ1MPreparedCorrection *iq1m_correction = nullptr)
    {
        constexpr bool OrderedK32 = Order != NativeMoEDotOrder::PartitionPartial;
        const uint8_t *payload_base = desc.payload;
        const uint16_t *scale_base = static_cast<const uint16_t *>(desc.scales);
        const uint16_t *min_base = static_cast<const uint16_t *>(desc.mins);
        const uint32_t *emin_base = static_cast<const uint32_t *>(desc.emins);
        float acc = 0.0f;

#pragma unroll (Unroll > 0 ? Unroll : 1)
        for (int block_idx = FixedBlocks > 0 ? 0 : b_start;
             block_idx < (FixedBlocks > 0 ? FixedBlocks : b_end); ++block_idx)
        {
            const int32_t *a4 = reinterpret_cast<const int32_t *>(A_int8 + block_idx * 32);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860
            // Original Q8 rows and their K32 packets are 16-byte aligned.
            // Materialize both halves in registers for the measured readers.
            int32_t activation_words[8];
            if constexpr (Order == NativeMoEDotOrder::VerifierWideGrid ||
                          (OrderedK32 &&
                           (CodebookId == native_formats::Q5_1.codebook_id ||
                            CodebookId == native_formats::Q6_K.codebook_id ||
                            CodebookId == native_formats::Q3_K.codebook_id ||
                            CodebookId == native_formats::Q2_K.codebook_id ||
                            CodebookId == native_formats::IQ3_S.codebook_id ||
                            CodebookId == native_formats::IQ3_XXS.codebook_id ||
                            CodebookId == native_formats::IQ2_S.codebook_id ||
                            CodebookId == native_formats::IQ2_XS.codebook_id ||
                            CodebookId == native_formats::IQ1_S.codebook_id)))
            {
                const auto *packed = reinterpret_cast<const int4 *>(a4);
                const int4 lo = packed[0];
                const int4 hi = packed[1];
                activation_words[0] = lo.x;
                activation_words[1] = lo.y;
                activation_words[2] = lo.z;
                activation_words[3] = lo.w;
                activation_words[4] = hi.x;
                activation_words[5] = hi.y;
                activation_words[6] = hi.z;
                activation_words[7] = hi.w;
                a4 = activation_words;
            }
#endif
            const size_t linear = static_cast<size_t>(block_idx) * N + static_cast<size_t>(n);
            const uint8_t *payload = payload_base +
                                     linear * llaminar2::cuda_native_quantised::payload_bytes_for_codebook<CodebookId>();

            int32_t packed_groups[8];
            decode_moe_native_quantised_packet<CodebookId, Order, SharedIQ3>(payload, packed_groups, iq3_table);

            const float scale_a = scales_A_blockwise[block_idx];
            const float contribution =
                moe_native_quantised_block_contribution_rn<CodebookId, Order, SharedIQ1MCorrection>(
                    a4,
                    packed_groups,
                    payload,
                    scale_base,
                    min_base,
                    emin_base,
                    linear,
                    scale_a,
                    CodebookId,
                    SharedIQ1MCorrection ? iq1m_correction + block_idx : nullptr);
            if constexpr (OrderedK32)
                acc = moe_accumulate_rn(acc, __fadd_rn(0.0f, contribution));
            else
                acc = __fadd_rn(acc, contribution);
        }

        return acc;
    }

    /**
     * @brief Resolve a descriptor and its range before running the arithmetic.
     * @param desc Original prepared weight descriptor.
     * @param n Output column in a matrix of N columns.
     * @param A_int8 Quantized activation row of K elements.
     * @param scales_A_blockwise Original K32 activation scales.
     * @param N Output column pitch.
     * @param K Logical reduction width.
     * @param b_start First requested K32 packet.
     * @param b_end Exclusive end of the requested packets.
     * @return Ordered dot partial, or zero for an inactive/incompatible input.
     */
    template <uint8_t CodebookId, NativeMoEDotOrder Order = NativeMoEDotOrder::PartitionPartial>
    __device__ __forceinline__ float native_quantised_dot_desc_range(
        const DeviceNativeQuantisedMatrixDesc &desc,
        int n,
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        int N,
        int K,
        int b_start,
        int b_end)
    {
        const int blocks_per_row = K / 32;
        if (!A_int8 || !scales_A_blockwise || n < 0 || n >= N ||
            blocks_per_row <= 0 || !native_quantised_desc_shape_ok<CodebookId>(desc, N, K))
            return 0.0f;

        b_start = max(0, b_start);
        b_end = min(blocks_per_row, b_end);
        if constexpr (CodebookId == llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook)
        {
            // A mixed IQ4 family shares packed bytes but resolves the scale
            // program once. The physical reader does no checks inside K.
            if (desc.codebook_id == 4)
                return native_quantised_dot_admitted_range<4, Order>(
                    desc, n, A_int8, scales_A_blockwise, N, b_start, b_end);
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860
        if constexpr (Order == NativeMoEDotOrder::DecodeK32 &&
                      CodebookId == native_formats::Q5_1.codebook_id)
        {
            // The QFN route's full K32 traversal uses a fixed instruction
            // schedule and keeps the same ordered FP32 publication.
            if (N == 2560 && K == 320 && b_start == 0 && b_end == 10)
                return native_quantised_dot_admitted_range<CodebookId, Order, 10>(
                    desc, n, A_int8, scales_A_blockwise, N, b_start, b_end);
        }
#endif
        return native_quantised_dot_admitted_range<CodebookId, Order>(
            desc, n, A_int8, scales_A_blockwise, N, b_start, b_end);
    }

    /**
     * @brief Evaluate two resolved physical projections in one K-block traversal.
     *
     * Gate and up descriptors have identical geometry and consume the same
     * quantized activation row. The former implementation called
     * native_quantised_dot_desc_range() twice, which repeated activation/scaling
     * loads and all loop/index work. This paired primitive loads each
     * activation block once, decodes the two independent payloads in sequence,
     * and retains one accumulator per projection.
     *
     * Arithmetic is unchanged: each accumulator receives exactly the same
     * block contribution sequence and explicit round-to-nearest FP32 additions
     * as its standalone projection. Interleaving independent gate/up work does
     * not alter either dependency chain, so serial-row byte equivalence is
     * preserved.
     * @tparam GateCodebook Resolved physical gate codebook.
     * @tparam UpCodebook Resolved physical up codebook.
     * @param gate_desc Admitted gate weight planes.
     * @param up_desc Admitted up weight planes.
     * @param n Output column in both matrices.
     * @param A_int8 Original prepared activation codes.
     * @param scales_A_blockwise Original prepared activation scales.
     * @param N Shared output column pitch.
     * @param b_start First admitted K32 packet.
     * @param b_end Exclusive end of the admitted packets.
     * @param gate_acc Ordered gate partial.
     * @param up_acc Ordered up partial.
     */
    template <uint8_t GateCodebook, uint8_t UpCodebook>
    __device__ __forceinline__ void native_quantised_dot_admitted_pair_range(
        const DeviceNativeQuantisedMatrixDesc &gate_desc,
        const DeviceNativeQuantisedMatrixDesc &up_desc,
        int n,
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        int N,
        int b_start,
        int b_end,
        float &gate_acc,
        float &up_acc)
    {
        gate_acc = 0.0f;
        up_acc = 0.0f;

        const auto *gate_scales =
            static_cast<const uint16_t *>(gate_desc.scales);
        const auto *gate_mins =
            static_cast<const uint16_t *>(gate_desc.mins);
        const auto *gate_emins =
            static_cast<const uint32_t *>(gate_desc.emins);
        const auto *up_scales =
            static_cast<const uint16_t *>(up_desc.scales);
        const auto *up_mins =
            static_cast<const uint16_t *>(up_desc.mins);
        const auto *up_emins =
            static_cast<const uint32_t *>(up_desc.emins);

        const size_t gate_bytes =
            llaminar2::cuda_native_quantised::payload_bytes_for_codebook<GateCodebook>();
        const size_t up_bytes =
            llaminar2::cuda_native_quantised::payload_bytes_for_codebook<UpCodebook>();

#pragma unroll 1
        for (int block_idx = b_start; block_idx < b_end; ++block_idx)
        {
            const int32_t *a4 =
                reinterpret_cast<const int32_t *>(A_int8 + block_idx * 32);
            const float scale_a = scales_A_blockwise[block_idx];
            const size_t linear =
                static_cast<size_t>(block_idx) * N + static_cast<size_t>(n);

            int32_t packed_groups[8];
            const uint8_t *gate_payload =
                gate_desc.payload + linear * gate_bytes;
            llaminar2::cuda_native_quantised::decode_groups_vec<GateCodebook>(
                gate_payload, packed_groups);
            gate_acc = __fadd_rn(
                gate_acc,
                moe_native_quantised_block_contribution_rn<GateCodebook>(
                    a4,
                    packed_groups,
                    gate_payload,
                    gate_scales,
                    gate_mins,
                    gate_emins,
                    linear,
                    scale_a));

            const uint8_t *up_payload =
                up_desc.payload + linear * up_bytes;
            llaminar2::cuda_native_quantised::decode_groups_vec<UpCodebook>(
                up_payload, packed_groups);
            up_acc = __fadd_rn(
                up_acc,
                moe_native_quantised_block_contribution_rn<UpCodebook>(
                    a4,
                    packed_groups,
                    up_payload,
                    up_scales,
                    up_mins,
                    up_emins,
                    linear,
                    scale_a));
        }
    }

    /**
     * @brief Admit one gate/up tile and choose both physical readers before K.
     * @param gate_desc Original prepared gate descriptor.
     * @param up_desc Original prepared up descriptor.
     * @param n Output column in both matrices.
     * @param A_int8 Shared original activation codes.
     * @param scales_A_blockwise Shared original activation scales.
     * @param N Shared output column pitch.
     * @param K Logical reduction width.
     * @param b_start First requested K32 packet.
     * @param b_end Exclusive end of the requested packets.
     * @param gate_acc Ordered gate partial, zero when inactive/incompatible.
     * @param up_acc Ordered up partial, zero when inactive/incompatible.
     */
    template <uint8_t CodebookId>
    __device__ __forceinline__ void native_quantised_dot_desc_pair_range(
        const DeviceNativeQuantisedMatrixDesc &gate_desc,
        const DeviceNativeQuantisedMatrixDesc &up_desc,
        int n,
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        int N,
        int K,
        int b_start,
        int b_end,
        float &gate_acc,
        float &up_acc)
    {
        const int blocks_per_row = K / 32;
        if (!A_int8 || !scales_A_blockwise || n < 0 || n >= N ||
            blocks_per_row <= 0 || !native_quantised_pair_shape_ok<CodebookId>(gate_desc, up_desc, N, K))
        {
            gate_acc = up_acc = 0.0f;
            return;
        }

        b_start = max(0, b_start);
        b_end = min(blocks_per_row, b_end);
        const auto evaluate = [&]<uint8_t GateCodebook, uint8_t UpCodebook>()
        {
            native_quantised_dot_admitted_pair_range<GateCodebook, UpCodebook>(
                gate_desc, up_desc, n, A_int8, scales_A_blockwise, N, b_start, b_end, gate_acc, up_acc);
        };
        if constexpr (llaminar2::cuda_native_quantised::CodebookTraits<CodebookId>::is_iq4_lut)
        {
            constexpr auto iq4_source_scale = llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook;
            if (gate_desc.codebook_id == 4)
            {
                if (up_desc.codebook_id == 4)
                    evaluate.template operator()<4, 4>();
                else
                    evaluate.template operator()<4, iq4_source_scale>();
            }
            else if (up_desc.codebook_id == 4)
                evaluate.template operator()<iq4_source_scale, 4>();
            else
                evaluate.template operator()<iq4_source_scale, iq4_source_scale>();
        }
        else
            evaluate.template operator()<CodebookId, CodebookId>();
    }

    /**
     * @brief Full-K dot product wrapper (reduces all K-blocks). Preserves the
     *        original single-shot reduction semantics for non-split-K callers.
     */
    template <uint8_t CodebookId>
    __device__ __forceinline__ float native_quantised_dot_desc(
        const DeviceNativeQuantisedMatrixDesc &desc,
        int n,
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        int N,
        int K)
    {
        return native_quantised_dot_desc_range<CodebookId>(
            desc, n, A_int8, scales_A_blockwise, N, K, 0, K / 32);
    }

    template <NativeMoEDotOrder Order>
    __device__ __forceinline__ float native_quantised_dot_desc_range_dynamic(
        const DeviceNativeQuantisedMatrixDesc &desc,
        int n,
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        int N,
        int K,
        int b_start,
        int b_end)
    {
        switch (desc.codebook_id)
        {
        case 0:
            return native_quantised_dot_desc_range<0, Order>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 4:
            return native_quantised_dot_desc_range<4, Order>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook:
            return native_quantised_dot_desc_range<llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook, Order>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 5:
            return native_quantised_dot_desc_range<5, Order>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 6:
            return native_quantised_dot_desc_range<6, Order>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 7:
            return native_quantised_dot_desc_range<7, Order>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 8:
            return native_quantised_dot_desc_range<8, Order>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 9:
            return native_quantised_dot_desc_range<9, Order>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 10:
            return native_quantised_dot_desc_range<10, Order>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 11:
            return native_quantised_dot_desc_range<11, Order>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 12:
            return native_quantised_dot_desc_range<12, Order>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 13:
            return native_quantised_dot_desc_range<13, Order>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 14:
            return native_quantised_dot_desc_range<14, Order>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 15:
            return native_quantised_dot_desc_range<15, Order>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 16:
            return native_quantised_dot_desc_range<16, Order>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 17:
            return native_quantised_dot_desc_range<17, Order>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case 19:
            return native_quantised_dot_desc_range<19, Order>(desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        case llaminar2::kNativeQuantisedExpandedInt8MinCodebook:
            return native_quantised_dot_desc_range<
                llaminar2::kNativeQuantisedExpandedInt8MinCodebook, Order>(
                desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        default:
            return 0.0f;
        }
    }

    template <uint8_t CodebookId, NativeMoEDotOrder Order = NativeMoEDotOrder::PartitionPartial>
    __device__ __forceinline__ float native_quantised_dot_desc_range_dispatch(
        const DeviceNativeQuantisedMatrixDesc &desc,
        int n,
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        int N,
        int K,
        int b_start,
        int b_end)
    {
        if constexpr (CodebookId == kMixedCodebookSentinel)
        {
            return native_quantised_dot_desc_range_dynamic<Order>(
                desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        }
        else
        {
            return native_quantised_dot_desc_range<CodebookId, Order>(
                desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        }
    }

    template <uint8_t CodebookId>
    __device__ __forceinline__ float native_quantised_dot_desc_dispatch(
        const DeviceNativeQuantisedMatrixDesc &desc,
        int n,
        const int8_t *__restrict__ A_int8,
        const float *__restrict__ scales_A_blockwise,
        int N,
        int K)
    {
        return native_quantised_dot_desc_range_dispatch<CodebookId>(
            desc, n, A_int8, scales_A_blockwise, N, K, 0, K / 32);
    }

    constexpr int kQFNDecodeGateUpN = llaminar2::CUDAMoEBatchInvariantPolicy::qfn_gate_up_n;
    constexpr int kQFNDecodeGateUpK = llaminar2::CUDAMoEBatchInvariantPolicy::qfn_gate_up_k;
    constexpr int kQFNDecodeGateUpPackets = 5;
    constexpr unsigned kQFNVerifierRoutesPerToken =
        llaminar2::CUDAMoEBatchInvariantPolicy::qfn_verifier_routes_per_token;
    static_assert(kQFNDecodeGateUpK / 32 == kQFNDecodeGateUpPackets *
        llaminar2::CUDAMoEBatchInvariantPolicy::gate_up_k_partitions);

    /**
     * @brief Identify a measured SM86 owner from immutable physical admission.
     * @param device Ordinal of the CUDA participant that will capture the node.
     * @return Whether the admitted device has compute capability 8.6.
     */
    bool is_sm86_cuda_device(int device)
    {
        const auto &hardware = llaminar2::cudaHardwareMetadata().at(device);
        return hardware.sm_major == 8 && hardware.sm_minor == 6;
    }

    /** @brief Identify the prepared layouts measured with the fixed QFN schedule. */
    template <uint8_t CodebookId>
    inline constexpr bool kQFNDecodeGateUpCodebook =
        CodebookId == native_formats::Q4_K.codebook_id ||
        CodebookId == native_formats::Q5_K.codebook_id ||
        CodebookId == native_formats::Q8_0.codebook_id;

    /** @brief Prepared layouts with a measured fixed-geometry reader for M=1. */
    template <uint8_t CodebookId>
    inline constexpr bool kQFNDecodeFixedGateUpCodebook =
        kQFNDecodeGateUpCodebook<CodebookId> ||
        llaminar2::CUDAMoEBatchInvariantPolicy::qfnPackedGateUp(CodebookId);

    /**
     * @brief Evaluate one fixed Gate/Up partition with shared activation registers.
     * @tparam CodebookId Physical decoder selected before recording the kernel.
     * @tparam Pitch Physical output-column count of each prepared K32 plane.
     * @tparam Packets Ordered number of K32 packets per partition.
     * @tparam PacketOrder Equivalent packed integer operand schedule, independent
     *         of the fixed FP32 split-K publication order below.
     * @tparam SplitIQ1SDot Use the measured independent integer chains for IQ1_S.
     * @tparam SignSchedule The packed IQ2 sign arithmetic selected with the tile.
     * @param gate Original admitted Gate planes.
     * @param up Original admitted Up planes.
     * @param activations Aligned original Q8 activation row.
     * @param activation_scales Original K32 activation scales.
     * @param column Output column in both projections.
     * @param first_packet First K32 packet of this ordered partition.
     * @param gate_sum Receives the original ordered Gate partial.
     * @param up_sum Receives the original ordered Up partial.
     * @pre Both weight planes are validated at the captured pitch and
     *      each admitted partition contains exactly Packets complete packets.
     */
    template <uint8_t CodebookId, int Pitch, int Packets,
              NativeMoEDotOrder PacketOrder = NativeMoEDotOrder::PartitionPartial,
              bool SplitIQ1SDot = false,
              NativeMoEIQ2SignSchedule SignSchedule = NativeMoEIQ2SignSchedule::RegisterShifts>
    __device__ __forceinline__ void native_quantised_gate_up_fixed_partition(
        const DeviceNativeQuantisedMatrixDesc &gate, const DeviceNativeQuantisedMatrixDesc &up,
        const int8_t *activations, const float *activation_scales,
        int column, int first_packet, float &gate_sum, float &up_sum)
    {
        static_assert((Pitch == 0 || Pitch % 4 == 0) && Packets > 0);
        using Traits = llaminar2::cuda_native_quantised::CodebookTraits<CodebookId>;
        constexpr int payload_bytes = Traits::payload_bytes;
        const int pitch = Pitch > 0 ? Pitch : gate.n;
        // Five independently decoded packets hide the latency of the packed
        // readers on the measured verifier tile. Larger byte streams keep
        // their shorter instruction body.
        constexpr bool wide_packed_unroll = (Pitch == 256 || Pitch == kQFNDecodeGateUpN) && Packets == 5 &&
            (Traits::is_iq_grid || CodebookId == native_formats::Q2_K.codebook_id ||
             CodebookId == native_formats::Q3_K.codebook_id || CodebookId == native_formats::Q5_0.codebook_id ||
             CodebookId == native_formats::Q6_K.codebook_id);
        constexpr int unroll = PacketOrder == NativeMoEDotOrder::PartitionPartial || wide_packed_unroll
            ? Packets : 1;
        const auto *a = reinterpret_cast<const int4 *>(activations + first_packet * 32);
        const int start = first_packet * pitch + column;
        gate_sum = 0.0f;
        up_sum = 0.0f;
#pragma unroll unroll
        for (int packet = 0; packet < Packets; ++packet)
        {
            const int linear = start + packet * pitch;
            const int4 low = a[2 * packet];
            const int4 high = a[2 * packet + 1];
            const int32_t codes[8] = {low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
            const float scale_a = activation_scales[first_packet + packet];
            const auto contribution = [&](const DeviceNativeQuantisedMatrixDesc &desc) {
                const uint8_t *payload = desc.payload + linear * payload_bytes;
                int32_t weights[8];
                decode_moe_native_quantised_packet<CodebookId, PacketOrder, false, SignSchedule>(payload, weights);
                return moe_native_quantised_block_contribution_rn<CodebookId, PacketOrder, false, SplitIQ1SDot>(
                    codes, weights, payload, static_cast<const uint16_t *>(desc.scales),
                    static_cast<const uint16_t *>(desc.mins), static_cast<const uint32_t *>(desc.emins),
                    linear, scale_a);
            };
            gate_sum = __fadd_rn(gate_sum, contribution(gate));
            up_sum = __fadd_rn(up_sum, contribution(up));
        }
    }


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 860
    /** @brief Packed activation bytes in the two contiguous IQ4 nibble groups. */
    struct alignas(16) MoEIQ4InterleavedActivation
    {
        int32_t first[4];
        int32_t last[4];
    };

    /**
     * @brief Arrange one activation packet in the order of the packed weight nibbles.
     * @param codes Aligned K32 packet shared by the two projections.
     * @return Signed DP4A operands containing a byte from each activation half.
     */
    __device__ __forceinline__ MoEIQ4InterleavedActivation
    moe_iq4_interleaved_activation(const int8_t *codes)
    {
        const auto *packets = reinterpret_cast<const uint4 *>(codes);
        const uint4 lo = packets[0];
        const uint4 hi = packets[1];
        const uint32_t left[] = {lo.x, lo.y, lo.z, lo.w};
        const uint32_t right[] = {hi.x, hi.y, hi.z, hi.w};
        MoEIQ4InterleavedActivation result{};
#pragma unroll
        for (int group = 0; group < 4; ++group)
        {
            result.first[group] = static_cast<int32_t>(__byte_perm(left[group], right[group], 0x5140));
            result.last[group] = static_cast<int32_t>(__byte_perm(left[group], right[group], 0x7362));
        }
        return result;
    }

    /**
     * @brief Evaluate one IQ4 packet from the two contiguous nibble groups.
     * @tparam Codebook Resolved physical scale program.
     * @param desc Admitted original weight planes.
     * @param linear Prepared payload and scale packet index.
     * @param codes Activation bytes already in nibble order.
     * @param activation_scale Original K32 activation scale.
     * @return Canonical single-scale contribution with the original FP32 rounding.
     */
    template <uint8_t Codebook>
    __device__ __forceinline__ float moe_iq4_interleaved_contribution(
        const DeviceNativeQuantisedMatrixDesc &desc, int linear,
        const MoEIQ4InterleavedActivation &codes, float activation_scale)
    {
        const uint4 packet = reinterpret_cast<const uint4 *>(desc.payload)[linear];
        const uint32_t payload[4] = {packet.x, packet.y, packet.z, packet.w};
        int first = 0;
        int last = 0;
#pragma unroll
        for (int group = 0; group < 4; ++group)
        {
            uint32_t first_weights;
            uint32_t last_weights;
            llaminar2::cuda_native_quantised::iq4nl_decode_adjacent_nibbles(
                payload[group], first_weights, last_weights);
            // The integer products fit in INT32 so the two independent
            // chains recombine exactly before the canonical scale.
            first = __dp4a(codes.first[group], static_cast<int32_t>(first_weights), first);
            last = __dp4a(codes.last[group], static_cast<int32_t>(last_weights), last);
        }

        const uint16_t primary = static_cast<const uint16_t *>(desc.scales)[linear];
        float scale;
        if constexpr (Codebook == llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook)
        {
            scale = llaminar2::cuda_native_quantised::native_quantised_single_scale_rn<Codebook>(
                primary, static_cast<const uint16_t *>(desc.mins)[linear]);
        }
        else
            scale = llaminar2::cuda_native_quantised::fp16_bits_to_float(primary);
        return llaminar2::device_native_quantised_contract::singleScaleBlock(
            first + last, scale, activation_scale);
    }

    /**
     * @brief Evaluate one fixed Gate/Up interval using a shared activation layout.
     * @tparam Pitch Admitted output-column pitch of each K32 weight plane.
     * @tparam Packets Exact K32 interval in the canonical split-K tree.
     * @param gate Admitted Gate planes.
     * @param up Admitted Up planes.
     * @param codes Aligned original Q8 activation row.
     * @param scales Original K32 activation scales.
     * @param column Output column in both projections.
     * @param first_packet First packet of the ordered partition.
     * @param gate_out Receives the ordered Gate partial.
     * @param up_out Receives the ordered Up partial.
     */
    template <int Pitch, int Packets>
    __device__ __forceinline__ void moe_iq4_interleaved_gate_up(
        const DeviceNativeQuantisedMatrixDesc &gate, const DeviceNativeQuantisedMatrixDesc &up,
        const int8_t *codes, const float *scales, int column, int first_packet,
        float &gate_out, float &up_out)
    {
        constexpr auto nl = native_formats::IQ4_NL.codebook_id;
        constexpr auto xs = llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook;
        const int base = first_packet * Pitch + column;
        const auto evaluate = [&]<uint8_t Gate, uint8_t Up>() {
            float gate_sum = 0.0f;
            float up_sum = 0.0f;
#pragma unroll
            for (int index = 0; index < Packets; ++index)
            {
                const int linear = base + index * Pitch;
                const auto activation = moe_iq4_interleaved_activation(codes + (first_packet + index) * 32);
                const float activation_scale = scales[first_packet + index];
                gate_sum = __fadd_rn(gate_sum, moe_iq4_interleaved_contribution<Gate>(
                    gate, linear, activation, activation_scale));
                up_sum = __fadd_rn(up_sum, moe_iq4_interleaved_contribution<Up>(
                    up, linear, activation, activation_scale));
            }
            gate_out = gate_sum;
            up_out = up_sum;
        };

        // The paired payload family supports either scale contract in either
        // projection. Resolve both once before the first packet.
        if (gate.codebook_id == nl)
        {
            if (up.codebook_id == nl)
                evaluate.template operator()<nl, nl>();
            else
                evaluate.template operator()<nl, xs>();
        }
        else if (up.codebook_id == nl)
            evaluate.template operator()<xs, nl>();
        else
            evaluate.template operator()<xs, xs>();
    }

    /**
     * @brief Publish one fixed-width IQ4 Down column with the original ordered K32 fold.
     * @tparam Codebook Admitted IQ4 physical family.
     * @tparam ReductionWidth Admitted complete K32 reduction width.
     * @param desc Original Down descriptor, validated before evaluation.
     * @param codes Aligned SwiGLU activation row.
     * @param scales Original SwiGLU K32 scales.
     * @param column Output column.
     * @return Byte-equivalent final Down projection value.
     */
    template <uint8_t Codebook, int ReductionWidth>
    __device__ __forceinline__ float moe_iq4_interleaved_down(
        const DeviceNativeQuantisedMatrixDesc &desc, const int8_t *codes,
        const float *scales, int column)
    {
        constexpr auto nl = native_formats::IQ4_NL.codebook_id;
        constexpr auto xs = llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook;
        if (!native_quantised_desc_shape_ok<Codebook>(desc, kWideVerifierDownN, ReductionWidth))
            return 0.0f;
        constexpr int packets = ReductionWidth / 32;
        constexpr bool stage_activation = Codebook == nl && ReductionWidth == kWideVerifierDownK;
        struct alignas(16) PreparedActivation
        {
            MoEIQ4InterleavedActivation codes[packets];
            float scales[packets];
        };
        __shared__ PreparedActivation prepared;
        if constexpr (stage_activation)
        {
            static_assert(kWideVerifierDownN % kVerifierDownTileColumns == 0);
            static_assert(packets <= kVerifierDownTileColumns);
            // This complete CTA owns one route. Transform each K32 activation
            // once before the output lanes consume the shared dot operands.
            if (threadIdx.x < packets)
            {
                const int packet = threadIdx.x;
                prepared.codes[packet] = moe_iq4_interleaved_activation(codes + packet * 32);
                prepared.scales[packet] = scales[packet];
            }
            __syncthreads();
        }
        const auto evaluate = [&]<uint8_t Physical>() {
            float sum = 0.0f;
#pragma unroll
            for (int packet = 0; packet < packets; ++packet)
            {
                const auto activation = stage_activation ? prepared.codes[packet]
                    : moe_iq4_interleaved_activation(codes + packet * 32);
                const int index = packet * kWideVerifierDownN + column;
                const float contribution = moe_iq4_interleaved_contribution<Physical>(
                    desc, index, activation, stage_activation ? prepared.scales[packet] : scales[packet]);
                sum = moe_accumulate_rn(sum, __fadd_rn(0.0f, contribution));
            }
            return sum;
        };
        if constexpr (Codebook == nl)
            return evaluate.template operator()<nl>();
        else
            return desc.codebook_id == xs ? evaluate.template operator()<xs>()
                                         : evaluate.template operator()<nl>();
    }
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860
    /**
     * @brief Read a complete SM86 Down tile with a fixed weight pitch.
     *
     * Each CTA addresses one route. Descriptor checks and shared activation or
     * lookup preparation finish before the ordered K32 dot fold. A K32
     * contribution enters the FP32 sum in the same order as serial decode.
     *
     * @tparam CodebookId Physical reader selected during capture.
     * @tparam ReductionWidth Complete local K admitted by the launcher.
     * @tparam TileColumns Exact output width of this CTA.
     * @param desc Device-published descriptor for the selected expert.
     * @param column Output column owned by the calling lane.
     * @param codes Prepared SwiGLU activation row.
     * @param scales Original FP32 SwiGLU K32 scales.
     * @return The unweighted route contribution for this column.
     * @pre The whole CTA calls with the same route and descriptor.
     */
    template <uint8_t CodebookId, int ReductionWidth, int TileColumns>
    __device__ __forceinline__ float native_quantised_moe_fixed_down(
        const DeviceNativeQuantisedMatrixDesc &desc, int column,
        const int8_t *codes, const float *scales)
    {
        constexpr auto order = NativeMoEDotOrder::VerifierWideGrid;
        static_assert(kWideVerifierDownN % TileColumns == 0);
        float sum = 0.0f;
        using Traits = llaminar2::cuda_native_quantised::CodebookTraits<CodebookId>;
        if constexpr (Traits::is_iq4_lut)
            sum = moe_iq4_interleaved_down<CodebookId, ReductionWidth>(desc, codes, scales, column);
        else if constexpr (Traits::is_iq3_grid)
        {
            if (native_quantised_desc_shape_ok<CodebookId>(desc, kWideVerifierDownN, ReductionWidth))
            {
                constexpr bool source_s = CodebookId == native_formats::IQ3_S.codebook_id;
                constexpr int entries = source_s ? 512 : 256;
                constexpr int packets = ReductionWidth / 32;
                __shared__ __align__(16) uint32_t table[entries];
                const auto *original = source_s
                    ? llaminar2::cuda_native_quantised::d_iq3s_grid
                    : llaminar2::cuda_native_quantised::d_iq3xxs_grid;
                // The fixed-width CTA owns one route and a complete tile.
                // Its lanes load the table together before any K32 math.
#pragma unroll
                for (int i = threadIdx.x; i < entries; i += TileColumns)
                    table[i] = __ldg(original + i);
                __syncthreads();
                sum = native_quantised_dot_admitted_range<CodebookId, order, packets, 1, true>(
                    desc, column, codes, scales, kWideVerifierDownN, 0, packets, table);
            }
        }
        else if constexpr (Traits::is_iq1_m)
        {
            if (native_quantised_desc_shape_ok<CodebookId>(desc, kWideVerifierDownN, ReductionWidth))
            {
                constexpr int packets = ReductionWidth / 32;
                __shared__ MoEIQ1MPreparedCorrection corrections[packets];
                if (threadIdx.x < packets * 2)
                {
                    constexpr int32_t ones = 0x01010101;
                    const auto *words = reinterpret_cast<const int32_t *>(codes);
                    const int first = threadIdx.x * 4;
                    const int32_t sum0 = __dp4a(words[first + 1], ones, __dp4a(words[first], ones, 0));
                    const int32_t sum1 = __dp4a(words[first + 3], ones, __dp4a(words[first + 2], ones, 0));
                    const float group0 = __fmul_rn(0.125f, static_cast<float>(sum0));
                    const float group1 = __fmul_rn(0.125f, static_cast<float>(sum1));
                    auto &pair = corrections[threadIdx.x / 2].halves[threadIdx.x % 2];
                    pair[0] = __fadd_rn(group0, group1);
                    pair[1] = __fadd_rn(-group0, group1);
                    pair[2] = __fadd_rn(group0, -group1);
                    pair[3] = __fadd_rn(-group0, -group1);
                }
                // Stage each signed activation combination once. Explicit
                // rounded adds preserve the canonical signed-zero cases.
                __syncthreads();
                sum = native_quantised_dot_admitted_range<CodebookId, order, packets, 2, false, true>(
                    desc, column, codes, scales, kWideVerifierDownN, 0, packets, nullptr, corrections);
            }
        }
        else if constexpr (CodebookId == native_formats::IQ1_S.codebook_id ||
                           CodebookId == native_formats::IQ2_XXS.codebook_id)
        {
            constexpr int packets = ReductionWidth / 32;
            if (native_quantised_desc_shape_ok<CodebookId>(desc, kWideVerifierDownN, ReductionWidth))
                sum = native_quantised_dot_admitted_range<CodebookId, order, packets, 2>(
                    desc, column, codes, scales, kWideVerifierDownN, 0, packets);
        }
        else
            sum = native_quantised_dot_desc_range_dispatch<CodebookId, order>(
                desc, column, codes, scales, kWideVerifierDownN, ReductionWidth, 0, ReductionWidth / 32);
        return sum;
    }
#endif

} // namespace
