/**
 * @file CanonicalMoEPublicationTestOracle.h
 * @brief Deterministic fixtures for canonical LocalTP MoE publication tests.
 *
 * The canonical GPU publication transaction transports routed-expert rows and
 * shared-expert rank banks in one FP32 collective payload. These helpers build
 * backend-neutral byte witnesses for that layout so CUDA and ROCm certify the
 * same geometry, active-row masking, participant ordering, and serial-row
 * extraction rules. The oracle never evaluates MoE arithmetic: expected
 * outputs must come from repeated production M=1 backend kernels.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace llaminar2::test
{
    /**
     * @brief Complete grouped-verifier row inventory including serial decode.
     */
    inline constexpr std::array<int, 17>
        kCanonicalMoEPublicationRows = {
            1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 31};

    /**
     * @brief Width boundaries that exercise scalar and float4 publication.
     *
     * Qwen's production width is represented by 2048. The adjacent small and
     * tile-boundary widths prove that the interface remains total rather than
     * silently relying on a four-float-aligned model geometry.
     */
    inline constexpr std::array<int, 10>
        kCanonicalMoEPublicationWidths = {
            1, 3, 4, 5, 255, 256, 257, 512, 513, 2048};

    /**
     * @brief Runtime-M probes paired with every non-production test width.
     */
    inline constexpr std::array<int, 4>
        kCanonicalMoEPublicationGeometryRows = {1, 2, 16, 31};

    /**
     * @brief One row extracted from a grouped canonical publication.
     */
    struct CanonicalMoEPublicationRow
    {
        std::vector<float> input;
        std::vector<float> publication;
    };

    /**
     * @brief Complete deterministic input for one canonical publication case.
     *
     * `canonical_reduced` models the payload after the rooted collective: route
     * slots are row-major `[M, top_k, d_model]`, followed by shared banks in
     * participant-major `[participant_count, M, d_model]` order.
     * `publisher_initial` and `publisher_expected` isolate one participant's
     * pre-collective overwrite contract.
     */
    struct CanonicalMoEPublicationFixture
    {
        int seq_len = 0;
        int top_k = 0;
        int d_model = 0;
        int participant_count = 0;
        int participant_index = 0;
        int effective_seq_len = 0;
        std::vector<float> input;
        std::vector<float> gate;
        std::vector<float> canonical_reduced;
        std::vector<float> publisher_shared;
        std::vector<float> publisher_initial;
        std::vector<float> publisher_expected;
    };

    /**
     * @brief Build a nontrivial, byte-stable canonical publication fixture.
     *
     * The generated values are intentionally nonzero and sign-varying so a
     * missing route, rank bank, gate, or combine operation cannot pass by
     * publishing zeros. Only standard scalar host math is used to construct
     * inputs; it does not serve as the arithmetic correctness oracle.
     *
     * @param seq_len Physical graph row capacity.
     * @param top_k Number of canonical routed slots per row.
     * @param d_model Hidden width.
     * @param participant_count Number of shared rank banks.
     * @param participant_index Bank owned by the publisher under test.
     * @param effective_seq_len Device-visible live row prefix.
     * @return Fully populated grouped and publisher fixtures.
     */
    inline CanonicalMoEPublicationFixture
    makeCanonicalMoEPublicationFixture(
        int seq_len,
        int top_k,
        int d_model,
        int participant_count,
        int participant_index,
        int effective_seq_len)
    {
        CanonicalMoEPublicationFixture fixture{
            .seq_len = seq_len,
            .top_k = top_k,
            .d_model = d_model,
            .participant_count = participant_count,
            .participant_index = participant_index,
            .effective_seq_len = effective_seq_len,
        };

        const size_t row_elements =
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
        const size_t route_elements =
            row_elements * static_cast<size_t>(top_k);
        const size_t publication_elements =
            row_elements * static_cast<size_t>(top_k + participant_count);

        fixture.input.resize(row_elements);
        fixture.gate.resize(static_cast<size_t>(d_model));
        fixture.canonical_reduced.resize(publication_elements);
        fixture.publisher_shared.resize(row_elements);
        fixture.publisher_initial.resize(publication_elements);
        fixture.publisher_expected.resize(publication_elements);

        for (int row = 0; row < seq_len; ++row)
        {
            for (int column = 0; column < d_model; ++column)
            {
                const size_t element =
                    static_cast<size_t>(row) * d_model +
                    static_cast<size_t>(column);
                fixture.input[element] =
                    0.071f * std::sin(
                                 0.017f * static_cast<float>(element + 3)) -
                    0.043f * std::cos(
                                 0.011f * static_cast<float>(element + 13)) +
                    0.00037f * static_cast<float>(
                                     static_cast<int>(element % 29) - 14);
            }
        }

        for (int column = 0; column < d_model; ++column)
        {
            fixture.gate[static_cast<size_t>(column)] =
                0.019f * std::sin(
                             0.023f * static_cast<float>(column + 5)) +
                0.013f * std::cos(
                             0.007f * static_cast<float>(column + 19)) -
                0.00021f * static_cast<float>((column % 31) - 15);
        }

        for (int row = 0; row < seq_len; ++row)
        {
            for (int route = 0; route < top_k; ++route)
            {
                for (int column = 0; column < d_model; ++column)
                {
                    const size_t index =
                        (static_cast<size_t>(row) * top_k +
                         static_cast<size_t>(route)) *
                            static_cast<size_t>(d_model) +
                        static_cast<size_t>(column);
                    fixture.canonical_reduced[index] =
                        (0.031f + 0.004f * static_cast<float>(route)) *
                            std::sin(
                                0.009f * static_cast<float>(index + 7)) -
                        (0.017f - 0.001f * static_cast<float>(route)) *
                            std::cos(
                                0.015f * static_cast<float>(index + 23));
                }
            }
        }

        for (int participant = 0;
             participant < participant_count;
             ++participant)
        {
            for (int row = 0; row < seq_len; ++row)
            {
                for (int column = 0; column < d_model; ++column)
                {
                    const size_t row_element =
                        static_cast<size_t>(row) * d_model +
                        static_cast<size_t>(column);
                    const float value =
                        (0.053f + 0.006f * participant) *
                            std::sin(
                                0.013f * static_cast<float>(
                                             row_element + 11 + participant)) +
                        (0.029f - 0.003f * participant) *
                            std::cos(
                                0.005f * static_cast<float>(
                                             row_element + 17));
                    const size_t bank_index =
                        route_elements +
                        static_cast<size_t>(participant) * row_elements +
                        row_element;
                    fixture.canonical_reduced[bank_index] = value;
                    if (participant == participant_index)
                        fixture.publisher_shared[row_element] = value;
                }
            }
        }

        /*
         * Route slots are real routed evidence and must remain untouched by the
         * publisher. Rank banks begin with a conspicuous stale value so the
         * expected image proves that every local, peer, and padded byte was
         * overwritten rather than accidentally inherited.
         */
        std::copy_n(
            fixture.canonical_reduced.begin(),
            route_elements,
            fixture.publisher_initial.begin());
        std::fill(
            fixture.publisher_initial.begin() +
                static_cast<std::ptrdiff_t>(route_elements),
            fixture.publisher_initial.end(),
            -127.25f);
        fixture.publisher_expected = fixture.publisher_initial;

        for (int participant = 0;
             participant < participant_count;
             ++participant)
        {
            for (int row = 0; row < seq_len; ++row)
            {
                for (int column = 0; column < d_model; ++column)
                {
                    const size_t row_element =
                        static_cast<size_t>(row) * d_model +
                        static_cast<size_t>(column);
                    const size_t bank_index =
                        route_elements +
                        static_cast<size_t>(participant) * row_elements +
                        row_element;
                    fixture.publisher_expected[bank_index] =
                        participant == participant_index &&
                                row < effective_seq_len
                            ? fixture.publisher_shared[row_element]
                            : 0.0f;
                }
            }
        }

        return fixture;
    }

    /**
     * @brief Extract one M=1 production-kernel input from grouped storage.
     *
     * @param fixture Grouped canonical publication fixture.
     * @param row Physical row to extract.
     * @return M=1 input and canonical payload with unchanged route/rank order.
     */
    inline CanonicalMoEPublicationRow extractCanonicalMoEPublicationRow(
        const CanonicalMoEPublicationFixture &fixture,
        int row)
    {
        CanonicalMoEPublicationRow extracted;
        extracted.input.resize(static_cast<size_t>(fixture.d_model));
        extracted.publication.resize(
            static_cast<size_t>(fixture.top_k + fixture.participant_count) *
            static_cast<size_t>(fixture.d_model));

        const size_t grouped_row_elements =
            static_cast<size_t>(fixture.seq_len) *
            static_cast<size_t>(fixture.d_model);
        const size_t grouped_route_elements =
            grouped_row_elements * static_cast<size_t>(fixture.top_k);
        const size_t source_row =
            static_cast<size_t>(row) * static_cast<size_t>(fixture.d_model);

        std::copy_n(
            fixture.input.begin() + static_cast<std::ptrdiff_t>(source_row),
            fixture.d_model,
            extracted.input.begin());

        for (int route = 0; route < fixture.top_k; ++route)
        {
            const size_t source =
                (static_cast<size_t>(row) * fixture.top_k +
                 static_cast<size_t>(route)) *
                static_cast<size_t>(fixture.d_model);
            const size_t destination =
                static_cast<size_t>(route) *
                static_cast<size_t>(fixture.d_model);
            std::copy_n(
                fixture.canonical_reduced.begin() +
                    static_cast<std::ptrdiff_t>(source),
                fixture.d_model,
                extracted.publication.begin() +
                    static_cast<std::ptrdiff_t>(destination));
        }

        for (int participant = 0;
             participant < fixture.participant_count;
             ++participant)
        {
            const size_t source =
                grouped_route_elements +
                static_cast<size_t>(participant) * grouped_row_elements +
                source_row;
            const size_t destination =
                static_cast<size_t>(fixture.top_k + participant) *
                static_cast<size_t>(fixture.d_model);
            std::copy_n(
                fixture.canonical_reduced.begin() +
                    static_cast<std::ptrdiff_t>(source),
                fixture.d_model,
                extracted.publication.begin() +
                    static_cast<std::ptrdiff_t>(destination));
        }

        return extracted;
    }
} // namespace llaminar2::test
