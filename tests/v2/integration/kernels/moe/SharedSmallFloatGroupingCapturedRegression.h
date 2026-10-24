/**
 * @file SharedSmallFloatGroupingCapturedRegression.h
 * @brief Backend-shared production bridge, captured topology and publication proof.
 * Existing backend stream/graph owners retain capture transactions. This helper
 * owns fixed test tensors and terminal observations only; it introduces no device
 * execution policy, route-count shadow, allocation ledger or alternate launcher.
 */
#pragma once
#include "SmallFloatGroupingTestOracle.h"
#include "../../../utils/TestTensorFactory.h"
#include "transfer/TransferEngine.h"
#include <array>
#include <cstring>
#include <cstdint>
#include <exception>
#include <memory>
#include <charconv>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <stdexcept>
namespace llaminar2::test
{
    /** @brief Borrowed original 13-argument production grouping coordinates. */
    struct SharedSmallFloatGroupingCaptureInputs
    {
        const float *indices, *weights; ///< Borrowed original route IDs and literal FP32 weights.
        int *counts, *offsets, *tokens, *inverse,
            *experts;           ///< Disjoint integer publication planes.
        float *grouped_weights; ///< Stable expert-major copied weight publication.
        int *active;            ///< Sorted active-expert publication, bounded by max_active.
        int slots, num_experts, top_k,
            max_active; ///< Immutable captured geometry and active-list extent.
    };
    /**
     * @brief Fence outstanding fixture copies/replays before their borrowed owners.
     * @tparam Backend Thin existing runtime/graph adapter, never numerical code.
     */
    template <class Backend> class SharedSmallFloatGroupingRetirement final
    {
      public:
        /** @param stream Exact nonreserved stream whose inputs/outputs remain alive. */
        explicit SharedSmallFloatGroupingRetirement(typename Backend::Stream stream)
            : stream_(stream)
        {
        }
        /** @brief Terminal test-only join; failures terminate instead of freeing live storage. */
        ~SharedSmallFloatGroupingRetirement()
        {
            if (armed_ && !Backend::syncNoThrow(stream_))
                std::terminate();
        }
        /** @brief Disarm only after a successful explicit terminal observation. */
        void dismiss() noexcept
        {
            armed_ = false;
        }

      private:
        typename Backend::Stream stream_;
        bool armed_ = true;
    };
    /** @brief Test observation policy, never a production execution selector. */
    enum class SharedSmallFloatGroupingObservation
    {
        BytesOnly,
        NativeEvents
    };
    /**
     * @brief Parse one disabled-event cell before allocating or capturing anything.
     * @param text Exact four unsigned decimal E,M,K,A coordinates.
     * @return Immutable selected geometry; backend row admission is checked by runner.
     * @throws std::invalid_argument For malformed or missing coordinates.
     */
    inline SmallFloatGroupingGeometry sharedSmallFloatGroupingEventGeometry(const char *text)
    {
        if (!text)
            throw std::invalid_argument(
                "LLAMINAR_TEST_SMALL_FLOAT_GROUP_GEOMETRY requires E,M,K,A");
        int values[4]{};
        const char *cursor = text;
        for (int i = 0; i < 4; ++i)
        {
            const char *end = cursor;
            while (*end && *end != ',')
                ++end;
            const auto parsed = std::from_chars(cursor, end, values[i]);
            if (cursor == end || parsed.ec != std::errc{} || parsed.ptr != end ||
                (i < 3 ? *end != ',' : *end != '\0'))
                throw std::invalid_argument("Grouping event geometry must be exact E,M,K,A");
            cursor = i < 3 ? end + 1 : end;
        }
        return {values[0], values[1], values[2], values[3]};
    }
    /**
     * @brief Prove real production grouping for the backend's complete bounded matrix.
     * @tparam Backend Thin runtime adapter using existing stream/capture owners.
     * @param max_rows Actual backend compact admission (CUDA 4,ROCm 64).
     * @param independent_max Test-owned exact backend policy boundary (CUDA8,ROCm32).
     * @param geometries Exact ordered diagnostic cells, or the canonical matrix.
     * @param observation Whether to perform byte-only proof or isolated native timing.
     * @throws std::runtime_error On a native contract, topology, publication or byte defect.
     * @details One exact stream and fixed TensorFactory storage survive all cells.
     * Capture is fully completed inside Backend::capture before returning. A local
     * retirement guard joins pending replay/copies before graph/host/tensor destruction,
     * including assertion/exception paths. Each external input joins its published
     * event on the exact capture stream before recording begins. All eight dirty
     * patterns replay identical
     * immutable native arguments; no eager production execution or host count is used.
     */
    template <class Backend>
    void runSharedSmallFloatGroupingProductionCells(
        int max_rows,
        int independent_max,
        const std::vector<SmallFloatGroupingGeometry> &geometries,
        SharedSmallFloatGroupingObservation observation)
    {
        const bool events = observation == SharedSmallFloatGroupingObservation::NativeEvents;
        if (events && geometries.size() != 1)
            throw std::invalid_argument("Events require one exact grouping cell");
        for (const auto g : geometries)
            if (g.experts < 1 || g.experts > 512 || g.rows < 1 || g.rows > max_rows ||
                g.top_k < 1 || g.top_k > 16 || g.rows > 64 / g.top_k || g.max_active < 1 ||
                g.max_active > std::min(g.experts, g.rows * g.top_k))
                throw std::invalid_argument("Grouping observation outside backend admission");
        typename Backend::StreamOwner stream_owner(0);
        Backend::check(stream_owner.status());
        const auto stream = stream_owner.get();
        if (!stream || reinterpret_cast<std::uintptr_t>(stream) <= 2)
            throw std::runtime_error("Grouping test requires an exact stream");
        constexpr int Slots = 64, Experts = 512, Guard = 8,
                      Capacity = 2 * Experts + 4 * Slots + 6 * Guard;
        std::vector<float> hi(Slots + Guard), hw(Slots + Guard), read_i(Slots + Guard),
            read_w(Slots + Guard), expected_w(Slots + Guard), actual_w(Slots + Guard);
        std::vector<int> expected(Capacity), actual(Capacity);
        auto ids = TestTensorFactory::createFP32Zeros({Slots + Guard});
        auto weights = TestTensorFactory::createFP32Zeros({Slots + Guard});
        auto output_weights = TestTensorFactory::createFP32Zeros({Slots + Guard});
        auto metadata = TestTensorFactory::createINT32({Capacity});
        SharedSmallFloatGroupingRetirement<Backend> owner_retirement(stream);
        for (auto *tensor : {static_cast<ITensor *>(ids.get()),
                             static_cast<ITensor *>(weights.get()),
                             static_cast<ITensor *>(output_weights.get()),
                             static_cast<ITensor *>(metadata.get())})
            TransferEngine::prepareDeviceInput(tensor, Backend::device(), stream);
        auto *base = static_cast<int *>(metadata->gpu_data_ptr());
        for (const auto g : geometries)
        {
            SCOPED_TRACE(::testing::Message() << "E=" << g.experts << " M=" << g.rows
                                              << " K=" << g.top_k << " A=" << g.max_active);
            const int slots = g.rows * g.top_k;
            int cursor = 0;
            const auto plane = [&](int count)
            {
                int *p = base + cursor;
                cursor += count + Guard;
                return p;
            };
            SharedSmallFloatGroupingCaptureInputs p{
                static_cast<const float *>(ids->gpu_data_ptr()),
                static_cast<const float *>(weights->gpu_data_ptr()),
                plane(g.experts),
                plane(g.experts),
                plane(slots),
                plane(slots),
                plane(slots),
                static_cast<float *>(output_weights->gpu_data_ptr()),
                plane(g.max_active),
                slots,
                g.experts,
                g.top_k,
                g.max_active};
            if (cursor > Capacity)
                throw std::runtime_error("Grouping test publication exceeds fixed storage");
            // The capture helper's local owner closes an aborted recording before
            // this outer retirement can ever synchronize the exact stream.
            // Placement does not itself establish the tensor's recorded external
            // producer-event/consumer-stream join. Import that exact event outside
            // recording; a blocking stream fence is not a coherence substitute.
            for (auto *tensor : {static_cast<ITensor *>(ids.get()),
                                 static_cast<ITensor *>(weights.get()),
                                 static_cast<ITensor *>(output_weights.get()),
                                 static_cast<ITensor *>(metadata.get())})
                TransferEngine::requireDeviceInput(tensor, Backend::device(), stream);
            auto graph = Backend::capture(stream, p, events ? 32 : 1);
            SharedSmallFloatGroupingRetirement<Backend> capture_retirement(stream);
            Backend::inspect(*graph, p, stream, events ? 32 : 1, events);
            for (int pattern = 0; pattern < 8; ++pattern)
            {
                SCOPED_TRACE(::testing::Message() << "dirty replay=" << pattern);
                const auto reference = makeSmallFloatGroupingPublicationCase(g, pattern);
                std::memset(hi.data(), 0xa5, hi.size() * sizeof(float));
                std::memset(hw.data(), 0xa5, hw.size() * sizeof(float));
                std::copy(
                    reference.routing_indices.begin(), reference.routing_indices.end(), hi.begin());
                std::copy(
                    reference.routing_weights.begin(), reference.routing_weights.end(), hw.begin());
                std::memset(expected.data(), 0xa5, expected.size() * sizeof(int));
                std::memset(expected_w.data(), 0xa5, expected_w.size() * sizeof(float));
                const auto put = [&](const auto &values, int *target) {
                    std::memcpy(expected.data() + (target - base),
                                values.data(),
                                values.size() * sizeof(int));
                };
                put(reference.expert_counts, p.counts);
                put(reference.expert_offsets, p.offsets);
                put(reference.grouped_token_indices, p.tokens);
                put(reference.original_to_grouped, p.inverse);
                put(reference.original_expert_ids, p.experts);
                put(reference.active_expert_ids, p.active);
                std::memcpy(
                    expected_w.data(), reference.grouped_weights.data(), slots * sizeof(float));
                Backend::upload(
                    const_cast<float *>(p.indices), hi.data(), hi.size() * sizeof(float), stream);
                Backend::upload(
                    const_cast<float *>(p.weights), hw.data(), hw.size() * sizeof(float), stream);
                Backend::poison(base, Capacity * sizeof(int), stream);
                Backend::poison(p.grouped_weights, (Slots + Guard) * sizeof(float), stream);
                if (!graph->launch())
                    throw std::runtime_error("Production grouping replay failed");
                Backend::download(actual.data(), base, actual.size() * sizeof(int), stream);
                Backend::download(
                    actual_w.data(), p.grouped_weights, actual_w.size() * sizeof(float), stream);
                Backend::download(read_i.data(), p.indices, read_i.size() * sizeof(float), stream);
                Backend::download(read_w.data(), p.weights, read_w.size() * sizeof(float), stream);
                Backend::check(Backend::sync(stream));
                if (std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(int)) ||
                    std::memcmp(
                        expected_w.data(), actual_w.data(), expected_w.size() * sizeof(float)) ||
                    std::memcmp(hi.data(), read_i.data(), hi.size() * sizeof(float)) ||
                    std::memcmp(hw.data(), read_w.data(), hw.size() * sizeof(float)))
                    throw std::runtime_error(
                        "Production grouping metadata/weight/input/poison bytes differ");
            }
            if (events)
            {
                const auto marker = [&](const char *phase)
                {
                    std::cout << "SMALL_FLOAT_GROUP_PRODUCTION_BYTES E=" << g.experts
                              << " M=" << g.rows << " K=" << g.top_k << " A=" << g.max_active
                              << " phase=" << phase << " patterns=8 guards=8 pass=1\n";
                };
                marker("before");
                for (int i = 0; i < 5; ++i)
                    if (!graph->launch())
                        throw std::runtime_error("Grouping event warmup failed");
                Backend::check(Backend::sync(stream));
                for (int sample = 0; sample < 12; ++sample)
                {
                    const double us = Backend::sample(*graph, stream) / 32;
                    if (!std::isfinite(us) || us <= 0)
                        throw std::runtime_error("Invalid native grouping event duration");
                    std::cout << "SMALL_FLOAT_GROUP_PRODUCTION_EVENT E=" << g.experts
                              << " M=" << g.rows << " K=" << g.top_k << " A=" << g.max_active
                              << " sample=" << sample << " repeats=32 warmups=5 value_us=" << us
                              << " unit=us_per_complete_grouping\n";
                }
                if (!graph->launch())
                    throw std::runtime_error("Grouping post-event replay failed");
                Backend::download(actual.data(), base, actual.size() * sizeof(int), stream);
                Backend::download(
                    actual_w.data(), p.grouped_weights, actual_w.size() * sizeof(float), stream);
                Backend::download(read_i.data(), p.indices, read_i.size() * sizeof(float), stream);
                Backend::download(read_w.data(), p.weights, read_w.size() * sizeof(float), stream);
                Backend::check(Backend::sync(stream));
                if (std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(int)) ||
                    std::memcmp(
                        expected_w.data(), actual_w.data(), expected_w.size() * sizeof(float)) ||
                    std::memcmp(hi.data(), read_i.data(), hi.size() * sizeof(float)) ||
                    std::memcmp(hw.data(), read_w.data(), hw.size() * sizeof(float)))
                    throw std::runtime_error("Production grouping post-event bytes differ");
                marker("after");
            }
            capture_retirement.dismiss();
        }
        Backend::check(Backend::sync(stream));
        owner_retirement.dismiss();
    }
    /**
     * @brief Run the canonical backend matrix without timing or environment policy.
     * @tparam Backend Existing native owner adapter.
     * @param max_rows Exact backend compact row admission.
     * @param independent_max Independently expected measured native prefix boundary.
     */
    template <class Backend>
    void runSharedSmallFloatGroupingProductionMatrix(int max_rows, int independent_max)
    {
        auto cells = smallFloatGroupingPublicationGeometries(max_rows, independent_max);
        if (cells.size() != std::size_t(max_rows == 4 ? 2354 : 5354))
            throw std::runtime_error("Grouping matrix extent drift");
        runSharedSmallFloatGroupingProductionCells<Backend>(
            max_rows, independent_max, cells, SharedSmallFloatGroupingObservation::BytesOnly);
    }
    /**
     * @brief Observe exactly one production cell with native events and byte gates.
     * @tparam Backend Existing native owner adapter.
     * @param max_rows Exact backend compact row admission.
     * @param independent_max Independently expected measured native prefix boundary.
     * @param geometry Original immutable E,M,K,A, not a live route-count shadow.
     */
    template <class Backend>
    void runSharedSmallFloatGroupingProductionEvents(int max_rows,
                                                     int independent_max,
                                                     SmallFloatGroupingGeometry geometry)
    {
        runSharedSmallFloatGroupingProductionCells<Backend>(
            max_rows,
            independent_max,
            {geometry},
            SharedSmallFloatGroupingObservation::NativeEvents);
    }
} // namespace llaminar2::test
