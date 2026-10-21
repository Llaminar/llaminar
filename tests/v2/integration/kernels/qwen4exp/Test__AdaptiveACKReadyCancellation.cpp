/**
 * @file Test__AdaptiveACKReadyCancellation.cpp
 * @brief Deterministic original-source cancellation after ACK/fill and before first Ready.
 *
 * One retained native graph serves warm, empty, cancelled and Preserve requests.
 * Setup authenticates actual production function identities/arguments/edges, then
 * adds one diagnostic controller node. Every native source, grant, ACK, slot and
 * reporting operation remains production code. This is a component interleaving
 * proof; it does not advertise a new serving policy or mid-Ready-loop injection.
 */
#include "AdaptiveACKReadyCancellation.h"
#include "fixtures/qwen4exp/AdaptiveLookupNativeComponent.h"
#include "kernels/common/SamplingMath.h"
#include "kernels/qwen4exp/QwenAdaptiveTerminalCounters.h"
#include <algorithm>
#include <array>
#include <gtest/gtest.h>

namespace llaminar2::test::adaptive_native
{
namespace
{
/** @return Original complete native operands, authenticated against all four captured nodes. */
qwen4exp::AdaptiveLookupDeviceArgs arguments(Component &c)
{
    return {c.geometry->nativePlan().parameters(),
            {c.cache ? static_cast<std::byte *>(c.cache->gpu_data_ptr()) : nullptr,
             static_cast<std::byte *>(c.frame->gpu_data_ptr()),
             static_cast<qwen4exp::ImmutableLookupDeviceState *>(c.protocol->gpu_data_ptr())},
            *c.query, c.sourceBinding->deviceView(), c.control->healthy(), c.control->error(), c.control->row()};
}
/**
 * @brief Capture all original production operations and splice before the first Ready once.
 * @param c Actual source/service/cache/controller owner, already initialized and admitted.
 * @note The Component's explicit pair family retains the sole executable claim.
 * An assertion failure unwinds through Component's exact-stream double-fence retirement.
 */
void capture(Component &c)
{
    c.graph = c.worker.createGraphCapture(c.stream);
    c.terminal = c.worker.createGraphCapture(c.stream);
    require(c.graph && c.terminal, "ACK cancellation capture owners missing");
    {
        ScopedBackendGraphCapture scope(c.worker, *c.graph, "adaptive actual ACK cancellation");
        require(scope.begin(), "ACK cancellation capture begin failed");
        c.kernels->gather(*c.storage, *c.query, *c.control, c.stream);
        c.kernels->closeWhenTerminal(*c.storage, *c.control, c.stream);
        scope.finish();
    }
    const auto original = arguments(c);
#ifdef HAVE_CUDA
    if (c.device.is_cuda())
        spliceCUDAAdaptiveACKReadyCancellation(*c.graph, original, c.stream);
#endif
#ifdef HAVE_ROCM
    if (c.device.is_rocm())
        spliceROCmAdaptiveACKReadyCancellation(*c.graph, original, c.stream);
#endif
    {
        ScopedBackendGraphCapture scope(c.worker, *c.terminal, "adaptive ACK cancellation terminal");
        require(scope.begin(), "ACK cancellation terminal capture begin failed");
        c.kernels->closeWhenTerminal(*c.storage, *c.control, c.stream);
        scope.finish();
    }
    require(c.graph->instantiate() && c.terminal->instantiate(), "ACK cancellation instantiate failed");
}
/** @return Original repeated row7, with target9 appended only in the cancellation frame. */
std::vector<std::int64_t> idsFor(int live, bool cancelled)
{
    std::vector<std::int64_t> ids(static_cast<std::size_t>(live) * 16, 7);
    if (cancelled)
        ids.back() = 9;
    return ids;
}
/** @return Actual direct owner reporting after complete native frame retirement. */
qwen4exp::QwenAdaptiveCounterTotals counters(Component &c)
{
    const auto row = c.read(c.control->row(), sampling_math::kDeviceGenerationControlCount);
    qwen4exp::QwenAdaptiveCounterTotals result;
    require(qwen4exp::readQwenAdaptiveCounters(row.data(), row.size(), result),
            "ACK cancellation complete controller extent changed");
    return result;
}
/** @brief Assert original native response bytes, unique ACK grants and inactive response poison. */
void response(Component &c, std::uint64_t rows)
{
    const auto frame = c.frameState();
    ASSERT_LE(frame.unique_rows, c.maximumIds);
    ASSERT_LE(frame.source_rows, frame.unique_rows);
    ASSERT_EQ(frame.source_rows, rows);
    const auto *base = static_cast<const std::byte *>(c.frame->gpu_data_ptr());
    const auto unique = c.read(reinterpret_cast<const policy::UniqueRecord *>(base + c.geometry->frame().unique),
                               frame.unique_rows);
    const auto acks = c.read(reinterpret_cast<const ::llaminar2::ImmutableRowInvalidationAcknowledgement *>(
                                 base + c.geometry->frame().acknowledgements), rows);
    const auto native = c.read(base + c.geometry->frame().native_response,
                               c.maximumIds * c.geometry->rowBytes());
    std::uint64_t actualRows = 0;
    for (const auto &row : unique)
    {
        EXPECT_EQ(row.flags & 4U, 0u) << "Every original reader/reservation pin must retire";
        if (row.flags & policy::InitialHit)
            continue;
        ASSERT_LT(row.response_coordinate, acks.size());
        ASSERT_LT(row.row, c.source.rows);
        EXPECT_EQ(acks[row.response_coordinate].destination_grant,
                  row.flags & policy::ReservedFill ? row.expected_slot_generation : 0);
        EXPECT_EQ(std::memcmp(native.data() + row.response_coordinate * c.geometry->rowBytes(),
                              c.source.native.data() + row.row * c.geometry->rowBytes(),
                              c.geometry->rowBytes()), 0);
        ++actualRows;
    }
    EXPECT_EQ(actualRows, rows);
    EXPECT_TRUE(std::all_of(native.begin() + rows * c.geometry->rowBytes(), native.end(),
                            [](std::byte b) { return b == std::byte{0xa5}; }));
}
/** @brief Snapshot only already-retired actual cache metadata; no serving host directory is created. */
struct CacheObservation
{
    policy::OwnerControl owner{};
    std::vector<policy::Slot32> slots;
    std::vector<policy::RadixNode> nodes;
};
/** @return Exact terminal owner/slots/nodes for the C2 adversarial oracle. */
CacheObservation observeCache(Component &c)
{
    const auto *base = static_cast<const std::byte *>(c.cache->gpu_data_ptr());
    return {c.read(reinterpret_cast<const policy::OwnerControl *>(base + c.geometry->cache().owner), 1).front(),
            c.read(reinterpret_cast<const policy::Slot32 *>(base + c.geometry->cache().slots), 2),
            c.read(reinterpret_cast<const policy::RadixNode *>(base + c.geometry->cache().nodes), 1)};
}
/** @return Actual common directory lookup over an immutable terminal copy, without duplicating its algorithm. */
std::uint32_t find(CacheObservation &cache, policy::FrameControl &frame, Component &c, std::uint64_t row)
{
    const auto &p = c.geometry->nativePlan().parameters();
    policy::Operations<policy::Slot32> op({cache.slots.data(), cache.nodes.data(), &cache.owner, &frame,
                                         nullptr, nullptr, p.slots, p.protected_limit, p.lane_width,
                                         p.source_rows, p.policy.aging_accesses});
    return op.find(row);
}
/** @brief Assert full live output and inactive poison against direct original native decoding. */
void successfulOutput(Component &c, std::span<const std::int64_t> ids)
{
    const auto output = c.read(c.output(), c.maximumIds * c.source.columns);
    auto expected = std::vector<float>(output.size(), -12345.f);
    std::vector<std::byte> native(ids.size() * c.geometry->rowBytes());
    for (std::size_t i = 0; i < ids.size(); ++i)
        std::memcpy(native.data() + i * c.geometry->rowBytes(),
                    c.source.native.data() + ids[i] * c.geometry->rowBytes(), c.geometry->rowBytes());
    if (!ids.empty())
        qwen4exp::decodeCPUImmutableLookupRows(c.geometry->format(), c.source.columns, c.geometry->rowBytes(),
                                              native.data(), ids.size(), expected.data());
    EXPECT_EQ(std::memcmp(output.data(), expected.data(), output.size() * sizeof(float)), 0);
    EXPECT_EQ(c.read(c.control->error(), 1).front(), 0);
}
/**
 * @brief Execute the real source/ACK/fill/cancellation/reset interleaving twenty times.
 * @param device Exact explicit backend endpoint, with canonical CPU service startup.
 * @note C0 proves scratch-only progress; C2 proves a pinned initial hit survives
 * while its different acknowledged reservation never becomes Ready. All observed
 * state is copied only after complete native replay, not to control that replay.
 */
void cancellation(DeviceId device)
{
    auto *backend = getBackendFor(device);
    if (!backend || backend->deviceCount() <= device.ordinal)
    {
        GTEST_SKIP() << "Requested backend unavailable";
    }
    if (!hasCPUBackend())
        initCPUBackend(-1);
    auto &worker = GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&]
    {
        for (const auto slots : {0u, 2u})
            for (const auto capacity : {1, 17})
            {
                SCOPED_TRACE(::testing::Message() << device.toString() << " C=" << slots << " M=" << capacity);
                Source source(iq4Source(97, 160), 97, 160, GGUFTensorType::IQ4_NL);
                Component c(source, device, worker, capacity, slots, AdaptiveFixtureGraphFamily::LookupAndTerminal);
                c.prepareBundle();
                c.initializeMetadata();
                c.admit();
                c.workload(capacity, idsFor(capacity, false));
                capture(c);
                const auto *const graph = c.graph.get();
                const auto signature = c.kernels->captureIdentity(*c.storage);
                for (int repeat = 0; repeat < 20; ++repeat)
                {
                    SCOPED_TRACE(repeat);
                    if (repeat)
                        c.admit();
                    const auto warm = idsFor(capacity, false);
                    c.workload(capacity, warm);
                    c.launch();
                    c.join();
                    successfulOutput(c, warm);
                    ASSERT_TRUE(c.close());
                    auto warmFrame = c.frameState();
                    EXPECT_EQ(warmFrame.phase, policy::Phase::Idle);
                    EXPECT_EQ(warmFrame.status, policy::Status::Success);
                    EXPECT_EQ(warmFrame.held_pins, 0u);
                    CacheObservation before;
                    std::uint32_t warmSlot = policy::None;
                    std::uint64_t warmGeneration = 0;
                    if (slots)
                    {
                        before = observeCache(c);
                        warmSlot = find(before, warmFrame, c, 7);
                        ASSERT_LT(warmSlot, slots);
                        warmGeneration = before.slots[warmSlot].generation;
                        // filled is a high-water slot index, not live occupancy.
                        EXPECT_EQ(before.owner.filled, repeat ? 2u : 1u);
                    }
                    const auto sourceBefore = c.service->stats();
                    const auto sequenceBefore = c.protocolState().issuedSequence;
                    c.admit();
                    const int live = capacity == 1 ? 1 : (repeat % 3 == 0 ? 17 : repeat % 3 == 1 ? 1 : 9);
                    c.workload(live, idsFor(live, true));
                    const int commitBudget = 3;
                    c.upload(c.control->row() + sampling_math::kDeviceGenerationControlTransactionCommitBudget,
                             std::span<const int>(&commitBudget, 1));
                    c.launch();
                    c.join();
                    const auto row = c.read(c.control->row(), sampling_math::kDeviceGenerationControlCount);
                    EXPECT_EQ(row[sampling_math::kDeviceGenerationControlOk], 0);
                    EXPECT_EQ(row[sampling_math::kDeviceGenerationControlRequestComplete], 1);
                    EXPECT_EQ(row[sampling_math::kDeviceGenerationControlTransactionCommitBudget], 0);
                    EXPECT_EQ(row[sampling_math::kDeviceGenerationControlErrorCode],
                              static_cast<int>(sampling_math::DeviceGenerationError::InvalidController));
                    const auto output = c.read(c.output(), c.maximumIds * c.source.columns);
                    EXPECT_TRUE(std::all_of(output.begin(), output.end(), [](float x) { return x == -12345.f; }));
                    auto frame = c.frameState();
                    EXPECT_EQ(frame.phase, policy::Phase::Idle);
                    EXPECT_EQ(frame.status, policy::Status::Cancelled);
                    EXPECT_EQ(frame.held_pins, 0u);
                    const auto requested = slots ? 1u : 2u;
                    response(c, requested);
                    const qwen4exp::QwenAdaptiveCounterTotals expected{1, 0, slots ? 1u : 0u, requested,
                                                                      requested, requested * 90u, 0, 0, 1};
                    EXPECT_EQ(counters(c), expected);
                    const auto state = c.protocolState();
                    EXPECT_EQ(state.issuedSequence, sequenceBefore + 1);
                    EXPECT_EQ(state.issuedSequence, state.consumedSequence);
                    EXPECT_EQ(state.generation, c.generation);
                    EXPECT_EQ(state.admissionGeneration, c.generation);
                    EXPECT_EQ(state.reserved, 0u);
                    ASSERT_TRUE(c.close());
                    const auto sourceAfter = c.service->stats();
                    EXPECT_EQ(sourceAfter.requests, sourceBefore.requests + 1);
                    EXPECT_EQ(sourceAfter.ids, sourceBefore.ids + requested);
                    EXPECT_EQ(sourceAfter.idBytes, sourceBefore.idBytes + requested * 8u);
                    EXPECT_EQ(sourceAfter.nativeBytes, sourceBefore.nativeBytes + requested * 90u);
                    EXPECT_EQ(sourceAfter.failures, sourceBefore.failures);
                    if (slots)
                    {
                        auto after = observeCache(c);
                        EXPECT_EQ(after.owner.flags, policy::OwnerInitialized);
                        EXPECT_EQ(after.owner.filled, 2u);
                        EXPECT_EQ(std::count_if(after.slots.begin(), after.slots.end(),
                                                [](const policy::Slot32 &slot)
                                                { return policy::Operations<policy::Slot32>::state(slot) == policy::SlotState::Ready; }), 1);
                        EXPECT_EQ(after.owner.next_slot_generation, before.owner.next_slot_generation + 1);
                        EXPECT_EQ(find(after, frame, c, 9), policy::None);
                        EXPECT_EQ(find(after, frame, c, 7), warmSlot);
                        EXPECT_EQ(frame.status, policy::Status::Cancelled);
                        EXPECT_EQ(after.slots[warmSlot].generation, warmGeneration);
                        EXPECT_FALSE(policy::Operations<policy::Slot32>::pinned(after.slots[warmSlot]));
                        EXPECT_EQ(policy::Operations<policy::Slot32>::state(after.slots[warmSlot]), policy::SlotState::Ready);
                        const auto *cache = static_cast<const std::byte *>(c.cache->gpu_data_ptr());
                        const auto payload = c.read(cache + c.geometry->cache().payload + warmSlot * c.geometry->slotStride(), 90);
                        EXPECT_EQ(std::memcmp(payload.data(), source.native.data() + 7 * 90, 90), 0);
                    }
                    // Fresh request publication clears only its canonical data.
                    // Same executable + row7 proves warm preservation/U0 at C2;
                    // C0 uses actual empty input to prove handshake-free progress.
                    const auto stableSequence = c.protocolState().issuedSequence;
                    const auto idleBefore = c.service->stats();
                    c.admit();
                    const auto next = slots ? idsFor(1, false) : idsFor(0, false);
                    c.workload(slots ? 1 : 0, next);
                    c.launch();
                    c.join();
                    successfulOutput(c, next);
                    EXPECT_EQ(c.frameState().source_rows, 0u);
                    EXPECT_EQ(c.protocolState().issuedSequence, stableSequence);
                    ASSERT_TRUE(c.close());
                    const auto idleAfter = c.service->stats();
                    EXPECT_EQ(idleAfter.requests, idleBefore.requests);
                    EXPECT_EQ(idleAfter.ids, idleBefore.ids);
                    EXPECT_EQ(idleAfter.nativeBytes, idleBefore.nativeBytes);
                    EXPECT_EQ(c.graph.get(), graph);
                    EXPECT_EQ(c.kernels->captureIdentity(*c.storage), signature);
                    const auto zeroBefore = c.service->stats();
                    c.admit();
                    c.workload(0, idsFor(0, false));
                    c.launch();
                    c.join();
                    successfulOutput(c, {});
                    EXPECT_EQ(counters(c), (qwen4exp::QwenAdaptiveCounterTotals{1, 0, 0, 0, 0, 0, 0, 0, 0}));
                    EXPECT_EQ(c.protocolState().issuedSequence, stableSequence);
                    ASSERT_TRUE(c.close());
                    const auto zeroAfter = c.service->stats();
                    EXPECT_EQ(zeroAfter.requests, zeroBefore.requests);
                    EXPECT_EQ(zeroAfter.ids, zeroBefore.ids);
                    EXPECT_EQ(zeroAfter.nativeBytes, zeroBefore.nativeBytes);
                }
            }
    });
}
} // namespace
#ifdef HAVE_CUDA
TEST(AdaptiveACKReadyCancellation, CUDAAfterACKBeforeFirstReady)
{
    cancellation(DeviceId::cuda(0));
}
#endif
#ifdef HAVE_ROCM
TEST(AdaptiveACKReadyCancellation, ROCmAfterACKBeforeFirstReady)
{
    cancellation(DeviceId::rocm(0));
}
#endif
} // namespace llaminar2::test::adaptive_native
