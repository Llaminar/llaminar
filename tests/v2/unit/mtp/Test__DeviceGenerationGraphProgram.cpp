/**
 * @file Test__DeviceGenerationGraphProgram.cpp
 * @brief Device-free construction and exception-path proofs for generation graphs.
 *
 * Opaque host addresses model graph identity only. These doubles never access
 * a GPU, launch a kernel, or claim mathematical correctness. Integration tests
 * use the same builder with real captured sampling and ticket publications.
 */
#include "execution/mtp/DeviceGenerationGraphProgram.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../mocks/MockBackend.h"
#include "../../mocks/MockWorkerGPUContext.h"
#include <gtest/gtest.h>
#include <array>
#include <stdexcept>
#include <vector>

namespace
{
using namespace llaminar2;
using namespace llaminar2::sampling_math;

/** @brief Exact graph-construction trace with deliberately injectable failures. */
class ProgramGraph final : public IGPUGraphCapture
{
public:
    int stream_token = 0;
    void *stream = &stream_token;
    bool executable = false, native_supported = true, selector_supported = true;
    bool build_ok = true, instantiate_ok = true, begin_ok = true;
    int builds = 0, begins = 0, ends = 0, instantiations = 0;
    size_t recorded_nodes = 1;
    DeviceControlledLoopPredicate predicate{};
    DeviceControlledLoopSelector selector{};
    std::vector<const IGPUGraphCapture *> entry, initialization, iteration;
    /** @brief Record the shared continuation binding without reading its rows. */
    bool buildDeviceControlledWhileLoop(const DeviceControlledLoopProgram &program,
        const DeviceControlledLoopPredicate &value) override
    {
        ++builds;
        predicate = value;
        for (const auto &fragment : program.entry) entry.push_back(fragment.capture);
        for (const auto &fragment : program.initialization) initialization.push_back(fragment.capture);
        for (const auto &fragment : program.iteration) iteration.push_back(fragment.capture);
        return build_ok;
    }
    /** @brief Record the dynamic selector without launching any branch. */
    bool buildDeviceControlledSelectorWhileLoop(const DeviceControlledLoopProgram &program,
        const DeviceControlledLoopPredicate &value, const DeviceControlledLoopSelector &selection) override
    { selector = selection; return buildDeviceControlledWhileLoop(program, value); }
    /** @return Explicit native capability, never an inferred retry choice. */
    bool supportsDeviceControlledWhileLoop() const noexcept override { return native_supported; }
    /** @return Explicit selector capability for the same owner. */
    bool supportsDeviceControlledSelectorWhileLoop() const noexcept override { return selector_supported; }
    /** @brief Record entry to the publisher capture scope. */
    bool beginCapture() override { ++begins; return begin_ok; }
    /** @brief Record that normal or exceptional exit closes the capture once. */
    bool endCapture() override { ++ends; return true; }
    /** @brief Expose success only after the caller actually instantiated. */
    bool instantiate() override { ++instantiations; executable = instantiate_ok; return instantiate_ok; }
    /** @brief Setup must never run the model. */
    bool launch() override { ADD_FAILURE(); return false; }
    /** @return The exact opaque stream, including null for the rejection test. */
    void *executionStream() const noexcept override { return stream; }
    /** @brief This compiler cannot mutate retained executable topology. */
    GraphUpdateResult tryUpdate() override { ADD_FAILURE(); return GraphUpdateResult::Failed; }
    /** @return No executable updates are allowed in this protocol. */
    bool supportsExecutableUpdate() const noexcept override { return false; }
    /** @return Whether instantiation has completed. */
    bool hasExecutable() const override { return executable; }
    /** @return No real memory is owned by the test graph. */
    size_t residentMemoryBytes() const noexcept override { return 0; }
    /** @return A non-empty construction fixture, not a production node count. */
    size_t nodeCount() const override { return recorded_nodes; }
    /** @brief The caller, not the compiler, owns explicit retirement. */
    void reset() override { ADD_FAILURE(); }
    /** @return Diagnostic-only identity. */
    const char *backendName() const override { return "program-unit-double"; }
};

/** @brief Ticket-only backend checks stream and capture scope, then can fail/throw. */
class TicketBackend final : public test::MockBackend
{
public:
    /** @brief Select a GPU-shaped backend without loading a driver. */
    explicit TicketBackend(DeviceType type) : MockBackend(type) {}
    bool enqueue_ok = true, throw_on_enqueue = false;
    int calls = 0;
    void *expected_stream = nullptr;
    llaminar2::testing::MockWorkerGPUContext *context = nullptr;
    /** @brief A publisher observes capture metadata only, never live controller bytes. */
    bool enqueuePublishDeviceGenerationDispatchTickets(void *, int, int,
        const void *, void *, int ordinal, void *stream) override
    {
        ++calls;
        EXPECT_EQ(ordinal, 7);
        EXPECT_EQ(stream, expected_stream);
        EXPECT_TRUE(context->isDeviceGraphCaptureActive());
        EXPECT_TRUE(isGraphCaptureActive());
        if (throw_on_enqueue) throw std::runtime_error("native asynchronous failure");
        return enqueue_ok;
    }
};

TEST(DeviceGenerationGraphProgram, OrdinaryAndFixedSharePredicateDynamicAddsOnlySelector)
{
    std::array<int, kDeviceGenerationControlCount> rows{};
    ProgramGraph source;
    const DeviceControlledLoopFragment fragments[] = {{.name = "complete transaction", .capture = &source}};
    auto dynamic = DeviceGenerationPolicy::fixed(2);
    dynamic.mode = DeviceGenerationPolicyMode::Dynamic;
    dynamic.minimum_depth = 1;
    dynamic.maximum_depth = 15;
    for (auto policy : {DeviceGenerationPolicy::ordinary(), DeviceGenerationPolicy::forwardOnly(),
                        DeviceGenerationPolicy::fixed(1), DeviceGenerationPolicy::fixed(15), dynamic})
    {
        ProgramGraph parent;
        std::string error;
        ASSERT_TRUE(DeviceGenerationGraphProgram::native(parent,
            {policy, rows.data(), kDeviceGenerationControlCount, 1}, {.iteration = fragments}, error)) << error;
        EXPECT_EQ(parent.builds, 1);
        EXPECT_EQ(parent.instantiations, 1);
        EXPECT_EQ(parent.begins, 0);
        EXPECT_EQ(parent.predicate.control_rows_device, rows.data());
        EXPECT_EQ(parent.predicate.healthy_index, kDeviceGenerationControlOk);
        EXPECT_EQ(parent.predicate.complete_index, kDeviceGenerationControlRequestComplete);
        if (policy.mode == DeviceGenerationPolicyMode::Dynamic) {
            EXPECT_EQ(parent.selector.minimum_selector, 1);
            EXPECT_EQ(parent.selector.maximum_selector, 15);
            EXPECT_EQ(parent.selector.selector_index, kDeviceGenerationControlCurrentDraftDepth);
        }
        EXPECT_FALSE(DeviceGenerationGraphProgram::native(parent,
            {policy, rows.data(), kDeviceGenerationControlCount, 1}, {.iteration = fragments}, error));
        EXPECT_EQ(parent.builds, 1) << "Retained executables must never be silently replaced";
    }
}

TEST(DeviceGenerationGraphProgram, InvalidBindingsAndNativeFailuresCannotSelectHostedExecution)
{
    std::array<int, kDeviceGenerationControlCount> rows{};
    for (int failure = 0; failure < 10; ++failure) {
        SCOPED_TRACE(failure);
        ProgramGraph parent, source;
        source.recorded_nodes = failure == 9 ? 0 : 1;
        DeviceGenerationGraphControl control{DeviceGenerationPolicy::ordinary(), rows.data(), kDeviceGenerationControlCount, 1};
        DeviceControlledLoopFragment fragment{.name = "transaction", .capture = &source};
        if (failure == 0) control.rows = nullptr;
        if (failure == 1) control.stride = 1;
        if (failure == 2) control.policy = DeviceGenerationPolicy::fixed(0);
        if (failure == 3) parent.stream = nullptr;
        if (failure == 4) fragment.capture = &parent;
        if (failure == 5) parent.native_supported = false;
        if (failure == 6) { fragment.execution = DeviceControlledLoopFragmentExecution::IfDeviceSelectorAtLeast; fragment.minimum_selector = 1; }
        if (failure == 7) parent.build_ok = false;
        if (failure == 8) parent.instantiate_ok = false;
        std::string error;
        EXPECT_FALSE(DeviceGenerationGraphProgram::native(parent, control, {.iteration = {&fragment, 1}}, error));
        EXPECT_FALSE(error.empty());
        EXPECT_EQ(parent.begins, 0) << "No hosted retry may hide a native failure";
        EXPECT_EQ(parent.instantiations, failure == 8 ? 1 : 0);
    }
}

/** @test Initialization remains separate from repeated work for both native algorithms. */
TEST(DeviceGenerationGraphProgram, InitializationHasOrderedOnceOnlyIdentity)
{
    std::array<int, kDeviceGenerationControlCount> rows{};
    ProgramGraph receive, first, second, body;
    const DeviceControlledLoopEntryFragment entry[] = {
        {.name = "receive current authority command", .capture = &receive}};
    const DeviceControlledLoopFragment initialization[] = {
        {.name = "prefill sample", .capture = &first},
        {.name = "prefill publication", .capture = &second}};
    const DeviceControlledLoopFragment iteration[] = {{.name = "decode transaction", .capture = &body}};
    auto dynamic = DeviceGenerationPolicy::fixed(2);
    dynamic.mode = DeviceGenerationPolicyMode::Dynamic;
    dynamic.minimum_depth = 1;
    dynamic.maximum_depth = 15;
    for (auto policy : {DeviceGenerationPolicy::ordinary(), dynamic})
    {
        ProgramGraph parent;
        std::string error;
        ASSERT_TRUE(DeviceGenerationGraphProgram::native(parent,
            {policy, rows.data(), kDeviceGenerationControlCount, 1},
            {.entry = entry, .initialization = initialization, .iteration = iteration}, error)) << error;
        EXPECT_EQ(parent.entry, (std::vector<const IGPUGraphCapture *>{&receive}));
        EXPECT_EQ(parent.initialization, (std::vector<const IGPUGraphCapture *>{&first, &second}));
        EXPECT_EQ(parent.iteration, (std::vector<const IGPUGraphCapture *>{&body}));
        EXPECT_EQ(parent.builds, 1);
        EXPECT_EQ(parent.instantiations, 1);
    }
}

/** @test Arrival cannot be conditional, incomplete or a self-importing graph. */
TEST(DeviceGenerationGraphProgram, InvalidArrivalCannotMutateDestination)
{
    std::array<int, kDeviceGenerationControlCount> rows{};
    for (int failure = 0; failure < 5; ++failure)
    {
        SCOPED_TRACE(failure);
        ProgramGraph parent, arrival, body;
        DeviceControlledLoopEntryFragment entry{
            .name = "receive authority command", .capture = &arrival};
        const DeviceControlledLoopFragment iteration{.name = "transaction", .capture = &body};
        if (failure == 0) entry.capture = nullptr;
        if (failure == 1) entry.capture = &parent;
        if (failure == 2) arrival.recorded_nodes = 0;
        if (failure == 3) entry.name = "";
        if (failure == 4) entry.name = nullptr;
        std::string error;
        EXPECT_FALSE(DeviceGenerationGraphProgram::native(parent,
            {DeviceGenerationPolicy::ordinary(), rows.data(), kDeviceGenerationControlCount, 1},
            {.entry = {&entry, 1}, .iteration = {&iteration, 1}}, error));
        EXPECT_FALSE(error.empty());
        EXPECT_EQ(parent.builds, 0);
        EXPECT_EQ(parent.instantiations, 0);
    }
}

/** @test Incomplete initialization never reaches a mutating backend construction call. */
TEST(DeviceGenerationGraphProgram, InvalidInitializationFailsBeforeConstruction)
{
    std::array<int, kDeviceGenerationControlCount> rows{};
    for (int failure = 0; failure < 6; ++failure)
    {
        SCOPED_TRACE(failure);
        ProgramGraph parent, initial, body;
        DeviceControlledLoopFragment initialization{.name = "prefill sample", .capture = &initial};
        const DeviceControlledLoopFragment iteration{.name = "decode", .capture = &body};
        if (failure == 0) initialization.capture = nullptr;
        if (failure == 1) initialization.capture = &parent;
        if (failure == 2) initial.recorded_nodes = 0;
        if (failure == 3) initialization.name = "";
        if (failure == 4) {
            initialization.execution = DeviceControlledLoopFragmentExecution::IfDeviceWordZero;
            initialization.condition_word_device = nullptr;
        }
        if (failure == 5) {
            initialization.execution = DeviceControlledLoopFragmentExecution::IfDeviceSelectorAtLeast;
            initialization.minimum_selector = 1;
        }
        std::string error;
        EXPECT_FALSE(DeviceGenerationGraphProgram::native(parent,
            {DeviceGenerationPolicy::ordinary(), rows.data(), kDeviceGenerationControlCount, 1},
            {.initialization = {&initialization, 1}, .iteration = {&iteration, 1}}, error));
        EXPECT_FALSE(error.empty());
        EXPECT_EQ(parent.builds, 0);
        EXPECT_EQ(parent.instantiations, 0);
    }
}

/** @brief A rejected nested compiler cannot clear the enclosing recording's ownership. */
TEST(DeviceGenerationGraphProgram, RejectsNestedRecordingWithoutTouchingOuterOwner)
{
    for (auto type : {DeviceType::CUDA, DeviceType::ROCm}) {
        ProgramGraph parent, source;
        source.executable = true;
        llaminar2::testing::MockWorkerGPUContext context(7);
        TicketBackend backend(type);
        std::array<int, kDeviceGenerationControlCount> rows{};
        DeviceGenerationDispatchTicket ticket{};
        const DeviceGenerationGraphControl control{DeviceGenerationPolicy::ordinary(), rows.data(), kDeviceGenerationControlCount, 1};
        const DeviceControlledLoopFragment fragment{.name = "transaction", .capture = &source};
        std::string error;
        const auto build_ticket = [&] {
            return DeviceGenerationGraphProgram::ticketPublisher(
                parent, context, backend, {type, 7}, control, nullptr, &ticket, error);
        };
        context.setGraphCaptureActive(true);
        EXPECT_FALSE(build_ticket());
        EXPECT_TRUE(context.isDeviceGraphCaptureActive());
        context.setGraphCaptureActive(false);
        {
            GraphCaptureGuard enclosing;
            EXPECT_FALSE(build_ticket());
            EXPECT_FALSE(DeviceGenerationGraphProgram::native(parent, control, {.iteration = {&fragment, 1}}, error));
            EXPECT_TRUE(isGraphCaptureActive());
        }
        EXPECT_EQ(parent.begins, 0);
        EXPECT_EQ(parent.ends, 0);
        EXPECT_EQ(parent.builds, 0);
        EXPECT_EQ(parent.instantiations, 0);
        EXPECT_EQ(backend.calls, 0);
    }
}

TEST(DeviceGenerationGraphProgram, TicketCaptureClosesOnSuccessFailureAndException)
{
    for (auto type : {DeviceType::CUDA, DeviceType::ROCm})
    for (int failure = 0; failure < 5; ++failure) {
        SCOPED_TRACE(failure);
        ProgramGraph parent;
        llaminar2::testing::MockWorkerGPUContext context(7);
        TicketBackend backend(type);
        backend.context = &context;
        backend.expected_stream = parent.stream;
        backend.enqueue_ok = failure != 1;
        backend.throw_on_enqueue = failure == 2;
        parent.begin_ok = failure != 3;
        parent.instantiate_ok = failure != 4;
        std::array<int, kDeviceGenerationControlCount> rows{};
        DeviceGenerationDispatchTicket ticket{};
        const DeviceGenerationGraphControl control{DeviceGenerationPolicy::ordinary(), rows.data(), kDeviceGenerationControlCount, 1};
        std::string error;
        auto build = [&] { return DeviceGenerationGraphProgram::ticketPublisher(
            parent, context, backend, {type, 7}, control, nullptr, &ticket, error); };
        if (failure == 2) EXPECT_THROW(build(), std::runtime_error);
        else EXPECT_EQ(build(), failure == 0) << error;
        EXPECT_EQ(parent.begins, 1);
        EXPECT_EQ(parent.ends, failure == 3 ? 0 : 1);
        EXPECT_EQ(parent.instantiations, failure == 0 || failure == 4 ? 1 : 0);
        EXPECT_FALSE(context.isDeviceGraphCaptureActive());
        EXPECT_FALSE(isGraphCaptureActive());
    }
}
} // namespace
