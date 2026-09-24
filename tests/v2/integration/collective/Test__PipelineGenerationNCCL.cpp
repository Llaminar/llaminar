/**
 * @file Test__PipelineGenerationNCCL.cpp
 * @brief Native pipeline arrival and terminal-command ordering on real CUDA devices.
 *
 * The tail alone owns a request budget. Its follower owns only local work and
 * the command received over NCCL. Retained graphs carry activations forward
 * and commands back without host iteration, polling, or command repair. This
 * is a model-free protocol regression, not a whole-model PP certificate.
 */
#include <gtest/gtest.h>

#ifdef HAVE_CUDA
#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "backends/cuda/CUDAGraphCapture.h"
#include "collective/LocalTPContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <cuda_runtime.h>
#include <array>
#include <atomic>
#include <barrier>
#include <memory>
#include <thread>
#include <vector>

namespace
{
using namespace llaminar2;

/**
 * @brief Fixture storage and stream whose lifetime exceeds all imported graphs.
 *
 * The four command words are the only tail state sent to followers. The local
 * commit clock measures executed work independently; it is never copied from
 * the tail, compared as a response ledger, or used to decide follower exit.
 */
class PipelineParticipant
{
public:
    enum Word : int { Healthy, Terminal, Selector, Error, CommandWords,
        Committed = CommandWords, Remaining, Due, Advanced, Selected, Initialized, InitialStop,
        Activation, Words };
    int device;
    IBackend &backend;
    cudaStream_t stream = nullptr;
    uint32_t *data = nullptr;
    std::unique_ptr<CUDAGraphCapture> arrival, initialization, work, selected, publication, parent;

    /** @brief Bind ownership without allocating before the test can report errors. */
    PipelineParticipant(int ordinal, IBackend &owner) : device(ordinal), backend(owner) {}

    /** @brief Retire parents before borrowed recordings, storage and stream. */
    ~PipelineParticipant()
    {
        cudaSetDevice(device);
        parent.reset();
        publication.reset();
        selected.reset();
        work.reset();
        initialization.reset();
        arrival.reset();
        if (data) backend.free(data, device);
        if (stream) cudaStreamDestroy(stream);
    }

    /** @return True after allocating exact fixture storage through the backend authority. */
    bool prepare()
    {
        if (cudaSetDevice(device) != cudaSuccess ||
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
            return false;
        data = static_cast<uint32_t *>(backend.allocate(Words * sizeof(uint32_t), device));
        if (!data) return false;
        // Deliberately invalid old commands must not decide first entry.
        const std::array<uint32_t, Words> stale{0u, 1u, 999u};
        if (cudaMemcpyAsync(data, stale.data(), sizeof(stale), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess)
            return false;
        for (auto *graph : {&arrival, &initialization, &work, &selected, &publication, &parent})
            *graph = std::make_unique<CUDAGraphCapture>(stream, device);
        return true;
    }

    /** @return True after one real captured work clock advances, without a host counter. */
    bool recordWorkClock()
    {
        return backend.enqueuePublishSerialDecodeCommitBoundary(data + Committed,
                   data + Remaining, data + Due, data + Advanced, device, stream) &&
            backend.enqueueAcknowledgeDecodeCommitBoundary(data + Remaining,
                data + Due, data + Advanced, device, stream);
    }
};

/**
 * @test Arrival precedes both native predicates, including replay after terminal.
 *
 * Reverse the vocabulary owner's physical ordinal and graph submission order.
 * Zero work models first-token EOS; positive budgets cover short and sustained
 * requests. A follower begins with an unhealthy terminal command and later
 * retains the preceding request's terminal command. Neither is host-reset.
 */
TEST(PipelineGenerationNCCL, TailCommandPrecedesPredicateAndRetiresEveryParticipant)
{
    auto *backend = getCUDABackend();
    ASSERT_NE(backend, nullptr);
    if (backend->deviceCount() < 2)
        GTEST_SKIP() << "Requires two CUDA devices";
    using P = PipelineParticipant;
    constexpr std::array<uint32_t, 6> budgets{0u, 1u, 2u, 17u, 31u, 384u};
    for (const int tail : {0, 1})
    for (const bool dynamic : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << "tail=" << tail << " dynamic=" << dynamic);
        auto context = createLocalTPContext(
            {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}, {}, CollectiveBackendType::NCCL);
        ASSERT_NE(context, nullptr);
        std::array<std::unique_ptr<P>, 2> participants{
            std::make_unique<P>(0, *backend), std::make_unique<P>(1, *backend)};
        for (auto &p : participants) ASSERT_TRUE(p->prepare());
        std::barrier boundary(2);
        std::atomic<bool> failed{false};
        const auto check = [&](bool value) {
            EXPECT_TRUE(value);
            if (!value) failed.store(true);
            return value;
        };
        const auto record = [&](int rank) {
            auto &p = *participants[rank];
            check(cudaSetDevice(rank) == cudaSuccess);
            const auto command = [&] {
                return context->broadcastRawOnStream(p.data, p.data, P::CommandWords,
                    CollectiveDataType::INT32, tail, rank, p.stream, "pipeline_tail_command");
            };
            // Warm up communicator infrastructure, not model state or control.
            check(command());
            check(cudaStreamSynchronize(p.stream) == cudaSuccess);
            boundary.arrive_and_wait();
            // Both participants enter/leave every recording together, even on
            // error. A failed capture must not strand its peer at a collective.
            const auto capture = [&](CUDAGraphCapture &graph, auto &&enqueue) {
                const bool began = graph.beginCapture();
                check(began);
                boundary.arrive_and_wait();
                const bool admitted = !failed.load();
                boundary.arrive_and_wait();
                if (admitted) {
                    GraphCaptureGuard scope;
                    check(enqueue());
                }
                boundary.arrive_and_wait();
                if (began) check(graph.endCapture());
                boundary.arrive_and_wait();
            };
            capture(*p.arrival, command);
            capture(*p.initialization, [&] {
                // Model a tail's prefill sampler terminating the request before
                // any forward. Arrival must remain outside this admitted IF.
                return cudaMemsetAsync(p.data + P::Initialized, 2, sizeof(uint32_t), p.stream) == cudaSuccess &&
                    cudaMemcpyAsync(p.data + P::Terminal, p.data + P::InitialStop,
                        sizeof(uint32_t), cudaMemcpyDeviceToDevice, p.stream) == cudaSuccess;
            });
            capture(*p.work, [&] {
                if (rank != tail && !p.recordWorkClock()) return false;
                // The head's actual local count travels as an activation, not
                // as a host-maintained proxy for either stage's progress.
                const CollectiveP2POp edge{
                    .kind = rank == tail ? CollectiveP2POpKind::Recv : CollectiveP2POpKind::Send,
                    .send_buffer = rank == tail ? nullptr : p.data + P::Committed,
                    .recv_buffer = rank == tail ? p.data + P::Activation : nullptr,
                    .count = 1, .dtype = CollectiveDataType::INT32, .peer = 1 - rank};
                if (!context->groupedP2PRawOnStream({edge}, rank, p.stream, "pipeline_activation")) return false;
                return rank != tail || p.recordWorkClock();
            });
            capture(*p.selected, [&] {
                return cudaMemsetAsync(p.data + P::Selected, 1, sizeof(uint32_t), p.stream) == cudaSuccess;
            });
            capture(*p.publication, [&] {
                if (rank == tail && cudaMemcpyAsync(p.data + P::Terminal, p.data + P::Due,
                        sizeof(uint32_t), cudaMemcpyDeviceToDevice, p.stream) != cudaSuccess)
                    return false;
                return command(); // Terminal decisions must also reach every follower.
            });
            if (failed.load()) return;
            const DeviceControlledLoopEntryFragment entry[] = {{"receive tail command", p.arrival.get()}};
            const DeviceControlledLoopFragment initialization[] = {{"admitted initialization", p.initialization.get()}};
            std::vector<DeviceControlledLoopFragment> body{{"local work and activation", p.work.get()}};
            if (dynamic) body.push_back({.name = "selected local work", .capture = p.selected.get(),
                .execution = DeviceControlledLoopFragmentExecution::IfDeviceSelectorAtLeast, .minimum_selector = 2});
            body.push_back({"publish next or terminal command", p.publication.get()});
            const DeviceControlledLoopProgram program{.entry = entry,
                .initialization = rank == tail ? std::span<const DeviceControlledLoopFragment>(initialization)
                                              : std::span<const DeviceControlledLoopFragment>{},
                .iteration = body};
            const DeviceControlledLoopPredicate predicate{
                .control_rows_device = reinterpret_cast<const int *>(p.data),
                .control_stride = P::CommandWords, .request_count = 1,
                .healthy_index = P::Healthy, .complete_index = P::Terminal};
            const bool built = dynamic
                ? p.parent->buildDeviceControlledSelectorWhileLoop(program, predicate, {
                    .control_rows_device = reinterpret_cast<int *>(p.data),
                    .control_stride = P::CommandWords, .request_count = 1,
                    .healthy_index = P::Healthy, .complete_index = P::Terminal,
                    .selector_index = P::Selector, .error_index = P::Error,
                    .minimum_selector = 1, .maximum_selector = 2, .invalid_selector_error = 7})
                : p.parent->buildDeviceControlledWhileLoop(program, predicate);
            check(built && p.parent->instantiate());
        };
        std::thread first(record, 0), second(record, 1);
        first.join();
        second.join();
        ASSERT_FALSE(failed.load());
        for (int replay = 0; replay < 20; ++replay)
        {
            const uint32_t budget = budgets[static_cast<size_t>(replay) % budgets.size()];
            const uint32_t selector = 1u + static_cast<uint32_t>(replay % 2);
            const bool initial_stop = replay % 7 == 3;
            const bool unhealthy = replay % 9 == 5;
            const uint32_t expected_work = initial_stop || unhealthy ? 0u : budget;
            SCOPED_TRACE(::testing::Message() << "replay=" << replay << " budget=" << budget);
            const std::array<uint32_t, P::CommandWords> command{
                unhealthy ? 0u : 1u, budget == 0 ? 1u : 0u, selector, unhealthy ? 7u : 0u};
            // Only the tail receives new request policy. Followers retain old
            // commands; only their own model/work state is reset on admission.
            std::array<std::array<uint32_t, P::Words - P::CommandWords>, 2> local{};
            for (int rank = 0; rank < 2; ++rank) {
                auto &p = *participants[rank];
                local[rank][P::Remaining - P::CommandWords] = rank == tail ? budget : 4096u;
                local[rank][P::InitialStop - P::CommandWords] = initial_stop ? 1u : 0u;
                ASSERT_EQ(cudaSetDevice(rank), cudaSuccess);
                ASSERT_EQ(cudaMemcpyAsync(p.data + P::CommandWords, local[rank].data(), sizeof(local[rank]),
                    cudaMemcpyHostToDevice, p.stream), cudaSuccess);
                if (rank == tail) ASSERT_EQ(cudaMemcpyAsync(p.data, command.data(), sizeof(command),
                    cudaMemcpyHostToDevice, p.stream), cudaSuccess);
            }
            for (int slot = 0; slot < 2; ++slot)
                ASSERT_TRUE(participants[(slot + replay) % 2]->parent->launch());
            for (int rank = 0; rank < 2; ++rank) {
                auto &p = *participants[rank];
                ASSERT_EQ(cudaSetDevice(rank), cudaSuccess);
                ASSERT_EQ(cudaStreamSynchronize(p.stream), cudaSuccess);
                std::array<uint32_t, P::Words> observed{};
                ASSERT_TRUE(backend->deviceToHost(observed.data(), p.data, sizeof(observed), rank, p.stream));
                EXPECT_EQ(observed[P::Healthy], unhealthy ? 0u : 1u);
                EXPECT_EQ(observed[P::Terminal], unhealthy ? command[P::Terminal] : 1u);
                EXPECT_EQ(observed[P::Selector], selector);
                EXPECT_EQ(observed[P::Error], unhealthy ? 7u : 0u);
                EXPECT_EQ(observed[P::Committed], expected_work);
                EXPECT_EQ(observed[P::Remaining], (rank == tail ? budget : 4096u) - expected_work);
                EXPECT_EQ(observed[P::Initialized], rank == tail && budget && !unhealthy ? 0x02020202u : 0u);
                EXPECT_EQ(observed[P::Selected], expected_work && dynamic && selector == 2u ? 0x01010101u : 0u);
                if (rank == tail) EXPECT_EQ(observed[P::Activation], expected_work);
            }
            // An absorbing terminal replay still participates in arrival, but
            // executes no local body. Both devices must finish without reset.
            for (auto &p : participants) ASSERT_TRUE(p->parent->launch());
            for (auto &p : participants) {
                ASSERT_EQ(cudaSetDevice(p->device), cudaSuccess);
                ASSERT_EQ(cudaStreamSynchronize(p->stream), cudaSuccess);
                uint32_t committed = 0;
                ASSERT_TRUE(backend->deviceToHost(&committed, p->data + P::Committed,
                    sizeof(committed), p->device, p->stream));
                EXPECT_EQ(committed, expected_work);
            }
        }
    }
}
} // namespace
#endif
