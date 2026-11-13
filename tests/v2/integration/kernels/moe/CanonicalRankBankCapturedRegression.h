/**
 * @file CanonicalRankBankCapturedRegression.h
 * @brief Shared real-bridge retained rank-bank correctness and isolated native-event A/B.
 *
 * Both native owners call the original production native launch bridge. A separately
 * compiled frozen original entry is a diagnostic oracle only. PMA-backed
 * transfer buffers retain immutable pointers, explicit streams, route slots
 * and both guards across full/partial/empty/reset histories. Timing excludes
 * setup and byte checks;
  profiler modes construct no timing events.
 */
#pragma once
#include "backends/BackendManager.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "transfer/TransferEngine.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <string>
#include <utility>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace llaminar2::test::canonical_rank_bank
{
/** @brief Fixture-only execution selection, never a production dispatch switch. */
enum class Run { Functional, Paired, ProfileOriginal, ProfileProduction };
/** @brief Frozen original native launch signature, identical to the original bridge geometry. */
using NativeLauncher = bool (*)(const float *, float *, int, int, int, int, int, const int *, int, void *);
/** @brief Reject incomplete fixture infrastructure before observing numerical success.
 * @param value Required infrastructure predicate.
 * @param message Precise failing lifecycle/ownership boundary.
 * @throws std::runtime_error When the predicate is false.
 */
inline void require(bool value, const char *message)
{
    if (!value)
        throw std::runtime_error(message);
}
/** @brief Persistent device storage and executables, retired before their pointers disappear. */
struct Storage
{
    IBackend *backend;
    DeviceId device;
    void *stream;
    std::vector<std::shared_ptr<DeviceTransferBuffer>> owners;
    std::unique_ptr<IGPUGraphCapture> production, original;
    /** @brief Drain terminal test work on every exit before graph/buffer destruction. */
    ~Storage()
    {
        (void)backend->synchronizeStream(stream, device.ordinal);
        original.reset();
        production.reset();
    }
    /** @brief Admit setup-only storage through the public transfer allocator.
     * @tparam T Native element type.
     * @param count Exact retained element capacity.
     * @return Stable device pointer owned by this storage lifetime.
     */
    template <class T> T *allocate(std::size_t count)
    {
        auto owner = TransferEngine::instance().allocateDeviceTransferBuffer(count*sizeof(T), device);
        require(owner && owner->isBound(), "Rank-bank fixture storage admission failed");
        auto *pointer = static_cast<T *>(owner->mutableDeviceData());
        owners.push_back(std::move(owner));
        return pointer;
    }
    /** @brief Upload immutable inputs/dirty state outside capture and timing.
     * @tparam T Native input element type.
     * @param pointer Existing retained device allocation.
     * @param values Complete original host fixture values.
     */
    template <class T> void upload(T *pointer, const std::vector<T> & values)
    {
        require(backend->hostToDeviceOnStream(pointer, values.data(), values.size()*sizeof(T), device.ordinal, stream), "Rank-bank fixture upload failed");
        require(backend->synchronizeStream(stream, device.ordinal), "Rank-bank setup upload retirement failed");
    }
    /** @brief Collect terminal bytes after the exact producer stream.
     * @tparam T Native output element type.
     * @param pointer Retained original device allocation.
     * @param count Complete observed element extent, including guards.
     * @return Terminal host observation; never an inference control input.
     */
    template <class T> std::vector<T> download(const T *pointer, std::size_t count)
    {
        std::vector<T> result(count);
        require(backend->deviceToHostOnStream(result.data(), pointer, count*sizeof(T), device.ordinal, stream), "Rank-bank fixture download failed");
        require(backend->synchronizeStream(stream, device.ordinal), "Rank-bank terminal observation failed");
        return result;
    }
};
/** @brief Native events owned only by the unprofiled paired measurement mode. */
struct Events
{
    Storage &storage;
    void *first = nullptr;
    void *last = nullptr;
    /** @brief Admit persistent timing events before collecting any samples.
     * @param s Retained device/stream owner that outlives both events.
     */
    explicit Events(Storage &s):storage(s)
    {
        first = s.backend->createTimingEvent(s.device.ordinal);
        last = s.backend->createTimingEvent(s.device.ordinal);
        if (!first || !last)
        {
            if (first)
                s.backend->destroyEvent(first, s.device.ordinal);
            if (last)
                s.backend->destroyEvent(last, s.device.ordinal);
            throw std::runtime_error("Rank-bank fixture event admission failed");
        }
    }
    /** @brief Drain pending event use before destruction on every exit. */
    ~Events()
    {
        (void)storage.backend->synchronizeStream(storage.stream, storage.device.ordinal);
        storage.backend->destroyEvent(last, storage.device.ordinal);
        storage.backend->destroyEvent(first, storage.device.ordinal);
    }
    /** @brief Measure one complete captured batch in amortized microseconds per publication.
     * @param graph Exact retained original or production variant.
     * @param copies Complete operations per retained graph.
     * @return Native-event microseconds, including device scheduling and all required bank stores.
     */
    float sample(IGPUGraphCapture &graph, int copies)
    {
        auto &s = storage;
        require(s.backend->recordEvent(first, s.device.ordinal, s.stream), "Rank-bank first event failed");
        require(graph.launch(), "Rank-bank timed replay failed");
        require(s.backend->recordEvent(last, s.device.ordinal, s.stream), "Rank-bank last event failed");
        require(s.backend->waitForEvent(last, s.device.ordinal), "Rank-bank event observation failed");
        float ms = 0;
        require(s.backend->eventElapsedTimeMs(first, last, s.device.ordinal, &ms) && std::isfinite(ms) && ms>0, "Rank-bank invalid elapsed time");
        return ms*1000.f/static_cast<float>(copies);
    }
};
/** @brief Prove exact original bytes and retained overwrite semantics on a real production bridge.
 * @tparam GraphFactory Factory returning the native graph owner.
 * @param production Actual production native bridge, without a second implementation.
 * @param device Exact selected backend endpoint.
 * @param stream Non-default original stream.
 * @param factory Native graph factory preserving that device/stream owner.
 * @param original Frozen old kernel bridge, unavailable to production dispatch.
 * @param rows Physical capture rows.
 * @param hidden Actual row width, including scalar tails.
 * @param participant Local bank in the unchanged four-bank suffix.
 * @param run Functional, interleaved events or one-variant event-free profiling.
 * @details Top10 and four participant banks match original QFN prefill. The
 * complete routed prefix remains untouched. Device effective rows change data,
 * never geometry, pointers, graph topology or launch count.
 */
template <class GraphFactory>
void oneCase(NativeLauncher production, DeviceId device, void *stream, GraphFactory factory,
    NativeLauncher original, int rows, int hidden, int participant, Run run)
{
    constexpr int topK = 10, participants = 4, copies = 32;
    constexpr std::size_t guard = 32;
    const bool profile = run == Run::ProfileOriginal || run == Run::ProfileProduction;
    const bool oldOnly = run == Run::ProfileOriginal;
    const int operations = run == Run::Functional?1:copies;
    const auto sourceCount = std::size_t(rows)*hidden, payloadCount = sourceCount*(topK+participants);
    const auto sourceTotal = sourceCount+2*guard, payloadTotal = payloadCount+2*guard;
    const float sentinel = std::bit_cast<float>(std::uint32_t{0x4b123456});
    // Staging owners precede stream-draining Storage on every exceptional exit.
    std::vector<float> source(sourceTotal, sentinel), initial(payloadTotal, sentinel), expected(payloadTotal);
    std::vector<int> liveHost(1, rows);
    auto *backend = getBackendFor(device);
    require(backend != nullptr && stream, "Rank-bank explicit native owner missing");
    Storage s{backend, device, stream, {}, {}, {}};
    auto *input = s.allocate<float>(sourceTotal), *output = s.allocate<float>(payloadTotal), *oldOutput = s.allocate<float>(payloadTotal);
    auto *live = s.allocate<int>(1);
    const auto capture = [&](bool historical){
        auto graph = factory(stream);
        require(bool(graph), "Rank-bank graph admission failed");
        ScopedBackendGraphCapture recording(*graph, "Original shared rank-bank publication");
        require(recording.begin(), "Rank-bank capture failed");
        for (int i = 0;i<operations;++i)
        {
            if (historical)
            {
                require(original(input + guard, oldOutput + guard, rows, topK, hidden,
                            participant, participants, live, device.ordinal, stream),
                    "Original rank-bank submission failed");
            }
            else
            {
                require(production(input + guard, output + guard, rows, topK, hidden,
                            participant, participants, live, device.ordinal, stream),
                    "Production rank-bank submission failed");
            }
        }
        recording.finish();
        std::vector<GPUGraphKernelNodeInfo> nodes;
        std::string detail;
        require(graph->inspectKernelNodes(nodes, &detail), detail.c_str());
        require(nodes.size() == std::size_t(operations) && graph->nodeCount() == nodes.size(), "Rank-bank graph contains hidden or missing operations");
        const unsigned work = device.is_cuda()?(unsigned(hidden)+3)/4:((hidden&3) == 0?unsigned(hidden)/4:unsigned(hidden));
        const unsigned gridX = (work+255)/256;
        for (const auto &node:nodes)
        {
            require(node.valid() && node.name_resolved && node.grid_x == gridX && node.grid_y == unsigned(rows*participants) && node.grid_z == 1 &&
                node.block_x == 256 && node.block_y == 1 && node.block_z == 1 && node.dynamic_shared_memory_bytes == 0,
                "Rank-bank native entry/geometry differs from original publication");
            require(node.name.find(historical?"originalSharedRankBankKernel":"publish_shared_expert_rank_bank_kernel") != std::string::npos,
                "Rank-bank captured a foreign kernel");
            if (!historical)
                require(node.local_memory_bytes_per_thread == 0, "Rank-bank repaired entry retains addressable private storage");
        }
        const auto &n = nodes.front();
        // Compiler-owned static shared reservations are architecture-specific;
        // retain both measured footprints without imposing a universal zero.
        std::cout << "RANK_BANK_RESOURCE," << device.toString() << ','
                  << (historical ? "original" : "production")
                  << ",registers=" << n.registers_per_thread
                  << ",static_shared_bytes=" << n.static_shared_memory_bytes
                  << ",dynamic_shared_bytes=" << n.dynamic_shared_memory_bytes
                  << ",private_bytes=" << n.local_memory_bytes_per_thread
                  << ",max_active_blocks=" << n.max_active_blocks_per_sm << '\n';
        std::cout<<"RANK_BANK_INVENTORY,"<<device.toString()<<','<<(historical?"original":"production")<<','<<rows<<','<<hidden<<','<<participant<<','<<topK<<','<<participants<<','<<operations<<','<<n.name<<','<<n.grid_x<<','<<n.grid_y<<','<<n.block_x<<','<<n.registers_per_thread<<','<<n.local_memory_bytes_per_thread<<'\n';
        require(graph->instantiate(), "Rank-bank graph instantiate failed");
        return graph;
    };
    if (!profile || !oldOnly)
        s.production = capture(false);
    if (!profile || oldOnly)
        s.original = capture(true);
    const auto setup = [&](int history, int active, bool special){
        const std::array<std::uint32_t, 12> words{0u, 0x80000000u, 1u, 0x80000001u, 0x007fffffu, 0x00800000u,
            0x7f800000u, 0xff800000u, 0x7fc01234u, 0xffc05678u, 0x7f801234u, 0xff805678u};
        for (std::size_t i = 0;i<sourceCount;++i)
            source[guard+i] = special?std::bit_cast<float>(words[(i+history)%words.size()]):float(int((i*17+history*11)%97)-48)/29.f;
        for (std::size_t i = 0;i<payloadCount;++i)
            initial[guard+i] = std::bit_cast<float>(std::uint32_t{0x3f000000u}+std::uint32_t((i+history*31)%0x007fffffu));
        expected = initial;
        const int effective = std::clamp(active, 0, rows);
        for (int bank = 0; bank < participants; ++bank)
        for (int row = 0; row < rows; ++row)
        {
            auto *destination = expected.data()+guard+sourceCount*topK+std::size_t(bank)*sourceCount+std::size_t(row)*hidden;
            if (bank == participant && row<effective)
                std::memcpy(destination, source.data()+guard+std::size_t(row)*hidden, std::size_t(hidden)*sizeof(float));
            else
                std::memset(destination, 0, std::size_t(hidden)*sizeof(float));
        }
        liveHost[0] = active;
        s.upload(input, source);
        s.upload(output, initial);
        s.upload(oldOutput, initial);
        s.upload(live, liveHost);
    };
    const auto compare = [&]{
        const auto got = s.download(output, payloadTotal), prior = s.download(oldOutput, payloadTotal), unchanged = s.download(input, sourceTotal);
        EXPECT_EQ(std::memcmp(got.data(), expected.data(), payloadTotal*sizeof(float)), 0)<<"M="<<rows<<" H="<<hidden<<" rank="<<participant<<" live="<<liveHost[0];
        EXPECT_EQ(std::memcmp(prior.data(), expected.data(), payloadTotal*sizeof(float)), 0)<<"original M="<<rows<<" H="<<hidden;
        EXPECT_EQ(std::memcmp(got.data(), prior.data(), payloadTotal*sizeof(float)), 0);
        EXPECT_EQ(std::memcmp(unchanged.data(), source.data(), sourceTotal*sizeof(float)), 0);
    };
    if (profile)
    {
        setup(0, rows, false);
        // All setup DMA is retired; no timing event is ever created.
        auto &graph = oldOnly?*s.original:*s.production;
        std::cout<<"RANK_BANK_PROFILE,"<<device.toString()<<','<<(oldOnly?"original":"production")<<",M="<<rows<<",H="<<hidden<<",topK=10,banks=4,copies=32,warmups=8,observations=31,timing_events=0,repeated_addresses=1\n";
        for (int i = 0;i<8+31;++i)
            require(graph.launch(), "Rank-bank selected profiler replay failed");
        require(backend->synchronizeStream(stream, device.ordinal), "Rank-bank profiler terminal retirement failed");
        return;
    }
    const std::vector<int> histories = run == Run::Functional?std::vector<int>{rows, std::min(rows, 127), 0, std::min(rows, 3), rows, -1, rows+3}:std::vector<int>{rows};
    for (std::size_t history = 0;history<histories.size();++history)
    {
        setup(int(history), histories[history], run == Run::Functional);
        require(s.production->launch() && s.original->launch(), "Rank-bank dirty retained replay failed");
        compare();
    }
    if (run != Run::Paired)
        return;
    require(!::testing::Test::HasFailure(), "Rank-bank byte proof failed before timing");
    Events events(s);
    for (int i = 0; i < 8; ++i)
    {
        (void)events.sample(*s.original, copies);
        (void)events.sample(*s.production, copies);
    }
    for (int round = 0; round < 31; ++round)
    for (int slot = 0; slot < 4; ++slot)
    {
        const bool historical = (slot == 0 || slot == 3) != bool(round&1);
        const auto us = events.sample(historical?*s.original:*s.production, copies);
        std::cout<<"RANK_BANK_EVENT,"<<device.toString()<<','<<rows<<','<<hidden<<','<<round<<','<<slot<<','<<(historical?"original":"production")<<','<<copies<<','<<us<<'\n';
    }
    compare();
}
/** @brief Existing native owners share the exact shape sweep and one-shape performance policy.
 * @tparam GraphFactory Factory returning the native graph owner.
 * @param production Original production native bridge.
 * @param device Exact backend endpoint.
 * @param stream Explicit original execution stream.
 * @param factory Native graph constructor for that device and stream.
 * @param original Frozen original native launcher.
 * @param mode Explicit fixture diagnostic demand.
 */
template <class GraphFactory>
void run(NativeLauncher production, DeviceId device, void *stream, GraphFactory factory, NativeLauncher original, Run mode)
{
    if (mode != Run::Functional)
    {
        oneCase(production, device, stream, factory, original, 512, 2560, 0, mode);
        return;
    }
    for (int participant = 0;participant<4;++participant)
        oneCase(production, device, stream, factory, original, 512, 2560, participant, mode);
    for (const auto shape:std::array{std::array{1, 1}, std::array{3, 3}, std::array{31, 4}, std::array{33, 7},
        std::array{511, 2560}, std::array{513, 2560}, std::array{17, 2559}, std::array{17, 2561}})
        oneCase(production, device, stream, factory, original, shape[0], shape[1], 2, mode);
}
} // namespace llaminar2::test::canonical_rank_bank
