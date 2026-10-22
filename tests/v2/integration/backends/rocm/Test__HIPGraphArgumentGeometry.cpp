/**
* @file Test__HIPGraphArgumentGeometry.cpp
* @brief Exact heavy-by-value HIP graph geometry, clone/update and family-budget proof.
*
* Four retained native definitions share one original output owner and one typed
* PMA family. Every kernel directly consumes its original argument words; there
* is no automatic packet copy, private array, model or hot allocation. Native
* code-object kernarg metadata must accompany the focused runtime proof because
* hidden ABI arguments can exceed the declared4096-byte packet.
*/
#include "backends/GPUGraphMemoryContract.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "planning/PlanningObservedResource.h"
#include "utils/MPIContext.h"
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include "HIPGraphArgumentGeometryABI.h"
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace llaminar2::hip_argument_geometry_test {
    /** @brief Reject any native error without alternate execution or enlarged reservations. */
    inline void check(hipError_t status, const char *what)
    {
        if (status != hipSuccess) throw std::runtime_error(std::string(what)+": "+hipGetErrorString(status));
    }
    /**
    * @brief Author independent original packet contents and expected modulo32-bit lane sums.
    * @param output Original allocation shared by mutually exclusive retained variants.
    * @param slot Original graph node identity.
    * @param changed Whether this node belongs to the explicit updated variant.
    * @return Full source packet with distinct first/interior/last words.
    */
    inline Packet packet(std::uint32_t *output, unsigned slot, bool changed = false)
    {
        Packet p{};
        p.output = output;
        p.slot = slot;
        p.salt = 0x10203040u+slot+(changed?17u:0u);
        for (unsigned i = 0; i < 1020; ++i)p.words[i] = 0x9e3779b9u*(1+i)+slot*31u+(changed?i+1:0);
        return p;
    }
    /** @brief One native executable retires before its original graph definition. */
    struct Variant {
        hipGraph_t graph = nullptr;
        hipGraphExec_t exec = nullptr;
        ~Variant()
        {
            if (exec)(void)hipGraphExecDestroy(exec);
            if (graph)(void)hipGraphDestroy(graph);
        }
    };
    /**
    * @brief Own exact terminal, original allocations and PMA claims through final retirement.
    * @note Unknown pending work terminates the disposable test rather than freeing live owners.
    */
    struct Owner {
        hipStream_t stream = nullptr;
        hipEvent_t terminal = nullptr;
        std::uint32_t *output = nullptr;
        PhysicalMemoryAllocationLease data_claim;
        PhysicalMemoryOwnerReservation graphs_claim;
        std::array<Variant, Variants> variants;
        bool pending = false;
        ~Owner()
        {
            if (pending)
            {
                std::fputs("FATAL unobserved HIP argument-geometry work\n", stderr);
                std::_Exit(EXIT_FAILURE);
            }
            for (auto &v:variants)
            {
                if (v.exec)
                {
                    (void)hipGraphExecDestroy(v.exec);
                    v.exec = nullptr;
                }
                if (v.graph)
                {
                    (void)hipGraphDestroy(v.graph);
                    v.graph = nullptr;
                }
            }
            if (output)(void)hipFree(output);
            if (terminal)(void)hipEventDestroy(terminal);
            if (stream)(void)hipStreamDestroy(stream);
        }
        /** @brief Join only the exact terminal with the ordinary30-second test boundary. */
        void await()
        {
            const auto deadline = std::chrono::steady_clock::now()+std::chrono::seconds(30);
            while (true)
            {
                const auto s = hipEventQuery(terminal);
                if (s == hipSuccess)break;
                if (s != hipErrorNotReady)check(s, "terminal query");
                if (std::chrono::steady_clock::now() >= deadline)throw std::runtime_error("argument-geometry terminal timeout");
                std::this_thread::yield();
            }
            pending = false;
        }
    };
    /**
    * @brief Materialize one actual four-executable family and prove original/clone/update/child bytes.
    * @param ordinal Exact original worker device; caller runs on that worker.
    * @note Infrastructure-only free-memory observations match the production graph observer;
    * no memory/count observation occurs in captured work or between graph nodes.
    */
    inline void prove(int ordinal)
    {
        const DeviceId device = DeviceId::rocm(ordinal);
        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        if (!inventory || mpi->world_size() != 1)throw std::runtime_error("Original one-rank inventory required");
        constexpr std::size_t data_bytes = (Nodes*Threads+16)*sizeof(std::uint32_t);
        const auto budget = Variants*GPUGraphMemoryContract::reservationBytesPerExecutable(device);
        PhysicalMemoryPlanBuilder plan;
        const auto resource = planningObservedResource(inventory->ranks.at(mpi->rank()), device);
        plan.add(resource, PhysicalMemoryOwner::ActivationTransportStaging, data_bytes);
        plan.add(resource, PhysicalMemoryOwner::NativeGraphExecutable, budget);
        auto authority = std::make_shared<PhysicalMemoryAuthority>(std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.build()), mpi->rank());
        {
            Owner owner;
            owner.data_claim = authority->claimNewAllocation(device, PhysicalMemoryOwner::ActivationTransportStaging, data_bytes);
            owner.graphs_claim = authority->reserveNewAllocations(device, PhysicalMemoryOwner::NativeGraphExecutable, budget);
            check(hipSetDevice(ordinal), "select original device");
            check(hipStreamCreateWithFlags(&owner.stream, hipStreamNonBlocking), "exact stream");
            check(hipEventCreateWithFlags(&owner.terminal, hipEventDisableTiming), "terminal");
            check(hipMalloc(&owner.output, data_bytes), "original output");
            std::array<Packet, Nodes> originals;
            std::array<hipGraphNode_t, Nodes> nodes{};
            check(hipGraphCreate(&owner.variants[0].graph, 0), "original graph");
            for (unsigned node = 0; node < Nodes; ++node)
            {
                originals[node] = packet(owner.output, node);
                void *arguments[]{&originals[node]};
                hipKernelNodeParams params{};
                params.func = hipGraphArgumentGeometryKernel();
                params.gridDim = dim3(1);
                params.blockDim = dim3(Threads);
                params.kernelParams = arguments;
                check(hipGraphAddKernelNode(&nodes[node], owner.variants[0].graph, node?&nodes[node-1]:nullptr, node?1:0, &params), "actual argument node");
            }
            check(hipGraphClone(&owner.variants[1].graph, owner.variants[0].graph), "native clone");
            check(hipGraphClone(&owner.variants[2].graph, owner.variants[0].graph), "update clone");
            hipGraphNode_t updated_node = nullptr;
            check(hipGraphNodeFindInClone(&updated_node, nodes[48], owner.variants[2].graph), "original cloned node");
            auto updated = packet(owner.output, 48, true);
            void *arguments[]{&updated};
            hipKernelNodeParams params{};
            params.func = hipGraphArgumentGeometryKernel();
            params.gridDim = dim3(1);
            params.blockDim = dim3(Threads);
            params.kernelParams = arguments;
            check(hipGraphKernelNodeSetParams(updated_node, &params), "native graph argument update");
            check(hipGraphCreate(&owner.variants[3].graph, 0), "child parent");
            hipGraphNode_t child = nullptr;
            check(hipGraphAddChildGraphNode(&child, owner.variants[3].graph, nullptr, 0, owner.variants[0].graph), "original child clone");
            // Native constructors/clone/update must own copies, not caller stack arrays.
            for (auto &p:originals)p = Packet{};
            updated = Packet{};
            std::size_t observed = 0;
            for (unsigned variant = 0; variant < Variants; ++variant)
            {
                std::size_t before = 0, total_before = 0, after = 0, total_after = 0;
                check(hipMemGetInfo(&before, &total_before), "cold before");
                check(hipGraphInstantiate(&owner.variants[variant].exec, owner.variants[variant].graph, nullptr, nullptr, 0), "instantiate actual family");
                check(hipMemGetInfo(&after, &total_after), "cold after");
                if (total_before != total_after) throw std::runtime_error("Native total-memory identity changed during instantiation");
                const auto delta = before>after?before-after:0;
                observed+= delta;
                std::printf("HIP_KERNARG_GEOMETRY variant=%u,nodes=%u,declared_packet_bytes=%zu,observed_growth=%zu,family_growth=%zu,family_budget=%zu,total_memory_before=%zu,total_memory_after=%zu\n", variant, Nodes, sizeof(Packet), delta, observed, budget, total_before, total_after);
            }
            if (!GPUGraphMemoryContract::acceptsFamilyObservation(device, observed, budget))throw std::runtime_error("Argument-heavy native family exceeds original typed PMA budget");
            std::vector<std::uint32_t> actual(Nodes*Threads+16);
            for (unsigned variant = 0; variant < Variants; ++variant)for (unsigned replay = 0; replay < Replays; ++replay)
            {
                owner.pending = true;
                check(hipMemsetAsync(owner.output, 0xa5, data_bytes, owner.stream), "poison original output");
                check(hipGraphLaunch(owner.variants[variant].exec, owner.stream), "unchanged graph replay");
                check(hipMemcpyAsync(actual.data(), owner.output, data_bytes, hipMemcpyDeviceToHost, owner.stream), "terminal output");
                check(hipEventRecord(owner.terminal, owner.stream), "exact terminal");
                owner.await();
                for (unsigned node = 0; node < Nodes; ++node)
                {
                    const auto p = packet(owner.output, node, variant == 2&&node == 48);
                    for (unsigned lane = 0; lane < Threads; ++lane)
                    {
                        std::uint32_t expected = p.salt;
                        for (unsigned word = lane; word<1020; word+= Threads)expected+= p.words[word];
                        if (actual[node*Threads+lane] != expected)throw std::runtime_error("Original/clone/update/child argument bytes differ");
                    }
                }
                for (unsigned i = Nodes*Threads; i < actual.size(); ++i)
                if (actual[i] != 0xa5a5a5a5u)throw std::runtime_error("Argument geometry output suffix changed");
            }
            int final_device = -1;
            check(hipGetDevice(&final_device), "final original device");
            if (final_device != ordinal)throw std::runtime_error("Native graph changed original worker device");
        }
        // All graph/data owners retire before the authority is inspected.
        if (authority->claimedBytes(device, PhysicalMemoryOwner::ActivationTransportStaging, PhysicalMemoryMaterializationKind::NewAllocation) != 0 ||
        authority->reservedBytes(device, PhysicalMemoryOwner::NativeGraphExecutable) != 0)
        throw std::runtime_error("HIP argument-geometry PMA claims did not retire to zero");
    }
}
// namespace
/** @test Heavy actual arguments retain exact original geometry through all native copies. */
TEST(HIPGraphArgumentGeometry, OriginalCloneUpdateChildFamily)
{
    llaminar2::ensureAMDFactoryRegistered();
    if (!llaminar2::GPUDeviceContextPool::instance().hasAMDSupport())GTEST_SKIP()<<"ROCm unavailable";
    auto &worker = llaminar2::GPUDeviceContextPool::instance().getAMDContext(0);
    ASSERT_NO_THROW(worker.submitAndWait([]{
        llaminar2::hip_argument_geometry_test::prove(0);}));
}
