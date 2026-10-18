/**
 * @file rccl_protocol_geometry_probe.cpp
 * @brief Exercise the actual RCCL protocol consumers without initializing a GPU.
 *
 * The build supplies its native dependency translation units and compile
 * definitions. Immutable synthetic topology descriptors cover independent
 * communicator lifetimes and concurrent first use. No replacement geometry
 * implementation, device enumeration, communicator initialization or kernel is
 * used. Unrelated native functions are discarded at link time.
 */
#include LLAMINAR_RCCL_WRAP_SOURCE
#include LLAMINAR_RCCL_ARCH_SOURCE

#include <array>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace
{
    /** @brief Independent input architectures, including the supported suffix syntax. */
    constexpr std::array<const char*, 6> architectures{
        "gfx906", "gfx950", "gfx90a", "gfx942", "gfx1100", "gfx950:xnack-"};

    /** @brief Own valid C++ topology/communicator objects; no native initialization. */
    struct GeometryOwner
    {
        std::unique_ptr<ncclComm> comm = std::make_unique<ncclComm>();
        std::unique_ptr<ncclTopoSystem> topology = std::make_unique<ncclTopoSystem>();

        /** @brief Bind caller-owned immutable topology to the production descriptor. */
        GeometryOwner()
        {
            comm->topo = topology.get();
            comm->WarpSize = 64;
            topology->nodes[GPU].count = 1;
        }

        /**
         * @brief Reconfigure an unused synthetic descriptor between independent calls.
         * @param architecture Exact GPU architecture under test.
         * @param nodes Number of hosts represented by the descriptor.
         * @param ranks Number of collective participants, always positive.
         */
        void configure(const char* architecture, int nodes, int ranks)
        {
            comm->nRanks = ranks;
            comm->nNodes = nodes;
            std::snprintf(topology->nodes[GPU].nodes[0].gpu.gcn,
                          sizeof(topology->nodes[GPU].nodes[0].gpu.gcn), "%s", architecture);
        }
    };

    /** @brief Count every checked consumer result without worker-thread logging. */
    struct Observations
    {
        std::uint64_t checks = 0;
        std::uint64_t failures = 0;

        /**
         * @brief Retain a failed invariant while allowing the matrix to complete.
         * @param passed Result of one independent consumer assertion.
         */
        void check(bool passed)
        {
            ++checks;
            failures += !passed;
        }
    };

    /**
     * @brief Prove buffers and launch limits for one complete architecture matrix.
     * @param owner Descriptor owned exclusively by this calling thread.
     * @param offset Rotate the first architecture to cover initialization order.
     * @return Counts of executed assertions and failed invariants.
     */
    Observations verify(GeometryOwner& owner, std::size_t offset)
    {
        Observations observed;
        for (std::size_t index = 0; index < architectures.size(); ++index)
        {
            const auto ordinal = (index + offset) % architectures.size();
            const auto architecture = architectures[ordinal];
            // These two reviewed architecture classes have different launch
            // limits. Do not ask the function under test for its own oracle.
            const int maximum = ordinal == 1 || ordinal == 5 ? 512 : 256;
            for (const int nodes : {1, 2})
                for (const int ranks : {2, 4, 8})
                {
                    owner.configure(architecture, nodes, ranks);
                    int buffers[NCCL_NUM_PROTOCOLS] = {-1, -1, -1};
                    rcclSetDefaultBuffSizes(owner.comm.get(), buffers);
                    observed.check(buffers[NCCL_PROTO_LL] == 262144);
                    observed.check(buffers[NCCL_PROTO_LL128] == (maximum == 512 ? 917504 : 458752));
                    observed.check(buffers[NCCL_PROTO_SIMPLE] == 4194304);

                    for (const int algorithm : {NCCL_ALGO_TREE, NCCL_ALGO_PAT, NCCL_ALGO_RING})
                        for (const int protocol : {NCCL_PROTO_LL, NCCL_PROTO_LL128, NCCL_PROTO_SIMPLE})
                        {
                            ncclTaskColl task{};
                            task.func = ncclFuncAllReduce;
                            task.algorithm = algorithm;
                            task.protocol = protocol;
                            int threads = 384;
                            rcclOptThreadBlockSize(owner.comm.get(), &task, 4194304, threads);
                            const int expected = nodes == 1 || protocol == NCCL_PROTO_LL ? 256 :
                                algorithm == NCCL_ALGO_RING ? 384 : maximum;
                            observed.check(threads == expected);

                            task.func = ncclFuncReduceScatter;
                            // The inclusive small-message boundary overrides
                            // the multi-node TREE/PAT limit for every protocol.
                            threads = 384;
                            rcclOptThreadBlockSize(owner.comm.get(), &task, std::size_t{524288} * ranks, threads);
                            observed.check(threads == 256);
                            threads = 384;
                            rcclOptThreadBlockSize(owner.comm.get(), &task, std::size_t{524288} * ranks + 1, threads);
                            observed.check(threads == expected);
                        }
                }
        }
        return observed;
    }

    /** @brief A one-use CPU start gate gives every worker a concurrent first call. */
    class StartGate
    {
    public:
        /**
         * @brief Require all workers to arrive before any consumer executes.
         * @param count Positive number of CPU worker participants.
         */
        explicit StartGate(unsigned count) : remaining_(count) {}

        /** @brief Publish the last arrival to all waiting CPU workers. */
        void arrive()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (--remaining_ == 0)
                ready_.notify_all();
            else
                ready_.wait(lock, [this] { return remaining_ == 0; });
        }

    private:
        unsigned remaining_;
        std::mutex mutex_;
        std::condition_variable ready_;
    };
}

/**
 * @brief Exercise forward/reverse architecture order or simultaneous cold consumers.
 * @param argc Exactly one mode argument is required.
 * @param argv Mode must be forward, reverse or concurrent.
 * @return Zero only when every real consumer result matches the reviewed defaults.
 */
int main(int argc, char** argv)
{
    if (argc != 2)
        return 2;
    const std::string mode(argv[1]);
    Observations total;
    if (mode == "forward" || mode == "reverse")
    {
        GeometryOwner owner;
        total = verify(owner, mode == "reverse" ? 1 : 0);
    }
    else if (mode == "concurrent")
    {
        constexpr unsigned participants = 16;
        StartGate start(participants);
        std::array<Observations, participants> results{};
        std::vector<std::thread> workers;
        for (unsigned rank = 0; rank < participants; ++rank)
            workers.emplace_back([&, rank] {
                GeometryOwner owner;
                start.arrive();
                for (unsigned repetition = 0; repetition < 100; ++repetition)
                {
                    const auto observed = verify(owner, rank + repetition);
                    results[rank].checks += observed.checks;
                    results[rank].failures += observed.failures;
                }
            });
        for (auto& worker : workers)
            worker.join();
        for (const auto& observed : results)
        {
            total.checks += observed.checks;
            total.failures += observed.failures;
        }
    }
    else
        return 2;
    std::printf("{\"mode\":\"%s\",\"checks\":%llu,\"failures\":%llu}\n", mode.c_str(),
                static_cast<unsigned long long>(total.checks),
                static_cast<unsigned long long>(total.failures));
    return total.failures == 0 ? 0 : 1;
}
