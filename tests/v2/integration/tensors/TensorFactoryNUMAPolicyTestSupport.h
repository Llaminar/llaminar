/**
 * @file TensorFactoryNUMAPolicyTestSupport.h
 * @brief Bounded Linux policy observations for model-free factory integration.
 *
 * Observations read only kernel NUMA metadata. The fixture restores the caller's
 * incoming thread policy after all small tensors and new workers retire; it does
 * not claim production bound factories restore policy. No GPU, model, physical
 * page migration, timing threshold or host capacity override is involved.
 */
#pragma once

#include "v2/tensors/TensorFactory.h"
#include <gtest/gtest.h>
#include <array>
#include <cerrno>
#include <climits>
#include <cstring>
#include <exception>
#include <numa.h>
#include <numaif.h>
#include <stdexcept>
#include <string>
#include <thread>

namespace llaminar2::test::tensor_factory_numa
{
/** @brief Fixed bounded kernel nodemask, sufficient for up to 1024 node IDs. */
inline constexpr unsigned long PolicyBits = 1024;
/** @brief Exact thread memory policy, not a guessed CPU-affinity placement. */
struct Policy
{
    int mode = MPOL_DEFAULT;
    std::array<unsigned long, PolicyBits / (sizeof(unsigned long) * CHAR_BIT)> nodes{};
    bool operator==(const Policy &) const = default;
};
/** @return Complete current thread policy; denied observations fail precisely. */
inline Policy observe()
{
    Policy result;
    if (::get_mempolicy(&result.mode, result.nodes.data(), PolicyBits, nullptr, 0))
        throw std::runtime_error(std::string("get_mempolicy: ") + std::strerror(errno));
    return result;
}
/**
 * @brief Install a fixture policy; never continue after kernel denial.
 * @param policy Exact mode and nodemask to install on this calling thread.
 */
inline void install(const Policy &policy)
{
    const bool is_default = policy.mode == MPOL_DEFAULT;
    if (::set_mempolicy(policy.mode, is_default ? nullptr : policy.nodes.data(),
                        is_default ? 0 : PolicyBits))
        throw std::runtime_error(std::string("set_mempolicy: ") + std::strerror(errno));
}
/** @return Last actually allowed node, exercising a nonzero node where possible. */
inline int allowedNode()
{
    if (::numa_available() < 0 || ::numa_max_node() >= static_cast<int>(PolicyBits))
        throw std::runtime_error("NUMA policy integration needs an observable node ID below 1024");
    Policy allowed;
    if (::get_mempolicy(nullptr, allowed.nodes.data(), PolicyBits, nullptr, MPOL_F_MEMS_ALLOWED))
        throw std::runtime_error(std::string("get_mempolicy allowed nodes: ") + std::strerror(errno));
    constexpr auto WordBits = sizeof(unsigned long) * CHAR_BIT;
    for (int node = ::numa_max_node(); node >= 0; --node)
        if ((allowed.nodes[node / WordBits] >> (node % WordBits)) & 1ul)
            return node;
    throw std::runtime_error("No allowed NUMA node for integration control");
}
/**
 * @param node Actually allowed NUMA node selected from kernel metadata.
 * @return Explicit singleton bind, independent of MPI rank/current CPU.
 */
inline Policy bindPolicy(int node)
{
    if (node < 0 || node >= static_cast<int>(PolicyBits))
        throw std::invalid_argument("Fixture node outside bounded policy mask");
    Policy policy;
    policy.mode = MPOL_BIND;
    constexpr auto WordBits = sizeof(unsigned long) * CHAR_BIT;
    policy.nodes[node / WordBits] = 1ul << (node % WordBits);
    return policy;
}
/** @brief Restore the exact incoming policy even after an assertion/exception. */
class RestorePolicy final
{
    Policy incoming_ = observe();
public:
    /** @brief Retain current policy before the test installs a contrasting one. */
    RestorePolicy() = default;
    RestorePolicy(const RestorePolicy &) = delete;
    RestorePolicy &operator=(const RestorePolicy &) = delete;
    /** @brief Kernel restoration failure is a test failure, not ignored cleanup. */
    ~RestorePolicy()
    {
        try { install(incoming_); }
        catch (const std::exception &error) { ADD_FAILURE() << error.what(); }
    }
};
/**
 * @brief Observe two newly created worker threads after the factory's work.
 * @param expected Exact caller policy that new workers must inherit.
 *
 * Fresh workers avoid attributing a pre-existing OpenMP worker pool's policy to
 * this factory. Linux inherits the creating thread policy at clone, the same
 * mechanism that previously propagated a factory-created bind to packing teams.
 */
inline void expectFreshWorkerPolicy(const Policy &expected)
{
    std::array<Policy, 2> actual;
    std::array<std::exception_ptr, 2> errors{};
    std::array<std::jthread, 2> workers;
    for (size_t i = 0; i < workers.size(); ++i)
        workers[i] = std::jthread([&, i] {
            try { actual[i] = observe(); }
            catch (...) { errors[i] = std::current_exception(); }
        });
    for (auto &worker : workers) worker.join();
    for (size_t i = 0; i < actual.size(); ++i)
    {
        if (errors[i]) std::rethrow_exception(errors[i]);
        EXPECT_EQ(actual[i], expected) << "New worker " << i << " inherited a changed policy";
    }
}
} // namespace llaminar2::test::tensor_factory_numa
