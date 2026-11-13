/**
 * @file Test__GPUContextProvenance.cpp
 * @brief Replay original pool provenance and cross-backend alias rejection without GPUs.
 *
 * Recorded worker constructors use the production pool. No PMA, source family,
 * native enumeration or caller-authored provenance capability is involved.
 */
#include "backends/GPUDeviceContextPool.h"
#include "mocks/MockWorkerGPUContext.h"
#include <gtest/gtest.h>
#include <array>
namespace llaminar2::test {

/** @test Invalid identities never enter a pool lookup or install retirement state. */
TEST(GPUContextProvenance, InvalidRetirementEndpointsFailBeforeLookupReplay)
{
    auto &pool = GPUDeviceContextPool::instance();
    const std::array devices{DeviceId::cpu(), DeviceId::invalid(),
                             DeviceId::cuda(-1), DeviceId::rocm(-1)};
    for (int replay = 0; replay < 3; ++replay)
        for (const auto device : devices)
        {
            EXPECT_THROW((void)pool.retireExclusiveGeneration(device), std::invalid_argument);
            EXPECT_THROW((void)pool.beginExclusiveGenerationRetirement(device), std::invalid_argument);
        }
}

TEST(GPUContextProvenance, ExactBackendControlBlockAndRetainedGenerationReplay){
    auto& pool=GPUDeviceContextPool::instance();auto births=std::make_shared<std::atomic<unsigned>>(0);
    const auto creator=[births](int ordinal){++*births;return std::unique_ptr<IWorkerGPUContext>(new testing::MockWorkerGPUContext(ordinal));};
    pool.registerNvidiaFactory(creator,1);pool.registerAMDFactory(creator,1);
    const auto cuda=DeviceId::cuda(0),hip=DeviceId::rocm(0);
    EXPECT_THROW(pool.retainExistingContextProvenance(cuda),std::logic_error);
        EXPECT_EQ(births->load(),0u);
    std::uint64_t generation=0;
    {
        auto cudaWorker=pool.retainNvidiaContext(0),hipWorker=pool.retainAMDContext(0);
        auto cudaProof=pool.retainExistingContextProvenance(cuda),hipProof=pool.retainExistingContextProvenance(hip);
        EXPECT_EQ(cudaProof.device(),cuda);
        EXPECT_EQ(hipProof.device(),hip);generation=cudaProof.generation();
        EXPECT_NE(generation,0u);
        EXPECT_NO_THROW(cudaProof.requireWorker(cudaWorker));
        EXPECT_NO_THROW(hipProof.requireWorker(hipWorker));
        EXPECT_THROW(cudaProof.requireWorker(hipWorker),std::logic_error);
        EXPECT_THROW(hipProof.requireWorker(cudaWorker),std::logic_error);
        auto alias=std::shared_ptr<IWorkerGPUContext>(cudaWorker.get(),[](auto*){});
        EXPECT_THROW(cudaProof.requireWorker(alias),std::logic_error);
        auto moved=std::move(cudaProof);
        EXPECT_THROW(cudaProof.requireWorker(nullptr),std::logic_error);
        EXPECT_THROW(moved.requireWorker(nullptr),std::logic_error);
        EXPECT_NO_THROW(moved.requireWorker(cudaWorker));
        EXPECT_THROW((void)pool.beginExclusiveGenerationRetirement(cuda),std::logic_error);
        EXPECT_THROW(pool.retainExistingContextProvenance(DeviceId::cuda(1)),std::logic_error);
        EXPECT_EQ(births->load(),2u);
    }
    {
        auto first=pool.beginExclusiveGenerationRetirement(cuda),second=pool.beginExclusiveGenerationRetirement(hip);
        EXPECT_THROW(pool.retainExistingContextProvenance(cuda),std::logic_error);
        EXPECT_THROW((void)pool.retainNvidiaContext(0),std::logic_error);
        EXPECT_EQ(births->load(),2u);
    }
    {
        auto worker=pool.retainNvidiaContext(0);auto proof=pool.retainExistingContextProvenance(cuda);
        EXPECT_GT(proof.generation(),generation);
        EXPECT_EQ(births->load(),3u);
        EXPECT_NO_THROW(proof.requireWorker(worker));
    }
    {auto scope=pool.beginExclusiveGenerationRetirement(cuda);}
    EXPECT_THROW(pool.retainExistingContextProvenance(cuda),std::logic_error);
}
}
