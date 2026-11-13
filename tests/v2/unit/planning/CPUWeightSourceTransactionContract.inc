/**
 * @file CPUWeightSourceTransactionContract.inc
 * @brief Actual original-PMA source serialization and lifetime without worker/native readiness.
 */
#include <future>
#include <optional>
#include <chrono>
#include <type_traits>

namespace {
/** @return One actual final admitted CPU source allocator, with no fabricated preparation phase. */
std::shared_ptr<PhysicalMemoryAuthority> sourceTransactionMemory()
{
    PhysicalMemoryPlanBuilder budget;
    budget.add({0,DeviceId::cpu(),4096,4096},PhysicalMemoryOwner::WeightLoadStaging,4096);
    return std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()),0);
}
}

/** @test One original source owner serializes callers while leaving accounting progress available. */
TEST(CPUWeightSourceTransactionContract, SharedAuthoritySerializesAndIndependentAuthoritiesRemainIndependent)
{
    using namespace std::chrono_literals;
    static_assert(!std::is_default_constructible_v<PhysicalMemoryCPUWeightSourceLease>);
    static_assert(!std::is_copy_constructible_v<PhysicalMemoryCPUWeightSourceLease>);
    static_assert(std::is_nothrow_move_constructible_v<PhysicalMemoryCPUWeightSourceLease>);
    const auto shared=sourceTransactionMemory(),independent=sourceTransactionMemory();
    std::optional<PhysicalMemoryCPUWeightSourceLease> first;
    first.emplace(shared->acquireCPUWeightSourceTransaction());
    std::promise<void> attempted;auto attempting=attempted.get_future();
    std::future<size_t> same;std::future<bool> other;
    try {
        same=std::async(std::launch::async,[&] {
            attempted.set_value();
            auto lease=shared->acquireCPUWeightSourceTransaction();
            return shared->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,
                PhysicalMemoryMaterializationKind::NewAllocation);
        });
        attempting.get();
        const bool blocked=same.wait_for(20ms)==std::future_status::timeout;
        other=std::async(std::launch::async,[&] {
            auto lease=independent->acquireCPUWeightSourceTransaction();return true;
        });
        const bool independent_progress=other.wait_for(100ms)==std::future_status::ready;
        // A waiter must not hold the accounting mutex needed to retire a source.
        {auto allocation=shared->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::WeightLoadStaging,64);}
        first.reset(); // Release before consuming/joining any results, including failures.
        EXPECT_TRUE(blocked);EXPECT_TRUE(independent_progress);
        EXPECT_TRUE(other.get());EXPECT_EQ(same.get(),0u);
    } catch(...) {
        const auto original_error=std::current_exception();first.reset();
        try{if(other.valid())(void)other.get();}catch(...){}
        try{if(same.valid())(void)same.get();}catch(...){}
        std::rethrow_exception(original_error);
    }
}

/** @test The move-only source token retains its original authority until the final lock retires. */
TEST(CPUWeightSourceTransactionContract, TokenRetainsOriginalAuthorityAndRejectsMissingSharedCpuAdmission)
{
    auto memory=sourceTransactionMemory();std::weak_ptr<PhysicalMemoryAuthority> original=memory;
    std::optional<PhysicalMemoryCPUWeightSourceLease> lease;
    lease.emplace(memory->acquireCPUWeightSourceTransaction());
    memory.reset();EXPECT_FALSE(original.expired());
    auto moved=std::move(*lease);lease.reset();EXPECT_FALSE(original.expired());
    {std::optional<PhysicalMemoryCPUWeightSourceLease> last;last.emplace(std::move(moved));last.reset();}
    EXPECT_TRUE(original.expired());
    PhysicalMemoryPlanBuilder no_cpu;
    no_cpu.add({0,DeviceId::cuda(0),4096,4096},PhysicalMemoryOwner::PrimaryModelWeights,64);
    const auto gpu_only=std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(no_cpu.build()),0);
    EXPECT_THROW((void)gpu_only->acquireCPUWeightSourceTransaction(),std::logic_error);
    const auto certificate=std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(no_cpu.build());
    PhysicalMemoryAuthority unshared(certificate,0);
    EXPECT_THROW((void)unshared.acquireCPUWeightSourceTransaction(),std::logic_error);
}

/** @test The original admitted CPU allocator is explicit, independent of NUMA/affinity or CPU0. */
TEST(CPUWeightSourceTransactionContract, NonzeroCpuAllocatorRequiresItsExactOriginalAdmission)
{
    const DeviceId cpu1{DeviceType::CPU,1};
    PhysicalMemoryPlanBuilder budget;
    budget.add({0,cpu1,4096,4096},PhysicalMemoryOwner::WeightLoadStaging,4096);
    const auto memory=std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()),0);
    EXPECT_NO_THROW((void)memory->acquireCPUWeightSourceTransaction(cpu1));
    EXPECT_THROW((void)memory->acquireCPUWeightSourceTransaction(DeviceId::cpu()),std::logic_error);
    EXPECT_THROW((void)memory->acquireCPUWeightSourceTransaction(DeviceId{DeviceType::CPU,2}),std::logic_error);
    EXPECT_THROW((void)memory->acquireCPUWeightSourceTransaction(DeviceId::cuda(0)),std::logic_error);
}
