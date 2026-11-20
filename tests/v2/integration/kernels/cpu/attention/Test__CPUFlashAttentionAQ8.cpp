/**
 * @file Test__CPUFlashAttentionAQ8.cpp
 * @brief Actual native AQ8 FA2 tensor/request entry and grouped serial-byte regressions.
 *
 * The production ring factory owns key basis, encoding, slots and compressed
 * values. The fixture registers its physical NUMA endpoint and allocates a
 * setup-owned DeviceWorkspaceManager, not a production PMA admission or a
 * context-summary lifetime certificate. These model-free integration cases cannot
 * pass merely because an ARM codec leaf compiled while the FA2 adapter remained
 * inside an x86-only guard. Serial calls are test oracles, never production replay.
 */
#include <gtest/gtest.h>
#include "backends/BackendManager.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "kernels/cpu/CPURingKVCache.h"
#include "kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "../../../../utils/TestTensorFactory.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <limits>
#include <span>
#include <vector>
#include <numa.h>
#include <sched.h>
#include <omp.h>

namespace
{
using namespace llaminar2;

/** @brief Lexically restore the caller's team preference after native entry tests. */
class AQ8TestWorkers final
{
public:
    /** @brief Select a positive test team; @param count Explicit correctness budget. */
    explicit AQ8TestWorkers(int count):previous_(omp_get_max_threads()){omp_set_num_threads(count);}
    /** @brief Restore the original preference on success or an assertion return. */
    ~AQ8TestWorkers(){omp_set_num_threads(previous_);}
    AQ8TestWorkers(const AQ8TestWorkers&)=delete;
    AQ8TestWorkers&operator=(const AQ8TestWorkers&)=delete;
private:
    int previous_;
};

/** @brief Invoke the actual tensor entry with explicit original ring geometry.
 * @param kernel Workspace-bound production kernel; @param query Original FP32 queries.
 * @param cache Original ring owner; @param output Complete caller-owned output.
 * @param rows Original query count; @param visible Original visible key prefix.
 * @param dimension Complete head dimension; @return Public entry's success status.
 */
bool aq8TensorEntry(CPUFlashAttentionKernelT<ActivationPrecision::FP32>&kernel,
                    const FP32Tensor&query,ICPUKVCache&cache,FP32Tensor&output,
                    int rows,int visible,int dimension)
{
    return kernel.compute_tensor(&query,cache.get_k(0),cache.get_v(0),&output,
        1,rows,visible,4,2,dimension,true,7,nullptr,nullptr,nullptr,-1,0,-1,-1,0,{},
        {.logical_row_origin=0,.physical_row_capacity=17});
}

/** @brief Poison the complete publication bank so no-op/partial writers cannot inherit a pass.
 * @param output Original caller-owned tensor, outside production execution.
 */
void poisonAQ8Output(FP32Tensor&output)
{
    std::fill_n(output.mutable_data(),output.numel(),std::numeric_limits<float>::quiet_NaN());
}

/** @brief Check every result is finite before accepting exact complete-bank byte equality.
 * @param output Actual public-entry tensor; @param expected Complete original serial bank.
 */
void expectAQ8Words(const FP32Tensor&output,std::span<const float>expected)
{
    ASSERT_EQ(output.numel(),expected.size());
    for(std::size_t i=0;i<expected.size();++i)
    {
        ASSERT_TRUE(std::isfinite(output.data()[i]))<<i;
        ASSERT_TRUE(std::isfinite(expected[i]))<<i;
    }
    ASSERT_EQ(std::memcmp(output.data(),expected.data(),expected.size_bytes()),0);
}
} // namespace

/** @test ARM and x86 execute both actual AQ8 entry routes and preserve original serial-row words. */
TEST(CPUFlashAttentionAQ8, NativeTensorAndRequestEntriesRetainSerialBytes)
{
    const int cpu=sched_getcpu();ASSERT_GE(cpu,0);
    const int node=numa_node_of_cpu(cpu);ASSERT_GE(node,0);
    initCPUBackend(node);
    constexpr int positions=17,rows=3,requests=2,heads=4,kv_heads=2;
    MPIContext mpi(0,1,MPI_COMM_WORLD);
    for(const int dimension:{64,128,256})
    {
        TurboQuantContext context(dimension,42);
        for(const auto precision:{ActivationPrecision::Q8_1,ActivationPrecision::TQ4,ActivationPrecision::TQ8})
        {
            SCOPED_TRACE(::testing::Message()<<"D="<<dimension<<" V="<<static_cast<int>(precision));
            const std::size_t columns=heads*dimension,kv_columns=kv_heads*dimension;
            std::array<std::unique_ptr<ICPUKVCache>,requests> caches;
            for(int request=0;request<requests;++request)
            {
                caches[request]=createCPURingKVCache(precision,mpi,1,1,positions,kv_heads,dimension,
                    DeviceId::cpu(),KVCacheLayoutMode::POSITION_MAJOR,&context);
                ASSERT_NE(caches[request],nullptr);
                auto keys=test::TestTensorFactory::createFP32Random({positions,kv_columns},-2,2,71+request);
                auto values=test::TestTensorFactory::createFP32Random({positions,kv_columns},-1,1,79+request);
                ASSERT_TRUE(caches[request]->append(0,0,keys.get(),values.get(),positions));
                ASSERT_EQ(caches[request]->get_k(0)->native_type(),TensorType::AQ8);
            }
            auto queries=test::TestTensorFactory::createFP32Random({requests*rows,columns},-1,1,83);
            auto one_query=test::TestTensorFactory::createFP32({1,columns});
            auto one_output=test::TestTensorFactory::createFP32({1,columns});
            CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel;
            const auto requirements=kernel.getWorkspaceRequirements(requests*rows,heads,dimension);
            DeviceWorkspaceManager workspace(DeviceId::cpu(),requirements.total_bytes_with_alignment());
            ASSERT_TRUE(workspace.allocate(requirements));kernel.bindWorkspace(&workspace);
            std::vector<float> expected(requests*rows*columns);
            {
                AQ8TestWorkers workers(1);
                for(int request=0;request<requests;++request)
                    for(int row=0;row<rows;++row)
                    {
                        const std::size_t first=(request*rows+row)*columns;
                        std::memcpy(one_query->mutable_data(),queries->data()+first,columns*sizeof(float));
                        poisonAQ8Output(*one_output);
                        ASSERT_TRUE(aq8TensorEntry(kernel,*one_query,*caches[request],*one_output,1,
                                                  positions-rows+row+1,dimension));
                        std::copy_n(one_output->data(),columns,expected.data()+first);
                    }
            }
            auto local_query=test::TestTensorFactory::createFP32({rows,columns});
            auto local_output=test::TestTensorFactory::createFP32({rows,columns});
            auto request_output=test::TestTensorFactory::createFP32({requests*rows,columns});
            std::array<const ITensor*,requests> keys{caches[0]->get_k(0),caches[1]->get_k(0)};
            std::array<const ITensor*,requests> values{caches[0]->get_v(0),caches[1]->get_v(0)};
            std::array<int,requests> lengths{positions,positions};
            std::array<attention::AttentionKVLogicalView,requests> views{};
            for(auto &view:views)view={.logical_row_origin=0,.physical_row_capacity=positions};
            for(const int count:{1,3,10})
            {
                AQ8TestWorkers workers(count);
                for(int request=0;request<requests;++request)
                {
                    std::memcpy(local_query->mutable_data(),queries->data()+request*rows*columns,rows*columns*sizeof(float));
                    poisonAQ8Output(*local_output);
                    ASSERT_TRUE(aq8TensorEntry(kernel,*local_query,*caches[request],*local_output,rows,positions,dimension));
                    expectAQ8Words(*local_output,{expected.data()+request*rows*columns,rows*columns});
                    poisonAQ8Output(*local_output);
                    ASSERT_TRUE(kernel.compute_verifier_rows_decode_equivalent(local_query.get(),keys[request],values[request],
                        local_output.get(),rows,positions,heads,kv_heads,dimension,true,7,nullptr,-1,0,0,views[request]));
                    expectAQ8Words(*local_output,{expected.data()+request*rows*columns,rows*columns});
                }
                poisonAQ8Output(*request_output);
                ASSERT_TRUE(kernel.compute_request_batch_decode_equivalent(queries.get(),keys.data(),values.data(),lengths.data(),
                    request_output.get(),requests,rows,heads,kv_heads,dimension,true,7,nullptr,-1,0,0,{},views.data()));
                expectAQ8Words(*request_output,expected);
            }
        }
    }
}
