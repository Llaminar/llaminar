/**
 * @file ReplicatedGQADirectGuardProof.h
 * @brief Actual native FP32 entrypoint proof for original global replicated GQA.
 *
 * This diagnostic uses the production typed attention kernel, its named
 * workspace requirements and exact worker stream. Constant dyadic values make
 * the independent head-selection oracle exact without changing tolerances.
 * Captured stage arithmetic is tested separately by each backend fixture.
 */
#pragma once
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <exception>
#include <cstring>
#include <vector>

namespace llaminar2::test
{
/**
 * @brief Check direct FP32 native admission and original global KV selection.
 * @tparam Kernel Actual CUDA or ROCm FP32 attention kernel specialization.
 * @param device Exact native participant; preparation and retirement are local.
 */
template<class Kernel> void proveReplicatedGQADirectGuard(DeviceId device)
{
    auto *backend=getBackendFor(device);
    if(!backend || backend->deviceCount()<=device.ordinal)GTEST_SKIP()<<"GPU unavailable";
    auto context=IDeviceContext::create(device,1);
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&] {
        const auto stream=ExplicitGPUStream(worker.defaultStream()).get();
        for(const int rows:{1,3}) {
            constexpr int heads=3,kvHeads=2,dim=256,kvRows=2;
            FP32Tensor q({std::size_t(rows),heads*dim}),k({kvRows,kvHeads*dim}),
                v({kvRows,kvHeads*dim}),output({std::size_t(rows),heads*dim});
            std::fill_n(q.mutable_data(),rows*heads*dim,0.0f);
            std::fill_n(k.mutable_data(),kvRows*kvHeads*dim,0.0f);
            for(int row=0;row<kvRows;++row)for(int head=0;head<kvHeads;++head)for(int d=0;d<dim;++d)
                v.mutable_data()[(row*kvHeads+head)*dim+d]=float(2+head)/4.0f+float(d)/1024.0f;
            std::fill_n(output.mutable_data(),rows*heads*dim,-71.0f);
            Kernel kernel(device.ordinal);
            kernel.setGPUStream(stream);
            const auto requirements=kernel.getWorkspaceRequirements(rows,heads,dim);
            DeviceWorkspaceManager workspace(device,requirements.total_bytes_with_alignment());
            ASSERT_TRUE(workspace.allocate(requirements));
            kernel.bindWorkspace(&workspace);
            std::vector<float> observed(rows*heads*dim);
            // Declared last: even an assertion/exception retires queued kernels
            // and terminal copies before host planes, workspace or tensors die.
            struct Retirement {
                IBackend *backend;void *stream;int ordinal;
                /** @brief Fence this diagnostic stream before every borrowed owner. */
                ~Retirement() noexcept {if(!backend->synchronizeStream(stream,ordinal))std::terminate();}
            } retirement{backend,stream,device.ordinal};
            for(auto *tensor:{static_cast<TensorBase*>(&q),static_cast<TensorBase*>(&k),
                              static_cast<TensorBase*>(&v),static_cast<TensorBase*>(&output)}) {
                TransferEngine::prepareDeviceInput(tensor,device,stream);
                TransferEngine::requireDeviceInput(tensor,device,stream);
            }
            ASSERT_TRUE(kernel.prepareDynamicAttnParams(kvRows,0,rows,stream));
            const auto apply=[&](int start,int ratio,int visibleKV=2) {
                return kernel.apply_typed(static_cast<const float*>(q.gpu_data_ptr()),
                    static_cast<const float*>(k.gpu_data_ptr()),static_cast<const float*>(v.gpu_data_ptr()),
                    static_cast<float*>(output.gpu_data_ptr()),1,rows,kvRows,heads,visibleKV,dim,
                    false,-1,0,device.ordinal,nullptr,nullptr,start,ratio);
            };
            const auto collect=[&] {
                return backend->deviceToHostOnStream(observed.data(),output.gpu_data_ptr(),
                    observed.size()*sizeof(float),device.ordinal,stream) &&
                    backend->synchronizeStream(stream,device.ordinal);
            };
            // All eight real TP8 coordinates, plus a boundary-crossing query
            // interval, exercise both global KV groups without local modulo.
            for(const int start:{0,3,6,9,12,15,18,21,11}) {
                ASSERT_TRUE(apply(start,12));ASSERT_TRUE(collect());
                for(int row=0;row<rows;++row)for(int head=0;head<heads;++head)for(int d=0;d<dim;++d)
                    ASSERT_EQ(observed[(row*heads+head)*dim+d],
                        float(2+(start+head)/12)/4.0f+float(d)/1024.0f)
                        <<"rows="<<rows<<" start="<<start<<" head="<<head<<" dim="<<d;
            }
            const auto lastSuccess=observed;
            // Rejected geometry must return before enqueue and preserve every
            // previously produced output byte, including complete interval ends.
            for(const auto bad:std::array<std::array<int,3>,7>{{
                    {{-1,12,2}},{{22,12,2}},{{0,-1,2}},{{0,0,2}},
                    {{21,1,2}},{{0,12,0}},{{21,12,1}}}}) {
                EXPECT_FALSE(apply(bad[0],bad[1],bad[2]));ASSERT_TRUE(collect());
                EXPECT_EQ(std::memcmp(observed.data(),lastSuccess.data(),observed.size()*sizeof(float)),0);
            }
        }
    }).get();
}
} // namespace llaminar2::test
