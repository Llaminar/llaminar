/**
 * @file Test__RouterHadamardQ8.cpp
 * @brief Independent signed-H32 oracle, captured logits and original expert-bank proof.
 *
 * CPU butterflies generate rotated floating operands independently of GPU
 * shuffles. Existing Native Q8 bridges provide the dot oracle, including serial
 * row comparisons. The candidate graph must retain exactly three nodes with
 * zero private memory, and leave source weights/inputs and inactive tails intact.
 * Tensor-core prefill additionally authenticates the exact captured kernel family
 * and both router bases across the supported M range and wider K partitions.
 */
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/common/RouterQ8Launch.h"
#ifdef HAVE_CUDA
#include "kernels/cuda/moe/CUDARouterQ8Prefill.h"
#endif
#include "kernels/common/DeviceQ8ActivationNumericalContract.h"
#include "utils/TestTensorFactory.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace
{
using namespace llaminar2;
using namespace llaminar2::test;
/** @brief Explicit test path; both must retain identical Q8 arithmetic. */
enum class HRoute { DecodeEquivalent, TensorCorePrefill };
/** @brief Preserve a failed native edge across the device worker boundary. */
void requireH(bool value, const char *edge) { if (!value) throw std::runtime_error(edge); }
/** @brief Complete test observations before borrowed graph addresses retire. */
struct HJoin { IBackend *backend; DeviceId device; void *stream;
    ~HJoin() { (void)backend->synchronizeStream(stream, device.ordinal); } };
/** @brief Independent array butterfly with standard seed31 signs and explicit CPU RNE stores. */
std::vector<float> rotateH(const std::vector<float> &source)
{
    std::mt19937_64 rng(31); std::array<float,32> signs{};
    for (auto &sign : signs) sign = (rng() & 1u) ? 1.f : -1.f;
    auto result = source;
    for (size_t base = 0; base < result.size(); base += 32)
    {
        for (int i = 0; i < 32; ++i) result[base+i] *= signs[i];
        for (int stride = 1; stride < 32; stride *= 2)
            for (int start = 0; start < 32; start += 2*stride)
                for (int j = 0; j < stride; ++j)
                {
                    const float a = result[base+start+j], b = result[base+start+j+stride];
                    volatile float sum = a+b, difference = a-b;
                    result[base+start+j] = sum; result[base+start+j+stride] = difference;
                }
        for (int i = 0; i < 32; ++i) { volatile float value=result[base+i]*0x1.6a09e6p-3f; result[base+i]=value; }
    }
    return result;
}
/** @brief Prepare only the selected backend's retained gate bytes. */
bool gateH(DeviceId d, const float *w, int8_t *q, float *s, int k, int e, void *stream, RouterQ8Basis basis)
{
#ifdef HAVE_CUDA
    if(d.is_cuda()) return cudaMoE_quantize_router_gate_q8(w,q,s,k,e,d.ordinal,stream,basis);
#endif
#ifdef HAVE_ROCM
    if(d.is_rocm()) return hipMoE_quantize_router_gate_q8(w,q,s,k,e,d.ordinal,stream,basis);
#endif
    throw std::invalid_argument("Hadamard gate requires compiled backend");
}
/** @brief Dispatch serial or grouped production logits; output bank always uses original basis. */
bool logitsH(DeviceId d, const float *h, int8_t *q, float *s, const int8_t *w, const float *ws,
    float *out, int m, int k, int e, void *stream, const int *live, RouterQ8Basis basis, bool serial=false)
{
#ifdef HAVE_CUDA
    if(d.is_cuda()) return serial ? cudaMoE_gate_logits_single_token_q8_weights(h,q,s,w,ws,out,k,e,d.ordinal,stream,basis) :
        cudaMoE_gate_logits_q8_weights_decode_equivalent_rows(h,q,s,w,ws,out,m,k,e,d.ordinal,stream,basis);
#endif
#ifdef HAVE_ROCM
    if(d.is_rocm()) return serial ? hipMoE_gate_logits_single_token_q8_weights(h,q,s,w,ws,out,k,e,d.ordinal,stream,basis) :
        hipMoE_gate_logits_q8_weights_decode_equivalent_rows(h,q,s,w,ws,out,m,k,e,d.ordinal,stream,live,basis);
#endif
    throw std::invalid_argument("Hadamard logits require compiled backend");
}
/** @brief Check a captured shape, independent source preservation and large/small/empty replay. */
void proveH(DeviceId device, int m, int k, int experts,
    HRoute route = HRoute::DecodeEquivalent, RouterQ8Basis basis = RouterQ8Basis::SignedHadamard32)
{
    auto &context=GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&]
    {
        SCOPED_TRACE(::testing::Message()<<device.toString()<<" M="<<m<<" K="<<k);
        auto *backend=getBackendFor(device); void *stream=context.defaultStream();
        requireH(backend && stream,"H32 backend stream");
#ifdef HAVE_CUDA
        cuda::RouterQ8PrefillHardware hardware;
        if (route == HRoute::TensorCorePrefill)
        {
            EXPECT_FALSE(cuda::prepareRouterQ8PrefillHardware(device.ordinal, nullptr, hardware));
            requireH(cuda::prepareRouterQ8PrefillHardware(device.ordinal, stream, hardware), "router hardware preparation");
            ASSERT_EQ(hardware.device, device.ordinal);
        }
#endif
        const size_t extent=size_t(m+1)*k, gate_extent=size_t(experts)*k;
        std::vector<float> input(extent), weights(gate_extent);
        for(size_t i=0;i<input.size();++i) input[i]=i%32==3?8.f:float(int(i*71%199)-99)/129.f;
        for(size_t i=0;i<weights.size();++i) weights[i]=i%32==19?-2.f:float(int(i*53%199)-99)/1027.f;
        auto rotated_input=basis==RouterQ8Basis::SignedHadamard32?rotateH(input):input;
        auto rotated_weights=basis==RouterQ8Basis::SignedHadamard32?rotateH(weights):weights;
        auto h=TestTensorFactory::createFP32({extent}), hr=TestTensorFactory::createFP32({extent});
        auto w=TestTensorFactory::createFP32({gate_extent}), wr=TestTensorFactory::createFP32({gate_extent});
        std::copy(input.begin(),input.end(),h->mutable_data());std::copy(rotated_input.begin(),rotated_input.end(),hr->mutable_data());
        std::copy(weights.begin(),weights.end(),w->mutable_data());std::copy(rotated_weights.begin(),rotated_weights.end(),wr->mutable_data());
        auto q=TestTensorFactory::createINT32({extent/4}), qr=TestTensorFactory::createINT32({extent/4});
        auto g=TestTensorFactory::createINT32({gate_extent/4}), gr=TestTensorFactory::createINT32({gate_extent/4});
        auto s=TestTensorFactory::createFP32({extent/32}), sr=TestTensorFactory::createFP32({extent/32});
        auto gs=TestTensorFactory::createFP32({gate_extent/32}), gsr=TestTensorFactory::createFP32({gate_extent/32});
        auto out=TestTensorFactory::createFP32({size_t(m+1)*experts}), ref=TestTensorFactory::createFP32({size_t(m+1)*experts});
        auto count=TestTensorFactory::createINT32({1});count->mutable_int32_data()[0]=m;
        auto graph=context.createGraphCapture(stream); HJoin join{backend,device,stream};
        for(TensorBase *tensor:{h.get(),hr.get(),w.get(),wr.get(),s.get(),sr.get(),gs.get(),gsr.get(),out.get(),ref.get()})
            requireH(tensor->ensureOnDevice(device,stream),"H32 floating fixture storage");
        for(auto *tensor:{q.get(),qr.get(),g.get(),gr.get(),count.get()})
            requireH(tensor->ensureOnDevice(device,stream),"H32 byte fixture storage");
        auto fp=[](auto &t){return static_cast<float*>(t->gpu_data_ptr());};
        auto qp=[](auto &t){return static_cast<int8_t*>(t->gpu_data_ptr());};
        const auto *live=static_cast<const int*>(count->gpu_data_ptr());
        const auto rotated=basis; constexpr auto original=RouterQ8Basis::Original;
        EXPECT_FALSE(gateH(device,fp(w),qp(g),fp(gs),k,experts,nullptr,rotated));
        EXPECT_FALSE(logitsH(device,fp(h),qp(q),fp(s),qp(g),fp(gs),fp(out),m,k,experts,nullptr,live,rotated));
        EXPECT_FALSE(gateH(device,fp(w),qp(g),fp(gs),k,experts,stream,static_cast<RouterQ8Basis>(99)));
        requireH(gateH(device,fp(w),qp(g),fp(gs),k,experts,stream,rotated) &&
            gateH(device,fp(wr),qp(gr),fp(gsr),k,experts,stream,original),"H32 first weight conversion");
        std::vector<int8_t> actual_gate(gate_extent),expected_gate(gate_extent);
        std::vector<float> actual_scales(gate_extent/32),expected_scales(gate_extent/32);
        requireH(backend->deviceToHost(actual_gate.data(),qp(g),gate_extent,device.ordinal,stream) &&
            backend->deviceToHost(expected_gate.data(),qp(gr),gate_extent,device.ordinal,stream) &&
            backend->deviceToHost(actual_scales.data(),fp(gs),actual_scales.size()*4,device.ordinal,stream) &&
            backend->deviceToHost(expected_scales.data(),fp(gsr),expected_scales.size()*4,device.ordinal,stream) &&
            backend->synchronizeStream(stream,device.ordinal),"H32 independent gate observation");
        ASSERT_EQ(actual_gate,expected_gate); ASSERT_EQ(std::memcmp(actual_scales.data(),expected_scales.data(),actual_scales.size()*4),0);
        const auto capturedLogits = [&](void *producer)
        {
#ifdef HAVE_CUDA
            if (route == HRoute::TensorCorePrefill)
            {
                requireH(device.is_cuda(), "tensor-core test requires CUDA");
                return cudaMoE_gate_logits_q8_weights_prefill_rows(fp(h), qp(q), fp(s),
                    qp(g), fp(gs), fp(out), m, k, experts, device.ordinal, producer, live, rotated, hardware);
            }
#endif
            return logitsH(device,fp(h),qp(q),fp(s),qp(g),fp(gs),fp(out),m,k,experts,producer,live,rotated,m==1);
        };
        EXPECT_FALSE(capturedLogits(nullptr));
        {
            ScopedBackendGraphCapture capture(context,*graph,"signed router H32"); requireH(capture.begin(),"H32 capture begin");
#ifdef HAVE_CUDA
            if (route == HRoute::TensorCorePrefill)
            {
                cuda::RouterQ8PrefillHardware unavailable;
                EXPECT_FALSE(cuda::prepareRouterQ8PrefillHardware(device.ordinal, stream, unavailable));
                EXPECT_FALSE(unavailable.valid());
            }
#endif
            requireH(capturedLogits(stream),"H32 captured logits");
            capture.finish();
        }
        std::vector<GPUGraphKernelNodeInfo> nodes; ASSERT_TRUE(graph->inspectKernelNodes(nodes));
        ASSERT_EQ(nodes.size(), basis==RouterQ8Basis::SignedHadamard32 ? 3u : 2u);
        bool found=false, tensor_core=false;
        for(const auto &node:nodes)
        {
            ASSERT_EQ(node.local_memory_bytes_per_thread,0u);
            found |= node.name.find("quantize_signed_hadamard32")!=std::string::npos;
            tensor_core |= node.name.find("router_q8_prefill_detail")!=std::string::npos;
#ifdef HAVE_CUDA
            if (node.name.find("router_q8_prefill_detail")!=std::string::npos)
            {
                const auto tile = cuda::routerQ8PrefillGeometry(m,k,experts,hardware);
                EXPECT_EQ(node.block_x, 32u * tile.warps);
                EXPECT_EQ(node.grid_x, unsigned((experts+tile.experts-1)/tile.experts));
                EXPECT_EQ(node.grid_y, unsigned((m+tile.rows-1)/tile.rows));
                EXPECT_EQ(node.static_shared_memory_bytes, tile.k_warps>1 ? 512u*tile.k_warps : 0u);
            }
#endif
        }
        ASSERT_EQ(found,basis==RouterQ8Basis::SignedHadamard32);
        ASSERT_EQ(tensor_core,route==HRoute::TensorCorePrefill);
        requireH(graph->instantiate(),"H32 instantiate");
        std::vector<float> actual(out->numel()),expected(out->numel()),poison(out->numel(),-12345.f),scales(extent/32);
        std::vector<int8_t> bytes(extent),poison_bytes(extent,23);
        std::vector<float> poison_scales(extent/32,-17.f);
        for(int active: ((route==HRoute::TensorCorePrefill || (device.is_rocm() && m>1))?std::vector<int>{m,0,1,m/2,m}:std::vector<int>{m,m}))
        {
            requireH(backend->hostToDevice(count->gpu_data_ptr(),&active,sizeof(active),device.ordinal,stream) &&
                backend->hostToDevice(fp(out),poison.data(),poison.size()*4,device.ordinal,stream) &&
                backend->hostToDevice(fp(ref),poison.data(),poison.size()*4,device.ordinal,stream) &&
                backend->hostToDevice(qp(q),poison_bytes.data(),extent,device.ordinal,stream) &&
                backend->hostToDevice(fp(s),poison_scales.data(),poison_scales.size()*4,device.ordinal,stream),"H32 poisoned replay inputs");
            if(active>0)
            {
                if(m<=31) for(int row=0;row<active;++row)
                    requireH(logitsH(device,fp(hr)+size_t(row)*k,qp(qr)+size_t(row)*k,fp(sr)+size_t(row)*k/32,
                        qp(gr),fp(gsr),fp(ref)+size_t(row)*experts,1,k,experts,stream,nullptr,original,true),"H32 serial row oracle");
                else requireH(logitsH(device,fp(hr),qp(qr),fp(sr),qp(gr),fp(gsr),fp(ref),active,k,experts,stream,nullptr,original),"H32 grouped native oracle");
            }
            requireH(graph->launch(),"H32 retained replay");
            requireH(backend->deviceToHost(actual.data(),fp(out),actual.size()*4,device.ordinal,stream) &&
                backend->deviceToHost(expected.data(),fp(ref),expected.size()*4,device.ordinal,stream) &&
                backend->deviceToHost(bytes.data(),qp(q),extent,device.ordinal,stream) &&
                backend->deviceToHost(scales.data(),fp(s),scales.size()*4,device.ordinal,stream) &&
                backend->synchronizeStream(stream,device.ordinal),"H32 terminal observations");
            ASSERT_EQ(std::memcmp(actual.data(),expected.data(),actual.size()*4),0)<<"live="<<active;
            for(size_t block=0;block<scales.size();++block)
            {
                if(block>=size_t(active)*k/32) { ASSERT_EQ(scales[block],-17.f);for(int lane=0;lane<32;++lane) ASSERT_EQ(bytes[block*32+lane],23);continue; }
                float maximum=0.f;for(int lane=0;lane<32;++lane) maximum=std::max(maximum,std::abs(input[block*32+lane]));
                const float scale=device_q8_activation_contract::scale(maximum);
                ASSERT_EQ(std::bit_cast<uint32_t>(scales[block]),std::bit_cast<uint32_t>(scale));
                for(int lane=0;lane<32;++lane) ASSERT_EQ(bytes[block*32+lane],device_q8_activation_contract::quantize(input[block*32+lane],scale));
            }
        }
        std::vector<float> returned(input.size()),returned_weights(weights.size());
        requireH(backend->deviceToHost(returned.data(),fp(h),returned.size()*4,device.ordinal,stream) &&
            backend->deviceToHost(returned_weights.data(),fp(w),returned_weights.size()*4,device.ordinal,stream) &&
            backend->synchronizeStream(stream,device.ordinal),"H32 source ownership observation");
        EXPECT_EQ(returned,input); EXPECT_EQ(returned_weights,weights);
    });
}
/** @brief Sweep M totality, ragged wave groups and the 128-lane K partition boundary. */
void proveHBackend(DeviceId device)
{
    for(int m=1;m<=31;++m) { proveH(device,m,96,17);if(::testing::Test::HasFatalFailure()) return; }
    for(int k:{32,2048,2080,3072,4160}) proveH(device,17,k,17);
    for(int m:{32,65,511,512,1024,2048}) proveH(device,m,96,17);
    proveH(device,512,2048,256);
}
#ifdef HAVE_CUDA
/** @test CUDA paired rotation, serial equivalence and original expert publication. */
TEST(RouterHadamardQ8, CUDA) { proveHBackend(DeviceId::cuda(0)); }
/** @test Tensor-core prefill preserves raw logits, source operands and expert-bank bytes. */
TEST(RouterHadamardQ8, CUDATensorCorePrefill)
{
    for (auto basis : {RouterQ8Basis::Original, RouterQ8Basis::SignedHadamard32})
    {
        for (int m=1; m<=65; ++m) proveH(DeviceId::cuda(0),m,96,17,HRoute::TensorCorePrefill,basis);
        for (int k : {32,64,96,128,256,384,768,1024,1536,2048,2080,2560,3072,4160,6144,8192,12288})
            proveH(DeviceId::cuda(0),17,k,17,HRoute::TensorCorePrefill,basis);
        for (int m : {128,256,448,511,512,513,768,1024,1536,2048,4096})
            proveH(DeviceId::cuda(0),m,96,17,HRoute::TensorCorePrefill,basis);
        for (int m : {512,1024,2048})
            proveH(DeviceId::cuda(0),m,2048,256,HRoute::TensorCorePrefill,basis);
        for (int experts : {32,64,128,257,512,1024})
            for (int m : {128,513,1537})
                proveH(DeviceId::cuda(0),m,2048,experts,HRoute::TensorCorePrefill,basis);
    }
}
#endif
#ifdef HAVE_ROCM
/** @test ROCm wave64 grouping, live counts and original expert publication. */
TEST(RouterHadamardQ8, ROCm) { proveHBackend(DeviceId::rocm(0)); }
#endif
}
