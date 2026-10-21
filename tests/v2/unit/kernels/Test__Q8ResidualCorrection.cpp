/**
 * @file Test__Q8ResidualCorrection.cpp
 * @brief Device-free prepared-engine proof of explicit two-term Q8 arithmetic.
 *
 * Real Q8_0 tensors enter the actual prepared store; no alternate weight pack
 * certifies the operation. Ragged retained rows, exact serial-row bytes, empty
 * replay, unchanged SingleQ8 output/source identities and minimal scratch are
 * checked through the public engine API. An independent FP64 native-source
 * oracle proves the correction improves activation approximation error. An
 * authored cancellation-sensitive projection feeds the actual nonzero read stage.
 */
#include <gtest/gtest.h>
#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "loaders/PreparedWeightStore.h"
#include "kernels/KernelFactory.h"
#include "tensors/Tensors.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "../../utils/PreparedWeightTestHarness.h"
#include "../../utils/CPUProjectionTestWorkspace.h"
#include "../../utils/HCUpCancellationFixture.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPrimitiveStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "utils/OpenMPUtils.h"
#include <array>
#include <cmath>
#include <cstring>
#include <vector>
using namespace llaminar2;
namespace {
/** @brief Construct unchanged native blocks with signed source bytes and distinct exact scales. */
std::vector<std::uint8_t> nativeWeights(int n,int k){std::vector<std::uint8_t>raw(std::size_t(n)*k/32*sizeof(Q8_0Block));for(std::size_t b=0;b<raw.size()/34;++b){Q8_0Block block{};block.d=fp32_to_fp16(.002f*float(1+b%11));for(int i=0;i<32;++i)block.qs[i]=static_cast<std::int8_t>((b*13+i*17)%255-127);std::memcpy(raw.data()+b*34,&block,34);}return raw;}
/** @brief Retain one actual immutable prepared binding and its exact primary/secondary arena. */
struct Fixture {
 int m,n,k;std::vector<std::uint8_t> raw;Q8_0Tensor weight;PreparedWeightStore store{ModelContextId{801933}};
 FP32Tensor input,output,single,serial,baseline,again;WeightBinding binding;ITensorGemm*kernel;std::unique_ptr<test::CPUProjectionTestWorkspace>work;
 /** @brief Reuse the actual prepared fixture with optional authored original source bytes. */
 Fixture(int rows,int outputs,int columns=320,const std::vector<std::uint8_t>*authored=nullptr):m(rows),n(outputs),k(columns),raw(authored?*authored:nativeWeights(n,k)),weight({std::size_t(n),std::size_t(k)},raw),input({std::size_t(m),std::size_t(k)},DeviceId::cpu()),output({std::size_t(m),std::size_t(n)},DeviceId::cpu()),single({1,std::size_t(k)},DeviceId::cpu()),serial({1,std::size_t(n)},DeviceId::cpu()),baseline({std::size_t(m),std::size_t(n)},DeviceId::cpu()),again({std::size_t(m),std::size_t(n)},DeviceId::cpu()) {
  binding=test::makePreparedWeightTestBinding(&weight,DeviceId::cpu(),"hc.attn_up.weight",store.modelId());
  // Construct and register the actual production engine: neither a test
  // weight pack nor an alternate quantizer certifies this operation.
  using Factory=llaminar::v2::kernels::KernelFactory;
  auto handle=std::make_shared<Factory::PreparedGemmHandle>();handle->tensor=&weight;handle->device_id=DeviceId::cpu();handle->kind=Factory::GemmPreparationKind::CPU_PACKED;handle->variant=static_cast<int>(TensorType::Q8_0);
  handle->prepared_weights=std::make_shared<Factory::PreparedGemmWeights>();handle->prepared_weights->kind=handle->kind;
  handle->prepared_weights->owned_kernel=std::make_shared<cpu::native_vnni::CPUNativeVNNIGemmKernel>(&weight);handle->prepared_weights->kernel=handle->prepared_weights->owned_kernel.get();
  binding.prepared=store.registerPreparedGemmHandle(binding,PreparedWeightKind::CpuPackedGemm,DeviceId::cpu(),std::move(handle));kernel=store.gemmKernel(*binding.prepared);if(!kernel)throw std::runtime_error("Prepared Q8 source missing");
  auto requirements=kernel->q8ResidualCorrectionWorkspaceRequirements(m,n,k);
  if(requirements.buffers.size()!=2)throw std::runtime_error("Two-term CPU Q8 scratch is not two exact activation banks");
  requirements.merge(dynamic_cast<IWorkspaceConsumer*>(kernel)->getWorkspaceRequirements(m,n,k));work=std::make_unique<test::CPUProjectionTestWorkspace>(requirements);
  for(int row=0;row<m;++row)for(int c=0;c<k;++c)input.mutable_data()[std::size_t(row)*k+c]=c%32==0?float((row%7)+1)*17.25f:std::sin(float(c*7+row*11))*.31f;
 }
};
}
/** @test Every live retained row matches independent serial bytes and holes survive large/small/empty reuse. */
TEST(Q8ResidualCorrection, PreparedSourceSerialRowsAndExactScratch){
 omp_set_num_threads(3);
 for(auto geometry:std::array{std::array{1,65},std::array{15,127},std::array{512,64},std::array{1024,64},std::array{1,10240},std::array{15,10240},std::array{16,10240},std::array{17,10240}}){Fixture f(geometry[0],geometry[1]);SCOPED_TRACE(::testing::Message()<<"M="<<f.m<<" N="<<f.n);const auto*source=f.kernel->exportCPUNativeVNNIPackedWeights();
 ASSERT_NE(source,nullptr);const auto*identity=source->native_interleaved.data();std::vector<std::uint8_t>original(source->native_interleaved.begin(),source->native_interleaved.end());
 ASSERT_TRUE(f.kernel->multiply_tensor(&f.input,&f.baseline,f.m,f.n,f.k,true,1,0,nullptr,nullptr,-1,f.work->get()));
 const auto*primary=static_cast<const Q8_1Block*>(f.work->get()->getBuffer(kCPUProjectionQ8));
 std::vector<Q8_1Block>ordinary(primary,primary+std::size_t(f.m)*f.k/32);
 for(int live:std::array{f.m,std::min(2,f.m),0,f.m}){
  std::fill_n(f.output.mutable_data(),f.output.numel(),-12345.f);
  f.kernel->multiply_q8_residual_corrected(&f.input,&f.output,DeviceRowRange::deviceCounted(f.m,&live),f.n,f.k,f.work->get());
  if(live)EXPECT_EQ(std::memcmp(primary,ordinary.data(),std::size_t(live)*f.k/32*sizeof(Q8_1Block)),0);
  for(int row=0;row<live;++row){std::memcpy(f.single.mutable_data(),f.input.data()+std::size_t(row)*f.k,f.k*4);f.kernel->multiply_q8_residual_corrected(&f.single,&f.serial,DeviceRowRange::fullyActive(1),f.n,f.k,f.work->get());
 EXPECT_EQ(std::memcmp(f.output.data()+std::size_t(row)*f.n,f.serial.data(),f.n*4),0)<<"row="<<row;}
  for(std::size_t i=std::size_t(live)*f.n;i<f.output.numel();++i)ASSERT_EQ(f.output.data()[i],-12345.f);
 }
 EXPECT_EQ(source->native_interleaved.data(),identity);
 EXPECT_TRUE(std::equal(original.begin(),original.end(),source->native_interleaved.begin()));
 EXPECT_EQ(f.store.gemmKernel(*f.binding.prepared),f.kernel);
 ASSERT_TRUE(f.kernel->multiply_tensor(&f.input,&f.again,f.m,f.n,f.k,true,1,0,nullptr,nullptr,-1,f.work->get()));
 EXPECT_EQ(std::memcmp(f.again.data(),f.baseline.data(),f.again.numel()*4),0);
 }
}
/** @test Two quantized operands reduce approximation error against exact native source weights. */
TEST(Q8ResidualCorrection, ReducesActivationErrorWithoutChangingNativeSource){Fixture f(1,64);
 ASSERT_TRUE(f.kernel->multiply_tensor(&f.input,&f.baseline,1,f.n,f.k,true,1,0,nullptr,nullptr,-1,f.work->get()));f.kernel->multiply_q8_residual_corrected(&f.input,&f.output,DeviceRowRange::fullyActive(1),f.n,f.k,f.work->get());long double error0=0,error1=0;for(int n=0;n<f.n;++n){long double exact=0;for(int k=0;k<f.k;++k){Q8_0Block block;std::memcpy(&block,f.raw.data()+(std::size_t(n)*(f.k/32)+k/32)*34,34);exact+=static_cast<long double>(f.input.data()[k])*fp16_to_fp32(block.d)*block.qs[k%32];}const auto d0=f.baseline.data()[n]-exact,d1=f.output.data()[n]-exact;error0+=d0*d0;error1+=d1*d1;}EXPECT_LT(error1,error0/10000);}
/** @test Unsupported source/geometry, absent banks and corrupt count fail before mutation. */
TEST(Q8ResidualCorrection, FailsClosedBeforeMutation){Fixture f(15,64);std::fill_n(f.output.mutable_data(),f.output.numel(),-12345.f);int invalid=16;
 EXPECT_THROW(f.kernel->multiply_q8_residual_corrected(&f.input,&f.output,DeviceRowRange::deviceCounted(15,&invalid),64,320,f.work->get()),std::invalid_argument);
 EXPECT_THROW(f.kernel->multiply_q8_residual_corrected(&f.input,&f.output,DeviceRowRange::fullyActive(15),64,320,nullptr),std::runtime_error);for(std::size_t i=0;i<f.output.numel();++i)ASSERT_EQ(f.output.data()[i],-12345.f);
 EXPECT_THROW(f.kernel->q8ResidualCorrectionWorkspaceRequirements(15,64,319),std::invalid_argument);}
/** @test The exact primary publication agrees with the ordinary prepared engine on adversarial blocks. */
TEST(Q8ResidualCorrection, AuthenticatesNativePrimaryZeroSubnormalAndTieBlocks)
{
 Fixture f(15,64);
 for(int pattern=0;pattern<4;++pattern) {
  for(int row=0;row<f.m;++row)for(int c=0;c<f.k;++c) {
   float value=0;
   if(pattern==1)value=std::numeric_limits<float>::denorm_min()*float(c%32+1);
   if(pattern==2)value=(c%32==0)?127.f:float(c%11-5)+.5f;
   if(pattern==3)value=1e-11f*float(c%7-3);
   f.input.mutable_data()[std::size_t(row)*f.k+c]=value;
  }
  ASSERT_TRUE(f.kernel->multiply_tensor(&f.input,&f.baseline,f.m,f.n,f.k,true,1,0,nullptr,nullptr,-1,f.work->get()));
  auto*published=static_cast<const Q8_1Block*>(f.work->get()->getBuffer(kCPUProjectionQ8));
  std::vector<Q8_1Block>ordinary(published,published+std::size_t(f.m)*f.k/32);
  f.kernel->multiply_q8_residual_corrected(&f.input,&f.output,DeviceRowRange::fullyActive(f.m),f.n,f.k,f.work->get());
  EXPECT_EQ(std::memcmp(ordinary.data(),published,ordinary.size()*sizeof(Q8_1Block)),0)<<"pattern="<<pattern;
 }
}
/** @test Runtime values outside the native half-scale range fail without changing any output. */
TEST(Q8ResidualCorrection, RejectsNonfiniteAndNativeScaleOverflowBeforeOutput)
{
 Fixture f(3,64);
 for(float invalid:std::array{1e9f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
  std::fill_n(f.output.mutable_data(),f.output.numel(),-12345.f);
  f.input.mutable_data()[f.input.numel()-1]=invalid;
  EXPECT_THROW(f.kernel->multiply_q8_residual_corrected(&f.input,&f.output,DeviceRowRange::fullyActive(f.m),f.n,f.k,f.work->get()),std::invalid_argument);
  for(std::size_t i=0;i<f.output.numel();++i)ASSERT_EQ(f.output.data()[i],-12345.f);
  f.input.mutable_data()[f.input.numel()-1]=.25f;
 }
}

/** @test Exact physical operand capacities fail before scratch or output publication. */
TEST(Q8ResidualCorrection, RejectsUndersizedAndInteriorAliasedOperandsBeforeMutation)
{
 Fixture f(3,64);
 std::fill_n(f.output.mutable_data(),f.output.numel(),-12345.f);
 FP32Tensor shortInput({2,320},DeviceId::cpu()),shortOutput({2,64},DeviceId::cpu());
 EXPECT_THROW(f.kernel->multiply_q8_residual_corrected(&shortInput,&f.output,DeviceRowRange::fullyActive(3),64,320,f.work->get()),std::invalid_argument);
 EXPECT_THROW(f.kernel->multiply_q8_residual_corrected(&f.input,&shortOutput,DeviceRowRange::fullyActive(3),64,320,f.work->get()),std::invalid_argument);
 auto sharedInput=std::make_shared<FP32Tensor>(std::vector<std::size_t>{3,320},DeviceId::cpu());
 std::copy_n(f.input.data(),f.input.numel(),sharedInput->mutable_data());
 auto alias=sharedInput->create_view({3,64},1);
 ASSERT_NE(alias,nullptr);
 GEMMStage::Params aliasParams{.device_id=DeviceId::cpu(),.A=sharedInput.get(),.B=&f.weight,.C=alias.get(),.m=3,.n=64,.k=320,
  .prepared_ref=f.binding.prepared,.prepared_store=&f.store,.quantized_operand_precision=QuantizedActivationOperandPrecision::TwoTermQ8};
 EXPECT_THROW(GEMMStage invalid(aliasParams),std::invalid_argument);
 const std::vector<float>before(sharedInput->data(),sharedInput->data()+sharedInput->numel());
 EXPECT_THROW(f.kernel->multiply_q8_residual_corrected(sharedInput.get(),alias.get(),DeviceRowRange::fullyActive(3),64,320,f.work->get()),std::invalid_argument);
 EXPECT_EQ(std::memcmp(before.data(),sharedInput->data(),before.size()*sizeof(float)),0);
 for(std::size_t i=0;i<f.output.numel();++i)ASSERT_EQ(f.output.data()[i],-12345.f);
 const auto*packed=f.kernel->exportCPUNativeVNNIPackedWeights();ASSERT_NE(packed,nullptr);
 auto*primary=static_cast<Q8_1Block*>(f.work->get()->getBuffer(kCPUProjectionQ8));
 const auto blocks=cpuProjectionQ8Blocks(3,320);
 // Different starting addresses still alias one physical activation bank.
 EXPECT_THROW(cpu::native_vnni::cpuQ8ResidualCorrectedProjection(*packed,f.input.data(),f.output.mutable_data(),DeviceRowRange::fullyActive(3),
  std::span<Q8_1Block>(primary,blocks),std::span<Q8_1Block>(primary+1,blocks)),std::invalid_argument);
}
/** @test Device-free metadata checks cover exact GPU/CPU byte intervals without dereferencing native pointers. */
TEST(Q8ResidualCorrection, ValidatesExactByteIntervalsAndReadOnlyAliasing)
{
 alignas(64)std::array<std::byte,512>storage;
 const std::array writes{Q8ResidualStorageSpan{storage.data(),64,16},Q8ResidualStorageSpan{storage.data()+64,64,16}};
 const std::array reads{Q8ResidualStorageSpan{storage.data()+128,64,16},Q8ResidualStorageSpan{storage.data()+144,16,16},Q8ResidualStorageSpan{nullptr,0,4}};
 EXPECT_NO_THROW(validateQ8ResidualStorage(writes,reads));
 const std::array writeOverlap{writes[0],Q8ResidualStorageSpan{storage.data()+63,64,1}};
 EXPECT_THROW(validateQ8ResidualStorage(writeOverlap,reads),std::invalid_argument);
 const std::array readOverlap{Q8ResidualStorageSpan{storage.data()+127,8,1}};
 EXPECT_THROW(validateQ8ResidualStorage(writes,readOverlap),std::invalid_argument);
 const std::array misaligned{Q8ResidualStorageSpan{storage.data()+129,4,4}};
 EXPECT_THROW(validateQ8ResidualStorage(writes,misaligned),std::invalid_argument);
 const std::array overflow{Q8ResidualStorageSpan{reinterpret_cast<const void*>(std::numeric_limits<std::uintptr_t>::max()-3),8,1}};
 EXPECT_THROW(validateQ8ResidualStorage(writes,overflow),std::invalid_argument);
 const std::array absent{Q8ResidualStorageSpan{nullptr,4,4}};
 EXPECT_THROW(validateQ8ResidualStorage(writes,absent),std::invalid_argument);
}
/** @test Arithmetic policy and count-address/slice metadata distinguish captures, while live count changes do not. */
TEST(Q8ResidualCorrection, StageCaptureIdentityFreezesPrecisionBackendGeometryAndCountOwner)
{
 Fixture f(3,64);int count=-123,other=-456;
 GEMMStage::Params p{.device_id=DeviceId::cpu(),.A=&f.input,.B=&f.weight,.C=&f.output,.m=3,.n=64,.k=320,
  .prepared_ref=f.binding.prepared,.prepared_store=&f.store};
 GEMMStage ordinary(p);EXPECT_EQ(ordinary.graphCaptureVariantSignature(),0u);
 p.quantized_operand_precision=QuantizedActivationOperandPrecision::TwoTermQ8;
 p.verifier_row_range=DeviceRowRange::deviceCounted(6,&count).slice(1,3);
 GEMMStage corrected(p),identical(p);const auto signature=corrected.graphCaptureVariantSignature();
 EXPECT_NE(signature,0u);EXPECT_EQ(signature,identical.graphCaptureVariantSignature());
 count=0;EXPECT_EQ(signature,corrected.graphCaptureVariantSignature());
 p.verifier_row_range=DeviceRowRange::deviceCounted(6,&other).slice(1,3);
 GEMMStage differentOwner(p);EXPECT_NE(signature,differentOwner.graphCaptureVariantSignature());
 p.verifier_row_range=DeviceRowRange::deviceCounted(6,&count).slice(2,3);
 GEMMStage differentOrigin(p);EXPECT_NE(signature,differentOrigin.graphCaptureVariantSignature());
 p.verifier_row_range=DeviceRowRange::deviceCounted(6,&count).slice(1,3);p.device_id=DeviceId::cuda(0);
 GEMMStage cuda(p);EXPECT_NE(signature,cuda.graphCaptureVariantSignature());
 p.device_id=DeviceId::rocm(0);GEMMStage rocm(p);EXPECT_NE(cuda.graphCaptureVariantSignature(),rocm.graphCaptureVariantSignature());
 p.device_id=DeviceId::cpu();p.m=2;p.verifier_row_range.reset();GEMMStage differentRows(p);EXPECT_NE(signature,differentRows.graphCaptureVariantSignature());
 p.m=4;EXPECT_THROW(GEMMStage invalid(p),std::invalid_argument);
}

/**
 * @test An authored cancellation-sensitive original Q8_0 projection retains a
 * nonzero read signal through the actual CPU GEMM and ResidualRead stages.
 * Absolute motif bounds also pass matching CUDA/ROCm validation; this is
 * neither a model tolerance nor a hardcoded historical improvement ratio.
 */
TEST(Q8ResidualCorrection, HCUpCancellationRetainsReadSignal)
{
    namespace hc = test::hc_up_cancellation;
    llaminar::v2::ThreadCountGuard team(1);
    constexpr int m = 1, n = 64, k = hc::kColumns, hidden = n / 4;
    const auto raw = hc::weights(n);
    const auto input = hc::activations(m);
    const auto branches = hc::normalized(m, n);
    const auto oracle = hc::reference(raw, input, branches, m, n);
    Fixture f(m, n, k, &raw);
    std::copy(input.begin(), input.end(), f.input.mutable_data());
    FP32Tensor normalized({m, n}, DeviceId::cpu());
    FP32Tensor singleRead({m, hidden}, DeviceId::cpu()), correctedRead({m, hidden}, DeviceId::cpu());
    std::copy(branches.begin(), branches.end(), normalized.mutable_data());
    CPUDeviceContext context(DeviceId::cpu(), 1);
    using Primitive = Qwen4ExpPrimitiveStage;
    using Float = Qwen4ExpFloatBinding;
    Primitive readSingle({.rows={m,{}}, .operation=Primitive::ResidualRead{hidden,
        Float(normalized,BufferId::NORMALIZED), Float(f.baseline,BufferId::GATE_PROJ),
        Float(singleRead,BufferId::ATTN_OUTPUT)}});
    Primitive readCorrected({.rows={m,{}}, .operation=Primitive::ResidualRead{hidden,
        Float(normalized,BufferId::NORMALIZED), Float(f.output,BufferId::GATE_PROJ),
        Float(correctedRead,BufferId::ATTN_OUTPUT)}});
    GEMMStage corrected({.device_id=DeviceId::cpu(), .A=&f.input, .B=&f.weight, .C=&f.output,
        .m=m, .n=n, .k=k, .a_buffer_id=BufferId::NORMALIZED, .c_buffer_id=BufferId::GATE_PROJ,
        .prepared_ref=f.binding.prepared, .prepared_store=&f.store,
        .quantized_operand_precision=QuantizedActivationOperandPrecision::TwoTermQ8});
    corrected.bindWorkspace(f.work->get());
    ASSERT_TRUE(f.kernel->multiply_tensor(&f.input,&f.baseline,m,n,k,true,1,0,nullptr,nullptr,-1,f.work->get()));
    ASSERT_TRUE(readSingle.execute(&context));
    ASSERT_TRUE(corrected.execute(&context));
    ASSERT_TRUE(readCorrected.execute(&context));
    for (int column=0; column<n; ++column)
    {
        ASSERT_TRUE(std::isfinite(f.output.data()[column]));
        EXPECT_NEAR(f.output.data()[column],oracle.logits[column],hc::kLogitAbsoluteError);
    }
    for (int channel=0; channel<hidden; ++channel)
    {
        EXPECT_GT(std::abs(oracle.read[channel]),.2);
        ASSERT_TRUE(std::isfinite(singleRead.data()[channel]));
        EXPECT_GT(std::abs(double(singleRead.data()[channel])-oracle.read[channel]),hc::kSingleReadErrorFloor);
        ASSERT_TRUE(std::isfinite(correctedRead.data()[channel]));
        EXPECT_NEAR(correctedRead.data()[channel],oracle.read[channel],hc::kReadAbsoluteError);
    }
    EXPECT_EQ(std::memcmp(raw.data(),f.weight.raw_data(),raw.size()),0);
    corrected.unbindWorkspace();
}
