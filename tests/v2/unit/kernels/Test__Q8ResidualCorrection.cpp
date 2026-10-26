/**
 * @file Test__Q8ResidualCorrection.cpp
 * @brief Device-free prepared-engine proof of explicit two-term Q8 arithmetic.
 *
 * Real Q8_0 tensors enter the actual prepared store; no alternate weight pack
 * certifies the operation. Ragged retained rows, exact serial-row bytes, empty
 * replay, unchanged SingleQ8 output/source identities and minimal scratch are
 * checked through the public engine API. Actual stage submissions additionally
 * prove semantic HC-only observation, failures and single-owner CPU worksharing.
 * An independent FP64 native-source
 * oracle proves the correction improves activation approximation error. An
 * authored cancellation-sensitive projection feeds the actual nonzero read stage.
 */
#include <gtest/gtest.h>
#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "loaders/PreparedWeightStore.h"
#include "kernels/KernelFactory.h"
#include "tensors/Tensors.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/compute_stages/stages/GDNProjectionStage.h"
#include "../../utils/QuantizedVerifierFormats.h"
#include "../../utils/PreparedWeightTestHarness.h"
#include "../../utils/CPUProjectionTestWorkspace.h"
#include "../../utils/HCUpCancellationFixture.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPrimitiveStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "utils/OpenMPUtils.h"
#include "utils/PerfStatsCollector.h"
#include <cstdlib>
#include <optional>
#include <string>
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
 /** @brief Reuse the actual prepared fixture with optional authored source bytes.
  * @param rows Physical retained input rows.
  * @param outputs Prepared output columns.
  * @param columns Complete native input columns.
  * @param authored Optional exact native block fixture.
  * @param sourceName Original semantic binding name; old tests retain their prior default.
  */
 Fixture(int rows,int outputs,int columns=320,const std::vector<std::uint8_t>*authored=nullptr,
         const char*sourceName="hc.attn_up.weight"):m(rows),n(outputs),k(columns),raw(authored?*authored:nativeWeights(n,k)),weight({std::size_t(n),std::size_t(k)},raw),input({std::size_t(m),std::size_t(k)},DeviceId::cpu()),output({std::size_t(m),std::size_t(n)},DeviceId::cpu()),single({1,std::size_t(k)},DeviceId::cpu()),serial({1,std::size_t(n)},DeviceId::cpu()),baseline({std::size_t(m),std::size_t(n)},DeviceId::cpu()),again({std::size_t(m),std::size_t(n)},DeviceId::cpu()) {
  binding=test::makePreparedWeightTestBinding(&weight,DeviceId::cpu(),sourceName,store.modelId());
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

/**
 * @test Real GDN bundle setup authenticates all canonical prepared QKV sources.
 *
 * One immutable source is deliberately shared by the four algebraically valid
 * equal-width siblings. The actual store prepares its real CPU engine; no mock
 * descriptor certifies original provenance. Numerical all-format arithmetic is
 * owned by the separate generic correction sweep, while this regression proves
 * the consumer reaches that engine without a Q8_0-only admission island.
 */
TEST(Q8ResidualCorrection, GDNBundleAdmitsEveryOriginalNativeSource)
{
 llaminar::v2::ThreadCountGuard team(1);
 constexpr int m=3,n=64,k=256;
 for(const auto &format:test::quantizedVerifierFormats())
 {
  SCOPED_TRACE(format.label);
  auto source=format.create({n,k},917);
  auto prepared=test::makePreparedGemmFixture(source.get(),DeviceId::cpu(),"blk.0.attn_qkv.weight");
  FP32Tensor input({m,k}),qkv({m,n}),z({m,n}),alpha({m,n}),beta({m,n});
  int32_t count=m;
  GDNProjectionStage::Params params;
  params.device_id=DeviceId::cpu();params.input=&input;params.m=m;params.k=k;
  params.w_qkv=source.get();params.output_qkv=&qkv;params.n_qkv=n;
  params.w_z=source.get();params.output_z=&z;params.n_z=n;
  params.w_a=source.get();params.output_a=&alpha;params.n_a=n;
  params.w_b=source.get();params.output_b=&beta;params.n_b=n;
  params.prepared_ref_qkv=prepared.ref;params.prepared_ref_z=prepared.ref;
  params.prepared_ref_a=prepared.ref;params.prepared_ref_b=prepared.ref;
  params.prepared_store=prepared.store.get();params.qkv_operand_precision=QuantizedActivationOperandPrecision::TwoTermQ8;
  params.qkv_row_range=DeviceRowRange::deviceCounted(m,&count);
  GDNProjectionStage stage(params);
  std::string error;
  EXPECT_TRUE(stage.validatePreparedWeights(&error))<<error;
  const auto identity=stage.graphCaptureVariantSignature();
  EXPECT_NE(identity,0u);
  const auto requirements=stage.getWorkspaceRequirements(m,n,k);
  const auto *residual=requirements.find(kCPUProjectionResidualQ8);
  ASSERT_NE(residual,nullptr);
  EXPECT_EQ(residual->size_bytes,cpuProjectionQ8Blocks(m,k)*sizeof(Q8_1Block));
  for(const int live:{0,1,m})
  {
   count=live;
   EXPECT_EQ(stage.graphCaptureVariantSignature(),identity);
  }
  auto wrong_rows=params;wrong_rows.qkv_row_range=DeviceRowRange::fullyActive(m+1);
  EXPECT_THROW((void)GDNProjectionStage(wrong_rows),std::invalid_argument);
  auto foreign=params;foreign.prepared_store=nullptr;
  GDNProjectionStage rejected(foreign);
  EXPECT_FALSE(rejected.validatePreparedWeights(&error));
 }
}

namespace {
/** @brief Scope only passive collector configuration; never enable a backend or stage timer. */
class HCObservationSettings final {
 std::optional<std::string> json_,filter_;
 /** @return Original optional environment value, preserving absence. */
 static std::optional<std::string> prior(const char*name){const char*value=std::getenv(name);return value?std::optional<std::string>{value}:std::nullopt;}
 /** @brief Restore one original environment value exactly. */
 static void restore(const char*name,const std::optional<std::string>&value){if(value)setenv(name,value->c_str(),1);else unsetenv(name);}
public:
 /** @brief Select the exact host-submission family and clear earlier fixture observations. */
 HCObservationSettings():json_(prior("LLAMINAR_PERF_STATS_JSON")),filter_(prior("LLAMINAR_PERF_STATS_FILTER")){
  setenv("LLAMINAR_PERF_STATS_JSON","1",1);setenv("LLAMINAR_PERF_STATS_FILTER","kernel.hc_up_projection_submissions",1);
  PerfStatsCollector::reloadConfigurationFromEnvironment();PerfStatsCollector::reset();
 }
 /** @brief Retire fixture records and restore the original collection policy. */
 ~HCObservationSettings(){PerfStatsCollector::reset();restore("LLAMINAR_PERF_STATS_JSON",json_);restore("LLAMINAR_PERF_STATS_FILTER",filter_);PerfStatsCollector::reloadConfigurationFromEnvironment();}
 HCObservationSettings(const HCObservationSettings&)=delete;
 HCObservationSettings&operator=(const HCObservationSettings&)=delete;
};
/** @return Only actual exact-family records, not dotted neighbors or unrelated MTP counters. */
std::vector<PerfStatRecord> hcSubmissions(){
 auto records=PerfStatsCollector::snapshot({"kernel.hc_up_projection_submissions"});
 records.erase(std::remove_if(records.begin(),records.end(),[](const auto&r){return r.domain!="kernel"||r.name!="hc_up_projection_submissions";}),records.end());return records;
}
/**
 * @brief Bind the existing tiny native fixture to the actual prepared HC stage.
 * @param f Original source, store, operands and canonical workspace.
 * @param policy Actual selected activation arithmetic.
 * @param verifier Whether the caller requests grouped decode-equivalent arithmetic.
 * @param count Optional original CPU count for the default whole-prefix descriptor.
 * @param consumedRows Optional exact tiled descriptor overriding that default.
 * @return One actual stage borrowing only fixture-owned source and row metadata.
 */
std::unique_ptr<GEMMStage> hcStage(Fixture&f,QuantizedActivationOperandPrecision policy,bool verifier=false,
 const std::int32_t*count=nullptr,std::optional<DeviceRowRange> consumedRows=std::nullopt){
 auto stage=std::make_unique<GEMMStage>(GEMMStage::Params{.device_id=DeviceId::cpu(),.A=&f.input,.B=&f.weight,.C=&f.output,
  .m=f.m,.n=f.n,.k=f.k,.force_decode_equivalent_verifier_prefill=verifier,
  .verifier_row_range=consumedRows?consumedRows:count?std::optional<DeviceRowRange>{DeviceRowRange::deviceCounted(f.m,count)}:std::nullopt,
  .prepared_ref=f.binding.prepared,.prepared_store=&f.store,.quantized_operand_precision=policy});
 stage->bindWorkspace(f.work->get());return stage;
}
}

/** @test Actual original CPU stages report selected successful paths and immutable row geometry only. */
TEST(Q8ResidualCorrection, HCUpSubmissionCountsActualPoliciesAndDoesNotCountLiveRows)
{
 llaminar::v2::ThreadCountGuard workers(1);HCObservationSettings settings;CPUDeviceContext context(DeviceId::cpu(),1);
 Fixture f(3,64,64,nullptr,"blk.2.hc_ffn_up.weight");
 std::int32_t unconsumed=-123;
 auto single=hcStage(f,QuantizedActivationOperandPrecision::SingleQ8,false,&unconsumed);
 ASSERT_TRUE(single->execute(&context));
 auto records=hcSubmissions();ASSERT_EQ(records.size(),1u);
 EXPECT_EQ(records[0].kind,PerfStatRecord::Kind::Counter);EXPECT_EQ(records[0].count,1u);EXPECT_DOUBLE_EQ(records[0].value,1);
 EXPECT_EQ(records[0].device,DeviceId::cpu().toString());EXPECT_EQ(records[0].phase,"submission");
 EXPECT_EQ(records[0].tags.at("operand_precision"),"single_q8");EXPECT_EQ(records[0].tags.at("operation"),"ordinary");
 EXPECT_EQ(records[0].tags.at("canonical_name"),"blk.2.hc_ffn_up.weight");
 EXPECT_EQ(records[0].tags.at("observation"),"host_execute_or_capture");EXPECT_EQ(records[0].tags.at("row_binding"),"fully_active");
 EXPECT_EQ(unconsumed,-123);
 PerfStatsCollector::reset();
 std::int32_t count=2;auto two=hcStage(f,QuantizedActivationOperandPrecision::TwoTermQ8,true,&count);
 ASSERT_TRUE(two->execute(&context));
 std::fill_n(f.output.mutable_data(),f.output.numel(),-12345.f);count=0;
 ASSERT_TRUE(two->execute(&context));
 for(std::size_t i=0;i<f.output.numel();++i)ASSERT_EQ(f.output.data()[i],-12345.f);
 records=hcSubmissions();ASSERT_EQ(records.size(),1u);EXPECT_EQ(records[0].count,2u);EXPECT_DOUBLE_EQ(records[0].value,2);
 EXPECT_EQ(records[0].tags.at("operand_precision"),"two_term_q8");EXPECT_EQ(records[0].tags.at("operation"),"q8_residual_corrected");
 EXPECT_EQ(records[0].tags.at("m"),"3");EXPECT_EQ(records[0].tags.at("n"),"64");EXPECT_EQ(records[0].tags.at("k"),"64");
 EXPECT_EQ(records[0].tags.at("row_binding"),"counted_prefix");EXPECT_EQ(records[0].tags.at("count_capacity"),"3");
 EXPECT_EQ(records[0].tags.at("verifier_requested"),"true");
 EXPECT_FALSE(records[0].tags.contains("live_rows"));
 // A zero-live accepted call is one host submission, not zero or three model rows.
 two->resetSessionStatePreservingCapturedReplay();PerfStatsCollector::reset();count=1;
 ASSERT_TRUE(two->execute(&context));ASSERT_EQ(hcSubmissions().size(),1u);EXPECT_EQ(hcSubmissions()[0].count,1u);
 PerfStatsCollector::reset();auto grouped=hcStage(f,QuantizedActivationOperandPrecision::SingleQ8,true,&count);
 ASSERT_TRUE(grouped->execute(&context));records=hcSubmissions();ASSERT_EQ(records.size(),1u);
 EXPECT_EQ(records[0].tags.at("operation"),"decode_equivalent");EXPECT_EQ(records[0].tags.at("operand_precision"),"single_q8");
 PerfStatsCollector::reset();setenv("LLAMINAR_PERF_STATS_FILTER","forward_graph",1);
 PerfStatsCollector::reloadConfigurationFromEnvironment();
 ASSERT_TRUE(two->execute(&context));EXPECT_TRUE(hcSubmissions().empty());
}

/** @test Failed/throwing calls and successful unrelated prepared sources cannot mint HC success. */
TEST(Q8ResidualCorrection, HCUpSubmissionRejectsFailedCallsAndForeignSemanticSources)
{
 llaminar::v2::ThreadCountGuard workers(1);HCObservationSettings settings;CPUDeviceContext context(DeviceId::cpu(),1);
 Fixture f(3,64,64,nullptr,"blk.2.hc_ffn_up.weight");auto stage=hcStage(f,QuantizedActivationOperandPrecision::TwoTermQ8);
 EXPECT_FALSE(stage->execute(nullptr));EXPECT_TRUE(hcSubmissions().empty());
 f.input.mutable_data()[f.input.numel()-1]=std::numeric_limits<float>::quiet_NaN();
 EXPECT_THROW(stage->execute(&context),std::invalid_argument);EXPECT_TRUE(hcSubmissions().empty());
 f.input.mutable_data()[f.input.numel()-1]=.25f;stage->unbindWorkspace();
 EXPECT_THROW(stage->execute(&context),std::exception);EXPECT_TRUE(hcSubmissions().empty());
 Fixture foreign(3,64,64,nullptr,"blk.2.ffn_up.weight");auto other=hcStage(foreign,QuantizedActivationOperandPrecision::TwoTermQ8);
 ASSERT_TRUE(other->execute(&context));EXPECT_TRUE(hcSubmissions().empty());
 auto ordinary=hcStage(f,QuantizedActivationOperandPrecision::SingleQ8);
 EXPECT_FALSE(ordinary->execute(nullptr));EXPECT_TRUE(hcSubmissions().empty());
 GEMMStage rejected({.device_id=DeviceId::cpu(),.A=&f.input,.B=&f.weight,.C=&f.output,.m=f.m,.n=f.n-1,.k=f.k,
  .prepared_ref=f.binding.prepared,.prepared_store=&f.store});
 rejected.bindWorkspace(f.work->get());
 std::fill_n(f.output.mutable_data(),f.output.numel(),-12345.f);
 EXPECT_FALSE(rejected.execute(&context));EXPECT_TRUE(hcSubmissions().empty());
 for(std::size_t i=0;i<f.output.numel();++i)ASSERT_EQ(f.output.data()[i],-12345.f);
}

/** @test Existing CPU worksharing publishes one receipt from its owner, never once per worker. */
TEST(Q8ResidualCorrection, HCUpSubmissionCPUWorkshareHasOneObserver)
{
 llaminar::v2::ThreadCountGuard workers(3);HCObservationSettings settings;CPUDeviceContext context(DeviceId::cpu(),3);
 for(auto policy:{QuantizedActivationOperandPrecision::SingleQ8,QuantizedActivationOperandPrecision::TwoTermQ8}) {
  Fixture f(3,64,64,nullptr,"blk.2.hc_ffn_up.weight");auto stage=hcStage(f,policy);
  // Resolve the existing shared execution kernel before entering its team.
  ASSERT_TRUE(stage->execute(&context));PerfStatsCollector::reset();
  std::array<int,3>success{};
#pragma omp parallel num_threads(3)
  {success[omp_get_thread_num()]=stage->execute(&context)?1:0;}
  EXPECT_EQ(success,(std::array<int,3>{1,1,1}));
  const auto records=hcSubmissions();ASSERT_EQ(records.size(),1u);EXPECT_EQ(records[0].count,1u);EXPECT_DOUBLE_EQ(records[0].value,1);
  PerfStatsCollector::reset();
 }
}

/** @test A consumed two-row tile is distinguishable from its three-row retained envelope. */
TEST(Q8ResidualCorrection, HCUpSubmissionPreservesConsumedSliceExtent)
{
 llaminar::v2::ThreadCountGuard workers(1);HCObservationSettings settings;CPUDeviceContext context(DeviceId::cpu(),1);
 Fixture f(3,64,64,nullptr,"blk.2.hc_ffn_up.weight");
 const std::int32_t count=3;
 const auto whole=DeviceRowRange::deviceCounted(f.m,&count);
 const auto tile=whole.slice(0,2);
 auto sliced=hcStage(f,QuantizedActivationOperandPrecision::TwoTermQ8,true,&count,tile);
 std::fill_n(f.output.mutable_data(),f.output.numel(),-12345.f);
 ASSERT_TRUE(sliced->execute(&context));
 for(std::size_t i=0;i<std::size_t(2*f.n);++i) {
  EXPECT_TRUE(std::isfinite(f.output.data()[i]));EXPECT_NE(f.output.data()[i],-12345.f);
 }
 for(std::size_t i=std::size_t(2*f.n);i<f.output.numel();++i) { EXPECT_EQ(f.output.data()[i],-12345.f); }
 auto records=hcSubmissions();ASSERT_EQ(records.size(),1u);EXPECT_EQ(records[0].count,1u);
 const auto sliced_tags=records[0].tags;
 EXPECT_EQ(sliced_tags.at("m"),"3");EXPECT_EQ(sliced_tags.at("count_capacity"),"3");
 EXPECT_EQ(sliced_tags.at("physical_rows"),"2");EXPECT_EQ(sliced_tags.at("first_physical_row"),"0");
 EXPECT_EQ(sliced_tags.at("repeats_per_row"),"1");
 EXPECT_EQ(sliced_tags.at("row_schema_version"),std::to_string(tile.captureIdentity().schema_version));
 // The same source, input, native count and enclosing M consume a wider tile.
 PerfStatsCollector::reset();auto full=hcStage(f,QuantizedActivationOperandPrecision::TwoTermQ8,true,&count,whole);
 ASSERT_TRUE(full->execute(&context));records=hcSubmissions();ASSERT_EQ(records.size(),1u);
 EXPECT_EQ(records[0].count,1u);EXPECT_EQ(records[0].tags.at("physical_rows"),"3");
 auto full_tags=records[0].tags;auto prior_tags=sliced_tags;
 full_tags.erase("physical_rows");prior_tags.erase("physical_rows");EXPECT_EQ(full_tags,prior_tags);
 for(std::size_t i=std::size_t(2*f.n);i<f.output.numel();++i) {
  EXPECT_TRUE(std::isfinite(f.output.data()[i]));EXPECT_NE(f.output.data()[i],-12345.f);
 }
}

/** @test Actual original all-format HC submissions retain source labels, with floating execution explicitly native. */
TEST(Q8ResidualCorrection, HCUpSubmissionAuthenticatesAllOriginalSourceTags)
{
    llaminar::v2::ThreadCountGuard workers(1);
    HCObservationSettings settings;
    CPUDeviceContext context(DeviceId::cpu(),1);
    constexpr int m=1,n=5,k=256;
    const auto check=[&](TensorBase &source, bool floating) {
        auto prepared=test::makePreparedGemmFixture(&source,DeviceId::cpu(),"blk.0.hc_attn_up.weight");
        FP32Tensor input({m,k}),output({m,n});
        std::fill_n(input.mutable_data(),input.numel(),0.125f);
        for(const auto policy:{QuantizedActivationOperandPrecision::SingleQ8,QuantizedActivationOperandPrecision::TwoTermQ8})
        {
            if(floating && policy==QuantizedActivationOperandPrecision::TwoTermQ8) continue;
            PerfStatsCollector::reset();
            GEMMStage stage(GEMMStage::Params{.device_id=DeviceId::cpu(),.A=&input,.B=&source,.C=&output,
                .m=m,.n=n,.k=k,.prepared_ref=prepared.ref,.prepared_store=prepared.store.get(),.quantized_operand_precision=policy});
            auto requirements=test::cpuProjectionTestRequirements(m,{prepared.store->gemmKernel(prepared.ref)});
            requirements.merge(stage.getWorkspaceRequirements(m,n,k));
            test::CPUProjectionTestWorkspace workspace(m,k,requirements);
            stage.bindWorkspace(workspace.get());
            ASSERT_TRUE(stage.execute(&context));
            stage.unbindWorkspace();
            auto records=hcSubmissions(); ASSERT_EQ(records.size(),1u);
            EXPECT_EQ(records[0].tags.at("source_format"),tensorTypeName(source.native_type()));
            EXPECT_EQ(records[0].tags.at("source_arithmetic"),floating ? "floating_native" :
                policy==QuantizedActivationOperandPrecision::TwoTermQ8 ? "quantized_two_term_q8" : "quantized_single_q8");
            EXPECT_EQ(records[0].count,1u);
        }
    };
    for(const auto &format:test::quantizedVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        auto source=format.create({n,k},917);
        check(*source,false);
    }
    FP32Tensor floating({n,k});
    std::fill_n(floating.mutable_data(),floating.numel(),0.25f);
    check(floating,true);
}

/** @test Floating HC request holes retain their own immutable length authority, never a fabricated prefix. */
TEST(Q8ResidualCorrection, HCUpSubmissionRetainsActualFloatingRequestRows)
{
    llaminar::v2::ThreadCountGuard workers(1);
    HCObservationSettings settings;
    CPUDeviceContext context(DeviceId::cpu(),1);
    constexpr int m=4,n=5,k=32;
    FP32Tensor source({n,k}),input({m,k}),output({m,n});
    std::fill_n(source.mutable_data(),source.numel(),0.25f);
    std::fill_n(input.mutable_data(),input.numel(),0.5f);
    auto prepared=test::makePreparedGemmFixture(&source,DeviceId::cpu(),"blk.0.hc_ffn_up.weight");
    // These deliberately invalid live publications are not inspected at setup.
    // Only the real CPU consumer sees valid producer lengths at execution.
    std::int32_t lengths[2]{-99,99};
    GEMMStage stage(GEMMStage::Params{.device_id=DeviceId::cpu(),.A=&input,.B=&source,.C=&output,
        .m=m,.n=n,.k=k,.force_decode_equivalent_verifier_prefill=true,
        .request_rows=DeviceRequestRowRanges::deviceCounted(2,2,lengths),
        .prepared_ref=prepared.ref,.prepared_store=prepared.store.get()});
    auto requirements=test::cpuProjectionTestRequirements(m,{prepared.store->gemmKernel(prepared.ref)});
    requirements.merge(stage.getWorkspaceRequirements(m,n,k));
    test::CPUProjectionTestWorkspace workspace(m,k,requirements);
    stage.bindWorkspace(workspace.get());
    const auto capture_identity=stage.graphCaptureVariantSignature();
    for(int visit=0;visit<2;++visit)
    {
        PerfStatsCollector::reset();
        lengths[0]=visit==0 ? 0 : 1; lengths[1]=visit==0 ? 2 : 0;
        std::fill_n(output.mutable_data(),output.numel(),-12345.f);
        ASSERT_TRUE(stage.execute(&context));
        EXPECT_EQ(stage.graphCaptureVariantSignature(),capture_identity);
        auto records=hcSubmissions(); ASSERT_EQ(records.size(),1u);
        const auto &tags=records[0].tags;
        EXPECT_EQ(records[0].count,1u);
        EXPECT_EQ(tags.at("source_arithmetic"),"floating_native");
        EXPECT_EQ(tags.at("operation"),"decode_equivalent");
        EXPECT_EQ(tags.at("row_binding"),"request_counted_prefixes");
        EXPECT_EQ(tags.at("length_owner"),std::to_string(reinterpret_cast<uintptr_t>(lengths)));
        EXPECT_EQ(tags.at("requests"),"2"); EXPECT_EQ(tags.at("rows_per_request"),"2");
        EXPECT_EQ(tags.at("physical_rows"),"4");
        for(const auto *prefix_only:{"count_owner","count_capacity","first_physical_row","row_schema_version","repeats_per_row","live_rows"})
            EXPECT_FALSE(tags.contains(prefix_only));
        for(int row=0;row<m;++row)
            for(int column=0;column<n;++column)
                EXPECT_FLOAT_EQ(output.data()[row*n+column],row%2<lengths[row/2] ? 4.f : -12345.f);
    }
    stage.unbindWorkspace();
}

namespace
{
/** @return Exact GDN-output family records, excluding unchanged HC observations. */
std::vector<PerfStatRecord> gdnOutputSubmissions()
{
    auto records = PerfStatsCollector::snapshot({"kernel.gdn_output_projection_submissions"});
    records.erase(std::remove_if(records.begin(), records.end(), [](const auto &record) {
        return record.domain != "kernel" || record.name != "gdn_output_projection_submissions";
    }), records.end());
    return records;
}
/** @brief Select the existing passive family without changing any arithmetic policy. */
void selectGDNOutputObservation()
{
    setenv("LLAMINAR_PERF_STATS_FILTER", "kernel", 1);
    PerfStatsCollector::reloadConfigurationFromEnvironment();
    PerfStatsCollector::reset();
}
}

/** @test Actual prepared residual success records physical ownership, never live rows or completion. */
TEST(Q8ResidualCorrection, GDNOutputSubmissionCountsSuccessfulResidualAndRetainsRowBinding)
{
    llaminar::v2::ThreadCountGuard workers(1);
    HCObservationSettings settings;
    selectGDNOutputObservation();
    CPUDeviceContext context(DeviceId::cpu(), 1);
    Fixture source(3, 64, 64, nullptr, "blk.2.ssm_out.weight");
    ASSERT_EQ(source.binding.identity.role, WeightRole::GDNProjection);
    int count = 2;
    auto stage = hcStage(source, QuantizedActivationOperandPrecision::TwoTermQ8, true, &count);
    for (int live : {2, 0, 1})
    {
        count = live;
        ASSERT_TRUE(stage->execute(&context));
    }
    const auto records = gdnOutputSubmissions();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].count, 3u);
    EXPECT_EQ(records[0].device, DeviceId::cpu().toString());
    EXPECT_EQ(records[0].phase, "submission");
    EXPECT_EQ(records[0].tags.at("canonical_name"), "blk.2.ssm_out.weight");
    EXPECT_EQ(records[0].tags.at("weight_role"), "GDNProjection");
    EXPECT_EQ(records[0].tags.at("operand_precision"), "two_term_q8");
    EXPECT_EQ(records[0].tags.at("operation"), "q8_residual_corrected");
    EXPECT_EQ(records[0].tags.at("observation"), "host_execute_or_capture");
    EXPECT_EQ(records[0].tags.at("row_binding"), "counted_prefix");
    EXPECT_EQ(records[0].tags.at("physical_rows"), "3");
    EXPECT_FALSE(records[0].tags.contains("live_rows"));
    EXPECT_TRUE(hcSubmissions().empty());
    PerfStatsCollector::reset();
    setenv("LLAMINAR_PERF_STATS_FILTER", "forward_graph", 1);
    PerfStatsCollector::reloadConfigurationFromEnvironment();
    ASSERT_TRUE(stage->execute(&context));
    EXPECT_TRUE(gdnOutputSubmissions().empty());
}

/** @test Failed calls, ordinary policy and foreign sources cannot mint residual success. */
TEST(Q8ResidualCorrection, GDNOutputSubmissionRejectsFailureAndPreservesHCMeaning)
{
    llaminar::v2::ThreadCountGuard workers(1);
    HCObservationSettings settings;
    selectGDNOutputObservation();
    CPUDeviceContext context(DeviceId::cpu(), 1);
    Fixture source(1, 64, 64, nullptr, "blk.2.ssm_out.weight");
    auto stage = hcStage(source, QuantizedActivationOperandPrecision::TwoTermQ8);
    EXPECT_FALSE(stage->execute(nullptr));
    EXPECT_TRUE(gdnOutputSubmissions().empty());
    source.input.mutable_data()[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(stage->execute(&context), std::invalid_argument);
    EXPECT_TRUE(gdnOutputSubmissions().empty());
    source.input.mutable_data()[0] = .25f;
    auto single = hcStage(source, QuantizedActivationOperandPrecision::SingleQ8);
    ASSERT_TRUE(single->execute(&context));
    EXPECT_TRUE(gdnOutputSubmissions().empty());
    Fixture hc(1, 64, 64, nullptr, "blk.2.hc_ffn_up.weight");
    ASSERT_TRUE(hcStage(hc, QuantizedActivationOperandPrecision::TwoTermQ8)->execute(&context));
    EXPECT_TRUE(gdnOutputSubmissions().empty());
    ASSERT_EQ(hcSubmissions().size(), 1u);
    EXPECT_EQ(hcSubmissions()[0].count, 1u);
}

/** @test Existing CPU worksharing reports one successful submission, not one per worker. */
TEST(Q8ResidualCorrection, GDNOutputSubmissionCPUWorkshareHasOneObserver)
{
    llaminar::v2::ThreadCountGuard workers(3);
    HCObservationSettings settings;
    selectGDNOutputObservation();
    CPUDeviceContext context(DeviceId::cpu(), 3);
    Fixture source(3, 64, 64, nullptr, "blk.2.ssm_out.weight");
    auto stage = hcStage(source, QuantizedActivationOperandPrecision::TwoTermQ8);
    ASSERT_TRUE(stage->execute(&context));
    PerfStatsCollector::reset();
    std::array<int, 3> success{};
#pragma omp parallel num_threads(3)
    {success[omp_get_thread_num()] = stage->execute(&context) ? 1 : 0;}
    for (int value : success) ASSERT_EQ(value, 1);
    ASSERT_EQ(gdnOutputSubmissions().size(), 1u);
    EXPECT_EQ(gdnOutputSubmissions()[0].count, 1u);
}
