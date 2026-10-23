/**
 * @file Test__Q8ResidualAllFormatsCPU.cpp
 * @brief Explicit CPU integration sweep of original-format TwoTerm rows,
 * threads and tails.
 *
 * Source weights and captured-stage policy remain production-owned. The M1 row
 * loop exists only as an independent diagnostic oracle. ISA aliases start new
 * processes because the canonical runtime ISA policy is immutable after first
 * use.
 */
#include "../../utils/Q8ResidualAllFormatCPUHarness.h"
#include "utils/MPIBootstrap.h"
#include "../../utils/HC103CPUControlChain.h"

/** @test All positive worker teams execute every format on a tiny complete
 * source. */
TEST(Q8ResidualAllFormatsCPU, NativeRowsThreadsAndLargeSmallReuse) {
  const int budget = std::max(
      1, llaminar2::MPIBootstrap::detectCPUTopology().cores_per_socket);
  std::vector<int> workers(budget);
  std::iota(workers.begin(), workers.end(), 1);
  std::vector<int> representative{1, std::min(budget, 3), budget};
  std::sort(representative.begin(), representative.end());
  representative.erase(
      std::unique(representative.begin(), representative.end()),
      representative.end());
  for (const auto &format : llaminar2::test::quantizedVerifierFormats()) {
    const std::array tinyLives{3, 0, 1, 3};
    llaminar2::test::q8_all_formats::proveCPUThreads(format, 3, 5, 256, workers,
                                                     false, tinyLives);
    ASSERT_FALSE(::testing::Test::HasFailure());
    for (int rows : {1, 9, 15, 33, 64, 65}) {
      const std::array single{1};
      const std::span<const int> teams =
          rows == 65 ? std::span<const int>(representative)
                     : std::span<const int>(single);
      llaminar2::test::q8_all_formats::proveCPUThreads(format, rows, 65, 512,
                                                       teams);
      ASSERT_FALSE(::testing::Test::HasFailure());
    }
  }
}
/** @test Large physical work uses bounded distinct-row oracles and retains
 * poisoned suffixes. */
TEST(Q8ResidualAllFormatsCPU, NativeLargeTiledRows) {
  const std::array team{1};
  for (const auto &format : llaminar2::test::quantizedVerifierFormats()) {
    for (int rows : {512, 1024, 2048}) {
      const std::array lives{rows, 1, 0, std::min(rows, 15), rows};
      llaminar2::test::q8_all_formats::proveCPUThreads(format, rows, 65, 256,
                                                       team, true, lives);
      ASSERT_FALSE(::testing::Test::HasFailure());
    }
    // A separate untiled (seven distinct authored rows) large bank prevents the
    // one-row tiling specialization from being mistaken for general M support.
    const std::array lives{2048, 512, 1024, 1, 0, 2048};
    llaminar2::test::q8_all_formats::proveCPUThreads(format, 2048, 5, 256, team,
                                                     false, lives);
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
}

/** @test Calibrate frozen native CPU HC arithmetic against actual prepared control stages. */
TEST(Q8ResidualAllFormatsCPU, HC103OriginalPreparedControlChain) {
 const int budget=std::max(1,llaminar2::MPIBootstrap::detectCPUTopology().cores_per_socket);
 std::vector<int> teams{1,std::min(3,budget),budget};
 std::sort(teams.begin(),teams.end());teams.erase(std::unique(teams.begin(),teams.end()),teams.end());
 const std::array one{1};
 for(const auto &format:llaminar2::test::quantizedVerifierFormats()) {
  for(bool alias:{false,true}) {
   llaminar2::test::hc103_cpu::prove(format,3,64,256,teams,alias);
   ASSERT_FALSE(::testing::Test::HasFailure());
   for(int rows:{1,9,33}) {
    llaminar2::test::hc103_cpu::prove(format,rows,64,256,one,alias);
    ASSERT_FALSE(::testing::Test::HasFailure());
   }
   // Original256-value source blocks cannot represent K320. The canonical
   // non-superblock families cover original HC rank320 and partial activation CTA.
   if(!format.source_is_superblock) {
    llaminar2::test::hc103_cpu::prove(format,9,72,320,one,alias);
    ASSERT_FALSE(::testing::Test::HasFailure());
   }
  }
 }
}
namespace {
/** @brief Calibrate only real CPU primitive activation/read, without NativeVNNI admission. */
void hc103OriginalPrimitiveTails() {
 using namespace llaminar2;
 using namespace llaminar2::test;
 q8_all_formats::Threads workers(1);
 auto primitive=qwen4exp::createCPUQwen4ExpKernels();
 for(int width:{1,7,8,15,16,17,31,32,33,255,256,257,320}) {
  std::vector<float> raw(std::size_t(3)*width),actual(raw.size()),expected(raw.size());
  for(std::size_t i=0;i<raw.size();++i)raw[i]=std::array{-0.f,0.f,1.e-38f,-1.e-38f,.00000001f,-.00000001f,3.9999998f,-4.0000005f}[i%8];
  for(int live:{3,1,0}) {
   const qwen4exp::RowExtent rows{3,&live};
   std::fill(actual.begin(),actual.end(),-12345.f);expected=actual;
   primitive->residualDown({rows,width,raw.data(),actual.data()},nullptr);
   hc103_original::residualDown({rows,width,raw.data(),expected.data()},nullptr);
   ASSERT_EQ(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),0);
   actual=raw;expected=raw;
   primitive->residualDown({rows,width,actual.data(),actual.data()},nullptr);
   hc103_original::residualDown({rows,width,expected.data(),expected.data()},nullptr);
   ASSERT_EQ(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),0);
  }
  std::vector<float> normalized(std::size_t(3)*4*width),logits(normalized.size()),read(std::size_t(3)*width),gold(read.size());
  for(std::size_t i=0;i<logits.size();++i){logits[i]=raw[i%raw.size()];normalized[i]=float(int(i%29)-14)*.03125f;}
  for(int live:{3,1,0}) {
   const qwen4exp::RowExtent rows{3,&live};std::fill(read.begin(),read.end(),-12345.f);gold=read;
   primitive->residualRead({rows,width,normalized.data(),logits.data(),read.data()},nullptr);
   hc103_original::residualRead({rows,width,normalized.data(),logits.data(),gold.data()},nullptr);
   ASSERT_EQ(std::memcmp(read.data(),gold.data(),read.size()*sizeof(float)),0);
  }
 }
}
} // namespace
/** @test Freeze ISA-specific channel tails and actual TwoTerm finite/half-overflow admission. */
TEST(Q8ResidualAllFormatsCPU, HC103OriginalPrimitiveTailsAndPublicationAdmission) {
 using namespace llaminar2;
 using namespace llaminar2::test;
 hc103OriginalPrimitiveTails();ASSERT_FALSE(::testing::Test::HasFailure());
 q8_all_formats::Threads workers(1);
 const auto &formats=quantizedVerifierFormats();
 const auto it=std::find_if(formats.begin(),formats.end(),[](const auto &f){return f.tensor_type==TensorType::Q8_0;});
 ASSERT_NE(it,formats.end());q8_all_formats::CPUFixture up(*it,3,256,256,false);
 CPUDeviceContext context(DeviceId::cpu(),1);std::vector<Q8_1Block> p(24),r(24);
 for(float invalid:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),1.e20f}) {
  std::fill_n(up.input.mutable_data(),up.input.numel(),0.f);up.input.mutable_data()[0]=invalid;
  std::fill_n(up.output.mutable_data(),up.output.numel(),-12345.f);
  EXPECT_THROW(up.stage->execute(&context),std::invalid_argument);
  EXPECT_THROW(hc103_original::publish(up.input.data(),3,256,p.data(),r.data()),std::invalid_argument);
  for(std::size_t i=0;i<up.output.numel();++i)ASSERT_EQ(up.output.data()[i],-12345.f);
 }
}

/** @test Explicit scalar primitive policy retains every original tail/live/alias byte. */
TEST(Q8ResidualAllFormatsCPU, DISABLED_HC103OriginalScalarPrimitives) {
 ASSERT_EQ(llaminar2::activeISALevel(),llaminar2::ISALevel::Scalar);
 hc103OriginalPrimitiveTails();
}
/** @test Scalar is deliberately outside original NativeVNNI prepared-workspace geometry. */
TEST(Q8ResidualAllFormatsCPU, DISABLED_HC103ScalarRejectsNativePreparedGeometry) {
 using namespace llaminar2;
 using namespace llaminar2::test;
 ASSERT_EQ(activeISALevel(),ISALevel::Scalar);
 EXPECT_EQ(CPUExecutionGeometry::local().maximum_native_row_tile,0u);
 EXPECT_FALSE(CPUExecutionGeometry::local().isValid());
 const auto &formats=quantizedVerifierFormats();
 const auto it=std::find_if(formats.begin(),formats.end(),[](const auto &f){return f.tensor_type==TensorType::Q8_0;});
 ASSERT_NE(it,formats.end());
 try {
  q8_all_formats::CPUFixture unsupported(*it,3,256,256,false);
  FAIL()<<"Scalar must not fabricate NativeVNNI workspace execution geometry";
 } catch(const std::invalid_argument &error) {
  EXPECT_EQ(std::string(error.what()),"CPU projection workspace requires participant-owned worker/cache/ISA geometry");
 }
}
