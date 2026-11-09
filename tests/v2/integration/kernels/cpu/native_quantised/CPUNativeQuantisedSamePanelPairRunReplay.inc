/**
 * @file CPUNativeQuantisedSamePanelPairRunReplay.inc
 * @brief Actual paired kernels retain serial bytes under bounded same-panel ownership.
 *
 * These authored inputs reuse the production fused entrypoint and native source
 * preparation. Mixed row counts, live descriptor prefixes, output guards and
 * retained teams prove completion and coverage; the serial row kernel alone
 * supplies the arithmetic oracle. No copied decoder, timing winner or recorded
 * model-route claim is introduced. All source formats use legal native K512;
 * actual QFN Q4_K gate/up and Q5_1 down additionally retain their real shard K.
 */
#include <string_view>

namespace llaminar2::test {
namespace {
/**
 * @brief Replay one legal pair of prepared source matrices across mixed expert rows.
 * @param left First canonical source format, never inferred from a codebook alias.
 * @param right Adjacent source format or the same actual model format.
 * @param n0 First physical output width, including a possible N64 tail.
 * @param n1 Second physical output width.
 * @param k Actual legal native source width shared by both projections.
 * @param maximum_rows Retained row capacity; each descriptor publishes its own live M.
 * @param worker_counts Existing-team widths to exercise without changing arithmetic.
 */
void proveSamePanelPairRun(const QuantizedVerifierFormatCase &left,
                          const QuantizedVerifierFormatCase &right,
                          int n0, int n1, int k, int maximum_rows,
                          std::span<const int> worker_counts)
{
    using namespace cpu::native_quantised;
    constexpr float guard = -12345.0f;
    constexpr auto policy = CPUProjectionNumericalPolicy::GPUAlignedExpert;
    FusedCompletionScope restore(1, std::nullopt);
    requireQuantizedVerifierSourceShape(left, n0, k);
    requireQuantizedVerifierSourceShape(right, n1, k);
    auto source0 = left.create({size_t(n0),size_t(k)}, 2707);
    auto source1 = right.create({size_t(n1),size_t(k)}, 3709);
    CPUNativeQuantisedGemmKernel kernel0(source0.get(),0,-1,policy), kernel1(source1.get(),0,-1,policy);
    ASSERT_TRUE(kernel0.isValid() && kernel1.isValid());
    const std::array<const CPUNativeQuantisedPackedWeights *,2> packed{&kernel0.packedWeights(),&kernel1.packedWeights()};
    const std::array<int,2> widths{n0,n1};
    const int blocks=packed[0]->blocks_per_row;
    std::vector<float> activations(size_t(maximum_rows)*k);
    for(size_t i=0;i<activations.size();++i)activations[i]=float(int((i*37u+13u)%257u)-128)*0.003125f;
    std::vector<Q8_1Block> input(size_t(maximum_rows)*blocks);
    quantize_activations_to_q8_1(activations.data(),input.data(),maximum_rows,k,blocks,policy);
    std::array<std::vector<float>,2> reference,bias;
    for(int member=0;member<2;++member) {
        reference[member].assign(size_t(maximum_rows)*(widths[member]+7)+17u,guard);
        bias[member].resize(widths[member]);
        for(int column=0;column<widths[member];++column)bias[member][column]=float(column%19-9)*0.015625f;
        NativeQuantisedTestPartialStorage serial(*packed[member],1);
        for(int row=0;row<maximum_rows;++row)
            gemv_native_quantised_preq(*packed[member],input.data()+size_t(row)*blocks,
                reference[member].data()+size_t(row)*(widths[member]+7),serial.span());
    }
    std::array<std::vector<float>,4> output,expected;
    for(int member=0;member<4;++member)
        output[member].assign(reference[member%2].size(),guard);
    for(int workers:worker_counts) {
        omp_set_num_threads(workers);
        SCOPED_TRACE("workers="+std::to_string(workers));
        // The full case contains an actual singleton and a long odd/even expert.
        // The short case exercises the four-wave guard; live1 also retains the
        // original small-grid/SharedBank path. Allocation capacity never changes.
        // The actual gate shard additionally crosses the 1/2/4 ownership
        // thresholds at 28 workers; this is authored geometry, not a timed claim.
        const bool gate_thresholds = workers==28 && n0==320 && k==2560;
        const std::array<int,3> threshold_rows{32,128,256};
        for(int replay=0;replay<(gate_thresholds?9:6);++replay) {
            const bool full=replay==0 || replay==3;
            const bool threshold = replay>=6;
            const int live=threshold || replay==2?1:4;
            const std::array<int,4> rows=threshold
                ?std::array<int,4>{threshold_rows[size_t(replay-6)],1,1,1}
                :full?std::array<int,4>{1,maximum_rows,3,9}:std::array<int,4>{1,3,2,1};
            SCOPED_TRACE("replay="+std::to_string(replay)+" first M="+std::to_string(rows[0])+" live="+std::to_string(live));
            const auto purpose=replay==4?ProjectionRowsPurpose::MTPVerifier:
                replay==5?ProjectionRowsPurpose::Decode:ProjectionRowsPurpose::Prefill;
            std::array<FusedVerifierRowsDesc,5> descriptors;
            for(auto &descriptor:descriptors)descriptor={.packed=nullptr,.N=-11,.rows=-13};
            for(int member=0;member<4;++member) {
                std::fill(output[member].begin(),output[member].end(),guard);
                expected[member]=output[member];
                if(member>=live)continue;
                const int matrix=member%2,stride=widths[matrix]+7;
                const bool biased=member==1;
                for(int row=0;row<rows[member];++row)for(int column=0;column<widths[matrix];++column) {
                    float value=reference[matrix][size_t(row)*stride+column];
                    if(biased)value+=bias[matrix][column];
                    expected[member][size_t(row)*stride+column]=value;
                }
                descriptors[member]={.packed=packed[matrix],.output=output[member].data(),
                    .bias=biased?bias[matrix].data():nullptr,.N=widths[matrix],.ldc=stride,
                    .rows=rows[member],.input=input.data(),
                    .verifier_schedule=rows[member]>1?VerifierRowsPolicy::Pairwise:VerifierRowsPolicy::Auto,
                    .purpose=purpose};
            }
            const auto plan=planNativeQuantisedFusedRows({descriptors.data(),size_t(live)},1,workers);
            if(full || threshold)ASSERT_EQ(plan.partial_route,NativeQuantisedFusedPartialRoute::OutputTileLocal);
            if(threshold) {
                // Establish the actual geometry, then exercise the same
                // internal pure ownership selector used by production.
                const auto pairs=size_t((rows[0]+1)/2),columns=size_t(plan.projections[0].n_chunks);
                const auto two_jobs=((pairs+1u)/2u)*columns,four_jobs=((pairs+3u)/4u)*columns;
                const std::array<size_t,3> expected_two{40u,160u,320u},expected_four{20u,80u,160u};
                EXPECT_EQ(two_jobs,expected_two[size_t(replay-6)]);
                EXPECT_EQ(four_jobs,expected_four[size_t(replay-6)]);
                const auto expected_cap = replay==6?1u:replay==7?2u:4u;
                EXPECT_EQ(cpu::native_quantised::detail::selectSamePanelPrefillPairRunWidth(true,two_jobs,four_jobs,workers),expected_cap);
                SCOPED_TRACE("authored threshold M="+std::to_string(rows[0])+" expected maximum pairs="+std::to_string(expected_cap));
            }
            for(int member=0;member<live;++member) {
                ASSERT_EQ(descriptors[member].packed->numerical_policy,policy);
                if(rows[member]>1)ASSERT_EQ(plan.projections[member].effective_verifier_schedule,VerifierRowsPolicy::Pairwise);
            }
            NativeQuantisedTestPartialStorage partials(plan.sharedPartialFloats());
            const auto invoke=[&]() {
                return gemm_native_quantised_fused_verifier_rows_preq(nullptr,descriptors.data(),partials.span(),live,1,blocks);
            };
            const auto complete=[&]() {
                for(int member=0;member<4;++member)
                    if(std::memcmp(expected[member].data(),output[member].data(),expected[member].size()*sizeof(float)))return false;
                return true;
            };
            ASSERT_TRUE(invoke());ASSERT_TRUE(complete());
            if(full || threshold) {
                for(auto &values:output)std::fill(values.begin(),values.end(),guard);
                std::vector<int> accepted(size_t(workers),0);
#pragma omp parallel num_threads(workers)
                { accepted[size_t(omp_get_thread_num())]=invoke() && complete(); }
                for(int ok:accepted)ASSERT_EQ(ok,1);
            }
        }
    }
}

/** @test Every admitted source format preserves existing pair/tail bytes and bounded job coverage. */
TEST(CPUNativeQuantisedFusedCompletion, GPUAlignedPrefillSamePanelPairRunReplay)
{
    ::testing::Test::RecordProperty("input_scope","authored_byte_lifecycle_not_recorded_model_routes");
    using cpu::native_quantised::detail::selectSamePanelPrefillPairRunWidth;
    // Directly test the production scalar decision, not a copied selector.
    // The actual job decoder/arithmetic is exercised by the native replays below.
    for(int workers=1;workers<=64;++workers) {
        const auto floor=size_t(workers)*4u;
        EXPECT_EQ(selectSamePanelPrefillPairRunWidth(false,floor,floor,workers),1u);
        EXPECT_EQ(selectSamePanelPrefillPairRunWidth(true,floor-1u,floor-1u,workers),1u);
        EXPECT_EQ(selectSamePanelPrefillPairRunWidth(true,floor,floor-1u,workers),2u);
        EXPECT_EQ(selectSamePanelPrefillPairRunWidth(true,floor,floor,workers),4u);
    }
    EXPECT_EQ(selectSamePanelPrefillPairRunWidth(true,160u,80u,28),2u);
    EXPECT_THROW(static_cast<void>(selectSamePanelPrefillPairRunWidth(true,160u,80u,0)),std::invalid_argument);
    EXPECT_THROW(static_cast<void>(selectSamePanelPrefillPairRunWidth(true,80u,160u,28)),std::invalid_argument);
    const auto &formats=quantizedVerifierFormats();
    const std::array<int,3> workers{1,3,7};
    for(size_t format=0;format<formats.size();++format) {
        SCOPED_TRACE(formats[format].label);
        proveSamePanelPairRun(formats[format],formats[(format+1u)%formats.size()],65,129,512,64,workers);
    }
    const std::array<int,2> socket_workers{1,28};
    bool gate_source_seen=false,down_source_seen=false;
    for(const auto &format:formats) {
        if(std::string_view(format.label)=="Q4_K") {
            gate_source_seen=true;
            proveSamePanelPairRun(format,format,320,321,2560,448,socket_workers);
        }
        if(std::string_view(format.label)=="Q5_1") {
            down_source_seen=true;
            proveSamePanelPairRun(format,format,2560,2561,320,448,socket_workers);
        }
    }
    ASSERT_TRUE(gate_source_seen && down_source_seen);
}
} // namespace
} // namespace llaminar2::test
