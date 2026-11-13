/**
 * @file CPUNativeQuantisedFusedLivePrefixLifecycle.inc
 * @brief Actual batched projections retain only the current live descriptor prefix.
 *
 * Authored native lifecycle inputs reuse prepared matrices and output capacity
 * across large, small and reset invocations. Invalid inactive descriptors and
 * poisoned inactive output rows may never enter planning, LUT access or work.
 * The public serial kernel remains the independent byte oracle; this fixture
 * implements no alternative dot arithmetic and makes no timing assertion.
 * The private live plan additionally matches every public value-plan field,
 * preserves public default/copy semantics and leaves inactive raw backing poisoned.
 */
#include <utility>
#include <tuple>

namespace llaminar2::test
{
namespace
{
/** @return Equality of every observable field, without comparing unspecified object padding. */
bool sameFusedProjectionPlan(const cpu::native_quantised::FusedVerifierRowsPlan& a,
    const cpu::native_quantised::FusedVerifierRowsPlan& b)
{
    const auto fields=[](const auto& p) {
        const auto& v=p.verifier_resolution;
        return std::tie(p.rows,p.policy_n,p.n_chunks,p.n_block_chunks,p.total_blocks,p.k_tiles,
            p.k_blocks_per_tile,p.partial_sums_offset,p.partial_sums_size,
            p.requested_decode_schedule,p.effective_decode_schedule,p.requested_verifier_schedule,
            p.effective_verifier_schedule,p.execution_exact_hit,v.requested,v.effective,v.route,
            v.task_grid,v.physical_row_tile,v.ambient_n_block_chunks,v.n_block_chunks,v.n_chunks,
            v.n_blocks,v.producer_tasks,v.reduction_tasks);
    };
    return fields(a)==fields(b);
}

/** @return Equality of the public plan's original scalar result and bank admission. */
template<class A,class B>
bool sameFusedPlanSummary(const A& a,const B& b)
{
    return a.partial_route==b.partial_route && a.partial_floats==b.partial_floats &&
        a.worker_partial_stride==b.worker_partial_stride && a.use_avx512==b.use_avx512 &&
        a.use_avx2==b.use_avx2 && a.all_k_partitioned==b.all_k_partitioned &&
        a.execution_exact_class==b.execution_exact_class && a.sharedPartialFloats()==b.sharedPartialFloats();
}

/**
 * @brief Replay public value and private prefix plans under the actual caller's team identity.
 * @param descriptors Original prepared matrices/output bindings, bounded to the live prefix.
 * @param rows Original inherited row count.
 * @param workers Actual positive workers for this invocation, not a pre-team guess.
 * @return True only if all plans/copies/defaults agree and inactive raw bytes remain untouched.
 */
bool proveLivePlanValue(std::span<const cpu::native_quantised::FusedVerifierRowsDesc> descriptors,
    int rows,int workers)
{
    using namespace cpu::native_quantised;
    constexpr auto poison=std::byte{0xa7};
    alignas(FusedVerifierRowsPlan) std::array<std::byte,
        sizeof(FusedVerifierRowsPlan)*NativeQuantisedFusedRowsPlan::capacity> backing;
    std::fill(backing.begin(),backing.end(),poison);
    cpu::native_quantised::detail::NativeQuantisedFusedRowsLiveView live(backing,descriptors.size());
    const auto actual=cpu::native_quantised::detail::fillNativeQuantisedFusedRowsPlans(
        live.mutableProjections(),descriptors,rows,workers);
    const auto expected=planNativeQuantisedFusedRows(descriptors,rows,workers);
    const NativeQuantisedFusedRowsPlan copied(expected);
    NativeQuantisedFusedRowsPlan assigned;
    assigned=expected;
    if(!sameFusedPlanSummary(actual,expected) || !sameFusedPlanSummary(copied,expected) ||
       !sameFusedPlanSummary(assigned,expected) || live.projections().size()!=descriptors.size())return false;
    const FusedVerifierRowsPlan defaults;
    for(size_t index=0;index<NativeQuantisedFusedRowsPlan::capacity;++index)
    {
        if(!sameFusedProjectionPlan(expected.projections[index],copied.projections[index]) ||
           !sameFusedProjectionPlan(expected.projections[index],assigned.projections[index]))return false;
        if(index<descriptors.size())
        {
            if(!sameFusedProjectionPlan(live.projections()[index],expected.projections[index]))return false;
        }
        else if(!sameFusedProjectionPlan(expected.projections[index],defaults))return false;
    }
    return std::all_of(backing.begin()+descriptors.size()*sizeof(FusedVerifierRowsPlan),backing.end(),
        [&](std::byte value){return value==poison;});
}

/** @brief Check all-format native prefix reuse in standalone and existing teams. */
void proveLiveFusedPrefixLifecycle()
{
    using namespace cpu::native_quantised;
    using Kernel = CPUNativeQuantisedGemmKernel;
    using Desc = Kernel::BatchedPrequantizedProjectionDesc;
    constexpr int projections = 37, max_rows = 7, n = 80, k = 1024, stride = n + 7;
    constexpr int blocks = k / Q8_1Block::BLOCK_SIZE;
    constexpr float guard = -12345.0f;
    constexpr auto policy = CPUProjectionNumericalPolicy::GPUAlignedExpert;
    FusedCompletionScope restore(/*workers=*/1, std::nullopt);
    const auto &formats = quantizedVerifierFormats();
    std::vector<float> source(static_cast<size_t>(projections) * max_rows * k);
    for (size_t index = 0; index < source.size(); ++index)
        source[index] = float(int((index * 37 + 13) % 257) - 128) * 0.003125f;
    std::vector<Q8_1Block> q8(static_cast<size_t>(projections) * max_rows * blocks);
    quantize_activations_to_q8_1(source.data(), q8.data(), projections * max_rows, k, blocks, policy);

    for (size_t format = 0; format < formats.size(); ++format)
    {
        SCOPED_TRACE(formats[format].label);
        omp_set_num_threads(1);
        auto weights0 = formats[format].create({size_t(n), size_t(k)}, 9109 + format);
        auto weights1 = formats[(format + 1) % formats.size()].create({size_t(n), size_t(k)}, 9127 + format);
        Kernel kernel0(weights0.get(), 0, -1, policy), kernel1(weights1.get(), 0, -1, policy);
        ASSERT_TRUE(kernel0.isValid() && kernel1.isValid());
        std::array<Kernel *, 2> kernels{&kernel0, &kernel1};
        std::array<std::vector<float>, projections> outputs, serial;
        for (int projection = 0; projection < projections; ++projection)
        {
            outputs[projection].assign(size_t(max_rows) * stride + 17u, guard);
            serial[projection] = outputs[projection];
            auto &packed = kernels[projection % 2]->packedWeights();
            NativeQuantisedTestPartialStorage partial(packed, 1);
            for (int row = 0; row < max_rows; ++row)
                gemv_native_quantised_preq(packed,
                    q8.data() + size_t(projection * max_rows + row) * blocks,
                    serial[projection].data() + size_t(row) * stride, partial.span());
        }
        for (int workers : {1, 3, 7})
        {
            omp_set_num_threads(workers);
            // Inactive capacity is deliberately malformed, rather than a
            // second valid vector whose accidental execution could go unseen.
            std::array<Desc, projections + 4> descriptors;
            for (const auto [live, rows] : std::array<std::pair<int, int>, 4>{{
                    {projections, max_rows}, {1, 1}, {projections, 3}, {3, 2}}})
            {
                SCOPED_TRACE("workers=" + std::to_string(workers) + " live=" +
                    std::to_string(live) + " rows=" + std::to_string(rows));
                const auto purpose=rows==max_rows ? ProjectionRowsPurpose::Prefill : ProjectionRowsPurpose::MTPVerifier;
                for (auto &output : outputs)
                    std::fill(output.begin(), output.end(), guard);
                for (size_t projection = 0; projection < descriptors.size(); ++projection)
                {
                    descriptors[projection] = {.kernel = nullptr, .rows = -11, .n = -13};
                    if (projection < static_cast<size_t>(live))
                        descriptors[projection] = {
                            .kernel = kernels[projection % 2],
                            .input_q8 = q8.data() + projection * max_rows * blocks,
                            .output = outputs[projection].data(),
                            .rows = rows, .n = n, .ldc = stride,
                            .purpose = purpose};
                }
                std::array<FusedVerifierRowsDesc,projections> plan_descriptors;
                for(int projection=0;projection<live;++projection)
                    plan_descriptors[projection]={
                        .packed=&kernels[projection%2]->packedWeights(),.output=outputs[projection].data(),
                        .N=n,.ldc=stride,.rows=rows,
                        .input=q8.data()+size_t(projection)*max_rows*blocks,
                        .purpose=purpose};
                ASSERT_TRUE(proveLivePlanValue({plan_descriptors.data(),size_t(live)},rows,workers));
                NativeQuantisedTestPartialStorage partial(
                    NativeQuantisedTestPartialStorage::bundleFloats(descriptors.data(), live, 1));
                const auto complete = [&]()
                {
                    for (int projection = 0; projection < projections; ++projection)
                        for (size_t index = 0; index < outputs[projection].size(); ++index)
                        {
                            const bool active = projection < live && index / stride < static_cast<size_t>(rows)
                                && index % stride < static_cast<size_t>(n);
                            const float expected = active ? serial[projection][index] : guard;
                            if (std::memcmp(&outputs[projection][index], &expected, sizeof(float)) != 0)
                                return false;
                        }
                    return true;
                };
                const auto invoke = [&]()
                {
                    return Kernel::multiply_batched_preq_decode_equivalent(
                        descriptors.data(), live, k, partial.span());
                };
                ASSERT_TRUE(invoke());
                ASSERT_TRUE(complete());
                for (auto &output : outputs)
                    std::fill(output.begin(), output.end(), guard);
                std::array<int, 7> success;
                success.fill(1);
#pragma omp parallel
                {
                    const int worker = omp_get_thread_num();
                    if (!invoke() || !complete()) success[worker] = 0;
                }
                EXPECT_TRUE(std::all_of(success.begin(), success.end(), [](int value) { return value == 1; }));
            }
        }
    }
}
/** @brief Check MTP/decode defaults, inactive backing reuse and actual existing-team planning. */
void proveLivePlanPrefixReset()
{
    using namespace cpu::native_quantised;
    constexpr int maximum=37,rows_max=7,n=80,k=1024;
    FusedCompletionScope restore(1,std::nullopt);
    const auto& formats=quantizedVerifierFormats();
    auto weight=formats.front().create({size_t(n),size_t(k)},19341);
    CPUNativeQuantisedGemmKernel kernel(weight.get(),0,-1,CPUProjectionNumericalPolicy::GPUAlignedExpert);
    ASSERT_TRUE(kernel.isValid());
    std::array<std::array<float,n*rows_max>,maximum> outputs{};
    std::array<FusedVerifierRowsDesc,maximum> descriptors;
    const NativeQuantisedFusedRowsPlan original_default;
    const FusedVerifierRowsPlan projection_default;
    for(const auto& p:original_default.projections)ASSERT_TRUE(sameFusedProjectionPlan(p,projection_default));
    ASSERT_EQ(original_default.partial_route,NativeQuantisedFusedPartialRoute::FullK);
    ASSERT_EQ(original_default.sharedPartialFloats(),0u);
    EXPECT_EQ(original_default.partial_floats,0u);
    EXPECT_EQ(original_default.worker_partial_stride,0u);
    EXPECT_FALSE(original_default.use_avx512);
    EXPECT_FALSE(original_default.use_avx2);
    EXPECT_FALSE(original_default.all_k_partitioned);
    EXPECT_EQ(original_default.execution_exact_class,CPUExecutionExactBundleClass::None);
    alignas(FusedVerifierRowsPlan) std::array<std::byte,
        sizeof(FusedVerifierRowsPlan)*NativeQuantisedFusedRowsPlan::capacity> reusable;
    for(int workers:{1,3,7})
    {
        omp_set_num_threads(workers);
        for(const auto [count,rows]:std::array<std::pair<int,int>,5>{{{maximum,7},{1,1},{maximum,3},{3,2},{1,1}}})
        {
            for(int p=0;p<count;++p)descriptors[p]={.packed=&kernel.packedWeights(),.output=outputs[p].data(),
                .N=n,.ldc=n,.rows=rows,.purpose=ProjectionRowsPurpose::MTPVerifier};
            const auto original=std::span<const FusedVerifierRowsDesc>(descriptors.data(),size_t(count));
            ASSERT_TRUE(proveLivePlanValue(original,rows,workers));
            std::fill(reusable.begin(),reusable.end(),std::byte{0x5d});
            {
                cpu::native_quantised::detail::NativeQuantisedFusedRowsLiveView live(reusable,size_t(count));
                const auto filled=cpu::native_quantised::detail::fillNativeQuantisedFusedRowsPlans(live.mutableProjections(),original,rows,workers);
                EXPECT_TRUE(sameFusedPlanSummary(filled,planNativeQuantisedFusedRows(original,rows,workers)));
                EXPECT_TRUE(std::all_of(reusable.begin()+size_t(count)*sizeof(FusedVerifierRowsPlan),reusable.end(),
                    [](std::byte value){return value==std::byte{0x5d};}));
                EXPECT_THROW(cpu::native_quantised::detail::fillNativeQuantisedFusedRowsPlans(
                    live.mutableProjections().first(size_t(count)-1u),original,rows,workers),std::invalid_argument);
                auto incomplete=descriptors;
                incomplete[size_t(count)-1u].packed=nullptr;
                const auto failed=std::span<const FusedVerifierRowsDesc>(incomplete.data(),size_t(count));
                EXPECT_THROW(cpu::native_quantised::detail::fillNativeQuantisedFusedRowsPlans(
                    live.mutableProjections(),failed,rows,workers),std::invalid_argument);
                EXPECT_THROW(planNativeQuantisedFusedRows(failed,rows,workers),std::invalid_argument);
            }
            std::array<int,7> success;success.fill(1);
#pragma omp parallel
            {
                const int actual_workers=omp_get_num_threads();
                if(!proveLivePlanValue(original,rows,actual_workers))success[omp_get_thread_num()]=0;
            }
            EXPECT_TRUE(std::all_of(success.begin(),success.end(),[](int value){return value==1;}));
        }
    }
    {
        cpu::native_quantised::detail::NativeQuantisedFusedRowsLiveView full(reusable,NativeQuantisedFusedRowsPlan::capacity);
        EXPECT_EQ(full.projections().size(),size_t(NativeQuantisedFusedRowsPlan::capacity));
        for(const auto& projection:full.projections())EXPECT_TRUE(sameFusedProjectionPlan(projection,projection_default));
    }
    const auto invalid_before=reusable;
    EXPECT_THROW((cpu::native_quantised::detail::NativeQuantisedFusedRowsLiveView{reusable,0}),std::invalid_argument);
    EXPECT_THROW((cpu::native_quantised::detail::NativeQuantisedFusedRowsLiveView{reusable,NativeQuantisedFusedRowsPlan::capacity+1u}),std::invalid_argument);
    EXPECT_THROW((cpu::native_quantised::detail::NativeQuantisedFusedRowsLiveView{std::span(reusable).subspan(1),1}),std::invalid_argument);
    EXPECT_TRUE(reusable==invalid_before);
}
} // namespace

/** @test Public value plans and private live prefixes preserve defaults/copy/team identity through reset. */
TEST(CPUNativeQuantisedFusedCompletion,LivePlanFillValueAndPrefixReset)
{
    proveLivePlanPrefixReset();
}

/** @test Live-only metadata/LUT storage ignores invalid inactive capacity after reset. */
TEST(CPUNativeQuantisedFusedCompletion, LivePrefixStorageLargeSmallReset)
{
    proveLiveFusedPrefixLifecycle();
}
} // namespace llaminar2::test
