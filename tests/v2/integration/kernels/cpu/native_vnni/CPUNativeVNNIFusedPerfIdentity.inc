/**
 * @file CPUNativeVNNIFusedPerfIdentity.inc
 * @brief Retain meaningful native projection evidence independently of descriptor bundle positions.
 *
 * Real prepared CPU weights, original fused execution and serial-row output
 * bytes certify this instrumentation economy regression. The collector is
 * neither replaced nor capped: every actual projection must contribute once,
 * while unchanged geometry/policy observations share their original key.
 */
#include "utils/PerfStatsCollector.h"
#include <map>
#include <set>

namespace llaminar2::test {
namespace {
constexpr const char* kProjectionLaunchFamily =
    "kernel.cpu_native_vnni_fused_verifier_rows_projection_launch";

/** @brief Restore the caller's collection policy after three explicit test measurement lifetimes. */
class FusedProjectionCollectionScope final {
public:
    /** @brief Enable only the original family under test; no production collector policy is changed. */
    FusedProjectionCollectionScope() {
        if(const char* value=std::getenv("LLAMINAR_PERF_STATS_JSON")) prior_json_=value;
        if(const char* value=std::getenv("LLAMINAR_PERF_STATS_FILTER")) prior_filter_=value;
        setenv("LLAMINAR_PERF_STATS_JSON","1",1);
        setenv("LLAMINAR_PERF_STATS_FILTER",kProjectionLaunchFamily,1);
        PerfStatsCollector::reset();
    }
    /** @brief Drop only this test's observations and restore the original immutable-policy source. */
    ~FusedProjectionCollectionScope() {
        if(prior_json_)setenv("LLAMINAR_PERF_STATS_JSON",prior_json_->c_str(),1);
        else unsetenv("LLAMINAR_PERF_STATS_JSON");
        if(prior_filter_)setenv("LLAMINAR_PERF_STATS_FILTER",prior_filter_->c_str(),1);
        else unsetenv("LLAMINAR_PERF_STATS_FILTER");
        PerfStatsCollector::reset();
    }
    FusedProjectionCollectionScope(const FusedProjectionCollectionScope&)=delete;
    FusedProjectionCollectionScope& operator=(const FusedProjectionCollectionScope&)=delete;
private:
    std::optional<std::string> prior_json_,prior_filter_;
};

/** @test Transient bundle ordinals cannot multiply otherwise identical native projection records. */
TEST(CPUNativeVNNIFusedCompletion, ProjectionEvidenceUsesGeometryAndPolicyInsteadOfBundleOrdinal) {
    constexpr int rows=3,k=256,workers=2,max_bundle=32;
    constexpr float guard=-12345.0f;
    const std::array<int,2> widths{65,97};
    const size_t output_floats=size_t(rows)*(widths[1]+7);
    FusedCompletionScope runtime(workers,1);
    FusedProjectionCollectionScope collection;
    const auto& formats=quantizedVerifierFormats();
    ASSERT_GE(formats.size(),2u);
    ASSERT_STREQ(formats[0].label,"Q4_0");
    ASSERT_STREQ(formats[1].label,"IQ4_NL");
    auto weights0=formats[0].create({size_t(widths[0]),size_t(k)},431);
    auto weights1=formats[1].create({size_t(widths[1]),size_t(k)},659);
    ASSERT_NE(weights0,nullptr);
    ASSERT_NE(weights1,nullptr);
    CPUNativeVNNIGemmKernel kernel0(weights0.get()),kernel1(weights1.get());
    ASSERT_TRUE(kernel0.isValid());
    ASSERT_TRUE(kernel1.isValid());
    const std::array<const CPUNativeVNNIPackedWeights*,2> packed{
        &kernel0.packedWeights(),&kernel1.packedWeights()};
    ASSERT_NE(packed[0]->codebook_id,packed[1]->codebook_id);
    ASSERT_EQ(packed[0]->blocks_per_row,packed[1]->blocks_per_row);
    std::vector<float> source(size_t(rows)*k);
    for(size_t value=0;value<source.size();++value)
        source[value]=float(int((value*37+13)%257)-128)*0.003125f;
    std::vector<Q8_1Block> input(size_t(rows)*packed[0]->blocks_per_row);
    quantize_activations_to_q8_1(source.data(),input.data(),rows,k,packed[0]->blocks_per_row);
    std::array<std::vector<float>,2> references{
        std::vector<float>(output_floats,guard),std::vector<float>(output_floats,guard)};
    for(int semantic=0;semantic<2;++semantic) {
        NativeVNNITestPartialStorage serial(*packed[semantic],1);
        for(int row=0;row<rows;++row)
            gemv_native_vnni_preq(*packed[semantic],input.data()+size_t(row)*packed[semantic]->blocks_per_row,
                references[semantic].data()+size_t(row)*(widths[semantic]+7),serial.span());
    }
    std::array<std::vector<float>,max_bundle> outputs;
    for(auto& output:outputs)output.resize(output_floats,guard);
    std::vector<FusedVerifierRowsDesc> descriptors;
    std::vector<int> semantics;
    /** @brief Execute original fused work without output aliasing.
     * @param count Actual live descriptor count, never spare bundle capacity.
     * @param rotation Which meaningful geometry occupies the first transient position. */
    const auto invoke=[&](int count,int rotation) {
        descriptors.clear();semantics.clear();
        for(int slot=0;slot<count;++slot) {
            const int semantic=(slot+rotation)%2;
            semantics.push_back(semantic);
            std::fill(outputs[slot].begin(),outputs[slot].end(),guard);
            descriptors.push_back(FusedVerifierRowsDesc{.packed=packed[semantic],.output=outputs[slot].data(),
                .N=widths[semantic],.ldc=widths[semantic]+7,.rows=rows,
                .verifier_schedule=VerifierRowsPolicy::FullKTwoRowPairGridNbc1});
        }
        const auto plan=planNativeVNNIFusedRows(descriptors,rows,workers,ISAPath::AUTO);
        ASSERT_EQ(plan.partial_route,NativeVNNIFusedPartialRoute::FullK);
        NativeVNNITestPartialStorage partials(plan.sharedPartialFloats());
        ASSERT_TRUE(gemm_native_vnni_fused_verifier_rows_preq(input.data(),descriptors.data(),partials.span(),
            count,rows,packed[0]->blocks_per_row));
        for(int slot=0;slot<count;++slot)
            ASSERT_EQ(std::memcmp(outputs[slot].data(),references[semantics[slot]].data(),
                output_floats*sizeof(float)),0) << "slot=" << slot;
    };
    // The semantic identities below are deliberately independent of bundle
    // ordinal. Values come from real single-member production observations;
    // this oracle never duplicates the recorder's emission or policy formulas.
    const std::set<std::string> meaningful_keys{
        "m","n","policy_n","k","codebook","purpose","isa","k_tiles","k_tile_blocks",
        "ambient_n_block_chunks","n_block_chunks","requested_decode_policy","effective_decode_policy",
        "requested_verifier_policy","effective_verifier_policy","physical_row_tile","route","task_grid",
        "producer_tasks","reduction_tasks","bundle_scheduler"};
    invoke(1,0);invoke(1,1);
    const auto baseline=PerfStatsCollector::snapshot({kProjectionLaunchFamily});
    ASSERT_EQ(baseline.size(),2u);
    std::array<PerfStatsCollector::Tags,2> signatures;
    for(const auto& record:baseline) {
        const int semantic=record.tags.at("n")==std::to_string(widths[0])?0:1;
        EXPECT_EQ(record.tags.at("n"),std::to_string(widths[semantic]));
        EXPECT_EQ(record.tags.at("codebook"),std::to_string(packed[semantic]->codebook_id));
        EXPECT_EQ(record.tags.at("m"),std::to_string(rows));
        EXPECT_EQ(record.tags.at("k"),std::to_string(k));
        EXPECT_EQ(record.tags.at("purpose"),"decode");
        for(const auto& key:meaningful_keys) {
            ASSERT_TRUE(record.tags.contains(key)) << key;
            EXPECT_FALSE(record.tags.at(key).empty()) << key;
            signatures[semantic].emplace(key,record.tags.at(key));
        }
        EXPECT_EQ(record.tags.size(),meaningful_keys.size());
        EXPECT_EQ(record.count,1u);
        EXPECT_DOUBLE_EQ(record.value,1.0);
    }
    ASSERT_EQ(signatures[0].size(),21u);
    ASSERT_EQ(signatures[1].size(),21u);
    for(int lifetime=0;lifetime<3;++lifetime) {
        SCOPED_TRACE(lifetime);
        PerfStatsCollector::reset();
        std::array<uint64_t,2> expected_counts{};
        for(int count:{1,2,4,8,16,max_bundle})for(int rotation:{0,1}) {
            SCOPED_TRACE(count);
            SCOPED_TRACE(rotation);
            invoke(count,rotation);
            for(const int semantic:semantics)++expected_counts[semantic];
            const auto records=PerfStatsCollector::snapshot({kProjectionLaunchFamily});
            const size_t meaningful_count=size_t(expected_counts[0]!=0)+size_t(expected_counts[1]!=0);
            ASSERT_EQ(records.size(),meaningful_count) << "Distinct records must not grow with descriptor positions";
            uint64_t observations=0;
            double increments=0;
            std::array<uint64_t,2> semantic_counts{};
            for(const auto& record:records) {
                EXPECT_EQ(record.kind,PerfStatRecord::Kind::Counter);
                EXPECT_EQ(record.domain,"kernel");
                EXPECT_EQ(record.phase,"gemm");
                EXPECT_EQ(record.device,"cpu");
                const int semantic=record.tags.at("n")==std::to_string(widths[0])?0:1;
                EXPECT_EQ(record.tags,signatures[semantic]);
                EXPECT_FALSE(record.tags.contains("projection"));
                observations+=record.count;increments+=record.value;semantic_counts[semantic]+=record.count;
            }
            EXPECT_EQ(semantic_counts,expected_counts);
            EXPECT_EQ(observations,expected_counts[0]+expected_counts[1]);
            EXPECT_DOUBLE_EQ(increments,double(expected_counts[0]+expected_counts[1]));
        }
        EXPECT_EQ(expected_counts,(std::array<uint64_t,2>{63,63}));
        EXPECT_EQ(PerfStatsCollector::snapshot({kProjectionLaunchFamily}).size(),2u);
    }

    // Identical prepared weights can legitimately produce different semantic
    // records. Establish each complete key independently, then mix/reorder
    // those real descriptors to check purpose, policy and row identity too.
    struct SemanticCase {
        int rows;
        int policy_n;
        ProjectionRowsPurpose purpose;
        DecodeSchedulePolicy decode;
        VerifierRowsPolicy verifier;
    };
    const std::array<SemanticCase,7> cases{{
        {3,65,ProjectionRowsPurpose::Decode,DecodeSchedulePolicy::Auto,VerifierRowsPolicy::FullKTwoRowPairGridNbc1},
        {3,65,ProjectionRowsPurpose::Prefill,DecodeSchedulePolicy::Auto,VerifierRowsPolicy::FullKTwoRowPairGridNbc1},
        {2,65,ProjectionRowsPurpose::Decode,DecodeSchedulePolicy::Auto,VerifierRowsPolicy::FullKTwoRowPairGridNbc1},
        {3,65,ProjectionRowsPurpose::Decode,DecodeSchedulePolicy::Auto,VerifierRowsPolicy::Pairwise},
        {3,97,ProjectionRowsPurpose::Decode,DecodeSchedulePolicy::Auto,VerifierRowsPolicy::FullKTwoRowPairGridNbc1},
        {1,65,ProjectionRowsPurpose::Decode,DecodeSchedulePolicy::Auto,VerifierRowsPolicy::Auto},
        {1,65,ProjectionRowsPurpose::Decode,DecodeSchedulePolicy::Nbc2,VerifierRowsPolicy::Auto}}};
    std::array<std::vector<float>,cases.size()> case_references;
    for(size_t semantic=0;semantic<cases.size();++semantic) {
        auto& reference=case_references[semantic];reference.assign(output_floats,guard);
        NativeVNNITestPartialStorage serial(*packed[0],1);
        for(int row=0;row<cases[semantic].rows;++row)
            gemv_native_vnni_preq(*packed[0],input.data()+size_t(row)*packed[0]->blocks_per_row,
                reference.data()+size_t(row)*(widths[0]+7),serial.span());
    }
    /** @brief Execute only live case descriptors through the real fused scheduler.
     * @param live Case identities in the requested transient descriptor order. */
    const auto invoke_cases=[&](const std::vector<size_t>& live) {
        ASSERT_LE(live.size(),outputs.size());
        descriptors.clear();
        for(size_t slot=0;slot<live.size();++slot) {
            const auto& item=cases.at(live[slot]);
            std::fill(outputs[slot].begin(),outputs[slot].end(),guard);
            descriptors.push_back(FusedVerifierRowsDesc{.packed=packed[0],.output=outputs[slot].data(),
                .N=widths[0],.ldc=widths[0]+7,.rows=item.rows,.decode_schedule=item.decode,
                .verifier_schedule=item.verifier,.purpose=item.purpose,.policy_n=item.policy_n});
        }
        const auto plan=planNativeVNNIFusedRows(descriptors,rows,workers,ISAPath::AUTO);
        ASSERT_EQ(plan.partial_route,NativeVNNIFusedPartialRoute::FullK);
        NativeVNNITestPartialStorage partials(plan.sharedPartialFloats());
        ASSERT_TRUE(gemm_native_vnni_fused_verifier_rows_preq(input.data(),descriptors.data(),partials.span(),
            static_cast<int>(live.size()),rows,packed[0]->blocks_per_row));
        for(size_t slot=0;slot<live.size();++slot)
            ASSERT_EQ(std::memcmp(outputs[slot].data(),case_references[live[slot]].data(),
                output_floats*sizeof(float)),0) << "case=" << live[slot];
    };
    std::array<PerfStatsCollector::Tags,cases.size()> case_signatures;
    std::set<PerfStatsCollector::Tags> unique_signatures;
    for(size_t semantic=0;semantic<cases.size();++semantic) {
        PerfStatsCollector::reset();invoke_cases({semantic});
        const auto records=PerfStatsCollector::snapshot({kProjectionLaunchFamily});
        ASSERT_EQ(records.size(),1u);
        ASSERT_EQ(records[0].tags.size(),meaningful_keys.size());
        EXPECT_EQ(records[0].count,1u);EXPECT_DOUBLE_EQ(records[0].value,1.0);
        case_signatures[semantic]=records[0].tags;unique_signatures.insert(records[0].tags);
    }
    ASSERT_EQ(unique_signatures.size(),cases.size());
    for(int lifetime=0;lifetime<3;++lifetime) {
        PerfStatsCollector::reset();std::array<uint64_t,cases.size()> expected{};
        const std::array<std::vector<size_t>,4> trace{{
            {0,1,2,3,4,5,6,6,5,4,3,2,1,0},{4,0,6},{1},
            {6,0,5,1,4,2,3,3,2,4,1,5,0,6}}};
        for(const auto& live:trace) {
            invoke_cases(live);for(size_t semantic:live)++expected[semantic];
            const auto records=PerfStatsCollector::snapshot({kProjectionLaunchFamily});
            ASSERT_EQ(records.size(),cases.size());
            std::array<uint64_t,cases.size()> observed{};
            for(const auto& record:records) {
                const auto found=std::find(case_signatures.begin(),case_signatures.end(),record.tags);
                ASSERT_NE(found,case_signatures.end());
                const auto semantic=static_cast<size_t>(found-case_signatures.begin());
                observed[semantic]+=record.count;
                EXPECT_DOUBLE_EQ(record.value,static_cast<double>(record.count));
            }
            EXPECT_EQ(observed,expected);
        }
    }
}
} // namespace
} // namespace llaminar2::test
