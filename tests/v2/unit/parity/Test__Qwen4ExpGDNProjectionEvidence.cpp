/**
 * @file Test__Qwen4ExpGDNProjectionEvidence.cpp
 * @brief Device-free exact route export, source validation and reporting-window preservation.
 *
 * The real collector, file exporter and adaptive window boundary consume bounded
 * independently authored schema records. They do not claim native execution.
 */
#include "integration/parity/qwen4exp/Qwen4ExpGDNProjectionEvidence.h"
#include "integration/parity/qwen4exp/Qwen4ExpGPUAdaptiveLookupEvidence.h"
#include "../models/qwen4exp/Qwen4ExpMetadataFixture.h"
#include "utils/DebugEnv.h"
#include <gtest/gtest.h>
#include <chrono>
#include <algorithm>
#include <array>
#include <vector>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>

namespace llaminar2::test::parity::qwen4exp
{
namespace
{
/** @brief Restore actual process collection policy without enabling timing or a backend. */
struct Collection final
{
    const std::array<const char *, 8> names{"LLAMINAR_PERF_STATS_JSON", "LLAMINAR_PERF_STATS_FILTER",
        "LLAMINAR_PROFILE_KERNELS", "LLAMINAR_PROFILING", "LLAMINAR_PERF_STATS_CPU_STAGE_TIMING",
        "LLAMINAR_PERF_STATS_GPU_STAGE_TIMING", "LLAMINAR_GPU_STAGE_TIMING", "LLAMINAR_GPU_STAGE_TIMING_DETAIL"};
    std::array<std::optional<std::string>, 8> previous;
    /** @brief Enable only the exact operation export with all clock overrides absent. */
    Collection()
    {
        for (size_t i=0;i<names.size();++i)
        {
            if (const char *value=std::getenv(names[i])) previous[i]=value;
            ::unsetenv(names[i]);
        }
        ::setenv(names[0],"1",1); ::setenv(names[1],kGDNProjectionRouteFilter,1);
        mutableDebugEnv().reload();
        PerfStatsCollector::reset();
    }
    /** @brief Discard fixture records before restoring the original setup policy. */
    ~Collection()
    {
        PerfStatsCollector::reset();
        for(size_t i=0;i<names.size();++i)
            if(previous[i]) ::setenv(names[i],previous[i]->c_str(),1); else ::unsetenv(names[i]);
        mutableDebugEnv().reload();
        PerfStatsCollector::reloadConfigurationFromEnvironment();
    }
};
/** @return Source metadata only, with the original canonical main QKV tensor inventory. */
GGUFModel source()
{
    auto directory=test::qwen4MetadataFixture(false);
    const Qwen4ExpConfig geometry(directory);
    for(uint32_t layer=0;layer<geometry.mainLayers();++layer)
        if(geometry.mixers()[layer]==Qwen4ExpMixer::GatedDeltaNet)
            directory.tensors.push_back({.name="blk."+std::to_string(layer)+".attn_qkv.weight",
                .dimensions={10240,2560},.type=GGUFTensorType::Q8_0});
    return directory;
}
/** @return The actual producer schema, without manufacturing any source/prepared/native capability. */
PerfStatRecord selected()
{
    return {.kind=PerfStatRecord::Kind::Counter,.domain="kernel",.name="gdn_projection_route",
        .tags={{"route","qkv_two_term_q8"},{"m","9"},{"k","2560"},{"projections","1"},
            {"names","qkv"},{"n","10240"},{"codebooks","19"}},.count=2,.value=2};
}
}

/** @test Exact export retains only the actual operation family and survives the real adaptive window reset. */
TEST(Qwen4ExpGDNProjectionEvidence, ExactFamilyExportsAndSurvivesReportingReset)
{
    Collection collection;
    EXPECT_TRUE(PerfStatsCollector::isDomainEnabled("kernel"));
    EXPECT_FALSE(PerfStatsCollector::cpuStageTimingEnabled());
    EXPECT_FALSE(PerfStatsCollector::gpuStageEventTimingEnabled());
    const auto row=selected();
    PerfStatsCollector::addCounter(row.domain,row.name,1,{}, {},row.tags);
    PerfStatsCollector::addCounter(row.domain,row.name,1,{}, {},row.tags);
    PerfStatsCollector::addCounter("kernel","unrelated_diagnostic",7);
    PerfStatsCollector::addCounter("immutable_lookup","forward_only",3);
    const auto raw=PerfStatsCollector::csvString({kGDNProjectionRouteFilter});
    EXPECT_NE(raw.find("qkv_two_term_q8"),std::string::npos);
    EXPECT_EQ(raw.find("unrelated_diagnostic"),std::string::npos);
    EXPECT_EQ(raw.find("forward_only"),std::string::npos);
    const auto records=PerfStatsCollector::snapshot({kGDNProjectionRouteFilter});
    ASSERT_EQ(records.size(),1u);
    EXPECT_EQ(records[0].count,2u);
    EXPECT_NO_THROW(requireQwen4ExpGDNQKVRouteEvidence(records,source()));
    (void)beginQwen4ExpGPUAdaptiveGenerationEvidenceWindow();
    EXPECT_EQ(PerfStatsCollector::csvString({kGDNProjectionRouteFilter}),raw);
    const auto path=std::filesystem::temp_directory_path()/
        ("llaminar-qfn-route-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".csv");
    struct Remove
    {
        std::filesystem::path path;
        /** @brief Remove only this fixture's completed tiny CSV artifact. */
        ~Remove(){std::error_code ignored;std::filesystem::remove(path,ignored);}
    } remove{path};
    ASSERT_TRUE(PerfStatsCollector::writeCsv(path.string(),{kGDNProjectionRouteFilter}));
    std::ifstream file(path);std::ostringstream exported;exported<<file.rdbuf();
    EXPECT_EQ(exported.str(),raw);
}

/** @test Missing/foreign/malformed selected evidence cannot be replaced by a requested environment string. */
TEST(Qwen4ExpGDNProjectionEvidence, RejectsMissingMalformedAndForeignSelectedRecords)
{
    const auto directory=source();
    const std::vector<PerfStatRecord> empty;
    EXPECT_THROW(requireQwen4ExpGDNQKVRouteEvidence(empty,directory),std::invalid_argument);
    for(int mutation=0;mutation!=16;++mutation)
    {
        std::vector<PerfStatRecord> records{selected()};auto &row=records[0];
        if(mutation==0)row.tags["route"]="homogeneous_full";
        if(mutation==1)row.tags.erase("route");
        if(mutation==2)row.tags["names"]="z";
        if(mutation==3)row.tags["projections"]="2";
        if(mutation==4)row.tags["codebooks"]="8";
        if(mutation==5)row.tags["n"]="6144";
        if(mutation==6)row.tags["k"]="10240";
        if(mutation==7)row.tags["m"]="0";
        if(mutation==8)row.tags["m"]="9x";
        if(mutation==9)row.domain="requested_environment";
        if(mutation==10)row.name="gdn_qkv_requested";
        if(mutation==11)row.count=0;
        if(mutation==12)row.value=1;
        if(mutation==13)row.value=std::numeric_limits<double>::quiet_NaN();
        if(mutation==14)row.value=std::numeric_limits<double>::infinity();
        if(mutation==15)row.kind=PerfStatRecord::Kind::Timer;
        EXPECT_THROW(requireQwen4ExpGDNQKVRouteEvidence(records,directory),std::invalid_argument);
    }
}

/** @test Original model metadata owns source identity/type/NK, without a second derived layout. */
TEST(Qwen4ExpGDNProjectionEvidence, RejectsMissingOrConvertedOriginalSourceMetadata)
{
    const std::vector<PerfStatRecord> records{selected()};
    for(int mutation=0;mutation!=3;++mutation)
    {
        auto directory=source();
        auto found=std::find_if(directory.tensors.begin(),directory.tensors.end(),
            [](const auto &tensor){return tensor.name=="blk.0.attn_qkv.weight";});
        ASSERT_NE(found,directory.tensors.end());
        if(mutation==0)directory.tensors.erase(found);
        if(mutation==1)found->type=GGUFTensorType::F32;
        if(mutation==2)found->dimensions={0,2560};
        EXPECT_THROW(requireQwen4ExpGDNQKVRouteEvidence(records,directory),std::invalid_argument);
    }
}
} // namespace llaminar2::test::parity::qwen4exp
