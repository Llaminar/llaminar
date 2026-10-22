/**
 * @file Test__PreparedNativeVNNIInspection.cpp
 * @brief Captured immutable ROCm owner-plan regression through the original complete fixture.
 *
 * Backend-specific passive inspection lives in separate CUDA/HIP TUs so native
 * runtime ABI headers never collide. This TU adds no native runtime include.
 * The unchanged production fixture owns all numerical/lifetime/extent checks;
 * its host submission counters prove recording, never executed-token counts.
 */
#include "utils/MoEProjectionPipelineFixture.h"
#include "utils/PerfStatsCollector.h"
#include "kernels/rocm/gemm/ROCmMoERouteOwnedTuning.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>

namespace llaminar2::test
{
namespace
{
#ifdef HAVE_ROCM
/** @brief Own the two exact collector keys, restoring absence/empty/nonempty values. */
class OwnerCounters final
{
    std::optional<std::string> json_, filter_;
    /** @return Exact original environment value, including absence. */
    static std::optional<std::string> prior(const char *name)
    { const auto *v=std::getenv(name); return v ? std::optional<std::string>{v} : std::nullopt; }
    /** @brief Restore original collector policy; inability to restore is fatal. */
    static void restore(const char *name,const std::optional<std::string> &v)
    { if ((v ? setenv(name,v->c_str(),1) : unsetenv(name)) != 0) std::terminate(); }
public:
    /** @brief Enable only owner capture submission observations, not kernel timers. */
    OwnerCounters() : json_(prior("LLAMINAR_PERF_STATS_JSON")),filter_(prior("LLAMINAR_PERF_STATS_FILTER"))
    {
        if(setenv("LLAMINAR_PERF_STATS_JSON","1",1)!=0 ||
            setenv("LLAMINAR_PERF_STATS_FILTER","kernel.rocm_moe_owner_m1_partition_submissions",1)!=0)
        {
            restore("LLAMINAR_PERF_STATS_FILTER",filter_); restore("LLAMINAR_PERF_STATS_JSON",json_);
            throw std::runtime_error("Cannot enable owner capture evidence");
        }
        PerfStatsCollector::reloadConfigurationFromEnvironment(); PerfStatsCollector::reset();
    }
    /** @brief Clear fixture records and restore the exact outer collector configuration. */
    ~OwnerCounters()
    {
        PerfStatsCollector::reset(); restore("LLAMINAR_PERF_STATS_FILTER",filter_);
        restore("LLAMINAR_PERF_STATS_JSON",json_); PerfStatsCollector::reloadConfigurationFromEnvironment();
    }
    OwnerCounters(const OwnerCounters &)=delete;
    OwnerCounters &operator=(const OwnerCounters &)=delete;
};
/**
 * @brief Run the unchanged complete graph lowerer and assert its source-bound compiled policy.
 * @param participants Explicit physical communicator size; no device ordinal assumptions enter kernel policy.
 *
 * The existing fixture owns every lifecycle and byte/extent oracle. These checks
 * add actual accepted host capture metadata only. The per-table native tests
 * separately prove symbols and replay-stable capture counters. No profiler or
 * synthetic timing is part of this regression.
 */
void proveModelOwnerBinding(int participants)
{
    OwnerCounters observation;
    ProjectionPipelineFixtureConfig config;
    config.rows={1,2};
    config.geometry.intermediate=512;
    if (participants==4) config.format="Q4_K";
    config.exchange=ProjectionExchangeProof::DeviceCounted;
    config.service_phase=MoEOverlayServicePhaseHint::Decode;
    runProjectionPipelineFixture(DeviceId::rocm(0), participants, config);
    ASSERT_FALSE(::testing::Test::HasFailure());
    const bool selected=rocm::compiledMoERouteOwnedModelPartitionExecution()==rocm::MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes;
    const auto records=PerfStatsCollector::snapshot({"kernel.rocm_moe_owner_m1_partition_submissions"});
    if(!selected) { EXPECT_TRUE(records.empty()); return; }
    ASSERT_FALSE(records.empty());
    std::set<std::string> devices;
    for(const auto &record:records)
    {
        ASSERT_EQ(record.domain,"kernel"); ASSERT_EQ(record.name,"rocm_moe_owner_m1_partition_submissions");
        EXPECT_EQ(record.phase,"submission"); EXPECT_GT(record.count,0u);
        EXPECT_EQ(record.value,static_cast<double>(record.count));
        EXPECT_EQ(record.tags.at("observation"),"host_execute_or_capture");
        EXPECT_EQ(record.tags.at("partition_execution"),"four_worker_planes");
        EXPECT_EQ(record.tags.at("worker_planes"),"4"); EXPECT_EQ(record.tags.at("block_threads"),"256");
        EXPECT_EQ(record.tags.at("n"),"512"); EXPECT_EQ(record.tags.at("k"),"256");
        EXPECT_EQ(record.tags.at("grid_x"),"8"); EXPECT_EQ(record.tags.at("grid_z"),"2");
        devices.insert(record.device);
    }
    EXPECT_EQ(devices.size(),static_cast<std::size_t>(participants));
}
#endif
} // namespace
#ifdef HAVE_ROCM
/** @brief Every supported format retains captured runtime-owner/reset/empty histories under the compiled policy. */
TEST(MoEModelPartitionBinding, ROCm2) { proveModelOwnerBinding(2); }
/** @brief Four real ranks exercise source-indexed ownership beyond a paired-rank special case. */
TEST(MoEModelPartitionBinding, ROCm4) { proveModelOwnerBinding(4); }
#endif
} // namespace llaminar2::test
