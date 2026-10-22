/**
 * @file Test__PreparedNativeVNNIInspection.cpp
 * @brief Captured immutable ROCm owner-plan regression through the original complete fixture.
 *
 * Backend-specific passive inspection lives in separate CUDA/HIP TUs so native
 * runtime ABI headers never collide. This TU adds no native runtime include.
 * The unchanged production fixture owns all numerical/lifetime/extent checks;
 * its host submission counters prove recording, never executed-token counts.
 * Bounded Automatic has Q4_K/Q5_K exact-geometry four-device witnesses, with
 * independent startup-hardware expectations; legacy smaller
 * fixtures remain negative defaults and explicit Four controls.
 */
#include "utils/MoEProjectionPipelineFixture.h"
#include "utils/PerfStatsCollector.h"
#include "kernels/rocm/gemm/ROCmMoERouteOwnedTuning.h"
#include "backends/HardwareInventory.h"
#include <gtest/gtest.h>
#include <array>
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
 * @param measured_geometry Require compiled Automatic and select the measured operation tuple.
 * @param measured_format Actual original source family for a measured witness; Q4_K or Q5_K.
 *
 * The existing fixture owns every lifecycle and byte/extent oracle. These checks
 * add actual accepted host capture metadata only. The per-table native tests
 * separately prove symbols and replay-stable capture counters. No profiler or
 * synthetic timing is part of this regression.
 */
void proveModelOwnerBinding(int participants, bool measured_geometry = false, const char *measured_format = "Q4_K")
{
    const auto policy = rocm::compiledMoERouteOwnedModelPartitionPolicy();
    if (measured_geometry)
        ASSERT_EQ(policy, rocm::MoERouteOwnedModelPartitionPolicy::Automatic)
            << "The Automatic-specific witness cannot certify an explicitly forced mode";
    OwnerCounters observation;
    ProjectionPipelineFixtureConfig config;
    config.rows = measured_geometry ? std::vector<int>{1} : std::vector<int>{1,2};
    config.geometry.intermediate=512;
    if (measured_geometry) config.geometry = {512, 2560, 640, 10};
    if (participants==4) config.format=measured_geometry ? measured_format : "Q4_K";
    config.exchange=ProjectionExchangeProof::DeviceCounted;
    config.service_phase=MoEOverlayServicePhaseHint::Decode;
    runProjectionPipelineFixture(DeviceId::rocm(0), participants, config);
    ASSERT_FALSE(::testing::Test::HasFailure());
    const auto *startup = DeviceManager::instance().hardware();
    ASSERT_NE(startup, nullptr) << "the fixture must retain canonical startup publication";
    // Independent test expectation from original native startup facts, not the
    // production selector under test. Unqualified hardware is a negative case.
    bool measured_hardware = participants == 4;
    for (int ordinal = 0; ordinal < participants; ++ordinal)
    {
        int matches = 0;
        for (const auto &entry : startup->rocm_devices)
            if (entry.type == ComputeBackendType::GPU_ROCM && entry.device_id == ordinal)
            {
                ++matches;
                measured_hardware = measured_hardware && entry.compute_capability == 906 &&
                    entry.compute_units == 60 && entry.name.find("MI50") != std::string::npos;
            }
        measured_hardware = measured_hardware && matches == 1;
    }
    const bool selected = policy == rocm::MoERouteOwnedModelPartitionPolicy::FourWorkerPlanes ||
        (policy == rocm::MoERouteOwnedModelPartitionPolicy::Automatic && measured_geometry && measured_hardware);
    ::testing::Test::RecordProperty("automatic_hardware_qualified", measured_geometry && measured_hardware ? "true" : "false");
    if (measured_geometry)
        ::testing::Test::RecordProperty("measured_source_codebook", config.format == "Q5_K" ? "7" : "5");
    ::testing::Test::RecordProperty("compiled_owner_partition_policy", std::to_string(static_cast<int>(policy)));
    ::testing::Test::RecordProperty("resolved_owner_partition_execution", selected ? "four_worker_planes" : "serial");
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
        EXPECT_EQ(record.tags.at("n"),std::to_string(config.geometry.intermediate));
        EXPECT_EQ(record.tags.at("k"),std::to_string(config.geometry.width));
        EXPECT_EQ(record.tags.at("grid_x"),std::to_string((config.geometry.intermediate + 63) / 64));
        EXPECT_EQ(record.tags.at("grid_y"),std::to_string(config.geometry.top_k));
        EXPECT_EQ(record.tags.at("grid_z"),"2");
        EXPECT_NE(record.tags.at("native_function"),"0");
        if (measured_geometry) EXPECT_EQ(record.tags.at("dynamic_shared_bytes"),"4096");
        devices.insert(record.device);
    }
    std::set<std::string> expected_devices;
    for (int ordinal = 0; ordinal < participants; ++ordinal)
        expected_devices.insert(DeviceId::rocm(ordinal).toString());
    EXPECT_EQ(devices,expected_devices);
}
#endif
} // namespace
#ifdef HAVE_ROCM
/** @brief Every supported format retains captured runtime-owner/reset/empty histories under the compiled policy. */
TEST(MoEModelPartitionBinding, ROCm2) { proveModelOwnerBinding(2); }
/** @brief Four real ranks exercise source-indexed ownership beyond a paired-rank special case. */
TEST(MoEModelPartitionBinding, ROCm4) { proveModelOwnerBinding(4); }
/** @brief Exact admitted owner geometry drives real default selection and all twenty replay histories. */
TEST(MoEModelPartitionBinding, AutomaticMeasuredROCm4) { proveModelOwnerBinding(4, true, "Q4_K"); }
/** @brief The second measured original source family exercises actual prepared Automatic lowering. */
TEST(MoEModelPartitionBinding, AutomaticMeasuredQ5KROCm4) { proveModelOwnerBinding(4, true, "Q5_K"); }
#endif
} // namespace llaminar2::test
