/**
 * @file Test__IQ3SSourceEquations.cpp
 * @brief Lean captured IQ3 original-factor dense and fused expert regressions.
 *
 * Expected values come from immutable original blocks and fixed consumed Q8,
 * independently of the prepared tuple. Exact per-role counter scopes prevent a
 * previous role or backend from certifying a missing optimized producer.
 */
#include "utils/IQ3SOriginalEquationGpuWitness.h"
#include "utils/IQ3SFusedMoEGpuSourceWitness.h"
#include "utils/IQ3SPrefillSourceWitness.h"
#include "utils/DebugEnv.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <string>
namespace {
/** @brief Restore caller evidence policy after all graph owners retire. */
struct IQ3KernelStats {
    std::string old_summary,old_filter;bool had_summary=false,had_filter=false;
    /** @brief Enable exact producer counters before constructing workers. */
    IQ3KernelStats() {
        if(const char*v=std::getenv("LLAMINAR_PERF_STATS_SUMMARY")){had_summary=true;old_summary=v;}
        if(const char*v=std::getenv("LLAMINAR_PERF_STATS_FILTER")){had_filter=true;old_filter=v;}
        setenv("LLAMINAR_PERF_STATS_SUMMARY","1",1);setenv("LLAMINAR_PERF_STATS_FILTER","kernel",1);
        llaminar2::mutableDebugEnv().reload();llaminar2::PerfStatsCollector::reset();
    }
    /** @brief Restore only after synchronous test worker retirement. */
    ~IQ3KernelStats(){
        if(had_summary)setenv("LLAMINAR_PERF_STATS_SUMMARY",old_summary.c_str(),1);else unsetenv("LLAMINAR_PERF_STATS_SUMMARY");
        if(had_filter)setenv("LLAMINAR_PERF_STATS_FILTER",old_filter.c_str(),1);else unsetenv("LLAMINAR_PERF_STATS_FILTER");
        llaminar2::mutableDebugEnv().reload();llaminar2::PerfStatsCollector::reset();
    }
};
}
#ifdef HAVE_CUDA
/** @brief Actual default captured dense source-scale consumption and Q8 publication. */
TEST(IQ3SSourceCUDA,DenseOriginalEquation){IQ3KernelStats stats;llaminar2::test::iq3s_source_witness::runDense(llaminar2::DeviceId::cuda(0));}
/** @brief Isolated source gate/up/down roles through captured public FromRouting. */
TEST(IQ3SSourceCUDA,FusedOriginalEquation){IQ3KernelStats stats;llaminar2::test::iq3s_source_witness::runGpu(llaminar2::DeviceId::cuda(0));}
/** @brief Actual retained complete/partial grouped prefill for every source role. */
TEST(IQ3SSourceCUDA,GroupedPrefillOriginalEquation){IQ3KernelStats stats;llaminar2::test::iq3s_source_witness::runPrefill(llaminar2::DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
/** @brief Actual default captured dense source-scale consumption and Q8 publication. */
TEST(IQ3SSourceROCm,DenseOriginalEquation){IQ3KernelStats stats;llaminar2::test::iq3s_source_witness::runDense(llaminar2::DeviceId::rocm(0));}
/** @brief Isolated source gate/up/down roles through captured public FromRouting. */
TEST(IQ3SSourceROCm,FusedOriginalEquation){IQ3KernelStats stats;llaminar2::test::iq3s_source_witness::runGpu(llaminar2::DeviceId::rocm(0));}
/** @brief Actual retained complete/partial grouped prefill for every source role. */
TEST(IQ3SSourceROCm,GroupedPrefillOriginalEquation){IQ3KernelStats stats;llaminar2::test::iq3s_source_witness::runPrefill(llaminar2::DeviceId::rocm(0));}
#endif
