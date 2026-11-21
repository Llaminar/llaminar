/**
 * @file ROCmRouterQ8SubmissionProof.h
 * @brief Quiescent test collection of router submission topology, separate from replay.
 *
 * Tests enable this scope before entering a device worker. Captured launches
 * increment the resolved family; replay must not invent another host submission.
 * Exact immutable tags also prove the original capacity and ownership boundary.
 */
#pragma once
#include "utils/PerfStatsCollector.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <exception>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>

namespace llaminar2::test
{
/** @brief Preserve environment and immutable collector policy across a focused proof. */
class ROCmRouterQ8SubmissionCollection final
{
public:
    /** @brief Enable just the two router projection submission families. */
    ROCmRouterQ8SubmissionCollection()
    {
        for (const auto *name : {"LLAMINAR_PERF_STATS_JSON","LLAMINAR_PERF_STATS_FILTER"})
            previous_[name] = std::getenv(name) ? std::optional<std::string>(std::getenv(name)) : std::nullopt;
        if (setenv("LLAMINAR_PERF_STATS_JSON","1",1) || setenv("LLAMINAR_PERF_STATS_FILTER",
                "kernel.rocm_router_q8_wmma_projection_submissions,kernel.rocm_router_q8_dot4_projection_submissions",1))
            throw std::runtime_error("cannot enable router submission proof");
        PerfStatsCollector::reloadConfigurationFromEnvironment();
        if (!PerfStatsCollector::isRecordFamilyEnabled("kernel","rocm_router_q8_wmma_projection_submissions") ||
            !PerfStatsCollector::isRecordFamilyEnabled("kernel","rocm_router_q8_dot4_projection_submissions"))
            throw std::runtime_error("router submission collection did not enable both engine families");
    }
    /** @brief Restore the exact caller policy after all workers have joined. */
    ~ROCmRouterQ8SubmissionCollection()
    {
        for (const auto &[name,value] : previous_)
            if (value ? setenv(name.c_str(),value->c_str(),1) : unsetenv(name.c_str())) std::terminate();
        PerfStatsCollector::reloadConfigurationFromEnvironment();
    }
    ROCmRouterQ8SubmissionCollection(const ROCmRouterQ8SubmissionCollection &) = delete;
    ROCmRouterQ8SubmissionCollection &operator=(const ROCmRouterQ8SubmissionCollection &) = delete;
private:
    std::map<std::string,std::optional<std::string>> previous_;
};

/** @brief Observe only submission families; no runtime or device probe is involved. */
class ROCmRouterQ8SubmissionSnapshot final
{
public:
    /** @brief Freeze counter counts before a recording or replay. */
    ROCmRouterQ8SubmissionSnapshot() : counts_(read()) {}
    /** @brief Verify all submission families remain unchanged through GPU replay. */
    void expectUnchanged() const { EXPECT_EQ(read(),counts_); }
    /** @brief Require exactly the named successful projection submissions.
     * @param device Owning visible ordinal. @param capacity Original admitted rows.
     * @param width Hidden columns. @param experts Output columns.
     * @param basis Exact paired operand tag. @param ownership Whole or compact rows.
     * @param waves Native cap, or zero for Dot4. @param count Successful enqueues. */
    void expectAdded(int device, int capacity, int width, int experts, const char *basis,
        const char *ownership, int waves, uint64_t count) const
    {
        auto expected = counts_;
        PerfStatRecord record;
        record.name = waves ? "rocm_router_q8_wmma_projection_submissions" : "rocm_router_q8_dot4_projection_submissions";
        record.device = "rocm:" + std::to_string(device);
        record.phase = "router";
        record.tags = {{"capacity",std::to_string(capacity)}, {"width",std::to_string(width)},
            {"experts",std::to_string(experts)}, {"basis",basis}, {"ownership",ownership}, {"k_waves",std::to_string(waves)}};
        expected[key(record)] += count;
        EXPECT_EQ(read(),expected);
    }
private:
    /** @return Complete immutable identity of one recorded submission family. */
    static std::string key(const PerfStatRecord &record)
    {
        std::string result = record.name + "/" + record.device + "/" + record.phase;
        for (const auto &[name,value] : record.tags) result += "/" + name + "=" + value;
        return result;
    }
    /** @return Current exact counts, including both engines and every admitted shape. */
    static std::map<std::string,uint64_t> read()
    {
        std::map<std::string,uint64_t> result;
        for (const auto &record : PerfStatsCollector::snapshot({"kernel"}))
            if (record.name == "rocm_router_q8_wmma_projection_submissions" || record.name == "rocm_router_q8_dot4_projection_submissions")
                result[key(record)] = record.count;
        return result;
    }
    std::map<std::string,uint64_t> counts_;
};
}
