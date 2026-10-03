/**
 * @file Test__BenchmarkGeometrySelection.cpp
 * @brief Device-free proof that explicit benchmark shapes cannot silently change.
 *
 * Exercise the shared CUDA/ROCm selector before any model, GPU context or timing
 * is admitted. Invalid partial lists and integer overflow must reject the whole
 * workload; absence alone selects defaults and valid request order is retained.
 */
#include "utils/BenchmarkGeometrySelection.h"

#include <gtest/gtest.h>

#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
    using llaminar2::test::parseBenchmarkGeometry;

    /** Only an absent selector may select the standard inventory. */
    TEST(BenchmarkGeometrySelection, AbsenceAdmitsDeclaredDefaults)
    {
        EXPECT_EQ(parseBenchmarkGeometry("rows", std::nullopt, {4, 16}),
                  (std::vector<int>{4, 16}));
        EXPECT_THROW(parseBenchmarkGeometry("rows", std::string_view{}, {4, 16}),
                     std::runtime_error);
    }

    /** Preserve exact order, whitespace-normalized values and every valid boundary. */
    TEST(BenchmarkGeometrySelection, ExplicitInventoryPreservesRequestedOrder)
    {
        EXPECT_EQ(parseBenchmarkGeometry("rows", " 16, 4 ,1 ", {8}),
                  (std::vector<int>{16, 4, 1}));
        const auto maximum = std::to_string(std::numeric_limits<int>::max());
        EXPECT_EQ(parseBenchmarkGeometry("rows", maximum, {4}),
                  (std::vector<int>{std::numeric_limits<int>::max()}));
    }

    /** A valid prefix cannot hide a bad suffix, overflow or duplicate measurement. */
    TEST(BenchmarkGeometrySelection, MalformedInventoryRejectsTheEntireRequest)
    {
        for (const std::string_view value : {
                 " ", "0", "-1", "4junk", "4.0", "+4", ",4", "4,", "4,,16",
                 "4,0,16", "4,-1,16", "4,bad,16", "4,4", "4, 4 ",
                 "999999999999999999999999999999999999", "4,99999999999999999999"})
        {
            SCOPED_TRACE(value);
            EXPECT_THROW(parseBenchmarkGeometry("capacity", value, {4, 16}),
                         std::runtime_error);
        }
    }

    /** Typed defaults must satisfy the same contract instead of bypassing admission. */
    TEST(BenchmarkGeometrySelection, InvalidDefaultsAreRejected)
    {
        EXPECT_THROW(parseBenchmarkGeometry("rows", std::nullopt, {}), std::runtime_error);
        EXPECT_THROW(parseBenchmarkGeometry("rows", std::nullopt, {0}), std::runtime_error);
        EXPECT_THROW(parseBenchmarkGeometry("rows", std::nullopt, {4, 4}), std::runtime_error);
        EXPECT_THROW(parseBenchmarkGeometry("rows", std::nullopt, {-1, 4}), std::runtime_error);
    }

    /** Admission errors identify the requested dimension, not an unrelated probe. */
    TEST(BenchmarkGeometrySelection, FailureRetainsSelectorIdentity)
    {
        try
        {
            (void)parseBenchmarkGeometry("LLAMINAR_ROCM_MOE_MTP_CAPACITIES", "4,bad", {16});
            FAIL() << "Malformed explicit capacity was admitted";
        }
        catch (const std::runtime_error &error)
        {
            EXPECT_NE(std::string_view(error.what()).find("LLAMINAR_ROCM_MOE_MTP_CAPACITIES"),
                      std::string_view::npos);
        }
    }
}
