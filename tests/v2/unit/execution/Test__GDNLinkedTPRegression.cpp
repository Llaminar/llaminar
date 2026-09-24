/**
 * @file Test__GDNLinkedTPRegression.cpp
 * @brief CPU-only regressions for dependency-closed GDN tensor parallelism.
 *
 * These tests lock down the protocol used by production CUDA/ROCm live-state
 * allgather without occupying an accelerator. They cover the Qwen 122B GDN
 * geometry at TP=2/4/8, prove the local modulo relation needs offset zero, and
 * prove rank-major collective output reassembles to global semantic order.
 */

#include "config/GDNHeadAssignment.h"
#include "execution/compute_stages/stages/GDNLinkedLiveStateGeometry.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <limits>
#include <vector>

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Materialize the raw rank-major allgather result independently.
         *
         * `full` is group-major. Each simulated participant copies its Q/K
         * interval followed by the corresponding interval from every V repeat,
         * exactly as the production dependency-closed local kernels publish it.
         */
        std::vector<float> makeRankMajorGathered(
            const std::vector<float> &full,
            const GDNLinkedLiveStateShape &shape)
        {
            std::vector<float> gathered;
            gathered.reserve(full.size());
            const int local_key_group = shape.localKeyGroupFloats();
            const int full_key_group = shape.fullKeyGroupFloats();
            const int local_value_group = shape.localValueGroupFloats();
            const int full_value_group = shape.fullValueGroupFloats();
            const int full_prefix = shape.prefix_group_count * full_key_group;

            for (int participant = 0; participant < shape.degree; ++participant)
            {
                for (int group = 0;
                     group < shape.prefix_group_count;
                     ++group)
                {
                    const size_t begin =
                        static_cast<size_t>(group * full_key_group +
                                            participant * local_key_group);
                    gathered.insert(
                        gathered.end(),
                        full.begin() + static_cast<std::ptrdiff_t>(begin),
                        full.begin() + static_cast<std::ptrdiff_t>(
                                           begin + local_key_group));
                }
                for (int repeat = 0; repeat < shape.repeat_factor; ++repeat)
                {
                    const size_t begin =
                        static_cast<size_t>(full_prefix +
                                            repeat * full_value_group +
                                            participant * local_value_group);
                    gathered.insert(
                        gathered.end(),
                        full.begin() + static_cast<std::ptrdiff_t>(begin),
                        full.begin() + static_cast<std::ptrdiff_t>(
                                           begin + local_value_group));
                }
            }
            return gathered;
        }
    } // namespace

    TEST(Test__GDNLinkedTPRegression, Qwen122StateAccountingIsExactThroughTP8)
    {
        const GDNLinkedLiveStateGeometry geometry{
            .global_key_heads = 16,
            .global_value_heads = 64,
            .key_width = 128,
            .value_width = 128,
            .conv_history_length = 3,
        };

        for (const int degree : std::array{2, 4, 8})
        {
            const auto conv = geometry.resolve(
                GDNLinkedLiveStateKind::ConvHistory,
                degree);
            const auto recurrence = geometry.resolve(
                GDNLinkedLiveStateKind::Recurrence,
                degree);
            ASSERT_TRUE(conv.has_value()) << "TP=" << degree;
            ASSERT_TRUE(recurrence.has_value()) << "TP=" << degree;
            EXPECT_EQ(conv->repeat_factor, 4);
            EXPECT_EQ(conv->prefix_group_count, 2);
            EXPECT_EQ(conv->full_state_floats, 96 * 128 * 3);
            EXPECT_EQ(conv->local_state_floats * degree,
                      conv->full_state_floats);
            EXPECT_EQ(recurrence->prefix_group_count, 0);
            EXPECT_EQ(recurrence->full_state_floats, 64 * 128 * 128);
            EXPECT_EQ(recurrence->local_state_floats * degree,
                      recurrence->full_state_floats);
        }
    }

    TEST(Test__GDNLinkedTPRegression, RankMajorGatherReassemblesEverySemanticGroup)
    {
        const GDNLinkedLiveStateGeometry geometry{
            .global_key_heads = 8,
            .global_value_heads = 32,
            .key_width = 2,
            .value_width = 3,
            .conv_history_length = 2,
        };

        for (const int degree : std::array{2, 4, 8})
        {
            for (const auto kind : {
                     GDNLinkedLiveStateKind::ConvHistory,
                     GDNLinkedLiveStateKind::Recurrence,
                 })
            {
                const auto shape = geometry.resolve(kind, degree);
                ASSERT_TRUE(shape.has_value()) << "TP=" << degree;
                std::vector<float> full(
                    static_cast<size_t>(shape->full_state_floats));
                for (size_t i = 0; i < full.size(); ++i)
                    full[i] = static_cast<float>(i) + 0.25F;

                const std::vector<float> gathered =
                    makeRankMajorGathered(full, *shape);
                ASSERT_EQ(gathered.size(), full.size());
                std::vector<int> source_visits(gathered.size(), 0);
                for (size_t full_offset = 0;
                     full_offset < full.size();
                     ++full_offset)
                {
                    const auto gathered_offset =
                        geometry.gatheredOffsetForFullOffset(
                            kind,
                            degree,
                            full_offset);
                    ASSERT_TRUE(gathered_offset.has_value());
                    ASSERT_LT(*gathered_offset, gathered.size());
                    ++source_visits[*gathered_offset];
                    EXPECT_EQ(gathered[*gathered_offset], full[full_offset])
                        << "TP=" << degree
                        << " full_offset=" << full_offset;
                }
                for (const int visits : source_visits)
                    EXPECT_EQ(visits, 1) << "TP=" << degree;
            }
        }
    }

    TEST(Test__GDNLinkedTPRegression, LocalModuloRelationUsesZeroOffsetThroughTP8)
    {
        for (const int degree : std::array{2, 4, 8})
        {
            for (int participant = 0; participant < degree; ++participant)
            {
                const GDNHeadAssignment assignment =
                    GDNHeadAssignment::forEqualRank(
                        /*global_key_heads=*/16,
                        /*global_value_heads=*/64,
                        participant,
                        degree);
                for (int local_value = 0;
                     local_value < assignment.localValueHeads();
                     ++local_value)
                {
                    const int global_key =
                        assignment.globalValueHead(local_value) %
                        assignment.globalKeyHeads();
                    EXPECT_EQ(
                        global_key - assignment.keyHeads().start,
                        assignment.localKeyHeadForValue(local_value))
                        << "TP=" << degree
                        << " participant=" << participant
                        << " local_value=" << local_value;
                }
            }
        }
    }

    TEST(Test__GDNLinkedTPRegression, UnevenTPRejectsEqualCountStateCollective)
    {
        const GDNLinkedLiveStateGeometry geometry{
            .global_key_heads = 16,
            .global_value_heads = 64,
            .key_width = 128,
            .value_width = 128,
            .conv_history_length = 3,
        };
        EXPECT_FALSE(geometry.resolve(
            GDNLinkedLiveStateKind::ConvHistory,
            /*degree=*/3));
        EXPECT_FALSE(geometry.resolve(
            GDNLinkedLiveStateKind::Recurrence,
            /*degree=*/3));
    }
} // namespace llaminar2
