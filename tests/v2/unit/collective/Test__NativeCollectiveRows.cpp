/**
 * @file Test__NativeCollectiveRows.cpp
 * @brief Device-free proofs for capacity-stable, live-prefix native messages.
 *
 * Opaque addresses are never dereferenced. Exercise the real public geometry
 * and admission helpers over every supported scalar, operation and TP degree,
 * including aliases, empty publications, slices and arithmetic overflow.
 */
#include "collective/NativeCollectiveRowsContract.h"
#include "execution/local_execution/graph/IGraphBuilder.h"
#include "execution/compute_stages/stages/TPAllreduceStage.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>

using namespace llaminar2;
namespace
{
    /** @return Opaque test address; no physical memory is created or read. */
    void *opaque(std::uintptr_t address) { return reinterpret_cast<void *>(address); }
    /** @return Borrowed device operand deliberately inaccessible on the host. */
    const std::int32_t *countOwner() { return static_cast<const std::int32_t *>(opaque(4096)); }

    TEST(NativeCollectiveRows, RetainsPhysicalStridesWhileDevicePrefixShrinksAndGrows)
    {
        const NativeCollectiveRows rows(DeviceRowRange::deviceCounted(512, countOwner()), 1024);
        EXPECT_EQ(rows.bankElements(), 512u * 1024);
        EXPECT_EQ(rows.rows().countOwner(), countOwner());
        for (const int live : {512, 448, 1, 0, 7, 512, 32, 0})
        {
            EXPECT_EQ(rows.rows().activeRowsFor(live), live);
            EXPECT_EQ(rows.bankElements(), 512u * 1024);
        }
        EXPECT_EQ(rows.rows().activeRowsFor(-1), -1);
        EXPECT_EQ(rows.rows().activeRowsFor(513), -1);
        const NativeCollectiveRows fixed(DeviceRowRange::fullyActive(1), 1024);
        EXPECT_EQ(fixed.rows().countOwner(), nullptr);
        EXPECT_EQ(fixed.bankElements(), 1024u);
    }

    TEST(NativeCollectiveRows, RejectsSlicesEmptyRowsAndByteAbiOverflow)
    {
        const auto rows = DeviceRowRange::deviceCounted(512, countOwner());
        EXPECT_THROW((NativeCollectiveRows(rows.slice(1, 511), 4)), std::invalid_argument);
        EXPECT_THROW((NativeCollectiveRows(rows.slice(0, 511), 4)), std::invalid_argument);
        EXPECT_THROW((NativeCollectiveRows(rows, 0)), std::invalid_argument);
        EXPECT_THROW((NativeCollectiveRows(rows, std::size_t{1} << 32)), std::invalid_argument);
        const NativeCollectiveRows huge(DeviceRowRange::fullyActive(std::numeric_limits<int>::max()),
            std::numeric_limits<std::uint32_t>::max());
        EXPECT_FALSE(huge.byteGeometryValid(4, 2));
        EXPECT_FALSE(huge.byteGeometryValid(1, 8));
        const NativeCollectiveRows valid(rows, 7);
        for (auto bytes : {1u, 2u, 4u, 8u}) EXPECT_TRUE(valid.byteGeometryValid(bytes, 8));
        for (auto bytes : {0u, 3u, 16u}) EXPECT_FALSE(valid.byteGeometryValid(bytes, 2));
        EXPECT_FALSE(valid.byteGeometryValid(4, 0));
    }

    TEST(NativeCollectiveRows, AllFormatsOperationsDegreesAndParticipantCoordinates)
    {
        const NativeCollectiveRows rows(DeviceRowRange::deviceCounted(17, countOwner()), 37);
        for (auto dtype : {CollectiveDataType::FLOAT32, CollectiveDataType::FLOAT16,
                CollectiveDataType::BFLOAT16, CollectiveDataType::INT32, CollectiveDataType::INT8})
            for (auto operation : {NativeRowCollective::AllGather, NativeRowCollective::AllReduce,
                    NativeRowCollective::ReduceScatter})
                for (auto op : {CollectiveOp::ALLREDUCE_SUM, CollectiveOp::ALLREDUCE_MIN, CollectiveOp::ALLREDUCE_MAX})
                    for (int degree = 2; degree <= 8; ++degree)
                        for (int participant = 0; participant < degree; ++participant)
                        {
                            const auto reduction = operation == NativeRowCollective::AllGather ? CollectiveOp::ALLGATHER : op;
                            EXPECT_TRUE(nativeCollectiveRowsValid(operation, opaque(0x100000), opaque(0x200000),
                                rows, dtype, reduction, degree, participant, opaque(1)));
                            EXPECT_EQ(nativeCollectiveRowsValid(operation, opaque(0x100000), opaque(0x100000),
                                rows, dtype, reduction, degree, participant, opaque(1)),
                                operation == NativeRowCollective::AllReduce);
                        }
    }

    TEST(NativeCollectiveRows, RejectsMissingStreamBadEnumsAliasedCountAndWrappedRanges)
    {
        const NativeCollectiveRows rows(DeviceRowRange::deviceCounted(17, countOwner()), 37);
        const auto valid = [&](const void *send, void *receive, void *stream,
                CollectiveDataType dtype = CollectiveDataType::FLOAT32,
                CollectiveOp op = CollectiveOp::ALLREDUCE_SUM) {
            return nativeCollectiveRowsValid(NativeRowCollective::AllReduce, send, receive,
                rows, dtype, op, 2, 0, stream);
        };
        EXPECT_FALSE(valid(nullptr, opaque(0x200000), opaque(1)));
        EXPECT_FALSE(valid(opaque(0x100000), nullptr, opaque(1)));
        EXPECT_FALSE(valid(opaque(0x100000), opaque(0x200000), nullptr));
        EXPECT_FALSE(valid(opaque(0x100001), opaque(0x200000), opaque(1)));
        EXPECT_FALSE(valid(opaque(0x100000), opaque(0x100004), opaque(1)));
        EXPECT_FALSE(valid(opaque(0x100000), opaque(4096), opaque(1)));
        EXPECT_FALSE(valid(opaque(0x100000), opaque(4092), opaque(1)));
        EXPECT_FALSE(valid(opaque(0x100000), opaque(std::numeric_limits<std::uintptr_t>::max()-3), opaque(1)));
        EXPECT_FALSE(valid(opaque(0x100000), opaque(0x200000), opaque(1), static_cast<CollectiveDataType>(999)));
        EXPECT_FALSE(valid(opaque(0x100000), opaque(0x200000), opaque(1),
            CollectiveDataType::FLOAT32, CollectiveOp::BROADCAST));
        EXPECT_FALSE(nativeCollectiveRowsValid(static_cast<NativeRowCollective>(255),
            opaque(0x100000), opaque(0x200000), rows, CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM, 2, 0, opaque(1)));
        const NativeCollectiveRows misaligned(DeviceRowRange::deviceCounted(17,
            static_cast<const std::int32_t *>(opaque(4097))), 37);
        EXPECT_FALSE(nativeCollectiveRowsValid(NativeRowCollective::AllReduce,
            opaque(0x100000), opaque(0x200000), misaligned, CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM, 2, 0, opaque(1)));
    }

    TEST(NativeCollectiveRows, PrefillCountRequiresCompleteMaterializerOwnership)
    {
        std::array<std::int32_t, 8> tokens{};
        DevicePrefillChunkGraphBinding binding;
        binding.backend = static_cast<IBackend *>(opaque(1));
        binding.request_token_ids_device = &tokens[0];
        binding.request_position_ids_device = &tokens[1];
        binding.request_total_rows_device = &tokens[2];
        binding.cached_tokens_device = &tokens[3];
        binding.chunk_token_ids_device = &tokens[4];
        binding.chunk_position_ids_device = &tokens[5];
        binding.chunk_real_rows_device = &tokens[6];
        binding.chunk_row_stride_device = &tokens[7];
        binding.request_row_capacity = 1024;
        binding.bucket_seq_len = 512;
        binding.capture_identity = 1;
        ForwardInput input;
        input.device = DeviceId::rocm(0);
        input.batch_size = 1;
        input.seq_len = 512;
        input.token_ids_device = binding.chunk_token_ids_device;
        input.position_ids_device = binding.chunk_position_ids_device;
        input.position_policy = ForwardPositionPolicy::ExplicitRows;
        input.sequence_lengths_device = binding.chunk_real_rows_device;
        input.device_prefill_chunk = binding;
        ASSERT_TRUE(devicePrefillChunkOwnsInput(input));
        input.device = DeviceId::cuda(0);
        EXPECT_TRUE(devicePrefillChunkOwnsInput(input));
        const auto reject = [&](auto mutation) {
            auto invalid = input;
            mutation(invalid);
            EXPECT_FALSE(devicePrefillChunkOwnsInput(invalid));
        };
        reject([](auto &v) { v.device = DeviceId::cpu(); });
        reject([](auto &v) { v.device_prefill_chunk.reset(); });
        reject([](auto &v) { v.batch_size = 2; });
        reject([](auto &v) { v.seq_len = 511; });
        reject([&](auto &v) { v.token_ids = tokens.data(); });
        reject([&](auto &v) { v.position_ids = tokens.data(); });
        reject([](auto &v) { v.token_ids_device = nullptr; });
        reject([](auto &v) { v.position_ids_device = nullptr; });
        reject([](auto &v) { v.position_policy = ForwardPositionPolicy::ContiguousOffset; });
        reject([&](auto &v) { v.sequence_lengths_device = binding.request_total_rows_device; });
        reject([](auto &v) { v.device_prefill_chunk->capture_identity = 0; });
    }

    TEST(NativeCollectiveRows, StageRebindingCannotBypassRowGeometryAdmission)
    {
        TPAllreduceStage::Params params;
        params.device_id = DeviceId::cuda(0);
        params.count = 17 * 37;
        params.sideband_device_index = 0;
        params.live_rows.emplace(DeviceRowRange::deviceCounted(17, countOwner()), 37);
        TPAllreduceStage stage(params);
        const auto reject = [&](auto mutation) {
            auto invalid = params;
            mutation(invalid);
            EXPECT_THROW((TPAllreduceStage(invalid)), std::invalid_argument);
            EXPECT_THROW(stage.setParams(invalid), std::invalid_argument);
            EXPECT_EQ(stage.params().count, params.count);
            EXPECT_EQ(stage.params().live_rows->rows().countOwner(), countOwner());
        };
        reject([](auto &v) { v.count = 0; });
        reject([](auto &v) { ++v.count; });
        reject([](auto &v) { v.device_id = DeviceId::cpu(); });
        reject([](auto &v) { v.sideband_device_index = -1; });
    }
}
