/**
 * @file Test__PredictorRows.cpp
 * @brief Device-free original/expanded count-authority and unsupported-hole controls.
 */
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPredictorRows.h"
#include <gtest/gtest.h>
#include <limits>
using namespace llaminar2;
TEST(PredictorRows, ExplicitFullyActiveAdmissionStaysFullyActiveOnEveryBackend)
{
    for (DeviceId device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)}) {
        const auto rows = Qwen4ExpPredictorRows::admit(1, 3, device, nullptr, nullptr);
        EXPECT_EQ(rows.device(), device);
        EXPECT_EQ(rows.tokens().countOwner(), nullptr);
        EXPECT_EQ(rows.tokens().activeRowsFor(3), 3);
        EXPECT_EQ(rows.branches().activeRowsFor(3), 12);
        EXPECT_FALSE(rows.primitives().liveRows);
    }
}
TEST(PredictorRows, CPUPresentCountRemainsTheSameOwnerAcrossAllConsumers)
{
    const std::vector<int> count{std::numeric_limits<int>::min()};
    const auto rows = Qwen4ExpPredictorRows::admit(1, 9, DeviceId::cpu(), &count, nullptr);
    EXPECT_EQ(rows.tokens().countOwner(), count.data());
    EXPECT_EQ(rows.branches().countOwner(), count.data());
    EXPECT_EQ(rows.branches().countCapacity(), 9);
    const auto primitives = rows.primitives();
    ASSERT_TRUE(primitives.liveRows);
    const auto &owner = std::get<Qwen4ExpNativeIntInput>(primitives.liveRows->value);
    EXPECT_EQ(owner.data, count.data());
    EXPECT_EQ(owner.capacity, 1);
    for (int published = -1; published <= 10; ++published)
        EXPECT_EQ(rows.branches().activeRowsFor(published), published < 0 || published > 9 ? -1 : published * 4);
}
TEST(PredictorRows, NativeGPUAdmissionDoesNotDereferenceOrReplaceCount)
{
    const auto *native = reinterpret_cast<const std::int32_t *>(0x1234);
    for (DeviceId device : {DeviceId::cuda(0), DeviceId::rocm(0)}) {
        const auto rows = Qwen4ExpPredictorRows::admit(1, 17, device, nullptr, native);
        EXPECT_EQ(rows.tokens().countOwner(), native);
        EXPECT_EQ(rows.branches().countOwner(), native);
        const auto primitives = rows.primitives();
        const auto &owner = std::get<Qwen4ExpNativeIntInput>(primitives.liveRows->value);
        EXPECT_EQ(owner.owner, device);
        EXPECT_EQ(owner.data, native);
    }
}
TEST(PredictorRows, RejectsMultiRequestHolesWrongOwnerAndOverflow)
{
    const std::vector<int> one{1}, two{1, 1};
    const auto *native = reinterpret_cast<const std::int32_t *>(0x1234);
    EXPECT_THROW((void)Qwen4ExpPredictorRows::admit(2, 9, DeviceId::cuda(0), nullptr, native), std::invalid_argument);
    EXPECT_THROW((void)Qwen4ExpPredictorRows::admit(1, 0, DeviceId::cpu(), nullptr, nullptr), std::invalid_argument);
    EXPECT_THROW((void)Qwen4ExpPredictorRows::admit(1, 9, DeviceId::cuda(0), &one, nullptr), std::invalid_argument);
    EXPECT_THROW((void)Qwen4ExpPredictorRows::admit(1, 9, DeviceId::cpu(), nullptr, native), std::invalid_argument);
    EXPECT_THROW((void)Qwen4ExpPredictorRows::admit(1, 9, DeviceId::cpu(), &two, nullptr), std::invalid_argument);
    EXPECT_THROW((void)Qwen4ExpPredictorRows::admit(1, 9, DeviceId::rocm(0), nullptr,
        reinterpret_cast<const std::int32_t *>(0x1235)), std::invalid_argument);
    EXPECT_THROW((void)Qwen4ExpPredictorRows::admit(1, std::numeric_limits<int>::max(), DeviceId::cpu(), nullptr, nullptr), std::overflow_error);
}
TEST(PredictorRows, CPUCompactPrefillAndDecodeRetainExactOriginalExtent)
{
    for (const int capacity : {1,3,16}) {
        std::vector<int> counts{capacity};
        const auto rows=Qwen4ExpPredictorRows::admit(1,capacity,DeviceId::cpu(),&counts,nullptr);
        EXPECT_NO_THROW(rows.requireCompactCPUQuery());
        EXPECT_EQ(rows.tokens().countOwner(),counts.data());
        EXPECT_EQ(rows.branches().physicalRows(),capacity*4);
        EXPECT_EQ(rows.branches().countOwner(),counts.data());
        for (const int invalid : {0,capacity-1,capacity+1,-1}) {
            counts[0]=invalid;
            EXPECT_THROW(rows.requireCompactCPUQuery(),std::invalid_argument);
        }
    }
    EXPECT_NO_THROW(Qwen4ExpPredictorRows::admit(1,16,DeviceId::cpu(),nullptr,nullptr).requireCompactCPUQuery());
}
TEST(PredictorRows, CompactCPUAdmissionNeverReadsNativeGPUCounts)
{
    const auto *native=reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(0x1234));
    for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)}) {
        const auto rows=Qwen4ExpPredictorRows::admit(1,16,device,nullptr,native);
        EXPECT_NO_THROW(rows.requireCompactCPUQuery());
        EXPECT_EQ(rows.tokens().countOwner(),native);
    }
}
