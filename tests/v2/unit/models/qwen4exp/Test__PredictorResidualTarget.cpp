/**
 * @file Test__PredictorResidualTarget.cpp
 * @brief Device-free admitted-role/type/participant/width/overflow negatives for Dense0 writes.
 *
 * The fixture exercises the same metadata validator used by fromArena before
 * any native allocation or graph insertion. It does not pretend to authenticate
 * GPU storage, whose arena/root seal requires coherent integration coverage.
 */
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPredictorResidualBinding.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
using namespace llaminar2;
TEST(PredictorResidualTarget, ExactIntermediateAndRecursiveTerminalRolesRetain4H)
{
    const std::array<size_t,2> shape{16,10240};
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)}) {
        for (const auto role : {Qwen4ExpPredictorTrunk::Intermediate,Qwen4ExpPredictorTrunk::Terminal}) {
            const auto id = Qwen4ExpPredictorTrunkTarget::bufferFor(role);
            EXPECT_NO_THROW(Qwen4ExpPredictorTrunkTarget::requireDeclared(role,
                {id,TensorType::FP32,shape,device},4,10240,device));
            EXPECT_THROW(Qwen4ExpPredictorTrunkTarget::requireDeclared(role,
                {id,TensorType::FP32,shape,device},4,2560,device),std::invalid_argument);
        }
    }
}
TEST(PredictorResidualTarget, RejectsForeignRolesTypesDevicesAndInsufficientRows)
{
    const std::array<size_t,2> shape{4,10240};
    const auto role=Qwen4ExpPredictorTrunk::Terminal;
    EXPECT_THROW(Qwen4ExpPredictorTrunkTarget::requireDeclared(role,
        {BufferId::MTP_PROJECTED,TensorType::FP32,shape,DeviceId::cpu()},4,10240,DeviceId::cpu()),std::invalid_argument);
    EXPECT_THROW(Qwen4ExpPredictorTrunkTarget::requireDeclared(role,
        {BufferId::HIDDEN_STATE,TensorType::FP32,shape,DeviceId::cpu()},4,10240,DeviceId::cpu()),std::invalid_argument);
    EXPECT_THROW(Qwen4ExpPredictorTrunkTarget::requireDeclared(role,
        {BufferId::MTP_HIDDEN,TensorType::FP16,shape,DeviceId::cpu()},4,10240,DeviceId::cpu()),std::invalid_argument);
    EXPECT_THROW(Qwen4ExpPredictorTrunkTarget::requireDeclared(role,
        {BufferId::MTP_HIDDEN,TensorType::FP32,shape,DeviceId::cuda(0)},4,10240,DeviceId::rocm(0)),std::invalid_argument);
    EXPECT_THROW(Qwen4ExpPredictorTrunkTarget::requireDeclared(role,
        {BufferId::MTP_HIDDEN,TensorType::FP32,shape,DeviceId::cpu()},5,10240,DeviceId::cpu()),std::invalid_argument);
}
TEST(PredictorResidualTarget, RejectsMalformedRankZeroGeometryAndPhysicalByteOverflow)
{
    const std::array<size_t,3> rank3{4,4,2560};
    const std::array<size_t,2> overflow{std::numeric_limits<size_t>::max()/10240+1,10240};
    const std::array<size_t,2> bytesOverflow{std::numeric_limits<size_t>::max()/10240/4+1,10240};
    for (const auto shape : {std::span<const size_t>(overflow),std::span<const size_t>(bytesOverflow)})
        EXPECT_THROW(Qwen4ExpPredictorTrunkTarget::requireDeclared(Qwen4ExpPredictorTrunk::Terminal,
            {BufferId::MTP_HIDDEN,TensorType::FP32,shape,DeviceId::cpu()},1,10240,DeviceId::cpu()),std::overflow_error);
    EXPECT_THROW(Qwen4ExpPredictorTrunkTarget::requireDeclared(Qwen4ExpPredictorTrunk::Terminal,
        {BufferId::MTP_HIDDEN,TensorType::FP32,rank3,DeviceId::cpu()},1,10240,DeviceId::cpu()),std::invalid_argument);
    EXPECT_THROW(Qwen4ExpPredictorTrunkTarget::requireDeclared(Qwen4ExpPredictorTrunk::Terminal,
        {BufferId::MTP_HIDDEN,TensorType::FP32,overflow,DeviceId::cpu()},0,10240,DeviceId::cpu()),std::invalid_argument);
    EXPECT_THROW((void)Qwen4ExpPredictorTrunkTarget::bufferFor(static_cast<Qwen4ExpPredictorTrunk>(99)),std::invalid_argument);
}
