/**
 * @file CPUWeightPrecisionIntentContract.inc
 * @brief Registered model operand intent survives unrelated global CPU conversion policies.
 */
#include "tensors/NativeQuantisedFormatInfo.h"
#include "tensors/TensorType.h"

/** @test Shared source/metadata policy preserves the original immutable FP32 operator contract. */
TEST(CPUWeightPrecisionIntentContract, ImmutableOperatorRolesRetainFP32AcrossGlobalPolicies)
{
    using Contract = PreparedWeightRepresentationContract;
    using Representation = ModelPreparedWeightRepresentation;
    constexpr WeightRole roles[] = {WeightRole::HyperConnectionNorm, WeightRole::SparseIndexerNorm,
        WeightRole::PerLayerEmbeddingNorm, WeightRole::PerLayerEmbeddingConvolution, WeightRole::SharedExpertInputGate};
    std::vector<TensorType> sources{TensorType::FP32, TensorType::FP16, TensorType::BF16};
    for (const auto& format : native_quantised_formats::kAllSourceFormats) {
        const auto type = tensorTypeFromName(format.quant_type);
        ASSERT_TRUE(type);
        sources.push_back(*type);
    }
    for (const auto role : roles)
        for (const auto source : sources) {
            SCOPED_TRACE(static_cast<int>(role));
            SCOPED_TRACE(tensorTypeName(source));
            const auto representation = Contract::resolve(role, source, "operator.weight", "qwen4exp",
                GDNProjectionArithmetic::FP16, MoERouterArithmetic::FP16);
            EXPECT_EQ(representation, source == TensorType::FP32 ? Representation::SourceNative : Representation::FP32);
            EXPECT_EQ(Contract::resolve(role, tensorTypeName(source)), representation);
            const auto operand = Contract::operandPrecision(role, "operator.weight", "qwen4exp",
                GDNProjectionArithmetic::FP16, MoERouterArithmetic::FP16);
            ASSERT_EQ(operand, FloatingActivationOperandPrecision::Native);
            for (const auto requested : {WeightPrecision::NATIVE, WeightPrecision::CONVERT_TO_INT8,
                WeightPrecision::CONVERT_TO_FP16, WeightPrecision::CONVERT_TO_BF16, WeightPrecision::CONVERT_TO_FP32})
                EXPECT_EQ(Contract::resolveCPUWeightPrecision(role, source, requested, representation, operand),
                    source == TensorType::FP32 ? WeightPrecision::NATIVE : WeightPrecision::CONVERT_TO_FP32);
        }
}

TEST(CPUWeightPrecisionIntentContract, RegisteredAlreadyTypedAndConvertedSourcesRetainModelIntent)
{
    using Contract=PreparedWeightRepresentationContract;
    using Operand=FloatingActivationOperandPrecision;
    for(const auto requested:{WeightPrecision::NATIVE,WeightPrecision::CONVERT_TO_INT8,
        WeightPrecision::CONVERT_TO_FP32,WeightPrecision::CONVERT_TO_FP16,WeightPrecision::CONVERT_TO_BF16}) {
        for(const auto original:{TensorType::FP16,TensorType::FP32,TensorType::Q6_K}) {
            const auto half_execution=Contract::resolve(WeightRole::GDNAlphaBetaProjection,original,
                "blk.0.ssm_alpha.weight","qwen35moe",GDNProjectionArithmetic::FP16,MoERouterArithmetic::Native);
            const auto half_operand=Contract::operandPrecision(WeightRole::GDNAlphaBetaProjection,
                "blk.0.ssm_alpha.weight","qwen35moe",GDNProjectionArithmetic::FP16,MoERouterArithmetic::Native);
            ASSERT_EQ(half_operand,Operand::FP16RNE);
            EXPECT_EQ(Contract::resolveCPUWeightPrecision(WeightRole::GDNAlphaBetaProjection,original,requested,half_execution,half_operand),
                original==TensorType::FP16?WeightPrecision::NATIVE:WeightPrecision::CONVERT_TO_FP16);
            const auto float_execution=Contract::resolve(WeightRole::GDNAlphaBetaProjection,original,
                "blk.0.ssm_alpha.weight","qwen35moe",GDNProjectionArithmetic::FP32,MoERouterArithmetic::Native);
            const auto float_operand=Contract::operandPrecision(WeightRole::GDNAlphaBetaProjection,
                "blk.0.ssm_alpha.weight","qwen35moe",GDNProjectionArithmetic::FP32,MoERouterArithmetic::Native);
            ASSERT_EQ(float_operand,Operand::FP32);
            EXPECT_EQ(Contract::resolveCPUWeightPrecision(WeightRole::GDNAlphaBetaProjection,original,requested,float_execution,float_operand),
                original==TensorType::FP32?WeightPrecision::NATIVE:WeightPrecision::CONVERT_TO_FP32);
        }
    }
}

TEST(CPUWeightPrecisionIntentContract, NativeCodebooksAndUnscopedFloatingConversionsRemainDistinct)
{
    using Contract=PreparedWeightRepresentationContract;
    using Operand=FloatingActivationOperandPrecision;
    for(const auto& format:native_quantised_formats::kAllSourceFormats) {
        const auto original=tensorTypeFromName(format.quant_type);ASSERT_TRUE(original);
        for(const auto requested:{WeightPrecision::NATIVE,WeightPrecision::CONVERT_TO_INT8,
            WeightPrecision::CONVERT_TO_FP32,WeightPrecision::CONVERT_TO_FP16,WeightPrecision::CONVERT_TO_BF16})
            EXPECT_EQ(Contract::resolveCPUWeightPrecision(WeightRole::Other,*original,requested,
                ModelPreparedWeightRepresentation::SourceNative,Operand::Native),WeightPrecision::NATIVE);
    }
    EXPECT_EQ(Contract::resolveCPUWeightPrecision(WeightRole::Other,TensorType::FP32,WeightPrecision::CONVERT_TO_INT8,
        ModelPreparedWeightRepresentation::SourceNative,Operand::Native),WeightPrecision::CONVERT_TO_INT8);
    EXPECT_EQ(Contract::resolveCPUWeightPrecision(WeightRole::Other,TensorType::FP16,WeightPrecision::CONVERT_TO_FP32,
        ModelPreparedWeightRepresentation::SourceNative,Operand::Native),WeightPrecision::CONVERT_TO_FP32);
    EXPECT_THROW((void)Contract::resolveCPUWeightPrecision(WeightRole::Other,TensorType::FP32,WeightPrecision::NATIVE,
        ModelPreparedWeightRepresentation::FP16,Operand::FP32),std::invalid_argument);
    EXPECT_THROW((void)Contract::resolveCPUWeightPrecision(WeightRole::Other,TensorType::FP16,WeightPrecision::NATIVE,
        ModelPreparedWeightRepresentation::FP32,Operand::FP16RNE),std::invalid_argument);
    EXPECT_THROW((void)Contract::resolveCPUWeightPrecision(WeightRole::Other,TensorType::FP16,WeightPrecision::NATIVE,
        ModelPreparedWeightRepresentation::SourceNative,static_cast<Operand>(255)),std::invalid_argument);
}
