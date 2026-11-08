/**
 * @file CPUWeightPreparationGeometry.cpp
 * @brief Same bounded native unit and scratch consumed by CPU final-weight writers.
 *
 * This quote describes transient input and workspace only. Final storage is
 * allocated by the original persistent owner and never charged here again.
 */
#include "CPUWeightPreparationGeometry.h"
#include "CPUNativeVNNIWeightPacker.h"
#include "tensors/NativeVnniFormatInfo.h"
#include "tensors/W4A4MatrixContract.h"
#include <limits>
namespace llaminar2 {
std::optional<NativeWeightLoadStagingRequest> quoteCPUWeightPreparation(
    TensorType type,size_t n,size_t k,WeightPrecision precision,CPUWeightConsumer consumer,
    FloatingActivationOperandPrecision operand) {
    (void)operand;
    if(consumer!=CPUWeightConsumer::ResidentTensor && consumer!=CPUWeightConsumer::PreparedGemm)
        throw std::invalid_argument("CPU weight quote has invalid final consumer");
    switch(precision) {
      case WeightPrecision::NATIVE:case WeightPrecision::CONVERT_TO_FP32:
      case WeightPrecision::CONVERT_TO_FP16:case WeightPrecision::CONVERT_TO_BF16:
      case WeightPrecision::CONVERT_TO_INT8:break;
      default:throw std::invalid_argument("CPU weight quote has invalid resolved precision");
    }
    if(!n || !k)throw std::invalid_argument("CPU weight quote requires positive matrix geometry");
    if(isW4A4Format(type)) {
        if(precision!=WeightPrecision::NATIVE || operand!=FloatingActivationOperandPrecision::Native)
            throw std::invalid_argument("W4A4/H preparation must retain native weights and its FP32 online-A4 boundary");
        const auto extent=w4a4::MatrixContract::sourceExtent(n,k);
        if(!extent)throw std::invalid_argument(extent.error().message());
        // The reader writes original bytes directly into the one final owner.
        // Validation happens only after complete coverage, before publication.
        return std::nullopt;
    }
    const bool floating=type==TensorType::FP32 || type==TensorType::FP16 || type==TensorType::BF16;
    if(floating) {
        const auto source_bytes=type==TensorType::FP32?size_t(4):size_t(2);
        const bool unchanged=precision==WeightPrecision::NATIVE ||
            (type==TensorType::FP32 && precision==WeightPrecision::CONVERT_TO_FP32) ||
            (type==TensorType::FP16 && precision==WeightPrecision::CONVERT_TO_FP16) ||
            (type==TensorType::BF16 && precision==WeightPrecision::CONVERT_TO_BF16);
        if(unchanged)return std::nullopt;
        if(n>std::numeric_limits<size_t>::max()/k)throw std::overflow_error("CPU source population overflows");
        if(precision==WeightPrecision::CONVERT_TO_INT8) {
            if(k%32)throw std::invalid_argument("Q8_0 conversion requires complete native groups");
            const auto* q8=native_vnni_formats::forQuantType("Q8_0");
            const auto pack=consumer==CPUWeightConsumer::PreparedGemm
                ? sizeof(Q8_0Block)+cpu::native_vnni::NativePackGroupScratch::checked(*q8).bytes : 0;
            return NativeWeightLoadStagingRequest{32*source_bytes,32*sizeof(float)+pack,
                consumer==CPUWeightConsumer::PreparedGemm?64u:0u,n*(k/32)};
        }
        return NativeWeightLoadStagingRequest{source_bytes,0,0,n*k};
    }
    const auto* format=native_vnni_formats::forQuantType(tensorTypeName(type));
    if(!format)throw std::invalid_argument("CPU weight quote has unsupported original native format");
    const auto elements=format->is_superblock?size_t(256):size_t(32);
    if(k%elements)throw std::invalid_argument("CPU native source K must preserve original group boundaries");
    if(n>std::numeric_limits<size_t>::max()/(k/elements))throw std::overflow_error("CPU native group population overflows");
    const auto bytes=cpu::native_vnni::native_block_bytes_for_format(format->codebook_id,format->is_superblock);
    if(precision==WeightPrecision::NATIVE) {
        if(consumer==CPUWeightConsumer::ResidentTensor)return std::nullopt;
        return NativeWeightLoadStagingRequest{bytes,cpu::native_vnni::NativePackGroupScratch::checked(*format).bytes,0,n*(k/elements)};
    }
    if(precision==WeightPrecision::CONVERT_TO_INT8)
        throw std::invalid_argument("Original native quantized CPU weights cannot be requantized");
    return NativeWeightLoadStagingRequest{bytes,precision==WeightPrecision::CONVERT_TO_FP32?0:elements*sizeof(float),0,n*(k/elements)};
}
}
