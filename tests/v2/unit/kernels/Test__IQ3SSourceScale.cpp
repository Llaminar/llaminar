/**
 * @file Test__IQ3SSourceScale.cpp
 * @brief Device-free IQ3 source factors, geometry and exact inverse-grid proof.
 *
 * Expected arithmetic starts from original half/nibble metadata and literal
 * grid zero (four positive ones), never from prepared factors. Fixed Q8 bytes
 * isolate source-scale preservation. Exhaustive grid/sign/lane proofs validate
 * the bounded inverse used by tier movement without any device or allocation.
 */
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cstring>
#include <omp.h>
#include "tensors/TensorClasses.h"
#include "tensors/IQQuantTables.h"
#include "tensors/NativeVnniIQ3SScale.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemv.h"
#include "kernels/cpu/gemm/CPUNativeAVX2Gemv.h"
#include "../../utils/IQ3SOriginalEquationSource.h"
#include "kernels/PackedWeightsSerialization.h"
#include "loaders/gpu_pipeline/GpuPackedWeightsFormat.h"
#include "execution/moe/ExpertTierWeightDeviceLayout.h"

namespace llaminar2::test {
namespace scale=native_vnni_iq3s_scale;
/** @brief All source grid entries are strictly sorted and uniquely invertible. */
TEST(IQ3SSourceScale, AllGridSignsAndLanesRoundTrip)
{
    for (int index=0;index<512;++index) {
        if(index) ASSERT_LT(iq3s_grid[index-1],iq3s_grid[index]);
        for(int signs=0;signs<16;++signs) {
            int8_t values[4];
            for(int lane=0;lane<4;++lane) {
                const int magnitude=(iq3s_grid[index]>>(8*lane))&255;
                ASSERT_GT(magnitude,0);ASSERT_LE(magnitude,15);ASSERT_EQ(magnitude%2,1);
                values[lane]=static_cast<int8_t>((signs&(1<<lane))?-magnitude:magnitude);
            }
            const auto encoded=scale::encodeGridQuartet(values,iq3s_grid);
            ASSERT_TRUE(encoded.valid());EXPECT_EQ(encoded.index,index);EXPECT_EQ(encoded.signs,signs);
            for(int group=0;group<8;++group) {
                const uint8_t qs=static_cast<uint8_t>(encoded.index);
                const uint8_t qh=static_cast<uint8_t>((encoded.index>>8)<<group);
                EXPECT_EQ(unsigned(qs)|(((qh>>group)&1u)<<8),unsigned(index));
                for(int half=0;half<2;++half) {
                    const uint8_t sign_byte=static_cast<uint8_t>(encoded.signs<<(4*half));
                    EXPECT_EQ((sign_byte>>(4*half))&15,signs);
                }
            }
        }
    }
}
/** @brief Reserved coefficients and off-grid values have explicit invalid results. */
TEST(IQ3SSourceScale, InvalidMetadataAndGridRejected)
{
    for(int n=0;n<16;++n) {EXPECT_TRUE(scale::valid(scale::encodeSourceCoefficient(n)));EXPECT_EQ(scale::coefficient(n),1+2*n);}
    EXPECT_FALSE(scale::valid(16));EXPECT_FALSE(scale::valid(65535));
    EXPECT_EQ(scale::sourceGridIndex(0,iq3s_grid),scale::kInvalidGridIndex);
    EXPECT_EQ(scale::sourceGridIndex(0,nullptr),scale::kInvalidGridIndex);
    for(int bad:{0,2,16,-16,-128}) {int8_t values[4]={1,1,1,static_cast<int8_t>(bad)};EXPECT_FALSE(scale::encodeGridQuartet(values,iq3s_grid).valid());}
}
/** @brief Metadata admission cannot confuse raw IQ3 factors with legacy half scales. */
TEST(IQ3SSourceScale, MandatoryTupleMetadataAndOldGenerationsRejected)
{
    auto header=buildGpuPackedHeader(RepackFormat::IQ3_S,11,13,true,true,false,1,32);
    const size_t total=gpuPackedTotalSize(header);
    ASSERT_TRUE(validateGpuPackedHeader(header,total));
    auto old=header;old.version=3;EXPECT_FALSE(validateGpuPackedHeader(old,total));
    auto absent=header;absent.is_asymmetric=0;absent.mins_size=0;EXPECT_FALSE(validateGpuPackedHeader(absent,total));
    auto wrong=header;wrong.payload_bytes=32;EXPECT_FALSE(validateGpuPackedHeader(wrong,total));
    PackedWeightsHeader cpu_header{};using Encoding=cpu::native_vnni::CPUNativeVNNIEncoding;
    cpu_header.codebook_id=11;cpu_header.is_superblock=1;cpu_header.payload_bytes=13;
    cpu_header.prepared_encoding=static_cast<uint8_t>(Encoding::IQ3SExpandedSourceScale);
    cpu_header.data_stride=2048;cpu_header.interleaved_block_stride=2432;
    ASSERT_TRUE(packed_weights_serialization::validNativeSourceScaleTupleHeader(cpu_header));
    cpu_header.interleaved_block_stride=2304;EXPECT_FALSE(packed_weights_serialization::validNativeSourceScaleTupleHeader(cpu_header));
    cpu_header.interleaved_block_stride=2432;cpu_header.codebook_id=12;EXPECT_FALSE(packed_weights_serialization::validNativeSourceScaleTupleHeader(cpu_header));
}
/** @brief Keep explicit rotation encoding separate from ordinary source-factor preservation. */
TEST(IQ3SSourceScale, OrdinaryAndExplicitRotationEncodingBoundaries)
{
    using namespace cpu::native_vnni;
    EXPECT_EQ(preparedEncodingForCodebook(11,false),CPUNativeVNNIEncoding::IQ3SExpandedSourceScale);
    EXPECT_EQ(preparedEncodingForCodebook(11,true),CPUNativeVNNIEncoding::ExpandedInt8);
    PackedWeightsHeader header{};header.codebook_id=11;header.is_superblock=1;
    header.prepared_encoding=static_cast<uint8_t>(CPUNativeVNNIEncoding::ExpandedInt8);
    header.payload_bytes=32;header.data_stride=2048;header.interleaved_block_stride=2304;
    // This checks encoding validity only; archive metadata does not restore the
    // separately owned activation-rotation basis (an existing distinct concern).
    EXPECT_TRUE(packed_weights_serialization::validNativeSourceScaleTupleHeader(header));
}
/** @brief Raw source tuple bytes roundtrip in V7 and reject earlier V6 metadata. */
TEST(IQ3SSourceScale, ArchiveRoundTripRejectsV6BeforeAdoption)
{
    using namespace cpu::native_vnni;using namespace packed_weights_serialization;
    IQ3_SBlock original{};original.d=0x3c01;original.scales[0]=1;
    std::vector<uint8_t> raw(sizeof(original));std::memcpy(raw.data(),&original,sizeof(original));
    IQ3_STensor tensor({1,32},raw);CPUNativeVNNIPackedWeights packed;
    const int threads=omp_get_max_threads();omp_set_num_threads(1);
    const bool prepared=packWeightsCPUNativeVNNI(&tensor,packed);omp_set_num_threads(threads);
    ASSERT_TRUE(prepared);CPUPackedWeights owner(std::move(packed));
    auto blob=serialize(owner);auto restored=deserialize(blob.data(),blob.size());
    ASSERT_NE(restored,nullptr);const auto *native=dynamic_cast<const CPUPackedWeights*>(restored.get());
    ASSERT_NE(native,nullptr);EXPECT_TRUE(native->packed().usesIQ3SExpandedSourceScale());
    ASSERT_EQ(native->packed().native_interleaved.size(),owner.packed().native_interleaved.size());
    EXPECT_EQ(std::memcmp(native->packed().native_interleaved.data(),owner.packed().native_interleaved.data(),owner.packed().native_interleaved.size()),0);
    const uint32_t old_version=6;
    std::memcpy(blob.data()+offsetof(PackedWeightsHeader,version),&old_version,sizeof(old_version));
    EXPECT_EQ(deserialize(blob.data(),blob.size()),nullptr);
    PackedWeightsTransferDescriptor descriptor{};
    std::memcpy(&descriptor.header,blob.data(),sizeof(descriptor.header));
    std::memcpy(&descriptor.sections,blob.data()+sizeof(descriptor.header),sizeof(descriptor.sections));
    EXPECT_THROW((void)allocateTransferTarget(descriptor),std::invalid_argument);
}
} // namespace llaminar2::test
