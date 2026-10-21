/**
 * @file Test__ImmutableResidentLookup.cpp
 * @brief Device-free original-table identity, extent and alias checks for VRAM lookup.
 *
 * Opaque addresses are declaration identities only and are never dereferenced.
 * Native device execution belongs to the explicit CUDA/ROCm capture fixture.
 */
#include "kernels/qwen4exp/ImmutableResidentLookupKernels.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::qwen4exp;
namespace
{
/** @brief Deliberately separated opaque storage declarations for a fast pure fixture. */
struct Fixture
{
    static constexpr std::uintptr_t tableAddress=0x100000,idsAddress=0x200000,outputAddress=0x300000;
    const std::byte *data=reinterpret_cast<const std::byte *>(tableAddress);
    /** @return Complete source-native allocation identity without touching an accelerator. */
    ImmutableResidentTableView table(TensorType format=TensorType::IQ4_NL) const
    {
        const auto stride=format==TensorType::FP32?1024u:
            format==TensorType::FP16||format==TensorType::BF16?512u:
            NativeEmbeddingLayout::forSource(format,1,256).row_stride_bytes;
        return ImmutableResidentTableView::bind(DeviceId::cuda(0),format,data,37,256,stride,37*stride,9);
    }
    /** @return Exact sixteen-head output and borrowed independent control declarations. */
    ImmutableResidentGatherArgs args() const
    {
        return {{15,reinterpret_cast<const int *>(0x400000)},
            reinterpret_cast<const std::int64_t *>(idsAddress),15*16,
            reinterpret_cast<float *>(outputAddress),15*16*256,
            reinterpret_cast<int *>(0x500000),reinterpret_cast<int *>(0x600000)};
    }
};
}
/** @test Every original codebook and floating source retains native byte/generation identity. */
TEST(ImmutableResidentLookup, ExactSourceInventoryAndIdentity)
{
    Fixture fixture;
    std::vector<TensorType> formats{TensorType::FP32,TensorType::FP16,TensorType::BF16};
#define RESIDENT_FORMAT(F) formats.push_back(TensorType::F);
    QFN_LOOKUP_QUANT_FORMATS(RESIDENT_FORMAT)
#undef RESIDENT_FORMAT
    ASSERT_EQ(formats.size(),24u);
    for(const auto format:formats)
    {
        const auto table=fixture.table(format);
        EXPECT_EQ(table.format(),format);EXPECT_EQ(table.nativeData(),fixture.data);
        EXPECT_EQ(table.rowCount(),37u);EXPECT_EQ(table.rowElements(),256);
        EXPECT_EQ(table.nativeBytes(),37u*table.rowBytes());EXPECT_EQ(table.storageGeneration(),9u);
        EXPECT_EQ(table,fixture.table(format));
        const auto other=ImmutableResidentTableView::bind(DeviceId::cuda(0),format,fixture.data,37,256,
            table.rowBytes(),table.nativeBytes(),10);
        EXPECT_NE(table,other);
        EXPECT_NO_THROW(resident_lookup_detail::validate(table,fixture.args(),reinterpret_cast<void *>(0x700000)));
    }
}
/** @test Missing/changed source authority, incomplete original rows and overflow fail before capture. */
TEST(ImmutableResidentLookup, InvalidTableDeclarationsFailBeforeCapture)
{
    Fixture fixture;
    const auto bind=[&](DeviceId device,const std::byte *data,std::uint64_t rows,int width,
                        std::size_t stride,std::size_t bytes,std::uint64_t generation) {
        return ImmutableResidentTableView::bind(device,TensorType::IQ4_NL,data,rows,width,stride,bytes,generation);
    };
    EXPECT_THROW((void)ImmutableResidentTableView::bind(DeviceId::cuda(0),TensorType::INT32,fixture.data,37,256,1024,37*1024,1),std::invalid_argument);
    EXPECT_THROW((void)bind(DeviceId::cuda(0),reinterpret_cast<const std::byte *>(Fixture::tableAddress+1),37,256,144,37*144,1),std::invalid_argument);
    EXPECT_THROW((void)bind(DeviceId::cpu(),fixture.data,37,256,144,37*144,1),std::invalid_argument);
    EXPECT_THROW((void)bind(DeviceId::cuda(0),nullptr,37,256,144,37*144,1),std::invalid_argument);
    EXPECT_THROW((void)bind(DeviceId::cuda(0),fixture.data,0,256,144,0,1),std::invalid_argument);
    EXPECT_THROW((void)bind(DeviceId::cuda(0),fixture.data,37,256,144,37*144,0),std::invalid_argument);
    EXPECT_THROW((void)bind(DeviceId::cuda(0),fixture.data,37,255,144,37*144,1),std::invalid_argument);
    EXPECT_THROW((void)bind(DeviceId::cuda(0),fixture.data,37,256,143,37*144,1),std::invalid_argument);
    EXPECT_THROW((void)bind(DeviceId::cuda(0),fixture.data,37,256,144,37*144-1,1),std::invalid_argument);
    EXPECT_THROW((void)bind(DeviceId::cuda(0),fixture.data,std::numeric_limits<std::uint64_t>::max(),256,144,1,1),std::overflow_error);
    EXPECT_THROW((void)bind(DeviceId::cuda(0),reinterpret_cast<const std::byte *>(~std::uintptr_t{0}-15),37,256,144,37*144,1),std::overflow_error);
}
/** @test Exact admitted capacity, non-default stream and disjoint physical ownership are mandatory. */
TEST(ImmutableResidentLookup, InvocationRejectsCapacityAliasAndStreamDrift)
{
    Fixture fixture;const auto table=fixture.table();
    const auto check=[&](auto change) {auto args=fixture.args();change(args);
        EXPECT_THROW(resident_lookup_detail::validate(table,args,reinterpret_cast<void *>(0x700000)),std::invalid_argument);};
    for(const auto stream:{std::uintptr_t{0},std::uintptr_t{1},std::uintptr_t{2}})
        EXPECT_THROW(resident_lookup_detail::validate(table,fixture.args(),reinterpret_cast<void *>(stream)),std::invalid_argument);
    check([](auto &a){a.tokenRows.rows=0;});check([](auto &a){--a.idCapacity;});
    check([](auto &a){--a.outputElements;});check([](auto &a){a.ids=nullptr;});
    check([](auto &a){a.output=nullptr;});check([](auto &a){a.healthy=nullptr;});
    check([](auto &a){a.error=a.healthy;});check([](auto &a){a.output=reinterpret_cast<float *>(Fixture::idsAddress);});
    check([](auto &a){a.ids=reinterpret_cast<const std::int64_t *>(Fixture::tableAddress);});
    check([](auto &a){a.tokenRows.liveRows=a.healthy;});
    check([](auto &a){a.ids=reinterpret_cast<const std::int64_t *>(Fixture::idsAddress+1);});
    auto full=fixture.args();full.tokenRows.liveRows=nullptr;
    EXPECT_NO_THROW(resident_lookup_detail::validate(table,full,reinterpret_cast<void *>(0x700000)));
}
