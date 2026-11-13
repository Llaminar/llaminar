/**
 * @file Test__ModelPayloadAccessPattern.cpp
 * @brief Device-free admission ordering for separately owned native row tables.
 *
 * Tiny sparse GGUF sources exercise the production directory parser and mmap
 * path. Their payload stays unread: mincore inspects only the distant native
 * table page. A separate pure-policy sweep proves unchanged ordinary sources
 * and explicit device-staging/selected consumers without constructing devices.
 */
#include "loaders/ModelLoader.h"
#include "loaders/ModelPayloadAccessPattern.h"
#include <gtest/gtest.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <sys/mman.h>
#include <unistd.h>

namespace llaminar2::test
{
namespace
{
/** @brief Serialize one exact GGUF scalar without inspecting any tensor bytes. */
template<class T> void field(std::ostream &stream,T value)
{
    stream.write(reinterpret_cast<const char*>(&value),sizeof(value));
}
/** @brief Serialize one original length-prefixed GGUF string. */
void stringField(std::ostream &stream,const std::string &text)
{
    field(stream,uint64_t{text.size()});stream.write(text.data(),text.size());
}
/** @brief Own a sparse original GGUF with one distant native IQ4_NL table row. */
class SparseGGUF final
{
public:
    static constexpr size_t Bytes=16u*1024u*1024u;
    static constexpr uint64_t TableOffset=8u*1024u*1024u;
    std::string path;
    /** @brief Write only directory metadata; the distant payload remains sparse and cold. */
    SparseGGUF()
    {
        char pattern[]="/tmp/llaminar-selected-source-XXXXXX";
        const int descriptor=::mkstemp(pattern);
        if(descriptor<0)throw std::runtime_error("cannot create source policy fixture");
        path=pattern;
        const int status=::ftruncate(descriptor,Bytes);::close(descriptor);
        if(status){::unlink(path.c_str());throw std::runtime_error("cannot size source policy fixture");}
        std::fstream stream(path,std::ios::in|std::ios::out|std::ios::binary);
        field(stream,uint32_t{0x46554747});field(stream,uint32_t{3});
        field(stream,uint64_t{1});field(stream,uint64_t{1});
        stringField(stream,"general.architecture");field(stream,uint32_t{8});stringField(stream,"qwen4exp");
        stringField(stream,"per_layer_token_embd.weight");field(stream,uint32_t{2});
        field(stream,uint64_t{160});field(stream,uint64_t{1});
        field(stream,static_cast<uint32_t>(GGUFTensorType::IQ4_NL));field(stream,TableOffset);
        stream.flush();
        if(!stream)throw std::runtime_error("cannot write sparse GGUF directory");
    }
    /** @brief Retire only this private source after all mapping leases have ended. */
    ~SparseGGUF(){::unlink(path.c_str());}
    SparseGGUF(const SparseGGUF&)=delete;
    SparseGGUF &operator=(const SparseGGUF&)=delete;
};

/** @test Source ownership selects mapping density independently of execution device. */
TEST(ModelPayloadAccessPattern, ImmutableTablePreventsDenseSourceWarmupOnly)
{
    using P=ModelPayloadAccessPattern;
    for(const auto pattern:{P::DenseCpuResident,P::DeviceStaging,P::SparseSelection})
    {
        EXPECT_EQ(resolveModelPayloadAccessPattern(pattern,WeightRole::ImmutableRowTable),
                  pattern==P::DenseCpuResident?P::SparseSelection:pattern);
        for(const auto role:{WeightRole::Embedding,WeightRole::MoEExpertGate,
                            WeightRole::HyperConnectionProjection,WeightRole::Other})
            EXPECT_EQ(resolveModelPayloadAccessPattern(pattern,role),pattern);
    }
}

/** @test Direct low-level loading resolves original row ownership before source mmap. */
TEST(ModelPayloadAccessPattern, DirectLoaderLeavesUnselectedNativeTablePageCold)
{
    SparseGGUF source;
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel(source.path));
    const auto &directory=loader.getModel();ASSERT_EQ(directory.tensors.size(),1u);
    const auto mapping=loader.getMmapRegion(&directory.tensors.front());ASSERT_NE(mapping,nullptr);
    EXPECT_FALSE(mapping->wasEagerPrefaulted());
    const auto page=static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    const auto offset=(directory.data_offset+SparseGGUF::TableOffset)/page*page;
    unsigned char resident=0;
    ASSERT_EQ(::mincore(const_cast<uint8_t*>(mapping->data())+offset,page,&resident),0);
    EXPECT_EQ(resident&1u,0u) << "Table capacity must not be materialized before its typed physical owner";
}
} // namespace
} // namespace llaminar2::test
