/**
 * @file Test__ModelLoaderSharedPredictor.cpp
 * @brief Device-free native source-routing and immutable binding regressions.
 *
 * Sparse generated files carry exact published geometry but only selected live
 * native rows/expert bytes are written. No model is loaded, downloaded, hashed,
 * or expanded into a full expert/table payload during these interface tests.
 */
#include "Qwen4ExpSidecarFixture.h"
#include "models/qwen4exp/Qwen4ExpSidecarManifest.h"
#include "loaders/ModelLoader.h"
#include "execution/moe/MoEOverlayPhysicalResidencyFabric.h"
#include "execution/moe/MoEOverlayInferenceTransaction.h"
#include "execution/mtp/MTPWeightManifest.h"
#include "loaders/ModelContext.h"
#include "utils/Sha256.h"
#include "tensors/Tensors.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace llaminar2::test
{
namespace
{
/** @brief Generated sparse original GGUF artifacts with bounded physical payloads. */
class SidecarFiles final
{
public:
    Qwen4ExpSidecarFixture fixture;
    std::filesystem::path root;
    std::string main_path, predictor_path;
    /**
     * @brief Serialize two independent directories and native source identities.
     * @param routed_main Add exact original main expert parents for routed interval tests.
     */
    explicit SidecarFiles(bool routed_main=false)
    {
        if(routed_main)
            for(int layer=0;layer<48;++layer)
                for(const char *role:{"ffn_gate_exps","ffn_up_exps","ffn_down_exps"})
                {
                    const bool down=std::string_view(role)=="ffn_down_exps";
                    GGUFTensorInfo tensor{.name="blk."+std::to_string(layer)+"."+role+".weight",
                        .dimensions=down ? std::vector<uint64_t>{640,2560,512} : std::vector<uint64_t>{2560,640,512},
                        .type=GGUFTensorType::Q4_0};
                    Qwen4ExpSidecarFixture::sealBytes(tensor);
                    fixture.main.tensors.push_back(std::move(tensor));
                }
        char pattern[]="/tmp/qfn-shared-predictor-XXXXXX";
        const auto *created=::mkdtemp(pattern);
        if(!created)throw std::runtime_error("Cannot create shared predictor fixture directory");
        root=created;main_path=(root/"target.gguf").string();predictor_path=(root/"selected.gguf").string();
        try { write(main_path,fixture.main);write(predictor_path,fixture.predictor); }
        catch(...) { std::filesystem::remove_all(root);throw; }
    }
    /** @brief Retire only this fixture's generated files, including sparse extents. */
    ~SidecarFiles(){std::error_code ignored;std::filesystem::remove_all(root,ignored);}
    SidecarFiles(const SidecarFiles&)=delete;
    SidecarFiles &operator=(const SidecarFiles&)=delete;
    /** @brief Write only exact selected bytes at original tensor coordinates. */
    void payload(bool predictor,const std::string &name,uint64_t offset,const std::vector<uint8_t>&bytes)
    {
        const auto &model=predictor?fixture.predictor:fixture.main;
        const auto *tensor=model.findTensor(name);
        if(!tensor||offset>tensor->size_bytes||bytes.size()>tensor->size_bytes-offset)
            throw std::invalid_argument("Invalid selected fixture payload");
        std::fstream stream(predictor?predictor_path:main_path,std::ios::binary|std::ios::in|std::ios::out);
        stream.seekp(model.data_offset+tensor->offset+offset);
        stream.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
        if(!stream)throw std::runtime_error("Cannot write selected fixture payload");
    }
private:
    /** @brief Encode the same little-endian scalar representation parsed by GGUF. */
    template<class T> static void scalar(std::ostream &out,T value)
    { out.write(reinterpret_cast<const char*>(&value),sizeof(value)); }
    /** @brief Encode one GGUF string without a trailing terminator. */
    static void string(std::ostream &out,const std::string &value)
    { scalar(out,uint64_t(value.size()));out.write(value.data(),value.size()); }
    /** @brief Encode exact metadata subtypes, including token strings and integer arrays. */
    static void value(std::ostream &out,const GGUFValue &entry)
    {
        scalar(out,uint32_t(entry.type));
        if(entry.type==GGUFValueType::STRING)string(out,entry.asString());
        else if(entry.type==GGUFValueType::ARRAY)
        {
            scalar(out,uint32_t(entry.array_element_type.value()));scalar(out,entry.array_length);
            if(entry.array_element_type==GGUFValueType::STRING)
                for(const auto &token:entry.string_array_value)string(out,token);
            else out.write(reinterpret_cast<const char*>(entry.data.data()),entry.data.size());
        }
        else out.write(reinterpret_cast<const char*>(entry.data.data()),entry.data.size());
    }
    /** @brief Serialize native coordinates with sparse payload holes, never materializing capacity. */
    static void write(const std::string &path,GGUFModel &model)
    {
        GGUFValue architecture{.type=GGUFValueType::STRING,.data=std::vector<uint8_t>(16)};
        const uint64_t length=8;std::memcpy(architecture.data.data(),&length,8);
        std::memcpy(architecture.data.data()+8,"qwen4exp",8);
        model.metadata["general.architecture"]=std::move(architecture);
        uint64_t extent=0;
        for(auto &tensor:model.tensors)
        {
            extent=(extent+31)/32*32;tensor.offset=extent;tensor.split_idx=0;extent+=tensor.size_bytes;
        }
        std::ofstream out(path,std::ios::binary|std::ios::trunc);
        out.write("GGUF",4);scalar(out,uint32_t(3));scalar(out,uint64_t(model.tensors.size()));scalar(out,uint64_t(model.metadata.size()));
        for(const auto &[key,entry]:model.metadata){string(out,key);value(out,entry);}
        for(const auto &tensor:model.tensors)
        {
            string(out,tensor.name);scalar(out,uint32_t(tensor.dimensions.size()));
            auto dimensions=tensor.dimensions;
            if(dimensions.size()==2)std::swap(dimensions[0],dimensions[1]);
            for(auto width:dimensions)scalar(out,width);
            scalar(out,uint32_t(tensor.type));scalar(out,tensor.offset);
        }
        model.data_offset=(uint64_t(out.tellp())+31)/32*32;
        out.seekp(model.data_offset+extent-1);out.put(0);
        if(!out)throw std::runtime_error("Cannot write sparse shared predictor fixture");
    }
};

/** @test Native global roles keep target coordinates; dense dormant roles remain directory-only. */
TEST(ModelLoaderSharedPredictor, BindsOneSeparateDirectoryAndPreservesBorrowedGlobalIdentity)
{
    SidecarFiles files;ModelLoader loader;loader.setUseMmap(true);
    loader.setPayloadAccessPattern(ModelPayloadAccessPattern::DeviceStaging);
    ASSERT_TRUE(loader.loadModel(files.main_path));
    const auto original=loader.getModel().findTensor("token_embd.weight")->offset;
    const auto main_count=loader.tensorCount();
    EXPECT_EQ(loader.ggufPredictorDirectory(),nullptr);EXPECT_EQ(loader.predictorManifest(),nullptr);
    loader.bindSharedPredictor(SharedPredictorGGUFSource(files.predictor_path), SharedPredictorMaterialization::Active);
    ASSERT_NE(loader.ggufPredictorDirectory(),nullptr);ASSERT_NE(loader.predictorManifest(),nullptr);
    EXPECT_EQ(loader.blockCount(),48u);EXPECT_EQ(loader.getModel().tensors.size(),main_count);
    EXPECT_EQ(loader.tensorCount(),main_count+32);EXPECT_EQ(loader.tensorNames().size(),main_count+32);
    const auto active=loader.materializationNames();EXPECT_EQ(active.size(),main_count+28);
    for(const char *role:{"q_proj","k_proj","q_norm","k_norm"})
    {
        const auto name=std::string("blk.48.indexer.")+role+".weight";
        EXPECT_TRUE(loader.hasTensor(name));
        EXPECT_EQ(std::count(active.begin(),active.end(),name),0);
    }
    for(const auto &tensor:loader.getModel().tensors)
        EXPECT_EQ(std::count(active.begin(),active.end(),tensor.name),1);
    EXPECT_EQ(loader.artifactPaths(),(std::vector<std::string>{files.main_path,files.predictor_path}));
    EXPECT_EQ(loader.predictorManifest()->attention(),Qwen4ExpPredictorAttention::Dense);
    EXPECT_EQ(loader.predictorManifest()->tensor("token_embd.weight").tensor.offset,original);
    EXPECT_EQ(loader.predictorManifest()->tensor("token_embd.weight").source,Qwen4ExpTensorSource::Main);
    EXPECT_EQ(loader.getMmapRegion(loader.getModel().findTensor("token_embd.weight"))->path(),files.main_path);
    EXPECT_EQ(loader.getMmapRegion(loader.ggufPredictorDirectory()->findTensor("blk.48.nextn.hnorm.weight"))->path(),files.predictor_path);
    EXPECT_EQ(loader.getTensorShape("blk.48.ffn_gate_exps.weight"),(std::vector<size_t>{2560,640,512}));
    EXPECT_EQ(loader.totalBytes(),loader.getModel().findTensor("per_layer_token_embd.weight")->size_bytes+
        [&]{size_t bytes=0;for(const auto &t:loader.getModel().tensors)if(t.name!="per_layer_token_embd.weight")bytes+=t.size_bytes;
        for(const auto &t:loader.ggufPredictorDirectory()->tensors)bytes+=t.size_bytes;return bytes;}());
    EXPECT_THROW(loader.bindSharedPredictor(SharedPredictorGGUFSource(files.predictor_path), SharedPredictorMaterialization::Active),std::logic_error);
    EXPECT_THROW(loader.loadModel(files.main_path),std::logic_error);
}

/** @test An admitted inactive source owns identity but cannot leak into preload or native preparation. */
TEST(ModelLoaderSharedPredictor, InactiveSidecarRemainsDiscoverableButRejectsNativeMaterialization)
{
    SidecarFiles files;ModelLoader loader;loader.setUseMmap(false);
    ASSERT_TRUE(loader.loadModel(files.main_path));const auto main_names=loader.tensorNames();
    loader.bindSharedPredictor(SharedPredictorGGUFSource(files.predictor_path),SharedPredictorMaterialization::Inactive);
    EXPECT_EQ(loader.predictorMaterialization(),SharedPredictorMaterialization::Inactive);
    EXPECT_EQ(loader.materializationNames(),main_names);
    EXPECT_EQ(loader.tensorNames().size(),main_names.size()+32);
    EXPECT_EQ(loader.artifactPaths(),(std::vector<std::string>{files.main_path,files.predictor_path}));
    EXPECT_TRUE(loader.hasTensor("blk.48.nextn.hnorm.weight"));
    EXPECT_THROW(loader.loadTensor("blk.48.nextn.hnorm.weight"),std::logic_error);
    EXPECT_THROW(loader.loadTensorRowSlice("blk.48.attn_q.weight",0,1),std::logic_error);
    EXPECT_THROW(loader.loadTensorColumnSlice("blk.48.attn_q.weight",0,32),std::logic_error);
    EXPECT_THROW(loader.loadTensorExpertSlice("blk.48.ffn_gate_exps.weight",0,1),std::logic_error);
    EXPECT_THROW(loader.loadTensorExpertSelection("blk.48.ffn_gate_exps.weight",{511}),std::logic_error);
    EXPECT_EQ(loader.blockCount(),48u);
}

/** @test Every materialization API routes to exact native source bytes, including the final expert. */
TEST(ModelLoaderSharedPredictor, LoadsNativeFullRowColumnAndExpertSelectionsFromOriginalSource)
{
    SidecarFiles files;
    const std::string norm="blk.48.nextn.hnorm.weight",query="blk.48.attn_q.weight",expert="blk.48.ffn_gate_exps.weight";
    const std::vector<uint8_t> norm_bytes(10240*4,0x3f),embedding_bytes(2560*2,0x29);
    const std::vector<uint8_t> query_bytes(2560/32*34,0x35);
    const size_t expert_bytes=2560*640/32*34;
    const std::vector<uint8_t> first(expert_bytes,0x21),last(expert_bytes,0x53);
    files.payload(true,norm,0,norm_bytes);files.payload(true,query,0,query_bytes);
    files.payload(true,expert,0,first);files.payload(true,expert,511*expert_bytes,last);
    files.payload(false,"token_embd.weight",0,embedding_bytes);
    ModelLoader loader;loader.setUseMmap(false);ASSERT_TRUE(loader.loadModel(files.main_path));
    loader.bindSharedPredictor(SharedPredictorGGUFSource(files.predictor_path), SharedPredictorMaterialization::Active);
    auto full=loader.loadTensor(norm);ASSERT_NE(full,nullptr);EXPECT_EQ(full->native_type(),TensorType::FP32);
    EXPECT_EQ(std::memcmp(full->raw_data(),norm_bytes.data(),norm_bytes.size()),0);
    auto row=loader.loadTensorRowSlice(query,0,1);ASSERT_NE(row,nullptr);EXPECT_EQ(row->native_type(),TensorType::Q8_0);
    EXPECT_EQ(std::memcmp(row->raw_data(),query_bytes.data(),query_bytes.size()),0);
    auto column=loader.loadTensorColumnSlice(query,0,32);ASSERT_NE(column,nullptr);EXPECT_EQ(column->native_type(),TensorType::Q8_0);
    EXPECT_EQ(std::memcmp(column->raw_data(),query_bytes.data(),34),0);
    auto selected=loader.loadTensorExpertSelection(expert,{0,511});ASSERT_NE(selected,nullptr);
    EXPECT_EQ(selected->native_type(),TensorType::Q8_0);EXPECT_EQ(selected->shape(),(std::vector<size_t>{2560,640,2}));
    const auto *packed=static_cast<const uint8_t*>(selected->raw_data());
    EXPECT_EQ(std::memcmp(packed,first.data(),expert_bytes),0);EXPECT_EQ(std::memcmp(packed+expert_bytes,last.data(),expert_bytes),0);
    auto interval=loader.loadTensorExpertSlice(expert,511,512);ASSERT_NE(interval,nullptr);
    EXPECT_EQ(std::memcmp(interval->raw_data(),last.data(),expert_bytes),0);
    auto embedding=loader.loadTensorRowSlice("token_embd.weight",0,1);ASSERT_NE(embedding,nullptr);
    EXPECT_EQ(embedding->native_type(),TensorType::FP16);EXPECT_EQ(std::memcmp(embedding->raw_data(),embedding_bytes.data(),embedding_bytes.size()),0);
}

/** @test Failed admission is atomic, no source is discovered, and payload entry permanently seals identity. */
TEST(ModelLoaderSharedPredictor, RejectsLateOrInvalidBindingWithoutMutatingTarget)
{
    SidecarFiles files;ModelLoader loader;loader.setUseMmap(false);ASSERT_TRUE(loader.loadModel(files.main_path));
    EXPECT_THROW(SharedPredictorGGUFSource(""),std::invalid_argument);
    EXPECT_THROW(loader.bindSharedPredictor(SharedPredictorGGUFSource(files.main_path), SharedPredictorMaterialization::Active),std::invalid_argument);
    EXPECT_EQ(loader.ggufPredictorDirectory(),nullptr);EXPECT_EQ(loader.artifactPaths(),(std::vector<std::string>{files.main_path}));
    EXPECT_FALSE(loader.hasTensor("blk.48.nextn.hnorm.weight"));
    EXPECT_EQ(loader.loadTensor("missing.weight"),nullptr);
    EXPECT_THROW(loader.bindSharedPredictor(SharedPredictorGGUFSource(files.predictor_path), SharedPredictorMaterialization::Active),std::logic_error);
    EXPECT_EQ(loader.blockCount(),48u);EXPECT_EQ(loader.ggufPredictorDirectory(),nullptr);
    auto conflicting=files.fixture.main;
    conflicting.tensors.push_back(files.fixture.predictor.tensors.front());
    EXPECT_THROW(Qwen4ExpSidecarManifest(conflicting,files.fixture.predictor),std::invalid_argument);
    auto foreign_policy=files.fixture.predictor;
    foreign_policy.metadata["tokenizer.chat_template_eos_token_id"]=qwen4Scalar(GGUFValueType::UINT32,uint32_t(1));
    EXPECT_THROW(Qwen4ExpSidecarManifest(files.fixture.main,foreign_policy),std::invalid_argument);
    ModelLoader admitted;admitted.setUseMmap(false);ASSERT_TRUE(admitted.loadModel(files.main_path));
    admitted.sealTensorSources();
    EXPECT_THROW(admitted.bindSharedPredictor(SharedPredictorGGUFSource(files.predictor_path), SharedPredictorMaterialization::Active),std::logic_error);
}
/** @test ModelContext binds before the sole weight owner and publishes both source identities through its interface. */
TEST(ModelLoaderSharedPredictor, ModelContextOwnsAndSealsTheCompleteSourceSet)
{
    SidecarFiles files;ModelContextConfig config;config.use_mmap=false;
    config.shared_predictor.emplace(files.predictor_path);
    config.predictor_materialization = SharedPredictorMaterialization::Active;
    auto context=ModelContext::create(files.main_path,config);ASSERT_NE(context,nullptr);
    EXPECT_EQ(context->artifactPaths(),(std::vector<std::string>{files.main_path,files.predictor_path}));
    const auto loader=context->loader();ASSERT_NE(loader,nullptr);
    ASSERT_NE(loader->ggufPredictorDirectory(),nullptr);ASSERT_NE(loader->predictorManifest(),nullptr);
    EXPECT_EQ(loader->ggufDirectory()->block_count,48u);
    EXPECT_EQ(loader->ggufPredictorDirectory()->block_count,49u);
    EXPECT_EQ(loader->materializationNames().size(),loader->tensorNames().size()-4);
    std::vector<std::filesystem::path> paths;
    for(const auto &path:context->artifactPaths())paths.emplace_back(path);
    const auto before=sha256FileSetIdentityHex(paths);ASSERT_TRUE(before);
    const auto target_only=sha256FileSetIdentityHex({files.main_path});ASSERT_TRUE(target_only);
    EXPECT_NE(before,target_only);
    std::filesystem::resize_file(files.predictor_path,std::filesystem::file_size(files.predictor_path)+1);
    const auto replaced=sha256FileSetIdentityHex(paths);ASSERT_TRUE(replaced);EXPECT_NE(before,replaced);
    EXPECT_THROW(context->concreteLoader().bindSharedPredictor(SharedPredictorGGUFSource(files.predictor_path), SharedPredictorMaterialization::Active),std::logic_error);
}
} // namespace
/** @test Dense attention retains its real MoE source48 in a distinct immutable artifact. */
TEST(ModelLoaderSharedPredictor, RoutedManifestUsesOriginalSharedPredictorAuthority)
{
    SidecarFiles files(true); ModelLoader loader; loader.setUseMmap(false);
    ASSERT_TRUE(loader.loadModel(files.main_path));
    const auto main_count=loader.getModel().tensors.size();
    EXPECT_EQ(loader.tensorInfo("blk.48.ffn_gate_exps.weight"),nullptr);
    EXPECT_THROW((void)buildMoEOverlayLayerWeightManifestFromLoader(loader,49,512),std::invalid_argument);
    loader.bindSharedPredictor(SharedPredictorGGUFSource(files.predictor_path),SharedPredictorMaterialization::Active);
    ASSERT_NE(loader.predictorManifest(),nullptr);
    EXPECT_EQ(loader.predictorManifest()->attention(),Qwen4ExpPredictorAttention::Dense);
    EXPECT_EQ(loader.getModel().tensors.size(),main_count);
    EXPECT_EQ(loader.getModel().findTensor("blk.48.ffn_gate_exps.weight"),nullptr);
    ASSERT_NE(loader.ggufPredictorDirectory(),nullptr);
    EXPECT_EQ(loader.tensorInfo("blk.48.ffn_gate_exps.weight"),
        loader.ggufPredictorDirectory()->findTensor("blk.48.ffn_gate_exps.weight"));
    const auto learned=discoverMTPWeightManifest(loader,"qwen4exp",48,true);
    ASSERT_TRUE(learned.available)<<learned.diagnostic;
    ASSERT_EQ(learned.depths.size(),1u);EXPECT_TRUE(learned.depths.front().moe_ffn_layout);
    const auto family=resolveMoEOverlayInferenceGraphFamilyIdentity(loader,"qwen4exp",48,
        MoEOverlayMTPGraphFamilyPolicy::RetainModelSidecars,1,16,16,1,15);
    EXPECT_EQ(family.main_layer_count,48);EXPECT_EQ(family.routedLayerCapacity(),49);
    EXPECT_EQ(family.mtp_source_layers,(std::vector<int>{48}));
    const auto terminal=family.forRoutedLayerInterval(40,9);
    EXPECT_EQ(terminal.main_layer_count,8);EXPECT_EQ(terminal.routedLayerCapacity(),9);
    EXPECT_EQ(terminal.mtp_source_layers,(std::vector<int>{48}));
    const auto main=buildMoEOverlayLayerWeightManifestFromGGUF(loader.getModel(),48,512);
    // Original failure: source48 is not in the main-only directory. Retrying
    // or dropping that legitimate expert owner would hide the source defect.
    EXPECT_THROW((void)buildMoEOverlayLayerWeightManifestFromGGUF(loader.getModel(),49,512),std::invalid_argument);
    const auto all=buildMoEOverlayLayerWeightManifestFromLoader(loader,49,512);
    ASSERT_EQ(all.size(),49u);
    EXPECT_TRUE(std::equal(main.begin(),main.end(),all.begin()));
    const auto predictor=buildMoEOverlayLayerWeightManifestFromGGUF(*loader.ggufPredictorDirectory(),1,512,48);
    EXPECT_EQ(all.back(),predictor.front());EXPECT_EQ(all.back().layer_idx,48);
    EXPECT_EQ(buildMoEOverlayLayerWeightManifestFromLoader(loader,9,512,40),
        (std::vector<MoEOverlayLayerWeightManifest>(all.begin()+40,all.end())));
    EXPECT_NE(all.front().projections.front().format,all.back().projections.front().format);
    EXPECT_EQ(all.back().projections[0].N,640);EXPECT_EQ(all.back().projections[0].K,2560);
    EXPECT_EQ(all.back().projections[2].N,2560);EXPECT_EQ(all.back().projections[2].K,640);
    const auto replicas=learned.participantReplicaNames(MTPRoutedExpertWeightAuthority::ExpertOverlay);
    EXPECT_EQ(std::count(replicas.begin(),replicas.end(),"blk.48.ffn_gate_exps.weight"),0);
    EXPECT_EQ(std::count(replicas.begin(),replicas.end(),"blk.48.nextn.eh_proj.weight"),1);
    EXPECT_EQ(std::count(replicas.begin(),replicas.end(),"blk.48.ffn_gate_shexp.weight"),1);
}

/** @test Separate-source resolution keeps strict missing/type/shape/interval rejection. */
TEST(ModelLoaderSharedPredictor, RoutedManifestRejectsIncompleteOrUnsupportedSidecarMetadata)
{
    SidecarFiles files(true); ModelLoader loader; loader.setUseMmap(false);
    ASSERT_TRUE(loader.loadModel(files.main_path));
    loader.bindSharedPredictor(SharedPredictorGGUFSource(files.predictor_path),SharedPredictorMaterialization::Active);
    auto *tensor=const_cast<GGUFTensorInfo *>(loader.tensorInfo("blk.48.ffn_gate_exps.weight"));
    ASSERT_NE(tensor,nullptr);
    const auto original=*tensor;
    tensor->name="removed-sidecar-projection";
    EXPECT_THROW((void)buildMoEOverlayLayerWeightManifestFromLoader(loader,49,512),std::invalid_argument);
    *tensor=original;tensor->dimensions[2]=511;
    EXPECT_THROW((void)buildMoEOverlayLayerWeightManifestFromLoader(loader,49,512),std::invalid_argument);
    *tensor=original;tensor->type=static_cast<GGUFTensorType>(0x7fffffff);
    EXPECT_THROW((void)buildMoEOverlayLayerWeightManifestFromLoader(loader,49,512),std::invalid_argument);
    *tensor=original;
    EXPECT_THROW((void)buildMoEOverlayLayerWeightManifestFromLoader(loader,2,512,48),std::invalid_argument);
    EXPECT_THROW((void)buildMoEOverlayLayerWeightManifestFromLoader(loader,1,512,-1),std::invalid_argument);
    EXPECT_EQ(buildMoEOverlayLayerWeightManifestFromLoader(loader,1,512,48).front().layer_idx,48);
}

} // namespace llaminar2::test
