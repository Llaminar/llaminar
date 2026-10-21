/**
 * @file Qwen4ExpSidecarFixture.h
 * @brief Native-directory fixture for the published32-role shared predictor.
 *
 * All large tensor extents are metadata only. An empty synthetic vocabulary
 * still has the exact declared extent so pairing checks cover every token ID
 * without opening models or inventing a tokenizer identity shortcut.
 */
#pragma once
#include "Qwen4ExpMetadataFixture.h"

namespace llaminar2::test
{
/** @brief Separate original directories with the actual shared dense export policy. */
struct Qwen4ExpSidecarFixture
{
    GGUFModel main = qwen4MetadataFixture(false);
    GGUFModel predictor;
    /** @brief Build native tensor extents without materializing any weight payload. */
    Qwen4ExpSidecarFixture()
    {
        GGUFValue vocabulary{.type=GGUFValueType::ARRAY,.array_length=248320,
            .array_element_type=GGUFValueType::STRING};
        vocabulary.string_array_value.resize(vocabulary.array_length);
        main.metadata["tokenizer.ggml.tokens"]=std::move(vocabulary);
        for(auto &tensor:main.tensors)sealBytes(tensor);
        predictor.architecture="qwen4exp";
        predictor.metadata=main.metadata;
        for(auto it=predictor.metadata.begin();it!=predictor.metadata.end();)
            if(it->first.rfind("qwen4exp.ple.",0)==0)it=predictor.metadata.erase(it);else ++it;
        predictor.metadata.erase("qwen4exp.attention.recurrent_layers");
        predictor.metadata["qwen4exp.block_count"]=qwen4Scalar(GGUFValueType::UINT32,uint32_t(49));
        predictor.metadata["qwen4exp.nextn_predict_layers"]=qwen4Scalar(GGUFValueType::UINT32,uint32_t(1));
        predictor.metadata["qwen4exp.nextn_shared_target_tensors"]={.type=GGUFValueType::BOOL,.data={1}};
        auto ratios=main.metadata.at("qwen4exp.attention.compress_ratios").asUInt64Array();
        ratios.push_back(0);
        predictor.metadata["qwen4exp.attention.compress_ratios"]=qwen4Array(GGUFValueType::UINT64,ratios);
        const auto add=[&](const std::string &suffix,std::vector<uint64_t> shape,GGUFTensorType type=GGUFTensorType::Q8_0)
        {
            predictor.tensors.push_back({.name="blk.48."+suffix+".weight",.dimensions=std::move(shape),.type=type});
            sealBytes(predictor.tensors.back());
        };
        add("attn_q",{12288,2560});add("attn_k",{512,2560});add("attn_v",{512,2560});
        add("attn_output",{2560,6144});add("attn_q_norm",{256},GGUFTensorType::F32);add("attn_k_norm",{256},GGUFTensorType::F32);
        add("ffn_gate_inp",{512,2560},GGUFTensorType::F32);add("ffn_gate_inp_shexp",{2560},GGUFTensorType::F32);
        add("ffn_gate_exps",{2560,640,512});add("ffn_up_exps",{2560,640,512});add("ffn_down_exps",{640,2560,512});
        add("ffn_gate_shexp",{640,2560});add("ffn_up_shexp",{640,2560});add("ffn_down_shexp",{2560,640});
        for(const char *role:{"attn","ffn"})
        {
            const auto name=std::string("hc_")+role;
            add(name+"_norm",{10240},GGUFTensorType::F32);add(name+"_down",{320,10240});
            add(name+"_up",{10240,320});add(name+"_inject",{4,10240});
        }
        add("indexer.q_proj",{512,2560},GGUFTensorType::BF16);add("indexer.k_proj",{128,2560},GGUFTensorType::BF16);
        add("indexer.q_norm",{128},GGUFTensorType::F32);add("indexer.k_norm",{128},GGUFTensorType::F32);
        add("nextn.eh_proj",{2560,5120});add("nextn.enorm",{2560},GGUFTensorType::F32);add("nextn.hnorm",{10240},GGUFTensorType::F32);
        add("nextn.hc_head_norm",{10240},GGUFTensorType::F32);add("nextn.hc_head_down",{320,10240});add("nextn.hc_head_up",{10240,320});
    }
    /** @brief Compute exact block bytes for synthetic original native tensor metadata. */
    static void sealBytes(GGUFTensorInfo &tensor)
    {
        uint64_t elements=1;
        for(auto width:tensor.dimensions)elements*=width;
        tensor.size_bytes=elements/(tensor.isQuantized()?tensor.getBlockSize():1)*tensor.getTypeSize();
    }
};
} // namespace llaminar2::test
