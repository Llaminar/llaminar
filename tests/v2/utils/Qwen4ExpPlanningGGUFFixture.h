/**
 * @file Qwen4ExpPlanningGGUFFixture.h
 * @brief Complete synthetic QFN source-directory owner for device-free admission.
 *
 * Every layer has real FA/GDN/MoE/HC source geometry and native tensor extents.
 * Payloads are sparse file holes, never loaded, mapped, hashed or presented as
 * numerical model evidence. The original table retains its IQ4_NL row format.
 */
#pragma once
#include "../unit/models/qwen4exp/Qwen4ExpMetadataFixture.h"
#include <filesystem>
#include <fstream>
#include <limits>
#include <unistd.h>
namespace llaminar2::test {
/** @brief One move-forbidden metadata source with bounded physical directory bytes. */
class Qwen4ExpPlanningGGUFFixture final {
public:
    /** @brief Create complete published geometry without materializing weight payloads. */
    Qwen4ExpPlanningGGUFFixture()
    {
        char pattern[]="/tmp/qfn-candidate-source-XXXXXX";const int fd=::mkstemp(pattern);
        if(fd<0)throw std::runtime_error("Cannot create QFN planning source");
        ::close(fd);path_=pattern;
        try {
            auto model=qwen4MetadataFixture(false);
            const auto add=[&](std::string name,std::vector<std::uint64_t> shape,GGUFTensorType type=GGUFTensorType::Q4_0) {
                model.tensors.push_back({.name=std::move(name),.dimensions=std::move(shape),.type=type});
            };
            add("output_norm.weight",{2560},GGUFTensorType::F32);
            for(int layer=0;layer<48;++layer) {
                const auto p="blk."+std::to_string(layer)+".";
                for(const auto *role:{"attn_norm","post_attention_norm"})add(p+role+".weight",{2560},GGUFTensorType::F32);
                if((layer+1)%4==0) {
                    add(p+"attn_q.weight",{12288,2560});add(p+"attn_gate.weight",{6144,2560});
                    for(const auto *role:{"attn_k","attn_v"})add(p+role+".weight",{512,2560});
                    add(p+"attn_output.weight",{2560,6144});
                    for(const auto *role:{"attn_q_norm","attn_k_norm"})add(p+role+".weight",{256},GGUFTensorType::F32);
                } else {
                    add(p+"attn_qkv.weight",{10240,2560});add(p+"ssm_out.weight",{2560,6144});
                    add(p+"attn_gate.weight",{6144,2560});
                    for(const auto *role:{"ssm_alpha","ssm_beta"})add(p+role+".weight",{48,2560},GGUFTensorType::F16);
                    add(p+"ssm_conv1d.weight",{10240,4},GGUFTensorType::F32);
                }
                add(p+"ffn_gate_exps.weight",{2560,640,512});add(p+"ffn_up_exps.weight",{2560,640,512});
                add(p+"ffn_down_exps.weight",{640,2560,512});
                add(p+"ffn_gate_inp.weight",{512,2560},GGUFTensorType::F32);
                add(p+"ffn_gate_inp_shexp.weight",{2560},GGUFTensorType::F32);
                add(p+"ffn_gate_shexp.weight",{640,2560});add(p+"ffn_up_shexp.weight",{640,2560});
                add(p+"ffn_down_shexp.weight",{2560,640});
                for(const auto *role:{"attn","ffn"}) {
                    const auto h=p+"hc_"+role;
                    add(h+"_norm.weight",{10240},GGUFTensorType::F32);add(h+"_down.weight",{320,10240});
                    add(h+"_up.weight",{10240,320});add(h+"_inject.weight",{4,10240});
                }
            }
            write(model);
        } catch(...) {::unlink(path_.c_str());throw;}
    }
    /** @brief Remove only the private sparse source after all metadata borrowers retire. */
    ~Qwen4ExpPlanningGGUFFixture(){::unlink(path_.c_str());}
    Qwen4ExpPlanningGGUFFixture(const Qwen4ExpPlanningGGUFFixture&)=delete;
    Qwen4ExpPlanningGGUFFixture &operator=(const Qwen4ExpPlanningGGUFFixture&)=delete;
    /** @return Stable original path consumed by PlanningModelSource. */
    const std::string &path()const noexcept{return path_;}
private:
    /** @brief Encode one parser-native little-endian scalar. */
    template<class T>static void scalar(std::ostream &out,T value){out.write(reinterpret_cast<const char*>(&value),sizeof(value));}
    /** @brief Encode one GGUF string with its exact length. */
    static void string(std::ostream &out,const std::string &value){scalar(out,std::uint64_t(value.size()));out.write(value.data(),value.size());}
    /** @brief Preserve scalar/array metadata subtypes without float conversion. */
    static void value(std::ostream &out,const GGUFValue &entry) {
        scalar(out,std::uint32_t(entry.type));
        if(entry.type==GGUFValueType::STRING)string(out,entry.asString());
        else if(entry.type==GGUFValueType::ARRAY) {
            scalar(out,std::uint32_t(entry.array_element_type.value()));scalar(out,entry.array_length);
            if(entry.array_element_type==GGUFValueType::STRING)for(const auto &v:entry.string_array_value)string(out,v);
            else out.write(reinterpret_cast<const char*>(entry.data.data()),entry.data.size());
        } else out.write(reinterpret_cast<const char*>(entry.data.data()),entry.data.size());
    }
    /** @brief Serialize exact original extents; only the directory and final sparse byte are written. */
    void write(GGUFModel &model) {
        GGUFValue architecture{.type=GGUFValueType::STRING,.data=std::vector<std::uint8_t>(16)};
        const std::uint64_t length=8;std::memcpy(architecture.data.data(),&length,8);std::memcpy(architecture.data.data()+8,"qwen4exp",8);
        model.metadata["general.architecture"]=std::move(architecture);
        std::uint64_t extent=0;
        for(auto &tensor:model.tensors) {
            std::uint64_t elements=1;
            for(auto width:tensor.dimensions) {
                if(!width||elements>UINT64_MAX/width)throw std::overflow_error("QFN fixture tensor geometry overflow");elements*=width;
            }
            const auto block=tensor.isQuantized()?tensor.getBlockSize():1;
            if(elements%block)throw std::invalid_argument("QFN fixture requires complete original native blocks");
            tensor.size_bytes=elements/block*tensor.getTypeSize();extent=(extent+31)/32*32;
            tensor.offset=extent;tensor.split_idx=0;extent+=tensor.size_bytes;
        }
        std::ofstream out(path_,std::ios::binary|std::ios::trunc);out.exceptions(std::ios::badbit|std::ios::failbit);
        out.write("GGUF",4);scalar(out,std::uint32_t(3));scalar(out,std::uint64_t(model.tensors.size()));scalar(out,std::uint64_t(model.metadata.size()));
        for(const auto &[key,entry]:model.metadata){string(out,key);value(out,entry);}
        for(const auto &tensor:model.tensors) {
            string(out,tensor.name);scalar(out,std::uint32_t(tensor.dimensions.size()));auto shape=tensor.dimensions;
            if(shape.size()==2)std::swap(shape[0],shape[1]);for(auto width:shape)scalar(out,width);
            scalar(out,std::uint32_t(tensor.type));scalar(out,tensor.offset);
        }
        const auto start=(std::uint64_t(out.tellp())+31)/32*32;
        out.seekp(start+extent-1);out.put(0);
    }
    std::string path_;
};
}
