/**
 * @file audit.cpp
 * @brief Authenticate QFN shared/routed TP cuts and native source bytes without opening a GPU.
 */
#include "loaders/ModelLoader.h"
#include "planning/TensorParallelSourceDeclaration.h"
#include "execution/moe/MoEExpertShardCohort.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <set>

using namespace llaminar2;

namespace {
using Interval = MoEExpertShardCohort::Interval;

/// Read one bounded original GGUF expert or shared bank through the production reader.
std::vector<std::byte> read(ModelLoader& loader, const std::string& name,
                            const NativeWeightSourceGeometry& g, std::size_t expert,
                            Interval rows, Interval columns)
{
    const auto row_bytes = columns.columns / g.native_group_elements * g.native_group_bytes;
    const NativeWeightSourceWindow window{.original_expert=expert,.selected_expert_ordinal=expert,
        .first_row=std::size_t(rows.first_column),.rows=std::size_t(rows.columns),
        .first_column=std::size_t(columns.first_column),.columns=std::size_t(columns.columns),
        .native_row_bytes=row_bytes};
    std::vector<std::byte> bytes(window.nativeBytes());
    loader.readNativeWeightSourceWindow(name,window,bytes);
    return bytes;
}

/// Verify every prepared coordinate against the corresponding original full-bank native bytes.
std::size_t compare(ModelLoader& loader, const std::string& name, std::size_t expert,
                    bool down, const std::vector<Interval>& intervals)
{
    const auto g=loader.nativeWeightSourceGeometry(name);
    const auto K=int(g.original_shape.at(0));
    const auto N=int(g.original_shape.at(1));
    const auto whole=read(loader,name,g,expert,{0,N},{0,K});
    const auto row_bytes=K / g.native_group_elements * g.native_group_bytes;
    int next=0;
    std::size_t sampled_bytes=0;
    for(const auto cut:intervals) {
        if(cut.first_column!=next || cut.columns<=0)
            throw std::runtime_error("noncontiguous or empty interval: "+name);
        next+=cut.columns;
        const auto rows=down?Interval{0,N}:cut;
        const auto columns=down?cut:Interval{0,K};
        const auto local=read(loader,name,g,expert,rows,columns);
        const auto bytes=local.size()/rows.columns;
        const auto col_offset=columns.first_column/g.native_group_elements*g.native_group_bytes;
        for(int row=0;row<rows.columns;++row)
            if(std::memcmp(whole.data()+(rows.first_column+row)*row_bytes+col_offset,
                           local.data()+row*bytes,bytes)!=0)
                throw std::runtime_error("native bytes differ: "+name);
        sampled_bytes+=local.size();
    }
    if(next!=(down?K:N) || sampled_bytes!=whole.size())
        throw std::runtime_error("incomplete native source ownership: "+name);
    return sampled_bytes;
}
}

int main(int argc, char** argv)
try {
    if(argc!=10) throw std::invalid_argument("usage: audit GGUF ORDERED_GPU_ORDINALS[8]");
    ModelLoader loader;
    loader.setUseMmap(true);
    if(!loader.loadModel(argv[1])) throw std::runtime_error("GGUF load failed");
    const auto& gguf=loader.getModel();
    const auto profile=ModelMemoryProfile::fromGGUF(gguf);
    const auto manifest=buildMoEOverlayLayerWeightManifestFromLoader(
        loader,profile.mainLayerCount(),profile.expert_count);
    std::cout << "source H=" << profile.d_model << " d_ff=" << profile.d_ff
      << " E=" << profile.expert_count << " K=" << profile.expert_used_count
      << " layers=" << profile.mainLayerCount() << "/" << profile.totalLayerCount()
      << " q=" << profile.n_heads << " kv=" << profile.n_kv_heads
      << " gdn=" << profile.gdn_group_count << "/" << profile.gdn_time_step_rank << std::endl;
    for(const int width : {1,2,4,8}) {
        std::vector<DeviceId> devices;
        for(int slot=0;slot<width;++slot) devices.push_back(DeviceId::rocm(std::stoi(argv[2+slot])));
        try {
            const auto mtp=resolveQfnTPDeclaration(gguf,TensorParallelMainPlacement::FullMainRankLocal,
                MTPRuntimeConfig{.enabled=true},DenseParallelPolicy::TensorParallel,devices);
            const auto assignment=mtp.partition(devices,{},profile.d_ff,profile.vocab_size);
            std::cout << "width=" << width << " retained_mtp=partitioned coverage=" << int(mtp.coverage())
                      << " assignments=" << assignment.worldSize() << std::endl;
        } catch(const std::invalid_argument& e) {
            std::cout << "width=" << width << " retained_mtp=rejected " << e.what() << std::endl;
        }
        const auto declaration=resolveQfnTPDeclaration(gguf,TensorParallelMainPlacement::FullMainRankLocal,
            MTPRuntimeConfig{},DenseParallelPolicy::TensorParallel,devices);
        const auto tp=declaration.partition(devices,{},profile.d_ff,profile.vocab_size);
        if(width>1) {
            auto permuted=devices;
            std::reverse(permuted.begin(),permuted.end());
            bool rejected=false;
            try {
                declaration.validate(tp,permuted);
            } catch(const std::invalid_argument&) {
                rejected=true;
            }
            if(!rejected)
                throw std::runtime_error("foreign ordered physical owner accepted the retained TP assignment");
        }
        std::vector<Interval> shared;
        std::cout << "width=" << width << " coverage=" << int(declaration.coverage());
        for(int slot=0;slot<width;++slot) {
            const auto& assignment=tp.forRank(slot);
            shared.push_back({assignment.d_ff_start,assignment.d_ff_count});
            std::cout << " owner=" << assignment.device.ordinal
              << ":S" << assignment.d_ff_start << "+" << assignment.d_ff_count
              << ":Q" << assignment.head_start << "+" << assignment.head_count
              << ":KV" << assignment.kv_head_start << "+" << assignment.kv_head_count;
        }
        std::cout << std::endl;
        std::size_t sampled_bytes=0;
        for(const auto& layer:manifest) {
            const auto id=layer.layer_idx;
            const auto routed=MoEExpertShardCohort::gpuIntervals(layer,width,id);
            const auto expert=(std::size_t(id)*37)%profile.expert_count;
            std::cout << "width=" << width << " layer=" << id << " expert=" << expert;
            for(const auto cut:routed) std::cout << " F" << cut.first_column << "+" << cut.columns;
            for(const auto role:{"gate","up","down"}) {
                const auto prefix="blk."+std::to_string(id)+".ffn_"+role;
                const auto routed_name=prefix+"_exps.weight";
                const auto shared_name=prefix+"_shexp.weight";
                const bool down=std::string_view(role)=="down";
                const auto g=loader.nativeWeightSourceGeometry(routed_name);
                const auto s=loader.nativeWeightSourceGeometry(shared_name);
                std::cout << " type=" << int(g.original_type) << "/" << int(s.original_type);
                sampled_bytes+=compare(loader,routed_name,expert,down,routed);
                sampled_bytes+=compare(loader,shared_name,0,down,shared);
            }
            std::cout << std::endl;
        }
        std::cout << "width=" << width << " verified_native_bytes=" << sampled_bytes << std::endl;
    }
} catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
