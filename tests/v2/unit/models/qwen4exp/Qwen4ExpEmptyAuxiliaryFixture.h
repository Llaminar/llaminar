/**
 * @file Qwen4ExpEmptyAuxiliaryFixture.h
 * @brief Exact metadata/cache fixtures for absent participant-local QSA/PLE state.
 *
 * The authenticated source owns PLE placement and the heterogeneous mixer map.
 * These model-free fixtures preserve both; no layer or GDN geometry is invented.
 */
#pragma once
#include "Qwen4ExpMetadataFixture.h"
#include "models/qwen4exp/Qwen4ExpGraphConfigBuilder.h"
#include "models/GraphTypes.h"
#include "kernels/KernelFactory.h"
#include "kernels/HybridKVCacheConfig.h"
#include <stdexcept>
#include <vector>

namespace llaminar2::test
{
/** @brief Project an exact main-model PP interval without constructing model weights. */
inline GraphConfig emptyAuxiliaryConfig(DeviceId device,int first,int count=1)
{
    GraphConfig config;
    config.default_device=device;config.pp_layer_offset=first;config.n_layers=count;
    config.max_seq_len=8;config.max_request_count=1;config.max_activation_rows=1;
    Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(false),config);
    config.local_n_heads=config.n_heads;config.local_n_kv_heads=config.n_kv_heads;
    return config;
}
/** @brief Find every source-declared recurrent layer without lexical ownership. */
inline std::vector<int> emptyAuxiliaryLayers()
{
    const Qwen4ExpConfig source(qwen4MetadataFixture(false));
    std::vector<int> result;
    for(int layer=0;layer<static_cast<int>(source.mainLayers());++layer)
        if(source.mixers()[layer]==Qwen4ExpMixer::GatedDeltaNet &&
            layer!=static_cast<int>(source.ngrams().decoder_layer))result.push_back(layer);
    if(result.empty())throw std::logic_error("Exact fixture has no GDN-only participant interval");
    return result;
}
/** @brief Project the exact cache-owned recurrent geometry from the authenticated graph interval. */
inline HybridKVCacheConfig emptyAuxiliaryHybridConfig(const GraphConfig &config)
{
    HybridKVCacheConfig hybrid;
    hybrid.first_layer_index=config.pp_layer_offset;
    hybrid.layer_types.assign(config.layer_types.begin()+config.pp_layer_offset,
        config.layer_types.begin()+config.pp_layer_offset+config.n_layers);
    hybrid.gdn_conv_kernel_size=config.gdn.conv_kernel_size;
    hybrid.gdn_state_size=config.gdn.state_size;hybrid.gdn_inner_size=config.gdn.inner_size;
    hybrid.gdn_group_count=config.gdn.group_count;hybrid.gdn_time_step_rank=config.gdn.time_step_rank;
    hybrid.n_heads=config.n_heads;hybrid.local_n_heads=config.local_n_heads;
    return hybrid;
}
/**
 * @brief Build the production cache descriptor before admission or materialization.
 * @param config Exact graph geometry and device.
 * @param mpi Fixture rank context retained through cache destruction.
 * @param hybrid Borrowed exact geometry retained through the factory invocation.
 * @return Descriptor whose estimateBytes() owns KV/frontier accounting.
 */
inline llaminar::v2::kernels::KVCacheConfig emptyAuxiliaryKVConfig(
    const GraphConfig &config,const IMPIContext &mpi,const HybridKVCacheConfig &hybrid)
{
    llaminar::v2::kernels::KVCacheConfig cache;
    cache.precision=ActivationPrecision::FP32;cache.device=config.default_device;
    cache.num_layers=config.n_layers;cache.first_layer_index=config.pp_layer_offset;
    cache.batch_size=config.max_request_count;cache.max_seq_len=config.max_seq_len;
    cache.n_kv_heads=config.n_kv_heads;cache.head_dim=config.head_dim;
    cache.mpi_ctx=&mpi;cache.hybrid_config=&hybrid;
    return cache;
}
/** @brief Construct a standalone CPU fixture cache using the same exact geometry projection. */
inline std::unique_ptr<IKVCache> emptyAuxiliaryCache(const GraphConfig &config,const IMPIContext &mpi)
{
    const auto hybrid=emptyAuxiliaryHybridConfig(config);
    return llaminar::v2::kernels::KernelFactory::createKVCache(emptyAuxiliaryKVConfig(config,mpi,hybrid));
}
} // namespace llaminar2::test
