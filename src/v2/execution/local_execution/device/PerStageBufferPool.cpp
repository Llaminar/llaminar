/**
 * @file PerStageBufferPool.cpp
 * @brief Implementation of per-PP-stage buffer allocation
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "PerStageBufferPool.h"
#include "../../../tensors/TensorFactory.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/Logger.h"
#include "../../../utils/MPIContext.h"
#include <stdexcept>

namespace llaminar2
{

PerStageBufferPool::~PerStageBufferPool()
{
    release();
}

bool PerStageBufferPool::initialize(const PipelineConfig &config, const PPStageBufferSpec &spec,
                                    const MPIContext *mpi_ctx)
{
    // Release any existing allocations
    release();

    config_ = &config;
    spec_ = spec;
    stats_.reset();

    // Create tensor factory
    if (mpi_ctx)
    {
        tensor_factory_ = std::make_unique<TensorFactory>(*mpi_ctx);
    }
    else
    {
        // Create a minimal MPI context for single-rank operation
        static MPIContext single_rank_ctx(0, 1);
        tensor_factory_ = std::make_unique<TensorFactory>(single_rank_ctx);
    }

    LOG_DEBUG("[PerStageBufferPool] Initializing for " << config.numStages()
                                                       << " PP stages, batch=" << spec.batch_size
                                                       << " seq=" << spec.seq_len);

    // Allocate buffers for each stage
    for (const auto &stage : config.pp_stages)
    {
        const TPDomainConfig *domain = config.getDomainForStage(stage.stage_id);
        if (!domain)
        {
            LOG_ERROR("[PerStageBufferPool] No domain found for stage " << stage.stage_id);
            release();
            return false;
        }

        DeviceId device = domain->primaryDevice();
        LOG_DEBUG("[PerStageBufferPool] Stage " << stage.stage_id
                                                << " on device " << device.to_string());

        if (!allocateStageBuffers(stage.stage_id, device, spec))
        {
            LOG_ERROR("[PerStageBufferPool] Failed to allocate buffers for stage "
                      << stage.stage_id);
            release();
            return false;
        }
    }

    LOG_INFO("[PerStageBufferPool] Initialized " << stage_buffers_.size()
                                                 << " stage buffer pools, total "
                                                 << stats_.total_bytes() << " bytes");
    return true;
}

ActivationBuffers &PerStageBufferPool::forStage(int stage_id)
{
    auto it = stage_buffers_.find(stage_id);
    if (it == stage_buffers_.end())
    {
        throw std::out_of_range("PerStageBufferPool::forStage: invalid stage_id " +
                                std::to_string(stage_id));
    }
    return it->second;
}

const ActivationBuffers &PerStageBufferPool::forStage(int stage_id) const
{
    auto it = stage_buffers_.find(stage_id);
    if (it == stage_buffers_.end())
    {
        throw std::out_of_range("PerStageBufferPool::forStage: invalid stage_id " +
                                std::to_string(stage_id));
    }
    return it->second;
}

ActivationBuffers &PerStageBufferPool::forLayer(int layer_idx)
{
    if (!config_)
    {
        throw std::runtime_error("PerStageBufferPool::forLayer: not initialized");
    }
    int stage_id = config_->getStageIdForLayer(layer_idx);
    if (stage_id < 0)
    {
        throw std::out_of_range("PerStageBufferPool::forLayer: layer " +
                                std::to_string(layer_idx) + " not covered by any stage");
    }
    return forStage(stage_id);
}

const ActivationBuffers &PerStageBufferPool::forLayer(int layer_idx) const
{
    if (!config_)
    {
        throw std::runtime_error("PerStageBufferPool::forLayer: not initialized");
    }
    int stage_id = config_->getStageIdForLayer(layer_idx);
    if (stage_id < 0)
    {
        throw std::out_of_range("PerStageBufferPool::forLayer: layer " +
                                std::to_string(layer_idx) + " not covered by any stage");
    }
    return forStage(stage_id);
}

DeviceId PerStageBufferPool::deviceForStage(int stage_id) const
{
    if (!config_)
    {
        throw std::runtime_error("PerStageBufferPool::deviceForStage: not initialized");
    }
    const TPDomainConfig *domain = config_->getDomainForStage(stage_id);
    if (!domain)
    {
        throw std::out_of_range("PerStageBufferPool::deviceForStage: invalid stage_id " +
                                std::to_string(stage_id));
    }
    return domain->primaryDevice();
}

int PerStageBufferPool::numStages() const
{
    return config_ ? config_->numStages() : 0;
}

void PerStageBufferPool::release()
{
    // Clear buffers first (they hold raw pointers to tensors)
    stage_buffers_.clear();
    // Then release tensors
    tensor_storage_.clear();
    config_ = nullptr;
    tensor_factory_.reset();
    stats_.reset();
    LOG_DEBUG("[PerStageBufferPool] Released all buffers");
}

bool PerStageBufferPool::allocateStageBuffers(int stage_id, DeviceId device,
                                              const PPStageBufferSpec &spec)
{
    auto &storage = tensor_storage_[stage_id];
    auto &buffers = stage_buffers_[stage_id];

    // Helper to allocate and track a tensor
    auto alloc = [&](const std::vector<size_t> &shape, TensorType dtype) -> TensorBase *
    {
        auto tensor = createTensor(device, shape, dtype);
        if (!tensor)
            return nullptr;
        TensorBase *ptr = tensor.get();
        storage.push_back(std::move(tensor));

        // Update stats
        size_t bytes = ptr->size_bytes();
        stats_.bytes_per_device[device] += bytes;
        stats_.buffers_per_device[device]++;
        if (device.is_gpu())
        {
            stats_.gpu_bytes_allocated += bytes;
            stats_.gpu_buffer_count++;
        }
        else
        {
            stats_.cpu_bytes_allocated += bytes;
            stats_.cpu_buffer_count++;
        }
        return ptr;
    };

    // Determine data types based on precision
    TensorType activation_dtype = (spec.precision == ActivationPrecision::HybridQ16)
                                      ? TensorType::Q16_1
                                      : TensorType::FP32;
    TensorType fp32_dtype = TensorType::FP32;

    size_t tokens = spec.total_tokens();
    size_t d_model = spec.d_model;
    size_t n_heads = spec.n_heads;
    size_t head_dim = spec.head_dim;
    size_t kv_dim = spec.kv_dim();
    size_t intermediate = spec.intermediate_size;

    // Allocate core activation buffers
    buffers.residual = alloc({tokens, d_model}, activation_dtype);
    buffers.normalized = alloc({tokens, d_model}, fp32_dtype);
    buffers.current_hidden = alloc({tokens, d_model}, fp32_dtype);

    // QKV buffers
    buffers.Q = alloc({tokens, n_heads * head_dim}, fp32_dtype);
    buffers.K = alloc({tokens, kv_dim}, fp32_dtype);
    buffers.V = alloc({tokens, kv_dim}, fp32_dtype);

    // Attention output
    buffers.attn_output = alloc({tokens, n_heads * head_dim}, fp32_dtype);
    buffers.attn_proj = alloc({tokens, d_model}, fp32_dtype);

    // FFN buffers
    buffers.gate = alloc({tokens, intermediate}, fp32_dtype);
    buffers.up = alloc({tokens, intermediate}, fp32_dtype);
    buffers.ffn_output = alloc({tokens, d_model}, fp32_dtype);

    // Workspace buffers for attention
    // Note: workspace sizes depend on implementation, using conservative estimates
    buffers.workspace_scores = alloc({tokens, tokens}, fp32_dtype);   // seq x seq for scores
    buffers.workspace_context = alloc({tokens, n_heads * head_dim}, fp32_dtype);

    // Hybrid mode buffers (Q/K after RoPE)
    if (spec.precision == ActivationPrecision::HybridQ16 ||
        spec.precision == ActivationPrecision::Hybrid)
    {
        buffers.Q_rope = alloc({tokens, n_heads * head_dim}, fp32_dtype);
        buffers.K_rope = alloc({tokens, kv_dim}, fp32_dtype);
        buffers.V_dequant = alloc({tokens, kv_dim}, fp32_dtype);
    }

    // Snapshot buffers (optional)
    if (spec.enable_snapshots)
    {
        buffers.context_snapshot = alloc({tokens, n_heads * head_dim}, fp32_dtype);
        buffers.attention_output_snapshot = alloc({tokens, d_model}, fp32_dtype);
        buffers.attention_residual_snapshot = alloc({tokens, d_model}, fp32_dtype);
    }

    LOG_DEBUG("[PerStageBufferPool] Allocated " << storage.size()
                                                << " tensors for stage " << stage_id
                                                << " on " << device.to_string());

    return true;
}

std::unique_ptr<TensorBase> PerStageBufferPool::createTensor(
    DeviceId device,
    const std::vector<size_t> &shape,
    TensorType dtype)
{
    if (!tensor_factory_)
    {
        LOG_ERROR("[PerStageBufferPool::createTensor] TensorFactory not initialized");
        return nullptr;
    }

    // Use TensorFactory for device-aware allocation
    switch (dtype)
    {
    case TensorType::FP32:
        return tensor_factory_->createFP32(shape, device);

    case TensorType::Q8_1:
        return tensor_factory_->createQ8_1(shape, device);

    case TensorType::Q16_1:
        return tensor_factory_->createQ16_1(shape, device);

    case TensorType::FP16:
        return tensor_factory_->createFP16(shape);

    case TensorType::BF16:
        return tensor_factory_->createBF16(shape);

    default:
        LOG_ERROR("[PerStageBufferPool::createTensor] Unsupported dtype: "
                  << tensorTypeName(dtype));
        return nullptr;
    }
}

} // namespace llaminar2
