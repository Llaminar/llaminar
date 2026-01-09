/**
 * @file TensorFactory.cpp
 * @brief Factory implementation for creating tensors with NUMA-aware allocation
 * @author David Sanftenberg
 */

#include "TensorFactory.h"
#include "BlockStructures.h"
#include "../utils/Logger.h"
#include <stdexcept>
#include <sstream>

#ifdef HAVE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

namespace llaminar2
{
    TensorFactory::TensorFactory(const MPIContext &mpi_ctx)
        : mpi_rank_(mpi_ctx.rank()), numa_node_(-1)
    {
        // Determine NUMA node for this MPI rank
        numa_node_ = getNumaNodeForRank(mpi_rank_);

        // Bind current thread to NUMA node if available
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }
    }

    std::unique_ptr<FP32Tensor> TensorFactory::createFP32(const std::vector<size_t> &shape, DeviceId device)
    {
        // Ensure we're on the correct NUMA node
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        return std::make_unique<FP32Tensor>(shape, device);
    }

    std::unique_ptr<FP16Tensor> TensorFactory::createFP16(const std::vector<size_t> &shape)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        return std::make_unique<FP16Tensor>(shape);
    }

    std::unique_ptr<FP16Tensor> TensorFactory::createFP16(const std::vector<size_t> &shape,
                                                          const std::vector<uint16_t> &fp16_data)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        return std::make_unique<FP16Tensor>(shape, fp16_data);
    }

    std::unique_ptr<BF16Tensor> TensorFactory::createBF16(const std::vector<size_t> &shape)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        return std::make_unique<BF16Tensor>(shape);
    }

    std::unique_ptr<BF16Tensor> TensorFactory::createBF16(const std::vector<size_t> &shape,
                                                          const std::vector<uint16_t> &bf16_data)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        return std::make_unique<BF16Tensor>(shape, bf16_data);
    }

    std::unique_ptr<INT32Tensor> TensorFactory::createINT32(const std::vector<size_t> &shape)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        return std::make_unique<INT32Tensor>(shape);
    }

    std::unique_ptr<Q8_1Tensor> TensorFactory::createQ8_1(const std::vector<size_t> &shape, DeviceId device)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        // Use the mutable activation buffer constructor (no raw_data, allocates internally)
        // This creates a Q8_1Tensor that supports mutable_data() and quantize_from_cache()
        return std::make_unique<Q8_1Tensor>(shape, device.toLegacyIndex());
    }

    std::unique_ptr<Q8_1Tensor> TensorFactory::createQ8_1(const std::vector<size_t> &shape,
                                                          const std::vector<uint8_t> &raw_data)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        return std::make_unique<Q8_1Tensor>(shape, raw_data);
    }

    std::unique_ptr<Q16_1Tensor> TensorFactory::createQ16_1(const std::vector<size_t> &shape,
                                                            DeviceId device)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        // Use the mutable activation buffer constructor
        // Q16_1Tensor(shape, device) creates an empty tensor for residual accumulation
        return std::make_unique<Q16_1Tensor>(shape, device.toLegacyIndex());
    }

    std::unique_ptr<Q16_1Tensor> TensorFactory::createQ16_1(const std::vector<size_t> &shape,
                                                            Q16BlockSize block_size,
                                                            DeviceId device)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        // Use the mutable activation buffer constructor with custom block size
        return std::make_unique<Q16_1Tensor>(shape, block_size, device.toLegacyIndex());
    }

    std::unique_ptr<CPUTensorBase> TensorFactory::createActivation(const std::vector<size_t> &shape,
                                                                   ActivationPrecision precision,
                                                                   DeviceId device)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        std::unique_ptr<CPUTensorBase> tensor;

        switch (precision)
        {
        case ActivationPrecision::FP32:
            // FP32 createFP32 already accepts device
            return createFP32(shape, device);

        case ActivationPrecision::BF16:
            tensor = createBF16(shape);
            break;

        case ActivationPrecision::FP16:
            tensor = createFP16(shape);
            break;

        case ActivationPrecision::Q8_1:
            tensor = createQ8_1(shape);
            break;

        case ActivationPrecision::Q16_1:
            // Q16_1: High-precision quantized format for residual stream
            return createQ16_1(shape, device);

        case ActivationPrecision::HybridQ16:
            // For HybridQ16 mode, createActivation returns Q16_1 for residual buffers
            // Buffer allocation logic in GraphOrchestrator handles specific buffer types
            return createQ16_1(shape, device);

        default:
            LOG_ERROR("TensorFactory::createActivation: unknown precision, defaulting to FP32");
            return createFP32(shape, device);
        }

        // Set device on the created tensor to ensure consistent device tracking
        // This is critical for pipelines that use placement maps to route tensors
        // between devices. Without this, Q8_1/BF16/FP16 tensors would have device=-1
        // even when they should be associated with device 0 (CPU), causing spurious
        // "device transfer" attempts in prepareActivationForDevice().
        if (tensor && device.is_valid())
        {
            tensor->set_device(device.toLegacyIndex());
        }

        return tensor;
    }

    std::unique_ptr<CPUTensorBase> TensorFactory::createActivation(const std::vector<size_t> &shape,
                                                                   ActivationPrecision precision,
                                                                   int head_dim,
                                                                   DeviceId device)
    {
        // For Q16_1 and HybridQ16, use optimal block size based on head_dim
        if (precision == ActivationPrecision::Q16_1 ||
            precision == ActivationPrecision::HybridQ16)
        {
            Q16BlockSize block_size = optimal_q16_block_size(head_dim);
            LOG_DEBUG("TensorFactory::createActivation: Q16 with head_dim=" << head_dim
                                                                            << " -> block_size=" << static_cast<int>(block_size));
            return createQ16_1(shape, block_size, device);
        }

        // All other precisions: delegate to base overload
        return createActivation(shape, precision, device);
    }

    std::unique_ptr<CPUTensorBase> TensorFactory::createQuantized(TensorType type,
                                                                  const std::vector<size_t> &shape,
                                                                  const std::vector<uint8_t> &raw_data)
    {
        if (numa_node_ >= 0)
        {
            bindToNumaNode();
        }

        switch (type)
        {
        case TensorType::IQ4_NL:
            return std::make_unique<IQ4_NLTensor>(shape, raw_data);
        case TensorType::Q8_0:
            return std::make_unique<Q8_0Tensor>(shape, raw_data);
        case TensorType::Q4_0:
            return std::make_unique<Q4_0Tensor>(shape, raw_data);
        case TensorType::Q4_1:
            return std::make_unique<Q4_1Tensor>(shape, raw_data);
        case TensorType::Q5_0:
            return std::make_unique<Q5_0Tensor>(shape, raw_data);
        case TensorType::Q5_1:
            return std::make_unique<Q5_1Tensor>(shape, raw_data);
        case TensorType::Q6_K:
            return std::make_unique<Q6_KTensor>(shape, raw_data);
        case TensorType::Q2_K:
            return std::make_unique<Q2_KTensor>(shape, raw_data);
        case TensorType::Q5_K:
            return std::make_unique<Q5_KTensor>(shape, raw_data);
        case TensorType::Q3_K:
            return std::make_unique<Q3_KTensor>(shape, raw_data);
        case TensorType::Q4_K:
            return std::make_unique<Q4_KTensor>(shape, raw_data);
        case TensorType::Q8_K:
            return std::make_unique<Q8_KTensor>(shape, raw_data);
        case TensorType::IQ4_XS:
            return std::make_unique<IQ4_XSTensor>(shape, raw_data);
        case TensorType::IQ2_XXS:
            return std::make_unique<IQ2_XXSTensor>(shape, raw_data);
        case TensorType::IQ2_XS:
            return std::make_unique<IQ2_XSTensor>(shape, raw_data);
        case TensorType::IQ3_XXS:
            return std::make_unique<IQ3_XXSTensor>(shape, raw_data);
        case TensorType::IQ2_S:
            return std::make_unique<IQ2_STensor>(shape, raw_data);
        case TensorType::IQ3_S:
            return std::make_unique<IQ3_STensor>(shape, raw_data);
        case TensorType::IQ1_S:
            return std::make_unique<IQ1_STensor>(shape, raw_data);
        case TensorType::IQ1_M:
            return std::make_unique<IQ1_MTensor>(shape, raw_data);
        default:
            LOG_ERROR("TensorFactory::createQuantized: unsupported type " << static_cast<int>(type));
            std::ostringstream oss;
            oss << "TensorFactory::createQuantized: unsupported type " << static_cast<int>(type);
            throw std::runtime_error(oss.str());
        }
    }

    bool TensorFactory::isNumaAvailable()
    {
#ifdef HAVE_NUMA
        if (numa_available() < 0)
        {
            return false;
        }

        // Check if system has multiple NUMA nodes
        int max_node = numa_max_node();
        return max_node > 0;
#else
        return false;
#endif
    }

    void TensorFactory::bindToNumaNode()
    {
#ifdef HAVE_NUMA
        if (numa_node_ >= 0 && numa_available() >= 0)
        {
            // Bind current thread to NUMA node
            struct bitmask *mask = numa_allocate_nodemask();
            numa_bitmask_clearall(mask);
            numa_bitmask_setbit(mask, numa_node_);

            // Set memory binding policy for allocations
            numa_set_membind(mask);

            // Also bind CPU affinity (optional, helps with first-touch)
            // numa_run_on_node_mask(mask);

            numa_free_nodemask(mask);
        }
#endif
    }

    int TensorFactory::getNumaNodeForRank(int rank)
    {
#ifdef HAVE_NUMA
        if (numa_available() < 0)
        {
            return -1;
        }

        int max_node = numa_max_node();
        if (max_node <= 0)
        {
            return -1; // Only one NUMA node
        }

        // Simple round-robin mapping: rank % (max_node + 1)
        // More sophisticated mapping could query actual CPU topology
        return rank % (max_node + 1);
#else
        (void)rank; // Suppress unused parameter warning
        return -1;
#endif
    }

} // namespace llaminar2
