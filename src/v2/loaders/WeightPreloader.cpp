/**
 * @file WeightPreloader.cpp
 * @brief Implementation of weight pre-packing before graph execution
 */

#include "WeightPreloader.h"
#include "../kernels/KernelFactory.h"
#include "../utils/Logger.h"
#include "../backends/ComputeBackend.h"
#include "../backends/DeviceId.h"

#ifdef HAVE_ROCM
#include "../kernels/rocm/ROCmQuantisedGemmKernel.h"
#endif

namespace llaminar2
{

    WeightPreloader::WeightPreloader(
        std::shared_ptr<WeightManager> weight_manager,
        std::shared_ptr<WeightPlacementMap> placement_map)
        : weight_manager_(std::move(weight_manager)), placement_map_(std::move(placement_map))
    {
        if (!weight_manager_)
        {
            throw std::invalid_argument("WeightPreloader: weight_manager cannot be null");
        }
    }

    bool WeightPreloader::preloadAll(
        PreloadProgressCallback progress_callback,
        bool release_raw_data)
    {
        // Get all cached weight names from the weight manager
        // Note: We iterate the cache, so weights must already be loaded
        auto &cache = weight_manager_->cache_;
        if (cache.empty())
        {
            LOG_WARN("[WeightPreloader] No weights loaded in cache - nothing to preload");
            return true;
        }

        std::vector<std::string> weight_names;
        weight_names.reserve(cache.size());
        for (const auto &[name, tensor] : cache)
        {
            weight_names.push_back(name);
        }

        return preload(weight_names, progress_callback, release_raw_data);
    }

    bool WeightPreloader::preload(
        const std::vector<std::string> &weight_names,
        PreloadProgressCallback progress_callback,
        bool release_raw_data)
    {
        size_t total = weight_names.size();
        size_t current = 0;
        bool all_success = true;

        LOG_DEBUG("[WeightPreloader] Preloading " << total << " weights...");

        for (const auto &name : weight_names)
        {
            current++;

            // Get the tensor from cache
            auto tensor = weight_manager_->getWeight(name);
            if (!tensor)
            {
                LOG_WARN("[WeightPreloader] Weight not found: " << name);
                continue;
            }

            // Determine target device (now returns full DeviceId with ordinal)
            DeviceId target = getTargetDevice(name);

            // Report progress
            if (progress_callback)
            {
                progress_callback(current, total, name);
            }

            // Skip non-GEMM weights (embeddings, norms, biases)
            // These don't need packing
            if (!weight_manager_->isGemmWeight(name))
            {
                LOG_TRACE("[WeightPreloader] Skipping non-GEMM weight: " << name);
                continue;
            }

            // Pack the weight
            if (!packWeight(tensor.get(), target, release_raw_data))
            {
                LOG_ERROR("[WeightPreloader] Failed to pack weight: " << name);
                all_success = false;
            }
        }

        LOG_DEBUG("[WeightPreloader] Preloading complete: "
                  << num_cpu_packed_ << " CPU, " << num_gpu_packed_ << " GPU");

        return all_success;
    }

    bool WeightPreloader::preloadWithOverrideDevice(
        const std::vector<std::string> &weight_names,
        DeviceId override_device,
        PreloadProgressCallback progress_callback,
        bool release_raw_data)
    {
        size_t total = weight_names.size();
        size_t current = 0;
        bool all_success = true;

        LOG_DEBUG("[WeightPreloader] Preloading " << total << " weights to "
                                                  << override_device.to_string() << "...");

        for (const auto &name : weight_names)
        {
            current++;

            // Get the tensor from cache
            auto tensor = weight_manager_->getWeight(name);
            if (!tensor)
            {
                LOG_WARN("[WeightPreloader] Weight not found: " << name);
                continue;
            }

            // Report progress
            if (progress_callback)
            {
                progress_callback(current, total, name);
            }

            // Pack the weight to the override device
            if (!packWeight(tensor.get(), override_device, release_raw_data))
            {
                LOG_ERROR("[WeightPreloader] Failed to pack weight: " << name);
                all_success = false;
            }
        }

        LOG_DEBUG("[WeightPreloader] Preloading complete: "
                  << num_cpu_packed_ << " CPU, " << num_gpu_packed_ << " GPU");

        return all_success;
    }

    bool WeightPreloader::preloadForDevice(
        DeviceType target_device,
        PreloadProgressCallback progress_callback,
        bool release_raw_data)
    {
        auto &cache = weight_manager_->cache_;

        std::vector<std::string> matching_names;
        for (const auto &[name, tensor] : cache)
        {
            // When no placement map exists, load ALL GEMM weights to the target device
            // This is the common case for single-device inference
            if (!placement_map_)
            {
                // Only include GEMM weights (skip embeddings, norms, etc.)
                if (weight_manager_->isGemmWeight(name))
                {
                    matching_names.push_back(name);
                }
            }
            else if (getTargetDevice(name).type == target_device)
            {
                // Filter by device type when placement map exists
                matching_names.push_back(name);
            }
        }

        // Override target device in the preload call when no placement map exists
        // This ensures weights are packed for the actual target device, not CPU
        // Convert DeviceType to DeviceId (ordinal 0 by default since we don't know better)
        if (!placement_map_ && !matching_names.empty())
        {
            LOG_DEBUG("[WeightPreloader] No placement map - preloading all "
                      << matching_names.size() << " GEMM weights to target device");
            // Convert DeviceType to DeviceId with ordinal 0 (legacy path)
            DeviceId target_id = (target_device == DeviceType::CPU) ? DeviceId::cpu() : (target_device == DeviceType::CUDA) ? DeviceId::cuda(0)
                                                                                                                            : DeviceId::rocm(0);
            return preloadWithOverrideDevice(matching_names, target_id, progress_callback, release_raw_data);
        }

        return preload(matching_names, progress_callback, release_raw_data);
    }

    bool WeightPreloader::preloadForDevice(
        DeviceId target_device,
        PreloadProgressCallback progress_callback,
        bool release_raw_data)
    {
        // Use the KernelFactory's thread-local mechanism to pass the device ordinal
        // to kernel creation calls within this scope
        using namespace llaminar::v2::kernels;

        DeviceType device_type = target_device.type;

        // For ROCm devices, set the thread-local ordinal so kernels are created
        // on the correct device in multi-GPU setups
        if (target_device.is_rocm())
        {
            LOG_DEBUG("[WeightPreloader] Setting ROCm ordinal " << target_device.ordinal
                                                                << " for weight preloading");
            // Create guard - will be active for entire preload operation
            KernelFactory::ROCmOrdinalGuard guard(target_device.ordinal);

            // Now all kernel creation within this call will use the correct ordinal
            auto &cache = weight_manager_->cache_;
            std::vector<std::string> matching_names;
            for (const auto &[name, tensor] : cache)
            {
                if (!placement_map_)
                {
                    if (weight_manager_->isGemmWeight(name))
                    {
                        matching_names.push_back(name);
                    }
                }
                else if (getTargetDevice(name).type == device_type)
                {
                    // Filter by device type when placement map exists
                    matching_names.push_back(name);
                }
            }

            if (!placement_map_ && !matching_names.empty())
            {
                LOG_DEBUG("[WeightPreloader] No placement map - preloading all "
                          << matching_names.size() << " GEMM weights to ROCm device "
                          << target_device.ordinal);
                return preloadWithOverrideDevice(matching_names, target_device, progress_callback, release_raw_data);
            }
            return preload(matching_names, progress_callback, release_raw_data);
        }
        else
        {
            // For CPU and CUDA, just delegate to DeviceType version
            return preloadForDevice(device_type, progress_callback, release_raw_data);
        }
    }

    DeviceId WeightPreloader::getTargetDevice(const std::string &weight_name) const
    {
        if (!placement_map_)
        {
            return DeviceId::cpu(); // Default to CPU
        }

        DeviceId device_id = placement_map_->getDeviceForWeight(weight_name);
        if (!device_id.is_valid())
        {
            return DeviceId::cpu();
        }

        return device_id;
    }

    bool WeightPreloader::packWeight(
        TensorBase *tensor,
        DeviceId target_device,
        bool release_raw_data)
    {
        if (!tensor)
        {
            return false;
        }

        // Use device-targeted kernel creation API
        // This ensures the kernel is created for the correct device from the start
        using namespace llaminar::v2::kernels;

        // Create kernel for target device (this also packs weights appropriately)
        // Use the DeviceId overload to preserve device ordinal for multi-GPU
        auto *kernel = KernelFactory::getOrCreateGemm(tensor, target_device);
        bool success = (kernel != nullptr);

        if (success)
        {
            if (target_device.is_cpu())
            {
                num_cpu_packed_++;

                // For CPU, we can release raw data since packed weights are in CPU cache
                if (release_raw_data)
                {
                    tensor->release_raw_data();
                    LOG_TRACE("[WeightPreloader] Released raw data for: " << tensor->shape()[0]
                                                                          << "x" << tensor->shape()[1]);
                }
            }
            else
            {
                num_gpu_packed_++;

#ifdef HAVE_ROCM
                // For ROCm, force upload to device NOW to avoid hipMalloc during inference
                if (target_device.is_rocm())
                {
                    auto *rocm_kernel = dynamic_cast<llaminar2::rocm::ROCmQuantisedGemmKernel *>(kernel);
                    if (rocm_kernel)
                    {
                        LOG_TRACE("[WeightPreloader] Calling ensureWeightsConverted for ROCm tensor="
                                  << (void *)tensor << " kernel=" << (void *)kernel
                                  << " shape=" << tensor->shape()[0] << "x" << tensor->shape()[1]);
                        rocm_kernel->ensureWeightsConverted();
                        LOG_TRACE("[WeightPreloader] Uploaded ROCm weights: "
                                  << tensor->shape()[0] << "x" << tensor->shape()[1]);
                    }
                    else
                    {
                        LOG_WARN("[WeightPreloader] Cast to ROCmQuantisedGemmKernel failed!");
                    }
                }
#endif

                // For GPU, do NOT release raw data!
                // The tensor coherence system (ensureOnDevice) still needs the host data
                // to upload to GPU buffers during stage execution.
                LOG_TRACE("[WeightPreloader] GPU weight packed (keeping raw data): "
                          << tensor->shape()[0] << "x" << tensor->shape()[1]);
            }
        }

        return success;
    }

} // namespace llaminar2
