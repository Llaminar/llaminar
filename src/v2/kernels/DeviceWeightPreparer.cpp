/**
 * @file DeviceWeightPreparer.cpp
 * @brief Implementation of device-targeted weight preparation for PP stages
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#include "DeviceWeightPreparer.h"
#include "KernelFactory.h"
#include "../tensors/ITensor.h"
#include "../utils/Logger.h"

namespace llaminar2
{

    WeightPreparationResult DeviceWeightPreparer::prepare(
        const WeightViewSet &views,
        DeviceId target_device)
    {
        WeightPreparationResult result;

        LOG_DEBUG("[DeviceWeightPreparer] Preparing " << views.size()
                                                      << " weights for " << target_device.to_string());

        for (const auto &view : views)
        {
            if (!view.tensor)
            {
                LOG_WARN("[DeviceWeightPreparer] Null tensor for weight: " << view.name);
                result.failures++;
                result.failed_names.push_back(view.name);
                continue;
            }

            if (view.is_gemm_weight)
            {
                // GEMM weights: create backend-specific prepared representation
                if (!prepareGemmWeight(view, target_device))
                {
                    result.failures++;
                    result.failed_names.push_back(view.name);
                    continue;
                }
                result.gemm_weights_prepared++;
                result.total_device_bytes += view.tensor->size_bytes();
            }
            else
            {
                // Non-GEMM weights (norms, biases, embeddings): upload to device
                if (!prepareNonGemmWeight(view, target_device))
                {
                    result.failures++;
                    result.failed_names.push_back(view.name);
                    continue;
                }
                result.non_gemm_weights_uploaded++;
                result.total_device_bytes += view.tensor->size_bytes();
            }
        }

        LOG_INFO("[DeviceWeightPreparer] Prepared for " << target_device.to_string()
                                                        << ": " << result.gemm_weights_prepared << " GEMM"
                                                        << " + " << result.non_gemm_weights_uploaded << " non-GEMM"
                                                        << " (" << (result.total_device_bytes / (1024 * 1024)) << " MB)"
                                                        << (result.failures > 0
                                                                ? " — " + std::to_string(result.failures) + " FAILED"
                                                                : ""));

        return result;
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    bool DeviceWeightPreparer::prepareGemmWeight(
        const WeightView &view,
        DeviceId target_device)
    {
        try
        {
            const auto *handle = ::llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(
                view.tensor.get(), target_device);

            if (!handle)
            {
                LOG_ERROR("[DeviceWeightPreparer] Failed to prepare GEMM weight: " << view.name
                                                                                   << " for " << target_device.to_string());
                return false;
            }

            LOG_TRACE("[DeviceWeightPreparer] Prepared GEMM: " << view.name
                                                               << " → " << target_device.to_string());
            return true;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[DeviceWeightPreparer] Exception preparing GEMM weight '"
                      << view.name << "': " << e.what());
            return false;
        }
    }

    bool DeviceWeightPreparer::prepareNonGemmWeight(
        const WeightView &view,
        DeviceId target_device)
    {
        // CPU target: weights are already on host, nothing to do
        if (target_device.is_cpu())
        {
            LOG_TRACE("[DeviceWeightPreparer] Non-GEMM weight " << view.name
                                                                << " already on CPU, skipping");
            return true;
        }

        // GPU target: upload via ensureOnDevice
        try
        {
            // Cast to ITensor for coherence API (TensorBase → ITensor)
            auto *itensor = dynamic_cast<ITensor *>(view.tensor.get());
            if (!itensor)
            {
                LOG_ERROR("[DeviceWeightPreparer] Weight '" << view.name
                                                            << "' does not implement ITensor interface");
                return false;
            }

            bool ok = itensor->ensureOnDevice(target_device);
            if (!ok)
            {
                LOG_ERROR("[DeviceWeightPreparer] Failed to upload non-GEMM weight '"
                          << view.name << "' to " << target_device.to_string());
                return false;
            }

            LOG_TRACE("[DeviceWeightPreparer] Uploaded non-GEMM: " << view.name
                                                                   << " → " << target_device.to_string());
            return true;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[DeviceWeightPreparer] Exception uploading non-GEMM weight '"
                      << view.name << "': " << e.what());
            return false;
        }
    }

} // namespace llaminar2
