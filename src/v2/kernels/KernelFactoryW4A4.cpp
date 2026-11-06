/**
 * @file KernelFactoryW4A4.cpp
 * @brief Bind a sealed W4A4/H source to its native CPU arithmetic implementation.
 *
 * This factory never decodes floating weights or selects a weight-only VNNI
 * engine. The source tensor and prepared engine share the original immutable
 * physical owner. GPU preparation belongs to the native device load pipeline,
 * as for other weight tensors; a CPU source factory cannot select that path.
 */
#include "KernelFactory.h"
#include "cpu/gemm/CPUW4A4GemmKernel.h"
#include "../tensors/W4A4Tensor.h"

namespace llaminar::v2::kernels
{
    std::unique_ptr<llaminar2::ITensorGemm> KernelFactory::createGemm(
        const llaminar2::W4A4Tensor *tensor, DeviceType device, int device_ordinal)
    {
        if (!tensor) throw std::invalid_argument("W4A4 GEMM factory requires its native tensor authority");
        if (device != DeviceType::CPU || device_ordinal != -1)
            throw std::invalid_argument("W4A4 source GEMM factory is CPU-only; GPU weights require the native load pipeline");
        return std::make_unique<llaminar2::w4a4::CPUW4A4GemmKernel>(tensor->retainWeightStorage());
    }
}
