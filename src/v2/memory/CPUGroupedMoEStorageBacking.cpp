/**
 * @file CPUGroupedMoEStorageBacking.cpp
 * @brief Native backing construction and retirement for grouped CPU scratch.
 *
 * All storage is materialized from one checked immutable declaration. Member
 * RAII rolls back partial construction, and tensor/vector ownership retains the
 * exact physical claims beyond workspace-manager or graph metadata retirement.
 */
#include "CPUGroupedMoEStorageBacking.h"
#include "tensors/TensorClasses.h"
#include <cstring>

namespace llaminar2
{
CPUGroupedMoEStorageBacking::CPUGroupedMoEStorageBacking(
    const CPUGroupedMoEStorageLayout &layout,
    std::shared_ptr<PhysicalMemoryAuthority> authority)
    : layout_(layout)
{
    layout.validate();
    if (!authority)
        throw std::invalid_argument("CPU grouped-MoE backing requires physical-memory admission");
    const auto device = layout.batch.allocator();
    batch = std::make_shared<FP32Tensor>(
        std::vector<size_t>{layout.row_capacity, size_t(layout.d_model)},
        layout.batch.allocate(authority), device);
    gate = std::make_shared<FP32Tensor>(
        std::vector<size_t>{layout.route_capacity, size_t(layout.expert_intermediate)},
        layout.gate.allocate(authority), device);
    up = std::make_shared<FP32Tensor>(
        std::vector<size_t>{layout.route_capacity, size_t(layout.expert_intermediate)},
        layout.up.allocate(authority), device);
    output = std::make_shared<FP32Tensor>(
        std::vector<size_t>{layout.route_capacity, size_t(layout.d_model)},
        layout.output.allocate(authority), device);
    route_outputs = layout.route_outputs.allocate(authority);
    router_q8 = layout.router_q8.allocate(authority);
    swiglu_q8 = layout.swiglu_q8.allocate(authority);
    kernel_router_q8 = std::make_shared<AlignedVector<Q8_1Block>>(
        layout.kernel_router_q8.allocate(authority));
}

size_t CPUGroupedMoEStorageBacking::physicalBytes() const
{
    // Tensor capacity is fixed by the canonical declaration and is immutable
    // while owned here; independent vector roots expose actual allocator bytes.
    return (batch && !batch->is_raw_data_released() ? layout_.batch.physicalBytes() : 0)
        + (gate && !gate->is_raw_data_released() ? layout_.gate.physicalBytes() : 0)
        + (up && !up->is_raw_data_released() ? layout_.up.physicalBytes() : 0)
        + (output && !output->is_raw_data_released() ? layout_.output.physicalBytes() : 0)
        + route_outputs.allocationBytes() + router_q8.allocationBytes() + swiglu_q8.allocationBytes()
        + (kernel_router_q8 ? kernel_router_q8->allocationBytes() : 0);
}

bool CPUGroupedMoEStorageBacking::hasExternalOwners() const noexcept
{
    return batch.use_count() != 1 || gate.use_count() != 1 || up.use_count() != 1 ||
        output.use_count() != 1 || kernel_router_q8.use_count() != 1;
}

void CPUGroupedMoEStorageBacking::zero()
{
    if (physicalBytes() != layout_.physicalBytes())
        throw std::logic_error("CPU grouped-MoE backing was retired before zeroing");
    for (const auto &tensor : {batch, gate, up, output})
        std::memset(tensor->raw_mutable_data(), 0, tensor->size_bytes());
    const auto clear = [](auto &storage)
    {
        if (!storage.empty())
            std::memset(static_cast<void *>(storage.data()), 0,
                storage.size() * sizeof(typename std::remove_reference_t<decltype(storage)>::value_type));
    };
    clear(route_outputs);
    clear(router_q8);
    clear(swiglu_q8);
    clear(*kernel_router_q8);
}
}
