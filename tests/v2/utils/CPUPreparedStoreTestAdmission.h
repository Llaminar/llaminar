/**
 * @file CPUPreparedStoreTestAdmission.h
 * @brief Exact device-free CPU prepared-store admission for bounded fixtures.
 *
 * Fixtures contribute only their final packed matrices through the production
 * layout and allocator geometry. Borrowed native floating tensors allocate no
 * prepared copies. The resulting immutable authority is installed before any
 * production packing; tests never relax the production admission requirement.
 */
#pragma once
#include "loaders/PreparedWeightStore.h"
#include "kernels/cpu/gemm/CPUNativeVNNIWeightPacker.h"
#include "tensors/TensorSlice.h"
#include <initializer_list>
#include <limits>
#include <span>
#include <optional>
namespace llaminar2::test
{
/**
 * @brief Resolve one fixture's actual final packed allocation without packing.
 * @param tensor Source CPU matrix; native floating storage remains borrowed.
 * @param output_rows Exact output extent of a retained physical slice, absent for the full matrix.
 * @return Exact heap allocator extent of the retained NativeVNNI representation.
 */
inline size_t cpuPreparedStoreTestBytes(const TensorBase &tensor, std::optional<size_t> output_rows = std::nullopt)
{
    // The ordinary CPU factory unwraps a physically presliced tensor before
    // constructing the packer. Use that same source for shape and rotation;
    // the wrapper's own rotation state need not equal its inner tensor's state.
    const TensorBase *packed_source = &tensor;
    if (const auto *slice = dynamic_cast<const TensorSlice *>(&tensor);
        slice && slice->metadata().inner_is_presliced)
        packed_source = slice->inner();
    const auto *source = IINT8Unpackable::fromTensor(packed_source);
    if (!source) return 0;
    const auto *format = source->vnniFormatInfo();
    if (!format) return 0;
    if (packed_source->shape().size() != 2)
        throw std::invalid_argument("CPU prepared fixture requires a matrix with native packing metadata");
    const auto footprint = cpu::native_vnni::preparedFootprintForFormat(*format, packed_source->activationRotation() != nullptr);
    const size_t rows = output_rows.value_or(packed_source->shape()[0]);
    if (rows == 0 || rows > packed_source->shape()[0])
        throw std::invalid_argument("CPU prepared fixture slice exceeds its actual source rows");
    const size_t chunks = (rows + 63) / 64;
    const size_t blocks = (packed_source->shape()[1] + 31) / 32;
    const auto checked_product = [](size_t a, size_t b) {
        if (a && b > std::numeric_limits<size_t>::max() / a)
            throw std::overflow_error("CPU prepared fixture byte count overflows");
        return a * b;
    };
    return CPUWeightStoragePlacement::local().requiredAllocationBytes<uint8_t>(
        checked_product(checked_product(chunks, blocks), footprint.weight_bytes_per_n_chunk_k_block));
}
/** @brief One unique physical slice prepared from an original fixture matrix. */
struct CPUPreparedStoreTestSlice
{
    const TensorBase *tensor; ///< Exact immutable source.
    size_t first_row, end_row; ///< Half-open output interval, matching the CPU factory.
};
/**
 * @brief Admit all concurrently retained fixture matrices before preparation.
 * @param store Empty prepared store that receives the sole immutable authority.
 * @param tensors Complete CPU source set prepared during this fixture lifetime.
 * @param device Exact CPU allocator identity selected by the fixture.
 * @param owner Exact persistent-weight owner used by production preparation.
 * @param slices Complete distinct retained slice inventory, in addition to full matrices.
 */
inline void admitCPUPreparedStoreForTest(PreparedWeightStore &store,
    std::span<const TensorBase *const> tensors, DeviceId device = DeviceId::cpu(),
    PhysicalMemoryOwner owner = PhysicalMemoryOwner::PrimaryModelWeights,
    std::span<const CPUPreparedStoreTestSlice> slices = {})
{
    if (!device.is_cpu()) throw std::invalid_argument("CPU fixture admission requires a CPU allocator");
    if (owner != PhysicalMemoryOwner::PrimaryModelWeights &&
        owner != PhysicalMemoryOwner::AdditionalModelWeights &&
        owner != PhysicalMemoryOwner::RoutedExpertWeights)
        throw std::invalid_argument("CPU prepared fixture requires its persistent-weight owner");
    size_t bytes = 0;
    for (const auto *tensor : tensors)
    {
        if (!tensor) throw std::invalid_argument("CPU fixture admission received a null tensor");
        const auto add = cpuPreparedStoreTestBytes(*tensor);
        if (add > std::numeric_limits<size_t>::max() - bytes)
            throw std::overflow_error("CPU prepared fixture owner overflows");
        bytes += add;
    }
    for (const auto &slice : slices)
    {
        if (!slice.tensor || slice.end_row <= slice.first_row ||
            slice.end_row > slice.tensor->rows())
            throw std::invalid_argument("CPU prepared fixture requires valid physical slice intervals");
        const auto add = cpuPreparedStoreTestBytes(*slice.tensor, slice.end_row - slice.first_row);
        if (add > std::numeric_limits<size_t>::max() - bytes)
            throw std::overflow_error("CPU prepared fixture slice owner overflows");
        bytes += add;
    }
    // A borrowed-floating fixture has a real resource identity and zero new
    // prepared bytes. The resource observation must still be nonzero.
    const size_t capacity = std::max(size_t{1}, bytes);
    PhysicalMemoryPlanBuilder builder;
    builder.add(PhysicalMemoryResource{.world_rank = 0, .device = device,
        .total_bytes = capacity, .admission_available_bytes = capacity},
        owner, bytes);
    store.installPhysicalMemoryAuthority(std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0));
}
/** @brief Convenience overload for fixed sets of fixture-owned matrices. */
inline void admitCPUPreparedStoreForTest(PreparedWeightStore &store,
    std::initializer_list<const TensorBase *> tensors, DeviceId device = DeviceId::cpu(),
    PhysicalMemoryOwner owner = PhysicalMemoryOwner::PrimaryModelWeights)
{
    admitCPUPreparedStoreForTest(store, std::span<const TensorBase *const>(tensors.begin(), tensors.size()), device, owner);
}
}
