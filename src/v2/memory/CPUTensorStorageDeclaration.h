/**
 * @file CPUTensorStorageDeclaration.h
 * @brief Immutable native CPU tensor roots shared by schema planning and arenas.
 *
 * Logical matrix shape, supported native dtype and exact owner/device select
 * one typed CPUStorageRoot. Both BOM projection and materialization visit that
 * same immutable root. Tensor adoption transfers the backing-owned allocation
 * lease; aliases and native views therefore retain physical storage naturally.
 */
#pragma once
#include "CPUStorageRoot.h"
#include "execution/debug/BufferRole.h"
#include <array>
#include <memory>
#include <string_view>
#include <variant>
namespace llaminar2
{
class TensorBase;
class TensorFactory;
/** @brief Complete native CPU tensor declaration, with no allocation or live ledger. */
class CPUTensorStorageDeclaration final
{
public:
    /**
     * @brief Bind logical shape, native dtype and physical ownership before admission.
     * @param rows Flattened matrix rows selected by the declaring schema.
     * @param columns Flattened native row width.
     * @param dtype One of FP32, FP16, BF16, INT32, Q8_1 or Q16_1.
     * @param owner Explicit PMA owner of the actual backing.
     * @param allocator Exact admitted CPU DeviceId, independent of NUMA node.
     * @param placement Explicit allocation-kind/NUMA policy; heap remains the default.
     */
    CPUTensorStorageDeclaration(size_t rows, size_t columns, BufferTensorType dtype,
        PhysicalMemoryOwner owner, DeviceId allocator,
        CPUStoragePlacement placement = CPUStoragePlacement::local());
    /**
     * @brief Flatten one resolved graph descriptor with the arena's exact shape contract.
     * @param descriptor Nonempty resolved CPU shape/dtype/device declaration.
     * @param owner Explicit physical owner selected by the graph allocation inventory.
     * @return Immutable native root used identically by planning and materialization.
     * @throws std::invalid_argument For empty/zero shapes or unsupported dtype/device.
     * @throws std::overflow_error For overflowing trailing dimensions or total elements.
     */
    [[nodiscard]] static CPUTensorStorageDeclaration fromDescriptor(
        const BufferDescriptor &descriptor, PhysicalMemoryOwner owner);
    /** @return Structural identity of every shape, native dtype and physical-root policy. */
    bool operator==(const CPUTensorStorageDeclaration &) const = default;
    /** @return Declared flattened row count. */
    [[nodiscard]] size_t rows() const noexcept { return shape_[0]; }
    /** @return Declared native row width. */
    [[nodiscard]] size_t columns() const noexcept { return shape_[1]; }
    /** @return Exact native tensor dtype; no inference or fallback is applied. */
    [[nodiscard]] BufferTensorType dtype() const noexcept { return dtype_; }
    /** @return Explicit physical owner shared by declaration and actual backing. */
    [[nodiscard]] PhysicalMemoryOwner owner() const
    { return std::visit([](const auto &root){return root.owner();},root_); }
    /** @return Exact admitted CPU allocator, independent of NUMA placement. */
    [[nodiscard]] DeviceId allocator() const
    { return std::visit([](const auto &root){return root.allocator();},root_); }
    /** @return Explicit allocation-kind and NUMA policy. */
    [[nodiscard]] CPUStoragePlacement placement() const
    { return std::visit([](const auto &root){return root.placement();},root_); }
    /** @return Exact allocator-rounded physical extent of the retained native backing. */
    [[nodiscard]] size_t physicalBytes() const;
    /**
     * @brief Construct native storage from the same typed root projected by physicalBytes.
     * @param authority Mandatory admitted physical-memory authority.
     * @param factory Optional existing NUMA binding authority, applied before first touch.
     * @return Unpublished tensor adopting the actual zero-initialized aligned backing.
     *
     * Factory binding does not change allocator kind: a heap declaration stays
     * heap-backed. Dedicated mappings require explicit placement in the declaration.
     */
    [[nodiscard]] std::unique_ptr<TensorBase> materialize(
        std::shared_ptr<PhysicalMemoryAuthority> authority, TensorFactory *factory = nullptr) const;
    /**
     * @brief Parse the legacy arena registration boundary into a supported typed dtype.
     * @param dtype Explicit dtype spelling supplied by registration.
     * @return Supported dtype value, or a precise exception for unknown input.
     */
    [[nodiscard]] static BufferTensorType requireDtype(std::string_view dtype);
private:
    using Root = std::variant<CPUStorageRoot<float>, CPUStorageRoot<uint16_t>,
        CPUStorageRoot<int32_t>, CPUStorageRoot<uint8_t>>;
    /**
     * @brief Select the one native backing geometry from the complete declaration.
     * @param rows Flattened native matrix rows.
     * @param columns Native row width before block padding.
     * @param dtype Explicit supported native dtype.
     * @param owner Typed physical owner of the root.
     * @param allocator Exact admitted CPU allocator identity.
     * @param placement Explicit allocation kind and first-touch policy.
     * @return Single typed root used for both projection and materialization.
     */
    static Root declareRoot(size_t rows, size_t columns, BufferTensorType dtype,
        PhysicalMemoryOwner owner, DeviceId allocator, CPUStoragePlacement placement);
    const std::array<size_t,2> shape_; ///< Native logical matrix geometry.
    const BufferTensorType dtype_; ///< Exact dtype distinguishing shared native element types.
    const Root root_; ///< Sole immutable physical allocation declaration.
};
}
