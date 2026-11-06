/**
 * @file W4A4WeightStorage.cpp
 * @brief Native W4A4/H allocation adoption, mapping admission and retained row leases.
 *
 * Admission precedes publication. A failed source check retires any moved
 * allocation through its original PMA lease; no partial matrix escapes.
 * Slice construction changes only bounded view metadata and retains the
 * original physical owner, including mmap provenance and allocation identity.
 */
#include "W4A4WeightStorage.h"
#include "loaders/MmapRegion.h"

#include <sstream>
#include <variant>

namespace llaminar2::w4a4
{
    namespace
    {
        /**
         * @brief Translate source admission into a diagnostic including the exact malformed block.
         * @param transform Source-declared arithmetic.
         * @param rows Output rows.
         * @param columns Complete input width.
         * @param bytes Exact immutable native source, not an allocation capacity.
         * @return Admitted view bound to these bytes.
         * @throws std::invalid_argument On byte alignment, geometry or source metadata failure.
         */
        SourceView admit(Transform transform, std::size_t rows, std::size_t columns,
                         std::span<const std::uint8_t> bytes)
        {
            const auto extent = MatrixContract::sourceExtent(rows, columns);
            if (!extent) throw std::invalid_argument(extent.error().message());
            if (bytes.size() != *extent || !bytes.data() ||
                reinterpret_cast<std::uintptr_t>(bytes.data()) % alignof(SourceBlock))
                throw std::invalid_argument("W4A4 storage requires exact aligned native source bytes");
            const auto view = SourceView::admit(transform, rows, columns,
                {reinterpret_cast<const SourceBlock *>(bytes.data()), bytes.size() / sizeof(SourceBlock)});
            if (!view)
            {
                std::ostringstream diagnostic;
                diagnostic << view.error().message();
                if (view.error().location)
                    diagnostic << " at row " << view.error().location->row
                               << ", block " << view.error().location->block;
                throw std::invalid_argument(diagnostic.str());
            }
            return *view;
        }
    }

    /** @brief One physical owner; alternatives cannot simultaneously retain duplicate payloads. */
    struct WeightStorage::Backing
    {
        using Owner = std::variant<AlignedVector<std::uint8_t>, std::shared_ptr<const MmapRegion>>;
        Owner owner; ///< Existing physical allocation or exact file mapping.
        DeviceId allocator; ///< Actual allocation endpoint, not first-touch placement.

        /** @brief Retain already admitted CPU bytes without another allocation or lease. */
        Backing(AlignedVector<std::uint8_t> bytes, DeviceId device)
            : owner(std::in_place_index<0>, std::move(bytes)), allocator(device) {}
        /** @brief Retain the original mapping without a second tensor payload. */
        explicit Backing(std::shared_ptr<const MmapRegion> region)
            : owner(std::in_place_index<1>, std::move(region)), allocator(DeviceId::cpu()) {}
    };

    WeightStorage WeightStorage::adopt(Transform transform, std::size_t rows, std::size_t columns,
                                       AlignedVector<std::uint8_t> bytes, PhysicalMemoryOwner owner)
    {
        if (owner != PhysicalMemoryOwner::PrimaryModelWeights &&
            owner != PhysicalMemoryOwner::AdditionalModelWeights &&
            owner != PhysicalMemoryOwner::RoutedExpertWeights &&
            owner != PhysicalMemoryOwner::ModelSourcePayload)
            throw std::invalid_argument("W4A4 final storage requires an admitted physical weight or source owner");
        const auto allocator = bytes.requireAdmittedAllocator(owner);
        // Admission happens while the incoming vector still owns every byte.
        // Its move into the backing preserves the allocation and source address.
        const auto view = admit(transform, rows, columns, {bytes.data(), bytes.size()});
        return WeightStorage(std::make_shared<const Backing>(std::move(bytes), allocator), view);
    }

    WeightStorage WeightStorage::map(Transform transform, std::size_t rows, std::size_t columns,
                                     std::shared_ptr<const MmapRegion> region, std::size_t byte_offset)
    {
        const auto extent = MatrixContract::sourceExtent(rows, columns);
        if (!extent) throw std::invalid_argument(extent.error().message());
        if (!region || !region->data() || byte_offset > region->size() ||
            *extent > region->size() - byte_offset)
            throw std::invalid_argument("W4A4 matrix lies outside its retained source mapping");
        const auto view = admit(transform, rows, columns, {region->data() + byte_offset, *extent});
        return WeightStorage(std::make_shared<const Backing>(std::move(region)), view);
    }

    WeightStorage WeightStorage::rowSlice(std::size_t first, std::size_t count) const
    {
        const auto slice = view_.rowSlice(first, count);
        if (!slice) throw std::out_of_range(slice.error().message());
        return WeightStorage(backing_, *slice);
    }

    DeviceId WeightStorage::allocator() const noexcept { return backing_->allocator; }

    bool WeightStorage::isMapped() const noexcept
    {
        return std::holds_alternative<std::shared_ptr<const MmapRegion>>(backing_->owner);
    }
} // namespace llaminar2::w4a4
