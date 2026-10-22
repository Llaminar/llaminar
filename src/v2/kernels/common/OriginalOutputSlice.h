/**
 * @file OriginalOutputSlice.h
 * @brief Store-authenticated arithmetic identity of an original output interval.
 *
 * Only PreparedWeightStore can mint this immutable recording fact from its
 * canonical Factory-to-store publication. It retains the exact engine and source
 * owner snapshots, independently of mutable handle/index wrappers. This proves
 * lifetime and observable source metadata, not independent payload authenticity.
 * It is not device state, capacity, or permission to change K arithmetic.
 */
#pragma once
#include "backends/DeviceId.h"
#include "tensors/NativeVnniFormatInfo.h"
#include <cstdint>
#include <memory>
#include <utility>
#include <stdexcept>

namespace llaminar2
{
    class ITensorGemm;
    class TensorBase;
    class PreparedWeightStore;
    enum class PreparedWeightKind;

    /** @brief Exact source interval carried independently by one fused projection. */
    class OriginalOutputSlice
    {
    public:
        /** @return Original source output width used solely for arithmetic policy. */
        int policyColumns() const noexcept { return original_columns_; }
        /** @return Exact owned engine snapshot; later index replacement cannot rebind it. */
        ITensorGemm *retainedEngine() const noexcept { return engine_owner_.get(); }
        /** @return Exact original prepared representation, part of the frozen reference. */
        PreparedWeightKind preparedKind() const noexcept { return prepared_kind_; }
        /** @return First original output column represented by the prepared source. */
        int firstColumn() const noexcept { return first_column_; }
        /** @return Actual output/storage width of this prepared projection. */
        int physicalColumns() const noexcept { return physical_columns_; }
        /** @return Complete unsliced original reduction width. */
        int reductionColumns() const noexcept { return reduction_columns_; }
        /** @return Source binding identity retained in graph-cache identity. */
        std::uint64_t bindingId() const noexcept { return binding_id_; }
        /** @return Original model identity, not a local source-name alias. */
        std::uint64_t modelId() const noexcept { return model_id_; }
        /** @return Exact prepared endpoint authenticated by the store. */
        DeviceId device() const noexcept { return device_; }
        /**
         * @brief Reject a foreign engine or changed physical launch geometry.
         * @param engine Actual prepared engine, never another same-shape matrix.
         * @param columns Actual engine/launch output width.
         * @param reduction Actual engine/launch reduction width.
         * @throws std::invalid_argument Before any enqueue on identity mismatch.
         */
        void require(const ITensorGemm *engine, int columns, int reduction) const
        {
            if (!engine_owner_ || !source_owner_ ||
                engine != engine_owner_.get() || columns != physical_columns_ ||
                reduction != reduction_columns_)
                throw std::invalid_argument("Original output slice disagrees with prepared engine/N/K");
        }
        /** @return Original source format, distinct from normalized execution bytes. */
        NativeVnniSourceIdentity sourceIdentity() const noexcept { return source_identity_; }
        /** @return All frozen facts, including retained object addresses, for capture identity. */
        std::uint64_t recordingIdentity() const noexcept
        {
            std::uint64_t identity = 1469598103934665603ULL;
            const auto mix = [&](std::uint64_t word) { identity = (identity ^ word) * 1099511628211ULL; };
            mix(reinterpret_cast<std::uintptr_t>(engine_owner_.get())); mix(reinterpret_cast<std::uintptr_t>(source_owner_.get()));
            mix(binding_id_); mix(model_id_); mix(static_cast<std::uint64_t>(prepared_kind_)); mix(static_cast<std::uint64_t>(device_.type)); mix(device_.ordinal);
            mix(original_columns_); mix(first_column_); mix(physical_columns_); mix(reduction_columns_);
            mix(source_identity_.codebook_id); mix(source_identity_.is_superblock); mix(source_identity_.present);
            return identity;
        }
        /** @brief Compare all frozen source, endpoint and arithmetic facts. */
        bool operator==(const OriginalOutputSlice &) const = default;

    private:
        friend class PreparedWeightStore;
        /** @brief Mint only after the store authenticates its real frozen binding. */
        OriginalOutputSlice(std::shared_ptr<ITensorGemm> engine, std::shared_ptr<TensorBase> source,
            std::uint64_t binding_id, std::uint64_t model_id, PreparedWeightKind prepared_kind, DeviceId device,
            int original_columns, int first_column, int physical_columns, int reduction_columns,
            NativeVnniSourceIdentity source_identity)
            : source_owner_(std::move(source)), engine_owner_(std::move(engine)),
              binding_id_(binding_id), model_id_(model_id), prepared_kind_(prepared_kind),
              device_(device), original_columns_(original_columns), first_column_(first_column),
              physical_columns_(physical_columns), reduction_columns_(reduction_columns),
              source_identity_(source_identity) {}
        // Destruction releases the engine before its source; neither owner is a
        // mutable PreparedGemmHandle/PreparedGemmWeights publication wrapper.
        std::shared_ptr<TensorBase> source_owner_; ///< Exact binding.tensor_owner snapshot.
        std::shared_ptr<ITensorGemm> engine_owner_; ///< Exact owned_kernel snapshot, not an alias to a raw pointer.
        std::uint64_t binding_id_; ///< Store entry, not a duplicated weight ledger.
        std::uint64_t model_id_; ///< Model-context lifetime identity.
        PreparedWeightKind prepared_kind_; ///< Exact prepared-reference representation.
        DeviceId device_; ///< Physical prepared endpoint.
        int original_columns_; ///< Original mathematical output width.
        int first_column_; ///< Original source output origin.
        int physical_columns_; ///< Exact local output width.
        int reduction_columns_; ///< Exact original K, never a K slice.
        NativeVnniSourceIdentity source_identity_; ///< Frozen source family, never physical decoder inference.
    };
}
