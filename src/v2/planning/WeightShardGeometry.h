/**
 * @file WeightShardGeometry.h
 * @brief Schema-owned TP geometry shared by physical sizing and work estimation.
 *
 * This metadata-only contract retains the N/K distinction that an element
 * fraction loses. It owns no capacity, allocation, tensor, or device state.
 * Source offsets are deliberately not implied: fused QKV/GDN shapes can be
 * assembled from disjoint source intervals and must use the production slicer
 * when materialized. Expert residency counts are not routed-token frequencies.
 */
#pragma once
#include "backends/DeviceId.h"
#include "config/TensorParallelConfig.h"
#include "execution/local_execution/graph/GraphSchema.h"
#include <cstddef>
#include <optional>

namespace llaminar2
{
    struct ModelMemoryProfile;
    struct TensorSizeInfo;

    /** @brief Exact logical matrix shape, before native packing or padding. */
    struct WeightShardMatrix
    {
        size_t rows;
        size_t columns;
        size_t instances; ///< One for ordinary weights, resident cardinality for expert parents.
        /** @brief Value comparison includes the independent expert axis. */
        bool operator==(const WeightShardMatrix &) const = default;
    };

    /** @brief Validated participant-local geometry, with no memory-admission authority. */
    class WeightShardGeometry final
    {
    public:
        /** @return Logical elements, not allocated bytes or per-token memory traffic. */
        size_t elements() const noexcept { return elements_; }
        /** @return Exact matrix geometry if the source describes a matrix, otherwise absence. */
        const std::optional<WeightShardMatrix> &matrix() const noexcept { return matrix_; }
    private:
        friend class WeightShardGeometryResolver;
        /** @brief Only the validating resolver may publish complete geometry. */
        WeightShardGeometry(size_t elements, std::optional<WeightShardMatrix> matrix)
            : elements_(elements), matrix_(matrix) {}
        size_t elements_;
        std::optional<WeightShardMatrix> matrix_;
    };

    /**
     * @brief Participant-bound compiler that resolves schema rules once, not per tensor.
     *
     * The borrowed profile must outlive this setup-only compiler. Results retain
     * no profile reference, allocation owner, device handle or admission ledger.
     */
    class WeightShardGeometryResolver final
    {
    public:
        /**
         * @brief Bind exact TP ownership and the canonical model schema.
         * @param profile Immutable model metadata, borrowed for this compiler's lifetime.
         * @param device Exact physical participant.
         * @param shard_index Local TP coordinate, not discovery or execution rank.
         * @param total_shards Positive TP degree.
         * @param assignment Exact local assignment, absent for equal semantic partitions.
         * @throws std::invalid_argument for mismatched or invalid ownership.
         */
        WeightShardGeometryResolver(const ModelMemoryProfile &profile, DeviceId device,
            int shard_index = 0, int total_shards = 1,
            const std::optional<DeviceShardingAssignment> &assignment = {});
        /**
         * @brief Preserve source N/K/expert axes through schema-directed partitioning.
         * @param tensor One original native tensor inventory entry.
         * @param resident_experts Explicit overlay residency; replaces ordinary
         *        expert-axis apportionment without dividing individual experts.
         * @return Validated shape; vectors and element-only metadata have no matrix.
         * @throws std::exception for malformed ownership, geometry or overflow.
         *
         * A GEMM consumer must reject absent matrix geometry. Source intervals
         * are not implied by shape: fused QKV/GDN can use disjoint source spans.
         */
        WeightShardGeometry resolve(const TensorSizeInfo &tensor,
            std::optional<size_t> resident_experts = {}) const;
    private:
        const ModelMemoryProfile &profile_;
        int shard_index_, total_shards_;
        std::optional<DeviceShardingAssignment> assignment_;
        std::optional<WeightShardingConfig> sharding_;
    };
}
