/**
 * @file WeightShardGeometry.cpp
 * @brief Preserve schema-directed matrix axes instead of dividing flat weight bytes.
 *
 * GQA replication and modulo-linked GDN ownership use the same model dimensions
 * and DeviceShardingAssignment as production loading. An input shard keeps N
 * intact and reduces K; an output shard does the converse. Expert apportionment
 * changes only the number of complete matrices. Physical packing remains owned
 * by its existing format contracts and PhysicalMemoryAuthority admission.
 */
#include "planning/WeightShardGeometry.h"
#include "planning/ModelMemoryProfile.h"
#include "config/GDNHeadAssignment.h"
#include "execution/local_execution/graph/SchemaFactoryRegistry.h"
#include "loaders/WeightIdentity.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    namespace
    {
        /** @brief One typed interval in a model-owned logical dimension. */
        struct LogicalShardInterval
        {
            size_t start = 0u;
            size_t count = 0u;
            size_t total = 0u;
        };

        /** @brief Divide one semantic dimension while keeping remainders whole. */
        LogicalShardInterval equalLogicalShardInterval(
            size_t total,
            int shard_index,
            int total_shards,
            const std::string &tensor_name)
        {
            if (total == 0u || total_shards <= 0 || shard_index < 0 ||
                shard_index >= total_shards)
            {
                throw std::invalid_argument(
                    "Weight shard geometry cannot divide the semantic dimension for " +
                    tensor_name);
            }
            const size_t degree = static_cast<size_t>(total_shards);
            const size_t index = static_cast<size_t>(shard_index);
            const size_t quotient = total / degree;
            const size_t remainder = total % degree;
            const size_t count = quotient + (index < remainder ? 1u : 0u);
            if (count == 0u)
            {
                throw std::invalid_argument(
                    "Tensor-parallel degree exceeds the semantic dimension for " +
                    tensor_name);
            }
            return {
                .start = index * quotient + std::min(index, remainder),
                .count = count,
                .total = total,
            };
        }

        /**
         * @brief Resolve the assignment interval named by the schema.
         *
         * This function translates only typed dimension coordinates. Tensor
         * names and sharding policy remain owned by WeightShardingConfig.
         */
        LogicalShardInterval logicalShardInterval(
            WeightDimensionType dimension,
            const ModelMemoryProfile &profile,
            const DeviceShardingAssignment &assignment,
            const std::string &tensor_name)
        {
            const auto checked = [&tensor_name](
                                     int start,
                                     int count,
                                     int total,
                                     const char *dimension_name)
            {
                if (total <= 0 || start < 0 || count <= 0 ||
                    start > total - count)
                {
                    throw std::invalid_argument(
                        "Weight shard geometry received an invalid " +
                        std::string(dimension_name) + " assignment for " +
                        tensor_name);
                }
                return LogicalShardInterval{
                    .start = static_cast<size_t>(start),
                    .count = static_cast<size_t>(count),
                    .total = static_cast<size_t>(total),
                };
            };

            switch (dimension)
            {
            case WeightDimensionType::Heads:
            case WeightDimensionType::ProportionalHeads:
            case WeightDimensionType::FusedQKVHeads:
                return checked(
                    assignment.head_start,
                    assignment.head_count,
                    profile.n_heads,
                    "query-head");
            case WeightDimensionType::KVHeads:
                return checked(
                    assignment.kv_head_start,
                    assignment.kv_head_count,
                    profile.n_kv_heads,
                    "KV-head");
            case WeightDimensionType::FFNHidden:
                return checked(
                    assignment.d_ff_start,
                    assignment.d_ff_count,
                    profile.d_ff,
                    "FFN");
            case WeightDimensionType::Vocab:
                return checked(
                    assignment.vocab_start,
                    assignment.vocab_count,
                    profile.vocab_size,
                    "vocabulary");
            case WeightDimensionType::Bias1D:
            case WeightDimensionType::None:
                break;
            }
            throw std::invalid_argument(
                "Sharded weight has no typed slice dimension in the model schema: " +
                tensor_name);
        }

        /** @brief Resolve an equal cross-rank interval in a schema-owned dimension. */
        LogicalShardInterval equalSchemaInterval(
            WeightDimensionType dimension,
            const ModelMemoryProfile &profile,
            int shard_index,
            int total_shards,
            const std::string &tensor_name)
        {
            int total = 0;
            switch (dimension)
            {
            case WeightDimensionType::Heads:
            case WeightDimensionType::ProportionalHeads:
            case WeightDimensionType::FusedQKVHeads:
                total = profile.n_heads;
                break;
            case WeightDimensionType::KVHeads:
                total = profile.n_kv_heads;
                break;
            case WeightDimensionType::FFNHidden:
                total = profile.d_ff;
                break;
            case WeightDimensionType::Vocab:
                total = profile.vocab_size;
                break;
            case WeightDimensionType::Bias1D:
            case WeightDimensionType::None:
                throw std::invalid_argument(
                    "Sharded weight has no typed slice dimension in the model schema: " +
                    tensor_name);
            }
            if (total <= 0)
                throw std::invalid_argument("Weight shard semantic dimension is not positive: " + tensor_name);
            return equalLogicalShardInterval(
                static_cast<size_t>(total), shard_index, total_shards, tensor_name);
        }

        /** @brief Apply a logical interval to a flat element count exactly as a slice boundary. */
        size_t intervalElementCount(
            size_t complete_elements,
            LogicalShardInterval interval)
        {
            if (interval.total == 0u || interval.count > interval.total ||
                interval.start > interval.total - interval.count)
            {
                throw std::invalid_argument(
                    "Weight shard interval is invalid");
            }
            // Multiply in a wide intermediate: a valid shard of a representable
            // tensor must not overflow simply because its global coordinate is large.
            const size_t begin = static_cast<size_t>(
                static_cast<unsigned __int128>(complete_elements) * interval.start / interval.total);
            const size_t end = static_cast<size_t>(
                static_cast<unsigned __int128>(complete_elements) *
                (interval.start + interval.count) / interval.total);
            return end - begin;
        }

        /**
         * @brief Resolve a fused-QKV participant's exact element count.
         *
         * GDN uses modulo-linked key/value ownership while full attention uses
         * independent query and KV ranges. Both are already encoded in the
         * same typed model geometry and assignment consumed by weight loading.
         */
        size_t fusedQKVElementCount(
            const TensorSizeInfo &tensor,
            const ModelMemoryProfile &profile,
            const DeviceShardingAssignment &assignment)
        {
            if (tensor.K == 0u || tensor.elements % tensor.K != 0u)
            {
                throw std::invalid_argument(
                    "Fused-QKV tensor lacks an integral matrix shape: " +
                    tensor.name);
            }
            const size_t rows = tensor.elements / tensor.K;
            if (profile.gdn_group_count > 0 &&
                profile.gdn_time_step_rank > 0 &&
                profile.gdn_state_size > 0)
            {
                const size_t expected_gdn_rows =
                    (2u * static_cast<size_t>(profile.gdn_group_count) +
                     static_cast<size_t>(profile.gdn_time_step_rank)) *
                    static_cast<size_t>(profile.gdn_state_size);
                if (rows == expected_gdn_rows)
                {
                    const auto gdn = GDNHeadAssignment::fromPartition(
                        profile.gdn_group_count,
                        profile.gdn_time_step_rank,
                        assignment.head_start,
                        assignment.head_count,
                        profile.n_heads);
                    return gdn.localFusedRows(profile.gdn_state_size) *
                           tensor.K;
                }
            }

            if (profile.n_heads > 0 && profile.n_kv_heads > 0 &&
                profile.head_dim > 0)
            {
                const size_t expected_attention_rows =
                    static_cast<size_t>(profile.n_heads) *
                        static_cast<size_t>(profile.head_dim) +
                    2u * static_cast<size_t>(profile.n_kv_heads) *
                        static_cast<size_t>(profile.head_dim);
                if (rows == expected_attention_rows)
                {
                    const size_t local_rows =
                        static_cast<size_t>(assignment.head_count) *
                            static_cast<size_t>(profile.head_dim) +
                        2u * static_cast<size_t>(assignment.kv_head_count) *
                            static_cast<size_t>(profile.head_dim);
                    return local_rows * tensor.K;
                }
            }

            return intervalElementCount(
                tensor.elements,
                logicalShardInterval(
                    WeightDimensionType::FusedQKVHeads,
                    profile,
                    assignment,
                    tensor.name));
        }


        /** @brief Reject inconsistent matrix extents rather than floor-dividing them. */
        size_t checkedProduct(size_t a, size_t b)
        {
            if (b && a > std::numeric_limits<size_t>::max() / b)
                throw std::overflow_error("Weight shard matrix extent overflows size_t");
            return a * b;
        }
    }

    WeightShardGeometryResolver::WeightShardGeometryResolver(
        const ModelMemoryProfile &profile, DeviceId device, int shard_index, int total_shards,
        const std::optional<DeviceShardingAssignment> &assignment)
        : profile_(profile), shard_index_(shard_index), total_shards_(total_shards), assignment_(assignment)
    {
        if (total_shards <= 0 || shard_index < 0 || shard_index >= total_shards ||
            (assignment && (total_shards <= 1 || assignment->local_rank != shard_index ||
                assignment->device != device || !assignment->isValid())))
            throw std::invalid_argument("Weight shard geometry has an invalid TP participant");
        if (total_shards > 1)
            sharding_ = SchemaFactoryRegistry::getWeightShardingConfig(profile.architecture);
    }

    WeightShardGeometry WeightShardGeometryResolver::resolve(
        const TensorSizeInfo &tensor, std::optional<size_t> resident_experts) const
    {
        const auto &profile = profile_;
        const auto &assignment = assignment_;
        const int shard_index = shard_index_, total_shards = total_shards_;
        const WeightRole role = inferWeightRole(tensor.name);
        const bool expert = isRoutedExpertRole(role);
        if (resident_experts && (!expert || profile.expert_count <= 0 ||
                *resident_experts > static_cast<size_t>(profile.expert_count)))
            throw std::invalid_argument("Explicit expert residency does not match its parent tensor");
        if (tensor.elements == 0)
        {
            if (tensor.native_bytes != 0)
                throw std::invalid_argument("Empty weight metadata carries physical source bytes");
            return {0, std::nullopt};
        }

        size_t instances = 1;
        if (expert)
        {
            if (profile.expert_count <= 0 || tensor.elements % static_cast<size_t>(profile.expert_count))
                throw std::invalid_argument("Expert parent is not an integral set of model experts: " + tensor.name);
            instances = static_cast<size_t>(profile.expert_count);
        }
        // Norms and per-head scalar vectors are not one-row GEMM matrices.
        // Legacy metadata-only fixtures may also deliberately omit K entirely.
        const bool vector = role == WeightRole::Norm || role == WeightRole::OutputNorm ||
            role == WeightRole::Bias || (role == WeightRole::GDNSsmParam && tensor.K == tensor.elements);
        std::optional<WeightShardMatrix> matrix;
        if (tensor.K && !vector)
        {
            const size_t per_instance = tensor.elements / instances;
            if (per_instance % tensor.K)
                throw std::invalid_argument("Weight has a nonintegral logical matrix shape: " + tensor.name);
            matrix = WeightShardMatrix{per_instance / tensor.K, tensor.K, instances};
        }

        // An overlay already supplied physical residency. Applying the ordinary
        // expert TP share as well would divide ownership for a second time.
        if (resident_experts)
        {
            if (matrix) matrix->instances = *resident_experts;
            return {checkedProduct(tensor.elements / instances, *resident_experts), matrix};
        }
        if (total_shards == 1)
            return {tensor.elements, matrix};

        const auto [mode, dimension] = sharding_->getModeAndDimension(tensor.name);
        if (mode == WeightShardingMode::Replicate)
            return {tensor.elements, matrix};
        if (mode == WeightShardingMode::ExpertIdApportioned)
        {
            if (!expert)
                throw std::invalid_argument("Expert-axis policy applied to a non-expert tensor");
            const size_t degree = static_cast<size_t>(total_shards);
            const size_t local = instances / degree + (static_cast<size_t>(shard_index) < instances % degree);
            if (matrix) matrix->instances = local;
            return {checkedProduct(tensor.elements / instances, local), matrix};
        }
        if (expert)
            throw std::invalid_argument("Whole expert ownership cannot be replaced by N/K sharding");

        const auto interval = assignment
            ? logicalShardInterval(dimension, profile, *assignment, tensor.name)
            : equalSchemaInterval(dimension, profile, shard_index, total_shards, tensor.name);
        if (dimension == WeightDimensionType::FusedQKVHeads)
        {
            if (!matrix || mode != WeightShardingMode::ColumnParallel)
                throw std::invalid_argument("Fused-QKV sharding requires an output-sharded matrix");
            DeviceShardingAssignment local;
            if (assignment)
                local = *assignment;
            else
            {
                const auto kv = equalSchemaInterval(WeightDimensionType::KVHeads,
                    profile, shard_index, total_shards, tensor.name);
                local.device = DeviceId::cpu(); // Coordinate-only synthetic assignment, never an allocator.
                local.local_rank = shard_index;
                local.head_start = static_cast<int>(interval.start);
                local.head_count = static_cast<int>(interval.count);
                local.kv_head_start = static_cast<int>(kv.start);
                local.kv_head_count = static_cast<int>(kv.count);
            }
            const auto elements = fusedQKVElementCount(tensor, profile, local);
            if (elements % matrix->columns)
                throw std::invalid_argument("Fused-QKV local rows are not integral");
            matrix->rows = elements / matrix->columns;
            return {elements, matrix};
        }

        if (!matrix)
            return {intervalElementCount(tensor.elements, interval), std::nullopt};
        switch (mode)
        {
        case WeightShardingMode::ColumnParallel:
            matrix->rows = intervalElementCount(matrix->rows, interval);
            break;
        case WeightShardingMode::RowParallel:
        case WeightShardingMode::InputParallel:
            matrix->columns = intervalElementCount(matrix->columns, interval);
            break;
        default:
            throw std::invalid_argument("Unknown weight sharding mode");
        }
        if (!matrix->rows || !matrix->columns)
            throw std::invalid_argument("TP geometry assigns an empty matrix: " + tensor.name);
        return {checkedProduct(matrix->rows, matrix->columns), matrix};
    }
}
