/**
 * @file PlanningForwardStateWork.cpp
 * @brief Context-aware state work with canonical native payload and TP/GDN geometry.
 *
 * Algebra is intentionally bounded: causal attention and equivalent recurrent
 * updates provide compute demand, and a logical traversal provides a streaming
 * lower bound. It is not a reconstruction of every tile, cache miss or launch.
 * Costs retain this limitation instead of confusing resident bytes with traffic
 * or a cache-resident projection rate with independent memory bandwidth.
 */
#include "PlanningForwardStateWork.h"
#include "MemoryPlanner.h"
#include "PlanningModelMetadata.h"
#include "PersistentStateMemoryEstimator.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace llaminar2
{
    PlanningForwardInvocation::PlanningForwardInvocation(int rows, double preceding_context)
        : rows_(rows), preceding_context_(preceding_context)
    {
        if (rows <= 0 || !std::isfinite(preceding_context) || preceding_context < 0 ||
            !std::isfinite(causalPairs()))
            throw std::invalid_argument("Forward work needs positive rows and finite nonnegative preceding context");
    }

    double PlanningForwardInvocation::causalPairs() const noexcept
    {
        // Convert before arithmetic: a large, valid metadata horizon must not
        // overflow an integer intermediate or iterate once per decode token.
        const double rows = rows_;
        return rows * preceding_context_ + rows * (rows + 1.0) / 2.0;
    }

    std::vector<PlanningLayerStateWork> compilePlanningForwardStateWork(
        const PlanningModelMetadata &model, const DevicePlanConfig &device,
        PlanningMainForwardPhase phase, const PlanningForwardInvocation &invocation)
    {
        if (phase != PlanningMainForwardPhase::Prefill && phase != PlanningMainForwardPhase::Decode)
            throw std::invalid_argument("Unknown main-forward phase");
        if (device.execution_role == DeviceExecutionMemoryRole::RoutedExpertParticipant) return {};
        if (device.execution_role != DeviceExecutionMemoryRole::ContinuationGraph)
            throw std::invalid_argument("Unknown participant state-work role");
        const auto &profile = model.memoryProfile();
        const int capacity = device.max_seq_len > 0 ? device.max_seq_len : profile.max_seq_len;
        if (device.activation_seq_len < invocation.rows() || capacity <= 0 ||
            invocation.precedingContext() + invocation.rows() > capacity ||
            (phase == PlanningMainForwardPhase::Decode && invocation.rows() != 1))
            throw std::invalid_argument("Forward state work exceeds admitted rows/context or is not serial decode");
        if (device.total_shards <= 0 || device.shard_index < 0 || device.shard_index >= device.total_shards)
            throw std::invalid_argument("Forward state work has invalid tensor-parallel coordinates");

        const bool replicated = phase == PlanningMainForwardPhase::Decode &&
            std::find(device.additional_weight_sets.begin(), device.additional_weight_sets.end(),
                AdditionalPersistentWeightSet::ReplicatedDenseDecode) != device.additional_weight_sets.end();
        int head_start = 0, query_heads = profile.n_heads, kv_heads = profile.n_kv_heads;
        if (!replicated && device.tensor_parallel_assignment)
        {
            const auto &assignment = *device.tensor_parallel_assignment;
            if (device.total_shards <= 1 || assignment.device != device.device ||
                assignment.local_rank != device.shard_index || !assignment.isValid() ||
                assignment.kv_head_start < 0 || assignment.kv_head_count <= 0 ||
                assignment.kv_head_start > profile.n_kv_heads - assignment.kv_head_count)
                throw std::invalid_argument("Forward state work received a foreign TP assignment");
            head_start = assignment.head_start;
            query_heads = assignment.head_count;
            kv_heads = assignment.kv_head_count;
        }
        else if (!replicated && device.total_shards > 1)
        {
            // Cross-rank uniform TP is admitted only with integral ownership.
            // Rank-local uneven/GQA replication comes from the explicit slice,
            // never from dividing every dimension by the number of devices.
            if (profile.n_heads % device.total_shards || device.local_kv_heads <= 0)
                throw std::invalid_argument("Uniform state work requires exact compiled query/KV head ownership");
            query_heads = profile.n_heads / device.total_shards;
            head_start = device.shard_index * query_heads;
            kv_heads = device.local_kv_heads;
        }
        if (query_heads <= 0 || kv_heads <= 0 || profile.head_dim <= 0 || head_start < 0 ||
            head_start > profile.n_heads - query_heads || kv_heads > profile.n_kv_heads)
            throw std::invalid_argument("Forward state work has invalid local attention geometry");
        const int first = device.first_layer;
        const int last = device.last_layer < 0 ? model.mainLayerCount() - 1 :
            std::min(device.last_layer, model.mainLayerCount() - 1);
        if (first < 0 || first > last)
            throw std::invalid_argument("Forward state work has no owned main-model layers");
        const auto family = profile.gdn_conv_kernel_size > 0 && profile.gdn_state_size > 0 &&
            profile.full_attention_interval > 0 ? KVCacheFamily::Hybrid : KVCacheFamily::AttentionOnly;
        std::vector<PlanningLayerStateWork> result;
        const double rows = invocation.rows();
        for (int layer = first; layer <= last; ++layer)
        {
            if (PersistentStateMemoryEstimator::isFullAttentionLayer(profile, layer))
            {
                // The codec owner supplies position and fixed-anchor extents.
                // No CUDA linearization replica, CPU metadata or unused ring
                // horizon is charged as an attention read/write here.
                const auto one = KVCacheMemoryEstimator::logicalPayload(family, 1, kv_heads,
                    profile.head_dim, device.kv_precision, device.device).totalBytes();
                const auto two = KVCacheMemoryEstimator::logicalPayload(family, 2, kv_heads,
                    profile.head_dim, device.kv_precision, device.device).totalBytes();
                if (two <= one || two - one > one)
                    throw std::logic_error("KV payload does not have a positive position plus nonnegative anchor extent");
                const double position = two - one;
                const double anchor = one - (two - one);
                const double pairs = invocation.causalPairs();
                // Two dot products cost 4*D per pair. Softmax's subtract,
                // exponent, sum, reciprocal and scale form a scalar-op proxy,
                // not an assertion that transcendental instructions cost one FMA.
                const double operations = query_heads * pairs * (4.0 * profile.head_dim + 5.0);
                result.push_back({layer, invocation.rows(), PlanningAttentionStateWork{
                    query_heads, kv_heads, profile.head_dim, pairs, operations,
                    (invocation.precedingContext() + rows) * position + anchor,
                    rows * position,
                    sizeof(float) * rows * (2.0 * query_heads + 2.0 * kv_heads) * profile.head_dim}});
            }
            else
            {
                const auto geometry = HybridGDNStateGeometry::resolve(profile.n_heads, head_start, query_heads,
                    profile.gdn_group_count, profile.gdn_time_step_rank, profile.gdn_state_size,
                    profile.gdn_inner_size, profile.gdn_conv_kernel_size);
                const double matrix = geometry.local_recurrence_state_floats;
                const double values = static_cast<double>(geometry.local_value_heads) * geometry.d_v;
                // Equivalent recurrence: decay S (1), predict v (2), outer
                // update S (2), and output q*S (2) per matrix element. The
                // installed chunked prefill kernel may use a different DAG;
                // these are algebraic work, not serial-row production calls.
                const double recurrence = 7.0 * matrix + 3.0 * values;
                const double convolution = 2.0 * profile.gdn_conv_kernel_size * geometry.local_qkv_dim;
                const double normalize = 10.0 * geometry.local_key_heads * geometry.d_k;
                result.push_back({layer, invocation.rows(), PlanningGDNStateWork{geometry,
                    rows * (recurrence + convolution + normalize),
                    2.0 * geometry.localPayloadBytes(1),
                    sizeof(float) * rows * (geometry.local_qkv_dim + values)}});
            }
        }
        return result;
    }

    double planningStateServiceSeconds(const PlanningLayerStateWork &work,
        const PlanningFP32ArithmeticObservations &arithmetic, const PlanningMemoryBandwidthObservation &memory)
    {
        if (work.rows <= 0 || work.layer < 0 || arithmetic.device != memory.request.device() ||
            arithmetic.phases.size() != 2 || arithmetic.phases[0].rows != 1 ||
            arithmetic.phases[1].rows != PlanningFP32ArithmeticPlan::kPrefillRows)
            throw std::invalid_argument("State service requires matching FP32/streaming device and complete row evidence");
        if (arithmetic.device.is_cpu() && (!arithmetic.cpu ||
                arithmetic.cpu->workers != memory.request.workers() ||
                arithmetic.cpu->execution != memory.request.cpuGeometry()))
            throw std::invalid_argument("State service cannot combine different CPU workers or ISA geometry");
        return std::visit([&](const auto &operation) {
            using Operation = std::decay_t<decltype(operation)>;
            const auto demand = [](double value) {
                if (!std::isfinite(value) || value < 0)
                    throw std::invalid_argument("State work demand must be finite and nonnegative");
                return value;
            };
            const double traffic = [&] {
                if constexpr (std::is_same_v<Operation, PlanningAttentionStateWork>)
                    return demand(operation.kv_read_bytes) + demand(operation.kv_write_bytes) + demand(operation.activation_bytes);
                else return demand(operation.state_bytes) + demand(operation.activation_bytes);
            }();
            if (demand(operation.operations) == 0 || traffic == 0)
                throw std::invalid_argument("An owned state operation cannot have zero arithmetic or traffic demand");
            const double compute = planningArithmeticSeconds(arithmetic.phases[0].service,
                arithmetic.phases[1].rows, arithmetic.phases[1].service, work.rows, operation.operations);
            const double bytes = memory.service.secondsFor(PlanningWorkUnit::Bytes, traffic);
            const double result = std::max(compute, bytes);
            if (!std::isfinite(result) || result <= 0)
                throw std::invalid_argument("State service must have finite positive arithmetic and traffic demand");
            return result;
        }, work.operation);
    }
}
