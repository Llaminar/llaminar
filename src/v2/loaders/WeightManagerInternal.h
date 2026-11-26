/**
 * @file WeightManagerInternal.h
 * @brief Shared internal weight classification and preparation helpers for WeightManager shards.
 */
#pragma once
/**
 * @file WeightManager.cpp
 * @brief Model-weight loading, typed sharding, preparation, and residency.
 *
 * WeightManager turns declarative schema and tensor-parallel assignments into
 * native-format source slices, then publishes stable prepared-weight handles
 * for graph construction.  It is an I/O/materialization authority, not a
 * second sharding planner: semantic slice arithmetic belongs to WeightSlicer
 * and typed model-specific ownership values such as GDNHeadAssignment.
 *
 * Loading and packing occur before graph capture.  Production inference must
 * never call these paths, allocate replacement payloads, or repair residency
 * by synchronizing an execution stream.
 * A participant's completed upload does not end another participant's source
 * lifetime. Static expert source requests materialize exact original N/K
 * rectangles from their frozen ownership; they cannot fall back to a full parent.
 * Frozen host policies survive preparation and the complete graph
 * lifecycle owns reclamation of shared model bytes. Static original F shards
 * enter this same pool/staging pipeline only through complete declared frozen
 * participant sources; they never publish whole-expert aliases. Overlay
 * preparation authenticates each frozen fragment against its original exact
 * domain/rank/member/role request before publishing a prepared registry entry.
 * Each sharding operation retains an immutable TP assignment while parallel
 * participant construction may atomically republish the shared manager owner.
 * Preparation role classification shares the exact identity contract consumed
 * by physical creator/alias lowering; schema fallback retains its existing role.
 * CPU participant rank qualifies each cached shard independently of its shared
 * host storage DeviceId; ordinary/GPU cache access retains device identity.
 * GPU pipeline futures independently inherit the admitted OpenMP partition;
 * their creators and every CPU packing path preserve original placement.
 *
 * @author David Sanftenberg
 */

#include "WeightManager.h"
#include "loaders/BoundedGPUExpertSource.h"
#include <map>
#include <tuple>
#include "utils/ScopedOpenMPPartitionAffinity.h"
#include "CPUWeightRequirementOwner.h"
#include "tensors/NativeTensorExtent.h"
#include "loaders/NativeWeightSourceTransaction.h"
#include "kernels/cpu/gemm/CPUPreparedWeightWriter.h"
#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include "execution/moe/ExpertShardSourceSelection.h"
#include "loaders/PreparedWeightIdentityContract.h"
#include "WeightSlicer.h"
#include "MoEExpertSourceView.h"
#include "MmapRegion.h"
#include "PreparedWeightRepresentationContract.h"
#include "PreparedHalfWeights.h"
#include "PreparedWeightStore.h"
#include "PreparedOverlaySourceRetirement.h"
#include "GPUHostLoadPreflight.h"
#include "GPUVramPreflight.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "utils/VramBillOfMaterials.h"
#include "../execution/moe/ExpertWeightPayloadProvider.h"
#include "../execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "../execution/moe/MoEExpertOverlayRuntimePlan.h"
#include "../execution/moe/MoEExpertWeightService.h"
#include "../execution/moe/RoutedExpertOwnerAssignment.h"
#include "../utils/Logger.h"
#include "../utils/PerfStatsCollector.h"
#include "../utils/WeightLoadingProfiler.h"
#include "../utils/DebugEnv.h"
#include "../tensors/TensorFactory.h"
#include "../tensors/TensorSlice.h"
#include "../kernels/KernelFactory.h"
#include "../backends/BackendManager.h"
#include "../transfer/TransferEngine.h"

// GPU weight loading pipeline
#include "gpu_pipeline/LoadOrchestrator.h"
#include "gpu_pipeline/WeightVRAMPool.h"
#include "WeightLoadProgress.h"
#include "gpu_pipeline/RepackFormat.h"

#ifdef HAVE_ROCM
#include "../kernels/rocm/gemm/ROCmWeightPacker.h"
#endif
#include "../tensors/BlockStructures.h"
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <regex>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <iomanip>
#include <thread>
#include <future>
#include <limits>
#include <bit>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <vector>
#ifdef __linux__
#include <malloc.h> // malloc_trim
#endif


namespace llaminar2
{
    namespace
    {
        /**
         * @brief Publish the physical routed-expert selection used by a loader.
         *
         * This is setup-only evidence: it performs no inference-path transfer
         * and is dormant unless the `moe_placement` PerfStats domain is
         * requested.  Numerical parity then proves that the independently
         * configured graph owner map consumes this exact packed selection.
         */
        void recordRoutedExpertWeightSelection(
            const std::vector<int> &expert_ids,
            RoutedExpertOwnerOrder owner_order,
            int participant_index,
            int layer_idx,
            const DeviceId &device)
        {
            if (!PerfStatsCollector::isDomainEnabled("moe_placement"))
                return;

            PerfStatsCollector::addCounter(
                "moe_placement",
                "routed_expert_weight_selection",
                static_cast<double>(expert_ids.size()),
                "load",
                device.to_string(),
                {
                    {"layer", std::to_string(layer_idx)},
                    {"owner_order", routedExpertOwnerOrderToString(owner_order)},
                    {"participant", std::to_string(participant_index)},
                    {"selection_layout",
                     routed_expert_ownership::expertIdsFormContiguousSpan(
                         expert_ids)
                         ? "contiguous"
                         : "noncontiguous"},
                });
        }

        /**
         * @brief Assemble original native stripes in the existing linked head order.
         * @param source Original directory geometry in GGUF K/N order.
         * @param axis Concatenation axis in logical matrix order.
         * @param spans Exact ordered intervals along that axis.
         * @return Final source assembly without allocating or reading a parent tensor.
         */
        NativeWeightSourceAssembly orderedCPUNativeAssembly(
            const NativeWeightSourceGeometry &source, NativeWeightSourceAssembly::Axis axis,
            const std::vector<GDNHeadSpan> &spans)
        {
            using Axis = NativeWeightSourceAssembly::Axis;
            const auto &shape = source.original_shape;
            if (shape.empty() || shape.size() > 2 || spans.empty() ||
                !source.native_group_elements || !source.native_group_bytes)
                throw std::invalid_argument("CPU ordered assembly needs original native matrix or vector geometry");
            const size_t rows = shape.size() == 2 ? shape[1] : 1;
            const size_t columns = shape[0];
            const bool by_row = axis == Axis::Rows;
            NativeWeightSourceAssembly assembly{.axis = axis,
                .final_rows = by_row ? 0 : rows, .final_columns = by_row ? columns : 0};
            auto &assembled_extent = by_row ? assembly.final_rows : assembly.final_columns;
            const size_t original_extent = by_row ? rows : columns;
            assembly.segments.reserve(spans.size());
            for (const auto span : spans) {
                if (span.start < 0 || span.count <= 0)
                    throw std::invalid_argument("CPU ordered assembly has an invalid original interval");
                const auto first = static_cast<size_t>(span.start);
                const auto count = static_cast<size_t>(span.count);
                const auto first_column = by_row ? 0 : first;
                const auto stripe_columns = by_row ? columns : count;
                if (first > original_extent || count > original_extent - first ||
                    count > std::numeric_limits<size_t>::max() - assembled_extent ||
                    first_column % source.native_group_elements ||
                    stripe_columns % source.native_group_elements ||
                    stripe_columns / source.native_group_elements >
                        std::numeric_limits<size_t>::max() / source.native_group_bytes)
                    throw std::invalid_argument("CPU ordered assembly exceeds its source or splits native groups");
                assembly.segments.push_back({.original_expert = 0, .selected_expert_ordinal = 0,
                    .first_row = by_row ? first : 0, .rows = by_row ? count : rows,
                    .first_column = first_column, .columns = stripe_columns,
                    .native_row_bytes = stripe_columns / source.native_group_elements * source.native_group_bytes});
                assembled_extent += count;
            }
            return assembly;
        }

        uint64_t stableMaterializedBindingId(
            const WeightRequirement &requirement,
            const InferenceStrategy &strategy)
        {
            uint64_t hash = 1469598103934665603ull;
            auto mixByte = [&](uint8_t byte)
            {
                hash ^= static_cast<uint64_t>(byte);
                hash *= 1099511628211ull;
            };
            auto mixString = [&](const std::string &value)
            {
                for (unsigned char ch : value)
                    mixByte(ch);
                mixByte(0xffu);
            };
            auto mixInt = [&](int64_t value)
            {
                uint64_t bits = static_cast<uint64_t>(value);
                for (int i = 0; i < 8; ++i)
                    mixByte(static_cast<uint8_t>((bits >> (i * 8)) & 0xffu));
            };
            auto mixSize = [&](size_t value)
            {
                mixInt(static_cast<int64_t>(value));
            };

            mixInt(static_cast<int64_t>(strategy.model_id.value));
            mixString(requirement.canonical_name);
            mixString(requirement.source_name);
            mixInt(static_cast<int>(requirement.role));
            mixInt(static_cast<int>(requirement.derivation));
            mixInt(requirement.layer);
            mixInt(requirement.expert);
            mixInt(requirement.pp_stage);
            mixInt(requirement.tp_domain);
            mixInt(requirement.tp_rank_or_device_index);
            mixInt(requirement.bypass_tensor_parallel ? 1 : 0);
            mixInt(static_cast<int>(requirement.residency_category));
            mixString(requirement.overlay_domain);
            mixInt(requirement.overlay_participant_index);
            mixInt(requirement.overlay_participant_world_rank);
            mixInt(static_cast<int>(requirement.target_device.type));
            mixInt(requirement.target_device.ordinal);
            mixSize(requirement.slice.row_start);
            mixSize(requirement.slice.row_count);
            mixSize(requirement.slice.col_start);
            mixSize(requirement.slice.col_count);
            mixSize(requirement.slice.expert_start);
            mixSize(requirement.slice.expert_count);
            mixSize(requirement.slice.expert_ids.size());
            for (const int expert_id : requirement.slice.expert_ids)
                mixInt(expert_id);

            if (requirement.native_source_assembly) {
                mixString("original_native_source_assembly_v1");
                const auto &assembly=*requirement.native_source_assembly;
                mixInt(static_cast<int>(assembly.axis));mixSize(assembly.final_rows);mixSize(assembly.final_columns);
                mixSize(assembly.segments.size());
                for (const auto &segment:assembly.segments) {
                    mixSize(segment.original_expert);mixSize(segment.selected_expert_ordinal);
                    mixSize(segment.first_row);mixSize(segment.rows);mixSize(segment.first_column);mixSize(segment.columns);
                    mixSize(segment.native_row_bytes);
                }
            }

            if (requirement.projection_ownership)
            {
                // Extend only explicit new requests: old binding hashes stay
                // byte-for-byte unchanged. Shape equality omits peer intervals.
                mixString("original_native_expert_projection_v1");
                const auto &ownership = *requirement.projection_ownership;
                const auto geometry = ownership.geometry();
                mixInt(geometry.experts);
                mixInt(geometry.model_columns);
                mixInt(geometry.intermediate_columns);
                mixInt(ownership.participant());
                mixInt(ownership.participants());
                for (const auto interval : ownership.intermediateIntervals())
                {
                    mixInt(interval.first_column);
                    mixInt(interval.columns);
                }
            }
            if(requirement.joint_source) {
                const auto& member=requirement.joint_source->participant();
                mixString(member.device.is_cpu() ? "original_cpu_joint_fragment_v1" : "original_gpu_joint_fragment_v1");
                mixInt(member.participant_id);mixInt(member.world_rank);
                mixString(member.address.toString());
            }
            return hash == 0 ? 1 : hash;
        }

        bool hasVnniPackedQuantizedPayload(const TensorBase *tensor)
        {
            const auto *unpackable = IINT8Unpackable::fromTensor(tensor);
            return unpackable && unpackable->vnniFormatInfo() != nullptr;
        }

        bool hasExplicitSlice(const WeightSliceSpec &slice)
        {
            return slice.source_rows != 0 ||
                   slice.source_cols != 0 ||
                   slice.row_start != 0 ||
                   slice.row_count != 0 ||
                   slice.col_start != 0 ||
                   slice.col_count != 0 ||
                   slice.expert_start != 0 ||
                   slice.expert_count != 0 ||
                   !slice.expert_ids.empty() ||
                   slice.inner_is_presliced;
        }

        bool isEmptySliceSpec(const WeightSliceSpec &slice)
        {
            return !hasExplicitSlice(slice);
        }

        WeightSliceSpec mergeRequirementSlice(
            const WeightSliceSpec &materialized,
            const WeightSliceSpec &requirement)
        {
            if (!hasExplicitSlice(requirement))
                return materialized;
            if (isEmptySliceSpec(materialized))
                return requirement;

            WeightSliceSpec merged = materialized;
            if (requirement.source_rows != 0)
                merged.source_rows = requirement.source_rows;
            if (requirement.source_cols != 0)
                merged.source_cols = requirement.source_cols;

            if (requirement.row_start != 0 || requirement.row_count != 0)
            {
                merged.row_start = requirement.row_start;
                if (requirement.row_count != 0)
                    merged.row_count = requirement.row_count;
            }
            if (requirement.col_start != 0 || requirement.col_count != 0)
            {
                merged.col_start = requirement.col_start;
                if (requirement.col_count != 0)
                    merged.col_count = requirement.col_count;
            }
            if (requirement.expert_start != 0 || requirement.expert_count != 0)
            {
                merged.expert_start = requirement.expert_start;
                if (requirement.expert_count != 0)
                    merged.expert_count = requirement.expert_count;
            }
            if (!requirement.expert_ids.empty())
            {
                merged.expert_ids = requirement.expert_ids;
                merged.expert_start = static_cast<size_t>(requirement.expert_ids.front());
                merged.expert_count = requirement.expert_ids.size();
            }
            if (requirement.inner_is_presliced)
                merged.inner_is_presliced = true;
            return merged;
        }

        bool isRoutedExpertResidencyCategory(WeightResidencyCategory category)
        {
            return category == WeightResidencyCategory::AcceleratorRoutedExpert ||
                   category == WeightResidencyCategory::CpuFallbackExpert ||
                   category == WeightResidencyCategory::WorkerFallbackExpert;
        }

        bool isRoutedExpertBinding(const WeightBinding &binding)
        {
            return isRoutedExpertRole(binding.identity.role) ||
                   isRoutedExpertResidencyCategory(binding.identity.residency_category);
        }

        std::string formatExpertIds(const std::vector<int> &expert_ids)
        {
            std::ostringstream out;
            out << '[';
            for (size_t index = 0; index < expert_ids.size(); ++index)
            {
                if (index != 0u)
                    out << ',';
                out << expert_ids[index];
            }
            out << ']';
            return out.str();
        }

        bool bindingTargetsDevice(const WeightBinding &binding, DeviceId device)
        {
            return binding.residency.home_device == device ||
                   (binding.residency.resident_device.has_value() &&
                    *binding.residency.resident_device == device) ||
                   (binding.prepared.has_value() && binding.prepared->device == device);
        }

        /**
         * @brief Authenticate explicit static preparation before any backend or allocation.
         * @param frozen Exact owned source or projected GEMM bindings.
         * @param device Preparation participant being entered.
         * @param include_expert_jobs Whether this invocation actually prepares expert handles.
         *
         * Static GPU matrices retain their original source through both passes;
         * the dense pass defers their handles to explicit expert preparation.
         * CPU joint fragments can enter the dense pass with expert work excluded
         * when their native source transaction already committed them.
         */
        void validateStaticProjectionPreparation(const FrozenModelWeightSet &frozen, DeviceId device,
                                                   bool include_expert_jobs = true)
        {
            for (const auto &binding : frozen.bindings())
            {
                if (!binding.projection_ownership || !bindingTargetsDevice(binding, device))
                    continue;
                if (device.is_gpu())
                {
                    (void)ExpertGemmRegistry::staticProjectionScope(binding);
                    if (!binding.projection_source_owner && !binding.native_source)
                        throw std::invalid_argument("Static GPU preparation requires its original source owner");
                }
                else if (binding.joint_source)
                {
                    (void)ExpertGemmRegistry::staticProjectionScope(binding);
                    if (!binding.native_source && (!include_expert_jobs || !binding.projection_source_owner))
                        throw std::invalid_argument("CPU joint preparation requires its original source transaction or explicit expert work");
                }
                if (!binding.tensor || binding.tensor->shape().size() != 2)
                    throw std::invalid_argument("Static expert GEMM preparation requires projectedExpertMatrices, not a 3D source parent");
            }
        }

        bool shouldPrepareFrozenGemmBinding(
            const WeightManager &manager,
            const WeightBinding &binding,
            DeviceId device,
            bool include_expert_jobs)
        {
            const auto classification = PreparedWeightIdentityContract::classify(binding, device,
                include_expert_jobs ? PreparedExpertJobs::Included : PreparedExpertJobs::Excluded);
            if (classification == PreparedGemmClassification::Rejected) return false;
            if (classification == PreparedGemmClassification::Gemm) return true;

            try
            {
                /*
                 * WeightRole::Other bindings from older graph builders still
                 * require schema classification. Production model contexts
                 * always install that schema before materialization.
                 */
                return manager.isGemmWeight(binding.identity.canonical_name);
            }
            catch (const std::exception &)
            {
                return false;
            }
        }

        /** @return Whether @p value has the exact trailing substring @p suffix. */
        bool endsWith(const std::string &value, const std::string &suffix)
        {
            return value.size() >= suffix.size() &&
                   value.compare(
                       value.size() - suffix.size(),
                       suffix.size(),
                       suffix) == 0;
        }

    }

    namespace
    {

        /**
         * @brief Create a typed tensor from raw bytes (generic for all tensor types)
         *
         * Handles FP32, FP16, BF16, and all quantized tensor types.
         * Raw bytes are interpreted according to the tensor type's native storage format.
         */
        std::unique_ptr<TensorBase> createTensorFromRawData(
            TensorType type,
            const std::vector<size_t> &shape,
            std::vector<uint8_t> raw_data)
        {
            switch (type)
            {
            case TensorType::FP32:
            {
                auto tensor = std::make_unique<FP32Tensor>(shape);
                std::memcpy(tensor->mutable_data(), raw_data.data(), raw_data.size());
                return tensor;
            }
            case TensorType::Q4_0:
                return std::make_unique<Q4_0Tensor>(shape, raw_data);
            case TensorType::Q8_0:
                return std::make_unique<Q8_0Tensor>(shape, raw_data);
            case TensorType::Q8_1:
                return std::make_unique<Q8_1Tensor>(shape, raw_data);
            case TensorType::Q4_1:
                return std::make_unique<Q4_1Tensor>(shape, raw_data);
            case TensorType::Q5_0:
                return std::make_unique<Q5_0Tensor>(shape, raw_data);
            case TensorType::Q5_1:
                return std::make_unique<Q5_1Tensor>(shape, raw_data);
            case TensorType::Q6_K:
                return std::make_unique<Q6_KTensor>(shape, raw_data);
            case TensorType::Q2_K:
                return std::make_unique<Q2_KTensor>(shape, raw_data);
            case TensorType::Q5_K:
                return std::make_unique<Q5_KTensor>(shape, raw_data);
            case TensorType::Q3_K:
                return std::make_unique<Q3_KTensor>(shape, raw_data);
            case TensorType::Q4_K:
                return std::make_unique<Q4_KTensor>(shape, raw_data);
            case TensorType::Q8_K:
                return std::make_unique<Q8_KTensor>(shape, raw_data);
            case TensorType::IQ4_NL:
                return std::make_unique<IQ4_NLTensor>(shape, raw_data);
            case TensorType::IQ4_XS:
                return std::make_unique<IQ4_XSTensor>(shape, raw_data);
            case TensorType::IQ2_XXS:
                return std::make_unique<IQ2_XXSTensor>(shape, raw_data);
            case TensorType::IQ2_XS:
                return std::make_unique<IQ2_XSTensor>(shape, raw_data);
            case TensorType::IQ3_XXS:
                return std::make_unique<IQ3_XXSTensor>(shape, raw_data);
            case TensorType::IQ2_S:
                return std::make_unique<IQ2_STensor>(shape, raw_data);
            case TensorType::IQ3_S:
                return std::make_unique<IQ3_STensor>(shape, raw_data);
            case TensorType::IQ1_S:
                return std::make_unique<IQ1_STensor>(shape, raw_data);
            case TensorType::IQ1_M:
                return std::make_unique<IQ1_MTensor>(shape, raw_data);
            case TensorType::BF16:
            {
                std::vector<uint16_t> bf16_data(raw_data.size() / 2);
                std::memcpy(bf16_data.data(), raw_data.data(), raw_data.size());
                return std::make_unique<BF16Tensor>(shape, bf16_data);
            }
            case TensorType::FP16:
            {
                std::vector<uint16_t> fp16_data(raw_data.size() / 2);
                std::memcpy(fp16_data.data(), raw_data.data(), raw_data.size());
                return std::make_unique<FP16Tensor>(shape, fp16_data);
            }
            default:
                throw std::runtime_error("[WeightManager] Unsupported tensor type for createTensorFromRawData: " +
                                         std::to_string(static_cast<int>(type)));
            }
        }
    } // anonymous namespace
} // namespace llaminar2
