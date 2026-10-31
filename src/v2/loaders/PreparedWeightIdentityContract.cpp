/**
 * @file PreparedWeightIdentityContract.cpp
 * @brief Preserve actual prepared-store normalization and original-source ownership.
 *
 * Store/model/device/name and registry checks remain at their existing runtime
 * boundaries. This representation predicate is shared with metadata lowering;
 * it neither authorizes a new source nor manufactures a physical alias lease.
 */
#include "loaders/PreparedWeightIdentityContract.h"
#include "tensors/TensorClasses.h"
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** @return Existing explicit GPU GEMM-role declaration, before schema fallback. */
        bool declaredGemmRole(WeightRole role)
        {
            if (isFlashNextProjectionRole(role)) return true;
            switch (role)
            {
            case WeightRole::LMHead: case WeightRole::AttentionQ: case WeightRole::AttentionK:
            case WeightRole::AttentionV: case WeightRole::AttentionWO: case WeightRole::FusedQKV:
            case WeightRole::GDNProjection: case WeightRole::GDNAlphaBetaProjection:
            case WeightRole::FFNGate: case WeightRole::FFNUp: case WeightRole::FFNDown:
            case WeightRole::MoEExpertGate: case WeightRole::MoEExpertUp: case WeightRole::MoEExpertDown:
            case WeightRole::SharedExpertGate: case WeightRole::SharedExpertUp: case WeightRole::SharedExpertDown:
                return true;
            default: return false;
            }
        }
        /** @return Existing hard rejection: a prepared hint cannot reinterpret these operands. */
        bool rejectedRole(WeightRole role)
        {
            return isImmutableFP32WeightRole(role) || role == WeightRole::ImmutableRowTable ||
                role == WeightRole::Embedding || role == WeightRole::OutputNorm ||
                role == WeightRole::GDNSsmParam || role == WeightRole::MoERouter ||
                role == WeightRole::SharedExpertInputGate || role == WeightRole::Norm || role == WeightRole::Bias;
        }
    }
    PreparedWeightIdentityContract::PreparedWeightIdentityContract(std::vector<size_t> shape, TensorType type,
        FloatingActivationOperandPrecision precision, WeightSliceSpec slice,
        std::optional<MoEExpertProjectionOwnership> projection, std::shared_ptr<TensorBase> original_projection)
        : shape_(std::move(shape)), type_(type), precision_(precision), slice_(std::move(slice)),
          projection_(std::move(projection)), original_projection_(std::move(original_projection)) {}

    std::optional<PreparedWeightIdentityContract> PreparedWeightIdentityContract::fromBinding(const WeightBinding &b)
    {
        if (!b.tensor) return std::nullopt;
        return PreparedWeightIdentityContract(b.tensor->shape(), b.tensor->native_type(),
            b.floating_operand_precision, b.slice, b.projection_ownership, b.projection_source_owner);
    }
    PreparedWeightIdentityContract PreparedWeightIdentityContract::completeOriginalMatrix(size_t rows, size_t columns,
        TensorType type, FloatingActivationOperandPrecision precision)
    {
        WeightSliceSpec slice;
        slice.source_rows = slice.row_count = rows;
        slice.source_cols = slice.col_count = columns;
        return PreparedWeightIdentityContract({rows, columns}, type, precision, std::move(slice), {}, {});
    }
    bool PreparedWeightIdentityContract::emptySlice(const WeightSliceSpec &s)
    {
        return s.source_rows == 0 && s.source_cols == 0 && s.row_start == 0 && s.row_count == 0 &&
            s.col_start == 0 && s.col_count == 0 && s.expert_start == 0 && s.expert_count == 0 &&
            s.expert_ids.empty() && !s.inner_is_presliced;
    }
    bool PreparedWeightIdentityContract::fullSlice() const
    {
        return shape_.size() >= 2 && slice_.row_start == 0 && slice_.col_start == 0 &&
            slice_.expert_start == 0 && slice_.expert_count == 0 && slice_.expert_ids.empty() &&
            !slice_.inner_is_presliced && slice_.row_count == shape_[0] && slice_.col_count == shape_[1] &&
            (slice_.source_rows == 0 || slice_.source_rows == shape_[0]) &&
            (slice_.source_cols == 0 || slice_.source_cols == shape_[1]);
    }
    bool PreparedWeightIdentityContract::compatible(const PreparedWeightIdentityContract &r) const
    {
        if (projection_ != r.projection_ || shape_ != r.shape_ || type_ != r.type_ || precision_ != r.precision_)
            return false;
        if (projection_ && (original_projection_ != r.original_projection_ ||
            original_projection_.owner_before(r.original_projection_) ||
            r.original_projection_.owner_before(original_projection_))) return false;
        const bool empty = emptySlice(slice_), other_empty = emptySlice(r.slice_);
        if (empty && other_empty) return true;
        if (empty) return r.fullSlice();
        if (other_empty) return fullSlice();
        return slice_.source_rows == r.slice_.source_rows && slice_.source_cols == r.slice_.source_cols &&
            slice_.row_start == r.slice_.row_start && slice_.row_count == r.slice_.row_count &&
            slice_.col_start == r.slice_.col_start && slice_.col_count == r.slice_.col_count &&
            slice_.expert_start == r.slice_.expert_start && slice_.expert_count == r.slice_.expert_count &&
            slice_.expert_ids == r.slice_.expert_ids && slice_.inner_is_presliced == r.slice_.inner_is_presliced;
    }
    PreparedGemmClassification PreparedWeightIdentityContract::classifyOriginalMatrix(WeightRole role, DeviceId device)
    {
        if (rejectedRole(role)) return PreparedGemmClassification::Rejected;
        if (device.is_gpu() && declaredGemmRole(role)) return PreparedGemmClassification::Gemm;
        return PreparedGemmClassification::RequiresSchema;
    }
    PreparedGemmClassification PreparedWeightIdentityContract::classify(const WeightBinding &b, DeviceId device,
        PreparedExpertJobs experts)
    {
        const bool target = b.residency.home_device == device ||
            (b.residency.resident_device && *b.residency.resident_device == device) ||
            (b.prepared && b.prepared->device == device);
        if (!target || !b.tensor || b.tensor->shape().size() != 2 || rejectedRole(b.identity.role))
            return PreparedGemmClassification::Rejected;
        const auto category = b.identity.residency_category;
        const bool routed = isRoutedExpertRole(b.identity.role) ||
            category == WeightResidencyCategory::AcceleratorRoutedExpert ||
            category == WeightResidencyCategory::CpuFallbackExpert ||
            category == WeightResidencyCategory::WorkerFallbackExpert;
        if (experts == PreparedExpertJobs::Excluded && routed) return PreparedGemmClassification::Rejected;
        if (b.prepared && b.prepared->device == device &&
            ((device.is_cpu() && b.prepared->kind == PreparedWeightKind::CpuPackedGemm) ||
             (device.is_cuda() && b.prepared->kind == PreparedWeightKind::CudaInt8PackedGemm) ||
             (device.is_rocm() && b.prepared->kind == PreparedWeightKind::RocmInt8PackedGemm)))
            return PreparedGemmClassification::Gemm;
        return classifyOriginalMatrix(b.identity.role, device);
    }
}
