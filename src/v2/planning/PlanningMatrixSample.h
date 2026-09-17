/**
 * @file PlanningMatrixSample.h
 * @brief Sealed ordinary/source-sharded matrices for bounded production GEMM observations.
 *
 * Root validates source selection through the retained GGUF owner. Followers
 * consume identical native bytes without a local model file. Prepared engines
 * borrow the loaded matrix only while its PMA-backed owner is retained; source
 * geometry is never reconstructed from a timing or a prepared byte count.
 */
#pragma once
#include "PlanningModelMetadata.h"
#include "backends/DeviceId.h"
#include <functional>

namespace llaminar2
{
    class PlanningLoadedMatrixSample;

    /** @brief Immutable single-matrix source contract, independent of execution backend. */
    class PlanningMatrixSamplePlan final
    {
    public:
        /** @return Source-validated whole matrix, source-axis shard or expert matrix without I/O. */
        static PlanningMatrixSamplePlan resolve(const PlanningModelSource &source,
            const PlanningModelSampleRequest &request);
        /** @return Model path as provenance only; receivers must not open it. */
        const std::string &modelPath() const noexcept { return model_path_; }
        /** @return Original model architecture, not inferred from a tensor name. */
        const std::string &architecture() const noexcept { return architecture_; }
        /** @return Exact immutable source interval or expert index. */
        const PlanningModelSampleRequest &request() const noexcept { return request_; }
        /** @return Native source dimensions/extent, not packed-weight allocation demand. */
        const PlanningModelSampleGeometry &geometry() const noexcept { return geometry_; }
        /** @return Original source format; no dequantization or format substitution. */
        const std::string &format() const noexcept { return format_; }
        /** @return Validated native tensor type for source allocation/preparation. */
        TensorType tensorType() const;
        /** @return Runtime type from the canonical model preparation policy, not a sampling override. */
        TensorType executionType() const;
        /** @return Runtime format whose production projection implementation is measured. */
        std::string executionFormat() const;
        /** @return Checked logical runtime extent; source publication retains its native extent. */
        size_t executionPayloadBytes() const;
        /** @return Versioned immutable control record, excluding all tensor payload. */
        std::vector<uint8_t> serialize() const;
        /** @return Native/source-selection validated publication; malformed records throw. */
        static PlanningMatrixSamplePlan deserialize(std::span<const uint8_t> bytes);
        /**
         * @brief Load this exact source through its original metadata owner.
         * @param memory Sole authority admitting source payload and transient reader staging.
         * @param host_device Exact CPU allocator selected by the rank's setup policy.
         * @return Plan and loaded source ownership retained together through preparation.
         */
        PlanningLoadedMatrixSample load(const PlanningModelSource &source,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory, DeviceId host_device) const;
    private:
        /** @brief Only source resolution and the strict decoder construct a plan. */
        PlanningMatrixSamplePlan() = default;
        std::string model_path_, architecture_, format_;
        PlanningModelSampleRequest request_;
        PlanningModelSampleGeometry geometry_{};
        uint32_t gguf_type_ = 0;
    };

    /** @brief A complete native payload and its sealed selection cannot separate in transit. */
    class PlanningLoadedMatrixSample final
    {
    public:
        /** @return Immutable source identity used for this exact payload. */
        const PlanningMatrixSamplePlan &plan() const noexcept { return plan_; }
        /** @return Tensor borrow valid while this loaded owner (or a copy) remains alive. */
        const TensorBase &tensor() const { return sample_.tensor(); }
    private:
        friend class PlanningMatrixSamplePlan;
        friend class PlanningMatrixSamplePublication;
        /** @brief Seal only a source-loaded or completely received native owner. */
        PlanningLoadedMatrixSample(PlanningMatrixSamplePlan plan, PlanningLoadedModelSample sample)
            : plan_(std::move(plan)), sample_(std::move(sample)) {}
        PlanningMatrixSamplePlan plan_;
        PlanningLoadedModelSample sample_;
    };

    /** @brief Single-matrix adapter to the same failure-atomic native publication as expert samples. */
    class PlanningMatrixSamplePublication final
    {
    public:
        /** @return The identical root-owned source plan on every discovery rank. */
        static PlanningMatrixSamplePlan describe(const std::shared_ptr<IMPIContext> &mpi,
            const std::function<PlanningMatrixSamplePlan()> &root_describe);
        /**
         * @brief Publish into final admitted storage, without receive-side payload copies or file access.
         * @param mpi Discovery communicator; null explicitly denotes a process-local transaction.
         * @param plan Same published source selection on every participating rank.
         * @param memory Rank-bound admission prepared inside the enclosing collective consensus.
         * @param host_device CPU allocator retaining source ownership through native completion.
         * @param root_load Invoked only at root after all ranks authenticate the source plan.
         * @return Sealed payload only after native completion and all-rank acceptance.
         */
        static PlanningLoadedMatrixSample publish(const std::shared_ptr<IMPIContext> &mpi,
            const PlanningMatrixSamplePlan &plan, const std::shared_ptr<PhysicalMemoryAuthority> &memory,
            DeviceId host_device, const std::function<PlanningLoadedMatrixSample()> &root_load);
    };
}
