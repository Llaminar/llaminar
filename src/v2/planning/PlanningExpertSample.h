/**
 * @file PlanningExpertSample.h
 * @brief File-independent, admitted source triplets for distributed planning.
 *
 * A plan is immutable metadata resolved by the GGUF owner. A loaded sample is
 * the same plan plus three destruction-ordered native payload owners. Remote
 * ranks receive those bytes without opening the initiating host's model path;
 * CPU/GPU preparation consumes this identical interface in either case.
 */
#pragma once
#include "PlanningModelMetadata.h"
#include "backends/DeviceId.h"
#include "tensors/TensorType.h"
#include <array>
#include <functional>

namespace llaminar2
{
    class PlanningLoadedExpertSample;
    /** @brief Source materialization contract: only a GGUF reader owns transient read storage. */
    enum class PlanningSampleOrigin { LocalGGUF, PublishedPayload };

    /** @brief Sealed source identity and geometry; no payload or physical admission. */
    class PlanningExpertSamplePlan final
    {
    public:
        /** @return Validated complete triplet from the retained GGUF directory, without payload I/O. */
        static PlanningExpertSamplePlan resolve(const PlanningModelSource &source,
            const PlanningExpertSampleRequest &request);
        /** @return Source path as provenance only; followers must not open it. */
        const std::string &modelPath() const noexcept { return model_path_; }
        /** @return Exact source architecture used by workspace geometry. */
        const std::string &architecture() const noexcept { return architecture_; }
        /** @return One complete same-layer expert selection. */
        const PlanningExpertSampleRequest &request() const noexcept { return request_; }
        /** @return Source extents, formats and layer validated before allocation. */
        const PlanningExpertSampleDescription &description() const noexcept { return description_; }
        /** @return Canonical native tensor type for a projection; invalid indices throw. */
        TensorType tensorType(size_t projection) const;
        /** @return Versioned control envelope only, never model payload. */
        std::vector<uint8_t> serialize() const;
        /** @return Validated root envelope; malformed formats, extents or identities throw. */
        static PlanningExpertSamplePlan deserialize(std::span<const uint8_t> bytes);
        /**
         * @brief Load this exact plan through its original GGUF owner.
         * @param source Must still describe exactly the admitted plan.
         * @param memory Source and reader allocation authority.
         * @param host_device Exact CPU allocator selected for preparation.
         * @return Native triplet retaining all source claims, including after the reader retires.
         */
        PlanningLoadedExpertSample load(const PlanningModelSource &source,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory, DeviceId host_device) const;
    private:
        /** @brief Construct only within validated directory/decoder transactions. */
        PlanningExpertSamplePlan() = default;
        std::string model_path_, architecture_;
        PlanningExpertSampleRequest request_;
        PlanningExpertSampleDescription description_{};
        std::array<uint32_t, 3> gguf_types_{};
    };

    /** @brief Immutable borrowed tensors cannot outlive their coupled physical claims. */
    class PlanningLoadedExpertSample final
    {
    public:
        /** @return Exact source identity of these bytes, not caller-supplied timing labels. */
        const PlanningExpertSamplePlan &plan() const noexcept { return plan_; }
        /** @return Const source matrix valid for this owner's lifetime; invalid indices throw. */
        const TensorBase &tensor(size_t projection) const { return matrices_.at(projection).tensor(); }
    private:
        friend class PlanningExpertSamplePlan;
        friend class PlanningExpertSamplePublication;
        /** @brief Seal three fully populated native matrices after local I/O or MPI completion. */
        PlanningLoadedExpertSample(PlanningExpertSamplePlan plan,
            std::array<PlanningLoadedModelSample, 3> matrices)
            : plan_(std::move(plan)), matrices_(std::move(matrices)) {}
        PlanningExpertSamplePlan plan_;
        std::array<PlanningLoadedModelSample, 3> matrices_;
    };

    /** @brief Collective setup transport, separate from root-only selection callbacks. */
    class PlanningExpertSamplePublication final
    {
    public:
        /** @return One identical validated source plan; the producer runs only on rank zero. */
        static PlanningExpertSamplePlan describe(const std::shared_ptr<IMPIContext> &mpi,
            const std::function<PlanningExpertSamplePlan()> &root_describe);
        /**
         * @brief Publish one admitted triplet into every rank's final source storage.
         * @param mpi Exact discovery membership; null means process-local work.
         * @param plan Previously published plan, authenticated again before payload transfer.
         * @param memory This rank's source-payload authority; reader staging is needed only at root.
         * @param host_device Exact CPU allocator, never a remote device ordinal.
         * @param root_load Root-only loader; must return the exact published plan.
         * @return Sealed triplet only after all payloads and all ranks complete.
         * @throws std::runtime_error Collectively on pre-traffic failures.
         *
         * No extra payload copy/staging is allocated at followers. Native MPI
         * requests retain all owners until completion; a transport failure or
         * 30-second stall aborts rather than unwinding live receive addresses.
         */
        static PlanningLoadedExpertSample publish(const std::shared_ptr<IMPIContext> &mpi,
            const PlanningExpertSamplePlan &plan, const std::shared_ptr<PhysicalMemoryAuthority> &memory,
            DeviceId host_device, const std::function<PlanningLoadedExpertSample()> &root_load);
    };
}
