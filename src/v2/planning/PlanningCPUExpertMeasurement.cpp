/**
 * @file PlanningCPUExpertMeasurement.cpp
 * @brief Real source weights, canonical preparation, and one complete CPU FFN.
 *
 * Source reads and packing are bounded by one requested triplet. Every payload
 * is claimed before its allocation, and declaration order retires all engines
 * before their claims. Only immutable completed observations leave this scope.
 * The existing startup service producer owns invocation workspace and timing,
 * so planning does not introduce a parallel expert execution implementation.
 */
#include "PlanningCPUExpertMeasurement.h"
#include "execution/moe/MoEOverlayCPUServiceMeasurement.h"
#include "kernels/KernelFactory.h"
#include "kernels/cpu/gemm/CPUWeightPreparationMemory.h"
#include "kernels/cpu/gemm/CPUProjectionWorkspaceContract.h"
#include "kernels/cpu/CPUInvocationWorkspace.h"
#include "tensors/TensorClasses.h"
#include "utils/CPUFeatures.h"
#include <algorithm>
#include <limits>
#include <omp.h>
#include <optional>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        // Generated CPU dispatch distinguishes an AVX2-only artifact from an
        // AVX512 artifact executing its AVX2 runtime lane. Retain both identities.
#if LLAMINAR_COMPILED_WITH_AVX512
        constexpr ISALevel kCompiledISA = ISALevel::AVX512;
#else
        constexpr ISALevel kCompiledISA = ISALevel::AVX2;
#endif
        /** @brief Validated metadata/BOM terms, not a live allocation ledger. */
        struct ExpertDescription final : PlanningExpertSampleDescription
        {
            std::array<CPUWeightPreparationMemory, 3> preparation;
            size_t prepared_bytes = 0, staging_bytes = 0;
        };

        /** @brief Add independently allocated payloads without wrapping the BOM. */
        void addBytes(size_t &sum, size_t bytes)
        {
            if (bytes > std::numeric_limits<size_t>::max() - sum)
                throw std::overflow_error("CPU expert sample payload overflow");
            sum += bytes;
        }

        /** @return One complete, same-layer/source-expert description before any I/O. */
        ExpertDescription describe(const PlanningExpertSamplePlan &plan, PlanningSampleOrigin origin)
        {
            if (origin != PlanningSampleOrigin::LocalGGUF && origin != PlanningSampleOrigin::PublishedPayload)
                throw std::invalid_argument("Unknown planning sample source materialization contract");
            ExpertDescription result{plan.description(), {}, 0, 0};
            for (size_t i = 0; i < result.matrices.size(); ++i)
            {
                result.preparation[i] = CPUWeightPreparationMemory::sourceNative(
                    result.formats[i], result.matrices[i].n, result.matrices[i].k);
                addBytes(result.prepared_bytes, result.preparation[i].persistent_bytes);
                result.staging_bytes = std::max({result.staging_bytes,
                    origin == PlanningSampleOrigin::LocalGGUF ? result.matrices[i].source_bytes : size_t{0},
                    result.preparation[i].temporary_bytes});
            }
            return result;
        }

        /** @return Existing production workspace geometry, without another estimator. */
        MoEOverlayCPUServiceMeasurement::Geometry geometry(const ExpertDescription &description)
        {
            return {static_cast<int>(description.matrices[0].k), static_cast<int>(description.matrices[0].n)};
        }
    }

    void PlanningCPUExpertMeasurement::contributeMemory(const PlanningModelSource &source,
        const PlanningExpertSampleRequest &request, PhysicalMemoryResource resource, PhysicalMemoryPlanBuilder &builder,
        CPUExecutionGeometry execution, int workers)
    {
        contributeMemory(PlanningExpertSamplePlan::resolve(source, request), PlanningSampleOrigin::LocalGGUF,
            resource, builder, execution, workers);
    }

    void PlanningCPUExpertMeasurement::contributeMemory(const PlanningExpertSamplePlan &plan, PlanningSampleOrigin origin,
        PhysicalMemoryResource resource, PhysicalMemoryPlanBuilder &builder, CPUExecutionGeometry execution, int workers)
    {
        if (!resource.device.is_cpu()) throw std::invalid_argument("CPU expert sample requires a CPU physical resource");
        const auto description = describe(plan, origin);
        constexpr int rows = MoEOverlayCPUServiceMeasurement::kMaximumRows;
        auto workspace = cpuSwiGLUWorkspaceRequirements(rows, geometry(description).intermediate);
        for (size_t projection = 0; projection < description.matrices.size(); ++projection)
            workspace.merge(CPUProjectionWorkspaceContract::sourceNative(description.formats[projection],
                {.rows = rows, .n = static_cast<int>(description.matrices[projection].n),
                 .k = static_cast<int>(description.matrices[projection].k), .workers = workers,
                 .execution = execution, .numerical_policy = CPUProjectionNumericalPolicy::GPUAlignedExpert}));
        builder.add(resource, PhysicalMemoryOwner::ModelSourcePayload, description.source_bytes);
        builder.add(resource, PhysicalMemoryOwner::RoutedExpertWeights, description.prepared_bytes);
        builder.add(resource, PhysicalMemoryOwner::WeightLoadStaging, description.staging_bytes);
        builder.add(resource, PhysicalMemoryOwner::ExecutionWorkspace,
            MoEOverlayCPUServiceMeasurement::allocationBytes(geometry(description), workspace.total_bytes_with_alignment()));
    }

    PlanningCPUExpertObservations PlanningCPUExpertMeasurement::measure(const PlanningModelSource &source,
        const PlanningExpertSampleRequest &request, DeviceId device, const std::shared_ptr<PhysicalMemoryAuthority> &memory)
    {
        const auto sample = PlanningExpertSamplePlan::resolve(source, request).load(source, memory, device);
        return measure(sample, device, memory);
    }

    PlanningCPUExpertObservations PlanningCPUExpertMeasurement::measure(const PlanningLoadedExpertSample &sample,
        DeviceId device, const std::shared_ptr<PhysicalMemoryAuthority> &memory)
    {
        if (!device.is_cpu() || !memory || omp_in_parallel() || omp_get_dynamic())
            throw std::invalid_argument("CPU expert sampling requires admitted CPU storage and a fixed, non-nested production workshare");
        const auto description = describe(sample.plan(), PlanningSampleOrigin::PublishedPayload);
        const auto &request = sample.plan().request();
        const int workers = omp_get_max_threads();
        if (workers <= 0 || omp_get_thread_limit() < workers)
            throw std::invalid_argument("CPU expert sampling worker budget exceeds the OpenMP team limit");

        // Leases are declared before engines. The enclosing loaded sample owns
        // all source bytes through preparation, execution and engine retirement.
        std::vector<PhysicalMemoryAllocationLease> prepared_claims;
        std::array<std::shared_ptr<ITensorGemm>, 3> engines;
        const auto inputs = request.projections();
        using llaminar::v2::kernels::KernelFactory;
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            prepared_claims.push_back(memory->claimNewAllocation(device,
                PhysicalMemoryOwner::RoutedExpertWeights, description.preparation[i].persistent_bytes));
            std::optional<PhysicalMemoryAllocationLease> packing_claim;
            if (description.preparation[i].temporary_bytes)
                packing_claim.emplace(memory->claimNewAllocation(device,
                    PhysicalMemoryOwner::WeightLoadStaging, description.preparation[i].temporary_bytes));
            // A const shared borrow is kept inside the source-sample scope;
            // it cannot escape its PMA owner through a returned tensor view.
            engines[i] = KernelFactory::prepareExpertGemmLocal(sample.tensor(i).shared_from_this(), device);
            if (!engines[i] || !engines[i]->hasWeights() ||
                engines[i]->packedWeightBytes() != description.preparation[i].persistent_bytes)
                throw std::runtime_error("CPU expert preparation disagrees with its admitted native payload");
        }

        PlanningCPUExpertObservations result{device, workers, activeISALevel(), kCompiledISA,
            description.matrices[0].k, description.matrices[0].n, description.prepared_bytes,
            description.formats, request, {}};
        const MoEOverlayPreparedExpertTriplet triplet{engines[0], engines[1], engines[2]};
        for (const auto phase : {ExpertHistogramSource::DecodeToken, ExpertHistogramSource::PrefillChunk,
                                ExpertHistogramSource::GroupedVerifier})
        {
            const int rows = MoEOverlayCPUServiceMeasurement::rowsForPhase(phase);
            const uint64_t elapsed = MoEOverlayCPUServiceMeasurement::measurePrepared(
                triplet, device, description.layer, phase, geometry(description), memory);
            // Count the three full GEMM products, while timing the entire FFN
            // (including activation publication and SwiGLU). This is service
            // time for this exact repeated expert, not a DRAM-bandwidth probe.
            const double operations = 6.0 * rows * result.input_width * result.intermediate_width *
                MoEOverlayCPUServiceMeasurement::kMeasuredSamples;
            const std::string identity = "prepared CPU expert; model=" + sample.plan().modelPath() +
                "; source=" + request.gate.tensor_name + "/" + request.up.tensor_name + "/" + request.down.tensor_name +
                "; expert=" + std::to_string(std::get<PlanningExpertMatrix>(request.gate.selection).index) +
                "; device=" + device.toString() +
                "; workers=" + std::to_string(workers) + "; ISA=" + isaLevelName(result.isa) +
                "; compiled-ISA=" + isaLevelName(result.compiled_isa) +
                "; gate/up/down=" + result.formats[0] + "/" + result.formats[1] + "/" + result.formats[2] +
                "; input=" + std::to_string(result.input_width) + "; intermediate=" +
                std::to_string(result.intermediate_width) + "; rows=" + std::to_string(rows) +
                "; cache-regime=repeated-same-prepared-expert";
            result.phases.push_back({phase, rows, PlanningServiceObservation(
                PlanningWorkUnit::ArithmeticOperations, operations, double(elapsed) * 1e-9, identity)});
        }
        return result;
    }
}
