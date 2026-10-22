/**
 * @file Test__ROCmMoERouteOwnedGeometry.cpp
 * @brief Native resource admission and empty-capture rejection for route-owned tuning.
 *
 * Cold inspection authenticates every canonical physical decoder at each exact
 * K_LANES1 width, plus the unchanged fixed64 down controls. Public invalid calls
 * are observed in a real native capture and must leave zero nodes of any kind.
 * Admission/empty-capture cases guard scope and exact resource ownership.
 * The captured cases compare real production leaf transactions against an
 * independent serial-M1 byte oracle, including one real CPU-prepared promotion.
 * They do not certify QFN owner/compact geometry or
 * economy. Those larger operation/timing proofs belong to the canonical trainer.
 */
#include "backends/GPUDeviceContextPool.h"
#include "backends/BackendManager.h"
#include "backends/rocm/HIPGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/KernelFactory.h"
#include "kernels/IMoEKernel.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "utils/GpuPreparedGemmHarness.h"
#include "utils/TestTensorFactory.h"
#include "utils/QuantizedVerifierFormats.h"
#include <algorithm>
#include <cstring>
#include <vector>
#include "kernels/rocm/gemm/ROCmMoERouteOwnedTuning.h"
#include "tensors/NativeVnniFormatInfo.h"
#include "../kernels/moe/NativeVNNIExpertTransferParityTest.h"
#include <gtest/gtest.h>
#include <set>
#include <map>
#include <tuple>
#include <stdexcept>
#include <thread>
#include <cstdlib>
#include <string>
#include "utils/PerfStatsCollector.h"

#ifdef HAVE_ROCM
namespace
{
using namespace llaminar2;
using namespace llaminar2::rocm;

/** @brief Build exact physical identities from canonical source/migration authorities. */
std::set<std::uint8_t> compiledPhysicalCodebooks()
{
    std::set<std::uint8_t> result;
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
    {
        result.insert(canonicalDeviceVnniFormat(*source.metadata).codebook_id);
        result.insert(migrationStableDeviceVnniFormat(*source.metadata).codebook_id);
    }
    return result;
}

/** @brief Retain original source policy while inspecting every canonical/migration physical leaf. */
std::map<std::uint8_t, NativeVnniExecutionFormatEnvelope> compiledPhysicalEnvelopes()
{
    std::map<std::uint8_t, NativeVnniExecutionFormatEnvelope> result;
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
        for (const auto physical : {canonicalDeviceVnniFormat(*source.metadata).codebook_id,
                                   migrationStableDeviceVnniFormat(*source.metadata).codebook_id})
        {
            const NativeVnniSourceIdentity original{source.metadata->codebook_id, source.metadata->is_superblock, true};
            if (!addNativeVnniExecutionFormat(result[physical], physical, original, false))
                throw std::logic_error("canonical source could not construct its native resource envelope");
        }
    return result;
}

/** @brief Require real inspected allocation/residency at the actual leaf's launch geometry. */
void expectResources(const MoERouteOwnedKernelResources &resources, int threads)
{
    ASSERT_NE(resources.function, nullptr);
    ASSERT_NE(resources.specialization, nullptr);
    EXPECT_EQ(resources.block_threads, threads);
    EXPECT_GT(resources.registers_per_thread, 0);
    EXPECT_EQ(resources.local_memory_bytes_per_thread, 0u);
    EXPECT_GE(resources.max_threads_per_block, threads);
    EXPECT_GT(resources.max_active_blocks_per_sm, 0);
}
}

/** @brief Inspect all actual physical formats; source aliases do not become invented kernels. */
TEST(ROCmMoERouteOwnedGeometryNative, ExactLeafAdmissionAndScopeRetirement)
{
    auto &context = GPUDeviceContextPool::instance().getContext(DeviceId::rocm(0));
    context.submitAndWait([&]
    {
        for (const auto &[codebook, envelope] : compiledPhysicalEnvelopes())
        for (const auto geometry : kMoERouteOwnedGateUpGeometries)
        {
            SCOPED_TRACE(::testing::Message() << "codebook=" << int(codebook)
                << " columns=" << moeRouteOwnedGateUpColumns(geometry));
            const auto admission = admitMoERouteOwnedGateUpGeometry(geometry, envelope, 0);
            expectResources(admission.resources(), moeRouteOwnedGateUpColumns(geometry));
            EXPECT_EQ(admission.executionCodebook(), codebook);
            EXPECT_EQ(admission.deviceOrdinal(), 0);
            const auto expected_lds = codebook == 13 ? 1024u * sizeof(std::uint64_t) : 0u;
            EXPECT_EQ(admission.resources().dynamic_shared_memory_bytes, expected_lds);
            EXPECT_FALSE(currentMoERouteOwnedGateUpGeometry());
            {
                ScopedMoERouteOwnedGateUpGeometry scope(admission);
                EXPECT_EQ(currentMoERouteOwnedGateUpGeometry(), geometry);
                EXPECT_THROW((void)ScopedMoERouteOwnedGateUpGeometry{admission}, std::logic_error);
                EXPECT_EQ(currentMoERouteOwnedGateUpGeometry(), geometry);
                bool other_thread_inactive = false;
                std::thread other([&] { other_thread_inactive = !currentMoERouteOwnedGateUpGeometry(); });
                other.join();
                EXPECT_TRUE(other_thread_inactive);
            }
            EXPECT_FALSE(currentMoERouteOwnedGateUpGeometry());
        }
        for (const auto codebook : compiledPhysicalCodebooks())
        for (const auto layout : {MoEPrefillOutputLayout::OriginalRouteRows, MoEPrefillOutputLayout::TokenRows})
            expectResources(inspectMoERouteOwnedDown64(codebook, layout, 0), 64);
        EXPECT_THROW((void)admitMoERouteOwnedGateUpGeometry(
            MoERouteOwnedGateUpGeometry::TN64, 255, 0), std::invalid_argument);
        EXPECT_THROW((void)admitMoERouteOwnedGateUpGeometry(
            MoERouteOwnedGateUpGeometry::TN64, 19, -1), std::invalid_argument);
        EXPECT_THROW((void)inspectMoERouteOwnedDown64(
            19, static_cast<MoEPrefillOutputLayout>(255), 0), std::invalid_argument);
    });
}

/** @brief Public scoped failures precede preparation/device work and record no native node. */
TEST(ROCmMoERouteOwnedGeometryNative, PublicInvalidBindingsLeaveEmptyNativeCapture)
{
    const auto device = DeviceId::rocm(0);
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&]
    {
        auto *stream = context.defaultStream();
        ASSERT_NE(stream, nullptr);
        const auto admission = admitMoERouteOwnedGateUpGeometry(
            MoERouteOwnedGateUpGeometry::TN128, 19, 0);
        auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
        ASSERT_NE(kernel, nullptr);
        kernel->setGPUStream(stream);
        EXPECT_THROW(kernel->setGPUStream(nullptr), std::invalid_argument);
        auto graph = context.createGraphCapture(stream);
        ASSERT_NE(graph, nullptr);
        auto *native = dynamic_cast<HIPGraphCapture *>(graph.get());
        ASSERT_NE(native, nullptr);
        {
            ScopedMoERouteOwnedGateUpGeometry scope(admission);
            ScopedBackendGraphCapture recording(context, *graph, "route-owned invalid public calls");
            ASSERT_TRUE(recording.begin());
            EXPECT_THROW((void)ScopedMoERouteOwnedGateUpGeometry{admission}, std::logic_error);
            EXPECT_THROW((void)admitMoERouteOwnedGateUpGeometry(
                MoERouteOwnedGateUpGeometry::TN64, 19, 0), std::logic_error);
            EXPECT_THROW(validateMoERouteOwnedGateUpScopeBinding(0, nullptr, 1,
                MoEPrefillProjectionExecution::gateUp(256)), std::logic_error);
            for (void *invalid : {static_cast<void *>(hipStreamLegacy), static_cast<void *>(hipStreamPerThread)})
            {
                kernel->setGPUStream(invalid);
                EXPECT_THROW((void)kernel->executeGroupedPrefillProjection(nullptr, nullptr,
                    -1, -1, 1, 256, 256, 3, 2, MoEPrefillProjectionExecution::gateUp(256)), std::logic_error);
            }
            kernel->setGPUStream(stream);
            EXPECT_THROW((void)kernel->executeGroupedPrefillProjection(nullptr, nullptr,
                -1, -1, 2, 256, 256, 3, 2, MoEPrefillProjectionExecution::gateUp(256)), std::logic_error);
            EXPECT_THROW((void)kernel->executeGroupedPrefillProjection(nullptr, nullptr,
                -1, -1, 1, 256, 256, 3, 2, MoEPrefillProjectionExecution::down(256, 0, 64)), std::logic_error);
            EXPECT_THROW(validateMoERouteOwnedGateUpScopeBinding(1, stream, 1,
                MoEPrefillProjectionExecution::gateUp(256)), std::logic_error);
            validateMoERouteOwnedGateUpCodebook(1u << 19, 1u << 19);
            EXPECT_THROW(validateMoERouteOwnedGateUpCodebook(0, 1u << 19), std::logic_error);
            EXPECT_THROW(validateMoERouteOwnedGateUpCodebook(1u << 19, 0), std::logic_error);
            EXPECT_THROW(validateMoERouteOwnedGateUpCodebook((1u << 19) | 1u, 1u << 19), std::logic_error);
            EXPECT_THROW(validateMoERouteOwnedGateUpCodebook(1u << 19, (1u << 19) | 1u), std::logic_error);
            EXPECT_FALSE(lastMoERouteOwnedGateUpLaunch());
            recording.finish();
        }
        EXPECT_FALSE(currentMoERouteOwnedGateUpGeometry());
        std::size_t nodes = 0;
        ASSERT_EQ(hipGraphGetNodes(native->graph(), nullptr, &nodes), hipSuccess);
        EXPECT_EQ(nodes, 0u);
        // Ordinary failed preparation clears diagnostic observation without an override/HIP probe.
        EXPECT_FALSE(kernel->executeGroupedPrefillProjection(nullptr, nullptr,
            -1, -1, 0, 256, 256, 3, 2, MoEPrefillProjectionExecution::gateUp(256)));
        EXPECT_FALSE(lastMoERouteOwnedGateUpLaunch());
        try
        {
            ScopedMoERouteOwnedGateUpGeometry scope(admission);
            throw std::runtime_error("fixture unwind");
        }
        catch (const std::runtime_error &) {}
        EXPECT_FALSE(currentMoERouteOwnedGateUpGeometry());
    });
}
/**
 * @brief Inspect exact worker-plane resources and reject shape mismatches before any enqueue.
 *
 * Every physical decoder is explicitly admitted or rejected. The source-policy
 * resolver, not execution codebook or capacity64, owns the partial-bank extent.
 */
TEST(ROCmMoERouteOwnedGeometryNative, ParallelSourcePartitionAdmissionRejectsIncompatibleBindings)
{
    using P = MoERouteOwnedGateUpPartitionExecution;
    constexpr MoERouteOwnedProjectionShape shape{.output_columns = 96, .reduction_columns = 2560};
    const auto device = DeviceId::rocm(0);
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&]
    {
        for (const auto &[codebook, envelope] : compiledPhysicalEnvelopes())
        {
            if (moeRouteOwnedGateUpPartitionExecutionValid(P::FourWorkerPlanes, envelope, shape))
            {
                const auto parallel = admitMoERouteOwnedGateUpGeometry(MoERouteOwnedGateUpGeometry::TN64,
                    envelope, 0, P::FourWorkerPlanes, shape);
                const auto serial = admitMoERouteOwnedGateUpGeometry(MoERouteOwnedGateUpGeometry::TN64,
                    envelope, 0, P::Serial, shape);
                EXPECT_EQ(parallel.partitionExecution(), P::FourWorkerPlanes);
                EXPECT_EQ(parallel.reductionWidth(), shape.reduction_columns);
                EXPECT_EQ(parallel.outputWidth(), shape.output_columns);
                EXPECT_EQ(parallel.arithmeticPartitions(), serial.arithmeticPartitions());
                EXPECT_EQ(parallel.maximumKPartitions(), moeRouteOwnedGateUpMaximumKPartitions(
                    envelope, parallel.arithmeticPartitions(), shape.reduction_columns));
                expectResources(parallel.resources(), 256);
                EXPECT_EQ(parallel.resources().dynamic_shared_memory_bytes,
                    moeRouteOwnedGateUpK1SharedLayout(codebook, P::FourWorkerPlanes, parallel.maximumKPartitions()).total_bytes);
                EXPECT_NE(parallel.resources().function, serial.resources().function);
            }
            else EXPECT_THROW((void)admitMoERouteOwnedGateUpGeometry(MoERouteOwnedGateUpGeometry::TN64,
                envelope, 0, P::FourWorkerPlanes, shape), std::invalid_argument);
        }
        const NativeVnniExecutionFormatEnvelope corrected{1u << 5, 1u << 5};
        for (auto geometry : {MoERouteOwnedGateUpGeometry::TN128, MoERouteOwnedGateUpGeometry::TN256})
            EXPECT_THROW((void)admitMoERouteOwnedGateUpGeometry(geometry,
                corrected, 0, P::FourWorkerPlanes, shape), std::invalid_argument);
        for (const auto invalid : {MoERouteOwnedProjectionShape{}, {96, 0}, {0, 2560}, {96, 31}, {-1, 2560}})
            EXPECT_THROW((void)admitMoERouteOwnedGateUpGeometry(MoERouteOwnedGateUpGeometry::TN64,
                corrected, 0, P::FourWorkerPlanes, invalid), std::invalid_argument);
        EXPECT_THROW((void)admitMoERouteOwnedGateUpGeometry(MoERouteOwnedGateUpGeometry::TN64,
            corrected, 0, static_cast<P>(255), shape), std::invalid_argument);
        const auto admission = admitMoERouteOwnedGateUpGeometry(MoERouteOwnedGateUpGeometry::TN64,
            corrected, 0, P::FourWorkerPlanes, shape);
        auto *stream = context.defaultStream();
        auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
        ASSERT_NE(kernel, nullptr);
        kernel->setGPUStream(stream);
        auto graph = context.createGraphCapture(stream);
        ASSERT_NE(graph, nullptr);
        auto *native = dynamic_cast<HIPGraphCapture *>(graph.get());
        ASSERT_NE(native, nullptr);
        {
            ScopedMoERouteOwnedGateUpGeometry scope(admission);
            ScopedBackendGraphCapture recording(context, *graph, "partition workers reject foreign N/K before preparation");
            ASSERT_TRUE(recording.begin());
            EXPECT_THROW(validateMoERouteOwnedGateUpScopeBinding(0, stream, 1,
                MoEPrefillProjectionExecution::gateUp(2560), {96, 32}), std::logic_error);
            EXPECT_THROW(validateMoERouteOwnedGateUpScopeBinding(0, stream, 1,
                MoEPrefillProjectionExecution::gateUp(2560), {64, 2560}), std::logic_error);
            EXPECT_THROW((void)kernel->executeGroupedPrefillProjection(nullptr, nullptr,
                -1, -1, 1, 32, 96, 1, 1, MoEPrefillProjectionExecution::gateUp(32)), std::logic_error);
            EXPECT_THROW((void)kernel->executeGroupedPrefillProjection(nullptr, nullptr,
                -1, -1, 1, 2560, 64, 1, 1, MoEPrefillProjectionExecution::gateUp(2560)), std::logic_error);
            EXPECT_FALSE(lastMoERouteOwnedGateUpLaunchSet());
            recording.finish();
        }
        std::size_t nodes = 0;
        ASSERT_EQ(hipGraphGetNodes(native->graph(), nullptr, &nodes), hipSuccess);
        EXPECT_EQ(nodes, 0u);
    });
}

namespace
{
/** @brief Own only the exact collector filter; restore original environment values and policy. */
class OwnerSubmissionObservation final
{
    std::optional<std::string> json_, filter_;
    /** @return Exact previous value, distinguishing absence from the empty string. */
    static std::optional<std::string> prior(const char *name)
    { const auto *value = std::getenv(name); return value ? std::optional<std::string>{value} : std::nullopt; }
    /** @brief Restore exact collector configuration; restoration failure is fatal. */
    static void restore(const char *name, const std::optional<std::string> &value)
    { if ((value ? setenv(name, value->c_str(), 1) : unsetenv(name)) != 0) std::terminate(); }
public:
    /** @brief Observe accepted owner submissions without enabling timers or unrelated domains. */
    OwnerSubmissionObservation() : json_(prior("LLAMINAR_PERF_STATS_JSON")), filter_(prior("LLAMINAR_PERF_STATS_FILTER"))
    {
        if (setenv("LLAMINAR_PERF_STATS_JSON", "1", 1) != 0 ||
            setenv("LLAMINAR_PERF_STATS_FILTER", "kernel.rocm_moe_owner_m1_partition_submissions", 1) != 0)
        {
            restore("LLAMINAR_PERF_STATS_FILTER", filter_); restore("LLAMINAR_PERF_STATS_JSON", json_);
            throw std::runtime_error("Cannot select owner submission observation");
        }
        PerfStatsCollector::reloadConfigurationFromEnvironment(); PerfStatsCollector::reset();
    }
    /** @brief Retire test records and restore exact original collector policy. */
    ~OwnerSubmissionObservation()
    {
        PerfStatsCollector::reset(); restore("LLAMINAR_PERF_STATS_FILTER", filter_);
        restore("LLAMINAR_PERF_STATS_JSON", json_); PerfStatsCollector::reloadConfigurationFromEnvironment();
    }
    OwnerSubmissionObservation(const OwnerSubmissionObservation &) = delete;
    OwnerSubmissionObservation &operator=(const OwnerSubmissionObservation &) = delete;
};
/** @return Exact family only; neighboring counter names cannot satisfy replay assertions. */
std::vector<PerfStatRecord> ownerSubmissionRecords()
{
    auto records = PerfStatsCollector::snapshot({"kernel.rocm_moe_owner_m1_partition_submissions"});
    std::erase_if(records, [](const auto &record) { return record.domain != "kernel" || record.name != "rocm_moe_owner_m1_partition_submissions"; });
    return records;
}

/**
 * @brief Original serial-byte witness for serial and four-plane source-partition execution.
 * @param requested_execution Original serial policy or paired serial/four-plane TN64 execution.
 * @param H Actual reduction width; source-superblock cases require a 256-value multiple.
 * @param F Actual expert width, preserving a partial TN64 output tile when not divisible by64.
 * @param promote_asymmetric Exercise actual codebook23 values through original CPU preparation and promotion.
 * @param rows Actual token rows; two with topK9 exercises ordinary IQ2_S capacity18 without widening M1 scope.
 * @param top_k Actual routes per row, bounded by the original serial oracle.
 * @param iq2_only Restrict the capacity-boundary witness to the actual IQ2_S source.
 * @param model_binding Exercise table-owned explicit lowering without any diagnostic scope.
 *
 * Static and genuine runtime-placement tables retain their actual envelopes.
 * All threads reach partial-bank publication before N-tail exits. The independent serial
 * oracle and every partition/FP32 expression remain the original public path.
 */
void verifyCapturedPublicM1Sources(MoERouteOwnedGateUpPartitionExecution requested_execution, int H, int F,
    bool promote_asymmetric = false, int rows = 1, int top_k = 1, bool iq2_only = false, bool model_binding = false)
{
    using namespace llaminar2::test;
    constexpr int Guard = 16, Replays = 20;
    const int E = top_k, TopK = top_k;
    ASSERT_GT(rows, 0);
    ASSERT_LE(rows, 8);
    ASSERT_GT(TopK, 0);
    ASSERT_LE(TopK, 16);
    const int live = rows * H;
    const auto device = DeviceId::rocm(0);
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&]
    {
        auto *backend = getBackendFor(device);
        auto *stream = context.defaultStream();
        ASSERT_NE(backend, nullptr);
        ASSERT_NE(stream, nullptr);
        /** @brief Keep the fixture's stream joined before borrowed storage is retired. */
        struct Join
        {
            IBackend *backend; void *stream;
            /** @brief Join only the exact test stream at its observation/retirement boundary. */
            ~Join() { (void)backend->synchronizeStream(stream, 0); }
        };
        std::vector<std::tuple<QuantizedVerifierFormatCase, bool, bool, bool>> cases;
        for (const bool runtime_mutable : {false, true})
        {
            for (const auto &format : quantizedMoEVerifierFormats()) cases.emplace_back(format, false, runtime_mutable, false);
            cases.emplace_back(quantizedMoEVerifierFormat("Q4_K"), true, runtime_mutable, false);
            cases.emplace_back(quantizedMoEVerifierFormat("IQ4_XS"), false, runtime_mutable, true);
        }
        for (const auto &[format, q5_down, runtime_mutable, iq4_mixed] : cases)
        {
            if (model_binding && !runtime_mutable) continue;
            if (promote_asymmetric && std::string_view(format.label) != "Q5_K") continue;
            if (iq2_only && std::string_view(format.label) != "IQ2_S") continue;
            if (format.source_is_superblock && H % 256 != 0) continue;
            SCOPED_TRACE(std::string(format.label) + (q5_down ? "/Q5_1" : "/Q8_0") +
                (runtime_mutable ? "/RuntimePlacementTable" : "/StaticDescriptorTable") +
                (iq4_mixed ? "/IQ4_NL-up" : ""));
            // All original source aliases plus the dominant mixed QFN tuple.
            // Source K retains its block alignment; down K=F uses original Q8/Q5_1.
            auto make_gate = [&](unsigned seed) -> std::shared_ptr<TensorBase>
            { return format.create({static_cast<size_t>(F), static_cast<size_t>(H)}, seed); };
            std::vector<std::shared_ptr<TensorBase>> sources{
                make_gate(87200), iq4_mixed
                    ? std::shared_ptr<TensorBase>(quantizedMoEVerifierFormat("IQ4_NL").create({static_cast<size_t>(F), static_cast<size_t>(H)}, 87300))
                    : make_gate(87300),
                q5_down
                    ? std::shared_ptr<TensorBase>(TestTensorFactory::createQ5_1Random({static_cast<size_t>(H), static_cast<size_t>(F)}, 87400))
                    : std::shared_ptr<TensorBase>(TestTensorFactory::createQ8_0Random({static_cast<size_t>(H), static_cast<size_t>(F)}, 87400))};
            std::vector<GpuPreparedGemm> prepared;
            for (std::size_t i = 0; i < sources.size(); ++i)
                prepared.push_back(makeGpuPreparedGemm(sources[i].get(), device,
                    "test.route_owned.geometry." + std::string(format.label) + "." + std::to_string(q5_down) +
                    ".mutable" + std::to_string(runtime_mutable) + ".iq4mixed" + std::to_string(iq4_mixed) +
                    ".k" + std::to_string(H) + ".f" + std::to_string(F) + ".partition_execution" + std::to_string(static_cast<int>(requested_execution)) + "." + std::to_string(i)));
            DeviceNativeVNNIMatrixDesc gate{}, up{}, down{};
            ASSERT_TRUE(prepared[0].kernel->exportNativeVNNIMatrixDesc(gate));
            ASSERT_TRUE(prepared[1].kernel->exportNativeVNNIMatrixDesc(up));
            ASSERT_TRUE(prepared[2].kernel->exportNativeVNNIMatrixDesc(down));
            ASSERT_TRUE(areNativeVnniGateUpCodebooksCompatible(gate.codebook_id, up.codebook_id));
            const auto original_gate = gate;
            const auto original_up = up;
            // Retain genuine CPU-prepared bytes and original TransferEngine
            // promotion owners through every descriptor/capture/replay. No
            // source-format relabel or test-only weight repacking is used.
            using namespace native_vnni_transfer_parity_detail;
            std::array<cpu::native_vnni::CPUNativeVNNIPackedWeights, 2> cpu_weights;
            std::optional<PromotedCPUProjection> promoted_gate, promoted_up;
            if (promote_asymmetric)
            {
                for (std::size_t index = 0; index < cpu_weights.size(); ++index)
                {
                    ASSERT_TRUE(cpu::native_vnni::packWeightsCPUNativeVNNI(sources[index].get(), cpu_weights[index]));
                    ASSERT_TRUE(cpu_weights[index].usesExpandedInt8());
                    ASSERT_TRUE(cpu_weights[index].is_asymmetric);
                }
                promoted_gate = promoteCPUProjection(backend, device, cpu_weights[0], ExpertTierWeightProjection::Gate, 88801);
                promoted_up = promoteCPUProjection(backend, device, cpu_weights[1], ExpertTierWeightProjection::Up, 88802);
                gate = promoted_gate->descriptor;
                up = promoted_up->descriptor;
                ASSERT_EQ(gate.codebook_id, kNativeVnniExpandedInt8MinCodebook);
                ASSERT_EQ(up.codebook_id, kNativeVnniExpandedInt8MinCodebook);
                ASSERT_EQ(gate.source_codebook_id, original_gate.source_codebook_id);
                ASSERT_EQ(gate.source_is_superblock, original_gate.source_is_superblock);
                ASSERT_NE(gate.source_identity_present, 0u);
            }
            // Distinct route slots may borrow identical retained expert weights.
            // Host descriptor vectors outlive the metadata-upload retirement join.
            const std::vector<DeviceNativeVNNIMatrixDesc> gates(E, gate), ups(E, up), downs(E, down);
            const std::vector<DeviceNativeVNNIMatrixDesc> original_gates(E, original_gate), original_ups(E, original_up);
            const auto requirements = MoEWorkspaceBuffers::rocmMoE(rows, H, F, E, TopK);
            DeviceWorkspaceManager workspace(device, requirements.total_bytes_with_alignment());
            ASSERT_TRUE(workspace.allocate(requirements));
            auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
            ASSERT_NE(kernel, nullptr);
            kernel->setGPUStream(stream);
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
            ASSERT_NE(consumer, nullptr);
            consumer->bindWorkspace(&workspace);
            Join metadata_join{backend, stream};
            const auto origin = runtime_mutable ? MoEDecodeDescriptorSource::RuntimePlacementTable
                                                : MoEDecodeDescriptorSource::StaticDescriptorTable;
            NativeVnniExecutionFormatEnvelope envelope;
            for (const auto *desc : {&gate, &up})
                ASSERT_TRUE(addNativeVnniExecutionFormat(envelope, desc->codebook_id,
                    NativeVnniSourceIdentity{desc->source_codebook_id, desc->source_is_superblock != 0u,
                                            desc->source_identity_present != 0u}, runtime_mutable));
            const int gate_table = kernel->uploadGroupedExpertGateUpDescriptorTables(gates.data(), ups.data(), E, H, F, origin);
            const int down_table = kernel->uploadGroupedExpertDownDescriptorTable(downs.data(), E, H, F, origin);
            ASSERT_GE(gate_table, 0);
            ASSERT_GE(down_table, 0);
            // Only the independent public serial oracle uses immutable tables.
            // The actual capture below retains its genuine runtime envelope.
            const int oracle_gate_table = (runtime_mutable || promote_asymmetric)
                ? kernel->uploadGroupedExpertGateUpDescriptorTables(original_gates.data(), original_ups.data(), E, H, F,
                    MoEDecodeDescriptorSource::StaticDescriptorTable) : gate_table;
            const int oracle_down_table = runtime_mutable
                ? kernel->uploadGroupedExpertDownDescriptorTable(downs.data(), E, H, F,
                    MoEDecodeDescriptorSource::StaticDescriptorTable) : down_table;
            ASSERT_GE(oracle_gate_table, 0);
            ASSERT_GE(oracle_down_table, 0);
            auto hidden = TestTensorFactory::createFP32({static_cast<size_t>(rows), static_cast<size_t>(H)});
            auto ids = TestTensorFactory::createFP32({static_cast<size_t>(rows), static_cast<size_t>(TopK)});
            auto weights = TestTensorFactory::createFP32({static_cast<size_t>(rows), static_cast<size_t>(TopK)});
            auto serial = TestTensorFactory::createFP32({static_cast<size_t>(live + Guard)});
            auto output = TestTensorFactory::createFP32({static_cast<size_t>(live + Guard)});
            const std::vector<float> poison(live + Guard, -1234.5f);
            std::vector<float> route_ids(rows * TopK), route_weights(rows * TopK, 0.625f / TopK);
            for (int slot = 0; slot < rows * TopK; ++slot) route_ids[slot] = static_cast<float>(slot % TopK);
            std::vector<std::vector<float>> input(Replays, std::vector<float>(live));
            std::vector<std::vector<float>> expected(Replays, std::vector<float>(live + Guard));
            // The join is declared after every host/device transfer owner.
            Join storage_join{backend, stream};
            for (auto *tensor : {hidden.get(), ids.get(), weights.get(), serial.get(), output.get()})
                ASSERT_TRUE(tensor->ensureOnDevice(device, stream));
            ASSERT_TRUE(backend->hostToDevice(ids->gpu_data_ptr(), route_ids.data(), route_ids.size() * sizeof(float), 0, stream));
            ASSERT_TRUE(backend->hostToDevice(weights->gpu_data_ptr(), route_weights.data(), route_weights.size() * sizeof(float), 0, stream));
            for (int replay = 0; replay < Replays; ++replay)
            {
                for (int k = 0; k < live; ++k)
                {
                    // Capacity-boundary rows deliberately share input values,
                    // so each has the same independent public serial-M1 oracle.
                    input[replay][k] = .0071f * (int(((k % H) * 19 + replay * 13) % 101) - 50);
                    if (requested_execution == MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes)
                    {
                        // Cancellation, positive/negative zero and changing scale
                        // histories must retain the original partition fold bytes.
                        if (replay % 4 == 0) input[replay][k] = (k & 1 ? 1.f : -1.f) * (1 + replay / 4);
                        if (replay % 4 == 1) input[replay][k] = (k % 32 < 16 ? .125f : -.125f) * (1 + replay / 4);
                        if (replay % 4 == 2) input[replay][k] = k & 1 ? 0.0f : -0.0f;
                    }
                }
                ASSERT_TRUE(backend->hostToDevice(hidden->gpu_data_ptr(), input[replay].data(), live * sizeof(float), 0, stream));
                ASSERT_TRUE(backend->hostToDevice(serial->gpu_data_ptr(), poison.data(), poison.size() * sizeof(float), 0, stream));
                ASSERT_TRUE(kernel->groupedExpertDecodeFromRouting(hidden.get(), ids.get(), weights.get(),
                    oracle_gate_table, oracle_down_table, TopK, serial.get(), H, F));
                ASSERT_TRUE(backend->deviceToHost(expected[replay].data(), serial->gpu_data_ptr(),
                    expected[replay].size() * sizeof(float), 0, stream));
                ASSERT_TRUE(backend->synchronizeStream(stream, 0));
                // The public oracle computes one row only. Its untouched rest
                // proves it did not silently process or overwrite extra capacity.
                for (int column = H; column < live + Guard; ++column)
                    ASSERT_EQ(expected[replay][column], poison[column]);
                for (int row = 1; row < rows; ++row)
                    std::copy_n(expected[replay].begin(), H, expected[replay].begin() + row * H);
            }
            const auto complete = [&]
            {
                return kernel->prepareExpertGroupsAsync(ids.get(), weights.get(), rows, E, TopK) &&
                    kernel->executeGroupedPrefillPipeline(hidden.get(), output.get(), gate_table,
                        down_table, rows, H, F, E, TopK);
            };
            if (requested_execution == MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes)
            {
                // Capture the real unscoped production default independently. In
                // particular Q6_K/IQ2_S use adjacent-K_LANES defaults, not Serial64/K1.
                auto automatic = context.createGraphCapture(stream);
                ASSERT_NE(automatic, nullptr);
                std::vector<float> actual(live + Guard);
                Join automatic_join{backend, stream};
                ASSERT_TRUE(complete());
                ASSERT_TRUE(backend->synchronizeStream(stream, 0));
                {
                    ScopedBackendGraphCapture recording(context, *automatic, "route-owned untouched automatic M1 control");
                    ASSERT_TRUE(recording.begin());
                    ASSERT_TRUE(complete());
                    recording.finish();
                }
                ASSERT_FALSE(lastMoERouteOwnedGateUpLaunchSet());
                ASSERT_TRUE(automatic->instantiate());
                std::vector<GPUGraphKernelNodeInfo> nodes;
                std::string error;
                ASSERT_TRUE(automatic->inspectKernelNodes(nodes, &error)) << error;
                std::uint32_t seen = 0;
                for (const auto &node : nodes)
                {
                    if (node.name.find("moe_grouped_gate_up_decode_equivalent_prefill_kernel") == std::string::npos) continue;
                    ASSERT_TRUE(node.name_resolved);
                    std::uint8_t cb = 255;
                    const auto mask = moeRouteOwnedGateUpLaunchCodebookMask(envelope.execution_codebook_mask);
                    for (std::uint8_t candidate = 0; candidate < 32; ++candidate)
                        if ((mask & nativeVnniCodebookMaskBit(candidate)) &&
                            node.name.find("IL23NativeVNNIFormatPrefill" + std::to_string(candidate) + "E") != std::string::npos)
                            cb = candidate;
                    ASSERT_NE(cb, 255);
                    ASSERT_EQ(seen & nativeVnniCodebookMaskBit(cb), 0u);
                    seen |= nativeVnniCodebookMaskBit(cb);
                    const int columns = cb == 13 && rows * TopK > 16 ? 128 : 64;
                    const int k_lanes = cb == 8 || (cb == 13 && rows * TopK <= 16) ? 4 : cb == 13 ? 2 : 1;
                    const auto expected_lds = moeRouteOwnedGateUpCodebookSharedBytes(cb) +
                        (k_lanes > 1 ? std::size_t(H) + std::size_t(H / 32) * sizeof(float) : 0u);
                    EXPECT_EQ(node.grid_x, 1 + (F - 1) / columns);
                    EXPECT_EQ(node.grid_y, rows * TopK); EXPECT_EQ(node.grid_z, 2u);
                    EXPECT_EQ(node.block_x, columns * k_lanes);
                    EXPECT_EQ(node.dynamic_shared_memory_bytes, expected_lds);
                    EXPECT_EQ(node.local_memory_bytes_per_thread, 0u);
                }
                EXPECT_EQ(seen, moeRouteOwnedGateUpLaunchCodebookMask(envelope.execution_codebook_mask));
                for (int replay = 0; replay < Replays; ++replay)
                {
                    ASSERT_TRUE(backend->hostToDevice(hidden->gpu_data_ptr(), input[replay].data(), live * sizeof(float), 0, stream));
                    ASSERT_TRUE(backend->hostToDevice(output->gpu_data_ptr(), poison.data(), poison.size() * sizeof(float), 0, stream));
                    ASSERT_TRUE(automatic->launch());
                    ASSERT_TRUE(backend->deviceToHost(actual.data(), output->gpu_data_ptr(), actual.size() * sizeof(float), 0, stream));
                    ASSERT_TRUE(backend->synchronizeStream(stream, 0));
                    ASSERT_EQ(std::memcmp(actual.data(), expected[replay].data(), actual.size() * sizeof(float)), 0)
                        << "automatic source=" << format.label << " rows=" << rows << " replay=" << replay;
                    ASSERT_FALSE(lastMoERouteOwnedGateUpLaunchSet());
                }
            }
            if (model_binding)
            {
                ASSERT_EQ(rows, 1);
                ASSERT_TRUE(runtime_mutable);
                OwnerSubmissionObservation observation;
                const auto plan = planMoERouteOwnedGateUp(MoERouteOwnedGateUpGeometry::TN64, envelope, 0,
                    MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes,
                    {.output_columns = F, .reduction_columns = H});
                const auto intent_before_resources = plan.identity();
                const auto capability = admitMoERouteOwnedGateUpGeometry(plan);
                ASSERT_EQ(plan.identity(), intent_before_resources);
                const bool selected = compiledMoERouteOwnedModelPartitionPolicy() ==
                    MoERouteOwnedModelPartitionPolicy::FourWorkerPlanes;
                const auto expected_intent = selected ? plan.identity() : 0u;
                EXPECT_THROW((void)kernel->prepareOwnerGateUpCaptureBinding(gate_table, rows,
                    plan.identity() ^ 1u), std::logic_error);
                EXPECT_EQ(kernel->ownerGateUpCaptureBindingSignature(gate_table, rows), 0u);
                EXPECT_THROW(kernel->requireOwnerGateUpCaptureBinding(gate_table, rows, plan.identity()), std::logic_error);
                kernel->requireOwnerGateUpCaptureBinding(gate_table, rows, 0);
                {
                    auto missing = context.createGraphCapture(stream); ASSERT_NE(missing, nullptr);
                    auto *native_missing = dynamic_cast<HIPGraphCapture *>(missing.get()); ASSERT_NE(native_missing, nullptr);
                    Join missing_join{backend, stream};
                    {
                        ScopedBackendGraphCapture recording(context, *missing, "missing declared owner policy cannot record serial");
                        ASSERT_TRUE(recording.begin());
                        EXPECT_THROW(kernel->requireOwnerGateUpCaptureBinding(gate_table, rows, plan.identity()), std::logic_error);
                        recording.finish();
                    }
                    std::size_t missing_nodes=0;
                    ASSERT_EQ(hipGraphGetNodes(native_missing->graph(), nullptr, &missing_nodes), hipSuccess);
                    EXPECT_EQ(missing_nodes,0u);
                }
                ASSERT_EQ(kernel->prepareOwnerGateUpCaptureBinding(gate_table, rows, expected_intent), expected_intent);
                ASSERT_EQ(kernel->prepareOwnerGateUpCaptureBinding(gate_table, rows, expected_intent), expected_intent);
                ASSERT_EQ(kernel->ownerGateUpCaptureBindingSignature(gate_table, rows), expected_intent);
                // Pure identity is stable and cannot materialize resources during capture.
                auto rejected = context.createGraphCapture(stream);
                ASSERT_NE(rejected, nullptr);
                auto *native_rejected = dynamic_cast<HIPGraphCapture *>(rejected.get());
                ASSERT_NE(native_rejected, nullptr);
                Join rejected_join{backend, stream};
                {
                    ScopedBackendGraphCapture recording(context, *rejected, "owner binding rejects cold work during recording");
                    ASSERT_TRUE(recording.begin());
                    EXPECT_THROW((void)kernel->prepareOwnerGateUpCaptureBinding(gate_table, rows, expected_intent), std::logic_error);
                    EXPECT_EQ(kernel->ownerGateUpCaptureBindingSignature(gate_table, rows), expected_intent);
                    EXPECT_THROW(kernel->requireOwnerGateUpCaptureBinding(gate_table, rows,
                        selected ? (expected_intent ^ 1u) : plan.identity()), std::logic_error);
                    if (selected)
                        EXPECT_THROW((void)kernel->executeGroupedPrefillProjection(nullptr, nullptr, gate_table,
                            down_table, rows, H, F + 32, E, TopK, MoEPrefillProjectionExecution::gateUp(H)), std::logic_error);
                    recording.finish();
                }
                std::size_t rejected_nodes = 0;
                ASSERT_EQ(hipGraphGetNodes(native_rejected->graph(), nullptr, &rejected_nodes), hipSuccess);
                EXPECT_EQ(rejected_nodes, 0u);
                if (selected)
                {
                    ScopedMoERouteOwnedGateUpGeometry conflicting(capability);
                    EXPECT_THROW((void)kernel->executeGroupedPrefillProjection(nullptr, nullptr, gate_table,
                        down_table, rows, H, F, E, TopK, MoEPrefillProjectionExecution::gateUp(H)), std::logic_error);
                }
                EXPECT_TRUE(ownerSubmissionRecords().empty());
                // Ordinary component complete control remains separate. The model
                // candidate is explicitly two phases borrowing the same original
                // grouped scratch, not a second producer or a hidden scope.
                const auto attached = [&] {
                    return kernel->prepareExpertGroupsAsync(ids.get(), weights.get(), rows, E, TopK) &&
                        kernel->executeGroupedPrefillProjection(hidden.get(), nullptr, gate_table, down_table,
                            rows, H, F, E, TopK, MoEPrefillProjectionExecution::gateUp(H)) &&
                        kernel->executeGroupedPrefillProjection(nullptr, output.get(), gate_table, down_table,
                            rows, H, F, E, TopK, MoEPrefillProjectionExecution::down(H, 0, H));
                };
                ASSERT_TRUE(attached());
                ASSERT_TRUE(backend->synchronizeStream(stream, 0));
                PerfStatsCollector::reset();
                auto graph = context.createGraphCapture(stream);
                ASSERT_NE(graph, nullptr);
                std::vector<float> actual(live + Guard);
                Join graph_join{backend, stream};
                {
                    ScopedBackendGraphCapture recording(context, *graph, "explicit table-owned owner partition policy");
                    ASSERT_TRUE(recording.begin()); ASSERT_TRUE(attached()); recording.finish();
                }
                ASSERT_FALSE(currentMoERouteOwnedGateUpGeometry());
                ASSERT_FALSE(lastMoERouteOwnedGateUpLaunchSet()); // No model receipt is published through TLS.
                ASSERT_TRUE(graph->instantiate());
                std::vector<GPUGraphKernelNodeInfo> nodes; std::string error;
                ASSERT_TRUE(graph->inspectKernelNodes(nodes, &error)) << error;
                const auto captured_records = ownerSubmissionRecords();
                if (selected)
                {
                    ASSERT_EQ(captured_records.size(), capability.leafCount());
                    for (const auto &record : captured_records)
                    {
                        EXPECT_EQ(record.count, 1u); EXPECT_EQ(record.value, 1.0);
                        EXPECT_EQ(record.phase, "submission");
                        EXPECT_EQ(record.tags.at("observation"), "host_execute_or_capture");
                        EXPECT_EQ(record.tags.at("intent"), std::to_string(expected_intent));
                        EXPECT_EQ(record.tags.at("n"), std::to_string(F)); EXPECT_EQ(record.tags.at("k"), std::to_string(H));
                        EXPECT_EQ(record.tags.at("worker_planes"), "4");
                        const auto cb = static_cast<std::uint8_t>(std::stoi(record.tags.at("execution_codebook")));
                        const auto &resources = capability.leafResources(cb);
                        int matches = 0;
                        for (const auto &node : nodes)
                            if (node.function_identity == reinterpret_cast<std::uintptr_t>(resources.function))
                            {
                                ++matches;
                                EXPECT_EQ(node.grid_x, 1 + (F - 1) / 64); EXPECT_EQ(node.grid_y, rows * TopK); EXPECT_EQ(node.grid_z, 2u);
                                EXPECT_EQ(node.block_x, 256u); EXPECT_EQ(node.block_y, 1u); EXPECT_EQ(node.block_z, 1u);
                                EXPECT_EQ(node.dynamic_shared_memory_bytes, resources.dynamic_shared_memory_bytes);
                            }
                        EXPECT_EQ(matches, 1);
                    }
                }
                else EXPECT_TRUE(captured_records.empty());
                for (int replay = 0; replay < Replays; ++replay)
                {
                    ASSERT_TRUE(backend->hostToDevice(hidden->gpu_data_ptr(), input[replay].data(), live * sizeof(float), 0, stream));
                    ASSERT_TRUE(backend->hostToDevice(output->gpu_data_ptr(), poison.data(), poison.size() * sizeof(float), 0, stream));
                    ASSERT_TRUE(graph->launch());
                    ASSERT_TRUE(backend->deviceToHost(actual.data(), output->gpu_data_ptr(), actual.size() * sizeof(float), 0, stream));
                    ASSERT_TRUE(backend->synchronizeStream(stream, 0));
                    ASSERT_EQ(std::memcmp(actual.data(), expected[replay].data(), actual.size() * sizeof(float)), 0)
                        << "table-owned source=" << format.label << " replay=" << replay;
                    EXPECT_EQ(kernel->ownerGateUpCaptureBindingSignature(gate_table, rows), expected_intent);
                    const auto after = ownerSubmissionRecords(); ASSERT_EQ(after.size(), captured_records.size());
                    for (std::size_t i = 0; i < after.size(); ++i)
                    { EXPECT_EQ(after[i].tags, captured_records[i].tags); EXPECT_EQ(after[i].count, captured_records[i].count); EXPECT_EQ(after[i].value, captured_records[i].value); }
                    EXPECT_FALSE(lastMoERouteOwnedGateUpLaunchSet());
                }
                consumer->unbindWorkspace();
                continue;
            }
            if (rows != 1)
            {
                // The lexical diagnostic capability intentionally remains M1.
                // Multirow coverage above observes only the original automatic policy.
                consumer->unbindWorkspace();
                continue;
            }
            std::optional<MoERouteOwnedGateUpLaunchSet> execution_reference;
            const std::array executions{MoERouteOwnedGateUpPartitionExecution::Serial, requested_execution};
            const std::size_t execution_count = requested_execution == MoERouteOwnedGateUpPartitionExecution::Serial ? 1 : 2;
            for (std::size_t execution_index = 0; execution_index < execution_count; ++execution_index)
            {
                const auto partition_execution = executions[execution_index];
                std::optional<MoERouteOwnedGateUpLaunchSet> geometry_reference;
                for (const auto geometry : kMoERouteOwnedGateUpGeometries)
                {
                    if (requested_execution == MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes &&
                        geometry != MoERouteOwnedGateUpGeometry::TN64) continue;
                    const auto admission = admitMoERouteOwnedGateUpGeometry(geometry, envelope, 0, partition_execution,
                        {.output_columns = F, .reduction_columns = H});
                    auto graph = context.createGraphCapture(stream);
                    ASSERT_NE(graph, nullptr);
                    // Each TN owns a new capture; receipts cannot authenticate a
                    // cached graph created under another geometry capability.
                    ASSERT_FALSE(graph->hasExecutable());
                    ASSERT_EQ(graph->nodeCount(), 0u);
                    std::vector<float> actual(live + Guard);
                    Join graph_join{backend, stream};
                    MoERouteOwnedGateUpLaunchSet captured;
                    {
                        ScopedMoERouteOwnedGateUpGeometry scope(admission);
                        ASSERT_TRUE(complete());
                        ASSERT_TRUE(backend->synchronizeStream(stream, 0));
                        ScopedBackendGraphCapture recording(context, *graph, "route-owned actual M1 geometry");
                        ASSERT_TRUE(recording.begin());
                        ASSERT_TRUE(complete());
                        ASSERT_TRUE(lastMoERouteOwnedGateUpLaunchSet());
                        captured = *lastMoERouteOwnedGateUpLaunchSet();
                        recording.finish();
                    }
                    ASSERT_FALSE(currentMoERouteOwnedGateUpGeometry());
                    ASSERT_TRUE(graph->instantiate());
                    std::vector<GPUGraphKernelNodeInfo> nodes;
                    std::string error;
                    ASSERT_TRUE(graph->inspectKernelNodes(nodes, &error)) << error;
                    ASSERT_EQ(captured.execution_envelope.execution_codebook_mask, envelope.execution_codebook_mask);
                    ASSERT_EQ(captured.execution_envelope.policy_codebook_mask, envelope.policy_codebook_mask);
                    ASSERT_EQ(captured.leaf_count, admission.leafCount());
                    if (requested_execution == MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes)
                    {
                        if (execution_reference)
                        {
                            ASSERT_EQ(captured.leaf_count, execution_reference->leaf_count);
                            for (std::size_t leaf = 0; leaf < captured.leaf_count; ++leaf)
                            {
                                EXPECT_EQ(captured.leaves[leaf].execution_codebook, execution_reference->leaves[leaf].execution_codebook);
                                EXPECT_EQ(captured.leaves[leaf].arithmetic_partitions, execution_reference->leaves[leaf].arithmetic_partitions);
                                EXPECT_NE(captured.leaves[leaf].resources.function, execution_reference->leaves[leaf].resources.function);
                            }
                        }
                        else execution_reference = captured;
                    }
                    if (geometry_reference)
                    {
                        ASSERT_EQ(captured.leaf_count, geometry_reference->leaf_count);
                        for (std::size_t index = 0; index < captured.leaf_count; ++index)
                        {
                            EXPECT_EQ(captured.leaves[index].execution_codebook,
                                geometry_reference->leaves[index].execution_codebook);
                            EXPECT_EQ(captured.leaves[index].arithmetic_partitions,
                                geometry_reference->leaves[index].arithmetic_partitions);
                        }
                    }
                    else geometry_reference = captured;
                    EXPECT_EQ(lastMoERouteOwnedGateUpLaunch().has_value(), captured.leaf_count == 1);
                    std::uint32_t seen = 0;
                    for (std::size_t leaf_index = 0; leaf_index < captured.leaf_count; ++leaf_index)
                    {
                        const auto &leaf = captured.leaves[leaf_index];
                        const auto bit = nativeVnniCodebookMaskBit(leaf.execution_codebook);
                        ASSERT_EQ(seen & bit, 0u);
                        seen |= bit;
                        EXPECT_EQ(leaf.resources.function, admission.leafResources(leaf.execution_codebook).function);
                        int matching = 0;
                        for (const auto &node : nodes)
                            if (node.function_identity == reinterpret_cast<std::uintptr_t>(leaf.resources.function))
                            {
                                ++matching;
                                EXPECT_EQ(node.grid_x, leaf.grid_x); EXPECT_EQ(node.grid_y, leaf.grid_y);
                                EXPECT_EQ(node.grid_z, leaf.grid_z); EXPECT_EQ(node.block_x, leaf.block_x);
                                EXPECT_EQ(node.block_y, leaf.block_y); EXPECT_EQ(node.block_z, leaf.block_z);
                                EXPECT_EQ(node.dynamic_shared_memory_bytes, leaf.resources.dynamic_shared_memory_bytes);
                            }
                        ASSERT_EQ(matching, 1); // Even an empty reachable family remains captured.
                        EXPECT_EQ(leaf.n, F); EXPECT_EQ(leaf.k, H); EXPECT_EQ(leaf.route_capacity, rows * TopK);
                        EXPECT_EQ(leaf.geometry, geometry);
                        EXPECT_EQ(leaf.partition_execution, partition_execution);
                        EXPECT_EQ(leaf.partition_worker_planes,
                            partition_execution == MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes ? 4 : 1);
                        EXPECT_EQ(leaf.arithmetic_partitions, admission.arithmeticPartitions());
                        EXPECT_EQ(leaf.block_x, admission.leafResources(leaf.execution_codebook).block_threads);
                        if (partition_execution == MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes)
                            EXPECT_EQ(leaf.resources.dynamic_shared_memory_bytes,
                                moeRouteOwnedGateUpK1SharedLayout(leaf.execution_codebook, partition_execution, admission.maximumKPartitions()).total_bytes);
                        EXPECT_EQ(leaf.grid_x, 1 + (F - 1) / moeRouteOwnedGateUpColumns(geometry));
                        EXPECT_EQ(leaf.grid_z, 2);
                        EXPECT_EQ(leaf.resources.local_memory_bytes_per_thread, 0u);
                        EXPECT_EQ(leaf.arithmetic_partitions, captured.leaves[0].arithmetic_partitions);
                    }
                    EXPECT_EQ(seen, admission.launchCodebookMask());
                    if (iq4_mixed)
                    {
                        EXPECT_TRUE(envelope.execution_codebook_mask & nativeVnniCodebookMaskBit(4));
                        EXPECT_TRUE(envelope.execution_codebook_mask & nativeVnniCodebookMaskBit(kNativeVnniIQ4XSSourceScaleCodebook));
                        EXPECT_EQ(seen & nativeVnniCodebookMaskBit(4), 0u);
                        EXPECT_NE(seen & nativeVnniCodebookMaskBit(kNativeVnniIQ4XSSourceScaleCodebook), 0u);
                    }
                    if (captured.leaf_count != 1)
                    {
                        EXPECT_THROW((void)admission.executionCodebook(), std::logic_error);
                        EXPECT_THROW((void)admission.resources(), std::logic_error);
                    }
                    for (int replay = 0; replay < Replays; ++replay)
                    {
                        ASSERT_TRUE(backend->hostToDevice(hidden->gpu_data_ptr(), input[replay].data(), live * sizeof(float), 0, stream));
                        ASSERT_TRUE(backend->hostToDevice(output->gpu_data_ptr(), poison.data(), poison.size() * sizeof(float), 0, stream));
                        ASSERT_TRUE(graph->launch());
                        ASSERT_TRUE(backend->deviceToHost(actual.data(), output->gpu_data_ptr(), actual.size() * sizeof(float), 0, stream));
                        ASSERT_TRUE(backend->synchronizeStream(stream, 0));
                        ASSERT_EQ(std::memcmp(actual.data(), expected[replay].data(), actual.size() * sizeof(float)), 0)
                            << "source=" << format.label << " q5_down=" << q5_down << " replay=" << replay;
                        ASSERT_TRUE(lastMoERouteOwnedGateUpLaunchSet());
                        const auto after = *lastMoERouteOwnedGateUpLaunchSet();
                        ASSERT_EQ(after.leaf_count, captured.leaf_count);
                        EXPECT_EQ(after.execution_envelope.execution_codebook_mask, captured.execution_envelope.execution_codebook_mask);
                        EXPECT_EQ(after.execution_envelope.policy_codebook_mask, captured.execution_envelope.policy_codebook_mask);
                        for (std::size_t leaf = 0; leaf < captured.leaf_count; ++leaf)
                        {
                            EXPECT_EQ(after.leaves[leaf].resources.function, captured.leaves[leaf].resources.function);
                            EXPECT_EQ(after.leaves[leaf].execution_codebook, captured.leaves[leaf].execution_codebook);
                            EXPECT_EQ(after.leaves[leaf].geometry, geometry);
                            EXPECT_EQ(after.leaves[leaf].partition_execution, partition_execution);
                            EXPECT_EQ(after.leaves[leaf].partition_worker_planes, captured.leaves[leaf].partition_worker_planes);
                            EXPECT_EQ(after.leaves[leaf].arithmetic_partitions, captured.leaves[leaf].arithmetic_partitions);
                        }
                    }
                    ASSERT_TRUE(complete()); // Ordinary automatic path remains independent.
                    EXPECT_FALSE(lastMoERouteOwnedGateUpLaunchSet());
                    EXPECT_FALSE(lastMoERouteOwnedGateUpLaunch());
                    ASSERT_TRUE(backend->synchronizeStream(stream, 0));
                }
            }
            consumer->unbindWorkspace();
        }
    });
}

} // namespace

/** @brief Preserve the original all-format serial partition witness. */
TEST(ROCmMoERouteOwnedGeometryNative, CapturedPublicM1MixedSourcesMatchSerialBytes)
{
    verifyCapturedPublicM1Sources(MoERouteOwnedGateUpPartitionExecution::Serial, 256, 320);
}

/** @brief Original source partitions cover K32, nonempty multipartitions, N tails and20 captured histories. */
TEST(ROCmMoERouteOwnedGeometryNative, CapturedParallelSourcePartitionsMatchSerialBytes)
{
    verifyCapturedPublicM1Sources(MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes, 32, 32);
    if (::testing::Test::HasFatalFailure()) return;
    verifyCapturedPublicM1Sources(MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes, 64, 64);
    if (::testing::Test::HasFatalFailure()) return;
    verifyCapturedPublicM1Sources(MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes, 96, 96);
    if (::testing::Test::HasFatalFailure()) return;
    verifyCapturedPublicM1Sources(MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes, 256, 32);
    if (::testing::Test::HasFatalFailure()) return;
    verifyCapturedPublicM1Sources(MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes, 640, 32);
    if (::testing::Test::HasFatalFailure()) return;
    verifyCapturedPublicM1Sources(MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes, 768, 96);
    if (::testing::Test::HasFatalFailure()) return;
    verifyCapturedPublicM1Sources(MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes, 2560, 96);
}

/** @brief Actual IQ2_S automatic small/large route policies remain byte-exact against both explicit modes. */
TEST(ROCmMoERouteOwnedGeometryNative, CapturedIQ2AutomaticCapacityBranchesMatchParallelBytes)
{
    verifyCapturedPublicM1Sources(MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes, 256, 96, false, 1, 1, true);
    if (::testing::Test::HasFatalFailure()) return;
    verifyCapturedPublicM1Sources(MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes, 256, 96, false, 2, 9, true);
}

/** @brief Real CPU-prepared codebook23 values match original compact serial bytes under both captured execution modes. */
TEST(ROCmMoERouteOwnedGeometryNative, CapturedParallelSourcePartitionsPromotedQ5KMatchSerialBytes)
{
    verifyCapturedPublicM1Sources(MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes, 2560, 96, true);
}

/** @brief Actual runtime descriptor tables lower graph-owned caps without diagnostic scope, retaining serial bytes over twenty histories. */
TEST(ROCmMoERouteOwnedGeometryNative, CapturedModelOwnerBindingMatchesSerialBytes)
{
    verifyCapturedPublicM1Sources(MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes, 256, 96,
        false, 1, 1, false, true);
}
/** @brief Real promoted23 payloads retain exact source-policy identity in table-owned model lowering. */
TEST(ROCmMoERouteOwnedGeometryNative, CapturedModelOwnerBindingPromotedQ5KMatchesSerialBytes)
{
    verifyCapturedPublicM1Sources(MoERouteOwnedGateUpPartitionExecution::FourWorkerPlanes, 256, 96,
        true, 1, 1, false, true);
}

#endif
