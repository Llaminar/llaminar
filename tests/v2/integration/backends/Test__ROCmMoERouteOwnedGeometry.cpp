/**
 * @file Test__ROCmMoERouteOwnedGeometry.cpp
 * @brief Native resource admission and empty-capture rejection for route-owned tuning.
 *
 * Cold inspection authenticates every canonical physical decoder at each exact
 * K_LANES1 width, plus the unchanged fixed64 down controls. Public invalid calls
 * are observed in a real native capture and must leave zero nodes of any kind.
 * The first two cases guard admission and scope retirement. The final case
 * executes a small captured all-format leaf transaction against an independent
 * serial-M1 byte oracle. It does not certify QFN owner/compact geometry or
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
#include <gtest/gtest.h>
#include <set>
#include <map>
#include <tuple>
#include <stdexcept>
#include <thread>

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
 * @brief Replay each real producer geometry against the independent public serial-M1 operation.
 *
 * Every source alias retains a non-TN256-aligned F320 tail under static and
 * genuine runtime-placement table origins. Runtime envelopes keep all original
 * empty reachable decoder dispatches. The test proves every actual symbol/grid,
 * full output/guard bytes and stable complete replay receipts
 * after scope retirement. It deliberately is not QFN owner/compact certification;
 * that larger transaction belongs to the separately scoped canonical trainer.
 */
TEST(ROCmMoERouteOwnedGeometryNative, CapturedPublicM1MixedSourcesMatchSerialBytes)
{
    using namespace llaminar2::test;
    constexpr int H = 256, F = 320, E = 1, TopK = 1, Guard = 16, Replays = 20;
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
            SCOPED_TRACE(std::string(format.label) + (q5_down ? "/Q5_1" : "/Q8_0") +
                (runtime_mutable ? "/RuntimePlacementTable" : "/StaticDescriptorTable") +
                (iq4_mixed ? "/IQ4_NL-up" : ""));
            // All original source aliases plus the dominant mixed QFN tuple.
            // Full source K256 stays superblock aligned; down K320 uses Q8/Q5_1.
            auto make_gate = [&](unsigned seed) -> std::shared_ptr<TensorBase>
            { return format.create({F, H}, seed); };
            std::vector<std::shared_ptr<TensorBase>> sources{
                make_gate(87200), iq4_mixed
                    ? std::shared_ptr<TensorBase>(quantizedMoEVerifierFormat("IQ4_NL").create({F, H}, 87300))
                    : make_gate(87300),
                q5_down
                    ? std::shared_ptr<TensorBase>(TestTensorFactory::createQ5_1Random({H, F}, 87400))
                    : std::shared_ptr<TensorBase>(TestTensorFactory::createQ8_0Random({H, F}, 87400))};
            std::vector<GpuPreparedGemm> prepared;
            for (std::size_t i = 0; i < sources.size(); ++i)
                prepared.push_back(makeGpuPreparedGemm(sources[i].get(), device,
                    "test.route_owned.geometry." + std::string(format.label) + "." + std::to_string(q5_down) +
                    ".mutable" + std::to_string(runtime_mutable) + ".iq4mixed" + std::to_string(iq4_mixed) + "." + std::to_string(i)));
            DeviceNativeVNNIMatrixDesc gate{}, up{}, down{};
            ASSERT_TRUE(prepared[0].kernel->exportNativeVNNIMatrixDesc(gate));
            ASSERT_TRUE(prepared[1].kernel->exportNativeVNNIMatrixDesc(up));
            ASSERT_TRUE(prepared[2].kernel->exportNativeVNNIMatrixDesc(down));
            ASSERT_TRUE(areNativeVnniGateUpCodebooksCompatible(gate.codebook_id, up.codebook_id));
            const auto requirements = MoEWorkspaceBuffers::rocmMoE(1, H, F, E, TopK);
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
            const int gate_table = kernel->uploadGroupedExpertGateUpDescriptorTables(&gate, &up, E, H, F, origin);
            const int down_table = kernel->uploadGroupedExpertDownDescriptorTable(&down, E, H, F, origin);
            ASSERT_GE(gate_table, 0);
            ASSERT_GE(down_table, 0);
            // Only the independent public serial oracle uses immutable tables.
            // The actual capture below retains its genuine runtime envelope.
            const int oracle_gate_table = runtime_mutable
                ? kernel->uploadGroupedExpertGateUpDescriptorTables(&gate, &up, E, H, F,
                    MoEDecodeDescriptorSource::StaticDescriptorTable) : gate_table;
            const int oracle_down_table = runtime_mutable
                ? kernel->uploadGroupedExpertDownDescriptorTable(&down, E, H, F,
                    MoEDecodeDescriptorSource::StaticDescriptorTable) : down_table;
            ASSERT_GE(oracle_gate_table, 0);
            ASSERT_GE(oracle_down_table, 0);
            auto hidden = TestTensorFactory::createFP32({1, H});
            auto ids = TestTensorFactory::createFP32({1, TopK});
            auto weights = TestTensorFactory::createFP32({1, TopK});
            auto serial = TestTensorFactory::createFP32({H + Guard});
            auto output = TestTensorFactory::createFP32({H + Guard});
            const std::vector<float> poison(H + Guard, -1234.5f);
            const float id = 0.f, weight = 0.625f;
            std::vector<std::vector<float>> input(Replays, std::vector<float>(H));
            std::vector<std::vector<float>> expected(Replays, std::vector<float>(H + Guard));
            // The join is declared after every host/device transfer owner.
            Join storage_join{backend, stream};
            for (auto *tensor : {hidden.get(), ids.get(), weights.get(), serial.get(), output.get()})
                ASSERT_TRUE(tensor->ensureOnDevice(device, stream));
            ASSERT_TRUE(backend->hostToDevice(ids->gpu_data_ptr(), &id, sizeof(id), 0, stream));
            ASSERT_TRUE(backend->hostToDevice(weights->gpu_data_ptr(), &weight, sizeof(weight), 0, stream));
            for (int replay = 0; replay < Replays; ++replay)
            {
                for (int k = 0; k < H; ++k)
                    input[replay][k] = .0071f * (int((k * 19 + replay * 13) % 101) - 50);
                ASSERT_TRUE(backend->hostToDevice(hidden->gpu_data_ptr(), input[replay].data(), H * sizeof(float), 0, stream));
                ASSERT_TRUE(backend->hostToDevice(serial->gpu_data_ptr(), poison.data(), poison.size() * sizeof(float), 0, stream));
                ASSERT_TRUE(kernel->groupedExpertDecodeFromRouting(hidden.get(), ids.get(), weights.get(),
                    oracle_gate_table, oracle_down_table, TopK, serial.get(), H, F));
                ASSERT_TRUE(backend->deviceToHost(expected[replay].data(), serial->gpu_data_ptr(),
                    expected[replay].size() * sizeof(float), 0, stream));
                ASSERT_TRUE(backend->synchronizeStream(stream, 0));
            }
            const auto complete = [&]
            {
                return kernel->prepareExpertGroupsAsync(ids.get(), weights.get(), 1, E, TopK) &&
                    kernel->executeGroupedPrefillPipeline(hidden.get(), output.get(), gate_table,
                        down_table, 1, H, F, E, TopK);
            };
            std::optional<MoERouteOwnedGateUpLaunchSet> geometry_reference;
            for (const auto geometry : kMoERouteOwnedGateUpGeometries)
            {
                const auto admission = admitMoERouteOwnedGateUpGeometry(geometry, envelope, 0);
                auto graph = context.createGraphCapture(stream);
                ASSERT_NE(graph, nullptr);
                // Each TN owns a new capture; receipts cannot authenticate a
                // cached graph created under another geometry capability.
                ASSERT_FALSE(graph->hasExecutable());
                ASSERT_EQ(graph->nodeCount(), 0u);
                std::vector<float> actual(H + Guard);
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
                    EXPECT_EQ(leaf.n, F); EXPECT_EQ(leaf.k, H); EXPECT_EQ(leaf.route_capacity, TopK);
                    EXPECT_EQ(leaf.geometry, geometry);
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
                    ASSERT_TRUE(backend->hostToDevice(hidden->gpu_data_ptr(), input[replay].data(), H * sizeof(float), 0, stream));
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
                        EXPECT_EQ(after.leaves[leaf].arithmetic_partitions, captured.leaves[leaf].arithmetic_partitions);
                    }
                }
                ASSERT_TRUE(complete()); // Ordinary automatic path remains independent.
                EXPECT_FALSE(lastMoERouteOwnedGateUpLaunchSet());
                EXPECT_FALSE(lastMoERouteOwnedGateUpLaunch());
                ASSERT_TRUE(backend->synchronizeStream(stream, 0));
            }
            consumer->unbindWorkspace();
        }
    });
}

#endif
