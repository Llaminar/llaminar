/**
 * @file PreparedNativeVNNIInspectionFixture.h
 * @brief Backend-neutral prepared-metadata lifecycle proof shared by separate native TUs.
 *
 * CUDA/HIP runtime headers intentionally never meet in this fixture. Each
 * backend supplies its actual checked all-node graph observer. Inspection uses
 * the original production weight loader and retains joined graph/source/storage
 * owners; no source payload readback or materialization is introduced.
 */
#pragma once
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "GpuPreparedGemmHarness.h"
#include "TestTensorFactory.h"
#include <gtest/gtest.h>
#include <tuple>

namespace llaminar2::test::prepared_native_inspection
{
/** @return Every original semantic descriptor field, excluding unspecified struct padding. */
inline auto descriptorIdentity(const DeviceNativeVNNIMatrixDesc &d)
{
    return std::tuple{d.payload, d.scales, d.mins, d.emins, d.n, d.k, d.blocks_per_row,
        d.codebook_id, d.allocation_payload_bytes_per_block, d.allocation_has_mins,
        d.allocation_has_emins, d.source_codebook_id, d.source_is_superblock, d.source_identity_present};
}
/** @return Original source identity without a GPU/export/materialization operation. */
inline auto sourceIdentity(const ITensorGemm &kernel)
{
    NativeVnniSourceIdentity original{};
    const bool present = kernel.exportNativeVNNISourceIdentity(original);
    return std::tuple{present, original.present, original.codebook_id, original.is_superblock};
}
/**
 * @brief Verify passive absence and resident metadata under the actual native recording API.
 * @param device Exact backend endpoint, selected by the owning context worker.
 * @tparam Engine Concrete original GPU adapter; its source constructor remains unprepared.
 * @tparam NativeNodeCount Backend-specific checked all-node inspection; no kernel-only projection is accepted.
 * @param native_node_count Exact native graph API observer, compiled in the one owning backend TU.
 *
 * The legacy constructor is used only to prove inspection cannot cause its
 * deferred materialization. Positive metadata comes from the production VRAM
 * loader, not that legacy constructor. A joined stream protects all owners on
 * assertion/exception unwind, including graph inspection failure.
 */
template<class Engine, class NativeNodeCount>
void provePassiveInspection(DeviceId device, NativeNodeCount native_node_count)
{
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&] {
        auto *backend = getBackendFor(device); auto *stream = context.defaultStream();
        ASSERT_NE(backend, nullptr); ASSERT_NE(stream, nullptr);
        auto source = TestTensorFactory::createQ4_KRandom({32,256}, 891004);
        Engine unprepared(source.get(), device.ordinal);
        auto prepared = makeGpuPreparedGemm(source.get(), device, "test.passive_native_metadata");
        ASSERT_NE(prepared.kernel, nullptr);
        auto graph = context.createGraphCapture(stream); ASSERT_NE(graph, nullptr);
        /** @brief Retire the exact native stream before any prepared, graph or source owner. */
        struct Join { IBackend *backend; void *stream; ~Join() { (void)backend->synchronizeStream(stream, 0); } } join{backend,stream};
        ASSERT_FALSE(unprepared.weights_converted());
        const auto unprepared_source_before = sourceIdentity(unprepared);
        const auto prepared_source_before = sourceIdentity(*prepared.kernel);
        const bool converted_before = prepared.kernel->weights_converted();
        DeviceNativeVNNIMatrixDesc resident{}, exported{}, absent{};
        ASSERT_TRUE(prepared.kernel->inspectPreparedNativeVNNIMatrixDesc(resident));
        ASSERT_TRUE(prepared.kernel->exportNativeVNNIMatrixDesc(exported));
        ASSERT_EQ(descriptorIdentity(resident), descriptorIdentity(exported));
        absent.n = 999;
        ASSERT_FALSE(unprepared.inspectPreparedNativeVNNIMatrixDesc(absent));
        EXPECT_EQ(descriptorIdentity(absent), descriptorIdentity(DeviceNativeVNNIMatrixDesc{}));
        ASSERT_TRUE(backend->synchronizeStream(stream, 0));
        {
            ScopedBackendGraphCapture recording(context, *graph, "passive prepared metadata has no native work");
            ASSERT_TRUE(recording.begin());
            for (int observation = 0; observation < 3; ++observation)
            {
                DeviceNativeVNNIMatrixDesc current{}, missing{}; missing.n=999;
                EXPECT_TRUE(prepared.kernel->inspectPreparedNativeVNNIMatrixDesc(current));
                EXPECT_EQ(descriptorIdentity(current), descriptorIdentity(resident));
                EXPECT_FALSE(unprepared.inspectPreparedNativeVNNIMatrixDesc(missing));
                EXPECT_EQ(descriptorIdentity(missing), descriptorIdentity(DeviceNativeVNNIMatrixDesc{}));
                EXPECT_FALSE(unprepared.weights_converted());
                EXPECT_EQ(prepared.kernel->weights_converted(), converted_before);
                EXPECT_EQ(sourceIdentity(unprepared), unprepared_source_before);
                EXPECT_EQ(sourceIdentity(*prepared.kernel), prepared_source_before);
            }
            recording.finish();
        }
        EXPECT_EQ(native_node_count(*graph), 0u);
        DeviceNativeVNNIMatrixDesc final{};
        ASSERT_TRUE(prepared.kernel->inspectPreparedNativeVNNIMatrixDesc(final));
        EXPECT_EQ(descriptorIdentity(final), descriptorIdentity(resident));
        EXPECT_FALSE(unprepared.weights_converted());
    });
}
} // namespace llaminar2::test::prepared_native_inspection
