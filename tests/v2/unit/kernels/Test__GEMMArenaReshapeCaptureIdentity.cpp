/**
 * @file Test__GEMMArenaReshapeCaptureIdentity.cpp
 * @brief Pure capture identity controls with the expanded descriptor and no old-core linkage.
 */
#include "execution/compute_stages/stages/GEMMArenaReshapeStage.h"
#include <gtest/gtest.h>
using namespace llaminar2;
namespace
{
/** @brief Metadata-only declarations: no TensorBase instance, native pointer or device is accessed. */
struct Identity
{
    GEMMStage::Params params{.device_id = DeviceId::cuda(0), .m = 8, .n = 2, .k = 4};
    FP32ArenaReshape::DeclarationIdentity input{reinterpret_cast<const TensorBase *>(0x1000),
        BufferId::MTP_CONCAT, DeviceId::cuda(0), 128, {8, 4}};
    FP32ArenaReshape::DeclarationIdentity output{reinterpret_cast<const TensorBase *>(0x2000),
        BufferId::MTP_PROJECTED, DeviceId::cuda(0), 64, {8, 2}};
    uint64_t base = 0;
    /** @return Exact stage-owned pure declaration key. */
    uint64_t signature() const { return GEMMArenaReshapeStage::captureVariantForDeclaration(params, input, output, base); }
};
}
TEST(GEMMArenaReshapeCaptureIdentity, CapturesRootBufferDeviceCapacityAndShape)
{
    const Identity original;
    EXPECT_NE(original.signature(), 0u);
    for (int field = 0; field < 6; ++field) {
        Identity changed = original;
        if (field == 0) changed.input.root = reinterpret_cast<const TensorBase *>(0x3000);
        if (field == 1) changed.input.buffer = BufferId::Q_PROJ;
        if (field == 2) changed.output.device = DeviceId::rocm(0);
        if (field == 3) changed.output.bytes += 4;
        if (field == 4) changed.input.shape = {4, 8};
        if (field == 5) changed.base = 2;
        EXPECT_NE(changed.signature(), original.signature()) << field;
    }
}
TEST(GEMMArenaReshapeCaptureIdentity, CapturesAllExpandedRowsWithoutReadingCount)
{
    std::int32_t live = 1, other = 1;
    Identity original;
    original.params.verifier_row_range = DeviceRowRange::deviceCounted(2, &live).repeatEachRow(4);
    const auto signature = original.signature();
    live = 0;
    EXPECT_EQ(original.signature(), signature);
    for (int field = 0; field < 5; ++field) {
        Identity changed = original;
        if (field == 0) changed.params.verifier_row_range = DeviceRowRange::deviceCounted(2, &other).repeatEachRow(4);
        if (field == 1) changed.params.verifier_row_range = DeviceRowRange::deviceCounted(4, &live).repeatEachRow(2);
        if (field == 2) changed.params.verifier_row_range = original.params.verifier_row_range->slice(1, 7);
        if (field == 3) changed.params.verifier_row_range = original.params.verifier_row_range->slice(0, 7);
        if (field == 4) changed.params.verifier_row_range = DeviceRowRange::fullyActive(8);
        EXPECT_NE(changed.signature(), signature) << field;
    }
}
TEST(GEMMArenaReshapeCaptureIdentity, CapturesPreparedNativeWeightAndLaunchPolicy)
{
    const Identity original;
    for (int field = 0; field < 12; ++field) {
        Identity changed = original;
        if (field == 0) changed.params.B = reinterpret_cast<const ITensor *>(0x3000);
        if (field == 1) changed.params.alpha = 0.5f;
        if (field == 2) changed.params.beta = 0.5f;
        if (field == 3) changed.params.transpose_B = true;
        if (field == 4) changed.params.bias = reinterpret_cast<const float *>(0x4000);
        if (field == 5) changed.params.do_swiglu = true;
        if (field == 6) changed.params.output_range = {1, 2};
        if (field == 7) changed.params.force_decode_equivalent_verifier_prefill = true;
        if (field == 8) changed.params.prepared_ref = PreparedWeightRef{.binding_id = 9};
        if (field == 9) changed.params.prepared_store = reinterpret_cast<PreparedWeightStore *>(0x5000);
        if (field == 10) changed.params.m = 4;
        if (field == 11) changed.params.device_id = DeviceId::cuda(1);
        EXPECT_NE(changed.signature(), original.signature()) << field;
    }
}
