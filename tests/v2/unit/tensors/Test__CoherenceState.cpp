#include <gtest/gtest.h>

#include "tensors/CoherenceState.h"

using namespace llaminar2;

// ============================================================================
// TensorCoherenceState transition table tests
// ============================================================================

class Test__CoherenceState : public ::testing::Test
{
};

// ---------------------------------------------------------------------------
// HOST_ONLY transitions
// ---------------------------------------------------------------------------

TEST(Test__CoherenceState, HostOnly_Upload_BecomesSynced)
{
    auto result = coherenceTransition(TensorCoherenceState::HOST_ONLY, CoherenceOp::UPLOAD);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.new_state, TensorCoherenceState::SYNCED);
}

TEST(Test__CoherenceState, HostOnly_Download_StaysHostOnly)
{
    auto result = coherenceTransition(TensorCoherenceState::HOST_ONLY, CoherenceOp::DOWNLOAD);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.new_state, TensorCoherenceState::HOST_ONLY);
}

TEST(Test__CoherenceState, HostOnly_MarkDeviceDirty_Invalid)
{
    auto result = coherenceTransition(TensorCoherenceState::HOST_ONLY, CoherenceOp::MARK_DEVICE_DIRTY);
    EXPECT_FALSE(result.valid);
}

TEST(Test__CoherenceState, HostOnly_MutableHostAccess_StaysHostOnly)
{
    auto result = coherenceTransition(TensorCoherenceState::HOST_ONLY, CoherenceOp::MUTABLE_HOST_ACCESS);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.new_state, TensorCoherenceState::HOST_ONLY);
}

TEST(Test__CoherenceState, HostOnly_ReleaseDevice_StaysHostOnly)
{
    auto result = coherenceTransition(TensorCoherenceState::HOST_ONLY, CoherenceOp::RELEASE_DEVICE);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.new_state, TensorCoherenceState::HOST_ONLY);
}

// ---------------------------------------------------------------------------
// HOST_AUTHORITATIVE transitions
// ---------------------------------------------------------------------------

TEST(Test__CoherenceState, HostAuthoritative_Upload_BecomesSynced)
{
    auto result = coherenceTransition(TensorCoherenceState::HOST_AUTHORITATIVE, CoherenceOp::UPLOAD);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.new_state, TensorCoherenceState::SYNCED);
}

TEST(Test__CoherenceState, HostAuthoritative_Download_Stays)
{
    auto result = coherenceTransition(TensorCoherenceState::HOST_AUTHORITATIVE, CoherenceOp::DOWNLOAD);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.new_state, TensorCoherenceState::HOST_AUTHORITATIVE);
}

TEST(Test__CoherenceState, HostAuthoritative_MarkDeviceDirty_BecomesDeviceAuth)
{
    auto result = coherenceTransition(TensorCoherenceState::HOST_AUTHORITATIVE, CoherenceOp::MARK_DEVICE_DIRTY);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.new_state, TensorCoherenceState::DEVICE_AUTHORITATIVE);
}

TEST(Test__CoherenceState, HostAuthoritative_MutableHostAccess_Stays)
{
    auto result = coherenceTransition(TensorCoherenceState::HOST_AUTHORITATIVE, CoherenceOp::MUTABLE_HOST_ACCESS);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.new_state, TensorCoherenceState::HOST_AUTHORITATIVE);
}

TEST(Test__CoherenceState, HostAuthoritative_ReleaseDevice_BecomesHostOnly)
{
    auto result = coherenceTransition(TensorCoherenceState::HOST_AUTHORITATIVE, CoherenceOp::RELEASE_DEVICE);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.new_state, TensorCoherenceState::HOST_ONLY);
}

// ---------------------------------------------------------------------------
// DEVICE_AUTHORITATIVE transitions
// ---------------------------------------------------------------------------

TEST(Test__CoherenceState, DeviceAuth_Upload_BecomesSynced)
{
    // Device→host sync implied, then upload is effectively a re-sync
    auto result = coherenceTransition(TensorCoherenceState::DEVICE_AUTHORITATIVE, CoherenceOp::UPLOAD);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.new_state, TensorCoherenceState::SYNCED);
}

TEST(Test__CoherenceState, DeviceAuth_Download_BecomesSynced)
{
    auto result = coherenceTransition(TensorCoherenceState::DEVICE_AUTHORITATIVE, CoherenceOp::DOWNLOAD);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.new_state, TensorCoherenceState::SYNCED);
}

TEST(Test__CoherenceState, DeviceAuth_MarkDeviceDirty_Stays)
{
    auto result = coherenceTransition(TensorCoherenceState::DEVICE_AUTHORITATIVE, CoherenceOp::MARK_DEVICE_DIRTY);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.new_state, TensorCoherenceState::DEVICE_AUTHORITATIVE);
}

TEST(Test__CoherenceState, DeviceAuth_MutableHostAccess_BecomesSynced)
{
    // Implies an implicit device→host download
    auto result = coherenceTransition(TensorCoherenceState::DEVICE_AUTHORITATIVE, CoherenceOp::MUTABLE_HOST_ACCESS);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.new_state, TensorCoherenceState::SYNCED);
}

TEST(Test__CoherenceState, DeviceAuth_ReleaseDevice_BecomesHostOnly)
{
    // Must download device data first!
    auto result = coherenceTransition(TensorCoherenceState::DEVICE_AUTHORITATIVE, CoherenceOp::RELEASE_DEVICE);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.new_state, TensorCoherenceState::HOST_ONLY);
}

// ---------------------------------------------------------------------------
// SYNCED transitions
// ---------------------------------------------------------------------------

TEST(Test__CoherenceState, Synced_Upload_StaysSynced)
{
    auto result = coherenceTransition(TensorCoherenceState::SYNCED, CoherenceOp::UPLOAD);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.new_state, TensorCoherenceState::SYNCED);
}

TEST(Test__CoherenceState, Synced_Download_StaysSynced)
{
    auto result = coherenceTransition(TensorCoherenceState::SYNCED, CoherenceOp::DOWNLOAD);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.new_state, TensorCoherenceState::SYNCED);
}

TEST(Test__CoherenceState, Synced_MarkDeviceDirty_BecomesDeviceAuth)
{
    auto result = coherenceTransition(TensorCoherenceState::SYNCED, CoherenceOp::MARK_DEVICE_DIRTY);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.new_state, TensorCoherenceState::DEVICE_AUTHORITATIVE);
}

TEST(Test__CoherenceState, Synced_MutableHostAccess_BecomesHostAuth)
{
    auto result = coherenceTransition(TensorCoherenceState::SYNCED, CoherenceOp::MUTABLE_HOST_ACCESS);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.new_state, TensorCoherenceState::HOST_AUTHORITATIVE);
}

TEST(Test__CoherenceState, Synced_ReleaseDevice_BecomesHostOnly)
{
    auto result = coherenceTransition(TensorCoherenceState::SYNCED, CoherenceOp::RELEASE_DEVICE);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.new_state, TensorCoherenceState::HOST_ONLY);
}

// ---------------------------------------------------------------------------
// MAPPED — all transitions stay MAPPED
// ---------------------------------------------------------------------------

TEST(Test__CoherenceState, Mapped_AllOps_StaysMapped)
{
    for (int op_idx = 0; op_idx < NUM_OPS; ++op_idx)
    {
        auto op = static_cast<CoherenceOp>(op_idx);
        auto result = coherenceTransition(TensorCoherenceState::MAPPED, op);
        EXPECT_TRUE(result.valid) << "Op " << to_string(op) << " should be valid for MAPPED";
        EXPECT_EQ(result.new_state, TensorCoherenceState::MAPPED)
            << "Op " << to_string(op) << " should keep MAPPED state";
    }
}

// ---------------------------------------------------------------------------
// INVALID and out-of-range
// ---------------------------------------------------------------------------

TEST(Test__CoherenceState, InvalidState_ReturnsInvalid)
{
    auto result = coherenceTransition(TensorCoherenceState::INVALID, CoherenceOp::UPLOAD);
    EXPECT_FALSE(result.valid);
}

TEST(Test__CoherenceState, OutOfRangeState_ReturnsInvalid)
{
    auto result = coherenceTransition(static_cast<TensorCoherenceState>(99), CoherenceOp::UPLOAD);
    EXPECT_FALSE(result.valid);
}

TEST(Test__CoherenceState, OutOfRangeOp_ReturnsInvalid)
{
    auto result = coherenceTransition(TensorCoherenceState::HOST_ONLY, static_cast<CoherenceOp>(99));
    EXPECT_FALSE(result.valid);
}

// ---------------------------------------------------------------------------
// Helper function tests
// ---------------------------------------------------------------------------

TEST(Test__CoherenceState, NeedsDeviceToHostSync)
{
    EXPECT_TRUE(needsDeviceToHostSync(TensorCoherenceState::DEVICE_AUTHORITATIVE));
    EXPECT_FALSE(needsDeviceToHostSync(TensorCoherenceState::HOST_ONLY));
    EXPECT_FALSE(needsDeviceToHostSync(TensorCoherenceState::HOST_AUTHORITATIVE));
    EXPECT_FALSE(needsDeviceToHostSync(TensorCoherenceState::SYNCED));
    EXPECT_FALSE(needsDeviceToHostSync(TensorCoherenceState::MAPPED));
}

TEST(Test__CoherenceState, NeedsHostToDeviceUpload)
{
    EXPECT_TRUE(needsHostToDeviceUpload(TensorCoherenceState::HOST_ONLY));
    EXPECT_TRUE(needsHostToDeviceUpload(TensorCoherenceState::HOST_AUTHORITATIVE));
    EXPECT_FALSE(needsHostToDeviceUpload(TensorCoherenceState::DEVICE_AUTHORITATIVE));
    EXPECT_FALSE(needsHostToDeviceUpload(TensorCoherenceState::SYNCED));
    EXPECT_FALSE(needsHostToDeviceUpload(TensorCoherenceState::MAPPED));
}

TEST(Test__CoherenceState, IsHostValid)
{
    EXPECT_TRUE(isHostValid(TensorCoherenceState::HOST_ONLY));
    EXPECT_TRUE(isHostValid(TensorCoherenceState::HOST_AUTHORITATIVE));
    EXPECT_TRUE(isHostValid(TensorCoherenceState::SYNCED));
    EXPECT_TRUE(isHostValid(TensorCoherenceState::MAPPED));
    EXPECT_FALSE(isHostValid(TensorCoherenceState::DEVICE_AUTHORITATIVE));
    EXPECT_FALSE(isHostValid(TensorCoherenceState::INVALID));
}

TEST(Test__CoherenceState, IsDeviceValid)
{
    EXPECT_TRUE(isDeviceValid(TensorCoherenceState::DEVICE_AUTHORITATIVE));
    EXPECT_TRUE(isDeviceValid(TensorCoherenceState::SYNCED));
    EXPECT_TRUE(isDeviceValid(TensorCoherenceState::MAPPED));
    EXPECT_FALSE(isDeviceValid(TensorCoherenceState::HOST_ONLY));
    EXPECT_FALSE(isDeviceValid(TensorCoherenceState::HOST_AUTHORITATIVE));
    EXPECT_FALSE(isDeviceValid(TensorCoherenceState::INVALID));
}

TEST(Test__CoherenceState, ApplyTransition_Valid)
{
    EXPECT_EQ(applyTransition(TensorCoherenceState::HOST_ONLY, CoherenceOp::UPLOAD), TensorCoherenceState::SYNCED);
    EXPECT_EQ(applyTransition(TensorCoherenceState::SYNCED, CoherenceOp::MARK_DEVICE_DIRTY),
              TensorCoherenceState::DEVICE_AUTHORITATIVE);
    EXPECT_EQ(applyTransition(TensorCoherenceState::DEVICE_AUTHORITATIVE, CoherenceOp::DOWNLOAD),
              TensorCoherenceState::SYNCED);
}

TEST(Test__CoherenceState, ApplyTransition_Invalid_ReturnsInvalid)
{
    EXPECT_EQ(applyTransition(TensorCoherenceState::HOST_ONLY, CoherenceOp::MARK_DEVICE_DIRTY),
              TensorCoherenceState::INVALID);
}

TEST(Test__CoherenceState, IsValidTransition_Checks)
{
    EXPECT_TRUE(isValidTransition(TensorCoherenceState::HOST_ONLY, CoherenceOp::UPLOAD));
    EXPECT_FALSE(isValidTransition(TensorCoherenceState::HOST_ONLY, CoherenceOp::MARK_DEVICE_DIRTY));
    EXPECT_TRUE(isValidTransition(TensorCoherenceState::MAPPED, CoherenceOp::MARK_DEVICE_DIRTY));
}

// ---------------------------------------------------------------------------
// to_string tests
// ---------------------------------------------------------------------------

TEST(Test__CoherenceState, ToString_States)
{
    EXPECT_EQ(to_string(TensorCoherenceState::HOST_ONLY), "HOST_ONLY");
    EXPECT_EQ(to_string(TensorCoherenceState::HOST_AUTHORITATIVE), "HOST_AUTHORITATIVE");
    EXPECT_EQ(to_string(TensorCoherenceState::DEVICE_AUTHORITATIVE), "DEVICE_AUTHORITATIVE");
    EXPECT_EQ(to_string(TensorCoherenceState::SYNCED), "SYNCED");
    EXPECT_EQ(to_string(TensorCoherenceState::MAPPED), "MAPPED");
    EXPECT_EQ(to_string(TensorCoherenceState::INVALID), "INVALID");
}

TEST(Test__CoherenceState, ToString_Operations)
{
    EXPECT_EQ(to_string(CoherenceOp::UPLOAD), "UPLOAD");
    EXPECT_EQ(to_string(CoherenceOp::DOWNLOAD), "DOWNLOAD");
    EXPECT_EQ(to_string(CoherenceOp::MARK_DEVICE_DIRTY), "MARK_DEVICE_DIRTY");
    EXPECT_EQ(to_string(CoherenceOp::MUTABLE_HOST_ACCESS), "MUTABLE_HOST_ACCESS");
    EXPECT_EQ(to_string(CoherenceOp::RELEASE_DEVICE), "RELEASE_DEVICE");
}

TEST(Test__CoherenceState, ToString_Residency)
{
    EXPECT_EQ(to_string(MemoryResidency::STANDARD), "STANDARD");
    EXPECT_EQ(to_string(MemoryResidency::BAR_BACKED), "BAR_BACKED");
    EXPECT_EQ(to_string(MemoryResidency::MAPPED), "MAPPED");
}

// ---------------------------------------------------------------------------
// Full lifecycle sequences
// ---------------------------------------------------------------------------

TEST(Test__CoherenceState, FullLifecycle_UploadComputeDownload)
{
    TensorCoherenceState state = TensorCoherenceState::HOST_ONLY;

    // Upload to device
    state = applyTransition(state, CoherenceOp::UPLOAD);
    EXPECT_EQ(state, TensorCoherenceState::SYNCED);

    // GPU kernel writes
    state = applyTransition(state, CoherenceOp::MARK_DEVICE_DIRTY);
    EXPECT_EQ(state, TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // Read back to host
    state = applyTransition(state, CoherenceOp::DOWNLOAD);
    EXPECT_EQ(state, TensorCoherenceState::SYNCED);
}

TEST(Test__CoherenceState, FullLifecycle_HostModifyReupload)
{
    TensorCoherenceState state = TensorCoherenceState::HOST_ONLY;

    // Upload
    state = applyTransition(state, CoherenceOp::UPLOAD);
    EXPECT_EQ(state, TensorCoherenceState::SYNCED);

    // Host modifies
    state = applyTransition(state, CoherenceOp::MUTABLE_HOST_ACCESS);
    EXPECT_EQ(state, TensorCoherenceState::HOST_AUTHORITATIVE);

    // Re-upload
    state = applyTransition(state, CoherenceOp::UPLOAD);
    EXPECT_EQ(state, TensorCoherenceState::SYNCED);
}

TEST(Test__CoherenceState, FullLifecycle_DeviceDirtyThenMutableHost)
{
    TensorCoherenceState state = TensorCoherenceState::SYNCED;

    // GPU kernel writes
    state = applyTransition(state, CoherenceOp::MARK_DEVICE_DIRTY);
    EXPECT_EQ(state, TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // Host wants mutable access (implies download)
    state = applyTransition(state, CoherenceOp::MUTABLE_HOST_ACCESS);
    EXPECT_EQ(state, TensorCoherenceState::SYNCED);
}

TEST(Test__CoherenceState, FullLifecycle_ReleaseDevice)
{
    TensorCoherenceState state = TensorCoherenceState::SYNCED;

    // Release GPU buffer
    state = applyTransition(state, CoherenceOp::RELEASE_DEVICE);
    EXPECT_EQ(state, TensorCoherenceState::HOST_ONLY);

    // Can re-upload
    state = applyTransition(state, CoherenceOp::UPLOAD);
    EXPECT_EQ(state, TensorCoherenceState::SYNCED);
}

// ---------------------------------------------------------------------------
// Constexpr verification — compile-time tests
// ---------------------------------------------------------------------------

static_assert(coherenceTransition(TensorCoherenceState::HOST_ONLY, CoherenceOp::UPLOAD).valid);
static_assert(coherenceTransition(TensorCoherenceState::HOST_ONLY, CoherenceOp::UPLOAD).new_state == TensorCoherenceState::SYNCED);
static_assert(!coherenceTransition(TensorCoherenceState::HOST_ONLY, CoherenceOp::MARK_DEVICE_DIRTY).valid);
static_assert(coherenceTransition(TensorCoherenceState::MAPPED, CoherenceOp::UPLOAD).new_state == TensorCoherenceState::MAPPED);
static_assert(isHostValid(TensorCoherenceState::SYNCED));
static_assert(!isHostValid(TensorCoherenceState::DEVICE_AUTHORITATIVE));
static_assert(isDeviceValid(TensorCoherenceState::SYNCED));
static_assert(!isDeviceValid(TensorCoherenceState::HOST_ONLY));
