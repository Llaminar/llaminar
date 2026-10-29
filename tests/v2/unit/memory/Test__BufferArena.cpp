/**
 * @file Test__BufferArena.cpp
 * @brief Unit tests for BufferArena — the central buffer management system
 *
 * Tests cover:
 * - Buffer registration (owned and external)
 * - Allocation lifecycle
 * - Coherence state tracking
 * - Borrow tracking (read/write/aliasing conflicts)
 * - StageBufferContract building
 * - StageBoundBuffers typed access
 * - Exact physical preclaims, rejection, root/view lifetime, and replay
 */

#include "../../utils/CPUActivationArenaFixture.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <set>
#include <string>
#include "planning/PhysicalMemoryAuthority.h"
#include "memory/BufferArena.h"
#include "memory/CPUTensorStorageDeclaration.h"
#include "memory/BufferId.h"
#include "memory/BufferAccess.h"
#include "memory/StageBufferContract.h"
#include "memory/StageBoundBuffers.h"
#include "memory/CoherenceTracker.h"
#include "tensors/TensorClasses.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

namespace
{
/** @brief Admit an explicit device-free fixture CPU arena budget. */
std::shared_ptr<PhysicalMemoryAuthority> arenaAuthority(size_t bytes, DeviceId device = DeviceId::cpu())
{
    PhysicalMemoryPlanBuilder plan;
    plan.add(PhysicalMemoryResource{
        .world_rank = 0,
        .device = device,
        .total_bytes = bytes,
        .admission_available_bytes = bytes},
        PhysicalMemoryOwner::ActivationArena, bytes);
    return std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
}

/** @brief Supply fixture admission explicitly rather than changing production defaults. */
ArenaConfig fixtureArenaConfig()
{
    return test::cpuArenaFixtureConfig();
}
}


// ============================================================================
// Registration Tests
// ============================================================================

TEST(Test__BufferArena, RegisterBufferSucceeds)
{
    BufferArena arena(fixtureArenaConfig());
    EXPECT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu()));
    EXPECT_TRUE(arena.isRegistered(BufferId::HIDDEN_STATE));
    EXPECT_EQ(arena.registeredCount(), 1u);
}

TEST(Test__BufferArena, RegisterBufferRejectsDoubleRegistration)
{
    BufferArena arena(fixtureArenaConfig());
    EXPECT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu()));
    EXPECT_FALSE(arena.registerBuffer(BufferId::HIDDEN_STATE, 8, 896, "FP32", DeviceId::cpu()));
    EXPECT_EQ(arena.registeredCount(), 1u);
}

/**
 * @brief Graph registration must preserve the capacity of every logical axis.
 *
 * Canonical MoE publication uses a rank-three `[rows, routes, width]`
 * descriptor. The arena intentionally exposes matrix tensors to kernels, so it
 * must flatten the two trailing dimensions rather than dropping `width`.
 */
TEST(Test__BufferArena, GraphDescriptorRegistrationFlattensEveryTrailingAxis)
{
    BufferArena arena(fixtureArenaConfig());
    const BufferDescriptor descriptor = BufferDescriptor::scratch(
        "moe_canonical_route_contributions",
        {7, 3, 5},
        BufferTensorType::FP32);

    ASSERT_TRUE(arena.registerBuffer(
        BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS,
        descriptor));
    EXPECT_EQ(arena.getRows(BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS), 7u);
    EXPECT_EQ(arena.getCols(BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS), 15u);
}

TEST(Test__BufferArena, GraphDescriptorRegistrationPreservesRankOneCapacity)
{
    BufferArena arena(fixtureArenaConfig());
    const BufferDescriptor descriptor = BufferDescriptor::scratch(
        "one_dimensional_scratch",
        {4096},
        BufferTensorType::FP32);

    ASSERT_TRUE(arena.registerBuffer(BufferId::GEMM_WORKSPACE, descriptor));
    EXPECT_EQ(arena.getRows(BufferId::GEMM_WORKSPACE), 4096u);
    EXPECT_EQ(arena.getCols(BufferId::GEMM_WORKSPACE), 1u);
}

TEST(Test__BufferArena, GraphDescriptorRegistrationRejectsInvalidShapes)
{
    BufferArena arena(fixtureArenaConfig());

    EXPECT_THROW(
        arena.registerBuffer(
            BufferId::HIDDEN_STATE,
            BufferDescriptor::scratch("empty", {}, BufferTensorType::FP32)),
        std::invalid_argument);
    EXPECT_THROW(
        arena.registerBuffer(
            BufferId::HIDDEN_STATE,
            BufferDescriptor::scratch("zero", {4, 0, 8}, BufferTensorType::FP32)),
        std::invalid_argument);
    EXPECT_THROW(
        arena.registerBuffer(
            BufferId::HIDDEN_STATE,
            BufferDescriptor::scratch(
                "overflow",
                {2, std::numeric_limits<size_t>::max()},
                BufferTensorType::FP32)),
        std::overflow_error);
}

TEST(Test__BufferArena, RegisterMultipleBuffers)
{
    BufferArena arena(fixtureArenaConfig());
    EXPECT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu()));
    EXPECT_TRUE(arena.registerBuffer(BufferId::Q_PROJ, 4, 896, "FP32", DeviceId::cpu()));
    EXPECT_TRUE(arena.registerBuffer(BufferId::K_PROJ, 4, 128, "FP32", DeviceId::cpu()));
    EXPECT_EQ(arena.registeredCount(), 3u);
}

TEST(Test__BufferArena, RegisterExternalBufferSucceeds)
{
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 896}, DeviceId::cpu());
    BufferArena arena(fixtureArenaConfig());
    EXPECT_TRUE(arena.registerExternalBuffer(BufferId::LOGITS, tensor.get()));
    EXPECT_TRUE(arena.isRegistered(BufferId::LOGITS));
}

TEST(Test__BufferArena, BindExternalBufferCanReplaceDynamicScratch)
{
    auto first = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 128}, DeviceId::cpu());
    auto second = std::make_shared<FP32Tensor>(std::vector<size_t>{3, 128}, DeviceId::cpu());

    BufferArena arena(fixtureArenaConfig());
    ASSERT_TRUE(arena.bindExternalBuffer(BufferId::ALL_POSITION_LOGITS, first.get()));
    EXPECT_TRUE(arena.isRegistered(BufferId::ALL_POSITION_LOGITS));
    EXPECT_EQ(arena.getTensor(BufferId::ALL_POSITION_LOGITS), first.get());
    EXPECT_EQ(arena.getRows(BufferId::ALL_POSITION_LOGITS), 2u);

    arena.markWrittenFlagsOnly(BufferId::ALL_POSITION_LOGITS, DeviceId::cpu());
    EXPECT_EQ(arena.getCoherenceState(BufferId::ALL_POSITION_LOGITS).authority,
              CoherenceState::HOST);

    ASSERT_TRUE(arena.bindExternalBuffer(BufferId::ALL_POSITION_LOGITS, second.get()));
    EXPECT_EQ(arena.getTensor(BufferId::ALL_POSITION_LOGITS), second.get());
    EXPECT_EQ(arena.getRows(BufferId::ALL_POSITION_LOGITS), 3u);
    EXPECT_EQ(arena.getCoherenceState(BufferId::ALL_POSITION_LOGITS).authority,
              CoherenceState::UNINITIALIZED);
}

TEST(Test__BufferArena, ArenaBindingsInstallStableBufferDebugNames)
{
    auto external = std::make_shared<FP32Tensor>(
        std::vector<size_t>{2, 128},
        DeviceId::cpu());
    ASSERT_TRUE(external->debugName().empty());

    BufferArena arena(fixtureArenaConfig());
    ASSERT_TRUE(arena.bindExternalBuffer(
        BufferId::ALL_POSITION_LOGITS,
        external.get()));
    ASSERT_TRUE(arena.registerBuffer(
        BufferId::MTP_VERIFIER_INPUT_TOKENS,
        1,
        16,
        "INT32",
        DeviceId::cpu()));
    ASSERT_TRUE(arena.allocate());

    EXPECT_EQ(external->debugName(), "ALL_POSITION_LOGITS");
    auto *owned = dynamic_cast<TensorBase *>(
        arena.getTensor(BufferId::MTP_VERIFIER_INPUT_TOKENS));
    ASSERT_NE(owned, nullptr);
    EXPECT_EQ(owned->debugName(), "MTP_VERIFIER_INPUT_TOKENS");
}

/**
 * @brief Address diagnostics must expose real ranges in numeric order.
 *
 * Out-of-bounds GPU diagnostics depend on identifying the buffers immediately
 * below and above a suspect allocation. The formatter is tested independently
 * of logging so a logger-level change cannot silently remove that evidence.
 */
TEST(Test__BufferArena, AllocationAddressMapSortsRealBufferRanges)
{
    BufferArena arena(fixtureArenaConfig());
    ASSERT_TRUE(arena.registerBuffer(
        BufferId::REQUEST_POSITION_IDS,
        1,
        16,
        "INT32",
        DeviceId::cpu()));
    ASSERT_TRUE(arena.registerBuffer(
        BufferId::MTP_LOGICAL_SEQUENCE_STATE,
        7,
        1,
        "INT32",
        DeviceId::cpu()));
    ASSERT_TRUE(arena.allocate());

    const auto *positions =
        arena.getTensor(BufferId::REQUEST_POSITION_IDS);
    const auto *logical_state =
        arena.getTensor(BufferId::MTP_LOGICAL_SEQUENCE_STATE);
    ASSERT_NE(positions, nullptr);
    ASSERT_NE(logical_state, nullptr);

    const uintptr_t positions_address =
        reinterpret_cast<uintptr_t>(positions->raw_data());
    const uintptr_t logical_state_address =
        reinterpret_cast<uintptr_t>(logical_state->raw_data());
    const std::string map = arena.allocationAddressMap();

    EXPECT_NE(map.find("Allocation address map"), std::string::npos);
    EXPECT_NE(map.find("device=CPU"), std::string::npos);
    EXPECT_NE(map.find("name=REQUEST_POSITION_IDS"), std::string::npos);
    EXPECT_NE(map.find("name=MTP_LOGICAL_SEQUENCE_STATE"),
              std::string::npos);
    EXPECT_NE(map.find("range=[0x"), std::string::npos);
    EXPECT_NE(map.find("ownership=arena"), std::string::npos);
    EXPECT_NE(map.find("space=host"), std::string::npos);
    EXPECT_NE(map.find("shape=7x1"), std::string::npos);

    const size_t positions_offset =
        map.find("name=REQUEST_POSITION_IDS");
    const size_t logical_state_offset =
        map.find("name=MTP_LOGICAL_SEQUENCE_STATE");
    if (positions_address < logical_state_address)
        EXPECT_LT(positions_offset, logical_state_offset);
    else
        EXPECT_LT(logical_state_offset, positions_offset);
}

TEST(Test__BufferArena, BindExternalBufferRejectsArenaOwnedSlot)
{
    BufferArena arena(fixtureArenaConfig());
    ASSERT_TRUE(arena.registerBuffer(BufferId::LOGITS, 1, 128, "FP32", DeviceId::cpu()));

    auto external = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 128}, DeviceId::cpu());
    EXPECT_FALSE(arena.bindExternalBuffer(BufferId::LOGITS, external.get()));
}

TEST(Test__BufferArena, UnregisteredBufferReturnsNullTensor)
{
    BufferArena arena(fixtureArenaConfig());
    EXPECT_FALSE(arena.isRegistered(BufferId::HIDDEN_STATE));
    EXPECT_EQ(arena.getTensor(BufferId::HIDDEN_STATE), nullptr);
}

// ============================================================================
// Allocation Tests
// ============================================================================

TEST(Test__BufferArena, AllocateCreatesOwnedTensors)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::Q_PROJ, 4, 896, "FP32", DeviceId::cpu());

    EXPECT_TRUE(arena.allocate());
    EXPECT_TRUE(arena.isAllocated());

    // Owned tensors should be created
    auto *t1 = arena.getTensor(BufferId::HIDDEN_STATE);
    auto *t2 = arena.getTensor(BufferId::Q_PROJ);
    ASSERT_NE(t1, nullptr);
    ASSERT_NE(t2, nullptr);
    EXPECT_EQ(t1->rows(), 4u);
    EXPECT_EQ(t1->cols(), 896u);
}

TEST(Test__BufferArena, AllocateRejectsDoubleAllocation)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu());
    EXPECT_TRUE(arena.allocate());
    EXPECT_FALSE(arena.allocate()); // Second call should fail
}

TEST(Test__BufferArena, ExternalBufferNotAllocatedByArena)
{
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 896}, DeviceId::cpu());
    BufferArena arena(fixtureArenaConfig());
    arena.registerExternalBuffer(BufferId::LOGITS, tensor.get());
    EXPECT_TRUE(arena.allocate());

    // Should return the same external tensor
    EXPECT_EQ(arena.getTensor(BufferId::LOGITS), tensor.get());
}

// ============================================================================
// Coherence State Tests
// ============================================================================

TEST(Test__BufferArena, InitialCoherenceIsUninitialized)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu());
    arena.allocate();

    auto state = arena.getCoherenceState(BufferId::HIDDEN_STATE);
    EXPECT_EQ(state.authority, CoherenceState::UNINITIALIZED);
}

TEST(Test__BufferArena, MarkWrittenSetsHostAuthority)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu());
    arena.allocate();

    arena.markWritten(BufferId::HIDDEN_STATE, DeviceId::cpu());

    auto state = arena.getCoherenceState(BufferId::HIDDEN_STATE);
    EXPECT_EQ(state.authority, CoherenceState::HOST);
}

TEST(Test__BufferArena, PrepareForWriteThenMarkWritten)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::FFN_OUTPUT, 4, 896, "FP32", DeviceId::cpu());
    arena.allocate();

    EXPECT_TRUE(arena.prepareForWrite(
        BufferId::FFN_OUTPUT, DeviceId::cpu(), nullptr));
    arena.markWritten(BufferId::FFN_OUTPUT, DeviceId::cpu());

    auto state = arena.getCoherenceState(BufferId::FFN_OUTPUT);
    EXPECT_EQ(state.authority, CoherenceState::HOST);
}

TEST(Test__BufferArena, PrepareForReadOnCPUAfterCPUWrite)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu());
    arena.allocate();

    // Write on CPU
    arena.markWritten(BufferId::HIDDEN_STATE, DeviceId::cpu());

    // Read on CPU — no transfer needed
    EXPECT_TRUE(arena.prepareForRead(BufferId::HIDDEN_STATE, DeviceId::cpu()));
    auto state = arena.getCoherenceState(BufferId::HIDDEN_STATE);
    EXPECT_EQ(state.authority, CoherenceState::HOST);
}

TEST(Test__BufferArena, MarkWrittenWithStreamSetsAuthority)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu());
    arena.allocate();

    // Pass a non-null stream (just a dummy pointer for CPU test)
    int dummy_stream = 0;
    arena.markWritten(BufferId::HIDDEN_STATE, DeviceId::cpu(), &dummy_stream);

    auto state = arena.getCoherenceState(BufferId::HIDDEN_STATE);
    EXPECT_EQ(state.authority, CoherenceState::HOST);
}

TEST(Test__BufferArena, MarkWrittenFlagsOnlySetsAuthority)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu());
    arena.allocate();

    arena.markWrittenFlagsOnly(BufferId::HIDDEN_STATE, DeviceId::cpu());

    auto state = arena.getCoherenceState(BufferId::HIDDEN_STATE);
    EXPECT_EQ(state.authority, CoherenceState::HOST);
}

// ============================================================================
// CoherenceState Tests (direct)
// ============================================================================

TEST(Test__CoherenceState, UninitializedNeedsNoTransfer)
{
    CoherenceState state;
    EXPECT_EQ(state.authority, CoherenceState::UNINITIALIZED);
    EXPECT_FALSE(state.needsTransferTo(DeviceId::cpu()));
    EXPECT_FALSE(state.needsTransferTo(DeviceId::cuda(0)));
}

TEST(Test__CoherenceState, HostNeedsTransferToGPU)
{
    CoherenceState state;
    state.authority = CoherenceState::HOST;
    EXPECT_FALSE(state.needsTransferTo(DeviceId::cpu()));
    EXPECT_TRUE(state.needsTransferTo(DeviceId::cuda(0)));
}

TEST(Test__CoherenceState, DeviceNeedsTransferToCPU)
{
    CoherenceState state;
    state.authority = CoherenceState::DEVICE;
    state.authoritative_device = DeviceId::cuda(0);
    EXPECT_TRUE(state.needsTransferTo(DeviceId::cpu()));
    EXPECT_FALSE(state.needsTransferTo(DeviceId::cuda(0)));
}

TEST(Test__CoherenceState, DeviceNeedsTransferToDifferentGPU)
{
    CoherenceState state;
    state.authority = CoherenceState::DEVICE;
    state.authoritative_device = DeviceId::cuda(0);
    EXPECT_TRUE(state.needsTransferTo(DeviceId::cuda(1)));
}

// ============================================================================
// Borrow Tracking Tests
// ============================================================================

TEST(Test__BufferArena, ReadBorrowIncrementsAndDecrements)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::Q_PROJ, 4, 896, "FP32", DeviceId::cpu());
    arena.allocate();

    arena.acquireReadBorrow(BufferId::Q_PROJ);
    arena.acquireReadBorrow(BufferId::Q_PROJ);
    // Two read borrows should coexist
    EXPECT_FALSE(arena.validateNoBorrowsActive());

    arena.releaseReadBorrow(BufferId::Q_PROJ);
    arena.releaseReadBorrow(BufferId::Q_PROJ);
    EXPECT_TRUE(arena.validateNoBorrowsActive());
}

TEST(Test__BufferArena, WriteBorrowTracking)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::Q_PROJ, 4, 896, "FP32", DeviceId::cpu());
    arena.allocate();

    arena.acquireWriteBorrow(BufferId::Q_PROJ);
    EXPECT_FALSE(arena.validateNoBorrowsActive());

    arena.releaseWriteBorrow(BufferId::Q_PROJ);
    EXPECT_TRUE(arena.validateNoBorrowsActive());
}

#ifndef NDEBUG
// Death tests only work in debug builds where assert() is active

TEST(Test__BufferArena, WriteBorrowConflictsWithReadBorrow)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::Q_PROJ, 4, 896, "FP32", DeviceId::cpu());
    arena.allocate();

    arena.acquireReadBorrow(BufferId::Q_PROJ);
    EXPECT_DEATH(arena.acquireWriteBorrow(BufferId::Q_PROJ), "active read borrow");
    arena.releaseReadBorrow(BufferId::Q_PROJ);
}

TEST(Test__BufferArena, ReadBorrowConflictsWithWriteBorrow)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::Q_PROJ, 4, 896, "FP32", DeviceId::cpu());
    arena.allocate();

    arena.acquireWriteBorrow(BufferId::Q_PROJ);
    EXPECT_DEATH(arena.acquireReadBorrow(BufferId::Q_PROJ), "active write borrow");
    arena.releaseWriteBorrow(BufferId::Q_PROJ);
}

TEST(Test__BufferArena, DoubleWriteBorrowDies)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::Q_PROJ, 4, 896, "FP32", DeviceId::cpu());
    arena.allocate();

    arena.acquireWriteBorrow(BufferId::Q_PROJ);
    EXPECT_DEATH(arena.acquireWriteBorrow(BufferId::Q_PROJ), "double write borrow");
    arena.releaseWriteBorrow(BufferId::Q_PROJ);
}

#endif // NDEBUG

// ============================================================================
// Aliasing Tests
// ============================================================================

TEST(Test__BufferArena, AliasingGroupAssignment)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::ATTN_OUTPUT, 4, 896, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::GATE_PROJ, 4, 4864, "FP32", DeviceId::cpu());
    arena.registerAlias(BufferId::ATTN_OUTPUT, BufferId::GATE_PROJ);

    arena.allocate();

    // Both should be registered and usable
    EXPECT_NE(arena.getTensor(BufferId::ATTN_OUTPUT), nullptr);
    EXPECT_NE(arena.getTensor(BufferId::GATE_PROJ), nullptr);
}

// Verify that registerAlias transitively merges groups:
// alias(A,B) + alias(B,C) → A,B,C in the same group.
// A write borrow on A must block a write borrow on C.
TEST(Test__BufferArena, AliasingGroupMerge)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::ATTN_OUTPUT, 4, 896, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::GATE_PROJ, 4, 4864, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::UP_PROJ, 4, 4864, "FP32", DeviceId::cpu());

    // Create group {ATTN_OUTPUT, GATE_PROJ}, then merge UP_PROJ via GATE_PROJ
    arena.registerAlias(BufferId::ATTN_OUTPUT, BufferId::GATE_PROJ);
    arena.registerAlias(BufferId::GATE_PROJ, BufferId::UP_PROJ);

    arena.allocate();

    // Serial access across all three must work
    arena.acquireWriteBorrow(BufferId::ATTN_OUTPUT);
    arena.releaseWriteBorrow(BufferId::ATTN_OUTPUT);

    arena.acquireWriteBorrow(BufferId::UP_PROJ);
    arena.releaseWriteBorrow(BufferId::UP_PROJ);

    EXPECT_TRUE(arena.validateNoBorrowsActive());
}

// Verify multiple read borrows on aliased buffers are allowed simultaneously.
// Only writes conflict; reads can coexist freely.
TEST(Test__BufferArena, AliasingConcurrentReadsAllowed)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::ATTN_OUTPUT, 4, 896, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::GATE_PROJ, 4, 4864, "FP32", DeviceId::cpu());
    arena.registerAlias(BufferId::ATTN_OUTPUT, BufferId::GATE_PROJ);
    arena.allocate();

    // Two simultaneous read borrows on aliased buffers — should NOT conflict
    arena.acquireReadBorrow(BufferId::ATTN_OUTPUT);
    arena.acquireReadBorrow(BufferId::GATE_PROJ); // must not die
    arena.releaseReadBorrow(BufferId::GATE_PROJ);
    arena.releaseReadBorrow(BufferId::ATTN_OUTPUT);
    EXPECT_TRUE(arena.validateNoBorrowsActive());
}

// Verify aliasing is idempotent — calling registerAlias(A,B) twice should
// not create duplicate groups or cause any issues.
TEST(Test__BufferArena, AliasingIdempotent)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::ATTN_OUTPUT, 4, 896, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::GATE_PROJ, 4, 4864, "FP32", DeviceId::cpu());
    arena.registerAlias(BufferId::ATTN_OUTPUT, BufferId::GATE_PROJ);
    arena.registerAlias(BufferId::ATTN_OUTPUT, BufferId::GATE_PROJ); // no-op
    arena.allocate();

    // Serial borrow still works (group wasn't corrupted by double alias)
    arena.acquireWriteBorrow(BufferId::ATTN_OUTPUT);
    arena.releaseWriteBorrow(BufferId::ATTN_OUTPUT);
    arena.acquireWriteBorrow(BufferId::GATE_PROJ);
    arena.releaseWriteBorrow(BufferId::GATE_PROJ);
    EXPECT_TRUE(arena.validateNoBorrowsActive());
}

TEST(Test__BufferArena, AliasingSerialBorrowsSucceed)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::ATTN_OUTPUT, 4, 896, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::GATE_PROJ, 4, 4864, "FP32", DeviceId::cpu());
    arena.registerAlias(BufferId::ATTN_OUTPUT, BufferId::GATE_PROJ);
    arena.allocate();

    // Can borrow one at a time
    {
        arena.acquireWriteBorrow(BufferId::ATTN_OUTPUT);
        arena.releaseWriteBorrow(BufferId::ATTN_OUTPUT);
    }
    {
        arena.acquireWriteBorrow(BufferId::GATE_PROJ);
        arena.releaseWriteBorrow(BufferId::GATE_PROJ);
    }
    EXPECT_TRUE(arena.validateNoBorrowsActive());
}

#ifndef NDEBUG

TEST(Test__BufferArena, AliasingWriteConflictDies)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::ATTN_OUTPUT, 4, 896, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::GATE_PROJ, 4, 4864, "FP32", DeviceId::cpu());
    arena.registerAlias(BufferId::ATTN_OUTPUT, BufferId::GATE_PROJ);
    arena.allocate();

    arena.acquireWriteBorrow(BufferId::ATTN_OUTPUT);
    EXPECT_DEATH(arena.acquireWriteBorrow(BufferId::GATE_PROJ), "aliased buffer");
    arena.releaseWriteBorrow(BufferId::ATTN_OUTPUT);
}

TEST(Test__BufferArena, AliasingWriteConflictsWithReadDies)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::ATTN_OUTPUT, 4, 896, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::GATE_PROJ, 4, 4864, "FP32", DeviceId::cpu());
    arena.registerAlias(BufferId::ATTN_OUTPUT, BufferId::GATE_PROJ);
    arena.allocate();

    arena.acquireReadBorrow(BufferId::ATTN_OUTPUT);
    EXPECT_DEATH(arena.acquireWriteBorrow(BufferId::GATE_PROJ), "aliased buffer");
    arena.releaseReadBorrow(BufferId::ATTN_OUTPUT);
}

// Verify that group merging propagates conflicts: alias(A,B) + alias(B,C)
// means write on A must block write on C, even though they were never
// directly aliased.
TEST(Test__BufferArena, AliasingMergedGroupWriteConflictDies)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::ATTN_OUTPUT, 4, 896, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::GATE_PROJ, 4, 4864, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::UP_PROJ, 4, 4864, "FP32", DeviceId::cpu());
    arena.registerAlias(BufferId::ATTN_OUTPUT, BufferId::GATE_PROJ);
    arena.registerAlias(BufferId::GATE_PROJ, BufferId::UP_PROJ);
    arena.allocate();

    arena.acquireWriteBorrow(BufferId::ATTN_OUTPUT);
    // UP_PROJ was never directly aliased with ATTN_OUTPUT, but they share
    // a group via GATE_PROJ — must still conflict
    EXPECT_DEATH(arena.acquireWriteBorrow(BufferId::UP_PROJ), "aliased buffer");
    arena.releaseWriteBorrow(BufferId::ATTN_OUTPUT);
}

// Read on aliased buffer must block write on a different member (reversed direction)
TEST(Test__BufferArena, AliasingReadBlocksWriteOnOtherMemberDies)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::ATTN_OUTPUT, 4, 896, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::GATE_PROJ, 4, 4864, "FP32", DeviceId::cpu());
    arena.registerAlias(BufferId::ATTN_OUTPUT, BufferId::GATE_PROJ);
    arena.allocate();

    // Write on GATE_PROJ while ATTN_OUTPUT has read borrow (reversed param order)
    arena.acquireReadBorrow(BufferId::GATE_PROJ);
    EXPECT_DEATH(arena.acquireWriteBorrow(BufferId::ATTN_OUTPUT), "aliased buffer");
    arena.releaseReadBorrow(BufferId::GATE_PROJ);
}

#endif // NDEBUG

// ============================================================================
// Dimensions & DevicePtr Tests
// ============================================================================

TEST(Test__BufferArena, RowsColsMatchRegistration)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu());
    arena.allocate();

    EXPECT_EQ(arena.getRows(BufferId::HIDDEN_STATE), 4u);
    EXPECT_EQ(arena.getCols(BufferId::HIDDEN_STATE), 896u);
}

TEST(Test__BufferArena, DevicePtrReturnsCPUPointer)
{
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu());
    arena.allocate();

    void *ptr = arena.getDevicePtr(BufferId::HIDDEN_STATE, DeviceId::cpu());
    EXPECT_NE(ptr, nullptr);
}

// ============================================================================
// StageBufferContract Tests
// ============================================================================

TEST(Test__StageBufferContract, EmptyContractIsEmpty)
{
    auto contract = StageBufferContract::build();
    EXPECT_TRUE(contract.empty());
    EXPECT_EQ(contract.bindingCount(), 0u);
}

TEST(Test__StageBufferContract, FluentBuilderAddsBindings)
{
    // Create a dummy tensor to use as a weight
    auto weight_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{64, 896}, DeviceId::cpu());

    auto contract = StageBufferContract::build()
                        .addInput(BufferId::HIDDEN_STATE, "FP32")
                        .addWeight(weight_tensor.get())
                        .addOutput(BufferId::Q_PROJ, "FP32");

    EXPECT_FALSE(contract.empty());
    EXPECT_EQ(contract.inputs.size(), 1u);
    EXPECT_EQ(contract.weight_tensors.size(), 1u);
    EXPECT_EQ(contract.outputs.size(), 1u);
    EXPECT_EQ(contract.bindingCount(), 3u);
}

TEST(Test__StageBufferContract, NullWeightIsIgnored)
{
    auto contract = StageBufferContract::build()
                        .addWeight(nullptr);

    EXPECT_TRUE(contract.empty());
    EXPECT_EQ(contract.weight_tensors.size(), 0u);
}

TEST(Test__StageBufferContract, InOutBinding)
{
    auto contract = StageBufferContract::build()
                        .addInOut(BufferId::HIDDEN_STATE, "FP32");

    EXPECT_EQ(contract.inouts.size(), 1u);
    EXPECT_EQ(contract.inouts[0].access, BufferAccess::READWRITE);
    EXPECT_TRUE(contract.inouts[0].prepare_write_storage);
}

TEST(Test__StageBufferContract, PreallocatedInOutSkipsWritePreparationOnly)
{
    auto contract = StageBufferContract::build()
                        .addOutput(BufferId::ATTN_OUTPUT)
                        .addPreallocatedInOut(BufferId::HIDDEN_STATE, "FP32");

    ASSERT_EQ(contract.inouts.size(), 1u);
    EXPECT_EQ(contract.inouts[0].access, BufferAccess::READWRITE);
    EXPECT_FALSE(contract.inouts[0].prepare_write_storage);

    auto reads = contract.allArenaReads();
    EXPECT_EQ(reads.size(), 1u);
    EXPECT_EQ(reads[0].id, BufferId::HIDDEN_STATE);

    auto writes = contract.allWrites();
    EXPECT_EQ(writes.size(), 2u);

    auto write_preps = contract.writesRequiringPrepare();
    ASSERT_EQ(write_preps.size(), 1u);
    EXPECT_EQ(write_preps[0].id, BufferId::ATTN_OUTPUT);
}

TEST(Test__StageBufferContract, PreallocatedOutputIsWriteOnlyAndSkipsRebinding)
{
    auto contract = StageBufferContract::build()
                        .addInput(BufferId::STOCHASTIC_BATCH_OUTPUT_META)
                        .addPreallocatedOutput(
                            BufferId::MTP_CONDITION_TOKEN,
                            "INT32");

    ASSERT_EQ(contract.outputs.size(), 1u);
    EXPECT_EQ(contract.outputs[0].id, BufferId::MTP_CONDITION_TOKEN);
    EXPECT_EQ(contract.outputs[0].access, BufferAccess::WRITE);
    EXPECT_FALSE(contract.outputs[0].prepare_write_storage);

    const auto reads = contract.allArenaReads();
    ASSERT_EQ(reads.size(), 1u);
    EXPECT_EQ(reads[0].id, BufferId::STOCHASTIC_BATCH_OUTPUT_META)
        << "A persistent output mailbox must not become a graph-frontier read";

    const auto writes = contract.allWrites();
    ASSERT_EQ(writes.size(), 1u);
    EXPECT_EQ(writes[0].id, BufferId::MTP_CONDITION_TOKEN);
    EXPECT_TRUE(contract.writesRequiringPrepare().empty())
        << "Setup-owned mailbox storage must retain its capture-time address";
}

TEST(Test__StageBufferContract, WorkspaceBinding)
{
    auto contract = StageBufferContract::build()
                        .addWorkspace("attn_scores", 1024 * 1024, 64, true);

    EXPECT_EQ(contract.workspaces.size(), 1u);
    EXPECT_STREQ(contract.workspaces[0].name, "attn_scores");
    EXPECT_EQ(contract.workspaces[0].size_bytes, 1024u * 1024u);
}

TEST(Test__StageBufferContract, AllArenaReadsGathersInputsAndInouts)
{
    auto weight_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{64, 896}, DeviceId::cpu());

    auto contract = StageBufferContract::build()
                        .addInput(BufferId::HIDDEN_STATE)
                        .addWeight(weight_tensor.get())
                        .addInOut(BufferId::RESIDUAL)
                        .addOutput(BufferId::ATTN_OUTPUT);

    auto reads = contract.allArenaReads();
    EXPECT_EQ(reads.size(), 2u); // input + inout (weights excluded from arena reads)
}

TEST(Test__StageBufferContract, AllWritesGathersOutputsInouts)
{
    auto contract = StageBufferContract::build()
                        .addOutput(BufferId::ATTN_OUTPUT)
                        .addInOut(BufferId::RESIDUAL);

    auto writes = contract.allWrites();
    EXPECT_EQ(writes.size(), 2u); // output + inout
}

TEST(Test__StageBufferContract,
     GraphDependencyTrackerExcludesProducedEmbeddingFromCaptureInputs)
{
    GraphArenaDependencyTracker tracker;

    const auto embedding = StageBufferContract::build()
                               .addOutput(BufferId::MTP_EMBEDDING);
    EXPECT_TRUE(tracker.observeStage(embedding).empty());

    const auto embedding_allreduce = StageBufferContract::build()
                                         .addPreallocatedInOut(
                                             BufferId::MTP_EMBEDDING);
    EXPECT_TRUE(tracker.observeStage(embedding_allreduce).empty())
        << "An in-place collective consumes the embedding produced inside the same graph";

    const auto norm_embedding = StageBufferContract::build()
                                    .addInput(BufferId::MTP_EMBEDDING)
                                    .addInput(BufferId::HIDDEN_STATE)
                                    .addOutput(BufferId::MTP_NORM_EMBEDDING);
    const auto external_reads = tracker.observeStage(norm_embedding);
    ASSERT_EQ(external_reads.size(), 1u);
    EXPECT_EQ(external_reads.front().id, BufferId::HIDDEN_STATE)
        << "Only the graph-external hidden state requires a pre-capture event join";
}

TEST(Test__StageBufferContract,
     GraphDependencyTrackerTreatsUnproducedInOutAsExternal)
{
    GraphArenaDependencyTracker tracker;
    const auto external_allreduce = StageBufferContract::build()
                                        .addPreallocatedInOut(
                                            BufferId::MTP_EMBEDDING);

    const auto external_reads = tracker.observeStage(external_allreduce);
    ASSERT_EQ(external_reads.size(), 1u);
    EXPECT_EQ(external_reads.front().id, BufferId::MTP_EMBEDDING)
        << "A capture unit beginning with an in-place consumer must join its prior segment";
}

// ============================================================================
// StageBoundBuffers Tests
// ============================================================================

TEST(Test__StageBoundBuffers, AddAndRetrieveEntry)
{
    StageBoundBuffers bound;

    float dummy_data[100] = {};
    StageBoundBuffers::Entry entry;
    entry.id = BufferId::HIDDEN_STATE;
    entry.access = BufferAccess::READ;
    entry.device_ptr = dummy_data;
    entry.tensor = nullptr;
    entry.rows = 4;
    entry.cols = 25;
    entry.device = DeviceId::cpu();

    bound.addEntry(entry);

    EXPECT_TRUE(bound.has(BufferId::HIDDEN_STATE));
    EXPECT_FALSE(bound.has(BufferId::Q_PROJ));
    EXPECT_EQ(bound.size(), 1u);

    auto view = bound.input<float>(BufferId::HIDDEN_STATE);
    EXPECT_EQ(view.rows(), 4u);
    EXPECT_EQ(view.cols(), 25u);
    EXPECT_EQ(view.read_ptr(), dummy_data);
    EXPECT_TRUE(view.valid());
}

TEST(Test__StageBoundBuffers, OutputViewProvidesMutablePointer)
{
    StageBoundBuffers bound;

    float dummy_data[100] = {};
    StageBoundBuffers::Entry entry;
    entry.id = BufferId::Q_PROJ;
    entry.access = BufferAccess::WRITE;
    entry.device_ptr = dummy_data;
    entry.rows = 4;
    entry.cols = 25;
    entry.device = DeviceId::cpu();

    bound.addEntry(entry);

    auto view = bound.output<float>(BufferId::Q_PROJ);
    EXPECT_EQ(view.write_ptr(), dummy_data);

    // Can actually write through the view
    view.write_ptr()[0] = 42.0f;
    EXPECT_FLOAT_EQ(dummy_data[0], 42.0f);
}

TEST(Test__StageBoundBuffers, InOutViewProvidesBothPointers)
{
    StageBoundBuffers bound;

    float dummy_data[100] = {};
    dummy_data[0] = 1.5f;

    StageBoundBuffers::Entry entry;
    entry.id = BufferId::RESIDUAL;
    entry.access = BufferAccess::READWRITE;
    entry.device_ptr = dummy_data;
    entry.rows = 2;
    entry.cols = 50;
    entry.device = DeviceId::cpu();

    bound.addEntry(entry);

    auto view = bound.inout<float>(BufferId::RESIDUAL);
    EXPECT_FLOAT_EQ(view.read_ptr()[0], 1.5f);
    view.write_ptr()[1] = 2.5f;
    EXPECT_FLOAT_EQ(dummy_data[1], 2.5f);
}

TEST(Test__StageBoundBuffers, WorkspaceReturnsNullForUnknown)
{
    StageBoundBuffers bound;
    EXPECT_EQ(bound.workspace("nonexistent"), nullptr);
}

TEST(Test__StageBoundBuffers, WorkspaceReturnsPointer)
{
    StageBoundBuffers bound;
    float dummy[10];
    bound.addWorkspace("scratch", dummy);
    EXPECT_EQ(bound.workspace("scratch"), dummy);
}

TEST(Test__StageBoundBuffers, MissingBufferThrows)
{
    StageBoundBuffers bound;
    EXPECT_THROW(bound.input<float>(BufferId::HIDDEN_STATE), std::runtime_error);
}

// ============================================================================
// BufferView Tests
// ============================================================================

TEST(Test__BufferView, ReadViewProperties)
{
    float data[100] = {};
    BufferView<float, BufferAccess::READ> view(data, nullptr, 10, 10, DeviceId::cpu());

    EXPECT_EQ(view.rows(), 10u);
    EXPECT_EQ(view.cols(), 10u);
    EXPECT_EQ(view.numel(), 100u);
    EXPECT_EQ(view.read_ptr(), data);
    EXPECT_TRUE(view.valid());
    EXPECT_TRUE(view.device().is_cpu());
}

TEST(Test__BufferView, WriteViewProperties)
{
    float data[100] = {};
    BufferView<float, BufferAccess::WRITE> view(data, nullptr, 5, 20, DeviceId::cpu());

    EXPECT_EQ(view.write_ptr(), data);
    view.write_ptr()[0] = 3.14f;
    EXPECT_FLOAT_EQ(data[0], 3.14f);
}

TEST(Test__BufferView, DefaultConstructedIsInvalid)
{
    BufferView<float, BufferAccess::READ> view;
    EXPECT_FALSE(view.valid());
    EXPECT_EQ(view.read_ptr(), nullptr);
}

// ============================================================================
// BufferId Tests
// ============================================================================

TEST(Test__BufferId, NameRoundTrips)
{
    EXPECT_STREQ(bufferIdName(BufferId::HIDDEN_STATE), "HIDDEN_STATE");
    EXPECT_STREQ(bufferIdName(BufferId::Q_PROJ), "Q_PROJ");
    EXPECT_STREQ(bufferIdName(BufferId::ALLREDUCE_STAGING), "ALLREDUCE_STAGING");
    EXPECT_STREQ(bufferIdName(BufferId::MTP_PROJECTED), "MTP_PROJECTED");
    EXPECT_STREQ(bufferIdName(BufferId::MTP_LOGITS), "MTP_LOGITS");
    EXPECT_STREQ(
        bufferIdName(BufferId::MTP_LOGITS_GATHERED),
        "MTP_LOGITS_GATHERED");
}

/** Every declared ID must fit the arena's enum-sized registry without a magic ceiling. */
TEST(Test__BufferId, EveryDeclaredIdHasUniqueNameAndArenaSlot)
{
    const auto count = static_cast<size_t>(BufferId::_COUNT);
    ASSERT_GT(count, 0u);
    BufferArena arena(fixtureArenaConfig());
    std::set<std::string> names;
    for (size_t index = 0; index < count; ++index)
    {
        const auto id = static_cast<BufferId>(index);
        SCOPED_TRACE(index);
        EXPECT_STRNE(bufferIdName(id), "UNKNOWN");
        EXPECT_TRUE(names.insert(bufferIdName(id)).second);
        // Registration is metadata-only, so even device scratch roles can be
        // checked on CPU without making this Unit test allocate GPU storage.
        ASSERT_TRUE(arena.registerBuffer(id, 1, index + 1, "FP32", DeviceId::cpu()));
        EXPECT_TRUE(arena.isRegistered(id));
        EXPECT_EQ(arena.getCols(id), index + 1);
    }
    EXPECT_EQ(arena.registeredCount(), count);
}

// ============================================================================
// KVBufferId Tests
// ============================================================================

TEST(Test__KVBufferId, EqualityComparison)
{
    KVBufferId a{0, KVBufferId::KEY};
    KVBufferId b{0, KVBufferId::KEY};
    KVBufferId c{0, KVBufferId::VALUE};
    KVBufferId d{1, KVBufferId::KEY};

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
    EXPECT_NE(a, d);
}

TEST(Test__KVBufferId, ToString)
{
    KVBufferId key{3, KVBufferId::KEY};
    KVBufferId val{5, KVBufferId::VALUE};

    EXPECT_EQ(key.to_string(), "KV_KEY_L3");
    EXPECT_EQ(val.to_string(), "KV_VALUE_L5");
}

// ============================================================================
// Integration: Arena + Contract + BoundBuffers
// ============================================================================

TEST(Test__BufferArena, EndToEndCPUWorkflow)
{
    // 1. Register buffers
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::HIDDEN_STATE, 2, 64, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::Q_PROJ, 2, 64, "FP32", DeviceId::cpu());
    ASSERT_TRUE(arena.allocate());

    // 2. Define a stage contract
    auto contract = StageBufferContract::build()
                        .addInput(BufferId::HIDDEN_STATE, "FP32")
                        .addOutput(BufferId::Q_PROJ, "FP32");

    // 3. Simulate executor: prepare for read/write
    DeviceId cpu = DeviceId::cpu();

    // Write some input data
    auto *hs = dynamic_cast<TensorBase *>(arena.getTensor(BufferId::HIDDEN_STATE));
    ASSERT_NE(hs, nullptr);
    float *hs_ptr = dynamic_cast<FP32Tensor *>(hs)->mutable_data();
    for (int i = 0; i < 128; ++i)
        hs_ptr[i] = static_cast<float>(i);
    arena.markWritten(BufferId::HIDDEN_STATE, cpu);

    // 4. Prepare for stage execution
    EXPECT_TRUE(arena.prepareForRead(BufferId::HIDDEN_STATE, cpu));
    EXPECT_TRUE(arena.prepareForWrite(BufferId::Q_PROJ, cpu, nullptr));

    // 5. Build bound buffers
    arena.acquireReadBorrow(BufferId::HIDDEN_STATE);
    arena.acquireWriteBorrow(BufferId::Q_PROJ);

    StageBoundBuffers bound;
    StageBoundBuffers::Entry input_entry;
    input_entry.id = BufferId::HIDDEN_STATE;
    input_entry.access = BufferAccess::READ;
    input_entry.tensor = arena.getTensor(BufferId::HIDDEN_STATE);
    input_entry.device_ptr = arena.getDevicePtr(BufferId::HIDDEN_STATE, cpu);
    input_entry.rows = arena.getRows(BufferId::HIDDEN_STATE);
    input_entry.cols = arena.getCols(BufferId::HIDDEN_STATE);
    input_entry.device = cpu;
    bound.addEntry(input_entry);

    StageBoundBuffers::Entry output_entry;
    output_entry.id = BufferId::Q_PROJ;
    output_entry.access = BufferAccess::WRITE;
    output_entry.tensor = arena.getTensor(BufferId::Q_PROJ);
    output_entry.device_ptr = arena.getDevicePtr(BufferId::Q_PROJ, cpu);
    output_entry.rows = arena.getRows(BufferId::Q_PROJ);
    output_entry.cols = arena.getCols(BufferId::Q_PROJ);
    output_entry.device = cpu;
    bound.addEntry(output_entry);

    // 6. "Execute" the stage (just copy input to output)
    auto inp = bound.input<float>(BufferId::HIDDEN_STATE);
    auto out = bound.output<float>(BufferId::Q_PROJ);
    for (size_t i = 0; i < inp.numel(); ++i)
        out.write_ptr()[i] = inp.read_ptr()[i] * 2.0f;

    // 7. Release borrows and mark output written
    arena.releaseReadBorrow(BufferId::HIDDEN_STATE);
    arena.releaseWriteBorrow(BufferId::Q_PROJ);
    arena.markWritten(BufferId::Q_PROJ, cpu);

    // 8. Validate results
    EXPECT_TRUE(arena.validateNoBorrowsActive());
    auto state = arena.getCoherenceState(BufferId::Q_PROJ);
    EXPECT_EQ(state.authority, CoherenceState::HOST);

    // Verify the data was actually written
    auto *q_tensor = dynamic_cast<FP32Tensor *>(arena.getTensor(BufferId::Q_PROJ));
    ASSERT_NE(q_tensor, nullptr);
    EXPECT_FLOAT_EQ(q_tensor->data()[0], 0.0f);
    EXPECT_FLOAT_EQ(q_tensor->data()[1], 2.0f);
    EXPECT_FLOAT_EQ(q_tensor->data()[2], 4.0f);
}

// ============================================================================
// Phase 2: Arena + Contract Coherence Integration
// ============================================================================

TEST(Test__ArenaContractCoherence, PrepareForReadWriteFromContract)
{
    // Simulate the executor's contract-based coherence flow on CPU:
    // 1. Register external buffers matching SwiGLU's contract
    // 2. Use the contract to drive prepareForRead/Write
    // 3. Execute SwiGLU
    // 4. markWritten for outputs

    BufferArena arena(fixtureArenaConfig());
    DeviceId cpu = DeviceId::cpu();

    // Create tensors (simulating what buffer_manager_ allocates)
    auto gate = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 8}, DeviceId::cpu());
    auto up = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 8}, DeviceId::cpu());

    // Fill gate and up with test data
    for (size_t i = 0; i < 16; ++i)
    {
        gate->mutable_data()[i] = static_cast<float>(i) * 0.1f;
        up->mutable_data()[i] = static_cast<float>(i) * 0.2f;
    }

    // Register as external buffers (same as initializeArena does)
    ASSERT_TRUE(arena.registerExternalBuffer(BufferId::GATE_PROJ, gate.get()));
    ASSERT_TRUE(arena.registerExternalBuffer(BufferId::UP_PROJ, up.get()));

    // Build contract (same as SwiGLUStage::bufferContract would return)
    auto contract = StageBufferContract::build()
                        .addInput(BufferId::GATE_PROJ)
                        .addInOut(BufferId::UP_PROJ);

    // Simulate executor coherence: prepare reads
    for (const auto &binding : contract.allArenaReads())
    {
        EXPECT_TRUE(arena.prepareForRead(binding.id, cpu));
    }

    // Simulate executor coherence: prepare writes
    for (const auto &binding : contract.allWrites())
    {
        EXPECT_TRUE(arena.prepareForWrite(binding.id, cpu, nullptr));
    }

    // (Stage would execute here — we just verify coherence tracking)

    // Mark written on outputs
    for (const auto &binding : contract.allWrites())
    {
        arena.markWritten(binding.id, cpu);
    }

    // Verify coherence: UP_PROJ should be HOST-authoritative after CPU write
    auto state = arena.getCoherenceState(BufferId::UP_PROJ);
    EXPECT_EQ(state.authority, CoherenceState::HOST);

    // GATE_PROJ was only read, should still be HOST-authoritative
    auto gate_state = arena.getCoherenceState(BufferId::GATE_PROJ);
    EXPECT_EQ(gate_state.authority, CoherenceState::HOST);
}

TEST(Test__ArenaContractCoherence, MarkWrittenFlagsOnlyDoesNotRecordEvent)
{
    // On CPU, markWrittenFlagsOnly should update coherence state
    // without trying to record a GPU event
    BufferArena arena(fixtureArenaConfig());
    DeviceId cpu = DeviceId::cpu();

    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 4}, DeviceId::cpu());
    ASSERT_TRUE(arena.registerExternalBuffer(BufferId::FFN_OUTPUT, tensor.get()));

    // No stream (CPU) — markWrittenFlagsOnly should work
    arena.markWrittenFlagsOnly(BufferId::FFN_OUTPUT, cpu);

    auto state = arena.getCoherenceState(BufferId::FFN_OUTPUT);
    EXPECT_EQ(state.authority, CoherenceState::HOST);
}

// ============================================================================
// Factory-based Allocation Tests (Phase 2)
// ============================================================================

TEST(Test__BufferArena, AllocateWithFactoryCreatesFP32)
{
    // Create a real TensorFactory for the test
    MPIContext mpi_ctx(0, 1);
    TensorFactory factory(mpi_ctx);

    ArenaConfig config = fixtureArenaConfig();
    config.factory = &factory;

    BufferArena arena(config);
    arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::Q_PROJ, 4, 128, "FP32", DeviceId::cpu());

    EXPECT_TRUE(arena.allocate());

    auto *t1 = arena.getTensor(BufferId::HIDDEN_STATE);
    auto *t2 = arena.getTensor(BufferId::Q_PROJ);
    ASSERT_NE(t1, nullptr);
    ASSERT_NE(t2, nullptr);
    EXPECT_EQ(t1->rows(), 4u);
    EXPECT_EQ(t1->cols(), 896u);
    EXPECT_EQ(t2->rows(), 4u);
    EXPECT_EQ(t2->cols(), 128u);
}

TEST(Test__BufferArena, GpuActivationAndPublicationBuffersRemainDeviceLocal)
{
    MPIContext mpi_ctx(0, 1);
    TensorFactory factory(mpi_ctx);

    ArenaConfig config = fixtureArenaConfig();
    config.factory = &factory;

    BufferArena arena(config);
    ASSERT_TRUE(arena.registerBuffer(BufferId::GDN_ALPHA, 256, 16, "FP32", DeviceId::rocm(0)));
    ASSERT_TRUE(arena.registerBuffer(BufferId::GDN_BETA, 256, 16, "FP32", DeviceId::rocm(0)));
    ASSERT_TRUE(arena.registerBuffer(BufferId::LOGITS, 16, 4096, "FP32", DeviceId::rocm(0)));
    ASSERT_TRUE(arena.registerBuffer(BufferId::LOGITS_LOCAL, 16, 2048, "FP32", DeviceId::rocm(0)));
    ASSERT_TRUE(arena.registerBuffer(BufferId::ALL_POSITION_LOGITS, 16, 4096, "FP32", DeviceId::rocm(0)));
    ASSERT_TRUE(arena.registerBuffer(BufferId::ALL_POSITION_LOGITS_LOCAL, 16, 2048, "FP32", DeviceId::rocm(0)));
    ASSERT_TRUE(arena.registerBuffer(BufferId::PREFIX_TERMINAL_LOGITS, 1, 4096, "FP32", DeviceId::rocm(0)));
    ASSERT_TRUE(arena.registerBuffer(BufferId::STOCHASTIC_PROCESSED_LOGITS, 1, 4096, "FP32", DeviceId::rocm(0)));
    ASSERT_TRUE(arena.registerBuffer(BufferId::MTP_LOGITS, 16, 4096, "FP32", DeviceId::rocm(0)));
    ASSERT_TRUE(arena.registerBuffer(BufferId::MTP_LOGITS_GATHERED, 16, 8192, "FP32", DeviceId::rocm(0)));

    ASSERT_TRUE(arena.allocate());

    const std::array<BufferId, 10> graph_buffers{
        BufferId::GDN_ALPHA,
        BufferId::GDN_BETA,
        BufferId::LOGITS,
        BufferId::LOGITS_LOCAL,
        BufferId::ALL_POSITION_LOGITS,
        BufferId::ALL_POSITION_LOGITS_LOCAL,
        BufferId::PREFIX_TERMINAL_LOGITS,
        BufferId::STOCHASTIC_PROCESSED_LOGITS,
        BufferId::MTP_LOGITS,
        BufferId::MTP_LOGITS_GATHERED,
    };
    for (const BufferId id : graph_buffers)
    {
        auto *tensor = dynamic_cast<TensorBase *>(arena.getTensor(id));
        ASSERT_NE(tensor, nullptr) << bufferIdName(id);
        EXPECT_FALSE(tensor->isMapped())
            << bufferIdName(id)
            << " is graph activation/publication storage; host-visible mappings "
               "must be explicit TransferEngine regions";
        EXPECT_EQ(tensor->home_device(), DeviceId::rocm(0))
            << bufferIdName(id);
    }
}

TEST(Test__BufferArena, AllocateWithFactoryCreatesBF16)
{
    MPIContext mpi_ctx(0, 1);
    TensorFactory factory(mpi_ctx);

    ArenaConfig config = fixtureArenaConfig();
    config.factory = &factory;

    BufferArena arena(config);
    arena.registerBuffer(BufferId::NORMALIZED, 4, 896, "BF16", DeviceId::cpu());

    EXPECT_TRUE(arena.allocate());

    auto *t = arena.getTensor(BufferId::NORMALIZED);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->rows(), 4u);
    EXPECT_EQ(t->cols(), 896u);
}

TEST(Test__BufferArena, AllocateWithFactoryCreatesQ8_1)
{
    MPIContext mpi_ctx(0, 1);
    TensorFactory factory(mpi_ctx);

    ArenaConfig config = fixtureArenaConfig();
    config.factory = &factory;

    BufferArena arena(config);
    arena.registerBuffer(BufferId::Q_QUANTIZED, 4, 128, "Q8_1", DeviceId::cpu());

    EXPECT_TRUE(arena.allocate());

    auto *t = arena.getTensor(BufferId::Q_QUANTIZED);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->rows(), 4u);
}

TEST(Test__BufferArena, AllocationStatsTracked)
{
    MPIContext mpi_ctx(0, 1);
    TensorFactory factory(mpi_ctx);

    ArenaConfig config = fixtureArenaConfig();
    config.factory = &factory;

    BufferArena arena(config);
    arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::Q_PROJ, 4, 128, "FP32", DeviceId::cpu());

    EXPECT_TRUE(arena.allocate());

    const auto &stats = arena.stats();
    EXPECT_EQ(stats.total_buffers, 2u);
    EXPECT_GT(stats.total_bytes, 0u);
    // FP32: 4*896*4 + 4*128*4 = 14336 + 2048 = 16384 bytes
    EXPECT_EQ(stats.total_bytes, 4u * 896u * 4u + 4u * 128u * 4u);
}

TEST(Test__BufferArena, SetConfigAfterConstruction)
{
    MPIContext mpi_ctx(0, 1);
    TensorFactory factory(mpi_ctx);

    BufferArena arena(fixtureArenaConfig()); // No config initially
    arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu());

    // Set config before allocate
    ArenaConfig config = fixtureArenaConfig();
    config.factory = &factory;
    arena.setConfig(config);

    EXPECT_TRUE(arena.allocate());
    EXPECT_NE(arena.getTensor(BufferId::HIDDEN_STATE), nullptr);
    EXPECT_EQ(arena.stats().total_buffers, 1u);
}

TEST(Test__BufferArena, TypedStorageWithoutFactory)
{
    // Explicit CPU storage works without a factory; its admitted dtype is preserved.
    BufferArena arena(fixtureArenaConfig());
    arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu());

    EXPECT_TRUE(arena.allocate());
    auto *t = arena.getTensor(BufferId::HIDDEN_STATE);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->rows(), 4u);
    EXPECT_EQ(t->cols(), 896u);
    EXPECT_EQ(arena.stats().total_buffers, 1u);
}

/** @test CPU physical ownership is mandatory at the allocating boundary. */
TEST(Test__BufferArenaPhysicalMemory, RejectsMissingAuthority)
{
    BufferArena arena;
    ASSERT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE, 1, 1, "FP32", DeviceId::cpu()));
    EXPECT_THROW(arena.allocate(), std::logic_error);
    EXPECT_EQ(arena.getSharedTensor(BufferId::HIDDEN_STATE), nullptr);
}

/** @test Every activation representation claims its allocator's exact capacity. */
TEST(Test__BufferArenaPhysicalMemory, ClaimsRoundedPhysicalBytesForEveryDtype)
{
    MPIContext mpi(0, 1);
    TensorFactory factory(mpi);
    const std::array cases{
        std::pair{"FP32", size_t{2 * 33 * sizeof(float)}},
        std::pair{"FP16", size_t{2 * 33 * sizeof(uint16_t)}},
        std::pair{"BF16", size_t{2 * 33 * sizeof(uint16_t)}},
        std::pair{"INT32", size_t{2 * 33 * sizeof(int32_t)}},
        std::pair{"Q8_1", size_t{2 * 2 * sizeof(Q8_1Block)}},
        std::pair{"Q16_1", size_t{2 * 2 * sizeof(Q16_1Block)}}};
    for (const auto &[dtype, payload] : cases)
    {
        SCOPED_TRACE(dtype);
        const auto bytes = AlignedVector<uint8_t>::requiredAllocationBytes(payload);
        auto authority = arenaAuthority(bytes);
        {
            BufferArena arena({.factory = &factory,
                               .physical_memory_authority = authority});
            ASSERT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE, 2, 33, dtype, DeviceId::cpu()));
            ASSERT_TRUE(arena.allocate());
            auto root = arena.getSharedTensor(BufferId::HIDDEN_STATE);
            auto shared_root = root->shared_from_this();
            EXPECT_EQ(shared_root.get(), root.get());
            EXPECT_FALSE(shared_root.owner_before(root));
            EXPECT_FALSE(root.owner_before(shared_root));
            EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationArena, PhysicalMemoryMaterializationKind::NewAllocation), bytes);
            EXPECT_FALSE(arena.allocate());
            EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationArena, PhysicalMemoryMaterializationKind::NewAllocation), bytes);
        }
        EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationArena, PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    }
}

/** @test Admission rejects a bounded over-budget extent before any root is constructed. */
TEST(Test__BufferArenaPhysicalMemory, RejectsCapacityBeforeConstruction)
{
    auto authority = arenaAuthority(4096);
    BufferArena arena({.physical_memory_authority = authority});
    ASSERT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE, 1, 1025, "FP32", DeviceId::cpu()));
    EXPECT_THROW(arena.allocate(), std::logic_error);
    EXPECT_EQ(arena.getSharedTensor(BufferId::HIDDEN_STATE), nullptr);
    EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationArena, PhysicalMemoryMaterializationKind::NewAllocation), 0u);
}

/** @test A structural view and shared family reference retain exactly one physical claim. */
TEST(Test__BufferArenaPhysicalMemory, ClaimFollowsFinalRootAndViewLifetime)
{
    const auto bytes = AlignedVector<float>::requiredAllocationBytes(33);
    auto authority = arenaAuthority(bytes);
    std::shared_ptr<TensorBase> retained, view;
    std::weak_ptr<TensorBase> weak;
    {
        BufferArena arena({.physical_memory_authority = authority});
        ASSERT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE, 1, 33, "FP32", DeviceId::cpu()));
        ASSERT_TRUE(arena.allocate());
        retained = arena.getSharedTensor(BufferId::HIDDEN_STATE);
        weak = retained;
        view = retained->create_view({1, 16}, 1);
        ASSERT_NE(view, nullptr);
        BufferArena family;
        ASSERT_TRUE(family.registerExternalBuffer(BufferId::HIDDEN_STATE, retained.get()));
        ASSERT_TRUE(family.registerExternalBuffer(BufferId::Q_PROJ, view.get()));
        ASSERT_TRUE(family.allocate());
        EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationArena, PhysicalMemoryMaterializationKind::NewAllocation), bytes);
    }
    retained.reset();
    EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationArena, PhysicalMemoryMaterializationKind::NewAllocation), bytes);
    view.reset();
    EXPECT_TRUE(weak.expired());
    // Retaining only a weak control block must not retain materialized bytes.
    EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationArena, PhysicalMemoryMaterializationKind::NewAllocation), 0u);
}

/** @test Declared borrow aliases own distinct CPU allocations and each is claimed once. */
TEST(Test__BufferArenaPhysicalMemory, AliasDeclarationsDoNotInventSharedCpuStorage)
{
    const auto bytes = AlignedVector<float>::requiredAllocationBytes(1);
    auto authority = arenaAuthority(2 * bytes);
    BufferArena arena({.physical_memory_authority = authority});
    ASSERT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE, 1, 1, "FP32", DeviceId::cpu()));
    ASSERT_TRUE(arena.registerBuffer(BufferId::Q_PROJ, 1, 1, "FP32", DeviceId::cpu()));
    arena.registerAlias(BufferId::HIDDEN_STATE, BufferId::Q_PROJ);
    ASSERT_TRUE(arena.allocate());
    EXPECT_NE(arena.getSharedTensor(BufferId::HIDDEN_STATE)->raw_data(),
              arena.getSharedTensor(BufferId::Q_PROJ)->raw_data());
    EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationArena, PhysicalMemoryMaterializationKind::NewAllocation), 2 * bytes);
}

/** @test Replay across page/huge-page thresholds retires each exact allocation. */
TEST(Test__BufferArenaPhysicalMemory, RepeatedSmallLargeSmallRootsUseExactAllocatorRounding)
{
    const std::array<size_t, 5> counts{1, 1025, 524289, 1025, 1};
    const auto maximum = AlignedVector<float>::requiredAllocationBytes(524289);
    auto authority = arenaAuthority(maximum);
    for (const auto count : counts)
    {
        SCOPED_TRACE(count);
        {
            BufferArena arena({.physical_memory_authority = authority});
            ASSERT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE, 1, count, "FP32", DeviceId::cpu()));
            ASSERT_TRUE(arena.allocate());
            EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationArena,
                                             PhysicalMemoryMaterializationKind::NewAllocation),
                      AlignedVector<float>::requiredAllocationBytes(count));
        }
        EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationArena,
                                         PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    }
}

/** @test Partial setup failure keeps only constructed roots live until their final owner retires. */
TEST(Test__BufferArenaPhysicalMemory, PartialAllocationFailureRetiresOnlyConstructedRoots)
{
    const auto bytes = AlignedVector<float>::requiredAllocationBytes(1);
    auto authority = arenaAuthority(bytes);
    std::shared_ptr<TensorBase> retained;
    {
        BufferArena arena({.physical_memory_authority = authority});
        // HIDDEN_STATE precedes Q_PROJ in the canonical BufferId order.
        ASSERT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE, 1, 1, "FP32", DeviceId::cpu()));
        ASSERT_TRUE(arena.registerBuffer(BufferId::Q_PROJ, 1, 1025, "FP32", DeviceId::cpu()));
        EXPECT_THROW(arena.allocate(), std::logic_error);
        retained = arena.getSharedTensor(BufferId::HIDDEN_STATE);
        ASSERT_NE(retained, nullptr);
        EXPECT_EQ(arena.getSharedTensor(BufferId::Q_PROJ), nullptr);
        EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationArena,
            PhysicalMemoryMaterializationKind::NewAllocation), bytes);
        EXPECT_THROW(arena.allocate(), std::logic_error);
        EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationArena,
            PhysicalMemoryMaterializationKind::NewAllocation), bytes);
    }
    EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationArena,
        PhysicalMemoryMaterializationKind::NewAllocation), bytes);
    retained.reset();
    EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationArena,
        PhysicalMemoryMaterializationKind::NewAllocation), 0u);
}

/** @test A NUMA CPU participant charges its admitted allocator, never CPU ordinal zero. */
TEST(Test__BufferArenaPhysicalMemory, UsesExactCpuAllocatorIdentity)
{
    const DeviceId device(DeviceType::CPU, 1);
    const auto bytes = AlignedVector<float>::requiredAllocationBytes(1);
    auto authority = arenaAuthority(bytes, device);
    BufferArena arena({.physical_memory_authority = authority});
    ASSERT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE, 1, 1, "FP32", device));
    ASSERT_TRUE(arena.allocate());
    EXPECT_EQ(authority->claimedBytes(device, PhysicalMemoryOwner::ActivationArena,
        PhysicalMemoryMaterializationKind::NewAllocation), bytes);
    EXPECT_FALSE(authority->contains(DeviceId::cpu()));
}

/** @test The shared declaration governs all six CPU dtypes even without a factory. */
TEST(Test__BufferArenaPhysicalMemory, NativeDeclarationAllSixDtypesWithoutFactory)
{
    const DeviceId cpu1{DeviceType::CPU,1};
    for(const auto dtype : {"FP32","FP16","BF16","INT32","Q8_1","Q16_1"})
    {
        SCOPED_TRACE(dtype);
        const auto type=CPUTensorStorageDeclaration::requireDtype(dtype);
        const CPUTensorStorageDeclaration declaration(2,33,type,PhysicalMemoryOwner::ActivationArena,cpu1);
        auto memory=arenaAuthority(declaration.physicalBytes(),cpu1);
        std::shared_ptr<TensorBase> retained;
        std::weak_ptr<TensorBase> weak;
        {
            BufferArena arena({.physical_memory_authority=memory});
            ASSERT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE,2,33,dtype,cpu1));
            ASSERT_TRUE(arena.allocate());
            retained=arena.getSharedTensor(BufferId::HIDDEN_STATE);
            ASSERT_NE(retained,nullptr);
            EXPECT_EQ(retained->home_device(),cpu1);
            EXPECT_EQ(retained->native_type(),type==BufferTensorType::FP32?TensorType::FP32:
                type==BufferTensorType::FP16?TensorType::FP16:type==BufferTensorType::BF16?TensorType::BF16:
                type==BufferTensorType::INT32?TensorType::INT32:type==BufferTensorType::Q8_1?TensorType::Q8_1:TensorType::Q16_1);
            const auto *raw=static_cast<const uint8_t *>(retained->raw_data());
            ASSERT_NE(raw,nullptr);
            for(size_t i=0;i<retained->size_bytes();++i) EXPECT_EQ(raw[i],0u);
            EXPECT_EQ(memory->claimedBytes(cpu1,PhysicalMemoryOwner::ActivationArena,
                PhysicalMemoryMaterializationKind::NewAllocation),declaration.physicalBytes());
            weak=retained;
        }
        EXPECT_FALSE(weak.expired());
        retained.reset();
        EXPECT_TRUE(weak.expired());
        EXPECT_EQ(memory->claimedBytes(cpu1,PhysicalMemoryOwner::ActivationArena,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }
}
/** @test Actual backing retirement ends the claim while tensor control ownership survives. */
TEST(Test__BufferArenaPhysicalMemory, NativeBackingRetirementEndsLeaseBeforeTensorDestruction)
{
    for(const auto dtype : {"FP32","FP16","BF16","INT32","Q8_1","Q16_1"})
    {
        SCOPED_TRACE(dtype);
        const CPUTensorStorageDeclaration declaration(2,33,CPUTensorStorageDeclaration::requireDtype(dtype),
            PhysicalMemoryOwner::ActivationArena,DeviceId::cpu());
        auto memory=arenaAuthority(declaration.physicalBytes());
        BufferArena arena({.physical_memory_authority=memory});
        ASSERT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE,2,33,dtype,DeviceId::cpu()));
        ASSERT_TRUE(arena.allocate());
        auto retained=arena.getSharedTensor(BufferId::HIDDEN_STATE);
        retained->release_raw_data();
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ActivationArena,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }
}
/** @test Unsupported dtype never becomes an anonymous FP32 root. */
TEST(Test__BufferArenaPhysicalMemory, NativeDeclarationRejectsUnknownDtypeAndExactBudgetShortage)
{
    EXPECT_THROW(CPUTensorStorageDeclaration::requireDtype("Q8_0"),std::invalid_argument);
    for(const auto dtype : {"FP32","FP16","BF16","INT32","Q8_1","Q16_1"})
    {
        const CPUTensorStorageDeclaration declaration(2,33,CPUTensorStorageDeclaration::requireDtype(dtype),
            PhysicalMemoryOwner::ActivationArena,DeviceId::cpu());
        auto memory=arenaAuthority(declaration.physicalBytes()-1);
        BufferArena arena({.physical_memory_authority=memory});
        ASSERT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE,2,33,dtype,DeviceId::cpu()));
        EXPECT_THROW(arena.allocate(),std::exception);
        EXPECT_EQ(arena.getSharedTensor(BufferId::HIDDEN_STATE),nullptr);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ActivationArena,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }
    auto memory=arenaAuthority(4096);
    BufferArena arena({.physical_memory_authority=memory});
    ASSERT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE,2,33,"unrecognized",DeviceId::cpu()));
    EXPECT_THROW(arena.allocate(),std::invalid_argument);
}

/** @test Planning and registration consume the same flattened resolved descriptor. */
TEST(Test__BufferArenaPhysicalMemory, DescriptorDeclarationSharesExactFlatteningContract)
{
    for(const auto shape : {std::vector<size_t>{7},std::vector<size_t>{7,3,5}})
    {
        auto descriptor=BufferDescriptor::scratch("typed_cpu_root",shape,BufferTensorType::Q8_1);
        descriptor.device=DeviceId{DeviceType::CPU,1};
        const auto declaration=CPUTensorStorageDeclaration::fromDescriptor(descriptor,PhysicalMemoryOwner::ActivationArena);
        EXPECT_EQ(declaration.rows(),7u);
        EXPECT_EQ(declaration.columns(),shape.size()==1?1u:15u);
        auto memory=arenaAuthority(declaration.physicalBytes(),descriptor.device);
        BufferArena arena({.physical_memory_authority=memory});
        ASSERT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE,descriptor));
        ASSERT_TRUE(arena.allocate());
        EXPECT_EQ(arena.getRows(BufferId::HIDDEN_STATE),declaration.rows());
        EXPECT_EQ(arena.getCols(BufferId::HIDDEN_STATE),declaration.columns());
        EXPECT_EQ(memory->claimedBytes(descriptor.device,PhysicalMemoryOwner::ActivationArena,
            PhysicalMemoryMaterializationKind::NewAllocation),declaration.physicalBytes());
    }
}
