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
 */

#include <gtest/gtest.h>
#include "memory/BufferArena.h"
#include "memory/BufferId.h"
#include "memory/BufferAccess.h"
#include "memory/StageBufferContract.h"
#include "memory/StageBoundBuffers.h"
#include "memory/CoherenceTracker.h"
#include "tensors/TensorClasses.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

// ============================================================================
// Registration Tests
// ============================================================================

TEST(Test__BufferArena, RegisterBufferSucceeds)
{
    BufferArena arena;
    EXPECT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu()));
    EXPECT_TRUE(arena.isRegistered(BufferId::HIDDEN_STATE));
    EXPECT_EQ(arena.registeredCount(), 1u);
}

TEST(Test__BufferArena, RegisterBufferRejectsDoubleRegistration)
{
    BufferArena arena;
    EXPECT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu()));
    EXPECT_FALSE(arena.registerBuffer(BufferId::HIDDEN_STATE, 8, 896, "FP32", DeviceId::cpu()));
    EXPECT_EQ(arena.registeredCount(), 1u);
}

TEST(Test__BufferArena, RegisterMultipleBuffers)
{
    BufferArena arena;
    EXPECT_TRUE(arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu()));
    EXPECT_TRUE(arena.registerBuffer(BufferId::Q_PROJ, 4, 896, "FP32", DeviceId::cpu()));
    EXPECT_TRUE(arena.registerBuffer(BufferId::K_PROJ, 4, 128, "FP32", DeviceId::cpu()));
    EXPECT_EQ(arena.registeredCount(), 3u);
}

TEST(Test__BufferArena, RegisterExternalBufferSucceeds)
{
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 896}, DeviceId::cpu());
    BufferArena arena;
    EXPECT_TRUE(arena.registerExternalBuffer(BufferId::LOGITS, tensor.get()));
    EXPECT_TRUE(arena.isRegistered(BufferId::LOGITS));
}

TEST(Test__BufferArena, UnregisteredBufferReturnsNullTensor)
{
    BufferArena arena;
    EXPECT_FALSE(arena.isRegistered(BufferId::HIDDEN_STATE));
    EXPECT_EQ(arena.getTensor(BufferId::HIDDEN_STATE), nullptr);
}

// ============================================================================
// Allocation Tests
// ============================================================================

TEST(Test__BufferArena, AllocateCreatesOwnedTensors)
{
    BufferArena arena;
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
    BufferArena arena;
    arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu());
    EXPECT_TRUE(arena.allocate());
    EXPECT_FALSE(arena.allocate()); // Second call should fail
}

TEST(Test__BufferArena, ExternalBufferNotAllocatedByArena)
{
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 896}, DeviceId::cpu());
    BufferArena arena;
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
    BufferArena arena;
    arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu());
    arena.allocate();

    auto state = arena.getCoherenceState(BufferId::HIDDEN_STATE);
    EXPECT_EQ(state.authority, CoherenceState::UNINITIALIZED);
}

TEST(Test__BufferArena, MarkWrittenSetsHostAuthority)
{
    BufferArena arena;
    arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu());
    arena.allocate();

    arena.markWritten(BufferId::HIDDEN_STATE, DeviceId::cpu());

    auto state = arena.getCoherenceState(BufferId::HIDDEN_STATE);
    EXPECT_EQ(state.authority, CoherenceState::HOST);
}

TEST(Test__BufferArena, PrepareForWriteThenMarkWritten)
{
    BufferArena arena;
    arena.registerBuffer(BufferId::FFN_OUTPUT, 4, 896, "FP32", DeviceId::cpu());
    arena.allocate();

    EXPECT_TRUE(arena.prepareForWrite(BufferId::FFN_OUTPUT, DeviceId::cpu()));
    arena.markWritten(BufferId::FFN_OUTPUT, DeviceId::cpu());

    auto state = arena.getCoherenceState(BufferId::FFN_OUTPUT);
    EXPECT_EQ(state.authority, CoherenceState::HOST);
}

TEST(Test__BufferArena, PrepareForReadOnCPUAfterCPUWrite)
{
    BufferArena arena;
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
    BufferArena arena;
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
    BufferArena arena;
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
    BufferArena arena;
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
    BufferArena arena;
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
    BufferArena arena;
    arena.registerBuffer(BufferId::Q_PROJ, 4, 896, "FP32", DeviceId::cpu());
    arena.allocate();

    arena.acquireReadBorrow(BufferId::Q_PROJ);
    EXPECT_DEATH(arena.acquireWriteBorrow(BufferId::Q_PROJ), "active read borrow");
    arena.releaseReadBorrow(BufferId::Q_PROJ);
}

TEST(Test__BufferArena, ReadBorrowConflictsWithWriteBorrow)
{
    BufferArena arena;
    arena.registerBuffer(BufferId::Q_PROJ, 4, 896, "FP32", DeviceId::cpu());
    arena.allocate();

    arena.acquireWriteBorrow(BufferId::Q_PROJ);
    EXPECT_DEATH(arena.acquireReadBorrow(BufferId::Q_PROJ), "active write borrow");
    arena.releaseWriteBorrow(BufferId::Q_PROJ);
}

TEST(Test__BufferArena, DoubleWriteBorrowDies)
{
    BufferArena arena;
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
    BufferArena arena;
    arena.registerBuffer(BufferId::ATTN_OUTPUT, 4, 896, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::GATE_PROJ, 4, 4864, "FP32", DeviceId::cpu());
    arena.registerAlias(BufferId::ATTN_OUTPUT, BufferId::GATE_PROJ);

    arena.allocate();

    // Both should be registered and usable
    EXPECT_NE(arena.getTensor(BufferId::ATTN_OUTPUT), nullptr);
    EXPECT_NE(arena.getTensor(BufferId::GATE_PROJ), nullptr);
}

TEST(Test__BufferArena, AliasingSerialBorrowsSucceed)
{
    BufferArena arena;
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
    BufferArena arena;
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
    BufferArena arena;
    arena.registerBuffer(BufferId::ATTN_OUTPUT, 4, 896, "FP32", DeviceId::cpu());
    arena.registerBuffer(BufferId::GATE_PROJ, 4, 4864, "FP32", DeviceId::cpu());
    arena.registerAlias(BufferId::ATTN_OUTPUT, BufferId::GATE_PROJ);
    arena.allocate();

    arena.acquireReadBorrow(BufferId::ATTN_OUTPUT);
    EXPECT_DEATH(arena.acquireWriteBorrow(BufferId::GATE_PROJ), "aliased buffer");
    arena.releaseReadBorrow(BufferId::ATTN_OUTPUT);
}

#endif // NDEBUG

// ============================================================================
// Dimensions & DevicePtr Tests
// ============================================================================

TEST(Test__BufferArena, RowsColsMatchRegistration)
{
    BufferArena arena;
    arena.registerBuffer(BufferId::HIDDEN_STATE, 4, 896, "FP32", DeviceId::cpu());
    arena.allocate();

    EXPECT_EQ(arena.getRows(BufferId::HIDDEN_STATE), 4u);
    EXPECT_EQ(arena.getCols(BufferId::HIDDEN_STATE), 896u);
}

TEST(Test__BufferArena, DevicePtrReturnsCPUPointer)
{
    BufferArena arena;
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
}

TEST(Test__BufferId, CountIsReasonable)
{
    auto count = static_cast<size_t>(BufferId::_COUNT);
    EXPECT_GT(count, 10u);
    EXPECT_LT(count, 100u);
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
    BufferArena arena;
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
    EXPECT_TRUE(arena.prepareForWrite(BufferId::Q_PROJ, cpu));

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

    BufferArena arena;
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
        EXPECT_TRUE(arena.prepareForWrite(binding.id, cpu));
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
    BufferArena arena;
    DeviceId cpu = DeviceId::cpu();

    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 4}, DeviceId::cpu());
    ASSERT_TRUE(arena.registerExternalBuffer(BufferId::FFN_OUTPUT, tensor.get()));

    // No stream (CPU) — markWrittenFlagsOnly should work
    arena.markWrittenFlagsOnly(BufferId::FFN_OUTPUT, cpu);

    auto state = arena.getCoherenceState(BufferId::FFN_OUTPUT);
    EXPECT_EQ(state.authority, CoherenceState::HOST);
}
