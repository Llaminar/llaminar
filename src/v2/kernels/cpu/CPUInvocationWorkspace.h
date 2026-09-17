/**
 * @file CPUInvocationWorkspace.h
 * @brief Participant-owned scratch for immutable, concurrently callable CPU kernels.
 *
 * A prepared kernel may be shared, but a scratch address may not. The stage
 * retains its own DeviceWorkspaceManager and supplies it to each invocation.
 * This boundary only borrows already admitted storage: it cannot allocate,
 * resize, retain a thread-local buffer or substitute another participant's
 * workspace. PhysicalMemoryAuthority remains the allocation ledger.
 */
#pragma once
#include "interfaces/IWorkspaceConsumer.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "tensors/BlockStructures.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include <cstdint>
#include <algorithm>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    /** @brief Stable buffer ABI for the CPU down projection's input transform. */
    // Materialize metadata once, not a long temporary std::string per lookup.
    inline const std::string kCPUSwiGLUInput = "cpu_swiglu_input";
    /** @brief CPU-native activation blocks, shared only by serial projections. */
    inline const std::string kCPUProjectionQ8 = "cpu_projection_q8";
    /** @brief Rotated FP32 inputs coexist with the unmodified caller's input. */
    inline const std::string kCPUProjectionRotation = "cpu_projection_rotation";
    /** @brief Unscaled product retained while the beta epilogue reads prior output. */
    inline const std::string kCPUProjectionProduct = "cpu_projection_product";
    /** @brief Shared producer/reducer bank for ordered CPU K partitions. */
    inline const std::string kCPUProjectionPartials = "cpu_projection_partials";

    /**
     * @brief Physical FP32 partial extent, including each masked 64-column tail.
     * @param rows Simultaneously live output rows.
     * @param columns Physical output width, not a mirrored head's policy width.
     * @param partitions Arithmetic partition count selected by the serial policy.
     * @return Zero for a full-K route; otherwise its exact producer/reducer bank.
     */
    inline size_t cpuProjectionPartialFloats(int rows, int columns, int partitions)
    {
        if (rows <= 0 || columns <= 0 || partitions < 0)
            throw std::invalid_argument("CPU partial workspace requires valid geometry");
        if (partitions <= 1) return 0;
        const size_t chunks = (static_cast<size_t>(columns) + 63) / 64;
        const size_t row_floats = chunks * static_cast<size_t>(partitions) * 64;
        if (static_cast<size_t>(rows) > std::numeric_limits<size_t>::max() / sizeof(float) / row_floats)
            throw std::overflow_error("CPU partial workspace extent overflow");
        return static_cast<size_t>(rows) * row_floats;
    }

    /** @return The named partial bank; full-K projections declare no allocation. */
    inline WorkspaceRequirements cpuProjectionPartialWorkspaceRequirements(size_t floats)
    {
        if (floats > std::numeric_limits<size_t>::max() / sizeof(float))
            throw std::overflow_error("CPU partial workspace byte extent overflow");
        WorkspaceRequirements requirements;
        if (floats) requirements.buffers.push_back({kCPUProjectionPartials, floats * sizeof(float), 64});
        return requirements;
    }

    /**
     * @brief Named slab envelope for serial projections and fused worker grids.
     * @param rows Maximum live rows for one projection, including retained smaller calls.
     * @param columns Physical output width, including its masked 64-column tail.
     * @param partitions This member's serial arithmetic tree; never the anchor's codebook.
     * @param workers Admitted workshare width, not the number of model experts.
     * @param row_tile Maximum physical rows one selected microkernel can own (1..4).
     * @return Checked FP32 capacity contributed by this member; merge members by maximum.
     *
     * Mixed full-K/partitioned bundles retire each partial reader at their existing
     * reducer barrier, so ordinary projection capacity covers their reused bank.
     * An all-partitioned bundle uses shared producers only while its output grid
     * has fewer than workers tiles: (workers-1)*row_tile*partitions*64 bounds the
     * simultaneous bank, independently of expert count. Larger grids use private
     * output owners. Trees beyond the stack contract get one exclusive slab tile
     * per worker instead of an unbounded stack or a changed reduction tree.
     */
    inline size_t cpuProjectionPartialEnvelopeFloats(
        int rows, int columns, int partitions, int workers, int row_tile)
    {
        const size_t serial = cpuProjectionPartialFloats(rows, columns, partitions);
        if (workers <= 0 || row_tile <= 0 || row_tile > 4)
            throw std::invalid_argument("CPU partial envelope requires a valid worker/microkernel grid");
        if (partitions <= 1) return 0;
        const size_t tile = cpuProjectionPartialFloats(std::min(rows, row_tile), 64, partitions);
        const size_t slots = partitions > MoEProjectionNumericalContract::ordered_k_partitions
            ? static_cast<size_t>(workers) : static_cast<size_t>(workers - 1);
        if (slots > std::numeric_limits<size_t>::max() / sizeof(float) / tile)
            throw std::overflow_error("CPU fused worker slab extent overflow");
        return std::max(serial, slots * tile);
    }

    /** @return Checked complete Q8 block count, shared by declaration and invocation. */
    inline size_t cpuProjectionQ8Blocks(int rows, int columns)
    {
        if (rows <= 0 || columns <= 0)
            throw std::invalid_argument("CPU Q8 workspace requires positive geometry");
        const size_t blocks = (static_cast<size_t>(columns) + Q8_1Block::BLOCK_SIZE - 1) /
            Q8_1Block::BLOCK_SIZE;
        if (static_cast<size_t>(rows) > std::numeric_limits<size_t>::max() / sizeof(Q8_1Block) / blocks)
            throw std::overflow_error("CPU Q8 workspace extent overflow");
        return static_cast<size_t>(rows) * blocks;
    }

    /**
     * @brief Exact Q8 input bank; codebooks share the same activation ABI.
     * @param rows Simultaneously live input rows, including distinct expert inputs.
     * @param columns Logical K width; the final block includes a masked tail.
     * @return One aligned bank, merged by maximum across serial invocations.
     */
    inline WorkspaceRequirements cpuProjectionQ8WorkspaceRequirements(int rows, int columns)
    {
        WorkspaceRequirements result;
        result.buffers.push_back({kCPUProjectionQ8, cpuProjectionQ8Blocks(rows, columns) * sizeof(Q8_1Block), 64});
        return result;
    }

    /**
     * @brief Exact FP32 tile shared by optional CPU projection operations.
     * @param name Stable named-buffer identity for one independent lifetime.
     * @param rows Maximum simultaneous activation rows.
     * @param columns Logical FP32 elements per row.
     * @return One aligned bank; only serial operations may merge the same name.
     */
    inline WorkspaceRequirements cpuProjectionFP32WorkspaceRequirements(
        const std::string &name, int rows, int columns)
    {
        if (rows <= 0 || columns <= 0)
            throw std::invalid_argument("CPU FP32 workspace requires positive geometry");
        const size_t width = static_cast<size_t>(columns) * sizeof(float);
        if (static_cast<size_t>(rows) > std::numeric_limits<size_t>::max() / width)
            throw std::overflow_error("CPU FP32 workspace extent overflow");
        WorkspaceRequirements result;
        result.buffers.push_back({name, static_cast<size_t>(rows) * width, 64});
        return result;
    }

    /** @return Exact down-projection input transform; independent of its output. */
    inline WorkspaceRequirements cpuSwiGLUWorkspaceRequirements(int rows, int columns)
    {
        return cpuProjectionFP32WorkspaceRequirements(kCPUSwiGLUInput, rows, columns);
    }

    /** @brief Workspace declaration without mutable bindings on a shared CPU engine. */
    class CPUInvocationWorkspaceConsumer : public IWorkspaceConsumer
    {
    public:
        /** @return No ordinary-projection scratch declared by this ownership adapter. */
        WorkspaceRequirements getWorkspaceRequirements(int, int = 0, int = 0) const override { return {}; }
        /** @brief Add only the input transform selected by the owning stage. */
        void appendSwiGLUWorkspaceRequirements(WorkspaceRequirements &requirements, int m, int k) const override
        {
            requirements.merge(cpuSwiGLUWorkspaceRequirements(m, k));
        }
        /** @return The stage, not this prepared engine, owns the active scratch binding. */
        WorkspaceBindingPolicy workspaceBindingPolicy() const noexcept final
        {
            return WorkspaceBindingPolicy::Invocation;
        }
        /**
         * @brief Stage preparation cannot install a shared mutable kernel binding.
         * @param workspace Stage-owned storage; supplied again by the invocation.
         *
         * The existing stage adapter retains this pointer itself. This method
         * deliberately stores nothing; a bare kernel invocation without its
         * workspace argument still fails when it requests scratch.
         */
        void bindWorkspace(DeviceWorkspaceManager *workspace) final { (void)workspace; }
        /** @brief No engine-owned pointer survives an invocation or needs clearing. */
        void unbindWorkspace() final {}
        /** @return False: this engine never owns a persistent workspace binding. */
        bool hasWorkspace() const final { return false; }
        /** @return No implicit workspace; callers must supply their participant's arena. */
        DeviceWorkspaceManager *getWorkspace() const final { return nullptr; }
    };

    /** @brief Checked CPU scratch access shared by ordinary and fused projection paths. */
    class CPUInvocationWorkspace final
    {
    public:
        /** @return Checked payload product for both declaration and runtime validation. */
        static size_t multiply(size_t count, size_t width)
        {
            if (width && count > std::numeric_limits<size_t>::max() / width)
                throw std::overflow_error("CPU invocation workspace extent overflow");
            return count * width;
        }

        /**
         * @brief Borrow a declared bank whose exact used prefix is resolved by a fused plan.
         * @return Empty when no bank was declared; execution must still validate its demand.
         *
         * This never allocates or invents capacity. A full-K/private-stack call
         * legitimately requires no bank. Any positive demand is checked against
         * this span before the fused engine launches its first producer.
         */
        template<class T>
        static std::span<T> available(DeviceWorkspaceManager *workspace, const std::string &name)
        {
            if (!workspace) return {};
            if (!workspace->device().is_cpu())
                throw std::runtime_error("CPU invocation cannot borrow a GPU workspace");
            const size_t bytes = workspace->getBufferSize(name);
            if (bytes % sizeof(T))
                throw std::runtime_error("CPU invocation workspace has an incomplete typed element");
            return require<T>(workspace, name, bytes / sizeof(T));
        }

        /**
         * @brief Borrow an exact typed prefix of a preallocated CPU workspace buffer.
         * @param workspace Invocation-owned CPU allocator; null is never an allocation request.
         * @param name Stable buffer ABI name shared with the kernel's declaration.
         * @param elements Required live elements; zero requires no buffer.
         * @return Non-owning span valid until the enclosing stage invocation completes.
         * @throws std::runtime_error on missing, foreign-device, undersized or misaligned storage.
         */
        template<class T>
        static std::span<T> require(DeviceWorkspaceManager *workspace, const std::string &name, size_t elements)
        {
            if (!elements) return {};
            const size_t bytes = multiply(elements, sizeof(T));
            if (!workspace || !workspace->device().is_cpu() || workspace->getBufferSize(name) < bytes)
                throw std::runtime_error("CPU invocation requires admitted workspace buffer: " + name);
            auto *address = workspace->getBuffer(name);
            if (!address || reinterpret_cast<uintptr_t>(address) % alignof(T))
                throw std::runtime_error("CPU invocation workspace address is absent or misaligned: " + name);
            return {static_cast<T *>(address), elements};
        }
    };
}
