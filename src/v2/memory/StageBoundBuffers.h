/**
 * @file StageBoundBuffers.h
 * @brief Device-ready buffer delivery for stage execution
 *
 * StageBoundBuffers is an immutable collection of BufferView handles
 * bound to a stage for one execution. The GraphExecutor builds it from
 * BufferArena borrows and passes it to stage->execute().
 *
 * Stages use it to retrieve typed, access-controlled views:
 *   auto inp = buffers.input<float>(BufferId::HIDDEN_STATE);
 *   auto out = buffers.output<float>(BufferId::Q_PROJ);
 */

#pragma once

#include <cstddef>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include "BufferId.h"
#include "BufferAccess.h"
#include "backends/DeviceId.h"
#include "tensors/ITensor.h"

namespace llaminar2
{

    /**
     * @brief Immutable set of buffer views for a single stage execution.
     *
     * Only constructible by the GraphExecutor (friend).
     */
    class StageBoundBuffers
    {
    public:
        /// Entry stored per buffer binding
        struct Entry
        {
            BufferId id;
            BufferAccess access;
            ITensor *tensor = nullptr;  ///< For kernel dispatch
            void *device_ptr = nullptr; ///< Already on target device
            size_t rows = 0;
            size_t cols = 0;
            DeviceId device;
        };

        StageBoundBuffers() = default;

        // ── Typed accessors (compile-time access control) ───────────────────

        /// Get a read-only view for an input buffer
        template <typename T>
        BufferView<T, BufferAccess::READ> input(BufferId id) const
        {
            return getView<T, BufferAccess::READ>(id, BufferAccess::READ);
        }

        /// Get a write-only view for an output buffer
        template <typename T>
        BufferView<T, BufferAccess::WRITE> output(BufferId id) const
        {
            return getView<T, BufferAccess::WRITE>(id, BufferAccess::WRITE);
        }

        /// Get a read-only view for a weight buffer
        template <typename T>
        BufferView<T, BufferAccess::READ> weight(BufferId id) const
        {
            return getView<T, BufferAccess::READ>(id, BufferAccess::READ);
        }

        /// Get a read-write view for an in-place buffer
        template <typename T>
        BufferView<T, BufferAccess::READWRITE> inout(BufferId id) const
        {
            return getView<T, BufferAccess::READWRITE>(id, BufferAccess::READWRITE);
        }

        /// Get underlying ITensor* for kernel dispatch
        ITensor *weightTensor(BufferId id) const
        {
            auto it = entries_.find(static_cast<uint32_t>(id));
            if (it == entries_.end())
            {
                throw std::runtime_error(std::string("StageBoundBuffers: no entry for ") + bufferIdName(id));
            }
            return it->second.tensor;
        }

        /// Get raw workspace pointer by name
        void *workspace(const char *name) const
        {
            if (!name)
                return nullptr;
            auto it = workspaces_.find(std::string(name));
            if (it == workspaces_.end())
                return nullptr;
            return it->second;
        }

        // ── Introspection ───────────────────────────────────────────────────

        /// Number of bound entries
        size_t size() const { return entries_.size(); }

        /// Check if a buffer is bound
        bool has(BufferId id) const { return entries_.count(static_cast<uint32_t>(id)) > 0; }

        // ── Builder (used by GraphExecutor / BufferArena) ───────────────────

        void addEntry(const Entry &entry)
        {
            entries_[static_cast<uint32_t>(entry.id)] = entry;
        }

        void addWorkspace(const char *name, void *ptr)
        {
            if (name)
                workspaces_[std::string(name)] = ptr;
        }

    private:
        template <typename T, BufferAccess Access>
        BufferView<T, Access> getView(BufferId id, BufferAccess /*requested_access*/) const
        {
            auto it = entries_.find(static_cast<uint32_t>(id));
            if (it == entries_.end())
            {
                throw std::runtime_error(std::string("StageBoundBuffers: buffer not bound: ") + bufferIdName(id));
            }
            const Entry &e = it->second;
            return BufferView<T, Access>(e.device_ptr, e.tensor, e.rows, e.cols, e.device);
        }

        std::unordered_map<uint32_t, Entry> entries_;
        std::unordered_map<std::string, void *> workspaces_;
    };

} // namespace llaminar2
