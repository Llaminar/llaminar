/**
 * @file ROCmEmbeddingExecutionBinding.h
 * @brief Checked borrowed bindings for one immutable ROCm embedding participant.
 *
 * KernelFactory already separates embedding implementations by device. This
 * object therefore retains one exact owner, not an ambient-device map or a
 * last-device substitute. Binding authenticates before committing; getters
 * perform no runtime work. The workspace type parameter permits device-free
 * replay of this actual state machine without constructing a GPU workspace.
 * The surrounding kernel's existing serialized submission owns mutation.
 */
#pragma once
#include "backends/DeviceId.h"
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

namespace llaminar2
{
    /** @brief One participant's non-owning stream/workspace execution bindings. */
    template <typename Workspace>
    class ROCmEmbeddingExecutionBinding final
    {
    public:
        /** @brief Require an explicit native ordinal; no current-device/default-zero mode exists.
         * @param ordinal Exact backend-local nonnegative participant.
         * @throws std::invalid_argument When no explicit participant is supplied. */
        explicit ROCmEmbeddingExecutionBinding(int ordinal) : ordinal_(ordinal)
        {
            if (ordinal < 0)
                throw std::invalid_argument("ROCm embedding requires an explicit nonnegative device owner");
        }
        /** @return Immutable owner; independent of the calling thread's current device. */
        int ordinal() const noexcept { return ordinal_; }
        /** @brief Resolve the API's omitted ordinal against this instance, rejecting foreign overrides.
         * @param requested Exact owner or -1 to use this already explicit instance.
         * @return Immutable participant ordinal.
         * @throws std::invalid_argument For a foreign or malformed override. */
        int resolveOrdinal(int requested) const
        {
            if (requested != -1 && requested != ordinal_)
                throw std::invalid_argument("ROCm embedding launch device differs from its immutable owner");
            return ordinal_;
        }
        /**
         * @brief Authenticate a non-null stream before replacing a valid binding.
         * @param stream Exact borrowed native stream.
         * @param query Checked native stream-owner adapter, or recorded replay adapter.
         * @tparam Query Callable returning native integer status and writing the observed ordinal.
         * @throws On a failed observation or foreign owner, retaining the old binding.
         */
        template <typename Query>
        void bindStream(void *stream, Query &&query)
        {
            if (!stream) throw std::invalid_argument("ROCm embedding requires a non-null stream");
            int observed = -1;
            const int status = static_cast<int>(std::invoke(std::forward<Query>(query), stream, &observed));
            if (status != 0)
                throw std::runtime_error("hipStreamGetDevice failed before ROCm embedding binding: status=" +
                    std::to_string(status));
            if (observed != ordinal_)
                throw std::invalid_argument("ROCm embedding stream belongs to a foreign device");
            stream_ = stream;
        }
        /** @brief Clear a borrowed stream without changing owner, workspace or native topology. */
        void clearStream() noexcept { stream_ = nullptr; }
        /** @return Checked borrowed stream, without any device or hardware query. */
        void *stream() const
        {
            if (!stream_) throw std::runtime_error("ROCmEmbeddingKernelT requires an explicit non-null GPU stream binding before launch");
            return stream_;
        }
        /**
         * @brief Bind or explicitly clear the exact owner's borrowed workspace.
         * @param workspace Borrowed owner, or null to clear.
         * @param device Workspace's declared device; ignored only for explicit clear.
         * @throws Before modifying the prior binding if a non-null workspace is foreign.
         */
        void bindWorkspace(Workspace *workspace, DeviceId device)
        {
            if (workspace && device != DeviceId::rocm(ordinal_))
                throw std::invalid_argument("ROCm embedding workspace belongs to a foreign device");
            workspace_ = workspace;
        }
        /** @return Borrowed exact-owner workspace, or absence; never another device's last binding. */
        Workspace *workspace() const noexcept { return workspace_; }
    private:
        const int ordinal_; ///< Device identity lives as long as the factory-owned kernel.
        void *stream_ = nullptr; ///< Exact producer stream, owned by the execution context.
        Workspace *workspace_ = nullptr; ///< Workspace lifetime remains with the graph/arena owner.
    };
}
