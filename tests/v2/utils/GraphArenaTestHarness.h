/**
 * @file GraphArenaTestHarness.h
 * @brief Stable BufferArena and tensor ownership for graph integration tests.
 *
 * GPU graph tests must model the same address-lifetime and coherence rules as
 * production. This helper owns one BufferArena plus externally bound tensors,
 * binds the arena to DeviceGraphExecutor, and rejects duplicate BufferId
 * registration. It deliberately does not upload tensors or synchronize a
 * device: the executor and stage contracts remain responsible for stream-
 * ordered preparation exactly as they are during inference.
 */

#pragma once

#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "memory/BufferArena.h"
#include "tensors/ITensor.h"

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace llaminar2::test
{
    /**
     * @brief Composable arena fixture for CPU-only and GPU integration tests.
     *
     * The helper is intentionally independent of `testing::Test` so specialized
     * fixtures can compose it without creating another inheritance hierarchy.
     * Declare it before executors/engines/hosts in a fixture so reverse member
     * destruction keeps its arena and tensors alive until every graph cache has
     * released its stage pointers.
     */
    class GraphArenaTestHarness final
    {
    public:
        GraphArenaTestHarness() = default;
        GraphArenaTestHarness(const GraphArenaTestHarness &) = delete;
        GraphArenaTestHarness &operator=(const GraphArenaTestHarness &) = delete;

        /**
         * @brief Bind this stable arena to an executor before its first graph.
         *
         * @param executor Executor whose stage contracts will resolve BufferIds
         *        through this harness.
         */
        void bindExecutor(DeviceGraphExecutor &executor)
        {
            executor.setArena(&arena_);
        }

        /**
         * @brief Construct, own, and bind one persistent tensor.
         *
         * Every BufferId may be established exactly once. Tests that need
         * multiple graph geometries should allocate the largest required shape
         * and vary active dimensions in stage parameters, matching production
         * graph-cache address stability.
         *
         * @tparam TensorT Concrete ITensor implementation.
         * @tparam Args Constructor argument types for TensorT.
         * @param id Semantic arena slot for the tensor.
         * @param args Arguments forwarded to TensorT's constructor.
         * @return Stable typed pointer owned by this harness.
         * @throws std::logic_error when the BufferId is already bound.
         * @throws std::runtime_error when BufferArena rejects registration.
         */
        template <typename TensorT, typename... Args>
        TensorT *createPersistentTensor(BufferId id, Args &&...args)
        {
            static_assert(
                std::is_base_of_v<ITensor, TensorT>,
                "GraphArenaTestHarness tensors must implement ITensor");

            const size_t index = static_cast<size_t>(id);
            if (index >= tensors_.size())
                throw std::out_of_range("GraphArenaTestHarness received an invalid BufferId");
            if (tensors_[index] || arena_.isRegistered(id))
                throw std::logic_error(
                    std::string("GraphArenaTestHarness duplicate BufferId: ") +
                    bufferIdName(id));

            auto tensor = std::make_unique<TensorT>(
                std::forward<Args>(args)...);
            TensorT *const result = tensor.get();
            if (!arena_.registerExternalBuffer(id, result))
            {
                throw std::runtime_error(
                    std::string("GraphArenaTestHarness failed to bind BufferId: ") +
                    bufferIdName(id));
            }
            tensors_[index] = std::move(tensor);
            return result;
        }

        /// @brief Access the arena for assertions or explicit host-boundary publication.
        BufferArena &arena() { return arena_; }
        const BufferArena &arena() const { return arena_; }

        /// @brief Return the tensor currently bound to a semantic arena slot.
        ITensor *tensor(BufferId id) const
        {
            return arena_.getTensor(id);
        }

    private:
        static constexpr size_t kBufferCount =
            static_cast<size_t>(BufferId::_COUNT);

        BufferArena arena_;
        std::array<std::unique_ptr<ITensor>, kBufferCount> tensors_{};
    };

} // namespace llaminar2::test
