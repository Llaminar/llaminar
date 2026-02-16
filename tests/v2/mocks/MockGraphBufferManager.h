/**
 * @file MockDeviceGraphBufferManager.h
 * @brief Mock graph buffer manager for unit testing without actual tensor allocation
 *
 * This mock enables:
 * - Testing graph execution logic without actual memory allocation
 * - Testing buffer binding and retrieval in isolation
 * - Simulating buffer aliasing scenarios
 * - Call tracking for verification
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "execution/local_execution/graph/IGraphBufferManager.h"
#include "execution/debug/BufferRole.h"
#include "execution/local_execution/graph/LivenessAnalyzer.h"       // For AliasingGroup
#include "execution/local_execution/graph/DeviceGraphExecutor.h"          // For ComputeGraph
#include "execution/local_execution/collective/CollectiveContext.h" // For CollectiveContext (Phase 3)
#include "tensors/Tensors.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <stdexcept>
#include <functional>

namespace llaminar2::test
{

    // Forward declaration
    class MockGraphBufferManagerBuilder;

    /**
     * @brief Mock graph buffer manager for unit testing
     *
     * Provides configurable buffer management for testing:
     * - Pre-registered mock buffers
     * - Configurable allocation behavior
     * - Call tracking for verification
     * - Failure injection for robustness testing
     *
     * Usage:
     * @code
     * // Simple usage with builder
     * auto output = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 896});
     * auto mock = MockGraphBufferManagerBuilder()
     *     .addBuffer("layer0_attn", "output", output.get())
     *     .setAllocateForGraphSucceeds(true)
     *     .build();
     *
     * auto* buffer = mock->getBuffer("layer0_attn", "output");
     * EXPECT_EQ(mock->getBuffer_call_count(), 1);
     * @endcode
     */
    class MockGraphBufferManager : public IGraphBufferManager
    {
    public:
        // =========================================================================
        // Configuration
        // =========================================================================

        /**
         * @brief Configuration options for the mock
         */
        struct Config
        {
            bool track_calls = true;                     ///< Record call counts for verification
            bool allocate_for_graph_succeeds = true;     ///< Return value for allocateForGraph()
            bool allocate_buffer_succeeds = true;        ///< Return value for allocateBuffer()
            bool allocate_with_aliasing_succeeds = true; ///< Return value for allocateWithAliasing()
            bool bind_buffer_succeeds = true;            ///< Return value for bindBuffer()
            double aliasing_savings_percent = 0.0;       ///< Simulated aliasing savings
            size_t aliasing_group_count = 0;             ///< Simulated aliasing groups
        };

        // =========================================================================
        // Presets for Common Configurations
        // =========================================================================

        /**
         * @brief Create a minimal mock with no buffers
         *
         * Suitable for testing code paths that don't use buffers.
         */
        static std::shared_ptr<MockGraphBufferManager> createEmpty();

        /**
         * @brief Create a mock that always succeeds
         *
         * All allocation methods return true, no buffers pre-registered.
         */
        static std::shared_ptr<MockGraphBufferManager> createSuccessful();

        /**
         * @brief Create a mock that always fails allocations
         *
         * All allocation methods return false.
         */
        static std::shared_ptr<MockGraphBufferManager> createFailing();

        // =========================================================================
        // Construction
        // =========================================================================

        MockGraphBufferManager();
        explicit MockGraphBufferManager(const Config &config);
        ~MockGraphBufferManager() override = default;

        // Non-copyable but movable
        MockGraphBufferManager(const MockGraphBufferManager &) = delete;
        MockGraphBufferManager &operator=(const MockGraphBufferManager &) = delete;
        MockGraphBufferManager(MockGraphBufferManager &&) = default;
        MockGraphBufferManager &operator=(MockGraphBufferManager &&) = default;

        // =========================================================================
        // IGraphBufferManager Implementation - Allocation
        // =========================================================================

        bool allocateForGraph(ComputeGraph &graph) override
        {
            if (config_.track_calls)
            {
                allocate_for_graph_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            if (allocate_for_graph_callback_)
            {
                return allocate_for_graph_callback_(graph);
            }
            return config_.allocate_for_graph_succeeds;
        }

        bool allocateBuffer(const std::string &node_name, const BufferDescriptor &desc) override
        {
            if (config_.track_calls)
            {
                allocate_buffer_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            if (allocate_buffer_callback_)
            {
                return allocate_buffer_callback_(node_name, desc);
            }
            return config_.allocate_buffer_succeeds;
        }

        bool allocateWithAliasing(ComputeGraph &graph) override
        {
            if (config_.track_calls)
            {
                allocate_with_aliasing_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            if (allocate_with_aliasing_callback_)
            {
                return allocate_with_aliasing_callback_(graph);
            }
            return config_.allocate_with_aliasing_succeeds;
        }

        void releaseAll() override
        {
            if (config_.track_calls)
            {
                release_all_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            // Clear registered buffers (but don't delete - we don't own them)
            buffers_.clear();
            buffer_count_ = 0;
            total_bytes_ = 0;
            stats_.reset();
        }

        // =========================================================================
        // IGraphBufferManager Implementation - Collective Context (Phase 3)
        // =========================================================================

        void setCollectiveContext(std::shared_ptr<CollectiveContext> ctx) override
        {
            if (config_.track_calls)
            {
                set_collective_context_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            collective_ctx_ = std::move(ctx);
        }

        std::shared_ptr<CollectiveContext> collectiveContext() const override
        {
            if (config_.track_calls)
            {
                get_collective_context_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            return collective_ctx_;
        }

        // =========================================================================
        // IGraphBufferManager Implementation - Buffer Retrieval
        // =========================================================================

        TensorBase *getBuffer(const std::string &node_name, const std::string &buffer_name) override
        {
            if (config_.track_calls)
            {
                get_buffer_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            BufferKey key{node_name, buffer_name};
            return getBuffer(key);
        }

        TensorBase *getBuffer(const BufferKey &key) override
        {
            if (config_.track_calls)
            {
                get_buffer_by_key_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            auto it = buffers_.find(key);
            if (it != buffers_.end())
            {
                return it->second;
            }
            return nullptr;
        }

        bool hasBuffer(const std::string &node_name, const std::string &buffer_name) const override
        {
            if (config_.track_calls)
            {
                has_buffer_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            BufferKey key{node_name, buffer_name};
            return buffers_.find(key) != buffers_.end();
        }

        std::vector<BufferKey> getAllBufferKeys() const override
        {
            if (config_.track_calls)
            {
                get_all_keys_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            std::vector<BufferKey> keys;
            keys.reserve(buffers_.size());
            for (const auto &[key, _] : buffers_)
            {
                keys.push_back(key);
            }
            return keys;
        }

        // =========================================================================
        // IGraphBufferManager Implementation - Binding
        // =========================================================================

        bool bindBuffer(const std::string &node_name,
                        const std::string &buffer_name,
                        TensorBase **target_ptr) override
        {
            if (config_.track_calls)
            {
                bind_buffer_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            if (!config_.bind_buffer_succeeds)
            {
                return false;
            }
            if (target_ptr)
            {
                *target_ptr = getBuffer(node_name, buffer_name);
            }
            return true;
        }

        // =========================================================================
        // IGraphBufferManager Implementation - Statistics
        // =========================================================================

        const BufferAllocationStats &stats() const override
        {
            return stats_;
        }

        void resetStats() override
        {
            stats_.reset();
        }

        size_t bufferCount() const override
        {
            return buffer_count_;
        }

        size_t totalAllocatedBytes() const override
        {
            return total_bytes_;
        }

        double aliasingSavingsPercent() const override
        {
            return config_.aliasing_savings_percent;
        }

        size_t aliasingGroupCount() const override
        {
            return config_.aliasing_group_count;
        }

        const std::vector<AliasingGroup> &aliasingGroups() const override
        {
            return aliasing_groups_;
        }

        // =========================================================================
        // IGraphBufferManager Implementation - Debug
        // =========================================================================

        void dumpBufferInventory() const override
        {
            if (config_.track_calls)
            {
                dump_inventory_calls_.fetch_add(1, std::memory_order_relaxed);
            }
            // No-op in mock
        }

        // =========================================================================
        // Mock Configuration - Buffer Registration
        // =========================================================================

        /**
         * @brief Register a mock buffer
         * @param node_name Node name in the graph
         * @param buffer_name Buffer name within the stage
         * @param buffer Pointer to tensor (mock does NOT take ownership)
         */
        void registerBuffer(const std::string &node_name,
                            const std::string &buffer_name,
                            TensorBase *buffer)
        {
            BufferKey key{node_name, buffer_name};
            buffers_[key] = buffer;
            buffer_count_ = buffers_.size();
            if (buffer)
            {
                // Use size_bytes() from ITensor interface
                total_bytes_ += buffer->size_bytes();
                stats_.total_bytes = total_bytes_;
                stats_.total_buffers = buffer_count_;
            }
        }

        /**
         * @brief Register a mock buffer with a descriptor
         * @param node_name Node name in the graph
         * @param desc Buffer descriptor with name and metadata
         * @param buffer Pointer to tensor (mock does NOT take ownership)
         */
        void registerBuffer(const std::string &node_name,
                            const BufferDescriptor &desc,
                            TensorBase *buffer)
        {
            registerBuffer(node_name, desc.name, buffer);
            descriptors_[{node_name, desc.name}] = desc;
        }

        /**
         * @brief Unregister a buffer
         * @param node_name Node name
         * @param buffer_name Buffer name
         */
        void unregisterBuffer(const std::string &node_name, const std::string &buffer_name)
        {
            BufferKey key{node_name, buffer_name};
            auto it = buffers_.find(key);
            if (it != buffers_.end())
            {
                if (it->second)
                {
                    // Use size_bytes() from ITensor interface
                    total_bytes_ -= it->second->size_bytes();
                }
                buffers_.erase(it);
                buffer_count_ = buffers_.size();
                stats_.total_bytes = total_bytes_;
                stats_.total_buffers = buffer_count_;
            }
            descriptors_.erase(key);
        }

        /**
         * @brief Clear all registered buffers
         */
        void clearBuffers()
        {
            buffers_.clear();
            descriptors_.clear();
            buffer_count_ = 0;
            total_bytes_ = 0;
            stats_.reset();
        }

        // =========================================================================
        // Mock Configuration - Behavior Control
        // =========================================================================

        /**
         * @brief Set callback for allocateForGraph
         */
        void setAllocateForGraphCallback(std::function<bool(ComputeGraph &)> callback)
        {
            allocate_for_graph_callback_ = std::move(callback);
        }

        /**
         * @brief Set callback for allocateBuffer
         */
        void setAllocateBufferCallback(
            std::function<bool(const std::string &, const BufferDescriptor &)> callback)
        {
            allocate_buffer_callback_ = std::move(callback);
        }

        /**
         * @brief Set callback for allocateWithAliasing
         */
        void setAllocateWithAliasingCallback(std::function<bool(ComputeGraph &)> callback)
        {
            allocate_with_aliasing_callback_ = std::move(callback);
        }

        /**
         * @brief Set whether allocateForGraph should succeed
         */
        void setAllocateForGraphSucceeds(bool succeeds)
        {
            config_.allocate_for_graph_succeeds = succeeds;
        }

        /**
         * @brief Set whether allocateBuffer should succeed
         */
        void setAllocateBufferSucceeds(bool succeeds)
        {
            config_.allocate_buffer_succeeds = succeeds;
        }

        /**
         * @brief Set whether allocateWithAliasing should succeed
         */
        void setAllocateWithAliasingSucceeds(bool succeeds)
        {
            config_.allocate_with_aliasing_succeeds = succeeds;
        }

        /**
         * @brief Set whether bindBuffer should succeed
         */
        void setBindBufferSucceeds(bool succeeds)
        {
            config_.bind_buffer_succeeds = succeeds;
        }

        /**
         * @brief Set simulated aliasing savings percentage
         */
        void setAliasingSavingsPercent(double percent)
        {
            config_.aliasing_savings_percent = percent;
        }

        /**
         * @brief Set simulated aliasing group count
         */
        void setAliasingGroupCount(size_t count)
        {
            config_.aliasing_group_count = count;
        }

        /**
         * @brief Add a simulated aliasing group
         */
        void addAliasingGroup(const AliasingGroup &group)
        {
            aliasing_groups_.push_back(group);
            config_.aliasing_group_count = aliasing_groups_.size();
        }

        /**
         * @brief Clear aliasing groups
         */
        void clearAliasingGroups()
        {
            aliasing_groups_.clear();
            config_.aliasing_group_count = 0;
        }

        // =========================================================================
        // Test Utilities - Call Counting
        // =========================================================================

        /**
         * @brief Get the number of allocateForGraph() calls
         */
        size_t allocateForGraph_call_count() const
        {
            return allocate_for_graph_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of allocateBuffer() calls
         */
        size_t allocateBuffer_call_count() const
        {
            return allocate_buffer_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of allocateWithAliasing() calls
         */
        size_t allocateWithAliasing_call_count() const
        {
            return allocate_with_aliasing_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of releaseAll() calls
         */
        size_t releaseAll_call_count() const
        {
            return release_all_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of getBuffer(node_name, buffer_name) calls
         */
        size_t getBuffer_call_count() const
        {
            return get_buffer_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of getBuffer(BufferKey) calls
         */
        size_t getBufferByKey_call_count() const
        {
            return get_buffer_by_key_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of hasBuffer() calls
         */
        size_t hasBuffer_call_count() const
        {
            return has_buffer_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of getAllBufferKeys() calls
         */
        size_t getAllBufferKeys_call_count() const
        {
            return get_all_keys_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of bindBuffer() calls
         */
        size_t bindBuffer_call_count() const
        {
            return bind_buffer_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get the number of dumpBufferInventory() calls
         */
        size_t dumpBufferInventory_call_count() const
        {
            return dump_inventory_calls_.load(std::memory_order_relaxed);
        }

        /**
         * @brief Get total method call count
         */
        size_t total_call_count() const
        {
            return allocateForGraph_call_count() + allocateBuffer_call_count() +
                   allocateWithAliasing_call_count() + releaseAll_call_count() +
                   getBuffer_call_count() + getBufferByKey_call_count() +
                   hasBuffer_call_count() + getAllBufferKeys_call_count() +
                   bindBuffer_call_count() + dumpBufferInventory_call_count();
        }

        /**
         * @brief Reset all call counters
         */
        void reset_call_counts()
        {
            allocate_for_graph_calls_.store(0, std::memory_order_relaxed);
            allocate_buffer_calls_.store(0, std::memory_order_relaxed);
            allocate_with_aliasing_calls_.store(0, std::memory_order_relaxed);
            release_all_calls_.store(0, std::memory_order_relaxed);
            get_buffer_calls_.store(0, std::memory_order_relaxed);
            get_buffer_by_key_calls_.store(0, std::memory_order_relaxed);
            has_buffer_calls_.store(0, std::memory_order_relaxed);
            get_all_keys_calls_.store(0, std::memory_order_relaxed);
            bind_buffer_calls_.store(0, std::memory_order_relaxed);
            dump_inventory_calls_.store(0, std::memory_order_relaxed);
        }

        /**
         * @brief Get current configuration (read-only)
         */
        const Config &config() const { return config_; }

    private:
        Config config_;

        // Buffer storage (mock doesn't own these)
        std::unordered_map<BufferKey, TensorBase *, BufferKeyHash> buffers_;
        std::unordered_map<BufferKey, BufferDescriptor, BufferKeyHash> descriptors_;

        // Statistics
        BufferAllocationStats stats_;
        size_t buffer_count_ = 0;
        size_t total_bytes_ = 0;
        std::vector<AliasingGroup> aliasing_groups_;

        // Callbacks for customizing behavior
        std::function<bool(ComputeGraph &)> allocate_for_graph_callback_;
        std::function<bool(const std::string &, const BufferDescriptor &)> allocate_buffer_callback_;
        std::function<bool(ComputeGraph &)> allocate_with_aliasing_callback_;

        // Atomic call counters (thread-safe)
        mutable std::atomic<size_t> allocate_for_graph_calls_{0};
        mutable std::atomic<size_t> allocate_buffer_calls_{0};
        mutable std::atomic<size_t> allocate_with_aliasing_calls_{0};
        mutable std::atomic<size_t> release_all_calls_{0};
        mutable std::atomic<size_t> get_buffer_calls_{0};
        mutable std::atomic<size_t> get_buffer_by_key_calls_{0};
        mutable std::atomic<size_t> has_buffer_calls_{0};
        mutable std::atomic<size_t> get_all_keys_calls_{0};
        mutable std::atomic<size_t> bind_buffer_calls_{0};
        mutable std::atomic<size_t> dump_inventory_calls_{0};
        mutable std::atomic<size_t> set_collective_context_calls_{0};
        mutable std::atomic<size_t> get_collective_context_calls_{0};

        // Collective context (Phase 3)
        std::shared_ptr<CollectiveContext> collective_ctx_;
    };

    // =============================================================================
    // Implementation
    // =============================================================================

    inline MockGraphBufferManager::MockGraphBufferManager()
        : config_{} {}

    inline MockGraphBufferManager::MockGraphBufferManager(const Config &config)
        : config_(config) {}

    inline std::shared_ptr<MockGraphBufferManager> MockGraphBufferManager::createEmpty()
    {
        return std::make_shared<MockGraphBufferManager>();
    }

    inline std::shared_ptr<MockGraphBufferManager> MockGraphBufferManager::createSuccessful()
    {
        Config config;
        config.allocate_for_graph_succeeds = true;
        config.allocate_buffer_succeeds = true;
        config.allocate_with_aliasing_succeeds = true;
        config.bind_buffer_succeeds = true;
        return std::make_shared<MockGraphBufferManager>(config);
    }

    inline std::shared_ptr<MockGraphBufferManager> MockGraphBufferManager::createFailing()
    {
        Config config;
        config.allocate_for_graph_succeeds = false;
        config.allocate_buffer_succeeds = false;
        config.allocate_with_aliasing_succeeds = false;
        config.bind_buffer_succeeds = false;
        return std::make_shared<MockGraphBufferManager>(config);
    }

    // =============================================================================
    // Builder Pattern
    // =============================================================================

    /**
     * @brief Builder for MockGraphBufferManager with fluent interface
     *
     * Usage:
     * @code
     * auto mock = MockGraphBufferManagerBuilder()
     *     .setAllocateForGraphSucceeds(true)
     *     .addBuffer("layer0", "output", output_tensor.get())
     *     .setAliasingSavingsPercent(25.0)
     *     .build();
     * @endcode
     */
    class MockGraphBufferManagerBuilder
    {
    public:
        MockGraphBufferManagerBuilder() = default;

        /**
         * @brief Enable/disable call tracking
         */
        MockGraphBufferManagerBuilder &setTrackCalls(bool track)
        {
            config_.track_calls = track;
            return *this;
        }

        /**
         * @brief Set whether allocateForGraph should succeed
         */
        MockGraphBufferManagerBuilder &setAllocateForGraphSucceeds(bool succeeds)
        {
            config_.allocate_for_graph_succeeds = succeeds;
            return *this;
        }

        /**
         * @brief Set whether allocateBuffer should succeed
         */
        MockGraphBufferManagerBuilder &setAllocateBufferSucceeds(bool succeeds)
        {
            config_.allocate_buffer_succeeds = succeeds;
            return *this;
        }

        /**
         * @brief Set whether allocateWithAliasing should succeed
         */
        MockGraphBufferManagerBuilder &setAllocateWithAliasingSucceeds(bool succeeds)
        {
            config_.allocate_with_aliasing_succeeds = succeeds;
            return *this;
        }

        /**
         * @brief Set whether bindBuffer should succeed
         */
        MockGraphBufferManagerBuilder &setBindBufferSucceeds(bool succeeds)
        {
            config_.bind_buffer_succeeds = succeeds;
            return *this;
        }

        /**
         * @brief Set simulated aliasing savings percentage
         */
        MockGraphBufferManagerBuilder &setAliasingSavingsPercent(double percent)
        {
            config_.aliasing_savings_percent = percent;
            return *this;
        }

        /**
         * @brief Set simulated aliasing group count
         */
        MockGraphBufferManagerBuilder &setAliasingGroupCount(size_t count)
        {
            config_.aliasing_group_count = count;
            return *this;
        }

        /**
         * @brief Add a pre-registered buffer
         * @param node_name Node name in the graph
         * @param buffer_name Buffer name within the stage
         * @param buffer Pointer to tensor (caller retains ownership)
         */
        MockGraphBufferManagerBuilder &addBuffer(const std::string &node_name,
                                                 const std::string &buffer_name,
                                                 TensorBase *buffer)
        {
            pending_buffers_.push_back({node_name, buffer_name, buffer});
            return *this;
        }

        /**
         * @brief Add multiple buffers with a common prefix
         * @param prefix Node name prefix
         * @param buffers Map of buffer_name -> tensor pointer
         */
        MockGraphBufferManagerBuilder &addBuffers(
            const std::string &prefix,
            const std::unordered_map<std::string, TensorBase *> &buffers)
        {
            for (const auto &[name, tensor] : buffers)
            {
                pending_buffers_.push_back({prefix, name, tensor});
            }
            return *this;
        }

        /**
         * @brief Add a simulated aliasing group
         */
        MockGraphBufferManagerBuilder &addAliasingGroup(const AliasingGroup &group)
        {
            pending_aliasing_groups_.push_back(group);
            return *this;
        }

        /**
         * @brief Build the mock
         */
        std::shared_ptr<MockGraphBufferManager> build()
        {
            auto mock = std::make_shared<MockGraphBufferManager>(config_);

            // Register pending buffers
            for (const auto &[node, name, buffer] : pending_buffers_)
            {
                mock->registerBuffer(node, name, buffer);
            }

            // Add pending aliasing groups
            for (const auto &group : pending_aliasing_groups_)
            {
                mock->addAliasingGroup(group);
            }

            return mock;
        }

    private:
        MockGraphBufferManager::Config config_;
        std::vector<std::tuple<std::string, std::string, TensorBase *>> pending_buffers_;
        std::vector<AliasingGroup> pending_aliasing_groups_;
    };

} // namespace llaminar2::test
