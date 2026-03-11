/**
 * @file MockComputeStage.h
 * @brief Mock implementations of execution graph components for testing
 * @author David Sanftenberg
 * @date December 2025
 *
 * These mocks enable testing of:
 * 1. Graph execution order (without real kernels)
 * 2. Dependency resolution
 * 3. Error handling
 * 4. Device context interactions
 * 5. LayerExecutor orchestration (via MockLayerExecutor)
 */

#pragma once

#include "execution/compute_stages/ComputeStages.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/model/ILayerExecutor.h" // For ILayerExecutor interface
#include "execution/local_execution/model/LayerExecutor.h"  // For ComputeGraph
#include "backends/DeviceId.h"
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>
#include <memory>
#include <algorithm>

namespace llaminar2
{
    namespace testing
    {

        /**
         * @brief Mock compute stage for testing graph execution
         *
         * Tracks execution calls and can be configured to succeed or fail.
         * Useful for testing:
         * - Graph topology (execution order)
         * - Error propagation
         * - Dependency handling
         */
        class MockComputeStage : public IComputeStage
        {
        public:
            explicit MockComputeStage(ComputeStageType type = ComputeStageType::GEMM,
                                      std::string name = "MockStage",
                                      DeviceId device = DeviceId::cpu())
                : IComputeStage(device), type_(type), name_(std::move(name)) {}

            bool execute(IDeviceContext *ctx) override
            {
                execution_count_++;
                last_ctx_ = ctx;

                // Record execution for order verification
                if (execution_log_)
                {
                    execution_log_->push_back(name_);
                }

                // Call custom execution hook if set
                if (on_execute_)
                {
                    on_execute_(ctx);
                }

                return should_succeed_;
            }

            ComputeStageType type() const override { return type_; }
            std::string name() const override { return name_; }
            CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
            size_t estimatedFlops() const override { return estimated_flops_; }
            size_t estimatedMemoryBytes() const override { return estimated_memory_; }
            bool requiresAllreduce() const override { return requires_allreduce_; }

            bool supportsBackend(ComputeBackendType backend) const override
            {
                return supported_backends_.empty() ||
                       std::find(supported_backends_.begin(), supported_backends_.end(), backend) != supported_backends_.end();
            }

            StageDumpInfo buildDumpInfoImpl() const override
            {
                return dump_info_;
            }

            // =========================================================================
            // Test Configuration
            // =========================================================================

            /// Set whether execute() should return success or failure
            void setShouldSucceed(bool succeed) { should_succeed_ = succeed; }

            /// Set estimated FLOPS for load balancing tests
            void setEstimatedFlops(size_t flops) { estimated_flops_ = flops; }

            /// Set estimated memory for memory planning tests
            void setEstimatedMemory(size_t bytes) { estimated_memory_ = bytes; }

            /// Set whether stage requires allreduce
            void setRequiresAllreduce(bool needs_allreduce) { requires_allreduce_ =
                                                                  needs_allreduce; }

            /// Set supported backends (empty = all)
            void setSupportedBackends(std::vector<ComputeBackendType> backends)
            {
                supported_backends_ = std::move(backends);
            }

            /// Set dump info for debugging tests
            void setDumpInfo(StageDumpInfo info) { dump_info_ = std::move(info); }

            /// Set execution log vector (for order verification)
            void setExecutionLog(std::vector<std::string> *log) { execution_log_ = log; }

            /// Set custom execution hook
            void setOnExecute(std::function<void(IDeviceContext *)> hook)
            {
                on_execute_ = std::move(hook);
            }

            // =========================================================================
            // Test Verification
            // =========================================================================

            /// Number of times execute() was called
            int executionCount() const { return execution_count_; }

            /// Last device context passed to execute()
            IDeviceContext *lastContext() const { return last_ctx_; }

            /// Reset execution tracking
            void reset()
            {
                execution_count_ = 0;
                last_ctx_ = nullptr;
            }

        private:
            ComputeStageType type_;
            std::string name_;

            // Configuration
            bool should_succeed_ = true;
            size_t estimated_flops_ = 0;
            size_t estimated_memory_ = 0;
            bool requires_allreduce_ = false;
            std::vector<ComputeBackendType> supported_backends_;
            StageDumpInfo dump_info_;

            // Execution tracking
            int execution_count_ = 0;
            IDeviceContext *last_ctx_ = nullptr;
            std::vector<std::string> *execution_log_ = nullptr;
            std::function<void(IDeviceContext *)> on_execute_;
        };

        /**
         * @brief Mock device context for testing
         *
         * Simulates a device context without real hardware.
         * Tracks operations for verification.
         */
        class MockDeviceContext : public IDeviceContext
        {
        public:
            explicit MockDeviceContext(DeviceId device = DeviceId::cpu(),
                                       ComputeBackendType backend = ComputeBackendType::CPU)
                : device_id_(device), backend_(backend) {}

            // Legacy constructor for backward compatibility
            explicit MockDeviceContext(int device_idx,
                                       ComputeBackendType backend)
                : device_id_(device_idx == -1 || device_idx == 0 ? DeviceId::cpu() : DeviceId::cuda(device_idx - 1)),
                  backend_(backend) {}

            DeviceId deviceId() const override { return device_id_; }
            int deviceIndex() const override { return device_id_.toKernelDeviceIndex(); }
            ComputeBackendType backendType() const override { return backend_; }
            DeviceType deviceType() const override
            {
                switch (backend_)
                {
                case ComputeBackendType::GPU_CUDA:
                    return DeviceType::CUDA;
                case ComputeBackendType::GPU_ROCM:
                    return DeviceType::ROCm;
                case ComputeBackendType::GPU_VULKAN:
                    return DeviceType::Vulkan;
                case ComputeBackendType::GPU_METAL:
                    return DeviceType::Metal;
                default:
                    return DeviceType::CPU;
                }
            }
            bool isGPU() const override
            {
                return backend_ == ComputeBackendType::GPU_CUDA ||
                       backend_ == ComputeBackendType::GPU_ROCM ||
                       backend_ == ComputeBackendType::GPU_VULKAN ||
                       backend_ == ComputeBackendType::GPU_METAL;
            }
            std::string deviceName() const override { return "MockDevice_" + device_id_.toString(); }

            void synchronize() override { sync_count_++; }
            void barrier() override { barrier_count_++; }

            void *allocate(size_t bytes) override
            {
                allocations_.push_back(bytes);
                total_allocated_ += bytes;
                // Return a dummy pointer (don't actually allocate)
                return reinterpret_cast<void *>(0xDEADBEEF + allocations_.size());
            }

            void free(void *ptr) override
            {
                free_count_++;
            }

            void *getWorkspace(size_t bytes) override
            {
                workspace_requests_.push_back(bytes);
                return reinterpret_cast<void *>(0xBEEFCAFE);
            }

            size_t availableMemory() const override { return available_memory_; }
            size_t totalMemory() const override { return total_memory_; }

            bool copyToDevice(void *dst, const void *src, size_t bytes) override
            {
                h2d_transfers_.push_back(bytes);
                return true;
            }

            bool copyToHost(void *dst, const void *src, size_t bytes) override
            {
                d2h_transfers_.push_back(bytes);
                return true;
            }

            bool copyFromDevice(void *dst, const void *src, size_t bytes,
                                IDeviceContext *src_ctx) override
            {
                d2d_transfers_.push_back(bytes);
                return true;
            }

            void runParallel(std::function<void(int, int)> work) override
            {
                parallel_runs_++;
                work(0, 1); // Single-threaded for testing
            }

            void runFor(size_t start, size_t end, std::function<void(size_t)> work) override
            {
                for_runs_++;
                for (size_t i = start; i < end; ++i)
                {
                    work(i);
                }
            }

            int numThreads() const override { return num_threads_; }

            // =========================================================================
            // Test Configuration
            // =========================================================================

            void setAvailableMemory(size_t bytes) { available_memory_ = bytes; }
            void setTotalMemory(size_t bytes) { total_memory_ = bytes; }
            void setNumThreads(int threads) { num_threads_ = threads; }

            // =========================================================================
            // Test Verification
            // =========================================================================

            int syncCount() const { return sync_count_; }
            int barrierCount() const { return barrier_count_; }
            int freeCount() const { return free_count_; }
            int parallelRuns() const { return parallel_runs_; }
            int forRuns() const { return for_runs_; }
            size_t totalAllocated() const { return total_allocated_; }
            const std::vector<size_t> &allocations() const { return allocations_; }
            const std::vector<size_t> &workspaceRequests() const { return workspace_requests_; }
            const std::vector<size_t> &h2dTransfers() const { return h2d_transfers_; }
            const std::vector<size_t> &d2hTransfers() const { return d2h_transfers_; }
            const std::vector<size_t> &d2dTransfers() const { return d2d_transfers_; }

            void reset()
            {
                sync_count_ = 0;
                barrier_count_ = 0;
                free_count_ = 0;
                parallel_runs_ = 0;
                for_runs_ = 0;
                total_allocated_ = 0;
                allocations_.clear();
                workspace_requests_.clear();
                h2d_transfers_.clear();
                d2h_transfers_.clear();
                d2d_transfers_.clear();
            }

        private:
            DeviceId device_id_;
            ComputeBackendType backend_;

            // Configuration
            size_t available_memory_ = 16ULL * 1024 * 1024 * 1024; // 16GB default
            size_t total_memory_ = 16ULL * 1024 * 1024 * 1024;
            int num_threads_ = 4;

            // Tracking
            int sync_count_ = 0;
            int barrier_count_ = 0;
            int free_count_ = 0;
            int parallel_runs_ = 0;
            int for_runs_ = 0;
            size_t total_allocated_ = 0;
            std::vector<size_t> allocations_;
            std::vector<size_t> workspace_requests_;
            std::vector<size_t> h2d_transfers_;
            std::vector<size_t> d2h_transfers_;
            std::vector<size_t> d2d_transfers_;
        };

        /**
         * @brief Helper to create a mock graph with configurable topology
         *
         * Example usage:
         * @code
         * MockGraphBuilder builder;
         * builder.addNode("A", ComputeStageType::RMS_NORM);
         * builder.addNode("B", ComputeStageType::GEMM);
         * builder.addNode("C", ComputeStageType::GEMM);
         * builder.addDependency("B", "A");
         * builder.addDependency("C", "A");
         * auto graph = builder.build();
         * @endcode
         */
        class MockGraphBuilder
        {
        public:
            MockGraphBuilder &addNode(const std::string &name,
                                      ComputeStageType type = ComputeStageType::GEMM,
                                      DeviceId device = DeviceId::cpu())
            {
                auto stage = std::make_unique<MockComputeStage>(type, name);
                stage->setExecutionLog(&execution_log_);
                stages_[name] = stage.get();
                nodes_.push_back({name, std::move(stage), device});
                return *this;
            }

            MockGraphBuilder &addDependency(const std::string &node, const std::string &depends_on)
            {
                dependencies_.push_back({node, depends_on});
                return *this;
            }

            /// Configure a specific stage
            MockComputeStage *getStage(const std::string &name)
            {
                auto it = stages_.find(name);
                return it != stages_.end() ? it->second : nullptr;
            }

            /// Build the compute graph
            ComputeGraph build()
            {
                ComputeGraph graph;
                for (auto &[name, stage, device] : nodes_)
                {
                    graph.addNode(name, std::move(stage), device);
                }
                for (auto &[node, depends_on] : dependencies_)
                {
                    graph.addDependency(node, depends_on);
                }
                return graph;
            }

            /// Get execution log after graph execution
            const std::vector<std::string> &executionLog() const { return execution_log_; }

            /// Clear execution log
            void clearLog() { execution_log_.clear(); }

        private:
            struct NodeInfo
            {
                std::string name;
                std::unique_ptr<MockComputeStage> stage;
                DeviceId device;
            };

            std::vector<NodeInfo> nodes_;
            std::vector<std::pair<std::string, std::string>> dependencies_;
            std::unordered_map<std::string, MockComputeStage *> stages_;
            std::vector<std::string> execution_log_;
        };

        /**
         * @brief Mock LayerExecutor for testing higher-level components
         *
         * Allows testing of pipeline code without executing real compute graphs.
         * Tracks execute() calls and can be configured to succeed or fail.
         */
        class MockLayerExecutor : public ILayerExecutor
        {
        public:
            explicit MockLayerExecutor(const LayerExecutorConfig &config = LayerExecutorConfig())
                : config_(config) {}

            ~MockLayerExecutor() override = default;

            // Configuration (ILayerExecutor interface)
            const LayerExecutorConfig &config() const override { return config_; }
            void setExecutionMode(ExecutionMode mode) override { config_.mode = mode; }
            ExecutionMode executionMode() const override { return config_.mode; }
            void setProfilingEnabled(bool enabled) override { config_.enable_profiling = enabled; }
            void setValidationEnabled(bool enabled) override { config_.enable_validation = enabled; }
            void setSnapshotCallback(StageSnapshotCallback callback) override
            {
                config_.snapshot_callback = std::move(callback);
            }
            void setCurrentLayerIdx(int layer_idx) override { config_.current_layer_idx = layer_idx; }
            void setCurrentIteration(int iteration) override { config_.current_iteration = iteration; }
            void setMPIRank(int rank) override { config_.mpi_rank = rank; }

            // Execution (ILayerExecutor interface - mock implementation)
            bool execute(ComputeGraph &graph, IDeviceContext *ctx) override
            {
                execute_count_++;
                last_graph_ = &graph;
                last_ctx_ = ctx;

                if (on_execute_)
                {
                    on_execute_(graph, ctx);
                }

                // Optionally call snapshot callback for each node
                if (config_.snapshot_callback && call_snapshot_callback_)
                {
                    for (const auto &name : graph.getExecutionOrder())
                    {
                        auto *node = graph.getNode(name);
                        if (node && node->stage)
                        {
                            config_.snapshot_callback(name, node->stage->getDumpInfo());
                        }
                    }
                }

                return should_succeed_;
            }

            bool executeMultiDevice(
                ComputeGraph &graph,
                const std::unordered_map<DeviceId, IDeviceContext *> &contexts) override
            {
                multi_execute_count_++;
                return should_succeed_;
            }

            // Statistics (ILayerExecutor interface)
            const LayerExecutorStats &stats() const override { return stats_; }
            void resetStats() override { stats_.reset(); }

            // =========================================================================
            // Test Configuration
            // =========================================================================

            void setShouldSucceed(bool succeed) { should_succeed_ = succeed; }
            void setCallSnapshotCallback(bool call) { call_snapshot_callback_ = call; }
            void setOnExecute(std::function<void(ComputeGraph &, IDeviceContext *)> hook)
            {
                on_execute_ = std::move(hook);
            }

            // =========================================================================
            // Test Verification
            // =========================================================================

            int executeCount() const { return execute_count_; }
            int multiExecuteCount() const { return multi_execute_count_; }
            ComputeGraph *lastGraph() const { return last_graph_; }
            IDeviceContext *lastContext() const { return last_ctx_; }

            void reset()
            {
                execute_count_ = 0;
                multi_execute_count_ = 0;
                last_graph_ = nullptr;
                last_ctx_ = nullptr;
            }

        private:
            LayerExecutorConfig config_;
            LayerExecutorStats stats_;

            // Configuration
            bool should_succeed_ = true;
            bool call_snapshot_callback_ = false;
            std::function<void(ComputeGraph &, IDeviceContext *)> on_execute_;

            // Tracking
            int execute_count_ = 0;
            int multi_execute_count_ = 0;
            ComputeGraph *last_graph_ = nullptr;
            IDeviceContext *last_ctx_ = nullptr;
        };

        /**
         * @brief Factory function to create MockComputeStage with common configurations
         */
        inline std::unique_ptr<MockComputeStage> createMockStage(
            const std::string &name,
            ComputeStageType type = ComputeStageType::GEMM,
            size_t estimated_flops = 0,
            bool should_succeed = true)
        {
            auto stage = std::make_unique<MockComputeStage>(type, name);
            stage->setEstimatedFlops(estimated_flops);
            stage->setShouldSucceed(should_succeed);
            return stage;
        }

    } // namespace testing
} // namespace llaminar2
