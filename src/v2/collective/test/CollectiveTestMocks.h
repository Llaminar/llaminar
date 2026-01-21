/**
 * @file CollectiveTestMocks.h
 * @brief Mock implementations for testing collective infrastructure
 *
 * Provides mock implementations of:
 * - ICollectiveBackend
 * - IBackendFactory
 * - IBackendRouter
 *
 * These mocks enable unit testing of:
 * - CollectiveContext
 * - BackendRouter
 * - GraphExecutor (collective execution path)
 *
 * Without requiring:
 * - Real MPI environment
 * - NCCL/RCCL libraries
 * - Actual GPU hardware
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../ICollectiveBackend.h"
#include "../BackendRouter.h"
#include "../DeviceGroup.h"
#include "../../config/TPDomain.h"
#include <algorithm>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{
    namespace test
    {

        // =========================================================================
        // MockCollectiveBackend
        // =========================================================================

        /**
         * @brief Mock backend for testing collective operations
         *
         * Records all calls and returns configurable results.
         *
         * Usage:
         *   MockCollectiveBackend mock(CollectiveBackendType::MPI);
         *   mock.setAllreduceResult(true);  // Will succeed
         *   mock.setAllreduceCallback([](void* buf, size_t n) {
         *       // Verify buffer contents, simulate reduction, etc.
         *   });
         *
         *   // Use mock in tests
         *   mock.allreduce(buffer, 100, dtype, op, group);
         *
         *   // Verify calls
         *   EXPECT_EQ(mock.allreduceCallCount(), 1);
         */
        class MockCollectiveBackend : public ICollectiveBackend
        {
        public:
            explicit MockCollectiveBackend(CollectiveBackendType type = CollectiveBackendType::HOST)
                : type_(type), name_(toString(type) + "_Mock") {}

            // Identity
            CollectiveBackendType type() const override { return type_; }
            std::string name() const override { return name_; }

            // Capability queries - configurable
            bool supportsDevice(DeviceType type) const override
            {
                return supported_devices_.empty() ||
                       std::find(supported_devices_.begin(), supported_devices_.end(), type) != supported_devices_.end();
            }

            bool supportsDirectTransfer(DeviceId src, DeviceId dst) const override
            {
                return supports_direct_transfer_;
            }

            bool isAvailable() const override { return is_available_; }

            // Lifecycle
            bool initialize(const DeviceGroup &group) override
            {
                initialized_ = true;
                init_group_ = group;
                return init_result_;
            }

            void shutdown() override
            {
                initialized_ = false;
                shutdown_called_ = true;
            }

            bool isInitialized() const override { return initialized_; }

            // Collective operations (match ICollectiveBackend interface - no DeviceGroup param)
            bool allreduce(
                void *buffer,
                size_t count,
                CollectiveDataType dtype,
                CollectiveOp op) override
            {
                allreduce_calls_++;
                last_allreduce_count_ = count;
                last_allreduce_dtype_ = dtype;
                last_allreduce_op_ = op;

                if (allreduce_callback_)
                {
                    allreduce_callback_(buffer, count, dtype, op);
                }
                return allreduce_result_;
            }

            bool allgather(
                const void *send_buf,
                void *recv_buf,
                size_t send_count,
                CollectiveDataType dtype) override
            {
                allgather_calls_++;
                last_allgather_count_ = send_count;

                if (allgather_callback_)
                {
                    allgather_callback_(send_buf, recv_buf, send_count, dtype);
                }
                return allgather_result_;
            }

            bool allgatherv(
                const void * /*send_buf*/,
                size_t /*send_count*/,
                void * /*recv_buf*/,
                const std::vector<int> & /*recv_counts*/,
                const std::vector<int> & /*displacements*/,
                CollectiveDataType /*dtype*/) override
            {
                allgatherv_calls_++;
                return allgatherv_result_;
            }

            bool reduceScatter(
                const void *send_buf,
                void *recv_buf,
                size_t recv_count,
                CollectiveDataType dtype,
                CollectiveOp op) override
            {
                reduce_scatter_calls_++;
                return reduce_scatter_result_;
            }

            bool broadcast(
                void *buffer,
                size_t count,
                CollectiveDataType dtype,
                int root_rank) override
            {
                broadcast_calls_++;
                return broadcast_result_;
            }

            bool synchronize() override
            {
                sync_calls_++;
                return true;
            }

            // =====================================================================
            // Configuration methods (for test setup)
            // =====================================================================

            void setAvailable(bool available) { is_available_ = available; }
            void setInitResult(bool result) { init_result_ = result; }
            void setAllreduceResult(bool result) { allreduce_result_ = result; }
            void setAllgatherResult(bool result) { allgather_result_ = result; }
            void setAllgathervResult(bool result) { allgatherv_result_ = result; }
            void setReduceScatterResult(bool result) { reduce_scatter_result_ = result; }
            void setBroadcastResult(bool result) { broadcast_result_ = result; }
            void setSupportsDirectTransfer(bool supports) { supports_direct_transfer_ = supports; }
            void setSupportedDevices(std::vector<DeviceType> types) { supported_devices_ = std::move(types); }

            using AllreduceCallback = std::function<void(void *, size_t, CollectiveDataType, CollectiveOp)>;
            using AllgatherCallback = std::function<void(const void *, void *, size_t, CollectiveDataType)>;

            void setAllreduceCallback(AllreduceCallback cb) { allreduce_callback_ = std::move(cb); }
            void setAllgatherCallback(AllgatherCallback cb) { allgather_callback_ = std::move(cb); }

            // =====================================================================
            // Verification methods (for test assertions)
            // =====================================================================

            int allreduceCallCount() const { return allreduce_calls_; }
            int allgatherCallCount() const { return allgather_calls_; }
            int allgathervCallCount() const { return allgatherv_calls_; }
            int reduceScatterCallCount() const { return reduce_scatter_calls_; }
            int broadcastCallCount() const { return broadcast_calls_; }
            int syncCallCount() const { return sync_calls_; }

            size_t lastAllreduceCount() const { return last_allreduce_count_; }
            CollectiveDataType lastAllreduceDtype() const { return last_allreduce_dtype_; }
            CollectiveOp lastAllreduceOp() const { return last_allreduce_op_; }

            bool wasInitialized() const { return initialized_; }
            bool wasShutdownCalled() const { return shutdown_called_; }
            const DeviceGroup &initGroup() const { return init_group_; }

            void reset()
            {
                allreduce_calls_ = 0;
                allgather_calls_ = 0;
                allgatherv_calls_ = 0;
                reduce_scatter_calls_ = 0;
                broadcast_calls_ = 0;
                sync_calls_ = 0;
                initialized_ = false;
                shutdown_called_ = false;
            }

        private:
            CollectiveBackendType type_;
            std::string name_;

            // Configuration
            bool is_available_ = true;
            bool init_result_ = true;
            bool allreduce_result_ = true;
            bool allgather_result_ = true;
            bool allgatherv_result_ = true;
            bool reduce_scatter_result_ = true;
            bool broadcast_result_ = true;
            bool supports_direct_transfer_ = true;
            std::vector<DeviceType> supported_devices_;

            AllreduceCallback allreduce_callback_;
            AllgatherCallback allgather_callback_;

            // State
            bool initialized_ = false;
            bool shutdown_called_ = false;
            DeviceGroup init_group_;

            // Call tracking
            int allreduce_calls_ = 0;
            int allgather_calls_ = 0;
            int allgatherv_calls_ = 0;
            int reduce_scatter_calls_ = 0;
            int broadcast_calls_ = 0;
            int sync_calls_ = 0;

            size_t last_allreduce_count_ = 0;
            CollectiveDataType last_allreduce_dtype_ = CollectiveDataType::FLOAT32;
            CollectiveOp last_allreduce_op_ = CollectiveOp::ALLREDUCE_SUM;
            size_t last_allgather_count_ = 0;
        };

        // =========================================================================
        // MockBackendFactory
        // =========================================================================

        /**
         * @brief Mock factory that returns configurable mock backends
         *
         * Usage:
         *   auto factory = std::make_unique<MockBackendFactory>();
         *   auto* mock_mpi = factory->addMockBackend(CollectiveBackendType::MPI);
         *   auto* mock_nccl = factory->addMockBackend(CollectiveBackendType::NCCL);
         *
         *   // Configure mocks
         *   mock_mpi->setAllreduceResult(true);
         *   mock_nccl->setAvailable(false);  // Simulate NCCL not compiled in
         *
         *   // Inject into BackendRouter
         *   BackendRouter router(mpi_ctx, inventory, std::move(factory));
         */
        class MockBackendFactory : public IBackendFactory
        {
        public:
            std::unique_ptr<ICollectiveBackend> createBackend(
                CollectiveBackendType type,
                std::shared_ptr<MPIContext> mpi_ctx) override
            {
                create_calls_++;

                // Return pre-configured mock if available
                auto it = mock_backends_.find(type);
                if (it != mock_backends_.end())
                {
                    // Transfer ownership of pre-configured mock
                    return std::move(it->second);
                }

                // Create new mock with default config
                auto mock = std::make_unique<MockCollectiveBackend>(type);
                mock->setAvailable(isAvailable(type));
                return mock;
            }

            bool isAvailable(CollectiveBackendType type) const override
            {
                auto it = availability_.find(type);
                if (it != availability_.end())
                {
                    return it->second;
                }
                // Default: MPI and HOST always available, NCCL/RCCL not
                return type == CollectiveBackendType::MPI ||
                       type == CollectiveBackendType::HOST;
            }

            // =====================================================================
            // Configuration methods
            // =====================================================================

            /**
             * @brief Add a pre-configured mock backend
             *
             * Returns raw pointer for configuration before factory is used.
             * Ownership transfers to factory until createBackend() is called.
             */
            MockCollectiveBackend *addMockBackend(CollectiveBackendType type)
            {
                auto mock = std::make_unique<MockCollectiveBackend>(type);
                auto *ptr = mock.get();
                mock_backends_[type] = std::move(mock);
                return ptr;
            }

            void setAvailable(CollectiveBackendType type, bool available)
            {
                availability_[type] = available;
            }

            int createCallCount() const { return create_calls_; }

        private:
            std::unordered_map<CollectiveBackendType, std::unique_ptr<MockCollectiveBackend>> mock_backends_;
            std::unordered_map<CollectiveBackendType, bool> availability_;
            int create_calls_ = 0;
        };

        // =========================================================================
        // MockBackendRouter
        // =========================================================================

        /**
         * @brief Mock router for testing CollectiveContext
         *
         * Usage:
         *   auto router = std::make_unique<MockBackendRouter>();
         *   router->setDefaultBackend(&mock_backend);
         *
         *   CollectiveContext ctx(std::move(router), mpi_ctx, devices);
         */
        class MockBackendRouter : public IBackendRouter
        {
        public:
            ICollectiveBackend *getBackend(const DeviceGroup &group) override
            {
                get_backend_calls_++;
                last_group_ = group;
                return default_backend_;
            }

            ICollectiveBackend *getBackend(CollectiveBackendType type) override
            {
                auto it = backends_.find(type);
                return it != backends_.end() ? it->second : default_backend_;
            }

            BackendSelection selectBackend(const DeviceGroup &group) const override
            {
                return selection_;
            }

            bool isAvailable(CollectiveBackendType type) const override
            {
                auto it = availability_.find(type);
                return it != availability_.end() ? it->second : false;
            }

            bool executeHeterogeneousAllReduce(
                const DeviceGroup &group,
                void *buffer,
                size_t count,
                CollectiveDataType dtype,
                CollectiveOp op) override
            {
                hetero_allreduce_calls_++;
                return hetero_allreduce_result_;
            }

            bool executeHeterogeneousAllGather(
                const DeviceGroup &group,
                const void *send_buf,
                void *recv_buf,
                size_t send_count,
                CollectiveDataType dtype) override
            {
                hetero_allgather_calls_++;
                return hetero_allgather_result_;
            }

            ICollectiveBackend *selectBackendForDomain(const TPDomain *domain) override
            {
                select_domain_calls_++;
                last_domain_ = domain;
                return domain_backend_;
            }

            bool hasDomainSupport() const override
            {
                return has_domain_support_;
            }

            // =====================================================================
            // Configuration
            // =====================================================================

            void setDefaultBackend(ICollectiveBackend *backend)
            {
                default_backend_ = backend;
            }

            void setBackend(CollectiveBackendType type, ICollectiveBackend *backend)
            {
                backends_[type] = backend;
            }

            void setSelection(BackendSelection sel) { selection_ = sel; }
            void setAvailable(CollectiveBackendType type, bool avail) { availability_[type] = avail; }
            void setHeteroAllreduceResult(bool result) { hetero_allreduce_result_ = result; }
            void setHeteroAllgatherResult(bool result) { hetero_allgather_result_ = result; }
            void setDomainBackend(ICollectiveBackend *backend) { domain_backend_ = backend; }
            void setHasDomainSupport(bool has) { has_domain_support_ = has; }

            // =====================================================================
            // Verification
            // =====================================================================

            int getBackendCallCount() const { return get_backend_calls_; }
            const DeviceGroup &lastGroup() const { return last_group_; }
            int heteroAllreduceCallCount() const { return hetero_allreduce_calls_; }
            int heteroAllgatherCallCount() const { return hetero_allgather_calls_; }
            int selectDomainCallCount() const { return select_domain_calls_; }
            const TPDomain *lastDomain() const { return last_domain_; }

        private:
            ICollectiveBackend *default_backend_ = nullptr;
            ICollectiveBackend *domain_backend_ = nullptr;
            std::unordered_map<CollectiveBackendType, ICollectiveBackend *> backends_;
            std::unordered_map<CollectiveBackendType, bool> availability_;
            BackendSelection selection_;
            DeviceGroup last_group_;
            const TPDomain *last_domain_ = nullptr;

            int get_backend_calls_ = 0;
            int hetero_allreduce_calls_ = 0;
            int hetero_allgather_calls_ = 0;
            int select_domain_calls_ = 0;
            bool hetero_allreduce_result_ = true;
            bool hetero_allgather_result_ = true;
            bool has_domain_support_ = false;
        };

        // =========================================================================
        // Test Helpers
        // =========================================================================

        /**
         * @brief Create a simple device group for testing
         */
        inline DeviceGroup createTestDeviceGroup(
            const std::string &name,
            std::vector<DeviceId> devices,
            int local_rank = 0)
        {
            DeviceGroupBuilder builder;
            builder.setName(name)
                .addDevices(devices)
                .setLocalRank(local_rank);
            return builder.build();
        }

        /**
         * @brief Create a homogeneous CUDA device group for testing
         */
        inline DeviceGroup createTestCUDAGroup(int num_gpus, int local_rank = 0)
        {
            std::vector<DeviceId> devices;
            for (int i = 0; i < num_gpus; ++i)
            {
                devices.push_back(DeviceId::cuda(i));
            }
            return createTestDeviceGroup("test_cuda_group", devices, local_rank);
        }

        /**
         * @brief Create a heterogeneous device group for testing
         */
        inline DeviceGroup createTestHeterogeneousGroup()
        {
            return createTestDeviceGroup("test_hetero_group",
                                         {DeviceId::cuda(0), DeviceId::cuda(1), DeviceId::cpu()},
                                         0);
        }

    } // namespace test
} // namespace llaminar2
