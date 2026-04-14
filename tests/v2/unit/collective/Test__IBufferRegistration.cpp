/**
 * @file Test__IBufferRegistration.cpp
 * @brief Unit tests for IBufferRegistration interface
 *
 * Tests the buffer registration interface used by collective backends
 * to track buffer locations for cross-device communication.
 */

#include <gtest/gtest.h>
#include "v2/collective/IBufferRegistration.h"
#include "v2/collective/ICollectiveBackend.h"
#include "v2/collective/DeviceGroup.h"
#include "v2/backends/DeviceId.h"

namespace llaminar2::test
{

    // ═══════════════════════════════════════════════════════════════════════════
    // RegisteredBuffer Struct Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(Test__RegisteredBuffer, DefaultConstructor_CreatesInvalidBuffer)
    {
        RegisteredBuffer buf;

        EXPECT_FALSE(buf.device.is_valid());
        EXPECT_EQ(buf.ptr, nullptr);
        EXPECT_EQ(buf.size, 0u);
        EXPECT_EQ(buf.bar_offset, 0u);
        EXPECT_FALSE(buf.is_primary);
        EXPECT_FALSE(buf.isValid());
    }

    TEST(Test__RegisteredBuffer, FullConstructor_SetsAllFields)
    {
        void *test_ptr = reinterpret_cast<void *>(0x12345678);
        DeviceId cuda_device = DeviceId::cuda(0);

        RegisteredBuffer buf(cuda_device, test_ptr, 1024, 512, true);

        EXPECT_TRUE(buf.device.is_cuda());
        EXPECT_EQ(buf.device.ordinal, 0);
        EXPECT_EQ(buf.ptr, test_ptr);
        EXPECT_EQ(buf.size, 1024u);
        EXPECT_EQ(buf.bar_offset, 512u);
        EXPECT_TRUE(buf.is_primary);
        EXPECT_TRUE(buf.isValid());
    }

    TEST(Test__RegisteredBuffer, ConstructorWithDefaults_SetsOptionalFieldsToZero)
    {
        void *test_ptr = reinterpret_cast<void *>(0xABCDEF00);
        DeviceId rocm_device = DeviceId::rocm(1);

        RegisteredBuffer buf(rocm_device, test_ptr, 2048);

        EXPECT_TRUE(buf.device.is_rocm());
        EXPECT_EQ(buf.device.ordinal, 1);
        EXPECT_EQ(buf.ptr, test_ptr);
        EXPECT_EQ(buf.size, 2048u);
        EXPECT_EQ(buf.bar_offset, 0u); // Default
        EXPECT_FALSE(buf.is_primary);  // Default
        EXPECT_TRUE(buf.isValid());
    }

    TEST(Test__RegisteredBuffer, IsValid_ReturnsFalseForNullPointer)
    {
        RegisteredBuffer buf(DeviceId::cuda(0), nullptr, 1024);

        EXPECT_FALSE(buf.isValid());
    }

    TEST(Test__RegisteredBuffer, IsValid_ReturnsFalseForInvalidDevice)
    {
        void *test_ptr = reinterpret_cast<void *>(0x12345678);
        RegisteredBuffer buf(DeviceId::invalid(), test_ptr, 1024);

        EXPECT_FALSE(buf.isValid());
    }

    TEST(Test__RegisteredBuffer, CPUDevice_IsValid)
    {
        void *test_ptr = reinterpret_cast<void *>(0x12345678);
        RegisteredBuffer buf(DeviceId::cpu(), test_ptr, 4096);

        EXPECT_TRUE(buf.device.is_cpu());
        EXPECT_TRUE(buf.isValid());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Mock Backend for Interface Testing
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Minimal mock backend to test IBufferRegistration interface
     *
     * Implements required pure virtual methods with minimal stubs,
     * and inherits default buffer registration behavior.
     */
    class MockBackendWithDefaults : public ICollectiveBackend
    {
    public:
        // Identity
        CollectiveBackendType type() const override { return CollectiveBackendType::HOST; }
        std::string name() const override { return "MockBackend"; }

        // Capability queries
        bool supportsDevice(DeviceType /*type*/) const override { return true; }
        bool supportsDirectTransfer(DeviceId /*src*/, DeviceId /*dst*/) const override { return true; }
        bool isAvailable() const override { return true; }

        // Lifecycle
        bool initialize(const DeviceGroup & /*group*/) override { return true; }
        bool isInitialized() const override { return true; }
        void shutdown() override {}

        // Collective operations
        bool allreduce(void * /*buffer*/, size_t /*count*/,
                       CollectiveDataType /*dtype*/, CollectiveOp /*op*/) override
        {
            return true;
        }

        bool allgather(const void * /*send_buf*/, void * /*recv_buf*/,
                       size_t /*send_count*/, CollectiveDataType /*dtype*/) override
        {
            return true;
        }

        bool allgatherv(const void * /*send_buf*/, size_t /*send_count*/,
                        void * /*recv_buf*/, const std::vector<int> & /*recv_counts*/,
                        const std::vector<int> & /*displacements*/,
                        CollectiveDataType /*dtype*/) override
        {
            return true;
        }

        bool reduceScatter(const void * /*send_buf*/, void * /*recv_buf*/,
                           size_t /*recv_count*/, CollectiveDataType /*dtype*/,
                           CollectiveOp /*op*/) override
        {
            return true;
        }

        bool broadcast(void * /*buffer*/, size_t /*count*/,
                       CollectiveDataType /*dtype*/, int /*root_rank*/) override
        {
            return true;
        }

        bool synchronize() override { return true; }
    };

    /**
     * @brief Mock backend that implements buffer registration
     *
     * Stores registrations in a map for testing purposes.
     */
    class MockBackendWithRegistration : public ICollectiveBackend
    {
    public:
        // Identity
        CollectiveBackendType type() const override { return CollectiveBackendType::HOST; }
        std::string name() const override { return "MockRegistrationBackend"; }

        // Capability queries
        bool supportsDevice(DeviceType /*type*/) const override { return true; }
        bool supportsDirectTransfer(DeviceId /*src*/, DeviceId /*dst*/) const override { return true; }
        bool isAvailable() const override { return true; }

        // Lifecycle
        bool initialize(const DeviceGroup & /*group*/) override { return true; }
        bool isInitialized() const override { return true; }
        void shutdown() override { registrations_.clear(); }

        // Collective operations (stubs)
        bool allreduce(void * /*buffer*/, size_t /*count*/,
                       CollectiveDataType /*dtype*/, CollectiveOp /*op*/) override
        {
            return true;
        }

        bool allgather(const void * /*send_buf*/, void * /*recv_buf*/,
                       size_t /*send_count*/, CollectiveDataType /*dtype*/) override
        {
            return true;
        }

        bool allgatherv(const void * /*send_buf*/, size_t /*send_count*/,
                        void * /*recv_buf*/, const std::vector<int> & /*recv_counts*/,
                        const std::vector<int> & /*displacements*/,
                        CollectiveDataType /*dtype*/) override
        {
            return true;
        }

        bool reduceScatter(const void * /*send_buf*/, void * /*recv_buf*/,
                           size_t /*recv_count*/, CollectiveDataType /*dtype*/,
                           CollectiveOp /*op*/) override
        {
            return true;
        }

        bool broadcast(void * /*buffer*/, size_t /*count*/,
                       CollectiveDataType /*dtype*/, int /*root_rank*/) override
        {
            return true;
        }

        bool synchronize() override { return true; }

        // Override buffer registration
        bool registerBuffer(const std::string &collective_id,
                            DeviceId device,
                            void *buffer,
                            size_t size) override
        {
            std::string key = makeKey(collective_id, device);
            if (registrations_.count(key) > 0)
            {
                return false; // Duplicate registration
            }
            registrations_[key] = RegisteredBuffer(device, buffer, size, next_bar_offset_, false);
            next_bar_offset_ += size;
            return true;
        }

        void unregisterBuffer(const std::string &collective_id,
                              DeviceId device) override
        {
            std::string key = makeKey(collective_id, device);
            registrations_.erase(key);
        }

        std::optional<RegisteredBuffer> getBuffer(const std::string &collective_id,
                                                  DeviceId device) const override
        {
            std::string key = makeKey(collective_id, device);
            auto it = registrations_.find(key);
            if (it == registrations_.end())
            {
                return std::nullopt;
            }
            return it->second;
        }

        bool requiresBufferRegistration() const override
        {
            return true;
        }

        bool allreduceRegistered(const std::string &collective_id,
                                 size_t count,
                                 CollectiveDataType dtype,
                                 CollectiveOp op) override
        {
            // Record the call for testing
            last_allreduce_id_ = collective_id;
            last_allreduce_count_ = count;
            last_allreduce_dtype_ = dtype;
            last_allreduce_op_ = op;
            return true;
        }

        // Test helpers
        size_t registrationCount() const { return registrations_.size(); }
        std::string lastAllreduceId() const { return last_allreduce_id_; }
        size_t lastAllreduceCount() const { return last_allreduce_count_; }

    private:
        static std::string makeKey(const std::string &collective_id, DeviceId device)
        {
            return collective_id + "_" + std::to_string(static_cast<int>(device.type)) +
                   "_" + std::to_string(device.ordinal);
        }

        std::unordered_map<std::string, RegisteredBuffer> registrations_;
        size_t next_bar_offset_ = 0;
        std::string last_allreduce_id_;
        size_t last_allreduce_count_ = 0;
        CollectiveDataType last_allreduce_dtype_ = CollectiveDataType::FLOAT32;
        CollectiveOp last_allreduce_op_ = CollectiveOp::ALLREDUCE_SUM;
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // ICollectiveBackend Default Implementation Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(Test__IBufferRegistration, DefaultRegisterBuffer_ReturnsTrue)
    {
        MockBackendWithDefaults backend;
        void *test_ptr = reinterpret_cast<void *>(0x12345678);

        EXPECT_TRUE(backend.registerBuffer("test_collective", DeviceId::cuda(0), test_ptr, 1024));
    }

    TEST(Test__IBufferRegistration, DefaultUnregisterBuffer_DoesNotThrow)
    {
        MockBackendWithDefaults backend;

        // Should not throw, even for non-existent registration
        EXPECT_NO_THROW(backend.unregisterBuffer("test_collective", DeviceId::cuda(0)));
    }

    TEST(Test__IBufferRegistration, DefaultGetBuffer_ReturnsNullopt)
    {
        MockBackendWithDefaults backend;
        void *test_ptr = reinterpret_cast<void *>(0x12345678);

        // Even after "registering", getBuffer returns nullopt because default is no-op
        backend.registerBuffer("test_collective", DeviceId::cuda(0), test_ptr, 1024);
        auto result = backend.getBuffer("test_collective", DeviceId::cuda(0));

        EXPECT_FALSE(result.has_value());
    }

    TEST(Test__IBufferRegistration, DefaultRequiresBufferRegistration_ReturnsFalse)
    {
        MockBackendWithDefaults backend;

        EXPECT_FALSE(backend.requiresBufferRegistration());
    }

    TEST(Test__IBufferRegistration, DefaultAllreduceRegistered_ReturnsFalse)
    {
        MockBackendWithDefaults backend;

        EXPECT_FALSE(backend.allreduceRegistered(
            "test_collective", 1024, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Custom Implementation Tests (MockBackendWithRegistration)
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(Test__IBufferRegistration, CustomBackend_RegisterBuffer_StoresRegistration)
    {
        MockBackendWithRegistration backend;
        void *test_ptr = reinterpret_cast<void *>(0x12345678);

        EXPECT_TRUE(backend.registerBuffer("layer0_attn", DeviceId::cuda(0), test_ptr, 1024));
        EXPECT_EQ(backend.registrationCount(), 1u);
    }

    TEST(Test__IBufferRegistration, CustomBackend_RegisterBuffer_RejectsDuplicate)
    {
        MockBackendWithRegistration backend;
        void *ptr1 = reinterpret_cast<void *>(0x12345678);
        void *ptr2 = reinterpret_cast<void *>(0xABCDEF00);

        EXPECT_TRUE(backend.registerBuffer("layer0_attn", DeviceId::cuda(0), ptr1, 1024));
        EXPECT_FALSE(backend.registerBuffer("layer0_attn", DeviceId::cuda(0), ptr2, 2048));
        EXPECT_EQ(backend.registrationCount(), 1u);
    }

    TEST(Test__IBufferRegistration, CustomBackend_RegisterBuffer_AllowsDifferentDevices)
    {
        MockBackendWithRegistration backend;
        void *ptr_cuda = reinterpret_cast<void *>(0x12345678);
        void *ptr_rocm = reinterpret_cast<void *>(0xABCDEF00);

        EXPECT_TRUE(backend.registerBuffer("layer0_attn", DeviceId::cuda(0), ptr_cuda, 1024));
        EXPECT_TRUE(backend.registerBuffer("layer0_attn", DeviceId::rocm(0), ptr_rocm, 1024));
        EXPECT_EQ(backend.registrationCount(), 2u);
    }

    TEST(Test__IBufferRegistration, CustomBackend_GetBuffer_ReturnsRegistration)
    {
        MockBackendWithRegistration backend;
        void *test_ptr = reinterpret_cast<void *>(0x12345678);

        backend.registerBuffer("layer0_attn", DeviceId::cuda(0), test_ptr, 1024);

        auto result = backend.getBuffer("layer0_attn", DeviceId::cuda(0));

        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->ptr, test_ptr);
        EXPECT_EQ(result->size, 1024u);
        EXPECT_TRUE(result->device.is_cuda());
        EXPECT_EQ(result->device.ordinal, 0);
    }

    TEST(Test__IBufferRegistration, CustomBackend_GetBuffer_ReturnsNulloptForUnknown)
    {
        MockBackendWithRegistration backend;
        void *test_ptr = reinterpret_cast<void *>(0x12345678);

        backend.registerBuffer("layer0_attn", DeviceId::cuda(0), test_ptr, 1024);

        // Different collective_id
        EXPECT_FALSE(backend.getBuffer("layer1_attn", DeviceId::cuda(0)).has_value());

        // Different device
        EXPECT_FALSE(backend.getBuffer("layer0_attn", DeviceId::rocm(0)).has_value());
    }

    TEST(Test__IBufferRegistration, CustomBackend_UnregisterBuffer_RemovesRegistration)
    {
        MockBackendWithRegistration backend;
        void *test_ptr = reinterpret_cast<void *>(0x12345678);

        backend.registerBuffer("layer0_attn", DeviceId::cuda(0), test_ptr, 1024);
        EXPECT_TRUE(backend.getBuffer("layer0_attn", DeviceId::cuda(0)).has_value());

        backend.unregisterBuffer("layer0_attn", DeviceId::cuda(0));

        EXPECT_FALSE(backend.getBuffer("layer0_attn", DeviceId::cuda(0)).has_value());
        EXPECT_EQ(backend.registrationCount(), 0u);
    }

    TEST(Test__IBufferRegistration, CustomBackend_UnregisterBuffer_SafeForNonExistent)
    {
        MockBackendWithRegistration backend;

        // Should not throw for non-existent registration
        EXPECT_NO_THROW(backend.unregisterBuffer("nonexistent", DeviceId::cuda(0)));
        EXPECT_EQ(backend.registrationCount(), 0u);
    }

    TEST(Test__IBufferRegistration, CustomBackend_RequiresBufferRegistration_ReturnsTrue)
    {
        MockBackendWithRegistration backend;

        EXPECT_TRUE(backend.requiresBufferRegistration());
    }

    TEST(Test__IBufferRegistration, CustomBackend_AllreduceRegistered_Works)
    {
        MockBackendWithRegistration backend;
        void *ptr_cuda = reinterpret_cast<void *>(0x12345678);
        void *ptr_rocm = reinterpret_cast<void *>(0xABCDEF00);

        // Register buffers
        backend.registerBuffer("layer0_attn", DeviceId::cuda(0), ptr_cuda, 1024);
        backend.registerBuffer("layer0_attn", DeviceId::rocm(0), ptr_rocm, 1024);

        // Perform registered allreduce
        EXPECT_TRUE(backend.allreduceRegistered(
            "layer0_attn", 256, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));

        // Verify the call was recorded
        EXPECT_EQ(backend.lastAllreduceId(), "layer0_attn");
        EXPECT_EQ(backend.lastAllreduceCount(), 256u);
    }

    TEST(Test__IBufferRegistration, CustomBackend_BAROffset_AssignedSequentially)
    {
        MockBackendWithRegistration backend;
        void *ptr1 = reinterpret_cast<void *>(0x11111111);
        void *ptr2 = reinterpret_cast<void *>(0x22222222);
        void *ptr3 = reinterpret_cast<void *>(0x33333333);

        backend.registerBuffer("col1", DeviceId::cuda(0), ptr1, 1024);
        backend.registerBuffer("col2", DeviceId::cuda(0), ptr2, 2048);
        backend.registerBuffer("col3", DeviceId::cuda(0), ptr3, 512);

        auto buf1 = backend.getBuffer("col1", DeviceId::cuda(0));
        auto buf2 = backend.getBuffer("col2", DeviceId::cuda(0));
        auto buf3 = backend.getBuffer("col3", DeviceId::cuda(0));

        ASSERT_TRUE(buf1.has_value());
        ASSERT_TRUE(buf2.has_value());
        ASSERT_TRUE(buf3.has_value());

        EXPECT_EQ(buf1->bar_offset, 0u);
        EXPECT_EQ(buf2->bar_offset, 1024u);
        EXPECT_EQ(buf3->bar_offset, 3072u);
    }

    TEST(Test__IBufferRegistration, CustomBackend_Shutdown_ClearsRegistrations)
    {
        MockBackendWithRegistration backend;
        void *ptr1 = reinterpret_cast<void *>(0x11111111);
        void *ptr2 = reinterpret_cast<void *>(0x22222222);

        backend.registerBuffer("col1", DeviceId::cuda(0), ptr1, 1024);
        backend.registerBuffer("col2", DeviceId::rocm(0), ptr2, 2048);
        EXPECT_EQ(backend.registrationCount(), 2u);

        backend.shutdown();

        EXPECT_EQ(backend.registrationCount(), 0u);
        EXPECT_FALSE(backend.getBuffer("col1", DeviceId::cuda(0)).has_value());
    }

} // namespace llaminar2::test
