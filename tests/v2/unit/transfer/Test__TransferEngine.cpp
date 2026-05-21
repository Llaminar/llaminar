#include <gtest/gtest.h>

#include "tensors/CoherenceState.h"
#include "transfer/TransferEngine.h"
#include "transfer/TransferMethod.h"

// For execute tests
#include "backends/DeviceId.h"
#include "tensors/TensorClasses.h"
#include "../../mocks/MockBackend.h"
#include "../../utils/TestTensorFactory.h"

#include <cstring>
#include <memory>

using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// planTransfer() tests — pure logic, no GPU needed
// ============================================================================

class Test__TransferEngine_Plan : public ::testing::Test
{
};

TEST(Test__TransferEngine_Plan, CpuToCuda_HostToDevice)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cpu(), DeviceId::cuda(0), MemoryResidency::STANDARD);
    EXPECT_EQ(method, TransferMethod::HOST_TO_DEVICE);
}

TEST(Test__TransferEngine_Plan, CpuToRocm_HostToDevice)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cpu(), DeviceId::rocm(0), MemoryResidency::STANDARD);
    EXPECT_EQ(method, TransferMethod::HOST_TO_DEVICE);
}

TEST(Test__TransferEngine_Plan, CudaToCpu_DeviceToHost)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cuda(0), DeviceId::cpu(), MemoryResidency::STANDARD);
    EXPECT_EQ(method, TransferMethod::DEVICE_TO_HOST);
}

TEST(Test__TransferEngine_Plan, RocmToCpu_DeviceToHost)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::rocm(0), DeviceId::cpu(), MemoryResidency::STANDARD);
    EXPECT_EQ(method, TransferMethod::DEVICE_TO_HOST);
}

TEST(Test__TransferEngine_Plan, CudaToCuda_SameBackend)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cuda(0), DeviceId::cuda(1), MemoryResidency::STANDARD);
    EXPECT_EQ(method, TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND);
}

TEST(Test__TransferEngine_Plan, RocmToRocm_SameBackend)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::rocm(0), DeviceId::rocm(1), MemoryResidency::STANDARD);
    EXPECT_EQ(method, TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND);
}

TEST(Test__TransferEngine_Plan, CudaToRocm_HostStaged)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cuda(0), DeviceId::rocm(0), MemoryResidency::STANDARD);
    EXPECT_EQ(method, TransferMethod::HOST_STAGED);
}

TEST(Test__TransferEngine_Plan, RocmToCuda_HostStaged)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::rocm(0), DeviceId::cuda(0), MemoryResidency::STANDARD);
    EXPECT_EQ(method, TransferMethod::HOST_STAGED);
}

TEST(Test__TransferEngine_Plan, SameDevice_Noop)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cuda(0), DeviceId::cuda(0), MemoryResidency::STANDARD);
    EXPECT_EQ(method, TransferMethod::NOOP);
}

TEST(Test__TransferEngine_Plan, CpuToCpu_Noop)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cpu(), DeviceId::cpu(), MemoryResidency::STANDARD);
    EXPECT_EQ(method, TransferMethod::NOOP);
}

TEST(Test__TransferEngine_Plan, Mapped_AlwaysMappedNoop)
{
    // Mapped memory is always a no-op regardless of devices
    EXPECT_EQ(TransferEngine::planTransfer(
                  DeviceId::cpu(), DeviceId::cuda(0), MemoryResidency::MAPPED),
              TransferMethod::MAPPED_NOOP);

    EXPECT_EQ(TransferEngine::planTransfer(
                  DeviceId::cuda(0), DeviceId::cpu(), MemoryResidency::MAPPED),
              TransferMethod::MAPPED_NOOP);

    EXPECT_EQ(TransferEngine::planTransfer(
                  DeviceId::cuda(0), DeviceId::rocm(0), MemoryResidency::MAPPED),
              TransferMethod::MAPPED_NOOP);
}

TEST(Test__TransferEngine_Plan, HostResident_AlwaysNoop)
{
    // HOST_RESIDENT tensors never move to device — always NOOP regardless of direction
    EXPECT_EQ(TransferEngine::planTransfer(
                  DeviceId::cpu(), DeviceId::cuda(0), MemoryResidency::HOST_RESIDENT),
              TransferMethod::NOOP);

    EXPECT_EQ(TransferEngine::planTransfer(
                  DeviceId::cpu(), DeviceId::rocm(0), MemoryResidency::HOST_RESIDENT),
              TransferMethod::NOOP);

    EXPECT_EQ(TransferEngine::planTransfer(
                  DeviceId::cuda(0), DeviceId::cpu(), MemoryResidency::HOST_RESIDENT),
              TransferMethod::NOOP);

    EXPECT_EQ(TransferEngine::planTransfer(
                  DeviceId::cuda(0), DeviceId::rocm(0), MemoryResidency::HOST_RESIDENT),
              TransferMethod::NOOP);

    // Same device also NOOP (both paths agree)
    EXPECT_EQ(TransferEngine::planTransfer(
                  DeviceId::cpu(), DeviceId::cpu(), MemoryResidency::HOST_RESIDENT),
              TransferMethod::NOOP);
}

TEST(Test__TransferEngine_Plan, HostResident_PrecedesOtherChecks)
{
    // HOST_RESIDENT should NOOP even for cross-vendor transfers that would
    // normally be HOST_STAGED — residency takes priority.
    EXPECT_EQ(TransferEngine::planTransfer(
                  DeviceId::rocm(0), DeviceId::cuda(0), MemoryResidency::HOST_RESIDENT),
              TransferMethod::NOOP);
}

TEST(Test__TransferEngine_Plan, DescribeTransferPlan_HumanReadable)
{
    auto desc = TransferEngine::describeTransferPlan(
        DeviceId::cpu(), DeviceId::cuda(0), MemoryResidency::STANDARD);

    // Should contain source, destination, method
    EXPECT_NE(desc.find("HOST_TO_DEVICE"), std::string::npos);
}

TEST(Test__TransferEngine_Plan, DescribeTransferPlan_HostResident)
{
    auto desc = TransferEngine::describeTransferPlan(
        DeviceId::cpu(), DeviceId::cuda(0), MemoryResidency::HOST_RESIDENT);

    EXPECT_NE(desc.find("HOST_RESIDENT"), std::string::npos);
    EXPECT_NE(desc.find("NOOP"), std::string::npos);
}

// ============================================================================
// execute() tests — uses MockBackend
// ============================================================================

class Test__TransferEngine_Execute : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mock_backend_ = std::make_shared<MockBackend>(DeviceType::CUDA);

        // Create engine with custom resolver that returns our mock
        resolver_ = [](DeviceId) -> IBackend *
        {
            // The lambda captures nothing; we use a static pointer.
            return s_mock_;
        };
    }

    // Static mock pointer for the resolver lambda
    static MockBackend *s_mock_;
    std::shared_ptr<MockBackend> mock_backend_;
    TransferEngine::BackendResolver resolver_;
};

MockBackend *Test__TransferEngine_Execute::s_mock_ = nullptr;

TEST_F(Test__TransferEngine_Execute, HostToDevice_RecordsTransfer)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    // Prepare host data
    float host_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    // Allocate "device" memory via mock
    void *device_ptr = mock_backend_->allocate(sizeof(host_data), 0);
    ASSERT_NE(device_ptr, nullptr);

    // Build request
    MemoryDescriptor desc;
    desc.host_ptr = host_data;
    desc.device = DeviceId::cpu();
    desc.size_bytes = sizeof(host_data);
    desc.residency = MemoryResidency::STANDARD;

    TransferRequest req;
    req.source = desc;
    req.target_device = DeviceId::cuda(0);
    req.method = TransferMethod::HOST_TO_DEVICE;
    req.target_ptr = device_ptr;

    auto result = engine.execute(req);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::HOST_TO_DEVICE);

    // Verify mock recorded the transfer
    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 1u);
    EXPECT_EQ(stats.h2d_bytes, sizeof(host_data));

    // Verify data was copied (MockBackend does real memcpy)
    auto *result_data = static_cast<float *>(device_ptr);
    EXPECT_FLOAT_EQ(result_data[0], 1.0f);
    EXPECT_FLOAT_EQ(result_data[3], 4.0f);

    mock_backend_->free(device_ptr, 0);
}

TEST_F(Test__TransferEngine_Execute, DeviceToHost_RecordsTransfer)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    // Allocate "device" memory and fill it
    constexpr size_t bytes = 4 * sizeof(float);
    void *device_ptr = mock_backend_->allocate(bytes, 0);
    ASSERT_NE(device_ptr, nullptr);
    float device_data[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    std::memcpy(device_ptr, device_data, bytes);

    // Host destination
    float host_data[4] = {0};

    MemoryDescriptor desc;
    desc.host_ptr = host_data;
    desc.device_ptr = device_ptr;
    desc.device = DeviceId::cuda(0);
    desc.size_bytes = bytes;
    desc.residency = MemoryResidency::STANDARD;

    TransferRequest req;
    req.source = desc;
    req.target_device = DeviceId::cpu();
    req.method = TransferMethod::DEVICE_TO_HOST;

    auto result = engine.execute(req);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::DEVICE_TO_HOST);

    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.d2h_count, 1u);
    EXPECT_EQ(stats.d2h_bytes, bytes);

    // Verify data
    EXPECT_FLOAT_EQ(host_data[0], 10.0f);
    EXPECT_FLOAT_EQ(host_data[3], 40.0f);

    mock_backend_->free(device_ptr, 0);
}

TEST_F(Test__TransferEngine_Execute, HostStaged_RecordsBothTransfers)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    // Source "device" memory (simulating CUDA)
    constexpr size_t bytes = 4 * sizeof(float);
    void *src_device = mock_backend_->allocate(bytes, 0);
    ASSERT_NE(src_device, nullptr);
    float src_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::memcpy(src_device, src_data, bytes);

    // Target "device" memory (simulating ROCm)
    void *dst_device = mock_backend_->allocate(bytes, 0);
    ASSERT_NE(dst_device, nullptr);

    // Host bounce buffer
    float host_bounce[4] = {0};

    MemoryDescriptor desc;
    desc.host_ptr = host_bounce;
    desc.device_ptr = src_device;
    desc.device = DeviceId::cuda(0);
    desc.size_bytes = bytes;
    desc.residency = MemoryResidency::STANDARD;

    TransferRequest req;
    req.source = desc;
    req.target_device = DeviceId::rocm(0);
    req.method = TransferMethod::HOST_STAGED;
    req.target_ptr = dst_device;

    auto result = engine.execute(req);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::HOST_STAGED);

    // Should have done D2H + H2D
    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.d2h_count, 1u);
    EXPECT_EQ(stats.h2d_count, 1u);

    // Verify data arrived at destination
    auto *result_data = static_cast<float *>(dst_device);
    EXPECT_FLOAT_EQ(result_data[0], 1.0f);
    EXPECT_FLOAT_EQ(result_data[3], 4.0f);

    mock_backend_->free(src_device, 0);
    mock_backend_->free(dst_device, 0);
}

TEST_F(Test__TransferEngine_Execute, Noop_NoBackendCall)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    MemoryDescriptor desc;
    desc.device = DeviceId::cuda(0);
    desc.size_bytes = 100;

    TransferRequest req;
    req.source = desc;
    req.target_device = DeviceId::cuda(0);
    req.method = TransferMethod::NOOP;

    auto result = engine.execute(req);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::NOOP);

    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 0u);
    EXPECT_EQ(stats.d2h_count, 0u);
}

TEST_F(Test__TransferEngine_Execute, MappedNoop_NoBackendCall)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    MemoryDescriptor desc;
    desc.device = DeviceId::cuda(0);
    desc.size_bytes = 100;
    desc.residency = MemoryResidency::MAPPED;

    TransferRequest req;
    req.source = desc;
    req.target_device = DeviceId::cpu();
    req.method = TransferMethod::MAPPED_NOOP;

    auto result = engine.execute(req);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::MAPPED_NOOP);

    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 0u);
    EXPECT_EQ(stats.d2h_count, 0u);
}

// ============================================================================
// HOST_RESIDENT high-level tests — verifies upload/uploadFull/transferActivation
// skip device allocation and transfer for HOST_RESIDENT tensors.
// All paths exit before any backend interaction, so MockBackend is sufficient.
// ============================================================================

TEST_F(Test__TransferEngine_Execute, Upload_HostResident_SkipsTransfer)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    // Create a tensor and mark it HOST_RESIDENT
    auto tensor = TestTensorFactory::createFP32Random({4, 4});
    tensor->setHostResident();

    EXPECT_TRUE(tensor->isHostResident());

    // Upload should succeed as NOOP — no backend calls
    auto result = engine.upload(tensor.get(), DeviceId::cuda(0));
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::NOOP);

    // Verify zero backend interaction
    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 0u);
    EXPECT_EQ(stats.d2h_count, 0u);
}

TEST_F(Test__TransferEngine_Execute, UploadFull_HostResident_SkipsTransfer)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    auto tensor = TestTensorFactory::createFP32Random({8, 8});
    tensor->setHostResident();

    // uploadFull should also NOOP for HOST_RESIDENT
    auto result = engine.uploadFull(tensor.get(), DeviceId::rocm(0));
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::NOOP);

    // Zero backend interaction
    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 0u);
    EXPECT_EQ(stats.d2h_count, 0u);
}

TEST_F(Test__TransferEngine_Execute, TransferActivation_HostResident_SkipsTransfer)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    auto tensor = TestTensorFactory::createFP32Random({2, 2});
    tensor->setHostResident();

    // transferActivation should NOOP for HOST_RESIDENT
    auto result = engine.transferActivation(tensor.get(), DeviceId::cuda(1));
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::NOOP);

    // Zero backend interaction
    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 0u);
    EXPECT_EQ(stats.d2h_count, 0u);
}

TEST_F(Test__TransferEngine_Execute, Upload_HostResident_HostDataStillAccessible)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setHostResident();

    // Upload does nothing for HOST_RESIDENT
    auto result = engine.upload(tensor.get(), DeviceId::cuda(0));
    EXPECT_TRUE(result.success);

    // Host data remains accessible and valid — no device buffer was allocated
    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);
    for (size_t i = 0; i < 16; ++i)
    {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

TEST_F(Test__TransferEngine_Execute, Upload_HostResident_MultipleCallsStillNoop)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    auto tensor = TestTensorFactory::createFP32Random({4, 4});
    tensor->setHostResident();

    // Multiple uploads to different devices should all NOOP
    for (int i = 0; i < 3; ++i)
    {
        auto result = engine.upload(tensor.get(), DeviceId::rocm(i));
        EXPECT_TRUE(result.success);
        EXPECT_EQ(result.method_used, TransferMethod::NOOP);
    }

    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 0u);
    EXPECT_EQ(stats.d2h_count, 0u);
}

// ============================================================================
// Error handling tests
// ============================================================================

TEST_F(Test__TransferEngine_Execute, HostToDevice_NullSourceHostPtr_Fails)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    MemoryDescriptor desc;
    desc.host_ptr = nullptr; // No host data!
    desc.size_bytes = 100;

    TransferRequest req;
    req.source = desc;
    req.target_device = DeviceId::cuda(0);
    req.method = TransferMethod::HOST_TO_DEVICE;
    req.target_ptr = reinterpret_cast<void *>(0xDEAD);

    auto result = engine.execute(req);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error.empty());
}

TEST_F(Test__TransferEngine_Execute, DeviceToHost_NullDevicePtr_Fails)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    float host_buf[4];
    MemoryDescriptor desc;
    desc.host_ptr = host_buf;
    desc.device_ptr = nullptr; // No device data!
    desc.device = DeviceId::cuda(0);
    desc.size_bytes = sizeof(host_buf);

    TransferRequest req;
    req.source = desc;
    req.target_device = DeviceId::cpu();
    req.method = TransferMethod::DEVICE_TO_HOST;

    auto result = engine.execute(req);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error.empty());
}

// ============================================================================
// to_string tests
// ============================================================================

TEST(Test__TransferEngine_Strings, TransferMethodToString)
{
    EXPECT_EQ(to_string(TransferMethod::NOOP), "NOOP");
    EXPECT_EQ(to_string(TransferMethod::HOST_TO_DEVICE), "HOST_TO_DEVICE");
    EXPECT_EQ(to_string(TransferMethod::DEVICE_TO_HOST), "DEVICE_TO_HOST");
    EXPECT_EQ(to_string(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND), "DEVICE_TO_DEVICE_SAME_BACKEND");
    EXPECT_EQ(to_string(TransferMethod::HOST_STAGED), "HOST_STAGED");
    EXPECT_EQ(to_string(TransferMethod::MAPPED_NOOP), "MAPPED_NOOP");
}

// ============================================================================
// MemoryDescriptor tests
// ============================================================================

TEST(Test__TransferEngine_Descriptor, Describe_ContainsDevice)
{
    MemoryDescriptor desc;
    desc.device = DeviceId::cuda(0);
    desc.size_bytes = 1024;
    desc.residency = MemoryResidency::STANDARD;

    std::string s = desc.describe();
    EXPECT_NE(s.find("1024"), std::string::npos);
    EXPECT_NE(s.find("STANDARD"), std::string::npos);
}

TEST(Test__TransferEngine_Descriptor, TransferResult_Ok)
{
    auto r = TransferResult::ok(TransferMethod::HOST_TO_DEVICE, 42);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.method_used, TransferMethod::HOST_TO_DEVICE);
    EXPECT_EQ(r.elapsed_ns, 42u);
    EXPECT_TRUE(r.error.empty());
}

TEST(Test__TransferEngine_Descriptor, TransferResult_Fail)
{
    auto r = TransferResult::fail(TransferMethod::DEVICE_TO_HOST, "test error");
    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.error, "test error");
}

// ============================================================================
// GPU_ONLY planTransfer tests — GPU_ONLY uses standard transfer logic
// ============================================================================

TEST(Test__TransferEngine_Plan, GpuOnly_CpuToCuda_HostToDevice)
{
    // GPU_ONLY does NOT affect planTransfer — data still moves H2D normally.
    // The host release happens AFTER the transfer, not instead of it.
    auto method = TransferEngine::planTransfer(
        DeviceId::cpu(), DeviceId::cuda(0), MemoryResidency::GPU_ONLY);
    EXPECT_EQ(method, TransferMethod::HOST_TO_DEVICE);
}

TEST(Test__TransferEngine_Plan, GpuOnly_CpuToRocm_HostToDevice)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cpu(), DeviceId::rocm(0), MemoryResidency::GPU_ONLY);
    EXPECT_EQ(method, TransferMethod::HOST_TO_DEVICE);
}

TEST(Test__TransferEngine_Plan, GpuOnly_SameDevice_Noop)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cuda(0), DeviceId::cuda(0), MemoryResidency::GPU_ONLY);
    EXPECT_EQ(method, TransferMethod::NOOP);
}

TEST(Test__TransferEngine_Plan, GpuOnly_GpuToCpu_DeviceToHost)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cuda(0), DeviceId::cpu(), MemoryResidency::GPU_ONLY);
    EXPECT_EQ(method, TransferMethod::DEVICE_TO_HOST);
}

// ============================================================================
// GPU_ONLY TensorBase API tests
// ============================================================================

TEST(Test__TransferEngine_GpuOnly, SetGpuOnly_SetsResidency)
{
    auto tensor = TestTensorFactory::createFP32Random({4, 4});
    EXPECT_FALSE(tensor->isGpuOnly());
    EXPECT_EQ(tensor->memoryResidency(), MemoryResidency::STANDARD);

    tensor->setGpuOnly();
    EXPECT_TRUE(tensor->isGpuOnly());
    EXPECT_EQ(tensor->memoryResidency(), MemoryResidency::GPU_ONLY);
}

TEST(Test__TransferEngine_GpuOnly, GpuOnly_MutuallyExclusive_WithHostResident)
{
    auto tensor = TestTensorFactory::createFP32Random({4, 4});

    tensor->setGpuOnly();
    EXPECT_TRUE(tensor->isGpuOnly());
    EXPECT_FALSE(tensor->isHostResident());

    tensor->setHostResident();
    EXPECT_FALSE(tensor->isGpuOnly());
    EXPECT_TRUE(tensor->isHostResident());
}

TEST(Test__TransferEngine_GpuOnly, MemoryResidency_ToString)
{
    EXPECT_EQ(to_string(MemoryResidency::GPU_ONLY), "GPU_ONLY");
}

TEST(Test__TransferEngine_GpuOnly, ReleaseHostWeightData_FreesMemory)
{
    auto tensor = TestTensorFactory::createFP32Ones({32, 32});
    EXPECT_FALSE(tensor->is_raw_data_released());

    tensor->release_host_weight_data();
    EXPECT_TRUE(tensor->is_raw_data_released());
}

TEST(Test__TransferEngine_GpuOnly, ReleaseHostWeightData_Idempotent)
{
    auto tensor = TestTensorFactory::createFP32Ones({32, 32});

    tensor->release_host_weight_data();
    EXPECT_TRUE(tensor->is_raw_data_released());

    // Second call is safe (no-op)
    tensor->release_host_weight_data();
    EXPECT_TRUE(tensor->is_raw_data_released());
}

// ============================================================================
// Event wait failure tests — lock in hard-error behavior for downloadFull
// and fallback behavior for uploadFull
// ============================================================================

namespace
{
    /**
     * @brief MockBackend subclass with configurable event wait failure.
     *
     * Allows tests to make waitForEvent() return false on demand,
     * simulating corrupted or invalid completion events (e.g., events
     * recorded during CUDA graph capture that are invalid for synchronize).
     */
    class FailableEventMockBackend : public MockBackend
    {
    public:
        FailableEventMockBackend() : MockBackend(DeviceType::CUDA) {}

        bool waitForEvent(void *event, int device_id) override
        {
            // Still record the operation for test inspection
            MockBackend::waitForEvent(event, device_id);
            return !fail_event_wait_;
        }

        bool synchronize(int device_id) override
        {
            // Track that synchronize was called as a fallback
            sync_fallback_count_++;
            return !fail_synchronize_;
        }

        /// Make waitForEvent() return false from now on
        void setEventWaitFails(bool fail) { fail_event_wait_ = fail; }

        /// Make synchronize() return false from now on
        void setSynchronizeFails(bool fail) { fail_synchronize_ = fail; }

        /// Number of times synchronize() was called (for verifying fallback behavior)
        size_t getSyncFallbackCount() const { return sync_fallback_count_; }

    private:
        bool fail_event_wait_ = false;
        bool fail_synchronize_ = false;
        size_t sync_fallback_count_ = 0;
    };
} // namespace

class Test__TransferEngine_EventFailure : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mock_ = std::make_shared<FailableEventMockBackend>();

        // Resolver returns our failable mock
        s_failable_mock_ = mock_.get();
        resolver_ = [](DeviceId) -> IBackend *
        {
            return s_failable_mock_;
        };
    }

    /// Helper: set up a tensor on CUDA device with a completion event.
    /// Returns the tensor in DEVICE_AUTHORITATIVE state with a valid event.
    std::unique_ptr<FP32Tensor> createTensorOnDeviceWithEvent()
    {
        auto tensor = TestTensorFactory::createFP32Ones({4, 4});
        tensor->setBackendForTesting(mock_.get());

        // Upload to device (allocates GPU buffer, sets state to SYNCED)
        bool ok = tensor->ensureOnDevice(DeviceId::cuda(0));
        EXPECT_TRUE(ok);

        // Transition to DEVICE_AUTHORITATIVE with a completion event
        // This creates an event and records it on the mock backend
        tensor->transitionToWithEvent(TensorCoherenceState::DEVICE_AUTHORITATIVE,
                                      DeviceId::cuda(0));

        return tensor;
    }

    static FailableEventMockBackend *s_failable_mock_;
    std::shared_ptr<FailableEventMockBackend> mock_;
    TransferEngine::BackendResolver resolver_;
};

FailableEventMockBackend *Test__TransferEngine_EventFailure::s_failable_mock_ = nullptr;

// -----------------------------------------------------------------------------
// downloadFull: event wait failure → HARD ERROR (TransferResult::fail)
// -----------------------------------------------------------------------------

TEST_F(Test__TransferEngine_EventFailure, DownloadFull_EventWaitFail_ReturnsHardError)
{
    TransferEngine engine(resolver_);

    auto tensor = createTensorOnDeviceWithEvent();

    // Now make event wait fail — simulates corrupted event from graph capture
    mock_->setEventWaitFails(true);

    auto result = engine.downloadFull(tensor.get());

    // Must be a hard failure — no silent fallback
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::DEVICE_TO_HOST);
    EXPECT_NE(result.error.find("Event wait failed"), std::string::npos);
}

TEST_F(Test__TransferEngine_EventFailure, DownloadFull_EventWaitFail_ErrorMessageMentionsInvalidEvent)
{
    TransferEngine engine(resolver_);

    auto tensor = createTensorOnDeviceWithEvent();
    tensor->setDebugName("test_attention_output");

    mock_->setEventWaitFails(true);

    auto result = engine.downloadFull(tensor.get());

    EXPECT_FALSE(result.success);
    // Error should mention "invalid" to help diagnosis
    EXPECT_NE(result.error.find("invalid"), std::string::npos);
}

TEST_F(Test__TransferEngine_EventFailure, DownloadFull_EventWaitFail_NoD2HTransferOccurs)
{
    TransferEngine engine(resolver_);

    auto tensor = createTensorOnDeviceWithEvent();

    mock_->setEventWaitFails(true);
    mock_->resetTransferStats();

    engine.downloadFull(tensor.get());

    // The D2H transfer should NOT have occurred — we failed before reaching memcpy
    auto stats = mock_->getTransferStats();
    EXPECT_EQ(stats.d2h_count, 0u);
}

TEST_F(Test__TransferEngine_EventFailure, DownloadFull_EventWaitSuccess_TransferSucceeds)
{
    TransferEngine engine(resolver_);

    auto tensor = createTensorOnDeviceWithEvent();

    // Event wait succeeds (default) — download should work
    mock_->setEventWaitFails(false);
    mock_->resetTransferStats();

    auto result = engine.downloadFull(tensor.get());

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::DEVICE_TO_HOST);

    auto stats = mock_->getTransferStats();
    EXPECT_EQ(stats.d2h_count, 1u);
}

TEST_F(Test__TransferEngine_EventFailure, DownloadFull_NoEvent_FallsBackToFullSync)
{
    TransferEngine engine(resolver_);

    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());

    // Upload to device but DON'T set a completion event
    tensor->ensureOnDevice(DeviceId::cuda(0));
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, DeviceId::cuda(0));

    mock_->resetTransferStats();

    auto result = engine.downloadFull(tensor.get());

    // Should succeed via full device sync (no event to wait on)
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::DEVICE_TO_HOST);

    // synchronize() was called as fallback (no event path)
    EXPECT_GE(mock_->getSyncFallbackCount(), 1u);
}

// -----------------------------------------------------------------------------
// uploadFull: event wait failure → HARD ERROR (same as downloadFull)
// No fallback to full device synchronize — invalid events must fail loudly.
// -----------------------------------------------------------------------------

TEST_F(Test__TransferEngine_EventFailure, UploadFull_EventWaitFail_ReturnsHardError)
{
    TransferEngine engine(resolver_);

    auto tensor = createTensorOnDeviceWithEvent();

    // Event wait fails — simulates corrupted event from graph capture
    mock_->setEventWaitFails(true);

    // uploadFull to the SAME device where tensor already resides (triggers event wait path)
    auto result = engine.uploadFull(tensor.get(), DeviceId::cuda(0));

    // Must be a hard failure — no silent fallback to synchronize
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.error.find("Event wait failed"), std::string::npos);

    // synchronize() must NOT have been called as fallback
    EXPECT_EQ(mock_->getSyncFallbackCount(), 0u);
}

TEST_F(Test__TransferEngine_EventFailure, UploadFull_EventWaitFail_ErrorMessageMentionsInvalidEvent)
{
    TransferEngine engine(resolver_);

    auto tensor = createTensorOnDeviceWithEvent();
    tensor->setDebugName("test_hidden_state");

    mock_->setEventWaitFails(true);

    auto result = engine.uploadFull(tensor.get(), DeviceId::cuda(0));

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.error.find("invalid"), std::string::npos);
}

TEST_F(Test__TransferEngine_EventFailure, UploadFull_EventWaitSuccess_Succeeds)
{
    TransferEngine engine(resolver_);

    auto tensor = createTensorOnDeviceWithEvent();

    // Event wait succeeds
    mock_->setEventWaitFails(false);

    auto result = engine.uploadFull(tensor.get(), DeviceId::cuda(0));

    // Should succeed without needing synchronize fallback
    EXPECT_TRUE(result.success);
    EXPECT_EQ(mock_->getSyncFallbackCount(), 0u);
}
