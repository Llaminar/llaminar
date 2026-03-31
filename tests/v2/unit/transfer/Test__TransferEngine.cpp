#include <gtest/gtest.h>

#include "tensors/CoherenceState.h"
#include "transfer/TransferEngine.h"
#include "transfer/TransferMethod.h"

// For execute tests
#include "backends/DeviceId.h"
#include "tensors/TensorClasses.h"
#include "../../mocks/MockBackend.h"

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

TEST(Test__TransferEngine_Plan, CudaToRocm_BarBacked_HostBounce)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cuda(0), DeviceId::rocm(0), MemoryResidency::BAR_BACKED);
    EXPECT_EQ(method, TransferMethod::BAR_HOST_BOUNCE);
}

TEST(Test__TransferEngine_Plan, RocmToCuda_BarBacked_HostBounce)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::rocm(0), DeviceId::cuda(0), MemoryResidency::BAR_BACKED);
    EXPECT_EQ(method, TransferMethod::BAR_HOST_BOUNCE);
}

TEST(Test__TransferEngine_Plan, CudaToRocm_NoBar_HostStaged)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cuda(0), DeviceId::rocm(0), MemoryResidency::STANDARD);
    EXPECT_EQ(method, TransferMethod::HOST_STAGED);
}

TEST(Test__TransferEngine_Plan, RocmToCuda_NoBar_HostStaged)
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

TEST(Test__TransferEngine_Plan, DescribeTransferPlan_HumanReadable)
{
    auto desc = TransferEngine::describeTransferPlan(
        DeviceId::cpu(), DeviceId::cuda(0), MemoryResidency::STANDARD);

    // Should contain source, destination, method
    EXPECT_NE(desc.find("HOST_TO_DEVICE"), std::string::npos);
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

TEST_F(Test__TransferEngine_Execute, BarHostBounce_UsesStagingPtr)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    constexpr size_t bytes = 4 * sizeof(float);

    // Staging buffer (simulating HIP VRAM)
    void *staging = mock_backend_->allocate(bytes, 0);
    ASSERT_NE(staging, nullptr);
    float staging_data[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    std::memcpy(staging, staging_data, bytes);

    // Target "device" memory
    void *dst_device = mock_backend_->allocate(bytes, 0);
    ASSERT_NE(dst_device, nullptr);

    // Host bounce buffer
    float host_bounce[4] = {0};

    MemoryDescriptor desc;
    desc.host_ptr = host_bounce;
    desc.device_ptr = nullptr; // BAR tensors may not have generic device_ptr
    desc.device = DeviceId::rocm(0);
    desc.size_bytes = bytes;
    desc.residency = MemoryResidency::BAR_BACKED;
    desc.bar_staging_ptr = staging;
    desc.bar_host_device = DeviceId::rocm(0);

    TransferRequest req;
    req.source = desc;
    req.target_device = DeviceId::cuda(0);
    req.method = TransferMethod::BAR_HOST_BOUNCE;
    req.target_ptr = dst_device;

    auto result = engine.execute(req);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::BAR_HOST_BOUNCE);

    // Should have done D2H from staging + H2D to target
    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.d2h_count, 1u);
    EXPECT_EQ(stats.h2d_count, 1u);

    // Data should have arrived
    auto *result_data = static_cast<float *>(dst_device);
    EXPECT_FLOAT_EQ(result_data[0], 5.0f);
    EXPECT_FLOAT_EQ(result_data[3], 8.0f);

    mock_backend_->free(staging, 0);
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
    EXPECT_EQ(to_string(TransferMethod::BAR_HOST_BOUNCE), "BAR_HOST_BOUNCE");
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

TEST(Test__TransferEngine_Descriptor, Describe_BarBacked_ShowsBarFields)
{
    MemoryDescriptor desc;
    desc.device = DeviceId::rocm(0);
    desc.size_bytes = 2048;
    desc.residency = MemoryResidency::BAR_BACKED;
    desc.bar_staging_ptr = reinterpret_cast<void *>(0x1);

    std::string s = desc.describe();
    EXPECT_NE(s.find("BAR_BACKED"), std::string::npos);
    EXPECT_NE(s.find("bar_staging=1"), std::string::npos);
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
