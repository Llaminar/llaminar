#include <gtest/gtest.h>

#include "tensors/CoherenceState.h"
#include "transfer/TransferEngine.h"
#include "transfer/TransferMethod.h"

// For execute tests
#include "backends/DeviceId.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/TensorClasses.h"
#include "tensors/TensorSlice.h"
#include "../../mocks/MockBackend.h"
#include "../../mocks/MockWorkerGPUContext.h"
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
        llaminar2::testing::installHardwareFreeGPUContextFactories();
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

TEST_F(Test__TransferEngine_Execute, Upload_TensorSliceMutatesBackingStorageOwner)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    auto inner = TestTensorFactory::createFP32Ones({4, 4});
    inner->setBackendForTesting(mock_backend_.get());
    TensorBase *inner_ptr = inner.get();

    SliceMetadata metadata{
        .mode = SliceMode::ROW_PARALLEL,
        .original_rows = 4,
        .original_cols = 4,
        .slice_start = 0,
        .slice_end = 4,
        .rank = 0,
        .world_size = 1,
        .inner_is_presliced = true,
    };
    std::unique_ptr<TensorBase> storage_owner = std::move(inner);
    TensorSlice slice(std::move(storage_owner), metadata);

    const auto result = engine.upload(&slice, DeviceId::cuda(0));

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.method_used, TransferMethod::HOST_TO_DEVICE);
    EXPECT_EQ(slice.transferStorageOwner(), inner_ptr);
    EXPECT_EQ(slice.current_device(), DeviceId::cuda(0));
    EXPECT_NE(slice.gpu_data_ptr(), nullptr);
    EXPECT_EQ(slice.gpu_data_ptr(), inner_ptr->gpu_data_ptr());

    const auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 1u);
    EXPECT_EQ(stats.d2h_count, 0u);
}

TEST_F(Test__TransferEngine_Execute, Download_TensorSliceReadsBackingStorageOwner)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    auto inner = TestTensorFactory::createFP32Ones({4, 4});
    inner->setBackendForTesting(mock_backend_.get());
    TensorBase *inner_ptr = inner.get();

    SliceMetadata metadata{
        .mode = SliceMode::ROW_PARALLEL,
        .original_rows = 4,
        .original_cols = 4,
        .slice_start = 0,
        .slice_end = 4,
        .rank = 0,
        .world_size = 1,
        .inner_is_presliced = true,
    };
    std::unique_ptr<TensorBase> storage_owner = std::move(inner);
    TensorSlice slice(std::move(storage_owner), metadata);

    ASSERT_TRUE(engine.upload(&slice, DeviceId::cuda(0)).success);
    TransferEngine::publishDeviceWrite(
        &slice,
        DeviceId::cuda(0),
        reinterpret_cast<void *>(0x1234));
    mock_backend_->resetTransferStats();

    const auto result = engine.download(&slice);

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.method_used, TransferMethod::DEVICE_TO_HOST);
    EXPECT_EQ(slice.transferStorageOwner(), inner_ptr);
    EXPECT_TRUE(slice.hostValid());

    const auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 0u);
    EXPECT_EQ(stats.d2h_count, 1u);
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
// copyActivation() tests — tensor→tensor copy with transport auto-selection.
//
// GPU-buffer paths (same-device D2D, same-vendor peer copy, cross-vendor
// host-staging) require real device backends and are exercised in the
// integration suite (integration/transfer/Test__TransferEngine_CopyActivation).
// Here we cover the host/CPU-destination path and the argument guards, which
// are fully deterministic without a GPU.
// ============================================================================

TEST_F(Test__TransferEngine_Execute, CopyActivation_NullSrc_Fails)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    auto dst = TestTensorFactory::createFP32({4, 4});
    auto result = engine.copyActivation(nullptr, dst.get(), DeviceId::cpu(), 64);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error.empty());
}

TEST_F(Test__TransferEngine_Execute, CopyActivation_NullDst_Fails)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    auto src = TestTensorFactory::createFP32({4, 4});
    auto result = engine.copyActivation(src.get(), nullptr, DeviceId::cpu(), 64);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error.empty());
}

TEST_F(Test__TransferEngine_Execute, CopyActivation_ZeroBytes_Noop)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    auto src = TestTensorFactory::createFP32({4, 4});
    auto dst = TestTensorFactory::createFP32({4, 4});

    auto result = engine.copyActivation(src.get(), dst.get(), DeviceId::cpu(), 0);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::NOOP);

    // No backend interaction for a zero-byte copy
    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 0u);
    EXPECT_EQ(stats.d2h_count, 0u);
    EXPECT_EQ(stats.d2d_count, 0u);
}

TEST_F(Test__TransferEngine_Execute, CopyActivation_CpuDestination_HostMemcpy)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    // Source filled with a known pattern; destination starts zeroed.
    auto src = TestTensorFactory::createFP32({4, 4});
    auto dst = TestTensorFactory::createFP32({4, 4});
    float *src_data = src->mutable_data();
    float *dst_data = dst->mutable_data();
    for (size_t i = 0; i < src->numel(); ++i)
    {
        src_data[i] = static_cast<float>(i) + 0.5f;
        dst_data[i] = 0.0f;
    }

    const size_t bytes = src->numel() * sizeof(float);
    auto result = engine.copyActivation(src.get(), dst.get(), DeviceId::cpu(), bytes);

    EXPECT_TRUE(result.success);

    // CPU destination is a pure host-side copy: no device transfers occur.
    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 0u);
    EXPECT_EQ(stats.d2h_count, 0u);
    EXPECT_EQ(stats.d2d_count, 0u);

    // Data must have landed in the destination host buffer.
    const float *out = dst->data();
    for (size_t i = 0; i < dst->numel(); ++i)
    {
        EXPECT_FLOAT_EQ(out[i], static_cast<float>(i) + 0.5f);
    }
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
// Event publication/wait failures — every required dependency fails closed.
// ============================================================================

namespace
{
    /// Non-null sentinel used as a mock producer stream without GPU work.
    void *mockProducerStream()
    {
        return reinterpret_cast<void *>(0x5055424C);
    }

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
            host_event_wait_count_++;
            return !fail_event_wait_;
        }

        bool streamWaitEvent(void *stream, void *event, int device_id) override
        {
            MockBackend::streamWaitEvent(stream, event, device_id);
            stream_event_wait_count_++;
            return !fail_stream_event_wait_;
        }

        void *createEvent(int device_id) override
        {
            if (fail_event_create_)
                return nullptr;
            return MockBackend::createEvent(device_id);
        }

        bool recordEvent(void *event, int device_id, void *stream = nullptr) override
        {
            MockBackend::recordEvent(event, device_id, stream);
            return !fail_event_record_;
        }

        bool synchronize(int device_id) override
        {
            // Track that synchronize was called as a fallback
            sync_fallback_count_++;
            return !fail_synchronize_;
        }

        /// Make waitForEvent() return false from now on
        void setEventWaitFails(bool fail) { fail_event_wait_ = fail; }

        /// Make streamWaitEvent() return false from now on
        void setStreamEventWaitFails(bool fail) { fail_stream_event_wait_ = fail; }

        /// Make createEvent() fail from now on
        void setEventCreateFails(bool fail) { fail_event_create_ = fail; }

        /// Make recordEvent() fail from now on
        void setEventRecordFails(bool fail) { fail_event_record_ = fail; }

        /// Make synchronize() return false from now on
        void setSynchronizeFails(bool fail) { fail_synchronize_ = fail; }

        /// Number of times synchronize() was called (for verifying fallback behavior)
        size_t getSyncFallbackCount() const { return sync_fallback_count_; }

        size_t getHostEventWaitCount() const { return host_event_wait_count_; }
        size_t getStreamEventWaitCount() const { return stream_event_wait_count_; }

    private:
        bool fail_event_wait_ = false;
        bool fail_stream_event_wait_ = false;
        bool fail_event_create_ = false;
        bool fail_event_record_ = false;
        bool fail_synchronize_ = false;
        size_t sync_fallback_count_ = 0;
        size_t host_event_wait_count_ = 0;
        size_t stream_event_wait_count_ = 0;
    };
} // namespace

class Test__TransferEngine_EventFailure : public ::testing::Test
{
protected:
    void SetUp() override
    {
        llaminar2::testing::installHardwareFreeGPUContextFactories();
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

        // Publish DEVICE_AUTHORITATIVE with a completion event on the mock backend.
        TransferEngine::publishDeviceWrite(
            tensor,
            DeviceId::cuda(0),
            mockProducerStream());

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

TEST_F(Test__TransferEngine_EventFailure, DownloadFull_ExplicitStreamBypassesStaleCompletionEvent)
{
    TransferEngine engine(resolver_);

    auto tensor = createTensorOnDeviceWithEvent();
    mock_->setEventWaitFails(true);
    mock_->resetTransferStats();
    mock_->resetEventRecords();

    void *producer_stream = reinterpret_cast<void *>(0x1234);
    auto result = engine.downloadFull(tensor.get(), producer_stream);

    EXPECT_TRUE(result.success)
        << "An explicit producer stream should order D2H directly and must not "
           "host-wait a stale graph-capture completion event";
    EXPECT_EQ(result.method_used, TransferMethod::DEVICE_TO_HOST);
    EXPECT_EQ(mock_->getEventWaitCount(), 0u)
        << "Explicit-stream publication must bypass tensor completion events.";
    auto stats = mock_->getTransferStats();
    EXPECT_EQ(stats.d2h_count, 1u);
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

TEST_F(Test__TransferEngine_EventFailure, DownloadFull_NoEvent_FailsClosedWithoutTransfer)
{
    TransferEngine engine(resolver_);

    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());

    // Upload to device but DON'T set a completion event
    tensor->ensureOnDevice(DeviceId::cuda(0));
    TransferEngine::publishGraphOwnedDeviceWrite(tensor, DeviceId::cuda(0));

    mock_->resetTransferStats();

    auto result = engine.downloadFull(tensor.get());

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::DEVICE_TO_HOST);
    EXPECT_NE(result.error.find("no completion event"), std::string::npos);

    const auto stats = mock_->getTransferStats();
    EXPECT_EQ(stats.d2h_count, 0u);
    EXPECT_EQ(mock_->getSyncFallbackCount(), 0u);
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

TEST_F(Test__TransferEngine_EventFailure, UploadFull_StreamWaitFail_DoesNotHostWait)
{
    TransferEngine engine(resolver_);

    auto tensor = createTensorOnDeviceWithEvent();
    mock_->setStreamEventWaitFails(true);

    auto result = engine.uploadFull(
        tensor.get(),
        DeviceId::cuda(0),
        reinterpret_cast<void *>(0x1234));

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.error.find("Stream event wait failed"), std::string::npos);
    EXPECT_EQ(mock_->getStreamEventWaitCount(), 1u);
    EXPECT_EQ(mock_->getHostEventWaitCount(), 0u)
        << "A failed device-side dependency must never degrade into a "
           "host-blocking event wait.";
    EXPECT_EQ(mock_->getSyncFallbackCount(), 0u);
}

TEST_F(Test__TransferEngine_EventFailure, PublicationCreateFail_DoesNotPublishDeviceAuthority)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    ASSERT_TRUE(tensor->ensureOnDevice(DeviceId::cuda(0)));
    const TensorCoherenceState state_before = tensor->coherenceState();

    mock_->setEventCreateFails(true);
    EXPECT_THROW(
        TransferEngine::publishDeviceWrite(
            tensor,
            DeviceId::cuda(0),
            mockProducerStream()),
        std::runtime_error);
    EXPECT_EQ(tensor->coherenceState(), state_before)
        << "Authority cannot become externally visible without its event.";
}

TEST_F(Test__TransferEngine_EventFailure, NullProducerStream_DoesNotPublishDeviceAuthority)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    ASSERT_TRUE(tensor->ensureOnDevice(DeviceId::cuda(0)));
    const TensorCoherenceState state_before = tensor->coherenceState();
    mock_->resetEventRecords();

    EXPECT_THROW(
        TransferEngine::publishDeviceWrite(
            tensor,
            DeviceId::cuda(0),
            nullptr),
        std::invalid_argument);
    EXPECT_EQ(tensor->coherenceState(), state_before);
    EXPECT_EQ(mock_->getEventCreateCount(), 0u);
    EXPECT_EQ(mock_->getEventRecordCount(), 0u);
}

TEST_F(Test__TransferEngine_EventFailure, NullInputPreparationStreamFailsBeforePlacement)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());

    EXPECT_THROW(
        TransferEngine::prepareDeviceInput(
            tensor.get(),
            DeviceId::cuda(0),
            nullptr),
        std::invalid_argument);
    EXPECT_FALSE(tensor->current_device().has_value());
}

TEST_F(
    Test__TransferEngine_EventFailure,
    RequireDeviceInputRejectsHostOnlyTensorWithoutAllocatingOrUploading)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    const size_t allocations_before = mock_->getAllocationCount();
    const auto transfers_before = mock_->getTransferStats();

    EXPECT_THROW(
        TransferEngine::requireDeviceInput(
            tensor.get(),
            DeviceId::cuda(0),
            reinterpret_cast<void *>(0x1234)),
        std::runtime_error);

    EXPECT_EQ(mock_->getAllocationCount(), allocations_before);
    const auto transfers_after = mock_->getTransferStats();
    EXPECT_EQ(transfers_after.h2d_count, transfers_before.h2d_count);
    EXPECT_EQ(transfers_after.d2h_count, transfers_before.d2h_count);
    EXPECT_FALSE(tensor->current_device().has_value());
}

TEST_F(
    Test__TransferEngine_EventFailure,
    RequireDeviceInputJoinsProducerEventWithoutTransfer)
{
    auto tensor = createTensorOnDeviceWithEvent();
    const size_t allocations_before = mock_->getAllocationCount();
    const auto transfers_before = mock_->getTransferStats();

    TransferEngine::requireDeviceInput(
        tensor.get(),
        DeviceId::cuda(0),
        reinterpret_cast<void *>(0x1234));

    EXPECT_EQ(mock_->getStreamEventWaitCount(), 1u);
    EXPECT_EQ(mock_->getHostEventWaitCount(), 0u);
    EXPECT_EQ(mock_->getAllocationCount(), allocations_before);
    const auto transfers_after = mock_->getTransferStats();
    EXPECT_EQ(transfers_after.h2d_count, transfers_before.h2d_count);
    EXPECT_EQ(transfers_after.d2h_count, transfers_before.d2h_count);
}

TEST_F(
    Test__TransferEngine_EventFailure,
    PreCaptureJoinMakesExactEventAndStreamCaptureSafe)
{
    auto tensor = createTensorOnDeviceWithEvent();
    void *capture_stream = reinterpret_cast<void *>(0xCA970001);

    TransferEngine::requireDeviceInput(
        tensor.get(),
        DeviceId::cuda(0),
        capture_stream);
    ASSERT_EQ(mock_->getStreamEventWaitCount(), 1u);

    {
        GraphCaptureGuard capture_guard;
        EXPECT_NO_THROW(
            TransferEngine::requireDeviceInput(
                tensor.get(),
                DeviceId::cuda(0),
                capture_stream));
    }

    EXPECT_EQ(mock_->getStreamEventWaitCount(), 1u)
        << "Capture must consume the prejoined dependency without importing "
           "the external producer event into the graph.";
}

TEST_F(
    Test__TransferEngine_EventFailure,
    CaptureRejectsUnpreparedOrDifferentConsumerStreamWithoutBackendWait)
{
    auto tensor = createTensorOnDeviceWithEvent();
    void *prepared_stream = reinterpret_cast<void *>(0xCA970001);
    void *different_stream = reinterpret_cast<void *>(0xCA970002);

    TransferEngine::requireDeviceInput(
        tensor.get(),
        DeviceId::cuda(0),
        prepared_stream);
    ASSERT_EQ(mock_->getStreamEventWaitCount(), 1u);

    {
        GraphCaptureGuard capture_guard;
        EXPECT_THROW(
            TransferEngine::requireDeviceInput(
                tensor.get(),
                DeviceId::cuda(0),
                different_stream),
            std::runtime_error);
    }

    EXPECT_EQ(mock_->getStreamEventWaitCount(), 1u)
        << "A missing pre-capture dependency is fatal; capture must never try "
           "the backend wait and hope the runtime accepts it.";
}

TEST_F(
    Test__TransferEngine_EventFailure,
    NewDevicePublicationInvalidatesPriorPreCaptureJoin)
{
    auto tensor = createTensorOnDeviceWithEvent();
    void *capture_stream = reinterpret_cast<void *>(0xCA970001);

    TransferEngine::requireDeviceInput(
        tensor.get(),
        DeviceId::cuda(0),
        capture_stream);
    ASSERT_EQ(mock_->getStreamEventWaitCount(), 1u);

    TransferEngine::publishDeviceWrite(
        tensor.get(),
        DeviceId::cuda(0),
        mockProducerStream());

    {
        GraphCaptureGuard capture_guard;
        EXPECT_THROW(
            TransferEngine::requireDeviceInput(
                tensor.get(),
                DeviceId::cuda(0),
                capture_stream),
            std::runtime_error);
    }

    EXPECT_EQ(mock_->getStreamEventWaitCount(), 1u);
}

TEST_F(
    Test__TransferEngine_EventFailure,
    RequireDeviceInputFailsClosedWhenProducerEventCannotBeJoined)
{
    auto tensor = createTensorOnDeviceWithEvent();
    mock_->setStreamEventWaitFails(true);

    EXPECT_THROW(
        TransferEngine::requireDeviceInput(
            tensor.get(),
            DeviceId::cuda(0),
            reinterpret_cast<void *>(0x1234)),
        std::runtime_error);

    EXPECT_EQ(mock_->getStreamEventWaitCount(), 1u);
    EXPECT_EQ(mock_->getHostEventWaitCount(), 0u);
    EXPECT_EQ(mock_->getSyncFallbackCount(), 0u);
}

TEST_F(
    Test__TransferEngine_EventFailure,
    RequireDeviceInputRejectsNullConsumerStreamBeforeResidencyInspection)
{
    auto tensor = createTensorOnDeviceWithEvent();

    EXPECT_THROW(
        TransferEngine::requireDeviceInput(
            tensor.get(),
            DeviceId::cuda(0),
            nullptr),
        std::invalid_argument);
    EXPECT_EQ(mock_->getStreamEventWaitCount(), 0u);
}

TEST_F(Test__TransferEngine_EventFailure, NullOutputPreparationStreamFailsBeforeAllocation)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());

    EXPECT_THROW(
        TransferEngine::prepareDeviceOutput(
            tensor.get(),
            DeviceId::cuda(0),
            nullptr),
        std::invalid_argument);
    EXPECT_FALSE(tensor->current_device().has_value());
}

TEST_F(
    Test__TransferEngine_EventFailure,
    RequireDeviceOutputAcceptsInvalidPreallocatedBytesWithoutAllocation)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    TransferEngine::allocateDeviceStorage(
        tensor.get(),
        DeviceId::cuda(0));
    ASSERT_FALSE(tensor->deviceValid());
    const size_t allocations_before = mock_->getAllocationCount();

    EXPECT_NO_THROW(
        TransferEngine::requireDeviceOutput(
            tensor.get(),
            DeviceId::cuda(0),
            reinterpret_cast<void *>(0x0A117001)));

    EXPECT_EQ(mock_->getAllocationCount(), allocations_before)
        << "Execution-time output validation must never allocate.";
    EXPECT_FALSE(tensor->deviceValid())
        << "Storage validation must not publish unwritten bytes.";
}

TEST_F(
    Test__TransferEngine_EventFailure,
    CaptureRejectsPlacementCapableInputEvenWhenDeviceBytesAreValid)
{
    auto tensor = createTensorOnDeviceWithEvent();
    GraphCaptureGuard capture_guard;

    EXPECT_THROW(
        TransferEngine::prepareDeviceInput(
            tensor.get(),
            DeviceId::cuda(0),
            reinterpret_cast<void *>(0x0A117002)),
        std::logic_error)
        << "A no-op placement call can still import stale generation state into capture.";
}

TEST_F(
    Test__TransferEngine_EventFailure,
    CaptureRejectsPlacementCapableOutputEvenWhenStorageExists)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    TransferEngine::allocateDeviceStorage(
        tensor.get(),
        DeviceId::cuda(0));
    const size_t allocations_before = mock_->getAllocationCount();
    GraphCaptureGuard capture_guard;

    EXPECT_THROW(
        TransferEngine::prepareDeviceOutput(
            tensor.get(),
            DeviceId::cuda(0),
            reinterpret_cast<void *>(0x0A117003)),
        std::logic_error);
    EXPECT_EQ(mock_->getAllocationCount(), allocations_before);
}

TEST_F(Test__TransferEngine_EventFailure, AllocationOnlyStorageDoesNotPublishAuthority)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    mock_->resetEventRecords();

    TransferEngine::allocateDeviceStorage(
        tensor.get(),
        DeviceId::cuda(0));

    ASSERT_TRUE(tensor->current_device().has_value());
    EXPECT_EQ(*tensor->current_device(), DeviceId::cuda(0));
    EXPECT_NE(tensor->gpu_data_ptr(), nullptr);
    EXPECT_FALSE(tensor->deviceValid())
        << "Allocation alone must not publish unwritten device bytes";
    EXPECT_EQ(mock_->getEventCreateCount(), 0u);
    EXPECT_EQ(mock_->getEventRecordCount(), 0u);
}

TEST_F(
    Test__TransferEngine_EventFailure,
    CaptureLedgerAdmitsOnlyAnEarlierRecordedInternalProducer)
{
    auto internal = TestTensorFactory::createFP32Ones({4, 4});
    auto external = TestTensorFactory::createFP32Ones({4, 4});
    internal->setBackendForTesting(mock_.get());
    external->setBackendForTesting(mock_.get());
    TransferEngine::allocateDeviceStorage(internal.get(), DeviceId::cuda(0));
    TransferEngine::allocateDeviceStorage(external.get(), DeviceId::cuda(0));
    ASSERT_FALSE(internal->deviceValid());
    ASSERT_FALSE(external->deviceValid());

    int producer_stage = 0;
    int consumer_stage = 0;
    void *capture_stream = reinterpret_cast<void *>(0xCA970001);
    std::vector<GraphCaptureDependencyLedger::StagePlan> stages;
    stages.push_back({
        .stage_identity = &producer_stage,
        .stage_name = "producer",
        .outputs = {internal->transferStorageOwner()},
    });
    stages.push_back({
        .stage_identity = &consumer_stage,
        .stage_name = "consumer",
        .external_inputs = {external->transferStorageOwner()},
        .internal_inputs = {{
            .tensor = internal->transferStorageOwner(),
            .producer_stage_index = 0,
        }},
    });
    GraphCaptureDependencyLedger ledger(
        DeviceId::cuda(0), capture_stream, std::move(stages), "unit_capture");

    {
        GraphCaptureGuard capture_guard(&ledger);
        {
            ScopedGraphCaptureStage producer_scope(&producer_stage);
            EXPECT_NO_THROW(TransferEngine::publishDeviceWrite(
                internal.get(), DeviceId::cuda(0), capture_stream));
            EXPECT_FALSE(internal->deviceValid())
                << "Recording a producer must not publish unexecuted bytes";
            EXPECT_EQ(mock_->getEventRecordCount(), 0u)
                << "Captured stage publication must not create per-tensor events";
            producer_scope.complete();
        }
        {
            ScopedGraphCaptureStage consumer_scope(&consumer_stage);
            EXPECT_NO_THROW(TransferEngine::requireDeviceInput(
                internal.get(), DeviceId::cuda(0), capture_stream));
            EXPECT_THROW(
                TransferEngine::requireDeviceInput(
                    external.get(), DeviceId::cuda(0), capture_stream),
                std::runtime_error)
                << "An allocated external tensor still needs globally valid bytes";
            consumer_scope.complete();
        }
    }

    EXPECT_FALSE(internal->deviceValid());
    EXPECT_THROW(
        TransferEngine::requireDeviceInput(
            internal.get(), DeviceId::cuda(0), capture_stream),
        std::runtime_error)
        << "The internal-edge proof must not escape its capture transaction";
}

TEST_F(
    Test__TransferEngine_EventFailure,
    CaptureLedgerRejectsInternalInputWhoseProducerIsNotEarlier)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    TransferEngine::allocateDeviceStorage(tensor.get(), DeviceId::cuda(0));

    int invalid_consumer_stage = 0;
    void *capture_stream = reinterpret_cast<void *>(0xCA970002);
    std::vector<GraphCaptureDependencyLedger::StagePlan> stages = {{
        .stage_identity = &invalid_consumer_stage,
        .stage_name = "consumer_before_producer",
        .internal_inputs = {{
            .tensor = tensor->transferStorageOwner(),
            .producer_stage_index = 0,
        }},
    }};
    GraphCaptureDependencyLedger ledger(
        DeviceId::cuda(0), capture_stream, std::move(stages), "invalid_order");

    GraphCaptureGuard capture_guard(&ledger);
    ScopedGraphCaptureStage consumer_scope(&invalid_consumer_stage);
    EXPECT_THROW(
        TransferEngine::requireDeviceInput(
            tensor.get(), DeviceId::cuda(0), capture_stream),
        std::logic_error);
    consumer_scope.complete();
}

TEST_F(
    Test__TransferEngine_EventFailure,
    CaptureLedgerRejectsDifferentConsumerStream)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    TransferEngine::allocateDeviceStorage(tensor.get(), DeviceId::cuda(0));

    int producer_stage = 0;
    int consumer_stage = 0;
    void *capture_stream = reinterpret_cast<void *>(0xCA970003);
    void *different_stream = reinterpret_cast<void *>(0xCA970004);
    std::vector<GraphCaptureDependencyLedger::StagePlan> stages = {
        {
            .stage_identity = &producer_stage,
            .stage_name = "producer",
            .outputs = {tensor->transferStorageOwner()},
        },
        {
            .stage_identity = &consumer_stage,
            .stage_name = "consumer",
            .internal_inputs = {{
                .tensor = tensor->transferStorageOwner(),
                .producer_stage_index = 0,
            }},
        },
    };
    GraphCaptureDependencyLedger ledger(
        DeviceId::cuda(0), capture_stream, std::move(stages), "stream_identity");

    GraphCaptureGuard capture_guard(&ledger);
    {
        ScopedGraphCaptureStage producer_scope(&producer_stage);
        producer_scope.complete();
    }
    {
        ScopedGraphCaptureStage consumer_scope(&consumer_stage);
        EXPECT_THROW(
            TransferEngine::requireDeviceInput(
                tensor.get(), DeviceId::cuda(0), different_stream),
            std::logic_error);
        consumer_scope.complete();
    }
}

TEST_F(Test__TransferEngine_EventFailure, CurrentDevicePublicationRejectsNullStream)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    ASSERT_TRUE(tensor->ensureOnDevice(DeviceId::cuda(0)));
    const TensorCoherenceState state_before = tensor->coherenceState();
    mock_->resetEventRecords();

    EXPECT_THROW(
        TransferEngine::publishCurrentDeviceWrite(tensor, nullptr),
        std::invalid_argument);
    EXPECT_EQ(tensor->coherenceState(), state_before);
    EXPECT_EQ(mock_->getEventCreateCount(), 0u);
    EXPECT_EQ(mock_->getEventRecordCount(), 0u);
}

TEST_F(Test__TransferEngine_EventFailure, PublicationRecordFail_DoesNotPublishDeviceAuthority)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    ASSERT_TRUE(tensor->ensureOnDevice(DeviceId::cuda(0)));
    const TensorCoherenceState state_before = tensor->coherenceState();

    mock_->setEventRecordFails(true);
    EXPECT_THROW(
        TransferEngine::publishDeviceWrite(
            tensor.get(),
            DeviceId::cuda(0),
            reinterpret_cast<void *>(0x1234)),
        std::runtime_error);
    EXPECT_EQ(tensor->coherenceState(), state_before)
        << "A failed record must not expose an eventless GPU write.";
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
