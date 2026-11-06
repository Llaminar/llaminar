/**
 * @file Test__W4A4Tensor.cpp
 * @brief Native tensor identity, immutable-source boundaries and retained-engine lifecycles.
 *
 * Real PMA allocations feed the public tensor and KernelFactory interfaces.
 * Complete-row aliases and engines must retain one original physical owner;
 * conversions, writes, relabeling and partial-K views cannot bypass admission.
 * Recorded numerical coverage uses W4A4CPUGemmReplayDriver, not these fixtures.
 */
#include "tensors/W4A4Tensor.h"
#include "kernels/KernelFactory.h"
#include "memory/CPUStorageRoot.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <utility>

namespace llaminar2::w4a4
{
    namespace
    {
        constexpr std::size_t kRows = 2, kColumns = 256, kBytes = kRows * sizeof(SourceBlock);
        constexpr auto kOwner = PhysicalMemoryOwner::PrimaryModelWeights;
        const DeviceId kCPU{DeviceType::CPU, 1};

        /** @brief Fixture keeps only the real PMA and tensor; no second live-byte bookkeeping. */
        struct Fixture
        {
            std::shared_ptr<PhysicalMemoryAuthority> memory;
            std::shared_ptr<W4A4Tensor> tensor;
        };

        /** @return A native tensor whose original allocator differs from NUMA first-touch policy. */
        Fixture fixture(Transform transform)
        {
            const CPUStorageRoot<std::uint8_t> root(kBytes, kOwner, kCPU,
                CPUStoragePlacement::local(), CPUStorageInitialization::Uninitialized);
            const auto bytes = root.physicalBytes();
            PhysicalMemoryPlanBuilder builder;
            builder.add(PhysicalMemoryResource{.world_rank = 0, .device = kCPU,
                .total_bytes = bytes, .admission_available_bytes = bytes}, kOwner, bytes);
            auto memory = std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
            auto allocated = root.allocate(memory);
            std::array<SourceBlock, kRows> source{};
            for (auto &block : source)
            {
                block.unit = 0.125f;
                block.descriptor = 0x60001111;
                block.rotation_length = transform == Transform::Identity ? kColumns : 128;
                std::fill_n(block.codes, 128, std::uint8_t{0x11});
            }
            std::memcpy(allocated.data(), source.data(), kBytes);
            return {memory, std::make_shared<W4A4Tensor>(WeightStorage::adopt(
                transform, kRows, kColumns, std::move(allocated), kOwner))};
        }

        /** @return The canonical authority's remaining original physical allocation. */
        std::size_t live(const Fixture &value)
        {
            return value.memory->claimedBytes(kCPU, kOwner, PhysicalMemoryMaterializationKind::NewAllocation);
        }
    }

    TEST(W4A4TensorLifecycle, CompleteRowsAndFactoryEnginesOutliveSourceRetirement)
    {
        for (auto transform : {Transform::Identity, Transform::PaleySylvester})
        {
            auto value = fixture(transform);
            const auto physical = live(value);
            const auto *original = value.tensor->typed_data();
            EXPECT_GE(physical, kBytes);
            EXPECT_EQ(value.tensor->home_device(), kCPU);
            EXPECT_EQ(value.tensor->numel(), kRows * kColumns);
            EXPECT_EQ(value.tensor->size_bytes(), kBytes);
            EXPECT_EQ(value.tensor->raw_data(), original);
            EXPECT_FALSE(value.tensor->is_view());
            EXPECT_FALSE(value.tensor->is_mmap_data());
            auto tail = value.tensor->create_view({1, kColumns}, kColumns);
            EXPECT_TRUE(tail->is_view());
            EXPECT_EQ(tail->raw_data(), original + 1);
            EXPECT_EQ(tail->home_device(), kCPU);
            auto engine = value.tensor->createGemm();
            auto sliced = tail->createGemm();
            EXPECT_EQ(engine->get_n(), 2);
            EXPECT_EQ(sliced->get_n(), 1);
            EXPECT_EQ(sliced->get_k(), kColumns);
            EXPECT_TRUE(engine->canReleaseSourceWeightTensor());
            EXPECT_EQ(engine->packedWeightBytes(), kBytes);
            EXPECT_EQ(sliced->packedWeightBytes(), sizeof(SourceBlock));
            value.tensor->release_raw_data();
            value.tensor->release_host_weight_data();
            EXPECT_TRUE(value.tensor->is_raw_data_released());
            EXPECT_EQ(value.tensor->raw_data(), nullptr);
            EXPECT_EQ(value.tensor->active_data_ptr(), nullptr);
            EXPECT_EQ(value.tensor->size_bytes(), kBytes);
            EXPECT_EQ(value.tensor->shape(), (std::vector<std::size_t>{kRows, kColumns}));
            EXPECT_THROW(static_cast<void>(value.tensor->retainWeightStorage()), std::logic_error);
            EXPECT_THROW(static_cast<void>(value.tensor->typed_data()), std::logic_error);
            EXPECT_THROW(value.tensor->createGemm(), std::logic_error);
            EXPECT_EQ(tail->raw_data(), original + 1);
            tail->release_raw_data();
            value.tensor.reset();
            tail.reset();
            EXPECT_EQ(live(value), physical);
            engine.reset();
            EXPECT_EQ(live(value), physical);
            sliced.reset();
            EXPECT_EQ(live(value), 0u);
        }
    }

    TEST(W4A4TensorContract, NativeIdentityCannotBecomeWeightOnlyArithmeticOrMutableStorage)
    {
        using llaminar::v2::kernels::KernelFactory;
        for (auto transform : {Transform::Identity, Transform::PaleySylvester})
        {
            auto value = fixture(transform);
            auto &tensor = *value.tensor;
            const auto type = transform == Transform::Identity ? TensorType::W4A4 : TensorType::W4A4H;
            EXPECT_EQ(tensor.native_type(), type);
            EXPECT_EQ(tensorTypeFromName(tensorTypeName(type)), type);
            EXPECT_FALSE(isNativeVnniFormat(type));
            EXPECT_FALSE(isInt8VnniFormat(type));
            EXPECT_FALSE(tensor.is_fp32_backed());
            EXPECT_THROW(tensor.data(), std::logic_error);
            EXPECT_THROW(tensor.fp32_data(), std::logic_error);
            EXPECT_THROW(tensor.mutable_data(), std::logic_error);
            EXPECT_THROW(tensor.mutable_fp32_data(), std::logic_error);
            EXPECT_THROW(tensor.raw_mutable_data(), std::logic_error);
            EXPECT_THROW(tensor.active_mutable_data_ptr(), std::logic_error);
            EXPECT_THROW(tensor.copyFrom(&tensor), std::logic_error);
            EXPECT_THROW(tensor.to_fp32(nullptr), std::logic_error);
            EXPECT_THROW(tensor.to_bf16(nullptr), std::logic_error);
            EXPECT_THROW(tensor.to_fp16(nullptr), std::logic_error);
            EXPECT_THROW(tensor.to_int8_blocked(nullptr, nullptr), std::logic_error);
            EXPECT_THROW(tensor.to_int8_perchannel(nullptr, nullptr), std::logic_error);
            EXPECT_THROW(tensor.to_fp32_row(0, nullptr), std::logic_error);
            EXPECT_THROW(tensor.to_fp32_span(0, 1, nullptr), std::logic_error);
            EXPECT_THROW(tensor.transferStorageOwner(), std::logic_error);
            EXPECT_THROW(std::as_const(tensor).transferStorageOwner(), std::logic_error);
            EXPECT_THROW(tensor.ensureOnDevice(DeviceId::cpu()), std::logic_error);
            EXPECT_THROW(tensor.allocateOnDevice(DeviceId::cpu()), std::logic_error);
            EXPECT_TRUE(tensor.ensureOnHost());
            EXPECT_THROW(tensor.setLayout(TensorLayout::UNKNOWN), std::invalid_argument);
            EXPECT_THROW(tensor.create_view({kRows * kColumns}), std::invalid_argument);
            EXPECT_THROW(tensor.create_view({1, kColumns / 2}), std::invalid_argument);
            EXPECT_THROW(tensor.create_view({1, kColumns}, 1), std::invalid_argument);
            EXPECT_THROW(tensor.create_view({0, kColumns}), std::out_of_range);
            EXPECT_THROW(tensor.create_view({1, kColumns}, kRows * kColumns), std::out_of_range);
            EXPECT_THROW(KernelFactory::createGemm(&tensor, DeviceType::CUDA, 0), std::invalid_argument);
            EXPECT_THROW(KernelFactory::createGemm(&tensor, DeviceType::ROCm, 0), std::invalid_argument);
            EXPECT_THROW(KernelFactory::createGemm(&tensor, DeviceType::CPU, 0), std::invalid_argument);
            EXPECT_THROW(KernelFactory::createGemm(static_cast<W4A4Tensor *>(nullptr), DeviceType::CPU), std::invalid_argument);
            value.tensor.reset();
            EXPECT_EQ(live(value), 0u);
        }
    }
} // namespace llaminar2::w4a4
