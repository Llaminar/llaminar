/**
 * @file Test__MoEPhaseD_NoGlobalRegistry.cpp
 * @brief Phase D unit tests: verify prepareExpertGemmLocal() does NOT register
 *        in KernelFactory's global prepared_gemm_registry_.
 *
 * Tests:
 *   1. prepareExpertGemmLocal returns a valid engine (shared_ptr)
 *   2. No global registry entry created after prepareExpertGemmLocal
 *   3. Engine lifetime governed solely by shared_ptr ownership
 *   4. Multiple experts get independent engine instances
 *   5. GPU device correctly rejected (returns nullptr)
 *   6. Null tensor returns nullptr (no throw)
 *   7. Contrast: explicit transfer registration is global, prepareExpertGemmLocal is not
 */

#include <gtest/gtest.h>

#include "kernels/KernelFactory.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h"
#include "backends/DeviceId.h"
#include "utils/TestTensorFactory.h"

using namespace llaminar2;
using KF = llaminar::v2::kernels::KernelFactory;
using namespace llaminar2::test;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class Test__MoEPhaseD_NoGlobalRegistry : public ::testing::Test
{
protected:
    void TearDown() override
    {
        KF::clearCache();
    }
};

// ---------------------------------------------------------------------------
// 1. prepareExpertGemmLocal returns a valid engine
// ---------------------------------------------------------------------------

TEST_F(Test__MoEPhaseD_NoGlobalRegistry, LocalPrepReturnsValidEngine)
{
    auto tensor = TestTensorFactory::createFP32Random({64, 128});
    ASSERT_NE(tensor, nullptr);

    auto engine = KF::prepareExpertGemmLocal(tensor.get(), DeviceId::cpu());
    ASSERT_NE(engine, nullptr) << "prepareExpertGemmLocal should return a valid engine for FP32 CPU";
}

// ---------------------------------------------------------------------------
// 2. prepareExpertGemmLocal does NOT register in the global registry
// ---------------------------------------------------------------------------

TEST_F(Test__MoEPhaseD_NoGlobalRegistry, LocalPrepDoesNotRegisterGlobally)
{
    auto tensor = TestTensorFactory::createFP32Random({64, 128});

    auto engine = KF::prepareExpertGemmLocal(tensor.get(), DeviceId::cpu());
    ASSERT_NE(engine, nullptr);

    // The global registry should have NO entry for this tensor
    const auto *handle = KF::findPreparedGemmWeights(tensor.get(), DeviceId::cpu());
    EXPECT_EQ(handle, nullptr)
        << "prepareExpertGemmLocal must NOT insert into the global prepared_gemm_registry_";
}

// ---------------------------------------------------------------------------
// 3. Engine lifetime is governed by shared_ptr ownership
// ---------------------------------------------------------------------------

TEST_F(Test__MoEPhaseD_NoGlobalRegistry, EngineLifetimeViaSharedPtr)
{
    auto tensor = TestTensorFactory::createFP32Random({64, 128});

    auto engine = KF::prepareExpertGemmLocal(tensor.get(), DeviceId::cpu());
    ASSERT_NE(engine, nullptr);

    // The caller is the sole owner
    EXPECT_EQ(engine.use_count(), 1);

    // Capture raw pointer before release
    ITensorGemm *raw = engine.get();
    EXPECT_NE(raw, nullptr);

    // After reset, use_count drops and the engine is destroyed
    engine.reset();
    EXPECT_EQ(engine, nullptr);
    EXPECT_EQ(engine.use_count(), 0);
}

// ---------------------------------------------------------------------------
// 4. Multiple experts get independent engines
// ---------------------------------------------------------------------------

TEST_F(Test__MoEPhaseD_NoGlobalRegistry, MultipleExpertsGetIndependentEngines)
{
    constexpr int NUM_EXPERTS = 4;
    std::vector<std::unique_ptr<FP32Tensor>> tensors;
    std::vector<std::shared_ptr<ITensorGemm>> engines;

    for (int i = 0; i < NUM_EXPERTS; ++i)
    {
        tensors.push_back(TestTensorFactory::createFP32Random({64, 128}, /*seed=*/100 + i));
        auto engine = KF::prepareExpertGemmLocal(tensors.back().get(), DeviceId::cpu());
        ASSERT_NE(engine, nullptr) << "Expert " << i << " engine should be non-null";
        engines.push_back(std::move(engine));
    }

    // All engines must be distinct objects
    for (int i = 0; i < NUM_EXPERTS; ++i)
    {
        for (int j = i + 1; j < NUM_EXPERTS; ++j)
        {
            EXPECT_NE(engines[i].get(), engines[j].get())
                << "Expert " << i << " and " << j << " must have independent engine instances";
        }
    }

    // None should appear in the global registry
    for (int i = 0; i < NUM_EXPERTS; ++i)
    {
        const auto *handle = KF::findPreparedGemmWeights(tensors[i].get(), DeviceId::cpu());
        EXPECT_EQ(handle, nullptr)
            << "Expert " << i << " must NOT appear in the global registry";
    }
}

// ---------------------------------------------------------------------------
// 5. GPU device correctly rejected
// ---------------------------------------------------------------------------

TEST_F(Test__MoEPhaseD_NoGlobalRegistry, GpuDeviceReturnsNullptr)
{
    auto tensor = TestTensorFactory::createQ8_0Random({64, 128});

    auto engine = KF::prepareExpertGemmLocal(tensor.get(), DeviceId::cuda(0));
    EXPECT_EQ(engine, nullptr)
        << "prepareExpertGemmLocal must reject GPU devices (should use LoadOrchestrator)";
}

// ---------------------------------------------------------------------------
// 6. Null tensor returns nullptr (no throw)
// ---------------------------------------------------------------------------

TEST_F(Test__MoEPhaseD_NoGlobalRegistry, NullTensorReturnsNullptr)
{
    auto engine = KF::prepareExpertGemmLocal(nullptr, DeviceId::cpu());
    EXPECT_EQ(engine, nullptr) << "Null tensor should return nullptr without throwing";
}

// ---------------------------------------------------------------------------
// 7. Contrast: explicit global registration path registers, local path does not
// ---------------------------------------------------------------------------

TEST_F(Test__MoEPhaseD_NoGlobalRegistry, GlobalPathRegistersLocalPathDoesNot)
{
    // Tensor A — use the explicit global transfer registration path
    auto tensor_a = TestTensorFactory::createFP32Random({64, 128}, /*seed=*/42);
    auto kernel_a = std::make_unique<cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor_a.get());
    const auto *handle_a = KF::registerPreparedGemmFromTransfer(
        tensor_a.get(), DeviceId::cpu(), std::move(kernel_a));
    ASSERT_NE(handle_a, nullptr) << "Global path must return a valid handle";

    // Verify it IS in the global registry
    const auto *found_a = KF::findPreparedGemmWeights(tensor_a.get(), DeviceId::cpu());
    EXPECT_NE(found_a, nullptr) << "Global path must register in the global registry";

    // Tensor B — use the local (non-registering) path
    auto tensor_b = TestTensorFactory::createFP32Random({64, 128}, /*seed=*/99);
    auto engine_b = KF::prepareExpertGemmLocal(tensor_b.get(), DeviceId::cpu());
    ASSERT_NE(engine_b, nullptr) << "Local path must return a valid engine";

    // Verify tensor B is NOT in the global registry
    const auto *found_b = KF::findPreparedGemmWeights(tensor_b.get(), DeviceId::cpu());
    EXPECT_EQ(found_b, nullptr) << "Local path must NOT register in the global registry";

    // Tensor A should still be findable
    const auto *refind_a = KF::findPreparedGemmWeights(tensor_a.get(), DeviceId::cpu());
    EXPECT_NE(refind_a, nullptr) << "Global registration for tensor A must survive";
}
