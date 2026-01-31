/**
 * @file Test__LocalTPContext_BARManagement.cpp
 * @brief Unit tests for LocalTPContext BAR-backed tensor management
 * @author David Sanftenberg
 * @date January 2026
 *
 * Tests the BAR-backed tensor registry in LocalTPContext:
 * - registerBARBackedOutput(): Registers BAR-backed tensors per stage/device
 * - getBARBackedOutputs(): Retrieves registered tensors in device order
 * - hasBARBackedOutputs(): Checks if a stage has BAR-backed outputs
 * - clearBARBackedOutputs(): Clears all registrations
 *
 * These methods support zero-copy allreduce by tracking which stage outputs
 * are allocated in BAR memory (AMD VRAM accessible via PCIe BAR).
 */

#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

#include "collective/LocalTPContext.h"
#include "backends/GlobalDeviceAddress.h"
#include "tensors/TensorClasses.h"
#include "backends/DeviceId.h"

namespace llaminar2
{
    namespace test
    {

        /**
         * @brief Mock BAR-backed FP32 tensor for testing
         *
         * Since actual BAR allocation requires DirectP2PEngine with hardware,
         * we create a mock tensor that reports isBARBacked() = true by calling
         * initBARBackedDirect() with mock pointers.
         */
        class MockBARBackedTensor : public FP32Tensor
        {
        public:
            /**
             * @brief Create a mock BAR-backed tensor
             * @param shape Tensor dimensions
             */
            explicit MockBARBackedTensor(const std::vector<size_t> &shape)
                : FP32Tensor(shape)
            {
                // Set up mock BAR state - use small non-null pointers as mock addresses
                // Note: These are not real GPU pointers, just for testing isBARBacked()
                initBARBackedDirect(
                    reinterpret_cast<void *>(0x1000), // Mock ROCm pointer
                    reinterpret_cast<void *>(0x2000), // Mock CUDA pointer
                    DeviceId::rocm(0),                // Mock ROCm device
                    DeviceId::cuda(0),                // Mock CUDA device
                    numel() * sizeof(float)           // Size in bytes
                );
            }
        };

        // =========================================================================
        // Test Fixture
        // =========================================================================

        class Test__LocalTPContext_BARManagement : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                // Create test devices (heterogeneous CUDA + ROCm)
                cuda0_ = GlobalDeviceAddress::cuda(0, 0);
                rocm0_ = GlobalDeviceAddress::rocm(0, 0);

                // Create context with 2 devices
                devices_ = {cuda0_, rocm0_};
                weights_ = {0.5f, 0.5f};

                // Use HOST backend (doesn't require actual GPUs for these unit tests)
                ctx_ = createLocalTPContext(devices_, weights_, CollectiveBackendType::HOST);

                // Create mock BAR-backed tensors
                tensor1_ = std::make_unique<MockBARBackedTensor>(std::vector<size_t>{32, 64});
                tensor2_ = std::make_unique<MockBARBackedTensor>(std::vector<size_t>{32, 64});
                tensor3_ = std::make_unique<MockBARBackedTensor>(std::vector<size_t>{64, 128});

                // Create a regular (non-BAR-backed) tensor for negative tests
                regular_tensor_ = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 64});
            }

            GlobalDeviceAddress cuda0_;
            GlobalDeviceAddress rocm0_;
            std::vector<GlobalDeviceAddress> devices_;
            std::vector<float> weights_;
            std::unique_ptr<ILocalTPContext> ctx_;
            std::unique_ptr<MockBARBackedTensor> tensor1_;
            std::unique_ptr<MockBARBackedTensor> tensor2_;
            std::unique_ptr<MockBARBackedTensor> tensor3_;
            std::unique_ptr<FP32Tensor> regular_tensor_;

            // Helper to get concrete LocalTPContext for testing non-interface methods
            LocalTPContext *getConcreteContext()
            {
                return dynamic_cast<LocalTPContext *>(ctx_.get());
            }
        };

        // =========================================================================
        // Test: RegisterBAROutput_Success
        // =========================================================================

        /**
         * @test Can register a BAR-backed tensor for a stage
         */
        TEST_F(Test__LocalTPContext_BARManagement, RegisterBAROutput_Success)
        {
            auto *ctx = getConcreteContext();
            ASSERT_NE(ctx, nullptr);

            // Verify tensor is BAR-backed
            ASSERT_TRUE(tensor1_->isBARBacked());

            // Register should not throw
            EXPECT_NO_THROW(ctx->registerBARBackedOutput(
                "layer0_ffn_down_allreduce",
                cuda0_,
                tensor1_.get()));

            // Verify registration
            EXPECT_TRUE(ctx->hasBARBackedOutputs("layer0_ffn_down_allreduce"));
        }

        // =========================================================================
        // Test: RegisterBAROutput_InvalidDevice
        // =========================================================================

        /**
         * @test Throws for device not in context
         */
        TEST_F(Test__LocalTPContext_BARManagement, RegisterBAROutput_InvalidDevice)
        {
            auto *ctx = getConcreteContext();
            ASSERT_NE(ctx, nullptr);

            // Create a device that's not in the context
            GlobalDeviceAddress invalid_device = GlobalDeviceAddress::cuda(99, 0);

            EXPECT_THROW(
                ctx->registerBARBackedOutput(
                    "layer0_ffn_down_allreduce",
                    invalid_device,
                    tensor1_.get()),
                std::invalid_argument);
        }

        // =========================================================================
        // Test: RegisterBAROutput_NullTensor
        // =========================================================================

        /**
         * @test Throws for null tensor
         */
        TEST_F(Test__LocalTPContext_BARManagement, RegisterBAROutput_NullTensor)
        {
            auto *ctx = getConcreteContext();
            ASSERT_NE(ctx, nullptr);

            EXPECT_THROW(
                ctx->registerBARBackedOutput(
                    "layer0_ffn_down_allreduce",
                    cuda0_,
                    nullptr),
                std::invalid_argument);
        }

        // =========================================================================
        // Test: RegisterBAROutput_NonBARTensor
        // =========================================================================

        /**
         * @test Throws for non-BAR-backed tensor
         */
        TEST_F(Test__LocalTPContext_BARManagement, RegisterBAROutput_NonBARTensor)
        {
            auto *ctx = getConcreteContext();
            ASSERT_NE(ctx, nullptr);

            // Verify regular tensor is NOT BAR-backed
            ASSERT_FALSE(regular_tensor_->isBARBacked());

            EXPECT_THROW(
                ctx->registerBARBackedOutput(
                    "layer0_ffn_down_allreduce",
                    cuda0_,
                    regular_tensor_.get()),
                std::invalid_argument);
        }

        // =========================================================================
        // Test: GetBAROutputs_Registered
        // =========================================================================

        /**
         * @test Returns registered tensors in device order
         */
        TEST_F(Test__LocalTPContext_BARManagement, GetBAROutputs_Registered)
        {
            auto *ctx = getConcreteContext();
            ASSERT_NE(ctx, nullptr);

            // Register tensors for both devices
            ctx->registerBARBackedOutput("layer0_ffn_down_allreduce", cuda0_, tensor1_.get());
            ctx->registerBARBackedOutput("layer0_ffn_down_allreduce", rocm0_, tensor2_.get());

            // Get outputs
            auto outputs = ctx->getBARBackedOutputs("layer0_ffn_down_allreduce");

            // Should have degree() entries
            ASSERT_EQ(outputs.size(), static_cast<size_t>(ctx->degree()));

            // Index 0 = cuda0_ tensor, Index 1 = rocm0_ tensor
            EXPECT_EQ(outputs[0], tensor1_.get());
            EXPECT_EQ(outputs[1], tensor2_.get());
        }

        // =========================================================================
        // Test: GetBAROutputs_PartialRegistration
        // =========================================================================

        /**
         * @test Returns nullptr for devices without registration
         */
        TEST_F(Test__LocalTPContext_BARManagement, GetBAROutputs_PartialRegistration)
        {
            auto *ctx = getConcreteContext();
            ASSERT_NE(ctx, nullptr);

            // Only register for cuda0_, not rocm0_
            ctx->registerBARBackedOutput("layer0_wo_allreduce", cuda0_, tensor1_.get());

            auto outputs = ctx->getBARBackedOutputs("layer0_wo_allreduce");

            ASSERT_EQ(outputs.size(), static_cast<size_t>(ctx->degree()));
            EXPECT_EQ(outputs[0], tensor1_.get()); // cuda0_ registered
            EXPECT_EQ(outputs[1], nullptr);        // rocm0_ not registered
        }

        // =========================================================================
        // Test: GetBAROutputs_Unregistered
        // =========================================================================

        /**
         * @test Returns vector of nullptrs for unknown stage
         */
        TEST_F(Test__LocalTPContext_BARManagement, GetBAROutputs_Unregistered)
        {
            auto *ctx = getConcreteContext();
            ASSERT_NE(ctx, nullptr);

            auto outputs = ctx->getBARBackedOutputs("nonexistent_stage");

            // Should return degree() nullptrs
            ASSERT_EQ(outputs.size(), static_cast<size_t>(ctx->degree()));
            for (auto *tensor : outputs)
            {
                EXPECT_EQ(tensor, nullptr);
            }
        }

        // =========================================================================
        // Test: HasBAROutputs_True
        // =========================================================================

        /**
         * @test Returns true when at least one device has registration
         */
        TEST_F(Test__LocalTPContext_BARManagement, HasBAROutputs_True)
        {
            auto *ctx = getConcreteContext();
            ASSERT_NE(ctx, nullptr);

            // No registration yet
            EXPECT_FALSE(ctx->hasBARBackedOutputs("layer0_ffn_down_allreduce"));

            // Register one device
            ctx->registerBARBackedOutput("layer0_ffn_down_allreduce", cuda0_, tensor1_.get());

            // Should now return true
            EXPECT_TRUE(ctx->hasBARBackedOutputs("layer0_ffn_down_allreduce"));
        }

        // =========================================================================
        // Test: HasBAROutputs_False
        // =========================================================================

        /**
         * @test Returns false for unknown stage
         */
        TEST_F(Test__LocalTPContext_BARManagement, HasBAROutputs_False)
        {
            auto *ctx = getConcreteContext();
            ASSERT_NE(ctx, nullptr);

            EXPECT_FALSE(ctx->hasBARBackedOutputs("nonexistent_stage"));
        }

        // =========================================================================
        // Test: ClearBAROutputs
        // =========================================================================

        /**
         * @test Clears all registrations
         */
        TEST_F(Test__LocalTPContext_BARManagement, ClearBAROutputs)
        {
            auto *ctx = getConcreteContext();
            ASSERT_NE(ctx, nullptr);

            // Register tensors for multiple stages
            ctx->registerBARBackedOutput("layer0_ffn_down_allreduce", cuda0_, tensor1_.get());
            ctx->registerBARBackedOutput("layer0_wo_allreduce", cuda0_, tensor2_.get());

            // Verify registrations exist
            EXPECT_TRUE(ctx->hasBARBackedOutputs("layer0_ffn_down_allreduce"));
            EXPECT_TRUE(ctx->hasBARBackedOutputs("layer0_wo_allreduce"));

            // Clear all
            ctx->clearBARBackedOutputs();

            // Verify all cleared
            EXPECT_FALSE(ctx->hasBARBackedOutputs("layer0_ffn_down_allreduce"));
            EXPECT_FALSE(ctx->hasBARBackedOutputs("layer0_wo_allreduce"));

            // getBARBackedOutputs should return nullptrs
            auto outputs = ctx->getBARBackedOutputs("layer0_ffn_down_allreduce");
            for (auto *tensor : outputs)
            {
                EXPECT_EQ(tensor, nullptr);
            }
        }

        // =========================================================================
        // Test: MultipleStages
        // =========================================================================

        /**
         * @test Tracks tensors per-stage correctly
         */
        TEST_F(Test__LocalTPContext_BARManagement, MultipleStages)
        {
            auto *ctx = getConcreteContext();
            ASSERT_NE(ctx, nullptr);

            // Register different tensors for different stages
            ctx->registerBARBackedOutput("layer0_ffn_down_allreduce", cuda0_, tensor1_.get());
            ctx->registerBARBackedOutput("layer0_wo_allreduce", cuda0_, tensor2_.get());
            ctx->registerBARBackedOutput("layer1_ffn_down_allreduce", cuda0_, tensor3_.get());

            // Each stage should have its own tensor
            auto ffn_outputs = ctx->getBARBackedOutputs("layer0_ffn_down_allreduce");
            auto wo_outputs = ctx->getBARBackedOutputs("layer0_wo_allreduce");
            auto ffn1_outputs = ctx->getBARBackedOutputs("layer1_ffn_down_allreduce");

            EXPECT_EQ(ffn_outputs[0], tensor1_.get());
            EXPECT_EQ(wo_outputs[0], tensor2_.get());
            EXPECT_EQ(ffn1_outputs[0], tensor3_.get());

            // They should all be different
            EXPECT_NE(ffn_outputs[0], wo_outputs[0]);
            EXPECT_NE(ffn_outputs[0], ffn1_outputs[0]);
            EXPECT_NE(wo_outputs[0], ffn1_outputs[0]);

            // All should report having BAR outputs
            EXPECT_TRUE(ctx->hasBARBackedOutputs("layer0_ffn_down_allreduce"));
            EXPECT_TRUE(ctx->hasBARBackedOutputs("layer0_wo_allreduce"));
            EXPECT_TRUE(ctx->hasBARBackedOutputs("layer1_ffn_down_allreduce"));
        }

        // =========================================================================
        // Test: Overwrite Registration
        // =========================================================================

        /**
         * @test Can overwrite an existing registration for same stage/device
         */
        TEST_F(Test__LocalTPContext_BARManagement, OverwriteRegistration)
        {
            auto *ctx = getConcreteContext();
            ASSERT_NE(ctx, nullptr);

            // Register tensor1 first
            ctx->registerBARBackedOutput("layer0_ffn_down_allreduce", cuda0_, tensor1_.get());

            auto outputs1 = ctx->getBARBackedOutputs("layer0_ffn_down_allreduce");
            EXPECT_EQ(outputs1[0], tensor1_.get());

            // Overwrite with tensor2
            ctx->registerBARBackedOutput("layer0_ffn_down_allreduce", cuda0_, tensor2_.get());

            auto outputs2 = ctx->getBARBackedOutputs("layer0_ffn_down_allreduce");
            EXPECT_EQ(outputs2[0], tensor2_.get());
            EXPECT_NE(outputs2[0], tensor1_.get());
        }

        // =========================================================================
        // REGRESSION TESTS - MIGRATED TO INTEGRATION TESTS
        // =========================================================================
        // NOTE: Hardware-dependent PCIeBAR regression tests have been migrated to
        // integration tests in Test__LocalTPBackendBehavior.cpp:
        // - Regression_CUDATensorMustNotBeSkipped_PCIeBAR → BAR_CUDATensorMustNotBeSkipped
        // - Regression_BothDevicesMustHaveRegisteredTensors_PCIeBAR → BAR_BothDevicesMustHaveRegisteredTensors
        // - Regression_DistinctTensorPointersRequired → BAR_DistinctTensorPointersRequired
        //
        // Those tests require actual CUDA+ROCm hardware with PCIe BAR support.

    } // namespace test
} // namespace llaminar2
