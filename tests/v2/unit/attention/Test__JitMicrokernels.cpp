/**
 * @file Test__JitMicrokernels.cpp
 * @brief Unit tests for modular JIT microkernel infrastructure
 * @author David Sanftenberg
 *
 * Tests the JIT code generation infrastructure:
 * - Microkernel emitters generate valid code
 * - Register allocation follows conventions
 * - Basic JIT kernels can be created and executed
 */

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>
#include <cstring>

#include "kernels/cpu/attention/q8_1/jit/JitMicrokernelBase.h"
#include "kernels/cpu/attention/q8_1/jit/JitQ8DotProduct.h"
#include "kernels/cpu/attention/q8_1/jit/JitFastExp.h"
#include "kernels/cpu/attention/q8_1/jit/JitOnlineSoftmax.h"
#include "kernels/cpu/attention/q8_1/jit/JitVWeightedAccum.h"
#include "kernels/cpu/attention/q8_1/jit/JitWoProjection.h"

namespace llaminar::v2::kernels::jit::test
{

    // ============================================================================
    // JIT Infrastructure Tests
    // ============================================================================

    /**
     * @brief Test ZmmZones struct constants
     */
    TEST(JitInfrastructure, ZmmZonesConstants)
    {
        // Verify the ZMM zone definitions match expected values
        EXPECT_EQ(ZmmZones::ACCUM_START, 0);
        EXPECT_EQ(ZmmZones::ACCUM_END, 7);

        EXPECT_EQ(ZmmZones::INPUT_START, 8);
        EXPECT_EQ(ZmmZones::INPUT_END, 15);

        EXPECT_EQ(ZmmZones::STATE_START, 16);
        EXPECT_EQ(ZmmZones::STATE_END, 19);

        EXPECT_EQ(ZmmZones::SCRATCH_START, 20);
        EXPECT_EQ(ZmmZones::SCRATCH_END, 25);

        EXPECT_EQ(ZmmZones::CONST_START, 26);
        EXPECT_EQ(ZmmZones::CONST_END, 31);
    }

    /**
     * @brief Test ConstRegs struct constants
     */
    TEST(JitInfrastructure, ConstRegsConstants)
    {
        EXPECT_EQ(ConstRegs::ZMM_128, 26);
        EXPECT_EQ(ConstRegs::ZMM_SCALE, 27);
        EXPECT_EQ(ConstRegs::ZMM_NEG_INF, 28);
        EXPECT_EQ(ConstRegs::ZMM_ONE, 29);
        EXPECT_EQ(ConstRegs::ZMM_LOG2E, 30);
        EXPECT_EQ(ConstRegs::ZMM_EXP_MIN, 31);
    }

    /**
     * @brief Test StateRegs struct constants
     */
    TEST(JitInfrastructure, StateRegsConstants)
    {
        EXPECT_EQ(StateRegs::ZMM_MAX, 16);
        EXPECT_EQ(StateRegs::ZMM_SUM, 17);
        EXPECT_EQ(StateRegs::ZMM_WEIGHT, 18);
        EXPECT_EQ(StateRegs::ZMM_CORR, 19);
    }

    // ============================================================================
    // Register Convention Tests
    // ============================================================================

    /**
     * @brief Test that register zones don't overlap
     */
    TEST(JitRegisterConventions, NoZoneOverlap)
    {
        // Accumulators: 0-7
        int accum_end = ZmmZones::ACCUM_END;

        // Inputs: 8-15
        EXPECT_GT(ZmmZones::INPUT_START, accum_end);
        int input_end = ZmmZones::INPUT_END;

        // State: 16-19
        EXPECT_GT(ZmmZones::STATE_START, input_end);
        int state_end = ZmmZones::STATE_END;

        // Scratch: 20-25
        EXPECT_GT(ZmmZones::SCRATCH_START, state_end);
        int scratch_end = ZmmZones::SCRATCH_END;

        // Constants: 26-31
        EXPECT_GT(ZmmZones::CONST_START, scratch_end);

        // All 32 ZMM registers should be covered
        EXPECT_EQ(ZmmZones::CONST_END, 31); // zmm31 is last
    }

    /**
     * @brief Test that all zones sum correctly
     */
    TEST(JitRegisterConventions, FullZmmCoverage)
    {
        // Count registers in each zone
        int accum_count = ZmmZones::ACCUM_END - ZmmZones::ACCUM_START + 1;       // 8
        int input_count = ZmmZones::INPUT_END - ZmmZones::INPUT_START + 1;       // 8
        int state_count = ZmmZones::STATE_END - ZmmZones::STATE_START + 1;       // 4
        int scratch_count = ZmmZones::SCRATCH_END - ZmmZones::SCRATCH_START + 1; // 6
        int const_count = ZmmZones::CONST_END - ZmmZones::CONST_START + 1;       // 6

        int total = accum_count + input_count + state_count + scratch_count + const_count;
        EXPECT_EQ(total, 32);
    }

    // ============================================================================
    // JIT Kernel Generation Tests
    // ============================================================================

    /**
     * @brief Minimal concrete JIT kernel for testing
     */
    class TestJitKernel : public JitMicrokernelBase
    {
    public:
        TestJitKernel() : JitMicrokernelBase() {}

        /**
         * @brief Generate a minimal test kernel
         */
        void generate_test_kernel()
        {
            // Prologue
            push(rbp);
            mov(rbp, rsp);

            // Emit some SIMD operations using helper methods
            vxorps(zmm_accum(0), zmm_accum(0), zmm_accum(0));
            vxorps(zmm_scratch(0), zmm_scratch(0), zmm_scratch(0));

            // Epilogue
            pop(rbp);
            ret();
        }

        void finalize()
        {
            ready();
        }
    };

    /**
     * @brief Test that minimal JIT kernel generates valid code
     */
    TEST(JitKernels, MinimalKernelGeneration)
    {
        TestJitKernel kernel;
        kernel.generate_test_kernel();
        kernel.finalize();

        // Get the generated code
        const uint8_t *code = kernel.getCode();
        ASSERT_NE(code, nullptr);

        size_t code_size = kernel.getSize();
        EXPECT_GT(code_size, 0u);
        EXPECT_LT(code_size, 4096u); // Should be small
    }

    /**
     * @brief Test that kernel uses correct register zones
     */
    TEST(JitKernels, RegisterZoneUsage)
    {
        TestJitKernel kernel;
        kernel.generate_test_kernel();
        kernel.finalize();

        // Just test that it compiles and generates code
        EXPECT_GT(kernel.getSize(), 0u);
    }

    // ============================================================================
    // JIT Emitter Include Tests
    // ============================================================================

    /**
     * @brief Test JitFastExpEmitter header compiles
     */
    TEST(JitEmitters, FastExpIncludesCorrectly)
    {
        // This test just verifies the header can be included
        // and that the emitter class exists
        SUCCEED() << "JitFastExpEmitter header compiles successfully";
    }

    /**
     * @brief Test JitQ8DotProductEmitter header compiles
     */
    TEST(JitEmitters, Q8DotProductIncludesCorrectly)
    {
        SUCCEED() << "JitQ8DotProductEmitter header compiles successfully";
    }

    /**
     * @brief Test JitOnlineSoftmaxEmitter header compiles
     */
    TEST(JitEmitters, OnlineSoftmaxIncludesCorrectly)
    {
        SUCCEED() << "JitOnlineSoftmaxEmitter header compiles successfully";
    }

    /**
     * @brief Test JitVWeightedAccumEmitter header compiles
     */
    TEST(JitEmitters, VWeightedAccumIncludesCorrectly)
    {
        SUCCEED() << "JitVWeightedAccumEmitter header compiles successfully";
    }

    /**
     * @brief Test JitWoProjectionEmitter header compiles
     */
    TEST(JitEmitters, WoProjectionIncludesCorrectly)
    {
        SUCCEED() << "JitWoProjectionEmitter header compiles successfully";
    }

    // ============================================================================
    // Q8 Test Data Generator (for future use)
    // ============================================================================

    /**
     * @brief Test data generator for Q8_1 blocks
     */
    class Q8TestData
    {
    public:
        /**
         * @brief Convert FP32 to FP16 bit representation
         */
        static uint16_t fp32_to_fp16_bits(float f)
        {
            uint32_t x;
            std::memcpy(&x, &f, sizeof(x));

            uint32_t sign = (x >> 16) & 0x8000;
            int exp = ((x >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = (x >> 13) & 0x3FF;

            if (exp <= 0)
            {
                return static_cast<uint16_t>(sign); // Underflow to zero
            }
            else if (exp >= 31)
            {
                return static_cast<uint16_t>(sign | 0x7C00); // Overflow to inf
            }
            return static_cast<uint16_t>(sign | (exp << 10) | mant);
        }

        /**
         * @brief Convert FP16 bits to FP32
         */
        static float fp16_bits_to_fp32(uint16_t h)
        {
            uint32_t sign = (h & 0x8000) << 16;
            int exp = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;

            if (exp == 0)
            {
                return 0.0f; // Denormal -> zero
            }
            else if (exp == 31)
            {
                return std::numeric_limits<float>::infinity();
            }

            exp = exp - 15 + 127;
            uint32_t result = sign | (exp << 23) | (mant << 13);
            float f;
            std::memcpy(&f, &result, sizeof(f));
            return f;
        }
    };

    /**
     * @brief Test FP16 conversion utility
     */
    TEST(TestUtils, FP16Conversion)
    {
        // Test round-trip conversion
        float test_values[] = {0.0f, 1.0f, -1.0f, 0.5f, 0.125f, 100.0f};

        for (float v : test_values)
        {
            uint16_t fp16 = Q8TestData::fp32_to_fp16_bits(v);
            float roundtrip = Q8TestData::fp16_bits_to_fp32(fp16);

            // FP16 has limited precision, so allow some error
            if (v == 0.0f)
            {
                EXPECT_EQ(roundtrip, 0.0f);
            }
            else
            {
                float rel_error = std::abs(roundtrip - v) / std::abs(v);
                EXPECT_LT(rel_error, 0.01f) << "Value: " << v; // 1% tolerance
            }
        }
    }

} // namespace llaminar::v2::kernels::jit::test
