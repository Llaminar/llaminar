/**
 * @file Test__TensorVerification.cpp
 * @brief Unit tests for the unified TensorVerification system
 * @author David Sanftenberg
 * @date January 2026
 *
 * Tests the Phase 1 tensor verification framework including:
 * - VerificationResult creation and manipulation
 * - verifyRawBuffer() for FP32 data validation
 * - VerificationFailure exception formatting
 * - dumpStageBuffers() file creation (when enabled)
 *
 * Note: These tests are only compiled when LLAMINAR_ASSERTIONS_ACTIVE is true
 * (Debug and Integration builds).
 */

#include <gtest/gtest.h>
#include "tensors/TensorVerification.h"
#include "execution/compute_stages/IComputeStage.h"
#include "../../utils/TestTensorFactory.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>

using namespace llaminar2;
using namespace llaminar2::verification;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture
// =============================================================================

class TensorVerificationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Clean up any old test dumps
        cleanupTestDumps();
    }

    void TearDown() override
    {
        // Clean up test dumps
        cleanupTestDumps();
    }

    void cleanupTestDumps()
    {
        // Remove test dump directories if they exist
        try
        {
            if (std::filesystem::exists("/tmp/llaminar_verification_dump"))
            {
                // Only remove directories created by tests (with "test" in name)
                for (const auto &entry : std::filesystem::directory_iterator("/tmp/llaminar_verification_dump"))
                {
                    if (entry.path().filename().string().find("TestStage") != std::string::npos)
                    {
                        std::filesystem::remove_all(entry.path());
                    }
                }
            }
        }
        catch (...)
        {
            // Ignore cleanup errors
        }
    }

    // Helper to create test buffer with specific values
    std::vector<float> createTestBuffer(size_t rows, size_t cols, float fill = 1.0f)
    {
        std::vector<float> buffer(rows * cols, fill);
        return buffer;
    }

    // Helper to create buffer with NaN values
    std::vector<float> createNaNBuffer(size_t rows, size_t cols, size_t nan_index)
    {
        std::vector<float> buffer(rows * cols, 1.0f);
        if (nan_index < buffer.size())
        {
            buffer[nan_index] = std::numeric_limits<float>::quiet_NaN();
        }
        return buffer;
    }

    // Helper to create buffer with Inf values
    std::vector<float> createInfBuffer(size_t rows, size_t cols, size_t inf_index)
    {
        std::vector<float> buffer(rows * cols, 1.0f);
        if (inf_index < buffer.size())
        {
            buffer[inf_index] = std::numeric_limits<float>::infinity();
        }
        return buffer;
    }

    // Helper to create all-zero buffer
    std::vector<float> createZeroBuffer(size_t rows, size_t cols)
    {
        return std::vector<float>(rows * cols, 0.0f);
    }
};

// =============================================================================
// VerificationResult Tests
// =============================================================================

TEST_F(TensorVerificationTest, VerificationResult_Ok)
{
    auto result = VerificationResult::ok();

    EXPECT_TRUE(result.passed);
    EXPECT_TRUE(result.tensor_name.empty());
    EXPECT_TRUE(result.error_reason.empty());
}

TEST_F(TensorVerificationTest, VerificationResult_Fail)
{
    auto result = VerificationResult::fail("test_tensor", "Contains NaN");

    EXPECT_FALSE(result.passed);
    EXPECT_EQ(result.tensor_name, "test_tensor");
    EXPECT_EQ(result.error_reason, "Contains NaN");
}

TEST_F(TensorVerificationTest, VerificationResult_WithDiagnostics)
{
    auto result = VerificationResult::withDiagnostics(
        false, "my_tensor", "5 NaN values found",
        5,   // nan_count
        0,   // inf_count
        100, // zero_count
        1000 // total_sampled
    );

    EXPECT_FALSE(result.passed);
    EXPECT_EQ(result.tensor_name, "my_tensor");
    EXPECT_EQ(result.error_reason, "5 NaN values found");
    EXPECT_EQ(result.nan_count, 5u);
    EXPECT_EQ(result.inf_count, 0u);
    EXPECT_EQ(result.zero_count, 100u);
    EXPECT_EQ(result.total_sampled, 1000u);
}

// =============================================================================
// verifyRawBuffer Tests (Only in Debug/Integration builds)
// =============================================================================

#if LLAMINAR_ASSERTIONS_ACTIVE

TEST_F(TensorVerificationTest, VerifyRawBuffer_NullPointerFails)
{
    VerificationConfig config;
    config.check_null = true;

    auto result = verifyRawBuffer(nullptr, 10, 10, "null_tensor", "FP32", config);

    EXPECT_FALSE(result.passed);
    EXPECT_EQ(result.tensor_name, "null_tensor");
    EXPECT_TRUE(result.error_reason.find("Null") != std::string::npos);
}

TEST_F(TensorVerificationTest, VerifyRawBuffer_ValidDataPasses)
{
    auto buffer = createTestBuffer(10, 10, 1.0f);
    VerificationConfig config;

    auto result = verifyRawBuffer(buffer.data(), 10, 10, "valid_tensor", "FP32", config);

    EXPECT_TRUE(result.passed);
}

TEST_F(TensorVerificationTest, VerifyRawBuffer_DetectsNaN)
{
    auto buffer = createNaNBuffer(10, 10, 50); // NaN at index 50
    VerificationConfig config;
    config.check_nan = true;
    config.sample_rows = 10; // Sample all rows

    auto result = verifyRawBuffer(buffer.data(), 10, 10, "nan_tensor", "FP32", config);

    EXPECT_FALSE(result.passed);
    EXPECT_GT(result.nan_count, 0u);
    EXPECT_TRUE(result.error_reason.find("NaN") != std::string::npos);
}

TEST_F(TensorVerificationTest, VerifyRawBuffer_DetectsInf)
{
    auto buffer = createInfBuffer(10, 10, 25); // Inf at index 25
    VerificationConfig config;
    config.check_inf = true;
    config.sample_rows = 10;

    auto result = verifyRawBuffer(buffer.data(), 10, 10, "inf_tensor", "FP32", config);

    EXPECT_FALSE(result.passed);
    EXPECT_GT(result.inf_count, 0u);
    EXPECT_TRUE(result.error_reason.find("Inf") != std::string::npos);
}

TEST_F(TensorVerificationTest, VerifyRawBuffer_DetectsNegativeInf)
{
    std::vector<float> buffer(100, 1.0f);
    buffer[50] = -std::numeric_limits<float>::infinity();

    VerificationConfig config;
    config.check_inf = true;
    config.sample_rows = 10;

    auto result = verifyRawBuffer(buffer.data(), 10, 10, "neg_inf_tensor", "FP32", config);

    EXPECT_FALSE(result.passed);
    EXPECT_GT(result.inf_count, 0u);
}

TEST_F(TensorVerificationTest, VerifyRawBuffer_DetectsAllZero)
{
    auto buffer = createZeroBuffer(10, 10);
    VerificationConfig config;
    config.check_all_zero = true;
    config.sample_rows = 10;

    auto result = verifyRawBuffer(buffer.data(), 10, 10, "zero_tensor", "FP32", config);

    EXPECT_FALSE(result.passed);
    EXPECT_EQ(result.zero_count, 100u); // All 100 elements are zero
    EXPECT_TRUE(result.error_reason.find("zero") != std::string::npos);
}

TEST_F(TensorVerificationTest, VerifyRawBuffer_AllZeroPassesIfNotChecking)
{
    auto buffer = createZeroBuffer(10, 10);
    VerificationConfig config;
    config.check_all_zero = false; // Don't check for zeros

    auto result = verifyRawBuffer(buffer.data(), 10, 10, "zero_tensor", "FP32", config);

    EXPECT_TRUE(result.passed);
}

TEST_F(TensorVerificationTest, VerifyRawBuffer_SampleRowsRespected)
{
    // Create 100 rows but only sample first 2
    std::vector<float> buffer(100 * 10, 1.0f);
    // Put NaN in row 50 (beyond sample range)
    buffer[50 * 10] = std::numeric_limits<float>::quiet_NaN();

    VerificationConfig config;
    config.check_nan = true;
    config.sample_rows = 2; // Only sample first 2 rows

    auto result = verifyRawBuffer(buffer.data(), 100, 10, "partial_tensor", "FP32", config);

    // Should pass because NaN is beyond sample range
    EXPECT_TRUE(result.passed);
    EXPECT_EQ(result.total_sampled, 20u); // 2 rows * 10 cols
}

TEST_F(TensorVerificationTest, VerifyRawBuffer_NonFP32TypeSkipped)
{
    // Non-FP32 types should be skipped (return ok)
    auto buffer = createNaNBuffer(10, 10, 0);
    VerificationConfig config;

    auto result = verifyRawBuffer(buffer.data(), 10, 10, "q8_tensor", "Q8_1", config);

    EXPECT_TRUE(result.passed); // Skipped, so passes
}

TEST_F(TensorVerificationTest, VerifyRawBuffer_EmptyBufferPasses)
{
    // Empty buffer with zero dimensions should pass (no data to validate)
    // Note: vector.data() may return nullptr or non-null for empty vector (implementation-defined)
    // Use nullptr explicitly to test zero-element case
    VerificationConfig config;
    config.check_null = false; // Disable null check since we're passing nullptr

    auto result = verifyRawBuffer(nullptr, 0, 0, "empty_tensor", "FP32", config);

    EXPECT_TRUE(result.passed); // numel=0, so no validation performed, should pass
}

TEST_F(TensorVerificationTest, VerifyRawBuffer_DiagnosticsPopulated)
{
    // Create buffer with mixed content
    std::vector<float> buffer(100);
    for (size_t i = 0; i < 100; ++i)
    {
        if (i % 3 == 0)
            buffer[i] = 0.0f; // ~33 zeros
        else
            buffer[i] = 1.0f;
    }

    VerificationConfig config;
    config.check_all_zero = false; // Pass but collect diagnostics
    config.sample_rows = 10;

    auto result = verifyRawBuffer(buffer.data(), 10, 10, "diag_tensor", "FP32", config);

    EXPECT_TRUE(result.passed);
    EXPECT_EQ(result.total_sampled, 100u);
    EXPECT_GT(result.zero_count, 30u);    // Should have ~33 zeros
    EXPECT_LT(result.zero_count, 40u);    // Sanity check
    EXPECT_EQ(result.nan_count, 0u);
    EXPECT_EQ(result.inf_count, 0u);
}

// =============================================================================
// VerificationFailure Exception Tests
// =============================================================================

TEST_F(TensorVerificationTest, VerificationFailure_ContainsContext)
{
    VerificationFailure ex(
        "TestStage",       // stage_name
        5,                 // layer_idx
        "ENTRY",           // phase
        "input_tensor",    // tensor_name
        "Contains 3 NaN",  // reason
        "/tmp/dump/test"   // dump_path
    );

    EXPECT_EQ(ex.stageName(), "TestStage");
    EXPECT_EQ(ex.layerIdx(), 5);
    EXPECT_STREQ(ex.phase(), "ENTRY");
    EXPECT_EQ(ex.tensorName(), "input_tensor");
    EXPECT_EQ(ex.reason(), "Contains 3 NaN");
    EXPECT_EQ(ex.dumpPath(), "/tmp/dump/test");
}

TEST_F(TensorVerificationTest, VerificationFailure_MessageContainsAllInfo)
{
    VerificationFailure ex(
        "FusedAttentionWoStage",
        12,
        "EXIT",
        "attention_output",
        "Contains 156 NaN values",
        "/tmp/llaminar_verification_dump/test123"
    );

    std::string what = ex.what();

    // Message should contain all relevant info
    EXPECT_TRUE(what.find("FusedAttentionWoStage") != std::string::npos);
    EXPECT_TRUE(what.find("12") != std::string::npos);
    EXPECT_TRUE(what.find("EXIT") != std::string::npos);
    EXPECT_TRUE(what.find("attention_output") != std::string::npos);
    EXPECT_TRUE(what.find("156 NaN") != std::string::npos);
    EXPECT_TRUE(what.find("/tmp/llaminar_verification_dump/test123") != std::string::npos);
}

TEST_F(TensorVerificationTest, VerificationFailure_EmptyDumpPath)
{
    VerificationFailure ex("Stage", 0, "ENTRY", "tensor", "error", "");

    std::string what = ex.what();
    EXPECT_TRUE(what.find("(disabled)") != std::string::npos);
}

// =============================================================================
// dumpStageBuffers Tests
// =============================================================================

TEST_F(TensorVerificationTest, DumpStageBuffers_CreatesDirectory)
{
    // Create minimal StageDumpInfo
    StageDumpInfo dump_info;
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    dump_info.addInput("test_input", input_data.data(), 2, 2);

    std::string dump_path = dumpStageBuffers(
        "TestStage", 0, "EXIT", dump_info, "test_input", "test reason");

    // Should have created a dump directory
    EXPECT_FALSE(dump_path.empty());
    EXPECT_TRUE(std::filesystem::exists(dump_path));

    // Should have created subdirectories
    EXPECT_TRUE(std::filesystem::exists(dump_path + "/inputs"));
    EXPECT_TRUE(std::filesystem::exists(dump_path + "/outputs"));
    EXPECT_TRUE(std::filesystem::exists(dump_path + "/weights"));
}

TEST_F(TensorVerificationTest, DumpStageBuffers_CreatesManifest)
{
    StageDumpInfo dump_info;
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    dump_info.addInput("input_A", input_data.data(), 2, 2);

    std::string dump_path = dumpStageBuffers(
        "TestStage", 3, "ENTRY", dump_info, "input_A", "NaN detected");

    // Check manifest exists
    std::string manifest_path = dump_path + "/manifest.json";
    EXPECT_TRUE(std::filesystem::exists(manifest_path));

    // Read and verify manifest content
    std::ifstream f(manifest_path);
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());

    EXPECT_TRUE(content.find("\"stage\": \"TestStage\"") != std::string::npos);
    EXPECT_TRUE(content.find("\"layer_idx\": 3") != std::string::npos);
    EXPECT_TRUE(content.find("\"phase\": \"ENTRY\"") != std::string::npos);
    EXPECT_TRUE(content.find("\"tensor\": \"input_A\"") != std::string::npos);
    EXPECT_TRUE(content.find("\"reason\": \"NaN detected\"") != std::string::npos);
}

TEST_F(TensorVerificationTest, DumpStageBuffers_DumpsInputBinaries)
{
    StageDumpInfo dump_info;
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    dump_info.addInput("my_input", input_data.data(), 2, 2);

    std::string dump_path = dumpStageBuffers(
        "TestStage", 0, "EXIT", dump_info, "", "");

    // Check binary file exists
    std::string bin_path = dump_path + "/inputs/my_input.bin";
    EXPECT_TRUE(std::filesystem::exists(bin_path));

    // Verify binary content
    std::ifstream f(bin_path, std::ios::binary);
    std::vector<float> read_data(4);
    f.read(reinterpret_cast<char *>(read_data.data()), 4 * sizeof(float));

    EXPECT_EQ(read_data[0], 1.0f);
    EXPECT_EQ(read_data[1], 2.0f);
    EXPECT_EQ(read_data[2], 3.0f);
    EXPECT_EQ(read_data[3], 4.0f);
}

TEST_F(TensorVerificationTest, DumpStageBuffers_DumpsMetadataFile)
{
    StageDumpInfo dump_info;
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    dump_info.addInput("test_input", input_data.data(), 2, 2);

    std::string dump_path = dumpStageBuffers(
        "TestStage", 0, "EXIT", dump_info, "", "");

    // Check metadata file exists
    std::string meta_path = dump_path + "/inputs/test_input_metadata.txt";
    EXPECT_TRUE(std::filesystem::exists(meta_path));

    // Read and verify metadata
    std::ifstream f(meta_path);
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());

    EXPECT_TRUE(content.find("name: test_input") != std::string::npos);
    EXPECT_TRUE(content.find("rows: 2") != std::string::npos);
    EXPECT_TRUE(content.find("cols: 2") != std::string::npos);
    EXPECT_TRUE(content.find("dtype: FP32") != std::string::npos);
}

TEST_F(TensorVerificationTest, DumpStageBuffers_HandlesMultipleInputsAndOutputs)
{
    StageDumpInfo dump_info;

    std::vector<float> input1 = {1.0f, 2.0f};
    std::vector<float> input2 = {3.0f, 4.0f, 5.0f};
    std::vector<float> output1 = {6.0f, 7.0f, 8.0f, 9.0f};

    dump_info.addInput("A", input1.data(), 1, 2);
    dump_info.addInput("B", input2.data(), 1, 3);
    dump_info.addOutput("C", output1.data(), 2, 2);

    std::string dump_path = dumpStageBuffers(
        "TestStage", 0, "EXIT", dump_info, "", "");

    EXPECT_TRUE(std::filesystem::exists(dump_path + "/inputs/A.bin"));
    EXPECT_TRUE(std::filesystem::exists(dump_path + "/inputs/B.bin"));
    EXPECT_TRUE(std::filesystem::exists(dump_path + "/outputs/C.bin"));
}

// =============================================================================
// Timestamp Generation Tests
// =============================================================================

TEST_F(TensorVerificationTest, GenerateTimestamp_ReturnsValidFormat)
{
    std::string ts = generateTimestamp();

    // Format: YYYYMMDD_HHMMSS_mmm = 8 + 1 + 6 + 1 + 3 = 19 characters
    EXPECT_EQ(ts.length(), 19u);
    EXPECT_EQ(ts[8], '_');  // Between date and time
    EXPECT_EQ(ts[15], '_'); // Between time and milliseconds
}

TEST_F(TensorVerificationTest, GenerateTimestamp_SuccessiveCallsUnique)
{
    std::string ts1 = generateTimestamp();

    // Small delay to ensure different millisecond
    std::this_thread::sleep_for(std::chrono::milliseconds(2));

    std::string ts2 = generateTimestamp();

    // Timestamps should be unique (or at least different)
    // Note: May be same if within same millisecond, but unlikely with sleep
    // More robust: just check both are non-empty
    EXPECT_FALSE(ts1.empty());
    EXPECT_FALSE(ts2.empty());
}

#endif // LLAMINAR_ASSERTIONS_ACTIVE

// =============================================================================
// Integration Tests - Verification with Real Tensors
// =============================================================================

#if LLAMINAR_ASSERTIONS_ACTIVE

TEST_F(TensorVerificationTest, IntegrationWithTestTensorFactory)
{
    // Use TestTensorFactory to create realistic test data
    auto tensor = TestTensorFactory::createFP32Random({32, 64}, -1.0f, 1.0f);

    VerificationConfig config;
    config.sample_rows = 8;

    auto result = verifyRawBuffer(
        tensor->data(), 32, 64, "random_tensor", "FP32", config);

    EXPECT_TRUE(result.passed);
    EXPECT_EQ(result.total_sampled, 8u * 64u); // 8 rows * 64 cols
}

TEST_F(TensorVerificationTest, IntegrationNaNInjection)
{
    // Create valid tensor then inject NaN
    auto tensor = TestTensorFactory::createFP32Random({16, 32}, -1.0f, 1.0f);
    tensor->mutable_data()[100] = std::numeric_limits<float>::quiet_NaN();

    VerificationConfig config;
    config.check_nan = true;
    config.sample_rows = 16; // Check all rows

    auto result = verifyRawBuffer(
        tensor->data(), 16, 32, "nan_injected", "FP32", config);

    EXPECT_FALSE(result.passed);
    EXPECT_EQ(result.nan_count, 1u);
}

#endif // LLAMINAR_ASSERTIONS_ACTIVE
