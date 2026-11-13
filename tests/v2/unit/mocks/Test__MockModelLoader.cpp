/**
 * @file Test__MockModelLoader.cpp
 * @brief Unit tests for MockModelLoader
 *
 * Tests the mock model loader's ability to:
 * - Provide in-memory tensors without file I/O
 * - Support different tensor types (FP32, Q4_0, Q8_0, IQ4_NL)
 * - Return correct metadata for model configurations
 * - Create row/column slices for tensor parallelism testing
 * - Use builder pattern for fluent configuration
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "mocks/MockModelLoader.h"
#include "tensors/Tensors.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// BASIC CONSTRUCTION TESTS
// =============================================================================

TEST(Test__MockModelLoader, DefaultConstruction) {
    MockModelLoader mock;
    
    EXPECT_TRUE(mock.isLoaded());
    EXPECT_EQ(mock.architecture(), "qwen2");
    EXPECT_EQ(mock.tensorCount(), 0);
    EXPECT_EQ(mock.totalBytes(), 0);
}

TEST(Test__MockModelLoader, BuilderBasicConfiguration) {
    auto mock = MockModelLoaderBuilder()
        .setArchitecture("llama")
        .setBlockCount(32)
        .setEmbeddingLength(4096)
        .setHeadCount(32)
        .setHeadCountKV(8)
        .setVocabSize(32000)
        .setContextLength(2048)
        .setRopeTheta(10000.0f)
        .setRmsNormEps(1e-5f)
        .build();
    
    EXPECT_EQ(mock->architecture(), "llama");
    EXPECT_EQ(mock->blockCount(), 32);
    EXPECT_EQ(mock->embeddingLength(), 4096);
    EXPECT_EQ(mock->headCount(), 32);
    EXPECT_EQ(mock->headCountKV(), 8);
    EXPECT_EQ(mock->vocabSize(), 32000);
    EXPECT_EQ(mock->contextLength(), 2048);
    EXPECT_FLOAT_EQ(mock->ropeTheta(), 10000.0f);
    EXPECT_FLOAT_EQ(mock->rmsNormEps(), 1e-5f);
}

// =============================================================================
// TENSOR ADDITION TESTS
// =============================================================================

TEST(Test__MockModelLoader, AddFP32RandomTensor) {
    auto mock = MockModelLoaderBuilder()
        .addFP32RandomTensor("test_tensor", {64, 128}, -1.0f, 1.0f, 42)
        .build();
    
    EXPECT_TRUE(mock->hasTensor("test_tensor"));
    EXPECT_EQ(mock->tensorCount(), 1);
    
    auto tensor = mock->loadTensor("test_tensor");
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->rows(), 64);
    EXPECT_EQ(tensor->cols(), 128);
    
    // Verify it's actually FP32
    auto* fp32 = dynamic_cast<FP32Tensor*>(tensor.get());
    ASSERT_NE(fp32, nullptr);
    
    // Verify data is in range
    const float* data = fp32->data();
    for (size_t i = 0; i < 64 * 128; ++i) {
        EXPECT_GE(data[i], -1.0f);
        EXPECT_LE(data[i], 1.0f);
    }
}

TEST(Test__MockModelLoader, AddFP32ZerosTensor) {
    auto mock = MockModelLoaderBuilder()
        .addFP32ZerosTensor("zeros", {32, 64})
        .build();
    
    auto tensor = mock->loadTensor("zeros");
    ASSERT_NE(tensor, nullptr);
    
    auto* fp32 = dynamic_cast<FP32Tensor*>(tensor.get());
    ASSERT_NE(fp32, nullptr);
    
    const float* data = fp32->data();
    for (size_t i = 0; i < 32 * 64; ++i) {
        EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
}

TEST(Test__MockModelLoader, AddFP32OnesTensor) {
    auto mock = MockModelLoaderBuilder()
        .addFP32OnesTensor("ones", {16, 32})
        .build();
    
    auto tensor = mock->loadTensor("ones");
    ASSERT_NE(tensor, nullptr);
    
    auto* fp32 = dynamic_cast<FP32Tensor*>(tensor.get());
    ASSERT_NE(fp32, nullptr);
    
    const float* data = fp32->data();
    for (size_t i = 0; i < 16 * 32; ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

TEST(Test__MockModelLoader, AddQ4_0RandomTensor) {
    auto mock = MockModelLoaderBuilder()
        .addQ4_0RandomTensor("q4_weights", {256, 512}, -1.0f, 1.0f, 42)
        .build();
    
    EXPECT_TRUE(mock->hasTensor("q4_weights"));
    
    auto tensor = mock->loadTensor("q4_weights");
    ASSERT_NE(tensor, nullptr);
    
    auto* q4 = dynamic_cast<Q4_0Tensor*>(tensor.get());
    ASSERT_NE(q4, nullptr);
    EXPECT_EQ(q4->rows(), 256);
    EXPECT_EQ(q4->cols(), 512);
}

TEST(Test__MockModelLoader, AddQ8_0RandomTensor) {
    auto mock = MockModelLoaderBuilder()
        .addQ8_0RandomTensor("q8_weights", {128, 256}, -0.5f, 0.5f, 123)
        .build();
    
    auto tensor = mock->loadTensor("q8_weights");
    ASSERT_NE(tensor, nullptr);
    
    auto* q8 = dynamic_cast<Q8_0Tensor*>(tensor.get());
    ASSERT_NE(q8, nullptr);
    EXPECT_EQ(q8->rows(), 128);
    EXPECT_EQ(q8->cols(), 256);
}

TEST(Test__MockModelLoader, AddIQ4_NLRandomTensor) {
    auto mock = MockModelLoaderBuilder()
        .addIQ4_NLRandomTensor("iq4_weights", {64, 128}, -1.0f, 1.0f, 456)
        .build();
    
    auto tensor = mock->loadTensor("iq4_weights");
    ASSERT_NE(tensor, nullptr);
    
    auto* iq4 = dynamic_cast<IQ4_NLTensor*>(tensor.get());
    ASSERT_NE(iq4, nullptr);
    EXPECT_EQ(iq4->rows(), 64);
    EXPECT_EQ(iq4->cols(), 128);
}

// =============================================================================
// TENSOR NAMES AND LOOKUP TESTS
// =============================================================================

TEST(Test__MockModelLoader, TensorNamesReturnsAllTensors) {
    auto mock = MockModelLoaderBuilder()
        .addFP32RandomTensor("tensor_a", {10, 20})
        .addFP32RandomTensor("tensor_b", {30, 40})
        .addQ4_0RandomTensor("tensor_c", {50, 60})
        .build();
    
    auto names = mock->tensorNames();
    EXPECT_EQ(names.size(), 3);
    
    // Check all names are present (order not guaranteed)
    EXPECT_TRUE(std::find(names.begin(), names.end(), "tensor_a") != names.end());
    EXPECT_TRUE(std::find(names.begin(), names.end(), "tensor_b") != names.end());
    EXPECT_TRUE(std::find(names.begin(), names.end(), "tensor_c") != names.end());
}

TEST(Test__MockModelLoader, HasTensorReturnsFalseForMissing) {
    auto mock = MockModelLoaderBuilder()
        .addFP32RandomTensor("exists", {10, 20})
        .build();
    
    EXPECT_TRUE(mock->hasTensor("exists"));
    EXPECT_FALSE(mock->hasTensor("does_not_exist"));
}

TEST(Test__MockModelLoader, LoadTensorReturnsNullForMissing) {
    auto mock = MockModelLoaderBuilder().build();
    
    auto tensor = mock->loadTensor("nonexistent");
    EXPECT_EQ(tensor, nullptr);
    
    // Check missing requests tracking
    EXPECT_EQ(mock->missingTensorRequests().size(), 1);
    EXPECT_EQ(mock->missingTensorRequests()[0], "nonexistent");
}

// =============================================================================
// METADATA ACCESS TESTS
// =============================================================================

TEST(Test__MockModelLoader, GetIntReturnsConfiguredValues) {
    auto mock = MockModelLoaderBuilder()
        .setBlockCount(24)
        .setHeadCount(14)
        .setHeadCountKV(2)
        .build();
    
    EXPECT_EQ(mock->getInt("block_count"), 24);
    EXPECT_EQ(mock->getInt("head_count"), 14);
    EXPECT_EQ(mock->getInt("head_count_kv"), 2);
    EXPECT_EQ(mock->getInt("unknown", 999), 999);
}

TEST(Test__MockModelLoader, GetUInt64ReturnsConfiguredValues) {
    auto mock = MockModelLoaderBuilder()
        .setEmbeddingLength(896)
        .setContextLength(32768)
        .setVocabSize(151936)
        .build();
    
    EXPECT_EQ(mock->getUInt64("embedding_length"), 896);
    EXPECT_EQ(mock->getUInt64("context_length"), 32768);
    EXPECT_EQ(mock->getUInt64("vocab_size"), 151936);
    EXPECT_EQ(mock->getUInt64("unknown", 12345), 12345);
}

TEST(Test__MockModelLoader, GetFloatReturnsConfiguredValues) {
    auto mock = MockModelLoaderBuilder()
        .setRopeTheta(1000000.0f)
        .setRmsNormEps(1e-6f)
        .build();
    
    EXPECT_FLOAT_EQ(mock->getFloat("rope_theta"), 1000000.0f);
    EXPECT_FLOAT_EQ(mock->getFloat("rms_norm_eps"), 1e-6f);
    EXPECT_FLOAT_EQ(mock->getFloat("unknown", 3.14f), 3.14f);
}

TEST(Test__MockModelLoader, GetStringReturnsConfiguredValues) {
    auto mock = MockModelLoaderBuilder()
        .setArchitecture("qwen2")
        .build();
    
    EXPECT_EQ(mock->getString("architecture"), "qwen2");
    EXPECT_EQ(mock->getString("unknown", "default"), "default");
}

TEST(Test__MockModelLoader, CustomParametersWork) {
    auto mock = MockModelLoaderBuilder()
        .setInt("custom_int", 42)
        .setFloat("custom_float", 2.718f)
        .setString("custom_string", "hello")
        .build();
    
    EXPECT_EQ(mock->getInt("custom_int"), 42);
    EXPECT_FLOAT_EQ(mock->getFloat("custom_float"), 2.718f);
    EXPECT_EQ(mock->getString("custom_string"), "hello");
}

// =============================================================================
// ROW/COLUMN SLICE TESTS
// =============================================================================

TEST(Test__MockModelLoader, LoadTensorRowSlice) {
    auto mock = MockModelLoaderBuilder()
        .addFP32RandomTensor("matrix", {100, 200}, -1.0f, 1.0f, 42)
        .build();
    
    // Get full tensor for reference
    auto full = mock->loadTensor("matrix");
    ASSERT_NE(full, nullptr);
    auto* full_fp32 = dynamic_cast<FP32Tensor*>(full.get());
    ASSERT_NE(full_fp32, nullptr);
    
    // Get row slice [20, 50)
    auto slice = mock->loadTensorRowSlice("matrix", 20, 50);
    ASSERT_NE(slice, nullptr);
    EXPECT_EQ(slice->rows(), 30);
    EXPECT_EQ(slice->cols(), 200);
    
    auto* slice_fp32 = dynamic_cast<FP32Tensor*>(slice.get());
    ASSERT_NE(slice_fp32, nullptr);
    
    // Verify slice data matches original
    for (size_t r = 0; r < 30; ++r) {
        for (size_t c = 0; c < 200; ++c) {
            EXPECT_FLOAT_EQ(slice_fp32->data()[r * 200 + c],
                           full_fp32->data()[(r + 20) * 200 + c])
                << "Mismatch at row " << r << ", col " << c;
        }
    }
}

TEST(Test__MockModelLoader, LoadTensorColumnSlice) {
    auto mock = MockModelLoaderBuilder()
        .addFP32RandomTensor("matrix", {50, 100}, -1.0f, 1.0f, 42)
        .build();
    
    auto full = mock->loadTensor("matrix");
    auto* full_fp32 = dynamic_cast<FP32Tensor*>(full.get());
    ASSERT_NE(full_fp32, nullptr);
    
    // Get column slice [10, 40)
    auto slice = mock->loadTensorColumnSlice("matrix", 10, 40);
    ASSERT_NE(slice, nullptr);
    EXPECT_EQ(slice->rows(), 50);
    EXPECT_EQ(slice->cols(), 30);
    
    auto* slice_fp32 = dynamic_cast<FP32Tensor*>(slice.get());
    ASSERT_NE(slice_fp32, nullptr);
    
    // Verify slice data matches original
    for (size_t r = 0; r < 50; ++r) {
        for (size_t c = 0; c < 30; ++c) {
            EXPECT_FLOAT_EQ(slice_fp32->data()[r * 30 + c],
                           full_fp32->data()[r * 100 + c + 10])
                << "Mismatch at row " << r << ", col " << c;
        }
    }
}

TEST(Test__MockModelLoader, SliceOfMissingTensorReturnsNull) {
    auto mock = MockModelLoaderBuilder().build();
    
    auto row_slice = mock->loadTensorRowSlice("nonexistent", 0, 10);
    EXPECT_EQ(row_slice, nullptr);
    
    auto col_slice = mock->loadTensorColumnSlice("nonexistent", 0, 10);
    EXPECT_EQ(col_slice, nullptr);
}

// =============================================================================
// PRESET TESTS
// =============================================================================

TEST(Test__MockModelLoader, CreateMinimalPreset) {
    auto mock = MockModelLoader::createMinimal();
    
    EXPECT_TRUE(mock->isLoaded());
    EXPECT_EQ(mock->architecture(), "qwen2");
    EXPECT_EQ(mock->blockCount(), 1);
    EXPECT_EQ(mock->embeddingLength(), 128);
    EXPECT_EQ(mock->headCount(), 4);
    EXPECT_EQ(mock->headCountKV(), 2);
    
    // Should have embedding, one layer, and output
    EXPECT_TRUE(mock->hasTensor("token_embd.weight"));
    EXPECT_TRUE(mock->hasTensor("blk.0.attn_norm.weight"));
    EXPECT_TRUE(mock->hasTensor("blk.0.attn_q.weight"));
    EXPECT_TRUE(mock->hasTensor("output.weight"));
}

TEST(Test__MockModelLoader, CreateQwen2_05BPreset) {
    // Presets only set metadata - no tensors by default (for fast tests)
    auto mock = MockModelLoader::createQwen2_05B();
    
    EXPECT_EQ(mock->architecture(), "qwen2");
    EXPECT_EQ(mock->blockCount(), 24);
    EXPECT_EQ(mock->embeddingLength(), 896);
    EXPECT_EQ(mock->headCount(), 14);
    EXPECT_EQ(mock->headCountKV(), 2);
    EXPECT_EQ(mock->vocabSize(), 151936);
}

TEST(Test__MockModelLoader, CreateQwen2_7BPreset) {
    // Presets only set metadata - no tensors by default (for fast tests)
    auto mock = MockModelLoader::createQwen2_7B();
    
    EXPECT_EQ(mock->architecture(), "qwen2");
    EXPECT_EQ(mock->blockCount(), 28);
    EXPECT_EQ(mock->embeddingLength(), 3584);
    EXPECT_EQ(mock->headCount(), 28);
    EXPECT_EQ(mock->headCountKV(), 4);
}

TEST(Test__MockModelLoader, CreateLlama3_8BPreset) {
    // Presets only set metadata - no tensors by default (for fast tests)
    auto mock = MockModelLoader::createLlama3_8B();
    
    EXPECT_EQ(mock->architecture(), "llama");
    EXPECT_EQ(mock->blockCount(), 32);
    EXPECT_EQ(mock->embeddingLength(), 4096);
    EXPECT_EQ(mock->headCount(), 32);
    EXPECT_EQ(mock->headCountKV(), 8);
}

// =============================================================================
// CALL TRACKING TESTS
// =============================================================================

TEST(Test__MockModelLoader, LoadTensorCallCountTracking) {
    auto mock = MockModelLoaderBuilder()
        .addFP32RandomTensor("tensor", {10, 20})
        .build();
    
    EXPECT_EQ(mock->loadTensorCallCount(), 0);
    
    mock->loadTensor("tensor");
    EXPECT_EQ(mock->loadTensorCallCount(), 1);
    
    mock->loadTensor("tensor");
    mock->loadTensor("missing");
    EXPECT_EQ(mock->loadTensorCallCount(), 3);
    
    mock->resetCounters();
    EXPECT_EQ(mock->loadTensorCallCount(), 0);
}

TEST(Test__MockModelLoader, MissingRequestsTracking) {
    auto mock = MockModelLoaderBuilder()
        .addFP32RandomTensor("exists", {10, 20})
        .build();
    
    mock->loadTensor("exists");  // Should succeed
    mock->loadTensor("missing1");  // Missing
    mock->loadTensor("missing2");  // Missing
    
    const auto& missing = mock->missingTensorRequests();
    EXPECT_EQ(missing.size(), 2);
    EXPECT_EQ(missing[0], "missing1");
    EXPECT_EQ(missing[1], "missing2");
    
    mock->resetCounters();
    EXPECT_TRUE(mock->missingTensorRequests().empty());
}

// =============================================================================
// TOTAL BYTES CALCULATION TEST
// =============================================================================

TEST(Test__MockModelLoader, TotalBytesCalculation) {
    auto mock = MockModelLoaderBuilder()
        .addFP32RandomTensor("fp32_tensor", {10, 20})  // 10*20 = 200 elements
        .addFP32RandomTensor("fp32_tensor2", {5, 10})  // 5*10 = 50 elements
        .build();
    
    // Mock returns element count * sizeof(float) as approximation
    size_t expected = (10 * 20 + 5 * 10) * sizeof(float);
    EXPECT_EQ(mock->totalBytes(), expected);
}

// =============================================================================
// REPRODUCIBILITY TEST
// =============================================================================

TEST(Test__MockModelLoader, RandomTensorsAreReproducible) {
    auto mock1 = MockModelLoaderBuilder()
        .addFP32RandomTensor("tensor", {32, 64}, -1.0f, 1.0f, 12345)
        .build();
    
    auto mock2 = MockModelLoaderBuilder()
        .addFP32RandomTensor("tensor", {32, 64}, -1.0f, 1.0f, 12345)
        .build();
    
    auto tensor1 = mock1->loadTensor("tensor");
    auto tensor2 = mock2->loadTensor("tensor");
    
    auto* fp1 = dynamic_cast<FP32Tensor*>(tensor1.get());
    auto* fp2 = dynamic_cast<FP32Tensor*>(tensor2.get());
    ASSERT_NE(fp1, nullptr);
    ASSERT_NE(fp2, nullptr);
    
    // Same seed should produce identical data
    for (size_t i = 0; i < 32 * 64; ++i) {
        EXPECT_FLOAT_EQ(fp1->data()[i], fp2->data()[i]);
    }
}

TEST(Test__MockModelLoader, DifferentSeedsProduceDifferentData) {
    auto mock1 = MockModelLoaderBuilder()
        .addFP32RandomTensor("tensor", {32, 64}, -1.0f, 1.0f, 111)
        .build();
    
    auto mock2 = MockModelLoaderBuilder()
        .addFP32RandomTensor("tensor", {32, 64}, -1.0f, 1.0f, 222)
        .build();
    
    auto tensor1 = mock1->loadTensor("tensor");
    auto tensor2 = mock2->loadTensor("tensor");
    
    auto* fp1 = dynamic_cast<FP32Tensor*>(tensor1.get());
    auto* fp2 = dynamic_cast<FP32Tensor*>(tensor2.get());
    
    // Different seeds should produce different data
    bool any_different = false;
    for (size_t i = 0; i < 32 * 64; ++i) {
        if (fp1->data()[i] != fp2->data()[i]) {
            any_different = true;
            break;
        }
    }
    EXPECT_TRUE(any_different);
}

// =============================================================================
// INTERFACE COMPLIANCE TEST
// =============================================================================

TEST(Test__MockModelLoader, ImplementsIModelLoaderInterface) {
    // Verify MockModelLoader can be used through IModelLoader pointer
    std::shared_ptr<IModelLoader> loader = MockModelLoader::createMinimal();
    
    EXPECT_TRUE(loader->isLoaded());
    EXPECT_FALSE(loader->architecture().empty());
    EXPECT_GT(loader->tensorCount(), 0);
    EXPECT_GT(loader->blockCount(), 0);
    
    // Can load tensors through interface
    auto names = loader->tensorNames();
    EXPECT_FALSE(names.empty());
    
    auto tensor = loader->loadTensor(names[0]);
    EXPECT_NE(tensor, nullptr);
}

/** @test In-memory source fixtures implement the same original-coordinate contract with genuine roots. */
TEST(Test__MockModelLoader, NativeSourceWindowsRetainOriginalTensorBytesAndRejectInvalidPrefix)
{
    MockModelLoader loader;
    auto ordinary=std::make_shared<FP32Tensor>(std::vector<size_t>{3,8});
    for(size_t i=0;i<24;++i)ordinary->mutable_data()[i]=float(i);
    loader.addTensor("ordinary",ordinary);
    EXPECT_EQ(loader.nativeWeightSourcePlacement(),CPUStoragePlacement::local());
    const auto source=loader.nativeWeightSourceGeometry("ordinary");
    EXPECT_EQ(source.original_shape,(std::vector<size_t>{8,3}));
    EXPECT_EQ(source.original_type,TensorType::FP32);
    EXPECT_EQ(source.native_group_elements,1u);EXPECT_EQ(source.native_group_bytes,sizeof(float));
    std::array<float,5> result;result.fill(-7.f);
    NativeWeightSourceWindow window{.first_row=1,.rows=1,.first_column=2,.columns=3,.native_row_bytes=3*sizeof(float)};
    loader.readNativeWeightSourceWindow("ordinary",window,
        {reinterpret_cast<std::byte*>(result.data()),window.nativeBytes()});
    EXPECT_EQ(result,(std::array<float,5>{10.f,11.f,12.f,-7.f,-7.f}));
    EXPECT_EQ(loader.loadTensorCallCount(),0u);EXPECT_EQ(loader.nativeSourceReadCallCount(),1u);
    window.first_row=3;
    EXPECT_THROW(loader.readNativeWeightSourceWindow("ordinary",window,
        {reinterpret_cast<std::byte*>(result.data()),3*sizeof(float)}),std::invalid_argument);
    EXPECT_EQ(result,(std::array<float,5>{10.f,11.f,12.f,-7.f,-7.f}));
    EXPECT_EQ(loader.nativeSourceReadCallCount(),1u);
    ordinary->release_raw_data();
    const auto retired=loader.nativeWeightSourceGeometry("ordinary");
    EXPECT_EQ(retired.original_shape,source.original_shape);
    EXPECT_EQ(retired.original_type,source.original_type);
    EXPECT_EQ(retired.native_group_elements,source.native_group_elements);
    EXPECT_EQ(retired.native_group_bytes,source.native_group_bytes);
    window.first_row=1;
    EXPECT_THROW(loader.readNativeWeightSourceWindow("ordinary",window,
        {reinterpret_cast<std::byte*>(result.data()),window.nativeBytes()}),std::invalid_argument);
    EXPECT_EQ(result,(std::array<float,5>{10.f,11.f,12.f,-7.f,-7.f}));
    EXPECT_EQ(loader.nativeSourceReadCallCount(),1u);
}

/** @test Real native blocks and original expert axes remain unchanged by the in-memory reader. */
TEST(Test__MockModelLoader, NativeSourceWindowsPreserveQuantizedGroupsAndExpertCoordinates)
{
    MockModelLoader loader;loader.addQ4_0RandomTensor("quant",{3,64},-1.f,1.f,19);
    const auto source=loader.nativeWeightSourceGeometry("quant");
    ASSERT_EQ(source.original_shape,(std::vector<size_t>{64,3}));
    ASSERT_EQ(source.native_group_elements,32u);
    const auto root=loader.loadTensor("quant");ASSERT_TRUE(root);
    std::vector<std::byte> output(source.native_group_bytes+8,std::byte{0xcd});
    NativeWeightSourceWindow window{.first_row=1,.rows=1,.first_column=32,.columns=32,.native_row_bytes=source.native_group_bytes};
    loader.readNativeWeightSourceWindow("quant",window,{output.data(),source.native_group_bytes});
    EXPECT_EQ(std::memcmp(output.data(),static_cast<const std::byte*>(root->raw_data())+3*source.native_group_bytes,
        source.native_group_bytes),0);
    for(size_t i=source.native_group_bytes;i<output.size();++i)EXPECT_EQ(output[i],std::byte{0xcd});
    const auto populated=output;
    root->release_host_weight_data();
    const auto retired=loader.nativeWeightSourceGeometry("quant");
    EXPECT_EQ(retired.original_shape,source.original_shape);
    EXPECT_EQ(retired.original_type,source.original_type);
    EXPECT_EQ(retired.native_group_bytes,source.native_group_bytes);
    EXPECT_THROW(loader.readNativeWeightSourceWindow("quant",window,
        {output.data(),source.native_group_bytes}),std::invalid_argument);
    EXPECT_EQ(output,populated);
    EXPECT_EQ(loader.nativeSourceReadCallCount(),1u);
    auto experts=std::make_shared<FP32Tensor>(std::vector<size_t>{8,3,2});
    for(size_t i=0;i<48;++i)experts->mutable_data()[i]=float(i);
    loader.addTensor("experts",experts);
    EXPECT_EQ(loader.nativeWeightSourceGeometry("experts").original_shape,(std::vector<size_t>{8,3,2}));
    std::array<float,3> values{-7.f,-7.f,-7.f};
    const NativeWeightSourceWindow selected{.original_expert=1,.selected_expert_ordinal=0,.first_row=2,.rows=1,
        .first_column=3,.columns=2,.native_row_bytes=2*sizeof(float)};
    loader.readNativeWeightSourceWindow("experts",selected,
        {reinterpret_cast<std::byte*>(values.data()),selected.nativeBytes()});
    EXPECT_EQ(values,(std::array<float,3>{43.f,44.f,-7.f}));
    loader.resetCounters();EXPECT_EQ(loader.nativeSourceReadCallCount(),0u);
}
