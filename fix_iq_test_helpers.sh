#!/bin/bash
# Add missing helper methods to IQ tensor test files

for file in Test__IQ1_MTensor Test__IQ1_STensor Test__IQ2_XXSTensor Test__IQ2_XSTensor Test__IQ3_XXSTensor Test__IQ2_STensor; do
    test_file="tests/v2/unit/tensors/${file}.cpp"
    
    if [ ! -f "$test_file" ]; then
        echo "File not found: $test_file"
        continue
    fi
    
    # Check if already has the methods
    if grep -q "createRandomTensor" "$test_file"; then
        echo "Skipping $file - already has helper methods"
        continue
    fi
    
    echo "Updating $file..."
    
    # Add #include <random> after other includes
    sed -i '/#include <cstring>/a #include <random>' "$test_file"
    
    # Find the class definition and add helper methods before the first createBlock method
    # This is a bit tricky - we'll add it after "protected:" and before the first method
    
    cat > /tmp/gemm_helpers.txt << 'EOF'

    void SetUp() override {
        rng_.seed(42); // Reproducible tests
    }
    
    // Helper: Create a random tensor for GEMM testing
    template<typename TensorType, typename BlockType>
    std::unique_ptr<TensorType> createRandomTensorImpl(size_t rows, size_t cols) {
        std::vector<size_t> shape = {rows, cols};
        size_t blocks_per_row = (cols + 255) / 256; // 256 elements per block
        size_t total_blocks = rows * blocks_per_row;
        std::vector<uint8_t> raw_data(total_blocks * sizeof(BlockType));

        std::uniform_real_distribution<float> scale_dist(0.1f, 2.0f);
        std::uniform_int_distribution<uint8_t> byte_dist(0, 255);

        for (size_t b = 0; b < total_blocks; ++b) {
            BlockType *block = reinterpret_cast<BlockType *>(raw_data.data() + b * sizeof(BlockType));
            // Fill with random data (specific initialization per type would be better, but this works for testing)
            for (size_t i = 0; i < sizeof(BlockType); ++i) {
                reinterpret_cast<uint8_t*>(block)[i] = byte_dist(rng_);
            }
            // Set a reasonable scale
            block->d = fp32_to_fp16(scale_dist(rng_));
        }

        return std::make_unique<TensorType>(shape, raw_data);
    }

    // Helper: Compute reference GEMM: C = A @ B^T
    void referenceGEMM(const float *A, const float *B, float *C, int m, int n, int k) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int l = 0; l < k; ++l) {
                    sum += A[i * k + l] * B[j * k + l]; // B is [n, k]
                }
                C[i * n + j] = sum;
            }
        }
    }

    // Helper: Check if two matrices are approximately equal
    bool matricesEqual(const float *A, const float *B, int size, float tolerance = 1e-4f) {
        for (int i = 0; i < size; ++i) {
            float diff = std::abs(A[i] - B[i]);
            float rel_error = diff / (std::abs(A[i]) + 1e-8f);
            if (diff > tolerance && rel_error > tolerance) {
                return false;
            }
        }
        return true;
    }

EOF
    
done

echo "Done!"
