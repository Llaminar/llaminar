/**
 * @file test_cosma_conversion.cpp
 * @brief Test float<->double conversions in COSMA prefill manager
 * @author David Sanftenberg
 *
 * This test verifies that the float->double->float roundtrip in CosmaPrefillManager
 * preserves values correctly and doesn't cause the parity failures we're seeing.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

using cosma_scalar_t = double; // Matches cosma_prefill_manager.h

void test_simple_conversion()
{
    std::cout << "=== Test 1: Simple Conversion ===" << std::endl;

    // Test float -> double -> float roundtrip
    std::vector<float> input = {1.0f, 2.5f, -3.7f, 0.0001f, 12345.6f};
    std::vector<double> intermediate(input.size());
    std::vector<float> output(input.size());

    // Convert float -> double (like fill_activation does)
    for (size_t i = 0; i < input.size(); ++i)
    {
        intermediate[i] = static_cast<cosma_scalar_t>(input[i]);
    }

    // Convert double -> float (like reconstruct_matrix does)
    for (size_t i = 0; i < intermediate.size(); ++i)
    {
        output[i] = static_cast<float>(intermediate[i]);
    }

    // Check if values match
    bool passed = true;
    for (size_t i = 0; i < input.size(); ++i)
    {
        if (input[i] != output[i])
        {
            std::cout << "MISMATCH at index " << i << ": "
                      << std::setprecision(10)
                      << input[i] << " -> " << intermediate[i] << " -> " << output[i]
                      << std::endl;
            passed = false;
        }
    }

    std::cout << "Simple conversion test: " << (passed ? "PASSED" : "FAILED") << std::endl;
}

void test_matrix_conversion()
{
    std::cout << "\n=== Test 2: Matrix Conversion ===" << std::endl;

    const int m = 32, k = 896;
    std::vector<float> input_matrix(m * k);

    // Fill with test pattern
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto &val : input_matrix)
    {
        val = dist(gen);
    }

    // Simulate fill_activation conversion
    std::vector<double> cosma_storage(m * k);
    for (int i = 0; i < m * k; ++i)
    {
        cosma_storage[i] = static_cast<cosma_scalar_t>(input_matrix[i]);
    }

    // Simulate reconstruct_matrix conversion
    std::vector<float> output_matrix(m * k);
    for (int i = 0; i < m * k; ++i)
    {
        output_matrix[i] = static_cast<float>(cosma_storage[i]);
    }

    // Calculate errors
    double max_abs_error = 0.0;
    double sum_sq_error = 0.0;
    double sum_sq_original = 0.0;
    int mismatch_count = 0;

    for (int i = 0; i < m * k; ++i)
    {
        double error = std::abs(input_matrix[i] - output_matrix[i]);
        if (error > 1e-6)
        {
            mismatch_count++;
            if (mismatch_count <= 5)
            {
                std::cout << "Mismatch at index " << i << ": "
                          << std::setprecision(10)
                          << input_matrix[i] << " -> " << output_matrix[i]
                          << " (error=" << error << ")" << std::endl;
            }
        }
        max_abs_error = std::max(max_abs_error, error);
        sum_sq_error += error * error;
        sum_sq_original += input_matrix[i] * input_matrix[i];
    }

    double rel_l2 = std::sqrt(sum_sq_error) / (std::sqrt(sum_sq_original) + 1e-30);

    std::cout << "Matrix " << m << "x" << k << " conversion results:" << std::endl;
    std::cout << "  Max absolute error: " << std::setprecision(10) << max_abs_error << std::endl;
    std::cout << "  Relative L2 error:  " << rel_l2 << std::endl;
    std::cout << "  Mismatches:         " << mismatch_count << " / " << (m * k) << std::endl;
    std::cout << "Matrix conversion test: " << ((max_abs_error < 1e-6) ? "PASSED" : "FAILED") << std::endl;
}

void test_accumulation_precision()
{
    std::cout << "\n=== Test 3: Accumulation Precision ===" << std::endl;

    // Test if double precision helps with accumulation
    const int n = 10000;
    std::vector<float> values(n, 0.1f);

    // Float accumulation (how OpenBLAS does it internally)
    float sum_float = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        sum_float += values[i];
    }

    // Double accumulation (how COSMA does it)
    double sum_double = 0.0;
    for (int i = 0; i < n; ++i)
    {
        sum_double += static_cast<double>(values[i]);
    }
    float sum_double_converted = static_cast<float>(sum_double);

    std::cout << "Summing " << n << " values of 0.1:" << std::endl;
    std::cout << "  Float accumulation:  " << std::setprecision(10) << sum_float << std::endl;
    std::cout << "  Double accumulation: " << sum_double_converted << std::endl;
    std::cout << "  Expected:            " << (n * 0.1f) << std::endl;
    std::cout << "  Difference:          " << std::abs(sum_double_converted - sum_float) << std::endl;

    // Double precision should be more accurate
    bool passed = std::abs(sum_double_converted - n * 0.1f) < std::abs(sum_float - n * 0.1f);
    std::cout << "Accumulation precision test: " << (passed ? "PASSED" : "FAILED") << std::endl;
}

int main()
{
    std::cout << "Testing COSMA float<->double conversion correctness" << std::endl;
    std::cout << "sizeof(cosma_scalar_t) = " << sizeof(cosma_scalar_t) << " bytes" << std::endl;
    std::cout << std::endl;

    test_simple_conversion();
    test_matrix_conversion();
    test_accumulation_precision();

    std::cout << "\n=== Analysis ===" << std::endl;
    std::cout << "If all tests PASSED: The float<->double conversion itself is NOT the bug." << std::endl;
    std::cout << "If tests FAILED: We found the conversion bug!" << std::endl;
    std::cout << "\nIf conversion is correct, the bug must be in:" << std::endl;
    std::cout << "  1. Matrix layout/stride handling in scatter_row_major_dest_local()" << std::endl;
    std::cout << "  2. COSMA coordinate mapping (local_coordinates/global_coordinates)" << std::endl;
    std::cout << "  3. Ownership normalization in reconstruct_matrix()" << std::endl;
    std::cout << "  4. MPI communication/synchronization in the distributed path" << std::endl;

    return 0;
}
