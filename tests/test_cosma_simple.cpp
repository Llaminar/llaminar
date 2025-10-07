// Simple test to check if COSMA multiply is consistent
#include <cosma/cosma.hpp>
#include <cosma/strategy.hpp>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Simple 4×4×4 test
    const int m = 4, n = 4, k = 4;
    
    cosma::Strategy strategy(m, n, k, size);
    
    if (rank == 0) {
        std::cout << "Strategy: " << strategy << std::endl;
    }
    
    cosma::CosmaMatrix<float> A('A', strategy, rank, false);
    cosma::CosmaMatrix<float> B('B', strategy, rank, false);
    cosma::CosmaMatrix<float> C('C', strategy, rank, false);
    
    A.allocate();
    B.allocate();
    C.allocate();
    
    std::cout << "[Rank " << rank << "] A.matrix_size()=" << A.matrix_size() << std::endl;
    std::cout << "[Rank " << rank << "] B.matrix_size()=" << B.matrix_size() << std::endl;
    std::cout << "[Rank " << rank << "] C.matrix_size()=" << C.matrix_size() << std::endl;
    
    // Fill with simple pattern
    float* a_ptr = A.matrix_pointer();
    float* b_ptr = B.matrix_pointer();
    float* c_ptr = C.matrix_pointer();
    
    for (size_t i = 0; i < A.matrix_size(); ++i) {
        auto [row, col] = A.global_coordinates(static_cast<int>(i));
        a_ptr[i] = (row >= 0 && col >= 0) ? (row * k + col + 1.0f) : 0.0f;
        if (rank == 0 && i < 5) {
            std::cout << "  A[" << i << "] -> global(" << row << "," << col << ") = " << a_ptr[i] << std::endl;
        }
    }
    
    for (size_t i = 0; i < B.matrix_size(); ++i) {
        auto [row, col] = B.global_coordinates(static_cast<int>(i));
        b_ptr[i] = (row >= 0 && col >= 0) ? (row * n + col + 1.0f) : 0.0f;
    }
    
    for (size_t i = 0; i < C.matrix_size(); ++i) {
        c_ptr[i] = 0.0f;
    }
    
    // Multiply
    cosma::multiply(A, B, C, strategy, MPI_COMM_WORLD, 1.0f, 0.0f);
    
    // Check result on rank 0
    if (rank == 0) {
        std::cout << "\nResult C (first 5 elements):" << std::endl;
        for (size_t i = 0; i < std::min(5ul, C.matrix_size()); ++i) {
            auto [row, col] = C.global_coordinates(static_cast<int>(i));
            std::cout << "  C[" << i << "] -> global(" << row << "," << col << ") = " << c_ptr[i] << std::endl;
        }
    }
    
    MPI_Finalize();
    return 0;
}
