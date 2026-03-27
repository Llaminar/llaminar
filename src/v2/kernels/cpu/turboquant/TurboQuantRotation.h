/**
 * @file TurboQuantRotation.h
 * @brief Random orthogonal rotation matrix for TurboQuant
 * @author David Sanftenberg
 *
 * Generates a deterministic random orthogonal matrix Π ∈ R^{d×d} via QR
 * decomposition of a Gaussian random matrix. The same seed + head_dim always
 * produces the same rotation, ensuring reproducibility.
 *
 * The rotation serves to decorrelate vector coordinates before scalar
 * quantization, which is key to TurboQuant's near-optimal distortion rate.
 *
 * Provides:
 *   - generate_rotation_matrix(): QR decomposition of N(0,1) random matrix
 *   - apply_rotation():    y = Π · x  (quantize path)
 *   - apply_rotation_transpose(): x = Π^T · y  (dequantize path)
 *
 * The QR decomposition uses Modified Gram-Schmidt (numerically stable,
 * no LAPACK dependency). For head_dim ≤ 128, this runs in < 1ms.
 *
 * The matvec operations use AVX-512 when available for production performance.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2
{

    /**
     * @brief Holds a d×d orthogonal rotation matrix (row-major).
     *
     * The matrix is stored as a contiguous float array of size d*d.
     * Row i of the matrix is at offset i*d.
     *
     * Π[i][j] = matrix_[i * dim + j]
     */
    struct TurboQuantRotation
    {
        int dim;                   ///< Dimension d (head_dim)
        std::vector<float> matrix; ///< Row-major d×d orthogonal matrix
        uint64_t seed;             ///< Seed used for generation

        /** @brief Access Π[row][col] */
        float at(int row, int col) const { return matrix[row * dim + col]; }

        /** @brief Pointer to row `r` */
        const float *row_ptr(int r) const { return matrix.data() + r * dim; }
        float *row_ptr(int r) { return matrix.data() + r * dim; }
    };

    // ========================================================================
    // Rotation Matrix Generation
    // ========================================================================

    /**
     * @brief Generate deterministic d×d orthogonal rotation matrix.
     *
     * Algorithm:
     *   1. Fill d×d matrix A with N(0,1) random values (seeded)
     *   2. QR decomposition via Modified Gram-Schmidt: A = Q · R
     *   3. Fix sign ambiguity: ensure R[i][i] > 0 (Haar-distributed Q)
     *   4. Return Q as the rotation matrix
     *
     * The seed is derived from head_dim to be model-independent but
     * dimension-specific. Different head_dims get different rotations.
     *
     * @param head_dim Dimension of the rotation matrix
     * @param seed Optional seed override (default: derived from head_dim)
     * @return TurboQuantRotation with the generated matrix
     */
    inline TurboQuantRotation generate_rotation_matrix(
        int head_dim, uint64_t seed = 0)
    {
        // NOTE: This is a one-time init cost (called once per head_dim at model load).
        // Heap allocations below are acceptable; the hot path is apply_rotation().
        if (seed == 0)
            seed = 31ULL;

        const int d = head_dim;
        TurboQuantRotation rot;
        rot.dim = d;
        rot.seed = seed;
        rot.matrix.resize(d * d);

        // Step 1: Fill with N(0,1) random values
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < d * d; ++i)
            rot.matrix[i] = dist(rng);

        // Step 2: Modified Gram-Schmidt QR decomposition
        // Process columns of A to produce orthonormal columns of Q.
        // We work in-place: rot.matrix holds Q columns after orthogonalization.

        // Temporary storage for R diagonal (for sign fixing)
        std::vector<float> r_diag(d);

        for (int j = 0; j < d; ++j)
        {
            // Orthogonalize column j against all previous columns
            for (int i = 0; i < j; ++i)
            {
                // r_ij = <q_i, a_j>
                float dot = 0.0f;
                for (int k = 0; k < d; ++k)
                    dot += rot.matrix[k * d + i] * rot.matrix[k * d + j];

                // a_j -= r_ij * q_i
                for (int k = 0; k < d; ++k)
                    rot.matrix[k * d + j] -= dot * rot.matrix[k * d + i];
            }

            // Normalize column j
            float norm_sq = 0.0f;
            for (int k = 0; k < d; ++k)
                norm_sq += rot.matrix[k * d + j] * rot.matrix[k * d + j];
            float norm = std::sqrt(norm_sq);

            r_diag[j] = norm;

            if (norm > 1e-10f)
            {
                float inv_norm = 1.0f / norm;
                for (int k = 0; k < d; ++k)
                    rot.matrix[k * d + j] *= inv_norm;
            }
        }

        // Step 3: Fix sign — ensure R[j][j] > 0 for Haar-distributed Q
        // If R[j][j] < 0, negate column j of Q
        for (int j = 0; j < d; ++j)
        {
            if (r_diag[j] < 0.0f)
            {
                for (int k = 0; k < d; ++k)
                    rot.matrix[k * d + j] = -rot.matrix[k * d + j];
            }
        }

        // Step 4: Transpose from column-major Q to row-major Π
        // Currently Q is stored column-major in rot.matrix (MGS produced columns).
        // We need row-major for efficient matvec Π·x.
        std::vector<float> transposed(d * d);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                transposed[i * d + j] = rot.matrix[j * d + i];

        rot.matrix = std::move(transposed);

        return rot;
    }

    // ========================================================================
    // Rotation application: y = Π · x
    // ========================================================================

    /**
     * @brief Apply rotation: y = Π · x
     *
     * Computes d dot products (one per row of Π) against the input vector x.
     * Uses AVX-512 when available.
     *
     * @param rot Rotation matrix
     * @param x Input vector (length = rot.dim)
     * @param y Output vector (length = rot.dim)
     */
    inline void apply_rotation(const TurboQuantRotation &rot,
                               const float *x, float *y)
    {
        const int d = rot.dim;

#if defined(__AVX512F__)
        // AVX-512: 4-way row unroll — computes 4 dot products per outer iteration,
        // loading x once per 4 rows instead of per row. Improves ILP and cache reuse.
        // d must be a multiple of 4 (always true for 64, 128).
        int i = 0;
        for (; i + 4 <= d; i += 4)
        {
            const float *r0 = rot.row_ptr(i);
            const float *r1 = rot.row_ptr(i + 1);
            const float *r2 = rot.row_ptr(i + 2);
            const float *r3 = rot.row_ptr(i + 3);
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (int j = 0; j < d; j += 16)
            {
                __m512 v = _mm512_loadu_ps(x + j);
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(r0 + j), v, acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(r1 + j), v, acc1);
                acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(r2 + j), v, acc2);
                acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(r3 + j), v, acc3);
            }

            y[i] = _mm512_reduce_add_ps(acc0);
            y[i + 1] = _mm512_reduce_add_ps(acc1);
            y[i + 2] = _mm512_reduce_add_ps(acc2);
            y[i + 3] = _mm512_reduce_add_ps(acc3);
        }
        // Tail: remaining rows (if d is not a multiple of 4)
        for (; i < d; ++i)
        {
            const float *row = rot.row_ptr(i);
            __m512 acc = _mm512_setzero_ps();
            for (int j = 0; j < d; j += 16)
            {
                __m512 r = _mm512_loadu_ps(row + j);
                __m512 v = _mm512_loadu_ps(x + j);
                acc = _mm512_fmadd_ps(r, v, acc);
            }
            y[i] = _mm512_reduce_add_ps(acc);
        }
#elif defined(__AVX2__)
        // AVX2: process 8 floats per iteration
        for (int i = 0; i < d; ++i)
        {
            const float *row = rot.row_ptr(i);
            __m256 acc = _mm256_setzero_ps();

            int j = 0;
            for (; j + 8 <= d; j += 8)
            {
                __m256 r = _mm256_loadu_ps(row + j);
                __m256 v = _mm256_loadu_ps(x + j);
                acc = _mm256_fmadd_ps(r, v, acc);
            }

            // Horizontal sum
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 sum128 = _mm_add_ps(lo, hi);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float sum = _mm_cvtss_f32(sum128);

            for (; j < d; ++j)
                sum += row[j] * x[j];

            y[i] = sum;
        }
#else
        // Scalar fallback
        for (int i = 0; i < d; ++i)
        {
            const float *row = rot.row_ptr(i);
            float sum = 0.0f;
            for (int j = 0; j < d; ++j)
                sum += row[j] * x[j];
            y[i] = sum;
        }
#endif
    }

    /**
     * @brief Apply inverse rotation: x = Π^T · y
     *
     * Since Π is orthogonal, Π^(-1) = Π^T. This computes x[j] = Σ_i Π[i][j] · y[i],
     * which is the same as transposing Π and doing a regular matvec.
     *
     * For efficiency with row-major Π, we accumulate: x += y[i] * row_i(Π)
     * which is a series of scaled vector additions (saxpy-like).
     *
     * @param rot Rotation matrix
     * @param y Input vector (length = rot.dim)
     * @param x Output vector (length = rot.dim), will be overwritten
     */
    inline void apply_rotation_transpose(const TurboQuantRotation &rot,
                                         const float *y, float *x)
    {
        const int d = rot.dim;

        // Zero output with AVX-512
#if defined(__AVX512F__)
        for (int j = 0; j < d; j += 16)
            _mm512_storeu_ps(x + j, _mm512_setzero_ps());
#else
        for (int j = 0; j < d; ++j)
            x[j] = 0.0f;
#endif

#if defined(__AVX512F__)
        // 4-way row unroll: accumulate 4 scaled rows per outer iteration.
        // Inner loop does 4 FMAs per x load/store → 4x fewer memory ops.
        int i = 0;
        for (; i + 4 <= d; i += 4)
        {
            const float *r0 = rot.row_ptr(i);
            const float *r1 = rot.row_ptr(i + 1);
            const float *r2 = rot.row_ptr(i + 2);
            const float *r3 = rot.row_ptr(i + 3);
            __m512 y0 = _mm512_set1_ps(y[i]);
            __m512 y1 = _mm512_set1_ps(y[i + 1]);
            __m512 y2 = _mm512_set1_ps(y[i + 2]);
            __m512 y3 = _mm512_set1_ps(y[i + 3]);

            for (int j = 0; j < d; j += 16)
            {
                __m512 xv = _mm512_loadu_ps(x + j);
                xv = _mm512_fmadd_ps(y0, _mm512_loadu_ps(r0 + j), xv);
                xv = _mm512_fmadd_ps(y1, _mm512_loadu_ps(r1 + j), xv);
                xv = _mm512_fmadd_ps(y2, _mm512_loadu_ps(r2 + j), xv);
                xv = _mm512_fmadd_ps(y3, _mm512_loadu_ps(r3 + j), xv);
                _mm512_storeu_ps(x + j, xv);
            }
        }
        // Tail: remaining rows
        for (; i < d; ++i)
        {
            const float *row = rot.row_ptr(i);
            __m512 yi = _mm512_set1_ps(y[i]);
            for (int j = 0; j < d; j += 16)
            {
                __m512 r = _mm512_loadu_ps(row + j);
                __m512 xv = _mm512_loadu_ps(x + j);
                xv = _mm512_fmadd_ps(yi, r, xv);
                _mm512_storeu_ps(x + j, xv);
            }
        }
#elif defined(__AVX2__)
        for (int i = 0; i < d; ++i)
        {
            const float *row = rot.row_ptr(i);
            __m256 yi = _mm256_set1_ps(y[i]);

            int j = 0;
            for (; j + 8 <= d; j += 8)
            {
                __m256 r = _mm256_loadu_ps(row + j);
                __m256 xv = _mm256_loadu_ps(x + j);
                xv = _mm256_fmadd_ps(yi, r, xv);
                _mm256_storeu_ps(x + j, xv);
            }

            for (; j < d; ++j)
                x[j] += y[i] * row[j];
        }
#else
        // Scalar fallback
        for (int i = 0; i < d; ++i)
        {
            const float *row = rot.row_ptr(i);
            float yi = y[i];
            for (int j = 0; j < d; ++j)
                x[j] += yi * row[j];
        }
#endif
    }

    // ========================================================================
    // Verification utilities
    // ========================================================================

    /**
     * @brief Verify that a rotation matrix is orthogonal: Π · Π^T ≈ I
     *
     * Returns the maximum absolute off-diagonal element of Π · Π^T
     * and the maximum deviation of diagonal elements from 1.0.
     *
     * @param rot Rotation matrix to verify
     * @return Max absolute error from identity matrix
     */
    inline float verify_orthogonality(const TurboQuantRotation &rot)
    {
        const int d = rot.dim;
        float max_err = 0.0f;

        for (int i = 0; i < d; ++i)
        {
            for (int j = 0; j < d; ++j)
            {
                // Compute (Π · Π^T)[i][j] = Σ_k Π[i][k] · Π[j][k]
                float dot = 0.0f;
                for (int k = 0; k < d; ++k)
                    dot += rot.at(i, k) * rot.at(j, k);

                float expected = (i == j) ? 1.0f : 0.0f;
                float err = std::abs(dot - expected);
                if (err > max_err)
                    max_err = err;
            }
        }

        return max_err;
    }

} // namespace llaminar2
