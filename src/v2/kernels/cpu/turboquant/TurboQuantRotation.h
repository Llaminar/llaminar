/**
 * @file TurboQuantRotation.h
 * @brief Random orthogonal rotation matrix for TurboQuant
 * @author David Sanftenberg
 *
 * Generates a deterministic random orthogonal matrix Π ∈ R^{d×d} via QR
 * decomposition of a Gaussian random matrix. The existing seeded RNG and
 * floating generation program are retained; cross-standard-library matrix-byte
 * reproducibility is a separate qualification, not inferred from seed identity.
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
 * Matvecs select native NEON/AVX2/AVX-512 without unavailable-ISA substitutes.
 * Original matrix/query/output borrows are checked before dispatch. Rotation
 * generation remains a cold builder; inference allocates no matrix/head mirror.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <stdexcept>
#include <random>
#include <vector>

#include "../../../utils/CPUFeatures.h"
#include "../primitives/VectorPrimitives.h"

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

        /** @brief Read one original matrix word.
         * @param row Valid original row; @param col Valid original column.
         * @return Original row-major coefficient; no storage is materialized.
         */
        float at(int row, int col) const { return matrix[std::size_t(row) * std::size_t(dim) + col]; }

        /** @brief Borrow one original immutable row.
         * @param r Valid original row; @return Original row root with widened addressing.
         */
        const float *row_ptr(int r) const { return matrix.data() + std::size_t(r) * std::size_t(dim); }
        /** @brief Borrow one row during cold matrix construction.
         * @param r Valid original row; @return Original writable builder row.
         */
        float *row_ptr(int r) { return matrix.data() + std::size_t(r) * std::size_t(dim); }
    };

    namespace rotation_detail
    {
        /** @brief Price exact matrix geometry without allocating or maintaining a capacity ledger.
         * @param d Nonnegative original dimension.
         * @return Exact square element count, with a representable complete byte envelope.
         * @throws std::invalid_argument For negative or overflowing geometry.
         */
        inline std::size_t matrixElements(int d)
        {
            if (d < 0) throw std::invalid_argument("CPU rotation dimension must be nonnegative");
            const auto width = std::size_t(d);
            if (width && (width > std::numeric_limits<std::size_t>::max() / width ||
                width * width > std::numeric_limits<std::size_t>::max() / sizeof(float)))
                throw std::invalid_argument("CPU rotation matrix byte geometry overflows");
            return width * width;
        }
        /** @brief Require complete original matrix storage before any coefficient access.
         * @param rotation Original matrix owner and dimension.
         * @return Exact complete square element count.
         * @throws std::invalid_argument If the original owner does not match its geometry.
         */
        inline std::size_t requireMatrix(const TurboQuantRotation &rotation)
        {
            const auto elements = matrixElements(rotation.dim);
            if (rotation.matrix.size() != elements)
                throw std::invalid_argument("CPU rotation matrix extent disagrees with its dimension");
            return elements;
        }
        /** @brief Compare borrowed byte envelopes without overflowing end addresses.
         * @param a First root; @param an Its bytes; @param b Second root; @param bn Its bytes.
         * @return Whether the nonempty original envelopes intersect.
         */
        inline bool overlaps(const void *a, std::size_t an, const void *b, std::size_t bn) noexcept
        {
            if (!an || !bn) return false;
            const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
            return x <= y ? y - x < an : x - y < bn;
        }
        /** @brief Checked synchronous original operands; no owner lifetime or device state is copied. */
        class Invocation final
        {
        public:
            /** @brief Exclude all output corruption of original input, matrix and descriptor bytes.
             * @param rotation Original immutable complete matrix; @param input Original FP32 vector.
             * @param output Original writable result, disjoint from every retained read authority.
             * @return Sealed synchronous borrows; empty geometry accesses no vector bytes.
             * @throws std::invalid_argument For malformed, missing or aliased storage.
             */
            static Invocation bind(const TurboQuantRotation &rotation, const float *input, float *output)
            {
                (void)requireMatrix(rotation);
                const auto bytes = std::size_t(rotation.dim) * sizeof(float);
                // The vector owns unused capacity too. A result may not turn
                // those physical bytes into a competing mutable authority.
                const auto matrix_bytes = rotation.matrix.capacity() * sizeof(float);
                if (bytes && (!input || !output || overlaps(output, bytes, input, bytes) ||
                    overlaps(output, bytes, rotation.matrix.data(), matrix_bytes) ||
                    overlaps(output, bytes, &rotation, sizeof(rotation))))
                    throw std::invalid_argument("CPU rotation result aliases or lacks original vector/matrix/descriptor storage");
                return Invocation(rotation, input, output);
            }
            /** @return Original immutable matrix owner/geometry, not a transposed copy. */
            const TurboQuantRotation &rotation() const noexcept { return *rotation_; }
            /** @return Original immutable input root. */
            const float *input() const noexcept { return input_; }
            /** @return Original disjoint result root. */
            float *output() const noexcept { return output_; }
            /** @return Checked complete dimension; zero is a noncollective empty operation. */
            int dimensions() const noexcept { return rotation_->dim; }
        private:
            /** @brief Seal original checked borrows without allocation or publication.
             * @param rotation Checked complete matrix; @param input Checked input; @param output Checked result.
             */
            Invocation(const TurboQuantRotation &rotation, const float *input, float *output) noexcept
                : rotation_(&rotation), input_(input), output_(output) {}
            const TurboQuantRotation *rotation_;
            const float *input_;
            float *output_;
        };
    } // namespace rotation_detail

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
     * @throws std::invalid_argument For nonpositive or unrepresentable matrix geometry.
     */
    inline TurboQuantRotation generate_rotation_matrix(
        int head_dim, uint64_t seed = 0)
    {
        // NOTE: This is a one-time init cost (called once per head_dim at model load).
        // Heap allocations below are acceptable; the hot path is apply_rotation().
        if (seed == 0)
            seed = 31ULL;

        const int d = head_dim;
        if (d <= 0) throw std::invalid_argument("A generated CPU rotation requires a positive dimension");
        const auto elements = rotation_detail::matrixElements(d);
        TurboQuantRotation rot;
        rot.dim = d;
        rot.seed = seed;
        rot.matrix.resize(elements);

        // Step 1: Fill with N(0,1) random values
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (std::size_t i = 0; i < elements; ++i)
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
                    dot += rot.matrix[std::size_t(k) * d + i] * rot.matrix[std::size_t(k) * d + j];

                // a_j -= r_ij * q_i
                for (int k = 0; k < d; ++k)
                    rot.matrix[std::size_t(k) * d + j] -= dot * rot.matrix[std::size_t(k) * d + i];
            }

            // Normalize column j
            float norm_sq = 0.0f;
            for (int k = 0; k < d; ++k)
                norm_sq += rot.matrix[std::size_t(k) * d + j] * rot.matrix[std::size_t(k) * d + j];
            float norm = std::sqrt(norm_sq);

            r_diag[j] = norm;

            if (norm > 1e-10f)
            {
                float inv_norm = 1.0f / norm;
                for (int k = 0; k < d; ++k)
                    rot.matrix[std::size_t(k) * d + j] *= inv_norm;
            }
        }

        // Step 3: Fix sign — ensure R[j][j] > 0 for Haar-distributed Q
        // If R[j][j] < 0, negate column j of Q
        for (int j = 0; j < d; ++j)
        {
            if (r_diag[j] < 0.0f)
            {
                for (int k = 0; k < d; ++k)
                    rot.matrix[std::size_t(k) * d + j] = -rot.matrix[std::size_t(k) * d + j];
            }
        }

        // Step 4: Transpose from column-major Q to row-major Π
        // Currently Q is stored column-major in rot.matrix (MGS produced columns).
        // We need row-major for efficient matvec Π·x.
        std::vector<float> transposed(elements);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                transposed[std::size_t(i) * d + j] = rot.matrix[std::size_t(j) * d + i];

        rot.matrix = std::move(transposed);

        return rot;
    }

    // ========================================================================
    // Rotation application: y = Π · x
    // ========================================================================

    /**
     * @brief Apply rotation: y = Π · x
     *
     * Computes d scalar diagnostic dot products in original increasing K order.
     * Production callers bind original operands through apply_rotation().
     *
     * @param rot Rotation matrix
     * @param x Input vector (length = rot.dim)
     * @param y Output vector (length = rot.dim)
     */
    inline void apply_rotation_scalar(const TurboQuantRotation &rot,
                                      const float *x, float *y)
    {
        const int d = rot.dim;
        for (int i = 0; i < d; ++i)
        {
            const float *row = rot.row_ptr(i);
            float sum = 0.0f;
            for (int j = 0; j < d; ++j)
                sum += row[j] * x[j];
            y[i] = sum;
        }
    }

#if defined(__AVX2__)
    /** @brief Existing AVX2 forward reduction with a bounded original K fringe.
     * @param rot Checked original matrix; @param x Complete immutable input; @param y Disjoint result.
     */
    inline void apply_rotation_avx2(const TurboQuantRotation &rot,
                                    const float *x, float *y)
    {
        const int d = rot.dim;
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
    }
#endif

#if defined(__AVX512F__)
    /** @brief Existing AVX-512 forward tree, sharing loads across four original output rows.
     * @param rot Checked original matrix; @param x Complete immutable input; @param y Disjoint result.
     * @details Partial K vectors remain bounded and finish after the same full-vector fold.
     */
    inline void apply_rotation_avx512(const TurboQuantRotation &rot,
                                      const float *x, float *y)
    {
        const int d = rot.dim;
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

            int j = 0;
            for (; j + 16 <= d; j += 16)
            {
                __m512 v = _mm512_loadu_ps(x + j);
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(r0 + j), v, acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(r1 + j), v, acc1);
                acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(r2 + j), v, acc2);
                acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(r3 + j), v, acc3);
            }
            float s0 = _mm512_reduce_add_ps(acc0), s1 = _mm512_reduce_add_ps(acc1);
            float s2 = _mm512_reduce_add_ps(acc2), s3 = _mm512_reduce_add_ps(acc3);
            for (; j < d; ++j)
            {
                s0 = std::fma(r0[j], x[j], s0); s1 = std::fma(r1[j], x[j], s1);
                s2 = std::fma(r2[j], x[j], s2); s3 = std::fma(r3[j], x[j], s3);
            }
            y[i] = s0; y[i + 1] = s1; y[i + 2] = s2; y[i + 3] = s3;
        }
        for (; i < d; ++i)
        {
            const float *row = rot.row_ptr(i);
            __m512 acc = _mm512_setzero_ps();
            int j = 0;
            for (; j + 16 <= d; j += 16)
            {
                __m512 r = _mm512_loadu_ps(row + j);
                __m512 v = _mm512_loadu_ps(x + j);
                acc = _mm512_fmadd_ps(r, v, acc);
            }
            float sum = _mm512_reduce_add_ps(acc);
            for (; j < d; ++j) sum = std::fma(row[j], x[j], sum);
            y[i] = sum;
        }
    }
#endif


    /** @brief Explicit scalar diagnostic transpose program with original increasing row order.
     * @param rot Complete original matrix; @param y Immutable input; @param x Disjoint result.
     */
    inline void apply_rotation_transpose_scalar(const TurboQuantRotation &rot,
                                                const float *y, float *x)
    {
        const int d = rot.dim;
        for (int j = 0; j < d; ++j)
            x[j] = 0.0f;
        for (int i = 0; i < d; ++i)
        {
            const float *row = rot.row_ptr(i);
            float yi = y[i];
            for (int j = 0; j < d; ++j)
                x[j] += yi * row[j];
        }
    }

#if defined(__AVX2__)
    /** @brief Existing AVX2 coordinate-local transpose FMA program with bounded zeroing/tails.
     * @param rot Checked original matrix; @param y Immutable input; @param x Disjoint result.
     */
    inline void apply_rotation_transpose_avx2(const TurboQuantRotation &rot,
                                              const float *y, float *x)
    {
        const int d = rot.dim;
        int zero_column = 0;
        for (; zero_column + 8 <= d; zero_column += 8)
            _mm256_storeu_ps(x + zero_column, _mm256_setzero_ps());
        for (; zero_column < d; ++zero_column) x[zero_column] = 0.0f;

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
    }
#endif

#if defined(__AVX512F__)
    /** @brief Existing AVX-512 ordered four-row transpose update, including bounded output fringes.
     * @param rot Checked original matrix; @param y Immutable input; @param x Disjoint result.
     */
    inline void apply_rotation_transpose_avx512(const TurboQuantRotation &rot,
                                                const float *y, float *x)
    {
        const int d = rot.dim;
        int zero_column = 0;
        for (; zero_column + 16 <= d; zero_column += 16)
            _mm512_storeu_ps(x + zero_column, _mm512_setzero_ps());
        for (; zero_column < d; ++zero_column) x[zero_column] = 0.0f;

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

            int j = 0;
            for (; j + 16 <= d; j += 16)
            {
                __m512 xv = _mm512_loadu_ps(x + j);
                xv = _mm512_fmadd_ps(y0, _mm512_loadu_ps(r0 + j), xv);
                xv = _mm512_fmadd_ps(y1, _mm512_loadu_ps(r1 + j), xv);
                xv = _mm512_fmadd_ps(y2, _mm512_loadu_ps(r2 + j), xv);
                xv = _mm512_fmadd_ps(y3, _mm512_loadu_ps(r3 + j), xv);
                _mm512_storeu_ps(x + j, xv);
            }
            for (; j < d; ++j)
            {
                float value = std::fma(y[i], r0[j], x[j]);
                value = std::fma(y[i + 1], r1[j], value);
                value = std::fma(y[i + 2], r2[j], value);
                x[j] = std::fma(y[i + 3], r3[j], value);
            }
        }
        for (; i < d; ++i)
        {
            const float *row = rot.row_ptr(i);
            __m512 yi = _mm512_set1_ps(y[i]);
            int j = 0;
            for (; j + 16 <= d; j += 16)
            {
                __m512 r = _mm512_loadu_ps(row + j);
                __m512 xv = _mm512_loadu_ps(x + j);
                xv = _mm512_fmadd_ps(yi, r, xv);
                _mm512_storeu_ps(x + j, xv);
            }
            for (; j < d; ++j) x[j] = std::fma(y[i], row[j], x[j]);
        }
    }
#endif


    namespace rotation_detail
    {
        /** @brief Execute explicitly requested scalar diagnostic forward arithmetic.
         * @param input Checked original matrix/vector/result borrows.
         */
        inline void apply_scalar(const Invocation &input)
        { apply_rotation_scalar(input.rotation(), input.input(), input.output()); }
        /** @brief Execute explicitly requested scalar diagnostic transpose arithmetic.
         * @param input Checked original matrix/vector/result borrows.
         */
        inline void transpose_scalar(const Invocation &input)
        { apply_rotation_transpose_scalar(input.rotation(), input.input(), input.output()); }
#if defined(__AVX2__)
        /** @brief Execute the existing AVX2 forward program on checked original storage.
         * @param input Complete disjoint original invocation.
         */
        inline void apply_avx2(const Invocation &input)
        { apply_rotation_avx2(input.rotation(), input.input(), input.output()); }
        /** @brief Execute bounded AVX2 transpose updates without changing original row order.
         * @param input Complete disjoint original invocation.
         */
        inline void transpose_avx2(const Invocation &input)
        { apply_rotation_transpose_avx2(input.rotation(), input.input(), input.output()); }
#endif
#if defined(__AVX512F__)
        /** @brief Execute the existing AVX-512 forward tree, with bounded partial-vector tails.
         * @param input Complete disjoint original invocation.
         */
        inline void apply_avx512(const Invocation &input)
        { apply_rotation_avx512(input.rotation(), input.input(), input.output()); }
        /** @brief Execute ordered AVX-512 transpose FMAs without crossing the original result extent.
         * @param input Complete disjoint original invocation.
         */
        inline void transpose_avx512(const Invocation &input)
        { apply_rotation_transpose_avx512(input.rotation(), input.input(), input.output()); }
#endif
#if LLAMINAR_COMPILED_WITH_NEON
        /** @brief Share each original query load across four native matrix rows.
         * @param input Checked matrix/query/result borrows, with no head-sized mirror.
         * @details The shared vector primitive proves the same two FMA chains,
         * pair fold and bounded tail as native serial M1; residual output rows
         * use that identical serial arithmetic, not a grouped-row replay path.
         */
        inline void apply_neon(const Invocation &input)
        {
            const int d = input.dimensions();
            int row = 0;
            for (; row + 4 <= d; row += 4)
                primitives::vec_dot_four_rows(input.input(), input.rotation().row_ptr(row), d,
                    std::size_t(d), input.output() + row, 1);
            for (; row < d; ++row)
                input.output()[row] = primitives::vec_dot(input.input(), input.rotation().row_ptr(row), d);
        }
        /** @brief Stream original rows through native coordinate-local FMAs in increasing row order.
         * @param input Checked original operands; the output is neither the input nor a matrix mirror.
         * @details The existing native AXPY leaf preserves each coordinate's
         * FMA order, including its bounded tail. This leaf is noncollective;
         * independent caller-owned heads may execute in an existing team.
         */
        inline void transpose_neon(const Invocation &input)
        {
            const int d = input.dimensions();
            if (d == 0) return;
            std::fill(input.output(), input.output() + d, 0.0f);
            for (int row = 0; row < d; ++row)
                primitives::vec_axpy(input.output(), input.rotation().row_ptr(row), input.input()[row], d);
        }
#endif
    } // namespace rotation_detail

    /** @brief Apply native forward rotation to original disjoint operands.
     * @param rot Original complete immutable matrix; @param x Original FP32 input.
     * @param y Original writable result; may not alias input, matrix or descriptor bytes.
     * @throws std::invalid_argument For malformed geometry or missing/aliased original storage.
     * @details No worker-team creation, allocation, cache mirror or matrix transposition occurs.
     */
    inline void apply_rotation(const TurboQuantRotation &rot, const float *x, float *y)
    {
        const auto input = rotation_detail::Invocation::bind(rot, x, y);
        ISA_DISPATCH_VOID(rotation_detail::apply, input);
    }
    /** @brief Apply the original matrix transpose without materializing a transposed matrix.
     * @param rot Original immutable matrix; @param y Original FP32 input; @param x Disjoint result.
     * @throws std::invalid_argument For malformed geometry or missing/aliased original storage.
     * @details Every output coordinate consumes original rows in increasing order.
     */
    inline void apply_rotation_transpose(const TurboQuantRotation &rot, const float *y, float *x)
    {
        const auto input = rotation_detail::Invocation::bind(rot, y, x);
        ISA_DISPATCH_VOID(rotation_detail::transpose, input);
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
     * @throws std::invalid_argument For a malformed original matrix extent.
     * @throws std::domain_error For a nonfinite verification dot; NaN must never pass as zero error.
     */
    inline float verify_orthogonality(const TurboQuantRotation &rot)
    {
        (void)rotation_detail::requireMatrix(rot);
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

                // NaN compares false against max_err and would otherwise
                // certify a corrupted matrix as exactly orthogonal. This is
                // cold verification, not a hot-path full-matrix prescan.
                if (!std::isfinite(dot))
                    throw std::domain_error("CPU rotation orthogonality verification observed a nonfinite dot");
                float expected = (i == j) ? 1.0f : 0.0f;
                float err = std::abs(dot - expected);
                if (err > max_err)
                    max_err = err;
            }
        }

        return max_err;
    }

} // namespace llaminar2
