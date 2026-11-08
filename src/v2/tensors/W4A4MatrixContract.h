/**
 * @file W4A4MatrixContract.h
 * @brief Checked, immutable source policy for native W4A4/H matrices.
 *
 * Checked geometry bounds allocation; complete source metadata is validated
 * before immutable publication or captured launch. The sealed value owns
 * geometry/policy, not memory; physical storage remains with the tensor and
 * PhysicalMemoryAuthority. The same checked extent feeds loader views and BOM
 * inputs rather than independent capacity arithmetic. K slices cannot masquerade
 * as complete rows: their activation-unit and Hadamard semantics differ.
 */
#pragma once

#include "W4A4Format.h"

#include <expected>
#include <optional>
#include <span>
#include <utility>

namespace llaminar2::w4a4
{
    /** @brief Precise admission failures for untrusted native GGUF bytes. */
    enum class AdmissionFailure : std::uint8_t
    {
        UnsupportedTransform, InvalidGeometry, IntegerRange, SizeOverflow,
        SourceExtent, InvalidUnit, InconsistentRowUnit, InvalidRotation,
        InconsistentRotation, InvalidClip, InconsistentClip, ZeroMultiplier
    };

    /** @brief Position of a malformed block within its declared matrix. */
    struct SourceLocation
    {
        std::size_t row;   ///< Logical output row.
        std::size_t block; ///< Source block within that row.
    };

    /** @brief Structured user-input error; geometry errors have no source location. */
    struct AdmissionError
    {
        AdmissionFailure failure;               ///< Failed numerical/storage invariant.
        std::optional<SourceLocation> location;  ///< Exact malformed block when available.
        /** @return Stable diagnostic text; the loader supplies the tensor name/location. */
        [[nodiscard]] const char *message() const noexcept;
    };

    /**
     * @brief Find the pinned ascending-Paley factorization without allocating.
     * @param length Positive transform length from source metadata.
     * @return Exact geometry, or null when the transform is not supported.
     */
    [[nodiscard]] std::optional<HadamardGeometry> factorHadamard(int length) noexcept;

    class MatrixAdmission;

    /** @brief Immutable admitted geometry and operand policy for one complete native matrix. */
    class MatrixContract final
    {
    public:
        /**
         * @brief Admit the actual source, rejecting malformed data without throwing.
         * @param transform Explicit source type's online transform.
         * @param rows Number of complete output rows.
         * @param columns Complete input width, not an independently quantized K shard.
         * @param source Exact typed block extent; ownership remains with the caller.
         * @return Sealed policy or a precise source/geometry failure.
         */
        [[nodiscard]] static std::expected<MatrixContract, AdmissionError> admit(
            Transform transform, std::size_t rows, std::size_t columns,
            std::span<const SourceBlock> source) noexcept;

        /**
         * @brief Compute checked source bytes for admission/BOM construction.
         * @param rows Logical output count.
         * @param columns Complete reduction width.
         * @return Exact bytes or the same geometry error used by source admission.
         */
        [[nodiscard]] static std::expected<std::size_t, AdmissionError> sourceExtent(
            std::size_t rows, std::size_t columns) noexcept;

        /**
         * @brief Derive complete output rows from this already admitted immutable matrix.
         * @param first First output row in this matrix, not a source K coordinate.
         * @param count Positive number of complete rows to retain.
         * @return Same arithmetic policy with checked subset geometry, or an extent error.
         *
         * No metadata rescan is needed: every retained block was admitted and
         * remains immutable. This operation cannot create a K-fragment policy.
         */
        [[nodiscard]] std::expected<MatrixContract, AdmissionError> rowSlice(
            std::size_t first, std::size_t count) const noexcept
        {
            if (first > static_cast<std::size_t>(rows_) || count == 0 ||
                count > static_cast<std::size_t>(rows_) - first)
                return std::unexpected(AdmissionError{AdmissionFailure::InvalidGeometry, std::nullopt});
            const auto extent = sourceExtent(count, columns_);
            if (!extent) return std::unexpected(extent.error());
            return MatrixContract(static_cast<int>(count), columns_, *extent, transform_,
                                  rotation_length_, hadamard_, clip_bits_, clip_);
        }

        /** @return Admitted output-row count, representable by kernel dimension types. */
        [[nodiscard]] int rows() const noexcept { return rows_; }
        /** @return Complete reduction width, with a proven int32 accumulation bound. */
        [[nodiscard]] int columns() const noexcept { return columns_; }
        /** @return Native source blocks in one complete row. */
        [[nodiscard]] int blocksPerRow() const noexcept { return columns_ / kBlockElements; }
        /** @return Arithmetic groups in one complete activation row. */
        [[nodiscard]] int groupsPerRow() const noexcept { return columns_ / kGroupElements; }
        /** @return Exact source-storage bytes, not a live communication extent. */
        [[nodiscard]] std::size_t sourceBytes() const noexcept { return source_bytes_; }
        /** @return Source-declared transform; never an automatic dispatch alternative. */
        [[nodiscard]] Transform transform() const noexcept { return transform_; }
        /** @return Explicit stored row/rotation length. */
        [[nodiscard]] int rotationLength() const noexcept { return rotation_length_; }
        /** @return Admitted transform geometry, absent for ordinary W4A4. */
        [[nodiscard]] const std::optional<HadamardGeometry> &hadamard() const noexcept { return hadamard_; }
        /** @return Original encoded clipping bits for source/graph identity. */
        [[nodiscard]] std::uint32_t encodedClipBits() const noexcept { return clip_bits_; }
        /** @return Resolved finite positive FP32 clipping scalar; unset zero resolves to one. */
        [[nodiscard]] float activationClip() const noexcept { return clip_; }

    private:
        friend class MatrixAdmission;
        /** @brief Construct only after complete admission; no default/partially sealed state exists. */
        MatrixContract(int rows, int columns, std::size_t bytes, Transform transform,
                       int rotation_length, std::optional<HadamardGeometry> hadamard,
                       std::uint32_t clip_bits, float clip) noexcept
            : rows_(rows), columns_(columns), source_bytes_(bytes), transform_(transform),
              rotation_length_(rotation_length), hadamard_(hadamard), clip_bits_(clip_bits), clip_(clip) {}

        int rows_;                               ///< Admitted N.
        int columns_;                            ///< Admitted complete K.
        std::size_t source_bytes_;               ///< Checked native extent, shared with the physical BOM.
        Transform transform_;                    ///< Source numerical policy.
        int rotation_length_;                    ///< Source header's complete row/transform block width.
        std::optional<HadamardGeometry> hadamard_; ///< Typed absence for identity, not a sentinel geometry.
        std::uint32_t clip_bits_;                 ///< Original encoding retained independently of resolution.
        float clip_;                             ///< Exact resolved scalar embedded into execution identity.
    };
    /**
     * @brief Move-only complete-source admission across bounded original block windows.
     *
     * The state retains only geometry, the current row's unit and immutable
     * header policy. Payload bytes remain in the caller's unpublished final
     * allocation. Arbitrary block windows may split rows, but only complete
     * original coverage can publish a MatrixContract. Failure is absorbing.
     */
    class MatrixAdmission final
    {
    public:
        /**
         * @brief Validate original geometry before accepting any payload.
         * @param transform Source type's explicit online transform.
         * @param rows Complete original output extent, not a selected subset.
         * @param columns Complete original K.
         * @return Unpublished admission state or the precise geometry failure.
         */
        [[nodiscard]] static std::expected<MatrixAdmission, AdmissionError> begin(
            Transform transform, std::size_t rows, std::size_t columns) noexcept;
        /**
         * @brief Check consecutive complete source blocks without retaining their bytes.
         * @param source Next exact block-aligned byte extent; address alignment is unrestricted.
         * @return Success or the first error in original row/block coordinates.
         */
        [[nodiscard]] std::expected<void, AdmissionError> append(std::span<const std::byte> source) noexcept;
        /**
         * @brief Consume this state and seal only complete, successful source coverage.
         * @return Immutable matrix policy or the original failure/incomplete-extent error.
         */
        [[nodiscard]] std::expected<MatrixContract, AdmissionError> finish() && noexcept;
        /** @brief Transfer the sole unpublished state; the previous owner becomes consumed.
         * @param other Original admission owner.
         */
        MatrixAdmission(MatrixAdmission &&other) noexcept : state_(std::exchange(other.state_, std::nullopt)) {}
        /** @brief Retire the current state and transfer the sole incoming authority.
         * @param other Original admission owner.
         * @return This new owner.
         */
        MatrixAdmission &operator=(MatrixAdmission &&other) noexcept
        {
            if (this != &other) state_ = std::exchange(other.state_, std::nullopt);
            return *this;
        }
        MatrixAdmission(const MatrixAdmission &) = delete;
        MatrixAdmission &operator=(const MatrixAdmission &) = delete;

    private:
        /** @brief Fixed metadata only; no physical storage or allocation ledger. */
        struct State
        {
            int rows, columns;
            std::size_t source_bytes;
            Transform transform;
            std::size_t blocks_seen = 0;
            std::uint32_t row_unit = 0;
            int rotation = 0;
            std::optional<HadamardGeometry> hadamard{};
            std::uint32_t clip_bits = 0;
            float clip = 0;
            std::optional<AdmissionError> failure{};
        };
        /** @brief Create only after checked geometry; partial policy cannot escape finish(). */
        explicit MatrixAdmission(State state) noexcept : state_(std::move(state)) {}
        std::optional<State> state_; ///< Absence means consumed/moved-from, never a second publisher.
    };
} // namespace llaminar2::w4a4
