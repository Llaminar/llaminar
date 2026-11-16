/**
 * @file MoEPrefillProjectionExecution.h
 * @brief Exact-arithmetic boundaries in a grouped expert projection pipeline.
 *
 * A routed FFN may end a graph stage after producing its existing quantized or
 * FP32 SwiGLU rows, or begin a stage at those rows and publish a down-projection
 * column slice. This changes neither the activation representation nor the
 * order of the reduction over K. In particular, a column slice inherits the
 * full matrix's serial-M1 arithmetic policy, not a policy selected for its
 * smaller physical N. The slice's prepared weights must contain exactly the
 * declared columns in their normal backend-native layout.
 *
 * This capture-time value owns no memory or execution state. Its factories
 * reject impossible geometry before recording a graph. Producers and consumers
 * still require explicit event/collective edges and persistent, matching route
 * maps; this type does not imply that an orchestration topology is supported.
 * Public down outputs begin with unspecified bytes. A successful complete
 * producer must overwrite every token or original-route contribution, including
 * zero contributions for absent routes, before publishing the output event.
 */
#pragma once

#include "WorkspaceOverwriteContract.h"
#include "DeviceRowRange.h"
#include "MoEGroupingPackedDirectory.h"

#include <stdexcept>
#include <cstdint>
#include <utility>
#include <optional>
#include <exception>
#include <memory>

namespace llaminar2
{
    /** @brief Physical publication layout, independent of allocation capacity. */
    enum class MoEPrefillOutputLayout
    {
        TokenRows, ///< One ordered sum for each physical token and output column.
        OriginalRouteRows ///< One contribution for each original token/top-k slot.
    };

    /**
     * @brief Capture-time provenance of the grouping producer, never live counts.
     *
     * Unspecified and selected-route publications cannot certify an all-route
     * measurement, even when a particular replay happens to contain all routes.
     */
    enum class MoEPrefillRouteScope : std::uint8_t
    {
        Unspecified, ///< No successful all-route grouping producer is bound.
        AllRoutes, ///< The ordinary unmasked grouping producer owns every route.
        SelectedRoutes, ///< A mask or published runtime assignment may select routes.
        SharedVerifierCountedOriginalRows ///< Explicit verifier, original counted rows and one implicit shared route.
    };

    /** @brief Shared grouping's declared execution purpose, independent of live count contents. */
    enum class MoESharedExpertGroupKind : std::uint8_t { Ordinary, DecodeEquivalentVerifier };

    /**
     * @brief Own static grouping-origin publication for one kernel instance.
     *
     * This records which setup producer successfully enqueued a grouping, not
     * its live rows, experts or device counts. Beginning a replacement revokes
     * prior provenance immediately. Failure, exception and abandoned setup keep
     * it unspecified; only a completed producer may publish its fixed origin.
     */
    class MoEPrefillGroupingProvenance final
    {
    public:
        /** @brief Start with no successfully bound grouping producer. */
        MoEPrefillGroupingProvenance() = default;
        /** @brief Producer provenance cannot be copied to another kernel owner. */
        MoEPrefillGroupingProvenance(const MoEPrefillGroupingProvenance &) = delete;
        /** @brief Rebinding an owner must invalidate rather than copy provenance. */
        MoEPrefillGroupingProvenance &operator=(const MoEPrefillGroupingProvenance &) = delete;

        /** @brief Revoke provenance when grouping buffers or their owner change. */
        void invalidate() noexcept
        {
            scope_ = MoEPrefillRouteScope::Unspecified;
            shared_rows_.reset();
            revokeDirectory();
        }

        /** @brief Revoke work-bank authority before another planner overwrites it, retaining route origin. */
        void revokeDirectory() noexcept
        {
            packed_directory_.reset();
            if (++generation_ == 0) std::terminate();
        }

        /** @brief Non-owning, non-detachable directory authority for one synchronous submission. */
        class DirectoryBorrow final
        {
        public:
            DirectoryBorrow(const DirectoryBorrow &) = delete;
            DirectoryBorrow &operator=(const DirectoryBorrow &) = delete;
            /** @return Whether this original generation published the optional layout.
             * @throws std::logic_error If reset or another grouping revoked the borrow. */
            [[nodiscard]] bool present() const { validate(); return directory_.has_value(); }
            /** @brief Authenticate all original banks and geometry before recording a consumer.
             * @param expected Exact current consuming workspace/stream/geometry declaration.
             * @return Original published bindings. @throws std::logic_error For absence or mismatch. */
            [[nodiscard]] const MoEGroupingPackedDirectory &requireMatching(
                const MoEGroupingPackedDirectory &expected) const
            {
                validate();
                if (!directory_ || *directory_ != expected)
                    throw std::logic_error("grouped packed directory consumer changed its original producer binding");
                return *directory_;
            }
            /** @brief Authenticate the actual native consumer packet against this live borrow.
             * @param counts Original count bank. @param offsets Original offset bank.
             * @param words Packed work bank. @param rows Physical token capacity.
             * @param top_k Original route multiplicity. @param experts Expert domain.
             * @param device Exact owning device. @param stream Exact ordered stream.
             * @param layout Exact work encoding expected by this consumer.
             * @return Original published binding; no authority is transferred.
             * @throws std::logic_error For a revoked, missing or incompatible publication. */
            [[nodiscard]] const MoEGroupingPackedDirectory &requireProjection(
                const int *counts, const int *offsets, const uint32_t *words,
                int rows, int top_k, int experts, DeviceId device, void *stream,
                MoEGroupingPackedLayout layout = MoEGroupingPackedLayout::WideFloor8) const
            {
                validate();
                if (!directory_ || directory_->counts() != counts || directory_->offsets() != offsets ||
                    directory_->words() != words || directory_->rows() != rows || directory_->topK() != top_k ||
                    directory_->experts() != experts || directory_->device() != device || directory_->stream() != stream ||
                    directory_->layout() != layout)
                    throw std::logic_error("grouped packed directory does not bind the submitted projection packet");
                return *directory_;
            }
        private:
            friend class MoEPrefillGroupingProvenance;
            /** @brief Snapshot immutable metadata while retaining its sole publication authority. */
            explicit DirectoryBorrow(const MoEPrefillGroupingProvenance &owner)
                : owner_(owner), lifetime_(owner.lifetime_), generation_(owner.generation_), directory_(owner.packed_directory_) {}
            /** @brief A copied pointer set cannot resurrect revoked publication authority. */
            void validate() const
            {
                if (lifetime_.expired() || owner_.preparing_ || owner_.generation_ != generation_)
                    throw std::logic_error("grouped packed directory borrow was revoked by preparation or reset");
            }
            const MoEPrefillGroupingProvenance &owner_;
            std::weak_ptr<const uint8_t> lifetime_; ///< Cold owner identity; checked before dereferencing its storage.
            uint64_t generation_;
            std::optional<MoEGroupingPackedDirectory> directory_;
        };

        /** @return A lexical borrow that cannot be copied or moved out as ownership.
         * @throws std::logic_error While an incomplete grouping still owns publication. */
        [[nodiscard]] DirectoryBorrow borrowDirectory() const
        {
            if (preparing_) throw std::logic_error("cannot borrow a directory from incomplete grouping");
            return DirectoryBorrow(*this);
        }

        /** @return The last successfully published static producer's origin. */
        [[nodiscard]] MoEPrefillRouteScope scope() const noexcept { return scope_; }

        /** @brief Authenticate a consuming projection against its original grouping.
         * @param rows Actual physical projection rows, not the current live count.
         * @param top_k Original route slots per token.
         * @return The successfully published origin for this exact consumer.
         * @throws std::logic_error If a shared verifier reuses a different row authority's grouping. */
        [[nodiscard]] MoEPrefillRouteScope scopeForProjection(int rows, int top_k) const
        {
            if (scope_ == MoEPrefillRouteScope::SharedVerifierCountedOriginalRows &&
                (!shared_rows_ || shared_rows_->physical_rows != rows || shared_rows_->count_capacity != rows || top_k != 1))
                throw std::logic_error("shared verifier projection changed its prepared row geometry");
            return scope_;
        }

        /** @brief One noncopyable setup publication; it never owns device work. */
        class Preparation final
        {
        public:
            /** @brief One setup transaction has exactly one publication owner. */
            Preparation(const Preparation &) = delete;
            /** @brief A live setup transaction cannot be reassigned. */
            Preparation &operator=(const Preparation &) = delete;

            /** @brief Failed or abandoned setup cannot retain old provenance. */
            ~Preparation()
            {
                if (owner_ && !lifetime_.expired())
                {
                    owner_->invalidate();
                    owner_->preparing_ = false;
                }
            }

            /**
             * @brief Publish only after the entire grouping producer succeeded.
             * @param directory Optional original packed layout submitted by the
             * combined grouping publisher; other producers must leave it absent.
             * @throws std::logic_error if a consumed preparation is reused.
             * @note Invocation success means enqueued on its existing stream;
             *       no device count is inspected and no completion wait is added.
             */
            void publish(std::optional<MoEGroupingPackedDirectory> directory = std::nullopt) &&
            {
                if (!owner_ || lifetime_.expired())
                    throw std::logic_error("grouped MoE origin publication already consumed");
                if (directory && owner_->generation_ != generation_)
                    throw std::logic_error("packed directory preparation was revoked before publication");
                if (directory && scope_ != MoEPrefillRouteScope::AllRoutes)
                    throw std::logic_error("packed work directory requires an ordinary all-route grouping producer");
                owner_->scope_ = scope_;
                owner_->shared_rows_ = shared_rows_;
                owner_->packed_directory_ = std::move(directory);
                owner_->preparing_ = false;
                owner_ = nullptr;
            }

        private:
            friend class MoEPrefillGroupingProvenance;
            /**
             * @brief Revoke the prior producer before any replacement can fail.
             * @param owner Kernel-instance metadata receiving this publication.
             * @param scope Fixed origin selected by the typed producer factory.
             * @param shared_rows Original immutable verifier row identity, never a count value.
             * @throws std::logic_error before altering an already active owner.
             */
            Preparation(MoEPrefillGroupingProvenance &owner, MoEPrefillRouteScope scope,
                std::optional<DeviceRowRange::CaptureIdentity> shared_rows = std::nullopt)
                : owner_(&owner), lifetime_(owner.lifetime_), scope_(scope), shared_rows_(shared_rows)
            {
                if (owner.preparing_)
                    throw std::logic_error("grouped MoE origin preparation overlaps an active producer");
                owner.preparing_ = true;
                owner.invalidate();
                generation_ = owner.generation_;
            }
            MoEPrefillGroupingProvenance *owner_;
            std::weak_ptr<const uint8_t> lifetime_; ///< Preparation cannot publish through a destroyed/replaced owner.
            MoEPrefillRouteScope scope_;
            std::optional<DeviceRowRange::CaptureIdentity> shared_rows_;
            uint64_t generation_ = 0; ///< Packed storage cannot survive an intervening invalidation.
        };

        /**
         * @return An unpublished transaction for ordinary unmasked grouping.
         * @throws std::logic_error if another producer still owns preparation.
         */
        [[nodiscard]] Preparation beginAllRoutes()
        {
            return Preparation(*this, MoEPrefillRouteScope::AllRoutes);
        }

        /**
         * @return An unpublished transaction for a mask-selected grouping.
         * @throws std::logic_error if another producer still owns preparation.
         */
        [[nodiscard]] Preparation beginSelectedRoutes()
        {
            return Preparation(*this, MoEPrefillRouteScope::SelectedRoutes);
        }

        /** @brief Begin an explicit shared grouping without inspecting device state.
         * @param kind Declared ordinary or serial-equivalent verifier operation.
         * @param rows Original capture geometry and borrowed count authority.
         * @return Unpublished origin; only original counted verifier rows qualify.
         * @throws std::invalid_argument For an unknown purpose, revoking old provenance.
         * @throws std::logic_error If another preparation owns this instance. */
        [[nodiscard]] Preparation beginSharedExpert(MoESharedExpertGroupKind kind, DeviceRowRange rows)
        {
            if (preparing_)
                throw std::logic_error("shared grouping origin overlaps an active producer");
            invalidate();
            if (kind != MoESharedExpertGroupKind::Ordinary && kind != MoESharedExpertGroupKind::DecodeEquivalentVerifier)
                throw std::invalid_argument("unknown shared expert grouping purpose");
            const auto identity = rows.captureIdentity();
            const bool counted_verifier = kind == MoESharedExpertGroupKind::DecodeEquivalentVerifier &&
                identity.count && identity.count_capacity == identity.physical_rows &&
                identity.first_physical_row == 0 && identity.repeats_per_row == 1;
            return Preparation(*this, counted_verifier ? MoEPrefillRouteScope::SharedVerifierCountedOriginalRows
                                                       : MoEPrefillRouteScope::Unspecified,
                counted_verifier ? std::optional{identity} : std::nullopt);
        }

    private:
        MoEPrefillRouteScope scope_ = MoEPrefillRouteScope::Unspecified;
        std::optional<DeviceRowRange::CaptureIdentity> shared_rows_; ///< Capture metadata only; device owns the count value.
        std::optional<MoEGroupingPackedDirectory> packed_directory_; ///< Original binding metadata, never live route counts.
        const std::shared_ptr<const uint8_t> lifetime_ = std::make_shared<const uint8_t>(0); ///< Allocated only with the cold kernel owner.
        uint64_t generation_ = 1; ///< Revokes lexical borrows on every replacement/reset.
        bool preparing_ = false; ///< Exactly one setup publication owns this instance.
    };

    /**
     * @brief Meaning of the already-installed complete-operation measurements.
     *
     * This is a qualification of existing corpus rows, not a new phase policy.
     * Unmeasured transactions retain the unchanged generic geometry and their
     * original arithmetic key. New operation families need independent evidence.
     */
    enum class MoEPrefillMeasuredOperation : std::uint8_t
    {
        Unmeasured,
        CompleteFullWidthTokenRowsAllRoutes,
        CompleteFullWidthSharedVerifierTokenRows
    };

    /** @brief One explicit projection transaction, without implicit replay. */
    class MoEPrefillProjectionExecution final
    {
    public:
        /** @brief Record the existing gate/up, activation, down and publication. */
        [[nodiscard]] static MoEPrefillProjectionExecution complete(int model_columns)
        {
            return {Phase::Complete, model_columns, 0, model_columns};
        }

        /**
         * @brief Stop after ordinary grouped SwiGLU publication.
         *
         * The caller retains the existing activation representation and route
         * map: Q8 plus FP32 block scales for quants, FP32 for floating experts.
         * Down output and down scratch are not publication targets here.
         */
        [[nodiscard]] static MoEPrefillProjectionExecution gateUp(int model_columns)
        {
            return {Phase::GateUp, model_columns, 0, model_columns};
        }

        /**
         * @brief Consume complete intermediate rows and publish an output slice.
         * @param model_columns Original unsharded output width and policy key.
         * @param first_column First source weight row in the prepared slice.
         * @param column_count Number of prepared rows and compact output stride.
         *
         * K and its reduction order are unchanged. Each output column is
         * independent, so concatenating these slices requires no FP32 sum.
         */
        [[nodiscard]] static MoEPrefillProjectionExecution down(
            int model_columns, int first_column, int column_count)
        {
            return {Phase::Down, model_columns, first_column, column_count};
        }

        /** @return Whether this transaction owns input quantization and gate/up. */
        [[nodiscard]] bool executesGateUp() const noexcept { return phase_ != Phase::Down; }

        /** @return Whether this transaction owns down projection and publication. */
        [[nodiscard]] bool executesDown() const noexcept { return phase_ != Phase::GateUp; }

        /** @return Full source width used to resolve serial-M1 arithmetic. */
        [[nodiscard]] int modelColumns() const noexcept { return model_columns_; }

        /** @return Original matrix column at the start of the prepared slice. */
        [[nodiscard]] int firstColumn() const noexcept { return first_column_; }

        /** @return Physical down-descriptor width and compact output stride. */
        [[nodiscard]] int columnCount() const noexcept { return column_count_; }

        /**
         * @brief Qualify the operation represented by existing complete corpora.
         * @param layout Actual down publication target, not allocation shape.
         * @param scope Immutable provenance of the current grouping producer.
         * @return The installed complete-operation identity only for full-width
         *         gate/up plus down, token publication and unmasked all routes.
         * @throws std::invalid_argument for an unknown layout or scope.
         * @note A full-width down-only operation is still not a complete FFN.
         *       This decision never reads mutable route counts or changes K.
         */
        [[nodiscard]] MoEPrefillMeasuredOperation measuredOperation(
            MoEPrefillOutputLayout layout, MoEPrefillRouteScope scope) const
        {
            if (layout != MoEPrefillOutputLayout::TokenRows &&
                layout != MoEPrefillOutputLayout::OriginalRouteRows)
                throw std::invalid_argument("unknown grouped MoE measured publication layout");
            if (scope != MoEPrefillRouteScope::Unspecified &&
                scope != MoEPrefillRouteScope::AllRoutes &&
                scope != MoEPrefillRouteScope::SelectedRoutes &&
                scope != MoEPrefillRouteScope::SharedVerifierCountedOriginalRows)
                throw std::invalid_argument("unknown grouped MoE measured route scope");
            if (phase_ != Phase::Complete || first_column_ != 0 || column_count_ != model_columns_ ||
                layout != MoEPrefillOutputLayout::TokenRows)
                return MoEPrefillMeasuredOperation::Unmeasured;
            if (scope == MoEPrefillRouteScope::AllRoutes)
                return MoEPrefillMeasuredOperation::CompleteFullWidthTokenRowsAllRoutes;
            if (scope == MoEPrefillRouteScope::SharedVerifierCountedOriginalRows)
                return MoEPrefillMeasuredOperation::CompleteFullWidthSharedVerifierTokenRows;
            return MoEPrefillMeasuredOperation::Unmeasured;
        }

        /**
         * @brief Describe all down-output elements that this producer must define.
         * @param rows Physical token rows consumed by the next graph stage.
         * @param top_k Positive original route count per token.
         * @param layout Ordered token sums or original-route contributions.
         * @return Exact consumed extent with the prepared slice's compact stride.
         * @throws std::logic_error if gate/up attempts to own a down publication.
         * @throws std::invalid_argument for negative rows or nonpositive top-k.
         * @throws std::overflow_error if the live product is unrepresentable.
         * @note An inactive route still owns a public zero value. Compact,
         *       inactive intermediate scratch is not part of this extent.
         */
        [[nodiscard]] WorkspaceOverwriteExtent outputOverwriteExtent(
            int rows, int top_k, MoEPrefillOutputLayout layout) const
        {
            if (!executesDown())
                throw std::logic_error("gate/up does not own a grouped MoE down publication");
            if (rows < 0 || top_k <= 0)
                throw std::invalid_argument("grouped MoE output overwrite requires nonnegative rows and positive top-k");
            std::size_t planes;
            switch (layout)
            {
            case MoEPrefillOutputLayout::TokenRows: planes = 1u; break;
            case MoEPrefillOutputLayout::OriginalRouteRows:
                planes = static_cast<std::size_t>(top_k); break;
            default: throw std::invalid_argument("unknown grouped MoE output overwrite layout");
            }
            return WorkspaceOverwriteExtent::matrix(
                static_cast<std::size_t>(rows), static_cast<std::size_t>(column_count_), planes);
        }

        /**
         * @brief Enqueue the complete projection before admitting output publication.
         * @param storage Bound persistent FP32 publication storage; dirty bytes are valid.
         * @param capacity_bytes Accessible storage at that pointer, never the write extent.
         * @param rows Consumed physical token rows, including zero-valued inactive rows.
         * @param top_k Positive number of original routes per token.
         * @param layout Typed publication layout.
         * @param stream Exact non-null producer/publication stream.
         * @param producer Immediate launch builder receiving the writable live span
         *        and exact stream. Gate/up alone receives an empty output span.
         * @param publisher Immediate event/coherence publisher receiving the ordered
         *        read receipt; never invoked for gate/up or a failed producer.
         * @return Complete producer/publication success, without a clear or wait.
         * @throws std::invalid_argument for invalid geometry, storage or stream.
         * @throws std::length_error if the publication cannot fit its bound storage.
         * @throws std::overflow_error if publication bytes are unrepresentable.
         */
        template <class Producer, class Publisher>
        [[nodiscard]] bool overwriteOutputThenPublish(
            float *storage, std::size_t capacity_bytes, int rows, int top_k,
            MoEPrefillOutputLayout layout, void *stream,
            Producer &&producer, Publisher &&publisher) const
        {
            if (!stream || rows < 0 || top_k <= 0)
                throw std::invalid_argument("grouped MoE overwrite requires an exact stream and valid row/route geometry");
            if (!executesDown())
                return std::forward<Producer>(producer)(std::span<float>{}, stream);
            return WorkspaceOverwrite<float>::bind(
                storage, capacity_bytes, outputOverwriteExtent(rows, top_k, layout), stream)
                .overwriteThenRead(std::forward<Producer>(producer),
                                   std::forward<Publisher>(publisher));
        }

    private:
        /** @brief Closed transaction kinds; callers construct only valid cases. */
        enum class Phase { Complete, GateUp, Down };

        /** @brief Reject invalid or overflowing ranges before capture begins. */
        MoEPrefillProjectionExecution(
            Phase phase, int model_columns, int first_column, int column_count)
            : phase_(phase), model_columns_(model_columns),
              first_column_(first_column), column_count_(column_count)
        {
            // A floating source need not have block-quantized geometry. Each
            // backend's quantized launch validates its own block alignment;
            // projection ownership itself only defines the source interval.
            if (model_columns <= 0 ||
                first_column < 0 || first_column >= model_columns ||
                column_count <= 0 || column_count > model_columns - first_column)
                throw std::invalid_argument("grouped MoE projection requires a valid source column range");
        }

        Phase phase_;
        int model_columns_;
        int first_column_;
        int column_count_;
    };
} // namespace llaminar2
