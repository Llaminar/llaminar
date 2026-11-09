/**
 * @file CPUNativeNEONGemv.h
 * @brief Native ARM implementation of the canonical CPU prepared-projection API.
 *
 * The CPU factory, prepared owners, workspace names and request coordinates are
 * unchanged. Synchronous NEON batches consume the published interleaved bank;
 * all arithmetic stays in native shared-row tiles. Ordinary rows own full-K
 * chains, experts own their canonical register-resident ordered tree. Neither
 * requires a shared K-partial allocation. No x86 intrinsic, weight repack, row
 * inference replay, allocation or coherence publication belongs in this layer.
 */
#pragma once
#include "CPUNativeQuantisedExecutionContract.h"
#include "CPUNativeQuantisedPackedWeights.h"
#include "CPUNativeNEONProjectionBatch.h"
#include "CPUNativeQuantisedScalarReference.h"
#include "CPUNativeQuantisedWeightPacker.h"
#include "CPUProjectionQ8Publication.h"
#include "execution/moe/DeviceMoERuntimeABI.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>

namespace llaminar2::cpu::native_quantised
{
    namespace neon_execution
    {
        /** @brief Resolve one authenticated instruction family before worksharing.
         * @param path Explicit native ISA or automatic process policy.
         * @return Native signed-byte dot family, never a diagnostic/scalar path.
         * @throws std::invalid_argument For a foreign, uncompiled or non-native route.
         */
        inline multi_scale::NEONByteDot instruction(ISAPath path)
        {
            requireNativeQuantisedISAPath(path);
            const auto active = activeISALevel();
            if (path == ISAPath::NEON || (path == ISAPath::AUTO && active == ISALevel::NEON))
                return multi_scale::NEONByteDot::ASIMD;
#if defined(__ARM_FEATURE_DOTPROD)
            if (path == ISAPath::NEON_DOTPROD || (path == ISAPath::AUTO && active == ISALevel::NEONDotProd))
            {
                ARMCPUCapabilities::native().require(ARMCPUFeature::DotProd);
                return multi_scale::NEONByteDot::DotProd;
            }
#endif
            throw std::invalid_argument("Native ARM projection requires an admitted native execution ISA");
        }

        /** @brief Validate metadata without reading payloads or consulting local cache state.
         * @param packed Actual source/encoding/arithmetic descriptor.
         * @param rows Positive physical row capacity.
         * @param stride Physical output row stride, at least N.
         * @throws std::invalid_argument For incomplete geometry/source/policy metadata.
         */
        inline void geometry(const CPUNativeQuantisedPackedWeights &packed, int rows, int stride)
        {
            if (rows <= 0 || packed.N <= 0 || packed.K <= 0 || stride < packed.N ||
                std::int64_t(packed.N_padded) != (std::int64_t(packed.N) + 63) / 64 * 64 ||
                std::int64_t(packed.blocks_per_row) != (std::int64_t(packed.K) + 31) / 32 ||
                !native_quantised_formats::forSourceIdentity(packed.codebook_id, packed.is_superblock) ||
                (packed.numerical_policy != CPUProjectionNumericalPolicy::BackendNative &&
                 packed.numerical_policy != CPUProjectionNumericalPolicy::GPUAlignedExpert) ||
                (packed.numerical_policy == CPUProjectionNumericalPolicy::GPUAlignedExpert && packed.K % 32 != 0))
                throw std::invalid_argument("Native ARM projection requires exact positive source/policy geometry");
        }

        /** @brief Select physical shared-row width independently of the K arithmetic.
         * @param requested Explicit physical native reuse or automatic total policy.
         * @return Native two/four-row tile, including bounded useful tails.
         * @throws std::invalid_argument For an x86-only or unknown candidate.
         */
        inline NEONProjectionRowTile rowTile(VerifierRowsPolicy requested)
        {
            switch (requested)
            {
            case VerifierRowsPolicy::Auto:
            case VerifierRowsPolicy::WideRows: return NEONProjectionRowTile::Four;
            case VerifierRowsPolicy::Pairwise: return NEONProjectionRowTile::Pair;
            default: throw std::invalid_argument("ARM grouped projection requires a native Pairwise/WideRows candidate");
            }
        }

        /** @brief Resolve native physical N ownership without borrowing x86 timing labels.
         * @param requested Automatic, explicit Nbc, or the M1 diagnostic ownership oracle.
         * @return Positive adjacent 64-column chunks per task.
         */
        inline int nChunks(DecodeSchedulePolicy requested)
        {
            if (requested == DecodeSchedulePolicy::Auto || requested == DecodeSchedulePolicy::FrozenSerialOracle) return 1;
            return decodeScheduleNBlockChunks(requested);
        }

        /** @brief Build checked whole-bank spans from one actual retained prepared owner.
         * @param packed Actual immutable prepared owner; no deferred workspace binding.
         * @param input Persistent original-order Q8 rows.
         * @param output Exclusive original-order live output bank.
         * @param rows Physical M capacity.
         * @param stride Original physical output row stride.
         * @param tile Explicit native row reuse.
         * @param chunks Positive adjacent column chunks per owner.
         * @param work Optional immutable original request publication.
         * @param bias Optional complete immutable bias row.
         * @return Allocation-free borrowed native binding.
         */
        inline CPUNativeNEONProjectionBatch bind(const CPUNativeQuantisedPackedWeights &packed,
            const Q8_1Block *input, float *output, int rows, int stride, NEONProjectionRowTile tile,
            int chunks, const ProjectionRowWork *work = nullptr, const float *bias = nullptr)
        {
            geometry(packed, rows, stride);
            if (!input || !output) throw std::invalid_argument("Native ARM projection requires retained input/output storage");
            storage_proof::requireDisjoint({output, (std::size_t(rows - 1) * stride + packed.N) * sizeof(float)},
                {&packed, sizeof(packed)});
            return CPUNativeNEONProjectionBatch::bind(packed.readView(),
                {input, std::size_t(rows) * packed.blocks_per_row},
                {output, std::size_t(rows - 1) * stride + packed.N}, rows, stride, tile, work,
                bias ? std::span<const float>(bias, packed.N) : std::span<const float>{}, chunks);
        }

        /** @brief Execute a complete native binding after one ISA decision.
         * @param batch Bound original-coordinate synchronous operation.
         * @param dot Authenticated native integer instruction family.
         */
        inline void execute(const CPUNativeNEONProjectionBatch &batch, multi_scale::NEONByteDot dot)
        {
#if defined(__ARM_FEATURE_DOTPROD)
            if (dot == multi_scale::NEONByteDot::DotProd) return batch.execute<multi_scale::NEONByteDot::DotProd>();
#endif
            if (dot != multi_scale::NEONByteDot::ASIMD)
                throw std::invalid_argument("Native ARM projection instruction is unknown or outside this compiler envelope");
            batch.execute<multi_scale::NEONByteDot::ASIMD>();
        }

        /** @brief Publish original FP32 rows and project them within one retained/new team.
         * @param packed Retained canonical prepared owner.
         * @param input Original host-owned FP32 rows.
         * @param output Exclusive live FP32 output bank.
         * @param rows Physical row capacity, not useful row count.
         * @param stride Original output row stride.
         * @param activation Caller-owned admitted persistent Q8 workspace.
         * @param work Optional original-coordinate row publication.
         */
        inline void full(const CPUNativeQuantisedPackedWeights &packed, const float *input, float *output,
            int rows, int stride, std::span<Q8_1Block> activation, const ProjectionRowWork *work = nullptr)
        {
            geometry(packed, rows, stride);
            const CPUProjectionRowGrid grid(rows, work);
            if (!grid.beginWorkshare()) return;
            if (!input || activation.size() < std::size_t(rows) * packed.blocks_per_row)
                throw std::invalid_argument("Native ARM projection requires its admitted physical-M Q8 workspace");
            const auto used_q8_bytes = std::size_t(rows) * packed.blocks_per_row * sizeof(Q8_1Block);
            const storage_proof::Bytes writes{activation.data(), used_q8_bytes};
            const auto view = packed.readView();
            storage_proof::requireDisjoint(writes, {input, std::size_t(rows) * packed.K * sizeof(float)});
            storage_proof::requireDisjoint(writes, {view.bytes().data(), view.bytes().size()});
            storage_proof::requireDisjoint(writes, {&packed, sizeof(packed)});
            storage_proof::requireOriginalWorkReadOnly(writes, work);
            const auto dot = instruction(ISAPath::AUTO);
            const auto batch = bind(packed, activation.data(), output, rows, stride, NEONProjectionRowTile::Four, 1, work);
            auto run = [&] {
                quantize_activations_to_q8_1(input, activation.data(), rows, packed.K, packed.blocks_per_row,
                    packed.numerical_policy, work);
                execute(batch, dot);
            };
            OMP_WORKSHARE_REGION(run);
        }

        /** @brief Test bounded byte aliases without forming overflowing end addresses.
         * @param a First original storage root.
         * @param an First live byte extent.
         * @param b Second original root.
         * @param bn Second live byte extent.
         * @return Whether nonempty byte ranges intersect.
         */
        inline bool overlaps(const void *a, std::size_t an, const void *b, std::size_t bn) noexcept
        {
            if (!an || !bn) return false;
            const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
            return x <= y ? y - x < an : x - y < bn;
        }
    }

    /** @brief ARM's full-K/register-tree implementation requires no shared partial bank.
     * @param packed Actual source/arithmetic geometry.
     * @param rows Physical invocation row capacity.
     * @return Zero; canonical expert partials live only in bounded native registers.
     */
    inline std::size_t nativeQuantisedProjectionPartialFloats(const CPUNativeQuantisedPackedWeights &packed, int rows)
    { neon_execution::geometry(packed, rows, packed.N); return 0; }

    /** @copydoc nativeQuantisedProjectionPartialFloats */
    inline std::size_t nativeQuantisedProjectionPartialEnvelopeFloats(const CPUNativeQuantisedPackedWeights &packed, int rows)
    { return nativeQuantisedProjectionPartialFloats(packed, rows); }

    /** @brief Execute the canonical native M1 prequantized entrypoint.
     * @param packed Retained prepared source.
     * @param input Complete original Q8 K row.
     * @param output Exclusive N-value destination.
     * @param partial_storage Existing ABI bank; ARM needs no shared partials.
     * @param isa_path Explicit native or automatic ISA.
     * @param schedule_override Physical N ownership only.
     */
    inline void gemv_native_quantised_preq(const CPUNativeQuantisedPackedWeights &packed, const Q8_1Block *input,
        float *output, std::span<float> partial_storage, ISAPath isa_path = ISAPath::AUTO,
        DecodeSchedulePolicy schedule_override = DecodeSchedulePolicy::Auto)
    {
        (void)partial_storage;
        const auto dot = neon_execution::instruction(isa_path);
        (void)cpuNativeQuantisedSerialEquivalentPolicyN(packed.N);
        const auto batch = neon_execution::bind(packed, input, output, 1, packed.N,
            NEONProjectionRowTile::Four, neon_execution::nChunks(schedule_override));
        neon_execution::execute(batch, dot);
    }

    /** @brief Execute grouped decode through native shared-row tiles, never M1 replay.
     * @param packed Retained prepared source and arithmetic.
     * @param input Original physical-M Q8 bank.
     * @param output Exclusive output with original row stride.
     * @param partial_storage Existing ABI bank; no shared partials required on ARM.
     * @param rows Positive physical M.
     * @param stride Physical output stride in floats.
     * @param isa_path Explicit native or automatic ISA.
     * @param policy Native physical row reuse candidate.
     * @param publish_verifier_route Existing telemetry intent, not arithmetic policy.
     * @param chunks Zero for total native policy, otherwise positive exact N ownership.
     * @param work Optional original row/frontier publication.
     */
    inline void gemm_native_quantised_preq_decode_equivalent_rows(const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *input, float *output, std::span<float> partial_storage, int rows, int stride,
        ISAPath isa_path = ISAPath::AUTO, VerifierRowsPolicy policy = VerifierRowsPolicy::Auto,
        bool publish_verifier_route = true, int chunks = 0, const ProjectionRowWork *work = nullptr)
    {
        (void)partial_storage; (void)publish_verifier_route;
        requireNativeQuantisedISAPath(isa_path);
        neon_execution::geometry(packed, rows, stride);
        if (chunks < 0) throw std::invalid_argument("Native ARM N ownership cannot be negative");
        (void)cpuNativeQuantisedSerialEquivalentPolicyN(packed.N);
        const CPUProjectionRowGrid grid(rows, work);
        if (!grid.beginWorkshare()) return;
        const auto batch = neon_execution::bind(packed, input, output, rows, stride,
            neon_execution::rowTile(policy), chunks ? chunks : 1, work);
        neon_execution::execute(batch, neon_execution::instruction(isa_path));
    }

    /** @brief Ordinary native prefill retains the exact same per-row arithmetic.
     * @param packed Retained prepared source.
     * @param input Original physical-M Q8 bank.
     * @param output Exclusive physical output.
     * @param partial_storage Existing ABI bank, unused by native full-K ownership.
     * @param rows Positive physical M.
     * @param stride Original output row stride.
     * @param isa_path Explicit native or automatic ISA.
     * @param policy Explicit native row candidate or automatic selection.
     * @param schedule Explicit native prefill reuse; unsupported x86 candidates fail.
     * @param chunks Positive exact N ownership or zero for total native policy.
     * @param work Optional original row publication.
     */
    inline void gemm_native_quantised_preq(const CPUNativeQuantisedPackedWeights &packed, const Q8_1Block *input,
        float *output, std::span<float> partial_storage, int rows, int stride,
        ISAPath isa_path = ISAPath::AUTO, VerifierRowsPolicy policy = VerifierRowsPolicy::Auto,
        PrefillSchedulePolicy schedule = PrefillSchedulePolicy::Auto, int chunks = 0,
        const ProjectionRowWork *work = nullptr)
    {
        if (schedule == PrefillSchedulePolicy::TwoRowPairGrid)
        {
            if (policy != VerifierRowsPolicy::Auto && policy != VerifierRowsPolicy::Pairwise)
                throw std::invalid_argument("Native ARM prefill row candidates conflict");
            policy = VerifierRowsPolicy::Pairwise;
        }
        else if (schedule == PrefillSchedulePolicy::FourRowGrid)
        {
            if (policy != VerifierRowsPolicy::Auto && policy != VerifierRowsPolicy::WideRows)
                throw std::invalid_argument("Native ARM prefill row candidates conflict");
            policy = VerifierRowsPolicy::WideRows;
        }
        else if (schedule != PrefillSchedulePolicy::Auto)
            throw std::invalid_argument("ARM prefill requires native shared-row grid ownership");
        gemm_native_quantised_preq_decode_equivalent_rows(packed, input, output, partial_storage,
            rows, stride, isa_path, policy, false, chunks, work);
    }

    /** @brief Canonical FP32 M1 publication and native projection over admitted scratch.
     * @param packed Actual prepared source.
     * @param input Original FP32 K row.
     * @param output Exclusive N-value destination.
     * @param partial_storage Existing ABI bank, unused on ARM.
     * @param activation_storage Admitted persistent complete Q8 input bank.
     */
    inline void gemv_native_quantised(const CPUNativeQuantisedPackedWeights &packed, const float *input,
        float *output, std::span<float> partial_storage, std::span<Q8_1Block> activation_storage)
    { (void)partial_storage; neon_execution::full(packed, input, output, 1, packed.N, activation_storage); }

    /** @brief Canonical ordinary FP32 prefill over original physical rows.
     * @param packed Actual prepared source.
     * @param input Original FP32 row-major matrix.
     * @param output Exclusive strided destination.
     * @param partial_storage Existing ABI bank, unused on ARM.
     * @param rows Positive physical M.
     * @param stride Original output stride.
     * @param activation_storage Participant-owned admitted complete Q8 bank.
     */
    inline void gemm_native_quantised(const CPUNativeQuantisedPackedWeights &packed, const float *input,
        float *output, std::span<float> partial_storage, int rows, int stride, std::span<Q8_1Block> activation_storage)
    { (void)partial_storage; neon_execution::full(packed, input, output, rows, stride, activation_storage); }

    /** @brief Add immutable bias after every native row's complete K result.
     * @param output Exclusive original physical destination.
     * @param bias Retained immutable N-value bias row.
     * @param rows Physical row capacity.
     * @param columns Live columns per row.
     * @param stride Original output row stride.
     * @param work Optional original row publication.
     */
    inline void apply_bias_epilogue(float *output, const float *bias, int rows, int columns, int stride,
        const ProjectionRowWork *work = nullptr)
    {
        const CPUProjectionRowGrid grid(rows, work);
        if (columns <= 0 || stride < columns || !output || !bias)
            throw std::invalid_argument("Native ARM bias requires exact positive retained geometry");
        if (!grid.beginWorkshare()) return;
        auto add = [&] {
#pragma omp for schedule(static)
            for (int row = 0; row < rows; ++row)
            {
                if (!grid.active(row)) continue;
                auto *destination = output + std::size_t(row) * stride;
                int column = 0;
                for (; column + 4 <= columns; column += 4)
                    vst1q_f32(destination + column, vaddq_f32(vld1q_f32(destination + column), vld1q_f32(bias + column)));
                for (; column < columns; ++column) destination[column] += bias[column];
            }
        };
        OMP_WORKSHARE_REGION(add);
    }

    /** @brief Canonical BLAS epilogue over one complete native product, not serial row replay.
     * @param packed Retained prepared source.
     * @param input Original FP32 rows.
     * @param output Exclusive destination and optional prior beta addend.
     * @param rows Original physical M.
     * @param columns Exact prepared N and output stride.
     * @param alpha Product scale.
     * @param beta Prior-output scale.
     * @param activation Admitted persistent Q8 input bank.
     * @param product Admitted distinct product bank when beta is nonzero.
     * @param partial Existing ABI bank; ARM uses register-resident K trees.
     * @param work Optional original row publication.
     */
    inline void multiply_native_quantised_with_epilogue(const CPUNativeQuantisedPackedWeights &packed,
        const float *input, float *output, int rows, int columns, float alpha, float beta,
        std::span<Q8_1Block> activation, std::span<float> product, std::span<float> partial,
        const ProjectionRowWork *work = nullptr)
    {
        (void)partial;
        neon_execution::geometry(packed, rows, columns);
        if (columns != packed.N) throw std::invalid_argument("Native ARM BLAS epilogue requires the exact prepared output width");
        const CPUProjectionRowGrid grid(rows, work);
        if (!grid.beginWorkshare()) return;
        const auto elements = std::size_t(rows) * columns;
        if (!output) throw std::invalid_argument("Native ARM BLAS output is missing");
        storage_proof::requireOriginalWorkReadOnly({output, elements * sizeof(float)}, work);
        storage_proof::requireDisjoint({output, elements * sizeof(float)}, {&packed, sizeof(packed)});
        if (beta != 0.0f)
        {
            if (product.size() < elements || !product.data() ||
                neon_execution::overlaps(product.data(), elements * sizeof(float), output, elements * sizeof(float)) ||
                neon_execution::overlaps(output, elements * sizeof(float), activation.data(), activation.size_bytes()) ||
                neon_execution::overlaps(output, elements * sizeof(float), packed.readView().bytes().data(), packed.readView().bytes().size()))
                throw std::invalid_argument("Native ARM beta epilogue requires distinct admitted retained banks");
        }
        float *destination = beta == 0.0f ? output : product.data();
        neon_execution::full(packed, input, destination, rows, columns, activation, work);
        if (beta == 0.0f && alpha == 1.0f) return;
        auto epilogue = [&] {
#pragma omp for schedule(static)
            for (int row = 0; row < rows; ++row)
            {
                if (!grid.active(row)) continue;
                const auto offset = std::size_t(row) * columns;
                int column = 0;
                for (; column + 4 <= columns; column += 4)
                {
                    const auto a = persistRoundedNEON(vmulq_n_f32(vld1q_f32(destination + offset + column), alpha));
                    const auto result = beta == 0.0f ? a : vaddq_f32(a,
                        persistRoundedNEON(vmulq_n_f32(vld1q_f32(output + offset + column), beta)));
                    vst1q_f32(output + offset + column, result);
                }
                for (; column < columns; ++column)
                {
                    const auto a = device_fp32_contract::multiply(destination[offset + column], alpha);
                    output[offset + column] = beta == 0.0f ? a : device_fp32_contract::add(a,
                        device_fp32_contract::multiply(output[offset + column], beta));
                }
            }
        };
        OMP_WORKSHARE_REGION(epilogue);
    }

    /** @brief Bounded native fused planning, with no payload reads or shared partial bank. */
    struct NativeQuantisedFusedRowsPlan
    {
        static constexpr int capacity = 2 * (moe_runtime_abi::kMaxExperts + 1);
        int count = 0; ///< Number of complete independent projections.
        std::array<int, capacity> rows{}; ///< Original per-projection physical row capacity.
        std::array<int, capacity> chunks{}; ///< Exact physical N ownership per task.
        std::array<NEONProjectionRowTile, capacity> tiles{}; ///< Native shared-row reuse.
        /** @return Zero: full-K/native ordered trees do not consume shared partial memory. */
        constexpr std::size_t sharedPartialFloats() const noexcept { return 0; }
    };

    /**
     * @brief Admit an output-job ownership request on the native ARM schedule.
     * @param ownership Typed request from the shared batched-projection ABI.
     * @throws std::invalid_argument For an unknown value or an explicit x86 control.
     * @details ARM physical N ownership is already selected by the native schedule
     * (plan.chunks), so Auto and Current name that existing mapping. Contiguous and
     * PanelBlocks are x86 diagnostic job layouts with no NEON implementation; an
     * explicit request is rejected before any payload write, never reinterpreted.
     */
    inline void requireNativeNEONFusedOutputJobOwnership(FusedOutputJobOwnership ownership)
    {
        (void)fusedOutputJobOwnershipName(ownership);
        if (ownership != FusedOutputJobOwnership::Auto && ownership != FusedOutputJobOwnership::Current)
            throw std::invalid_argument(std::string("Native ARM fused rows do not implement explicit output-job ownership ") +
                fusedOutputJobOwnershipName(ownership));
    }

    /** @brief Resolve independent native bundle geometry without reading any operand data.
     * @param descriptors Actual prepared metadata and original physical output geometry.
     * @param rows Default positive physical M for members without an explicit row count.
     * @param workers Actual positive participant worker budget.
     * @param isa_path Explicit native or automatic ISA envelope.
     * @return Bounded native schedule with zero shared partial demand.
     */
    inline NativeQuantisedFusedRowsPlan planNativeQuantisedFusedRows(std::span<const FusedVerifierRowsDesc> descriptors,
        int rows, int workers, ISAPath isa_path = ISAPath::AUTO)
    {
        requireNativeQuantisedISAPath(isa_path);
        if (descriptors.empty() || descriptors.size() > NativeQuantisedFusedRowsPlan::capacity || rows <= 0 || workers <= 0)
            throw std::invalid_argument("Native ARM fused plan requires complete descriptors and positive rows/workers");
        NativeQuantisedFusedRowsPlan plan;
        plan.count = static_cast<int>(descriptors.size());
        int blocks = 0;
        for (int p = 0; p < plan.count; ++p)
        {
            const auto &d = descriptors[p];
            if (!d.packed || d.rows < 0 || d.N != d.packed->N || d.policy_n < 0 ||
                (d.purpose != ProjectionRowsPurpose::Decode && d.purpose != ProjectionRowsPurpose::Prefill))
                throw std::invalid_argument("Native ARM fused descriptor has incomplete original geometry");
            plan.rows[p] = d.rows ? d.rows : rows;
            neon_execution::geometry(*d.packed, plan.rows[p], d.ldc);
            if (blocks && blocks != d.packed->blocks_per_row)
                throw std::invalid_argument("Native ARM fused members require the same physical K width");
            blocks = d.packed->blocks_per_row;
            if (d.policy_n == 0) (void)cpuNativeQuantisedSerialEquivalentPolicyN(d.N);
            plan.tiles[p] = neon_execution::rowTile(d.verifier_schedule);
            plan.chunks[p] = neon_execution::nChunks(d.decode_schedule);
        }
        return plan;
    }

    /** @brief Execute native fused members within one retained/new OpenMP team.
     * @param input Shared persistent Q8 bank, overridden by a member's explicit input.
     * @param descriptors Independent retained prepared/input/output descriptions.
     * @param partial_storage Existing ABI bank; native partial demand is zero.
     * @param count Number of complete descriptors.
     * @param rows Default positive physical M.
     * @param blocks Exact complete Q8 blocks per row shared by all members.
     * @param isa_path Explicit native or automatic ISA.
     * @return True after successful native completion; invalid bundles throw, never select another implementation.
     */
    inline bool gemm_native_quantised_fused_verifier_rows_preq(const Q8_1Block *input,
        const FusedVerifierRowsDesc *descriptors, std::span<float> partial_storage, int count,
        int rows, int blocks, ISAPath isa_path = ISAPath::AUTO,
        FusedOutputJobOwnership job_ownership = FusedOutputJobOwnership::Auto)
    {
        (void)partial_storage;
        requireNativeNEONFusedOutputJobOwnership(job_ownership);
        if (!descriptors || count <= 0 || count > NativeQuantisedFusedRowsPlan::capacity || blocks <= 0)
            throw std::invalid_argument("Native ARM fused execution requires complete descriptor geometry");
        const auto plan = planNativeQuantisedFusedRows({descriptors, std::size_t(count)}, rows,
            nativeQuantisedInvocationWorkerCount(), isa_path);
        const auto dot = neon_execution::instruction(isa_path);
        std::array<std::optional<CPUNativeNEONProjectionBatch>, NativeQuantisedFusedRowsPlan::capacity> batches;
        for (int p = 0; p < count; ++p)
        {
            const auto &d = descriptors[p];
            if (d.packed->blocks_per_row != blocks)
                throw std::invalid_argument("Native ARM fused K width disagrees with its Q8 input declaration");
            batches[p].emplace(neon_execution::bind(*d.packed, d.input ? d.input : input, d.output,
                plan.rows[p], d.ldc, plan.tiles[p], plan.chunks[p], nullptr, d.bias));
            // A bundle denotes independent projections. A member's destination
            // cannot mutate any sibling input, weight, bias or destination.
            const auto output_bytes = (std::size_t(plan.rows[p] - 1) * d.ldc + d.N) * sizeof(float);
            storage_proof::requireDisjoint({d.output, output_bytes}, {descriptors, std::size_t(count) * sizeof(FusedVerifierRowsDesc)});
            for (int other = 0; other < count; ++other)
            {
                if (other == p) continue;
                const auto &s = descriptors[other];
                const auto other_rows = plan.rows[other];
                const auto view = s.packed->readView();
                storage_proof::requireDisjoint({d.output, output_bytes}, {s.packed, sizeof(*s.packed)});
                if (neon_execution::overlaps(d.output, output_bytes, s.input ? s.input : input,
                        std::size_t(other_rows) * blocks * sizeof(Q8_1Block)) ||
                    neon_execution::overlaps(d.output, output_bytes, view.bytes().data(), view.bytes().size()) ||
                    neon_execution::overlaps(d.output, output_bytes, s.output,
                        (std::size_t(other_rows - 1) * s.ldc + s.N) * sizeof(float)) ||
                    (s.bias && neon_execution::overlaps(d.output, output_bytes, s.bias, std::size_t(s.N) * sizeof(float))))
                    throw std::invalid_argument("Native ARM fused destination aliases a sibling publication");
            }
        }
        auto run = [&] {
            for (int p = 0; p < count; ++p) neon_execution::execute(*batches[p], dot);
        };
        OMP_WORKSHARE_REGION(run);
        return true;
    }

    /** @brief Adapt serial shared-input descriptors into the same native fused authority.
     * @param input Complete retained shared Q8 K row.
     * @param descriptors Complete retained output/source members.
     * @param partial_storage Existing ABI bank, unused on ARM.
     * @param count Positive bounded member count.
     * @param isa_path Explicit native or automatic ISA.
     * @param schedule_override Physical N ownership candidate.
     */
    inline void gemv_native_quantised_fused_preq(const Q8_1Block *input, const FusedGemvDesc *descriptors,
        std::span<float> partial_storage, int count, ISAPath isa_path = ISAPath::AUTO,
        DecodeSchedulePolicy schedule_override = DecodeSchedulePolicy::Auto)
    {
        if (!descriptors || count <= 0 || count > NativeQuantisedFusedRowsPlan::capacity)
            throw std::invalid_argument("Native ARM fused M1 requires complete bounded members");
        std::array<FusedVerifierRowsDesc, NativeQuantisedFusedRowsPlan::capacity> members{};
        for (int p = 0; p < count; ++p)
        {
            const auto &d = descriptors[p];
            members[p] = {.packed = d.packed, .output = d.output, .bias = d.bias, .N = d.N, .ldc = d.N,
                .decode_schedule = schedule_override, .policy_n = d.policy_n};
        }
        if (!descriptors[0].packed) throw std::invalid_argument("Native ARM fused M1 has no prepared source");
        (void)gemm_native_quantised_fused_verifier_rows_preq(input, members.data(), partial_storage, count, 1,
            descriptors[0].packed->blocks_per_row, isa_path);
    }

    /** @brief Adapt independent expert inputs into the same native fused authority.
     * @param descriptors Members with original retained input pointers.
     * @param partial_storage Existing ABI bank, unused on ARM.
     * @param count Positive bounded member count.
     * @param isa_path Explicit native or automatic ISA.
     * @param schedule_override Physical N ownership candidate.
     */
    inline void gemv_fused_multi_input_preq(const FusedGemvMultiInputDesc *descriptors,
        std::span<float> partial_storage, int count, ISAPath isa_path = ISAPath::AUTO,
        DecodeSchedulePolicy schedule_override = DecodeSchedulePolicy::Auto)
    {
        if (!descriptors || count <= 0 || count > NativeQuantisedFusedRowsPlan::capacity)
            throw std::invalid_argument("Native ARM fused expert M1 requires complete bounded members");
        std::array<FusedVerifierRowsDesc, NativeQuantisedFusedRowsPlan::capacity> members{};
        for (int p = 0; p < count; ++p)
        {
            const auto &d = descriptors[p];
            members[p] = {.packed = d.packed, .output = d.output, .N = d.N, .ldc = d.N, .input = d.A_q8,
                .decode_schedule = schedule_override};
        }
        if (!descriptors[0].packed) throw std::invalid_argument("Native ARM expert M1 has no prepared source");
        (void)gemm_native_quantised_fused_verifier_rows_preq(nullptr, members.data(), partial_storage, count, 1,
            descriptors[0].packed->blocks_per_row, isa_path);
    }
} // namespace llaminar2::cpu::native_quantised
