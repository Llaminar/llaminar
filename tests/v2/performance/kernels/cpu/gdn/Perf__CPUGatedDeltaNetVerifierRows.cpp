/**
 * @file Perf__CPUGatedDeltaNetVerifierRows.cpp
 * @brief Production-shaped CPU GDN grouped-verifier economy benchmark.
 *
 * This harness compares one grouped merged-QKV verifier transaction with the
 * exact production M=1 merged-QKV recurrence repeated in serial row order.  It
 * includes production immutable checkpoint and shared raw-log publication.
 * Acceptance retains pending updates without flushing; canonical flat snapshots
 * are materialized only outside timing for byte authentication.
 * Incoming epoch phases are explicit evidence, independent of grouping.
 * Capture-only persistent fixture buffers use 256-byte backing matching the production
 * arena contract. Retained capacity is declared independently of live M; set
 * LLAMINAR_CPU_GDN_VERIFIER_CAPACITY to reproduce an admitted verifier bucket.
 * Allocation, input generation, capacity setup, and state reset stay outside
 * canonical timing samples. A poisoned borrowed input bank additionally proves
 * every accepted row depends only on completed retained checkpoint/log bytes.
 *
 * Correctness is a byte contract.  Every grouped output row and every published
 * recurrence-state snapshot must match the corresponding production M=1 row.
 * The default inventory contains M=1 as a control, every supported MTP verifier
 * row count, and the canonical extended runtime-M probes.
 * The separate publication scope uses ordinary FP32 tensor owners and the
 * real stage's 256-byte compact workspace rather than claiming that every
 * TensorFactory allocation has the capture-only fixture's 256-byte backing.
 * A separate named observation times a real GDNRecurrenceStage through the
 * production CPU MTPSpecStatePublisher, including accepted canonical-state
 * scatter. Its actual measured endpoint is checked before any reset/replay;
 * full and partial accepted prefixes also prove subsequent serial continuation.
 * A disabled diagnostic additionally compares FullEncoded and Shared epoch retention
 * using native kernel capture/restore plus ordinary continuation, without claiming
 * that FullEncoded is supported by the production stage/publisher contract.
 * This local stateful-stage boundary excludes model/KV/sampling orchestration
 * and does not claim full-model MTP throughput.
 */

#include <gtest/gtest.h>

#include "kernels/cpu/gdn/CPUGatedDeltaNet.h"
#include "backends/BackendManager.h"
#include "execution/compute_stages/stages/GDNRecurrenceStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/mtp/MTPSpecStatePublisher.h"
#include "tensors/Tensors.h"
#include "kernels/CPUSharedGDNSnapshots.h"
#include "utils/VerifierRowTestInventory.h"

#include <algorithm>
#include <chrono>
#include <array>
#include <cstddef>
#include <memory>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <cerrno>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>
#include <new>

using namespace llaminar2;

namespace
{
    /** @brief Timing and geometry evidence emitted for one runtime-M case. */
    struct BenchmarkResult
    {
        int rows = 0;
        int n_k_heads = 0;
        int n_heads = 0;
        int d_k = 0;
        int d_v = 0;
        int initial_phase = 0;
        int retained_capacity = 0; ///< Declared row capacity, never inferred from bytes.
        size_t snapshot_bytes = 0;
        size_t work_bytes = 0;
        bool byte_exact = false;
        double grouped_ms = 0.0;
        double serial_ms = 0.0;
        double speedup = 0.0;
    };

    /** @return Explicit incoming phases, validated before constructing a case. */
    std::vector<int> envPhases()
    {
        const char *raw = std::getenv("LLAMINAR_CPU_GDN_VERIFIER_PHASES");
        if (!raw || !*raw) return {0};
        if (std::strcmp(raw, "all") == 0) return {0, 1, 2, 3, 4, 5, 6, 7};
        std::vector<int> result;
        std::stringstream input(raw);
        std::string field;
        while (std::getline(input, field, ','))
        {
            char *end = nullptr;
            const long phase = std::strtol(field.c_str(), &end, 10);
            if (end == field.c_str() || *end != '\0' || phase < 0 || phase >= 8)
                throw std::invalid_argument("CPU GDN verifier phases must be 0..7 or all");
            result.push_back(static_cast<int>(phase));
        }
        if (result.empty()) throw std::invalid_argument("CPU GDN verifier phases cannot be empty");
        return result;
    }

    /** @brief Parse one strictly positive integer environment override. */
    int envInt(const char *name, int fallback)
    {
        const char *raw = std::getenv(name);
        if (!raw || !*raw)
            return fallback;

        char *end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end == raw || *end != '\0' || parsed <= 0 ||
            parsed > std::numeric_limits<int>::max())
        {
            return fallback;
        }
        return static_cast<int>(parsed);
    }

    /** @brief Parse one strictly positive floating-point environment override. */
    double envDouble(const char *name, double fallback)
    {
        const char *raw = std::getenv(name);
        if (!raw || !*raw)
            return fallback;

        char *end = nullptr;
        const double parsed = std::strtod(raw, &end);
        if (end == raw || *end != '\0' || parsed <= 0.0)
            return fallback;
        return parsed;
    }

    /**
     * @brief Select runtime-M cases without imposing an artificial upper bound.
     *
     * An explicit comma-separated override is useful for profiler runs.  The
     * ordinary benchmark uses the shared complete verifier inventory, preceded
     * by M=1 so the serial boundary remains visible.
     */
    std::vector<int> envRows()
    {
        const char *raw = std::getenv("LLAMINAR_CPU_GDN_VERIFIER_M");
        if (!raw || !*raw)
        {
            std::vector<int> rows = {1};
            rows.insert(
                rows.end(),
                test::kGroupedVerifierRuntimeRows.begin(),
                test::kGroupedVerifierRuntimeRows.end());
            return rows;
        }

        std::vector<int> rows;
        std::stringstream stream(raw);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            char *end = nullptr;
            const long parsed = std::strtol(token.c_str(), &end, 10);
            if (end != token.c_str() && *end == '\0' && parsed > 0 &&
                parsed <= std::numeric_limits<int>::max())
            {
                rows.push_back(static_cast<int>(parsed));
            }
        }

        if (rows.empty())
            rows = {1, 2, 3, 4, 8, 15, 16};
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        return rows;
    }

    /**
     * @brief Report the first FP32 mismatch after an exact byte comparison.
     */
    void expectByteExact(
        const float *actual,
        const float *expected,
        size_t count,
        const std::string &context)
    {
        if (std::memcmp(actual, expected, count * sizeof(float)) == 0)
            return;

        for (size_t index = 0; index < count; ++index)
        {
            if (std::memcmp(actual + index, expected + index, sizeof(float)) != 0)
            {
                uint32_t actual_bits = 0;
                uint32_t expected_bits = 0;
                std::memcpy(&actual_bits, actual + index, sizeof(actual_bits));
                std::memcpy(&expected_bits, expected + index, sizeof(expected_bits));
                ADD_FAILURE() << context << " first mismatch at element " << index
                              << " actual=" << actual[index]
                              << " expected=" << expected[index]
                              << " actual_bits=" << actual_bits
                              << " expected_bits=" << expected_bits;
                return;
            }
        }
    }

    /**
     * @brief Stable 256-byte backing for persistent production-shaped fixture data.
     *
     * Allocation and copy construction are setup/oracle operations. Measured
     * grouped/serial invocations never construct, grow, move, or rebind these
     * buffers. Canonical state rows retain their real encoded head stride;
     * aligning the allocation must not invent a different state representation.
     */
    class AlignedFloatStorage
    {
        float *data_ = nullptr; ///< Sole owner of this fixture allocation.
        size_t size_ = 0; ///< Immutable live extent after initial setup.
    public:
        /** @brief Construct empty storage awaiting explicit fixture admission. */
        AlignedFloatStorage() = default;
        /** @brief Allocate a fixed extent before warmup. @param count Live floats. */
        explicit AlignedFloatStorage(size_t count) { resize(count); }
        /**
         * @brief Preserve a correctness observation outside canonical measurement.
         * @param other Immutable source with independently owned backing.
         */
        AlignedFloatStorage(const AlignedFloatStorage &other)
        {
            if (other.size_ == 0) return;
            resize(other.size_);
            std::memcpy(data_, other.data_, size_ * sizeof(float));
        }
        /**
         * @brief Restore an oracle observation without reallocating or rebinding.
         * @param other Equal-extent immutable source.
         * @return This stable destination.
         * @throws std::logic_error if a copy would change admitted geometry.
         */
        AlignedFloatStorage &operator=(const AlignedFloatStorage &other)
        {
            if (this == &other) return *this;
            if (size_ != other.size_)
                throw std::logic_error("CPU GDN fixture buffer extent cannot change after setup");
            if (size_ != 0) std::memcpy(data_, other.data_, size_ * sizeof(float));
            return *this;
        }
        /** @brief Release fixture backing after all measurement/oracle work. */
        ~AlignedFloatStorage() { std::free(data_); }
        /**
         * @brief Admit the sole allocation for one persistent fixture buffer.
         * @param count Positive live float count, without anonymous reserve.
         * @throws std::logic_error if this storage was already admitted.
         * @throws std::length_error for an absent or overflowing byte extent.
         * @throws std::bad_alloc if aligned allocation cannot satisfy the request.
         */
        void resize(size_t count)
        {
            if (data_) throw std::logic_error("CPU GDN fixture buffer is already admitted");
            if (count == 0 || count > std::numeric_limits<size_t>::max() / sizeof(float))
                throw std::length_error("CPU GDN fixture float extent is invalid");
            void *memory = nullptr;
            if (posix_memalign(&memory,256,count * sizeof(float)) != 0)
                throw std::bad_alloc();
            data_ = static_cast<float *>(memory);
            size_ = count;
            std::fill_n(data_,size_,0.0f);
        }
        /** @return Stable mutable allocation base. */
        float *data() { return data_; }
        /** @return Stable immutable allocation base. */
        const float *data() const { return data_; }
        /** @return First mutable live element. */
        float *begin() { return data_; }
        /** @return Exclusive live endpoint. */
        float *end() { return data_ ? data_ + size_ : nullptr; }
        /** @return First immutable live element. */
        const float *begin() const { return data_; }
        /** @return Exclusive immutable live endpoint. */
        const float *end() const { return data_ ? data_ + size_ : nullptr; }
        /** @return Indexed mutable element within the declared fixture extent. */
        float &operator[](size_t index) { return data_[index]; }
        /** @return Indexed immutable element within the declared fixture extent. */
        const float &operator[](size_t index) const { return data_[index]; }
        /** @return Exact live float count; allocation alignment adds no logical rows. */
        size_t size() const { return size_; }
    };

    /**
     * @brief Own admitted persistent storage for one complete verifier workload.
     *
     * Grouped and serial kernels have distinct instances and state banks. Every
     * persistent fixture allocation is 256-byte aligned before warmup. Logical
     * live M and declared retained row capacity remain separate identities.
     */
    class GDNVerifierCase
    {
    public:
        GDNVerifierCase(
            int rows,
            int n_k_heads,
            int n_heads,
            int d_k,
            int d_v,
            int initial_phase)
            : rows_(rows),
              n_k_heads_(n_k_heads),
              n_heads_(n_heads),
              d_k_(d_k),
              d_v_(d_v),
              q_src_dim_(n_k_heads * d_k),
              k_src_dim_(n_k_heads * d_k),
              value_dim_(n_heads * d_v),
              qkv_stride_(q_src_dim_ + k_src_dim_ + value_dim_),
              state_floats_(llaminar2::CPUGDNStateLayout(n_heads,d_k,d_v).floats()),
              merged_qkv_(static_cast<size_t>(rows) * qkv_stride_),
              alpha_(static_cast<size_t>(rows) * n_heads),
              beta_(static_cast<size_t>(rows) * n_heads),
              a_log_(n_heads),
              dt_bias_(n_heads),
              initial_state_(state_floats_),
              grouped_live_state_(state_floats_),
              serial_state_(state_floats_),
              grouped_output_(static_cast<size_t>(rows) * value_dim_),
              serial_output_(static_cast<size_t>(rows) * value_dim_),
              grouped_snapshots_(static_cast<size_t>(rows) * state_floats_),
              serial_snapshots_(static_cast<size_t>(rows) * state_floats_)
        {
            const CPUGDNStateLayout layout(n_heads,d_k,d_v);
            retained_capacity_ = envInt("LLAMINAR_CPU_GDN_VERIFIER_CAPACITY",rows);
            if (retained_capacity_ < rows)
                throw std::invalid_argument("CPU GDN retained capacity is smaller than live M");
            const CPUSharedGDNLogicalCapacity capacity(retained_capacity_,1);
            shared_slots_.resize(capacity.slotFloats(layout));
            grouped_kernel_.bindVerifierStateCaptureWorkspace(shared_slots_.data(),capacity.flatRows(),state_floats_);
            grouped_kernel_.bindCPUSharedStateCapture(shared_slots_.data(),shared_slots_.size(),capacity.flatRows());
            fillInputs();
            // Establish real pending history outside timing using ordinary
            // M=1 arithmetic, so phases include their actual live raw records.
            CPUGatedDeltaNet phase_kernel;
            std::vector<float> phase_output(value_dim_);
            for (int p = 0; p < initial_phase; ++p)
                if (!phase_kernel.chunkForwardMergedQKV(
                        merged_qkv_.data(), qkv_stride_, alpha_.data(), beta_.data(),
                        a_log_.data(), dt_bias_.data(), phase_output.data(), initial_state_.data(),
                        1, n_k_heads_, n_heads_, d_k_, d_v_, 0, 64, true))
                    throw std::runtime_error("CPU GDN verifier phase preparation failed");
            resetGrouped();
            resetSerial();
        }

        /** @brief Restore the grouped live-state source outside measured time. */
        void resetGrouped()
        {
            std::memcpy(
                grouped_live_state_.data(),
                initial_state_.data(),
                static_cast<size_t>(state_floats_) * sizeof(float));
        }

        /** @brief Restore serial production state outside measured time. */
        void resetSerial()
        {
            std::memcpy(
                serial_state_.data(),
                initial_state_.data(),
                static_cast<size_t>(state_floats_) * sizeof(float));
        }

        /**
         * @brief Run one grouped production-ABI verifier transaction.
         *
         * Uses the same bound compact arena transaction as production serving;
         * initial immutable checkpoint, crossed checkpoints and raw log are timed.
         */
        bool runGrouped()
        {
            return grouped_kernel_.chunkForwardMergedQKV(
                merged_qkv_.data(),
                qkv_stride_,
                alpha_.data(),
                beta_.data(),
                a_log_.data(),
                dt_bias_.data(),
                grouped_output_.data(),
                grouped_live_state_.data(),
                rows_,
                n_k_heads_,
                n_heads_,
                d_k_,
                d_v_,
                /*global_v_head_offset=*/0,
                /*chunk_size=*/64,
                /*use_qk_l2norm=*/true);
        }

        /**
         * @brief Run the actual production M=1 merged-QKV path for every row.
         *
         * @param publish_snapshots Whether to copy each post-row state for the
         *        correctness oracle. Canonical timing disables these diagnostic
         *        copies because ordinary serial decode does not publish them.
         * @param prefix_rows Accepted prefix to execute; zero retains the original full-row call.
         */
        bool runSerial(bool publish_snapshots, int prefix_rows = 0)
        {
            const int live_rows = prefix_rows == 0 ? rows_ : prefix_rows;
            if (live_rows <= 0 || live_rows > rows_) return false;
            for (int row = 0; row < live_rows; ++row)
            {
                if (!serial_kernel_.chunkForwardMergedQKV(
                        merged_qkv_.data() + static_cast<size_t>(row) * qkv_stride_,
                        qkv_stride_,
                        alpha_.data() + static_cast<size_t>(row) * n_heads_,
                        beta_.data() + static_cast<size_t>(row) * n_heads_,
                        a_log_.data(),
                        dt_bias_.data(),
                        serial_output_.data() + static_cast<size_t>(row) * value_dim_,
                        serial_state_.data(),
                        /*seq_len=*/1,
                        n_k_heads_,
                        n_heads_,
                        d_k_,
                        d_v_,
                        /*global_v_head_offset=*/0,
                        /*chunk_size=*/64,
                        /*use_qk_l2norm=*/true))
                {
                    return false;
                }

                if (publish_snapshots)
                {
                    std::memcpy(
                        serial_snapshots_.data() +
                            static_cast<size_t>(row) * state_floats_,
                        serial_state_.data(),
                        static_cast<size_t>(state_floats_) * sizeof(float));
                }
            }
            return true;
        }

        /** @brief Prove grouped output, snapshots, and live-state isolation. */
        void verifyByteEquality()
        {
            byte_exact_ = false;
            resetGrouped();
            resetSerial();
            ASSERT_TRUE(runGrouped());
            ASSERT_TRUE(runSerial(/*publish_snapshots=*/true));

            const std::string prefix =
                "CPU GDN merged verifier M=" + std::to_string(rows_);
            expectByteExact(
                grouped_output_.data(),
                serial_output_.data(),
                grouped_output_.size(),
                prefix + " output");
            const auto preserved_live = grouped_live_state_;
            std::fill(grouped_live_state_.begin(), grouped_live_state_.end(), -777.0f);
            for (int row=0;row<rows_;++row)
                ASSERT_TRUE(grouped_kernel_.restoreVerifierStateCaptureRow(grouped_snapshots_.data()+static_cast<size_t>(row)*state_floats_,row,nullptr));
            grouped_live_state_ = preserved_live;
            expectByteExact(
                grouped_snapshots_.data(),
                serial_snapshots_.data(),
                grouped_snapshots_.size(),
                prefix + " state snapshots");
            expectByteExact(
                grouped_live_state_.data(),
                initial_state_.data(),
                initial_state_.size(),
                prefix + " live-state isolation");
            byte_exact_ = std::memcmp(grouped_output_.data(), serial_output_.data(), grouped_output_.size()*sizeof(float)) == 0 &&
                std::memcmp(grouped_snapshots_.data(), serial_snapshots_.data(), serial_snapshots_.size()*sizeof(float)) == 0 &&
                std::memcmp(grouped_live_state_.data(), initial_state_.data(), initial_state_.size()*sizeof(float)) == 0;
        }

        /**
         * @brief Bind a real production stage and its persistent publication workspace.
         * @details Setup only: tensor copies, allocator admission and context creation
         * finish before warmup/timing. This stage owns the compact generation and
         * returns its actual restore plan; no test-authored plan or kernel restore
         * API substitutes for MTPSpecStatePublisher in the measured scope.
         */
        void enableProductionPublication()
        {
            if (publication_stage_) throw std::logic_error("Publication owner was already bound");
            if (!byte_exact_) throw std::logic_error("Publication setup needs the serial byte oracle first");
            if (!hasCPUBackend()) initCPUBackend(-1);
            const auto tensor = [](std::vector<size_t> shape, const float *source)
            {
                auto result = std::make_unique<FP32Tensor>(shape, DeviceId::cpu());
                if (source) std::memcpy(result->mutable_data(), source, result->numel() * sizeof(float));
                return result;
            };
            publication_qkv_ = tensor({static_cast<size_t>(rows_), static_cast<size_t>(qkv_stride_)}, merged_qkv_.data());
            publication_alpha_ = tensor({static_cast<size_t>(rows_), static_cast<size_t>(n_heads_)}, alpha_.data());
            publication_beta_ = tensor({static_cast<size_t>(rows_), static_cast<size_t>(n_heads_)}, beta_.data());
            publication_a_log_ = tensor({static_cast<size_t>(n_heads_)}, a_log_.data());
            publication_dt_bias_ = tensor({static_cast<size_t>(n_heads_)}, dt_bias_.data());
            publication_output_ = tensor({static_cast<size_t>(rows_), static_cast<size_t>(value_dim_)}, nullptr);
            publication_expected_output_.assign(serial_output_.data(), serial_output_.data() + serial_output_.size());
            publication_context_ = std::make_unique<CPUDeviceContext>(DeviceId::cpu());
            GDNRecurrenceStage::Params params;
            params.device_id = DeviceId::cpu(); params.kernel = &grouped_kernel_;
            params.Q = params.K = params.V = publication_qkv_.get();
            params.alpha = publication_alpha_.get(); params.beta = publication_beta_.get();
            params.A_log = publication_a_log_.get(); params.dt_bias = publication_dt_bias_.get();
            params.output = publication_output_.get(); params.recurrence_state = grouped_live_state_.data();
            params.seq_len = params.request_seq_len = rows_;
            params.n_heads = n_heads_; params.n_k_heads = n_k_heads_; params.d_k = d_k_; params.d_v = d_v_;
            params.speculative_state_slot_rows = params.verifier_state_capture_rows = retained_capacity_;
            params.workspace_namespace = "cpu_verifier_publication_perf";
            publication_stage_ = std::make_unique<GDNRecurrenceStage>(params);
            const auto requirements = publication_stage_->getWorkspaceRequirements(retained_capacity_);
            if (requirements.buffers.size() != 1)
                throw std::logic_error("CPU GDN publication requires exactly its typed compact slot buffer");
            publication_workspace_ = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::cpu(), requirements.total_bytes_with_alignment());
            if (!publication_workspace_->allocate(requirements))
                throw std::runtime_error("CPU GDN publication workspace admission failed");
            publication_slots_name_ = requirements.buffers.front().name;
            publication_stage_->bindWorkspace(publication_workspace_.get());
            publication_stages_.push_back(publication_stage_.get());
            publication_plan_.request_id = 1;
            // The CPU host publisher's legacy plan counts retained draft rows;
            // its extra Pending target row is not another captured candidate.
            // This fixture certifies the local state publication ABI, not an
            // enclosing depth-controller or model-generation policy.
            publication_plan_.draft_count = rows_;
            publication_plan_.target_rows = rows_ + 1;
        }

        /**
         * @brief Restore the canonical input and select one accepted prefix outside timing.
         * @param accepted Number of captured rows to make live, in 1..rows().
         */
        void prepareProductionPublication(int accepted)
        {
            if (!publication_stage_ || accepted <= 0 || accepted > rows_ || rows_ == std::numeric_limits<int>::max())
                throw std::invalid_argument("Invalid CPU publication prefix");
            resetGrouped();
            if (std::memcmp(grouped_live_state_.data(), initial_state_.data(), initial_state_.size() * sizeof(float)) != 0)
                throw std::runtime_error("CPU publication input-state reset was incomplete");
            std::fill_n(publication_output_->mutable_data(), publication_output_->numel(), -12345.0f);
            publication_plan_.accepted_count = accepted;
            publication_plan_.target_cached_tokens = accepted;
        }

        /**
         * @brief Capture through the real stage and publish through the actual CPU publisher.
         * @return True only for a successful capture and exactly one restored stage.
         */
        bool runProductionPublication()
        {
            if (!publication_stage_->execute(publication_context_.get())) return false;
            const auto result = publishAcceptedMTPSpecStateFromVerifierRow(publication_plan_,
                publication_plan_.accepted_count - 1, publication_stages_, DeviceId::cpu(), nullptr, true);
            if (!result.ok) std::cerr << "[CPU GDN production publication] " << result.error << '\n';
            return result.ok && result.restored_stage_count == 1 && result.skipped_stage_count == 0;
        }

        /**
         * @brief Freeze the independently checked capture payload before timed publication.
         * @param accepted Valid full or partial prefix for this observation.
         * @details The actual publisher must preserve every checkpoint/raw-log
         * byte. No timed endpoint is discarded to produce this reference.
         */
        void primeProductionPublication(int accepted)
        {
            prepareProductionPublication(accepted);
            // Dirty the retained allocation only for the untimed first oracle.
            // Per-sample slot poisoning would change the cache/economy workload.
            std::memset(publication_workspace_->getBuffer(publication_slots_name_), 0xa5,
                publication_workspace_->getBufferSize(publication_slots_name_));
            if (!publication_stage_->execute(publication_context_.get()))
                throw std::runtime_error("CPU production capture oracle failed");
            expectByteExact(grouped_live_state_.data(), initial_state_.data(), initial_state_.size(), "stage capture live-state isolation");
            // Every candidate is independently materialized only for this
            // untimed oracle; the timed path below uses the real publisher.
            for (int row = 0; row < rows_; ++row)
                if (!grouped_kernel_.restoreVerifierStateCaptureRow(grouped_snapshots_.data() +
                        static_cast<size_t>(row) * state_floats_, row, nullptr))
                    throw std::runtime_error("CPU stage candidate materialization oracle failed");
            expectByteExact(grouped_snapshots_.data(), serial_snapshots_.data(), serial_snapshots_.size(), "stage captured candidates");
            if (std::memcmp(grouped_live_state_.data(), initial_state_.data(), initial_state_.size() * sizeof(float)) != 0 ||
                std::memcmp(grouped_snapshots_.data(), serial_snapshots_.data(), serial_snapshots_.size() * sizeof(float)) != 0)
                throw std::runtime_error("CPU stage candidate oracle changed state or candidate bytes");
            const auto *slots = static_cast<const std::byte *>(publication_workspace_->getBuffer(publication_slots_name_));
            const size_t bytes = publication_workspace_->getBufferSize(publication_slots_name_);
            publication_expected_slots_.assign(slots, slots + bytes);
            const auto result = publishAcceptedMTPSpecStateFromVerifierRow(publication_plan_, accepted - 1,
                publication_stages_, DeviceId::cpu(), nullptr, true);
            if (!result.ok || result.restored_stage_count != 1 || !checkProductionPublication(accepted))
                throw std::runtime_error("CPU production publication oracle failed");
        }

        /**
         * @brief Check the actual just-completed measured endpoint without launching or resetting.
         * @param accepted Number of rows published by the last invocation.
         * @return Exact output/state/input/immutable-slot equality at this endpoint.
         */
        bool checkProductionPublication(int accepted) const
        {
            const float *expected_state = serial_snapshots_.data() + static_cast<size_t>(accepted - 1) * state_floats_;
            const auto *slots = publication_workspace_->getBuffer(publication_slots_name_);
            const auto equal = [](const void *a, const void *b, size_t bytes) { return std::memcmp(a, b, bytes) == 0; };
            const bool output = equal(publication_output_->data(), publication_expected_output_.data(), publication_expected_output_.size() * sizeof(float));
            const bool state = equal(grouped_live_state_.data(), expected_state, static_cast<size_t>(state_floats_) * sizeof(float));
            const bool retained = publication_expected_slots_.size() == publication_workspace_->getBufferSize(publication_slots_name_) &&
                equal(slots, publication_expected_slots_.data(), publication_expected_slots_.size());
            const bool inputs = equal(publication_qkv_->data(), merged_qkv_.data(), merged_qkv_.size() * sizeof(float)) &&
                equal(publication_alpha_->data(), alpha_.data(), alpha_.size() * sizeof(float)) &&
                equal(publication_beta_->data(), beta_.data(), beta_.size() * sizeof(float)) &&
                equal(publication_a_log_->data(), a_log_.data(), a_log_.size() * sizeof(float)) &&
                equal(publication_dt_bias_->data(), dt_bias_.data(), dt_bias_.size() * sizeof(float));
            EXPECT_TRUE(output) << "Actual timed production-stage output changed";
            EXPECT_TRUE(state) << "Actual timed accepted canonical state changed";
            EXPECT_TRUE(retained) << "Actual timed publication mutated immutable candidate bytes";
            EXPECT_TRUE(inputs) << "Actual timed publication mutated source tensors";
            return output && state && retained && inputs;
        }

        /** @brief Restore and authenticate ordinary serial input outside the timed interval. */
        void prepareSerialPublication()
        {
            resetSerial();
            std::fill_n(serial_output_.data(), serial_output_.size(), -12345.0f);
            if (std::memcmp(serial_state_.data(), initial_state_.data(), initial_state_.size() * sizeof(float)) != 0)
                throw std::runtime_error("CPU serial input-state reset was incomplete");
        }

        /**
         * @brief Check the just-completed ordinary serial prefix without replay or retention.
         * @param accepted Number of ordinary rows executed in the last timed call.
         * @return Exact prefix output and terminal-state equality.
         */
        bool checkSerialPrefix(int accepted) const
        {
            const bool output = std::memcmp(serial_output_.data(), publication_expected_output_.data(),
                static_cast<size_t>(accepted) * value_dim_ * sizeof(float)) == 0;
            const bool state = std::memcmp(serial_state_.data(), serial_snapshots_.data() +
                static_cast<size_t>(accepted - 1) * state_floats_, static_cast<size_t>(state_floats_) * sizeof(float)) == 0;
            EXPECT_TRUE(output); EXPECT_TRUE(state);
            return output && state;
        }

        /**
         * @brief Prove the published endpoint supports the same next row as serial decode.
         * @param accepted Full or partial prefix currently live in the grouped bank.
         * @return Whether continuation output/state and retired payload remain byte exact.
         * @details Endpoint checks precede detachment. Detachment retires only
         * compact protocol words; continuation cannot mutate the retained payload.
         * The existing stage/workspace is rebound only outside timing afterward.
         */
        bool verifyPublicationContinuation(int accepted)
        {
            if (!checkProductionPublication(accepted)) throw std::runtime_error("Invalid pre-continuation endpoint");
            std::vector<float> expected_state(serial_snapshots_.data() + static_cast<size_t>(accepted - 1) * state_floats_,
                serial_snapshots_.data() + static_cast<size_t>(accepted) * state_floats_);
            std::vector<float> expected(value_dim_), actual(value_dim_);
            CPUGatedDeltaNet continuation_oracle;
            if (!continuation_oracle.chunkForwardMergedQKV(merged_qkv_.data(), qkv_stride_, alpha_.data(), beta_.data(),
                    a_log_.data(), dt_bias_.data(), expected.data(), expected_state.data(), 1,
                    n_k_heads_, n_heads_, d_k_, d_v_, 0, 64, true))
                throw std::runtime_error("CPU serial continuation oracle failed");
            publication_stage_->unbindWorkspace();
            auto retired = publication_expected_slots_;
            const CPUGDNStateLayout layout(n_heads_, d_k_, d_v_);
            const size_t head_floats = CPUSharedGDNSnapshots::headFloats(layout, retained_capacity_);
            const std::array<uint32_t, 3> protocol{};
            for (int head = 0; head < n_heads_; ++head)
                std::memcpy(retired.data() + static_cast<size_t>(head) * head_floats * sizeof(float), protocol.data(), sizeof(protocol));
            if (!grouped_kernel_.chunkForwardMergedQKV(merged_qkv_.data(), qkv_stride_, alpha_.data(), beta_.data(),
                    a_log_.data(), dt_bias_.data(), actual.data(), grouped_live_state_.data(), 1,
                    n_k_heads_, n_heads_, d_k_, d_v_, 0, 64, true))
                throw std::runtime_error("CPU accepted-state continuation failed");
            expectByteExact(actual.data(), expected.data(), expected.size(), "published prefix continuation output");
            expectByteExact(grouped_live_state_.data(), expected_state.data(), expected_state.size(), "published prefix continuation state");
            const bool retained = std::memcmp(publication_workspace_->getBuffer(publication_slots_name_), retired.data(), retired.size()) == 0;
            EXPECT_TRUE(retained) << "Continuation changed retired compact payload";
            publication_stage_->bindWorkspace(publication_workspace_.get());
            return retained && std::memcmp(actual.data(), expected.data(), expected.size() * sizeof(float)) == 0 &&
                std::memcmp(grouped_live_state_.data(), expected_state.data(), expected_state.size() * sizeof(float)) == 0;
        }

        /** @return Live verifier rows in this case. */
        int rows() const { return rows_; }
        /** @return Explicit admitted row capacity, independent of physical rounding. */
        int retainedCapacity() const { return retained_capacity_; }
        /** @return Whether every output, candidate and initial-state byte matched. */
        bool byteExact() const { return byte_exact_; }
        /** @return Arena payload bytes of compact candidate capture. */
        size_t snapshotBytes() const { return shared_slots_.size() * sizeof(float); }
        /** @return Exclusive mutable bank bytes, separate from immutable capture. */
        size_t workBytes() const { return 0; }

        /** @brief Admit an explicit kernel-only FullEncoded comparator before any samples.
         * @details Full stores M complete post-row CPU epoch images. Negative
         * publication is NoOp, not prefix0 restoration, so it needs no additional
         * immutable cp0. Existing run() uses its last snapshot as working state.
         * Shared keeps its existing immutable checkpoint/raw-log retention.
         */
        void enableDiagnosticRetentionComparison()
        {
            if (!byte_exact_ || diagnostic_full_slots_.size() || retained_capacity_ != rows_)
                throw std::logic_error("CPU retention diagnostic requires exact fresh physical-M owners");
            const auto state = static_cast<size_t>(state_floats_);
            if (static_cast<size_t>(rows_) > std::numeric_limits<size_t>::max() / state)
                throw std::overflow_error("CPU FullEncoded diagnostic extent overflow");
            diagnostic_full_slots_.resize(static_cast<size_t>(rows_) * state);
            diagnostic_full_live_.resize(state);
            diagnostic_full_output_.resize(grouped_output_.size());
            diagnostic_full_next_.resize(value_dim_);
            diagnostic_shared_next_.resize(value_dim_);
            diagnostic_full_kernel_.bindVerifierStateCaptureWorkspace(diagnostic_full_slots_.data(), rows_,
                                                                      state_floats_);
            // Full serialization is explicit; it never impersonates shared slots.
            prepareDiagnosticRetention(false);
            prepareDiagnosticRetention(true);
            if (!captureDiagnosticRetention(false) || !captureDiagnosticRetention(true))
                throw std::runtime_error("CPU retention diagnostic cold capture failed");
            expectByteExact(diagnostic_full_live_.data(), initial_state_.data(), state,
                            "FullEncoded capture preserves original source");
            expectByteExact(grouped_live_state_.data(), initial_state_.data(), state,
                            "Shared capture preserves original source");
            expectByteExact(diagnostic_full_slots_.data(), serial_snapshots_.data(),
                            serial_snapshots_.size(), "FullEncoded every epoch image");
            expectByteExact(diagnostic_full_output_.data(), serial_output_.data(), serial_output_.size(),
                            "FullEncoded original epoch readout");
            diagnostic_expected_full_.assign(diagnostic_full_slots_.begin(), diagnostic_full_slots_.end());
            diagnostic_expected_shared_.assign(shared_slots_.begin(), shared_slots_.end());
        }

        /** @brief Reset only canonical data and output poison outside the timed interval.
         * @param full Select FullEncoded versus Shared diagnostic kernel arm. */
        void prepareDiagnosticRetention(bool full)
        {
            auto &live = full ? diagnostic_full_live_ : grouped_live_state_;
            auto &output = full ? diagnostic_full_output_ : grouped_output_;
            auto &next = full ? diagnostic_full_next_ : diagnostic_shared_next_;
            std::memcpy(live.data(), initial_state_.data(), initial_state_.size() * sizeof(float));
            std::fill(output.begin(), output.end(), -12345.0f);
            std::fill(next.begin(), next.end(), -12345.0f);
        }

        /** @brief Execute one explicit same-arithmetic retained capture without a stage facade.
         * @param full Select FullEncoded versus Shared immutable transaction.
         * @return Existing native CPU kernel completed; initial live bytes remain untouched. */
        bool captureDiagnosticRetention(bool full)
        {
            if (!full)
                return runGrouped();
            return diagnostic_full_kernel_.chunkForwardMergedQKVWithStateSnapshots(
                merged_qkv_.data(), qkv_stride_, alpha_.data(), beta_.data(), a_log_.data(),
                dt_bias_.data(), diagnostic_full_output_.data(), diagnostic_full_live_.data(), rows_,
                n_k_heads_, n_heads_, d_k_, d_v_, 0, 64, true, diagnostic_full_slots_.data(), state_floats_,
                rows_);
        }

        /** @brief Use the existing native R1 row-vector publication semantics on either format.
         * @param full Explicit FullEncoded versus Shared arm.
         * @param row Positive prefix A names row A-1; negative means preserve current live state.
         * @return Actual native publication succeeded; no test-authored stage restore plan. */
        bool publishDiagnosticRetention(bool full, int row)
        {
            auto &kernel = full ? diagnostic_full_kernel_ : grouped_kernel_;
            auto &live = full ? diagnostic_full_live_ : grouped_live_state_;
            return kernel.restoreVerifierStateCaptureRows(live.data(), &row, 1, nullptr);
        }

        /** @brief Prepare an independent serial next-row oracle outside timing.
         * @param row Captured row0..M-1 or negative NoOp from original state. */
        void expectDiagnosticRetentionNext(int row)
        {
            const float *source = row < 0
                                      ? initial_state_.data()
                                      : serial_snapshots_.data() + static_cast<size_t>(row) * state_floats_;
            diagnostic_expected_next_state_.assign(source, source + state_floats_);
            diagnostic_expected_next_output_.resize(value_dim_);
            CPUGatedDeltaNet oracle;
            if (!oracle.chunkForwardMergedQKV(merged_qkv_.data(), qkv_stride_, alpha_.data(), beta_.data(),
                                              a_log_.data(), dt_bias_.data(),
                                              diagnostic_expected_next_output_.data(),
                                              diagnostic_expected_next_state_.data(), 1, n_k_heads_,
                                              n_heads_, d_k_, d_v_, 0, 64, true))
                throw std::runtime_error("CPU retention diagnostic next-row oracle failed");
        }

        /** @brief Complete timed kernel capture, native accepted restore and one ordinary M1.
         * @param full Explicit retained arm; neither choice changes epoch arithmetic.
         * @param row Actual accepted row or negative NoOp; prefix0 restore is not exposed.
         * @return Every native operation succeeded; no fixture allocation/rebinding/reset occurs here. */
        bool runDiagnosticRetentionComplete(bool full, int row)
        {
            if (!captureDiagnosticRetention(full) || !publishDiagnosticRetention(full, row))
                return false;
            auto &live = full ? diagnostic_full_live_ : grouped_live_state_;
            auto &next = full ? diagnostic_full_next_ : diagnostic_shared_next_;
            auto &kernel = full ? diagnostic_full_next_kernel_ : diagnostic_shared_next_kernel_;
            return kernel.chunkForwardMergedQKV(merged_qkv_.data(), qkv_stride_, alpha_.data(),
                                                beta_.data(), a_log_.data(), dt_bias_.data(), next.data(),
                                                live.data(), 1, n_k_heads_, n_heads_, d_k_, d_v_, 0, 64,
                                                true);
        }

        /** @brief Check the actual timed endpoint before another capture/reset can replace it.
         * @param full Completed explicit diagnostic arm.
         * @return All verifier output, accepted continuation and immutable retained bytes match. */
        bool checkDiagnosticRetentionComplete(bool full) const
        {
            const auto &live = full ? diagnostic_full_live_ : grouped_live_state_;
            const auto &output = full ? diagnostic_full_output_ : grouped_output_;
            const auto &next = full ? diagnostic_full_next_ : diagnostic_shared_next_;
            const auto &slots = full ? diagnostic_full_slots_ : shared_slots_;
            const auto &expected_slots = full ? diagnostic_expected_full_ : diagnostic_expected_shared_;
            const bool exact =
                std::memcmp(output.data(), serial_output_.data(), output.size() * sizeof(float)) == 0 &&
                std::memcmp(live.data(), diagnostic_expected_next_state_.data(),
                            live.size() * sizeof(float)) == 0 &&
                std::memcmp(next.data(), diagnostic_expected_next_output_.data(),
                            next.size() * sizeof(float)) == 0 &&
                std::memcmp(slots.data(), expected_slots.data(), slots.size() * sizeof(float)) == 0;
            EXPECT_TRUE(exact) << "Actual timed FullEncoded/Shared diagnostic endpoint differs";
            return exact;
        }

        /** @brief Prove repeated positive restores and negative NoOp without a new capture.
         * @return Original live isolation, every selected row and immutable payload remain exact. */
        bool verifyDiagnosticRetentionRevisions()
        {
            prepareDiagnosticRetention(false);
            prepareDiagnosticRetention(true);
            if (!captureDiagnosticRetention(false) || !captureDiagnosticRetention(true))
                return false;
            if (std::memcmp(grouped_live_state_.data(), initial_state_.data(),
                            initial_state_.size() * sizeof(float)) ||
                std::memcmp(diagnostic_full_live_.data(), initial_state_.data(),
                            initial_state_.size() * sizeof(float)))
                return false;
            std::vector<float> previous(initial_state_.begin(), initial_state_.end());
            for (int row : {rows_ - 1, -1, 0, rows_ / 2, rows_ - 1})
            {
                if (!publishDiagnosticRetention(false, row) || !publishDiagnosticRetention(true, row))
                    return false;
                if (row >= 0)
                    std::copy_n(serial_snapshots_.data() + static_cast<size_t>(row) * state_floats_,
                                state_floats_, previous.data());
                if (std::memcmp(grouped_live_state_.data(), previous.data(),
                                previous.size() * sizeof(float)) ||
                    std::memcmp(diagnostic_full_live_.data(), previous.data(),
                                previous.size() * sizeof(float)) ||
                    std::memcmp(shared_slots_.data(), diagnostic_expected_shared_.data(),
                                shared_slots_.size() * sizeof(float)) ||
                    std::memcmp(diagnostic_full_slots_.data(), diagnostic_expected_full_.data(),
                                diagnostic_full_slots_.size() * sizeof(float)))
                    return false;
            }
            return true;
        }

        /** @return Complete FullEncoded retained bytes; no redundant immutable origin image. */
        size_t diagnosticFullRetainedBytes() const { return diagnostic_full_slots_.size() * sizeof(float); }

    private:
        int retained_capacity_ = 0; ///< Explicit declared row capacity, independent of live M.
        bool byte_exact_ = false; ///< Independent oracle result for the latest check.
        /** @brief Fill deterministic finite values spanning every source region. */
        void fillInputs()
        {
            for (int row = 0; row < rows_; ++row)
            {
                float *source =
                    merged_qkv_.data() + static_cast<size_t>(row) * qkv_stride_;
                for (int index = 0; index < q_src_dim_; ++index)
                {
                    source[index] =
                        0.002f * static_cast<float>((row * 17 + index) % 29 - 14);
                }
                for (int index = 0; index < k_src_dim_; ++index)
                {
                    source[q_src_dim_ + index] =
                        0.0015f * static_cast<float>((row * 19 + index) % 31 - 15);
                }
                for (int index = 0; index < value_dim_; ++index)
                {
                    source[q_src_dim_ + k_src_dim_ + index] =
                        0.001f * static_cast<float>((row * 23 + index) % 23 - 11);
                }
            }

            for (size_t index = 0; index < alpha_.size(); ++index)
            {
                alpha_[index] =
                    -0.2f + 0.004f * static_cast<float>(index % 37);
                beta_[index] =
                    0.1f - 0.003f * static_cast<float>(index % 41);
            }
            for (int head = 0; head < n_heads_; ++head)
            {
                a_log_[static_cast<size_t>(head)] =
                    -0.35f - 0.002f * static_cast<float>(head);
                dt_bias_[static_cast<size_t>(head)] =
                    0.01f * static_cast<float>(head % 7 - 3);
            }
            for (int index = 0; index < state_floats_; ++index)
            {
                initial_state_[static_cast<size_t>(index)] =
                    0.0001f * static_cast<float>(index % 43 - 21);
            }
            const CPUGDNStateLayout canonical(n_heads_, d_k_, d_v_);
            for (int head = 0; head < n_heads_; ++head) {
                float *bank = initial_state_.data() + static_cast<size_t>(head) * canonical.headFloats();
                bank[0] = 0;
                std::fill(canonical.record(bank, 0), bank + canonical.headFloats(), 0.0f);
            }
        }

        int rows_ = 0;
        int n_k_heads_ = 0;
        int n_heads_ = 0;
        int d_k_ = 0;
        int d_v_ = 0;
        int q_src_dim_ = 0;
        int k_src_dim_ = 0;
        int value_dim_ = 0;
        int qkv_stride_ = 0;
        int state_floats_ = 0;

        AlignedFloatStorage merged_qkv_;
        AlignedFloatStorage alpha_;
        AlignedFloatStorage beta_;
        AlignedFloatStorage a_log_;
        AlignedFloatStorage dt_bias_;
        AlignedFloatStorage initial_state_;
        AlignedFloatStorage grouped_live_state_;
        AlignedFloatStorage serial_state_;
        AlignedFloatStorage grouped_output_;
        AlignedFloatStorage serial_output_;
        AlignedFloatStorage shared_slots_; ///< Preallocated production transaction arenas.
        AlignedFloatStorage grouped_snapshots_;
        AlignedFloatStorage serial_snapshots_;
        CPUGatedDeltaNet grouped_kernel_;
        CPUGatedDeltaNet serial_kernel_;
        // Diagnostic-only persistent owners: admitted once before warmup, never a serving selector.
        AlignedFloatStorage diagnostic_full_slots_, diagnostic_full_live_, diagnostic_full_output_;
        AlignedFloatStorage diagnostic_full_next_, diagnostic_shared_next_;
        CPUGatedDeltaNet diagnostic_full_kernel_, diagnostic_full_next_kernel_,
            diagnostic_shared_next_kernel_;
        std::vector<float> diagnostic_expected_full_, diagnostic_expected_shared_;
        std::vector<float> diagnostic_expected_next_state_, diagnostic_expected_next_output_;

        // All publication tensors/workspace/context outlive the stage's borrows.
        std::unique_ptr<FP32Tensor> publication_qkv_, publication_alpha_, publication_beta_;
        std::unique_ptr<FP32Tensor> publication_a_log_, publication_dt_bias_, publication_output_;
        std::unique_ptr<CPUDeviceContext> publication_context_;
        std::unique_ptr<DeviceWorkspaceManager> publication_workspace_;
        std::unique_ptr<GDNRecurrenceStage> publication_stage_;
        std::vector<IComputeStage *> publication_stages_;
        std::string publication_slots_name_;
        MTPSpecStepPlan publication_plan_;
        std::vector<float> publication_expected_output_;
        std::vector<std::byte> publication_expected_slots_;
    };

    /** @brief Measure one invocation after untimed state preparation. */
    double timeOneMs(
        const std::function<void()> &prepare,
        const std::function<bool()> &invoke)
    {
        prepare();
        const auto start = std::chrono::steady_clock::now();
        const bool ok = invoke();
        const auto end = std::chrono::steady_clock::now();
        EXPECT_TRUE(ok);
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    /**
     * @brief Alternate grouped and serial order and retain each best stable run.
     *
     * Alternation prevents one path from systematically inheriting hotter code
     * or data.  Best-of-N estimates the kernel ceiling and is intentionally not
     * an end-to-end latency percentile.
     */
    void timePairedBestMs(
        GDNVerifierCase *benchmark_case,
        int iterations,
        double *grouped_ms,
        double *serial_ms)
    {
        double best_grouped = std::numeric_limits<double>::infinity();
        double best_serial = std::numeric_limits<double>::infinity();

        const auto measure_grouped = [&]()
        {
            best_grouped = std::min(
                best_grouped,
                timeOneMs(
                    [&]() { benchmark_case->resetGrouped(); },
                    [&]() { return benchmark_case->runGrouped(); }));
        };
        const auto measure_serial = [&]()
        {
            best_serial = std::min(
                best_serial,
                timeOneMs(
                    [&]() { benchmark_case->resetSerial(); },
                    [&]() { return benchmark_case->runSerial(false); }));
        };

        for (int iteration = 0; iteration < iterations; ++iteration)
        {
            if ((iteration & 1) == 0)
            {
                measure_grouped();
                measure_serial();
            }
            else
            {
                measure_serial();
                measure_grouped();
            }
        }

        *grouped_ms = best_grouped;
        *serial_ms = best_serial;
    }

    /** @brief Run correctness, warmup, and measurement for one runtime M. */
    BenchmarkResult runCase(
        int rows,
        int n_k_heads,
        int n_heads,
        int d_k,
        int d_v,
        int initial_phase)
    {
        const int warmups = envInt("LLAMINAR_CPU_GDN_VERIFIER_WARMUP", 3);
        const int iterations = envInt("LLAMINAR_CPU_GDN_VERIFIER_ITERS", 12);
        GDNVerifierCase benchmark_case(rows, n_k_heads, n_heads, d_k, d_v, initial_phase);

        benchmark_case.verifyByteEquality();
        const bool initial_byte_exact = benchmark_case.byteExact();
        for (int warmup = 0; warmup < warmups; ++warmup)
        {
            benchmark_case.resetGrouped();
            EXPECT_TRUE(benchmark_case.runGrouped());
            benchmark_case.resetSerial();
            EXPECT_TRUE(benchmark_case.runSerial(false));
        }

        BenchmarkResult result;
        result.rows = rows;
        result.n_k_heads = n_k_heads;
        result.n_heads = n_heads;
        result.d_k = d_k;
        result.d_v = d_v;
        result.initial_phase = initial_phase;
        result.retained_capacity = benchmark_case.retainedCapacity();
        result.snapshot_bytes = benchmark_case.snapshotBytes();
        result.work_bytes = benchmark_case.workBytes();
        timePairedBestMs(
            &benchmark_case,
            iterations,
            &result.grouped_ms,
            &result.serial_ms);
        result.speedup = result.serial_ms / std::max(result.grouped_ms, 1e-9);

        benchmark_case.verifyByteEquality();
        result.byte_exact = initial_byte_exact && benchmark_case.byteExact();
        return result;
    }

    /** @brief Persist concise additive evidence when a CSV path is requested. */
    void writeCsv(const std::vector<BenchmarkResult> &results)
    {
        const char *path = std::getenv("LLAMINAR_CPU_GDN_VERIFIER_CSV");
        if (!path || !*path)
            return;

        FILE *file = std::fopen(path, "w");
        ASSERT_NE(file, nullptr) << "failed to open " << path;
        std::fprintf(
            file,
            "backend,phase,rows,n_k_heads,n_heads,d_k,d_v,grouped_ms,"
            "serial_ms,speedup,byte_exact,initial_phase,snapshot_bytes,work_bytes,retained_capacity_rows,fixture_alignment_bytes\n");
        for (const BenchmarkResult &result : results)
        {
            std::fprintf(
                file,
                "cpu,gdn_verifier_rows,%d,%d,%d,%d,%d,%.9f,%.9f,%.9f,%d,%d,%zu,%zu,%d,256\n",
                result.rows,
                result.n_k_heads,
                result.n_heads,
                result.d_k,
                result.d_v,
                result.grouped_ms,
                result.serial_ms,
                result.speedup, result.byte_exact ? 1 : 0, result.initial_phase,
                result.snapshot_bytes, result.work_bytes, result.retained_capacity);
        }
        std::fclose(file);
    }
} // namespace

TEST(Perf__CPUGatedDeltaNetVerifierRows, RuntimeM_GroupedVsSerial_ProductionMergedQKV)
{
    const int n_heads = envInt("LLAMINAR_CPU_GDN_VERIFIER_HEADS", 16);
    const int n_k_heads =
        envInt("LLAMINAR_CPU_GDN_VERIFIER_K_HEADS", n_heads);
    const int d_k = envInt("LLAMINAR_CPU_GDN_VERIFIER_DK", 128);
    const int d_v = envInt("LLAMINAR_CPU_GDN_VERIFIER_DV", 128);
    const double min_speedup =
        envDouble("LLAMINAR_CPU_GDN_VERIFIER_MIN_SPEEDUP", 1.0);

    std::vector<BenchmarkResult> results;
    for (const int initial_phase : envPhases())
    for (const int rows : envRows())
    {
        BenchmarkResult result =
            runCase(rows, n_k_heads, n_heads, d_k, d_v, initial_phase);
        results.push_back(result);

        std::cerr << "[CPU GDN verifier rows] M=" << result.rows
                  << " initial_phase=" << result.initial_phase
                  << " retained_capacity=" << result.retained_capacity
                  << " fixture_alignment=256"
                  << " k_heads=" << result.n_k_heads
                  << " v_heads=" << result.n_heads
                  << " d_k=" << result.d_k
                  << " d_v=" << result.d_v
                  << " grouped_ms=" << result.grouped_ms
                  << " serial_ms=" << result.serial_ms
                  << " speedup=" << result.speedup
                  << " byte_exact=" << result.byte_exact << '\n';

        if (rows > 1)
        {
            EXPECT_GT(result.speedup, min_speedup)
                << "M=" << rows
                << " grouped_ms=" << result.grouped_ms
                << " serial_ms=" << result.serial_ms
                << " min_speedup=" << min_speedup;
        }
    }
    writeCsv(results);
}


/**
 * @brief Time real GDN stage capture through production accepted-state publication.
 *
 * Full acceptance keeps the original economy threshold. Partial acceptance is
 * explicitly diagnostic: all verifier rows execute, while the serial comparator
 * computes only its accepted rows. Both endpoints and a following ordinary row
 * must be byte exact. Best-of-N remains a local ceiling, not model throughput.
 * Every measured pair is emitted after both actual endpoint checks. When the
 * existing summary CSV path is supplied, a fresh .samples.csv sidecar retains
 * those raw pairs without changing the summary schema or capture-only scope.
 */
TEST(Perf__CPUGatedDeltaNetVerifierRows, RuntimeM_CaptureAndAcceptedPublication_ProductionStage)
{
    const int n_heads = envInt("LLAMINAR_CPU_GDN_VERIFIER_HEADS", 16);
    const int n_k_heads = envInt("LLAMINAR_CPU_GDN_VERIFIER_K_HEADS", n_heads);
    const int d_k = envInt("LLAMINAR_CPU_GDN_VERIFIER_DK", 128);
    const int d_v = envInt("LLAMINAR_CPU_GDN_VERIFIER_DV", 128);
    const int warmups = envInt("LLAMINAR_CPU_GDN_VERIFIER_WARMUP", 3);
    const int iterations = envInt("LLAMINAR_CPU_GDN_VERIFIER_ITERS", 12);
    const double min_speedup = envDouble("LLAMINAR_CPU_GDN_VERIFIER_MIN_SPEEDUP", 1.0);
    const char *path = std::getenv("LLAMINAR_CPU_GDN_PUBLICATION_CSV");
    std::unique_ptr<FILE, decltype(&std::fclose)> csv(path && *path ? std::fopen(path, "wx") : nullptr, &std::fclose);
    if (path && *path) ASSERT_NE(csv, nullptr) << "Publication CSV must be a fresh file: " << path;
    if (csv) std::fprintf(csv.get(), "scope,rows,accepted_rows,initial_phase,n_k_heads,n_heads,d_k,d_v,capture_publication_ms,serial_prefix_ms,speedup,byte_exact,economy_required,retained_capacity_rows\n");
    // A separate additive sidecar preserves the existing summary schema. Every
    // measured pair is also printed when no CSV was requested; warmups are not samples.
    const std::string sample_path = path && *path ? std::string(path) + ".samples.csv" : std::string{};
    std::unique_ptr<FILE, decltype(&std::fclose)> samples(
        sample_path.empty() ? nullptr : std::fopen(sample_path.c_str(), "wx"), &std::fclose);
    if (!sample_path.empty()) ASSERT_NE(samples, nullptr) << "Raw publication sample CSV must be fresh: " << sample_path;
    if (samples) std::fprintf(samples.get(), "rows,initial_phase,accepted_rows,sample,order,n_k_heads,n_heads,d_k,d_v,grouped_ms,serial_ms,grouped_endpoint_exact,serial_endpoint_exact,endpoint_exact\n");
    for (const int initial_phase : envPhases())
    for (const int rows : envRows())
    {
        GDNVerifierCase benchmark_case(rows, n_k_heads, n_heads, d_k, d_v, initial_phase);
        benchmark_case.verifyByteEquality();
        ASSERT_TRUE(benchmark_case.byteExact());
        benchmark_case.enableProductionPublication();
        std::vector<int> prefixes{rows};
        if (rows > 1) prefixes.push_back(std::max(1, rows / 2));
        for (const int accepted : prefixes)
        {
            SCOPED_TRACE("M=" + std::to_string(rows) + " accepted=" + std::to_string(accepted) + " phase=" + std::to_string(initial_phase));
            benchmark_case.primeProductionPublication(accepted);
            ASSERT_TRUE(benchmark_case.verifyPublicationContinuation(accepted));
            for (int warmup = 0; warmup < warmups; ++warmup)
            {
                benchmark_case.prepareProductionPublication(accepted);
                ASSERT_TRUE(benchmark_case.runProductionPublication());
                ASSERT_TRUE(benchmark_case.checkProductionPublication(accepted));
                benchmark_case.prepareSerialPublication();
                ASSERT_TRUE(benchmark_case.runSerial(false, accepted));
                ASSERT_TRUE(benchmark_case.checkSerialPrefix(accepted));
            }
            double best_grouped = std::numeric_limits<double>::infinity();
            double best_serial = std::numeric_limits<double>::infinity();
            bool endpoint_bytes = true;
            for (int iteration = 0; iteration < iterations; ++iteration)
            {
                double grouped_ms = std::numeric_limits<double>::quiet_NaN();
                double serial_ms = std::numeric_limits<double>::quiet_NaN();
                bool grouped_endpoint_exact = false, serial_endpoint_exact = false;
                const auto grouped = [&]()
                {
                    bool completed = false;
                    grouped_ms = timeOneMs(
                        [&]() { benchmark_case.prepareProductionPublication(accepted); },
                        [&]() { completed = benchmark_case.runProductionPublication(); return completed; });
                    best_grouped = std::min(best_grouped, grouped_ms);
                    // Inspect the actual measured result before another reset,
                    // capture, fresh oracle or continuation can replace it.
                    const bool checked = benchmark_case.checkProductionPublication(accepted);
                    grouped_endpoint_exact = completed && checked;
                    endpoint_bytes &= grouped_endpoint_exact;
                };
                const auto serial = [&]()
                {
                    bool completed = false;
                    serial_ms = timeOneMs(
                        [&]() { benchmark_case.prepareSerialPublication(); },
                        [&]() { completed = benchmark_case.runSerial(false, accepted); return completed; });
                    best_serial = std::min(best_serial, serial_ms);
                    const bool checked = benchmark_case.checkSerialPrefix(accepted);
                    serial_endpoint_exact = completed && checked;
                    endpoint_bytes &= serial_endpoint_exact;
                };
                if (iteration & 1) { serial(); grouped(); }
                else { grouped(); serial(); }
                // Both native intervals and their actual endpoint checks are
                // complete before any formatting or durable sample publication.
                const char *order = iteration & 1 ? "serial_grouped" : "grouped_serial";
                const bool pair_exact = grouped_endpoint_exact && serial_endpoint_exact;
                std::cerr << "[CPU GDN publication sample] M=" << rows
                          << " phase=" << initial_phase << " accepted=" << accepted
                          << " sample=" << iteration << " order=" << order
                          << " grouped_ms=" << grouped_ms << " serial_ms=" << serial_ms
                          << " endpoint_exact=" << pair_exact << '\n';
                if (samples)
                {
                    std::fprintf(samples.get(), "%d,%d,%d,%d,%s,%d,%d,%d,%d,%.9f,%.9f,%d,%d,%d\n",
                        rows, initial_phase, accepted, iteration, order, n_k_heads, n_heads, d_k, d_v,
                        grouped_ms, serial_ms, grouped_endpoint_exact ? 1 : 0,
                        serial_endpoint_exact ? 1 : 0, pair_exact ? 1 : 0);
                    std::fflush(samples.get());
                }
            }
            // Independent serial work does not touch the grouped bank, so this
            // is the last actual timed publication endpoint, never a fresh replay.
            endpoint_bytes &= benchmark_case.verifyPublicationContinuation(accepted);
            const double speedup = best_serial / std::max(best_grouped, 1e-9);
            const bool economy_required = rows > 1 && accepted == rows;
            std::cerr << "[CPU GDN publication boundary] scope=production_stage_capture_and_cpu_publisher"
                      << " M=" << rows << " accepted=" << accepted << " phase=" << initial_phase
                      << " capture_publication_ms=" << best_grouped << " serial_prefix_ms=" << best_serial
                      << " speedup=" << speedup << " byte_exact=" << endpoint_bytes
                      << " economy_required=" << economy_required << '\n';
            if (csv) {
                std::fprintf(csv.get(), "production_stage_capture_and_cpu_publisher,%d,%d,%d,%d,%d,%d,%d,%.9f,%.9f,%.9f,%d,%d,%d\n",
                    rows, accepted, initial_phase, n_k_heads, n_heads, d_k, d_v,
                    best_grouped, best_serial, speedup, endpoint_bytes ? 1 : 0,
                    economy_required ? 1 : 0, benchmark_case.retainedCapacity());
                std::fflush(csv.get());
            }
            EXPECT_TRUE(endpoint_bytes);
            if (economy_required) EXPECT_GT(speedup, min_speedup);
        }
    }
}

/**
 * @brief Same-epoch FullEncoded versus Shared complete kernel boundary, not serving certification.
 *
 * Both arms include immutable capture, existing native R1 restore and the same
 * ordinary next row. The actual stage and MTPSpecStatePublisher are deliberately
 * absent because their current typed CPU contract admits Shared only. Negative
 * selection is NoOp; positive accepted A uses row A-1. Prefix0 restore is absent.
 * Original capture-only and real-stage tests/schemas above remain unchanged.
 */
TEST(Perf__CPUGatedDeltaNetVerifierRows, DISABLED_FullEncodedVsSharedRetainedKernelBoundary)
{
    const char *worker_request = std::getenv("OMP_NUM_THREADS");
    ASSERT_NE(worker_request, nullptr) << "Diagnostic requires an explicit worker request";
    char *worker_end = nullptr;
    errno = 0;
    const long parsed_workers = std::strtol(worker_request, &worker_end, 10);
    ASSERT_TRUE(worker_end != worker_request && *worker_end == '\0' && errno != ERANGE &&
                parsed_workers > 0 && parsed_workers <= std::numeric_limits<int>::max())
        << "Diagnostic requires one complete positive OMP_NUM_THREADS integer";
    const int requested_workers = static_cast<int>(parsed_workers);
    const int warmups = envInt("LLAMINAR_CPU_GDN_VERIFIER_WARMUP", 3);
    const int samples = envInt("LLAMINAR_CPU_GDN_VERIFIER_ITERS", 12);
    for (int phase = 0; phase < CPUGDNStateLayout::kPeriod; ++phase)
        for (int rows : {1, 2, 4, 16})
        {
            GDNVerifierCase cell(rows, 16, 32, 128, 128, phase);
            cell.verifyByteEquality();
            ASSERT_FALSE(::testing::Test::HasFailure());
            ASSERT_TRUE(cell.byteExact());
            cell.enableDiagnosticRetentionComparison();
            ASSERT_FALSE(::testing::Test::HasFailure());
            ASSERT_TRUE(cell.verifyDiagnosticRetentionRevisions());
            for (int row = -1; row < rows; ++row)
            {
                // -1 is an actual native NoOp; it is never relabeled cp0 restore.
                cell.expectDiagnosticRetentionNext(row);
                for (int sample = -warmups; sample < samples; ++sample)
                {
                    double full_ms = 0.0, shared_ms = 0.0;
                    bool full_exact = false, shared_exact = false;
                    const auto arm = [&](bool full)
                    {
                        bool completed = false;
                        const double milliseconds =
                            timeOneMs([&] { cell.prepareDiagnosticRetention(full); },
                                      [&]
                                      {
                                          completed =
                                              cell.runDiagnosticRetentionComplete(full, row);
                                          return completed;
                                      });
                        const bool exact = completed && cell.checkDiagnosticRetentionComplete(full);
                        if (full)
                        {
                            full_ms = milliseconds;
                            full_exact = exact;
                        }
                        else
                        {
                            shared_ms = milliseconds;
                            shared_exact = exact;
                        }
                    };
                    if ((sample + warmups) & 1)
                    {
                        arm(false);
                        arm(true);
                    }
                    else
                    {
                        arm(true);
                        arm(false);
                    }
                    ASSERT_TRUE(full_exact && shared_exact)
                        << "Actual measured retained/next endpoint failed before reset";
                    if (sample >= 0)
                        std::cout
                            << std::setprecision(12)
                            << "CPU_GDN_RETAINED_DIAGNOSTIC_PAIR "
                               "scope=kernel_capture_native_restore_next"
                            << " production_stage=0 production_publisher=0 serving_qualification=0"
                            << " arithmetic=fixed_epoch8_backward_fma_v1 "
                               "persistent_state=cpu_fixed_epoch"
                            << " heads=32 key_heads=16 DK=128 DV=128 requests=1 M=" << rows
                            << " live_rows=" << rows << " initial_phase=" << phase
                            << " requested_workers=" << requested_workers << " row=" << row
                            << " accepted_rows=" << (row < 0 ? 0 : row + 1)
                            << " selection=" << (row < 0 ? "negative_no_op" : "positive_post_row")
                            << " prefix0_restore_supported=0 full_extra_cp0_copy=0"
                            << " full_slots_bytes=" << cell.diagnosticFullRetainedBytes()
                            << " shared_slots_bytes=" << cell.snapshotBytes()
                            << " sample=" << sample << " samples=" << samples
                            << " warmups=" << warmups << " order="
                            << (((sample + warmups) & 1) ? "shared_full" : "full_shared")
                            << " full_ms=" << full_ms << " shared_ms=" << shared_ms
                            << " full_endpoint_exact=1 shared_endpoint_exact=1 "
                               "capture_input_preserved=1"
                            << " immutable_slots_exact=1 next_row_in_timed_boundary=1 "
                               "serial_oracle=1 passed=1\n";
                }
            }
        }
}

/** @test Source-compatible explicit full serialization and Shared retain every positive/NoOp
 * revision. */
TEST(Perf__CPUGatedDeltaNetVerifierRows, DISABLED_FullEncodedVsSharedRevisionBytes)
{
    for (int phase = 0; phase < CPUGDNStateLayout::kPeriod; ++phase)
        for (int rows : {1, 2, 4, 16})
        {
            GDNVerifierCase cell(rows, 16, 32, 128, 128, phase);
            cell.verifyByteEquality();
            ASSERT_FALSE(::testing::Test::HasFailure());
            ASSERT_TRUE(cell.byteExact());
            cell.enableDiagnosticRetentionComparison();
            ASSERT_FALSE(::testing::Test::HasFailure());
            ASSERT_TRUE(cell.verifyDiagnosticRetentionRevisions());
        }
}
