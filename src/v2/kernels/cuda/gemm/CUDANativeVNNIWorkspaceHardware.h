/**
 * @file CUDANativeVNNIWorkspaceHardware.h
 * @brief Explicit physical policy inputs for device-free CUDA projection workspace planning.
 *
 * The runtime binds these facts from its admitted physical inventory; metadata
 * planners bind the same ordinal/SM count from their typed rank observation.
 * No global synthetic inventory, runtime context, allocation or capacity ledger
 * is created. The installed canonical-M1 arithmetic policy consumes this count.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <optional>

namespace llaminar2::cuda
{
    /** @brief Explicit participant binding; only SM-dependent policies require an SM observation. */
    class NativeVNNIWorkspaceHardware final
    {
    public:
        /**
         * @brief Bind the observed participant and compute-unit count before workspace planning.
         * @param ordinal Exact nonnegative CUDA participant ordinal.
         * @param sm_count Positive SM count from the admitted hardware observation.
         * @return Immutable policy inputs carrying the original observation.
         * @throws std::invalid_argument For a negative ordinal or nonpositive SM count.
         */
        static NativeVNNIWorkspaceHardware checked(int ordinal, int sm_count)
        {
            if (ordinal < 0 || sm_count <= 0)
                throw std::invalid_argument("CUDA workspace planning requires exact ordinal and positive SM count");
            return NativeVNNIWorkspaceHardware(ordinal, sm_count);
        }
        /**
         * @brief Bind geometry with no physical SM observation, never a guessed device.
         * @param ordinal Declared participant ordinal.
         * @return Explicitly unobserved hardware; SM-dependent policy must reject it.
         * @throws std::invalid_argument For a negative participant ordinal.
         */
        static NativeVNNIWorkspaceHardware withoutSMObservation(int ordinal)
        {
            if (ordinal < 0) throw std::invalid_argument("CUDA workspace planning requires a participant ordinal");
            return NativeVNNIWorkspaceHardware(ordinal, std::nullopt);
        }
        /** @brief Derived-plan identity includes the exact ordinal and optional SM observation. */
        bool operator==(const NativeVNNIWorkspaceHardware &) const = default;
        /** @return The original optional observation for complete derived-plan cache identity. */
        std::optional<int> observedSMCount() const noexcept { return sm_count_; }
        /** @return Exact observed participant ordinal, without native work. */
        int deviceOrdinal() const noexcept { return ordinal_; }
        /**
         * @brief Require the observation needed by an SM-dependent workspace policy.
         * @return Exact observed SM count without querying a runtime or creating a context.
         * @throws std::invalid_argument If planning supplied no SM observation.
         */
        int smCount() const
        {
            if (!sm_count_) throw std::invalid_argument("CUDA canonical workspace policy requires an explicit SM observation");
            return *sm_count_;
        }
    private:
        /** @brief Retain only checked immutable policy inputs. */
        NativeVNNIWorkspaceHardware(int ordinal, std::optional<int> sm_count) : ordinal_(ordinal), sm_count_(sm_count) {}
        int ordinal_;
        std::optional<int> sm_count_;
    };

    /**
     * @brief Run the existing installed prefill envelope policy with explicit hardware facts.
     * @param codebook Execution decoder. @param arithmetic_codebook Original arithmetic policy.
     * @param max_rows Retained row envelope. @param n Output columns. @param k Reduction width.
     * @param hardware Checked participant facts. @param bytes Unpadded canonical scratch output.
     * @param partitions Canonical M1 partition output. @param rows Envelope-owning row output.
     * @return Whether the unchanged installed policy admits the complete envelope.
     */
    bool nativeVNNIPrefillWorkspaceEnvelopeForHardware(std::uint8_t codebook,
        std::uint8_t arithmetic_codebook, int max_rows, int n, int k,
        const NativeVNNIWorkspaceHardware &hardware, std::size_t *bytes, int *partitions, int *rows);
}
