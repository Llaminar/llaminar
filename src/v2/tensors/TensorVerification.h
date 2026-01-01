/**
 * @file TensorVerification.h
 * @brief Unified tensor verification system for debug builds
 * @author David Sanftenberg
 * @date January 2026
 *
 * Provides comprehensive tensor verification at stage boundaries:
 * - Input validation BEFORE stage execution (catch corruption early)
 * - Output validation AFTER stage execution (detect stage bugs)
 * - Automatic buffer dump on failure for debugging
 * - Throws VerificationFailure with full context
 *
 * Design Philosophy:
 * - Uses LLAMINAR_ASSERTIONS_ACTIVE for build-type awareness
 * - Zero overhead in Release builds (all code compiled out)
 * - Throws exception on failure (fail-fast, unlike warning-only approaches)
 * - Captures full context: layer, stage, phase, tensor, dump path
 *
 * Integration with existing infrastructure:
 * - Absorbs TensorValidation.h functionality (tensorAppearsZero, tensorHasNaNOrInf)
 * - Uses StageDumper patterns for buffer dumping
 * - Respects debugEnv().validation configuration
 *
 * @see TensorValidation.h (legacy, functionality absorbed here)
 * @see StageDumper.h (reuses dump patterns)
 * @see Assertions.h (follows same LLAMINAR_ASSERTIONS_ACTIVE pattern)
 */

#pragma once

#include "../utils/Assertions.h"
#include "../utils/Logger.h"
#include "../utils/DebugEnv.h"
#include "../execution/compute_stages/IComputeStage.h"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace llaminar2::verification
{

    // =========================================================================
    // Result Structures
    // =========================================================================

    /**
     * @brief Result of tensor verification
     *
     * Contains pass/fail status along with diagnostics about what was found.
     * Used by verifyRawBuffer() and verifyTensor().
     */
    struct VerificationResult
    {
        bool passed = true;
        std::string tensor_name;
        std::string error_reason;

        // Diagnostics (populated even on success for analysis)
        size_t nan_count = 0;
        size_t inf_count = 0;
        size_t zero_count = 0;
        size_t total_sampled = 0;

        /// @brief Create a successful result
        static VerificationResult ok() { return {true, "", ""}; }

        /// @brief Create a failure result
        static VerificationResult fail(const std::string &tensor, const std::string &reason)
        {
            VerificationResult r;
            r.passed = false;
            r.tensor_name = tensor;
            r.error_reason = reason;
            return r;
        }

        /// @brief Create result with diagnostics
        static VerificationResult withDiagnostics(
            bool passed,
            const std::string &tensor,
            const std::string &reason,
            size_t nan_count,
            size_t inf_count,
            size_t zero_count,
            size_t total_sampled)
        {
            VerificationResult r;
            r.passed = passed;
            r.tensor_name = tensor;
            r.error_reason = reason;
            r.nan_count = nan_count;
            r.inf_count = inf_count;
            r.zero_count = zero_count;
            r.total_sampled = total_sampled;
            return r;
        }
    };

    // =========================================================================
    // Verification Failure Exception
    // =========================================================================

    /**
     * @brief Exception thrown when tensor verification fails
     *
     * Contains all context needed to diagnose the failure:
     * - Stage name and layer index
     * - Phase (ENTRY or EXIT)
     * - Tensor name and failure reason
     * - Path to dumped buffers
     */
    class VerificationFailure : public std::runtime_error
    {
    public:
        VerificationFailure(
            const std::string &stage_name,
            int layer_idx,
            const char *phase,
            const std::string &tensor_name,
            const std::string &reason,
            const std::string &dump_path)
            : std::runtime_error(formatMessage(stage_name, layer_idx, phase, tensor_name, reason, dump_path)),
              stage_name_(stage_name),
              layer_idx_(layer_idx),
              phase_(phase),
              tensor_name_(tensor_name),
              reason_(reason),
              dump_path_(dump_path)
        {
        }

        const std::string &stageName() const { return stage_name_; }
        int layerIdx() const { return layer_idx_; }
        const char *phase() const { return phase_; }
        const std::string &tensorName() const { return tensor_name_; }
        const std::string &reason() const { return reason_; }
        const std::string &dumpPath() const { return dump_path_; }

    private:
        static std::string formatMessage(
            const std::string &stage_name,
            int layer_idx,
            const char *phase,
            const std::string &tensor_name,
            const std::string &reason,
            const std::string &dump_path)
        {
            std::ostringstream oss;
            oss << "\n"
                << "╔══════════════════════════════════════════════════════════════════╗\n"
                << "║               TENSOR VERIFICATION FAILED                          ║\n"
                << "╠══════════════════════════════════════════════════════════════════╣\n"
                << "║ Layer:  " << layer_idx << "\n"
                << "║ Stage:  " << stage_name << "\n"
                << "║ Phase:  " << phase << "\n"
                << "║ Tensor: " << tensor_name << "\n"
                << "║ Reason: " << reason << "\n"
                << "║\n"
                << "║ Dump:   " << (dump_path.empty() ? "(disabled)" : dump_path) << "\n"
                << "╚══════════════════════════════════════════════════════════════════╝\n";
            return oss.str();
        }

        std::string stage_name_;
        int layer_idx_;
        const char *phase_;
        std::string tensor_name_;
        std::string reason_;
        std::string dump_path_;
    };

    // =========================================================================
    // Verification Configuration
    // =========================================================================

    /**
     * @brief Configuration for verification operations
     *
     * Controls what checks are performed and how failures are handled.
     * Typically populated from debugEnv().validation.
     */
    struct VerificationConfig
    {
        int sample_rows = 8;        ///< Check first N rows for efficiency
        bool check_null = true;     ///< Fail on null pointer
        bool check_nan = true;      ///< Fail on NaN values
        bool check_inf = true;      ///< Fail on Inf values
        bool check_all_zero = false; ///< Fail on all-zero tensor (usually warnings only)
        bool dump_on_failure = true; ///< Dump all buffers to disk when verification fails
    };

    // =========================================================================
    // Core Verification Functions
    // =========================================================================

#if LLAMINAR_ASSERTIONS_ACTIVE

    /**
     * @brief Verify raw FP32 buffer data
     *
     * Samples the first N rows of the buffer to check for:
     * - Null pointer
     * - NaN values
     * - Inf values
     * - All-zero data (optional)
     *
     * @param data Pointer to buffer data
     * @param rows Number of rows
     * @param cols Number of columns
     * @param name Tensor name for error messages
     * @param dtype Data type string ("FP32", "Q8_1", etc.)
     * @param config Verification configuration
     * @return VerificationResult with pass/fail and diagnostics
     */
    inline VerificationResult verifyRawBuffer(
        const void *data,
        size_t rows,
        size_t cols,
        const char *name,
        const char *dtype,
        const VerificationConfig &config = {})
    {
        // Null check
        if (config.check_null && data == nullptr)
        {
            return VerificationResult::fail(name, "Null pointer");
        }

        // Skip non-FP32 for now (can't easily check NaN/Inf)
        if (dtype == nullptr || std::string(dtype) != "FP32")
        {
            return VerificationResult::ok();
        }

        const float *fp32_data = static_cast<const float *>(data);
        size_t numel = rows * cols;

        if (numel == 0)
        {
            return VerificationResult::ok();
        }

        // Sample elements for efficiency
        size_t sample_rows = static_cast<size_t>(config.sample_rows);
        size_t check_rows = std::min(rows, sample_rows);
        size_t elements_to_check = check_rows * cols;

        size_t nan_count = 0;
        size_t inf_count = 0;
        size_t zero_count = 0;

        for (size_t i = 0; i < elements_to_check; ++i)
        {
            float val = fp32_data[i];
            if (std::isnan(val))
            {
                ++nan_count;
            }
            else if (std::isinf(val))
            {
                ++inf_count;
            }
            else if (val == 0.0f)
            {
                ++zero_count;
            }
        }

        // Check for NaN
        if (config.check_nan && nan_count > 0)
        {
            std::ostringstream oss;
            oss << "Contains " << nan_count << " NaN values in first " << check_rows << " rows";
            return VerificationResult::withDiagnostics(
                false, name, oss.str(), nan_count, inf_count, zero_count, elements_to_check);
        }

        // Check for Inf
        if (config.check_inf && inf_count > 0)
        {
            std::ostringstream oss;
            oss << "Contains " << inf_count << " Inf values in first " << check_rows << " rows";
            return VerificationResult::withDiagnostics(
                false, name, oss.str(), nan_count, inf_count, zero_count, elements_to_check);
        }

        // Check for all-zero (optional)
        if (config.check_all_zero && zero_count == elements_to_check)
        {
            std::ostringstream oss;
            oss << "All " << elements_to_check << " sampled elements are zero (likely uninitialized)";
            return VerificationResult::withDiagnostics(
                false, name, oss.str(), nan_count, inf_count, zero_count, elements_to_check);
        }

        return VerificationResult::withDiagnostics(
            true, name, "", nan_count, inf_count, zero_count, elements_to_check);
    }

    /**
     * @brief Generate timestamp string for dump directory names
     */
    inline std::string generateTimestamp()
    {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm tm_now;
        localtime_r(&time_t_now, &tm_now);

        std::ostringstream oss;
        oss << std::put_time(&tm_now, "%Y%m%d_%H%M%S");

        // Add milliseconds for uniqueness
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) %
                  1000;
        oss << "_" << std::setfill('0') << std::setw(3) << ms.count();

        return oss.str();
    }

    /**
     * @brief Create dump directory for verification failure
     *
     * Creates directory: /tmp/llaminar_verification_dump/<timestamp>_layer<N>_<stage>_<phase>/
     *
     * @param stage_name Name of the stage
     * @param layer_idx Layer index
     * @param phase "ENTRY" or "EXIT"
     * @return Path to created directory, or empty string on failure
     */
    inline std::string createDumpDirectory(
        const std::string &stage_name,
        int layer_idx,
        const char *phase)
    {
        std::ostringstream dir_name;
        dir_name << "/tmp/llaminar_verification_dump/"
                 << generateTimestamp()
                 << "_layer" << layer_idx
                 << "_" << stage_name
                 << "_" << phase;

        std::string path = dir_name.str();

        try
        {
            std::filesystem::create_directories(path);
            std::filesystem::create_directories(path + "/inputs");
            std::filesystem::create_directories(path + "/outputs");
            std::filesystem::create_directories(path + "/weights");
            return path;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[TensorVerification] Failed to create dump directory: " << e.what());
            return "";
        }
    }

    /**
     * @brief Write tensor metadata file
     */
    inline void writeTensorMetadata(
        const std::string &path,
        const char *name,
        size_t rows,
        size_t cols,
        const char *dtype,
        const VerificationResult *result = nullptr)
    {
        std::ofstream f(path);
        if (!f)
        {
            LOG_WARN("[TensorVerification] Failed to write metadata: " << path);
            return;
        }

        f << "name: " << name << "\n";
        f << "rows: " << rows << "\n";
        f << "cols: " << cols << "\n";
        f << "elements: " << (rows * cols) << "\n";
        f << "dtype: " << dtype << "\n";

        if (result)
        {
            f << "\n# Verification diagnostics\n";
            f << "passed: " << (result->passed ? "true" : "false") << "\n";
            f << "nan_count: " << result->nan_count << "\n";
            f << "inf_count: " << result->inf_count << "\n";
            f << "zero_count: " << result->zero_count << "\n";
            f << "total_sampled: " << result->total_sampled << "\n";
            if (!result->error_reason.empty())
            {
                f << "error: " << result->error_reason << "\n";
            }
        }
    }

    /**
     * @brief Dump FP32 buffer to binary file
     */
    inline void dumpFP32Buffer(
        const std::string &path,
        const float *data,
        size_t count)
    {
        std::ofstream f(path, std::ios::binary);
        if (!f)
        {
            LOG_WARN("[TensorVerification] Failed to write buffer: " << path);
            return;
        }
        f.write(reinterpret_cast<const char *>(data), count * sizeof(float));
    }

    /**
     * @brief Dump raw buffer to binary file
     */
    inline void dumpRawBuffer(
        const std::string &path,
        const void *data,
        size_t bytes)
    {
        std::ofstream f(path, std::ios::binary);
        if (!f)
        {
            LOG_WARN("[TensorVerification] Failed to write buffer: " << path);
            return;
        }
        f.write(reinterpret_cast<const char *>(data), bytes);
    }

    /**
     * @brief Write manifest.json file summarizing the dump
     */
    inline void writeManifest(
        const std::string &dump_path,
        const std::string &stage_name,
        int layer_idx,
        const char *phase,
        const std::string &failed_tensor,
        const std::string &failure_reason,
        const StageDumpInfo &dump_info)
    {
        std::string manifest_path = dump_path + "/manifest.json";
        std::ofstream f(manifest_path);
        if (!f)
        {
            LOG_WARN("[TensorVerification] Failed to write manifest");
            return;
        }

        f << "{\n";
        f << "    \"stage\": \"" << stage_name << "\",\n";
        f << "    \"layer_idx\": " << layer_idx << ",\n";
        f << "    \"phase\": \"" << phase << "\",\n";
        f << "    \"timestamp\": \"" << generateTimestamp() << "\",\n";
        f << "    \"failure\": {\n";
        f << "        \"tensor\": \"" << failed_tensor << "\",\n";
        f << "        \"reason\": \"" << failure_reason << "\"\n";
        f << "    },\n";

        // Inputs
        f << "    \"inputs\": [\n";
        for (size_t i = 0; i < dump_info.inputs.size(); ++i)
        {
            const auto &input = dump_info.inputs[i];
            f << "        {\"name\": \"" << (input.name ? input.name : "unnamed")
              << "\", \"shape\": [" << input.rows << ", " << input.cols
              << "], \"dtype\": \"" << (input.dtype ? input.dtype : "unknown") << "\"}";
            if (i < dump_info.inputs.size() - 1)
                f << ",";
            f << "\n";
        }
        f << "    ],\n";

        // Outputs
        f << "    \"outputs\": [\n";
        for (size_t i = 0; i < dump_info.outputs.size(); ++i)
        {
            const auto &output = dump_info.outputs[i];
            f << "        {\"name\": \"" << (output.name ? output.name : "unnamed")
              << "\", \"shape\": [" << output.rows << ", " << output.cols
              << "], \"dtype\": \"" << (output.dtype ? output.dtype : "unknown") << "\"}";
            if (i < dump_info.outputs.size() - 1)
                f << ",";
            f << "\n";
        }
        f << "    ]\n";

        f << "}\n";
    }

    /**
     * @brief Dump all stage buffers to disk for debugging
     *
     * Creates directory: /tmp/llaminar_verification_dump/<timestamp>_layer<N>_<stage>_<phase>/
     *
     * Dumps:
     * - All input buffers
     * - All output buffers
     * - Manifest file with metadata
     *
     * @param stage_name Name of the failing stage
     * @param layer_idx Layer index (-1 if unknown)
     * @param phase "ENTRY" or "EXIT"
     * @param dump_info Stage's dump info with all buffers
     * @param failed_tensor Name of tensor that failed verification
     * @param failure_reason Why verification failed
     * @return Path to dump directory, or empty string if disabled/failed
     */
    inline std::string dumpStageBuffers(
        const std::string &stage_name,
        int layer_idx,
        const char *phase,
        const StageDumpInfo &dump_info,
        const std::string &failed_tensor = "",
        const std::string &failure_reason = "")
    {
        std::string dump_path = createDumpDirectory(stage_name, layer_idx, phase);
        if (dump_path.empty())
        {
            return "";
        }

        LOG_INFO("[TensorVerification] Dumping stage buffers to: " << dump_path);

        // Dump inputs
        for (const auto &input : dump_info.inputs)
        {
            if (!input.data)
                continue;

            std::string name = input.name ? input.name : "unnamed";
            std::string base_path = dump_path + "/inputs/" + name;

            if (std::string(input.dtype) == "FP32")
            {
                dumpFP32Buffer(base_path + ".bin",
                               static_cast<const float *>(input.data),
                               input.rows * input.cols);
            }
            else
            {
                dumpRawBuffer(base_path + ".bin",
                              input.data,
                              input.rows * input.cols * input.element_size);
            }

            writeTensorMetadata(base_path + "_metadata.txt",
                                input.name, input.rows, input.cols, input.dtype);
        }

        // Dump outputs
        for (const auto &output : dump_info.outputs)
        {
            if (!output.data)
                continue;

            std::string name = output.name ? output.name : "unnamed";
            std::string base_path = dump_path + "/outputs/" + name;

            if (std::string(output.dtype) == "FP32")
            {
                dumpFP32Buffer(base_path + ".bin",
                               static_cast<const float *>(output.data),
                               output.rows * output.cols);
            }
            else
            {
                dumpRawBuffer(base_path + ".bin",
                              output.data,
                              output.rows * output.cols * output.element_size);
            }

            writeTensorMetadata(base_path + "_metadata.txt",
                                output.name, output.rows, output.cols, output.dtype);
        }

        // Write manifest
        writeManifest(dump_path, stage_name, layer_idx, phase,
                      failed_tensor, failure_reason, dump_info);

        return dump_path;
    }

#else // !LLAMINAR_ASSERTIONS_ACTIVE

    // Release builds: all verification functions are no-ops

    inline VerificationResult verifyRawBuffer(
        const void *,
        size_t,
        size_t,
        const char *,
        const char *,
        const VerificationConfig & = {})
    {
        return VerificationResult::ok();
    }

    inline std::string dumpStageBuffers(
        const std::string &,
        int,
        const char *,
        const StageDumpInfo &,
        const std::string & = "",
        const std::string & = "")
    {
        return "";
    }

#endif // LLAMINAR_ASSERTIONS_ACTIVE

} // namespace llaminar2::verification
