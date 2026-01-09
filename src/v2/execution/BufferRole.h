/**
 * @file BufferRole.h
 * @brief Buffer role classification and descriptor types for GraphExecutor
 * @author David Sanftenberg
 * @date December 2025
 *
 * This header defines the type system for buffer management in the GraphExecutor
 * framework. Every buffer used by a ComputeStage has a formal role that determines:
 * - Read/write permissions
 * - Persistence after execution
 * - Eligibility for buffer reuse/aliasing
 *
 * @see GraphBufferManager for buffer allocation
 * @see IComputeStage::getBufferRequirements() for stage integration
 */

#pragma once

#include "../tensors/TensorLayout.h"
#include "../backends/DeviceId.h"
#include <cstddef>
#include <string>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class CPUTensorBase;
    using TensorBase = CPUTensorBase; // Backward compatibility alias

    // =========================================================================
    // Buffer Role Enum
    // =========================================================================

    /**
     * @brief Formal classification of buffer roles in compute stages
     *
     * Each buffer in a stage's Params struct should have a clear role that
     * determines its semantics during execution.
     *
     * ## Role Semantics
     *
     * | Role    | Read | Write | Persists | Aliasable | Typical Use |
     * |---------|------|-------|----------|-----------|-------------|
     * | INPUT   | Yes  | No    | Yes      | No        | Activations from previous stage |
     * | OUTPUT  | No   | Yes   | Yes      | No        | Results for downstream stages |
     * | INOUT   | Yes  | Yes   | Yes      | No        | Residual accumulators |
     * | SCRATCH | Maybe| Yes   | No       | Yes       | Temporary workspace |
     * | WEIGHT  | Yes  | No    | Yes      | No        | Model parameters |
     *
     * ## Buffer Aliasing
     *
     * Only SCRATCH buffers are eligible for aliasing (sharing physical memory).
     * GraphBufferManager performs liveness analysis to identify non-overlapping
     * SCRATCH buffers that can share the same allocation.
     *
     * ## Example Usage
     *
     * @code
     * struct MyStageParams {
     *     const TensorBase* input = nullptr;    // [INPUT] Read-only activation
     *     const TensorBase* gamma = nullptr;    // [WEIGHT] Norm weights
     *     TensorBase* output = nullptr;         // [OUTPUT] Result tensor
     *     TensorBase* residual = nullptr;       // [INOUT] Accumulator
     *     TensorBase* workspace = nullptr;      // [SCRATCH] Temporary buffer
     * };
     * @endcode
     */
    enum class BufferRole
    {
        /**
         * @brief Read-only input consumed by the stage
         *
         * - Stage MUST NOT modify this buffer
         * - Data comes from previous stage's OUTPUT or external source
         * - Persists after stage execution (owned externally)
         * - NOT eligible for aliasing
         *
         * Use `const TensorBase*` in Params struct.
         */
        INPUT,

        /**
         * @brief Write-only output produced by the stage
         *
         * - Stage MAY assume buffer contents are undefined on entry
         * - Stage MUST write meaningful data before returning
         * - Persists after stage execution (consumed by downstream)
         * - NOT eligible for aliasing
         *
         * Use non-const `TensorBase*` in Params struct.
         */
        OUTPUT,

        /**
         * @brief Read-modify-write buffer
         *
         * - Stage reads existing value, modifies it, writes back
         * - Typical use: residual accumulation (residual += attention_output)
         * - Persists after stage execution (accumulator pattern)
         * - NOT eligible for aliasing
         *
         * Use non-const `TensorBase*` in Params struct.
         */
        INOUT,

        /**
         * @brief Temporary workspace buffer
         *
         * - Contents are UNDEFINED after execute() returns
         * - Stage has exclusive write access during execution
         * - Does NOT persist (can be overwritten by next stage)
         * - ELIGIBLE for aliasing with non-overlapping SCRATCH buffers
         *
         * Use non-const `TensorBase*` in Params struct.
         * GraphBufferManager may alias multiple SCRATCH buffers to save memory.
         */
        SCRATCH,

        /**
         * @brief Read-only model weights
         *
         * - Never modified during inference
         * - Persists for entire model lifetime
         * - NOT eligible for aliasing
         * - May be quantized (IQ4_NL, Q8_0, etc.)
         *
         * Use `const TensorBase*` in Params struct.
         */
        WEIGHT,
    };

    /**
     * @brief Convert BufferRole to string for logging/debugging
     */
    inline const char *bufferRoleName(BufferRole role)
    {
        switch (role)
        {
        case BufferRole::INPUT:
            return "INPUT";
        case BufferRole::OUTPUT:
            return "OUTPUT";
        case BufferRole::INOUT:
            return "INOUT";
        case BufferRole::SCRATCH:
            return "SCRATCH";
        case BufferRole::WEIGHT:
            return "WEIGHT";
        default:
            return "UNKNOWN";
        }
    }

    // =========================================================================
    // Tensor Type (for buffer allocation)
    // =========================================================================

    /**
     * @brief Tensor data type for buffer allocation
     *
     * Matches TensorBase::native_type() values but as a simple enum
     * for use in buffer descriptors before tensors exist.
     */
    enum class BufferTensorType
    {
        FP32,    ///< 32-bit floating point
        FP16,    ///< 16-bit floating point
        BF16,    ///< Brain float 16
        Q8_1,    ///< 8-bit quantized with scales
        Q8_0,    ///< 8-bit quantized
        Q4_0,    ///< 4-bit quantized
        IQ4_NL,  ///< 4-bit importance quantized
        INT32,   ///< 32-bit integer (position IDs, etc.)
        UNKNOWN, ///< Unspecified (infer from context)
    };

    /**
     * @brief Get element size in bytes for a tensor type
     *
     * For block-quantized types, returns the block size.
     */
    inline size_t bufferTensorTypeSize(BufferTensorType type)
    {
        switch (type)
        {
        case BufferTensorType::FP32:
            return 4;
        case BufferTensorType::FP16:
            return 2;
        case BufferTensorType::BF16:
            return 2;
        case BufferTensorType::INT32:
            return 4;
        case BufferTensorType::Q8_1:
            return 36; // 32 bytes data + 4 bytes scale (per 32 elements)
        case BufferTensorType::Q8_0:
            return 34; // 32 bytes data + 2 bytes scale
        case BufferTensorType::Q4_0:
            return 18; // 16 bytes data + 2 bytes scale (per 32 elements)
        case BufferTensorType::IQ4_NL:
            return 18; // Similar to Q4_0
        default:
            return 4; // Default to FP32 size
        }
    }

    /**
     * @brief Convert BufferTensorType to string
     */
    inline const char *bufferTensorTypeName(BufferTensorType type)
    {
        switch (type)
        {
        case BufferTensorType::FP32:
            return "FP32";
        case BufferTensorType::FP16:
            return "FP16";
        case BufferTensorType::BF16:
            return "BF16";
        case BufferTensorType::Q8_1:
            return "Q8_1";
        case BufferTensorType::Q8_0:
            return "Q8_0";
        case BufferTensorType::Q4_0:
            return "Q4_0";
        case BufferTensorType::IQ4_NL:
            return "IQ4_NL";
        case BufferTensorType::INT32:
            return "INT32";
        default:
            return "UNKNOWN";
        }
    }

    // =========================================================================
    // Buffer Descriptor
    // =========================================================================

    /**
     * @brief Describes a single buffer's requirements
     *
     * Used by stages to declare what buffers they need, and by
     * GraphBufferManager to allocate and track buffers.
     *
     * ## Example
     *
     * @code
     * BufferDescriptor desc{
     *     .name = "workspace_scores",
     *     .role = BufferRole::SCRATCH,
     *     .shape = {n_heads, seq_len, kv_len},
     *     .tensor_type = BufferTensorType::FP32,
     *     .required = true,
     *     .alignment = 64,
     * };
     * @endcode
     */
    struct BufferDescriptor
    {
        /// Human-readable buffer name (unique within stage)
        std::string name;

        /// Buffer role determining semantics
        BufferRole role = BufferRole::SCRATCH;

        /// Required tensor shape (may be empty for scalar-like buffers)
        std::vector<size_t> shape;

        /// Tensor data type
        BufferTensorType tensor_type = BufferTensorType::FP32;

        /// Whether this buffer is required (false = optional)
        bool required = true;

        /// Memory alignment requirement in bytes
        size_t alignment = 64;

        /// Target device
        DeviceId device = DeviceId::cpu();

        // =====================================================================
        // Producer/Consumer Contract Fields (Phase 2: Buffer Validation)
        // =====================================================================

        /// Stage name that produces this buffer (empty = external/pre-allocated)
        /// Used by GraphValidator to verify buffer flow
        std::string producer_stage;

        /// Whether this buffer must be non-zero after producer executes
        /// When true, GraphValidator will check tensor is populated
        bool validate_populated = false;

        // =====================================================================
        // Layout Contract Fields (Phase 3: Tensor Layout Contracts)
        // =====================================================================

        /// Expected TensorLayout for this buffer (UNKNOWN = no validation)
        /// Used by GraphExecutor for automatic layout validation
        TensorLayout expected_layout = TensorLayout::UNKNOWN;

        // =====================================================================
        // Convenience Methods
        // =====================================================================

        /**
         * @brief Calculate total elements in buffer
         */
        size_t numel() const
        {
            if (shape.empty())
                return 0;
            size_t n = 1;
            for (size_t dim : shape)
                n *= dim;
            return n;
        }

        /**
         * @brief Calculate total bytes needed
         *
         * For quantized types, accounts for block structure.
         */
        size_t sizeBytes() const
        {
            size_t elements = numel();
            if (elements == 0)
                return 0;

            switch (tensor_type)
            {
            case BufferTensorType::Q8_1:
            case BufferTensorType::Q8_0:
            case BufferTensorType::Q4_0:
            case BufferTensorType::IQ4_NL:
            {
                // Block-quantized: round up to block boundary
                constexpr size_t BLOCK_SIZE = 32;
                size_t blocks = (elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
                return blocks * bufferTensorTypeSize(tensor_type);
            }
            default:
                return elements * bufferTensorTypeSize(tensor_type);
            }
        }

        /**
         * @brief Check if buffer is aliasable (only SCRATCH)
         */
        bool isAliasable() const
        {
            return role == BufferRole::SCRATCH;
        }

        /**
         * @brief Check if buffer is read-only
         */
        bool isReadOnly() const
        {
            return role == BufferRole::INPUT || role == BufferRole::WEIGHT;
        }

        // =====================================================================
        // Builder Pattern
        // =====================================================================

        static BufferDescriptor input(const std::string &name,
                                      std::vector<size_t> shape,
                                      BufferTensorType type = BufferTensorType::FP32)
        {
            return BufferDescriptor{name, BufferRole::INPUT, std::move(shape), type, true, 64, DeviceId::cpu()};
        }

        static BufferDescriptor output(const std::string &name,
                                       std::vector<size_t> shape,
                                       BufferTensorType type = BufferTensorType::FP32)
        {
            return BufferDescriptor{name, BufferRole::OUTPUT, std::move(shape), type, true, 64, DeviceId::cpu()};
        }

        static BufferDescriptor inout(const std::string &name,
                                      std::vector<size_t> shape,
                                      BufferTensorType type = BufferTensorType::FP32)
        {
            return BufferDescriptor{name, BufferRole::INOUT, std::move(shape), type, true, 64, DeviceId::cpu()};
        }

        static BufferDescriptor scratch(const std::string &name,
                                        std::vector<size_t> shape,
                                        BufferTensorType type = BufferTensorType::FP32)
        {
            return BufferDescriptor{name, BufferRole::SCRATCH, std::move(shape), type, true, 64, DeviceId::cpu()};
        }

        static BufferDescriptor weight(const std::string &name,
                                       std::vector<size_t> shape,
                                       BufferTensorType type = BufferTensorType::FP32)
        {
            return BufferDescriptor{name, BufferRole::WEIGHT, std::move(shape), type, true, 64, DeviceId::cpu()};
        }

        // =====================================================================
        // Fluent Builder Extensions (Producer/Consumer Contracts)
        // =====================================================================

        /**
         * @brief Declare which stage produces this buffer
         *
         * Used by GraphValidator to verify buffer flow - ensures that
         * every buffer with a producer_stage actually gets populated.
         *
         * @param stage Stage name (e.g., "kv_append", "rope")
         * @return Reference for chaining
         */
        BufferDescriptor &withProducer(const std::string &stage)
        {
            producer_stage = stage;
            return *this;
        }

        /**
         * @brief Mark buffer for population validation
         *
         * When set, debug builds will verify the buffer is non-zero
         * after the producer stage executes.
         *
         * @return Reference for chaining
         */
        BufferDescriptor &validatePopulated()
        {
            validate_populated = true;
            return *this;
        }

        /**
         * @brief Declare expected tensor layout for this buffer
         *
         * Used by GraphExecutor for automatic layout validation in debug builds.
         * When set, the executor verifies the tensor's declared layout matches
         * this expectation and that the tensor's shape is consistent with the
         * layout given the model's dimension parameters.
         *
         * @param layout Expected TensorLayout (UNKNOWN = no validation)
         * @return Reference for chaining
         *
         * @code
         * reqs.addInput("Q", {seq_len, qkv_dim}, BufferTensorType::FP32)
         *     .withLayout(TensorLayout::Q_SEQ_HEAD_DIM);
         * @endcode
         */
        BufferDescriptor &withLayout(TensorLayout layout)
        {
            expected_layout = layout;
            return *this;
        }

        /**
         * @brief Check if buffer has a declared producer
         */
        bool hasProducer() const
        {
            return !producer_stage.empty();
        }
    };

    // =========================================================================
    // Stage Buffer Requirements
    // =========================================================================

    /**
     * @brief Complete buffer requirements for a compute stage
     *
     * Returned by IComputeStage::getBufferRequirements() to declare
     * all buffers needed for execution.
     *
     * ## Usage in Stage Implementation
     *
     * @code
     * StageBufferRequirements MyStage::getBufferRequirements() const {
     *     StageBufferRequirements reqs;
     *     reqs.addInput("input", {seq_len_, d_model_}, BufferTensorType::FP32);
     *     reqs.addOutput("output", {seq_len_, d_model_}, BufferTensorType::FP32);
     *     reqs.addScratch("workspace", {seq_len_ * 4}, BufferTensorType::FP32);
     *     reqs.addWeight("gamma", {d_model_}, BufferTensorType::FP32);
     *     return reqs;
     * }
     * @endcode
     */
    struct StageBufferRequirements
    {
        /// All buffer descriptors for this stage
        std::vector<BufferDescriptor> buffers;

        // =====================================================================
        // Builder Methods
        // These return BufferDescriptor& to enable fluent chaining with
        // .withLayout(), .withProducer(), .validatePopulated()
        // =====================================================================

        /**
         * @brief Add an INPUT buffer requirement
         * @param name Buffer name (unique within stage)
         * @param shape Required tensor shape
         * @param type Tensor data type (default: FP32)
         * @param layout Expected tensor layout (default: UNKNOWN = no validation)
         * @return Reference to this for fluent chaining
         */
        StageBufferRequirements &addInput(const std::string &name,
                                          std::vector<size_t> shape,
                                          BufferTensorType type = BufferTensorType::FP32,
                                          TensorLayout layout = TensorLayout::UNKNOWN)
        {
            auto desc = BufferDescriptor::input(name, std::move(shape), type);
            desc.expected_layout = layout;
            buffers.push_back(std::move(desc));
            return *this;
        }

        /**
         * @brief Add an OUTPUT buffer requirement
         * @param name Buffer name (unique within stage)
         * @param shape Required tensor shape
         * @param type Tensor data type (default: FP32)
         * @param layout Expected tensor layout (default: UNKNOWN = no validation)
         * @return Reference to this for fluent chaining
         */
        StageBufferRequirements &addOutput(const std::string &name,
                                           std::vector<size_t> shape,
                                           BufferTensorType type = BufferTensorType::FP32,
                                           TensorLayout layout = TensorLayout::UNKNOWN)
        {
            auto desc = BufferDescriptor::output(name, std::move(shape), type);
            desc.expected_layout = layout;
            buffers.push_back(std::move(desc));
            return *this;
        }

        /**
         * @brief Add an INOUT buffer requirement
         * @param name Buffer name (unique within stage)
         * @param shape Required tensor shape
         * @param type Tensor data type (default: FP32)
         * @param layout Expected tensor layout (default: UNKNOWN = no validation)
         * @return Reference to this for fluent chaining
         */
        StageBufferRequirements &addInout(const std::string &name,
                                          std::vector<size_t> shape,
                                          BufferTensorType type = BufferTensorType::FP32,
                                          TensorLayout layout = TensorLayout::UNKNOWN)
        {
            auto desc = BufferDescriptor::inout(name, std::move(shape), type);
            desc.expected_layout = layout;
            buffers.push_back(std::move(desc));
            return *this;
        }

        /**
         * @brief Add a SCRATCH buffer requirement
         * @param name Buffer name (unique within stage)
         * @param shape Required tensor shape
         * @param type Tensor data type (default: FP32)
         * @param layout Expected tensor layout (default: UNKNOWN = no validation)
         * @return Reference to this for fluent chaining
         */
        StageBufferRequirements &addScratch(const std::string &name,
                                            std::vector<size_t> shape,
                                            BufferTensorType type = BufferTensorType::FP32,
                                            TensorLayout layout = TensorLayout::UNKNOWN)
        {
            auto desc = BufferDescriptor::scratch(name, std::move(shape), type);
            desc.expected_layout = layout;
            buffers.push_back(std::move(desc));
            return *this;
        }

        /**
         * @brief Add a WEIGHT buffer requirement
         * @param name Buffer name (unique within stage)
         * @param shape Required tensor shape
         * @param type Tensor data type (default: FP32)
         * @param layout Expected tensor layout (default: UNKNOWN = no validation)
         * @return Reference to this for fluent chaining
         */
        StageBufferRequirements &addWeight(const std::string &name,
                                           std::vector<size_t> shape,
                                           BufferTensorType type = BufferTensorType::FP32,
                                           TensorLayout layout = TensorLayout::UNKNOWN)
        {
            auto desc = BufferDescriptor::weight(name, std::move(shape), type);
            desc.expected_layout = layout;
            buffers.push_back(std::move(desc));
            return *this;
        }

        /**
         * @brief Add a custom buffer descriptor
         */
        StageBufferRequirements &add(BufferDescriptor desc)
        {
            buffers.push_back(std::move(desc));
            return *this;
        }

        // =====================================================================
        // Query Methods
        // =====================================================================

        /**
         * @brief Check if requirements are empty
         */
        bool empty() const { return buffers.empty(); }

        /**
         * @brief Get number of buffer requirements
         */
        size_t size() const { return buffers.size(); }

        /**
         * @brief Get buffers by role
         */
        std::vector<const BufferDescriptor *> getByRole(BufferRole role) const
        {
            std::vector<const BufferDescriptor *> result;
            for (const auto &buf : buffers)
            {
                if (buf.role == role)
                    result.push_back(&buf);
            }
            return result;
        }

        /**
         * @brief Get buffer by name (nullptr if not found)
         */
        const BufferDescriptor *getByName(const std::string &name) const
        {
            for (const auto &buf : buffers)
            {
                if (buf.name == name)
                    return &buf;
            }
            return nullptr;
        }

        /**
         * @brief Total bytes for INPUT buffers
         */
        size_t totalInputBytes() const
        {
            size_t total = 0;
            for (const auto &buf : buffers)
            {
                if (buf.role == BufferRole::INPUT)
                    total += buf.sizeBytes();
            }
            return total;
        }

        /**
         * @brief Total bytes for OUTPUT buffers
         */
        size_t totalOutputBytes() const
        {
            size_t total = 0;
            for (const auto &buf : buffers)
            {
                if (buf.role == BufferRole::OUTPUT)
                    total += buf.sizeBytes();
            }
            return total;
        }

        /**
         * @brief Total bytes for SCRATCH buffers
         */
        size_t totalScratchBytes() const
        {
            size_t total = 0;
            for (const auto &buf : buffers)
            {
                if (buf.role == BufferRole::SCRATCH)
                    total += buf.sizeBytes();
            }
            return total;
        }

        /**
         * @brief Total bytes for all buffers
         */
        size_t totalBytes() const
        {
            size_t total = 0;
            for (const auto &buf : buffers)
            {
                total += buf.sizeBytes();
            }
            return total;
        }
    };

    // =========================================================================
    // Tensor Type Conversion Helpers
    // =========================================================================

    // Forward declaration (see Tensors.h for full definition)
    enum class TensorType;

    /**
     * @brief Convert TensorBase::native_type() to BufferTensorType
     *
     * This helper enables stages to derive buffer requirements from
     * existing tensor parameters.
     *
     * @param type TensorType from TensorBase::native_type()
     * @return Corresponding BufferTensorType
     */
    inline BufferTensorType toBufferTensorType(TensorType type)
    {
        // TensorType values from Tensors.h - cast to underlying int
        switch (static_cast<int>(type))
        {
        case 0: // FP32
            return BufferTensorType::FP32;
        case 1: // BF16
            return BufferTensorType::BF16;
        case 2: // FP16
            return BufferTensorType::FP16;
        case 4: // INT32
            return BufferTensorType::INT32;
        case 5: // IQ4_NL
            return BufferTensorType::IQ4_NL;
        case 7: // Q8_0
            return BufferTensorType::Q8_0;
        case 8: // Q8_1
            return BufferTensorType::Q8_1;
        case 9: // Q4_0
            return BufferTensorType::Q4_0;
        default:
            return BufferTensorType::UNKNOWN;
        }
    }

} // namespace llaminar2
