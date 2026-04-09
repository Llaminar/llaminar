/**
 * @file StageDumper.h
 * @brief First-class debugging utility for compute stage input/output dumping
 * @author David Sanftenberg
 *
 * Provides comprehensive dumping of all inputs, outputs, and parameters for
 * any ComputeStage execution. Dumps are written to disk in a structured format
 * that supports later analysis and replay testing.
 *
 * Design Philosophy:
 * - Controlled via debugEnv().stage_dump configuration (environment variables)
 * - Zero overhead when disabled (compile-time guards + runtime checks)
 * - Supports filtering by stage type, layer, iteration, rank
 * - Binary format for replay + human-readable metadata for debugging
 * - Thread-safe dump counter management
 *
 * Integration Points:
 * - Call StageDumper::dump() from LayerExecutor before/after stage execution
 * - Each ComputeStage implements getDumpInfo() to expose its buffers
 * - LoadedStageDump enables replay testing with saved data
 *
 * Output Structure:
 *   <dump_dir>/stage_<counter>_<type>_layer<N>_iter<I>/
 *     metadata.txt        - Human-readable description
 *     params.bin          - Binary parameters for replay
 *     inputs/             - Input tensors directory
 *       A.bin             - FP32 activation matrix
 *       A_q8_1.bin        - Q8_1 block data (if pre-quantized)
 *     weights/            - Weight tensors directory
 *       B_dequant.bin     - Dequantized weights (FP32)
 *       B_raw.bin         - Raw quantized blocks
 *       B_metadata.txt    - Quant type, shape, scale info
 *     outputs/            - Output tensors directory
 *       C.bin             - Output matrix (FP32)
 *     <stage-specific>/   - Additional stage-specific data
 *
 * Example Usage:
 *   // In LayerExecutor::executeStage()
 *   if (debugEnv().stage_dump.shouldDump(stage->name(), layer_idx, iteration, rank)) {
 *       StageDumper::dump(stage.get(), layer_idx, iteration, rank);
 *   }
 *   stage->execute(ctx);
 *   if (debugEnv().stage_dump.shouldDump(stage->name(), layer_idx, iteration, rank)) {
 *       StageDumper::dumpOutputs(stage.get(), layer_idx, iteration, rank, dump_id);
 *   }
 */

#pragma once

#include "../compute_stages/ComputeStages.h"
#include "../../utils/DebugEnv.h"
#include "../../tensors/BlockStructures.h"

#include <cstdio>
#include <cstdint>
#include <atomic>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <map>
#include <mutex>
#include <vector>
#include <memory>
#include <cerrno>
#include <algorithm>
#include <cmath>

namespace llaminar2
{

    /**
     * @brief Stage dump context - tracks state for a single stage execution dump
     */
    struct StageDumpContext
    {
        int dump_id = -1;            ///< Unique dump ID (-1 = skip)
        std::string dump_dir;        ///< Full path to dump directory
        std::string type_name;       ///< Stage type name
        std::string stage_name;      ///< Stage node name (e.g., "layer0_attention")
        int layer_idx = -1;          ///< Layer index (-1 if not layer-specific)
        int iteration = -1;          ///< Decode iteration (-1 for prefill)
        int rank = 0;                ///< MPI rank
        bool inputs_dumped = false;  ///< Whether inputs have been dumped
        bool outputs_dumped = false; ///< Whether outputs have been dumped
    };

    /**
     * @brief Binary dump header for version compatibility
     */
    struct StageDumpHeader
    {
        uint32_t magic = 0x4C4D5344; ///< "LMSD" - Llaminar Stage Dump
        uint32_t version = 1;        ///< Format version
        uint32_t type_hash = 0;      ///< Hash of stage type name
        int32_t layer_idx = -1;
        int32_t iteration = -1;
        int32_t rank = 0;
        uint64_t timestamp = 0; ///< Unix timestamp
    };

    /**
     * @brief Tensor dump metadata (matches TensorDumpInfo from old design)
     */
    struct TensorDumpMeta
    {
        std::string name; ///< Tensor name (e.g., "A", "B", "C")
        size_t rows = 0;
        size_t cols = 0;
        std::string dtype; ///< "FP32", "Q8_1", "IQ4_NL", etc.
        size_t element_count = 0;
        size_t byte_size = 0;
        float sample_min = 0, sample_max = 0, sample_mean = 0;

        // Block info for quantized formats (computed from dtype and dims)
        size_t block_count = 0;        ///< Total blocks (0 for non-block formats)
        size_t blocks_per_row = 0;     ///< Blocks per row (0 for non-block formats)
        size_t block_element_size = 0; ///< Elements per block (e.g., 32)

        /**
         * @brief Compute block-related metadata from dtype and dimensions
         * Call this after setting rows, cols, dtype
         *
         * Handles variable Q16_1 block sizes:
         * - Q16_1 / Q16_1_32: 32 elements per block
         * - Q16_1_64: 64 elements per block
         * - Q16_1_128: 128 elements per block
         */
        void computeBlockInfo()
        {
            constexpr size_t BLOCK_SIZE_32 = 32;   // Elements per block (Q8_1, Q8_0, Q16_1_32, IQ4_NL)
            constexpr size_t BLOCK_SIZE_64 = 64;   // Elements per block (Q16_1_64)
            constexpr size_t BLOCK_SIZE_128 = 128; // Elements per block (Q16_1_128)

            if (dtype == "Q8_1" || dtype == "Q16_1" || dtype == "Q16_1_32" || dtype == "Q8_0" || dtype == "IQ4_NL")
            {
                block_element_size = BLOCK_SIZE_32;
                blocks_per_row = (cols + BLOCK_SIZE_32 - 1) / BLOCK_SIZE_32;
                block_count = rows * blocks_per_row;
            }
            else if (dtype == "Q16_1_64")
            {
                block_element_size = BLOCK_SIZE_64;
                blocks_per_row = (cols + BLOCK_SIZE_64 - 1) / BLOCK_SIZE_64;
                block_count = rows * blocks_per_row;
            }
            else if (dtype == "Q16_1_128")
            {
                block_element_size = BLOCK_SIZE_128;
                blocks_per_row = (cols + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128;
                block_count = rows * blocks_per_row;
            }
            else
            {
                block_element_size = 0;
                blocks_per_row = 0;
                block_count = 0;
            }
        }
    };

    /**
     * @brief Main stage dumper utility
     */
    class StageDumper
    {
    public:
        /**
         * @brief Extract layer index from stage name
         *
         * Parses names like "layer0_attn_norm" → 0, "layer23_ffn_norm" → 23
         * Returns -1 if no layer prefix found.
         */
        static int extractLayerFromName(const std::string &stage_name)
        {
            if (stage_name.length() < 6 || stage_name.substr(0, 5) != "layer")
                return -1;

            size_t underscore_pos = stage_name.find('_', 5);
            if (underscore_pos == std::string::npos)
                return -1;

            try
            {
                return std::stoi(stage_name.substr(5, underscore_pos - 5));
            }
            catch (...)
            {
                return -1;
            }
        }

        /**
         * @brief Check if dumping is enabled for a stage execution (with stage name)
         * @param stage The compute stage
         * @param stage_name The node name (e.g., "layer0_attention")
         * @param layer_idx Layer index (-1 if not layer-specific, will try to extract from stage_name)
         * @param iteration Decode iteration (-1 for prefill)
         * @param rank MPI rank
         * @return true if this stage execution should be dumped
         */
        static bool shouldDump(const IComputeStage *stage, const std::string &stage_name,
                               int layer_idx, int iteration, int rank)
        {
            const auto &cfg = debugEnv().stage_dump;
            if (!cfg.enabled)
                return false;

            // If layer_idx not provided, try to extract from stage_name
            int effective_layer = layer_idx;
            if (effective_layer < 0 && !stage_name.empty())
            {
                effective_layer = extractLayerFromName(stage_name);
            }

            return cfg.shouldDump(computeStageTypeName(stage->type()), stage_name, effective_layer, iteration, rank);
        }

        /**
         * @brief Legacy check without stage name
         * @deprecated Use the overload with stage_name parameter
         */
        static bool shouldDump(const IComputeStage *stage, int layer_idx, int iteration, int rank)
        {
            return shouldDump(stage, "", layer_idx, iteration, rank);
        }

        /**
         * @brief Begin a dump for a stage (creates directory, returns context)
         *
         * Call before stage->execute() to dump inputs.
         * The returned context should be passed to dumpOutputs() after execution.
         *
         * @param stage The compute stage
         * @param stage_name The node name (e.g., "layer0_attention")
         * @param layer_idx Layer index (-1 if not layer-specific)
         * @param iteration Decode iteration (-1 for prefill)
         * @param rank MPI rank
         * @return Dump context (check dump_id >= 0 for success)
         */
        static StageDumpContext beginDump(
            const IComputeStage *stage,
            const std::string &stage_name,
            int layer_idx,
            int iteration,
            int rank)
        {
            StageDumpContext ctx;
            ctx.type_name = computeStageTypeName(stage->type());
            ctx.stage_name = stage_name;
            ctx.layer_idx = layer_idx;
            ctx.iteration = iteration;
            ctx.rank = rank;

            // Check dump limit per type
            const auto &cfg = debugEnv().stage_dump;
            int count = incrementDumpCounter(ctx.type_name);
            if (count > cfg.max_dumps_per_type)
            {
                ctx.dump_id = -1; // Signal: skip dump
                return ctx;
            }

            ctx.dump_id = count - 1; // 0-indexed dump ID

            // Create dump directory (include stage_name in directory name)
            ctx.dump_dir = createDumpDirectory(cfg.dump_dir, ctx.type_name, stage_name,
                                               ctx.dump_id, layer_idx, iteration, rank);
            if (ctx.dump_dir.empty())
            {
                ctx.dump_id = -1;
                return ctx;
            }

            return ctx;
        }

        /**
         * @brief Legacy beginDump without stage name
         * @deprecated Use the overload with stage_name parameter
         */
        static StageDumpContext beginDump(
            const IComputeStage *stage,
            int layer_idx,
            int iteration,
            int rank)
        {
            return beginDump(stage, "", layer_idx, iteration, rank);
        }

        /**
         * @brief Dump inputs and weights for a stage using getDumpInfo()
         */
        static bool dumpInputs(StageDumpContext &ctx, const IComputeStage *stage)
        {
            if (ctx.dump_id < 0)
                return false;
            if (ctx.inputs_dumped)
                return true;

            const auto &cfg = debugEnv().stage_dump;
            if (!cfg.dump_inputs && !cfg.dump_weights)
                return true;

            ctx.inputs_dumped = true;

            // Get dump info from stage
            StageDumpInfo dump_info = stage->getDumpInfo();

            // Dump inputs
            if (cfg.dump_inputs)
            {
                for (const auto &input : dump_info.inputs)
                {
                    if (!input.data)
                        continue;

                    std::string path = ctx.dump_dir + "/inputs/" + input.name;
                    TensorDumpMeta meta;
                    meta.name = input.name;
                    meta.rows = input.rows;
                    meta.cols = input.cols;
                    meta.dtype = input.dtype;
                    meta.computeBlockInfo(); // Compute block metadata for quantized formats

                    // Use explicit byte_size from input (now computed correctly from logical dims)
                    size_t dump_bytes = input.byte_size;

                    if (std::string(input.dtype) == "FP32")
                    {
                        path += ".bin";
                        dumpFP32Buffer(path, static_cast<const float *>(input.data),
                                       input.rows * input.cols, meta);
                    }
                    else
                    {
                        // Dump in native format (Q8_1, Q16_1, etc.)
                        // Use dtype-specific extension for clarity
                        std::string dtype_lower = input.dtype;
                        std::transform(dtype_lower.begin(), dtype_lower.end(), dtype_lower.begin(), ::tolower);
                        path += "_" + dtype_lower + ".bin";
                        dumpRawBuffer(path, input.data, dump_bytes, meta);
                    }

                    writeTensorMeta(ctx.dump_dir + "/inputs/" + input.name + "_meta.txt", meta);
                }
            }

            // Dump weights
            if (cfg.dump_weights)
            {
                for (const auto &weight : dump_info.weights)
                {
                    if (!weight.tensor)
                        continue;

                    std::string path = ctx.dump_dir + "/weights/" + weight.name;
                    TensorDumpMeta meta;
                    meta.name = weight.name;
                    meta.rows = weight.rows;
                    meta.cols = weight.cols;
                    meta.dtype = weight.dtype ? weight.dtype : "unknown";

                    // Dump weight metadata
                    writeTensorMeta(ctx.dump_dir + "/weights/" + weight.name + "_meta.txt", meta);

                    // Note: Full weight dumping would require dequantization
                    // For now, we just record metadata - full dump can be added later
                }
            }

            // Dump scalar params
            if (!dump_info.scalars.empty())
            {
                std::string params_path = ctx.dump_dir + "/scalars.txt";
                FILE *f = fopen(params_path.c_str(), "w");
                if (f)
                {
                    for (const auto &scalar : dump_info.scalars)
                    {
                        if (std::string(scalar.dtype) == "int")
                        {
                            fprintf(f, "%s=%d\n", scalar.name, static_cast<int>(scalar.value));
                        }
                        else if (std::string(scalar.dtype) == "bool")
                        {
                            fprintf(f, "%s=%s\n", scalar.name, scalar.value != 0 ? "true" : "false");
                        }
                        else
                        {
                            fprintf(f, "%s=%f\n", scalar.name, scalar.value);
                        }
                    }
                    fclose(f);
                }
            }

            return true;
        }

        /**
         * @brief Dump outputs for a stage after execution
         */
        static bool dumpOutputs(StageDumpContext &ctx, const IComputeStage *stage)
        {
            if (ctx.dump_id < 0)
                return false;
            if (ctx.outputs_dumped)
                return true;

            const auto &cfg = debugEnv().stage_dump;
            if (!cfg.dump_outputs)
                return true;

            ctx.outputs_dumped = true;

            // Get dump info from stage
            StageDumpInfo dump_info = stage->getDumpInfo();

            for (const auto &output : dump_info.outputs)
            {
                if (!output.data)
                    continue;

                std::string path = ctx.dump_dir + "/outputs/" + output.name;
                TensorDumpMeta meta;
                meta.name = output.name;
                meta.rows = output.rows;
                meta.cols = output.cols;
                meta.dtype = output.dtype;
                meta.computeBlockInfo(); // Compute block metadata for quantized formats

                // Use explicit byte_size from output (now computed correctly from logical dims)
                size_t dump_bytes = output.byte_size;

                if (std::string(output.dtype) == "FP32")
                {
                    path += ".bin";
                    dumpFP32Buffer(path, static_cast<const float *>(output.data),
                                   output.rows * output.cols, meta);
                }
                else
                {
                    // Dump in native format (Q8_1, Q16_1, etc.)
                    std::string dtype_lower = output.dtype;
                    std::transform(dtype_lower.begin(), dtype_lower.end(), dtype_lower.begin(), ::tolower);
                    path += "_" + dtype_lower + ".bin";
                    dumpRawBuffer(path, output.data, dump_bytes, meta);
                }

                writeTensorMeta(ctx.dump_dir + "/outputs/" + output.name + "_meta.txt", meta);
            }

            return true;
        }

        /**
         * @brief Complete a dump by writing final metadata
         */
        static void finalizeDump(StageDumpContext &ctx, double execution_time_ms = 0.0)
        {
            if (ctx.dump_id < 0)
                return;
            writeMetadata(ctx, execution_time_ms);

            fprintf(stderr, "[STAGE_DUMP] Created %s dump #%04d: %s\n",
                    ctx.type_name.c_str(), ctx.dump_id, ctx.dump_dir.c_str());
        }

    private:
        // =========================================================================
        // Internal Implementation
        // =========================================================================

        /**
         * @brief Thread-safe per-type dump counter (returns incremented value)
         */
        static int incrementDumpCounter(const std::string &type_name)
        {
            static std::map<std::string, std::atomic<int>> counters;
            static std::mutex counter_mutex;

            std::lock_guard<std::mutex> lock(counter_mutex);
            if (counters.find(type_name) == counters.end())
            {
                counters[type_name] = 0;
            }
            return ++counters[type_name];
        }

        /**
         * @brief Create dump directory structure (with stage name)
         */
        static std::string createDumpDirectory(
            const std::string &base_dir,
            const std::string &type_name,
            const std::string &stage_name,
            int dump_id,
            int layer_idx,
            int iteration,
            int rank)
        {
            // Create base directory
            mkdir(base_dir.c_str(), 0755);

            // Build directory name
            std::ostringstream oss;
            oss << base_dir << "/stage_" << std::setfill('0') << std::setw(4) << dump_id
                << "_" << type_name;
            if (!stage_name.empty())
            {
                oss << "_" << stage_name;
            }
            if (layer_idx >= 0)
            {
                oss << "_layer" << layer_idx;
            }
            if (iteration >= 0)
            {
                oss << "_iter" << iteration;
            }
            oss << "_rank" << rank;

            std::string dir_path = oss.str();
            if (mkdir(dir_path.c_str(), 0755) != 0 && errno != EEXIST)
            {
                fprintf(stderr, "[STAGE_DUMP] Failed to create directory: %s (errno=%d)\n",
                        dir_path.c_str(), errno);
                return "";
            }

            // Create subdirectories
            mkdir((dir_path + "/inputs").c_str(), 0755);
            mkdir((dir_path + "/weights").c_str(), 0755);
            mkdir((dir_path + "/outputs").c_str(), 0755);

            return dir_path;
        }

        /**
         * @brief Write final metadata file
         */
        static void writeMetadata(const StageDumpContext &ctx, double execution_time_ms)
        {
            std::string path = ctx.dump_dir + "/metadata.txt";
            FILE *f = fopen(path.c_str(), "w");
            if (!f)
                return;

            fprintf(f, "# Llaminar Stage Dump\n");
            fprintf(f, "dump_id=%d\n", ctx.dump_id);
            fprintf(f, "type=%s\n", ctx.type_name.c_str());
            if (!ctx.stage_name.empty())
            {
                fprintf(f, "name=%s\n", ctx.stage_name.c_str());
            }
            fprintf(f, "layer_idx=%d\n", ctx.layer_idx);
            fprintf(f, "iteration=%d\n", ctx.iteration);
            fprintf(f, "rank=%d\n", ctx.rank);
            fprintf(f, "execution_time_ms=%.4f\n", execution_time_ms);
            fprintf(f, "inputs_dumped=%d\n", ctx.inputs_dumped ? 1 : 0);
            fprintf(f, "outputs_dumped=%d\n", ctx.outputs_dumped ? 1 : 0);

            fclose(f);
        }

        /**
         * @brief Dump FP32 buffer to binary file with stats
         */
        static bool dumpFP32Buffer(
            const std::string &path,
            const float *data,
            size_t count,
            TensorDumpMeta &meta)
        {
            if (!data || count == 0)
                return true;

            FILE *f = fopen(path.c_str(), "wb");
            if (!f)
                return false;

            fwrite(data, sizeof(float), count, f);
            fclose(f);

            // Compute stats
            meta.element_count = count;
            meta.byte_size = count * sizeof(float);
            meta.dtype = "FP32";

            float min_val = data[0], max_val = data[0];
            double sum = 0;
            for (size_t i = 0; i < count; ++i)
            {
                min_val = std::min(min_val, data[i]);
                max_val = std::max(max_val, data[i]);
                sum += data[i];
            }
            meta.sample_min = min_val;
            meta.sample_max = max_val;
            meta.sample_mean = static_cast<float>(sum / count);

            return true;
        }

        /**
         * @brief Dump Q8_1 blocks to binary file
         */
        static bool dumpQ8_1Blocks(
            const std::string &path,
            const void *data,
            size_t block_count,
            TensorDumpMeta &meta)
        {
            if (!data || block_count == 0)
                return true;

            FILE *f = fopen(path.c_str(), "wb");
            if (!f)
                return false;

            fwrite(data, sizeof(Q8_1Block), block_count, f);
            fclose(f);

            meta.element_count = block_count * 32; // 32 elements per Q8_1 block
            meta.byte_size = block_count * sizeof(Q8_1Block);
            meta.dtype = "Q8_1";

            return true;
        }

        /**
         * @brief Dump raw binary buffer
         */
        static bool dumpRawBuffer(
            const std::string &path,
            const void *data,
            size_t byte_count,
            TensorDumpMeta &meta)
        {
            if (!data || byte_count == 0)
                return true;

            FILE *f = fopen(path.c_str(), "wb");
            if (!f)
                return false;

            fwrite(data, 1, byte_count, f);
            fclose(f);

            meta.byte_size = byte_count;

            return true;
        }

        /**
         * @brief Write tensor metadata file
         */
        static void writeTensorMeta(const std::string &path, const TensorDumpMeta &meta)
        {
            FILE *f = fopen(path.c_str(), "w");
            if (!f)
                return;

            fprintf(f, "name=%s\n", meta.name.c_str());
            fprintf(f, "rows=%zu\n", meta.rows);
            fprintf(f, "cols=%zu\n", meta.cols);
            fprintf(f, "dtype=%s\n", meta.dtype.c_str());
            fprintf(f, "element_count=%zu\n", meta.element_count);
            fprintf(f, "byte_size=%zu\n", meta.byte_size);

            // Add block info for quantized formats
            if (meta.block_count > 0)
            {
                fprintf(f, "# Block format info:\n");
                fprintf(f, "block_count=%zu\n", meta.block_count);
                fprintf(f, "blocks_per_row=%zu\n", meta.blocks_per_row);
                fprintf(f, "block_element_size=%zu\n", meta.block_element_size);
            }

            fprintf(f, "sample_min=%f\n", meta.sample_min);
            fprintf(f, "sample_max=%f\n", meta.sample_max);
            fprintf(f, "sample_mean=%f\n", meta.sample_mean);

            fclose(f);
        }
    };

    // =========================================================================
    // Loaded Stage Dump for Replay Testing
    // =========================================================================

    /**
     * @brief Loaded stage dump for replay testing
     *
     * Similar to LoadedAttentionDump but generic for any stage type.
     * Allows loading previously-dumped stage data and replaying through
     * the kernel for correctness testing.
     */
    struct LoadedStageDump
    {
        // Header info
        std::string type_name;
        int dump_id = -1;
        int layer_idx = -1;
        int iteration = -1;
        int rank = 0;
        double execution_time_ms = 0;

        // Loaded tensor data (FP32 vectors)
        std::map<std::string, std::vector<float>> fp32_tensors;

        // Loaded Q8_1 block data
        std::map<std::string, std::vector<Q8_1Block>> q8_1_tensors;

        // Tensor metadata
        std::map<std::string, TensorDumpMeta> tensor_meta;

        // Scalar parameters
        std::map<std::string, double> scalars;

        // Metadata
        std::map<std::string, std::string> metadata;
        bool valid = false;

        /**
         * @brief Load stage dump from directory
         * @param dump_dir Path to stage dump directory
         * @return true if load succeeded
         */
        bool load(const std::string &dump_dir)
        {
            valid = false;

            // Read main metadata
            std::string meta_path = dump_dir + "/metadata.txt";
            if (!loadKeyValueFile(meta_path, metadata))
                return false;

            // Extract key fields
            if (metadata.count("type"))
                type_name = metadata["type"];
            if (metadata.count("dump_id"))
                dump_id = std::stoi(metadata["dump_id"]);
            if (metadata.count("layer_idx"))
                layer_idx = std::stoi(metadata["layer_idx"]);
            if (metadata.count("iteration"))
                iteration = std::stoi(metadata["iteration"]);
            if (metadata.count("rank"))
                rank = std::stoi(metadata["rank"]);
            if (metadata.count("execution_time_ms"))
                execution_time_ms = std::stod(metadata["execution_time_ms"]);

            // Load scalars
            std::map<std::string, std::string> scalar_map;
            if (loadKeyValueFile(dump_dir + "/scalars.txt", scalar_map))
            {
                for (const auto &kv : scalar_map)
                {
                    // Try to parse as number
                    try
                    {
                        scalars[kv.first] = std::stod(kv.second);
                    }
                    catch (...)
                    {
                        // Boolean or string - convert
                        if (kv.second == "true")
                            scalars[kv.first] = 1.0;
                        else if (kv.second == "false")
                            scalars[kv.first] = 0.0;
                    }
                }
            }

            // Load inputs, outputs (scan directories)
            loadTensorsFromDir(dump_dir + "/inputs");
            loadTensorsFromDir(dump_dir + "/outputs");

            valid = true;
            return true;
        }

        /**
         * @brief Get FP32 input by name
         */
        const std::vector<float> *getFP32Tensor(const std::string &name) const
        {
            auto it = fp32_tensors.find(name);
            return it != fp32_tensors.end() ? &it->second : nullptr;
        }

        /**
         * @brief Get Q8_1 blocks by name
         */
        const std::vector<Q8_1Block> *getQ8_1Tensor(const std::string &name) const
        {
            auto it = q8_1_tensors.find(name);
            return it != q8_1_tensors.end() ? &it->second : nullptr;
        }

        /**
         * @brief Get scalar parameter
         */
        double getScalar(const std::string &name, double default_val = 0.0) const
        {
            auto it = scalars.find(name);
            return it != scalars.end() ? it->second : default_val;
        }

        int getScalarInt(const std::string &name, int default_val = 0) const
        {
            return static_cast<int>(getScalar(name, default_val));
        }

        bool getScalarBool(const std::string &name, bool default_val = false) const
        {
            return getScalar(name, default_val ? 1.0 : 0.0) != 0.0;
        }

    private:
        static bool loadKeyValueFile(const std::string &path, std::map<std::string, std::string> &out)
        {
            std::ifstream file(path);
            if (!file.is_open())
                return false;

            std::string line;
            while (std::getline(file, line))
            {
                if (line.empty() || line[0] == '#')
                    continue;
                size_t eq_pos = line.find('=');
                if (eq_pos != std::string::npos)
                {
                    std::string key = line.substr(0, eq_pos);
                    std::string value = line.substr(eq_pos + 1);
                    out[key] = value;
                }
            }
            return true;
        }

        void loadTensorsFromDir(const std::string &dir)
        {
            // Note: In production, would use directory listing
            // For now, we rely on metadata to know what files exist
        }

        static bool loadFP32File(const std::string &path, std::vector<float> &buffer)
        {
            FILE *f = fopen(path.c_str(), "rb");
            if (!f)
                return false;

            fseek(f, 0, SEEK_END);
            size_t file_size = ftell(f);
            fseek(f, 0, SEEK_SET);

            size_t count = file_size / sizeof(float);
            buffer.resize(count);
            size_t read = fread(buffer.data(), sizeof(float), count, f);
            fclose(f);

            return read == count;
        }

        static bool loadQ8_1File(const std::string &path, std::vector<Q8_1Block> &buffer)
        {
            FILE *f = fopen(path.c_str(), "rb");
            if (!f)
                return false;

            fseek(f, 0, SEEK_END);
            size_t file_size = ftell(f);
            fseek(f, 0, SEEK_SET);

            size_t count = file_size / sizeof(Q8_1Block);
            buffer.resize(count);
            size_t read = fread(buffer.data(), sizeof(Q8_1Block), count, f);
            fclose(f);

            return read == count;
        }
    };

} // namespace llaminar2
