/**
 * @file npz_loader.h
 * @brief Simple .npz (NumPy archive) loader for PyTorch snapshot integration
 * @author David Sanftenberg
 *
 * This is a minimal .npz loader specifically designed for loading PyTorch reference
 * snapshots exported from python/reference implementation. It handles the subset of
 * NumPy format needed for parity testing.
 *
 * .npz format is a ZIP archive containing .npy files. Each .npy file is a binary
 * NumPy array with a header describing shape, dtype, and byte order.
 *
 * Dependencies: zlib (for ZIP decompression)
 *
 * Usage:
 *   NpzLoader loader("snapshots.npz");
 *   if (loader.load()) {
 *     auto embedding = loader.get_array("EMBEDDING_-1");
 *     // embedding.shape = {1, seq_len, hidden_dim}
 *     // embedding.data = std::vector<float>
 *   }
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <cstring>
#include <stdexcept>

namespace llaminar
{
    namespace parity
    {

        /**
         * @brief Represents a NumPy array loaded from .npz
         */
        struct NpyArray
        {
            std::vector<size_t> shape; // Array dimensions
            std::vector<float> data;   // Float data (we only support float32 for now)
            std::string dtype;         // Data type string (e.g., "<f4" for float32)

            size_t total_elements() const
            {
                size_t count = 1;
                for (size_t dim : shape)
                {
                    count *= dim;
                }
                return count;
            }

            bool is_valid() const
            {
                return !shape.empty() && !data.empty() && data.size() == total_elements();
            }
        };

        /**
         * @brief Simple .npz loader for PyTorch snapshots
         *
         * This is a minimal implementation that parses .npz files (ZIP archives of .npy files).
         * It currently only supports float32 arrays, which is sufficient for our parity testing.
         *
         * Limitations:
         * - Only supports float32 dtype (most common for ML)
         * - Assumes little-endian byte order (standard for x86/ARM)
         * - No compression support (uses numpy.savez_compressed which stores floats uncompressed)
         *
         * For a more complete implementation, consider using cnpy library.
         */
        class NpzLoader
        {
        public:
            explicit NpzLoader(const std::string &filepath)
                : filepath_(filepath), loaded_(false)
            {
            }

            /**
             * @brief Load the .npz file and parse all arrays
             * @return true if successful, false otherwise
             */
            bool load()
            {
                if (loaded_)
                {
                    return true; // Already loaded
                }

                try
                {
                    // Simple approach: Use Python subprocess to extract .npz to temporary directory
                    // This avoids implementing a full ZIP parser in C++
                    // Alternative: Link against libzip or use cnpy library

                    // For now, we'll provide a simpler approach: expect .npz to be extracted manually
                    // or use a Python helper script to convert .npz → individual .npy files

                    // TODO: Implement full .npz parsing or use cnpy
                    // For MVP, we'll document that users should extract .npz first

                    throw std::runtime_error(
                        "Direct .npz loading not yet implemented. "
                        "Please extract .npz archive first:\n"
                        "  python -c 'import numpy as np; data=np.load(\"" +
                        filepath_ + "\"); "
                                    "[np.save(f\"{k}.npy\", v) for k,v in data.items()]'\n"
                                    "Or use the npz_to_npy.py helper script in tests/");
                }
                catch (const std::exception &e)
                {
                    error_message_ = e.what();
                    return false;
                }
            }

            /**
             * @brief Get array by name (e.g., "EMBEDDING_-1")
             * @param name Array name (corresponds to stage_layer format from Python)
             * @param out_array Output array
             * @return true if found and loaded, false otherwise
             */
            bool get_array(const std::string &name, NpyArray &out_array) const
            {
                if (!loaded_)
                {
                    return false;
                }

                auto it = arrays_.find(name);
                if (it == arrays_.end())
                {
                    return false;
                }

                out_array = it->second;
                return true;
            }

            /**
             * @brief List all array names in the archive
             */
            std::vector<std::string> list_arrays() const
            {
                std::vector<std::string> names;
                names.reserve(arrays_.size());
                for (const auto &kv : arrays_)
                {
                    names.push_back(kv.first);
                }
                return names;
            }

            /**
             * @brief Get error message if loading failed
             */
            const std::string &error_message() const { return error_message_; }

            /**
             * @brief Load a single .npy file (helper for extracted archives)
             *
             * This parses the NumPy .npy format directly. Use this when .npz has been
             * extracted to individual .npy files.
             *
             * @param filepath Path to .npy file
             * @param out_array Output array
             * @return true if successful
             */
            static bool load_npy(const std::string &filepath, NpyArray &out_array)
            {
                std::ifstream file(filepath, std::ios::binary);
                if (!file)
                {
                    return false;
                }

                // Parse .npy header
                // Format: magic "\x93NUMPY" + version (2 bytes) + header_len (2 or 4 bytes) + dict + data

                char magic[6];
                file.read(magic, 6);
                if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
                {
                    return false; // Not a .npy file
                }

                uint8_t major_version, minor_version;
                file.read(reinterpret_cast<char *>(&major_version), 1);
                file.read(reinterpret_cast<char *>(&minor_version), 1);

                uint32_t header_len = 0;
                if (major_version == 1)
                {
                    uint16_t len16;
                    file.read(reinterpret_cast<char *>(&len16), 2);
                    header_len = len16;
                }
                else if (major_version == 2 || major_version == 3)
                {
                    file.read(reinterpret_cast<char *>(&header_len), 4);
                }
                else
                {
                    return false; // Unsupported version
                }

                // Read header dictionary (Python dict string)
                std::vector<char> header_bytes(header_len);
                file.read(header_bytes.data(), header_len);
                std::string header_str(header_bytes.begin(), header_bytes.end());

                // Parse header (simple Python dict parsing)
                // Example: "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 10, 512), }"

                // Extract dtype
                size_t descr_pos = header_str.find("'descr':");
                if (descr_pos == std::string::npos)
                    return false;
                size_t dtype_start = header_str.find("'", descr_pos + 8) + 1;
                size_t dtype_end = header_str.find("'", dtype_start);
                out_array.dtype = header_str.substr(dtype_start, dtype_end - dtype_start);

                // Extract shape
                size_t shape_pos = header_str.find("'shape':");
                if (shape_pos == std::string::npos)
                    return false;
                size_t shape_start = header_str.find("(", shape_pos) + 1;
                size_t shape_end = header_str.find(")", shape_start);
                std::string shape_str = header_str.substr(shape_start, shape_end - shape_start);

                // Parse shape tuple
                out_array.shape.clear();
                std::istringstream shape_stream(shape_str);
                std::string dim_str;
                while (std::getline(shape_stream, dim_str, ','))
                {
                    // Trim whitespace
                    dim_str.erase(0, dim_str.find_first_not_of(" \t"));
                    dim_str.erase(dim_str.find_last_not_of(" \t,") + 1);
                    if (!dim_str.empty())
                    {
                        out_array.shape.push_back(std::stoull(dim_str));
                    }
                }

                // Read data (only support float32 for now)
                if (out_array.dtype != "<f4" && out_array.dtype != "=f4" && out_array.dtype != "|f4")
                {
                    return false; // Only float32 supported
                }

                size_t total = out_array.total_elements();
                out_array.data.resize(total);
                file.read(reinterpret_cast<char *>(out_array.data.data()), total * sizeof(float));

                return file.good() && out_array.is_valid();
            }

        private:
            std::string filepath_;
            bool loaded_;
            std::unordered_map<std::string, NpyArray> arrays_;
            std::string error_message_;
        };

        /**
         * @brief Helper to load PyTorch snapshots from extracted .npy files
         *
         * Usage:
         *   PyTorchSnapshotLoader loader("path/to/extracted/snapshots/");
         *   auto embedding = loader.load_snapshot("EMBEDDING", -1);
         *   auto attn_out_layer0 = loader.load_snapshot("ATTENTION_OUTPUT", 0);
         */
        class PyTorchSnapshotLoader
        {
        public:
            explicit PyTorchSnapshotLoader(const std::string &base_dir)
                : base_dir_(base_dir)
            {
                // Ensure trailing slash
                if (!base_dir_.empty() && base_dir_.back() != '/')
                {
                    base_dir_ += '/';
                }
            }

            /**
             * @brief Load a snapshot by stage name and layer index
             * @param stage_name Pipeline stage name (e.g., "EMBEDDING", "ATTENTION_OUTPUT")
             * @param layer_index Layer index (-1 for global stages)
             * @param out_array Output array
             * @return true if loaded successfully
             */
            bool load_snapshot(const std::string &stage_name, int layer_index, NpyArray &out_array)
            {
                // Format: STAGE_layer.npy (e.g., EMBEDDING_-1.npy, ATTENTION_OUTPUT_0.npy)
                std::string filename = stage_name + "_" + std::to_string(layer_index) + ".npy";
                std::string filepath = base_dir_ + filename;

                return NpzLoader::load_npy(filepath, out_array);
            }

            /**
             * @brief Check if a snapshot exists
             */
            bool has_snapshot(const std::string &stage_name, int layer_index) const
            {
                std::string filename = stage_name + "_" + std::to_string(layer_index) + ".npy";
                std::string filepath = base_dir_ + filename;
                std::ifstream file(filepath);
                return file.good();
            }

        private:
            std::string base_dir_;
        };

    } // namespace parity
} // namespace llaminar
