/**
 * @file CudaGemmNeuralNetwork.cu
 * @brief ONNX Runtime neural network implementation
 *
 * @author David Sanftenberg
 * @date November 2, 2025
 */

#ifdef HAVE_ONNX_RUNTIME

#include "CudaGemmNeuralNetwork.h"
#include "../../utils/Logger.h"
#include <cmath>
#include <cassert>
#include <fstream>
#include <algorithm>
#include <sstream>

namespace llaminar2
{
    namespace cuda
    {

        CudaGemmNeuralNetwork &CudaGemmNeuralNetwork::instance()
        {
            static CudaGemmNeuralNetwork instance;
            return instance;
        }

        CudaGemmNeuralNetwork::CudaGemmNeuralNetwork()
        {
            initialize();
        }

        void CudaGemmNeuralNetwork::initialize()
        {
            // Determine model path (relative to source directory)
            // Assume model is in src/v2/kernels/cuda/
            model_path_ = "src/v2/kernels/cuda/cuda_heuristic_nn.onnx";
            scaler_path_ = "src/v2/kernels/cuda/cuda_heuristic_scaler.txt"; // Changed from .npz to .txt

            // Check if files exist
            std::ifstream model_file(model_path_);
            if (!model_file.good())
            {
                LOG_WARN("[CUDA NN] ONNX model not found: " << model_path_);
                LOG_WARN("[CUDA NN] Train with: python3 src/v2/kernels/cuda/python/train_cuda_neural_network.py");
                initialized_ = false;
                return;
            }

            // Load scaler parameters
            if (!loadScalerParameters())
            {
                LOG_WARN("[CUDA NN] Failed to load scaler parameters, using identity transform");
                // Continue anyway with identity transform (mean=0, scale=1)
                feature_mean_.fill(0.0f);
                feature_scale_.fill(1.0f);
            }

            // Initialize ONNX Runtime
            try
            {
                session_options_.SetIntraOpNumThreads(1); // Single-threaded inference
                session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

                session_ = std::make_unique<Ort::Session>(env_, model_path_.c_str(), session_options_);

                // Get input/output names and store them
                Ort::AllocatorWithDefaultOptions allocator;

                // Input name - copy to string storage
                auto input_name_ptr = session_->GetInputNameAllocated(0, allocator);
                input_name_storage_ = std::string(input_name_ptr.get());
                input_names_.push_back(input_name_storage_.c_str());

                // Output name - copy to string storage
                auto output_name_ptr = session_->GetOutputNameAllocated(0, allocator);
                output_name_storage_ = std::string(output_name_ptr.get());
                output_names_.push_back(output_name_storage_.c_str());

                initialized_ = true;
                LOG_INFO("[CUDA NN] Neural network initialized successfully");
                LOG_INFO("[CUDA NN] Model: " << model_path_);
                LOG_INFO("[CUDA NN] Input: " << input_names_[0] << " (73 features)");
                LOG_INFO("[CUDA NN] Output: " << output_names_[0] << " (GFLOPS prediction)");
                LOG_INFO("[CUDA NN] Expected Test R²: 0.96-0.97 (96-97% accuracy)");
                LOG_INFO("[CUDA NN] Scaler: RobustScaler (median/IQR, better outlier handling)");
            }
            catch (const Ort::Exception &e)
            {
                LOG_ERROR("[CUDA NN] ONNX Runtime initialization failed: " << e.what());
                initialized_ = false;
            }
        }

        bool CudaGemmNeuralNetwork::loadScalerParameters()
        {
            std::ifstream file(scaler_path_);
            if (!file.good())
            {
                LOG_WARN("[CUDA NN] Scaler file not found: " << scaler_path_);
                return false;
            }

            std::string line;

            // Read "MEAN" line (actually median for RobustScaler)
            if (!std::getline(file, line) || line != "MEAN")
            {
                LOG_WARN("[CUDA NN] Invalid scaler file format (expected MEAN)");
                return false;
            }

            // Read 73 center values (median for RobustScaler)
            for (int i = 0; i < 73; i++)
            {
                if (!std::getline(file, line))
                {
                    LOG_WARN("[CUDA NN] Unexpected end of file reading center values at index " << i);
                    return false;
                }
                feature_mean_[i] = std::stof(line);
            }

            // Read "SCALE" line (actually IQR for RobustScaler)
            if (!std::getline(file, line) || line != "SCALE")
            {
                LOG_WARN("[CUDA NN] Invalid scaler file format (expected SCALE)");
                return false;
            }

            // Read 73 scale values (IQR for RobustScaler)
            for (int i = 0; i < 73; i++)
            {
                if (!std::getline(file, line))
                {
                    LOG_WARN("[CUDA NN] Unexpected end of file reading scale values at index " << i);
                    return false;
                }
                feature_scale_[i] = std::stof(line);
            }

            LOG_INFO("[CUDA NN] Loaded RobustScaler parameters from " << scaler_path_);
            LOG_INFO("[CUDA NN] Features: 73 (center=median, scale=IQR)");
            return true;
        }

        std::array<float, 73> CudaGemmNeuralNetwork::extractFeatures(
            const CudaGemmConfig &config, int m, int n, int k)
        {

            std::array<float, 73> features;
            int idx = 0;

            // Raw features (13 total)
            features[idx++] = static_cast<float>(config.tile_m);
            features[idx++] = static_cast<float>(config.tile_n);
            features[idx++] = static_cast<float>(config.tile_k);
            features[idx++] = static_cast<float>(config.threads_m);
            features[idx++] = static_cast<float>(config.threads_n);
            features[idx++] = static_cast<float>(config.work_per_thread_m); // work_m → work_per_thread_m
            features[idx++] = static_cast<float>(config.work_per_thread_n); // work_n → work_per_thread_n
            features[idx++] = static_cast<float>(config.prefetch_stages);
            features[idx++] = static_cast<float>(config.transpose_smem);
            features[idx++] = static_cast<float>(config.vectorize_load);
            features[idx++] = static_cast<float>(m);
            features[idx++] = static_cast<float>(n);
            features[idx++] = static_cast<float>(k);

            // Base engineered features (19 total)
            int threads_per_block = config.threads_m * config.threads_n;
            int tile_size = config.tile_m * config.tile_n;
            int tile_area = config.tile_m * config.tile_n * config.tile_k;
            int work_per_thread = config.work_per_thread_m * config.work_per_thread_n; // Updated

            features[idx++] = static_cast<float>(threads_per_block);
            features[idx++] = static_cast<float>(tile_size);
            features[idx++] = static_cast<float>(tile_area);
            features[idx++] = static_cast<float>(work_per_thread);

            // Occupancy estimate
            float smem_per_block = tile_area * 4.0f; // 4 bytes per float
            float occupancy_estimate = std::min(
                48000.0f / std::max(smem_per_block, 1.0f),
                1024.0f / std::max(static_cast<float>(threads_per_block), 1.0f));
            features[idx++] = occupancy_estimate;

            // Arithmetic intensity
            float arithmetic_intensity = (2.0f * config.tile_m * config.tile_n * config.tile_k) /
                                         ((config.tile_m * config.tile_k + config.tile_k * config.tile_n) * 4.0f);
            features[idx++] = arithmetic_intensity;

            // Problem size ratios
            features[idx++] = static_cast<float>(m) / std::max(static_cast<float>(config.tile_m), 1.0f);
            features[idx++] = static_cast<float>(n) / std::max(static_cast<float>(config.tile_n), 1.0f);
            features[idx++] = static_cast<float>(k) / std::max(static_cast<float>(config.tile_k), 1.0f);

            // Tile coverage
            features[idx++] = static_cast<float>(m % config.tile_m) / std::max(static_cast<float>(config.tile_m), 1.0f);
            features[idx++] = static_cast<float>(n % config.tile_n) / std::max(static_cast<float>(config.tile_n), 1.0f);

            // Size categories
            size_t total_size = static_cast<size_t>(m) * n * k;
            features[idx++] = (total_size < 1000000) ? 1.0f : 0.0f;                             // is_tiny
            features[idx++] = (total_size >= 1000000 && total_size < 10000000) ? 1.0f : 0.0f;   // is_small
            features[idx++] = (total_size >= 10000000 && total_size < 100000000) ? 1.0f : 0.0f; // is_medium
            features[idx++] = (total_size >= 100000000) ? 1.0f : 0.0f;                          // is_large

            // Shape features
            features[idx++] = (n == k) ? 1.0f : 0.0f;                                                                // is_square
            features[idx++] = static_cast<float>(n) / std::max(static_cast<float>(k), 1.0f);                         // aspect_ratio
            features[idx++] = static_cast<float>(config.tile_n) / std::max(static_cast<float>(config.tile_k), 1.0f); // tile_aspect_ratio
            float aspect_ratio = static_cast<float>(n) / std::max(static_cast<float>(k), 1.0f);
            float tile_aspect_ratio = static_cast<float>(config.tile_n) / std::max(static_cast<float>(config.tile_k), 1.0f);
            features[idx++] = std::abs(aspect_ratio - tile_aspect_ratio); // tile_shape_match

            // Enhanced features (25 total)

            // Batch-aware (3)
            features[idx++] = std::log2(std::max(static_cast<float>(m), 1.0f)); // batch_size_log2
            features[idx++] = (m == 1) ? 1.0f : 0.0f;                           // is_single_token
            features[idx++] = ((m & (m - 1)) == 0) ? 1.0f : 0.0f;               // is_power_of_2_batch

            // Alignment (6)
            features[idx++] = (n % 16 == 0) ? 1.0f : 0.0f;            // n_aligned_16
            features[idx++] = (n % 32 == 0) ? 1.0f : 0.0f;            // n_aligned_32
            features[idx++] = (n % 64 == 0) ? 1.0f : 0.0f;            // n_aligned_64
            features[idx++] = (k % 32 == 0) ? 1.0f : 0.0f;            // k_aligned_32
            features[idx++] = (n % config.tile_n == 0) ? 1.0f : 0.0f; // n_aligned_tile
            features[idx++] = (k % config.tile_k == 0) ? 1.0f : 0.0f; // k_aligned_tile

            // Efficiency (5)
            features[idx++] = static_cast<float>(threads_per_block) / 32.0f;                                         // warp_efficiency
            features[idx++] = 48000.0f / std::max(static_cast<float>(threads_per_block), 32.0f);                     // blocks_per_sm_estimate
            features[idx++] = static_cast<float>((m % config.tile_m) + (n % config.tile_n));                         // work_imbalance
            features[idx++] = static_cast<float>(config.tile_m * config.tile_n * config.tile_k);                     // work_total
            features[idx++] = static_cast<float>(tile_area) / std::max(static_cast<float>(threads_per_block), 1.0f); // work_per_thread_normalized

            // Memory/compute (3)
            features[idx++] = 1.0f / std::max(arithmetic_intensity, 0.1f);                       // bytes_loaded_per_flop
            features[idx++] = static_cast<float>(config.prefetch_stages) * arithmetic_intensity; // prefetch_benefit
            features[idx++] = ((config.vectorize_load * 4) <= 16) ? 1.0f : 0.0f;                 // vec_load_aligned

            // Tile coverage (5)
            features[idx++] = std::ceil(static_cast<float>(m) / config.tile_m); // m_tiles
            features[idx++] = std::ceil(static_cast<float>(n) / config.tile_n); // n_tiles
            features[idx++] = std::ceil(static_cast<float>(k) / config.tile_k); // k_tiles
            features[idx++] = ((m % config.tile_m != 0) ? 1.0f : 0.0f) +
                              ((n % config.tile_n != 0) ? 1.0f : 0.0f); // partial_tiles

            // Size category (1)
            int size_category = 0;
            if (total_size >= 1000000 && total_size < 10000000)
                size_category = 1;
            else if (total_size >= 10000000 && total_size < 100000000)
                size_category = 2;
            else if (total_size >= 100000000)
                size_category = 3;
            features[idx++] = static_cast<float>(size_category);

            // Interaction features (3)
            features[idx++] = static_cast<float>(tile_size) * std::log2(std::max(static_cast<float>(m), 1.0f)); // tile_size_x_batch
            features[idx++] = occupancy_estimate * arithmetic_intensity;                                        // occupancy_x_intensity
            features[idx++] = static_cast<float>(work_per_thread) * m;                                          // work_per_thread_x_batch

            // Hardware-aware features (NEW - 8 features)
            features[idx++] = static_cast<float>(threads_per_block) / 32.0f;       // warps_per_block
            features[idx++] = static_cast<float>(threads_per_block % 32) / 32.0f;  // warp_utilization (partial warp inefficiency)
            features[idx++] = static_cast<float>(std::max(config.tile_n % 32, 1)); // smem_bank_conflicts (32 banks on most GPUs)
            features[idx++] = ((n % 128) != 0) ? 1.0f : 0.0f;                      // coalescing_penalty (128-byte cache lines)
            features[idx++] = static_cast<float>(config.vectorize_load) / 4.0f;    // vec_load_efficiency (normalized 0-1)

            float m_tiles = std::ceil(static_cast<float>(m) / config.tile_m);
            float n_tiles = std::ceil(static_cast<float>(n) / config.tile_n);
            float k_tiles = std::ceil(static_cast<float>(k) / config.tile_k);
            features[idx++] = std::ceil(m_tiles * n_tiles / 80.0f);                                                  // tiles_per_sm (~80 SMs on A100)
            features[idx++] = k_tiles;                                                                               // tile_reuse_factor (how many times tiles are reused)
            features[idx++] = static_cast<float>(tile_area) / std::max(static_cast<float>(threads_per_block), 1.0f); // tile_compute_density

            // PHASE 2: Estimated profiler metrics (8 features - ZERO runtime cost!)
            // These approximate what ncu profiler would measure, but computed from config

            // Bank conflict risk (0-1 scale, higher = more conflicts)
            features[idx++] = std::min(static_cast<float>(config.tile_n % 32), 16.0f) / 16.0f; // bank_conflict_risk

            // Coalescing efficiency score (0-1, higher = better)
            float coalescing_score = 0.0f;
            if (n % 128 == 0)
                coalescing_score += 1.0f;
            if (n % 64 == 0)
                coalescing_score += 0.5f;
            if (n % 32 == 0)
                coalescing_score += 0.25f;
            features[idx++] = coalescing_score;

            // Register pressure (work per thread)
            features[idx++] = static_cast<float>(tile_area) / std::max(static_cast<float>(threads_per_block), 1.0f); // register_pressure

            // Memory vs compute ratio (higher = more memory-bound)
            features[idx++] = (1.0f / std::max(arithmetic_intensity, 0.1f)) * 1000.0f; // mem_compute_ratio

            // Warp divergence risk (0-1, higher = more divergence)
            features[idx++] = ((threads_per_block & (threads_per_block - 1)) == 0) ? 0.0f : 1.0f; // warp_divergence_risk

            // Shared memory pressure (KB per block)
            float smem_kb = (static_cast<float>(tile_area) * 4.0f) / 1024.0f;
            features[idx++] = std::min(smem_kb, 48.0f); // smem_kb_per_block

            // L1 cache pressure (0-1, higher = more pressure)
            features[idx++] = std::min((static_cast<float>(tile_area) * 4.0f) / (1024.0f * 128.0f), 1.0f); // l1_cache_pressure

            // Occupancy limiter (0=none, 1=smem, 2=threads, 3=registers)
            float occupancy_limiter = 0.0f;
            if (smem_kb > 32.0f)
                occupancy_limiter = 1.0f; // SMEM limited
            if (threads_per_block > 512)
                occupancy_limiter = 2.0f; // Thread limited
            if (tile_area / std::max(static_cast<float>(threads_per_block), 1.0f) > 128.0f)
                occupancy_limiter = 3.0f; // Register limited
            features[idx++] = occupancy_limiter;

            // Should be exactly 73 features
            assert(idx == 73);

            return features;
        }

        double CudaGemmNeuralNetwork::predict(const CudaGemmConfig &config, int m, int n, int k)
        {
            if (!initialized_)
            {
                LOG_WARN("[CUDA NN] Neural network not initialized, using fallback score");
                return 0.0; // Fallback to manual heuristic
            }

            try
            {
                // Extract and scale features
                auto features = extractFeatures(config, m, n, k);

                // Apply RobustScaler: (x - center) / scale
                // center = median, scale = IQR (interquartile range)
                for (int i = 0; i < 73; i++)
                {
                    features[i] = (features[i] - feature_mean_[i]) / feature_scale_[i];
                }

                // Create input tensor
                std::vector<int64_t> input_shape = {1, 73}; // Batch size 1, 73 features
                Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                    memory_info, features.data(), 73, input_shape.data(), input_shape.size());

                // Run inference
                auto output_tensors = session_->Run(
                    Ort::RunOptions{nullptr},
                    input_names_.data(), &input_tensor, 1,
                    output_names_.data(), 1);

                // Extract prediction
                float *output_data = output_tensors[0].GetTensorMutableData<float>();
                double predicted_gflops = static_cast<double>(output_data[0]);

                return predicted_gflops;
            }
            catch (const Ort::Exception &e)
            {
                LOG_ERROR("[CUDA NN] Inference failed: " << e.what());
                return 0.0;
            }
        }

    } // namespace cuda
} // namespace llaminar2

#endif // HAVE_ONNX_RUNTIME
