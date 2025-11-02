/**
 * @file CudaGemmAutoTuner.cu
 * @brief Implementation of CUDA GEMM auto-tuner
 *
 * @author David Sanftenberg
 * @date October 31, 2025
 * @updated November 1, 2025 - Added ML-based tile predictor (Phase 3)
 * @updated November 2, 2025 - Added ONNX neural network heuristic (Phase 24)
 */

#include "CudaGemmAutoTuner.h"
#include "CudaGemmVariantsBaseline.h"
#include "generated/cuda_heuristic_weights.h"
#include "generated/cuda_heuristic_lookup.h"
#include "../../tensors/FP16Utils.h"
#include "../../../../build_v2/autotuner_models/GemmAutoTunerML.h" // ML-based predictor
#ifdef HAVE_ONNX_RUNTIME
#include "CudaGemmNeuralNetwork.h"
#endif
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <random>
#include <cmath>
#include <iostream>

// Simple logging macros (avoid dependency on V2 Logging.h which may not be CUDA-compatible)
#define LOG_INFO(msg) std::cout << "[INFO] " << msg << std::endl
#define LOG_ERROR(msg) std::cerr << "[ERROR] " << msg << std::endl
#define LOG_DEBUG(msg) std::cout << "[DEBUG] " << msg << std::endl
#define LOG_WARN(msg) std::cerr << "[WARN] " << msg << std::endl

namespace llaminar2
{
    namespace cuda
    {

        // Environment flags to control heuristic selection
        static bool useMLHeuristic()
        {
            const char *use_ml = std::getenv("LLAMINAR_USE_ML_HEURISTIC");
            return use_ml && std::atoi(use_ml) != 0;
        }

        static bool useNNHeuristic()
        {
            const char *use_nn = std::getenv("LLAMINAR_USE_NN_HEURISTIC");
            return use_nn && std::atoi(use_nn) != 0;
        }

        // ============================================================================
        // Singleton and Lifecycle
        // ============================================================================

        CudaGemmAutoTuner &CudaGemmAutoTuner::instance()
        {
            static CudaGemmAutoTuner instance;
            return instance;
        }

        CudaGemmAutoTuner::CudaGemmAutoTuner()
        {
            // Read environment variables
            const char *disable_env = std::getenv("LLAMINAR_DISABLE_CUDA_AUTOTUNE");
            if (disable_env && std::atoi(disable_env) != 0)
            {
                auto_tuning_enabled_ = false;
                LOG_INFO("[CUDA AutoTuner] Auto-tuning disabled via LLAMINAR_DISABLE_CUDA_AUTOTUNE");
            }

            const char *max_candidates_env = std::getenv("LLAMINAR_CUDA_AUTOTUNE_CANDIDATES");
            if (max_candidates_env)
            {
                max_candidates_ = std::atoi(max_candidates_env);
                LOG_INFO("[CUDA AutoTuner] Max candidates set to " << max_candidates_);
            }

            const char *iterations_env = std::getenv("LLAMINAR_CUDA_AUTOTUNE_ITERATIONS");
            if (iterations_env)
            {
                benchmark_iterations_ = std::atoi(iterations_env);
                LOG_INFO("[CUDA AutoTuner] Benchmark iterations set to " << benchmark_iterations_);
            }

            // Get device properties
            cudaGetDevice(&device_id_);
            cudaGetDeviceProperties(&device_props_, device_id_);

            LOG_INFO("[CUDA AutoTuner] Device: " << device_props_.name);
            LOG_INFO("[CUDA AutoTuner] Compute capability: " << device_props_.major << "." << device_props_.minor);
            LOG_INFO("[CUDA AutoTuner] Max shared memory per block: " << device_props_.sharedMemPerBlock / 1024 << " KB");
            LOG_INFO("[CUDA AutoTuner] Max registers per block: " << device_props_.regsPerBlock);

            // Create CUDA resources
            cudaStreamCreate(&benchmark_stream_);
            cudaEventCreate(&start_event_);
            cudaEventCreate(&stop_event_);
        }

        CudaGemmAutoTuner::~CudaGemmAutoTuner()
        {
            freeTestData();

            if (benchmark_stream_)
                cudaStreamDestroy(benchmark_stream_);
            if (start_event_)
                cudaEventDestroy(start_event_);
            if (stop_event_)
                cudaEventDestroy(stop_event_);
        }

        // ============================================================================
        // Public API
        // ============================================================================

        CudaGemmConfig CudaGemmAutoTuner::getOptimalConfig(int m, int n, int k)
        {
            auto shape = std::make_tuple(m, n, k);

            // Check cache
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                auto it = optimal_configs_.find(shape);
                if (it != optimal_configs_.end())
                {
                    return it->second;
                }
            }

            // Run auto-tuning
            CudaGemmConfig config;
            if (auto_tuning_enabled_)
            {
                config = autoTune(m, n, k);
            }
            else
            {
                config = selectHeuristic(m, n, k);
            }

            // Cache result
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                optimal_configs_[shape] = config;
            }

            return config;
        }

        void CudaGemmAutoTuner::setOptimalConfig(int m, int n, int k, const CudaGemmConfig &config)
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            optimal_configs_[std::make_tuple(m, n, k)] = config;
        }

        void CudaGemmAutoTuner::clearCache()
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            optimal_configs_.clear();
            benchmark_history_.clear();
        }

        std::vector<CudaBenchmarkResult> CudaGemmAutoTuner::getBenchmarkResults(int m, int n, int k) const
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            auto it = benchmark_history_.find(std::make_tuple(m, n, k));
            if (it != benchmark_history_.end())
            {
                return it->second;
            }
            return {};
        }

        void CudaGemmAutoTuner::setAutoTuningEnabled(bool enabled)
        {
            auto_tuning_enabled_ = enabled;
        }

        void CudaGemmAutoTuner::setMaxCandidates(int max_candidates)
        {
            max_candidates_ = max_candidates;
        }

        void CudaGemmAutoTuner::setBenchmarkIterations(int iterations)
        {
            benchmark_iterations_ = iterations;
        }

        void CudaGemmAutoTuner::setWarmupIterations(int iterations)
        {
            warmup_iterations_ = iterations;
        }

        // ============================================================================
        // Auto-Tuning Pipeline
        // ============================================================================

        CudaGemmConfig CudaGemmAutoTuner::autoTune(int m, int n, int k)
        {
            LOG_INFO("[CUDA AutoTuner] Auto-tuning for shape (" << m << ", " << n << ", " << k << ")");

            // Generate all candidates
            auto candidates = generateCandidates();
            LOG_INFO("[CUDA AutoTuner] Generated " << candidates.size() << " candidate configurations");

            // Filter by problem size
            candidates = filterByProblemSize(candidates, m, n, k);
            LOG_INFO("[CUDA AutoTuner] After problem-size filtering: " << candidates.size() << " candidates");

            // Filter by GPU resources
            candidates = filterByResources(candidates);
            LOG_INFO("[CUDA AutoTuner] After resource filtering: " << candidates.size() << " candidates");

            // Rank by analytical model
            candidates = rankByPerformanceModel(candidates, m, n, k);

            // Select top candidates for benchmarking
            candidates = selectTopCandidates(candidates, max_candidates_);
            LOG_INFO("[CUDA AutoTuner] Benchmarking top " << candidates.size() << " candidates");

            // Benchmark each candidate
            std::vector<CudaBenchmarkResult> results;
            for (const auto &config : candidates)
            {
                auto result = benchmarkConfig(config, m, n, k);
                results.push_back(result);
                LOG_DEBUG("[CUDA AutoTuner]   " << config.id() << ": "
                                                << result.gflops << " GFLOPS (" << result.time_ms << " ms)");
            }

            // Sort by performance
            std::sort(results.begin(), results.end());

            // Cache benchmark results
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                benchmark_history_[std::make_tuple(m, n, k)] = results;
            }

            // Return best configuration
            LOG_INFO("[CUDA AutoTuner] Best config: " << results[0].config.id()
                                                      << " (" << results[0].gflops << " GFLOPS)");
            return results[0].config;
        }

        // ============================================================================
        // Candidate Generation
        // ============================================================================

        std::vector<CudaGemmConfig> CudaGemmAutoTuner::generateCandidates()
        {
            std::vector<CudaGemmConfig> candidates;

            // CRITICAL: These parameters must EXACTLY match generate_cuda_gemm_variants.py
            // Otherwise the auto-tuner will select configurations that don't have compiled kernels

            // Tile dimensions (shared memory constraints apply)
            const std::vector<int> tile_m_values = {16, 32, 64};
            const std::vector<int> tile_n_values = {16, 32, 64};
            const std::vector<int> tile_k_values = {32}; // Fixed for IQ4_NL block size

            // Thread block configurations
            const std::vector<int> threads_m_values = {8, 16};
            const std::vector<int> threads_n_values = {8, 16};

            // Work per thread (register pressure)
            const std::vector<int> work_m_values = {1, 2, 4, 8};
            const std::vector<int> work_n_values = {1, 2, 4, 8};

            // Prefetching stages (0 = no prefetch, 1-2 = double/triple buffering)
            const std::vector<int> prefetch_values = {0, 1, 2};

            // Shared memory optimizations
            const std::vector<bool> transpose_values = {false, true};

            // Vectorized loads (1 = scalar, 2/4 = vector)
            const std::vector<int> vectorize_values = {1, 2, 4};

            // NEW: Tensor Core atom configuration
            // atom_type: 0 = SM80_16x8x16 (K=16, more K per iteration, better for medium/large K)
            //            1 = SM80_16x8x8  (K=8, smaller footprint, may help small matrices)
            const std::vector<int> atom_type_values = {0, 1}; // Both atom types for diversity

            // Atom layout: how many atoms to tile together
            // Layout 2×2×1 = 4 atoms → 32×16 output tile (was hardcoded before, good baseline)
            // Layout 1×1×1 = 1 atom → 16×8 output tile (smaller for tiny matrices)
            // Layout 4×4×1 = 16 atoms → 64×32 output tile (larger for big matrices)
            //
            // PRACTICAL SUBSET: Using 3 representative layouts to keep config count manageable
            // Full space would be 648 × 2 atom_types × 9 layouts = 11,664 configs (too many)
            // Current space: 648 × 2 atom_types × 3 layouts = 3,888 configs (good balance)
            const std::vector<int> atom_layout_m_values = {1, 2, 4}; // Representative: small, medium, large
            const std::vector<int> atom_layout_n_values = {1, 2, 4}; // Representative: small, medium, large
            const std::vector<int> atom_layout_k_values = {1};       // Always 1 for SM80

            // Explode parameter space (must match Python generator's nested loop order if used)
            for (int atom_type : atom_type_values)
            {
                for (int atom_m : atom_layout_m_values)
                {
                    for (int atom_n : atom_layout_n_values)
                    {
                        for (int atom_k : atom_layout_k_values)
                        {
                            // FILTER: Only use square/balanced atom layouts for practical subset
                            // 1×1×1 (tiny), 2×2×1 (medium), 4×4×1 (large)
                            // Skip asymmetric layouts like 1×2, 2×4, etc. to keep config count manageable
                            if (atom_m != atom_n)
                            {
                                continue; // Skip non-square layouts
                            }

                            for (int tm : tile_m_values)
                            {
                                for (int tn : tile_n_values)
                                {
                                    for (int tk : tile_k_values)
                                    {
                                        for (int thm : threads_m_values)
                                        {
                                            for (int thn : threads_n_values)
                                            {
                                                for (int wm : work_m_values)
                                                {
                                                    for (int wn : work_n_values)
                                                    {
                                                        for (int prefetch : prefetch_values)
                                                        {
                                                            for (bool transpose : transpose_values)
                                                            {
                                                                for (int vectorize : vectorize_values)
                                                                {
                                                                    CudaGemmConfig config{
                                                                        .tile_m = tm,
                                                                        .tile_n = tn,
                                                                        .tile_k = tk,
                                                                        .threads_m = thm,
                                                                        .threads_n = thn,
                                                                        .work_per_thread_m = wm,
                                                                        .work_per_thread_n = wn,
                                                                        .prefetch_stages = prefetch,
                                                                        .transpose_smem = transpose,
                                                                        .vectorize_load = vectorize,
                                                                        .atom_type = atom_type,
                                                                        .atom_layout_m = atom_m,
                                                                        .atom_layout_n = atom_n,
                                                                        .atom_layout_k = atom_k};

                                                                    if (config.isValid())
                                                                    {
                                                                        candidates.push_back(config);
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return candidates;
        }

        std::vector<CudaGemmConfig> CudaGemmAutoTuner::getAvailableConfigs()
        {
            return generateCandidates();
        }

        // ============================================================================
        // Filtering and Ranking
        // ============================================================================

        std::vector<CudaGemmConfig> CudaGemmAutoTuner::filterByProblemSize(
            const std::vector<CudaGemmConfig> &candidates,
            int m, int n, int k)
        {
            std::vector<CudaGemmConfig> filtered;

            for (const auto &config : candidates)
            {
                // Tile too large for matrix
                if (config.tile_m > m * 2 || config.tile_n > n * 2)
                    continue;

                // Tile too small for large matrix (inefficient)
                if (m >= 512 && config.tile_m < 32)
                    continue;
                if (n >= 512 && config.tile_n < 32)
                    continue;

                // Prefer tiles that evenly divide matrix dimensions
                bool good_m_fit = (m % config.tile_m == 0) || (m / config.tile_m >= 4);
                bool good_n_fit = (n % config.tile_n == 0) || (n / config.tile_n >= 4);

                if (good_m_fit && good_n_fit)
                {
                    filtered.push_back(config);
                }
            }

            // If filter too aggressive, keep all valid candidates
            return filtered.empty() ? candidates : filtered;
        }

        std::vector<CudaGemmConfig> CudaGemmAutoTuner::filterByResources(
            const std::vector<CudaGemmConfig> &candidates)
        {
            std::vector<CudaGemmConfig> filtered;

            for (const auto &config : candidates)
            {
                // Check occupancy
                float occupancy = config.estimateOccupancy();
                if (occupancy < 0.25f)
                    continue; // Too low

                // Check register pressure
                int reg_pressure = config.estimateRegisterPressure();
                if (reg_pressure > 64)
                    continue; // Too high (reduces occupancy)

                // Check shared memory
                int smem_bytes = (config.tile_m * config.tile_k + config.tile_n * config.tile_k) *
                                 (1 + config.prefetch_stages) * sizeof(float);
                if (smem_bytes > static_cast<int>(device_props_.sharedMemPerBlock))
                    continue;

                filtered.push_back(config);
            }

            return filtered;
        }

        std::vector<CudaGemmConfig> CudaGemmAutoTuner::rankByPerformanceModel(
            const std::vector<CudaGemmConfig> &candidates,
            int m, int n, int k)
        {
            struct ScoredConfig
            {
                CudaGemmConfig config;
                double score;

                bool operator<(const ScoredConfig &other) const
                {
                    return score > other.score; // Higher score is better
                }
            };

            std::vector<ScoredConfig> scored;

            // Check which heuristic to use (priority: NN > ML > Manual)
            bool use_nn = useNNHeuristic();
            bool use_ml = useMLHeuristic();

            // Log which heuristic is active (only once)
            static bool logged = false;
            if (!logged)
            {
                if (use_nn)
                {
                    LOG_INFO("[CUDA AutoTuner] Using ONNX neural network heuristic (LLAMINAR_USE_NN_HEURISTIC=1)");
                }
                else if (use_ml)
                {
                    LOG_INFO("[CUDA AutoTuner] Using ML-learned heuristic (LLAMINAR_USE_ML_HEURISTIC=1)");
                }
                else
                {
                    LOG_INFO("[CUDA AutoTuner] Using manual heuristic (set LLAMINAR_USE_ML_HEURISTIC=1 for ML or LLAMINAR_USE_NN_HEURISTIC=1 for NN)");
                }
                logged = true;
            }

            // Track lookup statistics
            static int total_lookups = 0;
            static int lookup_hits = 0;
            static int lookup_misses = 0;

            for (const auto &config : candidates)
            {
                double score = 0.0;

                if (use_nn)
                {
                    // ONNX neural network heuristic: Works on ANY shape (no lookup table needed)
                    // Trained on 0.5B/7B/72B, validated on 4B/32B, tested on 14B/235B/671B
                    // Test R²=0.9569 (95.69% accuracy on unseen model sizes)
#ifdef HAVE_ONNX_RUNTIME
                    auto &nn = CudaGemmNeuralNetwork::instance();
                    if (nn.isInitialized())
                    {
                        score = nn.predict(config, m, n, k); // Predicted GFLOPS (higher is better)
                    }
                    else
                    {
                        // NN not initialized - fallback to manual heuristic
                        LOG_WARN("[CUDA AutoTuner] ONNX neural network not initialized, using manual heuristic");
                        score = manualHeuristicScore(config, m, n, k);
                    }
#else
                    // ONNX Runtime not available - fallback to manual heuristic
                    static bool warned_once = false;
                    if (!warned_once)
                    {
                        LOG_WARN("[CUDA AutoTuner] ONNX Runtime not available (rebuild with -DUSE_ONNX_HEURISTIC=ON), using manual heuristic");
                        warned_once = true;
                    }
                    score = manualHeuristicScore(config, m, n, k);
#endif
                }
                else if (use_ml)
                {
                    // ML-learned heuristic: Use Gradient Boosting predictions from lookup table
                    // This gives us exact GB accuracy (R²=0.9999) without porting decision trees
                    using llaminar::cuda::gb_predictions;

                    uint64_t config_hash = config.hash(m, n, k); // Include problem size in hash
                    total_lookups++;

                    // Lookup GB prediction
                    auto it = gb_predictions.find(config_hash);
                    if (it != gb_predictions.end())
                    {
                        // Found in lookup table - use GB prediction directly
                        score = it->second; // Predicted GFLOPS (higher is better)
                        lookup_hits++;
                    }
                    else
                    {
                        // Not in lookup table - fallback to manual heuristic
                        // (This should rarely happen since we benchmarked all generated configs)
                        lookup_misses++;

                        // Log first few misses for debugging
                        static int logged_misses = 0;
                        if (logged_misses < 5)
                        {
                            LOG_INFO("[CUDA AutoTuner] Lookup MISS for m=" << m << " n=" << n << " k=" << k
                                                                           << " hash=0x" << std::hex << config_hash << std::dec
                                                                           << " config=" << config.id());
                            logged_misses++;
                        }

                        score = manualHeuristicScore(config, m, n, k);
                    }
                }
                else
                {
                    // Manual heuristic (kept for comparison)
                    score = manualHeuristicScore(config, m, n, k);
                }

                scored.push_back({config, score});
            }

            // Log lookup statistics periodically
            if (use_ml && total_lookups > 0)
            {
                static int last_logged = 0;
                if (total_lookups - last_logged >= 500)
                {
                    float hit_rate = 100.0f * lookup_hits / total_lookups;
                    LOG_INFO("[CUDA AutoTuner] Lookup stats: " << lookup_hits << "/" << total_lookups
                                                               << " hits (" << hit_rate << "%), " << lookup_misses << " misses");
                    last_logged = total_lookups;
                }
            }

            // Sort by score (higher is better)
            std::sort(scored.begin(), scored.end());

            // Log top 5 scored configs for debugging
            if (use_ml && scored.size() > 0)
            {
                LOG_INFO("[CUDA AutoTuner] Top 5 scored configs for m=" << m << " n=" << n << " k=" << k << ":");
                for (size_t i = 0; i < std::min(size_t(5), scored.size()); i++)
                {
                    LOG_INFO("  #" << (i + 1) << ": " << scored[i].config.id() << " (score=" << scored[i].score << ")");
                }
            }

            // Extract sorted configs
            std::vector<CudaGemmConfig> ranked;
            ranked.reserve(candidates.size());
            for (const auto &sc : scored)
            {
                ranked.push_back(sc.config);
            }

            return ranked;
        }

        double CudaGemmAutoTuner::manualHeuristicScore(
            const CudaGemmConfig &config,
            int m, int n, int k)
        {
            double score = 0.0;

            // Original manual heuristic (DEPRECATED - has -12,000 correlation!)
            // Kept for comparison only

            // Occupancy (0.0 to 1.0, weight: 30%)
            score += config.estimateOccupancy() * 30.0;

            // Arithmetic intensity (FLOPs per byte loaded)
            // Each output element: 2*k FLOPs, loads: k*sizeof(A) + k*sizeof(B_encoded)
            double flops_per_elem = 2.0 * k;
            double bytes_per_elem = k * sizeof(float) + (k / 32.0) * sizeof(IQ4_NLBlock);
            double arithmetic_intensity = flops_per_elem / bytes_per_elem;
            score += std::min(arithmetic_intensity / 10.0, 10.0) * 20.0; // Cap at 10, weight: 20%

            // Tile size efficiency (larger tiles amortize overhead, weight: 20%)
            double tile_efficiency = static_cast<double>(config.tile_m * config.tile_n) / 16384.0; // Normalize to 128x128
            score += std::min(tile_efficiency, 1.0) * 20.0;

            // Work per thread (higher = better register reuse, weight: 15%)
            double work_efficiency = static_cast<double>(config.work_per_thread_m * config.work_per_thread_n) / 64.0; // Normalize to 8x8
            score += std::min(work_efficiency, 1.0) * 15.0;

            // Prefetch bonus (reduces memory stalls, weight: 10%)
            score += config.prefetch_stages * 5.0;

            // Transpose bonus (reduces bank conflicts, weight: 5%)
            if (config.transpose_smem)
                score += 5.0;

            return score;
        }

        // ============================================================================
        // Candidate Selection
        // ============================================================================

        std::vector<CudaGemmConfig> CudaGemmAutoTuner::selectTopCandidates(
            const std::vector<CudaGemmConfig> &ranked,
            int max_candidates)
        {
            std::vector<CudaGemmConfig> top;
            int n = std::min(static_cast<int>(ranked.size()), max_candidates);
            for (int i = 0; i < n; ++i)
            {
                top.push_back(ranked[i]);
            }
            return top;
        }

        // ============================================================================
        // Heuristic Selection
        // ============================================================================

        CudaGemmConfig CudaGemmAutoTuner::selectHeuristic(int m, int n, int k)
        {
            // Phase 3 (Nov 2025): Use ML-based tile predictor by default
            // Environment flag to fallback to old heuristic: LLAMINAR_USE_OLD_HEURISTIC=1
            const char *use_old = std::getenv("LLAMINAR_USE_OLD_HEURISTIC");
            bool use_old_heuristic = use_old && std::atoi(use_old) != 0;

            if (!use_old_heuristic)
            {
                // ML-based predictor (Phase 3)
                auto ml_tile_config = GemmAutoTunerML::predict(m, n, k);

                CudaGemmConfig config;
                config.tile_m = ml_tile_config.tile_m;
                config.tile_n = ml_tile_config.tile_n;
                config.tile_k = ml_tile_config.tile_k;

                // Derive other parameters from tile sizes
                // Use 8x8 thread block (64 threads) as default
                config.threads_m = 8;
                config.threads_n = 8;

                // Work per thread based on tile size
                config.work_per_thread_m = ml_tile_config.tile_m / config.threads_m;
                config.work_per_thread_n = ml_tile_config.tile_n / config.threads_n;

                // No prefetching by default (Phase 2.7 showed minimal gains)
                config.prefetch_stages = 0;

                // No shared memory transpose by default
                config.transpose_smem = false;

                // Vectorize load based on tile_n
                // Wider tiles (64) can use wider vectors
                if (ml_tile_config.tile_n >= 64)
                {
                    config.vectorize_load = 4;
                }
                else if (ml_tile_config.tile_n >= 32)
                {
                    config.vectorize_load = 2;
                }
                else
                {
                    config.vectorize_load = 1;
                }

                LOG_DEBUG("[CUDA AutoTuner] ML predictor selected: " << GemmAutoTunerML::getConfigName(ml_tile_config)
                                                                     << " for shape (" << m << ", " << n << ", " << k << ")");

                return config;
            }

            // Fallback to old size-based heuristic
            LOG_DEBUG("[CUDA AutoTuner] Using old heuristic (LLAMINAR_USE_OLD_HEURISTIC=1)");
            if (m < 128 || n < 128)
            {
                return presets::small();
            }
            else if (m < 512 || n < 512)
            {
                return presets::medium();
            }
            else if (m > n * 2)
            {
                return presets::tall();
            }
            else if (n > m * 2)
            {
                return presets::wide();
            }
            else
            {
                return presets::large();
            }
        }

        // ============================================================================
        // Benchmarking
        // ============================================================================

        CudaBenchmarkResult CudaGemmAutoTuner::benchmarkConfig(
            const CudaGemmConfig &config,
            int m, int n, int k)
        {
            // Allocate test data if needed
            if (m > allocated_m_ || n > allocated_n_ || k > allocated_k_)
            {
                allocateTestData(m, n, k);
            }

            CudaBenchmarkResult result;
            result.config = config;

            // Warmup iterations
            for (int i = 0; i < warmup_iterations_; ++i)
            {
                auto err = launchIQ4NLGemmVariant(
                    test_A_device_, test_B_device_, test_C_device_,
                    m, n, k, config, benchmark_stream_);
                if (err != cudaSuccess)
                {
                    LOG_ERROR("[CUDA AutoTuner] Warmup launch failed: " << cudaGetErrorString(err));
                    result.gflops = 0.0;
                    result.time_ms = 1e9;
                    return result;
                }
            }
            cudaStreamSynchronize(benchmark_stream_);

            // Timed iterations
            cudaEventRecord(start_event_, benchmark_stream_);

            for (int i = 0; i < benchmark_iterations_; ++i)
            {
                auto err = launchIQ4NLGemmVariant(
                    test_A_device_, test_B_device_, test_C_device_,
                    m, n, k, config, benchmark_stream_);
                if (err != cudaSuccess)
                {
                    LOG_ERROR("[CUDA AutoTuner] Benchmark launch failed: " << cudaGetErrorString(err));
                    result.gflops = 0.0;
                    result.time_ms = 1e9;
                    return result;
                }
            }

            cudaEventRecord(stop_event_, benchmark_stream_);
            cudaEventSynchronize(stop_event_);

            // Compute metrics
            float elapsed_ms;
            cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_);

            result.time_ms = elapsed_ms / benchmark_iterations_;
            result.iterations = benchmark_iterations_;

            // GFLOPS: 2*m*n*k FLOPs per GEMM
            double flops = 2.0 * static_cast<double>(m) * n * k;
            result.gflops = (flops / 1e9) / (result.time_ms / 1000.0);

            result.occupancy = config.estimateOccupancy();
            result.shared_memory_bytes = (config.tile_m * config.tile_k + config.tile_n * config.tile_k) *
                                         (1 + config.prefetch_stages) * sizeof(float);

            return result;
        }

        // ============================================================================
        // Memory Management
        // ============================================================================

        void CudaGemmAutoTuner::allocateTestData(int m, int n, int k)
        {
            // Free old allocation
            freeTestData();

            // Allocate A matrix (FP32)
            size_t A_size = static_cast<size_t>(m) * k * sizeof(float);
            cudaMalloc(&test_A_device_, A_size);

            // Allocate B matrix (IQ4_NL blocks)
            int num_blocks = n * (k / 32);
            size_t B_size = static_cast<size_t>(num_blocks) * sizeof(IQ4_NLBlock);
            cudaMalloc(&test_B_device_, B_size);

            // Allocate C matrix (FP32)
            size_t C_size = static_cast<size_t>(m) * n * sizeof(float);
            cudaMalloc(&test_C_device_, C_size);

            // Initialize with random data on host, then copy to device
            std::vector<float> A_host(m * k);
            std::vector<IQ4_NLBlock> B_host(num_blocks);

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

            for (auto &val : A_host)
                val = dist(gen);

            // Create random IQ4_NL blocks
            for (auto &block : B_host)
            {
                block.d = fp32_to_fp16(dist(gen) * 0.1f); // Small scale
                for (int i = 0; i < 16; ++i)
                {
                    block.qs[i] = static_cast<uint8_t>(gen() & 0xFF);
                }
            }

            cudaMemcpy(test_A_device_, A_host.data(), A_size, cudaMemcpyHostToDevice);
            cudaMemcpy(test_B_device_, B_host.data(), B_size, cudaMemcpyHostToDevice);

            allocated_m_ = m;
            allocated_n_ = n;
            allocated_k_ = k;

            LOG_DEBUG("[CUDA AutoTuner] Allocated test data: " << m << "x" << n << "x" << k);
        }

        void CudaGemmAutoTuner::freeTestData()
        {
            if (test_A_device_)
            {
                cudaFree(test_A_device_);
                test_A_device_ = nullptr;
            }
            if (test_B_device_)
            {
                cudaFree(test_B_device_);
                test_B_device_ = nullptr;
            }
            if (test_C_device_)
            {
                cudaFree(test_C_device_);
                test_C_device_ = nullptr;
            }
            allocated_m_ = allocated_n_ = allocated_k_ = 0;
        }

    } // namespace cuda
} // namespace llaminar2
