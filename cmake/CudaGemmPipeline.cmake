# CUDA GEMM AutoTuner Training Pipeline
#
# This CMake module defines targets for the complete training pipeline:
# 1. Profile kernel configurations on 0.5B, 4B, 7B Qwen models
# 2. Select top/bottom 10 configs per shape for diversity
# 3. Train neural network with collected data
# 4. Validate on unseen shapes
# 5. Replace ONNX model and scaler files
# 6. Rebuild CUDA backend to use new model
#
# Usage:
#   cmake --build build_v2_release --target cuda_gemm_retrain_pipeline
#
# Or individual stages:
#   cmake --build build_v2_release --target cuda_gemm_profile
#   cmake --build build_v2_release --target cuda_gemm_train
#   cmake --build build_v2_release --target cuda_gemm_validate
#   cmake --build build_v2_release --target cuda_gemm_deploy

# ==============================================================================
# Configuration
# ==============================================================================

set(CUDA_GEMM_PIPELINE_DIR "${CMAKE_SOURCE_DIR}/../../python")
set(CUDA_GEMM_PROFILING_SCRIPT "${CUDA_GEMM_PIPELINE_DIR}/collect_profiling_data.py")
set(CUDA_GEMM_TRAINING_SCRIPT "${CUDA_GEMM_PIPELINE_DIR}/train_cuda_neural_network.py")
set(CUDA_GEMM_VALIDATION_SCRIPT "${CUDA_GEMM_PIPELINE_DIR}/validate_heuristic.py")
set(CUDA_GEMM_SCALER_EXPORT_SCRIPT "${CUDA_GEMM_PIPELINE_DIR}/export_scaler_for_cpp.py")

set(CUDA_GEMM_MODEL_DIR "${CMAKE_SOURCE_DIR}/kernels/cuda")
set(CUDA_GEMM_ONNX_MODEL "${CUDA_GEMM_MODEL_DIR}/cuda_heuristic_nn.onnx")
set(CUDA_GEMM_SCALER_FILE "${CUDA_GEMM_MODEL_DIR}/cuda_heuristic_scaler.txt")

set(CUDA_GEMM_DATA_DIR "${CMAKE_SOURCE_DIR}/../..")
set(CUDA_GEMM_PROFILING_CSV "${CUDA_GEMM_DATA_DIR}/cuda_gemm_profiling_data.csv")
set(CUDA_GEMM_TRAINING_METRICS "${CUDA_GEMM_DATA_DIR}/training_metrics.json")
set(CUDA_GEMM_VALIDATION_RESULTS "${CUDA_GEMM_DATA_DIR}/validation_results.json")

# Model sizes to profile (Qwen models)
set(CUDA_GEMM_MODEL_SIZES "0.5B" "4B" "7B")

# Number of top/bottom configs to profile per shape
set(CUDA_GEMM_TOP_N 10)
set(CUDA_GEMM_BOTTOM_N 10)

# ==============================================================================
# Stage 1: Profiling
# ==============================================================================

add_custom_target(cuda_gemm_profile
    COMMAND ${CMAKE_COMMAND} -E echo "========================================="
    COMMAND ${CMAKE_COMMAND} -E echo "CUDA GEMM Pipeline: Profiling Stage"
    COMMAND ${CMAKE_COMMAND} -E echo "========================================="
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Model sizes: ${CUDA_GEMM_MODEL_SIZES}"
    COMMAND ${CMAKE_COMMAND} -E echo "Top configs: ${CUDA_GEMM_TOP_N}"
    COMMAND ${CMAKE_COMMAND} -E echo "Bottom configs: ${CUDA_GEMM_BOTTOM_N}"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    
    # Remove old profiling data
    COMMAND ${CMAKE_COMMAND} -E remove -f "${CUDA_GEMM_PROFILING_CSV}"
    
    # Run profiling script
    COMMAND ${CMAKE_COMMAND} -E env
        LLAMINAR_USE_NN_HEURISTIC=1
        ${Python3_EXECUTABLE} "${CUDA_GEMM_PROFILING_SCRIPT}"
        --models ${CUDA_GEMM_MODEL_SIZES}
        --top-n ${CUDA_GEMM_TOP_N}
        --bottom-n ${CUDA_GEMM_BOTTOM_N}
        --output "${CUDA_GEMM_PROFILING_CSV}"
        --verbose
    
    # Verify profiling data exists
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Profiling complete. Checking output..."
    COMMAND test -f "${CUDA_GEMM_PROFILING_CSV}" || (echo "ERROR: Profiling data not found!" && exit 1)
    
    # Show statistics
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Profiling data statistics:"
    COMMAND wc -l "${CUDA_GEMM_PROFILING_CSV}" || true
    COMMAND ${CMAKE_COMMAND} -E echo ""
    
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Profiling CUDA GEMM configurations for model sizes: ${CUDA_GEMM_MODEL_SIZES}"
    VERBATIM
)

# ==============================================================================
# Stage 2: Training
# ==============================================================================

add_custom_target(cuda_gemm_train
    COMMAND ${CMAKE_COMMAND} -E echo "========================================="
    COMMAND ${CMAKE_COMMAND} -E echo "CUDA GEMM Pipeline: Training Stage"
    COMMAND ${CMAKE_COMMAND} -E echo "========================================="
    COMMAND ${CMAKE_COMMAND} -E echo ""
    
    # Verify profiling data exists
    COMMAND test -f "${CUDA_GEMM_PROFILING_CSV}" || (echo "ERROR: Profiling data not found! Run 'cuda_gemm_profile' first." && exit 1)
    
    # Backup old model if it exists
    COMMAND ${CMAKE_COMMAND} -E echo "Backing up existing model..."
    COMMAND if [ -f "${CUDA_GEMM_ONNX_MODEL}" ]; then 
        cp "${CUDA_GEMM_ONNX_MODEL}" "${CUDA_GEMM_ONNX_MODEL}.backup.$(date +%Y%m%d_%H%M%S)";
        fi
    COMMAND if [ -f "${CUDA_GEMM_SCALER_FILE}" ]; then 
        cp "${CUDA_GEMM_SCALER_FILE}" "${CUDA_GEMM_SCALER_FILE}.backup.$(date +%Y%m%d_%H%M%S)";
        fi
    
    # Run training script
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Training neural network..."
    COMMAND ${Python3_EXECUTABLE} "${CUDA_GEMM_TRAINING_SCRIPT}"
        --input "${CUDA_GEMM_PROFILING_CSV}"
        --output-model "${CUDA_GEMM_ONNX_MODEL}.new"
        --output-scaler "${CUDA_GEMM_SCALER_FILE}.new"
        --epochs 10
        --batch-size 32
        --learning-rate 0.0001
        --hidden-dims 256 128 64
        --validation-split 0.2
        --metrics "${CUDA_GEMM_TRAINING_METRICS}"
        --verbose
    
    # Verify training outputs
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Verifying training outputs..."
    COMMAND test -f "${CUDA_GEMM_ONNX_MODEL}.new" || (echo "ERROR: ONNX model not generated!" && exit 1)
    COMMAND test -f "${CUDA_GEMM_SCALER_FILE}.new" || (echo "ERROR: Scaler file not generated!" && exit 1)
    COMMAND test -f "${CUDA_GEMM_TRAINING_METRICS}" || (echo "ERROR: Training metrics not found!" && exit 1)
    
    # Show training metrics
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Training metrics:"
    COMMAND cat "${CUDA_GEMM_TRAINING_METRICS}" || true
    COMMAND ${CMAKE_COMMAND} -E echo ""
    
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Training CUDA GEMM neural network heuristic"
    VERBATIM
)

# ==============================================================================
# Stage 3: Validation
# ==============================================================================

add_custom_target(cuda_gemm_validate
    COMMAND ${CMAKE_COMMAND} -E echo "========================================="
    COMMAND ${CMAKE_COMMAND} -E echo "CUDA GEMM Pipeline: Validation Stage"
    COMMAND ${CMAKE_COMMAND} -E echo "========================================="
    COMMAND ${CMAKE_COMMAND} -E echo ""
    
    # Verify training outputs exist
    COMMAND test -f "${CUDA_GEMM_ONNX_MODEL}.new" || (echo "ERROR: New model not found! Run 'cuda_gemm_train' first." && exit 1)
    COMMAND test -f "${CUDA_GEMM_SCALER_FILE}.new" || (echo "ERROR: New scaler not found! Run 'cuda_gemm_train' first." && exit 1)
    
    # Temporarily install new model for validation
    COMMAND ${CMAKE_COMMAND} -E echo "Installing new model for validation..."
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_GEMM_ONNX_MODEL}.new" "${CUDA_GEMM_ONNX_MODEL}.validation"
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_GEMM_SCALER_FILE}.new" "${CUDA_GEMM_SCALER_FILE}.validation"
    
    # Run validation script
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Validating neural network on unseen shapes..."
    COMMAND ${CMAKE_COMMAND} -E env
        LLAMINAR_CUDA_HEURISTIC_MODEL="${CUDA_GEMM_ONNX_MODEL}.validation"
        LLAMINAR_CUDA_HEURISTIC_SCALER="${CUDA_GEMM_SCALER_FILE}.validation"
        ${Python3_EXECUTABLE} "${CUDA_GEMM_VALIDATION_SCRIPT}"
        --model "${CUDA_GEMM_ONNX_MODEL}.validation"
        --scaler "${CUDA_GEMM_SCALER_FILE}.validation"
        --output "${CUDA_GEMM_VALIDATION_RESULTS}"
        --verbose
    
    # Show validation results
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Validation results:"
    COMMAND cat "${CUDA_GEMM_VALIDATION_RESULTS}" || true
    COMMAND ${CMAKE_COMMAND} -E echo ""
    
    # Check validation success
    COMMAND ${CMAKE_COMMAND} -E echo "Checking validation success criteria..."
    COMMAND ${Python3_EXECUTABLE} -c "
import json
import sys
with open('${CUDA_GEMM_VALIDATION_RESULTS}') as f:
    results = json.load(f)
    hit_rate = results.get('top_30_hit_rate', 0)
    print(f'Top-30 hit rate: {hit_rate*100:.1f}%')
    if hit_rate < 0.95:
        print('ERROR: Hit rate below 95% threshold!')
        sys.exit(1)
    print('✓ Validation passed!')
"
    
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Validating CUDA GEMM neural network on unseen shapes"
    VERBATIM
)

# ==============================================================================
# Stage 4: Deployment
# ==============================================================================

add_custom_target(cuda_gemm_deploy
    COMMAND ${CMAKE_COMMAND} -E echo "========================================="
    COMMAND ${CMAKE_COMMAND} -E echo "CUDA GEMM Pipeline: Deployment Stage"
    COMMAND ${CMAKE_COMMAND} -E echo "========================================="
    COMMAND ${CMAKE_COMMAND} -E echo ""
    
    # Verify new model exists and validated
    COMMAND test -f "${CUDA_GEMM_ONNX_MODEL}.new" || (echo "ERROR: New model not found! Run 'cuda_gemm_train' first." && exit 1)
    COMMAND test -f "${CUDA_GEMM_VALIDATION_RESULTS}" || (echo "ERROR: Validation results not found! Run 'cuda_gemm_validate' first." && exit 1)
    
    # Replace production model
    COMMAND ${CMAKE_COMMAND} -E echo "Deploying new model to production..."
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_GEMM_ONNX_MODEL}.new" "${CUDA_GEMM_ONNX_MODEL}"
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_GEMM_SCALER_FILE}.new" "${CUDA_GEMM_SCALER_FILE}"
    
    # Clean up temporary files
    COMMAND ${CMAKE_COMMAND} -E remove -f "${CUDA_GEMM_ONNX_MODEL}.new"
    COMMAND ${CMAKE_COMMAND} -E remove -f "${CUDA_GEMM_SCALER_FILE}.new"
    COMMAND ${CMAKE_COMMAND} -E remove -f "${CUDA_GEMM_ONNX_MODEL}.validation"
    COMMAND ${CMAKE_COMMAND} -E remove -f "${CUDA_GEMM_SCALER_FILE}.validation"
    
    # Rebuild CUDA backend
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Rebuilding CUDA backend with new model..."
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target cuda_backend --parallel
    
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "========================================="
    COMMAND ${CMAKE_COMMAND} -E echo "✓ Deployment complete!"
    COMMAND ${CMAKE_COMMAND} -E echo "========================================="
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "New model deployed: ${CUDA_GEMM_ONNX_MODEL}"
    COMMAND ${CMAKE_COMMAND} -E echo "New scaler deployed: ${CUDA_GEMM_SCALER_FILE}"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Deploying new CUDA GEMM model and rebuilding backend"
    VERBATIM
)

# ==============================================================================
# Complete Pipeline
# ==============================================================================

add_custom_target(cuda_gemm_retrain_pipeline
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "========================================="
    COMMAND ${CMAKE_COMMAND} -E echo "CUDA GEMM AutoTuner Retraining Pipeline"
    COMMAND ${CMAKE_COMMAND} -E echo "========================================="
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "This pipeline will:"
    COMMAND ${CMAKE_COMMAND} -E echo "  1. Profile kernel configs on 0.5B, 4B, 7B Qwen models"
    COMMAND ${CMAKE_COMMAND} -E echo "  2. Select top/bottom ${CUDA_GEMM_TOP_N} configs per shape"
    COMMAND ${CMAKE_COMMAND} -E echo "  3. Train neural network heuristic"
    COMMAND ${CMAKE_COMMAND} -E echo "  4. Validate on unseen shapes (≥95% hit rate)"
    COMMAND ${CMAKE_COMMAND} -E echo "  5. Deploy new model and rebuild CUDA backend"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Estimated time: 10-20 minutes"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Press Ctrl+C to abort, or wait 5 seconds to continue..."
    COMMAND ${CMAKE_COMMAND} -E sleep 5
    COMMAND ${CMAKE_COMMAND} -E echo ""
    
    # Run all stages in sequence
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target cuda_gemm_profile
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target cuda_gemm_train
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target cuda_gemm_validate
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target cuda_gemm_deploy
    
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "========================================="
    COMMAND ${CMAKE_COMMAND} -E echo "✓ Pipeline complete!"
    COMMAND ${CMAKE_COMMAND} -E echo "========================================="
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Summary:"
    COMMAND ${CMAKE_COMMAND} -E echo "  - Profiling data: ${CUDA_GEMM_PROFILING_CSV}"
    COMMAND ${CMAKE_COMMAND} -E echo "  - Training metrics: ${CUDA_GEMM_TRAINING_METRICS}"
    COMMAND ${CMAKE_COMMAND} -E echo "  - Validation results: ${CUDA_GEMM_VALIDATION_RESULTS}"
    COMMAND ${CMAKE_COMMAND} -E echo "  - Production model: ${CUDA_GEMM_ONNX_MODEL}"
    COMMAND ${CMAKE_COMMAND} -E echo "  - Production scaler: ${CUDA_GEMM_SCALER_FILE}"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "CUDA backend rebuilt and ready to use new model!"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running complete CUDA GEMM retraining pipeline"
    VERBATIM
)

# ==============================================================================
# Helper Targets
# ==============================================================================

# Clean all pipeline artifacts
add_custom_target(cuda_gemm_clean_pipeline
    COMMAND ${CMAKE_COMMAND} -E echo "Cleaning CUDA GEMM pipeline artifacts..."
    COMMAND ${CMAKE_COMMAND} -E remove -f "${CUDA_GEMM_PROFILING_CSV}"
    COMMAND ${CMAKE_COMMAND} -E remove -f "${CUDA_GEMM_TRAINING_METRICS}"
    COMMAND ${CMAKE_COMMAND} -E remove -f "${CUDA_GEMM_VALIDATION_RESULTS}"
    COMMAND ${CMAKE_COMMAND} -E remove -f "${CUDA_GEMM_ONNX_MODEL}.new"
    COMMAND ${CMAKE_COMMAND} -E remove -f "${CUDA_GEMM_SCALER_FILE}.new"
    COMMAND ${CMAKE_COMMAND} -E remove -f "${CUDA_GEMM_ONNX_MODEL}.validation"
    COMMAND ${CMAKE_COMMAND} -E remove -f "${CUDA_GEMM_SCALER_FILE}.validation"
    COMMAND ${CMAKE_COMMAND} -E remove -f "${CUDA_GEMM_ONNX_MODEL}.backup.*"
    COMMAND ${CMAKE_COMMAND} -E remove -f "${CUDA_GEMM_SCALER_FILE}.backup.*"
    COMMAND ${CMAKE_COMMAND} -E echo "✓ Cleanup complete"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Cleaning CUDA GEMM pipeline artifacts"
    VERBATIM
)

# Show pipeline status
add_custom_target(cuda_gemm_pipeline_status
    COMMAND ${CMAKE_COMMAND} -E echo "========================================="
    COMMAND ${CMAKE_COMMAND} -E echo "CUDA GEMM Pipeline Status"
    COMMAND ${CMAKE_COMMAND} -E echo "========================================="
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Production Files:"
    COMMAND test -f "${CUDA_GEMM_ONNX_MODEL}" && ls -lh "${CUDA_GEMM_ONNX_MODEL}" || ${CMAKE_COMMAND} -E echo "  Model: Not found"
    COMMAND test -f "${CUDA_GEMM_SCALER_FILE}" && ls -lh "${CUDA_GEMM_SCALER_FILE}" || ${CMAKE_COMMAND} -E echo "  Scaler: Not found"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Pipeline Artifacts:"
    COMMAND test -f "${CUDA_GEMM_PROFILING_CSV}" && ls -lh "${CUDA_GEMM_PROFILING_CSV}" || ${CMAKE_COMMAND} -E echo "  Profiling data: Not found"
    COMMAND test -f "${CUDA_GEMM_TRAINING_METRICS}" && ls -lh "${CUDA_GEMM_TRAINING_METRICS}" || ${CMAKE_COMMAND} -E echo "  Training metrics: Not found"
    COMMAND test -f "${CUDA_GEMM_VALIDATION_RESULTS}" && ls -lh "${CUDA_GEMM_VALIDATION_RESULTS}" || ${CMAKE_COMMAND} -E echo "  Validation results: Not found"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Showing CUDA GEMM pipeline status"
)

message(STATUS "CUDA GEMM Pipeline configured:")
message(STATUS "  - Profile: cmake --build build --target cuda_gemm_profile")
message(STATUS "  - Train: cmake --build build --target cuda_gemm_train")
message(STATUS "  - Validate: cmake --build build --target cuda_gemm_validate")
message(STATUS "  - Deploy: cmake --build build --target cuda_gemm_deploy")
message(STATUS "  - Full pipeline: cmake --build build --target cuda_gemm_retrain_pipeline")
message(STATUS "  - Clean: cmake --build build --target cuda_gemm_clean_pipeline")
message(STATUS "  - Status: cmake --build build --target cuda_gemm_pipeline_status")
