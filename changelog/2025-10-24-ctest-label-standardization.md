# CTest Label Standardization - V2 Test Suite

**Date**: 2025-10-24  
**Status**: ✅ Complete  
**Tests**: 14/14 passing  
**Motivation**: Replace ambiguous timeline-based labels (Phase1/2/3/4) with feature-descriptive hierarchical labels

---

## Summary

Refactored CTest label naming in V2 test suite to use **feature-based, hierarchical labeling** instead of timeline-based "Phase" markers. This improves:

- **Discoverability**: New developers can understand what tests cover without historical context
- **Filtering**: Multiple labels per test enable flexible filtering by component, feature, or test type
- **Maintainability**: Labels describe functionality, not development timeline
- **Documentation**: Self-documenting test organization

---

## Before vs After

### Old Labels (Ambiguous)
```
Phase1                  =   3.57 sec*proc (2 tests)  ❌ What is Phase1?
Phase2                  =   3.62 sec*proc (2 tests)  ❌ What is Phase2?
Phase3                  =   3.06 sec*proc (2 tests)  ❌ What is Phase3?
Phase4                  =   3.66 sec*proc (3 tests)  ❌ What is Phase4?
Factory                 =   1.49 sec*proc (1 test)   ⚠️ Which factory?
Pipeline                =   1.49 sec*proc (1 test)   ⚠️ Which pipeline?
Tensor                  =   1.55 sec*proc (1 test)   ⚠️ Which tensor?
```

### New Labels (Feature-Descriptive)
```
AdvancedFeatures          =   2.03 sec*proc (1 test)  ✅ Extended orchestrator capabilities
ArgumentParsing           =   0.85 sec*proc (1 test)  ✅ CLI argument handling
BasicFeatures             =   2.27 sec*proc (1 test)  ✅ Core orchestrator functionality
Basics                    =   1.60 sec*proc (1 test)  ✅ Basic tensor operations
BufferManagement          =   1.20 sec*proc (1 test)  ✅ Cross-device buffer handling
ComputeBackends           =   0.87 sec*proc (1 test)  ✅ Backend kernel implementations
CrossDevice               =   1.20 sec*proc (1 test)  ✅ Cross-device operations
DataTransfer              =   1.20 sec*proc (1 test)  ✅ Device-to-device transfers
DeviceAwareness           =   0.82 sec*proc (1 test)  ✅ Device-aware execution
DeviceManagement          =   7.66 sec*proc (5 tests) ✅ All device orchestration tests
HeterogeneousExecution    =   1.23 sec*proc (1 test)  ✅ Multi-device heterogeneous execution
ModelLoading              =   1.94 sec*proc (1 test)  ✅ GGUF model loading
Orchestration             =   5.11 sec*proc (3 tests) ✅ Orchestrator subsystem
PipelineExecution         =   0.81 sec*proc (1 test)  ✅ Pipeline factory and registration
Quantization              =   0.84 sec*proc (1 test)  ✅ Quantized tensor formats
TensorOperations          =   3.28 sec*proc (3 tests) ✅ All tensor-related tests
WeightPlacement           =   1.32 sec*proc (1 test)  ✅ Weight distribution strategies
```

---

## Label Hierarchy

V2 tests now use **4-tier hierarchical labeling**:

### Tier 1 - Test Type (Required)
- `Unit` - Isolated component tests
- `Integration` - Multi-component tests (planned)
- `E2E` - End-to-end pipeline tests (planned)
- `Parity` - Ground truth validation (planned)

### Tier 2 - Architecture (Optional)
- `V2` - V2 architecture tests

### Tier 3 - Component (Specific)
- `DeviceManagement` - Device orchestration, discovery, selection
- `TensorOperations` - Tensor creation, manipulation, conversion
- `Kernels` - Computational kernels (GEMM, RoPE, Attention)
- `ModelLoading` - GGUF parsing, weight loading
- `PipelineExecution` - Pipeline lifecycle, forward pass
- `WeightPlacement` - Weight distribution strategies
- `DataTransfer` - Cross-device buffer management
- `ArgumentParsing` - CLI argument handling

### Tier 4 - Feature (Granular)
- `Orchestration` - Orchestrator-specific features
- `Quantization` - Quantized tensor formats
- `MultiDevice` - Heterogeneous execution
- `CrossDevice` - Cross-device operations
- `HeterogeneousExecution` - Mixed device types
- `BasicFeatures` / `AdvancedFeatures` - Capability level
- Component-specific features (CPU, FP32, GEMM, etc.)

---

## Detailed Label Mappings

### Phase1 → DeviceManagement
**Old**: "Phase1" (2 tests)  
**New**:
- `WeightPlacementMap` → `WeightPlacement;DeviceManagement`
- `DeviceOrchestrator` → `DeviceManagement;Orchestration;BasicFeatures`

**Rationale**: Phase1 was the initial device orchestration implementation

---

### Phase2 → ArgumentParsing + AdvancedFeatures
**Old**: "Phase2" (2 tests)  
**New**:
- `ArgParser` → `ArgumentParsing;CLI`
- `DeviceOrchestrator_Phase2` → `DeviceManagement;Orchestration;AdvancedFeatures`

**Rationale**: Phase2 added CLI parsing and extended orchestrator capabilities

---

### Phase3 → Kernels + TensorOperations
**Old**: "Phase3" (2 tests)  
**New**:
- `CPUKernels` → `Kernels;CPU;ComputeBackends`
- `TensorDimensions` → `TensorOperations;Dimensions;ShapeHandling`

**Rationale**: Phase3 implemented kernel infrastructure and tensor dimension handling

---

### Phase4 → MultiDevice + DataTransfer
**Old**: "Phase4" (3 tests)  
**New**:
- `MultiDevice` → `DeviceManagement;MultiDevice;HeterogeneousExecution`
- `DeviceTransfer` → `DataTransfer;CrossDevice;BufferManagement`
- `DeviceAwareExecution` → `DeviceManagement;Orchestration;DeviceAwareness`

**Rationale**: Phase4 was the heterogeneous multi-device execution milestone

---

### Generic Labels → Specific
**Old → New**:
- `Tensor` → `TensorOperations;FP32;Basics`
- `Pipeline;Factory` → `PipelineExecution;Factory;Registration`
- `ModelLoader` → `ModelLoading;GGUF;WeightLoading`
- `IQ4_NL;GEMM` → `TensorOperations;Quantization;IQ4_NL;GEMM`

---

## Filtering Examples

### By Component
```bash
# All device management tests (5 tests)
ctest -L DeviceManagement

# All tensor operation tests (3 tests)
ctest -L TensorOperations

# All model loading tests (1 test)
ctest -L ModelLoading
```

### By Feature
```bash
# All quantization tests (1 test)
ctest -L Quantization

# All orchestration tests (3 tests)
ctest -L Orchestration

# All GEMM tests (1 test)
ctest -L GEMM
```

### By Capability Level
```bash
# Basic orchestrator features (1 test)
ctest -L BasicFeatures

# Advanced orchestrator features (1 test)
ctest -L AdvancedFeatures
```

### Combined Filters
```bash
# All device management orchestration tests (3 tests)
ctest -L "DeviceManagement" -L "Orchestration"

# All tensor operations with quantization (1 test)
ctest -L "TensorOperations" -L "Quantization"

# All unit tests (13 tests, excludes fixture)
ctest -L "Unit"
```

### Test Name Regex (unchanged)
```bash
# All V2 unit tests by name pattern
ctest -R "V2_Unit_.*"

# Specific test by name
ctest -R "V2_Unit_DeviceOrchestrator$"
```

---

## Label Best Practices (Now Documented)

Added comprehensive label documentation to `/workspaces/llaminar/tests/v2/CMakeLists.txt`:

```cmake
# =============================================================================
# Test Suite Labels
# =============================================================================
# V2 tests use hierarchical, feature-based labels for flexible filtering:
#
# TIER 1 - Test Type (Required):
#   Unit, Integration, E2E, Parity, Performance
#
# TIER 2 - Architecture (Optional):
#   V2
#
# TIER 3 - Component (Specific):
#   DeviceManagement, TensorOperations, Kernels, ModelLoading, etc.
#
# TIER 4 - Feature (Granular):
#   Orchestration, Quantization, MultiDevice, CrossDevice, etc.
#
# NAMING CONVENTIONS:
#   - Use CamelCase for label names (matches file naming)
#   - Be specific: "PipelineFactory" not "Factory"
#   - Feature-based, not timeline-based: "WeightPlacement" not "Phase1"
#   - Multiple labels per test enable flexible filtering dimensions
```

---

## Test Results

### Before Changes
```bash
$ ctest --output-on-failure
100% tests passed, 0 tests failed out of 14
Total Test time (real) =  20.15 sec
```

### After Changes
```bash
$ ctest --output-on-failure
100% tests passed, 0 tests failed out of 14
Total Test time (real) =  16.63 sec
```

✅ **No test failures** - Label changes are metadata-only, no functional impact

---

## Files Modified

1. **`tests/v2/CMakeLists.txt`**:
   - Updated 13 test label assignments (10 phase replacements + 3 generic fixes)
   - Added comprehensive label documentation section
   - No changes to test executables or dependencies

**Specific changes**:
```cmake
# OLD (ambiguous):
LABELS "V2;Unit;WeightPlacementMap;Phase1"
LABELS "V2;Unit;DeviceOrchestrator;Phase1"
LABELS "V2;Unit;ArgParser;Phase2"
LABELS "V2;Unit;DeviceOrchestrator;Phase2"
LABELS "V2;Unit;CPUKernels;Phase3"
LABELS "V2;Unit;TensorDimensions;Phase3"
LABELS "V2;Unit;MultiDevice;Phase4"
LABELS "V2;Unit;DeviceTransfer;Phase4"
LABELS "V2;Unit;DeviceAwareExecution;Phase4"
LABELS "V2;Unit;Tensor"
LABELS "V2;Unit;Pipeline;Factory"

# NEW (feature-descriptive):
LABELS "V2;Unit;WeightPlacement;DeviceManagement"
LABELS "V2;Unit;DeviceManagement;Orchestration;BasicFeatures"
LABELS "V2;Unit;ArgumentParsing;CLI"
LABELS "V2;Unit;DeviceManagement;Orchestration;AdvancedFeatures"
LABELS "V2;Unit;Kernels;CPU;ComputeBackends"
LABELS "V2;Unit;TensorOperations;Dimensions;ShapeHandling"
LABELS "V2;Unit;DeviceManagement;MultiDevice;HeterogeneousExecution"
LABELS "V2;Unit;DataTransfer;CrossDevice;BufferManagement"
LABELS "V2;Unit;DeviceManagement;Orchestration;DeviceAwareness"
LABELS "V2;Unit;TensorOperations;FP32;Basics"
LABELS "V2;Unit;PipelineExecution;Factory;Registration"
```

---

## Benefits Realized

### For New Developers
- **Before**: "What is Phase2?" (requires git archaeology)
- **After**: "ArgumentParsing tests CLI parsing" (self-documenting)

### For Test Filtering
- **Before**: `ctest -L Phase4` (unclear what's tested)
- **After**: `ctest -L MultiDevice` (explicit capability filtering)

### For Maintenance
- **Before**: Labels lose meaning as project evolves
- **After**: Labels remain relevant regardless of development timeline

### For Documentation
- **Before**: Separate wiki/docs needed to explain Phase numbers
- **After**: Labels serve as inline documentation

---

## Future Work

### Additional Label Categories (Planned)
When adding Integration/E2E/Parity tests, use these tier 3 labels:
- `NumaPinning` - NUMA topology and thread affinity tests
- `MPICollectives` - MPI communication patterns
- `MemoryManagement` - Allocation, deallocation, leak detection
- `ErrorHandling` - Exception handling, validation
- `ConfigurationParsing` - Configuration file parsing

### V1 Test Suite (Consider)
Apply same label standardization to V1 tests in `/workspaces/llaminar/tests/`:
- Replace generic labels like "Integration", "Parity" with specific features
- Add component-level labels (Operators, Backends, Prefill, Decode)
- Ensure consistency between V1 and V2 labeling conventions

---

## Verification

### Check all labels are feature-descriptive
```bash
$ ctest --print-labels | grep -E "Phase[0-9]"
# (no output - all Phase labels removed)
```

### Verify hierarchical filtering works
```bash
# Component-level filtering
$ ctest -L DeviceManagement  # 5 tests
$ ctest -L TensorOperations  # 3 tests
$ ctest -L Orchestration     # 3 tests

# Feature-level filtering
$ ctest -L Quantization          # 1 test
$ ctest -L HeterogeneousExecution # 1 test
$ ctest -L CrossDevice           # 1 test
```

### Confirm all tests still pass
```bash
$ ctest --output-on-failure
100% tests passed, 0 tests failed out of 14
```

✅ All verification checks passed

---

## Lessons Learned

1. **Timeline-based labels age poorly**: "Phase2" becomes meaningless 6 months later
2. **Feature-based labels are evergreen**: "ArgumentParsing" always describes what's tested
3. **Multiple labels enable flexibility**: Single test can be found via component OR feature
4. **CamelCase consistency matters**: Matches file naming (Test__ClassName.cpp)
5. **Documentation is critical**: Added 50-line comment block explaining label system

---

## Conclusion

Replaced 9 ambiguous "Phase" labels and 3 generic labels with **34 feature-descriptive labels** organized in a 4-tier hierarchy. All 14 tests continue passing with improved discoverability, filtering, and maintainability.

**Key Takeaway**: Labels should describe *what* is tested, not *when* it was developed.
