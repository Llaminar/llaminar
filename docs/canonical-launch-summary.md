# Llaminar Canonical Launch Script Summary

## Files Created/Updated

### 1. **run-llaminar.sh** (New Canonical Script)
- **Purpose**: Single canonical way to run Llaminar with optimal MPI/OpenMP settings
- **Auto-Detection**: Automatically detects system topology using same logic as C++ `detectCPUTopology()`
- **Dynamic Configuration**: Calculates cores per socket from `/proc/cpuinfo` parsing
- **Environment**: Sets all optimal environment variables for MPI and OpenMP
- **Usage**: `./run-llaminar.sh [llaminar arguments]`

### 2. **Updated Documentation**

#### `.github/copilot-instructions.md`
- Added "Canonical Runtime Configuration" section
- Documents the canonical run script as the preferred method
- Includes environment variable reference
- Provides manual MPI execution fallback

#### `.github/instructions/llaminar-architecture.instructions.md`
- Added comprehensive "Canonical Runtime Configuration" section
- Documents optimal launch settings and environment configuration
- Includes system requirements and usage examples
- Covers both basic execution and model inference scenarios

#### `README.md`
- Updated with Quick Start section featuring canonical script
- Shows building and running examples
- Documents the automatic MPI/OpenMP configuration

## Canonical Settings Summary

### OpenMP Configuration (Dynamic Detection)
```bash
export OMP_NUM_THREADS=<detected>     # Auto-detected physical cores per socket
export OMP_PLACES=sockets             # Thread placement strategy
export OMP_PROC_BIND=close            # Bind threads close together
export OMP_NESTED=false               # Disable nested parallelism
export OMP_DYNAMIC=false              # Disable dynamic adjustment
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0                # Minimize thread blocking
```

### MPI Configuration (Optimal)
```bash
export OMPI_MCA_mpi_leave_pinned=1                     # Memory pinning
export OMPI_MCA_btl_vader_single_copy_mechanism=none   # NUMA optimization
export OMPI_MCA_btl_openib_allow_ib=1                  # InfiniBand support

# Execution pattern: 1 process per socket
mpirun -np ${SOCKETS} \
  --bind-to socket \
  --map-by socket \
  --mca mpi_leave_pinned 1 \
  --mca btl_vader_single_copy_mechanism none \
  --report-bindings \
  ./build/llaminar "$@"
```

## Key Benefits

1. **Dynamic Topology Detection**: Automatically detects physical cores per socket using same logic as C++ code
2. **Consistent Performance**: All users get optimal settings automatically without manual configuration
3. **NUMA Awareness**: 1 MPI process per socket for optimal memory locality
4. **Thread Optimization**: Auto-detected threads per socket with proper binding
5. **Memory Efficiency**: Memory pinning and NUMA-aware allocation
6. **Automatic Detection**: No manual configuration needed across different hardware
7. **Production Ready**: Empirically-tuned settings from performance testing

## Usage Examples

```bash
# System topology and benchmarking
./run-llaminar.sh -v --print-topology
./run-llaminar.sh -vv --matrix-size 2048

# Model inference
./run-llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -v
./run-llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -vv --profile

# Interactive usage
./run-llaminar.sh --help
```

This canonical approach ensures that all Llaminar deployments achieve optimal performance with the empirically-validated MPI and OpenMP configurations from our extensive testing work.