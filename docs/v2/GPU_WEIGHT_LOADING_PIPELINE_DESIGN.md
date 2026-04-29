# GPU-Side Pipelined Weight Loading & Repacking

## Design Document — Llaminar V2

**Status**: Design  
**Goal**: Make model loading purely I/O-bound (disk read + PCIe bandwidth), not CPU-bound or throttled by serial `hipMalloc` overhead.

---

## 1. Problem Statement

### Current Architecture

```
GGUF File ──mmap──► Host Memory ──CPU pack──► Host VNNI Buffer ──hipMalloc──► Device ──hipMemcpy──► Device VNNI
                                  (serial)       (per-weight)      (serial)              (per-weight)
```

### Bottlenecks Measured (Qwen3.5-35B-A3B MoE, 2× MI50)

| Phase | Time | Bottleneck |
|-------|------|-----------|
| GGUF mmap + preload | ~29s | Expert slicing + `ensureOnDevice` overhead for non-HOST_RESIDENT tensors |
| `packGemmWeights` (ROCm:0) | 25.7s | CPU VNNI packing + serial hipMalloc per weight + per-weight stream creation |
| `packGemmWeights` (ROCm:1) | 23.8s | Same, sequential after ROCm:0 |
| MoE expert packing | ~1.2s | Already efficient (batch-packed, single slab) |
| **Total** | **~84s** | |

### Theoretical Minimum (I/O-bound)

| Resource | Bandwidth | Model Size | Time |
|----------|-----------|-----------|------|
| Disk → Host (mmap, NVMe) | ~3.5 GB/s | 21.2 GB | 6.1s |
| Host → Device (PCIe 3.0 ×16) | ~12 GB/s | ~10.6 GB/device | 0.9s/device |
| **Theoretical floor** | | | **~8s** |

We're 10× slower than the I/O floor. The gap is dominated by:
1. **CPU VNNI packing**: `packVnniBlock()` called per-block per-row, millions of times per weight
2. **Serial `hipMalloc`**: One allocation per weight matrix (5-6 per layer × 40 layers = ~200 allocations)
3. **Per-weight stream lifecycle**: `hipStreamCreate` + `hipHostMalloc` + `hipStreamSync` + `hipStreamDestroy` per weight
4. **Sequential device processing**: ROCm:1 waits for ROCm:0 to finish entirely

---

## 2. Design Overview

### Target Architecture: GPU-Side Repack Pipeline

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                   LOAD ORCHESTRATOR                      │
                    │  (single coordinator, multi-device, multi-stream)        │
                    └──────────────┬────────────────┬─────────────────────────┘
                                   │                │
                    ┌──────────────▼───┐    ┌───────▼──────────────┐
                    │   DevicePipeline │    │   DevicePipeline     │
                    │     (ROCm:0)     │    │     (ROCm:1)         │
                    └──────────────┬───┘    └───────┬──────────────┘
                                   │                │
            ┌──────────────────────┼────────────────┼──────────────────────┐
            │                      │                │                      │
    ┌───────▼───────┐   ┌─────────▼──────┐  ┌──────▼────────┐  ┌─────────▼──────┐
    │  H2D Stream 0 │   │ H2D Stream 1   │  │ H2D Stream 2  │  │ Repack Stream  │
    │  (weight N)   │   │ (weight N+1)   │  │ (weight N+2)  │  │ (GPU kernels)  │
    └───────┬───────┘   └────────┬───────┘  └──────┬────────┘  └────────┬───────┘
            │                    │                  │                    │
            │              ┌─────▼─────────────────▼──────┐             │
            └─────────────►│     Pre-allocated VRAM Pool    │◄───────────┘
                           │  (single hipMalloc at init)    │
                           └────────────────────────────────┘
```

### Key Principles

1. **Pre-allocate everything**: Single `hipMalloc` per device for all weight VRAM at orchestration init. Zero `hipMalloc` during weight loading.
2. **GPU-side repack**: Upload raw GGUF blocks directly to a staging region on GPU, then run a HIP kernel to repack into VNNI in-place on device.
3. **Multi-stream overlap**: N H2D transfer streams + 1 repack stream per device. While stream 0 transfers weight N, stream 1 transfers weight N+1, and the repack stream processes weight N-1.
4. **Parallel device loading**: All devices load simultaneously, not sequentially.
5. **Pinned ring buffer**: Reusable pinned host staging buffer (not per-weight `hipHostMalloc`).

---

## 3. Component Design

### 3.1 `WeightVRAMPool` — Pre-allocated Device Memory

**Replaces**: Per-weight `hipMalloc` calls in `ensureWeightsConverted()`.

```
┌────────────────────────────────────────────────────────────────┐
│                    WeightVRAMPool (per device)                  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │  VNNI Payload Region                                  │      │
│  │  [weight0_payload | weight1_payload | ... | weightN]  │      │
│  │  Total: sum of all weight native_vnni sizes           │      │
│  └──────────────────────────────────────────────────────┘      │
│  ┌──────────────────────────────────────────────────────┐      │
│  │  Scales Region                                        │      │
│  │  [weight0_scales | weight1_scales | ... | weightN]    │      │
│  └──────────────────────────────────────────────────────┘      │
│  ┌──────────────────────────────────────────────────────┐      │
│  │  Mins Region (asymmetric formats only)                │      │
│  │  [weight0_mins | weight1_mins | ... | weightN]        │      │
│  └──────────────────────────────────────────────────────┘      │
│  ┌──────────────────────────────────────────────────────┐      │
│  │  Staging Region (raw GGUF blocks for GPU repack)      │      │
│  │  [ring buffer, 2-3 weight slots for overlap]          │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
│  Allocated: 1× hipMalloc(total_bytes) at init                  │
│  Layout:    Bump-allocator assigns offsets per weight           │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

**Interface**:

```cpp
class WeightVRAMPool {
public:
    struct WeightSlot {
        uint8_t* d_native_vnni_payload;   // Offset into payload region
        uint16_t* d_native_vnni_scales;   // Offset into scales region
        uint16_t* d_native_vnni_mins;     // Offset into mins region (or nullptr)
        uint32_t* d_native_vnni_emins;    // Offset into emins region (or nullptr)
        float* d_int8_scales;             // Per-row scales for INT8 path
        uint8_t* d_staging;               // Temporary: raw GGUF blocks on device
        size_t staging_bytes;             // Size of staging region for this weight
    };

    // Phase 1: Plan (calculate sizes, no allocation)
    void planWeight(const std::string& name, const NativeVnniFormatInfo& info,
                    int N, int K, bool needs_int8_scales);
    
    // Phase 2: Allocate (single hipMalloc)
    bool allocate(int rocm_device_id);
    
    // Phase 3: Get slot (zero-cost offset lookup)
    WeightSlot getSlot(const std::string& name) const;
    
    // Phase 4: Release staging region after all repacks complete
    void releaseStagingRegion();

private:
    void* d_base_ = nullptr;             // Single allocation base
    size_t total_bytes_ = 0;
    std::unordered_map<std::string, WeightSlot> slots_;
};
```

**Sizing calculation** (at `finalizeForDevices` time):

```cpp
// For each GEMM weight assigned to this device:
for (auto& [name, tensor] : gemm_weights) {
    auto* info = quant->vnniFormatInfo();
    int N = tensor->rows(), K = tensor->cols();
    int blocks_per_row = K / 32;
    
    size_t payload_bytes = blocks_per_row * N * info->payload_bytes;
    size_t scales_bytes  = blocks_per_row * N * sizeof(uint16_t);
    size_t mins_bytes    = info->is_asymmetric ? scales_bytes : 0;
    size_t staging_bytes = tensor->raw_byte_count();  // Raw GGUF blocks
    
    pool.planWeight(name, *info, N, K, /*needs_int8_scales=*/true);
}
pool.allocate(rocm_device_id);
```

### 3.2 `PinnedRingBuffer` — Reusable Host Staging

**Replaces**: Per-weight `hipHostMalloc` + `hipHostFree` in `startupMemcpyAsyncOrSync()`.

```cpp
class PinnedRingBuffer {
public:
    PinnedRingBuffer(size_t slot_size, int num_slots);
    ~PinnedRingBuffer();  // hipHostFree once
    
    // Get next available slot (blocks if all in-flight)
    void* acquireSlot(hipEvent_t* completion_event);
    
    // Mark slot as available after H2D completes
    void releaseSlot(void* slot_ptr);

private:
    void* pinned_base_ = nullptr;     // hipHostMalloc(slot_size * num_slots) once
    size_t slot_size_;
    int num_slots_;
    std::vector<hipEvent_t> events_;  // Completion tracking
    int head_ = 0;                    // Next slot to acquire
};
```

**Sizing**: `slot_size` = max weight raw byte size across all GEMM weights (typically ~100-200 MB for large models). `num_slots` = number of H2D streams (3-4). Total pinned memory: ~400-800 MB.

### 3.3 `GpuVnniRepackKernel` — GPU-Side Format Transformation

**Replaces**: CPU-side `packVnniBlock()` per-block per-row loop.

**Core insight**: The VNNI repack is a pure data layout transformation — no floating-point computation for native VNNI formats. The GPU kernel reads raw GGUF blocks from a staging area and writes the interleaved VNNI layout to the final destination.

#### Kernel Design (Q4_K example)

```
Input:  Raw Q4_K super-blocks in GGUF layout
        blocks[row * sb_per_row + sb_idx] = { d, dmin, scales[12], qs[128] }
        
Output: Interleaved VNNI layout
        payload[b * N + n]  = 16 repacked nibble bytes
        scales[b * N + n]   = FP16 (d * sc)
        mins[b * N + n]     = FP16 (-dmin * m)
```

```cpp
// HIP kernel: one thread per (n, b) pair
__global__ void repack_q4k_to_vnni(
    const Q4_KBlock* __restrict__ d_raw_blocks,  // Raw GGUF super-blocks
    uint8_t* __restrict__         d_payload,     // Output VNNI payload
    uint16_t* __restrict__        d_scales,      // Output VNNI scales  
    uint16_t* __restrict__        d_mins,        // Output VNNI mins
    int N,                                        // Output features (rows)
    int K,                                        // Input features (cols)
    int sb_per_row)                               // Super-blocks per row
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= N) return;
    
    const int blocks_per_row = K / 32;
    if (b >= blocks_per_row) return;
    
    const int sb_idx = b / 8;
    const int sub_idx = b % 8;
    const Q4_KBlock& blk = d_raw_blocks[n * sb_per_row + sb_idx];
    
    // Extract group scale/min from packed scales[12]
    const int group_idx = sub_idx / 2;
    const int is_high = sub_idx & 1;
    const uint8_t* src32 = blk.qs + group_idx * 32;
    
    // Repack nibbles (same logic as CPU packVnniBlock)
    uint8_t repacked[16];
    if (is_high) {
        for (int i = 0; i < 16; ++i)
            repacked[i] = (src32[i] >> 4) | (src32[i + 16] & 0xF0);
    } else {
        for (int i = 0; i < 16; ++i)
            repacked[i] = (src32[i] & 0xF) | ((src32[i + 16] & 0xF) << 4);
    }
    
    // Write to interleaved VNNI layout
    const size_t linear = (size_t)b * N + n;
    // 16-byte payload write (4× uint32 stores for coalescing)
    uint32_t* dst = reinterpret_cast<uint32_t*>(d_payload + linear * 16);
    const uint32_t* src = reinterpret_cast<const uint32_t*>(repacked);
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
    
    // Extract and write scale/min
    uint8_t sc, m_val;
    device_get_scale_min_k4(sub_idx, blk.scales, &sc, &m_val);
    d_scales[linear] = __float2half_rn(
        __half2float(blk.d) * (float)sc);
    d_mins[linear] = __float2half_rn(
        -__half2float(blk.dmin) * (float)m_val);
}
```

**Performance characteristics**:
- Grid: `(ceil(N/64), ceil(blocks_per_row/4))` blocks, `(64, 4)` threads
- Each thread handles one 32-element block → one 16-byte payload write
- For a 2048×512 weight (e.g., MoE expert): 2048 × 16 = 32,768 threads → ~512 wavefronts
- GPU throughput: ~50 GB/s read + ~30 GB/s write on MI50 → completes in microseconds
- **Compared to CPU**: ~100ms CPU time per weight → ~0.1ms GPU time = **~1000× speedup**

#### Format Dispatch

Each quantization format gets its own repack kernel (template specialization on codebook_id):

| Format | Complexity | GPU Kernel Notes |
|--------|-----------|-----------------|
| Q4_0 | Simple: 16-byte memcpy + FP16 scale | Direct copy, trivial |
| Q4_K | Medium: nibble repack + scale extraction from packed `scales[12]` | Shown above |
| Q5_0/Q5_1 | Medium: 20-byte payload (qs + qh) | Two reads, one write |
| Q6_K | Complex: super-block with dual scales | Similar to Q4_K with extra fields |
| IQ4_NL | Simple: 16-byte copy + LUT (codebook already on device) | Trivial |
| Q2_K | Complex: super-block + embedded mins | Most complex, still ~microseconds |

### 3.4 `DeviceLoadPipeline` — Per-Device Orchestration

**Replaces**: Sequential `packGemmWeights()` → `ensureWeightsConverted()` per weight.

```
                    Time ──────────────────────────────────────────►
                    
H2D Stream 0:       [==== W0 H2D ====]                [==== W3 H2D ====]
H2D Stream 1:            [==== W1 H2D ====]                [==== W4 H2D ====]
H2D Stream 2:                 [==== W2 H2D ====]                [==== W5 H2D ====]
Repack Stream:            [=W0 repack=]                 [=W3 repack=]
                               [=W1 repack=]                 [=W4 repack=]
                                    [=W2 repack=]                 [=W5 repack=]
                    
                    ◄── Triple-buffered: H2D and repack overlap ──►
```

```cpp
class DeviceLoadPipeline {
public:
    DeviceLoadPipeline(int rocm_device_id, int num_h2d_streams = 3);
    
    struct WeightJob {
        std::string name;
        const void* host_raw_data;        // mmap'd GGUF pointer
        size_t raw_bytes;
        WeightVRAMPool::WeightSlot slot;  // Pre-assigned device offsets
        NativeVnniFormatInfo format;
        int N, K;
    };
    
    // Submit all weights for this device (non-blocking)
    void submitBatch(std::vector<WeightJob>&& jobs);
    
    // Wait for all transfers + repacks to complete
    void synchronize();
    
private:
    int device_id_;
    std::vector<hipStream_t> h2d_streams_;   // N transfer streams
    hipStream_t repack_stream_;               // 1 repack stream
    std::vector<hipEvent_t> transfer_done_;   // Per-stream events
    PinnedRingBuffer pinned_ring_;            // Reusable pinned staging
};
```

**Pipeline execution** (per weight job):

```cpp
void DeviceLoadPipeline::processJob(int stream_idx, const WeightJob& job) {
    hipStream_t h2d = h2d_streams_[stream_idx];
    
    // 1. Acquire pinned slot
    void* pinned = pinned_ring_.acquireSlot(nullptr);
    
    // 2. CPU memcpy: mmap → pinned (could be overlapped with prior GPU work)
    std::memcpy(pinned, job.host_raw_data, job.raw_bytes);
    
    // 3. H2D async: pinned → device staging
    hipMemcpyAsync(job.slot.d_staging, pinned, job.raw_bytes,
                   hipMemcpyHostToDevice, h2d);
    
    // 4. Record event when H2D completes
    hipEventRecord(transfer_done_[stream_idx], h2d);
    
    // 5. Repack stream waits for this specific H2D
    hipStreamWaitEvent(repack_stream_, transfer_done_[stream_idx], 0);
    
    // 6. Launch GPU repack kernel on repack stream
    launchRepackKernel(repack_stream_, job);
    
    // 7. Release pinned slot after H2D (not after repack)
    //    Use event callback or track in ring buffer
}
```

### 3.5 `LoadOrchestrator` — Multi-Device Coordination

**Replaces**: Sequential per-device loop in `finalizeForDevices()`.

```cpp
class LoadOrchestrator {
public:
    LoadOrchestrator(const std::vector<DeviceId>& devices,
                     const ModelLoader& loader);
    
    // Phase 1: Scan model to plan memory layout
    //   - Iterate all GEMM + MoE weights
    //   - Calculate per-device VNNI sizes (with TP sharding)
    //   - Plan WeightVRAMPool layout per device
    bool plan();
    
    // Phase 2: Allocate (one hipMalloc per device, parallel)
    bool allocate();
    
    // Phase 3: Load (all devices simultaneously)
    //   - Warm page cache on leading rank (sequential mmap read)
    //   - Dispatch DeviceLoadPipeline per device (parallel)
    //   - Each pipeline: H2D + GPU repack, triple-buffered
    bool load();
    
    // Phase 4: Finalize
    //   - Release staging regions
    //   - Create kernel objects pointing into WeightVRAMPool slots
    //   - Free host raw data if desired
    bool finalize();

private:
    std::vector<std::unique_ptr<DeviceLoadPipeline>> pipelines_;
    std::vector<std::unique_ptr<WeightVRAMPool>> pools_;
};
```

---

## 4. Page Cache Warming Strategy

The GGUF file is mmap'd. On first access, each page faults into the kernel's page cache. With 21 GB of model data, cold page faults during parallel H2D transfers cause unpredictable latency spikes.

**Strategy**: The leading rank does a sequential read-ahead pass before dispatching to devices:

```cpp
void LoadOrchestrator::warmPageCache(const void* mmap_base, size_t file_size) {
    // Sequential scan to warm page cache (Linux read-ahead optimized)
    // Use madvise(MADV_SEQUENTIAL) for kernel-level prefetch
    madvise(const_cast<void*>(mmap_base), file_size, MADV_SEQUENTIAL);
    
    // Touch every page (4KB) to ensure it's in RAM
    volatile char sink = 0;
    const char* base = static_cast<const char*>(mmap_base);
    for (size_t offset = 0; offset < file_size; offset += 4096) {
        sink += base[offset];
    }
    // After this: all pages in memory, subsequent reads are memcpy speed
    
    // Switch to random access pattern for parallel device reads
    madvise(const_cast<void*>(mmap_base), file_size, MADV_RANDOM);
}
```

**Expected time**: 21.2 GB / ~3.5 GB/s NVMe = ~6s. This overlaps with `hipMalloc` for VRAM pools.

---

## 5. INT8 VNNI Path: GPU-Side Requantization

The native VNNI path (§3.3) handles formats like Q4_K directly. But the INT8 VNNI path (`int8_data_vnni`) requires actual requantization: Q4_K → dequant → find max_abs → requant to INT8 → interleave.

This is more compute-intensive and benefits even more from GPU execution:

```cpp
// Phase 1: Per-row max-abs computation (GPU reduction)
__global__ void compute_row_scales_q4k(
    const Q4_KBlock* __restrict__ d_raw,
    float* __restrict__ d_row_scales,    // [N] output
    int N, int K, int sb_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    
    float max_abs = 0.0f;
    for (int sb = 0; sb < sb_per_row; ++sb) {
        const Q4_KBlock& blk = d_raw[n * sb_per_row + sb];
        float d = __half2float(blk.d);
        float dmin = __half2float(blk.dmin);
        // Iterate sub-blocks, compute dequantized max
        for (int sub = 0; sub < 8; ++sub) {
            uint8_t sc, m;
            device_get_scale_min_k4(sub, blk.scales, &sc, &m);
            float block_max = fabsf(d * sc * 15.0f - dmin * m);  // Conservative max
            max_abs = fmaxf(max_abs, block_max);
        }
    }
    d_row_scales[n] = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
}

// Phase 2: Requant + VNNI interleave (GPU kernel)
__global__ void requant_q4k_to_int8_vnni(
    const Q4_KBlock* __restrict__ d_raw,
    const float* __restrict__     d_row_scales,  // [N] from Phase 1
    int8_t* __restrict__          d_int8_vnni,   // [K/4][N][4] output
    int N, int K, int sb_per_row)
{
    // Each thread handles one 32-element block for one row
    // Dequantizes Q4_K → FP32 → requantizes to INT8 → writes VNNI interleaved
    // ...
}
```

---

## 6. Integration with Existing Architecture

### 6.1 Entry Point Change

**Current** (`WeightManager::finalizeForDevices()`):
```cpp
for (auto& dev : devices) {
    packGemmWeights(dev);  // Sequential: CPU pack + per-weight hipMalloc + H2D
}
```

**New**:
```cpp
LoadOrchestrator orchestrator(devices, loader_);
orchestrator.plan();       // Scan weights, calculate VRAM layouts
orchestrator.allocate();   // One hipMalloc per device (parallel)
orchestrator.load();       // All devices simultaneously, GPU repack
orchestrator.finalize();   // Create kernel objects
```

### 6.2 Kernel Object Creation

After `LoadOrchestrator::load()` completes, kernel objects must point into the `WeightVRAMPool` instead of per-weight device allocations.

**Current**: `ROCmQuantisedGemmKernel::ensureWeightsConverted()` does hipMalloc + hipMemcpy per weight.
**New**: Constructor receives pre-computed device pointers from `WeightVRAMPool::WeightSlot`:

```cpp
// New constructor: weights already on device
ROCmQuantisedGemmKernel::ROCmQuantisedGemmKernel(
    const WeightVRAMPool::WeightSlot& slot,
    int N, int K, int rocm_device_id,
    uint8_t codebook_id, uint32_t blocks_per_row)
    : packed_(nullptr), rocm_device_id_(rocm_device_id),
      N_(N), K_(K), weights_converted_(true)  // Already on device!
{
    impl_ = std::make_unique<Impl>();
    impl_->d_weights_native_vnni = slot.d_native_vnni_payload;
    impl_->d_weights_native_scales = slot.d_native_vnni_scales;
    impl_->d_weights_native_mins = slot.d_native_vnni_mins;
    impl_->d_scales_B = reinterpret_cast<float*>(slot.d_int8_scales);
    impl_->has_native_vnni = true;
    impl_->native_vnni_codebook_id = codebook_id;
    impl_->native_vnni_blocks_per_row = blocks_per_row;
}
```

### 6.3 MoE Expert Weights

MoE expert weights already use batch packing (`packMoEExpertsROCm`), which is efficient. The new pipeline extends this:

1. **Plan**: Include all MoE expert slabs in `WeightVRAMPool` 
2. **Load**: Upload raw expert tensor slices to staging, GPU repack into VNNI slab
3. **Finalize**: Create per-expert kernel objects pointing into slab offsets (same as current)

The per-expert VNNI interleave kernel is the same as regular weights but operating on a contiguous slab of `num_experts × rows_per_expert` rows.

### 6.4 Backward Compatibility

The new pipeline is opt-in:
- **ROCm**: Default path for `DeviceType::ROCM` when `WeightVRAMPool` is available
- **CUDA**: Falls back to current (legacy) pipeline. CUDA support is a future phase — not urgent because there are no CUDA MoE kernels yet, and `cudaMalloc` is faster than `hipMalloc` so the ROCm-specific bottlenecks are less severe on CUDA. The `WeightVRAMPool` and `LoadOrchestrator` interfaces are backend-agnostic; CUDA repack kernels (`.cu` equivalents of the `.hip` kernels) can be added later without architectural changes.
- **CPU**: Not applicable (no device memory)
- **Env var**: `LLAMINAR_GPU_WEIGHT_PIPELINE=0` to disable and use legacy path

---

## 7. Expected Performance

### Time Budget (Target)

| Phase | Current | New | Notes |
|-------|---------|-----|-------|
| Page cache warm | 0s (cold faults during load) | ~6s (sequential mmap scan) | One-time, overlaps with hipMalloc |
| VRAM pool alloc | N/A (200× serial hipMalloc = ~10s) | ~0.01s (1× hipMalloc per device) | Eliminates fragmentation |
| H2D transfer | ~5s (serial per-weight, with overhead) | ~1.8s (parallel streams, all devices) | PCIe 3.0 ×16 @ 12 GB/s |
| GPU repack | N/A (CPU: ~40s) | ~0.2s (GPU kernels) | ~1000× faster than CPU |
| Kernel creation | ~2s | ~0.5s (pre-computed slots) | No hipMalloc, just pointer assignment |
| **Total** | **~84s** | **~9s** | **~9.3× speedup** |

### Memory Overhead

| Resource | Current | New | Delta |
|----------|---------|-----|-------|
| Device VNNI weights | Same | Same | 0 |
| Device staging (temporary) | 0 | ~200 MB (largest weight) × 3 streams | +600 MB/device (freed after load) |
| Pinned host ring buffer | Per-weight alloc/free | 200 MB × 3 slots = 600 MB | +600 MB (fixed, freed after load) |
| Host VNNI buffers | ~10 GB (packed_ host copies) | 0 (GPU repacks from raw GGUF) | **-10 GB host RAM saved** |

---

## 8. Implementation Plan

### Phase 1: Foundation (Non-breaking)
1. `WeightVRAMPool` — plan/allocate/getSlot interface
2. `PinnedRingBuffer` — reusable pinned staging
3. `LoadOrchestrator` — multi-device coordination skeleton
4. Unit tests: pool sizing, slot assignment, ring buffer lifecycle

### Phase 2: GPU Repack Kernels
5. `repack_q4_0_to_vnni` — simplest format, proof of concept
6. `repack_q4_k_to_vnni` — most common format in production models
7. `repack_iq4_nl_to_vnni` — LUT-based format (codebook already on device)
8. Format dispatch table (codebook_id → kernel function pointer)
9. Parity tests: GPU-repacked VNNI vs CPU-repacked VNNI (bit-exact for pure copy formats, tolerance for FP16 scale computation)

### Phase 3: Pipeline Integration
10. `DeviceLoadPipeline` — triple-buffered H2D + repack
11. Wire into `WeightManager::finalizeForDevices()` as alternative path
12. Integration tests: full model load + inference produces identical output

### Phase 4: INT8 Path
13. GPU-side `compute_row_scales` kernel (per-row max-abs)
14. GPU-side `requant_to_int8_vnni` kernel
15. Parity tests: INT8 requantized weights match CPU path within tolerance

### Phase 5: MoE + Polish
16. MoE expert batch repack on GPU (single slab, one kernel launch)
17. Page cache warming with `madvise`
18. Performance benchmarks and tuning
19. Remove legacy `ensureWeightsConverted` code path for ROCm (behind feature flag)

### Phase 6: CUDA Support (Future, Low Priority)
20. Port HIP repack kernels to CUDA (`.hip` → `.cu`, mostly mechanical)
21. CUDA `WeightVRAMPool` backend (`cudaMalloc` / `cudaHostAlloc`)
22. Parity tests on CUDA devices
23. Evaluate whether CUDA benefits enough to justify as default (CUDA's `cudaMalloc` is already much faster than `hipMalloc`, so the ROCm-specific serial allocation bottleneck is less severe)

> **Note**: Phase 6 is deferred because (a) there are no CUDA MoE kernels yet, so the MoE loading path isn't exercised on CUDA, and (b) CUDA's allocator doesn't suffer from the same `hipMalloc` serialization overhead that makes ROCm loading 10× slower than the I/O floor. The pipeline interfaces (`WeightVRAMPool`, `LoadOrchestrator`, `DeviceLoadPipeline`) are backend-agnostic by design, so CUDA support is additive — no refactoring required.

---

## 9. File Layout

```
src/v2/loaders/
├── gpu_pipeline/
│   ├── WeightVRAMPool.h              // Pre-allocated device memory pool
│   ├── WeightVRAMPool.cpp
│   ├── PinnedRingBuffer.h            // Reusable pinned host staging
│   ├── PinnedRingBuffer.cpp
│   ├── DeviceLoadPipeline.h          // Per-device triple-buffered pipeline
│   ├── DeviceLoadPipeline.cpp
│   ├── LoadOrchestrator.h            // Multi-device coordination
│   └── LoadOrchestrator.cpp
│
src/v2/kernels/rocm/repack/
│   ├── VnniRepackKernels.hip         // GPU-side VNNI repack kernels (all formats, ROCm)
│   ├── VnniRepackKernels.h           // Host-side launch wrappers
│   ├── Int8RequantKernels.hip        // GPU-side INT8 requantization (ROCm)
│   └── Int8RequantKernels.h
│
# Future (Phase 6):
src/v2/kernels/cuda/repack/
│   ├── VnniRepackKernels.cu          // CUDA equivalents (mechanical port from .hip)
│   ├── VnniRepackKernels.h
│   ├── Int8RequantKernels.cu
│   └── Int8RequantKernels.h
│
tests/v2/unit/
│   ├── Test__WeightVRAMPool.cpp
│   ├── Test__PinnedRingBuffer.cpp
│   └── Test__VnniRepackKernels.cpp
tests/v2/integration/
│   └── Test__GpuWeightPipeline.cpp   // End-to-end: load → repack → inference parity
```

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| GPU repack kernel correctness | Wrong weights → garbage inference | Bit-exact parity tests (native VNNI path), tolerance tests (INT8 path) |
| VRAM fragmentation from staging region | OOM for weights after staging freed | Staging region at end of pool allocation; `hipFree` + re-`hipMalloc` if needed |
| FP16 scale precision on GPU vs CPU | Numerical divergence | Use `__float2half_rn` (round-to-nearest) matching CPU `fp32_to_fp16` |
| `madvise(MADV_SEQUENTIAL)` not available | Slower page warming | Graceful fallback to simple sequential read |
| Pinned memory exhaustion | `hipHostMalloc` fails | Fall back to pageable staging with sync memcpy |
| MI50 doesn't support hipMallocAsync | Can't use async pool API | Not needed — single `hipMalloc` per device avoids the issue entirely |

---

## 11. Open Questions

1. **Staging region lifetime**: Can the staging region be reclaimed as workspace memory after loading? This would save ~600 MB/device that could be used for KV cache or scratch buffers.

2. **CUDA parity**: Deferred to Phase 6. CUDA's `cudaMalloc` is significantly faster than ROCm's `hipMalloc`, so the serial allocation bottleneck is less severe. No CUDA MoE kernels exist yet, further reducing urgency. The pipeline interfaces are backend-agnostic; CUDA `.cu` repack kernels can be added later without architectural changes.

3. **Weight streaming**: If `LLAMINAR_WEIGHT_STREAMING=1` is enabled (for VRAM-constrained systems), the VRAM pool approach conflicts. Need a separate "streaming pool" with LRU eviction that still uses GPU repack.

4. **Multi-file GGUF**: Split GGUF files (`--split`) have tensors across multiple files. The page cache warming and mmap strategy must handle file boundaries.
