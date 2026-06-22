# Llaminar V2: End-to-End Execution Architecture

This document provides detailed diagrams showing how Llaminar constructs and executes inference graphs across heterogeneous multi-device, multi-node configurations.

## Table of Contents
- [Architecture Layers](#architecture-layers)
- [MPI Orchestration and Topology Discovery](#mpi-orchestration-and-topology-discovery)
- [Memory Management and GPU Workspace Architecture](#memory-management-and-gpu-workspace-architecture)
- [Scenario 1: Single Machine, 2 Ranks, Heterogeneous](#scenario-1-single-machine-2-ranks-heterogeneous)
- [Scenario 2: Six Machines, 12 Ranks, Complex Topology](#scenario-2-six-machines-12-ranks-complex-topology)
- [Collective Backend Selection](#collective-backend-selection)
- [Graph Construction Flow](#graph-construction-flow)
- [Execution Flow](#execution-flow)

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION LAYER                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  llaminar2 CLI  →  InferenceRunnerFactory  →  IInferenceRunner      │    │
│  │    ├── createInferenceRunner(model_ctx, mpi_ctx, device, config)    │    │
│  │    └── createTestableInferenceRunner(IModelContext*, device, config)│    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ORCHESTRATION LAYER                               │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  GraphOrchestrator (implements IInferenceRunner)                      │   │
│  │    ├── InferenceState (hidden, logits, kv_cache, positions)          │   │
│  │    ├── DeviceGraphExecutor (DAG execution engine)                          │   │
│  │    ├── LayerGraphCache (decode-mode graph caching)                   │   │
│  │    ├── Qwen2Graph (IGraphBuilder - declarative graph construction)   │   │
│  │    └── CollectiveContext (MPI backend routing)                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│         ┌────────────────────────────┼────────────────────────────┐          │
│         ▼                            ▼                            ▼          │
│  ┌──────────────────┐   ┌─────────────────────────┐   ┌──────────────────┐  │
│  │   MPITopology    │   │   PlacementStrategy     │   │  DeviceManager   │  │
│  │ (IMPITopology)   │   │  (device→stage mapping) │   │ (device registry)│  │
│  ├──────────────────┤   ├─────────────────────────┤   ├──────────────────┤  │
│  │ • world_size     │   │ • CPUOnlyStrategy       │   │ • DeviceInventory│  │
│  │ • rank           │   │ • GPUFirstStrategy      │   │ • DeviceId       │  │
│  │ • node_rank      │   │ • HybridOptimalStrategy │   │ • DeviceGroup    │  │
│  │ • local_rank     │──▶│                         │◀──│ • CUDA/ROCm query│  │
│  │ • ranks_per_node │   │ Outputs: PlacementPlan  │   │ • memory/compute │  │
│  │ • all_hostnames  │   │ • LayerPlacement[]      │   │   capabilities   │  │
│  │ • WorkRange APIs │   │ • GlobalTensorPlacement │   │                  │  │
│  └────────┬─────────┘   │ • decode_devices[]      │   └────────┬─────────┘  │
│           │             │ • weight_fractions[]    │            │            │
│           │             └─────────────────────────┘            │            │
│           │                         ▲                          │            │
│           │             ┌───────────┴───────────┐              │            │
│           ▼             ▼                       ▼              ▼            │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                       NUMATopology                                  │    │
│  │  ├── Socket/Core detection (/proc/cpuinfo)                         │    │
│  │  ├── NUMA node affinity                                            │    │
│  │  ├── CPU memory bandwidth estimation                               │    │
│  │  │     └── DDR4/DDR5 detection, channel count, transfers/sec       │    │
│  │  └── estimateCPUBandwidth() → {bandwidth_gbps, memory_channels}    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                     PlacementInput → PlacementPlan                  │    │
│  │                                                                     │    │
│  │   PlacementInput (provided to strategy):                           │    │
│  │     • n_layers, n_heads, hidden_dim, vocab_size                    │    │
│  │     • bytes_per_layer, total_model_bytes                           │    │
│  │     • world_size, cluster_inventory (ClusterInventory)             │    │
│  │     • gpu_memory_bandwidth, cpu_memory_bandwidth                   │    │
│  │     ────────────────────────────────────────────                   │    │
│  │     • getPhaseDeviceWeights(phase) → {gpu_weight, cpu_weight}      │    │
│  │     • cpuShouldParticipate(phase) → bool (≥5% of bandwidth)        │    │
│  │     • getTotalDecodeBandwidth() → combined GPU + CPU GB/s          │    │
│  │                                                                     │    │
│  │   PlacementPlan (output from strategy):                            │    │
│  │     • layers[]: LayerPlacement per transformer layer               │    │
│  │     • global: embedding, lm_head, final_norm device placement      │    │
│  │     ────────────────────────────────────────────                   │    │
│  │     LayerPlacement:                                                │    │
│  │       • device (PREFILL primary)                                   │    │
│  │       • decode_devices[] (DECODE participants)                     │    │
│  │       • decode_weight_fractions[] (bandwidth-proportional)         │    │
│  │       • cpu_participates_in_decode: bool                           │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GRAPH LAYER                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ComputeGraph (DAG of ComputeNodes)                                  │    │
│  │    ├── ComputeNode {name, stage, dependencies, device, completed}   │    │
│  │    ├── addNode(), addDependency(), getExecutionOrder()              │    │
│  │    ├── merge() - combine subgraphs with dependency linking          │    │
│  │    └── getRootNodes(), getLeafNodes() for DAG traversal             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  IGraphBuilder (model-specific graph construction)                   │    │
│  │    ├── Qwen2Graph: buildAttentionGraph(), buildFFNGraph()           │    │
│  │    ├── buildForwardGraph() → complete embedding→layers→LM head      │    │
│  │    └── buildLayerGraph() → single transformer layer                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  DeviceGraphBufferManager (IGraphBufferManager)                            │    │
│  │    ├── queryAvailableMemory(device) → total, free, usable          │    │
│  │    ├── computeWorkspaceBudget(device, config) → budget bytes       │    │
│  │    ├── allocateWithAliasing(graph) → LivenessAnalyzer optimization │    │
│  │    └── getBuffer(node, buffer) → tensor ptr                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  LivenessAnalyzer (buffer aliasing optimization)                    │    │
│  │    ├── analyze(graph) → BufferLiveness[] (first/last use per buffer)│    │
│  │    ├── computeAliasingGroups() → non-overlapping SCRATCH buffers   │    │
│  │    └── computeMemoryUsage() → (original_bytes, optimized_bytes)    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            EXECUTION LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  DeviceGraphExecutor (IGraphExecutor)                                      │    │
│  │    ├── execute(graph, ctx) → topological order execution            │    │
│  │    ├── executeMultiDevice(graph, contexts) → multi-GPU support      │    │
│  │    ├── setSnapshotCallback() for debugging/parity tests            │    │
│  │    └── verifyStageEntry/Exit() - debug validation (ASSERTIONS)     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  StageCoherence (GPU↔CPU memory synchronization)                    │    │
│  │    ├── cohereInputs(buffers, device) → ensureOnDevice()            │    │
│  │    ├── cohereOutputs(buffers, device) → allocate GPU buffers       │    │
│  │    ├── markOutputsDirty(buffers) → mark device as authoritative    │    │
│  │    └── CoherencePolicy: NONE, INPUT, OUTPUT, FULL                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  DeviceWorkspaceManager (per-device workspace allocation)           │    │
│  │    ├── Single contiguous block allocation at startup                │    │
│  │    ├── Named buffer suballocation at aligned offsets                │    │
│  │    ├── getBuffer(name), getBufferSize(name)                        │    │
│  │    └── Zero-allocation hot path during inference                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  TensorVerification (debug/integration builds only)                 │    │
│  │    ├── NaN/Inf detection at stage boundaries                       │    │
│  │    ├── Zero-tensor detection with configurable failure             │    │
│  │    ├── Auto-dump on verification failure                           │    │
│  │    └── VerificationFailure exception with full context             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPUTE STAGE LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  IComputeStage (base interface for all stages)                       │    │
│  │    ├── execute(ctx) → run computation on device                     │    │
│  │    ├── type() → ComputeStageType enum                               │    │
│  │    ├── getDumpInfo() → StageDumpInfo (inputs/outputs/weights)       │    │
│  │    ├── getBufferRequirements() → StageBufferRequirements            │    │
│  │    └── coherencePolicy() → CoherencePolicy for auto-sync            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐   │
│  │   GEMM Stages │ │Attention Stage│ │  Norm Stages  │ │Collective Stg │   │
│  ├───────────────┤ ├───────────────┤ ├───────────────┤ ├───────────────┤   │
│  │ GEMMStage     │ │ FusedAttnWo   │ │ RMSNormStage  │ │ AllreduceStage│   │
│  │ FusedQKVGEMM  │ │ AttnCompute   │ │ EmbeddingStage│ │ AllGatherStage│   │
│  │ FusedGateUp   │ │ KVCacheAppend │ │ LMHeadStage   │ │               │   │
│  │ LMHeadStage   │ │ KVCacheGather │ │               │ │               │   │
│  └───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘   │
│                                                                              │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐                      │
│  │ Activation Stg│ │Transform Stg  │ │ Utility Stages│                      │
│  ├───────────────┤ ├───────────────┤ ├───────────────┤                      │
│  │ SwiGLUStage   │ │ RoPEStage     │ │ResidualAddStg │                      │
│  │               │ │ QuantizeQ16_1 │ │               │                      │
│  └───────────────┘ └───────────────┘ └───────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           COLLECTIVE LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  CollectiveContext (ICollectiveContext)                              │    │
│  │    ├── executeAllreduce(buffer, count, device, op)                  │    │
│  │    ├── executeAllgather(local, full, seq_len, device)               │    │
│  │    ├── executeBroadcast(buffer, count, root, device)                │    │
│  │    └── Bridges abstract stages → concrete backends                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  BackendRouter (IBackendRouter)                                      │    │
│  │    ├── selectBackend(DeviceGroup) → BackendSelection               │    │
│  │    ├── getBackend(group) → ICollectiveBackend*                      │    │
│  │    └── executeHeterogeneousAllReduce() for mixed device groups     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Backends (ICollectiveBackend)                                       │    │
│  │    ├── NCCLBackend   - CUDA↔CUDA (NVLink/PCIe)                      │    │
│  │    ├── RCCLBackend   - ROCm↔ROCm (xGMI/PCIe)                        │    │
│  │    ├── MPIBackend    - Inter-node (Infiniband/Ethernet)             │    │
│  │    ├── HostBackend   - Heterogeneous fallback (CPU staging)         │    │
│  │    └── PCIeBARBackend- Direct CUDA↔ROCm via PCIe BAR mapping        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             KERNEL LAYER                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  KernelFactory (centralized kernel dispatch)                         │    │
│  │    ├── getDeviceType(DeviceId) → DeviceType                         │    │
│  │    ├── isActivationWeightCompatible(act_type, wgt_type) → bool      │    │
│  │    ├── createGemm<TensorType>(tensor, dev_type) → ITensorGemm       │    │
│  │    ├── createKVCache(config) → IKVCache                             │    │
│  │    └── CPU/CUDA/ROCm dispatch based on device type                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Kernel Interfaces                                                   │    │
│  │    ├── ITensorGemm - GEMM between activations and/or weights        │    │
│  │    ├── ITensorAttention - Full attention computation                │    │
│  │    ├── ITensorRoPE - Rotary position embeddings                     │    │
│  │    ├── ITensorSwiGLU - SwiGLU activation                            │    │
│  │    ├── ITensorRMSNorm - RMS normalization                           │    │
│  │    └── IWorkspaceConsumer - workspace buffer management interface   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐                      │
│  │  CPU Kernels  │ │ CUDA Kernels  │ │ ROCm Kernels  │                      │
│  ├───────────────┤ ├───────────────┤ ├───────────────┤                      │
│  │ OpenBLAS GEMM │ │ cuBLAS GEMM   │ │ rocBLAS GEMM  │                      │
│  │ MKL GEMM      │ │ FlashAttn     │ │ comp_kernel   │                      │
│  │ JIT Attention │ │ TensorCore    │ │ MatrixCore    │                      │
│  │ SIMD Prims    │ │ WMMA kernels  │ │ HIP kernels   │                      │
│  │ Quantized GEMM│ │ CUDAQuantised │ │ ROCmQuantised │                      │
│  └───────────────┘ └───────────────┘ └───────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             BACKEND LAYER                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  BackendManager (unified device memory interface)                    │    │
│  │    ├── getBackendFor(DeviceId) → IBackend*                          │    │
│  │    ├── getCPUBackend() → CPUBackend*                                │    │
│  │    ├── getCUDABackend() → CUDABackend* (if available)               │    │
│  │    └── getROCmBackend() → ROCmBackend* (if available)               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  IBackend (per-device-type memory management)                        │    │
│  │    ├── deviceToHost(), hostToDevice() - memory transfers            │    │
│  │    ├── allocate(), free() - device memory allocation                │    │
│  │    ├── deviceMemoryTotal(), deviceMemoryFree() - memory queries     │    │
│  │    └── synchronize() - device synchronization                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐                      │
│  │  CPUBackend   │ │ CUDABackend   │ │ ROCmBackend   │                      │
│  ├───────────────┤ ├───────────────┤ ├───────────────┤                      │
│  │ Rank-local    │ │ cudaMalloc    │ │ hipMalloc     │                      │
│  │ NUMA node     │ │ cudaMemcpy    │ │ hipMemcpy     │                      │
│  │ /sys/node     │ │ cudaMemGetInfo│ │ hipMemGetInfo │                      │
│  │ 64B align     │ │ 256B align    │ │ 256B align    │                      │
│  └───────────────┘ └───────────────┘ └───────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             TENSOR LAYER                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ITensor (runtime polymorphism interface)                            │    │
│  │    ├── native_type(), dtype_name() - type introspection             │    │
│  │    ├── shape(), rows(), cols(), numel(), size_bytes()               │    │
│  │    ├── home_device() → DeviceId (CPU, CUDA:0, ROCm:1)               │    │
│  │    └── typed_as<T>(), try_as<T>() - type-safe downcasting           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  CPUTensorBase (TensorBase alias) - base with device coherence       │    │
│  │    ├── ensureOnDevice(device) → upload to GPU if needed             │    │
│  │    ├── mark_device_dirty() → mark GPU as authoritative              │    │
│  │    ├── data() → host data (syncs from GPU if device-dirty)          │    │
│  │    └── mutable_data() → host data (marks CPU as authoritative)      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  TypedTensorBase<Derived, DataType> (CRTP for zero-overhead access)  │    │
│  │    ├── typed_data() → const DataType* (native storage access)       │    │
│  │    └── mutable_typed_data() → DataType* (mutable native access)     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────────────────┐ │
│  │ Activation    │ │ Quantized     │ │ Special Tensors                   │ │
│  │ Tensors       │ │ Weight Tensors│ │                                   │ │
│  ├───────────────┤ ├───────────────┤ ├───────────────────────────────────┤ │
│  │ FP32Tensor    │ │ IQ4_NLTensor  │ │ Q8_1Tensor  (activation quant)    │ │
│  │ FP16Tensor    │ │ Q4_0Tensor    │ │ Q16_1Tensor (KV cache quant)      │ │
│  │ BF16Tensor    │ │ Q6_KTensor    │ │ INT8Tensor  (dequant weights)     │ │
│  │ INT32Tensor   │ │ Q8_0Tensor    │ │ INT32Tensor (GEMM accumulator)    │ │
│  │               │ │ Q4_K, Q5_K... │ │                                   │ │
│  └───────────────┘ └───────────────┘ └───────────────────────────────────┘ │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  IActivationTensor (interface for mutable activations)               │    │
│  │    ├── createRoPE(), createSwiGLU(), createSoftmax(), createRMSNorm()│    │
│  │    ├── applyRMSNorm(), applyRoPE() - in-place transformations       │    │
│  │    └── to_int8_activation_pack() - per-row quantization for INT8 GEMM│    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ITensorGemmTileDataProvider (strategy pattern for quantized decode) │    │
│  │    ├── decode_block_at(row, k_offset, output) - format-specific     │    │
│  │    ├── block_size() → 32 elements for most formats                  │    │
│  │    └── Enables single generic GEMM kernel for all quant formats     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## MPI Orchestration and Topology Discovery

This section details how Llaminar coordinates across MPI ranks, discovers hardware topology, and makes placement decisions.

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TOPOLOGY & PLACEMENT COMPONENT FLOW                      │
│                                                                              │
│   MPI Bootstrap (llaminar2 --mpi-procs N)                                   │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────┐    MPI_Allgather     ┌──────────────┐                    │
│   │  MPIContext  │◀────────────────────▶│  MPIContext  │  (other ranks)     │
│   │  rank=0      │    hostname, gpus    │  rank=1..N   │                    │
│   └──────┬───────┘                      └──────────────┘                    │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │                        MPITopology                                │      │
│   │  Global view: world_size, ranks_per_node, all hostnames          │      │
│   │  Per-rank: node_rank, local_rank, numa_node, socket_id           │      │
│   │  Calls: NUMATopology, DeviceManager for local discovery          │      │
│   └──────┬───────────────────────────────────────────────────────────┘      │
│          │                                                                   │
│          ├─────────────────────┬──────────────────────────┐                 │
│          ▼                     ▼                          ▼                 │
│   ┌──────────────┐      ┌──────────────┐          ┌──────────────┐         │
│   │ NUMATopology │      │DeviceInventory│          │ DeviceGroup  │         │
│   │              │      │              │          │              │         │
│   │ • sockets    │      │ • cuda_count │          │ • ranks[]    │         │
│   │ • cores/sock │      │ • rocm_count │          │ • devices[]  │         │
│   │ • numa_nodes │      │ • gpu_memory │          │ • is_local   │         │
│   │ • cpu_bw_gbps│      │ • gpu_bw_gbps│          │ • DeviceIds  │         │
│   │ • DDR type   │      │ • vendor     │          │              │         │
│   └──────────────┘      └──────────────┘          └──────────────┘         │
│          │                     │                          │                 │
│          └─────────────────────┴──────────────────────────┘                 │
│                                │                                             │
│                                ▼                                             │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │                     PlacementInput                                │      │
│   │  Aggregates model info + topology for strategy decision          │      │
│   │  • n_layers, hidden_dim, bytes_per_layer                         │      │
│   │  • available_devices[], gpu_memory_bytes, cpu_memory_bandwidth   │      │
│   │  • gpu_memory_bandwidth (from DeviceInventory)                   │      │
│   │  • world_size (from MPITopology)                                 │      │
│   └──────┬───────────────────────────────────────────────────────────┘      │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │                   PlacementStrategy.compute()                     │      │
│   │  Strategies: CPUOnly, GPUFirst, HybridOptimal                    │      │
│   │                                                                   │      │
│   │  HybridOptimal Algorithm:                                        │      │
│   │    1. Calculate GPU memory budget vs model size                  │      │
│   │    2. Assign layers to GPUs (most layers fit in VRAM)            │      │
│   │    3. Overflow layers go to CPU (if any)                         │      │
│   │    4. For DECODE phase: compute bandwidth-proportional weights   │      │
│   │       → GPU + CPU both participate, weighted by memory BW        │      │
│   └──────┬───────────────────────────────────────────────────────────┘      │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │                       PlacementPlan                               │      │
│   │  Final output consumed by GraphOrchestrator                      │      │
│   │                                                                   │      │
│   │  LayerPlacement (per layer):                                     │      │
│   │    • device: PlacementDevice (CPU or GPU:N for PREFILL)          │      │
│   │    • decode_devices[]: [GPU:0, CPU:0] (DECODE participants)      │      │
│   │    • decode_weight_fractions[]: [0.80, 0.20] (bandwidth ratio)   │      │
│   │    • cpu_participates_in_decode: true/false                      │      │
│   │                                                                   │      │
│   │  GlobalTensorPlacement:                                          │      │
│   │    • embedding_device, lm_head_device, final_norm_device         │      │
│   │    • shard_embedding, shard_lm_head (for large vocab)            │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Startup Sequence (Per-Rank)

```cpp
// Each MPI rank executes this initialization sequence:

void initialize_topology() {
    // 1. NUMATopology: Parse /proc/cpuinfo, detect memory config
    NUMATopology numa;
    int sockets = numa.getNumSockets();           // e.g., 2
    int cores_per_socket = numa.getCoresPerSocket(); // e.g., 28
    auto cpu_bw = NUMATopology::estimateCPUBandwidth();
    //   cpu_bw = { bandwidth_gbps: 307.2, memory_channels: 8 }  (Xeon 8-ch DDR5)

    // 2. DeviceInventory: Query CUDA/ROCm devices
    DeviceInventory inventory = DeviceManager::getInventory();
    //   inventory = {
    //       cuda_devices: [{id: CUDA:0, memory: 24GB, bandwidth: 936 GB/s}],
    //       rocm_devices: [],
    //       cpu_bandwidth_gbps: 307.2  // from NUMATopology
    //   }

    // 3. MPITopology: Exchange topology with all ranks
    MPITopology topo(mpi_context);
    //   topo.world_size = 2
    //   topo.ranks_per_node = 2  (both ranks on same machine)
    //   topo.local_devices = [CPU:0, CUDA:0]
    //   topo.all_devices = [CPU:0, CUDA:0, CPU:1, ROCm:0, ROCm:1]

    // 4. DeviceGroup: Create groups for collective operations
    DeviceGroup local_group = DeviceGroup::createLocalAllDevicesGroup(topo);
    //   local_group.devices = [CUDA:0, CPU:0]  // rank 0's devices
    DeviceGroup global_gpu_group = DeviceGroup::createGlobalGPUGroup(topo);
    //   global_gpu_group.devices = [CUDA:0, ROCm:0, ROCm:1]  // all GPUs across ranks
}
```

### CPU Memory Bandwidth Detection

```cpp
// NUMATopology::estimateCPUBandwidth() implementation logic:

CPUBandwidthInfo estimateCPUBandwidth() {
    std::string cpu_model = parseCpuModel();  // From /proc/cpuinfo "model name"
    
    // Detect memory type and speed
    MemoryInfo mem = detectMemoryConfig();
    // mem = { ddr_type: DDR5, speed_mt: 4800, channels: 8 }
    
    // Calculate theoretical bandwidth
    // Formula: transfers_per_sec × 8 bytes × channels / 1000 = GB/s
    double bandwidth = (mem.speed_mt * 1e6) * 8 * mem.channels / 1e9;
    
    // Apply efficiency factor (~80% practical vs theoretical)
    bandwidth *= 0.8;
    
    return { bandwidth_gbps: bandwidth, memory_channels: mem.channels };
}

// Example results by CPU type:
//   Intel Xeon (8-ch DDR5-4800):  307 GB/s
//   AMD EPYC (12-ch DDR5-4800):   461 GB/s
//   Consumer (2-ch DDR5-5600):     72 GB/s
```

### Phase-Aware Device Selection

```cpp
// PlacementInput::getPhaseDeviceWeights() - Returns device participation weights

std::pair<float, float> getPhaseDeviceWeights(InferencePhase phase) const {
    if (phase == InferencePhase::PREFILL) {
        // Compute-bound: GPU only, CPU weight = 0
        return {1.0f, 0.0f};  // {gpu_weight, cpu_weight}
    }
    
    // DECODE: Bandwidth-bound, weight by memory bandwidth
    float total_bw = gpu_memory_bandwidth + cpu_memory_bandwidth;
    if (total_bw <= 0) return {1.0f, 0.0f};
    
    float gpu_weight = gpu_memory_bandwidth / total_bw;
    float cpu_weight = cpu_memory_bandwidth / total_bw;
    
    return {gpu_weight, cpu_weight};
    // Example (RTX 3090 936 GB/s + Xeon 8-ch 307 GB/s):
    //   → {0.75, 0.25}  (GPU handles 75%, CPU handles 25%)
}

bool cpuShouldParticipate(InferencePhase phase) const {
    if (phase == InferencePhase::PREFILL) return false;
    
    auto [gpu_w, cpu_w] = getPhaseDeviceWeights(InferencePhase::DECODE);
    return cpu_w >= 0.05f;  // CPU participates if ≥5% of total bandwidth
}
```

### DeviceId and PlacementDevice

```cpp
// DeviceId: Identifies a specific compute device
struct DeviceId {
    DeviceType type;   // CPU, CUDA, ROCm
    int index;         // Device index within type (0, 1, 2, ...)
    
    static DeviceId cpu(int idx = 0) { return {DeviceType::CPU, idx}; }
    static DeviceId cuda(int idx)    { return {DeviceType::CUDA, idx}; }
    static DeviceId rocm(int idx)    { return {DeviceType::ROCm, idx}; }
    
    std::string toString() const;  // "CPU:0", "CUDA:1", "ROCm:0"
};

// PlacementDevice: Wrapper used in PlacementPlan
struct PlacementDevice {
    bool is_cpu;
    int gpu_index;  // -1 for CPU
    
    static PlacementDevice cpu()      { return {true, -1}; }
    static PlacementDevice gpu(int i) { return {false, i}; }
    
    DeviceId toDeviceId(DeviceType gpu_type = DeviceType::CUDA) const;
};
```

### DeviceGroup for Collective Operations

```cpp
// DeviceGroup: Defines a set of devices that participate in a collective op

class DeviceGroup {
public:
    std::vector<int> ranks;          // MPI ranks involved
    std::vector<DeviceId> devices;   // Devices per rank
    
    bool isLocalOnly() const;        // All devices on same node?
    bool isHomogeneous() const;      // All same device type?
    bool containsCPU() const;        // Any CPU devices?
    
    // Factory methods
    static DeviceGroup createLocalAllDevicesGroup(const MPITopology& topo);
    static DeviceGroup createGlobalGPUGroup(const MPITopology& topo);
    static DeviceGroup createRankDeviceGroup(int rank, const std::vector<DeviceId>& devs);
};

// BackendRouter uses DeviceGroup to select collective backend:
//   - Local homogeneous GPUs → NCCL/RCCL
//   - Cross-node GPUs → MPI + NCCL
//   - Heterogeneous (CPU + GPU) → HostBackend (CPU staging)
```

---

## Memory Management and GPU Workspace Architecture

This section details the unified memory management system that enables efficient workspace buffer reuse across heterogeneous devices (CPU, CUDA, ROCm).

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MEMORY MANAGEMENT COMPONENT FLOW                         │
│                                                                              │
│   Model Load Complete (weights in VRAM/RAM)                                 │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │                    DeviceGraphBufferManager                             │      │
│   │  Central coordinator for buffer allocation and workspace budgets │      │
│   │                                                                   │      │
│   │  queryAvailableMemory(device):                                   │      │
│   │    → Calls BackendManager::getBackendFor(device)                 │      │
│   │    → Returns { total_bytes, free_bytes, usable_bytes }           │      │
│   │                                                                   │      │
│   │  computeWorkspaceBudget(device, config):                         │      │
│   │    → usable = free_bytes - headroom                              │      │
│   │    → budget = clamp(usable * fraction, min, max)                 │      │
│   │                                                                   │      │
│   │  allocateDeviceWorkspace(device, requirements):                     │      │
│   │    → Creates DeviceWorkspaceManager for the device                  │      │
│   │    → Pre-allocates contiguous block                              │      │
│   └──────┬───────────────────────────────────────────────────────────┘      │
│          │                                                                   │
│          ├────────────────────────────┬──────────────────────────┐          │
│          ▼                            ▼                          ▼          │
│   ┌──────────────────┐    ┌──────────────────────┐    ┌──────────────────┐ │
│   │ BackendManager   │    │ DeviceWorkspaceManager  │    │ WorkspaceBudget  │ │
│   │                  │    │   (per device)       │    │    Config        │ │
│   ├──────────────────┤    ├──────────────────────┤    ├──────────────────┤ │
│   │ getBackendFor()  │    │ • device: DeviceId   │    │ • gpu_fraction   │ │
│   │   → IBackend*    │    │ • budget_bytes       │    │   (default 0.8)  │ │
│   │                  │    │ • block_ptr          │    │ • cpu_fraction   │ │
│   │ getCPUBackend()  │    │ • named_buffers{}    │    │   (default 0.3)  │ │
│   │ getCUDABackend() │    │                      │    │ • min_budget     │ │
│   │ getROCmBackend() │    │ allocate(reqs)       │    │   (64 MB)        │ │
│   │                  │    │ getBuffer(name)      │    │ • max_budget     │ │
│   └────────┬─────────┘    │ release()            │    │   (4 GB)         │ │
│            │              └──────────┬───────────┘    │ • headroom       │ │
│            │                         │                │   (128 MB)       │ │
│            ▼                         │                └──────────────────┘ │
│   ┌──────────────────────────────────┴───────────────────────────────┐     │
│   │                        IBackend Interface                         │     │
│   │                                                                   │     │
│   │  Methods (unified for CPU, CUDA, ROCm):                          │     │
│   │    • deviceCount() → int                                         │     │
│   │    • deviceMemoryTotal(device_id) → size_t                       │     │
│   │    • deviceMemoryFree(device_id) → size_t                        │     │
│   │    • allocate(device_id, size, alignment) → void*                │     │
│   │    • free(device_id, ptr)                                        │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│            │                                                                │
│            ├─────────────────────────┬────────────────────────┐            │
│            ▼                         ▼                        ▼            │
│   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    │
│   │   CPUBackend     │    │   CUDABackend    │    │   ROCmBackend    │    │
│   │                  │    │                  │    │                  │    │
│   │ Rank-local NUMA  │    │ cudaMalloc/Free  │    │ hipMalloc/Free   │    │
│   │ /sys/node/nodeN  │    │ cudaMemGetInfo   │    │ hipMemGetInfo    │    │
│   │ 64-byte align    │    │ 256-byte align   │    │ 256-byte align   │    │
│   │ deviceCount()→1  │    │ deviceCount()→N  │    │ deviceCount()→M  │    │
│   └──────────────────┘    └──────────────────┘    └──────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### GPU Workspace Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      WORKSPACE LIFECYCLE FLOW                                │
│                                                                              │
│   1. BUDGET COMPUTATION (after model load)                                  │
│   ─────────────────────────────────────────                                 │
│                                                                              │
│   DeviceGraphBufferManager::computeWorkspaceBudget(CUDA:0, config)                │
│          │                                                                   │
│          ▼                                                                   │
│   ┌─────────────────────────────────────────────────────────┐               │
│   │  IBackend::deviceMemoryFree(0) → 18.5 GB (after weights)│               │
│   │  usable = 18.5 GB - 128 MB headroom = 18.37 GB          │               │
│   │  budget = clamp(18.37 GB * 0.8, 64 MB, 4 GB) = 4 GB     │               │
│   └─────────────────────────────────────────────────────────┘               │
│                                                                              │
│   2. REQUIREMENTS AGGREGATION                                               │
│   ───────────────────────────────                                           │
│                                                                              │
│   Kernels declare workspace needs via IWorkspaceConsumer:                │
│                                                                              │
│   ┌─────────────────────────┐    ┌─────────────────────────┐               │
│   │  ROCmQuantisedGemmKernel│    │  CUDAQuantisedGemmKernel│               │
│   │                         │    │                         │               │
│   │  getWorkspaceRequirements():│  getWorkspaceRequirements():             │
│   │  ┌───────────────────┐  │    │  ┌───────────────────┐  │               │
│   │  │ "fp16_activations"│  │    │  │ "fp16_activations"│  │               │
│   │  │   M×K×2 bytes     │  │    │  │   M×K×2 bytes     │  │               │
│   │  │ "fp16_output"     │  │    │  │ "fp16_output"     │  │               │
│   │  │   M×N×2 bytes     │  │    │  │   M×N×2 bytes     │  │               │
│   │  └───────────────────┘  │    │  └───────────────────┘  │               │
│   └─────────────────────────┘    └─────────────────────────┘               │
│                                                                              │
│   3. ALLOCATION (single contiguous block)                                   │
│   ────────────────────────────────────────                                  │
│                                                                              │
│   DeviceWorkspaceManager::allocate(requirements):                              │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                 CONTIGUOUS GPU MEMORY BLOCK                          │   │
│   │                     (single allocation)                              │   │
│   │                                                                      │   │
│   │  ┌──────────────────┐ ┌──────────────────┐ ┌─────────────────────┐  │   │
│   │  │ fp16_activations │ │ fp16_output      │ │ (unused headroom)   │  │   │
│   │  │   offset: 0      │ │   offset: 17MB   │ │                     │  │   │
│   │  │   size: 17MB     │ │   size: 17MB     │ │                     │  │   │
│   │  │   align: 256     │ │   align: 256     │ │                     │  │   │
│   │  └──────────────────┘ └──────────────────┘ └─────────────────────┘  │   │
│   │  ◄────────────────── total_required: 34MB ─────────────────────────►│   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   4. BINDING (kernels receive workspace)                                    │
│   ──────────────────────────────────────                                    │
│                                                                              │
│   kernel.bindWorkspace(&workspace_manager):                                 │
│     → kernel.hasWorkspace() = true                                          │
│     → kernel uses getBuffer("fp16_activations") instead of internal alloc   │
│                                                                              │
│   5. INFERENCE (zero-allocation hot path)                                   │
│   ────────────────────────────────────────                                  │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │   for each token:                                                    │   │
│   │       kernel->multiply(input, output, M, N, K)                       │   │
│   │       // Uses pre-allocated workspace buffers                        │   │
│   │       // NO hipMalloc/cudaMalloc calls during inference              │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### CPUBackend Rank-Local Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CPUBackend RANK-LOCAL ARCHITECTURE                        │
│                                                                              │
│   2-socket machine with 2 MPI ranks (each bound to one socket):             │
│                                                                              │
│   ┌─────────────────────────────────┬─────────────────────────────────┐     │
│   │        Socket 0 / NUMA Node 0   │        Socket 1 / NUMA Node 1   │     │
│   │            (MPI Rank 0)         │            (MPI Rank 1)         │     │
│   │                                 │                                 │     │
│   │  ┌───────────────────────────┐  │  ┌───────────────────────────┐  │     │
│   │  │  CPUBackend(numa_node=0)  │  │  │  CPUBackend(numa_node=1)  │  │     │
│   │  │                           │  │  │                           │  │     │
│   │  │  deviceCount() → 1        │  │  │  deviceCount() → 1        │  │     │
│   │  │  backendName() → "CPU"    │  │  │  backendName() → "CPU"    │  │     │
│   │  │  deviceName(0) →          │  │  │  deviceName(0) →          │  │     │
│   │  │    "CPU:NUMA0"            │  │  │    "CPU:NUMA1"            │  │     │
│   │  │                           │  │  │                           │  │     │
│   │  │  Memory Queries:          │  │  │  Memory Queries:          │  │     │
│   │  │  (reads /sys/node/node0)  │  │  │  (reads /sys/node/node1)  │  │     │
│   │  │                           │  │  │                           │  │     │
│   │  │  deviceMemoryTotal(0)     │  │  │  deviceMemoryTotal(0)     │  │     │
│   │  │    → 64 GB (half of 128GB)│  │  │    → 64 GB (half of 128GB)│  │     │
│   │  │  deviceMemoryFree(0)      │  │  │  deviceMemoryFree(0)      │  │     │
│   │  │    → 58 GB                │  │  │    → 60 GB                │  │     │
│   │  │                           │  │  │                           │  │     │
│   │  │  allocate(0, size, align) │  │  │  allocate(0, size, align) │  │     │
│   │  │    → numa_alloc_onnode()  │  │  │    → numa_alloc_onnode()  │  │     │
│   │  │       or aligned_alloc()  │  │  │       or aligned_alloc()  │  │     │
│   │  └───────────────────────────┘  │  └───────────────────────────┘  │     │
│   │                                 │                                 │     │
│   └─────────────────────────────────┴─────────────────────────────────┘     │
│                                                                              │
│   Why Rank-Local?                                                           │
│   ────────────────                                                          │
│   1. Consistent API: deviceCount()=1 like single-GPU systems               │
│   2. NUMA-aware: Memory queries return local socket's memory               │
│   3. Unified access: BackendManager::getBackendFor(CPU:0) works            │
│   4. No cross-socket memory: Each rank only sees its local NUMA node       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Kernel Dual-Mode Operation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KERNEL WORKSPACE DUAL-MODE DESIGN                         │
│                                                                              │
│   Kernels implementing IWorkspaceConsumer support both modes:            │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    IWorkspaceConsumer Interface                   │   │
│   │                                                                      │   │
│   │  • getWorkspaceRequirements() → WorkspaceRequirements            │   │
│   │  • bindWorkspace(DeviceWorkspaceManager*) → void                       │   │
│   │  • hasWorkspace() → bool                                            │   │
│   │  • getWorkspace() → DeviceWorkspaceManager*                            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────┐    ┌─────────────────────────────┐       │
│   │     LEGACY MODE             │    │     MANAGED MODE            │       │
│   │     (hasWorkspace()=false)  │    │     (hasWorkspace()=true)   │       │
│   ├─────────────────────────────┤    ├─────────────────────────────┤       │
│   │                             │    │                             │       │
│   │  On each multiply() call:   │    │  On bindWorkspace() call:   │       │
│   │                             │    │    workspace_ = mgr         │       │
│   │  void* fp16_act;            │    │                             │       │
│   │  hipMalloc(&fp16_act, ...)  │    │  On each multiply() call:   │       │
│   │                             │    │                             │       │
│   │  // ... compute ...         │    │  void* fp16_act =           │       │
│   │                             │    │    workspace_->getBuffer(   │       │
│   │  hipFree(fp16_act)          │    │      "fp16_activations")    │       │
│   │                             │    │                             │       │
│   │  Problem:                   │    │  // ... compute ...         │       │
│   │  ~17MB alloc per call       │    │                             │       │
│   │  Allocation overhead: ~1ms  │    │  // No free needed!         │       │
│   │                             │    │                             │       │
│   │                             │    │  Benefit:                   │       │
│   │                             │    │  Zero allocation hot path   │       │
│   │                             │    │  Allocation overhead: 0ms   │       │
│   └─────────────────────────────┘    └─────────────────────────────┘       │
│                                                                              │
│   Usage in DeviceGraphExecutor:                                                   │
│   ───────────────────────                                                   │
│                                                                              │
│   // At graph construction time (once)                                      │
│   DeviceGraphBufferManager gbm;                                                   │
│   auto budget = gbm.computeWorkspaceBudget(CUDA:0, config);                 │
│   auto reqs = kernel->getWorkspaceRequirements();                           │
│   auto mgr = gbm.allocateDeviceWorkspace(CUDA:0, reqs);                        │
│   kernel->bindWorkspace(mgr.get());                                         │
│                                                                              │
│   // At inference time (hot path)                                           │
│   for (token in sequence) {                                                 │
│       kernel->multiply(input, output, M, N, K);  // Zero allocations!       │
│   }                                                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Slab GEMM Configuration

For large FP16 GEMM operations that exceed workspace budget, the slab GEMM approach chunks the computation:

```cpp
// SlabGemmConfig: Budget-based slab dimension calculation

struct SlabGemmConfig {
    size_t slab_M;        // Rows per slab chunk
    size_t slab_N;        // Output columns (full)
    size_t slab_K;        // Input columns (full)
    size_t num_slabs;     // Total chunks needed
    
    // Factory methods
    static SlabGemmConfig fromBudget(size_t M, size_t N, size_t K, size_t budget_bytes);
    static SlabGemmConfig forDecode(size_t N, size_t K, size_t budget_bytes);   // M=1
    static SlabGemmConfig forPrefill(size_t seq_len, size_t N, size_t K, size_t budget_bytes);
};

// Example: 7B model attention projection
//   M=512 (sequence length), N=3584 (hidden), K=3584 (hidden)
//   Full FP16: 512×3584×2 + 3584×3584×2 = ~29 MB activation + ~26 MB output = ~55 MB
//   Budget: 32 MB
//   Slab config: slab_M=256, num_slabs=2 → processes 256 rows at a time
```

---

## Scenario 1: Single Machine, 2 Ranks, Heterogeneous

### Hardware Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SINGLE MACHINE                                       │
│                                                                              │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐   │
│  │       SOCKET 0 (NUMA Node 0)    │  │       SOCKET 1 (NUMA Node 1)    │   │
│  │         MPI Rank 0              │  │         MPI Rank 1              │   │
│  │                                 │  │                                 │   │
│  │  ┌─────────────────────────┐   │  │  ┌─────────────────────────┐   │   │
│  │  │    Xeon CPU (28 cores)  │   │  │  │    Xeon CPU (28 cores)  │   │   │
│  │  │    DeviceId: CPU:0      │   │  │  │    DeviceId: CPU:1      │   │   │
│  │  └─────────────────────────┘   │  │  └─────────────────────────┘   │   │
│  │              │                  │  │              │                  │   │
│  │              │ PCIe 4.0 x16     │  │              │ PCIe 4.0 x16     │   │
│  │              ▼                  │  │              ▼                  │   │
│  │  ┌─────────────────────────┐   │  │  ┌───────────┐ ┌───────────┐   │   │
│  │  │   NVIDIA RTX 3090       │   │  │  │ AMD Mi50  │ │ AMD Mi50  │   │   │
│  │  │   24GB VRAM             │   │  │  │ 16GB VRAM │ │ 16GB VRAM │   │   │
│  │  │   DeviceId: CUDA:0      │   │  │  │ ROCm:0    │ │ ROCm:1    │   │   │
│  │  └─────────────────────────┘   │  │  └───────────┘ └───────────┘   │   │
│  │                                 │  │        │ xGMI Interconnect │    │   │
│  └─────────────────────────────────┘  └─────────────────────────────────┘   │
│                    │                              │                          │
│                    └──────────────────────────────┘                          │
│                           QPI / UPI Interconnect                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Device Discovery (MPITopology + NUMATopology + DeviceManager)

```cpp
// On startup, each MPI rank discovers its local devices via the topology stack:

// Step 1: NUMATopology detects CPU configuration
NUMATopology numa;
// numa.getNumSockets() = 1 (per rank, bound to one socket)
// numa.getCoresPerSocket() = 28
// numa.estimateCPUBandwidth() = { 153.6 GB/s, 4 channels }  // Half of 8-ch for one socket

// Step 2: DeviceManager queries GPUs
DeviceInventory inv = DeviceManager::getInventory();
// Rank 0: inv.cuda_devices = [{ CUDA:0, 24GB, 936 GB/s }]
// Rank 1: inv.rocm_devices = [{ ROCm:0, 16GB, 717 GB/s }, { ROCm:1, 16GB, 717 GB/s }]

// Step 3: MPITopology aggregates and exchanges via MPI_Allgather
MPITopology topo(mpi_ctx);

// Rank 0 sees:
topo.rank = 0;
topo.local_devices = { CPU:0, CUDA:0 };
topo.device_capabilities[CPU:0] = { bandwidth_gbps: 153.6, memory_bytes: 128GB };
topo.device_capabilities[CUDA:0] = { bandwidth_gbps: 936, memory_bytes: 24GB };

// Rank 1 sees:
topo.rank = 1;
topo.local_devices = { CPU:1, ROCm:0, ROCm:1 };
topo.device_capabilities[CPU:1] = { bandwidth_gbps: 153.6, memory_bytes: 128GB };
topo.device_capabilities[ROCm:0] = { bandwidth_gbps: 717, memory_bytes: 16GB };
topo.device_capabilities[ROCm:1] = { bandwidth_gbps: 717, memory_bytes: 16GB };

// Step 4: PlacementInput aggregates for strategy
PlacementInput input = {
    .n_layers = 32,
    .bytes_per_layer = 410_MB,  // Qwen2.5-7B Q4_0
    .available_devices = topo.local_devices,
    .gpu_memory_bandwidth = 936,   // Rank 0: CUDA:0
    .cpu_memory_bandwidth = 153.6, // Rank 0: half of dual-socket
};

// Phase-aware weights:
input.getPhaseDeviceWeights(PREFILL) = { 1.0, 0.0 };   // GPU only
input.getPhaseDeviceWeights(DECODE)  = { 0.86, 0.14 }; // 936/(936+153.6), 153.6/(936+153.6)
input.cpuShouldParticipate(DECODE)   = true;           // 14% > 5% threshold
```

### Weight Sharding (Megatron-Style Tensor Parallelism)

```
Model: Qwen2.5-7B (32 layers, hidden_dim=3584, intermediate_dim=18944)

Weight Sharding across 2 MPI Ranks (TP=2):
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────┐  ┌─────────────────────────────────────┐
│           MPI RANK 0                 │  │           MPI RANK 1                 │
│  (loads first half of sharded dims) │  │ (loads second half of sharded dims) │
├─────────────────────────────────────┤  ├─────────────────────────────────────┤
│                                     │  │                                     │
│  token_embd: [151936, 3584] FULL    │  │  token_embd: [151936, 3584] FULL    │
│                                     │  │                                     │
│  Per-Layer (×32):                   │  │  Per-Layer (×32):                   │
│  ┌─────────────────────────────┐   │  │  ┌─────────────────────────────┐   │
│  │ attn_q:  [3584, 1792]       │   │  │  │ attn_q:  [3584, 1792]       │   │
│  │ attn_k:  [3584, 256]        │   │  │  │ attn_k:  [3584, 256]        │   │
│  │ attn_v:  [3584, 256]        │   │  │  │ attn_v:  [3584, 256]        │   │
│  │ attn_wo: [1792, 3584]       │   │  │  │ attn_wo: [1792, 3584]       │   │
│  │ ffn_gate:[3584, 9472]       │   │  │  │ ffn_gate:[3584, 9472]       │   │
│  │ ffn_up:  [3584, 9472]       │   │  │  │ ffn_up:  [3584, 9472]       │   │
│  │ ffn_down:[9472, 3584]       │   │  │  │ ffn_down:[9472, 3584]       │   │
│  │ attn_norm, ffn_norm: FULL   │   │  │  │ attn_norm, ffn_norm: FULL   │   │
│  └─────────────────────────────┘   │  │  └─────────────────────────────┘   │
│                                     │  │                                     │
│  output: [3584, 75968]  (half)     │  │  output: [3584, 75968]  (half)     │
│                                     │  │                                     │
└─────────────────────────────────────┘  └─────────────────────────────────────┘
```

### Intra-Rank Device Parallelism (PlacementStrategy)

Within each rank, work is distributed across local devices. **Critically, CPUs are first-class compute participants during decode, not just staging buffers.**

#### Memory Bandwidth Reality Check

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MEMORY BANDWIDTH COMPARISON                               │
│                                                                              │
│  Device                    │ Memory BW   │ Decode Token/s (7B, Q4_0)        │
│  ──────────────────────────┼─────────────┼──────────────────────────────────│
│  RTX 3090 (24GB GDDR6X)    │  936 GB/s   │  ~200 tok/s                      │
│  AMD Mi50 (16GB HBM2)      │  717 GB/s   │  ~150 tok/s                      │
│  AMD 7900 XTX (24GB GDDR6) │  960 GB/s   │  ~200 tok/s                      │
│  Xeon 6c DDR5-4800 (1 sock)│  230 GB/s   │  ~50 tok/s   ◄── NOT NEGLIGIBLE │
│  Xeon 8c DDR5-5600 (1 sock)│  358 GB/s   │  ~75 tok/s   ◄── SIGNIFICANT    │
│                                                                              │
│  Insight: For memory-bound decode, CPUs contribute meaningfully!            │
│  A dual-socket Xeon with 8 channels/socket = 716 GB/s = Mi50 equivalent    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Phase-Aware Placement Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PREFILL vs DECODE PLACEMENT                               │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        PREFILL PHASE                                   │  │
│  │                   (Compute-bound, batch of tokens)                     │  │
│  │                                                                        │  │
│  │   CPU: Idle (compute too slow for large matrix ops)                   │  │
│  │   GPU: All compute (GEMM, Attention, FFN)                             │  │
│  │                                                                        │  │
│  │   Rationale: Prefill is TFLOPS-bound. GPUs have 35-100× more TFLOPS. │  │
│  │   CPU participation would only slow things down.                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        DECODE PHASE                                    │  │
│  │                (Memory-bandwidth-bound, single token)                  │  │
│  │                                                                        │  │
│  │   CPU: PARTICIPATES with weight shard + compute                       │  │
│  │   GPU: Participates with weight shard + compute                       │  │
│  │                                                                        │  │
│  │   Rationale: Decode is GB/s-bound. CPUs have competitive bandwidth.  │  │
│  │   Every device with memory bandwidth should contribute.               │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Weight Placement for Phase-Aware Execution

**The Problem:** Weights must exist somewhere for compute to happen, but:
- PREFILL is compute-bound → CPU participation would bottleneck GPU (50-100× slower)
- DECODE is bandwidth-bound → CPU participation helps (contributes ~20% bandwidth)
- Weights can't magically move between phases

**Options Considered:**

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A: Duplicate** | GPU has 100% weights, CPU has decode shard | Clean phase separation, optimal perf | Memory overhead (~20% extra) |
| **B: Shuffle** | Move weights GPU↔CPU at phase transition | No duplication | PCIe latency (~2GB/s), complexity |
| **C: CPU always** | Sharded weights, CPU in both phases | Simplest | Prefill tanks (allreduce waits for slow CPU) |

**Why Option C is unacceptable:**
```
PREFILL 512 tokens on 7B model with CPU participating:
  - GPU GEMM for 75% of weights: ~10ms
  - CPU GEMM for 25% of weights: ~500ms (50× slower!)
  - Allreduce waits for slowest participant
  - Result: 50× prefill slowdown for ~20% decode speedup

  NOT WORTH IT.
```

**Selected: Option A (Selective Duplication)**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              WEIGHT PLACEMENT STRATEGY: SELECTIVE DUPLICATION                │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════   │
│  MODEL LOADING (happens once at startup):                                   │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│  GPU Memory:                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  100% of model weights (for prefill)                                │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ attn_q: [3584, 3584]  attn_k: [3584, 512]  attn_v: [3584, 512]│   │   │
│  │  │ attn_wo: [3584, 3584] ffn_gate: [3584, 18944] ...            │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │  + KV cache for prefill                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  CPU Memory:                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Decode shard ONLY (~20% of model, DUPLICATED from GPU portion)     │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ attn_q: [3584, 717]   attn_k: [3584, 102]  attn_v: [3584, 102]│  │   │
│  │  │ attn_wo: [717, 3584]  ffn_gate: [3584, 3789] ...             │   │   │
│  │  │ (These are the SAME weights as the last 20% on GPU)          │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │  + KV cache shard for decode (CPU's portion of heads)               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════   │
│  MEMORY OVERHEAD ANALYSIS (7B Q4_0 model):                                  │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│    Base model: ~4 GB                                                        │
│    GPU: 4 GB (100% of weights)                                              │
│    CPU: 0.8 GB (20% decode shard)                                           │
│    ──────────────────────────────────                                       │
│    Total: 4.8 GB                                                            │
│    Overhead: 20% (~800 MB on a 128GB+ server = negligible)                 │
│                                                                              │
│  For 70B model:                                                             │
│    Base: ~40 GB, CPU decode shard: ~8 GB                                    │
│    Still acceptable on high-memory servers                                  │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════   │
│  PHASE EXECUTION:                                                           │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│  PREFILL:                                                                   │
│    GPU executes alone using its 100% weight copy                           │
│    CPU is IDLE (no compute, no synchronization)                            │
│    → Full GPU speed, no CPU bottleneck                                     │
│                                                                              │
│  DECODE:                                                                    │
│    GPU uses first 80% of its weights (not all 100%)                        │
│    CPU uses its 20% shard (the duplicate)                                  │
│    Both compute in parallel → allreduce to combine                         │
│    → 20% more decode bandwidth utilized                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation Notes:**

1. **Loading Strategy:** 
   - GGUF loader reads weights once
   - GPU receives full copy
   - CPU receives tail slice (last `cpu_decode_fraction` of each weight dim)

2. **No Runtime Shuffling:**
   - Weights are placed at load time and never moved
   - Phase transition is instant (just different code paths)

3. **Alternative for Memory-Constrained Systems:**
   - If GPU VRAM is tight, use Option C (CPU always participates)
   - Accept prefill slowdown in exchange for fitting the model
   - Configuration: `LLAMINAR_CPU_PREFILL_PARTICIPATE=1`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MPI RANK 0 - INTRA-RANK PLACEMENT                         │
│                                                                              │
│  PlacementStrategy analyzes:                                                │
│    - RTX 3090: 24GB VRAM, 936 GB/s bandwidth, 35.6 TFLOPS FP32             │
│    - CPU: 28 cores, 230 GB/s bandwidth (6-ch DDR5), ~1 TFLOPS FP32         │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════   │
│  PREFILL: GPU-only (compute-bound)                                          │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      RTX 3090 (CUDA:0)                               │   │
│  │  100% of prefill compute:                                            │   │
│  │    • QKV projections (large batch GEMM)                             │   │
│  │    • Attention (Flash Attention, compute-heavy)                     │   │
│  │    • FFN (large batch GEMM)                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         CPU (CPU:0) - IDLE during prefill            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════   │
│  DECODE: ALL devices participate (bandwidth-bound)                          │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│  Weight sharding during decode (bandwidth-proportional):                    │
│    RTX 3090: 936 GB/s → 80% of weights (uses first 80% of its full copy)   │
│    CPU:      230 GB/s → 20% of weights (uses its dedicated decode shard)   │
│                                                                              │
│  Note: GPU has 100% of weights but only uses 80% during decode.            │
│  The last 20% is duplicated on CPU to avoid cross-device access.           │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      RTX 3090 (CUDA:0)                               │   │
│  │  Decode compute (80% of layer):                                      │   │
│  │    • QKV projection slice (80% of heads)                            │   │
│  │    • Attention for assigned heads                                   │   │
│  │    • FFN slice (80% of intermediate dim)                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              │ Allreduce partial sums                       │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         CPU (CPU:0)                                  │   │
│  │  Decode compute (20% of layer):                                      │   │
│  │    • QKV projection slice (20% of heads)                            │   │
│  │    • Attention for assigned heads (JIT kernels)                     │   │
│  │    • FFN slice (20% of intermediate dim)                            │   │
│  │    • RMSNorm, RoPE, Residuals (always on CPU after allreduce)       │   │
│  │    • Collective aggregation point                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MPI RANK 1 - INTRA-RANK PLACEMENT                         │
│                                                                              │
│  PlacementStrategy analyzes:                                                │
│    - Mi50 #1: 16GB VRAM, 717 GB/s HBM2 bandwidth, 13.4 TFLOPS FP32         │
│    - Mi50 #2: 16GB VRAM, 717 GB/s HBM2 bandwidth, 13.4 TFLOPS FP32         │
│    - CPU: 28 cores, 230 GB/s bandwidth (6-ch DDR5), ~1 TFLOPS FP32         │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════   │
│  PREFILL: GPUs only                                                         │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│  ┌──────────────────────────┐     ┌──────────────────────────┐             │
│  │     Mi50 #1 (ROCm:0)     │◄───►│     Mi50 #2 (ROCm:1)     │             │
│  │                          │xGMI │                          │             │
│  │  50% of prefill compute  │     │  50% of prefill compute  │             │
│  └──────────────────────────┘     └──────────────────────────┘             │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CPU (CPU:1) - IDLE during prefill                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════   │
│  DECODE: ALL devices participate (bandwidth-proportional sharding)          │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│  Weight sharding by bandwidth ratio:                                        │
│    Mi50 #1: 717 GB/s → 43% of weights                                      │
│    Mi50 #2: 717 GB/s → 43% of weights                                      │
│    CPU:     230 GB/s → 14% of weights                                      │
│                                                                              │
│  ┌──────────────────────────┐     ┌──────────────────────────┐             │
│  │     Mi50 #1 (ROCm:0)     │◄───►│     Mi50 #2 (ROCm:1)     │             │
│  │                          │xGMI │                          │             │
│  │  43% decode compute:     │     │  43% decode compute:     │             │
│  │    • QKV slice           │     │    • QKV slice           │             │
│  │    • Attention heads     │     │    • Attention heads     │             │
│  │    • FFN slice           │     │    • FFN slice           │             │
│  └──────────────────────────┘     └──────────────────────────┘             │
│              │                                │                             │
│              │ Intra-rank allreduce           │                             │
│              ▼                                ▼                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         CPU (CPU:1)                                  │   │
│  │  14% decode compute:                                                 │   │
│  │    • QKV slice (OpenBLAS GEMM)                                      │   │
│  │    • Attention heads (JIT Q8 kernels)                               │   │
│  │    • FFN slice (OpenBLAS GEMM)                                      │   │
│  │  Always on CPU:                                                      │   │
│  │    • RMSNorm, RoPE, Residuals                                       │   │
│  │    • Collective aggregation point                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Graph Construction (Scenario 1)

```cpp
// GraphOrchestrator::buildGraph() creates the compute DAG

ComputeGraph buildGraph(const Model& model, const GlobalTopology& topology) {
    ComputeGraph graph;
    
    // 1. Create DeviceGroups for collective operations
    DeviceGroup global_group = DeviceGroupFactory::createGlobal(topology);
    // global_group = {
    //   devices: [CPU:0, CUDA:0, CPU:1, ROCm:0, ROCm:1],
    //   scope: GLOBAL,
    //   local_rank: mpi_ctx->rank()
    // }
    
    DeviceGroup rank0_local = DeviceGroupFactory::createLocal(0, topology);
    // rank0_local = { devices: [CPU:0, CUDA:0], scope: LOCAL }
    
    DeviceGroup rank1_local = DeviceGroupFactory::createLocal(1, topology);
    // rank1_local = { devices: [CPU:1, ROCm:0, ROCm:1], scope: LOCAL }
    
    // 2. Build per-layer stages with device affinity
    for (int layer = 0; layer < model.n_layers; layer++) {
        // Each stage knows which device it runs on
        auto qkv_stage = createQKVProjectionStage(layer, primary_gpu);
        auto attn_stage = createAttentionStage(layer, primary_gpu);
        auto wo_stage = createWoProjectionStage(layer, primary_gpu);
        
        // ALLREDUCE after Wo (partial sums from TP sharding)
        auto wo_allreduce = std::make_unique<AllreduceStage>(
            wo_output_buffer, 
            hidden_dim,
            collective_ctx  // Uses new infrastructure!
        );
        
        // FFN stages...
        auto ffn_gate_up = createFFNGateUpStage(layer, primary_gpu);
        auto swiglu = createSwiGLUStage(layer, primary_gpu);
        auto ffn_down = createFFNDownStage(layer, primary_gpu);
        
        // ALLREDUCE after FFN down
        auto ffn_allreduce = std::make_unique<AllreduceStage>(
            ffn_output_buffer,
            hidden_dim,
            collective_ctx
        );
        
        // Norms on CPU (memory-bound, not worth GPU transfer)
        auto attn_norm = createRMSNormStage(layer, cpu_device);
        auto ffn_norm = createRMSNormStage(layer, cpu_device);
        
        // Wire dependencies
        graph.addEdge(attn_norm, qkv_stage);
        graph.addEdge(qkv_stage, attn_stage);
        graph.addEdge(attn_stage, wo_stage);
        graph.addEdge(wo_stage, wo_allreduce);
        graph.addEdge(wo_allreduce, residual_add);
        // ... etc
    }
    
    // 3. Final LM head with ALLGATHER
    auto lm_head = createLMHeadStage(primary_gpu);
    auto lm_allgather = std::make_unique<AllGatherStage>(
        local_logits,       // [batch, vocab/world_size]
        global_logits,      // [batch, vocab]
        vocab_size / world_size,
        collective_ctx
    );
    
    return graph;
}
```

### Collective Backend Selection (Scenario 1)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COLLECTIVE OPERATIONS IN SCENARIO 1                       │
│                                                                              │
│  Operation: ALLREDUCE after Wo projection                                   │
│  Participants: Rank 0 (CUDA:0 output) + Rank 1 (ROCm:0/1 outputs)          │
│                                                                              │
│  BackendRouter decision tree:                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. Is DeviceGroup homogeneous?                                      │   │
│  │    - Rank 0 has: CUDA                                               │   │
│  │    - Rank 1 has: ROCm                                               │   │
│  │    → NO, heterogeneous (CUDA + ROCm)                                │   │
│  │                                                                      │   │
│  │ 2. Can use vendor-specific backend?                                 │   │
│  │    - NCCLBackend: requires all CUDA → NO                            │   │
│  │    - RCCLBackend: requires all ROCm → NO                            │   │
│  │                                                                      │   │
│  │ 3. Is this inter-node?                                              │   │
│  │    - Same machine, different NUMA nodes                             │   │
│  │    → Could use MPI (shared memory transport)                        │   │
│  │                                                                      │   │
│  │ 4. Final selection: HostBackend                                     │   │
│  │    - Stages GPU data to CPU                                         │   │
│  │    - Performs allreduce on CPU                                      │   │
│  │    - Copies result back to GPUs                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Execution flow:                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │   CUDA:0 (Rank 0)              ROCm:0/1 (Rank 1)                    │   │
│  │   ┌──────────────┐             ┌──────────────┐                     │   │
│  │   │ partial_sum  │             │ partial_sum  │                     │   │
│  │   │ [1, 3584]    │             │ [1, 3584]    │                     │   │
│  │   └──────┬───────┘             └──────┬───────┘                     │   │
│  │          │ cudaMemcpyD2H              │ hipMemcpyD2H                │   │
│  │          ▼                            ▼                             │   │
│  │   ┌──────────────┐             ┌──────────────┐                     │   │
│  │   │  CPU:0 buf   │             │  CPU:1 buf   │                     │   │
│  │   │  (staging)   │             │  (staging)   │                     │   │
│  │   └──────┬───────┘             └──────┬───────┘                     │   │
│  │          │                            │                             │   │
│  │          └──────────┬─────────────────┘                             │   │
│  │                     │ MPI_Allreduce (shared mem)                    │   │
│  │                     ▼                                               │   │
│  │             ┌──────────────┐                                        │   │
│  │             │  reduced_sum │                                        │   │
│  │             │  [1, 3584]   │                                        │   │
│  │             └──────┬───────┘                                        │   │
│  │          ┌─────────┴─────────┐                                      │   │
│  │          ▼                   ▼                                      │   │
│  │   ┌──────────────┐    ┌──────────────┐                              │   │
│  │   │  CPU:0 buf   │    │  CPU:1 buf   │                              │   │
│  │   └──────┬───────┘    └──────┬───────┘                              │   │
│  │          │ cudaMemcpyH2D     │ hipMemcpyH2D                         │   │
│  │          ▼                   ▼                                      │   │
│  │   ┌──────────────┐    ┌──────────────┐                              │   │
│  │   │   CUDA:0     │    │  ROCm:0/1    │                              │   │
│  │   │   result     │    │   result     │                              │   │
│  │   └──────────────┘    └──────────────┘                              │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Execution Timeline (Scenario 1, Single Token Decode)

**Key insight: During decode, CPUs process their bandwidth-proportional weight shard IN PARALLEL with GPUs.**

```
Time ──────────────────────────────────────────────────────────────────────────►

     MPI Rank 0                           MPI Rank 1
     CUDA:0 (80%) + CPU:0 (20%)           ROCm:0 (43%) + ROCm:1 (43%) + CPU:1 (14%)
     ══════════════════════════           ══════════════════════════════════════════
     
t0   ┌─────────────────────────┐          ┌─────────────────────────┐
     │ Embedding lookup (CPU)  │          │ Embedding lookup (CPU)  │
     │ → distribute to devices │          │ → distribute to devices │
     └────────────┬────────────┘          └────────────┬────────────┘
                  │                                    │
     ╔════════════▼════════════╗          ╔════════════▼════════════╗
     ║    LAYER 0 (×32)        ║          ║    LAYER 0 (×32)        ║
     ╠═════════════════════════╣          ╠═════════════════════════╣
t1   ║ RMSNorm (CPU:0)         ║          ║ RMSNorm (CPU:1)         ║
     ║ (full hidden state)     ║          ║ (full hidden state)     ║
     ╟─────────────────────────╢          ╟─────────────────────────╢
     ║                         ║          ║                         ║
t2   ║ QKV Proj - PARALLEL:    ║          ║ QKV Proj - PARALLEL:    ║
     ║ ┌─────────┬─────────┐   ║          ║ ┌───────┬───────┬─────┐ ║
     ║ │ CUDA:0  │  CPU:0  │   ║          ║ │ROCm:0 │ROCm:1 │CPU:1│ ║
     ║ │  80%    │   20%   │   ║          ║ │ 43%   │ 43%   │ 14% │ ║
     ║ │ heads   │  heads  │   ║          ║ │heads  │heads  │heads│ ║
     ║ │ cuBLAS  │OpenBLAS │   ║          ║ │rocBLAS│rocBLAS│OBLAS│ ║
     ║ └────┬────┴────┬────┘   ║          ║ └──┬────┴──┬────┴──┬──┘ ║
     ╟──────┼─────────┼────────╢          ╟────┼───────┼───────┼────╢
     ║      │         │        ║          ║    │       │       │    ║
t3   ║ Attn - PARALLEL:        ║          ║ Attn - PARALLEL:        ║
     ║ ┌─────────┬─────────┐   ║          ║ ┌───────┬───────┬─────┐ ║
     ║ │ CUDA:0  │  CPU:0  │   ║          ║ │ROCm:0 │ROCm:1 │CPU:1│ ║
     ║ │ Flash   │JIT Q8   │   ║          ║ │ CK    │ CK    │JIT  │ ║
     ║ │ Attn    │Attn     │   ║          ║ │ Attn  │ Attn  │Attn │ ║
     ║ └────┬────┴────┬────┘   ║          ║ └──┬────┴──┬────┴──┬──┘ ║
     ╟──────┼─────────┼────────╢          ╟────┼───────┼───────┼────╢
     ║      │         │        ║          ║    │       │       │    ║
t4   ║ Wo Proj - PARALLEL:     ║          ║ Wo Proj - PARALLEL:     ║
     ║ ┌─────────┬─────────┐   ║          ║ ┌───────┬───────┬─────┐ ║
     ║ │ CUDA:0  │  CPU:0  │   ║          ║ │ROCm:0 │ROCm:1 │CPU:1│ ║
     ║ │ partial │ partial │   ║          ║ │partial│partial│part.│ ║
     ║ │  sum    │   sum   │   ║          ║ │ sum   │ sum   │sum  │ ║
     ║ └────┬────┴────┬────┘   ║          ║ └──┬────┴──┬────┴──┬──┘ ║
     ║      │         │        ║          ║    │       │       │    ║
     ║      └────┬────┘        ║          ║    └───┬───┴───┬───┘    ║
     ║           │ local reduce║          ║        │ local reduce   ║
     ╟───────────┼─────────────╢          ╟────────┼───────────────╢
t5   ║┌──────────▼────────────┐║          ║┌───────▼──────────────┐║
     ║│ ALLREDUCE (inter-rank) │◄─────────►│ ALLREDUCE (inter-rank)│║
     ║│ CPU:0 ↔ CPU:1 via MPI │║          ║│ CPU:0 ↔ CPU:1 via MPI│║
     ║└───────────────────────┘║          ║└──────────────────────┘║
     ╟─────────────────────────╢          ╟─────────────────────────╢
t6   ║ Residual Add (CPU:0)    ║          ║ Residual Add (CPU:1)    ║
     ╟─────────────────────────╢          ╟─────────────────────────╢
t7   ║ FFN Norm (CPU:0)        ║          ║ FFN Norm (CPU:1)        ║
     ╟─────────────────────────╢          ╟─────────────────────────╢
     ║                         ║          ║                         ║
t8   ║ FFN Gate+Up - PARALLEL: ║          ║ FFN Gate+Up - PARALLEL: ║
     ║ ┌─────────┬─────────┐   ║          ║ ┌───────┬───────┬─────┐ ║
     ║ │ CUDA:0  │  CPU:0  │   ║          ║ │ROCm:0 │ROCm:1 │CPU:1│ ║
     ║ │  80%    │   20%   │   ║          ║ │ 43%   │ 43%   │ 14% │ ║
     ║ │ inter.  │ inter.  │   ║          ║ │inter. │inter. │int. │ ║
     ║ └────┬────┴────┬────┘   ║          ║ └──┬────┴──┬────┴──┬──┘ ║
     ╟──────┼─────────┼────────╢          ╟────┼───────┼───────┼────╢
t9   ║ SwiGLU (per-device)     ║          ║ SwiGLU (per-device)     ║
     ╟─────────────────────────╢          ╟─────────────────────────╢
     ║                         ║          ║                         ║
t10  ║ FFN Down - PARALLEL:    ║          ║ FFN Down - PARALLEL:    ║
     ║ ┌─────────┬─────────┐   ║          ║ ┌───────┬───────┬─────┐ ║
     ║ │ CUDA:0  │  CPU:0  │   ║          ║ │ROCm:0 │ROCm:1 │CPU:1│ ║
     ║ │ partial │ partial │   ║          ║ │partial│partial│part.│ ║
     ║ └────┬────┴────┬────┘   ║          ║ └──┬────┴──┬────┴──┬──┘ ║
     ║      └────┬────┘        ║          ║    └───┬───┴───┬───┘    ║
     ╟───────────┼─────────────╢          ╟────────┼───────────────╢
t11  ║┌──────────▼────────────┐║          ║┌───────▼──────────────┐║
     ║│ ALLREDUCE (inter-rank) │◄─────────►│ ALLREDUCE (inter-rank)│║
     ║└───────────────────────┘║          ║└──────────────────────┘║
     ╟─────────────────────────╢          ╟─────────────────────────╢
t12  ║ Residual Add (CPU:0)    ║          ║ Residual Add (CPU:1)    ║
     ╚═════════════════════════╝          ╚═════════════════════════╝
                  │ (repeat ×32 layers)                │
                  ▼                                    ▼
t13  ┌─────────────────────────┐          ┌─────────────────────────┐
     │ Final RMSNorm (CPU:0)   │          │ Final RMSNorm (CPU:1)   │
     └────────────┬────────────┘          └────────────┬────────────┘
                  │                                    │
t14  ┌────────────▼────────────┐          ┌────────────▼────────────┐
     │ LM Head - PARALLEL:     │          │ LM Head - PARALLEL:     │
     │ CUDA:0 (80%) + CPU (20%)│          │ ROCm×2 (86%) + CPU (14%)│
     │ → partial vocab each    │          │ → partial vocab each    │
     └────────────┬────────────┘          └────────────┬────────────┘
                  │ local reduce                       │ local reduce
t15  ┌────────────▼────────────┐          ┌────────────▼────────────┐
     │ ALLGATHER (inter-rank)  │◄─────────►│ ALLGATHER (inter-rank)  │
     │ [75968]→[151936] full   │          │ [75968]→[151936] full   │
     └────────────┬────────────┘          └────────────┬────────────┘
                  │                                    │
t16  ┌────────────▼────────────┐          
     │ Argmax → next token     │          (Rank 0 broadcasts token)
     │ (Rank 0 only)           │          
     └─────────────────────────┘          
```

### Bandwidth Utilization Analysis (Scenario 1)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DECODE THROUGHPUT PROJECTION                              │
│                                                                              │
│  Without CPU participation (GPU-only):                                      │
│    Rank 0: RTX 3090 = 936 GB/s → ~200 tok/s capacity                       │
│    Rank 1: 2× Mi50  = 1434 GB/s → ~300 tok/s capacity                      │
│    Bottleneck: Rank 0 at 200 tok/s (limited by single GPU)                 │
│    System throughput: 200 tok/s                                             │
│                                                                              │
│  With CPU participation:                                                    │
│    Rank 0: RTX 3090 + CPU = 936 + 230 = 1166 GB/s → ~250 tok/s capacity   │
│    Rank 1: 2× Mi50 + CPU = 1434 + 230 = 1664 GB/s → ~350 tok/s capacity   │
│    Bottleneck: Rank 0 at 250 tok/s                                         │
│    System throughput: 250 tok/s (+25% improvement!)                        │
│                                                                              │
│  Key insight: CPU memory bandwidth is not "free" - we're already paying   │
│  for it. Using it for decode reduces the imbalance between ranks.          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Scenario 2: Six Machines, 12 Ranks, Complex Topology

### Hardware Overview

```
6 Dual-Socket Xeon Machines × Infiniband RDMA
═══════════════════════════════════════════════════════════════════════════════

Total: 12 MPI Ranks (2 per machine, 1 per socket)
Total GPUs: 60 devices (mix of NVIDIA RTX 3090, AMD Mi50, AMD 7900XTX)
Total CPUs: 12 sockets (first-class decode participants!)

┌─────────────────────────────────────────────────────────────────────────────┐
│                            INFINIBAND FABRIC                                 │
│                         (200 Gb/s per port RDMA)                            │
│                                                                              │
│    ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐   │
│    │Node 0│    │Node 1│    │Node 2│    │Node 3│    │Node 4│    │Node 5│   │
│    │R0,R1 │    │R2,R3 │    │R4,R5 │    │R6,R7 │    │R8,R9 │    │R10,11│   │
│    └──┬───┘    └──┬───┘    └──┬───┘    └──┬───┘    └──┬───┘    └──┬───┘   │
│       │           │           │           │           │           │        │
│       └───────────┴───────────┴─────┬─────┴───────────┴───────────┘        │
│                                     │                                       │
│                            IB Switch Fabric                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Per-Node Device Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NODE 0 (Representative of all 6 nodes)                    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    SOCKET 0 (MPI Rank 0, 2, 4, 6, 8, 10)            │    │
│  │                                                                      │    │
│  │   ┌─────────────┐                                                   │    │
│  │   │ Xeon CPU    │  28 cores, DeviceId: CPU:0                        │    │
│  │   └──────┬──────┘                                                   │    │
│  │          │                                                           │    │
│  │          │ PCIe Root Complex                                         │    │
│  │          ├───────────────────────────────────────────────────┐      │    │
│  │          │                                                    │      │    │
│  │   ┌──────▼──────┐  ┌──────────────┐  ┌──────────────┐        │      │    │
│  │   │ RTX 3090 #1 │  │ RTX 3090 #2  │  │ RTX 3090 #3  │        │      │    │
│  │   │ CUDA:0      │  │ CUDA:1       │  │ CUDA:2       │        │      │    │
│  │   │ 24GB        │  │ 24GB         │  │ 24GB         │        │      │    │
│  │   └─────────────┘  └──────────────┘  └──────────────┘        │      │    │
│  │         │ NVLink 3.0 (if present, else PCIe peer)            │      │    │
│  │   ┌─────▼───────┐  ┌──────────────┐  ┌──────────────┐        │      │    │
│  │   │ AMD Mi50 #1 │  │ AMD Mi50 #2  │  │ AMD Mi50 #3  │        │      │    │
│  │   │ ROCm:0      │  │ ROCm:1       │  │ ROCm:2       │        │      │    │
│  │   │ 16GB        │  │ 16GB         │  │ 16GB         │        │      │    │
│  │   └─────────────┘  └──────────────┘  └──────────────┘        │      │    │
│  │                                                               │      │    │
│  │   Socket 0 Total: 3× RTX 3090 (72GB) + 3× Mi50 (48GB) = 120GB│      │    │
│  │                                                               │      │    │
│  └───────────────────────────────────────────────────────────────┘      │    │
│                                                                          │    │
│  ┌───────────────────────────────────────────────────────────────┐      │    │
│  │                    SOCKET 1 (MPI Rank 1, 3, 5, 7, 9, 11)      │      │    │
│  │                                                                │      │    │
│  │   ┌─────────────┐                                             │      │    │
│  │   │ Xeon CPU    │  28 cores, DeviceId: CPU:1                  │      │    │
│  │   └──────┬──────┘                                             │      │    │
│  │          │                                                     │      │    │
│  │          │ PCIe Root Complex                                   │      │    │
│  │          ├────────────────────────────────────────────────────┤      │    │
│  │          │                                                     │      │    │
│  │   ┌──────▼──────┐  ┌──────────────┐  ┌──────────────┐        │      │    │
│  │   │ AMD Mi50 #4 │  │ AMD Mi50 #5  │  │ AMD Mi50 #6  │        │      │    │
│  │   │ ROCm:3      │  │ ROCm:4       │  │ ROCm:5       │        │      │    │
│  │   │ 16GB        │  │ 16GB         │  │ 16GB         │        │      │    │
│  │   └─────────────┘  └──────────────┘  └──────────────┘        │      │    │
│  │          │ xGMI Interconnect                                  │      │    │
│  │   ┌──────▼──────┐  ┌──────────────┐  ┌──────────────┐        │      │    │
│  │   │ AMD Mi50 #7 │  │ AMD Mi50 #8  │  │ AMD Mi50 #9  │        │      │    │
│  │   │ ROCm:6      │  │ ROCm:7       │  │ ROCm:8       │        │      │    │
│  │   │ 16GB        │  │ 16GB         │  │ 16GB         │        │      │    │
│  │   └─────────────┘  └──────────────┘  └──────────────┘        │      │    │
│  │                                                               │      │    │
│  │   ┌─────────────────────────────────────────┐                │      │    │
│  │   │ AMD Radeon 7900 XTX                     │                │      │    │
│  │   │ ROCm:9 (RDNA3)                          │                │      │    │
│  │   │ 24GB                                    │                │      │    │
│  │   └─────────────────────────────────────────┘                │      │    │
│  │                                                               │      │    │
│  │   Socket 1 Total: 6× Mi50 (96GB) + 1× 7900XTX (24GB) = 120GB │      │    │
│  │                                                               │      │    │
│  └───────────────────────────────────────────────────────────────┘      │    │
│                                                                          │    │
│  Node Total: 240GB VRAM + 2× CPU = massive heterogeneous capacity       │    │
│                                                                          │    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Global Topology Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CLUSTER-WIDE DEVICE INVENTORY                         │
│                                                                              │
│  MPI World Size: 12 ranks                                                   │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ Device Type         │ Count │ Per-Device │ Total VRAM │ Mem BW    │    │
│  ├─────────────────────┼───────┼────────────┼────────────┼───────────┤    │
│  │ NVIDIA RTX 3090     │  18   │   24 GB    │   432 GB   │ 16.8 TB/s │    │
│  │ AMD Mi50 (Vega 20)  │  36   │   16 GB    │   576 GB   │ 25.8 TB/s │    │
│  │ AMD 7900 XTX (RDNA3)│   6   │   24 GB    │   144 GB   │  5.8 TB/s │    │
│  │ Xeon CPUs (8-ch)    │  12   │  ~512 GB   │  ~6 TB     │  4.3 TB/s │    │
│  ├─────────────────────┼───────┼────────────┼────────────┼───────────┤    │
│  │ TOTAL (Decode)      │  72   │    -       │  1.15 TB   │ 52.7 TB/s │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  CRITICAL: CPU bandwidth (4.3 TB/s) is 8% of total cluster bandwidth!       │
│  Ignoring CPUs during decode would leave 8% of memory bandwidth unused.     │
│                                                                              │
│  Interconnect Hierarchy:                                                    │
│    • Intra-GPU (same vendor): NVLink/xGMI - 600 GB/s                       │
│    • Intra-Socket: PCIe 4.0 - 32 GB/s                                      │
│    • Intra-Node (cross-socket): QPI/UPI - 50 GB/s                          │
│    • Inter-Node: Infiniband HDR - 25 GB/s                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### DeviceGroup Hierarchy (Scenario 2)

```cpp
// The BackendRouter creates a hierarchy of DeviceGroups for optimal collective routing
// NOTE: CPUs are included as compute participants, not just staging buffers!

// Level 1: Global group (all 12 ranks, all devices INCLUDING CPUs)
DeviceGroup global_group = {
    .scope = DeviceGroupScope::GLOBAL,
    .devices = { /* 60 GPUs + 12 CPUs = 72 compute devices */ },
    .mpi_ranks = {0,1,2,3,4,5,6,7,8,9,10,11},
    .is_homogeneous = false  // Mix of CUDA, ROCm, CPU
};

// Level 2: Per-node groups (ranks on same machine)
DeviceGroup node0_group = {
    .scope = DeviceGroupScope::NODE,
    .devices = {CUDA:0-2, ROCm:0-9, CPU:0, CPU:1},  // CPUs included!
    .mpi_ranks = {0, 1},  // Both sockets
    .is_homogeneous = false
};
// ... node1_group through node5_group similar

// Level 3: Per-socket groups (single MPI rank)
DeviceGroup socket0_rank0 = {
    .scope = DeviceGroupScope::LOCAL,
    .devices = {CUDA:0, CUDA:1, CUDA:2, ROCm:0, ROCm:1, ROCm:2, CPU:0},
    .mpi_ranks = {0},
    .is_homogeneous = false  // CUDA + ROCm mix
};

DeviceGroup socket1_rank1 = {
    .scope = DeviceGroupScope::LOCAL,
    .devices = {ROCm:3-9, CPU:1},  // All AMD
    .mpi_ranks = {1},
    .is_homogeneous = true  // All ROCm (except CPU staging)
};

// Level 4: Vendor-specific subgroups (for fast paths)
DeviceGroup cuda_only_rank0 = {
    .scope = DeviceGroupScope::LOCAL,
    .devices = {CUDA:0, CUDA:1, CUDA:2},
    .is_homogeneous = true  // Can use NCCL!
};

DeviceGroup rocm_only_rank0 = {
    .scope = DeviceGroupScope::LOCAL,
    .devices = {ROCm:0, ROCm:1, ROCm:2},
    .is_homogeneous = true  // Can use RCCL!
};
```

### Collective Backend Selection Matrix (Scenario 2)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BACKEND SELECTION FOR SCENARIO 2                          │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Collective Scope       │ Participants          │ Backend Selected     │  │
│  ├────────────────────────┼───────────────────────┼──────────────────────┤  │
│  │                        │                       │                      │  │
│  │ GLOBAL (all 12 ranks)  │ All 72 GPUs + CPUs    │ HostBackend          │  │
│  │                        │ Mixed CUDA/ROCm       │ (MPI over IB RDMA)   │  │
│  │                        │                       │                      │  │
│  │ NODE (2 ranks/node)    │ 10 GPUs + 2 CPUs      │ HostBackend          │  │
│  │                        │ Mixed CUDA/ROCm       │ (shared memory MPI)  │  │
│  │                        │                       │                      │  │
│  │ LOCAL (1 rank)         │ RTX 3090 only         │ NCCLBackend          │  │
│  │ CUDA homogeneous       │ 3 CUDA GPUs           │ (NVLink/PCIe P2P)    │  │
│  │                        │                       │                      │  │
│  │ LOCAL (1 rank)         │ Mi50 only             │ RCCLBackend          │  │
│  │ ROCm homogeneous       │ 3-6 ROCm GPUs         │ (xGMI/PCIe P2P)      │  │
│  │                        │                       │                      │  │
│  │ LOCAL (1 rank)         │ CUDA + ROCm mix       │ HostBackend          │  │
│  │ Heterogeneous          │ Mixed vendor          │ (CPU staging)        │  │
│  │                        │                       │                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  BackendRouter::selectBackend() decision tree:                              │
│                                                                              │
│     ┌──────────────────────────┐                                            │
│     │ Is group homogeneous?    │                                            │
│     └────────────┬─────────────┘                                            │
│            YES   │   NO                                                     │
│      ┌───────────┴───────────┐                                              │
│      ▼                       ▼                                              │
│  ┌────────────┐      ┌────────────────┐                                     │
│  │ All CUDA?  │      │ HostBackend    │◄─── Heterogeneous always            │
│  └─────┬──────┘      └────────────────┘     uses CPU staging                │
│   YES  │  NO                                                                │
│   ┌────┴────┐                                                               │
│   ▼         ▼                                                               │
│ ┌────────┐ ┌────────────┐                                                   │
│ │NCCL    │ │ All ROCm?  │                                                   │
│ │Backend │ └─────┬──────┘                                                   │
│ └────────┘  YES  │  NO                                                      │
│        ┌─────────┴─────────┐                                                │
│        ▼                   ▼                                                │
│    ┌────────┐        ┌────────────┐                                         │
│    │RCCL    │        │HostBackend │                                         │
│    │Backend │        │(CPU only)  │                                         │
│    └────────┘        └────────────┘                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Hierarchical Allreduce Strategy (Scenario 2)

For global allreduce across 12 ranks with mixed devices, we use a hierarchical approach:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           HIERARCHICAL ALLREDUCE: GLOBAL ACROSS 12 RANKS                     │
│                                                                              │
│  Goal: Sum partial results from all 72 GPUs across 6 machines               │
│                                                                              │
│  Step 1: INTRA-VENDOR REDUCE (use fast paths where possible)                │
│  ═══════════════════════════════════════════════════════════                │
│                                                                              │
│  Per-socket, reduce within same vendor:                                     │
│                                                                              │
│  Socket 0 (Rank 0):                                                         │
│    CUDA group: CUDA:0 + CUDA:1 + CUDA:2 ─────► CUDA:0 (NCCL reduce)        │
│    ROCm group: ROCm:0 + ROCm:1 + ROCm:2 ─────► ROCm:0 (RCCL reduce)        │
│                                                                              │
│  Socket 1 (Rank 1):                                                         │
│    ROCm group: ROCm:3-8 + 7900XTX ───────────► ROCm:3 (RCCL reduce)        │
│                                                                              │
│                                                                              │
│  Step 2: INTRA-SOCKET CROSS-VENDOR REDUCE (CPU staging)                     │
│  ═══════════════════════════════════════════════════════                    │
│                                                                              │
│  Socket 0: CUDA:0 result + ROCm:0 result ────► CPU:0 (HostBackend)          │
│  Socket 1: ROCm:3 result ────────────────────► CPU:1 (already on CPU)       │
│                                                                              │
│                                                                              │
│  Step 3: INTRA-NODE CROSS-SOCKET REDUCE (shared memory)                     │
│  ═══════════════════════════════════════════════════════                    │
│                                                                              │
│  Per-node: CPU:0 + CPU:1 ────────────────────► CPU:0 (MPI shared mem)       │
│                                                                              │
│                                                                              │
│  Step 4: INTER-NODE GLOBAL REDUCE (Infiniband RDMA)                         │
│  ═══════════════════════════════════════════════════                        │
│                                                                              │
│  Node 0 CPU:0 ◄───┐                                                         │
│  Node 1 CPU:0 ◄───┤                                                         │
│  Node 2 CPU:0 ◄───┼───► MPI_Allreduce over IB ───► All nodes get result    │
│  Node 3 CPU:0 ◄───┤                                                         │
│  Node 4 CPU:0 ◄───┤                                                         │
│  Node 5 CPU:0 ◄───┘                                                         │
│                                                                              │
│                                                                              │
│  Step 5: BROADCAST BACK (reverse of steps 1-3)                              │
│  ═════════════════════════════════════════════                              │
│                                                                              │
│  CPU:0 ──► GPU:0 (per vendor group) ──► all GPUs in group                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Data Movement Visualization:
════════════════════════════

        Node 0                    Node 1                    Node 2
   ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
   │ S0        S1    │       │ S0        S1    │       │ S0        S1    │
   │┌───┐    ┌───┐  │       │┌───┐    ┌───┐  │       │┌───┐    ┌───┐  │
   ││3090│    │Mi50│  │       ││3090│    │Mi50│  │       ││3090│    │Mi50│  │
   ││×3  │    │×6  │  │       ││×3  │    │×6  │  │       ││×3  │    │×6  │  │
   │└─┬─┘    └─┬─┘  │       │└─┬─┘    └─┬─┘  │       │└─┬─┘    └─┬─┘  │
   │  │NCCL    │RCCL │       │  │NCCL    │RCCL │       │  │NCCL    │RCCL │
   │  ▼        ▼    │       │  ▼        ▼    │       │  ▼        ▼    │
   │┌───┐    ┌───┐  │       │┌───┐    ┌───┐  │       │┌───┐    ┌───┐  │
   ││CPU│◄──►│CPU│  │       ││CPU│◄──►│CPU│  │       ││CPU│◄──►│CPU│  │
   │└─┬─┘    └───┘  │       │└─┬─┘    └───┘  │       │└─┬─┘    └───┘  │
   └──┼──────────────┘       └──┼──────────────┘       └──┼──────────────┘
      │                         │                         │
      └─────────────────────────┼─────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │    IB RDMA Fabric     │
                    │   MPI_Allreduce       │
                    └───────────────────────┘
```

### Execution Timeline (Scenario 2, Single Token Decode)

```
Time ───────────────────────────────────────────────────────────────────────────►

12 MPI Ranks executing in parallel (showing Rank 0 and Rank 1 of Node 0):

     Rank 0 (Socket 0)                    Rank 1 (Socket 1)
     3× RTX 3090 + 3× Mi50                6× Mi50 + 7900 XTX
     ══════════════════════               ══════════════════════

t0   ┌──────────────────────────┐         ┌──────────────────────────┐
     │ Embedding (CPU:0)        │         │ Embedding (CPU:1)        │
     └───────────┬──────────────┘         └───────────┬──────────────┘
                 │                                    │
t1   ┌───────────▼──────────────┐         ┌───────────▼──────────────┐
     │ Scatter to 6 local GPUs  │         │ Scatter to 7 local GPUs  │
     │ CUDA:0-2, ROCm:0-2       │         │ ROCm:3-8, 7900XTX        │
     └───────────┬──────────────┘         └───────────┬──────────────┘
                 │                                    │
     ╔═══════════▼══════════════╗         ╔═══════════▼══════════════╗
     ║    LAYER 0 (×N layers)   ║         ║    LAYER 0 (×N layers)   ║
     ╠══════════════════════════╣         ╠══════════════════════════╣
     ║                          ║         ║                          ║
t2   ║ RMSNorm (CPU:0)          ║         ║ RMSNorm (CPU:1)          ║
     ╟──────────────────────────╢         ╟──────────────────────────╢
     ║                          ║         ║                          ║
t3   ║ QKV Projection:          ║         ║ QKV Projection:          ║
     ║  ├─ CUDA:0 (cuBLAS)      ║         ║  ├─ ROCm:3 (rocBLAS)     ║
     ║  ├─ CUDA:1 (cuBLAS)      ║         ║  ├─ ROCm:4 (rocBLAS)     ║
     ║  ├─ CUDA:2 (cuBLAS)      ║         ║  ├─ ROCm:5 (rocBLAS)     ║
     ║  ├─ ROCm:0 (rocBLAS)     ║         ║  ├─ ROCm:6 (rocBLAS)     ║
     ║  ├─ ROCm:1 (rocBLAS)     ║         ║  ├─ ROCm:7 (rocBLAS)     ║
     ║  └─ ROCm:2 (rocBLAS)     ║         ║  ├─ ROCm:8 (rocBLAS)     ║
     ║                          ║         ║  └─ 7900XTX (rocBLAS)    ║
     ╟──────────────────────────╢         ╟──────────────────────────╢
     ║                          ║         ║                          ║
t4   ║ Attention (per GPU):     ║         ║ Attention (per GPU):     ║
     ║  ├─ CUDA: Flash Attn     ║         ║  ├─ Mi50: CK attn        ║
     ║  └─ ROCm: CK attn        ║         ║  └─ 7900: CK attn        ║
     ╟──────────────────────────╢         ╟──────────────────────────╢
     ║                          ║         ║                          ║
t5   ║ Wo Projection (per GPU)  ║         ║ Wo Projection (per GPU)  ║
     ╟──────────────────────────╢         ╟──────────────────────────╢
     ║                          ║         ║                          ║
t6   ║ LOCAL REDUCE (2 stages): ║         ║ LOCAL REDUCE:            ║
     ║  ├─ NCCL: 3090s→CUDA:0   ║         ║  └─ RCCL: all→ROCm:3     ║
     ║  └─ RCCL: Mi50s→ROCm:0   ║         ║                          ║
     ║                          ║         ║                          ║
t7   ║ CROSS-VENDOR (Host):     ║         ║                          ║
     ║  CUDA:0 + ROCm:0 → CPU:0 ║         ║  ROCm:3 → CPU:1          ║
     ╟──────────────────────────╢         ╟──────────────────────────╢
     ║                          ║         ║                          ║
t8   ║┌────────────────────────┐║         ║┌────────────────────────┐║
     ║│ INTRA-NODE ALLREDUCE   │◄─────────►│ INTRA-NODE ALLREDUCE   │║
     ║│ CPU:0 ↔ CPU:1 (shm)    │║         ║│ CPU:0 ↔ CPU:1 (shm)    │║
     ║└────────────────────────┘║         ║└────────────────────────┘║
     ╚══════════════════════════╝         ╚══════════════════════════╝
                 │                                    │
                 │      ┌─────────────────────────────┤
                 │      │ (Same pattern on Nodes 1-5) │
                 │      └─────────────────────────────┘
                 │
t9   ╔═══════════▼══════════════════════════════════════════════════════════╗
     ║              INTER-NODE ALLREDUCE (MPI over Infiniband)              ║
     ║                                                                       ║
     ║  Node 0 ◄──► Node 1 ◄──► Node 2 ◄──► Node 3 ◄──► Node 4 ◄──► Node 5  ║
     ║                                                                       ║
     ║  MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_FLOAT, MPI_SUM, ...)    ║
     ║  Transport: IB RDMA with GPUDirect (if supported)                    ║
     ╚══════════════════════════════════════════════════════════════════════╝
                 │
t10  ┌───────────▼──────────────┐
     │ BROADCAST BACK to GPUs   │
     │ (reverse of reduce path) │
     └───────────┬──────────────┘
                 │
t11  ┌───────────▼──────────────┐
     │ Residual Add + FFN...    │
     │ (same pattern)           │
     └───────────┬──────────────┘
                 │
     (layers 1..N-1)
                 │
t_final          ▼
     ┌──────────────────────────┐
     │ LM Head → ALLGATHER      │
     │ → Argmax (Rank 0 only)   │
     │ → Broadcast next token   │
     └──────────────────────────┘
```

---

## Graph Construction Flow

### Phase 1: Topology Discovery (MPITopology + NUMATopology + DeviceManager)

```cpp
// Each MPI rank executes topology discovery during GraphOrchestrator initialization:

void GraphOrchestrator::initialize(const MPIContext& mpi_ctx) {
    // 1. NUMATopology: Discover CPU configuration from /proc/cpuinfo
    NUMATopology numa;
    auto cpu_info = numa.estimateCPUBandwidth();
    // cpu_info = { bandwidth_gbps: 307.2, memory_channels: 8 }
    
    // 2. DeviceManager: Query CUDA/ROCm devices
    DeviceInventory inventory = DeviceManager::getInventory();
    // inventory.cuda_devices = [{ CUDA:0, memory: 24GB, bandwidth: 936 }]
    // inventory.rocm_devices = []
    // inventory.cpu_bandwidth_gbps = 307.2  // Populated from NUMATopology
    
    // 3. MPITopology: Exchange device info across all ranks
    MPITopology topo(mpi_ctx);
    topo.detect_device_capabilities();  // Populates bandwidth for each device
    // MPI_Allgather exchanges: hostname, device list, capabilities
    
    // Result: Each rank knows global topology
    // topo.world_size = 2
    // topo.all_devices = [CPU:0, CUDA:0, CPU:1, ROCm:0, ROCm:1]
    // topo.device_capabilities[CUDA:0] = { bandwidth: 936, memory: 24GB }
    // topo.device_capabilities[CPU:0] = { bandwidth: 307.2, memory: 128GB }
    
    topology_ = std::move(topo);
}
```

### Phase 2: PlacementStrategy → PlacementPlan

```cpp
// PlacementStrategy consumes topology and produces a PlacementPlan
// CRITICAL: HybridOptimalPlacementStrategy computes phase-aware device assignment

PlacementPlan HybridOptimalPlacementStrategy::compute(const PlacementInput& input) {
    PlacementPlan plan;
    plan.layers.resize(input.n_layers);
    
    // 1. Calculate phase-aware device weights
    auto [decode_gpu_weight, decode_cpu_weight] = 
        input.getPhaseDeviceWeights(InferencePhase::DECODE);
    bool cpu_should_decode = input.cpuShouldParticipate(InferencePhase::DECODE);
    
    // 2. Assign layers to GPUs for PREFILL (compute-bound)
    for (int layer = 0; layer < input.n_layers; ++layer) {
        LayerPlacement& lp = plan.layers[layer];
        lp.layer_idx = layer;
        lp.device = selectBestGPU(input);  // GPU:0 for PREFILL
        
        // 3. For DECODE: Set up bandwidth-proportional multi-device execution
        if (cpu_should_decode) {
            lp.decode_devices = { lp.device, PlacementDevice::cpu() };
            lp.decode_weight_fractions = { decode_gpu_weight, decode_cpu_weight };
            lp.cpu_participates_in_decode = true;
            // Example: GPU:0 (0.75), CPU:0 (0.25) for decode
        }
    }
    
    return plan;
}

// Example output for Qwen2.5-7B on RTX 3090 + 8-ch Xeon:
// PlacementPlan {
//   layers[0..31]: {
//     device: GPU:0,           // PREFILL primary
//     decode_devices: [GPU:0, CPU:0],
//     decode_weight_fractions: [0.75, 0.25],
//     cpu_participates_in_decode: true
//   },
//   global: {
//     embedding_device: GPU:0,
//     lm_head_device: GPU:0,
//     final_norm_device: GPU:0
//   }
// }
```

### Phase 3: Graph Building (Model-Specific)

```cpp
// Qwen2Graph builds the actual compute DAG
ComputeGraph Qwen2Graph::build(const Model& model, const GlobalTopology& topo) {
    ComputeGraph graph;
    PlacementStrategy placement;
    
    // Create collective context for this graph
    auto collective_ctx = std::make_unique<CollectiveContext>(
        DeviceGroupFactory::createGlobal(topo),
        mpi_ctx_
    );
    
    // Embedding (replicated on all ranks)
    auto embed = graph.addStage<EmbeddingStage>(
        model.token_embd, 
        placement.assignStage(/*...*/));
    
    ComputeNode* prev = embed;
    
    for (int layer = 0; layer < model.n_layers; ++layer) {
        // Attention sublayer
        auto attn_norm = graph.addStage<RMSNormStage>(/*...*/);
        auto qkv = graph.addStage<QKVProjectionStage>(/*...*/);
        auto attn = graph.addStage<AttentionStage>(/*...*/);
        auto wo = graph.addStage<WoProjectionStage>(/*...*/);
        
        // Wo produces partial sum → need allreduce
        auto wo_allreduce = graph.addStage<AllreduceStage>(
            wo->output_buffer(),
            hidden_dim,
            collective_ctx.get()  // NEW: uses collective infrastructure
        );
        
        auto attn_residual = graph.addStage<ResidualAddStage>(/*...*/);
        
        // Wire dependencies
        graph.addEdge(prev, attn_norm);
        graph.addEdge(attn_norm, qkv);
        graph.addEdge(qkv, attn);
        graph.addEdge(attn, wo);
        graph.addEdge(wo, wo_allreduce);
        graph.addEdge(wo_allreduce, attn_residual);
        
        // FFN sublayer (similar pattern)
        // ...
        
        prev = ffn_residual;
    }
    
    // LM head with allgather
    auto final_norm = graph.addStage<RMSNormStage>(/*...*/);
    auto lm_head = graph.addStage<LMHeadStage>(/*...*/);  // Partial vocab
    auto lm_allgather = graph.addStage<AllGatherStage>(
        lm_head->output_buffer(),   // [batch, vocab/world_size]
        full_logits_buffer,         // [batch, vocab]
        vocab_chunk_size,
        collective_ctx.get()
    );
    
    graph.addEdge(prev, final_norm);
    graph.addEdge(final_norm, lm_head);
    graph.addEdge(lm_head, lm_allgather);
    
    // Transfer ownership of collective context to graph
    graph.setCollectiveContext(std::move(collective_ctx));
    
    return graph;
}
```

---

## Execution Flow

### DeviceGraphExecutor Loop

```cpp
void DeviceGraphExecutor::execute(ComputeGraph& graph) {
    // Topological order ensures dependencies are satisfied
    for (ComputeNode* node : graph.topologicalOrder()) {
        
        // 1. Ensure inputs are on the correct device
        StageCoherence::ensureInputsOnDevice(node->stage, node->device);
        
        // 2. Execute the stage
        if (node->stage->isCollective()) {
            // Collective stages use CollectiveContext internally
            // which routes to appropriate backend
            node->stage->execute();
        } else {
            // Compute stages use KernelFactory for device-specific kernels
            node->stage->execute();
        }
        
        // 3. Mark outputs as potentially modified
        StageCoherence::markOutputsDirty(node->stage);
        
        // 4. Debug validation (Debug/Integration builds only)
        #if LLAMINAR_ASSERTIONS_ACTIVE
        TensorVerification::verifyStageOutputs(node->stage);
        #endif
    }
}
```

### AllreduceStage Execution Detail

```cpp
void AllreduceStage::execute() {
    if (params_.collective_ctx) {
        // NEW PATH: Use collective infrastructure
        // CollectiveContext handles backend selection internally
        
        bool success = params_.collective_ctx->allreduce(
            params_.buffer->mutable_data(),
            count_,
            CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM
        );
        
        if (!success) {
            LOG_ERROR("Allreduce failed: " << params_.collective_ctx->lastError());
        }
        
    } else if (params_.mpi_ctx) {
        // LEGACY PATH: Direct MPI call (backward compatibility)
        MPI_Allreduce(
            MPI_IN_PLACE,
            params_.buffer->mutable_data(),
            count_,
            MPI_FLOAT,
            MPI_SUM,
            MPI_COMM_WORLD
        );
    }
}
```

### CollectiveContext Internal Flow

```cpp
bool CollectiveContext::allreduce(void* buffer, size_t count, 
                                   CollectiveDataType dtype, CollectiveOp op) {
    // 1. Get or create appropriate backend
    ICollectiveBackend* backend = router_->getBackend(group_, op);
    
    // Backend selection already happened in router:
    // - If group is all CUDA → NCCLBackend
    // - If group is all ROCm → RCCLBackend  
    // - If group is mixed → HostBackend
    // - If group spans nodes → HostBackend (uses MPI internally)
    
    // 2. Ensure backend is initialized for this group
    if (!backend->isInitialized()) {
        if (!backend->initialize(group_)) {
            last_error_ = "Failed to initialize " + backend->name();
            return false;
        }
    }
    
    // 3. Execute the collective operation
    bool success = backend->allreduce(buffer, count, dtype, op);
    
    if (!success) {
        last_error_ = backend->lastError();
    }
    
    return success;
}
```

---

## Summary

| Aspect | Scenario 1 (2 Ranks) | Scenario 2 (12 Ranks) |
|--------|----------------------|------------------------|
| **Nodes** | 1 machine | 6 machines |
| **MPI Ranks** | 2 | 12 |
| **GPU Types** | CUDA + ROCm | CUDA + ROCm (mixed) |
| **GPU Count** | 3 (1+2) | 60 |
| **CPU Participation** | Yes (20% decode weight) | Yes (8% decode weight) |
| **Total Decode BW** | 1.88 TB/s (incl. CPU) | 52.7 TB/s (incl. CPU) |
| **Interconnect** | QPI (intra-node) | IB RDMA (inter-node) |
| **Primary Backend** | HostBackend | Hierarchical (NCCL/RCCL→Host→MPI) |
| **Weight Sharding** | Bandwidth-proportional | Bandwidth-proportional |

### Key Architectural Insights

1. **Model graphs remain device-agnostic** - they declare collective operations abstractly, and `CollectiveContext` + `BackendRouter` selects the optimal communication path at runtime.

2. **CPUs are first-class decode participants** - Modern Xeons with 6-8 memory channels provide 230-360 GB/s bandwidth. During decode (memory-bandwidth-bound), this is significant and should not be wasted.

3. **Phase-aware placement** - Prefill is compute-bound (GPUs only), Decode is bandwidth-bound (all devices including CPUs).

4. **Bandwidth-proportional sharding** - Weight distribution during decode is proportional to each device's memory bandwidth, ensuring balanced completion times.

5. **Hierarchical collectives** - Intra-device (vendor-specific), intra-rank (HostBackend), inter-rank (MPI) layers allow optimal path selection at each level.

