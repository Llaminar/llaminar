/**
 * @file Perf__Q8ResidualPreparedProjection.cpp
 * @brief Whole captured prepared-projection timing on an explicit original
 * native quantized slice.
 *
 * Optional source-format selection uses the canonical verifier catalog; omitted
 * selection retains the original Q8_0 invocation. Native source blocks are read
 * exactly and never re-encoded. Q8_1 uses its original typed raw-byte
 * factory; Q8_K retains full original superblocks. CPU timing is a separately labeled component
 * arm. This diagnostic reuses production packers, prepared-store ownership,
 * GEMMStage, named workspace requirements and outer native timing events. It
 * does not implement arithmetic, a dispatch override, a profiler mode or a
 * correctness oracle. The separately registered full-GDN execution fixture
 * supplies serial byte/primary-scale/poison controls before any timing is
 * authorized.
 *
 * Original source and input metadata are recorded by prepare_source.py. Native
 * packing is additionally checked byte-for-byte here. Explicit tiling of saved
 * activation rows is a geometry workload only, never an end-to-end inference
 * claim. Setup/terminal observations remain outside every timed region.
 * An explicit --recorded-head-request branch consumes qualified native FINAL_NORM
 * and original GGUF slices with captured LMHeadStage bytes before compute-only timing.
 * The --recorded-gdn-request branch captures the complete original four-projection
 * stage on its qualified normalized input and byte-checks every packed sibling.
 * Existing raw-slice modes retain their original semantics.
 * Optional raw output export retains only the already-observed live prefix,
 * after all timed/verification work completes; no extra native readback,
 * launch, precision policy, source conversion or model execution is introduced.
 */
#include "recorded/RecordedHeadRequest.h"
#include "recorded/RecordedGDNRequest.h"
#include "cpu/CPUQ8ResidualPreparedMeasurement.h"
#include "cpu/Q8ResidualFixtureExport.h"
#include <gtest/gtest.h>
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/KernelFactory.h"
#include "kernels/common/Q8ResidualGroupedGeometry.h"
#include "kernels/common/Q8ResidualShortKGeometry.h"
#ifdef HAVE_CUDA
#include "kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#endif
#ifdef HAVE_ROCM
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "kernels/rocm/gemm/ROCmWeightPacker.h"
#endif
#include "kernels/cpu/gemm/CPUNativeVNNIWeightPacker.h"
#include "tensors/TensorFactory.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include "utils/MPIContext.h"
#include "utils/PreparedWeightTestHarness.h"
#include "utils/QuantizedVerifierFormats.h"
#include "utils/OriginalQ8ResidualProjectionSource.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <string_view>
#include <vector>
using namespace llaminar2;
namespace {
using Factory = llaminar::v2::kernels::KernelFactory;
/** @brief Fail infrastructure before a retained graph can reference invalid
 * storage. */
void require(bool value, const char *why) {
  if (!value)
    throw std::runtime_error(why);
}
/** @brief Retain native allocations through the completed diagnostic stream. */
struct Storage {
  DeviceId device;
  IBackend *backend;
  void *stream;
  std::shared_ptr<std::vector<std::shared_ptr<DeviceTransferBuffer>>> owners =
      std::make_shared<std::vector<std::shared_ptr<DeviceTransferBuffer>>>();
  /** @brief Join submitted diagnostic work before captured native allocations
   * retire. */
  ~Storage() { retire(); }
  /** @brief Complete native work; teardown failure is fatal.
   * This is a diagnostic terminal boundary, never a per-stage inference sync.
   */
  void retire() noexcept {
    if (!backend->synchronizeStream(stream, device.ordinal))
      std::terminate();
  }
  /** @brief Retain one exact transfer-authority device allocation until graph
   * retirement. */
  template <class T> T *allocate(std::size_t n) {
    auto owner = TransferEngine::instance().allocateDeviceTransferBuffer(
        n * sizeof(T), device);
    require(owner && owner->isBound(),
            "Native correction fixture allocation failed");
    auto *p = static_cast<T *>(owner->mutableDeviceData());
    owners->push_back(std::move(owner));
    return p;
  }
  /** @brief Publish initial/replay diagnostic data on the exact stream, outside
   * measured work. */
  template <class T> void upload(T *p, const std::vector<T> &x) {
    const bool published = backend->hostToDeviceOnStream(
        p, x.data(), x.size() * sizeof(T), device.ordinal, stream);
    const bool joined = backend->synchronizeStream(stream, device.ordinal);
    require(published && joined,
            "Correction fixture publication/observation failed");
  }
  /** @brief Materialize and initialize an immutable fixture plane before
   * capture. */
  template <class T> T *input(const std::vector<T> &x) {
    auto *p = allocate<T>(x.size());
    upload(p, x);
    return p;
  }
  /** @brief Observe the exact diagnostic validation range; zero elements
   * transfer nothing. */
  template <class T> std::vector<T> download(const T *p, std::size_t n) {
    if (!n)
      return {};
    std::vector<T> x(n);
    const bool copied = backend->deviceToHostOnStream(
        x.data(), p, n * sizeof(T), device.ordinal, stream);
    const bool joined = backend->synchronizeStream(stream, device.ordinal);
    require(copied && joined,
            "Correction fixture terminal copy/observation failed");
    return x;
  }
};
/** @brief Retire captured work before every captured tensor/workspace owner.
 * Declare this scope after all such owners and before the first native submit;
 * exception unwind then follows the same completion-before-storage
 * order as successful replay. Storage remains alive outside the scope.
 */
struct CapturedWorkRetirement {
  Storage &storage;
  explicit CapturedWorkRetirement(Storage &s) : storage(s) {}
  CapturedWorkRetirement(const CapturedWorkRetirement &) = delete;
  CapturedWorkRetirement &operator=(const CapturedWorkRetirement &) = delete;
  ~CapturedWorkRetirement() { storage.retire(); }
};
/** @brief Admit only the supplied actual stage/engine named buffers. */
struct Workspace {
  std::shared_ptr<PhysicalMemoryAuthority> memory;
  std::unique_ptr<DeviceWorkspaceManager> value;
  Workspace(DeviceId device, const WorkspaceRequirements &r) {
    auto bytes = r.total_bytes_with_alignment();
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank = 0,
              .device = device,
              .total_bytes = bytes,
              .admission_available_bytes = bytes},
             PhysicalMemoryOwner::ExecutionWorkspace, bytes);
    memory = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),
        0);
    value = std::make_unique<DeviceWorkspaceManager>(device, bytes, memory);
    require(value->allocate(r),
            "Correction fixture workspace admission failed");
  }
};
/** @brief Authenticate exact native source bits before device publication. */
void authenticate(const std::vector<std::uint8_t> &raw,
                  const std::vector<std::uint8_t> &payload,
                  const std::vector<std::uint16_t> &scales, int n, int k) {
  require(payload.size() == std::size_t(n) * k &&
              scales.size() == std::size_t(n) * k / 32,
          "Packed source extent mismatch");
  for (int row = 0; row < n; ++row)
    for (int b = 0; b < k / 32; ++b) {
      const auto *source = raw.data() + (std::size_t(row) * (k / 32) + b) * 34;
      std::uint16_t d;
      std::memcpy(&d, source, 2);
      require(d == scales[std::size_t(b) * n + row],
              "Native source scale bits changed");
      require(std::memcmp(source + 2,
                          payload.data() + (std::size_t(b) * n + row) * 32,
                          32) == 0,
              "Native source payload bits changed");
    }
}
/** @brief Use actual production native packing and preuploaded prepared-engine
 * construction. */
std::shared_ptr<ITensorGemm> prepare(TensorBase &source,
                                     const std::vector<std::uint8_t> &raw,
                                     Storage &s, int n, int k) {
#ifdef HAVE_CUDA
  if (s.device.is_cuda()) {
    cuda::CUDAPackedWeights packed;
    require(cuda::packWeightsToCUDA(&source, packed),
            "Production CUDA native pack failed");
    if (source.native_type() == TensorType::Q8_0)
      authenticate(raw, packed.native_vnni, packed.native_scales, n, k);
    return std::make_shared<cuda::CUDAQuantisedGemmKernel>(
        n, k, s.device.ordinal, s.input(packed.native_vnni),
        s.input(packed.native_scales),
        packed.native_mins.empty() ? nullptr : s.input(packed.native_mins),
        packed.native_emins.empty() ? nullptr : s.input(packed.native_emins),
        packed.native_codebook_id, packed.native_blocks_per_row, s.owners,
        packed.native_source_identity);
  }
#endif
#ifdef HAVE_ROCM
  if (s.device.is_rocm()) {
    rocm::ROCmPackedWeights packed;
    require(rocm::packNativeVNNI(&source, packed),
            "Production HIP native pack failed");
    if (source.native_type() == TensorType::Q8_0)
      authenticate(raw, packed.native_vnni_payload, packed.native_vnni_scales,
                   n, k);
    return std::make_shared<rocm::ROCmQuantisedGemmKernel>(
        n, k, s.device.ordinal, s.input(packed.native_vnni_payload),
        s.input(packed.native_vnni_scales),
        packed.native_vnni_mins.empty() ? nullptr
                                        : s.input(packed.native_vnni_mins),
        packed.native_vnni_emins.empty() ? nullptr
                                         : s.input(packed.native_vnni_emins),
        packed.native_vnni_codebook_id, packed.native_vnni_blocks_per_row,
        s.owners, packed.native_source_identity);
  }
#endif
  throw std::runtime_error(
      "Correction fixture selected unavailable native backend");
}
/** @brief Register exact source/engine identities through the production
 * prepared-store boundary. */
WeightBinding registerEngine(PreparedWeightStore &store, TensorBase &source,
                             DeviceId device,
                             std::shared_ptr<ITensorGemm> engine,
                             const std::string &name) {
  auto binding = test::makePreparedWeightTestBinding(&source, device, name,
                                                     store.modelId());
  auto handle = std::make_shared<Factory::PreparedGemmHandle>();
  handle->tensor = &source;
  handle->device_id = device;
  handle->kind = device.is_cuda()
                     ? Factory::GemmPreparationKind::CUDA_INT8_PACKED
                     : Factory::GemmPreparationKind::ROCM_INT8_PACKED;
  handle->variant = static_cast<int>(source.native_type());
  handle->prepared_weights = std::make_shared<Factory::PreparedGemmWeights>();
  handle->prepared_weights->kind = handle->kind;
  handle->prepared_weights->owned_kernel = std::move(engine);
  handle->prepared_weights->kernel =
      handle->prepared_weights->owned_kernel.get();
  binding.prepared = store.registerPreparedGemmHandle(
      binding, test::preparedKindForDevice(device), device, std::move(handle));
  return binding;
}
/**
 * @brief Reject an accidental source-model path before allocating/reading
 * payload.
 * @param path Explicit selected slice/checkpoint file.
 * @param byteBound Exact required bytes or the admitted diagnostic maximum.
 * @param exact Require equality rather than a positive bounded multiple of T.
 * @return Only the statically checked selected diagnostic extent.
 * @throws std::exception For missing, empty, truncated or oversized input.
 */
template <class T>
std::vector<T> load(const char *path, std::size_t byteBound, bool exact) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  require(bool(file), "Native slice unavailable");
  const auto end = file.tellg();
  require(end > 0, "Native slice byte extent invalid");
  const auto bytes = static_cast<std::uintmax_t>(end);
  require(bytes <= byteBound && (!exact || bytes == byteBound) &&
              bytes % sizeof(T) == 0,
          "Native slice exceeds its exact/bounded diagnostic byte extent");
  std::vector<T> x(static_cast<std::size_t>(bytes) / sizeof(T));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(x.data()),
            static_cast<std::streamsize>(bytes));
  require(bool(file), "Native slice read failed");
  return x;
}
/** @brief Named diagnostic comparison; never a production selector or arithmetic policy. */
enum class ProjectionComparison { SingleVsTwoTerm, OriginalVsShortK };
/** @brief Component launch selection bound to one retained graph instance. */
enum class ProjectionKernel { Production, OriginalColumnReference };
/** @brief One explicitly authored source, row authority and timing workload. */
struct MeasureOptions {
  DeviceId device;
  ProjectionComparison comparison = ProjectionComparison::SingleVsTwoTerm;
  int m = 0, n = 0, k = 0, live = -1, iterations = 0, batches = 0, warmup = 0;
  bool verifier = false, serial = false, tile = false, grouped = false;
  std::string weights, input, sourceName, exportDirectory,
      sourceFormat = "Q8_0";
};
/** @brief Parse one complete decimal integer, rejecting trailing text. */
int integer(const std::string &v) {
  std::size_t end = 0;
  const int value = std::stoi(v, &end);
  require(end == v.size(), "Noninteger measurement option");
  return value;
}
/** @brief Parse bounded exact geometry; unknown/duplicate options fail before
 * native setup. */
MeasureOptions options(int argc, char **argv) {
  MeasureOptions o;
  std::set<std::string> seen;
  for (int i = 1; i < argc; i += 2) {
    require(i + 1 < argc, "Missing option value");
    const std::string key = argv[i], v = argv[i + 1];
    require(seen.insert(key).second, "Duplicate measurement option");
    if (key == "--backend") {
      require(v == "cuda" || v == "rocm", "Invalid backend");
      o.device = v == "cuda" ? DeviceId::cuda(0) : DeviceId::rocm(0);
    } else if (key == "--m")
      o.m = integer(v);
    else if (key == "--n")
      o.n = integer(v);
    else if (key == "--k")
      o.k = integer(v);
    else if (key == "--live")
      o.live = integer(v);
    else if (key == "--phase") {
      require(v == "ordinary" || v == "verifier" || v == "serial",
              "Invalid phase");
      o.verifier = v == "verifier";
      o.serial = v == "serial";
    } else if (key == "--implementation") {
      require(v == "vector" || v == "grouped",
              "Invalid compiled implementation expectation");
      o.grouped = v == "grouped";
    } else if (key == "--comparison") {
      require(v == "single_vs_two_term" || v == "original_vs_short_k", "Invalid comparison");
      o.comparison = v == "original_vs_short_k" ? ProjectionComparison::OriginalVsShortK
                                                 : ProjectionComparison::SingleVsTwoTerm;
    } else if (key == "--input-mode") {
      require(v == "exact" || v == "tile", "Invalid input mode");
      o.tile = v == "tile";
    } else if (key == "--export-output") {
      require(!v.empty(), "Empty export directory");
      o.exportDirectory = v;
    } else if (key == "--weights")
      o.weights = v;
    else if (key == "--input")
      o.input = v;
    else if (key == "--source-name")
      o.sourceName = v;
    else if (key == "--source-format")
      o.sourceFormat = v;
    else if (key == "--iterations")
      o.iterations = integer(v);
    else if (key == "--batches")
      o.batches = integer(v);
    else if (key == "--warmup")
      o.warmup = integer(v);
    else
      throw std::invalid_argument("Unknown measurement option");
  }
  require(seen.size() == 14 + seen.count("--export-output") +
                             seen.count("--source-format") + seen.count("--comparison"),
          "All14 legacy measurement options required; export/source-format are "
          "optional");
  const auto &format = test::quantizedVerifierFormat(o.sourceFormat);
  const auto sourceBlock = format.source_is_superblock ? 256u : 32u;
  const auto sourceBytes = cpu::native_vnni::native_block_bytes_for_format(
      format.source_codebook_id, format.source_is_superblock);
  require(o.device.is_gpu() && o.m > 0 && o.m <= 8192 && o.n > 0 &&
              o.n <= 65536 && o.k > 0 && o.k <= 65536 && o.k % 32 == 0,
          "Unsupported Q8 diagnostic geometry");
  require(std::size_t(o.m) * o.k * sizeof(float) <= (1u << 28) &&
              std::size_t(o.m) * o.n * sizeof(float) <= (1u << 28) &&
              std::size_t(o.n) *
                      ((std::size_t(o.k) + sourceBlock - 1) / sourceBlock) *
                      sourceBytes <=
                  (1u << 28),
          "Diagnostic tensor byte bound exceeded");
  require(o.live >= 0 && o.live <= o.m && !o.weights.empty() &&
              !o.input.empty() && !o.sourceName.empty(),
          "Exact source/input/live extent required");
  require(!o.serial || (o.m == 1 && o.live == 1),
          "Serial phase requires exactly one active row");
  require(o.iterations > 0 && o.iterations <= 1024 && o.batches > 0 &&
              o.batches <= 20 && o.batches % 2 == 0 && o.warmup > 0 &&
              o.warmup <= 64,
          "Bounded balanced timing repetitions required");
  if (o.comparison == ProjectionComparison::OriginalVsShortK) {
    require(o.m == 1 && !o.grouped && o.k >= 64 && o.k <= 512 && o.n >= 4*o.k,
            "Short-K comparison requires M1/K64..512/N>=4K/vector");
    require(o.exportDirectory.empty(), "Short-K comparison uses exact pair byte checks, not SingleQ8 export labels");
  }
  if (!o.exportDirectory.empty()) {
    const std::filesystem::path directory(o.exportDirectory);
    const auto parent = directory.has_parent_path()
                            ? directory.parent_path()
                            : std::filesystem::current_path();
    require(!std::filesystem::exists(directory) &&
                std::filesystem::is_directory(parent),
            "Export requires a fresh directory under an existing parent");
  }
  return o;
}
/** @brief One retained prepared projection owns its complete
 * graph/workspace/output lifetime. */
struct Projection {
  Storage &storage;
  const MeasureOptions &options;
  QuantizedActivationOperandPrecision precision;
  const ProjectionKernel selectedKernel;
  const DeviceRowRange rows;
  std::size_t signature = 0, used = 0;
  const void *outputPointer = nullptr;
  std::vector<const void *> banks;
  std::shared_ptr<FP32Tensor> output;
  std::unique_ptr<GEMMStage> stage;
  std::unique_ptr<Workspace> workspace;
  std::unique_ptr<IGPUGraphCapture> graph;
  /** @brief Bind the actual prepared source and normal stage workspace before
   * capture. */
  Projection(Storage &s, const MeasureOptions &o, FP32Tensor *input,
             TensorBase *weight, PreparedWeightStore &store,
             const WeightBinding &binding,
             QuantizedActivationOperandPrecision precision,
             const int32_t *count,
             ProjectionKernel selectedKernel = ProjectionKernel::Production)
      : storage(s), options(o), precision(precision), selectedKernel(selectedKernel),
        rows(count ? DeviceRowRange::deviceCounted(o.m,count) : DeviceRowRange::fullyActive(o.m)),
        output(std::make_shared<FP32Tensor>(
            std::vector<std::size_t>{std::size_t(o.m), std::size_t(o.n)},
            DeviceId::cpu())) {
    try {
      TransferEngine::prepareDeviceOutput(output.get(), o.device, s.stream);
      TransferEngine::requireDeviceOutput(output.get(), o.device, s.stream);
      stage = std::make_unique<GEMMStage>(GEMMStage::Params{
          .device_id = o.device,
          .A = input,
          .B = weight,
          .C = output.get(),
          .m = o.m,
          .n = o.n,
          .k = o.k,
          .alpha = 1,
          .beta = 0,
          .transpose_B = false,
          .a_buffer_id = BufferId::NORMALIZED,
          .c_buffer_id = BufferId::ATTN_PROJ,
          .force_decode_equivalent_verifier_prefill = o.verifier,
          .verifier_row_range =
              (precision == QuantizedActivationOperandPrecision::TwoTermQ8 ||
               o.verifier) &&
                      !o.serial
                  ? DeviceRowRange::deviceCounted(o.m, count)
                  : DeviceRowRange::fullyActive(o.m),
          .prepared_ref = binding.prepared,
          .prepared_store = &store,
          .quantized_operand_precision = precision});
      stage->setGPUStream(s.stream);
      workspace = std::make_unique<Workspace>(
          o.device, stage->getWorkspaceRequirements(o.m, o.n, o.k));
      stage->bindWorkspace(workspace->value.get());
      signature = stage->graphCaptureVariantSignature();
      used = workspace->value->used();
      outputPointer = output->gpu_data_ptr();
      std::cout << "WORKSPACE precision="
                << (precision == QuantizedActivationOperandPrecision::TwoTermQ8
                        ? "two_term"
                        : "single")
                << " admitted_named_bytes="
                << stage->getWorkspaceRequirements(o.m, o.n, o.k)
                       .total_bytes_with_alignment()
                << " used=" << used << " signature=" << signature
                << " output=" << outputPointer << "\n";
      if (precision == QuantizedActivationOperandPrecision::TwoTermQ8) {
        require(stage->getWorkspaceRequirements(o.m, o.n, o.k).buffers.size() ==
                    4,
                "TwoTerm requires exactly four named banks");
        for (const auto &name : std::array{
                 kGPUProjectionPrimaryQ8, kGPUProjectionPrimaryQ8Scales,
                 kGPUProjectionResidualQ8, kGPUProjectionResidualQ8Scales})
          banks.push_back(workspace->value->getBuffer(name));
      }
    } catch (...) {
      if (!s.backend->synchronizeStream(s.stream, s.device.ordinal))
        std::terminate();
      if (stage)
        stage->unbindWorkspace();
      throw;
    }
  }
  /** @brief Join/retire the graph before any captured workspace, stage or
   * output teardown. */
  ~Projection() {
    if (!storage.backend->synchronizeStream(storage.stream,
                                            storage.device.ordinal))
      std::terminate();
    if (graph) {
      graph->reset();
      graph.reset();
    }
    if (stage)
      stage->unbindWorkspace();
  }
  /**
   * @brief Submit the actual stage or one explicit diagnostic role on unchanged bindings.
   * @param context Original participant context, used only by actual stage execution.
   * @return True after the selected actual stage or independent reference accepts its native submissions.
   * @throws std::invalid_argument For a diagnostic role/backend that is not available.
   * @note The original-column role never consults the normal production selector.
   */
  bool submit(IDeviceContext *context) {
    if (selectedKernel == ProjectionKernel::Production) return stage->execute(context);
    require(precision == QuantizedActivationOperandPrecision::TwoTermQ8, "Original-column reference requires TwoTerm");
    const auto &p = stage->getParams();
    auto *engine = p.prepared_store->gemmKernel(*p.prepared_ref);
    DeviceNativeVNNIMatrixDesc source{};
    require(engine && engine->exportNativeVNNIMatrixDesc(source), "Short-K lost original prepared source");
    const GPUQ8ResidualCorrectionArgs args{
        .weights=source, .rows=rows,
        .quantizer=storage.device.is_cuda() ? Q8ResidualQuantizerPolicy::CUDAFloatScaleNativeReciprocal
                                          : Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal,
        .input=static_cast<const float *>(p.A->gpu_data_ptr()),
        .output=static_cast<float *>(output->gpu_data_ptr()),
        .primary=static_cast<signed char *>(workspace->value->getBuffer(kGPUProjectionPrimaryQ8)),
        .residual=static_cast<signed char *>(workspace->value->getBuffer(kGPUProjectionResidualQ8)),
        .primaryScales=static_cast<float *>(workspace->value->getBuffer(kGPUProjectionPrimaryQ8Scales)),
        .residualScales=static_cast<float *>(workspace->value->getBuffer(kGPUProjectionResidualQ8Scales))};
#ifdef HAVE_CUDA
    if (storage.device.is_cuda()) {
      cudaQ8ResidualOriginalColumnReference(args, storage.stream);
      return true;
    }
#endif
#ifdef HAVE_ROCM
    if (storage.device.is_rocm()) {
      rocmQ8ResidualOriginalColumnReference(args, storage.stream);
      return true;
    }
#endif
    throw std::invalid_argument("Short-K requested backend is not compiled");
  }
  /** @brief Capture the exact native stage after optional setup warmup outside
   * capture. */
  void capture(IWorkerGPUContext &worker, IDeviceContext *context,
               bool prepare) {
    if (prepare) {
      require(submit(context), "Prepared projection warmup failed");
      require(storage.backend->synchronizeStream(storage.stream,
                                                 storage.device.ordinal),
              "Setup projection observation failed");
    }
    graph = worker.createGraphCapture(storage.stream);
    require(bool(graph), "Native capture unavailable");
    {
      ScopedBackendGraphCapture capture(
          worker, *graph, "prepared original Q8 projection measurement");
      require(capture.begin(), "Projection capture failed");
      require(submit(context), "Projection stage capture failed");
      capture.finish();
    }
    require(graph->instantiate(), "Projection graph instantiation failed");
    std::vector<GPUGraphKernelNodeInfo> nodes;
    std::string why;
    require(graph->inspectKernelNodes(nodes, &why),
            "Native node inspection failed");
    if (precision == QuantizedActivationOperandPrecision::TwoTermQ8) {
      require(nodes.size() == 2,
              "TwoTerm capture must contain only publication and projection");
      DeviceNativeVNNIMatrixDesc source{};
      auto *engine = stage->getParams().prepared_store->gemmKernel(
          *stage->getParams().prepared_ref);
      require(engine && engine->exportNativeVNNIMatrixDesc(source),
              "Projection lost its original source descriptor");
      // The oracle's geometry must remain independent when production dispatch evolves.
      const auto geometry = selectedKernel == ProjectionKernel::OriginalColumnReference
          ? Q8ResidualGroupedGeometry{Q8ResidualGroupedStrategy::RowVector, 1, 128, 128,
              (static_cast<std::uint64_t>(options.n) + 127) / 128}
          : q8ResidualGroupedGeometry(
              storage.device.is_cuda() ? Q8ResidualGroupedBackend::CUDA
                                       : Q8ResidualGroupedBackend::ROCm,
              options.m, options.n, source.codebook_id);
      const bool expectedGroup =
          geometry.strategy == Q8ResidualGroupedStrategy::CooperativeMatrix;
      require(!expectedGroup || options.grouped,
              "Explicit implementation does not match the original "
              "format-owned physical dispatch");
      bool group = false, projection = false, publication = false;
      const auto publisherGrid =
          (std::size_t(options.m) * (options.k / 32) + 7) / 8;
      const auto shortGeometry = q8ResidualShortKGeometry(
          storage.device.is_cuda() ? Q8ResidualGroupedBackend::CUDA : Q8ResidualGroupedBackend::ROCm,
          options.m, options.n, options.k, source.codebook_id);
      const bool orderedShortK = selectedKernel == ProjectionKernel::Production && bool(shortGeometry);
      const auto projectionGrid = orderedShortK ? shortGeometry->blocks : geometry.blocks;
      const auto projectionThreads = orderedShortK ? shortGeometry->threads : geometry.threads;
      for (const auto &node : nodes) {
        require(node.grid_y == 1 && node.grid_z == 1 && node.block_y == 1 &&
                    node.block_z == 1,
                "Unexpected Q8 launch dimensions");
        if (node.name.find("publish") != std::string::npos) {
          publication = true;
          require(node.grid_x == publisherGrid && node.block_x == 256,
                  "Unexpected primary/residual publication launch");
        }
        if (node.name.find("project") != std::string::npos) {
          projection = true;
          group = !orderedShortK && selectedKernel == ProjectionKernel::Production && node.block_x == 256;
          if (orderedShortK) {
            require(node.name.find("projectShortK") != std::string::npos &&
                        node.dynamic_shared_memory_bytes == 0 &&
                        node.static_shared_memory_bytes == shortGeometry->shared_bytes,
                    "Short-K actual kernel/storage identity differs");
          }
          if (selectedKernel == ProjectionKernel::OriginalColumnReference) {
            require(node.name.find("projectShortK") == std::string::npos &&
                        (node.name.find("project<1>") != std::string::npos ||
                         node.name.find("projectILi1EE") != std::string::npos),
                    "Independent original-column graph selected another kernel");
          }
          require(node.grid_x == projectionGrid &&
                      node.block_x == static_cast<unsigned>(projectionThreads),
                  "Unexpected TwoTerm projection launch");
        }
      }
      require(publication && projection,
              "TwoTerm graph lacks its actual publication/projection kernels");
      require(group == expectedGroup,
              "Compiled projection implementation does not match requested "
              "vector/grouped expectation");
    }
    for (const auto &n : nodes) {
      require(n.local_memory_bytes_per_thread == 0,
              "Projection contains native local memory");
      std::cout << "NODE precision="
                << (precision == QuantizedActivationOperandPrecision::TwoTermQ8
                        ? "two_term"
                        : "single")
                << " name=" << n.name << " grid=" << n.grid_x << "x" << n.grid_y
                << "x" << n.grid_z << " block=" << n.block_x << "x" << n.block_y
                << "x" << n.block_z << " registers=" << n.registers_per_thread
                << " local=" << n.local_memory_bytes_per_thread << "\n";
    }
  }
  /** @brief Validate immutable stage/workspace addresses after completed
   * timing. */
  void validateIdentity() const {
    require(stage->graphCaptureVariantSignature() == signature &&
                workspace->value->used() == used &&
                output->gpu_data_ptr() == outputPointer,
            "Captured stage/workspace identity changed");
    if (!banks.empty()) {
      std::size_t i = 0;
      for (const auto &name :
           std::array{kGPUProjectionPrimaryQ8, kGPUProjectionPrimaryQ8Scales,
                      kGPUProjectionResidualQ8, kGPUProjectionResidualQ8Scales})
        require(banks[i++] == workspace->value->getBuffer(name),
                "Captured named bank changed");
    }
  }
  /** @brief Observe live output and validate untouched TwoTerm inactive rows
   * outside timing. */
  std::vector<float> observe() {
    validateIdentity();
    auto all =
        storage.download(static_cast<const float *>(output->gpu_data_ptr()),
                         std::size_t(options.m) * options.n);
    const auto live = std::size_t(options.live) * options.n;
    bool useful = false;
    for (std::size_t i = 0; i < live; ++i) {
      require(std::isfinite(all[i]),
              "Projection produced nonfinite live output");
      useful = useful || all[i] != 0;
    }
    require(!live || useful, "Projection produced no useful live output");
    if (precision == QuantizedActivationOperandPrecision::TwoTermQ8)
      require(std::all_of(all.begin() + live, all.end(),
                          [](float x) { return x == -12345.f; }),
              "TwoTerm modified inactive output");
    all.resize(live);
    return all;
  }
  /** @brief Reset diagnostic output before native replay; never called in timed
   * scope. */
  void dirtyOutput() {
    storage.upload(
        static_cast<float *>(output->gpu_data_ptr()),
        std::vector<float>(std::size_t(options.m) * options.n, -12345.f));
  }
};
/** @brief Timing events are recorded only outside event-free retained graph
 * replay. */
struct MeasureEvents {
  IBackend *backend;
  DeviceId device;
  std::vector<void *> owned;
  void *add() {
    auto *e = backend->createTimingEvent(device.ordinal);
    require(e, "Timing event unavailable");
    owned.push_back(e);
    return e;
  }
  ~MeasureEvents() {
    for (auto *e : owned)
      backend->destroyEvent(e, device.ordinal);
  }
};
/** @brief Measure complete submitted captured projections, including native
 * graph gaps. */
float elapsed(Projection &p, void *start, void *stop, int iterations) {
  auto &s = p.storage;
  require(s.backend->recordEvent(start, s.device.ordinal, s.stream),
          "Outer start record failed");
  for (int i = 0; i < iterations; ++i)
    require(p.graph->launch(), "Timed retained projection failed");
  require(s.backend->recordEvent(stop, s.device.ordinal, s.stream),
          "Outer stop record failed");
  require(s.backend->waitForEvent(stop, s.device.ordinal),
          "Terminal timing event failed");
  float ms = 0;
  require(s.backend->eventElapsedTimeMs(start, stop, s.device.ordinal, &ms),
          "Completed outer timing unavailable");
  return ms * 1000.f / iterations;
}
/**
 * @brief Pair the independent original-column oracle with actual production stage execution.
 *
 * Both objects retain their own original four-bank workspace and captured graph.
 * Every observed output and published bank is byte-equal before/after timing;
 * raw AB/BA event intervals include all submissions and retain every result.
 */
void measureShortK(Storage &storage,const MeasureOptions &o,IWorkerGPUContext &worker,
                   IDeviceContext *context,FP32Tensor *input,TensorBase *weight,
                   PreparedWeightStore &store,const WeightBinding &binding,const int32_t *count,
                   void *start,void *stop,const DeviceNativeVNNIMatrixDesc &sourceBefore,
                   const std::vector<std::uint8_t> &raw) {
  require(bool(q8ResidualShortKGeometry(o.device.is_cuda() ? Q8ResidualGroupedBackend::CUDA
                                                         : Q8ResidualGroupedBackend::ROCm,
                                        o.m,o.n,o.k,sourceBefore.codebook_id)),
          "Exact source is not eligible for short-K comparison");
  Projection original(storage,o,input,weight,store,binding,QuantizedActivationOperandPrecision::TwoTermQ8,count,
                      ProjectionKernel::OriginalColumnReference);
  Projection candidate(storage,o,input,weight,store,binding,QuantizedActivationOperandPrecision::TwoTermQ8,count,
                       ProjectionKernel::Production);
  std::cout << "SHORT_K_ROLE role=original kernel=independent_original_column_reference\n";
  original.capture(worker,context,true);
  std::cout << "SHORT_K_ROLE role=candidate kernel=production_ordered_short_k\n";
  candidate.capture(worker,context,true);
  original.dirtyOutput(); candidate.dirtyOutput();
  for(int i=0;i<o.warmup;++i) {
    require(original.graph->launch(),"Original TwoTerm warm replay failed");
    require(candidate.graph->launch(),"Short-K warm replay failed");
  }
  const auto expected=original.observe();
  const auto ownerCount=storage.owners->size(); const auto *inputPointer=input->gpu_data_ptr();
  const auto compare=[&] {
    const auto oldOutput=original.observe(),newOutput=candidate.observe();
    require(oldOutput.size()==expected.size() && oldOutput.size()==newOutput.size() &&
            (oldOutput.empty()||(std::memcmp(oldOutput.data(),expected.data(),oldOutput.size()*sizeof(float))==0 &&
                                std::memcmp(oldOutput.data(),newOutput.data(),oldOutput.size()*sizeof(float))==0)),
            "Short-K candidate differs from exact original TwoTerm bytes");
    for(std::size_t i=0;i<original.banks.size();++i) {
      const auto values = std::size_t(o.live)*o.k;
      const auto bytes = i%2 ? values/32*sizeof(float) : values;
      const auto a=storage.download(static_cast<const std::uint8_t *>(original.banks[i]),std::size_t(bytes));
      const auto b=storage.download(static_cast<const std::uint8_t *>(candidate.banks[i]),std::size_t(bytes));
      require(a==b,"Short-K published activation bank/scales differ");
    }
    DeviceNativeVNNIMatrixDesc after{};
    auto *engine=store.gemmKernel(*binding.prepared);
    require(engine && engine->exportNativeVNNIMatrixDesc(after) &&
            std::memcmp(&after,&sourceBefore,sizeof(after))==0 &&
            std::memcmp(weight->raw_data(),raw.data(),raw.size())==0 &&
            storage.owners->size()==ownerCount && input->gpu_data_ptr()==inputPointer,
            "Short-K retained source/allocation identity changed");
  };
  compare();
  for(int batch=0;batch<o.batches;++batch) {
    float a,b;
    if(batch%2==0) {a=elapsed(original,start,stop,o.iterations);b=elapsed(candidate,start,stop,o.iterations);}
    else {b=elapsed(candidate,start,stop,o.iterations);a=elapsed(original,start,stop,o.iterations);}
    compare();
    std::cout << "CAPTURED_SHORT_K_TIMING backend=" << o.device.toString()
              << " M=" << o.m << " N=" << o.n << " K=" << o.k << " live=" << o.live
              << " source=" << o.sourceName << " source_format=" << o.sourceFormat
              << " execution_codebook=" << unsigned(sourceBefore.codebook_id)
              << " weights=" << o.weights << " input=" << o.input
              << " batch=" << batch << " order=" << (batch%2 ? "BA" : "AB")
              << " graph_submissions=" << o.iterations
              << " original_two_term_us=" << a << " short_k_two_term_us=" << b << " ratio=" << b/a
              << " scope=outer_complete_native_transaction includes_submission_gaps=1 timed_host_observations=0\n";
  }
}
/**
 * @brief Export already-joined live FP32 observations after all timing
 * completes.
 * @param options Exact selected component invocation and optional fresh
 * directory.
 * @param savedRows Original recorded input row count, before any declared
 * tiling.
 * @param weightBytes Exact selected native Q8_0 byte extent already loaded.
 * @param single Latest existing SingleQ8 terminal observation, live prefix
 * only.
 * @param two Verified byte-stable TwoTermQ8 observation of the same live rows.
 *
 * This function performs host file I/O only. It never reads a device pointer or
 * submits native work. A new owned directory prevents overwriting evidence; the
 * metadata completeness marker follows both successfully closed raw files.
 * Source naming is caller declaration; an external original-GGUF slice manifest
 * must authenticate it before interpreting these component outputs as model
 * data.
 */
void exportObservedOutputs(const MeasureOptions &options, std::size_t savedRows,
                           std::size_t weightBytes,
                           const DeviceNativeVNNIMatrixDesc &source,
                           const std::vector<float> &single,
                           const std::vector<float> &two) {
  if (options.exportDirectory.empty())
    return;
  require(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559 &&
              std::endian::native == std::endian::little,
          "Raw export requires IEEE FP32 little-endian host");
  const auto elements = std::size_t(options.live) * std::size_t(options.n);
  require(single.size() == elements && two.size() == elements,
          "Export must contain exactly live*N elements");
  const std::filesystem::path directory(options.exportDirectory);
  require(std::filesystem::create_directory(directory),
          "Export directory was already published");
  const auto write = [&](const char *name, const std::vector<float> &values) {
    std::ofstream file(directory / name, std::ios::binary | std::ios::out);
    require(bool(file), "Cannot open owned raw export");
    if (!values.empty())
      file.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    file.flush();
    require(bool(file), "Incomplete raw FP32 export");
    file.close();
    require(!file.fail(), "Raw FP32 export close failed");
  };
  write("single.f32", single);
  write("two-term.f32", two);
  const nlohmann::json metadata = {
      {"schema", "llaminar.prepared-projection-output.v1"},
      {"complete", true},
      {"scope", "reference_input_or_recorded_input_prepared_component"},
      {"backend", options.device.toString()},
      {"physical_rows", options.m},
      {"live_rows", options.live},
      {"n", options.n},
      {"k", options.k},
      {"exported_elements", elements},
      {"bytes_per_output", elements * sizeof(float)},
      {"format", "float32_le_row_major"},
      {"phase", options.verifier ? "verifier"
                : options.serial ? "serial"
                                 : "ordinary"},
      {"implementation_expectation", options.grouped ? "grouped" : "vector"},
      {"declared_source_name", options.sourceName},
      {"source_format", options.sourceFormat},
      {"source_codebook_id", source.source_codebook_id},
      {"source_is_superblock", bool(source.source_is_superblock)},
      {"execution_codebook_id", source.codebook_id},
      {"source_identity_present", bool(source.source_identity_present)},
      {"selected_weight_file",
       std::filesystem::absolute(options.weights).lexically_normal().string()},
      {"selected_weight_bytes", weightBytes},
      {"input_file",
       std::filesystem::absolute(options.input).lexically_normal().string()},
      {"saved_input_rows", savedRows},
      {"input_bytes", savedRows * std::size_t(options.k) * sizeof(float)},
      {"input_mode", options.tile ? "tile_geometry_only" : "exact"},
      {"export_extent", "declared_live_prefix_only"},
      {"extra_native_readbacks", 0},
      {"timing_scope", "unchanged; export after all timing and verification"},
      {"outputs",
       {{"single", {{"file", "single.f32"}, {"precision", "SingleQ8"}}},
        {"two_term", {{"file", "two-term.f32"}, {"precision", "TwoTermQ8"}}}}}};
  const auto temporary = directory / "metadata.json.tmp";
  std::ofstream file(temporary);
  require(bool(file), "Cannot open export metadata");
  file << metadata.dump(2) << '\n';
  file.flush();
  require(bool(file), "Incomplete export metadata");
  file.close();
  require(!file.fail(), "Export metadata close failed");
  std::filesystem::rename(temporary, directory / "metadata.json");
}
/** @brief Time one exact original-source projection without profiler or timed
 * host observation. */
void measure(const MeasureOptions &o) {
  auto *backend = getBackendFor(o.device);
  require(backend && backend->deviceCount() > o.device.ordinal,
          "Requested backend unavailable");
  auto &worker = GPUDeviceContextPool::instance().getContext(o.device);
  auto context = IDeviceContext::create(o.device, 1);
  worker
      .submitAsync([&] {
        Storage storage{o.device, backend,
                        ExplicitGPUStream(worker.defaultStream()).get()};
        const auto &format = test::quantizedVerifierFormat(o.sourceFormat);
        const auto block = format.source_is_superblock ? 256u : 32u;
        const auto blockBytes = cpu::native_vnni::native_block_bytes_for_format(
            format.source_codebook_id, format.source_is_superblock);
        const auto sourceBytes = std::size_t(o.n) *
                                 ((std::size_t(o.k) + block - 1) / block) *
                                 blockBytes;
        auto raw = load<std::uint8_t>(o.weights.c_str(), sourceBytes, true);
        MPIContext sourceMPI(0, 1, MPI_COMM_NULL);
        TensorFactory sourceFactory(sourceMPI);
        auto weight = test::q8_all_formats::originalProjectionSource(
            sourceFactory, format.tensor_type, {std::size_t(o.n), std::size_t(o.k)}, raw);
        require(weight && weight->size_bytes() == sourceBytes &&
                    std::memcmp(weight->raw_data(), raw.data(), sourceBytes) ==
                        0,
                "Original native quantized source extent/bits invalid");
        PreparedWeightStore store{ModelContextId{801977}};
        auto input = std::make_shared<FP32Tensor>(
            std::vector<std::size_t>{std::size_t(o.m), std::size_t(o.k)},
            DeviceId::cpu());
        const auto saved =
            load<float>(o.input.c_str(),
                        o.tile ? std::size_t(1u << 28)
                               : std::size_t(o.m) * o.k * sizeof(float),
                        !o.tile);
        require(saved.size() % o.k == 0, "Saved activation row width invalid");
        const auto savedRows = saved.size() / o.k;
        require(o.tile || savedRows == std::size_t(o.m),
                "Exact input mode requires exactly M saved rows");
        for (int row = 0; row < o.m; ++row)
          std::copy_n(saved.data() + std::size_t(row % savedRows) * o.k, o.k,
                      input->mutable_data() + std::size_t(row) * o.k);
        require(std::all_of(saved.begin(), saved.end(),
                            [](float x) { return std::isfinite(x); }),
                "Saved activation contains nonfinite values");
        std::unique_ptr<Projection> single, two;
        MeasureEvents events{backend, o.device};
        CapturedWorkRetirement retirement(storage);
        auto binding = registerEngine(store, *weight, o.device,
                                      prepare(*weight, raw, storage, o.n, o.k),
                                      o.sourceName);
        DeviceNativeVNNIMatrixDesc sourceBefore;
        auto *kernel = store.gemmKernel(*binding.prepared);
        require(kernel && kernel->exportNativeVNNIMatrixDesc(sourceBefore),
                "Prepared original source unavailable");
        TransferEngine::prepareDeviceInput(input.get(), o.device,
                                           storage.stream);
        TransferEngine::requireDeviceInput(input.get(), o.device,
                                           storage.stream);
        const auto *count =
            o.serial ? nullptr : storage.input(std::vector<int32_t>{o.live});
        auto *start = events.add();
        auto *stop = events.add();
        if(o.comparison == ProjectionComparison::OriginalVsShortK) {
          measureShortK(storage,o,worker,context.get(),input.get(),weight.get(),store,binding,count,
                        start,stop,sourceBefore,raw);
          return;
        }
        single = std::make_unique<Projection>(
            storage, o, input.get(), weight.get(), store, binding,
            QuantizedActivationOperandPrecision::SingleQ8, count);
        two = std::make_unique<Projection>(
            storage, o, input.get(), weight.get(), store, binding,
            QuantizedActivationOperandPrecision::TwoTermQ8, count);
        single->capture(worker, context.get(), true);
        two->capture(worker, context.get(), true);
        single->dirtyOutput();
        two->dirtyOutput();
        const auto allocationOwners = storage.owners->size();
        const auto inputPointer = input->gpu_data_ptr();
        for (int i = 0; i < o.warmup; ++i) {
          require(single->graph->launch(), "Single warm replay failed");
          require(two->graph->launch(), "TwoTerm warm replay failed");
        }
        const auto expected = two->observe();
        auto observedSingle = single->observe();
        std::cout << "SOURCE name=" << o.sourceName
                  << " source_format=" << o.sourceFormat
                  << " execution_codebook="
                  << unsigned(sourceBefore.codebook_id)
                  << " native_path=" << o.weights << " bytes=" << raw.size()
                  << " input_path=" << o.input << " saved_rows=" << savedRows
                  << " input_mode=" << (o.tile ? "tile_geometry_only" : "exact")
                  << " implementation=" << (o.grouped ? "grouped" : "vector")
                  << " count_owner=" << count << "\n";
        for (int batch = 0; batch < o.batches; ++batch) {
          float a, b;
          if (batch % 2 == 0) {
            a = elapsed(*single, start, stop, o.iterations);
            b = elapsed(*two, start, stop, o.iterations);
          } else {
            b = elapsed(*two, start, stop, o.iterations);
            a = elapsed(*single, start, stop, o.iterations);
          }
          const auto actual = two->observe();
          observedSingle = single->observe();
          require(actual.size() == expected.size() &&
                      (actual.empty() ||
                       std::memcmp(expected.data(), actual.data(),
                                   actual.size() * sizeof(float)) == 0),
                  "Timed replay changed live candidate bytes");
          require(storage.owners->size() == allocationOwners &&
                      input->gpu_data_ptr() == inputPointer,
                  "Retained allocation identity changed");
          DeviceNativeVNNIMatrixDesc sourceAfter;
          require(kernel->exportNativeVNNIMatrixDesc(sourceAfter) &&
                      std::memcmp(&sourceBefore, &sourceAfter,
                                  sizeof(sourceBefore)) == 0 &&
                      std::memcmp(weight->raw_data(), raw.data(), raw.size()) ==
                          0,
                  "Immutable prepared native source changed");
          require(std::memcmp(raw.data(), weight->raw_data(), raw.size()) == 0,
                  "Original host native source changed");
          std::cout << "CAPTURED_PROJECTION_TIMING backend="
                    << o.device.toString() << " M=" << o.m << " N=" << o.n
                    << " K=" << o.k << " live=" << o.live << " phase="
                    << (o.verifier ? "verifier"
                        : o.serial ? "serial"
                                   : "ordinary")
                    << " batch=" << batch
                    << " order=" << (batch % 2 ? "BA" : "AB")
                    << " graph_submissions=" << o.iterations
                    << " single_us=" << a << " two_term_us=" << b
                    << " ratio=" << b / a
                    << " scope=outer_complete_native_transaction "
                       "includes_submission_gaps=1 timed_host_observations=0\n";
        }
        exportObservedOutputs(o, savedRows, raw.size(), sourceBefore,
                              observedSingle, expected);
      })
      .get();
}
} // namespace
#endif // Native GPU measurement implementation; CPU-only builds need no GPU
       // headers.
/** @brief Dispatch only an explicit CPU/native request; no backend
 * initialization precedes parsing. */
int main(int argc, char **argv) {
  try {
    for (int index=1;index<argc;++index)
      if (std::string_view(argv[index]).starts_with("--gtest_filter=")) {
        ::testing::InitGoogleTest(&argc,argv);
        return RUN_ALL_TESTS();
      }
    if (llaminar2::test::recorded_gdn::requested(argc, argv))
      return llaminar2::test::recorded_gdn::run(argc, argv);
    if (llaminar2::test::recorded_head::requested(argc, argv))
      return llaminar2::test::recorded_head::run(argc, argv);
    if (llaminar2::test::q8_all_formats::q8ResidualFixtureExportRequested(argc, argv))
      return llaminar2::test::q8_all_formats::exportQ8ResidualFixtures(argc, argv);
    if (llaminar2::test::q8_all_formats::cpuQ8ResidualMeasurementRequested(
            argc, argv))
      return llaminar2::test::q8_all_formats::runCPUQ8ResidualMeasurement(argc,
                                                                          argv);
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
    const auto o = options(argc, argv);
    measure(o);
    return 0;
#else
    throw std::invalid_argument(
        "This projection measurement was built without a native GPU backend; "
        "use explicit --backend cpu");
#endif
  } catch (const std::exception &e) {
    std::cerr << "FATAL " << e.what() << "\n";
    return 1;
  }
}
