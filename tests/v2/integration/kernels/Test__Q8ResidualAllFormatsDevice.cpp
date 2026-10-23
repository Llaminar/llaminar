/**
 * @file Test__Q8ResidualAllFormatsDevice.cpp
 * @brief Actual original-format native-pool counted/TwoTerm capture and independent
 * FP64 tests.
 *
 * Preparation uses existing LoadOrchestrator/WeightVRAMPool/store
 * infrastructure. Source/stateless coefficient decoding and FP64 dots never
 * call GEMM reduction helpers. Host copies and joins are diagnostic
 * setup/terminal boundaries only. Counted SingleQ8 regressions additionally
 * retain exact live output, activation-byte, scale and sums extents across
 * full/partial/empty replay; empty work cannot write native partial scratch.
 */
#include "utils/GpuPreparedGemmHarness.h"
#include "utils/PerfStatsCollector.h"
#include "moe/NativeVNNIExpertTransferParityTest.h"
#include "../../utils/Q8ResidualAllFormatOracle.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/compute_stages/ComputeStageUtils.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/common/GPUQ8ResidualNative.h"
#include "kernels/common/Q8ResidualGroupedGeometry.h"
#include "kernels/common/Q8ResidualShortKGeometry.h"
#include "transfer/TransferEngine.h"
#ifdef HAVE_CUDA
#include "kernels/cuda/gemm/CUDAQuantisedActivationProducers.h"
#endif
#ifdef HAVE_ROCM
#include "kernels/rocm/gemm/ROCmQuantisedActivationProducers.h"
#endif
#include <cstdlib>
#include <string>
#include <future>
#include <gtest/gtest.h>
#include <iostream>
#include <optional>
#include <map>
#include <tuple>
using namespace llaminar2;
namespace {
namespace oracle = test::q8_all_formats;
/** @brief Reject missing original fixture infrastructure before native
 * submission. */
void require(bool value, const char *why) {
  if (!value)
    throw std::runtime_error(why);
}
struct Storage {
  DeviceId device;
  IBackend *backend;
  void *stream;
  std::shared_ptr<std::vector<std::shared_ptr<DeviceTransferBuffer>>> owners =
      std::make_shared<std::vector<std::shared_ptr<DeviceTransferBuffer>>>();
  std::unique_ptr<IGPUGraphCapture> graph;
  /** @brief Retire submitted diagnostic work, then its graph, before captured
   * native allocations. */
  ~Storage() { retire(); }
  /** @brief Complete native work and retire its graph; teardown failure is
   * fatal. This is a diagnostic terminal boundary, never a per-stage inference
   * sync.
   */
  void retire() noexcept {
    if (!backend->synchronizeStream(stream, device.ordinal))
      std::terminate();
    if (graph) {
      graph->reset();
      graph.reset();
    }
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
    const bool copied = backend->hostToDeviceOnStream(
        p, x.data(), x.size() * sizeof(T), device.ordinal, stream);
    if (!backend->synchronizeStream(stream, device.ordinal))
      std::terminate();
    require(
        copied,
        "Correction fixture publication failed after its exact stream retired");
  }
  /** @brief Materialize and initialize an immutable fixture plane before
   * capture. */
  template <class T> T *input(const std::vector<T> &x) {
    auto *p = allocate<T>(x.size());
    upload(p, x);
    return p;
  }
  /** @brief Terminally observe only the explicitly requested live prefix; empty
   * observations transfer nothing. */
  template <class T> std::vector<T> download(const T *p, std::size_t n) {
    if (!n)
      return {};
    std::vector<T> x(n);
    const bool copied = backend->deviceToHostOnStream(
        x.data(), p, n * sizeof(T), device.ordinal, stream);
    if (!backend->synchronizeStream(stream, device.ordinal))
      std::terminate();
    require(copied, "Correction fixture terminal copy failed after its exact "
                    "stream retired");
    return x;
  }
};
/** @brief Retire captured work before every captured tensor/workspace owner.
 * Declare this scope after all such owners and before the first native submit;
 * exception and ASSERT unwind then follows the same completion-before-storage
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
/** @brief End the stage borrow before its admitted workspace owner. */
struct StageBorrow {
  GEMMStage &stage;
  StageBorrow(GEMMStage &s, DeviceWorkspaceManager *w) : stage(s) {
    try {
      stage.bindWorkspace(w);
    } catch (...) {
      stage.unbindWorkspace();
      throw;
    }
  }
  ~StageBorrow() { stage.unbindWorkspace(); }
};
/** @brief Actual native source planes, including declared reusable-capacity
 * bytes. */
struct Planes {
  std::vector<uint8_t> payload;
  std::vector<uint16_t> scales, mins;
  std::vector<uint32_t> emins;
  bool operator==(const Planes &) const = default;
};
/** @brief Terminal snapshot of actual source allocations, never a hot-path
 * source mirror. */
Planes readPlanes(Storage &s, const DeviceNativeVNNIMatrixDesc &d) {
  const auto physical = requireQ8ResidualNativeMatrix(d, 1, d.n, d.k);
  const auto blocks = std::size_t(d.n) * d.blocks_per_row;
  return {
      s.download(d.payload, blocks * (d.allocation_payload_bytes_per_block
                                          ? d.allocation_payload_bytes_per_block
                                          : physical.payload_bytes_per_block)),
      s.download(static_cast<const uint16_t *>(d.scales), blocks),
      d.mins ? s.download(static_cast<const uint16_t *>(d.mins), blocks)
             : std::vector<uint16_t>{},
      d.emins ? s.download(static_cast<const uint32_t *>(d.emins), blocks)
              : std::vector<uint32_t>{}};
}
/** @brief Scalar coefficient interpretation independent of GPU integer dots and
 * FP32 contribution/reduction helpers. */
std::vector<double> physicalWeights(const DeviceNativeVNNIMatrixDesc &d,
                                    const Planes &p) {
  const auto physical = requireQ8ResidualNativeMatrix(d, 1, d.n, d.k);
  std::vector<double> w(std::size_t(d.n) * d.k);
  for (int b = 0; b < d.k / 32; ++b)
    for (int n = 0; n < d.n; ++n) {
      const auto i = std::size_t(b) * d.n + n;
      const auto values = oracle::coefficients(
          d.codebook_id,
          std::span<const uint8_t>(p.payload.data() +
                                       i * physical.payload_bytes_per_block,
                                   physical.payload_bytes_per_block),
          p.scales[i], p.mins.empty() ? 0 : p.mins[i],
          p.emins.empty() ? 0 : p.emins[i]);
      std::copy(values.begin(), values.end(),
                w.begin() + std::size_t(n) * d.k + b * 32);
    }
  return w;
}
/** @brief Missing public-factory IQ view must fail before publishing any
 * operand or output. */
void rejectMissingIQ(Storage &s, const DeviceNativeVNNIMatrixDesc &d,
                     DeviceRowRange rows, FP32Tensor &input, FP32Tensor &output,
                     DeviceWorkspaceManager &w) {
  if (!q8ResidualNeedsIQTables(d.codebook_id))
    return;
  const GPUQ8ResidualCorrectionArgs a{
      d,
      rows,
      s.device.is_cuda()
          ? Q8ResidualQuantizerPolicy::CUDAFloatScaleNativeReciprocal
          : Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal,
      static_cast<const float *>(input.gpu_data_ptr()),
      static_cast<float *>(output.gpu_data_ptr()),
      static_cast<signed char *>(w.getBuffer(kGPUProjectionPrimaryQ8)),
      static_cast<signed char *>(w.getBuffer(kGPUProjectionResidualQ8)),
      static_cast<float *>(w.getBuffer(kGPUProjectionPrimaryQ8Scales)),
      static_cast<float *>(w.getBuffer(kGPUProjectionResidualQ8Scales))};
#ifdef HAVE_CUDA
  if (s.device.is_cuda())
    EXPECT_THROW(cudaQ8ResidualCorrectedProjection(a, s.stream),
                 std::invalid_argument);
#endif
#ifdef HAVE_ROCM
  if (s.device.is_rocm())
    EXPECT_THROW(rocmQ8ResidualCorrectedProjection(a, s.stream),
                 std::invalid_argument);
#endif
}
/**
 * @brief Prepare a genuine cold CPU expert, then adopt its real promoted GPU engine.
 * @param source Original authored Q5_1 tensor; its source bytes are never repacked by the test.
 * @param device Exact GPU receiving the production persistent migration stream.
 * @param backend Existing backend owner for that endpoint.
 * @return Store-owned prepared engine retaining the promoted separated allocation.
 *
 * This covers the generic explicit TwoTerm API on a migrated expert projection.
 * HC dense replicas do not participate in expert movement; the test neither
 * enables expert TwoTerm by default nor claims an HC migration production path.
 */
test::GpuPreparedGemm preparePromotedAsymmetric(TensorBase *source, DeviceId device,
                                                IBackend *backend) {
  namespace transfer = test::native_vnni_transfer_parity_detail;
  namespace factory = llaminar::v2::kernels;
  constexpr const char *name = "blk.2.ffn_down.weight";
  auto cold = test::makePreparedGemmFixture(source, DeviceId::cpu(), name, ModelContextId{871122});
  auto *coldEngine =
      dynamic_cast<cpu::native_vnni::CPUNativeVNNIGemmKernel *>(cold.store->gemmKernel(cold.ref));
  require(coldEngine && coldEngine->isValid(), "Actual cold CPU prepared expert is absent");
  const auto &packed = coldEngine->packedWeights();
  require(packed.usesExpandedInt8() && packed.is_asymmetric,
          "Authored asymmetric source must reach real expanded CPU storage");
  const std::vector<uint8_t> before(packed.native_interleaved.begin(),
                                    packed.native_interleaved.end());
  auto allocation =
      std::make_shared<transfer::PromotedCPUProjection>(transfer::promoteCPUProjection(
          backend, device, packed, ExpertTierWeightProjection::Down, 871123));
  require(std::equal(before.begin(), before.end(), packed.native_interleaved.begin()),
          "Actual promotion mutated its cold CPU execution source");
  const auto &descriptor = allocation->descriptor;
  require(descriptor.codebook_id == kNativeVnniExpandedInt8MinCodebook &&
              descriptor.source_codebook_id == native_vnni_formats::Q5_1.codebook_id &&
              descriptor.source_is_superblock == 0 && descriptor.source_identity_present == 1 &&
              descriptor.mins && !descriptor.emins,
          "Production CPU promotion did not publish the original asymmetric physical23 contract");
  auto engine = transfer::promotedEngine(device, descriptor, allocation);
  const auto kind = device.is_cuda()
                        ? factory::KernelFactory::GemmPreparationKind::CUDA_INT8_PACKED
                        : factory::KernelFactory::GemmPreparationKind::ROCM_INT8_PACKED;
  auto weights = std::make_shared<factory::KernelFactory::PreparedGemmWeights>();
  weights->kind = kind;
  weights->owned_kernel = engine;
  weights->kernel = engine.get();
  auto handle = std::make_shared<factory::KernelFactory::PreparedGemmHandle>();
  handle->tensor = source;
  handle->device_id = device;
  handle->kind = kind;
  handle->variant = static_cast<int>(source->native_type());
  handle->prepared_weights = std::move(weights);
  test::GpuPreparedGemm out;
  // Preserve the exact original source binding while changing only its
  // prepared execution representation and current endpoint residency.
  const auto coldBinding = cold.store->binding(cold.ref);
  require(coldBinding.has_value(), "Actual cold prepared source binding is absent");
  out.binding = *coldBinding;
  out.binding.prepared.reset();
  out.binding.residency.resident_device = device;
  out.store = std::make_unique<PreparedWeightStore>(ModelContextId{871122});
  out.ref = out.store->registerPreparedGemmHandle(out.binding, test::preparedKindForDevice(device),
                                                  device, std::move(handle));
  out.kernel = out.store->gemmKernel(out.ref);
  const auto registeredBinding = out.store->binding(out.ref);
  require(out.kernel == engine.get() && out.ref.model_id == cold.ref.model_id &&
              out.ref.binding_id == cold.ref.binding_id &&
              registeredBinding && registeredBinding->tensor == source,
          "Promoted prepared-store source identity changed");
  return out;
}
/** @brief Scope only the existing collector policy; restore original optional environment values. */
class ProjectionSubmissionObservation final {
    std::optional<std::string> json_, filter_;
    /**
     * @param name One of the two collector environment keys owned by this scope.
     * @return Exact prior value, distinguishing an empty value from absence.
     */
    static std::optional<std::string> prior(const char *name) {
        const char *value = std::getenv(name);
        return value ? std::optional<std::string>{value} : std::nullopt;
    }
    /**
     * @brief Restore exactly the saved value or original absence; failed restoration is fatal.
     * @param name Collector environment key being retired.
     * @param value Original value, or absence to restore with unsetenv.
     */
    static void restore(const char *name, const std::optional<std::string> &value) {
        const int result = value ? setenv(name,value->c_str(),1) : unsetenv(name);
        if (result != 0) std::terminate();
    }
public:
    /** @brief Select accepted host submissions without enabling timers or another domain. */
    ProjectionSubmissionObservation()
        : json_(prior("LLAMINAR_PERF_STATS_JSON")),
          filter_(prior("LLAMINAR_PERF_STATS_FILTER")) {
        if (setenv("LLAMINAR_PERF_STATS_JSON","1",1) != 0 ||
            setenv("LLAMINAR_PERF_STATS_FILTER","kernel.q8_residual_ordered_short_k_submissions",1) != 0) {
            restore("LLAMINAR_PERF_STATS_FILTER",filter_);
            restore("LLAMINAR_PERF_STATS_JSON",json_);
            throw std::runtime_error("Cannot select projection submission observation environment");
        }
        PerfStatsCollector::reloadConfigurationFromEnvironment();
        PerfStatsCollector::reset();
    }
    /** @brief Retire fixture evidence, restore exact environment absence/values and reload it. */
    ~ProjectionSubmissionObservation() {
        PerfStatsCollector::reset();
        restore("LLAMINAR_PERF_STATS_FILTER",filter_);
        restore("LLAMINAR_PERF_STATS_JSON",json_);
        PerfStatsCollector::reloadConfigurationFromEnvironment();
    }
    ProjectionSubmissionObservation(const ProjectionSubmissionObservation&) = delete;
    ProjectionSubmissionObservation& operator=(const ProjectionSubmissionObservation&) = delete;
};
/** @return Exact accepted-submission records, excluding dotted neighboring families. */
std::vector<PerfStatRecord> projectionSubmissions() {
    auto records = PerfStatsCollector::snapshot({"kernel.q8_residual_ordered_short_k_submissions"});
    records.erase(std::remove_if(records.begin(),records.end(),[](const auto &record) {
        return record.domain != "kernel" || record.name != "q8_residual_ordered_short_k_submissions";
    }),records.end());
    return records;
}
/** @brief Explicit test setup choice; inference arithmetic is unchanged. */
enum class SourcePreparation { NativeGpu, PromotedCpuAsymmetric };
/** @brief Serial oracle choice only; every tested production capture uses the actual GEMMStage. */
enum class SerialProjectionReference { Production, OriginalColumn };
/**
 * @brief Submit the independent original-column diagnostic without production dispatch.
 * @param device Actual source/bank owner backend.
 * @param args Original prepared source, live authority and independent output/scratch.
 * @param stream Exact capture stream retaining those owners.
 * @return No value; native submissions are accepted or a typed exception is raised.
 */
void submitOriginalColumnReference(DeviceId device, const GPUQ8ResidualCorrectionArgs &args,
                                   void *stream)
{
#ifdef HAVE_CUDA
    if (device.is_cuda()) {
        cudaQ8ResidualOriginalColumnReference(args, stream);
        return;
    }
#endif
#ifdef HAVE_ROCM
    if (device.is_rocm()) {
        rocmQ8ResidualOriginalColumnReference(args, stream);
        return;
    }
#endif
    throw std::invalid_argument("Diagnostic projection backend is not compiled");
}
/**
 * @brief Prove actual prepared-stage capture, source math and retained live-prefix replay.
 * @param device Actual participant and original prepared-source owner.
 * @param format Canonical original quantized source fixture.
 * @param m Physical rows, independent of later live counts.
 * @param n Output width, including final-column tails.
 * @param k Complete native K32 reduction width.
 * @param tile Whether authored equal input rows label a larger geometry workload.
 * @param concurrentCold Whether cold IQ preparation queries run concurrently.
 * @param sourcePreparation Original native or actual promoted source path.
 * @param reference Independent original-column or ordinary serial reference.
 */
void prove(DeviceId device, const test::QuantizedVerifierFormatCase &format,
           int m, int n, int k, bool tile, bool concurrentCold = false,
           SourcePreparation sourcePreparation = SourcePreparation::NativeGpu,
           SerialProjectionReference reference = SerialProjectionReference::Production) {
  const bool promotedAsymmetric = sourcePreparation == SourcePreparation::PromotedCpuAsymmetric;
  SCOPED_TRACE(format.label);
  SCOPED_TRACE(::testing::Message() << device.toString() << " M=" << m << " N="
                                    << n << " K=" << k << " tile=" << tile);
  auto *backend = getBackendFor(device);
  require(backend && backend->deviceCount() > device.ordinal,
          "Requested native backend unavailable");
  auto &worker = GPUDeviceContextPool::instance().getContext(device);
  auto context = IDeviceContext::create(device, 1);
  worker
      .submitAsync([&] {
        Storage s{device, backend,
                  ExplicitGPUStream(worker.defaultStream()).get()};
        auto weight = oracle::source(format, n, k);
        const auto raw = oracle::originalBytes(*weight);
        const auto original = oracle::originalWeights(*weight);
        auto prepared = promotedAsymmetric
                            ? preparePromotedAsymmetric(weight.get(), device, backend)
                            : test::makeGpuPreparedGemm(weight.get(), device,
                                                        "blk.2.hc_ffn_up.weight", ModelContextId{871002});
        auto *kernel = prepared.kernel;
        require(kernel, "Actual prepared native source absent");
        kernel->setGPUStream(s.stream);
        DeviceNativeVNNIMatrixDesc desc{};
        require(kernel->exportNativeVNNIMatrixDesc(desc),
                "Prepared descriptor absent");
        ASSERT_EQ(desc.n, n);
        ASSERT_EQ(desc.k, k);
        EXPECT_EQ(desc.source_codebook_id, format.source_codebook_id);
        EXPECT_EQ(bool(desc.source_is_superblock), format.source_is_superblock);
        EXPECT_EQ(desc.codebook_id, promotedAsymmetric ? kNativeVnniExpandedInt8MinCodebook
                                                       : format.device_execution_codebook_id);
        EXPECT_EQ(desc.source_identity_present, 1);
        // Per-engine publication starts cold even when another earlier engine
        // has already initialized the original backend's global codebook
        // storage.
        if (concurrentCold) {
          std::array<std::future<WorkspaceRequirements>, 4> queries;
          for (auto &q : queries)
            q = std::async(std::launch::async, [&] {
              return kernel->q8ResidualCorrectionWorkspaceRequirements(m, n, k);
            });
          std::optional<WorkspaceRequirements> first;
          for (auto &q : queries) {
            auto r = q.get();
            if (!first)
              first = r;
            ASSERT_EQ(r.buffers.size(), first->buffers.size());
            for (std::size_t i = 0; i < r.buffers.size(); ++i) {
              EXPECT_EQ(r.buffers[i].name, first->buffers[i].name);
              EXPECT_EQ(r.buffers[i].size_bytes, first->buffers[i].size_bytes);
            }
          }
        }
        const auto sourceBefore = readPlanes(s, desc);
        const auto native = physicalWeights(desc, sourceBefore);
        FP32Tensor input({std::size_t(m), std::size_t(k)}, DeviceId::cpu()),
            output({std::size_t(m), std::size_t(n)}, DeviceId::cpu()),
            baseline({std::size_t(m), std::size_t(n)}, DeviceId::cpu()),
            rowInput({1, std::size_t(k)}, DeviceId::cpu()),
            rowOutput({1, std::size_t(n)}, DeviceId::cpu());
        const auto values = oracle::input(m, k, tile);
        std::copy(values.begin(), values.end(), input.mutable_data());
        auto *count = s.allocate<int32_t>(1);
        GEMMStage stage(
            {.device_id = device,
             .A = &input,
             .B = weight.get(),
             .C = &output,
             .m = m,
             .n = n,
             .k = k,
             .a_buffer_id = BufferId::NORMALIZED,
             .c_buffer_id = BufferId::GATE_PROJ,
             .force_decode_equivalent_verifier_prefill = true,
             .verifier_row_range = DeviceRowRange::deviceCounted(m, count),
             .prepared_ref = prepared.ref,
             .prepared_store = prepared.store.get(),
             .quantized_operand_precision =
                 QuantizedActivationOperandPrecision::TwoTermQ8});
        stage.setGPUStream(s.stream);
        Workspace correction(device, stage.getWorkspaceRequirements(m, n, k));
        StageBorrow borrow(stage, correction.value.get());
        auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
        require(consumer, "Prepared native workspace consumer absent");
        auto ordinaryRequirements = consumer->getWorkspaceRequirements(m, n, k);
        const std::array columns{n};
        consumer->appendFusedProjectionWorkspaceRequirements(
            ordinaryRequirements, m, columns, k);
        Workspace ordinary(device, ordinaryRequirements);
        CapturedWorkRetirement retire(s);
        for (auto *t : std::array<TensorBase *, 5>{&input, &output, &baseline,
                                                   &rowInput, &rowOutput}) {
          TransferEngine::prepareDeviceInput(t, device, s.stream);
          TransferEngine::requireDeviceInput(t, device, s.stream);
        }
        s.upload(count, std::vector<int32_t>{m});
        const std::vector<ITensorGemm::TensorProjectionDesc> projection{
            {kernel, &baseline, n, nullptr, "original.native.source"}};
        if (m == 15) {
          auto scope = kernel->beginVerifierDecodeEquivalentScope(
              DeviceRowRange::fullyActive(m));
          ASSERT_TRUE(kernel->multiply_fused_verifier_rows_decode_equivalent(
              &input, projection, m, k, nullptr, ordinary.value.get()));
        } else
          ASSERT_TRUE(kernel->multiply_tensor(
              &input, &baseline, m, n, k, false, 1, 0, nullptr, nullptr,
              device.ordinal, ordinary.value.get()));
        const auto ordinaryPrimary =
            s.download(static_cast<const signed char *>(
                           ordinary.value->getBuffer(kGPUProjectionPrimaryQ8)),
                       std::size_t(m) * k);
        const auto ordinaryScales =
            s.download(static_cast<const float *>(ordinary.value->getBuffer(
                           kGPUProjectionPrimaryQ8Scales)),
                       std::size_t(m) * k / 32);
        const auto ordinaryOutput =
            s.download(static_cast<const float *>(baseline.gpu_data_ptr()),
                       std::size_t(m) * n);

        // Reuse M1 expectations only after proving exact source input row
        // identity.
        const int uniqueRows = tile ? 1 : std::min(m, 7);
        for (int r = 0; r < m; ++r)
          ASSERT_EQ(std::memcmp(values.data() + std::size_t(r) * k,
                                values.data() + std::size_t(r % uniqueRows) * k,
                                std::size_t(k) * sizeof(float)),
                    0);
        auto *serial = s.allocate<float>(std::size_t(uniqueRows) * n);
        auto *serial0 = s.allocate<signed char>(std::size_t(uniqueRows) * k);
        auto *serial1 = s.allocate<signed char>(std::size_t(uniqueRows) * k);
        auto *scale0 = s.allocate<float>(std::size_t(uniqueRows) * k / 32);
        auto *scale1 = s.allocate<float>(std::size_t(uniqueRows) * k / 32);
        const auto copy = [&](void *out, const void *in, std::size_t bytes) {
          require(
              backend->deviceToDevice(out, in, bytes, device.ordinal, s.stream),
              "Independent native row copy failed");
        };
        if (reference == SerialProjectionReference::OriginalColumn) {
          PerfStatsCollector::reset();
          const GPUQ8ResidualCorrectionArgs referenceArgs{
              .weights = desc, .rows = DeviceRowRange::fullyActive(1),
              .quantizer = device.is_cuda()
                  ? Q8ResidualQuantizerPolicy::CUDAFloatScaleNativeReciprocal
                  : Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal,
              .input = static_cast<const float *>(rowInput.gpu_data_ptr()),
              .output = static_cast<float *>(rowOutput.gpu_data_ptr()),
              .primary = static_cast<signed char *>(correction.value->getBuffer(kGPUProjectionPrimaryQ8)),
              .residual = static_cast<signed char *>(correction.value->getBuffer(kGPUProjectionResidualQ8)),
              .primaryScales = static_cast<float *>(correction.value->getBuffer(kGPUProjectionPrimaryQ8Scales)),
              .residualScales = static_cast<float *>(correction.value->getBuffer(kGPUProjectionResidualQ8Scales))};
          auto rejected = referenceArgs;
          rejected.input = nullptr;
#ifdef HAVE_CUDA
          if (device.is_cuda()) EXPECT_THROW(cudaQ8ResidualCorrectedProjection(rejected,s.stream),std::invalid_argument);
#endif
#ifdef HAVE_ROCM
          if (device.is_rocm()) EXPECT_THROW(rocmQ8ResidualCorrectedProjection(rejected,s.stream),std::invalid_argument);
#endif
          EXPECT_TRUE(projectionSubmissions().empty()) << "Rejected production calls must not issue receipts";
          s.graph = worker.createGraphCapture(s.stream);
          require(bool(s.graph), "Original-column reference capture unavailable");
          {
            ScopedBackendGraphCapture capture(worker, *s.graph, "Independent original-column reference");
            require(capture.begin(), "Original-column reference capture failed");
            auto unsupported = referenceArgs;
            unsupported.rows = DeviceRowRange::fullyActive(2);
            EXPECT_THROW(submitOriginalColumnReference(device, unsupported, s.stream), std::invalid_argument);
            unsupported = referenceArgs;
            unsupported.weights.codebook_id = 20;
            EXPECT_THROW(submitOriginalColumnReference(device, unsupported, s.stream), std::invalid_argument);
            submitOriginalColumnReference(device, referenceArgs, s.stream);
            capture.finish();
          }
          require(s.graph->instantiate(), "Original-column reference instantiate failed");
          std::vector<GPUGraphKernelNodeInfo> referenceNodes;
          std::string referenceError;
          require(s.graph->inspectKernelNodes(referenceNodes, &referenceError),
                  "Original-column reference inspection failed");
          ASSERT_EQ(referenceNodes.size(), 2u);
          unsigned referenceProjects = 0, referencePublishers = 0;
          for (const auto &node : referenceNodes) {
            ASSERT_EQ(node.name.find("projectShortK"), std::string::npos);
            EXPECT_TRUE(node.name_resolved);
            if (node.name.find("publish") != std::string::npos) {
              ++referencePublishers;
              EXPECT_EQ(node.block_x, 256u);
              EXPECT_EQ(node.grid_x, (std::uint64_t(k / 32) + 7) / 8);
            }
            if (node.name.find("project") != std::string::npos) {
              ++referenceProjects;
              EXPECT_TRUE(node.name.find("project<1>") != std::string::npos ||
                          node.name.find("projectILi1EE") != std::string::npos);
              EXPECT_EQ(node.block_x, 128u);
              EXPECT_EQ(node.grid_x, (std::uint64_t(n) + 127) / 128);
              EXPECT_EQ(node.grid_y, 1u); EXPECT_EQ(node.grid_z, 1u);
            }
          }
          EXPECT_EQ(referenceProjects, 1u);
          EXPECT_EQ(referencePublishers, 1u);
        }
        for (int r = 0; r < uniqueRows; ++r) {
          copy(rowInput.gpu_data_ptr(),
               static_cast<const float *>(input.gpu_data_ptr()) +
                   std::size_t(r) * k,
               std::size_t(k) * sizeof(float));
          if (reference == SerialProjectionReference::OriginalColumn)
            require(s.graph->launch(), "Independent original-column replay failed");
          else
            kernel->multiply_q8_residual_corrected(&rowInput, &rowOutput,
                                                 DeviceRowRange::fullyActive(1),
                                                 n, k, correction.value.get());
          copy(serial + std::size_t(r) * n, rowOutput.gpu_data_ptr(),
               std::size_t(n) * sizeof(float));
          copy(serial0 + std::size_t(r) * k,
               correction.value->getBuffer(kGPUProjectionPrimaryQ8),
               std::size_t(k));
          copy(serial1 + std::size_t(r) * k,
               correction.value->getBuffer(kGPUProjectionResidualQ8),
               std::size_t(k));
          copy(scale0 + std::size_t(r) * k / 32,
               correction.value->getBuffer(kGPUProjectionPrimaryQ8Scales),
               std::size_t(k) / 32 * sizeof(float));
          copy(scale1 + std::size_t(r) * k / 32,
               correction.value->getBuffer(kGPUProjectionResidualQ8Scales),
               std::size_t(k) / 32 * sizeof(float));
        }
        const auto serialOutput =
            s.download(serial, std::size_t(uniqueRows) * n);
        const auto serialScales0 =
            s.download(scale0, std::size_t(uniqueRows) * k / 32);
        const auto serialScales1 =
            s.download(scale1, std::size_t(uniqueRows) * k / 32);
        const auto serialPrimary =
            s.download(serial0, std::size_t(uniqueRows) * k);
        const auto serialResidual =
            s.download(serial1, std::size_t(uniqueRows) * k);
        // Reference replay is complete before its captured graph retires; all
        // original source/bank owners remain retained for actual stage capture.
        if (reference == SerialProjectionReference::OriginalColumn) {
          s.retire();
          EXPECT_TRUE(projectionSubmissions().empty()) << "Independent oracle must not issue production receipts";
        }
        for (int r = 0; r < m; ++r) {
          const auto frozen = r % uniqueRows;
          ASSERT_EQ(std::memcmp(ordinaryPrimary.data() + std::size_t(r) * k,
                                serialPrimary.data() + std::size_t(frozen) * k,
                                std::size_t(k)),
                    0);
          ASSERT_EQ(
              std::memcmp(ordinaryScales.data() + std::size_t(r) * k / 32,
                          serialScales0.data() + std::size_t(frozen) * k / 32,
                          std::size_t(k) / 32 * sizeof(float)),
              0);
        }
        // Independent FP64 work is bounded by distinct input rows, not replay
        // count.
        oracle::Banks frozenBanks;
        frozenBanks.primary.assign(serialPrimary.begin(), serialPrimary.end());
        frozenBanks.residual.assign(serialResidual.begin(),
                                    serialResidual.end());
        frozenBanks.d0 = serialScales0;
        frozenBanks.d1 = serialScales1;
        const auto expected =
            oracle::project(native, n, k, frozenBanks, uniqueRows, true);
        const auto unquantized =
            oracle::exact(native, n, k, values, uniqueRows);
        const auto sourceExact =
            oracle::exact(original, n, k, values, uniqueRows);
        double e0 = 0, e1 = 0, sourceError = 0;
        for (int r = 0; r < uniqueRows; ++r)
          for (int c = 0; c < n; ++c) {
            const auto i = std::size_t(r) * n + c;
            ASSERT_TRUE(std::isfinite(serialOutput[i]));
            EXPECT_NEAR(double(serialOutput[i]), expected[i],
                        oracle::roundingBound(native, c, k, frozenBanks, r));
            EXPECT_NEAR(double(serialOutput[i]), sourceExact[i],
                        oracle::originalProjectionBound(
                            original, native, c, k, frozenBanks, r, values));
            e0 += std::pow(double(ordinaryOutput[i]) - unquantized[i], 2);
            e1 += std::pow(double(serialOutput[i]) - unquantized[i], 2);
            sourceError +=
                std::pow(double(serialOutput[i]) - sourceExact[i], 2);
          }
        EXPECT_GT(e0, 0);
        EXPECT_LT(e1, e0 / 16);
        std::cout
            << "Q8_TWO_TERM_ORACLE scope=unique_prepared_native_rows backend="
            << device.toString() << " source=" << format.label
            << " physical_rows=" << m << " unique_rows=" << uniqueRows
            << " original_fp64_squared_error=" << sourceError
            << " prepared_fp64_squared_error=" << e1
            << " single_prepared_squared_error=" << e0 << "\n";
        ASSERT_FALSE(::testing::Test::HasFailure());
        auto *a0 = static_cast<signed char *>(
            correction.value->getBuffer(kGPUProjectionPrimaryQ8));
        auto *a1 = static_cast<signed char *>(
            correction.value->getBuffer(kGPUProjectionResidualQ8));
        auto *d0 = static_cast<float *>(
            correction.value->getBuffer(kGPUProjectionPrimaryQ8Scales));
        auto *d1 = static_cast<float *>(
            correction.value->getBuffer(kGPUProjectionResidualQ8Scales));
        const std::vector<signed char> dirty(std::size_t(m) * k, 90);
        const std::vector<float> dirtyScales(std::size_t(m) * k / 32, -8765.f),
            dirtyOutput(std::size_t(m) * n, -12345.f);
        s.upload(a0, dirty);
        s.upload(a1, dirty);
        s.upload(d0, dirtyScales);
        s.upload(d1, dirtyScales);
        s.upload(static_cast<float *>(output.gpu_data_ptr()), dirtyOutput);
        rejectMissingIQ(s, desc, DeviceRowRange::deviceCounted(m, count), input,
                        output, *correction.value);
        EXPECT_EQ(s.download(a0, dirty.size()), dirty);
        EXPECT_EQ(s.download(a1, dirty.size()), dirty);
        EXPECT_EQ(s.download(d0, dirtyScales.size()), dirtyScales);
        EXPECT_EQ(s.download(d1, dirtyScales.size()), dirtyScales);
        EXPECT_EQ(s.download(static_cast<const float *>(output.gpu_data_ptr()),
                             dirtyOutput.size()),
                  dirtyOutput);
        if (reference == SerialProjectionReference::OriginalColumn) {
          PerfStatsCollector::reset();
          // Zero is device-owned input data, not a host-submission suppression
          // condition. Capture accepts once; retained live/empty replays do not
          // revisit the factory or add host receipts.
          s.upload(count,std::vector<int32_t>{0});
        }
        s.graph = worker.createGraphCapture(s.stream);
        require(bool(s.graph), "Native graph capture unavailable");
        {
          ScopedBackendGraphCapture capture(worker, *s.graph,
                                            "Original all-format TwoTerm");
          require(capture.begin(), "Native all-format capture failed");
          require(stage.execute(context.get()), "Actual stage capture failed");
          capture.finish();
        }
        require(s.graph->instantiate(),
                "Native all-format instantiation failed");
        std::vector<GPUGraphKernelNodeInfo> nodes;
        std::string why;
        require(s.graph->inspectKernelNodes(nodes, &why),
                "Actual kernel metadata inspection failed");
        ASSERT_EQ(nodes.size(), 2u);
        const auto geometry = q8ResidualGroupedGeometry(
            device.is_cuda() ? Q8ResidualGroupedBackend::CUDA
                             : Q8ResidualGroupedBackend::ROCm,
            m, n, desc.codebook_id);
        const auto shortGeometry = q8ResidualShortKGeometry(
            device.is_cuda() ? Q8ResidualGroupedBackend::CUDA : Q8ResidualGroupedBackend::ROCm,
            m, n, k, desc.codebook_id);
        const auto threads = shortGeometry ? shortGeometry->threads : geometry.threads;
        const auto blocks = shortGeometry ? shortGeometry->blocks : geometry.blocks;
        unsigned publishers = 0, projects = 0;
        for (const auto &node : nodes) {
          EXPECT_EQ(node.local_memory_bytes_per_thread, 0u);
          EXPECT_EQ(node.grid_y, 1u);
          EXPECT_EQ(node.grid_z, 1u);
          EXPECT_EQ(node.block_y, 1u);
          EXPECT_EQ(node.block_z, 1u);
          if (node.name.find("publish") != std::string::npos) {
            ++publishers;
            EXPECT_EQ(node.block_x, 256u);
            EXPECT_EQ(node.grid_x, (std::uint64_t(m) * (k / 32) + 7) / 8);
          }
          if (node.name.find("project") != std::string::npos) {
            ++projects;
            EXPECT_EQ(node.block_x, static_cast<unsigned>(threads));
            EXPECT_EQ(node.grid_x, blocks);
            EXPECT_EQ(node.name.find("projectShortK") != std::string::npos, bool(shortGeometry));
            if (shortGeometry) {
              EXPECT_NE(node.name.find("projectShortK"), std::string::npos);
              EXPECT_EQ(node.dynamic_shared_memory_bytes, 0u);
              EXPECT_EQ(node.static_shared_memory_bytes, shortGeometry->shared_bytes);
            }
          }
        }
        EXPECT_EQ(publishers, 1u);
        EXPECT_EQ(projects, 1u);
        std::vector<PerfStatRecord> capturedSubmissions;
        if (reference == SerialProjectionReference::OriginalColumn) {
          capturedSubmissions = projectionSubmissions();
          ASSERT_EQ(capturedSubmissions.size(),shortGeometry?1u:0u);
          if (shortGeometry) {
            const auto &record = capturedSubmissions.front();
            EXPECT_EQ(record.kind,PerfStatRecord::Kind::Counter);
            EXPECT_EQ(record.phase,"submission"); EXPECT_TRUE(record.device.empty());
            EXPECT_EQ(record.count,1u); EXPECT_EQ(record.value,1.0);
            EXPECT_EQ(record.tags.at("backend"),device.is_cuda()?"cuda":"rocm");
            EXPECT_EQ(record.tags.at("observation"),"host_execute_or_capture");
            EXPECT_EQ(record.tags.at("kernel_family"),"ordered_short_k");
            EXPECT_EQ(record.tags.at("physical_rows"),std::to_string(m));
            EXPECT_EQ(record.tags.at("n"),std::to_string(n)); EXPECT_EQ(record.tags.at("k"),std::to_string(k));
            EXPECT_EQ(record.tags.at("execution_codebook"),std::to_string(desc.codebook_id));
            for(const auto &node:nodes) if(node.name.find("project")!=std::string::npos) {
              EXPECT_EQ(record.tags.at("launch_blocks"),std::to_string(node.grid_x));
              EXPECT_EQ(record.tags.at("launch_threads"),std::to_string(node.block_x));
              EXPECT_EQ(record.tags.at("static_shared_bytes"),std::to_string(node.static_shared_memory_bytes));
            }
          }
        }
        const auto signature = stage.graphCaptureVariantSignature(),
                   used = correction.value->used(),
                   ownerCount = s.owners->size();
        const auto *inputPointer = input.gpu_data_ptr();
        const auto *outputPointer = output.gpu_data_ptr();
        for (int live : std::array{m, std::min(m, 9), std::min(m, 1), 0,
                                   std::min(m, 15), m}) {
          SCOPED_TRACE(live);
          auto current = values;
          std::fill(current.begin() + std::size_t(live) * k, current.end(),
                    std::numeric_limits<float>::quiet_NaN());
          s.upload(static_cast<float *>(input.gpu_data_ptr()), current);
          s.upload(count, std::vector<int32_t>{live});
          s.upload(a0, dirty);
          s.upload(a1, dirty);
          s.upload(d0, dirtyScales);
          s.upload(d1, dirtyScales);
          s.upload(static_cast<float *>(output.gpu_data_ptr()), dirtyOutput);
          require(s.graph->launch(),
                  "Retained native all-format replay failed");
          const auto actual =
              s.download(static_cast<const float *>(output.gpu_data_ptr()),
                         std::size_t(m) * n);
          const auto p = s.download(a0, std::size_t(m) * k),
                     q = s.download(a1, std::size_t(m) * k);
          const auto x = s.download(d0, std::size_t(m) * k / 32),
                     y = s.download(d1, std::size_t(m) * k / 32);
          for (int r = 0; r < live; ++r) {
            const auto frozen = r % uniqueRows;
            ASSERT_EQ(std::memcmp(actual.data() + std::size_t(r) * n,
                                  serialOutput.data() + std::size_t(frozen) * n,
                                  std::size_t(n) * sizeof(float)),
                      0)
                << "row=" << r;
            ASSERT_EQ(
                std::memcmp(p.data() + std::size_t(r) * k,
                            serialPrimary.data() + std::size_t(frozen) * k,
                            std::size_t(k)),
                0)
                << "primary row=" << r;
            ASSERT_EQ(
                std::memcmp(q.data() + std::size_t(r) * k,
                            serialResidual.data() + std::size_t(frozen) * k,
                            std::size_t(k)),
                0)
                << "residual row=" << r;
            ASSERT_EQ(
                std::memcmp(x.data() + std::size_t(r) * k / 32,
                            serialScales0.data() + std::size_t(frozen) * k / 32,
                            std::size_t(k) / 32 * sizeof(float)),
                0);
            ASSERT_EQ(
                std::memcmp(y.data() + std::size_t(r) * k / 32,
                            serialScales1.data() + std::size_t(frozen) * k / 32,
                            std::size_t(k) / 32 * sizeof(float)),
                0);
          }
          for (std::size_t i = std::size_t(live) * n; i < actual.size(); ++i)
            ASSERT_EQ(actual[i], -12345.f);
          for (std::size_t i = std::size_t(live) * k; i < p.size(); ++i) {
            ASSERT_EQ(p[i], 90);
            ASSERT_EQ(q[i], 90);
          }
          for (std::size_t i = std::size_t(live) * k / 32; i < x.size(); ++i) {
            ASSERT_EQ(x[i], -8765.f);
            ASSERT_EQ(y[i], -8765.f);
          }
          if (reference == SerialProjectionReference::OriginalColumn) {
            const auto records = projectionSubmissions();
            ASSERT_EQ(records.size(),capturedSubmissions.size());
            if (shortGeometry) {
              EXPECT_EQ(records.front().count,capturedSubmissions.front().count);
              EXPECT_EQ(records.front().value,capturedSubmissions.front().value);
              EXPECT_EQ(records.front().tags,capturedSubmissions.front().tags);
            }
          }
          EXPECT_EQ(stage.graphCaptureVariantSignature(), signature);
          EXPECT_EQ(correction.value->used(), used);
          EXPECT_EQ(s.owners->size(), ownerCount);
          EXPECT_EQ(input.gpu_data_ptr(), inputPointer);
          EXPECT_EQ(output.gpu_data_ptr(), outputPointer);
          EXPECT_EQ(oracle::originalBytes(*weight), raw);
          DeviceNativeVNNIMatrixDesc after{};
          require(kernel->exportNativeVNNIMatrixDesc(after),
                  "Native source disappeared");
          EXPECT_EQ(std::memcmp(&desc, &after, sizeof(desc)), 0);
          ASSERT_FALSE(::testing::Test::HasFailure());
        }
        EXPECT_EQ(readPlanes(s, desc), sourceBefore);
        s.retire();
      })
      .get();
}
/** @brief Canonical21 source entries with vector/cooperative boundaries and
 * separately labeled tiled geometry. */
void sweep(DeviceId device, bool large) {
  for (const auto &format : test::quantizedVerifierFormats())
    for (int m : large ? std::vector<int>{512, 1024, 2048}
                       : std::vector<int>{1, 9, 15, 33, 64, 65}) {
      prove(device, format, m, m == 15 ? 127 : 65, large ? 256 : 512, large);
      ASSERT_FALSE(::testing::Test::HasFailure());
    }
}
/** @brief Minimum native K uses genuine original32 block source formats. */
void block32(DeviceId device) {
  for (const auto &format : test::quantizedVerifierFormats()) {
    if (format.source_is_superblock)
      continue;
    prove(device, format, 65, 65, 32, false);
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
}
/** @brief Every actual K-group specialization plus wide-N and final-column tails.
 * The independent original-column launcher computes each distinct serial row;
 * every tested capture is the actual GEMMStage with original prepared owners.
 * Matching grouped geometry must retain serial bytes across live/empty/reset.
 */
void shortK(DeviceId device) {
  ProjectionSubmissionObservation observation;
  const auto &q8 = test::quantizedVerifierFormat("Q8_0");
  for (int k = 64; k <= 512; k += 32) {
    prove(device, q8, 1, 4*k+1, k, false, false,
          SourcePreparation::NativeGpu, SerialProjectionReference::OriginalColumn);
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
  for (int n : {1279, 1280, 1311, 1312, 1313, 1343, 1344, 1345, 10240}) {
    prove(device, q8, 1, n, 320, false, false,
          SourcePreparation::NativeGpu, SerialProjectionReference::OriginalColumn);
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
  for (int m : {2, 3, 15, 32, 64, 65, 512}) {
    prove(device, q8, m, 1281, 320, false, false,
          SourcePreparation::NativeGpu, SerialProjectionReference::OriginalColumn);
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
  for (const char *name : {"Q8_1", "Q8_K"}) {
    prove(device, test::quantizedVerifierFormat(name), 1, 2049, 512, false, false,
          SourcePreparation::NativeGpu, SerialProjectionReference::OriginalColumn);
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
}
} // namespace
#ifdef HAVE_CUDA
TEST(Q8ResidualAllFormatsDevice, CUDAOrderedShortK) { shortK(DeviceId::cuda(0)); }
TEST(Q8ResidualAllFormatsDevice, CUDA) { sweep(DeviceId::cuda(0), false); }
TEST(Q8ResidualAllFormatsDevice, CUDALargeTiled) {
  sweep(DeviceId::cuda(0), true);
}
TEST(Q8ResidualAllFormatsDevice, CUDAUntiledLarge) {
  for (const auto &f : test::quantizedVerifierFormats()) {
    prove(DeviceId::cuda(0), f, 2048, 5, 256, false);
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
}
TEST(Q8ResidualAllFormatsDevice, CUDAOriginalBlock32) {
  block32(DeviceId::cuda(0));
}
/** @brief Actual CPU-cold Q5_1 promotion remains exact under explicit native TwoTerm. */
TEST(Q8ResidualAllFormatsDevice, CUDAExpandedMinimumPromotion) {
  for (int m : {1, 15, 65}) {
    prove(DeviceId::cuda(0), test::quantizedVerifierFormat("Q5_1"), m, 64, 96,
          false, false, SourcePreparation::PromotedCpuAsymmetric);
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
}
TEST(Q8ResidualAllFormatsDevice, CUDAConcurrentColdIQ) {
  prove(DeviceId::cuda(0), test::quantizedVerifierFormat("IQ2_S"), 3, 5, 256,
        false, true);
}
#endif
#ifdef HAVE_ROCM
TEST(Q8ResidualAllFormatsDevice, ROCmOrderedShortK) { shortK(DeviceId::rocm(0)); }
TEST(Q8ResidualAllFormatsDevice, ROCm) { sweep(DeviceId::rocm(0), false); }
TEST(Q8ResidualAllFormatsDevice, ROCmLargeTiled) {
  sweep(DeviceId::rocm(0), true);
}
TEST(Q8ResidualAllFormatsDevice, ROCmUntiledLarge) {
  for (const auto &f : test::quantizedVerifierFormats()) {
    prove(DeviceId::rocm(0), f, 2048, 5, 256, false);
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
}
TEST(Q8ResidualAllFormatsDevice, ROCmOriginalBlock32) {
  block32(DeviceId::rocm(0));
}
/** @brief Equivalent ROCm coverage for genuine physical23 CPU-cold promotion. */
TEST(Q8ResidualAllFormatsDevice, ROCmExpandedMinimumPromotion) {
  for (int m : {1, 15, 65}) {
    prove(DeviceId::rocm(0), test::quantizedVerifierFormat("Q5_1"), m, 64, 96,
          false, false, SourcePreparation::PromotedCpuAsymmetric);
    ASSERT_FALSE(::testing::Test::HasFailure());
  }
}
TEST(Q8ResidualAllFormatsDevice, ROCmConcurrentColdIQ) {
  prove(DeviceId::rocm(0), test::quantizedVerifierFormat("IQ2_S"), 3, 5, 256,
        false, true);
}
#endif


namespace {
/** @brief Distinct public production adapters sharing the counted native row authority. */
enum class CountedNativeOperation {Plain,Bias,SwiGLU,Fused};
/** @brief Poison and authenticate one actual named activation publication surface. */
struct CountedNativeBank {
 std::string name;size_t elementBytes,rowWidth,logicalElements;bool cudaBlockMajor;bool emptyOnly;
 std::vector<uint8_t> poison,gold;
};
/** @brief Prove exact live output/publication extent through original prepared native adapters.
 * @param device Exact CUDA/HIP participant.
 * @param format Actual original source format; canonical source factories own its blocks.
 * @param m Captured physical rows, including M1 where counted zero is meaningful.
 * @param operation Plain stage, fused SwiGLU stage or shared-input projection bundle.
 * @details Full-active native witnesses authenticate live bytes. This focused
 * regression certifies count/extent preservation, not a second mathematical
 * projection oracle. Twenty replays poison all consumed publication capacities.
 */
void proveCountedNativeExtent(DeviceId device,const test::QuantizedVerifierFormatCase&format,
 int m,CountedNativeOperation operation,const test::QuantizedVerifierFormatCase* secondFormat=nullptr) {
 constexpr int n=65,n2=97,k=256;
 auto*backend=getBackendFor(device);require(backend&&backend->deviceCount()>device.ordinal,"Counted native backend unavailable");
 auto&worker=GPUDeviceContextPool::instance().getContext(device);auto context=IDeviceContext::create(device,1);
 worker.submitAsync([&] {
  SCOPED_TRACE(::testing::Message()<<device.toString()<<" format="<<format.label<<" M="<<m<<" operation="<<int(operation));
  Storage s{device,backend,ExplicitGPUStream(worker.defaultStream()).get()};
  auto source=oracle::source(format,n,k),source2=oracle::source(secondFormat?*secondFormat:format,n2,k);
  auto prepared=test::makeGpuPreparedGemm(source.get(),device,"counted.native.first",ModelContextId{884001});
  test::GpuPreparedGemm prepared2;
  if(operation==CountedNativeOperation::Fused)
   prepared2=test::makeGpuPreparedGemm(source2.get(),device,"counted.native.second",ModelContextId{884002});
  auto*kernel=prepared.kernel;auto*second=operation==CountedNativeOperation::Fused?prepared2.kernel:prepared.kernel;
  require(kernel&&second,"Counted native prepared engines absent");kernel->setGPUStream(s.stream);second->setGPUStream(s.stream);
  FP32Tensor input({size_t(m+1),size_t(k)},DeviceId::cpu()),gate({size_t(m+1),size_t(k)},DeviceId::cpu()),bias({size_t(n)},DeviceId::cpu());
  for(int i=0;i<n;++i)bias.mutable_data()[i]=float(i%11-5)*.03125f;
  std::array<std::unique_ptr<FP32Tensor>,2> output,output2;
  for(int side=0;side<2;++side) {
   output[side]=std::make_unique<FP32Tensor>(std::vector<size_t>{size_t(m+1),size_t(n)},DeviceId::cpu());
   output2[side]=std::make_unique<FP32Tensor>(std::vector<size_t>{size_t(m+1),size_t(n2)},DeviceId::cpu());
  }
  for(size_t i=0;i<input.numel();++i) {
   // Inactive rows remain finite so the original defective producer yields
   // localized bank failures rather than a device trap before observation.
   input.mutable_data()[i]=.125f+std::sin(float(i*7+11))*.31f;
   gate.mutable_data()[i]=-.375f+std::cos(float(i*13+3))*.27f;
  }
  auto*count=s.allocate<int32_t>(1);const auto counted=DeviceRowRange::deviceCounted(m,count);
  auto*consumer=dynamic_cast<IWorkspaceConsumer*>(kernel);auto*secondConsumer=dynamic_cast<IWorkspaceConsumer*>(second);
  require(consumer&&secondConsumer,"Counted native workspace consumers absent");
  auto requirements=consumer->getWorkspaceRequirements(m,n,k);
  if(operation==CountedNativeOperation::SwiGLU)consumer->appendSwiGLUWorkspaceRequirements(requirements,m,k);
  if(operation==CountedNativeOperation::Fused) {
   requirements.merge(secondConsumer->getWorkspaceRequirements(m,n2,k));
   const std::array columns{n,n2};consumer->appendFusedProjectionWorkspaceRequirements(requirements,m,columns,k);
   // Match the actual fused-stage contract, including ordinary M1 side slots.
   addCudaConcurrentDecodeGemvSideStreamWorkspace(requirements,device,m,columns.size());
   require(kernel->prepareFusedProjectionGraphCapture(columns.size()),"Counted fused capture preparation failed");
  }
  std::array<std::unique_ptr<Workspace>,2> workspace;
  std::array<std::unique_ptr<GEMMStage>,2> stage;
  std::unique_ptr<GEMMStage> ordinaryBiasStage;
  for(int side=0;side<2;++side) {
   workspace[side]=std::make_unique<Workspace>(device,requirements);
   if(operation!=CountedNativeOperation::Fused) {
    stage[side]=std::make_unique<GEMMStage>(GEMMStage::Params{
     .device_id=device,.A=&input,.B=source.get(),.C=output[side].get(),.m=m,.n=n,.k=k,
     .bias_tensor=operation==CountedNativeOperation::Bias?&bias:nullptr,
     .gate_input=operation==CountedNativeOperation::SwiGLU?&gate:nullptr,
     .do_swiglu=operation==CountedNativeOperation::SwiGLU,
     .a_buffer_id=BufferId::NORMALIZED,.gate_buffer_id=BufferId::GATE_PROJ,.c_buffer_id=BufferId::ATTN_OUTPUT,
     .force_decode_equivalent_verifier_prefill=true,
     .verifier_row_range=side==0?counted:DeviceRowRange::fullyActive(m),
     .prepared_ref=prepared.ref,.prepared_store=prepared.store.get()});
    stage[side]->setGPUStream(s.stream);stage[side]->bindWorkspace(workspace[side]->value.get());
   }
  }
  /** @brief End borrowed bindings after graph retirement and before manager release. */
  struct Borrows {std::array<std::unique_ptr<GEMMStage>,2>&stage;std::unique_ptr<GEMMStage>&ordinary;
   ~Borrows(){if(ordinary)ordinary->unbindWorkspace();for(auto&s:stage)if(s)s->unbindWorkspace();}} borrows{stage,ordinaryBiasStage};
  /** @brief Retire graph work before releasing borrowed engine workspace state. */
  struct FusedBorrows {
   IWorkspaceConsumer* first;IWorkspaceConsumer* second;bool active;
   ~FusedBorrows(){if(active){first->unbindWorkspace();if(second!=first)second->unbindWorkspace();}}
  } fusedBorrows{consumer,secondConsumer,operation==CountedNativeOperation::Fused};
  // Binding is setup, outside capture. Each graph retains its side's addresses
  // by value; replay does not rebind or mutate the native workspace topology.
  const auto prepareFusedSide=[&](int side) {
   if(operation!=CountedNativeOperation::Fused)return;
   consumer->bindWorkspace(workspace[side]->value.get());
   secondConsumer->bindWorkspace(workspace[side]->value.get());
   kernel->setGPUStream(s.stream);second->setGPUStream(s.stream);
  };
  std::vector<CountedNativeBank> banks;
  const auto addBank=[&](const char*name,size_t bytesPerElement,size_t rowWidth,bool blockMajor=false,bool emptyOnly=false) {
   const auto bytes=workspace[0]->value->getBufferSize(name);if(!bytes)return;
   require(bytes%bytesPerElement==0,"Counted native bank has fractional elements");
   CountedNativeBank bank{name,bytesPerElement,rowWidth,size_t(m)*rowWidth,blockMajor,emptyOnly};
   bank.poison.resize(bytes);
   if(bytesPerElement==1)std::fill(bank.poison.begin(),bank.poison.end(),uint8_t{91});
   else {const uint32_t word=0xc706d720u;for(size_t offset=0;offset<bytes;offset+=4)std::memcpy(bank.poison.data()+offset,&word,4);}
   banks.push_back(std::move(bank));
  };
  addBank(GemmWorkspaceBuffers::QUANT_A,1,k);
  addBank(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE,sizeof(float),k/32);
  // CUDA's established sums ABI is block-major; HIP's is row-major.
  addBank(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE,sizeof(int32_t),k/32,device.is_cuda());
  require(banks.size()>=2,"Counted native primary publication banks absent");
  // Empty work has no consumed partials at all, independent of backend layout.
  // Partial-row KPAR packing is not inferred from allocation capacity.
  for(const auto*name:{GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL,GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED,
      GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS,GemmWorkspaceBuffers::CUDA_CONCURRENT_DECODE_GEMV_KPAR_PARTIALS,
      GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_CANONICAL_KPART_PARTIALS})
   addBank(name,sizeof(float),1,false,true);
  // Original ordinary M1 tensor calls, with actual bias, are an independent
  // live-output witness: neither expected value passes through GEMMStage.
  std::vector<std::unique_ptr<FP32Tensor>> biasInputs,biasOutputs;
  FP32Tensor unbiasedWitness({1u,size_t(n)},DeviceId::cpu()),ordinaryBiasOutput({1u,size_t(n)},DeviceId::cpu());
  if(operation==CountedNativeOperation::Bias)for(int row=0;row<m;++row) {
   auto in=std::make_unique<FP32Tensor>(std::vector<size_t>{1u,size_t(k)},DeviceId::cpu());
   auto out=std::make_unique<FP32Tensor>(std::vector<size_t>{1u,size_t(n)},DeviceId::cpu());
   std::copy_n(input.data()+size_t(row)*k,k,in->mutable_data());
   biasInputs.push_back(std::move(in));biasOutputs.push_back(std::move(out));
  }
  std::unique_ptr<IGPUGraphCapture> fullGraph,ordinaryM1Graph,biasOracleGraph;
  CapturedWorkRetirement retirement(s);
  for(auto*t:std::vector<TensorBase*>{&input,&gate,&bias,output[0].get(),output[1].get(),output2[0].get(),output2[1].get()}) {
   TransferEngine::prepareDeviceInput(t,device,s.stream);TransferEngine::requireDeviceInput(t,device,s.stream);
  }
  if(operation==CountedNativeOperation::Bias) {
   for(int row=0;row<m;++row)for(auto*t:std::array<TensorBase*,2>{biasInputs[row].get(),biasOutputs[row].get()}) {
    TransferEngine::prepareDeviceInput(t,device,s.stream);TransferEngine::requireDeviceInput(t,device,s.stream);
   }
   for(auto*t:std::array<TensorBase*,2>{&unbiasedWitness,&ordinaryBiasOutput}) {
    TransferEngine::prepareDeviceInput(t,device,s.stream);TransferEngine::requireDeviceInput(t,device,s.stream);
   }
   auto ordinaryParams=stage[1]->getParams();ordinaryParams.A=biasInputs[0].get();ordinaryParams.C=&ordinaryBiasOutput;
   ordinaryParams.m=1;ordinaryParams.force_decode_equivalent_verifier_prefill=false;ordinaryParams.verifier_row_range.reset();
   ordinaryBiasStage=std::make_unique<GEMMStage>(ordinaryParams);
   ordinaryBiasStage->setGPUStream(s.stream);ordinaryBiasStage->bindWorkspace(workspace[1]->value.get());
   // Workspace binding clears adapter stream state; direct oracle calls borrow
   // the exact fixture stream after binding, just as a production stage does.
   kernel->setGPUStream(s.stream);
   biasOracleGraph=worker.createGraphCapture(s.stream);
   {ScopedBackendGraphCapture recording(worker,*biasOracleGraph,"Independent ordinary biased M1 witnesses");
    require(recording.begin(),"Bias witness capture failed");
    for(int row=0;row<m;++row)
     require(kernel->multiply_tensor(biasInputs[row].get(),biasOutputs[row].get(),1,n,k,false,1.f,0.f,&bias,
        nullptr,device.ordinal,workspace[1]->value.get()),"Ordinary biased row witness rejected");
    require(kernel->multiply_tensor(biasInputs[0].get(),&unbiasedWitness,1,n,k,false,1.f,0.f,nullptr,
       nullptr,device.ordinal,workspace[1]->value.get()),"Ordinary unbiased sensitivity witness rejected");
    require(ordinaryBiasStage->execute(context.get()),"Ordinary biased GEMMStage witness rejected");
    recording.finish();
   }
   require(biasOracleGraph->instantiate(),"Bias witness instantiate failed");
  }
  s.upload(count,std::vector<int32_t>{m});
  const auto submit=[&](int side) {
   if(operation==CountedNativeOperation::Fused) {
    const auto rows=side==0?counted:DeviceRowRange::fullyActive(m);
    auto scope=kernel->beginVerifierDecodeEquivalentScope(rows);
    const std::vector<ITensorGemm::TensorProjectionDesc>projections{
     {kernel,output[side].get(),n,nullptr,"counted.first"},{second,output2[side].get(),n2,nullptr,"counted.second"}};
    require(kernel->multiply_fused_verifier_rows_decode_equivalent(&input,projections,m,k,nullptr,workspace[side]->value.get()),"Counted fused native adapter rejected");
   } else require(stage[side]->execute(context.get()),"Counted native stage adapter rejected");
  };
  std::array<std::vector<GPUGraphKernelNodeInfo>,2> inventory;
  const auto capture=[&](std::unique_ptr<IGPUGraphCapture>&graph,int side) {
   prepareFusedSide(side);
   graph=worker.createGraphCapture(s.stream);ScopedBackendGraphCapture recording(worker,*graph,"Counted native projection extent");
   require(recording.begin(),"Counted native capture begin failed");submit(side);recording.finish();
   require(graph->instantiate(),"Counted native instantiate failed");std::string error;
   require(graph->inspectKernelNodes(inventory[side],&error),"Counted native kernel inspection failed");
   ASSERT_FALSE(inventory[side].empty());EXPECT_EQ(graph->nodeCount(),inventory[side].size());
   for(const auto&node:inventory[side]) {
    ASSERT_TRUE(node.name_resolved);EXPECT_EQ(node.local_memory_bytes_per_thread,0u);
    std::cout<<"COUNTED_NATIVE_KERNEL,"<<device.toString()<<','<<format.label<<','<<m<<','<<int(operation)<<','<<side<<','
     <<node.name<<','<<node.grid_x<<','<<node.grid_y<<','<<node.grid_z<<','<<node.block_x<<','<<node.block_y<<','<<node.block_z<<'\n';
   }
  };
  capture(s.graph,0);capture(fullGraph,1);
  if(secondFormat&&device.is_rocm()) {
   // A mixed prepared bundle must reach the mixed raw bridge, rather than two
   // separately accepted single-format calls hiding its M1 admission boundary.
   EXPECT_TRUE(std::any_of(inventory[0].begin(),inventory[0].end(),[](const auto&node){
    return node.name.find("gemv_native_vnni_small_m_batched_shared_mixed_kernel_t")!=std::string::npos;
   }))<<"Counted M1 mixed bundle did not exercise the mixed producer";
  }
  const auto quantizers=std::count_if(inventory[0].begin(),inventory[0].end(),[](const auto&node){
   return node.name.find("quantize")!=std::string::npos;
  });
  bool ordinaryM1Same=true;
  if(m==1) {
   prepareFusedSide(1);
   ordinaryM1Graph=worker.createGraphCapture(s.stream);
   {ScopedBackendGraphCapture recording(worker,*ordinaryM1Graph,"Literal ordinary M1 inventory");
    require(recording.begin(),"Ordinary M1 capture failed");
    if(operation==CountedNativeOperation::Fused) {
     const std::vector<ITensorGemm::TensorProjectionDesc>projections{
      {kernel,output[1].get(),n,nullptr,"ordinary.first"},{second,output2[1].get(),n2,nullptr,"ordinary.second"}};
     require(kernel->multiply_fused_tensor(&input,projections,1,k,nullptr,workspace[1]->value.get()),"Ordinary M1 fused rejected");
    } else if(operation==CountedNativeOperation::Bias) {
     // The verifier stage submits one fused projection. Its ordinary peer has
     // the same operation boundary: HIP's multiply_tensor uses a different
     // established bias epilogue. Independent multiply_tensor output bytes are
     // checked by biasOracleGraph above, without conflating the two inventories.
     const std::vector<ITensorGemm::TensorProjectionDesc>projections{
      {kernel,output[1].get(),n,&bias,"ordinary.biased"}};
     require(kernel->multiply_fused_tensor(&input,projections,1,k,nullptr,workspace[1]->value.get()),"Ordinary M1 biased fused projection rejected");
    } else if(operation==CountedNativeOperation::SwiGLU)
     require(kernel->multiply_tensor_with_fused_swiglu(&gate,&input,output[1].get(),1,n,k,1.f,0.f,workspace[1]->value.get()),"Ordinary M1 SwiGLU rejected");
    else require(kernel->multiply_tensor(&input,output[1].get(),1,n,k,false,1.f,0.f,
       operation==CountedNativeOperation::Bias?&bias:nullptr,nullptr,device.ordinal,workspace[1]->value.get()),"Ordinary M1 projection rejected");
    recording.finish();
   }
   require(ordinaryM1Graph->instantiate(),"Ordinary M1 instantiate failed");
   std::vector<GPUGraphKernelNodeInfo>ordinary;std::string error;
   require(ordinaryM1Graph->inspectKernelNodes(ordinary,&error),"Ordinary M1 inventory failed");
   EXPECT_EQ(ordinaryM1Graph->nodeCount(),ordinary.size());
   using Key=std::tuple<uintptr_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t>;
   std::map<Key,int>literal,explicitFull;
   const auto add=[](auto&counts,const auto&nodes){for(const auto&node:nodes)
    ++counts[Key{node.function_identity,node.grid_x,node.grid_y,node.grid_z,node.block_x,node.block_y,node.block_z}];};
   add(literal,ordinary);add(explicitFull,inventory[1]);
   ordinaryM1Same=literal==explicitFull;
  }
  constexpr float poison=-34567.125f;
  const auto dirtyOutputs=[&](int side) {
   s.upload(static_cast<float*>(output[side]->gpu_data_ptr()),std::vector<float>(output[side]->numel(),poison));
   s.upload(static_cast<float*>(output2[side]->gpu_data_ptr()),std::vector<float>(output2[side]->numel(),poison));
  };
  dirtyOutputs(1);require(fullGraph->launch(),"Counted native full witness replay failed");
  auto full=s.download(static_cast<const float*>(output[1]->gpu_data_ptr()),output[1]->numel());
  if(operation==CountedNativeOperation::Bias) {
   require(biasOracleGraph->launch(),"Ordinary biased witness replay failed");
   std::vector<float> independent(full.size(),poison);
   for(int row=0;row<m;++row) {
    auto expected=s.download(static_cast<const float*>(biasOutputs[row]->gpu_data_ptr()),n);
    std::copy(expected.begin(),expected.end(),independent.begin()+size_t(row)*n);
   }
   auto noBias=s.download(static_cast<const float*>(unbiasedWitness.gpu_data_ptr()),n);
   ASSERT_NE(std::memcmp(independent.data(),noBias.data(),size_t(n)*sizeof(float)),0)
    <<"Actual typed bias must change the ordinary native witness";
   auto ordinary=s.download(static_cast<const float*>(ordinaryBiasOutput.gpu_data_ptr()),n);
   EXPECT_EQ(std::memcmp(ordinary.data(),independent.data(),size_t(n)*sizeof(float)),0)
    <<"Ordinary M1 GEMMStage discarded or changed actual bias";
   ASSERT_EQ(std::memcmp(full.data(),independent.data(),full.size()*sizeof(float)),0)
    <<"Full verifier GEMMStage discarded or changed actual bias";
   full=std::move(independent);
  }
  const auto full2=s.download(static_cast<const float*>(output2[1]->gpu_data_ptr()),output2[1]->numel());
  // Full counted publication is the extent witness, not a second quantizer.
  // Its live projection bytes must first match the independent uncounted call.
  dirtyOutputs(0);for(auto&bank:banks)s.upload(static_cast<uint8_t*>(workspace[0]->value->getBuffer(bank.name)),bank.poison);
  require(s.graph->launch(),"Counted native full-count replay failed");
  auto first=s.download(static_cast<const float*>(output[0]->gpu_data_ptr()),output[0]->numel());
  ASSERT_EQ(std::memcmp(first.data(),full.data(),full.size()*sizeof(float)),0);
  if(operation==CountedNativeOperation::Fused) {
   auto secondOutput=s.download(static_cast<const float*>(output2[0]->gpu_data_ptr()),output2[0]->numel());
   ASSERT_EQ(std::memcmp(secondOutput.data(),full2.data(),full2.size()*sizeof(float)),0);
  }
  for(auto&bank:banks)bank.gold=s.download(static_cast<const uint8_t*>(workspace[0]->value->getBuffer(bank.name)),bank.poison.size());
  for(int replay=0;replay<20;++replay) {
   const int live=replay%4==0?m:replay%4==1?0:replay%4==2?std::min(m,1):std::max(0,m-1);
   SCOPED_TRACE(::testing::Message()<<"replay="<<replay<<" live="<<live);
   dirtyOutputs(0);for(auto&bank:banks)s.upload(static_cast<uint8_t*>(workspace[0]->value->getBuffer(bank.name)),bank.poison);
   s.upload(count,std::vector<int32_t>{live});require(s.graph->launch(),"Counted native replay failed");
   const auto checkOutput=[&](const char*name,FP32Tensor&value,const std::vector<float>&gold,int width) {
    auto got=s.download(static_cast<const float*>(value.gpu_data_ptr()),value.numel());
    for(size_t i=0;i<got.size();++i) {
     const float expected=i<size_t(live)*width?gold[i]:poison;
     uint32_t actualBits,expectedBits;std::memcpy(&actualBits,&got[i],4);std::memcpy(&expectedBits,&expected,4);
     ASSERT_EQ(actualBits,expectedBits)<<name<<" word="<<i<<" row="<<i/width;
    }
   };
   checkOutput("first",*output[0],full,n);
   if(operation==CountedNativeOperation::Fused)checkOutput("second",*output2[0],full2,n2);
   for(const auto&bank:banks) {
    if(bank.emptyOnly&&live>0)continue;
    auto got=s.download(static_cast<const uint8_t*>(workspace[0]->value->getBuffer(bank.name)),bank.poison.size());
    for(size_t element=0;element<got.size()/bank.elementBytes;++element) {
     const size_t row=bank.cudaBlockMajor?element%size_t(m):element/bank.rowWidth;
     const bool active=!bank.emptyOnly&&element<bank.logicalElements&&row<size_t(live);
     const auto&gold=active?bank.gold:bank.poison;const auto offset=element*bank.elementBytes;
     ASSERT_EQ(std::memcmp(got.data()+offset,gold.data()+offset,bank.elementBytes),0)
      <<bank.name<<" element="<<element<<" row="<<row<<" active="<<active;
    }
   }
   ASSERT_FALSE(::testing::Test::HasFailure());
  }
  EXPECT_EQ(quantizers,1)<<"Counted adapters must capture one shared row-aware publication";
  EXPECT_TRUE(ordinaryM1Same)<<"Null-count/full-active M1 must retain the literal ordinary native route";
  for(const auto&bank:banks)if(bank.name==GemmWorkspaceBuffers::QUANT_A||bank.name==GemmWorkspaceBuffers::SCALES_A_BLOCKWISE)
   EXPECT_NE(bank.gold,bank.poison)<<bank.name<<" full counted publication was not exercised";
  s.retire();fullGraph.reset();ordinaryM1Graph.reset();biasOracleGraph.reset();
 }).get();
}
/** @brief Exercise original source totality with small, non-tile-aligned output widths. */
void proveCountedNativeAllFormats(DeviceId device,CountedNativeOperation operation) {
 for(const auto&format:test::quantizedVerifierFormats())for(int m:{1,3,9,17}) {
   proveCountedNativeExtent(device,format,m,operation);
   if(::testing::Test::HasFailure())return;
 }
}
}

#ifdef HAVE_CUDA
/** @test Counted M1/grouped CUDA Plain leaves inactive output/publication bytes untouched. */
TEST(Q8ResidualAllFormatsDevice,CUDACountedNativePlain){proveCountedNativeAllFormats(DeviceId::cuda(0),CountedNativeOperation::Plain);}
/** @test Counted M1/grouped CUDA Bias leaves inactive output/publication bytes untouched. */
TEST(Q8ResidualAllFormatsDevice,CUDACountedNativeBias){proveCountedNativeAllFormats(DeviceId::cuda(0),CountedNativeOperation::Bias);}
/** @test Counted M1/grouped CUDA SwiGLU leaves inactive output/publication bytes untouched. */
TEST(Q8ResidualAllFormatsDevice,CUDACountedNativeSwiGLU){proveCountedNativeAllFormats(DeviceId::cuda(0),CountedNativeOperation::SwiGLU);}
/** @test Counted M1/grouped CUDA Fused leaves inactive output/publication bytes untouched. */
TEST(Q8ResidualAllFormatsDevice,CUDACountedNativeFused){proveCountedNativeAllFormats(DeviceId::cuda(0),CountedNativeOperation::Fused);}
/** @test Distinct source formats reach the count-owned M1 shared projection bundle. */
TEST(Q8ResidualAllFormatsDevice,CUDACountedNativeMixedFusedM1) {
 const auto&first=test::quantizedVerifierFormat("Q4_0");
 const auto&second=test::quantizedVerifierFormat("IQ4_NL");
 proveCountedNativeExtent(DeviceId::cuda(0),first,1,CountedNativeOperation::Fused,&second);
}
#endif

#ifdef HAVE_ROCM
/** @test Counted M1/grouped ROCm Plain leaves inactive output/publication bytes untouched. */
TEST(Q8ResidualAllFormatsDevice,ROCmCountedNativePlain){proveCountedNativeAllFormats(DeviceId::rocm(0),CountedNativeOperation::Plain);}
/** @test Counted M1/grouped ROCm Bias leaves inactive output/publication bytes untouched. */
TEST(Q8ResidualAllFormatsDevice,ROCmCountedNativeBias){proveCountedNativeAllFormats(DeviceId::rocm(0),CountedNativeOperation::Bias);}
/** @test Grouped bias publication preserves inactive rows independently of M1 admission. */
TEST(Q8ResidualAllFormatsDevice,ROCmCountedBiasEpilogue) {
 for(const auto&format:test::quantizedVerifierFormats()) {
  proveCountedNativeExtent(DeviceId::rocm(0),format,3,CountedNativeOperation::Bias);
  if(::testing::Test::HasFailure())return;
 }
}
/** @test Counted M1/grouped ROCm SwiGLU leaves inactive output/publication bytes untouched. */
TEST(Q8ResidualAllFormatsDevice,ROCmCountedNativeSwiGLU){proveCountedNativeAllFormats(DeviceId::rocm(0),CountedNativeOperation::SwiGLU);}
/** @test Counted M1/grouped ROCm Fused leaves inactive output/publication bytes untouched. */
TEST(Q8ResidualAllFormatsDevice,ROCmCountedNativeFused){proveCountedNativeAllFormats(DeviceId::rocm(0),CountedNativeOperation::Fused);}
/** @test Distinct source formats reach the count-owned M1 shared projection bundle. */
TEST(Q8ResidualAllFormatsDevice,ROCmCountedNativeMixedFusedM1) {
 const auto&first=test::quantizedVerifierFormat("Q4_0");
 const auto&second=test::quantizedVerifierFormat("IQ4_NL");
 proveCountedNativeExtent(DeviceId::rocm(0),first,1,CountedNativeOperation::Fused,&second);
}
#endif

#include "CountedActivationProducerRegression.h"
