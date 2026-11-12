/**
 * @file Perf__CPUMoESingleTokenTransaction.cpp
 * @brief Diagnostic M=1 CPU MoE FFN economy gate: production single-token
 *        sequence versus the grouped FFN transaction at a one-row route.
 *
 * Qwen3.6-35B-A3B CPU decode executes every MoE layer's routed expert work as
 * the production single-token sequence: the pooled router-Q8-reused gate/up
 * bundle, the production serial SwiGLU, and the pooled multi-input expert
 * down projection, followed by the production weighted route publication.
 * The grouped path executes the same arithmetic as one retained-team
 * transaction (gate/up workshare, exact SwiGLU-to-Q8_1 publication, down
 * workshare) for grouped and verifier routes. This harness asks one
 * diagnostic question at the M=1 shape: does routing the single-token
 * sequence through the existing transaction primitive regress, tie, or win
 * on the observed placements — with byte-identical outputs.
 *
 * Both arms are the actual production helpers. Arm A calls the production
 * router Q8 publication, the production fused router-Q8 bundle, the
 * production serial SwiGLU primitive, the production fused multi-input down
 * launcher (with its internal per-expert Q8 quantization), and the
 * production route publication. Arm B submits the same descriptors through
 * `execute_moe_grouped_ffn_transaction_preq_decode_equivalent` with the
 * actual shared router-Q8 row pointer; it never re-quantizes or copies the
 * router row into per-expert inputs. There is no copied SwiGLU/Q8 oracle and
 * no forced ineligible path.
 *
 * Hard byte gates, evaluated before any timing: gate/up projections, the
 * SwiGLU Q8_1 publication (production activation quantization applied to
 * arm A's production SwiGLU rows versus arm B's transaction-published bank),
 * down projections, and the published route contributions must be
 * byte-identical across arms. Guard tails prove neither arm writes outside
 * its live extent; the byte gates against the proven-live arm B banks prove
 * arm A's full live extent. Sealed inputs are compared after every timed
 * phase. A separate unprofiled proof pass verifies the intended production
 * counters advanced for both arms. If the transaction rejects the actual
 * shared-router-Q8 M=1 shape, the harness reports the admission gap and
 * fails; it never falls back to a reshaped or production-modified variant.
 *
 * Timing is unprofiled: warmup pairs, then alternating paired A/B samples
 * with the within-pair order alternating to cancel order effects. The
 * collection policy is suppressed around clocks and restored for the proof
 * pass. A regression or a no-signal result closes the diagnostic with the
 * recorded numbers; `B <= A within noise` is a non-regression, not an
 * optimization win. Thread/OMP/policy context is whatever the invoking
 * environment provides and is identical for both arms; the canonical
 * evidence runs the fixture once per observed physical placement.
 *
 * Geometry: d_model=2048, 256 experts, top-8, intermediate=512 (16 gate/up
 * members of 512x2048 sharing one router Q8 row; 8 down members of
 * 2048x512). Default formats pair IQ2_S gate/up with IQ4_XS down, matching
 * the real UD artifact's per-tensor pairing; the canonical registry supplies
 * every format via environment. Engines are position-mapped onto the actual
 * route selection: the router runs the real 256-expert top-8 selection and
 * publishes the real Q8 row, and the eight prepared engines serve the eight
 * selected slots in route order. Shapes, arithmetic, and paths are
 * production-identical; expert identity carries no semantics here.
 */

#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "kernels/cpu/CPUInvocationWorkspace.h"
#include "kernels/cpu/gemm/CPUProjectionQ8Publication.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedGemmKernel.h"
#include "kernels/cpu/moe/CPUMoEKernel.h"
#include "kernels/cpu/primitives/SwiGLUPrimitives.h"
#include "kernels/cpu/primitives/VectorPrimitives.h"
#include "tensors/AlignedVector.h"
#include "tensors/TensorClasses.h"
#include "tensors/TensorKernels.h"
#include "utils/CPUFeatures.h"
#include "utils/CPURouterKernelTestAdmission.h"
#include "utils/CPUPerfStatsTimingCollectionScope.h"
#include "utils/PerfStatsCollector.h"
#include "utils/QuantizedVerifierFormats.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

using namespace llaminar2;
using llaminar2::cpu::native_quantised::CPUNativeQuantisedGemmKernel;

namespace
{
    constexpr int kDModel = 2048;
    constexpr int kIntermediate = 512;
    constexpr int kNumExperts = 256;
    constexpr int kTopK = 8;
    constexpr int kHiddenBlocks = kDModel / Q8_1Block::BLOCK_SIZE;
    constexpr int kActivationBlocks = kIntermediate / Q8_1Block::BLOCK_SIZE;
    /// Guard tail behind every live bank extent, in bytes.
    constexpr size_t kGuardTailBytes = 64;
    /// Canary byte pattern for guard tails and sealed-input comparison.
    constexpr uint8_t kGuardByte = 0xA5;

    /** @brief Guard tail extent for one element type, in elements. */
    template <typename T>
    constexpr size_t guardElements()
    {
        return (kGuardTailBytes + sizeof(T) - 1) / sizeof(T);
    }

    int envPositiveInt(const char *name, int fallback)
    {
        const char *raw = std::getenv(name);
        if (!raw || !*raw)
            return fallback;
        char *end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end == raw || *end != '\0' || parsed <= 0 ||
            parsed > std::numeric_limits<int>::max())
        {
            throw std::runtime_error(
                std::string(name) + " must be a positive integer");
        }
        return static_cast<int>(parsed);
    }

    /** @brief One prepared expert triplet with its packed native engines. */
    struct PreparedExpert
    {
        int rows = 0; ///< Always one for the single-token route.
        std::unique_ptr<CPUNativeQuantisedGemmKernel> gate;
        std::unique_ptr<CPUNativeQuantisedGemmKernel> up;
        std::unique_ptr<CPUNativeQuantisedGemmKernel> down;
    };

    /**
     * @brief RAII scope that enables one PerfStats domain for a proof pass.
     *
     * Mirrors the existing exact-timing suppression scope in reverse: the
     * caller's filter is retained and restored; the scope exists only around
     * unprofiled proof calls, never around clocks.
     */
    class KernelDomainProofScope final
    {
    public:
        KernelDomainProofScope()
        {
            if (const char *value = std::getenv("LLAMINAR_PERF_STATS_FILTER"))
                prior_filter_ = value;
            if (setenv("LLAMINAR_PERF_STATS_FILTER", "kernel", 1) != 0)
                throw std::runtime_error("Cannot enable the kernel proof domain");
            PerfStatsCollector::reloadConfigurationFromEnvironment();
        }
        ~KernelDomainProofScope() noexcept
        {
            const int status = prior_filter_
                ? setenv("LLAMINAR_PERF_STATS_FILTER", prior_filter_->c_str(), 1)
                : unsetenv("LLAMINAR_PERF_STATS_FILTER");
            if (status != 0) std::terminate();
            PerfStatsCollector::reloadConfigurationFromEnvironment();
        }
        KernelDomainProofScope(const KernelDomainProofScope &) = delete;
        KernelDomainProofScope &operator=(const KernelDomainProofScope &) = delete;

    private:
        std::optional<std::string> prior_filter_; ///< Caller filter, restored on exit.
    };

    /** @brief Fill the guard tail behind a live bank extent. */
    template <typename T>
    void fillGuardTail(AlignedVector<T> &bank, size_t live_elements)
    {
        auto *bytes = reinterpret_cast<uint8_t *>(bank.data());
        const size_t live_bytes = live_elements * sizeof(T);
        const size_t capacity_bytes = bank.size() * sizeof(T);
        if (capacity_bytes < live_bytes + kGuardTailBytes)
            throw std::logic_error("guard tail does not fit the bank capacity");
        std::memset(bytes + live_bytes, kGuardByte, kGuardTailBytes);
    }

    /** @brief Require an intact guard tail; any overlap is a hard failure. */
    template <typename T>
    void expectGuardTailIntact(
        const AlignedVector<T> &bank, size_t live_elements, const std::string &label)
    {
        const auto *bytes = reinterpret_cast<const uint8_t *>(bank.data());
        const size_t live_bytes = live_elements * sizeof(T);
        for (size_t offset = live_bytes; offset < live_bytes + kGuardTailBytes; ++offset)
        {
            if (bytes[offset] != kGuardByte)
            {
                ADD_FAILURE()
                    << label << " wrote outside its live extent at tail byte "
                    << (offset - live_bytes);
                return;
            }
        }
    }

    /** @brief Require byte equality and report the first divergent byte. */
    template <typename T>
    void expectByteEqual(
        const T *actual, const T *expected, size_t elements,
        const std::string &label)
    {
        const size_t bytes = elements * sizeof(T);
        if (std::memcmp(actual, expected, bytes) == 0)
            return;
        const auto *a = reinterpret_cast<const uint8_t *>(actual);
        const auto *e = reinterpret_cast<const uint8_t *>(expected);
        size_t first = 0;
        while (first < bytes && a[first] == e[first])
            ++first;
        ADD_FAILURE()
            << label << " differs at byte " << first
            << " actual=" << static_cast<unsigned>(a[first])
            << " expected=" << static_cast<unsigned>(e[first]);
    }

    /** @brief Deterministic hidden rows with non-trivial quantization tails. */
    AlignedVector<float> makeHiddenRows(int rows, int d_model)
    {
        AlignedVector<float> hidden(static_cast<size_t>(rows) *
                                    static_cast<size_t>(d_model));
        for (size_t index = 0; index < hidden.size(); ++index)
        {
            const int centered =
                static_cast<int>((index * 37u + index / 13u) % 257u) - 128;
            hidden[index] =
                static_cast<float>(centered) * 0.004125f +
                static_cast<float>(static_cast<int>(index % 11u) - 5) *
                    0.000071f;
        }
        return hidden;
    }

    /** @brief Deterministic production-sized router gate matrix. */
    AlignedVector<float> makeRouterGate()
    {
        AlignedVector<float> gate(
            static_cast<size_t>(kNumExperts) * static_cast<size_t>(kDModel));
        for (size_t index = 0; index < gate.size(); ++index)
        {
            const int centered =
                static_cast<int>((index * 29u + index / 37u) % 193u) - 96;
            gate[index] =
                static_cast<float>(centered) * 0.00042f +
                static_cast<float>(static_cast<int>(index % 5u) - 2) *
                    0.000013f;
        }
        return gate;
    }

    /** @brief Resolve the gate/up source format: default production pairing. */
    const test::QuantizedVerifierFormatCase &selectedGateUpFormat()
    {
        const char *raw = std::getenv("LLAMINAR_CPU_MOE_STT_GATE_UP_FORMAT");
        if (!raw || !*raw)
            raw = "IQ2_S";
        for (const auto &format : test::quantizedMoEVerifierFormats())
            if (format.label == raw)
                return format;
        throw std::runtime_error(
            std::string("LLAMINAR_CPU_MOE_STT_GATE_UP_FORMAT selected no format: ") +
            raw);
    }

    /** @brief Resolve the down source format: default production pairing. */
    const test::QuantizedVerifierFormatCase &selectedDownFormat()
    {
        const char *raw = std::getenv("LLAMINAR_CPU_MOE_STT_DOWN_FORMAT");
        if (!raw || !*raw)
            raw = "IQ4_XS";
        for (const auto &format : test::quantizedMoEVerifierFormats())
            if (format.label == raw)
                return format;
        throw std::runtime_error(
            std::string("LLAMINAR_CPU_MOE_STT_DOWN_FORMAT selected no format: ") +
            raw);
    }

    /** @brief Resolve the numerical policy; production MoE experts are aligned. */
    CPUProjectionNumericalPolicy selectedPolicy()
    {
        const char *raw = std::getenv("LLAMINAR_CPU_MOE_STT_POLICY");
        if (!raw || !*raw)
            return CPUProjectionNumericalPolicy::GPUAlignedExpert;
        if (std::string(raw) == "aligned")
            return CPUProjectionNumericalPolicy::GPUAlignedExpert;
        if (std::string(raw) == "native")
            return CPUProjectionNumericalPolicy::BackendNative;
        throw std::runtime_error(
            "LLAMINAR_CPU_MOE_STT_POLICY must be aligned or native");
    }

    /** @brief Median of a native-double sample list (upper median). */
    double medianUs(std::vector<double> samples)
    {
        if (samples.empty())
            throw std::logic_error("median requested without samples");
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2u];
    }

    /** @brief Arithmetic mean of a native-double sample list. */
    double meanUs(const std::vector<double> &samples)
    {
        if (samples.empty())
            throw std::logic_error("mean requested without samples");
        const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
        return sum / static_cast<double>(samples.size());
    }

    /** @brief Sample standard deviation of a native-double sample list. */
    double deviationUs(const std::vector<double> &samples)
    {
        if (samples.size() < 2)
            return 0.0;
        const double mean = meanUs(samples);
        double sum = 0.0;
        for (const double sample : samples)
            sum += (sample - mean) * (sample - mean);
        return std::sqrt(sum / static_cast<double>(samples.size() - 1));
    }

    /**
     * @brief Complete fixture state prepared and sealed before any timing.
     *
     * Every buffer is allocated once with its guard tail, every input is
     * sealed against later mutation, and both arms hold disjoint output
     * banks so neither arm can observe the other's bytes.
     */
    struct SingleTokenFixture
    {
        CPUProjectionNumericalPolicy policy =
            CPUProjectionNumericalPolicy::GPUAlignedExpert;
        std::vector<PreparedExpert> experts;
        CPUMoEKernel router;
        AlignedVector<float> hidden;       ///< One hidden row [d_model].
        AlignedVector<float> router_gate;  ///< Router gate [256, d_model].
        std::vector<float> sealed_hidden;
        std::vector<float> sealed_router_gate;
        std::vector<int> route_slots;      ///< Actual selected expert slots.
        std::vector<float> route_weights;  ///< Actual normalized top-k weights.
        const Q8_1Block *published_q8 = nullptr; ///< Production router Q8 row.
        std::vector<Q8_1Block> sealed_q8;        ///< Sealed publication copy.

        // Arm A banks (production single-token shapes; per-expert tensors).
        std::vector<std::unique_ptr<FP32Tensor>> a_gate;
        std::vector<std::unique_ptr<FP32Tensor>> a_up;
        std::vector<std::unique_ptr<FP32Tensor>> a_activated;
        std::vector<std::unique_ptr<FP32Tensor>> a_down;
        AlignedVector<float> a_route_output; ///< [d_model] + guard.

        // Arm B banks (grouped expert-major layout) with guard tails.
        AlignedVector<float> b_gate;         ///< [8, intermediate] + guard.
        AlignedVector<float> b_up;           ///< [8, intermediate] + guard.
        AlignedVector<Q8_1Block> b_activation; ///< [8, activation_blocks] + guard.
        AlignedVector<float> b_down;         ///< [8, d_model] + guard.
        AlignedVector<float> b_route_output; ///< [d_model] + guard.

        // Production activation quantization of arm A's SwiGLU rows: the
        // production helper applied to production SwiGLU outputs, used to
        // byte-gate the SwiGLU-Q8 contract against arm B's published bank.
        AlignedVector<Q8_1Block> a_q8_proxy; ///< [8, activation_blocks] + guard.

        DeviceWorkspaceManager workspace{DeviceId::cpu(), 1u << 20};
        std::vector<float> partial_storage;  ///< Admitted partial bank.

        size_t liveGate = 0;   ///< Live floats in one gate/up bank.
        size_t liveDown = 0;   ///< Live floats in the down bank.
        size_t liveRoute = 0;  ///< Live floats in one route output.
    };

    /** @brief Prepare the complete fixture; no timing has begun. */
    void prepareFixture(SingleTokenFixture &fixture)
    {
        fixture.policy = selectedPolicy();

        // Production router: real 256-expert gate, real top-8 selection, real
        // Q8 publication, under the admitted publication root.
        fixture.hidden = makeHiddenRows(1, kDModel);
        fixture.router_gate = makeRouterGate();
        llaminar2::test::admitCPURouterKernelForTest(fixture.router, 1, kDModel);
        MoERoutingResult route;
        if (!fixture.router.route(
                fixture.hidden.data(), fixture.router_gate.data(), 1, kDModel,
                kNumExperts, kTopK, true, route))
        {
            throw std::runtime_error("production router rejected valid geometry");
        }
        fixture.published_q8 = fixture.router.publishedRouterQ8Hidden(
            fixture.hidden.data(), 1, kDModel);
        if (!fixture.published_q8)
            throw std::runtime_error("production router did not publish its Q8 row");
        fixture.route_slots = route.expert_indices;
        fixture.route_weights = route.expert_weights;
        if (static_cast<int>(fixture.route_slots.size()) != kTopK ||
            static_cast<int>(fixture.route_weights.size()) != kTopK)
        {
            throw std::runtime_error("production router did not select top-8");
        }
        fixture.sealed_hidden.assign(fixture.hidden.begin(), fixture.hidden.end());
        fixture.sealed_router_gate.assign(
            fixture.router_gate.begin(), fixture.router_gate.end());
        fixture.sealed_q8.assign(
            fixture.published_q8,
            fixture.published_q8 + static_cast<size_t>(kHiddenBlocks));

        // Eight prepared engines, position-mapped onto the route slots.
        const auto &gate_up_format = selectedGateUpFormat();
        const auto &down_format = selectedDownFormat();
        fixture.experts.reserve(kTopK);
        for (int slot = 0; slot < kTopK; ++slot)
        {
            PreparedExpert expert;
            expert.rows = 1;
            const uint32_t seed = static_cast<uint32_t>(
                41000u + static_cast<unsigned>(slot) * 101u +
                gate_up_format.source_codebook_id);
            const auto gate_weights = gate_up_format.create(
                {static_cast<size_t>(kIntermediate),
                 static_cast<size_t>(kDModel)},
                seed + 1u);
            const auto up_weights = gate_up_format.create(
                {static_cast<size_t>(kIntermediate),
                 static_cast<size_t>(kDModel)},
                seed + 2u);
            const auto down_weights = down_format.create(
                {static_cast<size_t>(kDModel),
                 static_cast<size_t>(kIntermediate)},
                seed + 3u);
            if (!gate_weights || !up_weights || !down_weights)
            {
                throw std::runtime_error(
                    std::string("could not create CPU MoE weights for ") +
                    gate_up_format.label + "/" + down_format.label);
            }
            expert.gate = std::make_unique<CPUNativeQuantisedGemmKernel>(
                gate_weights.get(), 0, -1, fixture.policy);
            expert.up = std::make_unique<CPUNativeQuantisedGemmKernel>(
                up_weights.get(), 0, -1, fixture.policy);
            expert.down = std::make_unique<CPUNativeQuantisedGemmKernel>(
                down_weights.get(), 0, -1, fixture.policy);
            if (!expert.gate->isValid() || !expert.up->isValid() ||
                !expert.down->isValid())
            {
                throw std::runtime_error(
                    std::string("NativeQuantised preparation failed for ") +
                    gate_up_format.label + "/" + down_format.label);
            }
            fixture.experts.push_back(std::move(expert));
        }

        // Arm A per-expert production scratch (FP32Tensor as production uses).
        for (int slot = 0; slot < kTopK; ++slot)
        {
            fixture.a_gate.push_back(std::make_unique<FP32Tensor>(
                std::vector<size_t>{1u, static_cast<size_t>(kIntermediate)}));
            fixture.a_up.push_back(std::make_unique<FP32Tensor>(
                std::vector<size_t>{1u, static_cast<size_t>(kIntermediate)}));
            fixture.a_activated.push_back(std::make_unique<FP32Tensor>(
                std::vector<size_t>{1u, static_cast<size_t>(kIntermediate)}));
            fixture.a_down.push_back(std::make_unique<FP32Tensor>(
                std::vector<size_t>{1u, static_cast<size_t>(kDModel)}));
        }

        fixture.liveGate = static_cast<size_t>(kTopK) *
                           static_cast<size_t>(kIntermediate);
        fixture.liveDown = static_cast<size_t>(kTopK) *
                           static_cast<size_t>(kDModel);
        fixture.liveRoute = static_cast<size_t>(kDModel);

        // Arm B expert-major banks with guard tails.
        constexpr size_t q8_guard = guardElements<Q8_1Block>();
        fixture.b_gate = AlignedVector<float>(fixture.liveGate + guardElements<float>());
        fixture.b_up = AlignedVector<float>(fixture.liveGate + guardElements<float>());
        fixture.b_activation = AlignedVector<Q8_1Block>(
            static_cast<size_t>(kTopK) * kActivationBlocks + q8_guard);
        fixture.b_down = AlignedVector<float>(fixture.liveDown + guardElements<float>());
        fixture.b_route_output = AlignedVector<float>(fixture.liveRoute + guardElements<float>());
        fixture.a_route_output = AlignedVector<float>(fixture.liveRoute + guardElements<float>());
        fixture.a_q8_proxy = AlignedVector<Q8_1Block>(
            static_cast<size_t>(kTopK) * kActivationBlocks + q8_guard);
        fillGuardTail(fixture.b_gate, fixture.liveGate);
        fillGuardTail(fixture.b_up, fixture.liveGate);
        fillGuardTail(fixture.b_down, fixture.liveDown);
        fillGuardTail(fixture.b_route_output, fixture.liveRoute);
        fillGuardTail(fixture.a_route_output, fixture.liveRoute);

        // Admitted invocation workspace for the production down launcher:
        // the exact bank production declares for one M=1 invocation. The
        // two-argument workspace manager allocates through the device
        // backend, so the CPU backend must be initialized first (the same
        // fixture pattern every device-free CPU test uses).
        if (!llaminar2::hasCPUBackend())
            llaminar2::initCPUBackend(-1);
        const auto q8_requirements =
            cpuProjectionQ8WorkspaceRequirements(kTopK, kIntermediate);
        if (!fixture.workspace.allocate(q8_requirements))
            throw std::runtime_error("CPU workspace admission failed");

        // Partial bank sized for the largest member family at one row.
        size_t partial_floats = 0;
        for (const PreparedExpert &expert : fixture.experts)
            for (const auto *kernel :
                 {expert.gate.get(), expert.up.get(), expert.down.get()})
            {
                partial_floats = std::max(
                    partial_floats,
                    cpu::native_quantised::nativeQuantisedProjectionPartialFloats(
                        kernel->packedWeights(), 1));
            }
        fixture.partial_storage.resize(partial_floats);
    }

    /** @brief Arm A: the exact production single-token sequence. */
    void runProductionArm(SingleTokenFixture &fixture)
    {
        using TensorProjectionDesc = ITensorGemm::TensorProjectionDesc;
        std::vector<TensorProjectionDesc> projections;
        projections.reserve(static_cast<size_t>(kTopK) * 2u);
        for (int slot = 0; slot < kTopK; ++slot)
        {
            projections.emplace_back(
                fixture.experts[static_cast<size_t>(slot)].gate.get(),
                fixture.a_gate[static_cast<size_t>(slot)].get(),
                kIntermediate,
                nullptr,
                "stt_gate");
            projections.emplace_back(
                fixture.experts[static_cast<size_t>(slot)].up.get(),
                fixture.a_up[static_cast<size_t>(slot)].get(),
                kIntermediate,
                nullptr,
                "stt_up");
        }
        // Production hosts the bundle on the first active expert's gate
        // engine and reuses the published router Q8 row as the shared input.
        if (!fixture.experts.front().gate
                 ->multiply_fused_router_q8_hidden_grouped_decode_equivalent(
                     fixture.published_q8,
                     projections,
                     /*m=*/1,
                     kDModel,
                     fixture.partial_storage))
        {
            throw std::runtime_error("production router-Q8 bundle rejected M=1");
        }

        // Production serial SwiGLU, one expert at a time.
        for (int slot = 0; slot < kTopK; ++slot)
        {
            const float *gate_row = fixture.a_gate[static_cast<size_t>(slot)]->data();
            const float *up_row = fixture.a_up[static_cast<size_t>(slot)]->data();
            float *activated_row =
                fixture.a_activated[static_cast<size_t>(slot)]->mutable_data();
            if (fixture.policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
            {
                primitives::compute_swiglu_gpu_aligned_expert_serial(
                    gate_row, up_row, activated_row, kIntermediate);
            }
            else
            {
                primitives::compute_swiglu_serial(
                    gate_row, up_row, activated_row, kIntermediate);
            }
        }

        // Production fused multi-input down launcher: the launcher performs
        // its own internal per-expert Q8 quantization of these FP32 rows.
        std::vector<ITensorGemm::FusedExpertDownDesc> down_descs;
        down_descs.reserve(kTopK);
        for (int slot = 0; slot < kTopK; ++slot)
        {
            down_descs.push_back({
                .kernel = fixture.experts[static_cast<size_t>(slot)].down.get(),
                .input = fixture.a_activated[static_cast<size_t>(slot)]->data(),
                .output = fixture.a_down[static_cast<size_t>(slot)]->mutable_data(),
                .n = kDModel,
            });
        }
        if (!fixture.experts.front().down
                 ->multiply_fused_expert_down(
                     down_descs.data(), kTopK, /*m=*/1, kIntermediate,
                     &fixture.workspace))
        {
            throw std::runtime_error("production fused expert down rejected M=1");
        }

        // Production route publication: ordered weighted accumulation.
        std::fill_n(
            fixture.a_route_output.data(), fixture.liveRoute, 0.0f);
        for (int slot = 0; slot < kTopK; ++slot)
        {
            primitives::vec_axpy(
                fixture.a_route_output.data(),
                fixture.a_down[static_cast<size_t>(slot)]->data(),
                fixture.route_weights[static_cast<size_t>(slot)],
                kDModel);
        }
    }

    /** @brief Arm B: the existing grouped transaction at the one-row route. */
    void runTransactionArm(SingleTokenFixture &fixture)
    {
        using Descriptor =
            CPUNativeQuantisedGemmKernel::BatchedPrequantizedProjectionDesc;
        std::vector<Descriptor> gate_up_descriptors;
        std::vector<Descriptor> down_descriptors;
        gate_up_descriptors.reserve(static_cast<size_t>(kTopK) * 2u);
        down_descriptors.reserve(kTopK);
        for (int slot = 0; slot < kTopK; ++slot)
        {
            const size_t expert = static_cast<size_t>(slot);
            // The actual shared-router-Q8 shape: every gate/up member reads
            // the same published router Q8 row pointer. No copies.
            gate_up_descriptors.push_back({
                .kernel = fixture.experts[expert].gate.get(),
                .input_q8 = fixture.published_q8,
                .output = fixture.b_gate.data() +
                          expert * static_cast<size_t>(kIntermediate),
                .bias = nullptr,
                .rows = 1,
                .n = kIntermediate,
                .ldc = kIntermediate,
                .purpose = cpu::native_quantised::ProjectionRowsPurpose::Decode,
            });
            gate_up_descriptors.push_back({
                .kernel = fixture.experts[expert].up.get(),
                .input_q8 = fixture.published_q8,
                .output = fixture.b_up.data() +
                          expert * static_cast<size_t>(kIntermediate),
                .bias = nullptr,
                .rows = 1,
                .n = kIntermediate,
                .ldc = kIntermediate,
                .purpose = cpu::native_quantised::ProjectionRowsPurpose::Decode,
            });
            down_descriptors.push_back({
                .kernel = fixture.experts[expert].down.get(),
                .input_q8 = fixture.b_activation.data() +
                            expert * static_cast<size_t>(kActivationBlocks),
                .output = fixture.b_down.data() +
                          expert * static_cast<size_t>(kDModel),
                .bias = nullptr,
                .rows = 1,
                .n = kDModel,
                .ldc = kDModel,
                .purpose = cpu::native_quantised::ProjectionRowsPurpose::Decode,
            });
        }
        if (!CPUNativeQuantisedGemmKernel::
                execute_moe_grouped_ffn_transaction_preq_decode_equivalent(
                    gate_up_descriptors.data(),
                    static_cast<int>(gate_up_descriptors.size()),
                    kDModel,
                    fixture.b_gate.data(),
                    fixture.b_up.data(),
                    fixture.b_activation.data(),
                    /*route_rows=*/kTopK,
                    kIntermediate,
                    kActivationBlocks,
                    down_descriptors.data(),
                    static_cast<int>(down_descriptors.size()),
                    fixture.partial_storage))
        {
            // The admission gap IS the finding: the transaction rejected the
            // actual shared-router-Q8 M=1 shape. No fallback exists here.
            ADD_FAILURE()
                << "ADMISSION GAP: the grouped FFN transaction rejected the "
                << "actual shared-router-Q8 M=1 shape (16 gate/up + 8 down "
                << "descriptors, rows=1, shared input pointer). Report the "
                << "gap; do not reshape or modify production to admit it.";
            return;
        }

        // Same production route publication over arm B's down rows.
        std::fill_n(fixture.b_route_output.data(), fixture.liveRoute, 0.0f);
        for (int slot = 0; slot < kTopK; ++slot)
        {
            primitives::vec_axpy(
                fixture.b_route_output.data(),
                fixture.b_down.data() +
                    static_cast<size_t>(slot) * static_cast<size_t>(kDModel),
                fixture.route_weights[static_cast<size_t>(slot)],
                kDModel);
        }
    }

    /** @brief Run every hard byte gate and guard check; no timing involved. */
    void runByteGates(SingleTokenFixture &fixture)
    {
        runProductionArm(fixture);
        runTransactionArm(fixture);

        // Production activation quantization of arm A's SwiGLU rows: this is
        // the production helper applied to production SwiGLU outputs, giving
        // the byte-comparable proxy for arm A's internal launcher Q8 bank.
        for (int slot = 0; slot < kTopK; ++slot)
        {
            cpu::native_quantised::quantize_activations_to_q8_1(
                fixture.a_activated[static_cast<size_t>(slot)]->data(),
                fixture.a_q8_proxy.data() +
                    static_cast<size_t>(slot) * kActivationBlocks,
                1,
                kIntermediate,
                kActivationBlocks,
                fixture.policy);
        }

        const size_t gate_live = static_cast<size_t>(kIntermediate);
        const size_t activation_live = static_cast<size_t>(kActivationBlocks);
        for (int slot = 0; slot < kTopK; ++slot)
        {
            const size_t expert = static_cast<size_t>(slot);
            expectByteEqual(
                fixture.a_gate[expert]->data(),
                fixture.b_gate.data() + expert * gate_live,
                gate_live,
                "gate projection (arm A vs arm B, slot " + std::to_string(slot) + ")");
            expectByteEqual(
                fixture.a_up[expert]->data(),
                fixture.b_up.data() + expert * gate_live,
                gate_live,
                "up projection (arm A vs arm B, slot " + std::to_string(slot) + ")");
            expectByteEqual(
                fixture.a_q8_proxy.data() + expert * activation_live,
                fixture.b_activation.data() + expert * activation_live,
                activation_live,
                "SwiGLU Q8_1 publication (arm A proxy vs arm B bank, slot " +
                    std::to_string(slot) + ")");
            expectByteEqual(
                fixture.a_down[expert]->data(),
                fixture.b_down.data() + expert * static_cast<size_t>(kDModel),
                static_cast<size_t>(kDModel),
                "down projection (arm A vs arm B, slot " + std::to_string(slot) + ")");
        }
        expectByteEqual(
            fixture.a_route_output.data(),
            fixture.b_route_output.data(),
            fixture.liveRoute,
            "route contribution (arm A vs arm B)");

        expectGuardTailIntact(fixture.b_gate, fixture.liveGate, "arm B gate bank");
        expectGuardTailIntact(fixture.b_up, fixture.liveGate, "arm B up bank");
        expectGuardTailIntact(fixture.b_down, fixture.liveDown, "arm B down bank");
        expectGuardTailIntact(
            fixture.b_route_output, fixture.liveRoute, "arm B route output");
        expectGuardTailIntact(
            fixture.a_route_output, fixture.liveRoute, "arm A route output");

        // Sealed inputs must be untouched after every executed arm.
        expectByteEqual(
            fixture.hidden.data(), fixture.sealed_hidden.data(),
            fixture.sealed_hidden.size(), "hidden row immutability");
        expectByteEqual(
            fixture.router_gate.data(), fixture.sealed_router_gate.data(),
            fixture.sealed_router_gate.size(), "router gate immutability");
        expectByteEqual(
            fixture.published_q8, fixture.sealed_q8.data(),
            fixture.sealed_q8.size(), "published router Q8 immutability");
    }

    /** @brief Fresh-counter proof that both intended production paths ran. */
    void runPathProof(SingleTokenFixture &fixture)
    {
        constexpr const char *kBundleFamily =
            "kernel.cpu_moe_decode_router_q8_reuse_calls";
        constexpr const char *kDownFamily =
            "kernel.cpu_native_quantised_fused_expert_down_calls";
        constexpr const char *kTransactionFamily =
            "kernel.cpu_native_quantised_moe_grouped_ffn_transactions";

        KernelDomainProofScope scope;
        const auto before_bundle = PerfStatsCollector::snapshot({kBundleFamily});
        const auto before_down = PerfStatsCollector::snapshot({kDownFamily});
        const auto before_transaction =
            PerfStatsCollector::snapshot({kTransactionFamily});

        runProductionArm(fixture);
        const auto after_a_bundle = PerfStatsCollector::snapshot({kBundleFamily});
        const auto after_a_down = PerfStatsCollector::snapshot({kDownFamily});
        const auto fresh_bundle = [&]() {
            uint64_t count = 0;
            for (const auto &record : after_a_bundle)
            {
                const auto prior = std::find_if(
                    before_bundle.begin(), before_bundle.end(),
                    [&](const auto &entry) { return entry.tags == record.tags; });
                const uint64_t old =
                    prior == before_bundle.end() ? 0 : prior->count;
                if (record.count > old)
                    count += record.count - old;
            }
            return count;
        }();
        const auto fresh_down = [&]() {
            uint64_t count = 0;
            for (const auto &record : after_a_down)
            {
                const auto prior = std::find_if(
                    before_down.begin(), before_down.end(),
                    [&](const auto &entry) { return entry.tags == record.tags; });
                const uint64_t old =
                    prior == before_down.end() ? 0 : prior->count;
                if (record.count > old)
                    count += record.count - old;
            }
            return count;
        }();

        runTransactionArm(fixture);
        const auto after_b = PerfStatsCollector::snapshot({kTransactionFamily});
        const auto fresh_transaction = [&]() {
            uint64_t count = 0;
            for (const auto &record : after_b)
            {
                const auto prior = std::find_if(
                    before_transaction.begin(), before_transaction.end(),
                    [&](const auto &entry) { return entry.tags == record.tags; });
                const uint64_t old =
                    prior == before_transaction.end() ? 0 : prior->count;
                if (record.count > old)
                    count += record.count - old;
            }
            return count;
        }();

        EXPECT_EQ(fresh_bundle, 1u)
            << "arm A must advance the production router-Q8 bundle counter";
        EXPECT_EQ(fresh_down, 1u)
            << "arm A must advance the production fused expert-down counter";
        EXPECT_EQ(fresh_transaction, 1u)
            << "arm B must advance the grouped FFN transaction counter";
        std::cout << "CPU_STT_PROOF,bundle=" << fresh_bundle
                  << ",expert_down=" << fresh_down
                  << ",transaction=" << fresh_transaction
                  << ",outside_timing=1\n";
    }
} // namespace

TEST(Perf__CPUMoESingleTokenTransaction,
     ProductionSingleTokenVersusTransactionIsByteExactAndEconomical)
{
    const int warmup_pairs = envPositiveInt("LLAMINAR_CPU_MOE_STT_WARMUP", 8);
    const int sample_pairs =
        envPositiveInt("LLAMINAR_CPU_MOE_STT_SAMPLES", 120);
    if (sample_pairs < 100)
        throw std::runtime_error(
            "LLAMINAR_CPU_MOE_STT_SAMPLES must be at least 100 paired samples");

    SingleTokenFixture fixture;
    prepareFixture(fixture);

    // Hard byte gates first: no timing until both arms agree byte for byte.
    runByteGates(fixture);
    ASSERT_FALSE(::testing::Test::HasFailure())
        << "byte gates failed; no timing evidence is collected";

    const auto gate_up_label = selectedGateUpFormat().label;
    const auto down_label = selectedDownFormat().label;
    const char *policy_label =
        fixture.policy == CPUProjectionNumericalPolicy::GPUAlignedExpert
            ? "GPUAlignedExpert"
            : "BackendNative";
    std::cout << "CPU_STT_LAYOUT,format=" << gate_up_label
              << ",down_format=" << down_label << ",policy=" << policy_label
              << ",d_model=" << kDModel << ",intermediate=" << kIntermediate
              << ",experts=" << kTopK << ",route_rows=1"
              << ",threads=" << omp_get_max_threads()
              << ",isa=" << isaLevelName(activeISALevel())
              << ",outside_timing=1\n";

    // Unprofiled alternating paired samples; within-pair order alternates.
    // The collection scope is confined to this block so the post-timing byte
    // gates and the proof pass run under the restored caller policy.
    std::vector<double> a_samples;
    std::vector<double> b_samples;
    a_samples.reserve(static_cast<size_t>(sample_pairs));
    b_samples.reserve(static_cast<size_t>(sample_pairs));
    // Each arm is timed individually inside the pair to keep per-arm samples
    // independent of the other arm's duration.
    auto timed_arm = [&fixture](bool production) {
        const auto begin = std::chrono::steady_clock::now();
        if (production)
            runProductionArm(fixture);
        else
            runTransactionArm(fixture);
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(end - begin).count();
    };

    {
        test::CPUExactTimingCollectionScope timing_scope;
        for (int pair = 0; pair < warmup_pairs; ++pair)
        {
            runProductionArm(fixture);
            runTransactionArm(fixture);
        }

        for (int pair = 0; pair < sample_pairs; ++pair)
        {
            const bool production_first = (pair % 2) == 0;
            if (production_first)
            {
                a_samples.push_back(timed_arm(true));
                b_samples.push_back(timed_arm(false));
            }
            else
            {
                b_samples.push_back(timed_arm(false));
                a_samples.push_back(timed_arm(true));
            }
        }
    }

    // Post-timing byte gates: the timed arms must still agree byte for byte.
    runByteGates(fixture);
    ASSERT_FALSE(::testing::Test::HasFailure())
        << "post-timing byte gates failed";

    const double a_median = medianUs(a_samples);
    const double b_median = medianUs(b_samples);
    const double a_mean = meanUs(a_samples);
    const double b_mean = meanUs(b_samples);
    const double a_sd = deviationUs(a_samples);
    const double b_sd = deviationUs(b_samples);
    std::cout << "CPU_STT_TIMING,format=" << gate_up_label
              << ",down_format=" << down_label << ",policy=" << policy_label
              << ",threads=" << omp_get_max_threads()
              << ",pairs=" << sample_pairs
              << ",production_median_us=" << std::fixed
              << std::setprecision(3) << a_median
              << ",transaction_median_us=" << b_median
              << ",production_mean_us=" << a_mean
              << ",production_sd_us=" << a_sd
              << ",transaction_mean_us=" << b_mean
              << ",transaction_sd_us=" << b_sd
              << ",transaction_over_production=" << (b_median / a_median)
              << ",unprofiled=1\n";

    runPathProof(fixture);

    // Diagnostic honesty: a median ratio at or below one is a non-regression,
    // not an optimization win. No production economy assertion is made here.
    std::cout << "CPU_STT_VERDICT,transaction_vs_production_median="
              << std::fixed << std::setprecision(4) << (b_median / a_median)
              << ",interpretation=diagnostic_only,closed_by=numbers\n";
}
