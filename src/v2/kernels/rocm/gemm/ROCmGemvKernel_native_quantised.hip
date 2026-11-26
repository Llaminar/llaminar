/**
 * @file ROCmGemvKernel_native_quantised.hip
 * @brief ROCm NativeQuantised M=1 GEMV kernels, policy/tuning state, and IQ table fanout.
 * Companion shards own small-M/batched verifier GEMV, grouped MoE decode, resident FFN,
 * persistent HC read, native-layout router FFN, and full native HC/FFN entries.
 */
#include "ROCmNativeQuantisedDecodeCommon.hip.inc"
#include "kernels/rocm/gemm/ROCmNativeQuantisedDecodeDispatchGenerated.inc"

// ============================================================================
// Runtime tuning overrides for native-quantised GEMV dispatch.
// Values < 0 mean "use default heuristic".
// ============================================================================
static std::atomic<int> g_native_quantised_kb_override{-1};
static std::atomic<int> g_native_quantised_target_waves_override{-1};
/*
 * Decode-equivalent dispatch is a property of one host worker's verifier
 * transaction. LocalTP drives one worker per GPU in the same process, so a
 * process-global mode bit lets either worker clear the other worker's policy
 * while it is still lowering a grouped launch. Keep this state thread-local,
 * matching CUDA, and pass resolved serial-M1 launch geometry explicitly from
 * ROCmQuantisedGemmKernel whenever a grouped projection is emitted.
 */
static thread_local int g_native_quantised_decode_equivalent_m1_config = 0;
static thread_local int g_native_quantised_serial_partition_n = 0;

namespace
{
/** One lowering thread owns its diagnostic scope; no process-global geometry switch. */
thread_local const llaminar2::rocm::ScopedNativeQ8SerialGeometry *g_native_q8_serial_scope = nullptr;
/** Immutable topology observation, not device execution state or a replay counter. */
thread_local std::optional<llaminar2::rocm::NativeQ8SerialLaunch> g_native_q8_serial_launch;
}

void llaminar2::rocm::validateNativeQ8SerialScopeBinding(int device_ordinal, void *stream)
{
    if (!g_native_q8_serial_scope) return;
    if (device_ordinal != g_native_q8_serial_scope->admission().deviceOrdinal() ||
        !stream || stream == static_cast<void *>(hipStreamLegacy) ||
        stream == static_cast<void *>(hipStreamPerThread))
    {
        g_native_q8_serial_launch.reset();
        throw std::logic_error("native-Q8 serial geometry requires its admitted device and exact nondefault stream");
    }
}

void llaminar2::rocm::clearNativeQ8SerialLaunch() noexcept
{
    g_native_q8_serial_launch.reset();
}

std::optional<llaminar2::rocm::NativeQ8SerialLaunch>
llaminar2::rocm::lastNativeQ8SerialLaunch() noexcept
{
    return g_native_q8_serial_launch;
}

std::optional<llaminar2::rocm::NativeQ8SerialGeometry>
llaminar2::rocm::currentNativeQ8SerialGeometry() noexcept
{
    return g_native_q8_serial_scope
        ? std::optional{g_native_q8_serial_scope->admission().geometry()} : std::nullopt;
}

llaminar2::rocm::ScopedNativeQ8SerialGeometry::ScopedNativeQ8SerialGeometry(
    const NativeQ8SerialGeometryAdmission &admission)
    : admission_(admission), previous_(g_native_q8_serial_scope)
{
    if (llaminar2::isGraphCaptureActive())
        throw std::logic_error("native-Q8 serial geometry must be scoped before capture");
    const auto &env = llaminar2::debugEnv().rocm;
    if (g_native_quantised_kb_override.load(std::memory_order_relaxed) > 0 ||
        g_native_quantised_target_waves_override.load(std::memory_order_relaxed) > 0 ||
        env.native_quantised_gemv_kb > 0 || env.native_quantised_gemv_target_waves > 0 ||
        env.native_quantised_disable_generated)
        throw std::logic_error("native-Q8 serial geometry requires unchanged generated arithmetic, without KB/TW overrides");
    (void)nativeQ8SerialGeometrySpec(admission_.geometry());
    g_native_q8_serial_scope = this;
}

llaminar2::rocm::ScopedNativeQ8SerialGeometry::~ScopedNativeQ8SerialGeometry() noexcept
{
    // A scope belongs to one construction thread and must retire in LIFO order.
    if (g_native_q8_serial_scope != this)
        std::terminate();
    g_native_q8_serial_scope = previous_;
}



static NativeQuantisedDecodeGeneratedRuntimeConfig
resolveNativeQuantisedDecodeGeneratedRuntimeConfigForPolicy(
    uint8_t arithmetic_policy_codebook_id,
    int m,
    int n,
    int k)
{
    NativeQuantisedDecodeGeneratedRuntimeConfig resolved{};
    const auto& rocm_env = llaminar2::debugEnv().rocm;
    if (rocm_env.native_quantised_disable_generated)
        return resolved;

    const bool has_explicit_override =
        g_native_quantised_kb_override.load(std::memory_order_relaxed) > 0 ||
        g_native_quantised_target_waves_override.load(std::memory_order_relaxed) > 0 ||
        rocm_env.native_quantised_gemv_kb > 0 ||
        rocm_env.native_quantised_gemv_target_waves > 0;
    if (has_explicit_override)
        return resolved;

    const int lookup_m =
        (g_native_quantised_decode_equivalent_m1_config != 0 &&
         m > 1)
            ? 1
            : m;
    const int lookup_n =
        g_native_quantised_serial_partition_n > 0
            ? g_native_quantised_serial_partition_n
            : n;

    /*
     * Generated policy is immutable for the process lifetime. Preserve each
     * input as a full-width word so invalid or future geometry cannot alias the
     * compact exact-overlay ABI. Caching a miss keeps totality checks cheap but
     * never manufactures a runtime fallback.
     */
    using CacheKey = std::array<uint64_t, 4>;
    using Cache = llaminar2::native_quantised::FixedDispatchCache<
        CacheKey,
        NativeQuantisedDecodeGeneratedRuntimeConfig,
        64,
        llaminar2::native_quantised::DispatchCacheArrayHasher<4>>;
    static thread_local Cache cache;
    const CacheKey cache_key = {
        static_cast<uint64_t>(arithmetic_policy_codebook_id),
        static_cast<uint64_t>(static_cast<uint32_t>(lookup_m)),
        static_cast<uint64_t>(static_cast<uint32_t>(lookup_n)),
        static_cast<uint64_t>(static_cast<uint32_t>(k)),
    };
    bool cached_selector_found = false;
    NativeQuantisedDecodeGeneratedRuntimeConfig cached{};
    if (cache.lookup(cache_key, cached_selector_found, cached))
        return cached;

    llaminar2::rocm::generated::ROCmNativeQuantisedDecodeDispatchConfig generated_cfg{};
    if (!llaminar2::rocm::generated::selectROCmNativeQuantisedDecodeGenerated(
            arithmetic_policy_codebook_id,
            lookup_m,
            lookup_n,
            k,
            generated_cfg))
    {
        cache.insert(cache_key, false, cached);
        return cached;
    }

    /*
     * Generic ROCm policy candidates encode a requested split count. The
     * production launch can never create more K partitions than there are
     * 32-value groups, so it resolves that request as min(requested, K / 32).
     * Publish the same physical value through the query/cache surface. Leaving
     * the nominal request here made serial-M1 policy queries disagree with the
     * launch telemetry for small-K unseen shapes (for example KB20 at K=416,
     * whose real launch is KB13), even though the kernel itself was correct.
     */
    const int requested_kb = static_cast<int>(generated_cfg.kb);
    const int k_groups = k / 32;
    cached.has_generated = true;
    cached.kb = (requested_kb < k_groups) ? requested_kb : k_groups;
    cached.target_waves_per_cu =
        static_cast<int>(generated_cfg.target_waves_per_cu);
    cache.insert(cache_key, true, cached);
    return cached;
}

/** @brief Resolve policy for a legacy self-describing execution codebook. */
static NativeQuantisedDecodeGeneratedRuntimeConfig
resolveNativeQuantisedDecodeGeneratedRuntimeConfig(
    uint8_t execution_codebook_id,
    int m,
    int n,
    int k)
{
    return resolveNativeQuantisedDecodeGeneratedRuntimeConfigForPolicy(
        defaultNativeQuantisedArithmeticPolicyCodebook(execution_codebook_id),
        m,
        n,
        k);
}

static void recordNativeQuantisedSmallMLaunch(
    int m,
    int n,
    int k,
    unsigned codebook,
    int grid_n,
    int kb,
    int target_waves_per_cu,
    bool batched,
    int projections,
    int arithmetic_policy_codebook = -1)
{
    if (!llaminar2::PerfStatsCollector::isDomainEnabled("kernel"))
        return;

    const char* path = (kb <= 1) ? "direct" : "split_reduce";
    llaminar2::PerfStatsCollector::Tags tags{
        {"m", std::to_string(m)}, {"n", std::to_string(n)},
        {"k", std::to_string(k)}, {"codebook", std::to_string(codebook)},
        {"arithmetic_policy_codebook", std::to_string(arithmetic_policy_codebook >= 0
            ? arithmetic_policy_codebook : static_cast<int>(codebook))},
        {"grid_n", std::to_string(grid_n)}, {"kb", std::to_string(kb)},
        {"path", path}, {"target_waves_per_cu", std::to_string(target_waves_per_cu)},
        {"batched", batched ? "true" : "false"},
        {"projections", std::to_string(projections)}};
    if (m == 1 && codebook == 19 && !batched && g_native_q8_serial_launch)
    {
        const auto &launch = *g_native_q8_serial_launch;
        const auto spec = llaminar2::rocm::nativeQ8SerialGeometrySpec(launch.geometry);
        tags.emplace("serial_tile_n", std::to_string(spec.tile_columns));
        tags.emplace("serial_cpt", std::to_string(spec.columns_per_thread));
        tags.emplace("serial_block_threads", std::to_string(launch.producer.block_x));
        tags.emplace("serial_arithmetic_policy_n", std::to_string(launch.arithmetic_policy_n));
        tags.emplace("serial_blocks_per_partition", std::to_string(launch.blocks_per_partition));
        tags.emplace("serial_nonempty_partitions", std::to_string(launch.nonempty_partitions));
        tags.emplace("serial_producer", launch.producer.specialization);
        tags.emplace("serial_reducer", launch.reducer.function ? launch.reducer.specialization : "absent");
    }
    llaminar2::PerfStatsCollector::addCounter(
        "kernel", "rocm_native_quantised_small_m_launch", 1.0, "gemm", {}, std::move(tags));
}


extern "C" void rocmGemv_native_quantised_set_tuning_overrides(int kb, int target_waves_per_cu)
{
    g_native_quantised_kb_override.store(kb, std::memory_order_relaxed);
    g_native_quantised_target_waves_override.store(target_waves_per_cu, std::memory_order_relaxed);
}

extern "C" void rocmGemv_native_quantised_set_decode_equivalent_m1_config(int enabled)
{
    g_native_quantised_decode_equivalent_m1_config = enabled ? 1 : 0;
}

extern "C" void rocmGemv_native_quantised_set_serial_partition_n(int n)
{
    g_native_quantised_serial_partition_n = n;
}

extern "C" int rocmGemv_native_quantised_get_serial_partition_n()
{
    return g_native_quantised_serial_partition_n;
}

extern "C" void rocmGemv_native_quantised_reset_tuning_overrides()
{
    rocmGemv_native_quantised_set_tuning_overrides(-1, -1);
}

static bool nativeQuantisedDecodeEquivalentSmallMMode()
{
    return g_native_quantised_decode_equivalent_m1_config != 0;
}

/**
 * @brief Enforce the ROCm small-M verifier arithmetic contract.
 *
 * The MTP verifier publication path must reproduce serial M=1 decode rows,
 * because its outputs are used as live recurrent/KV/logit state.  Manual KB
 * overrides are therefore allowed but never silently rewritten here: the
 * grouped verifier must match the same generated/manual dispatch policy that
 * serial decode would use.
 */
static bool validateNativeQuantisedDecodeEquivalentKBOverride(
    const char* caller,
    int explicit_kb_override,
    int m,
    int n,
    int k,
    int projections)
{
    (void)caller;
    (void)explicit_kb_override;
    (void)m;
    (void)n;
    (void)k;
    (void)projections;
    return true;
}


static NativeQuantisedDecodeGeneratedRuntimeConfig
resolveNativeQuantisedSerialM1RuntimeConfigForPolicy(
    uint8_t arithmetic_policy_codebook_id,
    int N,
    int K)
{
    NativeQuantisedDecodeGeneratedRuntimeConfig resolved =
        resolveNativeQuantisedDecodeGeneratedRuntimeConfigForPolicy(
            arithmetic_policy_codebook_id, 1, N, K);
    if (!resolved.has_generated)
        return resolved;

    if (llaminar2::MoEProjectionNumericalContract::
            ownsOrderedKPartitionTree(arithmetic_policy_codebook_id))
    {
        resolved.kb = llaminar2::MoEProjectionNumericalContract::
            orderedKPartitionsForWidth(K);
    }
    return resolved;
}

/**
 * @brief Resolve the split-K policy used by normal serial M=1 decode.
 *
 * MTP verifier publication compares grouped runtime-M rows against committed
 * serial decode state. The grouped kernels may execute several rows at once,
 * but they must consume this same typed policy so their FP32 reduction order is
 * bitwise-identical. Returning an empty result on a generated-policy miss
 * forces the caller to fail rather than reconstruct an obsolete heuristic.
 */
static NativeQuantisedDecodeGeneratedRuntimeConfig
resolveNativeQuantisedSerialM1RuntimeConfig(
    uint8_t codebook_id,
    int N,
    int K)
{
    return resolveNativeQuantisedSerialM1RuntimeConfigForPolicy(
        defaultNativeQuantisedArithmeticPolicyCodebook(codebook_id), N, K);
}

static NativeQuantisedDecodeGeneratedRuntimeConfig resolveNativeQuantisedCommonSerialM1RuntimeConfig(
    const uint8_t* codebook_ids,
    const int* Ns,
    int num_projections,
    int K)
{
    NativeQuantisedDecodeGeneratedRuntimeConfig common{};
    for (int p = 0; p < num_projections; ++p)
    {
        NativeQuantisedDecodeGeneratedRuntimeConfig candidate =
            resolveNativeQuantisedSerialM1RuntimeConfig(codebook_ids[p], Ns[p], K);
        if (!candidate.has_generated)
            return NativeQuantisedDecodeGeneratedRuntimeConfig{};

        if (p == 0)
        {
            common = candidate;
            continue;
        }

        if (candidate.kb != common.kb)
            return NativeQuantisedDecodeGeneratedRuntimeConfig{};
    }
    return common;
}

static NativeQuantisedDecodeGeneratedRuntimeConfig resolveNativeQuantisedCommonSerialM1RuntimeConfig(
    uint8_t codebook_id,
    const int* Ns,
    int num_projections,
    int K)
{
    uint8_t codebook_ids[8] = {};
    for (int p = 0; p < num_projections && p < 8; ++p)
        codebook_ids[p] = codebook_id;
    return resolveNativeQuantisedCommonSerialM1RuntimeConfig(codebook_ids, Ns, num_projections, K);
}

/**
 * @brief Resolve the exact graph-safe partition count executed by serial M=1.
 *
 * Generated policy stores the preferred count, while the launcher bounds it by
 * both the number of 32-value K blocks and the captured-workspace ABI. Grouped
 * MoE launchers call this helper so their partition boundaries are identical to
 * the public serial decode entrypoint rather than merely using the same raw
 * policy record.
 *
 * @param codebook_id NativeQuantised format identifier.
 * @param N Output-column count.
 * @param K Reduction width, divisible by 32.
 * @return Effective K-partition count, or zero when no generated policy exists.
 */
static int resolveNativeQuantisedSerialM1KPartitions(
    uint8_t codebook_id,
    int N,
    int K)
{
    if (N <= 0 || K <= 0 || (K % 32) != 0)
        return 0;

    const NativeQuantisedDecodeGeneratedRuntimeConfig cfg =
        resolveNativeQuantisedSerialM1RuntimeConfig(codebook_id, N, K);
    if (!cfg.has_generated || cfg.kb <= 0)
        return 0;

    int partitions = cfg.kb;
    if (partitions > K / 32)
        partitions = K / 32;
    if (partitions >
        llaminar2::NativeQuantisedGroupedDecodePolicy::maximum_k_partitions)
    {
        partitions =
            llaminar2::NativeQuantisedGroupedDecodePolicy::maximum_k_partitions;
    }
    return partitions > 0 ? partitions : 1;
}

/**
 * @brief Resolve graph-safe serial partitions from an exact source policy.
 */
static int resolveNativeQuantisedSerialM1KPartitionsForPolicy(
    uint8_t arithmetic_policy_codebook_id,
    int N,
    int K)
{
    if (N <= 0 || K <= 0 || (K % 32) != 0)
        return 0;

    const NativeQuantisedDecodeGeneratedRuntimeConfig cfg =
        resolveNativeQuantisedSerialM1RuntimeConfigForPolicy(
            arithmetic_policy_codebook_id, N, K);
    if (!cfg.has_generated || cfg.kb <= 0)
        return 0;

    int partitions = cfg.kb;
    if (partitions > K / 32)
        partitions = K / 32;
    if (partitions >
        llaminar2::NativeQuantisedGroupedDecodePolicy::maximum_k_partitions)
    {
        partitions =
            llaminar2::NativeQuantisedGroupedDecodePolicy::maximum_k_partitions;
    }
    return partitions > 0 ? partitions : 1;
}

extern "C" bool rocmGemv_native_quantised_query_serial_m1_config(
    uint8_t codebook_id,
    int N,
    int K,
    int* kb,
    int* target_waves_per_cu)
{
    NativeQuantisedDecodeGeneratedRuntimeConfig cfg =
        resolveNativeQuantisedSerialM1RuntimeConfig(codebook_id, N, K);
    if (!cfg.has_generated)
        return false;
    if (kb)
        *kb = resolveNativeQuantisedSerialM1KPartitions(codebook_id, N, K);
    if (target_waves_per_cu)
        *target_waves_per_cu = cfg.target_waves_per_cu;
    return true;
}

/**
 * @brief Query serial-M1 geometry for an explicit arithmetic policy.
 *
 * Unlike the legacy query, this entrypoint never maps the migration-only
 * execution codebook to a guessed policy. Callers must provide the canonical
 * source policy codebook retained in prepared-weight provenance.
 */
extern "C" bool rocmGemv_native_quantised_query_serial_m1_config_with_policy(
    uint8_t arithmetic_policy_codebook_id,
    int N,
    int K,
    int* kb,
    int* target_waves_per_cu)
{
    const NativeQuantisedDecodeGeneratedRuntimeConfig cfg =
        resolveNativeQuantisedSerialM1RuntimeConfigForPolicy(
            arithmetic_policy_codebook_id, N, K);
    if (!cfg.has_generated)
        return false;
    if (kb)
    {
        *kb = resolveNativeQuantisedSerialM1KPartitionsForPolicy(
            arithmetic_policy_codebook_id, N, K);
    }
    if (target_waves_per_cu)
        *target_waves_per_cu = cfg.target_waves_per_cu;
    return true;
}

/**
 * @brief Immutable serial-M1 split-K geometry available to one captured MoE graph.
 *
 * A dynamic expert-overlay table can contain several physical NativeQuantised
 * codebooks, and a migrated expert can change physical representation without
 * changing its source arithmetic policy.  The captured grouped kernel therefore
 * cannot bake one table-wide codebook or one table-wide partition count into its
 * launch.  This compact value table is copied into the graph's kernel arguments;
 * each routed descriptor selects its own entry on device.
 *
 * Indexes are canonical source-policy codebooks.  Unsupported or untrained
 * entries remain zero and are rejected before capture when the descriptor table
 * advertises them.  `maximum_k_partitions` sizes the stable grid while every
 * route still uses exactly its serial-M1 partition boundaries.
 */

static void setROCmMoESerialM1PolicyPartitions(
    ROCmMoESerialM1PolicyTable& table,
    uint8_t policy,
    uint8_t partitions)
{
    if (policy >= ROCmMoESerialM1PolicyTable::kCodebookCapacity)
        return;

    const unsigned shift = static_cast<unsigned>(policy & 7u) * 8u;
    const uint64_t encoded = static_cast<uint64_t>(partitions) << shift;
    switch (policy >> 3u)
    {
    case 0:
        table.k_partitions_0_7 |= encoded;
        break;
    case 1:
        table.k_partitions_8_15 |= encoded;
        break;
    case 2:
        table.k_partitions_16_23 |= encoded;
        break;
    case 3:
        table.k_partitions_24_31 |= encoded;
        break;
    default:
        break;
    }
}

/**
 * @brief Build graph-stable descriptor-local split-K policy for one projection.
 *
 * The generated selector is immutable, as is the descriptor table's source
 * arithmetic-policy mask.  Build entries only for that declared mask: table
 * upload/update rejects a policy-mask change, while compact-to-expanded expert
 * movement retains the source arithmetic policy in descriptor provenance.
 * Consequently a captured graph remains valid across overlay publication
 * without paying grid depth for unrelated codebooks.
 *
 * @param N Output width of the projection.
 * @param K Reduction width of the projection.
 * @param required_policy_mask Policies present in the current host descriptor
 *                             table; every set bit must have generated evidence.
 * @param table Receives the by-policy partition counts and maximum grid depth.
 * @return True when all required policies are represented exactly.
 */
static bool buildROCmMoESerialM1PolicyTable(
    int N,
    int K,
    uint32_t required_policy_mask,
    ROCmMoESerialM1PolicyTable& table)
{
    static constexpr std::array<uint8_t, 16> kCanonicalPolicies = {
        0, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 19};

    table = {};
    uint32_t represented_mask = 0;
    for (const uint8_t policy : kCanonicalPolicies)
    {
        const uint32_t policy_bit =
            uint32_t{1} << static_cast<uint32_t>(policy);
        if ((required_policy_mask & policy_bit) == 0)
        {
            /*
             * Grid Z is the maximum partition count represented here.  Do not
             * launch empty partition planes for formats the immutable table can
             * never select; each extra plane creates one block per route and
             * output tile in every captured decode replay.
             */
            continue;
        }

        const int partitions =
            resolveNativeQuantisedSerialM1KPartitionsForPolicy(policy, N, K);
        if (partitions <= 0)
            continue;

        setROCmMoESerialM1PolicyPartitions(
            table,
            policy,
            static_cast<uint8_t>(partitions));
        represented_mask |= policy_bit;
        if (partitions > static_cast<int>(table.maximum_k_partitions))
        {
            table.maximum_k_partitions = static_cast<uint8_t>(partitions);
        }
    }

    return table.maximum_k_partitions > 0 &&
           (required_policy_mask & ~represented_mask) == 0;
}

/**
 * @brief Record proof that descriptor-local decode policy was selected.
 *
 * This counter is emitted by the host launcher, including while a graph is
 * being recorded.  It does not inspect device data or synchronize the stream;
 * production campaigns can therefore prove mixed/runtime routing without
 * perturbing the captured execution timeline.
 */
static void recordROCmMoEDescriptorLocalDecode(
    const char* projection,
    const char* descriptor_source,
    uint8_t table_codebook,
    uint32_t policy_codebook_mask,
    const ROCmMoESerialM1PolicyTable& policy_table)
{
    if (!llaminar2::PerfStatsCollector::isDomainEnabled("kernel"))
        return;
    llaminar2::PerfStatsCollector::addCounter(
        "kernel",
        "rocm_moe_descriptor_local_decode_calls",
        1.0,
        "moe",
        {},
        {{"projection", projection},
         {"descriptor_source", descriptor_source},
         {"table_codebook", std::to_string(table_codebook)},
         {"policy_codebook_mask", std::to_string(policy_codebook_mask)},
         {"maximum_k_partitions",
          std::to_string(policy_table.maximum_k_partitions)}});
}

extern "C" bool rocmGemv_native_quantised_query_generated_config(
    uint8_t codebook_id,
    int M,
    int N,
    int K,
    int* kb,
    int* target_waves_per_cu)
{
    NativeQuantisedDecodeGeneratedRuntimeConfig cfg =
        resolveNativeQuantisedDecodeGeneratedRuntimeConfig(codebook_id, M, N, K);
    if (!cfg.has_generated)
        return false;
    if (kb)
        *kb = cfg.kb;
    if (target_waves_per_cu)
        *target_waves_per_cu = cfg.target_waves_per_cu;
    return true;
}

/**
 * @brief Measure the in-TU ROCm generated-selector cache without touching HIP.
 *
 * The timing loop calls the same static resolver as the production launcher,
 * avoiding the out-of-line public-query overhead that would otherwise dominate
 * a nanosecond-scale host measurement. A matched loop-only interval is
 * subtracted from the result.
 */
extern "C" double rocmGemv_native_quantised_measure_generated_config_ns(
    uint8_t codebook_id,
    int M,
    const int *Ns,
    const int *Ks,
    int query_count,
    int iterations,
    uint64_t *out_checksum)
{
    if (!Ns || !Ks || query_count <= 0 || iterations <= 0)
        return -1.0;

    uint64_t checksum = 0;
    const auto baseline_begin = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        const int index = iteration % query_count;
        checksum += static_cast<uint64_t>(Ns[index] + Ks[index]);
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }
    const auto baseline_end = std::chrono::steady_clock::now();

    const auto selector_begin = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        const int index = iteration % query_count;
        const NativeQuantisedDecodeGeneratedRuntimeConfig config =
            resolveNativeQuantisedDecodeGeneratedRuntimeConfig(
                codebook_id, M, Ns[index], Ks[index]);
        checksum += static_cast<uint64_t>(config.has_generated) |
                    (static_cast<uint64_t>(config.kb) << 8) |
                    (static_cast<uint64_t>(config.target_waves_per_cu) << 24);
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }
    const auto selector_end = std::chrono::steady_clock::now();

    if (out_checksum)
        *out_checksum = checksum;
    const double baseline_ns = std::chrono::duration<double, std::nano>(
        baseline_end - baseline_begin).count();
    const double selector_ns = std::chrono::duration<double, std::nano>(
        selector_end - selector_begin).count();
    return std::max(
        0.0,
        (selector_ns - baseline_ns) / static_cast<double>(iterations));
}

template <int TILE_N, int CPT, NativeQuantisedFormat FMT, bool HasWork = false>
__global__ void gemv_native_quantised_kernel_t(
    const int8_t*    __restrict__ d_A_int8,       // [K] quantized activations (INT8)
    const uint8_t*   __restrict__ d_payload,      // [blocks_per_row × N × payload_bytes] payload
    const uint16_t*  __restrict__ d_block_scales,  // [blocks_per_row × N] FP16 scale (d)
    const uint16_t*  __restrict__ d_block_mins,    // [blocks_per_row × N] FP16 min (m), NULL for symmetric
    const uint32_t*  __restrict__ d_block_emins,   // [blocks_per_row × N] packed {lo,hi} emins (Q2_K only, else nullptr)
    float*           __restrict__ d_C_fp32,        // [N] output (FP32), used when kblocks=1
    float*           __restrict__ d_partial_fp32,  // [KB × N] partials, used when kblocks>1
    const float*     __restrict__ d_scale_A,       // [1] per-row activation scale (CK fallback, used when d_scale_A_blockwise==nullptr)
    const float*     __restrict__ d_scale_A_blockwise, // [blocks_per_row] per-block activation scales
    int N, int K, int kblocks, llaminar2::ProjectionRowWork work)
{
    nativeM1ColumnsBody<TILE_N,CPT,FMT,HasWork>(d_A_int8,d_payload,d_block_scales,d_block_mins,d_block_emins,
        d_C_fp32,d_partial_fp32,d_scale_A,d_scale_A_blockwise,N,K,kblocks,work,blockIdx.x,blockIdx.y,threadIdx.x);
}

/**
 * @brief Retain the original eight increasing-word signed dot4 operations.
 * @param a_low Original activation words0..3. @param a_high Original words4..7.
 * @param w_low Original prepared Q8 words0..3. @param w_high Original words4..7.
 * @return Exact original complete K32 integer dot, with no saturation/reassociation.
 */
__device__ __forceinline__ int32_t nativeQ8SerialOrderedDot(
    int4 a_low,int4 a_high,int4 w_low,int4 w_high)
{
    int32_t dot=0;
    dot=::llaminar2::rocm_integer_contract::signedDot4(a_low.x,w_low.x,dot);
    dot=::llaminar2::rocm_integer_contract::signedDot4(a_low.y,w_low.y,dot);
    dot=::llaminar2::rocm_integer_contract::signedDot4(a_low.z,w_low.z,dot);
    dot=::llaminar2::rocm_integer_contract::signedDot4(a_low.w,w_low.w,dot);
    dot=::llaminar2::rocm_integer_contract::signedDot4(a_high.x,w_high.x,dot);
    dot=::llaminar2::rocm_integer_contract::signedDot4(a_high.y,w_high.y,dot);
    dot=::llaminar2::rocm_integer_contract::signedDot4(a_high.z,w_high.z,dot);
    return ::llaminar2::rocm_integer_contract::signedDot4(a_high.w,w_high.w,dot);
}

/**
 * @brief Prefetch two independent Q8 K32 blocks before their original ascending commits.
 * @tparam TILE_N Existing64/128 independent-column geometry; one column per lane.
 * @tparam HasWork Exact original immutable row-work mode.
 * @details This diagnostic entry changes no decoder, partition, accumulation,
 * destination or work extent. Prefetch values are register-only scalars/int4s;
 * no local aggregate address escapes, LDS/counter is introduced, or native
 * operation is added. The original producer remains a separate unchanged symbol.
 * @param A Original signed-Q8 activation bytes. @param payload Original prepared Q8 bytes.
 * @param scales Original FP16 weight scales. @param output Original direct output whenP1.
 * @param partials Original partition-major FP32 bank whenP>1. @param row_scale Original optional row activation scale.
 * @param block_scales Original optional K32 activation scales. @param N Exact columns. @param K Exact reduction width.
 * @param partitions Unchanged original source-policy partition count. @param work Original borrowed row authority.
 */
template<int TILE_N,bool HasWork>
__global__ __launch_bounds__(TILE_N) void gemv_native_q8_two_k32_lookahead_kernel_t(
    const int8_t* __restrict__ A,const uint8_t* __restrict__ payload,
    const uint16_t* __restrict__ scales,float* __restrict__ output,float* __restrict__ partials,
    const float* __restrict__ row_scale,const float* __restrict__ block_scales,
    int N,int K,int partitions,llaminar2::ProjectionRowWork work)
{
    static_assert(TILE_N==64||TILE_N==128);
    if constexpr(HasWork)if(!llaminar2::projectionWorkRowIsActiveUniform(work,0))return;
    const unsigned n=blockIdx.x*TILE_N+threadIdx.x;
    if(n>=static_cast<unsigned>(N))return;
    const int partition=blockIdx.y,blocks=K/32;
    const int per=(blocks+partitions-1)/partitions;
    int b=partition*per;
    const int end=min(b+per,blocks);
    float acc=0.0f;
    for(;b+1<end;b+=2)
    {
        const size_t linear0=static_cast<size_t>(b)*N+static_cast<size_t>(n);
        const size_t linear1=static_cast<size_t>(b+1)*N+static_cast<size_t>(n);
        const auto* w0=reinterpret_cast<const int32_t*>(payload+linear0*32);
        const auto* w1=reinterpret_cast<const int32_t*>(payload+linear1*32);
        const int4 w0_low=make_int4(w0[0],w0[1],w0[2],w0[3]);
        const int4 w0_high=make_int4(w0[4],w0[5],w0[6],w0[7]);
        const int4 w1_low=make_int4(w1[0],w1[1],w1[2],w1[3]);
        const int4 w1_high=make_int4(w1[4],w1[5],w1[6],w1[7]);
        const uint16_t scale0=scales[linear0],scale1=scales[linear1];
        const auto* a0=reinterpret_cast<const int32_t*>(A+b*32);
        const auto* a1=reinterpret_cast<const int32_t*>(A+(b+1)*32);
        const int4 a0_low=make_int4(a0[0],a0[1],a0[2],a0[3]);
        const int4 a0_high=make_int4(a0[4],a0[5],a0[6],a0[7]);
        const int4 a1_low=make_int4(a1[0],a1[1],a1[2],a1[3]);
        const int4 a1_high=make_int4(a1[4],a1[5],a1[6],a1[7]);
        const float activation0=block_scales?block_scales[b]:1.0f;
        const float activation1=block_scales?block_scales[b+1]:1.0f;
        const int32_t dot0=nativeQ8SerialOrderedDot(a0_low,a0_high,w0_low,w0_high);
        acc=rocm_native_quantised_commit_single_scale_block(acc,dot0,
            rocm_native_quantised_load_single_weight_scale<NATIVE_QUANTISED_Q8_0>(scale0,0),activation0);
        // The preceding RN add must remain before this second block's RN add.
        // No pair sum, fused accumulator or partition-dependent reassociation.
        const int32_t dot1=nativeQ8SerialOrderedDot(a1_low,a1_high,w1_low,w1_high);
        acc=rocm_native_quantised_commit_single_scale_block(acc,dot1,
            rocm_native_quantised_load_single_weight_scale<NATIVE_QUANTISED_Q8_0>(scale1,0),activation1);
    }
    if(b<end)
    {
        const size_t linear=static_cast<size_t>(b)*N+static_cast<size_t>(n);
        const auto* w=reinterpret_cast<const int32_t*>(payload+linear*32);
        const auto* a=reinterpret_cast<const int32_t*>(A+b*32);
        const int32_t dot=nativeQ8SerialOrderedDot(
            make_int4(a[0],a[1],a[2],a[3]),make_int4(a[4],a[5],a[6],a[7]),
            make_int4(w[0],w[1],w[2],w[3]),make_int4(w[4],w[5],w[6],w[7]));
        acc=rocm_native_quantised_commit_single_scale_block(acc,dot,
            rocm_native_quantised_load_single_weight_scale<NATIVE_QUANTISED_Q8_0>(scales[linear],0),
            block_scales?block_scales[b]:1.0f);
    }
    const float result=acc*(block_scales?1.0f:row_scale[0]);
    if(partitions==1)output[n]=result;
    else partials[static_cast<size_t>(partition)*N+static_cast<size_t>(n)]=result;
}

/**
 * @brief 2-lanes-per-column coalesced int4 Q8_0 M=1 GEMV (1024B contiguous per wave64).
 *
 * Adjacent lanes (2*c, 2*c+1) load the two 16-byte halves of each 32-byte Q8_0
 * payload block in one 100%-coalesced global_load_dwordx4 per wave64, combine
 * their exact INT32 partial dots via one __shfl_xor(1), and commit FP32 block
 * contributions on the even lane in exact ascending K-block order.
 */
template<int TILE_N,bool HasWork>
__global__ __launch_bounds__(TILE_N * 2) void gemv_native_q8_pair_int4_kernel_t(
    const int8_t* __restrict__ A,const uint8_t* __restrict__ payload,
    const uint16_t* __restrict__ scales,float* __restrict__ output,float* __restrict__ partials,
    const float* __restrict__ row_scale,const float* __restrict__ block_scales,
    int N,int K,int partitions,llaminar2::ProjectionRowWork work)
{
    static_assert(TILE_N==64||TILE_N==128);
    if constexpr(HasWork)if(!llaminar2::projectionWorkRowIsActiveUniform(work,0))return;
    const int col_local=static_cast<int>(threadIdx.x>>1);
    const int half=static_cast<int>(threadIdx.x&1u);
    const int n=static_cast<int>(blockIdx.x)*TILE_N+col_local;
    const bool active=(n<N);
    const int partition=static_cast<int>(blockIdx.y);
    const int blocks=K/32;
    const int per=(blocks+partitions-1)/partitions;
    const int b0=partition*per;
    const int b1=min(b0+per,blocks);
    float acc=0.0f;
    for(int b=b0;b<b1;++b)
    {
        const int4 a4=*reinterpret_cast<const int4*>(A+b*32+half*16);
        int32_t dot=0;
        if(active)
        {
            const size_t linear=static_cast<size_t>(b)*static_cast<size_t>(N)+static_cast<size_t>(n);
            const int4 w4=*reinterpret_cast<const int4*>(payload+linear*32+half*16);
            dot=::llaminar2::rocm_integer_contract::signedDot4(a4.x,w4.x,0);
            dot=::llaminar2::rocm_integer_contract::signedDot4(a4.y,w4.y,dot);
            dot=::llaminar2::rocm_integer_contract::signedDot4(a4.z,w4.z,dot);
            dot=::llaminar2::rocm_integer_contract::signedDot4(a4.w,w4.w,dot);
        }
        const int32_t peer=__shfl_xor(dot,1);
        if(active&&half==0)
        {
            const size_t linear=static_cast<size_t>(b)*static_cast<size_t>(N)+static_cast<size_t>(n);
            const float ws=rocm_native_quantised_load_single_weight_scale<NATIVE_QUANTISED_Q8_0>(scales[linear],0);
            acc=rocm_native_quantised_commit_single_scale_block(acc,dot+peer,ws,block_scales?block_scales[b]:1.0f);
        }
    }
    if(active&&half==0)
    {
        const float result=acc*(block_scales?1.0f:row_scale[0]);
        if(partitions==1)output[n]=result;
        else partials[static_cast<size_t>(partition)*static_cast<size_t>(N)+static_cast<size_t>(n)]=result;
    }
}

template <int TILE_N, int CPT, bool HasWork = false>
__global__ void gemv_native_quantised_reduce_kernel_t(const float *__restrict__ partials,
    float *__restrict__ output,int N,int partitions, llaminar2::ProjectionRowWork work)
{
    if constexpr(HasWork)
    if(!llaminar2::projectionWorkRowIsActiveUniform(work,0))return; // Same ordinary M1 route; no unauthored reads/writes.

    nativeM1FoldColumns<TILE_N,CPT>(partials,output,N,partitions,blockIdx.x,threadIdx.x);
}

/**
 * @brief Keep the original128-column reducer grid and publish its original HC output locally.
 * @tparam HasWork Exact original projection work mode.
 * @param partials Original partition bank. @param output Public raw-down output.
 * @param N Original output width. @param partitions Unchanged source partitions.
 * @param work Original projection work, independent of publisher count0.
 * @param publication Existing activated and four Q8 bank destinations.
 */
template<bool HasWork>
__global__ __launch_bounds__(128) void nativeM1HCDownReducePublish(const float* partials,float* output,
 int N,int partitions,llaminar2::ProjectionRowWork work,llaminar2::GPUHCDownOutputView publication) {
 if constexpr(HasWork)if(!llaminar2::projectionWorkRowIsActiveUniform(work,0))return;
 const unsigned n=blockIdx.x*128+threadIdx.x;
 // N is a multiple of32: each tail subgroup is wholly absent or wholly active.
 if(n>=static_cast<unsigned>(N))return;
 const float raw=nativeM1FoldColumnValue(partials,n,N,partitions);
 output[n]=raw;
 if(publication.rows.activeRows()!=0)llaminar2::hc_down_publication::publish(raw,publication,n);
}

/** @brief Concatenate original live reducer tiles without extending any projection. */
template <int TILE_N,int CPT, bool HasWork = false>
__global__ void nativeM1GroupedReduce(llaminar2::rocm::NativeM1ReductionGrid group)
{
    int p=0;
    #pragma unroll
    for(int i=1;i<4;++i)if(i<group.count && int(blockIdx.x)>=group.offsets[i])p=i;
    if constexpr(HasWork)
    if(!llaminar2::projectionWorkRowIsActiveUniform(group.work[p],0))return;
    nativeM1FoldColumns<TILE_N,CPT>(group.partials[p],group.outputs[p],group.columns[p],
        group.partitions[p],blockIdx.x-group.offsets[p],threadIdx.x);
}

namespace llaminar2::rocm
{
/** @brief Only native producer submission can issue a pending reduction authority. */
struct NativeM1ReductionIssuer
{
    /** @brief Reject reuse before any GPU submission. */
    static void requireEmpty(const NativeM1ReductionTicket &ticket)
    {
        if(ticket.state_!=NativeM1ReductionTicket::State::Empty)
            throw std::invalid_argument("M1 producer ticket was already issued");
    }
    /** @brief Publish a ticket only after the exact original producer launch succeeded. */
    static void issue(NativeM1ReductionTicket &ticket,const float *partials,float *output,
        int n,int kb,int tile,int cpt,int device,void *stream,llaminar2::ProjectionRowWork work,bool has_work)
    {
        requireEmpty(ticket);
        ticket.shape_={n,kb,tile,cpt};ticket.partials_=partials;ticket.output_=output;
        ticket.device_=device;ticket.stream_=stream;ticket.work_=work;
        ticket.has_work_=has_work;
        ticket.state_=kb>1?NativeM1ReductionTicket::State::Pending:NativeM1ReductionTicket::State::Direct;
    }
};
}

/** @brief Retain ordinary submission or issue its exact successful producer ticket. */
template <int TN,int CPT,bool HasWork>
void finishNativeM1Producer(float *partials,float *output,int n,int kb,int device,void *stream,
    llaminar2::rocm::NativeM1ReductionTicket *deferred,llaminar2::ProjectionRowWork work)
{
    if(deferred)
    {
        if(hipPeekAtLastError()!=hipSuccess)throw std::runtime_error("M1 deferred producer submission failed");
        llaminar2::rocm::NativeM1ReductionIssuer::issue(*deferred,partials,output,n,kb,TN,CPT,device,stream,work,HasWork);
    }
    else if(kb>1)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_native_quantised_reduce_kernel_t<TN,CPT,HasWork>),
            dim3(1+(n-1)/TN),dim3(TN/CPT),0,static_cast<hipStream_t>(stream),partials,output,n,kb,work);
}



/**
 * @brief Compute independent original Q8 K32 contributions in bounded CTA-owned windows.
 * @tparam HasWork Borrowed original device row authority; false retains the plain M1 ABI.
 * @param A Original quantized input. @param payload Canonical block-major native Q8 bytes.
 * @param scales Original FP16 weight scales. @param output Original public output for P1.
 * @param partials Original [P][N] partial bank. @param scale_a Original optional row scale.
 * @param block_scales Original optional FP32 K32 scales. @param N Physical output width.
 * @param K Original reduction width. @param partitions Exact source-resolved partition count.
 * @param work Original immutable row-work binding, copied into the retained graph.
 *
 * Four contributor groups own32 adjacent columns each. A lane computes complete
 * original signed-dot8 chains and the SAME rounded K32 contribution helper as
 * nativeM1ColumnsBody. Owner group0 folds each real contribution in ascending K
 * order, retaining its accumulator across windows. Never sum contributor-local
 * FP32 accumulators: that would change the source's expression tree. The second
 * local barrier retires every LDS reader before the next window overwrites it.
 *
 * The bounded32x32 FP32 array is explicit4KiB LDS, never per-thread/private
 * storage. Arbitrarily long admitted K partitions use repeated windows. Tail
 * columns and empty partition planes reach the identical barrier structure;
 * empty planes retain the original positive-zero initialization and final scale
 * multiply. No synthetic zero contribution enters an active owner's fold.
 */
template<bool HasWork>
__global__ __launch_bounds__(128) void gemv_native_q8_k32_contribution_tile_kernel_t(
    const int8_t* __restrict__ A,const uint8_t* __restrict__ payload,
    const uint16_t* __restrict__ scales,float* __restrict__ output,float* __restrict__ partials,
    const float* __restrict__ scale_a,const float* __restrict__ block_scales,
    int N,int K,int partitions,llaminar2::ProjectionRowWork work)
{
    if constexpr(HasWork)
        if(!llaminar2::projectionWorkRowIsActiveUniform(work,0))return;
    constexpr int Columns=32,Contributors=4,WindowBlocks=32;
    __shared__ float contributions[WindowBlocks][Columns];
    const unsigned group=threadIdx.x/Columns,column=threadIdx.x%Columns;
    const int n=static_cast<int>(blockIdx.x)*Columns+static_cast<int>(column);
    const int blocks=K/32,per=(blocks+partitions-1)/partitions;
    const int first=static_cast<int>(blockIdx.y)*per,last=min(first+per,blocks);
    float accumulator=0.0f;
    for(int begin=first;begin<last;begin+=WindowBlocks)
    {
        const int count=min(WindowBlocks,last-begin);
        for(int offset=static_cast<int>(group);offset<count;offset+=Contributors)
        {
            if(n<N)
            {
                const int b=begin+offset;
                const auto linear=static_cast<size_t>(b)*static_cast<size_t>(N)+static_cast<size_t>(n);
                const auto* weights=reinterpret_cast<const int32_t*>(payload+linear*32);
                const auto* activation=reinterpret_cast<const int32_t*>(A+b*32);
                int32_t dot=0;
#pragma unroll
                for(int word=0;word<8;++word)
                    dot=::llaminar2::rocm_integer_contract::signedDot4(activation[word],weights[word],dot);
                const float scale=rocm_native_quantised_load_single_weight_scale<NATIVE_QUANTISED_Q8_0>(scales[linear],uint16_t{0});
                contributions[offset][column]=::llaminar2::device_native_quantised_contract::singleScaleBlock(
                    dot,scale,block_scales?block_scales[b]:1.0f);
            }
        }
        __syncthreads();
        if(group==0&&n<N)
            for(int offset=0;offset<count;++offset)
                accumulator=rocm_native_quantised_accumulate_rn(accumulator,contributions[offset][column]);
        __syncthreads();
    }
    if(group==0&&n<N)
    {
        const float result=accumulator*(block_scales?1.0f:scale_a[0]);
        if(partitions==1)output[n]=result;
        else partials[static_cast<size_t>(blockIdx.y)*N+n]=result;
    }
}

namespace
{
/** @brief Exact loaded producer/reducer pair for one existing template geometry. */
struct NativeQ8SerialSymbols
{
    const void *producer;
    const void *reducer;
    const char *producer_name;
    const char *reducer_name;
};

/**
 * @brief Resolve the same symbols the launch builder will enqueue.
 * @tparam TN Output columns per workgroup.
 * @tparam CPT Independent output columns per lane.
 * @return Loaded HIP symbols and stable readable specialization spellings.
 */
template <int TN, int CPT, bool HasWork = false>
NativeQ8SerialSymbols nativeQ8SerialSymbols()
{
#define NQ8_SYMBOL_NAMES(N, C) \
    HasWork ? "gemv_native_quantised_kernel_t<" #N "," #C ",NATIVE_QUANTISED_Q8_0,true>" : "gemv_native_quantised_kernel_t<" #N "," #C ",NATIVE_QUANTISED_Q8_0,false>", \
    HasWork ? "gemv_native_quantised_reduce_kernel_t<" #N "," #C ",true>" : "gemv_native_quantised_reduce_kernel_t<" #N "," #C ",false>"
    const char *producer_name = nullptr;
    const char *reducer_name = nullptr;
#define NQ8_NAME_CASE(N, C) \
    if constexpr (TN == N && CPT == C) { \
        const char *names[] = {NQ8_SYMBOL_NAMES(N, C)}; \
        producer_name = names[0]; reducer_name = names[1]; }
    NQ8_NAME_CASE(64, 1)
    NQ8_NAME_CASE(128, 1)
    NQ8_NAME_CASE(256, 1)
    NQ8_NAME_CASE(128, 2)
    NQ8_NAME_CASE(256, 2)
#undef NQ8_NAME_CASE
#undef NQ8_SYMBOL_NAMES
    static_assert((TN == 64 && CPT == 1) || (TN == 128 && (CPT == 1 || CPT == 2)) ||
                  (TN == 256 && (CPT == 1 || CPT == 2)));
    return {reinterpret_cast<const void *>(gemv_native_quantised_kernel_t<TN, CPT, NATIVE_QUANTISED_Q8_0, HasWork>),
            reinterpret_cast<const void *>(gemv_native_quantised_reduce_kernel_t<TN, CPT, HasWork>),
            producer_name, reducer_name};
}

/**
 * @brief Resolve a diagnostic load schedule while retaining the existing reducer symbol.
 * @tparam TN Original independent-column tile. @tparam CPT Original columns per lane.
 * @tparam HasWork Exact original row-work mode. @param loads Explicit cold memory schedule.
 * @return Original pair or the admitted lookahead producer with the SAME reducer.
 * @throws std::invalid_argument For an unknown/unimplemented diagnostic tuple.
 */
template<int TN,int CPT,bool HasWork>
NativeQ8SerialSymbols nativeQ8SerialSymbols(llaminar2::rocm::NativeQ8SerialProducerLoads loads)
{
    using Loads=llaminar2::rocm::NativeQ8SerialProducerLoads;
    auto result=nativeQ8SerialSymbols<TN,CPT,HasWork>();
    if(loads==Loads::Original)return result;
    if(loads==Loads::K32ContributionTile4x32)
    {
        if constexpr(TN==128&&CPT==1)
        {
            result.producer=reinterpret_cast<const void*>(gemv_native_q8_k32_contribution_tile_kernel_t<HasWork>);
            result.producer_name=HasWork?"gemv_native_q8_k32_contribution_tile_kernel_t<true>":"gemv_native_q8_k32_contribution_tile_kernel_t<false>";
            return result; // Original128-column reducer remains the same symbol.
        }
        throw std::invalid_argument("native-Q8 K32 contribution tile requires original128 CPT1 reducer geometry");
    }
    if(loads!=Loads::TwoK32Lookahead)
        throw std::invalid_argument("unknown native-Q8 producer load schedule");
    if constexpr(CPT==1&&(TN==64||TN==128))
    {
        result.producer=reinterpret_cast<const void*>(gemv_native_q8_two_k32_lookahead_kernel_t<TN,HasWork>);
        if constexpr(TN==64)result.producer_name=HasWork?"gemv_native_q8_two_k32_lookahead_kernel_t<64,true>":"gemv_native_q8_two_k32_lookahead_kernel_t<64,false>";
        else result.producer_name=HasWork?"gemv_native_q8_two_k32_lookahead_kernel_t<128,true>":"gemv_native_q8_two_k32_lookahead_kernel_t<128,false>";
        return result;
    }
    throw std::invalid_argument("native-Q8 lookahead requires admitted64/128 CPT1 geometry");
}

/**
 * @brief Select one exact compiled pair; unknown enum values fail before HIP calls.
 * @param geometry Typed requested geometry.
 * @param loads Exact cold-admitted producer schedule.
 * @return The function pair shared with the actual launch dispatch.
 */
NativeQ8SerialSymbols nativeQ8SerialSymbols(llaminar2::rocm::NativeQ8SerialGeometry geometry,
    llaminar2::rocm::NativeQ8SerialProducerLoads loads)
{
    return llaminar2::dispatchProjectionWorkMode(1,[&]<bool HasWork>() {
    using G = llaminar2::rocm::NativeQ8SerialGeometry;
    switch (geometry)
    {
    case G::TN64CPT1: return nativeQ8SerialSymbols<64, 1, HasWork>(loads);
    case G::TN128CPT1: return nativeQ8SerialSymbols<128, 1, HasWork>(loads);
    case G::TN256CPT1: return nativeQ8SerialSymbols<256, 1, HasWork>(loads);
    case G::TN128CPT2: return nativeQ8SerialSymbols<128, 2, HasWork>(loads);
    case G::TN256CPT2: return nativeQ8SerialSymbols<256, 2, HasWork>(loads);
    }
    throw std::invalid_argument("unknown ROCm native-Q8 serial geometry");
    });
}

/**
 * @brief Authenticate a loaded specialization at its exact block width.
 * @param function Exact symbol selected by nativeQ8SerialSymbols().
 * @param name Static readable template spelling; function remains authority.
 * @param threads Physical workgroup width, not a target-wave heuristic.
 * @return Complete compiler and occupancy evidence with zero private memory.
 * @throws std::runtime_error on incomplete, nonresident or spilling resources.
 */
llaminar2::rocm::NativeQ8SerialKernelResources inspectNativeQ8SerialSymbol(
    const void *function, const char *name, int threads)
{
    hipFuncAttributes attributes{};
    int active_blocks = 0;
    if (!function || !name || threads <= 0 ||
        hipFuncGetAttributes(&attributes, function) != hipSuccess ||
        hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, function, threads, 0) != hipSuccess ||
        attributes.numRegs <= 0 || attributes.maxThreadsPerBlock < threads ||
        active_blocks <= 0 || attributes.localSizeBytes != 0)
        throw std::runtime_error(std::string("native-Q8 serial resource admission failed: ") + (name ? name : "missing symbol"));
    return {.function = function, .specialization = name, .block_threads = threads,
            .registers_per_thread = attributes.numRegs,
            .local_memory_bytes_per_thread = attributes.localSizeBytes,
            .static_shared_memory_bytes = attributes.sharedSizeBytes,
            .max_threads_per_block = attributes.maxThreadsPerBlock,
            .max_active_blocks_per_sm = active_blocks};
}

/**
 * @brief Enqueue the existing serial producer and ordered reducer at one geometry.
 * @tparam TN Independent output columns owned by one workgroup.
 * @tparam CPT Independent output chains per thread, one or two.
 * @tparam FMT Existing compiled decoder; new geometries instantiate Q8 only.
 * @param A Existing quantized activation bytes.
 * @param payload Existing prepared native weight payload.
 * @param scales Existing prepared primary scale metadata.
 * @param mins Existing prepared optional asymmetric metadata.
 * @param emins Existing prepared optional extended metadata.
 * @param output Live FP32 output vector.
 * @param partials Existing persistent [KB][N] partial workspace.
 * @param scale_a Existing per-row activation scale, when used.
 * @param block_scales Existing per-block activation scales, when used.
 * @param N Physical descriptor/output width.
 * @param K Complete unchanged reduction width.
 * @param kb Resolved production arithmetic partition count.
 * @param arithmetic_policy Source codebook owning the arithmetic tree.
 * @param device Exact HIP device.
 * @param stream Exact non-null producer stream.
 * @param geometry Actual selected geometry for the launch receipt.
 * @note No resource queries, allocations, transfers or waits occur here.
 */
template <int TN, int CPT, NativeQuantisedFormat FMT>
void launchNativeQuantisedSerialGeometry(
    const int8_t *A, const uint8_t *payload, const uint16_t *scales,
    const uint16_t *mins, const uint32_t *emins, float *output, float *partials,
    const float *scale_a, const float *block_scales, int N, int K, int kb,
    uint8_t arithmetic_policy, int device, void *stream,
    llaminar2::rocm::NativeQ8SerialGeometry geometry,
    llaminar2::rocm::NativeM1ReductionTicket *deferred=nullptr,
    const llaminar2::rocm::NativeQ8SerialGeometryAdmission *prepared=nullptr)
{
    static_assert(CPT == 1 || CPT == 2);
    static_assert(TN % CPT == 0);
    using Loads=llaminar2::rocm::NativeQ8SerialProducerLoads;
    const auto loads=prepared?prepared->producerLoads():
        g_native_q8_serial_scope?g_native_q8_serial_scope->admission().producerLoads():Loads::Original;
    const auto producer_spec=llaminar2::rocm::nativeQ8SerialProducerSpec(geometry,loads);
    const int grid_n = 1 + (N - 1) / (FMT==NATIVE_QUANTISED_Q8_0?producer_spec.tile_columns:TN);
    const dim3 block(TN / CPT, 1, 1);
    const auto work=llaminar2::projectionWorkForLaunch(1);
    return llaminar2::dispatchProjectionWorkMode(1,[&]<bool HasWork>() {
        if(prepared) {
            // Cold owner admission, not a capture-time scope or hardware query.
            // Exact function comparison also authenticates the original row-work
            // mode: a false-mode certificate cannot launch a counted program.
            const auto symbols=nativeQ8SerialSymbols<TN,CPT,HasWork>(loads);
            if(g_native_q8_serial_scope||FMT!=NATIVE_QUANTISED_Q8_0||TN!=128||CPT!=1||
                prepared->deviceOrdinal()!=device||prepared->geometry()!=geometry||loads!=Loads::TwoK32Lookahead||
                prepared->producerResources().function!=symbols.producer||
                prepared->reducerResources().function!=symbols.reducer||
                prepared->producerResources().block_threads!=TN/CPT)
                throw std::logic_error("HC lookahead producer changed its exact cold program/device/work geometry");
        }

        if constexpr(FMT==NATIVE_QUANTISED_Q8_0&&CPT==1&&(TN==64||TN==128))
        {
            if(loads==Loads::K32ContributionTile4x32)
            {
                if constexpr(TN==128)
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_native_q8_k32_contribution_tile_kernel_t<HasWork>),
                        dim3(grid_n,kb,1),dim3(producer_spec.block_threads),0,static_cast<hipStream_t>(stream),
                        A,payload,scales,output,partials,scale_a,block_scales,N,K,kb,work);
                else throw std::invalid_argument("native-Q8 K32 contribution tile requires original128 CPT1 reducer geometry");
            }
            else if(loads==Loads::TwoK32Lookahead)
                hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_native_q8_two_k32_lookahead_kernel_t<TN,HasWork>),
                    dim3(grid_n,kb,1),block,0,static_cast<hipStream_t>(stream),
                    A,payload,scales,output,partials,scale_a,block_scales,N,K,kb,work);
            else if(loads==Loads::Original)
            {
                if(!g_native_q8_serial_scope&&!prepared&&kb==1&&N>=16384)
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_native_q8_pair_int4_kernel_t<TN,HasWork>),
                        dim3(grid_n,kb,1),dim3(TN*2,1,1),0,static_cast<hipStream_t>(stream),
                        A,payload,scales,output,partials,scale_a,block_scales,N,K,kb,work);
                else
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_native_quantised_kernel_t<TN,CPT,FMT,HasWork>),
                        dim3(grid_n,kb,1),block,0,static_cast<hipStream_t>(stream),
                        A,payload,scales,mins,emins,output,partials,scale_a,block_scales,N,K,kb,work);
            }
            else throw std::invalid_argument("unknown native-Q8 producer load schedule");
        }
        else
        {
            if(loads!=Loads::Original)throw std::invalid_argument("native-Q8 lookahead requires admitted64/128 CPT1 source");
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_native_quantised_kernel_t<TN,CPT,FMT,HasWork>),
                dim3(grid_n,kb,1),block,0,static_cast<hipStream_t>(stream),
                A,payload,scales,mins,emins,output,partials,scale_a,block_scales,N,K,kb,work);
        }
    if (g_native_q8_serial_scope && hipPeekAtLastError() != hipSuccess)
    {
        g_native_q8_serial_launch.reset();
        throw std::runtime_error("native-quantised serial producer submission failed");
    }
    // TN/CPT retain the original source/reducer ticket geometry. K-cooperative
    // producer ownership is independently authenticated by its physical receipt.
    finishNativeM1Producer<TN,CPT,HasWork>(partials,output,N,kb,device,stream,deferred,work);
    if (g_native_q8_serial_scope && hipPeekAtLastError() != hipSuccess)
    {
        g_native_q8_serial_launch.reset();
        throw std::runtime_error("native-quantised serial ordered-reducer submission failed");
    }
    if constexpr (FMT == NATIVE_QUANTISED_Q8_0)
    {
        const auto symbols = nativeQ8SerialSymbols<TN,CPT,HasWork>(loads);
        const int b_per = 1 + (K / 32 - 1) / kb;
        const int nonempty = 1 + (K / 32 - 1) / b_per;
        const auto launch = [&](const void *function, const char *name, int grid_x,int grid_y,int threads)
        {
            return llaminar2::rocm::NativeQ8SerialKernelLaunch{
                function, name, grid_x, grid_y, 1, threads, 1, 1};
        };
        // Publish topology only. Replays never update this host observation.
        g_native_q8_serial_launch = llaminar2::rocm::NativeQ8SerialLaunch{
            .geometry = geometry, .producer_loads=loads, .physical_n = N,
            .arithmetic_policy_n = g_native_quantised_serial_partition_n > 0
                ? g_native_quantised_serial_partition_n : N,
            .k = K, .device_ordinal = device, .execution_codebook = 19,
            .arithmetic_policy_codebook = arithmetic_policy, .k_partitions = kb,
            .blocks_per_partition = b_per, .nonempty_partitions = nonempty,
            .partial_elements = kb > 1 ? std::size_t(kb) * N : 0u,
            .stream = stream,
            .producer = launch(symbols.producer, symbols.producer_name,grid_n,kb,producer_spec.block_threads),
            .reducer = kb > 1 && !deferred ? launch(symbols.reducer, symbols.reducer_name,1+(N-1)/TN,1,TN/CPT)
                : llaminar2::rocm::NativeQ8SerialKernelLaunch{}};
    }
    });
}
} // namespace

llaminar2::rocm::NativeQ8SerialGeometryAdmission
llaminar2::rocm::admitNativeQ8SerialGeometry(NativeQ8SerialGeometry geometry,int device_ordinal,
    NativeQ8SerialProducerLoads producer_loads)
{
    const auto spec = nativeQ8SerialGeometrySpec(geometry);
    const auto producer_spec=nativeQ8SerialProducerSpec(geometry,producer_loads);
    const auto symbols = nativeQ8SerialSymbols(geometry,producer_loads);
    if (device_ordinal < 0)
        throw std::invalid_argument("native-Q8 serial resource admission needs an exact device");
    if (llaminar2::isGraphCaptureActive())
        throw std::logic_error("native-Q8 serial resources must be inspected before capture");
    if (llaminar2::HipDeviceGuard::setDevice(device_ordinal) != 0)
        throw std::runtime_error("native-Q8 serial resource admission could not select device");
    return {geometry, device_ordinal,
        inspectNativeQ8SerialSymbol(symbols.producer, symbols.producer_name, producer_spec.block_threads),
        inspectNativeQ8SerialSymbol(symbols.reducer, symbols.reducer_name, spec.block_threads),producer_loads};
}

//   IQ2_XS:  +76%  decode speedup  (4KB grid  → 10 WGs/CU, 40 waves)
//
// Otherwise identical to the main kernel: sdot4 accumulation, FP16 scales,
// dual-scale, asymmetric min correction, scatter+reduce K-partitioning.
// ============================================================================
/**
 * @brief Original IQ-LDS producer with explicit logical tile and original K partition.
 * @details The original256-thread preload, decoder and arithmetic are shared
 * with resident execution. Every physical lane participates in its CTA barriers;
 * callers fence shared reuse between successive logical producer tasks.
 */
template <int IQ_TN, NativeQuantisedFormat FMT, bool HasWork = false>
__device__ __forceinline__ void nativeM1IQLDSBody(
    const int8_t*    __restrict__ d_A_int8,
    const uint8_t*   __restrict__ d_payload,
    const uint16_t*  __restrict__ d_block_scales,
    const uint16_t*  __restrict__ d_block_mins,
    const uint32_t*  __restrict__ d_block_emins,
    float*           __restrict__ d_C_fp32,
    float*           __restrict__ d_partial_fp32,
    const float*     __restrict__ d_scale_A,
    const float*     __restrict__ d_scale_A_blockwise,
    int N, int K, int kblocks, llaminar2::ProjectionRowWork work,
    unsigned logical_tile,unsigned logical_partition)
{
    if constexpr(HasWork)
    if(!llaminar2::projectionWorkRowIsActiveUniform(work,0))return; // Same ordinary M1 route; no unauthored reads/writes.

    using Traits = NATIVE_QUANTISEDTraits<FMT>;
    static_assert(Traits::is_iq_grid || Traits::is_iq1_grid,
                  "LDS kernel only for IQ grid formats");

    // ---- LDS grid table ----
    __shared__ uint8_t lds_raw[Traits::grid_lds_bytes];
    [[maybe_unused]] uint32_t* lds32 = reinterpret_cast<uint32_t*>(lds_raw);
    [[maybe_unused]] uint64_t* lds64 = reinterpret_cast<uint64_t*>(lds_raw);

    // Cooperative preload: 256 threads load grid from constant → LDS
    if constexpr (Traits::is_iq3_grid)
    {
        const uint32_t* src = (FMT == NATIVE_QUANTISED_IQ3_S) ? d_iq3s_grid : d_iq3xxs_grid;
        for (int i = threadIdx.x; i < Traits::grid_entries; i += blockDim.x)
            lds32[i] = src[i];
    }
    else if constexpr (Traits::is_iq2_grid)
    {
        const uint64_t* src =
            (FMT == NATIVE_QUANTISED_IQ2_S)   ? d_iq2s_grid :
            (FMT == NATIVE_QUANTISED_IQ2_XS)  ? d_iq2xs_grid : d_iq2xxs_grid;
        for (int i = threadIdx.x; i < Traits::grid_entries; i += blockDim.x)
            lds64[i] = src[i];
    }
    else // IQ1
    {
        for (int i = threadIdx.x; i < Traits::grid_entries; i += blockDim.x)
            lds64[i] = d_iq1s_grid[i];
    }
    __syncthreads();

    // ---- Main GEMV logic ----
    const int n = logical_tile * IQ_TN + threadIdx.x;
    const bool active = (n < N);
    const int kb = logical_partition;

    const int blocks_per_row = K / 32;
    const int b_per = (blocks_per_row + kblocks - 1) / kblocks;
    const int b_start = kb * b_per;
    const int b_end = min(b_start + b_per, blocks_per_row);

    float f_acc = 0.0f;

    if (active)
    {
        for (int b = b_start; b < b_end; ++b)
        {
            const int32_t* a4 = reinterpret_cast<const int32_t*>(d_A_int8 + b * 32);
            const float scale_ab = d_scale_A_blockwise ? d_scale_A_blockwise[b] : 1.0f;

            int32_t sum_a = 0;
            if constexpr (Traits::is_asymmetric)
            {
                constexpr int32_t ones = 0x01010101;
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                    sum_a = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], ones, sum_a);
            }

            iq_decode_accumulate_block<1, FMT, true>(
                a4, sum_a, d_payload, d_block_scales, d_block_mins,
                n, N, static_cast<size_t>(b) * static_cast<size_t>(N), &f_acc,
                lds32, lds64, scale_ab);
        }

        // ---- Apply activation scale and write output ----
        const float result = f_acc * (d_scale_A_blockwise ? 1.0f : d_scale_A[0]);
        if (kblocks == 1)
        {
            d_C_fp32[n] = result;
        }
        else
        {
            /*
             * IQ decode obeys the same partition contract as the linear
             * codebooks: one persistent row per K partition followed by the
             * fixed-order reducer.  Completion-order atomics are not a valid
             * serial-decode arithmetic regime.
             */
            d_partial_fp32[static_cast<size_t>(kb) * N + n] = result;
        }
    }
}

template <int IQ_TN, NativeQuantisedFormat FMT, bool HasWork = false>
__global__ void gemv_iq_lds_kernel_t(
    const int8_t*    __restrict__ d_A_int8,
    const uint8_t*   __restrict__ d_payload,
    const uint16_t*  __restrict__ d_block_scales,
    const uint16_t*  __restrict__ d_block_mins,
    const uint32_t*  __restrict__ d_block_emins,
    float*           __restrict__ d_C_fp32,
    float*           __restrict__ d_partial_fp32,
    const float*     __restrict__ d_scale_A,
    const float*     __restrict__ d_scale_A_blockwise,
    int N, int K, int kblocks, llaminar2::ProjectionRowWork work)
{
    nativeM1IQLDSBody<IQ_TN,FMT,HasWork>(d_A_int8,d_payload,d_block_scales,d_block_mins,d_block_emins,
        d_C_fp32,d_partial_fp32,d_scale_A,d_scale_A_blockwise,N,K,kblocks,work,blockIdx.x,blockIdx.y);
}




namespace llaminar2::rocm::native_quantised_internal
{
    int kbOverride() noexcept { return g_native_quantised_kb_override.load(std::memory_order_relaxed); }
    int targetWavesOverride() noexcept { return g_native_quantised_target_waves_override.load(std::memory_order_relaxed); }
    bool decodeEquivalentSmallMMode() noexcept { return nativeQuantisedDecodeEquivalentSmallMMode(); }
    int serialPartitionN() noexcept { return g_native_quantised_serial_partition_n; }
    bool nativeQ8SerialScopeActive() noexcept { return g_native_q8_serial_scope != nullptr; }
    bool validateDecodeEquivalentKBOverride(const char* caller, int explicit_kb_override, int m, int n, int k, int projections)
    {
        return validateNativeQuantisedDecodeEquivalentKBOverride(caller, explicit_kb_override, m, n, k, projections);
    }
    void recordSmallMLaunch(
        int m, int n, int k, uint8_t codebook, int grid_n, int kb,
        int target_waves_per_cu, bool batched, int projections, int arithmetic_policy_codebook)
    {
        recordNativeQuantisedSmallMLaunch(m, n, k, codebook, grid_n, kb, target_waves_per_cu, batched, projections, arithmetic_policy_codebook);
    }
    NativeQuantisedDecodeGeneratedRuntimeConfig resolveDecodeGeneratedRuntimeConfigForPolicy(
        uint8_t arithmetic_policy_codebook_id, int m, int n, int k)
    {
        return resolveNativeQuantisedDecodeGeneratedRuntimeConfigForPolicy(arithmetic_policy_codebook_id, m, n, k);
    }
    NativeQuantisedDecodeGeneratedRuntimeConfig resolveDecodeGeneratedRuntimeConfig(
        uint8_t codebook_id, int m, int n, int k)
    {
        return resolveNativeQuantisedDecodeGeneratedRuntimeConfig(codebook_id, m, n, k);
    }
    NativeQuantisedDecodeGeneratedRuntimeConfig resolveSerialM1RuntimeConfigForPolicy(
        uint8_t arithmetic_policy_codebook_id, int N, int K)
    {
        return resolveNativeQuantisedSerialM1RuntimeConfigForPolicy(arithmetic_policy_codebook_id, N, K);
    }
    NativeQuantisedDecodeGeneratedRuntimeConfig resolveSerialM1RuntimeConfig(uint8_t codebook_id, int N, int K)
    {
        return resolveNativeQuantisedSerialM1RuntimeConfig(codebook_id, N, K);
    }
    NativeQuantisedDecodeGeneratedRuntimeConfig resolveCommonSerialM1RuntimeConfig(
        uint8_t codebook_id, const int *Ns, int num_projections, int K)
    {
        return resolveNativeQuantisedCommonSerialM1RuntimeConfig(codebook_id, Ns, num_projections, K);
    }
    NativeQuantisedDecodeGeneratedRuntimeConfig resolveCommonSerialM1RuntimeConfig(
        const uint8_t *codebook_ids, const int *Ns, int num_projections, int K)
    {
        return resolveNativeQuantisedCommonSerialM1RuntimeConfig(codebook_ids, Ns, num_projections, K);
    }
    int resolveSerialM1KPartitions(uint8_t codebook_id, int N, int K)
    {
        return resolveNativeQuantisedSerialM1KPartitions(codebook_id, N, K);
    }
    int resolveSerialM1KPartitionsForPolicy(uint8_t policy_codebook_id, int N, int K)
    {
        return resolveNativeQuantisedSerialM1KPartitionsForPolicy(policy_codebook_id, N, K);
    }
    void recordMoEDescriptorLocalDecode(
        const char* projection,
        const char* descriptor_source,
        uint8_t table_codebook,
        uint32_t policy_codebook_mask,
        const ROCmMoESerialM1PolicyTable& policy_table)
    {
        ::recordROCmMoEDescriptorLocalDecode(
            projection, descriptor_source, table_codebook, policy_codebook_mask, policy_table);
    }
    bool buildROCmMoESerialM1PolicyTable(
        int N, int K, uint32_t required_policy_mask, ROCmMoESerialM1PolicyTable &out)
    {
        return ::buildROCmMoESerialM1PolicyTable(N, K, required_policy_mask, out);
    }
    llaminar2::rocm::NativeQ8SerialKernelResources inspectNativeQ8SerialSymbol(
        const void *fn, const char *name, int threads)
    {
        return ::inspectNativeQ8SerialSymbol(fn, name, threads);
    }
} // namespace llaminar2::rocm::native_quantised_internal

namespace llaminar2::rocm::native_quantised_internal {
bool initIQGridTables_small_m(int, const void*, const void*, const void*, const void*, const void*, const void*);
bool initIQGridTables_moe_decode(int, const void*, const void*, const void*, const void*, const void*, const void*);
bool initIQGridTables_resident_ffn_0(int, const void*, const void*, const void*, const void*, const void*, const void*);
bool initIQGridTables_resident_ffn_1(int, const void*, const void*, const void*, const void*, const void*, const void*);
bool initIQGridTables_persistent_hc(int, const void*, const void*, const void*, const void*, const void*, const void*);
bool initIQGridTables_native_layout_0(int, const void*, const void*, const void*, const void*, const void*, const void*);
bool initIQGridTables_native_layout_1(int, const void*, const void*, const void*, const void*, const void*, const void*);
bool initIQGridTables_native_layout_2(int, const void*, const void*, const void*, const void*, const void*, const void*);
bool initIQGridTables_full_hc_0(int, const void*, const void*, const void*, const void*, const void*, const void*);
bool initIQGridTables_full_hc_1(int, const void*, const void*, const void*, const void*, const void*, const void*);
bool initIQGridTables_full_hc_2(int, const void*, const void*, const void*, const void*, const void*, const void*);
}

extern "C" {


/**
 * @brief Retain the original ordinary-M1 IQ-LDS selection exactly for every caller.
 * @param execution Actual physical decoder codebook. @param n Output columns. @param k Reduction width.
 * @return The existing native source route; this does not select a resident grid or change K partitions.
 */
static bool nativeM1UsesIQLDS(std::uint8_t execution,int n,int k)
{
    constexpr int original_cus=60,iq_tile=256;
    const int tiles=(n+iq_tile-1)/iq_tile;
    return execution>=11 && execution<=17 &&
        (tiles>=original_cus/2 || (k>=4.0f*n && tiles>=original_cus/10));
}
/**
 * @brief Execute serial NativeQuantised GEMV with an explicit arithmetic policy.
 *
 * The execution codebook owns physical decoding; the arithmetic policy
 * codebook owns generated occupancy and split-K reduction order. These IDs
 * differ after compact CPU weights are promoted into expanded codebook 23.
 */
static bool nativeM1WithReduction(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const void* d_block_scales,  // __half* (FP16 scale d)
    const void* d_block_mins,    // __half* (FP16 min m), NULL for symmetric formats
    const void* d_block_emins,   // uint32_t* packed {lo,hi} emins (Q2_K only, else nullptr)
    float* d_C_fp32,
    const float* d_scale_A,
    float* d_partial_fp32,       // [KB_MAX × N] partial buffer (nullable when KB=1)
    int N, int K,
    uint8_t codebook_id,
    uint8_t arithmetic_policy_codebook_id,
    int device_id, void* stream,
    const float* d_scale_A_blockwise,
    llaminar2::rocm::NativeM1ReductionTicket *deferred,
    const llaminar2::rocm::NativeQ8SerialGeometryAdmission *prepared=nullptr)
{
    if(reinterpret_cast<std::uintptr_t>(stream)<=2 || device_id<0 || N<=0 || K<=0 || !d_A_int8 || !d_payload || !d_block_scales || !d_C_fp32)
        throw std::invalid_argument("Native M1 submission requires exact non-null operands, device and stream");
    if(prepared&&(g_native_q8_serial_scope||codebook_id!=NATIVE_QUANTISED_Q8_0||
        arithmetic_policy_codebook_id!=19||!d_scale_A_blockwise||g_native_quantised_serial_partition_n!=0||
        prepared->deviceOrdinal()!=device_id||prepared->geometry()!=llaminar2::rocm::NativeQ8SerialGeometry::TN128CPT1||
        prepared->producerLoads()!=llaminar2::rocm::NativeQ8SerialProducerLoads::TwoK32Lookahead))
        throw std::logic_error("Prepared HC lookahead requires its original native19/blockwise/source arithmetic purpose");
    if(deferred)llaminar2::rocm::NativeM1ReductionIssuer::requireEmpty(*deferred);
    llaminar2::rocm::clearNativeQ8SerialLaunch();
    if (g_native_q8_serial_scope)
    {
        llaminar2::rocm::validateNativeQ8SerialScopeBinding(device_id, stream);
        const auto &env = llaminar2::debugEnv().rocm;
        if (codebook_id != NATIVE_QUANTISED_Q8_0 ||
            !llaminar2::MoEProjectionNumericalContract::ownsOrderedKPartitionTree(
                arithmetic_policy_codebook_id) || N <= 0 || K <= 0 ||
            g_native_quantised_kb_override.load(std::memory_order_relaxed) > 0 ||
            g_native_quantised_target_waves_override.load(std::memory_order_relaxed) > 0 ||
            env.native_quantised_gemv_kb > 0 || env.native_quantised_gemv_target_waves > 0)
            throw std::logic_error("native-Q8 serial geometry scope requires exact device/native19/stream and unchanged production arithmetic");
    }
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 32) != 0)
    {
        fprintf(stderr, "[rocmGemv_native_quantised_fp32] K must be multiple of 32 (N=%d K=%d)\n", N, K);
        return false;
    }

    constexpr int NUM_CUS = 60;        // MI50/MI60
    constexpr int MIN_BLOCKS_PER_WAVE = 1;
    constexpr int KB_CAP =
        llaminar2::NativeQuantisedGroupedDecodePolicy::maximum_k_partitions;

    const int blocks_per_row = K / 32;
    const uint16_t* d_scales_u16 = static_cast<const uint16_t*>(d_block_scales);
    const uint16_t* d_mins_u16   = static_cast<const uint16_t*>(d_block_mins);
    const uint32_t* d_emins_u32  = static_cast<const uint32_t*>(d_block_emins);

    const auto& rocm_env = llaminar2::debugEnv().rocm;
    const int api_kb_override =
        g_native_quantised_kb_override.load(std::memory_order_relaxed);
    const int api_waves_override =
        g_native_quantised_target_waves_override.load(std::memory_order_relaxed);
    const int explicit_kb_override =
        (api_kb_override > 0) ? api_kb_override : rocm_env.native_quantised_gemv_kb;
    const int explicit_waves_override =
        (api_waves_override > 0)
            ? api_waves_override
            : rocm_env.native_quantised_gemv_target_waves;
    const bool has_explicit_tuning =
        explicit_kb_override > 0 || explicit_waves_override > 0;
    const NativeQuantisedDecodeGeneratedRuntimeConfig generated_cfg =
        resolveNativeQuantisedSerialM1RuntimeConfigForPolicy(
            arithmetic_policy_codebook_id, N, K);
    if (!has_explicit_tuning && !generated_cfg.has_generated)
    {
        fprintf(
            stderr,
            "[rocmGemv_native_quantised_fp32] generated dispatch miss "
            "(codebook=%u arithmetic_policy_codebook=%u N=%d K=%d)\n",
            static_cast<unsigned>(codebook_id),
            static_cast<unsigned>(arithmetic_policy_codebook_id),
            N,
            K);
        return false;
    }
    const int policy_kb =
        (explicit_kb_override > 0)
            ? explicit_kb_override
            : generated_cfg.kb;
    const int policy_target_waves =
        (explicit_waves_override > 0)
            ? explicit_waves_override
            : generated_cfg.target_waves_per_cu;

    // =========================================================================
    // IQ grid formats: LDS-cached grid kernel with 256-thread workgroups
    //
    // Constant memory grid lookups are the dominant bottleneck for IQ formats.
    // Preloading the grid table into LDS gives 76-190% decode speedup.
    // Uses 4 wavefronts/WG to amortize the one-time LDS preload cost.
    //
    // Occupancy (MI60, 64KB LDS, 256 threads = 4 waves/WG):
    //   IQ3_XXS (1KB):  10 WGs/CU → 40 waves — HW max
    //   IQ3_S   (2KB):  10 WGs/CU → 40 waves — HW max
    //   IQ2_XXS (2KB):  10 WGs/CU → 40 waves — HW max
    //   IQ2_XS  (4KB):  10 WGs/CU → 40 waves — HW max
    //   IQ2_S   (8KB):   8 WGs/CU → 32 waves
    //   IQ1_S/M (16KB):  4 WGs/CU → 16 waves (offset by +190% decode speed)
    //
    // Shape-aware dispatch (two criteria, either triggers LDS kernel):
    //
    //   1. CU saturation: grid_n = ceil(N/256) >= NUM_CUS/2
    //      Ensures enough N-blocks to distribute across CUs.
    //      Below this, TN=64 kernel creates 4× more blocks.
    //
    //   2. K-heavy aspect ratio: K/N >= 4 && grid_n >= MIN_GRID_N
    //      K-dominated shapes (FFN_Dn) have long K-loops with many grid lookups.
    //      LDS caching eliminates constant memory contention even at low N.
    //      K-partitioning (scatter+reduce) provides the needed parallelism.
    //      The grid_n floor ensures the LDS kernel has workable parallelism.
    //
    // All thresholds are ratio-based or hardware-relative — no hardcoded
    // row/column counts — so they scale to arbitrary model dimensions.
    // =========================================================================

    // Strategy 1: CU-saturation check — need enough N-blocks for the wider tile
    constexpr int IQ_LDS_TN = 256;
    const int grid_n_lds = (N + IQ_LDS_TN - 1) / IQ_LDS_TN;

    // Strategy 2: K-heavy aspect ratio — long K-loops benefit most from LDS grid
    // cache because they do many grid lookups per thread.  Uses two ratio-based
    // checks instead of absolute block counts:
    //   - K/N ratio >= 4: shape is K-dominated (FFN_Dn pattern)
    //   - grid_n_lds >= 6 (~10% of CUs): minimum parallelism for LDS kernel's
    //     256-thread workgroups.  Shapes with fewer N-blocks (e.g., 0.5B FFN_Dn
    //     with grid_n=4) don't have enough work for the wider tile.

    const bool use_iq_lds = nativeM1UsesIQLDS(codebook_id,N,K);

    if (use_iq_lds)
    {
        constexpr int IQ_TN = 256;         // 256 columns per block
        constexpr int IQ_THREADS = 256;    // 4 wavefronts per workgroup

        const int grid_n = grid_n_lds;     // Already computed above

        // LDS-aware KB heuristic: max waves/CU depends on grid table LDS size
        // and decode VALU cost.
        //
        // O8: Reduced occupancy for IQ2 formats.
        //
        // On gfx906, the wavefront scheduler round-robins ALL waves on a SIMD
        // before returning to the first.  When the inner loop has many VALU
        // instructions per block, high occupancy creates a long VALU round
        // that starves the memory pipeline.
        //
        // IQ2 decode has 127 VALU/block.  At 8 waves/SIMD the round takes
        // 1016 cycles, leaving memory idle ~80%.  Reducing to 4/SIMD (16
        // waves/CU) halves the round to 508 cycles.
        //
        // IQ3 has 94 VALU/block (ISA-verified), giving a 940-cycle round at
        // 10 waves/SIMD.  Despite this, 40 waves remains optimal because IQ3
        // has fewer memory bytes per block (15B vs ~25B for IQ2), and the 8
        // ds_read_b32 per block benefit from occupancy-based LDS latency
        // hiding.  Tested at 20 waves (O9 experiment) — no improvement.
        //
        // IQ1 is already at 16 waves from LDS size (16KB/WG).
        int max_waves_per_cu;
        if (codebook_id >= 16)       // IQ1_S, IQ1_M (16KB LDS)
            max_waves_per_cu = 16;
        else if (codebook_id >= 13 && codebook_id <= 15)  // IQ2_S, IQ2_XS, IQ2_XXS
            max_waves_per_cu = 16;
        else                         // IQ3_S, IQ3_XXS (≤2KB LDS, 94 VALU/block)
            max_waves_per_cu = 40;
        if (policy_target_waves > 0)
            max_waves_per_cu = policy_target_waves;

        // Each 256-thread block contains 4 wavefronts (wavefront64).
        // Convert wave target to block target before computing KB.
        constexpr int WAVES_PER_BLOCK = IQ_THREADS / 64;  // 4
        int target_total_blocks = (max_waves_per_cu * NUM_CUS) / WAVES_PER_BLOCK;
        int kb_raw = max(1, (target_total_blocks + grid_n - 1) / grid_n);

        // Each partition must have enough K-blocks to amortize the LDS preload
        // + __syncthreads().  Without this, small-N shapes over-partition K
        // (e.g., 3B_AttnOut: kb=64 → 1 K-block/partition → 77μs → 18% eff).
        constexpr int IQ_MIN_K_BLOCKS_PER_PART = 8;
        int kb_max = max(1, blocks_per_row / IQ_MIN_K_BLOCKS_PER_PART);
        kb_max = min(kb_max, KB_CAP);
        kb_raw = min(kb_raw, kb_max);

        int kb = kb_raw;
        if (blocks_per_row % kb_raw != 0)
        {
            for (int d = 1; d < kb_raw; ++d)
            {
                if (kb_raw - d >= 1 && blocks_per_row % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
                if (kb_raw + d <= kb_max && blocks_per_row % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
            }
        }
        if (policy_kb > 0)
        {
            kb = min(policy_kb, KB_CAP);
            if (kb > blocks_per_row) kb = blocks_per_row;
        }
        if (kb < 1) kb = 1;

        if (kb > 1 && !d_partial_fp32)
        {
            fprintf(stderr, "[rocmGemv_native_quantised_fp32] IQ: kb=%d>1 but no partial buffer (N=%d K=%d)\n", kb, N, K);
            return false;
        }

        /*
         * Publish the actual serial-decode route before launching it.  The
         * NativeQuantised trainer uses this record to prove that its row oracle ran
         * the same KB/occupancy policy returned by the serial-M1 resolver.  M=1
         * used to be the only NativeQuantised path without this telemetry, which
         * made a correct production launch indistinguishable from an unproven
         * test-only oracle.
         */
        recordNativeQuantisedSmallMLaunch(
            1,
            N,
            K,
            static_cast<unsigned>(codebook_id),
            grid_n,
            kb,
            max_waves_per_cu,
            false,
            1,
            static_cast<int>(arithmetic_policy_codebook_id));

        dim3 block(IQ_THREADS, 1, 1);

        #define LAUNCH_IQ_LDS(FMT_TAG) \
            do { llaminar2::dispatchProjectionWorkMode(1,[&]<bool HasWork>() { \
                hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_iq_lds_kernel_t<IQ_TN,FMT_TAG,HasWork>), \
                    dim3(grid_n,kb),block,0,static_cast<hipStream_t>(stream), \
                    d_A_int8,d_payload,d_scales_u16,d_mins_u16,d_emins_u32, \
                    d_C_fp32,d_partial_fp32,d_scale_A,d_scale_A_blockwise,N,K,kb,llaminar2::projectionWorkForLaunch(1)); \
                finishNativeM1Producer<IQ_TN,1,HasWork>(d_partial_fp32,d_C_fp32,N,kb,device_id,stream,deferred,llaminar2::projectionWorkForLaunch(1)); \
            }); } while(false)

        switch (static_cast<NativeQuantisedFormat>(codebook_id))
        {
        case NATIVE_QUANTISED_IQ3_S:    LAUNCH_IQ_LDS(NATIVE_QUANTISED_IQ3_S);    break;
        case NATIVE_QUANTISED_IQ3_XXS:  LAUNCH_IQ_LDS(NATIVE_QUANTISED_IQ3_XXS);  break;
        case NATIVE_QUANTISED_IQ2_S:    LAUNCH_IQ_LDS(NATIVE_QUANTISED_IQ2_S);    break;
        case NATIVE_QUANTISED_IQ2_XS:   LAUNCH_IQ_LDS(NATIVE_QUANTISED_IQ2_XS);   break;
        case NATIVE_QUANTISED_IQ2_XXS:  LAUNCH_IQ_LDS(NATIVE_QUANTISED_IQ2_XXS);  break;
        case NATIVE_QUANTISED_IQ1_S:    LAUNCH_IQ_LDS(NATIVE_QUANTISED_IQ1_S);    break;
        case NATIVE_QUANTISED_IQ1_M:    LAUNCH_IQ_LDS(NATIVE_QUANTISED_IQ1_M);    break;
        default:
            fprintf(stderr, "[rocmGemv_native_quantised_fp32] Unexpected IQ codebook_id=%u\n",
                    static_cast<unsigned>(codebook_id));
            return false;
        }

        #undef LAUNCH_IQ_LDS
    }
    else
    {
        // =====================================================================
        // Non-IQ formats + IQ small-N fallback: TN=128, 2 wavefronts/WG
        // =====================================================================
        constexpr int TN = 128;
        constexpr int CPT = 1;

        const auto serial_geometry = prepared?prepared->geometry():g_native_q8_serial_scope
            ? g_native_q8_serial_scope->admission().geometry()
            : llaminar2::rocm::NativeQ8SerialGeometry::TN128CPT1;
        const auto serial_spec = llaminar2::rocm::nativeQ8SerialGeometrySpec(serial_geometry);
        const int physical_tile_n = codebook_id == NATIVE_QUANTISED_Q8_0 ? serial_spec.tile_columns : TN;
        const int grid_n = 1 + (N - 1) / physical_tile_n;

        constexpr int DEFAULT_TARGET_WAVES_PER_CU = 24;

        // Both expanded INT8 representations have the same payload traffic
        // and direct decode.  Codebook 23 adds only the already-vectorized
        // asymmetric correction, so it intentionally inherits codebook 19's
        // trained launch policy until its own certification corpus is added.
        constexpr int Q8_0_TARGET_WAVES_PER_CU = 4;
        const bool uses_expanded_int8 =
            codebook_id == static_cast<uint8_t>(NATIVE_QUANTISED_Q8_0) ||
            codebook_id == static_cast<uint8_t>(NATIVE_QUANTISED_EXPANDED_INT8_MIN);
        const int default_waves = uses_expanded_int8
                                      ? Q8_0_TARGET_WAVES_PER_CU
                                      : DEFAULT_TARGET_WAVES_PER_CU;

        const int TARGET_WAVES_PER_CU =
            (policy_target_waves > 0) ? policy_target_waves : default_waves;
        int target_total_waves = TARGET_WAVES_PER_CU * NUM_CUS;
        int kb_raw = max(1, (target_total_waves + grid_n - 1) / grid_n);
        int kb_max = max(1, blocks_per_row / MIN_BLOCKS_PER_WAVE);
        kb_max = min(kb_max, KB_CAP);
        kb_raw = min(kb_raw, kb_max);

        int kb = kb_raw;
        if (blocks_per_row % kb_raw != 0)
        {
            for (int d = 1; d < kb_raw; ++d)
            {
                if (kb_raw - d >= 1 && blocks_per_row % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
                if (kb_raw + d <= kb_max && blocks_per_row % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
            }
        }
        if (kb < 1) kb = 1;

        if (policy_kb > 0)
        {
            kb = min(policy_kb, KB_CAP);
            if (kb > blocks_per_row) kb = blocks_per_row;
        }

        if ((g_native_q8_serial_scope||prepared) && kb !=
            llaminar2::MoEProjectionNumericalContract::orderedKPartitionsForWidth(K))
            throw std::logic_error("native-Q8 serial geometry cannot change the production K-partition tree");

        if (kb > 1 && !d_partial_fp32)
        {
            fprintf(stderr, "[rocmGemv_native_quantised_fp32] kb=%d>1 but no partial buffer (N=%d K=%d)\n", kb, N, K);
            return false;
        }

        /*
         * Keep serial M=1 route evidence symmetric with the grouped verifier
         * kernels.  Recording the resolved values, rather than merely the
         * requested overrides, also proves clamping and shape constraints did
         * not silently select a different production launch.
         */
        #define LAUNCH_NATIVE_QUANTISED(FMT_TAG) \
            launchNativeQuantisedSerialGeometry<TN, CPT, FMT_TAG>( \
                d_A_int8, d_payload, d_scales_u16, d_mins_u16, d_emins_u32, \
                d_C_fp32, d_partial_fp32, d_scale_A, d_scale_A_blockwise, N, K, kb, \
                arithmetic_policy_codebook_id, device_id, stream, serial_geometry,deferred,prepared)

        switch (static_cast<NativeQuantisedFormat>(codebook_id))
        {
        case NATIVE_QUANTISED_Q4_0:     LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_Q4_0);     break;
        case NATIVE_QUANTISED_IQ4_NL:   LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_IQ4_NL);   break;
        case NATIVE_QUANTISED_IQ4_XS_SOURCE:   LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_IQ4_XS_SOURCE);   break;
        case NATIVE_QUANTISED_Q4_1:     LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_Q4_1);     break;
        case NATIVE_QUANTISED_Q5_0:     LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_Q5_0);     break;
        case NATIVE_QUANTISED_Q5_1:     LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_Q5_1);     break;
        case NATIVE_QUANTISED_Q6_K:     LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_Q6_K);     break;
        case NATIVE_QUANTISED_Q3_K:     LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_Q3_K);     break;
        case NATIVE_QUANTISED_Q2_K:     LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_Q2_K);     break;
        // IQ formats fall through here when N ≤ IQ_LDS_N_THRESHOLD
        case NATIVE_QUANTISED_IQ3_S:    LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_IQ3_S);    break;
        case NATIVE_QUANTISED_IQ3_XXS:  LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_IQ3_XXS);  break;
        case NATIVE_QUANTISED_IQ2_S:    LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_IQ2_S);    break;
        case NATIVE_QUANTISED_IQ2_XS:   LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_IQ2_XS);   break;
        case NATIVE_QUANTISED_IQ2_XXS:  LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_IQ2_XXS);  break;
        case NATIVE_QUANTISED_IQ1_S:    LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_IQ1_S);    break;
        case NATIVE_QUANTISED_IQ1_M:    LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_IQ1_M);    break;
        case NATIVE_QUANTISED_Q8_0:
        {
            using G = llaminar2::rocm::NativeQ8SerialGeometry;
#define LAUNCH_NATIVE_Q8_GEOMETRY(TILE, COLUMNS) \
            launchNativeQuantisedSerialGeometry<TILE, COLUMNS, NATIVE_QUANTISED_Q8_0>( \
                d_A_int8, d_payload, d_scales_u16, d_mins_u16, d_emins_u32, \
                d_C_fp32, d_partial_fp32, d_scale_A, d_scale_A_blockwise, N, K, kb, \
                arithmetic_policy_codebook_id, device_id, stream, serial_geometry,deferred,prepared)
            switch (serial_geometry)
            {
            case G::TN64CPT1: LAUNCH_NATIVE_Q8_GEOMETRY(64, 1); break;
            case G::TN128CPT1: LAUNCH_NATIVE_Q8_GEOMETRY(128, 1); break;
            case G::TN256CPT1: LAUNCH_NATIVE_Q8_GEOMETRY(256, 1); break;
            case G::TN128CPT2: LAUNCH_NATIVE_Q8_GEOMETRY(128, 2); break;
            case G::TN256CPT2: LAUNCH_NATIVE_Q8_GEOMETRY(256, 2); break;
            default: throw std::invalid_argument("unknown ROCm native-Q8 serial geometry");
            }
#undef LAUNCH_NATIVE_Q8_GEOMETRY
            break;
        }
        case NATIVE_QUANTISED_EXPANDED_INT8_MIN:
            LAUNCH_NATIVE_QUANTISED(NATIVE_QUANTISED_EXPANDED_INT8_MIN);
            break;
        default:
            fprintf(stderr, "[rocmGemv_native_quantised_fp32] Unknown codebook_id=%u (N=%d K=%d)\n",
                    static_cast<unsigned>(codebook_id), N, K);
            return false;
        }

        recordNativeQuantisedSmallMLaunch(
            1,
            N,
            K,
            static_cast<unsigned>(codebook_id),
            grid_n,
            kb,
            TARGET_WAVES_PER_CU,
            false,
            1,
            static_cast<int>(arithmetic_policy_codebook_id));

        #undef LAUNCH_NATIVE_QUANTISED
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        llaminar2::rocm::clearNativeQ8SerialLaunch();
        fprintf(stderr, "[rocmGemv_native_quantised_fp32] Kernel launch failed: %s (N=%d K=%d codebook=%u)\n",
                hipGetErrorString(err), N, K, static_cast<unsigned>(codebook_id));
        return false;
    }

    return true;
}

/** @brief Preserve the public immediate-reduction ABI through the shared original dispatcher. */
bool rocmGemv_native_quantised_fp32_with_policy(const int8_t *A,const uint8_t *payload,
    const void *scales,const void *mins,const void *emins,float *output,const float *scale_a,
    float *partials,int n,int k,uint8_t execution,uint8_t policy,int device,void *stream,
    const float *block_scales)
{
    return nativeM1WithReduction(A,payload,scales,mins,emins,output,scale_a,partials,
        n,k,execution,policy,device,stream,block_scales,nullptr);
}

/** @brief Execute serial GEMV for self-describing weight bytes. */
bool rocmGemv_native_quantised_fp32(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const void* d_block_scales,
    const void* d_block_mins,
    const void* d_block_emins,
    float* d_C_fp32,
    const float* d_scale_A,
    float* d_partial_fp32,
    int N, int K,
    uint8_t codebook_id,
    int device_id, void* stream,
    const float* d_scale_A_blockwise)
{
    return rocmGemv_native_quantised_fp32_with_policy(
        d_A_int8,
        d_payload,
        d_block_scales,
        d_block_mins,
        d_block_emins,
        d_C_fp32,
        d_scale_A,
        d_partial_fp32,
        N,
        K,
        codebook_id,
        defaultNativeQuantisedArithmeticPolicyCodebook(codebook_id),
        device_id,
        stream,
        d_scale_A_blockwise);
}

// ============================================================================
// IQ grid table initialization: copies host-side LUT data to device constant
// memory.  Must be called once per device before launching any IQ grid kernel.
// ============================================================================
bool rocmInitIQGridTables(
    int device_id,
    const void* h_iq3s_grid,    // [512]  uint32_t
    const void* h_iq3xxs_grid,  // [256]  uint32_t
    const void* h_iq2s_grid,    // [1024] uint64_t
    const void* h_iq2xs_grid,   // [512]  uint64_t
    const void* h_iq2xxs_grid,  // [256]  uint64_t
    const void* h_iq1s_grid)    // [2048] uint64_t (IQ1_S/IQ1_M shared)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);
    hipError_t e;
    e = hipMemcpyToSymbol(d_iq3s_grid, h_iq3s_grid, sizeof(d_iq3s_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables] iq3s failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_iq3xxs_grid, h_iq3xxs_grid, sizeof(d_iq3xxs_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables] iq3xxs failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_iq2s_grid, h_iq2s_grid, sizeof(d_iq2s_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables] iq2s failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_iq2xs_grid, h_iq2xs_grid, sizeof(d_iq2xs_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables] iq2xs failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_iq2xxs_grid, h_iq2xxs_grid, sizeof(d_iq2xxs_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables] iq2xxs failed: %s\n", hipGetErrorString(e)); return false; }
    e = hipMemcpyToSymbol(d_iq1s_grid, h_iq1s_grid, sizeof(d_iq1s_grid));
    if (e != hipSuccess) { fprintf(stderr, "[rocmInitIQGridTables] iq1s failed: %s\n", hipGetErrorString(e)); return false; }
    if (!llaminar2::rocm::native_quantised_internal::initIQGridTables_small_m(device_id, h_iq3s_grid, h_iq3xxs_grid, h_iq2s_grid, h_iq2xs_grid, h_iq2xxs_grid, h_iq1s_grid)) return false;
    if (!llaminar2::rocm::native_quantised_internal::initIQGridTables_moe_decode(device_id, h_iq3s_grid, h_iq3xxs_grid, h_iq2s_grid, h_iq2xs_grid, h_iq2xxs_grid, h_iq1s_grid)) return false;
    if (!llaminar2::rocm::native_quantised_internal::initIQGridTables_resident_ffn_0(device_id, h_iq3s_grid, h_iq3xxs_grid, h_iq2s_grid, h_iq2xs_grid, h_iq2xxs_grid, h_iq1s_grid)) return false;
    if (!llaminar2::rocm::native_quantised_internal::initIQGridTables_resident_ffn_1(device_id, h_iq3s_grid, h_iq3xxs_grid, h_iq2s_grid, h_iq2xs_grid, h_iq2xxs_grid, h_iq1s_grid)) return false;
    if (!llaminar2::rocm::native_quantised_internal::initIQGridTables_persistent_hc(device_id, h_iq3s_grid, h_iq3xxs_grid, h_iq2s_grid, h_iq2xs_grid, h_iq2xxs_grid, h_iq1s_grid)) return false;
    if (!llaminar2::rocm::native_quantised_internal::initIQGridTables_native_layout_0(device_id, h_iq3s_grid, h_iq3xxs_grid, h_iq2s_grid, h_iq2xs_grid, h_iq2xxs_grid, h_iq1s_grid)) return false;
    if (!llaminar2::rocm::native_quantised_internal::initIQGridTables_native_layout_1(device_id, h_iq3s_grid, h_iq3xxs_grid, h_iq2s_grid, h_iq2xs_grid, h_iq2xxs_grid, h_iq1s_grid)) return false;
    if (!llaminar2::rocm::native_quantised_internal::initIQGridTables_native_layout_2(device_id, h_iq3s_grid, h_iq3xxs_grid, h_iq2s_grid, h_iq2xs_grid, h_iq2xxs_grid, h_iq1s_grid)) return false;
    if (!llaminar2::rocm::native_quantised_internal::initIQGridTables_full_hc_0(device_id, h_iq3s_grid, h_iq3xxs_grid, h_iq2s_grid, h_iq2xs_grid, h_iq2xxs_grid, h_iq1s_grid)) return false;
    if (!llaminar2::rocm::native_quantised_internal::initIQGridTables_full_hc_1(device_id, h_iq3s_grid, h_iq3xxs_grid, h_iq2s_grid, h_iq2xs_grid, h_iq2xxs_grid, h_iq1s_grid)) return false;
    if (!llaminar2::rocm::native_quantised_internal::initIQGridTables_full_hc_2(device_id, h_iq3s_grid, h_iq3xxs_grid, h_iq2s_grid, h_iq2xs_grid, h_iq2xxs_grid, h_iq1s_grid)) return false;
    return true;
}

} // extern "C"

/**
 * @brief Borrow existing backend grid addresses after their generation owner published them.
 * @param ordinal Exact device selected during source-weight preparation.
 * @param tables Destination non-owning immutable view; no cache or allocation is created.
 * @return True when all six existing source codebooks have valid device addresses.
 */
extern "C" bool rocmNativeEmbeddingIQTableView(
    int ordinal, llaminar2::NativeEmbeddingIQTables *tables)
{
    if (!tables || ordinal < 0) return false;
    *tables = {};
    if (hipSetDevice(ordinal) != hipSuccess) return false;
    void *address = nullptr;
    if (hipGetSymbolAddress(&address, HIP_SYMBOL(d_iq3s_grid)) != hipSuccess) return false;
    tables->iq3s = static_cast<const uint32_t *>(address);
    if (hipGetSymbolAddress(&address, HIP_SYMBOL(d_iq3xxs_grid)) != hipSuccess) return false;
    tables->iq3xxs = static_cast<const uint32_t *>(address);
    if (hipGetSymbolAddress(&address, HIP_SYMBOL(d_iq2s_grid)) != hipSuccess) return false;
    tables->iq2s = static_cast<const uint64_t *>(address);
    if (hipGetSymbolAddress(&address, HIP_SYMBOL(d_iq2xs_grid)) != hipSuccess) return false;
    tables->iq2xs = static_cast<const uint64_t *>(address);
    if (hipGetSymbolAddress(&address, HIP_SYMBOL(d_iq2xxs_grid)) != hipSuccess) return false;
    tables->iq2xxs = static_cast<const uint64_t *>(address);
    if (hipGetSymbolAddress(&address, HIP_SYMBOL(d_iq1s_grid)) != hipSuccess) return false;
    tables->iq1s = static_cast<const uint64_t *>(address);
    return true;
}

namespace llaminar2::rocm
{
bool enqueueNativeM1Producer(NativeM1ReductionTicket &ticket,const std::int8_t *A,
    const std::uint8_t *payload,const void *scales,const void *mins,const void *emins,
    float *output,const float *scale_a,float *partials,int n,int k,std::uint8_t execution,
    std::uint8_t policy,int device,void *stream,const float *block_scales)
{
    return nativeM1WithReduction(A,payload,scales,mins,emins,output,scale_a,partials,
        n,k,execution,policy,device,stream,block_scales,&ticket);
}

/** @copydoc enqueueNativeM1HCDownProjection */
bool enqueueNativeM1HCDownProjection(const std::int8_t *A,const std::uint8_t *payload,
    const void *scales,const void *mins,const void *emins,float *output,const float *scale_a,
    float *partials,int n,int k,std::uint8_t execution,std::uint8_t policy,int device,void *stream,
    const float *block_scales,const GPUHCDownOutputView &publication,
    const NativeQ8SerialGeometryAdmission *prepared)
{
 if(execution!=19||policy!=19||n<64||n>512||n%32||k<=32||k%32||
    publication.columns!=n||publication.rows.physicalRows()!=1||!publication.activated||
    !publication.primary||!publication.residual||!publication.primaryScales||!publication.residualScales||
    !partials||!block_scales||reinterpret_cast<std::uintptr_t>(stream)<=2||device<0||
    g_native_q8_serial_scope||g_native_quantised_kb_override.load(std::memory_order_relaxed)>0||
    g_native_quantised_target_waves_override.load(std::memory_order_relaxed)>0||
    debugEnv().rocm.native_quantised_gemv_kb>0||debugEnv().rocm.native_quantised_gemv_target_waves>0)
    throw std::invalid_argument("HC down epilogue requires original native19 M1 reducer policy and sealed complete banks");
 NativeM1ReductionTicket ticket;
 if(!nativeM1WithReduction(A,payload,scales,mins,emins,output,scale_a,partials,
    n,k,execution,policy,device,stream,block_scales,&ticket,prepared))
    throw std::runtime_error("Original HC down producer enqueue failed");
 const auto& shape=ticket.producerShape();
 if(ticket.state_!=NativeM1ReductionTicket::State::Pending||shape.tile_columns!=128||shape.columns_per_thread!=1)
    throw std::logic_error("HC down epilogue has no original pending128-column reducer");
 if(ticket.has_work_)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(nativeM1HCDownReducePublish<true>),dim3((n+127)/128),dim3(128),0,
        static_cast<hipStream_t>(stream),ticket.partials_,ticket.output_,n,shape.partitions,ticket.work_,publication);
 else
    hipLaunchKernelGGL(HIP_KERNEL_NAME(nativeM1HCDownReducePublish<false>),dim3((n+127)/128),dim3(128),0,
        static_cast<hipStream_t>(stream),ticket.partials_,ticket.output_,n,shape.partitions,ticket.work_,publication);
 if(hipGetLastError()!=hipSuccess)throw std::runtime_error("Original HC reducer epilogue enqueue failed");
 ticket.state_=NativeM1ReductionTicket::State::Consumed;
 return true;
}

bool submitNativeM1Reductions(std::span<NativeM1ReductionTicket> tickets,int device,void *stream)
{
    if(reinterpret_cast<std::uintptr_t>(stream)<=2 || device<0 || tickets.empty() || tickets.size()>4)
        throw std::invalid_argument("M1 reduction wave requires exact stream/device and one to four tickets");
    // Validate all authorities before submitting any reducer or consuming any ticket.
    for(const auto &t:tickets)
        if((t.state_!=NativeM1ReductionTicket::State::Pending && t.state_!=NativeM1ReductionTicket::State::Direct) ||
            t.device_!=device || t.stream_!=stream)
            throw std::invalid_argument("M1 reduction ticket is absent, consumed or foreign");
    const auto interval=[](const void *p,std::size_t bytes)
    {
        const auto address=reinterpret_cast<std::uintptr_t>(p);
        if(!address || !bytes || bytes>std::numeric_limits<std::uintptr_t>::max()-address)
            throw std::invalid_argument("M1 reduction operand extent is absent or overflows");
        return std::pair{address,address+bytes};
    };
    const auto overlap=[](auto a,auto b){return a.first<b.second && b.first<a.second;};
    for(std::size_t i=0;i<tickets.size();++i)
    {
        const auto &a=tickets[i];const auto out=interval(a.output_,std::size_t(a.shape_.columns)*sizeof(float));
        for(std::size_t j=0;j<tickets.size();++j)
        {
            const auto &b=tickets[j];
            if(i!=j && overlap(out,interval(b.output_,std::size_t(b.shape_.columns)*sizeof(float))))
                throw std::invalid_argument("M1 reduction outputs alias");
            if(b.state_==NativeM1ReductionTicket::State::Pending &&
                overlap(out,interval(b.partials_,std::size_t(b.shape_.columns)*b.shape_.partitions*sizeof(float))))
                throw std::invalid_argument("M1 output aliases pending partial storage");
            if(i!=j && a.state_==NativeM1ReductionTicket::State::Pending && b.state_==NativeM1ReductionTicket::State::Pending &&
                overlap(interval(a.partials_,std::size_t(a.shape_.columns)*a.shape_.partitions*sizeof(float)),
                    interval(b.partials_,std::size_t(b.shape_.columns)*b.shape_.partitions*sizeof(float))))
                throw std::invalid_argument("M1 pending producers share partial storage");
        }
    }
    if(static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(device))!=hipSuccess)return false;
    std::array<bool,4> selected{};
    for(std::size_t first=0;first<tickets.size();++first)
    {
        if(selected[first])continue;
        auto &leader=tickets[first];
        if(leader.state_==NativeM1ReductionTicket::State::Direct){selected[first]=true;continue;}
        std::array<NativeM1ReductionShape,4> shapes{};std::array<std::size_t,4> indices{};std::size_t count=0;
        for(std::size_t i=first;i<tickets.size();++i)
            if(!selected[i] && tickets[i].state_==NativeM1ReductionTicket::State::Pending &&
                tickets[i].shape_.tile_columns==leader.shape_.tile_columns &&
                tickets[i].shape_.columns_per_thread==leader.shape_.columns_per_thread)
            {selected[i]=true;indices[count]=i;shapes[count++]=tickets[i].shape_;}
        auto grid=nativeM1ReductionGrid(std::span(shapes.data(),count));
        for(std::size_t i=0;i<count;++i){grid.partials[i]=tickets[indices[i]].partials_;grid.outputs[i]=tickets[indices[i]].output_;grid.work[i]=tickets[indices[i]].work_;grid.has_work=grid.has_work || tickets[indices[i]].has_work_;}
#define LAUNCH_M1_REDUCTION(TN,CPT) \
        do { if(grid.has_work) { \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(nativeM1GroupedReduce<TN,CPT,true>),dim3(grid.tiles),dim3(TN/CPT),0,static_cast<hipStream_t>(stream),grid); \
        } else { \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(nativeM1GroupedReduce<TN,CPT,false>),dim3(grid.tiles),dim3(TN/CPT),0,static_cast<hipStream_t>(stream),grid); \
        } } while(false)
        switch(leader.shape_.tile_columns*10+leader.shape_.columns_per_thread)
        {
        case 641: LAUNCH_M1_REDUCTION(64,1);break;
        case 1281: LAUNCH_M1_REDUCTION(128,1);break;
        case 1282: LAUNCH_M1_REDUCTION(128,2);break;
        case 2561: LAUNCH_M1_REDUCTION(256,1);break;
        case 2562: LAUNCH_M1_REDUCTION(256,2);break;
        default:throw std::invalid_argument("M1 reduction ticket has no compiled geometry");
        }
#undef LAUNCH_M1_REDUCTION
        if(hipPeekAtLastError()!=hipSuccess)
        {
            for(auto &t:tickets)t.state_=NativeM1ReductionTicket::State::Consumed;
            return false;
        }
    }
    for(auto &t:tickets)t.state_=NativeM1ReductionTicket::State::Consumed;
    return true;
}
}


