/**
 * @file ROCmQuantisedGemmKernel.cpp
 * @brief ROCmQuantisedGemmKernel lifecycle, weight conversion, workspace contract, and verifier dispatch.
 */
#include "ROCmQuantisedGemmKernelInternal.h"

namespace llaminar2
{
    namespace rocm
    {
        namespace rqg_detail
        {
            std::atomic<uint32_t> g_rocm_gemm_workspace_slice_counter{1};
            thread_local bool g_rocm_native_quantised_decode_equivalent_scope = false;
        }


        // =====================================================================
        // ROCmPackedWeights destructor
        // =====================================================================

        ROCmPackedWeights::~ROCmPackedWeights()
        {
            if (!device_uploads.empty())
            {
                for (auto &[device_id, upload] : device_uploads)
                {
                    if (upload.d_int8_data_vnni)
                        rocmQuantGemm_freeDevice(upload.d_int8_data_vnni, device_id);
                    if (upload.d_scales)
                        rocmQuantGemm_freeDevice(upload.d_scales, device_id);
                    if (upload.d_native_quantised_payload)
                        rocmQuantGemm_freeDevice(upload.d_native_quantised_payload, device_id);
#ifdef HAVE_ROCM
                    if (upload.d_native_quantised_scales)
                        rocmQuantGemm_freeDevice(upload.d_native_quantised_scales, device_id);
                    if (upload.d_native_quantised_mins)
                        rocmQuantGemm_freeDevice(upload.d_native_quantised_mins, device_id);
                    if (upload.d_native_quantised_emins)
                        rocmQuantGemm_freeDevice(upload.d_native_quantised_emins, device_id);
                    freeStartupPinnedStaging(upload);
                    if (upload.startup_h2d_stream)
                        (void)hipStreamDestroy(reinterpret_cast<hipStream_t>(upload.startup_h2d_stream));
#endif
                }
            }
            else
            {
                if (d_int8_data_vnni)
                    rocmQuantGemm_freeDevice(d_int8_data_vnni, rocm_device_id);
                if (d_scales)
                    rocmQuantGemm_freeDevice(d_scales, rocm_device_id);
            }
        }

        // =====================================================================
        // Weight packing: see ROCmWeightPacker.cpp
        // (packNativeQuantised, packWeightsToROCm_Q8_0_fast, packWeightsToROCm)
        // =====================================================================

        bool ROCmQuantisedGemmKernel::prepareFusedProjectionGraphCapture(
            size_t projection_count)
        {
            if (projection_count == 0)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel] Fused projection capture "
                          "preparation requires positive fan-out");
                return false;
            }
            if (isGraphCaptureActive())
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel] Fused projection resources "
                          "must be prepared before graph capture begins");
                return false;
            }

            GPUDeviceContextPool::instance().retainAMDContext(rocm_device_id_)->prepareProjectionResources();
            return true;
        }

        ROCmQuantisedGemmKernel::ROCmQuantisedGemmKernel(const TensorBase *weights, int rocm_device_id)
            : weights_(weights),
              packed_(nullptr),
              rocm_device_id_(rocm_device_id),
              N_(0),
              K_(0),
              weights_converted_(false),
              owns_weight_memory_(true), // Legacy path owns weight memory
              slice_id_(g_rocm_gemm_workspace_slice_counter.fetch_add(1, std::memory_order_relaxed)),
              impl_(std::make_unique<Impl>())
        {
            if (!weights)
            {
                throw std::runtime_error("[ROCmQuantisedGemmKernel] Null weight tensor");
            }

            // Get dimensions
            N_ = weights->rows(); // Output features
            K_ = weights->cols(); // Input features

            // Validate it's a quantized type supported by this kernel
            TensorType wt = weights->native_type();
            bool is_quantized = isNativeQuantisedFormat(wt) || isInt8VnniFormat(wt);

            if (!is_quantized)
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Weight tensor must be quantized type, got: " +
                    std::to_string(static_cast<int>(wt)));
            }

            impl_->owns_weight_memory = true;        // Legacy constructor owns weight memory
            impl_->rocm_device_id = rocm_device_id_; // Store device ID for cleanup

            const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(weights);
            const NativeQuantisedFormatInfo *format =
                unpackable ? unpackable->vnniFormatInfo() : nullptr;
            if (!format)
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Quantized tensor has no NativeQuantised source identity");
            }
            impl_->native_source_identity = {
                .codebook_id = format->codebook_id,
                .is_superblock = format->is_superblock,
                .present = true,
            };

            LOG_TRACE("[ROCmQuantisedGemmKernel] Created (legacy) for " << N_ << "x" << K_
                                                                        << " quantized weights (type=" << static_cast<int>(wt)
                                                                        << ") on ROCm device " << rocm_device_id_);
        }

        ROCmQuantisedGemmKernel::ROCmQuantisedGemmKernel(ROCmPackedWeights *packed, int rocm_device_id)
            : weights_(nullptr),
              packed_(packed),
              rocm_device_id_(rocm_device_id),
              N_(0),
              K_(0),
              weights_converted_(false),  // Not yet uploaded to device
              owns_weight_memory_(false), // ROCmPackedWeights owns the memory
              slice_id_(g_rocm_gemm_workspace_slice_counter.fetch_add(1, std::memory_order_relaxed)),
              impl_(std::make_unique<Impl>())
        {
            if (!packed)
            {
                throw std::runtime_error("[ROCmQuantisedGemmKernel] Null packed weights");
            }

            N_ = static_cast<size_t>(packed->N);
            K_ = static_cast<size_t>(packed->K);

            impl_->owns_weight_memory = false;       // Pre-packed path doesn't own weight memory
            impl_->rocm_device_id = rocm_device_id_; // Store device ID for cleanup
            impl_->native_source_identity = packed->native_source_identity;
            if (!impl_->native_source_identity.present)
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Pre-packed weights lost NativeQuantised source identity");
            }

            LOG_TRACE("[ROCmQuantisedGemmKernel] Created (pre-packed) for " << N_ << "x" << K_
                                                                            << " INT8 weights on ROCm device " << rocm_device_id_);
        }

        ROCmQuantisedGemmKernel::ROCmQuantisedGemmKernel(
            int N, int K, int rocm_device_id,
            uint8_t *d_native_quantised, void *d_native_scales,
            void *d_native_mins, void *d_native_emins,
            uint8_t codebook_id, uint32_t blocks_per_row,
            std::shared_ptr<void> lifetime_owner,
            NativeQuantisedSourceIdentity source_identity,
            NativeQuantisedReusableDeviceAllocationFormat allocation_format)
            : weights_(nullptr),
              packed_(nullptr),
              lifetime_owner_(std::move(lifetime_owner)),
              rocm_device_id_(rocm_device_id),
              N_(static_cast<size_t>(N)),
              K_(static_cast<size_t>(K)),
              weights_converted_(true),   // Already on device
              owns_weight_memory_(false), // Batch allocation owns the memory
              slice_id_(g_rocm_gemm_workspace_slice_counter.fetch_add(1, std::memory_order_relaxed)),
              impl_(std::make_unique<Impl>())
        {
            impl_->d_weights_native_quantised = d_native_quantised;
            impl_->d_weights_native_scales = d_native_scales;
            impl_->d_weights_native_mins = d_native_mins;
            impl_->d_weights_native_emins = d_native_emins;
            impl_->native_quantised_codebook_id = codebook_id;
            impl_->native_quantised_blocks_per_row = blocks_per_row;
            const NativeQuantisedFormatInfo *source_format =
                source_identity.present
                    ? native_quantised_formats::forSourceIdentity(
                          source_identity.codebook_id,
                          source_identity.is_superblock)
                    : nullptr;
            if (!source_format ||
                !deviceVnniExecutionCompatibleWithSource(*source_format, codebook_id) ||
                ((codebook_id == 11 || codebook_id == kNativeQuantisedExpandedInt8MinCodebook ||
                  codebook_id == kNativeQuantisedIQ4XSSourceScaleCodebook) && !d_native_mins))
            {
                throw std::invalid_argument(
                    "[ROCmQuantisedGemmKernel] Direct device weights require an exact, "
                    "execution-compatible NativeQuantised source identity");
            }
            impl_->native_source_identity = source_identity;
            if (allocation_format.payload_bytes_per_block != 0)
            {
                const auto execution_format =
                    codebook_id == canonicalDeviceVnniCodebookId(*source_format)
                        ? canonicalDeviceVnniFormat(*source_format)
                        : migrationStableDeviceVnniFormat(*source_format);
                if (allocation_format.payload_bytes_per_block <
                        execution_format.payload_bytes_per_block ||
                    (execution_format.is_asymmetric &&
                     !allocation_format.has_mins) ||
                    (execution_format.has_emins &&
                     !allocation_format.has_emins))
                {
                    throw std::invalid_argument(
                        "[ROCmQuantisedGemmKernel] Reusable allocation cannot represent its live execution format");
                }
            }
            // Resident payloads do not imply resident IQ constants. This public
            // constructor bypasses ensureWeightsConverted(), so finish the same
            // existing generation-owned publication before exposing the engine.
            // The owner initializes every GEMV/GEMM/MoE translation-unit table
            // atomically; no table state or setup operation belongs to replay.
            if (source_identity.codebook_id >= 11 &&
                source_identity.codebook_id <= 17 &&
                (isGraphCaptureActive() ||
                 !ensureIQGridTablesInitialized(rocm_device_id_)))
            {
                throw std::runtime_error(
                    "ROCm resident weight preparation could not publish "
                    "destination IQ tables before capture");
            }
            impl_->native_allocation_format = allocation_format;
            impl_->has_native_quantised = true;
            impl_->owns_weight_memory = false;
            impl_->rocm_device_id = rocm_device_id;

            LOG_TRACE("[ROCmQuantisedGemmKernel] Created (MoE batch device ptrs) for " << N_ << "x" << K_
                                                                                       << " on ROCm device " << rocm_device_id_);
        }

        ROCmQuantisedGemmKernel::~ROCmQuantisedGemmKernel() = default;

        ROCmQuantisedGemmKernel::ROCmQuantisedGemmKernel(ROCmQuantisedGemmKernel &&) noexcept = default;
        ROCmQuantisedGemmKernel &ROCmQuantisedGemmKernel::operator=(ROCmQuantisedGemmKernel &&) noexcept = default;

        /**
         * @brief Scoped ROCm NativeQuantised verifier dispatch contract.
         *
         * The grouped verifier has to match the normal M=1 serial decode path,
         * not merely an idealized direct GEMV. ROCm NativeQuantised dispatch is
         * generated per shape and M; choosing a grouped-M split policy can alter
         * FP32 partial-reduction order relative to the serial baseline. While
         * this scope is active, every grouped verifier depth reuses the generated M=1
         * policy for the same projection shape. That keeps the path economical
         * and graph-capturable while preserving serial-decode equivalence.
         */
        class ScopedNativeQuantisedDecodeEquivalentDispatch final : public ITensorGemm::VerifierKernelModeScope
        {
        public:
            explicit ScopedNativeQuantisedDecodeEquivalentDispatch(std::optional<DeviceRowRange> rows)
                : VerifierKernelModeScope(rows)
            {
                previous_ = g_rocm_native_quantised_decode_equivalent_scope;
                g_rocm_native_quantised_decode_equivalent_scope = true;
                rocmGemv_native_quantised_set_decode_equivalent_m1_config(1);
            }

            ~ScopedNativeQuantisedDecodeEquivalentDispatch()
            {
                g_rocm_native_quantised_decode_equivalent_scope = previous_;
                rocmGemv_native_quantised_set_decode_equivalent_m1_config(previous_ ? 1 : 0);
            }

            ScopedNativeQuantisedDecodeEquivalentDispatch(const ScopedNativeQuantisedDecodeEquivalentDispatch &) = delete;
            ScopedNativeQuantisedDecodeEquivalentDispatch &operator=(const ScopedNativeQuantisedDecodeEquivalentDispatch &) = delete;

        private:
            bool previous_ = false;
        };

        /**
         * @brief Bind generated NativeQuantised geometry to the serial TP shard width.
         */
        class ScopedNativeQuantisedSerialPartition final
            : public ITensorGemm::OutputPartitionEquivalenceScope
        {
        public:
            explicit ScopedNativeQuantisedSerialPartition(int serial_partition_n)
                : previous_(rocmGemv_native_quantised_get_serial_partition_n())
            {
                rocmGemv_native_quantised_set_serial_partition_n(serial_partition_n);
            }

            ~ScopedNativeQuantisedSerialPartition() override
            {
                rocmGemv_native_quantised_set_serial_partition_n(previous_);
            }

            ScopedNativeQuantisedSerialPartition(
                const ScopedNativeQuantisedSerialPartition &) = delete;
            ScopedNativeQuantisedSerialPartition &operator=(
                const ScopedNativeQuantisedSerialPartition &) = delete;

        private:
            int previous_ = 0;
        };

        std::unique_ptr<ITensorGemm::VerifierKernelModeScope>
        ROCmQuantisedGemmKernel::beginVerifierDecodeEquivalentScope(
            std::optional<DeviceRowRange> rows)
        {
            return std::make_unique<ScopedNativeQuantisedDecodeEquivalentDispatch>(rows);
        }

        std::unique_ptr<ITensorGemm::OutputPartitionEquivalenceScope>
        ROCmQuantisedGemmKernel::beginOutputPartitionEquivalenceScope(
            int actual_output_columns,
            int serial_partition_columns)
        {
            if (actual_output_columns <= 0 ||
                serial_partition_columns <= 0 ||
                actual_output_columns != static_cast<int>(N_) ||
                serial_partition_columns > actual_output_columns)
            {
                throw std::invalid_argument(
                    "[ROCmQuantisedGemmKernel] Invalid replicated-output serial partition contract");
            }
            return std::make_unique<ScopedNativeQuantisedSerialPartition>(
                serial_partition_columns);
        }

        std::unique_ptr<ITensorGemm::OutputPartitionEquivalenceScope>
        ROCmQuantisedGemmKernel::beginOriginalOutputSliceScope(const OriginalOutputSlice &slice)
        {
            slice.require(this, static_cast<int>(N_), static_cast<int>(K_));
            if (slice.device() != DeviceId::rocm(rocm_device_id_))
                throw std::invalid_argument("Original output slice belongs to a foreign ROCm endpoint");
            if (!impl_ || !(impl_->has_native_quantised && impl_->d_weights_native_quantised && impl_->d_weights_native_scales) ||
                !impl_->native_source_identity.present)
                throw std::invalid_argument("Original output slice requires ready source-authenticated native quantized weights");
            return std::make_unique<ScopedNativeQuantisedSerialPartition>(slice.policyColumns());
        }

        bool ROCmQuantisedGemmKernel::weights_converted() const
        {
            /*
             * Prepared-weight ROCm kernels can be constructed directly from
             * model-lifetime WeightVRAMPool native-quantised descriptors.  The
             * resident descriptor pointers are the execution contract for
             * grouped MoE verifier kernels; weights_converted_ remains the
             * legacy lazy-upload flag for host-weight constructors.
             */
            return weights_converted_ ||
                   (impl_ &&
                    impl_->d_weights_native_quantised &&
                    impl_->d_weights_native_scales &&
                    impl_->has_native_quantised &&
                    N_ > 0 &&
                    K_ > 0 &&
                    impl_->native_quantised_blocks_per_row > 0);
        }

        WorkspaceRequirements ROCmQuantisedGemmKernel::q8ResidualCorrectionWorkspaceRequirements(int m, int n,
                                                                                                 int k) const
        {
            validateQ8ResidualCorrectionGeometry(m, n, k);
            NativeQuantisedSourceIdentity source;
            if (!weights_converted() || !impl_ || !exportNativeQuantisedSourceIdentity(source) ||
                n != static_cast<int>(N_) || k != static_cast<int>(K_))
                throw std::invalid_argument(
                    "TwoTerm ROCm requires already prepared original source and exact geometry");
            const DeviceNativeQuantisedMatrixDesc descriptor{
                .payload = impl_->d_weights_native_quantised,
                .scales = impl_->d_weights_native_scales,
                .mins = impl_->d_weights_native_mins,
                .emins = impl_->d_weights_native_emins,
                .n = n,
                .k = k,
                .blocks_per_row = impl_->native_quantised_blocks_per_row,
                .codebook_id = impl_->native_quantised_codebook_id,
                .allocation_payload_bytes_per_block = impl_->native_allocation_format.payload_bytes_per_block,
                .allocation_has_mins = static_cast<std::uint8_t>(impl_->native_allocation_format.has_mins),
                .allocation_has_emins = static_cast<std::uint8_t>(impl_->native_allocation_format.has_emins),
                .source_codebook_id = source.codebook_id,
                .source_is_superblock = static_cast<std::uint8_t>(source.is_superblock),
                .source_identity_present = static_cast<std::uint8_t>(source.present)};
            (void)requireQ8ResidualNativeMatrix(descriptor, m, n, k);
            if (q8ResidualNeedsIQTables(descriptor.codebook_id))
            {
                auto *backend = getROCmBackend();
                const auto generation = backend ? backend->deviceRuntimeGeneration(rocm_device_id_) : 0;
                if (!generation)
                    throw std::logic_error("TwoTerm ROCm IQ prepared source generation is unavailable");
                if (!isGraphCaptureActive())
                {
                    // Multiple cold graph builders may query this same prepared
                    // engine. Publish once; no execution path enters this guard.
                    std::call_once(impl_->residual_iq_once, [&] {
                        NativeEmbeddingIQTables tables;
                        if (!prepareNativeEmbeddingTablesROCm(rocm_device_id_, TensorType::IQ3_S, tables))
                            throw std::logic_error("TwoTerm ROCm IQ table binding requires cold source setup");
                        if (backend->deviceRuntimeGeneration(rocm_device_id_) != generation)
                            throw std::logic_error("TwoTerm IQ backend generation changed during cold binding");
                        impl_->residual_iq_tables = tables;
                        impl_->residual_iq_generation.store(generation, std::memory_order_release);
                    });
                }
                // Capture never waits on cold setup or attempts table binding.
                // Acquire also makes the immutable pointer view safe to read.
                if (impl_->residual_iq_generation.load(std::memory_order_acquire) != generation)
                    throw std::logic_error("TwoTerm ROCm IQ declaration requires its original cold generation");
            }
            return gpuQ8ResidualCorrectionWorkspaceRequirements(m, n, k);
        }

        void ROCmQuantisedGemmKernel::multiply_q8_residual_corrected(const TensorBase *input,
                                                                     TensorBase *output, DeviceRowRange rows,
                                                                     int n, int k,
                                                                     DeviceWorkspaceManager *workspace)
        {
            validateQ8ResidualCorrectionGeometry(rows.physicalRows(), n, k);
            const DeviceId target = DeviceId::rocm(rocm_device_id_);
            if (!workspace || workspace->device() != target || !input || !output || input == output ||
                input->native_type() != TensorType::FP32 || output->native_type() != TensorType::FP32 ||
                output->isMapped() || n != static_cast<int>(N_) || k != static_cast<int>(K_) ||
                !weights_converted())
                throw std::invalid_argument("Two-term ROCm Q8 requires prepared FP32 tensor operands, exact "
                                            "participant workspace and immutable native weights");
            validateQ8ResidualTensorCapacities(rows.physicalRows(), n, k, input->numel(), output->numel());
            void *stream = gpu_stream_;
            if (!stream)
                throw std::invalid_argument("Two-term ROCm Q8 requires its exact producer stream");
            if (q8ResidualNeedsIQTables(impl_->native_quantised_codebook_id))
            {
                auto *backend = getROCmBackend();
                const auto generation = impl_->residual_iq_generation.load(std::memory_order_acquire);
                if (!backend || !generation || backend->deviceRuntimeGeneration(rocm_device_id_) != generation)
                    throw std::logic_error("TwoTerm ROCm IQ execution requires its original cold prepared generation");
            }
            DeviceNativeQuantisedMatrixDesc weights;
            if (!exportNativeQuantisedMatrixDesc(weights))
                throw std::logic_error("Prepared two-term ROCm Q8 source export failed");
            (void)requireQ8ResidualNativeMatrix(weights, rows.physicalRows(), n, k);
            TransferEngine::requireDeviceInput(const_cast<TensorBase *>(input), target, stream);
            TransferEngine::requireDeviceOutput(output, target, stream);
            const auto values = static_cast<std::size_t>(rows.physicalRows()) * k,
                       scales = values / 32 * sizeof(float);
            const auto requireBank = [&](const std::string &name, std::size_t bytes) -> void * {
                auto *pointer = workspace->getBuffer(name);
                if (!pointer || workspace->getBufferSize(name) < bytes)
                    throw std::logic_error(
                        "Two-term ROCm Q8 invocation lacks an admitted named activation bank");
                return pointer;
            };
            rocmQ8ResidualCorrectedProjection(
                {weights, rows, Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal,
                 static_cast<const float *>(input->gpu_data_ptr()),
                 static_cast<float *>(output->gpu_data_ptr()),
                 static_cast<signed char *>(requireBank(kGPUProjectionPrimaryQ8, values)),
                 static_cast<signed char *>(requireBank(kGPUProjectionResidualQ8, values)),
                 static_cast<float *>(requireBank(kGPUProjectionPrimaryQ8Scales, scales)),
                 static_cast<float *>(requireBank(kGPUProjectionResidualQ8Scales, scales))},
                stream, impl_->residual_iq_tables);
            publishROCmTensorWrite(*output, rocm_device_id_, stream);
        }

        /** @copydoc ITensorGemm::multiply_hc_down_projection */
        bool ROCmQuantisedGemmKernel::multiply_hc_down_projection(const HCDownProjectionInput& input,
            TensorBase* output,int n,int k,DeviceWorkspaceManager* workspace,const HCDownOutputPublication* output_publication)
        {
            if(output_publication&&output_publication->producerProgram()==HCDownProducerProgram::GroupedM4Ordered)
            {
                constexpr int rows=4, columns=320, width=10240, partitions=16;
                const auto target=DeviceId::rocm(rocm_device_id_);
                (void)ExplicitGPUStream(gpu_stream_);
                if(input.publication||!workspace||workspace->device()!=target||!workspace->isAllocated()||
                   !output||output->isMapped()||&input.normalized==output||n!=columns||k!=width||
                   input.normalized.native_type()!=TensorType::FP32||output->native_type()!=TensorType::FP32||
                   input.normalized.numel()<rows*width||output->numel()<rows*columns)
                    throw std::invalid_argument("Grouped HC-down reducer requires the original counted M4 storage");
                DeviceNativeQuantisedMatrixDesc weights;
                if(!inspectPreparedNativeQuantisedMatrixDesc(weights)||weights.codebook_id!=19||
                   !weights.source_identity_present||weights.source_codebook_id!=19||weights.source_is_superblock)
                    throw std::logic_error("Grouped HC-down reducer lost the original signed-Q8 source");
                (void)requireQ8ResidualNativeMatrix(weights,rows,n,k);
                const auto publication=output_publication->deviceView(output,workspace,gpu_stream_);
                const auto requireBank=[&](const auto& name,std::size_t bytes) {
                    void* data=workspace->getBuffer(name);
                    if(!data||workspace->getBufferSize(name)<bytes)
                        throw std::logic_error("Grouped HC-down reducer lost the exact original scratch extent");
                    return data;
                };
                auto* codes=static_cast<int8_t*>(requireBank(GemmWorkspaceBuffers::QUANT_A,rows*width));
                auto* scales=static_cast<float*>(requireBank(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE,
                    rows*(width/32)*sizeof(float)));
                auto* sums=static_cast<int32_t*>(requireBank(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE,
                    rows*(width/32)*sizeof(int32_t)));
                auto* partials=static_cast<float*>(requireBank(scatterPartialBufferName(),
                    partitions*rows*columns*sizeof(float)));
                TransferEngine::requireDeviceInput(const_cast<TensorBase*>(&input.normalized),target,gpu_stream_);
                TransferEngine::requireDeviceOutput(output,target,gpu_stream_);
                auto mode=beginVerifierDecodeEquivalentScope(publication.rows);
                if(!rocmQuantGemm_quantizeActivationsBlockwiseWithSums_rows(
                    static_cast<const float*>(input.normalized.gpu_data_ptr()),codes,scales,sums,
                    rows,width,rocm_device_id_,gpu_stream_,32,&publication.rows)||
                   !rocm::enqueueGroupedHCDownReducePublish(codes,weights.payload,
                    static_cast<const uint16_t*>(weights.scales),static_cast<float*>(output->gpu_data_ptr()),
                    scales,sums,partials,rocm_device_id_,gpu_stream_,publication))
                    return false;
                PerfStatsCollector::addCounter("gemm","qfn_grouped_hc_down_reduce_publish_nodes",1.0,
                    "gemm","rocm:"+std::to_string(rocm_device_id_));
                output_publication->recordPublished();
                publishROCmTensorWrite(*output,rocm_device_id_,gpu_stream_);
                return true;
            }
            const auto target=DeviceId::rocm(rocm_device_id_);
            (void)ExplicitGPUStream(gpu_stream_);
            if(!workspace||workspace->device()!=target||!workspace->isAllocated()||!output||output->isMapped()||
                input.normalized.native_type()!=TensorType::FP32||output->native_type()!=TensorType::FP32||
                &input.normalized==output||n!=static_cast<int>(N_)||k!=static_cast<int>(K_)||k<=0||k%32||
                input.normalized.numel()<static_cast<std::size_t>(k)||output->numel()<static_cast<std::size_t>(n))
                throw std::invalid_argument("HC-down publication requires original prepared plain ordinary M1 operands");
            const auto publication=output_publication
                ?std::optional{output_publication->deviceView(output,workspace,gpu_stream_)}:std::nullopt;
            DeviceNativeQuantisedMatrixDesc weights;
            if(!inspectPreparedNativeQuantisedMatrixDesc(weights))
                throw std::logic_error("HC-down publication requires its original already-prepared native source");
            (void)requireQ8ResidualNativeMatrix(weights,1,n,k);
            TransferEngine::requireDeviceInput(const_cast<TensorBase*>(&input.normalized),target,gpu_stream_);
            TransferEngine::requireDeviceOutput(output,target,gpu_stream_);
            auto* d_A_int8=static_cast<int8_t*>(workspace->getBuffer(GemmWorkspaceBuffers::QUANT_A));
            auto* blockscales=static_cast<float*>(workspace->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));
            if(!d_A_int8||!blockscales||workspace->getBufferSize(GemmWorkspaceBuffers::QUANT_A)<static_cast<std::size_t>(k)||
                workspace->getBufferSize(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE)<static_cast<std::size_t>(k/32)*sizeof(float))
                throw std::logic_error("HC-down publication lost its original input scratch banks");
            if(input.publication) {
                input.publication->requireRead(input.normalized,*output,workspace,gpu_stream_);
                if(input.publication->codes()!=d_A_int8||input.publication->scales()!=blockscales)
                    throw std::logic_error("HC-down publication changed the original input bank binding");
            } else {
                if(!rocmQuantGemm_quantizeActivationsBlockwiseScoped(
                    static_cast<const float*>(input.normalized.gpu_data_ptr()),d_A_int8,blockscales,1,k,rocm_device_id_,gpu_stream_))
                    return false;
            }
            auto* rowscale=static_cast<float*>(workspace->getBuffer(GemmWorkspaceBuffers::SCALES_A));
            auto* partials=static_cast<float*>(workspace->getBuffer(scatterPartialBufferName()));
            if(!rowscale||!partials)throw std::logic_error("HC-down publication lost original producer/reducer scratch");
            const auto policy=requireNativeQuantisedArithmeticPolicyCodebook(weights.codebook_id,impl_->native_source_identity);
            const bool success=publication
                ?enqueueNativeM1HCDownProjection(d_A_int8,weights.payload,weights.scales,weights.mins,weights.emins,
                    static_cast<float*>(output->gpu_data_ptr()),rowscale,partials,n,k,weights.codebook_id,policy,
                    rocm_device_id_,gpu_stream_,blockscales,*publication,output_publication->preparedProducer())
                :rocmGemv_native_quantised_fp32_with_policy(d_A_int8,weights.payload,
                    weights.scales,weights.mins,weights.emins,static_cast<float*>(output->gpu_data_ptr()),
                    rowscale,partials,n,k,weights.codebook_id,policy,rocm_device_id_,gpu_stream_,blockscales);
            if(!success)return false;
            if(output_publication)output_publication->recordPublished();
            if(input.publication)input.publication->consumeRead();
            publishROCmTensorWrite(*output,rocm_device_id_,gpu_stream_);
            return true;
        }

        /** @brief Retain original native source admission while fusing HC input publication. */
        void ROCmQuantisedGemmKernel::multiply_q8_residual_corrected(HCResidualDownInput hc,
                                                                     TensorBase *output, DeviceRowRange rows,
                                                                     int n, int k,
                                                                     DeviceWorkspaceManager *workspace)
        {
            multiplyHCResidualDownWithPolicy(LocalGPUExpertKernelPolicy::Generic,std::move(hc),output,rows,n,k,workspace);
        }

        void ROCmQuantisedGemmKernel::multiplyHCResidualDownWithPolicy(LocalGPUExpertKernelPolicy policy,
            HCResidualDownInput hc,TensorBase* output,DeviceRowRange rows,int n,int k,DeviceWorkspaceManager* workspace)
        {
            const auto*input=&hc.raw;auto*activated=&hc.activated;
            validateHCResidualDownRows(rows,rows.physicalRows());
            validateQ8ResidualCorrectionGeometry(rows.physicalRows(), n, k);
            const DeviceId target = DeviceId::rocm(rocm_device_id_);
            if (!workspace || workspace->device() != target || !input || !output || input == output ||
                input->native_type() != TensorType::FP32 || output->native_type() != TensorType::FP32 ||
                output->isMapped() || n != static_cast<int>(N_) || k != static_cast<int>(K_) ||
                !weights_converted())
                throw std::invalid_argument("Two-term ROCm Q8 requires prepared FP32 tensor operands, exact "
                                            "participant workspace and immutable native weights");
            validateQ8ResidualTensorCapacities(rows.physicalRows(), n, k, input->numel(), output->numel());
            void *stream = gpu_stream_;
            if (!stream)
                throw std::invalid_argument("Two-term ROCm Q8 requires its exact producer stream");
            if (q8ResidualNeedsIQTables(impl_->native_quantised_codebook_id))
            {
                auto *backend = getROCmBackend();
                const auto generation = impl_->residual_iq_generation.load(std::memory_order_acquire);
                if (!backend || !generation || backend->deviceRuntimeGeneration(rocm_device_id_) != generation)
                    throw std::logic_error("TwoTerm ROCm IQ execution requires its original cold prepared generation");
            }
            DeviceNativeQuantisedMatrixDesc weights;
            if (!exportNativeQuantisedMatrixDesc(weights))
                throw std::logic_error("Prepared two-term ROCm Q8 source export failed");
            (void)requireQ8ResidualNativeMatrix(weights, rows.physicalRows(), n, k);
            TransferEngine::requireDeviceInput(const_cast<TensorBase *>(input), target, stream);
            if(activated==output||activated->native_type()!=TensorType::FP32||activated->isMapped()||
               activated->numel()<static_cast<std::size_t>(rows.physicalRows())*k)
                throw std::invalid_argument("HC GPU requires its retained prepared FP32 activation output");
            if(hc.prepublished) {
                hc.prepublished->requirePublished(*input,*activated,rows,workspace,stream);
                TransferEngine::requireDeviceInput(activated,target,stream);
            } else TransferEngine::requireDeviceOutput(activated,target,stream);
            TransferEngine::requireDeviceOutput(output, target, stream);
            if(hc.short_k_read) {
                const auto& read=*hc.short_k_read;
                if(read.normalized.native_type()!=TensorType::FP32||read.mixed.native_type()!=TensorType::FP32||
                   read.mixed.isMapped()||rows.physicalRows()!=1||n%4||
                   read.normalized.numel()<static_cast<std::size_t>(n)||read.mixed.numel()<static_cast<std::size_t>(n/4))
                    throw std::invalid_argument("HC short-K read requires complete original M1 FP32 branch/mixed operands");
                TransferEngine::requireDeviceInput(const_cast<TensorBase*>(&read.normalized),target,stream);
                TransferEngine::requireDeviceOutput(&read.mixed,target,stream);
            }
            const auto values = static_cast<std::size_t>(rows.physicalRows()) * k,
                       scales = values / 32 * sizeof(float);
            const auto requireBank = [&](const std::string &name, std::size_t bytes) -> void * {
                auto *pointer = workspace->getBuffer(name);
                if (!pointer || workspace->getBufferSize(name) < bytes)
                    throw std::logic_error(
                        "Two-term ROCm Q8 invocation lacks an admitted named activation bank");
                return pointer;
            };
            const GPUHCResidualDownPublication publication{
                {weights, rows, Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal,
                 static_cast<const float *>(input->gpu_data_ptr()),
                 static_cast<float *>(output->gpu_data_ptr()),
                 static_cast<signed char *>(requireBank(kGPUProjectionPrimaryQ8, values)),
                 static_cast<signed char *>(requireBank(kGPUProjectionResidualQ8, values)),
                 static_cast<float *>(requireBank(kGPUProjectionPrimaryQ8Scales, scales)),
                 static_cast<float *>(requireBank(kGPUProjectionResidualQ8Scales, scales))},
                 static_cast<float*>(activated->gpu_data_ptr()),
                 hc.prepublished?HCDownQ8PublicationSource::OriginalReducer:HCDownQ8PublicationSource::Standalone};
            if(hc.short_k_read) {
                const auto& read=*hc.short_k_read;
                rocmHCResidualShortKReadQ8Projection({publication,
                    static_cast<const float*>(read.normalized.gpu_data_ptr()),
                    static_cast<float*>(read.mixed.gpu_data_ptr())},stream);
                publishROCmTensorWrite(read.mixed, rocm_device_id_, stream);
            } else rocmHCResidualDownQ8ProjectionWithPolicy(publication,stream,impl_->residual_iq_tables,
                usesGFX906MI50Kernels(policy));
            if(hc.prepublished)hc.prepublished->consume();
            else publishROCmTensorWrite(*activated, rocm_device_id_, stream);
            publishROCmTensorWrite(*output, rocm_device_id_, stream);
        }

        /**
         * @brief Consume fused norm publication within the original prepared GEMM owner.
         * @details Only the preparation launch changes. The same M1 native
         * producer/reducer and source arithmetic policy immediately consume the
         * original named workspace. No prepared activation escapes this call.
         */
        ROCmQuantisedGemmKernel::GatedNormExecutionBinding ROCmQuantisedGemmKernel::gatedNormExecutionBinding(
            GatedRMSNormProjectionInput input,TensorBase* output,int n,int k,
            DeviceWorkspaceManager* workspace) const
        {
            const auto target=DeviceId::rocm(rocm_device_id_);
            if(!gatedNormSingleQ8Geometry(1,k,input.parameters)||!output||
               output==&input.values||output==&input.gate||output==&input.gamma||
               &input.values==&input.gate||&input.values==&input.gamma||
               input.values.native_type()!=TensorType::FP32||input.gate.native_type()!=TensorType::FP32||
               input.gamma.native_type()!=TensorType::FP32||output->native_type()!=TensorType::FP32||
               input.values.current_device()!=std::optional{target}||input.gate.current_device()!=std::optional{target}||
               input.gamma.current_device()!=std::optional{target}||output->current_device()!=std::optional{target}||
               input.values.isMapped()||output->isMapped()||n!=static_cast<int>(N_)||k!=static_cast<int>(K_)||
               input.values.numel()<static_cast<size_t>(k)||input.gate.numel()<static_cast<size_t>(k)||
               input.gamma.numel()<128||output->numel()<static_cast<size_t>(n)||
               !workspace||workspace!=workspace_||workspace->device()!=target||!workspace->isAllocated()||
               !weights_converted()||currentProjectionRowWork(1)||countedVerifierRows(1))
                throw std::invalid_argument("Gated-norm SingleQ8 requires its ordinary scalar FP32 tensors, exact prepared source and original bound workspace");
            DeviceNativeQuantisedMatrixDesc weights;
            if(!inspectPreparedNativeQuantisedMatrixDesc(weights))
                throw std::logic_error("Gated-norm SingleQ8 requires already-prepared native source planes");
            const auto physical=requireQ8ResidualNativeMatrix(weights,1,n,k);
            const auto requirements=getWorkspaceRequirements(1,n,k);
            const auto requireBank=[&](const std::string& name,size_t minimum)->void* {
                const auto* needed=requirements.find(name);
                if(!needed)throw std::logic_error("Gated-norm SingleQ8 bank is absent from the original BOM");
                minimum=std::max(minimum,needed->size_bytes);
                auto* pointer=workspace->getBuffer(name);
                if(!pointer||workspace->getBufferSize(name)<minimum)
                    throw std::logic_error("Gated-norm SingleQ8 lacks an original admitted quantization/projection bank");
                return pointer;
            };
            auto* quant=static_cast<std::int8_t*>(requireBank(GemmWorkspaceBuffers::QUANT_A,k));
            auto* scales=static_cast<float*>(requireBank(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE,
                static_cast<size_t>(k/32)*sizeof(float)));
            auto* partials=static_cast<float*>(requireBank(scatterPartialBufferName(),sizeof(float)));
            auto* row_scales=static_cast<float*>(requireBank(GemmWorkspaceBuffers::SCALES_A,sizeof(float)));
            (void)requireBank(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE,static_cast<size_t>(k/32)*sizeof(std::int32_t));
            const auto blocks=static_cast<size_t>(n)*weights.blocks_per_row;
            // Protect complete original capacities, including unused immutable
            // allocation planes and the scalar consumer's untouched sums bank.
            const std::array writes{
                Q8ResidualStorageSpan{input.values.gpu_data_ptr(),std::max(input.values.size_bytes(),static_cast<size_t>(k)*sizeof(float)),alignof(float)},
                Q8ResidualStorageSpan{output->gpu_data_ptr(),std::max(output->size_bytes(),static_cast<size_t>(n)*sizeof(float)),alignof(float)},
                Q8ResidualStorageSpan{quant,workspace->getBufferSize(GemmWorkspaceBuffers::QUANT_A),16},
                Q8ResidualStorageSpan{scales,workspace->getBufferSize(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE),alignof(float)},
                Q8ResidualStorageSpan{partials,workspace->getBufferSize(scatterPartialBufferName()),alignof(float)}};
            const std::array reads{
                Q8ResidualStorageSpan{input.gate.gpu_data_ptr(),std::max(input.gate.size_bytes(),static_cast<size_t>(k)*sizeof(float)),alignof(float)},
                Q8ResidualStorageSpan{input.gamma.gpu_data_ptr(),std::max(input.gamma.size_bytes(),size_t{128}*sizeof(float)),alignof(float)},
                Q8ResidualStorageSpan{weights.payload,blocks*(weights.allocation_payload_bytes_per_block?
                    weights.allocation_payload_bytes_per_block:physical.payload_bytes_per_block),16},
                Q8ResidualStorageSpan{weights.scales,blocks*sizeof(std::uint16_t),alignof(std::uint16_t)},
                Q8ResidualStorageSpan{weights.mins,weights.mins?blocks*sizeof(std::uint16_t):0,alignof(std::uint16_t)},
                Q8ResidualStorageSpan{weights.emins,weights.emins?blocks*sizeof(std::uint32_t):0,alignof(std::uint32_t)},
                Q8ResidualStorageSpan{workspace->getBuffer(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE),
                    workspace->getBufferSize(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE),alignof(std::int32_t)},
                Q8ResidualStorageSpan{row_scales,workspace->getBufferSize(GemmWorkspaceBuffers::SCALES_A),alignof(float)}};
            validateQ8ResidualStorage(writes,reads);
            const rocm::GatedNormSingleQ8Input native{
                static_cast<const float*>(input.values.gpu_data_ptr()),
                static_cast<const float*>(input.gate.gpu_data_ptr()),
                static_cast<const float*>(input.gamma.gpu_data_ptr()),
                static_cast<float*>(input.values.gpu_data_ptr()),k,input.parameters,quant,scales};
            return {native,weights,row_scales,partials};
        }

        /** @copydoc validate_gated_rmsnorm_single_q8 */
        void ROCmQuantisedGemmKernel::validate_gated_rmsnorm_single_q8(
            GatedRMSNormProjectionInput input,TensorBase* output,int n,int k,
            DeviceWorkspaceManager* workspace) const
        { (void)gatedNormExecutionBinding(input,output,n,k,workspace); }

        /** @copydoc multiply_gated_rmsnorm_single_q8 */
        bool ROCmQuantisedGemmKernel::multiply_gated_rmsnorm_single_q8(
            GatedRMSNormProjectionInput input,TensorBase* output,int n,int k,
            DeviceWorkspaceManager* workspace)
        {
            (void)ExplicitGPUStream(gpu_stream_);
            const auto native=gatedNormExecutionBinding(input,output,n,k,workspace);
            const auto target=DeviceId::rocm(rocm_device_id_);
            TransferEngine::requireDeviceInput(&input.values,target,gpu_stream_);
            TransferEngine::requireDeviceInput(const_cast<TensorBase*>(&input.gate),target,gpu_stream_);
            TransferEngine::requireDeviceInput(const_cast<TensorBase*>(&input.gamma),target,gpu_stream_);
            TransferEngine::requireDeviceOutput(output,target,gpu_stream_);
            if(!rocm::launchGatedNormSingleQ8(native.input,rocm_device_id_,gpu_stream_))return false;
            const auto& weights=native.weights;
            if(!rocmGemv_native_quantised_fp32_with_policy(native.input.quant,weights.payload,weights.scales,weights.mins,
                weights.emins,static_cast<float*>(output->gpu_data_ptr()),native.row_scales,native.partials,n,k,
                weights.codebook_id,requireNativeQuantisedArithmeticPolicyCodebook(
                    impl_->native_quantised_codebook_id,impl_->native_source_identity),
                rocm_device_id_,gpu_stream_,native.input.scales))return false;
            publishROCmTensorWrite(input.values,rocm_device_id_,gpu_stream_);
            publishROCmTensorWrite(*output,rocm_device_id_,gpu_stream_);
            if(PerfStatsCollector::isDomainEnabled("gdn_output"))
                PerfStatsCollector::addCounter("gdn_output","normalization_submission",1.,"gemm",
                    "rocm:"+std::to_string(rocm_device_id_),PerfStatsCollector::Tags{
                        {"preparation","fused_single_q8"},{"m","1"},{"n",std::to_string(n)},
                        {"k",std::to_string(k)},{"norm_dim","128"},{"normalized_fp32","retained"},
                        {"codebook",std::to_string(weights.codebook_id)},
                        {"source_codebook",std::to_string(weights.source_codebook_id)},
                        {"activation",std::to_string(static_cast<unsigned>(input.parameters.activation))}});
            return true;
        }

        bool ROCmQuantisedGemmKernel::exportNativeQuantisedMatrixDesc(DeviceNativeQuantisedMatrixDesc &out)
        {
            out = {};
            ensureWeightsConverted();

            return inspectPreparedNativeQuantisedMatrixDesc(out);
        }

        /** @copydoc ITensorGemm::inspectPreparedNativeQuantisedMatrixDesc */
        bool ROCmQuantisedGemmKernel::inspectPreparedNativeQuantisedMatrixDesc(DeviceNativeQuantisedMatrixDesc &out) const
        {
            out = {};
            if (!impl_ || !impl_->has_native_quantised ||
                !impl_->d_weights_native_quantised || !impl_->d_weights_native_scales)
            {
                return false;
            }

            out.payload = impl_->d_weights_native_quantised;
            out.scales = impl_->d_weights_native_scales;
            out.mins = impl_->d_weights_native_mins;
            out.emins = impl_->d_weights_native_emins;
            out.n = static_cast<int>(N_);
            out.k = static_cast<int>(K_);
            out.blocks_per_row = impl_->native_quantised_blocks_per_row;
            out.codebook_id = impl_->native_quantised_codebook_id;
            out.allocation_payload_bytes_per_block =
                impl_->native_allocation_format.payload_bytes_per_block;
            out.allocation_has_mins = static_cast<uint8_t>(
                impl_->native_allocation_format.has_mins);
            out.allocation_has_emins = static_cast<uint8_t>(
                impl_->native_allocation_format.has_emins);
            out.source_codebook_id = impl_->native_source_identity.codebook_id;
            out.source_is_superblock = static_cast<uint8_t>(
                impl_->native_source_identity.is_superblock);
            out.source_identity_present = static_cast<uint8_t>(
                impl_->native_source_identity.present);
            return out.valid();
        }

        bool ROCmQuantisedGemmKernel::exportNativeQuantisedSourceIdentity(
            NativeQuantisedSourceIdentity &out) const
        {
            if (!impl_ || !impl_->native_source_identity.present)
            {
                out = {};
                return false;
            }
            out = impl_->native_source_identity;
            return native_quantised_formats::forSourceIdentity(
                       out.codebook_id, out.is_superblock) != nullptr;
        }

        /**
         * @brief Classify the best prefill route for the current shape and weight format.
         *
         * This is the single source of truth for dispatch routing decisions.
         * The dispatch policy is:
         *   - ≤6-bit weights (has_native_quantised) → NATIVE_QUANTISED
         *   - 8-bit weights (d_weights_int8_vnni) → INT8_VNNI_NATIVE
         *   - Otherwise → UNSUPPORTED (no viable path)
         *
         * UNSUPPORTED is returned when impl_ is missing, dimensions are invalid,
         * or no VNNI weights are available.
         */
        ROCmQuantisedGemmKernel::PrefillDispatchPath ROCmQuantisedGemmKernel::selectPrefillDispatchPath(int m, int n, int k) const
        {
            (void)n;

            if (m <= 1 || k <= 0 || (k % 4) != 0)
            {
                return PrefillDispatchPath::UNSUPPORTED;
            }

            if (!impl_)
            {
                return PrefillDispatchPath::UNSUPPORTED;
            }

            // CK path retired. No fallback path available.

            // ≤6-bit formats use native-quantised (lossless decode, FP16 block scales)
            if (impl_->has_native_quantised)
            {
                return PrefillDispatchPath::NATIVE_QUANTISED;
            }

            // 8-bit formats use INT8-VNNI (requantized symmetric INT8)
            if (impl_->d_weights_int8_vnni != nullptr)
            {
                return PrefillDispatchPath::INT8_VNNI_NATIVE;
            }

            LOG_WARN("[ROCmQuantisedGemmKernel] No viable prefill dispatch path:"
                     " has_native_quantised="
                     << impl_->has_native_quantised
                     << " d_weights_int8_vnni=" << (const void *)impl_->d_weights_int8_vnni
                     << " M=" << m << " N=" << n << " K=" << k);
            return PrefillDispatchPath::UNSUPPORTED;
        }

        /**
         * @brief Try native M>1 prefill execution for INT8 VNNI or ratio-VNNI formats.
         *
         * A false return means the selected INT8-VNNI or native-quantised prefill
         * path could not execute. There is no retired CK fallback path; callers
         * must fail the operation instead of silently switching implementations.
         */
        bool ROCmQuantisedGemmKernel::tryPrefillNativeGemm(
            const int8_t *d_A_int8,
            float *d_output,
            const float *d_scales_A,
            const float *d_scales_A_blockwise,
            const float *d_scales_B,
            const float *d_bias,
            int m, int n, int k,
            float alpha, float beta,
            const char *callsite,
            void *stream_override,
            int32_t *scratch_int32_override)
        {
            // Use overrides when provided (concurrent prefill dispatch)
            void *effective_stream = stream_override ? stream_override : gpu_stream_;
            int32_t *effective_scratch_int32 = scratch_int32_override ? scratch_int32_override : (impl_ ? impl_->d_C_int32 : nullptr);
            const auto path_label = [&](PrefillDispatchPath p) -> const char *
            {
                switch (p)
                {
                case PrefillDispatchPath::NATIVE_QUANTISED:
                    return "native_quantised";
                case PrefillDispatchPath::INT8_VNNI_NATIVE:
                    return "int8_vnni_native";
                case PrefillDispatchPath::UNSUPPORTED:
                default:
                    return "unsupported";
                }
            };

            const auto record_path_selected = [&](PrefillDispatchPath p)
            {
                if (!WeightLoadingProfiler::isEnabled())
                    return;
                WeightLoadingProfiler::addDetail(
                    std::string("prefill_gemm.path_selected.") + path_label(p),
                    1.0);
            };

            const auto record_fallback_reason = [&](const char *reason)
            {
                if (!WeightLoadingProfiler::isEnabled())
                    return;
                WeightLoadingProfiler::addDetail(
                    std::string("prefill_gemm.fallback_reason.") + reason,
                    1.0);
            };

            const auto record_kernel_ms = [&](PrefillDispatchPath p, double ms)
            {
                if (!WeightLoadingProfiler::isEnabled())
                    return;
                WeightLoadingProfiler::addDetail(
                    std::string("prefill_gemm.kernel_ms.") + path_label(p),
                    ms);
            };

            const auto record_epilogue_ms = [&](PrefillDispatchPath p, double ms)
            {
                if (!WeightLoadingProfiler::isEnabled())
                    return;
                WeightLoadingProfiler::addDetail(
                    std::string("prefill_gemm.epilogue_ms.") + path_label(p),
                    ms);
            };

            // This helper is for prefill only. Decode (M=1) is handled
            // by the dedicated GEMV fast path and should never come through here.
            if (m <= 1)
            {
                return false;
            }

            auto logFallback = [&](const char *reason)
            {
                static std::once_flag unsupported_shape_once;
                static std::once_flag missing_buffers_once;
                static std::once_flag launch_failure_once;

                std::once_flag *selected_once = &launch_failure_once;
                if (std::strcmp(reason, "shape") == 0)
                {
                    selected_once = &unsupported_shape_once;
                }
                else if (std::strcmp(reason, "buffers") == 0)
                {
                    selected_once = &missing_buffers_once;
                }

                record_path_selected(PrefillDispatchPath::UNSUPPORTED);
                record_fallback_reason(reason);

                std::call_once(*selected_once, [&]()
                               { LOG_WARN("[" << callsite << "] Prefill GEMM unavailable (reason="
                                              << reason << ")"); });
            };

            // NativeQuantised path uses impl_->d_weights_native_scales (not d_scales_B),
            // so d_scales_B may be null when only native-quantised weights are present
            // (e.g. weights loaded via GPU pipeline / MoE batch constructor).
            const bool native_quantised_available = impl_ && impl_->has_native_quantised;
            if (!impl_ || !d_A_int8 || !d_output || !d_scales_A ||
                (!d_scales_B && !native_quantised_available))
            {
                LOG_WARN("[" << callsite << "] Prefill GEMM null pointer diagnostic:"
                                            " impl="
                             << (const void *)impl_.get()
                             << " d_A_int8=" << (const void *)d_A_int8
                             << " d_output=" << (const void *)d_output
                             << " d_scales_A=" << (const void *)d_scales_A
                             << " d_scales_B=" << (const void *)d_scales_B
                             << " scratch=" << (const void *)effective_scratch_int32
                             << " workspace=" << (const void *)workspace_);
                logFallback("buffers");
                return false;
            }

            if ((k % 4) != 0)
            {
                logFallback("shape");
                return false;
            }

            const PrefillDispatchPath path = selectPrefillDispatchPath(m, n, k);
            if (path == PrefillDispatchPath::INT8_VNNI_NATIVE &&
                !effective_scratch_int32)
            {
                logFallback("buffers");
                return false;
            }
            record_path_selected(path);
            bool native_ok = false;
            const bool profiling_enabled = debugEnv().profile.enabled;
            std::chrono::high_resolution_clock::time_point kernel_start{};
            std::chrono::high_resolution_clock::time_point kernel_end{};
            if (profiling_enabled)
                kernel_start = std::chrono::high_resolution_clock::now();

            if (path == PrefillDispatchPath::NATIVE_QUANTISED)
            {
                LOG_TRACE("[" << callsite << "] Trying native-quantised prefill (M=" << m
                              << " N=" << n << " K=" << k << ")");

                if (!impl_->d_weights_native_quantised || !impl_->d_weights_native_scales)
                {
                    LOG_WARN("[" << callsite << "] NativeQuantised prefill missing weight buffers:"
                                                " d_weights_native_quantised="
                                 << (const void *)impl_->d_weights_native_quantised
                                 << " d_weights_native_scales=" << (const void *)impl_->d_weights_native_scales);
                    logFallback("buffers");
                    return false;
                }

                if (beta != 0.0f && d_output == impl_->d_C_fp32)
                {
                    LOG_WARN("[" << callsite << "] NativeQuantised prefill beta path needs a distinct existing-output buffer");
                    logFallback("native_beta_alias");
                    return false;
                }

                float *d_native_output = (beta != 0.0f) ? impl_->d_C_fp32 : d_output;
                native_ok = rocmGemm_native_quantised_fp32_with_policy(
                    d_A_int8,
                    impl_->d_weights_native_quantised,
                    impl_->d_weights_native_scales,
                    impl_->d_weights_native_mins,
                    impl_->d_weights_native_emins,
                    d_native_output,
                    d_scales_A,
                    d_scales_A_blockwise,
                    m, n, k,
                    impl_->native_quantised_codebook_id,
                    requireNativeQuantisedArithmeticPolicyCodebook(
                        impl_->native_quantised_codebook_id,
                        impl_->native_source_identity),
                    rocm_device_id_, effective_stream);

                if (profiling_enabled)
                    kernel_end = std::chrono::high_resolution_clock::now();

                if (!native_ok)
                {
                    logFallback("launch_error");
                    return false;
                }

                if (profiling_enabled)
                {
                    record_kernel_ms(
                        path,
                        std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count());
                }

                if (beta != 0.0f)
                {
                    std::chrono::high_resolution_clock::time_point epilogue_start{};
                    if (profiling_enabled)
                        epilogue_start = std::chrono::high_resolution_clock::now();

                    if (!rocmQuantGemm_applyFp32CombineEpilogue(
                            d_native_output,
                            d_output,
                            d_output,
                            d_bias,
                            m,
                            n,
                            alpha,
                            beta,
                            rocm_device_id_,
                            effective_stream))
                    {
                        logFallback("launch_error");
                        return false;
                    }

                    if (profiling_enabled)
                    {
                        const auto epilogue_end = std::chrono::high_resolution_clock::now();
                        record_epilogue_ms(
                            path,
                            std::chrono::duration<double, std::milli>(epilogue_end - epilogue_start).count());
                    }
                }
                else if (alpha != 1.0f || d_bias)
                {
                    std::chrono::high_resolution_clock::time_point epilogue_start{};
                    if (profiling_enabled)
                        epilogue_start = std::chrono::high_resolution_clock::now();

                    if (!rocmQuantGemm_applyFp32Epilogue(
                            d_output,
                            d_bias,
                            m,
                            n,
                            alpha,
                            rocm_device_id_,
                            effective_stream))
                    {
                        logFallback("launch_error");
                        return false;
                    }

                    if (profiling_enabled)
                    {
                        const auto epilogue_end = std::chrono::high_resolution_clock::now();
                        record_epilogue_ms(
                            path,
                            std::chrono::duration<double, std::milli>(epilogue_end - epilogue_start).count());
                    }
                }

                return true;
            }

            if (path == PrefillDispatchPath::INT8_VNNI_NATIVE)
            {
                LOG_TRACE("[" << callsite << "] Trying INT8 VNNI native prefill (M=" << m
                              << " N=" << n << " K=" << k << ")");
                struct VnniPrefillLaunchConfig
                {
                    bool use_grid_kpar = false;
                    int split_k_slices = 1;
                    int tile_variant = -1; // -1 => HIP-side auto tile select
                    int cpt = 1;
                    int kernel_body_variant = 0;
                    int grid_swizzle_variant = 0;
                    const char *profile = "default";
                    bool use_wide_tile = false; // Wide-tile kernel for extreme-wide shapes (M≤128, N>>K)
                };

                const auto &rocm_env = debugEnv().rocm;
                auto select_vnni_prefill_launch = [&rocm_env](int M, int N, int K) -> VnniPrefillLaunchConfig
                {
                    const double safe_k = static_cast<double>(std::max(K, 1));
                    const double aspect_n_over_k = static_cast<double>(N) / safe_k;
                    const int64_t work = static_cast<int64_t>(M) * static_cast<int64_t>(N) * static_cast<int64_t>(K);
                    const int k_groups = std::max(1, K / 4);

                    constexpr int64_t kSmallWorkThreshold = 1200000000LL; // 0.5B-like vs 3B-like split
                    constexpr int64_t kGridKparMinWork = 300000000LL;

                    const bool is_ffn_up_like = (aspect_n_over_k >= 2.0 && aspect_n_over_k < 16.0);
                    if (is_ffn_up_like && M >= 64 && rocm_env.vnni_prefill_ffn_override)
                    {
                        const bool use_grid_kpar = (rocm_env.vnni_prefill_ffn_override_grid_kpar < 0)
                                                       ? true
                                                       : (rocm_env.vnni_prefill_ffn_override_grid_kpar != 0);
                        const int split_k_slices = (rocm_env.vnni_prefill_ffn_override_splits > 0)
                                                       ? rocm_env.vnni_prefill_ffn_override_splits
                                                       : 2;
                        const int tile_variant = (rocm_env.vnni_prefill_ffn_override_variant >= 0)
                                                     ? rocm_env.vnni_prefill_ffn_override_variant
                                                     : 1;
                        const int cpt = (rocm_env.vnni_prefill_ffn_override_cpt > 0)
                                            ? rocm_env.vnni_prefill_ffn_override_cpt
                                            : 4;
                        const int kernel_body_variant = rocm_env.vnni_prefill_ffn_override_kernel_body;
                        const int grid_swizzle_variant = (rocm_env.vnni_prefill_ffn_override_grid_swizzle >= 0)
                                                             ? rocm_env.vnni_prefill_ffn_override_grid_swizzle
                                                             : 1;

                        if (use_grid_kpar)
                        {
                            return VnniPrefillLaunchConfig{true, split_k_slices, tile_variant, cpt, kernel_body_variant, grid_swizzle_variant, "ffn_override_env_gridkpar"};
                        }
                        return VnniPrefillLaunchConfig{false, split_k_slices, tile_variant, cpt, kernel_body_variant, grid_swizzle_variant, "ffn_override_env_baseline"};
                    }

                    // Shape classes (ratio-first, then work-size split).

                    // ── Wide-tile kernel for ALL prefill shapes ──
                    // V3/V7 kernels use a 2D grid: (ceil(N/N_TILE), ceil(M/128)).
                    // Each workgroup covers up to 128 M-rows × N_TILE N-cols.
                    // Key advantages over grid-kpar:
                    //   - No hipMemsetAsync (no output zeroing needed)
                    //   - No atomicAdd (each output element written by exactly one WG)
                    //   - No extra applyScaling kernel launch
                    // V3/V7 selection (K-heavy vs N-heavy) happens in the dispatch code.
                    {
                        return VnniPrefillLaunchConfig{false, 1, 1, 1, 0, 0, "wide_tile", true};
                    }

                    // ── Legacy grid-kpar paths below are unreachable in default policy ──
                    // Kept for env-var override compatibility (vnni_prefill_ffn_override, etc.)

                    if (aspect_n_over_k >= 32.0)
                    {
                        // LM-head/extreme-wide: fallback grid-kpar (unreachable in default path)
                        return VnniPrefillLaunchConfig{true, 1, 1, 4, 2, 1, "extreme_wide_gridkpar_32x8_cpt4"};
                    }

                    if (aspect_n_over_k >= 2.0)
                    {
                        // FFN up/gate (wide): strategy-lab favored split-k style behavior.
                        if (work < kGridKparMinWork || M < 16 || k_groups < 64)
                        {
                            return VnniPrefillLaunchConfig{false, 1, 1, 1, 0, 0, "wide_guardrail_baseline_32x8_cpt1"};
                        }

                        const int slices = (work <= kSmallWorkThreshold) ? 4 : 6;
                        return VnniPrefillLaunchConfig{true, slices, 1, 4, 2, 1, "wide_gridkpar_32x8_cpt4"};
                    }

                    if (aspect_n_over_k < 0.75)
                    {
                        // FFN down (tall): keep grid-kpar with moderate split count.
                        if (work < kGridKparMinWork || M < 16 || k_groups < 64)
                        {
                            return VnniPrefillLaunchConfig{false, 1, 1, 1, 0, 0, "tall_guardrail_baseline_32x8_cpt1"};
                        }

                        const int slices = (work <= kSmallWorkThreshold) ? 4 : 6;
                        return VnniPrefillLaunchConfig{true, slices, 1, 4, 2, 1, "tall_gridkpar_32x8_cpt4"};
                    }

                    // Attention-ish / near-square.
                    {
                        const int slices = (work <= kSmallWorkThreshold) ? 4 : 6;
                        return VnniPrefillLaunchConfig{true, slices, 1, 4, 2, 1, "balanced_gridkpar_32x8_cpt4"};
                    }
                };

                const VnniPrefillLaunchConfig policy_cfg = select_vnni_prefill_launch(m, n, k);
                if (WeightLoadingProfiler::isEnabled())
                {
                    WeightLoadingProfiler::addDetail(
                        std::string("prefill_gemm.int8_policy.") + policy_cfg.profile,
                        1.0);
                    if (std::strncmp(policy_cfg.profile, "ffn_override_env_", 17) == 0)
                    {
                        WeightLoadingProfiler::addDetail("prefill_gemm.int8_policy.ffn_up_override_applied", 1.0);
                    }
                }

                // ── Fast path: wide-tile (default policy, no env overrides) ──
                // Check wide-tile eligibility BEFORE computing grid-kpar variables.
                // In the default path (no manual overrides), policy always returns
                // use_wide_tile=true, so we skip ~100ns of dead grid-kpar arithmetic.
                const bool has_manual_override =
                    (rocm_env.vnni_prefill_variant >= 0) ||
                    (rocm_env.vnni_prefill_grid_variant >= 0) ||
                    (rocm_env.vnni_prefill_grid_kpar_splits > 0) ||
                    (rocm_env.vnni_prefill_grid_kpar_kb > 0) ||
                    (rocm_env.vnni_prefill_grid_swizzle >= 0) ||
                    (rocm_env.vnni_prefill_cpt != 1) ||
                    rocm_env.vnni_prefill_grid_kpar;

                if (!has_manual_override)
                {
                    static std::once_flag vnni_prefill_policy_once;
                    std::call_once(vnni_prefill_policy_once, [&]()
                                   { LOG_DEBUG("[" << callsite << "] INT8 prefill policy enabled (ratio+work map): "
                                                   << policy_cfg.profile
                                                   << ", M=" << m << ", N=" << n << ", K=" << k
                                                   << ", N/K=" << std::fixed << std::setprecision(2)
                                                   << (static_cast<double>(n) / static_cast<double>(std::max(k, 1)))
                                                   << ")"); });
                }

                // Wide-tile path: covers all M-rows in one block (extreme-wide shapes)
                //
                // BLOCKWISE ACTIVATION SCALES: when d_scales_A_blockwise is provided,
                // use blockwise V3/V7 kernels that apply per-block activation scales
                // during accumulation, outputting FP32 directly. Then use weight-only
                // epilogue (scale_B only). K must be a multiple of 32 for block alignment.
                if (d_scales_A_blockwise != nullptr && (k % 32) == 0)
                {
                    // Reinterpret d_C_int32 buffer as float* — same 4-byte-per-element
                    // layout, device memory, no reallocation needed.
                    float *d_C_fp32_blockwise = reinterpret_cast<float *>(effective_scratch_int32);

                    const auto &bw_env = debugEnv().rocm;
                    // Data-driven dispatch (tuned on gfx906 MI50/MI60, UNROLL_KK sweep):
                    //   V7/MT64/U2 wins every shape×M in the perf sweep (1.02–1.45x
                    //   over best V3).  The old k>=n→V3 heuristic is retired.
                    //   Force overrides still take priority for experimentation.
                    // Q8 has a distinct prepared INT8 producer. Its measured
                    // key cannot enter a low-bit decoder. Explicit diagnostic
                    // controls retain precedence over the immutable overlay.
                    const bool blockwise_override = bw_env.blockwise_force_v3 ||
                        bw_env.blockwise_force_v7 || bw_env.blockwise_v3_mt >= 0 ||
                        bw_env.blockwise_v7_mt >= 0 ||
                        bw_env.blockwise_v3_unroll >= 0 ||
                        bw_env.blockwise_v7_unroll >= 0;
                    const auto exact = !has_manual_override && !blockwise_override
                        ? selectDensePrefillExactConfig(
                            19, m, n, k, DensePrefillWeightFamily::Int8Blockwise)
                        : std::nullopt;
                    const bool use_v3 = exact
                        ? exact->producer == VNNIPrefillProducer::Int8BlockwiseV3
                        : bw_env.blockwise_force_v3;

                    if (use_v3)
                    {
                        // V3 M_TILE scales with M for K>=N region (tuning data):
                        //   M <= 128 → MT16, M <= 256 → MT32, M > 256 → MT64
                        int v3_mt = exact ? exact->m_tile : bw_env.blockwise_v3_mt;
                        if (v3_mt == 0)
                        {
                            if (m <= 128)
                                v3_mt = 16;
                            else if (m <= 256)
                                v3_mt = 32;
                            else
                                v3_mt = 64;
                        }
                        native_ok = rocmQuantGemm_int8_int8_fp32_vnni_prefill_wide_tile_v3_blockwise(
                            d_A_int8,
                            impl_->d_weights_int8_vnni,
                            d_C_fp32_blockwise,
                            d_scales_A_blockwise,
                            m, n, k,
                            rocm_device_id_, effective_stream,
                            v3_mt,
                            exact ? exact->unroll : bw_env.blockwise_v3_unroll);
                    }
                    else
                    {
                        native_ok = rocmQuantGemm_int8_int8_fp32_vnni_prefill_wide_tile_v7_blockwise(
                            d_A_int8,
                            impl_->d_weights_int8_vnni,
                            d_C_fp32_blockwise,
                            d_scales_A_blockwise,
                            m, n, k,
                            rocm_device_id_, effective_stream,
                            exact ? exact->m_tile : bw_env.blockwise_v7_mt,
                            exact ? exact->unroll : bw_env.blockwise_v7_unroll);
                    }

                    if (native_ok)
                    {
                        if (profiling_enabled)
                            kernel_end = std::chrono::high_resolution_clock::now();

                        if (profiling_enabled)
                        {
                            record_kernel_ms(
                                path,
                                std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count());
                        }

                        const float *d_existing = nullptr;
                        if (beta != 0.0f)
                        {
                            if (d_output == impl_->d_C_fp32)
                            {
                                logFallback("beta_alias");
                                return false;
                            }
                            hipError_t copy_err = hipMemcpyAsync(
                                impl_->d_C_fp32,
                                d_output,
                                static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float),
                                hipMemcpyDeviceToDevice,
                                static_cast<hipStream_t>(effective_stream));
                            if (copy_err != hipSuccess)
                            {
                                LOG_ERROR("[" << callsite << "] Failed to snapshot beta existing output: "
                                              << hipGetErrorString(copy_err));
                                logFallback("launch_error");
                                return false;
                            }
                            d_existing = impl_->d_C_fp32;
                        }

                        std::chrono::high_resolution_clock::time_point epilogue_start{};
                        if (profiling_enabled)
                            epilogue_start = std::chrono::high_resolution_clock::now();

                        if (!rocmQuantGemm_applyScaling_weightOnly(
                                d_C_fp32_blockwise,
                                d_output,
                                d_scales_B,
                                m, n,
                                alpha, beta,
                                d_existing,
                                d_bias,
                                rocm_device_id_, effective_stream))
                        {
                            logFallback("launch_error");
                            return false;
                        }

                        if (profiling_enabled)
                        {
                            const auto epilogue_end = std::chrono::high_resolution_clock::now();
                            record_epilogue_ms(
                                path,
                                std::chrono::duration<double, std::milli>(epilogue_end - epilogue_start).count());
                        }

                        return true;
                    }
                    else
                    {
                        // Blockwise kernel launch failed.
                        static std::once_flag bw_fallback_once;
                        std::call_once(bw_fallback_once, [&]()
                                       { LOG_ERROR("[" << callsite << "] Blockwise INT8 kernel failed (M=" << m
                                                       << " N=" << n << " K=" << k << ")"); });
                        logFallback("launch_error");
                        return false;
                    }
                }

                if (!has_manual_override && policy_cfg.use_wide_tile)
                {
                    if (rocm_env.wide_tile_v7)
                    {
                        // V7 forced via env var: safe-tile split, branchless interior loop
                        native_ok = rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v7(
                            d_A_int8,
                            impl_->d_weights_int8_vnni,
                            effective_scratch_int32,
                            m, n, k,
                            rocm_env.wide_tile_kt,
                            rocm_device_id_, effective_stream);
                    }
                    else if (rocm_env.wide_tile_v3)
                    {
                        // V3 forced via env var: LDS double-buffered, N64
                        native_ok = rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v3(
                            d_A_int8,
                            impl_->d_weights_int8_vnni,
                            effective_scratch_int32,
                            m, n, k,
                            rocm_env.wide_tile_kt,
                            rocm_device_id_, effective_stream);
                    }
                    else if (k >= n)
                    {
                        // Auto V3 for K-heavy shapes (Attention: K==N, FFN_Down: K>>N).
                        // V3's LDS double-buffering hides global load latency in the long
                        // K-loop, giving +2-5% over V1 on these shapes.
                        // N-heavy shapes (FFN_Up/Gate, LM_Head: N > K) use V5 below.
                        native_ok = rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v3(
                            d_A_int8,
                            impl_->d_weights_int8_vnni,
                            effective_scratch_int32,
                            m, n, k,
                            16, // KT=16: best throughput for V3 auto-dispatch (perf sweep shows KT=16 wins 10/12 shapes)
                            rocm_device_id_, effective_stream);
                    }
                    else
                    {
                        // Auto V7/KT16 for N-heavy shapes (FFN_Up/Gate, LM_Head: N > K).
                        // V7's safe-tile split eliminates boundary branches from the hot
                        // loop — for aligned shapes (M=128, N%128==0, K%32==0) the boundary
                        // loop never executes, producing maximally tight ISA.
                        // Benchmark results (wallclock, 7B FFN_Up M=128 N=18944 K=3584):
                        //   V7/KT16: 7.300ms  V5/KT8: 7.397ms  CK: 7.323ms
                        // V7/KT16 matches or beats CK on all FFN_Up and LM_Head shapes.
                        native_ok = rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v7(
                            d_A_int8,
                            impl_->d_weights_int8_vnni,
                            effective_scratch_int32,
                            m, n, k,
                            16, // KT=16: best for V7 (safe-tile split, 2 waves/SIMD)
                            rocm_device_id_, effective_stream);
                    }
                    if (!native_ok)
                    {
                        static std::once_flag wide_tile_failure_once;
                        std::call_once(wide_tile_failure_once, [&]()
                                       { LOG_ERROR("[" << callsite << "] Wide-tile INT8 prefill kernel failed; refusing fallback"); });
                        logFallback("launch_error");
                        return false;
                    }
                }

                // Grid-kpar / baseline variables — only computed when wide-tile
                // didn't fire or didn't succeed, avoiding ~100ns of dead arithmetic
                // on the hot path.
                if (!native_ok)
                {
                    const bool try_grid_kpar = has_manual_override
                                                   ? rocm_env.vnni_prefill_grid_kpar
                                                   : policy_cfg.use_grid_kpar;

                    const int baseline_variant = has_manual_override ? rocm_env.vnni_prefill_variant : policy_cfg.tile_variant;
                    const int grid_variant = has_manual_override ? rocm_env.vnni_prefill_grid_variant : policy_cfg.tile_variant;
                    const int cpt = has_manual_override ? rocm_env.vnni_prefill_cpt : policy_cfg.cpt;
                    const int kernel_body_variant = policy_cfg.kernel_body_variant;
                    const int grid_swizzle_variant = (rocm_env.vnni_prefill_grid_swizzle >= 0)
                                                         ? rocm_env.vnni_prefill_grid_swizzle
                                                         : policy_cfg.grid_swizzle_variant;

                    const int resolved_variant = [&]()
                    {
                        if (try_grid_kpar)
                        {
                            return grid_variant;
                        }
                        return baseline_variant;
                    }();

                    const int tile_x = [&]()
                    {
                        if (resolved_variant == 1)
                            return 32;
                        if (resolved_variant == 2 || resolved_variant == 3)
                            return 8;
                        return 16;
                    }();

                    const int logical_tile_n = std::max(1, tile_x * std::max(1, cpt));
                    const int grid_n = std::max(1, (n + logical_tile_n - 1) / logical_tile_n);
                    const int64_t logical_tiles = static_cast<int64_t>(m) * static_cast<int64_t>(grid_n);
                    const int k_groups = std::max(1, k / 4);
                    const bool low_parallelism_guardrail = (!has_manual_override) && try_grid_kpar &&
                                                           (logical_tiles < 512 || m < 16 || k_groups < 64);

                    const bool use_grid_kpar = low_parallelism_guardrail ? false : try_grid_kpar;

                    if (low_parallelism_guardrail)
                    {
                        if (WeightLoadingProfiler::isEnabled())
                        {
                            WeightLoadingProfiler::addDetail(
                                "prefill_gemm.int8_policy.guardrail_disable_grid_kpar",
                                1.0);
                        }
                    }

                    if (use_grid_kpar)
                    {
                        const int requested_slices_env = rocm_env.vnni_prefill_grid_kpar_splits;
                        const int requested_slices = [&]()
                        {
                            const int kb_override = rocm_env.vnni_prefill_grid_kpar_kb;
                            if (kb_override > 0)
                            {
                                return kb_override;
                            }

                            if (requested_slices_env > 0)
                            {
                                return requested_slices_env;
                            }

                            if (!has_manual_override)
                            {
                                return policy_cfg.split_k_slices;
                            }

                            // Auto heuristic (Phase 4+): target enough K-groups per split
                            // while keeping a minimum level of wavefront availability.
                            const int k_groups = std::max(1, k / 4);
                            constexpr int target_groups_per_slice = 64;
                            constexpr int target_min_waves_per_cu = 8;
                            constexpr int num_cus = 60;

                            int auto_slices = std::max(1, k_groups / target_groups_per_slice);

                            const int cpt_auto = std::max(1, cpt);
                            int tile_x = 16;
                            const int variant = grid_variant;
                            if (variant == 1)
                                tile_x = 32;
                            else if (variant == 2 || variant == 3)
                                tile_x = 8;

                            const int threads_x = std::max(1, tile_x);
                            const int logical_tile_n = std::max(1, threads_x * cpt_auto);
                            const int grid_n = std::max(1, (n + logical_tile_n - 1) / logical_tile_n);
                            const int min_slices_for_waves = std::max(1, (target_min_waves_per_cu * num_cus + grid_n - 1) / grid_n);
                            auto_slices = std::max(auto_slices, min_slices_for_waves);

                            // Prefer fewer atomics for very small M while still enabling
                            // parallel K reduction for wider prefill cases.
                            if (m < 8)
                            {
                                auto_slices = std::min(auto_slices, 2);
                            }
                            else if (m >= 24 && k >= 1024)
                            {
                                auto_slices = std::max(auto_slices, 4);
                            }

                            const int n_tiles = std::max(1, (n + 127) / 128);
                            const int hard_cap = std::min(8, std::max(1, n_tiles * 2));
                            return std::clamp(auto_slices, 1, hard_cap);
                        }();
                        native_ok = rocmQuantGemm_int8_int8_int32_vnni_prefill_grid_kpar(
                            d_A_int8,
                            impl_->d_weights_int8_vnni,
                            effective_scratch_int32,
                            m, n, k,
                            requested_slices,
                            grid_variant,
                            cpt,
                            kernel_body_variant,
                            grid_swizzle_variant,
                            rocm_device_id_, effective_stream);

                        if (!native_ok)
                        {
                            static std::once_flag grid_kpar_failure_once;
                            std::call_once(grid_kpar_failure_once, [&]()
                                           { LOG_ERROR("[" << callsite << "] INT8 prefill grid-kpar launch failed; refusing baseline fallback"); });
                            logFallback("launch_error");
                            return false;
                        }
                    }

                    if (!native_ok)
                    {
                        native_ok = rocmQuantGemm_int8_int8_int32_vnni_prefill(
                            d_A_int8,
                            impl_->d_weights_int8_vnni,
                            effective_scratch_int32,
                            m, n, k,
                            baseline_variant,
                            cpt,
                            rocm_device_id_, effective_stream);
                    }
                } // end if (!native_ok) — grid-kpar / baseline fallback scope
                if (profiling_enabled)
                    kernel_end = std::chrono::high_resolution_clock::now();
            }
            else
            {
                logFallback("shape");
                return false;
            }

            if (profiling_enabled)
            {
                record_kernel_ms(
                    path,
                    std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count());
            }

            if (!native_ok)
            {
                logFallback("launch_error");
                return false;
            }

            // Epilogue: apply activation/weight scales, alpha/beta, and optional bias.
            const float *d_existing = nullptr;
            if (beta != 0.0f)
            {
                if (d_output == impl_->d_C_fp32)
                {
                    logFallback("beta_alias");
                    return false;
                }
                hipError_t copy_err = hipMemcpyAsync(
                    impl_->d_C_fp32,
                    d_output,
                    static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float),
                    hipMemcpyDeviceToDevice,
                    static_cast<hipStream_t>(effective_stream));
                if (copy_err != hipSuccess)
                {
                    LOG_ERROR("[" << callsite << "] Failed to snapshot beta existing output: "
                                  << hipGetErrorString(copy_err));
                    logFallback("launch_error");
                    return false;
                }
                d_existing = impl_->d_C_fp32;
            }

            std::chrono::high_resolution_clock::time_point epilogue_start{};
            if (profiling_enabled)
                epilogue_start = std::chrono::high_resolution_clock::now();
            if (!rocmQuantGemm_applyScaling(
                    effective_scratch_int32,
                    d_output,
                    d_scales_A,
                    d_scales_B,
                    m, n,
                    alpha, beta,
                    d_existing,
                    d_bias,
                    rocm_device_id_, effective_stream))
            {
                logFallback("launch_error");
                return false;
            }
            if (profiling_enabled)
            {
                const auto epilogue_end = std::chrono::high_resolution_clock::now();
                record_epilogue_ms(
                    path,
                    std::chrono::duration<double, std::milli>(epilogue_end - epilogue_start).count());
            }

            return true;
        }


        bool ROCmQuantisedGemmKernel::multiply_fused_verifier_rows_decode_equivalent(
            const TensorBase *input,
            const std::vector<TensorProjectionDesc> &projections,
            int m, int k,
            const IMPIContext *mpi_ctx,
            DeviceWorkspaceManager *workspace)
        {
            if (m < 1)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel] grouped verifier projection requires M>=1, got M="
                          << m);
                return false;
            }
            auto decode_equivalent_dispatch = beginVerifierDecodeEquivalentScope();
            std::optional<NativeQuantisedVerifierBundleOperation> bundle_operation;
            const auto *original_bundle = ScopedVerifierProjectionBundle::current();
            const auto *declared_rows = VerifierKernelModeScope::rowsFor(m);
            const auto original_rows = declared_rows ? *declared_rows : DeviceRowRange::fullyActive(m);
            const bool materialized_head = original_bundle &&
                original_bundle->kind() == VerifierProjectionBundleKind::MaterializedLMHead;
            const bool admitted_extent = materialized_head
                ? original_rows.captureIdentity() == DeviceRowRange::fullyActive(m).captureIdentity()
                : ScopedCapturedVerifierExtent::current().matches(DeviceId::rocm(rocm_device_id_), original_rows);
            if (original_bundle && m >= 2 && m <= 16 && workspace &&
                original_bundle->matches(DeviceId::rocm(rocm_device_id_), input, original_rows, k, projections) &&
                admitted_extent &&
                (!materialized_head || (input && input->native_type() == TensorType::FP32 && !input->isMapped())) &&
                !currentProjectionRowWork(m))
            {
                NativeQuantisedVerifierBundleOperation candidate{
                    .kind = original_bundle->kind(), .device = rocm_device_id_, .k = k,
                    .count = static_cast<int>(projections.size()), .rows = original_rows,
                    .activations = static_cast<const int8_t *>(workspace->getBuffer(GemmWorkspaceBuffers::QUANT_A)),
                    .activation_scales = static_cast<const float *>(workspace->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE)),
                    .activation_sums = static_cast<const int32_t *>(workspace->getBuffer(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE))};
                bool valid = candidate.activations && candidate.activation_scales && candidate.activation_sums &&
                    (materialized_head ? candidate.count == 1 : candidate.count >= 2) &&
                    candidate.count <= static_cast<int>(candidate.members.size());
                for (int index = 0; valid && index < candidate.count; ++index)
                {
                    const auto &projection = projections[index];
                    auto *owner = dynamic_cast<ROCmQuantisedGemmKernel *>(projection.kernel);
                    DeviceNativeQuantisedMatrixDesc descriptor;
                    const auto &slice = projection.original_output_slice;
                    // Inspection only: preparation/allocation belongs to the
                    // original stage. Unknown/mixed policies retain Dot4.
                    valid = owner && owner->rocm_device_id_ == rocm_device_id_ &&
                        owner->inspectPreparedNativeQuantisedMatrixDesc(descriptor) &&
                        projection.output && !projection.output->isMapped() && !projection.bias &&
                        (!materialized_head || projection.output->native_type() == TensorType::FP32) &&
                        descriptor.n == projection.n && descriptor.k == k && descriptor.source_identity_present &&
                        (!slice || (slice->firstColumn() == 0 && slice->physicalColumns() == slice->policyColumns()));
                    if (valid)
                    {
                        candidate.members[index] = {
                            .weights = descriptor.payload,
                            .scales = static_cast<const uint16_t *>(descriptor.scales),
                            .mins = static_cast<const uint16_t *>(descriptor.mins),
                            .emins = static_cast<const uint32_t *>(descriptor.emins),
                            .output = static_cast<float *>(projection.output->gpu_data_ptr()),
                            .n = projection.n,
                            .source = {descriptor.source_codebook_id, bool(descriptor.source_is_superblock), true}};
                        valid = candidate.members[index].output != nullptr;
                    }
                }
                if (valid) bundle_operation = candidate;
            }
            std::optional<NativeQuantisedGDNVerifierOperation> gdn_operation;
            const auto *gdn_subgroup = ScopedVerifierGDNSubgroup::current();
            if (gdn_subgroup && input && input->native_type()==TensorType::FP32 && m >= 2 && m <= 16 && workspace &&
                gdn_subgroup->matches(DeviceId::rocm(rocm_device_id_),input,original_rows,k,projections) &&
                ScopedCapturedVerifierExtent::current().matches(DeviceId::rocm(rocm_device_id_),original_rows) &&
                !currentProjectionRowWork(m))
            {
                const auto &original = gdn_subgroup->original();
                const auto all = original.projections();
                NativeQuantisedGDNVerifierOperation candidate{
                    .device=rocm_device_id_,.k=k,.input=input,.rows=original_rows,
                    .precision=original.precision(),.correction_rows=original.correctionRows(),
                    .activations=static_cast<const int8_t *>(workspace->getBuffer(GemmWorkspaceBuffers::QUANT_A)),
                    .activation_scales=static_cast<const float *>(workspace->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE)),
                    .activation_sums=static_cast<const int32_t *>(workspace->getBuffer(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE))};
                bool valid = candidate.activations && candidate.activation_scales && candidate.activation_sums;
                const std::array native_members{&candidate.qkv,&candidate.z};
                const std::array controls{&candidate.alpha,&candidate.beta};
                for (int role=0;valid && role<4;++role)
                {
                    const auto &projection=all[role];
                    const auto &slice=projection.original_output_slice;
                    valid=projection.output && projection.output->native_type()==TensorType::FP32 && !projection.output->isMapped() && !projection.bias &&
                        (!slice || (slice->firstColumn()==0 && slice->physicalColumns()==slice->policyColumns()));
                    if(!valid)break;
                    auto *output=static_cast<float *>(projection.output->gpu_data_ptr());
                    if(role<2)
                    {
                        auto *owner=dynamic_cast<ROCmQuantisedGemmKernel *>(projection.kernel);
                        DeviceNativeQuantisedMatrixDesc descriptor;
                        valid=owner && owner->rocm_device_id_==rocm_device_id_ && output &&
                            owner->inspectPreparedNativeQuantisedMatrixDesc(descriptor) &&
                            descriptor.n==projection.n && descriptor.k==k && descriptor.source_identity_present;
                        if(valid)*native_members[role]={.weights=descriptor.payload,
                            .scales=static_cast<const uint16_t *>(descriptor.scales),
                            .mins=static_cast<const uint16_t *>(descriptor.mins),
                            .emins=static_cast<const uint32_t *>(descriptor.emins),.output=output,.n=projection.n,
                            .source={descriptor.source_codebook_id,bool(descriptor.source_is_superblock),true}};
                    }
                    else
                    {
                        auto *owner=dynamic_cast<ROCmFloatingPointGemmKernel *>(projection.kernel);
                        ContiguousFloatingPointWeightDescriptor descriptor;
                        valid=owner && owner->rocm_device_id()==rocm_device_id_ && output &&
                            owner->exportContiguousFloatingPointWeights(descriptor) && descriptor.valid() &&
                            descriptor.n==projection.n && descriptor.k==k;
                        if(valid)*controls[role-2]={.engine=owner,.weights=descriptor,.output=output,
                            .precision=owner->floatingActivationOperandPrecision()};
                    }
                }
                if(valid)gdn_operation=candidate;
            }
            // Even an unmeasured nested API hides the enclosing native bundle.
            const ScopedNativeQuantisedVerifierBundle original_native_bundle(bundle_operation);
            const ScopedNativeQuantisedGDNVerifier original_native_gdn(gdn_operation);
            std::optional<NativeQuantisedVerifierOperation> operation;
            if (projections.size() == 1)
            {
                const auto &projection = projections.front();
                auto *owner = dynamic_cast<ROCmQuantisedGemmKernel *>(projection.kernel);
                const auto &slice = projection.original_output_slice;
                const int policy_n = rocmGemv_native_quantised_get_serial_partition_n();
                if (owner && owner->impl_ && owner->impl_->has_native_quantised &&
                    projection.output && !projection.output->isMapped() && !projection.bias &&
                    owner->rocm_device_id_ == rocm_device_id_ && projection.n == static_cast<int>(owner->N_) &&
                    k == static_cast<int>(owner->K_) && (policy_n <= 0 || policy_n == projection.n) &&
                    (!slice || (slice->firstColumn() == 0 && slice->physicalColumns() == slice->policyColumns())))
                    operation = NativeQuantisedVerifierOperation{NativeQuantisedVerifierOperationKind::Projection,
                        rocm_device_id_, m, projection.n, k, owner->impl_->d_weights_native_quantised,
                        owner->impl_->native_source_identity};
            }
            const ScopedNativeQuantisedVerifierOperation original_operation(operation);
            return multiply_fused_tensor(input, projections, m, k, mpi_ctx, workspace);
        }

        bool ROCmQuantisedGemmKernel::multiply_activations(
            const float *A, const float *B, float *C,
            int m, int n, int k,
            bool transpose_B,
            float alpha, float beta,
            const IMPIContext *mpi_ctx,
            int device_idx)
        {
            // Decode work admission never changes the selected arithmetic program.
            if (m > 1 && currentProjectionRowWork(m) && !VerifierKernelModeScope::active())
                throw std::invalid_argument("Projection row intervals require explicit grouped Decode arithmetic for M>1");

            // Activation-activation GEMM is not supported by quantized kernel
            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_activations] Not supported - use dedicated attention kernel");
            return false;
        }

        bool ROCmQuantisedGemmKernel::multiply_activations_strided(
            const float *A, const float *B, float *C,
            int m, int n, int k,
            int lda, int ldb, int ldc,
            bool transpose_B,
            float alpha, float beta,
            const IMPIContext *mpi_ctx,
            int device_idx)
        {
            // Decode work admission never changes the selected arithmetic program.
            if (m > 1 && currentProjectionRowWork(m) && !VerifierKernelModeScope::active())
                throw std::invalid_argument("Projection row intervals require explicit grouped Decode arithmetic for M>1");

            // Strided activation GEMM is not supported
            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_activations_strided] Not supported");
            return false;
        }

        bool ROCmQuantisedGemmKernel::supports_device(int device_idx) const
        {
            // Supports any ROCm GPU device
            return device_idx >= 0;
        }

        KernelSnapshotInfo ROCmQuantisedGemmKernel::getKernelSnapshotInfo() const
        {
            return KernelSnapshotInfo::gemm()
                .withInput("A", "input activations [m, k]", KernelBufferDtype::FP32)
                .withWeight("B", "quantized weight matrix [n, k] (converted to INT8)", KernelBufferDtype::INT8)
                .withOutput("C", "output matrix [m, n]", KernelBufferDtype::FP32)
                .withScalar("N", "output features", KernelBufferDtype::INT32)
                .withScalar("K", "input features", KernelBufferDtype::INT32)
                .withScalar("rocm_device_id", "ROCm device ID", KernelBufferDtype::INT32)
                .withScalar("weights_converted", "whether weights are converted to INT8", KernelBufferDtype::INT32);
        }

        // =====================================================================
        // IWorkspaceConsumer Interface Implementation
        // =====================================================================

        WorkspaceRequirements ROCmQuantisedGemmKernel::getWorkspaceRequirements(
            int m, int n, int k) const
        {
            // Use internal dimensions if not specified
            if (n == 0)
                n = static_cast<int>(N_);
            if (k == 0)
                k = static_cast<int>(K_);
            WorkspaceRequirements reqs =
                quantized_gemm_workspace::projectionRequirements(m, n, k);

            LOG_TRACE("[ROCmQuantisedGemmKernel::getWorkspaceRequirements] INT8 path: "
                      << "quant_a="
                      << (reqs.find(GemmWorkspaceBuffers::QUANT_A)->size_bytes / 1024)
                      << "KB, scales_a="
                      << reqs.find(GemmWorkspaceBuffers::SCALES_A)->size_bytes
                      << "B, scales_a_blockwise="
                      << reqs.find(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE)->size_bytes
                      << "B, acc="
                      << (reqs.find(GemmWorkspaceBuffers::ACC_INT32)->size_bytes / 1024)
                      << "KB");

            return reqs;
        }

        void ROCmQuantisedGemmKernel::appendFusedProjectionWorkspaceRequirements(
            WorkspaceRequirements &requirements,
            int m,
            std::span<const int> projection_columns,
            int k) const
        {
            (void)k;
            quantized_gemm_workspace::appendFusedProjectionRequirements(
                requirements, m, projection_columns);
        }

        void ROCmQuantisedGemmKernel::bindWorkspace(DeviceWorkspaceManager *workspace)
        {
            workspace_ = workspace;
            // Invalidate workspace validation cache so next validateWorkspace() re-populates
            if (impl_)
            {
                impl_->validated_workspace = nullptr;
            }
            if (workspace)
            {
                LOG_TRACE("[ROCmQuantisedGemmKernel] Bound workspace manager at " << (void *)workspace
                                                                                  << ", entering managed mode");
            }
            else
            {
                LOG_TRACE("[ROCmQuantisedGemmKernel] Unbound workspace, returning to legacy mode");
            }
        }

        bool ROCmQuantisedGemmKernel::hasWorkspace() const
        {
            return workspace_ != nullptr;
        }

        DeviceWorkspaceManager *ROCmQuantisedGemmKernel::getWorkspace() const
        {
            return workspace_;
        }

        // =====================================================================
        // Internal methods - STUBS
        // =====================================================================

        /**
         * @brief Materialize host/prepacked weights and their destination resources.
         * @details Resident constructors publish their IQ constants before exposing
         * the engine. Host-backed weights publish IQ constants here before uploading
         * weights. Pure CPU packing therefore needs no live GPU and replay retains
         * the existing immediate weights_converted_ fast path.
         * @throws std::runtime_error If destination IQ table publication fails.
         */
        void ROCmQuantisedGemmKernel::ensureWeightsConverted()
        {
            if (weights_converted_)
            {
                return; // Already converted and uploaded
            }

            const auto source_codebook = impl_->native_source_identity.codebook_id;
            if (source_codebook >= 11 && source_codebook <= 17 &&
                !ensureIQGridTablesInitialized(rocm_device_id_))
            {
                throw std::runtime_error(
                    "ROCm weight preparation could not publish destination IQ tables");
            }

            // Path 1: Pre-packed weights (ROCmPackedWeights* passed to constructor)
            if (packed_)
            {
                std::lock_guard<std::mutex> lock(packed_->upload_mutex);
                auto upload_it = packed_->device_uploads.find(rocm_device_id_);

                if (upload_it == packed_->device_uploads.end())
                {
                    ROCmPackedWeights::DeviceUpload upload;

                    // Upload only VNNI layout + scales to device.
                    rocmQuantGemm_setDevice(rocm_device_id_);

                    // Create a dedicated H2D upload stream for this weight.
                    // Using an explicit stream instead of the NULL stream avoids
                    // contention when multiple upload threads work on the same
                    // device concurrently, and the async path's pinned staging
                    // (hipHostMalloc + memcpy + hipMemcpyAsync) naturally works
                    // around ROCm hipMemcpy failures on pageable brk-heap memory.
                    bool async_upload_enabled = false;
                    hipStream_t h2d_upload_stream = nullptr;
#ifdef HAVE_ROCM
                    {
                        hipError_t stream_err = hipStreamCreateWithFlags(&h2d_upload_stream, hipStreamNonBlocking);
                        if (stream_err == hipSuccess)
                        {
                            upload.startup_h2d_stream = reinterpret_cast<void *>(h2d_upload_stream);
                            async_upload_enabled = true;
                        }
                        else
                        {
                            throw std::runtime_error(
                                std::string("[ROCmQuantisedGemmKernel] Failed to create explicit H2D stream: ") +
                                hipGetErrorString(stream_err));
                        }
                    }
#endif

                    auto cleanup_startup_async_resources = [&upload, &h2d_upload_stream]()
                    {
#ifdef HAVE_ROCM
                        freeStartupPinnedStaging(upload);
                        if (upload.startup_h2d_stream)
                        {
                            (void)hipStreamDestroy(reinterpret_cast<hipStream_t>(upload.startup_h2d_stream));
                        }
                        upload.startup_h2d_stream = nullptr;
                        h2d_upload_stream = nullptr;
#endif
                    };
                    auto cleanup_startup_device_upload = [&upload, this]()
                    {
                        if (upload.d_int8_data_vnni)
                        {
                            rocmQuantGemm_freeDevice(upload.d_int8_data_vnni, rocm_device_id_);
                            upload.d_int8_data_vnni = nullptr;
                        }
                        if (upload.d_scales)
                        {
                            rocmQuantGemm_freeDevice(upload.d_scales, rocm_device_id_);
                            upload.d_scales = nullptr;
                        }
                        if (upload.d_native_quantised_payload)
                        {
                            rocmQuantGemm_freeDevice(upload.d_native_quantised_payload, rocm_device_id_);
                            upload.d_native_quantised_payload = nullptr;
                        }
#ifdef HAVE_ROCM
                        if (upload.d_native_quantised_scales)
                        {
                            rocmQuantGemm_freeDevice(
                                upload.d_native_quantised_scales, rocm_device_id_);
                            upload.d_native_quantised_scales = nullptr;
                        }
                        if (upload.d_native_quantised_mins)
                        {
                            rocmQuantGemm_freeDevice(
                                upload.d_native_quantised_mins, rocm_device_id_);
                            upload.d_native_quantised_mins = nullptr;
                        }
                        if (upload.d_native_quantised_emins)
                        {
                            rocmQuantGemm_freeDevice(
                                upload.d_native_quantised_emins, rocm_device_id_);
                            upload.d_native_quantised_emins = nullptr;
                        }
#endif
                    };

                    {
                        ScopedWeightLoadDetailTimer h2d_timer("weights.gemm_pack.h2d_stage");

                        if (!rocmQuantGemm_allocFloat(&upload.d_scales,
                                                      static_cast<size_t>(packed_->N),
                                                      rocm_device_id_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc scales");
                            cleanup_startup_async_resources();
                            return;
                        }

                        hipError_t err = startupMemcpyAsyncOrSync(
                            upload.d_scales,
                            packed_->scales.data(),
                            static_cast<size_t>(packed_->N) * sizeof(float),
                            async_upload_enabled,
                            upload.startup_h2d_stream,
                            &upload.startup_h2d_pinned_scales,
                            "ROCmQuantisedGemmKernel::ensureWeightsConverted",
                            "scales");
                        if (err != hipSuccess)
                        {
                            rocmQuantGemm_freeDevice(upload.d_scales, rocm_device_id_);
                            upload.d_scales = nullptr;
                            cleanup_startup_async_resources();
                            LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload scales: "
                                      << hipGetErrorString(err));
                            return;
                        }

                        // Upload VNNI layout (sole weight copy on device)
                        if (!packed_->int8_data_vnni.empty())
                        {
                            if (!rocmQuantGemm_allocInt8(&upload.d_int8_data_vnni,
                                                         packed_->int8_data_vnni.size(),
                                                         rocm_device_id_))
                            {
                                rocmQuantGemm_freeDevice(upload.d_scales, rocm_device_id_);
                                upload.d_scales = nullptr;
                                cleanup_startup_async_resources();
                                LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc VNNI weights");
                                return;
                            }

                            err = startupMemcpyAsyncOrSync(
                                upload.d_int8_data_vnni,
                                packed_->int8_data_vnni.data(),
                                packed_->int8_data_vnni.size() * sizeof(int8_t),
                                async_upload_enabled,
                                upload.startup_h2d_stream,
                                &upload.startup_h2d_pinned_vnni,
                                "ROCmQuantisedGemmKernel::ensureWeightsConverted",
                                "vnni");
                            if (err != hipSuccess)
                            {
                                rocmQuantGemm_freeDevice(upload.d_int8_data_vnni, rocm_device_id_);
                                upload.d_int8_data_vnni = nullptr;
                                rocmQuantGemm_freeDevice(upload.d_scales, rocm_device_id_);
                                upload.d_scales = nullptr;
                                cleanup_startup_async_resources();
                                LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload VNNI weights: "
                                          << hipGetErrorString(err)
                                          << " (device=" << rocm_device_id_
                                          << " N=" << packed_->N << " K=" << packed_->K
                                          << " vnni_size=" << packed_->int8_data_vnni.size()
                                          << " src=" << (void *)packed_->int8_data_vnni.data()
                                          << " uploads=" << packed_->device_uploads.size() << ")");
                                return;
                            }

                            LOG_DEBUG("[ROCmQuantisedGemmKernel] Uploaded VNNI weights (Option B, sole device copy): "
                                      << (packed_->int8_data_vnni.size() / 1024) << " KB");
                        }

                        // Row-major weights are NOT uploaded here.  The CK prefill
                        // path repacks VNNI→row-major on-demand into the shared
                        // workspace scratch buffer (d_B_rowmajor_scratch), which
                        // reuses a single allocation for all GEMM calls.  Uploading
                        // a persistent row-major copy per weight would double GPU
                        // memory usage (VNNI + row-major) and OOM on large models.

                        // Upload native-quantised payload + scales + mins (Q4_0, Q4_1, Q5_0, Q5_1,
                        // Q6_K, Q3_K, Q2_K, Q4_K, Q5_K, IQ4_NL, IQ4_XS, IQ3_S, IQ3_XXS,
                        // IQ2_S, IQ2_XS, IQ2_XXS, IQ1_S, IQ1_M)
                        if (!packed_->native_quantised_payload.empty() && !packed_->native_quantised_scales.empty())
                        {
                            // Allocate payload buffer
                            if (!rocmQuantGemm_allocInt8(reinterpret_cast<int8_t **>(&upload.d_native_quantised_payload),
                                                         packed_->native_quantised_payload.size(),
                                                         rocm_device_id_))
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc native-quantised payload; refusing alternate GEMM path");
                                cleanup_startup_device_upload();
                                cleanup_startup_async_resources();
                                return;
                            }
                            else
                            {
                                err = startupMemcpyAsyncOrSync(
                                    upload.d_native_quantised_payload,
                                    packed_->native_quantised_payload.data(),
                                    packed_->native_quantised_payload.size() * sizeof(uint8_t),
                                    async_upload_enabled,
                                    upload.startup_h2d_stream,
                                    &upload.startup_h2d_pinned_native_quantised,
                                    "ROCmQuantisedGemmKernel::ensureWeightsConverted",
                                    "native_quantised_payload");
                                if (err != hipSuccess)
                                {
                                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload native-quantised payload: "
                                              << hipGetErrorString(err)
                                              << "; refusing alternate GEMM path");
                                    cleanup_startup_device_upload();
                                    cleanup_startup_async_resources();
                                    return;
                                }
                            }

                            // Allocate FP16 scales buffer
                            if (upload.d_native_quantised_payload)
                            {
                                const size_t scales_bytes = packed_->native_quantised_scales.size() * sizeof(uint16_t);
                                void *d_scales_tmp = nullptr;
                                if (!rocmQuantGemm_allocInt8(
                                        reinterpret_cast<int8_t **>(&d_scales_tmp),
                                        scales_bytes,
                                        rocm_device_id_))
                                {
                                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to allocate native-quantised scales");
                                    cleanup_startup_device_upload();
                                    cleanup_startup_async_resources();
                                    return;
                                }
                                upload.d_native_quantised_scales = d_scales_tmp;
                                if (d_scales_tmp)
                                {
                                    err = startupMemcpyAsyncOrSync(
                                        d_scales_tmp,
                                        packed_->native_quantised_scales.data(),
                                        scales_bytes,
                                        async_upload_enabled,
                                        upload.startup_h2d_stream,
                                        &upload.startup_h2d_pinned_native_scales,
                                        "ROCmQuantisedGemmKernel::ensureWeightsConverted",
                                        "native_quantised_scales");
                                    if (err != hipSuccess)
                                    {
                                        LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload native-quantised scales: "
                                                  << hipGetErrorString(err)
                                                  << "; refusing alternate GEMM path");
                                        cleanup_startup_device_upload();
                                        cleanup_startup_async_resources();
                                        return;
                                    }
                                }
                            }

                            if (upload.d_native_quantised_payload && upload.d_native_quantised_scales)
                            {
                                LOG_DEBUG("[ROCmQuantisedGemmKernel] Uploaded native-quantised payload="
                                          << (packed_->native_quantised_payload.size() / 1024) << " KB scales="
                                          << (packed_->native_quantised_scales.size() * 2 / 1024) << " KB");
                            }

                            // Allocate FP16 mins buffer (asymmetric formats: Q4_1, Q5_1)
                            if (upload.d_native_quantised_payload && !packed_->native_quantised_mins.empty())
                            {
                                const size_t mins_bytes = packed_->native_quantised_mins.size() * sizeof(uint16_t);
                                void *d_mins_tmp = nullptr;
                                if (!rocmQuantGemm_allocInt8(
                                        reinterpret_cast<int8_t **>(&d_mins_tmp),
                                        mins_bytes,
                                        rocm_device_id_))
                                {
                                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to allocate native-quantised mins");
                                    cleanup_startup_device_upload();
                                    cleanup_startup_async_resources();
                                    return;
                                }
                                upload.d_native_quantised_mins = d_mins_tmp;
                                if (d_mins_tmp)
                                {
                                    err = startupMemcpyAsyncOrSync(
                                        d_mins_tmp,
                                        packed_->native_quantised_mins.data(),
                                        mins_bytes,
                                        async_upload_enabled,
                                        upload.startup_h2d_stream,
                                        &upload.startup_h2d_pinned_native_mins,
                                        "ROCmQuantisedGemmKernel::ensureWeightsConverted",
                                        "native_quantised_mins");
                                    if (err != hipSuccess)
                                    {
                                        LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload native-quantised mins: "
                                                  << hipGetErrorString(err)
                                                  << "; refusing alternate GEMM path");
                                        cleanup_startup_device_upload();
                                        cleanup_startup_async_resources();
                                        return;
                                    }
                                    else
                                    {
                                        LOG_DEBUG("[ROCmQuantisedGemmKernel] Uploaded native-quantised mins="
                                                  << (mins_bytes / 1024) << " KB");
                                    }
                                }
                            }

                            // Allocate uint32 emins buffer (Q2_K only: packed {emin_lo, emin_hi} FP16)
                            if (upload.d_native_quantised_payload && !packed_->native_quantised_emins.empty())
                            {
                                const size_t emins_bytes = packed_->native_quantised_emins.size() * sizeof(uint32_t);
                                void *d_emins_tmp = nullptr;
                                if (!rocmQuantGemm_allocInt8(
                                        reinterpret_cast<int8_t **>(&d_emins_tmp),
                                        emins_bytes,
                                        rocm_device_id_))
                                {
                                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to allocate native-quantised emins");
                                    cleanup_startup_device_upload();
                                    cleanup_startup_async_resources();
                                    return;
                                }
                                upload.d_native_quantised_emins = d_emins_tmp;
                                if (d_emins_tmp)
                                {
                                    err = startupMemcpyAsyncOrSync(
                                        d_emins_tmp,
                                        packed_->native_quantised_emins.data(),
                                        emins_bytes,
                                        async_upload_enabled,
                                        upload.startup_h2d_stream,
                                        &upload.startup_h2d_pinned_native_emins,
                                        "ROCmQuantisedGemmKernel::ensureWeightsConverted",
                                        "native_quantised_emins");
                                    if (err != hipSuccess)
                                    {
                                        LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload native-quantised emins: "
                                                  << hipGetErrorString(err)
                                                  << "; refusing alternate GEMM path");
                                        cleanup_startup_device_upload();
                                        cleanup_startup_async_resources();
                                        return;
                                    }
                                    else
                                    {
                                        LOG_DEBUG("[ROCmQuantisedGemmKernel] Uploaded native-quantised emins="
                                                  << (emins_bytes / 1024) << " KB");
                                    }
                                }
                            }
                        }

                        if (packed_->int8_data_vnni.empty() && packed_->native_quantised_payload.empty())
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel] No VNNI layout available; refusing implicit CPU or alternate packing path.");
                            cleanup_startup_device_upload();
                            cleanup_startup_async_resources();
                            return;
                        }

                        // Synchronize and destroy the per-weight H2D stream.
                        // All hipMemcpyAsync calls on this stream must complete
                        // before we commit the upload and release pinned staging.
#ifdef HAVE_ROCM
                        if (h2d_upload_stream)
                        {
                            (void)hipStreamSynchronize(h2d_upload_stream);
                            (void)hipStreamDestroy(h2d_upload_stream);
                            h2d_upload_stream = nullptr;
                            upload.startup_h2d_stream = nullptr;
                        }
#endif
                        // Free pinned staging buffers now that the stream is synced
                        freeStartupPinnedStaging(upload);
                    }

                    {
                        ScopedWeightLoadDetailTimer commit_timer("weights.gemm_pack.commit_publish");
                        auto emplaced = packed_->device_uploads.emplace(rocm_device_id_, upload);
                        upload_it = emplaced.first;
                    }

                    LOG_DEBUG("[ROCmQuantisedGemmKernel] Uploaded pre-packed weights to ROCm:" << rocm_device_id_
                                                                                               << " " << packed_->N << "x" << packed_->K
                                                                                               << " vnni=" << (packed_->int8_data_vnni.size() / 1024) << " KB"
                                                                                               << " (VNNI-only)");
                }

                {
                    ScopedWeightLoadDetailTimer commit_timer("weights.gemm_pack.commit_publish");
                    auto &upload = upload_it->second;
                    packed_->d_int8_data_vnni = upload.d_int8_data_vnni;
                    packed_->d_scales = upload.d_scales;
                    packed_->d_native_quantised_payload = upload.d_native_quantised_payload;
                    packed_->d_native_quantised_scales = upload.d_native_quantised_scales;
                    packed_->d_native_quantised_mins = upload.d_native_quantised_mins;
                    packed_->d_native_quantised_emins = upload.d_native_quantised_emins;
                    packed_->uploaded = true;
                    packed_->rocm_device_id = rocm_device_id_;

                    // Point impl_ to packed_ device pointers
                    impl_->d_weights_int8_vnni = upload.d_int8_data_vnni;
                    impl_->d_scales_B = upload.d_scales;
                    impl_->d_weights_native_quantised = upload.d_native_quantised_payload;
                    impl_->d_weights_native_scales = upload.d_native_quantised_scales;
                    impl_->d_weights_native_mins = upload.d_native_quantised_mins;
                    impl_->d_weights_native_emins = upload.d_native_quantised_emins;
                    impl_->native_quantised_codebook_id = packed_->native_quantised_codebook_id;
                    impl_->native_quantised_blocks_per_row = packed_->native_quantised_blocks_per_row;
                    impl_->native_source_identity = packed_->native_source_identity;
                    impl_->has_native_quantised = (upload.d_native_quantised_payload != nullptr && upload.d_native_quantised_scales != nullptr);
                    impl_->startup_h2d_pinned_scales = upload.startup_h2d_pinned_scales;
                    impl_->startup_h2d_pinned_vnni = upload.startup_h2d_pinned_vnni;
                    impl_->startup_h2d_pinned_native_quantised = upload.startup_h2d_pinned_native_quantised;
                    impl_->startup_h2d_pinned_native_scales = upload.startup_h2d_pinned_native_scales;
                    impl_->startup_h2d_pinned_native_mins = upload.startup_h2d_pinned_native_mins;
                    impl_->startup_h2d_pinned_native_emins = upload.startup_h2d_pinned_native_emins;
                }
                weights_converted_ = true;

                // Release host-side packing buffers — data is now on GPU.
                // This saves ~2× model_size of host memory for large models.
                // Only safe when this packed_ won't be uploaded to additional devices;
                // we check device_uploads.size() == 1 as a proxy (TP shards have
                // separate packed_ per shard, so this is typically the only device).
                if (packed_->device_uploads.size() <= 1)
                {
                    const size_t freed_bytes =
                        packed_->int8_data.capacity() +
                        packed_->int8_data_vnni.capacity() +
                        packed_->scales.capacity() * sizeof(float) +
                        packed_->native_quantised_payload.capacity() +
                        packed_->native_quantised_scales.capacity() * sizeof(uint16_t) +
                        packed_->native_quantised_mins.capacity() * sizeof(uint16_t);
                    packed_->int8_data.clear();
                    packed_->int8_data.shrink_to_fit();
                    packed_->int8_data_vnni.clear();
                    packed_->int8_data_vnni.shrink_to_fit();
                    packed_->scales.clear();
                    packed_->scales.shrink_to_fit();
                    packed_->native_quantised_payload.clear();
                    packed_->native_quantised_payload.shrink_to_fit();
                    packed_->native_quantised_scales.clear();
                    packed_->native_quantised_scales.shrink_to_fit();
                    packed_->native_quantised_mins.clear();
                    packed_->native_quantised_mins.shrink_to_fit();
                    if (freed_bytes > 0)
                    {
                        LOG_DEBUG("[ROCmQuantisedGemmKernel] Released host packing buffers: "
                                  << (freed_bytes / (1024 * 1024)) << " MB");
                    }
                }
                return;
            }

            // Path 2: Legacy path - convert from TensorBase*
            if (!weights_)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel] No weights tensor or packed weights!");
                return;
            }

            // Pack on host
            ROCmPackedWeights host_packed;
            if (!packWeightsToROCm(weights_, host_packed))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to pack weights");
                return;
            }

            // Legacy path: upload the packed layouts produced by packWeightsToROCm.
            // INT8 formats provide VNNI + scales; native-quantised formats provide
            // payload + per-block metadata for the dedicated native kernels.
            rocmQuantGemm_setDevice(rocm_device_id_);

            // Upload scales
            if (!rocmQuantGemm_allocFloat(&impl_->d_scales_B,
                                          static_cast<size_t>(host_packed.N),
                                          rocm_device_id_))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc scales");
                return;
            }

            hipError_t err = hipMemcpy(impl_->d_scales_B,
                                       host_packed.scales.data(),
                                       static_cast<size_t>(host_packed.N) * sizeof(float),
                                       hipMemcpyHostToDevice);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload scales: "
                          << hipGetErrorString(err));
                return;
            }

            // Upload VNNI layout
            if (!host_packed.int8_data_vnni.empty())
            {
                if (!rocmQuantGemm_allocInt8(&impl_->d_weights_int8_vnni,
                                             host_packed.int8_data_vnni.size(),
                                             rocm_device_id_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc VNNI weights");
                    return;
                }

                err = hipMemcpy(impl_->d_weights_int8_vnni,
                                host_packed.int8_data_vnni.data(),
                                host_packed.int8_data_vnni.size() * sizeof(int8_t),
                                hipMemcpyHostToDevice);
                if (err != hipSuccess)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload VNNI weights: "
                              << hipGetErrorString(err));
                    return;
                }
            }

            // Upload native-quantised payload + metadata when present.
            if (!host_packed.native_quantised_payload.empty() && !host_packed.native_quantised_scales.empty())
            {
                if (!rocmQuantGemm_allocInt8(reinterpret_cast<int8_t **>(&impl_->d_weights_native_quantised),
                                             host_packed.native_quantised_payload.size(),
                                             rocm_device_id_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to alloc native-quantised payload");
                    return;
                }

                err = hipMemcpy(impl_->d_weights_native_quantised,
                                host_packed.native_quantised_payload.data(),
                                host_packed.native_quantised_payload.size() * sizeof(uint8_t),
                                hipMemcpyHostToDevice);
                if (err != hipSuccess)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload native-quantised payload: "
                              << hipGetErrorString(err));
                    return;
                }

                const size_t native_scales_bytes = host_packed.native_quantised_scales.size() * sizeof(uint16_t);
                if (!rocmQuantGemm_allocInt8(
                        reinterpret_cast<int8_t **>(&impl_->d_weights_native_scales),
                        native_scales_bytes,
                        rocm_device_id_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to allocate native-quantised scales");
                    return;
                }

                err = hipMemcpy(impl_->d_weights_native_scales,
                                host_packed.native_quantised_scales.data(),
                                native_scales_bytes,
                                hipMemcpyHostToDevice);
                if (err != hipSuccess)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload native-quantised scales: "
                              << hipGetErrorString(err));
                    return;
                }

                if (!host_packed.native_quantised_mins.empty())
                {
                    const size_t native_mins_bytes = host_packed.native_quantised_mins.size() * sizeof(uint16_t);
                    if (!rocmQuantGemm_allocInt8(
                            reinterpret_cast<int8_t **>(&impl_->d_weights_native_mins),
                            native_mins_bytes,
                            rocm_device_id_))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to allocate native-quantised mins");
                        return;
                    }

                    err = hipMemcpy(impl_->d_weights_native_mins,
                                    host_packed.native_quantised_mins.data(),
                                    native_mins_bytes,
                                    hipMemcpyHostToDevice);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload native-quantised mins: "
                                  << hipGetErrorString(err));
                        return;
                    }
                }

                if (!host_packed.native_quantised_emins.empty())
                {
                    const size_t native_emins_bytes = host_packed.native_quantised_emins.size() * sizeof(uint32_t);
                    if (!rocmQuantGemm_allocInt8(
                            reinterpret_cast<int8_t **>(&impl_->d_weights_native_emins),
                            native_emins_bytes,
                            rocm_device_id_))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to allocate native-quantised emins");
                        return;
                    }

                    err = hipMemcpy(impl_->d_weights_native_emins,
                                    host_packed.native_quantised_emins.data(),
                                    native_emins_bytes,
                                    hipMemcpyHostToDevice);
                    if (err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to upload native-quantised emins: "
                                  << hipGetErrorString(err));
                        return;
                    }
                }

                impl_->native_quantised_codebook_id = host_packed.native_quantised_codebook_id;
                impl_->native_quantised_blocks_per_row = host_packed.native_quantised_blocks_per_row;
                impl_->native_source_identity = host_packed.native_source_identity;
            }

            impl_->has_native_quantised = (impl_->d_weights_native_quantised != nullptr &&
                                      impl_->d_weights_native_scales != nullptr);

            impl_->owns_weight_memory = true; // We now own the device memory
            weights_converted_ = true;

            LOG_DEBUG("[ROCmQuantisedGemmKernel] Converted and uploaded weights: "
                      << N_ << "x" << K_);
        }

        void ROCmQuantisedGemmKernel::validateWorkspace() const
        {
            // =========================================================================
            // Fast path: skip re-validation if same workspace was already validated.
            // During steady-state inference, the workspace pointer never changes,
            // so this avoids ~20 unordered_map<string> lookups per GEMM call.
            // (~2260 lookups/token × ~100ns each = ~0.2ms/token savings)
            // =========================================================================
            if (workspace_ && impl_->validated_workspace == workspace_)
            {
                return;
            }

            // =========================================================================
            // Workspace is REQUIRED - no fallback allocation
            // =========================================================================
            if (!validateROCmWorkspaceBinding(workspace_, rocm_device_id_, "ROCmQuantisedGemmKernel"))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Invalid workspace binding");
            }

            // Validate required buffers exist
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::QUANT_A))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: QUANT_A");
            }
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::SCALES_A))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: SCALES_A");
            }
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: SCALES_A_BLOCKWISE");
            }
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: SUMS_A_BLOCKWISE");
            }
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::ACC_INT32))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: ACC_INT32");
            }
            // Persistent FP32 temporaries also own grouped verifier combine/copy.
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::TEMP_A_FP32))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: TEMP_A_FP32");
            }
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::TEMP_C_FP32))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: TEMP_C_FP32");
            }
            // NOTE: CK-specific buffers (ROCM_CK_INT32, ROCM_A_PADDED,
            // ROCM_SCALE_A_PADDED, ROCM_E_PADDED, ROCM_B_REPACK) are no
            // longer requested — the CK path is retired. Skipped here.
            const std::string scatter_partial_name = scatterPartialBufferName();
            if (!workspace_->hasBuffer(scatter_partial_name))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: " +
                    scatter_partial_name);
            }
            const std::string batched_scatter_partial_name = scatterPartialBatchedBufferName();
            /*
             * The batched arena is conditional graph topology, not a scalar
             * GEMM requirement. appendFusedProjectionWorkspaceRequirements()
             * declares it for graphs that can launch two or more projections
             * concurrently; ordinary decode, LM-head, and standalone kernel
             * harnesses deliberately omit it. Fused launch sites validate the
             * pointer and exact capacity before publishing any work, so making
             * it optional here preserves the single source of workspace
             * accounting truth instead of forcing every scalar GEMM to reserve
             * the largest fused bundle.
             */
            if (!workspace_->hasBuffer(GemmWorkspaceBuffers::ROCM_SELFREDUCE_COUNTERS))
            {
                throw std::runtime_error(
                    "[ROCmQuantisedGemmKernel] Workspace missing required buffer: ROCM_SELFREDUCE_COUNTERS");
            }

            // Populate impl_ pointers from workspace
            impl_->d_A_int8 = static_cast<int8_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A));
            impl_->d_scales_A = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A));
            impl_->d_scales_A_blockwise = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE));
            impl_->d_sums_A_blockwise = static_cast<int32_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::SUMS_A_BLOCKWISE));
            impl_->d_C_int32 = static_cast<int32_t *>(workspace_->getBuffer(GemmWorkspaceBuffers::ACC_INT32));
            impl_->d_A_fp32 = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::TEMP_A_FP32));
            impl_->d_C_fp32 = static_cast<float *>(workspace_->getBuffer(GemmWorkspaceBuffers::TEMP_C_FP32));
            impl_->d_scatter_partial = static_cast<float *>(workspace_->getBuffer(scatter_partial_name));
            impl_->d_scatter_partial_batched =
                workspace_->hasBuffer(batched_scatter_partial_name)
                    ? static_cast<float *>(workspace_->getBuffer(
                          batched_scatter_partial_name))
                    : nullptr;
            impl_->d_selfreduce_counters =
                static_cast<int *>(workspace_->getBuffer(GemmWorkspaceBuffers::ROCM_SELFREDUCE_COUNTERS));

            // Set capacity values to max (workspace is pre-sized for maximum dimensions)
            // These are used by code paths that check capacity before use
            impl_->d_A_fp32_capacity = SIZE_MAX;
            impl_->d_C_fp32_capacity = SIZE_MAX;

            // Mark workspace as validated to skip re-validation on next call
            impl_->validated_workspace = workspace_;

            LOG_TRACE("[ROCmQuantisedGemmKernel::validateWorkspace] Workspace validated, pointers populated"
                      << " A_int8=" << (void *)impl_->d_A_int8
                      << " scales_A=" << (void *)impl_->d_scales_A
                      << " C_int32=" << (void *)impl_->d_C_int32);
        }

        bool ROCmQuantisedGemmKernel::multiply_q8_to_fp32(
            const Q8_1Block *d_A_q8, float *d_C,
            int m, int n, int k,
            float alpha, float beta)
        {
            // Decode work admission never changes the selected arithmetic program.
            if (m > 1 && currentProjectionRowWork(m) && !VerifierKernelModeScope::active())
                throw std::invalid_argument("Projection row intervals require explicit grouped Decode arithmetic for M>1");

            // TODO: Implement in Phase 5
            return false;
        }

        bool ROCmQuantisedGemmKernel::multiply_q8_to_q8(
            const Q8_1Block *d_A_q8, Q8_1Block *d_C_q8,
            int m, int n, int k)
        {
            // Decode work admission never changes the selected arithmetic program.
            if (m > 1 && currentProjectionRowWork(m) && !VerifierKernelModeScope::active())
                throw std::invalid_argument("Projection row intervals require explicit grouped Decode arithmetic for M>1");

            // TODO: Implement in Phase 5
            return false;
        }

        // =====================================================================
        // Fused SwiGLU + GEMM
        // =====================================================================

        extern "C"
        {
            bool hipOps_swiglu_fp32(
                const float *gate, const float *up, float *output,
                int size, int device_idx, void *stream);
        }

        bool ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int m, int n, int k,
            float alpha, float beta,
            DeviceWorkspaceManager *workspace)
        {
            // Decode work admission never changes the selected arithmetic program.
            if (m > 1 && currentProjectionRowWork(m) && !VerifierKernelModeScope::active())
                throw std::invalid_argument("Projection row intervals require explicit grouped Decode arithmetic for M>1");

            (void)requireGPUStream();

            DeviceWorkspaceManager *ws = workspace ? workspace : workspace_;
            DeviceWorkspaceManager *saved_workspace = workspace_;
            if (ws && ws != workspace_)
            {
                workspace_ = ws;
            }
            auto restore_workspace = [&](void *)
            {
                if (ws && ws != saved_workspace)
                {
                    workspace_ = saved_workspace;
                }
            };
            std::unique_ptr<void, decltype(restore_workspace)> workspace_restore_guard(
                (ws && ws != saved_workspace) ? static_cast<void *>(this) : nullptr,
                restore_workspace);

            /*
             * Gate and up may have been produced by separate projection
             * streams. Join both completion events to this down-projection
             * stream before launching fused SwiGLU; pointer residency alone is
             * not an ordering guarantee.
             */
            const DeviceId target_device =
                DeviceId::rocm(rocm_device_id_);
            TransferEngine::requireDeviceInput(
                const_cast<TensorBase *>(gate),
                target_device,
                gpu_stream_);
            TransferEngine::requireDeviceInput(
                const_cast<TensorBase *>(up),
                target_device,
                gpu_stream_);
            if (beta != 0.0f)
                TransferEngine::requireDeviceInput(
                    output, target_device, gpu_stream_);
            else
                TransferEngine::requireDeviceOutput(
                    output, target_device, gpu_stream_);

            // Get GPU data pointers after both producer events are joined.
            const float *d_gate = static_cast<const float *>(gate->gpu_data_ptr());
            const float *d_up = static_cast<const float *>(up->gpu_data_ptr());
            float *d_C = static_cast<float *>(output->gpu_data_ptr());

            if (!d_gate || !d_up || !d_C)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu] "
                          "Null GPU pointer: gate="
                          << (void *)d_gate
                          << " up=" << (void *)d_up << " C=" << (void *)d_C);
                return false;
            }

            ensureWeightsConverted();
            if (currentProjectionRowWork(m) && (!impl_ || !impl_->has_native_quantised))
                throw std::invalid_argument("Decode projection work requires the prepared all-codebook NativeQuantised route");
            validateWorkspace();

            if (!impl_)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu] "
                          "No impl_ (workspace buffers not initialized)");
                return false;
            }

            if (m > 1 || countedVerifierRows(m))
            {
                // M>1 PREFILL: Fused SwiGLU + blockwise quantize → INT8 GEMM
                if (!impl_->d_A_int8 || !impl_->d_scales_A_blockwise)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu] "
                              "Missing workspace buffers for fused SwiGLU+quant (d_A_int8 or d_scales_A_blockwise)");
                    return false;
                }

                // Step 1: Fused SwiGLU + blockwise quantize → d_A_int8 + d_scales_A_blockwise
                const bool needs_block_sums =
                    g_rocm_native_quantised_decode_equivalent_scope &&
                    impl_->has_native_quantised && (m >= 2 || countedVerifierRows(m)) && (k % 32) == 0;
                const bool swiglu_quant_ok = needs_block_sums
                    ? hipOps_fused_swiglu_quantize_blockwise_with_sumsScoped(
                          d_gate, d_up,
                          impl_->d_A_int8, impl_->d_scales_A_blockwise,
                          impl_->d_sums_A_blockwise,
                          m, k, rocm_device_id_, gpu_stream_)
                    : hipOps_fused_swiglu_quantize_blockwiseScoped(
                          d_gate, d_up,
                          impl_->d_A_int8, impl_->d_scales_A_blockwise,
                          m, k, rocm_device_id_, gpu_stream_);
                if (!swiglu_quant_ok)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu] "
                              "Fused SwiGLU+quantize kernel failed");
                    return false;
                }

                // Step 2: NativeQuantised grouped verifier path. Once the explicit
                // verifier API selects this route, runtime M must stay on the
                // grouped decode-equivalent kernel instead of generic prefill.
                const bool supports_native_small_m =
                    g_rocm_native_quantised_decode_equivalent_scope &&
                    impl_->has_native_quantised && (m >= 2 || countedVerifierRows(m)) &&
                    alpha == 1.0f && beta == 0.0f;
                if (supports_native_small_m)
                {
                    if ((k % 32) != 0)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu] "
                                  "NativeQuantised fused SwiGLU small-M verifier requires K multiple of 32, got M="
                                  << m << " K=" << k);
                        return false;
                    }

                    if (!impl_->d_scatter_partial ||
                        !impl_->d_weights_native_quantised ||
                        !impl_->d_weights_native_scales)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu] "
                                  "NativeQuantised fused SwiGLU small-M verifier missing graph-native buffers"
                                  << " weights=" << static_cast<const void *>(impl_->d_weights_native_quantised)
                                  << " scales=" << static_cast<const void *>(impl_->d_weights_native_scales)
                                  << " partial=" << static_cast<const void *>(impl_->d_scatter_partial));
                        return false;
                    }

                    if (!rocmGemv_native_quantised_small_m_fp32_with_sums_policy(
                            impl_->d_A_int8,
                            impl_->d_weights_native_quantised,
                            impl_->d_weights_native_scales,
                            impl_->d_weights_native_mins,
                            impl_->d_weights_native_emins,
                            d_C,
                            impl_->d_scales_A_blockwise,
                            impl_->d_sums_A_blockwise,
                            impl_->d_scatter_partial,
                            m, n, k,
                            impl_->native_quantised_codebook_id,
                            requireNativeQuantisedArithmeticPolicyCodebook(
                                impl_->native_quantised_codebook_id,
                                impl_->native_source_identity),
                            rocm_device_id_, gpu_stream_,
                            VerifierKernelModeScope::rowsFor(m)))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu] "
                                  "NativeQuantised fused SwiGLU small-M verifier GEMV failed");
                        return false;
                    }

            if (PerfStatsCollector::isDomainEnabled("kernel"))
                    {
                        PerfStatsCollector::addCounter(
                            "kernel",
                            "rocm_native_quantised_small_m_calls",
                            1.0,
                            "gemm",
                            "rocm:" + std::to_string(rocm_device_id_),
                            PerfStatsCollector::Tags{
                                {"m", std::to_string(m)},
                                {"codebook", std::to_string(static_cast<int>(impl_->native_quantised_codebook_id))},
                                {"n", std::to_string(n)},
                                {"k", std::to_string(k)},
                                {"source", "fused_swiglu"}});

                        PerfStatsCollector::addCounter(
                            "kernel",
                            "rocm_native_quantised_small_m_rows",
                            static_cast<double>(m),
                            "gemm",
                            "rocm:" + std::to_string(rocm_device_id_),
                            PerfStatsCollector::Tags{
                                {"m", std::to_string(m)},
                                {"codebook", std::to_string(static_cast<int>(impl_->native_quantised_codebook_id))},
                                {"n", std::to_string(n)},
                                {"k", std::to_string(k)},
                                {"source", "fused_swiglu"}});
                    }

                    publishROCmTensorWrite(*output, rocm_device_id_, gpu_stream_);
                    return true;
                }

                // Step 2 fallback for non-small-M prefill shapes.
                if (impl_->has_native_quantised && alpha == 1.0f && beta == 0.0f)
                {
                    const uint8_t cb_id = impl_->native_quantised_codebook_id;
                    if (rocmGemm_native_quantised_fp32_with_policy(
                            impl_->d_A_int8,
                            impl_->d_weights_native_quantised,
                            impl_->d_weights_native_scales,
                            impl_->d_weights_native_mins,
                            impl_->d_weights_native_emins,
                            d_C,
                            impl_->d_scales_A,
                            impl_->d_scales_A_blockwise,
                            m, n, k,
                            cb_id,
                            requireNativeQuantisedArithmeticPolicyCodebook(
                                cb_id,
                                impl_->native_source_identity),
                            rocm_device_id_, gpu_stream_))
                    {
                        publishROCmTensorWrite(*output, rocm_device_id_, gpu_stream_);
                        return true;
                    }
                }

                // Resolve d_scales_B (same logic as multiply_tensor)
                float *d_scales_B = nullptr;
                if (packed_)
                    d_scales_B = packed_->d_scales;
                else if (impl_)
                    d_scales_B = impl_->d_scales_B;

                // Fallback: tryPrefillNativeGemm (INT8 VNNI path)
                if (tryPrefillNativeGemm(
                        impl_->d_A_int8, d_C,
                        impl_->d_scales_A, impl_->d_scales_A_blockwise,
                        d_scales_B, nullptr,
                        m, n, k, alpha, beta,
                        "ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu"))
                {
                    publishROCmTensorWrite(*output, rocm_device_id_, gpu_stream_);
                    return true;
                }

                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu] "
                          "No GEMM path available for M>1 (m="
                          << m << " n=" << n << " k=" << k << ")");
                return false;
            }
            else
            {
                // M=1 DECODE: SwiGLU → FP32 temp → standard FP32→FP32 GEMV path
                if (!impl_->d_A_fp32)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu] "
                              "No temp FP32 workspace buffer for M=1 decode path");
                    return false;
                }

                // Step 1: Compute SwiGLU on GPU: temp = silu(gate) * up
                if (!hipOps_swiglu_fp32(d_gate, d_up, impl_->d_A_fp32, m * k,
                                        rocm_device_id_, gpu_stream_))
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu] "
                              "SwiGLU kernel failed for decode");
                    return false;
                }

                // Step 2: Quantize + GEMV via existing FP32→FP32 path
                const bool decode_ok =
                    multiply_fp32_to_fp32(impl_->d_A_fp32, d_C, m, n, k, alpha, beta);
                if (decode_ok)
                {
                    publishROCmTensorWrite(*output, rocm_device_id_, gpu_stream_);
                }
                return decode_ok;
            }
        }

        bool ROCmQuantisedGemmKernel::multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int m, int n, int k,
            float alpha,
            float beta,
            DeviceWorkspaceManager *workspace)
        {
            if (m < 1)
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel] grouped verifier SwiGLU requires M>=1, got M="
                          << m);
                return false;
            }
            auto decode_equivalent_dispatch = beginVerifierDecodeEquivalentScope();
            std::optional<NativeQuantisedVerifierOperation> operation;
            const int policy_n = rocmGemv_native_quantised_get_serial_partition_n();
            if (impl_ && impl_->has_native_quantised && output && !output->isMapped() &&
                n == static_cast<int>(N_) && k == static_cast<int>(K_) &&
                alpha == 1.0f && beta == 0.0f && (policy_n <= 0 || policy_n == n))
                operation = NativeQuantisedVerifierOperation{NativeQuantisedVerifierOperationKind::SwiGLUDown,
                    rocm_device_id_, m, n, k, impl_->d_weights_native_quantised, impl_->native_source_identity};
            const ScopedNativeQuantisedVerifierOperation original_operation(operation);
            return multiply_tensor_with_fused_swiglu(
                gate, up, output, m, n, k, alpha, beta, workspace);
        }

        bool ROCmQuantisedGemmKernel::multiply_fp32_to_fp32(
            const float *d_A, float *d_C,
            int m, int n, int k,
            float alpha, float beta)
        {
            // Decode work admission never changes the selected arithmetic program.
            if (m > 1 && currentProjectionRowWork(m) && !VerifierKernelModeScope::active())
                throw std::invalid_argument("Projection row intervals require explicit grouped Decode arithmetic for M>1");

            return multiply_fp32_to_fp32_with_bias(d_A, d_C, nullptr, m, n, k, alpha, beta);
        }

        bool ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias(
            const float *d_A, float *d_C, const float *d_bias,
            int m, int n, int k,
            float alpha, float beta)
        {
            // Decode work admission never changes the selected arithmetic program.
            if (m > 1 && currentProjectionRowWork(m) && !VerifierKernelModeScope::active())
                throw std::invalid_argument("Projection row intervals require explicit grouped Decode arithmetic for M>1");

            (void)requireGPUStream();

            LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] m=" << m << " n=" << n << " k=" << k
                                                                                      << " alpha=" << alpha << " beta=" << beta
                                                                                      << " d_A=" << static_cast<const void *>(d_A)
                                                                                      << " d_C=" << static_cast<void *>(d_C)
                                                                                      << " d_bias=" << static_cast<const void *>(d_bias));

            // DECODE FAST PATH: M=1 GEMV with fused bias
            if (m == 1 && !countedVerifierRows(m))
            {
                ensureWeightsConverted();
            if (currentProjectionRowWork(m) && (!impl_ || !impl_->has_native_quantised))
                throw std::invalid_argument("Decode projection work requires the prepared all-codebook NativeQuantised route");
                // Option B: weights are VNNI and/or ratio-VNNI on device
                int8_t *d_vnni = impl_ ? impl_->d_weights_int8_vnni : nullptr;
                float *d_s = packed_ ? packed_->d_scales : (impl_ ? impl_->d_scales_B : nullptr);
                if ((impl_ && impl_->has_native_quantised) || (d_vnni && d_s))
                {
                    LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] GEMV fast path M=1 +bias");

                    if (!impl_)
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] INT8 GEMV requires impl_ buffers");
                        return false;
                    }

                    validateWorkspace();

                    // NativeQuantised path: lossless Q4/IQ4 decode, typed block metadata, then optional alpha/beta/bias epilogue.
                    if (impl_->has_native_quantised)
                    {
                        if (!rocmQuantGemm_quantizeActivationsBlockwiseScoped(
                                d_A, impl_->d_A_int8, impl_->d_scales_A_blockwise, m, k, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] NativeQuantised blockwise activation quantization failed");
                            return false;
                        }

                        float *d_native_output = (beta != 0.0f) ? impl_->d_C_fp32 : d_C;
                        if (!rocmGemv_native_quantised_fp32_with_policy(
                                impl_->d_A_int8,
                                impl_->d_weights_native_quantised,
                                impl_->d_weights_native_scales,
                                impl_->d_weights_native_mins,
                                impl_->d_weights_native_emins,
                                d_native_output,
                                impl_->d_scales_A,
                                impl_->d_scatter_partial,
                                n, k,
                                impl_->native_quantised_codebook_id,
                                requireNativeQuantisedArithmeticPolicyCodebook(
                                    impl_->native_quantised_codebook_id,
                                    impl_->native_source_identity),
                                rocm_device_id_, gpu_stream_,
                                impl_->d_scales_A_blockwise))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] NativeQuantised GEMV failed");
                            return false;
                        }

                        if (beta != 0.0f)
                        {
                            if (!rocmQuantGemm_applyFp32CombineEpilogue(
                                    d_native_output,
                                    d_C,
                                    d_C,
                                    d_bias,
                                    m,
                                    n,
                                    alpha,
                                    beta,
                                    rocm_device_id_,
                                    gpu_stream_))
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] NativeQuantised beta epilogue failed");
                                return false;
                            }
                        }
                        else if (alpha != 1.0f || d_bias)
                        {
                            if (!rocmQuantGemm_applyFp32Epilogue(
                                    d_C,
                                    d_bias,
                                    m,
                                    n,
                                    alpha,
                                    rocm_device_id_,
                                    gpu_stream_))
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] NativeQuantised epilogue failed");
                                return false;
                            }
                        }
                        return true;
                    }

                    // Use INT8 scatter+reduce GEMV when alpha=1, beta=0 AND VNNI weights available
                    // Bias handled inside scatter reduce kernel
                    if (alpha == 1.0f && beta == 0.0f && d_vnni)
                    {
                        if (!rocmQuantGemm_quantizeActivationsBlockwiseScoped(
                                d_A, impl_->d_A_int8, impl_->d_scales_A_blockwise, m, k, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] INT8 blockwise activation quantization failed");
                            return false;
                        }
                        bool ok = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                            impl_->d_A_int8, d_vnni, d_C,
                            impl_->d_scales_A_blockwise, d_s,
                            n, k,
                            alpha, beta,
                            nullptr, d_bias,
                            rocm_device_id_, gpu_stream_);

                        if (!ok)
                        {
                            ok = rocmGemv_int8_scatter_vnni_blockwise(
                                impl_->d_A_int8, d_vnni, d_C, impl_->d_scales_A_blockwise, d_s, d_bias,
                                impl_->d_scatter_partial, impl_->d_selfreduce_counters,
                                n, k, alpha, beta, nullptr,
                                rocm_device_id_, gpu_stream_);
                        }

                        return ok;
                    }

                    // Fallback: blockwise quantize → blockwise GEMV for non-standard alpha/beta
                    if (!rocmQuantGemm_quantizeActivationsBlockwiseScoped(
                            d_A, impl_->d_A_int8, impl_->d_scales_A_blockwise, m, k, rocm_device_id_, gpu_stream_))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] Activation quantization failed");
                        return false;
                    }

                    if (d_vnni)
                    {
                        float *d_int8_output = (beta != 0.0f) ? impl_->d_C_fp32 : d_C;
                        const float *d_existing = (beta != 0.0f) ? d_C : nullptr;
                        bool ok = rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
                            impl_->d_A_int8, d_vnni, d_int8_output,
                            impl_->d_scales_A_blockwise, d_s,
                            n, k,
                            alpha, beta,
                            d_existing, d_bias,
                            rocm_device_id_, gpu_stream_);

                        if (!ok)
                        {
                            ok = rocmGemv_int8_scatter_vnni_blockwise(
                                impl_->d_A_int8, d_vnni, d_int8_output, impl_->d_scales_A_blockwise, d_s, d_bias,
                                impl_->d_scatter_partial, impl_->d_selfreduce_counters,
                                n, k, alpha, beta, d_existing,
                                rocm_device_id_, gpu_stream_);
                        }

                        if (!ok)
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] INT8 GEMV (VNNI blockwise) failed");
                            return false;
                        }
                        if (beta != 0.0f)
                        {
                            hipError_t copy_err = hipMemcpyAsync(
                                d_C,
                                d_int8_output,
                                static_cast<size_t>(n) * sizeof(float),
                                hipMemcpyDeviceToDevice,
                                static_cast<hipStream_t>(gpu_stream_));
                            if (copy_err != hipSuccess)
                            {
                                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] INT8 beta output copy failed: "
                                          << hipGetErrorString(copy_err));
                                return false;
                            }
                        }
                        return true;
                    }

                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] No VNNI weights for M=1 decode GEMV");
                    return false;
                }
            }

            // Ensure weights converted and validate workspace
            ensureWeightsConverted();
            if (currentProjectionRowWork(m) && (!impl_ || !impl_->has_native_quantised))
                throw std::invalid_argument("Decode projection work requires the prepared all-codebook NativeQuantised route");
            validateWorkspace();

            // Step 1: Quantize FP32 activations to INT8 (blockwise)
            if (!rocmQuantGemm_quantizeActivationsBlockwiseScoped(
                    d_A, impl_->d_A_int8, impl_->d_scales_A_blockwise, m, k, rocm_device_id_, gpu_stream_))
            {
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] Blockwise activation quantization failed");
                return false;
            }

            // Step 1b: Try native-quantised GEMM (halved HBM bandwidth, no epilogue)
            // Same pattern as multiply_tensor and multiply_fp32_to_fp32: native-quantised
            // produces final FP32 output with per-block scales applied inline.
            // Bias is applied via a lightweight biasAdd kernel after the GEMM.
            if (m > 1 && impl_->has_native_quantised && alpha == 1.0f && beta == 0.0f)
            {
                const uint8_t cb_id = impl_->native_quantised_codebook_id;
                if (rocmGemm_native_quantised_fp32_with_policy(
                        impl_->d_A_int8,
                        impl_->d_weights_native_quantised,
                        impl_->d_weights_native_scales,
                        impl_->d_weights_native_mins,
                        impl_->d_weights_native_emins,
                        d_C,
                        impl_->d_scales_A,
                        impl_->d_scales_A_blockwise,
                        m, n, k,
                        cb_id,
                        requireNativeQuantisedArithmeticPolicyCodebook(
                            cb_id,
                            impl_->native_source_identity),
                        rocm_device_id_, gpu_stream_))
                {
                    LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] "
                              "NativeQuantised GEMM succeeded (M="
                              << m << " N=" << n << " K=" << k
                              << " codebook=" << static_cast<int>(cb_id) << " +bias)");

                    if (d_bias)
                    {
                        if (!rocmQuantGemm_biasAdd(d_C, d_bias, m, n, rocm_device_id_, gpu_stream_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] NativeQuantised GEMM bias add failed");
                            return false;
                        }
                    }
                    return true;
                }
                LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] "
                          "NativeQuantised GEMM failed; refusing INT8 GEMM fallback");
                return false;
            }

            if (m > 1 && tryPrefillNativeGemm(
                             impl_->d_A_int8,
                             d_C,
                             impl_->d_scales_A,
                             impl_->d_scales_A_blockwise,
                             impl_->d_scales_B,
                             d_bias,
                             m, n, k,
                             alpha, beta,
                             "ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias"))
            {
                return true;
            }

            // CK fallback retired — native-quantised / INT8-VNNI are the only supported paths.
            LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_fp32_to_fp32_with_bias] "
                      "NativeQuantised and INT8-VNNI prefill both failed (m="
                      << m
                      << " n=" << n << " k=" << k << "); CK fallback has been removed.");
            return false;
        }

        bool ROCmQuantisedGemmKernel::multiply_fp32_to_q8(
            const float *d_A, Q8_1Block *d_C_q8,
            int m, int n, int k)
        {
            // Decode work admission never changes the selected arithmetic program.
            if (m > 1 && currentProjectionRowWork(m) && !VerifierKernelModeScope::active())
                throw std::invalid_argument("Projection row intervals require explicit grouped Decode arithmetic for M>1");

            // TODO: Implement in Phase 5
            return false;
        }

    } // namespace rocm
} // namespace llaminar2

#include "ROCmPersistentHCRead.h"
#include "ROCmFP32ProjectionLoadPolicy.h"
namespace llaminar2::rocm {
/** @copydoc resolvePersistentHCReadSources */
void resolvePersistentHCReadSources(PersistentHCReadArgs& args,const ITensorGemm& down,
 const ITensorGemm& up,const ITensorGemm* injection) {
 if(isGraphCaptureActive()||!down.weights_converted()||!up.weights_converted()||
    !down.inspectPreparedNativeQuantisedMatrixDesc(args.down_weights)||
    !up.inspectPreparedNativeQuantisedMatrixDesc(args.up.weights))
  throw std::invalid_argument("Persistent HC requires original already-prepared native source owners cold");
 if(injection) {
  ContiguousFloatingPointWeightDescriptor weights;
  if(!injection->weights_converted()||!injection->exportContiguousFloatingPointWeights(weights)||
     !weights.valid()||weights.type!=TensorType::FP32||weights.n!=4||weights.k!=args.down_weights.k)
   throw std::invalid_argument("Persistent HC injection requires its original prepared contiguous FP32 four-row source");
  args.injection.weights=static_cast<const float*>(weights.data);
  args.injection.load_pairs=static_cast<int>(selectFP32ProjectionLoadPairs(1,4,1,weights.k));
 } else if(args.injection.weights||args.injection.output)
  throw std::invalid_argument("Persistent HC injection binding has no original prepared engine owner");
}
} // namespace llaminar2::rocm
