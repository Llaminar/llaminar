/**
 * @file CPUW4A4GemmKernel.cpp
 * @brief Native CPU W4A4/H stage adapters and shared-operand projection workshares.
 *
 * Validation precedes data production. Every selected projection consumes a
 * complete-K native operand policy. Equal policies share A4 preparation; a
 * different source clip/rotation requires its own exact preparation, not reuse
 * of a vaguely compatible bank. One OpenMP team owns the fused transaction and
 * its barriers retire each reader before the same arena region is overwritten.
 */
#include "CPUW4A4GemmKernel.h"

#include "kernels/cpu/primitives/SwiGLUPrimitives.h"
#include "tensors/TensorClasses.h"
#include "utils/OpenMPUtils.h"
#include "utils/PerfStatsCollector.h"

#include <cmath>
#include <memory>
#include <type_traits>

namespace llaminar2::w4a4
{
    namespace
    {
        /** @return Checked FP32 extent; no allocation or private reservation is made. */
        std::size_t elements(int rows, int width)
        {
            if (rows < 0 || width <= 0) throw std::invalid_argument("W4A4 GEMM requires valid matrix geometry");
            const auto count = CPUInvocationWorkspace::multiply(static_cast<std::size_t>(rows), width);
            static_cast<void>(CPUInvocationWorkspace::multiply(count, sizeof(float)));
            return count;
        }

        /** @brief Require existing FP32 host authority, including typed FP32 tensor views.
         * @param tensor Operand whose transfer/publication is already complete.
         * @param count Required prefix, in FP32 elements.
         * @throws std::invalid_argument For absent, wrong-format, short or non-host storage.
         */
        void requireFP32(const TensorBase *tensor, std::size_t count)
        {
            if (!tensor || tensor->native_type() != TensorType::FP32 || !tensor->is_fp32_backed()
                || !tensor->isOnCPU() || tensor->numel() < count)
                throw std::invalid_argument("W4A4 GEMM requires complete already-published CPU FP32 tensors");
        }

        /** @return Typed immutable FP32 prefix; this accessor cannot perform a conversion or transfer. */
        std::span<const float> readFP32(const TensorBase *tensor, std::size_t count)
        {
            requireFP32(tensor, count);
            const auto *pointer = tensor->fp32_data();
            if (!pointer && count) throw std::invalid_argument("W4A4 GEMM has no admitted FP32 input storage");
            return {pointer, count};
        }

        /** @return Typed writable FP32 prefix, after existing host authority has been checked. */
        std::span<float> writeFP32(TensorBase *tensor, std::size_t count)
        {
            requireFP32(tensor, count);
            auto *pointer = tensor->mutable_fp32_data();
            if (!pointer && count) throw std::invalid_argument("W4A4 GEMM has no admitted FP32 output storage");
            return {pointer, count};
        }

        /** @brief Reject aliasing without constructing overflowing end pointers.
         * @param left First borrowed byte extent.
         * @param right Second borrowed byte extent.
         * @throws std::invalid_argument If a producer would overwrite a live operand.
         */
        void requireSeparate(std::span<const std::byte> left, std::span<const std::byte> right)
        {
            if (left.empty() || right.empty()) return;
            const auto a = reinterpret_cast<std::uintptr_t>(left.data());
            const auto b = reinterpret_cast<std::uintptr_t>(right.data());
            if (a <= b ? b - a < left.size() : a - b < right.size())
                throw std::invalid_argument("W4A4 GEMM requires disjoint live operands and invocation scratch");
        }

        /** @brief Prove the native preparation cannot overwrite an already-live external operand. */
        void requireScratchSeparate(CPUActivationBuffers scratch, std::span<const std::byte> operand)
        {
            requireSeparate(std::as_bytes(scratch.blocks), operand);
            requireSeparate(std::as_bytes(scratch.ranges), operand);
            requireSeparate(std::as_bytes(scratch.row_ranges), operand);
            requireSeparate(std::as_bytes(scratch.butterfly), operand);
            requireSeparate(std::as_bytes(scratch.transformed), operand);
            requireSeparate(std::as_bytes(scratch.active_rows), operand);
        }

        /** @brief Resolve borrowed CPU work before entering the workshare.
         * @param rows Original physical matrix rows.
         * @return Immutable original-coordinate publication metadata, if present.
         * @throws std::invalid_argument or std::logic_error For inconsistent geometry/counts.
         */
        std::optional<ProjectionRowWork> invocationWork(int rows)
        {
            std::optional<ProjectionRowWork> work;
            if (const auto *scoped = currentProjectionRowWork(rows)) work = *scoped;
            if (const auto *requests = ITensorGemm::VerifierKernelModeScope::requestsFor(rows))
                work = work ? work->withInheritedRequests(*requests) : ProjectionRowWork::fromRequests(*requests);
            else if (const auto *prefix = ITensorGemm::VerifierKernelModeScope::prefixForFloating(rows))
                work = work ? work->withInheritedPrefix(*prefix) : ProjectionRowWork::fromPrefix(*prefix);
            if (work)
                for (int request = 0; request < work->requests(); ++request)
                    static_cast<void>(work->resolveHost(request));
            return work;
        }

        /** @brief Preserve native projection rounding, then apply the graph's FP32 epilogue.
         * @param prepared Completed operands and their sealed original-row validity.
         * @param product Unscaled native results, possibly the output itself when beta is zero.
         * @param output Original destination, preserving the beta addend until this workshare.
         * @param bias Optional complete output bias.
         * @param columns Complete output width.
         * @param alpha Product multiplier, applied after native arithmetic.
         * @param beta Existing output multiplier; zero never reads dirty destination bytes.
         */
        void epilogue(const CPUActivations &prepared, std::span<const float> product,
            std::span<float> output, std::span<const float> bias, int columns, float alpha, float beta)
        {
            if (alpha == 1.0f && beta == 0.0f && bias.empty()) return;
            auto work = [&] {
#pragma omp for collapse(2) schedule(static)
                for (int row = 0; row < prepared.rows(); ++row)
                    for (int column = 0; column < columns; ++column)
                    {
                        if (!prepared.active(row)) continue;
                        const auto index = static_cast<std::size_t>(row) * columns + column;
                        float value = product[index];
                        if (alpha != 1.0f) value = alpha * value;
                        if (beta != 0.0f) value = value + beta * output[index];
                        if (!bias.empty()) value = value + bias[column];
                        output[index] = value;
                    }
            };
            OMP_WORKSHARE_REGION(work);
        }

        /** @brief Reject unsupported epilogue state before any destination is written. */
        void requireEpilogue(float alpha, float beta)
        {
            if (!std::isfinite(alpha) || !std::isfinite(beta))
                throw std::invalid_argument("W4A4 GEMM requires finite alpha and beta");
        }

        /** @brief Record actual native invocations without enabling instrumentation-dependent arithmetic. */
        void recordCall(const char *name)
        {
            if (PerfStatsCollector::isDomainEnabled("kernel"))
                PerfStatsCollector::addCounter("kernel", name, 1.0, "gemm", "cpu");
        }
    }

    CPUW4A4GemmKernel::CPUW4A4GemmKernel(WeightStorage weights) : weights_(std::move(weights)) {}
    bool CPUW4A4GemmKernel::supports_device(int device_idx) const { return device_idx == -1; }
    int CPUW4A4GemmKernel::get_n() const { return weights_.view().contract().rows(); }
    int CPUW4A4GemmKernel::get_k() const { return weights_.view().contract().columns(); }
    std::size_t CPUW4A4GemmKernel::packedWeightBytes() const { return weights_.view().blocks().size_bytes(); }
    bool CPUW4A4GemmKernel::canReleaseSourceWeightTensor() const { return true; }
    bool CPUW4A4GemmKernel::supports_fused_projection() const { return true; }

    KernelSnapshotInfo CPUW4A4GemmKernel::getKernelSnapshotInfo() const
    {
        return KernelSnapshotInfo::gemm()
            .withInput("A", "native W4A4/H FP32 activation boundary", KernelBufferDtype::FP32)
            .withOutput("C", "native integer-group projection result", KernelBufferDtype::FP32);
    }

    void CPUW4A4GemmKernel::requireGeometry(int m, int n, int k) const
    {
        if (m <= 0 || n != get_n() || k != get_k())
            throw std::invalid_argument("W4A4 GEMM requires positive original rows and its exact complete-K matrix");
    }

    WorkspaceRequirements CPUW4A4GemmKernel::getWorkspaceRequirements(int m, int n, int k) const
    {
        if ((n && n != get_n()) || (k && k != get_k()))
            throw std::invalid_argument("W4A4 workspace declaration belongs to a different native matrix");
        return cpuWorkspaceRequirements(weights_.view().contract(), m);
    }

    void CPUW4A4GemmKernel::appendOutputAccumulationWorkspaceRequirements(
        WorkspaceRequirements &requirements, int m, int n) const
    {
        requireGeometry(m, n ? n : get_n(), get_k());
        requirements.merge(cpuProjectionFP32WorkspaceRequirements(kCPUProjectionProduct, m, get_n()));
    }

    void CPUW4A4GemmKernel::appendFusedProjectionWorkspaceRequirements(
        WorkspaceRequirements &requirements, int m, std::span<const int> columns, int k) const
    {
        requireGeometry(m, get_n(), k);
        if (columns.empty()) throw std::invalid_argument("W4A4 fusion requires a nonempty projection inventory");
        for (int n : columns)
            if (n <= 0) throw std::invalid_argument("W4A4 fusion requires positive physical output widths");
        requirements.merge(cpuFusedProjectionWorkspaceRequirements(columns.size()));
    }

    std::unique_ptr<ITensorGemm::OutputPartitionEquivalenceScope>
    CPUW4A4GemmKernel::beginOutputPartitionEquivalenceScope(int actual, int serial)
    {
        if (actual != get_n() || serial <= 0 || serial > actual)
            throw std::invalid_argument("W4A4 output partition disagrees with its retained row lease");
        // Every column owns the same fixed integer group order independently
        // of N. No alternate arithmetic dispatch or mutable TLS is necessary.
        return nullptr;
    }

    std::unique_ptr<ITensorGemm::OutputPartitionEquivalenceScope>
    CPUW4A4GemmKernel::beginOriginalOutputSliceScope(const OriginalOutputSlice &slice)
    {
        slice.require(this, get_n(), get_k());
        if (!slice.device().is_cpu())
            throw std::invalid_argument("W4A4 original output slice belongs to a non-CPU engine");
        return nullptr;
    }

    bool CPUW4A4GemmKernel::multiply_tensor(const TensorBase *input, TensorBase *output,
        int m, int n, int k, bool transpose_B, float alpha, float beta, const TensorBase *bias,
        const IMPIContext *, int device_idx, DeviceWorkspaceManager *workspace, int offset)
    {
        requireGeometry(m, n, k);
        requireEpilogue(alpha, beta);
        // As with the other native prepared operands, the retained matrix
        // owns [N,K]. Ordinary graph stages pass false; direct callers may
        // pass true. This legacy floating-BLAS flag cannot transpose sealed
        // packed storage or change the native arithmetic.
        (void)transpose_B;
        if (device_idx != -1 || offset < 0)
            throw std::invalid_argument("W4A4 GEMM requires CPU execution and a valid input offset");
        const auto work = invocationWork(m);
        const auto prefix = elements(offset, k), input_count = elements(m, k);
        if (prefix > std::numeric_limits<std::size_t>::max() / sizeof(float) - input_count)
            throw std::overflow_error("W4A4 input row offset overflows its physical extent");
        const auto values = readFP32(input, prefix + input_count).subspan(prefix);
        auto destination = writeFP32(output, elements(m, n));
        const auto bias_values = bias ? readFP32(bias, n) : std::span<const float>{};
        const auto scratch = borrowCPUWorkspace(weights_.view().contract(), m, workspace);
        auto product = beta == 0.0f ? destination
            : CPUInvocationWorkspace::require<float>(workspace, kCPUProjectionProduct, destination.size());
        for (const auto operand : {std::as_bytes(values), std::as_bytes(destination), std::as_bytes(bias_values),
                                   std::as_bytes(weights_.view().blocks())})
            requireScratchSeparate(scratch, operand);
        requireSeparate(std::as_bytes(destination), std::as_bytes(values));
        requireSeparate(std::as_bytes(destination), std::as_bytes(bias_values));
        if (beta != 0.0f)
        {
            requireSeparate(std::as_bytes(product), std::as_bytes(destination));
            requireSeparate(std::as_bytes(product), std::as_bytes(bias_values));
            requireSeparate(std::as_bytes(product), std::as_bytes(values));
        }
        const auto prepared = prepareCPUActivations(weights_.view().contract(), values, m, k, scratch, work);
        projectCPU(weights_.view(), prepared, product, n);
        epilogue(prepared, product, destination, bias_values, n, alpha, beta);
        recordCall("cpu_w4a4_projection_calls");
        return true;
    }

    bool CPUW4A4GemmKernel::multiply_tensor_with_fused_swiglu(const TensorBase *gate, const TensorBase *up,
        TensorBase *output, int m, int n, int k, float alpha, float beta, DeviceWorkspaceManager *workspace)
    {
        requireGeometry(m, n, k);
        requireEpilogue(alpha, beta);
        const auto work = invocationWork(m);
        const auto count = elements(m, k);
        const auto gate_values = readFP32(gate, count), up_values = readFP32(up, count);
        auto destination = writeFP32(output, elements(m, n));
        const auto scratch = borrowCPUWorkspace(weights_.view().contract(), m, workspace);
        auto activated = CPUInvocationWorkspace::require<float>(workspace, kCPUSwiGLUInput, count);
        auto product = beta == 0.0f ? destination
            : CPUInvocationWorkspace::require<float>(workspace, kCPUProjectionProduct, destination.size());
        for (const auto operand : {std::as_bytes(gate_values), std::as_bytes(up_values),
                                   std::as_bytes(destination), std::as_bytes(weights_.view().blocks())})
        {
            requireScratchSeparate(scratch, operand);
            requireSeparate(std::as_bytes(activated), operand);
        }
        requireScratchSeparate(scratch, std::as_bytes(activated));
        requireSeparate(std::as_bytes(destination), std::as_bytes(gate_values));
        requireSeparate(std::as_bytes(destination), std::as_bytes(up_values));
        if (beta != 0.0f)
        {
            requireSeparate(std::as_bytes(product), std::as_bytes(destination));
            requireSeparate(std::as_bytes(product), std::as_bytes(gate_values));
            requireSeparate(std::as_bytes(product), std::as_bytes(up_values));
        }
        auto execute = [&] {
            // Native K is divisible by 256. Each chunk is also a complete SIMD
            // extent, retaining ordinary CPU SwiGLU bytes without row GEMM replay.
#pragma omp for collapse(2) schedule(static)
            for (int row = 0; row < m; ++row)
                for (int block = 0; block < k / kBlockElements; ++block)
                {
                    if (work && !work->resolveHost(work->requestForRow(row)).contains(row)) continue;
                    const auto first = static_cast<std::size_t>(row) * k + block * kBlockElements;
                    primitives::compute_swiglu_serial(gate_values.data() + first, up_values.data() + first,
                                                      activated.data() + first, kBlockElements);
                }
            const auto prepared = prepareCPUActivations(weights_.view().contract(), activated, m, k, scratch, work);
            projectCPU(weights_.view(), prepared, product, n);
            epilogue(prepared, product, destination, {}, n, alpha, beta);
        };
        OMP_WORKSHARE_REGION(execute);
        recordCall("cpu_w4a4_swiglu_projection_calls");
        return true;
    }

    bool CPUW4A4GemmKernel::multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
        const TensorBase *gate, const TensorBase *up, TensorBase *output, int m, int n, int k,
        float alpha, float beta, DeviceWorkspaceManager *workspace)
    {
        // This delegates the complete grouped operation, never a loop over rows.
        return multiply_tensor_with_fused_swiglu(gate, up, output, m, n, k, alpha, beta, workspace);
    }

    bool CPUW4A4GemmKernel::multiply_fused_tensor(const TensorBase *input,
        const std::vector<TensorProjectionDesc> &projections, int m, int k,
        const IMPIContext *, DeviceWorkspaceManager *workspace)
    {
        requireGeometry(m, get_n(), k);
        if (projections.empty()) throw std::invalid_argument("W4A4 fusion requires native projections");
        const auto work = invocationWork(m);
        const auto values = readFP32(input, elements(m, k));
        // A single projection needs only fixed borrowed references, not an
        // arena metadata array. Both geometries use the identical validation,
        // grouped transform/reduction and epilogue below. Tensor scratch always
        // remains in the admitted workspace, including singleton verifier work.
        ProjectionInvocation single{};
        auto invocations = projections.size() == 1
            ? std::span<ProjectionInvocation>(&single, 1)
            : CPUInvocationWorkspace::require<ProjectionInvocation>(
                workspace, kCPUProjectionBundle, projections.size());
        const auto metadata = std::as_bytes(invocations);
        requireSeparate(metadata, std::as_bytes(values));
        // First validate the entire bundle, including later leaves, before
        // writing metadata that could otherwise alias an unvalidated operand.
        for (const auto &projection : projections)
        {
            const auto *kernel = dynamic_cast<const CPUW4A4GemmKernel *>(projection.kernel);
            if (!kernel) throw std::invalid_argument("W4A4 fusion requires native W4A4/H engines on every leaf");
            kernel->requireGeometry(m, projection.n, k);
            projection.requireOriginalOutputSlice(k);
            const auto output = writeFP32(projection.output, elements(m, projection.n));
            const auto bias = projection.bias ? readFP32(projection.bias, projection.n) : std::span<const float>{};
            const auto scratch = borrowCPUWorkspace(kernel->weights_.view().contract(), m, workspace);
            for (const auto operand : {std::as_bytes(output), std::as_bytes(bias),
                                       std::as_bytes(kernel->weights_.view().blocks())})
            {
                requireSeparate(metadata, operand);
                requireScratchSeparate(scratch, operand);
            }
            requireScratchSeparate(scratch, std::as_bytes(values));
            requireScratchSeparate(scratch, metadata);
            requireSeparate(std::as_bytes(output), std::as_bytes(values));
            // Cross-leaf operands remain live through the entire bundle.
            for (const auto &other : projections)
            {
                const auto *other_kernel = dynamic_cast<const CPUW4A4GemmKernel *>(other.kernel);
                if (!other_kernel)
                    throw std::invalid_argument("W4A4 fusion requires native W4A4/H engines on every leaf");
                requireSeparate(std::as_bytes(output), std::as_bytes(other_kernel->weights_.view().blocks()));
                const auto other_scratch = borrowCPUWorkspace(other_kernel->weights_.view().contract(), m, workspace);
                requireScratchSeparate(other_scratch, std::as_bytes(output));
                requireScratchSeparate(other_scratch, std::as_bytes(bias));
                if (&other != &projection)
                    requireSeparate(std::as_bytes(output), std::as_bytes(readFP32(other.output, elements(m, other.n))));
                if (other.bias)
                    requireSeparate(std::as_bytes(output), std::as_bytes(readFP32(other.bias, other.n)));
            }
            if (projection.original_output_slice)
                static_cast<void>(projection.kernel->beginOriginalOutputSliceScope(*projection.original_output_slice));
        }
        for (std::size_t i = 0; i < projections.size(); ++i)
        {
            const auto &projection = projections[i];
            const auto *kernel = static_cast<const CPUW4A4GemmKernel *>(projection.kernel);
            std::construct_at(&invocations[i], ProjectionInvocation{
                &kernel->weights_.view(), borrowCPUWorkspace(kernel->weights_.view().contract(), m, workspace),
                writeFP32(projection.output, elements(m, projection.n)),
                projection.bias ? readFP32(projection.bias, projection.n) : std::span<const float>{}});
        }
        auto execute = [&] {
            std::optional<CPUActivations> prepared;
            for (const auto &invocation : invocations)
            {
                const auto &policy = invocation.weights->contract();
                if (!prepared || !prepared->matches(policy))
                    prepared = prepareCPUActivations(policy, values, m, k, invocation.activation, work);
                projectCPU(*invocation.weights, *prepared, invocation.output, policy.rows());
                epilogue(*prepared, invocation.output, invocation.output, invocation.bias, policy.rows(), 1.0f, 0.0f);
            }
        };
        OMP_WORKSHARE_REGION(execute);
        recordCall("cpu_w4a4_fused_projection_calls");
        return true;
    }

    bool CPUW4A4GemmKernel::multiply_fused_verifier_rows_decode_equivalent(const TensorBase *input,
        const std::vector<TensorProjectionDesc> &projections, int m, int k,
        const IMPIContext *mpi_ctx, DeviceWorkspaceManager *workspace)
    {
        return multiply_fused_tensor(input, projections, m, k, mpi_ctx, workspace);
    }
} // namespace llaminar2::w4a4
