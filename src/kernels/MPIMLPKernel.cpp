#include "MPIMLPKernel.h"
#include "../adaptive_matmul.h"
#include "../backends/prefill_backend.h"
#include "../backends/inference_backend.h"
#include "../logger.h"
#include "../tensors/tp_generic_matmul_executor.h" // Added for TPGemmExecutor / TPGemmExecConfig
#include <algorithm>
#include <cmath>
#include "../utils/perf_counters.h"

namespace llaminar
{

    MPIMLPKernel::MPIMLPKernel()
    {
        initializeMPI();
        LOG_DEBUG("MPIMLPKernel initialized on rank " << rank_ << "/" << size_);
    }

    bool MPIMLPKernel::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                               std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        if (!validate(inputs, outputs))
        {
            LOG_ERROR("MPIMLPKernel validation failed");
            return false;
        }

        auto input = inputs[0];
        auto w_gate = inputs[1];
        auto w_up = inputs[2];
        auto w_down = inputs[3];
        auto output = outputs[0];

        size_t seq_len = input->shape()[0];
        size_t d_model = input->shape()[1];
        // Capture full (unpartitioned) feed-forward dimension before any TP slicing
        size_t global_d_ff_full = w_gate->shape()[1];

        // Detect sharding of feed-forward dimension (column sharding of gate/up)
        bool ff_sharded = false;
        size_t global_d_ff = 0; // only valid if sharded
        size_t local_d_ff = w_gate->shape()[1];
        if (auto *ss = dynamic_cast<ShardedSimpleTensor *>(w_gate.get()))
        {
            const ShardSpec &spec = ss->shard_spec();
            if (spec.is_sharded())
            {
                ff_sharded = true;
                global_d_ff = spec.global_dim;
                local_d_ff = spec.local_dim;
            }
        }

        LOG_DEBUG("MPIMLPKernel processing: seq_len=" << seq_len
                                                      << ", d_model=" << d_model << ", local_d_ff=" << local_d_ff);

        // Create temporary buffers for intermediate results
        auto gate_proj = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(local_d_ff)});
        auto up_proj = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(local_d_ff)});
        auto activated = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(local_d_ff)});
        // Down projection always produces the full d_model columns locally (partial over rows of w_down if row sharded)
        auto local_output = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(d_model)});

        // Backend selection: treat large seq_len as prefill (use PrefillBackend) else InferenceBackend.
        bool is_prefill_like = seq_len >= static_cast<size_t>(debugEnv().cosma.prefill_threshold);
        static bool logged_once = false;
        if (!logged_once && getRank() == 0)
        {
            logged_once = true;
            LOG_INFO("BACKEND_DECISION_SUMMARY component=MLP seq_len=" << seq_len
                                                                       << " threshold=" << debugEnv().cosma.prefill_threshold
                                                                       << " path=" << (is_prefill_like ? "prefill" : "inference")
                                                                       << " prefill_backend=cpu_stub inference_backend=cpu_stub fallback=adaptive_matmul");
        }
        auto prefill_backend = PrefillBackendFactory::create();
        auto infer_backend = InferenceBackendFactory::create();

        auto run_matmul = [&](const char *tag,
                              const std::shared_ptr<TensorBase> &A,
                              const std::shared_ptr<TensorBase> &B,
                              const std::shared_ptr<TensorBase> &C,
                              size_t M, size_t N, size_t K)
        {
            bool ok = false;
            if (is_prefill_like)
            {
                PrefillOpDesc desc;
                desc.kind = PrefillOpKind::MatMul;
                desc.M = (int64_t)M;
                desc.N = (int64_t)N;
                desc.K = (int64_t)K;
                desc.is_prefill = true;
                PrefillLaunchContext ctx{A->data(), B->data(), C->data()};
                auto decision = prefill_backend->launch(desc, ctx);
                if (decision.status == PrefillStatus::Success)
                    ok = true;
                else
                {
                    LOG_WARN("MLP " << tag << " backend fallback due to status=" << (int)decision.status << " reason=" << decision.reason);
                }
            }
            if (!ok)
            {
                if (!adaptive_matmul(A->data(), B->data(), C->data(), (int)M, (int)N, (int)K, is_prefill_like))
                {
                    LOG_ERROR("MLP " << tag << " adaptive_matmul failed M=" << M << " N=" << N << " K=" << K);
                    throw std::runtime_error("MLP matmul failure");
                }
            }
        };

        const auto &env = debugEnv();
        bool tp_mlp_enabled = env.mlp_tp.enable && env.mlp_tp.partitions > 1;
        int tp_parts = env.mlp_tp.partitions;
        // Gate/up TP only applies if weights are not already feature sharded externally
        // (If already sharded, we treat current local slice as authoritative.)
        bool weight_already_sharded = ff_sharded; // reuse earlier detection
        if (tp_mlp_enabled && weight_already_sharded)
        {
            if (getRank() == 0)
                LOG_INFO("[MLP_TP] Requested TP but weights already sharded; skipping TP executor");
            tp_mlp_enabled = false;
        }

        // Track TP slice offset (within d_ff) when partitioning columns of gate/up
        size_t tp_ff_offset = 0;
        if (tp_mlp_enabled)
        {
            if (getRank() == 0)
            {
                LOG_INFO("MLP_TP_ENABLED partitions=" << tp_parts
                                                      << " d_model=" << d_model
                                                      << " global_d_ff(local)=" << local_d_ff);
            }
            // Use generic TP GEMM executor abstraction (column split for gate & up)
            size_t global_ff = global_d_ff_full; // full dimension prior to slicing
            // Partition N dimension (D_ff)
            // We'll run each local partition sequentially on this rank? No: simple simulation maps rank->tp_rank by modulo.
            int tp_rank = getRank() % tp_parts;
            using ExecCfg = TPGemmExecConfig;
            using Mode = TPGemmExecConfig::Mode;
            auto matmul_local = [&](const float *A, const float *B, float *C, std::size_t M, std::size_t N, std::size_t K) -> bool
            { return adaptive_matmul(A, B, C, (int)M, (int)N, (int)K, is_prefill_like); };
            ExecCfg cfg_gate{Mode::Column, tp_parts, tp_rank};
            TPGemmExecutor gate_exec(matmul_local, cfg_gate, seq_len, global_ff, d_model);
            ExecCfg cfg_up{Mode::Column, tp_parts, tp_rank};
            TPGemmExecutor up_exec(matmul_local, cfg_up, seq_len, global_ff, d_model);
            auto gate_part = gate_exec.run(input->data(), w_gate->data());
            auto up_part = up_exec.run(input->data(), w_up->data());
            // Resize gate_proj/up_proj to local dims and copy results
            if (gate_part.N_local != up_part.N_local)
                throw std::runtime_error("MLP_TP gate/up local dims mismatch");
            local_d_ff = gate_part.N_local;              // effective local slice width
            tp_ff_offset = gate_part.partB.local_offset; // record slice offset for down projection slice
            // Reallocate intermediates sized to local slice
            gate_proj = TensorFactory::create_simple({(int)seq_len, (int)local_d_ff});
            up_proj = TensorFactory::create_simple({(int)seq_len, (int)local_d_ff});
            std::memcpy(gate_proj->data(), gate_part.buffer.data(), sizeof(float) * seq_len * local_d_ff);
            std::memcpy(up_proj->data(), up_part.buffer.data(), sizeof(float) * seq_len * local_d_ff);
            // Adjust activated buffer shape
            activated = TensorFactory::create_simple({(int)seq_len, (int)local_d_ff});
        }
        else
        {
            // Step 1: Gate projection
            run_matmul("gate", input, w_gate, gate_proj, seq_len, local_d_ff, d_model);
            // Step 2: Up projection
            run_matmul("up", input, w_up, up_proj, seq_len, local_d_ff, d_model);
        }

        // Step 3: Apply SwiGLU activation locally (no MPI communication)
        applySwiGLU(gate_proj, up_proj, activated, seq_len, local_d_ff);

        // Step 4: Down projection
        if (tp_mlp_enabled)
        {
            // Down projection over this partition slice only: use corresponding rows of w_down
            const float *w_down_slice = w_down->data() + tp_ff_offset * d_model;
            if (!adaptive_matmul(activated->data(), w_down_slice, local_output->data(), (int)seq_len, (int)d_model, (int)local_d_ff, is_prefill_like))
            {
                LOG_ERROR("MLP down matmul (TP slice) failed seq_len=" << seq_len << " d_model=" << d_model << " local_d_ff=" << local_d_ff);
                throw std::runtime_error("MLP TP down matmul failure");
            }
        }
        else
        {
            run_matmul("down", activated, w_down, local_output, seq_len, d_model, local_d_ff);
        }

        // Step 5: Gather final output using MPI_Allreduce (only if this rank produced partial result)
        // Heuristic: if down weight is row-sharded (its first dimension < global d_ff) we need Allreduce.
        bool need_allreduce = true;
        if (auto *ss_down = dynamic_cast<ShardedSimpleTensor *>(w_down.get()))
        {
            const ShardSpec &spec = ss_down->shard_spec();
            if (spec.is_sharded())
            {
                // row sharded feed-forward contributions require summation
                need_allreduce = true;
            }
            else
            {
                need_allreduce = true; // replicated path; still Allreduce acts as identity but cheap for small sizes
            }
        }
        // If TP MLP enabled we always need reduction across partitions even if original weights replicated
        if (tp_mlp_enabled)
            need_allreduce = true;
        if (need_allreduce)
        {
            gatherFinalOutput(local_output, output, seq_len, d_model);
        }
        else
        {
            // Direct copy (should not happen with current strategies but keeps logic explicit)
            std::memcpy(output->data(), local_output->data(), sizeof(float) * seq_len * d_model);
        }

        // Optional local parity validation (small tensors only) when tp enabled
        if (tp_mlp_enabled && env.mlp_tp.validate && getRank() == 0)
        {
            // Reference uses full (global) feed-forward dimension
            size_t elems = seq_len * d_model * global_d_ff_full;
            if (elems <= 8ull * 1024ull * 1024ull) // guard very large shapes
            {
                // Recompute reference full path (serial) using adaptive matmul without partitioning
                std::vector<float> ref_gate(seq_len * global_d_ff_full);
                std::vector<float> ref_up(seq_len * global_d_ff_full);
                std::vector<float> ref_act(seq_len * global_d_ff_full);
                if (!adaptive_matmul(input->data(), w_gate->data(), ref_gate.data(), (int)seq_len, (int)global_d_ff_full, (int)d_model, is_prefill_like))
                    LOG_WARN("MLP_TP parity: reference gate matmul failed");
                if (!adaptive_matmul(input->data(), w_up->data(), ref_up.data(), (int)seq_len, (int)global_d_ff_full, (int)d_model, is_prefill_like))
                    LOG_WARN("MLP_TP parity: reference up matmul failed");
                // silu * up over full dimension
                for (size_t i = 0; i < seq_len * global_d_ff_full; ++i)
                {
                    float g = ref_gate[i];
                    ref_act[i] = (g / (1.0f + std::exp(-g))) * ref_up[i];
                }
                std::vector<float> ref_out(seq_len * d_model);
                if (!adaptive_matmul(ref_act.data(), w_down->data(), ref_out.data(), (int)seq_len, (int)d_model, (int)global_d_ff_full, is_prefill_like))
                    LOG_WARN("MLP_TP parity: reference down matmul failed");
                // Compare with output->data()
                double diff_sq = 0.0, ref_sq = 0.0, max_abs = 0.0;
                size_t worst = 0;
                size_t total = seq_len * d_model;
                const float *cur = output->data();
                for (size_t i = 0; i < total; ++i)
                {
                    double d = (double)cur[i] - (double)ref_out[i];
                    double ad = std::fabs(d);
                    if (ad > max_abs)
                    {
                        max_abs = ad;
                        worst = i;
                    }
                    diff_sq += d * d;
                    ref_sq += (double)ref_out[i] * (double)ref_out[i];
                }
                double rel_l2 = ref_sq > 0.0 ? std::sqrt(diff_sq) / std::sqrt(ref_sq) : 0.0;
                LOG_INFO("MLP_TP_PARITY rel_l2=" << rel_l2 << " max_abs=" << max_abs << " worst_index=" << worst);
            }
            else if (getRank() == 0)
            {
                LOG_INFO("MLP_TP_PARITY skip large_tensor elems=" << elems);
            }
        }

        return true;
    }

    bool MPIMLPKernel::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        if (inputs.size() != 4 || outputs.size() != 1)
        {
            LOG_ERROR("MPIMLPKernel: Expected 4 inputs and 1 output, got "
                      << inputs.size() << " inputs and " << outputs.size() << " outputs");
            return false;
        }

        // Basic null checks
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            if (!inputs[i])
            {
                LOG_ERROR("MPIMLPKernel: Input " << i << " is null");
                return false;
            }
        }

        if (!outputs[0])
        {
            LOG_ERROR("MPIMLPKernel: Output is null");
            return false;
        }

        auto input = inputs[0];
        auto w_gate = inputs[1];
        auto w_up = inputs[2];
        auto w_down = inputs[3];
        auto output = outputs[0];

        // Check input is 2D [seq_len, d_model]
        if (input->shape().size() != 2)
        {
            LOG_ERROR("MPIMLPKernel: Input must be 2D, got " << input->shape().size() << " dimensions");
            return false;
        }

        size_t seq_len = input->shape()[0];
        size_t d_model = input->shape()[1];

        // Check weight dimensions for distributed setting
        // Shard-aware validation: allow gate/up to be column sharded
        size_t gate_rows = w_gate->shape()[0];
        size_t gate_cols = w_gate->shape()[1];
        if (gate_rows != d_model)
        {
            LOG_ERROR("MPIMLPKernel: Gate weight row mismatch expected d_model=" << d_model << " got " << gate_rows);
            return false;
        }
        // Up must match gate local columns
        if (w_up->shape().size() != 2 || w_up->shape()[0] != d_model || w_up->shape()[1] != gate_cols)
        {
            LOG_ERROR("MPIMLPKernel: Up weight mismatch; gate local cols=" << gate_cols << " up shape=[" << w_up->shape()[0] << "," << w_up->shape()[1] << "]");
            return false;
        }
        // Down projection may be row sharded: its first dim must equal gate_cols (local) if sharded or global if replicated
        size_t down_rows = w_down->shape()[0];
        if (down_rows != gate_cols)
        {
            // If down is sharded, its local rows should still equal gate_cols (local slice) for consistency of local matmul.
            LOG_ERROR("MPIMLPKernel: Down weight row mismatch expected " << gate_cols << " got " << down_rows);
            return false;
        }
        if (w_down->shape()[1] != d_model)
        {
            LOG_ERROR("MPIMLPKernel: Down weight col mismatch expected d_model=" << d_model << " got " << w_down->shape()[1]);
            return false;
        }

        // Check output shape
        if (output->shape().size() != 2 || output->shape()[0] != seq_len || output->shape()[1] != d_model)
        {
            LOG_ERROR("MPIMLPKernel: Output shape mismatch. Expected ["
                      << seq_len << ", " << d_model << "], got ["
                      << output->shape()[0] << ", " << output->shape()[1] << "]");
            return false;
        }

        return true;
    }

    void MPIMLPKernel::computeGateProjection(const std::shared_ptr<TensorBase> &input,
                                             const std::shared_ptr<TensorBase> &w_gate,
                                             std::shared_ptr<TensorBase> &gate_output,
                                             size_t seq_len, size_t d_model, size_t local_d_ff)
    {
        // Use adaptive matrix multiplication: gate_output = input * w_gate
        // input: [seq_len, d_model], w_gate: [d_model, local_d_ff] -> gate_output: [seq_len, local_d_ff]
        if (!adaptive_matmul(input->data(), w_gate->data(), gate_output->data(),
                             seq_len, local_d_ff, d_model, false))
        {
            LOG_ERROR("MPIMLPKernel: Gate projection matrix multiplication failed");
            throw std::runtime_error("Gate projection matrix multiplication failed");
        }

        LOG_DEBUG("Rank " << rank_ << " completed gate projection: ["
                          << seq_len << ", " << d_model << "] * [" << d_model << ", " << local_d_ff
                          << "] -> [" << seq_len << ", " << local_d_ff << "]");
    }

    void MPIMLPKernel::computeUpProjection(const std::shared_ptr<TensorBase> &input,
                                           const std::shared_ptr<TensorBase> &w_up,
                                           std::shared_ptr<TensorBase> &up_output,
                                           size_t seq_len, size_t d_model, size_t local_d_ff)
    {
        // Use adaptive matrix multiplication: up_output = input * w_up
        // input: [seq_len, d_model], w_up: [d_model, local_d_ff] -> up_output: [seq_len, local_d_ff]
        if (!adaptive_matmul(input->data(), w_up->data(), up_output->data(),
                             seq_len, local_d_ff, d_model, false))
        {
            LOG_ERROR("MPIMLPKernel: Up projection matrix multiplication failed");
            throw std::runtime_error("Up projection matrix multiplication failed");
        }

        LOG_DEBUG("Rank " << rank_ << " completed up projection: ["
                          << seq_len << ", " << d_model << "] * [" << d_model << ", " << local_d_ff
                          << "] -> [" << seq_len << ", " << local_d_ff << "]");
    }

    void MPIMLPKernel::applySwiGLU(const std::shared_ptr<TensorBase> &gate_output,
                                   const std::shared_ptr<TensorBase> &up_output,
                                   std::shared_ptr<TensorBase> &activated_output,
                                   size_t seq_len, size_t local_d_ff)
    {
        // Apply SwiGLU: activated_output = silu(gate_output) * up_output
        // This is element-wise operation, no MPI communication needed
        const float *gate_data = gate_output->data();
        const float *up_data = up_output->data();
        float *output_data = activated_output->data();

        size_t total_elements = seq_len * local_d_ff;
        // OpenMP parallelization: use single thread for very small ops to avoid overhead
        if (total_elements < 8192)
        {
            for (size_t i = 0; i < total_elements; ++i)
            {
                output_data[i] = silu(gate_data[i]) * up_data[i];
            }
        }
        else
        {
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < total_elements; ++i)
            {
                // vectorization hint
                output_data[i] = silu(gate_data[i]) * up_data[i];
            }
        }

        LOG_DEBUG("Rank " << rank_ << " completed SwiGLU activation for "
                          << total_elements << " elements");
    }

    void MPIMLPKernel::computeDownProjection(const std::shared_ptr<TensorBase> &activated_input,
                                             const std::shared_ptr<TensorBase> &w_down,
                                             std::shared_ptr<TensorBase> &local_output,
                                             size_t seq_len, size_t local_d_ff, size_t d_model)
    {
        // Use adaptive matrix multiplication: local_output = activated_input * w_down
        // activated_input: [seq_len, local_d_ff], w_down: [local_d_ff, d_model] -> local_output: [seq_len, d_model]
        if (!adaptive_matmul(activated_input->data(), w_down->data(), local_output->data(),
                             seq_len, d_model, local_d_ff, false))
        {
            LOG_ERROR("MPIMLPKernel: Down projection matrix multiplication failed");
            throw std::runtime_error("Down projection matrix multiplication failed");
        }

        LOG_DEBUG("Rank " << rank_ << " completed down projection: ["
                          << seq_len << ", " << local_d_ff << "] * [" << local_d_ff << ", " << d_model
                          << "] -> [" << seq_len << ", " << d_model << "]");
    }

    void MPIMLPKernel::gatherFinalOutput(const std::shared_ptr<TensorBase> &local_output,
                                         std::shared_ptr<TensorBase> &global_output,
                                         size_t seq_len, size_t d_model)
    {
        // Sum contributions from all ranks using MPI_Allreduce
        checkMPIError(PerfAllreduce(local_output->data(), global_output->data(),
                                    static_cast<int>(seq_len * d_model), MPI_FLOAT, MPI_SUM, getComm()),
                      "MPI_Allreduce in gatherFinalOutput");

        LOG_DEBUG("Rank " << rank_ << " completed final output gather for "
                          << seq_len << " x " << d_model << " elements");
    }

    float MPIMLPKernel::silu(float x) const
    {
        // SiLU (Swish) activation: x * sigmoid(x) = x / (1 + exp(-x))
        return x / (1.0f + std::exp(-x));
    }

    size_t MPIMLPKernel::calculateLocalDff(size_t global_d_ff) const
    {
        // Distribute d_ff dimension across ranks
        size_t base_size = global_d_ff / size_;
        size_t remainder = global_d_ff % size_;

        // First 'remainder' ranks get an extra element
        return (rank_ < static_cast<int>(remainder)) ? base_size + 1 : base_size;
    }

} // namespace llaminar