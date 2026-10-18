/**
 * @file ModelKLDCanary.cpp
 * @brief Observe full-vocabulary production logits on an identical forced trajectory.
 *
 * RuntimeInitPhase owns model admission, MPI and captured execution. Explicit
 * diagnostic host-logit publication is enabled at request boundaries; its
 * timings are never used as canonical throughput. The forced-token result owns
 * whether a row was committed, preventing a guessed or double-forwarded token.
 * This diagnostic accepts one explicit CUDA or ROCm device. It uses the
 * production device runner host-logit consumer directly, with no interposition.
 * Frozen workload and artifact paths are explicit tool arguments, independent
 * of a workspace mount. Both backends use the same observer and phase schedule.
 */
#include "app/RuntimeInitPhase.h"
#include "utils/PerfStatsCollector.h"
#include "utils/DebugEnv.h"
#include "config/OrchestrationConfigParser.h"
#include "execution/runner/IOrchestrationRunner.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace llaminar2;
using json = nlohmann::json;

/** @brief Publish one complete report only after its output streams are closed. */
void publish(const std::filesystem::path &path, const json &value)
{
    std::ofstream output(path);
    output << value.dump(2) << '\n';
    if (!output) throw std::runtime_error("Cannot write accuracy report");
}

/** @brief Save one fresh finite full-vocabulary row at a joined result boundary. */
json saveLogits(IOrchestrationRunner &runner, std::ofstream &output, size_t position)
{
    /*
     * lastLogits() authenticates the current device publication, queues the
     * exact producer-event wait on the explicit host bridge, and returns only
     * after the full vocabulary D2H completes. Accuracy has no timing interval
     * to close. An extra benchmark rendezvous incorrectly applies a collective
     * deadline to all already-submitted prefill buckets on a single GPU.
     * Missing publications and transfer failures remain fatal in the native
     * consuming API. The passive lifetime observer records health throughout.
     */
    const float *logits = runner.lastLogits();
    const int vocabulary = runner.vocabSize();
    if (!logits || vocabulary <= 0)
        throw std::runtime_error("No complete host logits at the diagnostic result boundary");
    if (!std::all_of(logits, logits+vocabulary, [](float v) { return std::isfinite(v); }))
        throw std::runtime_error("Nonfinite next-token logits");
    output.write(reinterpret_cast<const char *>(logits), vocabulary*sizeof(float));
    if (!output) throw std::runtime_error("Could not persist full-vocabulary row");
    return {{"consumed_tokens",position}, {"argmax",std::max_element(logits,logits+vocabulary)-logits},
            {"vocabulary",vocabulary}};
}

/** @brief Execute the exact production runner with fixed token histories. */
int main(int argc, char **argv)
{
    json result={{"complete",false},{"passed",false},{"diagnostic_only",true},{"cases",json::array()}};
    std::filesystem::path output;
    try
    {
        if (argc == 2 && std::string_view(argv[1]) == "--help")
        {
            std::cout << "llaminar_model_kld_canary --workload FILE --output DIR -- ENGINE_ARGS\n";
            return 0;
        }
        if (argc < 8 || std::string_view(argv[1]) != "--workload" ||
            std::string_view(argv[3]) != "--output" || std::string_view(argv[5]) != "--")
            throw std::invalid_argument("Require --workload FILE --output DIR -- ENGINE_ARGS");
        const std::filesystem::path input = argv[2];
        output = argv[4];
        std::vector<char *> engine_argv{argv[0]};
        for (int index=6; index<argc; ++index) engine_argv.push_back(argv[index]);
        int engine_argc = static_cast<int>(engine_argv.size());
        engine_argv.push_back(nullptr);
        char **engine_argv_pointer = engine_argv.data();
        std::filesystem::create_directories(output);
        json workload;
        std::ifstream(input) >> workload;
        OrchestrationConfigParser parser;
        auto config = parser.parseArgs(engine_argc,engine_argv.data());
        if (config.mtp_activation_policy != MTPActivationPolicy::Disabled || config.mtp.enabled ||
            config.prefix_cache.enabled || config.activation_precision != "fp32" ||
            config.kv_cache_precision != "fp16" || config.prefill_max_bucket_size != 512 ||
            (!config.tp_devices.empty() || !config.device_for_this_rank.has_value() ||
             (config.device_for_this_rank->device_type != DeviceType::CUDA &&
              config.device_for_this_rank->device_type != DeviceType::ROCm) ||
             config.device_for_this_rank->device_ordinal != 0 ||
             config.device_for_this_rank->hostname != "localhost"))
            throw std::runtime_error("KL observer requires explicit cuda:0 or rocm:0, no TP, M512, FP32 activations, FP16 KV, MTP/prefix off");
        result["contract"] = {{"topology","single_device"},{"device",config.device_for_this_rank->device_type == DeviceType::CUDA ? "cuda:0" : "rocm:0"},
            {"physical_devices",1},{"local_tp_devices",0},{"prefill_max_bucket",512},
            {"activation_precision","fp32"},{"kv_precision","fp16"},{"mtp",false},
            {"prefix_cache",false},{"event_ordered_host_logits",true},
            {"gdn_arithmetic",gdnProjectionArithmeticName(debugEnv().gemm.gdn_projection_arithmetic)},
            {"router_arithmetic",moeRouterArithmeticName(debugEnv().gemm.moe_router_arithmetic)}};
        auto startup = RuntimeInitPhase().execute(config,engine_argc,engine_argv_pointer);
        if (!std::holds_alternative<AppContext>(startup))
            throw std::runtime_error("Production runtime initialization did not return an owner");
        auto context = std::move(std::get<AppContext>(startup));
        auto &runner = *context.runner;
        if (context.mpi_ctx->world_size()!=1)
            throw std::runtime_error("This observer requires exactly one MPI rank");
        runner.setSkipLogitsGatherDecode(false);
        runner.setSkipLogitsGatherPrefill(false);
        if (!runner.prepareForInference())
            throw std::runtime_error("Production preparation failed: "+runner.lastError());
        for (const auto &item : workload.at("cases"))
        {
            const auto prompt=item.at("prompt_ids").get<std::vector<int32_t>>();
            const auto continuation=item.at("continuation_ids").get<std::vector<int32_t>>();
            const auto name=item.at("name").get<std::string>();
            if (prompt.empty() || continuation.size()<2 || name.find('/')!=std::string::npos)
                throw std::runtime_error("Malformed teacher-forcing case");
            runner.clearCache();
            if (!runner.prefill(prompt)) throw std::runtime_error("Prefill failed: "+runner.lastError());
            std::ofstream logits(output/(name+".f32"),std::ios::binary);
            json rows=json::array();
            rows.push_back(saveLogits(runner,logits,prompt.size()));
            for (size_t step=0; step<continuation.size(); ++step)
            {
                const auto forced=runner.forceDecodeToken(continuation[step]);
                if (!forced.success() || forced.tokens.size()!=1 || forced.tokens[0]!=continuation[step])
                    throw std::runtime_error("Forced production token failed: "+forced.error);
                // This experiment deliberately measures serial target arithmetic.
                // Its public deferred contract means the first call publishes a
                // pending token, and each successor forwards exactly that token.
                if (forced.returned_token_commit!=ReturnedTokenCommitState::Pending)
                    throw std::runtime_error("Unexpected forced-token commit contract");
                if (step>0) rows.push_back(saveLogits(runner,logits,prompt.size()+step));
            }
            logits.close();
            result["cases"].push_back({{"name",name},{"prompt_tokens",prompt.size()},
                {"rows",rows},{"continuation_ids",continuation},
                {"logits_file",(output/(name+".f32")).string()}});
            publish(output/"logits.json",result);
        }
        runner.shutdown();
        context.runner.reset();
        if (!PerfStatsCollector::flushFromEnv())
            throw std::runtime_error("Accuracy observer could not publish production-path counters");
        result["passed"]=true;
    }
    catch (const std::exception &error)
    {
        result["error"]=error.what();
        std::cerr << error.what() << '\n';
    }
    result["complete"]=true;
    if (!output.empty()) publish(output/"logits.json",result);
    return result["passed"].get<bool>() ? 0 : 1;
}
