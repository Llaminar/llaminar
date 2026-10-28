/**
 * @file BenchmarkCampaign.cpp
 * @brief Strict campaign admission and incremental ordinary benchmark artifacts.
 *
 * All prompt bytes/token IDs and request policies are sealed before the first
 * cell. Each transition clears request data, publishes one supported policy,
 * and invokes BenchmarkRunner's existing warmup/measurement implementation.
 * Observe cells retain the existing controller's recommendation-only semantics.
 * Atomic per-cell files survive a later native request or output failure.
 */
#include "BenchmarkCampaign.h"
#include "config/OrchestrationConfigDocument.h"
#include "execution/mtp/MTPModelClassPolicy.h"
#include "execution/mtp/MTPRequestSamplingPolicy.h"
#include "execution/mtp/MTPVerifierPolicy.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>

namespace llaminar2
{
namespace
{
using Json = nlohmann::json;
/** @brief Reject ambiguous authoring at its exact input coordinate. */
[[noreturn]] void invalid(const std::string &where, const std::string &reason)
{ throw std::invalid_argument("Benchmark campaign " + where + ": " + reason); }
/** @brief Require one exact object schema; optional implicit overrides are forbidden. */
void fields(const Json &value, std::initializer_list<const char *> expected, const std::string &where)
{
    if (!value.is_object()) invalid(where, "expected an object");
    std::set<std::string> names;
    for (const auto *name : expected) names.emplace(name);
    if (value.size() != names.size()) invalid(where, "missing or unknown fields");
    for (const auto &[key, ignored] : value.items())
        if (!names.contains(key)) invalid(where, "unknown field '" + key + "'");
}
/** @brief Read a nonempty literal string without coercing numbers or booleans. */
std::string text(const Json &value, const std::string &where)
{
    if (!value.is_string() || value.get_ref<const std::string &>().empty()) invalid(where, "expected nonempty string");
    return value.get<std::string>();
}
/** @brief Keep authored IDs inside one owned result directory. */
std::string id(const Json &value, const std::string &where)
{
    auto result = text(value, where);
    if (result.size() > 128 || !std::isalnum(static_cast<unsigned char>(result.front())) ||
        !std::all_of(result.begin(), result.end(), [](unsigned char c) {
            return std::isalnum(c) || c == '-' || c == '_' || c == '.';
        })) invalid(where, "ID must be a bounded filename-safe literal");
    return result;
}
/** @brief Require exact integers in an explicit inclusive range. */
int integer(const Json &value, int low, int high, const std::string &where)
{
    if (!value.is_number_integer() || value < low || value > high) invalid(where, "integer outside supported range");
    return value.get<int>();
}
/** @brief Require finite sampling scalars without string/bool coercion. */
double number(const Json &value, double low, double high, const std::string &where)
{
    if (!value.is_number()) invalid(where, "expected finite number");
    const auto result = value.get<double>();
    if (!std::isfinite(result) || result < low || result > high) invalid(where, "number outside supported range");
    return result;
}
/** @brief Hash small immutable prompt/config metadata through the canonical resolver. */
std::string digest(const std::string &bytes)
{
    OrchestrationConfig source;
    source.prompt = bytes;
    source.prompt_was_explicitly_provided = true;
    return resolveBenchmarkPrompt(source).sha256;
}
/** @brief Write one complete JSON publication; earlier cell artifacts never change. */
void writeAtomic(const std::filesystem::path &path, const Json &value, bool replace = false)
{
    if (!replace && std::filesystem::exists(path)) throw std::runtime_error("Benchmark campaign output already exists: " + path.string());
    auto temporary = path;
    temporary += ".partial";
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot open benchmark campaign output: " + temporary.string());
    out << value.dump(2) << '\n';
    out.close();
    if (!out) throw std::runtime_error("Cannot finish benchmark campaign output: " + temporary.string());
    std::filesystem::rename(temporary, path);
}

}

PreparedBenchmarkCampaign prepareBenchmarkCampaign(const OrchestrationConfig &config, BenchmarkRunner &benchmark)
{
    std::ifstream stream(config.benchmark_campaign_path, std::ios::binary);
    if (!stream) invalid("$", "cannot read " + config.benchmark_campaign_path);
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    if (stream.bad()) invalid("$", "failed reading campaign document");
    const auto bytes = buffer.str();
    std::vector<std::set<std::string>> keys;
    auto callback = [&](int, Json::parse_event_t event, Json &value) {
        if (event == Json::parse_event_t::object_start) keys.emplace_back();
        else if (event == Json::parse_event_t::object_end) keys.pop_back();
        else if (event == Json::parse_event_t::key && !keys.back().insert(value.get<std::string>()).second)
            invalid("$", "duplicate field '" + value.get<std::string>() + "'");
        return true;
    };
    Json document;
    try { document = Json::parse(bytes, callback); }
    catch (const Json::exception &error) { invalid("$", error.what()); }
    fields(document, {"schema", "output_directory", "sampling", "prompts", "cells"}, "$");
    if (text(document.at("schema"), "$.schema") != "llaminar.benchmark-campaign.v1") invalid("$.schema", "unsupported schema");
    PreparedBenchmarkCampaign result;
    result.output_ = text(document.at("output_directory"), "$.output_directory");
    if (!result.output_.is_absolute() || std::filesystem::exists(result.output_) ||
        !std::filesystem::is_directory(result.output_.parent_path())) invalid("$.output_directory", "requires a fresh absolute directory under an existing parent");
    result.digest_ = digest(bytes);
    result.config_ = config;
    const auto &sampling = document.at("sampling");
    fields(sampling, {"temperature", "top_k", "top_p", "seed"}, "$.sampling");
    result.config_.temperature = static_cast<float>(number(sampling.at("temperature"), 0, std::numeric_limits<float>::max(), "$.sampling.temperature"));
    result.config_.top_k = integer(sampling.at("top_k"), 1, std::numeric_limits<int>::max(), "$.sampling.top_k");
    result.config_.top_p = static_cast<float>(number(sampling.at("top_p"), std::numeric_limits<float>::min(), 1, "$.sampling.top_p"));
    result.config_.seed = integer(sampling.at("seed"), 1, std::numeric_limits<int>::max(), "$.sampling.seed");
    SamplingParams law;
    law.temperature = result.config_.temperature;
    law.top_k = result.config_.top_k;
    law.top_p = result.config_.top_p;
    law.seed = result.config_.seed;
    const auto verify = law.is_greedy() ? MTPVerifyMode::Greedy : MTPVerifyMode::SpeculativeSampling;
    if (config.temperature != result.config_.temperature || config.top_k != result.config_.top_k ||
        config.top_p != result.config_.top_p || config.seed != result.config_.seed)
        invalid("$.sampling", "sampling cohort must exactly match the admitted startup sampling parameters");
    const auto initial_sampling_view = resolveMTPSamplingRequestConfig(config.mtp, makeMTPRequestPolicy(config.mtp), law);
    const bool verification_conflict = initial_sampling_view.verify_mode != verify;
    if (config.n_predict <= 0) invalid("$", "campaign requires explicit positive --n-predict");
    if (config.max_seq_len <= 0) invalid("$", "campaign requires an admitted positive context length");
    const auto &prompts = document.at("prompts");
    if (!prompts.is_array() || prompts.empty()) invalid("$.prompts", "expected nonempty prompt table");
    std::map<std::string, std::size_t> prompt_indices;
    std::map<std::string, std::string> partitions_by_tokens;
    for (const auto &entry : prompts)
    {
        fields(entry, {"id", "partition", "text"}, "$.prompts[]");
        auto prompt_id = id(entry.at("id"), "$.prompts[].id");
        if (!prompt_indices.emplace(prompt_id, result.prompts_.size()).second) invalid("$.prompts", "duplicate prompt ID");
        auto partition = text(entry.at("partition"), "$.prompts[].partition");
        if (partition != "train" && partition != "holdout") invalid("$.prompts[].partition", "expected train or holdout");
        auto prompt_config = result.config_;
        prompt_config.prompt = text(entry.at("text"), "$.prompts[].text");
        prompt_config.prompt_was_explicitly_provided = true;
        prompt_config.benchmark_prompt_file_path.clear();
        prompt_config.benchmark_prompt_file_was_provided = false;
        auto input = benchmark.prepareInput(prompt_config);
        if (static_cast<std::uint64_t>(input.tokenIds().size()) + static_cast<std::uint64_t>(config.n_predict) >
            static_cast<std::uint64_t>(config.max_seq_len)) invalid("$.prompts." + prompt_id, "natural token IDs plus decode budget exceed admitted context; truncation is forbidden");
        const auto token_identity = Json(input.tokenIds()).dump();
        const auto [found, inserted] = partitions_by_tokens.emplace(token_identity, partition);
        if (!inserted && found->second != partition) invalid("$.prompts", "identical token history crosses train/holdout partitions");
        result.prompts_.push_back({std::move(prompt_id), std::move(partition), std::move(input)});
    }
    const auto &cells = document.at("cells");
    if (!cells.is_array() || cells.empty()) invalid("$.cells", "expected nonempty explicit request list");
    std::set<std::string> cell_ids;
    for (const auto &entry : cells)
    {
        fields(entry, {"id", "prompt_id", "mtp"}, "$.cells[]");
        auto cell_id = id(entry.at("id"), "$.cells[].id");
        if (cell_id == "summary" || !cell_ids.insert(cell_id).second) invalid("$.cells", "duplicate or reserved cell ID");
        const auto prompt_id = text(entry.at("prompt_id"), "$.cells[].prompt_id");
        const auto prompt = prompt_indices.find(prompt_id);
        if (prompt == prompt_indices.end()) invalid("$.cells[].prompt_id", "unknown immutable prompt ID");
        const auto &mtp = entry.at("mtp");
        if (!mtp.is_object() || !mtp.contains("mode")) invalid("$.cells[].mtp", "missing typed request mode");
        const auto mode = text(mtp.at("mode"), "$.cells[].mtp.mode");
        auto policy = makeMTPRequestPolicy(config.mtp);
        policy.depth_policy.mode = MTPDepthPolicyMode::Fixed;
        policy.draft_tokens = 1;
        if (mode == "off") { fields(mtp, {"mode"}, "$.cells[].mtp"); policy.enabled = false; }
        else if (mode == "fixed")
        {
            fields(mtp, {"mode", "depth"}, "$.cells[].mtp");
            policy.enabled = true;
            policy.draft_tokens = integer(mtp.at("depth"), 1, defaultMTPAdaptiveMaximumDraftDepth(), "$.cells[].mtp.depth");
            policy.depth_policy.min_depth = policy.draft_tokens;
            policy.depth_policy.max_depth = policy.draft_tokens;
            policy.depth_policy.initial_depth = policy.draft_tokens;
        }
        else if (mode == "observe" || mode == "dynamic")
        {
            fields(mtp, {"mode", "minimum_depth", "maximum_depth", "initial_depth"}, "$.cells[].mtp");
            policy.enabled = true;
            policy.depth_policy.mode = mode == "observe" ? MTPDepthPolicyMode::Observe : MTPDepthPolicyMode::Dynamic;
            policy.depth_policy.min_depth = integer(mtp.at("minimum_depth"), 1, defaultMTPAdaptiveMaximumDraftDepth(), "$.cells[].mtp.minimum_depth");
            policy.depth_policy.max_depth = integer(mtp.at("maximum_depth"), policy.depth_policy.min_depth, defaultMTPAdaptiveMaximumDraftDepth(), "$.cells[].mtp.maximum_depth");
            policy.depth_policy.initial_depth = integer(mtp.at("initial_depth"), 0, policy.depth_policy.max_depth, "$.cells[].mtp.initial_depth");
            policy.draft_tokens = policy.depth_policy.max_depth;
        }
        else invalid("$.cells[].mtp.mode", "expected off, fixed, observe or dynamic");
        if (policy.enabled && verification_conflict)
            invalid("$.cells." + cell_id, "sampling cohort conflicts with the admitted verification capability");
        if (const auto error = validateMTPRequestPolicy(policy, config.mtp)) invalid("$.cells." + cell_id, *error);
        result.cells_.push_back({std::move(cell_id), prompt->second, policy});
    }
    return result;
}

BenchmarkCampaignOutcome runBenchmarkCampaign(const PreparedBenchmarkCampaign &campaign,
    IOrchestrationRunner &runner, BenchmarkRunner &benchmark)
{
    BenchmarkCampaignOutcome outcome;
    const auto &config = campaign.configuration();
    SamplingParams law;
    law.temperature = config.temperature; law.top_k = config.top_k;
    law.top_p = config.top_p; law.seed = config.seed;
    // Model/geometry provenance comes from the loaded owner, never filenames.
    const auto admitted_scope = runner.mtpDepthPolicyScope();
    if (!admitted_scope.valid())
        throw std::invalid_argument("Benchmark campaign received malformed runtime training scope");
    const auto model_class = resolveMTPDepthPolicyModelClass(runner.modelContextForDiagnostics());
    const char *model_name = model_class == MTPDepthPolicyModelClass::MoE ? "moe" :
        model_class == MTPDepthPolicyModelClass::Dense ? "dense" : "any";
    auto device = runner.primaryDeviceId().toString();
    std::transform(device.begin(), device.end(), device.begin(), [](unsigned char c) { return std::tolower(c); });
    const auto runtime_document = serializeOrchestrationConfig(runner.config());
    const auto runtime_digest = digest(runtime_document);
    if (!std::filesystem::create_directory(campaign.outputDirectory()))
        throw std::runtime_error("Benchmark campaign refuses existing output directory");
    Json summary{{"schema", "llaminar.benchmark-campaign-result.v1"}, {"complete", false}, {"success", false},
        {"cell_count", campaign.cells().size()}, {"completed_cells", 0}, {"failure_reason", ""},
        {"document_sha256", campaign.documentDigest()}, {"cells", Json::array()}};
    writeAtomic(campaign.outputDirectory() / "summary.json", summary);
    for (std::size_t index = 0; index < campaign.cells().size(); ++index)
    {
        const auto &cell = campaign.cells()[index];
        const auto &prompt = campaign.prompts()[cell.prompt_index];
        try
        {
            auto request = config;
            request.mtp = resolveMTPSamplingRequestConfig(config.mtp, cell.policy, law);
            auto scope = admitted_scope;
            BenchmarkResult result;
            try
            {
                runner.clearCache();
                if (!runner.configureMTPRequestPolicy(cell.policy))
                    throw std::runtime_error("request policy publication rejected: " + runner.lastError());
                const auto installed = runner.mtpRequestPolicy();
                if (installed.enabled != cell.policy.enabled || installed.draft_tokens != cell.policy.draft_tokens ||
                    installed.verify_mode != cell.policy.verify_mode || installed.depth_policy.mode != cell.policy.depth_policy.mode ||
                    installed.depth_policy.min_depth != cell.policy.depth_policy.min_depth ||
                    installed.depth_policy.max_depth != cell.policy.depth_policy.max_depth ||
                    installed.depth_policy.initial_depth != cell.policy.depth_policy.initial_depth)
                    throw std::runtime_error("request policy publication did not install the complete authored execution policy");
                request.mtp = resolveMTPSamplingRequestConfig(runner.config().mtp, installed, law);
                scope = runner.mtpDepthPolicyScope();
                if (!scope.valid()) throw std::runtime_error("request policy produced malformed training scope");
                request.mtp = resolveMTPRequestDepthPolicyContext(request.mtp, runner.primaryDeviceId(), runner.modelContextForDiagnostics());
                request.mtp.depth_policy.scope = scope;
                request.benchmark_json_output_path = (campaign.outputDirectory() / (cell.id + ".json")).string();
                result = benchmark.run(request, prompt.input);
            }
            catch (const std::exception &error)
            {
                result.success = false;
                result.failure_reason = error.what();
            }
            // Even a rejected request publication retains its immutable input
            // identity in the failed cell; no result or completed prefix is erased.
            result.prompt_source = prompt.input.prompt().source;
            result.prompt_file_path = prompt.input.prompt().file_path;
            result.prompt_bytes = prompt.input.prompt().text.size();
            result.prompt_sha256 = prompt.input.prompt().sha256;
            result.prompt_token_ids = prompt.input.tokenIds();
            auto document = Json::parse(benchmarkResultToJsonString(result, &request));
            document["campaign"] = {{"schema", "llaminar.benchmark-campaign.v1"},
                {"document_sha256", campaign.documentDigest()}, {"prompt_id", prompt.id},
                {"partition", prompt.partition}, {"cell_id", cell.id}, {"cell_index", index}};
            document["device"] = device;
            document["model_class"] = model_name;
            document["sampling_mode"] = law.is_greedy() ? "greedy" : "stochastic";
            document["architecture"] = runner.architecture();
            document["runtime_configuration_sha256"] = runtime_digest;
            document["runtime_configuration"] = Json::parse(runtime_document);
            document["training_scope_generic_reason"] = scope.specified() ? "" : runner.mtpDepthPolicyScopeGenericReason();
            document["training_scope"] = {{"verifier_width_policy", mtpDepthPolicyWidthPolicyName(scope.verifier_width_policy)},
                {"tp_width", scope.tp_width}, {"pp_stages", scope.pp_stages}, {"retained_depth", scope.retained_depth},
                {"hidden_dim", scope.hidden_dim}, {"main_layers", scope.main_layers},
                {"expert_count", scope.expert_count}, {"experts_per_token", scope.experts_per_token},
                {"compute_capability_major", scope.compute_capability_major}, {"compute_capability_minor", scope.compute_capability_minor}};
            if (request.shared_predictor) document["config"]["shared_predictor"] = request.shared_predictor->path();
            document["config"]["mtp_terminal_head_policy"] = mtpTerminalHeadPolicyToString(request.mtp.terminal_head_policy);
            document["config"]["activation_precision"] = request.activation_precision;
            document["config"]["kv_cache_precision"] = request.kv_cache_precision;
            const bool fixed = request.mtp.depth_policy.mode == MTPDepthPolicyMode::Fixed;
            const auto initial = request.mtp.enabled ? resolveMTPDepthPolicyInitialDepth(
                request.mtp.depth_policy, request.mtp.draft_tokens, request.mtp.verify_mode) : 0;
            document["request_policy"] = {{"mode", !request.mtp.enabled ? "off" : mtpDepthPolicyModeToString(request.mtp.depth_policy.mode)},
                {"enabled", request.mtp.enabled}, {"verify_mode", mtpVerifyModeToString(request.mtp.verify_mode)},
                {"draft_tokens", request.mtp.enabled ? request.mtp.draft_tokens : 0},
                {"minimum_depth", !request.mtp.enabled ? 0 : fixed ? request.mtp.draft_tokens : request.mtp.depth_policy.min_depth},
                {"maximum_depth", !request.mtp.enabled ? 0 : fixed ? request.mtp.draft_tokens : request.mtp.depth_policy.max_depth},
                {"initial_depth", !request.mtp.enabled ? 0 : fixed ? request.mtp.draft_tokens : request.mtp.depth_policy.initial_depth},
                {"resolved_initial_depth", initial}};
            // These existing JSON keys describe effective execution, while retained
            // capacity remains the unmodified physical startup authority.
            document["config"]["mtp_draft_tokens"] = document["request_policy"]["draft_tokens"];
            document["config"]["mtp_min_draft_tokens"] = document["request_policy"]["minimum_depth"];
            document["config"]["mtp_max_draft_tokens"] = document["request_policy"]["maximum_depth"];
            document["config"]["mtp_initial_draft_tokens"] = document["request_policy"]["initial_depth"];
            writeAtomic(campaign.outputDirectory() / (cell.id + ".json"), document);
            summary["cells"].push_back({{"id", cell.id}, {"file", cell.id + ".json"}, {"success", result.success}});
            benchmark.printResults(result);
            if (result.success) ++outcome.completed_cells;
            else outcome.failure_reason = "cell '" + cell.id + "': " + result.failure_reason;
            summary["completed_cells"] = outcome.completed_cells;
            summary["failure_reason"] = outcome.failure_reason;
            summary["complete"] = !result.success || index + 1 == campaign.cells().size();
            summary["success"] = result.success && index + 1 == campaign.cells().size();
            writeAtomic(campaign.outputDirectory() / "summary.json", summary, true);
            if (!result.success) return outcome;
        }
        catch (const std::exception &error)
        {
            outcome.failure_reason = "cell '" + cell.id + "': " + error.what();
            summary["complete"] = true;
            summary["success"] = false;
            summary["completed_cells"] = outcome.completed_cells;
            summary["failure_reason"] = outcome.failure_reason;
            writeAtomic(campaign.outputDirectory() / "summary.json", summary, true);
            return outcome;
        }
    }
    outcome.success = true;
    return outcome;
}
}
