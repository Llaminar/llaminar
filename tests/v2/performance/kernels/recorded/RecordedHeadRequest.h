/**
 * @file RecordedHeadRequest.h
 * @brief Closed, device-free request for replaying one recorded native inference row.
 *
 * This is an explicit performance-tool contract, not a serving configuration or
 * an inference receipt. The caller binds the original native observation and
 * artifact stats. Execution independently checks those stats, GGUF metadata,
 * prepared ABI, captured LM-head byte oracle and immutable replay inputs. A
 * shared recorded row across slice widths is always local_shape evidence.
 */
#pragma once
#include "RecordedObservation.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace llaminar2::test::recorded_head
{
// Preserve the existing head-tool API while all recorded operations share one
// canonical metadata parser and original-export receipt validator.
using recorded_observation::FileIdentity;
using recorded_observation::require;
using recorded_observation::integer;
using recorded_observation::unsignedInteger;
using recorded_observation::fileIdentity;
using recorded_observation::csvFields;
using recorded_observation::validateObservationReceipt;
/** @brief Explicit setup files plus original arithmetic/source coordinates. */
struct Request
{
    std::string backend,output,observation,phase,source_tensor;
    int ordinal=0,original_n=0,k=0,degree=0,participant=0,first=0,n=0,serial_width=0;
    int physical_rows=0,selected_row=0,warmup=0,samples=0,step=0;
    std::string snapshot_backend,snapshot_key,logits_key;
    std::uint64_t generation=0,model_context=0;
    FileIdentity input,logits,snapshot_manifest,observation_receipt;
    std::vector<FileIdentity> model_artifacts;
    nlohmann::json precision,document;
};
/**
 * @brief Validate an external qualified native row and original equal output slice.
 * @param j Explicit request loaded before any backend/model initialization.
 * @return Closed immutable work; no default baseline or synthetic origin is invented.
 * @note Nonfinite FP32 values are not filtered or changed. The original stage is
 *       the byte oracle; this parser introduces no new finite-domain policy.
 */
inline Request parse(const nlohmann::json &j)
{
    require(j.at("schema")=="llaminar.recorded_head_projection.v1" && j.at("scope")=="local_shape" &&
        j.at("input_origin")=="llaminar_captured_stage_output" && j.at("producer_stage")=="final_norm" &&
        j.at("semantic_tensor")=="FINAL_NORM" && j.at("dtype")=="float32_le" &&
        j.at("head_operand_precision")=="single_q8" && j.at("source_format")=="Q8_0" &&
        j.at("source_gguf_type")==8 && j.at("source_execution_codebook")==19 &&
        j.at("mtp_enabled")==false && j.at("mtp_graph_capacity")==0,
        "Recorded head accepts ordinary captured FINAL_NORM/native Q8_0/SingleQ8 only");
    Request r; r.document=j;
    r.backend=j.at("backend").get<std::string>();r.ordinal=integer(j.at("device_ordinal"));
    r.output=j.at("output").get<std::string>();r.observation=j.at("observation_id").get<std::string>();
    r.phase=j.at("phase").get<std::string>();r.source_tensor=j.at("source_tensor").get<std::string>();
    r.original_n=integer(j.at("original_n"));r.k=integer(j.at("k"));
    r.serial_width=integer(j.at("serial_equivalent_partition_width"));
    r.degree=integer(j.at("slice_degree"));r.participant=integer(j.at("slice_participant"));
    r.first=integer(j.at("first_column"));r.n=integer(j.at("local_n"));
    r.physical_rows=integer(j.at("observation_physical_rows"));r.selected_row=integer(j.at("observation_selected_row"));
    r.warmup=integer(j.at("warmup"));r.samples=integer(j.at("samples"));
    r.step=integer(j.at("step"));r.snapshot_backend=j.at("snapshot_backend").get<std::string>();
    r.snapshot_key=j.at("snapshot_key").get<std::string>();r.logits_key=j.at("logits_key").get<std::string>();
    r.generation=unsignedInteger(j.at("generation"));r.model_context=unsignedInteger(j.at("model_context"));
    r.input=fileIdentity(j.at("input"));r.logits=fileIdentity(j.at("logits"));
    r.snapshot_manifest=fileIdentity(j.at("snapshot_manifest"));r.observation_receipt=fileIdentity(j.at("observation_receipt"));
    for(const auto &f:j.at("model_artifacts"))r.model_artifacts.push_back(fileIdentity(f));
    r.precision=j.at("six_precision_selectors");
    require((r.backend=="cuda"||r.backend=="rocm") && r.ordinal>=0 &&
        !r.output.empty() && r.output.front()=='/' && !r.observation.empty() && r.generation && r.model_context &&
        (r.phase=="decode"||r.phase=="prefill_terminal") && r.source_tensor=="output.weight" &&
        r.original_n>0 && r.k>0 && r.k%32==0 && (r.degree==1||r.degree==2||r.degree==4) &&
        r.participant>=0 && r.participant<r.degree && r.original_n%r.degree==0 &&
        r.n==r.original_n/r.degree && r.first==r.participant*r.n && r.serial_width>=0 && r.serial_width<=r.n && r.physical_rows>0 &&
        r.selected_row>=0 && r.selected_row<r.physical_rows &&
        (r.phase!="decode" || (r.physical_rows==1 && r.selected_row==0)) &&
        !r.snapshot_backend.empty() && r.snapshot_key=="FINAL_NORM" && r.logits_key=="LM_HEAD" && (r.phase=="decode"?r.step>=0:r.step==-1) &&
        r.warmup>=1 && r.warmup<=1024 && r.samples>=31 && r.samples<=4096 && !r.model_artifacts.empty(),
        "Recorded head geometry, trajectory, samples or source identity is incomplete");
    require(std::uint64_t(r.physical_rows)<=std::numeric_limits<std::uint64_t>::max()/4/std::uint64_t(r.k),
        "Recorded snapshot byte extent overflows");
    require(std::uint64_t(r.physical_rows)*std::uint64_t(r.k)*sizeof(float)==r.input.bytes &&
        std::uint64_t(r.original_n)*sizeof(float)==r.logits.bytes && r.precision.is_object() && r.precision.size()==6,
        "Recorded head requires complete FP32 snapshot, one full-vocabulary oracle row and all six precision selectors");
    std::vector<std::string> seen;
    for(const auto &f:r.model_artifacts)
    {require(std::find(seen.begin(),seen.end(),f.path)==seen.end(),"Duplicate original model artifact");seen.push_back(f.path);}
    for(const auto &[key,value]:r.precision.items())
        require(!key.empty() && value.is_string() && !value.get<std::string>().empty(),"Precision selector identity is incomplete");
    return r;
}
/**
 * @brief Bind exactly one canonical row for each actual input and head oracle.
 * @param r Closed immutable request, whose paths already have complete stat identities.
 * @param lines Original complete manifest lines, including exact scalar-type header.
 * @note A duplicate matching row is rejected rather than silently selecting a generation.
 */
inline void validateSnapshotManifest(const Request &r,const std::vector<std::string> &lines)
{
    recorded_observation::validateSnapshotBindings(r.snapshot_backend,r.phase=="decode"?"decode":"prefill",r.step,
        {{r.snapshot_key,r.input,r.input.bytes/4},{r.logits_key,r.logits,r.logits.bytes/4}},lines);
}
/** @brief A custom CLI branch consumes the explicit request file, never outer environment state. */
bool requested(int argc,char **argv) noexcept;
/** @brief Run the qualified local-shape diagnostic through the existing performance executable. */
int run(int argc,char **argv);
} // namespace llaminar2::test::recorded_head
