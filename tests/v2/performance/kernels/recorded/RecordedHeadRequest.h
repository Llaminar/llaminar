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
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace llaminar2::test::recorded_head
{
/** @brief Stat-bound file identity; contents are never hashed by this consumer. */
struct FileIdentity
{
    std::string path;
    std::uint64_t device=0,inode=0,bytes=0;
    std::int64_t mtime_ns=0,ctime_ns=0;
};
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
/** @brief Fail closed for a missing, wrong-typed or out-of-domain request field. */
inline void require(bool value,const char *message)
{if(!value)throw std::invalid_argument(message);}
/** @brief Reject numeric narrowing/coercion instead of truncating request axes. */
inline int integer(const nlohmann::json &v)
{
    require(v.is_number_integer(),"Recorded integer field has wrong scalar type");
    if(v.is_number_unsigned())require(v.get<std::uint64_t>()<=std::uint64_t(std::numeric_limits<int>::max()),"Recorded integer overflows");
    const auto n=v.get<std::int64_t>();
    require(n>=std::numeric_limits<int>::min() && n<=std::numeric_limits<int>::max(),"Recorded integer overflows");return int(n);
}
/** @brief Exact unsigned identity field; negative/float inputs cannot become wrapped integers. */
inline std::uint64_t unsignedInteger(const nlohmann::json &v)
{
    require(v.is_number_integer() && (v.is_number_unsigned() || v.get<std::int64_t>()>=0),"Recorded identity requires an unsigned integer");
    return v.get<std::uint64_t>();
}
/** @brief Parse one complete absolute file identity without opening its payload. */
inline FileIdentity fileIdentity(const nlohmann::json &j)
{
    FileIdentity f{j.at("path").get<std::string>(),unsignedInteger(j.at("device")),
        unsignedInteger(j.at("inode")),unsignedInteger(j.at("bytes")),
        j.at("mtime_ns").get<std::int64_t>(),j.at("ctime_ns").get<std::int64_t>()};
    require(!f.path.empty() && f.path.front()=='/' && f.inode && f.bytes && f.mtime_ns>0 && f.ctime_ns>0,
        "Recorded head requires absolute, complete file stat identities");
    return f;
}
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
/** @brief Parse the canonical quoted CSV dialect without trimming or coercing fields. */
inline std::vector<std::string> csvFields(const std::string &line)
{
    std::vector<std::string> out;std::string field;bool quoted=false,closed=false;
    for(std::size_t i=0;i<line.size();++i)
    {
        const char c=line[i];
        if(quoted)
        {if(c=='"'){if(i+1<line.size() && line[i+1]=='"'){field+='"';++i;}else{quoted=false;closed=true;}}else field+=c;}
        else if(c==','){out.push_back(field);field.clear();closed=false;}
        else if(c=='"'){require(field.empty()&&!closed,"Malformed canonical snapshot CSV quote");quoted=true;}
        else {require(!closed,"Unexpected bytes after snapshot CSV quote");field+=c;}
    }
    require(!quoted,"Unterminated canonical snapshot CSV quote");out.push_back(field);return out;
}
/**
 * @brief Bind exactly one canonical row for each actual input and head oracle.
 * @param r Closed immutable request, whose paths already have complete stat identities.
 * @param lines Original complete manifest lines, including exact scalar-type header.
 * @note A duplicate matching row is rejected rather than silently selecting a generation.
 */
inline void validateSnapshotManifest(const Request &r,const std::vector<std::string> &lines)
{
    require(!lines.empty() && lines[0]=="backend,phase,step,key,elements,exported_elements,file,scalar_type",
        "Recorded head requires the canonical typed snapshot manifest header");
    std::size_t inputs=0,logits=0;
    const std::string phase=r.phase=="decode"?"decode":"prefill";
    const auto check=[&](const std::vector<std::string>&f,const FileIdentity &file,const std::string &key,std::uint64_t elements)
    {
        require(f[0]==r.snapshot_backend && f[1]==phase && f[2]==std::to_string(r.step) && f[3]==key &&
            f[4]==std::to_string(elements) && f[5]==f[4] && f[6]==file.path && f[7]=="FP32",
            "Recorded head canonical row has different trajectory, extent, path or scalar type");
    };
    for(std::size_t i=1;i<lines.size();++i)
    {
        if(lines[i].empty())continue;
        const auto f=csvFields(lines[i]);require(f.size()==8,"Malformed canonical snapshot manifest row");
        if(f[6]==r.input.path){check(f,r.input,r.snapshot_key,r.input.bytes/4);++inputs;}
        if(f[6]==r.logits.path){check(f,r.logits,r.logits_key,r.logits.bytes/4);++logits;}
    }
    require(inputs==1 && logits==1,"Recorded head canonical input/oracle row is missing or duplicated");
}
/**
 * @brief Consume the actual root-owned focused canonical export result before admission.
 * @param r Recorded source/snapshot request, whose result file stat is separately checked.
 * @param j Original root-owned focused export result.json, not server-controller fields.
 * @note The producer wrapper owns registered-cell selection, XML/native/driver
 *       validation. This consumer checks retained original result fields; it
 *       does not invent a campaign certificate or infer head partition policy.
 */
inline void validateObservationReceipt(const Request &r,const nlohmann::json &j)
{
    require(j.at("complete")==true && j.at("passed")==true && j.at("returncode")==0 &&
        j.at("native_exit").at("exit_valid")==true && j.at("driver_window_clean")==true &&
        j.at("observation_errors").is_array() && j.at("observation_errors").empty() &&
        j.at("received_signals").is_array() && j.at("received_signals").empty() && !j.contains("error"),
        "Recorded canonical export result is not complete/passed/retired/driver-clean");
    const auto &cases=j.at("gtest_cases");
    require(cases.is_array() && cases.size()==1,"Recorded export must prove exactly one original selected GTest");
    const auto &cell=cases.at(0);
    require(cell.at("classname").get<std::string>()+"."+cell.at("name").get<std::string>()==r.observation &&
        cell.at("status")=="run" && cell.at("result")=="completed" && cell.at("failures")==0 && cell.at("skipped")==0,
        "Recorded export GTest cell is different, failed or skipped");
    require(j.at("command").is_array() && j.at("command").size()>=2 && j.at("source_models").is_array(),
        "Recorded export command/source inventory has wrong container type");
    std::size_t filters=0;
    for(const auto &arg:j.at("command"))
    {const auto word=arg.get<std::string>();if(word.starts_with("--gtest_filter="))
        {require(word=="--gtest_filter="+r.observation,"Recorded export command/cell mismatch");++filters;}}
    require(filters==1,"Recorded export original selected filter is missing or duplicated");
    const auto &env=j.at("explicit_environment");
    require(env.at("LLAMINAR_PARITY_EXPORT_SNAPSHOT_KEYS")=="FINAL_NORM,LM_HEAD" &&
        env.at("LLAMINAR_PARITY_EXPORT_SNAPSHOT_MAX_ELEMENTS")=="0",
        "Recorded export did not request complete original native head snapshots");
    const std::vector<std::string> precision_keys={"LLAMINAR_GDN_PROJECTION_ARITHMETIC","LLAMINAR_MOE_ROUTER_ARITHMETIC",
        "LLAMINAR_QFN_FA_OUTPUT_ACTIVATION","LLAMINAR_QFN_GDN_OUTPUT_ACTIVATION",
        "LLAMINAR_QFN_GDN_QKV_ACTIVATION","LLAMINAR_QFN_HC_UP_ACTIVATION"};
    for(const auto &key:precision_keys)
        require(r.precision.contains(key) && env.at(key)==r.precision.at(key),"Recorded six-selector identity differs from actual export");
    const auto same=[](const FileIdentity &a,const FileIdentity &b)
    {return a.path==b.path && a.device==b.device && a.inode==b.inode && a.bytes==b.bytes && a.mtime_ns==b.mtime_ns && a.ctime_ns==b.ctime_ns;};
    for(const auto &f:r.model_artifacts)
    {
        std::size_t found=0;
        for(const auto &source:j.at("source_models"))
            if(source.at("path")==f.path){require(same(fileIdentity(source),f),"Recorded GGUF differs from qualified source stat");++found;}
        require(found==1,"Recorded export source artifact is missing or duplicated");
    }
    const auto &manifests=j.at("snapshot_manifests");
    require(manifests.is_array() && manifests.size()==1 && same(fileIdentity(manifests.at(0)),r.snapshot_manifest),
        "Recorded export manifest differs from qualified original manifest");
}
/** @brief A custom CLI branch consumes the explicit request file, never outer environment state. */
bool requested(int argc,char **argv) noexcept;
/** @brief Run the qualified local-shape diagnostic through the existing performance executable. */
int run(int argc,char **argv);
} // namespace llaminar2::test::recorded_head
