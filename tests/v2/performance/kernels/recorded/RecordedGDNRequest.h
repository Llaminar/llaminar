/**
 * @file RecordedGDNRequest.h
 * @brief Closed original-input admission for the complete native GDN projection bundle.
 *
 * The four production siblings remain one operation. QKV and all value-associated
 * matrices use the production modulo-linked head authority; contiguous N/degree
 * slices are not an equivalent source. Native manifest/receipt validation is
 * shared with the existing recorded-head diagnostic and opens no payload.
 */
#pragma once
#include "RecordedObservation.h"
#include "../../../../../src/v2/config/GDNHeadAssignment.h"
#include "../../../../../src/v2/kernels/common/DeviceNativeVNNIMatrixDesc.h"
#include <array>

namespace llaminar2::test::recorded_gdn
{
using recorded_observation::FileIdentity;
using recorded_observation::require;
using recorded_observation::integer;
using recorded_observation::unsignedInteger;
using recorded_observation::fileIdentity;
/**
 * @brief Describe the actual admitted native prepared source with semantic JSON types.
 * @param descriptor Original engine descriptor already admitted by the native caller.
 * @return Numeric dimensions/codebooks and a boolean original block-family flag.
 * @note The device ABI stores flags in uint8_t. Publishing that byte directly
 *       would emit a JSON number instead of the source-family boolean contract.
 */
inline nlohmann::json nativePreparedSourceMetadata(const DeviceNativeVNNIMatrixDesc &descriptor)
{
    return {{"n",descriptor.n},{"k",descriptor.k},{"codebook",descriptor.codebook_id},
        {"source_codebook",descriptor.source_codebook_id},{"source_is_superblock",descriptor.source_is_superblock!=0}};
}
/** @brief Original sibling order is the actual GDNProjectionStage output order. */
enum class Projection : std::size_t { QKV, Z, Alpha, Beta };
/** @brief A projection source and complete native output oracle; no prepared payload is mirrored. */
struct Source
{
    std::string name,key;
    int original_n=0,gguf_type=0;
    FileIdentity oracle;
};
/** @brief Explicit native observation, original geometry and one diagnostic participant. */
struct Request
{
    std::string backend,output,observation,phase,snapshot_backend;
    int ordinal=0,degree=0,participant=0,rows=0,k=0,key_heads=0,value_heads=0,state=0;
    int step=0,warmup=0,samples=0;
    std::uint64_t generation=0,model_context=0;
    FileIdentity input,snapshot_manifest,observation_receipt;
    std::vector<FileIdentity> model_artifacts;
    std::array<Source,4> sources;
    nlohmann::json precision,document;
};
/** @brief Exact original key closure, shared by receipt validation and native producer selection. */
inline constexpr const char *snapshotKeys=
    "layer0_ATTENTION_NORM,layer0_QKV_PROJECTION,layer0_GDN_Z_PROJECTION,layer0_GDN_ALPHA,layer0_GDN_BETA";
/**
 * @brief Parse a complete ordinary layer-zero GDN bundle before any runtime initialization.
 * @param j Explicit closed operation request, never a serving configuration.
 * @return Exact qualified source work and complete physical snapshot row extent.
 * @throws std::invalid_argument For unsupported arithmetic, incomplete sources or geometry.
 * @note This first adapter supports ordinary SingleQ8 QKV and original FP16 prepared
 *       alpha/beta. Other policies need their own complete source/row contracts.
 */
inline Request parse(const nlohmann::json &j)
{
    require(j.at("schema")=="llaminar.recorded_gdn_projection.v1" && j.at("scope")=="local_shape" &&
        j.at("operation_boundary")=="gdn_projection_bundle_compute" &&
        j.at("input_origin")=="llaminar_captured_stage_output" &&
        j.at("producer_stage")=="layer0_attn_norm" && j.at("semantic_tensor")=="layer0_ATTENTION_NORM" &&
        j.at("dtype")=="float32_le" && j.at("mtp_enabled")==false && j.at("mtp_graph_capacity")==0,
        "Recorded GDN requires its complete ordinary native layer-zero projection bundle");
    Request r;r.document=j;
    r.backend=j.at("backend").get<std::string>();r.ordinal=integer(j.at("device_ordinal"));
    r.output=j.at("output").get<std::string>();r.observation=j.at("observation_id").get<std::string>();
    r.phase=j.at("phase").get<std::string>();r.snapshot_backend=j.at("snapshot_backend").get<std::string>();
    r.degree=integer(j.at("slice_degree"));r.participant=integer(j.at("slice_participant"));
    r.rows=integer(j.at("observation_physical_rows"));r.k=integer(j.at("k"));
    r.key_heads=integer(j.at("gdn_key_heads"));r.value_heads=integer(j.at("gdn_value_heads"));r.state=integer(j.at("gdn_state"));
    r.step=integer(j.at("step"));r.warmup=integer(j.at("warmup"));r.samples=integer(j.at("samples"));
    r.generation=unsignedInteger(j.at("generation"));r.model_context=unsignedInteger(j.at("model_context"));
    r.input=fileIdentity(j.at("input"));r.snapshot_manifest=fileIdentity(j.at("snapshot_manifest"));
    r.observation_receipt=fileIdentity(j.at("observation_receipt"));r.precision=j.at("six_precision_selectors");
    require((r.backend=="cuda" || r.backend=="rocm") && r.ordinal>=0 &&
        !r.output.empty() && r.output.front()=='/' && !r.observation.empty() && !r.snapshot_backend.empty() &&
        r.generation && r.model_context && (r.phase=="decode" || r.phase=="prefill") &&
        r.rows>0 && r.k>0 && r.k%32==0 && r.key_heads>0 && r.value_heads>0 && r.state>0 &&
        (r.degree==1 || r.degree==2 || r.degree==4) && r.participant>=0 && r.participant<r.degree &&
        (r.phase=="decode"?(r.rows==1 && r.step>=0):r.step==-1) &&
        r.warmup>=1 && r.warmup<=1024 && r.samples>=31 && r.samples<=4096,
        "Recorded GDN geometry, trajectory or sampling is incomplete");
    require(std::uint64_t(r.rows)<=std::numeric_limits<std::uint64_t>::max()/4/std::uint64_t(r.k) &&
        r.input.bytes==std::uint64_t(r.rows)*std::uint64_t(r.k)*4,
        "Recorded GDN input must cover every original physical FP32 row");
    // Validate positive products before the production authority's int arithmetic.
    require(std::uint64_t(r.state)*(2*std::uint64_t(r.key_heads)+std::uint64_t(r.value_heads))<=
        std::uint64_t(std::numeric_limits<int>::max()),"Recorded GDN original output geometry overflows");
    (void)GDNHeadAssignment::forEqualRank(r.key_heads,r.value_heads,r.participant,r.degree);
    const std::array<std::string,4> names={"blk.0.attn_qkv.weight","blk.0.attn_gate.weight","blk.0.ssm_alpha.weight","blk.0.ssm_beta.weight"};
    const std::array<std::string,4> keys={"layer0_QKV_PROJECTION","layer0_GDN_Z_PROJECTION","layer0_GDN_ALPHA","layer0_GDN_BETA"};
    const std::array<int,4> columns={(2*r.key_heads+r.value_heads)*r.state,r.value_heads*r.state,r.value_heads,r.value_heads};
    require(j.at("sources").is_array() && j.at("sources").size()==4,"Recorded GDN requires all four ordered original sources");
    for(std::size_t i=0;i<4;++i)
    {
        const auto &v=j.at("sources").at(i);auto &s=r.sources[i];
        s.name=v.at("source_tensor").get<std::string>();s.key=v.at("snapshot_key").get<std::string>();
        s.original_n=integer(v.at("original_n"));s.gguf_type=integer(v.at("source_gguf_type"));
        s.oracle=fileIdentity(v.at("oracle"));
        require(s.name==names[i] && s.key==keys[i] && s.original_n==columns[i] &&
            s.gguf_type==(i<2?8:0) && v.at("source_format")== (i<2?"Q8_0":"F32") &&
            v.at("prepared_format")== (i<2?"NativeVNNI_Q8_0":"FP16") &&
            (i<2?v.at("source_execution_codebook")==19:v.at("source_execution_codebook").is_null()),
            "Recorded GDN sibling order, source geometry, format or arithmetic differs");
        require(std::uint64_t(r.rows)<=std::numeric_limits<std::uint64_t>::max()/4/std::uint64_t(s.original_n) &&
            s.oracle.bytes==std::uint64_t(r.rows)*std::uint64_t(s.original_n)*4,
            "Recorded GDN output oracle must cover every original FP32 row");
    }
    require(r.precision.is_object() && r.precision.size()==6 &&
        r.precision.at("LLAMINAR_GDN_PROJECTION_ARITHMETIC")=="fp16_moe_alpha_beta" &&
        r.precision.at("LLAMINAR_QFN_GDN_QKV_ACTIVATION")=="single_q8",
        "Recorded GDN requires the original native QKV and FP16 prepared control policy");
    const std::array<const char *,6> selectors={"LLAMINAR_GDN_PROJECTION_ARITHMETIC","LLAMINAR_MOE_ROUTER_ARITHMETIC",
        "LLAMINAR_QFN_FA_OUTPUT_ACTIVATION","LLAMINAR_QFN_GDN_OUTPUT_ACTIVATION","LLAMINAR_QFN_GDN_QKV_ACTIVATION","LLAMINAR_QFN_HC_UP_ACTIVATION"};
    for(const char *key:selectors)require(r.precision.at(key).is_string() && !r.precision.at(key).get<std::string>().empty(),
        "Recorded GDN six-selector identity is incomplete");
    std::vector<std::string> paths;
    for(const auto &v:j.at("model_artifacts"))
    {
        auto f=fileIdentity(v);require(std::find(paths.begin(),paths.end(),f.path)==paths.end(),"Duplicate recorded GDN source artifact");
        paths.push_back(f.path);r.model_artifacts.push_back(std::move(f));
    }
    require(!r.model_artifacts.empty(),"Recorded GDN source artifact closure is empty");
    return r;
}
/**
 * @brief Project original output oracle intervals through the same production head authority.
 * @param r Already validated original geometry and equal participant membership.
 * @return Source row spans in actual participant-local packed order for every sibling.
 * @note These are source coordinates, not allocations or a second sharding policy.
 */
inline std::array<std::vector<GDNHeadSpan>,4> sourceSpans(const Request &r)
{
    const auto ownership=GDNHeadAssignment::forEqualRank(r.key_heads,r.value_heads,r.participant,r.degree);
    const auto key=ownership.keyElementSpan(r.state);
    const int key_rows=r.key_heads*r.state;
    std::array<std::vector<GDNHeadSpan>,4> spans;
    spans[0]={key,{key_rows+key.start,key.count}};
    for(auto value:ownership.valueElementSpans(r.state))
    {value.start+=2*key_rows;spans[0].push_back(value);}
    spans[1]=ownership.valueElementSpans(r.state);
    spans[2]=ownership.valueElementSpans(1);spans[3]=spans[2];
    return spans;
}
/** @brief Validate complete original input and all four sibling oracle manifest rows. */
inline void validateSnapshotManifest(const Request &r,const std::vector<std::string> &lines)
{
    std::vector<recorded_observation::SnapshotBinding> snapshots{{"layer0_ATTENTION_NORM",r.input,r.input.bytes/4}};
    for(const auto &s:r.sources)snapshots.push_back({s.key,s.oracle,s.oracle.bytes/4});
    recorded_observation::validateSnapshotBindings(r.snapshot_backend,r.phase,r.step,snapshots,lines);
}
/** @brief Bind the unchanged root-owned original export receipt and exact snapshot closure. */
inline void validateObservationReceipt(const Request &r,const nlohmann::json &j)
{recorded_observation::validateObservationReceipt(r,j,snapshotKeys);}
/** @brief One explicit concurrent cohort; member inputs and original sources cannot diverge. */
class CohortRequest final
{
public:
    /** @brief Validate complete, distinct physical membership before any backend work. */
    static CohortRequest fromJson(const nlohmann::json &j)
    {
        require(j.is_object() && j.size()==5 && j.at("schema")=="llaminar.recorded_gdn_cohort.v1" &&
            j.at("scope")=="actual_tp" && j.at("operation_boundary")=="gdn_projection_bundle_compute",
            "Recorded GDN requires the exact complete-cohort schema");
        auto source=parse(j.at("member_template"));require(source.participant==0,"Recorded GDN cohort must start at logical rank zero");
        const auto &devices=j.at("device_ordinals");require(devices.is_array() && devices.size()==std::size_t(source.degree),
            "Recorded GDN cohort membership differs from original slice degree");
        std::vector<Request> members;std::vector<int> seen;
        for(std::size_t rank=0;rank<devices.size();++rank)
        {
            const auto ordinal=integer(devices.at(rank));require(ordinal>=0 && std::find(seen.begin(),seen.end(),ordinal)==seen.end(),
                "Recorded GDN cohort has duplicate or invalid physical members");
            seen.push_back(ordinal);auto member=source.document;member["device_ordinal"]=ordinal;member["slice_participant"]=rank;
            members.push_back(parse(member));
        }
        require(members.front().ordinal==source.ordinal,"Recorded GDN template must name its first physical member");
        return CohortRequest(std::move(source),std::move(members),j);
    }
    /** @return Original unchanged input/source observation. */
    const Request &source() const noexcept{return source_;}
    /** @return Complete original logical membership. */
    const std::vector<Request> &members() const noexcept{return members_;}
    /** @return Authored external cohort document. */
    const nlohmann::json &document() const noexcept{return document_;}
private:
    /** @brief Restrict construction to total successful admission. */
    CohortRequest(Request source,std::vector<Request> members,nlohmann::json document)
        :source_(std::move(source)),members_(std::move(members)),document_(std::move(document)){}
    Request source_;std::vector<Request> members_;nlohmann::json document_;
};
/** @brief Explicit CLI dispatch in the existing prepared-projection performance executable. */
bool requested(int argc,char **argv) noexcept;
/** @brief Run only the admitted original recorded full-bundle capture diagnostic. */
int run(int argc,char **argv);
} // namespace llaminar2::test::recorded_gdn
