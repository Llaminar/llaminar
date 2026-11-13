/**
 * @file RecordedObservation.h
 * @brief Shared metadata-only admission for exact original native observations.
 *
 * Recorded operation tools share this CSV/stat/receipt parser. These checks
 * authenticate retained producer metadata; runtime consumers independently
 * validate the real file stats, prepared sources and captured output bytes.
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
namespace llaminar2::test::recorded_observation
{
/** @brief Stat-bound file identity; contents are never hashed by this consumer. */
struct FileIdentity
{
    std::string path;
    std::uint64_t device=0,inode=0,bytes=0;
    std::int64_t mtime_ns=0,ctime_ns=0;
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
/** @brief One complete FP32 snapshot bound by its actual semantic key and file identity. */
struct SnapshotBinding
{
    std::string key;
    FileIdentity file;
    std::uint64_t elements=0;
};
/**
 * @brief Validate the original canonical manifest for every operation input/oracle.
 * @param backend Original snapshot producer identity, not the benchmark backend name.
 * @param phase Exact original manifest phase.
 * @param step Exact original trajectory step.
 * @param snapshots Complete required operation closure, with no duplicate keys or paths.
 * @param lines Original manifest, never a derived local inventory.
 * @throws std::invalid_argument For absent, duplicate, partial or conflicting observations.
 */
inline void validateSnapshotBindings(const std::string &backend,const std::string &phase,int step,
    const std::vector<SnapshotBinding> &snapshots,const std::vector<std::string> &lines)
{
    require(!snapshots.empty() && !lines.empty() &&
        lines[0]=="backend,phase,step,key,elements,exported_elements,file,scalar_type",
        "Recorded operation requires the canonical typed snapshot manifest");
    std::vector<std::size_t> found(snapshots.size());
    for(std::size_t i=0;i<snapshots.size();++i)
    {
        const auto &s=snapshots[i];
        require(!s.key.empty() && s.elements>0 && s.elements<=std::numeric_limits<std::uint64_t>::max()/4 &&
            s.elements*4==s.file.bytes,"Recorded snapshot has incomplete FP32 extent");
        for(std::size_t j=0;j<i;++j)
            require(s.key!=snapshots[j].key && s.file.path!=snapshots[j].file.path,
                "Recorded snapshot keys and file paths must be distinct");
    }
    for(std::size_t i=1;i<lines.size();++i)
    {
        if(lines[i].empty())continue;
        const auto fields=csvFields(lines[i]);require(fields.size()==8,"Malformed canonical snapshot manifest row");
        for(std::size_t j=0;j<snapshots.size();++j)
        {
            const auto &s=snapshots[j];
            if(fields[6]!=s.file.path && !(fields[0]==backend && fields[1]==phase &&
                fields[2]==std::to_string(step) && fields[3]==s.key))continue;
            require(fields[0]==backend && fields[1]==phase && fields[2]==std::to_string(step) &&
                fields[3]==s.key && fields[4]==std::to_string(s.elements) && fields[5]==fields[4] &&
                fields[6]==s.file.path && fields[7]=="FP32",
                "Recorded canonical row differs in trajectory, key, full extent, path or type");
            ++found[j];
        }
    }
    for(const auto count:found)require(count==1,"Recorded canonical snapshot is absent or duplicated");
}
/**
 * @brief Consume the actual root-owned focused canonical export result before admission.
 * @param r Recorded source/snapshot request, whose result file stat is separately checked.
 * @param j Original root-owned focused export result.json, not server-controller fields.
 * @note The producer wrapper owns registered-cell selection, XML/native/driver
 *       validation. This consumer checks retained original result fields; it
 *       does not invent a campaign certificate or infer head partition policy.
 */
template<class RecordedRequest>
inline void validateObservationReceipt(const RecordedRequest &r,const nlohmann::json &j,
    std::string_view expected_snapshot_keys="FINAL_NORM,LM_HEAD")
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
    require(env.at("LLAMINAR_PARITY_EXPORT_SNAPSHOT_KEYS")==expected_snapshot_keys &&
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
} // namespace llaminar2::test::recorded_observation
