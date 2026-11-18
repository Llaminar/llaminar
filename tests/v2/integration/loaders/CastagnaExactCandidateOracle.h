/**
 * @file CastagnaExactCandidateOracle.h
 * @brief Bounded test-only old-B/new-B terminal byte oracle, never device state.
 *
 * WriteBaseline creates one new immutable file only after all observations pass.
 * CompareBaseline opens read-only and cannot create, replace or repair a file.
 * Exact source/input metadata precedes20 explicit joined output/route records.
 */
#pragma once
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <cerrno>
#include <sstream>
#include <type_traits>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
namespace llaminar2::test {
/** @brief Explicit diagnostic action; never a production or automatic mode. */
enum class CastagnaOracleAction {Disabled,WriteBaseline,CompareBaseline};
/** @brief Immutable path/action selected before fixture preparation. */
struct CastagnaOracleConfig {CastagnaOracleAction action=CastagnaOracleAction::Disabled;std::filesystem::path path;};
/** @brief Append exact typed fields, bounded before allocation or input parsing. */
class CastagnaOracleBytes {
public:
    static constexpr std::size_t limit=8*1024*1024;
    /** @brief Append bytes without encoding, padding or implicit type conversion. */
    void append(std::span<const std::byte> bytes) {
        if(bytes.size()>limit-data.size())throw std::invalid_argument("Castagna oracle exceeds8MiB");
        data.insert(data.end(),bytes.begin(),bytes.end());
    }
    /** @brief Append one fixed-width native field; header requires little endian. */
    template<class T> void scalar(const T& value) {static_assert(std::is_trivially_copyable_v<T>);append(std::as_bytes(std::span(&value,1)));}
    /** @brief Prefix a bounded string with its exact byte extent. */
    void text(std::string_view value) {if(value.size()>4096)throw std::invalid_argument("Castagna oracle string too large");scalar(std::uint32_t(value.size()));append(std::as_bytes(std::span(value.data(),value.size())));}
    /** @brief Add a named raw field for precise first-mismatch diagnostics. */
    void field(std::string name,std::span<const std::byte> value) {
        scalar(std::uint32_t(value.size()));fields.push_back({std::move(name),data.size(),value.size()});append(value);
    }
    struct Field {std::string name;std::size_t offset,size;};
    std::vector<std::byte> data;std::vector<Field> fields;
};
/** @brief Exact declared metadata/source identity, independent of candidate implementation. */
inline void castagnaOracleFileIdentity(CastagnaOracleBytes& out,const std::string& path) {
    struct stat value{};
    if(::stat(path.c_str(),&value)||!S_ISREG(value.st_mode))throw std::invalid_argument("Castagna oracle source missing: "+path);
    out.text(std::filesystem::canonical(path).string());out.scalar(std::uint64_t(value.st_dev));out.scalar(std::uint64_t(value.st_ino));
    out.scalar(std::uint64_t(value.st_size));out.scalar(std::int64_t(value.st_mtim.tv_sec));out.scalar(std::int64_t(value.st_mtim.tv_nsec));
}
/** @brief Write/compare exact record sequence with no candidate-side replacement path. */
class CastagnaExactCandidateOracle {
public:
    /** @brief Bind expected metadata; compare mode validates complete bounded file before records. */
    CastagnaExactCandidateOracle(CastagnaOracleConfig config,const CastagnaOracleBytes& metadata):config_(std::move(config)) {
        static_assert(std::endian::native==std::endian::little&&sizeof(float)==4&&sizeof(std::int32_t)==4);
        if(config_.action==CastagnaOracleAction::Disabled)return;
        if(config_.path.empty()||(config_.action!=CastagnaOracleAction::WriteBaseline&&config_.action!=CastagnaOracleAction::CompareBaseline))
            throw std::invalid_argument("Castagna oracle requires explicit valid action/path");
        bytes_.text("CastagnaExactCandidateOracle");bytes_.scalar(std::uint32_t(1));bytes_.scalar(std::uint32_t(0x01020304));
        bytes_.scalar(std::uint32_t(sizeof(float)));bytes_.scalar(std::uint32_t(20));bytes_.scalar(std::uint32_t(metadata.data.size()));bytes_.append(metadata.data);
        if(config_.action==CastagnaOracleAction::CompareBaseline) {
            std::ifstream in(config_.path,std::ios::binary|std::ios::ate);
            if(!in)throw std::invalid_argument("Castagna oracle baseline missing");
            const auto size=in.tellg();if(size<0||std::uint64_t(size)>CastagnaOracleBytes::limit)throw std::invalid_argument("Castagna oracle baseline size invalid");
            expected_.resize(std::size_t(size));in.seekg(0);in.read(reinterpret_cast<char*>(expected_.data()),expected_.size());
            if(!in||expected_.size()<bytes_.data.size()||std::memcmp(expected_.data(),bytes_.data.data(),bytes_.data.size()))
                throw std::invalid_argument("Castagna oracle source/geometry/policy metadata mismatch");
        } else if(config_.action==CastagnaOracleAction::WriteBaseline&&std::filesystem::exists(config_.path))
            throw std::invalid_argument("Castagna oracle baseline already exists");
    }
    /** @brief Observe the next exact input/output record after its existing GPU completion. */
    void observe(unsigned history,const CastagnaOracleBytes& record) {
        if(config_.action==CastagnaOracleAction::Disabled)return;
        if(history!=records_||history>=20)throw std::invalid_argument("Castagna oracle history sequence invalid");
        const auto begin=bytes_.data.size();bytes_.scalar(std::uint32_t(history));bytes_.scalar(std::uint32_t(record.data.size()));bytes_.append(record.data);
        if(config_.action==CastagnaOracleAction::CompareBaseline) {
            if(expected_.size()<bytes_.data.size())throw std::invalid_argument("Castagna oracle truncated history "+std::to_string(history));
            for(std::size_t i=begin;i<bytes_.data.size();++i)if(bytes_.data[i]!=expected_[i]) {
                std::ostringstream message;message<<"Castagna oracle mismatch history="<<history<<" record_byte="<<(i-begin);
                const auto offset=i-begin;
                for(const auto& field:record.fields)if(offset>=8+field.offset&&offset<8+field.offset+field.size) {
                    const auto index=(offset-8-field.offset)/4;std::uint32_t old_bits=0,new_bits=0;
                    const auto at=begin+8+field.offset+index*4;
                    if(index*4+4<=field.size) {std::memcpy(&old_bits,expected_.data()+at,4);std::memcpy(&new_bits,bytes_.data.data()+at,4);}
                    message<<" field="<<field.name<<" index="<<index<<" old_bits=0x"<<std::hex<<old_bits<<" new_bits=0x"<<new_bits;break;
                }
                throw std::invalid_argument(message.str());
            }
        }
        ++records_;
    }
    /** @brief Require totality; successful writer uses O_EXCL and never overwrites. */
    void finish() {
        if(config_.action==CastagnaOracleAction::Disabled)return;
        if(records_!=20)throw std::invalid_argument("Castagna oracle incomplete histories");
        if(config_.action==CastagnaOracleAction::CompareBaseline) {
            if(expected_.size()!=bytes_.data.size())throw std::invalid_argument("Castagna oracle trailing bytes");
            return;
        }
        const int fd=::open(config_.path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0444);
        if(fd<0)throw std::runtime_error("Castagna oracle exclusive create failed");
        std::size_t offset=0;bool okay=true;
        while(offset<bytes_.data.size()) {const auto n=::write(fd,bytes_.data.data()+offset,bytes_.data.size()-offset);if(n<0&&errno==EINTR)continue;if(n<=0){okay=false;break;}offset+=std::size_t(n);}
        if(::close(fd))okay=false;
        if(!okay)throw std::runtime_error("Castagna oracle incomplete exclusive write; retained file is invalid");
    }
private:
    CastagnaOracleConfig config_;CastagnaOracleBytes bytes_;std::vector<std::byte> expected_;unsigned records_=0;
};
}
