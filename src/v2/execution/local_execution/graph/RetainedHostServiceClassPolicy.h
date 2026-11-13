/** @file RetainedHostServiceClassPolicy.h
 * @brief One conservative declared CPU opaque policy, independent of sample counts and exact owned geometry.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
namespace llaminar2::detail {
/** @brief Existing explicit-stack joinable GNU pthread ABI; ordinary graph/code edits do not change it. */
enum class RetainedHostServiceCreatorVersion : std::uint32_t {MappedPthreadV1=1};
inline constexpr auto retainedHostServiceCreatorVersion=RetainedHostServiceCreatorVersion::MappedPthreadV1;
/** @brief Reject genuinely unsupported creator ABI, never an ELF/package byte difference. */
inline void requireRetainedHostServiceABI(std::string_view libcVersion,std::size_t pointerBytes){
    unsigned major=0,minor=0;std::size_t i=0;
    while(i<libcVersion.size()&&libcVersion[i]>='0'&&libcVersion[i]<='9'){if(major>1000)throw std::invalid_argument("Unsupported pthread ABI version");major=major*10+unsigned(libcVersion[i++]-'0');}
    if(i==0||i==libcVersion.size()||libcVersion[i++]!='.')throw std::invalid_argument("Unsupported pthread ABI version");
    const auto start=i;while(i<libcVersion.size()&&libcVersion[i]>='0'&&libcVersion[i]<='9'){if(minor>1000)throw std::invalid_argument("Unsupported pthread ABI version");minor=minor*10+unsigned(libcVersion[i++]-'0');}
    if(i==start||major!=2||minor<34||pointerBytes!=8)throw std::invalid_argument("Unsupported mapped GNU pthread ABI family");
}
/** @return Checked planning contribution from already-known TLS extent, never a live-byte probe/reprice. */
inline std::size_t plannedRetainedHostServiceOpaqueBytes(std::size_t tlsBytes){
    constexpr std::size_t unit=2u<<20,headroom=1u<<20;
    if(tlsBytes>std::numeric_limits<std::size_t>::max()-headroom)throw std::overflow_error("CPU declared TLS extent overflow");
    const auto need=tlsBytes+headroom;
    if(need>std::numeric_limits<std::size_t>::max()-(unit-1))throw std::overflow_error("CPU declared opaque rounding overflow");
    const auto rounded=((need+unit-1)/unit)*unit;return rounded<unit?unit:rounded;
}
}
