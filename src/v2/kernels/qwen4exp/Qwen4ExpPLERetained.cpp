/**
 * @file Qwen4ExpPLERetained.cpp
 * @brief Checked CPU live retention and immutable accepted-history splicing.
 *
 * Validation precedes every write. In particular, later invalid request decisions
 * cannot partially publish an earlier request. Native payloads are copied as bits;
 * no float arithmetic, precision conversion or history-sized temporary is needed.
 */
#include "Qwen4ExpPLERetained.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace llaminar2::qwen4exp {
namespace {
/** @brief Reject invalid immutable setup before any pointer arithmetic. */
void require(bool value){if(!value)throw std::invalid_argument("Invalid PLE retained transaction geometry/alias");}
/** @brief One checked native interval used only for setup alias validation. */
struct Span{std::uintptr_t begin,end;};
/** @brief Form an aligned overflow-checked nonempty native interval. */
Span span(const void *p,std::size_t bytes,std::size_t alignment=4){
 const auto b=reinterpret_cast<std::uintptr_t>(p);
 require(p&&bytes&&b%alignment==0&&bytes<=std::numeric_limits<std::uintptr_t>::max()-b);return{b,b+bytes};}
/** @brief Reject every overlap between distinct live owners. */
void distinct(Span a,Span b){require(a.end<=b.begin||b.end<=a.begin);}
/** @brief Validate request geometry without inspecting execution-owned words. */
void geometry(DeviceRequestRowRanges rows,int r,int c){require(r>=0&&r<rows.requests()&&c>0&&c<=(std::numeric_limits<int>::max()-2)/9&&std::size_t(c)<=std::numeric_limits<std::size_t>::max()/36&&std::size_t(rows.rowsPerRequest())<=std::numeric_limits<std::size_t>::max()/4/std::size_t(c));}
/** @brief Resolve a CPU-owned live count and reject corruption before writes. */
int live(DeviceRequestRowRanges rows,int r){const int n=rows.lengthOwner()?rows.lengthOwner()[r]:rows.rowsPerRequest();if(rows.activeRowsFor(r,n)<0)throw std::invalid_argument("Invalid PLE transaction live count");return n;}
}
void validatePLESnapshot(const PLESnapshotPrior &p){
 geometry(p.sourceRows,p.requestIndex,p.channels);
 const Span reads[]{span(p.acceptedTokens,8),span(p.acceptedConvolution,std::size_t(p.channels)*36)};
 const Span writes[]{span(p.priorTokens,8),span(p.priorConvolution,std::size_t(p.channels)*36),span(p.transactionLive,std::size_t(p.sourceRows.requests())*4),span(p.copiedBytes,8,8)};
 for(auto a:writes)for(auto b:reads)distinct(a,b);
 for(std::size_t a=0;a<std::size(reads);++a)for(std::size_t b=a+1;b<std::size(reads);++b)distinct(reads[a],reads[b]);
 for(int a=0;a<4;++a)for(int b=a+1;b<4;++b)distinct(writes[a],writes[b]);
 if(p.sourceRows.lengthOwner())for(auto a:writes)distinct(a,span(p.sourceRows.lengthOwner(),std::size_t(p.sourceRows.requests())*4));
}
void validatePLERetention(const PLERetainCandidates &p){
 geometry(p.transactionRows,p.requestIndex,p.channels);require(p.transactionRows.lengthOwner());
 const auto n=std::size_t(p.transactionRows.rowsPerRequest());
 const Span reads[]{span(p.tokens,n*4),span(p.normalized,n*std::size_t(p.channels)*4),span(p.transactionRows.lengthOwner(),std::size_t(p.transactionRows.requests())*4)};
 const Span writes[]{span(p.retainedTokens,n*4),span(p.retainedNormalized,n*std::size_t(p.channels)*4),span(p.copiedBytes,8,8)};
 for(auto a:writes)for(auto b:reads)distinct(a,b);
 for(std::size_t a=0;a<std::size(reads);++a)for(std::size_t b=a+1;b<std::size(reads);++b)distinct(reads[a],reads[b]);
 for(int a=0;a<3;++a)for(int b=a+1;b<3;++b)distinct(writes[a],writes[b]);
}
void validatePLEPublication(const PLEPublishAccepted &p){
 const auto rows=p.acceptance.candidates();geometry(rows,p.requestIndex,p.channels);require(rows.lengthOwner());
 const auto n=std::size_t(rows.rowsPerRequest());
 const Span reads[]{span(p.priorTokens,8),span(p.priorConvolution,std::size_t(p.channels)*36),span(p.retainedTokens,n*4),span(p.retainedNormalized,n*std::size_t(p.channels)*4),span(rows.lengthOwner(),std::size_t(rows.requests())*4)};
 const Span writes[]{span(p.acceptedTokens,8),span(p.acceptedConvolution,std::size_t(p.channels)*36)};
 for(auto a:writes)for(auto b:reads)distinct(a,b);
 for(std::size_t a=0;a<std::size(reads);++a)for(std::size_t b=a+1;b<std::size(reads);++b)distinct(reads[a],reads[b]);distinct(writes[0],writes[1]);
 if(p.acceptance.decisionWords())for(auto a:writes)distinct(a,span(p.acceptance.decisionWords(),(std::size_t(rows.requests()-1)*p.acceptance.decisionWordStride()+1)*4));
 if(p.acceptance.origins().words())for(auto a:writes)distinct(a,span(p.acceptance.origins().words(),(std::size_t(rows.requests()-1)*p.acceptance.origins().wordStride()+1)*4));
}
bool validPLEDecisions(const AcceptedRequestPrefixes &p) noexcept {for(int r=0;r<p.candidates().requests();++r)if(p.resolveNative(r)<0)return false;return true;}
void cpuPLESnapshotPrior(const PLESnapshotPrior &p){
 validatePLESnapshot(p);const int n=live(p.sourceRows,p.requestIndex);
 if(n){std::memcpy(p.priorTokens,p.acceptedTokens,8);std::memcpy(p.priorConvolution,p.acceptedConvolution,std::size_t(p.channels)*36);}
 p.transactionLive[p.requestIndex]=n;*p.copiedBytes=n?8+std::uint64_t(p.channels)*36:0;
}
void cpuPLERetainCandidates(const PLERetainCandidates &p){
 validatePLERetention(p);const int n=live(p.transactionRows,p.requestIndex);
 if(n){std::memcpy(p.retainedTokens,p.tokens,std::size_t(n)*4);std::memcpy(p.retainedNormalized,p.normalized,std::size_t(n)*p.channels*4);}
 *p.copiedBytes=std::uint64_t(n)*(std::uint64_t(p.channels)+1)*4;
}
void cpuPLEPublishAccepted(const PLEPublishAccepted &p){
 validatePLEPublication(p);if(!validPLEDecisions(p.acceptance))throw std::invalid_argument("Invalid PLE acceptance vector before publication");
 if(!live(p.acceptance.candidates(),p.requestIndex))return;
 const int accepted=p.acceptance.resolveNative(p.requestIndex);
 for(int row=0;row<2;++row){const int source=accepted-2+row;p.acceptedTokens[row]=source<0?p.priorTokens[source+2]:p.retainedTokens[source];}
 for(int row=0;row<9;++row){const int source=accepted-9+row;const auto *data=source<0?p.priorConvolution+std::size_t(source+9)*p.channels:p.retainedNormalized+std::size_t(source)*p.channels;std::memcpy(p.acceptedConvolution+std::size_t(row)*p.channels,data,std::size_t(p.channels)*4);}
}
} // namespace llaminar2::qwen4exp
