/**
 * @file ROCmCTAOwnedHCRead.h
 * @brief Ordinary CTA-owned HC completion over original prepared sources and scratch.
 *
 * Norm and injection retain their original graph nodes. The original SingleQ8
 * producer and deferred down producer precede one ordinary completion kernel.
 * Each CTA owns a wide mixed-output tile and reconstructs its small immutable
 * down row locally; there is no inter-CTA protocol or additional allocation.
 */
#pragma once
#include "ROCmPersistentHCRead.h"
#include "ROCmNativeQuantisedSerialGeometry.h"
namespace llaminar2::rocm {
/** @brief Pointer-free eligibility, including the unauthored-P1-partials alias boundary.
 * @param hidden Original branch width. @param partitions Actual original producer partitions.
 * @param aliased Exact raw/activated alias. @throws std::invalid_argument For unsafe P1 aliasing.
 */
inline void requireCTAOwnedHCTailGeometry(int hidden,int partitions,bool aliased) {
 if(hidden<=0||hidden%8||partitions<=0)
  throw std::invalid_argument("CTA-owned HC requires complete original K32 input and positive original partitions");
 if(partitions==1&&aliased&&hidden>48)
  throw std::invalid_argument("CTA-owned HC P1 raw/activated alias requires one output CTA; P1 partials are unauthored");
}
/** @brief Cold exact-function capability; stage ownership authenticates its physical binding. */
class CTAOwnedHCReadProgram {
public:
 /** @return Exact cold native resources, including actual static LDS and private bytes. */
 const NativeQ8SerialKernelResources& resources() const {return resources_;}
 /** @return Existing device/workspace program generation. */
 std::uint64_t generation() const {return generation_;}
 /** @return Exact original participant. */
 DeviceId device() const {return device_;}
 /** @brief Authenticate exact immutable program geometry without a runtime query or post-preparation signature drift.
  * @param device Original participant. @param args Frozen original binding. @param generation Original workspace identity.
  */
 void requireBinding(DeviceId device,const PersistentHCReadArgs& args,std::uint64_t generation) const {
  if(device_!=device||!generation||generation_!=generation||hidden_!=args.norm.hidden||rank_!=args.down_weights.n||
     partitions_!=args.partitions||!resources_.function||resources_.block_threads!=1024)
   throw std::logic_error("CTA-owned HC exact function/group/physical geometry changed");
 }
private:
 friend CTAOwnedHCReadProgram prepareCTAOwnedHCRead(DeviceId,PersistentHCReadArgs&,std::uint64_t);
 friend void launchCTAOwnedHCRead(const CTAOwnedHCReadProgram&,const PersistentHCReadArgs&,void*,std::uint64_t);
 NativeQ8SerialKernelResources resources_{};
 DeviceId device_;std::uint64_t generation_=0;int hidden_=0,rank_=0,partitions_=0;
};
/** @brief Admit the exact ordinary1024-thread completion cold; no cooperative launch/queue or new buffer.
 * @param device Original participant. @param args Original prepared sources/banks; receives actual partitions.
 * @param generation Original retained workspace generation. @return Sealed actual function/resources.
 */
CTAOwnedHCReadProgram prepareCTAOwnedHCRead(DeviceId device,PersistentHCReadArgs& args,std::uint64_t generation);
/** @brief Submit original SingleQ8, original deferred down producer and ordinary CTA-owned completion.
 * @param program Cold exact function. @param args Original frozen sources and physical owners.
 * @param stream Exact prepared non-default stream. @param generation Original workspace generation.
 */
void launchCTAOwnedHCRead(const CTAOwnedHCReadProgram& program,const PersistentHCReadArgs& args,void* stream,std::uint64_t generation);
}
