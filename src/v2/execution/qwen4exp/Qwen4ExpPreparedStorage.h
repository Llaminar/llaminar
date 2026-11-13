/**
 * @file Qwen4ExpPreparedStorage.h
 * @brief Stage-owned immutable native-storage seals after ordinary preparation.
 *
 * Graph variants may be queried before arena GPU addresses exist. Declaration
 * hashes therefore stay stable; capture-only preparation seals the canonical
 * root address/device/runtime generation. Every later identity query validates
 * that seal and fails on drift. This metadata neither allocates model storage
 * nor publishes coherence, execution state or a second admission ledger.
 */
#pragma once
#include "../../tensors/TensorClasses.h"
#include "../../backends/IBackend.h"
#include "../local_execution/graph/GraphCaptureGuard.h"
#include <span>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <vector>
namespace llaminar2 {
/** @brief One immutable set of borrowed root declarations and a prepared physical seal. */
class Qwen4ExpPreparedStorage final {
public:
 /**
  * @brief Retain declarations only; GPU storage need not exist yet.
  * @param device Exact participant whose ordinary preparation owns the bytes.
  * @param tensors Existing canonical arena roots; duplicates share one seal.
  */
 Qwen4ExpPreparedStorage(DeviceId device,std::span<TensorBase*const>tensors):device_(device){
  if(!device.is_valid()||tensors.empty())throw std::invalid_argument("QFN storage seal needs explicit participant/root declarations");
  roots_.reserve(tensors.size());
  for(auto*t:tensors){if(!t||t->transferStorageOwner()!=t||!t->size_bytes())throw std::invalid_argument("QFN storage seal requires canonical nonempty arena roots");
   if(std::none_of(roots_.begin(),roots_.end(),[&](const Root&r){return r.tensor==t;}))roots_.push_back({t,t->size_bytes(),t->native_type(),nullptr});
  }
 }
 /** @brief Seal already-prepared addresses outside capture; never allocate, copy or rebind. */
 void seal(){
  if(sealed_){validate();return;}
  if(isGraphCaptureActive())throw std::logic_error("QFN native storage must be sealed before graph capture");
  validateDeclarations();IBackend*backend=nullptr;std::uint64_t generation=0;
  for(const auto&r:roots_){
   const auto*address=device_.is_gpu()?r.tensor->gpu_data_ptr():r.tensor->raw_data();
   if(!address)throw std::logic_error("QFN storage seal requires ordinary prepared native addresses");
   if(device_.is_gpu()){
    if(r.tensor->current_device()!=device_)throw std::logic_error("QFN storage seal participant differs from prepared root");
    auto*b=r.tensor->resolveBackend(device_);if(!b)throw std::logic_error("QFN storage seal has no canonical backend");
    if(!backend){backend=b;generation=b->deviceRuntimeGeneration(device_.ordinal);}
    if(b!=backend||!generation||b->deviceRuntimeGeneration(device_.ordinal)!=generation)throw std::logic_error("QFN storage seal requires one authenticated backend runtime generation");
   }
  }
  // Validation completes before committing any physical seal metadata.
  for(auto&r:roots_)r.address=device_.is_gpu()?r.tensor->gpu_data_ptr():r.tensor->raw_data();
  backend_=backend;generation_=generation;sealed_=true;
 }
 /** @brief Validate declarations before preparation, and every physical field after sealing. */
 void validateIfSealed()const{validateDeclarations();if(sealed_)validate();}
 /** @brief Reject absent seals, released/rebound roots and changed backend generations. */
 void validate()const{
  if(!sealed_)throw std::logic_error("QFN native storage has not been sealed for capture");
  validateDeclarations();
  for(const auto&r:roots_){
   if((device_.is_gpu()?r.tensor->gpu_data_ptr():r.tensor->raw_data())!=r.address)throw std::logic_error("QFN native storage address changed after capture seal");
   if(device_.is_gpu()&&(r.tensor->current_device()!=device_||r.tensor->resolveBackend(device_)!=backend_))throw std::logic_error("QFN native storage participant/backend changed after capture seal");
  }
  if(backend_&&backend_->deviceRuntimeGeneration(device_.ordinal)!=generation_)throw std::logic_error("QFN native storage runtime generation changed after capture seal");
 }
 /**
  * @return Complete immutable prepared identity, including backend/runtime and every original root.
  * @throws std::logic_error Unless ordinary preparation installed a still-valid seal.
  *
  * Components combine this projection with their own source/geometry/policy
  * identity. No mutable tensor value, device counter or payload is observed.
  */
 std::uint64_t preparedIdentity() const
 {
     validate();
     std::uint64_t h = 1469598103934665603ULL;
     const auto mix = [&](std::uint64_t word)
     {
         h ^= word;
         h *= 1099511628211ULL;
     };
     mix(1);
     mix(static_cast<std::uint64_t>(device_.type));
     mix(device_.ordinal);
     mix(reinterpret_cast<std::uintptr_t>(backend_));
     mix(generation_);
     mix(roots_.size());
     for (const auto &r : roots_)
     {
         mix(reinterpret_cast<std::uintptr_t>(r.tensor));
         mix(r.bytes);
         mix(static_cast<std::uint64_t>(r.type));
         mix(reinterpret_cast<std::uintptr_t>(r.address));
     }
     return h ? h : 1;
 }
 /** @return Whether ordinary preparation has installed a complete physical seal. */
 bool isSealed()const noexcept{return sealed_;}
private:
 /** @brief Borrowed declaration plus immutable prepared address, never tensor contents. */
 struct Root{TensorBase*tensor;std::size_t bytes;TensorType type;const void*address;};
 /** @brief Reject root ownership/type/capacity changes without touching live values. */
 void validateDeclarations()const{
  for(const auto&r:roots_)if(r.tensor->transferStorageOwner()!=r.tensor||r.tensor->size_bytes()!=r.bytes||r.tensor->native_type()!=r.type)
   throw std::logic_error("QFN arena root declaration changed after stage construction");
 }
 DeviceId device_;std::vector<Root>roots_;IBackend*backend_=nullptr;std::uint64_t generation_=0;bool sealed_=false;
};
}
