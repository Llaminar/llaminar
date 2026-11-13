/** @file RecordedNativeContextBirthKey.h
 * @brief Valid metadata for resource-family selector replay; contains no native observations or allowances.
 */
#pragma once
#include "backends/NativeContextOpaqueContract.h"
namespace llaminar2::test {
/** @return Complete synthetic original metadata independent from installed resource units. */
inline NativeContextBirthKey recordedNativeContextBirthKey(DeviceType backend,NativeContextBirthPopulation population){
 NativeContextBirthKey k;k.device=backend==DeviceType::CUDA?DeviceId::cuda(0):DeviceId::rocm(0);k.population=population;
 k.physical_uuid="recorded-physical-device";k.architecture="recorded-architecture";k.runtime_version="recorded-runtime";k.driver_version="recorded-driver";
 k.backend_sha256=k.runtime_sha256=k.driver_sha256=k.runtime_policy_sha256=std::string(64,'a');
 k.hsa_sha256=k.device.is_rocm()?std::string(64,'b'):std::string{};k.visibility_identity="recorded-visibility";k.pthread_defaults_identity="recorded-defaults";
 k.visible_device_count=2;k.compute_units=80;k.native_wave_width=k.device.is_cuda()?32:64;k.worker_stack_bytes=8192;k.worker_guard_bytes=4096;k.context_object_bytes=128;k.context_object_alignment=8;
 k.primary_streams=k.primary_blas_handles=k.secondary_blas_handles=1;k.dependency_events=32;return k;
}
}
