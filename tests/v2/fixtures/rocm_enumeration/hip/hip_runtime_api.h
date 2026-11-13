/**
 * @file hip_runtime_api.h
 * @brief Exact native entrypoint declarations for the hardware-free enumeration replay.
 *
 * Only the calls made by the unchanged production body are doubled. No HIP
 * runtime, context, allocation or backend factory is linked. A separate actual
 * SDK signature compilation authenticates this deliberately small interface.
 */
#pragma once
#include <cstddef>
typedef int hipDevice_t;
enum hipError_t {hipSuccess=0,hipErrorInvalidValue=1,hipErrorInvalidDevice=101,hipErrorUnknown=999};
extern "C" {
hipError_t hipGetDevice(int* device);
hipError_t hipSetDevice(int device);
hipError_t hipMemGetInfo(std::size_t* free_bytes,std::size_t* total_bytes);
hipError_t hipDeviceGet(hipDevice_t* device,int ordinal);
hipError_t hipDevicePrimaryCtxGetState(hipDevice_t device,unsigned int* flags,int* active);
hipError_t hipDevicePrimaryCtxRelease(hipDevice_t device);
hipError_t hipDeviceCanAccessPeer(int* can_access,int source,int destination);
const char* hipGetErrorString(hipError_t error);
}
