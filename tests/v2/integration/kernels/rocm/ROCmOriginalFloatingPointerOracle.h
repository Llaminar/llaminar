/**
 * @file ROCmOriginalFloatingPointerOracle.h
 * @brief Test-only ABI for the frozen immediate-pre102 staged GPU pipeline.
 */
#pragma once
#include "kernels/common/FloatingProjectionRows.h"
#include <stdexcept>
/**
 * @brief Capture original pointer publication and the original custom consumer.
 * @param a Host-recording immutable FP32 activation addresses.
 * @param b Host-recording immutable native weight addresses.
 * @param c Host-recording immutable FP32 output addresses.
 * @param da Persistent original device A table.
 * @param db Persistent original device B table.
 * @param dc Persistent original device C table.
 * @param m Physical rows. @param n Columns. @param k Reduction width.
 * @param count Live projections in1..8. @param type FP32=0,FP16=1,BF16=2.
 * @param stream Exact explicit HIP stream.
 * @param prefix Original prefix authority, mutually exclusive with requests.
 * @param requests Original request-major live-hole authority.
 * @return Whether both original asynchronous submissions succeeded.
 * @throws std::invalid_argument For incomplete capture metadata.
 */
bool qfnPointer102OriginalStaged(const float *const *a,const void *const *b,float *const *c,
 const float **da,const float **db,float **dc,int m,int n,int k,int count,int type,
 void *stream,const llaminar2::DeviceRowRange *prefix=nullptr,
 const llaminar2::DeviceRequestRowRanges *requests=nullptr);
