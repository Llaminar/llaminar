/**
 * @file Test__PooledImmutableOwnerLaunchPlan.cpp
 * @brief Device-free replay of actual shared CUDA/HIP owner launch multiplicity.
 *
 * Native graph declarations must count kernel invocations rather than stages.
 * These checks exercise the same compile-time role plan consumed by production
 * dispatch, including immutable C0 and positive geometries across reset/repeat.
 */
#include <gtest/gtest.h>
#include "kernels/qwen4exp/PooledImmutableOwnerLaunchPlan.h"
namespace llaminar2::qwen4exp::pooled_native {
TEST(PooledImmutableOwnerLaunchPlan,ActualOwnerDispatchAndEnumHolesReplay) {
    for(unsigned epoch=0;epoch<20;++epoch)
        for(const std::uint32_t capacity:{0u,1u,64u,512u,2u,0u})
            for(unsigned value=0;value<=33;++value){
                const auto action=static_cast<OwnerAction>(value);
                if(value==29 || value==30 || value==31 || value==33){
                    EXPECT_THROW(ownerKernelCount(action,capacity),std::invalid_argument);
                    continue;
                }
                unsigned expected=1;
                switch(action){
                case OwnerAction::Pin:case OwnerAction::ExecuteFree:case OwnerAction::ExecuteVictim:
                case OwnerAction::CancelFree:case OwnerAction::CancelVictim:
                    expected+=capacity!=0;break;
                case OwnerAction::ReadRows:case OwnerAction::ApplySourceDelivery:
                    expected+=1+(capacity!=0);break;
                default:break;
                }
                EXPECT_EQ(ownerKernelCount(action,capacity),expected)<<value<<" capacity="<<capacity;
            }
}
}
