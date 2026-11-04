/**
 * @file Test__CPUExpertSocketMembership.cpp
 * @brief Device-free replay of the actual native distinct-socket membership predicate.
 *
 * Recorded 1–8 endpoint geometry proves launch validation only, not unavailable
 * socket hardware. Repeated/oversubscribed physical sockets must fail before
 * an admitted native domain is created.
 */
#include "../../fixtures/collective/CPUExpertDomainSocketMembership.h"
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
using namespace llaminar2::test;
TEST(CPUExpertSocketMembership, CompleteOneThroughEightRetainsOriginalRanks) {
 for(int count=1;count<=8;++count){
  std::vector<CPUExpertSocketEndpoint> input;
  for(int rank=0;rank<count;++rank)input.push_back({rank,rank,rank});
  EXPECT_TRUE(validCPUExpertSocketMembership(input,count));
  std::reverse(input.begin(),input.end());EXPECT_TRUE(validCPUExpertSocketMembership(input,count));
 }
}
TEST(CPUExpertSocketMembership, OversubscribedOrIncompleteNeverQualifies) {
 std::vector<CPUExpertSocketEndpoint> input{{0,0,0},{1,1,1}};
 EXPECT_TRUE(validCPUExpertSocketMembership(input,2));
 EXPECT_FALSE(validCPUExpertSocketMembership(input,3));
 EXPECT_FALSE(validCPUExpertSocketMembership({},0));EXPECT_FALSE(validCPUExpertSocketMembership(input,9));
 for(int count:{3,4,8}){
  input.clear();for(int rank=0;rank<count;++rank)input.push_back({rank,rank%2,rank%2});
  EXPECT_FALSE(validCPUExpertSocketMembership(input,count));
 }
 input={{0,0,0},{1,1,1}};
 input[1].socket=0;EXPECT_FALSE(validCPUExpertSocketMembership(input,2));
 input[1]={0,1,1};EXPECT_FALSE(validCPUExpertSocketMembership(input,2));
 input[1]={1,0,1};EXPECT_FALSE(validCPUExpertSocketMembership(input,2));
 input[1]={1,1,-1};EXPECT_FALSE(validCPUExpertSocketMembership(input,2));
}
