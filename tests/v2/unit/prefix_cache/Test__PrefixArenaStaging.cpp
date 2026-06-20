#include <gtest/gtest.h>

#include "memory/BufferArena.h"
#include "memory/BufferId.h"

using namespace llaminar2;

TEST(Test__PrefixArenaStaging, BufferIdsHaveNamesAndStringMappings)
{
    EXPECT_STREQ(bufferIdName(BufferId::PREFIX_K_STAGING), "PREFIX_K_STAGING");
    EXPECT_STREQ(bufferIdName(BufferId::PREFIX_V_STAGING), "PREFIX_V_STAGING");
    EXPECT_STREQ(bufferIdName(BufferId::PREFIX_HYBRID_STATE_STAGING), "PREFIX_HYBRID_STATE_STAGING");
    EXPECT_STREQ(bufferIdName(BufferId::PREFIX_MTP_K_STAGING), "PREFIX_MTP_K_STAGING");
    EXPECT_STREQ(bufferIdName(BufferId::PREFIX_MTP_V_STAGING), "PREFIX_MTP_V_STAGING");
    EXPECT_STREQ(bufferIdName(BufferId::PREFIX_TERMINAL_HIDDEN), "PREFIX_TERMINAL_HIDDEN");
    EXPECT_STREQ(bufferIdName(BufferId::PREFIX_TERMINAL_LOGITS), "PREFIX_TERMINAL_LOGITS");
    EXPECT_STREQ(bufferIdName(BufferId::MTP_FA_Q_RAW), "MTP_FA_Q_RAW");
    EXPECT_STREQ(bufferIdName(BufferId::MTP_FA_GATE), "MTP_FA_GATE");
    EXPECT_STREQ(bufferIdName(BufferId::ALL_POSITION_LOGITS), "ALL_POSITION_LOGITS");
    EXPECT_STREQ(bufferIdName(BufferId::ALL_POSITION_LOGITS_LOCAL), "ALL_POSITION_LOGITS_LOCAL");

    EXPECT_EQ(BufferArena::bufferNameToId("prefix_k_staging"), BufferId::PREFIX_K_STAGING);
    EXPECT_EQ(BufferArena::bufferNameToId("prefix_v_staging"), BufferId::PREFIX_V_STAGING);
    EXPECT_EQ(BufferArena::bufferNameToId("prefix_hybrid_state_staging"), BufferId::PREFIX_HYBRID_STATE_STAGING);
    EXPECT_EQ(BufferArena::bufferNameToId("prefix_mtp_k_staging"), BufferId::PREFIX_MTP_K_STAGING);
    EXPECT_EQ(BufferArena::bufferNameToId("prefix_mtp_v_staging"), BufferId::PREFIX_MTP_V_STAGING);
    EXPECT_EQ(BufferArena::bufferNameToId("prefix_terminal_hidden"), BufferId::PREFIX_TERMINAL_HIDDEN);
    EXPECT_EQ(BufferArena::bufferNameToId("prefix_terminal_logits"), BufferId::PREFIX_TERMINAL_LOGITS);
    EXPECT_EQ(BufferArena::bufferNameToId("mtp_q_raw"), BufferId::MTP_FA_Q_RAW);
    EXPECT_EQ(BufferArena::bufferNameToId("mtp_q_gate"), BufferId::MTP_FA_GATE);
    EXPECT_EQ(BufferArena::bufferNameToId("all_position_logits"), BufferId::ALL_POSITION_LOGITS);
    EXPECT_EQ(BufferArena::bufferNameToId("all_position_logits_local"), BufferId::ALL_POSITION_LOGITS_LOCAL);
}
