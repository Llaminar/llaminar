/**
 * @file Test__MPIContextP2P.cpp
 * @brief Integration tests for MPIContext point-to-point operations
 *
 * Tests MPI send/recv, isend/irecv, wait, waitAll, probe, and iprobe
 * operations required for pipeline parallelism.
 *
 * Run with: mpirun -np 2 ./v2_integration_mpi_context_p2p
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

#include "utils/MPIContext.h"
#include "utils/MPITags.h"
#include "utils/Logger.h"

using namespace llaminar2;

class MPIContextP2P : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_);
    }

    void TearDown() override
    {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    int rank_;
    int world_size_;
    std::shared_ptr<MPIContext> mpi_ctx_;
};

// ============================================================================
// Test 1: Basic Send/Recv with Float Array
// ============================================================================

TEST_F(MPIContextP2P, SendRecv_FloatArray)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    constexpr size_t COUNT = 128;
    constexpr int TAG = mpi_tags::TENSOR_DATA;

    std::vector<float> data(COUNT);

    if (rank_ == 0)
    {
        // Rank 0 sends
        for (size_t i = 0; i < COUNT; ++i)
        {
            data[i] = static_cast<float>(i) * 1.5f;
        }
        mpi_ctx_->sendFloat(data.data(), COUNT, 1, TAG);
        LOG_INFO("[Rank 0] Sent " << COUNT << " floats to rank 1");
    }
    else if (rank_ == 1)
    {
        // Rank 1 receives
        mpi_ctx_->recvFloat(data.data(), COUNT, 0, TAG);
        LOG_INFO("[Rank 1] Received " << COUNT << " floats from rank 0");

        // Verify data
        for (size_t i = 0; i < COUNT; ++i)
        {
            EXPECT_FLOAT_EQ(data[i], static_cast<float>(i) * 1.5f)
                << "Data mismatch at index " << i;
        }
    }
}

// ============================================================================
// Test 2: Large Buffer Transfer (1MB)
// ============================================================================

TEST_F(MPIContextP2P, SendRecv_LargeBuffer)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    // 1MB of floats = 256K elements
    constexpr size_t COUNT = 256 * 1024;
    constexpr int TAG = mpi_tags::ACTIVATION_FORWARD;

    std::vector<float> data(COUNT);

    if (rank_ == 0)
    {
        // Fill with pattern: i % 1000 + rank offset
        for (size_t i = 0; i < COUNT; ++i)
        {
            data[i] = static_cast<float>(i % 1000);
        }

        auto start = std::chrono::high_resolution_clock::now();
        mpi_ctx_->sendFloat(data.data(), COUNT, 1, TAG);
        auto end = std::chrono::high_resolution_clock::now();

        auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        LOG_INFO("[Rank 0] Sent 1MB in " << ms << " μs");
    }
    else if (rank_ == 1)
    {
        auto start = std::chrono::high_resolution_clock::now();
        mpi_ctx_->recvFloat(data.data(), COUNT, 0, TAG);
        auto end = std::chrono::high_resolution_clock::now();

        auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        LOG_INFO("[Rank 1] Received 1MB in " << ms << " μs");

        // Verify first and last elements
        EXPECT_FLOAT_EQ(data[0], 0.0f);
        EXPECT_FLOAT_EQ(data[999], 999.0f);
        EXPECT_FLOAT_EQ(data[COUNT - 1], static_cast<float>((COUNT - 1) % 1000));
    }
}

// ============================================================================
// Test 3: Non-blocking Isend/Irecv
// ============================================================================

TEST_F(MPIContextP2P, IsendIrecv_Async)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    constexpr size_t COUNT = 64;
    constexpr int TAG = mpi_tags::forwardTag(0); // Layer 0 activation

    std::vector<float> send_buf(COUNT);
    std::vector<float> recv_buf(COUNT, -1.0f);

    // Both ranks send and receive
    int partner = (rank_ == 0) ? 1 : 0;

    // Fill send buffer with rank-specific data
    for (size_t i = 0; i < COUNT; ++i)
    {
        send_buf[i] = static_cast<float>(rank_ * 1000 + i);
    }

    // Post non-blocking recv first, then send
    MPI_Request recv_req = mpi_ctx_->irecv(recv_buf.data(), COUNT, MPI_FLOAT, partner, TAG);
    MPI_Request send_req = mpi_ctx_->isend(send_buf.data(), COUNT, MPI_FLOAT, partner, TAG);

    // Wait for both to complete
    mpi_ctx_->wait(&recv_req);
    mpi_ctx_->wait(&send_req);

    // Verify received data is from partner
    float expected_base = static_cast<float>(partner * 1000);
    for (size_t i = 0; i < COUNT; ++i)
    {
        EXPECT_FLOAT_EQ(recv_buf[i], expected_base + static_cast<float>(i))
            << "Data mismatch at index " << i << " on rank " << rank_;
    }

    LOG_INFO("[Rank " << rank_ << "] Async exchange with rank " << partner << " successful");
}

// ============================================================================
// Test 4: WaitAll with Multiple Requests
// ============================================================================

TEST_F(MPIContextP2P, WaitAll_MultipleRequests)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    constexpr size_t COUNT = 32;
    constexpr int NUM_MESSAGES = 4;

    // Multiple messages with different tags
    std::vector<std::vector<float>> send_bufs(NUM_MESSAGES, std::vector<float>(COUNT));
    std::vector<std::vector<float>> recv_bufs(NUM_MESSAGES, std::vector<float>(COUNT, -1.0f));
    std::vector<MPI_Request> requests;

    int partner = (rank_ == 0) ? 1 : 0;

    // Post all recvs first
    for (int m = 0; m < NUM_MESSAGES; ++m)
    {
        int tag = mpi_tags::TENSOR_DATA + m;
        MPI_Request req = mpi_ctx_->irecv(recv_bufs[m].data(), COUNT, MPI_FLOAT, partner, tag);
        requests.push_back(req);
    }

    // Fill send buffers and post all sends
    for (int m = 0; m < NUM_MESSAGES; ++m)
    {
        for (size_t i = 0; i < COUNT; ++i)
        {
            send_bufs[m][i] = static_cast<float>(rank_ * 10000 + m * 1000 + i);
        }
        int tag = mpi_tags::TENSOR_DATA + m;
        MPI_Request req = mpi_ctx_->isend(send_bufs[m].data(), COUNT, MPI_FLOAT, partner, tag);
        requests.push_back(req);
    }

    // Wait for all operations
    mpi_ctx_->waitAll(requests);

    // Verify all requests are completed (MPI_REQUEST_NULL)
    for (const auto &req : requests)
    {
        EXPECT_EQ(req, MPI_REQUEST_NULL) << "Request not completed";
    }

    // Verify received data
    for (int m = 0; m < NUM_MESSAGES; ++m)
    {
        float expected_base = static_cast<float>(partner * 10000 + m * 1000);
        for (size_t i = 0; i < COUNT; ++i)
        {
            EXPECT_FLOAT_EQ(recv_bufs[m][i], expected_base + static_cast<float>(i))
                << "Message " << m << " data mismatch at index " << i;
        }
    }

    LOG_INFO("[Rank " << rank_ << "] WaitAll with " << requests.size() << " requests completed");
}

// ============================================================================
// Test 5: Probe for Pending Message
// ============================================================================

TEST_F(MPIContextP2P, Probe_MessageAvailable)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    constexpr size_t COUNT = 16;
    constexpr int TAG = mpi_tags::CONTROL + 99;

    std::vector<float> data(COUNT);

    if (rank_ == 0)
    {
        // Send data
        std::iota(data.begin(), data.end(), 0.0f);
        mpi_ctx_->sendFloat(data.data(), COUNT, 1, TAG);
        LOG_INFO("[Rank 0] Sent message with tag " << TAG);
    }
    else if (rank_ == 1)
    {
        // Probe first to check message availability
        MPI_Status status;
        mpi_ctx_->probe(0, TAG, &status);

        // Verify probe results
        EXPECT_EQ(status.MPI_SOURCE, 0) << "Unexpected source rank";
        EXPECT_EQ(status.MPI_TAG, TAG) << "Unexpected tag";

        // Get count from status
        int count = mpi_ctx_->getCount(status, MPI_FLOAT);
        EXPECT_EQ(count, static_cast<int>(COUNT)) << "Unexpected message count";

        // Now receive the data
        mpi_ctx_->recvFloat(data.data(), COUNT, 0, TAG);

        // Verify data
        for (size_t i = 0; i < COUNT; ++i)
        {
            EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
        }

        LOG_INFO("[Rank 1] Probe detected " << count << " elements, received successfully");
    }
}

// ============================================================================
// Test 6: Different Tags Don't Interfere
// ============================================================================

TEST_F(MPIContextP2P, SendRecv_WithTags)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    constexpr size_t COUNT = 8;

    // Use different tag namespaces
    constexpr int TAG_A = mpi_tags::ACTIVATION_FORWARD;
    constexpr int TAG_B = mpi_tags::KV_CACHE;
    constexpr int TAG_C = mpi_tags::CONTROL;

    std::vector<float> send_a(COUNT), send_b(COUNT), send_c(COUNT);
    std::vector<float> recv_a(COUNT), recv_b(COUNT), recv_c(COUNT);

    // Fill with distinct patterns
    std::fill(send_a.begin(), send_a.end(), 1.0f);
    std::fill(send_b.begin(), send_b.end(), 2.0f);
    std::fill(send_c.begin(), send_c.end(), 3.0f);

    if (rank_ == 0)
    {
        // Send in order: A, B, C
        mpi_ctx_->sendFloat(send_a.data(), COUNT, 1, TAG_A);
        mpi_ctx_->sendFloat(send_b.data(), COUNT, 1, TAG_B);
        mpi_ctx_->sendFloat(send_c.data(), COUNT, 1, TAG_C);
    }
    else if (rank_ == 1)
    {
        // Receive in different order: C, A, B (tags should match correctly)
        mpi_ctx_->recvFloat(recv_c.data(), COUNT, 0, TAG_C);
        mpi_ctx_->recvFloat(recv_a.data(), COUNT, 0, TAG_A);
        mpi_ctx_->recvFloat(recv_b.data(), COUNT, 0, TAG_B);

        // Verify each buffer received correct data
        for (size_t i = 0; i < COUNT; ++i)
        {
            EXPECT_FLOAT_EQ(recv_a[i], 1.0f) << "TAG_A data mismatch at " << i;
            EXPECT_FLOAT_EQ(recv_b[i], 2.0f) << "TAG_B data mismatch at " << i;
            EXPECT_FLOAT_EQ(recv_c[i], 3.0f) << "TAG_C data mismatch at " << i;
        }

        LOG_INFO("[Rank 1] Received messages with different tags correctly");
    }
}

// ============================================================================
// Test 7: Bidirectional Send/Recv
// ============================================================================

TEST_F(MPIContextP2P, SendRecv_Bidirectional)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    constexpr size_t COUNT = 64;
    constexpr int TAG = mpi_tags::ACTIVATION_FORWARD;

    std::vector<float> send_buf(COUNT);
    std::vector<float> recv_buf(COUNT, -1.0f);

    int partner = (rank_ == 0) ? 1 : 0;

    // Fill send buffer
    for (size_t i = 0; i < COUNT; ++i)
    {
        send_buf[i] = static_cast<float>(rank_ * 100 + i);
    }

    // Use async to avoid deadlock - both ranks send and receive
    MPI_Request send_req = mpi_ctx_->isend(send_buf.data(), COUNT, MPI_FLOAT, partner, TAG);
    MPI_Request recv_req = mpi_ctx_->irecv(recv_buf.data(), COUNT, MPI_FLOAT, partner, TAG);

    // Wait for both
    std::vector<MPI_Request> requests = {send_req, recv_req};
    mpi_ctx_->waitAll(requests);

    // Verify received data
    float expected_base = static_cast<float>(partner * 100);
    for (size_t i = 0; i < COUNT; ++i)
    {
        EXPECT_FLOAT_EQ(recv_buf[i], expected_base + static_cast<float>(i))
            << "Bidirectional data mismatch at index " << i;
    }

    LOG_INFO("[Rank " << rank_ << "] Bidirectional exchange with rank " << partner << " successful");
}

// ============================================================================
// Test 8: Non-blocking Probe (iprobe)
// ============================================================================

TEST_F(MPIContextP2P, IProbe_NoMessagePending)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    // Ensure no messages are pending first
    MPI_Barrier(MPI_COMM_WORLD);

    // iprobe should return false when no message is available
    MPI_Status status;
    bool message_available = mpi_ctx_->iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, &status);

    EXPECT_FALSE(message_available) << "No message should be pending after barrier";
    LOG_INFO("[Rank " << rank_ << "] iprobe correctly reports no pending messages");
}

// ============================================================================
// Test 9: Raw Bytes Transfer
// ============================================================================

TEST_F(MPIContextP2P, SendRecv_RawBytes)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    // Test struct to send as raw bytes
    struct TestData
    {
        int32_t seq;
        float value;
        uint64_t timestamp;
    };

    constexpr size_t BYTE_COUNT = sizeof(TestData);
    constexpr int TAG = mpi_tags::CONTROL;

    TestData send_data, recv_data;

    if (rank_ == 0)
    {
        send_data = {42, 3.14159f, 1234567890ULL};
        mpi_ctx_->sendBytes(&send_data, BYTE_COUNT, 1, TAG);
        LOG_INFO("[Rank 0] Sent " << BYTE_COUNT << " bytes of struct data");
    }
    else if (rank_ == 1)
    {
        mpi_ctx_->recvBytes(&recv_data, BYTE_COUNT, 0, TAG);

        EXPECT_EQ(recv_data.seq, 42);
        EXPECT_FLOAT_EQ(recv_data.value, 3.14159f);
        EXPECT_EQ(recv_data.timestamp, 1234567890ULL);

        LOG_INFO("[Rank 1] Received struct: seq=" << recv_data.seq
                                                  << " value=" << recv_data.value
                                                  << " timestamp=" << recv_data.timestamp);
    }
}

// ============================================================================
// Test 10: Pipeline Tag Helpers
// ============================================================================

TEST_F(MPIContextP2P, TagHelpers_LayerTags)
{
    // Test that tag helpers generate correct values
    EXPECT_EQ(mpi_tags::forwardTag(0), mpi_tags::ACTIVATION_FORWARD + 0);
    EXPECT_EQ(mpi_tags::forwardTag(5), mpi_tags::ACTIVATION_FORWARD + 5);
    EXPECT_EQ(mpi_tags::forwardTag(31), mpi_tags::ACTIVATION_FORWARD + 31);

    EXPECT_EQ(mpi_tags::backwardTag(0), mpi_tags::ACTIVATION_BACKWARD + 0);
    EXPECT_EQ(mpi_tags::backwardTag(5), mpi_tags::ACTIVATION_BACKWARD + 5);

    EXPECT_EQ(mpi_tags::kvCacheTag(0), mpi_tags::KV_CACHE + 0);
    EXPECT_EQ(mpi_tags::kvCacheTag(10), mpi_tags::KV_CACHE + 10);

    // KV key/value tags should not overlap
    EXPECT_NE(mpi_tags::kvCacheKeyTag(5), mpi_tags::kvCacheValueTag(5));

    LOG_INFO("Tag helper tests passed");
}

// ============================================================================
// Test 11: Control Message Tags
// ============================================================================

TEST_F(MPIContextP2P, ControlMessages_ReadySignal)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    // Simulate pipeline ready signaling
    int ready_signal = 1;
    int received_signal = 0;

    if (rank_ == 0)
    {
        // Wait for ready signal from rank 1
        mpi_ctx_->recv(&received_signal, 1, MPI_INT, 1, mpi_tags::control::READY);
        EXPECT_EQ(received_signal, 1) << "Expected ready signal";
        LOG_INFO("[Rank 0] Received READY signal from rank 1");
    }
    else if (rank_ == 1)
    {
        // Send ready signal to rank 0
        mpi_ctx_->send(&ready_signal, 1, MPI_INT, 0, mpi_tags::control::READY);
        LOG_INFO("[Rank 1] Sent READY signal to rank 0");
    }
}
