#include "test_timeout_guard.h"
#include "mpi_kernel_base.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <chrono>
#include <memory>

using namespace llaminar;

class MPIKernelBaseTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize MPI if not already done
        int flag;
        MPI_Initialized(&flag);
        if (!flag)
        {
            int provided;
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        }
    }

    void TearDown() override
    {
        // Note: Don't finalize MPI here as it might be used by other tests
    }
};

class TestMPIKernel : public MPIKernelBase
{
public:
    TestMPIKernel(MPI_Comm comm = MPI_COMM_WORLD) : MPIKernelBase(comm, false) {}

    // Implement pure virtual methods from KernelBase
    bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                 std::vector<std::shared_ptr<TensorBase>> &outputs) override
    {
        // Dummy implementation for testing
        return true;
    }

    bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                  const std::vector<std::shared_ptr<TensorBase>> &outputs) const override
    {
        return true;
    }

    std::string getKernelType() const override
    {
        return "TestMPIKernel";
    }

    size_t getExpectedInputCount() const override
    {
        return 1;
    }

    size_t getExpectedOutputCount() const override
    {
        return 1;
    }

    // Test helper methods to access protected functionality
    std::pair<int, int> testGetRowDistribution(int global_size, int rank = -1)
    {
        return getRowDistribution(global_size, rank);
    }

    std::pair<int, int> testGetColDistribution(int global_size, int rank = -1)
    {
        return getColDistribution(global_size, rank);
    }

    std::tuple<int, int, int, int> testGetBlockDistribution(int rows, int cols, int rank = -1)
    {
        return getBlockDistribution(rows, cols, rank);
    }

    void testSynchronize() const
    {
        synchronize();
    }

    void testBroadcast(float *data, int count, int root = 0) const
    {
        broadcast(data, count, root);
    }

    void testAllReduceSum(const float *send_data, float *recv_data, int count) const
    {
        allReduceSum(send_data, recv_data, count);
    }

    void testAllReduceMax(const float *send_data, float *recv_data, int count) const
    {
        allReduceMax(send_data, recv_data, count);
    }

    void testAllGather(const float *send_data, int send_count, std::vector<float> &recv_data) const
    {
        allGather(send_data, send_count, recv_data);
    }

    MPI_Comm testGetCommunicator() const
    {
        return getComm();
    }
};

TEST_F(MPIKernelBaseTest, BasicMPIFunctionality)
{
    TestMPIKernel kernel;

    // Test basic MPI properties
    EXPECT_GE(kernel.getRank(), 0);
    EXPECT_GE(kernel.getSize(), 1);
    EXPECT_EQ(kernel.testGetCommunicator(), MPI_COMM_WORLD);
}

TEST_F(MPIKernelBaseTest, DistributionCalculations)
{
    TestMPIKernel kernel;

    int size = kernel.getSize();

    // Test row distribution
    {
        int global_size = 100;
        auto [local_size, offset] = kernel.testGetRowDistribution(global_size, 0);

        EXPECT_GT(local_size, 0);
        EXPECT_GE(offset, 0);
        EXPECT_LE(offset + local_size, global_size);
    }

    // Test column distribution
    {
        int global_size = 100;
        auto [local_size, offset] = kernel.testGetColDistribution(global_size, 0);

        EXPECT_GT(local_size, 0);
        EXPECT_GE(offset, 0);
        EXPECT_LE(offset + local_size, global_size);
    }

    // Test block distribution
    {
        int rows = 64, cols = 64;
        auto [local_rows, local_cols, row_offset, col_offset] = kernel.testGetBlockDistribution(rows, cols, 0);

        EXPECT_GT(local_rows, 0);
        EXPECT_GT(local_cols, 0);
        EXPECT_GE(row_offset, 0);
        EXPECT_GE(col_offset, 0);
        EXPECT_LE(row_offset + local_rows, rows);
        EXPECT_LE(col_offset + local_cols, cols);
    }
}

TEST_F(MPIKernelBaseTest, DistributionConsistency)
{
    TestMPIKernel kernel;

    int size = kernel.getSize();
    int global_size = 100;

    // Test that all ranks get consistent distribution
    int total_elements = 0;
    for (int rank = 0; rank < size; ++rank)
    {
        auto [local_size, offset] = kernel.testGetRowDistribution(global_size, rank);
        total_elements += local_size;

        EXPECT_GT(local_size, 0);
        EXPECT_GE(offset, 0);
    }

    // Total elements should equal global size
    EXPECT_EQ(total_elements, global_size);
}

TEST_F(MPIKernelBaseTest, CommunicationOperations)
{
    TestMPIKernel kernel;

    // Test synchronization
    EXPECT_NO_THROW(kernel.testSynchronize());

    // Test broadcast if we have multiple processes
    if (kernel.getSize() > 1)
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f};
        EXPECT_NO_THROW(kernel.testBroadcast(data.data(), data.size(), 0));
    }

    // Test allreduce operations
    {
        std::vector<float> send_data = {1.0f, 2.0f, 3.0f};
        std::vector<float> recv_data(send_data.size());

        EXPECT_NO_THROW(kernel.testAllReduceSum(send_data.data(), recv_data.data(), send_data.size()));
        EXPECT_NO_THROW(kernel.testAllReduceMax(send_data.data(), recv_data.data(), send_data.size()));
    }

    // Test allgather
    {
        std::vector<float> send_data = {1.0f, 2.0f};
        std::vector<float> recv_data;

        EXPECT_NO_THROW(kernel.testAllGather(send_data.data(), send_data.size(), recv_data));
        EXPECT_EQ(recv_data.size(), send_data.size() * kernel.getSize());
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI for testing
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    auto timeout = llaminar::test_util::TestTimeoutGuard::ResolveTimeout(
        {"LLAMINAR_TEST_TIMEOUT_MS"}, std::chrono::milliseconds(60000));
    llaminar::test_util::TestTimeoutGuard watchdog("MPIKernelBaseTest", timeout);

    int result = RUN_ALL_TESTS();

    watchdog.disarm();

    // Finalize MPI
    MPI_Finalize();

    return result;
}