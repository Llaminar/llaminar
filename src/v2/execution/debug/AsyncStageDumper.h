/**
 * @file AsyncStageDumper.h
 * @brief Asynchronous stage dumping for high-performance debugging
 * @author David Sanftenberg
 * @date January 2026
 *
 * Provides zero-blocking stage buffer dumping by using background I/O threads.
 * The main execution thread only copies tensor data to a memory queue, while
 * dedicated I/O threads handle the actual file writes in the background.
 *
 * Key Benefits:
 * - Near-zero impact on inference latency
 * - Memory copies are fast (memcpy is ~10GB/s)
 * - File I/O happens in parallel with computation
 * - Automatic backpressure if I/O can't keep up
 *
 * Architecture:
 *   Main Thread                    I/O Thread Pool (2-4 threads)
 *   ───────────────────────────────────────────────────────────
 *   copyToQueue(data)  ──queue──>  dequeueAndWrite()
 *   (fast memcpy)                  (slow file I/O)
 *   continue execution             write to disk in background
 *
 * Usage:
 *   // At startup
 *   AsyncStageDumper::initialize(num_io_threads);
 *
 *   // During execution (replaces StageDumper calls)
 *   if (should_dump) {
 *       AsyncStageDumper::enqueue(dump_ctx, stage->getDumpInfo(), "inputs");
 *   }
 *
 *   // At shutdown
 *   AsyncStageDumper::shutdown();  // Flushes remaining writes
 */

#pragma once

#include "StageDumper.h"
#include "../../utils/Logger.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace llaminar2
{

    /**
     * @brief A buffer snapshot to be written asynchronously
     */
    struct DumpTask
    {
        std::string file_path;             ///< Full path to output file
        std::unique_ptr<uint8_t[]> data;   ///< Copied buffer data (owned)
        size_t byte_size = 0;              ///< Size of data in bytes
        TensorDumpMeta meta;               ///< Metadata for the tensor
        std::string meta_path;             ///< Path for metadata file
        std::function<void()> on_complete; ///< Optional callback when done
    };

    /**
     * @brief Thread-safe task queue for async I/O
     */
    class DumpTaskQueue
    {
    public:
        void push(DumpTask &&task)
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                queue_.push(std::move(task));
                pending_count_.fetch_add(1, std::memory_order_relaxed);
            }
            cv_.notify_one();
        }

        bool pop(DumpTask &task)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]
                     { return !queue_.empty() || shutdown_; });

            if (shutdown_ && queue_.empty())
                return false;

            task = std::move(queue_.front());
            queue_.pop();
            return true;
        }

        void shutdown()
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                shutdown_ = true;
            }
            cv_.notify_all();
        }

        size_t size() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.size();
        }

        void markCompleted()
        {
            pending_count_.fetch_sub(1, std::memory_order_relaxed);
        }

        size_t pendingCount() const
        {
            return pending_count_.load(std::memory_order_relaxed);
        }

        bool isEmpty() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.empty();
        }

    private:
        mutable std::mutex mutex_;
        std::condition_variable cv_;
        std::queue<DumpTask> queue_;
        std::atomic<size_t> pending_count_{0};
        bool shutdown_ = false;
    };

    /**
     * @brief Asynchronous stage dumper with background I/O threads
     */
    class AsyncStageDumper
    {
    public:
        /**
         * @brief Initialize the async dumper with specified thread count
         * @param num_threads Number of I/O threads (default: 2)
         * @param max_queue_size Max queued tasks before blocking (0 = unlimited)
         */
        static void initialize(size_t num_threads = 2, size_t max_queue_size = 100)
        {
            auto &instance = getInstance();
            std::lock_guard<std::mutex> lock(instance.init_mutex_);

            if (instance.initialized_)
            {
                LOG_WARN("[AsyncStageDumper] Already initialized");
                return;
            }

            instance.max_queue_size_ = max_queue_size;

            // Start I/O worker threads
            for (size_t i = 0; i < num_threads; ++i)
            {
                instance.workers_.emplace_back([&instance, i]()
                                               { instance.workerLoop(i); });
            }

            instance.initialized_ = true;
            LOG_DEBUG("[AsyncStageDumper] Initialized with " << num_threads << " I/O threads");
        }

        /**
         * @brief Shutdown the dumper, waiting for all pending writes to complete
         */
        static void shutdown()
        {
            auto &instance = getInstance();
            std::lock_guard<std::mutex> lock(instance.init_mutex_);

            if (!instance.initialized_)
                return;

            LOG_DEBUG("[AsyncStageDumper] Shutting down, "
                     << instance.queue_.pendingCount() << " tasks pending...");

            instance.queue_.shutdown();

            for (auto &worker : instance.workers_)
            {
                if (worker.joinable())
                    worker.join();
            }

            instance.workers_.clear();
            instance.initialized_ = false;

            LOG_DEBUG("[AsyncStageDumper] Shutdown complete. Total writes: "
                     << instance.total_writes_.load()
                     << ", bytes: " << instance.total_bytes_.load());
        }

        /**
         * @brief Check if async dumper is initialized
         */
        static bool isInitialized()
        {
            return getInstance().initialized_;
        }

        /**
         * @brief Enqueue a tensor buffer for async dump
         *
         * This copies the buffer to an internal queue and returns immediately.
         * The actual file write happens in a background thread.
         *
         * @param file_path Full path to output file
         * @param data Pointer to source data
         * @param byte_size Size of data in bytes
         * @param meta Tensor metadata
         * @param meta_path Path for metadata file (empty to skip)
         * @return true if enqueued successfully, false if queue is full
         */
        static bool enqueue(
            const std::string &file_path,
            const void *data,
            size_t byte_size,
            const TensorDumpMeta &meta,
            const std::string &meta_path = "")
        {
            auto &instance = getInstance();

            if (!instance.initialized_)
            {
                LOG_WARN("[AsyncStageDumper] Not initialized, falling back to sync write");
                return syncWrite(file_path, data, byte_size, meta, meta_path);
            }

            // Check backpressure
            if (instance.max_queue_size_ > 0 &&
                instance.queue_.size() >= instance.max_queue_size_)
            {
                LOG_WARN("[AsyncStageDumper] Queue full (" << instance.max_queue_size_
                                                           << " tasks), blocking...");
                // Block until queue has space (backpressure)
                while (instance.queue_.size() >= instance.max_queue_size_)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }

            // Create task with data copy
            DumpTask task;
            task.file_path = file_path;
            task.byte_size = byte_size;
            task.meta = meta;
            task.meta_path = meta_path;

            // Copy buffer data
            task.data = std::make_unique<uint8_t[]>(byte_size);
            std::memcpy(task.data.get(), data, byte_size);

            instance.queue_.push(std::move(task));
            return true;
        }

        /**
         * @brief Enqueue inputs from a StageDumpInfo
         * @param ctx Dump context with paths
         * @param dump_info Stage dump info
         * @param cfg Debug config (for dump_inputs flag)
         */
        static void enqueueInputs(
            const StageDumpContext &ctx,
            const StageDumpInfo &dump_info)
        {
            const auto &cfg = debugEnv().stage_dump;
            if (!cfg.dump_inputs && !cfg.dump_weights)
                return;

            StageDumper::writeWeightMetadata(ctx, dump_info);
            StageDumper::writeScalars(ctx, dump_info);

            if (!cfg.dump_inputs)
                return;

            for (const auto &input : dump_info.inputs)
            {
                const void *input_data = StageDumper::hostDataForDump(input);
                if (!input_data)
                    continue;

                std::string name = input.name;
                std::string path = ctx.dump_dir + "/inputs/" + name;

                TensorDumpMeta meta;
                meta.name = input.name;
                meta.rows = input.rows;
                meta.cols = input.cols;
                meta.dtype = input.dtype;
                meta.computeBlockInfo();

                size_t byte_size = input.byte_size;
                std::string dtype_lower = input.dtype;
                std::transform(dtype_lower.begin(), dtype_lower.end(),
                               dtype_lower.begin(), ::tolower);

                if (dtype_lower == "fp32")
                {
                    path += ".bin";
                    meta.element_count = input.rows * input.cols;
                    meta.byte_size = byte_size;
                }
                else
                {
                    path += "_" + dtype_lower + ".bin";
                    meta.byte_size = byte_size;
                }

                enqueue(path, input_data, byte_size, meta,
                        ctx.dump_dir + "/inputs/" + name + "_meta.txt");
            }
        }

        /**
         * @brief Enqueue outputs from a StageDumpInfo
         */
        static void enqueueOutputs(
            const StageDumpContext &ctx,
            const StageDumpInfo &dump_info)
        {
            const auto &cfg = debugEnv().stage_dump;
            if (!cfg.dump_outputs)
                return;

            dump_info.ensureOutputsOnHost();

            for (const auto &output : dump_info.outputs)
            {
                const void *output_data = StageDumper::hostDataForDump(output);
                if (!output_data)
                    continue;

                std::string name = output.name;
                std::string path = ctx.dump_dir + "/outputs/" + name;

                TensorDumpMeta meta;
                meta.name = output.name;
                meta.rows = output.rows;
                meta.cols = output.cols;
                meta.dtype = output.dtype;
                meta.computeBlockInfo();

                size_t byte_size = output.byte_size;
                std::string dtype_lower = output.dtype;
                std::transform(dtype_lower.begin(), dtype_lower.end(),
                               dtype_lower.begin(), ::tolower);

                if (dtype_lower == "fp32")
                {
                    path += ".bin";
                    meta.element_count = output.rows * output.cols;
                    meta.byte_size = byte_size;
                }
                else
                {
                    path += "_" + dtype_lower + ".bin";
                    meta.byte_size = byte_size;
                }

                enqueue(path, output_data, byte_size, meta,
                        ctx.dump_dir + "/outputs/" + name + "_meta.txt");
            }

            StageDumper::writeScalars(ctx, dump_info);
        }

        /**
         * @brief Get number of pending tasks
         */
        static size_t pendingTasks()
        {
            return getInstance().queue_.pendingCount();
        }

        /**
         * @brief Wait for all pending writes to complete
         */
        static void waitForCompletion()
        {
            auto &instance = getInstance();
            while (instance.queue_.pendingCount() > 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

    private:
        static AsyncStageDumper &getInstance()
        {
            static AsyncStageDumper instance;
            return instance;
        }

        // Private constructor and destructor for singleton
        AsyncStageDumper() = default;

        ~AsyncStageDumper()
        {
            // Ensure clean shutdown on destruction
            if (initialized_)
            {
                LOG_DEBUG("[AsyncStageDumper] Destructor called, shutting down...");
                queue_.shutdown();
                for (auto &worker : workers_)
                {
                    if (worker.joinable())
                        worker.join();
                }
                workers_.clear();
                initialized_ = false;
            }
        }

        // Prevent copy/move
        AsyncStageDumper(const AsyncStageDumper &) = delete;
        AsyncStageDumper &operator=(const AsyncStageDumper &) = delete;

        void workerLoop(size_t worker_id)
        {
            LOG_DEBUG("[AsyncStageDumper] Worker " << worker_id << " started");

            try
            {
                DumpTask task;
                while (queue_.pop(task))
                {
                    try
                    {
                        writeTask(task);
                    }
                    catch (const std::exception &e)
                    {
                        LOG_ERROR("[AsyncStageDumper] Worker " << worker_id
                                                               << " task error: " << e.what());
                    }
                    queue_.markCompleted();
                }
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[AsyncStageDumper] Worker " << worker_id
                                                       << " fatal error: " << e.what());
            }

            LOG_DEBUG("[AsyncStageDumper] Worker " << worker_id << " exiting");
        }

        void writeTask(const DumpTask &task)
        {
            // Ensure directory exists
            std::string dir = task.file_path.substr(0, task.file_path.rfind('/'));
            mkdir(dir.c_str(), 0755);

            // Write binary data
            FILE *f = fopen(task.file_path.c_str(), "wb");
            if (f)
            {
                fwrite(task.data.get(), 1, task.byte_size, f);
                fclose(f);

                total_bytes_.fetch_add(task.byte_size, std::memory_order_relaxed);
                total_writes_.fetch_add(1, std::memory_order_relaxed);
            }
            else
            {
                LOG_ERROR("[AsyncStageDumper] Failed to write: " << task.file_path);
            }

            // Write metadata if path provided
            if (!task.meta_path.empty())
            {
                writeTensorMeta(task.meta_path, task.meta);
            }

            if (task.on_complete)
            {
                task.on_complete();
            }
        }

        /**
         * @brief Write tensor metadata file (inline implementation)
         */
        static void writeTensorMeta(const std::string &path, const TensorDumpMeta &meta)
        {
            FILE *f = fopen(path.c_str(), "w");
            if (!f)
                return;

            fprintf(f, "name=%s\n", meta.name.c_str());
            fprintf(f, "rows=%zu\n", meta.rows);
            fprintf(f, "cols=%zu\n", meta.cols);
            fprintf(f, "dtype=%s\n", meta.dtype.c_str());
            fprintf(f, "element_count=%zu\n", meta.element_count);
            fprintf(f, "byte_size=%zu\n", meta.byte_size);

            if (meta.block_count > 0)
            {
                fprintf(f, "# Block format info:\n");
                fprintf(f, "block_count=%zu\n", meta.block_count);
                fprintf(f, "blocks_per_row=%zu\n", meta.blocks_per_row);
                fprintf(f, "block_element_size=%zu\n", meta.block_element_size);
            }

            if (meta.sample_min != 0 || meta.sample_max != 0 || meta.sample_mean != 0)
            {
                fprintf(f, "sample_min=%f\n", meta.sample_min);
                fprintf(f, "sample_max=%f\n", meta.sample_max);
                fprintf(f, "sample_mean=%f\n", meta.sample_mean);
            }

            fclose(f);
        }

        /**
         * @brief Synchronous write fallback when not initialized
         */
        static bool syncWrite(
            const std::string &file_path,
            const void *data,
            size_t byte_size,
            const TensorDumpMeta &meta,
            const std::string &meta_path)
        {
            // Ensure directory exists
            std::string dir = file_path.substr(0, file_path.rfind('/'));
            mkdir(dir.c_str(), 0755);

            FILE *f = fopen(file_path.c_str(), "wb");
            if (!f)
                return false;

            fwrite(data, 1, byte_size, f);
            fclose(f);

            if (!meta_path.empty())
            {
                writeTensorMeta(meta_path, meta);
            }

            return true;
        }

        // Member variables
        std::mutex init_mutex_;
        bool initialized_ = false;
        size_t max_queue_size_ = 100;
        DumpTaskQueue queue_;
        std::vector<std::thread> workers_;
        std::atomic<size_t> total_writes_{0};
        std::atomic<size_t> total_bytes_{0};
    };

} // namespace llaminar2
