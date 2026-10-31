/**
 * @file NCCLLoaderOriginalLoad.inc
 * @brief Literal former production load function used only as a negative oracle.
 * The bare default SONAME can select a different DSO from CMake's certified path.
 * This function is substituted into a disposable copy of the actual loader;
 * no production switch or alternative loader implementation selects it.
 */
        bool load(const char *library_path)
        {
            std::lock_guard<std::mutex> lock(g_mutex);

            if (g_library_handle)
            {
                // Already loaded
                return true;
            }

            // The distinct SONAME is deliberate: an unpatched system library
            // fails when a retained parent resumes capture with the same ID.
            // Resolve exactly one dependency; never hide a missing installation
            // by loading the incompatible distribution package instead.
            const char *path = library_path ? library_path : "libllaminar_nccl.so.2";

            // Clear any previous dlerror
            dlerror();

            LOG_DEBUG("NCCL Dynamic Loader: Trying to load '" << path << "'");

            // Resolve immediately, but never export NCCL symbols into RCCL's
            // namespace. The two vendors intentionally use identical names.
            g_library_handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
            if (g_library_handle)
                LOG_INFO("NCCL Dynamic Loader: Successfully loaded '" << path << "'");
            else
                LOG_DEBUG("NCCL Dynamic Loader: Failed to load '" << path << "': " << dlerror());

            if (!g_library_handle)
            {
                g_last_error = std::string("Failed to load canonical NCCL library: ") + path;
                LOG_ERROR("NCCL Dynamic Loader: " << g_last_error);
                return false;
            }

            // Load all required symbols
            bool success = true;
            success = success && loadSymbol(fp_ncclGetVersion, "ncclGetVersion");
            success = success && loadSymbol(fp_ncclGetUniqueId, "ncclGetUniqueId");
            success = success && loadSymbol(fp_ncclCommInitRank, "ncclCommInitRank");
            success = success && loadSymbol(fp_ncclCommInitRankConfig, "ncclCommInitRankConfig");
            success = success && loadSymbol(fp_ncclCommDestroy, "ncclCommDestroy");
            success = success && loadSymbol(fp_ncclCommAbort, "ncclCommAbort");
            success = success && loadSymbol(fp_ncclCommCount, "ncclCommCount");
            success = success && loadSymbol(fp_ncclCommCuDevice, "ncclCommCuDevice");
            success = success && loadSymbol(fp_ncclCommUserRank, "ncclCommUserRank");
            success = success && loadSymbol(fp_ncclGetErrorString, "ncclGetErrorString");
            success = success && loadSymbol(fp_ncclAllReduce, "ncclAllReduce");
            success = success && loadSymbol(fp_ncclBroadcast, "ncclBroadcast");
            success = success && loadSymbol(fp_ncclReduce, "ncclReduce");
            success = success && loadSymbol(fp_ncclAllGather, "ncclAllGather");
            success = success && loadSymbol(fp_ncclReduceScatter, "ncclReduceScatter");
            success = success && loadSymbol(fp_device_rows_gather, "llaminarNcclAllGatherDeviceRows");
            success = success && loadSymbol(fp_device_rows_reduce, "llaminarNcclAllReduceDeviceRows");
            success = success && loadSymbol(fp_device_rows_scatter, "llaminarNcclReduceScatterDeviceRows");
            success = success && loadSymbol(fp_ncclSend, "ncclSend");
            success = success && loadSymbol(fp_ncclRecv, "ncclRecv");
            success = success && loadSymbol(fp_ncclGroupStart, "ncclGroupStart");
            success = success && loadSymbol(fp_ncclGroupEnd, "ncclGroupEnd");

            if (!success)
            {
                LOG_ERROR("NCCL Dynamic Loader: " << g_last_error);
                dlclose(g_library_handle);
                g_library_handle = nullptr;
                return false;
            }

            LOG_INFO("NCCL Dynamic Loader: All " << 19 << " symbols loaded successfully");
            return true;
        }
