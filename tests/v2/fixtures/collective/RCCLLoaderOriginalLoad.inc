/**
 * @file RCCLLoaderOriginalLoad.inc
 * @brief Literal cached-handle-before-path-validation defect, test control only.
 * The original complete load definition preserves native optional-symbol policy
 * and partial resolution cleanup; it accepts mismatched/empty paths when loaded.
 */
bool load(const char *library_path)
        {
            std::lock_guard<std::mutex> lock(g_mutex);

            if (g_library_handle)
            {
                // Already loaded
                return true;
            }

            // A search list can silently select packaged GPU code that differs
            // from the dependency admitted by the build. In particular, moving
            // a checkout into /src must not change which RCCL executes.
            const char *path = library_path;
#ifdef LLAMINAR_RCCL_LIBRARY_PATH
            if (!path)
                path = LLAMINAR_RCCL_LIBRARY_PATH;
#endif
            if (!path || !*path)
            {
                g_last_error = "No RCCL library configured; set RCCL_LIBRARY when configuring the build";
                LOG_ERROR("RCCL Dynamic Loader: " << g_last_error);
                return false;
            }

            // Clear any previous dlerror
            dlerror();

            LOG_DEBUG("RCCL Dynamic Loader: Loading configured dependency '" << path << "'");
            // Resolve symbols immediately and keep them local so CUDA's NCCL
            // and ROCm's RCCL can coexist without sharing function addresses.
            g_library_handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
            if (!g_library_handle)
            {
                g_last_error = std::string("Failed to load configured RCCL library '") + path + "': " + dlerror();
                LOG_ERROR("RCCL Dynamic Loader: " << g_last_error);
                return false;
            }
            LOG_INFO("RCCL Dynamic Loader: Successfully loaded '" << path << "'");

            // Load all required symbols
            bool success = true;
            success = success && loadSymbol(fp_ncclGetUniqueId, "ncclGetUniqueId");
            success = success && loadSymbol(fp_ncclCommInitRank, "ncclCommInitRank");
            success = success && loadSymbol(fp_ncclCommInitAll, "ncclCommInitAll");
            success = success && loadSymbol(fp_ncclCommDestroy, "ncclCommDestroy");

            // ncclCommAbort is optional - don't fail if not found
            // It's available in RCCL 2.13+ and provides safer cleanup for unused communicators
            if (!loadSymbol(fp_ncclCommAbort, "ncclCommAbort"))
            {
                LOG_DEBUG("RCCL Dynamic Loader: ncclCommAbort not available (optional, RCCL 2.13+)");
                fp_ncclCommAbort = nullptr;
            }
            // ncclCommFinalize is optional - available in RCCL 2.14+
            if (!loadSymbol(fp_ncclCommFinalize, "ncclCommFinalize"))
            {
                LOG_DEBUG("RCCL Dynamic Loader: ncclCommFinalize not available (optional, RCCL 2.14+)");
                fp_ncclCommFinalize = nullptr;
            }
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
            success = success && loadSymbol(fp_host_transport_storage, "llaminarRcclHostTransportStorage");
            success = success && loadSymbol(fp_ncclSend, "ncclSend");
            success = success && loadSymbol(fp_ncclRecv, "ncclRecv");
            success = success && loadSymbol(fp_ncclGroupStart, "ncclGroupStart");
            success = success && loadSymbol(fp_ncclGroupEnd, "ncclGroupEnd");

            if (!success)
            {
                LOG_ERROR("RCCL Dynamic Loader: " << g_last_error);
                clearResolvedSymbols();
                dlclose(g_library_handle);
                g_library_handle = nullptr;
                return false;
            }

            LOG_INFO("RCCL Dynamic Loader: All required symbols loaded successfully");
            return true;
        }
