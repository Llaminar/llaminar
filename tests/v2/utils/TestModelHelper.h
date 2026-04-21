/**
 * @file TestModelHelper.h
 * @brief Shared helper for loading GGUF model fixtures in tests.
 * @author David Sanftenberg
 *
 * `ModelLoader::loadModel()` now throws std::runtime_error on missing or
 * corrupt GGUF files (commit c5c1b46d). Tests that used the pre-change
 * bool-return skip pattern would spuriously fail on CI runners without the
 * model fixture. This helper swallows both the false-return and any
 * thrown exception, returning false for either — preserving the original
 * `if (!tryLoadModel(...)) GTEST_SKIP();` ergonomics.
 *
 * Usage:
 *   #include "utils/TestModelHelper.h"
 *
 *   if (!llaminar2::test::tryLoadModel(loader, model_path)) {
 *       GTEST_SKIP() << "Model file unavailable: " << model_path;
 *   }
 */

#pragma once

#include "loaders/ModelLoader.h"
#include <exception>
#include <string>

namespace llaminar2
{

    /**
     * Attempts to load a GGUF model, returning false instead of throwing
     * if the file is missing, unreadable, or corrupt.
     *
     * ModelLoader::loadModel() was hardened in commit c5c1b46d to throw
     * std::runtime_error on unrecoverable errors. This wrapper preserves
     * the pre-change bool-return contract so existing `if (!loader.loadModel(...))
     * GTEST_SKIP()` patterns continue to work when the fixture is absent.
     *
     * Placed in namespace `llaminar2` so ADL finds it at every call site
     * (ModelLoader's namespace), regardless of the surrounding test namespace.
     *
     * @param loader  ModelLoader instance.
     * @param path    Filesystem path to the .gguf file.
     * @return true on success; false on any failure (missing, bad magic,
     *         truncated header, etc.).
     */
    inline bool tryLoadModel(ModelLoader &loader, const std::string &path)
    {
        try
        {
            return loader.loadModel(path);
        }
        catch (const std::exception &)
        {
            return false;
        }
    }

    inline bool tryLoadModel(ModelLoader *loader, const std::string &path)
    {
        return tryLoadModel(*loader, path);
    }

} // namespace llaminar2
