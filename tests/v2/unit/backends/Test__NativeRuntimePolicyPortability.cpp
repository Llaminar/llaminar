/**
 * @file Test__NativeRuntimePolicyPortability.cpp
 * @brief Replay portable native resource classes independently from strict live physical selectors.
 */
#include "backends/NativeContextRuntimePolicy.h"
#include "backends/rocm/AMDGPUModuleIdentity.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <array>
#include <optional>
namespace llaminar2::test {
/** @test Authenticate either a versioned module or the complete loaded GNU note and source version. */
TEST(NativeRuntimePolicyPortability, LoadedAMDGPUModuleRetainsOriginalIdentity) {
    std::string pattern = (::testing::TempDir() + "/amdgpu-module-XXXXXX");
    ASSERT_NE(::mkdtemp(pattern.data()), nullptr);
    const std::filesystem::path module(pattern);
    struct Cleanup {
        std::filesystem::path directory;
        ~Cleanup() { std::error_code error; std::filesystem::remove_all(directory, error); }
    } cleanup{module};
    std::filesystem::create_directory(module / "notes");
    std::ofstream(module / "srcversion") << "source\n";
    std::ofstream(module / "version") << "dkms\n";
    const auto versioned = AMDGPUModuleIdentity::observe(module, "release");
    EXPECT_EQ(versioned.version, "dkms\n:release");
    EXPECT_EQ(versioned.source, "source\n");

    std::filesystem::remove(module / "version");
    EXPECT_THROW(AMDGPUModuleIdentity::observe(module, "release"), std::runtime_error);
    const auto note = module / "notes/.note.gnu.build-id";
    const Elf32_Nhdr header{4, 4, NT_GNU_BUILD_ID};
    const auto writeNote = [&](const char* payload) {
        std::ofstream file(note, std::ios::binary);
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
        file.write("GNU", 4);
        file.write(payload, 4);
    };
    writeNote("\x00\xff\x08\x10");
    const auto linked = AMDGPUModuleIdentity::observe(module, "release");
    EXPECT_EQ(linked.version, "amdgpu-build-id-00ff0810:release");
    EXPECT_EQ(linked.source, versioned.source);
    writeNote("next");
    EXPECT_NE(AMDGPUModuleIdentity::observe(module, "release").version, linked.version);
    std::filesystem::resize_file(note, sizeof(header) + 5);
    EXPECT_THROW(AMDGPUModuleIdentity::observe(module, "release"), std::runtime_error);
    std::ofstream(module / "version");
    EXPECT_THROW(AMDGPUModuleIdentity::observe(module, "release"), std::runtime_error);
    std::ofstream(module / "version") << "dkms";
    std::filesystem::remove(module / "srcversion");
    EXPECT_THROW(AMDGPUModuleIdentity::observe(module, "release"), std::runtime_error);
}
/** @test Identical hardware geometry and queue policy do not become a physical-UUID catalog allowlist. */
TEST(NativeRuntimePolicyPortability, PhysicalSelectorsDoNotChangeResourceClass){
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}){
        const bool cuda=backend==DeviceType::CUDA;
        const auto selector=cuda?"CUDA_VISIBLE_DEVICES=":"ROCR_VISIBLE_DEVICES=";
        const auto queue=cuda?"CUDA_DEVICE_MAX_CONNECTIONS=8":"ROC_AQL_QUEUE_SIZE=128";
        const std::array first{std::string(selector)+"GPU-uuid-first",std::string(queue)};
        const std::array second{std::string(selector)+"GPU-uuid-second",std::string(queue)};
        EXPECT_EQ(NativeContextRuntimePolicy::resourceIdentity(backend,first),NativeContextRuntimePolicy::resourceIdentity(backend,second));
        auto changed=second;changed[1]=cuda?"CUDA_DEVICE_MAX_CONNECTIONS=16":"ROC_AQL_QUEUE_SIZE=256";
        EXPECT_NE(NativeContextRuntimePolicy::resourceIdentity(backend,first),NativeContextRuntimePolicy::resourceIdentity(backend,changed));
    }
}
/** @test Changed live UUID selection rejects even when portable resource policy stays equal. */
TEST(NativeRuntimePolicyPortability, LiveSelectorChangeStillRejects){
    for(const auto& [backend,name]:std::array{std::pair{DeviceType::CUDA,"CUDA_VISIBLE_DEVICES"},std::pair{DeviceType::ROCm,"ROCR_VISIBLE_DEVICES"}}){
        const auto& original=NativeContextRuntimePolicy::original(backend);const auto resource=original.identity(),live=original.liveIdentity();
        const auto* previous=std::getenv(name);const std::optional<std::string> saved=previous?std::optional<std::string>(previous):std::nullopt;
        const auto changed=saved.value_or("")+"-different-physical-UUID";
        ASSERT_EQ(::setenv(name,changed.c_str(),1),0);
        EXPECT_THROW((void)original.identity(),std::logic_error);
        EXPECT_THROW((void)original.liveIdentity(),std::logic_error);
        if(saved)ASSERT_EQ(::setenv(name,saved->c_str(),1),0);else ASSERT_EQ(::unsetenv(name),0);
        EXPECT_EQ(original.identity(),resource);EXPECT_EQ(original.liveIdentity(),live);
    }
}
}
