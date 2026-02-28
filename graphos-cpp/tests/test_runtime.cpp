#include <gtest/gtest.h>
#include "graphos/kernel/runtime.hpp"

using namespace graphos;

// Runtime tests that don't need real models
TEST(KernelRuntime, DeviceFallback) {
    // If NPU not available, should fall back to CPU without crashing
    EXPECT_NO_THROW({
        KernelRuntime runtime("NPU");
        // May be "NPU" or "CPU" depending on hardware
        EXPECT_FALSE(runtime.device().empty());
    });
}

TEST(KernelRuntime, EmptyProgramList) {
    KernelRuntime runtime("CPU");
    auto progs = runtime.programs();
    EXPECT_TRUE(progs.empty());
}

TEST(KernelRuntime, LoadNonexistent) {
    KernelRuntime runtime("CPU");
    ProgramSpec spec{"test", "nonexistent.onnx", {1, 64}, {1, 3}, ""};
    EXPECT_THROW(runtime.load(spec), std::exception);
}

TEST(KernelRuntime, HealthEmpty) {
    KernelRuntime runtime("CPU");
    auto h = runtime.health();
    EXPECT_TRUE(h.programs.empty());
    EXPECT_EQ(h.exec_count, 0u);
    EXPECT_EQ(h.errors, 0u);
    EXPECT_TRUE(h.healthy);
}

TEST(KernelRuntime, UnloadNonexistent) {
    KernelRuntime runtime("CPU");
    EXPECT_THROW(runtime.unload("nothing"), KernelError);
}

TEST(KernelRuntime, HasProgram) {
    KernelRuntime runtime("CPU");
    EXPECT_FALSE(runtime.has("classifier"));
}

TEST(KernelRuntime, AvailableDevices) {
    KernelRuntime runtime("CPU");
    auto devs = runtime.available_devices();
    // CPU should always be available
    bool has_cpu = false;
    for (const auto& d : devs) {
        if (d == "CPU") has_cpu = true;
    }
    EXPECT_TRUE(has_cpu);
}
