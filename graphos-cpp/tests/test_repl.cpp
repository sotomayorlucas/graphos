#include <gtest/gtest.h>
#include "graphos/shell/repl.hpp"

using namespace graphos;

TEST(GraphOSShell, Construction) {
    KernelRuntime runtime("CPU");
    EXPECT_NO_THROW(GraphOSShell shell(runtime, DEFAULT_BATCH_SIZE));
}

// REPL interactive tests are difficult to unit test.
// Main validation via integration testing.
