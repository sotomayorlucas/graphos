#include <gtest/gtest.h>
#include "graphos/kernel/program.hpp"

using namespace graphos;

TEST(ProgramSpec, Construction) {
    ProgramSpec spec{
        "classifier",
        "models/classifier.onnx",
        {64, 64},
        {64, 3},
        "Packet classifier"
    };
    EXPECT_EQ(spec.name, "classifier");
    EXPECT_EQ(spec.input_shape.size(), 2u);
    EXPECT_EQ(spec.input_shape[0], 64);
    EXPECT_EQ(spec.input_shape[1], 64);
    EXPECT_EQ(spec.output_shape[0], 64);
    EXPECT_EQ(spec.output_shape[1], 3);
}

// NOTE: Program::execute tests require OpenVINO + a real ONNX model.
// These are integration tests — run only when models/ dir has artifacts.
// Use: ctest -R test_program -L integration
