#include <gtest/gtest.h>
#include "graphos/kernel/compose.hpp"
#include <cstring>

using namespace graphos;

TEST(ConcatAdapter, ConcatenatesRows) {
    auto adapter = make_concat_adapter(2, 3, 2); // batch=2, left=3, right=2
    // left: [[1,2,3],[4,5,6]], right: [[7,8],[9,10]]
    float left[] = {1, 2, 3, 4, 5, 6};
    float right[] = {7, 8, 9, 10};

    auto result = adapter.execute_concat(left, 6, right, 4);
    // Expected: [[1,2,3,7,8],[4,5,6,9,10]]
    ASSERT_EQ(result.size(), 10u);
    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[3], 7.0f);
    EXPECT_FLOAT_EQ(result[4], 8.0f);
    EXPECT_FLOAT_EQ(result[5], 4.0f);
    EXPECT_FLOAT_EQ(result[8], 9.0f);
    EXPECT_FLOAT_EQ(result[9], 10.0f);
}

TEST(PadAdapter, ZeroPadsRows) {
    auto adapter = make_pad_adapter(2, 3, 5); // batch=2, in=3, out=5
    float input[] = {1, 2, 3, 4, 5, 6};

    auto result = adapter.execute_single(input, 6);
    ASSERT_EQ(result.size(), 10u);
    // Row 0: [1,2,3,0,0]
    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[2], 3.0f);
    EXPECT_FLOAT_EQ(result[3], 0.0f);
    EXPECT_FLOAT_EQ(result[4], 0.0f);
    // Row 1: [4,5,6,0,0]
    EXPECT_FLOAT_EQ(result[5], 4.0f);
    EXPECT_FLOAT_EQ(result[7], 6.0f);
    EXPECT_FLOAT_EQ(result[8], 0.0f);
}

TEST(PadAdapter, TruncatesRows) {
    auto adapter = make_pad_adapter(1, 5, 3); // truncate to 3
    float input[] = {1, 2, 3, 4, 5};

    auto result = adapter.execute_single(input, 5);
    ASSERT_EQ(result.size(), 3u);
    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[1], 2.0f);
    EXPECT_FLOAT_EQ(result[2], 3.0f);
}

TEST(ProgramPipeline, FluentAPI) {
    ProgramPipeline pipeline;
    ProgramSpec spec1{"prog1", "a.onnx", {64,64}, {64,3}, ""};
    auto adapter = make_concat_adapter(64, 64, 3);
    ProgramSpec spec2{"prog2", "b.onnx", {64,67}, {64,4}, ""};

    pipeline.add_program("prog1", spec1)
            .add_adapter(std::move(adapter))
            .add_program("prog2", spec2)
            .with_raw_passthrough("raw");

    EXPECT_EQ(pipeline.stages().size(), 3u);
    EXPECT_EQ(pipeline.stages()[0].kind, ProgramPipeline::StageKind::Program);
    EXPECT_EQ(pipeline.stages()[1].kind, ProgramPipeline::StageKind::Adapter);
    EXPECT_EQ(pipeline.stages()[2].kind, ProgramPipeline::StageKind::Program);

    auto desc = pipeline.describe();
    EXPECT_NE(desc.find("prog1"), std::string::npos);
    EXPECT_NE(desc.find("concat"), std::string::npos);
    EXPECT_NE(desc.find("prog2"), std::string::npos);
    EXPECT_NE(desc.find("raw"), std::string::npos);
}
