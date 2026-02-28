cmake_minimum_required(VERSION 3.20)
project(graphos VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Build type default ──
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── Aggressive optimization flags ──
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -flto")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG -march=native")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /DNDEBUG /GL /arch:AVX2")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG")
endif()

# ── OpenVINO (required) ──
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# ── libpcap (optional — for capture sources) ──
option(GRAPHOS_ENABLE_PCAP "Enable pcap capture sources" ON)
if(GRAPHOS_ENABLE_PCAP)
    if(WIN32)
        # Npcap SDK — user sets NPCAP_SDK_DIR
        if(DEFINED ENV{NPCAP_SDK_DIR})
            set(NPCAP_SDK_DIR "$ENV{NPCAP_SDK_DIR}")
        elseif(NOT NPCAP_SDK_DIR)
            set(NPCAP_SDK_DIR "C:/npcap-sdk")
        endif()
        if(EXISTS "${NPCAP_SDK_DIR}/Include")
            set(PCAP_INCLUDE_DIRS "${NPCAP_SDK_DIR}/Include")
            if(CMAKE_SIZEOF_VOID_P EQUAL 8)
                set(PCAP_LIBRARIES "${NPCAP_SDK_DIR}/Lib/x64/wpcap.lib")
            else()
                set(PCAP_LIBRARIES "${NPCAP_SDK_DIR}/Lib/wpcap.lib")
            endif()
            set(PCAP_FOUND TRUE)
        endif()
    else()
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(PCAP QUIET libpcap)
        endif()
        if(NOT PCAP_FOUND)
            find_library(PCAP_LIBRARIES NAMES pcap)
            find_path(PCAP_INCLUDE_DIRS NAMES pcap/pcap.h pcap.h)
            if(PCAP_LIBRARIES AND PCAP_INCLUDE_DIRS)
                set(PCAP_FOUND TRUE)
            endif()
        endif()
    endif()
endif()

# ── DPDK (optional, Linux only) ──
option(GRAPHOS_ENABLE_DPDK "Enable DPDK capture source" OFF)
if(GRAPHOS_ENABLE_DPDK)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(DPDK REQUIRED libdpdk)
endif()

# ── SIMD detection ──
include(CheckCXXCompilerFlag)
if(MSVC)
    check_cxx_compiler_flag("/arch:AVX2" HAS_AVX2)
else()
    check_cxx_compiler_flag("-mavx2" HAS_AVX2)
endif()

# ── Main library ──
set(LIB_SOURCES
    src/core/types.cpp
    src/dataflow/graph.cpp
    src/dataflow/scheduler.cpp
    src/nodes/batch_node.cpp
    src/nodes/tensor_node.cpp
    src/nodes/infer_node.cpp
    src/nodes/sink_node.cpp
    src/nodes/tee_node.cpp
    src/nodes/router_sink.cpp
    src/nodes/adapter_node.cpp
    src/kernel/program.cpp
    src/kernel/runtime.cpp
    src/kernel/health.cpp
    src/kernel/compose.cpp
    src/kernel/loop.cpp
    src/actions/counter.cpp
    src/actions/log_action.cpp
    src/shell/repl.cpp
)

if(PCAP_FOUND)
    list(APPEND LIB_SOURCES src/capture/pcap_source.cpp)
    list(APPEND LIB_SOURCES src/actions/pcap_writer.cpp)
endif()

if(GRAPHOS_ENABLE_DPDK)
    list(APPEND LIB_SOURCES src/capture/dpdk_source.cpp)
endif()

add_library(graphos_lib STATIC ${LIB_SOURCES})
target_include_directories(graphos_lib PUBLIC include/)
target_link_libraries(graphos_lib PUBLIC openvino::runtime)

# SIMD flags
if(HAS_AVX2)
    target_compile_definitions(graphos_lib PUBLIC GRAPHOS_HAS_AVX2)
    if(MSVC)
        target_compile_options(graphos_lib PRIVATE /arch:AVX2)
    else()
        target_compile_options(graphos_lib PRIVATE -mavx2)
    endif()
endif()

# pcap
if(PCAP_FOUND)
    target_compile_definitions(graphos_lib PUBLIC GRAPHOS_HAS_PCAP)
    target_include_directories(graphos_lib PRIVATE ${PCAP_INCLUDE_DIRS})
    target_link_libraries(graphos_lib PRIVATE ${PCAP_LIBRARIES})
endif()

# DPDK
if(GRAPHOS_ENABLE_DPDK)
    target_compile_definitions(graphos_lib PUBLIC GRAPHOS_ENABLE_DPDK)
    target_include_directories(graphos_lib PRIVATE ${DPDK_INCLUDE_DIRS})
    target_link_libraries(graphos_lib PRIVATE ${DPDK_LIBRARIES})
endif()

# ── Main executable ──
add_executable(graphos src/main.cpp)
target_link_libraries(graphos PRIVATE graphos_lib)

# ── Tests (FetchContent GoogleTest) ──
option(GRAPHOS_BUILD_TESTS "Build unit tests" ON)
if(GRAPHOS_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    set(TEST_SOURCES
        tests/test_constants.cpp
        tests/test_channel.cpp
        tests/test_graph.cpp
        tests/test_scheduler.cpp
        tests/test_nodes.cpp
        tests/test_actions.cpp
        tests/test_compose.cpp
        tests/test_program.cpp
        tests/test_runtime.cpp
        tests/test_repl.cpp
    )

    if(PCAP_FOUND)
        list(APPEND TEST_SOURCES tests/test_capture.cpp)
    endif()

    add_executable(graphos_tests ${TEST_SOURCES})
    target_link_libraries(graphos_tests PRIVATE graphos_lib GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(graphos_tests)
endif()

# ── Benchmarks (FetchContent Google Benchmark) ──
option(GRAPHOS_BUILD_BENCHMARKS "Build benchmarks" ON)
if(GRAPHOS_BUILD_BENCHMARKS)
    include(FetchContent)
    FetchContent_Declare(
        benchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)

    add_executable(graphos_bench bench/bench_channel.cpp)
    target_link_libraries(graphos_bench PRIVATE graphos_lib benchmark::benchmark)
endif()

# ── Install ──
install(TARGETS graphos DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
