# GraphOS — RouterGraph Packet Classifier

## Project Overview
Dataflow microkernel MVP: ONNX graph on Intel NPU classifies network packets (HTTP/DNS/OTHER).
Proves that NPU tensor ops can replace traditional C switch-based packet classification.

## Hardware
- Intel Core Ultra 7 155H (Meteor Lake), NPU 3720 (11 TOPS)
- Windows 11 userspace prototype via OpenVINO NPU plugin

## Conventions
- Python 3.10+
- Run modules with `python -m <module>` (e.g., `python -m router.train`)
- Tests via `pytest tests/`
- ONNX models go in `models/` directory
- Tensor format: 64-byte raw packet normalized to FP32 [0,1], shape (1, 64)
- 3 classes: TCP_HTTP (0), UDP_DNS (1), OTHER (2)
- NPU requires static shapes — no dynamic axes in ONNX export
- Model uses raw logits (no softmax) + argmax for speed

## Key Commands
```bash
python setup_npu.py              # Verify NPU
python -m router.train           # Train model
python -m router.export_onnx     # Export ONNX (batch=1)
python -m router.export_onnx --batch-size 64   # Export batched ONNX
python -m runtime.benchmark      # Benchmark all modes
python -m runtime.benchmark --batch-size 64 --n-packets 10000  # Custom benchmark
python -m dataflow.demo          # Dataflow pipeline demo
python -m dataflow.demo --batch-size 64 --n-packets 1000      # Custom demo
python -m capture.cli --pcap FILE              # Classify packets from pcap file
python -m capture.cli --iface Ethernet         # Live capture (needs Npcap)
python -m capture.cli --pcap FILE --batch-size 64  # Custom batch size
python -m capture.cli --pcap FILE --route                    # Route with count action
python -m capture.cli --pcap FILE --route --action log       # Route + log packets
python -m capture.cli --pcap FILE --route --action pcap      # Route + split to per-class pcaps
python -m kernel.programs.export_route_table --batch-size 64  # Export route table ONNX
python -m kernel                                               # Kernel demo
python -m kernel.programs.export_composed --batch-size 64      # Export composed router ONNX
python -m kernel.repl                                           # GraphOS interactive shell
pytest tests/                    # Run tests
pytest tests/test_dataflow.py -v # Dataflow framework tests
pytest tests/test_capture.py -v  # Capture framework tests
pytest tests/test_routing.py -v  # Routing framework tests
pytest tests/test_kernel.py -v   # Kernel runtime tests
pytest tests/test_route_table.py -v  # Route table tests
pytest tests/test_kernel_loop.py -v  # Kernel loop tests
pytest tests/test_compose.py -v      # Composition framework tests
pytest tests/test_composed_router.py -v  # Composed router tests
pytest tests/test_repl.py -v         # REPL shell tests

# C++ Build & Test (graphos-cpp/)
cd graphos-cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release   # Configure
cmake --build . -j                     # Build
ctest --output-on-failure              # Run tests
./graphos demo --model-dir ../../models --batch-size 64   # Demo
./graphos shell --model-dir ../../models                   # REPL
./graphos bench --model-dir ../../models --n-packets 10000 # Benchmark
./graphos_bench                        # Microbenchmarks (channel, tensor, argmax)

# DPDK kernel-bypass (Linux only, -DGRAPHOS_ENABLE_DPDK=ON)
./graphos dpdk --port 0 --model-dir ../../models          # Capture + classify on DPDK port 0
./graphos dpdk --port 0 --burst-size 64 --log             # Custom burst size + logging
./graphos dpdk --port 0 -- -l 0-3 -n 4                    # Forward EAL args after --
```

## C++ Architecture (graphos-cpp/)
- C++20, CMake 3.20+, OpenVINO C++ API
- Lock-free SPSC ring buffer (cache-line padded, power-of-2 capacity)
- Zero-copy inference: float* wrapped as ov::Tensor (no memcpy)
- AVX2 SIMD normalization: 8 bytes → 8 floats per instruction
- jthread-per-node with stop_token (no GIL, true parallelism)
- Pimpl pattern hides OpenVINO headers from public API
- Optional: DPDK kernel-bypass capture (Linux, -DGRAPHOS_ENABLE_DPDK=ON)
- Optional: libpcap/Npcap capture (dev mode)
