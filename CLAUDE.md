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
pytest tests/                    # Run tests
pytest tests/test_dataflow.py -v # Dataflow framework tests
pytest tests/test_capture.py -v  # Capture framework tests
pytest tests/test_routing.py -v  # Routing framework tests
```
