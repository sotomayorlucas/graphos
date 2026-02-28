"""Verify Intel NPU availability via OpenVINO."""

import sys


def check_npu():
    try:
        import openvino as ov
    except ImportError:
        print("ERROR: OpenVINO not installed.")
        print("Install with: pip install openvino>=2024.0")
        print("If NPU plugin is missing, download the Archive distribution from:")
        print("  https://docs.openvino.ai/2024/get-started/install-openvino.html")
        sys.exit(1)

    core = ov.Core()
    devices = core.available_devices
    print(f"Available devices: {devices}")

    if "NPU" not in devices:
        print(f"\nWARNING: NPU not found among available devices: {devices}")
        print("Possible causes:")
        print("  1. Intel NPU driver not installed")
        print("  2. OpenVINO NPU plugin not available (try Archive distribution)")
        print("  3. Hardware does not have an NPU")
        print("\nThe benchmark will fall back to CPU inference.")
        return False

    full_name = core.get_property("NPU", "FULL_DEVICE_NAME")
    print(f"\nNPU found: {full_name}")
    return True


if __name__ == "__main__":
    success = check_npu()
    sys.exit(0 if success else 1)
