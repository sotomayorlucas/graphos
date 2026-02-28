"""Allow running as python -m dataflow.demo."""
from dataflow.demo import run_demo

if __name__ == "__main__":
    import argparse
    from core.constants import DEFAULT_BATCH_SIZE

    parser = argparse.ArgumentParser(description="Dataflow pipeline demo")
    parser.add_argument("--n-packets", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    run_demo(n_packets=args.n_packets, batch_size=args.batch_size)
