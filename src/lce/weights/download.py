"""lce.weights.download — CLI weight manager."""
from __future__ import annotations
import argparse
from lce.models.registry import available, ensure, model_path, weights_dir


def main() -> None:
    ap = argparse.ArgumentParser(prog="lce-weights")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("list",     help="List available models")
    sub.add_parser("download", help="Download all models")
    sp = sub.add_parser("get", help="Download one model by id")
    sp.add_argument("id")

    args = ap.parse_args()
    if args.cmd == "list":
        print(f"Weights dir: {weights_dir()}\n")
        for m in available():
            local = model_path(m["id"])
            status = "✓" if local else " "
            print(f"  [{status}] {m['id']:30s}  {m['license']:15s}  {m.get('size_bytes', 0) // 1_000_000:>4} MB  {m['description']}")
    elif args.cmd == "download":
        for m in available():
            ensure(m["id"])
        print("All models ready.")
    elif args.cmd == "get":
        path = ensure(args.id)
        print(f"Ready: {path}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
