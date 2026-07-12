"""Command-line interface for swiftlm-coreai."""

from __future__ import annotations

import argparse
from pathlib import Path

from .errors import ExportError
from .exporter import export_model, validate_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swiftlm-coreai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("document", type=Path)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("document", type=Path)
    export_parser.add_argument("model_id")
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument("--overwrite", action="store_true")
    export_parser.add_argument(
        "--stateful",
        action="store_true",
        help="Export LFM2/LFM2 MoE with mutable KV and convolution states",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "validate":
            document = validate_document(args.document)
            print(
                f"valid format={document.raw['formatVersion']} "
                f"model_type={document.model_type} target={document.target}"
            )
            return 0

        result = export_model(
            args.document,
            args.model_id,
            args.output_dir,
            overwrite=args.overwrite,
            stateful=args.stateful,
        )
        print(result)
        return 0
    except ExportError as error:
        print(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
