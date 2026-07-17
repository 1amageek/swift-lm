"""Apple Core AI language-model bundle metadata and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .errors import ExportError

METADATA_VERSION = "0.2"


def write_language_bundle_metadata(
    bundle_path: Path,
    *,
    name: str,
    model_id: str,
    vocab_size: int,
    max_context_length: int,
    asset_name: str | None = None,
    contract_name: str = "swiftlm-program.json",
    compression: str | None = None,
    function_map: dict[str, list[str]] | None = None,
) -> Path:
    """Write the metadata.json contract consumed by Apple's language runtime."""
    _validate_positive_int("vocab_size", vocab_size)
    _validate_positive_int("max_context_length", max_context_length)
    if not name:
        raise ExportError("invalid_bundle_metadata", "name must not be empty")
    if not model_id:
        raise ExportError("invalid_bundle_metadata", "model_id must not be empty")

    resolved_asset_name = asset_name or f"{name}.aimodel"
    contract_path = bundle_path / contract_name
    try:
        contract_sha256 = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    except OSError as error:
        raise ExportError(
            "invalid_bundle_metadata",
            f"Could not hash contract {contract_path}: {error}",
        ) from error
    metadata: dict[str, Any] = {
        "metadata_version": METADATA_VERSION,
        "kind": "llm",
        "name": name,
        "assets": {
            "main": resolved_asset_name,
            "contract": contract_name,
            "contract_sha256": contract_sha256,
        },
        "language": {
            "tokenizer": "tokenizer",
            "vocab_size": vocab_size,
            "max_context_length": max_context_length,
            "embedded_tokenizer": True,
            "function_map": function_map or {"main": ["main"]},
        },
        "source": {
            "model_definition": "swift_lmir",
            "format_version": 2,
            "hf_model_id": model_id,
        },
        "compression": compression,
    }

    bundle_path.mkdir(parents=True, exist_ok=True)
    metadata_path = bundle_path / "metadata.json"
    temporary_path = metadata_path.with_suffix(".json.tmp")
    try:
        temporary_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(metadata_path)
    except OSError as error:
        raise ExportError(
            "bundle_metadata_write_failed",
            f"Could not write {metadata_path}: {error}",
        ) from error
    return metadata_path


def validate_language_bundle(
    bundle_path: Path,
    *,
    expected_name: str,
    expected_model_id: str,
    expected_vocab_size: int,
    expected_max_context_length: int,
) -> dict[str, Any]:
    """Validate an exported bundle against Apple's metadata_version 0.2 schema."""
    metadata_path = bundle_path / "metadata.json"
    if not bundle_path.is_dir():
        raise ExportError("invalid_bundle", f"Bundle directory does not exist: {bundle_path}")
    if not metadata_path.is_file():
        raise ExportError("invalid_bundle", f"Missing bundle metadata: {metadata_path}")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExportError("invalid_bundle", f"Could not read {metadata_path}: {error}") from error

    if not isinstance(metadata, dict):
        raise ExportError("invalid_bundle", "metadata.json root must be an object")
    if metadata.get("metadata_version") != METADATA_VERSION:
        raise ExportError(
            "invalid_bundle",
            f"metadata_version must be {METADATA_VERSION!r}",
        )
    if metadata.get("kind") != "llm":
        raise ExportError("invalid_bundle", "metadata.kind must be 'llm'")
    if metadata.get("name") != expected_name:
        raise ExportError(
            "invalid_bundle",
            f"metadata.name must be {expected_name!r}",
        )

    assets = metadata.get("assets")
    if not isinstance(assets, dict) or not isinstance(assets.get("main"), str):
        raise ExportError("invalid_bundle", "metadata.assets.main must be a string")
    asset_path = _resolve_bundle_asset(bundle_path, assets["main"])
    if asset_path.suffix != ".aimodel" or not asset_path.is_dir():
        raise ExportError(
            "invalid_bundle",
            f"metadata.assets.main must reference an existing .aimodel directory: {asset_path}",
        )
    contract_name = assets.get("contract")
    if not isinstance(contract_name, str):
        raise ExportError("invalid_bundle", "metadata.assets.contract must be a string")
    contract_path = _resolve_bundle_asset(bundle_path, contract_name)
    if not contract_path.is_file():
        raise ExportError("invalid_bundle", f"Missing Swift LMIR contract: {contract_path}")
    expected_contract_hash = assets.get("contract_sha256")
    if not isinstance(expected_contract_hash, str) or len(expected_contract_hash) != 64:
        raise ExportError("invalid_bundle", "metadata.assets.contract_sha256 is invalid")
    try:
        contract_bytes = contract_path.read_bytes()
    except OSError as error:
        raise ExportError("invalid_bundle", f"Could not hash {contract_path}") from error
    actual_contract_hash = hashlib.sha256(contract_bytes).hexdigest()
    if actual_contract_hash != expected_contract_hash:
        raise ExportError("invalid_bundle", "Swift LMIR contract hash does not match metadata")
    try:
        contract = json.loads(contract_bytes)
    except json.JSONDecodeError as error:
        raise ExportError("invalid_bundle", f"Invalid Swift LMIR contract: {contract_path}") from error
    if not isinstance(contract, dict) or contract.get("formatVersion") != 2:
        raise ExportError("invalid_bundle", "Swift LMIR contract must use format version 2")
    program = contract.get("program")
    if not isinstance(program, dict) or program.get("source") != "swift_lmir":
        raise ExportError("invalid_bundle", "Bundle contract must originate from swift_lmir")

    source = metadata.get("source")
    if not isinstance(source, dict) or source.get("model_definition") != "swift_lmir":
        raise ExportError("invalid_bundle", "metadata.source.model_definition must be swift_lmir")
    if source.get("format_version") != 2:
        raise ExportError("invalid_bundle", "metadata.source.format_version must be 2")
    if source.get("hf_model_id") != expected_model_id:
        raise ExportError(
            "invalid_bundle",
            f"metadata.source.hf_model_id must be {expected_model_id!r}",
        )

    language = metadata.get("language")
    if not isinstance(language, dict):
        raise ExportError("invalid_bundle", "metadata.language must be an object")
    tokenizer_name = language.get("tokenizer")
    if not isinstance(tokenizer_name, str) or not tokenizer_name:
        raise ExportError("invalid_bundle", "language.tokenizer must be a relative path")
    if language.get("vocab_size") != expected_vocab_size:
        raise ExportError(
            "invalid_bundle",
            f"language.vocab_size must be {expected_vocab_size}",
        )
    if language.get("max_context_length") != expected_max_context_length:
        raise ExportError(
            "invalid_bundle",
            f"language.max_context_length must be {expected_max_context_length}",
        )

    embedded_tokenizer = language.get("embedded_tokenizer", True)
    if not isinstance(embedded_tokenizer, bool):
        raise ExportError("invalid_bundle", "language.embedded_tokenizer must be a boolean")
    if embedded_tokenizer:
        tokenizer_path = _resolve_bundle_asset(bundle_path, tokenizer_name)
        tokenizer_json = tokenizer_path / "tokenizer.json"
        if not tokenizer_path.is_dir() or not tokenizer_json.is_file():
            raise ExportError(
                "invalid_bundle",
                f"Embedded tokenizer is missing: {tokenizer_json}",
            )

    function_map = language.get("function_map")
    if function_map is not None:
        _validate_function_map(function_map)
    return metadata


def _resolve_bundle_asset(bundle_path: Path, relative_path: str) -> Path:
    asset_path = (bundle_path / relative_path).resolve()
    bundle_root = bundle_path.resolve()
    if not asset_path.is_relative_to(bundle_root):
        raise ExportError("invalid_bundle", "bundle-relative paths must stay inside the bundle")
    return asset_path


def _validate_function_map(function_map: Any) -> None:
    if not isinstance(function_map, dict):
        raise ExportError("invalid_bundle", "language.function_map must be an object")
    for role, names in function_map.items():
        if not isinstance(role, str) or not isinstance(names, list):
            raise ExportError("invalid_bundle", "language.function_map values must be string arrays")
        if not names or not all(isinstance(name, str) and name for name in names):
            raise ExportError("invalid_bundle", "language.function_map entries must not be empty")


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExportError("invalid_bundle_metadata", f"{name} must be positive")
