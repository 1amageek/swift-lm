"""Export Swift-authored LMIR graphs through Core AI Torch."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from .bundle import validate_language_bundle, write_language_bundle_metadata
from .document import ExportDocument
from .errors import ExportError
from .lowering import TorchGraphLowerer
from .program import export_torch_module
from .weights import SafetensorWeightStore


def export_ir_language_model(
    document: ExportDocument,
    model_id: str,
    output_dir: Path,
    *,
    compute_precision: str = "float16",
    overwrite: bool = False,
) -> Path:
    """Compile the Swift graph and its explicit parameter bindings to Core AI."""
    try:
        import torch
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ExportError(
            "missing_coreai_dependencies",
            "Install the pinned Python dependencies from python/pyproject.toml",
        ) from error

    model_directory = _resolve_model_directory(model_id, snapshot_download)
    config = _load_config(model_directory)
    _validate_source_contract(document, config)

    dtype = _resolve_dtype(torch, compute_precision)
    weights = SafetensorWeightStore(model_directory, torch, dtype)
    model = TorchGraphLowerer(document, weights, torch).make_model().eval()

    name = document.metadata["name"]
    bundle_path = output_dir / name
    asset_path = bundle_path / f"{name}.aimodel"
    if asset_path.exists():
        if not overwrite:
            raise ExportError("output_exists", f"Output already exists: {asset_path}")
        shutil.rmtree(asset_path)
    bundle_path.mkdir(parents=True, exist_ok=True)

    context_length = document.metadata["maxContextLength"]
    reference_inputs, dynamic_shapes = _build_reference_inputs(document, torch)
    if document.execution == "stateful":
        _export_stateful_model(
            document,
            model,
            reference_inputs,
            dynamic_shapes,
            asset_path,
        )
    else:
        export_torch_module(
            model,
            reference_inputs,
            asset_path,
            input_names=[tensor["name"] for tensor in document.function["inputs"]],
            output_names=[tensor["name"] for tensor in document.function["outputs"]],
            dynamic_shapes=dynamic_shapes,
        )
    _copy_tokenizer_assets(model_directory, bundle_path / "tokenizer")

    contract_path = bundle_path / "swiftlm-program.json"
    try:
        contract_path.write_text(
            json.dumps(document.raw, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise ExportError("bundle_contract_write_failed", str(contract_path)) from error

    write_language_bundle_metadata(
        bundle_path,
        name=name,
        model_id=model_id,
        vocab_size=document.metadata["vocabSize"],
        max_context_length=context_length,
    )
    validate_language_bundle(
        bundle_path,
        expected_name=name,
        expected_model_id=model_id,
        expected_vocab_size=document.metadata["vocabSize"],
        expected_max_context_length=context_length,
    )
    return bundle_path


def _build_reference_inputs(
    document: ExportDocument,
    torch: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    symbols: dict[str, Any] = {}

    def tensor_shape(tensor: dict[str, Any]) -> tuple[list[int], dict[int, Any]]:
        shape: list[int] = []
        dynamic: dict[int, Any] = {}
        for axis, dimension in enumerate(tensor["dimensions"]):
            if dimension["kind"] == "fixed":
                shape.append(dimension["size"])
                continue
            minimum = dimension["minimum"]
            maximum = dimension["maximum"]
            reference = min(maximum, max(minimum, 8))
            shape.append(reference)
            if minimum != maximum:
                symbol = dimension["symbol"]
                dynamic[axis] = symbols.setdefault(
                    symbol,
                    torch.export.Dim(symbol, min=minimum, max=maximum),
                )
        return shape, dynamic

    def make_tensor(tensor: dict[str, Any], *, position: bool = False) -> tuple[Any, dict[int, Any]]:
        shape, dynamic = tensor_shape(tensor)
        dtype = _torch_dtype(torch, tensor["dataType"])
        if position:
            if len(shape) != 2 or shape[0] != 1:
                raise ExportError("invalid_contract", "position_ids must have shape [1, sequence]")
            value = torch.arange(shape[1], dtype=dtype).unsqueeze(0)
        elif tensor["name"] == "input_ids":
            value = torch.ones(tuple(shape), dtype=dtype)
        else:
            value = torch.zeros(tuple(shape), dtype=dtype)
        return value, dynamic

    inputs: dict[str, Any] = {}
    dynamic_shapes: dict[str, Any] = {}
    for tensor in document.function["inputs"]:
        value, dynamic = make_tensor(tensor, position=tensor["name"] == "position_ids")
        inputs[tensor["name"]] = value
        dynamic_shapes[tensor["name"]] = dynamic

    if document.execution == "stateful":
        state_values = []
        state_shapes = []
        for tensor in document.function["states"]:
            value, dynamic = make_tensor(tensor)
            state_values.append(value)
            state_shapes.append(dynamic)
        inputs["states"] = tuple(state_values)
        dynamic_shapes["states"] = tuple(state_shapes)
    return inputs, dynamic_shapes


def _export_stateful_model(
    document: ExportDocument,
    model: Any,
    reference_inputs: dict[str, Any],
    dynamic_shapes: dict[str, Any],
    asset_path: Path,
) -> None:
    try:
        from coreai_models.export.macos import export_to_coreai
    except ImportError as error:
        raise ExportError(
            "missing_coreai_dependencies",
            "Stateful export requires Apple's coreai-models package",
        ) from error

    try:
        program = export_to_coreai(
            model,
            reference_inputs,
            dynamic_shapes=dynamic_shapes,
            input_names=tuple(tensor["name"] for tensor in document.function["inputs"]),
            output_names=tuple(tensor["name"] for tensor in document.function["outputs"]),
            state_names=tuple(tensor["name"] for tensor in document.function["states"]),
        )
        program.optimize()
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        program.save_asset(asset_path)
    except Exception as error:
        raise ExportError("coreai_program_export_failed", str(error)) from error


def _torch_dtype(torch: Any, data_type: str) -> Any:
    try:
        return {
            "int32": torch.int32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[data_type]
    except KeyError as error:
        raise ExportError("invalid_contract", f"Unsupported tensor data type {data_type}") from error


def _resolve_model_directory(model_id: str, snapshot_download: Any) -> Path:
    candidate = Path(model_id).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    try:
        return Path(snapshot_download(repo_id=model_id))
    except Exception as error:
        raise ExportError("hf_bundle_load_failed", f"Could not load {model_id}") from error


def _load_config(model_directory: Path) -> dict[str, Any]:
    path = model_directory / "config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExportError("hf_config_load_failed", str(path)) from error
    if not isinstance(config, dict):
        raise ExportError("hf_config_load_failed", f"Invalid config root: {path}")
    return config


def _copy_tokenizer_assets(model_directory: Path, output_directory: Path) -> None:
    asset_names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "sentencepiece.bpe.model",
        "spiece.model",
    )
    tokenizer_json = model_directory / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise ExportError(
            "tokenizer_export_failed",
            f"HF bundle is missing tokenizer.json: {tokenizer_json}",
        )
    try:
        if output_directory.exists():
            shutil.rmtree(output_directory)
        output_directory.mkdir(parents=True)
        for name in asset_names:
            source = model_directory / name
            if source.is_file():
                shutil.copy2(source, output_directory / name)
    except OSError as error:
        raise ExportError("tokenizer_export_failed", str(error)) from error


def _validate_source_contract(
    document: ExportDocument,
    config: dict[str, Any],
) -> None:
    actual_model_type = config.get("model_type")
    if actual_model_type != document.model_type:
        raise ExportError(
            "model_type_mismatch",
            f"document declares {document.model_type!r}, HF config declares {actual_model_type!r}",
        )
    language_config = config.get("text_config", config)
    if not isinstance(language_config, dict):
        raise ExportError(
            "invalid_source_config",
            "HF text_config must be an object",
        )
    actual_vocab_size = language_config.get("vocab_size")
    if actual_vocab_size != document.metadata["vocabSize"]:
        raise ExportError(
            "vocab_size_mismatch",
            f"document declares {document.metadata['vocabSize']}, HF config declares {actual_vocab_size!r}",
        )


def _resolve_dtype(torch: Any, value: str) -> Any:
    try:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[value]
    except KeyError as error:
        raise ExportError(
            "invalid_compute_precision",
            "compute_precision must be float16, bfloat16, or float32",
        ) from error


def default_output_name(model_id: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id.rsplit("/", 1)[-1]).strip("_")
    if not name:
        raise ExportError("invalid_output", "model_id does not provide an output name")
    return name
