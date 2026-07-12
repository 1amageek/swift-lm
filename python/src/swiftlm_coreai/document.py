"""Parser and validator for the Core AI export document emitted by Swift."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import ExportError

SUPPORTED_FORMAT_VERSION = 1
SUPPORTED_TARGETS = {"macos_dynamic", "ios_static"}
SUPPORTED_PRIMITIVES = {
    "attention",
    "layer_scale",
    "linear",
    "mlp",
    "moe",
    "rms_norm",
    "layer_norm",
    "output_head",
    "patch_embedding",
    "per_layer_input",
    "pooling",
    "position_embedding",
    "rope",
    "short_conv",
    "standardize",
    "state_space",
    "token_embedding",
}


class ExportDocument:
    """Validated representation of a versioned Swift export document."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw

    @classmethod
    def load(cls, path: Path) -> "ExportDocument":
        try:
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except FileNotFoundError as error:
            raise ExportError("document_not_found", str(path)) from error
        except json.JSONDecodeError as error:
            raise ExportError("invalid_json", f"{path}: {error}") from error
        if not isinstance(raw, dict):
            raise ExportError("invalid_document", "root must be a JSON object")
        document = cls(raw)
        document.validate()
        return document

    @property
    def metadata(self) -> dict[str, Any]:
        value = self.raw["metadata"]
        assert isinstance(value, dict)
        return value

    @property
    def target(self) -> str:
        return self.metadata["target"]

    @property
    def model_type(self) -> str:
        return self.metadata["modelType"]

    def validate(self) -> None:
        version = self.raw.get("formatVersion")
        if version != SUPPORTED_FORMAT_VERSION:
            raise ExportError(
                "unsupported_format_version",
                f"expected {SUPPORTED_FORMAT_VERSION}, got {version!r}",
            )

        metadata = self.raw.get("metadata")
        if not isinstance(metadata, dict):
            raise ExportError("invalid_metadata", "metadata must be an object")
        for key in ("name", "modelType", "target", "maxContextLength", "vocabSize"):
            if key not in metadata:
                raise ExportError("invalid_metadata", f"missing metadata.{key}")
        if metadata["target"] not in SUPPORTED_TARGETS:
            raise ExportError("unsupported_target", str(metadata["target"]))
        if not self._is_integer(metadata["maxContextLength"]) or metadata["maxContextLength"] <= 0:
            raise ExportError("invalid_metadata", "metadata.maxContextLength must be positive")
        if not self._is_integer(metadata["vocabSize"]) or metadata["vocabSize"] <= 0:
            raise ExportError("invalid_metadata", "metadata.vocabSize must be positive")

        root = self.raw.get("rootRegion")
        if not isinstance(root, dict):
            raise ExportError("invalid_graph", "rootRegion must be an object")
        self._validate_region(root, "root")

    def _validate_region(self, region: dict[str, Any], path: str) -> None:
        parameters = self._require_ids(region, "parameters", path)
        operations = region.get("operations")
        results = self._require_ids(region, "results", path)
        if not isinstance(operations, list):
            raise ExportError("invalid_graph", f"{path}.operations must be an array")

        defined = set(parameters)
        for index, operation in enumerate(operations):
            operation_path = f"{path}.operations[{index}]"
            if not isinstance(operation, dict):
                raise ExportError("invalid_graph", f"{operation_path} must be an object")
            operands = self._require_ids(operation, "operands", operation_path)
            for value in operands:
                if value not in defined:
                    raise ExportError("invalid_graph", f"undefined value {value} at {operation_path}")
            result_ids = self._require_ids(operation, "results", operation_path)
            kind = operation.get("kind")
            if not isinstance(kind, dict):
                raise ExportError("invalid_graph", f"{operation_path}.kind must be an object")
            self._validate_kind(kind, operation_path)
            defined.update(result_ids)

        for value in results:
            if value not in defined:
                raise ExportError("invalid_graph", f"undefined region result {value} at {path}")

    def _validate_kind(self, kind: dict[str, Any], path: str) -> None:
        tag = kind.get("tag")
        if tag == "primitive":
            primitive = kind.get("primitive")
            if not isinstance(primitive, dict):
                raise ExportError("invalid_graph", f"{path}.primitive must be an object")
            opcode = primitive.get("opcode")
            if opcode not in SUPPORTED_PRIMITIVES:
                raise ExportError("unsupported_primitive", str(opcode))
            if "attributes" not in primitive:
                raise ExportError("invalid_graph", f"{path}.primitive.attributes is missing")
            return
        if tag == "residual":
            self._validate_nested_region(kind, "body", path)
            return
        if tag == "parallel":
            branches = kind.get("branches")
            if not isinstance(branches, list):
                raise ExportError("invalid_graph", f"{path}.branches must be an array")
            for index, branch in enumerate(branches):
                if not isinstance(branch, dict):
                    raise ExportError("invalid_graph", f"{path}.branches[{index}] must be an object")
                self._validate_region(branch, f"{path}.branches[{index}]")
            return
        if tag == "repeating":
            count = kind.get("count")
            if not self._is_integer(count) or count < 0:
                raise ExportError("invalid_graph", f"{path}.count must be non-negative")
            self._validate_nested_region(kind, "body", path)
            return
        if tag == "conditional":
            self._validate_nested_region(kind, "then", path)
            self._validate_nested_region(kind, "else", path)
            return
        raise ExportError("invalid_graph", f"unsupported operation tag {tag!r} at {path}")

    def _validate_nested_region(self, kind: dict[str, Any], key: str, path: str) -> None:
        value = kind.get(key)
        if not isinstance(value, dict):
            raise ExportError("invalid_graph", f"{path}.{key} must be an object")
        self._validate_region(value, f"{path}.{key}")

    @staticmethod
    def _require_ids(container: dict[str, Any], key: str, path: str) -> list[int]:
        value = container.get(key)
        if not isinstance(value, list) or not all(ExportDocument._is_integer(item) for item in value):
            raise ExportError("invalid_graph", f"{path}.{key} must be an integer array")
        return value

    @staticmethod
    def _is_integer(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)
