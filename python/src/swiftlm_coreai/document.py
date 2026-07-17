"""Parser and validator for the Core AI export document emitted by Swift."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import ExportError

SUPPORTED_FORMAT_VERSION = 2
SUPPORTED_TARGETS = {"macos_dynamic", "ios_static"}
SUPPORTED_EXECUTION_MODES = {"stateless", "stateful"}
SUPPORTED_DATA_TYPES = {"int32", "float16", "bfloat16", "float32"}
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

    @property
    def program(self) -> dict[str, Any]:
        value = self.raw["program"]
        assert isinstance(value, dict)
        return value

    @property
    def function(self) -> dict[str, Any]:
        functions = self.program["functions"]
        assert isinstance(functions, list)
        value = functions[0]
        assert isinstance(value, dict)
        return value

    @property
    def execution(self) -> str:
        value = self.program["execution"]
        assert isinstance(value, str)
        return value

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

        state_names = self._validate_program()

        root = self.raw.get("rootRegion")
        if not isinstance(root, dict):
            raise ExportError("invalid_graph", "rootRegion must be an object")
        self._validate_region(root, "root", state_names)

    def _validate_program(self) -> set[str]:
        program = self.raw.get("program")
        if not isinstance(program, dict):
            raise ExportError("invalid_program", "program must be an object")
        if program.get("source") != "swift_lmir":
            raise ExportError("invalid_program", "program.source must be swift_lmir")
        execution = program.get("execution")
        if execution not in SUPPORTED_EXECUTION_MODES:
            raise ExportError("invalid_program", f"unsupported execution mode {execution!r}")
        functions = program.get("functions")
        if not isinstance(functions, list) or len(functions) != 1:
            raise ExportError("invalid_program", "program must contain exactly one function")
        function = functions[0]
        if not isinstance(function, dict) or function.get("name") != "main":
            raise ExportError("invalid_program", "program function must be named main")

        names: set[str] = set()
        input_names = self._validate_tensors(function, "inputs", names)
        output_names = self._validate_tensors(function, "outputs", names)
        state_names = self._validate_tensors(function, "states", names)
        if not input_names or not output_names:
            raise ExportError("invalid_program", "main requires inputs and outputs")
        if execution == "stateless" and state_names:
            raise ExportError("invalid_program", "stateless main must not declare states")
        if execution == "stateful" and not state_names:
            raise ExportError("invalid_program", "stateful main must declare states")
        return state_names

    def _validate_tensors(
        self,
        function: dict[str, Any],
        key: str,
        all_names: set[str],
    ) -> set[str]:
        tensors = function.get(key)
        if not isinstance(tensors, list):
            raise ExportError("invalid_program", f"main.{key} must be an array")
        names: set[str] = set()
        for index, tensor in enumerate(tensors):
            path = f"main.{key}[{index}]"
            if not isinstance(tensor, dict):
                raise ExportError("invalid_program", f"{path} must be an object")
            name = tensor.get("name")
            if not isinstance(name, str) or not name:
                raise ExportError("invalid_program", f"{path}.name must be non-empty")
            if name in all_names:
                raise ExportError("invalid_program", f"duplicate tensor name {name!r}")
            data_type = tensor.get("dataType")
            if data_type not in SUPPORTED_DATA_TYPES:
                raise ExportError("invalid_program", f"unsupported data type {data_type!r} at {path}")
            dimensions = tensor.get("dimensions")
            if not isinstance(dimensions, list) or not dimensions:
                raise ExportError("invalid_program", f"{path}.dimensions must be non-empty")
            for dimension_index, dimension in enumerate(dimensions):
                self._validate_dimension(dimension, f"{path}.dimensions[{dimension_index}]")
            all_names.add(name)
            names.add(name)
        return names

    def _validate_dimension(self, dimension: Any, path: str) -> None:
        if not isinstance(dimension, dict):
            raise ExportError("invalid_program", f"{path} must be an object")
        kind = dimension.get("kind")
        if kind == "fixed":
            if not self._is_integer(dimension.get("size")) or dimension["size"] <= 0:
                raise ExportError("invalid_program", f"{path}.size must be positive")
            return
        if kind == "dynamic":
            symbol = dimension.get("symbol")
            minimum = dimension.get("minimum")
            maximum = dimension.get("maximum")
            if not isinstance(symbol, str) or not symbol:
                raise ExportError("invalid_program", f"{path}.symbol must be non-empty")
            if not self._is_integer(minimum) or not self._is_integer(maximum):
                raise ExportError("invalid_program", f"{path} bounds must be integers")
            if minimum <= 0 or maximum < minimum:
                raise ExportError("invalid_program", f"{path} bounds are invalid")
            return
        raise ExportError("invalid_program", f"unsupported dimension kind {kind!r} at {path}")

    def _validate_region(
        self,
        region: dict[str, Any],
        path: str,
        state_names: set[str],
    ) -> None:
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
            if len(set(result_ids)) != len(result_ids):
                raise ExportError("invalid_graph", f"duplicate result IDs at {operation_path}")
            for value in result_ids:
                if value in defined:
                    raise ExportError(
                        "invalid_graph",
                        f"redefined value {value} at {operation_path}",
                    )
            self._validate_parameter_bindings(operation, operation_path)
            self._validate_state_bindings(operation, operation_path, state_names)
            kind = operation.get("kind")
            if not isinstance(kind, dict):
                raise ExportError("invalid_graph", f"{operation_path}.kind must be an object")
            self._validate_kind(kind, operation_path, state_names)
            defined.update(result_ids)

        for value in results:
            if value not in defined:
                raise ExportError("invalid_graph", f"undefined region result {value} at {path}")

    def _validate_parameter_bindings(
        self,
        operation: dict[str, Any],
        path: str,
    ) -> None:
        bindings = operation.get("parameterBindings")
        if not isinstance(bindings, list):
            raise ExportError("invalid_graph", f"{path}.parameterBindings must be an array")
        roles: set[str] = set()
        for index, binding in enumerate(bindings):
            binding_path = f"{path}.parameterBindings[{index}]"
            if not isinstance(binding, dict):
                raise ExportError("invalid_graph", f"{binding_path} must be an object")
            role = binding.get("role")
            tensor_name = binding.get("tensorName")
            if not isinstance(role, str) or not role or role in roles:
                raise ExportError("invalid_graph", f"invalid or duplicate role at {binding_path}")
            if not isinstance(tensor_name, str) or not tensor_name:
                raise ExportError("invalid_graph", f"{binding_path}.tensorName must be non-empty")
            roles.add(role)

    def _validate_state_bindings(
        self,
        operation: dict[str, Any],
        path: str,
        state_names: set[str],
    ) -> None:
        bindings = operation.get("stateBindings")
        if not isinstance(bindings, list):
            raise ExportError("invalid_graph", f"{path}.stateBindings must be an array")
        if self.execution == "stateless" and bindings:
            raise ExportError("invalid_graph", f"stateless operation has state at {path}")
        roles: set[str] = set()
        for index, binding in enumerate(bindings):
            binding_path = f"{path}.stateBindings[{index}]"
            if not isinstance(binding, dict):
                raise ExportError("invalid_graph", f"{binding_path} must be an object")
            role = binding.get("role")
            state = binding.get("state")
            axis_index = binding.get("axisIndex")
            if not isinstance(role, str) or not role or role in roles:
                raise ExportError("invalid_graph", f"invalid or duplicate role at {binding_path}")
            if state not in state_names:
                raise ExportError("invalid_graph", f"unknown state {state!r} at {binding_path}")
            if not self._is_integer(axis_index) or axis_index < 0:
                raise ExportError("invalid_graph", f"{binding_path}.axisIndex must be non-negative")
            roles.add(role)

    def _validate_kind(
        self,
        kind: dict[str, Any],
        path: str,
        state_names: set[str],
    ) -> None:
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
            self._validate_nested_region(kind, "body", path, state_names)
            return
        if tag == "parallel":
            branches = kind.get("branches")
            if not isinstance(branches, list):
                raise ExportError("invalid_graph", f"{path}.branches must be an array")
            for index, branch in enumerate(branches):
                if not isinstance(branch, dict):
                    raise ExportError("invalid_graph", f"{path}.branches[{index}] must be an object")
                self._validate_region(branch, f"{path}.branches[{index}]", state_names)
            return
        if tag == "repeating":
            count = kind.get("count")
            if not self._is_integer(count) or count < 0:
                raise ExportError("invalid_graph", f"{path}.count must be non-negative")
            self._validate_nested_region(kind, "body", path, state_names)
            return
        if tag == "conditional":
            self._validate_nested_region(kind, "then", path, state_names)
            self._validate_nested_region(kind, "else", path, state_names)
            return
        raise ExportError("invalid_graph", f"unsupported operation tag {tag!r} at {path}")

    def _validate_nested_region(
        self,
        kind: dict[str, Any],
        key: str,
        path: str,
        state_names: set[str],
    ) -> None:
        value = kind.get(key)
        if not isinstance(value, dict):
            raise ExportError("invalid_graph", f"{path}.{key} must be an object")
        self._validate_region(value, f"{path}.{key}", state_names)

    @staticmethod
    def _require_ids(container: dict[str, Any], key: str, path: str) -> list[int]:
        value = container.get(key)
        if not isinstance(value, list) or not all(ExportDocument._is_integer(item) for item in value):
            raise ExportError("invalid_graph", f"{path}.{key} must be an integer array")
        return value

    @staticmethod
    def _is_integer(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)
