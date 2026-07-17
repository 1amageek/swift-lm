"""Lazy safetensors access for Core AI graph lowering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import ExportError


class SafetensorWeightStore:
    """Resolve Swift parameter bindings directly from Hugging Face safetensors."""

    def __init__(self, model_directory: Path, torch: Any, dtype: Any) -> None:
        self.model_directory = model_directory
        self.torch = torch
        self.dtype = dtype
        self._weight_map = self._load_weight_map()
        self._unindexed_files = sorted(model_directory.glob("*.safetensors"))
        if not self._weight_map and not self._unindexed_files:
            raise ExportError(
                "weights_not_found",
                f"No safetensors files found in {model_directory}",
            )

    def tensor(self, name: str) -> Any:
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise ExportError(
                "missing_coreai_dependencies",
                "Install the pinned Python dependencies from python/pyproject.toml",
            ) from error

        candidates: list[Path]
        if name in self._weight_map:
            candidates = [self.model_directory / self._weight_map[name]]
        else:
            candidates = self._unindexed_files

        for path in candidates:
            with safe_open(path, framework="pt", device="cpu") as handle:
                if name in handle.keys():
                    return handle.get_tensor(name).to(dtype=self.dtype).contiguous()
        raise ExportError("weight_not_found", name)

    def _load_weight_map(self) -> dict[str, str]:
        index_files = sorted(self.model_directory.glob("*.safetensors.index.json"))
        if not index_files:
            return {}
        try:
            payload = json.loads(index_files[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ExportError("invalid_weight_index", str(index_files[0])) from error
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in weight_map.items()
        ):
            raise ExportError("invalid_weight_index", str(index_files[0]))
        return weight_map
