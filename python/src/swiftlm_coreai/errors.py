"""Typed errors for the swift-lm Core AI exporter."""


class ExportError(Exception):
    """An export request cannot be completed under the declared contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"
