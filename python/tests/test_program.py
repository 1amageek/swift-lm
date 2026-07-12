import tempfile
import unittest
from pathlib import Path


class StatefulProgramTests(unittest.TestCase):
    def test_stateful_module_export(self) -> None:
        try:
            import torch
            from swiftlm_coreai.program import export_torch_module
        except ImportError as error:
            self.skipTest(f"Core AI Python dependencies unavailable: {error}")

        class Accumulator(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("state", torch.zeros((1,), dtype=torch.float32))

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                self.state.add_(input)
                return self.state + input * 0

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "Accumulator.aimodel"
            export_torch_module(
                Accumulator(),
                {"input": torch.ones((1,), dtype=torch.float32)},
                output,
                input_names=["input"],
                output_names=["output"],
                state_names=["state"],
            )
            self.assertTrue(output.is_dir())


if __name__ == "__main__":
    unittest.main()
