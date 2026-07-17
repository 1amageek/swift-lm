import os
import unittest
from pathlib import Path


class RealLFM2LoweringTests(unittest.TestCase):
    def test_swift_graph_matches_hugging_face_and_stateful_execution(self) -> None:
        model_path = os.environ.get("SWIFTLM_COREAI_TEST_MODEL")
        stateless_path = os.environ.get("SWIFTLM_COREAI_STATELESS_DOCUMENT")
        stateful_path = os.environ.get("SWIFTLM_COREAI_STATEFUL_DOCUMENT")
        if not model_path or not stateless_path or not stateful_path:
            self.skipTest("Real LFM2 model and Swift documents are not configured")

        try:
            import torch
            from transformers import AutoModelForCausalLM
            from swiftlm_coreai.document import ExportDocument
            from swiftlm_coreai.lowering import TorchGraphLowerer
            from swiftlm_coreai.weights import SafetensorWeightStore
        except ImportError as error:
            self.skipTest(f"Lowering dependencies unavailable: {error}")

        model_directory = Path(model_path)
        stateless_document = ExportDocument.load(Path(stateless_path))
        stateful_document = ExportDocument.load(Path(stateful_path))
        reference_weights = SafetensorWeightStore(
            model_directory,
            torch,
            torch.float32,
        )
        stateless_reference = TorchGraphLowerer(
            stateless_document,
            reference_weights,
            torch,
        ).make_stateless_model().eval()
        execution_weights = SafetensorWeightStore(
            model_directory,
            torch,
            torch.float16,
        )
        stateless_execution = TorchGraphLowerer(
            stateless_document,
            execution_weights,
            torch,
        ).make_stateless_model().eval()
        stateful = TorchGraphLowerer(
            stateful_document,
            execution_weights,
            torch,
        ).make_stateful_model().eval()
        reference = AutoModelForCausalLM.from_pretrained(
            model_directory,
            dtype=torch.float32,
        ).eval()

        tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
        positions = torch.arange(tokens.shape[1], dtype=torch.int32).unsqueeze(0)
        with torch.no_grad():
            expected = reference(input_ids=tokens.long()).logits
            actual = stateless_reference(tokens, positions)
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
        self.assertEqual(
            actual.topk(5, dim=-1).indices.tolist(),
            expected.topk(5, dim=-1).indices.tolist(),
        )

        states = tuple(
            torch.zeros(
                _resolved_state_shape(tensor, tokens.shape[1]),
                dtype=_torch_dtype(torch, tensor["dataType"]),
            )
            for tensor in stateful_document.function["states"]
        )
        with torch.no_grad():
            for index in range(tokens.shape[1]):
                prefix_positions = positions[:, : index + 1]
                decode_logits = stateful(
                    tokens[:, index : index + 1],
                    prefix_positions,
                    states,
                )
                full_logits = stateless_execution(
                    tokens[:, : index + 1],
                    prefix_positions,
                )[:, -1:, :]
                torch.testing.assert_close(
                    decode_logits,
                    full_logits,
                    rtol=1e-5,
                    atol=1e-5,
                )


def _resolved_state_shape(tensor: dict, context_length: int) -> tuple[int, ...]:
    return tuple(
        dimension["size"]
        if dimension["kind"] == "fixed"
        else context_length
        for dimension in tensor["dimensions"]
    )


def _torch_dtype(torch, data_type: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[data_type]


if __name__ == "__main__":
    unittest.main()
