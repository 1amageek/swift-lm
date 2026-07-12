# Core AI Architecture

## Scope

Core AI is the primary execution and distribution target for the 0.11 line.
The repository keeps the 0.10 direct Metal implementation for compatibility,
but new model declarations and public integration should be designed around
Core AI assets and Apple's model APIs.

The input contract remains a Hugging Face bundle. `config.json` and the model
declaration are the source inputs; `.aimodel` is a derived deployment asset.
The repository does not make `.aimodel` authoritative over the original
weights.

## Layered Flow

```text
Hugging Face bundle
  config.json + safetensors + tokenizer metadata
                  |
                  v
ModelDeclarations
  family-level declarative Swift components
                  |
                  v
LMArchitecture -> LMIR
  normalized graph, regions, operations, and parameter bindings
                  |
                  v
CoreAIExport
  versioned JSON export document
                  |
                  +------------------------------+
                  |                              |
                  v                              v
Apple Core AI exporter                 Core AI Torch converter
standard Transformer                    custom/stateful families
                  |                              |
                  +--------------+---------------+
                                 v
                              .aimodel
                                 |
                                 v
                   AIModel / InferenceFunction
                                 |
                                 v
                      Core AI runtime on macOS/iOS 27+
```

The Swift export document is deliberately backend-neutral at the graph level.
It contains stable operation tags, JSON attributes, SSA value IDs, structural
regions, and explicit tensor-name bindings. Python is responsible for turning
that validated contract into Apple's Core AI asset format.

## Swift Products

| Product | Responsibility |
|---|---|
| `LMIR` | Backend-independent graph and operation model |
| `LMArchitecture` | Declarative model components and graph validation |
| `ModelDeclarations` | Transformer and LFM2 family declarations |
| `CoreAIExport` | Versioned document and graph-to-document exporter |
| `SwiftLMCoreAI` | Asset validation, specialization, state allocation, and inference |
| `SwiftLMFoundationModels` | Apple `CoreAILanguageModels` bundle adapter |
| `SwiftLMIR` | `config.json` to export-document command-line tool |
| `MetalCompiler` / `SwiftLM` | 0.10 direct Metal compatibility path |

## Apple API Boundary

The runtime wrapper uses the following public Core AI API boundaries:

1. `AIModelAsset` validates `.aimodel` files and exposes function/state
   descriptors.
2. `AIModel.specialize` compiles or loads a specialized model through
   `AIModelCache`.
3. `InferenceFunction` owns an executable function and accepts `NDArray`
   inputs, outputs, and state views.
4. `ComputeStream` allows callers to provide an explicit scheduling stream.
5. `CoreAILanguageModels.LanguageBundle` and `CoreAILanguageModel` provide the
   high-level Apple runtime for supported language-model bundles.

`CoreAIModelAsset` rejects unsupported specialization settings explicitly.
The current beta has a reproducible failure when
`expectFrequentReshapes` is enabled, so callers must resolve dynamic shapes
before execution and use the default specialization policy.

## Model-Family Routing

| Family | Export route | Runtime route |
|---|---|---|
| Standard Transformer | `swiftlm-ir` + `swiftlm-coreai export` | `SwiftLMFoundationModels` / `CoreAILanguageModel` |
| LFM2 / LFM2 MoE | `swiftlm-coreai export` uses the Hugging Face Torch adapter and emits a dynamic stateless asset; `export_torch_module` remains the custom stateful extension point | `SwiftLMCoreAI` / `CoreAIStateSession` |
| Unsupported family | explicit validation error | no fallback |

The low-level session supports heterogeneous persistent states such as key
cache, value cache, convolution state, and recurrent state. Callers must pass
the complete resolved shape for every dynamic state through `stateShapes`, and
the complete resolved shape for every dynamic output through `outputShapes` on
each run. This avoids Core AI's fatal unresolved-dimension path and prevents
the wrapper from inventing a context length. At most four NDArray state tensors
and one NDArray output are currently supported by the Swift wrapper; image
states and larger output sets fail with typed errors.

## Export Commands

```bash
# Build the Swift graph exporter.
xcrun swift build -c release --product swiftlm-ir

# Emit a deterministic document.
xcrun swift run swiftlm-ir \
  --config /path/to/config.json \
  --output /tmp/model.json \
  --name model \
  --target macos

# Validate the document before invoking Apple's exporter.
PYTHONPATH=python/src python3 -m swiftlm_coreai.cli validate /tmp/model.json

# LFM2 and LFM2 MoE use the low-level Torch adapter.
PYTHONPATH=python/src python3 -m swiftlm_coreai.cli export \
  /tmp/lfm2.json LiquidAI/LFM2.5-1.2B-Instruct \
  --output-dir /tmp/coreai-lfm2 --overwrite

# Inspect a generated asset with Apple's toolchain.
xcrun coreai-build inspect --json /tmp/coreai-model/model.aimodel
```

The Python project pins the Core AI beta packages in
`python/pyproject.toml`. The Swift package uses the corresponding Apple
`coreai-models` revision and its transitive `xgrammar` dependency. Keep those
versions aligned when updating the Xcode beta.

## Verification Contract

The minimum validation order is:

```text
graph validation
  -> deterministic document and binding tests
  -> Core AI asset inspection
  -> stateful runtime smoke test
  -> real model output correctness
  -> performance measurement
```

Performance results are not release evidence when graph export, model output,
state reset, or prompt-state behavior is unresolved. The derived asset should
be regenerated whenever the Hugging Face weights or graph declaration changes.
