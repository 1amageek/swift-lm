# swift-lm

Declarative model graphs and Core AI model tooling for Apple platforms.

`swift-lm` loads Hugging Face model metadata, builds a backend-independent graph from a declarative Swift model definition, validates the graph, and exports a stable document for Apple's Core AI tooling. Standard Transformer models use Apple's high-level `CoreAILM` exporter and runtime. Model families outside that registry can use the low-level Core AI program exporter and runtime session.

The direct Metal runtime remains available as the 0.10 compatibility path. New model support and public API work should target Core AI first.

## What It Provides

- Declarative `LMIR` and `LMArchitecture` model graphs.
- Deterministic Core AI export documents with explicit graph, operation, and weight-binding contracts.
- Swift CLI for converting local Hugging Face `config.json` files into export documents.
- Python tooling that validates documents and invokes Apple's official Core AI exporter.
- Low-level stateful Core AI sessions for custom model families such as LFM2.
- The 0.10 direct Metal loader, generation API, embeddings, and multimodal runtime for compatibility.

## Requirements

- Xcode 27 beta or later
- Swift 6.4+
- macOS 27.0+ or iOS 27.0+ as declared by `Package.swift`
- Apple Silicon for local Core AI execution
- A Hugging Face model bundle containing:
  - `config.json`
  - model weights and tokenizer metadata when running a complete model

Optional files used when present:

- `tokenizer_config.json`
- `special_tokens_map.json`
- `chat_template.jinja`
- `preprocessor_config.json`
- `processor_config.json`

## Add the Package

```swift
dependencies: [
    .package(url: "https://github.com/1amageek/swift-lm.git", from: "0.11.0")
],
targets: [
    .target(
        name: "MyApp",
        dependencies: [
            .product(name: "SwiftLMFoundationModels", package: "swift-lm")
        ]
    )
]
```

## Core AI Workflow

The Core AI path separates model declaration, validation, export, and runtime:

```text
Hugging Face config.json
        |
        v
ModelDeclarations -> LMIR -> CoreAIExportDocument
                                      |
                                      v
                         coreai-models / coreai-torch
                                      |
                                      v
                                  .aimodel
                                      |
                                      v
                                  Core AI
```

Generate and validate a document from a local model configuration:

```bash
xcrun swift run swiftlm-ir \
    --config /path/to/config.json \
    --output /tmp/model.json \
    --name model \
    --target macos

PYTHONPATH=python/src python3 -m swiftlm_coreai.cli validate /tmp/model.json
```

Install the Python exporter in an isolated environment, then export a standard
Transformer model through Apple's pipeline:

```bash
python3 -m venv .venv
.venv/bin/pip install -e python
.venv/bin/swiftlm-coreai export /tmp/model.json \
    Qwen/Qwen3-0.6B \
    --output-dir /tmp/coreai-model \
    --overwrite
```

For LFM2 and LFM2 MoE, the same command selects the low-level Hugging Face
Torch adapter and emits a dynamic stateless asset:

```bash
.venv/bin/swiftlm-coreai export /tmp/lfm2.json \
    LiquidAI/LFM2.5-1.2B-Instruct \
    --output-dir /tmp/coreai-lfm2 \
    --overwrite
```

Use `--stateful` when the application owns a serial Core AI context. The
stateful asset exposes `keyCache`, `valueCache`, and `convCache`; its input is
one token per call and `position_ids` contains the complete prefix position
range:

```bash
.venv/bin/swiftlm-coreai export /tmp/lfm2.json \
    LiquidAI/LFM2.5-1.2B-Instruct \
    --output-dir /tmp/coreai-lfm2-stateful \
    --stateful \
    --overwrite
```

Use `CoreAIExport` when an application needs to build or inspect a document in
Swift. Use `SwiftLMFoundationModels` for Apple-supported language bundles and
`SwiftLMCoreAI` for direct asset inspection, specialization, and stateful
inference. LFM2 uses the low-level exporter because Apple's high-level
language-model registry does not provide its hybrid state layout. The current
LFM2 exporter emits a dynamic stateless asset by default and provides an
explicit stateful variant for serial token-by-token execution.

For a low-level asset with dynamic shapes, resolve every dynamic state at
session creation and every dynamic output at execution:

```swift
let session = try CoreAIStateSession(
    model: model,
    functionName: "main",
    stateShapes: [
        "keyCache": [2, 1, 1, 40960, 32],
        "valueCache": [2, 1, 1, 40960, 32]
    ]
)
let outputs = try await session.run(
    inputs: ["input_ids": inputIDs, "position_ids": positionIDs],
    outputShapes: ["logits": [1, 1, bundle.vocabSize]]
)
```

## Legacy Direct Metal API (0.10)

The following container, context, embedding, and multimodal APIs are retained
for 0.10 compatibility. They are not the Core AI runtime surface.

## Public API Shape

The public API is centered on immutable containers, explicit mutable contexts, and request value types.

For generation:

- `ModelBundleLoader`
- `LanguageModelContainer`
- `LanguageModelContext`
- `ModelInput`
- `GenerationParameters`
- `PromptSnapshot`

For text embeddings:

- `ModelBundleLoader`
- `TextEmbeddingContainer`
- `TextEmbeddingContext`
- `TextEmbeddingInput`

For most application code, use the container-level one-shot APIs. Use contexts when you need explicit runtime ownership, prompt staging, cache reset, or prompt snapshot reuse.

## Text Generation

Load a model from Hugging Face and generate from chat input:

```swift
import SwiftLM

let container = try await ModelBundleLoader().load(
    repo: "LiquidAI/LFM2.5-1.2B-Instruct"
)

let input = ModelInput(
    chat: [
        .system("You are a concise assistant."),
        .user("Write a haiku about Metal shaders.")
    ],
    promptOptions: .init(isThinkingEnabled: true)
)

let stream = try await container.generate(
    input,
    parameters: GenerationParameters(
        maxTokens: 128,
        streamChunkTokenCount: 8,
        temperature: 0.6,
        topP: 0.9,
        reasoning: .separate
    )
)

for await event in stream {
    switch event {
    case .text(let text):
        print(text, terminator: "")
    case .reasoning(let reasoning):
        fputs(reasoning, stderr)
    case .completed(let info):
        print("\nGenerated \(info.tokenCount) tokens at \(info.tokensPerSecond) tok/s")
    }
}
```

Load from a local Hugging Face snapshot directory:

```swift
import Foundation
import SwiftLM

let directory = URL(fileURLWithPath: "/path/to/model-snapshot")
let container = try await ModelBundleLoader().load(directory: directory)
```

`ModelBundleLoader` creates `model.staf` next to the source weights when the executable cache needs to be generated. The cache can be deleted and rebuilt from `safetensors`.

## Prompt Preparation and Reasoning

`ModelInput` is the primary generation request type:

- `ModelInput(prompt:)` for plain text
- `ModelInput(chat:)` for chat prompts
- `InputMessage.Content.image(...)` and `.video(...)` for multimodal chat content when supported

Prompt-time options and output-time options are intentionally separate:

- `PromptPreparationOptions.isThinkingEnabled` controls chat-template rendering for bundles that expose `enable_thinking`.
- `PromptPreparationOptions.templateVariables` passes extra values to template rendering.
- `GenerationParameters.reasoning` controls whether reasoning content is hidden, inline, or emitted as `.reasoning(String)` events.

## Prompt Reuse

Use `PromptSnapshot` when many requests share the same prefix.

```swift
let context = try LanguageModelContext(container)

let snapshot = try await PromptSnapshot(
    from: ModelInput(chat: [
        .system("You are a careful code reviewer."),
        .user("Review this patch.")
    ]),
    using: context
)

let stream = try context.generate(
    from: snapshot,
    parameters: GenerationParameters(maxTokens: 64)
)

for await event in stream {
    if case .text(let text) = event {
        print(text, terminator: "")
    }
}
```

Stateless tokenizer helpers live on `LanguageModelContainer`:

```swift
let tokenIDs = container.encode("Hello")
let text = container.decode(tokenIDs)
```

Mutable runtime state is owned by `LanguageModelContext`:

```swift
let context = try LanguageModelContext(container)
context.resetState()
```

## Text Embeddings

`ModelBundleLoader.loadTextEmbeddings(...)` loads sentence-transformers style embedding bundles and returns a `TextEmbeddingContainer`.

```swift
import SwiftLM

let embeddings = try await ModelBundleLoader().loadTextEmbeddings(
    repo: "google/embeddinggemma-300m"
)

let vector = try embeddings.embed(
    TextEmbeddingInput(
        "swift metal inference",
        promptName: embeddings.defaultPromptName
    )
)

print(vector.count)
```

Use `TextEmbeddingContext` only when you need explicit mutable runtime ownership:

```swift
let context = try TextEmbeddingContext(embeddings)
let vector = try context.embed(
    TextEmbeddingInput(
        "swift metal inference",
        promptName: embeddings.defaultPromptName
    )
)
```

EmbeddingGemma support includes:

- `google/embeddinggemma-300m`
- `mlx-community/embeddinggemma-300m-bf16`
- `mlx-community/embeddinggemma-300m-4bit`

## Multimodal Support

Runtime capabilities are exposed through `ModelConfiguration`:

- `configuration.inputCapabilities`
- `configuration.executionCapabilities`
- `configuration.vision`

Inspect these values before showing image or video UI. A bundle can declare multimodal metadata even when a specific runtime path is unavailable.

Current implementation notes:

- Qwen vision families support image prompt preparation and image execution when compatible vision metadata and weights are present.
- Qwen vision families support video prompt preparation and video execution when compatible video processor metadata and weights are present.
- Gemma4 supports image prompt preparation and image execution when the bundle includes compatible vision metadata and weights.
- Gemma4 video execution is not implemented.

Example:

```swift
if container.configuration.executionCapabilities.supportsImageExecution {
    let input = ModelInput(chat: [
        .user([
            .text("Describe this image."),
            .image(InputImage(fileURL: URL(fileURLWithPath: "/path/to/image.jpg")))
        ])
    ])

    let stream = try await container.generate(
        input,
        parameters: GenerationParameters(maxTokens: 128)
    )

    for await event in stream {
        if case .text(let chunk) = event {
            print(chunk, terminator: "")
        }
    }
}
```

## Supported Model Families

The loader resolves these families from `config.json["model_type"]`:

| Family | `model_type` values |
|---|---|
| Transformer | `llama`, `qwen2`, `qwen3`, `mistral`, `gemma`, `gemma2`, `phi`, `phi3`, `starcoder2`, `gpt_neox`, `internlm2`, `deepseek`, `yi`, `baichuan`, `chatglm`, `mixtral`, `qwen2_moe`, `deepseek_v2`, `arctic`, `dbrx` |
| Gemma3 text / EmbeddingGemma | `gemma3_text` |
| Gemma4 | `gemma4`, `gemma4_text` |
| Qwen 3.5 hybrid / Qwen vision text backbone | `qwen3_5`, `qwen3_vl`, `qwen2_5_vl`, `qwen2_vl` |
| LFM2 / LFM2.5 hybrid | `lfm2`, `lfm2_moe` |
| Cohere | `cohere`, `command-r` |

`nemotron_h` is explicitly rejected by the current loader. Unsupported or incomplete families fail during loading or graph construction rather than falling back silently.

## Architecture

The repository is split into five layers:

```text
LMIR
  Backend-independent graph and operation model.

LMArchitecture
  Declarative model DSL and validation.

ModelDeclarations
  Family-specific model declarations.

MetalCompiler
  IR lowering, fragment planning, dispatch optimization, kernel generation,
  STAF loading, and direct Metal execution planning.

SwiftLM
  Public loading, prompt preparation, tokenization, generation, and embedding API.
```

Dependency direction:

```text
LMIR  <-  LMArchitecture  <-  ModelDeclarations
  |                               |
  +--------  MetalCompiler  ------+
                  |
                  +--------  SwiftLM
```

Design constraints:

- `LMIR` stays semantic and backend-independent.
- Model declarations describe architecture families, not backend shortcuts.
- `safetensors` is canonical; STAF is a regenerable execution cache.
- Component-local optimization belongs in `MetalCompilable`.
- Cross-component fusion belongs in the compiler.
- Runtime-critical failures should be reported explicitly, not hidden behind silent fallbacks.

## Extending Models and Compiler

Model declarations and Metal execution are connected through a strict chain:

```text
ModelComponent (declaration)
  -> OperationAttributes (LMIR)
  -> MetalCompilable (MetalCompiler bridge)
  -> MetalKernelFragment / PrimitiveMetalKernelFragment (Metal)
```

This boundary is important:

- `LMIR` defines semantic attributes and must not import Metal or backend-specific types.
- `LMArchitecture` exposes reusable model-building components.
- `ModelDeclarations` assembles family/product graphs from those components.
- `MetalCompiler` adds retroactive `MetalCompilable` conformances and turns IR attributes into optimized fragment trees.
- The compiler walks contracts. It should not grow model-family switches for ordinary primitives.

### Adding a Model Component

Add a primitive in this order.

First, define backend-independent attributes in `Sources/LMIR/IR`:

```swift
public struct MyOpAttributes: OperationAttributes, Codable, Equatable {
    public let dimension: Int
    public let epsilon: Float

    public init(dimension: Int, epsilon: Float) {
        self.dimension = dimension
        self.epsilon = epsilon
    }
}
```

Then define the public declaration component in `Sources/LMArchitecture/Declaration/Components`:

```swift
public struct MyOp: ModelComponent {
    public typealias Attributes = MyOpAttributes

    public let dimension: Int
    public let epsilon: Float

    public init(dimension: Int, epsilon: Float = 1e-6) {
        precondition(dimension > 0, "dimension must be positive")
        self.dimension = dimension
        self.epsilon = epsilon
    }

    public var attributes: MyOpAttributes {
        MyOpAttributes(dimension: dimension, epsilon: epsilon)
    }
}
```

Add the Metal bridge in `Sources/MetalCompiler/Compilable`:

```swift
extension MyOpAttributes: MetalCompilable {
    func fragment(context: KernelContext) -> some MetalKernelFragment {
        Reduction(dimension: dimension, epsilon: epsilon, weightBias: 0)
    }
}
```

When an operation expands to multiple fragments, return the already-optimized component-local composition from `MetalCompilable`:

```swift
extension MyOpAttributes: MetalCompilable {
    @MetalKernelFragmentBuilder
    func fragment(context: KernelContext) -> some MetalKernelFragment {
        BatchedProjection(projections: [
            .init(field: "gate_proj", inputDimension: inputSize, outputDimension: intermediateSize),
            .init(field: "up_proj", inputDimension: inputSize, outputDimension: intermediateSize),
        ])
        ElementwiseFragment(count: intermediateSize, kind: .swiglu)
        LinearFragment(field: "down_proj", inputDimension: intermediateSize, outputDimension: outputSize)
    }
}
```

Use the component from model declarations only after the IR contract and Metal bridge exist:

```swift
struct MyModel: ModelArchitecture {
    var body: some ModelComponent {
        TokenEmbedding(vocabSize: vocabSize, embeddingSize: hiddenSize)
        Repeat(count: layerCount) {
            Residual {
                MyOp(dimension: hiddenSize)
            }
            Residual {
                Attention(
                    hiddenSize: hiddenSize,
                    headCount: headCount,
                    kvHeadCount: kvHeadCount
                )
            }
        }
        OutputHead(inputSize: hiddenSize, vocabSize: vocabSize)
    }
}
```

### Adding a Fragment

A fragment is a reusable Metal kernel building block. Fragments are composed by `MetalCompilable` conformances and consumed by the compiler through protocols and contracts.

Primitive fragments should describe:

- dispatch dimensions
- pipeline cache identity
- kernel source or composable kernel body
- decode bindings, including precise `writeBufferIndices`
- prefill steps
- fusion contract, if the fragment can be safely fused
- capability protocol conformances, if the fragment needs compiler resources

Skeleton:

```swift
public struct MyFragment: PrimitiveMetalKernelFragment {
    public let dimension: Int

    public var dispatchDimension: MetalDispatchDimension {
        .reduction(dimension: dimension)
    }

    public func kernelName(context: KernelContext) -> String {
        "my_fragment_\(context.bufferPrecision.suffix)"
    }

    public func kernelSource(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        """
        kernel void \(name)(
            device const \(bufferPrecision.typeName)* input [[buffer(0)]],
            device \(bufferPrecision.typeName)* output [[buffer(1)]],
            uint gid [[thread_position_in_grid]]
        ) {
            output[gid] = input[gid];
        }
        """
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        FragmentBindings(
            buffers: [
                (index: 0, buffer: context.currentInputBuffer, offset: context.currentInputOffset),
                (index: 1, buffer: context.bufferSet.hidden, offset: 0),
            ],
            bytes: [],
            outputIsHidden: true,
            writeBufferIndices: [1]
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        // Return dispatch steps for prefill execution.
    }
}
```

Declare `FusionContract` only when the fragment can be fused without changing graph semantics:

```swift
extension MyFragment {
    public var fusionContract: FusionContract? {
        FusionContract(
            ports: [
                FusionPort(
                    name: "input",
                    direction: .input,
                    role: .buffer,
                    accessPattern: .singlePass,
                    bufferIntent: .dataFlow
                ),
                FusionPort(
                    name: "output",
                    direction: .output,
                    role: .buffer,
                    accessPattern: .singlePass,
                    bufferIntent: .dataFlow
                ),
            ],
            parallelism: .perRow(dimension: dimension),
            threadgroupMemoryBytes: 0
        )
    }

    public func kernelBody(
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String? {
        """
        float value = input[gid];
        output[gid] = value;
        """
    }
}
```

Capability protocols should match fragment semantics:

| Protocol | Use for |
|---|---|
| `ProjectionDescribing` | Weight x input projections, GEMV/GEMM sizing, weight resolution, quantization planning, output marking |
| `ConvStateRequiring` | Temporal convolution with persistent state |
| `RecurrentStateRequiring` | Sequential recurrence with persistent state |
| `PerLayerInputCapable` | Per-layer external input injection |

Adding an ordinary fragment should not require changes to:

- `MetalSourceGenerator`
- `MetalKernelSourceCatalog`
- `MetalPrefillStepBuilder`
- `MetalDispatchStepBuilder`
- `MetalEntryCollector`

If a change appears to require one of those switches, first check whether the missing contract belongs on the fragment instead.

## Build and Test

Build:

```bash
swift build
```

Run a focused test target with a timeout:

```bash
perl -e 'alarm shift; exec @ARGV' 120 \
  xcodebuild test \
  -scheme swift-lm-Package \
  -destination 'platform=macOS' \
  -only-testing:SwiftLMTests
```

For real-model or Metal-heavy debugging, build once and run focused suites or cases one process at a time:

```bash
perl -e 'alarm shift; exec @ARGV' 120 \
  xcodebuild build-for-testing \
  -scheme swift-lm-Package \
  -destination 'platform=macOS'

perl -e 'alarm shift; exec @ARGV' 120 \
  xcodebuild test-without-building \
  -scheme swift-lm-Package \
  -destination 'platform=macOS' \
  -only-testing:SwiftLMTests/ReleaseSmokePromptStateTests
```

Useful runners:

- Qwen3.5+ multimodal suites: `scripts/benchmarks/run-qwen35-vision-tests.sh`
- Generation benchmark pipeline: `scripts/benchmarks/run-generation-pipeline.sh`
- Prefill route promotion artifact gate: `scripts/benchmarks/run-prefill-artifact-validation.sh`
- Xcode timeout wrapper: `scripts/xcodebuild/test-timeout.sh`
- Xcode hang guard: `scripts/xcodebuild/test-hang-guard.sh`

## Documentation

- Core AI architecture: [docs/design/core-ai.md](docs/design/core-ai.md)
- 0.11.0 alpha release notes: [docs/releases/0.11.0-alpha.1.md](docs/releases/0.11.0-alpha.1.md)
- Public API guide: [docs/using-swift-lm.md](docs/using-swift-lm.md)
- Production readiness gates: [docs/production-readiness.md](docs/production-readiness.md)
- Metal 4 design notes: [docs/design/metal4.md](docs/design/metal4.md)
- Quantization design notes: [docs/design/quantization.md](docs/design/quantization.md)
- Supported quantizations: [docs/design/supported-quantizations.md](docs/design/supported-quantizations.md)
- DocC sources: [Sources/SwiftLM/SwiftLM.docc](Sources/SwiftLM/SwiftLM.docc)
