# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

swift-mlx-lm is a Swift package providing GGUF-first language model inference on Apple Silicon via MLX/Metal. It is consumed by [AnyFoundationModels](https://github.com/1amageek) as the `MLXFoundationModels` backend.

**Core design principle**: GGUF is the standard input, `chat_template` is the canonical prompt formatter, MLX/Metal handles optimized execution. No dependency on swift-transformers — tokenizer, chat template evaluation, and model configuration are all restored from GGUF metadata.

## Build & Test

```bash
# Build
swift build

# Run all tests (always use xcodebuild — swift test crashes on Metal-dependent tests)
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS'

# Run specific module
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' -only-testing:MLXLMTests
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' -only-testing:GGUFParserTests
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' -only-testing:GGUFTokenizerTests
```

**Important**: `swift test` は使わない — Metal metallib が見つからずクラッシュするため。テスト実行は常に `xcodebuild test` を使用すること。

Swift tools version: 6.2. Platforms: macOS 15, iOS 18, visionOS 2.

## Architecture

### Design Philosophy (vs mlx-swift-lm)

This project differs fundamentally from mlx-swift-lm:

| | mlx-swift-lm | swift-mlx-lm |
|---|---|---|
| Input format | safetensors + config.json + tokenizer.json | GGUF single file |
| Tokenizer | swift-transformers dependency | Self-contained from GGUF metadata |
| Chat template | swift-transformers Jinja evaluator | swift-jinja (huggingface/swift-jinja) |
| Quantization | MLX post-load quantization | GGUF native (Q4_K, Q8_0, etc.) direct on Metal |
| Dependencies | mlx-swift + swift-transformers + HubApi | mlx-swift only |

### Integration Point

AnyFoundationModels (`/Users/1amageek/Desktop/AnyFoundationModels`) consumes this package through `MLXFoundationModels` module. The key interface is `ModelContainer`, which provides:
- `prepare(input:) async` — tokenization, prompt processing, and vision encoding
- `preparePrefix(input:) async` — prefix-only preparation (no generation prompt)
- `perform(values:operation:)` — generation with exclusive model access
- `generate(input:parameters:)` — streaming generation via `AsyncStream<Generation>`
- `decode(tokens:)` — reverse tokenization

### Module Structure

- **GGUFParser** — Binary parser, metadata extraction, tensor directory, mmap/lazy load. No external deps.
- **GGUFTokenizer** — BPE tokenizers from GGUF metadata. Merges-based (GPT-2/Llama3/Qwen2) and scores-based (SentencePiece/Llama1/2). Depends on GGUFParser.
- **MLXLM** — Model architectures, GGUF→MLX bridge, generation engine. Depends on GGUFParser, GGUFTokenizer, swift-jinja, mlx-swift.

External dependencies: `mlx-swift` (0.30.6+), `swift-jinja` (2.3.2+, library name `Jinja`).

### MLXLM Data Flow

```
GGUF URL → GGUFModelLoader.load(url:)
  ├─ GGUFFile.parse()          → metadata + tensor directory
  ├─ GGUFConfigExtractor       → LlamaConfiguration / Qwen25VLConfiguration
  ├─ Model(config)             → empty model (LlamaModel or Qwen25VLModel)
  ├─ GGUFTensorBridge          → dequantize tensors → MLXArray
  ├─ TensorNameMapper          → GGUF names → MLX weight paths
  ├─ model.update(parameters:) → loaded model
  ├─ (VLM) GGUFVisionLoader   → mmproj GGUF → vision encoder weights
  ├─ GGUFTokenizerFactory      → Tokenizer
  ├─ ChatTemplateRenderer      → Jinja prompt formatter
  ├─ UserInputProcessor        → GGUFUserInputProcessor or VLMUserInputProcessor
  └─ ModelContainer(context:)  → ready for generation
```

### Multimodal Input Flow (VLM)

```
UserInput(chat: [.user(text, images: [.url(...)])], tools:, additionalContext:)
  → VLMUserInputProcessor.prepare(input:) async
    ├─ Image preprocessing (resize, normalize)
    ├─ Vision token placeholder injection
    ├─ ChatTemplateRenderer.render() with additionalContext
    ├─ Tokenization
    └─ LMInput(text:, image: ProcessedImage, video: ProcessedVideo)
  → ModelContainer.generate(input:parameters:)
    ├─ Qwen25VLModel.encodeVision() → vision embeddings
    ├─ Merge vision embeddings into text sequence
    ├─ M-RoPE position IDs (temporal/height/width)
    └─ TokenIterator → AsyncStream<Generation>
```

### MLXLM Key Types

- `ModelContainer` (actor) — Serializes all model access, async prepare/generate
- `ModelContext` — Bundles model + tokenizer + processor + config
- `GGUFModelLoader` — End-to-end GGUF → ModelContainer (supports mmproj for VLM)
- `GGUFTensorBridge` — Dequantizes F16/F32/BF16/Q4_0/Q8_0/Q2_K-Q6_K
- `LlamaModel` — Llama/Mistral/Qwen2 architecture
- `Qwen25VLModel` — Qwen2.5-VL vision-language model (M-RoPE, vision encoder)
- `TokenIterator` — Autoregressive token generation loop (iterative prefill)
- `generate()` — Top-level AsyncStream generation function

### Chat & Input Types

- `Chat.Message` — Message with role, content, images, videos
- `UserInput` — Chat messages + tools + additionalContext + media processing
- `UserInputProcessor` (protocol, async) — Converts UserInput → LMInput
- `GGUFUserInputProcessor` — Text-only processor (chat_template + tokenizer)
- `VLMUserInputProcessor` — Vision processor (image preprocessing + vision tokens)
- `LMInput` — Tokenized input with optional ProcessedImage/ProcessedVideo
- `ChatTemplateRenderer` — Jinja template evaluation with additionalContext passthrough
- `ToolCallFormat` — Extensible format (.json, .xmlFunction, .lfm2, .glm4, .gemma)

### Generic vs Model-Specific Boundary

The text model side demonstrates the right pattern: `TransformerModel` handles 10+ architectures through a single `TransformerConfiguration` because all transformer decoders share the same computation graph (attention + FFN + norm) with minor flag-driven variations. All values come from GGUF metadata via `GGUFConfigExtractor`.

**Model-specific implementation is justified when the computation graph differs** — different layer types, different operations, different data flow. VLM vision encoders (Conv3d vs Conv2d, M-RoPE vs 1D RoPE, window attention vs global attention, spatial merge vs no merge) have fundamentally different computation graphs, so separate model types (e.g., `Qwen25VLModel`, future `LLaVAModel`) are correct.

**Model-specific implementation is NOT justified for configuration values.** These must always come from GGUF/mmproj metadata:

| Must extract from metadata | Never hardcode |
|---|---|
| Token IDs (image, video, vision_start, vision_end) | Magic numbers like `151655` |
| Vision encoder dimensions (hidden, depth, heads) | Default values that assume a specific model variant |
| Image normalization (mean, std) | Values like `(0.5, 0.5, 0.5)` |
| Patch size, spatial merge size, window size | Architecture-specific constants |
| M-RoPE sections | Head dimension splits like `[16, 24, 24]` |
| Full attention block pattern | Fixed arrays like `[7, 15, 23, 31]` |

**Design rules**:
- Protocols define the interface: `VisionLanguageModel`, `VisionEncoder`, `UserInputProcessor`
- Model-specific code is isolated in its own directory (e.g., `Models/Qwen25VL/`)
- The loader dispatches based on architecture string from GGUF metadata
- Config extraction throws on missing required metadata instead of falling back to model-specific defaults
- Token IDs are resolved from tokenizer vocabulary or GGUF metadata, not hardcoded

### Key Constraints

- GGUF single-file self-contained — no external tokenizer.json or config.json (VLM uses separate mmproj GGUF for vision encoder)
- No hand-written per-model formatters — `chat_template` evaluation is canonical
- No float16 fallback for unsupported quantization — all major GGUF quant types must run natively
- All public types must be `Sendable`
- `UserInputProcessor.prepare/preparePrefix` are async — VLM processors need async image loading
- additionalContext flows through to Jinja template context — supports model-specific flags like `enable_thinking`
