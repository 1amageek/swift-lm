# Troubleshooting

## Loader Errors

`ModelBundleLoader` currently exposes these public errors:

- ``ModelBundleLoaderError/noMetalDevice``
- ``ModelBundleLoaderError/noSafetensorsFiles(_:)``
- ``ModelBundleLoaderError/invalidConfig(_:)``

Common causes include:

- the local model directory is missing `config.json`
- the bundle has no `.safetensors` files
- the current machine does not expose a usable Metal device
- the model metadata does not include required fields such as `hidden_size`, `num_hidden_layers`, or `vocab_size`

## Generation Behavior

For predictable behavior, set ``GenerateParameters/maxTokens`` explicitly. If `maxTokens` is `nil`, the current runtime uses its default cap.

Generation is stream-based. If you need a single final string, collect all ``Generation/chunk`` values yourself.

## Current Public API Limits

- public input is text-only or chat-only through ``UserInput``
- multimodal image or video input is not part of the current public API
- tool calling and structured function-calling APIs are not part of the current public API

## Building Documentation

Use Xcode's package scheme to build the DocC archive:

```bash
xcodebuild docbuild -scheme swift-lm-Package -destination 'platform=macOS,arch=arm64' CODE_SIGNING_ALLOWED=NO
```

The generic macOS destination may try to include `x86_64`, which currently fails in `MetalCompiler`. Use the `arm64` destination on Apple Silicon.
