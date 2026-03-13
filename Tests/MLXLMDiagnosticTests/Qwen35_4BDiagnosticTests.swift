import Foundation
import GGUFParser
import GGUFValidation
import GGUFTokenizer
import GGUFToolingCore
import MLX
import Testing
import TestHeartbeat
@testable import MLXCompiler
@testable import MLXLM

@Suite("Qwen3.5-4B Diagnostics", .tags(.diagnostic), .heartbeat)
struct Qwen35_4BDiagnosticTests {

    private static let repo = "unsloth/Qwen3.5-4B-GGUF"
    private static let filename = "Qwen3.5-4B-Q4_K_M.gguf"
    private static let repairedFilename = "Qwen3.5-4B-Q4_K_M.repaired.gguf"
    private static let smallRepo = "unsloth/Qwen3.5-0.8B-GGUF"
    private static let smallFilename = "Qwen3.5-0.8B-Q4_K_M.gguf"
    private static let smallRepairedFilename = "Qwen3.5-0.8B-Q4_K_M.repaired.gguf"

    private func downloadModel() async throws -> URL {
        for candidate in localModelCandidates() {
            if FileManager.default.fileExists(atPath: candidate.path) {
                print("[qwen35-4b][cache] using local model at \(candidate.path)")
                return candidate
            }
        }

        let downloader = HuggingFaceDownloader()
        return try await downloader.download(
            repo: Self.repo,
            filename: Self.filename
        )
    }

    private func localModelCandidates() -> [URL] {
        let home = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
        return [
            home
                .appendingPathComponent("Library/Containers/team.stamp.JARDIS.ml/Data/Library/Application Support/swift-mlx-lm/huggingface", isDirectory: true)
                .appendingPathComponent("unsloth--Qwen3.5-4B-GGUF/main", isDirectory: true)
                .appendingPathComponent(Self.repairedFilename, isDirectory: false),
            home
                .appendingPathComponent("Library/Application Support/swift-mlx-lm/huggingface", isDirectory: true)
                .appendingPathComponent("unsloth--Qwen3.5-4B-GGUF/main", isDirectory: true)
                .appendingPathComponent(Self.filename, isDirectory: false),
            home
                .appendingPathComponent("Library/Containers/team.stamp.JARDIS.ml/Data/Library/Application Support/swift-mlx-lm/huggingface", isDirectory: true)
                .appendingPathComponent("unsloth--Qwen3.5-4B-GGUF/main", isDirectory: true)
                .appendingPathComponent(Self.filename, isDirectory: false),
        ]
    }

    private func originalModelURL() throws -> URL {
        let home = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
        let candidates = [
            home
                .appendingPathComponent("Library/Containers/team.stamp.JARDIS.ml/Data/Library/Application Support/swift-mlx-lm/huggingface", isDirectory: true)
                .appendingPathComponent("unsloth--Qwen3.5-4B-GGUF/main", isDirectory: true)
                .appendingPathComponent(Self.filename, isDirectory: false),
            home
                .appendingPathComponent("Library/Application Support/swift-mlx-lm/huggingface", isDirectory: true)
                .appendingPathComponent("unsloth--Qwen3.5-4B-GGUF/main", isDirectory: true)
                .appendingPathComponent(Self.filename, isDirectory: false),
        ]

        for candidate in candidates where FileManager.default.fileExists(atPath: candidate.path) {
            return candidate
        }

        throw CocoaError(.fileNoSuchFile)
    }

    private func smallModelCandidates() -> [URL] {
        let home = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
        return [
            home
                .appendingPathComponent("Library/Containers/team.stamp.JARDIS.ml/Data/Library/Application Support/swift-mlx-lm/huggingface", isDirectory: true)
                .appendingPathComponent("unsloth--Qwen3.5-0.8B-GGUF/main", isDirectory: true)
                .appendingPathComponent(Self.smallRepairedFilename, isDirectory: false),
            home
                .appendingPathComponent("Library/Containers/team.stamp.JARDIS.ml/Data/Library/Application Support/swift-mlx-lm/huggingface", isDirectory: true)
                .appendingPathComponent("unsloth--Qwen3.5-0.8B-GGUF/main", isDirectory: true)
                .appendingPathComponent(Self.smallFilename, isDirectory: false),
            home
                .appendingPathComponent("Library/Application Support/swift-mlx-lm/huggingface", isDirectory: true)
                .appendingPathComponent("unsloth--Qwen3.5-0.8B-GGUF/main", isDirectory: true)
                .appendingPathComponent(Self.smallFilename, isDirectory: false),
        ]
    }

    private func downloadSmallModel() async throws -> URL {
        for candidate in smallModelCandidates() where FileManager.default.fileExists(atPath: candidate.path) {
            return candidate
        }

        let downloader = HuggingFaceDownloader()
        return try await downloader.download(
            repo: Self.smallRepo,
            filename: Self.smallFilename
        )
    }

    private func repairedURLIfNeeded(sourceURL: URL) throws -> URL {
        let file = try GGUFFile.parse(url: sourceURL)
        let plan = GGUFValidationRegistry.default.makeRepairPlan(
            file: file,
            mode: .includeInferredRepairs,
            sourceURL: sourceURL
        )
        guard !plan.actions.isEmpty else {
            return sourceURL
        }

        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(sourceURL.deletingPathExtension().lastPathComponent, isDirectory: false)
            .appendingPathExtension("diagnostic.repaired.gguf")

        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }

        try GGUFFileRewriter().applying(
            GGUFMetadataPatch(actions: plan.actions),
            to: sourceURL,
            outputURL: outputURL
        )
        return outputURL
    }

    @Test("Compiled 4B prefill reports config and finite logits")
    func compiledPrefillReportsFiniteLogits() async throws {
        let url = try await downloadModel()
        let file = try GGUFFile.parse(url: url)

        let buildResult = try GGUFGraphBuilder().build(file: file)
        let config = buildResult.config

        print("[qwen35-4b][config] detected=\(buildResult.architecture)")
        print("[qwen35-4b][config] attentionHeads=\(config.attentionHeads) kvHeads=\(config.kvHeads) headDim=\(config.headDim)")
        print("[qwen35-4b][config] ssmNumHeads=\(config.ssmNumHeads ?? -1) ssmKeyHeadDim=\(config.ssmKeyHeadDim ?? -1) ssmValueHeadDim=\(config.ssmValueHeadDim ?? -1)")
        print("[qwen35-4b][metadata] ssm.group_count=\(file.ssmGroupCount ?? -1) ssm.time_step_rank=\(file.ssmNumHeads ?? -1) ssm.state_size=\(file.ssmStateSize ?? -1) ssm.inner_size=\(file.ssmInnerSize ?? -1)")

        if let ssmNumHeads = config.ssmNumHeads,
           let valueHeadDim = config.ssmValueHeadDim,
           let innerSize = file.ssmInnerSize {
            #expect(ssmNumHeads * valueHeadDim == innerSize)
        }

        let loader = GGUFModelLoader()
        let context = try loader.loadCompiledContext(url: url)
        let lmInput = try await context.processor.prepare(input: UserInput(prompt: "hi"))
        let cache = context.model.newCache(parameters: nil)
        let output = try runPrefill(
            model: context.model,
            input: lmInput,
            cache: cache,
            windowSize: 1024
        )

        let logits = nextTokenLogits(from: output)
        eval(logits)
        let values = logits.flattened().asType(.float32).asArray(Float.self)

        let hasNaN = values.contains { $0.isNaN }
        let hasInf = values.contains { !$0.isFinite }
        let minLogit = values.min() ?? .nan
        let maxLogit = values.max() ?? .nan

        print("[qwen35-4b][prefill] promptTokens=\(lmInput.text.tokens.dim(1)) cacheOffset=\(cache.first?.offset ?? -1)")
        print("[qwen35-4b][prefill] hasNaN=\(hasNaN) hasInf=\(hasInf) min=\(minLogit) max=\(maxLogit)")

        printTopCandidates(
            label: "qwen35-4b/compiled/prefill",
            logits: logits,
            tokenizer: context.tokenizer,
            limit: 10
        )

        #expect(!hasNaN)
        #expect(!hasInf)
    }

    @Test("4B DeltaNet coefficients are inspectable")
    func deltaNetCoefficientInspection() async throws {
        let url = try await downloadModel()
        let loader = GGUFModelLoader()
        let context = try loader.loadContext(url: url)
        let model = try #require(context.model as? HybridDeltaNetAttentionModel)
        let delta0 = try #require(model.model.layers[0].deltaNet)

        eval(delta0.A_log, delta0.dt_bias)
        let aValues = delta0.A_log.asType(.float32).asArray(Float.self)
        let dtValues = delta0.dt_bias.asType(.float32).asArray(Float.self)

        let aMin = aValues.min() ?? .nan
        let aMax = aValues.max() ?? .nan
        let dtMin = dtValues.min() ?? .nan
        let dtMax = dtValues.max() ?? .nan

        let expOfA = aValues.prefix(8).map { Foundation.exp(Double($0)) }
        let negExpOfA = aValues.prefix(8).map { -Foundation.exp(Double($0)) }

        print("[qwen35-4b][delta0] A_log shape=\(delta0.A_log.shape) min=\(aMin) max=\(aMax) values=\(aValues.prefix(16))")
        print("[qwen35-4b][delta0] dt_bias shape=\(delta0.dt_bias.shape) min=\(dtMin) max=\(dtMax) values=\(dtValues.prefix(16))")
        print("[qwen35-4b][delta0] exp(A_log prefix)=\(expOfA)")
        print("[qwen35-4b][delta0] -exp(A_log prefix)=\(negExpOfA)")

        #expect(!aValues.isEmpty)
        #expect(!dtValues.isEmpty)
    }

    @Test("Compiled 4B prompt and first decode steps are traceable")
    func compiledPromptAndDecodeTrace() async throws {
        let url = try await downloadModel()
        let loader = GGUFModelLoader()
        let context = try loader.loadCompiledContext(url: url)

        let lmInput = try await context.processor.prepare(input: UserInput(prompt: "hi"))
        let promptTokens = tokenIDs(from: lmInput.text.tokens)
        let promptTail = Array(promptTokens.suffix(24))
        let promptTailPieces = promptTail.map { context.tokenizer.tokenToString($0) ?? "<?>" }
        let promptTailJoined = promptTailPieces.joined()

        print("[qwen35-4b][prompt-tail] ids=\(promptTail)")
        print("[qwen35-4b][prompt-tail] pieces=\(promptTailPieces)")
        print("[qwen35-4b][prompt-tail] joined=\(quoted(promptTailJoined))")

        let promptTokenCount = lmInput.text.tokens.dim(1)
        let cache = context.model.newCache(parameters: nil)
        let prefillOutput = try runPrefill(
            model: context.model,
            input: lmInput,
            cache: cache,
            windowSize: 1024
        )

        let prefillLogits = nextTokenLogits(from: prefillOutput)
        printTopCandidates(
            label: "qwen35-4b/compiled/prefill",
            logits: prefillLogits,
            tokenizer: context.tokenizer,
            limit: 10
        )

        let sampler = ArgMaxSampler()
        var tokenID = Int(sampler.sample(logits: prefillLogits).item(Int32.self))
        var generated: [Int] = []

        for step in 0..<4 {
            generated.append(tokenID)
            let piece = context.tokenizer.tokenToString(tokenID) ?? "<??>"
            let decoded = context.tokenizer.decode(tokens: [tokenID])
            print(
                "[qwen35-4b][decode-step] step=\(step + 1) sampledID=\(tokenID) piece=\(quoted(piece)) decoded=\(quoted(decoded))"
            )

            let decodeInput = LMInput.Text(
                tokens: MLXArray([Int32(tokenID)]).reshaped([1, 1])
            )
            let decodeOutput = context.model.callAsFunction(decodeInput, cache: cache, state: nil)
            let offset = cache.first?.offset ?? -1
            #expect(offset == promptTokenCount + step + 1)

            let nextLogits = nextTokenLogits(from: decodeOutput)
            printTopCandidates(
                label: "qwen35-4b/compiled/decode-step-\(step + 1)",
                logits: nextLogits,
                tokenizer: context.tokenizer,
                limit: 10
            )

            tokenID = Int(sampler.sample(logits: nextLogits).item(Int32.self))
        }

        let generatedText = context.tokenizer.decode(tokens: generated)
        print("[qwen35-4b][decode-trace] tokens=\(generated)")
        print("[qwen35-4b][decode-trace] text=\(quoted(generatedText))")
    }

    @Test("Compiled 4B chat template render reveals thinking defaults")
    func compiledChatTemplateThinkingDefaults() async throws {
        let url = try await downloadModel()
        let file = try GGUFFile.parse(url: url)
        let template = try #require(file.chatTemplate)

        let loader = GGUFModelLoader()
        let context = try loader.loadCompiledContext(url: url)
        let bosToken = context.tokenizer.bosTokenID.flatMap { context.tokenizer.tokenToString($0) }
        let eosToken = context.tokenizer.eosTokenID.flatMap { context.tokenizer.tokenToString($0) }
        let renderer = try ChatTemplateRenderer(
            templateString: template,
            bosToken: bosToken,
            eosToken: eosToken
        )

        let containsThink = template.contains("think")
        print("[qwen35-4b][template][contains-think]=\(containsThink)")
        print("[qwen35-4b][template][snippet]=\(quoted(templateSnippet(template)))")

        let messages = [Chat.Message.user("hi")]
        let renderedDefault = try renderer.render(
            messages: messages,
            tools: nil,
            additionalContext: nil,
            addGenerationPrompt: true
        )
        let renderedThinkingOff = try renderer.render(
            messages: messages,
            tools: nil,
            additionalContext: ["enable_thinking": false],
            addGenerationPrompt: true
        )
        let renderedThinkingOn = try renderer.render(
            messages: messages,
            tools: nil,
            additionalContext: ["enable_thinking": true],
            addGenerationPrompt: true
        )

        print("[qwen35-4b][render][default]=\(quoted(renderedDefault))")
        print("[qwen35-4b][render][thinking=false]=\(quoted(renderedThinkingOff))")
        print("[qwen35-4b][render][thinking=true]=\(quoted(renderedThinkingOn))")
    }

    @Test("0.8B and 4B tensor directory shapes for DeltaNet are inspectable")
    func tensorDirectoryShapes() async throws {
        let models: [(String, URL)] = try await [
            (
                "0.8B",
                HuggingFaceDownloader().download(
                    repo: "unsloth/Qwen3.5-0.8B-GGUF",
                    filename: "Qwen3.5-0.8B-Q4_K_M.gguf"
                )
            ),
            ("4B", downloadModel()),
        ].asyncMap { label, url in
            (label, try await url)
        }

        let interestingNames = [
            "blk.0.attn_qkv.weight",
            "blk.0.attn_gate.weight",
            "blk.0.ssm_beta.weight",
            "blk.0.ssm_alpha.weight",
            "blk.0.ssm_conv1d.weight",
            "blk.0.ssm_dt.bias",
            "blk.0.ssm_a",
            "blk.0.ssm_norm.weight",
            "blk.0.ssm_out.weight",
        ]

        for (label, url) in models {
            let file = try GGUFFile.parse(url: url)
            print("[qwen35-shapes][\(label)][metadata] attention.key_length=\(file.headDimension ?? -1) ssm.group_count=\(file.ssmGroupCount ?? -1) ssm.time_step_rank=\(file.ssmNumHeads ?? -1) ssm.state_size=\(file.ssmStateSize ?? -1) ssm.inner_size=\(file.ssmInnerSize ?? -1)")

            for name in interestingNames {
                let tensor = try #require(file.tensors.first { $0.name == name })
                print("[qwen35-shapes][\(label)] \(name) dims=\(tensor.dimensions) q=\(tensor.quantizationType)")
            }
        }
    }

    @Test("0.8B and 4B full-attention tensor layout is inspectable")
    func fullAttentionTensorShapes() async throws {
        let smallURL = try repairedURLIfNeeded(sourceURL: try await downloadSmallModel())
        let largeURL = try repairedURLIfNeeded(sourceURL: try await downloadModel())
        let models = [("0.8B", smallURL), ("4B", largeURL)]

        let interestingNames = [
            "blk.3.attn_q.weight",
            "blk.3.attn_k.weight",
            "blk.3.attn_v.weight",
            "blk.3.attn_output.weight",
            "blk.3.attn_q_norm.weight",
            "blk.3.attn_k_norm.weight",
            "blk.3.attn_gate.weight",
        ]

        for (label, url) in models {
            let file = try GGUFFile.parse(url: url)
            for name in interestingNames {
                if let tensor = file.tensors.first(where: { $0.name == name }) {
                    print("[qwen35-full-attn][\(label)] \(name) dims=\(tensor.dimensions) q=\(tensor.quantizationType)")
                } else {
                    print("[qwen35-full-attn][\(label)] \(name) missing")
                }
            }
        }
    }

    @Test("Compiled and standard 4B prefill logits are comparable on repaired GGUF")
    func compiledVsStandardPrefillComparison() async throws {
        let url = try await downloadModel()
        let loader = GGUFModelLoader()
        let compiled = try loader.loadCompiledContext(url: url)
        let standard = try loader.loadContext(url: url)

        let compiledInput = try await compiled.processor.prepare(input: UserInput(prompt: "hi"))
        let standardInput = try await standard.processor.prepare(input: UserInput(prompt: "hi"))

        let compiledOutput = try runPrefill(
            model: compiled.model,
            input: compiledInput,
            cache: compiled.model.newCache(parameters: nil),
            windowSize: 1024
        )
        let standardOutput = try runPrefill(
            model: standard.model,
            input: standardInput,
            cache: standard.model.newCache(parameters: nil),
            windowSize: 1024
        )

        let compiledLogits = nextTokenLogits(from: compiledOutput)
        let standardLogits = nextTokenLogits(from: standardOutput)

        printTopCandidates(
            label: "qwen35-4b/compiled/prefill-repaired",
            logits: compiledLogits,
            tokenizer: compiled.tokenizer,
            limit: 10
        )
        printTopCandidates(
            label: "qwen35-4b/standard/prefill-repaired",
            logits: standardLogits,
            tokenizer: standard.tokenizer,
            limit: 10
        )

        let compiledTop = topTokenIDs(from: compiledLogits, limit: 10)
        let standardTop = topTokenIDs(from: standardLogits, limit: 10)
        let overlap = Set(compiledTop).intersection(Set(standardTop))

        print("[qwen35-4b][prefill-compare] compiledTop=\(compiledTop)")
        print("[qwen35-4b][prefill-compare] standardTop=\(standardTop)")
        print("[qwen35-4b][prefill-compare] overlap=\(overlap.sorted()) count=\(overlap.count)")
    }

    @Test("Repaired 4B preserves tensor directory and sampled tensor payloads")
    func repairedFilePreservesTensorPayload() async throws {
        let originalURL = try originalModelURL()
        let repairedURL = try await downloadModel()

        let original = try GGUFFile.parse(url: originalURL)
        let repaired = try GGUFFile.parse(url: repairedURL)

        #expect(repaired.tensors.count == original.tensors.count)
        #expect(repaired.tensors.map(\.name) == original.tensors.map(\.name))
        #expect(repaired.tensors.map(\.quantizationType) == original.tensors.map(\.quantizationType))
        #expect(repaired.tensors.map(\.dimensions) == original.tensors.map(\.dimensions))
        #expect(repaired.tensors.map(\.offset) == original.tensors.map(\.offset))

        let sampledNames = [
            "token_embd.weight",
            "blk.0.attn_qkv.weight",
            "blk.3.attn_q.weight",
            "blk.16.ssm_out.weight",
            "blk.31.ffn_up.weight",
        ]

        for name in sampledNames {
            let originalTensor = try #require(original.tensors.first { $0.name == name })
            let repairedTensor = try #require(repaired.tensors.first { $0.name == name })
            let originalData = try original.tensorData(for: originalTensor)
            let repairedData = try repaired.tensorData(for: repairedTensor)

            #expect(originalData == repairedData)
            print("[qwen35-4b][repair-check] \(name) bytes=\(originalData.count) preserved=true")
        }
    }

    @Test("4B thinking=true changes prompt tail and prefill candidates")
    func compiledPrefillWithThinkingEnabled() async throws {
        let url = try await downloadModel()
        let loader = GGUFModelLoader()
        let context = try loader.loadCompiledContext(url: url)

        let defaultInput = try await context.processor.prepare(input: UserInput(prompt: "hi"))
        let thinkingInput = try await context.processor.prepare(
            input: UserInput(
                chat: [.user("hi")],
                additionalContext: ["enable_thinking": true]
            )
        )

        let defaultTokens = tokenIDs(from: defaultInput.text.tokens)
        let thinkingTokens = tokenIDs(from: thinkingInput.text.tokens)
        let defaultTail = Array(defaultTokens.suffix(16))
        let thinkingTail = Array(thinkingTokens.suffix(16))
        let defaultTailPieces = defaultTail.map { context.tokenizer.tokenToString($0) ?? "<?>" }
        let thinkingTailPieces = thinkingTail.map { context.tokenizer.tokenToString($0) ?? "<?>" }

        print("[qwen35-4b][thinking][default] tokens=\(defaultTokens.count) tail=\(defaultTailPieces)")
        print("[qwen35-4b][thinking][enabled] tokens=\(thinkingTokens.count) tail=\(thinkingTailPieces)")

        let defaultOutput = try runPrefill(
            model: context.model,
            input: defaultInput,
            cache: context.model.newCache(parameters: nil),
            windowSize: 1024
        )
        let thinkingOutput = try runPrefill(
            model: context.model,
            input: thinkingInput,
            cache: context.model.newCache(parameters: nil),
            windowSize: 1024
        )

        let defaultLogits = nextTokenLogits(from: defaultOutput)
        let thinkingLogits = nextTokenLogits(from: thinkingOutput)

        printTopCandidates(
            label: "qwen35-4b/default-thinking/prefill",
            logits: defaultLogits,
            tokenizer: context.tokenizer,
            limit: 10
        )
        printTopCandidates(
            label: "qwen35-4b/enabled-thinking/prefill",
            logits: thinkingLogits,
            tokenizer: context.tokenizer,
            limit: 10
        )
    }

    @Test("4B thinking=true decode trace is inspectable")
    func compiledDecodeTraceWithThinkingEnabled() async throws {
        let url = try await downloadModel()
        let loader = GGUFModelLoader()
        let context = try loader.loadCompiledContext(url: url)

        let input = try await context.processor.prepare(
            input: UserInput(
                chat: [.user("hi")],
                additionalContext: ["enable_thinking": true]
            )
        )

        let cache = context.model.newCache(parameters: nil)
        let prefillOutput = try runPrefill(
            model: context.model,
            input: input,
            cache: cache,
            windowSize: 1024
        )

        let logits = nextTokenLogits(from: prefillOutput)
        printTopCandidates(
            label: "qwen35-4b/enabled-thinking/decode-start",
            logits: logits,
            tokenizer: context.tokenizer,
            limit: 10
        )

        let sampler = ArgMaxSampler()
        var tokenID = Int(sampler.sample(logits: logits).item(Int32.self))
        var generated: [Int] = []

        for step in 0..<8 {
            generated.append(tokenID)
            let piece = context.tokenizer.tokenToString(tokenID) ?? "<??>"
            let decoded = context.tokenizer.decode(tokens: [tokenID])
            print(
                "[qwen35-4b][thinking-enabled][decode-step] step=\(step + 1) sampledID=\(tokenID) piece=\(quoted(piece)) decoded=\(quoted(decoded))"
            )

            let decodeInput = LMInput.Text(
                tokens: MLXArray([Int32(tokenID)]).reshaped([1, 1])
            )
            let decodeOutput = context.model.callAsFunction(decodeInput, cache: cache, state: nil)
            let nextLogits = nextTokenLogits(from: decodeOutput)
            tokenID = Int(sampler.sample(logits: nextLogits).item(Int32.self))
        }

        let generatedText = context.tokenizer.decode(tokens: generated)
        print("[qwen35-4b][thinking-enabled][decode-trace] tokens=\(generated)")
        print("[qwen35-4b][thinking-enabled][decode-trace] text=\(quoted(generatedText))")
    }

    @Test("JARDIS app prompt differentiates 0.8B and 4B")
    func jardisAppPromptComparison() async throws {
        let smallURL = try repairedURLIfNeeded(sourceURL: try await downloadSmallModel())
        let largeURL = try repairedURLIfNeeded(sourceURL: try await downloadModel())
        let modelURLs = [("0.8B", smallURL), ("4B", largeURL)]

        let userInput = UserInput(
            chat: [
                .system(makeJardisAppSystemPrompt()),
                .user("hi"),
            ],
            additionalContext: ["enable_thinking": true]
        )

        var renderedPrompts: [String: String] = [:]
        var tokenizedPrompts: [String: [Int]] = [:]

        for (label, url) in modelURLs {
            let file = try GGUFFile.parse(url: url)
            let loader = GGUFModelLoader()
            let context = try loader.loadCompiledContext(url: url)

            let rendered = try renderPrompt(
                file: file,
                tokenizer: context.tokenizer,
                input: userInput
            )
            renderedPrompts[label] = rendered

            let lmInput = try await context.processor.prepare(input: userInput)
            let promptTokens = tokenIDs(from: lmInput.text.tokens)
            tokenizedPrompts[label] = promptTokens

            print("[jardis-prompt][\(label)] chars=\(rendered.count) tokens=\(promptTokens.count)")
            print("[jardis-prompt][\(label)] tail=\(Array(promptTokens.suffix(16)).map { context.tokenizer.tokenToString($0) ?? "<?>" })")

            let cache = context.model.newCache(parameters: nil)
            let output = try runPrefill(
                model: context.model,
                input: lmInput,
                cache: cache,
                windowSize: 1024
            )
            let logits = nextTokenLogits(from: output)
            printTopCandidates(
                label: "jardis/\(label)/prefill",
                logits: logits,
                tokenizer: context.tokenizer,
                limit: 10
            )

            let sampler = ArgMaxSampler()
            var tokenID = Int(sampler.sample(logits: logits).item(Int32.self))
            var generated: [Int] = []
            for _ in 0..<12 {
                generated.append(tokenID)
                let decodeInput = LMInput.Text(tokens: MLXArray([Int32(tokenID)]).reshaped([1, 1]))
                let decodeOutput = context.model.callAsFunction(decodeInput, cache: cache, state: nil)
                tokenID = Int(sampler.sample(logits: nextTokenLogits(from: decodeOutput)).item(Int32.self))
            }
            let text = context.tokenizer.decode(tokens: generated)
            print("[jardis-prompt][\(label)] decode=\(quoted(text))")
        }

        if let smallRendered = renderedPrompts["0.8B"], let largeRendered = renderedPrompts["4B"] {
            #expect(smallRendered == largeRendered)
        }
        if let smallTokens = tokenizedPrompts["0.8B"], let largeTokens = tokenizedPrompts["4B"] {
            #expect(smallTokens == largeTokens)
            #expect(smallTokens.count > 900)
        }
    }

    @Test("JARDIS app prompt layer-wise logits reveal 4B divergence point")
    func jardisLayerwiseLogitTrace() async throws {
        let smallURL = try repairedURLIfNeeded(sourceURL: try await downloadSmallModel())
        let largeURL = try repairedURLIfNeeded(sourceURL: try await downloadModel())
        let userInput = UserInput(
            chat: [
                .system(makeJardisAppSystemPrompt()),
                .user("hi"),
            ],
            additionalContext: ["enable_thinking": true]
        )

        try await traceLayerwiseLogits(label: "0.8B", url: smallURL, userInput: userInput)
        try await traceLayerwiseLogits(label: "4B", url: largeURL, userInput: userInput)
    }

    @Test("JARDIS app prompt first DeltaNet layer components reveal the broken branch")
    func jardisFirstLayerComponentComparison() async throws {
        let smallURL = try repairedURLIfNeeded(sourceURL: try await downloadSmallModel())
        let largeURL = try repairedURLIfNeeded(sourceURL: try await downloadModel())
        let userInput = UserInput(
            chat: [
                .system(makeJardisAppSystemPrompt()),
                .user("hi"),
            ],
            additionalContext: ["enable_thinking": true]
        )

        try await traceFirstLayerComponents(label: "0.8B", url: smallURL, userInput: userInput)
        try await traceFirstLayerComponents(label: "4B", url: largeURL, userInput: userInput)
    }

    private func renderPrompt(
        file: GGUFFile,
        tokenizer: any Tokenizer,
        input: UserInput
    ) throws -> String {
        let template = try #require(file.chatTemplate)
        let renderer = try ChatTemplateRenderer(
            templateString: template,
            bosToken: tokenizer.bosTokenID.flatMap { tokenizer.tokenToString($0) },
            eosToken: tokenizer.eosTokenID.flatMap { tokenizer.tokenToString($0) }
        )
        return try renderer.render(
            messages: input.chat,
            tools: input.tools,
            additionalContext: input.additionalContext,
            addGenerationPrompt: true
        )
    }

    private func traceLayerwiseLogits(
        label: String,
        url: URL,
        userInput: UserInput
    ) async throws {
        let loader = GGUFModelLoader()
        let context = try loader.loadContext(url: url)
        guard let model = context.model as? HybridDeltaNetAttentionModel else {
            Issue.record("[\(label)] Expected HybridDeltaNetAttentionModel")
            return
        }

        let lmInput = try await context.processor.prepare(input: userInput)
        let inputTokens = tokenIDs(from: lmInput.text.tokens)
        print("[layer-trace][\(label)] promptTokens=\(inputTokens.count)")

        var hidden = model.model.embedTokens(lmInput.text.tokens)
        printActivationStats(label: "layer-trace/\(label)/embed", tensor: hidden)

        let selectedIndices = selectedLayerIndices(
            totalCount: model.model.layers.count,
            fullAttentionInterval: model.configuration.fullAttentionInterval
        )

        for (layerIndex, layer) in model.model.layers.enumerated() {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            if layer.isFullAttention && hidden.dim(1) > 1 {
                mask = .causal
            } else {
                mask = .none
            }

            hidden = layer(hidden, mask: mask, cache: nil, positionIds: nil)

            guard selectedIndices.contains(layerIndex) else {
                continue
            }

            eval(hidden)
            let kind = layer.isFullAttention ? "full-attn" : "deltanet"
            printActivationStats(
                label: "layer-trace/\(label)/layer-\(layerIndex)/\(kind)/hidden",
                tensor: hidden
            )

            let logits = intermediateLogits(model: model, hidden: hidden)
            printTopCandidates(
                label: "layer-trace/\(label)/layer-\(layerIndex)/\(kind)",
                logits: logits,
                tokenizer: context.tokenizer,
                limit: 10
            )
        }
    }

    private func traceFirstLayerComponents(
        label: String,
        url: URL,
        userInput: UserInput
    ) async throws {
        let loader = GGUFModelLoader()
        let context = try loader.loadContext(url: url)
        guard let model = context.model as? HybridDeltaNetAttentionModel else {
            Issue.record("[\(label)] Expected HybridDeltaNetAttentionModel")
            return
        }
        let layer = try #require(model.model.layers.first)
        guard let deltaNet = layer.deltaNet else {
            Issue.record("[\(label)] Expected first layer to be DeltaNet")
            return
        }

        let lmInput = try await context.processor.prepare(input: userInput)
        let embedded = model.model.embedTokens(lmInput.text.tokens)
        let normedInput = layer.inputLayerNorm(embedded)
        let deltaBody = deltaNet(normedInput, cache: nil)
        let residual = embedded + deltaBody
        let mlpBody = layer.mlp(layer.postAttentionLayerNorm(residual))
        let output = residual + mlpBody

        printActivationStats(label: "layer-components/\(label)/embed", tensor: embedded)
        printActivationStats(label: "layer-components/\(label)/input-norm", tensor: normedInput)
        printActivationStats(label: "layer-components/\(label)/delta-body", tensor: deltaBody)
        printActivationStats(label: "layer-components/\(label)/after-delta", tensor: residual)
        printActivationStats(label: "layer-components/\(label)/mlp-body", tensor: mlpBody)
        printActivationStats(label: "layer-components/\(label)/layer-output", tensor: output)

        let afterDeltaLogits = intermediateLogits(model: model, hidden: residual)
        let outputLogits = intermediateLogits(model: model, hidden: output)

        printTopCandidates(
            label: "layer-components/\(label)/after-delta",
            logits: afterDeltaLogits,
            tokenizer: context.tokenizer,
            limit: 10
        )
        printTopCandidates(
            label: "layer-components/\(label)/layer-output",
            logits: outputLogits,
            tokenizer: context.tokenizer,
            limit: 10
        )
    }

    private func intermediateLogits(
        model: HybridDeltaNetAttentionModel,
        hidden: MLXArray
    ) -> MLXArray {
        let normalized = model.model.norm(hidden)
        let logits: MLXArray
        if let lmHead = model.lmHead {
            logits = lmHead(normalized)
        } else {
            logits = model.model.embedTokens.asLinear(normalized)
        }
        return nextTokenLogits(from: LMOutput(logits: logits))
    }

    private func selectedLayerIndices(
        totalCount: Int,
        fullAttentionInterval: Int
    ) -> Set<Int> {
        var indices: Set<Int> = [0, 1, 2, 3, totalCount - 1]
        for index in 0..<totalCount where (index + 1).isMultiple(of: fullAttentionInterval) {
            indices.insert(index)
        }
        return indices.filter { $0 >= 0 && $0 < totalCount }
    }

    private func makeJardisAppSystemPrompt(now: Date = Date()) -> String {
        let behavior = """
            # Behavior
            - Use available tools proactively to gather information. Verify rather than guess.
            - Understand the target before acting. Do not modify without first confirming the current state.
            - Focus on what was requested. Do not make unrequested changes or additions.
            - Confirm with the user before taking actions that are difficult to reverse.
            - When an approach fails, consider alternatives instead of retrying the same action.
            - Respond in the same language as the user's message.
            - Be concise, but do not omit necessary information.
            """

        let coreUISchema = """
            # CoreUI Artifact

            To display rich views, wrap a CoreUI JSON document in `<artifact type="KIND">JSON</artifact>`.
            KIND must match the view kind (map, image, places, calendar, health).

            ## Document

            - schemaVersion (string): Always "1.1".
            - message (string): A short one-line summary.
            - ui (object): layout (string, always "v"), views (array of view items).

            ## View Item

            - kind (string): "map", "image", "places", "calendar", or "health".
            - payload (object): Data specific to the kind.

            ## Coordinate

            Shared object: { "lat": number, "lng": number }.

            ## kind: "map" — Map Snapshot

            - center (Coordinate): Map center.
            - pins (array): [{ id, title, coord (Coordinate) }].
            - summary (array of string, optional): Lines below the map.
            - geofenceRadiusMeters (number, optional): Circle overlay radius.
            - route (object, optional): { originLabel, destinationLabel, distanceM (number), etaMin (integer) }.

            ## kind: "map" — Map Route

            When displaying a route with a polyline path, use this format instead.

            - path (array of Coordinate): Route polyline. Must have >= 2 points.
            - origin (object, optional): { id, title, coord }.
            - destination (object, optional): { id, title, coord }.
            - waypoints (array, optional): [{ id, title, coord }].
            - route (object, optional): { originLabel, destinationLabel, distanceM, etaMin }.
            - transport (string, optional): Transport mode.
            - steps (array, optional): [{ stepID, text, distanceM (optional) }].
            - summary (array of string, optional): Lines below the map.

            ## kind: "image" — Single Image

            - url (string): Image URL. Required if no placeholder.
            - placeholder (string, optional): SF Symbol name as fallback.
            - title (string, optional): Title text.
            - subtitle (string, optional): Description.
            - meta (object, optional): Key-value string pairs displayed as metadata grid.

            ## kind: "image" — Image Gallery

            Use when there are multiple images.

            - images (array): [{ id, url, title (optional), caption (optional) }].

            ## kind: "places" — Place List

            - places (array): [{ id, name, address, category (optional), phone (optional), coordinate (Coordinate, optional) }].

            ## kind: "calendar" — Calendar Timeline

            ALWAYS render this artifact after calling calendar_list_events, even when events is empty.

            - timezone (string, optional): IANA timezone identifier.
            - events (array, may be empty): [{ id (optional, auto-generated), title, start (ISO 8601), end (ISO 8601), location (optional), travelMin (integer, optional), conflict (boolean, default false) }].

            ## kind: "health" — Health Trend

            - period (string, optional): Period label (e.g. "Past 7 days").
            - metrics (array): [{ id (optional), label, unit, current (number), prev (number), series (array of number) }].
            - alerts (array of string, optional): Alert messages.
            """

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        formatter.timeZone = TimeZone.current
        let datetime = formatter.string(from: now)
        let timezone = TimeZone.current.identifier
        let environment = """
            <env>
            Datetime: \(datetime) (\(timezone))
            </env>
            """

        return """
            You are Jardis.

            \(behavior)
            # Rich view rendering
            - Focus on solving the user's request with tool calls and concise text.
            - Tool results are automatically rendered as rich views in the UI. Do not repeat information that will be displayed visually.
            - Do not call view-specific formatter/validator tools.
            - When your response includes visual data (coordinates, image URLs), embed a CoreUI artifact in your response.

            \(coreUISchema)

            \(environment)
            """
    }

    // MARK: - Layer-wise Activation Instrumentation

    @Test("4B layer-wise activation statistics reveal divergence point")
    func layerWiseActivationStats() async throws {
        let url = try await downloadModel()
        let loader = GGUFModelLoader()
        let context = try loader.loadCompiledContext(url: url)

        guard let compiled = context.model as? CompiledLanguageModel else {
            Issue.record("Model is not CompiledLanguageModel")
            return
        }

        let input = try await context.processor.prepare(
            input: UserInput(
                chat: [
                    .system(makeJardisAppSystemPrompt()),
                    .user("hi"),
                ],
                additionalContext: ["enable_thinking": true]
            )
        )

        var state = compiled.lowered.makeState()
        let options = ExecutionOptions()
        let steps = compiled.lowered.prefill.steps

        var h = input.text.tokens
        var layerIndex = 0

        for (stepIndex, step) in steps.enumerated() {
            switch step {
            case .op(let op):
                h = executeOp(op, input: h, state: &state, options: options)
                eval(h)
                let label = opLabel(op)
                printActivationStats(
                    label: "step-\(stepIndex)/\(label)",
                    tensor: h
                )

            case .residual(let body):
                let residualInput = h
                let bodyOut = executeSteps(body, input: h, state: &state, options: options)
                h = residualInput + bodyOut
                eval(h)

                // Determine layer type from body contents
                let kind = residualLayerKind(body)
                printActivationStats(
                    label: "step-\(stepIndex)/layer-\(layerIndex)/\(kind)/residual-out",
                    tensor: h
                )
                printActivationStats(
                    label: "step-\(stepIndex)/layer-\(layerIndex)/\(kind)/body-out",
                    tensor: bodyOut
                )
                layerIndex += 1

            case .parallel(let merge, let branches):
                let results = branches.map { branch in
                    executeSteps(branch, input: h, state: &state, options: options)
                }
                h = mergeResults(results, strategy: merge)
                eval(h)
                printActivationStats(
                    label: "step-\(stepIndex)/parallel",
                    tensor: h
                )
            }
        }

        let seqLen = input.text.tokens.dim(input.text.tokens.ndim - 1)
        state.nextPosition += seqLen

        // Final logits analysis
        let logits = h[0..., (-1)..., 0...].squeezed(axis: 0)
        printTopCandidates(
            label: "qwen35-4b/instrumented/prefill",
            logits: logits,
            tokenizer: context.tokenizer,
            limit: 10
        )
    }

    private func opLabel(_ op: LoweredInferenceOp) -> String {
        switch op {
        case .tokenEmbedding: return "embedding"
        case .attention: return "attention"
        case .mlp: return "mlp"
        case .moe: return "moe"
        case .norm: return "norm"
        case .outputHead: return "output-head"
        case .deltaNet: return "deltanet"
        case .rope: return "rope"
        case .positionalEmbedding: return "pos-emb"
        case .linear: return "linear"
        }
    }

    private func residualLayerKind(_ body: [LoweredStep]) -> String {
        for step in body {
            switch step {
            case .op(let op):
                switch op {
                case .attention: return "attn"
                case .deltaNet: return "dn"
                case .mlp: return "mlp"
                case .moe: return "moe"
                default: continue
                }
            case .residual(let inner):
                let inner = residualLayerKind(inner)
                if inner != "unknown" { return inner }
            case .parallel:
                continue
            }
        }
        return "unknown"
    }

    private func printActivationStats(label: String, tensor: MLXArray) {
        let flat = tensor.flattened().asType(.float32)
        eval(flat)
        let values = flat.asArray(Float.self)
        guard !values.isEmpty else {
            print("[activation][\(label)] empty")
            return
        }

        let count = values.count
        var sum: Double = 0
        var sumSq: Double = 0
        var absMax: Float = 0
        var nanCount = 0
        var infCount = 0

        for v in values {
            if v.isNaN { nanCount += 1; continue }
            if !v.isFinite { infCount += 1; continue }
            let d = Double(v)
            sum += d
            sumSq += d * d
            let a = abs(v)
            if a > absMax { absMax = a }
        }

        let finiteCount = count - nanCount - infCount
        let mean = finiteCount > 0 ? sum / Double(finiteCount) : .nan
        let variance = finiteCount > 0 ? sumSq / Double(finiteCount) - mean * mean : .nan
        let std = variance >= 0 ? variance.squareRoot() : .nan

        print("[activation][\(label)] shape=\(tensor.shape) mean=\(String(format: "%.6f", mean)) std=\(String(format: "%.6f", std)) absMax=\(String(format: "%.4f", absMax)) nan=\(nanCount) inf=\(infCount) count=\(count)")
    }

    private func runPrefill(
        model: any LanguageModel,
        input: LMInput,
        cache: [KVCache],
        windowSize: Int
    ) throws -> LMOutput {
        var currentInput = input

        while true {
            let result = try model.prepare(currentInput, cache: cache, windowSize: windowSize)
            switch result {
            case .tokens(let remaining):
                currentInput = LMInput(
                    text: remaining,
                    image: currentInput.image,
                    video: currentInput.video
                )
            case .logits(let output):
                return output
            }
        }
    }

    private func nextTokenLogits(from output: LMOutput) -> MLXArray {
        output.logits[0..., (-1)..., 0...].squeezed(axis: 0)
    }

    private func tokenIDs(from tokens: MLXArray) -> [Int] {
        tokens.flattened().asArray(Int32.self).map(Int.init)
    }

    private func topTokenIDs(from logits: MLXArray, limit: Int) -> [Int] {
        let values = logits.flattened().asType(.float32).asArray(Float.self)
        return values.enumerated()
            .sorted { $0.element > $1.element }
            .prefix(limit)
            .map(\.offset)
    }

    private func templateSnippet(_ template: String) -> String {
        guard let range = template.range(of: "think") else {
            return String(template.prefix(240))
        }
        let lower = template.index(range.lowerBound, offsetBy: -120, limitedBy: template.startIndex)
            ?? template.startIndex
        let upper = template.index(range.upperBound, offsetBy: 120, limitedBy: template.endIndex)
            ?? template.endIndex
        return String(template[lower..<upper])
    }

    private func printTopCandidates(
        label: String,
        logits: MLXArray,
        tokenizer: any Tokenizer,
        limit: Int
    ) {
        let flatLogits = logits.flattened().asType(.float32)
        let sorted = MLX.argSort(flatLogits, axis: -1)
        eval(flatLogits, sorted)

        let values = flatLogits.asArray(Float.self)
        let topIDs = Array(sorted.asArray(Int32.self).suffix(limit).reversed()).map(Int.init)

        print("[\(label)][topk] count=\(topIDs.count)")
        for (rank, tokenID) in topIDs.enumerated() {
            let piece = tokenizer.tokenToString(tokenID) ?? "<??>"
            let decoded = tokenizer.decode(tokens: [tokenID])
            let value = values[tokenID]
            print(
                "[\(label)][topk] rank=\(rank + 1) id=\(tokenID) logit=\(value) piece=\(quoted(piece)) decoded=\(quoted(decoded))"
            )
        }
    }

    @Test("4B reference vs compiled path output comparison")
    func referenceVsCompiledComparison() async throws {
        let url = try repairedURLIfNeeded(sourceURL: try await downloadModel())
        let loader = GGUFModelLoader()

        // Use the jardis prompt (1015 tokens) to test long-sequence behavior
        let prompt = UserInput(
            chat: [
                .system(makeJardisAppSystemPrompt()),
                .user("hi"),
            ],
            additionalContext: ["enable_thinking": true]
        )

        // Reference path
        let refContext = try loader.loadContext(url: url)
        let refInput = try await refContext.processor.prepare(input: prompt)
        print("[4B/refVsComp] prompt tokens=\(refInput.text.tokens.dim(1))")

        let refCache = refContext.model.newCache(parameters: nil)
        let refOutput = try runPrefill(
            model: refContext.model,
            input: refInput,
            cache: refCache,
            windowSize: 1024
        )
        let refLogits = nextTokenLogits(from: refOutput)
        printTopCandidates(
            label: "4B/reference/jardis",
            logits: refLogits,
            tokenizer: refContext.tokenizer,
            limit: 10
        )

        let sampler = ArgMaxSampler()
        var refTokenID = Int(sampler.sample(logits: refLogits).item(Int32.self))
        var refGenerated: [Int] = []
        for _ in 0..<12 {
            refGenerated.append(refTokenID)
            let decodeInput = LMInput.Text(tokens: MLXArray([Int32(refTokenID)]).reshaped([1, 1]))
            let decodeOutput = refContext.model.callAsFunction(decodeInput, cache: refCache, state: nil)
            refTokenID = Int(sampler.sample(logits: nextTokenLogits(from: decodeOutput)).item(Int32.self))
        }
        let refText = refContext.tokenizer.decode(tokens: refGenerated)
        print("[4B/reference/jardis] decode=\(quoted(refText))")

        // Compiled path
        let compContext = try loader.loadCompiledContext(url: url)
        let compInput = try await compContext.processor.prepare(input: prompt)
        let compCache = compContext.model.newCache(parameters: nil)
        let compOutput = try runPrefill(
            model: compContext.model,
            input: compInput,
            cache: compCache,
            windowSize: 1024
        )
        let compLogits = nextTokenLogits(from: compOutput)
        printTopCandidates(
            label: "4B/compiled/jardis",
            logits: compLogits,
            tokenizer: compContext.tokenizer,
            limit: 10
        )

        var compTokenID = Int(sampler.sample(logits: compLogits).item(Int32.self))
        var compGenerated: [Int] = []
        for _ in 0..<12 {
            compGenerated.append(compTokenID)
            let decodeInput = LMInput.Text(tokens: MLXArray([Int32(compTokenID)]).reshaped([1, 1]))
            let decodeOutput = compContext.model.callAsFunction(decodeInput, cache: compCache, state: nil)
            compTokenID = Int(sampler.sample(logits: nextTokenLogits(from: decodeOutput)).item(Int32.self))
        }
        let compText = compContext.tokenizer.decode(tokens: compGenerated)
        print("[4B/compiled/jardis] decode=\(quoted(compText))")

        // Compare
        let refTopIDs = topTokenIDs(from: refLogits, limit: 5)
        let compTopIDs = topTokenIDs(from: compLogits, limit: 5)
        print("[4B/jardis/comparison] ref top5=\(refTopIDs) comp top5=\(compTopIDs)")
        print("[4B/jardis/comparison] ref text=\(quoted(refText))")
        print("[4B/jardis/comparison] comp text=\(quoted(compText))")
    }

    @Test("4B short vs long prompt (reference path)")
    func shortVsLongPromptReference() async throws {
        let url = try repairedURLIfNeeded(sourceURL: try await downloadModel())
        let loader = GGUFModelLoader()
        let context = try loader.loadContext(url: url)

        // Short prompt: stays within single DeltaNet chunk (< 64 tokens)
        let shortPrompt = "Hello, how are you?"
        // Long prompt: triggers multi-chunk DeltaNet (> 64 tokens)
        let longPrompt = String(repeating: "The quick brown fox jumps over the lazy dog. ", count: 20)

        for (promptLabel, text) in [("short", shortPrompt), ("long", longPrompt)] {
            let prompt = UserInput(prompt: text)
            let input = try await context.processor.prepare(input: prompt)
            let tokenCount = input.text.tokens.dim(1)
            let tokenIDs = input.text.tokens.squeezed(axis: 0).asArray(Int32.self)
            let first10 = tokenIDs.prefix(10).map { String($0) }.joined(separator: ", ")
            let last10 = tokenIDs.suffix(10).map { String($0) }.joined(separator: ", ")
            print("[4B/ref/\(promptLabel)] tokens=\(tokenCount) first10=[\(first10)] last10=[\(last10)]")
            // Decode the last few tokens to verify chat template suffix
            let lastTokenTexts = tokenIDs.suffix(5).map { context.tokenizer.decode(tokens: [Int($0)]) }
            print("[4B/ref/\(promptLabel)] lastTokenTexts=\(lastTokenTexts)")

            let cache = context.model.newCache(parameters: nil)
            let output = try runPrefill(
                model: context.model,
                input: input,
                cache: cache,
                windowSize: 1024
            )
            let logits = nextTokenLogits(from: output)
            printTopCandidates(
                label: "4B/ref/\(promptLabel)",
                logits: logits,
                tokenizer: context.tokenizer,
                limit: 10
            )

            let sampler = ArgMaxSampler()
            var tokenID = Int(sampler.sample(logits: logits).item(Int32.self))
            var generated: [Int] = []
            for _ in 0..<12 {
                generated.append(tokenID)
                let decodeInput = LMInput.Text(tokens: MLXArray([Int32(tokenID)]).reshaped([1, 1]))
                let decodeOutput = context.model.callAsFunction(decodeInput, cache: cache, state: nil)
                tokenID = Int(sampler.sample(logits: nextTokenLogits(from: decodeOutput)).item(Int32.self))
            }
            let decoded = context.tokenizer.decode(tokens: generated)
            print("[4B/ref/\(promptLabel)] decode=\(quoted(decoded))")
        }
    }

    @Test("Chunked vs sequential recurrence parity for asymmetric heads")
    func chunkedVsSequentialRecurrenceParity() async throws {
        // Test the chunked WY recurrence against token-by-token recurrence
        // using 4B-like dimensions: 32 heads with Q/K repeated from 16 groups
        let B = 1
        let T = 192  // 3 chunks of 64 — enough to test multi-chunk
        let H = 32   // numHeads (value heads)
        let keyGroups = 16
        let dk = 128
        let dv = 128
        let repeatFactor = H / keyGroups
        let dtype: DType = .float32

        // Generate random inputs matching 4B DeltaNet dimensions
        MLXRandom.seed(42)
        let rawQ = MLXRandom.normal([B, T, keyGroups, dk]).asType(dtype)
        let rawK = MLXRandom.normal([B, T, keyGroups, dk]).asType(dtype)
        let value = MLXRandom.normal([B, T, H, dv]).asType(dtype)
        let gateLog = MLXRandom.uniform(-2.0 ..< -0.1, [B, T, H]).asType(dtype)
        let beta = MLX.sigmoid(MLXRandom.normal([B, T, H])).asType(dtype)
        let state = MLXArray.zeros([B, H, dk, dv], dtype: dtype)

        // Repeat Q/K from keyGroups to H (like 4B model does)
        let query = repeated(rawQ, count: repeatFactor, axis: 2)
        let key = repeated(rawK, count: repeatFactor, axis: 2)
        eval(query, key, value, gateLog, beta)

        let scale: Float = 1.0 / Float(dk).squareRoot()

        // L2-normalize Q and K, apply scale
        func l2Norm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
            x / MLX.sqrt((x * x).sum(axis: -1, keepDims: true) + MLXArray(eps))
        }
        let qN = l2Norm(query) * MLXArray(scale).asType(dtype)
        let kN = l2Norm(key)
        eval(qN, kN)

        // --- Sequential (token-by-token) recurrence ---
        let decay = MLX.exp(gateLog)
        var seqS = state
        var seqOutputs = [MLXArray]()
        for t in 0..<T {
            let qt = qN[0..., t..<(t + 1), 0..., 0...].squeezed(axis: 1)
            let kt = kN[0..., t..<(t + 1), 0..., 0...].squeezed(axis: 1)
            let vt = value[0..., t..<(t + 1), 0..., 0...].squeezed(axis: 1)
            let gt = decay[0..., t..<(t + 1), 0...].squeezed(axis: 1)
            let bt = beta[0..., t..<(t + 1), 0...].squeezed(axis: 1)

            let gE = gt.expandedDimensions(axis: -1).expandedDimensions(axis: -1)
            seqS = seqS * gE

            let kE = kt.expandedDimensions(axis: -1)
            let kvMem = (seqS * kE).sum(axis: -2)
            let delta = bt.expandedDimensions(axis: -1) * (vt - kvMem)
            seqS = seqS + kE * delta.expandedDimensions(axis: -2)

            let qE = qt.expandedDimensions(axis: -1)
            let ot = (seqS * qE).sum(axis: -2)
            seqOutputs.append(ot.expandedDimensions(axis: 1))

            if (t + 1) % 64 == 0 { eval(seqS) }
        }
        let seqOutput = concatenated(seqOutputs, axis: 1)  // [B, T, H, dv]
        eval(seqOutput, seqS)

        // --- Chunked WY recurrence ---
        let (chunkOutput, chunkState) = chunkedGatedDeltaNetRecurrence(
            query: qN, key: kN, value: value,
            gateLog: gateLog, beta: beta, state: state,
            chunkSize: 64, dtype: dtype
        )
        eval(chunkOutput, chunkState)

        // --- Compare outputs ---
        let seqFlat = seqOutput.flattened().asType(.float32).asArray(Float.self)
        let chunkFlat = chunkOutput.flattened().asType(.float32).asArray(Float.self)

        var maxDiff: Float = 0
        var sumDiffSq: Double = 0
        for i in 0..<seqFlat.count {
            let d = abs(seqFlat[i] - chunkFlat[i])
            maxDiff = max(maxDiff, d)
            sumDiffSq += Double(d * d)
        }
        let rmsDiff = Float(sqrt(sumDiffSq / Double(seqFlat.count)))

        print("[chunked-vs-seq] B=\(B) T=\(T) H=\(H) dk=\(dk) dv=\(dv) keyGroups=\(keyGroups)")
        print("[chunked-vs-seq] output maxDiff=\(maxDiff) rmsDiff=\(rmsDiff)")
        print("[chunked-vs-seq] output seqMean=\(seqFlat.reduce(0, +) / Float(seqFlat.count)) chunkMean=\(chunkFlat.reduce(0, +) / Float(chunkFlat.count))")

        // Compare states
        let seqStateFlat = seqS.flattened().asType(.float32).asArray(Float.self)
        let chunkStateFlat = chunkState.flattened().asType(.float32).asArray(Float.self)
        var stateMaxDiff: Float = 0
        for i in 0..<seqStateFlat.count {
            stateMaxDiff = max(stateMaxDiff, abs(seqStateFlat[i] - chunkStateFlat[i]))
        }
        print("[chunked-vs-seq] state maxDiff=\(stateMaxDiff)")

        // Also test symmetric case (should definitely match)
        let symQ = MLXRandom.normal([B, T, H, dk]).asType(dtype)
        let symK = MLXRandom.normal([B, T, H, dk]).asType(dtype)
        let symQN = l2Norm(symQ) * MLXArray(scale).asType(dtype)
        let symKN = l2Norm(symK)
        eval(symQN, symKN)

        var symSeqS = state
        var symSeqOutputs = [MLXArray]()
        let symDecay = MLX.exp(gateLog)
        for t in 0..<T {
            let qt = symQN[0..., t..<(t + 1), 0..., 0...].squeezed(axis: 1)
            let kt = symKN[0..., t..<(t + 1), 0..., 0...].squeezed(axis: 1)
            let vt = value[0..., t..<(t + 1), 0..., 0...].squeezed(axis: 1)
            let gt = symDecay[0..., t..<(t + 1), 0...].squeezed(axis: 1)
            let bt = beta[0..., t..<(t + 1), 0...].squeezed(axis: 1)

            let gE = gt.expandedDimensions(axis: -1).expandedDimensions(axis: -1)
            symSeqS = symSeqS * gE

            let kE = kt.expandedDimensions(axis: -1)
            let kvMem = (symSeqS * kE).sum(axis: -2)
            let delta = bt.expandedDimensions(axis: -1) * (vt - kvMem)
            symSeqS = symSeqS + kE * delta.expandedDimensions(axis: -2)

            let qE = qt.expandedDimensions(axis: -1)
            let ot = (symSeqS * qE).sum(axis: -2)
            symSeqOutputs.append(ot.expandedDimensions(axis: 1))

            if (t + 1) % 64 == 0 { eval(symSeqS) }
        }
        let symSeqOutput = concatenated(symSeqOutputs, axis: 1)
        eval(symSeqOutput, symSeqS)

        let (symChunkOutput, symChunkState) = chunkedGatedDeltaNetRecurrence(
            query: symQN, key: symKN, value: value,
            gateLog: gateLog, beta: beta, state: state,
            chunkSize: 64, dtype: dtype
        )
        eval(symChunkOutput, symChunkState)

        let symSeqFlat = symSeqOutput.flattened().asType(.float32).asArray(Float.self)
        let symChunkFlat = symChunkOutput.flattened().asType(.float32).asArray(Float.self)
        var symMaxDiff: Float = 0
        for i in 0..<symSeqFlat.count {
            symMaxDiff = max(symMaxDiff, abs(symSeqFlat[i] - symChunkFlat[i]))
        }
        print("[chunked-vs-seq] symmetric maxDiff=\(symMaxDiff)")

        // Expect reasonable agreement (allow some float32 accumulation error)
        #expect(maxDiff < 0.01, "Asymmetric chunked recurrence diverges from sequential: maxDiff=\(maxDiff)")
        #expect(symMaxDiff < 0.01, "Symmetric chunked recurrence diverges from sequential: maxDiff=\(symMaxDiff)")
    }

    private func quoted(_ string: String) -> String {
        var escaped = "\""
        for scalar in string.unicodeScalars {
            switch scalar.value {
            case 0x0A:
                escaped += "\\n"
            case 0x0D:
                escaped += "\\r"
            case 0x09:
                escaped += "\\t"
            case 0x22:
                escaped += "\\\""
            case 0x5C:
                escaped += "\\\\"
            default:
                escaped.unicodeScalars.append(scalar)
            }
        }
        escaped += "\""
        return escaped
    }
}

private extension Array {
    func asyncMap<T>(_ transform: (Element) async throws -> T) async rethrows -> [T] {
        var result: [T] = []
        result.reserveCapacity(count)
        for element in self {
            result.append(try await transform(element))
        }
        return result
    }
}
