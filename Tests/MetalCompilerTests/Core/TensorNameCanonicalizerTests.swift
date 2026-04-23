import Testing
@testable import MetalCompiler

@Suite("TensorNameCanonicalizer")
struct TensorNameCanonicalizerTests {

    @Test("identity leaves names unchanged")
    func identityPassesThrough() {
        let canonicalizer = TensorNameCanonicalizer.identity
        #expect(canonicalizer.canonicalize("model.language_model.embed_tokens.weight")
                == "model.language_model.embed_tokens.weight")
        #expect(canonicalizer.canonicalize("model.visual.patch_embed.weight")
                == "model.visual.patch_embed.weight")
        #expect(canonicalizer.canonicalize("lm_head.weight") == "lm_head.weight")
    }

    @Test("MLX VLM prefix is rewritten to HF form")
    func mlxVLMPrefixRewrite() {
        let canonicalizer = TensorNameCanonicalizer.mlxVLMToHuggingFace
        #expect(canonicalizer.canonicalize("language_model.model.embed_tokens.weight")
                == "model.language_model.embed_tokens.weight")
        #expect(canonicalizer.canonicalize("language_model.model.layers.0.self_attn.q_proj.weight")
                == "model.language_model.layers.0.self_attn.q_proj.weight")
        #expect(canonicalizer.canonicalize("language_model.model.layers.0.self_attn.q_proj.scales")
                == "model.language_model.layers.0.self_attn.q_proj.scales")
    }

    @Test("non-matching names pass through under MLX rule")
    func mlxRuleLeavesNonMatchingNames() {
        let canonicalizer = TensorNameCanonicalizer.mlxVLMToHuggingFace
        #expect(canonicalizer.canonicalize("visual.blocks.0.attn.qkv.weight")
                == "visual.blocks.0.attn.qkv.weight")
        #expect(canonicalizer.canonicalize("lm_head.weight") == "lm_head.weight")
        // Already canonical form — must not double-rewrite.
        #expect(canonicalizer.canonicalize("model.language_model.embed_tokens.weight")
                == "model.language_model.embed_tokens.weight")
    }

    @Test("detect picks MLX rule when any MLX-style name is present")
    func detectMLXConvention() {
        let mlxNames: Set<String> = [
            "language_model.model.embed_tokens.weight",
            "visual.blocks.0.attn.qkv.weight",
        ]
        let canonicalizer = TensorNameCanonicalizer.detect(from: mlxNames)
        #expect(canonicalizer.canonicalize("language_model.model.embed_tokens.weight")
                == "model.language_model.embed_tokens.weight")
    }

    @Test("detect returns identity for HF bundles")
    func detectHuggingFaceConvention() {
        let hfNames: Set<String> = [
            "model.language_model.embed_tokens.weight",
            "model.visual.patch_embed.weight",
            "lm_head.weight",
        ]
        let canonicalizer = TensorNameCanonicalizer.detect(from: hfNames)
        // No rewrite — identity preserves every input.
        for name in hfNames {
            #expect(canonicalizer.canonicalize(name) == name)
        }
    }

    @Test("detect returns identity for text-only HF bundles")
    func detectTextOnlyHuggingFaceConvention() {
        let names: Set<String> = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "lm_head.weight",
        ]
        let canonicalizer = TensorNameCanonicalizer.detect(from: names)
        for name in names {
            #expect(canonicalizer.canonicalize(name) == name)
        }
    }

    @Test("first matching rule wins")
    func firstMatchingRuleWins() {
        let canonicalizer = TensorNameCanonicalizer(rules: [
            .init(from: "a.", to: "x."),
            .init(from: "a.b.", to: "y."),
        ])
        // The first rule matches first, so `a.b.c` → `x.b.c`, not `y.c`.
        #expect(canonicalizer.canonicalize("a.b.c") == "x.b.c")
    }
}
