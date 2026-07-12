// swift-tools-version: 6.4
import PackageDescription
import Foundation

let enableMetalProbes = ProcessInfo.processInfo.environment["ENABLE_METAL_PROBES"] == "1"
let metalProbeSwiftSettings: [SwiftSetting] = enableMetalProbes
    ? [.define("ENABLE_METAL_PROBES")]
    : []

let package = Package(
    name: "swift-lm",
    platforms: [.macOS("27.0"), .iOS("27.0")],
    products: [
        .library(name: "LMIR", targets: ["LMIR"]),
        .library(name: "LMArchitecture", targets: ["LMArchitecture"]),
        .library(name: "ModelDeclarations", targets: ["ModelDeclarations"]),
        .library(name: "CoreAIExport", targets: ["CoreAIExport"]),
        .library(name: "SwiftLMCoreAI", targets: ["SwiftLMCoreAI"]),
        .library(name: "SwiftLMFoundationModels", targets: ["SwiftLMFoundationModels"]),
        .library(name: "MetalCompiler", targets: ["MetalCompiler"]),
        .library(name: "SwiftLM", targets: ["SwiftLM"]),
        .executable(name: "swiftlm-ir", targets: ["SwiftLMIR"]),
        .executable(name: "lfm25-a1b-benchmark", targets: ["LFM25A1BBenchmark"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-collections.git", from: "1.4.1"),
        .package(url: "https://github.com/huggingface/swift-jinja", from: "2.3.5"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/mattt/JSONSchema", from: "1.3.1"),
        .package(url: "https://github.com/1amageek/swift-testing-heartbeat", from: "0.1.0"),
        .package(
            url: "https://github.com/apple/coreai-models.git",
            revision: "938d0b8943b942ce66438b94ab017c5631d1aef4"
        ),
    ],
    targets: [
        .target(
            name: "LMIR"
        ),
        .target(
            name: "LMArchitecture",
            dependencies: [
                "LMIR",
                .product(name: "JSONSchema", package: "JSONSchema"),
            ]
        ),
        .target(name: "ModelDeclarations", dependencies: ["LMArchitecture"], path: "Sources/Models"),
        .target(
            name: "CoreAIExport",
            dependencies: ["LMArchitecture"]
        ),
        .target(
            name: "SwiftLMCoreAI",
            dependencies: ["CoreAIExport"]
        ),
        .target(
            name: "SwiftLMFoundationModels",
            dependencies: [
                .product(name: "CoreAILM", package: "coreai-models"),
            ]
        ),
        .executableTarget(
            name: "SwiftLMIR",
            dependencies: ["CoreAIExport", "ModelDeclarations"]
        ),
        .target(
            name: "MetalCompiler",
            dependencies: ["LMIR"],
            exclude: [
                "STAF/KVCacheSpec.md",
                "STAF/README.md",
            ],
            swiftSettings: metalProbeSwiftSettings
        ),
        .target(
            name: "SwiftLM",
            dependencies: [
                "LMIR",
                "LMArchitecture",
                "MetalCompiler",
                "ModelDeclarations",
                .product(name: "Jinja", package: "swift-jinja"),
                .product(name: "OrderedCollections", package: "swift-collections"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            swiftSettings: [
                .define("ACCELERATE_NEW_LAPACK"),
            ] + metalProbeSwiftSettings
        ),
        .executableTarget(
            name: "LFM25A1BBenchmark",
            dependencies: ["SwiftLM"]
        ),
        .testTarget(name: "MetalCompilerTests", dependencies: [
            "MetalCompiler", "LMArchitecture", "ModelDeclarations", "SwiftLM",
            .product(name: "TestHeartbeat", package: "swift-testing-heartbeat"),
            .product(name: "Tokenizers", package: "swift-transformers"),
            .product(name: "Jinja", package: "swift-jinja"),
            .product(name: "OrderedCollections", package: "swift-collections"),
        ], swiftSettings: metalProbeSwiftSettings),
        .testTarget(name: "ModelsTests", dependencies: [
            "ModelDeclarations", "LMArchitecture",
            .product(name: "TestHeartbeat", package: "swift-testing-heartbeat"),
        ]),
        .testTarget(name: "CoreAIExportTests", dependencies: [
            "CoreAIExport", "ModelDeclarations", "LMArchitecture",
        ]),
        .testTarget(name: "SwiftLMCoreAITests", dependencies: [
            "SwiftLMCoreAI",
        ]),
        .testTarget(name: "SwiftLMTests", dependencies: [
            "SwiftLM",
            .product(name: "TestHeartbeat", package: "swift-testing-heartbeat"),
        ], resources: [
            .process("TestData")
        ], swiftSettings: metalProbeSwiftSettings),
    ]
)
