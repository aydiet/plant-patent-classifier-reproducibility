// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "apple-translation-cli",
    platforms: [
        // Keep manifest compatible with older SwiftPMs; runtime will enforce macOS 26+.
        .macOS(.v15)
    ],
    products: [
        .executable(name: "apple-translation-cli", targets: ["AppleTranslationCLI"])
    ],
    targets: [
        .executableTarget(
            name: "AppleTranslationCLI",
            path: "Sources/AppleTranslationCLI"
        )
    ]
)
