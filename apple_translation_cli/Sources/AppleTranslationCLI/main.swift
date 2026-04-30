import Foundation
@preconcurrency import Translation

struct Candidate: Codable {
    let docdb_family_id: Int
    let field: String
    let source_appln_id: Int
    let source_lg: String?
    let target_lg: String
    let source_text: String
    let source_text_len: Int?
    let source_text_sha256: String
}

struct ResultRow: Codable {
    let docdb_family_id: Int
    let field: String
    let source_appln_id: Int
    let source_lg: String?
    let target_lg: String
    let source_text_sha256: String

    let translated_text: String?
    let ok: Bool
    let error: String?

    let engine: String
    let os_version: String
    let created_at_utc: String
}

@available(macOS 26.0, *)
extension TranslationSession.Request: @retroactive @unchecked Sendable {}

func iso8601NowUTC() -> String {
    let formatter = ISO8601DateFormatter()
    formatter.timeZone = TimeZone(secondsFromGMT: 0)
    return formatter.string(from: Date())
}

func readJSONL<T: Decodable>(_ path: String, as type: T.Type) throws -> [T] {
    let url = URL(fileURLWithPath: path)
    let data = try Data(contentsOf: url)
    guard let content = String(data: data, encoding: .utf8) else {
        throw NSError(domain: "apple-translation-cli", code: 2, userInfo: [NSLocalizedDescriptionKey: "Input is not valid UTF-8"])
    }
    let decoder = JSONDecoder()
    var out: [T] = []
    for (idx, line) in content.split(separator: "\n", omittingEmptySubsequences: true).enumerated() {
        do {
            let obj = try decoder.decode(T.self, from: Data(line.utf8))
            out.append(obj)
        } catch {
            throw NSError(
                domain: "apple-translation-cli",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Failed to parse JSONL at line \(idx + 1): \(error)"]
            )
        }
    }
    return out
}

func writeJSONL<T: Encodable>(_ rows: [T], to path: String) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.withoutEscapingSlashes]
    var lines: [String] = []
    lines.reserveCapacity(rows.count)
    for row in rows {
        let data = try encoder.encode(row)
        guard let s = String(data: data, encoding: .utf8) else { continue }
        lines.append(s)
    }
    let joined = lines.joined(separator: "\n") + (lines.isEmpty ? "" : "\n")
    try joined.write(to: URL(fileURLWithPath: path), atomically: true, encoding: .utf8)
}

@main
struct AppleTranslationCLI {
    static func main() async {
        do {
            guard #available(macOS 26.0, *) else {
                fputs("ERROR: Apple Translation CLI requires macOS 26.0 or newer (TranslationSession installedSource initializer).\n", stderr)
                exit(2)
            }

            let args = CommandLine.arguments
            guard args.count >= 3 else {
                fputs("Usage: apple-translation-cli <input_candidates.jsonl> <output_results.jsonl> [--chunk N]\n", stderr)
                exit(2)
            }
            let inputPath = args[1]
            let outputPath = args[2]
            var chunkSize = 50
            if let idx = args.firstIndex(of: "--chunk"), idx + 1 < args.count {
                chunkSize = Int(args[idx + 1]) ?? chunkSize
            }
            if chunkSize < 1 { chunkSize = 1 }

            let candidates = try readJSONL(inputPath, as: Candidate.self)
            let osVersion = ProcessInfo.processInfo.operatingSystemVersionString
            let createdAt = iso8601NowUTC()
            let engine = "apple_translation_framework"

            // Group by (field, source_lg, target_lg). Batch APIs require a consistent language pairing.
            let grouped = Dictionary(grouping: candidates) { c in
                let src = c.source_lg ?? ""
                return "\(c.field)\t\(src)\t\(c.target_lg)"
            }

            var results: [ResultRow] = []
            results.reserveCapacity(candidates.count)

            var missingPairs: Set<String> = []

            for (groupKey, groupCandidates) in grouped {
                let parts = groupKey.split(separator: "\t", omittingEmptySubsequences: false)
                _ = String(parts[0])
                let src = String(parts[1])
                let tgt = String(parts[2])
                let srcLang: Locale.Language
                if src.isEmpty {
                    // Unknown source language; translate will likely fail. Mark as error.
                    for c in groupCandidates {
                        results.append(ResultRow(
                            docdb_family_id: c.docdb_family_id,
                            field: c.field,
                            source_appln_id: c.source_appln_id,
                            source_lg: c.source_lg,
                            target_lg: c.target_lg,
                            source_text_sha256: c.source_text_sha256,
                            translated_text: nil,
                            ok: false,
                            error: "missing_source_language",
                            engine: engine,
                            os_version: osVersion,
                            created_at_utc: createdAt
                        ))
                    }
                    continue
                }

                srcLang = Locale.Language(identifier: src)
                let tgtLang = Locale.Language(identifier: tgt)

                // Verify availability to provide actionable error messages.
                let availability = LanguageAvailability()
                let status = await availability.status(from: srcLang, to: tgtLang)
                switch status {
                case .installed:
                    break
                case .supported:
                    // Supported but not installed. CLI can't request downloads.
                    missingPairs.insert("\(src)→\(tgt)")
                    for c in groupCandidates {
                        results.append(ResultRow(
                            docdb_family_id: c.docdb_family_id,
                            field: c.field,
                            source_appln_id: c.source_appln_id,
                            source_lg: c.source_lg,
                            target_lg: c.target_lg,
                            source_text_sha256: c.source_text_sha256,
                            translated_text: nil,
                            ok: false,
                            error: "language_pair_not_installed",
                            engine: engine,
                            os_version: osVersion,
                            created_at_utc: createdAt
                        ))
                    }
                    continue
                case .unsupported:
                    for c in groupCandidates {
                        results.append(ResultRow(
                            docdb_family_id: c.docdb_family_id,
                            field: c.field,
                            source_appln_id: c.source_appln_id,
                            source_lg: c.source_lg,
                            target_lg: c.target_lg,
                            source_text_sha256: c.source_text_sha256,
                            translated_text: nil,
                            ok: false,
                            error: "language_pair_unsupported",
                            engine: engine,
                            os_version: osVersion,
                            created_at_utc: createdAt
                        ))
                    }
                    continue
                @unknown default:
                    for c in groupCandidates {
                        results.append(ResultRow(
                            docdb_family_id: c.docdb_family_id,
                            field: c.field,
                            source_appln_id: c.source_appln_id,
                            source_lg: c.source_lg,
                            target_lg: c.target_lg,
                            source_text_sha256: c.source_text_sha256,
                            translated_text: nil,
                            ok: false,
                            error: "language_pair_status_unknown",
                            engine: engine,
                            os_version: osVersion,
                            created_at_utc: createdAt
                        ))
                    }
                    continue
                }

                // Translate in chunks
                do {
                    let session = TranslationSession(installedSource: srcLang, target: tgtLang)

                    var idx = 0
                    while idx < groupCandidates.count {
                        let end = min(idx + chunkSize, groupCandidates.count)
                        let chunk = Array(groupCandidates[idx..<end])
                        idx = end

                        let requests: [TranslationSession.Request] = chunk.map { c in
                            TranslationSession.Request(sourceText: c.source_text, clientIdentifier: c.source_text_sha256)
                        }

                        let responses = try await session.translations(from: requests)
                        // Map responses by clientIdentifier; handle duplicates by keeping first
                        var byId: [String: String] = [:]
                        for r in responses {
                            guard let id = r.clientIdentifier else { continue }
                            if byId[id] == nil {
                                byId[id] = r.targetText
                            }
                        }

                        for c in chunk {
                            let translated = byId[c.source_text_sha256]
                            if let translated {
                                results.append(ResultRow(
                                    docdb_family_id: c.docdb_family_id,
                                    field: c.field,
                                    source_appln_id: c.source_appln_id,
                                    source_lg: c.source_lg,
                                    target_lg: c.target_lg,
                                    source_text_sha256: c.source_text_sha256,
                                    translated_text: translated,
                                    ok: true,
                                    error: nil,
                                    engine: engine,
                                    os_version: osVersion,
                                    created_at_utc: createdAt
                                ))
                            } else {
                                results.append(ResultRow(
                                    docdb_family_id: c.docdb_family_id,
                                    field: c.field,
                                    source_appln_id: c.source_appln_id,
                                    source_lg: c.source_lg,
                                    target_lg: c.target_lg,
                                    source_text_sha256: c.source_text_sha256,
                                    translated_text: nil,
                                    ok: false,
                                    error: "missing_response",
                                    engine: engine,
                                    os_version: osVersion,
                                    created_at_utc: createdAt
                                ))
                            }
                        }
                    }
                } catch {
                    // Session init or translation error. Record per-row errors.
                    for c in groupCandidates {
                        results.append(ResultRow(
                            docdb_family_id: c.docdb_family_id,
                            field: c.field,
                            source_appln_id: c.source_appln_id,
                            source_lg: c.source_lg,
                            target_lg: c.target_lg,
                            source_text_sha256: c.source_text_sha256,
                            translated_text: nil,
                            ok: false,
                            error: "translation_error: \(error)",
                            engine: engine,
                            os_version: osVersion,
                            created_at_utc: createdAt
                        ))
                    }
                }
            }

            try writeJSONL(results, to: outputPath)

            if !missingPairs.isEmpty {
                let sorted = missingPairs.sorted()
                fputs("WARNING: Some language pairs are supported but not installed. Install them in System Settings before rerunning.\n", stderr)
                for p in sorted {
                    fputs("  - \(p)\n", stderr)
                }
            }

            print("Wrote results: \(outputPath)")
        } catch {
            fputs("ERROR: \(error)\n", stderr)
            exit(1)
        }
    }
}
