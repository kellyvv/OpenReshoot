import Combine
import CoreML
import Foundation

enum ReconstructionModelInstallState: Equatable, Sendable {
    case notInstalled
    case checkingSource
    case downloading
    case downloaded
    case bundled
    case failed(String)
}

struct ReconstructionDownloadProgress: Equatable, Sendable {
    var bytesReceived: Int64 = 0
    var totalBytes: Int64?
    var bytesPerSecond: Double?
    var activeFilePath: String?
    var activeSourceLabel: String?

    var fractionCompleted: Double? {
        guard let totalBytes, totalBytes > 0 else { return nil }
        return min(1, max(0, Double(bytesReceived) / Double(totalBytes)))
    }
}

private enum ReconstructionModelDownloadError: LocalizedError {
    case invalidURL
    case invalidResponse(String)

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "模型下载 URL 无效"
        case .invalidResponse(let message):
            return message
        }
    }
}

/// PhoneClaw-style model installer state, scoped to OpenReshot's single Core ML asset.
final class ReconstructionModelStore: ObservableObject {
    @Published private(set) var installState: ReconstructionModelInstallState = .notInstalled
    @Published private(set) var progress = ReconstructionDownloadProgress()

    private var downloadTask: Task<Void, Never>?

    init() {
        installState = Self.currentInstallState()
        Task { [weak self] in
            await self?.restoreResumeProgressIfNeeded()
        }
    }

    var isDownloading: Bool {
        if case .downloading = installState { return true }
        if case .checkingSource = installState { return true }
        return false
    }

    var activeModelLabel: String {
        switch installState {
        case .downloaded:
            return "已下载"
        case .bundled:
            return "内置"
        case .downloading:
            return "下载中"
        case .checkingSource:
            return "连接中"
        case .failed:
            return hasDownloadedModel ? "已下载" : (Self.bundledModelURL == nil ? "下载" : "内置")
        case .notInstalled:
            return "下载"
        }
    }

    var hasDownloadedModel: Bool {
        FileManager.default.fileExists(atPath: Self.downloadedModelURL.path)
    }

    var hasResumeProgress: Bool {
        progress.bytesReceived > 0 && !hasDownloadedModel
    }

    var primaryDownloadSourceLabel: String {
        Self.builtInDownloadBaseURL.map {
            Self.sourceLabel(for: $0, fallback: "下载源")
        } ?? "下载源"
    }

    nonisolated func activeModelURL() -> URL? {
        Self.activeModelURL()
    }

    @MainActor
    func refreshInstallState() {
        installState = Self.currentInstallState()
    }

    @MainActor
    func installModel(onInstalled: @MainActor @escaping () -> Void = {}) {
        guard downloadTask == nil else { return }

        installState = .checkingSource
        downloadTask = Task { [weak self] in
            guard let store = self else { return }
            do {
                if Self.builtInDownloadBaseURL != nil {
                    try await Self.downloadAndInstallPackage { progress in
                        await store.updateDownloadProgress(progress)
                    }
                } else {
                    try Self.installBundledModel()
                }
                store.finishInstall(onInstalled: onInstalled)
            } catch is CancellationError {
                store.finishCancellation()
            } catch {
                store.finishFailure(error)
            }
        }
    }

    @MainActor
    func cancelDownload() {
        guard let downloadTask else {
            refreshInstallState()
            return
        }
        downloadTask.cancel()
    }

    @MainActor
    func removeDownloadedModel(onRemoved: @MainActor @escaping () -> Void = {}) {
        downloadTask?.cancel()
        downloadTask = nil
        do {
            if FileManager.default.fileExists(atPath: Self.downloadedModelURL.path) {
                try FileManager.default.removeItem(at: Self.downloadedModelURL)
            }
            if FileManager.default.fileExists(atPath: Self.legacyDownloadedRawModelURL.path) {
                try FileManager.default.removeItem(at: Self.legacyDownloadedRawModelURL)
            }
            if FileManager.default.fileExists(atPath: Self.downloadWorkspaceURL.path) {
                try FileManager.default.removeItem(at: Self.downloadWorkspaceURL)
            }
            onRemoved()
            refreshInstallState()
        } catch {
            installState = .failed(error.localizedDescription)
        }
    }

    @MainActor
    private func updateDownloadProgress(_ nextProgress: ReconstructionDownloadProgress) {
        progress = nextProgress
        installState = .downloading
    }

    @MainActor
    private func finishInstall(onInstalled: @MainActor () -> Void) {
        installState = .downloaded
        downloadTask = nil
        onInstalled()
    }

    @MainActor
    private func finishCancellation() {
        downloadTask = nil
        refreshInstallState()
    }

    @MainActor
    private func finishFailure(_ error: Error) {
        downloadTask = nil
        installState = .failed(error.localizedDescription)
    }

    @MainActor
    private func restoreResumeProgressIfNeeded() async {
        guard !hasDownloadedModel else { return }
        guard let resume = try? await Self.manifestStore.resumeState(for: Self.assetID) else { return }
        progress = ReconstructionDownloadProgress(
            bytesReceived: resume.downloadedBytes,
            totalBytes: resume.totalBytes,
            bytesPerSecond: nil
        )
    }

    nonisolated static func activeModelURL() -> URL? {
        if FileManager.default.fileExists(atPath: downloadedModelURL.path) {
            return downloadedModelURL
        }
        return bundledModelURL
    }

    nonisolated private static func currentInstallState() -> ReconstructionModelInstallState {
        if FileManager.default.fileExists(atPath: downloadedModelURL.path) {
            return .downloaded
        }
        if bundledModelURL != nil {
            return .bundled
        }
        return .notInstalled
    }

    nonisolated static var bundledModelURL: URL? {
        Bundle.main.url(forResource: "SHARP", withExtension: "mlmodelc")
    }

    nonisolated static var downloadedModelURL: URL {
        modelsDirectory.appendingPathComponent("SHARP.mlmodelc", isDirectory: true)
    }

    nonisolated private static var legacyDownloadedRawModelURL: URL {
        modelsDirectory.appendingPathComponent("SHARP-source-model", isDirectory: false)
    }

    nonisolated private static var downloadWorkspaceURL: URL {
        modelsDirectory
            .appendingPathComponent(DownloadManifestStore.workspaceDirectoryName, isDirectory: true)
            .appendingPathComponent(assetID, isDirectory: true)
    }

    nonisolated private static var modelsDirectory: URL {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return support
            .appendingPathComponent("OpenReshot", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }

    nonisolated private static var builtInDownloadBaseURL: URL? {
        guard let value = Bundle.main.object(forInfoDictionaryKey: "OpenReshotModelDownloadBaseURL") as? String else {
            return nil
        }
        return downloadBaseURL(from: value)
    }

    nonisolated private static var mirrorBaseURLs: [URL] {
        if let values = Bundle.main.object(forInfoDictionaryKey: "OpenReshotModelDownloadMirrorBaseURLs") as? [String] {
            return values.compactMap(downloadBaseURL(from:))
        }

        if let value = Bundle.main.object(forInfoDictionaryKey: "OpenReshotModelDownloadMirrorBaseURLs") as? String {
            return value
                .split(separator: ",")
                .map(String.init)
                .compactMap(downloadBaseURL(from:))
        }

        return []
    }

    nonisolated private static let assetID = "openreshot-sharp-coreml"

    nonisolated private static let packageFileSpecs: [(relativePath: String, expectedBytes: Int64)] = [
        ("Manifest.json", 617),
        ("Data/com.apple.CoreML/model.mlmodel", 1_019_167),
        ("Data/com.apple.CoreML/weights/weight.bin", 1_322_157_376),
    ]

    nonisolated private static let manifestStore = DownloadManifestStore(rootDirectory: modelsDirectory)

    nonisolated private static func installBundledModel() throws {
        guard let bundledModelURL else {
            throw ReconstructionModelDownloadError.invalidURL
        }
        let fileManager = FileManager.default
        try createDirectoryExcludedFromBackup(modelsDirectory)
        let stagingURL = modelsDirectory.appendingPathComponent("SHARP.mlmodelc.installing-\(UUID().uuidString)", isDirectory: true)
        try? fileManager.removeItem(at: stagingURL)
        try fileManager.copyItem(at: bundledModelURL, to: stagingURL)
        try? fileManager.removeItem(at: downloadedModelURL)
        try fileManager.moveItem(at: stagingURL, to: downloadedModelURL)
    }

    nonisolated private static func downloadAndInstallPackage(
        progress: @escaping @Sendable (ReconstructionDownloadProgress) async -> Void
    ) async throws {
        guard builtInDownloadBaseURL != nil else {
            throw ReconstructionModelDownloadError.invalidURL
        }

        try createDirectoryExcludedFromBackup(modelsDirectory)
        try await manifestStore.pruneOrphans(knownAssetIDs: [assetID])

        if let resumeState = try await manifestStore.resumeState(for: assetID) {
            await progress(ReconstructionDownloadProgress(
                bytesReceived: resumeState.downloadedBytes,
                totalBytes: resumeState.totalBytes,
                bytesPerSecond: nil
            ))
        }

        let asset = try await modelDownloadAsset()
        let observer = ReconstructionDownloadObserver(progress: progress)
        let downloader = ResumableAssetDownloader(
            manifestStore: manifestStore,
            observer: observer
        )

        _ = try await downloader.download(asset: asset)
        try Task.checkCancellation()

        let compiledURL = try await MLModel.compileModel(at: asset.destinationDirectory)
        try Task.checkCancellation()

        let fileManager = FileManager.default
        let stagingURL = modelsDirectory.appendingPathComponent("SHARP.mlmodelc.installing-\(UUID().uuidString)", isDirectory: true)
        try? fileManager.removeItem(at: stagingURL)
        try fileManager.moveItem(at: compiledURL, to: stagingURL)
        try? fileManager.removeItem(at: downloadedModelURL)
        try? fileManager.removeItem(at: legacyDownloadedRawModelURL)
        try fileManager.moveItem(at: stagingURL, to: downloadedModelURL)
        try await manifestStore.purge(assetID: assetID)
    }

    nonisolated private static func modelDownloadAsset() async throws -> DownloadAsset {
        let stagingDirectory = try await manifestStore.stagingDirectory(for: assetID)
        let packageDirectory = stagingDirectory.appendingPathComponent("SHARP.mlpackage", isDirectory: true)
        return DownloadAsset(
            id: assetID,
            displayName: "空间模型",
            destinationDirectory: packageDirectory,
            files: try packageFileSpecs.map { spec in
                DownloadFile(
                    relativePath: spec.relativePath,
                    expectedSize: spec.expectedBytes,
                    sources: try downloadSources(for: spec.relativePath)
                )
            },
            preservesWorkspaceOnCompletion: true
        )
    }

    nonisolated private static func downloadSources(for relativePath: String) throws -> [DownloadFile.Source] {
        guard let primaryBase = builtInDownloadBaseURL else {
            throw ReconstructionModelDownloadError.invalidURL
        }

        let bases = [primaryBase] + mirrorBaseURLs
        var sources: [DownloadFile.Source] = []
        for (index, baseURL) in bases.enumerated() {
            guard let sourceURL = URL(string: relativePath, relativeTo: baseURL)?.absoluteURL else {
                continue
            }
            let label = sourceLabel(for: baseURL, fallback: index == 0 ? "Hugging Face" : "Mirror \(index)")
            sources.append(DownloadFile.Source(label: label, url: sourceURL, priority: index))
        }

        guard !sources.isEmpty else {
            throw ReconstructionModelDownloadError.invalidURL
        }
        return sources
    }

    nonisolated private static func downloadBaseURL(from value: String) -> URL? {
        let trimmed = value
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard !trimmed.isEmpty,
              let url = URL(string: trimmed + "/"),
              let scheme = url.scheme?.lowercased(),
              ["http", "https"].contains(scheme) else {
            return nil
        }
        return url
    }

    nonisolated private static func sourceLabel(for url: URL, fallback: String) -> String {
        guard let host = url.host?.lowercased() else { return fallback }
        if host.contains("huggingface.co") { return "Hugging Face" }
        if host.contains("hf-mirror.com") { return "HF Mirror" }
        if host.contains("modelscope") { return "ModelScope" }
        return host
    }
}

private struct ReconstructionDownloadObserver: DownloadObserver {
    let progress: @Sendable (ReconstructionDownloadProgress) async -> Void

    func onProgress(_ snapshot: DownloadProgressSnapshot) async {
        await progress(ReconstructionDownloadProgress(
            bytesReceived: snapshot.downloadedBytes,
            totalBytes: snapshot.totalBytes,
            bytesPerSecond: snapshot.bytesPerSecond,
            activeFilePath: snapshot.activeFilePath,
            activeSourceLabel: snapshot.activeSourceLabel
        ))
    }

    func onRetry(
        assetID: String,
        filePath: String,
        source: DownloadFile.Source,
        attempt: Int,
        error: DownloadFailure
    ) async {
        print("↻ [OpenReshot] download retry asset=\(assetID) file=\(filePath) source=\(source.label) attempt=\(attempt) error=\(error.localizedDescription)")
    }

    func onSourceSwitch(
        assetID: String,
        filePath: String,
        from: DownloadFile.Source?,
        to: DownloadFile.Source,
        reason: DownloadFailure?
    ) async {
        print("↔︎ [OpenReshot] switching download source asset=\(assetID) file=\(filePath) from=\(from?.label ?? "-") to=\(to.label) reason=\(reason?.localizedDescription ?? "-")")
    }

    func onFailure(assetID: String, failure: DownloadFailure) async {
        print("❌ [OpenReshot] download failed asset=\(assetID) error=\(failure.localizedDescription)")
    }
}

private struct DownloadAsset: Equatable, Identifiable, Sendable {
    let id: String
    let displayName: String
    let destinationDirectory: URL
    let files: [DownloadFile]
    let preservesWorkspaceOnCompletion: Bool

    init(
        id: String,
        displayName: String,
        destinationDirectory: URL,
        files: [DownloadFile],
        preservesWorkspaceOnCompletion: Bool = false
    ) {
        precondition(Self.isValidID(id), "DownloadAsset.id may only contain letters, digits, '.', '_', or '-'")
        self.id = id
        self.displayName = displayName
        self.destinationDirectory = destinationDirectory
        self.files = files
        self.preservesWorkspaceOnCompletion = preservesWorkspaceOnCompletion
    }

    static func isValidID(_ id: String) -> Bool {
        guard !id.isEmpty else { return false }
        return id.range(of: #"^[A-Za-z0-9._-]+$"#, options: .regularExpression) != nil
    }
}

private struct DownloadFile: Codable, Equatable, Identifiable, Sendable {
    struct Source: Codable, Equatable, Sendable {
        let label: String
        let url: URL
        /// Lower values are tried first. `0` is the default highest priority.
        let priority: Int

        init(label: String, url: URL, priority: Int = 0) {
            self.label = label
            self.url = url
            self.priority = priority
        }
    }

    let relativePath: String
    let expectedSize: Int64?
    let expectedSHA256: String?
    let sources: [Source]

    var id: String { relativePath }

    init(
        relativePath: String,
        expectedSize: Int64? = nil,
        expectedSHA256: String? = nil,
        sources: [Source]
    ) {
        self.relativePath = relativePath
        self.expectedSize = expectedSize
        self.expectedSHA256 = expectedSHA256
        self.sources = sources
    }
}

private struct DownloadManifest: Codable, Equatable, Sendable {
    static let currentSchemaVersion = 1

    let schemaVersion: Int
    let assetID: String
    let createdAt: Date
    let updatedAt: Date
    let files: [DownloadManifestFile]

    init(
        schemaVersion: Int = DownloadManifest.currentSchemaVersion,
        assetID: String,
        createdAt: Date,
        updatedAt: Date,
        files: [DownloadManifestFile]
    ) {
        self.schemaVersion = schemaVersion
        self.assetID = assetID
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.files = files
    }
}

private struct DownloadManifestFile: Codable, Equatable, Sendable {
    let relativePath: String
    let state: DownloadManifestFileState
    let downloadedBytes: Int64
    let expectedBytes: Int64?
    let selectedSourceLabel: String?
    let metadata: DownloadFileMetadata?

    init(
        relativePath: String,
        state: DownloadManifestFileState,
        downloadedBytes: Int64,
        expectedBytes: Int64? = nil,
        selectedSourceLabel: String? = nil,
        metadata: DownloadFileMetadata? = nil
    ) {
        self.relativePath = relativePath
        self.state = state
        self.downloadedBytes = downloadedBytes
        self.expectedBytes = expectedBytes
        self.selectedSourceLabel = selectedSourceLabel
        self.metadata = metadata
    }
}

private enum DownloadManifestFileState: String, Codable, Equatable, Sendable {
    case pending
    case downloading
    case paused
    case complete
    case failed
}

private struct DownloadFileMetadata: Codable, Equatable, Sendable {
    let sourceURL: URL?
    let sourceHost: String?
    let etag: String?
    let contentLength: Int64?
    let lastModified: String?
    let checksumSHA256: String?
    let updatedAt: Date?

    init(
        sourceURL: URL? = nil,
        sourceHost: String? = nil,
        etag: String? = nil,
        contentLength: Int64? = nil,
        lastModified: String? = nil,
        checksumSHA256: String? = nil,
        updatedAt: Date? = nil
    ) {
        self.sourceURL = sourceURL
        self.sourceHost = sourceHost
        self.etag = etag
        self.contentLength = contentLength
        self.lastModified = lastModified
        self.checksumSHA256 = checksumSHA256
        self.updatedAt = updatedAt
    }
}

private struct DownloadResumeState: Equatable, Sendable {
    let downloadedBytes: Int64
    let totalBytes: Int64?
    let resumableFileCount: Int
}

private struct DownloadProgressSnapshot: Codable, Equatable, Sendable {
    let assetID: String
    let completedFileCount: Int
    let totalFileCount: Int
    let downloadedBytes: Int64
    let totalBytes: Int64?
    let bytesPerSecond: Double?
    let activeFilePath: String?
    let activeSourceLabel: String?
    let phase: DownloadProgressPhase
    let updatedAt: Date

    var byteFraction: Double? {
        guard let totalBytes, totalBytes > 0 else { return nil }
        return min(1, max(0, Double(downloadedBytes) / Double(totalBytes)))
    }

    var fileFraction: Double {
        guard totalFileCount > 0 else { return 0 }
        return min(1, max(0, Double(completedFileCount) / Double(totalFileCount)))
    }

    var combinedFraction: Double? {
        byteFraction ?? fileFraction
    }
}

private enum DownloadProgressPhase: String, Codable, Equatable, Sendable {
    case idle
    case preflighting
    case downloading
    case validating
    case paused
    case complete
    case failed
}

private enum DownloadFailure: Error, Equatable, Sendable {
    case invalidURL(String)
    case invalidResponse(String)
    case httpStatus(Int)
    case validatorMismatch(expected: String, actual: String, field: String)
    case insufficientDiskSpace(required: Int64, available: Int64)
    case manifestCorrupt(String)
    case fileSystem(String)
    case cancelled
}

extension DownloadFailure: LocalizedError {
    var errorDescription: String? {
        switch self {
        case .invalidURL(let value):
            return "模型下载 URL 无效：\(value)"
        case .invalidResponse(let message):
            return "下载源响应无效：\(message)"
        case .httpStatus(let code):
            return "模型下载失败，HTTP \(code)"
        case .validatorMismatch(let expected, let actual, let field):
            return "\(field) 校验失败：期望 \(expected)，实际 \(actual)"
        case .insufficientDiskSpace(let required, let available):
            let formatter = ByteCountFormatter()
            formatter.countStyle = .file
            return "可用空间不足，需要 \(formatter.string(fromByteCount: required))，当前约 \(formatter.string(fromByteCount: available))"
        case .manifestCorrupt:
            return "下载记录损坏，已准备重新下载"
        case .fileSystem(let message):
            return "文件写入失败：\(message)"
        case .cancelled:
            return "下载已取消"
        }
    }
}

private enum DownloadPreflight: Sendable {
    static func availableCapacity(for directory: URL) throws -> Int64? {
        let values = try directory.resourceValues(forKeys: [
            .volumeAvailableCapacityForImportantUsageKey,
            .volumeAvailableCapacityKey,
        ])
        if let important = values.volumeAvailableCapacityForImportantUsage, important > 0 {
            return important
        }
        if let general = values.volumeAvailableCapacity, general > 0 {
            return Int64(general)
        }
        return values.volumeAvailableCapacityForImportantUsage
    }

    static func validateDiskSpace(requiredBytes: Int64, at directory: URL) throws {
        guard let available = try availableCapacity(for: directory) else { return }
        guard available >= requiredBytes else {
            throw DownloadFailure.insufficientDiskSpace(required: requiredBytes, available: available)
        }
    }
}

private protocol DownloadObserver: Sendable {
    func onProgress(_ snapshot: DownloadProgressSnapshot) async
    func onRetry(
        assetID: String,
        filePath: String,
        source: DownloadFile.Source,
        attempt: Int,
        error: DownloadFailure
    ) async
    func onSourceSwitch(
        assetID: String,
        filePath: String,
        from: DownloadFile.Source?,
        to: DownloadFile.Source,
        reason: DownloadFailure?
    ) async
    func onFailure(assetID: String, failure: DownloadFailure) async
}

private extension DownloadObserver {
    func onProgress(_ snapshot: DownloadProgressSnapshot) async {}

    func onRetry(
        assetID: String,
        filePath: String,
        source: DownloadFile.Source,
        attempt: Int,
        error: DownloadFailure
    ) async {}

    func onSourceSwitch(
        assetID: String,
        filePath: String,
        from: DownloadFile.Source?,
        to: DownloadFile.Source,
        reason: DownloadFailure?
    ) async {}

    func onFailure(assetID: String, failure: DownloadFailure) async {}
}

private struct NoopDownloadObserver: DownloadObserver {}

private actor DownloadManifestStore {
    static let workspaceDirectoryName = ".downloads"
    static let manifestFileName = ".download-manifest.json"

    private let rootDirectory: URL
    private let fileManager: FileManager

    init(rootDirectory: URL, fileManager: FileManager = .default) {
        self.rootDirectory = rootDirectory
        self.fileManager = fileManager
    }

    func workspaceRootDirectory() throws -> URL {
        let url = rootDirectory.appendingPathComponent(Self.workspaceDirectoryName, isDirectory: true)
        try ensureDirectory(url, excludedFromBackup: true)
        return url
    }

    func workspaceDirectory(for assetID: String) throws -> URL {
        let url = try workspaceRootDirectory()
            .appendingPathComponent(sanitizedAssetID(assetID), isDirectory: true)
        try ensureDirectory(url, excludedFromBackup: true)
        return url
    }

    func manifestURL(for assetID: String) throws -> URL {
        try workspaceDirectory(for: assetID)
            .appendingPathComponent(Self.manifestFileName, isDirectory: false)
    }

    func partialFileURL(for assetID: String, relativePath: String) throws -> URL {
        let url = try workspaceDirectory(for: assetID)
            .appendingPathComponent(relativePath, isDirectory: false)
            .appendingPathExtension("part")
        try ensureDirectory(url.deletingLastPathComponent(), excludedFromBackup: true)
        return url
    }

    func resumeDataURL(for assetID: String, relativePath: String) throws -> URL {
        let url = try workspaceDirectory(for: assetID)
            .appendingPathComponent(relativePath, isDirectory: false)
            .appendingPathExtension("resumeData")
        try ensureDirectory(url.deletingLastPathComponent(), excludedFromBackup: true)
        return url
    }

    func stagingDirectory(for assetID: String) throws -> URL {
        let url = try workspaceDirectory(for: assetID)
            .appendingPathComponent("staging", isDirectory: true)
        try ensureDirectory(url, excludedFromBackup: true)
        return url
    }

    func readManifest(for assetID: String) throws -> DownloadManifest? {
        let url = rawManifestURL(for: assetID)
        guard fileManager.fileExists(atPath: url.path) else { return nil }

        do {
            let data = try Data(contentsOf: url)
            let manifest = try JSONDecoder.downloadManifestDecoder.decode(DownloadManifest.self, from: data)
            guard manifest.schemaVersion <= DownloadManifest.currentSchemaVersion else {
                throw DownloadFailure.manifestCorrupt(
                    "schema v\(manifest.schemaVersion) > v\(DownloadManifest.currentSchemaVersion)"
                )
            }
            return manifest
        } catch let failure as DownloadFailure {
            throw failure
        } catch {
            throw DownloadFailure.manifestCorrupt(error.localizedDescription)
        }
    }

    func resumeState(for assetID: String) throws -> DownloadResumeState? {
        guard let manifest = try readManifest(for: assetID) else { return nil }

        var downloadedBytes: Int64 = 0
        var totalBytes: Int64 = 0
        var hasUnknownTotal = false
        var resumableFileCount = 0

        for file in manifest.files where file.state != .complete {
            let partialURL = try partialFileURL(for: assetID, relativePath: file.relativePath)
            let resumeDataURL = try resumeDataURL(for: assetID, relativePath: file.relativePath)
            let partialBytes = fileSize(at: partialURL)
            let hasResumeData = fileSize(at: resumeDataURL) > 0
            let expectedBytes = file.metadata?.contentLength ?? file.expectedBytes
            let bytes = clampedResumeBytes(max(partialBytes, file.downloadedBytes), expectedBytes: expectedBytes)
            guard bytes > 0 || hasResumeData else { continue }

            downloadedBytes += bytes
            resumableFileCount += 1

            if let expected = expectedBytes, expected > 0 {
                totalBytes += expected
            } else {
                hasUnknownTotal = true
            }
        }

        guard resumableFileCount > 0 else { return nil }
        return DownloadResumeState(
            downloadedBytes: downloadedBytes,
            totalBytes: hasUnknownTotal ? nil : totalBytes,
            resumableFileCount: resumableFileCount
        )
    }

    func writeManifest(_ manifest: DownloadManifest, for assetID: String) throws {
        let url = try manifestURL(for: assetID)
        let data = try JSONEncoder.downloadManifestEncoder.encode(manifest)
        try atomicWrite(data, to: url)
    }

    func purge(assetID: String) throws {
        let url = try workspaceDirectory(for: assetID)
        if fileManager.fileExists(atPath: url.path) {
            try fileManager.removeItem(at: url)
        }
    }

    func pruneOrphans(knownAssetIDs: Set<String>) throws {
        let workspaceRoot = try workspaceRootDirectory()
        let knownDirectoryNames = Set(knownAssetIDs.map(sanitizedAssetID))
        guard let children = try? fileManager.contentsOfDirectory(
            at: workspaceRoot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return
        }

        for child in children {
            let values = try child.resourceValues(forKeys: [.isDirectoryKey])
            guard values.isDirectory == true else { continue }
            guard !knownDirectoryNames.contains(child.lastPathComponent) else { continue }
            try fileManager.removeItem(at: child)
        }
    }

    private func ensureDirectory(_ url: URL, excludedFromBackup: Bool) throws {
        if !fileManager.fileExists(atPath: url.path) {
            try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
        }

        guard excludedFromBackup else { return }
        var mutableURL = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try mutableURL.setResourceValues(values)
    }

    private func rawWorkspaceDirectory(for assetID: String) -> URL {
        rootDirectory
            .appendingPathComponent(Self.workspaceDirectoryName, isDirectory: true)
            .appendingPathComponent(sanitizedAssetID(assetID), isDirectory: true)
    }

    private func rawManifestURL(for assetID: String) -> URL {
        rawWorkspaceDirectory(for: assetID)
            .appendingPathComponent(Self.manifestFileName, isDirectory: false)
    }

    private func fileSize(at url: URL) -> Int64 {
        guard let attributes = try? fileManager.attributesOfItem(atPath: url.path) else {
            return 0
        }
        return (attributes[.size] as? Int64) ?? 0
    }

    private func clampedResumeBytes(_ bytes: Int64, expectedBytes: Int64?) -> Int64 {
        let positiveBytes = max(0, bytes)
        guard let expectedBytes, expectedBytes > 0 else { return positiveBytes }
        return min(positiveBytes, expectedBytes)
    }

    private func atomicWrite(_ data: Data, to destinationURL: URL) throws {
        let parentDirectory = destinationURL.deletingLastPathComponent()
        try ensureDirectory(parentDirectory, excludedFromBackup: true)

        let temporaryURL = parentDirectory
            .appendingPathComponent(".\(destinationURL.lastPathComponent).\(UUID().uuidString).tmp")

        do {
            try data.write(to: temporaryURL, options: [.completeFileProtectionUntilFirstUserAuthentication])
            if fileManager.fileExists(atPath: destinationURL.path) {
                _ = try fileManager.replaceItemAt(destinationURL, withItemAt: temporaryURL)
            } else {
                try fileManager.moveItem(at: temporaryURL, to: destinationURL)
            }
        } catch {
            try? fileManager.removeItem(at: temporaryURL)
            throw DownloadFailure.fileSystem(error.localizedDescription)
        }
    }

    private func sanitizedAssetID(_ assetID: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        let scalars = assetID.unicodeScalars.map { scalar -> String in
            allowed.contains(scalar) ? String(scalar) : "_"
        }
        let sanitized = scalars.joined()
        return sanitized.isEmpty ? "asset" : sanitized
    }
}

private extension JSONEncoder {
    static var downloadManifestEncoder: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }
}

private extension JSONDecoder {
    static var downloadManifestDecoder: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}

private actor DownloadProgressRecorder {
    private let manifestStore: DownloadManifestStore
    private let observer: any DownloadObserver
    private let assetID: String
    private let minimumManifestWriteInterval: TimeInterval
    private var pendingManifest: DownloadManifest?
    private var scheduledFlushTask: Task<Void, Never>?
    private var lastManifestWriteAt: Date?
    private var latestObservedBytes: Int64 = -1
    private var isFinished = false

    init(
        manifestStore: DownloadManifestStore,
        observer: any DownloadObserver,
        assetID: String,
        minimumManifestWriteInterval: TimeInterval = 0.5
    ) {
        self.manifestStore = manifestStore
        self.observer = observer
        self.assetID = assetID
        self.minimumManifestWriteInterval = minimumManifestWriteInterval
    }

    func record(manifest: DownloadManifest, snapshot: DownloadProgressSnapshot, resetsBaseline: Bool = false) async {
        guard !isFinished else { return }
        if resetsBaseline {
            latestObservedBytes = -1
        }
        guard snapshot.downloadedBytes >= latestObservedBytes else { return }
        latestObservedBytes = snapshot.downloadedBytes
        pendingManifest = manifest

        await observer.onProgress(snapshot)
        await writeManifestIfDue()
    }

    func finish() async {
        isFinished = true
        scheduledFlushTask?.cancel()
        scheduledFlushTask = nil
        await flushPendingManifest()
    }

    private func writeManifestIfDue() async {
        guard !isFinished else { return }
        let now = Date()
        guard let lastManifestWriteAt else {
            await flushPendingManifest(now: now)
            return
        }

        let elapsed = now.timeIntervalSince(lastManifestWriteAt)
        if elapsed >= minimumManifestWriteInterval {
            await flushPendingManifest(now: now)
        } else {
            scheduleFlush(after: minimumManifestWriteInterval - elapsed)
        }
    }

    private func scheduleFlush(after delay: TimeInterval) {
        guard !isFinished, pendingManifest != nil, scheduledFlushTask == nil else { return }
        let nanoseconds = UInt64(max(0, delay) * 1_000_000_000)
        scheduledFlushTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: nanoseconds)
            guard !Task.isCancelled else { return }
            await self?.flushScheduledManifest()
        }
    }

    private func flushScheduledManifest() async {
        scheduledFlushTask = nil
        guard !isFinished else { return }
        await flushPendingManifest()
    }

    private func flushPendingManifest(now: Date = Date()) async {
        guard let manifest = pendingManifest else { return }
        pendingManifest = nil
        do {
            try await manifestStore.writeManifest(manifest, for: assetID)
            lastManifestWriteAt = now
        } catch {
            print("[OpenReshot] manifest progress write failed: \(error.localizedDescription)")
        }
    }
}

private actor ResumableAssetDownloader {
    private let manifestStore: DownloadManifestStore
    private let observer: any DownloadObserver
    private let fileManager: FileManager
    private let urlSession: URLSession

    init(
        manifestStore: DownloadManifestStore,
        observer: any DownloadObserver = NoopDownloadObserver(),
        fileManager: FileManager = .default,
        urlSession: URLSession = .shared
    ) {
        self.manifestStore = manifestStore
        self.observer = observer
        self.fileManager = fileManager
        self.urlSession = urlSession
    }

    func download(asset: DownloadAsset) async throws -> DownloadProgressSnapshot {
        guard !asset.files.isEmpty else {
            throw DownloadFailure.invalidResponse("Download asset has no files")
        }

        try createDirectoryExcludedFromBackup(asset.destinationDirectory)
        try await preflightDiskSpace(for: asset)

        var manifest = try await readManifestOrRestart(assetID: asset.id)
            ?? freshManifest(for: asset, now: Date())
        let totalBytes = totalExpectedBytes(for: asset)
        var completedFiles = 0
        var completedBytes: Int64 = 0

        for file in asset.files {
            try Task.checkCancellation()
            let finalURL = asset.destinationDirectory.appendingPathComponent(file.relativePath, isDirectory: false)
            if fileManager.fileExists(atPath: finalURL.path) {
                let size = fileSize(finalURL)
                let isUsable: Bool
                if let expected = file.expectedSize {
                    isUsable = size >= expected * 9 / 10
                } else {
                    isUsable = size > 0
                }
                if isUsable {
                    completedFiles += 1
                    completedBytes += size
                    continue
                }
            }

            let result = try await downloadFile(
                file,
                asset: asset,
                manifest: manifest,
                completedFiles: completedFiles,
                completedBytes: completedBytes,
                totalBytes: totalBytes
            )
            manifest = result.manifest
            completedFiles += 1
            completedBytes += result.bytesWritten
        }

        let snapshot = DownloadProgressSnapshot(
            assetID: asset.id,
            completedFileCount: completedFiles,
            totalFileCount: asset.files.count,
            downloadedBytes: completedBytes,
            totalBytes: completedBytes,
            bytesPerSecond: nil,
            activeFilePath: nil,
            activeSourceLabel: nil,
            phase: .complete,
            updatedAt: Date()
        )
        await observer.onProgress(snapshot)
        if !asset.preservesWorkspaceOnCompletion {
            try await manifestStore.purge(assetID: asset.id)
        }
        return snapshot
    }

    func purge(assetID: String) async throws {
        try await manifestStore.purge(assetID: assetID)
    }

    func pruneOrphans(knownAssetIDs: Set<String>) async throws {
        try await manifestStore.pruneOrphans(knownAssetIDs: knownAssetIDs)
    }

    private func downloadFile(
        _ file: DownloadFile,
        asset: DownloadAsset,
        manifest: DownloadManifest,
        completedFiles: Int,
        completedBytes: Int64,
        totalBytes: Int64?
    ) async throws -> (manifest: DownloadManifest, bytesWritten: Int64) {
        let partialURL = try await manifestStore.partialFileURL(for: asset.id, relativePath: file.relativePath)
        let finalURL = asset.destinationDirectory.appendingPathComponent(file.relativePath, isDirectory: false)
        try createDirectoryExcludedFromBackup(finalURL.deletingLastPathComponent())

        var currentManifest = manifest
        var lastError: Error?
        var previousSource: DownloadFile.Source?
        let sources = orderedSources(file.sources)

        for (attempt, source) in sources.enumerated() {
            if let previousSource {
                await observer.onSourceSwitch(
                    assetID: asset.id,
                    filePath: file.relativePath,
                    from: previousSource,
                    to: source,
                    reason: lastError.map(downloadFailure(from:))
                )
            }
            previousSource = source

            do {
                let plan = try await resumePlan(
                    assetID: asset.id,
                    file: file,
                    source: source,
                    partialURL: partialURL,
                    manifest: currentManifest
                )

                if plan.restart {
                    try? fileManager.removeItem(at: partialURL)
                    if let resumeDataURL = try? await manifestStore.resumeDataURL(for: asset.id, relativePath: file.relativePath) {
                        try? fileManager.removeItem(at: resumeDataURL)
                    }
                }

                let result = try await transfer(
                    file: file,
                    asset: asset,
                    source: source,
                    partialURL: partialURL,
                    initialOffset: plan.restart ? 0 : plan.offset,
                    metadata: plan.metadata,
                    manifest: currentManifest,
                    completedFiles: completedFiles,
                    completedBytes: completedBytes,
                    totalBytes: totalBytes
                )
                currentManifest = result.manifest

                try? fileManager.removeItem(at: finalURL)
                try createDirectoryExcludedFromBackup(finalURL.deletingLastPathComponent())
                try fileManager.moveItem(at: partialURL, to: finalURL)

                currentManifest = updatedManifest(
                    currentManifest,
                    asset: asset,
                    replacing: DownloadManifestFile(
                        relativePath: file.relativePath,
                        state: .complete,
                        downloadedBytes: result.bytesWritten,
                        expectedBytes: result.expectedBytes,
                        selectedSourceLabel: source.label,
                        metadata: result.metadata
                    )
                )
                try await manifestStore.writeManifest(currentManifest, for: asset.id)
                return (currentManifest, result.bytesWritten)
            } catch {
                if isCancellation(error) {
                    let latestManifest = (try? await manifestStore.readManifest(for: asset.id)) ?? currentManifest
                    let existingEntry = latestManifest.files.first(where: { $0.relativePath == file.relativePath })
                    let bytes = max(fileSize(partialURL), existingEntry?.downloadedBytes ?? 0)
                    currentManifest = updatedManifest(
                        latestManifest,
                        asset: asset,
                        replacing: DownloadManifestFile(
                            relativePath: file.relativePath,
                            state: .paused,
                            downloadedBytes: bytes,
                            expectedBytes: existingEntry?.expectedBytes ?? file.expectedSize,
                            selectedSourceLabel: source.label,
                            metadata: existingEntry?.metadata
                        )
                    )
                    try? await manifestStore.writeManifest(currentManifest, for: asset.id)
                    throw CancellationError()
                }

                lastError = error
                let failure = downloadFailure(from: error)
                await observer.onRetry(
                    assetID: asset.id,
                    filePath: file.relativePath,
                    source: source,
                    attempt: attempt + 1,
                    error: failure
                )
                let latestManifest = (try? await manifestStore.readManifest(for: asset.id)) ?? currentManifest
                let existingEntry = latestManifest.files.first(where: { $0.relativePath == file.relativePath })
                let bytes = max(fileSize(partialURL), existingEntry?.downloadedBytes ?? 0)
                currentManifest = updatedManifest(
                    latestManifest,
                    asset: asset,
                    replacing: DownloadManifestFile(
                        relativePath: file.relativePath,
                        state: .failed,
                        downloadedBytes: bytes,
                        expectedBytes: existingEntry?.expectedBytes ?? file.expectedSize,
                        selectedSourceLabel: source.label,
                        metadata: existingEntry?.metadata
                    )
                )
                try? await manifestStore.writeManifest(currentManifest, for: asset.id)
            }
        }

        let failure = downloadFailure(from: lastError ?? DownloadFailure.invalidResponse("No source succeeded"))
        await observer.onFailure(assetID: asset.id, failure: failure)
        throw lastError ?? failure
    }

    private func transfer(
        file: DownloadFile,
        asset: DownloadAsset,
        source: DownloadFile.Source,
        partialURL: URL,
        initialOffset: Int64,
        metadata: DownloadFileMetadata?,
        manifest: DownloadManifest,
        completedFiles: Int,
        completedBytes: Int64,
        totalBytes: Int64?
    ) async throws -> (
        manifest: DownloadManifest,
        bytesWritten: Int64,
        expectedBytes: Int64?,
        metadata: DownloadFileMetadata?
    ) {
        var request = URLRequest(url: source.url)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.timeoutInterval = 60

        let offset = initialOffset
        let resumeDataURL = try await manifestStore.resumeDataURL(for: asset.id, relativePath: file.relativePath)
        let manifestEntry = manifest.files.first { $0.relativePath == file.relativePath }
        let knownExpectedBytes = manifestEntry?.metadata?.contentLength ?? manifestEntry?.expectedBytes ?? file.expectedSize
        let resumeDataSourceMatches =
            manifestEntry?.selectedSourceLabel == source.label &&
            (manifestEntry?.metadata?.sourceURL == nil || manifestEntry?.metadata?.sourceURL == source.url)
        let resumeData = offset == 0 && resumeDataSourceMatches
            ? (try? Data(contentsOf: resumeDataURL))
            : nil
        let usingResumeData = resumeData?.isEmpty == false
        let resumeDataProgressBase = usingResumeData
            ? clampedDownloadedBytes(max(offset, manifestEntry?.downloadedBytes ?? 0), expectedBytes: knownExpectedBytes)
            : 0

        if offset == 0 && !resumeDataSourceMatches {
            try? fileManager.removeItem(at: resumeDataURL)
        }

        if offset > 0, !usingResumeData {
            request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range")
            if let ifRange = metadata?.etag ?? metadata?.lastModified {
                request.setValue(ifRange, forHTTPHeaderField: "If-Range")
            }
        }

        let normalizer = DownloadProgressNormalizer(
            usingResumeData: usingResumeData,
            rangeOffset: offset,
            resumeBase: resumeDataProgressBase,
            expectedBytes: knownExpectedBytes
        )
        let tracker = DownloadProgressAccumulator(
            downloadedBytes: usingResumeData ? resumeDataProgressBase : offset,
            expectedBytes: knownExpectedBytes
        )
        let progressRecorder = DownloadProgressRecorder(
            manifestStore: manifestStore,
            observer: observer,
            assetID: asset.id
        )

        let transfer = ModelFileTransfer()
        let result: ModelFileDownloadResult
        do {
            result = try await transfer.start(
                request: request,
                resumeData: resumeData
            ) { bytesWritten, totalBytesExpected in
                let expectedBytes: Int64? = {
                    if totalBytesExpected > 0 {
                        if usingResumeData {
                            return expectedBytesForResumeDataProgress(
                                taskBytesExpected: totalBytesExpected,
                                resumeBase: resumeDataProgressBase,
                                declaredExpectedBytes: knownExpectedBytes
                            )
                        }
                        return offset + totalBytesExpected
                    }
                    return knownExpectedBytes
                }()
                let normalizedProgress = normalizer.progress(
                    taskBytesWritten: bytesWritten,
                    taskBytesExpected: totalBytesExpected,
                    currentExpectedBytes: expectedBytes
                )
                let progress = tracker.update(
                    downloadedBytes: normalizedProgress.downloadedBytes,
                    expectedBytes: expectedBytes,
                    resetBaseline: normalizedProgress.resetBaseline
                )
                let downloadedBytes = progress.downloadedBytes
                let bytesPerSecond = progress.bytesPerSecond

                Task {
                    let updated = updatedManifestForProgress(
                        manifest,
                        asset: asset,
                        file: file,
                        source: source,
                        downloadedBytes: downloadedBytes,
                        expectedBytes: expectedBytes,
                        metadata: metadata
                    )
                    let snapshot = DownloadProgressSnapshot(
                        assetID: asset.id,
                        completedFileCount: completedFiles,
                        totalFileCount: asset.files.count,
                        downloadedBytes: completedBytes + downloadedBytes,
                        totalBytes: adjustedTotalBytes(
                            configuredTotalBytes: totalBytes,
                            completedBytes: completedBytes,
                            file: file,
                            expectedBytes: expectedBytes
                        ),
                        bytesPerSecond: bytesPerSecond,
                        activeFilePath: file.relativePath,
                        activeSourceLabel: source.label,
                        phase: .downloading,
                        updatedAt: Date()
                    )
                    await progressRecorder.record(
                        manifest: updated,
                        snapshot: snapshot,
                        resetsBaseline: normalizedProgress.resetBaseline
                    )
                }
            }
            await progressRecorder.finish()
            try? fileManager.removeItem(at: resumeDataURL)
        } catch let error as ModelFileTransferError {
            await progressRecorder.finish()
            if let resumeData = error.resumeData {
                try? resumeData.write(to: resumeDataURL, options: [.atomic])
            }
            if error.isCancellation {
                throw CancellationError()
            }
            throw error.underlyingError ?? error
        }

        let httpResponse = result.response
        let temporaryFileURL = result.fileURL

        if offset > 0, !usingResumeData, httpResponse.statusCode == 416 {
            try? fileManager.removeItem(at: partialURL)
            try? fileManager.removeItem(at: resumeDataURL)
            throw DownloadFailure.validatorMismatch(
                expected: "valid byte range from \(offset)",
                actual: "HTTP 416",
                field: "Range"
            )
        }

        guard (200..<300).contains(httpResponse.statusCode) else {
            throw DownloadFailure.httpStatus(httpResponse.statusCode)
        }

        let rangeWasAccepted = offset > 0 && !usingResumeData && httpResponse.statusCode == 206
        let serverIgnoredRange = offset > 0 && !usingResumeData && httpResponse.statusCode == 200

        if offset > 0, !usingResumeData, !rangeWasAccepted, !serverIgnoredRange {
            throw DownloadFailure.validatorMismatch(
                expected: "206 Partial Content or restartable 200 OK",
                actual: "HTTP \(httpResponse.statusCode)",
                field: "Range"
            )
        }

        let responseMetadata = makeMetadata(
            from: httpResponse,
            source: source,
            offset: rangeWasAccepted ? offset : 0,
            requireContentRangeForTotalLength: usingResumeData
        )
        let serverAuthoritativeBytes = responseMetadata.contentLength ?? metadata?.contentLength
        let expectedBytes = serverAuthoritativeBytes ?? tracker.expectedBytes ?? file.expectedSize

        if rangeWasAccepted {
            try appendDownloadedFile(temporaryFileURL, to: partialURL)
        } else {
            try? fileManager.removeItem(at: partialURL)
            try fileManager.moveItem(at: temporaryFileURL, to: partialURL)
        }
        try? fileManager.removeItem(at: temporaryFileURL)

        let bytesReceived = fileSize(partialURL)
        if let authBytes = serverAuthoritativeBytes,
           let sizeMismatch = finalSizeMismatch(bytesReceived, expectedBytes: authBytes) {
            print("[OpenReshot] file-size mismatch; deleting partial. source=\(source.label) actual=\(bytesReceived) expected=\(authBytes) tolerance=\(sizeMismatch.expected)")
            try? fileManager.removeItem(at: partialURL)
            try? fileManager.removeItem(at: resumeDataURL)
            let resetManifest = updatedManifest(
                (try? await manifestStore.readManifest(for: asset.id)) ?? manifest,
                asset: asset,
                replacing: DownloadManifestFile(
                    relativePath: file.relativePath,
                    state: .pending,
                    downloadedBytes: 0,
                    expectedBytes: expectedBytes,
                    selectedSourceLabel: source.label,
                    metadata: responseMetadata
                )
            )
            try? await manifestStore.writeManifest(resetManifest, for: asset.id)
            throw DownloadFailure.validatorMismatch(
                expected: sizeMismatch.expected,
                actual: sizeMismatch.actual,
                field: "file-size"
            )
        } else if serverAuthoritativeBytes == nil {
            print("[OpenReshot] no Content-Length from \(source.label); trusting received \(bytesReceived) bytes")
        }

        var currentManifest = (try? await manifestStore.readManifest(for: asset.id)) ?? manifest
        currentManifest = try await persistProgress(
            manifest: currentManifest,
            asset: asset,
            file: file,
            source: source,
            bytesReceived: bytesReceived,
            expectedBytes: expectedBytes,
            metadata: responseMetadata,
            completedFiles: completedFiles,
            completedBytes: completedBytes,
            totalBytes: totalBytes,
            bytesPerSecond: 0
        )

        return (currentManifest, bytesReceived, expectedBytes, responseMetadata)
    }

    private func persistProgress(
        manifest: DownloadManifest,
        asset: DownloadAsset,
        file: DownloadFile,
        source: DownloadFile.Source,
        bytesReceived: Int64,
        expectedBytes: Int64?,
        metadata: DownloadFileMetadata?,
        completedFiles: Int,
        completedBytes: Int64,
        totalBytes: Int64?,
        bytesPerSecond: Double
    ) async throws -> DownloadManifest {
        let updated = updatedManifest(
            manifest,
            asset: asset,
            replacing: DownloadManifestFile(
                relativePath: file.relativePath,
                state: .downloading,
                downloadedBytes: bytesReceived,
                expectedBytes: expectedBytes,
                selectedSourceLabel: source.label,
                metadata: metadata
            )
        )
        try await manifestStore.writeManifest(updated, for: asset.id)
        await observer.onProgress(
            DownloadProgressSnapshot(
                assetID: asset.id,
                completedFileCount: completedFiles,
                totalFileCount: asset.files.count,
                downloadedBytes: completedBytes + bytesReceived,
                totalBytes: adjustedTotalBytes(
                    configuredTotalBytes: totalBytes,
                    completedBytes: completedBytes,
                    file: file,
                    expectedBytes: expectedBytes
                ),
                bytesPerSecond: bytesPerSecond > 0 ? bytesPerSecond : nil,
                activeFilePath: file.relativePath,
                activeSourceLabel: source.label,
                phase: .downloading,
                updatedAt: Date()
            )
        )
        return updated
    }

    private func resumePlan(
        assetID: String,
        file: DownloadFile,
        source: DownloadFile.Source,
        partialURL: URL,
        manifest: DownloadManifest
    ) async throws -> (offset: Int64, metadata: DownloadFileMetadata?, restart: Bool) {
        let existingBytes = fileSize(partialURL)
        guard existingBytes > 0 else { return (0, nil, false) }

        if let expected = file.expectedSize, existingBytes > expected {
            return (0, nil, true)
        }

        guard let entry = manifest.files.first(where: { $0.relativePath == file.relativePath }) else {
            return (existingBytes, nil, false)
        }

        let headMetadata = try? await fetchHeadMetadata(for: source)

        guard let storedMetadata = entry.metadata else {
            if let headMetadata {
                if let currentLength = headMetadata.contentLength, currentLength < existingBytes {
                    return (0, headMetadata, true)
                }
                return (existingBytes, headMetadata, false)
            }

            return (existingBytes, nil, false)
        }

        if let headMetadata, validatorsMatch(stored: storedMetadata, current: headMetadata) {
            return (existingBytes, headMetadata, false)
        }

        if headMetadata == nil {
            return (existingBytes, storedMetadata.sourceURL == source.url ? storedMetadata : nil, false)
        }

        if let currentLength = headMetadata?.contentLength {
            return currentLength >= existingBytes ? (existingBytes, headMetadata, false) : (0, headMetadata, true)
        }

        return (0, headMetadata, true)
    }

    private func fetchHeadMetadata(for source: DownloadFile.Source) async throws -> DownloadFileMetadata {
        var request = URLRequest(url: source.url)
        request.httpMethod = "HEAD"
        request.timeoutInterval = 20

        let (_, response) = try await urlSession.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw DownloadFailure.invalidResponse("Missing HEAD response")
        }
        guard (200..<400).contains(httpResponse.statusCode) else {
            throw DownloadFailure.httpStatus(httpResponse.statusCode)
        }
        return makeMetadata(from: httpResponse, source: source, offset: 0)
    }

    private func validatorsMatch(stored: DownloadFileMetadata, current: DownloadFileMetadata) -> Bool {
        if let storedChecksum = stored.checksumSHA256,
           let currentChecksum = current.checksumSHA256,
           storedChecksum == currentChecksum {
            return true
        }
        if let storedETag = stored.etag, let currentETag = current.etag, storedETag == currentETag {
            return true
        }
        if let storedLength = stored.contentLength,
           let currentLength = current.contentLength,
           storedLength == currentLength {
            return true
        }
        if let storedModified = stored.lastModified,
           let currentModified = current.lastModified,
           storedModified == currentModified {
            return true
        }
        return false
    }

    private func makeMetadata(
        from response: HTTPURLResponse,
        source: DownloadFile.Source,
        offset: Int64,
        requireContentRangeForTotalLength: Bool = false
    ) -> DownloadFileMetadata {
        let responseLength = response.expectedContentLength > 0 ? response.expectedContentLength : nil
        let contentRangeTotal = header("Content-Range", from: response).flatMap(parseContentRangeTotal)
        let totalLength: Int64? = requireContentRangeForTotalLength
            ? contentRangeTotal
            : (contentRangeTotal ?? responseLength.map { offset + $0 })

        return DownloadFileMetadata(
            sourceURL: source.url,
            sourceHost: source.url.host,
            etag: header("ETag", from: response),
            contentLength: totalLength,
            lastModified: header("Last-Modified", from: response),
            checksumSHA256: nil,
            updatedAt: Date()
        )
    }

    private func parseContentRangeTotal(_ value: String) -> Int64? {
        guard let slashIndex = value.lastIndex(of: "/") else { return nil }
        let suffix = value[value.index(after: slashIndex)...]
        guard suffix != "*" else { return nil }
        return Int64(suffix)
    }

    private func header(_ name: String, from response: HTTPURLResponse) -> String? {
        for (key, value) in response.allHeaderFields {
            if String(describing: key).caseInsensitiveCompare(name) == .orderedSame {
                return String(describing: value)
            }
        }
        return nil
    }

    private func appendDownloadedFile(_ sourceURL: URL, to destinationURL: URL) throws {
        try createDirectoryExcludedFromBackup(destinationURL.deletingLastPathComponent())
        if !fileManager.fileExists(atPath: destinationURL.path) {
            fileManager.createFile(atPath: destinationURL.path, contents: nil)
        }

        let input = try FileHandle(forReadingFrom: sourceURL)
        defer { try? input.close() }
        let output = try FileHandle(forWritingTo: destinationURL)
        defer { try? output.close() }
        try output.seekToEnd()

        while true {
            let data = try input.read(upToCount: 1024 * 1024) ?? Data()
            if data.isEmpty { break }
            try output.write(contentsOf: data)
        }
    }

    private func readManifestOrRestart(assetID: String) async throws -> DownloadManifest? {
        do {
            return try await manifestStore.readManifest(for: assetID)
        } catch let failure as DownloadFailure {
            if case .manifestCorrupt = failure {
                print("[OpenReshot] manifest corrupt for \(assetID); restarting download")
                try await manifestStore.purge(assetID: assetID)
                return nil
            }
            throw failure
        }
    }

    private func preflightDiskSpace(for asset: DownloadAsset) async throws {
        let required = try await remainingExpectedBytes(for: asset)
        guard required > 0 else { return }
        try DownloadPreflight.validateDiskSpace(requiredBytes: required, at: asset.destinationDirectory)
    }

    private func remainingExpectedBytes(for asset: DownloadAsset) async throws -> Int64 {
        var required: Int64 = 0
        for file in asset.files {
            guard let expected = file.expectedSize, expected > 0 else { continue }
            let finalURL = asset.destinationDirectory.appendingPathComponent(file.relativePath, isDirectory: false)
            let finalBytes = fileSize(finalURL)
            if finalBytes >= expected * 9 / 10 { continue }

            let partialURL = try await manifestStore.partialFileURL(for: asset.id, relativePath: file.relativePath)
            let remaining = max(0, expected - fileSize(partialURL))
            required += remaining
        }
        return required
    }

    private func totalExpectedBytes(for asset: DownloadAsset) -> Int64? {
        var total: Int64 = 0
        for file in asset.files {
            guard let expected = file.expectedSize, expected > 0 else { return nil }
            total += expected
        }
        return total
    }

    private func freshManifest(for asset: DownloadAsset, now: Date) -> DownloadManifest {
        DownloadManifest(
            assetID: asset.id,
            createdAt: now,
            updatedAt: now,
            files: asset.files.map {
                DownloadManifestFile(
                    relativePath: $0.relativePath,
                    state: .pending,
                    downloadedBytes: 0,
                    expectedBytes: $0.expectedSize
                )
            }
        )
    }

    private func updatedManifest(
        _ manifest: DownloadManifest,
        asset: DownloadAsset,
        replacing replacement: DownloadManifestFile
    ) -> DownloadManifest {
        let existing = Dictionary(uniqueKeysWithValues: manifest.files.map { ($0.relativePath, $0) })
        let files = asset.files.map { file -> DownloadManifestFile in
            if file.relativePath == replacement.relativePath {
                return replacement
            }
            return existing[file.relativePath] ?? DownloadManifestFile(
                relativePath: file.relativePath,
                state: .pending,
                downloadedBytes: 0,
                expectedBytes: file.expectedSize
            )
        }

        return DownloadManifest(
            schemaVersion: manifest.schemaVersion,
            assetID: manifest.assetID,
            createdAt: manifest.createdAt,
            updatedAt: Date(),
            files: files
        )
    }

    private func orderedSources(_ sources: [DownloadFile.Source]) -> [DownloadFile.Source] {
        sources.sorted {
            if $0.priority == $1.priority {
                return $0.label < $1.label
            }
            return $0.priority < $1.priority
        }
    }

    private func fileSize(_ url: URL) -> Int64 {
        guard let attrs = try? fileManager.attributesOfItem(atPath: url.path) else { return 0 }
        return (attrs[.size] as? Int64) ?? 0
    }

    private func finalSizeMismatch(_ bytes: Int64, expectedBytes: Int64?) -> (expected: String, actual: String)? {
        guard let expectedBytes, expectedBytes > 0 else { return nil }
        let tolerance = progressTolerance(for: expectedBytes)
        let lowerBound = max(0, expectedBytes - tolerance)
        let upperBound = expectedBytes + tolerance
        guard bytes < lowerBound || bytes > upperBound else { return nil }
        return ("\(lowerBound)...\(upperBound)", "\(bytes)")
    }

    private func downloadFailure(from error: Error) -> DownloadFailure {
        if let failure = error as? DownloadFailure {
            return failure
        }
        if isCancellation(error) {
            return .cancelled
        }
        return .invalidResponse(error.localizedDescription)
    }

    private func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError || Task.isCancelled {
            return true
        }
        if let transferError = error as? ModelFileTransferError, transferError.isCancellation {
            return true
        }
        if let urlError = error as? URLError, urlError.code == .cancelled {
            return true
        }
        return false
    }
}

private struct NormalizedDownloadProgress: Sendable {
    let downloadedBytes: Int64
    let resetBaseline: Bool
}

private struct DownloadProgressNormalizer: Sendable {
    let usingResumeData: Bool
    let rangeOffset: Int64
    let resumeBase: Int64
    let expectedBytes: Int64?

    func progress(
        taskBytesWritten: Int64,
        taskBytesExpected: Int64,
        currentExpectedBytes: Int64?
    ) -> NormalizedDownloadProgress {
        let expectedBytes = currentExpectedBytes ?? self.expectedBytes

        if usingResumeData {
            let resumeProgress = resumeDataProgress(
                taskBytesWritten: taskBytesWritten,
                taskBytesExpected: taskBytesExpected,
                expectedBytes: expectedBytes
            )
            return NormalizedDownloadProgress(
                downloadedBytes: clampedDownloadedBytes(resumeProgress.downloadedBytes, expectedBytes: expectedBytes),
                resetBaseline: resumeProgress.resetBaseline
            )
        } else {
            return NormalizedDownloadProgress(
                downloadedBytes: clampedDownloadedBytes(rangeOffset + taskBytesWritten, expectedBytes: expectedBytes),
                resetBaseline: false
            )
        }
    }

    private func resumeDataProgress(
        taskBytesWritten: Int64,
        taskBytesExpected: Int64,
        expectedBytes: Int64?
    ) -> (downloadedBytes: Int64, resetBaseline: Bool) {
        guard resumeBase > 0 else { return (taskBytesWritten, false) }

        if let expectedBytes, expectedBytes > 0, taskBytesExpected > 0 {
            let tolerance = progressTolerance(for: expectedBytes)

            if taskBytesExpected >= expectedBytes - tolerance {
                return (taskBytesWritten, taskBytesWritten < resumeBase)
            }

            if resumeBase + taskBytesExpected <= expectedBytes + tolerance {
                return (resumeBase + taskBytesWritten, false)
            }
        }

        if taskBytesWritten >= resumeBase {
            return (taskBytesWritten, false)
        }

        return (resumeBase + taskBytesWritten, false)
    }
}

private struct ModelFileDownloadResult: Sendable {
    let fileURL: URL
    let response: HTTPURLResponse
}

private enum ModelFileTransferError: Error {
    case cancelled(resumeData: Data?)
    case failed(Error, resumeData: Data?)
    case missingDownloadedFile
    case invalidResponse

    var resumeData: Data? {
        switch self {
        case .cancelled(let resumeData), .failed(_, let resumeData):
            return resumeData
        case .missingDownloadedFile, .invalidResponse:
            return nil
        }
    }

    var underlyingError: Error? {
        switch self {
        case .failed(let error, _):
            return error
        case .cancelled, .missingDownloadedFile, .invalidResponse:
            return nil
        }
    }

    var isCancellation: Bool {
        if case .cancelled = self { return true }
        return false
    }
}

private final class ModelFileTransfer: NSObject, URLSessionDownloadDelegate, URLSessionTaskDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var session: URLSession?
    private var task: URLSessionDownloadTask?
    private var continuation: CheckedContinuation<ModelFileDownloadResult, Error>?
    private var result: Result<ModelFileDownloadResult, Error>?
    private var downloadedFileURL: URL?
    private var progress: (@Sendable (_ bytesWritten: Int64, _ totalBytesExpected: Int64) -> Void)?

    func start(
        request: URLRequest,
        resumeData: Data?,
        progress: @escaping @Sendable (_ bytesWritten: Int64, _ totalBytesExpected: Int64) -> Void
    ) async throws -> ModelFileDownloadResult {
        self.progress = progress
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let configuration = URLSessionConfiguration.default
                configuration.waitsForConnectivity = true
                configuration.timeoutIntervalForRequest = 60
                configuration.timeoutIntervalForResource = 60 * 60 * 12
                configuration.allowsConstrainedNetworkAccess = true
                configuration.allowsExpensiveNetworkAccess = true

                let session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
                let task: URLSessionDownloadTask
                if let resumeData, !resumeData.isEmpty {
                    task = session.downloadTask(withResumeData: resumeData)
                } else {
                    task = session.downloadTask(with: request)
                }
                task.taskDescription = request.url?.absoluteString

                lock.lock()
                self.session = session
                self.task = task
                self.continuation = continuation
                let pendingResult = self.result
                lock.unlock()

                if let pendingResult {
                    continuation.resume(with: pendingResult)
                    return
                }

                task.resume()
            }
        } onCancel: {
            self.cancel()
        }
    }

    func cancel() {
        lock.lock()
        let currentTask = task
        lock.unlock()

        guard let currentTask else {
            finish(.failure(ModelFileTransferError.cancelled(resumeData: nil)))
            return
        }

        currentTask.cancel { [weak self] resumeData in
            self?.finish(.failure(ModelFileTransferError.cancelled(resumeData: resumeData)))
        }
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        progress?(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        do {
            let destination = try makeTemporaryDownloadURL()
            try? FileManager.default.removeItem(at: destination)
            do {
                try FileManager.default.moveItem(at: location, to: destination)
            } catch {
                try FileManager.default.copyItem(at: location, to: destination)
            }

            lock.lock()
            downloadedFileURL = destination
            lock.unlock()
        } catch {
            finish(.failure(ModelFileTransferError.failed(error, resumeData: nil)))
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        if let error {
            let nsError = error as NSError
            let resumeData = nsError.userInfo[NSURLSessionDownloadTaskResumeData] as? Data
            if nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorCancelled {
                finish(.failure(ModelFileTransferError.cancelled(resumeData: resumeData)))
            } else {
                finish(.failure(ModelFileTransferError.failed(error, resumeData: resumeData)))
            }
            return
        }

        guard let response = task.response as? HTTPURLResponse else {
            finish(.failure(ModelFileTransferError.invalidResponse))
            return
        }

        lock.lock()
        let fileURL = downloadedFileURL
        lock.unlock()

        guard let fileURL else {
            finish(.failure(ModelFileTransferError.missingDownloadedFile))
            return
        }
        finish(.success(ModelFileDownloadResult(fileURL: fileURL, response: response)))
    }

    private func finish(_ result: Result<ModelFileDownloadResult, Error>) {
        var continuationToResume: CheckedContinuation<ModelFileDownloadResult, Error>?
        lock.lock()
        guard self.result == nil else {
            lock.unlock()
            return
        }
        self.result = result
        continuationToResume = continuation
        continuation = nil
        task = nil
        let sessionToInvalidate = session
        session = nil
        lock.unlock()

        sessionToInvalidate?.invalidateAndCancel()
        continuationToResume?.resume(with: result)
    }

    private func makeTemporaryDownloadURL() throws -> URL {
        let directory = FileManager.default
            .urls(for: .cachesDirectory, in: .userDomainMask)
            .first!
            .appendingPathComponent("OpenReshotBackgroundDownloads", isDirectory: true)
        try createDirectoryExcludedFromBackup(directory)
        return directory.appendingPathComponent(UUID().uuidString, isDirectory: false)
    }
}

private final class DownloadProgressAccumulator: @unchecked Sendable {
    private let lock = NSLock()
    private var storedDownloadedBytes: Int64
    private var storedExpectedBytes: Int64?
    private var lastSpeedUpdate: CFAbsoluteTime
    private var lastSpeedBytes: Int64
    private var smoothedBytesPerSecond: Double = 0

    init(downloadedBytes: Int64, expectedBytes: Int64?) {
        self.storedDownloadedBytes = clampedDownloadedBytes(downloadedBytes, expectedBytes: expectedBytes)
        self.storedExpectedBytes = expectedBytes
        self.lastSpeedUpdate = CFAbsoluteTimeGetCurrent()
        self.lastSpeedBytes = self.storedDownloadedBytes
    }

    var downloadedBytes: Int64 {
        lock.lock()
        defer { lock.unlock() }
        return storedDownloadedBytes
    }

    var expectedBytes: Int64? {
        lock.lock()
        defer { lock.unlock() }
        return storedExpectedBytes
    }

    func update(
        downloadedBytes: Int64,
        expectedBytes: Int64?,
        resetBaseline: Bool = false
    ) -> (downloadedBytes: Int64, bytesPerSecond: Double?) {
        lock.lock()
        let effectiveDownloadedBytes = clampedDownloadedBytes(downloadedBytes, expectedBytes: expectedBytes)
        let nextDownloadedBytes = resetBaseline
            ? effectiveDownloadedBytes
            : max(storedDownloadedBytes, effectiveDownloadedBytes)
        storedDownloadedBytes = nextDownloadedBytes
        if let expectedBytes {
            storedExpectedBytes = expectedBytes
        }
        let now = CFAbsoluteTimeGetCurrent()
        let elapsed = now - lastSpeedUpdate
        if resetBaseline {
            smoothedBytesPerSecond = 0
            lastSpeedUpdate = now
            lastSpeedBytes = nextDownloadedBytes
        } else if elapsed > 0.5 {
            let delta = nextDownloadedBytes - lastSpeedBytes
            if delta > 0 {
                let instantSpeed = Double(delta) / elapsed
                smoothedBytesPerSecond = smoothedBytesPerSecond > 0
                    ? smoothedBytesPerSecond * 0.7 + instantSpeed * 0.3
                    : instantSpeed
            }
            lastSpeedUpdate = now
            lastSpeedBytes = nextDownloadedBytes
        }
        let speed = smoothedBytesPerSecond > 0 ? smoothedBytesPerSecond : nil
        lock.unlock()
        return (nextDownloadedBytes, speed)
    }
}

private func createDirectoryExcludedFromBackup(_ url: URL) throws {
    let fileManager = FileManager.default
    if !fileManager.fileExists(atPath: url.path) {
        try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
    }
    var mutableURL = url
    var values = URLResourceValues()
    values.isExcludedFromBackup = true
    try mutableURL.setResourceValues(values)
}

private func clampedDownloadedBytes(_ bytes: Int64, expectedBytes: Int64?) -> Int64 {
    let positiveBytes = max(0, bytes)
    guard let expectedBytes, expectedBytes > 0 else { return positiveBytes }
    return min(positiveBytes, expectedBytes)
}

private func expectedBytesForResumeDataProgress(
    taskBytesExpected: Int64,
    resumeBase: Int64,
    declaredExpectedBytes: Int64?
) -> Int64 {
    guard taskBytesExpected > 0 else {
        return declaredExpectedBytes ?? max(0, resumeBase)
    }

    guard let declaredExpectedBytes, declaredExpectedBytes > 0, resumeBase > 0 else {
        return taskBytesExpected
    }

    let fullTransferDelta = abs(taskBytesExpected - declaredExpectedBytes)
    let incrementalTransferTotal = resumeBase + taskBytesExpected
    let incrementalTransferDelta = abs(incrementalTransferTotal - declaredExpectedBytes)
    return fullTransferDelta <= incrementalTransferDelta
        ? taskBytesExpected
        : incrementalTransferTotal
}

private func adjustedTotalBytes(
    configuredTotalBytes: Int64?,
    completedBytes: Int64,
    file: DownloadFile,
    expectedBytes: Int64?
) -> Int64? {
    guard let expectedBytes, expectedBytes > 0 else {
        return configuredTotalBytes
    }
    guard let configuredTotalBytes, let configuredFileBytes = file.expectedSize, configuredFileBytes > 0 else {
        return completedBytes + expectedBytes
    }
    return max(0, configuredTotalBytes - configuredFileBytes + expectedBytes)
}

private func progressTolerance(for expectedBytes: Int64) -> Int64 {
    max(1, min(Int64(1_048_576), expectedBytes / 1000))
}

private func updatedManifestForProgress(
    _ manifest: DownloadManifest,
    asset: DownloadAsset,
    file: DownloadFile,
    source: DownloadFile.Source,
    downloadedBytes: Int64,
    expectedBytes: Int64?,
    metadata: DownloadFileMetadata?
) -> DownloadManifest {
    let replacement = DownloadManifestFile(
        relativePath: file.relativePath,
        state: .downloading,
        downloadedBytes: downloadedBytes,
        expectedBytes: expectedBytes,
        selectedSourceLabel: source.label,
        metadata: metadata
    )
    let existing = Dictionary(uniqueKeysWithValues: manifest.files.map { ($0.relativePath, $0) })
    let files = asset.files.map { candidate -> DownloadManifestFile in
        if candidate.relativePath == file.relativePath {
            return replacement
        }
        return existing[candidate.relativePath] ?? DownloadManifestFile(
            relativePath: candidate.relativePath,
            state: .pending,
            downloadedBytes: 0,
            expectedBytes: candidate.expectedSize
        )
    }
    return DownloadManifest(
        schemaVersion: manifest.schemaVersion,
        assetID: manifest.assetID,
        createdAt: manifest.createdAt,
        updatedAt: Date(),
        files: files
    )
}
