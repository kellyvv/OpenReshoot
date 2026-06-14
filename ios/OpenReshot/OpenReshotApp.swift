import SwiftUI
import Foundation
import Photos
import PhotosUI
import MetalKit
import AVFoundation
import CoreML
import CoreImage
import CoreImage.CIFilterBuiltins
import ImageIO
import UniformTypeIdentifiers
import Vision
import simd
import SplatIO
import CryptoKit

@main
struct OpenReshotApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

enum RenderQuality: String, CaseIterable, Identifiable {
    case high
    case smooth

    var id: String { rawValue }

    var title: String {
        switch self {
        case .high: return "高清"
        case .smooth: return "流畅"
        }
    }

    var systemImage: String {
        switch self {
        case .high: return "sparkles"
        case .smooth: return "bolt.fill"
        }
    }

    func splatBudget(memoryGB: UInt64) -> Int {
        switch self {
        case .high:
            return memoryGB >= 8 ? 1_200_000 : 320_000
        case .smooth:
            return memoryGB >= 8 ? 320_000 : 180_000
        }
    }

    func renderScale(memoryGB: UInt64) -> CGFloat {
        switch self {
        case .high:
            return memoryGB >= 8 ? 2.0 : 1.5
        case .smooth:
            return memoryGB >= 8 ? 1.35 : 1.2
        }
    }

    var preferredFrameRate: Int {
        switch self {
        case .high: return 24
        case .smooth: return 30
        }
    }

}

enum ReshotViewAngleMode: String, CaseIterable, Identifiable {
    case standard
    case wide

    var id: String { rawValue }

    var title: String {
        switch self {
        case .standard: return "标准"
        case .wide: return "大角度"
        }
    }

    var motionScale: Float {
        switch self {
        case .standard: return 0.58
        case .wide: return 1.0
        }
    }

    var exportTiltRange: Float {
        switch self {
        case .standard: return 0.46
        case .wide: return 0.72
        }
    }
}

enum SaveState {
    case idle
    case saving
    case saved
    case failed
}

enum MotionExportState {
    case idle
    case rendering
    case saved
    case failed
}

enum MotionExportFormat: String, CaseIterable, Identifiable {
    case mp4
    case livePhoto
    case momentsLivePhoto
    case gif

    var id: String { rawValue }

    var isLivePhotoPackage: Bool {
        switch self {
        case .livePhoto, .momentsLivePhoto: return true
        case .mp4, .gif: return false
        }
    }

    var title: String {
        switch self {
        case .mp4: return "MP4"
        case .livePhoto: return "Live"
        case .momentsLivePhoto: return "朋友圈"
        case .gif: return "GIF"
        }
    }

    var systemImage: String {
        switch self {
        case .mp4: return "play.rectangle"
        case .livePhoto: return "livephoto"
        case .momentsLivePhoto: return "bubble.left.and.bubble.right"
        case .gif: return "sparkles.rectangle.stack"
        }
    }

    var savedMessage: String {
        switch self {
        case .mp4: return "MP4 已保存"
        case .livePhoto: return "Live Photo 已保存"
        case .momentsLivePhoto: return "朋友圈 Live 已保存"
        case .gif: return "GIF 已保存"
        }
    }
}

enum ProcessFailureKind: Equatable {
    case imageLoad
    case missingModel
    case reconstruction
    case missingAPIKey
    case enhancement
}

private struct PreviewExportMetadata: Codable, Sendable {
    let formatVersion: Int
    let splatFormat: String
    let splatCount: Int
    let focus: Float
    let focalPixels: Float
    let width: Int
    let height: Int
    let sourceImageName: String
    let splatFileName: String
}

struct ReshotCacheItem: Codable, Identifiable, Equatable {
    let id: String
    let formatVersion: Int
    let createdAt: Date
    let sourceSignature: String
    let quality: String
    let splatCount: Int
    let focus: Float
    let focalPixels: Float
    let width: Int
    let height: Int
    let sourceImageName: String
    let thumbnailImageName: String
    let splatFileName: String
}

private enum ReshotCacheStore {
    private static let bundledDemoID = "openreshot-bundled-demo"
    private static let bundledDemoResource = "DemoFLOW"
    private static let folderName = "ReshotCache"
    private static let metadataFileName = "Reshot.json"

    static func galleryItems() -> [ReshotCacheItem] {
        if let demo = bundledDemoItem {
            return [demo] + loadItems()
        }
        return loadItems()
    }

    static func loadItems() -> [ReshotCacheItem] {
        guard let root = try? rootDirectory(create: false),
              let urls = try? FileManager.default.contentsOfDirectory(
                at: root,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
              ) else {
            return []
        }

        return urls.compactMap { directory -> ReshotCacheItem? in
            guard (try? directory.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true else { return nil }
            let metadataURL = directory.appendingPathComponent(metadataFileName)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            guard let data = try? Data(contentsOf: metadataURL),
                  let item = try? decoder.decode(ReshotCacheItem.self, from: data) else {
                return nil
            }
            return item
        }
        .sorted { $0.createdAt > $1.createdAt }
    }

    static func item(matching sourceSignature: String, quality: RenderQuality) -> ReshotCacheItem? {
        let items = loadItems().filter { $0.sourceSignature == sourceSignature }
        return items.first { $0.quality == quality.rawValue } ?? items.first
    }

    static func sourceURL(for item: ReshotCacheItem) -> URL? {
        if isBundledItem(item) {
            return Bundle.main.url(forResource: bundledDemoResource, withExtension: "png")
        }
        return cacheDirectory(for: item)?.appendingPathComponent(item.sourceImageName)
    }

    static func thumbnailURL(for item: ReshotCacheItem) -> URL? {
        if isBundledItem(item) {
            return Bundle.main.url(forResource: bundledDemoResource, withExtension: "png")
        }
        return cacheDirectory(for: item)?.appendingPathComponent(item.thumbnailImageName)
    }

    static func splatURL(for item: ReshotCacheItem) -> URL? {
        if isBundledItem(item) {
            return Bundle.main.url(forResource: bundledDemoResource, withExtension: "ply")
        }
        return cacheDirectory(for: item)?.appendingPathComponent(item.splatFileName)
    }

    static func isBundledItem(_ item: ReshotCacheItem) -> Bool {
        item.id == bundledDemoID
    }

    static func delete(_ item: ReshotCacheItem) throws {
        guard !isBundledItem(item) else { return }
        guard let directory = cacheDirectory(for: item) else { return }
        try FileManager.default.removeItem(at: directory)
    }

    static func save(
        points: [SplatPoint],
        sourceImage: UIImage,
        sourceSignature: String,
        quality: RenderQuality,
        focus: Float,
        focalPixels: Float,
        width: Int,
        height: Int
    ) async throws -> ReshotCacheItem {
        try await Task.detached(priority: .utility) {
            let id = "\(Int(Date().timeIntervalSince1970))-\(UUID().uuidString.prefix(8))"
            let sourceImageName = "source.jpg"
            let thumbnailImageName = "thumb.jpg"
            let splatFileName = "scene.ply"
            let item = ReshotCacheItem(
                id: id,
                formatVersion: 1,
                createdAt: Date(),
                sourceSignature: sourceSignature,
                quality: quality.rawValue,
                splatCount: points.count,
                focus: focus,
                focalPixels: focalPixels,
                width: width,
                height: height,
                sourceImageName: sourceImageName,
                thumbnailImageName: thumbnailImageName,
                splatFileName: splatFileName
            )

            let root = try rootDirectory(create: true)
            let finalDirectory = root.appendingPathComponent(id, isDirectory: true)
            let temporaryDirectory = root.appendingPathComponent(".\(id)-writing", isDirectory: true)
            let fileManager = FileManager.default
            try? fileManager.removeItem(at: temporaryDirectory)
            try fileManager.createDirectory(at: temporaryDirectory, withIntermediateDirectories: true)

            guard let sourceData = sourceImage.jpegData(compressionQuality: 0.94),
                  let thumbnailData = thumbnailImage(from: sourceImage).jpegData(compressionQuality: 0.82) else {
                throw err("缓存图片编码失败")
            }
            try sourceData.write(to: temporaryDirectory.appendingPathComponent(sourceImageName), options: [.atomic])
            try thumbnailData.write(to: temporaryDirectory.appendingPathComponent(thumbnailImageName), options: [.atomic])

            let plyURL = temporaryDirectory.appendingPathComponent(splatFileName)
            let writer = try SplatPLYSceneWriter(toFileAtPath: plyURL.path)
            try await writer.start(sphericalHarmonicDegree: 0, binary: true, pointCount: points.count)
            try await writer.write(points)
            try await writer.close()

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            try encoder.encode(item).write(to: temporaryDirectory.appendingPathComponent(metadataFileName), options: [.atomic])

            try? fileManager.removeItem(at: finalDirectory)
            try fileManager.moveItem(at: temporaryDirectory, to: finalDirectory)
            return item
        }.value
    }

    private static func rootDirectory(create: Bool) throws -> URL {
        let base = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let root = base.appendingPathComponent(folderName, isDirectory: true)
        if create {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        }
        return root
    }

    private static func cacheDirectory(for item: ReshotCacheItem) -> URL? {
        try? rootDirectory(create: false).appendingPathComponent(item.id, isDirectory: true)
    }

    private static var bundledDemoItem: ReshotCacheItem? {
        guard Bundle.main.url(forResource: bundledDemoResource, withExtension: "png") != nil,
              Bundle.main.url(forResource: bundledDemoResource, withExtension: "ply") != nil else {
            return nil
        }
        return ReshotCacheItem(
            id: bundledDemoID,
            formatVersion: 1,
            createdAt: Date(timeIntervalSince1970: 0),
            sourceSignature: bundledDemoID,
            quality: "demo",
            splatCount: 1_179_648,
            focus: 1.4492188,
            focalPixels: 1330.3168,
            width: 941,
            height: 1672,
            sourceImageName: "\(bundledDemoResource).png",
            thumbnailImageName: "\(bundledDemoResource).png",
            splatFileName: "\(bundledDemoResource).ply"
        )
    }

    private static func thumbnailImage(from image: UIImage) -> UIImage {
        let maxSide: CGFloat = 420
        let longest = max(image.size.width, image.size.height, 1)
        let scale = min(1, maxSide / longest)
        let targetSize = CGSize(width: image.size.width * scale, height: image.size.height * scale)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        return UIGraphicsImageRenderer(size: targetSize, format: format).image { _ in
            image.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }
}

/// Holds the model + renderer and drives reconstruction off the main thread.
final class AppState: ObservableObject {
    @Published var status = ""
    @Published var hasCloud = false
    @Published var rendererReady = false
    @Published var inputImage: UIImage?
    @Published var imageAspect: CGFloat = 1
    @Published var quality: RenderQuality = .high
    @Published var sheenAmount: CGFloat = 0
    @Published var sheenTiltX: CGFloat = 0
    @Published var sheenTiltY: CGFloat = 0
    @Published var motionTilt = SIMD2<Float>(0, 0)
    @Published var previewMode = false
    @Published var loadingPreviewScene = false
    @Published var reconstructingScene = false
    @Published var reconstructingFrame = false
    @Published var processFailed = false
    @Published var processFailureKind: ProcessFailureKind?
    @Published var capturedFrame: UIImage?
    @Published var resultImage: UIImage?
    @Published var saveState: SaveState = .idle
    @Published var motionExportState: MotionExportState = .idle
    @Published var motionExportFormat: MotionExportFormat?
    @Published var motionExportProgress: Double = 0
    @Published var motionExportFailureMessage: String?
    @Published var subjectProtectionMask: UIImage?
    @Published var geminiKey: String
    @Published var hasExportablePreview = false
    @Published var lensFocusDepth: Float = 1
    @Published var lensFocusMin: Float = 0.25
    @Published var lensFocusMax: Float = 4
    @Published var lensFNumber: Float = 16
    @Published var lensDolly: Float = 0
    @Published var viewAngleMode: ReshotViewAngleMode = .wide
    @Published var galleryItems: [ReshotCacheItem] = []
    let modelStore = ReconstructionModelStore()
    var renderer: ReshootRenderer?
    private let modelQueue = DispatchQueue(label: "OpenReshot.model", qos: .userInitiated)
    private var cachedModel: SharpModel?
    private var currentSourceData: Data?
    private var memoryWarningObserver: NSObjectProtocol?
    private var activeTaskID = UUID()
    private var subjectMaskRequestID = UUID()
    private var reconstructionRequestID = UUID()
    private var enhanceTask: Task<Void, Never>?
    private var motionExportTask: Task<Void, Never>?
    private var currentCloudPoints: [SplatPoint]?
    private var currentPreviewMetadata: PreviewExportMetadata?
    private var activeCacheItemID: String?
    private var pendingCachedReshot: ReshotCacheItem?
    private var demoScenePending = false
    private var didLoadDemoScene = false
    private var didStartDemoModelPrefetch = false
    private var lensDefaultFocusDepth: Float = 1
    private static let geminiInputMaxSide: CGFloat = 1024
    private static let motionExportFPS = 24
    private static let motionExportFrameCount = 96
    private static let motionExportTiltRange: Float = 0.34
    private static let livePhotoStillFrameIndex = 0
    private static let momentsLivePhotoFPS = 30
    private static let momentsLivePhotoFrameCount = 90
    private static let momentsLivePhotoStillFrameIndex = 45
    private static let momentsLivePhotoTiltRange: Float = 0.24
    private static let motionExportMaxLongSide: CGFloat = 1920
    private static let geminiModel = "gemini-3.1-flash-image"
    private static let demoSceneResource = "DemoFLOW"
    private static let demoSceneFocalPixels: Float = 1330.3168
    private static let demoSceneFocus: Float = 1.4492188
    private static let demoSceneWidth = 941
    private static let demoSceneHeight = 1672
    private static let didAutoloadDemoSceneKey = "OpenReshot.didAutoloadDemoScene"
    private static var memoryGB: UInt64 { ProcessInfo.processInfo.physicalMemory / 1_073_741_824 }
    private static let enhancePrompt = """
    This image is a novel-view render produced from a 3D Gaussian Splatting reconstruction. Because the camera viewpoint changed, some newly exposed edges, disoccluded regions, stretched areas, warped details, holes, and blurry splat artifacts may appear.

    Repair only those rendering artifacts. Turn blurry splat regions into clear, coherent details. Correct warped, stretched, distorted, or wrong-looking areas, and complete any missing, exposed, or broken parts with realistic detail consistent with the surrounding scene and the original photo. Clarify the composition while keeping the same camera framing, perspective, subject placement, lighting, colors, materials, and overall layout. Do not restyle, beautify, replace, or add new main objects.

    If people are visible, preserve them exactly as shown: same identity, appearance, hair, clothing, pose, expression, and framing. Do not alter people or create any new person.

    Output one single complete sharp and clear photo. 这是 3DGS 新视角渲染结果，把模糊区域变清晰，把拉伸、变形、错误、露底、空洞、缺失和破碎区域修正并补齐，对构图进行清晰化处理，但保持原主体、视角、透视、布局、光线和颜色不变；如果画面里有人，人物必须保持原样。
    """

    init() {
        geminiKey = UserDefaults.standard.string(forKey: "OpenReshot.geminiKey") ?? ""
        galleryItems = ReshotCacheStore.galleryItems()
        memoryWarningObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.modelQueue.async {
                self?.cachedModel = nil
                print("🧹 [OpenReshot] released cached reconstruction model after memory warning")
            }
        }
    }

    deinit {
        if let memoryWarningObserver {
            NotificationCenter.default.removeObserver(memoryWarningObserver)
        }
    }

    @MainActor
    func attachRenderer(_ renderer: ReshootRenderer) {
        self.renderer = renderer
        renderer.setMotionRangeScale(viewAngleMode.motionScale)
        renderer.onReady = { [weak self] in
            guard let self, self.inputImage != nil else { return }
            self.rendererReady = true
            self.loadingPreviewScene = false
            self.reconstructingScene = false
            self.processFailed = false
            self.processFailureKind = nil
            self.status = ""
        }
        renderer.onFailure = { [weak self] message in
            guard let self, self.inputImage != nil else { return }
            self.loadingPreviewScene = false
            self.reconstructingScene = false
            self.rendererReady = false
            self.processFailed = true
            self.processFailureKind = .imageLoad
            self.status = ""
            print("❌ [OpenReshot] renderer scene load failed: \(message)")
        }
        startPendingDemoSceneIfPossible()
        startPendingCachedReshotIfPossible()
    }

    @MainActor
    func loadDemoSceneIfNeeded() {
        guard !didLoadDemoScene, inputImage == nil else { return }
        guard !UserDefaults.standard.bool(forKey: Self.didAutoloadDemoSceneKey) else { return }
        guard let imageURL = Bundle.main.url(forResource: Self.demoSceneResource, withExtension: "png"),
              let sceneURL = Bundle.main.url(forResource: Self.demoSceneResource, withExtension: "ply"),
              let data = try? Data(contentsOf: imageURL),
              let image = UIImage(data: data) else {
            print("⚠️ [OpenReshot] bundled demo scene missing")
            return
        }

        print("🌄 [OpenReshot] loading bundled demo scene \(sceneURL.lastPathComponent)")
        didLoadDemoScene = true
        UserDefaults.standard.set(true, forKey: Self.didAutoloadDemoSceneKey)
        let taskID = UUID()
        activeTaskID = taskID
        reconstructionRequestID = taskID
        subjectMaskRequestID = taskID
        activeCacheItemID = nil
        pendingCachedReshot = nil
        enhanceTask?.cancel()
        enhanceTask = nil
        renderer?.clearCloud()
        currentSourceData = nil
        currentCloudPoints = nil
        currentPreviewMetadata = nil
        hasExportablePreview = false
        resetLensControls(focus: Self.demoSceneFocus)
        inputImage = image
        imageAspect = max(0.1, CGFloat(Self.demoSceneWidth) / CGFloat(Self.demoSceneHeight))
        hasCloud = false
        rendererReady = false
        sheenAmount = 0
        sheenTiltX = 0
        sheenTiltY = 0
        motionTilt = .zero
        previewMode = true
        loadingPreviewScene = true
        reconstructingScene = false
        reconstructingFrame = false
        processFailed = false
        processFailureKind = nil
        capturedFrame = nil
        resultImage = nil
        saveState = .idle
        subjectProtectionMask = nil
        status = ""
        demoScenePending = true
        startPendingDemoSceneIfPossible()
        prefetchModelForDemoIfNeeded()
    }

    @MainActor
    func beginImageLoadTask() -> UUID {
        let taskID = UUID()
        resetTaskState(taskID: taskID, clearImage: true)
        return taskID
    }

    @MainActor
    func cancelCurrentTaskAndClear() {
        resetTaskState(taskID: UUID(), clearImage: true)
    }

    @MainActor
    func isCurrentTask(_ taskID: UUID) -> Bool {
        activeTaskID == taskID
    }

    @MainActor
    func failTaskIfCurrent(_ taskID: UUID, kind: ProcessFailureKind = .imageLoad) {
        guard isCurrentTask(taskID) else { return }
        reconstructingScene = false
        reconstructingFrame = false
        processFailed = true
        processFailureKind = kind
        status = ""
    }

    @MainActor
    func reconstruct(_ image: UIImage, sourceData: Data? = nil, taskID: UUID? = nil) {
        if let taskID, !isCurrentTask(taskID) {
            print("↩️ [OpenReshot] ignored stale picked photo")
            return
        }
        print("🔧 [OpenReshot] reconstruct start: \(Int(image.size.width))x\(Int(image.size.height)) @\(image.scale)x")
        let displayImage = SharpModel.normalized(image)
        let requestID = taskID ?? UUID()
        let hasInstalledModel = modelStore.activeModelURL() != nil
        let selectedQuality = quality
        let sourceSignature = Self.sourceSignature(for: displayImage)
        if let sourceSignature,
           let cachedItem = ReshotCacheStore.item(matching: sourceSignature, quality: selectedQuality) {
            print("♻️ [OpenReshot] loading cached reconstruction \(cachedItem.id)")
            loadCachedReshot(cachedItem, sourceOverride: displayImage, sourceData: sourceData, taskID: requestID)
            return
        }
        activeTaskID = requestID
        reconstructionRequestID = requestID
        currentSourceData = sourceData
        activeCacheItemID = nil
        pendingCachedReshot = nil
        enhanceTask?.cancel()
        enhanceTask = nil
        renderer?.clearCloud()
        previewMode = false
        loadingPreviewScene = false
        demoScenePending = false
        currentCloudPoints = nil
        currentPreviewMetadata = nil
        hasExportablePreview = false
        inputImage = displayImage
        imageAspect = max(0.1, displayImage.size.width / max(displayImage.size.height, 1))
        hasCloud = false
        rendererReady = false
        sheenAmount = 0
        sheenTiltX = 0
        sheenTiltY = 0
        motionTilt = .zero
        reconstructingScene = hasInstalledModel
        reconstructingFrame = false
        processFailed = !hasInstalledModel
        processFailureKind = hasInstalledModel ? nil : .missingModel
        capturedFrame = nil
        resultImage = nil
        saveState = .idle
        subjectProtectionMask = nil
        status = ""
        subjectMaskRequestID = requestID
        updateSubjectProtectionMask(for: displayImage, requestID: requestID)
        guard hasInstalledModel else {
            print("⚠️ [OpenReshot] reconstruction model missing; prompting download")
            return
        }
        modelQueue.async { [weak self] in
            guard let self else { return }
            do {
                let model = try self.loadCachedModel()
                print("✅ [OpenReshot] running inference (\(selectedQuality.title))…")
                let t0 = Date()
                let out = try model.reconstruct(displayImage, sourceData: sourceData)
                print("✅ [OpenReshot] inference done in \(Int(-t0.timeIntervalSinceNow))s, \(out.count) gaussians")
                let (g, focus) = GaussianCloud.build(from: out, quality: selectedQuality)
                print("✅ [OpenReshot] cloud built, focus=\(focus)")
                DispatchQueue.main.async {
                    guard self.activeTaskID == requestID, self.reconstructionRequestID == requestID else {
                        print("↩️ [OpenReshot] ignored stale reconstruction result")
                        return
                    }
                    if self.renderer == nil { print("❌ [OpenReshot] renderer is nil (Metal init failed)") }
                    self.renderer?.setCloud(g, focus: focus,
                                            fpx: out.fpx, width: out.width, height: out.height)
                    self.resetLensControls(focus: focus)
                    self.hasCloud = true
                    self.currentCloudPoints = g
                    self.currentPreviewMetadata = PreviewExportMetadata(
                        formatVersion: 1,
                        splatFormat: "ply-sh0",
                        splatCount: g.count,
                        focus: focus,
                        focalPixels: out.fpx,
                        width: out.width,
                        height: out.height,
                        sourceImageName: "OpenReshotPreview.png",
                        splatFileName: "OpenReshotPreview.ply"
                    )
                    self.hasExportablePreview = true
                    self.status = ""
                    if let sourceSignature {
                        self.persistReshotCache(
                            points: g,
                            sourceImage: displayImage,
                            sourceSignature: sourceSignature,
                            quality: selectedQuality,
                            focus: focus,
                            focalPixels: out.fpx,
                            width: out.width,
                            height: out.height,
                            taskID: requestID
                        )
                    }
                }
            } catch {
                print("❌ [OpenReshot] reconstruct error: \(error)")
                DispatchQueue.main.async {
                    guard self.activeTaskID == requestID, self.reconstructionRequestID == requestID else { return }
                    self.reconstructingScene = false
                    self.processFailed = true
                    self.processFailureKind = .reconstruction
                    self.status = ""
                }
            }
        }
    }

    private func updateSubjectProtectionMask(for image: UIImage, requestID: UUID) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let mask = Self.makeSubjectProtectionMask(from: image)
            DispatchQueue.main.async { [weak self] in
                guard let self, self.activeTaskID == requestID, self.subjectMaskRequestID == requestID else { return }
                self.subjectProtectionMask = mask
                print(mask == nil
                      ? "⚠️ [OpenReshot] no foreground subject mask"
                      : "✅ [OpenReshot] foreground subject mask ready")
            }
        }
    }

    private func loadCachedModel() throws -> SharpModel {
        if let cachedModel {
            print("♻️ [OpenReshot] reusing cached reconstruction model")
            return cachedModel
        }
        print("⏳ [OpenReshot] loading reconstruction model…")
        let model = try SharpModel(modelURL: modelStore.activeModelURL())
        cachedModel = model
        return model
    }

    func invalidateModelCache() {
        modelQueue.async { [weak self] in
            self?.cachedModel = nil
            print("🧹 [OpenReshot] reconstruction model cache invalidated")
        }
    }

    @MainActor
    func modelInstallationDidFinish() {
        invalidateModelCache()
        guard !previewMode else { return }
        guard processFailed,
              processFailureKind == .missingModel || processFailureKind == .reconstruction,
              let inputImage else { return }
        reconstruct(inputImage, sourceData: currentSourceData)
    }

    @MainActor
    func retryCurrentReconstruction() {
        guard !previewMode else { return }
        guard let inputImage else { return }
        reconstruct(inputImage, sourceData: currentSourceData)
    }

    @MainActor
    func refreshGallery() {
        galleryItems = ReshotCacheStore.galleryItems()
    }

    @MainActor
    func openCachedReshot(_ item: ReshotCacheItem) {
        loadCachedReshot(item)
    }

    @MainActor
    func deleteCachedReshot(_ item: ReshotCacheItem) {
        do {
            try ReshotCacheStore.delete(item)
            galleryItems.removeAll { $0.id == item.id }
            if activeCacheItemID == item.id {
                resetSession()
            }
        } catch {
            print("❌ [OpenReshot] delete cached reconstruction error: \(error)")
        }
    }

    @MainActor
    func exportCurrentPreviewPackage() async throws -> [URL] {
        guard let points = currentCloudPoints,
              let metadata = currentPreviewMetadata,
              let imageData = inputImage?.pngData() else {
            throw err("当前没有可导出的 3D 预览")
        }

        let urls = try await Self.writePreviewExportPackage(
            points: points,
            imageData: imageData,
            metadata: metadata
        )
        return urls
    }

    @MainActor
    func updateSheen(for tilt: SIMD2<Float>) {
        motionTilt = tilt
        refreshSheenState()
    }

    @MainActor
    func setLensFocusDepth(_ depth: Float) {
        lensFocusDepth = min(max(depth, lensFocusMin), lensFocusMax)
        lensDolly = min(max(lensDolly, lensDollyRange.lowerBound), lensDollyRange.upperBound)
        if lensFNumber >= 15.9 {
            lensFNumber = 4
        }
        renderer?.setLens(focusDepth: lensFocusDepth, fNumber: lensFNumber, dolly: lensDolly)
        refreshSheenState()
    }

    @MainActor
    func setLensFNumber(_ value: Float) {
        lensFNumber = min(max(value, 1.4), 16)
        renderer?.setLens(fNumber: lensFNumber)
    }

    @MainActor
    func setLensDolly(_ value: Float) {
        lensDolly = min(max(value, lensDollyRange.lowerBound), lensDollyRange.upperBound)
        renderer?.setLens(dolly: lensDolly)
        refreshSheenState()
    }

    @MainActor
    private func refreshSheenState() {
        let tiltAmount = CGFloat(simd_length(motionTilt))
        let dollyExtent = max(abs(lensDollyRange.lowerBound), abs(lensDollyRange.upperBound), 0.001)
        let dolly = CGFloat(lensDolly / dollyExtent)
        let dollyAmount = min(1, abs(dolly))
        let dollyDirection = dolly == 0 ? CGFloat(0) : (dolly > 0 ? CGFloat(1) : CGFloat(-1))

        sheenAmount = min(1, max(tiltAmount, dollyAmount * 0.95))
        sheenTiltX = CGFloat(motionTilt.x) + dollyDirection * dollyAmount * 0.55
        sheenTiltY = CGFloat(motionTilt.y) + dollyAmount * 0.18
    }

    @MainActor
    func setViewAngleMode(_ mode: ReshotViewAngleMode) {
        viewAngleMode = mode
        renderer?.setMotionRangeScale(mode.motionScale)
    }

    @MainActor
    func resetLensControlsToDefault() {
        let focus = min(max(lensDefaultFocusDepth, lensFocusMin), lensFocusMax)
        resetLensControls(focus: focus)
    }

    @MainActor
    func reconstructCurrentFrame() {
        guard !previewMode else { return }
        guard rendererReady, !reconstructingFrame else { return }
        guard !geminiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            processFailed = true
            processFailureKind = .missingAPIKey
            reconstructingFrame = false
            status = ""
            print("⚠️ [OpenReshot] Gemini API key missing; prompting input")
            return
        }
        let taskID = activeTaskID
        let startedAt = Date()
        enhanceTask?.cancel()
        resultImage = nil
        saveState = .idle
        reconstructingFrame = true
        processFailed = false
        processFailureKind = nil
        status = ""
        let rendered = renderer?.snapshotImage()
        guard let frame = Self.composeEnhanceFrame(rendered: rendered, source: inputImage),
              frame.size.width > 0,
              frame.size.height > 0 else {
            processFailed = true
            processFailureKind = .enhancement
            reconstructingFrame = false
            return
        }
        capturedFrame = frame
        print("⏱️ [OpenReshot] enhance capture+compose \(Self.ms(since: startedAt))ms, frame \(Self.describe(frame))")
        let key = geminiKey
        enhanceTask = Task { [weak self, frame, key, taskID] in
            guard let self else { return }
            do {
                let payload = try await Self.makeGeminiPayload(from: frame)
                try Task.checkCancellation()
                let result = try await Self.requestGeminiEnhance(imagePNG: payload, key: key)
                try Task.checkCancellation()
                await MainActor.run {
                    guard self.activeTaskID == taskID else { return }
                    self.resultImage = result
                    self.saveState = .idle
                    self.processFailed = false
                    self.processFailureKind = nil
                    self.reconstructingFrame = false
                    self.enhanceTask = nil
                }
            } catch is CancellationError {
                await MainActor.run {
                    guard self.activeTaskID == taskID else { return }
                    self.reconstructingFrame = false
                    self.enhanceTask = nil
                }
            } catch {
                print("❌ [OpenReshot] enhance error: \(error)")
                await MainActor.run {
                    guard self.activeTaskID == taskID else { return }
                    self.processFailed = true
                    self.processFailureKind = .enhancement
                    self.status = ""
                    self.reconstructingFrame = false
                    self.enhanceTask = nil
                }
            }
        }
    }

    @MainActor
    func closeResult() {
        resultImage = nil
        saveState = .idle
        motionExportState = .idle
        motionExportFormat = nil
        motionExportProgress = 0
        motionExportFailureMessage = nil
    }

    @MainActor
    func resetSession() {
        cancelCurrentTaskAndClear()
    }

    @MainActor
    private func resetTaskState(taskID: UUID, clearImage: Bool) {
        activeTaskID = taskID
        reconstructionRequestID = taskID
        subjectMaskRequestID = taskID
        enhanceTask?.cancel()
        enhanceTask = nil
        motionExportTask?.cancel()
        motionExportTask = nil
        renderer?.clearCloud()
        demoScenePending = false
        pendingCachedReshot = nil
        activeCacheItemID = nil
        previewMode = false
        loadingPreviewScene = false
        currentCloudPoints = nil
        currentPreviewMetadata = nil
        hasExportablePreview = false
        resetLensControls(focus: 1)
        status = ""
        hasCloud = false
        rendererReady = false
        if clearImage {
            inputImage = nil
            imageAspect = 1
            currentSourceData = nil
        }
        sheenAmount = 0
        sheenTiltX = 0
        sheenTiltY = 0
        motionTilt = .zero
        reconstructingScene = false
        reconstructingFrame = false
        processFailed = false
        processFailureKind = nil
        capturedFrame = nil
        resultImage = nil
        saveState = .idle
        motionExportState = .idle
        motionExportFormat = nil
        motionExportProgress = 0
        motionExportFailureMessage = nil
        subjectProtectionMask = nil
    }

    @MainActor
    private func startPendingDemoSceneIfPossible() {
        guard demoScenePending, let renderer else { return }
        guard let sceneURL = Bundle.main.url(forResource: Self.demoSceneResource, withExtension: "ply") else {
            demoScenePending = false
            previewMode = false
            loadingPreviewScene = false
            reconstructingScene = false
            processFailed = true
            processFailureKind = .imageLoad
            print("❌ [OpenReshot] missing bundled demo PLY")
            return
        }

        loadingPreviewScene = true
        let didStart = renderer.setCloud(from: sceneURL,
                                         focus: Self.demoSceneFocus,
                                         fpx: Self.demoSceneFocalPixels,
                                         width: Self.demoSceneWidth,
                                         height: Self.demoSceneHeight)
        guard didStart else { return }
        demoScenePending = false
        resetLensControls(focus: Self.demoSceneFocus)
        hasCloud = true
    }

    @MainActor
    private func loadCachedReshot(
        _ item: ReshotCacheItem,
        sourceOverride: UIImage? = nil,
        sourceData: Data? = nil,
        taskID: UUID = UUID()
    ) {
        guard let sourceURL = ReshotCacheStore.sourceURL(for: item),
              let sourceImage = sourceOverride ?? UIImage(contentsOfFile: sourceURL.path) else {
            processFailed = true
            processFailureKind = .imageLoad
            print("❌ [OpenReshot] cached source missing for \(item.id)")
            refreshGallery()
            return
        }

        activeTaskID = taskID
        reconstructionRequestID = taskID
        subjectMaskRequestID = taskID
        activeCacheItemID = item.id
        pendingCachedReshot = item
        currentSourceData = sourceData ?? (try? Data(contentsOf: sourceURL))
        enhanceTask?.cancel()
        enhanceTask = nil
        motionExportTask?.cancel()
        motionExportTask = nil
        renderer?.clearCloud()
        demoScenePending = false
        previewMode = false
        loadingPreviewScene = true
        reconstructingScene = false
        reconstructingFrame = false
        processFailed = false
        processFailureKind = nil
        currentCloudPoints = nil
        currentPreviewMetadata = nil
        hasExportablePreview = false
        inputImage = sourceImage
        imageAspect = max(0.1, CGFloat(item.width) / CGFloat(max(item.height, 1)))
        hasCloud = false
        rendererReady = false
        sheenAmount = 0
        sheenTiltX = 0
        sheenTiltY = 0
        motionTilt = .zero
        capturedFrame = nil
        resultImage = nil
        saveState = .idle
        motionExportState = .idle
        motionExportFormat = nil
        motionExportProgress = 0
        motionExportFailureMessage = nil
        subjectProtectionMask = nil
        status = ""
        resetLensControls(focus: item.focus)
        updateSubjectProtectionMask(for: sourceImage, requestID: taskID)
        startPendingCachedReshotIfPossible()
    }

    @MainActor
    private func startPendingCachedReshotIfPossible() {
        guard let item = pendingCachedReshot, let renderer else { return }
        guard let splatURL = ReshotCacheStore.splatURL(for: item) else {
            pendingCachedReshot = nil
            loadingPreviewScene = false
            processFailed = true
            processFailureKind = .imageLoad
            print("❌ [OpenReshot] cached PLY missing for \(item.id)")
            refreshGallery()
            return
        }

        loadingPreviewScene = true
        let didStart = renderer.setCloud(from: splatURL,
                                         focus: item.focus,
                                         fpx: item.focalPixels,
                                         width: item.width,
                                         height: item.height)
        guard didStart else { return }
        pendingCachedReshot = nil
        resetLensControls(focus: item.focus)
        hasCloud = true
    }

    @MainActor
    private func resetLensControls(focus: Float) {
        let safeFocus = max(0.08, focus)
        lensDefaultFocusDepth = safeFocus
        lensFocusMin = max(0.05, safeFocus * 0.35)
        lensFocusMax = max(lensFocusMin + 0.25, safeFocus * 2.6)
        lensFocusDepth = min(max(safeFocus, lensFocusMin), lensFocusMax)
        // Default preview must stay sharp; f/16 disables the DOF shader blur.
        lensFNumber = 16
        lensDolly = 0
        viewAngleMode = .wide
        renderer?.setMotionRangeScale(viewAngleMode.motionScale)
        renderer?.setLens(focusDepth: lensFocusDepth, fNumber: lensFNumber, dolly: lensDolly)
    }

    private var maxForwardLensDolly: Float {
        min(0.8, max(0, lensFocusDepth - 0.08))
    }

    var lensDollyRange: ClosedRange<Float> {
        let backward = -min(0.8, max(0.08, lensFocusDepth * 0.40))
        return backward...maxForwardLensDolly
    }

    @MainActor
    private func prefetchModelForDemoIfNeeded() {
        guard !didStartDemoModelPrefetch,
              modelStore.activeModelURL() == nil,
              !modelStore.isDownloading else { return }
        didStartDemoModelPrefetch = true
        modelStore.installModel()
    }

    private static func writePreviewExportPackage(
        points: [SplatPoint],
        imageData: Data,
        metadata: PreviewExportMetadata
    ) async throws -> [URL] {
        try await Task.detached(priority: .userInitiated) {
            let fileManager = FileManager.default
            let directory = fileManager.temporaryDirectory
                .appendingPathComponent("OpenReshotPreview-\(UUID().uuidString)", isDirectory: true)
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)

            let plyURL = directory.appendingPathComponent(metadata.splatFileName)
            let imageURL = directory.appendingPathComponent(metadata.sourceImageName)
            let jsonURL = directory.appendingPathComponent("OpenReshotPreview.json")

            let writer = try SplatPLYSceneWriter(toFileAtPath: plyURL.path)
            try await writer.start(sphericalHarmonicDegree: 0, binary: true, pointCount: points.count)
            try await writer.write(points)
            try await writer.close()

            try imageData.write(to: imageURL, options: [.atomic])
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(metadata).write(to: jsonURL, options: [.atomic])
            return [plyURL, imageURL, jsonURL]
        }.value
    }

    private func persistReshotCache(
        points: [SplatPoint],
        sourceImage: UIImage,
        sourceSignature: String,
        quality: RenderQuality,
        focus: Float,
        focalPixels: Float,
        width: Int,
        height: Int,
        taskID: UUID
    ) {
        Task { [weak self, points, sourceImage, sourceSignature, quality] in
            do {
                let item = try await ReshotCacheStore.save(
                    points: points,
                    sourceImage: sourceImage,
                    sourceSignature: sourceSignature,
                    quality: quality,
                    focus: focus,
                    focalPixels: focalPixels,
                    width: width,
                    height: height
                )
                await MainActor.run {
                    guard self?.activeTaskID == taskID else { return }
                    self?.activeCacheItemID = item.id
                    self?.galleryItems.removeAll { $0.id == item.id }
                    self?.galleryItems.insert(item, at: 0)
                    print("💾 [OpenReshot] cached reconstruction \(item.id)")
                }
            } catch {
                print("⚠️ [OpenReshot] cache write failed: \(error)")
            }
        }
    }

    private static func sourceSignature(for image: UIImage) -> String? {
        guard let data = image.pngData() else { return nil }
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    @MainActor
    func saveResultImage() {
        guard let resultImage, saveState != .saving else { return }
        let taskID = activeTaskID
        saveState = .saving
        Task { [weak self, taskID] in
            do {
                try await Self.saveToPhotoLibrary(resultImage)
                await MainActor.run {
                    guard self?.activeTaskID == taskID else { return }
                    self?.saveState = .saved
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { [weak self] in
                        guard self?.activeTaskID == taskID, self?.saveState == .saved else { return }
                        self?.saveState = .idle
                    }
                }
            } catch {
                await MainActor.run {
                    guard self?.activeTaskID == taskID else { return }
                    print("❌ [OpenReshot] save result error: \(error)")
                    self?.saveState = .failed
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.6) { [weak self] in
                        guard self?.activeTaskID == taskID, self?.saveState == .failed else { return }
                        self?.saveState = .idle
                    }
                }
            }
        }
    }

    @MainActor
    func exportMotion(_ format: MotionExportFormat) {
        guard motionExportState != .rendering else { return }
        guard rendererReady, let renderer else { return }

        let taskID = activeTaskID
        motionExportFormat = format
        motionExportState = .rendering
        motionExportProgress = 0
        motionExportFailureMessage = nil

        motionExportTask?.cancel()
        motionExportTask = Task { @MainActor [weak self, renderer, taskID, format] in
            guard let self else { return }
            do {
                let package = try await self.renderMotionExportPackage(format: format, renderer: renderer)
                if format.isLivePhotoPackage {
                    try await Self.validateLivePhotoPackage(package)
                }
                try await Self.saveMotionExportPackage(package)
                guard self.activeTaskID == taskID else { return }
                self.motionExportProgress = 1
                self.motionExportState = .saved
                self.motionExportFailureMessage = nil
                self.motionExportTask = nil
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.3) { [weak self] in
                    guard self?.activeTaskID == taskID, self?.motionExportState == .saved else { return }
                    self?.motionExportState = .idle
                    self?.motionExportFormat = nil
                    self?.motionExportProgress = 0
                    self?.motionExportFailureMessage = nil
                }
            } catch is CancellationError {
                guard self.activeTaskID == taskID else { return }
                self.motionExportState = .idle
                self.motionExportFormat = nil
                self.motionExportProgress = 0
                self.motionExportFailureMessage = nil
                self.motionExportTask = nil
            } catch {
                guard self.activeTaskID == taskID else { return }
                print("❌ [OpenReshot] motion export error: \(error)")
                self.motionExportFailureMessage = Self.motionExportFailureMessage(format: format, error: error)
                self.motionExportState = .failed
                self.motionExportTask = nil
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.8) { [weak self] in
                    guard self?.activeTaskID == taskID, self?.motionExportState == .failed else { return }
                    self?.motionExportState = .idle
                    self?.motionExportFormat = nil
                    self?.motionExportProgress = 0
                    self?.motionExportFailureMessage = nil
                }
            }
        }
    }

    private static func motionExportFailureMessage(format: MotionExportFormat, error: Error) -> String {
        if format.isLivePhotoPackage {
            return "Live Photo 配对失败"
        }
        if error is CancellationError {
            return "导出已取消"
        }
        return "\(format.title) 导出失败"
    }

    @MainActor
    private func renderMotionExportPackage(format: MotionExportFormat, renderer: ReshootRenderer) async throws -> MotionExportPackage {
        let plan = motionFramePlan(for: format)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("OpenReshotMotion-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        switch format {
        case .gif:
            let gifURL = directory.appendingPathComponent("OpenReshot.gif")
            let destination = try MotionGIFWriter.makeDestination(url: gifURL, frameCount: plan.frameCount)
            for frameIndex in 0..<plan.frameCount {
                try Task.checkCancellation()
                let frame = try motionExportFrame(renderer: renderer, plan: plan, frameIndex: frameIndex)
                try MotionGIFWriter.addFrame(frame, to: destination, delay: plan.frameDuration)
                motionExportProgress = Double(frameIndex + 1) / Double(plan.frameCount) * 0.92
                if frameIndex % 4 == 0 {
                    await Task.yield()
                }
            }
            try MotionGIFWriter.finalize(destination)
            return MotionExportPackage(format: .gif, videoURL: nil, gifURL: gifURL, livePhotoImageURL: nil, livePhotoVideoURL: nil)

        case .mp4, .livePhoto, .momentsLivePhoto:
            let livePhotoIdentifier = format.isLivePhotoPackage ? Self.makeLivePhotoAssetIdentifier() : nil
            let stillFrameIndex = Self.livePhotoStillFrameIndex(for: format)
            let videoURL = directory.appendingPathComponent(format.isLivePhotoPackage ? "OpenReshot.mov" : "OpenReshot.mp4")
            let writer = try MotionVideoWriter(
                url: videoURL,
                fileType: format.isLivePhotoPackage ? .mov : .mp4,
                videoCodec: format == .momentsLivePhoto ? .hevc : .h264,
                size: plan.size,
                fps: plan.fps,
                contentIdentifier: livePhotoIdentifier,
                stillImageFrameIndex: stillFrameIndex
            )
            for frameIndex in 0..<plan.frameCount {
                try Task.checkCancellation()
                let frame = try motionExportFrame(renderer: renderer, plan: plan, frameIndex: frameIndex)
                try await writer.append(frame, frameIndex: frameIndex)
                motionExportProgress = Double(frameIndex + 1) / Double(plan.frameCount) * 0.92
                if frameIndex % 4 == 0 {
                    await Task.yield()
                }
            }
            try await writer.finish()

            guard format.isLivePhotoPackage else {
                return MotionExportPackage(format: .mp4, videoURL: videoURL, gifURL: nil, livePhotoImageURL: nil, livePhotoVideoURL: nil)
            }
            guard let livePhotoIdentifier, let stillFrameIndex else {
                throw err("Live Photo identifier missing")
            }
            let stillFormat = Self.livePhotoStillFormat(for: format)
            let stillURL = directory.appendingPathComponent("OpenReshot.\(stillFormat.preferredFilenameExtension ?? "jpg")")
            let stillFrame = try motionExportFrame(renderer: renderer, plan: plan, frameIndex: stillFrameIndex)
            try Self.writeLivePhotoStillImage(stillFrame, assetIdentifier: livePhotoIdentifier, type: stillFormat, to: stillURL)
            return MotionExportPackage(format: format, videoURL: nil, gifURL: nil, livePhotoImageURL: stillURL, livePhotoVideoURL: videoURL)
        }
    }

    @MainActor
    private func motionExportFrame(renderer: ReshootRenderer, plan: MotionFramePlan, frameIndex: Int) throws -> UIImage {
        let frame = try renderer.renderMotionFrame(tilt: plan.tilt(at: frameIndex), size: plan.size)
        return Self.watermarkedExportFrame(frame)
    }

    private func motionFramePlan(for format: MotionExportFormat) -> MotionFramePlan {
        if format == .momentsLivePhoto {
            return MotionFramePlan(
                size: Self.motionExportSize(for: imageAspect),
                fps: Self.momentsLivePhotoFPS,
                frameCount: Self.momentsLivePhotoFrameCount,
                tiltRange: Self.momentsLivePhotoTiltRange,
                path: .centeredOrbit(clockwise: true, stillFrameIndex: Self.momentsLivePhotoStillFrameIndex)
            )
        }
        return MotionFramePlan(
            size: Self.motionExportSize(for: imageAspect),
            fps: Self.motionExportFPS,
            frameCount: Self.motionExportFrameCount,
            tiltRange: Self.motionExportTiltRange,
            path: .centerOrbit(clockwise: true)
        )
    }

    private static func livePhotoStillFrameIndex(for format: MotionExportFormat) -> Int? {
        switch format {
        case .livePhoto: return livePhotoStillFrameIndex
        case .momentsLivePhoto: return momentsLivePhotoStillFrameIndex
        case .mp4, .gif: return nil
        }
    }

    private static func motionExportSize(for aspect: CGFloat) -> CGSize {
        let safeAspect = max(aspect, 0.1)
        let longSide = motionExportMaxLongSide
        let rawWidth: CGFloat
        let rawHeight: CGFloat
        if safeAspect >= 1 {
            rawWidth = longSide
            rawHeight = longSide / safeAspect
        } else {
            rawHeight = longSide
            rawWidth = longSide * safeAspect
        }
        return CGSize(width: evenDimension(rawWidth), height: evenDimension(rawHeight))
    }

    private static func evenDimension(_ value: CGFloat) -> CGFloat {
        max(2, CGFloat(Int(value.rounded(.toNearestOrAwayFromZero)) / 2 * 2))
    }

    private static func watermarkedExportFrame(_ image: UIImage) -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let size = image.size
        return UIGraphicsImageRenderer(size: size, format: format).image { _ in
            image.draw(in: CGRect(origin: .zero, size: size))

            let label = "Reshot"
            let fontSize = max(11, min(size.width, size.height) * 0.026)
            let attributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: fontSize, weight: .semibold),
                .foregroundColor: UIColor.white.withAlphaComponent(0.86)
            ]
            let labelSize = (label as NSString).size(withAttributes: attributes)
            let horizontalPadding = fontSize * 0.72
            let verticalPadding = fontSize * 0.40
            let badgeSize = CGSize(
                width: labelSize.width + horizontalPadding * 2,
                height: labelSize.height + verticalPadding * 2
            )
            let inset = max(12, min(size.width, size.height) * 0.025)
            let badgeRect = CGRect(
                x: size.width - badgeSize.width - inset,
                y: size.height - badgeSize.height - inset,
                width: badgeSize.width,
                height: badgeSize.height
            )
            UIColor.black.withAlphaComponent(0.32).setFill()
            UIBezierPath(roundedRect: badgeRect, cornerRadius: badgeRect.height / 2).fill()
            (label as NSString).draw(
                in: CGRect(
                    x: badgeRect.minX + horizontalPadding,
                    y: badgeRect.minY + verticalPadding,
                    width: labelSize.width,
                    height: labelSize.height
                ),
                withAttributes: attributes
            )
        }
    }

    private static func makeLivePhotoAssetIdentifier() -> String {
        UUID().uuidString
    }

    private static func livePhotoStillFormat(for format: MotionExportFormat) -> UTType {
        format == .momentsLivePhoto ? .heic : .jpeg
    }

    private static func writeLivePhotoStillImage(_ image: UIImage, assetIdentifier: String, type: UTType, to url: URL) throws {
        try? FileManager.default.removeItem(at: url)
        guard assetIdentifier.count == 36 else {
            throw err("Live Photo asset identifier must be a 36-character UUID")
        }
        guard let cgImage = image.cgImage,
              let destination = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil) else {
            throw err("Live Photo still image encode failed")
        }
        let metadata: [CFString: Any] = [
            kCGImageDestinationLossyCompressionQuality: 0.96,
            kCGImagePropertyMakerAppleDictionary: [
                "17": assetIdentifier
            ]
        ]
        CGImageDestinationAddImage(destination, cgImage, metadata as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            throw err("Live Photo still image finalize failed")
        }
    }

    private static func saveMotionExportPackage(_ package: MotionExportPackage) async throws {
        switch package.format {
        case .mp4:
            guard let url = package.videoURL else { throw err("MP4 export missing") }
            try await saveVideoToPhotoLibrary(url)
        case .gif:
            guard let url = package.gifURL else { throw err("GIF export missing") }
            try await saveImageFileToPhotoLibrary(url)
        case .livePhoto:
            guard let imageURL = package.livePhotoImageURL,
                  let videoURL = package.livePhotoVideoURL else {
                throw err("Live Photo export missing")
            }
            try await saveLivePhotoToPhotoLibrary(imageURL: imageURL, imageTypeIdentifier: livePhotoStillFormat(for: package.format).identifier, videoURL: videoURL)
        case .momentsLivePhoto:
            guard let imageURL = package.livePhotoImageURL,
                  let videoURL = package.livePhotoVideoURL else {
                throw err("Live Photo export missing")
            }
            try await saveLivePhotoToPhotoLibrary(imageURL: imageURL, imageTypeIdentifier: livePhotoStillFormat(for: package.format).identifier, videoURL: videoURL)
        }
    }

    private static func validateLivePhotoPackage(_ package: MotionExportPackage) async throws {
        guard let imageURL = package.livePhotoImageURL,
              let videoURL = package.livePhotoVideoURL else {
            throw err("Live Photo export missing")
        }
        _ = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<PHLivePhoto, Error>) in
            var completed = false
            var requestID = PHLivePhotoRequestIDInvalid
            requestID = PHLivePhoto.request(
                withResourceFileURLs: [imageURL, videoURL],
                placeholderImage: nil,
                targetSize: CGSize(width: 320, height: 320),
                contentMode: .aspectFit
            ) { livePhoto, info in
                if completed { return }
                let cancelled = (info[PHLivePhotoInfoCancelledKey] as? NSNumber)?.boolValue ?? false
                if cancelled {
                    completed = true
                    continuation.resume(throwing: err("Live Photo validation cancelled"))
                    return
                }
                if let error = info[PHLivePhotoInfoErrorKey] as? Error {
                    completed = true
                    continuation.resume(throwing: error)
                    return
                }
                let degraded = (info[PHLivePhotoInfoIsDegradedKey] as? NSNumber)?.boolValue ?? false
                if let livePhoto, !degraded {
                    completed = true
                    continuation.resume(returning: livePhoto)
                }
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 8) {
                guard !completed else { return }
                completed = true
                if requestID != PHLivePhotoRequestIDInvalid {
                    PHLivePhoto.cancelRequest(withRequestID: requestID)
                }
                continuation.resume(throwing: err("Live Photo validation timed out"))
            }
        }
    }

    private static func ensurePhotoLibraryAddAuthorization() async throws {
        let currentStatus = PHPhotoLibrary.authorizationStatus(for: .addOnly)
        let status: PHAuthorizationStatus
        if currentStatus == .notDetermined {
            status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        } else {
            status = currentStatus
        }
        guard status == .authorized || status == .limited else {
            throw err("Photo library add permission denied")
        }
    }

    private static func saveVideoToPhotoLibrary(_ url: URL) async throws {
        try await ensurePhotoLibraryAddAuthorization()
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
            } completionHandler: { success, error in
                if let error {
                    continuation.resume(throwing: error)
                } else if success {
                    continuation.resume()
                } else {
                    continuation.resume(throwing: err("Photo library video save failed"))
                }
            }
        }
    }

    private static func saveImageFileToPhotoLibrary(_ url: URL) async throws {
        try await ensurePhotoLibraryAddAuthorization()
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.creationRequestForAssetFromImage(atFileURL: url)
            } completionHandler: { success, error in
                if let error {
                    continuation.resume(throwing: error)
                } else if success {
                    continuation.resume()
                } else {
                    continuation.resume(throwing: err("Photo library image save failed"))
                }
            }
        }
    }

    private static func saveLivePhotoToPhotoLibrary(imageURL: URL, imageTypeIdentifier: String, videoURL: URL) async throws {
        try await ensurePhotoLibraryAddAuthorization()
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            PHPhotoLibrary.shared().performChanges {
                let request = PHAssetCreationRequest.forAsset()
                let photoOptions = PHAssetResourceCreationOptions()
                photoOptions.uniformTypeIdentifier = imageTypeIdentifier
                let videoOptions = PHAssetResourceCreationOptions()
                videoOptions.uniformTypeIdentifier = UTType.quickTimeMovie.identifier
                request.addResource(with: .photo, fileURL: imageURL, options: photoOptions)
                request.addResource(with: .pairedVideo, fileURL: videoURL, options: videoOptions)
            } completionHandler: { success, error in
                if let error {
                    continuation.resume(throwing: error)
                } else if success {
                    continuation.resume()
                } else {
                    continuation.resume(throwing: err("Photo library Live Photo save failed"))
                }
            }
        }
    }

    func saveEnhanceSettings(key: String) {
        geminiKey = key.trimmingCharacters(in: .whitespacesAndNewlines)
        UserDefaults.standard.set(geminiKey, forKey: "OpenReshot.geminiKey")
        if !geminiKey.isEmpty, processFailureKind == .missingAPIKey {
            processFailed = false
            processFailureKind = nil
        }
    }

    private static func requestGeminiEnhance(imagePNG: Data, key: String) async throws -> UIImage {
        let requestStartedAt = Date()
        let trimmedKey = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedKey.isEmpty else { throw err("Missing Gemini API key") }
        guard var components = URLComponents(string: "https://generativelanguage.googleapis.com/v1beta/models/\(geminiModel):generateContent") else {
            throw err("Invalid Gemini endpoint")
        }
        components.queryItems = [URLQueryItem(name: "key", value: trimmedKey)]
        guard let url = components.url else { throw err("Invalid Gemini endpoint") }
        var request = URLRequest(url: url, timeoutInterval: 180)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = [
            "contents": [
                [
                    "role": "user",
                    "parts": [
                        ["text": enhancePrompt],
                        ["inlineData": [
                            "mimeType": "image/png",
                            "data": imagePNG.base64EncodedString()
                        ]]
                    ]
                ]
            ],
            "generationConfig": [
                "thinkingConfig": [
                    "thinkingLevel": "MINIMAL"
                ],
                "imageConfig": [
                    "imageSize": "1K"
                ],
                "responseModalities": ["IMAGE", "TEXT"]
            ]
        ]
        let requestData = try JSONSerialization.data(withJSONObject: body)
        print("⏱️ [OpenReshot] Gemini upload JSON \(requestData.count / 1024)KB")
        let (data, response) = try await URLSession.shared.upload(for: request, from: requestData)
        print("⏱️ [OpenReshot] Gemini response \(ms(since: requestStartedAt))ms, \(data.count / 1024)KB")
        guard let http = response as? HTTPURLResponse else {
            throw err("无效响应")
        }
        guard (200..<300).contains(http.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? "HTTP \(http.statusCode)"
            throw err(String(body.prefix(160)))
        }
        guard let image = imageFromGeminiResponse(data) else {
            let detail = geminiResponseDetail(data)
            throw err(detail.isEmpty ? "Gemini 没有返回图片" : "Gemini 没有返回图片: \(detail)")
        }
        return image
    }

    private static func makeGeminiPayload(from frame: UIImage) async throws -> Data {
        try await Task.detached(priority: .userInitiated) {
            let encodeStartedAt = Date()
            let input = geminiInputImage(from: frame)
            guard let payload = input.pngData() else {
                throw err("Gemini input encode failed")
            }
            print("⏱️ [OpenReshot] enhance resize+png \(ms(since: encodeStartedAt))ms, input \(describe(input)), payload \(payload.count / 1024)KB")
            return payload
        }.value
    }

    private static func geminiInputImage(from image: UIImage) -> UIImage {
        let pixelWidth = CGFloat(image.cgImage?.width ?? Int(image.size.width * image.scale))
        let pixelHeight = CGFloat(image.cgImage?.height ?? Int(image.size.height * image.scale))
        let maxSide = max(pixelWidth, pixelHeight)
        let ratio = min(1, geminiInputMaxSide / max(maxSide, 1))
        let targetSize = CGSize(width: pixelWidth * ratio, height: pixelHeight * ratio)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        return UIGraphicsImageRenderer(size: targetSize, format: format).image { _ in
            image.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }

    private static func describe(_ image: UIImage) -> String {
        let pixelWidth = image.cgImage?.width ?? Int(image.size.width * image.scale)
        let pixelHeight = image.cgImage?.height ?? Int(image.size.height * image.scale)
        return "\(Int(image.size.width))x\(Int(image.size.height))@\(String(format: "%.1f", image.scale)) / \(pixelWidth)x\(pixelHeight)px"
    }

    private static func ms(since date: Date) -> Int {
        Int(-date.timeIntervalSinceNow * 1000)
    }

    private static func saveToPhotoLibrary(_ image: UIImage) async throws {
        let currentStatus = PHPhotoLibrary.authorizationStatus(for: .addOnly)
        let status: PHAuthorizationStatus
        if currentStatus == .notDetermined {
            status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        } else {
            status = currentStatus
        }
        guard status == .authorized || status == .limited else {
            throw err("Photo library add permission denied")
        }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.creationRequestForAsset(from: image)
            } completionHandler: { success, error in
                if let error {
                    continuation.resume(throwing: error)
                } else if success {
                    continuation.resume()
                } else {
                    continuation.resume(throwing: err("Photo library save failed"))
                }
            }
        }
    }

    private static func imageFromGeminiResponse(_ data: Data) -> UIImage? {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let image = imageFromJSONValue(root) else {
            return nil
        }
        return image
    }

    private static func imageFromJSONValue(_ value: Any?) -> UIImage? {
        if let string = value as? String {
            return imageFromEncodedString(string)
        }
        if let dictionary = value as? [String: Any] {
            for key in ["image", "imageBase64", "base64", "data", "result", "output"] {
                if let image = imageFromJSONValue(dictionary[key]) {
                    return image
                }
            }
            let inline = (dictionary["inlineData"] as? [String: Any]) ?? (dictionary["inline_data"] as? [String: Any])
            if let encoded = inline?["data"] as? String,
               let image = imageFromEncodedString(encoded) {
                return image
            }
            if let candidates = dictionary["candidates"] as? [[String: Any]] {
                return imageFromInlineCandidates(candidates)
            }
        }
        if let array = value as? [Any] {
            for item in array {
                if let image = imageFromJSONValue(item) {
                    return image
                }
            }
        }
        return nil
    }

    private static func imageFromEncodedString(_ value: String) -> UIImage? {
        let encoded: String
        if let comma = value.firstIndex(of: ","),
           value[..<comma].lowercased().contains("base64") {
            encoded = String(value[value.index(after: comma)...])
        } else {
            encoded = value
        }
        guard let imageData = Data(base64Encoded: encoded, options: [.ignoreUnknownCharacters]),
              let image = UIImage(data: imageData) else {
            return nil
        }
        return image
    }

    private static func imageFromInlineCandidates(_ candidates: [[String: Any]]) -> UIImage? {
        for candidate in candidates {
            guard let content = candidate["content"] as? [String: Any],
                  let parts = content["parts"] as? [[String: Any]] else { continue }
            for part in parts {
                let inline = (part["inlineData"] as? [String: Any]) ?? (part["inline_data"] as? [String: Any])
                guard let encoded = inline?["data"] as? String,
                      let image = imageFromEncodedString(encoded) else {
                    continue
                }
                return image
            }
        }
        return nil
    }

    private static func geminiResponseDetail(_ data: Data) -> String {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return String(data: data, encoding: .utf8)?.prefix(240).description ?? ""
        }
        var parts: [String] = []
        if let promptFeedback = root["promptFeedback"] as? [String: Any] {
            parts.append("promptFeedback=\(promptFeedback)")
        }
        if let candidates = root["candidates"] as? [[String: Any]] {
            for candidate in candidates {
                if let finishReason = candidate["finishReason"] as? String {
                    parts.append("finishReason=\(finishReason)")
                }
                if let content = candidate["content"] as? [String: Any],
                   let responseParts = content["parts"] as? [[String: Any]] {
                    for part in responseParts {
                        if let text = part["text"] as? String, !text.isEmpty {
                            parts.append("text=\(String(text.prefix(180)))")
                        }
                    }
                }
            }
        }
        if parts.isEmpty,
           let raw = String(data: data, encoding: .utf8) {
            parts.append(String(raw.prefix(240)))
        }
        return parts.joined(separator: " | ")
    }

    private static func composeEnhanceFrame(rendered: UIImage?, source: UIImage?) -> UIImage? {
        guard rendered != nil || source != nil else { return nil }
        let targetSize = rendered?.size ?? source?.size ?? .zero
        guard targetSize.width > 0, targetSize.height > 0 else { return rendered ?? source }
        let format = UIGraphicsImageRendererFormat()
        format.scale = rendered?.scale ?? source?.scale ?? UIScreen.main.scale
        format.opaque = true
        return UIGraphicsImageRenderer(size: targetSize, format: format).image { _ in
            if let source {
                let cover = coverImage(source, targetSize: targetSize, scale: format.scale)
                (blurred(cover, radius: 16) ?? cover).draw(in: CGRect(origin: .zero, size: targetSize))
            } else {
                UIColor.black.setFill()
                UIRectFill(CGRect(origin: .zero, size: targetSize))
            }
            rendered?.draw(in: CGRect(origin: .zero, size: targetSize), blendMode: .normal, alpha: 1)
        }
    }

    private static func coverImage(_ image: UIImage, targetSize: CGSize, scale: CGFloat) -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = scale
        format.opaque = true
        return UIGraphicsImageRenderer(size: targetSize, format: format).image { _ in
            UIColor.black.setFill()
            UIRectFill(CGRect(origin: .zero, size: targetSize))
            image.draw(in: aspectFillRect(imageSize: image.size, targetSize: targetSize, overscan: 1.2))
        }
    }

    private static func aspectFillRect(imageSize: CGSize, targetSize: CGSize, overscan: CGFloat = 1) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0 else {
            return CGRect(origin: .zero, size: targetSize)
        }
        let scale = max(targetSize.width / imageSize.width, targetSize.height / imageSize.height) * overscan
        let size = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
        return CGRect(x: (targetSize.width - size.width) / 2,
                      y: (targetSize.height - size.height) / 2,
                      width: size.width,
                      height: size.height)
    }

    private static func blurred(_ image: UIImage, radius: CGFloat) -> UIImage? {
        guard let input = CIImage(image: image) else { return nil }
        let filter = CIFilter.gaussianBlur()
        filter.inputImage = input.clampedToExtent()
        filter.radius = Float(radius)
        guard let output = filter.outputImage?.cropped(to: input.extent),
              let cgImage = CIContext().createCGImage(output, from: input.extent) else {
            return nil
        }
        return UIImage(cgImage: cgImage, scale: image.scale, orientation: image.imageOrientation)
    }

    private static func makeSubjectProtectionMask(from image: UIImage) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up, options: [:])
        do {
            if #available(iOS 17.0, *) {
                let request = VNGenerateForegroundInstanceMaskRequest()
                try handler.perform([request])
                guard let observation = request.results?.first,
                      !observation.allInstances.isEmpty else {
                    return nil
                }
                let maskBuffer = try observation.generateScaledMaskForImage(
                    forInstances: observation.allInstances,
                    from: handler
                )
                return protectionMaskImage(from: maskBuffer, scale: image.scale)
            } else {
                let request = VNGeneratePersonSegmentationRequest()
                request.qualityLevel = .balanced
                request.outputPixelFormat = kCVPixelFormatType_OneComponent8
                try handler.perform([request])
                guard let observation = request.results?.first else { return nil }
                return protectionMaskImage(from: observation.pixelBuffer, scale: image.scale)
            }
        } catch {
            print("⚠️ [OpenReshot] subject mask failed: \(error)")
            return nil
        }
    }

    private static func protectionMaskImage(from pixelBuffer: CVPixelBuffer, scale: CGFloat) -> UIImage? {
        let context = CIContext()
        let extent = CIImage(cvPixelBuffer: pixelBuffer).extent
        var output = CIImage(cvPixelBuffer: pixelBuffer)
        output = output
            .applyingFilter("CIMorphologyMaximum", parameters: [kCIInputRadiusKey: 5])
            .clampedToExtent()
            .applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: 4])
            .cropped(to: extent)
            .applyingFilter("CIColorInvert")
        guard let cgImage = context.createCGImage(output, from: extent) else { return nil }
        return UIImage(cgImage: cgImage, scale: scale, orientation: .up)
    }
}

private struct FluidPressButtonStyle: ButtonStyle {
    var pressedScale: CGFloat = 0.92

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? pressedScale : 1)
            .brightness(configuration.isPressed ? 0.06 : 0)
            .animation(.interactiveSpring(response: 0.28, dampingFraction: 0.78), value: configuration.isPressed)
    }
}

private struct OpenReshotGlassCircle: ViewModifier {
    let tint: Color?
    let interactive: Bool

    @ViewBuilder
    func body(content: Content) -> some View {
        if #available(iOS 26.0, *) {
            if let tint {
                content.glassEffect(.regular.tint(tint).interactive(interactive), in: Circle())
            } else {
                content.glassEffect(.regular.interactive(interactive), in: Circle())
            }
        } else {
            content
                .background(.ultraThinMaterial, in: Circle())
                .overlay(Circle().strokeBorder(.white.opacity(0.22), lineWidth: 0.8))
        }
    }
}

private struct OpenReshotGlassCapsule: ViewModifier {
    let tint: Color?
    let interactive: Bool

    @ViewBuilder
    func body(content: Content) -> some View {
        if #available(iOS 26.0, *) {
            if let tint {
                content.glassEffect(.regular.tint(tint).interactive(interactive), in: Capsule())
            } else {
                content.glassEffect(.regular.interactive(interactive), in: Capsule())
            }
        } else {
            content
                .background(.ultraThinMaterial, in: Capsule())
                .overlay(Capsule().strokeBorder(.white.opacity(0.24), lineWidth: 0.8))
        }
    }
}

private extension View {
    func openReshootGlassCircle(tint: Color? = nil, interactive: Bool = true) -> some View {
        modifier(OpenReshotGlassCircle(tint: tint, interactive: interactive))
    }

    func openReshootGlassCapsule(tint: Color? = nil, interactive: Bool = true) -> some View {
        modifier(OpenReshotGlassCapsule(tint: tint, interactive: interactive))
    }
}

private enum OpenReshotPalette {
    static let bg = Color(red: 248.0 / 255.0, green: 245.0 / 255.0, blue: 239.0 / 255.0)
    static let bgElevated = Color.white
    static let bgHover = Color(red: 234.0 / 255.0, green: 229.0 / 255.0, blue: 219.0 / 255.0)
    static let textPrimary = Color(red: 58.0 / 255.0, green: 52.0 / 255.0, blue: 46.0 / 255.0)
    static let textSecondary = Color(red: 112.0 / 255.0, green: 103.0 / 255.0, blue: 94.0 / 255.0)
    static let textTertiary = Color(red: 174.0 / 255.0, green: 180.0 / 255.0, blue: 192.0 / 255.0)
    static let accent = Color(red: 255.0 / 255.0, green: 138.0 / 255.0, blue: 91.0 / 255.0)
    static let accentMuted = Color(red: 98.0 / 255.0, green: 217.0 / 255.0, blue: 255.0 / 255.0)
    static let coolMist = Color(red: 138.0 / 255.0, green: 255.0 / 255.0, blue: 184.0 / 255.0)
    static let border = Color(red: 224.0 / 255.0, green: 222.0 / 255.0, blue: 215.0 / 255.0)
    static let borderSubtle = Color(red: 240.0 / 255.0, green: 235.0 / 255.0, blue: 226.0 / 255.0)
    static let twilightTop = Color(red: 9.0 / 255.0, green: 11.0 / 255.0, blue: 16.0 / 255.0)
    static let twilightMid = Color(red: 13.0 / 255.0, green: 18.0 / 255.0, blue: 30.0 / 255.0)
    static let twilightBottom = Color(red: 5.0 / 255.0, green: 7.0 / 255.0, blue: 13.0 / 255.0)
    static let twilightText = Color(red: 247.0 / 255.0, green: 248.0 / 255.0, blue: 250.0 / 255.0)
    static let twilightAccent = Color(red: 255.0 / 255.0, green: 179.0 / 255.0, blue: 106.0 / 255.0)
    static let twilightCoral = Color(red: 255.0 / 255.0, green: 122.0 / 255.0, blue: 92.0 / 255.0)
    static let twilightElectric = Color(red: 98.0 / 255.0, green: 217.0 / 255.0, blue: 255.0 / 255.0)
    static let twilightMint = Color(red: 139.0 / 255.0, green: 255.0 / 255.0, blue: 184.0 / 255.0)
    static let twilightInk = Color(red: 8.0 / 255.0, green: 11.0 / 255.0, blue: 18.0 / 255.0)
    static let twilightButtonText = Color(red: 20.0 / 255.0, green: 16.0 / 255.0, blue: 23.0 / 255.0)
    static let twilightPrimaryGradient = [
        Color(red: 255.0 / 255.0, green: 179.0 / 255.0, blue: 106.0 / 255.0),
        Color(red: 255.0 / 255.0, green: 122.0 / 255.0, blue: 92.0 / 255.0),
        Color(red: 98.0 / 255.0, green: 217.0 / 255.0, blue: 255.0 / 255.0)
    ]
    static let twilightRingGradient = [
        Color(red: 255.0 / 255.0, green: 179.0 / 255.0, blue: 106.0 / 255.0),
        Color(red: 255.0 / 255.0, green: 122.0 / 255.0, blue: 92.0 / 255.0),
        Color(red: 98.0 / 255.0, green: 217.0 / 255.0, blue: 255.0 / 255.0),
        Color(red: 139.0 / 255.0, green: 255.0 / 255.0, blue: 184.0 / 255.0),
        Color(red: 255.0 / 255.0, green: 179.0 / 255.0, blue: 106.0 / 255.0)
    ]
}

private struct EmptyHomeDot {
    let offsetX: CGFloat
    let offsetY: CGFloat
    let opacity: Double
}

private enum ReshotFlowPhase {
    case previewLoading
    case preview
    case inferring
    case failed
    case compose
    case generating
    case result
}

private struct CircleSectorShape: Shape {
    var progress: CGFloat

    var animatableData: CGFloat {
        get { progress }
        set { progress = newValue }
    }

    func path(in rect: CGRect) -> Path {
        let clamped = min(max(progress, 0), 1)
        guard clamped > 0 else { return Path() }
        let center = CGPoint(x: rect.midX, y: rect.midY)
        let radius = min(rect.width, rect.height) / 2
        var path = Path()
        path.move(to: center)
        path.addArc(
            center: center,
            radius: radius,
            startAngle: .degrees(-90),
            endAngle: .degrees(-90 + 360 * Double(clamped)),
            clockwise: false
        )
        path.closeSubpath()
        return path
    }
}

struct ContentView: View {
    @StateObject private var app = AppState()
    @State private var pickerItem: PhotosPickerItem?
    @State private var photoLoadTask: Task<Void, Never>?
    @State private var showingPhotoPicker = false
    @State private var glowSpin = false
    @State private var showingSettings = false
    @State private var showingGallery = false
    @State private var comparingResult = false
    @State private var comparisonWipePosition: CGFloat = 0.5
    @State private var generationStartedAt = Date()
    @State private var resultFlash = false
    @State private var saveToastVisible = false
    @State private var saveToastMessage = "已保存到相册"
    @State private var saveToastSystemImage = "checkmark"
    @State private var dragBaseTilt = SIMD2<Float>(0, 0)
    @State private var draggingStage = false
    @State private var showDragHint = false
    @State private var shutterMenuExpanded = false
    @State private var shutterLongPressTriggered = false
    @State private var frameStartPending = false
    @State private var resultMenuExpanded = false
    @State private var resultReplacementPending = false
    @State private var lensDockExpanded = false
    @State private var motionExportMenuExpanded = false
    @State private var dollyZoomTask: Task<Void, Never>?
    @State private var dollyZoomRunning = false
    @State private var focusGeminiKeyWhenSettingsOpen = false
    private let toolbarHeight: CGFloat = 88
    private static let didShowDragHintKey = "OpenReshot.didShowDragHint"

    var body: some View {
        return ZStack {
            GeometryReader { geo in
                if app.inputImage == nil {
                    emptyHome(size: geo.size)
                } else {
                    twilightPhotoFlow(size: geo.size)
                    .animation(.spring(response: 0.46, dampingFraction: 0.88), value: app.inputImage != nil)
                    .animation(.spring(response: 0.36, dampingFraction: 0.86), value: app.resultImage != nil)
                }
            }
        }
        .onAppear {
            glowSpin = true
            app.refreshGallery()
            // Headless self-test: launch with `-autotest` to run the bundled image, no UI taps.
            if ProcessInfo.processInfo.arguments.contains("-autotest"),
               let url = Bundle.main.url(forResource: "koala", withExtension: "png"),
               let data = try? Data(contentsOf: url), let img = UIImage(data: data) {
                print("🧪 [OpenReshot] autotest: reconstructing bundled koala.png")
                app.reconstruct(img, sourceData: data)
            } else {
                app.loadDemoSceneIfNeeded()
            }
        }
        .onChange(of: pickerItem) { _, item in
            guard let item else { return }
            print("📸 [OpenReshot] photo picked")
            photoLoadTask?.cancel()
            let taskID = app.beginImageLoadTask()
            showDragHint = false
            saveToastVisible = false
            comparingResult = false
            comparisonWipePosition = 0.5
            shutterMenuExpanded = false
            resultMenuExpanded = false
            frameStartPending = false
            resultReplacementPending = false
            photoLoadTask = Task { @MainActor in
                do {
                    guard let data = try await item.loadTransferable(type: Data.self) else {
                        guard app.isCurrentTask(taskID), !Task.isCancelled else { return }
                        print("❌ [OpenReshot] loadTransferable returned nil")
                        app.failTaskIfCurrent(taskID)
                        return
                    }
                    guard app.isCurrentTask(taskID), !Task.isCancelled else { return }
                    print("✅ [OpenReshot] loaded \(data.count) bytes")
                    guard let img = UIImage(data: data) else {
                        guard app.isCurrentTask(taskID), !Task.isCancelled else { return }
                        print("❌ [OpenReshot] UIImage(data:) failed")
                        app.failTaskIfCurrent(taskID)
                        return
                    }
                    app.reconstruct(img, sourceData: data, taskID: taskID)
                } catch is CancellationError {
                    return
                } catch {
                    guard app.isCurrentTask(taskID), !Task.isCancelled else { return }
                    print("❌ [OpenReshot] load error: \(error)")
                    app.failTaskIfCurrent(taskID)
                }
            }
        }
        .onChange(of: app.reconstructingFrame) { _, isGenerating in
            if isGenerating {
                showDragHint = false
                lensDockExpanded = false
                motionExportMenuExpanded = false
                cancelDollyZoom(reset: false)
                resultMenuExpanded = false
                generationStartedAt = Date()
            }
        }
        .onChange(of: app.rendererReady) { _, ready in
            guard ready, app.inputImage != nil else {
                lensDockExpanded = false
                motionExportMenuExpanded = false
                cancelDollyZoom(reset: true)
                return
            }
            revealShutterMenuAfterReconstruction()
            revealDragHintAfterRendererReady()
        }
        .onChange(of: app.resultImage) { _, image in
            guard image != nil else {
                comparingResult = false
                comparisonWipePosition = 0.5
                resultMenuExpanded = false
                resultReplacementPending = false
                motionExportMenuExpanded = false
                return
            }
            showDragHint = false
            shutterMenuExpanded = false
            lensDockExpanded = false
            motionExportMenuExpanded = false
            cancelDollyZoom(reset: false)
            resultMenuExpanded = false
            resultFlash = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.22) {
                resultFlash = false
            }
            revealResultMenuAfterImageAppears()
        }
        .onChange(of: app.saveState) { _, state in
            if state == .saved {
                saveToastMessage = "已保存到相册"
                saveToastSystemImage = "checkmark"
                withAnimation(.easeOut(duration: 0.20)) {
                    saveToastVisible = true
                }
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                    withAnimation(.easeInOut(duration: 0.28)) {
                        saveToastVisible = false
                    }
                }
            }
        }
        .onChange(of: app.motionExportState) { _, state in
            if state == .saved {
                motionExportMenuExpanded = false
                saveToastMessage = app.motionExportFormat?.savedMessage ?? "已保存到相册"
                saveToastSystemImage = "checkmark"
                withAnimation(.easeOut(duration: 0.20)) {
                    saveToastVisible = true
                }
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                    withAnimation(.easeInOut(duration: 0.28)) {
                        saveToastVisible = false
                    }
                }
            } else if state == .failed {
                saveToastMessage = app.motionExportFailureMessage ?? "导出失败"
                saveToastSystemImage = "exclamationmark.triangle"
                withAnimation(.easeOut(duration: 0.20)) {
                    saveToastVisible = true
                }
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                    withAnimation(.easeInOut(duration: 0.28)) {
                        saveToastVisible = false
                    }
                }
            }
        }
        .onDisappear {
            cancelDollyZoom(reset: true)
        }
        .sheet(isPresented: $showingSettings) {
            SettingsView(app: app, focusGeminiKeyOnAppear: focusGeminiKeyWhenSettingsOpen)
                .onDisappear {
                    focusGeminiKeyWhenSettingsOpen = false
                }
        }
        .sheet(isPresented: $showingGallery) {
            ReshotGalleryView(app: app) { item in
                openCachedGalleryItem(item)
            }
        }
        .photosPicker(isPresented: $showingPhotoPicker, selection: $pickerItem, matching: .images)
        .tint(OpenReshotPalette.accent)
    }

    private var appBackdrop: some View {
        LinearGradient(
            colors: [
                OpenReshotPalette.bgElevated,
                OpenReshotPalette.bg,
                OpenReshotPalette.bgHover.opacity(0.72)
            ],
            startPoint: .top,
            endPoint: .bottom
        )
        .ignoresSafeArea()
    }

    private func emptyHome(size: CGSize) -> some View {
        let scale = min(max(min(size.width / 402, size.height / 874), 0.78), 1.08)
        let stageWidth = min(size.width - 44, 330 * scale)
        let stageHeight = 452 * scale

        return ZStack {
            twilightHomeBackdrop

            VStack(spacing: 0) {
                Color.clear
                    .frame(height: 104 * scale)

                emptyHomeStage(width: stageWidth, height: stageHeight)

                Text("换个机位，开始拍摄")
                    .font(.system(size: 15 * scale, weight: .medium, design: .rounded))
                    .tracking(-0.2)
                    .foregroundStyle(OpenReshotPalette.twilightText.opacity(0.60))
                    .padding(.top, 20 * scale)
                    .frame(minHeight: 20 * scale)

                Spacer(minLength: 0)
            }
            .frame(width: size.width, height: size.height)

            bottomActionLayer(size: size, scale: scale) {
                PhotosPicker(selection: $pickerItem, matching: .images) {
                    emptyBottomAction(scale: scale)
                }
                .buttonStyle(FluidPressButtonStyle(pressedScale: 0.94))
                .accessibilityLabel("选择照片")
            }

            emptyHomeSettingsButton(size: size, scale: scale)
        }
    }

    private var twilightHomeBackdrop: some View {
        LinearGradient(
            colors: [
                OpenReshotPalette.twilightTop,
                OpenReshotPalette.twilightMid,
                OpenReshotPalette.twilightBottom
            ],
            startPoint: .top,
            endPoint: .bottom
        )
        .ignoresSafeArea()
    }

    private func emptyHomeSettingsButton(size: CGSize, scale: CGFloat) -> some View {
        VStack {
            HStack {
                Button {
                    app.refreshGallery()
                    showingGallery = true
                } label: {
                    HStack(spacing: 6 * scale) {
                        Image(systemName: "photo.stack")
                            .font(.system(size: 12 * scale, weight: .semibold))

                        Text("空间画廊")
                            .font(.system(size: 11 * scale, weight: .semibold, design: .rounded))
                            .tracking(0.6)
                    }
                    .foregroundStyle(OpenReshotPalette.twilightText.opacity(0.58))
                    .padding(.horizontal, 12 * scale)
                    .frame(height: 34 * scale)
                    .background(OpenReshotPalette.twilightBottom.opacity(0.42), in: Capsule())
                    .overlay(
                        Capsule()
                            .strokeBorder(OpenReshotPalette.twilightText.opacity(0.10), lineWidth: 1)
                    )
                    .contentShape(Capsule())
                }
                .buttonStyle(FluidPressButtonStyle(pressedScale: 0.92))
                .accessibilityLabel("打开空间画廊")
                .padding(.leading, 22)
                .padding(.top, 13 * scale)

                Spacer()

                Button {
                    openSettings()
                } label: {
                    Text("···")
                        .font(.system(size: 18 * scale, weight: .bold, design: .rounded))
                        .tracking(2)
                        .foregroundStyle(OpenReshotPalette.twilightText.opacity(0.45))
                        .frame(width: 48, height: 44)
                        .contentShape(Rectangle())
                }
                .buttonStyle(FluidPressButtonStyle(pressedScale: 0.88))
                .accessibilityLabel("设置")
                .padding(.trailing, 22)
                .padding(.top, 8 * scale)
            }
            Spacer()
        }
        .frame(width: size.width, height: size.height)
        .zIndex(10)
    }

    private func emptyHomeStage(width: CGFloat, height: CGFloat) -> some View {
        let illustrationWidth = width * 322 / 330
        let illustrationHeight = illustrationWidth * 4 / 3

        return ZStack(alignment: .center) {
            ZStack(alignment: .topLeading) {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(
                        OpenReshotPalette.twilightText.opacity(0.30),
                        style: StrokeStyle(lineWidth: 1.5, lineCap: .round, dash: [5, 4])
                    )
                    .frame(width: illustrationWidth * 0.50, height: illustrationWidth * 0.50 * 1.30)
                    .rotation3DEffect(.degrees(19), axis: (x: 0, y: 1, z: 0), perspective: 0.82)
                    .rotationEffect(.degrees(3.2))
                    .offset(x: illustrationWidth * 0.39, y: illustrationHeight * 0.16)

                ForEach(emptyHomeDots(width: illustrationWidth, height: illustrationHeight), id: \.offsetX) { dot in
                    Circle()
                        .fill(OpenReshotPalette.twilightAccent.opacity(dot.opacity))
                        .frame(width: 5, height: 5)
                        .offset(x: dot.offsetX, y: dot.offsetY)
                }

                TimelineView(.animation) { timeline in
                    let seconds = timeline.date.timeIntervalSinceReferenceDate
                    let phase = (sin(seconds * .pi * 2 / 5) + 1) / 2

                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(
                            LinearGradient(
                                colors: OpenReshotPalette.twilightPrimaryGradient,
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: illustrationWidth * 0.515, height: illustrationWidth * 0.515 * 1.31)
                        .shadow(color: OpenReshotPalette.twilightElectric.opacity(0.22), radius: 26, y: 22)
                        .rotation3DEffect(.degrees(-15 + 3 * phase), axis: (x: 0, y: 1, z: 0), perspective: 0.82)
                        .rotationEffect(.degrees(-2.8 + phase))
                        .offset(x: illustrationWidth * 0.13, y: illustrationHeight * 0.25 - 7 * phase)
                }
            }
            .frame(width: illustrationWidth, height: illustrationHeight, alignment: .topLeading)
        }
        .frame(width: width, height: height)
    }

    private func emptyHomeRingButton(scale: CGFloat) -> some View {
        return ZStack {
            Circle()
                .fill(
                    AngularGradient(
                        colors: OpenReshotPalette.twilightRingGradient,
                        center: .center,
                        angle: .degrees(220)
                    )
                )
                .frame(width: 74 * scale, height: 74 * scale)
                .shadow(color: OpenReshotPalette.twilightElectric.opacity(0.24), radius: 30, y: 10)

            Circle()
                .fill(OpenReshotPalette.twilightMid)
                .frame(width: 64 * scale, height: 64 * scale)

            Image(systemName: "plus")
                .font(.system(size: 25 * scale, weight: .regular, design: .rounded))
                .foregroundStyle(OpenReshotPalette.twilightText)
        }
        .frame(width: 82 * scale, height: 82 * scale)
    }

    private func emptyBottomAction(scale: CGFloat) -> some View {
        VStack(spacing: 14 * scale) {
            emptyHomeRingButton(scale: scale * 0.86)

            Text("放入照片")
                .font(.system(size: 11 * scale, weight: .bold, design: .rounded))
                .tracking(3)
                .foregroundStyle(OpenReshotPalette.twilightAccent.opacity(0.80))
                .frame(height: 12 * scale)
        }
        .contentShape(Rectangle())
    }

    private func bottomActionSlotHeight(scale: CGFloat) -> CGFloat {
        (82 * 0.86 + 14 + 12) * scale
    }

    private func bottomActionLayer<Content: View>(
        size: CGSize,
        scale: CGFloat,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(spacing: 0) {
            Spacer(minLength: 0)
            content()
                .frame(height: bottomActionSlotHeight(scale: scale), alignment: .top)
            Color.clear
                .frame(height: 0)
        }
        .frame(width: size.width, height: size.height)
        .zIndex(5)
    }

    private func emptyHomeDots(width: CGFloat, height: CGFloat) -> [EmptyHomeDot] {
        [
            EmptyHomeDot(offsetX: width * 0.64, offsetY: height * 0.12, opacity: 0.30),
            EmptyHomeDot(offsetX: width * 0.54, offsetY: height * 0.09, opacity: 0.50),
            EmptyHomeDot(offsetX: width * 0.43, offsetY: height * 0.09, opacity: 0.75),
            EmptyHomeDot(offsetX: width * 0.33, offsetY: height * 0.125, opacity: 1.00)
        ]
    }

    private var reshotFlowPhase: ReshotFlowPhase {
        if app.resultImage != nil { return .result }
        if app.reconstructingFrame { return .generating }
        if app.processFailed { return .failed }
        if app.loadingPreviewScene { return .previewLoading }
        if app.previewMode { return .preview }
        if app.reconstructingScene || !app.rendererReady { return .inferring }
        return .compose
    }

    private func twilightPhotoFlow(size: CGSize) -> some View {
        let scale = min(max(min(size.width / 402, size.height / 874), 0.78), 1.08)
        let photoTopInset = 66 * scale
        let bottomInset: CGFloat = 0
        let bottomActionHeight = bottomActionSlotHeight(scale: scale)
        let photoBottomGap = 20 * scale
        let bottomActionTop = size.height - bottomInset - bottomActionHeight
        let maxStageWidth = min(size.width - 8 * scale, 396 * scale)
        let maxStageHeight = max(1, bottomActionTop - photoTopInset - photoBottomGap)
        let stageSize = fittedPhotoStageSize(maxWidth: maxStageWidth, maxHeight: maxStageHeight)
        let phase = reshotFlowPhase

        return ZStack {
            twilightHomeBackdrop

            VStack(spacing: 0) {
                Color.clear
                    .frame(height: photoTopInset)

                twilightPhotoStage(width: stageSize.width, height: stageSize.height)
                    .overlay(alignment: .bottom) {
                        photoStageCaption(phase: phase, scale: scale)
                        .padding(.bottom, 12 * scale)
                    }

                Spacer(minLength: 0)
            }
            .frame(width: size.width, height: size.height)

            bottomActionLayer(size: size, scale: scale) {
                flowBottomAction(phase: phase, scale: scale)
            }

            let floatingDockWidth = max(1, min(stageSize.width - 50 * scale, 286 * scale))
            let floatingDockY = max(photoTopInset + 78 * scale, bottomActionTop - 98 * scale)

            if motionExportMenuExpanded || app.motionExportState == .rendering {
                motionExportDock(width: floatingDockWidth, scale: scale)
                    .position(x: size.width / 2, y: floatingDockY)
                    .transition(.asymmetric(
                        insertion: .opacity.combined(with: .scale(scale: 0.96)).animation(.spring(response: 0.30, dampingFraction: 0.84)),
                        removal: .opacity.animation(.easeOut(duration: 0.18))
                    ))
                    .zIndex(6)
            }

            if lensDockExpanded, phase == .compose, app.rendererReady, app.resultImage == nil {
                lensDock(width: floatingDockWidth, scale: scale)
                    .position(x: size.width / 2, y: floatingDockY)
                    .transition(.asymmetric(
                        insertion: .opacity.combined(with: .scale(scale: 0.96)).animation(.spring(response: 0.30, dampingFraction: 0.84)),
                        removal: .opacity.animation(.easeOut(duration: 0.18))
                    ))
                    .zIndex(6)
            }

            if saveToastVisible {
                saveToast(size: size, scale: scale)
                    .transition(.opacity.combined(with: .scale(scale: 0.96)))
            }

            emptyHomeSettingsButton(size: size, scale: scale)
        }
    }

    private func fittedPhotoStageSize(maxWidth: CGFloat, maxHeight: CGFloat) -> CGSize {
        let aspect = max(CGFloat(app.imageAspect), 0.01)
        let heightAtMaxWidth = maxWidth / aspect

        if heightAtMaxWidth <= maxHeight {
            return CGSize(width: maxWidth, height: heightAtMaxWidth)
        }

        return CGSize(width: maxHeight * aspect, height: maxHeight)
    }

    private var photoTiltLimit: SIMD2<Float> {
        let aspect = max(Float(app.imageAspect), 0.1)
        let wideAmount = Self.clamp((aspect - 1.15) / 0.85, 0, 1)
        let tallAmount = Self.clamp((1.0 / aspect - 1.0) / 1.0, 0, 1)
        let baseLimit = Self.clamp(0.72 - 0.24 * wideAmount + 0.10 * tallAmount, 0.42, 0.82)
        let limit = Self.clamp(baseLimit * app.viewAngleMode.motionScale, 0.12, 0.82)
        return SIMD2(limit, limit)
    }

    @ViewBuilder
    private func flowBottomAction(phase: ReshotFlowPhase, scale: CGFloat) -> some View {
        if phase == .result {
            resultActionRow(scale: scale)
        } else if phase == .preview {
            Button {
                openPhotoPickerForReplacement()
            } label: {
                emptyBottomAction(scale: scale)
            }
            .buttonStyle(FluidPressButtonStyle(pressedScale: 0.94))
            .accessibilityLabel("放入照片")
        } else if phase == .failed {
            failedActionRow(scale: scale)
        } else if phase == .compose {
            composeActionRow(scale: scale)
        } else {
            VStack(spacing: 5 * scale) {
                Button {
                } label: {
                    flowRingButton(phase: phase, scale: scale * 0.86)
                }
                .disabled(true)
                .buttonStyle(FluidPressButtonStyle(pressedScale: 0.93))
                .accessibilityLabel(flowRingLabel(for: phase))

                Text(flowRingLabel(for: phase))
                    .font(.system(size: 10 * scale, weight: .semibold, design: .rounded))
                    .tracking(0.8)
                    .foregroundStyle(OpenReshotPalette.twilightAccent.opacity(0.80))
                    .frame(minHeight: 12 * scale)
            }
        }
    }

    private func failedActionRow(scale: CGFloat) -> some View {
        ZStack {
            Button {
                openPhotoPickerForReplacement()
            } label: {
                composeAuxiliaryButton(systemImage: "photo.on.rectangle", title: "换图", scale: scale)
            }
            .buttonStyle(FluidPressButtonStyle(pressedScale: 0.94))
            .accessibilityLabel("换一张照片")
            .offset(x: -88 * scale)

            Button {
                retryFailedReconstruction()
            } label: {
                VStack(spacing: 5 * scale) {
                    flowRingButton(phase: .failed, scale: scale * 0.86)

                    Text(flowRingLabel(for: .failed))
                        .font(.system(size: 10 * scale, weight: .semibold, design: .rounded))
                        .tracking(0.8)
                        .foregroundStyle(OpenReshotPalette.twilightAccent.opacity(0.80))
                        .frame(minHeight: 12 * scale)
                }
            }
            .buttonStyle(FluidPressButtonStyle(pressedScale: 0.93))
            .accessibilityLabel(failedRetryAccessibilityLabel)

            if shouldShowFailedSettingsButton {
                Button {
                    openFailedSettingsAction()
                } label: {
                    composeAuxiliaryButton(systemImage: failedSettingsIcon, title: failedSettingsTitle, scale: scale)
                }
                .buttonStyle(FluidPressButtonStyle(pressedScale: 0.94))
                .accessibilityLabel(failedSettingsTitle)
                .offset(x: 88 * scale)
            }
        }
        .frame(maxWidth: .infinity)
        .contentShape(Rectangle())
        .padding(.horizontal, 42 * scale)
    }

    private func retryFailedReconstruction() {
        switch app.processFailureKind {
        case .missingAPIKey:
            openSettings(focusGeminiKey: true)
        case .enhancement:
            app.reconstructCurrentFrame()
        case .imageLoad:
            openPhotoPickerForReplacement()
        case .missingModel:
            openSettings()
        case .reconstruction, nil:
            guard app.modelStore.activeModelURL() != nil else {
                openSettings()
                return
            }
            app.retryCurrentReconstruction()
        }
    }

    private var failedRetryAccessibilityLabel: String {
        switch app.processFailureKind {
        case .missingAPIKey:
            return "输入 Gemini API Key"
        case .enhancement:
            return "重新拍摄"
        case .imageLoad:
            return "重新选择照片"
        case .missingModel:
            return "下载模型"
        case .reconstruction, nil:
            return app.modelStore.activeModelURL() == nil ? "下载模型" : "重试构建空间"
        }
    }

    private var failedRetrySystemImage: String {
        switch app.processFailureKind {
        case .missingAPIKey:
            return "key.fill"
        case .enhancement:
            return "arrow.clockwise"
        case .imageLoad:
            return "photo.on.rectangle"
        case .missingModel:
            return "arrow.down"
        case .reconstruction, nil:
            guard app.modelStore.activeModelURL() != nil else {
                return "arrow.down"
            }
            return "arrow.clockwise"
        }
    }

    private var failedRetryLabel: String {
        switch app.processFailureKind {
        case .missingAPIKey:
            return "输入 API"
        case .enhancement:
            return "重试"
        case .imageLoad:
            return "换图"
        case .missingModel:
            return "下载模型"
        case .reconstruction, nil:
            guard app.modelStore.activeModelURL() != nil else {
                return "下载模型"
            }
            return "重试"
        }
    }

    private var failedCaption: String {
        switch app.processFailureKind {
        case .missingAPIKey:
            return "需要输入 API Key 后拍摄"
        case .enhancement:
            return "拍摄失败,请检查网络后重试"
        case .imageLoad:
            return "载入失败,请重新选择照片"
        case .missingModel:
            return "需要下载模型后构建空间"
        case .reconstruction, nil:
            return "构建失败,请检查模型或重试"
        }
    }

    private var failedSettingsTitle: String {
        switch app.processFailureKind {
        case .missingAPIKey:
            return "API"
        case .enhancement:
            return "设置"
        case .imageLoad:
            return "设置"
        case .missingModel:
            return "模型"
        case .reconstruction, nil:
            guard app.modelStore.activeModelURL() != nil else {
                return "模型"
            }
            return "设置"
        }
    }

    private var failedSettingsIcon: String {
        switch app.processFailureKind {
        case .missingAPIKey:
            return "key.fill"
        case .missingModel:
            return "arrow.down.circle"
        case .reconstruction, nil:
            guard app.modelStore.activeModelURL() != nil else {
                return "arrow.down.circle"
            }
            return "gearshape"
        default:
            return "gearshape"
        }
    }

    private var shouldShowFailedSettingsButton: Bool {
        app.processFailureKind != .imageLoad
    }

    private func openFailedSettingsAction() {
        if app.processFailureKind == .missingAPIKey {
            openSettings(focusGeminiKey: true)
            return
        }
        if app.processFailureKind == .reconstruction, app.modelStore.activeModelURL() == nil {
            openSettings()
            return
        }
        openSettings()
    }

    private func openSettings(focusGeminiKey: Bool = false) {
        focusGeminiKeyWhenSettingsOpen = focusGeminiKey
        showingSettings = true
    }

    private func openCachedGalleryItem(_ item: ReshotCacheItem) {
        frameStartPending = false
        resultReplacementPending = false
        resultMenuExpanded = false
        shutterMenuExpanded = false
        lensDockExpanded = false
        motionExportMenuExpanded = false
        showDragHint = false
        saveToastVisible = false
        comparingResult = false
        comparisonWipePosition = 0.5
        dragBaseTilt = .zero
        cancelDollyZoom(reset: true)
        pickerItem = nil
        app.openCachedReshot(item)
    }

    private func composeActionRow(scale: CGFloat) -> some View {
        ZStack {
            Button {
                openPhotoPickerForReplacement()
            } label: {
                composeAuxiliaryButton(systemImage: "photo.on.rectangle", title: "换图", scale: scale)
            }
            .buttonStyle(FluidPressButtonStyle(pressedScale: 0.94))
            .accessibilityLabel("换一张照片")
            .offset(x: shutterMenuExpanded ? -132 * scale : 0, y: shutterMenuExpanded ? 0 : -2 * scale)
            .scaleEffect(shutterMenuExpanded ? 1 : 0.24)
            .opacity(shutterMenuExpanded ? 1 : 0)
            .blur(radius: shutterMenuExpanded ? 0 : 5)
            .allowsHitTesting(shutterMenuExpanded)

            Button {
                resetLensAndViewpoint()
                withAnimation(.spring(response: 0.30, dampingFraction: 0.84)) {
                    lensDockExpanded = false
                    motionExportMenuExpanded = false
                    showDragHint = false
                }
            } label: {
                composeAuxiliaryButton(systemImage: "arrow.counterclockwise", title: "复位", scale: scale)
            }
            .buttonStyle(FluidPressButtonStyle(pressedScale: 0.94))
            .accessibilityLabel("复位镜头")
            .offset(x: shutterMenuExpanded ? -66 * scale : 0, y: shutterMenuExpanded ? 0 : -2 * scale)
            .scaleEffect(shutterMenuExpanded ? 1 : 0.24)
            .opacity(shutterMenuExpanded ? 1 : 0)
            .blur(radius: shutterMenuExpanded ? 0 : 5)
            .allowsHitTesting(shutterMenuExpanded)

            Button {
                withAnimation(.spring(response: 0.30, dampingFraction: 0.84)) {
                    lensDockExpanded.toggle()
                    motionExportMenuExpanded = false
                    showDragHint = false
                }
            } label: {
                composeAuxiliaryButton(systemImage: "camera.aperture", title: "镜头", scale: scale)
            }
            .buttonStyle(FluidPressButtonStyle(pressedScale: 0.94))
            .accessibilityLabel(lensDockExpanded ? "收起镜头控制" : "打开镜头控制")
            .offset(x: shutterMenuExpanded ? 66 * scale : 0, y: shutterMenuExpanded ? 0 : -2 * scale)
            .scaleEffect(shutterMenuExpanded ? 1 : 0.24)
            .opacity(shutterMenuExpanded ? 1 : 0)
            .blur(radius: shutterMenuExpanded ? 0 : 5)
            .allowsHitTesting(shutterMenuExpanded)

            Button {
                withAnimation(.spring(response: 0.30, dampingFraction: 0.84)) {
                    motionExportMenuExpanded.toggle()
                    lensDockExpanded = false
                    showDragHint = false
                }
            } label: {
                composeAuxiliaryButton(systemImage: "square.and.arrow.up", title: "导出", scale: scale)
            }
            .buttonStyle(FluidPressButtonStyle(pressedScale: 0.94))
            .accessibilityLabel(motionExportMenuExpanded ? "收起动效导出" : "打开动效导出")
            .offset(x: shutterMenuExpanded ? 132 * scale : 0, y: shutterMenuExpanded ? 0 : -2 * scale)
            .scaleEffect(shutterMenuExpanded ? 1 : 0.24)
            .opacity(shutterMenuExpanded ? 1 : 0)
            .blur(radius: shutterMenuExpanded ? 0 : 5)
            .allowsHitTesting(shutterMenuExpanded)

            VStack(spacing: 5 * scale) {
                Button {
                    if shutterLongPressTriggered {
                        shutterLongPressTriggered = false
                        return
                    }
                    collapseShutterMenuThenReconstruct()
                } label: {
                    flowRingButton(phase: .compose, scale: scale * 0.86)
                }
                .simultaneousGesture(
                    LongPressGesture(minimumDuration: 0.34)
                        .onEnded { _ in
                            shutterLongPressTriggered = true
                            withAnimation(.spring(response: 0.28, dampingFraction: 0.78)) {
                                shutterMenuExpanded = true
                            }
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                shutterLongPressTriggered = false
                            }
                        }
                )
                .buttonStyle(FluidPressButtonStyle(pressedScale: 0.93))
                .accessibilityLabel("开始拍摄")

                Text(flowRingLabel(for: .compose))
                    .font(.system(size: 10 * scale, weight: .semibold, design: .rounded))
                    .tracking(0.8)
                    .foregroundStyle(OpenReshotPalette.twilightAccent.opacity(0.80))
                    .frame(minHeight: 12 * scale)
            }
        }
        .frame(maxWidth: .infinity)
        .contentShape(Rectangle())
        .padding(.horizontal, 42 * scale)
    }

    private func revealShutterMenuAfterReconstruction() {
        guard !app.previewMode, app.inputImage != nil, app.rendererReady, app.resultImage == nil, !app.reconstructingFrame else { return }
        frameStartPending = false
        withAnimation(.spring(response: 0.46, dampingFraction: 0.70, blendDuration: 0.06)) {
            shutterMenuExpanded = true
        }
    }

    private func revealDragHintAfterRendererReady() {
        guard app.inputImage != nil, app.rendererReady, app.resultImage == nil, !app.reconstructingFrame else { return }
        guard !UserDefaults.standard.bool(forKey: Self.didShowDragHintKey) else { return }
        UserDefaults.standard.set(true, forKey: Self.didShowDragHintKey)
        withAnimation(.easeOut(duration: 0.24)) {
            showDragHint = true
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.2) {
            guard showDragHint, app.rendererReady, app.resultImage == nil, !app.reconstructingFrame, !draggingStage else { return }
            withAnimation(.easeOut(duration: 0.32)) {
                showDragHint = false
            }
        }
    }

    private func collapseShutterMenuThenReconstruct() {
        guard !app.previewMode else { return }
        guard !frameStartPending else { return }
        frameStartPending = true
        generationStartedAt = Date()
        withAnimation(.spring(response: 0.24, dampingFraction: 0.88)) {
            shutterMenuExpanded = false
            lensDockExpanded = false
            motionExportMenuExpanded = false
        }
        cancelDollyZoom(reset: false)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.20) {
            guard frameStartPending else { return }
            app.reconstructCurrentFrame()
            frameStartPending = false
        }
    }

    private func revealResultMenuAfterImageAppears() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) {
            guard app.resultImage != nil, !app.reconstructingFrame else { return }
            withAnimation(.spring(response: 0.46, dampingFraction: 0.70, blendDuration: 0.06)) {
                resultMenuExpanded = true
            }
        }
    }

    private func collapseResultMenuThenOpenPhotoPicker() {
        guard !resultReplacementPending else { return }
        resultReplacementPending = true
        withAnimation(.spring(response: 0.24, dampingFraction: 0.88)) {
            resultMenuExpanded = false
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.20) {
            guard resultReplacementPending else { return }
            resultReplacementPending = false
            openPhotoPickerForReplacement()
        }
    }

    private func openPhotoPickerForReplacement() {
        frameStartPending = false
        resultReplacementPending = false
        resultMenuExpanded = false
        photoLoadTask?.cancel()
        photoLoadTask = nil
        app.cancelCurrentTaskAndClear()
        showDragHint = false
        lensDockExpanded = false
        motionExportMenuExpanded = false
        cancelDollyZoom(reset: true)
        saveToastVisible = false
        comparingResult = false
        dragBaseTilt = .zero
        pickerItem = nil
        showingPhotoPicker = true
    }

    private func composeAuxiliaryButton(systemImage: String, title: String, scale: CGFloat) -> some View {
        VStack(spacing: 5 * scale) {
            ZStack {
                Circle()
                    .fill(OpenReshotPalette.twilightBottom.opacity(0.72))
                    .frame(width: 42 * scale, height: 42 * scale)
                    .overlay(
                        Circle()
                            .strokeBorder(OpenReshotPalette.twilightText.opacity(0.30), lineWidth: 1.2)
                    )

                Image(systemName: systemImage)
                    .font(.system(size: 15 * scale, weight: .medium))
                    .symbolRenderingMode(.monochrome)
                    .foregroundStyle(OpenReshotPalette.twilightText.opacity(0.82))
            }
            .shadow(color: .black.opacity(0.30), radius: 14, y: 6)

            Text(title)
                .font(.system(size: 9 * scale, weight: .semibold, design: .rounded))
                .tracking(0.6)
                .foregroundStyle(OpenReshotPalette.twilightText.opacity(0.52))
        }
        .frame(width: 54 * scale, height: 66 * scale)
    }

    private func resetPhotoViewpoint() {
        dragBaseTilt = .zero
        app.renderer?.setTiltTarget(.zero)
        app.updateSheen(for: .zero)
    }

    private func resetLensAndViewpoint() {
        cancelDollyZoom(reset: true)
        resetPhotoViewpoint()
        app.resetLensControlsToDefault()
    }

    private func setLensViewAngleMode(_ mode: ReshotViewAngleMode) {
        app.setViewAngleMode(mode)
        let limit = photoTiltLimit
        let tilt = SIMD2(
            Self.clamp(app.motionTilt.x, -limit.x, limit.x),
            Self.clamp(app.motionTilt.y, -limit.y, limit.y)
        )
        app.renderer?.setTiltTarget(tilt)
        app.updateSheen(for: tilt)
    }

    private func playDollyZoom() {
        guard !dollyZoomRunning else { return }
        cancelDollyZoom(reset: false)
        let focusDepth = max(app.lensFocusDepth, 0.1)
        let forward = min(0.58, max(0.12, focusDepth * 0.30), max(0.05, focusDepth - 0.10))
        let backward = -min(0.32, max(0.08, focusDepth * 0.16))

        dollyZoomRunning = true
        dollyZoomTask = Task { @MainActor in
            defer {
                dollyZoomRunning = false
                dollyZoomTask = nil
            }

            await animateLensDolly(from: app.lensDolly, to: forward, duration: 0.78)
            guard !Task.isCancelled else { return }
            await animateLensDolly(from: forward, to: backward, duration: 1.05)
            guard !Task.isCancelled else { return }
            await animateLensDolly(from: backward, to: 0, duration: 0.62)
            guard !Task.isCancelled else { return }
            app.setLensDolly(0)
        }
    }

    private func cancelDollyZoom(reset: Bool) {
        dollyZoomTask?.cancel()
        dollyZoomTask = nil
        dollyZoomRunning = false
        if reset {
            app.setLensDolly(0)
        }
    }

    @MainActor
    private func animateLensDolly(from start: Float, to end: Float, duration: TimeInterval) async {
        let frames = max(1, Int(duration * 60))
        for frame in 0...frames {
            if Task.isCancelled { return }
            let t = Float(frame) / Float(frames)
            let eased = t * t * (3 - 2 * t)
            app.setLensDolly(start + (end - start) * eased)
            try? await Task.sleep(nanoseconds: 16_666_667)
        }
    }

    private func lensDock(width: CGFloat, scale: CGFloat) -> some View {
        glassDockGroup(spacing: 7 * scale) {
            VStack(spacing: 7 * scale) {
                lensQuickControlDock(width: width, scale: scale)
                lensSliderDock(width: width, scale: scale)
            }
        }
    }

    private func lensQuickControlDock(width: CGFloat, scale: CGFloat) -> some View {
        HStack(spacing: 6 * scale) {
            ForEach(ReshotViewAngleMode.allCases) { mode in
                lensAngleModeButton(mode, scale: scale)
            }

            Button {
                playDollyZoom()
            } label: {
                dockIconButton(systemName: dollyZoomRunning ? "pause.fill" : "play.fill", highlighted: dollyZoomRunning, scale: scale)
            }
            .disabled(dollyZoomRunning)
            .buttonStyle(FluidPressButtonStyle(pressedScale: 0.90))
            .accessibilityLabel("播放推轨变焦")

            Button {
                resetLensAndViewpoint()
            } label: {
                dockIconButton(systemName: "arrow.counterclockwise", scale: scale)
            }
            .buttonStyle(FluidPressButtonStyle(pressedScale: 0.90))
            .accessibilityLabel("复位镜头")
        }
        .padding(.horizontal, 11 * scale)
        .padding(.vertical, 10 * scale)
        .frame(width: width)
        .openReshotGlassPanel(cornerRadius: 22)
    }

    private func lensSliderDock(width: CGFloat, scale: CGFloat) -> some View {
        VStack(spacing: 7 * scale) {
            lensValueSlider(
                systemImage: "scope",
                leading: "对焦",
                value: Binding(
                    get: { Double(app.lensFocusDepth) },
                    set: {
                        cancelDollyZoom(reset: false)
                        app.setLensFocusDepth(Float($0))
                    }
                ),
                range: Double(app.lensFocusMin)...Double(app.lensFocusMax),
                valueText: lensFocusLabel,
                scale: scale
            )

            lensValueSlider(
                systemImage: "camera.aperture",
                leading: "光圈",
                value: Binding(
                    get: { Double(app.lensFNumber) },
                    set: { app.setLensFNumber(Float($0)) }
                ),
                range: 1.4...16.0,
                valueText: lensFNumberLabel,
                scale: scale
            )

            lensValueSlider(
                systemImage: "arrow.left.and.right",
                leading: "推轨",
                value: Binding(
                    get: { Double(app.lensDolly) },
                    set: {
                        cancelDollyZoom(reset: false)
                        app.setLensDolly(Float($0))
                    }
                ),
                range: Double(app.lensDollyRange.lowerBound)...Double(app.lensDollyRange.upperBound),
                valueText: lensDollyLabel,
                centered: true,
                scale: scale
            )
        }
        .padding(.horizontal, 11 * scale)
        .padding(.vertical, 10 * scale)
        .frame(width: width)
        .openReshotGlassPanel(cornerRadius: 22)
    }

    @ViewBuilder
    private func glassDockGroup<Content: View>(
        spacing: CGFloat,
        @ViewBuilder content: () -> Content
    ) -> some View {
        if #available(iOS 26.0, *) {
            GlassEffectContainer(spacing: spacing) {
                content()
            }
        } else {
            content()
        }
    }

    private func lensAngleModeButton(_ mode: ReshotViewAngleMode, scale: CGFloat) -> some View {
        let selected = app.viewAngleMode == mode
        return Button {
            setLensViewAngleMode(mode)
        } label: {
            Text(mode.title)
                .font(.system(size: 8.5 * scale, weight: .bold, design: .rounded))
                .foregroundStyle(selected ? OpenReshotPalette.twilightText.opacity(0.86) : OpenReshotPalette.twilightText.opacity(0.56))
                .lineLimit(1)
                .minimumScaleFactor(0.78)
                .frame(maxWidth: .infinity)
                .frame(height: 40 * scale)
                .background(
                    selected ? OpenReshotPalette.twilightText.opacity(0.16) : OpenReshotPalette.twilightText.opacity(0.065),
                    in: RoundedRectangle(cornerRadius: 12 * scale, style: .continuous)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 12 * scale, style: .continuous)
                        .strokeBorder(selected ? OpenReshotPalette.twilightAccent.opacity(0.36) : OpenReshotPalette.twilightText.opacity(0.06), lineWidth: 0.8)
                )
        }
        .buttonStyle(FluidPressButtonStyle(pressedScale: 0.96))
        .accessibilityLabel("切换到\(mode.title)视角")
    }

    private func motionExportDock(width: CGFloat, scale: CGFloat) -> some View {
        VStack(spacing: 7 * scale) {
            HStack(spacing: 6 * scale) {
                ForEach(MotionExportFormat.allCases) { format in
                    motionExportFormatButton(format, scale: scale)
                }
            }

            if app.motionExportState == .rendering {
                ProgressView(value: app.motionExportProgress)
                    .tint(OpenReshotPalette.twilightAccent.opacity(0.92))
                    .frame(height: 3 * scale)
                    .clipShape(Capsule())
            }
        }
        .padding(.horizontal, 11 * scale)
        .padding(.vertical, 10 * scale)
        .frame(width: width)
        .openReshotGlassPanel(cornerRadius: 22)
    }

    private func motionExportFormatButton(_ format: MotionExportFormat, scale: CGFloat) -> some View {
        let enabled = motionExportEnabled(format)
        let active = app.motionExportFormat == format && app.motionExportState == .rendering
        return Button {
            app.exportMotion(format)
        } label: {
            VStack(spacing: 5 * scale) {
                Image(systemName: motionExportIcon(for: format))
                    .font(.system(size: 12.5 * scale, weight: .semibold))
                    .symbolRenderingMode(.hierarchical)

                Text(motionExportTitle(for: format))
                    .font(.system(size: 8.5 * scale, weight: .bold, design: .rounded))
                    .lineLimit(1)
                    .minimumScaleFactor(0.78)
            }
            .foregroundStyle(enabled || active ? OpenReshotPalette.twilightText.opacity(0.76) : OpenReshotPalette.twilightText.opacity(0.26))
            .frame(maxWidth: .infinity)
            .frame(height: 40 * scale)
            .background(
                active ? OpenReshotPalette.twilightText.opacity(0.16) : OpenReshotPalette.twilightText.opacity(enabled ? 0.065 : 0.03),
                in: RoundedRectangle(cornerRadius: 12 * scale, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12 * scale, style: .continuous)
                    .strokeBorder(active ? OpenReshotPalette.twilightAccent.opacity(0.36) : OpenReshotPalette.twilightText.opacity(0.06), lineWidth: 0.8)
            )
        }
        .disabled(!enabled)
        .buttonStyle(FluidPressButtonStyle(pressedScale: 0.96))
        .accessibilityLabel("导出\(format.title)")
    }

    private func motionExportEnabled(_ format: MotionExportFormat) -> Bool {
        guard app.motionExportState != .rendering, app.rendererReady, app.resultImage == nil else { return false }
        return true
    }

    private func motionExportIcon(for format: MotionExportFormat) -> String {
        if app.motionExportState == .rendering, app.motionExportFormat == format {
            return "hourglass"
        }
        if app.motionExportState == .saved, app.motionExportFormat == format {
            return "checkmark"
        }
        return format.systemImage
    }

    private func motionExportTitle(for format: MotionExportFormat) -> String {
        if app.motionExportState == .rendering, app.motionExportFormat == format {
            return "\(Int(app.motionExportProgress * 100))%"
        }
        if app.motionExportState == .failed, app.motionExportFormat == format {
            return "重试"
        }
        return format.title
    }

    private func dockIconButton(systemName: String, highlighted: Bool = false, scale: CGFloat) -> some View {
        Image(systemName: systemName)
            .font(.system(size: 10.5 * scale, weight: .semibold))
            .foregroundStyle(OpenReshotPalette.twilightText.opacity(highlighted ? 0.92 : 0.72))
            .frame(width: 24 * scale, height: 24 * scale)
            .background(OpenReshotPalette.twilightText.opacity(highlighted ? 0.14 : 0.075), in: Circle())
            .overlay(
                Circle()
                    .strokeBorder(OpenReshotPalette.twilightText.opacity(highlighted ? 0.12 : 0.07), lineWidth: 0.8)
            )
            .contentShape(Circle())
    }

    private func lensValueSlider(
        systemImage: String,
        leading: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
        valueText: String,
        centered: Bool = false,
        scale: CGFloat
    ) -> some View {
        HStack(spacing: 7 * scale) {
            Image(systemName: systemImage)
                .font(.system(size: 10 * scale, weight: .semibold))
                .foregroundStyle(OpenReshotPalette.twilightText.opacity(0.58))
                .frame(width: 14 * scale)

            Text(leading)
                .font(.system(size: 8.5 * scale, weight: .semibold, design: .rounded))
                .foregroundStyle(OpenReshotPalette.twilightText.opacity(0.52))
                .frame(width: 25 * scale, alignment: .leading)

            LensControlSlider(
                value: value,
                range: range,
                centered: centered,
                scale: scale
            )

            Text(valueText)
                .font(.system(size: 8.5 * scale, weight: .semibold, design: .rounded))
                .foregroundStyle(OpenReshotPalette.twilightText.opacity(0.62))
                .monospacedDigit()
                .lineLimit(1)
                .minimumScaleFactor(0.82)
                .frame(width: 38 * scale, alignment: .trailing)
        }
        .frame(height: 22 * scale)
    }

    private struct LensControlSlider: View {
        @Binding var value: Double
        let range: ClosedRange<Double>
        let centered: Bool
        let scale: CGFloat

        var body: some View {
            GeometryReader { proxy in
                let width = max(proxy.size.width, 1)
                let progress = CGFloat(sliderProgress(value))
                let center = centered ? CGFloat(sliderProgress(0)) : 0

                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(OpenReshotPalette.twilightText.opacity(0.13))
                        .frame(height: 2.4 * scale)

                    Capsule()
                        .fill(OpenReshotPalette.twilightAccent.opacity(0.92))
                        .frame(width: max(2.4 * scale, abs(progress - center) * width), height: 2.4 * scale)
                        .offset(x: min(progress, center) * width)

                    Circle()
                        .fill(OpenReshotPalette.twilightText.opacity(0.96))
                        .frame(width: 15.5 * scale, height: 15.5 * scale)
                        .shadow(color: .black.opacity(0.25), radius: 6, y: 3)
                        .offset(x: progress * width - 7.75 * scale)
                }
                .frame(height: proxy.size.height)
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { gesture in
                            let x = min(max(gesture.location.x, 0), width)
                            let nextProgress = Double(x / width)
                            value = range.lowerBound + nextProgress * (range.upperBound - range.lowerBound)
                        }
                )
            }
            .frame(height: 24 * scale)
        }

        private func sliderProgress(_ candidate: Double) -> Double {
            let span = max(range.upperBound - range.lowerBound, 0.0001)
            return min(max((candidate - range.lowerBound) / span, 0), 1)
        }
    }

    private var lensFNumberLabel: String {
        let value = app.lensFNumber
        if abs(value.rounded() - value) < 0.05 {
            return "f/\(Int(value.rounded()))"
        }
        return String(format: "f/%.1f", value)
    }

    private var lensFocusLabel: String {
        String(format: "%.1fm", app.lensFocusDepth)
    }

    private var lensDollyLabel: String {
        if abs(app.lensDolly) < 0.01 {
            return "0.00m"
        }
        return String(format: "%+.2fm", app.lensDolly)
    }

    private func twilightPhotoStage(width: CGFloat, height: CGFloat) -> some View {
        return ZStack {
            photoStage(width: width, height: height)
                .frame(width: width, height: height)
                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                .shadow(color: .black.opacity(0.50), radius: 30, y: 26)
        }
        .frame(width: width, height: height)
    }

    private func dragHintOverlay(scale: CGFloat) -> some View {
        TimelineView(.animation) { timeline in
            let phase = sin(timeline.date.timeIntervalSinceReferenceDate * 4.2)
            let drift = CGFloat(phase) * 8 * scale

            HStack {
                Image(systemName: "chevron.left")
                    .offset(x: -drift)

                Spacer()

                Image(systemName: "chevron.right")
                    .offset(x: drift)
            }
            .font(.system(size: 24 * scale, weight: .semibold))
            .foregroundStyle(OpenReshotPalette.twilightText.opacity(0.62))
            .padding(.horizontal, 16 * scale)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(
                LinearGradient(
                    colors: [
                        OpenReshotPalette.twilightBottom.opacity(0.28),
                        .clear,
                        OpenReshotPalette.twilightBottom.opacity(0.28)
                    ],
                    startPoint: .leading,
                    endPoint: .trailing
                )
            )
        }
        .allowsHitTesting(false)
    }

    @ViewBuilder
    private func photoStageCaption(phase: ReshotFlowPhase, scale: CGFloat) -> some View {
        if let caption = flowCaption(for: phase) {
            Text(caption)
                .font(.system(size: 13 * scale, weight: .medium, design: .rounded))
                .tracking(-0.1)
                .foregroundStyle(OpenReshotPalette.twilightText.opacity(0.78))
                .padding(.horizontal, 13 * scale)
                .frame(height: 28 * scale)
                .background(OpenReshotPalette.twilightBottom.opacity(0.42), in: Capsule())
        }
    }

    private func flowCaption(for phase: ReshotFlowPhase) -> String? {
        switch phase {
        case .previewLoading, .preview, .inferring, .generating:
            return nil
        case .failed:
            return failedCaption
        case .compose:
            return nil
        case .result:
            return nil
        }
    }

    private func flowRingLabel(for phase: ReshotFlowPhase) -> String {
        switch phase {
        case .previewLoading:
            return "载入预览"
        case .preview:
            return ""
        case .inferring:
            return "构建空间"
        case .failed:
            return failedRetryLabel
        case .compose:
            return "开始拍摄"
        case .generating:
            return "正在拍摄"
        case .result:
            return ""
        }
    }

    private func flowRingButton(phase: ReshotFlowPhase, scale: CGFloat) -> some View {
        TimelineView(.animation) { timeline in
            let rotationSpeed = phase == .generating ? 0.95 : 1.1
            let rotation = (phase == .previewLoading || phase == .inferring || phase == .generating)
                ? timeline.date.timeIntervalSinceReferenceDate / rotationSpeed * 360
                : 0

            ZStack {
                ringBackground(phase: phase, scale: scale)
                    .frame(width: 74 * scale, height: 74 * scale)
                    .rotationEffect(.degrees(rotation))
                    .shadow(color: OpenReshotPalette.twilightElectric.opacity(0.24), radius: 30, y: 10)

                Circle()
                    .fill(ringInnerFill(phase: phase))
                    .frame(width: 64 * scale, height: 64 * scale)

                if phase == .generating {
                    HStack(spacing: 4 * scale) {
                        ForEach(0..<3, id: \.self) { index in
                            Circle()
                                .fill(OpenReshotPalette.twilightText)
                                .frame(width: 4.5 * scale, height: 4.5 * scale)
                                .opacity(0.32 + 0.50 * max(0, sin(timeline.date.timeIntervalSinceReferenceDate * 4.8 + Double(index) * 0.72)))
                        }
                    }
                } else if phase == .compose {
                    Image(systemName: "camera.aperture")
                        .font(.system(size: 21 * scale, weight: .regular))
                        .symbolRenderingMode(.monochrome)
                        .foregroundStyle(OpenReshotPalette.twilightText.opacity(0.86))
                } else if phase == .failed {
                    Image(systemName: failedRetrySystemImage)
                        .font(.system(size: 21 * scale, weight: .semibold))
                        .symbolRenderingMode(.monochrome)
                        .foregroundStyle(OpenReshotPalette.twilightText.opacity(0.86))
                }
            }
            .frame(width: 82 * scale, height: 82 * scale)
        }
    }

    @ViewBuilder
    private func ringBackground(phase: ReshotFlowPhase, scale: CGFloat) -> some View {
        if phase == .generating {
            ZStack {
                Circle()
                    .stroke(OpenReshotPalette.twilightText.opacity(0.14), lineWidth: 5 * scale)

                Circle()
                    .trim(from: 0.08, to: 0.74)
                    .stroke(
                        AngularGradient(
                            colors: OpenReshotPalette.twilightRingGradient,
                            center: .center,
                            angle: .degrees(220)
                        ),
                        style: StrokeStyle(lineWidth: 5 * scale, lineCap: .round)
                    )
            }
        } else {
            Circle()
                .fill(
                    AngularGradient(
                        colors: OpenReshotPalette.twilightRingGradient,
                        center: .center,
                        angle: .degrees(220)
                    )
                )
        }
    }

    private func ringInnerFill(phase: ReshotFlowPhase) -> AnyShapeStyle {
        if phase == .compose || phase == .failed {
            return AnyShapeStyle(OpenReshotPalette.twilightBottom.opacity(0.94))
        }
        return AnyShapeStyle(OpenReshotPalette.twilightMid)
    }

    private func resultActionRow(scale: CGFloat) -> some View {
        ZStack {
            Button {
                withAnimation(.easeInOut(duration: 0.18)) {
                    if !comparingResult {
                        comparisonWipePosition = 0.5
                    }
                    comparingResult.toggle()
                }
            } label: {
                resultAuxiliaryButton(
                    systemImage: comparingResult ? "photo" : "rectangle.on.rectangle",
                    title: "对比",
                    highlighted: comparingResult,
                    scale: scale
                )
            }
            .buttonStyle(FluidPressButtonStyle(pressedScale: 0.96))
            .accessibilityLabel(comparingResult ? "显示成片" : "对比原图")
            .offset(x: resultMenuExpanded ? -88 * scale : 0, y: resultMenuExpanded ? 0 : -2 * scale)
            .scaleEffect(resultMenuExpanded ? 1 : 0.24)
            .opacity(resultMenuExpanded ? 1 : 0)
            .blur(radius: resultMenuExpanded ? 0 : 5)
            .allowsHitTesting(resultMenuExpanded)

            Button {
                app.saveResultImage()
            } label: {
                resultAuxiliaryButton(
                    systemImage: saveButtonIcon,
                    title: saveButtonText,
                    highlighted: app.saveState == .saving || app.saveState == .saved,
                    scale: scale
                )
            }
            .disabled(app.saveState == .saving)
            .buttonStyle(FluidPressButtonStyle(pressedScale: 0.96))
            .accessibilityLabel("保存生成照片")
            .offset(x: resultMenuExpanded ? 88 * scale : 0, y: resultMenuExpanded ? 0 : -2 * scale)
            .scaleEffect(resultMenuExpanded ? 1 : 0.24)
            .opacity(resultMenuExpanded ? 1 : 0)
            .blur(radius: resultMenuExpanded ? 0 : 5)
            .allowsHitTesting(resultMenuExpanded && app.saveState != .saving)

            Button {
                collapseResultMenuThenOpenPhotoPicker()
            } label: {
                emptyBottomAction(scale: scale)
            }
            .buttonStyle(FluidPressButtonStyle(pressedScale: 0.94))
            .accessibilityLabel("放入照片")
        }
        .frame(maxWidth: .infinity)
        .contentShape(Rectangle())
        .padding(.horizontal, 42 * scale)
    }

    private func resultAuxiliaryButton(systemImage: String, title: String, highlighted: Bool, scale: CGFloat) -> some View {
        VStack(spacing: 5 * scale) {
            ZStack {
                Circle()
                    .fill(highlighted ? OpenReshotPalette.twilightText.opacity(0.11) : OpenReshotPalette.twilightBottom.opacity(0.72))
                    .frame(width: 42 * scale, height: 42 * scale)
                    .overlay(
                        Circle()
                            .strokeBorder(
                                highlighted ? OpenReshotPalette.twilightAccent.opacity(0.86) : OpenReshotPalette.twilightText.opacity(0.30),
                                lineWidth: 1.2
                            )
                    )

                Image(systemName: systemImage)
                    .font(.system(size: 15 * scale, weight: .medium))
                    .symbolRenderingMode(.monochrome)
                    .foregroundStyle(highlighted ? OpenReshotPalette.twilightAccent.opacity(0.94) : OpenReshotPalette.twilightText.opacity(0.82))
            }
            .shadow(color: .black.opacity(0.30), radius: 14, y: 6)

            Text(title)
                .font(.system(size: 9 * scale, weight: .semibold, design: .rounded))
                .tracking(0.6)
                .foregroundStyle(highlighted ? OpenReshotPalette.twilightAccent.opacity(0.72) : OpenReshotPalette.twilightText.opacity(0.52))
        }
        .frame(width: 54 * scale, height: 66 * scale)
    }

    private var saveButtonIcon: String {
        switch app.saveState {
        case .saving: return "hourglass"
        case .saved: return "checkmark"
        case .failed: return "arrow.clockwise"
        case .idle: return "square.and.arrow.down"
        }
    }

    private var saveButtonText: String {
        switch app.saveState {
        case .saving: return "保存中"
        case .saved: return "已保存"
        case .failed: return "重试"
        case .idle: return "保存"
        }
    }

    private func saveToast(size: CGSize, scale: CGFloat) -> some View {
        HStack(spacing: 7 * scale) {
            Image(systemName: saveToastSystemImage)
                .font(.system(size: 12 * scale, weight: .bold))

            Text(saveToastMessage)
                .font(.system(size: 12 * scale, weight: .semibold, design: .rounded))
        }
        .foregroundStyle(OpenReshotPalette.twilightText.opacity(0.88))
        .padding(.horizontal, 14 * scale)
        .frame(height: 32 * scale)
        .background(OpenReshotPalette.twilightBottom.opacity(0.68), in: Capsule())
        .overlay(
            Capsule()
                .strokeBorder(OpenReshotPalette.twilightText.opacity(0.12), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.42), radius: 18, y: 8)
        .position(x: size.width / 2, y: size.height - 118 * scale)
    }

    private func photoStage(width: CGFloat, height: CGFloat) -> some View {
        let hintScale = min(max(width / 390, 0.82), 1.08)

        return ZStack {
            photoGlow
            subjectBackfill

            MetalView(app: app)
                .opacity(app.inputImage == nil ? 0 : 1)

            directionalDisocclusionOverlay
            sheenOverlay

            if let image = app.inputImage {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .opacity(app.rendererReady ? 0 : 1)
                    .animation(.easeInOut(duration: 0.8), value: app.rendererReady)
                    .allowsHitTesting(false)
            }

            if app.reconstructingScene || app.loadingPreviewScene {
                reconstructionOverlay
                    .transition(.opacity.animation(.easeInOut(duration: 0.28)))
            }

            if app.processFailed {
                failureFlash
                    .transition(.opacity.animation(.easeOut(duration: 0.22)))
            }

            if let resultImage = app.resultImage {
                resultOverlay(resultImage)
                    .transition(.asymmetric(
                        insertion: .opacity.combined(with: .scale(scale: 0.985)).animation(.spring(response: 0.38, dampingFraction: 0.90)),
                        removal: .opacity.animation(.easeOut(duration: 0.20))
                    ))
            }

            if resultFlash {
                Rectangle()
                    .fill(.white)
                    .transition(.opacity.animation(.easeOut(duration: 0.22)))
                    .allowsHitTesting(false)
            }

            if showDragHint, app.rendererReady, app.resultImage == nil, !app.reconstructingFrame {
                dragHintOverlay(scale: hintScale)
                    .transition(.opacity.animation(.easeOut(duration: 0.22)))
            }
        }
        .background(OpenReshotPalette.twilightBottom)
        .clipped()
        .contentShape(Rectangle())
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { value in
                    guard app.rendererReady, app.resultImage == nil, !app.reconstructingFrame else { return }
                    if showDragHint {
                        withAnimation(.easeOut(duration: 0.16)) {
                            showDragHint = false
                        }
                    }
                    if !draggingStage {
                        dragBaseTilt = app.motionTilt
                        draggingStage = true
                    }
                    let limit = photoTiltLimit
                    let tilt = SIMD2(
                        Self.clamp(dragBaseTilt.x + Float(value.translation.width / 128), -limit.x, limit.x),
                        Self.clamp(dragBaseTilt.y + Float(value.translation.height / 128), -limit.y, limit.y)
                    )
                    app.renderer?.setTiltTarget(tilt)
                    app.updateSheen(for: tilt)
                }
                .onEnded { _ in
                    draggingStage = false
                    dragBaseTilt = app.motionTilt
                }
        )
        .simultaneousGesture(
            TapGesture(count: 2)
                .onEnded {
                    guard app.rendererReady, app.resultImage == nil, !app.reconstructingFrame else { return }
                    app.renderer?.setTiltTarget(.zero)
                    app.updateSheen(for: .zero)
                }
        )
    }

    private static func clamp(_ value: Float, _ lower: Float, _ upper: Float) -> Float {
        min(max(value, lower), upper)
    }

    @ViewBuilder
    private var photoGlow: some View {
        if let image = app.inputImage {
            Image(uiImage: image)
                .resizable()
                .scaledToFill()
                .scaleEffect(1.30)
                .blur(radius: 42)
                .saturation(2.9)
                .brightness(0.05)
                .opacity(app.rendererReady ? 1.0 : 0.58)
                .blendMode(.normal)

            AngularGradient(colors: [
                Color(red: 1.0, green: 0.18, blue: 0.49),
                Color(red: 0.48, green: 0.36, blue: 1.0),
                Color(red: 0.0, green: 0.83, blue: 1.0),
                Color(red: 0.0, green: 1.0, blue: 0.64),
                Color(red: 1.0, green: 0.90, blue: 0.0),
                Color(red: 1.0, green: 0.37, blue: 0.0),
                Color(red: 1.0, green: 0.18, blue: 0.49)
            ], center: .center)
            .rotationEffect(.degrees(glowSpin ? 360 : 0))
            .scaleEffect(1.35)
            .blur(radius: 55)
            .opacity(app.rendererReady ? 0.90 : 0.68)
            .blendMode(.overlay)
            .animation(.linear(duration: 8).repeatForever(autoreverses: false), value: glowSpin)
                .allowsHitTesting(false)
        }
    }

    @ViewBuilder
    private var subjectBackfill: some View {
        if let image = app.inputImage, let mask = app.subjectProtectionMask {
            Image(uiImage: image)
                .resizable()
                .scaledToFill()
                .mask(
                    Image(uiImage: mask)
                        .resizable()
                        .scaledToFill()
                        .colorInvert()
                        .luminanceToAlpha()
                )
                .opacity(app.rendererReady ? 0.96 : 0)
                .allowsHitTesting(false)
        }
    }

    private var sheenOverlay: some View {
        AngularGradient(colors: [
            Color(red: 1.0, green: 0.0, blue: 0.83),
            Color(red: 0.0, green: 0.90, blue: 1.0),
            Color(red: 0.0, green: 1.0, blue: 0.52),
            Color(red: 1.0, green: 0.90, blue: 0.0),
            Color(red: 1.0, green: 0.18, blue: 0.18),
            Color(red: 1.0, green: 0.0, blue: 0.83)
        ], center: .center)
        .rotationEffect(.degrees(glowSpin ? 360 : 0))
        .blur(radius: 22)
        .opacity(app.rendererReady ? 0.10 + 0.55 * app.sheenAmount : 0)
        .blendMode(.screen)
        .mask(
            RadialGradient(stops: [
                .init(color: .clear, location: 0.0),
                .init(color: .clear, location: 0.56),
                .init(color: .white, location: 1.0)
            ], center: .center, startRadius: 0, endRadius: 520)
        )
        .mask(subjectProtectionMaskView)
        .animation(.linear(duration: 14).repeatForever(autoreverses: false), value: glowSpin)
        .allowsHitTesting(false)
    }

    private var directionalDisocclusionOverlay: some View {
        AngularGradient(colors: [
            Color(red: 1.0, green: 0.18, blue: 0.49),
            Color(red: 0.48, green: 0.36, blue: 1.0),
            Color(red: 0.0, green: 0.83, blue: 1.0),
            Color(red: 0.0, green: 1.0, blue: 0.64),
            Color(red: 1.0, green: 0.90, blue: 0.0),
            Color(red: 1.0, green: 0.37, blue: 0.0),
            Color(red: 1.0, green: 0.18, blue: 0.49)
        ], center: .center)
        .rotationEffect(.degrees(glowSpin ? 360 : 0))
        .scaleEffect(1.35)
        .blur(radius: 34)
        .opacity(app.rendererReady ? 0.10 + 0.34 * app.sheenAmount : 0)
        .blendMode(.screen)
        .mask(directionalDisocclusionMask)
        .mask(subjectProtectionMaskView)
        .animation(.linear(duration: 8).repeatForever(autoreverses: false), value: glowSpin)
        .allowsHitTesting(false)
    }

    private var directionalDisocclusionMask: some View {
        GeometryReader { proxy in
            let edgeRadius = max(proxy.size.width, proxy.size.height) * 0.72
            let horizontal = min(1, abs(app.sheenTiltX))
            let vertical = min(1, abs(app.sheenTiltY))
            let showLeading = app.sheenTiltX >= 0
            let showTop = app.sheenTiltY >= 0

            ZStack {
                RadialGradient(stops: [
                    .init(color: .clear, location: 0.0),
                    .init(color: .clear, location: 0.52),
                    .init(color: .white.opacity(0.50), location: 0.76),
                    .init(color: .white, location: 1.0)
                ], center: .center, startRadius: 0, endRadius: edgeRadius)

                LinearGradient(stops: [
                    .init(color: .white, location: 0.0),
                    .init(color: .white.opacity(0.86), location: 0.08),
                    .init(color: .white.opacity(0.42), location: 0.18),
                    .init(color: .clear, location: 0.34)
                ], startPoint: showLeading ? .leading : .trailing,
                   endPoint: showLeading ? .trailing : .leading)
                .opacity(0.12 + 0.88 * horizontal)

                LinearGradient(stops: [
                    .init(color: .white.opacity(0.66), location: 0.0),
                    .init(color: .white.opacity(0.34), location: 0.16),
                    .init(color: .clear, location: 0.30)
                ], startPoint: showTop ? .top : .bottom,
                   endPoint: showTop ? .bottom : .top)
                .opacity(0.08 + 0.44 * vertical)
            }
        }
    }

    @ViewBuilder
    private var subjectProtectionMaskView: some View {
        if let mask = app.subjectProtectionMask {
            Image(uiImage: mask)
                .resizable()
                .scaledToFill()
                .luminanceToAlpha()
        } else {
            Rectangle().fill(.white)
        }
    }

    private var reconstructionOverlay: some View {
        TimelineView(.animation) { timeline in
            let t = timeline.date.timeIntervalSinceReferenceDate
            let rotation = (t.truncatingRemainder(dividingBy: 6) / 6) * 360
            ZStack {
                if let image = app.capturedFrame ?? app.inputImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                        .saturation(1.8)
                        .brightness(0.05)
                        .contrast(1.05)
                        .hueRotation(.degrees(rotation))
                }

                AngularGradient(colors: [
                    Color(red: 1.0, green: 0.18, blue: 0.49),
                    Color(red: 0.48, green: 0.36, blue: 1.0),
                    Color(red: 0.0, green: 0.83, blue: 1.0),
                    Color(red: 0.0, green: 1.0, blue: 0.64),
                    Color(red: 1.0, green: 0.90, blue: 0.0),
                    Color(red: 1.0, green: 0.37, blue: 0.0),
                    Color(red: 1.0, green: 0.18, blue: 0.49)
                ], center: .center)
                .rotationEffect(.degrees(rotation))
                .scaleEffect(1.4)
                .blur(radius: 40)
                .opacity(0.50)
                .blendMode(.screen)
            }
        }
        .transition(.opacity)
        .allowsHitTesting(false)
    }

    private var twilightScanOverlay: some View {
        GeometryReader { proxy in
            TimelineView(.animation) { timeline in
                let cycle = timeline.date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: 1.7) / 1.7
                let y = proxy.size.height * (-0.45 + 2.90 * cycle)

                ZStack {
                    OpenReshotPalette.twilightBottom.opacity(0.35)

                    LinearGradient(
                        colors: [
                            OpenReshotPalette.twilightText.opacity(0),
                            OpenReshotPalette.twilightText.opacity(0.22),
                            OpenReshotPalette.twilightText.opacity(0)
                        ],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                    .frame(height: proxy.size.height * 0.45)
                    .offset(y: y)
                }
            }
        }
        .allowsHitTesting(false)
    }

    private var failureFlash: some View {
        Rectangle()
            .fill(.red.opacity(0.08))
            .overlay(Rectangle().stroke(.red.opacity(0.42), lineWidth: 2))
            .allowsHitTesting(false)
            .transition(.opacity)
    }

    private func resultOverlay(_ image: UIImage) -> some View {
        let beforeImage = app.inputImage ?? app.capturedFrame ?? image

        return GeometryReader { proxy in
            let width = max(proxy.size.width, 1)
            let wipeX = width * comparisonWipePosition

            ZStack {
                Image(uiImage: comparingResult ? beforeImage : image)
                    .resizable()
                    .scaledToFit()
                    .frame(width: proxy.size.width, height: proxy.size.height)

                if comparingResult {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(width: proxy.size.width, height: proxy.size.height)
                        .mask(alignment: .leading) {
                            Rectangle()
                                .frame(width: wipeX)
                        }

                    comparisonWipeChrome(x: wipeX, size: proxy.size)
                }
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        guard comparingResult else { return }
                        comparisonWipePosition = min(max(value.location.x / width, 0), 1)
                    }
            )
        }
        .animation(.easeInOut(duration: 0.18), value: comparingResult)
    }

    private func comparisonWipeChrome(x: CGFloat, size: CGSize) -> some View {
        let clampedX = min(max(x, 0), size.width)

        return ZStack(alignment: .topLeading) {
            VStack {
                HStack {
                    comparisonBadge("原图")
                    Spacer()
                    comparisonBadge("成片")
                }
                .padding(12)

                Spacer()
            }

            Rectangle()
                .fill(OpenReshotPalette.twilightText.opacity(0.88))
                .frame(width: 1.4, height: size.height)
                .offset(x: clampedX)

            ZStack {
                Circle()
                    .fill(OpenReshotPalette.twilightBottom.opacity(0.72))
                    .frame(width: 38, height: 38)
                    .overlay(
                        Circle()
                            .strokeBorder(OpenReshotPalette.twilightText.opacity(0.42), lineWidth: 1)
                    )

                Image(systemName: "arrow.left.and.right")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(OpenReshotPalette.twilightText.opacity(0.90))
            }
            .shadow(color: .black.opacity(0.34), radius: 12, y: 6)
            .offset(x: clampedX - 19, y: max(14, size.height * 0.50 - 19))
        }
        .allowsHitTesting(false)
    }

    private func comparisonBadge(_ title: String) -> some View {
        Text(title)
            .font(.system(size: 11, weight: .bold, design: .rounded))
            .tracking(1.6)
            .foregroundStyle(OpenReshotPalette.twilightText.opacity(0.88))
            .padding(.horizontal, 11)
            .padding(.vertical, 5)
            .background(OpenReshotPalette.twilightBottom.opacity(0.60), in: Capsule())
            .overlay(
                Capsule()
                    .strokeBorder(OpenReshotPalette.twilightText.opacity(0.10), lineWidth: 1)
            )
    }

    private var toolbar: some View {
        HStack(spacing: 34) {
            if app.resultImage != nil {
                Button {
                    app.saveResultImage()
                } label: {
                    saveButtonLabel
                }
                .disabled(app.saveState == .saving)
                .accessibilityLabel("保存生成照片")
                .buttonStyle(FluidPressButtonStyle())
            } else {
                PhotosPicker(selection: $pickerItem, matching: .images) {
                    plainUtilityIcon(systemName: "plus")
                }
                .accessibilityLabel("选择照片")
                .buttonStyle(FluidPressButtonStyle())
            }

            Button {
                app.reconstructCurrentFrame()
            } label: {
                Label("重构", systemImage: "sparkles")
                    .font(.system(size: 15, weight: .semibold, design: .rounded))
                    .foregroundStyle(OpenReshotPalette.accent)
                    .symbolRenderingMode(.hierarchical)
                    .frame(width: 102, height: 44)
                    .contentShape(Rectangle())
            }
            .disabled(!app.rendererReady || app.reconstructingFrame)
            .opacity((app.rendererReady && !app.reconstructingFrame) ? 1 : 0.45)
            .buttonStyle(FluidPressButtonStyle(pressedScale: 0.965))

            if app.resultImage != nil {
                Button {
                    app.closeResult()
                } label: {
                    plainUtilityIcon(systemName: "xmark")
                }
                .accessibilityLabel("关闭结果")
                .buttonStyle(FluidPressButtonStyle())
            } else {
                Button {
                    openSettings()
                } label: {
                    plainUtilityIcon(systemName: "gearshape")
                }
                .accessibilityLabel("设置")
                .buttonStyle(FluidPressButtonStyle())
            }
        }
        .frame(height: 58)
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 44)
        .padding(.top, 4)
        .padding(.bottom, 20)
        .opacity(app.inputImage == nil ? 0 : 1)
        .transition(.move(edge: .bottom).combined(with: .opacity))
    }

    @ViewBuilder
    private var saveButtonLabel: some View {
        switch app.saveState {
        case .saving:
            ProgressView()
                .tint(OpenReshotPalette.accent)
                .frame(width: 44, height: 44)
        case .saved:
            plainUtilityIcon(systemName: "checkmark", foreground: Color(red: 0.36, green: 0.56, blue: 0.36))
        case .failed:
            plainUtilityIcon(systemName: "exclamationmark", foreground: .red)
        case .idle:
            plainUtilityIcon(systemName: "square.and.arrow.down")
        }
    }

    private func plainUtilityIcon(systemName: String, foreground: Color = OpenReshotPalette.textSecondary) -> some View {
        Image(systemName: systemName)
            .font(.system(size: 19, weight: .medium))
            .symbolRenderingMode(.hierarchical)
            .foregroundStyle(foreground.opacity(0.90))
            .frame(width: 44, height: 44)
            .contentShape(Rectangle())
    }
}

private struct OpenReshotGlassPanelModifier: ViewModifier {
    let cornerRadius: CGFloat

    @ViewBuilder
    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
        if #available(iOS 26.0, *) {
            content
                .glassEffect(.regular.interactive(), in: shape)
                .overlay(
                    shape.strokeBorder(OpenReshotPalette.twilightText.opacity(0.11), lineWidth: 0.8)
                )
                .shadow(color: .black.opacity(0.20), radius: 22, y: 10)
        } else {
            content
                .background(.ultraThinMaterial, in: shape)
                .background(OpenReshotPalette.twilightInk.opacity(0.08), in: shape)
                .overlay(
                    shape.strokeBorder(OpenReshotPalette.twilightText.opacity(0.09), lineWidth: 0.8)
                )
                .shadow(color: .black.opacity(0.22), radius: 20, y: 9)
        }
    }
}

private extension View {
    func openReshotGlassPanel(cornerRadius: CGFloat) -> some View {
        modifier(OpenReshotGlassPanelModifier(cornerRadius: cornerRadius))
    }
}

private struct ReshotGalleryView: View {
    @ObservedObject var app: AppState
    @Environment(\.dismiss) private var dismiss
    let onSelect: (ReshotCacheItem) -> Void

    private let columns = [
        GridItem(.adaptive(minimum: 138, maximum: 190), spacing: 12)
    ]

    var body: some View {
        ZStack {
            SettingsSheetStyle.pageBackground.ignoresSafeArea()

            VStack(spacing: 0) {
                Color.clear
                    .frame(height: 42)

                VStack(spacing: 0) {
                    grabber
                    topBar

                    if app.galleryItems.isEmpty {
                        emptyState
                    } else {
                        ScrollView {
                            LazyVGrid(columns: columns, spacing: 12) {
                                ForEach(app.galleryItems) { item in
                                    Button {
                                        onSelect(item)
                                        dismiss()
                                    } label: {
                                        galleryCard(item)
                                    }
                                    .buttonStyle(FluidPressButtonStyle(pressedScale: 0.97))
                                    .contextMenu {
                                        if !ReshotCacheStore.isBundledItem(item) {
                                            Button(role: .destructive) {
                                                app.deleteCachedReshot(item)
                                            } label: {
                                                Label("删除", systemImage: "trash")
                                            }
                                        }
                                    }
                                }
                            }
                            .padding(.horizontal, 18)
                            .padding(.top, 18)
                            .padding(.bottom, 46)
                        }
                        .scrollIndicators(.hidden)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(SettingsSheetStyle.panelBackground, in: UnevenRoundedRectangle(topLeadingRadius: 34, topTrailingRadius: 34))
                .overlay(alignment: .top) {
                    UnevenRoundedRectangle(topLeadingRadius: 34, topTrailingRadius: 34)
                        .strokeBorder(SettingsSheetStyle.panelStroke, lineWidth: 1)
                }
                .shadow(color: .black.opacity(0.55), radius: 38, y: -12)
            }
        }
        .onAppear {
            app.refreshGallery()
        }
    }

    private var grabber: some View {
        HStack {
            Spacer()
            Capsule()
                .fill(SettingsSheetStyle.primaryText.opacity(0.20))
                .frame(width: 42, height: 5)
            Spacer()
        }
        .padding(.top, 10)
        .padding(.bottom, 4)
    }

    private var topBar: some View {
        HStack {
            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark")
            }
            .settingsChromeButton()
            .accessibilityLabel("关闭空间画廊")

            Spacer()

            VStack(alignment: .trailing, spacing: 2) {
                Text("空间画廊")
                    .font(.system(size: 18, weight: .semibold, design: .rounded))
                    .foregroundStyle(SettingsSheetStyle.primaryText)

                Text("\(app.galleryItems.count) 个空间")
                    .font(.system(size: 11, weight: .medium, design: .rounded))
                    .foregroundStyle(SettingsSheetStyle.tertiaryText)
            }
        }
        .padding(.horizontal, 24)
        .padding(.top, 8)
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Spacer(minLength: 0)

            Image(systemName: "photo.stack")
                .font(.system(size: 30, weight: .regular))
                .foregroundStyle(SettingsSheetStyle.secondaryText)
                .frame(width: 58, height: 58)
                .background(SettingsSheetStyle.iconFill, in: Circle())

            Text("还没有空间")
                .font(.system(size: 18, weight: .semibold, design: .rounded))
                .foregroundStyle(SettingsSheetStyle.primaryText)

            Text("构建过的空间会自动保存在这里")
                .font(.system(size: 13, weight: .regular, design: .rounded))
                .foregroundStyle(SettingsSheetStyle.secondaryText)

            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(.horizontal, 24)
    }

    private func galleryCard(_ item: ReshotCacheItem) -> some View {
        VStack(alignment: .leading, spacing: 9) {
            ZStack {
                if let url = ReshotCacheStore.thumbnailURL(for: item),
                   let image = UIImage(contentsOfFile: url.path) {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                } else {
                    Rectangle()
                        .fill(SettingsSheetStyle.controlFill)
                        .overlay {
                            Image(systemName: "photo")
                                .font(.system(size: 20, weight: .regular))
                                .foregroundStyle(SettingsSheetStyle.tertiaryText)
                        }
                }
            }
            .frame(height: 174)
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(SettingsSheetStyle.hairline, lineWidth: 1)
            )

            VStack(alignment: .leading, spacing: 3) {
                Text(ReshotCacheStore.isBundledItem(item) ? "默认预览" : Self.dateFormatter.string(from: item.createdAt))
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(SettingsSheetStyle.primaryText)
                    .lineLimit(1)

                Text("\(qualityLabel(item.quality)) · \(Self.countFormatter.string(from: NSNumber(value: item.splatCount)) ?? "\(item.splatCount)") 点")
                    .font(.system(size: 11, weight: .medium, design: .rounded))
                    .foregroundStyle(SettingsSheetStyle.secondaryText)
                    .lineLimit(1)
            }
            .padding(.horizontal, 2)
            .padding(.bottom, 3)
        }
        .padding(7)
        .background(SettingsSheetStyle.cardFill, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(SettingsSheetStyle.cardStroke, lineWidth: 1)
        )
    }

    private func qualityLabel(_ rawValue: String) -> String {
        switch RenderQuality(rawValue: rawValue) {
        case .some(.high):
            return "高清"
        case .some(.smooth):
            return "流畅"
        case .none:
            if rawValue == "demo" {
                return "内置示例"
            }
            return "缓存"
        }
    }

    private static let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "M月d日 HH:mm"
        return formatter
    }()

    private static let countFormatter: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        return formatter
    }()
}

struct SettingsView: View {
    @ObservedObject var app: AppState
    @ObservedObject private var modelStore: ReconstructionModelStore
    @Environment(\.dismiss) private var dismiss
    @State private var geminiKey: String
    @State private var showingGeminiHelp = false
    @State private var didApplyInitialFocus = false
    @State private var exportingPreview = false
    @State private var previewExportError: String?
    @State private var previewExportURLs: [URL] = []
    @State private var showingPreviewExportShare = false
    @FocusState private var geminiKeyFocused: Bool
    private let focusGeminiKeyOnAppear: Bool
    private static let geminiAPIKeyURL = URL(string: "https://aistudio.google.com/app/apikey")!

    init(app: AppState, focusGeminiKeyOnAppear: Bool = false) {
        self.app = app
        self.focusGeminiKeyOnAppear = focusGeminiKeyOnAppear
        _modelStore = ObservedObject(wrappedValue: app.modelStore)
        _geminiKey = State(initialValue: app.geminiKey)
    }

    var body: some View {
        ZStack {
            settingsBackdrop

            VStack(spacing: 0) {
                Color.clear
                    .frame(height: 42)

                VStack(spacing: 0) {
                    grabber
                    settingsTopBar

                    ScrollView {
                        VStack(alignment: .leading, spacing: 0) {
                            Text("设置")
                                .font(.system(size: 29, weight: .semibold, design: .rounded))
                                .tracking(0.2)
                                .foregroundStyle(SettingsSheetStyle.primaryText)
                                .padding(.horizontal, 2)
                                .padding(.top, 28)
                                .padding(.bottom, 34)

                            modelSection
                            qualitySection
                                .padding(.top, 34)
                            if shouldShowPreviewExportSection {
                                previewExportSection
                                    .padding(.top, 34)
                            }
                            geminiSection
                                .padding(.top, 34)

                            Spacer(minLength: 70)
                        }
                        .padding(.horizontal, 24)
                        .frame(maxWidth: .infinity, minHeight: 745, alignment: .topLeading)
                    }
                    .scrollIndicators(.hidden)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(SettingsSheetStyle.panelBackground, in: UnevenRoundedRectangle(topLeadingRadius: 34, topTrailingRadius: 34))
                .overlay(alignment: .top) {
                    UnevenRoundedRectangle(topLeadingRadius: 34, topTrailingRadius: 34)
                        .strokeBorder(SettingsSheetStyle.panelStroke, lineWidth: 1)
                }
                .shadow(color: .black.opacity(0.55), radius: 38, y: -12)
            }

            if showingGeminiHelp {
                GeminiAPIHelpOverlay(url: Self.geminiAPIKeyURL) {
                    withAnimation(.easeInOut(duration: 0.18)) {
                        showingGeminiHelp = false
                    }
                }
                .transition(.opacity.combined(with: .scale(scale: 0.985)))
                .zIndex(10)
            }
        }
        .onAppear {
            guard focusGeminiKeyOnAppear, !didApplyInitialFocus else { return }
            didApplyInitialFocus = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) {
                geminiKeyFocused = true
            }
        }
        .sheet(isPresented: $showingPreviewExportShare) {
            ShareSheet(activityItems: previewExportURLs)
        }
    }

    private var settingsBackdrop: some View {
        SettingsSheetStyle.pageBackground.ignoresSafeArea()
    }

    private var grabber: some View {
        HStack {
            Spacer()
            Capsule()
                .fill(SettingsSheetStyle.primaryText.opacity(0.20))
                .frame(width: 42, height: 5)
            Spacer()
        }
        .padding(.top, 10)
        .padding(.bottom, 4)
    }

    private var settingsTopBar: some View {
        HStack {
            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark")
            }
            .settingsChromeButton()
            .accessibilityLabel("取消")

            Spacer()

            Button {
                app.saveEnhanceSettings(key: geminiKey)
                dismiss()
            } label: {
                Label("保存", systemImage: "checkmark")
            }
            .buttonStyle(SettingsPrimaryCapsuleButtonStyle())
        }
        .padding(.horizontal, 24)
        .padding(.top, 8)
    }

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            sectionLabel("空间引擎")

            settingsCard {
                VStack(spacing: 0) {
                    HStack(spacing: 14) {
                        ZStack {
                            RoundedRectangle(cornerRadius: 13, style: .continuous)
                                .fill(SettingsSheetStyle.iconFill)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 13, style: .continuous)
                                        .strokeBorder(SettingsSheetStyle.cardStroke, lineWidth: 1)
                                )

                            Image(systemName: modelStateSymbol)
                                .font(.system(size: 16, weight: .semibold))
                                .symbolRenderingMode(.hierarchical)
                                .foregroundStyle(SettingsSheetStyle.secondaryText)
                        }
                        .frame(width: 38, height: 38)

                        VStack(alignment: .leading, spacing: 4) {
                            Text("空间模型")
                                .font(.system(size: 16, weight: .semibold, design: .rounded))
                                .foregroundStyle(SettingsSheetStyle.primaryText)

                            Text(modelStateCaption)
                                .font(.system(size: 12, weight: .regular, design: .rounded))
                                .foregroundStyle(SettingsSheetStyle.secondaryText)
                                .lineLimit(1)
                        }

                        Spacer()

                        modelStatusControl
                    }
                    .padding(.horizontal, 18)
                    .padding(.top, 18)
                    .padding(.bottom, shouldShowModelActivity ? 14 : 18)

                    if shouldShowModelActivity {
                        VStack(alignment: .leading, spacing: 8) {
                            if case .downloading = modelStore.installState,
                               let fraction = modelStore.progress.fractionCompleted {
                                ProgressView(value: fraction)
                                    .tint(SettingsSheetStyle.accent)
                            } else {
                                ProgressView()
                                    .tint(SettingsSheetStyle.accent)
                            }

                            Text(modelActivityText)
                                .font(.system(size: 12, weight: .regular))
                                .foregroundStyle(SettingsSheetStyle.tertiaryText)
                        }
                        .padding(.horizontal, 18)
                        .padding(.bottom, 16)
                    }

                    if case let .failed(message) = modelStore.installState {
                        divider
                        Label(message.isEmpty ? "下载失败" : message, systemImage: "exclamationmark.triangle")
                            .font(.system(size: 12, weight: .regular))
                            .foregroundStyle(Color.red.opacity(0.88))
                            .padding(.vertical, 14)
                            .padding(.horizontal, 18)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var modelStatusControl: some View {
        switch modelStore.installState {
        case .downloaded:
            Menu {
                Button {
                    modelStore.installModel {
                        app.modelInstallationDidFinish()
                    }
                } label: {
                    Label("重新下载", systemImage: "arrow.clockwise")
                }

                Button(role: .destructive) {
                    modelStore.removeDownloadedModel {
                        app.invalidateModelCache()
                    }
                } label: {
                    Label("删除模型", systemImage: "trash")
                }
            } label: {
                modelStatusCapsule(title: "已下载", systemImage: "checkmark")
            }

        case .bundled:
            modelStatusCapsule(title: "内置", systemImage: "shippingbox")

        case .checkingSource:
            modelStatusCapsule(title: "连接中", systemImage: "arrow.down.circle")

        case .downloading:
            Button {
                modelStore.cancelDownload()
            } label: {
                modelStatusCapsule(title: "取消", systemImage: "xmark")
            }
            .buttonStyle(.plain)

        case .failed:
            Button {
                modelStore.installModel {
                    app.modelInstallationDidFinish()
                }
            } label: {
                modelStatusCapsule(title: "重试", systemImage: "arrow.clockwise")
            }
            .buttonStyle(.plain)

        case .notInstalled:
            Button {
                modelStore.installModel {
                    app.modelInstallationDidFinish()
                }
            } label: {
                modelStatusCapsule(title: modelStore.hasResumeProgress ? "继续" : "下载", systemImage: "arrow.down")
            }
            .buttonStyle(.plain)
        }
    }

    private func modelStatusCapsule(title: String, systemImage: String) -> some View {
        Label(title, systemImage: systemImage)
            .font(.system(size: 11, weight: .semibold, design: .rounded))
            .labelStyle(.titleAndIcon)
            .foregroundStyle(modelStatusForeground)
            .padding(.horizontal, 11)
            .frame(height: 30)
            .background(modelStatusBackground, in: Capsule())
            .overlay(
                Capsule()
                    .strokeBorder(modelStatusForeground.opacity(0.22), lineWidth: 1)
            )
            .contentShape(Capsule())
    }

    private var qualitySection: some View {
        VStack(alignment: .leading, spacing: 14) {
            sectionLabel("画质")

            settingsCard {
                HStack(spacing: 6) {
                    ForEach(RenderQuality.allCases) { quality in
                        Button {
                            withAnimation(.spring(response: 0.28, dampingFraction: 0.82)) {
                                app.quality = quality
                            }
                        } label: {
                            HStack(spacing: 7) {
                                Image(systemName: quality.systemImage)
                                    .font(.system(size: 13, weight: .semibold))
                                    .symbolRenderingMode(.hierarchical)

                                Text(quality.title)
                                    .font(.system(size: 14, weight: app.quality == quality ? .semibold : .medium, design: .rounded))
                            }
                            .foregroundStyle(app.quality == quality ? SettingsSheetStyle.selectedText : SettingsSheetStyle.secondaryText)
                            .frame(maxWidth: .infinity, minHeight: 42)
                            .background(
                                Group {
                                    if app.quality == quality {
                                        Capsule()
                                            .fill(SettingsSheetStyle.accent)
                                    } else {
                                        Capsule()
                                            .fill(Color.clear)
                                    }
                                }
                            )
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(5)
                .background(SettingsSheetStyle.controlFill, in: Capsule())
                .overlay(
                    Capsule()
                        .strokeBorder(SettingsSheetStyle.hairline, lineWidth: 1)
                )
                .padding(12)
            }
        }
    }

    private var shouldShowPreviewExportSection: Bool {
        app.hasExportablePreview || exportingPreview || previewExportError != nil
    }

    private var previewExportSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            sectionLabel("3D 预览")

            settingsCard {
                VStack(spacing: 0) {
                    Button {
                        exportCurrentPreview()
                    } label: {
                        HStack(spacing: 12) {
                            Image(systemName: exportingPreview ? "hourglass" : "square.and.arrow.up")
                                .font(.system(size: 15, weight: .semibold))
                                .symbolRenderingMode(.hierarchical)
                                .frame(width: 20)

                            Text(exportingPreview ? "导出中" : "导出 3D 预览")
                                .font(.system(size: 15, weight: .semibold, design: .rounded))

                            Spacer()
                        }
                        .foregroundStyle(app.hasExportablePreview ? SettingsSheetStyle.primaryText : SettingsSheetStyle.tertiaryText)
                        .padding(.horizontal, 18)
                        .frame(height: 52)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .disabled(!app.hasExportablePreview || exportingPreview)

                    if let previewExportError {
                        divider
                        Label(previewExportError, systemImage: "exclamationmark.triangle")
                            .font(.system(size: 12, weight: .regular))
                            .foregroundStyle(Color.red.opacity(0.88))
                            .padding(.vertical, 14)
                            .padding(.horizontal, 18)
                    }
                }
            }
        }
    }

    private func exportCurrentPreview() {
        guard !exportingPreview else { return }
        exportingPreview = true
        previewExportError = nil
        Task { @MainActor in
            do {
                previewExportURLs = try await app.exportCurrentPreviewPackage()
                showingPreviewExportShare = true
            } catch {
                previewExportError = error.localizedDescription
            }
            exportingPreview = false
        }
    }

    private var geminiSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            sectionLabel("重构引擎")

            settingsCard {
                VStack(alignment: .leading, spacing: 12) {
                    HStack(spacing: 12) {
                        Image(systemName: "key.fill")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(SettingsSheetStyle.accent.opacity(0.78))
                            .frame(width: 18)

                        SecureField("Gemini API Key", text: $geminiKey)
                            .font(.system(size: 14, weight: .regular, design: .rounded))
                            .tracking(0.2)
                            .foregroundStyle(SettingsSheetStyle.primaryText)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .privacySensitive()
                            .tint(SettingsSheetStyle.accent)
                            .focused($geminiKeyFocused)
                    }
                    .padding(.horizontal, 16)
                    .frame(height: 48)
                    .background(SettingsSheetStyle.controlFill, in: RoundedRectangle(cornerRadius: 15, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 15, style: .continuous)
                            .strokeBorder(SettingsSheetStyle.primaryText.opacity(0.14), lineWidth: 1)
                    )
                    .contentShape(RoundedRectangle(cornerRadius: 15, style: .continuous))
                    .onTapGesture {
                        geminiKeyFocused = true
                    }

                    Button {
                        withAnimation(.easeInOut(duration: 0.18)) {
                            showingGeminiHelp = true
                        }
                    } label: {
                        Text("如何获取 API Key?")
                            .font(.system(size: 12, weight: .medium, design: .rounded))
                            .tracking(0.3)
                            .foregroundStyle(SettingsSheetStyle.accent)
                            .padding(.leading, 2)
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 18)
                .padding(.vertical, 18)
            }
        }
    }

    private func sectionLabel(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 10, weight: .semibold, design: .rounded))
            .tracking(3.5)
            .foregroundStyle(SettingsSheetStyle.primaryText.opacity(0.40))
            .padding(.horizontal, 4)
    }

    private func settingsCard<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        content()
            .frame(maxWidth: .infinity)
            .background(SettingsSheetStyle.cardFill, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .strokeBorder(SettingsSheetStyle.cardStroke, lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.12), radius: 10, y: 6)
    }

    private var divider: some View {
        Rectangle()
            .fill(SettingsSheetStyle.hairline)
            .frame(height: 0.5)
            .padding(.horizontal, 20)
    }

    private var modelStateSymbol: String {
        switch modelStore.installState {
        case .downloaded:
            return "checkmark.circle"
        case .bundled:
            return "shippingbox"
        case .checkingSource, .downloading:
            return "arrow.down.circle"
        case .failed:
            return "exclamationmark.triangle"
        case .notInstalled:
            return "circle"
        }
    }

    private var downloadProgressText: String {
        let progress = modelStore.progress
        guard progress.bytesReceived > 0 else {
            return "正在连接 \(modelActiveSourceLabel)"
        }
        let received = Self.byteFormatter.string(fromByteCount: progress.bytesReceived)
        if let total = progress.totalBytes {
            let totalText = Self.byteFormatter.string(fromByteCount: total)
            let percent = Int(((progress.fractionCompleted ?? 0) * 100).rounded(.down))
            if let speed = progress.bytesPerSecond {
                let speedText = Self.byteFormatter.string(fromByteCount: Int64(speed))
                return "\(percent)% · \(received) / \(totalText) · \(speedText)/s"
            }
            return "\(percent)% · \(received) / \(totalText)"
        }
        return "已下载 \(received)"
    }

    private var shouldShowModelActivity: Bool {
        switch modelStore.installState {
        case .checkingSource, .downloading:
            return true
        default:
            return false
        }
    }

    private var modelActivityText: String {
        switch modelStore.installState {
        case .checkingSource:
            return modelStore.hasResumeProgress
                ? "正在准备继续下载"
                : "正在连接 \(modelStore.primaryDownloadSourceLabel)"
        case .downloading:
            return downloadProgressText
        default:
            return modelStateCaption
        }
    }

    private var modelActiveSourceLabel: String {
        modelStore.progress.activeSourceLabel ?? modelStore.primaryDownloadSourceLabel
    }

    private var modelTransferVerb: String {
        modelStore.hasResumeProgress ? "继续" : "正在"
    }

    private var modelStateCaption: String {
        switch modelStore.installState {
        case .downloaded:
            return "本机模型已就绪"
        case .bundled:
            return "随 App 内置可用"
        case .checkingSource:
            return modelStore.hasResumeProgress
                ? "正在准备继续下载"
                : "正在连接 \(modelStore.primaryDownloadSourceLabel)"
        case .downloading:
            return "\(modelTransferVerb)从 \(modelActiveSourceLabel) 下载"
        case .failed:
            return "模型状态需要处理"
        case .notInstalled:
            return modelStore.hasResumeProgress ? "可继续上次下载" : "首次重构前需要下载"
        }
    }

    private var modelStateColor: Color {
        switch modelStore.installState {
        case .downloaded, .bundled:
            return SettingsSheetStyle.successText
        case .failed:
            return SettingsSheetStyle.destructiveText
        default:
            return SettingsSheetStyle.accent
        }
    }

    private var modelStatusForeground: Color {
        switch modelStore.installState {
        case .downloaded, .bundled:
            return modelStateColor
        case .failed:
            return SettingsSheetStyle.destructiveText
        default:
            return SettingsSheetStyle.accent
        }
    }

    private var modelStatusBackground: Color {
        switch modelStore.installState {
        case .downloaded, .bundled:
            return modelStateColor.opacity(0.12)
        case .failed:
            return SettingsSheetStyle.destructiveText.opacity(0.11)
        default:
            return SettingsSheetStyle.accentSoft
        }
    }

    private static let byteFormatter: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useMB, .useGB]
        formatter.countStyle = .file
        return formatter
    }()
}

private enum SettingsSheetStyle {
    static let pageBackground = Color(red: 4.0 / 255.0, green: 6.0 / 255.0, blue: 10.0 / 255.0)
    static let panelBackground = Color(red: 7.0 / 255.0, green: 10.0 / 255.0, blue: 16.0 / 255.0)
    static let panelStroke = OpenReshotPalette.twilightText.opacity(0.045)
    static let cardFill = Color(red: 27.0 / 255.0, green: 30.0 / 255.0, blue: 38.0 / 255.0)
    static let cardStroke = OpenReshotPalette.twilightText.opacity(0.065)
    static let controlFill = Color.black.opacity(0.22)
    static let iconFill = Color.black.opacity(0.18)
    static let accent = Color(red: 235.0 / 255.0, green: 164.0 / 255.0, blue: 99.0 / 255.0)
    static let accentSoft = accent.opacity(0.13)
    static let primaryText = OpenReshotPalette.twilightText
    static let secondaryText = OpenReshotPalette.twilightText.opacity(0.52)
    static let tertiaryText = OpenReshotPalette.twilightText.opacity(0.34)
    static let selectedText = OpenReshotPalette.twilightButtonText
    static let hairline = OpenReshotPalette.twilightText.opacity(0.075)
    static let successText = Color(red: 128.0 / 255.0, green: 178.0 / 255.0, blue: 137.0 / 255.0)
    static let destructiveText = Color(red: 232.0 / 255.0, green: 96.0 / 255.0, blue: 76.0 / 255.0)
}

private struct SettingsChromeButtonModifier: ViewModifier {
    func body(content: Content) -> some View {
        content
            .font(.system(size: 13, weight: .semibold, design: .rounded))
            .foregroundStyle(SettingsSheetStyle.primaryText.opacity(0.58))
            .frame(width: 38, height: 38)
            .background(SettingsSheetStyle.iconFill, in: Circle())
            .overlay(
                Circle()
                    .strokeBorder(SettingsSheetStyle.cardStroke, lineWidth: 1)
            )
            .contentShape(Rectangle())
    }
}

private extension View {
    func settingsChromeButton() -> some View {
        modifier(SettingsChromeButtonModifier())
    }
}

private struct SettingsPrimaryCapsuleButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .semibold, design: .rounded))
            .labelStyle(.titleAndIcon)
            .foregroundStyle(SettingsSheetStyle.selectedText)
            .padding(.horizontal, 16)
            .frame(height: 38)
            .background(SettingsSheetStyle.accent, in: Capsule())
            .shadow(color: SettingsSheetStyle.accent.opacity(0.16), radius: 12, y: 5)
            .scaleEffect(configuration.isPressed ? 0.96 : 1)
    }
}

private struct GeminiAPIHelpOverlay: View {
    let url: URL
    let onDismiss: () -> Void

    var body: some View {
        ZStack {
            Color.black.opacity(0.34)
                .ignoresSafeArea()
                .contentShape(Rectangle())
                .onTapGesture(perform: onDismiss)

            VStack(alignment: .leading, spacing: 18) {
                HStack(alignment: .top, spacing: 12) {
                    Text("如何获取 Gemini API?")
                        .font(.system(size: 19, weight: .semibold))
                        .foregroundStyle(SettingsSheetStyle.primaryText)

                    Spacer(minLength: 12)

                    Button(action: onDismiss) {
                        Image(systemName: "xmark")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(SettingsSheetStyle.secondaryText)
                            .frame(width: 28, height: 28)
                            .background(SettingsSheetStyle.controlFill, in: Circle())
                    }
                    .buttonStyle(.plain)
                }

                VStack(alignment: .leading, spacing: 10) {
                    helpStep("1", "登录 Google AI Studio。")
                    helpStep("2", "点击 Create API key。")
                    helpStep("3", "复制生成的 key，回到这里粘贴保存。")
                }

                Text("API Key 只保存在本机。不要发到公开仓库、截图或聊天里。")
                    .font(.system(size: 13, weight: .regular))
                    .foregroundStyle(SettingsSheetStyle.secondaryText)
                    .lineSpacing(3)
                    .fixedSize(horizontal: false, vertical: true)

                Link(destination: url) {
                    Label("打开 Google AI Studio", systemImage: "key")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(SettingsSheetStyle.primaryText)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 13)
                        .background(SettingsSheetStyle.accentSoft, in: Capsule())
                }
            }
            .padding(22)
            .frame(maxWidth: 330)
            .background(SettingsSheetStyle.cardFill, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .strokeBorder(SettingsSheetStyle.hairline, lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.22), radius: 24, y: 18)
            .padding(.horizontal, 28)
        }
    }

    private func helpStep(_ number: String, _ text: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Text(number)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(SettingsSheetStyle.primaryText)
                .frame(width: 20, height: 20)
                .background(SettingsSheetStyle.controlFill, in: Circle())

            Text(text)
                .font(.system(size: 14, weight: .regular))
                .foregroundStyle(SettingsSheetStyle.primaryText)
                .lineSpacing(3)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [URL]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

struct MetalView: UIViewRepresentable {
    let app: AppState

    func makeUIView(context: Context) -> MTKView {
        let v = MTKView()
        v.preferredFramesPerSecond = app.quality.preferredFrameRate
        v.isPaused = true
        v.enableSetNeedsDisplay = true
        v.contentScaleFactor = app.quality.renderScale(memoryGB: Self.memoryGB)
        if let r = ReshootRenderer(v) {
            app.attachRenderer(r)
            print("✅ [OpenReshot] MetalSplatter renderer ready")
        } else {
            print("❌ [OpenReshot] ReshootRenderer init failed (Metal unavailable)")
        }
        return v
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        uiView.preferredFramesPerSecond = app.quality.preferredFrameRate
        uiView.contentScaleFactor = app.quality.renderScale(memoryGB: Self.memoryGB)
        if app.rendererReady {
            uiView.setNeedsDisplay()
        }
    }

    private static var memoryGB: UInt64 { ProcessInfo.processInfo.physicalMemory / 1_073_741_824 }
}
