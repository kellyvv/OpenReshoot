import SwiftUI
import Photos
import PhotosUI
import MetalKit
import CoreImage
import CoreImage.CIFilterBuiltins
import ImageIO
import Vision
import simd

@main
struct OpenReshootApp: App {
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

enum SaveState {
    case idle
    case saving
    case saved
    case failed
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
    @Published var reconstructingScene = false
    @Published var reconstructingFrame = false
    @Published var processFailed = false
    @Published var capturedFrame: UIImage?
    @Published var resultImage: UIImage?
    @Published var saveState: SaveState = .idle
    @Published var subjectProtectionMask: UIImage?
    @Published var geminiKey: String
    var renderer: ReshootRenderer?
    private let modelQueue = DispatchQueue(label: "OpenReshoot.model", qos: .userInitiated)
    private var cachedModel: SharpModel?
    private var memoryWarningObserver: NSObjectProtocol?
    private var subjectMaskRequestID = UUID()
    private static let geminiInputMaxSide: CGFloat = 1024
    private static let geminiModel = "gemini-3.1-flash-image"
    private static var memoryGB: UInt64 { ProcessInfo.processInfo.physicalMemory / 1_073_741_824 }
    private static let enhancePrompt = """
    This is a novel-view render of a photo: some edges, subject details, and disoccluded areas may be blurry, warped, or missing. Restore only those rendering artifacts with realistic detail that is consistent with the original image. Preserve every person exactly as shown: keep the same identity, age, face, body, skin, hair, clothing, pose, expression, framing, and composition. Do not beautify, sexualize, age-change, body-change, or create any new person. Output one single complete clear photo. 修复画面中的模糊、扭曲和缺失细节，但人物必须保持原样。
    """

    init() {
        geminiKey = UserDefaults.standard.string(forKey: "OpenReshoot.geminiKey") ?? ""
        memoryWarningObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.modelQueue.async {
                self?.cachedModel = nil
                print("🧹 [OpenReshoot] released cached reconstruction model after memory warning")
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
        renderer.onReady = { [weak self] in
            self?.rendererReady = true
            self?.reconstructingScene = false
            self?.processFailed = false
            self?.status = ""
        }
    }

    func reconstruct(_ image: UIImage, sourceData: Data? = nil) {
        print("🔧 [OpenReshoot] reconstruct start: \(Int(image.size.width))x\(Int(image.size.height)) @\(image.scale)x")
        let displayImage = SharpModel.normalized(image)
        inputImage = displayImage
        imageAspect = max(0.1, displayImage.size.width / max(displayImage.size.height, 1))
        hasCloud = false
        rendererReady = false
        sheenAmount = 0
        sheenTiltX = 0
        sheenTiltY = 0
        motionTilt = .zero
        reconstructingScene = true
        reconstructingFrame = false
        processFailed = false
        capturedFrame = nil
        resultImage = nil
        saveState = .idle
        subjectProtectionMask = nil
        status = ""
        let maskRequestID = UUID()
        subjectMaskRequestID = maskRequestID
        updateSubjectProtectionMask(for: displayImage, requestID: maskRequestID)
        let selectedQuality = quality
        modelQueue.async { [weak self] in
            guard let self else { return }
            do {
                let model = try self.loadCachedModel()
                print("✅ [OpenReshoot] running inference (\(selectedQuality.title))…")
                let t0 = Date()
                let out = try model.reconstruct(displayImage, sourceData: sourceData)
                print("✅ [OpenReshoot] inference done in \(Int(-t0.timeIntervalSinceNow))s, \(out.count) gaussians")
                let (g, focus) = GaussianCloud.build(from: out, quality: selectedQuality)
                print("✅ [OpenReshoot] cloud built, focus=\(focus)")
                DispatchQueue.main.async {
                    if self.renderer == nil { print("❌ [OpenReshoot] renderer is nil (Metal init failed)") }
                    self.renderer?.setCloud(g, focus: focus,
                                            fpx: out.fpx, width: out.width, height: out.height)
                    self.hasCloud = true
                    self.status = ""
                }
            } catch {
                print("❌ [OpenReshoot] reconstruct error: \(error)")
                DispatchQueue.main.async {
                    self.reconstructingScene = false
                    self.processFailed = true
                    self.status = ""
                }
            }
        }
    }

    private func updateSubjectProtectionMask(for image: UIImage, requestID: UUID) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let mask = Self.makeSubjectProtectionMask(from: image)
            DispatchQueue.main.async { [weak self] in
                guard let self, self.subjectMaskRequestID == requestID else { return }
                self.subjectProtectionMask = mask
                print(mask == nil
                      ? "⚠️ [OpenReshoot] no foreground subject mask"
                      : "✅ [OpenReshoot] foreground subject mask ready")
            }
        }
    }

    private func loadCachedModel() throws -> SharpModel {
        if let cachedModel {
            print("♻️ [OpenReshoot] reusing cached reconstruction model")
            return cachedModel
        }
        print("⏳ [OpenReshoot] loading reconstruction model…")
        let model = try SharpModel()
        cachedModel = model
        return model
    }

    @MainActor
    func updateSheen(for tilt: SIMD2<Float>) {
        motionTilt = tilt
        sheenAmount = min(1, CGFloat(simd_length(tilt)))
        sheenTiltX = CGFloat(tilt.x)
        sheenTiltY = CGFloat(tilt.y)
    }

    @MainActor
    func reconstructCurrentFrame() {
        guard rendererReady, !reconstructingFrame else { return }
        let startedAt = Date()
        resultImage = nil
        saveState = .idle
        reconstructingFrame = true
        processFailed = false
        status = ""
        let rendered = renderer?.snapshotImage()
        guard let frame = Self.composeEnhanceFrame(rendered: rendered, source: inputImage),
              frame.size.width > 0,
              frame.size.height > 0 else {
            processFailed = true
            reconstructingFrame = false
            return
        }
        capturedFrame = frame
        print("⏱️ [OpenReshoot] enhance capture+compose \(Self.ms(since: startedAt))ms, frame \(Self.describe(frame))")
        let key = geminiKey
        Task { [weak self, frame, key] in
            guard let self else { return }
            do {
                let payload = try await Self.makeGeminiPayload(from: frame)
                let result = try await Self.requestGeminiEnhance(imagePNG: payload, key: key)
                resultImage = result
                saveState = .idle
                reconstructingFrame = false
            } catch {
                print("❌ [OpenReshoot] enhance error: \(error)")
                processFailed = true
                status = ""
                reconstructingFrame = false
            }
        }
    }

    @MainActor
    func closeResult() {
        resultImage = nil
        saveState = .idle
    }

    @MainActor
    func saveResultImage() {
        guard let resultImage, saveState != .saving else { return }
        saveState = .saving
        Task { [weak self] in
            do {
                try await Self.saveToPhotoLibrary(resultImage)
                await MainActor.run {
                    self?.saveState = .saved
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { [weak self] in
                        guard self?.saveState == .saved else { return }
                        self?.saveState = .idle
                    }
                }
            } catch {
                await MainActor.run {
                    print("❌ [OpenReshoot] save result error: \(error)")
                    self?.saveState = .failed
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.6) { [weak self] in
                        guard self?.saveState == .failed else { return }
                        self?.saveState = .idle
                    }
                }
            }
        }
    }

    func saveEnhanceSettings(key: String) {
        geminiKey = key.trimmingCharacters(in: .whitespacesAndNewlines)
        UserDefaults.standard.set(geminiKey, forKey: "OpenReshoot.geminiKey")
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
        print("⏱️ [OpenReshoot] Gemini upload JSON \(requestData.count / 1024)KB")
        let (data, response) = try await URLSession.shared.upload(for: request, from: requestData)
        print("⏱️ [OpenReshoot] Gemini response \(ms(since: requestStartedAt))ms, \(data.count / 1024)KB")
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
            print("⏱️ [OpenReshoot] enhance resize+png \(ms(since: encodeStartedAt))ms, input \(describe(input)), payload \(payload.count / 1024)KB")
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
            print("⚠️ [OpenReshoot] subject mask failed: \(error)")
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

private struct OpenReshootGlassCircle: ViewModifier {
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

private struct OpenReshootGlassCapsule: ViewModifier {
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
        modifier(OpenReshootGlassCircle(tint: tint, interactive: interactive))
    }

    func openReshootGlassCapsule(tint: Color? = nil, interactive: Bool = true) -> some View {
        modifier(OpenReshootGlassCapsule(tint: tint, interactive: interactive))
    }
}

private enum OpenReshootPalette {
    static let bg = Color(red: 248.0 / 255.0, green: 245.0 / 255.0, blue: 239.0 / 255.0)
    static let bgElevated = Color.white
    static let bgHover = Color(red: 234.0 / 255.0, green: 229.0 / 255.0, blue: 219.0 / 255.0)
    static let textPrimary = Color(red: 58.0 / 255.0, green: 52.0 / 255.0, blue: 46.0 / 255.0)
    static let textSecondary = Color(red: 112.0 / 255.0, green: 103.0 / 255.0, blue: 94.0 / 255.0)
    static let textTertiary = Color(red: 184.0 / 255.0, green: 173.0 / 255.0, blue: 160.0 / 255.0)
    static let accent = Color(red: 199.0 / 255.0, green: 122.0 / 255.0, blue: 63.0 / 255.0)
    static let accentMuted = Color(red: 195.0 / 255.0, green: 150.0 / 255.0, blue: 96.0 / 255.0)
    static let coolMist = Color(red: 138.0 / 255.0, green: 166.0 / 255.0, blue: 188.0 / 255.0)
    static let border = Color(red: 224.0 / 255.0, green: 222.0 / 255.0, blue: 215.0 / 255.0)
    static let borderSubtle = Color(red: 240.0 / 255.0, green: 235.0 / 255.0, blue: 226.0 / 255.0)
}

struct ContentView: View {
    @StateObject private var app = AppState()
    @State private var pickerItem: PhotosPickerItem?
    @State private var glowSpin = false
    @State private var showingEnhanceSettings = false
    private let toolbarHeight: CGFloat = 88

    var body: some View {
        ZStack {
            appBackdrop

            GeometryReader { geo in
                if app.inputImage == nil {
                    emptyHome(size: geo.size)
                } else {
                    let horizontalInset = min(max(geo.size.width * 0.03, 10), 22)
                    let stageTopInset = CGFloat(14)
                    let stageBottomGap = CGFloat(12)
                    let availableWidth = max(1, geo.size.width - horizontalInset * 2)
                    let availableHeight = max(1, geo.size.height - toolbarHeight - stageTopInset - stageBottomGap)
                    let aspect = max(0.1, app.imageAspect)
                    let stageWidth = min(availableWidth, availableHeight * aspect)
                    let stageHeight = stageWidth / aspect

                    VStack(spacing: 0) {
                        Spacer(minLength: stageTopInset)
                        photoStage(width: stageWidth, height: stageHeight)
                            .frame(width: stageWidth, height: stageHeight)
                            .shadow(color: .black.opacity(0.16), radius: 18, y: 10)
                        Spacer(minLength: stageBottomGap)
                        toolbar
                            .frame(height: toolbarHeight)
                    }
                    .frame(width: geo.size.width, height: geo.size.height)
                    .animation(.spring(response: 0.46, dampingFraction: 0.88), value: app.inputImage != nil)
                    .animation(.spring(response: 0.36, dampingFraction: 0.86), value: app.resultImage != nil)
                }
            }
        }
        .onAppear {
            glowSpin = true
            // Headless self-test: launch with `-autotest` to run the bundled image, no UI taps.
            if ProcessInfo.processInfo.arguments.contains("-autotest"),
               let url = Bundle.main.url(forResource: "koala", withExtension: "png"),
               let data = try? Data(contentsOf: url), let img = UIImage(data: data) {
                print("🧪 [OpenReshoot] autotest: reconstructing bundled koala.png")
                app.reconstruct(img, sourceData: data)
            }
        }
        .onChange(of: pickerItem) { _, item in
            guard let item else { return }
            print("📸 [OpenReshoot] photo picked")
            Task {
                do {
                    guard let data = try await item.loadTransferable(type: Data.self) else {
                        print("❌ [OpenReshoot] loadTransferable returned nil")
                        app.processFailed = true
                        return
                    }
                    print("✅ [OpenReshoot] loaded \(data.count) bytes")
                    guard let img = UIImage(data: data) else {
                        print("❌ [OpenReshoot] UIImage(data:) failed")
                        app.processFailed = true
                        return
                    }
                    app.reconstruct(img, sourceData: data)
                } catch {
                    print("❌ [OpenReshoot] load error: \(error)")
                    app.processFailed = true
                }
            }
        }
        .sheet(isPresented: $showingEnhanceSettings) {
            EnhanceSettingsView(app: app)
        }
        .tint(OpenReshootPalette.accent)
    }

    private var appBackdrop: some View {
        LinearGradient(
            colors: [
                OpenReshootPalette.bgElevated,
                OpenReshootPalette.bg,
                OpenReshootPalette.bgHover.opacity(0.72)
            ],
            startPoint: .top,
            endPoint: .bottom
        )
        .ignoresSafeArea()
    }

    private func emptyHome(size: CGSize) -> some View {
        let frameWidth = min(size.width - 44, size.height * 0.43)
        let frameHeight = min(size.height * 0.54, frameWidth * 1.42)
        let topInset = max(size.height * 0.11, 78)

        return ZStack {
            appBackdrop

            VStack(spacing: 26) {
                Text("OpenReshoot")
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(OpenReshootPalette.textSecondary.opacity(0.74))
                    .padding(.top, topInset)

                PhotosPicker(selection: $pickerItem, matching: .images) {
                    VStack(spacing: 22) {
                        ZStack {
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .fill(OpenReshootPalette.bgElevated.opacity(0.82))

                            LinearGradient(
                                colors: [
                                    OpenReshootPalette.accent.opacity(0.10),
                                    OpenReshootPalette.coolMist.opacity(0.12),
                                    OpenReshootPalette.bgElevated.opacity(0.0)
                                ],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))

                            Image(systemName: "photo")
                                .font(.system(size: 28, weight: .regular))
                                .symbolRenderingMode(.hierarchical)
                                .foregroundStyle(OpenReshootPalette.textTertiary.opacity(0.72))

                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .strokeBorder(OpenReshootPalette.border.opacity(0.82), lineWidth: 1)
                        }
                        .frame(width: frameWidth, height: frameHeight)
                        .shadow(color: .black.opacity(0.06), radius: 24, y: 12)

                        Label("选择照片", systemImage: "plus")
                            .font(.system(size: 15, weight: .semibold, design: .rounded))
                            .foregroundStyle(OpenReshootPalette.bg)
                            .symbolRenderingMode(.hierarchical)
                            .frame(width: 128, height: 44)
                            .background(OpenReshootPalette.textPrimary, in: Capsule())
                            .overlay(
                                Capsule()
                                    .strokeBorder(OpenReshootPalette.accentMuted.opacity(0.24), lineWidth: 1)
                            )
                    }
                }
                .buttonStyle(FluidPressButtonStyle(pressedScale: 0.985))

                Spacer(minLength: 26)
            }
            .frame(width: size.width, height: size.height)
        }
    }

    private func photoStage(width: CGFloat, height: CGFloat) -> some View {
        ZStack {
            photoGlow
            subjectBackfill

            MetalView(app: app)
                .opacity(app.inputImage == nil ? 0 : 1)

            directionalDisocclusionOverlay
            sheenOverlay

            if let image = app.inputImage {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
                    .opacity(app.rendererReady ? 0 : 1)
                    .animation(.easeInOut(duration: 0.8), value: app.rendererReady)
                    .allowsHitTesting(false)
            }

            if app.reconstructingScene || app.reconstructingFrame {
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
        }
        .background(Color.black)
        .clipped()
        .contentShape(Rectangle())
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { value in
                    guard app.rendererReady else { return }
                    let positionX = Float(value.location.x / max(width, 1))
                    let positionY = Float(value.location.y / max(height, 1))
                    let tilt = SIMD2(
                        Self.clamp(positionX * 2 - 1, -1, 1),
                        Self.clamp(positionY * 2 - 1, -1, 1)
                    )
                    app.renderer?.setTiltTarget(tilt)
                    app.updateSheen(for: tilt)
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

    private var failureFlash: some View {
        Rectangle()
            .fill(.red.opacity(0.08))
            .overlay(Rectangle().stroke(.red.opacity(0.42), lineWidth: 2))
            .allowsHitTesting(false)
            .transition(.opacity)
    }

    private func resultOverlay(_ image: UIImage) -> some View {
        ZStack {
            Color.black.opacity(0.72)

            Image(uiImage: image)
                .resizable()
                .scaledToFit()
                .padding(18)
                .shadow(color: .black.opacity(0.38), radius: 22, y: 12)
        }
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
                    .foregroundStyle(OpenReshootPalette.accent)
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
                    showingEnhanceSettings = true
                } label: {
                    plainUtilityIcon(systemName: "key")
                }
                .accessibilityLabel("重构设置")
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
                .tint(OpenReshootPalette.accent)
                .frame(width: 44, height: 44)
        case .saved:
            plainUtilityIcon(systemName: "checkmark", foreground: Color(red: 0.36, green: 0.56, blue: 0.36))
        case .failed:
            plainUtilityIcon(systemName: "exclamationmark", foreground: .red)
        case .idle:
            plainUtilityIcon(systemName: "square.and.arrow.down")
        }
    }

    private func plainUtilityIcon(systemName: String, foreground: Color = OpenReshootPalette.textSecondary) -> some View {
        Image(systemName: systemName)
            .font(.system(size: 19, weight: .medium))
            .symbolRenderingMode(.hierarchical)
            .foregroundStyle(foreground.opacity(0.90))
            .frame(width: 44, height: 44)
            .contentShape(Rectangle())
    }
}

struct EnhanceSettingsView: View {
    @ObservedObject var app: AppState
    @Environment(\.dismiss) private var dismiss
    @State private var geminiKey: String

    init(app: AppState) {
        self.app = app
        _geminiKey = State(initialValue: app.geminiKey)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker("质量", selection: $app.quality) {
                        ForEach(RenderQuality.allCases) { quality in
                            Label(quality.title, systemImage: quality.systemImage)
                                .tag(quality)
                        }
                    }
                    .pickerStyle(.segmented)
                }
                Section {
                    SecureField("Gemini API Key", text: $geminiKey)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                } header: {
                    Text("Gemini 重构")
                } footer: {
                    Text("当前使用 gemini-3.1-flash-image。API Key 只保存在本机。")
                }
                Section {
                    Button(role: .destructive) {
                        geminiKey = ""
                    } label: {
                        Label("清除", systemImage: "trash")
                    }
                }
            }
            .navigationTitle("重构设置")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("取消") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("保存") {
                        app.saveEnhanceSettings(key: geminiKey)
                        dismiss()
                    }
                }
            }
        }
    }
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
            print("✅ [OpenReshoot] MetalSplatter renderer ready")
        } else {
            print("❌ [OpenReshoot] ReshootRenderer init failed (Metal unavailable)")
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
