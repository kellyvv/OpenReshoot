import CoreML
import Darwin
import ImageIO
import UIKit

/// Raw network outputs (NDC-space gaussians) + the intrinsics needed to unproject.
struct SharpOutput {
    let mean: MLMultiArray      // [1, N, 3]  NDC
    let scale: MLMultiArray     // [1, N, 3]  sigma
    let quat: MLMultiArray      // [1, N, 4]  (w, x, y, z), unnormalized
    let color: MLMultiArray     // [1, N, 3]  linearRGB
    let opacity: MLMultiArray   // [1, N]     0..1
    let internalResolution: Int
    let fpx: Float              // focal length in px at the ORIGINAL image resolution
    let width: Int
    let height: Int
    var count: Int { mean.shape[1].intValue }
}

private struct SharpOutputBackings {
    let arraysByName: [String: MLMultiArray]
    let totalByteCount: Int

    var predictionDictionary: [String: Any] {
        Dictionary(uniqueKeysWithValues: arraysByName.map { ($0.key, $0.value as Any) })
    }

    func logUsage(in output: MLFeatureProvider) {
        let usage = arraysByName.keys.sorted().map { name in
            let expected = arraysByName[name]
            let actual = output.featureValue(for: name)?.multiArrayValue
            return "\(name)=\(actual === expected)"
        }
        print("🧪 [OpenReshot] Core ML outputBackings used: \(usage.joined(separator: ", "))")
    }
}

enum SharpComputeBackend: String {
    case cpuOnly
    case cpuAndGPU
    case cpuAndNeuralEngine
    case all

    var computeUnits: MLComputeUnits {
        switch self {
        case .cpuOnly:
            return .cpuOnly
        case .cpuAndGPU:
            return .cpuAndGPU
        case .cpuAndNeuralEngine:
            return .cpuAndNeuralEngine
        case .all:
            return .all
        }
    }

    static var preferred: SharpComputeBackend {
        if OpenReshotDebugFlags.allComputeUnitsInference {
            return .all
        }
        if OpenReshotDebugFlags.cpuAndNeuralEngineInference {
            return .cpuAndNeuralEngine
        }
        if OpenReshotDebugFlags.fastGPUInference {
            return .cpuAndGPU
        }
        if OpenReshotDebugFlags.cpuOnlyInference {
            return .cpuOnly
        }

        return .cpuOnly
    }
}

/// Loads the reconstruction model and runs the network.
final class SharpModel {
    static let defaultInternalResolution = 1536
    static let workingImageMaxLongEdge = 3072
    private static let outputNames = ["mean", "scale", "quat", "color", "opacity"]
    private let model: MLModel
    let internalResolution: Int
    let backend: SharpComputeBackend

    init(modelURL: URL? = nil, backend: SharpComputeBackend = .preferred) throws {
        let resolvedURL: URL
        if let modelURL {
            resolvedURL = modelURL
        } else {
            throw err("Reconstruction model is not installed. Download the model in Settings first.")
        }
        self.backend = backend
        let cfg = MLModelConfiguration()
        // .all makes Core ML try to place this 700M/1536² model on the Neural Engine,
        // whose compile/partition step hangs for a model this large. GPU is plenty.
        cfg.computeUnits = backend.computeUnits
        var optimizationHints = MLOptimizationHints()
        optimizationHints.reshapeFrequency = OpenReshotDebugFlags.frequentReshape ? .frequent : .infrequent
        cfg.optimizationHints = optimizationHints
        print("🧪 [OpenReshot] Core ML optimization hints: reshapeFrequency=\(OpenReshotDebugFlags.frequentReshape ? "frequent" : "infrequent")")
        print("⏳ [OpenReshot] MLModel(contentsOf:) loading \(resolvedURL.lastPathComponent) with \(backend.rawValue)…")
        model = try MLModel(contentsOf: resolvedURL, configuration: cfg)
        internalResolution = Self.modelInternalResolution(model)
        print("✅ [OpenReshot] MLModel ready (\(backend.rawValue), \(internalResolution)x\(internalResolution))")
        Self.logModelIdentity(model, url: resolvedURL)
    }

    func reconstruct(_ image: UIImage, sourceData: Data? = nil) throws -> SharpOutput {
        OpenReshotDiagnostics.logSourceData(sourceData, label: "reconstruct source")
        OpenReshotDiagnostics.logUIImage(image, label: "reconstruct input")
        let normalized = Self.normalized(image, maxLongEdge: Self.workingImageMaxLongEdge)
        let w = Int((normalized.size.width * normalized.scale).rounded())
        let h = Int((normalized.size.height * normalized.scale).rounded())
        OpenReshotDiagnostics.logUIImage(normalized, label: "reconstruct normalized")
        let focal35mmFromMetadata = Self.focalLength35mm(from: sourceData)
        let focal35mm = focal35mmFromMetadata ?? 30
        let fpx = focal35mm * (Float(w * w + h * h)).squareRoot() / Float(36 * 36 + 24 * 24).squareRoot()
        let disp = fpx / Float(w)
        print("📷 [OpenReshot] focal35mm=\(focal35mm), focalSource=\(focal35mmFromMetadata == nil ? "fallback" : "metadata"), fpx=\(fpx), disp=\(disp)")

        let imageArr = try Self.preprocess(normalized, side: internalResolution)
        OpenReshotDiagnostics.logMultiArray(imageArr, name: "input.image")
        let dispArr = try MLMultiArray(shape: [1], dataType: .float32)
        dispArr[0] = NSNumber(value: disp)
        OpenReshotDiagnostics.logMultiArray(dispArr, name: "input.disparity_factor")

        let input = try MLDictionaryFeatureProvider(dictionary: ["image": imageArr, "disparity_factor": dispArr])
        let options = MLPredictionOptions()
        let outputBackings = makePredictionOutputBackingsIfEnabled()
        if let outputBackings {
            options.outputBackings = outputBackings.predictionDictionary
        }
        let out = try OpenReshotMemoryProbe.measure("Core ML prediction") {
            try model.prediction(from: input, options: options)
        }
        outputBackings?.logUsage(in: out)

        func arr(_ name: String) throws -> MLMultiArray {
            guard let v = out.featureValue(for: name)?.multiArrayValue else { throw err("missing output \(name)") }
            return v
        }
        let mean = try arr("mean")
        let scale = try arr("scale")
        let quat = try arr("quat")
        let color = try arr("color")
        let opacity = try arr("opacity")
        OpenReshotDiagnostics.logMultiArray(mean, name: "output.mean")
        OpenReshotDiagnostics.logMultiArray(scale, name: "output.scale")
        OpenReshotDiagnostics.logMultiArray(quat, name: "output.quat")
        OpenReshotDiagnostics.logMultiArray(color, name: "output.color")
        OpenReshotDiagnostics.logMultiArray(opacity, name: "output.opacity")
        return SharpOutput(mean: mean, scale: scale, quat: quat,
                           color: color, opacity: opacity,
                           internalResolution: internalResolution,
                           fpx: fpx, width: w, height: h)
    }

    private func makePredictionOutputBackingsIfEnabled() -> SharpOutputBackings? {
        guard !OpenReshotDebugFlags.noOutputBackings else {
            print("🧪 [OpenReshot] Core ML outputBackings disabled")
            return nil
        }
        do {
            OpenReshotMemoryProbe.log("before output backings allocation")
            let backings = try makePredictionOutputBackings()
            print("🧪 [OpenReshot] prepared Core ML outputBackings (\(Self.formatBytes(backings.totalByteCount)))")
            OpenReshotMemoryProbe.log("after output backings allocation")
            return backings
        } catch {
            print("⚠️ [OpenReshot] Core ML outputBackings unavailable: \(error.localizedDescription)")
            return nil
        }
    }

    private func makePredictionOutputBackings() throws -> SharpOutputBackings {
        var arraysByName: [String: MLMultiArray] = [:]
        var totalByteCount = 0

        for name in Self.outputNames {
            guard let description = model.modelDescription.outputDescriptionsByName[name],
                  description.type == .multiArray,
                  let constraint = description.multiArrayConstraint else {
                throw err("output \(name) does not describe a multi-array")
            }
            let shape = constraint.shape.map(\.intValue)
            guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else {
                throw err("output \(name) has unsupported dynamic shape \(constraint.shape)")
            }
            let strides = Self.contiguousStrides(for: shape)
            let logicalByteCount = try Self.byteCount(shape: shape, dataType: constraint.dataType)
            let array = try Self.pageAlignedMultiArray(shape: shape,
                                                       dataType: constraint.dataType,
                                                       strides: strides,
                                                       byteCount: logicalByteCount)
            let featureValue = MLFeatureValue(multiArray: array)
            guard description.isAllowedValue(featureValue) else {
                throw err("output \(name) backing rejected by model description")
            }
            arraysByName[name] = array
            totalByteCount += logicalByteCount
        }

        return SharpOutputBackings(arraysByName: arraysByName, totalByteCount: totalByteCount)
    }

    private static func pageAlignedMultiArray(shape: [Int],
                                              dataType: MLMultiArrayDataType,
                                              strides: [Int],
                                              byteCount: Int) throws -> MLMultiArray {
        let pageSize = Int(getpagesize())
        let allocationByteCount = ((byteCount + pageSize - 1) / pageSize) * pageSize
        var pointer: UnsafeMutableRawPointer?
        let result = posix_memalign(&pointer, pageSize, allocationByteCount)
        guard result == 0, let pointer else {
            throw err("output backing allocation failed (\(result), \(formatBytes(byteCount)))")
        }
        let nsShape = shape.map(NSNumber.init(value:))
        let nsStrides = strides.map(NSNumber.init(value:))
        do {
            return try MLMultiArray(dataPointer: pointer,
                                    shape: nsShape,
                                    dataType: dataType,
                                    strides: nsStrides) { bytes in
                free(bytes)
            }
        } catch {
            free(pointer)
            throw error
        }
    }

    private static func contiguousStrides(for shape: [Int]) -> [Int] {
        guard !shape.isEmpty else { return [] }
        var strides = Array(repeating: 1, count: shape.count)
        if shape.count > 1 {
            for index in stride(from: shape.count - 2, through: 0, by: -1) {
                strides[index] = strides[index + 1] * shape[index + 1]
            }
        }
        return strides
    }

    private static func byteCount(shape: [Int], dataType: MLMultiArrayDataType) throws -> Int {
        let elementCount = shape.reduce(1, *)
        return elementCount * (try bytesPerElement(dataType))
    }

    private static func bytesPerElement(_ dataType: MLMultiArrayDataType) throws -> Int {
        let bits = dataType.rawValue & 0xff
        guard bits > 0, bits % 8 == 0 else {
            throw err("unsupported multi-array data type \(dataType.rawValue)")
        }
        return bits / 8
    }

    private static func formatBytes(_ bytes: Int) -> String {
        String(format: "%.1fMB", Double(bytes) / 1_048_576.0)
    }

    private static func logModelIdentity(_ model: MLModel, url: URL) {
        let path = url.path
        let source: String
        if path.contains("/Application Support/") {
            source = "downloaded"
        } else if path.contains(".app/") {
            source = "bundled"
        } else {
            source = "custom"
        }
        print("🧪 [OpenReshot] active reconstruction model source=\(source), path=\(path)")

        guard let creatorDefined = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] else {
            print("🧪 [OpenReshot] Core ML metadata: OpenReshotCoreMLMode=<missing>")
            return
        }
        let mode = creatorDefined["OpenReshotCoreMLMode"] ?? "<missing>"
        let attention = creatorDefined["OpenReshotAttention"] ?? "<missing>"
        let resolution = creatorDefined["OpenReshotInternalResolution"] ?? "<missing>"
        let dialect = creatorDefined["com.github.apple.coremltools.source_dialect"] ?? "<missing>"
        print("🧪 [OpenReshot] Core ML metadata: mode=\(mode), attention=\(attention), internalResolution=\(resolution), dialect=\(dialect)")
    }

    private static func modelInternalResolution(_ model: MLModel) -> Int {
        if let creatorDefined = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String],
           let rawResolution = creatorDefined["OpenReshotInternalResolution"],
           let resolution = Int(rawResolution),
           resolution > 0 {
            return resolution
        }

        if let shape = model.modelDescription.inputDescriptionsByName["image"]?.multiArrayConstraint?.shape,
           let resolution = shape.last?.intValue,
           resolution > 0 {
            return resolution
        }

        print("⚠️ [OpenReshot] Core ML internal resolution missing; assuming \(defaultInternalResolution)")
        return defaultInternalResolution
    }

    static func focalLength35mm(from data: Data?) -> Float? {
        guard let data,
              let source = CGImageSourceCreateWithData(data as CFData, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as NSDictionary? else {
            return nil
        }
        let exif = props[kCGImagePropertyExifDictionary] as? NSDictionary
        if let f35 = number(exif, keys: ["FocalLenIn35mmFilm", "FocalLengthIn35mmFilm"]), f35 > 0 {
            return f35
        }
        if let focal = number(exif, keys: ["FocalLength"]), focal > 0 {
            return focal < 10 ? focal * 8.4 : focal
        }
        return nil
    }

    private static func number(_ dictionary: NSDictionary?, keys: [String]) -> Float? {
        guard let dictionary else { return nil }
        for key in keys {
            if let value = dictionary[key] as? NSNumber {
                return value.floatValue
            }
            if let value = dictionary[key] as? String, let parsed = Float(value) {
                return parsed
            }
        }
        return nil
    }

    /// Resize to side×side, RGB float32 in [0,1], CHW order, as MLMultiArray [1,3,side,side].
    static func preprocess(_ image: UIImage, side: Int) throws -> MLMultiArray {
        guard let cg = image.cgImage else { throw err("UIImage has no cgImage") }
        var pixels = [UInt8](repeating: 0, count: side * side * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &pixels, width: side, height: side, bitsPerComponent: 8,
                                  bytesPerRow: side * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw err("CGContext failed")
        }
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: side, height: side))

        let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: side), NSNumber(value: side)], dataType: .float32)
        let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: arr.count)
        let plane = side * side
        pixels.withUnsafeBufferPointer { px in
            for y in 0..<side {
                for x in 0..<side {
                    let p = (y * side + x) * 4
                    let idx = y * side + x
                    ptr[idx] = Float(px[p + 0]) / 255.0
                    ptr[plane + idx] = Float(px[p + 1]) / 255.0
                    ptr[2 * plane + idx] = Float(px[p + 2]) / 255.0
                }
            }
        }
        return arr
    }

    static func downsampledImage(from data: Data, maxLongEdge: Int = workingImageMaxLongEdge) -> UIImage? {
        let sourceOptions: [CFString: Any] = [
            kCGImageSourceShouldCache: false,
            kCGImageSourceShouldCacheImmediately: false
        ]
        guard let source = CGImageSourceCreateWithData(data as CFData, sourceOptions as CFDictionary) else {
            return nil
        }
        let thumbnailOptions: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxLongEdge,
            kCGImageSourceShouldCacheImmediately: true
        ]
        guard let cgImage = CGImageSourceCreateThumbnailAtIndex(source, 0, thumbnailOptions as CFDictionary) else {
            return nil
        }
        return UIImage(cgImage: cgImage, scale: 1, orientation: .up)
    }

    /// Match the PC loader's EXIF auto-rotation before square resize, optionally bounding working bitmap size.
    static func normalized(_ image: UIImage, maxLongEdge: Int? = nil) -> UIImage {
        let pixelWidth = max(1, Int((image.size.width * image.scale).rounded()))
        let pixelHeight = max(1, Int((image.size.height * image.scale).rounded()))
        var targetWidth = pixelWidth
        var targetHeight = pixelHeight
        if let maxLongEdge, maxLongEdge > 0 {
            let longEdge = max(pixelWidth, pixelHeight)
            if longEdge > maxLongEdge {
                let scale = Double(maxLongEdge) / Double(longEdge)
                targetWidth = max(1, Int((Double(pixelWidth) * scale).rounded()))
                targetHeight = max(1, Int((Double(pixelHeight) * scale).rounded()))
            }
        }

        guard image.imageOrientation != .up || targetWidth != pixelWidth || targetHeight != pixelHeight else {
            return image
        }

        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let targetSize = CGSize(width: targetWidth, height: targetHeight)
        return UIGraphicsImageRenderer(size: targetSize, format: format).image { _ in
            image.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }

    static func pixelSizeDescription(_ image: UIImage) -> String {
        let width = Int((image.size.width * image.scale).rounded())
        let height = Int((image.size.height * image.scale).rounded())
        return "\(width)x\(height)"
    }
}

func err(_ msg: String) -> NSError {
    NSError(domain: "OpenReshot", code: -1, userInfo: [NSLocalizedDescriptionKey: msg])
}
