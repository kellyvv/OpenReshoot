import CoreML
import ImageIO
import UIKit

/// Raw network outputs (NDC-space gaussians) + the intrinsics needed to unproject.
struct SharpOutput {
    let mean: MLMultiArray      // [1, N, 3]  NDC
    let scale: MLMultiArray     // [1, N, 3]  sigma
    let quat: MLMultiArray      // [1, N, 4]  (w, x, y, z), unnormalized
    let color: MLMultiArray     // [1, N, 3]  linearRGB
    let opacity: MLMultiArray   // [1, N]     0..1
    let fpx: Float              // focal length in px at the ORIGINAL image resolution
    let width: Int
    let height: Int
    var count: Int { mean.shape[1].intValue }
}

/// Loads the reconstruction model and runs the network.
final class SharpModel {
    static let internalRes = 1536
    private let model: MLModel

    init(modelURL: URL? = nil) throws {
        let resolvedURL: URL
        if let modelURL {
            resolvedURL = modelURL
        } else if let bundledURL = Bundle.main.url(forResource: "SHARP", withExtension: "mlmodelc") {
            resolvedURL = bundledURL
        } else {
            throw err("Reconstruction model not in bundle — add SHARP.mlpackage to the OpenReshot target.")
        }
        let cfg = MLModelConfiguration()
        // .all makes Core ML try to place this 700M/1536² model on the Neural Engine,
        // whose compile/partition step hangs for a model this large. GPU is plenty.
        cfg.computeUnits = .cpuAndGPU
        print("⏳ [OpenReshot] MLModel(contentsOf:) loading \(resolvedURL.lastPathComponent)…")
        model = try MLModel(contentsOf: resolvedURL, configuration: cfg)
        print("✅ [OpenReshot] MLModel ready")
    }

    func reconstruct(_ image: UIImage, sourceData: Data? = nil) throws -> SharpOutput {
        let normalized = Self.normalized(image)
        let w = Int((normalized.size.width * normalized.scale).rounded())
        let h = Int((normalized.size.height * normalized.scale).rounded())
        let focal35mm = Self.focalLength35mm(from: sourceData) ?? 30
        let fpx = focal35mm * (Float(w * w + h * h)).squareRoot() / Float(36 * 36 + 24 * 24).squareRoot()
        let disp = fpx / Float(w)
        print("📷 [OpenReshot] focal35mm=\(focal35mm), fpx=\(fpx), disp=\(disp)")

        let imageArr = try Self.preprocess(normalized, side: Self.internalRes)
        let dispArr = try MLMultiArray(shape: [1], dataType: .float32)
        dispArr[0] = NSNumber(value: disp)

        let input = try MLDictionaryFeatureProvider(dictionary: ["image": imageArr, "disparity_factor": dispArr])
        let out = try model.prediction(from: input)

        func arr(_ name: String) throws -> MLMultiArray {
            guard let v = out.featureValue(for: name)?.multiArrayValue else { throw err("missing output \(name)") }
            return v
        }
        return SharpOutput(mean: try arr("mean"), scale: try arr("scale"), quat: try arr("quat"),
                           color: try arr("color"), opacity: try arr("opacity"),
                           fpx: fpx, width: w, height: h)
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

    /// Match the PC loader's EXIF auto-rotation before square resize.
    static func normalized(_ image: UIImage) -> UIImage {
        guard image.imageOrientation != .up else { return image }
        let format = UIGraphicsImageRendererFormat()
        format.scale = image.scale
        format.opaque = true
        return UIGraphicsImageRenderer(size: image.size, format: format).image { _ in
            image.draw(in: CGRect(origin: .zero, size: image.size))
        }
    }
}

func err(_ msg: String) -> NSError {
    NSError(domain: "OpenReshot", code: -1, userInfo: [NSLocalizedDescriptionKey: msg])
}
