import CoreML
import CryptoKit
import Foundation
import ImageIO
import Metal
import UIKit
import Darwin

enum OpenReshotDiagnostics {
    private static let arguments = Set(ProcessInfo.processInfo.arguments)
    static let enabled = arguments.contains("-openreshotDiagnostics")

    static func logRuntimeEnvironment() {
        guard enabled else { return }
        let device = UIDevice.current
        let memoryMB = ProcessInfo.processInfo.physicalMemory / 1_048_576
        print("🧪 [OpenReshot][Diag] runtime: device=\(device.model), machine=\(machineIdentifier()), os=\(device.systemName) \(device.systemVersion), physicalMemory=\(memoryMB)MB, processors=\(ProcessInfo.processInfo.processorCount), thermal=\(thermalStateDescription(ProcessInfo.processInfo.thermalState))")
    }

    static func logMetalDevice(_ device: MTLDevice) {
        guard enabled else { return }
        print("🧪 [OpenReshot][Diag] metal device: name=\(device.name), registryID=\(String(device.registryID, radix: 16)), unified=\(device.hasUnifiedMemory), maxThreads=\(device.maxThreadsPerThreadgroup.width)x\(device.maxThreadsPerThreadgroup.height)x\(device.maxThreadsPerThreadgroup.depth), maxThreadgroupMemory=\(device.maxThreadgroupMemoryLength)")
    }

    static func logMetalView(label: String, renderScale: CGFloat, frameRate: Int, view: UIView) {
        guard enabled else { return }
        let screen = view.window?.screen ?? UIScreen.main
        print("🧪 [OpenReshot][Diag] \(label): renderScale=\(format(Double(renderScale))), preferredFPS=\(frameRate), viewScale=\(format(Double(view.contentScaleFactor))), screenScale=\(format(Double(screen.scale))), nativeScale=\(format(Double(screen.nativeScale))), bounds=\(formatSize(view.bounds.size))")
    }

    static func logSourceData(_ data: Data?, label: String) {
        guard enabled else { return }
        guard let data else {
            print("🧪 [OpenReshot][Diag] \(label) data: <nil>")
            return
        }
        var parts = [
            "bytes=\(data.count)",
            "sha=\(shortSHA256(data))"
        ]
        if let source = CGImageSourceCreateWithData(data as CFData, nil),
           let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? NSDictionary {
            parts.append("cg=\(number(props, kCGImagePropertyPixelWidth))x\(number(props, kCGImagePropertyPixelHeight))")
            parts.append("orientation=\(number(props, kCGImagePropertyOrientation))")
            let exif = props[kCGImagePropertyExifDictionary] as? NSDictionary
            let tiff = props[kCGImagePropertyTIFFDictionary] as? NSDictionary
            parts.append("exifFocal35=\(number(exif, "FocalLenIn35mmFilm") ?? number(exif, "FocalLengthIn35mmFilm"))")
            parts.append("exifFocal=\(number(exif, "FocalLength"))")
            parts.append("camera=\(string(tiff, kCGImagePropertyTIFFModel))")
            parts.append("lens=\(string(exif, "LensModel"))")
        } else {
            parts.append("metadata=<unreadable>")
        }
        print("🧪 [OpenReshot][Diag] \(label) data: \(parts.joined(separator: ", "))")
    }

    static func logUIImage(_ image: UIImage, label: String) {
        guard enabled else { return }
        let pixelWidth = Int((image.size.width * image.scale).rounded())
        let pixelHeight = Int((image.size.height * image.scale).rounded())
        var parts = [
            "points=\(formatSize(image.size))",
            "pixels=\(pixelWidth)x\(pixelHeight)",
            "scale=\(format(Double(image.scale)))",
            "orientation=\(image.imageOrientation.rawValue)"
        ]
        if let cg = image.cgImage {
            parts.append("cg=\(cg.width)x\(cg.height)")
            parts.append("bits=\(cg.bitsPerComponent)/\(cg.bitsPerPixel)")
            parts.append("bytesPerRow=\(cg.bytesPerRow)")
            parts.append("alpha=\(cg.alphaInfo.rawValue)")
            parts.append("bitmap=0x\(String(cg.bitmapInfo.rawValue, radix: 16))")
            if let colorSpaceName = cg.colorSpace?.name {
                parts.append("colorSpace=\(colorSpaceName)")
            }
        } else {
            parts.append("cg=<nil>")
        }
        print("🧪 [OpenReshot][Diag] \(label) image: \(parts.joined(separator: ", "))")
    }

    static func logMultiArray(_ array: MLMultiArray, name: String) {
        guard enabled else { return }
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        let byteCount = array.count * bytesPerElement(array.dataType)
        let sha = hashBytes(start: array.dataPointer, byteCount: byteCount)
        let stats = sampledFloatStats(array)
        print("🧪 [OpenReshot][Diag] multiarray \(name): shape=\(shape), strides=\(strides), dtype=\(dataTypeName(array.dataType)), count=\(array.count), bytes=\(byteCount), sha=\(sha), \(stats)")
    }

    static func shortFileSHA256(_ url: URL) -> String {
        do {
            let data = try Data(contentsOf: url, options: [.mappedIfSafe])
            return shortSHA256(data)
        } catch {
            return "<hash-error:\(error.localizedDescription)>"
        }
    }

    static func fileByteCount(_ url: URL) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value) ?? -1
    }

    static func shortSHA256(_ data: Data, length: Int = 16) -> String {
        shortDigest(SHA256.hash(data: data), length: length)
    }

    static func format(_ value: Double) -> String {
        if !value.isFinite { return "\(value)" }
        let magnitude = abs(value)
        if magnitude > 0, magnitude < 0.001 || magnitude >= 10_000 {
            return String(format: "%.3e", value)
        }
        return String(format: "%.5f", value)
    }

    private static func sampledFloatStats(_ array: MLMultiArray) -> String {
        let samples = min(array.count, 8192)
        guard samples > 0 else { return "samples=0" }
        let stride = max(1, array.count / samples)
        var minValue = Float.greatestFiniteMagnitude
        var maxValue = -Float.greatestFiniteMagnitude
        var sum: Double = 0
        var finiteCount = 0
        var nonFiniteCount = 0
        var firstValue: Float?

        func record(_ value: Float) {
            if firstValue == nil { firstValue = value }
            guard value.isFinite else {
                nonFiniteCount += 1
                return
            }
            finiteCount += 1
            minValue = min(minValue, value)
            maxValue = max(maxValue, value)
            sum += Double(value)
        }

        switch array.dataType {
        case .float32:
            let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
            for sampleIndex in 0..<samples {
                record(pointer[min(array.count - 1, sampleIndex * stride)])
            }
        case .float16:
            let pointer = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
            for sampleIndex in 0..<samples {
                record(Float(pointer[min(array.count - 1, sampleIndex * stride)]))
            }
        default:
            return "samples=\(samples), stats=<unsupported>"
        }

        guard finiteCount > 0 else {
            return "samples=\(samples), finite=0, nonFinite=\(nonFiniteCount), first=\(format(Double(firstValue ?? .nan)))"
        }
        return "samples=\(samples), finite=\(finiteCount), nonFinite=\(nonFiniteCount), min=\(format(Double(minValue))), max=\(format(Double(maxValue))), mean=\(format(sum / Double(finiteCount))), first=\(format(Double(firstValue ?? .nan)))"
    }

    private static func bytesPerElement(_ dataType: MLMultiArrayDataType) -> Int {
        switch dataType {
        case .float16:
            return 2
        case .float32, .int32:
            return 4
        case .double:
            return 8
        default:
            let bits = dataType.rawValue & 0xff
            return bits > 0 ? max(1, bits / 8) : 1
        }
    }

    private static func dataTypeName(_ dataType: MLMultiArrayDataType) -> String {
        switch dataType {
        case .float16: return "float16"
        case .float32: return "float32"
        case .double: return "double"
        case .int32: return "int32"
        default: return "\(dataType.rawValue)"
        }
    }

    private static func hashBytes(start: UnsafeMutableRawPointer, byteCount: Int, length: Int = 16) -> String {
        guard byteCount > 0 else { return shortSHA256(Data(), length: length) }
        let buffer = UnsafeRawBufferPointer(start: start, count: byteCount)
        var hasher = SHA256()
        hasher.update(bufferPointer: buffer)
        return shortDigest(hasher.finalize(), length: length)
    }

    private static func shortDigest<D: Sequence>(_ digest: D, length: Int) -> String where D.Element == UInt8 {
        String(digest.map { String(format: "%02x", $0) }.joined().prefix(length))
    }

    private static func machineIdentifier() -> String {
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var buffer = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("hw.machine", &buffer, &size, nil, 0)
        return String(cString: buffer)
    }

    private static func thermalStateDescription(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private static func formatSize(_ size: CGSize) -> String {
        "\(format(Double(size.width)))x\(format(Double(size.height)))"
    }

    private static func number(_ dictionary: NSDictionary?, _ key: CFString) -> String {
        number(dictionary, key as String) ?? "<nil>"
    }

    private static func number(_ dictionary: NSDictionary?, _ key: String) -> String? {
        guard let value = dictionary?[key] else { return nil }
        if let number = value as? NSNumber { return format(number.doubleValue) }
        return "\(value)"
    }

    private static func string(_ dictionary: NSDictionary?, _ key: CFString) -> String {
        string(dictionary, key as String)
    }

    private static func string(_ dictionary: NSDictionary?, _ key: String) -> String {
        guard let value = dictionary?[key] else { return "<nil>" }
        return "\(value)"
    }
}
