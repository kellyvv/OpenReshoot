import AVFoundation
import CoreGraphics
import CoreMedia
import ImageIO
import UIKit
import UniformTypeIdentifiers

struct MotionExportPackage {
    let format: MotionExportFormat
    let videoURL: URL?
    let gifURL: URL?
    let livePhotoImageURL: URL?
    let livePhotoVideoURL: URL?
}

struct MotionFramePlan {
    let size: CGSize
    let fps: Int
    let frameCount: Int
    let tiltRange: Float

    var frameDuration: TimeInterval {
        1.0 / Double(fps)
    }

    func tilt(at index: Int) -> SIMD2<Float> {
        let progress = Float(index) / Float(max(frameCount - 1, 1))
        let x: Float
        if progress < 0.46 {
            x = interpolate(from: -tiltRange, to: tiltRange, progress: progress / 0.46)
        } else {
            x = interpolate(from: tiltRange, to: 0, progress: (progress - 0.46) / 0.54)
        }
        let y = sin(progress * .pi * 2) * tiltRange * 0.06
        return SIMD2(x, y)
    }

    private func interpolate(from start: Float, to end: Float, progress: Float) -> Float {
        let t = min(max(progress, 0), 1)
        let eased = t * t * (3 - 2 * t)
        return start + (end - start) * eased
    }
}

final class MotionVideoWriter: @unchecked Sendable {
    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private let metadataInput: AVAssetWriterInput?
    private let metadataAdaptor: AVAssetWriterInputMetadataAdaptor?
    private let width: Int
    private let height: Int
    private let fps: Int
    private var didAppendStillImageTime = false
    private let stillImageFrameIndex: Int?

    init(
        url: URL,
        fileType: AVFileType = .mp4,
        size: CGSize,
        fps: Int,
        contentIdentifier: String? = nil,
        stillImageFrameIndex: Int? = nil
    ) throws {
        self.width = max(2, Int(size.width.rounded(.toNearestOrAwayFromZero)))
        self.height = max(2, Int(size.height.rounded(.toNearestOrAwayFromZero)))
        self.fps = fps
        self.stillImageFrameIndex = stillImageFrameIndex

        try? FileManager.default.removeItem(at: url)
        writer = try AVAssetWriter(outputURL: url, fileType: fileType)

        let compression: [String: Any] = [
            AVVideoAverageBitRateKey: max(2_000_000, width * height * 7),
            AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel
        ]
        let outputSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: compression
        ]
        input = AVAssetWriterInput(mediaType: .video, outputSettings: outputSettings)
        input.expectsMediaDataInRealTime = false

        let sourceAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: sourceAttributes
        )

        guard writer.canAdd(input) else {
            throw NSError(domain: "OpenReshot", code: -2101, userInfo: [NSLocalizedDescriptionKey: "Cannot add video writer input"])
        }
        writer.add(input)

        if let contentIdentifier {
            writer.metadata = [Self.contentIdentifierMetadataItem(contentIdentifier)]
            let stillInput = AVAssetWriterInput(
                mediaType: .metadata,
                outputSettings: nil,
                sourceFormatHint: try Self.stillImageTimeFormatDescription()
            )
            stillInput.expectsMediaDataInRealTime = false
            if writer.canAdd(stillInput) {
                writer.add(stillInput)
                metadataInput = stillInput
                metadataAdaptor = AVAssetWriterInputMetadataAdaptor(assetWriterInput: stillInput)
            } else {
                throw NSError(domain: "OpenReshot", code: -2106, userInfo: [NSLocalizedDescriptionKey: "Cannot add Live Photo still-time metadata input"])
            }
        } else {
            metadataInput = nil
            metadataAdaptor = nil
        }

        guard writer.startWriting() else {
            throw writer.error ?? NSError(domain: "OpenReshot", code: -2102, userInfo: [NSLocalizedDescriptionKey: "Failed to start video writer"])
        }
        writer.startSession(atSourceTime: .zero)
    }

    func append(_ image: UIImage, frameIndex: Int) async throws {
        while !input.isReadyForMoreMediaData {
            try Task.checkCancellation()
            try await Task.sleep(nanoseconds: 3_000_000)
        }
        guard let pixelBuffer = makePixelBuffer(from: image) else {
            throw NSError(domain: "OpenReshot", code: -2103, userInfo: [NSLocalizedDescriptionKey: "Failed to create video pixel buffer"])
        }
        let time = CMTime(value: CMTimeValue(frameIndex), timescale: CMTimeScale(fps))
        guard adaptor.append(pixelBuffer, withPresentationTime: time) else {
            throw writer.error ?? NSError(domain: "OpenReshot", code: -2104, userInfo: [NSLocalizedDescriptionKey: "Failed to append video frame"])
        }
        if frameIndex == stillImageFrameIndex {
            try await appendStillImageTimeIfNeeded(frameIndex: frameIndex)
        }
    }

    func finish() async throws {
        if stillImageFrameIndex != nil, !didAppendStillImageTime {
            throw NSError(domain: "OpenReshot", code: -2107, userInfo: [NSLocalizedDescriptionKey: "Live Photo still-time metadata was not written"])
        }
        input.markAsFinished()
        metadataInput?.markAsFinished()
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            writer.finishWriting {
                if let error = self.writer.error {
                    continuation.resume(throwing: error)
                } else if self.writer.status == .completed {
                    continuation.resume()
                } else {
                    continuation.resume(throwing: NSError(domain: "OpenReshot", code: -2105, userInfo: [NSLocalizedDescriptionKey: "Video writer did not complete"]))
                }
            }
        }
    }

    private static func contentIdentifierMetadataItem(_ identifier: String) -> AVMetadataItem {
        let item = AVMutableMetadataItem()
        item.identifier = .quickTimeMetadataContentIdentifier
        item.value = identifier as NSString
        item.dataType = kCMMetadataBaseDataType_UTF8 as String
        return item.copy() as! AVMetadataItem
    }

    private static func stillImageTimeFormatDescription() throws -> CMFormatDescription {
        let specification: [String: Any] = [
            kCMMetadataFormatDescriptionMetadataSpecificationKey_Identifier as String: "mdta/com.apple.quicktime.still-image-time",
            kCMMetadataFormatDescriptionMetadataSpecificationKey_DataType as String: kCMMetadataBaseDataType_SInt8 as String
        ]
        var formatDescription: CMFormatDescription?
        let status = CMMetadataFormatDescriptionCreateWithMetadataSpecifications(
            allocator: kCFAllocatorDefault,
            metadataType: kCMMetadataFormatType_Boxed,
            metadataSpecifications: [specification] as CFArray,
            formatDescriptionOut: &formatDescription
        )
        guard status == noErr, let formatDescription else {
            throw NSError(domain: "OpenReshot", code: -2108, userInfo: [NSLocalizedDescriptionKey: "Failed to describe Live Photo still-time metadata"])
        }
        return formatDescription
    }

    private func appendStillImageTimeIfNeeded(frameIndex: Int) async throws {
        guard !didAppendStillImageTime else { return }
        guard let metadataAdaptor, let metadataInput else {
            throw NSError(domain: "OpenReshot", code: -2109, userInfo: [NSLocalizedDescriptionKey: "Live Photo still-time metadata input missing"])
        }
        while !metadataInput.isReadyForMoreMediaData {
            try Task.checkCancellation()
            try await Task.sleep(nanoseconds: 3_000_000)
        }
        didAppendStillImageTime = true
        let start = CMTime(value: CMTimeValue(frameIndex), timescale: CMTimeScale(max(fps, 1)))
        let item = AVMutableMetadataItem()
        item.keySpace = .quickTimeMetadata
        item.key = "com.apple.quicktime.still-image-time" as NSString
        item.value = 0 as NSNumber
        item.dataType = kCMMetadataBaseDataType_SInt8 as String
        let group = AVTimedMetadataGroup(
            items: [item],
            timeRange: CMTimeRange(start: start, duration: CMTime(value: 1, timescale: CMTimeScale(max(fps, 1))))
        )
        guard metadataAdaptor.append(group) else {
            didAppendStillImageTime = false
            throw writer.error ?? NSError(domain: "OpenReshot", code: -2110, userInfo: [NSLocalizedDescriptionKey: "Failed to append Live Photo still-time metadata"])
        }
    }

    private func makePixelBuffer(from image: UIImage) -> CVPixelBuffer? {
        guard let pool = adaptor.pixelBufferPool else { return nil }
        var pixelBuffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pixelBuffer) == kCVReturnSuccess,
              let pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)
            .union(.byteOrder32Little)
        guard let context = CGContext(data: baseAddress,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: bytesPerRow,
                                      space: colorSpace,
                                      bitmapInfo: bitmapInfo.rawValue) else {
            return nil
        }

        context.clear(CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPushContext(context)
        image.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPopContext()
        return pixelBuffer
    }
}

enum MotionGIFWriter {
    static func makeDestination(url: URL, frameCount: Int) throws -> CGImageDestination {
        try? FileManager.default.removeItem(at: url)
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL,
            UTType.gif.identifier as CFString,
            frameCount,
            nil
        ) else {
            throw NSError(domain: "OpenReshot", code: -2201, userInfo: [NSLocalizedDescriptionKey: "Failed to create GIF destination"])
        }
        let properties: [CFString: Any] = [
            kCGImagePropertyGIFDictionary: [
                kCGImagePropertyGIFLoopCount: 0
            ]
        ]
        CGImageDestinationSetProperties(destination, properties as CFDictionary)
        return destination
    }

    static func addFrame(_ image: UIImage, to destination: CGImageDestination, delay: TimeInterval) throws {
        guard let cgImage = image.cgImage else {
            throw NSError(domain: "OpenReshot", code: -2202, userInfo: [NSLocalizedDescriptionKey: "Failed to read GIF frame"])
        }
        let properties: [CFString: Any] = [
            kCGImagePropertyGIFDictionary: [
                kCGImagePropertyGIFDelayTime: delay
            ]
        ]
        CGImageDestinationAddImage(destination, cgImage, properties as CFDictionary)
    }

    static func finalize(_ destination: CGImageDestination) throws {
        guard CGImageDestinationFinalize(destination) else {
            throw NSError(domain: "OpenReshot", code: -2203, userInfo: [NSLocalizedDescriptionKey: "Failed to finalize GIF"])
        }
    }
}
