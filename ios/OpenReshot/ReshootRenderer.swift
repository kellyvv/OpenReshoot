import Metal
import MetalKit
import QuartzCore
import UIKit
import simd
import MetalSplatter
import SplatIO

/// Drives MetalSplatter's renderer as a lightweight animated photo layer.
/// MetalSplatter does the heavy lifting: async GPU depth sort + tiled splat rasterization.
@MainActor
final class ReshootRenderer: NSObject, MTKViewDelegate {
    nonisolated private static let maxSimultaneousRenders = 3
    nonisolated private static let nearPlane: Float = 0.02
    nonisolated private static let farPlane: Float = 100
    nonisolated private static let splatDepthFormat: MTLPixelFormat = .depth32Float

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private weak var view: MTKView?
    private var splat: SplatRenderer?
    private var cloudTask: Task<Void, Never>?
    private var cloudTaskID = UUID()
    private var displayLink: CADisplayLink?
    private var readyNotified = false
    private let inFlight = DispatchSemaphore(value: ReshootRenderer.maxSimultaneousRenders)

    var onReady: (() -> Void)?
    var onFailure: ((String) -> Void)?

    private var tilt = SIMD2<Float>(0, 0)
    private var targetTilt = SIMD2<Float>(0, 0)
    private var fpx: Float = 1000, imgW: Float = 1, imgH: Float = 1, focus: Float = 1
    private var lensFocusDepth: Float = 1
    private var lensFNumber: Float = 16
    private var lensDolly: Float = 0
    private var motionRangeScale: Float = 1
    private var amplitude = SIMD2<Float>(0.06, 0.04)
    private var renderTargetSize = MTLSize(width: 0, height: 0, depth: 1)
    private var colorTarget: MTLTexture?
    private var depthTarget: MTLTexture?
    private var lensPipelineState: MTLRenderPipelineState?
    private var lensLibrary: MTLLibrary?

    private struct LensUniforms {
        var texelSize: SIMD2<Float>
        var focusDistance: Float
        var fNumber: Float
        var maxRadius: Float
        var nearPlane: Float
        var farPlane: Float
        var blurScale: Float
    }

    private final class SortCompletionSignal: @unchecked Sendable {
        private let lock = NSLock()
        private var continuation: CheckedContinuation<Void, Never>?
        private var didSignal = false

        func signal() {
            lock.lock()
            didSignal = true
            let pending = continuation
            continuation = nil
            lock.unlock()
            pending?.resume()
        }

        func wait() async {
            await withCheckedContinuation { continuation in
                lock.lock()
                if didSignal {
                    lock.unlock()
                    continuation.resume()
                } else {
                    self.continuation = continuation
                    lock.unlock()
                }
            }
        }
    }

    nonisolated private static func makeSingleChunk(
        from url: URL,
        expectedSplatCount: Int?,
        device: MTLDevice
    ) async throws -> (chunk: SplatChunk, splatCount: Int) {
        OpenReshotMemoryProbe.log("renderer makeSingleChunk start")
        let reader = try AutodetectSceneReader(url)
        let initialCapacity = max(expectedSplatCount ?? 1, 1)
        let splatBuffer = try MetalBuffer<EncodedSplatPoint>(device: device, capacity: initialCapacity)
        OpenReshotMemoryProbe.log("renderer makeSingleChunk buffer allocated")
        var splatCount = 0
        var sphericalHarmonicPoints: [SplatPoint]?

        for try await batch in try await reader.read() {
            if Task.isCancelled { throw CancellationError() }
            guard !batch.isEmpty else { continue }
            if sphericalHarmonicPoints != nil {
                sphericalHarmonicPoints?.append(contentsOf: batch)
                continue
            }
            let shDegree = batch.first?.color.shDegree ?? .sh0
            if shDegree > .sh0 {
                sphericalHarmonicPoints = []
                sphericalHarmonicPoints?.reserveCapacity(expectedSplatCount ?? batch.count)
                sphericalHarmonicPoints?.append(contentsOf: batch)
                continue
            }
            try splatBuffer.ensureCapacity(splatCount + batch.count)
            for point in batch {
                splatBuffer.values[splatCount] = EncodedSplatPoint(point)
                splatCount += 1
            }
        }

        if let sphericalHarmonicPoints {
            OpenReshotMemoryProbe.log("renderer makeSingleChunk SH fallback before chunk")
            return (try SplatChunk(device: device, from: sphericalHarmonicPoints), sphericalHarmonicPoints.count)
        }
        guard splatCount > 0 else {
            throw err("empty splat scene")
        }
        splatBuffer.count = splatCount
        OpenReshotMemoryProbe.log("renderer makeSingleChunk finished")
        return (SplatChunk(splats: splatBuffer, shDegree: .sh0), splatCount)
    }

    init?(_ v: MTKView) {
        guard let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else { return nil }
        device = dev; queue = q; view = v
        super.init()
        OpenReshotDiagnostics.logMetalDevice(dev)
        v.device = dev
        v.isOpaque = false
        v.backgroundColor = .clear
        v.layer.isOpaque = false
        v.colorPixelFormat = .bgra8Unorm_srgb
        v.preferredFramesPerSecond = 60
        // Depth is rendered into our own shader-readable texture for the lens pass.
        v.depthStencilPixelFormat = .invalid
        v.sampleCount = 1
        v.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        v.delegate = self
    }

    deinit {
        cloudTask?.cancel()
        displayLink?.invalidate()
    }

    func clearCloud() {
        OpenReshotMemoryProbe.log("renderer clearCloud start")
        cloudTask?.cancel()
        cloudTaskID = UUID()
        stopSmoothing()
        tilt = .zero
        targetTilt = .zero
        readyNotified = false
        splat = nil
        colorTarget = nil
        depthTarget = nil
        renderTargetSize = MTLSize(width: 0, height: 0, depth: 1)
        view?.setNeedsDisplay()
        OpenReshotMemoryProbe.log("renderer clearCloud end")
    }

    /// Build a MetalSplatter renderer from a PLY scene file without materializing all points at once.
    @discardableResult
    func setCloud(from url: URL, expectedSplatCount: Int? = nil, focus: Float, fpx: Float, width: Int, height: Int) -> Bool {
        self.fpx = fpx; imgW = Float(width); imgH = Float(height); self.focus = focus
        lensFocusDepth = focus
        lensFNumber = 16
        lensDolly = 0
        refreshMotionAmplitude()
        print("🎛️ [OpenReshot] scene motion amplitude x=\(amplitude.x), y=\(amplitude.y), focus=\(focus)")
        OpenReshotMemoryProbe.log("renderer setCloud start")
        guard let view else { return false }
        let cf = view.colorPixelFormat, sc = view.sampleCount
        cloudTask?.cancel()
        let taskID = UUID()
        cloudTaskID = taskID
        stopSmoothing()
        tilt = .zero
        targetTilt = .zero
        readyNotified = false
        splat = nil

        let metalDevice = device
        cloudTask = Task.detached(priority: .userInitiated) { [weak self, url] in
            do {
                OpenReshotMemoryProbe.log("renderer task before SplatRenderer init")
                let r = try SplatRenderer(device: metalDevice, colorFormat: cf, depthFormat: Self.splatDepthFormat,
                                          sampleCount: sc, maxViewCount: 1,
                                          maxSimultaneousRenders: Self.maxSimultaneousRenders,
                                          highQualityDepth: false)
                OpenReshotMemoryProbe.log("renderer task after SplatRenderer init")
                r.onSortComplete = { [weak self] _ in
                    Task { @MainActor in
                        guard let self else { return }
                        guard self.cloudTaskID == taskID else { return }
                        if self.displayLink == nil {
                            self.view?.setNeedsDisplay()
                        }
                    }
                }
                OpenReshotMemoryProbe.log("renderer task before chunk build")
                let chunkResult = try await Self.makeSingleChunk(
                    from: url,
                    expectedSplatCount: expectedSplatCount,
                    device: metalDevice
                )
                OpenReshotMemoryProbe.log("renderer task after chunk build")
                if Task.isCancelled { return }
                let sortSignal = SortCompletionSignal()
                r.afterNextSort {
                    sortSignal.signal()
                }

                OpenReshotMemoryProbe.log("renderer task before addChunk")
                let chunkID = await r.addChunk(chunkResult.chunk, enabled: false)
                OpenReshotMemoryProbe.log("renderer task after addChunk")
                await sortSignal.wait()
                OpenReshotMemoryProbe.log("renderer task after first sort")
                if Task.isCancelled { return }
                await r.setChunkEnabled(chunkID, enabled: true)
                OpenReshotMemoryProbe.log("renderer task after enable chunk")
                if Task.isCancelled { return }
                let finalLoadedCount = chunkResult.splatCount
                await MainActor.run { [weak self] in
                    guard let self, self.cloudTaskID == taskID else { return }
                    self.splat = r
                    self.view?.setNeedsDisplay()
                    if !self.readyNotified {
                        self.readyNotified = true
                        self.onReady?()
                    }
                    OpenReshotMemoryProbe.log("renderer ready assigned")
                    print("✅ [OpenReshot] splat scene ready, \(finalLoadedCount) splats in 1 chunk")
                }
            } catch is CancellationError {
                return
            } catch {
                await MainActor.run { [weak self] in
                    print("❌ [OpenReshot] bundled demo setup failed: \(error)")
                    guard let self, self.cloudTaskID == taskID else { return }
                    self.onFailure?(error.localizedDescription)
                }
            }
        }
        return true
    }

    func setLens(focusDepth: Float? = nil, fNumber: Float? = nil, dolly: Float? = nil) {
        if let focusDepth {
            lensFocusDepth = max(Self.nearPlane, focusDepth)
        }
        if let fNumber {
            lensFNumber = min(max(fNumber, 1.0), 22.0)
        }
        if let dolly {
            lensDolly = dolly
        }
        view?.setNeedsDisplay()
    }

    func setTiltTarget(_ newTilt: SIMD2<Float>) {
        targetTilt = newTilt
        startSmoothing()
    }

    func setMotionRangeScale(_ scale: Float) {
        motionRangeScale = min(max(scale, 0.2), 1.2)
        refreshMotionAmplitude()
        view?.setNeedsDisplay()
    }

    private func startSmoothing() {
        guard displayLink == nil else { return }
        let link = CADisplayLink(target: self, selector: #selector(stepPhotoMotion(_:)))
        link.preferredFrameRateRange = CAFrameRateRange(minimum: 30, maximum: 120, preferred: 60)
        link.add(to: .main, forMode: .common)
        displayLink = link
    }

    private func stopSmoothing() {
        displayLink?.invalidate()
        displayLink = nil
    }

    private func refreshMotionAmplitude() {
        amplitude = Self.motionAmplitude(for: focus, scale: motionRangeScale)
    }

    private static func motionAmplitude(for focus: Float, scale: Float) -> SIMD2<Float> {
        let amount = min(0.46, focus * 0.19) * scale
        return SIMD2(amount, amount)
    }

    @objc private func stepPhotoMotion(_ link: CADisplayLink) {
        let delta = targetTilt - tilt
        tilt += delta * 0.15
        if simd_length(delta) < 0.002 {
            tilt = targetTilt
            stopSmoothing()
        }
        view?.setNeedsDisplay()
    }

    private func cameraEye(for tilt: SIMD2<Float>) -> SIMD3<Float> {
        SIMD3<Float>(amplitude.x * tilt.x, amplitude.y * tilt.y, lensDolly)
    }

    private func focusTarget() -> SIMD3<Float> {
        SIMD3<Float>(0, 0, max(Self.nearPlane, lensFocusDepth))
    }

    private func currentFocusDistance(for tilt: SIMD2<Float>) -> Float {
        max(Self.nearPlane, simd_length(focusTarget() - cameraEye(for: tilt)))
    }

    /// OpenCV-space parallax look-at, then flip into MetalSplatter's right-hand / Y-up space.
    private func viewMatrix(for tilt: SIMD2<Float>) -> simd_float4x4 {
        let eye = cameraEye(for: tilt)
        let up = SIMD3<Float>(0, -1, 0)
        let front = simd_normalize(focusTarget() - eye)
        let right = simd_normalize(simd_cross(front, up))
        let down = simd_cross(front, right)
        let wcv = simd_float4x4(columns: (
            SIMD4(right.x, down.x, front.x, 0),
            SIMD4(right.y, down.y, front.y, 0),
            SIMD4(right.z, down.z, front.z, 0),
            SIMD4(-simd_dot(right, eye), -simd_dot(down, eye), -simd_dot(front, eye), 1)))
        let flip = simd_float4x4(diagonal: SIMD4(1, -1, -1, 1))   // OpenCV -> GL (right-hand, Y up)
        return flip * wcv
    }

    private func projection(aspect: Float) -> simd_float4x4 {
        let anchorDepth = max(Self.nearPlane, lensFocusDepth)
        let dollyCompensation = max(0.25, (anchorDepth - lensDolly) / max(anchorDepth, 1e-4))
        let effectiveFpx = max(1, fpx * dollyCompensation)
        let fovy = 2 * atan(imgH / (2 * effectiveFpx))
        let ys = 1 / tanf(fovy * 0.5)
        let xs = ys / aspect
        let near = Self.nearPlane, far = Self.farPlane
        let zs = far / (near - far)
        return simd_float4x4(columns: (
            SIMD4(xs, 0, 0, 0), SIMD4(0, ys, 0, 0), SIMD4(0, 0, zs, -1), SIMD4(0, 0, zs * near, 0)))
    }

    func snapshotImage() -> UIImage? {
        guard let view, view.bounds.width > 0, view.bounds.height > 0 else { return nil }
        view.draw()
        let format = UIGraphicsImageRendererFormat()
        format.scale = UIScreen.main.scale
        format.opaque = false
        return UIGraphicsImageRenderer(bounds: view.bounds, format: format).image { _ in
            view.drawHierarchy(in: view.bounds, afterScreenUpdates: true)
        }
    }

    func renderMotionFrame(tilt frameTilt: SIMD2<Float>, size: CGSize) throws -> UIImage {
        guard let splat, splat.isReadyToRender else {
            throw NSError(domain: "OpenReshot", code: -2001, userInfo: [NSLocalizedDescriptionKey: "Renderer is not ready"])
        }
        let width = max(2, Int(size.width.rounded(.toNearestOrAwayFromZero)))
        let height = max(2, Int(size.height.rounded(.toNearestOrAwayFromZero)))

        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm_srgb,
            width: width,
            height: height,
            mipmapped: false
        )
        colorDescriptor.usage = [.renderTarget, .shaderRead]
        colorDescriptor.storageMode = .private
        guard let colorTexture = device.makeTexture(descriptor: colorDescriptor) else {
            throw NSError(domain: "OpenReshot", code: -2002, userInfo: [NSLocalizedDescriptionKey: "Failed to create export color texture"])
        }

        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: Self.splatDepthFormat,
            width: width,
            height: height,
            mipmapped: false
        )
        depthDescriptor.usage = [.renderTarget, .shaderRead]
        depthDescriptor.storageMode = .private
        guard let depthTexture = device.makeTexture(descriptor: depthDescriptor) else {
            throw NSError(domain: "OpenReshot", code: -2003, userInfo: [NSLocalizedDescriptionKey: "Failed to create export depth texture"])
        }

        let outputDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm_srgb,
            width: width,
            height: height,
            mipmapped: false
        )
        outputDescriptor.usage = [.renderTarget, .shaderRead]
        outputDescriptor.storageMode = .private
        guard let outputTexture = device.makeTexture(descriptor: outputDescriptor) else {
            throw NSError(domain: "OpenReshot", code: -2004, userInfo: [NSLocalizedDescriptionKey: "Failed to create export output texture"])
        }

        let bytesPerRow = width * 4
        guard let readBuffer = device.makeBuffer(length: bytesPerRow * height, options: [.storageModeShared]) else {
            throw NSError(domain: "OpenReshot", code: -2005, userInfo: [NSLocalizedDescriptionKey: "Failed to create export read buffer"])
        }

        guard inFlight.wait(timeout: .now()) == .success else {
            throw NSError(domain: "OpenReshot", code: -2006, userInfo: [NSLocalizedDescriptionKey: "Renderer is busy"])
        }
        guard let cmd = queue.makeCommandBuffer() else {
            inFlight.signal()
            throw NSError(domain: "OpenReshot", code: -2007, userInfo: [NSLocalizedDescriptionKey: "Failed to create export command buffer"])
        }
        cmd.addCompletedHandler { [inFlight] _ in inFlight.signal() }

        let viewport = SplatRenderer.ViewportDescriptor(
            viewport: MTLViewport(originX: 0, originY: 0, width: Double(width), height: Double(height), znear: 0, zfar: 1),
            projectionMatrix: projection(aspect: Float(width) / Float(height)),
            viewMatrix: viewMatrix(for: frameTilt),
            screenSize: SIMD2(width, height)
        )
        let rendered = try splat.render(viewports: [viewport],
                                        colorTexture: colorTexture,
                                        colorStoreAction: .store,
                                        depthTexture: depthTexture,
                                        rasterizationRateMap: nil,
                                        renderTargetArrayLength: 0,
                                        accessTimeout: 0,
                                        sortTimeout: 0,
                                        to: cmd)
        guard rendered else {
            throw NSError(domain: "OpenReshot", code: -2008, userInfo: [NSLocalizedDescriptionKey: "Export render skipped"])
        }

        do {
            try encodeLensPass(colorTexture: colorTexture,
                               depthTexture: depthTexture,
                               destinationTexture: outputTexture,
                               focusDistance: currentFocusDistance(for: frameTilt),
                               commandBuffer: cmd)
        } catch {
            print("❌ [OpenReshot] export lens pass error: \(error)")
            copyColorTarget(colorTexture, to: outputTexture, commandBuffer: cmd)
        }

        guard let blit = cmd.makeBlitCommandEncoder() else {
            throw NSError(domain: "OpenReshot", code: -2009, userInfo: [NSLocalizedDescriptionKey: "Failed to create export blit encoder"])
        }
        blit.copy(from: outputTexture,
                  sourceSlice: 0,
                  sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: width, height: height, depth: 1),
                  to: readBuffer,
                  destinationOffset: 0,
                  destinationBytesPerRow: bytesPerRow,
                  destinationBytesPerImage: bytesPerRow * height)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        if let error = cmd.error {
            throw error
        }

        let data = Data(bytes: readBuffer.contents(), count: bytesPerRow * height)
        guard let provider = CGDataProvider(data: data as CFData) else {
            throw NSError(domain: "OpenReshot", code: -2010, userInfo: [NSLocalizedDescriptionKey: "Failed to create export data provider"])
        }
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)
            .union(.byteOrder32Little)
        guard let cgImage = CGImage(width: width,
                                    height: height,
                                    bitsPerComponent: 8,
                                    bitsPerPixel: 32,
                                    bytesPerRow: bytesPerRow,
                                    space: CGColorSpaceCreateDeviceRGB(),
                                    bitmapInfo: bitmapInfo,
                                    provider: provider,
                                    decode: nil,
                                    shouldInterpolate: false,
                                    intent: .defaultIntent) else {
            throw NSError(domain: "OpenReshot", code: -2011, userInfo: [NSLocalizedDescriptionKey: "Failed to create export image"])
        }
        return UIImage(cgImage: cgImage, scale: 1, orientation: .up)
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable else { return }
        guard let splat, splat.isReadyToRender else {
            clearDrawable(drawable, in: view)
            return
        }
        let w = view.drawableSize.width, h = view.drawableSize.height
        guard w > 0, h > 0 else { return }
        let targetWidth = max(1, Int(w.rounded(.up)))
        let targetHeight = max(1, Int(h.rounded(.up)))
        ensureRenderTargets(width: targetWidth, height: targetHeight, colorFormat: view.colorPixelFormat)
        guard let colorTarget, let depthTarget else {
            clearDrawable(drawable, in: view)
            return
        }

        guard inFlight.wait(timeout: .now()) == .success else { return }
        guard let cmd = queue.makeCommandBuffer() else {
            inFlight.signal()
            return
        }
        cmd.addCompletedHandler { [inFlight] _ in inFlight.signal() }

        let vp = SplatRenderer.ViewportDescriptor(
            viewport: MTLViewport(originX: 0, originY: 0, width: Double(w), height: Double(h), znear: 0, zfar: 1),
            projectionMatrix: projection(aspect: Float(w / h)),
            viewMatrix: viewMatrix(for: tilt),
            screenSize: SIMD2(Int(w), Int(h)))
        do {
            let ok = try splat.render(viewports: [vp],
                                      colorTexture: colorTarget,
                                      colorStoreAction: .store,
                                      depthTexture: depthTarget,
                                      rasterizationRateMap: nil,
                                      renderTargetArrayLength: 0,
                                      accessTimeout: 0,
                                      sortTimeout: 0,
                                      to: cmd)
            if ok {
                do {
                    try encodeLensPass(colorTexture: colorTarget,
                                       depthTexture: depthTarget,
                                       drawable: drawable,
                                       focusDistance: currentFocusDistance(for: tilt),
                                       commandBuffer: cmd)
                } catch {
                    print("❌ [OpenReshot] lens pass error: \(error)")
                    copyColorTarget(colorTarget, to: drawable.texture, commandBuffer: cmd)
                }
                cmd.present(drawable)
            }
        } catch {
            print("❌ [OpenReshot] render error: \(error)")
        }
        cmd.commit()
    }

    private func ensureRenderTargets(width: Int, height: Int, colorFormat: MTLPixelFormat) {
        guard renderTargetSize.width != width ||
              renderTargetSize.height != height ||
              colorTarget?.pixelFormat != colorFormat else { return }

        if let view {
            OpenReshotDiagnostics.logMetalView(label: "render target \(width)x\(height)", renderScale: CGFloat(view.contentScaleFactor), frameRate: view.preferredFramesPerSecond, view: view)
        }

        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: colorFormat,
            width: width,
            height: height,
            mipmapped: false
        )
        colorDescriptor.usage = [.renderTarget, .shaderRead]
        colorDescriptor.storageMode = .private
        let nextColorTarget = device.makeTexture(descriptor: colorDescriptor)
        nextColorTarget?.label = "OpenReshot lens color"

        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: Self.splatDepthFormat,
            width: width,
            height: height,
            mipmapped: false
        )
        depthDescriptor.usage = [.renderTarget, .shaderRead]
        depthDescriptor.storageMode = .private
        let nextDepthTarget = device.makeTexture(descriptor: depthDescriptor)
        nextDepthTarget?.label = "OpenReshot lens depth"

        colorTarget = nextColorTarget
        depthTarget = nextDepthTarget
        renderTargetSize = MTLSize(width: width, height: height, depth: 1)
    }

    private func encodeLensPass(
        colorTexture: MTLTexture,
        depthTexture: MTLTexture,
        drawable: CAMetalDrawable,
        focusDistance: Float,
        commandBuffer: MTLCommandBuffer
    ) throws {
        try buildLensPipelineIfNeeded(colorFormat: drawable.texture.pixelFormat)
        guard let lensPipelineState else {
            copyColorTarget(colorTexture, to: drawable.texture, commandBuffer: commandBuffer)
            return
        }

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = drawable.texture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = view?.clearColor ?? MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { return }
        encoder.label = "OpenReshot Lens DOF"
        encoder.setRenderPipelineState(lensPipelineState)
        encoder.setFragmentTexture(colorTexture, index: 0)
        encoder.setFragmentTexture(depthTexture, index: 1)
        var uniforms = LensUniforms(
            texelSize: SIMD2(1 / Float(max(colorTexture.width, 1)), 1 / Float(max(colorTexture.height, 1))),
            focusDistance: focusDistance,
            fNumber: lensFNumber,
            maxRadius: 28,
            nearPlane: Self.nearPlane,
            farPlane: Self.farPlane,
            blurScale: lensFNumber >= 15.9 ? 0 : 1
        )
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<LensUniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }

    private func encodeLensPass(
        colorTexture: MTLTexture,
        depthTexture: MTLTexture,
        destinationTexture: MTLTexture,
        focusDistance: Float,
        commandBuffer: MTLCommandBuffer
    ) throws {
        try buildLensPipelineIfNeeded(colorFormat: destinationTexture.pixelFormat)
        guard let lensPipelineState else {
            copyColorTarget(colorTexture, to: destinationTexture, commandBuffer: commandBuffer)
            return
        }

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = destinationTexture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { return }
        encoder.label = "OpenReshot Export Lens DOF"
        encoder.setRenderPipelineState(lensPipelineState)
        encoder.setFragmentTexture(colorTexture, index: 0)
        encoder.setFragmentTexture(depthTexture, index: 1)
        var uniforms = LensUniforms(
            texelSize: SIMD2(1 / Float(max(colorTexture.width, 1)), 1 / Float(max(colorTexture.height, 1))),
            focusDistance: focusDistance,
            fNumber: lensFNumber,
            maxRadius: 28,
            nearPlane: Self.nearPlane,
            farPlane: Self.farPlane,
            blurScale: lensFNumber >= 15.9 ? 0 : 1
        )
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<LensUniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }

    private func buildLensPipelineIfNeeded(colorFormat: MTLPixelFormat) throws {
        guard lensPipelineState == nil else { return }
        if lensLibrary == nil {
            lensLibrary = try device.makeLibrary(source: Self.lensShaderSource, options: nil)
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "OpenReshot Lens Pipeline"
        descriptor.vertexFunction = lensLibrary?.makeFunction(name: "lensVertex")
        descriptor.fragmentFunction = lensLibrary?.makeFunction(name: "lensDepthOfFieldFragment")
        descriptor.colorAttachments[0].pixelFormat = colorFormat
        lensPipelineState = try device.makeRenderPipelineState(descriptor: descriptor)
    }

    private func copyColorTarget(_ source: MTLTexture, to destination: MTLTexture, commandBuffer: MTLCommandBuffer) {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        let size = MTLSize(
            width: min(source.width, destination.width),
            height: min(source.height, destination.height),
            depth: 1
        )
        blit.copy(from: source,
                  sourceSlice: 0,
                  sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: size,
                  to: destination,
                  destinationSlice: 0,
                  destinationLevel: 0,
                  destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.endEncoding()
    }

    private func clearDrawable(_ drawable: CAMetalDrawable, in view: MTKView) {
        guard let cmd = queue.makeCommandBuffer() else { return }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = drawable.texture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = view.clearColor
        cmd.makeRenderCommandEncoder(descriptor: pass)?.endEncoding()
        cmd.present(drawable)
        cmd.commit()
    }

    private static let lensShaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct LensVertexOut {
        float4 position [[position]];
    };

    struct LensUniforms {
        float2 texelSize;
        float focusDistance;
        float fNumber;
        float maxRadius;
        float nearPlane;
        float farPlane;
        float blurScale;
    };

    vertex LensVertexOut lensVertex(uint vertexID [[vertex_id]]) {
        float2 positions[3] = {
            float2(-1.0, -1.0),
            float2( 3.0, -1.0),
            float2(-1.0,  3.0)
        };
        LensVertexOut out;
        out.position = float4(positions[vertexID], 0.0, 1.0);
        return out;
    }

    inline float linearDepth(float depth, constant LensUniforms &uniforms) {
        depth = clamp(depth, 0.0, 0.999999);
        float denom = max(uniforms.farPlane - depth * (uniforms.farPlane - uniforms.nearPlane), 1e-4);
        return (uniforms.nearPlane * uniforms.farPlane) / denom;
    }

    inline float blurRadius(float sceneDepth, constant LensUniforms &uniforms) {
        float focus = max(uniforms.focusDistance, uniforms.nearPlane);
        float depth = max(sceneDepth, uniforms.nearPlane);
        float coc = abs(depth - focus) / depth;
        float aperture = 2.8 / max(uniforms.fNumber, 1.0);
        return clamp(coc * uniforms.maxRadius * aperture, 0.0, uniforms.maxRadius) * uniforms.blurScale;
    }

    fragment half4 lensDepthOfFieldFragment(
        LensVertexOut in [[stage_in]],
        texture2d<half> colorTexture [[texture(0)]],
        depth2d<float> depthTexture [[texture(1)]],
        constant LensUniforms &uniforms [[buffer(0)]]
    ) {
        constexpr sampler linearSampler(coord::normalized, address::clamp_to_edge, filter::linear);
        constexpr sampler nearestSampler(coord::normalized, address::clamp_to_edge, filter::nearest);

        float2 uv = clamp(in.position.xy * uniforms.texelSize, float2(0.0), float2(1.0));
        half4 centerColor = colorTexture.sample(linearSampler, uv);
        if (centerColor.a <= half(0.001) || uniforms.blurScale <= 0.0) {
            return centerColor;
        }

        float centerDepth = linearDepth(depthTexture.sample(nearestSampler, uv), uniforms);
        float radius = blurRadius(centerDepth, uniforms);
        if (radius < 0.45) {
            return centerColor;
        }

        const float2 taps[16] = {
            float2( 1.000,  0.000), float2( 0.924,  0.383),
            float2( 0.707,  0.707), float2( 0.383,  0.924),
            float2( 0.000,  1.000), float2(-0.383,  0.924),
            float2(-0.707,  0.707), float2(-0.924,  0.383),
            float2(-1.000,  0.000), float2(-0.924, -0.383),
            float2(-0.707, -0.707), float2(-0.383, -0.924),
            float2( 0.000, -1.000), float2( 0.383, -0.924),
            float2( 0.707, -0.707), float2( 0.924, -0.383)
        };

        half4 accum = centerColor * half(2.0);
        float totalWeight = 2.0;
        for (uint i = 0; i < 16; ++i) {
            float ring = (i & 1) ? 0.62 : 1.0;
            float2 sampleUV = clamp(uv + taps[i] * radius * ring * uniforms.texelSize, float2(0.0), float2(1.0));
            half4 sampleColor = colorTexture.sample(linearSampler, sampleUV);
            float sampleDepth = linearDepth(depthTexture.sample(nearestSampler, sampleUV), uniforms);
            float sampleRadius = blurRadius(sampleDepth, uniforms);
            float weight = 1.0;
            if (sampleDepth + 0.02 < centerDepth && sampleRadius < radius) {
                weight = 0.35;
            }
            accum += sampleColor * half(weight);
            totalWeight += weight;
        }

        return accum / half(totalWeight);
    }
    """
}
