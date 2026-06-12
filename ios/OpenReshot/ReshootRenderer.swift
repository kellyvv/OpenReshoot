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

    private var tilt = SIMD2<Float>(0, 0)
    private var targetTilt = SIMD2<Float>(0, 0)
    private var fpx: Float = 1000, imgW: Float = 1, imgH: Float = 1, focus: Float = 1
    private var amplitude = SIMD2<Float>(0.06, 0.04)

    init?(_ v: MTKView) {
        guard let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else { return nil }
        device = dev; queue = q; view = v
        super.init()
        v.device = dev
        v.isOpaque = false
        v.backgroundColor = .clear
        v.layer.isOpaque = false
        v.colorPixelFormat = .bgra8Unorm_srgb
        v.preferredFramesPerSecond = 60
        // The viewer only needs back-to-front alpha blending. Disabling depth avoids
        // MetalSplatter's high-quality multi-stage depth path on device.
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
        cloudTask?.cancel()
        cloudTaskID = UUID()
        stopSmoothing()
        tilt = .zero
        targetTilt = .zero
        readyNotified = false
        splat = nil
        view?.setNeedsDisplay()
    }

    /// Build a MetalSplatter renderer from the reconstructed points (async: chunk + first sort).
    func setCloud(_ points: [SplatPoint], focus: Float, fpx: Float, width: Int, height: Int) {
        self.fpx = fpx; imgW = Float(width); imgH = Float(height); self.focus = focus
        amplitude = SIMD2(min(0.46, focus * 0.19), min(0.30, focus * 0.13))
        print("🎛️ [OpenReshot] motion amplitude x=\(amplitude.x), y=\(amplitude.y), focus=\(focus)")
        guard let view else { return }
        let cf = view.colorPixelFormat, df = view.depthStencilPixelFormat, sc = view.sampleCount
        cloudTask?.cancel()
        let taskID = UUID()
        cloudTaskID = taskID
        stopSmoothing()
        tilt = .zero
        targetTilt = .zero
        readyNotified = false
        splat = nil

        let metalDevice = device
        cloudTask = Task.detached(priority: .userInitiated) { [weak self, points] in
            do {
                let r = try SplatRenderer(device: metalDevice, colorFormat: cf, depthFormat: df,
                                          sampleCount: sc, maxViewCount: 1,
                                          maxSimultaneousRenders: Self.maxSimultaneousRenders,
                                          highQualityDepth: false)
                r.onSortComplete = { [weak self] _ in
                    Task { @MainActor in
                        guard let self else { return }
                        guard self.cloudTaskID == taskID else { return }
                        if self.displayLink == nil {
                            self.view?.setNeedsDisplay()
                        }
                        if !self.readyNotified {
                            self.readyNotified = true
                            self.onReady?()
                        }
                    }
                }
                let chunk = try SplatChunk(device: metalDevice, from: points)
                if Task.isCancelled { return }
                await r.addChunk(chunk)
                if Task.isCancelled { return }
                await MainActor.run { [weak self] in
                    guard let self, self.cloudTaskID == taskID else { return }
                    self.splat = r
                    self.view?.setNeedsDisplay()
                    print("✅ [OpenReshot] MetalSplatter ready, \(r.splatCount) splats")
                }
            } catch {
                await MainActor.run {
                    print("❌ [OpenReshot] MetalSplatter setup failed: \(error)")
                }
            }
        }
    }

    /// Build a MetalSplatter renderer from a bundled splat scene without running reconstruction.
    func setCloud(from url: URL, focus: Float, fpx: Float, width: Int, height: Int) {
        self.fpx = fpx; imgW = Float(width); imgH = Float(height); self.focus = focus
        amplitude = SIMD2(min(0.46, focus * 0.19), min(0.30, focus * 0.13))
        print("🎛️ [OpenReshot] demo motion amplitude x=\(amplitude.x), y=\(amplitude.y), focus=\(focus)")
        guard let view else { return }
        let cf = view.colorPixelFormat, df = view.depthStencilPixelFormat, sc = view.sampleCount
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
                let r = try SplatRenderer(device: metalDevice, colorFormat: cf, depthFormat: df,
                                          sampleCount: sc, maxViewCount: 1,
                                          maxSimultaneousRenders: Self.maxSimultaneousRenders,
                                          highQualityDepth: false)
                r.onSortComplete = { [weak self] _ in
                    Task { @MainActor in
                        guard let self else { return }
                        guard self.cloudTaskID == taskID else { return }
                        if self.displayLink == nil {
                            self.view?.setNeedsDisplay()
                        }
                        if !self.readyNotified {
                            self.readyNotified = true
                            self.onReady?()
                        }
                    }
                }
                let reader = try AutodetectSceneReader(url)
                let points = try await reader.readAll()
                if Task.isCancelled { return }
                let chunk = try SplatChunk(device: metalDevice, from: points)
                if Task.isCancelled { return }
                await r.addChunk(chunk)
                if Task.isCancelled { return }
                await MainActor.run { [weak self] in
                    guard let self, self.cloudTaskID == taskID else { return }
                    self.splat = r
                    self.view?.setNeedsDisplay()
                    print("✅ [OpenReshot] bundled demo ready, \(r.splatCount) splats")
                }
            } catch {
                await MainActor.run {
                    print("❌ [OpenReshot] bundled demo setup failed: \(error)")
                }
            }
        }
    }

    func setTiltTarget(_ newTilt: SIMD2<Float>) {
        targetTilt = newTilt
        startSmoothing()
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

    @objc private func stepPhotoMotion(_ link: CADisplayLink) {
        let delta = targetTilt - tilt
        tilt += delta * 0.15
        if simd_length(delta) < 0.002 {
            tilt = targetTilt
            stopSmoothing()
        }
        view?.setNeedsDisplay()
    }

    /// OpenCV-space parallax look-at, then flip into MetalSplatter's right-hand / Y-up space.
    private func viewMatrix() -> simd_float4x4 {
        let eye = SIMD3<Float>(amplitude.x * tilt.x, amplitude.y * tilt.y, 0)
        let up = SIMD3<Float>(0, -1, 0)
        let front = simd_normalize(SIMD3<Float>(0, 0, focus) - eye)
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
        let fovy = 2 * atan(imgH / (2 * fpx))
        let ys = 1 / tanf(fovy * 0.5)
        let xs = ys / aspect
        let near: Float = 0.02, far: Float = 100
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

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable else { return }
        guard let splat, splat.isReadyToRender else {
            clearDrawable(drawable, in: view)
            return
        }
        let w = view.drawableSize.width, h = view.drawableSize.height
        guard w > 0, h > 0 else { return }

        guard inFlight.wait(timeout: .now()) == .success else { return }
        guard let cmd = queue.makeCommandBuffer() else {
            inFlight.signal()
            return
        }
        cmd.addCompletedHandler { [inFlight] _ in inFlight.signal() }

        let vp = SplatRenderer.ViewportDescriptor(
            viewport: MTLViewport(originX: 0, originY: 0, width: Double(w), height: Double(h), znear: 0, zfar: 1),
            projectionMatrix: projection(aspect: Float(w / h)),
            viewMatrix: viewMatrix(),
            screenSize: SIMD2(Int(w), Int(h)))
        do {
            let ok = try splat.render(viewports: [vp],
                                      colorTexture: view.multisampleColorTexture ?? drawable.texture,
                                      colorStoreAction: view.multisampleColorTexture == nil ? .store : .multisampleResolve,
                                      depthTexture: view.depthStencilTexture,
                                      rasterizationRateMap: nil,
                                      renderTargetArrayLength: 0,
                                      accessTimeout: 0,
                                      sortTimeout: 0,
                                      to: cmd)
            if ok { cmd.present(drawable) }
        } catch {
            print("❌ [OpenReshot] render error: \(error)")
        }
        cmd.commit()
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
}
