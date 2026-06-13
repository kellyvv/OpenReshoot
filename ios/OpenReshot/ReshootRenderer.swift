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
    }

    /// Build a MetalSplatter renderer from the reconstructed points (async: chunk + first sort).
    @discardableResult
    func setCloud(_ points: [SplatPoint], focus: Float, fpx: Float, width: Int, height: Int) -> Bool {
        self.fpx = fpx; imgW = Float(width); imgH = Float(height); self.focus = focus
        lensFocusDepth = focus
        lensFNumber = 16
        lensDolly = 0
        amplitude = SIMD2(min(0.46, focus * 0.19), min(0.30, focus * 0.13))
        print("🎛️ [OpenReshot] motion amplitude x=\(amplitude.x), y=\(amplitude.y), focus=\(focus)")
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
        cloudTask = Task.detached(priority: .userInitiated) { [weak self, points] in
            do {
                let r = try SplatRenderer(device: metalDevice, colorFormat: cf, depthFormat: Self.splatDepthFormat,
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
                await MainActor.run { [weak self] in
                    print("❌ [OpenReshot] MetalSplatter setup failed: \(error)")
                    guard let self, self.cloudTaskID == taskID else { return }
                    self.onFailure?(error.localizedDescription)
                }
            }
        }
        return true
    }

    /// Build a MetalSplatter renderer from a bundled splat scene without running reconstruction.
    @discardableResult
    func setCloud(from url: URL, focus: Float, fpx: Float, width: Int, height: Int) -> Bool {
        self.fpx = fpx; imgW = Float(width); imgH = Float(height); self.focus = focus
        lensFocusDepth = focus
        lensFNumber = 16
        lensDolly = 0
        amplitude = SIMD2(min(0.46, focus * 0.19), min(0.30, focus * 0.13))
        print("🎛️ [OpenReshot] demo motion amplitude x=\(amplitude.x), y=\(amplitude.y), focus=\(focus)")
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
                let r = try SplatRenderer(device: metalDevice, colorFormat: cf, depthFormat: Self.splatDepthFormat,
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

    private func cameraEye() -> SIMD3<Float> {
        SIMD3<Float>(amplitude.x * tilt.x, amplitude.y * tilt.y, lensDolly)
    }

    private func focusTarget() -> SIMD3<Float> {
        SIMD3<Float>(0, 0, max(Self.nearPlane, lensFocusDepth))
    }

    private func currentFocusDistance() -> Float {
        max(Self.nearPlane, simd_length(focusTarget() - cameraEye()))
    }

    /// OpenCV-space parallax look-at, then flip into MetalSplatter's right-hand / Y-up space.
    private func viewMatrix() -> simd_float4x4 {
        let eye = cameraEye()
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
            viewMatrix: viewMatrix(),
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
            focusDistance: currentFocusDistance(),
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
