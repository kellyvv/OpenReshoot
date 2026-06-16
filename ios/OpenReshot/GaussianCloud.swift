import Foundation
import CoreML
import simd
import SplatIO

/// Turns the network's NDC gaussians into metric MetalSplatter `SplatPoint`s.
/// MetalSplatter wants position + scale + rotation(quat), so we decompose the
/// metric covariance with a 3×3 symmetric eigensolver (Jacobi).
/// Gaussians stay in OpenCV camera space; the OpenCV→GL flip is done in the view matrix.
enum GaussianCloud {
    struct FileBuildResult {
        let url: URL
        let splatCount: Int
        let focus: Float
    }

    static func writePLY(from o: SharpOutput, quality: RenderQuality, to url: URL) async throws -> FileBuildResult {
        let work = Work(o: o, quality: quality)
        let emittedPointCount = work.validPointCount()
        let writer = try SplatPLYSceneWriter(toFileAtPath: url.path)
        try await writer.start(sphericalHarmonicDegree: 0, binary: true, pointCount: emittedPointCount)

        let batchSize = 65_536
        var batch: [SplatPoint] = []
        batch.reserveCapacity(batchSize)
        var focusSamples: [Float] = []
        focusSamples.reserveCapacity(min(4096, emittedPointCount))
        let focusStride = max(1, emittedPointCount / 4096)
        var pointStats = OpenReshotDiagnostics.enabled
            ? PointCloudStats(sampleStride: max(1, emittedPointCount / 8192))
            : nil
        var emittedIndex = 0

        do {
            for k in 0..<work.pointCount {
                if let point = makePoint(work: work, outputIndex: work.outputIndex(for: k)) {
                    if emittedIndex % focusStride == 0, point.position.z > 0 {
                        focusSamples.append(point.position.z)
                    }
                    pointStats?.record(point, emittedIndex: emittedIndex)
                    emittedIndex += 1
                    batch.append(point)
                    if batch.count == batchSize {
                        try await writer.write(batch)
                        batch.removeAll(keepingCapacity: true)
                    }
                }
            }
            if !batch.isEmpty {
                try await writer.write(batch)
            }
            try await writer.close()
        } catch {
            try? await writer.close()
            throw error
        }

        focusSamples.sort()
        let focus = focusSamples.isEmpty ? 1.0 : focusSamples[focusSamples.count / 2]
        print("🎯 [OpenReshot] \(emittedPointCount) splats written (\(quality.title), grid \(work.res)², raw=\(work.rawCount), decimated=\(work.isDecimated), step=\(work.step), opacity ≥ \(work.minOpacity)), focus=\(focus)")
        pointStats?.log(rawCount: work.rawCount, emittedCount: emittedPointCount, focus: focus)
        if OpenReshotDiagnostics.enabled {
            print("🧪 [OpenReshot][Diag] ply file: bytes=\(OpenReshotDiagnostics.fileByteCount(url)), sha=\(OpenReshotDiagnostics.shortFileSHA256(url)), url=\(url.lastPathComponent)")
        }
        return FileBuildResult(url: url, splatCount: emittedPointCount, focus: focus)
    }

    private struct PointCloudStats {
        let sampleStride: Int
        var finiteCount = 0
        var nonFiniteCount = 0
        var nonPositiveZCount = 0
        var minPosition = SIMD3<Float>(repeating: Float.greatestFiniteMagnitude)
        var maxPosition = SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude)
        var zSamples: [Float] = []

        mutating func record(_ point: SplatPoint, emittedIndex: Int) {
            let p = point.position
            guard p.x.isFinite, p.y.isFinite, p.z.isFinite else {
                nonFiniteCount += 1
                return
            }
            finiteCount += 1
            if p.z <= 0 { nonPositiveZCount += 1 }
            minPosition = simd_min(minPosition, p)
            maxPosition = simd_max(maxPosition, p)
            if emittedIndex % sampleStride == 0 {
                zSamples.append(p.z)
            }
        }

        mutating func log(rawCount: Int, emittedCount: Int, focus: Float) {
            zSamples.sort()
            func percentile(_ fraction: Float) -> String {
                guard !zSamples.isEmpty else { return "<nil>" }
                let index = min(zSamples.count - 1, max(0, Int(Float(zSamples.count - 1) * fraction)))
                return OpenReshotDiagnostics.format(Double(zSamples[index]))
            }
            print("🧪 [OpenReshot][Diag] point cloud: raw=\(rawCount), emitted=\(emittedCount), finite=\(finiteCount), nonFinite=\(nonFiniteCount), z<=0=\(nonPositiveZCount), posMin=(\(OpenReshotDiagnostics.format(Double(minPosition.x))),\(OpenReshotDiagnostics.format(Double(minPosition.y))),\(OpenReshotDiagnostics.format(Double(minPosition.z)))), posMax=(\(OpenReshotDiagnostics.format(Double(maxPosition.x))),\(OpenReshotDiagnostics.format(Double(maxPosition.y))),\(OpenReshotDiagnostics.format(Double(maxPosition.z)))), zP01=\(percentile(0.01)), zP50=\(percentile(0.50)), zP99=\(percentile(0.99)), focus=\(OpenReshotDiagnostics.format(Double(focus)))")
        }
    }

    private struct Work {
        let rawCount: Int
        let mean: MultiArrayFloatReader
        let scaleA: MultiArrayFloatReader
        let quat: MultiArrayFloatReader
        let color: MultiArrayFloatReader
        let opac: MultiArrayFloatReader
        let res: Int
        let step: Int
        let kept: [Int]?
        let pointCount: Int
        let unprojectionA: simd_float3x3
        let unprojectionATranspose: simd_float3x3
        let unprojectionB: SIMD3<Float>
        let minOpacity: Float = 0
        var isDecimated: Bool { kept != nil }

        init(o: SharpOutput, quality: RenderQuality) {
            rawCount = o.count
            mean = MultiArrayFloatReader(o.mean)
            scaleA = MultiArrayFloatReader(o.scale)
            quat = MultiArrayFloatReader(o.quat)
            color = MultiArrayFloatReader(o.color)
            opac = MultiArrayFloatReader(o.opacity)

            let R = Float(o.internalResolution)
            let fxR = o.fpx * R / Float(o.width)
            let fyR = o.fpx * R / Float(o.height)
            let cc = R / 2.0
            let ndc = simd_float4x4(columns: (
                SIMD4(2 / R, 0, 0, 0), SIMD4(0, 2 / R, 0, 0), SIMD4(-1, -1, 1, 0), SIMD4(0, 0, 0, 1)))
            let intr = simd_float4x4(columns: (
                SIMD4(fxR, 0, 0, 0), SIMD4(0, fyR, 0, 0), SIMD4(cc, cc, 1, 0), SIMD4(0, 0, 0, 1)))
            let unproj = simd_inverse(ndc * intr)
            unprojectionA = simd_float3x3(columns: (unproj.columns.0.xyz, unproj.columns.1.xyz, unproj.columns.2.xyz))
            unprojectionATranspose = unprojectionA.transpose
            unprojectionB = unproj.columns.3.xyz

            res = o.internalResolution
            let memoryGB = ProcessInfo.processInfo.physicalMemory / 1_073_741_824
            let budget = quality.splatBudget(memoryGB: memoryGB)
            var nextStep = 1
            while (res / nextStep) * (res / nextStep) > budget { nextStep += 1 }
            step = nextStep

            if nextStep > 1 && o.count == res * res {
                let keepPerBlock = (quality == .high && memoryGB >= 8 && nextStep == 2) ? 2 : 1
                var k = [Int]()
                k.reserveCapacity((res / nextStep) * (res / nextStep) * keepPerBlock)
                var by = 0
                while by + nextStep <= res {
                    var bx = 0
                    while bx + nextStep <= res {
                        var first = -1
                        var second = -1
                        var firstScore = -Float.greatestFiniteMagnitude
                        var secondScore = -Float.greatestFiniteMagnitude
                        for dy in 0..<nextStep {
                            for dx in 0..<nextStep {
                                let i = (by + dy) * res + (bx + dx)
                                let score = opac[i] * max(scaleA[i * 3], 1e-6) * max(scaleA[i * 3 + 1], 1e-6) * max(scaleA[i * 3 + 2], 1e-6)
                                if score > firstScore {
                                    second = first
                                    secondScore = firstScore
                                    first = i
                                    firstScore = score
                                } else if score > secondScore {
                                    second = i
                                    secondScore = score
                                }
                            }
                        }
                        if first >= 0 { k.append(first) }
                        if keepPerBlock > 1 && second >= 0 { k.append(second) }
                        bx += nextStep
                    }
                    by += nextStep
                }
                kept = k
                pointCount = k.count
            } else {
                kept = nil
                pointCount = o.count
            }
        }

        func outputIndex(for pointIndex: Int) -> Int {
            kept?[pointIndex] ?? pointIndex
        }

        func validPointCount() -> Int {
            var count = 0
            for pointIndex in 0..<pointCount where GaussianCloud.isCandidatePoint(work: self, outputIndex: outputIndex(for: pointIndex)) {
                count += 1
            }
            return count
        }
    }

    private static func makePoint(work: Work, outputIndex i: Int) -> SplatPoint? {
        guard isCandidatePoint(work: work, outputIndex: i) else { return nil }
        let op = clampedOpacity(work.opac[i])
        let q = SIMD4<Float>(work.quat[i * 4], work.quat[i * 4 + 1], work.quat[i * 4 + 2], work.quat[i * 4 + 3])
        let Rm = rotMat(q)
        let s = SIMD3<Float>(work.scaleA[i * 3], work.scaleA[i * 3 + 1], work.scaleA[i * 3 + 2])
        let s2 = s * s
        let Rs = simd_float3x3(columns: (Rm.columns.0 * s2.x, Rm.columns.1 * s2.y, Rm.columns.2 * s2.z))
        let covNdc = Rs * Rm.transpose
        let mMetric = work.unprojectionA * SIMD3<Float>(work.mean[i * 3], work.mean[i * 3 + 1], work.mean[i * 3 + 2]) + work.unprojectionB
        let covMetric = work.unprojectionA * covNdc * work.unprojectionATranspose

        let (eval, evec) = jacobiEigen3(covMetric)
        let scaleVec = SIMD3<Float>(sqrtf(max(eval.x, 1e-12)), sqrtf(max(eval.y, 1e-12)), sqrtf(max(eval.z, 1e-12)))
        var Rmat = evec
        if Rmat.determinant < 0 { Rmat.columns.0 = -Rmat.columns.0 }

        let col = linearToSRGB(SIMD3<Float>(work.color[i * 3], work.color[i * 3 + 1], work.color[i * 3 + 2]))
        return SplatPoint(position: mMetric,
                          color: .sRGBUInt8(toU8(col * 255)),
                          opacity: .linearFloat(op),
                          scale: .linearFloat(scaleVec),
                          rotation: simd_quatf(Rmat).normalized)
    }

    private static func isCandidatePoint(work: Work, outputIndex i: Int) -> Bool {
        let op = work.opac[i]
        guard op.isFinite, op > work.minOpacity else { return false }
        let mean = SIMD3<Float>(work.mean[i * 3], work.mean[i * 3 + 1], work.mean[i * 3 + 2])
        let scale = SIMD3<Float>(work.scaleA[i * 3], work.scaleA[i * 3 + 1], work.scaleA[i * 3 + 2])
        let quat = SIMD4<Float>(work.quat[i * 4], work.quat[i * 4 + 1], work.quat[i * 4 + 2], work.quat[i * 4 + 3])
        let color = SIMD3<Float>(work.color[i * 3], work.color[i * 3 + 1], work.color[i * 3 + 2])
        guard mean.allFinite, scale.allFinite, quat.allFinite, color.allFinite else { return false }
        guard simd_length_squared(quat) > 1e-12 else { return false }
        return true
    }

    private static func clampedOpacity(_ value: Float) -> Float {
        min(1 - 1e-6, max(1e-6, value))
    }

}

// MARK: - math helpers

extension SIMD4 where Scalar == Float { var xyz: SIMD3<Float> { SIMD3(x, y, z) } }
extension SIMD3 where Scalar == Float { var allFinite: Bool { x.isFinite && y.isFinite && z.isFinite } }
extension SIMD4 where Scalar == Float { var allFinite: Bool { x.isFinite && y.isFinite && z.isFinite && w.isFinite } }

/// Quaternion (w,x,y,z), normalized -> rotation matrix from the model output convention.
func rotMat(_ qin: SIMD4<Float>) -> simd_float3x3 {
    let q = simd_normalize(qin)
    let w = q.x, x = q.y, y = q.z, z = q.w
    let xx = x * x, yy = y * y, zz = z * z
    let xy = x * y, xz = x * z, yz = y * z
    let wx = w * x, wy = w * y, wz = w * z
    return simd_float3x3(columns: (
        SIMD3(1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy)),
        SIMD3(2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx)),
        SIMD3(2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy))))
}

/// Cyclic Jacobi eigendecomposition for a symmetric 3×3. Returns (eigenvalues, eigenvectors-as-columns).
func jacobiEigen3(_ M: simd_float3x3) -> (SIMD3<Float>, simd_float3x3) {
    var a = M
    var v = matrix_identity_float3x3
    let pairs = [(0, 1), (0, 2), (1, 2)]
    for _ in 0..<12 {
        var off: Float = 0
        for (p, q) in pairs { off += abs(a[q][p]) }
        if off < 1e-9 { break }
        for (p, q) in pairs {
            let apq = a[q][p]
            if abs(apq) < 1e-20 { continue }
            let tau = (a[q][q] - a[p][p]) / (2 * apq)
            let t = (tau >= 0 ? Float(1) : Float(-1)) / (abs(tau) + (1 + tau * tau).squareRoot())
            let c = 1 / (1 + t * t).squareRoot()
            let s = t * c
            var J = matrix_identity_float3x3
            J[p][p] = c; J[q][q] = c
            J[q][p] = s         // element (row p, col q)
            J[p][q] = -s        // element (row q, col p)
            a = J.transpose * a * J
            v = v * J
        }
    }
    return (SIMD3(a[0][0], a[1][1], a[2][2]), v)
}

func linearToSRGB(_ c: SIMD3<Float>) -> SIMD3<Float> {
    func f(_ v: Float) -> Float { v <= 0.0031308 ? v * 12.92 : 1.055 * pow(v, 1 / 2.4) - 0.055 }
    return SIMD3(f(c.x), f(c.y), f(c.z))
}

func toU8(_ v: SIMD3<Float>) -> SIMD3<UInt8> {
    func u(_ x: Float) -> UInt8 { UInt8(max(0, min(255, x))) }
    return SIMD3<UInt8>(u(v.x), u(v.y), u(v.z))
}

private struct MultiArrayFloatReader {
    private let array: MLMultiArray
    private let float32Pointer: UnsafeMutablePointer<Float>?
    private let float16Pointer: UnsafeMutablePointer<Float16>?
    private let logicalShape: [Int]
    private let physicalStrides: [Int]

    init(_ array: MLMultiArray) {
        self.array = array
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        if shape.first == 1, shape.count > 1 {
            logicalShape = Array(shape.dropFirst())
            physicalStrides = Array(strides.dropFirst())
        } else {
            logicalShape = shape
            physicalStrides = strides
        }
        let storageElementCapacity = 1 + zip(shape, strides).reduce(0) { partial, item in
            partial + max(0, item.0 - 1) * item.1
        }
        switch array.dataType {
        case .float32:
            float32Pointer = array.dataPointer.bindMemory(to: Float.self, capacity: storageElementCapacity)
            float16Pointer = nil
        case .float16:
            float32Pointer = nil
            float16Pointer = array.dataPointer.bindMemory(to: Float16.self, capacity: storageElementCapacity)
        default:
            float32Pointer = nil
            float16Pointer = nil
        }
    }

    subscript(_ index: Int) -> Float {
        let offset = physicalOffset(forLogicalIndex: index)
        if let float32Pointer {
            return float32Pointer[offset]
        }
        if let float16Pointer {
            return Float(float16Pointer[offset])
        }
        return array[index].floatValue
    }

    private func physicalOffset(forLogicalIndex index: Int) -> Int {
        guard !logicalShape.isEmpty, logicalShape.count == physicalStrides.count else {
            return index
        }
        var remaining = index
        var offset = 0
        for dimension in stride(from: logicalShape.count - 1, through: 0, by: -1) {
            let size = logicalShape[dimension]
            guard size > 0 else { return offset }
            let coordinate = remaining % size
            remaining /= size
            offset += coordinate * physicalStrides[dimension]
        }
        return offset
    }
}
