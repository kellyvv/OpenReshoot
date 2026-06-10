import Foundation
import CoreML
import simd
import SplatIO

/// Turns the network's NDC gaussians into metric MetalSplatter `SplatPoint`s.
/// MetalSplatter wants position + scale + rotation(quat), so we decompose the
/// metric covariance with a 3×3 symmetric eigensolver (Jacobi).
/// Gaussians stay in OpenCV camera space; the OpenCV→GL flip is done in the view matrix.
enum GaussianCloud {

    static func build(from o: SharpOutput, quality: RenderQuality) -> (points: [SplatPoint], focus: Float) {
        let n = o.count
        let mean = toFloats(o.mean)      // n*3
        let scaleA = toFloats(o.scale)   // n*3
        let quat = toFloats(o.quat)      // n*4
        let color = toFloats(o.color)    // n*3
        let opac = toFloats(o.opacity)   // n

        // --- unprojection matrix: inv(ndc * intrinsics_resized), extrinsics = I ---
        let R = Float(SharpModel.internalRes)
        let fxR = o.fpx * R / Float(o.width)
        let fyR = o.fpx * R / Float(o.height)
        let cc = R / 2.0
        let ndc = simd_float4x4(columns: (
            SIMD4(2 / R, 0, 0, 0), SIMD4(0, 2 / R, 0, 0), SIMD4(-1, -1, 1, 0), SIMD4(0, 0, 0, 1)))
        let intr = simd_float4x4(columns: (
            SIMD4(fxR, 0, 0, 0), SIMD4(0, fyR, 0, 0), SIMD4(cc, cc, 1, 0), SIMD4(0, 0, 0, 1)))
        let unproj = simd_inverse(ndc * intr)
        let A = simd_float3x3(columns: (unproj.columns.0.xyz, unproj.columns.1.xyz, unproj.columns.2.xyz))
        let bvec = unproj.columns.3.xyz
        let At = A.transpose

        // --- Decimation: keep the most important gaussians per step×step block. ------
        // The model emits ~1 gaussian/pixel of its 1536² grid; a phone is comfortable with
        // a few hundred K. Match the PC viewer's importance idea (size * opacity) instead
        // of keeping opacity alone; high quality keeps extra contributors so semi-transparent
        // blurry/disoccluded areas behave closer to the PC viewer.
        let res = SharpModel.internalRes
        let memoryGB = ProcessInfo.processInfo.physicalMemory / 1_073_741_824
        let budget = quality.splatBudget(memoryGB: memoryGB)
        var step = 1
        while (res / step) * (res / step) > budget { step += 1 }
        func importance(_ i: Int) -> Float {
            let sx = max(scaleA[i * 3], 1e-6)
            let sy = max(scaleA[i * 3 + 1], 1e-6)
            let sz = max(scaleA[i * 3 + 2], 1e-6)
            return opac[i] * sx * sy * sz
        }
        var kept: [Int]
        if step > 1 && n == res * res {
            let keepPerBlock = (quality == .high && memoryGB >= 8 && step == 2) ? 2 : 1
            var k = [Int](); k.reserveCapacity((res / step) * (res / step) * keepPerBlock)
            var by = 0
            while by + step <= res {
                var bx = 0
                while bx + step <= res {
                    var first = -1
                    var second = -1
                    var firstScore = -Float.greatestFiniteMagnitude
                    var secondScore = -Float.greatestFiniteMagnitude
                    for dy in 0..<step { for dx in 0..<step {
                        let i = (by + dy) * res + (bx + dx)
                        let score = importance(i)
                        if score > firstScore {
                            second = first
                            secondScore = firstScore
                            first = i
                            firstScore = score
                        } else if score > secondScore {
                            second = i
                            secondScore = score
                        }
                    } }
                    if first >= 0 { k.append(first) }
                    if keepPerBlock > 1 && second >= 0 { k.append(second) }
                    bx += step
                }
                by += step
            }
            kept = k
        } else {
            kept = Array(0..<n)
        }
        // Do not inflate splat scale after decimation. The PC viewer renders the
        // model's native scales; multiplying by the grid step makes the photo soft.
        let sBoost: Float = 1
        // -----------------------------------------------------------------------------

        let minOpacity: Float = 0
        var pts = [SplatPoint?](repeating: nil, count: kept.count)
        pts.withUnsafeMutableBufferPointer { buf in
            DispatchQueue.concurrentPerform(iterations: kept.count) { k in
                let i = kept[k]
                let op = opac[i]
                if op <= minOpacity { return }
                let q = SIMD4<Float>(quat[i * 4], quat[i * 4 + 1], quat[i * 4 + 2], quat[i * 4 + 3])
                let Rm = rotMat(q)
                let s = SIMD3<Float>(scaleA[i * 3], scaleA[i * 3 + 1], scaleA[i * 3 + 2]) * sBoost
                let s2 = s * s
                let Rs = simd_float3x3(columns: (Rm.columns.0 * s2.x, Rm.columns.1 * s2.y, Rm.columns.2 * s2.z))
                let covNdc = Rs * Rm.transpose
                let mMetric = A * SIMD3<Float>(mean[i * 3], mean[i * 3 + 1], mean[i * 3 + 2]) + bvec
                let covMetric = A * covNdc * At

                // decompose metric covariance -> scale (sqrt eigenvalues) + rotation (eigenvectors)
                let (eval, evec) = jacobiEigen3(covMetric)
                let scaleVec = SIMD3<Float>(sqrtf(max(eval.x, 1e-12)), sqrtf(max(eval.y, 1e-12)), sqrtf(max(eval.z, 1e-12)))
                var Rmat = evec
                if Rmat.determinant < 0 { Rmat.columns.0 = -Rmat.columns.0 }  // proper rotation

                let col = linearToSRGB(SIMD3<Float>(color[i * 3], color[i * 3 + 1], color[i * 3 + 2]))
                buf[k] = SplatPoint(position: mMetric,
                                    color: .sRGBUInt8(toU8(col * 255)),
                                    opacity: .linearFloat(op),
                                    scale: .linearFloat(scaleVec),
                                    rotation: simd_quatf(Rmat).normalized)
            }
        }
        let points = pts.compactMap { $0 }

        // Lookat depth ~ median z of the bulk (OpenCV z = forward).
        var zs = stride(from: 0, to: points.count, by: max(1, points.count / 4096)).map { points[$0].position.z }.filter { $0 > 0 }
        zs.sort()
        let focus = zs.isEmpty ? 1.0 : zs[zs.count / 2]
        print("🎯 [OpenReshoot] \(points.count) splats (\(quality.title), grid \(res)², step \(step), \(n) raw, opacity ≥ \(minOpacity)), focus=\(focus)")
        return (points, focus)
    }
}

// MARK: - math helpers

extension SIMD4 where Scalar == Float { var xyz: SIMD3<Float> { SIMD3(x, y, z) } }

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

func toFloats(_ a: MLMultiArray) -> [Float] {
    let n = a.count
    var out = [Float](repeating: 0, count: n)
    switch a.dataType {
    case .float32:
        let p = a.dataPointer.bindMemory(to: Float.self, capacity: n)
        out.withUnsafeMutableBufferPointer { _ = memcpy($0.baseAddress!, p, n * 4) }
    case .float16:
        let p = a.dataPointer.bindMemory(to: Float16.self, capacity: n)
        for i in 0..<n { out[i] = Float(p[i]) }
    default:
        for i in 0..<n { out[i] = a[i].floatValue }
    }
    return out
}
