# OpenReshot iOS

OpenReshot iOS 是原生 SwiftUI + Core ML + Metal 版本。默认打包为空壳 App，不把 1GB+ 的 SHARP 模型放进安装包；首次使用时在设置里下载模型，下载完成后在设备本机编译并运行。

## 要求

- macOS + Xcode 16 或更新版本。
- XcodeGen：`brew install xcodegen`
- iOS 18 或更新版本的 iPhone / iPad 真机。
- 建议使用内存较大的设备，Pro / 8GB 内存机型更稳。

普通 iOS 构建不需要本地 Python、Core ML 转换脚本，也不需要提前准备 `SHARP.mlpackage`。

## 构建运行

```bash
cd ios
xcodegen generate
open OpenReshot.xcodeproj
```

在 Xcode 中选择 `OpenReshot` target，配置签名 Team，选择真机运行。默认 target 会排除 `OpenReshot/SHARP.mlpackage`，所以 App 包体积保持轻量。

首次运行流程：

1. 点首页右上角设置按钮。
2. 在 **模型** 区域点 **下载模型**。
3. 等待下载和本机编译完成。
4. 回到首页选择照片，等待重建完成后拖动照片视角。

## 模型下载

内置主下载地址在 `OpenReshotModelDownloadBaseURL`：

```text
https://modelscope.cn/models/kilywei/openreshoot-sharp-coreml/resolve/main/SHARP.mlpackage
```

备用下载地址在 `OpenReshotModelDownloadMirrorBaseURLs`：

```text
https://huggingface.co/kellyxiaowei/openreshoot-sharp-coreml/resolve/main/SHARP.mlpackage
```

公开模型仓库：

```text
ModelScope:
https://modelscope.cn/models/kilywei/openreshoot-sharp-coreml

Hugging Face:
https://huggingface.co/kellyxiaowei/openreshoot-sharp-coreml
```

App 不下载 zip，而是直接下载 ModelScope / Hugging Face 上可见的 `.mlpackage` 源文件：

```text
SHARP.mlpackage/
  Manifest.json
  Data/com.apple.CoreML/model.mlmodel
  Data/com.apple.CoreML/weights/weight.bin
```

下载完成后，App 会在本机调用 `MLModel.compileModel(at:)` 编译为 `SHARP.mlmodelc`，并保存到 Application Support：

```text
OpenReshot/Models/SHARP.mlmodelc
```

后续启动会优先使用已下载模型；没有下载模型时，设置页会显示未安装。

## Gemini API Key

照片动效和拖动预览只需要本机 SHARP 模型，不需要 Gemini API key。

点击 **重构** 生成最终照片时，App 会调用 `gemini-3.1-flash-image`。应用不内置公共 API key，需要用户在设置里保存自己的 Gemini API key。

创建 key：

```text
https://aistudio.google.com/apikey
```

不要把自己的 API key 提交到 GitHub。

## 主要文件

| file | role |
|---|---|
| `OpenReshotApp.swift` | SwiftUI 主界面、设置页、模型下载/安装状态、照片选择、重构流程 |
| `SharpModel.swift` | 加载已下载或 bundle 内的 Core ML 模型，预处理并运行推理 |
| `GaussianCloud.swift` | NDC 到 metric 坐标转换，构建 3D covariance |
| `Splat.metal` | EWA splatting shader |
| `SplatRenderer.swift` | Metal 渲染、固定画框视角拖动、实例化绘制 |
| `project.yml` | XcodeGen 配置，包含模型下载地址和 AppIcon 设置 |

## 可选：本地转换模型

只有在你要重新转换 Apple SHARP 模型时，才需要本地 Python/Core ML 流程：

```bash
pip install -r requirements/requirements-ios.txt
scripts/prepare_ios_model.sh
```

这个流程会生成 `ios/OpenReshot/SHARP.mlpackage`。当前默认 `project.yml` 会把它从 target sources 里排除，避免打包进 App；如果你明确要内置模型，需要手动调整 XcodeGen 配置。

## 注意事项

- 模型权重约 1.23 GiB，下载和编译都需要稳定网络、足够电量和足够存储空间。
- SHARP 1536x1536 模型的 ViT activations 很吃内存，低内存设备可能失败或被系统终止。
- 当前运行时动效定位是固定画框照片视差，不是完整 3DGS 场景查看器。
- iOS **重构** 步骤会使用用户保存的 Gemini API key 直接请求 Gemini。
- OpenReshot 自有源码采用 Apache 2.0；Apple SHARP 上游源码和研究模型分别受 `docs/LICENSE_APPLE_SHARP`、`docs/LICENSE_MODEL` 约束。不要把研究模型当作商业分发资产使用。
