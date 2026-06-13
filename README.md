<div align="center">
  <h1>OpenReshot</h1>

  <p><strong>把一张普通照片，换个角度重新拍出来。</strong></p>

  <p>
    <img alt="Swift" src="https://img.shields.io/badge/Swift-5-orange">
    <img alt="iOS" src="https://img.shields.io/badge/iOS-18%2B-black">
    <img alt="Core ML" src="https://img.shields.io/badge/Core%20ML-on%20device-0A84FF">
    <img alt="Metal" src="https://img.shields.io/badge/Metal-rendering-8E8E93">
    <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green">
  </p>

  <p>
    <a href="README.en.md">English</a> ·
    <a href="https://github.com/kellyvv/OpenReshot/issues">反馈问题</a> ·
    <a href="https://github.com/kellyvv/OpenReshot/issues">功能建议</a>
  </p>

  <p>
    <a href="#核心体验">核心体验</a> ·
    <a href="#最新更新">最新更新</a> ·
    <a href="#快速开始">快速开始</a> ·
    <a href="#模型下载">模型下载</a> ·
    <a href="#许可">许可</a>
  </p>
</div>

OpenReshot 是一个原生 iOS 照片动效应用。选择一张照片后，你可以在固定画框里轻轻拖动视角，让照片像是从另一个角度被重新拍摄；满意后点击重构，生成当前视角的最终照片，并保存回系统相册。

WWDC 2026 之后，空间照片和可变视角照片成为 iOS 27 时代很有吸引力的新体验。但很多用户会因为系统版本或设备限制暂时用不上。OpenReshot 想做的就是把这种“普通照片也能换角度看、换角度拍”的体验，带给更多 iPhone 和 iPad。

<div align="center">
  <video src="https://github.com/user-attachments/assets/57ccefa4-ae91-4185-ba59-5f2f672e9c61" width="760" controls muted playsinline></video>
</div>

## 核心体验

- 换个角度重新拍：拖动照片，找到比原图更合适的视角和构图。
- 生成最终照片：当前角度满意后，点重构生成一张更完整、更清晰的照片。

## 最新更新

**2026-06-14**

- UI 做了全新的相机化改版，新增 **镜头** 控制面板。
- 支持事后对焦、光圈景深和希区柯克变焦：照片拍完后还能重新选焦点、调 f-number、做推轨 + FOV 反向补偿。
- 结果页对比升级为 before / after 擦除滑块，用拖动分割线直接判断修复值不值得保存。
- 新增 **空间画廊**：重建结果会缓存在本机，再次打开同一张照片可直接加载空间，不必重复跑分钟级推理。
- 默认预览已放入空间画廊；首次启动会自动加载一次，之后启动不再自动进入任何空间照片。

<details>
<summary>历史更新</summary>

### 2026-06-12

- OpenReshot 已提交 Apple TestFlight 审核，当前等待 Apple 处理。
- TestFlight 审核通常需要 1-2 天；通过后会在这里补充 TestFlight 邀请链接。
- 后续体验会直接通过 TestFlight 安装，不需要再手动用 Xcode 打包到手机。

### 2026-06-11

- iOS 主流程已完成：选图、照片动效、当前角度重构、结果保存。
- iOS 默认打包为空壳 App，不内置 1GB+ 模型；首次使用可在设置里从 Hugging Face 下载 Core ML 模型。
- 查看器动效已补齐 glow、sheen、cover fade 和多彩缺失区域处理。
- 底部操作区改成更克制的 PhoneClaw 同系风格，按钮直接融合在背景里。
- 模型仓库已发布到 [Hugging Face](https://huggingface.co/kellyxiaowei/openreshoot-sharp-coreml)，可直接查看 `SHARP.mlpackage` 源文件。

</details>

## 快速开始

要求：

- macOS，Xcode 16 或更新版本。
- XcodeGen：`brew install xcodegen`
- iOS 18 或更新版本的 iPhone / iPad 真机。

准备工程：

```bash
git clone https://github.com/kellyvv/OpenReshot.git
cd OpenReshot
cd ios
xcodegen generate
open OpenReshot.xcodeproj
```

在 Xcode 中选择 `OpenReshot` target，配置签名 Team，选择真机运行。默认构建不包含 `SHARP.mlpackage`，App 包体积保持轻量。

iOS 细节见 [ios/README.md](ios/README.md)。

## 模型下载

首次运行后，点首页右上角设置按钮，进入 **设置 → 模型 → 下载模型**。

App 会从 Hugging Face 下载并在本机重建 Core ML package：

```text
https://huggingface.co/kellyxiaowei/openreshoot-sharp-coreml
```

下载的文件结构是：

```text
SHARP.mlpackage/
  Manifest.json
  Data/com.apple.CoreML/model.mlmodel
  Data/com.apple.CoreML/weights/weight.bin
```

下载完成后，App 会在本机编译为 `SHARP.mlmodelc`，之后选图和照片动效都在本机运行。

## 参与

欢迎提交 issue、效果截图和复现步骤。

## 许可

OpenReshot 自有源码采用 [Apache License 2.0](LICENSE)。
