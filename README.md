<div align="center">
  <img src="data/teaser.jpg" alt="OpenReshoot preview" width="760">

  <h1>OpenReshoot</h1>

  <p><strong>把一张普通照片，换个角度重新拍出来。</strong></p>

  <p>
    <img alt="Swift" src="https://img.shields.io/badge/Swift-5-orange">
    <img alt="iOS" src="https://img.shields.io/badge/iOS-18%2B-black">
    <img alt="Core ML" src="https://img.shields.io/badge/Core%20ML-on%20device-0A84FF">
    <img alt="Metal" src="https://img.shields.io/badge/Metal-rendering-8E8E93">
    <img alt="License" src="https://img.shields.io/badge/License-see%20LICENSE-lightgrey">
  </p>

  <p>
    <a href="README.en.md">English</a> ·
    <a href="https://github.com/kellyvv/OpenReshoot/issues">反馈问题</a> ·
    <a href="https://github.com/kellyvv/OpenReshoot/issues">功能建议</a>
  </p>

  <p>
    <a href="#核心体验">核心体验</a> ·
    <a href="#最新状态">最新状态</a> ·
    <a href="#快速开始">快速开始</a> ·
    <a href="#gemini-api-key">Gemini API Key</a> ·
    <a href="#本地-web-原型">本地 Web 原型</a> ·
    <a href="#许可">许可</a>
  </p>
</div>

OpenReshoot 是一个原生 iOS 照片动效应用。选择一张照片后，你可以在固定画框里轻轻拖动视角，让照片像是从另一个角度被重新拍摄；满意后点击重构，生成当前视角的最终照片，并保存回系统相册。

WWDC 2026 之后，空间照片和可变视角照片成为 iOS 27 时代很有吸引力的新体验。但很多用户会因为系统版本或设备限制暂时用不上。OpenReshoot 想做的就是把这种“普通照片也能换角度看、换角度拍”的体验，带给更多 iPhone 和 iPad。

## 核心体验

- 换个角度重新拍：拖动照片，找到比原图更合适的视角和构图。
- 照片动起来：照片本身产生视角变化，而不是整个查看器跟着晃动。
- 生成最终照片：当前角度满意后，点重构生成一张更完整、更清晰的照片。
- 原生 iOS 体验：使用系统照片选择器、原生按钮和相册保存流程，不需要打开网页。
- 本地照片动效：只看照片动效不需要 API key，也不需要连接 PC 服务端。

## 最新状态

2026-06-11

- iOS 主流程已对齐 PC：选图、照片动效、当前角度重构、结果保存。
- 查看器动效已补齐 glow、sheen、cover fade 和多彩缺失区域处理。
- 底部操作区改成更克制的 PhoneClaw 同系风格，按钮直接融合在背景里。
- 仓库 README、依赖检查脚本和模型准备脚本已整理，方便新的贡献者直接跑起来。

## 快速开始

要求：

- macOS，Xcode 15 或更新版本。
- XcodeGen：`brew install xcodegen`
- Python 依赖：`pip install -r requirements-ios.txt`
- iOS 18 或更新版本的 iPhone / iPad 真机。

准备工程：

```bash
git clone https://github.com/kellyvv/OpenReshoot.git
cd OpenReshoot
scripts/check_openreshoot_setup.sh
scripts/prepare_ios_model.sh
cd ios
xcodegen generate
open OpenReshoot.xcodeproj
```

在 Xcode 中选择 `OpenReshoot` target，配置签名 Team，选择真机运行。

iOS 细节见 [ios/README.md](ios/README.md)。

## Gemini API Key

OpenReshoot 当前使用 `gemini-3.1-flash-image` 生成重构后的最终照片。应用不内置公共 API key，用户需要填写自己的 Gemini API key。

设置方式：

1. 打开 [Google AI Studio API Keys](https://aistudio.google.com/apikey)，登录后创建 Gemini API key。
2. 在 iOS 应用里点右下角 key 按钮。
3. 粘贴 API key 并保存。
4. 回到照片页，拖到想要的角度后点击重构。

只看照片动效不需要 API key；只有点击重构时才会请求 Gemini。不要把自己的 API key 提交到 GitHub。

## 本地 Web 原型

仓库保留了 Python / MPS 的本地 Web 原型，主要用于贡献者对齐 PC 端效果。普通 iOS 使用不需要运行它。

```bash
pip install -r requirements-web.txt
export GEMINI_API_KEY="your-key"
bash run.sh
```

然后打开 `http://127.0.0.1:8765/splat/`。

## 参与

欢迎提交 issue、效果截图和复现步骤。与 iOS 查看器、照片动效、Gemini 重构、保存流程相关的问题，建议附上机型、iOS 版本和一段简短录屏。

## 许可

源码许可见 [LICENSE](LICENSE)，模型许可见 [LICENSE_MODEL](LICENSE_MODEL)。上游开源声明见 [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS)。
