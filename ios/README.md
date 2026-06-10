# OpenReshoot — iOS app

Upload a photo → OpenReshoot runs reconstruction **on-device via Core ML** → a metric 3D-gaussian cloud →
a lightweight **Metal photo layer** that adds fixed-frame drag parallax.

This is the iOS counterpart of the web app. The hard pieces (Core ML I/O, the
NDC→metric covariance math, the EWA splat renderer) are ported from the **validated
Python/MPS** pipeline; the Core ML model was converted and parity-checked on a Mac
(render PSNR 33 dB vs PyTorch, visually identical).

## Prerequisites
- macOS + **Xcode 15+**
- **XcodeGen**: `brew install xcodegen`
- An iPhone on **iOS 18+** (Pro / 8 GB recommended — see *Memory* below)
- Python dependencies from the repository root: `pip install -r requirements-ios.txt`
- The converted model: **`SHARP.mlpackage`** (prepared by `scripts/prepare_ios_model.sh` from the repository root)

## Build & run
```bash
# 1. from the repository root, check local setup
scripts/check_openreshoot_setup.sh

# 2. download/convert/copy the Core ML model
scripts/prepare_ios_model.sh

# 3. generate the Xcode project
cd ios
xcodegen generate                    # creates OpenReshoot.xcodeproj
open OpenReshoot.xcodeproj
```
In Xcode: select the **OpenReshoot** target → **Signing & Capabilities** → set your
**Team**; pick your **device**; **Run** (⌘R).

Then: tap **+**, choose a photo, wait ~10–30 s for reconstruction, then drag inside
the photo. The UI controls stay fixed; only the photo content changes angle.

For the **重构** button, tap the key button and save your Gemini API key. The
native app uses `gemini-3.1-flash-image` to generate the final photo from the
current-angle PNG.

Create a key at [Google AI Studio API Keys](https://aistudio.google.com/apikey).
Do not commit your API key to git.

## Files
| file | role |
|---|---|
| `SharpModel.swift` | load the bundled Core ML model, preprocess, run inference |
| `GaussianCloud.swift` | NDC→metric: build 3D covariance + unproject (**no SVD** — renderer takes covariance) |
| `Splat.metal` | EWA splatting (project mean+cov → 2D conic → gaussian) |
| `SplatRenderer.swift` | Metal pipeline + fixed-frame parallax camera + instanced draw |
| `OpenReshootApp.swift` | SwiftUI: photo picker, fixed controls, Metal photo layer, enhance flow |

## Caveats (read this)
- **This is an early native iOS implementation**. The logic is ported from the
  validated Python pipeline, but device performance depends heavily on memory and
  thermal limits.
- **App size ~1.2 GB** (the fp16 1536 model is bundled). Fine for personal device builds;
  for distribution you'd download the model on first launch instead.
- **Memory**: the 1536x1536 ViT activations are the real bottleneck (not just the
  1.2 GB weights). The default model is already FP16. The upstream network is not
  shape-agnostic; a simple 1024x1024 Core ML export fails because the sliding-pyramid
  patch encoder expects the 1536 -> 768 -> 384 pyramid.
- **Runtime motion** is intentionally limited to fixed-frame photo parallax. This app is
  not trying to be a full 3DGS scene viewer.
- The iOS **重构** step calls Gemini directly with the API key saved in the app.
  The current model is `gemini-3.1-flash-image`.
