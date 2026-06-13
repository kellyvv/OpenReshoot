<div align="center">
  <h1>OpenReshot</h1>

  <p><strong>Reshoot an ordinary photo from a new angle.</strong></p>

  <p>
    <img alt="Swift" src="https://img.shields.io/badge/Swift-5-orange">
    <img alt="iOS" src="https://img.shields.io/badge/iOS-18%2B-black">
    <img alt="Core ML" src="https://img.shields.io/badge/Core%20ML-on%20device-0A84FF">
    <img alt="Metal" src="https://img.shields.io/badge/Metal-rendering-8E8E93">
    <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green">
  </p>

  <p>
    <a href="README.md">中文</a> ·
    <a href="https://github.com/kellyvv/OpenReshot/issues">Issues</a> ·
    <a href="https://github.com/kellyvv/OpenReshot/issues">Feature requests</a>
  </p>

  <p>
    <a href="#core-experience">Core Experience</a> ·
    <a href="#latest-updates">Latest Updates</a> ·
    <a href="#quick-start">Quick Start</a> ·
    <a href="#model-download">Model Download</a> ·
    <a href="#license">License</a>
  </p>
</div>

OpenReshot is a native iOS photo motion app. Pick a photo, drag inside a fixed frame, and the photo feels like it was captured from a different angle. When the view looks right, tap Reshoot to generate the final photo for that angle and save it back to the system photo library.

After WWDC 2026, spatial photos and angle-aware photo viewing became one of the most interesting iOS 27-era directions. Many users still cannot try that experience immediately because of system or device limits. OpenReshot brings the idea of "view it from another angle, then reshoot it from that angle" to more iPhones and iPads.

<table>
  <tr>
    <td width="50%" align="center">
      <video src="https://github.com/user-attachments/assets/57ccefa4-ae91-4185-ba59-5f2f672e9c61" width="360" controls muted playsinline></video>
    </td>
    <td width="50%" align="center">
      <video src="https://github.com/user-attachments/assets/d785d845-1fd7-4bb6-978e-c9d87ec1572f" width="360" controls muted playsinline></video>
    </td>
  </tr>
</table>

## Core Experience

- Reshoot from a new angle: drag the photo and find a better angle or composition.
- Generate the final image: when the angle feels right, Reshoot creates a clearer, more complete photo.

## Latest Updates

**2026-06-14**

- The UI has been rebuilt around a more camera-like experience, with a new **Lens** control panel.
- Post-capture focus, aperture depth of field, and Hitchcock zoom are now supported: after taking a photo, you can pick a new focus point, adjust the f-number, and combine dolly movement with inverse FOV compensation.
- The result comparison is now a before / after wipe slider, so you can drag the divider and decide directly whether the repair is worth saving.
- A new **Spatial Gallery** caches reconstructed results locally. Reopening the same photo can load its spatial result directly instead of repeating minute-scale inference.
- The default preview is now part of the Spatial Gallery. First launch loads it once, and later launches no longer auto-open any spatial photo.

<details>
<summary>Previous updates</summary>

### 2026-06-12

- OpenReshot has been submitted to Apple TestFlight beta review and is currently waiting for Apple processing.
- TestFlight review usually takes about 1-2 days; the TestFlight invite link will be added here after approval.
- Future installs and testing will go through TestFlight directly, without manually building to a device from Xcode.

### 2026-06-11

- The iOS flow now covers picking a photo, viewing motion, reshooting the current angle, and saving the result.
- The iOS app is packaged as a lightweight shell by default and does not bundle the 1GB+ model; first use downloads the Core ML model from ModelScope in Settings, with Hugging Face as the fallback mirror.
- The viewer includes glow, sheen, cover fade, and colorful missing-area treatment.
- The bottom controls now use a quieter PhoneClaw-style visual direction and blend directly into the background.
- The model repository is public on [ModelScope](https://modelscope.cn/models/kilywei/openreshoot-sharp-coreml) and [Hugging Face](https://huggingface.co/kellyxiaowei/openreshoot-sharp-coreml), where the `SHARP.mlpackage` source files can be inspected directly.

</details>

## Quick Start

Requirements:

- macOS with Xcode 16 or newer.
- XcodeGen: `brew install xcodegen`
- A real iPhone or iPad on iOS 18 or newer.

Prepare the project:

```bash
git clone https://github.com/kellyvv/OpenReshot.git
cd OpenReshot
cd ios
xcodegen generate
open OpenReshot.xcodeproj
```

In Xcode, select the `OpenReshot` target, set your signing team, choose a real iOS device, and run. The default build does not include `SHARP.mlpackage`, keeping the app bundle lightweight.

More iOS-specific details are in [ios/README.md](ios/README.md).

## Model Download

After first launch, tap the Settings button in the top-right corner, then go to **Settings → Model → Download Model**.

The app downloads and rebuilds the Core ML package from ModelScope by default. If the primary source fails, it falls back to the Hugging Face mirror:

```text
ModelScope download source:
https://modelscope.cn/models/kilywei/openreshoot-sharp-coreml/resolve/main/SHARP.mlpackage

Hugging Face fallback:
https://huggingface.co/kellyxiaowei/openreshoot-sharp-coreml/resolve/main/SHARP.mlpackage
```

Public model repositories:

```text
ModelScope:
https://modelscope.cn/models/kilywei/openreshoot-sharp-coreml

Hugging Face:
https://huggingface.co/kellyxiaowei/openreshoot-sharp-coreml
```

Downloaded file layout:

```text
SHARP.mlpackage/
  Manifest.json
  Data/com.apple.CoreML/model.mlmodel
  Data/com.apple.CoreML/weights/weight.bin
```

After download, the app compiles the package locally into `SHARP.mlmodelc`. Photo selection and photo motion then run on device.

## Contributing

Issues, screenshots, and short reproduction videos are welcome.

## License

OpenReshot-owned source code is licensed under the [Apache License 2.0](LICENSE).
