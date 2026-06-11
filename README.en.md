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
    <a href="#gemini-api-key">Gemini API Key</a> ·
    <a href="#local-web-prototype">Local Web Prototype</a> ·
    <a href="#license">License</a>
  </p>
</div>

OpenReshot is a native iOS photo motion app. Pick a photo, drag inside a fixed frame, and the photo feels like it was captured from a different angle. When the view looks right, tap Reshoot to generate the final photo for that angle and save it back to the system photo library.

After WWDC 2026, spatial photos and angle-aware photo viewing became one of the most interesting iOS 27-era directions. Many users still cannot try that experience immediately because of system or device limits. OpenReshot brings the idea of "view it from another angle, then reshoot it from that angle" to more iPhones and iPads.

<div align="center">
  <video src="https://github.com/user-attachments/assets/57ccefa4-ae91-4185-ba59-5f2f672e9c61" width="760" controls muted playsinline></video>
</div>

## Core Experience

- Reshoot from a new angle: drag the photo and find a better angle or composition.
- Make still photos move: the photo changes perspective without moving the whole viewer.
- Generate the final image: when the angle feels right, Reshoot creates a clearer, more complete photo.
- Native iOS flow: system photo picker, native controls, and system photo library saving.
- Local photo motion: after downloading the model, photo motion runs on device and does not need a Gemini API key or a PC server.

## Latest Updates

**2026-06-12**

- OpenReshot has been submitted to Apple TestFlight beta review and is currently waiting for Apple processing.
- TestFlight review usually takes about 1-2 days; the TestFlight invite link will be added here after approval.
- Future installs and testing will go through TestFlight directly, without manually building to a device from Xcode.

<details>
<summary>Previous updates</summary>

### 2026-06-11

- The iOS flow now matches the PC flow: pick a photo, view motion, reshoot the current angle, and save the result.
- The iOS app is packaged as a lightweight shell by default and does not bundle the 1GB+ model; first use downloads the Core ML model from Hugging Face in Settings.
- The viewer includes glow, sheen, cover fade, and colorful missing-area treatment.
- The bottom controls now use a quieter PhoneClaw-style visual direction and blend directly into the background.
- The model repository is public on [Hugging Face](https://huggingface.co/kellyxiaowei/openreshoot-sharp-coreml), where the `SHARP.mlpackage` source files can be inspected directly.

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

The app downloads and rebuilds the Core ML package from Hugging Face:

```text
https://huggingface.co/kellyxiaowei/openreshoot-sharp-coreml
```

Downloaded file layout:

```text
SHARP.mlpackage/
  Manifest.json
  Data/com.apple.CoreML/model.mlmodel
  Data/com.apple.CoreML/weights/weight.bin
```

After download, the app compiles the package locally into `SHARP.mlmodelc`. Photo selection and photo motion then run through on-device Core ML inference. The model is based on the Apple SHARP research model; see [LICENSE_MODEL](LICENSE_MODEL) and the Hugging Face model card for model licensing.

## Gemini API Key

OpenReshot currently uses `gemini-3.1-flash-image` to generate the final Reshoot photo. The app does not include a shared public API key, so users need to enter their own Gemini API key.

Setup:

1. Open [Google AI Studio API Keys](https://aistudio.google.com/apikey), sign in, and create a Gemini API key.
2. In the iOS app, tap the Settings button.
3. Paste the API key and save it.
4. Return to the photo, drag to the angle you want, then tap Reshoot.

The local photo motion experience does not need a Gemini API key. Gemini is only used when the user taps Reshoot to generate the final image. Do not commit your API key to GitHub.

## Local Web Prototype

The repository keeps the Python / MPS web prototype mainly for contributors who want to compare the iOS result with the PC flow. Regular iOS use does not require it.

```bash
pip install -r requirements-web.txt
export GEMINI_API_KEY="your-key"
bash run.sh
```

Then open `http://127.0.0.1:8765/splat/`.

## Contributing

Issues, screenshots, and short reproduction videos are welcome. For iOS viewer, photo motion, Gemini Reshoot, or save-flow bugs, please include the device model, iOS version, and a concise reproduction path.

## License

OpenReshot-owned source code is licensed under the [Apache License 2.0](LICENSE), matching PhoneClaw.

Apple SHARP upstream source licensing is preserved in [LICENSE_APPLE_SHARP](LICENSE_APPLE_SHARP), and the released model license is in [LICENSE_MODEL](LICENSE_MODEL). Also see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for upstream open-source notices.
