# Contributing to OpenReshot

OpenReshot is an iOS-first photo motion app. Contributions should improve the
native app experience, the Core ML conversion path, the Metal renderer, or the
local comparison prototype.

## Good First Areas

- iOS performance and memory profiling.
- Core ML model packaging and first-run setup.
- Metal rendering quality for photo motion and disoccluded edges.
- Gemini enhancement UX and error handling.
- Documentation for building and running on real devices.

## Before Opening a Pull Request

- Keep the user-facing app behavior focused on OpenReshot.
- Do not commit generated models, checkpoints, uploads, derived data, or API keys.
- Run the relevant build or script checks before submitting.
- Keep model attribution and model-license requirements intact when touching the
  model conversion or Python package code.

By submitting a pull request, you confirm that you have the right to license your
contribution under this repository's license.
