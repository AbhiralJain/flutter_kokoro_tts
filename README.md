# flutter_kokoro_tts

A Flutter plugin for **Kokoro TTS** (text-to-speech) using ONNX Runtime and espeak-ng. Generates high-quality speech from text with multiple voices, running fully on-device (no cloud API required).

## Features

- **On-device inference** — ONNX model runs locally; no internet after initial model download.
- **Dynamic model management** — Check download status and perform downloading/extracting directly through the API.
- **Batch & Streaming synthesis** — Speak standard sentences or stream audio chunks sentence-by-sentence.
- **Multiple voices** — Supports pre-packaged voices (e.g., `Bella`, `Echo`).
- **Configurable speed** — Adjust speech rate per call.
- **24 kHz output** — Mono PCM at 24,000 Hz (Float32).
- **Android & iOS** — Supporting both platforms with native espeak-ng linking.

---

## Installation

Since this package is hosted on your GitHub, add it to your `pubspec.yaml` using the Git dependency format:

```yaml
dependencies:
  flutter_kokoro_tts:
    git:
      url: https://github.com/AbhiralJain/flutter_kokoro_tts.git
      ref: main
  path_provider: ^2.1.5  # Required to get the application documents directory
  path: ^1.9.1
```

Then run:

```bash
flutter pub get
```

### Platform Requirements

- **Flutter** — SDK 3.10+
- **Android** — minSdk 21+ (plugin uses NDK/CMake for native code). Install **CMake** and **NDK** via Android Studio if needed.
- **iOS** — iOS 11.0+

---

## Step-by-step Usage

### Step 1: Import the Package

```dart
import 'package:flutter_kokoro_tts/flutter_kokoro_tts.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
```

### Step 2: Check & Download Model Assets

Before using `KokoroTts`, ensure the required model and voice assets (~50 MB zip) are downloaded and extracted to the local device:

```dart
// Check if the assets are already downloaded
final bool downloaded = await KokoroTts.isDownloaded();

if (!downloaded) {
  // Download and extract model assets (from R2 R2.dev mirror)
  await KokoroTts.downloadAndExtractTts(
    onProgress: (progress) {
      // progress: 0.0 to 1.0
      print('Download/Extraction Progress: ${(progress * 100).toStringAsFixed(1)}%');
    },
  );
}
```

### Step 3: Instantiate the Engine

Instantiate the `KokoroTts` engine by passing the directory path where assets were extracted (the default download location is `getApplicationDocumentsDirectory() / tts`):

```dart
final Directory appDocDir = await getApplicationDocumentsDirectory();
final String ttsDir = p.join(appDocDir.path, 'tts');

final tts = KokoroTts(ttsDir: ttsDir);
```

You can also check if the configured path contains the files using:
```dart
bool exists = tts.doesModelExist();
```

### Step 4: Initialize the Engine

Initialize the engine to load the ONNX session and espeak phonemizer:

```dart
// Monitor engine state changes
tts.onStateChanged = (state) {
  print('State changed to: $state'); // TtsState.idle, initializing, ready, generating, error
};

await tts.initialize(
  onProgress: (progress, status) {
    print('Initialization: $status - ${(progress * 100).round()}%');
  },
);
```

### Step 5: Generate Speech

#### Batch Speech (Waits for full audio)
```dart
final Float32List audio = await tts.speak(
  'Hello! This is Kokoro running fully on device.',
  voice: 'Bella', // E.g., 'Bella' or 'Echo'
  speed: 1.0,     // Playback-speed multiplier
);
// Output is Float32List, mono, 24,000 Hz (KokoroTts.sampleRate)
```

#### Streaming Speech (Sentence-by-sentence chunks)
```dart
final stream = tts.speakStreaming(
  'This is a longer text. It will generate audio sentence by sentence. You can start playing the first chunk immediately.',
  voice: 'Bella',
  speed: 1.0,
);

await for (final Float32List chunk in stream) {
  // Feed audio chunk to your player
}
```
You can abort streaming between chunks by calling `tts.cancelStreaming()`.

### Step 6: Dispose when Done

```dart
await tts.dispose();
```

---

## Configuration & Customization

### Available Voices

Default configured voices:
- `Bella` (Female, style file: `af_bella.bin`)
- `Echo` (Male, style file: `am_echo.bin`)

You can supply custom voices via the `KokoroTts` constructor:
```dart
final tts = KokoroTts(
  ttsDir: ttsDir,
  voices: [
    KokoroVoice(name: 'Bella', binFile: 'af_bella.bin'),
    KokoroVoice(name: 'Echo', binFile: 'am_echo.bin'),
  ],
);
```

### Engine Lifecycle States (`TtsState`)

- `TtsState.idle`: `initialize` has not been called yet.
- `TtsState.initializing`: ONNX session and phonemizer are being loaded.
- `TtsState.ready`: Ready to speak or stream.
- `TtsState.generating`: Currently synthesizing audio.
- `TtsState.error`: A fatal error occurred (check `tts.errorMessage`).

---

## License

MIT. See [LICENSE](LICENSE).
