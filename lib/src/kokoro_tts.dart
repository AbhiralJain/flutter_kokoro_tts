// kokoro_tts.dart
//
// Pure-Dart Kokoro TTS engine. No Flutter, no ChangeNotifier, no Provider.
// Safe to use in any Dart/Flutter project or as a standalone package.
//
// Expects assets already extracted to a directory you supply via [ttsDir]:
//
//   <ttsDir>/
//     ├── kokoro.onnx          ← ONNX model (int8 or fp32)
//     ├── <voice>.bin          ← style embeddings  (one per voice)
//     └── espeak-ng-data/      ← phonemizer data
//
// Quick start:
//   final tts = KokoroTts(ttsDir: '/path/to/tts');
//   await tts.initialize();
//   final pcm = await tts.speak('Hello world');   // Float32List, 24 kHz mono
//   await tts.dispose();

import 'dart:async';
import 'dart:developer' as dev;
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:path/path.dart' as p;

import 'model_manager.dart';
import 'phonemizer.dart';
import 'text_cleaner.dart';
import 'text_preprocessor.dart';

// ── Voice model ───────────────────────────────────────────────────────────────

/// Describes a single TTS voice.
///
/// [name]      Display / lookup name (e.g. `'Bella'`).
/// [binFile]   Filename of the style-embedding binary inside [KokoroTts.ttsDir]
///             (e.g. `'af_bella.bin'`).
/// [speedPrior] Multiplied with the caller's [speed] before inference.
///              Values below 1.0 slow the voice down; the default of 0.8
///              matches the upstream Kokoro reference.
class KokoroVoice {
  const KokoroVoice({required this.name, required this.binFile, this.speedPrior = 0.8});

  final String name;
  final String binFile;
  final double speedPrior;
}

/// Built-in voices shipped with the reference asset zip.
/// Pass a subset or extend this list via [KokoroTts.voices].
const List<KokoroVoice> defaultVoices = [
  KokoroVoice(name: 'Bella', binFile: 'af_bella.bin'),
  KokoroVoice(name: 'Echo', binFile: 'am_echo.bin'),
];

// ── Engine state ──────────────────────────────────────────────────────────────

/// Lifecycle state of the [KokoroTts] engine.
enum TtsState {
  /// [KokoroTts.initialize] has not been called yet.
  idle,

  /// Engine is loading the ONNX session and phonemizer.
  initializing,

  /// Ready to accept [KokoroTts.speak] / [KokoroTts.speakStreaming] calls.
  ready,

  /// Currently synthesising audio.
  generating,

  /// A fatal error occurred; see [KokoroTts.errorMessage].
  error,
}

// ── Engine ────────────────────────────────────────────────────────────────────

/// Kokoro on-device TTS engine.
///
/// ```dart
/// final tts = KokoroTts(ttsDir: '/data/user/0/com.example/files/tts');
///
/// tts.onStateChanged = (s) => print('state → $s');
///
/// await tts.initialize(onProgress: (p, msg) => print('$p $msg'));
///
/// // Batch (waits for full audio before returning)
/// final pcm = await tts.speak('Hello!');
///
/// // Streaming (yields one sentence at a time)
/// await for (final chunk in tts.speakStreaming('Long text…')) {
///   player.feed(chunk);
/// }
///
/// await tts.dispose();
/// ```
class KokoroTts {
  /// Creates a new engine instance.
  ///
  /// [ttsDir]   Absolute path to the directory containing the model,
  ///            voice `.bin` files, and `espeak-ng-data/`.
  /// [voices]   Voice list; defaults to [defaultVoices].
  /// [modelFile] Model filename inside [ttsDir]; defaults to `'kokoro.onnx'`.
  KokoroTts({required this.ttsDir, List<KokoroVoice>? voices, this.modelFile = 'kokoro.onnx'})
    : voices = voices ?? List.unmodifiable(defaultVoices);

  // ── Configuration ─────────────────────────────────────────────────────────

  /// Absolute path to the directory that holds the model + assets.
  final String ttsDir;

  /// Model filename inside [ttsDir].
  final String modelFile;

  /// Available voices. Populated from the constructor argument.
  final List<KokoroVoice> voices;

  // ── Public state ──────────────────────────────────────────────────────────

  TtsState _state = TtsState.idle;

  /// Current engine state.
  TtsState get state => _state;

  /// Last error message, set when [state] is [TtsState.error].
  String? errorMessage;

  /// Called every time [state] changes.
  /// Use this to drive reactive UI without depending on Provider / Riverpod.
  ///
  /// ```dart
  /// tts.onStateChanged = (s) => setState(() => _ttsState = s);
  /// ```
  void Function(TtsState state)? onStateChanged;

  /// Output sample rate – always 24 000 Hz for Kokoro.
  static const int sampleRate = 24000;

  // ── Private fields ────────────────────────────────────────────────────────

  static const int _maxTokens = 500;
  static const int _styleDim = 256;

  OrtSession? _session;

  final _phonemizer = Phonemizer();
  final _textCleaner = TextCleaner();
  final _textPreprocessor = TextPreprocessor();

  final Map<String, Float32List> _voiceCache = {};

  Future<void>? _initFuture;
  bool _cancelStreaming = false;

  // ── Voice lookup helpers ──────────────────────────────────────────────────

  KokoroVoice? _findVoice(String name) {
    try {
      return voices.firstWhere((v) => v.name == name);
    } catch (_) {
      return null;
    }
  }

  // ── Initialisation ────────────────────────────────────────────────────────

  /// Initialises the engine (idempotent – safe to call multiple times).
  ///
  /// [onProgress] receives values in `[0.0, 1.0]` with a human-readable
  /// status string.  Pass `null` if you don't need progress updates.
  ///
  /// Throws on failure; call [initialize] again to retry.
  Future<void> initialize({void Function(double progress, String status)? onProgress}) {
    if (_state == TtsState.ready) return Future.value();
    _initFuture ??= _doInitialize(onProgress);
    return _initFuture!;
  }

  Future<void> _doInitialize(void Function(double progress, String status)? onProgress) async {
    _setState(TtsState.initializing);
    try {
      onProgress?.call(0.1, 'Locating assets…');
      _assertAssets();

      onProgress?.call(0.4, 'Loading phonemizer…');
      _phonemizer.initialize(dataPath: ttsDir);

      onProgress?.call(0.7, 'Loading ONNX model…');
      final ort = OnnxRuntime();
      _session = await ort.createSession(p.join(ttsDir, modelFile));

      _setState(TtsState.ready);
      onProgress?.call(1.0, 'Ready');
    } catch (e, st) {
      _initFuture = null; // allow retry
      errorMessage = e.toString();
      _setState(TtsState.error);
      dev.log('[KokoroTts] init error: $e\n$st', name: 'KokoroTts');
      rethrow;
    }
  }

  /// Throws a clear [Exception] for each missing required file.
  void _assertAssets() {
    // Model
    _requireFile(p.join(ttsDir, modelFile), 'ONNX model');

    // espeak-ng-data sentinel
    _requireFile(p.join(ttsDir, 'espeak-ng-data', 'phontab'), 'espeak-ng-data/phontab');

    // Voice .bin files
    for (final v in voices) {
      _requireFile(p.join(ttsDir, v.binFile), 'voice "${v.name}"');
    }
  }

  void _requireFile(String path, String label) {
    if (!File(path).existsSync()) {
      throw Exception(
        '[KokoroTts] Missing $label at $path.\n'
        'Ensure the asset zip was extracted to "$ttsDir" before calling initialize().',
      );
    }
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /// Returns true if the TTS files are already downloaded and present in the default location.
  static Future<bool> isDownloaded() async {
    return KokoroModelManager().isReady();
  }

  /// Downloads and extracts the TTS model and assets to the default location.
  static Future<void> downloadAndExtractTts({void Function(double progress)? onProgress}) async {
    await KokoroModelManager().downloadAndExtractTts(onProgress: onProgress);
  }

  /// Returns true if the assets required by this instance exist in [ttsDir].
  bool doesModelExist() {
    try {
      _assertAssets();
      return true;
    } catch (_) {
      return false;
    }
  }

  /// Synthesises [text] and returns the complete audio as a [Float32List].
  ///
  /// Waits for all sentences to be generated before returning.
  /// Use [speakStreaming] if you want audio to start playing sooner.
  ///
  /// - [voice]  voice name from [voices] (defaults to `voices.first.name`).
  /// - [speed]  playback-speed multiplier; combined with the voice's
  ///            [KokoroVoice.speedPrior].
  ///
  /// Returns an empty [Float32List] for blank input.
  /// Throws [StateError] if not [TtsState.ready].
  Future<Float32List> speak(String text, {String? voice, double speed = 1.0}) async {
    final v = _resolveVoice(voice);
    if (text.trim().isEmpty) return Float32List(0);
    _requireReady();

    _setState(TtsState.generating);
    try {
      final audio = await _generate(text, voice: v, speed: speed);
      _setState(TtsState.ready);
      return audio;
    } catch (e) {
      errorMessage = e.toString();
      _setState(TtsState.error);
      rethrow;
    }
  }

  /// Synthesises [text] sentence-by-sentence and yields each [Float32List]
  /// chunk as soon as it is ready.
  ///
  /// The caller can begin playback on the first event while the engine is
  /// still generating subsequent sentences.
  ///
  /// Call [cancelStreaming] to abort between chunks.
  ///
  /// Throws [StateError] if not [TtsState.ready].
  Stream<Float32List> speakStreaming(String text, {String? voice, double speed = 1.0}) async* {
    final v = _resolveVoice(voice);
    if (text.trim().isEmpty) return;
    _requireReady();

    _cancelStreaming = false;
    _setState(TtsState.generating);

    try {
      for (final chunk in _splitIntoChunks(text)) {
        if (_cancelStreaming) {
          dev.log('[KokoroTts] streaming cancelled', name: 'KokoroTts');
          break;
        }
        final audio = await _generateChunk(chunk, voice: v, speed: speed);
        if (audio.isNotEmpty) yield audio;
      }
    } catch (e) {
      errorMessage = e.toString();
      _setState(TtsState.error);
      rethrow;
    } finally {
      if (_state == TtsState.generating) _setState(TtsState.ready);
    }
  }

  /// Stops [speakStreaming] after the current in-flight chunk completes.
  /// No-op if not currently streaming.
  void cancelStreaming() {
    _cancelStreaming = true;
    dev.log('[KokoroTts] cancelStreaming()', name: 'KokoroTts');
  }

  // ── Core synthesis ────────────────────────────────────────────────────────

  Future<Float32List> _generate(String text, {required KokoroVoice voice, required double speed}) async {
    final parts = <Float32List>[];
    for (final chunk in _splitIntoChunks(text)) {
      final audio = await _generateChunk(chunk, voice: voice, speed: speed);
      if (audio.isNotEmpty) parts.add(audio);
    }
    return _concat(parts);
  }

  Future<Float32List> _generateChunk(String chunk, {required KokoroVoice voice, required double speed}) async {
    // ── Text pipeline ───────────────────────────────────────────────────────
    final cleaned = _addTrailingPunctuation(_textPreprocessor.process(chunk).trim());
    dev.log('[KokoroTts] cleaned: $cleaned', name: 'KokoroTts');

    final phonemes = _phonemizer.phonemize(cleaned);
    dev.log('[KokoroTts] phonemes: $phonemes', name: 'KokoroTts');

    final tokens = _textCleaner.encodeAndWrap(phonemes);
    dev.log('[KokoroTts] tokens (${tokens.length})', name: 'KokoroTts');

    // ── Token-limit guard ───────────────────────────────────────────────────
    if (tokens.length > _maxTokens) {
      dev.log('[KokoroTts] bisecting chunk', name: 'KokoroTts');
      final mid = chunk.length ~/ 2;
      int splitAt = chunk.lastIndexOf(RegExp(r'[.!?,;:\s]'), mid);
      if (splitAt <= 0) splitAt = mid;

      final parts = <Float32List>[];
      final first = chunk.substring(0, splitAt).trim();
      final second = chunk.substring(splitAt).trim();
      if (first.isNotEmpty) {
        parts.add(await _generateChunk(first, voice: voice, speed: speed));
      }
      if (second.isNotEmpty) {
        parts.add(await _generateChunk(second, voice: voice, speed: speed));
      }
      return _concat(parts);
    }

    // ── Style vector ────────────────────────────────────────────────────────
    final styleAll = await _loadVoice(voice);
    final numVectors = styleAll.length ~/ _styleDim;
    if (numVectors == 0) {
      throw Exception(
        '[KokoroTts] Voice "${voice.name}" has no style vectors '
        '(expected multiples of $_styleDim floats in ${voice.binFile}).',
      );
    }
    final row = tokens.length.clamp(0, numVectors - 1);
    final style = Float32List.sublistView(styleAll, row * _styleDim, (row + 1) * _styleDim);

    // ── ONNX inference ──────────────────────────────────────────────────────
    final effectiveSpeed = speed * voice.speedPrior;

    final inputIds = await OrtValue.fromList(Int64List.fromList(tokens), [1, tokens.length]);
    final styleTensor = await OrtValue.fromList(style, [1, _styleDim]);
    final speedTensor = await OrtValue.fromList([effectiveSpeed], [1]);

    final outputs = await _session!.run({'input_ids': inputIds, 'style': styleTensor, 'speed': speedTensor});

    await inputIds.dispose();
    await styleTensor.dispose();
    await speedTensor.dispose();

    // ── Extract audio tensor (largest output) ───────────────────────────────
    OrtValue? audioTensor;
    int maxLen = 0;
    for (final v in outputs.values) {
      final len = v.shape.fold<int>(1, (a, b) => a * b);
      if (len > maxLen) {
        maxLen = len;
        audioTensor = v;
      }
    }

    if (audioTensor == null) {
      for (final v in outputs.values) {
        await v.dispose();
      }
      return Float32List(0);
    }

    final rawList = await audioTensor.asFlattenedList();
    for (final v in outputs.values) {
      await v.dispose();
    }

    // ── Normalise to Float32List ─────────────────────────────────────────────
    Float32List audio;
    if (rawList is Float32List) {
      audio = rawList;
    } else if (rawList is List<double>) {
      audio = Float32List.fromList(rawList);
    } else {
      audio = Float32List.fromList((rawList).map((e) => (e as num).toDouble()).toList());
    }

    // Trim trailing silence padded by the model
    if (audio.length > 5000) {
      audio = Float32List.sublistView(audio, 0, audio.length - 5000);
    }

    dev.log('[KokoroTts] generated ${audio.length} samples', name: 'KokoroTts');
    return audio;
  }

  // ── Voice loading ─────────────────────────────────────────────────────────

  Future<Float32List> _loadVoice(KokoroVoice voice) async {
    if (_voiceCache.containsKey(voice.name)) return _voiceCache[voice.name]!;
    final path = p.join(ttsDir, voice.binFile);
    final bytes = await File(path).readAsBytes();
    final embedding = bytes.buffer.asFloat32List();
    _voiceCache[voice.name] = embedding;
    dev.log(
      '[KokoroTts] loaded "${voice.name}" – '
      '${embedding.length ~/ _styleDim} style vectors',
      name: 'KokoroTts',
    );
    return embedding;
  }

  // ── Text helpers ──────────────────────────────────────────────────────────

  List<String> _splitIntoChunks(String text, {int maxLen = 200}) {
    final sentences = text.split(RegExp(r'(?<=[.!?])\s+'));
    final chunks = <String>[];
    final buf = StringBuffer();

    for (final s in sentences) {
      final trimmed = s.trim();
      if (trimmed.isEmpty) continue;
      if (buf.length + trimmed.length + 1 > maxLen && buf.isNotEmpty) {
        chunks.add(_addTrailingPunctuation(buf.toString().trim()));
        buf.clear();
      }
      if (buf.isNotEmpty) buf.write(' ');
      buf.write(trimmed);
    }
    if (buf.isNotEmpty) {
      chunks.add(_addTrailingPunctuation(buf.toString().trim()));
    }
    return chunks.isEmpty ? [_addTrailingPunctuation(text.trim())] : chunks;
  }

  static String _addTrailingPunctuation(String t) {
    t = t.trim();
    if (t.isEmpty) return t;
    return '.!?,;:'.contains(t[t.length - 1]) ? t : '$t,';
  }

  static Float32List _concat(List<Float32List> parts) {
    if (parts.isEmpty) return Float32List(0);
    if (parts.length == 1) return parts.first;
    final total = parts.fold<int>(0, (sum, p) => sum + p.length);
    final result = Float32List(total);
    var offset = 0;
    for (final part in parts) {
      result.setRange(offset, offset + part.length, part);
      offset += part.length;
    }
    return result;
  }

  // ── Guards ────────────────────────────────────────────────────────────────

  KokoroVoice _resolveVoice(String? name) {
    final target = name ?? voices.first.name;
    final v = _findVoice(target);
    if (v == null) {
      throw ArgumentError(
        '[KokoroTts] Unknown voice "$target". '
        'Available: ${voices.map((v) => v.name).join(', ')}',
      );
    }
    return v;
  }

  void _requireReady() {
    if (_state != TtsState.ready) {
      throw StateError(
        '[KokoroTts] Engine is not ready (state: $_state). '
        'Call initialize() and await it first.',
      );
    }
  }

  // ── State management ──────────────────────────────────────────────────────

  void _setState(TtsState s) {
    _state = s;
    onStateChanged?.call(s);
  }

  // ── Disposal ──────────────────────────────────────────────────────────────

  /// Releases the ONNX session, phonemizer, and voice cache.
  /// The instance must not be used after this call.
  Future<void> dispose() async {
    await _session?.close();
    _session = null;
    _phonemizer.dispose();
    _voiceCache.clear();
    _state = TtsState.idle;
    _initFuture = null;
    onStateChanged = null;
  }
}
