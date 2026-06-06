import 'dart:io';

import 'package:path/path.dart' as p;

import 'espeak_ffi.dart';

/// Convert English text to IPA phonemes using espeak-ng via DArt FFI.

class Phonemizer {
  final EspeakFfi _espeak = EspeakFfi();

  bool get isInitialized => _espeak.isReady;

  /// Initialize espeak-ng. [dataPath] must be either:
  /// - The directory that contains an "espeak-ng-data" subfolder (with phontab, etc.), or
  /// - The espeak-ng-data directory itself (with phontab, phonindex, phondata inside).
  /// Throws if the required data files are not found (avoids native crash).
  void initialize({required String dataPath}) {
    if (dataPath.isEmpty) {
      throw Exception(
        'espeak-ng data path is empty. Provide a path to a directory that '
        'contains espeak-ng-data (with phontab, phonindex, phondata). '
        'See package README for how to obtain or bundle espeak-ng-data.',
      );
    }
    final dir = Directory(dataPath);
    if (!dir.existsSync()) {
      throw Exception(
        'espeak-ng data path does not exist: $dataPath. '
        'Bundle or download espeak-ng-data and pass its parent directory.',
      );
    }
    final withSubdir = File(p.join(dir.path, 'espeak-ng-data', 'phontab'));
    final direct = File(p.join(dir.path, 'phontab'));
    if (!withSubdir.existsSync() && !direct.existsSync()) {
      throw Exception(
        'espeak-ng data not found at $dataPath (missing phontab). '
        'Expected either "$dataPath/espeak-ng-data/phontab" or "$dataPath/phontab". '
        'Obtain espeak-ng-data from https://github.com/espeak-ng/espeak-ng (build from source) '
        'or pass a valid espeakDataPath to KokoroTts.initialize().',
      );
    }
    _espeak.load();
    _espeak.init(dataPath: dataPath);
  }

  // Convert text to IPA phonemes
  String phonemize(String text) {
    if (!isInitialized) throw Exception('Espeak-ng not initialized');

    // 1. Normalize hyphens/dashes that are acting as punctuation to em-dash (—)
    text = text.replaceAll(RegExp(r'\s+-\s+'), ' — ');
    text = text.replaceAll(RegExp(r'(?<=\w)-\s+'), ' — ');
    text = text.replaceAll(RegExp(r'\s+-(?=\w)'), ' — ');

    // 2. Split the text by punctuation marks that we want to preserve (including surrounding whitespace)
    final regex = RegExp(r'(\s*[;:,.!?¡¿—…"«»“”]+\s*)');
    final matches = regex.allMatches(text).toList();

    if (matches.isEmpty) {
      return _espeak.phonemize(text);
    }

    final buffer = StringBuffer();
    var lastIndex = 0;

    for (final match in matches) {
      final textSegment = text.substring(lastIndex, match.start);
      final punctuation = match.group(0)!;

      if (textSegment.isNotEmpty) {
        final phonemes = _espeak.phonemize(textSegment);
        buffer.write(phonemes);
      }
      buffer.write(punctuation);
      lastIndex = match.end;
    }

    if (lastIndex < text.length) {
      final remainingText = text.substring(lastIndex);
      if (remainingText.isNotEmpty) {
        final phonemes = _espeak.phonemize(remainingText);
        buffer.write(phonemes);
      }
    }

    return buffer.toString();
  }

  // Dispose of espeak-ng
  void dispose() {
    _espeak.dispose();
  }
}
