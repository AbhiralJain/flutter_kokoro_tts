import 'dart:io';

import 'package:archive/archive_io.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

class TtsDownloadException implements Exception {
  const TtsDownloadException(this.message, {this.cause});
  final String message;
  final Object? cause;

  @override
  String toString() =>
      cause != null ? 'TtsDownloadException: $message (caused by: $cause)' : 'TtsDownloadException: $message';
}

class TtsExtractionException implements Exception {
  const TtsExtractionException(this.message, {this.cause});
  final String message;
  final Object? cause;

  @override
  String toString() =>
      cause != null ? 'TtsExtractionException: $message (caused by: $cause)' : 'TtsExtractionException: $message';
}

class KokoroModelManager {
  static const String _ttsZipUrl = 'https://pub-b78a71a7a6ef4108ab104ae6f3b1d556.r2.dev/tts.zip';
  static const String _ttsDirName = 'tts';
  static const String _tempZipName = 'temp_tts.zip';

  /// Returns true when TTS files are already present and ready to use.
  Future<bool> isReady() async {
    final Directory appDocDir = await getApplicationDocumentsDirectory();
    final Directory ttsDir = Directory(p.join(appDocDir.path, _ttsDirName));
    return ttsDir.existsSync() && ttsDir.listSync().isNotEmpty;
  }

  Future<void> downloadAndExtractTts({void Function(double progress)? onProgress}) async {
    if (await isReady()) {
      onProgress?.call(1.0);
      return;
    }

    final Directory appDocDir = await getApplicationDocumentsDirectory();
    final Directory ttsDir = Directory(p.join(appDocDir.path, _ttsDirName));
    final File tempZipFile = File(p.join(appDocDir.path, _tempZipName));

    // Clean up any stale temp file left over from a previous crashed run.
    if (tempZipFile.existsSync()) {
      await tempZipFile.delete();
    }

    try {
      await _download(tempZipFile: tempZipFile, onProgress: onProgress);
      await _extract(tempZipFile: tempZipFile, ttsDir: ttsDir, onProgress: onProgress);
    } finally {
      // Always clean up the temp file — even if extraction throws.
      if (tempZipFile.existsSync()) {
        await tempZipFile.delete();
      }
    }
  }

  Future<void> _download({required File tempZipFile, void Function(double)? onProgress}) async {
    final http.Client client = http.Client();

    try {
      final http.StreamedResponse response;
      try {
        final http.Request request = http.Request('GET', Uri.parse(_ttsZipUrl));
        response = await client.send(request);
      } catch (e) {
        throw TtsDownloadException('Network request failed.', cause: e);
      }

      if (response.statusCode != 200) {
        throw TtsDownloadException('Unexpected HTTP status: ${response.statusCode}.');
      }

      final int totalBytes = response.contentLength ?? 0;
      int receivedBytes = 0;

      final IOSink sink = tempZipFile.openWrite();
      try {
        await for (final List<int> chunk in response.stream) {
          sink.add(chunk);
          receivedBytes += chunk.length;

          if (totalBytes > 0) {
            // Download accounts for ~90% of overall time.
            onProgress?.call((receivedBytes / totalBytes) * 0.9);
          }
        }
        await sink.flush();
      } catch (e) {
        throw TtsDownloadException('Download stream interrupted.', cause: e);
      } finally {
        await sink.close();
      }
    } finally {
      client.close();
    }
  }

  Future<void> _extract({
    required File tempZipFile,
    required Directory ttsDir,
    void Function(double)? onProgress,
  }) async {
    try {
      onProgress?.call(0.92);
      await extractFileToDisk(tempZipFile.path, ttsDir.path);
      onProgress?.call(1.0);
    } catch (e) {
      throw TtsExtractionException('Failed to extract ZIP file.', cause: e);
    }
  }
}
