import 'dart:convert';
import 'package:http/http.dart' as http;

class CursorApi {
  final String baseUrl;

  CursorApi({this.baseUrl = 'https://nhrsxv6h-8000.use.devtunnels.ms'});

  Future<http.Response> startTranscription({
    bool diarization = true,
    int speakers = 2,
    bool screen = false,
    bool debug = false,
  }) async {
    final url = '$baseUrl/start';

    final payload = {
      'diarization': diarization,
      'speakers': speakers,
      'screen': screen,
      'debug': debug,
    };

    try {
      final response = await http.post(
        Uri.parse(url),
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode(payload),
      );

      return response;
    } catch (e) {
      throw Exception('Failed to stop transcription: $e');
    }
  }

  Future<http.Response> stopTranscription({
    bool diarization = true,
    int speakers = 2,
    bool screen = false,
    bool debug = false,
  }) async {
    final url = '$baseUrl/stop';

    final payload = {
      'diarization': diarization,
      'speakers': speakers,
      'screen': screen,
      'debug': debug,
    };

    try {
      final response = await http.post(
        Uri.parse(url),
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode(payload),
      );

      return response;
    } catch (e) {
      throw Exception('Failed to stop transcription: $e');
    }
  }
}
