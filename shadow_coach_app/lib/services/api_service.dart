import 'dart:io';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../models/models.dart';

/// Service class for communicating with Shadow Coach REST API
class ApiService {
  // API base URL (change to your computer's IP for physical device testing)
  static const String baseUrl = 'http://localhost:5000';
  // For physical devices on same network, use: 'http://192.168.0.70:5000'

  /// Check if API server is healthy
  Future<bool> healthCheck() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/health'),
      ).timeout(const Duration(seconds: 5));

      return response.statusCode == 200;
    } catch (e) {
      print('[API] Health check failed: $e');
      return false;
    }
  }

  /// Analyze a video file from bytes (for web support)
  ///
  /// [videoBytes] - The video file bytes
  /// [fileName] - Name of the video file
  /// Returns [SessionMetrics] with detection results
  /// Throws [ApiException] if analysis fails
  Future<SessionMetrics> analyzeVideoBytes(Uint8List videoBytes, String fileName) async {
    try {
      print('[API] Uploading video: $fileName');
      print('[API] File size: ${videoBytes.length / (1024 * 1024)} MB');

      // Create multipart request
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/analyze'),
      );

      // Add video file from bytes
      request.files.add(
        http.MultipartFile.fromBytes(
          'video',
          videoBytes,
          filename: fileName,
        ),
      );

      print('[API] Sending request...');

      // Send request with timeout
      var streamedResponse = await request.send().timeout(
        const Duration(seconds: 60),
        onTimeout: () {
          throw ApiException('Request timed out after 60 seconds');
        },
      );

      // Get response
      var response = await http.Response.fromStream(streamedResponse);

      print('[API] Response status: ${response.statusCode}');

      if (response.statusCode == 200) {
        // Parse JSON response
        final jsonData = json.decode(response.body) as Map<String, dynamic>;
        print('[API] Analysis complete: ${jsonData['total_punches']} punches detected');

        // Convert to SessionMetrics
        return SessionMetrics.fromJson(jsonData);
      } else {
        // Error response
        final error = json.decode(response.body);
        throw ApiException(error['error'] ?? 'Unknown error');
      }
    } on SocketException {
      throw ApiException(
        'Cannot connect to server. Make sure the API is running at $baseUrl'
      );
    } on FormatException {
      throw ApiException('Invalid response from server');
    } catch (e) {
      if (e is ApiException) rethrow;
      throw ApiException('Analysis failed: ${e.toString()}');
    }
  }

  /// Analyze a video file and get jab detection results
  ///
  /// [videoFile] - The video file to analyze
  /// Returns [SessionMetrics] with detection results
  /// Throws [ApiException] if analysis fails
  Future<SessionMetrics> analyzeVideo(File videoFile) async {
    try {
      print('[API] Uploading video: ${videoFile.path}');
      print('[API] File size: ${(await videoFile.length()) / (1024 * 1024)} MB');

      // Create multipart request
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/analyze'),
      );

      // Add video file
      request.files.add(
        await http.MultipartFile.fromPath(
          'video',
          videoFile.path,
        ),
      );

      print('[API] Sending request...');

      // Send request with timeout
      var streamedResponse = await request.send().timeout(
        const Duration(seconds: 60),
        onTimeout: () {
          throw ApiException('Request timed out after 60 seconds');
        },
      );

      // Get response
      var response = await http.Response.fromStream(streamedResponse);

      print('[API] Response status: ${response.statusCode}');

      if (response.statusCode == 200) {
        // Parse JSON response
        final jsonData = json.decode(response.body) as Map<String, dynamic>;
        print('[API] Analysis complete: ${jsonData['total_punches']} punches detected');

        // Convert to SessionMetrics
        return SessionMetrics.fromJson(jsonData);
      } else {
        // Error response
        final error = json.decode(response.body);
        throw ApiException(error['error'] ?? 'Unknown error');
      }
    } on SocketException {
      throw ApiException(
        'Cannot connect to server. Make sure the API is running at $baseUrl'
      );
    } on FormatException {
      throw ApiException('Invalid response from server');
    } catch (e) {
      if (e is ApiException) rethrow;
      throw ApiException('Analysis failed: ${e.toString()}');
    }
  }

  /// Get API information
  Future<Map<String, dynamic>> getApiInfo() async {
    try {
      final response = await http.get(
        Uri.parse(baseUrl),
      ).timeout(const Duration(seconds: 5));

      if (response.statusCode == 200) {
        return json.decode(response.body) as Map<String, dynamic>;
      } else {
        throw ApiException('Failed to get API info');
      }
    } catch (e) {
      throw ApiException('Cannot connect to API: ${e.toString()}');
    }
  }
}

/// Custom exception for API errors
class ApiException implements Exception {
  final String message;

  ApiException(this.message);

  @override
  String toString() => message;
}
