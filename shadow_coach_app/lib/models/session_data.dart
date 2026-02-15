import 'dart:convert';
import 'session_metrics.dart';

/// Complete session data including metadata and analysis results
/// Used for storing and retrieving sessions from local database
class SessionData {
  final int? id; // SQLite auto-increment ID (null for new sessions)
  final DateTime timestamp;
  final String fileName;
  final SessionMetrics metrics;
  final double? videoDuration; // Optional video duration in seconds

  SessionData({
    this.id,
    required this.timestamp,
    required this.fileName,
    required this.metrics,
    this.videoDuration,
  });

  /// Create SessionData from database row
  factory SessionData.fromMap(Map<String, dynamic> map) {
    return SessionData(
      id: map['id'] as int?,
      timestamp: DateTime.parse(map['timestamp'] as String),
      fileName: map['file_name'] as String,
      metrics: SessionMetrics.fromJson(
        jsonDecode(map['metrics_json'] as String) as Map<String, dynamic>,
      ),
      videoDuration: map['video_duration'] as double?,
    );
  }

  /// Convert SessionData to database map
  Map<String, dynamic> toMap() {
    return {
      if (id != null) 'id': id,
      'timestamp': timestamp.toIso8601String(),
      'file_name': fileName,
      'metrics_json': jsonEncode(metrics.toJson()),
      'video_duration': videoDuration,
    };
  }

  /// Create a copy with updated fields
  SessionData copyWith({
    int? id,
    DateTime? timestamp,
    String? fileName,
    SessionMetrics? metrics,
    double? videoDuration,
  }) {
    return SessionData(
      id: id ?? this.id,
      timestamp: timestamp ?? this.timestamp,
      fileName: fileName ?? this.fileName,
      metrics: metrics ?? this.metrics,
      videoDuration: videoDuration ?? this.videoDuration,
    );
  }

  /// Get formatted date string for display
  String get formattedDate {
    final now = DateTime.now();
    final difference = now.difference(timestamp);

    if (difference.inDays == 0) {
      return 'Today ${timestamp.hour}:${timestamp.minute.toString().padLeft(2, '0')}';
    } else if (difference.inDays == 1) {
      return 'Yesterday';
    } else if (difference.inDays < 7) {
      return '${difference.inDays} days ago';
    } else {
      return '${timestamp.day}/${timestamp.month}/${timestamp.year}';
    }
  }

  /// Get session intensity rating (basic heuristic)
  String get intensityRating {
    if (metrics.punchesPerMinute >= 60) return 'High';
    if (metrics.punchesPerMinute >= 40) return 'Medium';
    return 'Low';
  }

  @override
  String toString() {
    return 'SessionData(id: $id, date: $formattedDate, file: $fileName, punches: ${metrics.totalPunches})';
  }
}
