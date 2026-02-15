import 'package:hive_flutter/hive_flutter.dart';
import '../models/models.dart';

/// Database service using Hive for storing and retrieving shadowboxing sessions
class DatabaseService {
  static final DatabaseService instance = DatabaseService._init();

  static const String _sessionsBoxName = 'sessions';
  Box<Map>? _sessionsBox;
  int _nextId = 1;

  DatabaseService._init();

  /// Initialize Hive database
  Future<void> init() async {
    await Hive.initFlutter();
    _sessionsBox = await Hive.openBox<Map>(_sessionsBoxName);

    // Initialize the next ID counter
    if (_sessionsBox!.isNotEmpty) {
      final keys = _sessionsBox!.keys.cast<int>();
      _nextId = keys.reduce((a, b) => a > b ? a : b) + 1;
    }

    print('[DB] Hive initialized with ${_sessionsBox!.length} sessions');
  }

  Box<Map> get _sessions {
    if (_sessionsBox == null || !_sessionsBox!.isOpen) {
      throw Exception('Database not initialized. Call init() first.');
    }
    return _sessionsBox!;
  }

  /// Save a session to the database
  Future<int> saveSession(SessionData session) async {
    final id = _nextId++;
    final data = session.toMap();
    data['id'] = id;

    await _sessions.put(id, data);
    print('[DB] Session saved with ID: $id');
    return id;
  }

  /// Get all sessions ordered by timestamp (newest first)
  Future<List<SessionData>> getAllSessions() async {
    final sessions = <SessionData>[];

    for (var key in _sessions.keys) {
      final data = Map<String, dynamic>.from(_sessions.get(key)!);
      sessions.add(SessionData.fromMap(data));
    }

    // Sort by timestamp descending (newest first)
    sessions.sort((a, b) => b.timestamp.compareTo(a.timestamp));

    return sessions;
  }

  /// Get a single session by ID
  Future<SessionData?> getSession(int id) async {
    final data = _sessions.get(id);

    if (data != null) {
      return SessionData.fromMap(Map<String, dynamic>.from(data));
    } else {
      return null;
    }
  }

  /// Delete a session by ID
  Future<void> deleteSession(int id) async {
    await _sessions.delete(id);
    print('[DB] Session $id deleted');
  }

  /// Get sessions count
  Future<int> getSessionCount() async {
    return _sessions.length;
  }

  /// Get total punches across all sessions
  Future<int> getTotalPunches() async {
    int total = 0;

    for (var key in _sessions.keys) {
      final data = Map<String, dynamic>.from(_sessions.get(key)!);
      total += (data['total_punches'] as int?) ?? 0;
    }

    return total;
  }

  /// Clear all sessions (for testing)
  Future<void> clearAll() async {
    await _sessions.clear();
    _nextId = 1;
    print('[DB] All sessions cleared');
  }

  /// Close the database
  Future<void> close() async {
    await _sessionsBox?.close();
  }
}
