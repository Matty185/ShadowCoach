import 'dart:math';
import '../models/models.dart';
import '../storage/database_service.dart';

/// Service for seeding mock session data for demo purposes.
/// Generates 7 sessions showing progressive improvement over a week.
class MockDataService {
  static final _random = Random(42); // Fixed seed for reproducibility

  /// Seeds 7 sample sessions only if the database is empty.
  static Future<void> seedIfEmpty() async {
    final count = await DatabaseService.instance.getSessionCount();
    if (count == 0) {
      await seedSampleSessions();
    }
  }

  /// Clears the database and seeds 7 fresh sample sessions.
  static Future<void> seedSampleSessions() async {
    await DatabaseService.instance.clearAll();

    final now = DateTime.now();
    final sessionConfigs = [
      _SessionConfig(daysAgo: 7, punches: 10, avgSpeed: 1.8, ppm: 18, duration: 55, activityPct: 0.35, fatigue: 'declining'),
      _SessionConfig(daysAgo: 6, punches: 13, avgSpeed: 2.1, ppm: 22, duration: 60, activityPct: 0.40, fatigue: 'declining'),
      _SessionConfig(daysAgo: 5, punches: 16, avgSpeed: 2.4, ppm: 28, duration: 65, activityPct: 0.45, fatigue: 'stable'),
      _SessionConfig(daysAgo: 4, punches: 19, avgSpeed: 2.7, ppm: 32, duration: 70, activityPct: 0.50, fatigue: 'stable'),
      _SessionConfig(daysAgo: 3, punches: 23, avgSpeed: 3.0, ppm: 38, duration: 75, activityPct: 0.55, fatigue: 'stable'),
      _SessionConfig(daysAgo: 2, punches: 27, avgSpeed: 3.3, ppm: 42, duration: 80, activityPct: 0.60, fatigue: 'improving'),
      _SessionConfig(daysAgo: 1, punches: 30, avgSpeed: 3.5, ppm: 48, duration: 85, activityPct: 0.65, fatigue: 'improving'),
    ];

    for (final config in sessionConfigs) {
      final timestamp = now.subtract(Duration(days: config.daysAgo));
      final session = _buildSession(config, timestamp);
      await DatabaseService.instance.saveSession(session);
    }

    print('[MockData] Seeded ${sessionConfigs.length} sample sessions');
  }

  static SessionData _buildSession(_SessionConfig c, DateTime timestamp) {
    final maxSpeed = c.avgSpeed * 1.3;
    final minSpeed = c.avgSpeed * 0.7;
    final speedVariance = (maxSpeed - minSpeed) / 2;
    final activeTime = c.duration * c.activityPct;
    final restTime = c.duration * (1 - c.activityPct);

    final punchEvents = _generatePunchEvents(c);
    final speedOverTime = _generateSpeedOverTime(c);
    final leftCount = (c.punches * 0.6).round();
    final rightCount = c.punches - leftCount;

    // Intensity score: normalized 0-100 based on PPM and speed
    final intensityScore = ((c.ppm / 60) * 50 + (c.avgSpeed / 5) * 50).clamp(0, 100).toDouble();

    // Punches by 15-second intervals
    final intervalCount = (c.duration / 15).ceil();
    final punchesByInterval = _distributePunches(c.punches, intervalCount);
    final peakIdx = punchesByInterval.indexOf(punchesByInterval.reduce(max));

    // Fatigue indicator: 0.0 (no fatigue) to 1.0 (severe fatigue)
    final fatigueIndicator = c.fatigue == 'declining'
        ? 0.7
        : c.fatigue == 'stable'
            ? 0.4
            : 0.2;

    final metrics = SessionMetrics(
      totalPunches: c.punches,
      averageSpeed: c.avgSpeed,
      punchesPerMinute: c.ppm.toDouble(),
      punchEvents: punchEvents,
      graphs: GraphData.fromJson({
        'speed_over_time': speedOverTime,
        'hand_distribution': {'left': leftCount, 'right': rightCount},
      }),
      sessionDuration: c.duration.toDouble(),
      performance: PerformanceMetrics(
        maxSpeed: maxSpeed,
        minSpeed: minSpeed,
        speedVariance: speedVariance,
        averagePunchDuration: 0.3 + (_random.nextDouble() * 0.2),
        fastestPunch: PunchDetail(
          index: _random.nextInt(c.punches),
          speed: maxSpeed,
          time: c.duration * 0.3,
          duration: 0.25,
        ),
        slowestPunch: PunchDetail(
          index: _random.nextInt(c.punches),
          speed: minSpeed,
          time: c.duration * 0.8,
          duration: 0.5,
        ),
      ),
      activity: ActivityMetrics(
        activeTime: activeTime,
        restTime: restTime,
        activityPercentage: c.activityPct * 100,
        averageRestPeriod: restTime / (c.punches + 1),
        maxRestPeriod: restTime / (c.punches + 1) * 2,
        minRestPeriod: restTime / (c.punches + 1) * 0.5,
      ),
      intensity: IntensityMetrics(
        score: intensityScore,
        punchesByInterval: punchesByInterval,
        peakInterval: PeakInterval(
          interval: peakIdx,
          punches: punchesByInterval[peakIdx],
        ),
      ),
      fatigue: FatigueMetrics(
        indicator: fatigueIndicator,
        speedTrend: c.fatigue,
      ),
    );

    return SessionData(
      timestamp: timestamp,
      fileName: 'session_day${8 - c.daysAgo}.mp4',
      metrics: metrics,
      videoDuration: c.duration.toDouble(),
    );
  }

  static List<PunchEvent> _generatePunchEvents(_SessionConfig c) {
    final events = <PunchEvent>[];
    final interval = c.duration / c.punches;

    for (int i = 0; i < c.punches; i++) {
      final startTime = i * interval + (_random.nextDouble() * interval * 0.3);
      final speed = c.avgSpeed + ((_random.nextDouble() - 0.5) * c.avgSpeed * 0.6);
      final hand = i % 5 < 3 ? 'left' : 'right'; // ~60/40 split
      final punchType = _random.nextBool() ? 'jab' : 'punch';

      events.add(PunchEvent(
        index: i,
        speed: speed.clamp(c.avgSpeed * 0.5, c.avgSpeed * 1.5),
        hand: hand,
        startTime: startTime,
        endTime: startTime + 0.2 + (_random.nextDouble() * 0.3),
        punchType: punchType,
      ));
    }

    return events;
  }

  static List<double> _generateSpeedOverTime(_SessionConfig c) {
    final points = <double>[];
    final count = c.punches;
    for (int i = 0; i < count; i++) {
      final base = c.avgSpeed;
      final variation = (_random.nextDouble() - 0.5) * base * 0.4;
      // Add slight fatigue trend for declining sessions
      final fatigueDrop = c.fatigue == 'declining' ? (i / count) * 0.3 : 0.0;
      points.add((base + variation - fatigueDrop).clamp(0.5, 6.0));
    }
    return points;
  }

  static List<int> _distributePunches(int total, int intervals) {
    if (intervals <= 0) return [total];
    final base = total ~/ intervals;
    final remainder = total % intervals;
    final result = List.filled(intervals, base);
    // Distribute remainder across early intervals
    for (int i = 0; i < remainder; i++) {
      result[i]++;
    }
    // Shuffle slightly for realism
    for (int i = result.length - 1; i > 0; i--) {
      if (_random.nextBool() && i > 0) {
        final temp = result[i];
        result[i] = result[i - 1];
        result[i - 1] = temp;
      }
    }
    return result;
  }
}

class _SessionConfig {
  final int daysAgo;
  final int punches;
  final double avgSpeed;
  final int ppm;
  final int duration;
  final double activityPct;
  final String fatigue;

  _SessionConfig({
    required this.daysAgo,
    required this.punches,
    required this.avgSpeed,
    required this.ppm,
    required this.duration,
    required this.activityPct,
    required this.fatigue,
  });
}
