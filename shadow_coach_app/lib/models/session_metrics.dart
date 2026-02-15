import 'punch_event.dart';

/// Performance metrics for punch quality analysis
class PerformanceMetrics {
  final double maxSpeed;
  final double minSpeed;
  final double speedVariance;
  final double averagePunchDuration;
  final PunchDetail? fastestPunch;
  final PunchDetail? slowestPunch;

  PerformanceMetrics({
    required this.maxSpeed,
    required this.minSpeed,
    required this.speedVariance,
    required this.averagePunchDuration,
    this.fastestPunch,
    this.slowestPunch,
  });

  factory PerformanceMetrics.fromJson(Map<String, dynamic> json) {
    return PerformanceMetrics(
      maxSpeed: (json['max_speed'] as num?)?.toDouble() ?? 0.0,
      minSpeed: (json['min_speed'] as num?)?.toDouble() ?? 0.0,
      speedVariance: (json['speed_variance'] as num?)?.toDouble() ?? 0.0,
      averagePunchDuration: (json['average_punch_duration'] as num?)?.toDouble() ?? 0.0,
      fastestPunch: json['fastest_punch'] != null
          ? PunchDetail.fromJson(json['fastest_punch'])
          : null,
      slowestPunch: json['slowest_punch'] != null
          ? PunchDetail.fromJson(json['slowest_punch'])
          : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'max_speed': maxSpeed,
      'min_speed': minSpeed,
      'speed_variance': speedVariance,
      'average_punch_duration': averagePunchDuration,
      'fastest_punch': fastestPunch?.toJson(),
      'slowest_punch': slowestPunch?.toJson(),
    };
  }
}

/// Details about a specific punch
class PunchDetail {
  final int index;
  final double speed;
  final double time;
  final double duration;

  PunchDetail({
    required this.index,
    required this.speed,
    required this.time,
    required this.duration,
  });

  factory PunchDetail.fromJson(Map<String, dynamic> json) {
    return PunchDetail(
      index: json['index'] as int,
      speed: (json['speed'] as num).toDouble(),
      time: (json['time'] as num).toDouble(),
      duration: (json['duration'] as num).toDouble(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'index': index,
      'speed': speed,
      'time': time,
      'duration': duration,
    };
  }
}

/// Activity and temporal metrics
class ActivityMetrics {
  final double activeTime;
  final double restTime;
  final double activityPercentage;
  final double averageRestPeriod;
  final double maxRestPeriod;
  final double minRestPeriod;

  ActivityMetrics({
    required this.activeTime,
    required this.restTime,
    required this.activityPercentage,
    required this.averageRestPeriod,
    required this.maxRestPeriod,
    required this.minRestPeriod,
  });

  factory ActivityMetrics.fromJson(Map<String, dynamic> json) {
    return ActivityMetrics(
      activeTime: (json['active_time'] as num?)?.toDouble() ?? 0.0,
      restTime: (json['rest_time'] as num?)?.toDouble() ?? 0.0,
      activityPercentage: (json['activity_percentage'] as num?)?.toDouble() ?? 0.0,
      averageRestPeriod: (json['average_rest_period'] as num?)?.toDouble() ?? 0.0,
      maxRestPeriod: (json['max_rest_period'] as num?)?.toDouble() ?? 0.0,
      minRestPeriod: (json['min_rest_period'] as num?)?.toDouble() ?? 0.0,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'active_time': activeTime,
      'rest_time': restTime,
      'activity_percentage': activityPercentage,
      'average_rest_period': averageRestPeriod,
      'max_rest_period': maxRestPeriod,
      'min_rest_period': minRestPeriod,
    };
  }
}

/// Intensity metrics
class IntensityMetrics {
  final double score;
  final List<int> punchesByInterval;
  final PeakInterval? peakInterval;

  IntensityMetrics({
    required this.score,
    required this.punchesByInterval,
    this.peakInterval,
  });

  factory IntensityMetrics.fromJson(Map<String, dynamic> json) {
    return IntensityMetrics(
      score: (json['score'] as num?)?.toDouble() ?? 0.0,
      punchesByInterval: (json['punches_by_interval'] as List<dynamic>?)
              ?.map((e) => (e as num).toInt())
              .toList() ??
          [],
      peakInterval: json['peak_interval'] != null
          ? PeakInterval.fromJson(json['peak_interval'])
          : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'score': score,
      'punches_by_interval': punchesByInterval,
      'peak_interval': peakInterval?.toJson(),
    };
  }
}

/// Peak interval information
class PeakInterval {
  final int interval;
  final int punches;

  PeakInterval({
    required this.interval,
    required this.punches,
  });

  factory PeakInterval.fromJson(Map<String, dynamic> json) {
    return PeakInterval(
      interval: json['interval'] as int,
      punches: json['punches'] as int,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'interval': interval,
      'punches': punches,
    };
  }
}

/// Fatigue indicators
class FatigueMetrics {
  final double indicator;
  final String speedTrend; // 'improving', 'stable', 'declining'

  FatigueMetrics({
    required this.indicator,
    required this.speedTrend,
  });

  factory FatigueMetrics.fromJson(Map<String, dynamic> json) {
    return FatigueMetrics(
      indicator: (json['indicator'] as num?)?.toDouble() ?? 0.0,
      speedTrend: json['speed_trend'] as String? ?? 'stable',
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'indicator': indicator,
      'speed_trend': speedTrend,
    };
  }

  String get trendDescription {
    if (speedTrend == 'improving') return 'Speed increasing';
    if (speedTrend == 'declining') return 'Fatigue detected';
    return 'Speed consistent';
  }
}

/// Graph data for visualizing punch analysis
class GraphData {
  final List<double> speedOverTime;
  final Map<String, int> handDistribution;

  GraphData({
    required this.speedOverTime,
    required this.handDistribution,
  });

  factory GraphData.fromJson(Map<String, dynamic> json) {
    return GraphData(
      speedOverTime: (json['speed_over_time'] as List<dynamic>?)
              ?.map((e) => (e as num).toDouble())
              .toList() ??
          [],
      handDistribution: Map<String, int>.from(json['hand_distribution'] ?? {}),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'speed_over_time': speedOverTime,
      'hand_distribution': handDistribution,
    };
  }

  int get leftPunches => handDistribution['left'] ?? 0;
  int get rightPunches => handDistribution['right'] ?? 0;
  int get totalPunches => leftPunches + rightPunches;
}

/// Complete metrics for a shadowboxing analysis session
class SessionMetrics {
  // Basic metrics (backward compatible)
  final int totalPunches;
  final double averageSpeed;
  final double punchesPerMinute;
  final List<PunchEvent> punchEvents;
  final GraphData graphs;

  // Session info
  final double sessionDuration;

  // Comprehensive metrics
  final PerformanceMetrics performance;
  final ActivityMetrics activity;
  final IntensityMetrics intensity;
  final FatigueMetrics fatigue;

  SessionMetrics({
    required this.totalPunches,
    required this.averageSpeed,
    required this.punchesPerMinute,
    required this.punchEvents,
    required this.graphs,
    required this.sessionDuration,
    required this.performance,
    required this.activity,
    required this.intensity,
    required this.fatigue,
  });

  /// Factory constructor to create SessionMetrics from Python backend JSON
  factory SessionMetrics.fromJson(Map<String, dynamic> json) {
    return SessionMetrics(
      totalPunches: json['total_punches'] as int,
      averageSpeed: (json['average_speed'] as num).toDouble(),
      punchesPerMinute: (json['punches_per_minute'] as num).toDouble(),
      punchEvents: (json['punch_events'] as List<dynamic>?)
              ?.map((e) => PunchEvent.fromJson(e as Map<String, dynamic>))
              .toList() ??
          [],
      graphs: GraphData.fromJson(json['graphs'] as Map<String, dynamic>? ?? {}),
      sessionDuration: (json['session_duration'] as num?)?.toDouble() ?? 0.0,
      performance: PerformanceMetrics.fromJson(
        json['performance'] as Map<String, dynamic>? ?? {},
      ),
      activity: ActivityMetrics.fromJson(
        json['activity'] as Map<String, dynamic>? ?? {},
      ),
      intensity: IntensityMetrics.fromJson(
        json['intensity'] as Map<String, dynamic>? ?? {},
      ),
      fatigue: FatigueMetrics.fromJson(
        json['fatigue'] as Map<String, dynamic>? ?? {},
      ),
    );
  }

  /// Convert SessionMetrics to JSON
  Map<String, dynamic> toJson() {
    return {
      'total_punches': totalPunches,
      'average_speed': averageSpeed,
      'punches_per_minute': punchesPerMinute,
      'punch_events': punchEvents.map((e) => e.toJson()).toList(),
      'graphs': graphs.toJson(),
      'session_duration': sessionDuration,
      'performance': performance.toJson(),
      'activity': activity.toJson(),
      'intensity': intensity.toJson(),
      'fatigue': fatigue.toJson(),
    };
  }

  /// Get punches filtered by hand
  List<PunchEvent> getPunchesByHand(String hand) {
    return punchEvents.where((p) => p.hand == hand).toList();
  }

  /// Calculate average speed for a specific hand
  double getAverageSpeedForHand(String hand) {
    final handPunches = getPunchesByHand(hand);
    if (handPunches.isEmpty) return 0.0;

    final totalSpeed = handPunches.fold<double>(
      0.0,
      (sum, punch) => sum + punch.speed,
    );
    return totalSpeed / handPunches.length;
  }

  @override
  String toString() {
    return 'SessionMetrics(total: $totalPunches, avgSpeed: ${averageSpeed.toStringAsFixed(2)}, ppm: ${punchesPerMinute.toStringAsFixed(1)})';
  }
}
