/// Represents a single punch detected in the analysis
class PunchEvent {
  final int index;
  final double speed; // meters per second or similar unit
  final String hand; // 'left' or 'right'
  final double startTime; // seconds
  final double endTime; // seconds
  final String punchType; // 'jab' or 'punch'

  PunchEvent({
    required this.index,
    required this.speed,
    required this.hand,
    required this.startTime,
    required this.endTime,
    this.punchType = 'punch', // Default to 'punch' for backward compatibility
  });

  /// Duration of the punch in seconds
  double get duration => endTime - startTime;

  /// Factory constructor to create PunchEvent from JSON
  factory PunchEvent.fromJson(Map<String, dynamic> json) {
    return PunchEvent(
      index: json['index'] as int,
      speed: (json['speed'] as num).toDouble(),
      hand: json['hand'] as String,
      startTime: (json['start'] as num).toDouble(),
      endTime: (json['end'] as num).toDouble(),
      punchType: json['punch_type'] as String? ?? 'punch', // Default to 'punch' if not provided
    );
  }

  /// Convert PunchEvent to JSON
  Map<String, dynamic> toJson() {
    return {
      'index': index,
      'speed': speed,
      'hand': hand,
      'start': startTime,
      'end': endTime,
      'punch_type': punchType,
    };
  }

  @override
  String toString() {
    return 'PunchEvent(index: $index, speed: ${speed.toStringAsFixed(2)}, hand: $hand, type: $punchType, duration: ${duration.toStringAsFixed(2)}s)';
  }
}
