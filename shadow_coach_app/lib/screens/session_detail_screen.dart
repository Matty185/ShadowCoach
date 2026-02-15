import 'package:flutter/material.dart';
import '../models/session_data.dart';
import '../models/models.dart';

/// Screen for viewing detailed session information
class SessionDetailScreen extends StatelessWidget {
  final SessionData session;

  const SessionDetailScreen({super.key, required this.session});

  @override
  Widget build(BuildContext context) {
    final metrics = session.metrics;

    return Scaffold(
      appBar: AppBar(
        title: const Text('SESSION DETAILS'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Session info card
            _buildInfoCard(context),
            const SizedBox(height: 16),

            // Summary card
            _buildSummaryCard(metrics),
            const SizedBox(height: 16),

            // Performance metrics
            _buildPerformanceCard(metrics),
            const SizedBox(height: 16),

            // Activity metrics
            _buildActivityCard(metrics),
            const SizedBox(height: 16),

            // Intensity & Fatigue
            Row(
              children: [
                Expanded(child: _buildIntensityCard(metrics)),
                const SizedBox(width: 16),
                Expanded(child: _buildFatigueCard(metrics)),
              ],
            ),
            const SizedBox(height: 24),

            // Punch events
            if (metrics.punchEvents.isNotEmpty) ...[
              const Text(
                'Detected Punches',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 12),
              ...metrics.punchEvents.map((punch) => _buildPunchCard(punch)),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildInfoCard(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.calendar_today, size: 20, color: Colors.blueAccent),
                const SizedBox(width: 8),
                const Text(
                  'Session Information',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const Divider(height: 24),
            _buildInfoRow('Date', session.formattedDate),
            const SizedBox(height: 12),
            _buildInfoRow('File', session.fileName),
            const SizedBox(height: 12),
            _buildInfoRow('Intensity', session.intensityRating),
            if (session.videoDuration != null) ...[
              const SizedBox(height: 12),
              _buildInfoRow('Duration', '${session.videoDuration!.toStringAsFixed(1)}s'),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: TextStyle(fontSize: 14, color: Colors.grey[400]),
        ),
        Text(
          value,
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }

  Widget _buildSummaryCard(SessionMetrics metrics) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            const Text(
              'Summary',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const Divider(height: 24),
            _buildMetricRow('Total Punches', metrics.totalPunches.toString()),
            const Divider(height: 24),
            _buildMetricRow('Average Speed', '${metrics.averageSpeed.toStringAsFixed(2)} m/s'),
            const Divider(height: 24),
            _buildMetricRow('Punches/Min', metrics.punchesPerMinute.toStringAsFixed(1)),
            const Divider(height: 24),
            _buildMetricRow('Left/Right', '${metrics.graphs.leftPunches}/${metrics.graphs.rightPunches}'),
          ],
        ),
      ),
    );
  }

  Widget _buildMetricRow(String label, String value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: TextStyle(fontSize: 16, color: Colors.grey[400]),
        ),
        Text(
          value,
          style: const TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
            color: Colors.blueAccent,
          ),
        ),
      ],
    );
  }

  Widget _buildPerformanceCard(SessionMetrics metrics) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.speed, color: Colors.orangeAccent, size: 20),
                const SizedBox(width: 8),
                const Text(
                  'Performance',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildStatColumn(
                  'Max Speed',
                  '${metrics.performance.maxSpeed.toStringAsFixed(2)} m/s',
                  Colors.greenAccent,
                ),
                _buildStatColumn(
                  'Min Speed',
                  '${metrics.performance.minSpeed.toStringAsFixed(2)} m/s',
                  Colors.orangeAccent,
                ),
                _buildStatColumn(
                  'Variance',
                  metrics.performance.speedVariance.toStringAsFixed(2),
                  Colors.blueAccent,
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildActivityCard(SessionMetrics metrics) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.timer, color: Colors.blueAccent, size: 20),
                const SizedBox(width: 8),
                const Text(
                  'Activity',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildStatColumn(
                  'Active',
                  '${metrics.activity.activeTime.toStringAsFixed(1)}s',
                  Colors.greenAccent,
                ),
                _buildStatColumn(
                  'Rest',
                  '${metrics.activity.restTime.toStringAsFixed(1)}s',
                  Colors.grey,
                ),
                _buildStatColumn(
                  'Activity %',
                  '${metrics.activity.activityPercentage.toStringAsFixed(1)}%',
                  Colors.blueAccent,
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildIntensityCard(SessionMetrics metrics) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.flash_on, color: Colors.yellowAccent, size: 20),
                const SizedBox(width: 8),
                const Text(
                  'Intensity',
                  style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              metrics.intensity.score.toStringAsFixed(2),
              style: const TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: Colors.yellowAccent),
            ),
            if (metrics.intensity.peakInterval != null)
              Text(
                'Peak: ${metrics.intensity.peakInterval!.punches} jabs in interval ${metrics.intensity.peakInterval!.interval + 1}',
                style: TextStyle(fontSize: 11, color: Colors.grey[400]),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildFatigueCard(SessionMetrics metrics) {
    Color trendColor;
    IconData trendIcon;

    if (metrics.fatigue.speedTrend == 'improving') {
      trendColor = Colors.greenAccent;
      trendIcon = Icons.trending_up;
    } else if (metrics.fatigue.speedTrend == 'declining') {
      trendColor = Colors.redAccent;
      trendIcon = Icons.trending_down;
    } else {
      trendColor = Colors.blueAccent;
      trendIcon = Icons.trending_flat;
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.battery_charging_full, color: trendColor, size: 20),
                const SizedBox(width: 8),
                const Text(
                  'Fatigue',
                  style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Icon(trendIcon, color: trendColor, size: 28),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    metrics.fatigue.trendDescription,
                    style: TextStyle(fontSize: 11, color: Colors.grey[400]),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatColumn(String label, String value, Color color) {
    return Column(
      children: [
        Text(
          value,
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: color),
        ),
        const SizedBox(height: 4),
        Text(
          label,
          style: TextStyle(fontSize: 11, color: Colors.grey[400]),
        ),
      ],
    );
  }

  Widget _buildPunchCard(PunchEvent punch) {
    final bool isJab = punch.punchType == 'jab';
    final Color avatarColor = isJab ? Colors.blueAccent : Colors.orangeAccent;
    final String punchLabel = isJab
        ? '${punch.hand.toUpperCase()} JAB'
        : '${punch.hand.toUpperCase()} PUNCH';

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: avatarColor,
          child: Text('${punch.index}'),
        ),
        title: Text(punchLabel),
        subtitle: Text(
          'Time: ${punch.startTime.toStringAsFixed(2)}s - ${punch.endTime.toStringAsFixed(2)}s',
        ),
        trailing: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text(
              '${punch.speed.toStringAsFixed(2)} m/s',
              style: const TextStyle(
                fontWeight: FontWeight.bold,
                color: Colors.greenAccent,
              ),
            ),
            Text(
              '${punch.duration.toStringAsFixed(2)}s',
              style: TextStyle(fontSize: 12, color: Colors.grey[400]),
            ),
          ],
        ),
      ),
    );
  }
}
