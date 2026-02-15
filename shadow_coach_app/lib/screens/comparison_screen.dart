import 'package:flutter/material.dart';
import '../models/session_data.dart';
import 'package:intl/intl.dart';

/// Side-by-side comparison of two sessions with improvement/regression indicators.
class ComparisonScreen extends StatelessWidget {
  final SessionData sessionA;
  final SessionData sessionB;

  const ComparisonScreen({
    super.key,
    required this.sessionA,
    required this.sessionB,
  });

  @override
  Widget build(BuildContext context) {
    final dateFormat = DateFormat('MMM d, h:mm a');
    return Scaffold(
      appBar: AppBar(
        title: const Text('SESSION COMPARISON'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            // Session headers
            Row(
              children: [
                Expanded(
                  child: _buildSessionHeader(
                    'Session A',
                    dateFormat.format(sessionA.timestamp),
                    Colors.blueAccent,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _buildSessionHeader(
                    'Session B',
                    dateFormat.format(sessionB.timestamp),
                    Colors.orangeAccent,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),

            // Comparison rows
            _buildComparisonRow(
              label: 'Average Speed',
              unit: 'm/s',
              valueA: sessionA.metrics.averageSpeed,
              valueB: sessionB.metrics.averageSpeed,
              higherIsBetter: true,
            ),
            _buildComparisonRow(
              label: 'Total Punches',
              unit: '',
              valueA: sessionA.metrics.totalPunches.toDouble(),
              valueB: sessionB.metrics.totalPunches.toDouble(),
              higherIsBetter: true,
            ),
            _buildComparisonRow(
              label: 'Punches/Min',
              unit: '',
              valueA: sessionA.metrics.punchesPerMinute,
              valueB: sessionB.metrics.punchesPerMinute,
              higherIsBetter: true,
            ),
            _buildComparisonRow(
              label: 'Activity',
              unit: '%',
              valueA: sessionA.metrics.activity.activityPercentage,
              valueB: sessionB.metrics.activity.activityPercentage,
              higherIsBetter: true,
            ),
            _buildComparisonRow(
              label: 'Intensity Score',
              unit: '',
              valueA: sessionA.metrics.intensity.score,
              valueB: sessionB.metrics.intensity.score,
              higherIsBetter: true,
            ),
            _buildComparisonRow(
              label: 'Max Speed',
              unit: 'm/s',
              valueA: sessionA.metrics.performance.maxSpeed,
              valueB: sessionB.metrics.performance.maxSpeed,
              higherIsBetter: true,
            ),
            _buildComparisonRow(
              label: 'Speed Variance',
              unit: '',
              valueA: sessionA.metrics.performance.speedVariance,
              valueB: sessionB.metrics.performance.speedVariance,
              higherIsBetter: false,
            ),
            _buildFatigueTrendRow(
              sessionA.metrics.fatigue.speedTrend,
              sessionB.metrics.fatigue.speedTrend,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSessionHeader(String label, String date, Color color) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.grey[850],
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Container(
            width: 10,
            height: 10,
            decoration: BoxDecoration(
              color: color,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                    color: color,
                  ),
                ),
                Text(
                  date,
                  style: TextStyle(fontSize: 11, color: Colors.grey[400]),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildComparisonRow({
    required String label,
    required String unit,
    required double valueA,
    required double valueB,
    required bool higherIsBetter,
  }) {
    final diff = valueB - valueA;
    final pctChange = valueA != 0 ? (diff / valueA.abs()) * 100 : 0.0;

    // Determine if change is good
    final isImproved = higherIsBetter ? diff > 0.01 : diff < -0.01;
    final isRegressed = higherIsBetter ? diff < -0.01 : diff > 0.01;

    Color changeColor;
    IconData changeIcon;
    if (isImproved) {
      changeColor = Colors.greenAccent;
      changeIcon = Icons.arrow_upward;
    } else if (isRegressed) {
      changeColor = Colors.redAccent;
      changeIcon = Icons.arrow_downward;
    } else {
      changeColor = Colors.grey;
      changeIcon = Icons.remove;
    }

    final suffix = unit.isNotEmpty ? ' $unit' : '';

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              label,
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: Colors.grey[400],
              ),
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                // Value A
                Expanded(
                  child: Text(
                    '${valueA.toStringAsFixed(1)}$suffix',
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.blueAccent,
                    ),
                  ),
                ),
                // Change indicator
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: changeColor.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(changeIcon, color: changeColor, size: 16),
                      const SizedBox(width: 4),
                      Text(
                        '${pctChange >= 0 ? '+' : ''}${pctChange.toStringAsFixed(1)}%',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          color: changeColor,
                        ),
                      ),
                    ],
                  ),
                ),
                // Value B
                Expanded(
                  child: Text(
                    '${valueB.toStringAsFixed(1)}$suffix',
                    textAlign: TextAlign.right,
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.orangeAccent,
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildFatigueTrendRow(String trendA, String trendB) {
    // Score: improving=2, stable=1, declining=0
    int score(String t) => t == 'improving' ? 2 : t == 'stable' ? 1 : 0;
    final diff = score(trendB) - score(trendA);

    Color changeColor;
    IconData changeIcon;
    String changeText;
    if (diff > 0) {
      changeColor = Colors.greenAccent;
      changeIcon = Icons.arrow_upward;
      changeText = 'Better';
    } else if (diff < 0) {
      changeColor = Colors.redAccent;
      changeIcon = Icons.arrow_downward;
      changeText = 'Worse';
    } else {
      changeColor = Colors.grey;
      changeIcon = Icons.remove;
      changeText = 'Same';
    }

    String displayTrend(String t) => t[0].toUpperCase() + t.substring(1);

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Fatigue Trend',
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: Colors.grey[400],
              ),
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: Text(
                    displayTrend(trendA),
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.blueAccent,
                    ),
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: changeColor.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(changeIcon, color: changeColor, size: 16),
                      const SizedBox(width: 4),
                      Text(
                        changeText,
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          color: changeColor,
                        ),
                      ),
                    ],
                  ),
                ),
                Expanded(
                  child: Text(
                    displayTrend(trendB),
                    textAlign: TextAlign.right,
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.orangeAccent,
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
