import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../storage/database_service.dart';
import '../models/session_data.dart';
import '../services/mock_data_service.dart';
import 'comparison_screen.dart';

class ProgressDashboardScreen extends StatefulWidget {
  const ProgressDashboardScreen({super.key});

  @override
  State<ProgressDashboardScreen> createState() => _ProgressDashboardScreenState();
}

class _ProgressDashboardScreenState extends State<ProgressDashboardScreen> {
  List<SessionData> _sessions = []; // chronological order
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadSessions();
  }

  Future<void> _loadSessions() async {
    setState(() => _isLoading = true);
    try {
      final sessions = await DatabaseService.instance.getAllSessions();
      setState(() {
        _sessions = sessions.reversed.toList(); // chronological
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _reseed() async {
    setState(() => _isLoading = true);
    await MockDataService.seedSampleSessions();
    await _loadSessions();
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Demo data reseeded'),
        backgroundColor: Colors.greenAccent,
      ),
    );
  }

  double get _improvementScore {
    if (_sessions.length < 2) return 0;
    final latest = _sessions.last;
    final priors = _sessions.sublist(0, _sessions.length - 1);

    final avgSpeed = priors.fold<double>(0, (s, e) => s + e.metrics.averageSpeed) / priors.length;
    final avgPpm = priors.fold<double>(0, (s, e) => s + e.metrics.punchesPerMinute) / priors.length;
    final avgActivity = priors.fold<double>(0, (s, e) => s + e.metrics.activity.activityPercentage) / priors.length;

    final speedImprovement = avgSpeed > 0 ? ((latest.metrics.averageSpeed - avgSpeed) / avgSpeed) * 100 : 0.0;
    final ppmImprovement = avgPpm > 0 ? ((latest.metrics.punchesPerMinute - avgPpm) / avgPpm) * 100 : 0.0;
    final activityImprovement = avgActivity > 0 ? ((latest.metrics.activity.activityPercentage - avgActivity) / avgActivity) * 100 : 0.0;

    return speedImprovement * 0.40 + ppmImprovement * 0.35 + activityImprovement * 0.25;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('PROGRESS'),
        actions: [
          IconButton(
            icon: const Icon(Icons.replay),
            onPressed: _reseed,
            tooltip: 'Reseed demo data',
          ),
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadSessions,
            tooltip: 'Refresh',
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _sessions.length < 2
              ? _buildEmptyState()
              : _buildDashboard(),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.insights, size: 80, color: Colors.grey[600]),
            const SizedBox(height: 24),
            const Text(
              'Need at least 2 sessions',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            Text(
              'Analyze more videos or seed demo data to see your progress trends.',
              style: TextStyle(color: Colors.grey[400]),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              onPressed: _reseed,
              icon: const Icon(Icons.auto_fix_high),
              label: const Text('SEED DEMO DATA'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.orangeAccent,
                foregroundColor: Colors.black,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDashboard() {
    final bestPpm = _sessions.map((s) => s.metrics.punchesPerMinute).reduce((a, b) => a > b ? a : b);
    final improvement = _improvementScore;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          // Summary Header Row
          Row(
            children: [
              _buildStatCard('Sessions', _sessions.length.toString(), Colors.blueAccent),
              const SizedBox(width: 12),
              _buildStatCard('Best PPM', bestPpm.toStringAsFixed(0), Colors.orangeAccent),
              const SizedBox(width: 12),
              _buildStatCard(
                'Improve',
                '${improvement >= 0 ? '+' : ''}${improvement.toStringAsFixed(0)}%',
                improvement >= 0 ? Colors.greenAccent : Colors.redAccent,
              ),
            ],
          ),
          const SizedBox(height: 20),

          // Speed Trend Chart
          _buildChartCard(
            title: 'Average Speed (m/s)',
            chart: _buildSpeedLineChart(),
          ),
          const SizedBox(height: 16),

          // Punch Count Bar Chart
          _buildChartCard(
            title: 'Total Punches',
            chart: _buildPunchBarChart(),
          ),
          const SizedBox(height: 16),

          // PPM Trend Chart
          _buildChartCard(
            title: 'Punches per Minute',
            chart: _buildPpmLineChart(),
          ),
          const SizedBox(height: 16),

          // Improvement Score Card
          _buildImprovementCard(improvement),
          const SizedBox(height: 16),

          // Compare Last Two button
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: () {
                if (_sessions.length >= 2) {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (_) => ComparisonScreen(
                        sessionA: _sessions[_sessions.length - 2],
                        sessionB: _sessions.last,
                      ),
                    ),
                  );
                }
              },
              icon: const Icon(Icons.compare_arrows),
              label: const Text('COMPARE LAST TWO SESSIONS'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blueAccent,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
            ),
          ),
          const SizedBox(height: 16),
        ],
      ),
    );
  }

  Widget _buildStatCard(String label, String value, Color color) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 12),
        decoration: BoxDecoration(
          color: Colors.grey[850],
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          children: [
            Text(
              value,
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              label,
              style: TextStyle(fontSize: 11, color: Colors.grey[400]),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildChartCard({required String title, required Widget chart}) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.grey[850],
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.bold,
              color: Colors.grey[300],
            ),
          ),
          const SizedBox(height: 16),
          SizedBox(height: 200, child: chart),
        ],
      ),
    );
  }

  Widget _buildImprovementCard(double improvement) {
    final isPositive = improvement >= 0;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Colors.grey[850],
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        children: [
          Text(
            'Overall Improvement',
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.bold,
              color: Colors.grey[300],
            ),
          ),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                isPositive ? Icons.trending_up : Icons.trending_down,
                color: isPositive ? Colors.greenAccent : Colors.redAccent,
                size: 36,
              ),
              const SizedBox(width: 12),
              Text(
                '${isPositive ? '+' : ''}${improvement.toStringAsFixed(1)}%',
                style: TextStyle(
                  fontSize: 40,
                  fontWeight: FontWeight.bold,
                  color: isPositive ? Colors.greenAccent : Colors.redAccent,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            '40% speed + 35% PPM + 25% activity',
            style: TextStyle(fontSize: 11, color: Colors.grey[500]),
          ),
        ],
      ),
    );
  }

  // ---- Chart builders ----

  List<FlSpot> _spotsFromMetric(double Function(SessionData) getter) {
    return List.generate(
      _sessions.length,
      (i) => FlSpot(i.toDouble(), getter(_sessions[i])),
    );
  }

  Widget _buildSpeedLineChart() {
    final spots = _spotsFromMetric((s) => s.metrics.averageSpeed);
    return LineChart(
      LineChartData(
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
          getDrawingHorizontalLine: (_) => FlLine(
            color: Colors.grey[700]!.withValues(alpha: 0.3),
            strokeWidth: 1,
          ),
        ),
        titlesData: _buildTitles(),
        borderData: FlBorderData(show: false),
        lineTouchData: LineTouchData(
          touchTooltipData: LineTouchTooltipData(
            getTooltipColor: (_) => Colors.grey[800]!,
          ),
        ),
        lineBarsData: [
          LineChartBarData(
            spots: spots,
            isCurved: true,
            color: Colors.blueAccent,
            barWidth: 3,
            dotData: FlDotData(
              show: true,
              getDotPainter: (p0, p1, p2, p3) => FlDotCirclePainter(
                radius: 4,
                color: Colors.blueAccent,
                strokeWidth: 2,
                strokeColor: Colors.white,
              ),
            ),
            belowBarData: BarAreaData(
              show: true,
              color: Colors.blueAccent.withValues(alpha: 0.15),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPunchBarChart() {
    return BarChart(
      BarChartData(
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
          getDrawingHorizontalLine: (_) => FlLine(
            color: Colors.grey[700]!.withValues(alpha: 0.3),
            strokeWidth: 1,
          ),
        ),
        titlesData: _buildTitles(),
        borderData: FlBorderData(show: false),
        barTouchData: BarTouchData(
          touchTooltipData: BarTouchTooltipData(
            getTooltipColor: (_) => Colors.grey[800]!,
          ),
        ),
        barGroups: List.generate(_sessions.length, (i) {
          return BarChartGroupData(
            x: i,
            barRods: [
              BarChartRodData(
                toY: _sessions[i].metrics.totalPunches.toDouble(),
                color: Colors.greenAccent,
                width: 20,
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(4),
                  topRight: Radius.circular(4),
                ),
              ),
            ],
          );
        }),
      ),
    );
  }

  Widget _buildPpmLineChart() {
    final spots = _spotsFromMetric((s) => s.metrics.punchesPerMinute);
    return LineChart(
      LineChartData(
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
          getDrawingHorizontalLine: (_) => FlLine(
            color: Colors.grey[700]!.withValues(alpha: 0.3),
            strokeWidth: 1,
          ),
        ),
        titlesData: _buildTitles(),
        borderData: FlBorderData(show: false),
        lineTouchData: LineTouchData(
          touchTooltipData: LineTouchTooltipData(
            getTooltipColor: (_) => Colors.grey[800]!,
          ),
        ),
        lineBarsData: [
          LineChartBarData(
            spots: spots,
            isCurved: true,
            color: Colors.orangeAccent,
            barWidth: 3,
            dotData: FlDotData(
              show: true,
              getDotPainter: (p0, p1, p2, p3) => FlDotCirclePainter(
                radius: 4,
                color: Colors.orangeAccent,
                strokeWidth: 2,
                strokeColor: Colors.white,
              ),
            ),
            belowBarData: BarAreaData(
              show: true,
              color: Colors.orangeAccent.withValues(alpha: 0.15),
            ),
          ),
        ],
      ),
    );
  }

  FlTitlesData _buildTitles() {
    return FlTitlesData(
      topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
      rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
      bottomTitles: AxisTitles(
        sideTitles: SideTitles(
          showTitles: true,
          getTitlesWidget: (value, meta) {
            final idx = value.toInt();
            if (idx < 0 || idx >= _sessions.length) return const SizedBox.shrink();
            return Text(
              'S${idx + 1}',
              style: TextStyle(fontSize: 10, color: Colors.grey[400]),
            );
          },
          reservedSize: 24,
        ),
      ),
      leftTitles: AxisTitles(
        sideTitles: SideTitles(
          showTitles: true,
          reservedSize: 36,
          getTitlesWidget: (value, meta) {
            return Text(
              value.toStringAsFixed(0),
              style: TextStyle(fontSize: 10, color: Colors.grey[400]),
            );
          },
        ),
      ),
    );
  }
}
