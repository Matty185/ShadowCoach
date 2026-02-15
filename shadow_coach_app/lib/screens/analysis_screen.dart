import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:io';
import '../services/api_service.dart';
import '../models/models.dart';
import '../storage/database_service.dart';

/// Screen for uploading and analyzing videos
class AnalysisScreen extends StatefulWidget {
  const AnalysisScreen({super.key});

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen> {
  final ApiService _apiService = ApiService();

  bool _isAnalyzing = false;
  bool _apiHealthy = false;
  SessionMetrics? _results;
  String? _error;
  String? _uploadedFileName;

  @override
  void initState() {
    super.initState();
    _checkApiHealth();
  }

  Future<void> _checkApiHealth() async {
    final healthy = await _apiService.healthCheck();
    setState(() {
      _apiHealthy = healthy;
    });
  }

  Future<void> _pickAndAnalyzeVideo() async {
    // Pick video file
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.video,
      allowMultiple: false,
    );

    if (result == null) return;

    final platformFile = result.files.single;

    // Analyze video
    setState(() {
      _isAnalyzing = true;
      _error = null;
      _results = null;
      _uploadedFileName = platformFile.name;
    });

    try {
      final SessionMetrics metrics;

      // Web platform uses bytes, other platforms use file path
      if (platformFile.bytes != null) {
        // Web platform
        metrics = await _apiService.analyzeVideoBytes(
          platformFile.bytes!,
          platformFile.name,
        );
      } else if (platformFile.path != null) {
        // Mobile/Desktop platform
        final videoFile = File(platformFile.path!);
        metrics = await _apiService.analyzeVideo(videoFile);
      } else {
        _showError('Unable to access file');
        setState(() {
          _isAnalyzing = false;
        });
        return;
      }

      setState(() {
        _results = metrics;
        _isAnalyzing = false;
      });
    } on ApiException catch (e) {
      _showError(e.toString());
      setState(() {
        _isAnalyzing = false;
      });
    }
  }

  void _showError(String message) {
    setState(() {
      _error = message;
    });
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.redAccent,
      ),
    );
  }

  Future<void> _saveSession() async {
    if (_results == null) return;

    try {
      // Create session data
      final sessionData = SessionData(
        timestamp: DateTime.now(),
        fileName: _uploadedFileName ?? 'Unknown Video',
        metrics: _results!,
        videoDuration: _results!.sessionDuration,
      );

      // Save to database
      final id = await DatabaseService.instance.saveSession(sessionData);

      // Show success message
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Session saved successfully! (ID: $id)'),
          backgroundColor: Colors.greenAccent,
          duration: const Duration(seconds: 2),
        ),
      );
    } catch (e) {
      // Show error message
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Failed to save session: $e'),
          backgroundColor: Colors.redAccent,
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('VIDEO ANALYSIS'),
        actions: [
          // API health indicator
          Padding(
            padding: const EdgeInsets.only(right: 16.0),
            child: Center(
              child: Row(
                children: [
                  Icon(
                    _apiHealthy ? Icons.check_circle : Icons.error,
                    color: _apiHealthy ? Colors.greenAccent : Colors.redAccent,
                    size: 20,
                  ),
                  const SizedBox(width: 8),
                  Text(
                    _apiHealthy ? 'API Ready' : 'API Offline',
                    style: TextStyle(
                      fontSize: 12,
                      color: _apiHealthy ? Colors.greenAccent : Colors.redAccent,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
      body: SafeArea(
        child: _isAnalyzing
            ? _buildAnalyzingView()
            : _results != null
                ? _buildResultsView()
                : _buildUploadView(),
      ),
    );
  }

  Widget _buildUploadView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.cloud_upload,
              size: 80,
              color: _apiHealthy ? Colors.blueAccent : Colors.grey,
            ),
            const SizedBox(height: 24),
            Text(
              'Upload Video for Analysis',
              style: const TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            Text(
              'Select a shadowboxing video to analyze',
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey[400],
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 48),
            ElevatedButton.icon(
              onPressed: _apiHealthy ? _pickAndAnalyzeVideo : null,
              icon: const Icon(Icons.video_file),
              label: const Text('SELECT VIDEO'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blueAccent,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
            ),
            if (!_apiHealthy) ...[
              const SizedBox(height: 24),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.redAccent.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.redAccent),
                ),
                child: Column(
                  children: [
                    const Icon(Icons.warning, color: Colors.redAccent),
                    const SizedBox(height: 8),
                    const Text(
                      'API Server Not Running',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        color: Colors.redAccent,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Start the server:\nenv\\Scripts\\python.exe api\\app.py',
                      style: TextStyle(
                        fontSize: 12,
                        fontFamily: 'monospace',
                        color: Colors.grey[400],
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildAnalyzingView() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const CircularProgressIndicator(
            color: Colors.blueAccent,
          ),
          const SizedBox(height: 24),
          const Text(
            'Analyzing Video...',
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          Text(
            'This may take 5-15 seconds',
            style: TextStyle(fontSize: 14, color: Colors.grey[400]),
          ),
        ],
      ),
    );
  }

  Widget _buildResultsView() {
    final metrics = _results!;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Success icon
          const Icon(
            Icons.check_circle,
            color: Colors.greenAccent,
            size: 64,
          ),
          const SizedBox(height: 16),
          const Text(
            'Analysis Complete!',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 32),

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

          const SizedBox(height: 32),

          // Action buttons
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () {
                    setState(() {
                      _results = null;
                    });
                  },
                  icon: const Icon(Icons.refresh),
                  label: const Text('ANALYZE ANOTHER'),
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: _saveSession,
                  icon: const Icon(Icons.save),
                  label: const Text('SAVE'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.greenAccent,
                    foregroundColor: Colors.black,
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildSummaryCard(SessionMetrics metrics) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'Summary',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                IconButton(
                  icon: const Icon(Icons.info_outline, size: 20),
                  onPressed: () => _showAnalysisExplanation(
                    'Summary Metrics',
                    'Total Punches: Total number of jabs detected throughout the session.\n\n'
                    'Average Speed: Mean speed of all detected punches, calculated from wrist movement velocity.\n\n'
                    'Punches/Min: Frequency of punches normalized to one minute, based on total punches divided by session duration.\n\n'
                    'Left/Right: Count of punches thrown with each hand (currently detecting left jabs only).',
                  ),
                ),
              ],
            ),
            const Divider(height: 16),
            _buildMetricRowWithInfo(
              'Total Punches',
              metrics.totalPunches.toString(),
              'The total number of jabs detected throughout your entire session. Each punch is identified by analyzing your hand movement patterns.',
            ),
            const Divider(height: 24),
            _buildMetricRowWithInfo(
              'Average Speed',
              '${metrics.averageSpeed.toStringAsFixed(2)} m/s',
              'The average speed of all your punches, calculated by measuring how fast your hand moves during each punch. Higher speeds indicate more powerful strikes.',
            ),
            const Divider(height: 24),
            _buildMetricRowWithInfo(
              'Punches/Min',
              metrics.punchesPerMinute.toStringAsFixed(1),
              'How many punches you throw per minute on average. This is calculated by dividing your total punches by your session duration, then scaling to one minute.',
            ),
            const Divider(height: 24),
            _buildMetricRowWithInfo(
              'Left/Right',
              '${metrics.graphs.leftPunches}/${metrics.graphs.rightPunches}',
              'The breakdown of punches by hand. Shows how many jabs you threw with your left hand versus your right hand, helping track balance in your training.',
            ),
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

  Widget _buildMetricRowWithInfo(String label, String value, String explanation) {
    return GestureDetector(
      onTap: () => _showAnalysisExplanation(label, explanation),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            children: [
              Text(
                label,
                style: TextStyle(fontSize: 16, color: Colors.grey[400]),
              ),
              const SizedBox(width: 6),
              Icon(Icons.info_outline, size: 16, color: Colors.grey[500]),
            ],
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
      ),
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
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    Icon(Icons.speed, color: Colors.orangeAccent, size: 20),
                    const SizedBox(width: 8),
                    const Text(
                      'Performance',
                      style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
                IconButton(
                  icon: const Icon(Icons.info_outline, size: 18),
                  onPressed: () => _showAnalysisExplanation(
                    'Performance Metrics',
                    'Max Speed: The fastest punch detected in your session, measured by how quickly your hand moved.\n\n'
                    'Min Speed: The slowest punch detected, showing your baseline speed.\n\n'
                    'Variance: How consistent your punch speeds are. Lower variance means more consistent performance.',
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildStatColumnWithInfo(
                  'Max Speed',
                  '${metrics.performance.maxSpeed.toStringAsFixed(2)} m/s',
                  Colors.greenAccent,
                  'The fastest punch in your session. Shows your peak performance capability.',
                ),
                _buildStatColumnWithInfo(
                  'Min Speed',
                  '${metrics.performance.minSpeed.toStringAsFixed(2)} m/s',
                  Colors.orangeAccent,
                  'The slowest punch detected. Helps identify your baseline speed.',
                ),
                _buildStatColumnWithInfo(
                  'Variance',
                  metrics.performance.speedVariance.toStringAsFixed(2),
                  Colors.blueAccent,
                  'Measures how consistent your speeds are. Lower values mean more consistent punching.',
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
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    Icon(Icons.timer, color: Colors.blueAccent, size: 20),
                    const SizedBox(width: 8),
                    const Text(
                      'Activity',
                      style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
                IconButton(
                  icon: const Icon(Icons.info_outline, size: 18),
                  onPressed: () => _showAnalysisExplanation(
                    'Activity Metrics',
                    'Active Time: Total time spent throwing punches during your session.\n\n'
                    'Rest Time: Total time between punches when you\'re not actively punching.\n\n'
                    'Activity %: Percentage of your session spent actively punching. Higher percentages mean more intense workouts.',
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildStatColumnWithInfo(
                  'Active',
                  '${metrics.activity.activeTime.toStringAsFixed(1)}s',
                  Colors.greenAccent,
                  'Total time you spent actively throwing punches. Higher values indicate more punching time.',
                ),
                _buildStatColumnWithInfo(
                  'Rest',
                  '${metrics.activity.restTime.toStringAsFixed(1)}s',
                  Colors.grey,
                  'Time spent between punches. Shows your recovery periods during the session.',
                ),
                _buildStatColumnWithInfo(
                  'Activity %',
                  '${metrics.activity.activityPercentage.toStringAsFixed(1)}%',
                  Colors.blueAccent,
                  'What percentage of your session was spent punching. 100% means continuous punching with no breaks.',
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
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    Icon(Icons.flash_on, color: Colors.yellowAccent, size: 20),
                    const SizedBox(width: 8),
                    const Text(
                      'Intensity',
                      style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
                IconButton(
                  icon: const Icon(Icons.info_outline, size: 18),
                  onPressed: () => _showAnalysisExplanation(
                    'Intensity Score',
                    'This score combines how fast and how frequently you punch. It\'s calculated by multiplying your average punch speed with your punch frequency.\n\n'
                    'Higher scores mean you\'re maintaining both speed and consistency throughout your session.\n\n'
                    'Peak Interval shows which 10-second period had the most punches, indicating your most intense burst of activity.',
                  ),
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
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
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
                IconButton(
                  icon: const Icon(Icons.info_outline, size: 18),
                  onPressed: () => _showAnalysisExplanation(
                    'Fatigue Indicator',
                    'This metric compares your punch speeds in the first half of your session versus the second half.\n\n'
                    'Improving: Your punches got faster as the session went on - great endurance!\n\n'
                    'Declining: Your speed decreased, which may indicate fatigue setting in.\n\n'
                    'Stable: Your speed remained consistent throughout - excellent consistency!',
                  ),
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

  Widget _buildStatColumnWithInfo(String label, String value, Color color, String explanation) {
    return GestureDetector(
      onTap: () => _showAnalysisExplanation(label, explanation),
      child: Column(
        children: [
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                value,
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: color),
              ),
              const SizedBox(width: 4),
              Icon(Icons.info_outline, size: 14, color: Colors.grey[600]),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            label,
            style: TextStyle(fontSize: 11, color: Colors.grey[400]),
          ),
        ],
      ),
    );
  }

  Widget _buildPunchCard(PunchEvent punch) {
    // Determine colors and labels based on punch type
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

  void _showAnalysisExplanation(String title, String message) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Row(
            children: [
              const Icon(Icons.info_outline, color: Colors.blueAccent),
              const SizedBox(width: 8),
              Text(title),
            ],
          ),
          content: SingleChildScrollView(
            child: Text(
              message,
              style: const TextStyle(fontSize: 14),
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Got it'),
            ),
          ],
        );
      },
    );
  }
}
