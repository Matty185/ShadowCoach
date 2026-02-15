import 'package:flutter/material.dart';
import '../storage/database_service.dart';
import '../models/session_data.dart';
import 'session_detail_screen.dart';
import 'comparison_screen.dart';

/// Screen for viewing session history
class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  List<SessionData> _sessions = [];
  bool _isLoading = true;
  bool _compareMode = false;
  final Set<int> _selectedIds = {};

  @override
  void initState() {
    super.initState();
    _loadSessions();
  }

  Future<void> _loadSessions() async {
    setState(() {
      _isLoading = true;
    });

    try {
      final sessions = await DatabaseService.instance.getAllSessions();
      setState(() {
        _sessions = sessions;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Failed to load sessions: $e'),
          backgroundColor: Colors.redAccent,
        ),
      );
    }
  }

  Future<void> _deleteSession(SessionData session) async {
    // Show confirmation dialog
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Session'),
        content: Text('Are you sure you want to delete the session from ${session.formattedDate}?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('CANCEL'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            style: TextButton.styleFrom(foregroundColor: Colors.redAccent),
            child: const Text('DELETE'),
          ),
        ],
      ),
    );

    if (confirmed != true || session.id == null) return;

    try {
      await DatabaseService.instance.deleteSession(session.id!);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Session deleted'),
          backgroundColor: Colors.greenAccent,
        ),
      );
      _loadSessions(); // Reload sessions
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Failed to delete session: $e'),
          backgroundColor: Colors.redAccent,
        ),
      );
    }
  }

  void _navigateToComparison() {
    final selected = _sessions.where((s) => _selectedIds.contains(s.id)).toList();
    if (selected.length != 2) return;
    // Sort so older session is A, newer is B
    selected.sort((a, b) => a.timestamp.compareTo(b.timestamp));
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => ComparisonScreen(
          sessionA: selected[0],
          sessionB: selected[1],
        ),
      ),
    );
  }

  void _toggleSelection(SessionData session) {
    if (session.id == null) return;
    setState(() {
      if (_selectedIds.contains(session.id)) {
        _selectedIds.remove(session.id);
      } else if (_selectedIds.length < 2) {
        _selectedIds.add(session.id!);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(_compareMode ? 'SELECT 2 SESSIONS' : 'SESSION HISTORY'),
        actions: [
          IconButton(
            icon: Icon(
              Icons.compare_arrows,
              color: _compareMode ? Colors.orangeAccent : null,
            ),
            onPressed: () {
              setState(() {
                _compareMode = !_compareMode;
                _selectedIds.clear();
              });
            },
            tooltip: 'Compare sessions',
          ),
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadSessions,
            tooltip: 'Refresh',
          ),
        ],
      ),
      floatingActionButton: _compareMode && _selectedIds.length == 2
          ? FloatingActionButton.extended(
              onPressed: _navigateToComparison,
              backgroundColor: Colors.orangeAccent,
              foregroundColor: Colors.black,
              icon: const Icon(Icons.compare_arrows),
              label: const Text('COMPARE'),
            )
          : null,
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _sessions.isEmpty
              ? _buildEmptyState()
              : _buildSessionList(),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.history,
              size: 80,
              color: Colors.grey[600],
            ),
            const SizedBox(height: 24),
            const Text(
              'No Sessions Yet',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            Text(
              'Analyze a video and save it to see your session history',
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey[400],
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 32),
            ElevatedButton.icon(
              onPressed: () => Navigator.pushNamed(context, '/analyze'),
              icon: const Icon(Icons.add),
              label: const Text('ANALYZE VIDEO'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blueAccent,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSessionList() {
    return Column(
      children: [
        // Summary card
        Container(
          margin: const EdgeInsets.all(16),
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.grey[850],
            borderRadius: BorderRadius.circular(12),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _buildSummaryItem(
                'Total Sessions',
                _sessions.length.toString(),
                Icons.video_library,
                Colors.blueAccent,
              ),
              _buildSummaryItem(
                'Total Punches',
                _sessions.fold<int>(0, (sum, s) => sum + s.metrics.totalPunches).toString(),
                Icons.sports_mma,
                Colors.orangeAccent,
              ),
            ],
          ),
        ),

        // Session list
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            itemCount: _sessions.length,
            itemBuilder: (context, index) {
              final session = _sessions[index];
              return _buildSessionCard(session);
            },
          ),
        ),
      ],
    );
  }

  Widget _buildSummaryItem(String label, String value, IconData icon, Color color) {
    return Column(
      children: [
        Icon(icon, color: color, size: 32),
        const SizedBox(height: 8),
        Text(
          value,
          style: TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
        Text(
          label,
          style: TextStyle(
            fontSize: 12,
            color: Colors.grey[400],
          ),
        ),
      ],
    );
  }

  Widget _buildSessionCard(SessionData session) {
    final isSelected = _compareMode && _selectedIds.contains(session.id);
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: isSelected
            ? const BorderSide(color: Colors.orangeAccent, width: 2)
            : BorderSide.none,
      ),
      child: InkWell(
        onTap: _compareMode
            ? () => _toggleSelection(session)
            : () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => SessionDetailScreen(session: session),
                  ),
                );
              },
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header row
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  if (_compareMode)
                    Padding(
                      padding: const EdgeInsets.only(right: 8),
                      child: Icon(
                        isSelected ? Icons.check_circle : Icons.radio_button_unchecked,
                        color: isSelected ? Colors.orangeAccent : Colors.grey,
                      ),
                    ),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          session.formattedDate,
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          session.fileName,
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.grey[400],
                          ),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ],
                    ),
                  ),
                  // Intensity badge
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: _getIntensityColor(session.intensityRating),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(
                      session.intensityRating,
                      style: const TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        color: Colors.black,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),

              // Stats row
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  _buildStatItem(
                    Icons.sports_mma,
                    session.metrics.totalPunches.toString(),
                    'Punches',
                  ),
                  _buildStatItem(
                    Icons.speed,
                    session.metrics.averageSpeed.toStringAsFixed(1),
                    'm/s',
                  ),
                  _buildStatItem(
                    Icons.timer,
                    session.metrics.sessionDuration.toStringAsFixed(0),
                    'seconds',
                  ),
                ],
              ),
              const SizedBox(height: 12),

              // Action buttons
              Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  TextButton.icon(
                    onPressed: () => _deleteSession(session),
                    icon: const Icon(Icons.delete, size: 18),
                    label: const Text('DELETE'),
                    style: TextButton.styleFrom(
                      foregroundColor: Colors.redAccent,
                    ),
                  ),
                  const SizedBox(width: 8),
                  ElevatedButton.icon(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => SessionDetailScreen(session: session),
                        ),
                      );
                    },
                    icon: const Icon(Icons.visibility, size: 18),
                    label: const Text('VIEW'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blueAccent,
                      foregroundColor: Colors.white,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStatItem(IconData icon, String value, String label) {
    return Column(
      children: [
        Icon(icon, size: 20, color: Colors.grey[400]),
        const SizedBox(height: 4),
        Text(
          value,
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
        Text(
          label,
          style: TextStyle(
            fontSize: 11,
            color: Colors.grey[500],
          ),
        ),
      ],
    );
  }

  Color _getIntensityColor(String intensity) {
    switch (intensity) {
      case 'High':
        return Colors.redAccent;
      case 'Medium':
        return Colors.orangeAccent;
      case 'Low':
        return Colors.greenAccent;
      default:
        return Colors.blueAccent;
    }
  }
}
