import 'package:flutter/material.dart';

/// Home screen - main entry point for Shadow Coach app
/// Provides three primary actions: Record, Upload, View History
class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[900],
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // App branding
              const Icon(
                Icons.sports_martial_arts,
                size: 80,
                color: Colors.redAccent,
              ),
              const SizedBox(height: 16),
              const Text(
                'SHADOW COACH',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                  letterSpacing: 2,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'Data-Driven Shadowboxing Analysis',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.grey[400],
                  letterSpacing: 1,
                ),
              ),
              const SizedBox(height: 40),

              // Record Video Button
              _ActionButton(
                icon: Icons.videocam,
                label: 'RECORD VIDEO',
                color: Colors.redAccent,
                onPressed: () {
                  Navigator.pushNamed(context, '/record');
                },
              ),
              const SizedBox(height: 20),

              // Upload Video Button
              _ActionButton(
                icon: Icons.upload_file,
                label: 'UPLOAD VIDEO',
                color: Colors.blueAccent,
                onPressed: () {
                  Navigator.pushNamed(context, '/analyze');
                },
              ),
              const SizedBox(height: 20),

              // View History Button
              _ActionButton(
                icon: Icons.history,
                label: 'VIEW HISTORY',
                color: Colors.greenAccent,
                onPressed: () {
                  Navigator.pushNamed(context, '/history');
                },
              ),
              const SizedBox(height: 20),

              // View Progress Button
              _ActionButton(
                icon: Icons.insights,
                label: 'VIEW PROGRESS',
                color: Colors.orangeAccent,
                onPressed: () {
                  Navigator.pushNamed(context, '/progress');
                },
              ),
              const SizedBox(height: 40),

              // Footer info
              Text(
                'Version 1.0.0',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.grey[600],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Reusable action button widget
class _ActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback onPressed;

  const _ActionButton({
    required this.icon,
    required this.label,
    required this.color,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: onPressed,
      style: ElevatedButton.styleFrom(
        backgroundColor: color,
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(vertical: 20),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
        elevation: 4,
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, size: 28),
          const SizedBox(width: 12),
          Text(
            label,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              letterSpacing: 1,
            ),
          ),
        ],
      ),
    );
  }
}
