import 'package:flutter/material.dart';
import 'screens/home_screen.dart';
import 'screens/analysis_screen.dart';
import 'screens/history_screen.dart';
import 'screens/progress_dashboard_screen.dart';
import 'storage/database_service.dart';
import 'services/mock_data_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize Hive database
  await DatabaseService.instance.init();

  // Seed demo data if database is empty
  await MockDataService.seedIfEmpty();

  runApp(const ShadowCoachApp());
}

class ShadowCoachApp extends StatelessWidget {
  const ShadowCoachApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Shadow Coach',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        primaryColor: Colors.redAccent,
        scaffoldBackgroundColor: Colors.grey[900],
        colorScheme: ColorScheme.dark(
          primary: Colors.redAccent,
          secondary: Colors.blueAccent,
          surface: Colors.grey[850]!,
        ),
        appBarTheme: AppBarTheme(
          backgroundColor: Colors.grey[850],
          elevation: 0,
          centerTitle: true,
          titleTextStyle: const TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
            letterSpacing: 1.5,
          ),
        ),
        cardTheme: CardThemeData(
          color: Colors.grey[850],
          elevation: 4,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
      ),
      home: const HomeScreen(),
      routes: {
        '/home': (context) => const HomeScreen(),
        '/record': (context) => const PlaceholderScreen(title: 'Record Video'),
        '/analyze': (context) => const AnalysisScreen(),
        '/history': (context) => const HistoryScreen(),
        '/results': (context) => const PlaceholderScreen(title: 'Results'),
        '/progress': (context) => const ProgressDashboardScreen(),
      },
    );
  }
}

/// Temporary placeholder screen for routes under development
class PlaceholderScreen extends StatelessWidget {
  final String title;

  const PlaceholderScreen({super.key, required this.title});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(
              Icons.construction,
              size: 64,
              color: Colors.orange,
            ),
            const SizedBox(height: 16),
            Text(
              '$title Screen',
              style: const TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'Coming soon...',
              style: TextStyle(fontSize: 16, color: Colors.grey),
            ),
            const SizedBox(height: 32),
            ElevatedButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Go Back'),
            ),
          ],
        ),
      ),
    );
  }
}
