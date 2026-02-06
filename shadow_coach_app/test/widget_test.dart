import 'package:flutter_test/flutter_test.dart';
import 'package:shadow_coach_app/main.dart';

void main() {
  testWidgets('Shadow Coach app launches and shows home screen', (WidgetTester tester) async {
    // Build our app and trigger a frame
    await tester.pumpWidget(const ShadowCoachApp());

    // Verify that the app title is displayed
    expect(find.text('SHADOW COACH'), findsOneWidget);

    // Verify that the three main action buttons are present
    expect(find.text('RECORD VIDEO'), findsOneWidget);
    expect(find.text('UPLOAD VIDEO'), findsOneWidget);
    expect(find.text('VIEW HISTORY'), findsOneWidget);

    // Verify that the subtitle is displayed
    expect(find.text('Data-Driven Shadowboxing Analysis'), findsOneWidget);
  });
}
