import 'package:flutter/material.dart';
import 'package:rb_demo/pages/home.dart';
import 'package:rb_demo/pages/history.dart';
import 'package:rb_demo/pages/settings.dart';

import '../pages/user-history.dart';

class AppRoutes {
  static const String home = '/';
  static const String userHistory = '/userHistory';
  static const String chatHistory = '/chatHistory';
  static const String settings = '/settings';

  // Define routes
  static Map<String, WidgetBuilder> routes = {
    home: (context) => const HomePage(),
    userHistory: (context) => const UserHistory(),
    chatHistory: (context) => const HistoryPage(),
    settings: (context) => const SettingsPage(),
  };

  // For handling named routes with arguments
  static Route<dynamic> generateRoute(RouteSettings settings) {
    final String? routeName = settings.name;

    if (routeName == home) {
      return MaterialPageRoute(builder: (context) => const HomePage());
    } else if (routeName == userHistory) {
      return MaterialPageRoute(builder: (context) => const HistoryPage());
    } else if (routeName == settings) {
      return MaterialPageRoute(builder: (context) => const SettingsPage());
    } else if (routeName == chatHistory) {
      return MaterialPageRoute(builder: (context) => const UserHistory());
    } else {
      return MaterialPageRoute(
        builder: (context) => Scaffold(
          body: Center(
            child: Text('No route defined for $routeName'),
          ),
        ),
      );
    }
  }
}
