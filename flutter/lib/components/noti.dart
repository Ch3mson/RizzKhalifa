import 'package:flutter/material.dart';
import 'package:toastification/toastification.dart';

/// A utility class for showing notifications throughout the app
class NotificationUtil {
  // Static list to keep track of active notifications
  static final List<ToastificationItem> _activeNotifications = [];
  static const int _maxNotifications = 3;
  static final Toastification _toastification = Toastification();

  /// Shows a notification with customizable parameters
  /// Limited to 3 notifications at a time
  ///
  /// [context] - The BuildContext for showing the notification
  /// [title] - The main message to display
  /// [description] - Optional additional details
  /// [type] - The type of notification (success, error, info, warning)
  /// [backgroundColor] - The background color of the notification
  /// [borderColor] - The border color of the notification
  /// [duration] - How long the notification stays visible
  static void showNotification({
    required BuildContext context,
    required String title,
    String? description,
    ToastificationType type = ToastificationType.success,
    Color backgroundColor = Colors.lightBlue,
    Color borderColor = Colors.blue,
    Duration duration = const Duration(seconds: 2),
  }) {
    if (_activeNotifications.length >= _maxNotifications) {
      final oldestNotification = _activeNotifications.removeAt(0);
      _toastification.dismiss(oldestNotification);
    }

    final notification = _toastification.show(
      context: context,
      title: title,
      description: description,
      type: type,
      style: ToastificationStyle.flatColored,
      autoCloseDuration: duration,
      backgroundColor: backgroundColor,
      primaryColor: borderColor,
      foregroundColor: Colors.black87,
      borderRadius: BorderRadius.circular(8),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      boxShadow: [
        BoxShadow(
          color: borderColor.withOpacity(0.2),
          blurRadius: 8,
          offset: const Offset(0, 4),
        ),
      ],
      callbacks: ToastificationCallbacks(
        onDismissed: (item) {
          _activeNotifications.remove(item);
        },
      ),
    );
    _activeNotifications.add(notification);
  }
}
