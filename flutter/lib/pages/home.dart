import 'package:flutter/material.dart';
import 'package:just_audio/just_audio.dart';
import 'package:rb_demo/services/cursor_api.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'dart:async';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:rb_demo/components/bottom_nav_bar.dart';
import 'package:toastification/toastification.dart';
import 'package:rb_demo/components/noti.dart';

class HomePage extends StatefulWidget {
  const HomePage({Key? key}) : super(key: key);

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final AudioPlayer _audioPlayer = AudioPlayer();
  final toastification = Toastification();
  List<Map<String, dynamic>> audioFiles = [];
  bool isLoading = false;
  bool isPlaying = false;
  bool isStreaming = false;
  String? currentlyPlaying;
  static const bucketName = "llm";
  StreamSubscription? _subscription;
  RealtimeChannel? _schemaSubscription;
  CursorApi cursorApi = CursorApi();

  int _currentIndex = 0;
  Timer? _streamingTimer;
  bool _canStopStreaming = true;

  @override
  void initState() {
    super.initState();

    _audioPlayer.playerStateStream.listen((state) {
      if (state.processingState == ProcessingState.completed) {
        setState(() {
          isPlaying = false;
        });
      }
    });
  }

  void subscribeToRealTimeUpdates() {
    if (isStreaming) {
      if (!_canStopStreaming) {
        NotificationUtil.showNotification(
          context: context,
          title: 'Please wait at least 10 seconds',
          type: ToastificationType.error,
          backgroundColor: Colors.red.shade100,
          borderColor: Colors.red.shade600,
        );
        return;
      }

      print('Cancelling stream');
      setState(() {
        _subscription?.cancel();
        _schemaSubscription?.unsubscribe();
        isStreaming = false;
      });
      cursorApi.stopTranscription();

      NotificationUtil.showNotification(
        context: context,
        title: 'Stream stopped successfully',
        type: ToastificationType.success,
        backgroundColor: Colors.lightGreen.shade100,
        borderColor: Colors.green.shade600,
      );
    } else {
      print('Starting stream');

      // Set flag to prevent immediate stopping
      _canStopStreaming = false;

      // Start a timer to enforce minimum streaming time
      _streamingTimer?.cancel();
      _streamingTimer = Timer(const Duration(seconds: 10), () {
        _canStopStreaming = true;
        print('Minimum streaming time reached');

        NotificationUtil.showNotification(
          context: context,
          title: 'You can now stop streaming',
          type: ToastificationType.info,
          backgroundColor: Colors.lightBlue.shade100,
          borderColor: Colors.blue.shade600,
        );
      });

      setState(() {
        isStreaming = true;
      });

      cursorApi.startTranscription();

      // Add success toast when starting stream
      NotificationUtil.showNotification(
        context: context,
        title: 'Stream started successfully',
        type: ToastificationType.success,
        backgroundColor: Colors.lightGreen.shade100,
        borderColor: Colors.green.shade600,
      );

      _schemaSubscription = Supabase.instance.client
          .channel('schema-db-changes')
          .onPostgresChanges(
              event: PostgresChangeEvent.insert,
              schema: 'public',
              table: 'agent-audio',
              callback: (payload) {
                print('Database change detected: ${payload.eventType}');
                print('Changed record: ${payload.newRecord}');
                if (payload.newRecord != null &&
                    payload.newRecord!['file_url'] != null) {
                  print(payload.newRecord!['file_url']);
                  final fileUrl = payload.newRecord['file_url'];
                  final fullUrl =
                      "https://ehysqseqcnewyndigvfo.supabase.co/storage/v1/object/public/llm/$fileUrl";

                  print('Fetched file URL: $fullUrl');

                  if (fileUrl is String) {
                    playMp3FromUrl(fullUrl);
                  }
                }
              })
          .subscribe();
    }
  }

  Future<void> playMp3(String fileName) async {
    try {
      if (currentlyPlaying == fileName && isPlaying) {
        await _audioPlayer.pause();
        setState(() {
          isPlaying = false;
        });
        return;
      }
      // If playing a different file or resuming
      if (currentlyPlaying != fileName) {
        final signedUrl = await Supabase.instance.client.storage
            .from(bucketName)
            .createSignedUrl(fileName, 60);
        await _audioPlayer.setUrl(signedUrl);
      }

      await _audioPlayer.play();
      setState(() {
        isPlaying = true;
        currentlyPlaying = fileName;
      });
    } catch (e) {
      print('Error playing MP3: $e');
    }
  }

  // Optional: Add a method to play from URL directly if needed
  Future<void> playMp3FromUrl(String fileUrl) async {
    try {
      await _audioPlayer.setUrl(fileUrl);
      await _audioPlayer.play();
      setState(() {
        isPlaying = true;
        currentlyPlaying = 'Streaming file';
      });
    } catch (e) {
      print('Error playing MP3 from URL: $e');
    }
  }

  @override
  void dispose() {
    _subscription?.cancel();
    _schemaSubscription?.unsubscribe();
    _audioPlayer.dispose();
    super.dispose();
  }

  // Method to handle page navigation
  void navigateToPage(int index) {
    if (index == _currentIndex) return; // Don't navigate if already on the page

    setState(() {
      _currentIndex = index;
    });
    switch (index) {
      case 0:
        break;
      case 1:
        Navigator.pushNamed(context, '/userHistory');
        break;
      case 2:
        Navigator.pushNamed(context, '/chatHistory');
        break;
      case 3:
        Navigator.pushNamed(context, '/settings');
        break;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Audio Streamer',
            style: TextStyle(fontWeight: FontWeight.bold)),
        elevation: 0,
        backgroundColor: Theme.of(context).scaffoldBackgroundColor,
        centerTitle: true,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Minimal status text
            Text(
              isStreaming ? 'STREAMING' : 'STREAM',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w600,
                letterSpacing: 2,
                color:
                    isStreaming ? Colors.green.shade700 : Colors.grey.shade700,
              ),
            ),

            const SizedBox(height: 30),

            // Cool animated button
            GestureDetector(
              onTap: subscribeToRealTimeUpdates,
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 500),
                width: isStreaming ? 200 : 180,
                height: isStreaming ? 200 : 180,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: isStreaming
                        ? [Colors.green.shade400, Colors.teal.shade700]
                        : [Colors.blue.shade300, Colors.indigo.shade600],
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: isStreaming
                          ? Colors.green.withOpacity(0.6)
                          : Colors.blue.withOpacity(0.4),
                      blurRadius: isStreaming ? 30 : 15,
                      spreadRadius: isStreaming ? 5 : 2,
                    ),
                  ],
                ),
                child: Center(
                  child: AnimatedSwitcher(
                    duration: const Duration(milliseconds: 300),
                    transitionBuilder:
                        (Widget child, Animation<double> animation) {
                      return ScaleTransition(scale: animation, child: child);
                    },
                    child: isStreaming
                        ? Stack(
                            key: const ValueKey('streaming'),
                            alignment: Alignment.center,
                            children: [
                              // Pulsing circles for streaming effect
                              ...List.generate(3, (index) {
                                return AnimatedBuilder(
                                  animation: AlwaysStoppedAnimation(index / 3),
                                  builder: (context, child) {
                                    return Container(
                                      width: 120 + (index * 20),
                                      height: 120 + (index * 20),
                                      decoration: BoxDecoration(
                                        shape: BoxShape.circle,
                                        border: Border.all(
                                          color: Colors.white
                                              .withOpacity(0.8 - (index * 0.2)),
                                          width: 2,
                                        ),
                                      ),
                                    );
                                  },
                                );
                              }),
                              const Icon(
                                Icons.podcasts,
                                size: 80,
                                color: Colors.white,
                              ),
                            ],
                          )
                        : const Icon(
                            Icons.play_arrow_rounded,
                            size: 80,
                            color: Colors.white,
                            key: ValueKey('not-streaming'),
                          ),
                  ),
                ),
              ),
            ),

            // Now Playing indicator (only when streaming and playing)
            if (isStreaming && isPlaying)
              Padding(
                padding: const EdgeInsets.only(top: 30),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.volume_up,
                        color: Colors.green.shade700, size: 18),
                    const SizedBox(width: 8),
                    Text(
                      'PLAYING',
                      style: TextStyle(
                        color: Colors.green.shade700,
                        fontWeight: FontWeight.w600,
                        letterSpacing: 1,
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
      bottomNavigationBar: CustomBottomAppBar(
        currentIndex: 0,
        onItemTapped: navigateToPage,
        items: const [
          BottomNavItem.fontAwesome(
              faIconData: FontAwesomeIcons.home, label: 'Home'),
          BottomNavItem.fontAwesome(
              faIconData: FontAwesomeIcons.userGroup, label: 'User-Info'),
          BottomNavItem.fontAwesome(
              faIconData: FontAwesomeIcons.clockRotateLeft, label: 'History'),
          BottomNavItem.fontAwesome(
              faIconData: FontAwesomeIcons.gear, label: 'Settings'),
        ],
        selectedItemColor: Colors.blue,
      ),
    );
  }
}
