import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:just_audio/just_audio.dart';
import 'package:intl/intl.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:rb_demo/components/bottom_nav_bar.dart';

class HistoryPage extends StatefulWidget {
  const HistoryPage({Key? key}) : super(key: key);

  @override
  _HistoryPageState createState() => _HistoryPageState();
}

class _HistoryPageState extends State<HistoryPage> {
  List<Map<String, dynamic>> audioFiles = [];
  bool isLoading = true;
  bool isPlaying = false;
  String? currentlyPlaying;
  final AudioPlayer _audioPlayer = AudioPlayer();
  static const bucketName = "agent-audio";
  int _currentIndex = 2; // Set to 1 since this is the History page

  @override
  void initState() {
    super.initState();
    fetchAudioHistory();

    _audioPlayer.playerStateStream.listen((state) {
      if (state.processingState == ProcessingState.completed) {
        setState(() {
          isPlaying = false;
        });
      }
    });
  }

  Future<void> fetchAudioHistory() async {
    setState(() => isLoading = true);
    try {
      final response = await Supabase.instance.client
          .from('agent-audio')
          .select()
          .order('created_at', ascending: false);

      setState(() {
        audioFiles = List<Map<String, dynamic>>.from(response);
        isLoading = false;
      });
    } catch (e) {
      print('Error fetching audio history: $e');
      setState(() => isLoading = false);
    }
  }

  Future<void> playAudio(String fileUrl, String fileName) async {
    print(fileUrl);
    try {
      fileUrl =
          "https://ehysqseqcnewyndigvfo.supabase.co/storage/v1/object/public/llm/$fileUrl";
      if (currentlyPlaying == fileName && isPlaying) {
        await _audioPlayer.pause();
        setState(() {
          isPlaying = false;
        });
        return;
      }

      // If it's a full URL, use it directly
      if (fileUrl.startsWith('http')) {
        await _audioPlayer.setUrl(fileUrl);
      }

      await _audioPlayer.play();
      setState(() {
        isPlaying = true;
        currentlyPlaying = fileName;
      });
    } catch (e) {
      print('Error playing audio: $e');
    }
  }

  @override
  void dispose() {
    _audioPlayer.dispose();
    super.dispose();
  }

  String _formatDate(String? dateString) {
    if (dateString == null) return 'Unknown date';
    try {
      final date = DateTime.parse(dateString);
      // Subtract 4 hours to adjust for EST timezone
      final adjustedDate = date.subtract(const Duration(hours: 4));
      return DateFormat('MMM d, yyyy • h:mm a').format(adjustedDate);
    } catch (e) {
      return 'Invalid date';
    }
  }

  void navigateToPage(int index) {
    if (index == _currentIndex) return; // Don't navigate if already on the page

    setState(() {
      _currentIndex = index;
    });
    switch (index) {
      case 0:
        Navigator.pushNamed(context, '/');
        break;
      case 1:
        Navigator.pushNamed(context, '/userHistory');
        break;
      case 2:
        break;
      case 3:
        Navigator.pushNamed(context, '/settings');
        break;
    }
  }

  @override
  Widget build(BuildContext context) {
    return WillPopScope(
      // Prevent back button navigation
      onWillPop: () async => false,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Audio History'),
          automaticallyImplyLeading: false,
          actions: [
            IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: fetchAudioHistory,
              tooltip: 'Refresh',
            ),
          ],
        ),
        body: isLoading
            ? const Center(child: CircularProgressIndicator())
            : audioFiles.isEmpty
                ? _buildEmptyState()
                : _buildAudioList(),
        bottomNavigationBar: CustomBottomAppBar(
          currentIndex: _currentIndex,
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
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.history,
            size: 80,
            color: Colors.grey.shade400,
          ),
          const SizedBox(height: 16),
          Text(
            'No audio history found',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w500,
              color: Colors.grey.shade600,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Audio files will appear here once created',
            style: TextStyle(
              fontSize: 14,
              color: Colors.grey.shade500,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAudioList() {
    return ListView.builder(
      padding: const EdgeInsets.symmetric(vertical: 8),
      itemCount: audioFiles.length + 1, // +1 for the sentiment overview
      itemBuilder: (context, index) {
        // Show sentiment overview at the top
        if (index == 0) {
          return _buildSentimentAggregateView();
        }

        // Adjust index for the actual audio files
        final fileIndex = index - 1;
        final file = audioFiles[fileIndex];
        final fileName = file['summary'] ?? file['file_name'];
        final fileUrl = file['file_url'] ?? '';
        final isCurrentlyPlaying = currentlyPlaying == fileName;

        // Extract sentiment data if available
        final Map<String, dynamic>? sentimentData = file['sentiment'];

        return Card(
          margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
          elevation: 2,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          child: Column(
            children: [
              ListTile(
                contentPadding: const EdgeInsets.all(12),
                leading: Container(
                  width: 48,
                  height: 48,
                  decoration: BoxDecoration(
                    color: isCurrentlyPlaying
                        ? Colors.blue.shade100
                        : Colors.grey.shade200,
                    borderRadius: BorderRadius.circular(24),
                  ),
                  child: IconButton(
                    icon: Icon(
                      isCurrentlyPlaying && isPlaying
                          ? Icons.pause
                          : Icons.play_arrow,
                      color: isCurrentlyPlaying
                          ? Colors.blue
                          : Colors.grey.shade700,
                    ),
                    onPressed: () => playAudio(fileUrl, fileName),
                  ),
                ),
                title: Text(
                  fileName,
                  style: const TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: 16,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
                subtitle: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const SizedBox(height: 4),
                    Text(
                      _formatDate(file['created_at']),
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey.shade600,
                      ),
                    ),
                    if (file['duration'] != null) ...[
                      const SizedBox(height: 4),
                      Text(
                        'Duration: ${file['duration']} seconds',
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.grey.shade600,
                        ),
                      ),
                    ],
                    if (sentimentData != null) ...[
                      const SizedBox(height: 4),
                      _buildSentimentSummary(sentimentData),
                    ],
                  ],
                ),
                trailing: IconButton(
                  icon: const Icon(Icons.more_vert),
                  onPressed: () {
                    // Show options menu (download, share, delete, etc.)
                    showModalBottomSheet(
                      context: context,
                      builder: (context) => _buildOptionsSheet(file),
                    );
                  },
                ),
              ),
              // Display detailed sentiment data if available
              if (sentimentData != null)
                Padding(
                  padding:
                      const EdgeInsets.only(left: 16, right: 16, bottom: 12),
                  child: _buildSentimentDetails(sentimentData),
                ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildOptionsSheet(Map<String, dynamic> file) {
    final Map<String, dynamic>? sentimentData = file['sentiment'];

    return Container(
      padding: const EdgeInsets.symmetric(vertical: 20),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Add sentiment header if sentiment data exists
          if (sentimentData != null) ...[
            Padding(
              padding: const EdgeInsets.only(left: 16, right: 16, bottom: 16),
              child: _buildSentimentHeader(sentimentData),
            ),
            const Divider(),
          ],
          ListTile(
            leading: const Icon(Icons.download),
            title: const Text('Download'),
            onTap: () {
              // Implement download functionality
              Navigator.pop(context);
            },
          ),
          ListTile(
            leading: const Icon(Icons.share),
            title: const Text('Share'),
            onTap: () {
              // Implement share functionality
              Navigator.pop(context);
            },
          ),
          ListTile(
            leading: const Icon(Icons.delete, color: Colors.red),
            title: const Text('Delete', style: TextStyle(color: Colors.red)),
            onTap: () {
              // Implement delete functionality
              Navigator.pop(context);
              // Show confirmation dialog
            },
          ),
        ],
      ),
    );
  }

  // New method to build sentiment header for the popup
  Widget _buildSentimentHeader(Map<String, dynamic> sentimentData) {
    final emotion = sentimentData['emotion']?.toString() ?? 'unknown';
    final sentiment = sentimentData['sentiment']?.toString() ?? 'neutral';
    final confidence = sentimentData['confidence'] ?? 0.0;
    final scores = sentimentData['scores'] as Map<String, dynamic>?;

    // Choose color based on sentiment
    Color sentimentColor;
    switch (sentiment.toLowerCase()) {
      case 'positive':
        sentimentColor = Colors.green;
        break;
      case 'negative':
        sentimentColor = Colors.red;
        break;
      case 'neutral':
      default:
        sentimentColor = Colors.grey;
        break;
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(
              _getEmotionIcon(emotion),
              size: 28,
              color: sentimentColor,
            ),
            const SizedBox(width: 12),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Emotion: ${_capitalizeFirst(emotion)}',
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  'Sentiment: ${_capitalizeFirst(sentiment)}',
                  style: TextStyle(
                    fontSize: 16,
                    color: sentimentColor,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
          ],
        ),
        const SizedBox(height: 12),
        Text(
          'Confidence: ${(confidence * 100).toStringAsFixed(1)}%',
          style: const TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w500,
          ),
        ),

        // Add score bars if scores exist
        if (scores != null) ...[
          const SizedBox(height: 16),
          const Text(
            'Sentiment Scores:',
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: _buildScoreBar('Positive',
                    scores['positive']?.toDouble() ?? 0.0, Colors.green),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _buildScoreBar('Negative',
                    scores['negative']?.toDouble() ?? 0.0, Colors.red),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Row(
            children: [
              Expanded(
                child: _buildScoreBar('Neutral',
                    scores['neutral']?.toDouble() ?? 0.0, Colors.grey),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _buildScoreBar(
                    'Compound',
                    (scores['compound']?.toDouble() ?? 0.0).abs(),
                    (scores['compound']?.toDouble() ?? 0.0) >= 0
                        ? Colors.blue
                        : Colors.purple),
              ),
            ],
          ),
        ],
      ],
    );
  }

  Widget _buildSentimentAggregateView() {
    if (audioFiles.isEmpty) return const SizedBox.shrink();

    // Calculate aggregate sentiment data
    int totalFiles = 0;
    int positiveCount = 0;
    int negativeCount = 0;
    int neutralCount = 0;
    Map<String, int> emotionCounts = {};
    double avgConfidence = 0;

    for (var file in audioFiles) {
      final sentiment = file['sentiment'];
      if (sentiment != null) {
        totalFiles++;

        switch (sentiment['sentiment']?.toString().toLowerCase()) {
          case 'positive':
            positiveCount++;
            break;
          case 'negative':
            negativeCount++;
            break;
          case 'neutral':
          default:
            neutralCount++;
            break;
        }

        // Count emotions
        final emotion = sentiment['emotion']?.toString() ?? 'unknown';
        emotionCounts[emotion] = (emotionCounts[emotion] ?? 0) + 1;

        // Sum confidence for average
        avgConfidence += sentiment['confidence']?.toDouble() ?? 0;
      }
    }

    // Calculate average confidence
    avgConfidence = totalFiles > 0 ? avgConfidence / totalFiles : 0;

    // Find most common emotion
    String mostCommonEmotion = 'none';
    int maxCount = 0;
    emotionCounts.forEach((emotion, count) {
      if (count > maxCount) {
        maxCount = count;
        mostCommonEmotion = emotion;
      }
    });

    return Card(
      margin: const EdgeInsets.all(16),
      elevation: 3,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Sentiment Overview',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            Text(
              'Based on $totalFiles analyzed recordings',
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey.shade600,
              ),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: _buildSentimentStat(
                    'Positive',
                    totalFiles > 0 ? positiveCount / totalFiles : 0,
                    Colors.green,
                    Icons.sentiment_very_satisfied,
                  ),
                ),
                Expanded(
                  child: _buildSentimentStat(
                    'Neutral',
                    totalFiles > 0 ? neutralCount / totalFiles : 0,
                    Colors.grey,
                    Icons.sentiment_neutral,
                  ),
                ),
                Expanded(
                  child: _buildSentimentStat(
                    'Negative',
                    totalFiles > 0 ? negativeCount / totalFiles : 0,
                    Colors.red,
                    Icons.sentiment_very_dissatisfied,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Icon(
                  _getEmotionIcon(mostCommonEmotion),
                  size: 24,
                  color: Colors.blue,
                ),
                const SizedBox(width: 8),
                Text(
                  'Most common emotion: ${_capitalizeFirst(mostCommonEmotion)}',
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              'Average confidence: ${(avgConfidence * 100).toStringAsFixed(1)}%',
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey.shade700,
              ),
            ),

            // Add a section for individual audio entries summary
            if (totalFiles > 0) ...[
              const SizedBox(height: 16),
              const Divider(),
              const SizedBox(height: 8),
              const Text(
                'Recent Recordings',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 8),
              _buildRecentEntriesSummary(),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildSentimentStat(
      String label, double percentage, Color color, IconData icon) {
    return Column(
      children: [
        Stack(
          alignment: Alignment.center,
          children: [
            SizedBox(
              height: 60,
              width: 60,
              child: CircularProgressIndicator(
                value: percentage,
                backgroundColor: color.withOpacity(0.2),
                valueColor: AlwaysStoppedAnimation<Color>(color),
                strokeWidth: 8,
              ),
            ),
            Icon(icon, color: color),
          ],
        ),
        const SizedBox(height: 8),
        Text(
          label,
          style: const TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w500,
          ),
        ),
        Text(
          '${(percentage * 100).toStringAsFixed(0)}%',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
      ],
    );
  }

  IconData _getEmotionIcon(String emotion) {
    switch (emotion.toString().toLowerCase()) {
      case 'happy':
        return Icons.sentiment_very_satisfied;
      case 'sad':
        return Icons.sentiment_dissatisfied;
      case 'angry':
        return Icons.sentiment_very_dissatisfied;
      case 'fear':
        return Icons.mood_bad;
      case 'surprise':
        return Icons.sentiment_satisfied_alt;
      default:
        return Icons.sentiment_neutral;
    }
  }

  // Widget to display a summary of sentiment data
  Widget _buildSentimentSummary(Map<String, dynamic> sentimentData) {
    final emotion = sentimentData['emotion']?.toString() ?? 'unknown';
    final sentiment = sentimentData['sentiment']?.toString() ?? 'neutral';
    final confidence = sentimentData['confidence'] ?? 0.0;

    // Choose color based on sentiment
    Color sentimentColor;
    switch (sentiment.toString().toLowerCase()) {
      case 'positive':
        sentimentColor = Colors.green;
        break;
      case 'negative':
        sentimentColor = Colors.red;
        break;
      case 'neutral':
      default:
        sentimentColor = Colors.grey;
        break;
    }

    return Row(
      children: [
        Icon(
          _getEmotionIcon(emotion),
          size: 14,
          color: sentimentColor,
        ),
        const SizedBox(width: 4),
        Text(
          '${_capitalizeFirst(emotion)} • ${(confidence * 100).toStringAsFixed(0)}% confidence',
          style: TextStyle(
            fontSize: 12,
            color: sentimentColor,
            fontWeight: FontWeight.w500,
          ),
        ),
      ],
    );
  }

  // Add this helper method to replace the extension
  String _capitalizeFirst(String text) {
    if (text.isEmpty) return text;
    return text[0].toUpperCase() + text.substring(1);
  }

  // Widget to display detailed sentiment scores
  Widget _buildSentimentDetails(Map<String, dynamic> sentimentData) {
    final scores = sentimentData['scores'] as Map<String, dynamic>?;
    if (scores == null) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Sentiment Analysis',
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: _buildScoreBar('Positive',
                  scores['positive']?.toDouble() ?? 0.0, Colors.green),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _buildScoreBar('Negative',
                  scores['negative']?.toDouble() ?? 0.0, Colors.red),
            ),
          ],
        ),
        const SizedBox(height: 4),
        Row(
          children: [
            Expanded(
              child: _buildScoreBar(
                  'Neutral', scores['neutral']?.toDouble() ?? 0.0, Colors.grey),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _buildScoreBar(
                  'Compound',
                  (scores['compound']?.toDouble() ?? 0.0).abs(),
                  (scores['compound']?.toDouble() ?? 0.0) >= 0
                      ? Colors.blue
                      : Colors.purple),
            ),
          ],
        ),
      ],
    );
  }

  // Helper widget to display a score bar
  Widget _buildScoreBar(String label, double value, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          '$label: ${(value * 100).toStringAsFixed(0)}%',
          style: const TextStyle(fontSize: 12),
        ),
        const SizedBox(height: 4),
        LinearProgressIndicator(
          value: value,
          backgroundColor: color.withOpacity(0.2),
          valueColor: AlwaysStoppedAnimation<Color>(color),
          minHeight: 8,
          borderRadius: BorderRadius.circular(4),
        ),
      ],
    );
  }

  // Add a new method to show recent entries summary
  Widget _buildRecentEntriesSummary() {
    // Show at most 3 recent entries
    final entriesToShow = audioFiles.length > 3 ? 3 : audioFiles.length;

    return Column(
      children: List.generate(entriesToShow, (index) {
        final file = audioFiles[index];
        final fileName = file['summary'] ?? file['file_name'];
        final Map<String, dynamic>? sentimentData = file['sentiment'];

        if (sentimentData == null) {
          return const SizedBox.shrink();
        }

        final emotion = sentimentData['emotion']?.toString() ?? 'unknown';
        final sentiment = sentimentData['sentiment']?.toString() ?? 'neutral';
        final confidence = sentimentData['confidence'] ?? 0.0;

        // Choose color based on sentiment
        Color sentimentColor;
        switch (sentiment.toLowerCase()) {
          case 'positive':
            sentimentColor = Colors.green;
            break;
          case 'negative':
            sentimentColor = Colors.red;
            break;
          case 'neutral':
          default:
            sentimentColor = Colors.grey;
            break;
        }

        return Padding(
          padding: const EdgeInsets.only(bottom: 8.0),
          child: Row(
            children: [
              Icon(
                _getEmotionIcon(emotion),
                size: 16,
                color: sentimentColor,
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      fileName,
                      style: const TextStyle(
                        fontWeight: FontWeight.w500,
                        fontSize: 14,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    Text(
                      '${_capitalizeFirst(emotion)} • ${_capitalizeFirst(sentiment)} • ${(confidence * 100).toStringAsFixed(0)}% confidence',
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey.shade600,
                      ),
                    ),
                  ],
                ),
              ),
              Text(
                _formatDate(file['created_at']).split('•')[0].trim(),
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.grey.shade500,
                ),
              ),
            ],
          ),
        );
      }),
    );
  }
}
