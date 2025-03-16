import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:just_audio/just_audio.dart';
import 'package:intl/intl.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

import '../components/bottom_nav_bar.dart';

class UserHistory extends StatefulWidget {
  const UserHistory({Key? key}) : super(key: key);

  @override
  _UserHistoryState createState() => _UserHistoryState();
}

class _UserHistoryState extends State<UserHistory> {
  List<Map<String, dynamic>> userHistoryData = [];
  bool isLoading = true;
  bool isPlaying = false;
  String? currentlyPlaying;
  final AudioPlayer _audioPlayer = AudioPlayer();
  static const bucketName = "agent-audio";
  int _currentIndex = 1;
  final Color primaryColor = Colors.blue;
  final Color secondaryColor = Colors.blue.shade700;

  @override
  void initState() {
    super.initState();
    fetchUserHistory();

    _audioPlayer.playerStateStream.listen((state) {
      if (state.processingState == ProcessingState.completed) {
        setState(() {
          isPlaying = false;
        });
      }
    });
  }

  Future<void> fetchUserHistory() async {
    setState(() => isLoading = true);
    try {
      final response = await Supabase.instance.client
          .from('user-history')
          .select()
          .order('created_at', ascending: false);

      setState(() {
        userHistoryData = List<Map<String, dynamic>>.from(response);
        isLoading = false;
      });
    } catch (e) {
      print('Error fetching user history: $e');
      setState(() => isLoading = false);
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
      return DateFormat('MM/dd/yy • h:mm a').format(adjustedDate);
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
    return WillPopScope(
      onWillPop: () async {
        return false;
      },
      child: Scaffold(
        body: NestedScrollView(
          headerSliverBuilder: (context, innerBoxIsScrolled) {
            return [
              SliverAppBar(
                expandedHeight: 180.0,
                floating: false,
                pinned: true,
                elevation: 0,
                backgroundColor: primaryColor,
                flexibleSpace: FlexibleSpaceBar(
                  title: const Text(
                    'User History',
                    style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  background: Container(
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                        colors: [primaryColor, secondaryColor],
                      ),
                    ),
                    child: Stack(
                      children: [
                        Positioned(
                          right: -50,
                          top: -20,
                          child: Container(
                            width: 150,
                            height: 150,
                            decoration: BoxDecoration(
                              color: Colors.white.withOpacity(0.1),
                              shape: BoxShape.circle,
                            ),
                          ),
                        ),
                        Positioned(
                          left: -30,
                          bottom: -10,
                          child: Container(
                            width: 120,
                            height: 120,
                            decoration: BoxDecoration(
                              color: Colors.white.withOpacity(0.1),
                              shape: BoxShape.circle,
                            ),
                          ),
                        ),
                        Align(
                          alignment: Alignment.center,
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              const SizedBox(height: 30),
                              Icon(
                                FontAwesomeIcons.userGroup,
                                size: 40,
                                color: Colors.white.withOpacity(0.8),
                              ),
                              const SizedBox(height: 8),
                              Text(
                                '${userHistoryData.length} Users',
                                style: TextStyle(
                                  color: Colors.white.withOpacity(0.9),
                                  fontSize: 16,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                actions: [
                  IconButton(
                    icon: const Icon(Icons.refresh, color: Colors.white),
                    onPressed: fetchUserHistory,
                    tooltip: 'Refresh',
                  ),
                  IconButton(
                    icon: const Icon(Icons.search, color: Colors.white),
                    onPressed: () {
                      // Implement search functionality
                    },
                    tooltip: 'Search',
                  ),
                ],
              ),
            ];
          },
          body: isLoading
              ? const Center(child: CircularProgressIndicator())
              : userHistoryData.isEmpty
                  ? _buildEmptyState()
                  : _buildUserHistoryList(),
        ),
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
          Container(
            width: 120,
            height: 120,
            decoration: BoxDecoration(
              color: Colors.grey.shade100,
              shape: BoxShape.circle,
            ),
            child: Icon(
              Icons.person_off,
              size: 60,
              color: Colors.grey.shade400,
            ),
          ),
          const SizedBox(height: 24),
          Text(
            'No user history found',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.w600,
              color: Colors.grey.shade700,
            ),
          ),
          const SizedBox(height: 12),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 40),
            child: Text(
              'User data will appear here once created',
              style: TextStyle(
                fontSize: 16,
                color: Colors.grey.shade500,
              ),
              textAlign: TextAlign.center,
            ),
          ),
          const SizedBox(height: 32),
          ElevatedButton.icon(
            icon: const Icon(Icons.refresh),
            label: const Text('Refresh'),
            onPressed: fetchUserHistory,
            style: ElevatedButton.styleFrom(
              backgroundColor: primaryColor,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(20),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildUserHistoryList() {
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: userHistoryData.length,
      itemBuilder: (context, index) {
        final userData = userHistoryData[index];
        final userId = userData['user_id']?.toString() ?? 'Unknown User';

        // Handle personal_info which could be a Map or a List
        Map<String, dynamic> personalInfo = {};
        if (userData['personal_info'] is Map) {
          personalInfo =
              Map<String, dynamic>.from(userData['personal_info'] as Map);
        } else if (userData['personal_info'] is List &&
            (userData['personal_info'] as List).isNotEmpty) {
          // If it's a list, try to use the first item if it's a map
          final firstItem = (userData['personal_info'] as List).first;
          if (firstItem is Map) {
            personalInfo = Map<String, dynamic>.from(firstItem);
          }
        }

        final profilePic = "https://ehysqseqcnewyndigvfo.supabase"
            ".co/storage/v1/object/public/avatars/${userData['profile_pic']}";

        // Extract personal info details if available
        final name = personalInfo['name'] as String? ?? 'User $userId';
        final email = personalInfo['email'] as String? ?? '';
        final additionalInfo = personalInfo['additional_info'] as String? ?? '';
        final createdAt = _formatDate(userData['created_at']?.toString());

        return Card(
          margin: const EdgeInsets.only(bottom: 16),
          elevation: 3,
          shadowColor: Colors.black.withOpacity(0.1),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          child: InkWell(
            onTap: () => _navigateToUserProfilePage(userData),
            borderRadius: BorderRadius.circular(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Header with gradient
                Container(
                  height: 80,
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: [primaryColor.withOpacity(0.8), secondaryColor],
                    ),
                    borderRadius: const BorderRadius.only(
                      topLeft: Radius.circular(16),
                      topRight: Radius.circular(16),
                    ),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Row(
                      children: [
                        // Profile picture with border
                        Container(
                          width: 50,
                          height: 50,
                          decoration: BoxDecoration(
                            color: Colors.white,
                            shape: BoxShape.circle,
                            border: Border.all(color: Colors.white, width: 2),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withOpacity(0.2),
                                blurRadius: 5,
                                spreadRadius: 1,
                              ),
                            ],
                            image: profilePic != null && profilePic.isNotEmpty
                                ? DecorationImage(
                                    image: NetworkImage(profilePic),
                                    fit: BoxFit.cover,
                                  )
                                : null,
                          ),
                          child: profilePic == null || profilePic.isEmpty
                              ? Icon(
                                  Icons.person,
                                  size: 30,
                                  color: Colors.grey.shade400,
                                )
                              : null,
                        ),
                        const SizedBox(width: 16),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Text(
                                name,
                                style: const TextStyle(
                                  fontWeight: FontWeight.bold,
                                  fontSize: 18,
                                  color: Colors.white,
                                ),
                                overflow: TextOverflow.ellipsis,
                              ),
                              if (email.isNotEmpty) ...[
                                const SizedBox(height: 4),
                                Text(
                                  email,
                                  style: TextStyle(
                                    fontSize: 14,
                                    color: Colors.white.withOpacity(0.9),
                                  ),
                                  overflow: TextOverflow.ellipsis,
                                ),
                              ],
                            ],
                          ),
                        ),
                        IconButton(
                          icon:
                              const Icon(Icons.more_vert, color: Colors.white),
                          onPressed: () {
                            showModalBottomSheet(
                              context: context,
                              shape: const RoundedRectangleBorder(
                                borderRadius: BorderRadius.vertical(
                                  top: Radius.circular(20),
                                ),
                              ),
                              builder: (context) =>
                                  _buildOptionsSheet(userData),
                            );
                          },
                        ),
                      ],
                    ),
                  ),
                ),

                // Content
                Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // User ID and created date - Make this scrollable
                      SingleChildScrollView(
                        scrollDirection: Axis.horizontal,
                        child: Row(
                          children: [
                            _buildInfoTag(Icons.fingerprint, 'ID: $userId'),
                            const SizedBox(width: 12),
                            _buildInfoTag(Icons.calendar_today, createdAt),
                          ],
                        ),
                      ),

                      const SizedBox(height: 16),

                      // Additional info if available
                      if (additionalInfo.isNotEmpty) ...[
                        const Text(
                          'Additional Information',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          additionalInfo,
                          style: TextStyle(
                            fontSize: 14,
                            color: Colors.grey.shade700,
                          ),
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                        ),
                        const SizedBox(height: 16),
                      ],

                      // Action button - Only keep View Profile
                      Center(
                        child: ElevatedButton.icon(
                          icon: const Icon(Icons.visibility, size: 16),
                          label: const Text('View Profile'),
                          onPressed: () => _navigateToUserProfilePage(userData),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: primaryColor,
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(
                                horizontal: 24, vertical: 12),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(20),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildInfoTag(IconData icon, String label) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.grey.shade100,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.grey.shade300),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: Colors.grey.shade700),
          const SizedBox(width: 6),
          Text(
            label,
            style: TextStyle(
              fontSize: 12,
              color: Colors.grey.shade700,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPersonalInfoDetails(Map<String, dynamic> personalInfo) {
    // Filter out already displayed fields
    final displayedKeys = ['name', 'email', 'additional_info'];
    final otherFields = personalInfo.entries
        .where((entry) =>
            !displayedKeys.contains(entry.key) && entry.value != null)
        .toList();

    if (otherFields.isEmpty) {
      return const SizedBox.shrink();
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Divider(),
        const SizedBox(height: 8),
        const Text(
          'User Details',
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 8),
        ...otherFields.map((entry) {
          return Padding(
            padding: const EdgeInsets.only(bottom: 4),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '${_capitalizeFirst(entry.key.replaceAll('_', ' '))}: ',
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                Expanded(
                  child: Text(
                    entry.value.toString(),
                    style: TextStyle(
                      fontSize: 14,
                      color: Colors.grey.shade700,
                    ),
                  ),
                ),
              ],
            ),
          );
        }).toList(),
      ],
    );
  }

  String _capitalizeFirst(String text) {
    if (text.isEmpty) return text;
    return text[0].toUpperCase() + text.substring(1);
  }

  Widget _buildOptionsSheet(Map<String, dynamic> userData) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 20),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 50,
            height: 5,
            margin: const EdgeInsets.only(bottom: 20),
            decoration: BoxDecoration(
              color: Colors.grey.shade300,
              borderRadius: BorderRadius.circular(10),
            ),
          ),
          ListTile(
            leading: Icon(Icons.person, color: primaryColor),
            title: const Text('View Full Profile'),
            onTap: () {
              Navigator.pop(context);
              _navigateToUserProfilePage(userData);
            },
          ),
          ListTile(
            leading: Icon(Icons.edit, color: primaryColor),
            title: const Text('Edit User Info'),
            onTap: () {
              // Implement edit functionality
              Navigator.pop(context);
            },
          ),
          const Divider(),
          ListTile(
            leading: const Icon(Icons.delete, color: Colors.red),
            title: const Text('Delete Record',
                style: TextStyle(color: Colors.red)),
            onTap: () {
              // Implement delete functionality
              Navigator.pop(context);
              // Show confirmation dialog
              showDialog(
                context: context,
                builder: (context) => AlertDialog(
                  title: const Text('Delete User Record'),
                  content: const Text(
                      'Are you sure you want to delete this user record? This action cannot be undone.'),
                  actions: [
                    TextButton(
                      onPressed: () => Navigator.pop(context),
                      child: const Text('Cancel'),
                    ),
                    TextButton(
                      onPressed: () {
                        // Implement delete
                        Navigator.pop(context);
                      },
                      child: const Text('Delete',
                          style: TextStyle(color: Colors.red)),
                    ),
                  ],
                ),
              );
            },
          ),
        ],
      ),
    );
  }

  void _navigateToUserProfilePage(Map<String, dynamic> userData) {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (context) => UserProfilePage(userData: userData),
      ),
    );
  }
}

class UserProfilePage extends StatefulWidget {
  final Map<String, dynamic> userData;

  const UserProfilePage({Key? key, required this.userData}) : super(key: key);

  @override
  _UserProfilePageState createState() => _UserProfilePageState();
}

class _UserProfilePageState extends State<UserProfilePage> {
  List<Map<String, dynamic>> chatHistory = [];
  bool isLoadingChats = true;
  final Color primaryColor = Colors.blue;
  final Color secondaryColor = Colors.blue.shade700;
  int _selectedTabIndex = 0; // 0 for Chats, 1 for Info

  @override
  void initState() {
    super.initState();
    fetchUserChatHistory();
  }

  Future<void> fetchUserChatHistory() async {
    setState(() => isLoadingChats = true);
    try {
      final userId = widget.userData['user_id']?.toString();
      if (userId == null) {
        throw Exception('User ID is null');
      }

      final response = await Supabase.instance.client
          .from('chat-history')
          .select()
          .eq('user_id', userId)
          .order('created_at', ascending: false);
      print(response);
      setState(() {
        chatHistory = List<Map<String, dynamic>>.from(response);
        isLoadingChats = false;
      });
    } catch (e) {
      print('Error fetching chat history: $e');
      setState(() => isLoadingChats = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    // Handle personal_info which could be a Map or a List
    Map<String, dynamic> personalInfo = {};
    List<Map<String, dynamic>> structuredData = [];

    // Process personal_info based on its type
    if (widget.userData['personal_info'] is Map) {
      personalInfo =
          Map<String, dynamic>.from(widget.userData['personal_info'] as Map);
    } else if (widget.userData['personal_info'] is List) {
      // If it's a list, use it directly as structured data
      structuredData = List<Map<String, dynamic>>.from(
          widget.userData['personal_info'] as List);

      // Also extract basic info for the header if available
      if (structuredData.isNotEmpty) {
        for (var item in structuredData) {
          if (item['type'] == 'name') {
            personalInfo['name'] = item['value'];
          } else if (item['type'] == 'email') {
            personalInfo['email'] = item['value'];
          }
        }
      }
    }

    final userId = widget.userData['user_id']?.toString() ?? 'Unknown User';
    final name = personalInfo['name'] as String? ?? 'User $userId';
    final email = personalInfo['email'] as String? ?? '';
    final profilePic =
        "https://ehysqseqcnewyndigvfo.supabase.co/storage/v1/object/public/avatars/${widget.userData['profile_pic']}";
    final timestamp = _formatDate(widget.userData['created_at']?.toString());

    return Scaffold(
      appBar: AppBar(
        title: Text(name),
        elevation: 0, // Remove shadow for a more modern look
        backgroundColor: primaryColor,
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: const Icon(Icons.edit),
            onPressed: () {
              // Implement edit functionality
            },
            tooltip: 'Edit Profile',
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            // Profile header
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [primaryColor, secondaryColor],
                ),
                borderRadius: const BorderRadius.only(
                  bottomLeft: Radius.circular(30),
                  bottomRight: Radius.circular(30),
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.2),
                    blurRadius: 10,
                    offset: const Offset(0, 5),
                  ),
                ],
              ),
              child: Column(
                children: [
                  // Profile picture with border
                  Container(
                    width: 120,
                    height: 120,
                    decoration: BoxDecoration(
                      color: Colors.white,
                      shape: BoxShape.circle,
                      border: Border.all(color: Colors.white, width: 4),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.2),
                          blurRadius: 10,
                          spreadRadius: 2,
                        ),
                      ],
                      image: profilePic != null && profilePic.isNotEmpty
                          ? DecorationImage(
                              image: NetworkImage(profilePic),
                              fit: BoxFit.cover,
                            )
                          : null,
                    ),
                    child: profilePic == null || profilePic.isEmpty
                        ? Icon(
                            Icons.person,
                            size: 60,
                            color: Colors.grey.shade400,
                          )
                        : null,
                  ),
                  const SizedBox(height: 16),
                  // Basic info
                  Column(
                    children: [
                      Text(
                        name,
                        style: const TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                      if (email.isNotEmpty) ...[
                        const SizedBox(height: 4),
                        Text(
                          email,
                          style: const TextStyle(
                            fontSize: 16,
                            color: Colors.white,
                          ),
                        ),
                      ],
                      const SizedBox(height: 12),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          _buildInfoChip(Icons.fingerprint, 'ID: $userId'),
                          const SizedBox(width: 12),
                          _buildInfoChip(Icons.calendar_today, timestamp),
                        ],
                      ),
                    ],
                  ),
                ],
              ),
            ),

            const SizedBox(height: 16),

            // Section tabs
            _buildSectionHeader(),

            const SizedBox(height: 16),

            // Content based on selected tab
            _selectedTabIndex == 0
                ? _buildChatHistorySection()
                : _buildUserInfoSection(structuredData),
          ],
        ),
      ),
    );
  }

  Widget _buildChatHistorySection() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Chat History',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
              IconButton(
                icon: const Icon(Icons.refresh),
                onPressed: fetchUserChatHistory,
                tooltip: 'Refresh',
              ),
            ],
          ),
          // Display chat history
          isLoadingChats
              ? const Center(
                  child: Padding(
                    padding: EdgeInsets.all(24.0),
                    child: CircularProgressIndicator(),
                  ),
                )
              : chatHistory.isEmpty
                  ? _buildEmptyChatHistory()
                  : _buildChatHistoryList(),
        ],
      ),
    );
  }

  Widget _buildUserInfoSection(List<Map<String, dynamic>> structuredData) {
    if (structuredData.isEmpty) {
      return Padding(
        padding: const EdgeInsets.all(16.0),
        child: _buildEmptyUserInfo(),
      );
    }

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'User Information',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
              IconButton(
                icon: const Icon(Icons.info_outline),
                onPressed: () {
                  // Show info about confidence scores
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text(
                          'Confidence scores indicate how certain we are about this information'),
                    ),
                  );
                },
                tooltip: 'About confidence scores',
              ),
            ],
          ),

          // Display structured data items
          ...structuredData
              .where((item) =>
                  item['type'] != null &&
                  item['value'] != null &&
                  item['type'] != 'name' &&
                  item['type'] != 'email')
              .map((item) => _buildInfoItem(item))
              .toList(),
        ],
      ),
    );
  }

  Widget _buildEmptyUserInfo() {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 40, horizontal: 20),
      margin: const EdgeInsets.symmetric(vertical: 16),
      decoration: BoxDecoration(
        color: Colors.grey.shade50,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.grey.shade200),
      ),
      alignment: Alignment.center,
      child: Column(
        children: [
          Icon(
            Icons.person_outline,
            size: 60,
            color: Colors.grey.shade400,
          ),
          const SizedBox(height: 16),
          Text(
            'No user information available',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w500,
              color: Colors.grey.shade600,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Additional user details will appear here when available',
            style: TextStyle(
              fontSize: 14,
              color: Colors.grey.shade500,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildInfoChip(IconData icon, String label) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.2),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: Colors.white),
          const SizedBox(width: 6),
          Text(
            label,
            style: const TextStyle(
              fontSize: 12,
              color: Colors.white,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionHeader() {
    return Container(
      height: 60,
      margin: const EdgeInsets.symmetric(horizontal: 16.0),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(30),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.2),
            spreadRadius: 2,
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          _buildTabItem(
              Icons.chat_bubble_outline, 'Chats', _selectedTabIndex == 0, 0),
          _buildTabItem(Icons.info_outline, 'Info', _selectedTabIndex == 1, 1),
        ],
      ),
    );
  }

  Widget _buildTabItem(IconData icon, String label, bool isActive, int index) {
    return GestureDetector(
      onTap: () {
        setState(() {
          _selectedTabIndex = index;
        });
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 8),
        decoration: BoxDecoration(
          color: isActive ? primaryColor.withOpacity(0.1) : Colors.transparent,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          children: [
            Icon(
              icon,
              size: 20,
              color: isActive ? primaryColor : Colors.grey,
            ),
            const SizedBox(width: 6),
            Text(
              label,
              style: TextStyle(
                fontSize: 14,
                fontWeight: isActive ? FontWeight.bold : FontWeight.normal,
                color: isActive ? primaryColor : Colors.grey,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildEmptyChatHistory() {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 40, horizontal: 20),
      margin: const EdgeInsets.symmetric(vertical: 16),
      decoration: BoxDecoration(
        color: Colors.grey.shade50,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.grey.shade200),
      ),
      alignment: Alignment.center,
      child: Column(
        children: [
          Icon(
            Icons.chat_bubble_outline,
            size: 60,
            color: Colors.grey.shade400,
          ),
          const SizedBox(height: 16),
          Text(
            'No chat history found',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w500,
              color: Colors.grey.shade600,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Conversations with this user will appear here',
            style: TextStyle(
              fontSize: 14,
              color: Colors.grey.shade500,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildChatHistoryList() {
    return ListView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      itemCount: chatHistory.length,
      itemBuilder: (context, index) {
        final chat = chatHistory[index];
        final timestamp = _formatDate(chat['created_at']?.toString());
        final message = chat['message'] as String? ?? 'No message content';

        return Card(
          margin: const EdgeInsets.only(bottom: 16),
          elevation: 2,
          shadowColor: Colors.black.withOpacity(0.1),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Row(
                      children: [
                        Icon(
                          Icons.chat_bubble_outline,
                          size: 18,
                          color: primaryColor,
                        ),
                        const SizedBox(width: 8),
                        const Text(
                          'Conversation',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.grey.shade100,
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        timestamp,
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.grey.shade700,
                        ),
                      ),
                    ),
                  ],
                ),
                const Padding(
                  padding: EdgeInsets.symmetric(vertical: 8.0),
                  child: Divider(),
                ),
                Text(
                  message,
                  style: const TextStyle(fontSize: 14),
                  maxLines: 3,
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 12),
                Row(
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [
                    OutlinedButton.icon(
                      icon: const Icon(Icons.visibility, size: 16),
                      label: const Text('View Full Conversation'),
                      onPressed: () {
                        // Navigate to full conversation view
                      },
                      style: OutlinedButton.styleFrom(
                        foregroundColor: primaryColor,
                        side: BorderSide(color: primaryColor),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(20),
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildInfoItem(Map<String, dynamic> item) {
    final type = item['type'] as String;
    final value = item['value'] as String;

    // Fix the confidence type casting issue
    final dynamic confidenceValue = item['confidence'];
    final double confidence = confidenceValue is int
        ? confidenceValue.toDouble()
        : (confidenceValue as double? ?? 1.0);

    final timestamp = item['timestamp'] as String?;

    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      elevation: 2,
      shadowColor: Colors.black.withOpacity(0.1),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    Icon(
                      _getIconForType(type),
                      size: 18,
                      color: primaryColor,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      _capitalizeFirst(type.replaceAll('_', ' ')),
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: _getConfidenceColor(confidence),
                    borderRadius: BorderRadius.circular(12),
                    boxShadow: [
                      BoxShadow(
                        color: _getConfidenceColor(confidence).withOpacity(0.4),
                        blurRadius: 4,
                        offset: const Offset(0, 2),
                      ),
                    ],
                  ),
                  child: Text(
                    '${(confidence * 100).toInt()}%',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                      color: _getConfidenceTextColor(confidence),
                    ),
                  ),
                ),
              ],
            ),
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 8.0),
              child: Divider(),
            ),
            Text(
              value,
              style: const TextStyle(
                fontSize: 16,
              ),
            ),
            if (timestamp != null) ...[
              const SizedBox(height: 12),
              Row(
                children: [
                  Icon(
                    Icons.access_time,
                    size: 14,
                    color: Colors.grey.shade500,
                  ),
                  const SizedBox(width: 4),
                  Text(
                    'Added: ${_formatTimestamp(timestamp)}',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.grey.shade600,
                      fontStyle: FontStyle.italic,
                    ),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }

  IconData _getIconForType(String type) {
    switch (type.toLowerCase()) {
      case 'phone':
      case 'phone_number':
        return Icons.phone;
      case 'email':
        return Icons.email;
      case 'address':
        return Icons.home;
      case 'birthday':
      case 'birth_date':
        return Icons.cake;
      case 'occupation':
      case 'job':
        return Icons.work;
      case 'education':
        return Icons.school;
      case 'hobby':
      case 'interest':
        return Icons.interests;
      case 'family':
        return Icons.family_restroom;
      case 'location':
        return Icons.location_on;
      default:
        return Icons.info_outline;
    }
  }

  Color _getConfidenceColor(double confidence) {
    if (confidence >= 0.9) return Colors.green.shade100;
    if (confidence >= 0.7) return Colors.lightGreen.shade100;
    if (confidence >= 0.5) return Colors.amber.shade100;
    return Colors.orange.shade100;
  }

  Color _getConfidenceTextColor(double confidence) {
    if (confidence >= 0.9) return Colors.green.shade800;
    if (confidence >= 0.7) return Colors.lightGreen.shade600;
    if (confidence >= 0.5) return Colors.amber.shade800;
    return Colors.orange.shade800;
  }

  String _formatTimestamp(String timestamp) {
    try {
      final date = DateTime.parse(timestamp);
      return DateFormat('MM/dd/yy • h:mm a').format(date);
    } catch (e) {
      return timestamp;
    }
  }

  String _formatDate(String? dateString) {
    if (dateString == null) return 'Unknown date';
    try {
      final date = DateTime.parse(dateString);
      final adjustedDate = date.subtract(const Duration(hours: 4));
      return DateFormat('MM/dd/yy • h:mm a').format(adjustedDate);
    } catch (e) {
      return 'Invalid date';
    }
  }

  String _capitalizeFirst(String text) {
    if (text.isEmpty) return text;
    return text[0].toUpperCase() + text.substring(1);
  }
}
