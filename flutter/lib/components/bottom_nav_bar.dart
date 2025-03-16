import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

class CustomBottomAppBar extends StatelessWidget {
  final Function(int)? onItemTapped;
  final int currentIndex;
  final List<BottomNavItem> items;
  final Color? backgroundColor;
  final Color? selectedItemColor;
  final Color? unselectedItemColor;

  const CustomBottomAppBar({
    Key? key,
    this.onItemTapped,
    this.currentIndex = 0,
    required this.items,
    this.backgroundColor,
    this.selectedItemColor,
    this.unselectedItemColor,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final selectedColor = selectedItemColor ?? theme.primaryColor;
    final unselectedColor = unselectedItemColor ?? theme.unselectedWidgetColor;

    return BottomAppBar(
      color: backgroundColor,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: _buildItems(selectedColor, unselectedColor),
      ),
    );
  }

  List<Widget> _buildItems(Color selectedColor, Color unselectedColor) {
    final result = <Widget>[];

    for (int i = 0; i < items.length; i++) {
      result.add(_buildTabItem(
        item: items[i],
        index: i,
        selectedColor: selectedColor,
        unselectedColor: unselectedColor,
      ));
    }

    return result;
  }

  Widget _buildTabItem({
    required BottomNavItem item,
    required int index,
    required Color selectedColor,
    required Color unselectedColor,
  }) {
    final isSelected = index == currentIndex;
    final color = isSelected ? selectedColor : unselectedColor;

    return Expanded(
      child: InkWell(
        onTap: () {
          if (onItemTapped != null) {
            onItemTapped!(index);
          }
        },
        child: Container(
          height: 56.0, // Fixed height for bottom nav
          padding: EdgeInsets.symmetric(vertical: 8.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            mainAxisAlignment:
                MainAxisAlignment.center, // Center content vertically
            children: [
              FaIcon(
                item.isIconData
                    ? item.iconData as IconData
                    : item.faIconData as IconData,
                color: color,
                size: 20, // Slightly smaller icon
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class BottomNavItem {
  final IconData? iconData;
  final IconData? faIconData;
  final String label;
  final bool isIconData;

  // Constructor for Material icons
  const BottomNavItem.material({
    required this.iconData,
    required this.label,
  })  : faIconData = null,
        isIconData = true;

  // Constructor for Font Awesome icons
  const BottomNavItem.fontAwesome({
    required this.faIconData,
    required this.label,
  })  : iconData = null,
        isIconData = false;
}

// Example of the original implementation for reference
class BottomNavBar extends StatelessWidget {
  final int currentIndex;
  final Function(int) onTap;

  const BottomNavBar({
    Key? key,
    required this.currentIndex,
    required this.onTap,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // Use the new CustomBottomAppBar with Font Awesome icons
    return CustomBottomAppBar(
      currentIndex: currentIndex,
      onItemTapped: onTap,
      items: const [
        BottomNavItem.fontAwesome(
            faIconData: FontAwesomeIcons.house, label: 'Home'),
        BottomNavItem.fontAwesome(
            faIconData: FontAwesomeIcons.magnifyingGlass, label: 'Search'),
        BottomNavItem.fontAwesome(
            faIconData: FontAwesomeIcons.heart, label: 'Favorites'),
        BottomNavItem.fontAwesome(
            faIconData: FontAwesomeIcons.user, label: 'Profile'),
      ],
    );
  }
}
