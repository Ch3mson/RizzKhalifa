"""
Migration helper for facial recognition module.
Provides utilities to migrate from the original facial_recognition.py to the new modular structure.
"""

import os
import sys
import logging
import shutil
from pathlib import Path
import importlib.util

class FacialRecognitionMigrator:
    """
    Helper class to migrate from the old monolithic structure to the new modular structure.
    """
    
    def __init__(self, 
                 original_module_path: str = "modules/facial_recognition.py",
                 backup_path: str = "modules/facial_recognition.py.bak"):
        """
        Initialize the migration helper.
        
        Args:
            original_module_path: Path to the original facial_recognition.py
            backup_path: Path to store a backup of the original file
        """
        self.logger = logging.getLogger("facial_recognition.migration")
        self.original_module_path = original_module_path
        self.backup_path = backup_path
        
    def create_backup(self) -> bool:
        """
        Create a backup of the original facial_recognition.py file.
        
        Returns:
            bool: Whether the backup was successful
        """
        try:
            if os.path.exists(self.original_module_path):
                shutil.copy2(self.original_module_path, self.backup_path)
                self.logger.info(f"Created backup of original module at {self.backup_path}")
                return True
            else:
                self.logger.error(f"Original module not found at {self.original_module_path}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return False
            
    def restore_backup(self) -> bool:
        """
        Restore the original module from backup.
        
        Returns:
            bool: Whether the restore was successful
        """
        try:
            if os.path.exists(self.backup_path):
                shutil.copy2(self.backup_path, self.original_module_path)
                self.logger.info(f"Restored original module from {self.backup_path}")
                return True
            else:
                self.logger.error(f"Backup not found at {self.backup_path}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to restore backup: {e}")
            return False
            
    def check_compatibility(self) -> bool:
        """
        Check if the original module is compatible with the migration.
        
        Returns:
            bool: Whether the module is compatible
        """
        try:
            if not os.path.exists(self.original_module_path):
                self.logger.error(f"Original module not found at {self.original_module_path}")
                return False
                
            # Import the original module
            module_name = "facial_recognition_original"
            spec = importlib.util.spec_from_file_location(module_name, self.original_module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for required classes
            if not hasattr(module, "ImprovedFaceAnalysis"):
                self.logger.error("ImprovedFaceAnalysis class not found in original module")
                return False
                
            if not hasattr(module, "FacialRecognitionModule"):
                self.logger.error("FacialRecognitionModule class not found in original module")
                return False
                
            # Basic check passed
            self.logger.info("Original module is compatible with migration")
            return True
        except Exception as e:
            self.logger.error(f"Failed to check compatibility: {e}")
            return False
            
    def create_migration_module(self) -> bool:
        """
        Create a module that provides backward compatibility.
        
        Returns:
            bool: Whether the creation was successful
        """
        try:
            migration_path = "modules/facial_recognition_compat.py"
            
            with open(migration_path, "w") as f:
                f.write("""
# This is an automatically generated compatibility module for facial_recognition.py
# It provides backward compatibility with code that uses the original module.

import logging
import warnings

# Import from the new modular structure
from modules.facial_recognition import (
    ImprovedFaceAnalysis,
    FacialRecognitionModule,
    FacialRecognitionPersistence,
    FaceRecognition,
    FaceVideoIntegration
)

# Show a deprecation warning
warnings.warn(
    "You are using the compatibility module for facial_recognition.py. "
    "Please update your imports to use the new modular structure: "
    "from modules.facial_recognition import ...",
    DeprecationWarning,
    stacklevel=2
)

logger = logging.getLogger("facial_recognition_compat")
logger.info("Using facial_recognition compatibility module")

# Re-export the classes for backward compatibility
__all__ = [
    'ImprovedFaceAnalysis',
    'FacialRecognitionModule'
]
                """)
                
            self.logger.info(f"Created migration module at {migration_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create migration module: {e}")
            return False
            
    def update_original_module(self) -> bool:
        """
        Update the original module to import from the new structure.
        
        Returns:
            bool: Whether the update was successful
        """
        try:
            with open(self.original_module_path, "w") as f:
                f.write("""
# This file is maintained for backward compatibility.
# It imports from the new modular structure.

import logging
import warnings

# Import from the new modular structure
from modules.facial_recognition import (
    ImprovedFaceAnalysis,
    FacialRecognitionModule
)

# Show a deprecation warning
warnings.warn(
    "You are using the original facial_recognition.py module which is now deprecated. "
    "Please update your imports to use the new modular structure: "
    "from modules.facial_recognition import ...",
    DeprecationWarning,
    stacklevel=2
)

logger = logging.getLogger("facial_recognition")
logger.info("Using deprecated facial_recognition.py - please update your imports")

# Re-export the classes for backward compatibility
__all__ = [
    'ImprovedFaceAnalysis',
    'FacialRecognitionModule'
]
                """)
                
            self.logger.info(f"Updated original module at {self.original_module_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update original module: {e}")
            return False
            
    def run_migration(self) -> bool:
        """
        Run the complete migration process.
        
        Returns:
            bool: Whether the migration was successful
        """
        try:
            # Create backup first
            if not self.create_backup():
                self.logger.error("Migration aborted: Failed to create backup")
                return False
                
            # Check compatibility
            if not self.check_compatibility():
                self.logger.error("Migration aborted: Compatibility check failed")
                return False
                
            # Create migration module
            if not self.create_migration_module():
                self.logger.error("Migration failed: Could not create migration module")
                return False
                
            # Update original module
            if not self.update_original_module():
                self.logger.error("Migration failed: Could not update original module")
                self.restore_backup()
                return False
                
            self.logger.info("Migration completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Migration failed with error: {e}")
            self.restore_backup()
            return False
            
def run_migration():
    """
    Run the migration process from command line.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("facial_recognition_migration")
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Migrate facial_recognition.py to modular structure")
    parser.add_argument("--original", default="modules/facial_recognition.py", help="Path to original module")
    parser.add_argument("--backup", default="modules/facial_recognition.py.bak", help="Path for backup")
    args = parser.parse_args()
    
    # Run migration
    migrator = FacialRecognitionMigrator(
        original_module_path=args.original,
        backup_path=args.backup
    )
    
    success = migrator.run_migration()
    
    if success:
        logger.info("Migration completed successfully")
        return 0
    else:
        logger.error("Migration failed")
        return 1
        
if __name__ == "__main__":
    sys.exit(run_migration()) 