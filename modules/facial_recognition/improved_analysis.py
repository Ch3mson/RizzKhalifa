"""
Improved face analysis module built on top of InsightFace.
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis

class ImprovedFaceAnalysis(FaceAnalysis):
    """
    Extension of InsightFace's FaceAnalysis that offers improved face detection capabilities.
    This class provides better handling of detection sizes and implements adaptive detection.
    """
    
    def get_with_multiple_sizes(self, img, max_num=0, sizes=None):
        """
        Attempts to detect faces using multiple detection sizes.
        
        Args:
            img: Input image
            max_num: Maximum number of faces to detect (0 for unlimited)
            sizes: List of detection sizes to try, e.g. [(640, 640), (320, 320)]
            
        Returns:
            List of detected faces
        """
        if sizes is None:
            sizes = [(640, 640), (320, 320), (480, 480), (720, 720), (960, 960)]
        
        print(f"Trying to detect faces with multiple detection sizes: {sizes}")
        faces = None
        
        for det_size in sizes:
            try:
                if hasattr(self.det_model, "input_size"):
                    self.det_model.input_size = det_size
                
                faces = self.get(img, max_num)
                if faces and len(faces) > 0:
                    print(f"Successfully detected {len(faces)} faces with detection size {det_size}")
                    return faces
            except Exception as e:
                print(f"Error with detection size {det_size}: {e}")
                continue
        
        if not faces or len(faces) == 0:
            print("No faces detected with any detection size")
            return []
        
        return faces 