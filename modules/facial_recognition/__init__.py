"""
Facial Recognition Module.
Provides face detection, recognition, and speaker diarization integration.
"""

from modules.facial_recognition.improved_analysis import ImprovedFaceAnalysis
from modules.facial_recognition.core import FacialRecognitionModule
from modules.facial_recognition.persistence import FacialRecognitionPersistence
from modules.facial_recognition.recognition import FaceRecognition
from modules.facial_recognition.integration import FaceVideoIntegration

__all__ = [
    'ImprovedFaceAnalysis',
    'FacialRecognitionModule',
    'FacialRecognitionPersistence',
    'FaceRecognition',
    'FaceVideoIntegration'
] 