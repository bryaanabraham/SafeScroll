"""
Singleton Face Detector using InsightFace
Loads model once and reuses across all images
"""

import cv2
import numpy as np
import os
import sys
from contextlib import contextmanager
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from typing import List, Dict, Optional
from pathlib import Path

import config


@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class FaceDetector:
    """Singleton face detector using InsightFace"""
    
    _instance: Optional['FaceDetector'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Initialize face analyzer with suppressed output
        with suppress_stdout_stderr():
            self.face_analyzer = FaceAnalysis(
                name=config.FACE_MODEL_NAME,
                providers=config.DEVICE.onnx_providers
            )
            
            # Prepare with GPU context
            ctx_id = config.DEVICE.device_id if config.DEVICE.device_id >= 0 else -1
            self.face_analyzer.prepare(
                ctx_id=ctx_id,
                det_size=config.FACE_DET_SIZE
            )
        
        self._initialized = True
    
    def detect_faces(self, image_path: str) -> List[Dict]:
        """
        Detect faces in a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of face detection results with embeddings
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.face_analyzer.get(rgb_frame)
        
        results = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            
            # Clamp to image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract face crop
            display_img = frame[y1:y2, x1:x2]
            
            # Align face if keypoints available
            if face.kps is not None:
                aligned = face_align.norm_crop(rgb_frame, face.kps)
                face_img = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
            else:
                face_img = display_img
            
            results.append({
                "face_img": face_img,
                "display_img": display_img,
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "embedding": face.embedding.astype("float32"),
                "det_score": float(face.det_score),
                "age": getattr(face, 'age', None),
                "gender": getattr(face, 'gender', None)
            })
        
        return results
    
    def process_batch(self, image_paths: List[str]) -> Dict[str, List[Dict]]:
        """
        Process multiple images efficiently
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary mapping image_path -> face detection results
        """
        results = {}
        for image_path in image_paths:
            try:
                results[image_path] = self.detect_faces(image_path)
            except Exception as e:
                # Silent failure, return empty list
                results[image_path] = []
        
        return results


# Global singleton instance getter
_face_detector_instance = None

def get_face_detector() -> FaceDetector:
    """Get or create the singleton face detector instance"""
    global _face_detector_instance
    if _face_detector_instance is None:
        _face_detector_instance = FaceDetector()
    return _face_detector_instance


def process_images_batch(image_paths: List[str]) -> Dict[str, List[Dict]]:
    """Convenience function to process a batch of images"""
    detector = get_face_detector()
    return detector.process_batch(image_paths)
