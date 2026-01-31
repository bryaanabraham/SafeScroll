"""Models module for Safe Scroll"""

from .face_detector import get_face_detector, process_images_batch
from .llm_client import LLMClient, get_llm_client

__all__ = [
    'get_face_detector',
    'process_images_batch',
    'LLMClient',
    'get_llm_client'
]
