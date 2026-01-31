"""
Image utilities for efficient image loading and validation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import config


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from disk
    
    Args:
        image_path: Path to image file
        
    Returns:
        Numpy array of image or None if failed
    """
    try:
        if not Path(image_path).exists():
            return None
        
        img = cv2.imread(image_path)
        return img
    except Exception:
        return None


def resize_image_for_llm(
    image_path: str,
    max_dimension: int,
    output_path: Optional[str] = None,
    quality: int = 85
) -> Optional[str]:
    """
    Resize image to reduce LLM processing load
    
    Args:
        image_path: Path to original image
        max_dimension: Maximum width or height
        output_path: Optional path for resized image (default: same dir with _resized suffix)
        quality: JPEG quality (1-100)
        
    Returns:
        Path to resized image or None if failed
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        height, width = img.shape[:2]
        
        # Check if resize needed
        if height <= max_dimension and width <= max_dimension:
            return image_path  # Already small enough
        
        # Calculate new dimensions
        if height > width:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        else:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        
        # Resize using high-quality interpolation
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Determine output path
        if output_path is None:
            path = Path(image_path)
            output_path = str(path.parent / f"{path.stem}_resized{path.suffix}")
        
        # Save with specified quality
        cv2.imwrite(output_path, resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        return output_path
    except Exception:
        return None


def create_llm_optimized_images(
    image_paths: List[str],
    max_dimension: int = config.IMAGE_MAX_DIMENSION_LLM,
    quality: int = config.IMAGE_QUALITY_JPEG
) -> List[str]:
    """
    Create downscaled versions of images for LLM processing
    
    Args:
        image_paths: List of original image paths
        max_dimension: Maximum dimension for resized images
        quality: JPEG quality
        
    Returns:
        List of paths to optimized images
    """
    optimized_paths = []
    for path in image_paths:
        resized_path = resize_image_for_llm(path, max_dimension, quality=quality)
        if resized_path:
            optimized_paths.append(resized_path)
    
    return optimized_paths


def validate_image_paths(image_paths: List[str]) -> List[str]:
    """
    Filter out invalid image paths
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        List of valid image paths only
    """
    valid_paths = []
    for path in image_paths:
        if Path(path).exists() and Path(path).is_file():
            valid_paths.append(path)
    
    return valid_paths
