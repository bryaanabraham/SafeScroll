"""
Configuration module for Safe Scroll
Handles device detection, logging, and global settings
"""

# CRITICAL: Set environment variables BEFORE any imports
import os
import sys

# Suppress ONNX Runtime verbose logging
os.environ['ORT_LOGGING_LEVEL'] = '4'  # Fatal errors only
os.environ['ONNXRUNTIME_LOG_SEVERITY_LEVEL'] = '4'

# Suppress TensorFlow/general warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
import warnings
import torch

# Suppress all Python warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Suppress library-specific warnings
logging.getLogger('insightface').setLevel(logging.ERROR)
logging.getLogger('onnxruntime').setLevel(logging.ERROR)
logging.getLogger('ollama').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)

# Configure root logger
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


class DeviceConfig:
    """Manages device detection and GPU configuration"""
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0
        
        # Prefer GPU 1 (GTX 1650) if available, otherwise GPU 0
        if self.device_count > 1:
            self.device_id = 1  # NVIDIA GTX 1650
            self.device_name = torch.cuda.get_device_name(1)
        elif self.device_count == 1:
            self.device_id = 0
            self.device_name = torch.cuda.get_device_name(0)
        else:
            self.device_id = -1  # CPU
            self.device_name = "CPU"
        
        # ONNX Runtime providers
        if self.cuda_available and self.device_id >= 0:
            self.onnx_providers = [
                ('CUDAExecutionProvider', {
                    'device_id': self.device_id,
                    'gpu_mem_limit': 3 * 1024 * 1024 * 1024,  # 3GB limit for GTX 1650
                }),
                'CPUExecutionProvider'
            ]
        else:
            self.onnx_providers = ['CPUExecutionProvider']
    
    def get_torch_device(self):
        """Get PyTorch device"""
        if self.cuda_available and self.device_id >= 0:
            return torch.device(f'cuda:{self.device_id}')
        return torch.device('cpu')
    
    def print_info(self):
        """Print device configuration info"""
        print(f"🔧 Device: {self.device_name}")
        if self.cuda_available:
            print(f"   GPU Memory: {torch.cuda.get_device_properties(self.device_id).total_memory / 1e9:.2f} GB")


# Global device configuration
DEVICE = DeviceConfig()

# Model configurations
FACE_MODEL_NAME = 'antelopev2'
FACE_DET_SIZE = (640, 640)

# LLM Model names
LLM_VISION_MODEL = 'gemma3:4b'  # For vision tasks (lightweight)
LLM_TEXT_MODEL = 'gemma3:4b'    # For text synthesis (consistent model)

# Download directory
DOWNLOAD_DIR = "downloads"

# Performance settings
ENABLE_GPU = DEVICE.cuda_available
MAX_BATCH_SIZE = 4  # Conservative for 4GB VRAM

# Image quality settings for LLM processing
# Downscale images to reduce LLM processing time
IMAGE_MAX_DIMENSION_LLM = 1024  # Max width/height for time/summary tasks
IMAGE_MAX_DIMENSION_OCR = 2048   # Higher quality for OCR accuracy
IMAGE_QUALITY_JPEG = 85          # JPEG quality for resized images

# API optimization settings
ENABLE_GEOLOCATION = False  # Set to True to enable geolocation (adds 1 extra API call per post)
