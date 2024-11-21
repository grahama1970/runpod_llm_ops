# config.py
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv('../.env')

HF_TOKEN = os.getenv("HF_TOKEN")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

if not RUNPOD_API_KEY:
    raise ValueError("RUNPOD_API_KEY is missing. Please set it in your environment variables.")

class ModelCapabilities:
    TEXT = "text"
    TEXT_AND_IMAGE = "text_and_image"

# Define model configurations
MODEL_CONFIGS = {
    # Corrected 0.5B QWEN model
    "SGLang-Qwen/Qwen2.5-1.5B-Instruct": {
        "name": "SGLang-Qwen/Qwen2.5-1.5B-Instruct",
        "image_name": "lmsysorg/sglang:latest",
        "docker_args": (
            "python3 -m sglang.launch_server "
            "--model-path Qwen/Qwen2.5-0.5B "
            "--mem-fraction-static 0.95 "
            "--host 0.0.0.0 "
            "--port 8000"
        ),
        "cloud_type": "SECURE",
        "volume_in_gb": 5,
        "ports": "8000/http",
        "container_disk_in_gb": 10,
        "volume_mount_path": "/root/.cache/huggingface",
        "env": {"HF_TOKEN": os.getenv("HF_TOKEN"), "HF_HUB_ENABLE_HF_TRANSFER": "1"},
        "preferred_gpu_names": ["RTX 4090", "RTX 4080", "RTX 6000 Ada", "RTX A6000"],
    },

    # 32B QWEN Instruct
    "SGLang-Qwen/Qwen2.5-Coder-32B-Instruct": {
        "name": "SGLang-Qwen/Qwen2.5-Coder-32B-Instruct",
        "image_name": "lmsysorg/sglang:latest",
        "docker_args": (
            "python3 -m sglang.launch_server "
            "--model-path Qwen/Qwen2.5-Coder-32B-Instruct "
            "--mem-fraction-static 0.95 "
            "--host 0.0.0.0 "
            "--port 8000"
        ),
        "cloud_type": "SECURE",
        "volume_in_gb": 100,
        "ports": "8000/http",
        "container_disk_in_gb": 50,
        "volume_mount_path": "/root/.cache/huggingface",
        "env": {"HF_TOKEN": os.getenv("HF_TOKEN"), "HF_HUB_ENABLE_HF_TRANSFER": "1"},
        "preferred_gpu_names": ["H100 PCIe", "H100 NVL", "H100 SXM", "RTX A6000"],
    },

}



# Default settings for pods
DEFAULT_POD_SETTINGS = {
    "image_name": "lmsysorg/sglang:latest",
    "cloud_type": "SECURE",
    "ports": "8000/http",
    "container_disk_in_gb": 10,
    "volume_in_gb": 100,
    "volume_mount_path": "/root/.cache/huggingface",
    "env": {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_TOKEN": HF_TOKEN
    },
    "scale_cooldown": 180,
    "metrics_window": 60,
    "monitor_interval": 15
}

# Configuration for the pipeline
pipeline_config = {
    "arango_config": {
        "host": "http://localhost:8529",
        "username": "root",
        "password": "openSesame",
        "db_name": "verifaix",
        "collection_name": "test_documents",
        "cache_collection_name": "litellm_cache",  # Store litellm responses
        "truncate_cache": True  # Truncate the cache collection before starting
    },
    "llm_config": {
        "model": "openai/Qwen/Qwen2.5-0.5B",
        "max_tokens": 50,
        "temperature": 0.7,
        "api_base": "api_base"  # This will be set dynamically
    }
}
