import os
from dotenv import load_dotenv

load_dotenv('../.env')

HF_TOKEN = os.getenv("HF_TOKEN")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

class ModelCapabilities:
    TEXT = "text"
    TEXT_AND_IMAGE = "text_and_image"

MODEL_CONFIGS = {
    # 32B QWEN model
    "SGLang-Qwen2.5-Coder-32B-Instruct": {
        "name": "SGLang-Qwen2.5-Coder-32B-Instruct",
        "capabilities": [ModelCapabilities.TEXT],
        "docker_args": "python -m sglang.launch_server --model-path Qwen/Qwen2.5-Coder-32B-Instruct --host 0.0.0.0 --port 8000",
        "volume_in_gb": 65,
        "min_instances": 1,
        "max_instances": 3,
        "min_tps": 20.0,
        "max_tps": 40.0,
        "gpus_per_instance": 1,
        "scale_cooldown": 240,
        "metrics_window": 60,
        "monitor_interval": 15,
        "internal_port": 8000  # Internal port the model is served on
    },
    # 1.5B QWEN model
    "SGLang-Qwen/Qwen2.5-1.5B-Instruct": {
        "name": "SGLang-Qwen2.5-Coder-1.5B",
        "capabilities": [ModelCapabilities.TEXT],
        "docker_args": "python -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000",
        "volume_in_gb": 5,
        "min_instances": 1,
        "max_instances": 5,
        "min_tps": 40.0,
        "max_tps": 80.0,
        "gpus_per_instance": 1,
        "scale_cooldown": 240,
        "metrics_window": 60,
        "monitor_interval": 15,
        "internal_port": 8000  # Internal port the model is served on
    }
}

DEFAULT_POD_SETTINGS = {
    "image_name": "lmsysorg/sglang:latest",
    "cloud_type": "SECURE",
    "ports": "8000/http",
    "container_disk_in_gb": 10,
    "volume_mount_path": "/root/.cache/huggingface",
    "env": {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_TOKEN": HF_TOKEN
    },
    "scale_cooldown": 180,
    "metrics_window": 60,
    "monitor_interval": 15
}


# put everything that is needed for the pipeline to run here, eventually we will have a config file for the pipeline
pipeline_config = {
    "arango_config": {
                "host": "http://localhost:8529",
                "username": "root",
                "password": "openSesame",
                "db_name": "verifaix",
                "collection_name": "test_documents"
    },
    "arango_config": {
            "host": "http://localhost:8529",
            "username": "root",
            "password": "openSesame",
            "db_name": "verifaix",
            "collection_name": "test_documents",
            "cache_collection_name": "litellm_cache" # store litellm responses
        },

        "llm_config": {
            "model": "openai/Qwen/Qwen2.5-0.5B",
            "max_tokens": 50,
            "temperature": 0.7,
            "api_base": "api_base"
        }
}
