import runpod
import os
import time
import re
from typing import Optional, Dict, Any, List
from .model_configs import MODEL_CONFIGS, DEFAULT_POD_SETTINGS
from .db.session import get_db
from .db.models import Pod, RunPodStatus
import logging
from .inference_utils import InferenceClient
from datetime import datetime, timedelta, UTC

# Constants
GPU_PREFERENCES = ["NVIDIA A40", "NVIDIA RTX A4500", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090"]

def extract_model_size(model_slug: str) -> int:
    """Extract model size in billions from model slug and calculate required volume."""
    match = re.search(r'(\d+)b', model_slug, re.IGNORECASE)
    if match:
        size_in_b = int(match.group(1))
        return (size_in_b * 2) + 10
    return 30  # default if no size found

def get_model_config(model_name: str, override_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get configuration for a model, with optional overrides."""
    # Start with default pod settings
    config = DEFAULT_POD_SETTINGS.copy()

    # Find config by model name
    if model_name in MODEL_CONFIGS:
        config.update(MODEL_CONFIGS[model_name])
    else:
        # Generate default config based on model name
        config.update({
            "name": model_name,  # Keep original case
            "docker_args": f"--model {model_name} --port 8000",
            "volume_in_gb": extract_model_size(model_name)
        })

    # Apply any override configs
    if override_config:
        config.update(override_config)

    return config

def start_pod(model_name: str, config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Start a pod for the specified model with optional configuration overrides."""
    pod_config = get_model_config(model_name, config_override)

    # Use original model name for pod name
    pod_config["name"] = model_name

    # Get number of GPUs and tensor parallel size
    gpus_per_instance = pod_config.get('gpus_per_instance', 1)
    tensor_parallel = gpus_per_instance
    print(f"GPUs per instance: {gpus_per_instance}")
    print(f"Tensor parallel size: {tensor_parallel}")

    # Handle docker args and tensor parallelism
    docker_args = pod_config.get("docker_args", "")
    # Remove any existing --tensor-parallel-size flag
    docker_args = re.sub(r'--tensor-parallel-size \d+', '', docker_args).strip()
    # Only add tensor parallel flag if greater than 1
    if tensor_parallel > 1:
        docker_args = f"{docker_args} --tensor-parallel-size {tensor_parallel}"
    pod_config["docker_args"] = docker_args

    # Remove scaling-specific and model-specific config items that RunPod API doesn't expect
    pod_creation_config = {k: v for k, v in pod_config.items() if k not in [
        'min_instances',
        'max_instances',
        'target_tps',
        'min_tps',
        'max_tps',
        'gpus_per_instance',
        'scale_cooldown',
        'metrics_window',
        'monitor_interval',
        'tensor_parallel',
        'capabilities'  # Added this to the exclusion list
    ]}

    # Set number of GPUs for the pod
    pod_creation_config["gpu_count"] = gpus_per_instance

    # Always set docker args with full model slug
    if "docker_args" not in pod_creation_config:
        pod_creation_config["docker_args"] = f"--model {model_name} --port 8000"

    last_error = None
    for gpu_type in GPU_PREFERENCES:
        try:
            pod_creation_config["gpu_type_id"] = gpu_type
            pod = runpod.create_pod(**pod_creation_config)
            print(f"\nCreated Pod with {gpu_type}:")
            print(pod)

            # Add pod to database
            with get_db() as db:
                pod_status = pod.get('desiredStatus', 'UNKNOWN')
                try:
                    status = RunPodStatus[pod_status]
                except KeyError:
                    status = RunPodStatus.UNKNOWN

                db_pod = db.query(Pod).filter(Pod.id == pod['id']).first()
                if db_pod:
                    # Update existing pod
                    db_pod.model = model_name
                    db_pod.runpod_status = status
                else:
                    # Create new pod
                    new_pod = Pod(
                        id=pod['id'],
                        model=model_name,
                        runpod_status=status,
                        is_ready=False
                    )
                    db.add(new_pod)
                db.commit()

            return pod

        except runpod.error.QueryError as err:
            print(f"Failed to create pod with {gpu_type}: {err}")
            last_error = err
            continue
        except ValueError as err:
            print(f"Invalid GPU configuration for {gpu_type}: {err}")
            last_error = err
            continue

    raise Exception(f"No preferred GPUs available. Last error: {last_error}")

def get_pod_status(pod_id: str) -> Dict[str, Any]:
    """
    Get the current status of a pod.

    Args:
        pod_id: The ID of the pod to check

    Returns:
        Dict containing pod status information
    """
    try:
        status = runpod.get_pod(pod_id)
        return status
    except runpod.error.QueryError as err:
        print(f"Error getting pod status: {err}")
        raise

def stop_pod(pod_id: str) -> None:
    """Stop and delete a pod, updating the database."""
    try:
        try:
            runpod.terminate_pod(pod_id)
        except runpod.error.QueryError as e:
            print(f"Warning: RunPod API error when terminating pod {pod_id}: {e}")

        with get_db() as db:
            pod = db.query(Pod).filter(Pod.id == pod_id).first()
            if pod:
                pod.runpod_status = RunPodStatus.TERMINATED
                pod.is_ready = False
                db.commit()
                print(f"Updated database status for pod {pod_id}")
            else:
                print(f"Warning: Pod {pod_id} not found in database")

    except Exception as e:
        print(f"Error in stop_pod: {e}")
        raise

def list_pods() -> List[Dict[str, Any]]:
    """
    Get a list of all running pods.

    Returns:
        List of pod information dictionaries
    """
    try:
        pods = runpod.get_pods()
        # logging.info("RunPod API Response:")
        # logging.info(pods)
        return pods
    except runpod.error.QueryError as err:
        print(f"Error listing pods: {err}")
        return []

async def sync_db() -> int:
    """Synchronize database with RunPod state and verify pod readiness."""
    try:
        runpod_pods = list_pods()
        pod_states = {pod['id']: pod for pod in runpod_pods}

        with get_db() as db:
            # First, clean up terminated pods older than 5 minutes
            cleanup_time = datetime.now(UTC) - timedelta(minutes=5)
            deleted = db.query(Pod).filter(
                Pod.runpod_status == RunPodStatus.TERMINATED,
                Pod.created_at < cleanup_time
            ).delete()

            if deleted:
                logging.info(f"Cleaned up {deleted} terminated pods")

            # Rest of the sync logic...
            db_pods = db.query(Pod).all()

            # Update existing pods
            for pod in db_pods:
                runpod_state = pod_states.get(pod.id)

                if not runpod_state:
                    # Pod no longer exists in RunPod
                    pod.runpod_status = RunPodStatus.TERMINATED
                    pod.is_ready = False
                    continue

                # Get RunPod status
                runpod_status = runpod_state.get('desiredStatus', 'UNKNOWN')
                try:
                    pod.runpod_status = RunPodStatus[runpod_status]
                except KeyError:
                    pod.runpod_status = RunPodStatus.UNKNOWN

                # Only check inference if RunPod shows it as running
                if runpod_status == "RUNNING":
                    client = InferenceClient(pod.id)
                    is_ready = await client.wait_for_service(max_retries=1, retry_delay=5)
                    pod.is_ready = is_ready and client.test_inference(model=pod.model)
                else:
                    pod.is_ready = False

            # Add new pods from RunPod
            for pod_id, pod_state in pod_states.items():
                if not db.query(Pod).filter(Pod.id == pod_id).first():
                    new_pod = Pod(
                        id=pod_id,
                        model=pod_state.get('name', '').split('/')[-1],
                        runpod_status=RunPodStatus[pod_state.get('desiredStatus', 'UNKNOWN')],
                        is_ready=False
                    )
                    db.add(new_pod)

            db.commit()
        return 0

    except Exception as e:
        logging.error(f"Error syncing database: {e}")
        return 1

# Example usage
if __name__ == "__main__":
    try:
        # Start a pod
        pod = start_pod()

        # Check its status
        status = get_pod_status(pod["id"])
        print(f"Pod status: {status}")

        # Stop it (uncomment to test)
        # stop_pod(pod["id"])

    except Exception as e:
        print(f"Error: {e}")