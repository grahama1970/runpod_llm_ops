# runpod_ops.py
import asyncio
from datetime import datetime, timezone
import httpx
import runpod
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from loguru import logger
from yaspin import yaspin
from yaspin.spinners import Spinners

from config import RUNPOD_API_KEY

# Set RunPod API key
runpod.api_key = RUNPOD_API_KEY

async def start_runpod_container(model_config):
    """
    Start a RunPod container with retry logic, prioritizing a list of preferred GPUs.
    Logs the time taken and handles pod recreation if needed.
    """
    available_gpus = runpod.get_gpus()  # Retrieve all available GPUs
    if not available_gpus:
        raise RuntimeError("No available GPUs found for pod creation.")

    # Prioritize GPUs based on the preferred list and ensure the order matches `preferred_gpu_names`
    preferred_gpus = sorted(
        [gpu for gpu in available_gpus if gpu["displayName"] in model_config["preferred_gpu_names"]],
        key=lambda gpu: model_config["preferred_gpu_names"].index(gpu["displayName"])
    )
    fallback_gpus = [gpu for gpu in available_gpus if gpu not in preferred_gpus]

    if not preferred_gpus and not fallback_gpus:
        raise RuntimeError("No suitable GPUs available for pod creation.")

    gpus_to_try = preferred_gpus + fallback_gpus

    for gpu in gpus_to_try:
        try:
            # Copy the model_config and remove non-relevant keys
            pod_config = {key: value for key, value in model_config.items() if key != "preferred_gpu_names"}
            pod_config["gpu_type_id"] = gpu["id"]  # Use the GPU type ID

            logger.info(f"Attempting to start pod with GPU: {gpu['displayName']} (ID: {gpu['id']})")
            start_time = datetime.now(timezone.utc)

            # Filter pod_config to only include valid keys for create_pod
            # NOTE: This is a workaround to avoid errors due to invalid keys in the pod_config
            valid_keys = [
                'name', 'image_name', 'gpu_type_id', 'cloud_type', 'support_public_ip',
                'start_ssh', 'data_center_id', 'country_code', 'gpu_count', 'volume_in_gb',
                'container_disk_in_gb', 'min_vcpu_count', 'min_memory_in_gb', 'docker_args',
                'ports', 'volume_mount_path', 'env', 'template_id', 'network_volume_id',
                'allowed_cuda_versions', 'min_download', 'min_upload'
            ]
            filtered_pod_config = {k: v for k, v in pod_config.items() if k in valid_keys}
            pod_config = filtered_pod_config

            pod = runpod.create_pod(**pod_config)

            logger.info(f"Successfully started pod with GPU: {gpu['displayName']} (ID: {gpu['id']}). Pod ID: {pod['id']}")
            end_time = datetime.now(timezone.utc)
            startup_duration = (end_time - start_time).total_seconds()
            logger.info(f"Container startup took {startup_duration:.2f} seconds.")
            return pod

        except Exception as e:
            logger.warning(f"Failed to start pod with GPU {gpu['displayName']} (ID: {gpu['id']}): {e}")

    logger.error("Failed to start a pod with any preferred or fallback GPU types.")
    raise RuntimeError("No GPUs could be used to start the RunPod container.")

async def wait_for_pod_to_run(pod):
    """
    Wait until the container is fully ready by checking its status and API readiness.
    Logs if the pod was recreated due to cleanup or error handling.
    """
    max_wait_time = 900  # 15 minutes
    start_time = datetime.now(timezone.utc)
    spinner = yaspin(Spinners.dots, text="Waiting for pod to initialize...")
    spinner.start()

    try:
        while True:
            elapsed_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            if elapsed_time > max_wait_time:
                raise TimeoutError("Pod startup timed out.")

            pod = runpod.get_pod(pod["id"])
            logger.info(f"Pod details: {pod}")

            if pod.get("desiredStatus") == "RUNNING":
                logger.info("Pod has reached 'RUNNING' status.")
                break

            await asyncio.sleep(10)

        # Construct the API base using the pod ID
        api_base = f"https://{pod['id']}-8000.proxy.runpod.net/v1"
        logger.info(f"API base URL: {api_base}")

        # Perform readiness check
        if await check_api_readiness(api_base):
            return pod, api_base
        else:
            logger.warning("Pod required recreation due to readiness failure.")
            raise RuntimeError("API readiness check failed.")

    except Exception as e:
        logger.exception("Error while waiting for the pod to be ready.")
        raise

    finally:
        spinner.stop()

@retry(
    stop=stop_after_attempt(30),  # Retry up to 30 times
    wait=wait_fixed(10),          # Wait 10 seconds between retries
    retry=retry_if_exception_type(Exception),  # Retry only on exceptions
)
async def check_api_readiness(api_base):
    """
    Check if the API is ready by pinging its endpoints.
    Retries until a response with status code 200 is received or retries are exhausted.

    Args:
        api_base (str): The base URL of the API.

    Returns:
        bool: True if the API is ready, raises an Exception otherwise.
    """
    endpoints = ["/models", "/health"]
    async with httpx.AsyncClient() as client:
        for endpoint in endpoints:
            response = await client.get(f"{api_base}{endpoint}", timeout=10)
            if response.status_code == 200:
                logger.info(f"API readiness confirmed at endpoint: {endpoint}.")
                return True

    raise RuntimeError("API readiness check failed.")

@retry(
    stop=stop_after_attempt(30),  # Retry up to 30 times
    wait=wait_fixed(5),           # Wait 5 seconds between retries
    retry=retry_if_exception_type(Exception),  # Retry on any exceptions
)
async def stop_runpod_container(pod_id, terminate_flag=False):
    """
    Stop the RunPod container and confirm it is fully stopped.
    If stopping fails and terminate_flag is True, attempt to terminate the container.

    Args:
        pod_id (str): The ID of the pod to stop.
        terminate_flag (bool): If True, attempt to terminate the pod if stopping fails.

    Raises:
        RuntimeError: If the pod fails to stop or terminate successfully.
    """
    logger.info(f"Initiating shutdown for pod with ID: {pod_id}")
    start_time = datetime.now(timezone.utc)

    # Attempt to stop the pod
    try:
        response = runpod.stop_pod(pod_id)
        logger.debug(f"RunPod stop response: {response}")

        # Ensure the response matches the expected pod ID
        if response.get("id") != pod_id:
            logger.warning(f"Unexpected pod ID in stop response: {response.get('id')}")

        # Check if the pod has stopped
        pod_details = runpod.get_pod(pod_id)
        current_status = pod_details.get("desiredStatus")
        logger.debug(f"Pod status is '{current_status}'.")

        if current_status not in ["EXITED", "STOPPED"]:
            raise RuntimeError(f"Pod with ID {pod_id} is still in state '{current_status}'. Retrying...")

        end_time = datetime.now(timezone.utc)
        shutdown_duration = (end_time - start_time).total_seconds()
        logger.info(f"Stopped pod with ID: {pod_id}. Container shutdown took {shutdown_duration:.2f} seconds.")
        return  # Exit function if the pod is successfully stopped

    except Exception as stop_exception:
        logger.warning(f"Failed to stop pod {pod_id}.")

        # Check if termination is enabled
        if terminate_flag:
            logger.info(f"Terminate flag is set. Attempting to terminate pod {pod_id}...")
            try:
                terminate_response = runpod.terminate_pod(pod_id)
                logger.info(f"RunPod terminate response: {terminate_response}")

                # Verify termination status
                terminated_pod = runpod.get_pod(pod_id)
                if terminated_pod.get("desiredStatus") == "TERMINATED":
                    logger.info(f"Pod with ID {pod_id} has been successfully terminated.")
                    return
                else:
                    raise RuntimeError(f"Pod with ID {pod_id} did not terminate successfully.")

            except Exception as terminate_exception:
                logger.exception(f"Failed to terminate pod {pod_id}. Termination error: {terminate_exception}")
                raise RuntimeError(f"Failed to stop or terminate pod {pod_id}: {stop_exception}, {terminate_exception}")
        else:
            logger.error(f"Terminate flag is not set. Pod {pod_id} remains in an unresolved state.")
            raise RuntimeError(f"Failed to stop pod {pod_id}, and termination was not attempted.")
