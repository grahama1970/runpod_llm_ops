import asyncio
import os
from typing import Any, Dict, List
import uuid
import httpx
import runpod
from datetime import datetime, timezone
from dotenv import load_dotenv
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from yaspin import yaspin
from yaspin.spinners import Spinners

from verifaix.arangodb_helper.arango_client import connect_to_arango_client, generate_safe_key
from verifaix.llm_client.get_litellm_response import get_litellm_response
from verifaix.pymupdf_extractor.utils.spinner import start_spinner, stop_spinner

# Load environment variables (e.g., API keys)
load_dotenv('../.env')

# Configure loguru for logging
logger.add("runpod_pipeline.log", rotation="1 MB", level="DEBUG")

# Set RunPod API key
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    raise ValueError("RUNPOD_API_KEY is missing. Please set it in your environment variables.")
runpod.api_key = RUNPOD_API_KEY


####
# RunPod Centric Functions
####
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
    spinner = start_spinner("Waiting for pod to initialize...")

    try:
        while True:
            elapsed_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            if elapsed_time > max_wait_time:
                raise TimeoutError("Pod startup timed out.")

            pod = runpod.get_pod(pod["id"])
            logger.info(f"Pod details: {pod}")

            if pod.get("desiredStatus") == "RUNNING": # running could be starting up
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
        stop_spinner(spinner)


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


async def send_query_via_curl(api_base, model_name, query, max_tokens=50, temperature=0.7):
    """
    Send a query to the model using a curl-equivalent POST request.

    Args:
        api_base (str): The base URL for the API.
        model_name (str): The name of the model to query.
        query (str): The query to send.
        max_tokens (int): The maximum number of tokens in the response.
        temperature (float): The temperature for the response generation.

    Returns:
        dict: The response from the model.
    """
    try:
        # Define the payload in OpenAI-like format
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Set the endpoint and headers
        endpoint = f"{api_base}/chat/completions"
        headers = {"Content-Type": "application/json"}

        # Send the POST request
        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, json=payload, headers=headers, timeout=10)

        # Check if the response was successful
        if response.status_code == 200:
            #logger.info(f"Curl response: {response.json()}")
            result = response.json()["choices"][0]["message"]["content"]
            logger.info(f"Curl result: {result}")
            return response.json()  # Parse the JSON response
        else:
            raise RuntimeError(
                f"Model query failed: {response.status_code}, {response.text}"
            )

    except Exception as e:
        raise RuntimeError(f"Error sending query via curl: {str(e)}")

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

####
# LLM Centric Functions
####

def add_hashes_to_requests(requests: List[List[Dict[str, str]]]) -> List[Dict[str, Any]]:
    """
    Add hashes to each conversation in the list of requests.

    Args:
        requests (list): A list of conversations (each a list of messages).

    Returns:
        list: A list of dictionaries, each with '_hash' and 'messages' keys.
    """
    def compute_hash(messages):
        # Generate a hash using the content of the messages.
        raw_key = " ".join(msg.get("content", "") for msg in messages)
        return generate_safe_key(raw_key)

    results = [
        {
            "_hash": compute_hash(messages),
            "messages": messages
        }
        for messages in requests
    ]
    return results

async def make_request(llm_params, request, db=None):
    """
    Make a request to the LLM API with given parameters and messages.
    Passes the hash directly into the LLM request.

    Args:
        llm_params (dict): LLM configuration parameters.
        request (dict): A dictionary containing '_hash' and 'messages'.
        db (object, optional): Database connection for additional context.

    Returns:
        dict: The response from the LLM API, including the hash.
    """
    try:
        # Extract hash and messages from the request
        request_hash = request["_hash"]
        messages = request["messages"]

        # Pass the hash into the LLM request
        response = await get_litellm_response(messages, llm_params, request_id=request_hash, db=db)
        return response
    except Exception as e:
        logger.exception(f"Failed to process request with hash {request['_hash']}: {e}")
        return {"_hidden_params": {"request_id": request["_hash"]}, "error": str(e)}

def merge_responses_with_requests(requests, responses):
    """
    Merge the response objects back into the original requests list using hashes.

    Args:
        requests (list): The original list of requests with '_hash' fields.
        responses (list): The list of response objects containing '_hidden_params' with hashes.

    Returns:
        list: The enriched list of requests, each with its corresponding response.
    """
    # Map request_id (hash) as key to response as value
    response_map = {
        response["_hidden_params"]["request_id"]: response
        for response in responses
    }

    # Merge the responses back into the original requests
    for request in requests:
        request_hash = request["_hash"]
        if request_hash in response_map:
            request["response"] = response_map[request_hash]

    return requests

# NOTE: Thurs Morning: You are ensuring that the hash connects
# back to the stored arangodb response. Then try again to make sure it works
# Then consider how you might make this script more modular and reusable.


####
# Main function
###
async def main():
    """
    Main function to start the container, wait for it to be ready,
    send a query, and stop the container.
    """

    model_config = {
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
        "volume_in_gb": 20,
        "ports": "8000/http",
        "container_disk_in_gb": 10,
        "volume_mount_path": "/root/.cache/huggingface",
        # "env": {"HF_TOKEN": os.getenv("HF_TOKEN"), "HF_HUB_ENABLE_HF_TRANSFER": "1"},
        "env": {"HF_TOKEN": os.getenv("HF_TOKEN")},
        "preferred_gpu_names": ["RTX 4090", "RTX 4080", "RTX 6000 Ada", "RTX A6000"],
    }

    try:
        # Check for existing pods
        logger.info("Checking for existing pods...")
        existing_pods = runpod.get_pods()
        matching_pod = next(
            (pod for pod in existing_pods if pod["name"] == model_config["name"]),
            None
        )

        if matching_pod:
            if matching_pod["desiredStatus"] == "RUNNING":
                logger.info(f"Using existing running pod: {matching_pod['id']}")
                pod = matching_pod
                api_base = f"https://{pod['id']}-8000.proxy.runpod.net/v1"

            elif matching_pod["desiredStatus"] in ["EXITED", "STOPPED"]:
                logger.info(f"Cleaning up existing exited pod: {matching_pod['id']}")

                # Terminate the stopped or exited pod to clean up resources
                runpod.terminate_pod(matching_pod["id"])
                logger.info(f"Terminated pod: {matching_pod['id']}")

                # Recreate a new pod using the model configuration
                logger.info("Recreating a new pod...")
                pod = await start_runpod_container(model_config)
                pod, api_base = await wait_for_pod_to_run(pod)
            else:
                logger.warning(f"Pod found with unexpected status: {matching_pod['desiredStatus']}. Starting a new pod...")
                pod = await start_runpod_container(model_config)
                pod, api_base = await wait_for_pod_to_run(pod)
        else:
            logger.info("No existing pod found. Starting a new one...")
            pod = await start_runpod_container(model_config)
            pod, api_base = await wait_for_pod_to_run(pod)

        # Test inference with a simple query
        # await send_query_via_curl(api_base, "Qwen/Qwen2.5-0.5B", "What is the capital of France?")

        # Test with litellm
        # messages = [
        #         {"role": "system", "content": "You are a helpful assistant."},
        #         {"role": "user", "content": "What is the capital of France?"}
        # ]

        # Testing async gather
        requests = [
            [
                {"role": "system", "content": "You are a knowledgeable historian who provides concise responses."},
                {"role": "user", "content": "Tell me about ancient Rome"},
                {"role": "assistant", "content": "Ancient Rome was a civilization centered in Italy."},
                {"role": "user", "content": "What were their major achievements?"}
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the most common color of an apple?"}
            ]
        ]

        # Step 1: Add UUIDs to each request
        requests_with_uuids = add_hashes_to_requests(requests)

        # Step 2: Connect to ArangoDB
        arango_config = {
            "host": "http://localhost:8529",
            "username": "root",
            "password": "openSesame",
            "db_name": "verifaix",
            "collection_name": "test_documents",
            "cache_collection_name": "litellm_cache" # store litellm responses
        }
        db = await asyncio.to_thread(connect_to_arango_client, arango_config)

        llm_params = {
            "model": "openai/Qwen/Qwen2.5-0.5B",
            "max_tokens": 50,
            "temperature": 0.7,
            "api_base": api_base
        }


        # Step 3: Create asyncio tasks for concurrent processing
        tasks = [make_request(llm_params, request) for request in requests_with_uuids]
        responses = await asyncio.gather(*tasks)

        # Step 4: Merge responses back into the original requests
        connected_requests = merge_responses_with_requests(requests_with_uuids, responses)

        # Log or process responses as needed
        for request in connected_requests:
            logger.info(f"Request ID: {request['_uuid']}")
            logger.info(f"Messages Object: {request['messages']}")
            result = request['response'].choices[0].message.content
            logger.info(f"Clean response: {result}\n\n")

        # Tests to Stop Container
        logger.info("Waiting 3 seconds before stopping the container...")
        await asyncio.sleep(3)

        logger.info(f"Stopping the runpod container: {pod['id']}")
        await stop_runpod_container(pod["id"])

        # Stop the container only if it was newly created
        if not matching_pod:
            await asyncio.sleep(10)
            await stop_runpod_container(pod["id"])

    except Exception as e:
        logger.exception("An error occurred in the main pipeline.")
    finally:
        if "pod" in locals() and not matching_pod:
            await stop_runpod_container(pod["id"])


if __name__ == "__main__":
    asyncio.run(main())
