import asyncio
import os
import runpod
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from verifaix.llm_client.get_litellm_response import get_litellm_response
from verifaix.arangodb_helper.arango_client import (
    connect_to_arango_client,
    ensure_collection_exists,
    upsert_document
)


# Load environment variables (e.g., API keys)
load_dotenv('../.env')

# Configure loguru for logging
logger.add("runpod_pipeline.log", rotation="1 MB", level="DEBUG")

# Set RunPod API key
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    raise ValueError("RUNPOD_API_KEY is missing. Please set it in your environment variables.")
runpod.api_key = RUNPOD_API_KEY

# Constants
INACTIVITY_THRESHOLD_MINUTES = 5
POD_STATUS_CHECK_INTERVAL = 10  # seconds

# Retry logic with tenacity for starting the container
@retry(
    stop=stop_after_attempt(3),  # Retry up to 3 times
    wait=wait_fixed(5),  # Wait 5 seconds between retries
    retry=retry_if_exception_type(Exception),  # Retry on any exception
    reraise=True  # Reraise the exception if retries are exhausted
)
async def start_runpod_container():
    """Start a RunPod container and log the time taken, with retry logic."""
    model_config = {
        "name": "SGLang-Qwen/Qwen2.5-1.5B-Instruct",
        "image_name": "lmsysorg/sglang:latest",
        "docker_args": "python -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000",
        "resources": {"gpu": "NVIDIA RTX 4090"},  # Replace gpu_type with the correct key
        "volume_in_gb": 20,
        "ports": "8000/http",
        "container_disk_in_gb": 10,
        "volume_mount_path": "/root/.cache/huggingface",
        "env": {"HF_TOKEN": os.getenv("HF_TOKEN")}
    }

    start_time = datetime.now(timezone.utc)
    pod = runpod.create_pod(**model_config)
    logger.info(f"Started pod with ID: {pod.id}")
    end_time = datetime.now(timezone.utc)
    startup_duration = (end_time - start_time).total_seconds()
    logger.info(f"Container startup took {startup_duration:.2f} seconds.")
    return pod

async def wait_for_pod_to_run(pod):
    """Wait until the container is running or handle errors, and retrieve port information."""
    max_wait_time = timedelta(minutes=10)
    start_time = datetime.now(timezone.utc)
    try:
        while pod.status != "RUNNING":
            elapsed_time = datetime.now(timezone.utc) - start_time
            if elapsed_time > max_wait_time:
                logger.error("Pod did not reach 'RUNNING' state within the maximum wait time.")
                raise TimeoutError("Pod startup timed out.")
            if pod.status in ["ERROR", "FAILED"]:
                logger.error(f"Pod entered an error state: {pod.status}")
                raise RuntimeError(f"Pod failed with status: {pod.status}")
            await asyncio.sleep(POD_STATUS_CHECK_INTERVAL)
            pod = runpod.get_pod(pod.id)
            logger.debug(f"Pod status: {pod.status}")

        # Retrieve exposed port details once the pod is running
        pod_details = runpod.get_pod(pod.id)
        exposed_port = None
        if "ports" in pod_details:
            exposed_port = pod_details["ports"][0]["external"]  # Adjust indexing as needed
            logger.info(f"Exposed port: {exposed_port}")

        logger.info("Pod is now running!")
        return pod, exposed_port
    except Exception as e:
        logger.exception("Error while waiting for the pod to be ready.")
        raise

async def get_pod_public_url(pod, port=8000):
    """Retrieve the public URL for the RunPod container with configurable port."""
    try:
        pod_details = runpod.get_pod(pod.id)
        public_ip = pod_details.get('network', {}).get('public_ip')  # Adjust if necessary
        if public_ip:
            api_base = f"http://{public_ip}:{port}/v1"  # Use the passed or configured port
            logger.info(f"Pod public URL: {api_base}")
            return api_base
        else:
            raise ValueError("Unable to retrieve the public IP for the pod.")
    except Exception as e:
        logger.exception("Error retrieving the public URL for the pod.")
        raise

async def process_single_query(query, llm_params, activity_tracker, db):
    """Process a single query asynchronously using litellm."""
    try:
        logger.info(f"Processing query: {query}")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
        response = await get_litellm_response(messages, llm_params, db=db)

        activity_tracker["last_activity"] = datetime.now(timezone.utc)
        result = response['choices'][0]['message']['content']
        logger.debug(f"Response: {result}")
        return {"query": query, "response": result}
    except Exception as e:
        logger.exception(f"Error processing query: {query}")
        return {"query": query, "error": str(e)}

async def process_queries(queries, llm_params, activity_tracker, db):
    """Process multiple queries concurrently using asyncio.gather."""
    tasks = [process_single_query(query, llm_params, activity_tracker, db) for query in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

async def monitor_inactivity(pod, activity_tracker):
    """Monitor for container inactivity and shut down after threshold."""
    try:
        while True:
            await asyncio.sleep(60)  # Check every minute
            elapsed_time = datetime.now(timezone.utc) - activity_tracker["last_activity"]
            if elapsed_time > timedelta(minutes=INACTIVITY_THRESHOLD_MINUTES):
                logger.info(f"No activity for {INACTIVITY_THRESHOLD_MINUTES} minutes. Initiating shutdown...")
                await stop_runpod_container(pod.id)
                break
    except Exception as e:
        logger.exception("Error during inactivity monitoring.")

async def stop_runpod_container(pod_id):
    """Stop the RunPod container and log the time taken."""
    try:
        start_time = datetime.now(timezone.utc)
        runpod.stop_pod(pod_id)
        end_time = datetime.now(timezone.utc)
        shutdown_duration = (end_time - start_time).total_seconds()
        logger.info(f"Stopped pod with ID: {pod_id}. Container shutdown took {shutdown_duration:.2f} seconds.")
    except Exception as e:
        logger.exception("Failed to stop RunPod container.")

async def run_pipeline():
    """
    Main pipeline function to start a RunPod container, process queries using litellm,
    and monitor for inactivity before shutting down the container.
    """
    # Step 1: Initialize the activity tracker for monitoring inactivity
    activity_tracker = {"last_activity": datetime.now(timezone.utc)}

    # Step 2: Connect to the ArangoDB

    arango_config =  {
            "host": "http://localhost:8529",
            "username": "root",
            "password": "openSesame",
            "db_name": "verifaix",
            "collection_name": "test_documents"
    }
    db  = await asyncio.to_thread(connect_to_arango_client, arango_config)


    try:
        # Step 3: Start the RunPod container and log startup time
        pod = await start_runpod_container()

        # Step 4: Wait until the container is running or handle errors if it fails to start
        pod, exposed_port = await wait_for_pod_to_run(pod)

        # Step 5: Configure LLM parameters for the litellm calls
        # Retrieve the public URL for the pod
        api_base = await get_pod_public_url(pod, port=exposed_port)

        llm_params = {
            "model": "SGLang-Qwen/Qwen2.5-1.5B-Instruct",  # Adjust model identifier as needed
            "max_tokens": 1024,
            "temperature": 0.2,
            "json_mode": True,
            "api_base": api_base  # Use the dynamically retrieved API base
        }

        # Step 6: Start a background task to monitor container inactivity
        monitor_task = asyncio.create_task(monitor_inactivity(pod, activity_tracker))

        # Step 7: Define the list of queries to be processed
        queries = [
            "What is the capital of France?",
            "What is the square root of 16?",
            "Who wrote 'Pride and Prejudice'?"
        ]

        # Step 8: Process the queries asynchronously using litellm and update activity tracker
        results = await process_queries(queries, llm_params, activity_tracker, db)
        logger.info(f"Query results: {results}")

        # Step 9: Wait for the monitor task to complete (this occurs after inactivity threshold)
        await monitor_task

    except Exception as e:
        # Handle any errors that occur during the pipeline execution
        logger.exception("Pipeline encountered an error.")

    finally:
        # Step 10: Ensure the pod is stopped even if an error occurs
        if 'pod' in locals():
            await stop_runpod_container(pod.id)

if __name__ == "__main__":
    asyncio.run(run_pipeline())
