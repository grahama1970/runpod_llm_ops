# pipeline.py
import asyncio
from datetime import datetime, timezone
import uuid
import jsonpickle
from loguru import logger
from verifaix.arangodb_helper.arango_client import connect_to_arango_client, upsert_document
import runpod

from config import pipeline_config, MODEL_CONFIGS, DEFAULT_POD_SETTINGS
from verifaix.runpod.utils.arango_utils import truncate_cache_collection
from verifaix.runpod.utils.runpod_ops import (
    start_runpod_container,
    wait_for_pod_to_run,
    stop_runpod_container
)
from verifaix.runpod.utils.llm_utils import (
    add_hashes_to_requests,
    make_request,
    merge_responses_with_requests
)

async def main():
    """
    Main function to start the container, wait for it to be ready,
    send a query, and stop the container.
    """

    # Step 1: Initialize Model Configuration
    logger.info("Step 1: Initialize Model Configuration")
    model_name = "SGLang-Qwen/Qwen2.5-1.5B-Instruct"
    model_config = MODEL_CONFIGS[model_name]

    # Merge default pod settings
    for key, value in DEFAULT_POD_SETTINGS.items():
        model_config.setdefault(key, value)

    try:
        # Step 2: Start or Reuse RunPod Container
        logger.info("Step 2: Start or Reuse RunPod Container")
        # Check for existing pods
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
                runpod.terminate_pod(matching_pod["id"])
                logger.info(f"Terminated pod: {matching_pod['id']}")
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

        # Step 3: Prepare Requests
        logger.info("Step 3: Prepare Requests")
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
        requests_with_hashes = add_hashes_to_requests(requests)

        # Step 4: Connect to Database
        logger.info("Step 4: Connect to Database")
        arango_config = pipeline_config['arango_config']
        db = await asyncio.to_thread(connect_to_arango_client, arango_config)

        # Truncate the litellm_cache collection (if set to True)
        truncate_cache = arango_config.get('truncate_cache', False)
        if truncate_cache:
            await truncate_cache_collection(pipeline_config['arango_config'], db)

        # Step 5: Prepare LLM Parameters
        logger.info("Step 5: Prepare LLM Parameters")
        llm_params = pipeline_config['llm_config']
        llm_params['api_base'] = api_base  # Ensure api_base is set

        # Step 6: Make Requests to LLM
        logger.info("Step 6: Make Requests to LLM")
        tasks = [make_request(llm_params, request, db=db) for request in requests_with_hashes]
        responses = await asyncio.gather(*tasks)

        # Step 7: Merge Responses with Requests
        logger.info("Step 7: Merge Responses with Requests")
        connected_requests = merge_responses_with_requests(requests_with_hashes, responses)

        # Step 8: Process and Store Results
        logger.info("Step 8: Process and Store Results")
        collection_name = arango_config.get('collection_name', 'default_collection')

        for request in connected_requests:
            try:
                completion = request.get("response").get("choices")[0].get("message").get("content")
                response_object = (
                    jsonpickle.encode(request.get("response"))
                    if request.get("response") else None
                )
                # Log the request for debugging
                logger.info(f"Request ID: {request['_hash']}")
                logger.info(f"Messages Object: {request['messages']}")
                logger.info(f"Completion: {completion}")
                # Prepare the document
                document = {
                    "_key": str(uuid.uuid4()),
                    "request_id": request["_hash"],  # Store the hash for traceability
                    "messages": request["messages"],
                    "completion": completion, # pulled out of response object
                    "response_object": response_object,
                    "_last_updated": datetime.now(timezone.utc).isoformat()
                }

                # Upsert the document (ArangoDB will generate the `_key`)
                await asyncio.to_thread(upsert_document, db, collection_name, document)
                logger.info(f"Upserted document with request_id: {request['_hash']} into collection '{collection_name}'")

            except Exception as e:
                logger.error(f"Failed to process request with request_id {request['_hash']}: {e}")

    # Step 9: Clean Up
    finally:
        logger.info("Step 9: Clean Up")
        if 'pod' in locals():
            try:
                logger.info(f"Stopping the runpod container: {pod['id']}")
                await stop_runpod_container(pod["id"])
            except Exception as cleanup_error:
                logger.error(f"Failed to stop the container: {cleanup_error}")

if __name__ == "__main__":
    asyncio.run(main())
