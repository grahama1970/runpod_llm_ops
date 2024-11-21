# llm_utils.py
import asyncio
from loguru import logger
from verifaix.arangodb_helper.arango_client import generate_safe_key
from verifaix.llm_client.get_litellm_response import get_litellm_response

def add_hashes_to_requests(requests):
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
