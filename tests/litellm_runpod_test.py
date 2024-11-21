import asyncio
from litellm import acompletion

async def debug_litellm():
    """
    Send a query using litellm's acompletion for debugging purposes.

    Args:
        api_base (str): Base URL for the API.
        model_name (str): The name of the model.
        query (str): The query text to send.
        max_tokens (int): Maximum number of tokens for the response.
        temperature (float): Sampling temperature for response generation.

    Returns:
        dict: The response from the model.
    """
    params ={
        'model': 'openai/Qwen/Qwen2.5-0.5B',
        'messages':  [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": 'What is the capital of France'}
        ],
        'max_tokens': 50,
        'temperature': 0.7,
        'response_format': None,
        'api_base': 'https://jqy6nr0djajtlf-8000.proxy.runpod.net/v1'
    }

    response = await acompletion(**params)
    print(response.choices[0].message.content)

    # try:
    #     # Perform an asynchronous completion request
    #     response = await acompletion(
    #         messages=[
    #             {"role": "system", "content": "You are a helpful assistant."},
    #             {"role": "user", "content": query}
    #         ],
    #         model=model_name,
    #         api_base=api_base,
    #         max_tokens=max_tokens,
    #         temperature=temperature,
    #     )

    #     # Print and return the response
    #     print("Model Response:", response)
    #     return response

    # except Exception as e:
    #     print("Error:", str(e))
    #     return None

if __name__ == "__main__":
    # Configuration for the model and API
    # Works
    # api_base = "https://jqy6nr0djajtlf-8000.proxy.runpod.net/v1"
    # model_name = "openai/Qwen/Qwen2.5-0.5B"
    # query = "What is the capital of France?"

    # Run the async function
    asyncio.run(debug_litellm())
