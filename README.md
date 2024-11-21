
# RunPod LLM Pipeline

This repository provides a robust pipeline script to manage the lifecycle of a machine learning container on RunPod, perform LLM (Large Language Model) inference, and store the results in an ArangoDB database. It is designed for efficient model deployment, request handling, and integration with caching mechanisms.

---

## Features

- **Model Lifecycle Management**: Automatically start, monitor, and stop RunPod containers for inference tasks.
- **Database Integration**: Connects to an ArangoDB instance for storing LLM requests and responses with support for caching.
- **Request Deduplication**: Generates unique hashes for LLM requests to prevent redundant computations.
- **Error Handling**: Incorporates retry mechanisms, logging, and robust cleanup steps for resource management.
- **Customizable Models**: Easily configurable to use different models, including `SGLang-Qwen` and other Hugging Face-hosted models.

---

## File Overview

### `runpod_llm_pipeline.py`
The main script that orchestrates the pipeline, including:
1. **Model Initialization**: Loads configurations for the model and RunPod container.
2. **RunPod Management**: Starts or reuses a container, checks readiness, and handles cleanup.
3. **Request Handling**: Prepares LLM requests with unique hashes.
4. **LLM Inference**: Sends requests to the model container and processes responses.
5. **Database Storage**: Upserts results into ArangoDB for persistence and traceability.

### `config.py`
Holds configuration details for:
- RunPod API and Hugging Face tokens.
- Model-specific settings such as GPU preferences, container disk size, and environmental variables.
- Pipeline configurations like ArangoDB connection details.

### `runpod_ops.py`
Utility functions for managing RunPod containers:
- Starting, stopping, and monitoring container status.
- Checking API readiness for deployed containers.

### `llm_utils.py`
Helper functions for:
- Adding hashes to requests to enable deduplication.
- Sending requests to the LLM and merging responses with the original requests.

---

## Prerequisites

1. **Python Environment**: Ensure Python 3.8+ is installed.
2. **Dependencies**:
   - Install required packages using `pip install -r requirements.txt`.
3. **ArangoDB**: Set up an instance of ArangoDB and ensure it is accessible.
4. **RunPod Account**: Obtain an API key from RunPod to manage containers.
5. **Hugging Face Token**: Required for downloading models during container initialization.

---

## Usage

1. **Configure Environment Variables**:
   - Create a `.env` file with the following variables:
     ```
     HF_TOKEN=<your_huggingface_token>
     RUNPOD_API_KEY=<your_runpod_api_key>
     ```

2. **Run the Pipeline**:
   Execute the script to start the pipeline:
   ```
   python runpod_llm_pipeline.py
   ```

3. **Monitor Logs**:
   Logs are generated at each step to ensure transparency and debugging.

---

## Key Steps in the Pipeline

1. **Start RunPod Container**:
   - Automatically starts a container with the specified model and preferred GPU.

2. **Prepare Requests**:
   - Each request is hashed for efficient deduplication.

3. **Send Requests to LLM**:
   - The requests are processed by the deployed model using the specified API parameters.

4. **Store Results in ArangoDB**:
   - Responses are upserted into a specified collection in ArangoDB for persistence.

5. **Cleanup**:
   - Ensures that containers are stopped and resources are released, even in case of errors.

---

## Customization

- **Model Configuration**:
  Modify `MODEL_CONFIGS` in `config.py` to add or update model-specific settings.

- **Database Settings**:
  Update the `pipeline_config['arango_config']` to customize ArangoDB settings like host, database name, and collection.

- **RunPod Parameters**:
  Adjust `DEFAULT_POD_SETTINGS` to optimize container resources or specify preferred GPUs.

---

## Error Handling

The pipeline includes robust mechanisms to handle errors:
- **Retry Logic**: Reattempts API requests and container operations using exponential backoff.
- **Graceful Cleanup**: Ensures that containers are stopped properly in the event of errors.

---

## Future Enhancements

- Support for multiple model types in parallel.
- Advanced analytics for token usage and request trends.
- Integration with monitoring dashboards for RunPod container status.

---

This pipeline is a scalable and modular solution for LLM deployment and inference. Contributions and suggestions are welcome!
