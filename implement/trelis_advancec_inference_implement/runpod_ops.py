import runpod
import asyncio
from config import RUNPOD_API_KEY, MODEL_CONFIGS, DEFAULT_POD_SETTINGS

runpod.api_key = RUNPOD_API_KEY

async def start_pod(model_name):
    model_config = MODEL_CONFIGS[model_name]
    pod_config = {**DEFAULT_POD_SETTINGS, **model_config}

    pod = runpod.create_pod(
        name=model_config['name'],
        image_name=DEFAULT_POD_SETTINGS['image_name'],
        gpu_type=model_config.get('gpu_type', 'NVIDIA RTX A4000'),  # Default to A4000 if not specified
        docker_args=model_config['docker_args'],
        volume_in_gb=model_config['volume_in_gb'],
        env=DEFAULT_POD_SETTINGS['env'],
        ports=DEFAULT_POD_SETTINGS['ports'],
        container_disk_in_gb=DEFAULT_POD_SETTINGS['container_disk_in_gb'],
        volume_mount_path=DEFAULT_POD_SETTINGS['volume_mount_path'],
        cloud_type=DEFAULT_POD_SETTINGS['cloud_type']
    )
    print(f"Started pod with ID: {pod.id}")
    return pod

async def wait_for_pod_to_run(pod):
    while pod.status != "RUNNING":
        await asyncio.sleep(10)
        pod = runpod.get_pod(pod.id)
        print(f"Pod status: {pod.status}")
    print("Pod is now running!")
    return pod

async def stop_pod(pod_id):
    runpod.stop_pod(pod_id)
    print(f"Stopped pod with ID: {pod_id}")

async def start_runpod_container(model_name):
    pod = await start_pod(model_name)
    pod = await wait_for_pod_to_run(pod)
    return pod.id

async def stop_runpod_container(pod_id):
    await stop_pod(pod_id)

# your_pipeline_script.py

import asyncio
from verifaix.runpod.utils.runpod_ops import start_runpod_container, stop_runpod_container

async def run_pipeline():
    # Start the container
    pod_id = await start_runpod_container("SGLang-Qwen2.5-Coder-32B-Instruct")
    print(f"Container started with ID: {pod_id}")

    # Do some work here...
    await asyncio.sleep(60)  # Simulating some work

    # Stop the container
    await stop_runpod_container(pod_id)
    print("Container stopped")

if __name__ == "__main__":
    asyncio.run(run_pipeline())