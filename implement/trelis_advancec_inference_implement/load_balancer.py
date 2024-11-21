import random
from typing import Optional, Dict, Any, Generator
from datetime import datetime
from .db.session import get_db
from .db.models import Pod, RunPodStatus, PodMetric
import logging
import time
from openai.types.chat import ChatCompletion
import tiktoken

class LoadBalancer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_available_pod(self, model: str) -> Optional[str]:
        """Find an available pod for the model directly from database."""
        self.logger.debug(f"Looking for pod for model: {model}")

        with get_db() as db:
            pods = db.query(Pod).filter(
                Pod.model == model,
                Pod.runpod_status == RunPodStatus.RUNNING,
                Pod.is_ready == True
            ).all()
            available_pods = [pod.id for pod in pods]
            self.logger.debug(f"Found ready pods: {available_pods}")

            if available_pods:
                chosen_pod = random.choice(available_pods)
                self.logger.debug(f"Selected pod: {chosen_pod}")
                return chosen_pod

            self.logger.warning(f"No available pods found for model: {model}")
            return None

    def handle_completion(
        self,
        client: Any,
        messages: list,
        model: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Handle a completion request and record metrics.
        Returns the response and metrics.
        """
        # Clean messages by removing None values
        cleaned_messages = []
        for msg in messages:
            if isinstance(msg["content"], list):
                # Handle multimodal content
                cleaned_content = []
                for content in msg["content"]:
                    cleaned_item = {
                        k: v for k, v in content.items()
                        if v is not None and (k != "image_url" or v.get("url"))
                    }
                    if cleaned_item:  # Only add if there are non-None values
                        cleaned_content.append(cleaned_item)
                cleaned_messages.append({
                    "role": msg["role"],
                    "content": cleaned_content
                })
            else:
                # Handle text-only content
                if msg["content"] is not None:
                    cleaned_messages.append(msg)

        start_time = time.time()
        first_token_time = None
        response_text = ""

        try:
            if stream:
                # Handle streaming case
                stream_response = client.client.chat.completions.create(
                    model=model,
                    messages=cleaned_messages,  # Use cleaned messages
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                )

                for chunk in stream_response:
                    if not first_token_time and chunk.choices[0].delta.content:
                        first_token_time = time.time() - start_time  # Store the delta instead of absolute time
                    if chunk.choices[0].delta.content:
                        response_text += chunk.choices[0].delta.content

                end_time = time.time()

            else:
                # Handle non-streaming case
                response = client.client.chat.completions.create(
                    model=model,
                    messages=cleaned_messages,  # Use cleaned messages
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                response_text = response.choices[0].message.content
                end_time = time.time()
                first_token_time = end_time - start_time  # For non-streaming, use total time

            # Calculate metrics
            token_count = self.count_tokens(response_text)
            total_time = end_time - start_time
            tps = token_count / total_time if token_count > 0 else None
            latency = first_token_time * 1000 if first_token_time else None  # Convert to ms

            # Record metrics
            self.record_metrics(
                pod_id=client.pod_id,
                model=model,
                tps=tps,
                latency=latency,
                silent=True
            )

            return {
                "response_text": response_text,
                "metrics": {
                    "token_count": token_count,
                    "total_time": total_time,
                    "tps": tps,
                    "latency": latency,
                    "time_to_first_token": first_token_time  # Renamed for clarity
                }
            }

        except Exception as e:
            self.logger.error(f"Error in completion: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception as e:
            self.logger.error(f"Error counting tokens: {e}")
            return 0

    def record_metrics(self, pod_id: str, model: str, tps: Optional[float], latency: Optional[float], silent: bool = False) -> None:
        """Record performance metrics for a pod."""
        # Format metrics for display, handling None values
        tps_str = f"{tps:.2f}" if tps is not None else "N/A"
        latency_str = f"{latency:.2f}ms" if latency is not None else "N/A"

        if not silent:
            print(f"\nRecording metrics - Pod: {pod_id}, TPS: {tps_str}, Latency: {latency_str}")

        try:
            # Normalize model name
            normalized_model = model.split('/')[-1]

            with get_db() as db:
                metric = PodMetric(
                    pod_id=pod_id,
                    model=normalized_model,
                    tokens_per_second=tps,
                    latency=latency,
                    timestamp=datetime.now()
                )
                db.add(metric)
                db.commit()

                if not silent:
                    print("✅ Metrics recorded successfully")

        except Exception as e:
            if not silent:
                print(f"❌ Failed to record metrics: {e}")
            raise