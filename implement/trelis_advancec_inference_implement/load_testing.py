import time
import asyncio
import statistics
from typing import List, Dict, Any
from .inference_utils import InferenceClient
from .load_balancer import LoadBalancer
from concurrent.futures import ThreadPoolExecutor
import logging
from .db.session import get_db
from .db.models import Pod, RunPodStatus

class LoadTester:
    def __init__(self, load_balancer: LoadBalancer, model: str):
        # Suppress OpenAI HTTP request logs
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self.load_balancer = load_balancer
        self.model = model
        self.results = []
        self.logger = logging.getLogger(__name__)

    def run_single_test(self) -> Dict[str, float]:
        """Run a single speed test and return metrics."""
        # Get a pod from the load balancer (which now only returns ready pods)
        pod_id = self.load_balancer.get_available_pod(self.model)
        if not pod_id:
            raise Exception("No pods available for testing")

        client = InferenceClient(pod_id)

        try:
            prompt = "Write a very detailed essay about the history of artificial intelligence."
            start_time = time.time()
            first_token_time = None

            # Stream the response
            stream = client.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.7,
                stream=True
            )

            full_response = ""
            for chunk in stream:
                if not first_token_time and chunk.choices[0].delta.content:
                    first_token_time = time.time()
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content

            end_time = time.time()

            # Calculate metrics
            token_count = len(full_response.split())  # Approximate
            total_time = end_time - start_time
            time_to_first = first_token_time - start_time if first_token_time else None

            return {
                "pod_id": pod_id,
                "time_to_first_token": time_to_first,
                "total_time": total_time,
                "tokens_per_second": token_count / total_time,
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"Test failed for pod {pod_id}: {e}")
            # Mark pod as potentially failed
            with get_db() as db:
                pod = db.query(Pod).filter(Pod.id == pod_id).first()
                if pod:
                    pod.status = RunPodStatus.ERROR
                    db.commit()
            raise

    async def run_load_test(self, concurrent_requests: int, duration: int) -> Dict[str, Any]:
        """Run a load test with multiple concurrent requests."""
        print(f"Starting load test with {concurrent_requests} concurrent requests for {duration}s...")

        # First verify we have at least one working pod
        pod_id = self.load_balancer.get_available_pod(self.model)
        if not pod_id:
            print("❌ No pods available for testing")
            return {"error": "No pods available"}

        self.results = []  # Reset results for new test
        self.start_time = time.time()
        end_time = self.start_time + duration

        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            while time.time() < end_time:
                # Submit new requests up to concurrent_requests
                futures = []
                for _ in range(concurrent_requests):
                    futures.append(
                        asyncio.get_event_loop().run_in_executor(
                            executor, self.run_single_test
                        )
                    )

                # Wait for all requests to complete
                results = await asyncio.gather(*futures)
                self.results.extend(results)

                # Record metrics silently
                for result in results:
                    self.load_balancer.record_metrics(
                        pod_id=result['pod_id'],
                        model=self.model,
                        tps=result['tokens_per_second'],
                        latency=result['time_to_first_token'] * 1000,
                        silent=True
                    )

                # Brief pause to prevent overwhelming
                await asyncio.sleep(0.1)

        print("Load test completed. Analyzing results...")
        return self.analyze_results()

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze test results and provide statistics."""
        if not self.results:
            return {"error": "No test results available"}

        tps_values = [r["tokens_per_second"] for r in self.results]
        ttft_values = [r["time_to_first_token"] for r in self.results if r["time_to_first_token"]]

        # Calculate actual test duration from start to now
        test_duration = time.time() - self.start_time

        return {
            "total_requests": len(self.results),
            "avg_tokens_per_second": statistics.mean(tps_values),
            "min_tokens_per_second": min(tps_values),
            "max_tokens_per_second": max(tps_values),
            "avg_time_to_first_token": statistics.mean(ttft_values),
            "p5_tokens_per_second": statistics.quantiles(tps_values, n=20)[0],  # 5th percentile - 95% are faster
            "test_duration": test_duration
        }