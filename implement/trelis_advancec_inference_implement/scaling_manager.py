import time
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from .pod_utils import start_pod, stop_pod, sync_db
from .model_configs import MODEL_CONFIGS, DEFAULT_POD_SETTINGS
from .db.models import Pod, PodMetric, ScalingEvent, RunPodStatus, ScalingAction
from .db.session import get_db
from sqlalchemy import func
import logging
import asyncio

class ScalingManager:
    def __init__(self, logger=None, model=None):
        self.running = False
        self.logger = logger or logging.getLogger(__name__)
        self.model = model
        self.last_scale_time = {}

        # Get timing configs from model config or defaults
        if model and model in MODEL_CONFIGS:
            config = MODEL_CONFIGS[model]
            self.scale_cooldown = config.get('scale_cooldown', DEFAULT_POD_SETTINGS['scale_cooldown'])
            self.metrics_window = config.get('metrics_window', DEFAULT_POD_SETTINGS['metrics_window'])
            self.monitor_interval = config.get('monitor_interval', DEFAULT_POD_SETTINGS['monitor_interval'])
        else:
            self.scale_cooldown = DEFAULT_POD_SETTINGS['scale_cooldown']
            self.metrics_window = DEFAULT_POD_SETTINGS['metrics_window']
            self.monitor_interval = DEFAULT_POD_SETTINGS['monitor_interval']

        self.logger.info(f"Initialized with cooldown={self.scale_cooldown}s, "
                        f"metrics_window={self.metrics_window}s, "
                        f"monitor_interval={self.monitor_interval}s")

    def get_pods_ready(self, model: str) -> List[str]:
        """Get list of ready pods for a model directly from database."""
        with get_db() as db:
            pods = db.query(Pod).filter(
                Pod.model == model,
                Pod.runpod_status == RunPodStatus.RUNNING,
                Pod.is_ready == True
            ).all()
            return [pod.id for pod in pods]

    def should_scale(self, model: str) -> Optional[str]:
        """Determine if scaling is needed."""
        if model not in MODEL_CONFIGS:
            self.logger.warning(f"No configuration found for model: {model}")
            return None

        config = MODEL_CONFIGS[model]
        metrics = self.get_pod_metrics(model)

        with get_db() as db:
            # Count both ready and initializing pods
            total_pods = db.query(Pod).filter(
                Pod.model == model,
                Pod.runpod_status == RunPodStatus.RUNNING
            ).count()

            ready_pods = db.query(Pod).filter(
                Pod.model == model,
                Pod.runpod_status == RunPodStatus.RUNNING,
                Pod.is_ready == True
            ).count()

        self.logger.info(f"Total pods: {total_pods} (Ready: {ready_pods})")

        # First, enforce max instances limit using total pods
        if total_pods >= config['max_instances']:
            self.logger.warning(f"Current pods ({total_pods}) at or exceed maximum allowed ({config['max_instances']})")
            return 'down' if ready_pods > config['min_instances'] else None

        # Don't scale up if we're already at max instances including initializing pods
        if metrics['avg_tps'] == 0:
            if ready_pods > config['min_instances']:
                return 'down'
            elif total_pods < config['min_instances']:  # Use total_pods here
                return 'up'
            return None

        # If we have traffic, scale based on TPS, but respect total pod count
        if metrics['avg_tps'] < config['min_tps'] and total_pods < config['max_instances']:
            return 'up'
        elif metrics['avg_tps'] > config['max_tps'] and ready_pods > config['min_instances']:
            return 'down'
        return None

    def scale_up(self, model: str):
        """Add a new pod for the model."""
        try:
            self.logger.info(f"Initiating scale up for {model}")
            pod = start_pod(model)
            self.logger.info(f"Started new pod: {pod['id']}")

            with get_db() as db:
                # Record scaling event only
                event = ScalingEvent(
                    model=model,
                    action=ScalingAction.UP,
                    reason=f"TPS below minimum threshold",
                    pods_before=db.query(Pod).filter(
                        Pod.model == model,
                        Pod.runpod_status == RunPodStatus.RUNNING,
                        Pod.is_ready == True
                    ).count() - 1,  # Subtract 1 since we just added a pod
                    pods_after=db.query(Pod).filter(
                        Pod.model == model,
                        Pod.runpod_status == RunPodStatus.RUNNING,
                        Pod.is_ready == True
                    ).count()
                )
                db.add(event)
                db.commit()

            asyncio.run(sync_db())
            self.last_scale_time[model] = time.time()
            self.logger.info(f"Scale up complete for {model}")

        except Exception as e:
            self.logger.error(f"Error scaling up {model}: {e}", exc_info=True)

    def scale_down(self, model: str):
        """Remove least efficient pod for the model."""
        try:
            self.logger.info(f"Initiating scale down for {model}")
            with get_db() as db:
                # Get all pods that are read for this model
                pods_ready = db.query(Pod).filter(
                    Pod.model == model,
                    Pod.runpod_status == RunPodStatus.RUNNING,
                    Pod.is_ready == True
                ).all()

                if not pods_ready:
                    self.logger.warning("No pods ready found to scale down")
                    return

                # Get pod metrics for the last 5 minutes
                cutoff = datetime.now() - timedelta(minutes=5)
                pod_metrics = {}

                for pod in pods_ready:
                    metrics = db.query(func.avg(PodMetric.tokens_per_second)).filter(
                        PodMetric.pod_id == pod.id,
                        PodMetric.timestamp > cutoff
                    ).scalar()
                    pod_metrics[pod.id] = metrics or 0.0

                # Find pod with lowest TPS
                pod_to_remove = min(pods_ready, key=lambda p: pod_metrics[p.id])

                if pod_to_remove:
                    self.logger.info(f"Selected pod {pod_to_remove.id} for removal (TPS: {pod_metrics[pod_to_remove.id]:.2f})")

                    # Stop the pod in RunPod
                    try:
                        stop_pod(pod_to_remove.id)
                        self.logger.info(f"Successfully stopped pod {pod_to_remove.id}")
                    except Exception as e:
                        self.logger.error(f"Failed to stop pod {pod_to_remove.id}: {e}")

                    # Update database
                    pod_to_remove.is_active = False
                    pod_to_remove.runpod_status = RunPodStatus.TERMINATED

                    # Record scaling event
                    event = ScalingEvent(
                        model=model,
                        action=ScalingAction.DOWN,
                        reason="TPS above maximum threshold",
                        pods_before=len(pods_ready),
                        pods_after=len(pods_ready) - 1
                    )
                    db.add(event)
                    db.commit()

            asyncio.run(sync_db())
            self.last_scale_time[model] = time.time()
            self.logger.info(f"Scale down complete for {model}")

        except Exception as e:
            self.logger.error(f"Error scaling down {model}: {e}", exc_info=True)

    def ensure_minimum_pods(self, model: str) -> None:
        """Ensure minimum number of pods are running for the model."""
        if model not in MODEL_CONFIGS:
            self.logger.warning(f"No configuration found for model: {model}")
            return

        # Add cooldown check here
        if (model in self.last_scale_time and
            time.time() - self.last_scale_time[model] < self.scale_cooldown):
            time_left = self.scale_cooldown - (time.time() - self.last_scale_time[model])
            self.logger.debug(f"Cooldown active for {model}. {time_left:.1f} seconds remaining")
            return

        config = MODEL_CONFIGS[model]
        min_instances = config.get('min_instances', 1)

        with get_db() as db:
            # Count ALL running pods, not just ready ones
            total_running_pods = db.query(Pod).filter(
                Pod.model == model,
                Pod.runpod_status == RunPodStatus.RUNNING
            ).count()

            # Get ready pods count for logging
            ready_pods = db.query(Pod).filter(
                Pod.model == model,
                Pod.runpod_status == RunPodStatus.RUNNING,
                Pod.is_ready == True
            ).count()

        self.logger.debug(f"Current pods - Total running: {total_running_pods}, Ready: {ready_pods}, Minimum required: {min_instances}")

        if total_running_pods < min_instances:
            pods_to_create = min_instances - total_running_pods
            self.logger.info(f"Creating {pods_to_create} initial pod(s) for {model}")

            for _ in range(pods_to_create):
                try:
                    self.scale_up(model)
                except Exception as e:
                    self.logger.error(f"Failed to create initial pod: {e}", exc_info=True)

    def monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Sync database with RunPod state first
                try:
                    sync_result = asyncio.run(sync_db())
                    if sync_result != 0:
                        self.logger.error("Database sync failed")
                    else:
                        self.logger.debug("Database synchronized with RunPod state")

                    # Debug logging for pod states
                    with get_db() as db:
                        running_pods = db.query(Pod).filter(
                            Pod.model == self.model,
                            Pod.runpod_status == RunPodStatus.RUNNING
                        ).all()
                        self.logger.debug(f"Current pods for {self.model}:")
                        for pod in running_pods:
                            self.logger.debug(f"  - Pod {pod.id}: Status={pod.runpod_status}, Ready={pod.is_ready}")

                except Exception as e:
                    self.logger.error(f"Error syncing database with RunPod: {e}", exc_info=True)
                    time.sleep(30)
                    continue

                # Clean up terminated pods before other operations
                self.cleanup_terminated_pods()

                # First, ensure minimum pods for our managed model
                if self.model:
                    self.ensure_minimum_pods(self.model)

                # Get list of models with any running pods (not just ready ones)
                with get_db() as db:
                    models = db.query(func.distinct(Pod.model)).filter(
                        Pod.runpod_status == RunPodStatus.RUNNING
                    ).all()
                    models = [m[0] for m in models]

                self.logger.debug(f"Found {len(models)} models to monitor: {models}")

                if not models:
                    if self.model:
                        self.logger.debug(f"No running pods found for {self.model}")
                    time.sleep(self.monitor_interval)
                    continue

                # Log status for each model at monitoring interval
                for model in models:
                    self.logger.debug(f"\n=== Checking scaling for {model} ===")

                    # Get current state and config
                    config = MODEL_CONFIGS.get(model)
                    if not config:
                        continue

                    with get_db() as db:
                        pods_ready = db.query(Pod).filter(
                            Pod.model == model,
                            Pod.runpod_status == RunPodStatus.RUNNING,
                            Pod.is_ready == True
                        ).count()

                    metrics = self.get_pod_metrics(model)
                    self.logger.info(
                        f"Model: {model} | "
                        f"Pods Ready: {pods_ready} | "
                        f"Avg TPS: {metrics['avg_tps']:.2f} | "
                        f"Total TPS: {metrics['total_tps']:.2f} | "
                        f"Avg Latency: {metrics['avg_latency']:.2f}ms"
                    )

                    self.logger.debug(f"Current pods ready: {pods_ready}")

                    # Check if we're over max instances - scale down immediately if we are
                    if pods_ready > config['max_instances']:
                        self.logger.warning(f"Current pods ({pods_ready}) exceed maximum allowed ({config['max_instances']})")
                        pods_to_remove = pods_ready - config['max_instances']
                        self.logger.info(f"Removing {pods_to_remove} excess pods")
                        for _ in range(pods_to_remove):
                            self.scale_down(model)
                        continue

                    # Normal scaling checks with cooldown
                    # Check scaling with cooldown
                    if (model in self.last_scale_time and
                        time.time() - self.last_scale_time[model] < self.scale_cooldown):
                        time_left = self.scale_cooldown - (time.time() - self.last_scale_time[model])
                        self.logger.debug(f"Cooldown active for {model}. {time_left:.1f} seconds remaining")
                        continue

                    scale_action = self.should_scale(model)
                    if scale_action == 'up':
                        self.logger.debug(f"Decision: Scale UP")
                        self.scale_up(model)
                    elif scale_action == 'down':
                        self.logger.debug(f"Decision: Scale DOWN")
                        self.scale_down(model)
                    else:
                        self.logger.debug(f"Decision: No scaling needed")

                    self.logger.debug("=== Scaling check complete ===\n")

            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}", exc_info=True)

            self.logger.debug(f"Sleeping for {self.monitor_interval} seconds...")
            time.sleep(self.monitor_interval)

    def start(self):
        """Start the scaling manager."""
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop)
        self.monitor_thread.start()

    def stop(self):
        """Stop the scaling manager."""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()

    def get_pod_metrics(self, model: str) -> Dict[str, float]:
        """Get aggregated metrics for all pods running this model."""
        try:
            with get_db() as db:
                # Get metrics from the last window period
                cutoff = datetime.now() - timedelta(seconds=self.metrics_window)
                metrics = db.query(PodMetric).filter(
                    PodMetric.model == model,
                    PodMetric.timestamp > cutoff
                ).all()

                if not metrics:
                    self.logger.debug(f"No metrics found for {model} in the last {self.metrics_window} seconds")
                    return {
                        'avg_tps': 0,
                        'avg_latency': 0,
                        'total_tps': 0
                    }

                # Calculate averages only for available metrics
                total_tps = sum(m.tokens_per_second for m in metrics if m.tokens_per_second is not None)
                count_tps = sum(1 for m in metrics if m.tokens_per_second is not None)

                # Only calculate latency if the column exists and has data
                total_latency = sum(m.latency for m in metrics if hasattr(m, 'latency') and m.latency is not None)
                count_latency = sum(1 for m in metrics if hasattr(m, 'latency') and m.latency is not None)

                return {
                    'avg_tps': total_tps / count_tps if count_tps > 0 else 0,
                    'avg_latency': total_latency / count_latency if count_latency > 0 else 0,
                    'total_tps': total_tps
                }

        except Exception as e:
            self.logger.error(f"Error getting pod metrics: {e}")
            return {
                'avg_tps': 0,
                'avg_latency': 0,
                'total_tps': 0
            }

    def cleanup_terminated_pods(self):
        """Clean up old terminated pods from the database."""
        try:
            with get_db() as db:
                # Find pods that have been terminated for more than 1 hour
                cutoff = datetime.now() - timedelta(hours=1)
                terminated_pods = db.query(Pod).filter(
                    Pod.runpod_status == RunPodStatus.TERMINATED,
                    Pod.is_ready == False
                ).all()

                for pod in terminated_pods:
                    # Delete associated metrics
                    db.query(PodMetric).filter(
                        PodMetric.pod_id == pod.id
                    ).delete()

                    # Delete the pod record
                    db.delete(pod)

                    self.logger.info(f"Cleaned up terminated pod {pod.id} and its metrics")

                db.commit()

        except Exception as e:
            self.logger.error(f"Error cleaning up terminated pods: {e}", exc_info=True)