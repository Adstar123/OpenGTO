"""Performance monitoring utilities for tracking model and system performance."""

import time
import psutil
import torch
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from contextlib import contextmanager
import logging


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    operation: str
    start_time: float
    end_time: float
    duration: float
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: Optional[float] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'operation': self.operation,
            'duration_seconds': self.duration,
            'cpu_percent': self.cpu_percent,
            'memory_mb': self.memory_mb,
            'gpu_memory_mb': self.gpu_memory_mb,
            'timestamp': datetime.fromtimestamp(self.start_time).isoformat(),
            **self.additional_metrics
        }


class PerformanceMonitor:
    """Monitor performance of various operations."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize performance monitor.
        
        Args:
            logger: Logger for performance messages
        """
        self.logger = logger or logging.getLogger(__name__)
        self.metrics_history: List[PerformanceMetrics] = []
        self.process = psutil.Process()
    
    @contextmanager
    def measure(self, operation: str, **kwargs):
        """Context manager to measure performance of an operation.
        
        Args:
            operation: Name of the operation
            **kwargs: Additional metrics to record
            
        Yields:
            Dict that can be updated with additional metrics
        """
        # Start measurements
        start_time = time.time()
        start_cpu = self.process.cpu_percent()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # GPU memory if available
        start_gpu_memory = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        
        # Additional metrics dict
        additional_metrics = kwargs.copy()
        
        try:
            yield additional_metrics
        finally:
            # End measurements
            end_time = time.time()
            duration = end_time - start_time
            
            # CPU and memory
            end_cpu = self.process.cpu_percent()
            end_memory = self.process.memory_info().rss / 1024 / 1024
            
            cpu_percent = (start_cpu + end_cpu) / 2
            memory_mb = end_memory
            
            # GPU memory
            gpu_memory_mb = None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_memory_mb = end_gpu_memory
            
            # Create metrics
            metrics = PerformanceMetrics(
                operation=operation,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                gpu_memory_mb=gpu_memory_mb,
                additional_metrics=additional_metrics
            )
            
            self.metrics_history.append(metrics)
            
            # Log if duration is significant
            if duration > 0.1:  # More than 100ms
                self.logger.info(
                    f"{operation} completed in {duration:.2f}s "
                    f"(CPU: {cpu_percent:.1f}%, Memory: {memory_mb:.1f}MB)"
                )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all recorded metrics.
        
        Returns:
            Summary dictionary
        """
        if not self.metrics_history:
            return {'message': 'No metrics recorded'}
        
        # Group by operation
        operation_metrics = {}
        for metric in self.metrics_history:
            if metric.operation not in operation_metrics:
                operation_metrics[metric.operation] = []
            operation_metrics[metric.operation].append(metric)
        
        # Calculate statistics
        summary = {
            'total_operations': len(self.metrics_history),
            'total_duration': sum(m.duration for m in self.metrics_history),
            'operations': {}
        }
        
        for operation, metrics in operation_metrics.items():
            durations = [m.duration for m in metrics]
            cpu_percents = [m.cpu_percent for m in metrics]
            memory_mbs = [m.memory_mb for m in metrics]
            
            summary['operations'][operation] = {
                'count': len(metrics),
                'total_duration': sum(durations),
                'avg_duration': np.mean(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'avg_cpu_percent': np.mean(cpu_percents),
                'avg_memory_mb': np.mean(memory_mbs),
            }
            
            # Add GPU stats if available
            gpu_memories = [m.gpu_memory_mb for m in metrics if m.gpu_memory_mb is not None]
            if gpu_memories:
                summary['operations'][operation]['avg_gpu_memory_mb'] = np.mean(gpu_memories)
        
        return summary
    
    def reset(self):
        """Reset metrics history."""
        self.metrics_history.clear()


class ModelPerformanceTracker:
    """Track model-specific performance metrics."""
    
    def __init__(self, model: torch.nn.Module):
        """Initialize model performance tracker.
        
        Args:
            model: PyTorch model to track
        """
        self.model = model
        self.inference_times: List[float] = []
        self.batch_sizes: List[int] = []
    
    def measure_inference_speed(
        self,
        input_tensor: torch.Tensor,
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """Measure model inference speed.
        
        Args:
            input_tensor: Sample input tensor
            num_iterations: Number of iterations to measure
            warmup: Number of warmup iterations
            
        Returns:
            Performance statistics
        """
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Warmup
        self.model.eval()
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(input_tensor)
        
        # Synchronize if using GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.time()
                _ = self.model(input_tensor)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.time()
                times.append(end - start)
        
        times = np.array(times)
        
        return {
            'mean_inference_time': np.mean(times),
            'std_inference_time': np.std(times),
            'min_inference_time': np.min(times),
            'max_inference_time': np.max(times),
            'throughput': input_tensor.shape[0] / np.mean(times),
            'device': str(device),
            'batch_size': input_tensor.shape[0],
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
        }
    
    def profile_model(self) -> Dict[str, Any]:
        """Profile model architecture and memory usage.
        
        Returns:
            Model profile information
        """
        profile = {
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            'model_size_mb': sum(
                p.numel() * p.element_size() for p in self.model.parameters()
            ) / 1024 / 1024,
            'layers': []
        }
        
        # Profile each layer
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                layer_params = sum(p.numel() for p in module.parameters())
                if layer_params > 0:
                    profile['layers'].append({
                        'name': name,
                        'type': module.__class__.__name__,
                        'parameters': layer_params,
                        'output_shape': self._get_output_shape(module)
                    })
        
        return profile
    
    def _get_output_shape(self, module: torch.nn.Module) -> Optional[str]:
        """Try to get output shape of a module."""
        # This is a simplified version - in practice you'd want more sophisticated shape inference
        if hasattr(module, 'out_features'):
            return f"({module.out_features},)"
        elif hasattr(module, 'out_channels'):
            return f"(C={module.out_channels},...)"
        return None


def benchmark_function(
    func: Callable,
    *args,
    num_runs: int = 100,
    warmup: int = 10,
    **kwargs
) -> Dict[str, float]:
    """Benchmark a function's performance.
    
    Args:
        func: Function to benchmark
        *args: Positional arguments for function
        num_runs: Number of runs to measure
        warmup: Number of warmup runs
        **kwargs: Keyword arguments for function
        
    Returns:
        Benchmark results
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Measure
    times = []
    for _ in range(num_runs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        times.append(end - start)
    
    times = np.array(times)
    
    return {
        'function': func.__name__,
        'num_runs': num_runs,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'total_time': np.sum(times),
    }