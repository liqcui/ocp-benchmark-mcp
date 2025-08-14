"""Configuration management for OpenShift Benchmark MCP Server."""
import configparser
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages configuration from properties files and YAML files."""
    
    def __init__(self):
        self.config_dir = Path(__file__).parent
        self.baseline_config = None
        self.metrics_config = None
        self._load_configs()
    
    def _load_configs(self):
        """Load baseline properties and metrics configuration."""
        # Load baseline properties
        baseline_path = self.config_dir / "baseline.properties"
        if baseline_path.exists():
            self.baseline_config = configparser.ConfigParser(interpolation=None)
            self.baseline_config.read(baseline_path)
        
        # Load metrics YAML
        metrics_path = self.config_dir / "metrics.yml"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                self.metrics_config = yaml.safe_load(f)
    
    def get_baseline(self, key: str, fallback: Optional[float] = None) -> Optional[float]:
        """Get baseline value by key."""
        if not self.baseline_config:
            return fallback
        
        # Try DEFAULT section first, then any section
        for section_name in self.baseline_config.sections() + ['DEFAULT']:
            if self.baseline_config.has_option(section_name, key):
                try:
                    return float(self.baseline_config.get(section_name, key))
                except (ValueError, TypeError):
                    continue
        
        return fallback
    
    def get_baseline_string(self, key: str, fallback: Optional[str] = None) -> Optional[str]:
        """Get baseline string value by key."""
        if not self.baseline_config:
            return fallback
        
        # Try DEFAULT section first, then any section
        for section_name in self.baseline_config.sections() + ['DEFAULT']:
            if self.baseline_config.has_option(section_name, key):
                return self.baseline_config.get(section_name, key)
        
        return fallback
    
    def get_metric_query(self, category: str, metric: str) -> Optional[str]:
        """Get PromQL query for a specific metric."""
        if not self.metrics_config:
            return None
        
        return self.metrics_config.get(category, {}).get(metric, {}).get('query')
    
    def get_metric_info(self, category: str, metric: str) -> Dict[str, Any]:
        """Get complete metric information."""
        if not self.metrics_config:
            return {}
        
        return self.metrics_config.get(category, {}).get(metric, {})
    
    def get_all_metrics(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Get all metrics for a category."""
        if not self.metrics_config:
            return {}
        
        return self.metrics_config.get(category, {})
    
    def get_cpu_baselines(self) -> Dict[str, float]:
        """Get all CPU-related baselines."""
        return {
            'min': self.get_baseline('cpu.baseline.min', 10.0),
            'max': self.get_baseline('cpu.baseline.max', 80.0),
            'mean': self.get_baseline('cpu.baseline.mean', 45.0),
            'variance': self.get_baseline('cpu.acceptable.variance', 10.0),
            'warning': self.get_baseline('performance.cpu.warning', 70.0),
            'critical': self.get_baseline('performance.cpu.critical', 90.0),
        }
    
    def get_memory_baselines(self) -> Dict[str, float]:
        """Get all memory-related baselines."""
        return {
            'min': self.get_baseline('memory.baseline.min', 20.0),
            'max': self.get_baseline('memory.baseline.max', 85.0),
            'mean': self.get_baseline('memory.baseline.mean', 50.0),
            'variance': self.get_baseline('memory.acceptable.variance', 15.0),
            'warning': self.get_baseline('performance.memory.warning', 75.0),
            'critical': self.get_baseline('performance.memory.critical', 90.0),
        }
    
    def get_disk_baselines(self) -> Dict[str, float]:
        """Get all disk I/O baselines."""
        return {
            'read_baseline': self.get_baseline('disk.io.read.baseline', 100.0),
            'write_baseline': self.get_baseline('disk.io.write.baseline', 50.0),
            'read_iops': self.get_baseline('disk.read.iops', 10000),
            'write_iops': self.get_baseline('disk.write.iops', 8000),
            'read_latency': self.get_baseline('disk.average.read.latency.ms', 5.0),
            'write_latency': self.get_baseline('disk.average.write.latency_ms', 8.0),
            'peak_latency': self.get_baseline('disk.peak.latency.threshold_ms', 50.0),
        }
    
    def get_network_baselines(self) -> Dict[str, float]:
        """Get all network baselines."""
        return {
            'rx_baseline': self.get_baseline('network.rx.baseline', 10.0),
            'tx_baseline': self.get_baseline('network.tx.baseline', 10.0),
            'max_throughput': self.get_baseline('network.max.throughput.mbps', 1000),
            'average_latency': self.get_baseline('network.average.latency.ms', 2.0),
            'packet_loss_threshold': self.get_baseline('network.packet.loss.threshold', 0.1),
            'max_connections': self.get_baseline('network.max.connections', 10000),
        }
    
    def get_api_baselines(self) -> Dict[str, float]:
        """Get all API latency baselines."""
        return {
            'p50': self.get_baseline('api.latency.p50.baseline', 100.0),
            'p95': self.get_baseline('api.latency.p95.baseline', 500.0),
            'p99': self.get_baseline('api.latency.p99.baseline', 1000.0),
            'etcd_response_time': self.get_baseline('etcd.response.time_ms', 10.0),
        }
    
    def get_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get warning and critical thresholds."""
        return {
            'cpu': {
                'warning': self.get_baseline('warning.cpu.usage', 80.0),
                'critical': self.get_baseline('critical.cpu.usage', 95.0),
            },
            'memory': {
                'warning': self.get_baseline('warning.memory.usage', 85.0),
                'critical': self.get_baseline('critical.memory.usage', 98.0),
            },
            'disk': {
                'warning': self.get_baseline('warning.disk.usage', 75.0),
                'critical': self.get_baseline('critical.disk.usage', 90.0),
            },
            'response_time': {
                'warning': self.get_baseline('warning.response.time_ms', 500.0),
                'critical': self.get_baseline('critical.response.time_ms', 1000.0),
            }
        }


# Global config instance
config_manager = ConfigManager()