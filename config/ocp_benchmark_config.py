#!/usr/bin/env python3
"""OpenShift Benchmark Configuration Module"""

import configparser
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BenchmarkConfig:
    """Configuration management for OpenShift benchmark MCP server"""
    
    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            config_dir = Path(__file__).parent
        
        self.config_dir = Path(config_dir)
        self.baseline_config = configparser.ConfigParser(interpolation=None)
        self.metrics_config = {}
        
        # Load configurations
        self._load_baseline_config()
        self._load_metrics_config()
    
    def _load_baseline_config(self) -> None:
        """Load baseline configuration from properties file"""
        baseline_path = self.config_dir / "baseline.properties"
        
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline configuration not found: {baseline_path}")
        
        self.baseline_config.read(baseline_path)
        logger.info(f"Loaded baseline configuration from {baseline_path}")
    
    def _load_metrics_config(self) -> None:
        """Load metrics configuration from YAML file"""
        metrics_path = self.config_dir / "metrics.yml"
        
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics configuration not found: {metrics_path}")
        
        with open(metrics_path, 'r') as file:
            self.metrics_config = yaml.safe_load(file)
        
        logger.info(f"Loaded metrics configuration from {metrics_path}")
    
    def get_baseline_value(self, section: str, key: str, fallback: Any = None) -> Any:
        """Get baseline configuration value"""
        try:
            value = self.baseline_config.get('DEFAULT', f"{section}.{key}")
            # Try to convert to appropriate type
            if '.' in value:
                return float(value)
            elif value.isdigit():
                return int(value)
            elif value.lower() in ('true', 'false'):
                return value.lower() == 'true'
            return value
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    def get_metric_query(self, category: str, metric: str) -> Optional[str]:
        """Get PromQL query for a specific metric"""
        try:
            return self.metrics_config['metrics'][category][metric]['query']
        except KeyError:
            logger.error(f"Metric not found: {category}.{metric}")
            return None
    
    def get_metric_description(self, category: str, metric: str) -> Optional[str]:
        """Get description for a specific metric"""
        try:
            return self.metrics_config['metrics'][category][metric]['description']
        except KeyError:
            return None
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics configuration"""
        return self.metrics_config.get('metrics', {})
    
    def get_cpu_baselines(self) -> Dict[str, float]:
        """Get CPU baseline values"""
        return {
            'min': self.get_baseline_value('cpu.baseline', 'min', 10.0),
            'max': self.get_baseline_value('cpu.baseline', 'max', 80.0),
            'mean': self.get_baseline_value('cpu.baseline', 'mean', 45.0),
            'variance': self.get_baseline_value('cpu.acceptable', 'variance', 10.0)
        }
    
    def get_memory_baselines(self) -> Dict[str, float]:
        """Get memory baseline values"""
        return {
            'min': self.get_baseline_value('memory.baseline', 'min', 20.0),
            'max': self.get_baseline_value('memory.baseline', 'max', 85.0),
            'mean': self.get_baseline_value('memory.baseline', 'mean', 50.0),
            'variance': self.get_baseline_value('memory.acceptable', 'variance', 15.0)
        }
    
    def get_disk_baselines(self) -> Dict[str, float]:
        """Get disk I/O baseline values"""
        return {
            'read_baseline': self.get_baseline_value('disk.io.read', 'baseline', 100.0),
            'write_baseline': self.get_baseline_value('disk.io.write', 'baseline', 50.0),
            'read_iops': self.get_baseline_value('disk.read', 'iops', 10000),
            'write_iops': self.get_baseline_value('disk.write', 'iops', 8000),
            'read_latency_ms': self.get_baseline_value('disk.average.read.latency', 'ms', 5.0),
            'write_latency_ms': self.get_baseline_value('disk.average.write', 'latency_ms', 8.0)
        }
    
    def get_network_baselines(self) -> Dict[str, float]:
        """Get network baseline values"""
        return {
            'rx_baseline': self.get_baseline_value('network.rx', 'baseline', 10.0),
            'tx_baseline': self.get_baseline_value('network.tx', 'baseline', 10.0),
            'max_throughput_mbps': self.get_baseline_value('network.max.throughput', 'mbps', 1000),
            'average_latency_ms': self.get_baseline_value('network.average.latency', 'ms', 2.0),
            'packet_loss_threshold': self.get_baseline_value('network.packet.loss', 'threshold', 0.1)
        }
    
    def get_api_baselines(self) -> Dict[str, float]:
        """Get API latency baseline values"""
        return {
            'p50': self.get_baseline_value('api.latency.p50', 'baseline', 100.0),
            'p95': self.get_baseline_value('api.latency.p95', 'baseline', 500.0),
            'p99': self.get_baseline_value('api.latency.p99', 'baseline', 1000.0)
        }
    
    def get_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get warning and critical thresholds"""
        return {
            'warning': {
                'cpu': self.get_baseline_value('warning.cpu', 'usage', 80.0),
                'memory': self.get_baseline_value('warning.memory', 'usage', 85.0),
                'disk': self.get_baseline_value('warning.disk', 'usage', 75.0),
                'response_time_ms': self.get_baseline_value('warning.response', 'time_ms', 500)
            },
            'critical': {
                'cpu': self.get_baseline_value('critical.cpu', 'usage', 95.0),
                'memory': self.get_baseline_value('critical.memory', 'usage', 98.0),
                'disk': self.get_baseline_value('critical.disk', 'usage', 90.0),
                'response_time_ms': self.get_baseline_value('critical.response', 'time_ms', 1000)
            }
        }

# Global configuration instance
config = BenchmarkConfig()