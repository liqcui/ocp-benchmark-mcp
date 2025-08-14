#!/usr/bin/env python3
"""Prometheus Node Usage Metrics Module"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import requests
import asyncio
from config.ocp_benchmark_config import config
from ocauth.ocp_benchmark_auth import auth

logger = logging.getLogger(__name__)

class NodeUsageCollector:
    """Collect node usage metrics from Prometheus"""
    
    def __init__(self):
        self.prometheus_url = None
        self.prometheus_token = None
        self._initialize_prometheus()
    
    def _initialize_prometheus(self):
        """Initialize Prometheus connection"""
        self.prometheus_url = auth.prometheus_url
        self.prometheus_token = auth.prometheus_token
    
    async def _query_prometheus(self, query: str, start_time: Optional[datetime] = None, 
                               end_time: Optional[datetime] = None, step: str = '1m') -> Optional[Dict[str, Any]]:
        """Execute Prometheus query"""
        try:
            if not self.prometheus_url or not self.prometheus_token:
                # Try to reinitialize
                self.prometheus_url, self.prometheus_token = await auth.initialize_prometheus_connection()
                if not self.prometheus_url or not self.prometheus_token:
                    logger.error("Prometheus connection not available")
                    return None
            
            headers = {
                'Authorization': f'Bearer {self.prometheus_token}',
                'Content-Type': 'application/json'
            }
            
            # Prepare query parameters
            params = {'query': query}
            
            if start_time and end_time:
                # Range query
                endpoint = f"{self.prometheus_url}/api/v1/query_range"
                params.update({
                    'start': start_time.timestamp(),
                    'end': end_time.timestamp(),
                    'step': step
                })
            else:
                # Instant query
                endpoint = f"{self.prometheus_url}/api/v1/query"
                if end_time:
                    params['time'] = end_time.timestamp()
            
            response = requests.get(
                endpoint,
                headers=headers,
                params=params,
                verify=False,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return data
                else:
                    logger.error(f"Prometheus query failed: {data.get('error', 'Unknown error')}")
            else:
                logger.error(f"Prometheus request failed: {response.status_code} - {response.text}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error querying Prometheus: {e}")
            return None
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate min, mean, max statistics from values"""
        if not values:
            return {'min': 0.0, 'mean': 0.0, 'max': 0.0, 'count': 0}
        
        return {
            'min': min(values),
            'mean': sum(values) / len(values),
            'max': max(values),
            'count': len(values)
        }
    
    def _extract_values_from_result(self, result: Dict[str, Any]) -> List[float]:
        """Extract numeric values from Prometheus result"""
        values = []
        
        if result.get('data', {}).get('result'):
            for series in result['data']['result']:
                if 'value' in series:
                    # Instant query result
                    try:
                        values.append(float(series['value'][1]))
                    except (ValueError, IndexError):
                        continue
                elif 'values' in series:
                    # Range query result
                    for value_pair in series['values']:
                        try:
                            values.append(float(value_pair[1]))
                        except (ValueError, IndexError):
                            continue
        
        return values
    
    async def get_node_cpu_usage(self, duration_hours: int = 1) -> Dict[str, Any]:
        """Get CPU usage statistics for all nodes"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=duration_hours)
            
            # Get CPU usage query from config
            cpu_query = config.get_metric_query('node_metrics', 'cpu_usage')
            if not cpu_query:
                logger.error("CPU usage query not found in config")
                return {'error': 'CPU usage query not configured'}
            
            result = await self._query_prometheus(cpu_query, start_time, end_time)
            if not result:
                return {'error': 'Failed to query Prometheus for CPU usage'}
            
            # Process results by node
            node_stats = {}
            
            for series in result['data']['result']:
                instance = series['metric'].get('instance', 'unknown')
                node_name = instance.split(':')[0] if ':' in instance else instance
                
                values = []
                if 'values' in series:
                    for value_pair in series['values']:
                        try:
                            values.append(float(value_pair[1]))
                        except (ValueError, IndexError):
                            continue
                
                if values:
                    node_stats[node_name] = self._calculate_stats(values)
            
            # Calculate overall statistics
            all_values = []
            for node, stats in node_stats.items():
                if 'values' in result['data']['result'][0]:
                    # Get all values from this node
                    for series in result['data']['result']:
                        if series['metric'].get('instance', '').startswith(node):
                            for value_pair in series.get('values', []):
                                try:
                                    all_values.append(float(value_pair[1]))
                                except (ValueError, IndexError):
                                    continue
            
            overall_stats = self._calculate_stats(all_values)
            
            return {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'duration_hours': duration_hours,
                'metric': 'cpu_usage_percentage',
                'overall_stats': overall_stats,
                'node_stats': node_stats,
                'baseline_comparison': self._compare_with_baseline(overall_stats, 'cpu')
            }
            
        except Exception as e:
            logger.error(f"Error getting node CPU usage: {e}")
            return {'error': str(e)}
    
    async def get_node_memory_usage(self, duration_hours: int = 1) -> Dict[str, Any]:
        """Get memory usage statistics for all nodes"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=duration_hours)
            
            # Get memory usage query from config
            memory_query = config.get_metric_query('node_metrics', 'memory_usage')
            if not memory_query:
                logger.error("Memory usage query not found in config")
                return {'error': 'Memory usage query not configured'}
            
            result = await self._query_prometheus(memory_query, start_time, end_time)
            if not result:
                return {'error': 'Failed to query Prometheus for memory usage'}
            
            # Process results by node
            node_stats = {}
            
            for series in result['data']['result']:
                instance = series['metric'].get('instance', 'unknown')
                node_name = instance.split(':')[0] if ':' in instance else instance
                
                values = []
                if 'values' in series:
                    for value_pair in series['values']:
                        try:
                            values.append(float(value_pair[1]))
                        except (ValueError, IndexError):
                            continue
                
                if values:
                    node_stats[node_name] = self._calculate_stats(values)
            
            # Calculate overall statistics
            all_values = []
            for series in result['data']['result']:
                for value_pair in series.get('values', []):
                    try:
                        all_values.append(float(value_pair[1]))
                    except (ValueError, IndexError):
                        continue
            
            overall_stats = self._calculate_stats(all_values)
            
            return {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'duration_hours': duration_hours,
                'metric': 'memory_usage_percentage',
                'overall_stats': overall_stats,
                'node_stats': node_stats,
                'baseline_comparison': self._compare_with_baseline(overall_stats, 'memory')
            }
            
        except Exception as e:
            logger.error(f"Error getting node memory usage: {e}")
            return {'error': str(e)}
    
    async def get_combined_node_usage(self, duration_hours: int = 1) -> Dict[str, Any]:
        """Get combined CPU and memory usage statistics"""
        try:
            cpu_task = self.get_node_cpu_usage(duration_hours)
            memory_task = self.get_node_memory_usage(duration_hours)
            
            cpu_result, memory_result = await asyncio.gather(cpu_task, memory_task)
            
            return {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'duration_hours': duration_hours,
                'cpu_usage': cpu_result,
                'memory_usage': memory_result
            }
            
        except Exception as e:
            logger.error(f"Error getting combined node usage: {e}")
            return {'error': str(e)}
    
    def _compare_with_baseline(self, stats: Dict[str, float], metric_type: str) -> Dict[str, Any]:
        """Compare statistics with baseline values"""
        try:
            if metric_type == 'cpu':
                baselines = config.get_cpu_baselines()
            elif metric_type == 'memory':
                baselines = config.get_memory_baselines()
            else:
                return {'error': f'Unknown metric type: {metric_type}'}
            
            comparison = {
                'baseline': baselines,
                'current': stats,
                'within_range': True,
                'deviations': {}
            }
            
            # Check if values are within acceptable range
            if stats['min'] < baselines['min']:
                comparison['within_range'] = False
                comparison['deviations']['min_below_baseline'] = baselines['min'] - stats['min']
            
            if stats['max'] > baselines['max']:
                comparison['within_range'] = False
                comparison['deviations']['max_above_baseline'] = stats['max'] - baselines['max']
            
            # Check if mean is within acceptable variance
            mean_deviation = abs(stats['mean'] - baselines['mean'])
            if mean_deviation > baselines['variance']:
                comparison['within_range'] = False
                comparison['deviations']['mean_variance_exceeded'] = mean_deviation - baselines['variance']
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing with baseline: {e}")
            return {'error': str(e)}

# Global node usage collector instance
node_usage_collector = NodeUsageCollector()

async def get_node_usage_json(duration_hours: int = 1) -> str:
    """Get node usage as JSON string"""
    usage = await node_usage_collector.get_combined_node_usage(duration_hours)
    return json.dumps(usage, indent=2)