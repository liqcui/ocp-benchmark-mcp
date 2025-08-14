"""Node usage metrics collection from Prometheus."""
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from tools.ocp_benchmark_prometheus_basequery import prometheus_client
from config.ocp_benchmark_config import config_manager


logger = logging.getLogger(__name__)


class NodesUsageCollector:
    """Collects CPU and RAM usage metrics for nodes."""
    
    def __init__(self):
        self.prometheus = prometheus_client
        self.config = config_manager
    
    def get_node_cpu_usage(self, 
                          start_time: datetime, 
                          end_time: datetime, 
                          step: str = '1m') -> Dict[str, Any]:
        """Get CPU usage for all nodes in a time range.
        
        Args:
            start_time: Start of the time range
            end_time: End of the time range
            step: Query resolution step
        
        Returns:
            Dictionary containing CPU usage data
        """
        query = self.config.get_metric_query('node_metrics', 'cpu_usage')
        if not query:
            query = '(1 - rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100'
        
        try:
            result = self.prometheus.query_range(query, start_time, end_time, step)
            formatted_result = self.prometheus.format_query_result(result)
            
            # Group by node
            nodes_data = {}
            for item in formatted_result['results']:
                instance = item['metric'].get('instance', 'unknown')
                node_name = item['metric'].get('instance', 'unknown').split(':')[0]  # Remove port if present
                
                nodes_data[node_name] = {
                    'metric_labels': item['metric'],
                    'values': item['values'],
                    'statistics': self.prometheus.calculate_statistics([{'values': [(v['timestamp'], v['value']) for v in item['values']]}])
                }
            
            return {
                'query': query,
                'metric_type': 'node_cpu_usage_percent',
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'step': step
                },
                'nodes': nodes_data,
                'cluster_statistics': formatted_result['statistics'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get node CPU usage: {e}")
            raise
    
    def get_node_memory_usage(self, 
                             start_time: datetime, 
                             end_time: datetime, 
                             step: str = '1m') -> Dict[str, Any]:
        """Get memory usage for all nodes in a time range.
        
        Args:
            start_time: Start of the time range
            end_time: End of the time range
            step: Query resolution step
        
        Returns:
            Dictionary containing memory usage data
        """
        query = self.config.get_metric_query('node_metrics', 'memory_usage')
        if not query:
            query = '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100'
        
        try:
            result = self.prometheus.query_range(query, start_time, end_time, step)
            formatted_result = self.prometheus.format_query_result(result)
            
            # Group by node
            nodes_data = {}
            for item in formatted_result['results']:
                instance = item['metric'].get('instance', 'unknown')
                node_name = item['metric'].get('instance', 'unknown').split(':')[0]  # Remove port if present
                
                nodes_data[node_name] = {
                    'metric_labels': item['metric'],
                    'values': item['values'],
                    'statistics': self.prometheus.calculate_statistics([{'values': [(v['timestamp'], v['value']) for v in item['values']]}])
                }
            
            return {
                'query': query,
                'metric_type': 'node_memory_usage_percent',
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'step': step
                },
                'nodes': nodes_data,
                'cluster_statistics': formatted_result['statistics'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get node memory usage: {e}")
            raise
    
    def get_node_memory_total(self) -> Dict[str, Any]:
        """Get total memory for all nodes."""
        query = self.config.get_metric_query('node_metrics', 'memory_total')
        if not query:
            query = 'node_memory_MemTotal_bytes'
        
        try:
            result = self.prometheus.query_instant(query)
            formatted_result = self.prometheus.format_query_result(result)
            
            # Group by node
            nodes_data = {}
            for item in formatted_result['results']:
                instance = item['metric'].get('instance', 'unknown')
                node_name = item['metric'].get('instance', 'unknown').split(':')[0]
                
                memory_bytes = item['values'][0]['value'] if item['values'] else 0
                memory_gb = memory_bytes / (1024 ** 3)
                
                nodes_data[node_name] = {
                    'metric_labels': item['metric'],
                    'memory_bytes': memory_bytes,
                    'memory_gb': round(memory_gb, 2)
                }
            
            return {
                'query': query,
                'metric_type': 'node_memory_total',
                'nodes': nodes_data,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get node total memory: {e}")
            raise
    
    def collect_nodes_usage(self, 
                           duration_hours: float = 1.0, 
                           step: str = '1m') -> Dict[str, Any]:
        """Collect CPU and RAM usage for all nodes over a specified duration.
        
        Args:
            duration_hours: Duration in hours to collect data for
            step: Query resolution step
        
        Returns:
            Dictionary containing usage data and statistics
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=duration_hours)
        
        try:
            # Get CPU usage data
            cpu_data = self.get_node_cpu_usage(start_time, end_time, step)
            
            # Get memory usage data
            memory_data = self.get_node_memory_usage(start_time, end_time, step)
            
            # Get total memory for context
            memory_total_data = self.get_node_memory_total()
            
            # Combine data by node
            combined_nodes = {}
            all_node_names = set(cpu_data.get('nodes', {}).keys()) | set(memory_data.get('nodes', {}).keys())
            
            for node_name in all_node_names:
                cpu_node_data = cpu_data.get('nodes', {}).get(node_name, {})
                memory_node_data = memory_data.get('nodes', {}).get(node_name, {})
                memory_total_node_data = memory_total_data.get('nodes', {}).get(node_name, {})
                
                combined_nodes[node_name] = {
                    'cpu_usage': {
                        'values': cpu_node_data.get('values', []),
                        'statistics': cpu_node_data.get('statistics', {}),
                        'unit': 'percent'
                    },
                    'memory_usage': {
                        'values': memory_node_data.get('values', []),
                        'statistics': memory_node_data.get('statistics', {}),
                        'unit': 'percent'
                    },
                    'memory_total': {
                        'bytes': memory_total_node_data.get('memory_bytes', 0),
                        'gb': memory_total_node_data.get('memory_gb', 0)
                    },
                    'labels': cpu_node_data.get('metric_labels', {})
                }
            
            # Calculate cluster-wide statistics
            cluster_stats = {
                'cpu_usage': cpu_data.get('cluster_statistics', {}),
                'memory_usage': memory_data.get('cluster_statistics', {}),
                'total_nodes': len(combined_nodes),
                'query_duration_hours': duration_hours
            }
            
            # Compare against baselines
            cpu_baselines = self.config.get_cpu_baselines()
            memory_baselines = self.config.get_memory_baselines()
            
            baseline_comparison = {
                'cpu': self._compare_against_baseline(
                    cluster_stats['cpu_usage'], cpu_baselines
                ),
                'memory': self._compare_against_baseline(
                    cluster_stats['memory_usage'], memory_baselines
                )
            }
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'collection_period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'duration_hours': duration_hours,
                    'step': step
                },
                'cluster_statistics': cluster_stats,
                'baseline_comparison': baseline_comparison,
                'nodes': combined_nodes,
                'queries_executed': {
                    'cpu_usage': cpu_data['query'],
                    'memory_usage': memory_data['query'],
                    'memory_total': memory_total_data['query']
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to collect nodes usage: {e}")
            raise
    
    def _compare_against_baseline(self, stats: Dict[str, float], baselines: Dict[str, float]) -> Dict[str, Any]:
        """Compare statistics against baseline values."""
        if not stats or 'mean' not in stats:
            return {'status': 'no_data', 'message': 'No statistics available for comparison'}
        
        mean_value = stats['mean']
        max_value = stats.get('max', mean_value)
        
        comparison = {
            'current_mean': round(mean_value, 2),
            'current_max': round(max_value, 2),
            'baseline_mean': baselines.get('mean', 0),
            'baseline_max': baselines.get('max', 0),
            'status': 'normal'
        }
        
        # Determine status based on thresholds
        if max_value >= baselines.get('critical', 90):
            comparison['status'] = 'critical'
            comparison['message'] = f"Maximum value ({max_value:.1f}%) exceeds critical threshold ({baselines.get('critical', 90):.1f}%)"
        elif max_value >= baselines.get('warning', 75):
            comparison['status'] = 'warning'
            comparison['message'] = f"Maximum value ({max_value:.1f}%) exceeds warning threshold ({baselines.get('warning', 75):.1f}%)"
        elif mean_value > baselines.get('max', 80):
            comparison['status'] = 'warning'
            comparison['message'] = f"Mean value ({mean_value:.1f}%) exceeds baseline maximum ({baselines.get('max', 80):.1f}%)"
        else:
            comparison['message'] = f"Values within normal range (mean: {mean_value:.1f}%, max: {max_value:.1f}%)"
        
        return comparison


def get_nodes_usage(duration_hours: float = 1.0, step: str = '1m') -> str:
    """Get nodes usage information and return as JSON string.
    
    Args:
        duration_hours: Duration in hours to collect data for (default: 1 hour)
        step: Query resolution step (default: '1m')
    
    Returns:
        JSON string containing nodes usage data
    """
    collector = NodesUsageCollector()
    usage_data = collector.collect_nodes_usage(duration_hours, step)
    return json.dumps(usage_data, indent=2)