"""Pod usage metrics collection from Prometheus."""
import json
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from tools.ocp_benchmark_prometheus_basequery import prometheus_client
from config.ocp_benchmark_config import config_manager


logger = logging.getLogger(__name__)


class PodsUsageCollector:
    """Collects CPU and RAM usage metrics for pods."""
    
    def __init__(self):
        self.prometheus = prometheus_client
        self.config = config_manager
    
    def get_pod_cpu_usage(self, 
                         start_time: datetime, 
                         end_time: datetime,
                         pod_regex: Optional[str] = None,
                         label_selectors: Optional[Dict[str, str]] = None,
                         step: str = '1m') -> Dict[str, Any]:
        """Get CPU usage for pods matching criteria.
        
        Args:
            start_time: Start of the time range
            end_time: End of the time range
            pod_regex: Regular expression to match pod names
            label_selectors: Dictionary of label key-value pairs to match
            step: Query resolution step
        
        Returns:
            Dictionary containing CPU usage data
        """
        query = self.config.get_metric_query('pod_metrics', 'cpu_usage')
        if not query:
            query = 'rate(container_cpu_usage_seconds_total{container!="",container!="POD"}[5m]) * 100'
        
        # Add label selectors to query if provided
        if label_selectors:
            label_filters = []
            for key, value in label_selectors.items():
                label_filters.append(f'{key}="{value}"')
            if label_filters:
                # Insert label filters into the query
                query = query.replace('{', '{' + ','.join(label_filters) + ',')
        
        try:
            result = self.prometheus.query_range(query, start_time, end_time, step)
            formatted_result = self.prometheus.format_query_result(result)
            
            # Filter by pod regex if provided and group by pod
            pods_data = {}
            for item in formatted_result['results']:
                pod_name = item['metric'].get('pod', 'unknown')
                namespace = item['metric'].get('namespace', 'unknown')
                container = item['metric'].get('container', 'unknown')
                
                # Apply regex filter if provided
                if pod_regex and not re.match(pod_regex, pod_name):
                    continue
                
                pod_key = f"{namespace}/{pod_name}"
                if pod_key not in pods_data:
                    pods_data[pod_key] = {
                        'namespace': namespace,
                        'pod_name': pod_name,
                        'containers': {},
                        'total_statistics': {'min': 0, 'max': 0, 'mean': 0, 'count': 0}
                    }
                
                container_stats = self.prometheus.calculate_statistics([{'values': [(v['timestamp'], v['value']) for v in item['values']]}])
                pods_data[pod_key]['containers'][container] = {
                    'metric_labels': item['metric'],
                    'values': item['values'],
                    'statistics': container_stats
                }
            
            # Calculate pod-level statistics (sum of all containers)
            for pod_key, pod_data in pods_data.items():
                container_means = [stats['statistics']['mean'] for stats in pod_data['containers'].values() if stats['statistics']['count'] > 0]
                container_maxs = [stats['statistics']['max'] for stats in pod_data['containers'].values() if stats['statistics']['count'] > 0]
                container_mins = [stats['statistics']['min'] for stats in pod_data['containers'].values() if stats['statistics']['count'] > 0]
                
                if container_means:
                    pod_data['total_statistics'] = {
                        'min': sum(container_mins),
                        'max': sum(container_maxs),
                        'mean': sum(container_means),
                        'count': len(container_means)
                    }
            
            return {
                'query': query,
                'metric_type': 'pod_cpu_usage_percent',
                'filters': {
                    'pod_regex': pod_regex,
                    'label_selectors': label_selectors
                },
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'step': step
                },
                'pods': pods_data,
                'cluster_statistics': formatted_result['statistics'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get pod CPU usage: {e}")
            raise
    
    def get_pod_memory_usage(self, 
                            start_time: datetime, 
                            end_time: datetime,
                            pod_regex: Optional[str] = None,
                            label_selectors: Optional[Dict[str, str]] = None,
                            step: str = '1m') -> Dict[str, Any]:
        """Get memory usage for pods matching criteria.
        
        Args:
            start_time: Start of the time range
            end_time: End of the time range
            pod_regex: Regular expression to match pod names
            label_selectors: Dictionary of label key-value pairs to match
            step: Query resolution step
        
        Returns:
            Dictionary containing memory usage data
        """
        query = self.config.get_metric_query('pod_metrics', 'memory_usage')
        if not query:
            query = 'container_memory_working_set_bytes{container!="",container!="POD"}'
        
        # Add label selectors to query if provided
        if label_selectors:
            label_filters = []
            for key, value in label_selectors.items():
                label_filters.append(f'{key}="{value}"')
            if label_filters:
                query = query.replace('{', '{' + ','.join(label_filters) + ',')
        
        try:
            result = self.prometheus.query_range(query, start_time, end_time, step)
            formatted_result = self.prometheus.format_query_result(result)
            
            # Filter by pod regex if provided and group by pod
            pods_data = {}
            for item in formatted_result['results']:
                pod_name = item['metric'].get('pod', 'unknown')
                namespace = item['metric'].get('namespace', 'unknown')
                container = item['metric'].get('container', 'unknown')
                
                # Apply regex filter if provided
                if pod_regex and not re.match(pod_regex, pod_name):
                    continue
                
                pod_key = f"{namespace}/{pod_name}"
                if pod_key not in pods_data:
                    pods_data[pod_key] = {
                        'namespace': namespace,
                        'pod_name': pod_name,
                        'containers': {},
                        'total_statistics': {'min': 0, 'max': 0, 'mean': 0, 'count': 0}
                    }
                
                # Convert bytes to MB for easier reading
                values_mb = [{'timestamp': v['timestamp'], 'value': v['value'] / (1024 * 1024)} for v in item['values']]
                container_stats = self.prometheus.calculate_statistics([{'values': [(v['timestamp'], v['value']) for v in values_mb]}])
                
                pods_data[pod_key]['containers'][container] = {
                    'metric_labels': item['metric'],
                    'values': values_mb,
                    'statistics': container_stats,
                    'unit': 'MB'
                }
            
            # Calculate pod-level statistics (sum of all containers)
            for pod_key, pod_data in pods_data.items():
                container_means = [stats['statistics']['mean'] for stats in pod_data['containers'].values() if stats['statistics']['count'] > 0]
                container_maxs = [stats['statistics']['max'] for stats in pod_data['containers'].values() if stats['statistics']['count'] > 0]
                container_mins = [stats['statistics']['min'] for stats in pod_data['containers'].values() if stats['statistics']['count'] > 0]
                
                if container_means:
                    pod_data['total_statistics'] = {
                        'min': sum(container_mins),
                        'max': sum(container_maxs),
                        'mean': sum(container_means),
                        'count': len(container_means),
                        'unit': 'MB'
                    }
            
            return {
                'query': query,
                'metric_type': 'pod_memory_usage_mb',
                'filters': {
                    'pod_regex': pod_regex,
                    'label_selectors': label_selectors
                },
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'step': step
                },
                'pods': pods_data,
                'cluster_statistics': formatted_result['statistics'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get pod memory usage: {e}")
            raise
    
    def get_pod_memory_usage_percent(self, 
                                   start_time: datetime, 
                                   end_time: datetime,
                                   pod_regex: Optional[str] = None,
                                   label_selectors: Optional[Dict[str, str]] = None,
                                   step: str = '1m') -> Dict[str, Any]:
        """Get memory usage as percentage of limit for pods matching criteria."""
        query = self.config.get_metric_query('pod_metrics', 'memory_usage_percent')
        if not query:
            query = '(container_memory_working_set_bytes{container!="",container!="POD"} / container_spec_memory_limit_bytes{container!="",container!="POD"}) * 100'
        
        # Add label selectors to query if provided
        if label_selectors:
            label_filters = []
            for key, value in label_selectors.items():
                label_filters.append(f'{key}="{value}"')
            if label_filters:
                query = query.replace('{', '{' + ','.join(label_filters) + ',')
        
        try:
            result = self.prometheus.query_range(query, start_time, end_time, step)
            formatted_result = self.prometheus.format_query_result(result)
            
            # Filter by pod regex if provided and group by pod
            pods_data = {}
            for item in formatted_result['results']:
                pod_name = item['metric'].get('pod', 'unknown')
                namespace = item['metric'].get('namespace', 'unknown')
                container = item['metric'].get('container', 'unknown')
                
                # Apply regex filter if provided
                if pod_regex and not re.match(pod_regex, pod_name):
                    continue
                
                pod_key = f"{namespace}/{pod_name}"
                if pod_key not in pods_data:
                    pods_data[pod_key] = {
                        'namespace': namespace,
                        'pod_name': pod_name,
                        'containers': {},
                    }
                
                container_stats = self.prometheus.calculate_statistics([{'values': [(v['timestamp'], v['value']) for v in item['values']]}])
                pods_data[pod_key]['containers'][container] = {
                    'metric_labels': item['metric'],
                    'values': item['values'],
                    'statistics': container_stats,
                    'unit': 'percent'
                }
            
            return {
                'query': query,
                'metric_type': 'pod_memory_usage_percent_of_limit',
                'filters': {
                    'pod_regex': pod_regex,
                    'label_selectors': label_selectors
                },
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'step': step
                },
                'pods': pods_data,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get pod memory usage percent: {e}")
            raise
    
    def collect_pods_usage(self, 
                          duration_hours: float = 1.0,
                          pod_regex: Optional[str] = None,
                          label_selectors: Optional[Dict[str, str]] = None,
                          step: str = '1m') -> Dict[str, Any]:
        """Collect CPU and RAM usage for pods matching criteria over a specified duration.
        
        Args:
            duration_hours: Duration in hours to collect data for
            pod_regex: Regular expression to match pod names
            label_selectors: Dictionary of label key-value pairs to match
            step: Query resolution step
        
        Returns:
            Dictionary containing usage data and statistics
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=duration_hours)
        
        try:
            # Get CPU usage data
            cpu_data = self.get_pod_cpu_usage(start_time, end_time, pod_regex, label_selectors, step)
            
            # Get memory usage data (absolute values)
            memory_data = self.get_pod_memory_usage(start_time, end_time, pod_regex, label_selectors, step)
            
            # Get memory usage as percentage of limit
            memory_percent_data = self.get_pod_memory_usage_percent(start_time, end_time, pod_regex, label_selectors, step)
            
            # Combine data by pod
            all_pod_keys = set(cpu_data.get('pods', {}).keys()) | set(memory_data.get('pods', {}).keys())
            combined_pods = {}
            
            for pod_key in all_pod_keys:
                cpu_pod_data = cpu_data.get('pods', {}).get(pod_key, {})
                memory_pod_data = memory_data.get('pods', {}).get(pod_key, {})
                memory_percent_pod_data = memory_percent_data.get('pods', {}).get(pod_key, {})
                
                # Get basic pod info
                namespace = cpu_pod_data.get('namespace') or memory_pod_data.get('namespace', 'unknown')
                pod_name = cpu_pod_data.get('pod_name') or memory_pod_data.get('pod_name', 'unknown')
                
                # Combine container data
                all_containers = set(cpu_pod_data.get('containers', {}).keys()) | set(memory_pod_data.get('containers', {}).keys())
                containers = {}
                
                for container_name in all_containers:
                    cpu_container = cpu_pod_data.get('containers', {}).get(container_name, {})
                    memory_container = memory_pod_data.get('containers', {}).get(container_name, {})
                    memory_percent_container = memory_percent_pod_data.get('containers', {}).get(container_name, {})
                    
                    containers[container_name] = {
                        'cpu_usage': {
                            'values': cpu_container.get('values', []),
                            'statistics': cpu_container.get('statistics', {}),
                            'unit': 'percent'
                        },
                        'memory_usage': {
                            'values': memory_container.get('values', []),
                            'statistics': memory_container.get('statistics', {}),
                            'unit': memory_container.get('unit', 'MB')
                        },
                        'memory_usage_percent_of_limit': {
                            'values': memory_percent_container.get('values', []),
                            'statistics': memory_percent_container.get('statistics', {}),
                            'unit': 'percent'
                        },
                        'labels': cpu_container.get('metric_labels', {}) or memory_container.get('metric_labels', {})
                    }
                
                combined_pods[pod_key] = {
                    'namespace': namespace,
                    'pod_name': pod_name,
                    'containers': containers,
                    'pod_totals': {
                        'cpu_usage': cpu_pod_data.get('total_statistics', {}),
                        'memory_usage': memory_pod_data.get('total_statistics', {})
                    }
                }
            
            # Calculate summary statistics
            total_pods = len(combined_pods)
            total_containers = sum(len(pod['containers']) for pod in combined_pods.values())
            
            # Group pods by namespace
            namespaces = {}
            for pod_key, pod_data in combined_pods.items():
                ns = pod_data['namespace']
                if ns not in namespaces:
                    namespaces[ns] = {'pod_count': 0, 'container_count': 0}
                namespaces[ns]['pod_count'] += 1
                namespaces[ns]['container_count'] += len(pod_data['containers'])
            
            # Performance analysis
            high_cpu_pods = []
            high_memory_pods = []
            
            cpu_threshold = 80.0  # 80% CPU
            memory_threshold = 80.0  # 80% memory
            
            for pod_key, pod_data in combined_pods.items():
                # Check CPU usage
                cpu_stats = pod_data['pod_totals'].get('cpu_usage', {})
                if cpu_stats.get('max', 0) > cpu_threshold:
                    high_cpu_pods.append({
                        'pod': pod_key,
                        'max_cpu_percent': cpu_stats.get('max', 0),
                        'mean_cpu_percent': cpu_stats.get('mean', 0)
                    })
                
                # Check memory usage (look at individual containers for memory limits)
                for container_name, container_data in pod_data['containers'].items():
                    memory_percent_stats = container_data.get('memory_usage_percent_of_limit', {}).get('statistics', {})
                    if memory_percent_stats.get('max', 0) > memory_threshold:
                        high_memory_pods.append({
                            'pod': pod_key,
                            'container': container_name,
                            'max_memory_percent': memory_percent_stats.get('max', 0),
                            'mean_memory_percent': memory_percent_stats.get('mean', 0)
                        })
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'collection_period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'duration_hours': duration_hours,
                    'step': step
                },
                'filters_applied': {
                    'pod_regex': pod_regex,
                    'label_selectors': label_selectors
                },
                'summary': {
                    'total_pods': total_pods,
                    'total_containers': total_containers,
                    'namespaces': namespaces,
                    'high_cpu_pods': len(high_cpu_pods),
                    'high_memory_pods': len(high_memory_pods)
                },
                'performance_alerts': {
                    'high_cpu_usage': high_cpu_pods,
                    'high_memory_usage': high_memory_pods,
                    'thresholds': {
                        'cpu_percent': cpu_threshold,
                        'memory_percent': memory_threshold
                    }
                },
                'pods': combined_pods,
                'queries_executed': {
                    'cpu_usage': cpu_data['query'],
                    'memory_usage': memory_data['query'],
                    'memory_usage_percent': memory_percent_data['query']
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to collect pods usage: {e}")
            raise


def get_pods_usage(duration_hours: float = 1.0, 
                   pod_regex: Optional[str] = None,
                   label_selectors: Optional[List[str]] = None,
                   step: str = '1m') -> str:
    """Get pods usage information and return as JSON string.
    
    Args:
        duration_hours: Duration in hours to collect data for (default: 1 hour)
        pod_regex: Regular expression to match pod names (optional)
        label_selectors: List of label selectors in format "key=value" (optional)
        step: Query resolution step (default: '1m')
    
    Returns:
        JSON string containing pods usage data
    """
    # Parse label selectors from list to dict
    label_dict = None
    if label_selectors:
        label_dict = {}
        for selector in label_selectors:
            if '=' in selector:
                key, value = selector.split('=', 1)
                label_dict[key.strip()] = value.strip()
    
    collector = PodsUsageCollector()
    usage_data = collector.collect_pods_usage(duration_hours, pod_regex, label_dict, step)
    return json.dumps(usage_data, indent=2)