"""API latency metrics collection from Prometheus."""
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from tools.ocp_benchmark_prometheus_basequery import prometheus_client
from config.ocp_benchmark_config import config_manager


logger = logging.getLogger(__name__)


class APILatencyCollector:
    """Collects API latency metrics from Prometheus."""
    
    def __init__(self):
        self.prometheus = prometheus_client
        self.config = config_manager
    
    def get_api_request_latency_p50(self, 
                                   start_time: datetime, 
                                   end_time: datetime, 
                                   step: str = '1m') -> Dict[str, Any]:
        """Get API server request latency 50th percentile."""
        query = self.config.get_metric_query('api_metrics', 'request_latency_p50')
        if not query:
            query = 'histogram_quantile(0.50, rate(apiserver_request_duration_seconds_bucket[5m])) * 1000'
        
        try:
            result = self.prometheus.query_range(query, start_time, end_time, step)
            formatted_result = self.prometheus.format_query_result(result)
            
            # Group by verb and resource if available
            grouped_data = {}
            for item in formatted_result['results']:
                verb = item['metric'].get('verb', 'unknown')
                resource = item['metric'].get('resource', 'unknown')
                key = f"{verb}:{resource}"
                
                stats = self.prometheus.calculate_statistics([item])
                grouped_data[key] = {
                    'metric_labels': item['metric'],
                    'values': item['values'],
                    'statistics': stats,
                    'unit': 'ms'
                }
            
            return {
                'query': query,
                'metric_type': 'api_request_latency_p50_ms',
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'step': step
                },
                'operations': grouped_data,
                'cluster_statistics': formatted_result['statistics'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get API request latency P50: {e}")
            raise
    
    def get_api_request_latency_p95(self, 
                                   start_time: datetime, 
                                   end_time: datetime, 
                                   step: str = '1m') -> Dict[str, Any]:
        """Get API server request latency 95th percentile."""
        query = self.config.get_metric_query('api_metrics', 'request_latency_p95')
        if not query:
            query = 'histogram_quantile(0.95, rate(apiserver_request_duration_seconds_bucket[5m])) * 1000'
        
        try:
            result = self.prometheus.query_range(query, start_time, end_time, step)
            formatted_result = self.prometheus.format_query_result(result)
            
            # Group by verb and resource if available
            grouped_data = {}
            for item in formatted_result['results']:
                verb = item['metric'].get('verb', 'unknown')
                resource = item['metric'].get('resource', 'unknown')
                key = f"{verb}:{resource}"
                
                stats = self.prometheus.calculate_statistics([item])
                grouped_data[key] = {
                    'metric_labels': item['metric'],
                    'values': item['values'],
                    'statistics': stats,
                    'unit': 'ms'
                }
            
            return {
                'query': query,
                'metric_type': 'api_request_latency_p95_ms',
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'step': step
                },
                'operations': grouped_data,
                'cluster_statistics': formatted_result['statistics'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get API request latency P95: {e}")
            raise
    
    def get_api_request_latency_p99(self, 
                                   start_time: datetime, 
                                   end_time: datetime, 
                                   step: str = '1m') -> Dict[str, Any]:
        """Get API server request latency 99th percentile."""
        query = self.config.get_metric_query('api_metrics', 'request_latency_p99')
        if not query:
            query = 'histogram_quantile(0.99, rate(apiserver_request_duration_seconds_bucket[5m])) * 1000'
        
        try:
            result = self.prometheus.query_range(query, start_time, end_time, step)
            formatted_result = self.prometheus.format_query_result(result)
            
            # Group by verb and resource if available
            grouped_data = {}
            for item in formatted_result['results']:
                verb = item['metric'].get('verb', 'unknown')
                resource = item['metric'].get('resource', 'unknown')
                key = f"{verb}:{resource}"
                
                stats = self.prometheus.calculate_statistics([item])
                grouped_data[key] = {
                    'metric_labels': item['metric'],
                    'values': item['values'],
                    'statistics': stats,
                    'unit': 'ms'
                }
            
            return {
                'query': query,
                'metric_type': 'api_request_latency_p99_ms',
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'step': step
                },
                'operations': grouped_data,
                'cluster_statistics': formatted_result['statistics'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get API request latency P99: {e}")
            raise
    
    def get_api_request_rate(self, 
                            start_time: datetime, 
                            end_time: datetime, 
                            step: str = '1m') -> Dict[str, Any]:
        """Get API server request rate."""
        query = self.config.get_metric_query('api_metrics', 'request_rate')
        if not query:
            query = 'rate(apiserver_request_total[5m])'
        
        try:
            result = self.prometheus.query_range(query, start_time, end_time, step)
            formatted_result = self.prometheus.format_query_result(result)
            
            # Group by verb and resource if available
            grouped_data = {}
            for item in formatted_result['results']:
                verb = item['metric'].get('verb', 'unknown')
                resource = item['metric'].get('resource', 'unknown')
                code = item['metric'].get('code', 'unknown')
                key = f"{verb}:{resource}:{code}"
                
                stats = self.prometheus.calculate_statistics([item])
                grouped_data[key] = {
                    'metric_labels': item['metric'],
                    'values': item['values'],
                    'statistics': stats,
                    'unit': 'requests/s'
                }
            
            return {
                'query': query,
                'metric_type': 'api_request_rate_per_second',
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'step': step
                },
                'operations': grouped_data,
                'cluster_statistics': formatted_result['statistics'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get API request rate: {e}")
            raise
    
    def get_etcd_latency(self, 
                        start_time: datetime, 
                        end_time: datetime, 
                        step: str = '1m') -> Dict[str, Any]:
        """Get etcd request latency."""
        query = self.config.get_metric_query('api_metrics', 'etcd_latency')
        print(f"Getting etcd latency: {query}")
        if not query:
            query = 'histogram_quantile(0.99, rate(etcd_request_duration_seconds_bucket[5m])) * 1000'
        
        try:
            result = self.prometheus.query_range(query, start_time, end_time, step)
            formatted_result = self.prometheus.format_query_result(result)
            
            # Group by operation type if available
            grouped_data = {}
            for item in formatted_result['results']:
                operation = item['metric'].get('operation', 'unknown')
                type_key = item['metric'].get('type', 'unknown')
                key = f"{operation}:{type_key}"
                
                stats = self.prometheus.calculate_statistics([item])
                grouped_data[key] = {
                    'metric_labels': item['metric'],
                    'values': item['values'],
                    'statistics': stats,
                    'unit': 'ms'
                }
            
            return {
                'query': query,
                'metric_type': 'etcd_request_latency_p99_ms',
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'step': step
                },
                'operations': grouped_data,
                'cluster_statistics': formatted_result['statistics'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get etcd latency: {e}")
            raise
    
    def collect_api_request_latency_metrics(self, 
                                   duration_hours: float = 1.0, 
                                   step: str = '1m') -> Dict[str, Any]:
        """Collect comprehensive API latency metrics.
        
        Args:
            duration_hours: Duration in hours to collect data for
            step: Query resolution step
        
        Returns:
            Dictionary containing API latency metrics and analysis
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=duration_hours)
        
        try:
            # Collect all API latency metrics
            # p50_data = self.get_api_request_latency_p50(start_time, end_time, step)
            p95_data = self.get_api_request_latency_p95(start_time, end_time, step)
            p99_data = self.get_api_request_latency_p99(start_time, end_time, step)
            
            # Combine latency data by operation
            combined_latency = {}
            all_operations = set()
            
            # Collect all operations from P50, P95, P99
            # for data in [p50_data, p95_data, p99_data]:
            for data in [p95_data,p99_data]:
                all_operations.update(data.get('operations', {}).keys())
            
            for operation in all_operations:
                combined_latency[operation] = {
                    # 'p50': p50_data.get('operations', {}).get(operation, {}),
                    'p95': p95_data.get('operations', {}).get(operation, {}),
                    'p99': p99_data.get('operations', {}).get(operation, {}),
                    'summary_stats': {}
                }
                
                # Calculate summary statistics across percentiles
                # p50_stats = combined_latency[operation]['p50'].get('statistics', {})
                p95_stats = combined_latency[operation]['p95'].get('statistics', {})
                p99_stats = combined_latency[operation]['p99'].get('statistics', {})
                
                if p99_stats:
                # if p50_stats and p95_stats and p99_stats:
                    combined_latency[operation]['summary_stats'] = {
                        # 'p50_mean_ms': round(p50_stats.get('mean', 0), 6),
                        'p95_mean_ms': round(p95_stats.get('mean', 0), 6),
                        'p99_mean_ms': round(p99_stats.get('mean', 0), 6),
                        # 'p50_max_ms': round(p50_stats.get('max', 0), 6),
                        'p95_max_ms': round(p95_stats.get('max', 0), 6),
                        'p99_max_ms': round(p99_stats.get('max', 0), 6)
                    }
        
            # Get baselines for comparison
            api_baselines = self.config.get_api_baselines()
            
            # Analyze performance against baselines
            performance_analysis = self._analyze_api_performance(
                combined_latency, {}, api_baselines
            )
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'collection_period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'duration_hours': duration_hours,
                    'step': step
                },
                'baselines': api_baselines,
                'performance_analysis': performance_analysis,
                'api_server_latency': {
                    'operations': combined_latency,
                    'cluster_summary': {
                        # 'p50_overall': p50_data.get('cluster_statistics', {}),
                        'p95_overall': p95_data.get('cluster_statistics', {}),
                        'p99_overall': p99_data.get('cluster_statistics', {})
                    }
                },
                'queries_executed': {
                    # 'latency_p50': p50_data['query'],
                    'latency_p95': p95_data['query'],
                    'latency_p99': p99_data['query']
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to collect API latency metrics: {e}")
            raise

    def collect_api_request_rate_metrics(self, 
                                   duration_hours: float = 1.0, 
                                   step: str = '1m') -> Dict[str, Any]:
        """Collect comprehensive API latency metrics.
        
        Args:
            duration_hours: Duration in hours to collect data for
            step: Query resolution step
        
        Returns:
            Dictionary containing API latency metrics and analysis
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=duration_hours)
        
        try:
            # Collect all API latency metrics
            rate_data = self.get_api_request_rate(start_time, end_time, step)
            
            # Combine latency data by operation
            combined_latency = {}
            all_operations = set()
            
            # Process request rate data
            request_rates = {}
            for operation, rate_info in rate_data.get('operations', {}).items():
                request_rates[operation] = {
                    'rate_stats': rate_info.get('statistics', {}),
                    'labels': rate_info.get('metric_labels', {})
                }
            
            # Get baselines for comparison
            api_baselines = self.config.get_api_baselines()
            
            # Analyze performance against baselines
            performance_analysis = self._analyze_api_performance(
                {}, {}, api_baselines
            )
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'collection_period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'duration_hours': duration_hours,
                    'step': step
                },
                'baselines': api_baselines,
                'performance_analysis': performance_analysis,
                'request_rates': request_rates,
                'queries_executed': {
                    'request_rate': rate_data['query']
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to collect API latency metrics: {e}")
            raise
    
    def collect_etcd_latency_metrics(self, 
                                   duration_hours: float = 1.0, 
                                   step: str = '1m') -> Dict[str, Any]:
        """Collect comprehensive API latency metrics.
        
        Args:
            duration_hours: Duration in hours to collect data for
            step: Query resolution step
        
        Returns:
            Dictionary containing API latency metrics and analysis
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=duration_hours)
        
        try:
            etcd_data = self.get_etcd_latency(start_time, end_time, step)
            
            # Combine latency data by operation
            combined_latency = {}
            all_operations = set()
        
            # Process etcd latency data
            etcd_operations = {}
            for operation, etcd_info in etcd_data.get('operations', {}).items():
                etcd_operations[operation] = {
                    'latency_stats': etcd_info.get('statistics', {}),
                    'labels': etcd_info.get('metric_labels', {})
                }
            
            # Get baselines for comparison
            api_baselines = self.config.get_api_baselines()
            
            # Analyze performance against baselines
            performance_analysis = self._analyze_api_performance(
                combined_latency, etcd_operations, api_baselines
            )
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'collection_period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'duration_hours': duration_hours,
                    'step': step
                },
                'baselines': api_baselines,
                'performance_analysis': performance_analysis,
                'etcd_latency': {
                    'operations': etcd_operations,
                    'cluster_summary': etcd_data.get('cluster_statistics', {})
                },
                'queries_executed': {
                    'etcd_latency': etcd_data['query']
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to collect API latency metrics: {e}")
            raise
        

    def _analyze_api_performance(self, 
                                latency_data: Dict[str, Any], 
                                etcd_data: Dict[str, Any],
                                baselines: Dict[str, float]) -> Dict[str, Any]:
        """Analyze API performance against baselines."""
        analysis = {
            'overall_status': 'normal',
            'alerts': [],
            'summary': {
                'total_operations': len(latency_data),
                'high_latency_operations': 0,
                'critical_latency_operations': 0,
                'etcd_operations': len(etcd_data),
                'high_etcd_latency': 0
            }
        }
        
        # Analyze API server latency
        for operation, operation_data in latency_data.items():
            summary_stats = operation_data.get('summary_stats', {})
            
            # Check P50 latency
            # p50_mean = summary_stats.get('p50_mean_ms', 0)
            # if p50_mean > baselines.get('p50', 100):
            #     analysis['alerts'].append({
            #         'type': 'high_p50_latency',
            #         'operation': operation,
            #         'current_ms': p50_mean,
            #         'baseline_ms': baselines.get('p50', 100),
            #         'severity': 'warning' if p50_mean < baselines.get('p50', 100) * 2 else 'critical'
            #     })
            #     analysis['summary']['high_latency_operations'] += 1
            
            # Check P95 latency
            p95_mean = summary_stats.get('p95_mean_ms', 0)
            if p95_mean > baselines.get('p95', 500):
                analysis['alerts'].append({
                    'type': 'high_p95_latency',
                    'operation': operation,
                    'current_ms': round(p95_mean, 6),
                    'baseline_ms': round(baselines.get('p95', 500), 6),
                    'severity': 'warning' if p95_mean < baselines.get('p95', 500) * 2 else 'critical'
                })
                analysis['summary']['high_latency_operations'] += 1
            
            # Check P99 latency
            p99_mean = summary_stats.get('p99_mean_ms', 0)
            if p99_mean > baselines.get('p99', 1000):
                analysis['alerts'].append({
                    'type': 'high_p99_latency',
                    'operation': operation,
                    'current_ms': round(p99_mean, 6),
                    'baseline_ms': round(baselines.get('p99', 1000), 6),
                    'severity': 'critical' if p99_mean > baselines.get('p99', 1000) * 2 else 'warning'
                })
                analysis['summary']['critical_latency_operations'] += 1
        
        # Analyze etcd latency
        for operation, etcd_info in etcd_data.items():
            latency_stats = etcd_info.get('latency_stats', {})
            mean_latency = latency_stats.get('mean', 0)
            
            if mean_latency > baselines.get('etcd_response_time', 10):
                analysis['alerts'].append({
                    'type': 'high_etcd_latency',
                    'operation': operation,
                    'current_ms': round(mean_latency, 6),
                    'baseline_ms': round(baselines.get('etcd_response_time', 10), 6),
                    'severity': 'critical' if mean_latency > baselines.get('etcd_response_time', 10) * 5 else 'warning'
                })
                analysis['summary']['high_etcd_latency'] += 1
        
        # Determine overall status
        if analysis['summary']['critical_latency_operations'] > 0 or analysis['summary']['high_etcd_latency'] > 0:
            analysis['overall_status'] = 'critical'
        elif analysis['summary']['high_latency_operations'] > 0:
            analysis['overall_status'] = 'degraded'
        elif len(analysis['alerts']) == 0:
            analysis['overall_status'] = 'optimal'
        
        return analysis


def get_api_request_latency(duration_hours: float = 1.0, step: str = '1m') -> str:
    """Get API latency metrics and return as JSON string.
    
    Args:
        duration_hours: Duration in hours to collect data for (default: 1 hour)
        step: Query resolution step (default: '1m')
    
    Returns:
        JSON string containing API latency metrics data
    """
    collector = APILatencyCollector()
    metrics_data = collector.collect_api_request_latency_metrics(duration_hours, step)
    return json.dumps(metrics_data, indent=2)

def get_api_request_rate(duration_hours: float = 1.0, step: str = '1m') -> str:
    """Get API latency metrics and return as JSON string.
    
    Args:
        duration_hours: Duration in hours to collect data for (default: 1 hour)
        step: Query resolution step (default: '1m')
    
    Returns:
        JSON string containing API latency metrics data
    """
    collector = APILatencyCollector()
    metrics_data = collector.collect_api_request_rate_metrics(duration_hours, step)
    return json.dumps(metrics_data, indent=2)

def get_etcd_latency(duration_hours: float = 1.0, step: str = '1m') -> str:
    """Get etcd latency metrics and return as JSON string.
    
    Args:
        duration_hours: Duration in hours to collect data for (default: 1 hour)
        step: Query resolution step (default: '1m')
    
    Returns:
        JSON string containing API latency metrics data
    """
    collector = APILatencyCollector()
    metrics_data = collector.collect_etcd_latency_metrics(duration_hours, step)
    return json.dumps(metrics_data, indent=2)

    