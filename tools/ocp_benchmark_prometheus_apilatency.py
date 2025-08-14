#!/usr/bin/env python3
"""Prometheus API Latency Metrics Module"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import requests
import asyncio
from config.ocp_benchmark_config import config
from ocauth.ocp_benchmark_auth import auth

logger = logging.getLogger(__name__)

class APILatencyCollector:
    """Collect API server latency metrics from Prometheus"""
    
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
    
    async def get_api_latency_percentiles(self, duration_hours: int = 1) -> Dict[str, Any]:
        """Get API server latency percentiles (P50, P95, P99)"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=duration_hours)
            
            # Get API latency queries from config
            p50_query = config.get_metric_query('api_latency', 'p50')
            p95_query = config.get_metric_query('api_latency', 'p95')
            p99_query = config.get_metric_query('api_latency', 'p99')
            
            if not all([p50_query, p95_query, p99_query]):
                return {'error': 'API latency queries not properly configured'}
            
            # Execute queries concurrently
            tasks = [
                self._query_prometheus(p50_query, start_time, end_time),
                self._query_prometheus(p95_query, start_time, end_time),
                self._query_prometheus(p99_query, start_time, end_time)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            p50_result, p95_result, p99_result = results
            
            # Process results
            metrics = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'duration_hours': duration_hours,
                'p50_latency': self._process_latency_result(p50_result, 'p50'),
                'p95_latency': self._process_latency_result(p95_result, 'p95'),
                'p99_latency': self._process_latency_result(p99_result, 'p99')
            }
            
            # Add baseline comparison
            baselines = config.get_api_baselines()
            metrics['baseline_comparison'] = self._compare_with_api_baseline(metrics, baselines)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting API latency percentiles: {e}")
            return {'error': str(e)}
    
    def _process_latency_result(self, result: Any, percentile: str) -> Dict[str, Any]:
        """Process Prometheus result for latency metrics"""
        if isinstance(result, Exception) or not result:
            return {'error': f'Failed to query {percentile} latency', 'unit': 'seconds'}
        
        verb_resource_stats = {}
        all_values = []
        
        for series in result['data']['result']:
            verb = series['metric'].get('verb', 'unknown')
            resource = series['metric'].get('resource', 'unknown')
            key = f"{verb}:{resource}"
            
            values = []
            if 'values' in series:
                for value_pair in series['values']:
                    try:
                        # Convert seconds to milliseconds
                        value_ms = float(value_pair[1]) * 1000
                        values.append(value_ms)
                        all_values.append(value_ms)
                    except (ValueError, IndexError):
                        continue
            
            if values:
                verb_resource_stats[key] = self._calculate_stats(values)
        
        overall_stats = self._calculate_stats(all_values)
        
        return {
            'percentile': percentile,
            'unit': 'milliseconds',
            'overall_stats': overall_stats,
            'verb_resource_stats': verb_resource_stats,
            'verb_resource_count': len(verb_resource_stats)
        }
    
    def _compare_with_api_baseline(self, metrics: Dict[str, Any], baselines: Dict[str, float]) -> Dict[str, Any]:
        """Compare API latency metrics with baseline values"""
        try:
            comparison = {
                'baselines': baselines,
                'within_thresholds': True,
                'issues': [],
                'performance_summary': {}
            }
            
            # Check P50 latency
            if 'p50_latency' in metrics and 'overall_stats' in metrics['p50_latency']:
                avg_p50_ms = metrics['p50_latency']['overall_stats'].get('mean', 0)
                comparison['performance_summary']['avg_p50_ms'] = round(avg_p50_ms, 2)
                
                if avg_p50_ms > baselines['p50']:
                    comparison['within_thresholds'] = False
                    comparison['issues'].append({
                        'metric': 'p50_latency',
                        'current_ms': avg_p50_ms,
                        'baseline_ms': baselines['p50'],
                        'status': 'above_baseline'
                    })
            
            # Check P95 latency
            if 'p95_latency' in metrics and 'overall_stats' in metrics['p95_latency']:
                avg_p95_ms = metrics['p95_latency']['overall_stats'].get('mean', 0)
                comparison['performance_summary']['avg_p95_ms'] = round(avg_p95_ms, 2)
                
                if avg_p95_ms > baselines['p95']:
                    comparison['within_thresholds'] = False
                    comparison['issues'].append({
                        'metric': 'p95_latency',
                        'current_ms': avg_p95_ms,
                        'baseline_ms': baselines['p95'],
                        'status': 'above_baseline'
                    })
            
            # Check P99 latency
            if 'p99_latency' in metrics and 'overall_stats' in metrics['p99_latency']:
                avg_p99_ms = metrics['p99_latency']['overall_stats'].get('mean', 0)
                comparison['performance_summary']['avg_p99_ms'] = round(avg_p99_ms, 2)
                
                if avg_p99_ms > baselines['p99']:
                    comparison['within_thresholds'] = False
                    comparison['issues'].append({
                        'metric': 'p99_latency',
                        'current_ms': avg_p99_ms,
                        'baseline_ms': baselines['p99'],
                        'status': 'above_baseline'
                    })
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing with API baseline: {e}")
            return {'error': str(e)}
    
    async def get_api_latency_by_verb_resource(self, duration_hours: int = 1) -> Dict[str, Any]:
        """Get API latency broken down by HTTP verb and resource type"""
        try:
            latency_metrics = await self.get_api_latency_percentiles(duration_hours)
            
            if 'error' in latency_metrics:
                return latency_metrics
            
            # Aggregate data by verb and resource
            verb_resource_summary = {}
            
            # Process each percentile
            for percentile in ['p50_latency', 'p95_latency', 'p99_latency']:
                if percentile in latency_metrics:
                    verb_resource_stats = latency_metrics[percentile].get('verb_resource_stats', {})
                    
                    for verb_resource_key, stats in verb_resource_stats.items():
                        if verb_resource_key not in verb_resource_summary:
                            verb, resource = verb_resource_key.split(':', 1)
                            verb_resource_summary[verb_resource_key] = {
                                'verb': verb,
                                'resource': resource,
                                'verb_resource_key': verb_resource_key
                            }
                        
                        percentile_name = percentile.replace('_latency', '')
                        verb_resource_summary[verb_resource_key][f'{percentile_name}_stats'] = stats
                        verb_resource_summary[verb_resource_key][f'{percentile_name}_avg_ms'] = round(stats.get('mean', 0), 2)
            
            # Sort by P95 latency (most impactful)
            sorted_entries = sorted(
                verb_resource_summary.values(),
                key=lambda x: x.get('p95_avg_ms', 0),
                reverse=True
            )
            
            return {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'duration_hours': duration_hours,
                'verb_resource_breakdown': sorted_entries,
                'total_verb_resource_combinations': len(sorted_entries)
            }
            
        except Exception as e:
            logger.error(f"Error getting API latency by verb/resource: {e}")
            return {'error': str(e)}
    
    async def get_slow_api_requests(self, duration_hours: int = 1, threshold_ms: float = 1000.0) -> Dict[str, Any]:
        """Identify slow API requests above threshold"""
        try:
            verb_resource_data = await self.get_api_latency_by_verb_resource(duration_hours)
            
            if 'error' in verb_resource_data:
                return verb_resource_data
            
            slow_requests = []
            
            for entry in verb_resource_data.get('verb_resource_breakdown', []):
                # Check if any percentile exceeds threshold
                p50_ms = entry.get('p50_avg_ms', 0)
                p95_ms = entry.get('p95_avg_ms', 0)
                p99_ms = entry.get('p99_avg_ms', 0)
                
                if p50_ms > threshold_ms or p95_ms > threshold_ms or p99_ms > threshold_ms:
                    slow_entry = entry.copy()
                    slow_entry['exceeded_threshold'] = []
                    
                    if p50_ms > threshold_ms:
                        slow_entry['exceeded_threshold'].append('p50')
                    if p95_ms > threshold_ms:
                        slow_entry['exceeded_threshold'].append('p95')
                    if p99_ms > threshold_ms:
                        slow_entry['exceeded_threshold'].append('p99')
                    
                    slow_requests.append(slow_entry)
            
            return {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'duration_hours': duration_hours,
                'threshold_ms': threshold_ms,
                'slow_requests': slow_requests,
                'slow_request_count': len(slow_requests),
                'total_analyzed': len(verb_resource_data.get('verb_resource_breakdown', []))
            }
            
        except Exception as e:
            logger.error(f"Error getting slow API requests: {e}")
            return {'error': str(e)}
    
    async def get_etcd_latency_metrics(self, duration_hours: int = 1) -> Dict[str, Any]:
        """Get etcd response time metrics"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=duration_hours)
            
            # Get etcd response time query from config
            etcd_query = config.get_metric_query('cluster_metrics', 'etcd_response_time')
            
            if not etcd_query:
                return {'error': 'etcd response time query not configured'}
            
            result = await self._query_prometheus(etcd_query, start_time, end_time)
            
            if not result:
                return {'error': 'Failed to query Prometheus for etcd latency'}
            
            # Process etcd latency results
            all_values = []
            operation_stats = {}
            
            for series in result['data']['result']:
                operation = series['metric'].get('operation', 'unknown')
                
                values = []
                if 'values' in series:
                    for value_pair in series['values']:
                        try:
                            # Convert seconds to milliseconds
                            value_ms = float(value_pair[1]) * 1000
                            values.append(value_ms)
                            all_values.append(value_ms)
                        except (ValueError, IndexError):
                            continue
                
                if values:
                    operation_stats[operation] = self._calculate_stats(values)
            
            overall_stats = self._calculate_stats(all_values)
            
            # Compare with baseline
            baselines = config.get_baseline_value('etcd.response', 'time_ms', 10.0)
            within_baseline = overall_stats.get('mean', 0) <= baselines
            
            return {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'duration_hours': duration_hours,
                'unit': 'milliseconds',
                'overall_stats': overall_stats,
                'operation_stats': operation_stats,
                'baseline_comparison': {
                    'baseline_ms': baselines,
                    'current_avg_ms': round(overall_stats.get('mean', 0), 2),
                    'within_baseline': within_baseline,
                    'deviation_ms': round(overall_stats.get('mean', 0) - baselines, 2) if not within_baseline else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting etcd latency metrics: {e}")
            return {'error': str(e)}
    
    async def get_combined_api_metrics(self, duration_hours: int = 1) -> Dict[str, Any]:
        """Get combined API server and etcd latency metrics"""
        try:
            api_task = self.get_api_latency_percentiles(duration_hours)
            etcd_task = self.get_etcd_latency_metrics(duration_hours)
            
            api_result, etcd_result = await asyncio.gather(api_task, etcd_task, return_exceptions=True)
            
            return {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'duration_hours': duration_hours,
                'api_server_latency': api_result if not isinstance(api_result, Exception) else {'error': str(api_result)},
                'etcd_latency': etcd_result if not isinstance(etcd_result, Exception) else {'error': str(etcd_result)},
                'summary': self._generate_api_summary(api_result, etcd_result)
            }
            
        except Exception as e:
            logger.error(f"Error getting combined API metrics: {e}")
            return {'error': str(e)}
    
    def _generate_api_summary(self, api_metrics: Any, etcd_metrics: Any) -> Dict[str, Any]:
        """Generate summary of API performance"""
        try:
            summary = {
                'overall_status': 'healthy',
                'issues_detected': [],
                'performance_indicators': {}
            }
            
            # Check API server performance
            if (not isinstance(api_metrics, Exception) and 
                'baseline_comparison' in api_metrics):
                
                comparison = api_metrics['baseline_comparison']
                if not comparison.get('within_thresholds', True):
                    summary['overall_status'] = 'degraded'
                    summary['issues_detected'].extend([
                        f"API {issue['metric']}: {issue['status']}" 
                        for issue in comparison.get('issues', [])
                    ])
                
                # Extract API performance indicators
                perf_summary = comparison.get('performance_summary', {})
                summary['performance_indicators'].update(perf_summary)
            
            # Check etcd performance
            if (not isinstance(etcd_metrics, Exception) and 
                'baseline_comparison' in etcd_metrics):
                
                etcd_comparison = etcd_metrics['baseline_comparison']
                if not etcd_comparison.get('within_baseline', True):
                    summary['overall_status'] = 'degraded'
                    summary['issues_detected'].append('etcd_latency: above_baseline')
                
                summary['performance_indicators']['etcd_avg_ms'] = etcd_comparison.get('current_avg_ms', 0)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating API summary: {e}")
            return {'error': str(e)}

# Global API latency collector instance
api_latency_collector = APILatencyCollector()

async def get_api_latency_json(duration_hours: int = 1) -> str:
    """Get API latency metrics as JSON string"""
    metrics = await api_latency_collector.get_combined_api_metrics(duration_hours)
    return json.dumps(metrics, indent=2)

async def get_slow_api_requests_json(duration_hours: int = 1, threshold_ms: float = 1000.0) -> str:
    """Get slow API requests as JSON string"""
    slow_requests = await api_latency_collector.get_slow_api_requests(duration_hours, threshold_ms)
    return json.dumps(slow_requests, indent=2)