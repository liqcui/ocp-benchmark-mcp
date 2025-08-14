#!/usr/bin/env python3
"""Prometheus Pod Usage Metrics Module"""

import json
import logging
import re
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import requests
import asyncio
from config.ocp_benchmark_config import config
from ocauth.ocp_benchmark_auth import auth

logger = logging.getLogger(__name__)

class PodUsageCollector:
    """Collect pod usage metrics from Prometheus"""
    
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
    
    def _match_pod_pattern(self, pod_name: str, patterns: List[str]) -> bool:
        """Check if pod name matches any of the given patterns"""
        for pattern in patterns:
            if re.search(pattern, pod_name):
                return True
        return False
    
    async def get_pods_by_labels(self, label_selectors: List[str]) -> List[str]:
        """Get pod names by label selectors using Kubernetes API"""
        try:
            from kubernetes import client
            v1 = client.CoreV1Api(auth.get_kube_client())
            
            all_pod_names = set()
            
            for label_selector in label_selectors:
                try:
                    pods = v1.list_pod_for_all_namespaces(label_selector=label_selector)
                    for pod in pods.items:
                        all_pod_names.add(pod.metadata.name)
                except Exception as e:
                    logger.warning(f"Error querying pods with label selector '{label_selector}': {e}")
                    continue
            
            return list(all_pod_names)
            
        except Exception as e:
            logger.error(f"Error getting pods by labels: {e}")
            return []
    
    async def get_pod_cpu_usage(self, pod_patterns: Optional[List[str]] = None, 
                               label_selectors: Optional[List[str]] = None, 
                               duration_hours: int = 1) -> Dict[str, Any]:
        """Get CPU usage statistics for pods"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=duration_hours)
            
            # Get CPU usage query from config
            cpu_query = config.get_metric_query('pod_metrics', 'cpu_usage')
            if not cpu_query:
                logger.error("Pod CPU usage query not found in config")
                return {'error': 'Pod CPU usage query not configured'}
            
            result = await self._query_prometheus(cpu_query, start_time, end_time)
            if not result:
                return {'error': 'Failed to query Prometheus for pod CPU usage'}
            
            # Get pod names if label selectors are provided
            target_pods = []
            if label_selectors:
                target_pods = await self.get_pods_by_labels(label_selectors)
            
            # Process results by pod
            pod_stats = {}
            
            for series in result['data']['result']:
                pod_name = series['metric'].get('pod', 'unknown')
                namespace = series['metric'].get('namespace', 'unknown')
                pod_key = f"{namespace}/{pod_name}"
                
                # Filter pods based on patterns or labels
                include_pod = True
                if pod_patterns and not self._match_pod_pattern(pod_name, pod_patterns):
                    include_pod = False
                if label_selectors and target_pods and pod_name not in target_pods:
                    include_pod = False
                
                if not include_pod:
                    continue
                
                values = []
                if 'values' in series:
                    for value_pair in series['values']:
                        try:
                            values.append(float(value_pair[1]))
                        except (ValueError, IndexError):
                            continue
                
                if values:
                    pod_stats[pod_key] = {
                        'pod_name': pod_name,
                        'namespace': namespace,
                        'stats': self._calculate_stats(values)
                    }
            
            # Calculate overall statistics
            all_values = []
            for pod_data in pod_stats.values():
                # Get raw values for overall calculation
                for series in result['data']['result']:
                    if (series['metric'].get('pod') == pod_data['pod_name'] and 
                        series['metric'].get('namespace') == pod_data['namespace']):
                        for value_pair in series.get('values', []):
                            try:
                                all_values.append(float(value_pair[1]))
                            except (ValueError, IndexError):
                                continue
            
            overall_stats = self._calculate_stats(all_values)
            
            return {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'duration_hours': duration_hours,
                'metric': 'pod_cpu_usage_percentage',
                'filters': {
                    'pod_patterns': pod_patterns or [],
                    'label_selectors': label_selectors or []
                },
                'overall_stats': overall_stats,
                'pod_count': len(pod_stats),
                'pod_stats': pod_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting pod CPU usage: {e}")
            return {'error': str(e)}
    
    async def get_pod_memory_usage(self, pod_patterns: Optional[List[str]] = None, 
                                  label_selectors: Optional[List[str]] = None, 
                                  duration_hours: int = 1) -> Dict[str, Any]:
        """Get memory usage statistics for pods"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=duration_hours)
            
            # Get memory usage query from config
            memory_query = config.get_metric_query('pod_metrics', 'memory_usage')
            if not memory_query:
                logger.error("Pod memory usage query not found in config")
                return {'error': 'Pod memory usage query not configured'}
            
            result = await self._query_prometheus(memory_query, start_time, end_time)
            if not result:
                return {'error': 'Failed to query Prometheus for pod memory usage'}
            
            # Get pod names if label selectors are provided
            target_pods = []
            if label_selectors:
                target_pods = await self.get_pods_by_labels(label_selectors)
            
            # Process results by pod
            pod_stats = {}
            
            for series in result['data']['result']:
                pod_name = series['metric'].get('pod', 'unknown')
                namespace = series['metric'].get('namespace', 'unknown')
                pod_key = f"{namespace}/{pod_name}"
                
                # Filter pods based on patterns or labels
                include_pod = True
                if pod_patterns and not self._match_pod_pattern(pod_name, pod_patterns):
                    include_pod = False
                if label_selectors and target_pods and pod_name not in target_pods:
                    include_pod = False
                
                if not include_pod:
                    continue
                
                values = []
                if 'values' in series:
                    for value_pair in series['values']:
                        try:
                            # Convert bytes to MB
                            memory_mb = float(value_pair[1]) / (1024 * 1024)
                            values.append(memory_mb)
                        except (ValueError, IndexError):
                            continue
                
                if values:
                    pod_stats[pod_key] = {
                        'pod_name': pod_name,
                        'namespace': namespace,
                        'stats': self._calculate_stats(values)
                    }
            
            # Calculate overall statistics
            all_values = []
            for pod_data in pod_stats.values():
                # Get raw values for overall calculation
                for series in result['data']['result']:
                    if (series['metric'].get('pod') == pod_data['pod_name'] and 
                        series['metric'].get('namespace') == pod_data['namespace']):
                        for value_pair in series.get('values', []):
                            try:
                                memory_mb = float(value_pair[1]) / (1024 * 1024)
                                all_values.append(memory_mb)
                            except (ValueError, IndexError):
                                continue
            
            overall_stats = self._calculate_stats(all_values)
            
            return {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'duration_hours': duration_hours,
                'metric': 'pod_memory_usage_mb',
                'filters': {
                    'pod_patterns': pod_patterns or [],
                    'label_selectors': label_selectors or []
                },
                'overall_stats': overall_stats,
                'pod_count': len(pod_stats),
                'pod_stats': pod_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting pod memory usage: {e}")
            return {'error': str(e)}
    
    async def get_combined_pod_usage(self, pod_patterns: Optional[List[str]] = None, 
                                    label_selectors: Optional[List[str]] = None, 
                                    duration_hours: int = 1) -> Dict[str, Any]:
        """Get combined CPU and memory usage statistics for pods"""
        try:
            cpu_task = self.get_pod_cpu_usage(pod_patterns, label_selectors, duration_hours)
            memory_task = self.get_pod_memory_usage(pod_patterns, label_selectors, duration_hours)
            
            cpu_result, memory_result = await asyncio.gather(cpu_task, memory_task)
            
            return {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'duration_hours': duration_hours,
                'filters': {
                    'pod_patterns': pod_patterns or [],
                    'label_selectors': label_selectors or []
                },
                'cpu_usage': cpu_result,
                'memory_usage': memory_result
            }
            
        except Exception as e:
            logger.error(f"Error getting combined pod usage: {e}")
            return {'error': str(e)}
    
    async def get_top_resource_consuming_pods(self, duration_hours: int = 1, 
                                             top_n: int = 10) -> Dict[str, Any]:
        """Get top resource consuming pods"""
        try:
            # Get all pod usage data
            combined_usage = await self.get_combined_pod_usage(duration_hours=duration_hours)
            
            if 'error' in combined_usage:
                return combined_usage
            
            # Combine CPU and memory data
            pods_data = {}
            
            # Process CPU data
            if 'cpu_usage' in combined_usage and 'pod_stats' in combined_usage['cpu_usage']:
                for pod_key, pod_data in combined_usage['cpu_usage']['pod_stats'].items():
                    if pod_key not in pods_data:
                        pods_data[pod_key] = {
                            'pod_name': pod_data['pod_name'],
                            'namespace': pod_data['namespace']
                        }
                    pods_data[pod_key]['cpu_stats'] = pod_data['stats']
            
            # Process memory data
            if 'memory_usage' in combined_usage and 'pod_stats' in combined_usage['memory_usage']:
                for pod_key, pod_data in combined_usage['memory_usage']['pod_stats'].items():
                    if pod_key not in pods_data:
                        pods_data[pod_key] = {
                            'pod_name': pod_data['pod_name'],
                            'namespace': pod_data['namespace']
                        }
                    pods_data[pod_key]['memory_stats'] = pod_data['stats']
            
            # Sort by CPU usage (mean) and get top N
            cpu_sorted = sorted(
                pods_data.items(),
                key=lambda x: x[1].get('cpu_stats', {}).get('mean', 0),
                reverse=True
            )[:top_n]
            
            # Sort by memory usage (mean) and get top N
            memory_sorted = sorted(
                pods_data.items(),
                key=lambda x: x[1].get('memory_stats', {}).get('mean', 0),
                reverse=True
            )[:top_n]
            
            return {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'duration_hours': duration_hours,
                'top_cpu_consumers': [{'pod_key': k, **v} for k, v in cpu_sorted],
                'top_memory_consumers': [{'pod_key': k, **v} for k, v in memory_sorted],
                'total_pods_analyzed': len(pods_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting top resource consuming pods: {e}")
            return {'error': str(e)}

# Global pod usage collector instance
pod_usage_collector = PodUsageCollector()

async def get_pod_usage_json(pod_patterns: Optional[List[str]] = None, 
                           label_selectors: Optional[List[str]] = None, 
                           duration_hours: int = 1) -> str:
    """Get pod usage as JSON string"""
    usage = await pod_usage_collector.get_combined_pod_usage(pod_patterns, label_selectors, duration_hours)
    return json.dumps(usage, indent=2)

async def get_top_pods_json(duration_hours: int = 1, top_n: int = 10) -> str:
    """Get top resource consuming pods as JSON string"""
    top_pods = await pod_usage_collector.get_top_resource_consuming_pods(duration_hours, top_n)
    return json.dumps(top_pods, indent=2)