#!/usr/bin/env python3
"""Prometheus Disk I/O Metrics Module"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import requests
import asyncio
from config.ocp_benchmark_config import config
from ocauth.ocp_benchmark_auth import auth

logger = logging.getLogger(__name__)

class DiskIOCollector:
    """Collect disk I/O metrics from Prometheus"""
    
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
    
    async def get_disk_read_metrics(self, duration_hours: int = 1) -> Dict[str, Any]:
        """Get disk read metrics (bytes and IOPS)"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=duration_hours)
            
            # Get disk read queries from config
            read_bytes_query = config.get_metric_query('disk_io', 'read_bytes')
            read_iops_query = config.get_metric_query('disk_io', 'read_iops')
            read_latency_query = config.get_metric_query('disk_io', 'read_latency')
            
            if not all([read_bytes_query, read_iops_query, read_latency_query]):
                return {'error': 'Disk read queries not properly configured'}
            
            # Execute queries concurrently
            tasks = [
                self._query_prometheus(read_bytes_query, start_time, end_time),
                self._query_prometheus(read_iops_query, start_time, end_time),
                self._query_prometheus(read_latency_query, start_time, end_time)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            bytes_result, iops_result, latency_result = results
            
            # Process results
            metrics = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'duration_hours': duration_hours,
                'read_bytes': self._process_disk_metric_result(bytes_result, 'bytes_per_second'),
                'read_iops': self._process_disk_metric_result(iops_result, 'operations_per_second'),
                'read_latency': self._process_disk_metric_result(latency_result, 'seconds', convert_to_ms=True)
            }
            
            # Add baseline comparison
            baselines = config.get_disk_baselines()
            metrics['baseline_comparison'] = self._compare_with_disk_baseline(metrics, baselines, 'read')
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting disk read metrics: {e}")
            return {'error': str(e)}
    
    async def get_disk_write_metrics(self, duration_hours: int = 1) -> Dict[str, Any]:
        """Get disk write metrics (bytes and IOPS)"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=duration_hours)
            
            # Get disk write queries from config
            write_bytes_query = config.get_metric_query('disk_io', 'write_bytes')
            write_iops_query = config.get_metric_query('disk_io', 'write_iops')
            write_latency_query = config.get_metric_query('disk_io', 'write_latency')
            
            if not all([write_bytes_query, write_iops_query, write_latency_query]):
                return {'error': 'Disk write queries not properly configured'}
            
            # Execute queries concurrently
            tasks = [
                self._query_prometheus(write_bytes_query, start_time, end_time),
                self._query_prometheus(write_iops_query, start_time, end_time),
                self._query_prometheus(write_latency_query, start_time, end_time)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            bytes_result, iops_result, latency_result = results
            
            # Process results
            metrics = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'duration_hours': duration_hours,
                'write_bytes': self._process_disk_metric_result(bytes_result, 'bytes_per_second'),
                'write_iops': self._process_disk_metric_result(iops_result, 'operations_per_second'),
                'write_latency': self._process_disk_metric_result(latency_result, 'seconds', convert_to_ms=True)
            }
            
            # Add baseline comparison
            baselines = config.get_disk_baselines()
            metrics['baseline_comparison'] = self._compare_with_disk_baseline(metrics, baselines, 'write')
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting disk write metrics: {e}")
            return {'error': str(e)}
    
    def _process_disk_metric_result(self, result: Any, unit: str, convert_to_ms: bool = False) -> Dict[str, Any]:
        """Process Prometheus result for disk metrics"""
        if isinstance(result, Exception) or not result:
            return {'error': 'Failed to query metric', 'unit': unit}
        
        device_stats = {}
        all_values = []
        
        for series in result['data']['result']:
            device = series['metric'].get('device', 'unknown')
            instance = series['metric'].get('instance', 'unknown')
            device_key = f"{instance}:{device}"
            
            values = []
            if 'values' in series:
                for value_pair in series['values']:
                    try:
                        value = float(value_pair[1])
                        if convert_to_ms:
                            value = value * 1000  # Convert seconds to milliseconds
                        values.append(value)
                        all_values.append(value)
                    except (ValueError, IndexError):
                        continue
            
            if values:
                device_stats[device_key] = self._calculate_stats(values)
        
        overall_stats = self._calculate_stats(all_values)
        
        return {
            'unit': 'milliseconds' if convert_to_ms else unit,
            'overall_stats': overall_stats,
            'device_stats': device_stats,
            'device_count': len(device_stats)
        }
    
    def _compare_with_disk_baseline(self, metrics: Dict[str, Any], baselines: Dict[str, float], 
                                   operation_type: str) -> Dict[str, Any]:
        """Compare disk metrics with baseline values"""
        try:
            comparison = {
                'operation_type': operation_type,
                'baselines': baselines,
                'within_thresholds': True,
                'issues': []
            }
            
            if operation_type == 'read':
                # Check read throughput (convert bytes/sec to MB/sec)
                if 'read_bytes' in metrics and 'overall_stats' in metrics['read_bytes']:
                    avg_bytes_per_sec = metrics['read_bytes']['overall_stats'].get('mean', 0)
                    avg_mb_per_sec = avg_bytes_per_sec / (1024 * 1024)
                    
                    if avg_mb_per_sec < baselines['read_baseline']:
                        comparison['within_thresholds'] = False
                        comparison['issues'].append({
                            'metric': 'read_throughput',
                            'current_mb_per_sec': avg_mb_per_sec,
                            'baseline_mb_per_sec': baselines['read_baseline'],
                            'status': 'below_baseline'
                        })
                
                # Check read IOPS
                if 'read_iops' in metrics and 'overall_stats' in metrics['read_iops']:
                    avg_iops = metrics['read_iops']['overall_stats'].get('mean', 0)
                    
                    if avg_iops < baselines['read_iops']:
                        comparison['within_thresholds'] = False
                        comparison['issues'].append({
                            'metric': 'read_iops',
                            'current_iops': avg_iops,
                            'baseline_iops': baselines['read_iops'],
                            'status': 'below_baseline'
                        })
                
                # Check read latency
                if 'read_latency' in metrics and 'overall_stats' in metrics['read_latency']:
                    avg_latency_ms = metrics['read_latency']['overall_stats'].get('mean', 0)
                    
                    if avg_latency_ms > baselines['read_latency_ms']:
                        comparison['within_thresholds'] = False
                        comparison['issues'].append({
                            'metric': 'read_latency',
                            'current_latency_ms': avg_latency_ms,
                            'baseline_latency_ms': baselines['read_latency_ms'],
                            'status': 'above_baseline'
                        })
            
            elif operation_type == 'write':
                # Check write throughput
                if 'write_bytes' in metrics and 'overall_stats' in metrics['write_bytes']:
                    avg_bytes_per_sec = metrics['write_bytes']['overall_stats'].get('mean', 0)
                    avg_mb_per_sec = avg_bytes_per_sec / (1024 * 1024)
                    
                    if avg_mb_per_sec < baselines['write_baseline']:
                        comparison['within_thresholds'] = False
                        comparison['issues'].append({
                            'metric': 'write_throughput',
                            'current_mb_per_sec': avg_mb_per_sec,
                            'baseline_mb_per_sec': baselines['write_baseline'],
                            'status': 'below_baseline'
                        })
                
                # Check write IOPS
                if 'write_iops' in metrics and 'overall_stats' in metrics['write_iops']:
                    avg_iops = metrics['write_iops']['overall_stats'].get('mean', 0)
                    
                    if avg_iops < baselines['write_iops']:
                        comparison['within_thresholds'] = False
                        comparison['issues'].append({
                            'metric': 'write_iops',
                            'current_iops': avg_iops,
                            'baseline_iops': baselines['write_iops'],
                            'status': 'below_baseline'
                        })
                
                # Check write latency
                if 'write_latency' in metrics and 'overall_stats' in metrics['write_latency']:
                    avg_latency_ms = metrics['write_latency']['overall_stats'].get('mean', 0)
                    
                    if avg_latency_ms > baselines.get('write_latency_ms', baselines.get('write_latency_ms', float('inf'))):
                        comparison['within_thresholds'] = False
                        comparison['issues'].append({
                            'metric': 'write_latency',
                            'current_latency_ms': avg_latency_ms,
                            'baseline_latency_ms': baselines.get('write_latency_ms', 0),
                            'status': 'above_baseline'
                        })
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing with disk baseline: {e}")
            return {'error': str(e)}
    
    async def get_combined_disk_metrics(self, duration_hours: int = 1) -> Dict[str, Any]:
        """Get combined disk read and write metrics"""
        try:
            read_task = self.get_disk_read_metrics(duration_hours)
            write_task = self.get_disk_write_metrics(duration_hours)
            
            read_result, write_result = await asyncio.gather(read_task, write_task)
            
            return {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'duration_hours': duration_hours,
                'disk_read_metrics': read_result,
                'disk_write_metrics': write_result,
                'summary': self._generate_disk_summary(read_result, write_result)
            }
            
        except Exception as e:
            logger.error(f"Error getting combined disk metrics: {e}")
            return {'error': str(e)}
    
    def _generate_disk_summary(self, read_metrics: Dict[str, Any], write_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of disk I/O performance"""
        try:
            summary = {
                'overall_status': 'healthy',
                'issues_detected': [],
                'performance_indicators': {}
            }
            
            # Check read performance
            if 'baseline_comparison' in read_metrics:
                read_comparison = read_metrics['baseline_comparison']
                if not read_comparison.get('within_thresholds', True):
                    summary['overall_status'] = 'degraded'
                    summary['issues_detected'].extend([
                        f"Read {issue['metric']}: {issue['status']}" 
                        for issue in read_comparison.get('issues', [])
                    ])
            
            # Check write performance
            if 'baseline_comparison' in write_metrics:
                write_comparison = write_metrics['baseline_comparison']
                if not write_comparison.get('within_thresholds', True):
                    summary['overall_status'] = 'degraded'
                    summary['issues_detected'].extend([
                        f"Write {issue['metric']}: {issue['status']}" 
                        for issue in write_comparison.get('issues', [])
                    ])
            
            # Extract key performance indicators
            if 'read_bytes' in read_metrics and 'overall_stats' in read_metrics['read_bytes']:
                read_mbps = read_metrics['read_bytes']['overall_stats'].get('mean', 0) / (1024 * 1024)
                summary['performance_indicators']['avg_read_mbps'] = round(read_mbps, 2)
            
            if 'write_bytes' in write_metrics and 'overall_stats' in write_metrics['write_bytes']:
                write_mbps = write_metrics['write_bytes']['overall_stats'].get('mean', 0) / (1024 * 1024)
                summary['performance_indicators']['avg_write_mbps'] = round(write_mbps, 2)
            
            if 'read_iops' in read_metrics and 'overall_stats' in read_metrics['read_iops']:
                summary['performance_indicators']['avg_read_iops'] = round(
                    read_metrics['read_iops']['overall_stats'].get('mean', 0), 2
                )
            
            if 'write_iops' in write_metrics and 'overall_stats' in write_metrics['write_iops']:
                summary['performance_indicators']['avg_write_iops'] = round(
                    write_metrics['write_iops']['overall_stats'].get('mean', 0), 2
                )
            
            if 'read_latency' in read_metrics and 'overall_stats' in read_metrics['read_latency']:
                summary['performance_indicators']['avg_read_latency_ms'] = round(
                    read_metrics['read_latency']['overall_stats'].get('mean', 0), 2
                )
            
            if 'write_latency' in write_metrics and 'overall_stats' in write_metrics['write_latency']:
                summary['performance_indicators']['avg_write_latency_ms'] = round(
                    write_metrics['write_latency']['overall_stats'].get('mean', 0), 2
                )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating disk summary: {e}")
            return {'error': str(e)}
    
    async def get_disk_utilization_by_device(self, duration_hours: int = 1) -> Dict[str, Any]:
        """Get disk utilization broken down by device"""
        try:
            combined_metrics = await self.get_combined_disk_metrics(duration_hours)
            
            if 'error' in combined_metrics:
                return combined_metrics
            
            devices_summary = {}
            
            # Process read metrics by device
            read_devices = combined_metrics.get('disk_read_metrics', {}).get('read_bytes', {}).get('device_stats', {})
            for device_key, stats in read_devices.items():
                if device_key not in devices_summary:
                    devices_summary[device_key] = {'device_key': device_key}
                
                devices_summary[device_key]['read_stats'] = stats
                devices_summary[device_key]['read_mbps'] = round(stats.get('mean', 0) / (1024 * 1024), 2)
            
            # Process write metrics by device
            write_devices = combined_metrics.get('disk_write_metrics', {}).get('write_bytes', {}).get('device_stats', {})
            for device_key, stats in write_devices.items():
                if device_key not in devices_summary:
                    devices_summary[device_key] = {'device_key': device_key}
                
                devices_summary[device_key]['write_stats'] = stats
                devices_summary[device_key]['write_mbps'] = round(stats.get('mean', 0) / (1024 * 1024), 2)
            
            # Add IOPS data
            read_iops_devices = combined_metrics.get('disk_read_metrics', {}).get('read_iops', {}).get('device_stats', {})
            for device_key, stats in read_iops_devices.items():
                if device_key in devices_summary:
                    devices_summary[device_key]['read_iops'] = round(stats.get('mean', 0), 2)
            
            write_iops_devices = combined_metrics.get('disk_write_metrics', {}).get('write_iops', {}).get('device_stats', {})
            for device_key, stats in write_iops_devices.items():
                if device_key in devices_summary:
                    devices_summary[device_key]['write_iops'] = round(stats.get('mean', 0), 2)
            
            # Add latency data
            read_latency_devices = combined_metrics.get('disk_read_metrics', {}).get('read_latency', {}).get('device_stats', {})
            for device_key, stats in read_latency_devices.items():
                if device_key in devices_summary:
                    devices_summary[device_key]['read_latency_ms'] = round(stats.get('mean', 0), 2)
            
            write_latency_devices = combined_metrics.get('disk_write_metrics', {}).get('write_latency', {}).get('device_stats', {})
            for device_key, stats in write_latency_devices.items():
                if device_key in devices_summary:
                    devices_summary[device_key]['write_latency_ms'] = round(stats.get('mean', 0), 2)
            
            return {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'duration_hours': duration_hours,
                'devices': list(devices_summary.values()),
                'device_count': len(devices_summary)
            }
            
        except Exception as e:
            logger.error(f"Error getting disk utilization by device: {e}")
            return {'error': str(e)}

# Global disk I/O collector instance
disk_io_collector = DiskIOCollector()

async def get_disk_metrics_json(duration_hours: int = 1) -> str:
    """Get disk I/O metrics as JSON string"""
    metrics = await disk_io_collector.get_combined_disk_metrics(duration_hours)
    return json.dumps(metrics, indent=2)

async def get_disk_utilization_json(duration_hours: int = 1) -> str:
    """Get disk utilization by device as JSON string"""
    utilization = await disk_io_collector.get_disk_utilization_by_device(duration_hours)
    return json.dumps(utilization, indent=2)