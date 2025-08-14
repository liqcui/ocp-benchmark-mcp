"""Disk I/O metrics collection from Prometheus."""
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from tools.ocp_benchmark_prometheus_basequery import prometheus_client
from config.ocp_benchmark_config import config_manager


logger = logging.getLogger(__name__)


class DiskIOCollector:
    """Collects disk I/O metrics from Prometheus."""
    
    def __init__(self):
        self.prometheus = prometheus_client
        self.config = config_manager
    
    def get_disk_read_bytes(self, 
                           start_time: datetime, 
                           end_time: datetime, 
                           step: str = '1m') -> Dict[str, Any]:
        """Get disk read bytes per second."""
        query = self.config.get_metric_query('disk_metrics', 'read_bytes')
        if not query:
            query = 'rate(node_disk_read_bytes_total[5m])'
        
        try:
            result = self.prometheus.query_range(query, start_time, end_time, step)
            formatted_result = self.prometheus.format_query_result(result)
            
            # Group by node and device
            nodes_data = {}
            for item in formatted_result['results']:
                instance = item['metric'].get('instance', 'unknown').split(':')[0]
                device = item['metric'].get('device', 'unknown')
                
                if instance not in nodes_data:
                    nodes_data[instance] = {}
                
                # Convert bytes to MB/s for easier reading
                values_mb = [{'timestamp': v['timestamp'], 'value': v['value'] / (1024 * 1024)} for v in item['values']]
                stats = self.prometheus.calculate_statistics([{'values': [(v['timestamp'], v['value']) for v in values_mb]}])
                
                nodes_data[instance][device] = {
                    'metric_labels': item['metric'],
                    'values': values_mb,
                    'statistics': stats,
                    'unit': 'MB/s'
                }
            
            return {
                'query': query,
                'metric_type': 'disk_read_bytes_per_second',
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
            logger.error(f"Failed to get disk read bytes: {e}")
            raise
    
    def get_disk_write_bytes(self, 
                            start_time: datetime, 
                            end_time: datetime, 
                            step: str = '1m') -> Dict[str, Any]:
        """Get disk write bytes per second."""
        query = self.config.get_metric_query('disk_metrics', 'write_bytes')
        if not query:
            query = 'rate(node_disk_written_bytes_total[5m])'
        
        try:
            result = self.prometheus.query_range(query, start_time, end_time, step)
            formatted_result = self.prometheus.format_query_result(result)
            
            # Group by node and device
            nodes_data = {}
            for item in formatted_result['results']:
                instance = item['metric'].get('instance', 'unknown').split(':')[0]
                device = item['metric'].get('device', 'unknown')
                
                if instance not in nodes_data:
                    nodes_data[instance] = {}
                
                # Convert bytes to MB/s for easier reading
                values_mb = [{'timestamp': v['timestamp'], 'value': v['value'] / (1024 * 1024)} for v in item['values']]
                stats = self.prometheus.calculate_statistics([{'values': [(v['timestamp'], v['value']) for v in values_mb]}])
                
                nodes_data[instance][device] = {
                    'metric_labels': item['metric'],
                    'values': values_mb,
                    'statistics': stats,
                    'unit': 'MB/s'
                }
            
            return {
                'query': query,
                'metric_type': 'disk_write_bytes_per_second',
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
            logger.error(f"Failed to get disk write bytes: {e}")
            raise
    
    def get_disk_read_iops(self, 
                          start_time: datetime, 
                          end_time: datetime, 
                          step: str = '1m') -> Dict[str, Any]:
        """Get disk read IOPS."""
        query = self.config.get_metric_query('disk_metrics', 'read_iops')
        if not query:
            query = 'rate(node_disk_reads_completed_total[5m])'
        
        try:
            result = self.prometheus.query_range(query, start_time, end_time, step)
            formatted_result = self.prometheus.format_query_result(result)
            
            # Group by node and device
            nodes_data = {}
            for item in formatted_result['results']:
                instance = item['metric'].get('instance', 'unknown').split(':')[0]
                device = item['metric'].get('device', 'unknown')
                
                if instance not in nodes_data:
                    nodes_data[instance] = {}
                
                stats = self.prometheus.calculate_statistics([item])
                nodes_data[instance][device] = {
                    'metric_labels': item['metric'],
                    'values': item['values'],
                    'statistics': stats,
                    'unit': 'ops/s'
                }
            
            return {
                'query': query,
                'metric_type': 'disk_read_iops',
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
            logger.error(f"Failed to get disk read IOPS: {e}")
            raise
    
    def get_disk_write_iops(self, 
                           start_time: datetime, 
                           end_time: datetime, 
                           step: str = '1m') -> Dict[str, Any]:
        """Get disk write IOPS."""
        query = self.config.get_metric_query('disk_metrics', 'write_iops')
        if not query:
            query = 'rate(node_disk_writes_completed_total[5m])'
        
        try:
            result = self.prometheus.query_range(query, start_time, end_time, step)
            formatted_result = self.prometheus.format_query_result(result)
            
            # Group by node and device
            nodes_data = {}
            for item in formatted_result['results']:
                instance = item['metric'].get('instance', 'unknown').split(':')[0]
                device = item['metric'].get('device', 'unknown')
                
                if instance not in nodes_data:
                    nodes_data[instance] = {}
                
                stats = self.prometheus.calculate_statistics([item])
                nodes_data[instance][device] = {
                    'metric_labels': item['metric'],
                    'values': item['values'],
                    'statistics': stats,
                    'unit': 'ops/s'
                }
            
            return {
                'query': query,
                'metric_type': 'disk_write_iops',
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
            logger.error(f"Failed to get disk write IOPS: {e}")
            raise
    
    def get_disk_read_latency(self, 
                             start_time: datetime, 
                             end_time: datetime, 
                             step: str = '1m') -> Dict[str, Any]:
        """Get disk read latency in milliseconds."""
        query = self.config.get_metric_query('disk_metrics', 'read_latency')
        if not query:
            query = 'rate(node_disk_read_time_seconds_total[5m]) / rate(node_disk_reads_completed_total[5m]) * 1000'
        
        try:
            result = self.prometheus.query_range(query, start_time, end_time, step)
            formatted_result = self.prometheus.format_query_result(result)
            
            # Group by node and device
            nodes_data = {}
            for item in formatted_result['results']:
                instance = item['metric'].get('instance', 'unknown').split(':')[0]
                device = item['metric'].get('device', 'unknown')
                
                if instance not in nodes_data:
                    nodes_data[instance] = {}
                
                stats = self.prometheus.calculate_statistics([item])
                nodes_data[instance][device] = {
                    'metric_labels': item['metric'],
                    'values': item['values'],
                    'statistics': stats,
                    'unit': 'ms'
                }
            
            return {
                'query': query,
                'metric_type': 'disk_read_latency_ms',
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
            logger.error(f"Failed to get disk read latency: {e}")
            raise
    
    def get_disk_write_latency(self, 
                              start_time: datetime, 
                              end_time: datetime, 
                              step: str = '1m') -> Dict[str, Any]:
        """Get disk write latency in milliseconds."""
        query = self.config.get_metric_query('disk_metrics', 'write_latency')
        if not query:
            query = 'rate(node_disk_write_time_seconds_total[5m]) / rate(node_disk_writes_completed_total[5m]) * 1000'
        
        try:
            result = self.prometheus.query_range(query, start_time, end_time, step)
            formatted_result = self.prometheus.format_query_result(result)
            
            # Group by node and device
            nodes_data = {}
            for item in formatted_result['results']:
                instance = item['metric'].get('instance', 'unknown').split(':')[0]
                device = item['metric'].get('device', 'unknown')
                
                if instance not in nodes_data:
                    nodes_data[instance] = {}
                
                stats = self.prometheus.calculate_statistics([item])
                nodes_data[instance][device] = {
                    'metric_labels': item['metric'],
                    'values': item['values'],
                    'statistics': stats,
                    'unit': 'ms'
                }
            
            return {
                'query': query,
                'metric_type': 'disk_write_latency_ms',
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
            logger.error(f"Failed to get disk write latency: {e}")
            raise
    
    def collect_disk_metrics(self, 
                            duration_hours: float = 1.0, 
                            step: str = '1m') -> Dict[str, Any]:
        """Collect comprehensive disk I/O metrics.
        
        Args:
            duration_hours: Duration in hours to collect data for
            step: Query resolution step
        
        Returns:
            Dictionary containing disk I/O metrics and analysis
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=duration_hours)
        
        try:
            # Collect all disk metrics
            read_bytes_data = self.get_disk_read_bytes(start_time, end_time, step)
            write_bytes_data = self.get_disk_write_bytes(start_time, end_time, step)
            read_iops_data = self.get_disk_read_iops(start_time, end_time, step)
            write_iops_data = self.get_disk_write_iops(start_time, end_time, step)
            read_latency_data = self.get_disk_read_latency(start_time, end_time, step)
            write_latency_data = self.get_disk_write_latency(start_time, end_time, step)
            
            # Combine data by node and device
            combined_data = {}
            all_nodes = set()
            
            # Collect all nodes from different metrics
            for data in [read_bytes_data, write_bytes_data, read_iops_data, 
                        write_iops_data, read_latency_data, write_latency_data]:
                all_nodes.update(data.get('nodes', {}).keys())
            
            for node in all_nodes:
                combined_data[node] = {'devices': {}}
                
                # Get all devices for this node
                all_devices = set()
                for data in [read_bytes_data, write_bytes_data, read_iops_data, 
                           write_iops_data, read_latency_data, write_latency_data]:
                    if node in data.get('nodes', {}):
                        all_devices.update(data['nodes'][node].keys())
                
                for device in all_devices:
                    combined_data[node]['devices'][device] = {
                        'read_throughput': read_bytes_data.get('nodes', {}).get(node, {}).get(device, {}),
                        'write_throughput': write_bytes_data.get('nodes', {}).get(node, {}).get(device, {}),
                        'read_iops': read_iops_data.get('nodes', {}).get(node, {}).get(device, {}),
                        'write_iops': write_iops_data.get('nodes', {}).get(node, {}).get(device, {}),
                        'read_latency': read_latency_data.get('nodes', {}).get(node, {}).get(device, {}),
                        'write_latency': write_latency_data.get('nodes', {}).get(node, {}).get(device, {})
                    }
            
            # Get baselines for comparison
            disk_baselines = self.config.get_disk_baselines()
            
            # Analyze performance against baselines
            performance_analysis = self._analyze_disk_performance(combined_data, disk_baselines)
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'collection_period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'duration_hours': duration_hours,
                    'step': step
                },
                'baselines': disk_baselines,
                'performance_analysis': performance_analysis,
                'nodes': combined_data,
                'queries_executed': {
                    'read_bytes': read_bytes_data['query'],
                    'write_bytes': write_bytes_data['query'],
                    'read_iops': read_iops_data['query'],
                    'write_iops': write_iops_data['query'],
                    'read_latency': read_latency_data['query'],
                    'write_latency': write_latency_data['query']
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to collect disk metrics: {e}")
            raise
    
    def _analyze_disk_performance(self, 
                                 disk_data: Dict[str, Any], 
                                 baselines: Dict[str, float]) -> Dict[str, Any]:
        """Analyze disk performance against baselines."""
        analysis = {
            'overall_status': 'normal',
            'alerts': [],
            'summary': {
                'total_nodes': len(disk_data),
                'total_devices': sum(len(node_data['devices']) for node_data in disk_data.values()),
                'high_latency_devices': 0,
                'low_throughput_devices': 0,
                'low_iops_devices': 0
            }
        }
        
        for node_name, node_data in disk_data.items():
            for device_name, device_data in node_data['devices'].items():
                device_id = f"{node_name}:{device_name}"
                
                # Check read latency
                read_latency_stats = device_data.get('read_latency', {}).get('statistics', {})
                if read_latency_stats.get('mean', 0) > baselines.get('peak_latency', 50):
                    analysis['alerts'].append({
                        'type': 'high_read_latency',
                        'device': device_id,
                        'current': read_latency_stats.get('mean', 0),
                        'baseline': baselines.get('peak_latency', 50),
                        'severity': 'warning'
                    })
                    analysis['summary']['high_latency_devices'] += 1
                
                # Check write latency
                write_latency_stats = device_data.get('write_latency', {}).get('statistics', {})
                if write_latency_stats.get('mean', 0) > baselines.get('peak_latency', 50):
                    analysis['alerts'].append({
                        'type': 'high_write_latency',
                        'device': device_id,
                        'current': write_latency_stats.get('mean', 0),
                        'baseline': baselines.get('peak_latency', 50),
                        'severity': 'warning'
                    })
                    analysis['summary']['high_latency_devices'] += 1
                
                # Check read throughput
                read_throughput_stats = device_data.get('read_throughput', {}).get('statistics', {})
                if read_throughput_stats.get('mean', 0) < baselines.get('read_baseline', 100) and read_throughput_stats.get('mean', 0) > 0:
                    analysis['alerts'].append({
                        'type': 'low_read_throughput',
                        'device': device_id,
                        'current': read_throughput_stats.get('mean', 0),
                        'baseline': baselines.get('read_baseline', 100),
                        'severity': 'info'
                    })
                    analysis['summary']['low_throughput_devices'] += 1
                
                # Check write throughput
                write_throughput_stats = device_data.get('write_throughput', {}).get('statistics', {})
                if write_throughput_stats.get('mean', 0) < baselines.get('write_baseline', 50) and write_throughput_stats.get('mean', 0) > 0:
                    analysis['alerts'].append({
                        'type': 'low_write_throughput',
                        'device': device_id,
                        'current': write_throughput_stats.get('mean', 0),
                        'baseline': baselines.get('write_baseline', 50),
                        'severity': 'info'
                    })
                    analysis['summary']['low_throughput_devices'] += 1
                
                # Check IOPS
                read_iops_stats = device_data.get('read_iops', {}).get('statistics', {})
                write_iops_stats = device_data.get('write_iops', {}).get('statistics', {})
                
                if (read_iops_stats.get('mean', 0) < baselines.get('read_iops', 10000) and 
                    read_iops_stats.get('mean', 0) > 0):
                    analysis['summary']['low_iops_devices'] += 1
                
                if (write_iops_stats.get('mean', 0) < baselines.get('write_iops', 8000) and 
                    write_iops_stats.get('mean', 0) > 0):
                    analysis['summary']['low_iops_devices'] += 1
        
        # Determine overall status
        if analysis['summary']['high_latency_devices'] > 0:
            analysis['overall_status'] = 'degraded'
        
        if len(analysis['alerts']) == 0:
            analysis['overall_status'] = 'optimal'
        
        return analysis


def get_disk_metrics(duration_hours: float = 1.0, step: str = '1m') -> str:
    """Get disk I/O metrics and return as JSON string.
    
    Args:
        duration_hours: Duration in hours to collect data for (default: 1 hour)
        step: Query resolution step (default: '1m')
    
    Returns:
        JSON string containing disk I/O metrics data
    """
    collector = DiskIOCollector()
    metrics_data = collector.collect_disk_metrics(duration_hours, step)
    return json.dumps(metrics_data, indent=2)