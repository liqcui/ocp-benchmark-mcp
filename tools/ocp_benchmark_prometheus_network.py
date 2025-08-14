"""Network metrics collection from Prometheus."""
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from tools.ocp_benchmark_prometheus_basequery import prometheus_client
from config.ocp_benchmark_config import config_manager


logger = logging.getLogger(__name__)


class NetworkMetricsCollector:
    """Collects network metrics from Prometheus."""
    
    def __init__(self):
        self.prometheus = prometheus_client
        self.config = config_manager
    
    def get_network_rx_bytes(self, 
                            start_time: datetime, 
                            end_time: datetime, 
                            step: str = '1m') -> Dict[str, Any]:
        """Get network receive bytes per second."""
        query = self.config.get_metric_query('network_metrics', 'rx_bytes')
        if not query:
            query = 'rate(node_network_receive_bytes_total{device!="lo"}[5m])'
        
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
                'metric_type': 'network_rx_bytes_per_second',
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
            logger.error(f"Failed to get network RX bytes: {e}")
            raise
    
    def get_network_tx_bytes(self, 
                            start_time: datetime, 
                            end_time: datetime, 
                            step: str = '1m') -> Dict[str, Any]:
        """Get network transmit bytes per second."""
        query = self.config.get_metric_query('network_metrics', 'tx_bytes')
        if not query:
            query = 'rate(node_network_transmit_bytes_total{device!="lo"}[5m])'
        
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
                'metric_type': 'network_tx_bytes_per_second',
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
            logger.error(f"Failed to get network TX bytes: {e}")
            raise
    
    def get_network_rx_packets(self, 
                              start_time: datetime, 
                              end_time: datetime, 
                              step: str = '1m') -> Dict[str, Any]:
        """Get network receive packets per second."""
        query = self.config.get_metric_query('network_metrics', 'rx_packets')
        if not query:
            query = 'rate(node_network_receive_packets_total{device!="lo"}[5m])'
        
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
                    'unit': 'packets/s'
                }
            
            return {
                'query': query,
                'metric_type': 'network_rx_packets_per_second',
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
            logger.error(f"Failed to get network RX packets: {e}")
            raise
    
    def get_network_tx_packets(self, 
                              start_time: datetime, 
                              end_time: datetime, 
                              step: str = '1m') -> Dict[str, Any]:
        """Get network transmit packets per second."""
        query = self.config.get_metric_query('network_metrics', 'tx_packets')
        if not query:
            query = 'rate(node_network_transmit_packets_total{device!="lo"}[5m])'
        
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
                    'unit': 'packets/s'
                }
            
            return {
                'query': query,
                'metric_type': 'network_tx_packets_per_second',
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
            logger.error(f"Failed to get network TX packets: {e}")
            raise
    
    def get_network_errors(self, 
                          start_time: datetime, 
                          end_time: datetime, 
                          step: str = '1m') -> Dict[str, Any]:
        """Get network error rates."""
        rx_errors_query = self.config.get_metric_query('network_metrics', 'rx_errors')
        tx_errors_query = self.config.get_metric_query('network_metrics', 'tx_errors')
        
        if not rx_errors_query:
            rx_errors_query = 'rate(node_network_receive_errs_total{device!="lo"}[5m])'
        if not tx_errors_query:
            tx_errors_query = 'rate(node_network_transmit_errs_total{device!="lo"}[5m])'
        
        try:
            # Get RX errors
            rx_result = self.prometheus.query_range(rx_errors_query, start_time, end_time, step)
            rx_formatted = self.prometheus.format_query_result(rx_result)
            
            # Get TX errors
            tx_result = self.prometheus.query_range(tx_errors_query, start_time, end_time, step)
            tx_formatted = self.prometheus.format_query_result(tx_result)
            
            # Combine error data by node and device
            nodes_data = {}
            
            # Process RX errors
            for item in rx_formatted['results']:
                instance = item['metric'].get('instance', 'unknown').split(':')[0]
                device = item['metric'].get('device', 'unknown')
                
                if instance not in nodes_data:
                    nodes_data[instance] = {}
                if device not in nodes_data[instance]:
                    nodes_data[instance][device] = {}
                
                stats = self.prometheus.calculate_statistics([item])
                nodes_data[instance][device]['rx_errors'] = {
                    'metric_labels': item['metric'],
                    'values': item['values'],
                    'statistics': stats,
                    'unit': 'errors/s'
                }
            
            # Process TX errors
            for item in tx_formatted['results']:
                instance = item['metric'].get('instance', 'unknown').split(':')[0]
                device = item['metric'].get('device', 'unknown')
                
                if instance not in nodes_data:
                    nodes_data[instance] = {}
                if device not in nodes_data[instance]:
                    nodes_data[instance][device] = {}
                
                stats = self.prometheus.calculate_statistics([item])
                nodes_data[instance][device]['tx_errors'] = {
                    'metric_labels': item['metric'],
                    'values': item['values'],
                    'statistics': stats,
                    'unit': 'errors/s'
                }
            
            return {
                'queries': {
                    'rx_errors': rx_errors_query,
                    'tx_errors': tx_errors_query
                },
                'metric_type': 'network_errors_per_second',
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'step': step
                },
                'nodes': nodes_data,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get network errors: {e}")
            raise
    
    def collect_network_metrics(self, 
                               duration_hours: float = 1.0, 
                               step: str = '1m') -> Dict[str, Any]:
        """Collect comprehensive network metrics.
        
        Args:
            duration_hours: Duration in hours to collect data for
            step: Query resolution step
        
        Returns:
            Dictionary containing network metrics and analysis
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=duration_hours)
        
        try:
            # Collect all network metrics
            rx_bytes_data = self.get_network_rx_bytes(start_time, end_time, step)
            tx_bytes_data = self.get_network_tx_bytes(start_time, end_time, step)
            rx_packets_data = self.get_network_rx_packets(start_time, end_time, step)
            tx_packets_data = self.get_network_tx_packets(start_time, end_time, step)
            errors_data = self.get_network_errors(start_time, end_time, step)
            
            # Combine data by node and device
            combined_data = {}
            all_nodes = set()
            
            # Collect all nodes from different metrics
            for data in [rx_bytes_data, tx_bytes_data, rx_packets_data, tx_packets_data]:
                all_nodes.update(data.get('nodes', {}).keys())
            all_nodes.update(errors_data.get('nodes', {}).keys())
            
            for node in all_nodes:
                combined_data[node] = {'interfaces': {}}
                
                # Get all network interfaces for this node
                all_interfaces = set()
                for data in [rx_bytes_data, tx_bytes_data, rx_packets_data, tx_packets_data]:
                    if node in data.get('nodes', {}):
                        all_interfaces.update(data['nodes'][node].keys())
                if node in errors_data.get('nodes', {}):
                    all_interfaces.update(errors_data['nodes'][node].keys())
                
                for interface in all_interfaces:
                    interface_data = {
                        'rx_bytes': rx_bytes_data.get('nodes', {}).get(node, {}).get(interface, {}),
                        'tx_bytes': tx_bytes_data.get('nodes', {}).get(node, {}).get(interface, {}),
                        'rx_packets': rx_packets_data.get('nodes', {}).get(node, {}).get(interface, {}),
                        'tx_packets': tx_packets_data.get('nodes', {}).get(node, {}).get(interface, {}),
                        'errors': errors_data.get('nodes', {}).get(node, {}).get(interface, {})
                    }
                    combined_data[node]['interfaces'][interface] = interface_data
            
            # Get baselines for comparison
            network_baselines = self.config.get_network_baselines()
            
            # Analyze performance against baselines
            performance_analysis = self._analyze_network_performance(combined_data, network_baselines)
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'collection_period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'duration_hours': duration_hours,
                    'step': step
                },
                'baselines': network_baselines,
                'performance_analysis': performance_analysis,
                'nodes': combined_data,
                'queries_executed': {
                    'rx_bytes': rx_bytes_data['query'],
                    'tx_bytes': tx_bytes_data['query'],
                    'rx_packets': rx_packets_data['query'],
                    'tx_packets': tx_packets_data['query'],
                    'rx_errors': errors_data['queries']['rx_errors'],
                    'tx_errors': errors_data['queries']['tx_errors']
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to collect network metrics: {e}")
            raise
    
    def _analyze_network_performance(self, 
                                    network_data: Dict[str, Any], 
                                    baselines: Dict[str, float]) -> Dict[str, Any]:
        """Analyze network performance against baselines."""
        analysis = {
            'overall_status': 'normal',
            'alerts': [],
            'summary': {
                'total_nodes': len(network_data),
                'total_interfaces': sum(len(node_data['interfaces']) for node_data in network_data.values()),
                'interfaces_with_errors': 0,
                'low_throughput_interfaces': 0,
                'high_utilization_interfaces': 0
            }
        }
        
        for node_name, node_data in network_data.items():
            for interface_name, interface_data in node_data['interfaces'].items():
                interface_id = f"{node_name}:{interface_name}"
                
                # Check for network errors
                rx_errors_stats = interface_data.get('errors', {}).get('rx_errors', {}).get('statistics', {})
                tx_errors_stats = interface_data.get('errors', {}).get('tx_errors', {}).get('statistics', {})
                
                if (rx_errors_stats.get('mean', 0) > 0 or tx_errors_stats.get('mean', 0) > 0):
                    analysis['alerts'].append({
                        'type': 'network_errors_detected',
                        'interface': interface_id,
                        'rx_errors_per_sec': rx_errors_stats.get('mean', 0),
                        'tx_errors_per_sec': tx_errors_stats.get('mean', 0),
                        'severity': 'warning'
                    })
                    analysis['summary']['interfaces_with_errors'] += 1
                
                # Check throughput
                rx_throughput_stats = interface_data.get('rx_bytes', {}).get('statistics', {})
                tx_throughput_stats = interface_data.get('tx_bytes', {}).get('statistics', {})
                
                # Check if throughput is below baseline (only if there's actual traffic)
                if (rx_throughput_stats.get('mean', 0) < baselines.get('rx_baseline', 10) and 
                    rx_throughput_stats.get('mean', 0) > 0.1):  # Ignore very low traffic
                    analysis['alerts'].append({
                        'type': 'low_rx_throughput',
                        'interface': interface_id,
                        'current_mbps': rx_throughput_stats.get('mean', 0),
                        'baseline_mbps': baselines.get('rx_baseline', 10),
                        'severity': 'info'
                    })
                    analysis['summary']['low_throughput_interfaces'] += 1
                
                if (tx_throughput_stats.get('mean', 0) < baselines.get('tx_baseline', 10) and 
                    tx_throughput_stats.get('mean', 0) > 0.1):  # Ignore very low traffic
                    analysis['alerts'].append({
                        'type': 'low_tx_throughput',
                        'interface': interface_id,
                        'current_mbps': tx_throughput_stats.get('mean', 0),
                        'baseline_mbps': baselines.get('tx_baseline', 10),
                        'severity': 'info'
                    })
                    analysis['summary']['low_throughput_interfaces'] += 1
                
                # Check for high utilization (approaching max throughput)
                max_throughput = baselines.get('max_throughput', 1000)  # 1Gbps default
                total_throughput = (rx_throughput_stats.get('max', 0) + tx_throughput_stats.get('max', 0))
                utilization_percent = (total_throughput / max_throughput) * 100
                
                if utilization_percent > 80:  # 80% utilization threshold
                    analysis['alerts'].append({
                        'type': 'high_network_utilization',
                        'interface': interface_id,
                        'utilization_percent': round(utilization_percent, 2),
                        'total_throughput_mbps': total_throughput,
                        'severity': 'warning' if utilization_percent > 90 else 'info'
                    })
                    analysis['summary']['high_utilization_interfaces'] += 1
        
        # Determine overall status
        if analysis['summary']['interfaces_with_errors'] > 0:
            analysis['overall_status'] = 'degraded'
        
        if analysis['summary']['high_utilization_interfaces'] > 0:
            if analysis['overall_status'] == 'normal':
                analysis['overall_status'] = 'stressed'
        
        if len(analysis['alerts']) == 0:
            analysis['overall_status'] = 'optimal'
        
        return analysis


def get_network_metrics(duration_hours: float = 1.0, step: str = '1m') -> str:
    """Get network metrics and return as JSON string.
    
    Args:
        duration_hours: Duration in hours to collect data for (default: 1 hour)
        step: Query resolution step (default: '1m')
    
    Returns:
        JSON string containing network metrics data
    """
    collector = NetworkMetricsCollector()
    metrics_data = collector.collect_network_metrics(duration_hours, step)
    return json.dumps(metrics_data, indent=2)

# Global network instance
network = NetworkMetricsCollector()