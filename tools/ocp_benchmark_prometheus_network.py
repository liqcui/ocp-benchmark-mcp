#!/usr/bin/env python3
"""Prometheus Network Metrics Module"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import requests
import asyncio
from config.ocp_benchmark_config import config
from ocauth.ocp_benchmark_auth import auth

logger = logging.getLogger(__name__)

class NetworkMetricsCollector:
    """Collect network metrics from Prometheus"""
    
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
    
    async def get_network_throughput_metrics(self, duration_hours: int = 1) -> Dict[str, Any]:
        """Get network throughput metrics (RX/TX bytes and packets)"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=duration_hours)
            
            # Get network queries from config
            rx_bytes_query = config.get_metric_query('network', 'receive_bytes')
            tx_bytes_query = config.get_metric_query('network', 'transmit_bytes')
            rx_packets_query = config.get_metric_query('network', 'receive_packets')
            tx_packets_query = config.get_metric_query('network', 'transmit_packets')
            
            if not all([rx_bytes_query, tx_bytes_query, rx_packets_query, tx_packets_query]):
                return {'error': 'Network throughput queries not properly configured'}
            
            # Execute queries concurrently
            tasks = [
                self._query_prometheus(rx_bytes_query, start_time, end_time),
                self._query_prometheus(tx_bytes_query, start_time, end_time),
                self._query_prometheus(rx_packets_query, start_time, end_time),
                self._query_prometheus(tx_packets_query, start_time, end_time)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            rx_bytes_result, tx_bytes_result, rx_packets_result, tx_packets_result = results
            
            # Process results
            metrics = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'duration_hours': duration_hours,
                'rx_bytes': self._process_network_metric_result(rx_bytes_result, 'bytes_per_second'),
                'tx_bytes': self._process_network_metric_result(tx_bytes_result, 'bytes_per_second'),
                'rx_packets': self._process_network_metric_result(rx_packets_result, 'packets_per_second'),
                'tx_packets': self._process_network_metric_result(tx_packets_result, 'packets_per_second')
            }
            
            # Add baseline comparison
            baselines = config.get_network_baselines()
            metrics['baseline_comparison'] = self._compare_with_network_baseline(metrics, baselines)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting network throughput metrics: {e}")
            return {'error': str(e)}
    
    async def get_network_packet_loss_metrics(self, duration_hours: int = 1) -> Dict[str, Any]:
        """Get network packet loss metrics"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=duration_hours)
            
            # Get packet loss query from config
            packet_loss_query = config.get_metric_query('network', 'packet_loss')
            
            if not packet_loss_query:
                return {'error': 'Network packet loss query not configured'}
            
            result = await self._query_prometheus(packet_loss_query, start_time, end_time)
            
            if not result:
                return {'error': 'Failed to query Prometheus for packet loss'}
            
            # Process packet loss results
            metrics = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'duration_hours': duration_hours,
                'packet_loss': self._process_network_metric_result(result, 'packets_per_second_dropped')
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting network packet loss metrics: {e}")
            return {'error': str(e)}
    
    def _process_network_metric_result(self, result: Any, unit: str) -> Dict[str, Any]:
        """Process Prometheus result for network metrics"""
        if isinstance(result, Exception) or not result:
            return {'error': 'Failed to query metric', 'unit': unit}
        
        interface_stats = {}
        all_values = []
        
        for series in result['data']['result']:
            device = series['metric'].get('device', 'unknown')
            instance = series['metric'].get('instance', 'unknown')
            interface_key = f"{instance}:{device}"
            
            # Skip loopback and other non-physical interfaces
            if device in ['lo', 'docker0'] or device.startswith('veth'):
                continue
            
            values = []
            if 'values' in series:
                for value_pair in series['values']:
                    try:
                        value = float(value_pair[1])
                        values.append(value)
                        all_values.append(value)
                    except (ValueError, IndexError):
                        continue
            
            if values:
                interface_stats[interface_key] = self._calculate_stats(values)
        
        overall_stats = self._calculate_stats(all_values)
        
        return {
            'unit': unit,
            'overall_stats': overall_stats,
            'interface_stats': interface_stats,
            'interface_count': len(interface_stats)
        }
    
    def _compare_with_network_baseline(self, metrics: Dict[str, Any], baselines: Dict[str, float]) -> Dict[str, Any]:
        """Compare network metrics with baseline values"""
        try:
            comparison = {
                'baselines': baselines,
                'within_thresholds': True,
                'issues': [],
                'performance_summary': {}
            }
            
            # Check RX throughput (convert bytes/sec to MB/sec)
            if 'rx_bytes' in metrics and 'overall_stats' in metrics['rx_bytes']:
                avg_bytes_per_sec = metrics['rx_bytes']['overall_stats'].get('mean', 0)
                avg_mb_per_sec = avg_bytes_per_sec / (1024 * 1024)
                comparison['performance_summary']['avg_rx_mbps'] = round(avg_mb_per_sec, 2)
                
                if avg_mb_per_sec < baselines['rx_baseline']:
                    comparison['within_thresholds'] = False
                    comparison['issues'].append({
                        'metric': 'rx_throughput',
                        'current_mbps': avg_mb_per_sec,
                        'baseline_mbps': baselines['rx_baseline'],
                        'status': 'below_baseline'
                    })
            
            # Check TX throughput
            if 'tx_bytes' in metrics and 'overall_stats' in metrics['tx_bytes']:
                avg_bytes_per_sec = metrics['tx_bytes']['overall_stats'].get('mean', 0)
                avg_mb_per_sec = avg_bytes_per_sec / (1024 * 1024)
                comparison['performance_summary']['avg_tx_mbps'] = round(avg_mb_per_sec, 2)
                
                if avg_mb_per_sec < baselines['tx_baseline']:
                    comparison['within_thresholds'] = False
                    comparison['issues'].append({
                        'metric': 'tx_throughput',
                        'current_mbps': avg_mb_per_sec,
                        'baseline_mbps': baselines['tx_baseline'],
                        'status': 'below_baseline'
                    })
            
            # Check combined throughput against max throughput
            total_mbps = comparison['performance_summary'].get('avg_rx_mbps', 0) + \
                        comparison['performance_summary'].get('avg_tx_mbps', 0)
            
            if total_mbps > baselines['max_throughput_mbps']:
                comparison['issues'].append({
                    'metric': 'total_throughput',
                    'current_mbps': total_mbps,
                    'max_throughput_mbps': baselines['max_throughput_mbps'],
                    'status': 'exceeds_maximum'
                })
            
            comparison['performance_summary']['total_throughput_mbps'] = round(total_mbps, 2)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing with network baseline: {e}")
            return {'error': str(e)}
    
    async def get_combined_network_metrics(self, duration_hours: int = 1) -> Dict[str, Any]:
        """Get combined network metrics including throughput and packet loss"""
        try:
            throughput_task = self.get_network_throughput_metrics(duration_hours)
            packet_loss_task = self.get_network_packet_loss_metrics(duration_hours)
            
            throughput_result, packet_loss_result = await asyncio.gather(
                throughput_task, packet_loss_task, return_exceptions=True
            )
            
            return {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'duration_hours': duration_hours,
                'throughput_metrics': throughput_result if not isinstance(throughput_result, Exception) else {'error': str(throughput_result)},
                'packet_loss_metrics': packet_loss_result if not isinstance(packet_loss_result, Exception) else {'error': str(packet_loss_result)},
                'summary': self._generate_network_summary(throughput_result, packet_loss_result)
            }
            
        except Exception as e:
            logger.error(f"Error getting combined network metrics: {e}")
            return {'error': str(e)}
    
    def _generate_network_summary(self, throughput_metrics: Any, packet_loss_metrics: Any) -> Dict[str, Any]:
        """Generate summary of network performance"""
        try:
            summary = {
                'overall_status': 'healthy',
                'issues_detected': [],
                'performance_indicators': {}
            }
            
            # Check throughput performance
            if (not isinstance(throughput_metrics, Exception) and 
                'baseline_comparison' in throughput_metrics):
                
                comparison = throughput_metrics['baseline_comparison']
                if not comparison.get('within_thresholds', True):
                    summary['overall_status'] = 'degraded'
                    summary['issues_detected'].extend([
                        f"{issue['metric']}: {issue['status']}" 
                        for issue in comparison.get('issues', [])
                    ])
                
                # Extract performance indicators
                perf_summary = comparison.get('performance_summary', {})
                summary['performance_indicators'].update(perf_summary)
            
            # Check packet loss
            if (not isinstance(packet_loss_metrics, Exception) and 
                'packet_loss' in packet_loss_metrics):
                
                packet_loss_stats = packet_loss_metrics['packet_loss'].get('overall_stats', {})
                avg_packet_loss = packet_loss_stats.get('mean', 0)
                max_packet_loss = packet_loss_stats.get('max', 0)
                
                summary['performance_indicators']['avg_packet_loss_per_sec'] = round(avg_packet_loss, 2)
                summary['performance_indicators']['max_packet_loss_per_sec'] = round(max_packet_loss, 2)
                
                # Check against baseline
                baselines = config.get_network_baselines()
                if avg_packet_loss > baselines.get('packet_loss_threshold', 0.1):
                    summary['overall_status'] = 'degraded'
                    summary['issues_detected'].append('packet_loss: above_threshold')
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating network summary: {e}")
            return {'error': str(e)}
    
    async def get_network_utilization_by_interface(self, duration_hours: int = 1) -> Dict[str, Any]:
        """Get network utilization broken down by interface"""
        try:
            combined_metrics = await self.get_combined_network_metrics(duration_hours)
            
            if 'error' in combined_metrics:
                return combined_metrics
            
            interfaces_summary = {}
            
            # Process throughput metrics by interface
            throughput_metrics = combined_metrics.get('throughput_metrics', {})
            
            # RX bytes by interface
            rx_interfaces = throughput_metrics.get('rx_bytes', {}).get('interface_stats', {})
            for interface_key, stats in rx_interfaces.items():
                if interface_key not in interfaces_summary:
                    interfaces_summary[interface_key] = {'interface_key': interface_key}
                
                interfaces_summary[interface_key]['rx_stats'] = stats
                interfaces_summary[interface_key]['rx_mbps'] = round(stats.get('mean', 0) / (1024 * 1024), 2)
            
            # TX bytes by interface
            tx_interfaces = throughput_metrics.get('tx_bytes', {}).get('interface_stats', {})
            for interface_key, stats in tx_interfaces.items():
                if interface_key not in interfaces_summary:
                    interfaces_summary[interface_key] = {'interface_key': interface_key}
                
                interfaces_summary[interface_key]['tx_stats'] = stats
                interfaces_summary[interface_key]['tx_mbps'] = round(stats.get('mean', 0) / (1024 * 1024), 2)
            
            # Add packet data
            rx_packet_interfaces = throughput_metrics.get('rx_packets', {}).get('interface_stats', {})
            for interface_key, stats in rx_packet_interfaces.items():
                if interface_key in interfaces_summary:
                    interfaces_summary[interface_key]['rx_packets_per_sec'] = round(stats.get('mean', 0), 2)
            
            tx_packet_interfaces = throughput_metrics.get('tx_packets', {}).get('interface_stats', {})
            for interface_key, stats in tx_packet_interfaces.items():
                if interface_key in interfaces_summary:
                    interfaces_summary[interface_key]['tx_packets_per_sec'] = round(stats.get('mean', 0), 2)
            
            # Calculate total utilization per interface
            for interface_key, interface_data in interfaces_summary.items():
                total_mbps = interface_data.get('rx_mbps', 0) + interface_data.get('tx_mbps', 0)
                interfaces_summary[interface_key]['total_mbps'] = round(total_mbps, 2)
            
            return {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'duration_hours': duration_hours,
                'interfaces': list(interfaces_summary.values()),
                'interface_count': len(interfaces_summary)
            }
            
        except Exception as e:
            logger.error(f"Error getting network utilization by interface: {e}")
            return {'error': str(e)}

# Global network metrics collector instance
network_metrics_collector = NetworkMetricsCollector()

async def get_network_metrics_json(duration_hours: int = 1) -> str:
    """Get network metrics as JSON string"""
    metrics = await network_metrics_collector.get_combined_network_metrics(duration_hours)
    return json.dumps(metrics, indent=2)

async def get_network_utilization_json(duration_hours: int = 1) -> str:
    """Get network utilization by interface as JSON string"""
    utilization = await network_metrics_collector.get_network_utilization_by_interface(duration_hours)
    return json.dumps(utilization, indent=2)