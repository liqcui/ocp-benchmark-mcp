"""Extract, Load, Transform module for OpenShift benchmark data."""
import json
import logging
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.ocp_benchmark_config import config_manager


logger = logging.getLogger(__name__)


class BenchmarkDataProcessor:
    """Processes and transforms benchmark data for analysis and reporting."""
    
    def __init__(self):
        self.config = config_manager
    
    def extract_json_data(self, json_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Extract and parse JSON data from MCP tools output.
        
        Args:
            json_data: JSON string or dictionary containing metrics data
        
        Returns:
            Parsed dictionary data
        """
        if isinstance(json_data, str):
            try:
                return json.loads(json_data)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON data: {e}")
                return {}
        elif isinstance(json_data, dict):
            return json_data
        else:
            logger.error(f"Unsupported data type: {type(json_data)}")
            return {}
    
    def nodes_usage_to_dataframe(self, nodes_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert nodes usage data to pandas DataFrame.
        
        Args:
            nodes_data: Nodes usage data from MCP tool
        
        Returns:
            DataFrame with nodes usage metrics
        """
        rows = []
        
        for node_name, node_data in nodes_data.get('nodes', {}).items():
            cpu_stats = node_data.get('cpu_usage', {}).get('statistics', {})
            memory_stats = node_data.get('memory_usage', {}).get('statistics', {})
            memory_total = node_data.get('memory_total', {})
            
            row = {
                'node_name': node_name,
                'cpu_min_percent': cpu_stats.get('min', 0),
                'cpu_max_percent': cpu_stats.get('max', 0),
                'cpu_mean_percent': cpu_stats.get('mean', 0),
                'memory_min_percent': memory_stats.get('min', 0),
                'memory_max_percent': memory_stats.get('max', 0),
                'memory_mean_percent': memory_stats.get('mean', 0),
                'memory_total_gb': memory_total.get('gb', 0),
                'timestamp': nodes_data.get('timestamp', ''),
                'collection_duration_hours': nodes_data.get('collection_period', {}).get('duration_hours', 0)
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def pods_usage_to_dataframe(self, pods_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert pods usage data to pandas DataFrame.
        
        Args:
            pods_data: Pods usage data from MCP tool
        
        Returns:
            DataFrame with pods usage metrics
        """
        rows = []
        
        for pod_key, pod_data in pods_data.get('pods', {}).items():
            namespace = pod_data.get('namespace', 'unknown')
            pod_name = pod_data.get('pod_name', 'unknown')
            
            # Get pod-level totals
            cpu_totals = pod_data.get('pod_totals', {}).get('cpu_usage', {})
            memory_totals = pod_data.get('pod_totals', {}).get('memory_usage', {})
            
            # Also process individual containers
            for container_name, container_data in pod_data.get('containers', {}).items():
                cpu_stats = container_data.get('cpu_usage', {}).get('statistics', {})
                memory_stats = container_data.get('memory_usage', {}).get('statistics', {})
                memory_percent_stats = container_data.get('memory_usage_percent_of_limit', {}).get('statistics', {})
                
                row = {
                    'namespace': namespace,
                    'pod_name': pod_name,
                    'container_name': container_name,
                    'cpu_min_percent': cpu_stats.get('min', 0),
                    'cpu_max_percent': cpu_stats.get('max', 0),
                    'cpu_mean_percent': cpu_stats.get('mean', 0),
                    'memory_min_mb': memory_stats.get('min', 0),
                    'memory_max_mb': memory_stats.get('max', 0),
                    'memory_mean_mb': memory_stats.get('mean', 0),
                    'memory_percent_of_limit_min': memory_percent_stats.get('min', 0),
                    'memory_percent_of_limit_max': memory_percent_stats.get('max', 0),
                    'memory_percent_of_limit_mean': memory_percent_stats.get('mean', 0),
                    'timestamp': pods_data.get('timestamp', ''),
                    'collection_duration_hours': pods_data.get('collection_period', {}).get('duration_hours', 0)
                }
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def disk_metrics_to_dataframe(self, disk_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert disk metrics data to pandas DataFrame."""
        rows = []
        
        for node_name, node_data in disk_data.get('nodes', {}).items():
            for device_name, device_data in node_data.get('devices', {}).items():
                
                # Extract statistics from each metric type
                read_throughput_stats = device_data.get('read_throughput', {}).get('statistics', {})
                write_throughput_stats = device_data.get('write_throughput', {}).get('statistics', {})
                read_iops_stats = device_data.get('read_iops', {}).get('statistics', {})
                write_iops_stats = device_data.get('write_iops', {}).get('statistics', {})
                read_latency_stats = device_data.get('read_latency', {}).get('statistics', {})
                write_latency_stats = device_data.get('write_latency', {}).get('statistics', {})
                
                row = {
                    'node_name': node_name,
                    'device_name': device_name,
                    'read_throughput_min_mbs': read_throughput_stats.get('min', 0),
                    'read_throughput_max_mbs': read_throughput_stats.get('max', 0),
                    'read_throughput_mean_mbs': read_throughput_stats.get('mean', 0),
                    'write_throughput_min_mbs': write_throughput_stats.get('min', 0),
                    'write_throughput_max_mbs': write_throughput_stats.get('max', 0),
                    'write_throughput_mean_mbs': write_throughput_stats.get('mean', 0),
                    'read_iops_min': read_iops_stats.get('min', 0),
                    'read_iops_max': read_iops_stats.get('max', 0),
                    'read_iops_mean': read_iops_stats.get('mean', 0),
                    'write_iops_min': write_iops_stats.get('min', 0),
                    'write_iops_max': write_iops_stats.get('max', 0),
                    'write_iops_mean': write_iops_stats.get('mean', 0),
                    'read_latency_min_ms': read_latency_stats.get('min', 0),
                    'read_latency_max_ms': read_latency_stats.get('max', 0),
                    'read_latency_mean_ms': read_latency_stats.get('mean', 0),
                    'write_latency_min_ms': write_latency_stats.get('min', 0),
                    'write_latency_max_ms': write_latency_stats.get('max', 0),
                    'write_latency_mean_ms': write_latency_stats.get('mean', 0),
                    'timestamp': disk_data.get('timestamp', ''),
                    'collection_duration_hours': disk_data.get('collection_period', {}).get('duration_hours', 0)
                }
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def network_metrics_to_dataframe(self, network_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert network metrics data to pandas DataFrame."""
        rows = []
        
        for node_name, node_data in network_data.get('nodes', {}).items():
            for interface_name, interface_data in node_data.get('interfaces', {}).items():
                
                # Extract statistics from each metric type
                rx_bytes_stats = interface_data.get('rx_bytes', {}).get('statistics', {})
                tx_bytes_stats = interface_data.get('tx_bytes', {}).get('statistics', {})
                rx_packets_stats = interface_data.get('rx_packets', {}).get('statistics', {})
                tx_packets_stats = interface_data.get('tx_packets', {}).get('statistics', {})
                
                # Extract error statistics if available
                errors_data = interface_data.get('errors', {})
                rx_errors_stats = errors_data.get('rx_errors', {}).get('statistics', {})
                tx_errors_stats = errors_data.get('tx_errors', {}).get('statistics', {})
                
                row = {
                    'node_name': node_name,
                    'interface_name': interface_name,
                    'rx_bytes_min_mbs': rx_bytes_stats.get('min', 0),
                    'rx_bytes_max_mbs': rx_bytes_stats.get('max', 0),
                    'rx_bytes_mean_mbs': rx_bytes_stats.get('mean', 0),
                    'tx_bytes_min_mbs': tx_bytes_stats.get('min', 0),
                    'tx_bytes_max_mbs': tx_bytes_stats.get('max', 0),
                    'tx_bytes_mean_mbs': tx_bytes_stats.get('mean', 0),
                    'rx_packets_min_pps': rx_packets_stats.get('min', 0),
                    'rx_packets_max_pps': rx_packets_stats.get('max', 0),
                    'rx_packets_mean_pps': rx_packets_stats.get('mean', 0),
                    'tx_packets_min_pps': tx_packets_stats.get('min', 0),
                    'tx_packets_max_pps': tx_packets_stats.get('max', 0),
                    'tx_packets_mean_pps': tx_packets_stats.get('mean', 0),
                    'rx_errors_per_sec': rx_errors_stats.get('mean', 0),
                    'tx_errors_per_sec': tx_errors_stats.get('mean', 0),
                    'timestamp': network_data.get('timestamp', ''),
                    'collection_duration_hours': network_data.get('collection_period', {}).get('duration_hours', 0)
                }
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def api_latency_to_dataframe(self, api_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert API latency data to pandas DataFrame."""
        rows = []
        
        # Process API server latency operations
        for operation, operation_data in api_data.get('api_server_latency', {}).get('operations', {}).items():
            summary_stats = operation_data.get('summary_stats', {})
            
            row = {
                'operation': operation,
                'metric_type': 'api_server_latency',
                # 'p50_mean_ms': summary_stats.get('p50_mean_ms', 0),
                # 'p95_mean_ms': summary_stats.get('p95_mean_ms', 0),
                'p99_mean_ms': summary_stats.get('p99_mean_ms', 0),
                # 'p50_max_ms': summary_stats.get('p50_max_ms', 0),
                # 'p95_max_ms': summary_stats.get('p95_max_ms', 0),
                'p99_max_ms': summary_stats.get('p99_max_ms', 0),
                'timestamp': api_data.get('timestamp', ''),
                'collection_duration_hours': api_data.get('collection_period', {}).get('duration_hours', 0)
            }
            rows.append(row)
        
        # Process etcd latency operations
        for operation, operation_data in api_data.get('etcd_latency', {}).get('operations', {}).items():
            latency_stats = operation_data.get('latency_stats', {})
            
            row = {
                'operation': operation,
                'metric_type': 'etcd_latency',
                # 'p50_mean_ms': 0,  # etcd only provides p99
                # 'p95_mean_ms': 0,
                'p99_mean_ms': latency_stats.get('mean', 0),
                # 'p50_max_ms': 0,
                # 'p95_max_ms': 0,
                'p99_max_ms': latency_stats.get('max', 0),
                'timestamp': api_data.get('timestamp', ''),
                'collection_duration_hours': api_data.get('collection_period', {}).get('duration_hours', 0)
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_summary_table(self, df: pd.DataFrame, title: str) -> str:
        """Generate a formatted summary table from DataFrame."""
        if df.empty:
            return f"\n{title}\n{'=' * len(title)}\nNo data available\n"
        
        # Create a summary string
        summary = f"\n{title}\n{'=' * len(title)}\n"
        summary += f"Total Records: {len(df)}\n"
        
        # Add column summary if DataFrame has numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns
        if len(numeric_columns) > 0:
            summary += "\nNumeric Column Statistics:\n"
            summary += "-" * 30 + "\n"
            # Always include stats (even if zero), round to 6 decimals
            for col in list(numeric_columns)[:10]:  # Limit to first 10 columns
                series = df[col].dropna()
                col_min = float(series.min()) if not series.empty else 0.0
                col_max = float(series.max()) if not series.empty else 0.0
                col_mean = float(series.mean()) if not series.empty else 0.0
                summary += f"{col}:\n"
                summary += f"  Min: {col_min:.6f}\n"
                summary += f"  Max: {col_max:.6f}\n"
                summary += f"  Mean: {col_mean:.6f}\n\n"
        
        # Add top rows preview
        summary += "\nData Preview (first 5 rows):\n"
        summary += "-" * 30 + "\n"
        try:
            preview = df.head().to_string(index=False, max_cols=8, max_colwidth=20)
        except TypeError:
            # Fallback for pandas versions without max_colwidth in to_string
            preview = df.head().to_string(index=False)
        if not preview.strip():
            preview = "(no rows to display)"
        summary += preview
        summary += "\n\n"
        
        return summary
    
    def analyze_performance_against_baselines(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance data against configured baselines."""
        analysis = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_status': 'normal',
            'findings': [],
            'recommendations': [],
            'baseline_comparisons': {}
        }
        
        # Analyze based on data type
        if 'nodes' in data and 'cluster_statistics' in data:
            # This is nodes usage data
            analysis['baseline_comparisons']['nodes'] = self._analyze_nodes_baselines(data)
        
        if 'pods' in data:
            # This is pods usage data
            analysis['baseline_comparisons']['pods'] = self._analyze_pods_performance(data)
        
        if 'performance_analysis' in data:
            # Data already contains performance analysis
            existing_analysis = data['performance_analysis']
            analysis['overall_status'] = existing_analysis.get('overall_status', 'normal')
            
            if 'alerts' in existing_analysis:
                for alert in existing_analysis['alerts']:
                    analysis['findings'].append({
                        'type': alert.get('type', 'unknown'),
                        'severity': alert.get('severity', 'info'),
                        'message': self._format_alert_message(alert)
                    })
        
        # Generate recommendations based on findings
        analysis['recommendations'] = self._generate_recommendations(analysis['findings'])
        
        # Determine overall status
        if any(f['severity'] == 'critical' for f in analysis['findings']):
            analysis['overall_status'] = 'critical'
        elif any(f['severity'] == 'warning' for f in analysis['findings']):
            analysis['overall_status'] = 'degraded'
        elif len(analysis['findings']) == 0:
            analysis['overall_status'] = 'optimal'
        
        return analysis
    
    def _analyze_nodes_baselines(self, nodes_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze nodes data against baselines."""
        cpu_baselines = self.config.get_cpu_baselines()
        memory_baselines = self.config.get_memory_baselines()
        
        cluster_stats = nodes_data.get('cluster_statistics', {})
        cpu_cluster_stats = cluster_stats.get('cpu_usage', {})
        memory_cluster_stats = cluster_stats.get('memory_usage', {})
        
        comparison = {
            'cpu': {
                'current_mean': cpu_cluster_stats.get('mean', 0),
                'baseline_mean': cpu_baselines['mean'],
                'status': 'normal',
                'variance_from_baseline': 0
            },
            'memory': {
                'current_mean': memory_cluster_stats.get('mean', 0),
                'baseline_mean': memory_baselines['mean'],
                'status': 'normal',
                'variance_from_baseline': 0
            }
        }
        
        # Calculate variance from baseline
        if cpu_baselines['mean'] > 0:
            comparison['cpu']['variance_from_baseline'] = (
                (comparison['cpu']['current_mean'] - cpu_baselines['mean']) / cpu_baselines['mean'] * 100
            )
        
        if memory_baselines['mean'] > 0:
            comparison['memory']['variance_from_baseline'] = (
                (comparison['memory']['current_mean'] - memory_baselines['mean']) / memory_baselines['mean'] * 100
            )
        
        # Determine status
        if comparison['cpu']['current_mean'] > cpu_baselines['critical']:
            comparison['cpu']['status'] = 'critical'
        elif comparison['cpu']['current_mean'] > cpu_baselines['warning']:
            comparison['cpu']['status'] = 'warning'
        
        if comparison['memory']['current_mean'] > memory_baselines['critical']:
            comparison['memory']['status'] = 'critical'
        elif comparison['memory']['current_mean'] > memory_baselines['warning']:
            comparison['memory']['status'] = 'warning'
        
        return comparison
    
    def _analyze_pods_performance(self, pods_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pods performance data."""
        summary = pods_data.get('summary', {})
        performance_alerts = pods_data.get('performance_alerts', {})
        
        return {
            'total_pods': summary.get('total_pods', 0),
            'high_cpu_pods': len(performance_alerts.get('high_cpu_usage', [])),
            'high_memory_pods': len(performance_alerts.get('high_memory_usage', [])),
            'namespaces_affected': len(summary.get('namespaces', {})),
            'status': 'degraded' if (performance_alerts.get('high_cpu_usage') or performance_alerts.get('high_memory_usage')) else 'normal'
        }
    
    def _format_alert_message(self, alert: Dict[str, Any]) -> str:
        """Format alert into human-readable message."""
        alert_type = alert.get('type', 'unknown')
        severity = alert.get('severity', 'info')
        
        if alert_type == 'high_cpu_usage':
            return f"High CPU usage detected: {alert.get('current', 0):.1f}% (threshold: {alert.get('baseline', 0):.1f}%)"
        elif alert_type == 'high_memory_usage':
            return f"High memory usage detected: {alert.get('current', 0):.1f}% (threshold: {alert.get('baseline', 0):.1f}%)"
        elif alert_type == 'high_read_latency':
            return f"High disk read latency on {alert.get('device', 'unknown')}: {alert.get('current', 0):.1f}ms"
        elif alert_type == 'high_write_latency':
            return f"High disk write latency on {alert.get('device', 'unknown')}: {alert.get('current', 0):.1f}ms"
        elif alert_type == 'network_errors_detected':
            return f"Network errors detected on {alert.get('interface', 'unknown')}: {alert.get('rx_errors_per_sec', 0):.2f} RX, {alert.get('tx_errors_per_sec', 0):.2f} TX errors/sec"
        elif alert_type.endswith('_latency'):
            return f"High {alert_type.replace('_', ' ')}: {alert.get('current_ms', 0):.1f}ms (baseline: {alert.get('baseline_ms', 0):.1f}ms)"
        else:
            return f"{alert_type}: {alert.get('message', 'No details available')}"
    
    def _generate_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []
        
        # CPU recommendations
        cpu_issues = [f for f in findings if 'cpu' in f['type'].lower()]
        if cpu_issues:
            recommendations.append("Consider scaling workloads or adding more CPU resources to nodes")
            recommendations.append("Review pod resource requests and limits for CPU")
        
        # Memory recommendations
        memory_issues = [f for f in findings if 'memory' in f['type'].lower()]
        if memory_issues:
            recommendations.append("Consider adding more memory to nodes or optimizing memory usage")
            recommendations.append("Review pod memory requests and limits")
        
        # Disk recommendations
        disk_issues = [f for f in findings if 'disk' in f['type'].lower() or 'latency' in f['type'].lower()]
        if disk_issues:
            recommendations.append("Consider upgrading to faster storage or optimizing I/O patterns")
            recommendations.append("Review persistent volume configurations")
        
        # Network recommendations
        network_issues = [f for f in findings if 'network' in f['type'].lower()]
        if network_issues:
            recommendations.append("Review network configuration and consider network optimizations")
            recommendations.append("Check for network congestion or hardware issues")
        
        # API recommendations
        api_issues = [f for f in findings if 'api' in f['type'].lower() or 'etcd' in f['type'].lower()]
        if api_issues:
            recommendations.append("Consider etcd performance tuning or API server scaling")
            recommendations.append("Review client request patterns and implement rate limiting")
        
        if not recommendations:
            recommendations.append("System performance is within acceptable parameters")
            recommendations.append("Continue regular monitoring and maintain current configuration")
        
        return recommendations


def analyze_performance_data(metrics_data: Dict[str, Any], analysis_type: str = "comprehensive") -> str:
    """Analyze performance data and return formatted results.
    
    Args:
        metrics_data: Dictionary containing metrics data from MCP tools
        analysis_type: Type of analysis to perform
    
    Returns:
        JSON string containing analysis results and formatted tables
    """
    processor = BenchmarkDataProcessor()
    
    try:
        # Extract and parse data
        if isinstance(metrics_data, str):
            data = processor.extract_json_data(metrics_data)
        else:
            data = metrics_data
        
        # Determine data type and process accordingly
        results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'analysis_type': analysis_type,
            'data_summary': {},
            'tables': {},
            'performance_analysis': {},
            'recommendations': []
        }
        
        # Process different types of data
        if 'nodes' in data and not 'pods' in data:
            # Nodes usage data
            df = processor.nodes_usage_to_dataframe(data)
            results['tables']['nodes_usage_summary'] = processor.generate_summary_table(df, "Nodes Usage Summary")
            results['data_summary']['nodes_count'] = len(df)
            
        elif 'pods' in data:
            # Pods usage data
            df = processor.pods_usage_to_dataframe(data)
            results['tables']['pods_usage_summary'] = processor.generate_summary_table(df, "Pods Usage Summary")
            results['data_summary']['pods_count'] = len(data.get('pods', {}))
            results['data_summary']['containers_count'] = len(df)
            
        elif 'nodes' in data and 'interfaces' in str(data):
            # Network metrics data
            df = processor.network_metrics_to_dataframe(data)
            results['tables']['network_metrics_summary'] = processor.generate_summary_table(df, "Network Metrics Summary")
            results['data_summary']['network_interfaces_count'] = len(df)
            
        elif 'nodes' in data and 'devices' in str(data):
            # Disk metrics data
            df = processor.disk_metrics_to_dataframe(data)
            results['tables']['disk_metrics_summary'] = processor.generate_summary_table(df, "Disk Metrics Summary")
            results['data_summary']['disk_devices_count'] = len(df)
            
        elif 'api_server_latency' in data:
            # API latency data
            df = processor.api_latency_to_dataframe(data)
            results['tables']['api_latency_summary'] = processor.generate_summary_table(df, "API Latency Summary")
            results['data_summary']['api_operations_count'] = len(df)
        
        # Perform performance analysis
        performance_analysis = processor.analyze_performance_against_baselines(data)
        results['performance_analysis'] = performance_analysis
        results['recommendations'] = performance_analysis.get('recommendations', [])
        
        # Add executive summary
        results['executive_summary'] = {
            'overall_status': performance_analysis.get('overall_status', 'unknown'),
            'total_findings': len(performance_analysis.get('findings', [])),
            'critical_issues': len([f for f in performance_analysis.get('findings', []) if f.get('severity') == 'critical']),
            'warning_issues': len([f for f in performance_analysis.get('findings', []) if f.get('severity') == 'warning']),
            'data_collection_timestamp': data.get('timestamp', ''),
            'analysis_timestamp': results['timestamp']
        }
        
        return json.dumps(results, indent=2)
    
    except Exception as e:
        logger.error(f"Failed to analyze performance data: {e}")
        error_result = {
            'error': f"Failed to analyze performance data: {str(e)}",
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'analysis_type': analysis_type
        }
        return json.dumps(error_result, indent=2)

# Global cluster info collector instance
benchmark_data_processor = BenchmarkDataProcessor()