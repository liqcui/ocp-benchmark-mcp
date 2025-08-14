#!/usr/bin/env python3
"""OpenShift Benchmark Performance Analysis Module"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import statistics

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """Advanced performance analysis for OpenShift clusters"""
    
    def __init__(self):
        self.severity_weights = {
            'critical': 1.0,
            'warning': 0.6,
            'info': 0.3
        }
    
    async def analyze_comprehensive_data(self, performance_data: Dict[str, Any], 
                                       baseline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze comprehensive performance data against baselines"""
        try:
            analysis = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'cluster_health_score': 0.0,
                'performance_trends': {},
                'baseline_deviations': {},
                'critical_issues': [],
                'warnings': [],
                'recommendations': [],
                'resource_utilization': {},
                'bottlenecks': [],
                'efficiency_metrics': {}
            }
            
            # Analyze each component
            if 'cluster_info' in performance_data:
                cluster_analysis = self._analyze_cluster_health(performance_data['cluster_info'])
                analysis.update(cluster_analysis)
            
            if 'node_usage' in performance_data:
                node_analysis = await self._analyze_node_usage(
                    performance_data['node_usage'], 
                    baseline_data.get('cpu_baselines', {}),
                    baseline_data.get('memory_baselines', {})
                )
                analysis['resource_utilization']['nodes'] = node_analysis
                analysis['performance_trends'].update(node_analysis.get('trends', {}))
            
            if 'disk_metrics' in performance_data:
                disk_analysis = await self._analyze_disk_performance(
                    performance_data['disk_metrics'],
                    baseline_data.get('disk_baselines', {})
                )
                analysis['resource_utilization']['disk'] = disk_analysis
                analysis['bottlenecks'].extend(disk_analysis.get('bottlenecks', []))
            
            if 'network_metrics' in performance_data:
                network_analysis = await self._analyze_network_performance(
                    performance_data['network_metrics'],
                    baseline_data.get('network_baselines', {})
                )
                analysis['resource_utilization']['network'] = network_analysis
            
            if 'api_latency' in performance_data:
                api_analysis = await self._analyze_api_performance(
                    performance_data['api_latency'],
                    baseline_data.get('api_baselines', {})
                )
                analysis['resource_utilization']['api'] = api_analysis
                if api_analysis.get('slow_endpoints'):
                    analysis['bottlenecks'].extend(api_analysis['slow_endpoints'])
            
            # Calculate overall health score
            analysis['cluster_health_score'] = self._calculate_health_score(analysis)
            
            # Identify critical issues
            analysis['critical_issues'] = self._identify_critical_issues(analysis)
            
            # Generate efficiency metrics
            analysis['efficiency_metrics'] = self._calculate_efficiency_metrics(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_cluster_health(self, cluster_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cluster health status"""
        try:
            health = {
                'cluster_status': 'healthy',
                'operator_health': {},
                'version_status': {},
                'infrastructure_status': {}
            }
            
            # Check cluster version
            if 'cluster_version' in cluster_info:
                version_info = cluster_info['cluster_version']
                if version_info and 'conditions' in version_info:
                    health['version_status'] = {
                        'version': version_info.get('version', 'unknown'),
                        'healthy': True
                    }
                    
                    for condition in version_info.get('conditions', []):
                        if condition.get('type') == 'Failing' and condition.get('status') == 'True':
                            health['version_status']['healthy'] = False
                            health['cluster_status'] = 'degraded'
            
            # Check operators
            if 'operators_status' in cluster_info:
                operators = cluster_info['operators_status']
                if operators:
                    total = operators.get('total_operators', 0)
                    available = operators.get('available', 0)
                    degraded = operators.get('degraded', 0)
                    
                    health['operator_health'] = {
                        'total': total,
                        'available': available,
                        'degraded': degraded,
                        'availability_percentage': (available / total * 100) if total > 0 else 0
                    }
                    
                    if degraded > 0:
                        health['cluster_status'] = 'degraded'
                    elif available < total:
                        health['cluster_status'] = 'warning'
            
            return health
            
        except Exception as e:
            logger.error(f"Error analyzing cluster health: {e}")
            return {'error': str(e)}
    
    async def _analyze_node_usage(self, node_usage: Dict[str, Any], 
                                 cpu_baselines: Dict[str, float],
                                 memory_baselines: Dict[str, float]) -> Dict[str, Any]:
        """Analyze node resource usage"""
        try:
            analysis = {
                'cpu_analysis': {},
                'memory_analysis': {},
                'trends': {},
                'outliers': [],
                'resource_pressure': []
            }
            
            # Analyze CPU usage
            if 'cpu_usage' in node_usage:
                cpu_data = node_usage['cpu_usage']
                
                if 'overall_stats' in cpu_data:
                    overall = cpu_data['overall_stats']
                    analysis['cpu_analysis'] = {
                        'current_stats': overall,
                        'baseline_comparison': self._compare_with_baseline(
                            overall, cpu_baselines, 'cpu'
                        ),
                        'utilization_level': self._categorize_utilization(overall.get('mean', 0))
                    }
            
            # Analyze memory usage
            if 'memory_usage' in node_usage:
                memory_data = node_usage['memory_usage']
                
                if 'overall_stats' in memory_data:
                    overall = memory_data['overall_stats']
                    analysis['memory_analysis'] = {
                        'current_stats': overall,
                        'baseline_comparison': self._compare_with_baseline(
                            overall, memory_baselines, 'memory'
                        ),
                        'utilization_level': self._categorize_utilization(overall.get('mean', 0))
                    }
            
            # Identify resource pressure
            cpu_mean = analysis.get('cpu_analysis', {}).get('current_stats', {}).get('mean', 0)
            memory_mean = analysis.get('memory_analysis', {}).get('current_stats', {}).get('mean', 0)
            
            if cpu_mean > 85:
                analysis['resource_pressure'].append({
                    'type': 'cpu',
                    'severity': 'critical' if cpu_mean > 95 else 'warning',
                    'value': cpu_mean,
                    'message': f"High CPU utilization: {cpu_mean:.1f}%"
                })
            
            if memory_mean > 85:
                analysis['resource_pressure'].append({
                    'type': 'memory',
                    'severity': 'critical' if memory_mean > 95 else 'warning',
                    'value': memory_mean,
                    'message': f"High memory utilization: {memory_mean:.1f}%"
                })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing node usage: {e}")
            return {'error': str(e)}
    
    async def _analyze_disk_performance(self, disk_metrics: Dict[str, Any],
                                       disk_baselines: Dict[str, float]) -> Dict[str, Any]:
        """Analyze disk I/O performance"""
        try:
            analysis = {
                'read_performance': {},
                'write_performance': {},
                'bottlenecks': [],
                'efficiency_score': 0.0
            }
            
            # Analyze read performance
            if 'disk_read_metrics' in disk_metrics:
                read_data = disk_metrics['disk_read_metrics']
                
                read_analysis = {
                    'throughput': self._extract_metric_stats(read_data, 'read_bytes'),
                    'iops': self._extract_metric_stats(read_data, 'read_iops'),
                    'latency': self._extract_metric_stats(read_data, 'read_latency')
                }
                
                # Check against baselines
                read_throughput = read_analysis.get('throughput', {}).get('mean', 0) / (1024*1024)  # MB/s
                if read_throughput < disk_baselines.get('read_baseline', 100):
                    analysis['bottlenecks'].append({
                        'type': 'disk_read_throughput',
                        'severity': 'warning',
                        'current': read_throughput,
                        'baseline': disk_baselines.get('read_baseline', 100),
                        'message': f"Read throughput ({read_throughput:.1f} MB/s) below baseline"
                    })
                
                analysis['read_performance'] = read_analysis
            
            # Analyze write performance
            if 'disk_write_metrics' in disk_metrics:
                write_data = disk_metrics['disk_write_metrics']
                
                write_analysis = {
                    'throughput': self._extract_metric_stats(write_data, 'write_bytes'),
                    'iops': self._extract_metric_stats(write_data, 'write_iops'),
                    'latency': self._extract_metric_stats(write_data, 'write_latency')
                }
                
                # Check against baselines
                write_throughput = write_analysis.get('throughput', {}).get('mean', 0) / (1024*1024)  # MB/s
                if write_throughput < disk_baselines.get('write_baseline', 50):
                    analysis['bottlenecks'].append({
                        'type': 'disk_write_throughput',
                        'severity': 'warning',
                        'current': write_throughput,
                        'baseline': disk_baselines.get('write_baseline', 50),
                        'message': f"Write throughput ({write_throughput:.1f} MB/s) below baseline"
                    })
                
                analysis['write_performance'] = write_analysis
            
            # Calculate efficiency score
            analysis['efficiency_score'] = self._calculate_disk_efficiency(analysis, disk_baselines)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing disk performance: {e}")
            return {'error': str(e)}
    
    async def _analyze_network_performance(self, network_metrics: Dict[str, Any],
                                          network_baselines: Dict[str, float]) -> Dict[str, Any]:
        """Analyze network performance"""
        try:
            analysis = {
                'throughput_analysis': {},
                'packet_loss_analysis': {},
                'bottlenecks': [],
                'efficiency_score': 0.0
            }
            
            # Analyze throughput
            if 'throughput_metrics' in network_metrics:
                throughput_data = network_metrics['throughput_metrics']
                
                rx_stats = self._extract_metric_stats(throughput_data, 'rx_bytes')
                tx_stats = self._extract_metric_stats(throughput_data, 'tx_bytes')
                
                analysis['throughput_analysis'] = {
                    'rx_throughput': rx_stats,
                    'tx_throughput': tx_stats,
                    'total_throughput': {
                        'mean': (rx_stats.get('mean', 0) + tx_stats.get('mean', 0)) / (1024*1024)  # MB/s
                    }
                }
                
                # Check against baselines
                rx_mbps = rx_stats.get('mean', 0) / (1024*1024)
                tx_mbps = tx_stats.get('mean', 0) / (1024*1024)
                
                if rx_mbps < network_baselines.get('rx_baseline', 10):
                    analysis['bottlenecks'].append({
                        'type': 'network_rx_throughput',
                        'severity': 'warning',
                        'current': rx_mbps,
                        'baseline': network_baselines.get('rx_baseline', 10),
                        'message': f"RX throughput ({rx_mbps:.1f} MB/s) below baseline"
                    })
                
                if tx_mbps < network_baselines.get('tx_baseline', 10):
                    analysis['bottlenecks'].append({
                        'type': 'network_tx_throughput',
                        'severity': 'warning',
                        'current': tx_mbps,
                        'baseline': network_baselines.get('tx_baseline', 10),
                        'message': f"TX throughput ({tx_mbps:.1f} MB/s) below baseline"
                    })
            
            # Analyze packet loss
            if 'packet_loss_metrics' in network_metrics:
                packet_loss_data = network_metrics['packet_loss_metrics']
                loss_stats = self._extract_metric_stats(packet_loss_data, 'packet_loss')
                
                analysis['packet_loss_analysis'] = loss_stats
                
                if loss_stats.get('mean', 0) > network_baselines.get('packet_loss_threshold', 0.1):
                    analysis['bottlenecks'].append({
                        'type': 'packet_loss',
                        'severity': 'critical',
                        'current': loss_stats.get('mean', 0),
                        'baseline': network_baselines.get('packet_loss_threshold', 0.1),
                        'message': f"Packet loss ({loss_stats.get('mean', 0):.3f}%) above threshold"
                    })
            
            # Calculate efficiency score
            analysis['efficiency_score'] = self._calculate_network_efficiency(analysis, network_baselines)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing network performance: {e}")
            return {'error': str(e)}
    
    async def _analyze_api_performance(self, api_metrics: Dict[str, Any],
                                      api_baselines: Dict[str, float]) -> Dict[str, Any]:
        """Analyze API server performance"""
        try:
            analysis = {
                'latency_analysis': {},
                'etcd_analysis': {},
                'slow_endpoints': [],
                'efficiency_score': 0.0
            }
            
            # Analyze API server latency
            if 'api_server_latency' in api_metrics:
                latency_data = api_metrics['api_server_latency']
                
                p50_stats = self._extract_metric_stats(latency_data, 'p50_latency')
                p95_stats = self._extract_metric_stats(latency_data, 'p95_latency')
                p99_stats = self._extract_metric_stats(latency_data, 'p99_latency')
                
                analysis['latency_analysis'] = {
                    'p50': p50_stats,
                    'p95': p95_stats,
                    'p99': p99_stats
                }
                
                # Check against baselines
                if p50_stats.get('mean', 0) > api_baselines.get('p50', 100):
                    analysis['slow_endpoints'].append({
                        'type': 'api_p50_latency',
                        'severity': 'warning',
                        'current': p50_stats.get('mean', 0),
                        'baseline': api_baselines.get('p50', 100),
                        'message': f"P50 latency ({p50_stats.get('mean', 0):.1f}ms) above baseline"
                    })
                
                if p95_stats.get('mean', 0) > api_baselines.get('p95', 500):
                    analysis['slow_endpoints'].append({
                        'type': 'api_p95_latency',
                        'severity': 'warning',
                        'current': p95_stats.get('mean', 0),
                        'baseline': api_baselines.get('p95', 500),
                        'message': f"P95 latency ({p95_stats.get('mean', 0):.1f}ms) above baseline"
                    })
                
                if p99_stats.get('mean', 0) > api_baselines.get('p99', 1000):
                    analysis['slow_endpoints'].append({
                        'type': 'api_p99_latency',
                        'severity': 'critical',
                        'current': p99_stats.get('mean', 0),
                        'baseline': api_baselines.get('p99', 1000),
                        'message': f"P99 latency ({p99_stats.get('mean', 0):.1f}ms) above baseline"
                    })
            
            # Analyze etcd performance
            if 'etcd_latency' in api_metrics:
                etcd_data = api_metrics['etcd_latency']
                etcd_stats = self._extract_metric_stats(etcd_data, 'overall_stats')
                
                analysis['etcd_analysis'] = etcd_stats
                
                if etcd_stats.get('mean', 0) > 10.0:  # 10ms baseline
                    analysis['slow_endpoints'].append({
                        'type': 'etcd_latency',
                        'severity': 'critical',
                        'current': etcd_stats.get('mean', 0),
                        'baseline': 10.0,
                        'message': f"etcd latency ({etcd_stats.get('mean', 0):.1f}ms) above baseline"
                    })
            
            # Calculate efficiency score
            analysis['efficiency_score'] = self._calculate_api_efficiency(analysis, api_baselines)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing API performance: {e}")
            return {'error': str(e)}
    
    def _extract_metric_stats(self, data: Dict[str, Any], metric_key: str) -> Dict[str, float]:
        """Extract statistical data from metric results"""
        try:
            if metric_key in data and 'overall_stats' in data[metric_key]:
                return data[metric_key]['overall_stats']
            elif 'overall_stats' in data:
                return data['overall_stats']
            else:
                return {'min': 0.0, 'mean': 0.0, 'max': 0.0, 'count': 0}
        except:
            return {'min': 0.0, 'mean': 0.0, 'max': 0.0, 'count': 0}
    
    def _compare_with_baseline(self, current_stats: Dict[str, float], 
                              baselines: Dict[str, float], metric_type: str) -> Dict[str, Any]:
        """Compare current statistics with baseline values"""
        try:
            comparison = {
                'within_range': True,
                'deviations': {},
                'score': 1.0
            }
            
            mean_value = current_stats.get('mean', 0)
            min_baseline = baselines.get('min', 0)
            max_baseline = baselines.get('max', 100)
            mean_baseline = baselines.get('mean', 50)
            variance_threshold = baselines.get('variance', 10)
            
            # Check range compliance
            if mean_value < min_baseline:
                comparison['within_range'] = False
                comparison['deviations']['below_minimum'] = min_baseline - mean_value
                comparison['score'] *= 0.8
            
            if mean_value > max_baseline:
                comparison['within_range'] = False
                comparison['deviations']['above_maximum'] = mean_value - max_baseline
                comparison['score'] *= 0.6
            
            # Check variance from mean baseline
            mean_deviation = abs(mean_value - mean_baseline)
            if mean_deviation > variance_threshold:
                comparison['within_range'] = False
                comparison['deviations']['variance_exceeded'] = mean_deviation - variance_threshold
                comparison['score'] *= 0.9
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing with baseline: {e}")
            return {'within_range': False, 'error': str(e)}
    
    def _categorize_utilization(self, value: float) -> str:
        """Categorize resource utilization level"""
        if value >= 90:
            return 'critical'
        elif value >= 75:
            return 'high'
        elif value >= 50:
            return 'moderate'
        elif value >= 25:
            return 'low'
        else:
            return 'minimal'
    
    def _calculate_health_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall cluster health score (0-10)"""
        try:
            base_score = 10.0
            
            # Deduct points for critical issues
            critical_issues = len(analysis.get('critical_issues', []))
            base_score -= critical_issues * 2.0
            
            # Deduct points for warnings
            warnings = len(analysis.get('warnings', []))
            base_score -= warnings * 0.5
            
            # Deduct points for bottlenecks
            bottlenecks = len(analysis.get('bottlenecks', []))
            base_score -= bottlenecks * 0.3
            
            # Factor in resource utilization efficiency
            resource_util = analysis.get('resource_utilization', {})
            efficiency_scores = []
            
            for component, data in resource_util.items():
                if isinstance(data, dict) and 'efficiency_score' in data:
                    efficiency_scores.append(data['efficiency_score'])
            
            if efficiency_scores:
                avg_efficiency = statistics.mean(efficiency_scores)
                base_score *= avg_efficiency
            
            return max(0.0, min(10.0, base_score))
            
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 5.0  # Default middle score
    
    def _identify_critical_issues(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify critical issues from analysis"""
        critical_issues = []
        
        try:
            # Check resource utilization
            resource_util = analysis.get('resource_utilization', {})
            
            # Node resource pressure
            if 'nodes' in resource_util:
                node_data = resource_util['nodes']
                for pressure in node_data.get('resource_pressure', []):
                    if pressure.get('severity') == 'critical':
                        critical_issues.append(pressure.get('message', 'Critical resource pressure'))
            
            # Disk bottlenecks
            if 'disk' in resource_util:
                disk_data = resource_util['disk']
                for bottleneck in disk_data.get('bottlenecks', []):
                    if bottleneck.get('severity') == 'critical':
                        critical_issues.append(bottleneck.get('message', 'Critical disk bottleneck'))
            
            # Network issues
            if 'network' in resource_util:
                network_data = resource_util['network']
                for bottleneck in network_data.get('bottlenecks', []):
                    if bottleneck.get('severity') == 'critical':
                        critical_issues.append(bottleneck.get('message', 'Critical network issue'))
            
            # API performance issues
            if 'api' in resource_util:
                api_data = resource_util['api']
                for endpoint in api_data.get('slow_endpoints', []):
                    if endpoint.get('severity') == 'critical':
                        critical_issues.append(endpoint.get('message', 'Critical API performance issue'))
            
        except Exception as e:
            logger.error(f"Error identifying critical issues: {e}")
        
        return critical_issues
    
    def _calculate_efficiency_metrics(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall efficiency metrics"""
        try:
            metrics = {
                'resource_efficiency': 0.0,
                'performance_efficiency': 0.0,
                'reliability_score': 0.0,
                'optimization_potential': 0.0
            }
            
            resource_util = analysis.get('resource_utilization', {})
            
            # Calculate resource efficiency
            efficiency_scores = []
            for component, data in resource_util.items():
                if isinstance(data, dict) and 'efficiency_score' in data:
                    efficiency_scores.append(data['efficiency_score'])
            
            if efficiency_scores:
                metrics['resource_efficiency'] = statistics.mean(efficiency_scores)
            
            # Performance efficiency based on baseline compliance
            baseline_scores = []
            for component, data in resource_util.items():
                if isinstance(data, dict):
                    # Look for baseline comparison scores
                    for sub_key, sub_data in data.items():
                        if isinstance(sub_data, dict) and 'baseline_comparison' in sub_data:
                            score = sub_data['baseline_comparison'].get('score', 1.0)
                            baseline_scores.append(score)
            
            if baseline_scores:
                metrics['performance_efficiency'] = statistics.mean(baseline_scores)
            
            # Reliability score (inverse of issues)
            total_issues = len(analysis.get('critical_issues', [])) + len(analysis.get('warnings', []))
            metrics['reliability_score'] = max(0.0, 1.0 - (total_issues * 0.1))
            
            # Optimization potential
            bottleneck_count = len(analysis.get('bottlenecks', []))
            metrics['optimization_potential'] = min(1.0, bottleneck_count * 0.2)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating efficiency metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_disk_efficiency(self, analysis: Dict[str, Any], 
                                  baselines: Dict[str, float]) -> float:
        """Calculate disk I/O efficiency score"""
        try:
            score = 1.0
            
            # Factor in throughput performance
            read_perf = analysis.get('read_performance', {})
            write_perf = analysis.get('write_performance', {})
            
            read_throughput = read_perf.get('throughput', {}).get('mean', 0) / (1024*1024)
            write_throughput = write_perf.get('throughput', {}).get('mean', 0) / (1024*1024)
            
            read_baseline = baselines.get('read_baseline', 100)
            write_baseline = baselines.get('write_baseline', 50)
            
            read_efficiency = min(1.0, read_throughput / read_baseline)
            write_efficiency = min(1.0, write_throughput / write_baseline)
            
            score = (read_efficiency + write_efficiency) / 2
            
            # Factor in latency
            read_latency = read_perf.get('latency', {}).get('mean', 0)
            write_latency = write_perf.get('latency', {}).get('mean', 0)
            
            read_latency_baseline = baselines.get('read_latency_ms', 5.0)
            write_latency_baseline = baselines.get('write_latency_ms', 8.0)
            
            if read_latency > read_latency_baseline:
                score *= 0.9
            if write_latency > write_latency_baseline:
                score *= 0.9
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating disk efficiency: {e}")
            return 0.5
    
    def _calculate_network_efficiency(self, analysis: Dict[str, Any],
                                     baselines: Dict[str, float]) -> float:
        """Calculate network efficiency score"""
        try:
            score = 1.0
            
            throughput_data = analysis.get('throughput_analysis', {})
            
            # Factor in throughput efficiency
            rx_throughput = throughput_data.get('rx_throughput', {}).get('mean', 0) / (1024*1024)
            tx_throughput = throughput_data.get('tx_throughput', {}).get('mean', 0) / (1024*1024)
            
            rx_baseline = baselines.get('rx_baseline', 10)
            tx_baseline = baselines.get('tx_baseline', 10)
            
            rx_efficiency = min(1.0, rx_throughput / rx_baseline)
            tx_efficiency = min(1.0, tx_throughput / tx_baseline)
            
            score = (rx_efficiency + tx_efficiency) / 2
            
            # Factor in packet loss
            packet_loss = analysis.get('packet_loss_analysis', {}).get('mean', 0)
            loss_threshold = baselines.get('packet_loss_threshold', 0.1)
            
            if packet_loss > loss_threshold:
                score *= max(0.1, 1.0 - (packet_loss / loss_threshold))
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating network efficiency: {e}")
            return 0.5
    
    def _calculate_api_efficiency(self, analysis: Dict[str, Any],
                                 baselines: Dict[str, float]) -> float:
        """Calculate API efficiency score"""
        try:
            score = 1.0
            
            latency_data = analysis.get('latency_analysis', {})
            
            # Factor in latency performance
            p50_latency = latency_data.get('p50', {}).get('mean', 0)
            p95_latency = latency_data.get('p95', {}).get('mean', 0)
            p99_latency = latency_data.get('p99', {}).get('mean', 0)
            
            p50_baseline = baselines.get('p50', 100)
            p95_baseline = baselines.get('p95', 500)
            p99_baseline = baselines.get('p99', 1000)
            
            p50_efficiency = min(1.0, p50_baseline / max(1, p50_latency))
            p95_efficiency = min(1.0, p95_baseline / max(1, p95_latency))
            p99_efficiency = min(1.0, p99_baseline / max(1, p99_latency))
            
            score = (p50_efficiency * 0.5 + p95_efficiency * 0.3 + p99_efficiency * 0.2)
            
            # Factor in etcd performance
            etcd_latency = analysis.get('etcd_analysis', {}).get('mean', 0)
            if etcd_latency > 10.0:  # 10ms baseline
                score *= max(0.1, 10.0 / etcd_latency)
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating API efficiency: {e}")
            return 0.5
    
    async def generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        try:
            # Resource utilization recommendations
            resource_util = analysis.get('resource_utilization', {})
            
            # Node resource recommendations
            if 'nodes' in resource_util:
                node_data = resource_util['nodes']
                
                for pressure in node_data.get('resource_pressure', []):
                    if pressure.get('type') == 'cpu' and pressure.get('severity') in ['critical', 'warning']:
                        recommendations.append("Consider horizontal pod autoscaling or vertical pod autoscaling for CPU-intensive workloads")
                        recommendations.append("Review and optimize CPU resource requests and limits for pods")
                        
                    if pressure.get('type') == 'memory' and pressure.get('severity') in ['critical', 'warning']:
                        recommendations.append("Review memory resource requests and limits to prevent memory pressure")
                        recommendations.append("Consider adding more worker nodes or scaling existing nodes")
            
            # Disk I/O recommendations
            if 'disk' in resource_util:
                disk_data = resource_util['disk']
                
                for bottleneck in disk_data.get('bottlenecks', []):
                    if 'read_throughput' in bottleneck.get('type', ''):
                        recommendations.append("Consider upgrading to faster storage (SSD) or implementing read caching")
                        recommendations.append("Review disk I/O patterns and optimize data access patterns")
                    
                    if 'write_throughput' in bottleneck.get('type', ''):
                        recommendations.append("Implement write buffering or batching for write-heavy applications")
                        recommendations.append("Consider storage optimization and defragmentation")
            
            # Network recommendations
            if 'network' in resource_util:
                network_data = resource_util['network']
                
                for bottleneck in network_data.get('bottlenecks', []):
                    if 'throughput' in bottleneck.get('type', ''):
                        recommendations.append("Review network bandwidth allocation and consider upgrading network infrastructure")
                        recommendations.append("Implement traffic shaping and load balancing optimization")
                    
                    if 'packet_loss' in bottleneck.get('type', ''):
                        recommendations.append("Investigate network configuration and hardware issues causing packet loss")
                        recommendations.append("Implement network monitoring and alerting for proactive issue detection")
            
            # API performance recommendations
            if 'api' in resource_util:
                api_data = resource_util['api']
                
                if api_data.get('slow_endpoints'):
                    recommendations.append("Optimize API server configuration and resource allocation")
                    recommendations.append("Review etcd performance and consider etcd cluster optimization")
                    recommendations.append("Implement API request caching and rate limiting")
            
            # General efficiency recommendations
            efficiency_metrics = analysis.get('efficiency_metrics', {})
            resource_efficiency = efficiency_metrics.get('resource_efficiency', 1.0)
            
            if resource_efficiency < 0.7:
                recommendations.append("Implement comprehensive resource monitoring and alerting")
                recommendations.append("Regular performance tuning and capacity planning reviews")
                recommendations.append("Consider workload optimization and rightsizing initiatives")
            
            # Health score recommendations
            health_score = analysis.get('cluster_health_score', 10.0)
            
            if health_score < 7.0:
                recommendations.append("Conduct thorough cluster health assessment and remediation")
                recommendations.append("Implement proactive monitoring and maintenance procedures")
            
            # Remove duplicates and return top recommendations
            unique_recommendations = list(dict.fromkeys(recommendations))
            return unique_recommendations[:15]  # Top 15 recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return [f"Error generating recommendations: {str(e)}"]