"""Performance analysis module for OpenShift benchmark data."""
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.ocp_benchmark_config import config_manager


logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Advanced performance analysis for OpenShift metrics."""
    
    def __init__(self):
        self.config = config_manager
        self.thresholds = self.config.get_thresholds()
        self.cpu_baselines = self.config.get_cpu_baselines()
        self.memory_baselines = self.config.get_memory_baselines()
        self.disk_baselines = self.config.get_disk_baselines()
        self.network_baselines = self.config.get_network_baselines()
        self.api_baselines = self.config.get_api_baselines()
    
    def analyze_trend(self, values: List[float], timestamps: List[str] = None) -> Dict[str, Any]:
        """Analyze trend in time series data."""
        if len(values) < 2:
            return {"trend": "insufficient_data", "slope": 0, "correlation": 0}
        
        # Convert to numpy array for analysis
        y = np.array(values)
        x = np.arange(len(values))
        
        # Calculate linear regression
        if len(values) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            correlation = np.corrcoef(x, y)[0, 1] if len(values) > 2 else 0
        else:
            slope, correlation = 0, 0
        
        # Determine trend direction
        if abs(slope) < 0.01:  # Nearly flat
            trend = "stable"
        elif slope > 0:
            if slope > 0.1:
                trend = "increasing_rapidly"
            else:
                trend = "increasing_slowly"
        else:
            if slope < -0.1:
                trend = "decreasing_rapidly"
            else:
                trend = "decreasing_slowly"
        
        # Calculate volatility (standard deviation)
        volatility = np.std(y) if len(y) > 1 else 0
        
        return {
            "trend": trend,
            "slope": float(slope),
            "correlation": float(correlation) if not np.isnan(correlation) else 0,
            "volatility": float(volatility),
            "trend_strength": abs(float(correlation)) if not np.isnan(correlation) else 0
        }
    
    def detect_anomalies(self, values: List[float], threshold_std: float = 2.0) -> List[int]:
        """Detect anomalies using standard deviation method."""
        if len(values) < 3:
            return []
        
        values_array = np.array(values)
        mean = np.mean(values_array)
        std = np.std(values_array)
        
        if std == 0:
            return []
        
        # Find values that are more than threshold_std standard deviations from mean
        z_scores = np.abs((values_array - mean) / std)
        anomaly_indices = np.where(z_scores > threshold_std)[0].tolist()
        
        return anomaly_indices
    
    def calculate_resource_efficiency(self, usage_stats: Dict[str, float], 
                                    capacity: float = 100.0) -> Dict[str, Any]:
        """Calculate resource efficiency metrics."""
        if not usage_stats or 'mean' not in usage_stats:
            return {"efficiency": 0, "utilization": 0, "waste": 0}
        
        mean_usage = usage_stats['mean']
        max_usage = usage_stats.get('max', mean_usage)
        
        # Calculate efficiency (how well resources are utilized)
        utilization = (mean_usage / capacity) * 100
        peak_utilization = (max_usage / capacity) * 100
        
        # Calculate waste (unused capacity)
        waste = capacity - mean_usage
        
        # Efficiency score (penalize both under and over utilization)
        if utilization < 20:
            efficiency_score = utilization / 20 * 50  # Under-utilized
        elif utilization > 80:
            efficiency_score = 100 - ((utilization - 80) / 20 * 30)  # Over-utilized
        else:
            efficiency_score = 50 + ((utilization - 20) / 60 * 50)  # Good range
        
        return {
            "efficiency_score": max(0, min(100, efficiency_score)),
            "utilization_percent": utilization,
            "peak_utilization_percent": peak_utilization,
            "waste_percent": (waste / capacity) * 100,
            "capacity": capacity
        }
    
    def analyze_cluster_health(self, cluster_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive cluster health analysis."""
        health_score = 100
        issues = []
        recommendations = []
        
        # Analyze different aspects of cluster health
        node_health = self._analyze_node_health(cluster_data)
        performance_health = self._analyze_performance_health(cluster_data)
        capacity_health = self._analyze_capacity_health(cluster_data)
        
        # Combine health scores
        health_components = {
            "node_health": node_health,
            "performance_health": performance_health,
            "capacity_health": capacity_health
        }
        
        # Calculate overall health score
        component_scores = [comp.get('score', 0) for comp in health_components.values()]
        overall_score = np.mean(component_scores) if component_scores else 0
        
        # Collect issues and recommendations
        for component in health_components.values():
            issues.extend(component.get('issues', []))
            recommendations.extend(component.get('recommendations', []))
        
        # Determine health status
        if overall_score >= 90:
            status = "excellent"
        elif overall_score >= 75:
            status = "good"
        elif overall_score >= 60:
            status = "fair"
        elif overall_score >= 40:
            status = "poor"
        else:
            status = "critical"
        
        return {
            "overall_health_score": round(overall_score, 1),
            "health_status": status,
            "component_scores": {k: v.get('score', 0) for k, v in health_components.items()},
            "issues": issues,
            "recommendations": list(set(recommendations)),  # Remove duplicates
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _analyze_node_health(self, cluster_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze node health."""
        score = 100
        issues = []
        recommendations = []
        
        # Check if we have node data
        if 'cluster_summary' in cluster_data:
            summary = cluster_data['cluster_summary']
            total_nodes = summary.get('total_nodes', 0)
            ready_nodes = summary.get('ready_nodes', 0)
            
            if total_nodes > 0:
                ready_percentage = (ready_nodes / total_nodes) * 100
                if ready_percentage < 100:
                    score -= (100 - ready_percentage) * 0.5  # Penalty for not ready nodes
                    issues.append(f"{total_nodes - ready_nodes} out of {total_nodes} nodes are not ready")
                    recommendations.append("Investigate non-ready nodes and resolve issues")
            else:
                score = 0
                issues.append("No node information available")
        
        # Check node roles distribution
        if 'nodes_by_role' in cluster_data:
            roles = cluster_data['nodes_by_role']
            if 'master' not in roles or roles['master']['count'] < 3:
                score -= 10
                issues.append("Less than 3 master nodes detected - high availability at risk")
                recommendations.append("Consider adding more master nodes for high availability")
        
        return {
            "score": max(0, score),
            "issues": issues,
            "recommendations": recommendations
        }
    
    def _analyze_performance_health(self, cluster_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance health."""
        score = 100
        issues = []
        recommendations = []
        
        # Check for performance analysis in data
        if 'performance_analysis' in cluster_data:
            perf_analysis = cluster_data['performance_analysis']
            
            # Check overall status
            status = perf_analysis.get('overall_status', 'normal')
            if status == 'critical':
                score -= 40
                issues.append("Critical performance issues detected")
                recommendations.append("Immediate attention required for critical performance issues")
            elif status == 'degraded':
                score -= 20
                issues.append("Performance degradation detected")
                recommendations.append("Investigate and resolve performance degradation")
            
            # Check alerts
            alerts = perf_analysis.get('alerts', [])
            for alert in alerts:
                severity = alert.get('severity', 'info')
                if severity == 'critical':
                    score -= 15
                elif severity == 'warning':
                    score -= 5
                
                issues.append(f"{severity.title()}: {alert.get('type', 'Unknown issue')}")
        
        # Check baseline comparisons
        if 'baseline_comparison' in cluster_data:
            baseline_comp = cluster_data['baseline_comparison']
            
            for resource, comparison in baseline_comp.items():
                if isinstance(comparison, dict) and 'status' in comparison:
                    if comparison['status'] == 'critical':
                        score -= 15
                        issues.append(f"{resource.title()} usage is critical")
                        recommendations.append(f"Scale up {resource} resources or optimize usage")
                    elif comparison['status'] == 'warning':
                        score -= 8
                        issues.append(f"{resource.title()} usage approaching limits")
                        recommendations.append(f"Monitor {resource} usage and prepare for scaling")
        
        return {
            "score": max(0, score),
            "issues": issues,
            "recommendations": recommendations
        }
    
    def _analyze_capacity_health(self, cluster_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze capacity and resource planning health."""
        score = 100
        issues = []
        recommendations = []
        
        # Check cluster statistics
        if 'cluster_statistics' in cluster_data:
            stats = cluster_data['cluster_statistics']
            
            # CPU capacity analysis
            cpu_stats = stats.get('cpu_usage', {})
            if cpu_stats.get('mean', 0) > 80:
                score -= 20
                issues.append(f"High average CPU usage: {cpu_stats.get('mean', 0):.1f}%")
                recommendations.append("Consider adding more CPU resources or optimizing workloads")
            elif cpu_stats.get('mean', 0) > 70:
                score -= 10
                issues.append(f"Elevated CPU usage: {cpu_stats.get('mean', 0):.1f}%")
                recommendations.append("Monitor CPU trends and prepare for capacity expansion")
            
            # Memory capacity analysis
            memory_stats = stats.get('memory_usage', {})
            if memory_stats.get('mean', 0) > 85:
                score -= 20
                issues.append(f"High average memory usage: {memory_stats.get('mean', 0):.1f}%")
                recommendations.append("Consider adding more memory or optimizing memory usage")
            elif memory_stats.get('mean', 0) > 75:
                score -= 10
                issues.append(f"Elevated memory usage: {memory_stats.get('mean', 0):.1f}%")
                recommendations.append("Monitor memory trends and prepare for capacity expansion")
        
        # Check for resource efficiency
        if 'nodes' in cluster_data:
            nodes = cluster_data['nodes']
            low_efficiency_nodes = 0
            
            for node_name, node_data in nodes.items():
                cpu_usage = node_data.get('cpu_usage', {}).get('statistics', {})
                memory_usage = node_data.get('memory_usage', {}).get('statistics', {})
                
                # Check for very low utilization (waste)
                if cpu_usage.get('mean', 0) < 10 and memory_usage.get('mean', 0) < 20:
                    low_efficiency_nodes += 1
            
            if low_efficiency_nodes > 0:
                score -= min(20, low_efficiency_nodes * 5)
                issues.append(f"{low_efficiency_nodes} nodes with very low resource utilization")
                recommendations.append("Consider consolidating workloads or rightsizing cluster")
        
        return {
            "score": max(0, score),
            "issues": issues,
            "recommendations": recommendations
        }
    
    def generate_capacity_forecast(self, usage_data: Dict[str, Any], 
                                 forecast_days: int = 30) -> Dict[str, Any]:
        """Generate capacity forecast based on usage trends."""
        forecasts = {}
        
        # Analyze CPU trends
        if 'cluster_statistics' in usage_data:
            stats = usage_data['cluster_statistics']
            
            # Extract time series data if available
            cpu_forecast = self._forecast_metric(
                stats.get('cpu_usage', {}), 
                'cpu', 
                forecast_days
            )
            memory_forecast = self._forecast_metric(
                stats.get('memory_usage', {}), 
                'memory', 
                forecast_days
            )
            
            forecasts.update({
                'cpu': cpu_forecast,
                'memory': memory_forecast
            })
        
        return {
            'forecasts': forecasts,
            'forecast_period_days': forecast_days,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'confidence': self._calculate_forecast_confidence(forecasts)
        }
    
    def _forecast_metric(self, metric_stats: Dict[str, Any], 
                        metric_name: str, 
                        forecast_days: int) -> Dict[str, Any]:
        """Forecast a single metric."""
        if not metric_stats or 'mean' not in metric_stats:
            return {
                'current_usage': 0,
                'predicted_usage': 0,
                'trend': 'insufficient_data',
                'risk_level': 'unknown'
            }
        
        current_usage = metric_stats.get('mean', 0)
        max_usage = metric_stats.get('max', current_usage)
        
        # Simple linear growth assumption (could be enhanced with ML)
        # Assume 5% monthly growth rate for CPU, 3% for memory
        growth_rates = {'cpu': 0.05, 'memory': 0.03}
        monthly_growth = growth_rates.get(metric_name, 0.04)
        daily_growth = monthly_growth / 30
        
        # Calculate predicted usage
        predicted_usage = current_usage * (1 + daily_growth * forecast_days)
        
        # Determine risk level
        thresholds = self.thresholds.get(metric_name, {'warning': 75, 'critical': 90})
        
        if predicted_usage >= thresholds['critical']:
            risk_level = 'high'
        elif predicted_usage >= thresholds['warning']:
            risk_level = 'medium'
        elif predicted_usage >= thresholds['warning'] * 0.8:
            risk_level = 'low'
        else:
            risk_level = 'minimal'
        
        # Estimate days until threshold
        days_to_warning = self._calculate_days_to_threshold(
            current_usage, thresholds['warning'], daily_growth
        )
        days_to_critical = self._calculate_days_to_threshold(
            current_usage, thresholds['critical'], daily_growth
        )
        
        return {
            'current_usage': round(current_usage, 2),
            'predicted_usage': round(predicted_usage, 2),
            'growth_rate_daily': round(daily_growth * 100, 3),
            'days_to_warning': days_to_warning,
            'days_to_critical': days_to_critical,
            'risk_level': risk_level,
            'max_observed': round(max_usage, 2)
        }
    
    def _calculate_days_to_threshold(self, current: float, threshold: float, 
                                   daily_growth: float) -> Optional[int]:
        """Calculate days until a threshold is reached."""
        if daily_growth <= 0 or current >= threshold:
            return None
        
        # Exponential growth formula: future = current * (1 + rate)^days
        # Solve for days: days = log(threshold/current) / log(1 + rate)
        try:
            import math
            days = math.log(threshold / current) / math.log(1 + daily_growth)
            return int(days) if days > 0 else None
        except (ValueError, ZeroDivisionError):
            return None
    
    def _calculate_forecast_confidence(self, forecasts: Dict[str, Any]) -> str:
        """Calculate confidence level for forecasts."""
        # Simple heuristic based on data availability
        if not forecasts:
            return 'low'
        
        data_points = len([f for f in forecasts.values() if f.get('current_usage', 0) > 0])
        
        if data_points >= 2:
            return 'high'
        elif data_points >= 1:
            return 'medium'
        else:
            return 'low'
    
    def generate_optimization_recommendations(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific optimization recommendations."""
        recommendations = []
        
        # CPU optimization recommendations
        if 'cluster_statistics' in analysis_data:
            stats = analysis_data['cluster_statistics']
            cpu_stats = stats.get('cpu_usage', {})
            memory_stats = stats.get('memory_usage', {})
            
            # High CPU usage recommendations
            if cpu_stats.get('mean', 0) > 80:
                recommendations.append({
                    'category': 'CPU Optimization',
                    'priority': 'high',
                    'issue': f"High CPU usage: {cpu_stats.get('mean', 0):.1f}%",
                    'recommendation': 'Scale horizontally by adding more nodes or vertically by upgrading CPU',
                    'impact': 'Improves response times and prevents CPU throttling',
                    'effort': 'medium'
                })
            
            elif cpu_stats.get('mean', 0) < 30:
                recommendations.append({
                    'category': 'CPU Optimization',
                    'priority': 'medium',
                    'issue': f"Low CPU utilization: {cpu_stats.get('mean', 0):.1f}%",
                    'recommendation': 'Consider consolidating workloads or reducing cluster size',
                    'impact': 'Reduces costs while maintaining performance',
                    'effort': 'medium'
                })
            
            # Memory optimization recommendations  
            if memory_stats.get('mean', 0) > 85:
                recommendations.append({
                    'category': 'Memory Optimization',
                    'priority': 'high',
                    'issue': f"High memory usage: {memory_stats.get('mean', 0):.1f}%",
                    'recommendation': 'Add more memory or optimize memory-intensive applications',
                    'impact': 'Prevents OOM kills and improves stability',
                    'effort': 'medium'
                })
        
        # Performance-based recommendations
        if 'performance_analysis' in analysis_data:
            perf_analysis = analysis_data['performance_analysis']
            alerts = perf_analysis.get('alerts', [])
            
            for alert in alerts:
                if alert.get('type') == 'high_read_latency':
                    recommendations.append({
                        'category': 'Storage Optimization',
                        'priority': 'high',
                        'issue': 'High disk read latency detected',
                        'recommendation': 'Consider upgrading to faster storage (NVMe SSD) or optimize I/O patterns',
                        'impact': 'Improves application performance and user experience',
                        'effort': 'high'
                    })
                
                elif alert.get('type') == 'network_errors_detected':
                    recommendations.append({
                        'category': 'Network Optimization',
                        'priority': 'high',
                        'issue': 'Network errors detected',
                        'recommendation': 'Investigate network hardware and configuration issues',
                        'impact': 'Improves reliability and prevents data loss',
                        'effort': 'medium'
                    })
        
        # Node-specific recommendations
        if 'nodes' in analysis_data:
            nodes = analysis_data['nodes']
            unbalanced_nodes = []
            
            for node_name, node_data in nodes.items():
                cpu_mean = node_data.get('cpu_usage', {}).get('statistics', {}).get('mean', 0)
                memory_mean = node_data.get('memory_usage', {}).get('statistics', {}).get('mean', 0)
                
                # Check for resource imbalance
                if abs(cpu_mean - memory_mean) > 40:  # More than 40% difference
                    unbalanced_nodes.append(node_name)
            
            if unbalanced_nodes:
                recommendations.append({
                    'category': 'Resource Balancing',
                    'priority': 'medium',
                    'issue': f'Resource imbalance detected on {len(unbalanced_nodes)} nodes',
                    'recommendation': 'Review pod placement and resource requests/limits to balance CPU and memory usage',
                    'impact': 'Improves overall resource utilization efficiency',
                    'effort': 'low'
                })
        
        return recommendations
    
    def compare_with_industry_benchmarks(self, cluster_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare cluster performance with industry benchmarks."""
        benchmarks = {
            'cpu_utilization': {'excellent': 60-75, 'good': 45-60, 'average': 30-45, 'poor': '<30 or >80'},
            'memory_utilization': {'excellent': 65-80, 'good': 50-65, 'average': 35-50, 'poor': '<35 or >85'},
            'api_latency_p99': {'excellent': '<100ms', 'good': '100-300ms', 'average': '300-500ms', 'poor': '>500ms'},
            'disk_latency': {'excellent': '<5ms', 'good': '5-10ms', 'average': '10-20ms', 'poor': '>20ms'},
            'network_throughput': {'excellent': '>80% capacity', 'good': '60-80%', 'average': '40-60%', 'poor': '<40%'}
        }
        
        results = {}
        
        # Compare CPU utilization
        if 'cluster_statistics' in cluster_data:
            stats = cluster_data['cluster_statistics']
            cpu_mean = stats.get('cpu_usage', {}).get('mean', 0)
            memory_mean = stats.get('memory_usage', {}).get('mean', 0)
            
            results['cpu_utilization'] = self._rate_against_benchmark(cpu_mean, 'cpu_utilization')
            results['memory_utilization'] = self._rate_against_benchmark(memory_mean, 'memory_utilization')
        
        # Compare API latency if available
        if 'api_server_latency' in cluster_data:
            api_stats = cluster_data['api_server_latency'].get('cluster_summary', {}).get('p99_overall', {})
            p99_latency = api_stats.get('mean', 0)
            results['api_latency'] = self._rate_against_benchmark(p99_latency, 'api_latency_p99')
        
        return {
            'benchmark_comparison': results,
            'overall_rating': self._calculate_overall_rating(results),
            'benchmark_date': datetime.now(timezone.utc).isoformat()
        }
    
    def _rate_against_benchmark(self, value: float, metric_type: str) -> Dict[str, Any]:
        """Rate a metric value against industry benchmarks."""
        # Simple rating logic (could be enhanced with more sophisticated benchmarks)
        if metric_type == 'cpu_utilization':
            if 60 <= value <= 75:
                rating = 'excellent'
            elif 45 <= value < 60 or 75 < value <= 80:
                rating = 'good'
            elif 30 <= value < 45:
                rating = 'average'
            else:
                rating = 'poor'
        elif metric_type == 'memory_utilization':
            if 65 <= value <= 80:
                rating = 'excellent'
            elif 50 <= value < 65:
                rating = 'good'
            elif 35 <= value < 50:
                rating = 'average'
            else:
                rating = 'poor'
        elif metric_type == 'api_latency_p99':
            if value < 100:
                rating = 'excellent'
            elif value < 300:
                rating = 'good'
            elif value < 500:
                rating = 'average'
            else:
                rating = 'poor'
        else:
            rating = 'unknown'
        
        return {
            'current_value': value,
            'rating': rating,
            'metric_type': metric_type
        }
    
    def _calculate_overall_rating(self, benchmark_results: Dict[str, Dict[str, Any]]) -> str:
        """Calculate overall benchmark rating."""
        if not benchmark_results:
            return 'unknown'
        
        rating_scores = {'excellent': 4, 'good': 3, 'average': 2, 'poor': 1, 'unknown': 0}
        
        total_score = sum(rating_scores.get(result.get('rating', 'unknown'), 0) 
                         for result in benchmark_results.values())
        avg_score = total_score / len(benchmark_results)
        
        if avg_score >= 3.5:
            return 'excellent'
        elif avg_score >= 2.5:
            return 'good'
        elif avg_score >= 1.5:
            return 'average'
        else:
            return 'poor'


def analyze_comprehensive_performance(data: Dict[str, Any]) -> str:
    """Comprehensive performance analysis entry point."""
    analyzer = PerformanceAnalyzer()
    
    try:
        # Perform various analyses
        health_analysis = analyzer.analyze_cluster_health(data)
        
        # Capacity forecast
        capacity_forecast = analyzer.generate_capacity_forecast(data)
        
        # Optimization recommendations
        optimization_recs = analyzer.generate_optimization_recommendations(data)
        
        # Industry benchmarks comparison
        benchmark_comparison = analyzer.compare_with_industry_benchmarks(data)
        
        # Combine results
        comprehensive_analysis = {
            'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
            'health_analysis': health_analysis,
            'capacity_forecast': capacity_forecast,
            'optimization_recommendations': optimization_recs,
            'benchmark_comparison': benchmark_comparison,
            'summary': {
                'overall_health': health_analysis.get('health_status', 'unknown'),
                'health_score': health_analysis.get('overall_health_score', 0),
                'total_recommendations': len(optimization_recs),
                'high_priority_recommendations': len([r for r in optimization_recs if r.get('priority') == 'high']),
                'benchmark_rating': benchmark_comparison.get('overall_rating', 'unknown')
            }
        }
        
        return json.dumps(comprehensive_analysis, indent=2)
    
    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {e}")
        return json.dumps({
            'error': f'Analysis failed: {str(e)}',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })


# Entry point for external usage
def analyze_performance_data(data: Dict[str, Any], analysis_type: str = "comprehensive") -> str:
    """Main entry point for performance analysis."""
    if analysis_type == "comprehensive":
        return analyze_comprehensive_performance(data)
    else:
        # Basic analysis fallback
        analyzer = PerformanceAnalyzer()
        health_analysis = analyzer.analyze_cluster_health(data)
        return json.dumps(health_analysis, indent=2)