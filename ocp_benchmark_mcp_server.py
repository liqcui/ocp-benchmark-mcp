#!/usr/bin/env python3
"""OpenShift Benchmark MCP Server"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import fastmcp
from fastmcp import FastMCP, Context
from pydantic import BaseModel, Field

# Import our modules
from config.ocp_benchmark_config import config
from ocauth.ocp_benchmark_auth import auth
from tools.ocp_benchmark_openshift_clusterinfo import get_cluster_info_json
from tools.ocp_benchmark_openshift_nodeinfo import get_nodes_info_json, get_nodes_summary_json
from tools.ocp_benchmark_prometheus_nodes_usage import get_node_usage_json
from tools.ocp_benchmark_prometheus_pods_usage import get_pod_usage_json, get_top_pods_json
from tools.ocp_benchmark_prometheus_diskio import get_disk_metrics_json, get_disk_utilization_json
from tools.ocp_benchmark_prometheus_network import get_network_metrics_json, get_network_utilization_json
from tools.ocp_benchmark_prometheus_apilatency import get_api_latency_json, get_slow_api_requests_json

# Set timezone to UTC
os.environ['TZ'] = 'UTC'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S UTC'
)
logger = logging.getLogger(__name__)

# Pydantic models for MCP tools
class BaseRequest(BaseModel):
    """Base request model with common configuration"""
    
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


class ClusterInfoRequest(BaseRequest):
    """Request model for cluster information"""
    include_operators: Optional[bool] = Field(default=True, description="Include cluster operators status")


class NodeInfoRequest(BaseRequest):
    """Request model for node information"""
    summary_only: Optional[bool] = Field(default=False, description="Return summary only instead of detailed info")
    role_filter: Optional[str] = Field(default=None, description="Filter nodes by role (master, worker, infra)")


class NodeUsageRequest(BaseRequest):
    """Request model for node usage metrics"""
    duration_hours: Optional[int] = Field(default=1, description="Time duration in hours for metrics collection")


class PodUsageRequest(BaseRequest):
    """Request model for pod usage metrics"""
    duration_hours: Optional[int] = Field(default=1, description="Time duration in hours for metrics collection")
    pod_patterns: Optional[List[str]] = Field(default=None, description="Regex patterns to filter pod names")
    label_selectors: Optional[List[str]] = Field(default=None, description="Label selectors to filter pods")
    top_n: Optional[int] = Field(default=10, description="Number of top resource consuming pods to return")


class DiskMetricsRequest(BaseRequest):
    """Request model for disk I/O metrics"""
    duration_hours: Optional[int] = Field(default=1, description="Time duration in hours for metrics collection")
    by_device: Optional[bool] = Field(default=False, description="Break down metrics by device")


class NetworkMetricsRequest(BaseRequest):
    """Request model for network metrics"""
    duration_hours: Optional[int] = Field(default=1, description="Time duration in hours for metrics collection")
    by_interface: Optional[bool] = Field(default=False, description="Break down metrics by network interface")


class APILatencyRequest(BaseRequest):
    """Request model for API latency metrics"""
    duration_hours: Optional[int] = Field(default=1, description="Time duration in hours for metrics collection")
    slow_threshold_ms: Optional[float] = Field(default=1000.0, description="Threshold in ms for identifying slow requests")
    include_slow_requests: Optional[bool] = Field(default=False, description="Include detailed slow request analysis")


# Initialize FastMCP server
mcp = FastMCP("OpenShift Benchmark MCP Server")


@mcp.tool()
async def get_cluster_info(ctx: Context, request: ClusterInfoRequest) -> str:
    """Get OpenShift cluster information including version, name, and infrastructure details.
    
    Returns comprehensive cluster information including:
    - Cluster version and update status
    - Cluster name and infrastructure details
    - Platform information (AWS, Azure, GCP, etc.)
    - API server URLs
    - Cluster operators status (if requested)
    """
    try:
        logger.info("Collecting cluster information")
        result = await get_cluster_info_json()
        logger.info("Successfully collected cluster information")
        return result
    except Exception as e:
        logger.error(f"Error getting cluster info: {e}")
        return f'{{"error": "Failed to collect cluster information: {str(e)}"}}'


@mcp.tool()
async def get_node_info(ctx: Context, request: NodeInfoRequest) -> str:
    """Get detailed information about OpenShift cluster nodes.
    
    Returns node information including:
    - Node count by role (master, worker, infra)
    - Instance types and resource allocation
    - CPU cores and RAM capacity per node
    - Node conditions and readiness status
    - Resource summary by node role
    """
    try:
        logger.info(f"Collecting node information (summary_only: {request.summary_only})")
        
        if request.summary_only:
            result = await get_nodes_summary_json()
        else:
            result = await get_nodes_info_json()
        
        logger.info("Successfully collected node information")
        return result
    except Exception as e:
        logger.error(f"Error getting node info: {e}")
        return f'{{"error": "Failed to collect node information: {str(e)}"}}'


@mcp.tool()
async def get_node_usage_metrics(ctx: Context, request: NodeUsageRequest) -> str:
    """Get CPU and memory usage metrics for cluster nodes from Prometheus.
    
    Returns usage statistics including:
    - Min, mean, max CPU usage percentage per node
    - Min, mean, max memory usage percentage per node
    - Overall cluster resource utilization
    - Comparison with baseline thresholds
    - Performance trend analysis
    """
    try:
        logger.info(f"Collecting node usage metrics for {request.duration_hours} hours")
        result = await get_node_usage_json(request.duration_hours)
        logger.info("Successfully collected node usage metrics")
        return result
    except Exception as e:
        logger.error(f"Error getting node usage: {e}")
        return f'{{"error": "Failed to collect node usage metrics: {str(e)}"}}'


@mcp.tool()
async def get_pod_usage_metrics(ctx: Context, request: PodUsageRequest) -> str:
    """Get CPU and memory usage metrics for pods from Prometheus.
    
    Returns pod usage statistics including:
    - Resource usage by individual pods
    - Filtering by pod name patterns or labels
    - Top resource-consuming pods
    - Usage trends over time
    - Namespace-level aggregation
    """
    try:
        logger.info(f"Collecting pod usage metrics for {request.duration_hours} hours")
        
        if request.top_n and request.top_n > 0:
            # Get top resource consuming pods
            result = await get_top_pods_json(request.duration_hours, request.top_n)
        else:
            # Get filtered pod usage
            result = await get_pod_usage_json(
                request.pod_patterns, 
                request.label_selectors, 
                request.duration_hours
            )
        
        logger.info("Successfully collected pod usage metrics")
        return result
    except Exception as e:
        logger.error(f"Error getting pod usage: {e}")
        return f'{{"error": "Failed to collect pod usage metrics: {str(e)}"}}'


@mcp.tool()
async def get_disk_io_metrics(ctx: Context, request: DiskMetricsRequest) -> str:
    """Get disk I/O performance metrics from Prometheus.
    
    Returns disk performance data including:
    - Read/write throughput (MB/s)
    - IOPS (Input/Output Operations Per Second)
    - Average and peak latency measurements
    - Per-device breakdown (if requested)
    - Comparison with baseline performance
    """
    try:
        logger.info(f"Collecting disk I/O metrics for {request.duration_hours} hours")
        
        if request.by_device:
            result = await get_disk_utilization_json(request.duration_hours)
        else:
            result = await get_disk_metrics_json(request.duration_hours)
        
        logger.info("Successfully collected disk I/O metrics")
        return result
    except Exception as e:
        logger.error(f"Error getting disk metrics: {e}")
        return f'{{"error": "Failed to collect disk I/O metrics: {str(e)}"}}'


@mcp.tool()
async def get_network_metrics(ctx: Context, request: NetworkMetricsRequest) -> str:
    """Get network performance metrics from Prometheus.
    
    Returns network performance data including:
    - Network throughput (RX/TX bytes and packets per second)
    - Packet loss rates
    - Network latency measurements
    - Per-interface breakdown (if requested)
    - Bandwidth utilization analysis
    """
    try:
        logger.info(f"Collecting network metrics for {request.duration_hours} hours")
        
        if request.by_interface:
            result = await get_network_utilization_json(request.duration_hours)
        else:
            result = await get_network_metrics_json(request.duration_hours)
        
        logger.info("Successfully collected network metrics")
        return result
    except Exception as e:
        logger.error(f"Error getting network metrics: {e}")
        return f'{{"error": "Failed to collect network metrics: {str(e)}"}}'


@mcp.tool()
async def get_api_latency_metrics(ctx: Context, request: APILatencyRequest) -> str:
    """Get Kubernetes API server latency metrics from Prometheus.
    
    Returns API performance data including:
    - P50, P95, P99 latency percentiles
    - Latency breakdown by HTTP verb and resource type
    - etcd response time metrics
    - Slow request identification
    - API server performance trends
    """
    try:
        logger.info(f"Collecting API latency metrics for {request.duration_hours} hours")
        
        if request.include_slow_requests:
            result = await get_slow_api_requests_json(request.duration_hours, request.slow_threshold_ms)
        else:
            result = await get_api_latency_json(request.duration_hours)
        
        logger.info("Successfully collected API latency metrics")
        return result
    except Exception as e:
        logger.error(f"Error getting API latency: {e}")
        return f'{{"error": "Failed to collect API latency metrics: {str(e)}"}}'


@mcp.tool()
async def get_baseline_configuration() -> str:
    """Get the current baseline configuration and thresholds.
    
    Returns the complete baseline configuration including:
    - CPU, memory, disk, and network performance baselines
    - Warning and critical thresholds
    - Performance variance acceptable ranges
    - Measurement units and expected values
    """
    try:
        logger.info("Retrieving baseline configuration")
        
        baseline_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'cpu_baselines': config.get_cpu_baselines(),
            'memory_baselines': config.get_memory_baselines(),
            'disk_baselines': config.get_disk_baselines(),
            'network_baselines': config.get_network_baselines(),
            'api_baselines': config.get_api_baselines(),
            'thresholds': config.get_thresholds()
        }
        
        import json
        return json.dumps(baseline_data, indent=2)
    except Exception as e:
        logger.error(f"Error getting baseline configuration: {e}")
        return f'{{"error": "Failed to retrieve baseline configuration: {str(e)}"}}'


@mcp.tool()
async def get_comprehensive_performance_report(ctx: Context, duration_hours: int = 1) -> str:
    """Generate a comprehensive performance report across all metrics.
    
    This tool collects and aggregates data from all monitoring sources:
    - Cluster and node information
    - Resource usage metrics (CPU, memory)
    - Disk I/O performance
    - Network throughput and latency
    - API server response times
    - Comparison with baseline values
    """
    try:
        logger.info(f"Generating comprehensive performance report for {duration_hours} hours")
        
        # Collect all metrics concurrently
        tasks = [
            get_cluster_info_json(),
            get_nodes_summary_json(),
            get_node_usage_json(duration_hours),
            get_disk_metrics_json(duration_hours),
            get_network_metrics_json(duration_hours),
            get_api_latency_json(duration_hours)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Parse results
        import json
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'duration_hours': duration_hours,
            'cluster_info': json.loads(results[0]) if not isinstance(results[0], Exception) else {'error': str(results[0])},
            'node_summary': json.loads(results[1]) if not isinstance(results[1], Exception) else {'error': str(results[1])},
            'node_usage': json.loads(results[2]) if not isinstance(results[2], Exception) else {'error': str(results[2])},
            'disk_metrics': json.loads(results[3]) if not isinstance(results[3], Exception) else {'error': str(results[3])},
            'network_metrics': json.loads(results[4]) if not isinstance(results[4], Exception) else {'error': str(results[4])},
            'api_latency': json.loads(results[5]) if not isinstance(results[5], Exception) else {'error': str(results[5])}
        }
        
        # Generate summary
        report['summary'] = _generate_performance_summary(report)
        
        logger.info("Successfully generated comprehensive performance report")
        return json.dumps(report, indent=2)
        
    except Exception as e:
        logger.error(f"Error generating comprehensive report: {e}")
        return f'{{"error": "Failed to generate comprehensive performance report: {str(e)}"}}'


def _generate_performance_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary of overall performance status"""
    summary = {
        'overall_status': 'healthy',
        'issues_detected': [],
        'recommendations': [],
        'key_metrics': {}
    }
    
    try:
        # Check node usage
        if 'node_usage' in report and 'error' not in report['node_usage']:
            node_usage = report['node_usage']
            
            # CPU usage
            if 'cpu_usage' in node_usage and 'baseline_comparison' in node_usage['cpu_usage']:
                cpu_comparison = node_usage['cpu_usage']['baseline_comparison']
                if not cpu_comparison.get('within_range', True):
                    summary['overall_status'] = 'degraded'
                    summary['issues_detected'].append('CPU usage outside baseline range')
            
            # Memory usage  
            if 'memory_usage' in node_usage and 'baseline_comparison' in node_usage['memory_usage']:
                memory_comparison = node_usage['memory_usage']['baseline_comparison']
                if not memory_comparison.get('within_range', True):
                    summary['overall_status'] = 'degraded'
                    summary['issues_detected'].append('Memory usage outside baseline range')
        
        # Check disk metrics
        if 'disk_metrics' in report and 'error' not in report['disk_metrics']:
            disk_metrics = report['disk_metrics']
            if 'summary' in disk_metrics and disk_metrics['summary'].get('overall_status') != 'healthy':
                summary['overall_status'] = 'degraded'
                summary['issues_detected'].extend(disk_metrics['summary'].get('issues_detected', []))
        
        # Check network metrics
        if 'network_metrics' in report and 'error' not in report['network_metrics']:
            network_metrics = report['network_metrics']
            if 'summary' in network_metrics and network_metrics['summary'].get('overall_status') != 'healthy':
                summary['overall_status'] = 'degraded'
                summary['issues_detected'].extend(network_metrics['summary'].get('issues_detected', []))
        
        # Check API latency
        if 'api_latency' in report and 'error' not in report['api_latency']:
            api_latency = report['api_latency']
            if 'summary' in api_latency and api_latency['summary'].get('overall_status') != 'healthy':
                summary['overall_status'] = 'degraded'
                summary['issues_detected'].extend(api_latency['summary'].get('issues_detected', []))
        
        # Generate recommendations based on issues
        if summary['issues_detected']:
            summary['recommendations'] = _generate_recommendations(summary['issues_detected'])
        
    except Exception as e:
        logger.error(f"Error generating performance summary: {e}")
        summary['error'] = str(e)
    
    return summary


def _generate_recommendations(issues: List[str]) -> List[str]:
    """Generate performance recommendations based on detected issues"""
    recommendations = []
    
    for issue in issues:
        if 'cpu' in issue.lower():
            recommendations.append("Consider scaling up nodes or optimizing workload CPU usage")
        if 'memory' in issue.lower():
            recommendations.append("Review memory requests/limits and consider node memory optimization")
        if 'disk' in issue.lower():
            recommendations.append("Investigate disk I/O bottlenecks and consider storage optimization")
        if 'network' in issue.lower():
            recommendations.append("Check network configuration and bandwidth utilization")
        if 'api' in issue.lower():
            recommendations.append("Investigate API server performance and etcd health")
    
    # Remove duplicates
    return list(set(recommendations))


async def initialize_server():
    """Initialize the MCP server with proper authentication and connections"""
    try:
        logger.info("Initializing OpenShift Benchmark MCP Server")
        
        # Initialize Prometheus connection
        prometheus_url, prometheus_token = await auth.initialize_prometheus_connection()
        
        if prometheus_url and prometheus_token:
            logger.info("Successfully connected to Prometheus")
        else:
            logger.warning("Could not establish Prometheus connection - some metrics may not be available")
        
        logger.info("MCP Server initialization completed")
        
    except Exception as e:
        logger.error(f"Error during server initialization: {e}")
        raise


async def main():
    """Main function to run the MCP server"""
    try:
        # Initialize server
        await initialize_server()
        
        # Run the MCP server with streamable HTTP transport
        await mcp.run_async(transport="streamable-http", host="0.0.0.0", port=8000)
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        logger.info("MCP Server stopped")


if __name__ == "__main__":
    asyncio.run(main())