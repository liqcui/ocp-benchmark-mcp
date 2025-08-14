#!/usr/bin/env python3
"""OpenShift Benchmark MCP Server.

This server provides MCP tools for OpenShift cluster performance monitoring and benchmarking.
"""
import asyncio
import logging
import os
import sys
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from pydantic import ConfigDict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pydantic import BaseModel, Field
from fastmcp import FastMCP

# Import our modules
from tools.ocp_benchmark_openshift_clusterinfo import get_cluster_info
from tools.ocp_benchmark_openshift_nodeinfo import get_nodes_info
from tools.ocp_benchmark_prometheus_nodes_usage import get_nodes_usage
from tools.ocp_benchmark_prometheus_pods_usage import get_pods_usage
from tools.ocp_benchmark_prometheus_diskio import get_disk_metrics
from tools.ocp_benchmark_prometheus_network import get_network_metrics
from tools.ocp_benchmark_prometheus_apilatency import get_api_request_latency,get_api_request_rate,get_etcd_latency
from elt.ocp_benchmark_elt import analyze_performance_data,benchmark_data_processor
from elt.ocp_benchmark_elt_extract_node_info import extract_node_info_from_json_data_as_json
from elt.ocp_benchmark_elt_extract_nodes_usage import extract_all_cluster_usage_info
from elt.ocp_benchmark_elt_extract_pods_usage import extract_summary,extract_pod_totals
from elt.ocp_benchmark_elt_extract_disk_io import extract_performance_analysis,extract_nodes_performance_data
from elt.ocp_benchmark_elt_extract_api_request_latency import extract_api_performance_analysis,extract_operation_statistics
from elt.ocp_benchmark_elt_extract_api_request_rate import extract_request_rate_performance_analysis,extract_active_request_rates
from elt.ocp_benchmark_elt_extract_etcd_latency import extract_etcd_performance_analysis,extract_active_operations
from elt.ocp_benchmark_extract_json import JSONExtractor
from config.ocp_benchmark_config import config_manager
from ocauth.ocp_benchmark_auth import ocp_auth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S UTC'
)
logger = logging.getLogger(__name__)

# Set timezone to UTC
os.environ['TZ'] = 'UTC'


# Pydantic models for tool parameters
class ClusterInfoParams(BaseModel):
    """Parameters for getting cluster information."""
    include_operators: Optional[bool] = Field(default=False, description="Include cluster operators status")
    detailed_status: Optional[bool] = Field(default=False, description="Return cluster information detailed status")
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")



class NodeInfoParams(BaseModel):
    """Parameters for getting node information."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class NodesUsageParams(BaseModel):
    """Parameters for getting nodes usage metrics."""
    duration_hours: float = Field(default=1.0, description="Duration in hours to collect data for")
    step: str = Field(default='1m', description="Query resolution step (e.g., '1m', '5m', '1h')")
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class PodsUsageParams(BaseModel):
    """Parameters for getting pods usage metrics."""
    duration_hours: float = Field(default=1.0, description="Duration in hours to collect data for")
    pod_regex: Optional[str] = Field(default=None, description="Regular expression to match pod names")
    label_selectors: Optional[List[str]] = Field(default=None, description="Label selectors in format 'key=value'")
    step: str = Field(default='1m', description="Query resolution step")
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class DiskMetricsParams(BaseModel):
    """Parameters for getting disk I/O metrics."""
    duration_hours: float = Field(default=1.0, description="Duration in hours to collect data for")
    step: str = Field(default='1m', description="Query resolution step")
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class NetworkMetricsParams(BaseModel):
    """Parameters for getting network metrics."""
    duration_hours: float = Field(default=1.0, description="Duration in hours to collect data for")
    step: str = Field(default='1m', description="Query resolution step")
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class APILatencyParams(BaseModel):
    """Parameters for getting API latency metrics."""
    duration_hours: float = Field(default=1.0, description="Duration in hours to collect data for")
    step: str = Field(default='1m', description="Query resolution step")
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class AnalyzePerformanceParams(BaseModel):
    """Parameters for performance analysis."""
    metrics_data: Dict[str, Any] = Field(description="Metrics data to analyze")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis to perform")
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class AnalyzeOverallParams(BaseModel):
    """Parameters for overall cluster performance analysis."""
    duration_hours: float = Field(default=1.0, description="Duration in hours to collect data for")
    step: str = Field(default='1m', description="Query resolution step")
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


# Initialize FastMCP
mcp = FastMCP("OpenShift Benchmark MCP Server")

async def initialize_server():
    """Initialize the MCP server with proper authentication and connections"""
    try:
        logger.info("Initializing OpenShift Benchmark MCP Server")
        
        # Initialize Prometheus connection
        if ocp_auth.test_prometheus_connection():
            logger.info("Successfully connected to Prometheus")
        else:
            logger.warning("Could not establish Prometheus connection - some metrics may not be available")
        
        logger.info("MCP Server initialization completed")
        
    except Exception as e:
        logger.error(f"Error during server initialization: {e}")
        raise

@mcp.tool()
async def get_cluster_information(params: ClusterInfoParams) -> str:
    """Get OpenShift cluster information including version, infrastructure, and cluster name.
    
    This tool retrieves comprehensive/detailed information about the OpenShift cluster including:
    - Cluster version and update channel
    - Infrastructure details (platform, API URL)
    - Cluster name and domain information
    
    Returns JSON string with cluster information.
    """
    try:
        if params.detailed_status:
            result = get_cluster_info()
            return result
        else:
            logger.info("Getting cluster information")
            result = get_cluster_info()
            print("result in get_cluster_information: \n",result)
            # Create extractor instance
            extractor = JSONExtractor(result)
            
            logger.info("=== EXTRACT CUSTOM FIELDS ===")
            custom_data = extractor.extract_custom_fields([
                "cluster_name",
                "version_info.version",
                "version_info.update_available",
                "infrastructure_info.platform",
                "nfrastructure_info.infrastructure_name",
                "summary.api_url"
            ])
            customized_cluster_info_json=json.dumps(custom_data, ensure_ascii=False, indent=2)
            logger.info(customized_cluster_info_json)
            logger.info("Successfully retrieved cluster information")
            return customized_cluster_info_json
    except Exception as e:
        logger.error(f"Failed to get cluster information: {e}")
        return f'{{"error": "Failed to get cluster information: {str(e)}"}}'

@mcp.tool()
async def get_node_information(params: NodeInfoParams) -> str:
    """Get detailed information about OpenShift cluster nodes.
    
    This tool retrieves comprehensive node information including:
    - Node roles (master, worker, infra)
    - CPU cores and memory capacity
    - Instance types and status
    - Node counts and resource totals by role
    
    Returns JSON string with node information grouped by role.
    """
    try:
        logger.info("Getting node information")
        result = get_nodes_info()
        extractor = JSONExtractor(result)
        cluster_summary_data = extractor.extract_custom_fields([
                "cluster_summary.total_nodes",
                "cluster_summary.ready_nodes",
                "cluster_summary.total_cpu_cores",
                "cluster_summary.total_memory_gb",
                "cluster_summary.node_roles_distribution.master",
                "cluster_summary.node_roles_distribution.worker"
            ])
        node_by_roles_master = extractor.extract_custom_fields([
                "nodes_by_role.master.count",
                "nodes_by_role.master.ready_count",
                "nodes_by_role.master.total_cpu_cores",
                "nodes_by_role.master.total_memory_gb",
                "nodes_by_role.master.average_cpu_cores",
                "nodes_by_role.master.average_memory_gb",
                "nodes_by_role.master.instance_types"
            ])
        
        node_by_roles_worker = extractor.extract_custom_fields([
                "nodes_by_role.worker.count",
                "nodes_by_role.worker.ready_count",
                "nodes_by_role.worker.total_cpu_cores",
                "nodes_by_role.worker.total_memory_gb",
                "nodes_by_role.worker.average_cpu_cores",
                "nodes_by_role.worker.average_memory_gb",
                "nodes_by_role.worker.instance_types"
            ])
        node_by_roles_infra = extractor.extract_custom_fields([
                "nodes_by_role.infra.count",
                "nodes_by_role.infra.ready_count",
                "nodes_by_role.infra.total_cpu_cores",
                "nodes_by_role.infra.total_memory_gb",
                "nodes_by_role.infra.average_cpu_cores",
                "nodes_by_role.infra.average_memory_gb",
                "nodes_by_role.infra.instance_types"
            ])
        detailed_node_info = json.loads(
            extract_node_info_from_json_data_as_json(json.loads(result))
        )            
        custom_data={
             "cluster_summary": cluster_summary_data,
             "node_by_roles": {
                 "master": node_by_roles_master,
                 "worker": node_by_roles_worker,
                 "infra": node_by_roles_infra
                                },
            "detailed_nodes_info": detailed_node_info
        }
        logger.info("Successfully retrieved node information")
        customized_nodes_info_json=json.dumps(custom_data, ensure_ascii=False, indent=2)
        return customized_nodes_info_json
    except Exception as e:
        logger.error(f"Failed to get node information: {e}")
        return f'{{"error": "Failed to get node information: {str(e)}"}}'

@mcp.tool()
async def get_nodes_usage_metrics(params: NodesUsageParams) -> str:
    """Get CPU and memory usage metrics for all cluster nodes over a time period.
    
    This tool queries Prometheus for node-level resource utilization including:
    - CPU usage percentages with min/max/mean statistics
    - Memory usage percentages with min/max/mean statistics
    - Comparison against configured baselines
    - Performance alerts and recommendations
    
    Args:
        duration_hours: Duration in hours to collect data for (default: 1.0)
        step: Query resolution step like '1m', '5m', '1h' (default: '1m')
    
    Returns JSON string with nodes usage metrics and analysis.
    """
    try:
        logger.info(f"Getting nodes usage metrics for {params.duration_hours} hours")
        result = get_nodes_usage(params.duration_hours, params.step)
        combined_result = extract_all_cluster_usage_info(result)
        print("✓ Combined extraction successful!")
        customized_node_usage_json=json.dumps(combined_result, indent=2)
        logger.info("Successfully retrieved nodes usage metrics")
        return customized_node_usage_json
    except Exception as e:
        logger.error(f"Failed to get nodes usage metrics: {e}")
        return f'{{"error": "Failed to get nodes usage metrics: {str(e)}"}}'


@mcp.tool()
async def get_pods_usage_metrics(params: PodsUsageParams) -> str:
    """Get CPU and memory usage metrics for pods with optional filtering.
    
    This tool queries Prometheus for pod-level resource utilization including:
    - CPU usage per container and pod totals
    - Memory usage in absolute values and percentage of limits
    - Filtering by pod name regex or label selectors
    - Performance alerts for high resource usage
    
    Args:
        duration_hours: Duration in hours to collect data for (default: 1.0)
        pod_regex: Regular expression to match pod names (optional)
        label_selectors: List of label selectors in format 'key=value' (optional)
        step: Query resolution step (default: '1m')
    
    Returns JSON string with pods usage metrics and analysis.
    """
    try:
        logger.info(f"Getting pods usage metrics for {params.duration_hours} hours")
        if params.pod_regex:
            logger.info(f"Filtering by pod regex: {params.pod_regex}")
        if params.label_selectors:
            logger.info(f"Filtering by labels: {params.label_selectors}")
        
        if params.pod_regex == None:
            params.pod_regex = "etcd.*|ovnkube.*|kube-apiserver.*|openshift-kube-scheduler.*|kube-controller-manager.*"

        result = get_pods_usage(
            params.duration_hours, 
            params.pod_regex, 
            params.label_selectors, 
            params.step
        )
        result_json=json.loads(result)
        extract_pods_usage_summary=extract_summary(result_json)
        extract_pods_usage_total_usage=extract_pod_totals(result_json)
        
        if extract_pods_usage_total_usage or extract_pods_usage_summary:
            combined_result={
                "pods_total_usage": extract_pods_usage_total_usage,
                "summary": extract_pods_usage_summary,
            }
        customized_pods_usage_json=json.dumps(combined_result, indent=2)
        logger.info("Successfully retrieved pods usage metrics")
        return customized_pods_usage_json
    except Exception as e:
        logger.error(f"Failed to get pods usage metrics: {e}")
        return f'{{"error": "Failed to get pods usage metrics: {str(e)}"}}'

async def get_etcd_pods_usage_metrics(params: PodsUsageParams) -> str:
    """Get CPU and memory usage metrics for etcd pods with optional filtering.
    
    This tool queries Prometheus for pod-level resource utilization including:
    - CPU usage per container and pod totals
    - Memory usage in absolute values and percentage of limits
    - Filtering by pod name etcd.*
    - Performance alerts for high resource usage
    
    Args:
        duration_hours: Duration in hours to collect data for (default: 1.0)
        pod_regex: Regular expression to match pod names (optional)
        label_selectors: List of label selectors in format 'key=value' (optional)
        step: Query resolution step (default: '1m')
    
    Returns JSON string with pods usage metrics and analysis.
    """
    try:
        logger.info(f"Getting pods usage metrics for {params.duration_hours} hours")
        if params.label_selectors:
            logger.info(f"Filtering by labels: {params.label_selectors}")
        
        result = get_pods_usage(
            params.duration_hours, 
            "etcd.*", 
            params.label_selectors, 
            params.step
        )
        result_json=json.loads(result)
        extract_pods_usage_summary=extract_summary(result_json)
        extract_pods_usage_total_usage=extract_pod_totals(result_json)
        
        if extract_pods_usage_total_usage or extract_pods_usage_summary:
            combined_result={
                "pods_total_usage": extract_pods_usage_total_usage,
                "summary": extract_pods_usage_summary,
            }
        customized_pods_usage_json=json.dumps(combined_result, indent=2)
        logger.info("Successfully retrieved pods usage metrics")
        return customized_pods_usage_json
    except Exception as e:
        logger.error(f"Failed to get pods usage metrics: {e}")
        return f'{{"error": "Failed to get pods usage metrics: {str(e)}"}}'

@mcp.tool()
async def get_disk_io_metrics(params: DiskMetricsParams) -> str:
    """Get comprehensive disk I/O performance metrics.
    
    This tool queries Prometheus for disk performance including:
    - Read/write throughput in MB/s
    - Read/write IOPS (operations per second)
    - Read/write latency in milliseconds
    - Comparison against configured baselines
    - Performance alerts and recommendations
    
    Args:
        duration_hours: Duration in hours to collect data for (default: 1.0)
        step: Query resolution step (default: '1m')
    
    Returns JSON string with disk I/O metrics and analysis.
    """
    try:
        logger.info(f"Getting disk I/O metrics for {params.duration_hours} hours")
        result = get_disk_metrics(params.duration_hours, params.step)
        logger.info("Successfully retrieved disk I/O metrics")
        print("=== Performance Analysis ===")
        performance_analysis = extract_performance_analysis(json.loads(result))
        
        # Extract simplified nodes performance data
        print("=== Nodes Performance Data (Statistics Only) ===")
        nodes_disk_io = extract_nodes_performance_data(json.loads(result))
        disk_io_usage={
            "performance_analysis": performance_analysis,
            "nodes_disk_io_usage": nodes_disk_io
        }
        disk_io_usage_json=json.dumps(disk_io_usage, indent=2)
        return disk_io_usage_json
    except Exception as e:
        logger.error(f"Failed to get disk I/O metrics: {e}")
        return f'{{"error": "Failed to get disk I/O metrics: {str(e)}"}}'


@mcp.tool()
async def get_network_performance_metrics(params: NetworkMetricsParams) -> str:
    """Get comprehensive network performance metrics.
    
    This tool queries Prometheus for network performance including:
    - Network throughput (RX/TX bytes and packets per second)
    - Network error rates
    - Interface utilization analysis
    - Comparison against configured baselines
    - Performance alerts and recommendations
    
    Args:
        duration_hours: Duration in hours to collect data for (default: 1.0)
        step: Query resolution step (default: '1m')
    
    Returns JSON string with network performance metrics and analysis.
    """
    try:
        logger.info(f"Getting network metrics for {params.duration_hours} hours")
        result = get_network_metrics(params.duration_hours, params.step)
        #print("result is:\n",result)
        performance_data = extract_performance_analysis(json.loads(result))
        print("performance_data is:\n",performance_data)
        customized_network_io_json=json.dumps(performance_data, indent=2)
        logger.info("Successfully retrieved network metrics")
        return customized_network_io_json
    except Exception as e:
        logger.error(f"Failed to get network metrics: {e}")
        return f'{{"error": "Failed to get network metrics: {str(e)}"}}'


@mcp.tool()
async def get_api_request_latency_metrics(params: APILatencyParams) -> str:
    """Get API server and etcd latency metrics.
    
    This tool queries Prometheus for API performance including:
    - API server request latency (P50, P95, P99 percentiles)
    - API request rates by operation and response code
    - etcd request latency metrics
    - Comparison against configured baselines
    - Performance alerts and recommendations
    
    Args:
        duration_hours: Duration in hours to collect data for (default: 1.0)
        step: Query resolution step (default: '1m')
    
    Returns JSON string with API latency metrics and analysis.
    """
    try:
        logger.info(f"Getting API latency metrics for {params.duration_hours} hours")
        raw_metrics = get_api_request_latency(params.duration_hours, params.step)
        metrics_json = json.loads(raw_metrics)

        performance_analysis = extract_api_performance_analysis(metrics_json)
        operations_stats = extract_operation_statistics(metrics_json)
        customized_api_request_latency = {
            "performance_analysis": performance_analysis
            # "operations_stats": operations_stats
        }

        logger.info("Successfully retrieved API latency metrics")
        return json.dumps(customized_api_request_latency, indent=2)
    except Exception as e:
        logger.error(f"Failed to get API latency metrics: {e}")
        return f'{{"error": "Failed to get API latency metrics: {str(e)}"}}'

@mcp.tool()
async def get_api_request_rate_metrics(params: APILatencyParams) -> str:
    """Get API server and etcd latency metrics.
    
    This tool queries Prometheus for API performance including:
    - API server request latency (P50, P95, P99 percentiles)
    - API request rates by operation and response code
    - etcd request latency metrics
    - Comparison against configured baselines
    - Performance alerts and recommendations
    
    Args:
        duration_hours: Duration in hours to collect data for (default: 1.0)
        step: Query resolution step (default: '1m')
    
    Returns JSON string with API latency metrics and analysis.
    """
    try:
        logger.info(f"Getting API latency metrics for {params.duration_hours} hours")
        raw_metrics = get_api_request_rate(params.duration_hours, params.step)
        # metrics_json = json.loads(raw_metrics)
        metrics_json = raw_metrics
        performance_analysis = json.loads(extract_request_rate_performance_analysis(metrics_json))
        request_rates = json.loads(extract_active_request_rates(metrics_json))
        customized_api_request_rate = {
            "performance_analysis": performance_analysis
            # "request_rates": request_rates
        }
        print("customized_api_request_rate is:\n",customized_api_request_rate)
        logger.info("Successfully retrieved API latency metrics")
        return json.dumps(customized_api_request_rate, indent=2)
    except Exception as e:
        logger.error(f"Failed to get API latency metrics: {e}")
        return f'{{"error": "Failed to get API latency metrics: {str(e)}"}}'

@mcp.tool()
async def get_etcd_latency_metrics(params: APILatencyParams) -> str:
    """Get API server and etcd latency metrics.
    
    This tool queries Prometheus for API performance including:
    - etcd request latency metrics
    - Comparison against configured baselines
    - Performance alerts and recommendations
    
    Args:
        duration_hours: Duration in hours to collect data for (default: 1.0)
        step: Query resolution step (default: '1m')
    
    Returns JSON string with API latency metrics and analysis.
    """
    try:
        logger.info(f"Getting etcd latency metrics for {params.duration_hours} hours")
        raw_metrics = get_etcd_latency(params.duration_hours, params.step)
        metrics_json = json.loads(raw_metrics)

        performance_analysis = extract_etcd_performance_analysis(metrics_json)
        operations_stats = extract_active_operations(metrics_json)
        customized_etcd_latency = {
            "performance_analysis": performance_analysis
            # "operations_stats": operations_stats
        }
         
        logger.info("Successfully retrieved API latency metrics")
        return json.dumps(customized_etcd_latency, indent=2)
    except Exception as e:
        logger.error(f"Failed to get etcd latency metrics: {e}")
        return f'{{"error": "Failed to get etcd latency metrics: {str(e)}"}}'
    
@mcp.tool()
async def analyze_ocp_overall_cluster_performance(params: AnalyzeOverallParams) -> str:
    """Collect key metrics and return an overall performance analysis snapshot.

    This tool orchestrates nodes, pods, disk, network, and API analyses and
    returns a compact JSON with component analyses plus an overall summary.
    """
    try:
        # Parameters
        duration_hours = params.duration_hours
        step = params.step

        # Collect component-level metrics (call underlying metric functions directly, not MCP tools)
        nodes_raw = get_nodes_usage(duration_hours, step)  # JSON string
        pods_raw = get_pods_usage(duration_hours, None, None, step)  # JSON string
        disk_raw = get_disk_metrics(duration_hours, step)  # JSON string
        network_raw = get_network_metrics(duration_hours, step)  # JSON string
        api_latency_raw = get_api_request_latency(duration_hours, step)  # JSON string

        # Perform higher-level analysis per component via ELT analyzer where appropriate
        from analysis.ocp_benchmark_performance_analysis import analyze_comprehensive_performance

        # Nodes analysis
        try:
            nodes_combined = extract_all_cluster_usage_info(nodes_raw)
            nodes_analysis = analyze_comprehensive_performance(nodes_combined)
        except Exception:
            nodes_analysis = json.dumps({})

        # Convert nodes_analysis into a concise 2-3 level JSON structure
        def _simplify_nodes_analysis(block: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
            try:
                parsed = json.loads(block) if isinstance(block, str) else block
                if not isinstance(parsed, dict):
                    return {}
                health = parsed.get('health_analysis', {}) or {}
                summary = parsed.get('summary', {}) or {}
                capacity = parsed.get('capacity_forecast', {}) or {}
                bench = parsed.get('benchmark_comparison', {}) or {}
                forecasts = capacity.get('forecasts', {}) if isinstance(capacity, dict) else {}
                cpu_fc = forecasts.get('cpu', {}) if isinstance(forecasts, dict) else {}
                mem_fc = forecasts.get('memory', {}) if isinstance(forecasts, dict) else {}

                simplified: Dict[str, Any] = {
                    "meta": {
                        "analysis_timestamp": parsed.get('analysis_timestamp')
                    },
                    "health": {
                        "score": health.get('overall_health_score', summary.get('health_score')),
                        "status": health.get('health_status', summary.get('overall_health')),
                        "component_scores": health.get('component_scores', {}),
                        "issues": health.get('issues', []),
                        "recommendations": health.get('recommendations', [])
                    },
                    "capacity": {
                        "period_days": capacity.get('forecast_period_days'),
                        "confidence": capacity.get('confidence'),
                        "cpu": {
                            "current_usage": cpu_fc.get('current_usage'),
                            "predicted_usage": cpu_fc.get('predicted_usage'),
                            "growth_rate_daily": cpu_fc.get('growth_rate_daily'),
                            "days_to_warning": cpu_fc.get('days_to_warning'),
                            "days_to_critical": cpu_fc.get('days_to_critical'),
                            "risk_level": cpu_fc.get('risk_level'),
                            "max_observed": cpu_fc.get('max_observed')
                        } if isinstance(cpu_fc, dict) else {},
                        "memory": {
                            "current_usage": mem_fc.get('current_usage'),
                            "predicted_usage": mem_fc.get('predicted_usage'),
                            "growth_rate_daily": mem_fc.get('growth_rate_daily'),
                            "days_to_warning": mem_fc.get('days_to_warning'),
                            "days_to_critical": mem_fc.get('days_to_critical'),
                            "risk_level": mem_fc.get('risk_level'),
                            "max_observed": mem_fc.get('max_observed')
                        } if isinstance(mem_fc, dict) else {}
                    },
                    "benchmark": {
                        "overall_rating": bench.get('overall_rating'),
                        "benchmark_date": bench.get('benchmark_date')
                    },
                    "summary": {
                        "overall_health": summary.get('overall_health'),
                        "health_score": summary.get('health_score'),
                        "total_recommendations": summary.get('total_recommendations'),
                        "high_priority_recommendations": summary.get('high_priority_recommendations'),
                        "benchmark_rating": summary.get('benchmark_rating')
                    }
                }
                return simplified
            except Exception:
                return {}

        nodes_analysis = json.dumps(_simplify_nodes_analysis(nodes_analysis), indent=2)

        # Pods analysis: create a light wrapper summary using existing structures
        pods_analysis = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "unknown",
            "findings": [],
            "recommendations": [],
            "baseline_comparisons": {}
        }

        # Disk analysis
        try:
            disk_analysis = analyze_comprehensive_performance(json.loads(disk_raw))
        except Exception:
            disk_analysis = json.dumps({})

        # Network analysis
        try:
            network_analysis = analyze_comprehensive_performance(json.loads(network_raw))
        except Exception:
            network_analysis = json.dumps({})

        # API analysis (use api latency payload directly)
        try:
            api_analysis = analyze_comprehensive_performance(json.loads(api_latency_raw))
        except Exception:
            api_analysis = json.dumps({})

        # Remove duplicate nested entries like benchmark_comparison.benchmark_comparison in api_analysis
        def _sanitize_benchmark_comparison(block: Union[str, Dict[str, Any]]) -> str:
            try:
                parsed = json.loads(block) if isinstance(block, str) else block
                if isinstance(parsed, dict):
                    bc = parsed.get('benchmark_comparison')
                    if isinstance(bc, dict) and 'benchmark_comparison' in bc:
                        # Drop the nested duplicate structure, keep top-level aggregates
                        bc.pop('benchmark_comparison', None)
                    return json.dumps(parsed, indent=2)
            except Exception:
                pass
            return block if isinstance(block, str) else json.dumps(block, indent=2)

        api_analysis = _sanitize_benchmark_comparison(api_analysis)

        # Build overall summary from component scores when available
        def extract_score(block: Union[str, Dict[str, Any]]) -> Optional[float]:
            try:
                parsed = json.loads(block) if isinstance(block, str) else block
                if isinstance(parsed, dict):
                    if isinstance(parsed.get('health_analysis'), dict):
                        val = parsed['health_analysis'].get('overall_health_score')
                        return float(val) if isinstance(val, (int, float)) else None
                    if isinstance(parsed.get('summary'), dict):
                        val = parsed['summary'].get('health_score')
                        return float(val) if isinstance(val, (int, float)) else None
            except Exception:
                return None
            return None

        scores = []
        comp_scores: Dict[str, float] = {}
        for name, block in (
            ("nodes_analysis", nodes_analysis),
            ("disk_analysis", disk_analysis),
            ("network_analysis", network_analysis),
            ("api_analysis", api_analysis),
        ):
            s = extract_score(block)
            if s is not None:
                comp_scores[name] = s
                scores.append(s)

        avg = round(sum(scores) / len(scores), 1) if scores else 0.0
        if avg >= 90:
            status = "excellent"
        elif avg >= 75:
            status = "good"
        elif avg >= 60:
            status = "fair"
        elif avg >= 40:
            status = "poor"
        else:
            status = "critical"

        overall_summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_health_score": avg,
            "health_status": status,
            "critical_issues": [],
            "warning_issues": [],
            "recommendations": [],
            "component_scores": comp_scores,
        }

        result = {
            "nodes_analysis": nodes_analysis,
            "pods_analysis": pods_analysis,
            "disk_analysis": disk_analysis,
            "network_analysis": network_analysis,
            "api_analysis": api_analysis,
            "overall_summary": overall_summary,
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Failed to analyze overall cluster performance: {e}")
        return f'{{"error": "Failed to analyze overall cluster performance: {str(e)}"}}'
@mcp.tool()
async def analyze_ocp_performance_data(params: AnalyzePerformanceParams) -> str:
    """Analyze performance data and generate insights.
    
    This tool processes collected metrics data to provide:
    - Performance analysis against baselines
    - Trend identification and anomaly detection
    - Resource utilization insights
    - Performance recommendations
    - Formatted reports in table format
    
    Args:
        metrics_data: Dictionary containing metrics data to analyze
        analysis_type: Type of analysis to perform (default: "comprehensive")
    
    Returns JSON string with analysis results and recommendations.
    """
    try:
        logger.info(f"Analyzing performance data with {params.analysis_type} analysis")
        result = analyze_performance_data(params.metrics_data, params.analysis_type)
        logger.info("Successfully analyzed performance data")
        return result
    except Exception as e:
        logger.error(f"Failed to analyze performance data: {e}")
        return f'{{"error": "Failed to analyze performance data: {str(e)}"}}'


async def main():
    """Main entry point for the MCP server."""
    logger.info("Starting OpenShift Benchmark MCP Server")
    
    # Verify environment
    kubeconfig = os.getenv('KUBECONFIG')
    if not kubeconfig:
        logger.warning("KUBECONFIG environment variable not set")
    else:
        logger.info(f"Using KUBECONFIG: {kubeconfig}")
    
    try:
        # Test basic connectivity
        await initialize_server()
        
        # Start the MCP server using streamable HTTP transport
        await mcp.run_async(transport="streamable-http", port=8000, host="0.0.0.0")
    
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)