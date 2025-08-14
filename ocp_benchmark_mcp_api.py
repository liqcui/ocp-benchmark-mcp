#!/usr/bin/env python3
"""OpenShift Benchmark MCP API Server - FastAPI wrapper for MCP tools"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Set timezone to UTC
os.environ['TZ'] = 'UTC'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S UTC'
)
logger = logging.getLogger(__name__)

# Global variables for MCP client
mcp_client = None
MCP_SERVER_URL = "http://localhost:8000"

# Pydantic models for API requests
class ClusterInfoRequest(BaseModel):
    """Request model for cluster information"""
    include_operators: Optional[bool] = Field(default=True, description="Include cluster operators status")

class NodeInfoRequest(BaseModel):
    """Request model for node information"""
    summary_only: Optional[bool] = Field(default=False, description="Return summary only")
    role_filter: Optional[str] = Field(default=None, description="Filter by role")

class NodeUsageRequest(BaseModel):
    """Request model for node usage metrics"""
    duration_hours: Optional[int] = Field(default=1, ge=1, le=24, description="Duration in hours")

class PodUsageRequest(BaseModel):
    """Request model for pod usage metrics"""
    duration_hours: Optional[int] = Field(default=1, ge=1, le=24, description="Duration in hours")
    pod_patterns: Optional[List[str]] = Field(default=None, description="Pod name patterns")
    label_selectors: Optional[List[str]] = Field(default=None, description="Label selectors")
    top_n: Optional[int] = Field(default=10, ge=1, le=100, description="Top N pods")

class DiskMetricsRequest(BaseModel):
    """Request model for disk metrics"""
    duration_hours: Optional[int] = Field(default=1, ge=1, le=24, description="Duration in hours")
    by_device: Optional[bool] = Field(default=False, description="Group by device")

class NetworkMetricsRequest(BaseModel):
    """Request model for network metrics"""
    duration_hours: Optional[int] = Field(default=1, ge=1, le=24, description="Duration in hours")
    by_interface: Optional[bool] = Field(default=False, description="Group by interface")

class APILatencyRequest(BaseModel):
    """Request model for API latency metrics"""
    duration_hours: Optional[int] = Field(default=1, ge=1, le=24, description="Duration in hours")
    slow_threshold_ms: Optional[float] = Field(default=1000.0, ge=0, description="Slow request threshold")
    include_slow_requests: Optional[bool] = Field(default=False, description="Include slow requests")

class ComprehensiveReportRequest(BaseModel):
    """Request model for comprehensive report"""
    duration_hours: Optional[int] = Field(default=1, ge=1, le=24, description="Duration in hours")

class MCPClient:
    """MCP Client for connecting to OCP Benchmark MCP Server."""
    
    def __init__(self, mcp_server_url: str = "http://localhost:8000/"):
        self.mcp_server_url = mcp_server_url.rstrip('/')
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        # self.session = aiohttp.ClientSession()
        url = f"{self.mcp_server_url}/mcp"
        # Connect to the server using Streamable HTTP
        async with streamablehttp_client(
                url,
                headers={"accept": "application/json"}
            ) as (
                read_stream,
                write_stream,
                get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize the connection
                self.session=session
                await self.session.initialize()
                logger.info("Successfully connected to MCP server")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return self
     
    async def call_tool(self, tool_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

        """Call a tool on the MCP tool via HTTP.

        Args:
            tool_name: Name of the tool to call
            params: Parameters for the tool

        Returns:
            Tool response data
        """    
        try:
            url = f"{self.mcp_server_url}/mcp"

            # Connect to the server using Streamable HTTP
            async with streamablehttp_client(
                url,
                headers={"accept": "application/json"}
                ) as (
                    read_stream,
                    write_stream,
                    get_session_id,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize the connection
                    await session.initialize()
 
                    # Get session id once connection established
                    session_id = get_session_id() 
                    print("Session ID: in call_tool", session_id)
            
                    print(f"Calling tool {tool_name} with params {params}",type(params))
                    
                    #Make a request to the server using HTTP, May convert the response to JSON if needed
                    request_data = {
                         "request": params or {}
                    }

                    print(f"Calling tool {tool_name} with params {request_data}",type(request_data))

                    result = await session.call_tool(tool_name, request_data)
                    print("#*"*50)
                    print("result in call_tool of mcp client:",result)
                    print("#*"*50)
                    print(f"{tool_name} = {result.content[0].text}")
                    #json_data = json.loads(result.content[0].text)
                    return result.content[0].text

        except Exception as e:
            print(f"Call MCP tool failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    global mcp_client
    logger.info("Starting MCP API Server")
    
    try:
        # Initialize HTTP client for MCP communication
        mcp_client = MCPClient(MCP_SERVER_URL)
        
        # Wait a moment for MCP server to be ready
        await asyncio.sleep(10)
        async with mcp_client:
            # Test connection
            if not mcp_client.session:
                raise HTTPException(status_code=503, detail="Failed to connect to MCP server")
 
        logger.info("MCP API Server started successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down MCP API Server")
    if mcp_client:
        await mcp_client.aclose()


# Initialize FastAPI app
app = FastAPI(
    title="OpenShift Benchmark MCP API",
    description="FastAPI wrapper for OpenShift Benchmark MCP tools",
    version="1.0.0",
    lifespan=lifespan
)


async def call_mcp_tool(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Call an MCP tool via HTTP"""
    try:
        if not mcp_client:
            raise HTTPException(status_code=503, detail="MCP client not initialized")
        
        
        request_data = {
            "request": params or {}
            }
    
        
        response = await mcp_client.call_tool(tool_name, request_data)
        print(f"Response from MCP tool {tool_name}: {response}")
        if response:
            return response
        else:
            raise HTTPException(status_code=response.status_code, detail=f"MCP tool call failed: {response.text}")
    
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Failed to communicate with MCP server: {str(e)}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"Invalid response from MCP server: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "OpenShift Benchmark MCP API",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoints": {
            "cluster": "/cluster/info",
            "nodes": "/nodes/info",
            "usage": "/metrics/usage",
            "disk": "/metrics/disk",
            "network": "/metrics/network",
            "api": "/metrics/api",
            "reports": "/reports/comprehensive"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mcp_server": MCP_SERVER_URL
    }


# Cluster Information Endpoints
@app.post("/cluster/info")
async def get_cluster_info(request: ClusterInfoRequest):
    """Get OpenShift cluster information"""

    # Get cluster info
    return await mcp_client.call_tool("get_cluster_info")



@app.get("/cluster/info")
async def get_cluster_info_get():
    """Get OpenShift cluster information (GET method)"""
    return await get_cluster_info(ClusterInfoRequest())


# Node Information Endpoints
@app.post("/nodes/info")
async def get_node_info(request: NodeInfoRequest):
    """Get OpenShift node information"""
    try:
        result = await call_mcp_tool("get_node_info", request.model_dump())
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in get_node_info: {e}")
        raise


@app.get("/nodes/info")
async def get_node_info_get(summary_only: bool = False, role_filter: Optional[str] = None):
    """Get OpenShift node information (GET method)"""
    request = NodeInfoRequest(summary_only=summary_only, role_filter=role_filter)
    return await get_node_info(request)


# Usage Metrics Endpoints
@app.post("/metrics/usage/nodes")
async def get_node_usage_metrics(request: NodeUsageRequest):
    """Get node usage metrics"""
    try:
        result = await call_mcp_tool("get_node_usage_metrics", request.model_dump())
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in get_node_usage_metrics: {e}")
        raise


@app.get("/metrics/usage/nodes")
async def get_node_usage_metrics_get(duration_hours: int = 1):
    """Get node usage metrics (GET method)"""
    request = NodeUsageRequest(duration_hours=duration_hours)
    return await get_node_usage_metrics(request)


@app.post("/metrics/usage/pods")
async def get_pod_usage_metrics(request: PodUsageRequest):
    """Get pod usage metrics"""
    try:
        result = await call_mcp_tool("get_pod_usage_metrics", request.model_dump())
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in get_pod_usage_metrics: {e}")
        raise


@app.get("/metrics/usage/pods")
async def get_pod_usage_metrics_get(duration_hours: int = 1, top_n: int = 10):
    """Get pod usage metrics (GET method)"""
    request = PodUsageRequest(duration_hours=duration_hours, top_n=top_n)
    return await get_pod_usage_metrics(request)


# Disk Metrics Endpoints
@app.post("/metrics/disk")
async def get_disk_metrics(request: DiskMetricsRequest):
    """Get disk I/O metrics"""
    try:
        result = await call_mcp_tool("get_disk_io_metrics", request.model_dump())
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in get_disk_metrics: {e}")
        raise


@app.get("/metrics/disk")
async def get_disk_metrics_get(duration_hours: int = 1, by_device: bool = False):
    """Get disk I/O metrics (GET method)"""
    request = DiskMetricsRequest(duration_hours=duration_hours, by_device=by_device)
    return await get_disk_metrics(request)


# Network Metrics Endpoints
@app.post("/metrics/network")
async def get_network_metrics(request: NetworkMetricsRequest):
    """Get network performance metrics"""
    try:
        result = await call_mcp_tool("get_network_metrics", request.model_dump())
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in get_network_metrics: {e}")
        raise


@app.get("/metrics/network")
async def get_network_metrics_get(duration_hours: int = 1, by_interface: bool = False):
    """Get network performance metrics (GET method)"""
    request = NetworkMetricsRequest(duration_hours=duration_hours, by_interface=by_interface)
    return await get_network_metrics(request)


# API Latency Endpoints
@app.post("/metrics/api")
async def get_api_latency_metrics(request: APILatencyRequest):
    """Get API server latency metrics"""
    try:
        result = await call_mcp_tool("get_api_latency_metrics", request.model_dump())
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in get_api_latency_metrics: {e}")
        raise


@app.get("/metrics/api")
async def get_api_latency_metrics_get(duration_hours: int = 1, slow_threshold_ms: float = 1000.0):
    """Get API server latency metrics (GET method)"""
    request = APILatencyRequest(duration_hours=duration_hours, slow_threshold_ms=slow_threshold_ms)
    return await get_api_latency_metrics(request)


# Baseline Configuration Endpoint
@app.get("/config/baseline")
async def get_baseline_configuration():
    """Get baseline configuration"""
    try:
        result = await call_mcp_tool("get_baseline_configuration", {})
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in get_baseline_configuration: {e}")
        raise


# Comprehensive Report Endpoints
@app.post("/reports/comprehensive")
async def get_comprehensive_report(request: ComprehensiveReportRequest):
    """Generate comprehensive performance report"""
    try:
        result = await call_mcp_tool("get_comprehensive_performance_report", request.model_dump())
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in get_comprehensive_report: {e}")
        raise


@app.get("/reports/comprehensive")
async def get_comprehensive_report_get(duration_hours: int = 1):
    """Generate comprehensive performance report (GET method)"""
    request = ComprehensiveReportRequest(duration_hours=duration_hours)
    return await get_comprehensive_report(request)


# Background task for periodic monitoring
@app.post("/monitoring/start")
async def start_monitoring(background_tasks: BackgroundTasks, interval_minutes: int = 5):
    """Start periodic monitoring"""
    async def periodic_monitoring():
        while True:
            try:
                logger.info("Running periodic monitoring check")
                # This could trigger alerts, store metrics, etc.
                await asyncio.sleep(interval_minutes * 60)
            except Exception as e:
                logger.error(f"Error in periodic monitoring: {e}")
    
    background_tasks.add_task(periodic_monitoring)
    return {"message": f"Monitoring started with {interval_minutes} minute intervals"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "ocp_benchmark_mcp_api:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
        access_log=True,
        reload=False
    )