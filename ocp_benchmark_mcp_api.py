#!/usr/bin/env python3
"""
FastAPI wrapper for OCP Benchmark MCP Server
Exposes MCP tools as REST API endpoints
"""

import asyncio
import logging
import os
import sys
import json
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, BackgroundTasks, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from elt.ocp_benchmark_json2table import json_to_table
from elt.ocp_benchmark_extract_json import JSONExtractor
# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import authentication module
from ocauth.ocp_benchmark_auth import ocp_auth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global variables for MCP client
mcp_client = None
MCP_SERVER_URL = "http://localhost:8000"

# Pydantic models for API requests - matching MCP server parameter models
class ClusterInfoRequest(BaseModel):
    """Request model for cluster information - no parameters required"""
    include_operators: Optional[bool] = Field(default=False, description="Include cluster operators status")
    detailed_status: Optional[bool] = Field(default=False, description="Return cluster information detailed status")
    

class NodeInfoRequest(BaseModel):
    """Request model for node information - no parameters required"""
    pass

class NodesUsageRequest(BaseModel):
    """Request model for nodes usage metrics"""
    duration_hours: float = Field(default=1.0, description="Duration in hours to collect data for")
    step: str = Field(default='1m', description="Query resolution step (e.g., '1m', '5m', '1h')")

class PodsUsageRequest(BaseModel):
    """Request model for pods usage metrics"""
    duration_hours: float = Field(default=1.0, description="Duration in hours to collect data for")
    pod_regex: Optional[str] = Field(default=None, description="Regular expression to match pod names")
    label_selectors: Optional[List[str]] = Field(default=None, description="Label selectors in format 'key=value'")
    step: str = Field(default='1m', description="Query resolution step")

class DiskMetricsRequest(BaseModel):
    """Request model for disk I/O metrics"""
    duration_hours: float = Field(default=1.0, description="Duration in hours to collect data for")
    step: str = Field(default='1m', description="Query resolution step")

class NetworkMetricsRequest(BaseModel):
    """Request model for network metrics"""
    duration_hours: float = Field(default=1.0, description="Duration in hours to collect data for")
    step: str = Field(default='1m', description="Query resolution step")

class APILatencyRequest(BaseModel):
    """Request model for API latency metrics"""
    duration_hours: float = Field(default=1.0, description="Duration in hours to collect data for")
    step: str = Field(default='1m', description="Query resolution step")

class AnalyzePerformanceRequest(BaseModel):
    """Request model for performance analysis"""
    metrics_data: Dict[str, Any] = Field(description="Metrics data to analyze")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis to perform")

class MCPClient:
    """MCP Client for connecting to OCP Benchmark MCP Server."""
    
    def __init__(self, mcp_server_url: str = "http://localhost:8000/"):
        self.mcp_server_url = mcp_server_url.rstrip('/')
    
    async def list_available_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server.
        
        Returns:
            List of tool definitions (name, description, input schema)
        """
        try:
            url = f"{self.mcp_server_url}/mcp"
            async with streamablehttp_client(url) as (
                read_stream,
                write_stream,
                get_session_id,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    serialized: List[Dict[str, Any]] = []
 
                    for tool in tools.tools:
                        try:
                            serialized.append({
                                "name": getattr(tool, "name", None) or (tool.get("name") if isinstance(tool, dict) else None),
                                "description": getattr(tool, "description", None) or (tool.get("description") if isinstance(tool, dict) else None),
                                "inputSchema": getattr(tool, "inputSchema", None) or (tool.get("inputSchema") if isinstance(tool, dict) else None)
                            })
                        except Exception:
                            serialized.append({"raw": str(tool)})
                    return serialized
        except Exception as e:
            logger.error(f"Failed to list MCP tools: {str(e)}")
            return []
    
    async def call_tool(self, tool_name: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Call a tool on the MCP server via HTTP.

        Args:
            tool_name: Name of the tool to call
            params: Parameters for the tool

        Returns:
            Tool response as JSON string
        """    
        try:
            url = f"{self.mcp_server_url}/mcp"
            
            logger.info(f"Calling MCP tool '{tool_name}' with params: {params}")

            # Connect to the server using Streamable HTTP
            async with streamablehttp_client(url) as (
                read_stream,
                write_stream,
                get_session_id,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize the connection
                    await session.initialize()
            
                    # Call the tool with the provided parameters under the expected 'params' key
                    payload = {"params": params or {}}
                    result = await session.call_tool(tool_name, payload)
                    
                    # Extract the text content from the result
                    if result.content and len(result.content) > 0:
                        response_text = result.content[0].text
                        logger.info(f"Successfully called tool '{tool_name}'")
                        return response_text
                    else:
                        logger.error(f"No content returned from tool '{tool_name}'")
                        return f'{{"error": "No content returned from tool"}}'

        except Exception as e:
            logger.error(f"Failed to call MCP tool '{tool_name}': {str(e)}")
            return f'{{"error": "Failed to call MCP tool: {str(e)}"}}'

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    global mcp_client
    
    try:
        logger.info("Starting OCP Benchmark API Server")
        logger.info("Testing Prometheus Connections...")
    
        # Test Prometheus connection
        if ocp_auth.test_prometheus_connection():
            logger.info("✅ Prometheus connection successful")
        else:
            logger.warning("⚠️ Prometheus connection failed")

        # Initialize MCP client
        mcp_client = MCPClient(MCP_SERVER_URL)
        
        logger.info("✅ MCP API Server started successfully")

    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MCP API Server")
    mcp_client = None

# Initialize FastAPI app
app = FastAPI(
    title="OpenShift Benchmark MCP API",
    description="FastAPI wrapper for OpenShift Benchmark MCP tools",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class APIResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# API endpoints
@app.get("/", response_model=APIResponse)
async def root():
    """Root endpoint with API information."""
    return APIResponse(
        success=True,
        data={
            "message": "OCP Benchmark API Server",
            "version": "1.0.0",
            "description": "FastAPI wrapper for OpenShift Benchmark MCP tools",
            "endpoints": [
                "/cluster-info - Get cluster information",
                "/node-info - Get node information", 
                "/nodes-usage - Get nodes usage metrics",
                "/pods-usage - Get pods usage metrics",
                "/disk-metrics - Get disk I/O metrics",
                "/network-metrics - Get network performance metrics",
                "/api-latency - Get API latency metrics",
                "/analyze-performance - Analyze performance data",
                "/tools - List available tools",
                "/tools/{tool_name} - Execute a tool (POST) or get its schema (GET)"
            ]
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


# Cluster Information Endpoints
@app.get("/cluster-info",  response_model=APIResponse)
async def get_cluster_info(
    include_operators: bool = Query(False, description="Include cluster operators status"),
    detailed_status: bool = Query(False, description="Return cluster information detailed status"),
):
    """Get OpenShift cluster information"""
    try:
        global mcp_client
        if not mcp_client:
            return APIResponse(success=False, error="MCP client not initialized")
        request_data = {
            "include_operators": include_operators,
            "detailed_status": detailed_status,
        }
        result = await mcp_client.call_tool("get_cluster_information", request_data)
        try:
            data = json.loads(result)
            if "error" in data:
                return APIResponse(success=False, error=data["error"])
            return APIResponse(success=True, data=data)
        except json.JSONDecodeError:
            # Fallback: return raw text when MCP server returns non-JSON
            return APIResponse(success=True, data={"result": result})
        
    except Exception as e:
        logger.error(f"Error in cluster-info endpoint: {e}")
        return APIResponse(success=False, error=str(e))

    
@app.get("/node-info", response_model=APIResponse)
async def get_node_info():
    """Get OpenShift node information."""
    try:
        global mcp_client
        if not mcp_client:
            return APIResponse(success=False, error="MCP client not initialized")
            
        result = await mcp_client.call_tool("get_node_information", {})
        
        # Parse JSON response
        try:
            data = json.loads(result)
            if "error" in data:
                return APIResponse(success=False, error=data["error"])
            return APIResponse(success=True, data=data)
        except json.JSONDecodeError:
            return APIResponse(success=True, data={"result": result})
        
    except Exception as e:
        logger.error(f"Error in node-info endpoint: {e}")
        return APIResponse(success=False, error=str(e))


@app.post("/nodes-usage", response_model=APIResponse)
async def get_nodes_usage_metrics(request: NodesUsageRequest):
    """Get nodes CPU and memory usage metrics."""
    try:
        global mcp_client
        if not mcp_client:
            return APIResponse(success=False, error="MCP client not initialized")
            
        params = {
            "duration_hours": request.duration_hours,
            "step": request.step
        }
        
        result = await mcp_client.call_tool("get_nodes_usage_metrics", params)
        
        # Parse JSON response
        try:
            data = json.loads(result)
            if "error" in data:
                return APIResponse(success=False, error=data["error"])
            return APIResponse(success=True, data=data)
        except json.JSONDecodeError:
            return APIResponse(success=True, data={"result": result})
        
    except Exception as e:
        logger.error(f"Error in nodes-usage endpoint: {e}")
        return APIResponse(success=False, error=str(e))


@app.post("/pods-usage", response_model=APIResponse)
async def get_pods_usage_metrics(request: PodsUsageRequest):
    """Get pods CPU and memory usage metrics."""
    try:
        global mcp_client
        if not mcp_client:
            return APIResponse(success=False, error="MCP client not initialized")
            
        params = {
            "duration_hours": request.duration_hours,
            "pod_regex": request.pod_regex,
            "label_selectors": request.label_selectors,
            "step": request.step
        }
        
        result = await mcp_client.call_tool("get_pods_usage_metrics", params)
        
        # Parse JSON response
        try:
            data = json.loads(result)
            if "error" in data:
                return APIResponse(success=False, error=data["error"])
            return APIResponse(success=True, data=data)
        except json.JSONDecodeError:
            return APIResponse(success=True, data={"result": result})
        
    except Exception as e:
        logger.error(f"Error in pods-usage endpoint: {e}")
        return APIResponse(success=False, error=str(e))


@app.post("/disk-metrics", response_model=APIResponse)
async def get_disk_io_metrics(request: DiskMetricsRequest):
    """Get disk I/O metrics."""
    try:
        global mcp_client
        if not mcp_client:
            return APIResponse(success=False, error="MCP client not initialized")
            
        params = {
            "duration_hours": request.duration_hours,
            "step": request.step
        }
        
        result = await mcp_client.call_tool("get_disk_io_metrics", params)
        
        # Parse JSON response
        try:
            data = json.loads(result)
            if "error" in data:
                return APIResponse(success=False, error=data["error"])
            return APIResponse(success=True, data=data)
        except json.JSONDecodeError:
            return APIResponse(success=True, data={"result": result})
        
    except Exception as e:
        logger.error(f"Error in disk-metrics endpoint: {e}")
        return APIResponse(success=False, error=str(e))


@app.post("/network-metrics", response_model=APIResponse)
async def get_network_performance_metrics(request: NetworkMetricsRequest):
    """Get network performance metrics."""
    try:
        global mcp_client
        if not mcp_client:
            return APIResponse(success=False, error="MCP client not initialized")
            
        params = {
            "duration_hours": request.duration_hours,
            "step": request.step
        }
        
        result = await mcp_client.call_tool("get_network_performance_metrics", params)
        
        # Parse JSON response
        try:
            data = json.loads(result)
            if "error" in data:
                return APIResponse(success=False, error=data["error"])
            return APIResponse(success=True, data=data)
        except json.JSONDecodeError:
            return APIResponse(success=True, data={"result": result})
        
    except Exception as e:
        logger.error(f"Error in network-metrics endpoint: {e}")
        return APIResponse(success=False, error=str(e))


@app.post("/api-request-latency", response_model=APIResponse)
async def get_api_request_latency_metrics(request: APILatencyRequest):
    """Get API latency metrics."""
    try:
        global mcp_client
        if not mcp_client:
            return APIResponse(success=False, error="MCP client not initialized")
            
        params = {
            "duration_hours": request.duration_hours,
            "step": request.step
        }
        
        result = await mcp_client.call_tool("get_api_request_latency_metrics", params)
        
        # Parse JSON response
        try:
            data = json.loads(result)
            if "error" in data:
                return APIResponse(success=False, error=data["error"])
            return APIResponse(success=True, data=data)
        except json.JSONDecodeError:
            return APIResponse(success=True, data={"result": result})
        
    except Exception as e:
        logger.error(f"Error in api-request-latency endpoint: {e}")
        return APIResponse(success=False, error=str(e))

@app.post("/api-request-rate", response_model=APIResponse)
async def get_api_request_rate_metrics(request: APILatencyRequest):
    """Get API latency metrics."""
    try:
        global mcp_client
        if not mcp_client:
            return APIResponse(success=False, error="MCP client not initialized")
            
        params = {
            "duration_hours": request.duration_hours,
            "step": request.step
        }
        
        result = await mcp_client.call_tool("get_api_request_rate_metrics", params)
        
        # Parse JSON response
        try:
            data = json.loads(result)
            if "error" in data:
                return APIResponse(success=False, error=data["error"])
            return APIResponse(success=True, data=data)
        except json.JSONDecodeError:
            return APIResponse(success=True, data={"result": result})
        
    except Exception as e:
        logger.error(f"Error in api-request-rate endpoint: {e}")
        return APIResponse(success=False, error=str(e))
    
@app.post("/etcd-latency", response_model=APIResponse)
async def get_etcd_latency_metrics(request: APILatencyRequest):
    """Get etcd latency metrics."""
    try:
        global mcp_client
        if not mcp_client:
            return APIResponse(success=False, error="MCP client not initialized")
            
        params = {
            "duration_hours": request.duration_hours,
            "step": request.step
        }
        
        result = await mcp_client.call_tool("get_etcd_latency_metrics", params)
        
        # Parse JSON response
        try:
            data = json.loads(result)
            if "error" in data:
                return APIResponse(success=False, error=data["error"])
            return APIResponse(success=True, data=data)
        except json.JSONDecodeError:
            return APIResponse(success=True, data={"result": result})
        
    except Exception as e:
        logger.error(f"Error in etcd-latency endpoint: {e}")
        return APIResponse(success=False, error=str(e))
    
@app.post("/analyze-performance", response_model=APIResponse)
async def analyze_performance_data(request: AnalyzePerformanceRequest):
    """Analyze performance data and generate insights."""
    try:
        global mcp_client
        if not mcp_client:
            return APIResponse(success=False, error="MCP client not initialized")
            
        params = {
            "metrics_data": request.metrics_data,
            "analysis_type": request.analysis_type
        }
        
        # Server tool is registered as 'analyze_ocp_performance_data'
        result = await mcp_client.call_tool("analyze_ocp_performance_data", params)
        
        # Parse JSON response
        try:
            data = json.loads(result)
            if "error" in data:
                return APIResponse(success=False, error=data["error"])
            return APIResponse(success=True, data=data)
        except json.JSONDecodeError:
            # Fallback: return raw text when MCP server returns non-JSON
            return APIResponse(success=True, data={"result": result})
        
    except Exception as e:
        logger.error(f"Error in analyze-performance endpoint: {e}")
        return APIResponse(success=False, error=str(e))


@app.get("/test-connection", response_model=APIResponse)
async def test_connection():
    """Test Prometheus connection."""
    try:
        if ocp_auth.test_prometheus_connection():
            return APIResponse(
                success=True, 
                data={"status": "connected", "message": "Prometheus connection successful"}
            )
        else:
            return APIResponse(
                success=False, 
                error="Prometheus connection failed"
            )
        
    except Exception as e:
        logger.error(f"Error in test-connection endpoint: {e}")
        return APIResponse(success=False, error=str(e))


@app.get("/tools", response_model=APIResponse)
async def list_tools():
    """List all available MCP tools exposed by the server."""
    try:
        global mcp_client
        if not mcp_client:
            return APIResponse(success=False, error="MCP client not initialized")
        tools = await mcp_client.list_available_tools()
        return APIResponse(success=True, data={"tools": tools})
    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        return APIResponse(success=False, error=str(e))


@app.get("/tools/{tool_name}", response_model=APIResponse)
async def get_tool_schema(tool_name: str):
    """Get a specific tool's definition (schema)."""
    try:
        global mcp_client
        if not mcp_client:
            return APIResponse(success=False, error="MCP client not initialized")
        tools = await mcp_client.list_available_tools()
        for tool in tools:
            if tool.get("name") == tool_name:
                return APIResponse(success=True, data={"tool": tool})
        return APIResponse(success=False, error=f"Tool '{tool_name}' not found")
    except Exception as e:
        logger.error(f"Error getting tool schema: {e}")
        return APIResponse(success=False, error=str(e))


@app.post("/tools/{tool_name}", response_model=APIResponse)
async def execute_tool(tool_name: str, params: Optional[Dict[str, Any]] = Body(default=None)):
    """Execute any MCP tool by name with arbitrary parameters."""
    try:
        global mcp_client
        if not mcp_client:
            return APIResponse(success=False, error="MCP client not initialized")
        result = await mcp_client.call_tool(tool_name, params or {})
        try:
            data = json.loads(result)
            if isinstance(data, dict) and "error" in data:
                return APIResponse(success=False, error=data["error"])
            return APIResponse(success=True, data=data if isinstance(data, dict) else {"result": data})
        except json.JSONDecodeError:
            return APIResponse(success=True, data={"result": result})
    except Exception as e:
        logger.error(f"Error executing tool '{tool_name}': {e}")
        return APIResponse(success=False, error=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "ocp_benchmark_mcp_api:app",
        host="0.0.0.0",
        port=8081,
        reload=False,
        log_level="info"
    )