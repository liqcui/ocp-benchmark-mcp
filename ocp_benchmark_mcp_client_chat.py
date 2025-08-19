#!/usr/bin/env python3
"""OpenShift Benchmark MCP Client with FastAPI and LangGraph integration."""
import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime,timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import uvicorn
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks,Request
from fastapi.responses import StreamingResponse, FileResponse,HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from elt.ocp_benchmark_elt import analyze_performance_data
from elt.ocp_benchmark_elt_reduce_json_level import convert_json_auto_reduce

from dotenv import load_dotenv
#MCP imports
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


# Set timezone to UTC
os.environ['TZ'] = 'UTC'
load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")    
api_key = os.getenv("GEMINI_API_KEY")

# Global variables
mcp_client = None
llm_agent = None
memory = None
MCP_SERVER_URL = "http://localhost:8000"

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    stream: Optional[bool] = True

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    tools_used: List[str] = []


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
                # headers={"accept": "application/json"}
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
                         "params": params or {}
                    }

                    print(f"Calling tool {tool_name} with params {request_data}",type(request_data))

                    result = await session.call_tool(tool_name, request_data)
                    # print("#*"*50)
                    # print("result in call_tool of mcp client:",result)
                    # print("#*"*50)
                    # print(f"{tool_name} = {result.content[0].text}")
                    json_data = json.loads(result.content[0].text)
                    return json_data

        except Exception as e:
            print(f"Call MCP tool failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

class MCPTool(BaseTool):
    """Custom tool for calling MCP server tools"""
    
    name: str
    description: str
    mcp_tool_name: str
    
    def _run(self, **kwargs) -> str:
        """Synchronous version - not used in async context"""
        raise NotImplementedError("Use async version")
    
    async def _arun(self, **kwargs) -> str:
        """Asynchronous tool execution"""
        try:
            if not mcp_client:
                return "Error: MCP client not initialized"
                  
            params= {
                    "arguments": kwargs or {}
                }
          
            
            # response = await mcp_client.post(f"{MCP_SERVER_URL}/mcp", json=payload)
            response=await mcp_client.call_tool(self.mcp_tool_name,params=params)
            print(response)
            if response is None:
                return "Error: No response from MCP tool"
            return json.dumps(response, indent=2)
        
        except Exception as e:
            logger.error(f"Error calling MCP tool {self.mcp_tool_name}: {e}")
            return f"Error calling MCP tool: {str(e)}"


# MCP Tools definitions
def create_mcp_tools() -> List[MCPTool]:
    """Create MCP tools for the LLM agent"""
    return [
        MCPTool(
            name="get_cluster_information",
            description="Get OpenShift cluster information including version, name, and infrastructure",
            mcp_tool_name="get_cluster_information"
        ),
        MCPTool(
            name="get_node_information",
            description="Get detailed information about cluster nodes including resources and status",
            mcp_tool_name="get_node_information"
        ),
        MCPTool(
            name="get_nodes_usage_metrics",
            description="Get CPU and memory usage metrics for cluster nodes. Parameters: duration_hours (int, default 1)",
            mcp_tool_name="get_nodes_usage_metrics"
        ),
        MCPTool(
            name="get_pods_usage_metrics",
            description="Get CPU and memory usage metrics for pods. Parameters: duration_hours (int), pod_patterns (list), label_selectors (list), top_n (int)",
            mcp_tool_name="get_pods_usage_metrics"
        ),
        MCPTool(
            name="get_etcd_pods_usage_metrics",
            description="Get CPU and memory usage metrics for etcd pods. Parameters: duration_hours (int), label_selectors (list), top_n (int)",
            mcp_tool_name="get_etcd_pods_usage_metrics"
        ),
        MCPTool(
            name="get_disk_io_metrics",
            description="Get disk I/O performance metrics. Parameters: duration_hours (int, default 1), by_device (bool, default False)",
            mcp_tool_name="get_disk_io_metrics"
        ),
        MCPTool(
            name="get_network_performance_metrics",
            description="Get network performance metrics. Parameters: duration_hours (int, default 1), by_interface (bool, default False)",
            mcp_tool_name="get_network_performance_metrics"
        ),
        MCPTool(
            name="get_api_request_latency_metrics",
            description="Get Kubernetes API server latency metrics. Parameters: duration_hours (int), slow_threshold_ms (float), include_slow_requests (bool)",
            mcp_tool_name="get_api_request_latency_metrics"
        ),
        MCPTool(
            name="get_api_request_rate_metrics",
            description="Get Kubernetes API server request rate metrics. Parameters: duration_hours (int), slow_threshold_ms (float), include_slow_requests (bool)",
            mcp_tool_name="get_api_request_rate_metrics"
        ),
        MCPTool(
            name="get_etcd_latency_metrics",
            description="Get Kubernetes etcd latency metrics. Parameters: duration_hours (int), slow_threshold_ms (float), include_slow_requests (bool)",
            mcp_tool_name="get_etcd_latency_metrics"
        ),                
        MCPTool(
            name="analyze_ocp_performance_data",
            description="Get the current baseline configuration and performance thresholds",
            mcp_tool_name="analyze_ocp_performance_data"
        ),
        MCPTool(
            name="analyze_ocp_overall_cluster_performance",
            description="Get the current cluster overall cluster performance",
            mcp_tool_name="analyze_ocp_overall_cluster_performance"
        ),
        MCPTool(
                name="identify_performance_bottlenecks",
                description="Analyze cluster performance bottlenecks",
                mcp_tool_name="identify_performance_bottlenecks"
            ),
        MCPTool(
            name="generate_performance_recommendations",
            description="Generate ocp cluster performance recommendations",
            mcp_tool_name="generate_performance_recommendations"
        )
        
    ]

def create_llm_agent():
    """Create the ReAct agent with LangGraph (create_react_agent)."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL")    

    llm = ChatOpenAI(
            model="gemini-1.5-flash",
            base_url=base_url,
            api_key=api_key,
            temperature=0.1,
            streaming=True         
        )
    # 2. Tools
    tools = create_mcp_tools()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are an expert OpenShift cluster performance analyst. You help users monitor, analyze, and troubleshoot OpenShift cluster performance using various metrics and tools.

Key capabilities:
- Analyze cluster health and performance metrics
- Monitor node and pod resource usage
- Investigate disk I/O and network performance issues
- Review API server latency and responsiveness
- Compare current performance against baseline thresholds
- Generate comprehensive performance reports
- Provide actionable recommendations for performance optimization
- Collecting cluster, node, and pod information
- Gathering performance metrics (CPU, memory, disk I/O, network, API latency)
- Analyzing performance data against baselines
- Identifying bottlenecks and performance issues
- Providing optimization recommendations


When presenting data:
- Format JSON responses as readable tables when possible
- Highlight critical issues and performance problems
- Provide context and explanations for metrics
- Suggest specific remediation steps
- Compare values against baseline thresholds
- Determine what data you need to collect
- Use the appropriate tools to gather metrics
- Analyze the data to identify issues
- Provide clear, actionable recommendations

Always explain your findings in business terms and provide specific, actionable recommendations.
Always be thorough in your analysis and provide actionable insights. and provide specific, actionable recommendations."""),
        MessagesPlaceholder(variable_name="messages"),
        # MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]) 

    # 4. 使用 LangGraph 的 MemorySaver 保存对话历史（等效于 ConversationBufferWindowMemory）
    memory = MemorySaver()
    # 5. 创建 ReAct agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
        checkpointer=memory,
    )

    return agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    global mcp_client, llm_agent
    logger.info("Starting MCP Client Chat Interface")
    
    try:
        # Initialize HTTP client for MCP communication
        mcp_client = MCPClient(MCP_SERVER_URL)
        
        # Initialize LLM agent
        llm_agent = create_llm_agent()
        
        # Wait for MCP server to be ready
        await asyncio.sleep(6)
        
        logger.info("MCP Client Chat Interface started successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down MCP Client Chat Interface")
    if mcp_client:
        await mcp_client.aclose()


# Initialize FastAPI app
app = FastAPI(
    title="OpenShift Benchmark MCP Chat",
    description="LLM-powered chat interface for OpenShift performance monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# Setup templates and static files
templates = Jinja2Templates(directory="html")


def format_json_as_table(data: str) -> str:
    """Format JSON data as HTML tables for better readability"""
    try:
        parsed_data = json.loads(data)
        return _json_to_html_table(parsed_data)
    except:
        return data

def _json_to_html_table(obj: Any, level: int = 0) -> str:
    """
    Convert JSON-like Python object (dict / list / primitive) to a nested HTML table.
    """
    margin = level * 20

    # ---------- dict ----------
    if isinstance(obj, dict):
        if not obj:
            return "Empty"

        rows = []
        # Compute simple widths for two columns (key/value)
        default_nested_width = 30
        max_key_len = 0
        max_val_len = 0
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                cell = _json_to_html_table(value, level + 1)
                val_len = default_nested_width
            else:
                cell = str(value)
                val_len = len(cell)
            rows.append(
                f"<tr>"
                f"<td style='padding: 0 1ch; white-space: nowrap;'><strong>{key}</strong></td>"
                f"<td style='padding: 0 1ch; white-space: nowrap;'>{cell}</td>"
                f"</tr>"
            )
            if isinstance(key, str):
                max_key_len = max(max_key_len, len(key))
            else:
                max_key_len = max(max_key_len, len(str(key)))
            max_val_len = max(max_val_len, val_len)

        # Add a little padding and cap widths
        key_w = min(max_key_len + 2, 60)
        val_w = min(max_val_len + 2, 120)

        return (
            f"<table class='table table-sm table-bordered' "
            f"style='margin-left: {margin}px; width: auto;'>"
            f"<colgroup><col style='width:{key_w}ch'><col style='width:{val_w}ch'></colgroup>"
            f"{''.join(rows)}</table>"
        )

    # ---------- list ----------
    elif isinstance(obj, list):
        if not obj:
            return "Empty list"

        # 如果列表里全是 dict，则做成一张横向表头统一的表格
        if all(isinstance(item, dict) for item in obj):
            headers = list({k for d in obj for k in d.keys()})
            thead = "<thead><tr>" + "".join(f"<th style='padding: 0 1ch; white-space: nowrap;'>{h}</th>" for h in headers) + "</tr></thead>"

            # Compute per-column widths based on max text length
            default_nested_width = 30
            col_widths = [len(str(h)) for h in headers]

            tbody_rows = []
            for item in obj:
                row_cells = []
                for idx, h in enumerate(headers):
                    v = item.get(h, "")
                    if isinstance(v, (dict, list)):
                        v_html = _json_to_html_table(v, level + 1)
                        display_len = default_nested_width
                        row_cells.append(f"<td style='padding: 0 1ch; white-space: nowrap;'>{v_html}</td>")
                    else:
                        v_str = str(v)
                        display_len = len(v_str)
                        row_cells.append(f"<td style='padding: 0 1ch; white-space: nowrap;'>{v_str}</td>")
                    col_widths[idx] = max(col_widths[idx], display_len)
                tbody_rows.append("<tr>" + "".join(row_cells) + "</tr>")
            tbody = "<tbody>" + "".join(tbody_rows) + "</tbody>"

            # Add padding and caps
            col_styles = []
            for w in col_widths:
                w2 = min(w + 2, 80)
                col_styles.append(f"<col style='width:{w2}ch'>")
            colgroup = "<colgroup>" + "".join(col_styles) + "</colgroup>"

            return (
                f"<table class='table table-sm table-striped table-bordered' "
                f"style='margin-left: {margin}px; width: auto;'>{colgroup}{thead}{tbody}</table>"
            )

        # 普通列表 → 简单 <ul>
        items = "".join(f"<li>{_json_to_html_table(v, level + 1)}</li>" for v in obj)
        return f"<ul style='margin-left: {margin}px; width: auto;'>{items}</ul>"

    # ---------- primitive ----------
    else:
        return str(obj)
    

@app.get("/", response_class=HTMLResponse)

async def chat_interface(request: Request):
    """Serve the chat interface"""
    return templates.TemplateResponse("ocp_benchmark_mcp_llm.html", {"request": request})


@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    """Chat endpoint for interacting with the LLM"""
    global llm_agent, memory
    try:
        if not llm_agent:
            raise HTTPException(status_code=503, detail="LLM agent not initialized")
        
        logger.info(f"Processing chat message: {chat_request.message}")
        
        if chat_request.stream:
 
            return StreamingResponse(
                stream_chat_response(chat_request.message),
                media_type="text/plain"
            )
        else:
            # Non-streaming response
            thread_uuid=uuid.uuid4()
            config = {"configurable": {"thread_id": "ocp_benchmark_mcp_chat"+str(thread_uuid)}}

            response = await llm_agent.ainvoke(
                # "input": chat_request.message
                 {
                  "messages": [HumanMessage(content=chat_request.message)],
                  },
           
                 config=config)
            print("#*"*30)
            print("response in chat Non-streaming response:\n",response)
            # print("#*"*30)
            # print("response type:",type(response))
            tool_msg = next( m for m in response["messages"]
                      if m.__class__.__name__ == "ToolMessage"
            )
            content_json = tool_msg.content          # 字符串
            print("content_json is ,",type(content_json))
            # output = json.dump(content_json)  # 变成 Python dict
            # print("cluster info:", output,type(output))
            # print("#*"*30)
            # formatted_response = format_json_as_table(content_json)
            formatted_response=content_json
            return ChatResponse(
                response=formatted_response,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


async def stream_chat_response(message: str):
    """Stream the chat response"""
    thread_uuid=uuid.uuid4()
    try:
        config = {"configurable": {"thread_id": "ocp_benchmark_mcp_stream_chat"+str(thread_uuid)}}
        response=[]
        response = await llm_agent.ainvoke(
            # "input": chat_request.message
                {
                "messages": [HumanMessage(content=message)],
                },        
                config=config)
        
        print("#*"*30)
        print("response in stream_chat_response:\n",response)
 
        tool_msg = next( m for m in response["messages"]
                    if m.__class__.__name__ == "ToolMessage"
        )
        content_json = tool_msg.content          # 字符串
        print("#*"*30)
        print("content_json is :\n",content_json)
        # print("content_json is ,",type(content_json))
        low_level_json = convert_json_auto_reduce(content_json)
        print("low_level_json is:\n",low_level_json)
        formatted_response = format_json_as_table(low_level_json)
        #formatted_response = content_json
        # Stream the response in chunks
        chunk_size = 50
        for i in range(0, len(formatted_response), chunk_size):
            chunk = formatted_response[i:i + chunk_size]
            # print(f"data: {json.dumps({'content': chunk, 'done': False})}\n\n")
            yield f"data: {json.dumps({'content': chunk, 'done': False})}\n\n"
            await asyncio.sleep(0.01)  # Small delay for better streaming effect
        
        # Send completion signal
        yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        yield f"data: {json.dumps({'content': error_msg, 'done': True, 'error': True})}\n\n"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mcp_server": MCP_SERVER_URL,
        "llm_agent": "initialized" if llm_agent else "not initialized"
    }


async def main():
    """Main entry point."""
    logger.info("Starting OpenShift Benchmark MCP Client")
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set, AI features will be limited")
    
    # Run the server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
        access_log=True
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Client shutdown requested")
    except Exception as e:
        logger.error(f"Client error: {e}")
        sys.exit(1)