#!/usr/bin/env python3
"""OpenShift Benchmark MCP Client with LLM Chat Interface"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Set timezone to UTC
os.environ['TZ'] = 'UTC'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S UTC'
)
logger = logging.getLogger(__name__)

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
            
            payload = {
                "method": "tools/call",
                "params": {
                    "name": self.mcp_tool_name,
                    "arguments": kwargs
                }
            }
            
            response = await mcp_client.post(f"{MCP_SERVER_URL}/mcp", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result and "content" in result["result"]:
                    content = result["result"]["content"][0]["text"]
                    return content
                else:
                    return json.dumps(result, indent=2)
            else:
                return f"Error: MCP tool call failed with status {response.status_code}"
        
        except Exception as e:
            logger.error(f"Error calling MCP tool {self.mcp_tool_name}: {e}")
            return f"Error calling MCP tool: {str(e)}"


# MCP Tools definitions
def create_mcp_tools() -> List[MCPTool]:
    """Create MCP tools for the LLM agent"""
    return [
        MCPTool(
            name="get_cluster_info",
            description="Get OpenShift cluster information including version, name, and infrastructure",
            mcp_tool_name="get_cluster_info"
        ),
        MCPTool(
            name="get_node_info",
            description="Get detailed information about cluster nodes including resources and status",
            mcp_tool_name="get_node_info"
        ),
        MCPTool(
            name="get_node_usage",
            description="Get CPU and memory usage metrics for cluster nodes. Parameters: duration_hours (int, default 1)",
            mcp_tool_name="get_node_usage_metrics"
        ),
        MCPTool(
            name="get_pod_usage",
            description="Get CPU and memory usage metrics for pods. Parameters: duration_hours (int), pod_patterns (list), label_selectors (list), top_n (int)",
            mcp_tool_name="get_pod_usage_metrics"
        ),
        MCPTool(
            name="get_disk_metrics",
            description="Get disk I/O performance metrics. Parameters: duration_hours (int, default 1), by_device (bool, default False)",
            mcp_tool_name="get_disk_io_metrics"
        ),
        MCPTool(
            name="get_network_metrics",
            description="Get network performance metrics. Parameters: duration_hours (int, default 1), by_interface (bool, default False)",
            mcp_tool_name="get_network_metrics"
        ),
        MCPTool(
            name="get_api_latency",
            description="Get Kubernetes API server latency metrics. Parameters: duration_hours (int), slow_threshold_ms (float), include_slow_requests (bool)",
            mcp_tool_name="get_api_latency_metrics"
        ),
        MCPTool(
            name="get_baseline_config",
            description="Get the current baseline configuration and performance thresholds",
            mcp_tool_name="get_baseline_configuration"
        ),
        MCPTool(
            name="get_comprehensive_report",
            description="Generate a comprehensive performance report across all metrics. Parameters: duration_hours (int, default 1)",
            mcp_tool_name="get_comprehensive_performance_report"
        )
    ]


def create_llm_agent():
    """Create the LLM agent with MCP tools"""
    global memory
    
    # Initialize memory
    memory = ConversationBufferWindowMemory(
        k=10,  # Keep last 10 conversations
        memory_key="chat_history",
        return_messages=True
    )
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Use a more cost-effective model for demos
        temperature=0,
        streaming=True
    )
    
    # Create tools
    tools = create_mcp_tools()
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert OpenShift cluster performance analyst. You help users monitor, analyze, and troubleshoot OpenShift cluster performance using various metrics and tools.

Key capabilities:
- Analyze cluster health and performance metrics
- Monitor node and pod resource usage
- Investigate disk I/O and network performance issues
- Review API server latency and responsiveness
- Compare current performance against baseline thresholds
- Generate comprehensive performance reports
- Provide actionable recommendations for performance optimization

When presenting data:
- Format JSON responses as readable tables when possible
- Highlight critical issues and performance problems
- Provide context and explanations for metrics
- Suggest specific remediation steps
- Compare values against baseline thresholds

Always be thorough in your analysis and provide actionable insights."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    return agent_executor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    global mcp_client, llm_agent
    logger.info("Starting MCP Client Chat Interface")
    
    try:
        # Initialize HTTP client for MCP communication
        mcp_client = httpx.AsyncClient(timeout=120.0)
        
        # Initialize LLM agent
        llm_agent = create_llm_agent()
        
        # Wait for MCP server to be ready
        await asyncio.sleep(2)
        
        # Test connection
        try:
            response = await mcp_client.get(f"{MCP_SERVER_URL}/health")
            logger.info("Successfully connected to MCP server")
        except Exception as e:
            logger.warning(f"Could not connect to MCP server: {e}")
        
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
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                cell = _json_to_html_table(value, level + 1)
            else:
                cell = str(value)
            rows.append(f"<tr><td><strong>{key}</strong></td><td>{cell}</td></tr>")

        return (
            f"<table class='table table-sm table-bordered' "
            f"style='margin-left: {margin}px;'>"
            f"{''.join(rows)}</table>"
        )

    # ---------- list ----------
    elif isinstance(obj, list):
        if not obj:
            return "Empty list"

        # 如果列表里全是 dict，则做成一张横向表头统一的表格
        if all(isinstance(item, dict) for item in obj):
            headers = list({k for d in obj for k in d.keys()})
            thead = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>"

            tbody_rows = []
            for item in obj:
                row_cells = []
                for h in headers:
                    v = item.get(h, "")
                    if isinstance(v, (dict, list)):
                        v = _json_to_html_table(v, level + 1)
                    row_cells.append(f"<td>{v}</td>")
                tbody_rows.append("<tr>" + "".join(row_cells) + "</tr>")
            tbody = "<tbody>" + "".join(tbody_rows) + "</tbody>"

            return (
                f"<table class='table table-sm table-striped table-bordered' "
                f"style='margin-left: {margin}px;'>{thead}{tbody}</table>"
            )

        # 普通列表 → 简单 <ul>
        items = "".join(f"<li>{_json_to_html_table(v, level + 1)}</li>" for v in obj)
        return f"<ul style='margin-left: {margin}px;'>{items}</ul>"

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
            response = await llm_agent.ainvoke({
                "input": chat_request.message
            })
            
            formatted_response = format_json_as_table(response.get("output", ""))
            
            return ChatResponse(
                response=formatted_response,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


async def stream_chat_response(message: str):
    """Stream the chat response"""
    try:
        response = await llm_agent.ainvoke({
            "input": message
        })
        
        output = response.get("output", "")
        formatted_output = format_json_as_table(output)
        
        # Stream the response in chunks
        chunk_size = 50
        for i in range(0, len(formatted_output), chunk_size):
            chunk = formatted_output[i:i + chunk_size]
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


@app.get("/chat/history")
async def get_chat_history():
    """Get chat history"""
    try:
        if not memory:
            return {"messages": []}
        
        messages = []
        for message in memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                messages.append({
                    "type": "human",
                    "content": message.content,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            elif isinstance(message, AIMessage):
                messages.append({
                    "type": "ai",
                    "content": message.content,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        
        return {"messages": messages}
    
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return {"messages": [], "error": str(e)}


@app.delete("/chat/history")
async def clear_chat_history():
    """Clear chat history"""
    try:
        if memory:
            memory.clear()
        return {"message": "Chat history cleared"}
    
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")


@app.get("/tools")
async def list_available_tools():
    """List available MCP tools"""
    tools = create_mcp_tools()
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "mcp_tool_name": tool.mcp_tool_name
            }
            for tool in tools
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "ocp_benchmark_mcp_client_chat:app",
        host="0.0.0.0",
        port=8081,
        log_level="info",
        access_log=True,
        reload=False
    )