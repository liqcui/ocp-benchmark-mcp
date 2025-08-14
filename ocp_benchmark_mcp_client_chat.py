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
from langchain.schema import HumanMessage, AIMessage,SystemMessage
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
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
            # if response.status_code == 200:
            #     result = response.json()
            #     if "result" in result and "content" in result["result"]:
            #         content = result["result"]["content"][0]["text"]
            #         return content
            #     else:
            #         return json.dumps(result, indent=2)
            # else:
            #     return f"Error: MCP tool call failed with status {response.status_code}"
        
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
    """Create the ReAct agent with LangGraph (create_react_agent)."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("BASE_URL")    
    # 1. LLM
    # llm = ChatOpenAI(
    #     model="gpt-4o-mini",
    #     temperature=0,
    #     streaming=True
    # )
    # llm=ChatOpenAI(
    #         model="deepseek-r1:14b",
    #         openai_api_base="xxxx",
    #         openai_api_key="anykey",
    #         temperature=0, 
    #         streaming=True
    #     ) 

    # llm = ChatOpenAI(
    #     model="gemini-2.0-flash-001",          # 任意支持 tools 的 Gemini 模型
    #     openai_api_base=base_url,
    #     openai_api_key=api_key,   # 字符串即可
    #     temperature=0,
    #     max_tokens=4096
    # )

    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0,
    max_output_tokens=4096
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

When presenting data:
- Format JSON responses as readable tables when possible
- Highlight critical issues and performance problems
- Provide context and explanations for metrics
- Suggest specific remediation steps
- Compare values against baseline thresholds

Always be thorough in your analysis and provide actionable insights."""),
        MessagesPlaceholder(variable_name="messages"),
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

    # 6. 返回一个可调用对象，用法与 AgentExecutor 一致：
    #    agent.invoke({"messages": [...]}, config={"configurable": {"thread_id": "abc123"}})
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
        await asyncio.sleep(15)
        
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
            print("#*"*30)
            print("chat_request.stream in", chat_request.message)
            return StreamingResponse(
                stream_chat_response(chat_request.message),
                media_type="text/plain"
            )
        else:
            # Non-streaming response
            config = {"configurable": {"thread_id": "ocp_benchmark_mcp_chat"}}
            print("#*"*30)
            print(chat_request.message)
            response = await llm_agent.ainvoke(
                # "input": chat_request.message
                 {"messages": [HumanMessage(content=chat_request.message)]},
                 config=config)
            print("#*"*30)
            print(response)            
            formatted_response = format_json_as_table(response.get("output", ""))
            
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
    try:
        config = {"configurable": {"thread_id": "ocp_benchmark_mcp_stream_chat"}}
        response = await llm_agent.ainvoke({
            "input": message
        },config=config)
        
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