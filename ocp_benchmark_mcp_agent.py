#!/usr/bin/env python3
"""OpenShift Benchmark MCP AI Agent using LangGraph"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

import httpx
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import pandas as pd

from analysis.ocp_benchmark_performance_anlysis import PerformanceAnalyzer

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
MCP_SERVER_URL = "http://localhost:8000"

# State definition for LangGraph
class AgentState(TypedDict):
    messages: List[BaseMessage]
    analysis_results: Dict[str, Any]
    performance_data: Dict[str, Any]
    recommendations: List[str]
    report_data: Dict[str, Any]
    current_task: str
    error: Optional[str]


class MCPAgentTool(BaseTool):
    """Custom tool for calling MCP server from agent"""
    
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
                return json.dumps({"error": "MCP client not initialized"})
            
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
                    return result["result"]["content"][0]["text"]
                else:
                    return json.dumps(result)
            else:
                return json.dumps({"error": f"MCP call failed: {response.status_code}"})
        
        except Exception as e:
            logger.error(f"Error calling MCP tool {self.mcp_tool_name}: {e}")
            return json.dumps({"error": str(e)})


def create_agent_tools() -> List[MCPAgentTool]:
    """Create tools for the AI agent"""
    return [
        MCPAgentTool(
            name="get_cluster_info",
            description="Get OpenShift cluster information and status",
            mcp_tool_name="get_cluster_info"
        ),
        MCPAgentTool(
            name="get_node_info",
            description="Get detailed node information and resources",
            mcp_tool_name="get_node_info"
        ),
        MCPAgentTool(
            name="get_comprehensive_report",
            description="Generate comprehensive performance report with all metrics",
            mcp_tool_name="get_comprehensive_performance_report"
        ),
        MCPAgentTool(
            name="get_baseline_config",
            description="Get baseline configuration and thresholds",
            mcp_tool_name="get_baseline_configuration"
        )
    ]


class OpenShiftBenchmarkAgent:
    """AI Agent for OpenShift benchmark analysis using LangGraph"""
    
    def __init__(self):
        self.mcp_client = None
        self.llm = None
        self.tools = []
        self.graph = None
        self.performance_analyzer = PerformanceAnalyzer()
        
    async def initialize(self):
        """Initialize the agent"""
        try:
            # Initialize HTTP client
            global mcp_client
            self.mcp_client = httpx.AsyncClient(timeout=120.0)
            mcp_client = self.mcp_client
            
            # Initialize LLM
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                streaming=False
            )
            
            # Initialize tools
            self.tools = create_agent_tools()
            
            # Build the graph
            self.graph = self._build_graph()
            
            # Test MCP connection
            await self._test_mcp_connection()
            
            logger.info("OpenShift Benchmark Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing agent: {e}")
            raise
    
    async def _test_mcp_connection(self):
        """Test connection to MCP server"""
        try:
            response = await self.mcp_client.get(f"{MCP_SERVER_URL}/health")
            if response.status_code == 200:
                logger.info("Successfully connected to MCP server")
            else:
                logger.warning(f"MCP server responded with status: {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to MCP server: {e}")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Define the workflow steps
        async def collect_data(state: AgentState) -> AgentState:
            """Collect performance data from MCP server"""
            logger.info("Collecting performance data...")
            
            try:
                # Get comprehensive report
                comprehensive_tool = next(t for t in self.tools if t.name == "get_comprehensive_report")
                report_data = await comprehensive_tool._arun(duration_hours=1)
                
                # Get baseline configuration
                baseline_tool = next(t for t in self.tools if t.name == "get_baseline_config")
                baseline_data = await baseline_tool._arun()
                
                # Parse the JSON responses
                report = json.loads(report_data) if isinstance(report_data, str) else report_data
                baseline = json.loads(baseline_data) if isinstance(baseline_data, str) else baseline_data
                
                state["performance_data"] = report
                state["analysis_results"] = {"baseline": baseline}
                state["current_task"] = "data_collected"
                
                logger.info("Successfully collected performance data")
                
            except Exception as e:
                logger.error(f"Error collecting data: {e}")
                state["error"] = str(e)
                state["current_task"] = "error"
            
            return state
        
        async def analyze_performance(state: AgentState) -> AgentState:
            """Analyze collected performance data"""
            logger.info("Analyzing performance data...")
            
            try:
                if "error" in state:
                    return state
                
                performance_data = state.get("performance_data", {})
                baseline = state.get("analysis_results", {}).get("baseline", {})
                
                # Use the performance analyzer
                analysis = await self.performance_analyzer.analyze_comprehensive_data(
                    performance_data, baseline
                )
                
                state["analysis_results"].update(analysis)
                state["current_task"] = "analysis_complete"
                
                logger.info("Performance analysis completed")
                
            except Exception as e:
                logger.error(f"Error analyzing performance: {e}")
                state["error"] = str(e)
                state["current_task"] = "error"
            
            return state
        
        async def generate_recommendations(state: AgentState) -> AgentState:
            """Generate recommendations based on analysis"""
            logger.info("Generating recommendations...")
            
            try:
                if "error" in state:
                    return state
                
                analysis_results = state.get("analysis_results", {})
                
                # Generate recommendations using the analyzer
                recommendations = await self.performance_analyzer.generate_recommendations(
                    analysis_results
                )
                
                state["recommendations"] = recommendations
                state["current_task"] = "recommendations_complete"
                
                logger.info("Recommendations generated")
                
            except Exception as e:
                logger.error(f"Error generating recommendations: {e}")
                state["error"] = str(e)
                state["current_task"] = "error"
            
            return state
        
        async def create_report(state: AgentState) -> AgentState:
            """Create final report with analysis and recommendations"""
            logger.info("Creating final report...")
            
            try:
                if "error" in state:
                    return state
                
                report_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "performance_data": state.get("performance_data", {}),
                    "analysis_results": state.get("analysis_results", {}),
                    "recommendations": state.get("recommendations", []),
                    "summary": self._generate_summary(state)
                }
                
                # Export reports
                await self._export_reports(report_data)
                
                state["report_data"] = report_data
                state["current_task"] = "report_complete"
                
                logger.info("Final report created and exported")
                
            except Exception as e:
                logger.error(f"Error creating report: {e}")
                state["error"] = str(e)
                state["current_task"] = "error"
            
            return state
        
        # Build the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("collect_data", collect_data)
        workflow.add_node("analyze_performance", analyze_performance)
        workflow.add_node("generate_recommendations", generate_recommendations)
        workflow.add_node("create_report", create_report)
        
        # Add edges
        workflow.set_entry_point("collect_data")
        workflow.add_edge("collect_data", "analyze_performance")
        workflow.add_edge("analyze_performance", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "create_report")
        workflow.add_edge("create_report", END)
        
        return workflow.compile()
    
    def _generate_summary(self, state: AgentState) -> Dict[str, Any]:
        """Generate executive summary of the analysis"""
        try:
            analysis = state.get("analysis_results", {})
            performance_data = state.get("performance_data", {})
            recommendations = state.get("recommendations", [])
            
            summary = {
                "overall_health": "healthy",
                "critical_issues": [],
                "key_findings": [],
                "priority_actions": recommendations[:3] if recommendations else []
            }
            
            # Determine overall health
            if analysis.get("critical_issues"):
                summary["overall_health"] = "critical"
                summary["critical_issues"] = analysis.get("critical_issues", [])
            elif analysis.get("warnings"):
                summary["overall_health"] = "warning"
            
            # Extract key findings
            if "cluster_health_score" in analysis:
                summary["key_findings"].append(
                    f"Cluster health score: {analysis['cluster_health_score']:.1f}/10"
                )
            
            if "performance_trends" in analysis:
                trends = analysis["performance_trends"]
                for metric, trend in trends.items():
                    if trend.get("trend") == "degrading":
                        summary["key_findings"].append(
                            f"{metric} performance is degrading"
                        )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {"error": str(e)}
    
    async def _export_reports(self, report_data: Dict[str, Any]):
        """Export reports to Excel and PDF"""
        try:
            # Ensure exports directory exists
            os.makedirs("exports", exist_ok=True)
            
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            
            # Export to Excel
            await self._export_to_excel(report_data, f"exports/performance_report_{timestamp}.xlsx")
            
            # Export to PDF
            await self._export_to_pdf(report_data, f"exports/performance_report_{timestamp}.pdf")
            
            logger.info(f"Reports exported with timestamp: {timestamp}")
            
        except Exception as e:
            logger.error(f"Error exporting reports: {e}")
    
    async def _export_to_excel(self, report_data: Dict[str, Any], filepath: str):
        """Export report to Excel format"""
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = []
                summary = report_data.get("summary", {})
                
                summary_data.append(["Overall Health", summary.get("overall_health", "unknown")])
                summary_data.append(["Timestamp", report_data.get("timestamp", "")])
                summary_data.append(["Critical Issues", len(summary.get("critical_issues", []))])
                summary_data.append(["Recommendations", len(report_data.get("recommendations", []))])
                
                summary_df = pd.DataFrame(summary_data, columns=["Metric", "Value"])
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                
                # Performance data sheet
                perf_data = report_data.get("performance_data", {})
                if perf_data:
                    # Flatten performance data for Excel
                    flattened_data = self._flatten_dict(perf_data)
                    perf_df = pd.DataFrame(list(flattened_data.items()), columns=["Metric", "Value"])
                    perf_df.to_excel(writer, sheet_name="Performance Data", index=False)
                
                # Recommendations sheet
                recommendations = report_data.get("recommendations", [])
                if recommendations:
                    rec_df = pd.DataFrame(recommendations, columns=["Recommendation"])
                    rec_df.to_excel(writer, sheet_name="Recommendations", index=False)
            
            logger.info(f"Excel report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
    
    async def _export_to_pdf(self, report_data: Dict[str, Any], filepath: str):
        """Export report to PDF format"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            
            doc = SimpleDocTemplate(filepath, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.darkblue
            )
            story.append(Paragraph("OpenShift Performance Report", title_style))
            story.append(Spacer(1, 12))
            
            # Summary section
            summary = report_data.get("summary", {})
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            
            summary_data = [
                ["Overall Health Status", summary.get("overall_health", "unknown")],
                ["Report Timestamp", report_data.get("timestamp", "")],
                ["Critical Issues Found", str(len(summary.get("critical_issues", [])))],
                ["Total Recommendations", str(len(report_data.get("recommendations", [])))]
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 20))
            
            # Key findings
            if summary.get("key_findings"):
                story.append(Paragraph("Key Findings", styles['Heading2']))
                for finding in summary.get("key_findings", []):
                    story.append(Paragraph(f"• {finding}", styles['Normal']))
                story.append(Spacer(1, 20))
            
            # Recommendations
            recommendations = report_data.get("recommendations", [])
            if recommendations:
                story.append(Paragraph("Recommendations", styles['Heading2']))
                for i, rec in enumerate(recommendations[:10], 1):  # Limit to top 10
                    story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
                    story.append(Spacer(1, 6))
            
            doc.build(story)
            logger.info(f"PDF report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting to PDF: {e}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary for Excel export"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, str(v)))
        return dict(items)
    
    async def analyze_cluster_performance(self, duration_hours: int = 1) -> Dict[str, Any]:
        """Main method to analyze cluster performance"""
        try:
            logger.info(f"Starting cluster performance analysis for {duration_hours} hours")
            
            initial_state = AgentState(
                messages=[HumanMessage(content=f"Analyze cluster performance for {duration_hours} hours")],
                analysis_results={},
                performance_data={},
                recommendations=[],
                report_data={},
                current_task="starting",
                error=None
            )
            
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)
            
            if final_state.get("error"):
                logger.error(f"Analysis failed: {final_state['error']}")
                return {"error": final_state["error"]}
            
            logger.info("Cluster performance analysis completed successfully")
            return final_state.get("report_data", {})
            
        except Exception as e:
            logger.error(f"Error in cluster performance analysis: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close the agent and cleanup resources"""
        if self.mcp_client:
            await self.mcp_client.aclose()
        logger.info("OpenShift Benchmark Agent closed")


async def main():
    """Main function to run the agent"""
    agent = OpenShiftBenchmarkAgent()
    
    try:
        await agent.initialize()
        
        # Run analysis
        result = await agent.analyze_cluster_performance(duration_hours=2)
        
        if "error" in result:
            print(f"Analysis failed: {result['error']}")
        else:
            print("Analysis completed successfully!")
            print(f"Report timestamp: {result.get('timestamp')}")
            print(f"Overall health: {result.get('summary', {}).get('overall_health')}")
            print(f"Recommendations: {len(result.get('recommendations', []))}")
    
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Error running agent: {e}")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())