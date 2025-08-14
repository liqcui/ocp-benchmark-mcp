ocp-benchmark-mcp/
├── README.md                           # Complete documentation
├── pyproject.toml                      # Python project configuration
├── ocp_benchmark_mcp_server.py         # Main MCP server
├── ocp_benchmark_mcp_client_chat.py    # Interactive client with FastAPI
├── ocp_benchmark_mcp_agent.py          # AI agent with StateGraph
├── ocp_benchmark_mcp_command.sh        # Startup script
├── ocauth/
│   └── ocp_benchmark_auth.py           # OpenShift authentication
├── tools/                              # MCP tools
│   ├── ocp_benchmark_openshift_clusterinfo.py
│   ├── ocp_benchmark_openshift_nodeinfo.py
│   ├── ocp_benchmark_prometheus_basequery.py
│   ├── ocp_benchmark_prometheus_nodes_usage.py
│   ├── ocp_benchmark_prometheus_pods_usage.py
│   ├── ocp_benchmark_prometheus_diskio.py
│   ├── ocp_benchmark_prometheus_network.py
│   └── ocp_benchmark_prometheus_apilatency.py
├── config/
│   ├── ocp_benchmark_config.py         # Configuration management
│   ├── baseline.properties             # Performance baselines
│   └── metrics.yml                     # PromQL queries
├── analysis/
│   └── ocp_benchmark_performance_analysis.py
├── elt/
│   └── ocp_benchmark_elt.py  # Data processing
├── exports/                            # Generated reports
└── html/
    └── ocp_benchmark_mcp_llm.html      # Web interface
 MCP Server (ocp_benchmark_mcp_server.py)

✅ Uses fastMCP with streamable-http transport (not stdio)
✅ Exposes 8 MCP tools for OpenShift monitoring
✅ Pydantic models with arbitrary_types_allowed = True and extra = "allow"
✅ UTC timezone configuration

2. Authentication (ocauth/ocp_benchmark_auth.py)

✅ Auto-discovers Prometheus URL from OpenShift monitoring namespace
✅ Creates SA tokens using oc create token or oc sa new-token
✅ Tests Prometheus connectivity

3. Performance Baselines (config/baseline.properties)

✅ Configurable CPU, memory, disk I/O, network, API latency baselines
✅ Warning and critical thresholds
✅ Uses configparser.ConfigParser(interpolation=None)

4. Prometheus Tools

✅ Base query client with instant and range queries
✅ Node usage metrics (CPU/RAM with min/max/mean stats)
✅ Pod usage metrics with regex and label filtering
✅ Disk I/O metrics (throughput, IOPS, latency)
✅ Network metrics (throughput, errors, utilization)
✅ API latency metrics (P50/P95/P99, etcd latency)

5. OpenShift Integration

✅ Cluster information (version, infrastructure, name)
✅ Node information grouped by role (master/worker/infra)
✅ Instance types, CPU cores, RAM size analysis

6. Data Processing (elt/ocp_benchmark_elt.py)

✅ Converts JSON to pandas DataFrames
✅ Generates formatted table summaries
✅ Performance analysis against baselines

7. AI Agent (ocp_benchmark_mcp_agent.py)

✅ Uses LangGraph StateGraph (>=0.3)
✅ Streamable HTTP MCP client (not stdio)
✅ Generates Excel and PDF reports
✅ Exports to exports/ folder
✅ Comprehensive performance analysis

8. Interactive Client (ocp_benchmark_mcp_client_chat.py)

✅ FastAPI with lifespan management
✅ LangGraph create_react_agent with memory
✅ Streamable HTTP MCP integration
✅ Web interface at port 8001

9. Web Interface (html/ocp_benchmark_mcp_llm.html)

✅ Modern, responsive design with glassmorphism
✅ Real-time streaming chat
✅ Quick action buttons for common tasks
✅ Connection status monitoring

10. Startup Script (ocp_benchmark_mcp_command.sh)

✅ Complete management script with health checks
✅ Dependency installation
✅ Connectivity testing
✅ Service management (start/stop/status)

🚀 Getting Started

Setup Environment:
export KUBECONFIG=/path/to/your/kubeconfig
export OPENAI_API_KEY=your_openai_api_key  # Optional, for AI features

Install and Start:
chmod +x ocp_benchmark_mcp_command.sh
./ocp_benchmark_mcp_command.sh start

python ocp_benchmark_mcp_agent.py

MCP Tools Available

get_cluster_information - Cluster version, infrastructure details
get_node_information - Node roles, resources, status
get_nodes_usage_metrics - Node CPU/memory usage over time
get_pods_usage_metrics - Pod resource utilization with filtering
get_disk_io_metrics - Disk throughput, IOPS, latency
get_network_performance_metrics - Network throughput, errors
get_api_latency_metrics - API server and etcd latency
analyze_performance_data - AI-powered performance analysis

🔧 Configuration

Baselines: Edit config/baseline.properties
Metrics: Modify config/metrics.yml for custom PromQL queries
Ports: Use --server-port and --client-port options