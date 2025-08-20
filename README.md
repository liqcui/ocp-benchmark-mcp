# OpenShift Benchmark MCP Server (AI‑Powered)

This AI‑assisted MCP platform fuses Prometheus metrics and OpenShift APIs with a Large Language Model agent to transform raw telemetry into actionable, executive‑ready performance insights.

## Features

- **AI Analysis Agent**: LLM‑driven reasoning and autonomous analysis flows (LangGraph)
- **AI‑Generated Reports**: Executive summaries, benchmarks, and prioritized recommendations (Excel/PDF)
- **Natural Language Chat**: Run analyses by asking questions—no query language required
- **Cluster Information**: Auto‑discover version and infrastructure
- **Node/Pod Monitoring**: CPU/memory utilization with baselines and alerts
- **Prometheus Integration**: Configurable PromQL queries
- **Web Interface**: Readable HTML tables and streaming responses

## AI‑Powered Highlights

- Autonomous data collection → analysis → report generation
- Capacity forecasting with risk scoring
- Baseline comparison and anomaly surfacing
- Conversational “tools” that map prompts to actions

## Architecture Topology
```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   MCP Server        │    │   MCP API Server    │    │   Chat Interface    │
│   (Port 8000)       │◄───│   (Port 8080)       │◄───│   (Port 8081)       │
│                     │    │                     │    │                     │
│   • MCP Tools       │    │   • REST Endpoints  │    │   • LLM Chat        │
│   • Prometheus      │    │   • FastAPI         │    │   • Web Interface   │
│   • OpenShift API   │    │   • Health Checks   │    │   • Streaming       │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
           │                                                       │
           ▼                                                       ▼
┌─────────────────────┐                            ┌─────────────────────┐
│   AI Agent          │                            │   Performance       │
│   (LangGraph)       │                            │   Analyzer          │
│                     │                            │                     │
│   • Analysis Flow   │                            │   • Baseline Comp   │
│   • Recommendations │                            │   • Trend Analysis  │
│   • Report Export   │                            │   • Efficiency      │
└─────────────────────┘                            └─────────────────────┘
```
```
┌─────────────────────────────────────────────────────────────────┐
│                    OCP Benchmark MCP Architecture                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐ │
│  │   MCP Client    │    │   AI Agent       │    │  FastAPI    │ │
│  │   (LangChain)   │◄──►│   (LangGraph)    │◄──►│  REST API   │ │
│  │                 │    │                  │    │             │ │
│  └─────────────────┘    └──────────────────┘    └─────────────┘ │
│           │                       │                      │      │
│           │              ┌────────▼────────┐             │      │
│           │              │  Performance    │             │      │
│           │              │  Analyzer       │             │      │
│           │              └─────────────────┘             │      │
│           │                                              │      │
│           ▼                                              ▼      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 MCP Server (FastMCP)                        │ │
│  │                Streamable HTTP Transport                    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│           │                       │                      │      │
│           ▼                       ▼                      ▼      │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐ │
│  │   OpenShift     │    │   Prometheus     │    │Elasticsearch│ │
│  │   API Server    │    │   Metrics        │    │   Logs      │ │
│  │                 │    │                  │    │             │ │
│  └─────────────────┘    └──────────────────┘    └─────────────┘ │
│           │                       │                      │      │
│           ▼                       ▼                      ▼      │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐ │
│  │   Cluster       │    │   Node Exporter  │    │  Website    │ │
│  │   Resources     │    │   Metrics        │    │  Scraper    │ │
│  └─────────────────┘    └──────────────────┘    └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -e .
```

## Configuration

1. Set up your OpenShift environment:
   ```bash
   export KUBECONFIG=/path/to/your/kubeconfig
   ```

2. Configure baselines in `config/baseline.properties`
3. Customize metrics in `config/metrics.yml`

## Usage

### Start the MCP Server
```bash
python ocp_benchmark_mcp_server.py
```

### Start the Interactive Client
```bash
python ocp_benchmark_mcp_client_chat.py
```
### Interactive Chat UI
<img width="1712" height="940" alt="截屏2025-08-19 23 16 39" src="https://github.com/user-attachments/assets/cb7835f2-d80d-4abb-9841-d9ec306ef40b" />

### Run Performance Analysis Agent
```bash
python ocp_benchmark_mcp_agent.py
```

The agent requires:
- `OPENAI_API_KEY` (and optionally `BASE_URL` for compatible providers)
- `KUBECONFIG` pointing to your cluster

It will:
- Connect to the MCP Server on `http://localhost:8000`
- Collect cluster, nodes, pods, disk, network, and API metrics
- Analyze against baselines and generate Excel/PDF reports in `exports/`

Example:
```bash
export OPENAI_API_KEY=sk-...
export KUBECONFIG=$HOME/.kube/config
python ocp_benchmark_mcp_agent.py
```

### Use the Startup Script
```bash
chmod +x ocp_benchmark_mcp_command.sh
./ocp_benchmark_mcp_command.sh
```

## Project Structure

```
ocp-benchmark-mcp-final-edition/
├── README.md
├── pyproject.toml
├── ocp_benchmark_mcp_server.py          # MCP tools server (port 8000)
├── ocp_benchmark_mcp_api.py             # REST API wrapper (port 8081)
├── ocp_benchmark_mcp_client_chat.py     # Web chat UI (port 8080)
├── ocp_benchmark_mcp_agent.py           # AI analysis agent
├── ocp_benchmark_command.sh             # Startup script
├── API_USAGE.md
├── EXTENSION_GUIDE.md
├── logs/
├── exports/
├── html/
│   └── ocp_benchmark_mcp_llm.html       # Web interface template
├── ocauth/
│   └── ocp_benchmark_auth.py            # OpenShift/Prometheus auth
├── config/
│   ├── ocp_benchmark_config.py          # Configuration management
│   ├── baseline.properties              # Performance baselines
│   └── metrics.yml                      # PromQL queries
├── tools/
│   ├── ocp_benchmark_es.py
│   ├── ocp_benchmark_openshift_clusterinfo.py
│   ├── ocp_benchmark_openshift_nodeinfo.py
│   ├── ocp_benchmark_prometheus_basequery.py
│   ├── ocp_benchmark_prometheus_nodes_usage.py
│   ├── ocp_benchmark_prometheus_pods_usage.py
│   ├── ocp_benchmark_prometheus_diskio.py
│   ├── ocp_benchmark_prometheus_network.py
│   ├── ocp_benchmark_prometheus_apilatency.py
│   └── ocp_benchmark_website.py
├── analysis/
│   └── ocp_benchmark_performance_analysis.py
└── elt/
    ├── ocp_benchmark_elt.py
    ├── ocp_benchmark_extract_json.py
    ├── ocp_benchmark_json2table.py
    ├── ocp_benchmark_elt_extract_node_info.py
    ├── ocp_benchmark_elt_extract_nodes_usage.py
    ├── ocp_benchmark_elt_extract_pods_usage.py
    ├── ocp_benchmark_elt_extract_disk_io.py
    ├── ocp_benchmark_elt_extract_network.py
    ├── ocp_benchmark_elt_extract_api_request_latency.py
    ├── ocp_benchmark_elt_extract_api_request_rate.py
    └── ocp_benchmark_elt_extract_etcd_latency.py
```

## API Endpoints

The MCP server provides the following tools:

- `get_cluster_info`: Retrieve cluster version and infrastructure details
- `get_node_info`: Get detailed node information grouped by role
- `get_nodes_usage`: Query CPU/RAM usage for nodes over time
- `get_pods_usage`: Monitor pod-level resource utilization
- `get_disk_metrics`: Analyze disk I/O performance
- `get_network_metrics`: Monitor network throughput and latency
- `get_api_latency`: Track API server response times
- `analyze_performance`: Compare metrics against baselines

## Environment Variables

- `KUBECONFIG`: Path to kubeconfig file (required)
- `PROMETHEUS_URL`: Override auto-discovered Prometheus URL (optional)
- `OPENAI_API_KEY`: Required for AI analysis features

## License

MIT License
 
---

## Quick Start

Follow these steps without removing any of the above instructions:

1) Environment
- Export `KUBECONFIG` to your cluster file
- (Optional) set `OPENAI_API_KEY` and `BASE_URL` for the chat UI

2) Run the three services (each in its own terminal):
```bash
python ocp_benchmark_mcp_server.py      # MCP Server (tools) on :8000
python ocp_benchmark_mcp_api.py         # REST API wrapper on :8081
python ocp_benchmark_mcp_client_chat.py # Web chat UI on :8080
```
Open http://localhost:8080

3) Quick API checks (via API Server :8081):
```bash
curl http://localhost:8081/health
curl http://localhost:8081/cluster-info
curl -X POST http://localhost:8081/nodes-usage \
  -H 'Content-Type: application/json' \
  -d '{"duration_hours": 1, "step": "1m"}'
curl -X POST http://localhost:8081/pods-usage \
  -H 'Content-Type: application/json' \
  -d '{"duration_hours": 0.5, "step": "1m"}'
curl -X POST http://localhost:8081/disk-metrics \
  -H 'Content-Type: application/json' \
  -d '{"duration_hours": 1, "step": "1m"}'
curl -X POST http://localhost:8081/network-metrics \
  -H 'Content-Type: application/json' \
  -d '{"duration_hours": 1, "step": "1m"}'
curl -X POST http://localhost:8081/api-request-latency \
  -H 'Content-Type: application/json' \
  -d '{"duration_hours": 1, "step": "1m"}'
curl -X POST http://localhost:8081/api-request-rate \
  -H 'Content-Type: application/json' \
  -d '{"duration_hours": 1, "step": "1m"}'
curl -X POST http://localhost:8081/etcd-latency \
  -H 'Content-Type: application/json' \
  -d '{"duration_hours": 1, "step": "1m"}'
```

## Output Conventions

- Floating-point numbers are rounded to 6 decimals (timestamps excluded)
- Extractors focus on relevant payloads (e.g., only `rate_stats`, only `latency_stats`)
- The web UI renders JSON as readable HTML tables with widened layout

## Ports Summary

- MCP Server (tools): 8000
- API Server (REST wrapper): 8081
- Chat Client (web UI + LLM): 8080
