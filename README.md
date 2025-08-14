# OpenShift Benchmark MCP Server

A comprehensive Model Context Protocol (MCP) server for OpenShift cluster performance monitoring, analysis, and benchmarking using AI-powered insights.

## Features

- **Comprehensive Monitoring**: Monitor cluster health, node resources, pod usage, disk I/O, network performance, and API latency
- **AI-Powered Analysis**: LLM-based performance analysis with actionable recommendations
- **Real-time Chat Interface**: Interactive web-based chat interface for querying cluster metrics
- **Baseline Comparison**: Compare current performance against configurable baseline thresholds
- **Multi-Format Reports**: Generate Excel and PDF performance reports
- **RESTful API**: FastAPI-based REST endpoints for programmatic access
- **LangGraph Integration**: Sophisticated AI agent workflow for automated analysis

## Architecture

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

## Quick Start

### Prerequisites

- Python 3.9+
- OpenShift CLI (`oc`) or kubectl
- Active OpenShift cluster connection (KUBECONFIG set)
- Access to cluster monitoring (Prometheus)

### Installation

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd ocp-benchmark-mcp
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   # OR use the startup script
   ./ocp_benchmark_mcp_command.sh start --install-deps
   ```

3. **Set Environment Variables**
   ```bash
   export KUBECONFIG=/path/to/your/kubeconfig
   export OPENAI_API_KEY=your_openai_api_key  # For AI features
   export TZ=UTC
   ```

4. **Start Services**
   ```bash
   chmod +x ocp_benchmark_mcp_command.sh
   ./ocp_benchmark_mcp_command.sh start
   ```

### Access Points

- **Chat Interface**: http://localhost:8081
- **REST API**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs
- **MCP Server**: http://localhost:8000

## Configuration

### Baseline Configuration

Edit `config/baseline.properties` to customize performance baselines:

```properties
# CPU baseline metrics (percentage)
cpu.baseline.min=10.0
cpu.baseline.max=80.0
cpu.baseline.mean=45.0

# Memory baseline metrics (percentage)
memory.baseline.min=20.0
memory.baseline.max=85.0
memory.baseline.mean=50.0

# Disk I/O baseline metrics
disk.io.read.baseline=100.0  # MB/s
disk.io.write.baseline=50.0  # MB/s
disk.read.iops=10000
disk.write.iops=8000
```

### Metrics Configuration

Customize PromQL queries in `config/metrics.yml`:

```yaml
metrics:
  node_metrics:
    cpu_usage:
      query: '100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)'
      description: "Node CPU usage percentage"
```

## Usage Examples

### Command Line Interface

```bash
# Start all services
./ocp_benchmark_mcp_command.sh start

# Check service status
./ocp_benchmark_mcp_command.sh status

# View logs
./ocp_benchmark_mcp_command.sh logs server
./ocp_benchmark_mcp_command.sh logs api
./ocp_benchmark_mcp_command.sh logs client

# Run AI analysis
./ocp_benchmark_mcp_command.sh analyze 2  # 2-hour analysis

# Test cluster connection
./ocp_benchmark_mcp_command.sh test

# Stop services
./ocp_benchmark_mcp_command.sh stop
```

### REST API Examples

```bash
# Get cluster information
curl http://localhost:8080/cluster/info

# Get node usage metrics
curl http://localhost:8080/metrics/usage/nodes?duration_hours=2

# Get top resource-consuming pods
curl http://localhost:8080/metrics/usage/pods?top_n=10

# Get comprehensive performance report
curl http://localhost:8080/reports/comprehensive?duration_hours=1
```

### Chat Interface Examples

Access the web interface at http://localhost:8081 and try these queries:

- "Show me cluster information"
- "What's the CPU usage for the last 2 hours?"
- "Which pods are consuming the most resources?"
- "Is there any disk I/O performance issue?"
- "Generate a comprehensive performance report"
- "Compare current metrics with baseline values"

## MCP Tools Available

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_cluster_info` | Cluster version and infrastructure | `include_operators` |
| `get_node_info` | Node resources and status | `summary_only`, `role_filter` |
| `get_node_usage_metrics` | CPU/memory usage statistics | `duration_hours` |
| `get_pod_usage_metrics` | Pod resource consumption | `duration_hours`, `pod_patterns`, `label_selectors` |
| `get_disk_io_metrics` | Disk I/O performance | `duration_hours`, `by_device` |
| `get_network_metrics` | Network throughput and latency | `duration_hours`, `by_interface` |
| `get_api_latency_metrics` | API server response times | `duration_hours`, `slow_threshold_ms` |
| `get_baseline_configuration` | Current baseline settings | None |
| `get_comprehensive_performance_report` | Full performance analysis | `duration_hours` |

## AI Analysis Features

### Performance Analysis

The AI agent provides:

- **Health Scoring**: 0-10 cluster health score
- **Trend Analysis**: Performance trend identification
- **Bottleneck Detection**: Resource constraint identification
- **Efficiency Metrics**: Resource utilization efficiency
- **Baseline Deviations**: Comparison with expected values

### Recommendations

AI-generated recommendations include:

- Resource optimization strategies
- Capacity planning suggestions
- Performance tuning advice
- Infrastructure improvements
- Monitoring enhancements

### Report Generation

Automated reports in multiple formats:

- **Excel**: Detailed metrics and analysis tables
- **PDF**: Executive summary with visualizations
- **JSON**: Raw data for programmatic access

## File Structure

```
ocp-benchmark-mcp/
├── README.md
├── requirements.txt
├── pyproject.toml
├── ocp_benchmark_mcp_server.py        # Main MCP server
├── ocp_benchmark_mcp_api.py           # FastAPI REST server
├── ocp_benchmark_mcp_client_chat.py   # Chat interface
├── ocp_benchmark_mcp_agent.py         # AI agent
├── ocp_benchmark_mcp_command.sh       # Startup script
├── ocauth/
│   └── ocp_benchmark_auth.py          # OpenShift authentication
├── tools/
│   ├── ocp_benchmark_openshift_clusterinfo.py
│   ├── ocp_benchmark_openshift_nodeinfo.py
│   ├── ocp_benchmark_prometheus_nodes_usage.py
│   ├── ocp_benchmark_prometheus_pods_usage.py
│   ├── ocp_benchmark_prometheus_diskio.py
│   ├── ocp_benchmark_prometheus_network.py
│   └── ocp_benchmark_prometheus_apilatency.py
├── config/
│   ├── ocp_benchmark_config.py
│   ├── baseline.properties
│   └── metrics.yml
├── analysis/
│   └── ocp_benchmark_performance_anlysis.py
├── exports/                           # Generated reports
├── html/
│   └── ocp_benchmark_mcp_llm.html     # Chat interface
├── logs/                              # Service logs
└── pids/                              # Process IDs
```

## Security Considerations

- **Authentication**: Uses OpenShift service account tokens
- **Authorization**: Requires appropriate RBAC permissions
- **Network**: Services bind to localhost by default
- **Data**: Metrics data is queried in real-time, not stored
- **Credentials**: API keys should be secured via environment variables

## Required Permissions

The service account needs these OpenShift permissions:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: benchmark-monitor
rules:
- apiGroups: [""]
  resources: ["nodes", "pods", "services"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["config.openshift.io"]
  resources: ["clusterversions", "infrastructures", "dnses", "clusteroperators"]
  verbs: ["get", "list"]
- apiGroups: ["route.openshift.io"]
  resources: ["routes"]
  verbs: ["get", "list"]
```

## Troubleshooting

### Common Issues

1. **Connection Issues**
   ```bash
   # Check KUBECONFIG
   echo $KUBECONFIG
   oc whoami
   
   # Test cluster access
   ./ocp_benchmark_mcp_command.sh test
   ```

2. **Prometheus Access**
   ```bash
   # Check if prometheus-k8s service account exists
   oc get sa prometheus-k8s -n openshift-monitoring
   
   # Create token manually if needed
   oc create token prometheus-k8s -n openshift-monitoring
   ```

3. **Port Conflicts**
   ```bash
   # Check if ports are in use
   netstat -tlnp | grep -E ":(8000|8080|8081)"
   
   # Modify ports in source code if needed
   ```

4. **Service Failures**
   ```bash
   # Check logs
   ./ocp_benchmark_mcp_command.sh logs server
   
   # Check service status
   ./ocp_benchmark_mcp_command.sh status
   ```

### Log Locations

- MCP Server: `logs/mcp_server.log`
- API Server: `logs/mcp_api.log`
- Chat Client: `logs/mcp_client.log`

## Performance Metrics

### Supported Metrics

- **Node Metrics**: CPU, Memory, Disk, Network utilization
- **Pod Metrics**: Resource consumption, limits, requests
- **Cluster Metrics**: API latency, etcd performance, operator status
- **Infrastructure**: Platform details, versions, capacity

### Time Ranges

- Default: 1 hour
- Configurable: 1-24 hours
- Real-time: Current values
- Historical: Trend analysis

## API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| GET/POST | `/cluster/info` | Cluster information |
| GET/POST | `/nodes/info` | Node details |
| GET/POST | `/metrics/usage/nodes` | Node resource usage |
| GET/POST | `/metrics/usage/pods` | Pod resource usage |
| GET/POST | `/metrics/disk` | Disk I/O metrics |
| GET/POST | `/metrics/network` | Network metrics |
| GET/POST | `/metrics/api` | API latency metrics |
| GET | `/config/baseline` | Baseline configuration |
| GET/POST | `/reports/comprehensive` | Full performance report |

### Example Queries (via chat or API):

"Show me cluster information"
"Get CPU usage for the last 2 hours"
"Which pods consume the most resources?"
"Check disk I/O performance against baselines"
"Generate comprehensive performance report"

### WebSocket Events

The chat interface supports real-time streaming responses via Server-Sent Events.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Update documentation
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review logs for error details
3. Verify OpenShift connectivity and permissions
4. Create an issue with detailed information

## Changelog

### Version 1.0.0

- Initial release
- MCP server implementation
- AI-powered analysis
- REST API interface
- Chat interface
- Comprehensive monitoring
- Report generation