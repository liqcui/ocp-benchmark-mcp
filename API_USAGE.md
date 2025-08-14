# OCP Benchmark MCP API Usage Guide

This FastAPI application provides REST API endpoints to expose OpenShift Benchmark MCP tools.

## Overview

The API serves as a wrapper around the OpenShift Benchmark MCP (Model Context Protocol) server, exposing its tools through HTTP REST endpoints.

## Architecture

```
Client → FastAPI API → MCP Client → MCP Server → OpenShift Tools
```

## Starting the Services

### 1. Start the MCP Server
```bash
cd /path/to/ocp-benchmark-mcp-final-edition
python ocp_benchmark_mcp_server.py
```
The MCP server will start on `http://localhost:8000`

### 2. Start the API Server
```bash
python ocp_benchmark_mcp_api.py
```
The API server will start on `http://localhost:8081`

## API Endpoints

### Basic Endpoints

- **GET /** - Root endpoint with API information
- **GET /health** - Health check endpoint
- **GET /test-connection** - Test Prometheus connection

### Cluster Information

- **GET /cluster-info** - Get OpenShift cluster information
- **GET /node-info** - Get detailed node information

### Metrics Endpoints (POST with JSON body)

#### Nodes Usage Metrics
```bash
POST /nodes-usage
Content-Type: application/json

{
  "duration_hours": 1.0,
  "step": "1m"
}
```

#### Pods Usage Metrics
```bash
POST /pods-usage
Content-Type: application/json

{
  "duration_hours": 1.0,
  "step": "1m",
  "pod_regex": "my-app-.*",
  "label_selectors": ["app=my-app", "tier=frontend"]
}
```

#### Disk I/O Metrics
```bash
POST /disk-metrics
Content-Type: application/json

{
  "duration_hours": 1.0,
  "step": "1m"
}
```

#### Network Performance Metrics
```bash
POST /network-metrics
Content-Type: application/json

{
  "duration_hours": 1.0,
  "step": "1m"
}
```

#### API Latency Metrics
```bash
POST /api-latency
Content-Type: application/json

{
  "duration_hours": 1.0,
  "step": "1m"
}
```

#### Performance Analysis
```bash
POST /analyze-performance
Content-Type: application/json

{
  "metrics_data": {
    "cpu_usage": [80, 85, 90],
    "memory_usage": [70, 75, 80]
  },
  "analysis_type": "comprehensive"
}
```

## Response Format

All endpoints return a consistent JSON response format:

```json
{
  "success": true,
  "data": {
    // Response data here
  },
  "error": null,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

For errors:
```json
{
  "success": false,
  "data": null,
  "error": "Error description",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Testing the API

Use the provided test script:
```bash
python test_api.py
```

This will test all endpoints and provide a summary of results.

## cURL Examples

### Get cluster information
```bash
curl -X GET http://localhost:8081/cluster-info
```

### Get nodes usage metrics
```bash
curl -X POST http://localhost:8081/nodes-usage \
  -H "Content-Type: application/json" \
  -d '{"duration_hours": 1.0, "step": "1m"}'
```

### Get pods usage with filtering
```bash
curl -X POST http://localhost:8081/pods-usage \
  -H "Content-Type: application/json" \
  -d '{
    "duration_hours": 2.0,
    "step": "5m",
    "pod_regex": "nginx-.*",
    "label_selectors": ["app=nginx"]
  }'
```

## API Documentation

Once the API is running, visit:
- **Swagger UI**: http://localhost:8081/docs
- **ReDoc**: http://localhost:8081/redoc

## Prerequisites

1. OpenShift cluster access with proper KUBECONFIG
2. Prometheus endpoint accessible
3. Required Python dependencies installed
4. MCP server running on port 8000

## Troubleshooting

### Common Issues

1. **MCP client not initialized** - Ensure the MCP server is running on port 8000
2. **Prometheus connection failed** - Check your cluster access and Prometheus configuration
3. **Invalid response format** - The MCP server may be returning non-JSON data

### Debugging

Check the API logs for detailed error information. The API uses structured logging with timestamps and log levels.

## Environment Variables

- `KUBECONFIG` - Path to your OpenShift/Kubernetes configuration file
- `PROMETHEUS_URL` - Custom Prometheus endpoint (if not using cluster default)

## Security Notes

- The API runs on `0.0.0.0:8081` by default
- CORS is enabled for all origins (configure appropriately for production)
- No authentication is implemented (add as needed for production use)
