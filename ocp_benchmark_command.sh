#!/bin/bash

# OpenShift Benchmark MCP Server Startup Script
# This script starts all components of the OpenShift Benchmark MCP system

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
PID_DIR="${SCRIPT_DIR}/pids"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    mkdir -p "${LOG_DIR}"
    mkdir -p "${PID_DIR}"
    mkdir -p "${SCRIPT_DIR}/exports"
    mkdir -p "${SCRIPT_DIR}/html"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [[ "$(echo "$PYTHON_VERSION < 3.9" | bc -l 2>/dev/null || echo 1)" == "1" ]]; then
        error "Python 3.9+ is required, found: $PYTHON_VERSION"
        exit 1
    fi
    
    # Check if KUBECONFIG is set or kubectl is configured
    if [[ -z "${KUBECONFIG}" ]] && ! kubectl config current-context &> /dev/null; then
        error "KUBECONFIG not set and kubectl not configured"
        exit 1
    fi
    
    # Check if oc command is available
    if ! command -v oc &> /dev/null; then
        warn "OpenShift CLI (oc) not found. Some features may not work properly."
    fi
    
    # Check if required packages are installed
    if ! python3 -c "import fastmcp, fastapi, langchain, kubernetes" &> /dev/null; then
        error "Required Python packages not installed. Please run: pip install -r requirements.txt"
        exit 1
    fi
    
    log "Prerequisites check completed"
}

# Set timezone to UTC
set_timezone() {
    log "Setting timezone to UTC..."
    export TZ=UTC
}

# Test OpenShift connection
test_openshift_connection() {
    log "Testing OpenShift connection..."
    
    if command -v oc &> /dev/null; then
        if ! oc whoami &> /dev/null; then
            error "Not logged into OpenShift cluster"
            exit 1
        fi
        
        CLUSTER_VERSION=$(oc version --client=false -o json 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('openshiftVersion', 'unknown'))" 2>/dev/null || echo "unknown")
        log "Connected to OpenShift cluster version: ${CLUSTER_VERSION}"
    else
        # Fallback to kubectl
        if ! kubectl cluster-info &> /dev/null; then
            error "Cannot connect to Kubernetes/OpenShift cluster"
            exit 1
        fi
        log "Connected to Kubernetes cluster"
    fi
}

# Install Python dependencies
install_dependencies() {
    if [[ "$1" == "--install-deps" ]]; then
        log "Installing Python dependencies..."
        python3 -m pip install --upgrade pip
        python3 -m pip install -r requirements.txt
        log "Dependencies installed"
    fi
}

# Start MCP Server
start_mcp_server() {
    log "Starting OpenShift Benchmark MCP Server..."
    
    cd "${SCRIPT_DIR}"
    nohup python3 ocp_benchmark_mcp_server.py > "${LOG_DIR}/mcp_server.log" 2>&1 &
    MCP_PID=$!
    echo $MCP_PID > "${PID_DIR}/mcp_server.pid"
    
    # Wait a bit and check if process is running
    sleep 3
    if ! kill -0 $MCP_PID 2>/dev/null; then
        error "MCP Server failed to start. Check log: ${LOG_DIR}/mcp_server.log"
        exit 1
    fi
    
    log "MCP Server started with PID: $MCP_PID"
}

# Start MCP API Server
start_mcp_api() {
    log "Starting MCP API Server..."
    
    cd "${SCRIPT_DIR}"
    nohup python3 ocp_benchmark_mcp_api.py > "${LOG_DIR}/mcp_api.log" 2>&1 &
    API_PID=$!
    echo $API_PID > "${PID_DIR}/mcp_api.pid"
    
    # Wait a bit and check if process is running
    sleep 3
    if ! kill -0 $API_PID 2>/dev/null; then
        error "MCP API Server failed to start. Check log: ${LOG_DIR}/mcp_api.log"
        exit 1
    fi
    
    log "MCP API Server started with PID: $API_PID"
}

# Start MCP Client Chat Interface
start_mcp_client() {
    log "Starting MCP Client Chat Interface..."
    
    cd "${SCRIPT_DIR}"
    nohup python3 ocp_benchmark_mcp_client_chat.py > "${LOG_DIR}/mcp_client.log" 2>&1 &
    CLIENT_PID=$!
    echo $CLIENT_PID > "${PID_DIR}/mcp_client.pid"
    
    # Wait a bit and check if process is running
    sleep 3
    if ! kill -0 $CLIENT_PID 2>/dev/null; then
        error "MCP Client failed to start. Check log: ${LOG_DIR}/mcp_client.log"
        exit 1
    fi
    
    log "MCP Client Chat Interface started with PID: $CLIENT_PID"
}

# Wait for services to be ready
wait_for_services() {
    log "Waiting for services to be ready..."
    
    # Wait for MCP Server
    for i in {1..30}; do
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            log "MCP Server is ready"
            break
        fi
        if [[ $i -eq 30 ]]; then
            error "MCP Server failed to become ready"
            exit 1
        fi
        sleep 2
    done
    
    # Wait for MCP API Server
    for i in {1..30}; do
        if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
            log "MCP API Server is ready"
            break
        fi
        if [[ $i -eq 30 ]]; then
            error "MCP API Server failed to become ready"
            exit 1
        fi
        sleep 2
    done
    
    # Wait for MCP Client
    for i in {1..30}; do
        if curl -sf http://localhost:8081/health > /dev/null 2>&1; then
            log "MCP Client Chat Interface is ready"
            break
        fi
        if [[ $i -eq 30 ]]; then
            error "MCP Client Chat Interface failed to become ready"
            exit 1
        fi
        sleep 2
    done
}

# Stop all services
stop_services() {
    log "Stopping all services..."
    
    # Stop MCP Client
    if [[ -f "${PID_DIR}/mcp_client.pid" ]]; then
        CLIENT_PID=$(cat "${PID_DIR}/mcp_client.pid")
        if kill -0 $CLIENT_PID 2>/dev/null; then
            kill $CLIENT_PID
            log "MCP Client stopped"
        fi
        rm -f "${PID_DIR}/mcp_client.pid"
    fi
    
    # Stop MCP API Server
    if [[ -f "${PID_DIR}/mcp_api.pid" ]]; then
        API_PID=$(cat "${PID_DIR}/mcp_api.pid")
        if kill -0 $API_PID 2>/dev/null; then
            kill $API_PID
            log "MCP API Server stopped"
        fi
        rm -f "${PID_DIR}/mcp_api.pid"
    fi
    
    # Stop MCP Server
    if [[ -f "${PID_DIR}/mcp_server.pid" ]]; then
        MCP_PID=$(cat "${PID_DIR}/mcp_server.pid")
        if kill -0 $MCP_PID 2>/dev/null; then
            kill $MCP_PID
            log "MCP Server stopped"
        fi
        rm -f "${PID_DIR}/mcp_server.pid"
    fi
}

# Show service status
show_status() {
    log "Service Status:"
    echo
    
    # Check MCP Server
    if [[ -f "${PID_DIR}/mcp_server.pid" ]]; then
        MCP_PID=$(cat "${PID_DIR}/mcp_server.pid")
        if kill -0 $MCP_PID 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} MCP Server (PID: $MCP_PID) - http://localhost:8000"
        else
            echo -e "  ${RED}✗${NC} MCP Server (not running)"
        fi
    else
        echo -e "  ${RED}✗${NC} MCP Server (not started)"
    fi
    
    # Check MCP API Server
    if [[ -f "${PID_DIR}/mcp_api.pid" ]]; then
        API_PID=$(cat "${PID_DIR}/mcp_api.pid")
        if kill -0 $API_PID 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} MCP API Server (PID: $API_PID) - http://localhost:8080"
        else
            echo -e "  ${RED}✗${NC} MCP API Server (not running)"
        fi
    else
        echo -e "  ${RED}✗${NC} MCP API Server (not started)"
    fi
    
    # Check MCP Client
    if [[ -f "${PID_DIR}/mcp_client.pid" ]]; then
        CLIENT_PID=$(cat "${PID_DIR}/mcp_client.pid")
        if kill -0 $CLIENT_PID 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} MCP Client Chat (PID: $CLIENT_PID) - http://localhost:8081"
        else
            echo -e "  ${RED}✗${NC} MCP Client Chat (not running)"
        fi
    else
        echo -e "  ${RED}✗${NC} MCP Client Chat (not started)"
    fi
    
    echo
}

# Show logs
show_logs() {
    local service="$1"
    
    case "$service" in
        "server"|"mcp")
            if [[ -f "${LOG_DIR}/mcp_server.log" ]]; then
                tail -f "${LOG_DIR}/mcp_server.log"
            else
                error "MCP Server log not found"
            fi
            ;;
        "api")
            if [[ -f "${LOG_DIR}/mcp_api.log" ]]; then
                tail -f "${LOG_DIR}/mcp_api.log"
            else
                error "MCP API log not found"
            fi
            ;;
        "client"|"chat")
            if [[ -f "${LOG_DIR}/mcp_client.log" ]]; then
                tail -f "${LOG_DIR}/mcp_client.log"
            else
                error "MCP Client log not found"
            fi
            ;;
        *)
            error "Unknown service: $service"
            echo "Available services: server, api, client"
            ;;
    esac
}

# Run AI agent analysis
run_analysis() {
    local duration="${1:-1}"
    
    log "Running AI agent performance analysis for ${duration} hours..."
    cd "${SCRIPT_DIR}"
    python3 ocp_benchmark_mcp_agent.py --duration-hours "$duration"
}

# Test cluster connection and gather basic info
test_cluster() {
    log "Testing cluster connection and gathering basic information..."
    
    cd "${SCRIPT_DIR}"
    python3 -c "
import asyncio
from tools.ocp_benchmark_openshift_clusterinfo import get_cluster_info_json
from tools.ocp_benchmark_openshift_nodeinfo import get_nodes_summary_json

async def test():
    try:
        print('\\n=== Cluster Information ===')
        cluster_info = await get_cluster_info_json()
        print(cluster_info)
        
        print('\\n=== Node Summary ===')
        node_info = await get_nodes_summary_json()
        print(node_info)
        
        print('\\n✓ Cluster connection test completed successfully')
    except Exception as e:
        print(f'✗ Cluster connection test failed: {e}')
        
asyncio.run(test())
"
}

# Show usage information
show_usage() {
    echo "OpenShift Benchmark MCP Server Management Script"
    echo
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Commands:"
    echo "  start [--install-deps]    Start all MCP services"
    echo "  stop                      Stop all MCP services"
    echo "  restart [--install-deps]  Restart all MCP services"
    echo "  status                    Show service status"
    echo "  logs <service>            Show logs (service: server|api|client)"
    echo "  test                      Test cluster connection"
    echo "  analyze [duration]        Run AI agent analysis (default: 1 hour)"
    echo "  help                      Show this help message"
    echo
    echo "Options:"
    echo "  --install-deps            Install Python dependencies before starting"
    echo
    echo "Examples:"
    echo "  $0 start --install-deps   # Install deps and start services"
    echo "  $0 logs server           # Show MCP server logs"
    echo "  $0 analyze 2             # Run 2-hour analysis"
    echo
    echo "Service URLs:"
    echo "  MCP Server:        http://localhost:8000"
    echo "  MCP API Server:    http://localhost:8080"
    echo "  Chat Interface:    http://localhost:8081"
}

# Signal handlers for cleanup
cleanup() {
    log "Received interrupt signal, stopping services..."
    stop_services
    exit 0
}

trap cleanup INT TERM

# Main script logic
main() {
    local command="${1:-help}"
    
    case "$command" in
        "start")
            create_directories
            check_prerequisites
            set_timezone
            test_openshift_connection
            install_dependencies "$2"
            
            start_mcp_server
            sleep 5
            start_mcp_api
            sleep 5
            start_mcp_client
            
            wait_for_services
            
            log "All services started successfully!"
            show_status
            
            info "Chat Interface: http://localhost:8081"
            info "API Documentation: http://localhost:8080/docs"
            info "Use '$0 logs <service>' to view logs"
            info "Use '$0 stop' to stop all services"
            ;;
            
        "stop")
            stop_services
            log "All services stopped"
            ;;
            
        "restart")
            stop_services
            sleep 2
            create_directories
            check_prerequisites
            set_timezone
            test_openshift_connection
            install_dependencies "$2"
            
            start_mcp_server
            sleep 5
            start_mcp_api
            sleep 5
            start_mcp_client
            
            wait_for_services
            
            log "All services restarted successfully!"
            show_status
            ;;
            
        "status")
            show_status
            ;;
            
        "logs")
            show_logs "$2"
            ;;
            
        "test")
            set_timezone
            test_openshift_connection
            test_cluster
            ;;
            
        "analyze")
            set_timezone
            run_analysis "$2"
            ;;
            
        "help"|"--help"|"-h")
            show_usage
            ;;
            
        *)
            error "Unknown command: $command"
            echo
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"