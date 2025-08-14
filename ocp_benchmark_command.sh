#!/bin/bash

# OpenShift Benchmark MCP Server Startup Script
# This script starts the OpenShift Benchmark MCP server and related services

set -e

# Configuration
PROJECT_NAME="ocp-benchmark-mcp"
MCP_SERVER_PORT="${MCP_SERVER_PORT:-8000}"
MCP_CLIENT_PORT="${MCP_CLIENT_PORT:-8001}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
TIMEZONE="${TZ:-UTC}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S UTC')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S UTC')] ERROR: $1${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S UTC')] WARNING: $1${NC}"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S UTC')] SUCCESS: $1${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Python 3.8+
    if ! python3 --version &> /dev/null; then
        error "Python 3 is required but not installed"
        exit 1
    fi
    
    local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
        error "Python 3.8+ is required, found $python_version"
        exit 1
    fi
    
    # Check KUBECONFIG
    if [[ -z "$KUBECONFIG" ]]; then
        error "KUBECONFIG environment variable is not set"
        exit 1
    fi
    
    if [[ ! -f "$KUBECONFIG" ]]; then
        error "KUBECONFIG file not found: $KUBECONFIG"
        exit 1
    fi
    
    # Check oc command
    if ! command -v oc &> /dev/null; then
        warn "oc command not found, some features may not work"
    fi
    
    # Check kubectl command as fallback
    if ! command -v kubectl &> /dev/null && ! command -v oc &> /dev/null; then
        error "Neither oc nor kubectl command found"
        exit 1
    fi
    
    # Set timezone
    export TZ="$TIMEZONE"
    
    success "Prerequisites check completed"
}

# Function to install dependencies
install_dependencies() {
    log "Installing Python dependencies..."
    
    if [[ -f "pyproject.toml" ]]; then
        pip3 install -e . --quiet
    else
        error "pyproject.toml not found. Please run this script from the project root directory."
        exit 1
    fi
    
    success "Dependencies installed successfully"
}

# Function to create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    local dirs=("exports" "logs")
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log "Created directory: $dir"
        fi
    done
    
    success "Directories created"
}

# Function to test OpenShift connectivity
test_connectivity() {
    log "Testing OpenShift connectivity..."
    
    # Test cluster access
    if command -v oc &> /dev/null; then
        if ! oc whoami &> /dev/null; then
            error "Cannot connect to OpenShift cluster. Please check your KUBECONFIG and login status."
            exit 1
        fi
        local current_project=$(oc project -q 2>/dev/null || echo "default")
        log "Connected to OpenShift cluster as $(oc whoami) in project $current_project"
    elif command -v kubectl &> /dev/null; then
        if ! kubectl cluster-info &> /dev/null; then
            error "Cannot connect to Kubernetes cluster. Please check your KUBECONFIG."
            exit 1
        fi
        log "Connected to Kubernetes cluster"
    fi
    
    # Test Prometheus access (if possible)
    log "Testing Prometheus connectivity (this may take a moment)..."
    python3 -c "
import sys
sys.path.append('.')
try:
    from ocauth.ocp_benchmark_auth import ocp_auth
    prometheus_url = ocp_auth.get_prometheus_url()
    if prometheus_url:
        print(f'Prometheus URL discovered: {prometheus_url}')
        if ocp_auth.test_prometheus_connection():
            print('Prometheus connection test: PASSED')
        else:
            print('Prometheus connection test: FAILED')
    else:
        print('Could not discover Prometheus URL')
except Exception as e:
    print(f'Prometheus test failed: {e}')
"
    
    success "Connectivity tests completed"
}

# Function to start MCP server
start_mcp_server() {
    log "Starting MCP Server on port $MCP_SERVER_PORT..."
    
    # Kill any existing process on the port
    if lsof -Pi :$MCP_SERVER_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        warn "Port $MCP_SERVER_PORT is already in use, attempting to kill existing process..."
        local pid=$(lsof -Pi :$MCP_SERVER_PORT -sTCP:LISTEN -t)
        kill -TERM $pid 2>/dev/null || true
        sleep 2
    fi
    
    # Start the server in background
    nohup python3 ocp_benchmark_mcp_server.py > logs/mcp_server.log 2>&1 &
    local server_pid=$!
    echo $server_pid > logs/mcp_server.pid
    
    # Wait a moment for server to start
    sleep 3
    
    # Check if server is running
    if kill -0 $server_pid 2>/dev/null; then
        success "MCP Server started successfully (PID: $server_pid)"
        log "Server logs: tail -f logs/mcp_server.log"
        log "Server URL: http://localhost:$MCP_SERVER_PORT"
    else
        error "Failed to start MCP Server"
        cat logs/mcp_server.log
        exit 1
    fi
}

# Function to start MCP client
start_mcp_client() {
    if [[ "$1" == "--client-only" ]]; then
        return 0
    fi
    
    log "Starting MCP Client on port $MCP_CLIENT_PORT..."
    
    # Kill any existing process on the port
    if lsof -Pi :$MCP_CLIENT_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        warn "Port $MCP_CLIENT_PORT is already in use, attempting to kill existing process..."
        local pid=$(lsof -Pi :$MCP_CLIENT_PORT -sTCP:LISTEN -t)
        kill -TERM $pid 2>/dev/null || true
        sleep 2
    fi
    
    # Start the client in background
    nohup python3 ocp_benchmark_mcp_client_chat.py > logs/mcp_client.log 2>&1 &
    local client_pid=$!
    echo $client_pid > logs/mcp_client.pid
    
    # Wait a moment for client to start
    sleep 3
    
    # Check if client is running
    if kill -0 $client_pid 2>/dev/null; then
        success "MCP Client started successfully (PID: $client_pid)"
        log "Client logs: tail -f logs/mcp_client.log"
        log "Client URL: http://localhost:$MCP_CLIENT_PORT"
    else
        error "Failed to start MCP Client"
        cat logs/mcp_client.log
        exit 1
    fi
}

# Function to stop services
stop_services() {
    log "Stopping services..."
    
    # Stop MCP Server
    if [[ -f logs/mcp_server.pid ]]; then
        local server_pid=$(cat logs/mcp_server.pid)
        if kill -0 $server_pid 2>/dev/null; then
            kill -TERM $server_pid
            log "Stopped MCP Server (PID: $server_pid)"
        fi
        rm -f logs/mcp_server.pid
    fi
    
    # Stop MCP Client
    if [[ -f logs/mcp_client.pid ]]; then
        local client_pid=$(cat logs/mcp_client.pid)
        if kill -0 $client_pid 2>/dev/null; then
            kill -TERM $client_pid
            log "Stopped MCP Client (PID: $client_pid)"
        fi
        rm -f logs/mcp_client.pid
    fi
    
    success "Services stopped"
}

# Function to show status
show_status() {
    log "Service Status:"
    
    # Check MCP Server
    if [[ -f logs/mcp_server.pid ]]; then
        local server_pid=$(cat logs/mcp_server.pid)
        if kill -0 $server_pid 2>/dev/null; then
            success "MCP Server: Running (PID: $server_pid, Port: $MCP_SERVER_PORT)"
        else
            error "MCP Server: Not running (stale PID file)"
        fi
    else
        warn "MCP Server: Not running"
    fi
    
    # Check MCP Client
    if [[ -f logs/mcp_client.pid ]]; then
        local client_pid=$(cat logs/mcp_client.pid)
        if kill -0 $client_pid 2>/dev/null; then
            success "MCP Client: Running (PID: $client_pid, Port: $MCP_CLIENT_PORT)"
        else
            error "MCP Client: Not running (stale PID file)"
        fi
    else
        warn "MCP Client: Not running"
    fi
    
    # Show resource usage if processes are running
    if [[ -f logs/mcp_server.pid ]] || [[ -f logs/mcp_client.pid ]]; then
        log "Resource Usage:"
        ps aux | head -1
        ps aux | grep -E "(ocp_benchmark_mcp|python3.*ocp)" | grep -v grep || true
    fi
}

# Function to run performance test
run_performance_test() {
    log "Running performance test..."
    
    python3 -c "
import sys
import asyncio
import json
sys.path.append('.')

async def test_tools():
    try:
        # Test cluster info
        from tools.ocp_benchmark_openshift_clusterinfo import get_cluster_info
        cluster_info = json.loads(get_cluster_info())
        print(f'✓ Cluster: {cluster_info.get(\"summary\", {}).get(\"cluster_name\", \"unknown\")}')
        
        # Test node info
        from tools.ocp_benchmark_openshift_nodeinfo import get_nodes_info
        node_info = json.loads(get_nodes_info())
        total_nodes = node_info.get('cluster_summary', {}).get('total_nodes', 0)
        print(f'✓ Nodes: {total_nodes} total nodes')
        
        # Test nodes usage (short duration)
        from tools.ocp_benchmark_prometheus_nodes_usage import get_nodes_usage
        nodes_usage = json.loads(get_nodes_usage(0.1))  # 6 minutes
        print(f'✓ Nodes Usage: Data collected for {len(nodes_usage.get(\"nodes\", {}))} nodes')
        
        print('✓ All tools working correctly')
        return True
        
    except Exception as e:
        print(f'✗ Test failed: {e}')
        return False

# Run the test
success = asyncio.run(test_tools())
sys.exit(0 if success else 1)
"
    
    if [[ $? -eq 0 ]]; then
        success "Performance test completed successfully"
    else
        error "Performance test failed"
        exit 1
    fi
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTIONS] [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start       Start both MCP server and client (default)"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  status      Show service status"
    echo "  test        Run connectivity and performance tests"
    echo "  server-only Start only the MCP server"
    echo "  client-only Start only the MCP client"
    echo "  help        Show this help message"
    echo ""
    echo "Options:"
    echo "  --server-port PORT    MCP Server port (default: 8000)"
    echo "  --client-port PORT    MCP Client port (default: 8001)"
    echo "  --log-level LEVEL     Log level (default: INFO)"
    echo "  --skip-deps          Skip dependency installation"
    echo "  --skip-test          Skip connectivity tests"
    echo ""
    echo "Environment Variables:"
    echo "  KUBECONFIG           Path to kubeconfig file (required)"
    echo "  OPENAI_API_KEY       OpenAI API key for AI features"
    echo "  MCP_SERVER_PORT      Override server port"
    echo "  MCP_CLIENT_PORT      Override client port"
    echo "  LOG_LEVEL            Override log level"
    echo "  TZ                   Timezone (default: UTC)"
    echo ""
    echo "Examples:"
    echo "  $0 start                    # Start both server and client"
    echo "  $0 server-only             # Start only server"
    echo "  $0 --server-port 9000 start # Start with custom port"
    echo "  $0 test                     # Run tests only"
    echo "  $0 status                   # Check service status"
}

# Signal handlers
cleanup() {
    log "Received interrupt signal, cleaning up..."
    stop_services
    exit 0
}

trap cleanup SIGINT SIGTERM

# Main execution
main() {
    local skip_deps=false
    local skip_test=false
    local command="start"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --server-port)
                MCP_SERVER_PORT="$2"
                shift 2
                ;;
            --client-port)
                MCP_CLIENT_PORT="$2"
                shift 2
                ;;
            --log-level)
                LOG_LEVEL="$2"
                shift 2
                ;;
            --skip-deps)
                skip_deps=true
                shift
                ;;
            --skip-test)
                skip_test=true
                shift
                ;;
            start|stop|restart|status|test|server-only|client-only|help)
                command="$1"
                shift
                ;;
            -h|--help)
                command="help"
                shift
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Set log level
    export LOG_LEVEL="$LOG_LEVEL"
    
    # Show banner
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║              OpenShift Benchmark MCP Server                     ║"
    echo "║                     Management Script                           ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Execute command
    case "$command" in
        help)
            show_help
            ;;
        test)
            check_prerequisites
            test_connectivity
            run_performance_test
            ;;
        status)
            show_status
            ;;
        stop)
            stop_services
            ;;
        restart)
            stop_services
            sleep 2
            check_prerequisites
            if [[ "$skip_deps" != true ]]; then
                install_dependencies
            fi
            create_directories
            if [[ "$skip_test" != true ]]; then
                test_connectivity
            fi
            start_mcp_server
            start_mcp_client
            show_status
            ;;
        server-only)
            check_prerequisites
            if [[ "$skip_deps" != true ]]; then
                install_dependencies
            fi
            create_directories
            if [[ "$skip_test" != true ]]; then
                test_connectivity
            fi
            start_mcp_server
            show_status
            ;;
        client-only)
            check_prerequisites
            if [[ "$skip_deps" != true ]]; then
                install_dependencies
            fi
            create_directories
            start_mcp_client
            show_status
            ;;
        start)
            check_prerequisites
            if [[ "$skip_deps" != true ]]; then
                install_dependencies
            fi
            create_directories
            if [[ "$skip_test" != true ]]; then
                test_connectivity
            fi
            start_mcp_server
            start_mcp_client
            show_status
            
            log "Services started successfully!"
            log "MCP Server: http://localhost:$MCP_SERVER_PORT"
            log "MCP Client: http://localhost:$MCP_CLIENT_PORT"
            log ""
            log "To stop services: $0 stop"
            log "To check status: $0 status"
            log "To view logs: tail -f logs/mcp_server.log"
            ;;
        *)
            error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"