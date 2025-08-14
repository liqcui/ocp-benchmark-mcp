import json
from typing import Dict, List, Any

def extract_performance_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract performance analysis data from the disk IO JSON file.
    Rounds floating point numbers to 2 decimal places.
    
    Args:
        data: The parsed JSON data
        
    Returns:
        Dictionary containing performance analysis with rounded values
    """
    performance_analysis = data.get("performance_analysis", {})
    
    # Round floating point values in alerts
    if "alerts" in performance_analysis:
        for alert in performance_analysis["alerts"]:
            if "current" in alert:
                alert["current"] = round(alert["current"], 2)
            if "baseline" in alert:
                alert["baseline"] = round(alert["baseline"], 2)
    
    return performance_analysis

def extract_container_info(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract container information from nodes data.
    This function dynamically extracts container names without hardcoding.
    
    Args:
        data: The parsed JSON data
        
    Returns:
        Dictionary containing container information for each node and device
    """
    container_info = {}
    nodes = data.get("nodes", {})
    
    for node_name, node_data in nodes.items():
        container_info[node_name] = {}
        devices = node_data.get("devices", {})
        
        for device_name, device_data in devices.items():
            container_info[node_name][device_name] = {}
            
            # Extract metrics with their container information
            for metric_name in ["read_throughput", "write_throughput", "read_iops", 
                              "write_iops", "read_latency", "write_latency"]:
                if metric_name in device_data:
                    metric_data = device_data[metric_name]
                    
                    # Extract container name from metric_labels if available
                    container_name = None
                    if "metric_labels" in metric_data:
                        container_name = metric_data["metric_labels"].get("container")
                    
                    # Store the container info along with statistics
                    container_info[node_name][device_name][metric_name] = {
                        "container": container_name,
                        "statistics": metric_data.get("statistics", {}),
                        "unit": metric_data.get("unit", "")
                    }
                    
                    # Round floating point values in statistics
                    stats = container_info[node_name][device_name][metric_name]["statistics"]
                    for key in ["min", "max", "mean"]:
                        if key in stats and isinstance(stats[key], (int, float)):
                            stats[key] = round(stats[key], 2)
    
    return container_info

def extract_nodes_performance_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract simplified nodes performance data with statistics only.
    Rounds floating point numbers to 2 decimal places.
    
    Args:
        data: The parsed JSON data
        
    Returns:
        Dictionary containing nodes data with only statistics and units
    """
    nodes_data = {}
    nodes = data.get("nodes", {})
    
    for node_name, node_data in nodes.items():
        nodes_data[node_name] = {"devices": {}}
        devices = node_data.get("devices", {})
        
        for device_name, device_data in devices.items():
            nodes_data[node_name]["devices"][device_name] = {}
            
            # Extract only statistics and units for each metric
            for metric_name in ["read_throughput", "write_throughput", "read_iops", 
                              "write_iops", "read_latency", "write_latency"]:
                if metric_name in device_data:
                    metric_data = device_data[metric_name]
                    
                    # Extract statistics and unit
                    statistics = metric_data.get("statistics", {}).copy()
                    unit = metric_data.get("unit", "")
                    
                    # Round floating point values in statistics
                    for key in ["min", "max", "mean"]:
                        if key in statistics and isinstance(statistics[key], (int, float)):
                            statistics[key] = round(statistics[key], 2)
                    
                    nodes_data[node_name]["devices"][device_name][metric_name] = {
                        "statistics": statistics,
                        "unit": unit
                    }
    
    return nodes_data

def main():
    """
    Main function to demonstrate the extraction functions.
    """
    # Read the JSON file
    try:
        with open('disk-io-full.json', 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Error: disk-io-full.json file not found")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return
    
    # Extract performance analysis
    print("=== Performance Analysis ===")
    performance_analysis = extract_performance_analysis(data)
    print(json.dumps(performance_analysis, indent=2))
    
    print("\n" + "="*50 + "\n")
    
    # Extract container information
    print("=== Container Information ===")
    container_info = extract_container_info(data)
    print(json.dumps(container_info, indent=2))
    
    print("\n" + "="*50 + "\n")
    
    # Extract simplified nodes performance data
    print("=== Nodes Performance Data (Statistics Only) ===")
    nodes_performance = extract_nodes_performance_data(data)
    print(json.dumps(nodes_performance, indent=2))

if __name__ == "__main__":
    main()