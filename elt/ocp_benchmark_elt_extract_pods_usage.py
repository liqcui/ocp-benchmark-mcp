import json
from typing import Any


def _round_decimals_in_obj(obj: Any, ndigits: int = 2) -> Any:
    """
    Recursively round all float values in a Python object (dict/list/scalars).
    Only float types are rounded; ints/str/bool/None are left untouched.
    """
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, list):
        return [_round_decimals_in_obj(item, ndigits) for item in obj]
    if isinstance(obj, dict):
        return {key: _round_decimals_in_obj(value, ndigits) for key, value in obj.items()}
    return obj


def safe_get(obj, key, default=None):
    """
    Safely get a value from an object, handling cases where obj might not be a dict.
    
    Args:
        obj: The object to get the value from
        key: The key to retrieve
        default: Default value if key doesn't exist or obj is not a dict
        
    Returns:
        The value or default
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def safe_round(value, decimals=2):
    """
    Safely round a numeric value, handling cases where value might not be numeric.
    
    Args:
        value: The value to round
        decimals: Number of decimal places
        
    Returns:
        Rounded value or 0 if value is not numeric
    """
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return round(float(value), decimals)
        return 0
    except (ValueError, TypeError):
        return 0


def extract_summary(data):
    """
    Extract summary information from the pods usage data.
    
    Args:
        data (dict): The loaded JSON data
        
    Returns:
        dict: Summary information with proper error handling
    """
    if not isinstance(data, dict):
        print(f"Warning: Expected dict, got {type(data)}")
        return {
            "total_pods": 0,
            "total_containers": 0,
            "namespaces": {},
            "high_cpu_pods": 0,
            "high_memory_pods": 0
        }
    
    summary = safe_get(data, "summary", {})
    
    # Extract basic counts
    extracted_summary = {
        "total_pods": safe_get(summary, "total_pods", 0),
        "total_containers": safe_get(summary, "total_containers", 0),
        "namespaces": {},
        "high_cpu_pods": safe_get(summary, "high_cpu_pods", 0),
        "high_memory_pods": safe_get(summary, "high_memory_pods", 0)
    }
    
    # Extract namespace information
    namespaces = safe_get(summary, "namespaces", {})
    if isinstance(namespaces, dict):
        for namespace, info in namespaces.items():
            if isinstance(info, dict):
                extracted_summary["namespaces"][namespace] = {
                    "pod_count": safe_get(info, "pod_count", 0),
                    "container_count": safe_get(info, "container_count", 0)
                }
    
    return _round_decimals_in_obj(extracted_summary, 2)


def extract_pod_totals(data):
    """
    Extract pod totals information from the pods usage data.
    Includes both aggregate totals and individual pod totals.
    
    Args:
        data (dict): The loaded JSON data
        
    Returns:
        dict: Pod totals information with individual pod data and aggregate totals
    """
    if not isinstance(data, dict):
        print(f"Warning: Expected dict, got {type(data)}")
        return {}
    
    extracted_totals = {}
    
    # First, extract the aggregate pod_totals if it exists
    if "pod_totals" in data:
        pod_totals = safe_get(data, "pod_totals", {})
        
        # Extract aggregate CPU usage stats
        cpu_usage = safe_get(pod_totals, "cpu_usage", {})
        if isinstance(cpu_usage, dict) and cpu_usage:
            extracted_totals["aggregate_totals"] = extracted_totals.get("aggregate_totals", {})
            extracted_totals["aggregate_totals"]["cpu_usage"] = {
                "min": safe_round(safe_get(cpu_usage, "min", 0)),
                "max": safe_round(safe_get(cpu_usage, "max", 0)),
                "mean": safe_round(safe_get(cpu_usage, "mean", 0)),
                "count": safe_get(cpu_usage, "count", 0)
            }
        
        # Extract aggregate memory usage stats
        memory_usage = safe_get(pod_totals, "memory_usage", {})
        if isinstance(memory_usage, dict) and memory_usage:
            extracted_totals["aggregate_totals"] = extracted_totals.get("aggregate_totals", {})
            extracted_totals["aggregate_totals"]["memory_usage"] = {
                "min": safe_round(safe_get(memory_usage, "min", 0)),
                "max": safe_round(safe_get(memory_usage, "max", 0)),
                "mean": safe_round(safe_get(memory_usage, "mean", 0)),
                "count": safe_get(memory_usage, "count", 0),
                "unit": safe_get(memory_usage, "unit", "MB")
            }
    
    # Extract individual pod totals
    pods = safe_get(data, "pods", {})
    if isinstance(pods, dict):
        extracted_totals["individual_pods"] = {}
        
        for pod_name, pod_data in pods.items():
            if not isinstance(pod_data, dict):
                continue
                
            pod_totals = safe_get(pod_data, "pod_totals", {})
            if isinstance(pod_totals, dict) and pod_totals:
                pod_stats = {}
                
                # Extract pod CPU usage
                cpu_usage = safe_get(pod_totals, "cpu_usage", {})
                if isinstance(cpu_usage, dict) and cpu_usage:
                    pod_stats["cpu_usage"] = {
                        "min": safe_round(safe_get(cpu_usage, "min", 0)),
                        "max": safe_round(safe_get(cpu_usage, "max", 0)),
                        "mean": safe_round(safe_get(cpu_usage, "mean", 0)),
                        "count": safe_get(cpu_usage, "count", 0),
                        "unit": safe_get(cpu_usage, "unit", "percent")
                    }
                
                # Extract pod memory usage
                memory_usage = safe_get(pod_totals, "memory_usage", {})
                if isinstance(memory_usage, dict) and memory_usage:
                    pod_stats["memory_usage"] = {
                        "min": safe_round(safe_get(memory_usage, "min", 0)),
                        "max": safe_round(safe_get(memory_usage, "max", 0)),
                        "mean": safe_round(safe_get(memory_usage, "mean", 0)),
                        "count": safe_get(memory_usage, "count", 0),
                        "unit": safe_get(memory_usage, "unit", "MB")
                    }
                
                if pod_stats:
                    extracted_totals["individual_pods"][pod_name] = pod_stats
    
    # If no individual pod totals found, try to calculate from container data
    if "individual_pods" not in extracted_totals or not extracted_totals["individual_pods"]:
        extracted_totals["individual_pods"] = calculate_pod_totals_from_containers(data)
    
    return _round_decimals_in_obj(extracted_totals, 2)


def calculate_pod_totals_from_containers(data):
    """
    Calculate pod totals from individual container data when pod_totals are not available.
    
    Args:
        data (dict): The loaded JSON data
        
    Returns:
        dict: Calculated pod totals from container data
    """
    pods = safe_get(data, "pods", {})
    if not isinstance(pods, dict):
        return {}
    
    pod_totals = {}
    
    for pod_name, pod_data in pods.items():
        if not isinstance(pod_data, dict):
            continue
            
        containers = safe_get(pod_data, "containers", {})
        if not isinstance(containers, dict):
            continue
        
        pod_cpu_stats = []
        pod_memory_stats = []
        
        # Collect all container stats for this pod
        for container_name, container_data in containers.items():
            if not isinstance(container_data, dict):
                continue
            
            # Get CPU stats
            cpu_usage = safe_get(container_data, "cpu_usage", {})
            if isinstance(cpu_usage, dict):
                cpu_stats = safe_get(cpu_usage, "statistics", {})
                if isinstance(cpu_stats, dict) and safe_get(cpu_stats, "mean"):
                    pod_cpu_stats.append({
                        "min": safe_get(cpu_stats, "min", 0),
                        "max": safe_get(cpu_stats, "max", 0),
                        "mean": safe_get(cpu_stats, "mean", 0)
                    })
            
            # Get Memory stats
            memory_usage = safe_get(container_data, "memory_usage", {})
            if isinstance(memory_usage, dict):
                memory_stats = safe_get(memory_usage, "statistics", {})
                if isinstance(memory_stats, dict) and safe_get(memory_stats, "mean"):
                    pod_memory_stats.append({
                        "min": safe_get(memory_stats, "min", 0),
                        "max": safe_get(memory_stats, "max", 0),
                        "mean": safe_get(memory_stats, "mean", 0)
                    })
        
        # Calculate pod aggregates
        if pod_cpu_stats or pod_memory_stats:
            pod_summary = {}
            
            if pod_cpu_stats:
                cpu_means = [stat["mean"] for stat in pod_cpu_stats if stat["mean"]]
                cpu_mins = [stat["min"] for stat in pod_cpu_stats if stat["min"]]
                cpu_maxs = [stat["max"] for stat in pod_cpu_stats if stat["max"]]
                
                if cpu_means:
                    pod_summary["cpu_usage"] = {
                        "min": safe_round(min(cpu_mins) if cpu_mins else 0),
                        "max": safe_round(max(cpu_maxs) if cpu_maxs else 0),
                        "mean": safe_round(sum(cpu_means) / len(cpu_means)),
                        "count": len(pod_cpu_stats),
                        "unit": "percent"
                    }
            
            if pod_memory_stats:
                memory_means = [stat["mean"] for stat in pod_memory_stats if stat["mean"]]
                memory_mins = [stat["min"] for stat in pod_memory_stats if stat["min"]]
                memory_maxs = [stat["max"] for stat in pod_memory_stats if stat["max"]]
                
                if memory_means:
                    pod_summary["memory_usage"] = {
                        "min": safe_round(min(memory_mins) if memory_mins else 0),
                        "max": safe_round(max(memory_maxs) if memory_maxs else 0),
                        "mean": safe_round(sum(memory_means) / len(memory_means)),
                        "count": len(pod_memory_stats),
                        "unit": "MB"
                    }
            
            if pod_summary:
                pod_totals[pod_name] = pod_summary
    
    return _round_decimals_in_obj(pod_totals, 2)


def extract_container_stats(data, pod_name=None, container_name=None):
    """
    Extract container statistics from the pods usage data.
    This function can extract stats for a specific container or all containers.
    
    Args:
        data (dict): The loaded JSON data
        pod_name (str, optional): Specific pod name to extract from
        container_name (str, optional): Specific container name to extract
        
    Returns:
        dict: Container statistics with rounded floating point numbers
    """
    if not isinstance(data, dict):
        print(f"Warning: Expected dict, got {type(data)}")
        return {}
    
    pods = safe_get(data, "pods", {})
    if not isinstance(pods, dict):
        print(f"Warning: 'pods' should be dict, got {type(pods)}")
        return {}
    
    container_stats = {}
    
    for pod_key, pod_info in pods.items():
        # Validate pod_info is a dictionary
        if not isinstance(pod_info, dict):
            print(f"Warning: Pod info for '{pod_key}' is not a dict, got {type(pod_info)}")
            continue
            
        # If pod_name is specified, only process that pod
        if pod_name and pod_name not in str(pod_key):
            continue
        
        containers = safe_get(pod_info, "containers", {})
        if not isinstance(containers, dict):
            print(f"Warning: Containers for pod '{pod_key}' is not a dict, got {type(containers)}")
            continue
        
        for container_key, container_info in containers.items():
            # Validate container_info is a dictionary
            if not isinstance(container_info, dict):
                print(f"Warning: Container info for '{container_key}' is not a dict, got {type(container_info)}")
                continue
                
            # If container_name is specified, only process that container
            if container_name and container_key != container_name:
                continue
            
            # Create a unique key for this container
            full_container_name = f"{pod_key}/{container_key}" if not container_name else container_key
            
            container_stats[full_container_name] = {}
            
            # Extract CPU usage statistics
            cpu_usage = safe_get(container_info, "cpu_usage", {})
            if isinstance(cpu_usage, dict):
                cpu_stats = safe_get(cpu_usage, "statistics", {})
                if isinstance(cpu_stats, dict) and cpu_stats:
                    container_stats[full_container_name]["cpu_usage"] = {
                        "statistics": {
                            "min": safe_round(safe_get(cpu_stats, "min", 0)),
                            "max": safe_round(safe_get(cpu_stats, "max", 0)),
                            "mean": safe_round(safe_get(cpu_stats, "mean", 0)),
                            "count": safe_get(cpu_stats, "count", 0)
                        },
                        "unit": safe_get(cpu_usage, "unit", "percent")
                    }
            
            # Extract memory usage statistics
            memory_usage = safe_get(container_info, "memory_usage", {})
            if isinstance(memory_usage, dict):
                memory_stats = safe_get(memory_usage, "statistics", {})
                if isinstance(memory_stats, dict) and memory_stats:
                    container_stats[full_container_name]["memory_usage"] = {
                        "statistics": {
                            "min": safe_round(safe_get(memory_stats, "min", 0)),
                            "max": safe_round(safe_get(memory_stats, "max", 0)),
                            "mean": safe_round(safe_get(memory_stats, "mean", 0)),
                            "count": safe_get(memory_stats, "count", 0)
                        },
                        "unit": safe_get(memory_usage, "unit", "MB")
                    }
    
    return _round_decimals_in_obj(container_stats, 2)


def validate_json_data(data):
    """
    Validate that the loaded data has the expected structure.
    
    Args:
        data: The loaded JSON data
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, f"Root data should be dict, got {type(data)}"
    
    # Check for required top-level keys
    if "summary" not in data and "pods" not in data:
        return False, "Data should contain either 'summary' or 'pods' key"
    
    return True, "Data structure is valid"


def get_pod_names_and_basic_stats(data):
    """
    Get a simple list of pod names with their basic resource usage statistics.
    
    Args:
        data (dict): The loaded JSON data
        
    Returns:
        dict: Pod names with basic CPU and memory stats
    """
    if not isinstance(data, dict):
        return {}
    
    pods = safe_get(data, "pods", {})
    if not isinstance(pods, dict):
        return {}
    
    pod_summary = {}
    
    for pod_name, pod_data in pods.items():
        if not isinstance(pod_data, dict):
            continue
        
        # Try to get pod totals first
        pod_totals = safe_get(pod_data, "pod_totals", {})
        
        if isinstance(pod_totals, dict) and pod_totals:
            # Use existing pod totals
            pod_stats = {}
            
            cpu_usage = safe_get(pod_totals, "cpu_usage", {})
            if isinstance(cpu_usage, dict) and safe_get(cpu_usage, "mean"):
                pod_stats["cpu_mean"] = safe_round(safe_get(cpu_usage, "mean", 0))
                pod_stats["cpu_unit"] = "percent"
            
            memory_usage = safe_get(pod_totals, "memory_usage", {})
            if isinstance(memory_usage, dict) and safe_get(memory_usage, "mean"):
                pod_stats["memory_mean"] = safe_round(safe_get(memory_usage, "mean", 0))
                pod_stats["memory_unit"] = safe_get(memory_usage, "unit", "MB")
            
            if pod_stats:
                pod_summary[pod_name] = pod_stats
        else:
            # Calculate from containers
            containers = safe_get(pod_data, "containers", {})
            if isinstance(containers, dict):
                cpu_means = []
                memory_means = []
                
                for container_data in containers.values():
                    if not isinstance(container_data, dict):
                        continue
                    
                    # CPU stats
                    cpu_usage = safe_get(container_data, "cpu_usage", {})
                    if isinstance(cpu_usage, dict):
                        cpu_stats = safe_get(cpu_usage, "statistics", {})
                        if isinstance(cpu_stats, dict):
                            cpu_mean = safe_get(cpu_stats, "mean", 0)
                            if cpu_mean:
                                cpu_means.append(cpu_mean)
                    
                    # Memory stats
                    memory_usage = safe_get(container_data, "memory_usage", {})
                    if isinstance(memory_usage, dict):
                        memory_stats = safe_get(memory_usage, "statistics", {})
                        if isinstance(memory_stats, dict):
                            memory_mean = safe_get(memory_stats, "mean", 0)
                            if memory_mean:
                                memory_means.append(memory_mean)
                
                pod_stats = {}
                if cpu_means:
                    pod_stats["cpu_mean"] = safe_round(sum(cpu_means) / len(cpu_means))
                    pod_stats["cpu_unit"] = "percent"
                
                if memory_means:
                    pod_stats["memory_mean"] = safe_round(sum(memory_means) / len(memory_means))
                    pod_stats["memory_unit"] = "MB"
                
                if pod_stats:
                    pod_summary[pod_name] = pod_stats
    
    return _round_decimals_in_obj(pod_summary, 2)


def main():
    """
    Main function to demonstrate the extraction functions with error handling.
    """
    # Read the JSON file
    try:
        with open("pods-usage.json", "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Error: pods-usage.json file not found!")
        print("Make sure the file exists in the current directory.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print("Please check that the file contains valid JSON.")
        return
    except Exception as e:
        print(f"Unexpected error reading file: {e}")
        return
    
    # Validate data structure
    is_valid, error_msg = validate_json_data(data)
    if not is_valid:
        print(f"Error: {error_msg}")
        return
    
    print("=== DATA VALIDATION PASSED ===\n")
    
    # Extract and display summary information
    try:
        print("=== SUMMARY ===")
        summary = extract_summary(data)
        print(json.dumps(summary, indent=2))
    except Exception as e:
        print(f"Error extracting summary: {e}")
    
    # Extract and display pod totals with individual pod names
    try:
        print("\n=== POD TOTALS (Including Individual Pods) ===")
        pod_totals = extract_pod_totals(data)
        if pod_totals:
            print(json.dumps(pod_totals, indent=2))
        else:
            print("No pod totals data found or extracted.")
    except Exception as e:
        print(f"Error extracting pod totals: {e}")
    
    # Extract and display simple pod names with basic stats
    try:
        print("\n=== POD NAMES WITH BASIC STATS ===")
        pod_names_stats = get_pod_names_and_basic_stats(data)
        if pod_names_stats:
            print(json.dumps(pod_names_stats, indent=2))
        else:
            print("No pod names/stats found or extracted.")
    except Exception as e:
        print(f"Error extracting pod names and stats: {e}")
    
    # Extract and display container statistics
    try:
        print("\n=== ALL CONTAINER STATISTICS ===")
        all_containers = extract_container_stats(data)
        if all_containers:
            print(json.dumps(all_containers, indent=2))
        else:
            print("No container statistics found or extracted.")
    except Exception as e:
        print(f"Error extracting container statistics: {e}")
    
    # Example: Extract specific container stats
    try:
        print("\n=== SPECIFIC CONTAINER (etcd-metrics) EXAMPLE ===")
        etcd_metrics_stats = extract_container_stats(data, container_name="etcd-metrics")
        if etcd_metrics_stats:
            print(json.dumps(etcd_metrics_stats, indent=2))
        else:
            print("No etcd-metrics container found.")
    except Exception as e:
        print(f"Error extracting etcd-metrics stats: {e}")


def get_all_container_names(data):
    """
    Get all unique container names from the data.
    
    Args:
        data (dict): The loaded JSON data
        
    Returns:
        list: List of all unique container names
    """
    if not isinstance(data, dict):
        return []
    
    container_names = set()
    pods = safe_get(data, "pods", {})
    
    if isinstance(pods, dict):
        for pod_info in pods.values():
            if isinstance(pod_info, dict):
                containers = safe_get(pod_info, "containers", {})
                if isinstance(containers, dict):
                    container_names.update(containers.keys())
    
    return list(container_names)


def extract_stats_by_container_name(data, container_name):
    """
    Extract statistics for all instances of a specific container name across all pods.
    
    Args:
        data (dict): The loaded JSON data
        container_name (str): Name of the container to extract stats for
        
    Returns:
        dict: Statistics for the specified container across all pods
    """
    return extract_container_stats(data, container_name=container_name)


if __name__ == "__main__":
    main()