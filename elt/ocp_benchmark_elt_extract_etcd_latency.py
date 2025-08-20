import json
import math


def _round_decimals_in_obj(obj, ndigits: int = 2):
    """
    Recursively round all float values to ndigits and convert NaN/inf to "NaN" strings.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return "NaN"
        return round(obj, ndigits)
    if isinstance(obj, list):
        return [_round_decimals_in_obj(item, ndigits) for item in obj]
    if isinstance(obj, dict):
        return {k: _round_decimals_in_obj(v, ndigits) for k, v in obj.items()}
    return obj


def extract_etcd_performance_analysis(data):
    """
    Extract performance analysis and cluster summary information from etcd latency data.
    
    Args:
        data (dict): Dictionary containing etcd latency data
        
    Returns:
        dict: JSON formatted result containing performance analysis and cluster summary
    """
    result = {}
    
    # Extract performance_analysis
    if "performance_analysis" in data:
        perf_analysis = data["performance_analysis"].copy()
        
        # Round numeric values in summary to 2 decimal places
        if "summary" in perf_analysis:
            summary = perf_analysis["summary"]
            for key, value in summary.items():
                if isinstance(value, (int, float)) and not math.isnan(value):
                    summary[key] = round(value, 2)
        
        result["performance_analysis"] = perf_analysis
    
    # Extract cluster_summary from etcd_latency operations
    if "etcd_latency" in data and "operations" in data["etcd_latency"]:
        operations = data["etcd_latency"]["operations"]
        if "cluster_summary" in operations:
            cluster_summary = operations["cluster_summary"].copy()
            
            # Round numeric values to 2 decimal places, handle NaN
            for key, value in cluster_summary.items():
                if isinstance(value, (int, float)):
                    if math.isnan(value):
                        cluster_summary[key] = "NaN"
                    else:
                        cluster_summary[key] = round(value, 2)
            
            result["cluster_summary"] = cluster_summary
    
    return _round_decimals_in_obj(result, 2)


def extract_active_operations(data):
    """
    Extract operations with latency stats where at least one of min/mean/max is greater than 0.0.
    
    Args:
        data (dict): Dictionary containing etcd latency data
        
    Returns:
        dict: JSON formatted result containing active operations with only latency stats
    """
    result = {"active_operations": {}}
    
    if "etcd_latency" in data and "operations" in data["etcd_latency"]:
        operations = data["etcd_latency"]["operations"]
        
        for operation_name, operation_data in operations.items():
            # Skip cluster_summary as it's handled separately
            if operation_name == "cluster_summary":
                continue
                
            if "latency_stats" in operation_data:
                stats = operation_data["latency_stats"]
                
                # Check if any of min, mean, max is greater than 0.0
                min_val = stats.get("min", 0.0)
                max_val = stats.get("max", 0.0) 
                mean_val = stats.get("mean", 0.0)
                
                if (min_val > 0.0 or max_val > 0.0 or mean_val > 0.0):
                    # Round values to 2 decimal places
                    rounded_stats = {}
                    for key, value in stats.items():
                        if isinstance(value, (int, float)) and not math.isnan(value):
                            rounded_stats[key] = round(value, 2)
                        else:
                            rounded_stats[key] = value

                    # Only include latency_stats keyed by operation name
                    result["active_operations"][operation_name] = {
                        "latency_stats": rounded_stats
                    }
    
    return _round_decimals_in_obj(result, 2)


def process_etcd_data(data):
    """
    Main function to process etcd latency data and extract all required information.
    
    Args:
        data (dict or str): Dictionary containing etcd data or JSON string
        
    Returns:
        dict: Complete extracted information in JSON format
    """
    # Handle both dict and JSON string inputs
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON format: {str(e)}"}
    
    if not isinstance(data, dict):
        return {"error": "Input must be a dictionary or valid JSON string"}
    
    # Extract performance analysis and cluster summary
    performance_data = extract_etcd_performance_analysis(data)
    
    # Extract active operations
    active_operations_data = extract_active_operations(data)
    
    # Combine results
    result = {
        **performance_data,
        **active_operations_data
    }
    
    return _round_decimals_in_obj(result, 2)


# Example usage
if __name__ == "__main__":
    # Example with the provided data
    sample_data = {
        "performance_analysis": {
            "overall_status": "optimal",
            "alerts": [],
            "summary": {
                "total_operations": 0,
                "high_latency_operations": 0,
                "critical_latency_operations": 0,
                "etcd_operations": 616,
                "high_etcd_latency": 0
            }
        },
        "etcd_latency": {
            "operations": {
                "cluster_summary": {
                    "min": float('nan'),
                    "max": float('nan'),
                    "mean": float('nan'),
                    "count": 37534
                },
                "get:pods": {
                    "latency_stats": {
                        "min": 0.1,
                        "max": 0.3,
                        "mean": 0.0,
                        "count": 0
                    },
                    "labels": {
                        "operation": "get",
                        "type": "pods"
                    }
                }
            }
        }
    }
    
    # Process the data
    result = process_etcd_data(sample_data)
    print(json.dumps(result, indent=2))