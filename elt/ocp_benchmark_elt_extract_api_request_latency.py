import json
import math


def _parse_input_data(input_data):
    """
    Helper function to parse input data from dict or JSON string.
    
    Args:
        input_data (dict or str): Input data as dictionary or JSON string
        
    Returns:
        dict: Parsed data dictionary
    """
    if isinstance(input_data, dict):
        return input_data
    elif isinstance(input_data, str):
        return json.loads(input_data)
    else:
        raise ValueError("Input must be a dictionary or JSON string")


def extract_api_performance_analysis(input_data):
    """
    Extract performance analysis information from the API latency data.
    
    Args:
        input_data (dict or str): API latency data as dictionary or JSON string
        
    Returns:
        dict: Performance analysis data with rounded floating point numbers
    """
    try:
        data = _parse_input_data(input_data)
        
        # Extract performance analysis section
        performance_analysis = data.get("performance_analysis", {})
        
        # Create the result dictionary
        result = {
            "performance_analysis": {
                "overall_status": performance_analysis.get("overall_status", ""),
                "alerts": performance_analysis.get("alerts", []),
                "summary": {
                    "total_operations": performance_analysis.get("summary", {}).get("total_operations", 0),
                    "high_latency_operations": performance_analysis.get("summary", {}).get("high_latency_operations", 0),
                    "critical_latency_operations": performance_analysis.get("summary", {}).get("critical_latency_operations", 0),
                    "etcd_operations": performance_analysis.get("summary", {}).get("etcd_operations", 0),
                    "high_etcd_latency": performance_analysis.get("summary", {}).get("high_etcd_latency", 0)
                }
            }
        }
        
        return result
        
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


def extract_operation_statistics(input_data, operation_name=None):
    """
    Extract statistics for operations, with optional filtering by operation name.
    
    Args:
        input_data (dict or str): API latency data as dictionary or JSON string
        operation_name (str, optional): Specific operation name to filter by (e.g., "LIST:helmchartrepositories")
        
    Returns:
        dict: Statistics data for operations, rounded to 2 decimal places
    """
    try:
        data = _parse_input_data(input_data)
        
        # Extract API server latency operations
        operations = data.get("api_server_latency", {}).get("operations", {})
        
        result = {}
        
        for op_name, op_data in operations.items():
            # Filter by operation name if specified
            if operation_name and op_name != operation_name:
                continue
            
            # Extract p95 and p99 statistics
            p95_stats = op_data.get("p95", {}).get("statistics", {})
            p99_stats = op_data.get("p99", {}).get("statistics", {})
            
            # Helper function to round numbers to 2 decimal places
            def round_value(value):
                if isinstance(value, (int, float)) and not math.isnan(value):
                    return round(value, 6)
                return 0.0
            
            result[op_name] = {
                "p95": {
                    "statistics": {
                        "min": round_value(p95_stats.get("min", 0.0)),
                        "max": round_value(p95_stats.get("max", 0.0)),
                        "mean": round_value(p95_stats.get("mean", 0.0)),
                        "count": p95_stats.get("count", 0)
                    },
                    "unit": op_data.get("p95", {}).get("unit", "ms")
                },
                "p99": {
                    "statistics": {
                        "min": round_value(p99_stats.get("min", 0.0)),
                        "max": round_value(p99_stats.get("max", 0.0)),
                        "mean": round_value(p99_stats.get("mean", 0.0)),
                        "count": p99_stats.get("count", 0)
                    },
                    "unit": op_data.get("p99", {}).get("unit", "ms")
                }
            }
        
        return result
        
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


def extract_cluster_summary(input_data):
    """
    Extract cluster summary information from the API latency data.
    
    Args:
        input_data (dict or str): API latency data as dictionary or JSON string
        
    Returns:
        dict: Cluster summary data with rounded floating point numbers
    """
    try:
        data = _parse_input_data(input_data)
        
        # Extract cluster summary section
        cluster_summary = data.get("api_server_latency", {}).get("cluster_summary", {})
        
        # Helper function to round numbers to 6 decimal places, handle NaN
        def round_value(value):
            if isinstance(value, (int, float)):
                if math.isnan(value):
                    return None  # or you can return "NaN" as string if preferred
                return round(value, 6)
            return 0.0
        
        # Extract p95_overall and p99_overall
        p95_overall = cluster_summary.get("p95_overall", {})
        p99_overall = cluster_summary.get("p99_overall", {})
        
        result = {
            "cluster_summary": {
                "p95_overall": {
                    "min": round_value(p95_overall.get("min")),
                    "max": round_value(p95_overall.get("max")),
                    "mean": round_value(p95_overall.get("mean")),
                    "count": p95_overall.get("count", 0)
                },
                "p99_overall": {
                    "min": round_value(p99_overall.get("min")),
                    "max": round_value(p99_overall.get("max")),
                    "mean": round_value(p99_overall.get("mean")),
                    "count": p99_overall.get("count", 0)
                }
            }
        }
        
        return result
        
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


def list_available_operations(input_data):
    """
    List all available operation names in the data.
    
    Args:
        input_data (dict or str): API latency data as dictionary or JSON string
        
    Returns:
        list: List of available operation names
    """
    try:
        data = _parse_input_data(input_data)
        
        operations = data.get("api_server_latency", {}).get("operations", {})
        return list(operations.keys())
        
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


# Example usage
if __name__ == "__main__":
    # Sample data - you can use either dict or JSON string
    sample_data = {
        "performance_analysis": {
            "overall_status": "optimal",
            "alerts": [],
            "summary": {
                "total_operations": 651,
                "high_latency_operations": 0,
                "critical_latency_operations": 0,
                "etcd_operations": 0,
                "high_etcd_latency": 0
            }
        },
        "api_server_latency": {
            "operations": {
                "LIST:helmchartrepositories": {
                    "p95": {
                        "statistics": {
                            "min": 1.5,
                            "max": 25.75,
                            "mean": 10.25,
                            "count": 100
                        },
                        "unit": "ms"
                    },
                    "p99": {
                        "statistics": {
                            "min": 2.1,
                            "max": 30.85,
                            "mean": 12.45,
                            "count": 100
                        },
                        "unit": "ms"
                    }
                }
            },
            "cluster_summary": {
                "p95_overall": {
                    "min": float('nan'),
                    "max": float('nan'),
                    "mean": float('nan'),
                    "count": 93757
                },
                "p99_overall": {
                    "min": float('nan'),
                    "max": float('nan'),
                    "mean": float('nan'),
                    "count": 93757
                }
            }
        }
    }
    
    # Example 1: Using dictionary input
    print("Performance Analysis (from dict):")
    print(json.dumps(extract_api_performance_analysis(sample_data), indent=2))
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Extract cluster summary
    print("Cluster Summary:")
    print(json.dumps(extract_cluster_summary(sample_data), indent=2))
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Using JSON string input
    json_string = json.dumps(sample_data)
    print("Cluster Summary (from JSON string):")
    print(json.dumps(extract_cluster_summary(json_string), indent=2))
    
    print("\n" + "="*50 + "\n")
    
    # Example 4: List available operations
    print("Available Operations:")
    operations = list_available_operations(sample_data)
    for op in operations:
        print(f"  - {op}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 5: Extract all operation statistics
    print("All Operation Statistics:")
    all_stats = extract_operation_statistics(sample_data)
    print(json.dumps(all_stats, indent=2))
    
    print("\n" + "="*50 + "\n")
    
    # Example 6: Extract specific operation statistics
    print("Specific Operation Statistics (LIST:helmchartrepositories):")
    specific_stats = extract_operation_statistics(sample_data, "LIST:helmchartrepositories")
    print(json.dumps(specific_stats, indent=2))
    
    print("\n" + "="*50 + "\n")
    
    # Example 7: Reading from the uploaded file (if you want to use it directly)
    print("Example of reading from your uploaded file:")
    print("# To use with your actual data from the file:")
    print("# with open('api-latency-simple.json', 'r') as f:")
    print("#     file_data = json.load(f)")
    print("#     result = extract_cluster_summary(file_data)")