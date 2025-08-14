import json
from typing import Dict, Any, Union

def round_float(value, decimal_places: int = 2):
    """
    Round a floating point number to specified decimal places.
    Returns the original value if it's not a number.
    """
    if isinstance(value, (int, float)) and value is not None:
        return round(float(value), decimal_places)
    return value

def extract_cluster_statistics(data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract cluster statistics and baseline comparison from node usage data.
    All floating point numbers are rounded to two decimal places.
    
    Args:
        data: Either a JSON string or dictionary containing cluster usage data
        
    Returns:
        Dictionary containing extracted cluster statistics and baseline comparison
        
    Raises:
        ValueError: If required fields are missing from the input data
        TypeError: If input data is not a string or dictionary
    """
    
    # Handle string input (JSON)
    if isinstance(data, str):
        try:
            parsed_data = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")
    elif isinstance(data, dict):
        parsed_data = data
    else:
        raise TypeError("Input data must be a JSON string or dictionary")
    
    # Validate required fields exist
    required_fields = ['cluster_statistics', 'baseline_comparison']
    for field in required_fields:
        if field not in parsed_data:
            raise ValueError(f"Missing required field: {field}")
    
    # Extract cluster statistics
    cluster_stats = parsed_data.get('cluster_statistics', {})
    baseline_comp = parsed_data.get('baseline_comparison', {})
    
    # Validate cluster statistics structure
    if 'cpu_usage' not in cluster_stats or 'memory_usage' not in cluster_stats:
        raise ValueError("Missing cpu_usage or memory_usage in cluster_statistics")
    
    # Validate baseline comparison structure
    if 'cpu' not in baseline_comp or 'memory' not in baseline_comp:
        raise ValueError("Missing cpu or memory in baseline_comparison")
    
    # Build the extracted result with rounded values
    result = {
        "cluster_statistics": {
            "cpu_usage": {
                "min": round_float(cluster_stats['cpu_usage'].get('min')),
                "max": round_float(cluster_stats['cpu_usage'].get('max')),
                "mean": round_float(cluster_stats['cpu_usage'].get('mean')),
                "count": cluster_stats['cpu_usage'].get('count')
            },
            "memory_usage": {
                "min": round_float(cluster_stats['memory_usage'].get('min')),
                "max": round_float(cluster_stats['memory_usage'].get('max')),
                "mean": round_float(cluster_stats['memory_usage'].get('mean')),
                "count": cluster_stats['memory_usage'].get('count')
            },
            "total_nodes": cluster_stats.get('total_nodes'),
            "query_duration_hours": round_float(cluster_stats.get('query_duration_hours'))
        },
        "baseline_comparison": {
            "cpu": {
                "current_mean": round_float(baseline_comp['cpu'].get('current_mean')),
                "current_max": round_float(baseline_comp['cpu'].get('current_max')),
                "baseline_mean": round_float(baseline_comp['cpu'].get('baseline_mean')),
                "baseline_max": round_float(baseline_comp['cpu'].get('baseline_max')),
                "status": baseline_comp['cpu'].get('status'),
                "message": baseline_comp['cpu'].get('message')
            },
            "memory": {
                "current_mean": round_float(baseline_comp['memory'].get('current_mean')),
                "current_max": round_float(baseline_comp['memory'].get('current_max')),
                "baseline_mean": round_float(baseline_comp['memory'].get('baseline_mean')),
                "baseline_max": round_float(baseline_comp['memory'].get('baseline_max')),
                "status": baseline_comp['memory'].get('status'),
                "message": baseline_comp['memory'].get('message')
            }
        }
    }
    
    return result

def extract_cluster_statistics_from_file(file_path: str) -> Dict[str, Any]:
    """
    Extract cluster statistics from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing cluster usage data
        
    Returns:
        Dictionary containing extracted cluster statistics and baseline comparison
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file contains invalid JSON or missing required fields
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
        return extract_cluster_statistics(data)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading file {file_path}: {e}")

# Example usage and test function
def test_extractor():
    """Test the extractor function with sample data"""
    
    # Sample data (simplified version of your actual data)
    sample_data = {
        "cluster_statistics": {
            "cpu_usage": {
                "min": 1.0561403508804745,
                "max": 6.2771929825019,
                "mean": 2.6924967644523123,
                "count": 9760
            },
            "memory_usage": {
                "min": 7.277930990202619,
                "max": 11.66060932412144,
                "mean": 9.449749953264506,
                "count": 305
            },
            "total_nodes": 5,
            "query_duration_hours": 1.0
        },
        "baseline_comparison": {
            "cpu": {
                "current_mean": 2.69,
                "current_max": 6.28,
                "baseline_mean": 45.0,
                "baseline_max": 80.0,
                "status": "normal",
                "message": "Values within normal range (mean: 2.7%, max: 6.3%)"
            },
            "memory": {
                "current_mean": 9.45,
                "current_max": 11.66,
                "baseline_mean": 50.0,
                "baseline_max": 85.0,
                "status": "normal",
                "message": "Values within normal range (mean: 9.4%, max: 11.7%)"
            }
        }
    }
    
    try:
        result = extract_cluster_statistics(sample_data)
        print("Extraction successful!")
        print(json.dumps(result, indent=2))
        return result
    except Exception as e:
        print(f"Test failed: {e}")
        return None

def extract_node_statistics(data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract individual node names and their CPU/memory statistics.
    All floating point numbers are rounded to two decimal places.
    
    Args:
        data: Either a JSON string or dictionary containing cluster usage data
        
    Returns:
        Dictionary containing node names and their statistics
        
    Raises:
        ValueError: If required fields are missing from the input data
        TypeError: If input data is not a string or dictionary
    """
    
    # Handle string input (JSON)
    if isinstance(data, str):
        try:
            parsed_data = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")
    elif isinstance(data, dict):
        parsed_data = data
    else:
        raise TypeError("Input data must be a JSON string or dictionary")
    
    # Validate nodes field exists
    if 'nodes' not in parsed_data:
        raise ValueError("Missing 'nodes' field in input data")
    
    nodes_data = parsed_data['nodes']
    result = {
        "nodes": {}
    }
    
    for node_name, node_info in nodes_data.items():
        node_stats = {
            "node_name": node_name,
            "cpu_usage": {
                "statistics": {
                    "min": round_float(node_info.get('cpu_usage', {}).get('statistics', {}).get('min')),
                    "max": round_float(node_info.get('cpu_usage', {}).get('statistics', {}).get('max')),
                    "mean": round_float(node_info.get('cpu_usage', {}).get('statistics', {}).get('mean')),
                    "count": node_info.get('cpu_usage', {}).get('statistics', {}).get('count')
                },
                "unit": node_info.get('cpu_usage', {}).get('unit', 'percent')
            },
            "memory_usage": {
                "statistics": {
                    "min": round_float(node_info.get('memory_usage', {}).get('statistics', {}).get('min')),
                    "max": round_float(node_info.get('memory_usage', {}).get('statistics', {}).get('max')),
                    "mean": round_float(node_info.get('memory_usage', {}).get('statistics', {}).get('mean')),
                    "count": node_info.get('memory_usage', {}).get('statistics', {}).get('count')
                },
                "unit": node_info.get('memory_usage', {}).get('unit', 'percent')
            },
            "memory_total": {
                "bytes": round_float(node_info.get('memory_total', {}).get('bytes')),
                "gb": round_float(node_info.get('memory_total', {}).get('gb'))
            }
        }
        
        result["nodes"][node_name] = node_stats
    
    return result

def extract_all_cluster_usage_info(data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract both cluster-wide statistics and individual node statistics.
    
    Args:
        data: Either a JSON string or dictionary containing cluster usage data
        
    Returns:
        Dictionary containing both cluster statistics and node-level statistics
    """
    
    cluster_stats = extract_cluster_statistics(data)
    node_stats = extract_node_statistics(data)
    
    return {
        "cluster_statistics": cluster_stats["cluster_statistics"],
        "baseline_comparison": cluster_stats["baseline_comparison"],
        "node_statistics": node_stats["nodes"]
    }

# Enhanced test function
def test_all_extractors():
    """Test all extractor functions with sample data"""
    
    # Sample data (simplified version of your actual data)
    sample_data = {
        "cluster_statistics": {
            "cpu_usage": {
                "min": 1.0561403508804745,
                "max": 6.2771929825019,
                "mean": 2.6924967644523123,
                "count": 9760
            },
            "memory_usage": {
                "min": 7.277930990202619,
                "max": 11.66060932412144,
                "mean": 9.449749953264506,
                "count": 305
            },
            "total_nodes": 5,
            "query_duration_hours": 1.0
        },
        "baseline_comparison": {
            "cpu": {
                "current_mean": 2.69,
                "current_max": 6.28,
                "baseline_mean": 45.0,
                "baseline_max": 80.0,
                "status": "normal",
                "message": "Values within normal range (mean: 2.7%, max: 6.3%)"
            },
            "memory": {
                "current_mean": 9.45,
                "current_max": 11.66,
                "baseline_mean": 50.0,
                "baseline_max": 85.0,
                "status": "normal",
                "message": "Values within normal range (mean: 9.4%, max: 11.7%)"
            }
        },
        "nodes": {
            "openshift-qe-018.lab.eng.rdu2.redhat.com": {
                "cpu_usage": {
                    "statistics": {
                        "min": 1.940350877182373,
                        "max": 3.5438596491146446,
                        "mean": 2.564279551334959,
                        "count": 61
                    },
                    "unit": "percent"
                },
                "memory_usage": {
                    "statistics": {
                        "min": 7.277930990202619,
                        "max": 7.4633469322673225,
                        "mean": 7.3892457144332,
                        "count": 61
                    },
                    "unit": "percent"
                },
                "memory_total": {
                    "bytes": 134606315520.0,
                    "gb": 125.36
                }
            },
            "openshift-qe-017.lab.eng.rdu2.redhat.com": {
                "cpu_usage": {
                    "statistics": {
                        "min": 2.2912280701852694,
                        "max": 3.540350877201992,
                        "mean": 2.7926373310311816,
                        "count": 61
                    },
                    "unit": "percent"
                },
                "memory_usage": {
                    "statistics": {
                        "min": 9.14867944903419,
                        "max": 9.265005245737179,
                        "mean": 9.205777166110526,
                        "count": 61
                    },
                    "unit": "percent"
                },
                "memory_total": {
                    "bytes": 134606331904.0,
                    "gb": 125.36
                }
            }
        }
    }
    
    print("Testing cluster statistics extraction:")
    try:
        cluster_result = extract_cluster_statistics(sample_data)
        print("✓ Cluster extraction successful!")
        print(json.dumps(cluster_result, indent=2))
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"✗ Cluster test failed: {e}")
    
    print("Testing node statistics extraction:")
    try:
        node_result = extract_node_statistics(sample_data)
        print("✓ Node extraction successful!")
        print(json.dumps(node_result, indent=2))
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"✗ Node test failed: {e}")
    
    print("Testing combined extraction:")
    try:
        combined_result = extract_all_cluster_usage_info(sample_data)
        print("✓ Combined extraction successful!")
        print(json.dumps(combined_result, indent=2))
    except Exception as e:
        print(f"✗ Combined test failed: {e}")

if __name__ == "__main__":
    # Run enhanced tests
    test_all_extractors()