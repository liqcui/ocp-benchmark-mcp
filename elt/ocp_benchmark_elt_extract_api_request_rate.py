import json

def extract_request_rate_performance_analysis(data):
    """
    Extract performance analysis information from the JSON data.
    
    Args:
        data: Can be a dictionary (parsed JSON) or a JSON string
    
    Returns:
        JSON string with performance_analysis section and floating point numbers rounded to 6 decimal places.
    """
    try:
        # Handle both dict and JSON string inputs
        if isinstance(data, str):
            data = json.loads(data)
        elif not isinstance(data, dict):
            return json.dumps({"error": "Input must be a dictionary or JSON string"}, indent=2)
        
        # Extract performance analysis
        performance_analysis = data.get('performance_analysis', {})
        
        # Round floating point numbers to 6 decimal places if they exist
        if 'summary' in performance_analysis:
            summary = performance_analysis['summary']
            for key, value in summary.items():
                if isinstance(value, float):
                    summary[key] = round(value, 6)
        
        result = {
            "performance_analysis": performance_analysis
        }
        
        return json.dumps(result, indent=2)
    
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON format"}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"An error occurred: {str(e)}"}, indent=2)


def extract_active_request_rates(data):
    """
    Extract request rates where at least one of min/mean/max values is greater than 0.0.
    
    Args:
        data: Can be a dictionary (parsed JSON) or a JSON string
    
    Returns:
        JSON string with only the request rates that have activity.
    """
    try:
        # Handle both dict and JSON string inputs
        if isinstance(data, str):
            data = json.loads(data)
        elif not isinstance(data, dict):
            return json.dumps({"error": "Input must be a dictionary or JSON string"}, indent=2)
        
        # Extract request rates
        request_rates = data.get('request_rates', {})
        active_request_rates = {}
        
        for request_key, request_data in request_rates.items():
            rate_stats = request_data.get('rate_stats', {})
            
            # Check if any of min, mean, max is greater than 0.0
            min_val = rate_stats.get('min', 0.0)
            max_val = rate_stats.get('max', 0.0)
            mean_val = rate_stats.get('mean', 0.0)
            
            if min_val > 0.0 or max_val > 0.0 or mean_val > 0.0:
                # Round floating point numbers to 6 decimal places
                rounded_stats = {}
                for stat_key, stat_value in rate_stats.items():
                    if isinstance(stat_value, float):
                        rounded_stats[stat_key] = round(stat_value, 6)
                    else:
                        rounded_stats[stat_key] = stat_value
                
                active_request_rates[request_key] = {
                    "rate_stats": rounded_stats
                }
        
        result = {
            "request_rates": active_request_rates
        }
        
        return json.dumps(result, indent=2)
    
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON format"}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"An error occurred: {str(e)}"}, indent=2)


# Example usage
if __name__ == "__main__":
    # Example with dictionary input
    sample_data = {
        "performance_analysis": {
            "overall_status": "optimal",
            "alerts": [],
            "summary": {
                "total_operations": 0,
                "high_latency_operations": 0,
                "critical_latency_operations": 0,
                "etcd_operations": 0,
                "high_etcd_latency": 0
            }
        },
        "request_rates": {
            "WATCH:pods:0": {
                "rate_stats": {
                    "min": 1.5,
                    "max": 3.2,
                    "mean": 2.1,
                    "count": 10
                },
                "labels": {"resource": "pods"}
            },
            "GET:nodes:200": {
                "rate_stats": {
                    "min": 0.0,
                    "max": 0.0,
                    "mean": 0.0,
                    "count": 0
                },
                "labels": {"resource": "nodes"}
            }
        }
    }
    
    print("=== Performance Analysis ===")
    performance_result = extract_request_rate_performance_analysis(sample_data)
    print(performance_result)
    
    print("\n=== Active Request Rates ===")
    active_rates_result = extract_active_request_rates(sample_data)
    print(active_rates_result)
    
    # Example with JSON string input
    json_string = json.dumps(sample_data)
    print("\n=== Using JSON String Input ===")
    performance_from_string = extract_request_rate_performance_analysis(json_string)
    print(performance_from_string)