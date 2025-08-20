#!/usr/bin/env python3
"""
Network Performance Analysis Data Extractor

This script extracts performance analysis information and network interface details
from network monitoring JSON files.
"""

import json
from typing import Dict, List, Optional, Any, Union
import os


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


def _load_data(input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Helper function to load data from various input types.
    
    Args:
        input_data: Can be a file path (str), JSON string (str), or dictionary
        
    Returns:
        Dict containing the loaded data
        
    Raises:
        ValueError: If input_data cannot be processed
        FileNotFoundError: If file path doesn't exist
        json.JSONDecodeError: If JSON string is invalid
    """
    if isinstance(input_data, dict):
        # Input is already a dictionary
        return input_data
    
    elif isinstance(input_data, str):
        # First try to parse as JSON string
        if input_data.strip().startswith(('{', '[')):
            try:
                data = json.loads(input_data)
                if isinstance(data, dict):
                    return data
                else:
                    raise ValueError(f"JSON string must parse to a dictionary, got: {type(data)}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string: {str(e)}")
        
        # If not JSON, check if it's a file path
        elif os.path.isfile(input_data):
            try:
                with open(input_data, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    if isinstance(data, dict):
                        return data
                    else:
                        raise ValueError(f"File content must be a JSON object (dictionary), got: {type(data)}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in file '{input_data}': {str(e)}")
        
        # If neither JSON nor existing file, try one more time as JSON (in case it doesn't start with { or [)
        else:
            try:
                data = json.loads(input_data)
                if isinstance(data, dict):
                    return data
                else:
                    raise ValueError(f"JSON string must parse to a dictionary, got: {type(data)}")
            except json.JSONDecodeError:
                raise ValueError(f"Input is neither a valid file path nor a valid JSON string. "
                               f"Input preview: {input_data[:100]}{'...' if len(input_data) > 100 else ''}")
    
    else:
        raise ValueError(f"Input must be a file path (str), JSON string (str), or dictionary (dict). Got: {type(input_data)}")


def extract_network_performance_analysis(input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract performance analysis information from network data.
    
    Args:
        input_data: Can be:
            - File path (str): Path to the JSON file
            - JSON string (str): JSON string containing the data
            - Dictionary (dict): Already parsed data dictionary
        
    Returns:
        Dict containing performance analysis with rounded floating point numbers
    """
    try:
        data = _load_data(input_data)
        
        # Ensure data is a dictionary
        if not isinstance(data, dict):
            raise ValueError(f"Loaded data must be a dictionary, got: {type(data)}")
        
        # Extract performance analysis section
        performance_analysis = data.get("performance_analysis", {})
        
        # Create a copy to avoid modifying original data
        result = {
            "overall_status": performance_analysis.get("overall_status"),
            "alerts": [],
            "summary": performance_analysis.get("summary", {})
        }
        
        # Process alerts and round floating point numbers to 2 decimal places
        for alert in performance_analysis.get("alerts", []):
            rounded_alert = {
                "type": alert.get("type"),
                "interface": alert.get("interface"),
                "current_mbps": round(float(alert.get("current_mbps", 0)), 2),
                "baseline_mbps": round(float(alert.get("baseline_mbps", 0)), 2),
                "severity": alert.get("severity")
            }
            result["alerts"].append(rounded_alert)
        
        return _round_decimals_in_obj(result, 2)
        
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Error: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error processing data: {e}")
        return {}


def extract_interface_details(input_data: Union[str, Dict[str, Any]], node_name: str, interface_name: str) -> Dict[str, Any]:
    """
    Extract detailed network interface information for a specific node and interface.
    
    Args:
        input_data: Can be:
            - File path (str): Path to the JSON file
            - JSON string (str): JSON string containing the data
            - Dictionary (dict): Already parsed data dictionary
        node_name (str): Name of the node (e.g., 'openshift-qe-021.lab.eng.rdu2.redhat.com')
        interface_name (str): Name of the interface (e.g., 'eno1')
        
    Returns:
        Dict containing interface details with rounded statistics
    """
    try:
        data = _load_data(input_data)
        
        # Ensure data is a dictionary
        if not isinstance(data, dict):
            raise ValueError(f"Loaded data must be a dictionary, got: {type(data)}")
        
        # Navigate to the specific node and interface
        nodes = data.get("nodes", {})
        if node_name not in nodes:
            print(f"Error: Node '{node_name}' not found.")
            return {}
        
        interfaces = nodes[node_name].get("interfaces", {})
        if interface_name not in interfaces:
            print(f"Error: Interface '{interface_name}' not found on node '{node_name}'.")
            return {}
        
        interface_data = interfaces[interface_name]
        
        # Helper function to round statistics
        def round_statistics(stats_dict: Dict[str, Any]) -> Dict[str, Any]:
            rounded_stats = {}
            for key, value in stats_dict.items():
                if key == "statistics":
                    rounded_stats[key] = {
                        stat_key: round(float(stat_value), 2) if isinstance(stat_value, (int, float)) else stat_value
                        for stat_key, stat_value in value.items()
                    }
                else:
                    rounded_stats[key] = value
            return rounded_stats
        
        # Process and round the interface data
        result = {}
        
        # Process rx_bytes, tx_bytes, rx_packets, tx_packets
        for metric in ['rx_bytes', 'tx_bytes', 'rx_packets', 'tx_packets']:
            if metric in interface_data:
                result[metric] = round_statistics(interface_data[metric])
        
        # Process errors section
        if 'errors' in interface_data:
            result['errors'] = {}
            for error_type in ['rx_errors', 'tx_errors']:
                if error_type in interface_data['errors']:
                    result['errors'][error_type] = round_statistics(interface_data['errors'][error_type])
        
        return _round_decimals_in_obj(result, 2)
        
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Error: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error processing data: {e}")
        return {}


def list_available_nodes_and_interfaces(input_data: Union[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    List all available nodes and their interfaces from network data.
    
    Args:
        input_data: Can be:
            - File path (str): Path to the JSON file
            - JSON string (str): JSON string containing the data
            - Dictionary (dict): Already parsed data dictionary
        
    Returns:
        Dict mapping node names to lists of interface names
    """
    try:
        data = _load_data(input_data)
        
        # Ensure data is a dictionary
        if not isinstance(data, dict):
            raise ValueError(f"Loaded data must be a dictionary, got: {type(data)}")
        
        nodes_interfaces = {}
        nodes = data.get("nodes", {})
        
        for node_name, node_data in nodes.items():
            interfaces = list(node_data.get("interfaces", {}).keys())
            nodes_interfaces[node_name] = interfaces
        
        return nodes_interfaces
        
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Error: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error processing data: {e}")
        return {}


def main():
    """
    Demonstration of the extraction functions with different input types.
    """
    # Example usage with different input types
    file_path = "network-full.json"  # Update this path as needed
    
    print("=== Demo with File Path ===")
    performance_data = extract_network_performance_analysis(file_path)
    if performance_data:
        print(f"Overall Status: {performance_data['overall_status']}")
        print(f"Number of alerts: {len(performance_data['alerts'])}")
    
    print("\n=== Demo with Dictionary ===")
    # Load data once and reuse as dictionary
    try:
        with open(file_path, 'r') as file:
            data_dict = json.load(file)
        
        # Use the dictionary directly
        performance_data = extract_network_performance_analysis(data_dict)
        nodes_interfaces = list_available_nodes_and_interfaces(data_dict)
        
        print(f"Performance status from dict: {performance_data.get('overall_status', 'N/A')}")
        print(f"Number of nodes from dict: {len(nodes_interfaces)}")
        
        # Example with interface extraction using dictionary
        if nodes_interfaces:
            first_node = list(nodes_interfaces.keys())[0]
            first_interface = nodes_interfaces[first_node][0] if nodes_interfaces[first_node] else None
            
            if first_interface:
                interface_details = extract_interface_details(data_dict, first_node, first_interface)
                if interface_details and 'rx_bytes' in interface_details:
                    rx_stats = interface_details['rx_bytes']['statistics']
                    print(f"Interface {first_interface} RX mean: {rx_stats.get('mean', 'N/A')} MB/s")
    
    except FileNotFoundError:
        print("File not found for dictionary demo")
    
    print("\n=== Demo with JSON String ===")
    # Example with JSON string (truncated for demo)
    json_string = '''
    {
        "performance_analysis": {
            "overall_status": "normal",
            "alerts": [
                {
                    "type": "low_rx_throughput",
                    "interface": "test-interface",
                    "current_mbps": 1.234567,
                    "baseline_mbps": 10.0,
                    "severity": "info"
                }
            ],
            "summary": {
                "total_nodes": 1,
                "total_interfaces": 5
            }
        },
        "nodes": {
            "test-node": {
                "interfaces": {
                    "eth0": {
                        "rx_bytes": {
                            "statistics": {
                                "min": 0.123456789,
                                "max": 1.987654321,
                                "mean": 1.111111111,
                                "count": 10
                            },
                            "unit": "MB/s"
                        }
                    }
                }
            }
        }
    }
    '''
    
    performance_from_json = extract_network_performance_analysis(json_string)
    nodes_from_json = list_available_nodes_and_interfaces(json_string)
    
    print(f"Performance status from JSON string: {performance_from_json.get('overall_status', 'N/A')}")
    print(f"Nodes from JSON string: {list(nodes_from_json.keys())}")
    
    # Test interface extraction with JSON string
    interface_from_json = extract_interface_details(json_string, "test-node", "eth0")
    if interface_from_json and 'rx_bytes' in interface_from_json:
        rx_stats = interface_from_json['rx_bytes']['statistics']
        print(f"Test interface RX mean (rounded): {rx_stats.get('mean', 'N/A')} MB/s")
    
    print("\n=== Usage Examples ===")
    print("# Using file path:")
    print('performance_data = extract_network_performance_analysis("network-full.json")')
    print("\n# Using dictionary:")
    print('with open("network-full.json") as f: data = json.load(f)')
    print('performance_data = extract_network_performance_analysis(data)')
    print("\n# Using JSON string:")
    print('json_str = \'{"performance_analysis": {...}}\'')
    print('performance_data = extract_network_performance_analysis(json_str)')


if __name__ == "__main__":
    main()