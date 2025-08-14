import json
from typing import List, Dict, Any, Optional

def extract_node_info(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract node information from OpenShift cluster node data file.
    
    Args:
        file_path (str): Path to the JSON file containing node information
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing node information
        Each dictionary contains: name, role, instance_type, and capacity details
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read().strip()
            
        # Handle case where JSON might be missing opening/closing braces
        if not content.startswith('{'):
            content = '{' + content
        if not content.endswith('}'):
            content = content + '}'
            
        data = json.loads(content)
        
        extracted_nodes = []
        
        # Extract nodes from both master and worker roles
        nodes_by_role = data.get('nodes_by_role', {})
        
        for role, role_data in nodes_by_role.items():
            nodes = role_data.get('nodes', [])
            
            for node in nodes:
                node_info = {
                    'name': node.get('name'),
                    'role': node.get('role'),
                    'instance_type': node.get('instance_type'),
                    'capacity': {
                        'cpu_cores': node.get('capacity', {}).get('cpu_cores'),
                        'memory_bytes': node.get('capacity', {}).get('memory_bytes'),
                        'memory_gb': node.get('capacity', {}).get('memory_gb')
                    }
                }
                extracted_nodes.append(node_info)
        
        return extracted_nodes
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        print(f"Content preview: {content[:200]}...")
        return []
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")
        return []

def extract_node_info_from_json_data(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract node information from already loaded JSON data.
    
    Args:
        json_data (Dict[str, Any]): Dictionary containing the node data
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing node information
    """
    try:
        extracted_nodes = []
        
        # Extract nodes from both master and worker roles
        nodes_by_role = json_data.get('nodes_by_role', {})
        
        for role, role_data in nodes_by_role.items():
            nodes = role_data.get('nodes', [])
            
            for node in nodes:
                node_info = {
                    'name': node.get('name'),
                    'role': node.get('role'),
                    'instance_type': node.get('instance_type'),
                    'capacity': {
                        'cpu_cores': node.get('capacity', {}).get('cpu_cores'),
                        'memory_bytes': node.get('capacity', {}).get('memory_bytes'),
                        'memory_gb': node.get('capacity', {}).get('memory_gb')
                    }
                }
                extracted_nodes.append(node_info)
        
        return extracted_nodes
        
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")
        return []

def print_node_summary(nodes: List[Dict[str, Any]]) -> None:
    """
    Print a formatted summary of the extracted node information.
    
    Args:
        nodes (List[Dict[str, Any]]): List of node information dictionaries
    """
    if not nodes:
        print("No nodes found.")
        return
    
    print(f"\n{'='*80}")
    print(f"OpenShift Cluster Node Summary")
    print(f"{'='*80}")
    print(f"Total Nodes: {len(nodes)}")
    
    # Group by role
    master_nodes = [n for n in nodes if n['role'] == 'master']
    worker_nodes = [n for n in nodes if n['role'] == 'worker']
    
    print(f"Master Nodes: {len(master_nodes)}")
    print(f"Worker Nodes: {len(worker_nodes)}")
    print(f"{'='*80}")
    
    for node in nodes:
        print(f"\nNode: {node['name']}")
        print(f"  Role: {node['role']}")
        print(f"  Instance Type: {node['instance_type']}")
        print(f"  CPU Cores: {node['capacity']['cpu_cores']}")
        print(f"  Memory (GB): {node['capacity']['memory_gb']:.2f}")
        print(f"  Memory (Bytes): {node['capacity']['memory_bytes']}")

def filter_nodes_by_role(nodes: List[Dict[str, Any]], role: str) -> List[Dict[str, Any]]:
    """
    Filter nodes by their role (master or worker).
    
    Args:
        nodes (List[Dict[str, Any]]): List of node information dictionaries
        role (str): Role to filter by ('master' or 'worker')
        
    Returns:
        List[Dict[str, Any]]: Filtered list of nodes
    """
    return [node for node in nodes if node['role'].lower() == role.lower()]

def extract_node_info_as_json(file_path: str, indent: int = 2) -> str:
    """
    Extract node information and return as JSON string.
    
    Args:
        file_path (str): Path to the JSON file containing node information
        indent (int): Number of spaces for JSON indentation (default: 2)
        
    Returns:
        str: JSON string containing the extracted node information
    """
    nodes = extract_node_info(file_path)
    
    if not nodes:
        return json.dumps({"error": "No nodes found or error occurred"}, indent=indent)
    
    # Create a structured output
    output = {
        "timestamp": "2025-08-17T10:39:49.119007Z",  # You can update this as needed
        "total_nodes": len(nodes),
        "master_nodes": len([n for n in nodes if n['role'] == 'master']),
        "worker_nodes": len([n for n in nodes if n['role'] == 'worker']),
        "nodes": nodes
    }
    
    return json.dumps(output, indent=indent)

def extract_node_info_from_json_data_as_json(json_data: Dict[str, Any], indent: int = 2) -> str:
    """
    Extract node information from JSON data and return as JSON string.
    
    Args:
        json_data (Dict[str, Any]): Dictionary containing the node data
        indent (int): Number of spaces for JSON indentation (default: 2)
        
    Returns:
        str: JSON string containing the extracted node information
    """
    nodes = extract_node_info_from_json_data(json_data)
    
    if not nodes:
        return json.dumps({"error": "No nodes found or error occurred"}, indent=indent)
    
    # Create a structured output
    output = {
        # "timestamp": json_data.get("timestamp", "unknown"),
        # "total_nodes": len(nodes),
        # "master_nodes": len([n for n in nodes if n['role'] == 'master']),
        # "worker_nodes": len([n for n in nodes if n['role'] == 'worker']),
        "nodes": nodes
    }
    
    return json.dumps(output, indent=indent)

def save_extracted_nodes_to_json(file_path: str, output_file: str) -> bool:
    """
    Extract node information and save to a JSON file.
    
    Args:
        file_path (str): Path to the input JSON file
        output_file (str): Path to save the extracted JSON data
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        json_output = extract_node_info_as_json(file_path)
        
        with open(output_file, 'w') as f:
            f.write(json_output)
        
        print(f"Extracted node information saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error saving to file: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Example 1: Extract from file and get JSON output
    print("=== JSON Output ===")
    json_result = extract_node_info_as_json('/Users/liqcui/Downloads/node-info.txt')
    print(json_result)
    
    # Example 2: Save to file
    print("\n=== Saving to file ===")
    success = save_extracted_nodes_to_json('/Users/liqcui/Downloads/node-info.txt', 'extracted_nodes.json')
    
    # Example 3: Get only specific role as JSON
    print("\n=== Worker Nodes Only (JSON) ===")
    nodes = extract_node_info('/Users/liqcui/Downloads/node-info.txt')
    if nodes:
        worker_nodes = filter_nodes_by_role(nodes, 'worker')
        worker_json = {
            "role_filter": "worker",
            "total_worker_nodes": len(worker_nodes),
            "nodes": worker_nodes
        }
        print(json.dumps(worker_json, indent=2))
    
    # Example 4: Get only specific role as JSON
    print("\n=== Master Nodes Only (JSON) ===")
    if nodes:
        master_nodes = filter_nodes_by_role(nodes, 'master')
        master_json = {
            "role_filter": "master", 
            "total_master_nodes": len(master_nodes),
            "nodes": master_nodes
        }
        print(json.dumps(master_json, indent=2))