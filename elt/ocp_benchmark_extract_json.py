import json
from typing import Union, Dict, Any, List, Optional

class JSONExtractor:
    """A common JSON data extractor with flexible field extraction capabilities."""
    
    def __init__(self, json_data: Union[str, Dict]):
        """
        Initialize the extractor with JSON data.
        
        Args:
            json_data: JSON string or dictionary
        """
        if isinstance(json_data, str):
            try:
                self.data = json.loads(json_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e}")
        else:
            self.data = json_data
    
    def extract_custom_fields(self, field_paths: List[str]) -> Dict[str, Any]:
        """
        Extract custom fields using dot notation paths.
        
        Args:
            field_paths: List of dot notation paths (e.g., ["data.cluster_name", "data.version"])
        
        Returns:
            Dictionary with extracted fields
        """
        result = {}
        
        for path in field_paths:
            try:
                value = self._get_nested_value(path)
                # Use the last part of the path as the key name
                key_name = path.split('.')[-1]
                result[key_name] = value
            except KeyError:
                result[path] = None
            except Exception as e:
                result[path] = f"Error: {str(e)}"
        
        return result
    
    def extract_all_by_key(self, key_name: str) -> List[Any]:
        """
        Find all values for a specific key throughout the JSON structure.
        
        Args:
            key_name: The key to search for
        
        Returns:
            List of all values found for the key
        """
        results = []
        
        def _search_recursive(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == key_name:
                        results.append(v)
                    _search_recursive(v)
            elif isinstance(obj, list):
                for item in obj:
                    _search_recursive(item)
        
        _search_recursive(self.data)
        return results
    
    def extract_section(self, section_path: str) -> Any:
        """
        Extract an entire section from the JSON.
        
        Args:
            section_path: Dot notation path to the section
        
        Returns:
            The section data
        """
        return self._get_nested_value(section_path)
    
    def flatten_data(self, separator: str = '.', max_depth: int = 10) -> Dict[str, Any]:
        """
        Flatten the entire JSON into a single-level dictionary.
        
        Args:
            separator: Separator for nested keys
            max_depth: Maximum depth to flatten
        
        Returns:
            Flattened dictionary
        """
        def _flatten(obj, parent_key='', depth=0):
            items = []
            
            if depth >= max_depth:
                items.append((parent_key, str(obj)))
                return dict(items)
            
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{parent_key}{separator}{k}" if parent_key else k
                    if isinstance(v, (dict, list)) and depth < max_depth:
                        items.extend(_flatten(v, new_key, depth + 1).items())
                    else:
                        items.append((new_key, v))
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    new_key = f"{parent_key}{separator}{i}" if parent_key else str(i)
                    if isinstance(v, (dict, list)) and depth < max_depth:
                        items.extend(_flatten(v, new_key, depth + 1).items())
                    else:
                        items.append((new_key, v))
            else:
                items.append((parent_key, obj))
            
            return dict(items)
        
        return _flatten(self.data)
    
    def _get_nested_value(self, path: str) -> Any:
        """
        Get value from nested JSON using dot notation.
        
        Args:
            path: Dot notation path
        
        Returns:
            Value at the specified path
        """
        keys = path.split('.')
        current = self.data
        
        for key in keys:
            if isinstance(current, dict):
                if key in current:
                    current = current[key]
                else:
                    raise KeyError(f"Key '{key}' not found in path '{path}'")
            elif isinstance(current, list):
                try:
                    index = int(key)
                    if 0 <= index < len(current):
                        current = current[index]
                    else:
                        raise KeyError(f"Index {index} out of range in path '{path}'")
                except ValueError:
                    raise KeyError(f"Invalid list index '{key}' in path '{path}'")
            else:
                raise KeyError(f"Cannot navigate through non-dict/list at '{key}' in path '{path}'")
        
        return current
    
    def get_all_keys(self, prefix: str = '') -> List[str]:
        """
        Get all available keys in the JSON structure.
        
        Args:
            prefix: Prefix for nested keys
        
        Returns:
            List of all keys in dot notation
        """
        keys = []
        
        def _collect_keys(obj, current_prefix=''):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_prefix = f"{current_prefix}.{k}" if current_prefix else k
                    keys.append(new_prefix)
                    if isinstance(v, (dict, list)):
                        _collect_keys(v, new_prefix)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    new_prefix = f"{current_prefix}.{i}" if current_prefix else str(i)
                    if isinstance(v, (dict, list)):
                        _collect_keys(v, new_prefix)
        
        _collect_keys(self.data, prefix)
        return sorted(keys)

# Convenience function for quick extraction
def extract_json_data(json_data: Union[str, Dict], 
                     custom_fields: Optional[List[str]] = None,
                     section: Optional[str] = None,
                     flatten: bool = False) -> Dict[str, Any]:
    """
    Common function to extract data from JSON with various options.
    
    Args:
        json_data: JSON string or dictionary
        custom_fields: List of custom field paths to extract
        section: Specific section to extract
        flatten: Whether to flatten the entire JSON
    
    Returns:
        Extracted data
    """
    extractor = JSONExtractor(json_data)
    
    if custom_fields:
        return extractor.extract_custom_fields(custom_fields)
    elif section:
        return extractor.extract_section(section)
    elif flatten:
        return extractor.flatten_data()
    else:
        return extractor.data

# Example usage with your OpenShift JSON
if __name__ == "__main__":
    # Your OpenShift JSON data
    openshift_json = '''
    {
      "success": true,
      "data": {
        "timestamp": "2025-08-16T14:41:58.462785Z",
        "cluster_name": "perfscale-qe-oc",
        "version_info": {
          "version": "4.19.1",
          "channel": "stable-4.19",
          "image": "xxxxxxx",
          "update_available": true
        },
        "infrastructure_info": {
          "infrastructure_name": "perfscale-qe-oc-xxxx",
          "platform": "BareMetal",
          "platform_status": {
            "baremetal": {
              "apiServerInternalIP": "1.2.3.4",
              "apiServerInternalIPs": ["2.3.4.5"],
              "ingressIP": "3.4.5.6",
              "ingressIPs": ["4.5.6.7"],
              "loadBalancer": {
                "type": "OpenShiftManagedDefault"
              },
              "machineNetworks": ["5.6.7.8/19"]
            },
            "type": "BareMetal"
          },
          "api_server_url": "xxxxxxxxx",
          "etcd_discovery_domain": ""
        },
        "collection_method": "mixed",
        "summary": {
          "cluster_name": "perfscale-qe-oc",
          "version": "4.19.1",
          "platform": "BareMetal",
          "api_url": "yyyyyyy"
        }
      },
      "error": null,
      "timestamp": "2025-08-16T14:41:58.490523+00:00"
    }
    '''
    
    # Create extractor instance
    extractor = JSONExtractor(openshift_json)
    
    print("=== EXTRACT CUSTOM FIELDS ===")
    custom_data = extractor.extract_custom_fields([
        "data.cluster_name",
        "data.version_info.version",
        "data.infrastructure_info.platform",
        "data.infrastructure_info.platform_status.baremetal.apiServerInternalIP",
        "data.infrastructure_info.platform_status.baremetal.ingressIP",
        "data.summary.api_url"
    ])
    print(custom_data)
    print("#*"*50)
    
    for key, value in custom_data.items():
        print(f"{key}: {value}")
    
    print("\n=== EXTRACT SUMMARY SECTION ===")
    summary = extractor.extract_section("data.summary")
    print(json.dumps(summary, indent=2))
    
    print("\n=== EXTRACT VERSION INFO ===")
    version_info = extractor.extract_section("data.version_info")
    print(json.dumps(version_info, indent=2))
    
    print("\n=== FIND ALL 'cluster_name' VALUES ===")
    cluster_names = extractor.extract_all_by_key("cluster_name")
    print(cluster_names)
    
    print("\n=== AVAILABLE KEYS (first 20) ===")
    all_keys = extractor.get_all_keys()
    for key in all_keys[:20]:
        print(key)
    if len(all_keys) > 20:
        print(f"... and {len(all_keys) - 20} more keys")
    
    print("\n=== CONVENIENCE FUNCTION USAGE ===")
    # Using the convenience function
    quick_extract = extract_json_data(
        openshift_json, 
        custom_fields=["data.cluster_name", "data.version_info.version", "success"]
    )
    print(json.dumps(quick_extract, indent=2))