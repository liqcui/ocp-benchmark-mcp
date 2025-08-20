import json
from typing import Any, Dict, Union

def auto_reduce_json_levels(data: Union[Dict, Any], separator: str = '.') -> Dict[str, Any]:
    """
    Automatically detects the depth of JSON and reduces it to 1-5 levels based on original depth.
    
    Reduction logic:
    - 1-5 levels: No change (already optimal)
    - 6-7 levels: Reduce to 5 levels
    - 8-9 levels: Reduce to 4 levels
    - 10-11 levels: Reduce to 3 levels
    - 12-13 levels: Reduce to 2 levels  
    - 14+ levels: Reduce to 1 level (completely flat)
    
    Args:
        data: The input JSON data (dict or any JSON-serializable object)
        separator: String used to join keys when flattening (default: '.')
    
    Returns:
        Dict with optimally reduced nesting levels (1-5 max)
    """
    
    def get_depth(obj, current_depth=0):
        """Calculate the maximum depth of nested dictionaries"""
        if not isinstance(obj, dict):
            return current_depth
        if not obj:  # Empty dict
            return current_depth
        return max(get_depth(value, current_depth + 1) for value in obj.values())
    
    def determine_target_levels(original_depth):
        """Automatically determine target levels based on original depth"""
        if original_depth <= 5:
            return original_depth  # No change needed
        elif original_depth <= 7:
            return 5  # Reduce to 5 levels
        elif original_depth <= 9:
            return 4  # Reduce to 4 levels
        elif original_depth <= 11:
            return 3  # Reduce to 3 levels
        elif original_depth <= 13:
            return 2  # Reduce to 2 levels
        else:
            return 1  # Completely flat for very deep structures
    
    def flatten_recursive(obj, parent_key='', current_level=1, target_levels=5):
        """Recursively flatten the object based on current level and target"""
        items = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}{separator}{key}" if parent_key else key
                
                # If we're at target_levels and the value is still a dict, flatten it completely
                if current_level >= target_levels and isinstance(value, dict):
                    # Flatten remaining levels completely
                    flattened = flatten_completely(value, new_key)
                    items.extend(flattened.items())
                elif isinstance(value, dict):
                    # Continue recursing if we haven't hit target levels
                    items.extend(flatten_recursive(value, new_key, current_level + 1, target_levels).items())
                else:
                    # Non-dict value, add as is
                    items.append((new_key, value))
        else:
            # Non-dict object
            items.append((parent_key, obj))
            
        return dict(items)
    
    def flatten_completely(obj, parent_key=''):
        """Completely flatten a nested structure (for levels beyond target)"""
        items = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}{separator}{key}" if parent_key else key
                
                if isinstance(value, dict):
                    items.extend(flatten_completely(value, new_key).items())
                elif isinstance(value, list):
                    # Handle lists by adding index
                    for i, item in enumerate(value):
                        list_key = f"{new_key}[{i}]"
                        if isinstance(item, dict):
                            items.extend(flatten_completely(item, list_key).items())
                        else:
                            items.append((list_key, item))
                else:
                    items.append((new_key, value))
        elif isinstance(obj, list):
            # Handle lists at the top level
            for i, item in enumerate(obj):
                list_key = f"{parent_key}[{i}]" if parent_key else f"[{i}]"
                if isinstance(item, dict):
                    items.extend(flatten_completely(item, list_key).items())
                else:
                    items.append((list_key, item))
        else:
            items.append((parent_key, obj))
            
        return dict(items)
    
    # Check if input is not a dict
    if not isinstance(data, dict):
        return {"value": data}
    
    # Get current depth
    original_depth = get_depth(data)
    
    # Determine target levels based on original depth
    target_levels = determine_target_levels(original_depth)
    
    # If no reduction needed, return original
    if target_levels == original_depth:
        return data
    
    # Flatten the structure to target levels
    return flatten_recursive(data, target_levels=target_levels)


def convert_json_auto_reduce(json_input: Union[str, Dict]) -> str:
    """
    Main function that automatically detects JSON depth and reduces to optimal 1-4 levels.
    
    Args:
        json_input: JSON string or dictionary
    
    Returns:
        JSON string with automatically optimized nesting levels (1-5 max)
    """
    
    # Parse JSON if it's a string
    if isinstance(json_input, str):
        try:
            data = json.loads(json_input)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")
    else:
        data = json_input
    
    # Auto-reduce the data
    reduced_data = auto_reduce_json_levels(data)
    
    # Return as JSON string
    return json.dumps(reduced_data, indent=2)


def get_json_depth_info(json_input: Union[str, Dict]) -> Dict[str, Any]:
    """
    Utility function to get depth information about JSON structure.
    
    Args:
        json_input: JSON string or dictionary
    
    Returns:
        Dictionary with depth analysis information
    """
    
    # Parse JSON if it's a string
    if isinstance(json_input, str):
        try:
            data = json.loads(json_input)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")
    else:
        data = json_input
    
    def get_depth(obj, current_depth=0):
        if not isinstance(obj, dict):
            return current_depth
        if not obj:
            return current_depth
        return max(get_depth(value, current_depth + 1) for value in obj.values())
    
    def determine_target_levels(original_depth):
        if original_depth <= 5:
            return original_depth
        elif original_depth <= 7:
            return 5
        elif original_depth <= 9:
            return 4
        elif original_depth <= 11:
            return 3
        elif original_depth <= 13:
            return 2
        else:
            return 1
    
    if not isinstance(data, dict):
        return {
            "original_depth": 0,
            "target_depth": 1,
            "reduction_needed": True,
            "reduction_reason": "Non-dict input"
        }
    
    original_depth = get_depth(data)
    target_depth = determine_target_levels(original_depth)
    
    return {
        "original_depth": original_depth,
        "target_depth": target_depth,
        "reduction_needed": target_depth != original_depth,
        "reduction_reason": f"Optimizing {original_depth} levels to {target_depth} levels"
    }


# Example usage and testing
if __name__ == "__main__":
    # Test case 1: Very deep JSON (18 levels) - should reduce to 1 level
    very_deep_json = {
        "a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": {"j": {"k": {"l": {"m": {"n": {"o": {"p": {"q": {"r": "extremely_deep_value"}}}}}}}}}}}}}}}}}
    }
    
    print("Test 1: Very deep JSON (18 levels)")
    info1 = get_json_depth_info(very_deep_json)
    print(f"Depth Info: {info1}")
    print("Original JSON (18 levels):")
    print(json.dumps(very_deep_json, indent=2))
    print("\nAuto-reduced JSON:")
    result1 = convert_json_auto_reduce(very_deep_json)
    print(result1)
    
    print("\n" + "="*70 + "\n")
    
    # Test case 2: JSON with 8 levels - should reduce to 4 levels
    eight_level_json = {
        "company": {
            "departments": {
                "engineering": {
                    "teams": {
                        "backend": {
                            "projects": {
                                "api": {
                                    "features": {
                                        "authentication": "oauth2",
                                        "logging": "enabled",
                                        "monitoring": "prometheus"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "metadata": {
            "version": "2.1",
            "created": "2024-01-15"
        }
    }
    
    print("Test 2: JSON with 8 levels")
    info2 = get_json_depth_info(eight_level_json)
    print(f"Depth Info: {info2}")
    print("Original JSON (8 levels):")
    print(json.dumps(eight_level_json, indent=2))
    print("\nAuto-reduced JSON:")
    result2 = convert_json_auto_reduce(eight_level_json)
    print(result2)
    
    print("\n" + "="*70 + "\n")
    
    # Test case 3: JSON with exactly 5 levels - should remain unchanged
    five_level_json = {
        "level1": {
            "level2": {
                "level3": {
                    "level4": {
                        "level5": {
                            "data": "max_allowed_depth",
                            "status": "active",
                            "count": 42
                        }
                    }
                }
            }
        }
    }
    
    print("Test 3: JSON with exactly 5 levels")
    info3 = get_json_depth_info(five_level_json)
    print(f"Depth Info: {info3}")
    print("Original JSON (5 levels):")
    print(json.dumps(five_level_json, indent=2))
    print("\nAuto-reduced JSON (should be unchanged):")
    result3 = convert_json_auto_reduce(five_level_json)
    print(result3)
    
    print("\n" + "="*70 + "\n")
    
    # Test case 4: JSON with 6 levels - should reduce to 5 levels
    six_level_json = {
        "app": {
            "modules": {
                "user": {
                    "components": {
                        "profile": {
                            "sections": {
                                "personal": "John Doe",
                                "contact": "john@example.com",
                                "preferences": {
                                    "theme": "dark",
                                    "language": "en",
                                    "notifications": True
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    print("Test 4: JSON with 6 levels")
    info4 = get_json_depth_info(six_level_json)
    print(f"Depth Info: {info4}")
    print("Original JSON (6 levels):")
    print(json.dumps(six_level_json, indent=2))
    print("\nAuto-reduced JSON:")
    result4 = convert_json_auto_reduce(six_level_json)
    print(result4)
    
    print("\n" + "="*70 + "\n")
    
    # Test case 5: JSON with 10 levels - should reduce to 3 levels
    ten_level_json = {
        "system": {
            "database": {
                "clusters": {
                    "primary": {
                        "nodes": {
                            "master": {
                                "config": {
                                    "storage": {
                                        "settings": {
                                            "cache": {
                                                "size": "1GB",
                                                "timeout": 300
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    print("Test 5: JSON with 10 levels")
    info5 = get_json_depth_info(ten_level_json)
    print(f"Depth Info: {info5}")
    print("Original JSON (10 levels):")
    print(json.dumps(ten_level_json, indent=2))
    print("\nAuto-reduced JSON:")
    result5 = convert_json_auto_reduce(ten_level_json)
    print(result5)
    
    print("\n" + "="*70 + "\n")
    
    # Test case 6: JSON with 12 levels - should reduce to 2 levels
    twelve_level_json = {
        "enterprise": {
            "organizations": {
                "divisions": {
                    "departments": {
                        "teams": {
                            "squads": {
                                "members": {
                                    "employees": {
                                        "profiles": {
                                            "personal": {
                                                "contact": {
                                                    "details": {
                                                        "phone": "555-0123",
                                                        "email": "employee@company.com"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    print("Test 6: JSON with 12 levels")
    info6 = get_json_depth_info(twelve_level_json)
    print(f"Depth Info: {info6}")
    print("Original JSON (12 levels):")
    print(json.dumps(twelve_level_json, indent=2))
    print("\nAuto-reduced JSON:")
    result6 = convert_json_auto_reduce(twelve_level_json)
    print(result6)
    
    print("\n" + "="*70 + "\n")
    
    # Test case 7: Show all reduction levels
    print("Summary of reduction logic (1-5 levels):")
    test_depths = [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18]
    for depth in test_depths:
        # Create a simple nested structure for testing
        current = {"value": f"depth_{depth}"}
        for i in range(depth - 1):
            current = {f"level_{depth-i}": current}
        
        info = get_json_depth_info(current)
        print(f"{depth} levels → {info['target_depth']} levels ({'No change' if not info['reduction_needed'] else 'Reduced'})")