import json
from typing import Union, List, Dict, Any

def json_to_table(data: Union[str, List[Dict], Dict], format_type: str = "text") -> str:
    """
    Convert JSON data to table format.
    
    Args:
        data: JSON string, list of dictionaries, or single dictionary
        format_type: "text", "html", or "markdown"
    
    Returns:
        Formatted table as string
    """
    # Parse JSON string if needed
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string")
    
    # Convert single dict to list
    if isinstance(data, dict):
        data = [data]
    
    if not data:
        return "No data available"
    
    # Get all unique keys
    all_keys = set()
    for item in data:
        if isinstance(item, dict):
            all_keys.update(item.keys())
    
    headers = list(all_keys)
    
    if format_type == "html":
        return _to_html_table(data, headers)
    elif format_type == "markdown":
        return _to_markdown_table(data, headers)
    else:
        return _to_text_table(data, headers)

def _to_text_table(data: List[Dict], headers: List[str]) -> str:
    """Convert to plain text table format"""
    if not data:
        return "No data"
    
    # Calculate column widths
    col_widths = {}
    for header in headers:
        col_widths[header] = len(str(header))
        for row in data:
            value = str(row.get(header, ""))
            col_widths[header] = max(col_widths[header], len(value))
    
    # Create table
    lines = []
    
    # Header row
    header_row = "| " + " | ".join(header.ljust(col_widths[header]) for header in headers) + " |"
    lines.append(header_row)
    
    # Separator row
    separator = "| " + " | ".join("-" * col_widths[header] for header in headers) + " |"
    lines.append(separator)
    
    # Data rows
    for row in data:
        data_row = "| " + " | ".join(str(row.get(header, "")).ljust(col_widths[header]) for header in headers) + " |"
        lines.append(data_row)
    
    return "\n".join(lines)

def _to_html_table(data: List[Dict], headers: List[str]) -> str:
    """Convert to HTML table format"""
    html = '<table border="1" style="border-collapse: collapse; width: 100%;">\n'
    
    # Header
    html += '  <thead>\n    <tr style="background-color: #f0f0f0;">\n'
    for header in headers:
        html += f'      <th style="padding: 8px; text-align: left;">{_escape_html(str(header))}</th>\n'
    html += '    </tr>\n  </thead>\n'
    
    # Body
    html += '  <tbody>\n'
    for i, row in enumerate(data):
        bg_color = "#f9f9f9" if i % 2 == 1 else "white"
        html += f'    <tr style="background-color: {bg_color};">\n'
        for header in headers:
            value = str(row.get(header, ""))
            html += f'      <td style="padding: 8px;">{_escape_html(value)}</td>\n'
        html += '    </tr>\n'
    html += '  </tbody>\n</table>'
    
    return html

def _to_markdown_table(data: List[Dict], headers: List[str]) -> str:
    """Convert to Markdown table format"""
    if not data:
        return "No data"
    
    lines = []
    
    # Header row
    header_row = "| " + " | ".join(headers) + " |"
    lines.append(header_row)
    
    # Separator row
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    lines.append(separator)
    
    # Data rows
    for row in data:
        data_row = "| " + " | ".join(str(row.get(header, "")) for header in headers) + " |"
        lines.append(data_row)
    
    return "\n".join(lines)

def _escape_html(text: str) -> str:
    """Escape HTML special characters"""
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))

# # Example usage and testing
# if __name__ == "__main__":
#     # Sample data
#     sample_data = [
#         {"name": "John Doe", "age": 30, "city": "New York", "salary": 75000},
#         {"name": "Jane Smith", "age": 25, "city": "Los Angeles", "salary": 80000},
#         {"name": "Bob Johnson", "age": 35, "city": "Chicago", "email": "bob@email.com"}
#     ]
    
#     print("=== TEXT TABLE ===")
#     print(json_to_table(sample_data, "text"))
    
#     print("\n=== HTML TABLE ===")
#     print(json_to_table(sample_data, "html"))
    
#     print("\n=== MARKDOWN TABLE ===")
#     print(json_to_table(sample_data, "markdown"))
    
#     print("\n=== FROM JSON STRING ===")
#     json_string = '[{"product": "Laptop", "price": 999}, {"product": "Phone", "price": 699}]'
#     print(json_to_table(json_string, "text"))
    
#     print("\n=== SINGLE OBJECT ===")
#     single_obj = {"name": "Alice", "role": "Developer", "experience": 5}
#     print(json_to_table(single_obj, "text"))