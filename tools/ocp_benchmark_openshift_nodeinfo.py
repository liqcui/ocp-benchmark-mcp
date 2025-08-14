"""OpenShift node information retrieval."""
import json
import logging
import subprocess
from typing import Dict, Any, List, Optional
from kubernetes import client
from kubernetes.client.rest import ApiException
from ocauth.ocp_benchmark_auth import ocp_auth


logger = logging.getLogger(__name__)


class NodeInfoCollector:
    """Collects OpenShift node information."""
    
    def __init__(self):
        self.k8s_client = ocp_auth.k8s_client
    
    def parse_resource_quantity(self, quantity: str) -> float:
        """Parse Kubernetes resource quantity (e.g., '4Gi', '1000m')."""
        if not quantity:
            return 0.0
        
        # Handle CPU units
        if quantity.endswith('m'):
            return float(quantity[:-1]) / 1000.0
        elif quantity.endswith('n'):
            return float(quantity[:-1]) / 1_000_000_000.0
        
        # Handle memory units
        elif quantity.endswith('Ki'):
            return float(quantity[:-2]) * 1024
        elif quantity.endswith('Mi'):
            return float(quantity[:-2]) * 1024 * 1024
        elif quantity.endswith('Gi'):
            return float(quantity[:-2]) * 1024 * 1024 * 1024
        elif quantity.endswith('Ti'):
            return float(quantity[:-2]) * 1024 * 1024 * 1024 * 1024
        
        # Handle plain numbers
        try:
            return float(quantity)
        except ValueError:
            return 0.0
    
    def get_node_role(self, node: Any) -> str:
        """Determine node role from labels."""
        labels = node.metadata.labels or {}
        
        if 'node-role.kubernetes.io/master' in labels or 'node-role.kubernetes.io/control-plane' in labels:
            return 'master'
        elif 'node-role.kubernetes.io/infra' in labels:
            return 'infra'
        elif 'node-role.kubernetes.io/worker' in labels:
            return 'worker'
        else:
            # Default to worker if no specific role found
            return 'worker'
    
    def get_instance_type(self, node: Any) -> Optional[str]:
        """Get instance type from node labels or annotations."""
        labels = node.metadata.labels or {}
        annotations = node.metadata.annotations or {}
        
        # Common cloud provider instance type labels
        instance_type_labels = [
            'node.kubernetes.io/instance-type',
            'beta.kubernetes.io/instance-type',
            'topology.kubernetes.io/instance-type'
        ]
        
        for label in instance_type_labels:
            if label in labels:
                return labels[label]
        
        # Try annotations
        if 'machine.openshift.io/instance-type' in annotations:
            return annotations['machine.openshift.io/instance-type']
        
        return 'unknown'
    
    def get_nodes_info_api(self) -> List[Dict[str, Any]]:
        """Get node information using Kubernetes API."""
        nodes_info = []
        
        try:
            v1 = client.CoreV1Api(self.k8s_client)
            nodes = v1.list_node()
            
            for node in nodes.items:
                # Parse capacity
                capacity = node.status.capacity or {}
                cpu_cores = self.parse_resource_quantity(capacity.get('cpu', '0'))
                memory_bytes = self.parse_resource_quantity(capacity.get('memory', '0'))
                
                # Parse allocatable
                allocatable = node.status.allocatable or {}
                allocatable_cpu = self.parse_resource_quantity(allocatable.get('cpu', '0'))
                allocatable_memory = self.parse_resource_quantity(allocatable.get('memory', '0'))
                
                # Get node conditions
                conditions = {}
                for condition in node.status.conditions or []:
                    conditions[condition.type] = condition.status == 'True'
                
                # Get addresses
                addresses = {}
                for addr in node.status.addresses or []:
                    addresses[addr.type] = addr.address
                
                node_info = {
                    'name': node.metadata.name,
                    'role': self.get_node_role(node),
                    'instance_type': self.get_instance_type(node),
                    'capacity': {
                        'cpu_cores': round(cpu_cores, 6),
                        'memory_bytes': memory_bytes,
                        'memory_gb': round(memory_bytes / (1024 ** 3), 6) if memory_bytes > 0 else 0
                    },
                    'allocatable': {
                        'cpu_cores': round(allocatable_cpu, 6),
                        'memory_bytes': allocatable_memory,
                        'memory_gb': round(allocatable_memory / (1024 ** 3), 6) if allocatable_memory > 0 else 0
                    },
                    'status': {
                        'ready': conditions.get('Ready', False),
                        'schedulable': not (node.spec.unschedulable or False),
                        'conditions': conditions
                    },
                    'addresses': addresses,
                    'labels': node.metadata.labels or {},
                    'annotations': dict(list((node.metadata.annotations or {}).items())[:5]),  # First 5 annotations
                    'kernel_version': node.status.node_info.kernel_version if node.status.node_info else 'unknown',
                    'os_image': node.status.node_info.os_image if node.status.node_info else 'unknown',
                    'kubelet_version': node.status.node_info.kubelet_version if node.status.node_info else 'unknown'
                }
                
                nodes_info.append(node_info)
        
        except Exception as e:
            logger.error(f"Failed to get nodes info with API: {e}")
            raise
        
        return nodes_info
    
    def get_nodes_info_oc(self) -> Optional[List[Dict[str, Any]]]:
        """Get node information using oc command."""
        try:
            result = subprocess.run([
                'oc', 'get', 'nodes', '-o', 'json'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                nodes_data = json.loads(result.stdout)
                nodes_info = []
                
                for node in nodes_data.get('items', []):
                    # Parse capacity
                    capacity = node.get('status', {}).get('capacity', {})
                    cpu_cores = self.parse_resource_quantity(capacity.get('cpu', '0'))
                    memory_bytes = self.parse_resource_quantity(capacity.get('memory', '0'))
                    
                    # Parse allocatable
                    allocatable = node.get('status', {}).get('allocatable', {})
                    allocatable_cpu = self.parse_resource_quantity(allocatable.get('cpu', '0'))
                    allocatable_memory = self.parse_resource_quantity(allocatable.get('memory', '0'))
                    
                    # Get node conditions
                    conditions = {}
                    for condition in node.get('status', {}).get('conditions', []):
                        conditions[condition['type']] = condition['status'] == 'True'
                    
                    # Get addresses
                    addresses = {}
                    for addr in node.get('status', {}).get('addresses', []):
                        addresses[addr['type']] = addr['address']
                    
                    # Determine role from labels
                    labels = node.get('metadata', {}).get('labels', {})
                    role = 'worker'  # default
                    if 'node-role.kubernetes.io/master' in labels or 'node-role.kubernetes.io/control-plane' in labels:
                        role = 'master'
                    elif 'node-role.kubernetes.io/infra' in labels:
                        role = 'infra'
                    elif 'node-role.kubernetes.io/worker' in labels:
                        role = 'worker'
                    
                    # Get instance type
                    instance_type = 'unknown'
                    instance_type_labels = [
                        'node.kubernetes.io/instance-type',
                        'beta.kubernetes.io/instance-type',
                        'topology.kubernetes.io/instance-type'
                    ]
                    for label in instance_type_labels:
                        if label in labels:
                            instance_type = labels[label]
                            break
                    
                    node_info = {
                        'name': node.get('metadata', {}).get('name'),
                        'role': role,
                        'instance_type': instance_type,
                        'capacity': {
                            'cpu_cores': round(cpu_cores, 6),
                            'memory_bytes': memory_bytes,
                            'memory_gb': round(memory_bytes / (1024 ** 3), 6) if memory_bytes > 0 else 0
                        },
                        'allocatable': {
                            'cpu_cores': round(allocatable_cpu, 6),
                            'memory_bytes': allocatable_memory,
                            'memory_gb': round(allocatable_memory / (1024 ** 3), 6) if allocatable_memory > 0 else 0
                        },
                        'status': {
                            'ready': conditions.get('Ready', False),
                            'schedulable': not node.get('spec', {}).get('unschedulable', False),
                            'conditions': conditions
                        },
                        'addresses': addresses,
                        'labels': labels,
                        'annotations': dict(list((node.get('metadata', {}).get('annotations', {})).items())[:5]),
                        'kernel_version': node.get('status', {}).get('nodeInfo', {}).get('kernelVersion', 'unknown'),
                        'os_image': node.get('status', {}).get('nodeInfo', {}).get('osImage', 'unknown'),
                        'kubelet_version': node.get('status', {}).get('nodeInfo', {}).get('kubeletVersion', 'unknown')
                    }
                    
                    nodes_info.append(node_info)
                
                return nodes_info
        
        except Exception as e:
            logger.warning(f"Failed to get nodes info with oc: {e}")
        
        return None
    
    def group_nodes_by_role(self, nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Group nodes by role and calculate statistics."""
        grouped = {
            'master': [],
            'worker': [],
            'infra': []
        }
        
        for node in nodes:
            role = node['role']
            if role in grouped:
                grouped[role].append(node)
        
        # Calculate statistics for each group
        result = {}
        for role, role_nodes in grouped.items():
            if not role_nodes:
                continue
            
            total_cpu = sum(node['capacity']['cpu_cores'] for node in role_nodes)
            total_memory_gb = sum(node['capacity']['memory_gb'] for node in role_nodes)
            ready_count = sum(1 for node in role_nodes if node['status']['ready'])
            
            result[role] = {
                'count': len(role_nodes),
                'ready_count': ready_count,
                'total_cpu_cores': round(total_cpu, 6),
                'total_memory_gb': round(total_memory_gb, 6),
                'average_cpu_cores': round(total_cpu / len(role_nodes), 6) if role_nodes else 0,
                'average_memory_gb': round(total_memory_gb / len(role_nodes), 6) if role_nodes else 0,
                'instance_types': list(set(node['instance_type'] for node in role_nodes)),
                'nodes': role_nodes
            }
        
        return result
    
    def collect_nodes_info(self) -> Dict[str, Any]:
        """Collect complete node information."""
        # Try API first, then oc command
        nodes = self.get_nodes_info_api()
        if not nodes:
            nodes = self.get_nodes_info_oc()
        
        if not nodes:
            raise RuntimeError("Could not retrieve node information")
        
        grouped_nodes = self.group_nodes_by_role(nodes)
        
        # Calculate cluster totals
        total_nodes = sum(group['count'] for group in grouped_nodes.values())
        total_cpu = sum(group['total_cpu_cores'] for group in grouped_nodes.values())
        total_memory = sum(group['total_memory_gb'] for group in grouped_nodes.values())
        total_ready = sum(group['ready_count'] for group in grouped_nodes.values())
        
        return {
            'timestamp': None,  # Will be set by caller
            'cluster_summary': {
                'total_nodes': total_nodes,
                'ready_nodes': total_ready,
                'total_cpu_cores': round(total_cpu, 6),
                'total_memory_gb': round(total_memory, 6),
                'node_roles_distribution': {
                    role: group['count'] for role, group in grouped_nodes.items()
                }
            },
            'nodes_by_role': grouped_nodes,
            'collection_method': 'api' if nodes else 'oc_command'
        }

def get_nodes_info() -> str:
    """Get nodes information and return as JSON string."""
    collector = NodeInfoCollector()
    info = collector.collect_nodes_info()
    
    # Add timestamp
    from datetime import datetime
    info['timestamp'] = datetime.utcnow().isoformat() + 'Z'
    
    return json.dumps(info, indent=2)

# Global cluster info collector instance
node_info_collector = NodeInfoCollector()
