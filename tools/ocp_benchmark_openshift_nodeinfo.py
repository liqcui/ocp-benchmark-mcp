#!/usr/bin/env python3
"""OpenShift Node Information Module"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from kubernetes import client
from ocauth.ocp_benchmark_auth import auth

logger = logging.getLogger(__name__)

class NodeInfoCollector:
    """Collect OpenShift node information"""
    
    def __init__(self):
        self.kube_client = auth.get_kube_client()
    
    def _get_node_role(self, node: Any) -> List[str]:
        """Determine node role from labels"""
        labels = node.metadata.labels or {}
        roles = []
        
        # Check for node role labels
        for label, value in labels.items():
            if label.startswith('node-role.kubernetes.io/'):
                role = label.replace('node-role.kubernetes.io/', '')
                if role:
                    roles.append(role)
        
        # Fallback to older label format
        if not roles:
            if labels.get('kubernetes.io/role') == 'master':
                roles.append('master')
            elif labels.get('kubernetes.io/role') == 'node':
                roles.append('worker')
        
        return roles if roles else ['worker']  # Default to worker
    
    def _extract_instance_type(self, node: Any) -> str:
        """Extract instance type from node labels"""
        labels = node.metadata.labels or {}
        
        # Common cloud provider instance type labels
        instance_type_labels = [
            'node.kubernetes.io/instance-type',
            'beta.kubernetes.io/instance-type',
            'kubernetes.io/instance-type',
            'failure-domain.beta.kubernetes.io/instance-type'
        ]
        
        for label in instance_type_labels:
            if label in labels:
                return labels[label]
        
        return 'unknown'
    
    def _extract_node_resources(self, node: Any) -> Dict[str, Any]:
        """Extract CPU and memory resources from node"""
        allocatable = node.status.allocatable or {}
        capacity = node.status.capacity or {}
        
        def parse_memory(memory_str: str) -> int:
            """Parse memory string to bytes"""
            if not memory_str:
                return 0
            
            # Handle Ki, Mi, Gi suffixes
            memory_str = memory_str.replace('i', '')  # Remove 'i' from Ki, Mi, Gi
            
            multipliers = {
                'K': 1024,
                'M': 1024 * 1024,
                'G': 1024 * 1024 * 1024,
                'T': 1024 * 1024 * 1024 * 1024
            }
            
            for suffix, multiplier in multipliers.items():
                if memory_str.endswith(suffix):
                    value = float(memory_str[:-1])
                    return int(value * multiplier)
            
            # If no suffix, assume bytes
            try:
                return int(memory_str)
            except ValueError:
                return 0
        
        def parse_cpu(cpu_str: str) -> float:
            """Parse CPU string to cores"""
            if not cpu_str:
                return 0.0
            
            if cpu_str.endswith('m'):
                # Millicores
                return float(cpu_str[:-1]) / 1000
            else:
                return float(cpu_str)
        
        return {
            'cpu': {
                'allocatable_cores': parse_cpu(allocatable.get('cpu', '0')),
                'capacity_cores': parse_cpu(capacity.get('cpu', '0'))
            },
            'memory': {
                'allocatable_bytes': parse_memory(allocatable.get('memory', '0')),
                'capacity_bytes': parse_memory(capacity.get('memory', '0')),
                'allocatable_gb': parse_memory(allocatable.get('memory', '0')) / (1024**3),
                'capacity_gb': parse_memory(capacity.get('memory', '0')) / (1024**3)
            },
            'storage': {
                'ephemeral_storage': allocatable.get('ephemeral-storage', '0')
            },
            'pods': {
                'allocatable': int(allocatable.get('pods', '0')),
                'capacity': int(capacity.get('pods', '0'))
            }
        }
    
    def _get_node_conditions(self, node: Any) -> Dict[str, Any]:
        """Get node conditions"""
        conditions = {}
        
        for condition in node.status.conditions or []:
            conditions[condition.type] = {
                'status': condition.status,
                'reason': condition.reason,
                'message': condition.message,
                'last_transition_time': condition.last_transition_time.isoformat() if condition.last_transition_time else None
            }
        
        return conditions
    
    async def get_all_nodes_info(self) -> Dict[str, Any]:
        """Get information for all nodes"""
        try:
            v1 = client.CoreV1Api(self.kube_client)
            nodes = v1.list_node()
            
            node_info = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'total_nodes': len(nodes.items),
                'summary': {
                    'master': {'count': 0, 'total_cpu': 0, 'total_memory_gb': 0},
                    'worker': {'count': 0, 'total_cpu': 0, 'total_memory_gb': 0},
                    'infra': {'count': 0, 'total_cpu': 0, 'total_memory_gb': 0}
                },
                'nodes': []
            }
            
            for node in nodes.items:
                roles = self._get_node_role(node)
                instance_type = self._extract_instance_type(node)
                resources = self._extract_node_resources(node)
                conditions = self._get_node_conditions(node)
                
                node_data = {
                    'name': node.metadata.name,
                    'roles': roles,
                    'instance_type': instance_type,
                    'resources': resources,
                    'conditions': conditions,
                    'ready': conditions.get('Ready', {}).get('status') == 'True',
                    'created': node.metadata.creation_timestamp.isoformat() if node.metadata.creation_timestamp else None,
                    'labels': node.metadata.labels or {},
                    'taints': []
                }
                
                # Extract taints
                if node.spec.taints:
                    for taint in node.spec.taints:
                        node_data['taints'].append({
                            'key': taint.key,
                            'value': taint.value,
                            'effect': taint.effect
                        })
                
                node_info['nodes'].append(node_data)
                
                # Update summary based on primary role
                primary_role = 'worker'  # Default
                if 'master' in roles or 'control-plane' in roles:
                    primary_role = 'master'
                elif 'infra' in roles:
                    primary_role = 'infra'
                
                node_info['summary'][primary_role]['count'] += 1
                node_info['summary'][primary_role]['total_cpu'] += resources['cpu']['capacity_cores']
                node_info['summary'][primary_role]['total_memory_gb'] += resources['memory']['capacity_gb']
            
            return node_info
            
        except Exception as e:
            logger.error(f"Error getting nodes info: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
    
    async def get_nodes_by_role(self, role: str) -> List[Dict[str, Any]]:
        """Get nodes filtered by role"""
        all_nodes = await self.get_all_nodes_info()
        
        if 'error' in all_nodes:
            return []
        
        filtered_nodes = []
        for node in all_nodes['nodes']:
            if role in node['roles']:
                filtered_nodes.append(node)
        
        return filtered_nodes
    
    async def get_node_summary(self) -> Dict[str, Any]:
        """Get summarized node information"""
        all_nodes = await self.get_all_nodes_info()
        
        if 'error' in all_nodes:
            return all_nodes
        
        summary = {
            'timestamp': all_nodes['timestamp'],
            'total_nodes': all_nodes['total_nodes'],
            'roles_summary': all_nodes['summary'],
            'instance_types': {},
            'ready_nodes': 0,
            'not_ready_nodes': 0
        }
        
        # Count instance types and ready status
        for node in all_nodes['nodes']:
            instance_type = node['instance_type']
            if instance_type not in summary['instance_types']:
                summary['instance_types'][instance_type] = {
                    'count': 0,
                    'total_cpu': 0,
                    'total_memory_gb': 0
                }
            
            summary['instance_types'][instance_type]['count'] += 1
            summary['instance_types'][instance_type]['total_cpu'] += node['resources']['cpu']['capacity_cores']
            summary['instance_types'][instance_type]['total_memory_gb'] += node['resources']['memory']['capacity_gb']
            
            if node['ready']:
                summary['ready_nodes'] += 1
            else:
                summary['not_ready_nodes'] += 1
        
        return summary

# Global node info collector instance
node_info_collector = NodeInfoCollector()

async def get_nodes_info_json() -> str:
    """Get nodes information as JSON string"""
    info = await node_info_collector.get_all_nodes_info()
    return json.dumps(info, indent=2)

async def get_nodes_summary_json() -> str:
    """Get nodes summary as JSON string"""
    summary = await node_info_collector.get_node_summary()
    return json.dumps(summary, indent=2)