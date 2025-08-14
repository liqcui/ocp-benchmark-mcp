#!/usr/bin/env python3
"""OpenShift Cluster Information Module"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional
from kubernetes import client
from ocauth.ocp_benchmark_auth import auth

logger = logging.getLogger(__name__)

class ClusterInfoCollector:
    """Collect OpenShift cluster information"""
    
    def __init__(self):
        self.kube_client = auth.get_kube_client()
    
    async def get_cluster_version(self) -> Optional[Dict[str, Any]]:
        """Get cluster version information"""
        try:
            custom_api = client.CustomObjectsApi(self.kube_client)
            
            # Get ClusterVersion resource
            cluster_versions = custom_api.list_cluster_custom_object(
                group="config.openshift.io",
                version="v1",
                plural="clusterversions"
            )
            
            if cluster_versions.get('items'):
                cv = cluster_versions['items'][0]
                
                version_info = {
                    'name': cv['metadata']['name'],
                    'version': cv['status'].get('desired', {}).get('version', 'unknown'),
                    'image': cv['status'].get('desired', {}).get('image', 'unknown'),
                    'channel': cv['spec'].get('channel', 'unknown'),
                    'cluster_id': cv['spec'].get('clusterID', 'unknown'),
                    'conditions': []
                }
                
                # Extract conditions
                for condition in cv['status'].get('conditions', []):
                    version_info['conditions'].append({
                        'type': condition.get('type'),
                        'status': condition.get('status'),
                        'reason': condition.get('reason'),
                        'message': condition.get('message', '')
                    })
                
                return version_info
            
            logger.warning("No cluster version information found")
            return None
            
        except Exception as e:
            logger.error(f"Error getting cluster version: {e}")
            return None
    
    async def get_cluster_name(self) -> Optional[str]:
        """Get cluster name from DNS or infrastructure"""
        try:
            custom_api = client.CustomObjectsApi(self.kube_client)
            
            # Try to get from DNS config
            try:
                dns_configs = custom_api.list_cluster_custom_object(
                    group="config.openshift.io",
                    version="v1",
                    plural="dnses"
                )
                
                if dns_configs.get('items'):
                    dns_config = dns_configs['items'][0]
                    cluster_domain = dns_config['spec'].get('baseDomain')
                    if cluster_domain:
                        # Extract cluster name from domain (typically first part)
                        cluster_name = cluster_domain.split('.')[0] if '.' in cluster_domain else cluster_domain
                        return cluster_name
            except Exception as dns_error:
                logger.warning(f"Could not get cluster name from DNS: {dns_error}")
            
            # Fallback: try from infrastructure
            try:
                infrastructures = custom_api.list_cluster_custom_object(
                    group="config.openshift.io",
                    version="v1",
                    plural="infrastructures"
                )
                
                if infrastructures.get('items'):
                    infra = infrastructures['items'][0]
                    return infra['status'].get('infrastructureName', 'unknown')
            except Exception as infra_error:
                logger.warning(f"Could not get cluster name from infrastructure: {infra_error}")
            
            return "unknown"
            
        except Exception as e:
            logger.error(f"Error getting cluster name: {e}")
            return None
    
    async def get_infrastructure_info(self) -> Optional[Dict[str, Any]]:
        """Get infrastructure information"""
        try:
            custom_api = client.CustomObjectsApi(self.kube_client)
            
            infrastructures = custom_api.list_cluster_custom_object(
                group="config.openshift.io",
                version="v1",
                plural="infrastructures"
            )
            
            if infrastructures.get('items'):
                infra = infrastructures['items'][0]
                
                infra_info = {
                    'name': infra['metadata']['name'],
                    'infrastructure_name': infra['status'].get('infrastructureName', 'unknown'),
                    'platform': infra['status'].get('platform', 'unknown'),
                    'platform_status': infra['status'].get('platformStatus', {}),
                    'api_server_internal_url': infra['status'].get('apiServerInternalURL', 'unknown'),
                    'api_server_url': infra['status'].get('apiServerURL', 'unknown'),
                    'etcd_discovery_domain': infra['status'].get('etcdDiscoveryDomain', 'unknown')
                }
                
                return infra_info
            
            logger.warning("No infrastructure information found")
            return None
            
        except Exception as e:
            logger.error(f"Error getting infrastructure info: {e}")
            return None
    
    async def get_cluster_operators_status(self) -> Optional[Dict[str, Any]]:
        """Get cluster operators status"""
        try:
            custom_api = client.CustomObjectsApi(self.kube_client)
            
            operators = custom_api.list_cluster_custom_object(
                group="config.openshift.io",
                version="v1",
                plural="clusteroperators"
            )
            
            operator_status = {
                'total_operators': len(operators.get('items', [])),
                'available': 0,
                'progressing': 0,
                'degraded': 0,
                'operators': []
            }
            
            for operator in operators.get('items', []):
                op_info = {
                    'name': operator['metadata']['name'],
                    'version': 'unknown',
                    'available': False,
                    'progressing': False,
                    'degraded': False
                }
                
                # Check conditions
                for condition in operator['status'].get('conditions', []):
                    if condition['type'] == 'Available':
                        op_info['available'] = condition['status'] == 'True'
                        if op_info['available']:
                            operator_status['available'] += 1
                    elif condition['type'] == 'Progressing':
                        op_info['progressing'] = condition['status'] == 'True'
                        if op_info['progressing']:
                            operator_status['progressing'] += 1
                    elif condition['type'] == 'Degraded':
                        op_info['degraded'] = condition['status'] == 'True'
                        if op_info['degraded']:
                            operator_status['degraded'] += 1
                
                # Get version if available
                for version in operator['status'].get('versions', []):
                    if version.get('name') == 'operator':
                        op_info['version'] = version.get('version', 'unknown')
                        break
                
                operator_status['operators'].append(op_info)
            
            return operator_status
            
        except Exception as e:
            logger.error(f"Error getting cluster operators status: {e}")
            return None
    
    async def get_complete_cluster_info(self) -> Dict[str, Any]:
        """Get complete cluster information"""
        try:
            cluster_info = {
                'timestamp': None,
                'cluster_version': None,
                'cluster_name': None,
                'infrastructure': None,
                'operators_status': None
            }
            
            # Set timestamp
            from datetime import datetime
            cluster_info['timestamp'] = datetime.utcnow().isoformat() + 'Z'
            
            # Collect all information concurrently
            tasks = [
                self.get_cluster_version(),
                self.get_cluster_name(),
                self.get_infrastructure_info(),
                self.get_cluster_operators_status()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            cluster_info['cluster_version'] = results[0] if not isinstance(results[0], Exception) else None
            cluster_info['cluster_name'] = results[1] if not isinstance(results[1], Exception) else None
            cluster_info['infrastructure'] = results[2] if not isinstance(results[2], Exception) else None
            cluster_info['operators_status'] = results[3] if not isinstance(results[3], Exception) else None
            
            return cluster_info
            
        except Exception as e:
            logger.error(f"Error collecting complete cluster info: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }

# Global cluster info collector instance
cluster_info_collector = ClusterInfoCollector()

async def get_cluster_info_json() -> str:
    """Get cluster information as JSON string"""
    info = await cluster_info_collector.get_complete_cluster_info()
    return json.dumps(info, indent=2)