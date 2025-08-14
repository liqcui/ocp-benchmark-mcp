"""OpenShift cluster information retrieval."""
import json
import logging
import subprocess
from typing import Dict, Any, Optional
from kubernetes import client
from kubernetes.client.rest import ApiException
from ocauth.ocp_benchmark_auth import ocp_auth


logger = logging.getLogger(__name__)


class ClusterInfoCollector:
    """Collects OpenShift cluster information."""
    
    def __init__(self):
        self.k8s_client = ocp_auth.k8s_client
    
    def get_cluster_version_oc(self) -> Optional[Dict[str, Any]]:
        """Get cluster version using oc command."""
        try:
            result = subprocess.run([
                'oc', 'get', 'clusterversion', 'version', '-o', 'json'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                version_data = json.loads(result.stdout)
                return {
                    'version': version_data.get('status', {}).get('desired', {}).get('version'),
                    'channel': version_data.get('spec', {}).get('channel'),
                    'image': version_data.get('status', {}).get('desired', {}).get('image'),
                    'update_available': len(version_data.get('status', {}).get('availableUpdates', [])) > 0
                }
        except Exception as e:
            logger.warning(f"Failed to get cluster version with oc: {e}")
        # Fallback: use 'oc version -o json' when ClusterVersion object is not accessible
        try:
            result = subprocess.run(['oc', 'version', '-o', 'json'], capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                # Common fields seen in oc version -o json
                openshift_version = data.get('openshiftVersion')
                server_version = data.get('serverVersion', {})
                server_git_version = server_version.get('gitVersion')
                computed_version = openshift_version or server_git_version
                if computed_version:
                    return {
                        'version': computed_version,
                        'channel': None,
                        'image': None,
                        'update_available': False
                    }
        except Exception as e:
            logger.warning(f"Fallback 'oc version' failed: {e}")

        return None
    
    def get_cluster_version_api(self) -> Optional[Dict[str, Any]]:
        """Get cluster version using Kubernetes API."""
        try:
            custom_api = client.CustomObjectsApi(self.k8s_client)
            version_obj = custom_api.get_cluster_custom_object(
                group="config.openshift.io",
                version="v1",
                plural="clusterversions",
                name="version"
            )
            
            return {
                'version': version_obj.get('status', {}).get('desired', {}).get('version'),
                'channel': version_obj.get('spec', {}).get('channel'),
                'image': version_obj.get('status', {}).get('desired', {}).get('image'),
                'update_available': len(version_obj.get('status', {}).get('availableUpdates', [])) > 0
            }
        except Exception as e:
            logger.warning(f"Failed to get cluster version with API: {e}")
        
        return None
    
    def get_infrastructure_info_oc(self) -> Optional[Dict[str, Any]]:
        """Get infrastructure information using oc command."""
        try:
            result = subprocess.run([
                'oc', 'get', 'infrastructure', 'cluster', '-o', 'json'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                infra_data = json.loads(result.stdout)
                status = infra_data.get('status', {})
                return {
                    'infrastructure_name': status.get('infrastructureName'),
                    'platform': status.get('platform'),
                    'platform_status': status.get('platformStatus', {}),
                    'api_server_url': status.get('apiServerURL'),
                    'etcd_discovery_domain': status.get('etcdDiscoveryDomain')
                }
        except Exception as e:
            logger.warning(f"Failed to get infrastructure info with oc: {e}")
        
        return None
    
    def get_infrastructure_info_api(self) -> Optional[Dict[str, Any]]:
        """Get infrastructure information using Kubernetes API."""
        try:
            custom_api = client.CustomObjectsApi(self.k8s_client)
            infra_obj = custom_api.get_cluster_custom_object(
                group="config.openshift.io",
                version="v1",
                plural="infrastructures",
                name="cluster"
            )
            
            status = infra_obj.get('status', {})
            return {
                'infrastructure_name': status.get('infrastructureName'),
                'platform': status.get('platform'),
                'platform_status': status.get('platformStatus', {}),
                'api_server_url': status.get('apiServerURL'),
                'etcd_discovery_domain': status.get('etcdDiscoveryDomain')
            }
        except Exception as e:
            logger.warning(f"Failed to get infrastructure info with API: {e}")
        
        return None
    
    def get_cluster_name_oc(self) -> Optional[str]:
        """Get cluster name using oc command."""
        try:
            # Try to get cluster name from DNS
            result = subprocess.run([
                'oc', 'get', 'dns', 'cluster', '-o', 'jsonpath={.spec.baseDomain}'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout.strip():
                base_domain = result.stdout.strip()
                # Extract cluster name from base domain
                parts = base_domain.split('.')
                if len(parts) >= 2:
                    return parts[0]  # Usually cluster name is the first part
            
            # Fallback: try infrastructure name
            result = subprocess.run([
                'oc', 'get', 'infrastructure', 'cluster', '-o', 'jsonpath={.status.infrastructureName}'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception as e:
            logger.warning(f"Failed to get cluster name with oc: {e}")
        
        return None
    
    def get_cluster_name_api(self) -> Optional[str]:
        """Get cluster name using Kubernetes API."""
        try:
            custom_api = client.CustomObjectsApi(self.k8s_client)
            
            # Try to get from DNS config
            dns_obj = custom_api.get_cluster_custom_object(
                group="config.openshift.io",
                version="v1",
                plural="dnses",
                name="cluster"
            )
            
            base_domain = dns_obj.get('spec', {}).get('baseDomain')
            if base_domain:
                parts = base_domain.split('.')
                if len(parts) >= 2:
                    return parts[0]
            
            # Fallback: try infrastructure name
            infra_obj = custom_api.get_cluster_custom_object(
                group="config.openshift.io",
                version="v1",
                plural="infrastructures",
                name="cluster"
            )
            
            infra_name = infra_obj.get('status', {}).get('infrastructureName')
            if infra_name:
                return infra_name
        except Exception as e:
            logger.warning(f"Failed to get cluster name with API: {e}")
        
        return None
    
    def collect_cluster_info(self) -> Dict[str, Any]:
        """Collect complete cluster information."""
        cluster_info = {
            'timestamp': None,  # Will be set by caller
            'cluster_name': None,
            'version_info': None,
            'infrastructure_info': None,
            'collection_method': 'mixed'
        }
        
        # Try to get cluster version
        version_info = self.get_cluster_version_oc()
        if not version_info:
            version_info = self.get_cluster_version_api()
        cluster_info['version_info'] = version_info
        
        # Try to get infrastructure info
        infra_info = self.get_infrastructure_info_oc()
        if not infra_info:
            infra_info = self.get_infrastructure_info_api()
        cluster_info['infrastructure_info'] = infra_info
        
        # Try to get cluster name
        cluster_name = self.get_cluster_name_oc()
        if not cluster_name:
            cluster_name = self.get_cluster_name_api()
        cluster_info['cluster_name'] = cluster_name
        
        # Add summary
        cluster_info['summary'] = {
            'cluster_name': cluster_name,
            'version': version_info.get('version') if version_info else 'unknown',
            'platform': infra_info.get('platform') if infra_info else 'unknown',
            'api_url': infra_info.get('api_server_url') if infra_info else 'unknown'
        }
        
        return cluster_info


def get_cluster_info() -> str:
    """Get cluster information and return as JSON string."""
    collector = ClusterInfoCollector()
    info = collector.collect_cluster_info()
    
    # Add timestamp
    from datetime import datetime
    info['timestamp'] = datetime.utcnow().isoformat() + 'Z'
    
    return json.dumps(info, indent=2)