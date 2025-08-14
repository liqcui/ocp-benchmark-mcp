"""OpenShift authentication and Prometheus connection management."""
import os
import subprocess
import logging
import requests
from typing import Optional, Tuple
from urllib.parse import urljoin
from kubernetes import client, config
from kubernetes.client.rest import ApiException

#Will fix warning instead of just disable it
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class OCPAuth:
    """Handles OpenShift authentication and Prometheus URL discovery."""
    
    def __init__(self):
        self.kubeconfig_path = os.getenv('KUBECONFIG')
        self.k8s_client = None
        self.prometheus_url = None
        self.prometheus_token = None
        self._setup_kubernetes_client()
    
    def _setup_kubernetes_client(self):
        """Initialize Kubernetes client using KUBECONFIG."""
        try:
            if self.kubeconfig_path and os.path.exists(self.kubeconfig_path):
                config.load_kube_config(config_file=self.kubeconfig_path)
            else:
                # Try in-cluster config if no kubeconfig
                config.load_incluster_config()
            
            self.k8s_client = client.ApiClient()
            logger.info("Kubernetes client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise

    def get_prometheus_url(self) -> Optional[str]:
        """Automatically discover Prometheus URL from OpenShift monitoring"""
        try:
            custom_api = client.CustomObjectsApi(self.k8s_client)
            routes = custom_api.list_namespaced_custom_object(
                group="route.openshift.io",
                version="v1",
                namespace="openshift-monitoring",
                plural="routes"
            )
            
            for route in routes.get('items', []):
                if "prometheus" in route['metadata']['name'].lower():
                    host = route['spec'].get('host')
                    if host:
                        self.prometheus_url = f"https://{host}"
                        logger.info(f"Found Prometheus route: {self.prometheus_url}")
                        return self.prometheus_url
                                    
            # Fallback: try to find service
            try:
                v1 = client.CoreV1Api(self.k8s_client)
                
                # Try to find prometheus service in openshift-monitoring namespace
                services = v1.list_namespaced_service(namespace="openshift-monitoring")
                
                for service in services.items:
                    if "prometheus" in service.metadata.name.lower():
                        if service.spec.ports:
                            port = service.spec.ports[0].port
                            # Use internal cluster DNS name
                            prometheus_url = f"https://{service.metadata.name}.openshift-monitoring.svc.cluster.local:{port}"
                            self.prometheus_url = prometheus_url
                            logger.info(f"Found Prometheus URL: {prometheus_url}")
                            return prometheus_url
                
                            
            except Exception as service_error:
                logger.warning(f"Could not fetch routes: {service_error}")
            
            logger.error("Could not discover Prometheus URL")
            return None
            
        except Exception as e:
            logger.error(f"Error discovering Prometheus URL: {e}")
            return None
    
    def get_prometheus_token(self) -> Optional[str]:
        """Get service account token for Prometheus access."""
        if self.prometheus_token:
            return self.prometheus_token
        
        try:
            # First try: create token using oc command
            result = subprocess.run([
                'oc', 'create', 'token', 'prometheus-k8s', '-n', 'openshift-monitoring'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.prometheus_token = result.stdout.strip()
                logger.info("Successfully created Prometheus token using 'oc create token'")
                return self.prometheus_token
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError) as e:
            logger.warning(f"Failed to create token with 'oc create token': {e}")
        
        try:
            # Fallback: try new-token (deprecated but might work)
            result = subprocess.run([
                'oc', 'sa', 'new-token', 'prometheus-k8s', '-n', 'openshift-monitoring'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.prometheus_token = result.stdout.strip()
                logger.info("Successfully created Prometheus token using 'oc sa new-token'")
                return self.prometheus_token
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError) as e:
            logger.warning(f"Failed to create token with 'oc sa new-token': {e}")
        
        # Try to get token from service account secret
        try:
            v1 = client.CoreV1Api(self.k8s_client)
            
            # Get service account
            sa = v1.read_namespaced_service_account(
                name="prometheus-k8s",
                namespace="openshift-monitoring"
            )
            
            # Look for token secrets
            for secret_ref in sa.secrets or []:
                secret = v1.read_namespaced_secret(
                    name=secret_ref.name,
                    namespace="openshift-monitoring"
                )
                
                if secret.type == "kubernetes.io/service-account-token":
                    token_data = secret.data.get('token')
                    if token_data:
                        import base64
                        self.prometheus_token = base64.b64decode(token_data).decode('utf-8')
                        logger.info("Successfully extracted token from service account secret")
                        return self.prometheus_token
            
        except Exception as e:
            logger.error(f"Failed to get token from service account secret: {e}")
        
        return None
    
    def test_prometheus_connection(self) -> bool:
        """Test connection to Prometheus with the obtained token."""
        prometheus_url = self.get_prometheus_url()
        token = self.get_prometheus_token()
        
        if not prometheus_url or not token:
            logger.error("Missing Prometheus URL or token")
            return False
        
        try:
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            # Test with a simple query
            test_url = urljoin(prometheus_url, '/api/v1/query')
            params = {'query': 'up'}
            
            response = requests.get(
                test_url,
                headers=headers,
                params=params,
                verify=False,  # Skip SSL verification for internal services
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Prometheus connection test successful")
                return True
            else:
                logger.error(f"Prometheus connection test failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Prometheus connection test failed: {e}")
            return False
    
    def get_auth_headers(self) -> dict:
        """Get authentication headers for Prometheus requests."""
        token = self.get_prometheus_token()
        if not token:
            raise ValueError("No Prometheus token available")
        
        return {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
    
    def setup_connection(self) -> Tuple[Optional[str], Optional[dict]]:
        """Setup complete Prometheus connection and return URL and headers."""
        prometheus_url = self.get_prometheus_url()
        
        if not prometheus_url:
            raise ValueError("Could not discover Prometheus URL")
        
        if not self.test_prometheus_connection():
            raise ValueError("Could not establish connection to Prometheus")
        
        return prometheus_url, self.get_auth_headers()


# Global auth instance
ocp_auth = OCPAuth()