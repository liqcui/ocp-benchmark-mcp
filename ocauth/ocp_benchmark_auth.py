#!/usr/bin/env python3
"""OpenShift Authentication and Prometheus Connection Module"""

import os
import subprocess
import logging
import asyncio
from typing import Optional, Dict, Any, Tuple
import kubernetes
from kubernetes import client, config
import requests
import json

logger = logging.getLogger(__name__)

class OpenShiftAuth:
    """OpenShift authentication and Prometheus connection management"""
    
    def __init__(self):
        self.kube_client = None
        self.prometheus_url = None
        self.prometheus_token = None
        self._initialize_kube_client()
    
    def _initialize_kube_client(self) -> None:
        """Initialize Kubernetes client using KUBECONFIG"""
        try:
            # Load kubeconfig from environment or default location
            if 'KUBECONFIG' in os.environ:
                config.load_kube_config(config_file=os.environ['KUBECONFIG'])
            else:
                config.load_kube_config()
            
            self.kube_client = client.ApiClient()
            logger.info("Successfully initialized Kubernetes client")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise
    
    async def get_prometheus_url(self) -> Optional[str]:
        """Automatically discover Prometheus URL from OpenShift monitoring"""
        try:
            v1 = client.CoreV1Api(self.kube_client)
            
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
            
            # Fallback: try to find route
            try:
                custom_api = client.CustomObjectsApi(self.kube_client)
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
                            
            except Exception as route_error:
                logger.warning(f"Could not fetch routes: {route_error}")
            
            logger.error("Could not discover Prometheus URL")
            return None
            
        except Exception as e:
            logger.error(f"Error discovering Prometheus URL: {e}")
            return None
    
    async def get_prometheus_token(self) -> Optional[str]:
        """Get service account token for Prometheus access"""
        try:
            # First try to create a token using oc command
            result = await self._run_oc_command([
                "create", "token", "-n", "openshift-monitoring", "prometheus-k8s"
            ])
            
            if result and result.strip():
                self.prometheus_token = result.strip()
                logger.info("Successfully created Prometheus token using 'oc create token'")
                return self.prometheus_token
            
        except Exception as e:
            logger.warning(f"Failed to create token with 'oc create token': {e}")
        
        try:
            # Fallback: try with sa new-token
            result = await self._run_oc_command([
                "sa", "new-token", "-n", "openshift-monitoring", "prometheus-k8s"
            ])
            
            if result and result.strip():
                self.prometheus_token = result.strip()
                logger.info("Successfully created Prometheus token using 'oc sa new-token'")
                return self.prometheus_token
                
        except Exception as e:
            logger.warning(f"Failed to create token with 'oc sa new-token': {e}")
        
        # Last resort: try to get token from secret
        try:
            token = await self._get_token_from_secret()
            if token:
                self.prometheus_token = token
                logger.info("Successfully retrieved token from service account secret")
                return token
        except Exception as e:
            logger.error(f"Failed to get token from secret: {e}")
        
        logger.error("Could not obtain Prometheus token")
        return None
    
    async def _run_oc_command(self, args: list) -> Optional[str]:
        """Run oc command and return output"""
        try:
            cmd = ["oc"] + args
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return stdout.decode('utf-8')
            else:
                logger.error(f"Command failed: {' '.join(cmd)}, Error: {stderr.decode('utf-8')}")
                return None
                
        except Exception as e:
            logger.error(f"Error running oc command: {e}")
            return None
    
    async def _get_token_from_secret(self) -> Optional[str]:
        """Get token from service account secret"""
        try:
            v1 = client.CoreV1Api(self.kube_client)
            
            # Get the service account
            sa = v1.read_namespaced_service_account(
                name="prometheus-k8s",
                namespace="openshift-monitoring"
            )
            
            # Look for token secrets
            if sa.secrets:
                for secret_ref in sa.secrets:
                    secret = v1.read_namespaced_secret(
                        name=secret_ref.name,
                        namespace="openshift-monitoring"
                    )
                    
                    if secret.type == "kubernetes.io/service-account-token":
                        token_data = secret.data.get("token")
                        if token_data:
                            import base64
                            return base64.b64decode(token_data).decode('utf-8')
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting token from secret: {e}")
            return None
    
    async def initialize_prometheus_connection(self) -> Tuple[Optional[str], Optional[str]]:
        """Initialize Prometheus connection - returns (url, token)"""
        url = await self.get_prometheus_url()
        token = await self.get_prometheus_token()
        
        if url and token:
            # Test the connection
            if await self.test_prometheus_connection(url, token):
                logger.info("Prometheus connection successfully established")
                return url, token
            else:
                logger.error("Prometheus connection test failed")
        
        return None, None
    
    async def test_prometheus_connection(self, url: str, token: str) -> bool:
        """Test Prometheus connection"""
        try:
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }
            
            # Use a simple query to test connection
            test_url = f"{url}/api/v1/query"
            params = {'query': 'up'}
            
            response = requests.get(
                test_url,
                headers=headers,
                params=params,
                verify=False,  # Skip SSL verification for internal services
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    logger.info("Prometheus connection test successful")
                    return True
            
            logger.error(f"Prometheus connection test failed: {response.status_code} - {response.text}")
            return False
            
        except Exception as e:
            logger.error(f"Error testing Prometheus connection: {e}")
            return False
    
    def get_kube_client(self) -> Optional[client.ApiClient]:
        """Get Kubernetes client"""
        return self.kube_client

# Global authentication instance
auth = OpenShiftAuth()