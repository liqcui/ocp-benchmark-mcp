"""Base Prometheus query functionality."""
import logging
import requests
import json
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urljoin
from ocauth.ocp_benchmark_auth import ocp_auth


logger = logging.getLogger(__name__)


class PrometheusClient:
    """Client for querying Prometheus metrics."""
    
    def __init__(self):
        self.prometheus_url = None
        self.headers = None
        self._setup_connection()
    
    def _setup_connection(self):
        """Setup Prometheus connection."""
        try:
            self.prometheus_url, self.headers = ocp_auth.setup_connection()
        except Exception as e:
            logger.error(f"Failed to setup Prometheus connection: {e}")
            raise
    
    def query_instant(self, 
                     query: str, 
                     time: Optional[Union[datetime, str]] = None) -> Dict[str, Any]:
        """Execute an instant query against Prometheus.
        
        Args:
            query: PromQL query string
            time: Optional timestamp for the query (defaults to now)
        
        Returns:
            Dictionary containing query results
        """
        if not self.prometheus_url or not self.headers:
            raise RuntimeError("Prometheus connection not established")
        
        url = urljoin(self.prometheus_url, '/api/v1/query')
        
        params = {'query': query}
        if time:
            if isinstance(time, datetime):
                params['time'] = time.timestamp()
            else:
                params['time'] = time
        
        try:
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                verify=False,  # Skip SSL verification for internal services
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'success':
                raise RuntimeError(f"Prometheus query failed: {data.get('error', 'Unknown error')}")
            
            return {
                'status': 'success',
                'query': query,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'data': data.get('data', {}),
                'result_type': data.get('data', {}).get('resultType'),
                'result': data.get('data', {}).get('result', [])
            }
        
        except requests.RequestException as e:
            logger.error(f"Failed to execute instant query: {e}")
            raise RuntimeError(f"Prometheus query failed: {e}")
    
    def query_range(self,
                   query: str,
                   start: Union[datetime, str],
                   end: Union[datetime, str],
                   step: str = '1m') -> Dict[str, Any]:
        """Execute a range query against Prometheus.
        
        Args:
            query: PromQL query string
            start: Start time for the range query
            end: End time for the range query
            step: Query resolution step width (e.g., '1m', '5m', '1h')
        
        Returns:
            Dictionary containing query results
        """
        if not self.prometheus_url or not self.headers:
            raise RuntimeError("Prometheus connection not established")
        
        url = urljoin(self.prometheus_url, '/api/v1/query_range')
        
        params = {
            'query': query,
            'step': step
        }
        
        # Convert datetime objects to timestamps
        if isinstance(start, datetime):
            params['start'] = start.timestamp()
        else:
            params['start'] = start
        
        if isinstance(end, datetime):
            params['end'] = end.timestamp()
        else:
            params['end'] = end
        
        try:
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                verify=False,  # Skip SSL verification for internal services
                timeout=60  # Longer timeout for range queries
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'success':
                raise RuntimeError(f"Prometheus range query failed: {data.get('error', 'Unknown error')}")
            
            return {
                'status': 'success',
                'query': query,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'start': params['start'],
                'end': params['end'],
                'step': step,
                'data': data.get('data', {}),
                'result_type': data.get('data', {}).get('resultType'),
                'result': data.get('data', {}).get('result', [])
            }
        
        except requests.RequestException as e:
            logger.error(f"Failed to execute range query: {e}")
            raise RuntimeError(f"Prometheus range query failed: {e}")
    
    def calculate_statistics(self, result_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate min, max, mean statistics from Prometheus result data.
        
        Args:
            result_data: List of result items from Prometheus query
        
        Returns:
            Dictionary with min, max, mean statistics
        """
        all_values: List[float] = []

        for item in result_data:
            values = item.get('values')
            single_value = item.get('value')

            # Handle range query results
            if values:
                # Two possible shapes:
                # 1) Raw Prometheus: [[timestamp, value], ...]
                # 2) Formatted result: [{ 'timestamp': ts, 'value': v }, ...]
                first = values[0] if isinstance(values, list) and len(values) > 0 else None
                if isinstance(first, (list, tuple)) and len(first) >= 2:
                    for ts, val in values:
                        try:
                            fv = float(val)
                            if not math.isnan(fv) and math.isfinite(fv):
                                all_values.append(fv)
                        except (ValueError, TypeError):
                            continue
                elif isinstance(first, dict):
                    for point in values:
                        try:
                            fv = float(point.get('value'))
                            if not math.isnan(fv) and math.isfinite(fv):
                                all_values.append(fv)
                        except (ValueError, TypeError):
                            continue

            # Handle instant query results
            elif single_value is not None:
                # Possible shapes:
                # 1) Raw Prometheus: [timestamp, value]
                # 2) Formatted result: {'timestamp': ts, 'value': v}
                try:
                    if isinstance(single_value, (list, tuple)) and len(single_value) >= 2:
                        _, val = single_value
                    elif isinstance(single_value, dict):
                        val = single_value.get('value')
                    else:
                        val = single_value

                    fv = float(val)
                    if not math.isnan(fv) and math.isfinite(fv):
                        all_values.append(fv)
                except (ValueError, TypeError, IndexError):
                    continue

        if not all_values:
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'count': 0}

        total = sum(all_values)
        count = len(all_values)
        return {
            'min': round(min(all_values), 6),
            'max': round(max(all_values), 6),
            'mean': round(total / count if count else 0.0, 6),
            'count': count
        }
    
    def format_query_result(self, 
                           query_result: Dict[str, Any], 
                           include_stats: bool = True) -> Dict[str, Any]:
        """Format Prometheus query result for easier consumption.
        
        Args:
            query_result: Raw query result from query_instant or query_range
            include_stats: Whether to include calculated statistics
        
        Returns:
            Formatted result dictionary
        """
        formatted = {
            'query': query_result.get('query'),
            'timestamp': query_result.get('timestamp'),
            'result_type': query_result.get('result_type'),
            'results': []
        }
        
        # Add range query specific fields
        if 'start' in query_result:
            formatted['start'] = query_result['start']
            formatted['end'] = query_result['end']
            formatted['step'] = query_result['step']
        
        # Process results
        for item in query_result.get('result', []):
            result_item = {
                'metric': item.get('metric', {}),
                'values': []
            }
            
            if item.get('values'):  # Range query
                result_item['values'] = [
                    {'timestamp': ts, 'value': round(float(val), 6)}
                    for ts, val in item['values']
                    if val != 'NaN'
                ]
            elif item.get('value'):  # Instant query
                timestamp, value = item['value']
                if value != 'NaN':
                    result_item['values'] = [{
                        'timestamp': timestamp,
                        'value': round(float(value), 6)
                    }]
            
            formatted['results'].append(result_item)
        
        # Add statistics if requested
        if include_stats:
            formatted['statistics'] = self.calculate_statistics(query_result.get('result', []))
        
        return formatted
    
    def test_connection(self) -> bool:
        """Test Prometheus connection with a simple query."""
        try:
            result = self.query_instant('up')
            return result.get('status') == 'success'
        except Exception as e:
            logger.error(f"Prometheus connection test failed: {e}")
            return False
    
    def get_available_metrics(self, limit: int = 100) -> List[str]:
        """Get list of available metrics from Prometheus.
        
        Args:
            limit: Maximum number of metrics to return
        
        Returns:
            List of available metric names
        """
        try:
            url = urljoin(self.prometheus_url, '/api/v1/label/__name__/values')
            response = requests.get(
                url,
                headers=self.headers,
                verify=False,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get('status') == 'success':
                metrics = data.get('data', [])
                return metrics[:limit] if limit else metrics
            else:
                logger.error(f"Failed to get metrics: {data.get('error')}")
                return []
        
        except Exception as e:
            logger.error(f"Failed to get available metrics: {e}")
            return []
    
    def validate_query(self, query: str) -> bool:
        """Validate a PromQL query without executing it.
        
        Args:
            query: PromQL query string
        
        Returns:
            True if query is valid, False otherwise
        """
        try:
            # Use a very short time range to minimize impact
            end = datetime.now(timezone.utc)
            start = end.replace(second=end.second - 1)  # 1 second range
            
            result = self.query_range(query, start, end, '1s')
            return result.get('status') == 'success'
        
        except Exception:
            return False


# Global Prometheus client instance
prometheus_client = PrometheusClient()