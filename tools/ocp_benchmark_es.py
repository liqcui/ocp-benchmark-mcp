"""Elasticsearch data retrieval tool for OpenShift benchmarking."""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta,timezone
from elasticsearch import Elasticsearch
from config.ocp_benchmark_config import config
import pytz
logger = logging.getLogger(__name__)


class ElasticsearchTool:
    """Tool for retrieving benchmark data from Elasticsearch."""
    
    def __init__(self):
        self.es_config = config.elasticsearch
        self.client = None
        self._connect()
        
    def _connect(self):
        """Connect to Elasticsearch cluster."""
        try:
            es_config = {
                'hosts': [self.es_config.url],
                'verify_certs': self.es_config.verify_ssl,
                'timeout': 30,
                'max_retries': 3,
                'retry_on_timeout': True
            }
            
            if self.es_config.username and self.es_config.password:
                es_config['http_auth'] = (self.es_config.username, self.es_config.password)
            
            self.client = Elasticsearch(**es_config)
            
            # Test connection
            if self.client.ping():
                logger.info("Successfully connected to Elasticsearch")
            else:
                logger.warning("Elasticsearch connection test failed")
                
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            self.client = None
    
    async def search_benchmark_results(self,
                                     query: Optional[Dict[str, Any]] = None,
                                     index_pattern: Optional[str] = None,
                                     time_range: Optional[Dict[str, str]] = None,
                                     size: int = 100) -> Dict[str, Any]:
        """Search for benchmark results in Elasticsearch."""
        try:
            if not self.client:
                raise RuntimeError("Elasticsearch client not available")
            
            index = index_pattern or self.es_config.index_pattern
            
            # Build search query
            search_query = self._build_search_query(query, time_range)
            
            # Execute search
            response = self.client.search(
                index=index,
                body=search_query,
                size=size
            )
            
            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": search_query,
                "total_hits": response['hits']['total']['value'] if 'total' in response['hits'] else 0,
                "results": []
            }
            
            # Process hits
            for hit in response['hits']['hits']:
                result["results"].append({
                    "index": hit.get("_index"),
                    "id": hit.get("_id"),
                    "score": hit.get("_score"),
                    "source": hit.get("_source", {})
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            raise
    
    async def get_kube_burner_results(self,
                                    test_name: Optional[str] = None,
                                    time_range: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Get kube-burner benchmark results."""
        try:
            query = {
                "bool": {
                    "must": [
                        {"term": {"benchmark_type.keyword": "kube-burner"}}
                    ]
                }
            }
            
            if test_name:
                query["bool"]["must"].append({
                    "term": {"test_name.keyword": test_name}
                })
            
            return await self.search_benchmark_results(
                query=query,
                time_range=time_range,
                index_pattern="kube-burner-*"
            )
            
        except Exception as e:
            logger.error(f"Failed to get kube-burner results: {e}")
            raise
    
    async def get_performance_metrics(self,
                                    metric_type: str,
                                    time_range: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Get performance metrics from Elasticsearch."""
        try:
            query = {
                "bool": {
                    "must": [
                        {"term": {"metric_type.keyword": metric_type}}
                    ]
                }
            }
            
            return await self.search_benchmark_results(
                query=query,
                time_range=time_range,
                index_pattern="performance-*"
            )
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            raise
    
    async def get_cluster_health_history(self,
                                       time_range: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Get cluster health history from Elasticsearch."""
        try:
            query = {
                "bool": {
                    "should": [
                        {"term": {"event_type.keyword": "cluster_health"}},
                        {"term": {"event_type.keyword": "node_status"}},
                        {"term": {"event_type.keyword": "pod_status"}}
                    ],
                    "minimum_should_match": 1
                }
            }
            
            # Add aggregation for health status over time
            aggs = {
                "health_over_time": {
                    "date_histogram": {
                        "field": "@timestamp",
                        "calendar_interval": "1h"
                    },
                    "aggs": {
                        "avg_healthy_nodes": {
                            "avg": {"field": "healthy_nodes"}
                        },
                        "avg_ready_pods": {
                            "avg": {"field": "ready_pods"}
                        }
                    }
                }
            }
            
            search_query = self._build_search_query(query, time_range)
            search_query["aggs"] = aggs
            
            response = self.client.search(
                index="cluster-health-*",
                body=search_query,
                size=0  # Only want aggregations
            )
            
            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_events": response['hits']['total']['value'] if 'total' in response['hits'] else 0,
                "health_timeline": []
            }
            
            # Process aggregations
            if "aggregations" in response:
                buckets = response["aggregations"]["health_over_time"]["buckets"]
                for bucket in buckets:
                    result["health_timeline"].append({
                        "timestamp": bucket["key_as_string"],
                        "avg_healthy_nodes": bucket["avg_healthy_nodes"]["value"],
                        "avg_ready_pods": bucket["avg_ready_pods"]["value"],
                        "doc_count": bucket["doc_count"]
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get cluster health history: {e}")
            raise
    
    async def get_application_logs(self,
                                 app_name: Optional[str] = None,
                                 namespace: Optional[str] = None,
                                 log_level: Optional[str] = None,
                                 time_range: Optional[Dict[str, str]] = None,
                                 size: int = 100) -> Dict[str, Any]:
        """Get application logs from Elasticsearch."""
        try:
            query = {"bool": {"must": []}}
            
            if app_name:
                query["bool"]["must"].append({
                    "term": {"kubernetes.labels.app.keyword": app_name}
                })
            
            if namespace:
                query["bool"]["must"].append({
                    "term": {"kubernetes.namespace_name.keyword": namespace}
                })
            
            if log_level:
                query["bool"]["must"].append({
                    "term": {"level.keyword": log_level.upper()}
                })
            
            # If no specific filters, get all application logs
            if not query["bool"]["must"]:
                query = {"match_all": {}}
            
            return await self.search_benchmark_results(
                query=query,
                time_range=time_range,
                index_pattern="app-*",
                size=size
            )
            
        except Exception as e:
            logger.error(f"Failed to get application logs: {e}")
            raise
    
    async def get_benchmark_summary(self,
                                  time_range: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Get benchmark summary with aggregations."""
        try:
            if not self.client:
                raise RuntimeError("Elasticsearch client not available")
            
            # Build aggregation query
            aggs = {
                "benchmark_types": {
                    "terms": {"field": "benchmark_type.keyword", "size": 20}
                },
                "test_results": {
                    "terms": {"field": "test_result.keyword", "size": 10}
                },
                "performance_metrics": {
                    "date_histogram": {
                        "field": "@timestamp", 
                        "calendar_interval": "1d"
                    },
                    "aggs": {
                        "avg_cpu_usage": {
                            "avg": {"field": "metrics.cpu_usage_percent"}
                        },
                        "avg_memory_usage": {
                            "avg": {"field": "metrics.memory_usage_percent"}
                        },
                        "avg_response_time": {
                            "avg": {"field": "metrics.response_time_ms"}
                        }
                    }
                }
            }
            
            search_query = self._build_search_query(None, time_range)
            search_query["aggs"] = aggs
            
            response = self.client.search(
                index=self.es_config.index_pattern,
                body=search_query,
                size=0  # Only want aggregations
            )
            
            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "summary": {
                    "total_documents": response['hits']['total']['value'] if 'total' in response['hits'] else 0,
                    "benchmark_types": [],
                    "test_results": [],
                    "performance_timeline": []
                }
            }
            
            # Process aggregations
            if "aggregations" in response:
                # Benchmark types
                for bucket in response["aggregations"]["benchmark_types"]["buckets"]:
                    result["summary"]["benchmark_types"].append({
                        "type": bucket["key"],
                        "count": bucket["doc_count"]
                    })
                
                # Test results
                for bucket in response["aggregations"]["test_results"]["buckets"]:
                    result["summary"]["test_results"].append({
                        "result": bucket["key"],
                        "count": bucket["doc_count"]
                    })
                
                # Performance timeline
                for bucket in response["aggregations"]["performance_metrics"]["buckets"]:
                    result["summary"]["performance_timeline"].append({
                        "timestamp": bucket["key_as_string"],
                        "avg_cpu_usage": bucket["avg_cpu_usage"]["value"],
                        "avg_memory_usage": bucket["avg_memory_usage"]["value"],
                        "avg_response_time": bucket["avg_response_time"]["value"],
                        "doc_count": bucket["doc_count"]
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get benchmark summary: {e}")
            raise
    
    def _build_search_query(self,
                           query: Optional[Dict[str, Any]] = None,
                           time_range: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Build Elasticsearch search query."""
        search_query = {
            "query": query or {"match_all": {}},
            "sort": [{"@timestamp": {"order": "desc"}}]
        }
        
        # Add time range filter
        if time_range:
            time_filter = {
                "range": {
                    "@timestamp": {}
                }
            }
            
            if "start" in time_range:
                time_filter["range"]["@timestamp"]["gte"] = time_range["start"]
            
            if "end" in time_range:
                time_filter["range"]["@timestamp"]["lte"] = time_range["end"]
            
            # Wrap existing query with bool and add time filter
            if isinstance(search_query["query"], dict) and "bool" in search_query["query"]:
                if "must" not in search_query["query"]["bool"]:
                    search_query["query"]["bool"]["must"] = []
                search_query["query"]["bool"]["must"].append(time_filter)
            else:
                search_query["query"] = {
                    "bool": {
                        "must": [
                            search_query["query"],
                            time_filter
                        ]
                    }
                }
        
        return search_query
    
    def _parse_time_range(self, duration: str) -> Dict[str, str]:
        """Parse duration to time range."""
        end_time = datetime.now(timezone.utc)
        
        duration = duration.lower().strip()
        if duration.endswith('h'):
            hours = int(duration[:-1])
            start_time = end_time - timedelta(hours=hours)
        elif duration.endswith('d'):
            days = int(duration[:-1])
            start_time = end_time - timedelta(days=days)
        elif duration.endswith('m'):
            minutes = int(duration[:-1])
            start_time = end_time - timedelta(minutes=minutes)
        else:
            # Default to 1 hour
            start_time = end_time - timedelta(hours=1)
        
        return {
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        }
    
    async def create_benchmark_index(self, index_name: str, mapping: Dict[str, Any]) -> bool:
        """Create a new index for benchmark data."""
        try:
            if not self.client:
                raise RuntimeError("Elasticsearch client not available")
            
            if self.client.indices.exists(index=index_name):
                logger.info(f"Index {index_name} already exists")
                return True
            
            response = self.client.indices.create(
                index=index_name,
                body={"mappings": mapping}
            )
            
            if response.get("acknowledged"):
                logger.info(f"Successfully created index {index_name}")
                return True
            else:
                logger.error(f"Failed to create index {index_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating index {index_name}: {e}")
            return False
    
    async def store_benchmark_result(self,
                                   index_name: str,
                                   document: Dict[str, Any],
                                   doc_id: Optional[str] = None) -> bool:
        """Store benchmark result in Elasticsearch."""
        try:
            if not self.client:
                raise RuntimeError("Elasticsearch client not available")
            
            # Add timestamp if not present
            if "@timestamp" not in document:
                document["@timestamp"] = datetime.now(timezone.utc).isoformat()
            
            response = self.client.index(
                index=index_name,
                body=document,
                id=doc_id
            )
            
            if response.get("result") in ["created", "updated"]:
                logger.info(f"Successfully stored document in {index_name}")
                return True
            else:
                logger.error(f"Failed to store document in {index_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            return False
    
    async def get_cluster_metrics_history(self,
                                        metric_types: List[str],
                                        time_range: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Get historical cluster metrics."""
        try:
            query = {
                "bool": {
                    "must": [
                        {"terms": {"metric_type.keyword": metric_types}}
                    ]
                }
            }
            
            # Add aggregation for metrics over time
            aggs = {
                "metrics_over_time": {
                    "date_histogram": {
                        "field": "@timestamp",
                        "calendar_interval": "5m"
                    },
                    "aggs": {
                        "metric_types": {
                            "terms": {"field": "metric_type.keyword"},
                            "aggs": {
                                "avg_value": {"avg": {"field": "value"}},
                                "max_value": {"max": {"field": "value"}},
                                "min_value": {"min": {"field": "value"}}
                            }
                        }
                    }
                }
            }
            
            search_query = self._build_search_query(query, time_range)
            search_query["aggs"] = aggs
            
            response = self.client.search(
                index="cluster-metrics-*",
                body=search_query,
                size=0
            )
            
            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metric_types": metric_types,
                "timeline": []
            }
            
            if "aggregations" in response:
                buckets = response["aggregations"]["metrics_over_time"]["buckets"]
                for bucket in buckets:
                    timeline_entry = {
                        "timestamp": bucket["key_as_string"],
                        "metrics": {}
                    }
                    
                    for metric_bucket in bucket["metric_types"]["buckets"]:
                        metric_type = metric_bucket["key"]
                        timeline_entry["metrics"][metric_type] = {
                            "avg": metric_bucket["avg_value"]["value"],
                            "max": metric_bucket["max_value"]["value"],
                            "min": metric_bucket["min_value"]["value"],
                            "count": metric_bucket["doc_count"]
                        }
                    
                    result["timeline"].append(timeline_entry)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get cluster metrics history: {e}")
            raise
    
    async def search_error_events(self,
                                severity: Optional[str] = None,
                                component: Optional[str] = None,
                                time_range: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Search for error events in logs."""
        try:
            query = {
                "bool": {
                    "must": [],
                    "should": [
                        {"term": {"level.keyword": "ERROR"}},
                        {"term": {"level.keyword": "FATAL"}},
                        {"term": {"level.keyword": "CRITICAL"}},
                        {"match": {"message": "error"}},
                        {"match": {"message": "failed"}},
                        {"match": {"message": "exception"}}
                    ],
                    "minimum_should_match": 1
                }
            }
            
            if severity:
                query["bool"]["must"].append({
                    "term": {"level.keyword": severity.upper()}
                })
            
            if component:
                query["bool"]["must"].append({
                    "term": {"kubernetes.labels.component.keyword": component}
                })
            
            return await self.search_benchmark_results(
                query=query,
                time_range=time_range,
                index_pattern="app-*",
                size=200
            )
            
        except Exception as e:
            logger.error(f"Failed to search error events: {e}")
            raise
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get Elasticsearch connection status."""
        status = {
            "connected": False,
            "cluster_info": {},
            "error": None
        }
        
        try:
            if self.client:
                if self.client.ping():
                    status["connected"] = True
                    cluster_info = self.client.info()
                    status["cluster_info"] = {
                        "cluster_name": cluster_info.get("cluster_name", "unknown"),
                        "version": cluster_info.get("version", {}).get("number", "unknown"),
                        "lucene_version": cluster_info.get("version", {}).get("lucene_version", "unknown")
                    }
                else:
                    status["error"] = "Ping failed"
            else:
                status["error"] = "Client not initialized"
                
        except Exception as e:
            status["error"] = str(e)
        
        return status
    
    def to_json(self, data: Dict[str, Any]) -> str:
        """Convert data to JSON string."""
        return json.dumps(data, indent=2, default=str)