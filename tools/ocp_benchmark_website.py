"""Website data retrieval tool for kube-burner documentation and resources."""

import json
import logging
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin, urlparse
from datetime import datetime
from config.ocp_benchmark_config import config
import pytz
logger = logging.getLogger(__name__)


class WebsiteTool:
    """Tool for retrieving data from kube-burner and related websites."""
    
    def __init__(self):
        self.web_config = config.web_scraping
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.web_config.user_agent
        })
        
    async def get_kube_burner_docs(self, 
                                  section: Optional[str] = None,
                                  deep_crawl: bool = False) -> Dict[str, Any]:
        """Retrieve kube-burner documentation."""
        try:
            base_url = self.web_config.kube_burner_url
            
            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "base_url": base_url,
                "sections": {},
                "links": [],
                "metadata": {}
            }
            
            # Get main page
            main_content = await self._fetch_page(base_url)
            if main_content:
                result["sections"]["main"] = main_content
                
                # Extract links for further crawling
                links = self._extract_links(main_content["content"], base_url)
                result["links"] = links
                
                # If deep crawl is enabled, fetch additional pages
                if deep_crawl:
                    for link in links[:10]:  # Limit to first 10 links
                        try:
                            page_content = await self._fetch_page(link["url"])
                            if page_content:
                                section_name = link.get("text", "unknown").lower().replace(" ", "_")
                                result["sections"][section_name] = page_content
                        except Exception as e:
                            logger.warning(f"Failed to fetch {link['url']}: {e}")
                
                # Extract specific section if requested
                if section:
                    section_content = self._extract_section(main_content["content"], section)
                    if section_content:
                        result["sections"][f"section_{section}"] = {
                            "title": section,
                            "content": section_content,
                            "extracted_at": datetime.now(timezone.utc).isoformat()
                        }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get kube-burner docs: {e}")
            raise
    
    async def get_benchmark_examples(self) -> Dict[str, Any]:
        """Get benchmark configuration examples from kube-burner docs."""
        try:
            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "examples": [],
                "configurations": {}
            }
            
            # Common kube-burner example URLs
            example_urls = [
                urljoin(self.web_config.kube_burner_url, "examples/"),
                urljoin(self.web_config.kube_burner_url, "configuration/"),
                urljoin(self.web_config.kube_burner_url, "metrics/")
            ]
            
            for url in example_urls:
                try:
                    content = await self._fetch_page(url)
                    if content:
                        # Extract code blocks and configuration examples
                        examples = self._extract_code_examples(content["content"])
                        if examples:
                            result["examples"].extend(examples)
                        
                        # Extract configuration sections
                        configs = self._extract_configurations(content["content"])
                        if configs:
                            result["configurations"].update(configs)
                            
                except Exception as e:
                    logger.warning(f"Failed to fetch examples from {url}: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get benchmark examples: {e}")
            raise
    
    async def get_performance_metrics_info(self) -> Dict[str, Any]:
        """Get information about performance metrics from documentation."""
        try:
            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics_info": {},
                "metric_types": [],
                "collection_methods": []
            }
            
            # Try to get metrics documentation
            metrics_url = urljoin(self.web_config.kube_burner_url, "metrics/")
            content = await self._fetch_page(metrics_url)
            
            if content:
                # Extract metrics information
                metrics_info = self._extract_metrics_info(content["content"])
                result["metrics_info"] = metrics_info
                
                # Extract metric types
                metric_types = self._extract_metric_types(content["content"])
                result["metric_types"] = metric_types
                
                # Extract collection methods
                collection_methods = self._extract_collection_methods(content["content"])
                result["collection_methods"] = collection_methods
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics info: {e}")
            raise
    
    async def search_best_practices(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """Search for best practices and recommendations."""
        try:
            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "topic": topic,
                "best_practices": [],
                "recommendations": [],
                "warnings": []
            }
            
            # Get main documentation
            main_content = await self._fetch_page(self.web_config.kube_burner_url)
            
            if main_content:
                # Extract best practices
                practices = self._extract_best_practices(main_content["content"], topic)
                result["best_practices"] = practices
                
                # Extract recommendations
                recommendations = self._extract_recommendations(main_content["content"], topic)
                result["recommendations"] = recommendations
                
                # Extract warnings or important notes
                warnings = self._extract_warnings(main_content["content"])
                result["warnings"] = warnings
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to search best practices: {e}")
            raise
    
    async def get_changelog_info(self) -> Dict[str, Any]:
        """Get changelog and version information."""
        try:
            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "versions": [],
                "latest_changes": [],
                "release_notes": {}
            }
            
            # Try GitHub releases API or changelog page
            changelog_urls = [
                "https://api.github.com/repos/kube-burner/kube-burner/releases",
                urljoin(self.web_config.kube_burner_url, "changelog/"),
                urljoin(self.web_config.kube_burner_url, "releases/")
            ]
            
            for url in changelog_urls:
                try:
                    if "api.github.com" in url:
                        # Handle GitHub API
                        response = self.session.get(url, timeout=self.web_config.timeout)
                        if response.status_code == 200:
                            releases = response.json()
                            for release in releases[:5]:  # Last 5 releases
                                result["versions"].append({
                                    "version": release.get("tag_name", "unknown"),
                                    "name": release.get("name", ""),
                                    "published_at": release.get("published_at", ""),
                                    "body": release.get("body", "")
                                })
                            break
                    else:
                        # Handle regular webpage
                        content = await self._fetch_page(url)
                        if content:
                            changes = self._extract_changelog(content["content"])
                            result["latest_changes"] = changes
                            break
                            
                except Exception as e:
                    logger.warning(f"Failed to fetch changelog from {url}: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get changelog info: {e}")
            raise
    
    async def _fetch_page(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch and parse a webpage."""
        try:
            response = self.session.get(url, timeout=self.web_config.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract metadata
            title = soup.find('title')
            title_text = title.get_text().strip() if title else 'Unknown'
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text_content = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return {
                "url": url,
                "title": title_text,
                "content": text,
                "html_content": str(soup),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "status_code": response.status_code
            }
            
        except Exception as e:
            logger.warning(f"Failed to fetch page {url}: {e}")
            return None
    
    def _extract_links(self, content: str, base_url: str) -> List[Dict[str, str]]:
        """Extract links from content."""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text().strip()
                
                # Convert relative URLs to absolute
                full_url = urljoin(base_url, href)
                
                # Only include relevant links
                if self._is_relevant_link(full_url, text):
                    links.append({
                        "url": full_url,
                        "text": text,
                        "title": link.get('title', '')
                    })
            
            return links[:50]  # Limit to 50 links
            
        except Exception as e:
            logger.warning(f"Failed to extract links: {e}")
            return []
    
    def _is_relevant_link(self, url: str, text: str) -> bool:
        """Check if a link is relevant for our purposes."""
        # Skip external links that are not related to documentation
        if not any(domain in url.lower() for domain in ['kube-burner', 'github.com']):
            return False
        
        # Skip certain file types
        if any(url.lower().endswith(ext) for ext in ['.jpg', '.png', '.gif', '.pdf']):
            return False
        
        # Include links with relevant text
        relevant_keywords = ['config', 'metric', 'example', 'guide', 'benchmark', 'performance', 'setup']
        return any(keyword in text.lower() for keyword in relevant_keywords)
    
    def _extract_section(self, content: str, section_name: str) -> Optional[str]:
        """Extract a specific section from content."""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Look for headings that match the section name
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                if section_name.lower() in heading.get_text().lower():
                    # Extract content until next heading of same or higher level
                    section_content = []
                    current = heading.next_sibling
                    
                    while current:
                        if current.name and current.name.startswith('h'):
                            # Check if it's a heading of same or higher level
                            current_level = int(current.name[1])
                            section_level = int(heading.name[1])
                            if current_level <= section_level:
                                break
                        
                        if hasattr(current, 'get_text'):
                            section_content.append(current.get_text().strip())
                        
                        current = current.next_sibling
                    
                    return ' '.join(section_content)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to extract section {section_name}: {e}")
            return None
    
    def _extract_code_examples(self, content: str) -> List[Dict[str, str]]:
        """Extract code examples from content."""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            examples = []
            
            # Find code blocks
            for code_block in soup.find_all(['pre', 'code']):
                code_text = code_block.get_text().strip()
                if len(code_text) > 50:  # Only include substantial code blocks
                    # Try to determine the language
                    language = 'unknown'
                    if code_block.get('class'):
                        classes = code_block.get('class')
                        for cls in classes:
                            if 'language-' in cls:
                                language = cls.replace('language-', '')
                            elif cls in ['yaml', 'json', 'bash', 'shell']:
                                language = cls
                    
                    examples.append({
                        "code": code_text,
                        "language": language,
                        "context": self._get_code_context(code_block)
                    })
            
            return examples
            
        except Exception as e:
            logger.warning(f"Failed to extract code examples: {e}")
            return []
    
    def _get_code_context(self, code_element) -> str:
        """Get context around a code element."""
        try:
            # Look for preceding heading or paragraph
            prev_element = code_element.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])
            if prev_element:
                return prev_element.get_text().strip()[:200]  # Limit to 200 chars
            return ""
        except:
            return ""
    
    def _extract_configurations(self, content: str) -> Dict[str, Any]:
        """Extract configuration information."""
        # This would contain logic to extract YAML/JSON configurations
        # For now, return a placeholder
        return {}
    
    def _extract_metrics_info(self, content: str) -> Dict[str, Any]:
        """Extract metrics information from content."""
        # Placeholder for metrics extraction logic
        return {}
    
    def _extract_metric_types(self, content: str) -> List[str]:
        """Extract available metric types."""
        # Placeholder for metric types extraction
        return []
    
    def _extract_collection_methods(self, content: str) -> List[str]:
        """Extract metric collection methods."""
        # Placeholder for collection methods extraction
        return []
    
    def _extract_best_practices(self, content: str, topic: Optional[str] = None) -> List[str]:
        """Extract best practices from content."""
        # Placeholder for best practices extraction
        return []
    
    def _extract_recommendations(self, content: str, topic: Optional[str] = None) -> List[str]:
        """Extract recommendations from content."""
        # Placeholder for recommendations extraction
        return []
    
    def _extract_warnings(self, content: str) -> List[str]:
        """Extract warnings or important notes."""
        # Placeholder for warnings extraction
        return []
    
    def _extract_changelog(self, content: str) -> List[Dict[str, str]]:
        """Extract changelog information."""
        # Placeholder for changelog extraction
        return []
    
    def to_json(self, data: Dict[str, Any]) -> str:
        """Convert data to JSON string."""
        return json.dumps(data, indent=2, default=str)