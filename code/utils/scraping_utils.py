"""
Shared utilities for web scraping operations.
This module contains common functions used by both schema extraction and content scraping scripts.

Features:
- Robust HTTP session management with retry strategies
- JSON-LD extraction from web pages
- Article content extraction using GNews
- Progress tracking and error handling
- Adaptive delay mechanisms for rate limiting

Author: Research Team
Version: 1.0.0
"""

import json
import logging
import os
import random
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from gnews import GNews

# Suppress urllib3 warnings and verbose logging
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)

# =============================================================================
# SESSION MANAGEMENT AND CONFIGURATION
# =============================================================================

def create_robust_session() -> requests.Session:
    """
    Create session with automatic retries and robust configuration.
    
    Returns:
        requests.Session: Configured session with retry strategy and headers
    """
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=5,  # Maximum 5 retries
        backoff_factor=2,  # Exponential time: 2, 4, 8, 16, 32 seconds
        status_forcelist=[429, 500, 502, 503, 504, 520, 521, 522, 524],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        raise_on_status=False
    )
    
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=10
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # More robust headers
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    })
    
    return session


def adaptive_delay(success_count: int, error_count: int, base_delay: float = 2) -> float:
    """
    Adaptive delay based on recent success/error rates.
    
    Args:
        success_count (int): Number of recent successful requests
        error_count (int): Number of recent failed requests
        base_delay (float): Base delay in seconds
        
    Returns:
        float: Calculated delay time in seconds
    """
    error_rate = error_count / (success_count + error_count + 1)
    
    if error_rate > 0.3:  # Many errors
        return base_delay * random.uniform(3, 6)
    elif error_rate > 0.1:  # Some errors
        return base_delay * random.uniform(2, 4)
    else:  # Few errors
        return base_delay * random.uniform(1, 2)


# =============================================================================
# DATA EXTRACTION FUNCTIONS
# =============================================================================
def extract_json_ld_robust(url: str, session: requests.Session, max_retries: int = 3) -> Tuple[Optional[int], str, Union[List[Dict], Dict[str, str]]]:
    """
    Robust JSON-LD extraction with specific error handling.
    
    Args:
        url (str): URL to extract JSON-LD from
        session (requests.Session): HTTP session to use
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        Tuple[Optional[int], str, Union[List[Dict], Dict[str, str]]]: 
            (status_code, status_reason, extracted_data_or_error)
    """
    
    for attempt in range(max_retries):
        try:
            # Progressive timeout
            timeout = (10 + attempt * 5, 30 + attempt * 10)  # (connect, read)
            
            response = session.get(url, timeout=timeout)
            status_code = response.status_code
            status_reason = response.reason

            if status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                json_ld_scripts = soup.find_all("script", {"type": "application/ld+json"})
                extracted_data = []

                for script in json_ld_scripts:
                    try:
                        if script.string:
                            json_data = json.loads(script.string)
                            if isinstance(json_data, dict):
                                extracted_data.append(json_data)
                            elif isinstance(json_data, list):
                                extracted_data.extend(json_data)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error for {url}: {e}")

                return status_code, status_reason, extracted_data
            
            elif status_code in [429, 503]:  # Rate limit or server busy
                wait_time = (2 ** attempt) * random.uniform(1, 3)
                print(f"Rate limited for {url}, waiting {wait_time:.2f}s before retry {attempt+1}")
                time.sleep(wait_time)
                continue
            
            else:
                return status_code, status_reason, {"error": f"HTTP {status_code}: {status_reason}"}

        except requests.exceptions.ConnectTimeout:
            print(f"Connection timeout for {url} (attempt {attempt+1})")
        except requests.exceptions.ReadTimeout:
            print(f"Read timeout for {url} (attempt {attempt+1})")
        except requests.exceptions.ConnectionError as e:
            if "Connection refused" in str(e):
                print(f"Connection refused for {url} (attempt {attempt+1})")
            else:
                print(f"Connection error for {url}: {str(e)} (attempt {attempt+1})")
        except requests.exceptions.RequestException as e:
            print(f"Request error for {url}: {str(e)} (attempt {attempt+1})")
        
        if attempt < max_retries - 1:
            # Exponential wait with jitter
            wait_time = (2 ** attempt) * random.uniform(1, 2) + random.uniform(0, 1)
            print(f"Waiting {wait_time:.2f}s before retry...")
            time.sleep(wait_time)
    
    return None, "Max retries exceeded", {"error": "Failed after all retries"}


def extract_article_info(url: str, session: Optional[requests.Session] = None, max_retries: int = 3) -> Tuple[Optional[int], str, Union[Dict[str, Any], Dict[str, str]]]:
    """
    Extract article metadata using GNews with robust error handling.
    
    Args:
        url (str): URL to extract article from
        session (Optional[requests.Session]): HTTP session to use
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        Tuple[Optional[int], str, Union[Dict[str, Any], Dict[str, str]]]: 
            (status_code, status_reason, article_data_or_error)
    """
    google_news = GNews()
    
    if session is None:
        session = create_robust_session()
    
    for attempt in range(max_retries):
        try:
            # Progressive timeout
            timeout = (10 + attempt * 5, 30 + attempt * 10)  # (connect, read)
            
            response = session.get(url, timeout=timeout)
            status_code = response.status_code
            status_reason = response.reason
            
            if status_code == 200:
                article = google_news.get_full_article(url)

                # Handle cases where article couldn't be fetched
                if not article or not article.text:
                    return None, "Article Fetch Failed", {"error": "Failed to retrieve article content"}

                extracted_data = {
                    "title": article.title if article.title else "No Title",
                    "author": article.authors if article.authors else "Unknown Author",
                    "content": article.text if article.text else "No Content",
                    "images": list(article.images) if isinstance(article.images, set) else article.images or []  # Convert set to list
                }

                return status_code, status_reason, extracted_data
            
            elif status_code in [429, 503]:  # Rate limit or server busy
                wait_time = (2 ** attempt) * random.uniform(1, 3)
                print(f"Rate limited for {url}, waiting {wait_time:.2f}s before retry {attempt+1}")
                time.sleep(wait_time)
                continue
            
            else:
                return status_code, status_reason, {"error": f"HTTP {status_code}: {status_reason}"}

        except requests.exceptions.ConnectTimeout:
            print(f"Connection timeout for {url} (attempt {attempt+1})")
        except requests.exceptions.ReadTimeout:
            print(f"Read timeout for {url} (attempt {attempt+1})")
        except requests.exceptions.ConnectionError as e:
            if "Connection refused" in str(e):
                print(f"Connection refused for {url} (attempt {attempt+1})")
            else:
                print(f"Connection error for {url}: {str(e)} (attempt {attempt+1})")
        except requests.exceptions.RequestException as e:
            print(f"Request error for {url}: {str(e)} (attempt {attempt+1})")
        
        if attempt < max_retries - 1:
            # Exponential wait with jitter
            wait_time = (2 ** attempt) * random.uniform(1, 2) + random.uniform(0, 1)
            print(f"Waiting {wait_time:.2f}s before retry...")
            time.sleep(wait_time)
    
    return None, "Max retries exceeded", {"error": "Failed after all retries"}


# =============================================================================
# FILE I/O AND DATA MANAGEMENT
# =============================================================================
def load_urls_from_file(input_file: str) -> Optional[List[str]]:
    """
    Load URLs from input file.
    
    Args:
        input_file (str): Path to the input file containing URLs
        
    Returns:
        Optional[List[str]]: List of URLs or None if file not found
    """
    urls = []
    if os.path.exists(input_file):
        with open(input_file, "r", encoding="utf-8") as file:
            urls = [line.strip() for line in file if line.strip()]
    else:
        print(f"Input file not found: {input_file}")
        return None
    
    print(f"Loaded {len(urls)} URLs from {input_file}")
    return urls


def load_existing_data(output_file: str) -> Dict[str, Dict]:
    """
    Load existing data from output file.
    
    Args:
        output_file (str): Path to the output JSON file
        
    Returns:
        Dict[str, Dict]: Dictionary mapping URLs to their data
    """
    existing_data = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as file:
                existing_data_list = json.load(file)
                existing_data = {entry["url"]: entry for entry in existing_data_list}
                print(f"Loaded {len(existing_data)} existing entries from {output_file}")
        except json.JSONDecodeError:
            print("Error reading existing data file, proceeding with fresh output.")
    return existing_data


def save_progress(existing_data: Dict[str, Dict], output_file: str) -> None:
    """
    Save progress safely to output file.
    
    Args:
        existing_data (Dict[str, Dict]): Dictionary of data to save
        output_file (str): Path to the output file
    """
    try:
        updated_data_list = list(existing_data.values())
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(updated_data_list, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving progress: {e}")


# =============================================================================
# URL PROCESSING AND WORKFLOW MANAGEMENT
# =============================================================================

def prepare_urls_to_process(urls: List[str], existing_data: Dict[str, Dict]) -> Optional[List[str]]:
    """
    Prepare list of URLs to process (failed + new).
    
    Args:
        urls (List[str]): All URLs to consider
        existing_data (Dict[str, Dict]): Previously processed data
        
    Returns:
        Optional[List[str]]: URLs that need processing or None if all done
    """
    # Check if we want to retry failed URLs (status_code is null)
    failed_urls = [entry["url"] for entry in existing_data.values() if entry.get("status_code") is None]
    new_urls = [url for url in urls if url not in existing_data]
    
    # Process both failed URLs and new URLs
    urls_to_process = failed_urls + new_urls
    
    print(f"Found {len(failed_urls)} failed URLs (status_code: null) to retry...")
    print(f"Found {len(new_urls)} completely new URLs to process...")
    print(f"Total URLs to process: {len(urls_to_process)}")
    
    if not urls_to_process:
        print("No URLs to process. All URLs have been successfully processed.")
        return None
    
    return urls_to_process


# =============================================================================
# ERROR HANDLING AND LOGGING
# =============================================================================

def handle_request_exceptions(url: str, attempt: int, max_retries: int) -> Dict:
    """
    Handle common request exceptions with appropriate logging.
    
    Args:
        url (str): URL that failed
        attempt (int): Current attempt number
        max_retries (int): Maximum retry attempts
        
    Returns:
        Dict: Exception handlers mapping
    """
    exception_handlers = {
        requests.exceptions.ConnectTimeout: lambda: print(f"Connection timeout for {url} (attempt {attempt+1})"),
        requests.exceptions.ReadTimeout: lambda: print(f"Read timeout for {url} (attempt {attempt+1})"),
        requests.exceptions.ConnectionError: lambda e: (
            print(f"Connection refused for {url} (attempt {attempt+1})") 
            if "Connection refused" in str(e) 
            else print(f"Connection error for {url}: {str(e)} (attempt {attempt+1})")
        ),
        requests.exceptions.RequestException: lambda e: print(f"Request error for {url}: {str(e)} (attempt {attempt+1})")
    }
    
    # Apply exponential wait with jitter if not last attempt
    if attempt < max_retries - 1:
        wait_time = (2 ** attempt) * random.uniform(1, 2) + random.uniform(0, 1)
        print(f"Waiting {wait_time:.2f}s before retry...")
        time.sleep(wait_time)
    
    return exception_handlers


def log_connection_attempt(url: str, attempt: int, status_code: Optional[int] = None, error_type: Optional[str] = None) -> None:
    """
    Standardized logging for connection attempts.
    
    Args:
        url (str): URL being accessed
        attempt (int): Attempt number
        status_code (Optional[int]): HTTP status code if available
        error_type (Optional[str]): Type of error if applicable
    """
    if status_code:
        if status_code == 200:
            print(f"✓ Successfully connected to {url}")
        elif status_code in [429, 503]:
            print(f"⚠ Rate limited for {url} (HTTP {status_code}) - attempt {attempt+1}")
        else:
            print(f"✗ HTTP {status_code} error for {url} (attempt {attempt+1})")
    elif error_type:
        print(f"✗ {error_type} for {url} (attempt {attempt+1})")
    else:
        print(f"→ Attempting connection to {url} (attempt {attempt+1})")


# =============================================================================
# STATISTICS AND REPORTING
# =============================================================================

def print_final_stats(existing_data: Dict[str, Dict], new_entries: List[str]) -> None:
    """
    Print final statistics for the scraping operation.
    
    Args:
        existing_data (Dict[str, Dict]): All processed data
        new_entries (List[str]): URLs that were newly processed
    """
    total_entries = len(existing_data)
    success_entries = len([entry for entry in existing_data.values() if entry['status_code'] == 200])
    
    print(f"\n{'='*50}")
    print(f"SCRAPING COMPLETED")
    print(f"{'='*50}")
    print(f"Total entries: {total_entries}")
    print(f"New entries processed: {len(new_entries)}")
    print(f"Successful: {success_entries} ({success_entries/total_entries*100:.1f}%)")
    print(f"Failed: {total_entries - success_entries}")
    print(f"{'='*50}")
