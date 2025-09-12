"""
Combined Web Scraper for Schema Extraction and Article Content Scraping
=======================================================================

This script combines the functionality of both schema extraction and article content scraping
into a single, configurable tool. It can perform either or both operations based on configuration.

Features:
- Schema extraction (JSON-LD) from web pages
- Article content extraction using GNews
- Robust error handling and retry mechanisms
- Progress tracking and adaptive delays
- Configurable operation modes

Usage:
    python combined_scraper.py --dataset mozambique --mode both
    python combined_scraper.py --dataset sudan --mode schema
    python combined_scraper.py --dataset drc --mode articles

Author: Research Team
Version: 1.0.0
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any

from tqdm import tqdm
from .scraping_utils import *


class NewsExtraction:
    """
    Combined scraper class for schema extraction and article content scraping.
    """
    
    def __init__(self, dataset_name: str, base_data_dir: str = "../data", 
                 input_file: Optional[str] = None,
                 schema_output_file: Optional[str] = None,
                 articles_output_file: Optional[str] = None):
        """
        Initialize the combined scraper.
        
        Args:
            dataset_name (str): Name of the dataset (e.g., 'mozambique', 'sudan', 'drc')
            base_data_dir (str): Base directory for data files
            input_file (Optional[str]): Custom path for input URLs file
            schema_output_file (Optional[str]): Custom path for schema output file
            articles_output_file (Optional[str]): Custom path for articles output file
        """
        self.dataset_name = dataset_name
        self.base_data_dir = base_data_dir
        self.data_dir = os.path.join(base_data_dir, dataset_name)
        
        # File paths - use custom paths if provided, otherwise use defaults
        self.input_file = input_file or os.path.join(self.data_dir, f"{dataset_name}_urls.txt")
        self.schema_output_file = schema_output_file or os.path.join(self.data_dir, f"{dataset_name}_schema.json")
        self.articles_output_file = articles_output_file or os.path.join(self.data_dir, f"{dataset_name}_articles_gnews.json")
        
        # Initialize session and counters
        self.session = create_robust_session()
        self.success_count = 0
        self.error_count = 0
        
    def extract_schema_data(self, urls_to_process: List[str]) -> Dict[str, Dict]:
        """
        Extract schema data (JSON-LD) from URLs.
        
        Args:
            urls_to_process (List[str]): URLs to process
            
        Returns:
            Dict[str, Dict]: Results of schema extraction
        """
        print(f"\n{'='*60}")
        print(f"SCHEMA EXTRACTION MODE")
        print(f"{'='*60}")
        
        # Load existing schema data
        existing_schema = load_existing_data(self.schema_output_file)
        
        # Filter URLs that need schema processing
        schema_urls = prepare_urls_to_process(urls_to_process, existing_schema)
        if schema_urls is None:
            print("No URLs need schema extraction.")
            return existing_schema
        
        new_entries = []
        self.success_count = 0
        self.error_count = 0
        
        print(f"Processing {len(schema_urls)} URLs for schema extraction...")
        
        for i, url in enumerate(tqdm(schema_urls, desc="Extracting Schema", unit="url")):
            status_code, status_reason, schema_data = extract_json_ld_robust(url, self.session)
            
            data_entry = {
                "url": url,
                "status_code": status_code,
                "status_reason": status_reason if status_code != 200 else None,
                "schema": schema_data
            }
            
            new_entries.append(data_entry)
            existing_schema[url] = data_entry
            
            # Update counters
            if status_code == 200:
                self.success_count += 1
            else:
                self.error_count += 1
            
            # Save progress every 10 URLs
            if (i + 1) % 10 == 0:
                save_progress(existing_schema, self.schema_output_file)
                current_success_rate = self.success_count/(self.success_count+self.error_count)*100 if (self.success_count+self.error_count) > 0 else 0
                print(f"Schema progress saved. Current success rate: {current_success_rate:.1f}%")
            
            # Adaptive delay
            if i < len(schema_urls) - 1:
                delay = adaptive_delay(self.success_count, self.error_count)
                time.sleep(delay)
        
        # Save final schema results
        save_progress(existing_schema, self.schema_output_file)
        print(f"\nSchema extraction completed!")
        print_final_stats(existing_schema, new_entries)
        
        return existing_schema
    
    def extract_article_data(self, urls_to_process: List[str]) -> Dict[str, Dict]:
        """
        Extract article content data using GNews.
        
        Args:
            urls_to_process (List[str]): URLs to process
            
        Returns:
            Dict[str, Dict]: Results of article extraction
        """
        print(f"\n{'='*60}")
        print(f"ARTICLE CONTENT EXTRACTION MODE")
        print(f"{'='*60}")
        
        # Load existing article data
        existing_articles = load_existing_data(self.articles_output_file)
        
        # Filter URLs that need article processing
        article_urls = prepare_urls_to_process(urls_to_process, existing_articles)
        if article_urls is None:
            print("No URLs need article extraction.")
            return existing_articles
        
        new_entries = []
        self.success_count = 0
        self.error_count = 0
        
        print(f"Processing {len(article_urls)} URLs for article extraction...")
        
        for i, url in enumerate(tqdm(article_urls, desc="Extracting Articles", unit="url")):
            status_code, status_reason, article_data = extract_article_info(url, self.session)
            
            data_entry = {
                "url": url,
                "status_code": status_code,
                "status_reason": status_reason if status_code != 200 else None,
                "data": article_data
            }
            
            new_entries.append(data_entry)
            existing_articles[url] = data_entry
            
            # Update counters
            if status_code == 200:
                self.success_count += 1
            else:
                self.error_count += 1
            
            # Save progress every 10 URLs
            if (i + 1) % 10 == 0:
                save_progress(existing_articles, self.articles_output_file)
                current_success_rate = self.success_count/(self.success_count+self.error_count)*100 if (self.success_count+self.error_count) > 0 else 0
                print(f"Article progress saved. Current success rate: {current_success_rate:.1f}%")
            
            # Adaptive delay
            if i < len(article_urls) - 1:
                delay = adaptive_delay(self.success_count, self.error_count)
                time.sleep(delay)
        
        # Save final article results
        save_progress(existing_articles, self.articles_output_file)
        print(f"\nArticle extraction completed!")
        print_final_stats(existing_articles, new_entries)
        
        return existing_articles
    
    def run(self, mode: str = "both") -> Dict[str, Any]:
        """
        Run the scraper in the specified mode.
        
        Args:
            mode (str): Operation mode - 'schema', 'articles', or 'both'
            
        Returns:
            Dict[str, Any]: Results of the scraping operation(s)
        """
        print(f"\n{'='*80}")
        print(f"COMBINED SCRAPER - Dataset: {self.dataset_name.upper()}")
        print(f"Mode: {mode.upper()}")
        print(f"{'='*80}")
        
        # Load URLs
        urls = load_urls_from_file(self.input_file)
        if urls is None:
            return {"error": "Could not load URLs"}
        
        results = {}
        
        if mode in ["schema", "both"]:
            results["schema"] = self.extract_schema_data(urls)
        
        if mode in ["articles", "both"]:
            results["articles"] = self.extract_article_data(urls)
        
        print(f"\n{'='*80}")
        print(f"ALL OPERATIONS COMPLETED FOR {self.dataset_name.upper()}")
        print(f"{'='*80}")
        
        # Print summary
        if "schema" in results:
            schema_success = len([e for e in results["schema"].values() if e.get('status_code') == 200])
            print(f"Schema extraction: {schema_success}/{len(results['schema'])} successful")
        
        if "articles" in results:
            article_success = len([e for e in results["articles"].values() if e.get('status_code') == 200])
            print(f"Article extraction: {article_success}/{len(results['articles'])} successful")
        
        return results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Combined web scraper for schema and article extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dataset mozambique --mode both
  %(prog)s --dataset sudan --mode schema
  %(prog)s --dataset drc --mode articles
  
Custom file paths:
  %(prog)s --dataset mozambique --input-file /path/to/urls.txt --schema-output /path/to/schema.json
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,  # Made optional
        help="Dataset name (e.g., mozambique, sudan, drc) - overrides the dataset_name variable in main()"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["schema", "articles", "both"],
        default="both",
        help="Scraping mode: 'schema' for JSON-LD extraction, 'articles' for content extraction, 'both' for both operations (default: both)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data",
        help="Base directory for data files (default: ../data)"
    )
    
    # Custom file path arguments
    parser.add_argument(
        "--input-file",
        type=str,
        help="Custom path for input URLs file (overrides default path based on dataset)"
    )
    
    parser.add_argument(
        "--schema-output",
        type=str,
        help="Custom path for schema output file (overrides default path)"
    )
    
    parser.add_argument(
        "--articles-output",
        type=str,
        help="Custom path for articles output file (overrides default path)"
    )
    
    return parser.parse_args()


