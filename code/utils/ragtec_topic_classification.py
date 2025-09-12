"""
RAGTEC: Retrieval-Augmented Generation for Topic Extraction and Classification

This module provides the topic classification component of the RAGTEC framework 
for news article topic assignment using predefined topics from the extraction phase.

RAGTEC Topic Classification Features:
- Article classification using predefined topics from extraction phase
- Token counting and cost analysis
- Comprehensive logging and result saving
- Analysis and reporting capabilities
- CSV and JSON output formats
"""

import os
import json
import tiktoken
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import re
from datetime import datetime

# LangChain imports
from langchain_openai import ChatOpenAI


class RAGTECTopicClassification:
    """
    RAGTEC: Retrieval-Augmented Generation for Topic Classification
    
    This class implements the topic classification component of the RAGTEC framework 
    which handles the complete pipeline for classifying news articles using predefined
    topics from the extraction phase, including comprehensive cost tracking and analysis.
    """
    
    def __init__(self, 
                 model: str = "gpt-4o-mini",
                 temperature: float = 0):
        """
        Initialize the RAGTEC Topic Classification module.
        
        Args:
            model: OpenAI model name for topic classification
            temperature: Temperature for LLM generation
        """
        self.model = model
        self.temperature = temperature
        
        # Initialize components
        self.llm = ChatOpenAI(model_name=model, temperature=temperature)
        self.encoding = tiktoken.encoding_for_model(model)
        
        # State tracking
        self.articles = []
        self.predefined_topics = []
        self.metadata = {}
        self.results = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def load_predefined_topics(self, topics_file: str) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Load predefined topics from the extraction phase output.
        
        Args:
            topics_file: Path to the extracted topics JSON file
            
        Returns:
            Tuple of (predefined_topics_list, metadata_dict)
        """
        print(f"Loading predefined topics from: {topics_file}")
        
        try:
            with open(topics_file, "r", encoding="utf-8") as f:
                inferred_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Topics file not found: {topics_file}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in topics file: {topics_file}")
        
        predefined_topics = []
        metadata = {
            "num_topics": 0,
            "explanation": "",
            "collection_topics": [],
            "collection_summary": ""
        }
        
        # Handle the structured JSON format from extraction phase
        if isinstance(inferred_data, dict):
            # New format with collection_summary, collection_topics, etc.
            metadata["num_topics"] = inferred_data.get("number_topics", 0)
            metadata["explanation"] = inferred_data.get("explanation", "")
            metadata["collection_summary"] = inferred_data.get("collection_summary", "")
            metadata["collection_topics"] = inferred_data.get("collection_topics", [])
            
            for topic in inferred_data.get("topics", []):
                predefined_topics.append({
                    "topic": topic.get("topic", ""),
                    "description": topic.get("description", ""),
                    "keywords": topic.get("keywords", [])
                })
        elif isinstance(inferred_data, list) and len(inferred_data) > 0:
            # Legacy format - list with single entry
            metadata["num_topics"] = inferred_data[0].get("number_topics", 0)
            metadata["explanation"] = inferred_data[0].get("explanation", "")
            metadata["collection_summary"] = ""
            metadata["collection_topics"] = []
            
            for topic in inferred_data[0].get("topics", []):
                predefined_topics.append({
                    "topic": topic.get("topic", ""),
                    "description": topic.get("description", ""),
                    "keywords": topic.get("keywords", [])
                })
        
        self.predefined_topics = predefined_topics
        self.metadata = metadata
        
        print(f"Loaded {len(predefined_topics)} predefined topics for classification")
        self.results['topics_loaded'] = {
            'count': len(predefined_topics),
            'metadata': metadata,
            'topics': predefined_topics
        }
        
        return predefined_topics, metadata
    
    def load_articles(self, articles_file: str, min_word_count: int = 30) -> List[Dict]:
        """
        Load and filter articles for classification.
        
        Args:
            articles_file: Path to the articles JSON file
            min_word_count: Minimum word count for article inclusion
            
        Returns:
            List of filtered articles
        """
        print(f"Loading articles from: {articles_file}")
        
        try:
            with open(articles_file, "r", encoding="utf-8") as f:
                articles = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Articles file not found: {articles_file}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in articles file: {articles_file}")
        
        # Filter by status code and word count
        filtered_articles = [
            entry for entry in articles
            if entry.get("status_code") == 200 and 
            len(entry.get("data", {}).get("content", "").split()) > min_word_count
        ]
        
        self.articles = filtered_articles
        
        print(f"Loaded {len(filtered_articles)} articles for classification (filtered from {len(articles)} total)")
        self.results['articles_loaded'] = {
            'total_articles': len(articles),
            'filtered_articles': len(filtered_articles),
            'min_word_count': min_word_count
        }
        
        return filtered_articles
    
    def build_classification_prompt(self, 
                                  article_text: str, 
                                  article_url: str, 
                                  prompt_template: str) -> str:
        """
        Build classification prompt for a single article.
        
        Args:
            article_text: Article content
            article_url: Article URL
            prompt_template: Template for classification prompt
            
        Returns:
            Formatted prompt string
        """
        # Create formatted topics text
        topics_text = "\n\n".join([
            f"TOPIC: {t['topic']}\nDESCRIPTION: {t['description']}\nKEYWORDS: {', '.join(t['keywords'])}"
            for t in self.predefined_topics
        ])
        
        # Build final prompt using string replacement
        final_prompt = prompt_template.replace("{topics}", topics_text)
        final_prompt = final_prompt.replace("{collection_summary}", self.metadata.get("collection_summary", ""))
        final_prompt = final_prompt.replace("{collection_topics}", ", ".join(self.metadata.get("collection_topics", [])))
        final_prompt = final_prompt.replace("{num_topics}", str(self.metadata.get("num_topics", 0)))
        final_prompt = final_prompt.replace("{explanation}", self.metadata.get("explanation", ""))
        final_prompt = final_prompt.replace("{url}", article_url)
        final_prompt = final_prompt.replace("{input}", article_text)
        
        return final_prompt
    
    def classify_single_article(self, 
                              article: Dict, 
                              prompt_template: str) -> Dict[str, Any]:
        """
        Classify a single article using the predefined topics.
        
        Args:
            article: Article dictionary with URL and content
            prompt_template: Template for classification prompt
            
        Returns:
            Classification result dictionary
        """
        url = article["url"]
        text = article["data"]["content"]
        
        # Build prompt
        final_prompt = self.build_classification_prompt(text, url, prompt_template)
        
        # Count input tokens
        input_tokens = len(self.encoding.encode(final_prompt))
        
        # Get LLM response
        print(f"Classifying article: {url}")
        response = self.llm.invoke(final_prompt)
        
        # Count output tokens
        output_tokens = len(self.encoding.encode(response.content))
        total_tokens = input_tokens + output_tokens
        
        # Track tokens
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        # Parse response
        try:
            response_json = self._parse_llm_response(response.content)
        except Exception as e:
            print(f"Error parsing response for {url}: {e}")
            response_json = {"error": str(e), "raw_response": response.content}
        
        # Create result
        result = {
            "url": url,
            "text_snippet": text[:200],  # First 200 characters
            "response": response_json,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }
        }
        
        return result
    
    def _parse_llm_response(self, response_content: str) -> Dict[str, Any]:
        """
        Parse the LLM response, handling both direct JSON and markdown-wrapped JSON.
        
        Args:
            response_content: Raw response from LLM
            
        Returns:
            Parsed JSON response as dictionary
        """
        try:
            return json.loads(response_content)
        except json.JSONDecodeError:
            # If the response is wrapped in markdown code blocks, extract the JSON
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                raise ValueError("Could not parse JSON from response")
    
    def perform_topic_classification(self, 
                                   prompt_template: str,
                                   batch_save_interval: int = None,
                                   output_dir: str = None,
                                   classification_output_file: str = None) -> List[Dict[str, Any]]:
        """
        Perform topic classification on all loaded articles using the RAGTEC framework.
        
        Args:
            prompt_template: Template for classification prompt
            
        Returns:
            List of classification results
        """
        if not self.articles:
            raise ValueError("No articles loaded. Call load_articles() first.")
        
        if not self.predefined_topics:
            raise ValueError("No predefined topics loaded. Call load_predefined_topics() first.")
        
        print(f"Starting classification of {len(self.articles)} articles...")
        
        results = []
        
        # Process articles with progress bar and batch saving
        for i, article in enumerate(tqdm(self.articles, desc="Classifying articles"), 1):
            try:
                result = self.classify_single_article(article, prompt_template)
                results.append(result)
                
                # Brief status update
                if "response" in result and isinstance(result["response"], dict):
                    if "topics" in result["response"]:
                        topic_names = [topic.get("main_topic", "Unknown") for topic in result["response"]["topics"]]
                        print(f"  Assigned topics: {', '.join(topic_names)}")
                
                # Batch save every N articles
                if batch_save_interval and i % batch_save_interval == 0:
                    if output_dir and classification_output_file:
                        print(f"  Saving progress: {i}/{len(self.articles)} articles to {classification_output_file}")
                        try:
                            with open(classification_output_file, 'w', encoding='utf-8') as f:
                                json.dump(results, f, indent=2, ensure_ascii=False)
                        except Exception as save_error:
                            print(f"  Warning: Could not save progress file: {save_error}")
                
            except Exception as e:
                print(f"Error processing {article['url']}: {str(e)}")
                results.append({
                    "url": article["url"],
                    "error": str(e),
                    "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                })
        
        # Calculate costs
        total_tokens = self.total_input_tokens + self.total_output_tokens
        input_cost = (self.total_input_tokens / 1000) * 0.000150
        output_cost = (self.total_output_tokens / 1000) * 0.000600
        total_cost = input_cost + output_cost
        
        # Store results
        self.results['classification'] = {
            'total_articles': len(self.articles),
            'successful_classifications': len([r for r in results if "error" not in r]),
            'failed_classifications': len([r for r in results if "error" in r]),
            'results': results,
            'token_usage': {
                'input_tokens': self.total_input_tokens,
                'output_tokens': self.total_output_tokens,
                'total_tokens': total_tokens,
                'input_cost_usd': input_cost,
                'output_cost_usd': output_cost,
                'total_cost_usd': total_cost,
                'model': self.model,
                'pricing_date': '2024_rates'
            }
        }
        
        # Print summary
        print(f"\nClassification Summary:")
        print(f"  Input tokens: {self.total_input_tokens:,} (${input_cost:.6f})")
        print(f"  Output tokens: {self.total_output_tokens:,} (${output_cost:.6f})")
        print(f"  Total cost: ${total_cost:.6f}")
        
        return results
    
    def parse_and_save_results(self, 
                             results: List[Dict], 
                             output_file: str = None,
                             save_to_file: bool = True) -> Optional[List[Dict]]:
        """
        Parse classification results and optionally save to file.
        
        Args:
            results: List of classification results
            output_file: Path to save parsed results (required if save_to_file=True)
            save_to_file: Whether to save results to file
            
        Returns:
            Parsed results list
        """
        try:
            # Save to file if requested
            if save_to_file and output_file:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                print(f"Saved classification results to: {output_file}")
            elif save_to_file and not output_file:
                print("Warning: save_to_file=True but no output_file provided. Skipping save.")
            
            self.results['parsed_output'] = results
            return results
            
        except Exception as e:
            print(f"Could not save classification results: {e}")
            return None
    
    def save_all_results(self, output_dir: str, save_components: Dict[str, bool] = None):
        """
        Save selected results and metadata to files.
        
        Args:
            output_dir: Directory to save all results
            save_components: Dictionary specifying what to save:
                - 'comprehensive': Save complete results JSON
                - 'token_usage': Save token usage and costs
                - 'classification': Save classification results
                - 'analysis': Save analysis report
                If None, saves all components
        """
        if save_components is None:
            save_components = {
                'comprehensive': True,
                'token_usage': True,
                'classification': True,
                'analysis': True
            }
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        # Save comprehensive results
        if save_components.get('comprehensive', False):
            results_file = output_path / "ragtec_classification_complete_results.json"
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, default=str)
            saved_files.append(str(results_file))
        
        # Save token usage separately
        if save_components.get('token_usage', False) and 'classification' in self.results:
            token_file = output_path / "token_usage_and_cost.json"
            with open(token_file, "w", encoding="utf-8") as f:
                json.dump(self.results['classification']['token_usage'], f, indent=2)
            saved_files.append(str(token_file))
        
        # Save classification results separately
        if save_components.get('classification', False) and 'classification' in self.results:
            classification_file = output_path / "classification_results.json"
            with open(classification_file, "w", encoding="utf-8") as f:
                json.dump(self.results['classification']['results'], f, indent=2)
            saved_files.append(str(classification_file))
        
        if saved_files:
            print(f"Results saved to: {output_path}")
            for file in saved_files:
                print(f"  - {Path(file).name}")
        else:
            print("No files were saved (all save options disabled)")
    
    def print_classification_summary(self):
        """
        Print the classification summary if available from the results.
        """
        if 'classification' not in self.results:
            print("\nNo classification results available to display summary.")
            return
        
        classification_results = self.results['classification']
        
        print("\n" + "="*50)
        print("CLASSIFICATION SUMMARY")
        print("="*50)
        
        print(f"Total articles processed: {classification_results['total_articles']}")
        print(f"Successful classifications: {classification_results['successful_classifications']}")
        print(f"Failed classifications: {classification_results['failed_classifications']}")
        
        # Print token usage
        token_usage = classification_results['token_usage']
        print(f"\nToken Usage:")
        print(f"  Input tokens: {token_usage['input_tokens']:,}")
        print(f"  Output tokens: {token_usage['output_tokens']:,}")
        print(f"  Total tokens: {token_usage['total_tokens']:,}")
        print(f"  Total cost: ${token_usage['total_cost_usd']:.6f}")
        
        print("-" * 50)
    
    def run_complete_pipeline(self, 
                            articles_file: str,
                            topics_file: str,
                            prompt_template: str,
                            output_dir: str = None,
                            classification_output_file: str = None,
                            save_results: bool = True,
                            save_components: Dict[str, bool] = None,
                            batch_save_interval: int = 10) -> Dict[str, Any]:
        """
        Run the complete RAGTEC pipeline for topic classification.
        
        Args:
            articles_file: Path to articles JSON file
            topics_file: Path to extracted topics JSON file
            prompt_template: Template for classification prompt
            output_dir: Directory for saving results (optional if save_results=False)
            classification_output_file: Specific file for classification results (optional if save_results=False)
            save_results: Whether to save results to files
            save_components: Dictionary specifying what to save (if save_results=True)
            
        Returns:
            Complete results dictionary
        """
        print("=" * 60)
        print("STARTING RAGTEC TOPIC CLASSIFICATION")
        print("Retrieval-Augmented Generation for Topic Classification")
        print("=" * 60)
        
        # Step 1: Load predefined topics
        self.load_predefined_topics(topics_file)
        
        # Step 2: Load articles
        self.load_articles(articles_file)
        
        # Step 3: Perform topic classification
        classification_results = self.perform_topic_classification(
            prompt_template=prompt_template,
            batch_save_interval=batch_save_interval,
            output_dir=output_dir,
            classification_output_file=classification_output_file
        )
        
        # Step 4: Parse results (and optionally save)
        if save_results and classification_output_file:
            self.parse_and_save_results(classification_results, classification_output_file, save_to_file=True)
        else:
            self.parse_and_save_results(classification_results, save_to_file=False)
        
        # Step 5: Save additional results if requested
        if save_results and output_dir:
            self.save_all_results(output_dir, save_components)
        
        # Print classification summary
        self.print_classification_summary()
        
        print("\n" + "=" * 60)
        print("RAGTEC TOPIC CLASSIFICATION COMPLETED")
        print("Topic Classification Finished")
        if not save_results:
            print("(Results available in memory - not saved to files)")
        print("=" * 60)
        
        return self.results


# Convenience function for simple usage
def run_ragtec_topic_classification(articles_file: str,
                                  topics_file: str,
                                  prompt_path: str,
                                  output_dir: str = None,
                                  model: str = "gpt-4o-mini",
                                  save_results: bool = True,
                                  batch_save_interval: int = 10,
                                  dataset_name: str = None) -> Dict[str, Any]:
    """
    Convenience function to run RAGTEC topic classification with file paths.
    
    Args:
        articles_file: Path to articles JSON file
        topics_file: Path to extracted topics JSON file
        prompt_path: Path to prompt template file
        output_dir: Output directory (required if save_results=True)
        model: OpenAI model to use
        save_results: Whether to save results to files
        batch_save_interval: Save progress every N articles (default: 10)
        dataset_name: Name of the dataset to include in output filenames
        
    Returns:
        Complete results dictionary
    """
    # Load template
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    
    # Initialize and run
    classifier = RAGTECTopicClassification(model=model)
    
    if save_results and output_dir:
        # Generate filename with dataset name if provided
        if dataset_name:
            classification_output_file = os.path.join(output_dir, f"{dataset_name}_ragtec_classification_results.json")
        else:
            classification_output_file = os.path.join(output_dir, "classification_results.json")
    else:
        classification_output_file = None
    
    return classifier.run_complete_pipeline(
        articles_file=articles_file,
        topics_file=topics_file,
        prompt_template=prompt_template,
        output_dir=output_dir,
        classification_output_file=classification_output_file,
        save_results=save_results,
        batch_save_interval=batch_save_interval
    )


if __name__ == "__main__":
    # Example usage
    print("RAGTEC Topic Classification Module")
    print("This module should be imported and used by other scripts.")
    print("See the convenience function for usage examples.")
