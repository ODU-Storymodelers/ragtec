"""
RAGTEC: Retrieval-Augmented Generation for Topic Extraction and Classification

This module provides the topic extraction component of the RAGTEC framework 
with optimal k selection for news article analysis.

RAGTEC Topic Extraction Features:
- Automatic optimal k calculation using AlNoamany et al. formula
- Vector store creation and management for article retrieval
- Intelligent article retrieval with similarity scoring
- Topic extraction using RAG with custom prompts
- Token counting and cost analysis
- Comprehensive logging and result saving
"""

import os
import json
import tiktoken
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from tqdm import tqdm

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Local imports
from utils.optimal_k import calculate_optimal_k, get_selection_strategy


class RAGTECTopicExtraction:
    """
    RAGTEC: Retrieval-Augmented Generation for Topic Extraction
    
    This class implements the topic extraction component of the RAGTEC framework 
    which handles the complete pipeline for extracting topics from news articles, 
    including optimal k selection and comprehensive cost tracking.
    """
    
    def __init__(self, 
                 model: str = "gpt-4o-mini",
                 temperature: float = 0,
                 embedding_model: Optional[str] = None):
        """
        Initialize the RAGTEC Topic Extraction module.
        
        Args:
            model: OpenAI model name for topic extraction
            temperature: Temperature for LLM generation
            embedding_model: Embedding model name (defaults to OpenAI default)
        """
        self.model = model
        self.temperature = temperature
        self.embedding_model = embedding_model
        
        # Initialize components
        self.llm = ChatOpenAI(model_name=model, temperature=temperature)
        self.embeddings = OpenAIEmbeddings()
        self.encoding = tiktoken.encoding_for_model(model)
        
        # State tracking
        self.articles = []
        self.collection_size = 0
        self.optimal_k = 0
        self.vectorstore = None
        self.results = {}
        
    def load_articles(self, articles_path: str, separator: str = "\n" + "="*80 + "\n") -> List[str]:
        """
        Load and parse articles from a formatted text file.
        
        Args:
            articles_path: Path to the articles file
            separator: Separator used between articles
            
        Returns:
            List of article texts
        """
        print(f"Loading articles from: {articles_path}")
        
        with open(articles_path, "r", encoding="utf-8") as f:
            articles_text = f.read()
        
        self.articles = [a.strip() for a in articles_text.split(separator) if a.strip()]
        self.collection_size = len(self.articles)
        
        print(f"Loaded {self.collection_size} articles.")
        return self.articles
    
    def calculate_optimal_selection(self, custom_k: int = None) -> Dict[str, Any]:
        """
        Calculate optimal k and get selection strategy information.
        
        Args:
            custom_k: Custom k value to use instead of formula-based optimal_k
        
        Returns:
            Dictionary with selection strategy details
        """
        if not self.articles:
            raise ValueError("No articles loaded. Call load_articles() first.")
        
        if custom_k is not None:
            # Use custom k value
            self.optimal_k = min(custom_k, self.collection_size)  # Ensure k doesn't exceed collection size
            selection_strategy = {
                'strategy_type': 'Custom Selection',
                'percentage_selected': (self.optimal_k / self.collection_size) * 100,
                'description': f'User-specified k={custom_k}',
                'formula_used': f'Custom k={custom_k}' + (f' (capped to {self.optimal_k})' if custom_k > self.collection_size else '')
            }
            print(f"Using custom k: {custom_k} (capped at collection size: {self.optimal_k})")
        else:
            # Use formula-based optimal k
            self.optimal_k = calculate_optimal_k(self.collection_size)
            selection_strategy = get_selection_strategy(self.collection_size)
        
        analysis_info = {
            'collection_size': self.collection_size,
            'optimal_k': self.optimal_k,
            'selection_strategy': selection_strategy
        }
        
        # Print analysis summary
        print(f"\nCollection Analysis:")
        print(f"  Total articles: {self.collection_size}")
        print(f"  Optimal k: {self.optimal_k} articles ({selection_strategy['percentage_selected']:.1f}% selection)")
        print(f"  Strategy: {selection_strategy['strategy_type']}")
        print(f"  Formula used: {selection_strategy['formula_used']}")
        
        self.results['analysis'] = analysis_info
        return analysis_info
    
    def build_vectorstore(self, persist_directory: Optional[str] = None) -> Chroma:
        """
        Build vector store from loaded articles.
        
        Args:
            persist_directory: Optional directory to persist the vector store
            
        Returns:
            Chroma vector store instance
        """
        if not self.articles:
            raise ValueError("No articles loaded. Call load_articles() first.")
        
        print("Building vector store...")
        
        # Create documents with metadata
        docs = [
            Document(
                page_content=article, 
                metadata={"index": idx, "length": len(article)}
            ) 
            for idx, article in enumerate(self.articles)
        ]
        
        # Build vector store
        if persist_directory:
            self.vectorstore = Chroma.from_documents(
                documents=docs, 
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
        else:
            self.vectorstore = Chroma.from_documents(
                documents=docs, 
                embedding=self.embeddings
            )
        
        print(f"Vector store created with {len(docs)} documents.")
        return self.vectorstore
    
    def retrieve_relevant_articles(self, 
                                 query: str, 
                                 k: Optional[int] = None) -> Tuple[List[Document], List[float]]:
        """
        Retrieve most relevant articles for the query.
        
        Args:
            query: Query for article retrieval
            k: Number of articles to retrieve (uses optimal_k if None)
            
        Returns:
            Tuple of (documents, similarity_scores)
        """
        if not self.vectorstore:
            raise ValueError("Vector store not built. Call build_vectorstore() first.")
        
        retrieval_k = k if k is not None else self.optimal_k
        if retrieval_k == 0:
            raise ValueError("Optimal k not calculated. Call calculate_optimal_selection() first.")
        
        print(f"Retrieving {retrieval_k} most relevant articles...")
        
        # Perform similarity search with cosine similarity scores
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query, 
            k=retrieval_k
        )
        
        docs = [doc for doc, score in docs_with_scores]
        # Convert L2 distances to cosine similarity scores (higher = more similar)
        # For cosine similarity: similarity = 1 - (L2_distance / 2)
        cosine_scores = [1 - (score / 2) for doc, score in docs_with_scores]
        
        # Store retrieval results
        retrieval_info = []
        for doc, cosine_score in zip(docs, cosine_scores):
            retrieval_info.append({
                "index": doc.metadata.get("index"),
                "cosine_similarity": float(cosine_score),
                "content_length": len(doc.page_content),
                "content": doc.page_content
            })
        
        self.results['retrieval'] = {
            'query': query,
            'k_retrieved': retrieval_k,
            'articles': retrieval_info,
            'average_score': sum(cosine_scores) / len(cosine_scores) if cosine_scores else 0,
            'score_range': (min(cosine_scores), max(cosine_scores)) if cosine_scores else (0, 0)
        }
        
        print(f"Retrieved {len(docs)} articles with cosine similarity scores: {min(cosine_scores):.4f} - {max(cosine_scores):.4f}")
        
        return docs, cosine_scores
    
    def build_context(self, 
                     docs: List[Document], 
                     scores: List[float], 
                     include_scores: bool = True) -> str:
        """
        Build context string from retrieved documents.
        
        Args:
            docs: Retrieved documents
            scores: Similarity scores
            include_scores: Whether to include similarity scores in context
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for doc, score in zip(docs, scores):
            if include_scores:
                context_parts.append(f"[Cosine Similarity: {score:.4f}]\n{doc.page_content}")
            else:
                context_parts.append(doc.page_content)
        
        context = "\n\n" + "="*50 + "\n\n".join(context_parts)
        
        # Count tokens
        token_count = len(self.encoding.encode(context))
        print(f"Context built: {token_count:,} tokens")
        
        self.results['context'] = {
            'token_count': token_count,
            'article_count': len(docs),
            'include_scores': include_scores
        }
        
        return context
    
    def perform_topic_modeling(self, 
                             prompt_template: str, 
                             context: str) -> Dict[str, Any]:
        """
        Perform topic extraction using the RAGTEC framework.
        
        Args:
            prompt_template: Template with {context} placeholder for topic extraction
            context: Retrieved articles context
            
        Returns:
            Dictionary with response and token usage information
        """
        # Build final prompt
        final_prompt = prompt_template.replace("{context}", context)
        
        # Count input tokens
        input_tokens = len(self.encoding.encode(final_prompt))
        print(f"Final prompt: {input_tokens:,} tokens")
        
        # Call LLM
        print("Calling LLM for topic extraction...")
        for _ in tqdm(range(1), desc="Processing"):
            response = self.llm.invoke(final_prompt)
        
        # Count output tokens
        output_tokens = len(self.encoding.encode(response.content))
        total_tokens = input_tokens + output_tokens
        
        # Calculate costs (GPT-4o-mini pricing)
        input_cost = (input_tokens / 1000) * 0.000150
        output_cost = (output_tokens / 1000) * 0.000600
        total_cost = input_cost + output_cost
        
        # Store results
        modeling_results = {
            'response': response.content,
            'token_usage': {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'input_cost_usd': input_cost,
                'output_cost_usd': output_cost,
                'total_cost_usd': total_cost,
                'model': self.model,
                'pricing_date': '2024_rates'
            }
        }
        
        # Print summary
        print(f"\nToken Usage Summary:")
        print(f"  Input tokens: {input_tokens:,} (${input_cost:.6f})")
        print(f"  Output tokens: {output_tokens:,} (${output_cost:.6f})")
        print(f"  Total cost: ${total_cost:.6f}")
        
        self.results['modeling'] = modeling_results
        return modeling_results
    
    def parse_and_save_results(self, 
                             response_content: str, 
                             output_file: str = None,
                             save_to_file: bool = True) -> Optional[Dict]:
        """
        Parse LLM response as JSON and optionally save results.
        
        Args:
            response_content: LLM response text
            output_file: Path to save parsed JSON (required if save_to_file=True)
            save_to_file: Whether to save results to file
            
        Returns:
            Parsed JSON dictionary or None if parsing failed
        """
        try:
            response_text = response_content.strip()
            
            # Handle markdown code blocks
            if "```json" in response_text:
                start_marker = "```json"
                end_marker = "```"
                start_idx = response_text.find(start_marker) + len(start_marker)
                end_idx = response_text.find(end_marker, start_idx)
                json_content = response_text[start_idx:end_idx].strip()
            else:
                json_content = response_text
            
            # Parse JSON
            result_json = json.loads(json_content)
            
            # Save to file if requested
            if save_to_file and output_file:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result_json, f, indent=2)
                print(f"Saved topic extraction results to: {output_file}")
            elif save_to_file and not output_file:
                print("Warning: save_to_file=True but no output_file provided. Skipping save.")
            
            self.results['parsed_output'] = result_json
            return result_json
            
        except Exception as e:
            print(f"Could not parse LLM response as JSON: {e}")
            
            if save_to_file and output_file:
                print("Saving raw response instead...")
                # Save raw response as fallback
                raw_file = output_file.replace('.json', '_raw.txt')
                with open(raw_file, "w", encoding="utf-8") as f:
                    f.write(response_content)
                print(f"Saved raw response to: {raw_file}")
            
            return None
    
    def save_all_results(self, output_dir: str, save_components: Dict[str, bool] = None, dataset_name: str = None):
        """
        Save selected results and metadata to files.
        
        Args:
            output_dir: Directory to save all results
            save_components: Dictionary specifying what to save:
                - 'comprehensive': Save complete results JSON
                - 'token_usage': Save token usage and costs
                - 'retrieval': Save retrieval results with scores  
                - 'analysis': Save collection analysis
                If None, saves all components
            dataset_name: Dataset name for file naming (optional)
        """
        if save_components is None:
            save_components = {
                'comprehensive': True,
                'token_usage': True,
                'retrieval': True,
                'analysis': True
            }
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine filename prefix
        file_prefix = f"{dataset_name}_" if dataset_name else ""
        
        saved_files = []
        
        # Save comprehensive results
        if save_components.get('comprehensive', False):
            results_file = output_path / f"{file_prefix}ragtec_complete_results.json"
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, default=str)
            saved_files.append(str(results_file))
        
        # Save token usage separately
        if save_components.get('token_usage', False) and 'modeling' in self.results:
            token_file = output_path / f"{file_prefix}token_usage_and_cost.json"
            with open(token_file, "w", encoding="utf-8") as f:
                json.dump(self.results['modeling']['token_usage'], f, indent=2)
            saved_files.append(str(token_file))
        
        # Save retrieval results separately
        if save_components.get('retrieval', False) and 'retrieval' in self.results:
            retrieval_file = output_path / f"{file_prefix}retrieved_articles_with_scores.json"
            with open(retrieval_file, "w", encoding="utf-8") as f:
                json.dump(self.results['retrieval']['articles'], f, indent=2)
            saved_files.append(str(retrieval_file))
        
        # Save analysis results separately
        if save_components.get('analysis', False) and 'analysis' in self.results:
            analysis_file = output_path / f"{file_prefix}collection_analysis.json"
            with open(analysis_file, "w", encoding="utf-8") as f:
                json.dump(self.results['analysis'], f, indent=2, default=str)
            saved_files.append(str(analysis_file))
        
        if saved_files:
            print(f"Results saved to: {output_path}")
            for file in saved_files:
                print(f"  - {Path(file).name}")
        else:
            print("No files were saved (all save options disabled)")
    
    def print_collection_summary(self):
        """
        Print the collection summary and topics if available from the parsed results.
        """
        if 'parsed_output' not in self.results:
            print("\nNo parsed results available to display summary.")
            return
        
        parsed_results = self.results['parsed_output']
        
        print("\n" + "="*50)
        print("COLLECTION SUMMARY")
        print("="*50)
        
        # Print collection summary
        if 'collection_summary' in parsed_results:
            print(f"Summary: {parsed_results['collection_summary']}")
        
        # Print number of topics identified
        if 'number_topics' in parsed_results:
            print(f"\nTopics identified: {parsed_results['number_topics']}")
        
        # Print explanation of approach
        if 'explanation' in parsed_results:
            print(f"Approach: {parsed_results['explanation']}")
        
        # Print collection topics list
        if 'collection_topics' in parsed_results and parsed_results['collection_topics']:
            print(f"\nIdentified Topics:")
            for i, topic in enumerate(parsed_results['collection_topics'], 1):
                print(f"  {i}. {topic}")
        
        # Print detailed topic information
        if 'topics' in parsed_results and parsed_results['topics']:
            print(f"\nTOPIC DETAILS:")
            print("-" * 50)
            for i, topic in enumerate(parsed_results['topics'], 1):
                print(f"\n{i}. {topic.get('topic', 'Unknown Topic')}")
                print(f"   Description: {topic.get('description', 'No description')}")
                if 'keywords' in topic and topic['keywords']:
                    keywords_str = ', '.join(topic['keywords'][:5])  # Show first 5 keywords
                    if len(topic['keywords']) > 5:
                        keywords_str += f" (and {len(topic['keywords']) - 5} more)"
                    print(f"   Keywords: {keywords_str}")
                if 'representative_documents' in topic and topic['representative_documents']:
                    doc_count = len(topic['representative_documents'])
                    print(f"   Representative documents: {doc_count}")
        
        print("-" * 50)
    
    def run_complete_pipeline(self, 
                            articles_path: str,
                            prompt_template: str,
                            query: str,
                            output_dir: str = None,
                            topic_output_file: str = None,
                            save_results: bool = True,
                            save_components: Dict[str, bool] = None,
                            dataset_name: str = None,
                            custom_k: int = None) -> Dict[str, Any]:
        """
        Run the complete RAGTEC pipeline for topic extraction.
        
        Args:
            articles_path: Path to articles file
            prompt_template: Template for topic extraction prompt
            query: Query for article retrieval
            output_dir: Directory for saving results (optional if save_results=False)
            topic_output_file: Specific file for topic results (optional if save_results=False)
            save_results: Whether to save results to files
            save_components: Dictionary specifying what to save (if save_results=True)
            dataset_name: Dataset name for file naming (optional)
            custom_k: Custom k value for article selection (overrides optimal_k formula if provided)
            
        Returns:
            Complete results dictionary
        """
        print("=" * 60)
        print("STARTING RAGTEC TOPIC EXTRACTION")
        print("Retrieval-Augmented Generation for Topic Extraction")
        print("=" * 60)
        
        # Step 1: Load articles
        self.load_articles(articles_path)
        
        # Step 2: Calculate optimal selection
        self.calculate_optimal_selection(custom_k)
        
        # Step 3: Build vector store
        self.build_vectorstore()
        
        # Step 4: Retrieve relevant articles
        docs, scores = self.retrieve_relevant_articles(query)
        
        # Step 5: Build context
        context = self.build_context(docs, scores)
        
        # Step 6: Perform topic modeling
        modeling_results = self.perform_topic_modeling(prompt_template, context)
        
        # Step 7: Parse results (and optionally save)
        if save_results and topic_output_file:
            self.parse_and_save_results(modeling_results['response'], topic_output_file, save_to_file=True)
        else:
            self.parse_and_save_results(modeling_results['response'], save_to_file=False)
        
        # Step 8: Save additional results if requested
        if save_results and output_dir:
            self.save_all_results(output_dir, save_components, dataset_name)
        
        # Print collection summary if available
        self.print_collection_summary()
        
        print("\n" + "=" * 60)
        print("RAGTEC TOPIC EXTRACTION COMPLETED")
        print("Topic Extraction Finished")
        if not save_results:
            print("(Results available in memory - not saved to files)")
        print("=" * 60)
        
        return self.results


# Convenience function for simple usage
def run_ragtec_topic_modeling(articles_path: str,
                             prompt_path: str,
                             query_path: str,
                             output_dir: str = None,
                             model: str = "gpt-4o-mini",
                             save_results: bool = True,
                             dataset_name: str = None,
                             custom_k: int = None) -> Dict[str, Any]:
    """
    Convenience function to run RAGTEC topic extraction with file paths.
    
    Args:
        articles_path: Path to articles file
        prompt_path: Path to prompt template file
        query_path: Path to query file
        output_dir: Output directory (required if save_results=True)
        model: OpenAI model to use
        save_results: Whether to save results to files
        dataset_name: Name of the dataset/region for prompt customization
        custom_k: Custom k value for article selection (overrides optimal_k formula if provided)
        
    Returns:        Complete results dictionary
    """
    # Load templates
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    
    # Auto-detect dataset name from articles_path if not provided
    if dataset_name is None:
        import re
        # Extract dataset name from path like "../data/drc/drc_articles_formatted.txt"
        match = re.search(r'/([^/]+)/[^/]*articles', articles_path)
        if match:
            dataset_name = match.group(1).upper()
        else:
            dataset_name = "the region"
    else:
        dataset_name = dataset_name.upper()
    
    # Replace placeholder in prompt template
    prompt_template = prompt_template.replace("{dataset_region}", dataset_name)
    
    with open(query_path, "r", encoding="utf-8") as f:
        query = f.read().strip()
    
    # Replace placeholder in query template
    query = query.replace("{dataset_region}", dataset_name)
    
    # Initialize and run
    modeler = RAGTECTopicExtraction(model=model)
    
    # Extract dataset name for file naming (use the raw dataset_name, not the uppercase version)
    if dataset_name is None:
        import re
        match = re.search(r'/([^/]+)/[^/]*articles', articles_path)
        file_dataset_name = match.group(1) if match else "dataset"
    else:
        # Use the original parameter value for filenames (before converting to uppercase)
        file_dataset_name = dataset_name.lower()
    
    if save_results and output_dir:
        topic_output_file = os.path.join(output_dir, f"{file_dataset_name}_ragtec_topic_results.json")
    else:
        topic_output_file = None
    
    return modeler.run_complete_pipeline(
        articles_path=articles_path,
        prompt_template=prompt_template,
        query=query,
        output_dir=output_dir,
        topic_output_file=topic_output_file,
        save_results=save_results,
        dataset_name=file_dataset_name,  # Pass dataset name for other file naming
        custom_k=custom_k  # Pass custom_k parameter
    )
