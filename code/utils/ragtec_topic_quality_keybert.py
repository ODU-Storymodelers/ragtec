"""
RAGTEC: Retrieval-Augmented Generation for Topic Extraction and Classification

This module provides the quality assessment component of the RAGTEC framework 
for evaluating topic modeling results from news article analysis using KeyBERT 
for keyword extraction from document content.

RAGTEC Quality Assessment Features:
- KeyBERT-based keyword extraction from document content
- Comprehensive coherence metrics (CV and UMass)  
- Topic diversity measures (Jaccard and Unique Word Proportion)
- Statistical analysis of topic characteristics
- Standardized result storage and export
- Multi-topic document assignment support
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict, Counter

# Gensim imports for topic quality metrics
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

# KeyBERT imports for keyword extraction
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer


class RAGTECTopicQualityKeyBert:
    """
    RAGTEC: Retrieval-Augmented Generation for Topic Quality Assessment with KeyBERT
    
    This class implements the quality assessment component of the RAGTEC framework 
    using KeyBERT for keyword extraction from document content, rather than using
    LLM-generated keywords.
    """
    
    def __init__(self, 
                 extraction_file: str,
                 classification_file: str,
                 articles_file: Optional[str] = None,
                 model_name: str = 'all-MiniLM-L6-v2',
                 top_k_keywords: int = 10,
                 ngram_range: Tuple[int, int] = (1, 1)):
        """
        Initialize the RAGTEC Topic Quality Assessment module with KeyBERT.
        
        Args:
            extraction_file: Path to the topic extraction results JSON file (contains master topic list)
            classification_file: Path to the classification results JSON file with document-topic assignments
            articles_file: Path to the formatted articles text file for coherence calculation (optional)
            model_name: Sentence transformer model name for KeyBERT. Popular options:
                       - 'all-MiniLM-L6-v2' (default, fast and lightweight)
                       - 'all-mpnet-base-v2' (higher quality, slower)
                       - 'paraphrase-MiniLM-L6-v2' (good for paraphrases)
                       - 'multi-qa-MiniLM-L6-cos-v1' (optimized for question-answering)
            top_k_keywords: Number of keywords to extract per topic (default: 10)
            ngram_range: Tuple specifying n-gram range for keyword extraction 
                        (1,1)=unigrams only, (1,2)=unigrams+bigrams, (2,2)=bigrams only
        """
        self.extraction_file = extraction_file
        self.classification_file = classification_file
        self.articles_file = articles_file
        self.model_name = model_name
        self.top_k_keywords = top_k_keywords
        self.ngram_range = ngram_range
        
        # Extract dataset name from file path for summary file naming
        self.dataset_name = os.path.basename(extraction_file).split('_')[0] if extraction_file else "unknown"
        
        # Initialize KeyBERT model
        try:
            self.keybert_model = KeyBERT(model=SentenceTransformer(model_name))
            print(f"Initialized KeyBERT with model: {model_name}")
        except ImportError:
            print("Warning: KeyBERT not available. Install with: pip install keybert")
            self.keybert_model = None
        
        # State tracking
        self.extraction_data = {}  # Master topic list from extraction
        self.master_topics = []    # List of topic names from extraction
        self.classification_data = {}
        self.topic_documents = {}  # Documents grouped by topic
        self.document_topic_assignments = {}
        self.topic_keywords = {}
        self.topic_counts = {}
        self.article_texts = []
        self.collection_size = 0
        
        # Data processing components
        self.dictionary = None
        self.corpus = None
        self.tokenized_texts = []
        self.topic_word_lists = []
        
        # Results storage
        self.results = {}
        
    def load_extraction_data(self) -> Dict[str, Any]:
        """
        Load the topic extraction results to get the master list of topics.
        
        Returns:
            Dictionary containing extraction results with master topic list
        """
        print(f"Loading topic extraction results from: {self.extraction_file}")
        
        with open(self.extraction_file, 'r', encoding='utf-8') as f:
            self.extraction_data = json.load(f)
        
        # Extract master topic list from the extraction results
        if 'topics' in self.extraction_data:
            self.master_topics = [topic['topic'] for topic in self.extraction_data['topics']]
            print(f"Found {len(self.master_topics)} master topics: {', '.join(self.master_topics)}")
        elif 'collection_topics' in self.extraction_data:
            self.master_topics = self.extraction_data['collection_topics']
            print(f"Found {len(self.master_topics)} collection topics: {', '.join(self.master_topics)}")
        else:
            raise ValueError("No 'topics' or 'collection_topics' field found in extraction data")
        
        return self.extraction_data

    def load_classification_data(self) -> Dict[str, Any]:
        """
        Load classification results and group documents by topics using only the topics field.
        Each document can belong to multiple topics based on its response.topics array.
        
        Returns:
            Dictionary containing classification results with document-topic assignments
        """
        if not self.master_topics:
            raise ValueError("Master topics not loaded. Call load_extraction_data() first.")
            
        print(f"Loading classification results from: {self.classification_file}")
        
        with open(self.classification_file, 'r', encoding='utf-8') as f:
            self.classification_data = json.load(f)
        
        # Initialize topic document groups
        self.topic_documents = {topic: [] for topic in self.master_topics}
        self.document_topic_assignments = {}
        
        processed_docs = 0
        skipped_docs = 0
        
        for idx, doc in enumerate(self.classification_data):
            # Check if response exists and has topics field
            if ('response' in doc and 
                isinstance(doc['response'], dict) and 
                'topics' in doc['response'] and
                isinstance(doc['response']['topics'], list)):
                
                doc_topics = []
                
                # Extract topic names from the topics array
                for topic_item in doc['response']['topics']:
                    if isinstance(topic_item, dict) and 'main_topic' in topic_item:
                        topic_name = topic_item['main_topic'].strip()
                        # Only include topics that are in our master list
                        if topic_name in self.master_topics:
                            doc_topics.append(topic_name)
                    elif isinstance(topic_item, str):
                        topic_name = topic_item.strip()
                        if topic_name in self.master_topics:
                            doc_topics.append(topic_name)
                
                if doc_topics:
                    # Document belongs to multiple topics
                    self.document_topic_assignments[idx] = doc_topics
                    
                    # Extract document content
                    content = ""
                    if 'text_snippet' in doc:
                        content = doc['text_snippet']
                    elif 'content' in doc:
                        content = doc['content']
                    elif 'data' in doc and isinstance(doc['data'], dict) and 'content' in doc['data']:
                        content = doc['data']['content']
                    elif 'text' in doc:
                        content = doc['text']
                    
                    if content and content.strip():
                        doc_info = {
                            'idx': idx,
                            'content': content.strip(),
                            'topics': doc_topics,
                            'url': doc.get('url', ''),
                        }
                        
                        # Add this document to each topic it belongs to
                        for topic in doc_topics:
                            self.topic_documents[topic].append(doc_info)
                        
                        processed_docs += 1
                    else:
                        skipped_docs += 1
                        print(f"  Warning: Document {idx} has no content, skipping")
                else:
                    skipped_docs += 1
            else:
                skipped_docs += 1
        
        print(f"Processed {processed_docs} documents, skipped {skipped_docs} documents")
        
        # Print topic distribution
        print("\nTopic Distribution:")
        for topic, docs in self.topic_documents.items():
            print(f"  {topic}: {len(docs)} documents")
        
        return self.classification_data
    
    def extract_keywords_with_keybert(self) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        """
        Extract keywords from document content using KeyBERT for each topic.
        Groups documents by topic and extracts representative keywords.
        
        Returns:
            Tuple of (topic_keywords, topic_counts)
        """
        if not self.topic_documents:
            raise ValueError("No topic documents loaded. Call load_classification_data() first.")
        
        if not self.keybert_model:
            raise ValueError("KeyBERT model not available. Install with: pip install keybert")
        
        print(f"Extracting keywords using KeyBERT for {len(self.master_topics)} topics...")
        
        topic_keywords = {}
        topic_counts = {}
        
        for topic in self.master_topics:
            docs = self.topic_documents.get(topic, [])
            
            if not docs:
                print(f"  Warning: No documents found for topic '{topic}', skipping")
                topic_keywords[topic] = []
                topic_counts[topic] = 0
                continue
            
            # Combine all document content for this topic
            combined_content = " ".join([doc['content'] for doc in docs])
            
            if not combined_content.strip():
                print(f"  Warning: Empty content for topic '{topic}', skipping")
                topic_keywords[topic] = []
                topic_counts[topic] = 0
                continue
            
            try:
                # Extract keywords using KeyBERT with compatible parameter handling
                try:
                    # Try the newer API first (with top_n parameter)
                    keywords_with_scores = self.keybert_model.extract_keywords(
                        combined_content,
                        keyphrase_ngram_range=self.ngram_range,  # User-configurable n-gram range
                        stop_words='english',
                        use_maxsum=True,  # Use MaxSum for diversity
                        nr_candidates=20,  # Consider more candidates
                        top_n=self.top_k_keywords  # Use top_n instead of top_k
                    )
                except TypeError as api_error:
                    # If top_n/top_k parameter fails, try legacy API with minimal parameters
                    if "unexpected keyword argument 'top_k'" in str(api_error) or "unexpected keyword argument 'top_n'" in str(api_error):
                        print(f"    Using legacy KeyBERT API - extracting {self.top_k_keywords} keywords")
                        
                        # Try without use_maxsum and with fewer parameters for legacy compatibility
                        try:
                            keywords_with_scores = self.keybert_model.extract_keywords(
                                combined_content,
                                keyphrase_ngram_range=self.ngram_range,
                                stop_words='english',
                                top_n=self.top_k_keywords
                            )
                            print(f"      Legacy API (minimal params) extracted {len(keywords_with_scores)} keywords")
                        except:
                            # Even more minimal approach - just the content
                            keywords_with_scores = self.keybert_model.extract_keywords(combined_content, top_n=self.top_k_keywords)
                            print(f"      Legacy API (basic) extracted {len(keywords_with_scores)} keywords")
                        
                        # Manually limit to top_k results (redundant, but keep for safety)
                        if len(keywords_with_scores) > self.top_k_keywords:
                            keywords_with_scores = keywords_with_scores[:self.top_k_keywords]
                            print(f"      Trimmed to top {len(keywords_with_scores)} keywords")
                        else:
                            print(f"      Kept all {len(keywords_with_scores)} keywords")
                    else:
                        # Re-raise if it's a different TypeError
                        raise api_error
                
                # Extract just the keywords (not scores)
                keywords = [kw[0] for kw in keywords_with_scores]
                
                topic_keywords[topic] = keywords
                topic_counts[topic] = len(keywords)
                
                print(f"  {topic}: extracted {len(keywords)} keywords from {len(docs)} documents")
                print(f"    Top keywords: {', '.join(keywords[:5])}")
                
            except Exception as e:
                print(f"  Error extracting keywords for topic '{topic}': {e}")
                topic_keywords[topic] = []
                topic_counts[topic] = 0
        
        # Store in class attributes
        self.topic_keywords = topic_keywords
        self.topic_counts = topic_counts
        
        # Store in results
        self.results['extraction'] = {
            'num_topics': len(topic_keywords),
            'total_keywords': sum(len(keywords) for keywords in topic_keywords.values()),
            'avg_keywords_per_topic': np.mean([len(keywords) for keywords in topic_keywords.values()]) if topic_keywords else 0
        }
        
        print(f"Successfully extracted keywords for {len(topic_keywords)} topics using KeyBERT")
        
        return topic_keywords, topic_counts

    def prepare_data_for_coherence(self) -> Tuple[Optional[Dictionary], Optional[List], Optional[List], Optional[List]]:
        """
        Prepare data structures needed for coherence calculation using KeyBERT-extracted keywords.
        
        Returns:
            Tuple of (dictionary, corpus, tokenized_texts, topic_word_lists)
        """
        if not self.topic_keywords:
            raise ValueError("No topic keywords extracted. Call extract_keywords_with_keybert() first.")
        
        # Use topic-specific documents for coherence calculation
        all_topic_texts = []
        for topic, docs in self.topic_documents.items():
            if docs:  # Only include topics that have documents
                topic_texts = [doc['content'] for doc in docs]
                all_topic_texts.extend(topic_texts)
        
        if not all_topic_texts:
            raise ValueError("No document content available for coherence calculation")
        
        # Tokenize all documents
        print("Tokenizing texts...")
        tokenized_texts = []
        for text in all_topic_texts:
            if text.strip():
                # Simple tokenization - split by spaces and clean
                tokens = [token.lower().strip('.,!?;:"()[]{}') for token in text.split() 
                         if len(token.strip('.,!?;:"()[]{}')) > 2]
                if tokens:
                    tokenized_texts.append(tokens)
        
        if not tokenized_texts:
            raise ValueError("No valid tokenized texts after processing")
        
        # Build dictionary and corpus
        print("Building dictionary and corpus...")
        dictionary = Dictionary(tokenized_texts)
        original_dict_size = len(dictionary)
        print(f"Original dictionary size: {original_dict_size}")
        
        # Filter dictionary - less aggressive filtering to keep more keywords
        dictionary.filter_extremes(no_below=1, no_above=0.95)
        filtered_dict_size = len(dictionary)
        print(f"Filtered dictionary size: {filtered_dict_size} (removed {original_dict_size - filtered_dict_size} terms)")
        
        if len(dictionary) == 0:
            raise ValueError("Dictionary is empty after filtering")
        
        # Build corpus
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        print(f"Corpus size: {len(corpus)} documents")
        
        # Format topic keywords for coherence models
        topic_word_lists = []
        total_keywords = 0
        keywords_in_dict = 0
        
        print("\n--- KEYWORD VALIDATION DETAILS ---")
        
        for topic_name, keywords in self.topic_keywords.items():
            if not keywords:  # Skip empty topics
                continue
                
            valid_keywords = []
            invalid_keywords = []
            
            print(f"Topic {topic_name}: Processing {len(keywords)} KeyBERT-extracted keywords")
            print(f"  Original keywords: {', '.join(keywords)}")
            
            # Process keywords - simplified for unigram extraction
            for keyword in keywords:  # Process ALL KeyBERT keywords
                # For unigram extraction (ngram_range=(1,1)), keywords should be single words
                if keyword in dictionary.token2id:
                    valid_keywords.append(keyword)
                    keywords_in_dict += 1
                    print(f"    ✓ '{keyword}' - found in dictionary")
                else:
                    invalid_keywords.append(keyword)
                    print(f"    ✗ '{keyword}' - not found in dictionary")
                
                total_keywords += 1
            
            # Display validation results
            print(f"Topic {topic_name}:")
            print(f"  - Valid keywords ({len(valid_keywords)}): {', '.join(valid_keywords)}")
            if invalid_keywords:
                print(f"  - Invalid keywords ({len(invalid_keywords)}): {', '.join(invalid_keywords)}")
            
            # Use only top 10 valid keywords for coherence calculation (method comparison)
            if valid_keywords:
                top_10_keywords = valid_keywords[:10]  # Limit to top 10 for fair comparison
                topic_word_lists.append(top_10_keywords)
                print(f"  - Top 10 for coherence: {', '.join(top_10_keywords)}")
            else:
                print(f"  ! Warning: No valid keywords found in dictionary for this topic")
        
        print("\n--- END KEYWORD VALIDATION ---")
        
        if total_keywords > 0:
            percentage = (keywords_in_dict/total_keywords)*100
            print(f"Keywords in dictionary: {keywords_in_dict} out of {total_keywords} ({percentage:.1f}%)")
        else:
            print("No keywords found in topics")
        
        # Store in class attributes
        self.dictionary = dictionary
        self.corpus = corpus
        self.tokenized_texts = tokenized_texts
        self.topic_word_lists = topic_word_lists
        
        # Store in results
        self.results['preparation'] = {
            'original_dict_size': original_dict_size,
            'filtered_dict_size': filtered_dict_size,
            'corpus_size': len(corpus),
            'num_topic_word_lists': len(topic_word_lists),
            'keyword_coverage': keywords_in_dict / total_keywords if total_keywords > 0 else 0
        }
        
        return dictionary, corpus, tokenized_texts, topic_word_lists

    def calculate_coherence_metrics(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate CV and UMass coherence scores for KeyBERT-extracted keywords.
        
        Returns:
            Tuple of (cv_coherence, umass_coherence)
        """
        if not self.topic_word_lists or len(self.topic_word_lists) < 2:
            print("Warning: Not enough valid topics for coherence calculation.")
            return None, None
        
        if not self.dictionary or len(self.dictionary) == 0 or not self.corpus:
            print("Warning: Dictionary or corpus is empty. Cannot compute coherence.")
            return None, None
        
        texts_to_use = self.tokenized_texts
        print(f"Using {len(texts_to_use)} texts for coherence calculation")
        
        # Calculate CV coherence
        cv_score = None
        try:
            print("Calculating CV coherence...")
            print(f"Topics for coherence: {len(self.topic_word_lists)}")
            print(f"Dictionary size: {len(self.dictionary)}")
            
            coherence_cv = CoherenceModel(
                topics=self.topic_word_lists, 
                texts=texts_to_use, 
                dictionary=self.dictionary, 
                coherence='c_v',
                processes=1,
                window_size=20
            )
            cv_score = coherence_cv.get_coherence()
            
            # Check for invalid values
            if np.isnan(cv_score) or np.isinf(cv_score):
                print("Warning: CV coherence calculation resulted in invalid value (NaN/Inf).")
                cv_score = None
            else:
                print(f"Successfully calculated CV coherence score: {cv_score}")
                
        except Exception as e:
            print(f"Error calculating CV coherence: {e}")
            cv_score = None
        
        # Calculate UMass coherence
        umass_score = None
        try:
            print("Calculating UMass coherence...")
            
            coherence_umass = CoherenceModel(
                topics=self.topic_word_lists, 
                texts=texts_to_use,
                dictionary=self.dictionary, 
                coherence='u_mass',
                processes=1
            )
            umass_score = coherence_umass.get_coherence()
            
            # Check for invalid values
            if np.isnan(umass_score) or np.isinf(umass_score):
                print("Warning: UMass coherence calculation resulted in invalid value (NaN/Inf).")
                umass_score = None
            else:
                print(f"Successfully calculated UMass coherence score: {umass_score}")
                
        except Exception as e:
            print(f"Error calculating UMass coherence: {e}")
            umass_score = None
        
        # Store in results
        self.results['coherence'] = {
            'cv_coherence': cv_score,
            'umass_coherence': umass_score,
            'num_topics_used': len(self.topic_word_lists),
            'calculation_method': 'KeyBERT_extracted_keywords'
        }
        
        return cv_score, umass_score

    def calculate_unique_word_proportion(self, top_n: int = 10) -> float:
        """
        Compute topic diversity as the proportion of unique words across topics.
        
        Args:
            top_n: Number of top keywords to consider per topic
            
        Returns:
            Diversity score (proportion of unique words)
        """
        if not self.topic_keywords:
            return 0.0
            
        unique_words = set()
        total_words = 0
        
        for keywords in self.topic_keywords.values():
            words = keywords[:min(len(keywords), top_n)]
            unique_words.update(words)
            total_words += len(words)
        
        diversity_score = len(unique_words) / total_words if total_words > 0 else 0.0
        
        # Store in results
        if 'diversity' not in self.results:
            self.results['diversity'] = {}
        self.results['diversity']['uwp_diversity'] = diversity_score
        
        return diversity_score

    def calculate_jaccard_diversity(self, top_n: int = 10) -> Optional[float]:
        """
        Calculate Jaccard-based topic diversity (average distance between topics).
        
        Args:
            top_n: Number of top keywords to consider per topic
            
        Returns:
            Jaccard diversity score or None if calculation fails
        """
        try:
            if not self.topic_keywords:
                return None
                
            topic_terms = []
            for keywords in self.topic_keywords.values():
                topic_terms.append(set(keywords[:min(len(keywords), top_n)]))
            
            if len(topic_terms) < 2:
                return None
            
            jaccard_distances = []
            for i in range(len(topic_terms)):
                for j in range(i + 1, len(topic_terms)):
                    intersection = len(topic_terms[i] & topic_terms[j])
                    union = len(topic_terms[i] | topic_terms[j])
                    jaccard_distance = 1 - (intersection / union) if union > 0 else 1
                    jaccard_distances.append(jaccard_distance)
            
            diversity_score = sum(jaccard_distances) / len(jaccard_distances) if jaccard_distances else 0.0
            
            # Store in results
            if 'diversity' not in self.results:
                self.results['diversity'] = {}
            self.results['diversity']['jaccard_diversity'] = diversity_score
            
            return diversity_score
        
        except Exception as e:
            print(f"Error calculating Jaccard diversity: {e}")
            return None

    def calculate_statistical_metrics(self) -> Dict[str, float]:
        """
        Calculate additional statistical metrics for KeyBERT-extracted topics.
        
        Returns:
            Dictionary containing statistical metrics
        """
        if not self.topic_keywords:
            return {}
            
        metrics = {}
        
        # Keywords per topic statistics
        keyword_counts = [len(keywords) for keywords in self.topic_keywords.values()]
        metrics['avg_keywords_per_topic'] = np.mean(keyword_counts) if keyword_counts else 0
        metrics['min_keywords_per_topic'] = min(keyword_counts) if keyword_counts else 0
        metrics['max_keywords_per_topic'] = max(keyword_counts) if keyword_counts else 0
        
        # Topic name length statistics
        topic_name_lengths = [len(name.split()) for name in self.topic_keywords.keys()]
        metrics['avg_topic_name_length'] = np.mean(topic_name_lengths) if topic_name_lengths else 0
        metrics['min_topic_name_length'] = min(topic_name_lengths) if topic_name_lengths else 0
        metrics['max_topic_name_length'] = max(topic_name_lengths) if topic_name_lengths else 0
        
        # Document distribution statistics
        doc_counts = [len(docs) for docs in self.topic_documents.values()]
        metrics['avg_docs_per_topic'] = np.mean(doc_counts) if doc_counts else 0
        metrics['min_docs_per_topic'] = min(doc_counts) if doc_counts else 0
        metrics['max_docs_per_topic'] = max(doc_counts) if doc_counts else 0
        
        # Store in results
        self.results['statistics'] = metrics
        
        return metrics

    def run_complete_quality_assessment(self) -> Dict[str, Any]:
        """
        Run the complete quality assessment pipeline using KeyBERT extraction.
        
        Returns:
            Dictionary containing all calculated metrics
        """
        try:
            print("=== STARTING RAGTEC KEYBERT QUALITY ASSESSMENT ===")
            
            # Configuration for method comparison
            COMPARISON_TOP_N = 10  # Fixed number of keywords for fair comparison between methods
            print(f"Using top {COMPARISON_TOP_N} keywords per topic for diversity metrics (method comparison)")
            
            # Step 1: Load extraction data (master topics)
            print("\n1. Loading extraction data...")
            self.load_extraction_data()
            
            # Step 2: Load classification data (document assignments)
            print("\n2. Loading classification data...")
            self.load_classification_data()
            
            # Step 3: Extract keywords using KeyBERT
            print("\n3. Extracting keywords using KeyBERT...")
            self.extract_keywords_with_keybert()
            
            # Step 4: Prepare data for coherence calculation
            print("\n4. Preparing data for coherence calculation...")
            self.prepare_data_for_coherence()
            
            # Step 5: Calculate coherence metrics
            print("\n5. Calculating coherence metrics...")
            cv_coherence, umass_coherence = self.calculate_coherence_metrics()
            
            # Step 6: Calculate diversity metrics
            print("\n6. Calculating diversity metrics...")
            # Use fixed top_n for method comparison (not all extracted keywords)
            jaccard_diversity = self.calculate_jaccard_diversity(top_n=COMPARISON_TOP_N)
            uwp_diversity = self.calculate_unique_word_proportion(top_n=COMPARISON_TOP_N)
            
            print(f"Diversity metrics calculated using top {COMPARISON_TOP_N} keywords per topic for method comparison")
            
            # Step 7: Calculate statistical metrics
            print("\n7. Calculating statistical metrics...")
            statistical_metrics = self.calculate_statistical_metrics()
            
            # Compile final results with KeyBERT-specific information
            final_results = {
                'num_topics': len(self.master_topics),
                'cv_coherence': cv_coherence,
                'umass_coherence': umass_coherence,
                'jaccard_diversity': jaccard_diversity,
                'uwp_diversity': uwp_diversity,
                'method': 'KeyBERT_extraction',
                'model_name': self.model_name,
                'ngram_range': str(self.ngram_range),
                'top_k_keywords': self.top_k_keywords,
                'topic_document_counts': {topic: len(docs) for topic, docs in self.topic_documents.items()},
                **statistical_metrics
            }
            
            self.results['final'] = final_results
            
            print("\n=== QUALITY ASSESSMENT COMPLETED ===")
            return final_results
            
        except Exception as e:
            print(f"Error in quality assessment pipeline: {e}")
            return {}

    def _save_detailed_summary(self, results: Dict[str, Any], output_dir: str) -> None:
        """
        Save detailed summary of quality assessment results.
        
        Args:
            results: Dictionary containing all calculated metrics
            output_dir: Directory to save the summary file
        """
        try:
            summary_file = os.path.join(output_dir, f"{self.dataset_name}_quality_assessment_summary.txt")
            os.makedirs(output_dir, exist_ok=True)
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("RAGTEC TOPIC QUALITY ASSESSMENT SUMMARY (KeyBERT)\n")
                f.write("=" * 80 + "\n\n")
                
                # Model and Configuration Information
                f.write("CONFIGURATION:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Model: {results.get('model_name', 'N/A')}\n")
                f.write(f"Keywords per topic: {results.get('top_k_keywords', 'N/A')}\n")
                f.write(f"N-gram range: {results.get('ngram_range', 'N/A')}\n")
                f.write(f"Extraction method: KeyBERT (content-based)\n\n")
                
                # Multi-topic assignment statistics
                if self.document_topic_assignments:
                    total_documents = len(self.document_topic_assignments)
                    total_assignments = sum(len(topics) if isinstance(topics, list) else 1 
                                          for topics in self.document_topic_assignments.values())
                    avg_topics_per_doc = total_assignments / total_documents if total_documents > 0 else 0
                    
                    f.write("MULTI-TOPIC ASSIGNMENT STATISTICS:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Total documents: {total_documents}\n")
                    f.write(f"Total topic assignments: {total_assignments}\n")
                    f.write(f"Average topics per document: {avg_topics_per_doc:.2f}\n")
                    
                    # Count documents by number of assigned topics
                    topic_count_distribution = {}
                    for topics in self.document_topic_assignments.values():
                        count = len(topics) if isinstance(topics, list) else 1
                        topic_count_distribution[count] = topic_count_distribution.get(count, 0) + 1
                    
                    f.write("Topic assignment distribution:\n")
                    for topic_count in sorted(topic_count_distribution.keys()):
                        doc_count = topic_count_distribution[topic_count]
                        percentage = (doc_count / total_documents) * 100
                        f.write(f"  {doc_count} documents ({percentage:.1f}%) → {topic_count} topic(s)\n")
                    f.write("\n")
                
                # Topic Overview
                f.write("TOPIC OVERVIEW:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total topics identified: {results['num_topics']}\n")
                f.write(f"Average keywords per topic: {results['avg_keywords_per_topic']:.1f}\n")
                f.write(f"Average topic name length: {results['avg_topic_name_length']:.1f} words\n\n")
                
                # Quality Metrics
                f.write("QUALITY METRICS:\n")
                f.write("-" * 40 + "\n")
                
                # Coherence Scores
                f.write("Coherence (Topic Consistency):\n")
                if results['cv_coherence'] is not None:
                    cv_score = results['cv_coherence']
                    cv_quality = self._interpret_cv_coherence(cv_score)
                    f.write(f"  • C_V Coherence: {cv_score:.3f} ({cv_quality})\n")
                else:
                    f.write("  • C_V Coherence: Not calculable\n")
                
                if results['umass_coherence'] is not None:
                    umass_score = results['umass_coherence']
                    umass_quality = self._interpret_umass_coherence(umass_score)
                    f.write(f"  • UMass Coherence: {umass_score:.3f} ({umass_quality})\n")
                else:
                    f.write("  • UMass Coherence: Not calculable\n")
                
                # Diversity Scores
                f.write("\nDiversity (Topic Distinctiveness):\n")
                if results['jaccard_diversity'] is not None:
                    jaccard_score = results['jaccard_diversity']
                    jaccard_quality = self._interpret_jaccard_diversity(jaccard_score)
                    f.write(f"  • Jaccard Diversity: {jaccard_score:.3f} ({jaccard_quality})\n")
                else:
                    f.write("  • Jaccard Diversity: Not calculable\n")
                
                if results['uwp_diversity'] is not None:
                    uwp_score = results['uwp_diversity']
                    uwp_quality = self._interpret_uwp_diversity(uwp_score)
                    f.write(f"  • Unique Word Proportion: {uwp_score:.3f} ({uwp_quality})\n")
                else:
                    f.write("  • Unique Word Proportion: Not calculable\n")
                
                # Overall Assessment
                f.write("\nOVERALL ASSESSMENT:\n")
                f.write("-" * 40 + "\n")
                overall_quality = self._get_overall_assessment(results)
                f.write(f"Quality Rating: {overall_quality['rating']}\n")
                f.write(f"Summary: {overall_quality['summary']}\n\n")
                
                # Topic Document Distribution
                if 'topic_document_counts' in results and results['topic_document_counts']:
                    f.write("TOPIC DOCUMENT DISTRIBUTION:\n")
                    f.write("-" * 40 + "\n")
                    for topic, count in results['topic_document_counts'].items():
                        f.write(f"  {topic}: {count} documents\n")
                    f.write("\n")
                
                # KeyBERT Extracted Keywords by Topic
                if hasattr(self, 'topic_keywords') and self.topic_keywords:
                    f.write("KEYBERT EXTRACTED KEYWORDS BY TOPIC:\n")
                    f.write("-" * 40 + "\n")
                    for topic_name, keywords in self.topic_keywords.items():
                        f.write(f"{topic_name}:\n")
                        if keywords:
                            f.write(f"  All Keywords ({len(keywords)}): {', '.join(keywords)}\n")
                            # Show top 10 used for evaluation
                            top_10_for_eval = keywords[:10]
                            f.write(f"  Top 10 for Evaluation: {', '.join(top_10_for_eval)}\n")
                        else:
                            f.write("  No keywords extracted\n")
                        f.write("\n")
                
                # Processing Information
                f.write("PROCESSING NOTES:\n")
                f.write("-" * 40 + "\n")
                f.write("• Keywords extracted using KeyBERT from document content\n")
                f.write("• Multi-topic assignment: Documents can belong to multiple topics\n")
                f.write("• All metrics use top 10 keywords per topic for fair method comparison\n")
                f.write("• Coherence measures topic internal consistency\n")
                f.write("• Diversity measures topic distinctiveness\n")
                f.write("• Higher coherence = more coherent topics\n")
                f.write("• Higher diversity = more distinct topics\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("End of Summary\n")
                f.write("=" * 80 + "\n")
            
            print(f"Detailed summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"Warning: Could not save detailed summary: {e}")

    def _interpret_cv_coherence(self, score: float) -> str:
        """Interpret C_V coherence score."""
        if score >= 0.5: return "Excellent"
        elif score >= 0.4: return "Good" 
        elif score >= 0.3: return "Fair"
        elif score >= 0.2: return "Poor"
        else: return "Very Poor"

    def _interpret_umass_coherence(self, score: float) -> str:
        """Interpret UMass coherence score."""
        if score >= -1: return "Excellent"
        elif score >= -2: return "Good"
        elif score >= -3: return "Fair" 
        elif score >= -4: return "Poor"
        else: return "Very Poor"

    def _interpret_jaccard_diversity(self, score: float) -> str:
        """Interpret Jaccard diversity score."""
        if score >= 0.8: return "Excellent"
        elif score >= 0.6: return "Good"
        elif score >= 0.4: return "Fair"
        elif score >= 0.2: return "Poor"
        else: return "Very Poor"

    def _interpret_uwp_diversity(self, score: float) -> str:
        """Interpret Unique Word Proportion diversity score."""
        if score >= 0.8: return "Excellent"
        elif score >= 0.6: return "Good"
        elif score >= 0.4: return "Fair"
        elif score >= 0.2: return "Poor"
        else: return "Very Poor"

    def _get_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate overall assessment based on all metrics."""
        scores = []
        
        # Normalize coherence scores (0-1 scale)
        if results['cv_coherence'] is not None:
            scores.append(min(1.0, max(0.0, results['cv_coherence'])))
        
        if results['umass_coherence'] is not None:
            # UMass is negative, normalize to 0-1 (0 = -10, 1 = 0)
            normalized_umass = max(0.0, min(1.0, (results['umass_coherence'] + 10) / 10))
            scores.append(normalized_umass)
        
        # Add diversity scores (already 0-1)
        if results['jaccard_diversity'] is not None:
            scores.append(results['jaccard_diversity'])
        
        if results['uwp_diversity'] is not None:
            scores.append(results['uwp_diversity'])
        
        if not scores:
            return {"rating": "Unable to assess", "summary": "Insufficient metrics calculated"}
        
        avg_score = np.mean(scores)
        
        if avg_score >= 0.7:
            return {"rating": "Excellent", "summary": "High-quality topics with good coherence and diversity"}
        elif avg_score >= 0.5:
            return {"rating": "Good", "summary": "Well-formed topics with reasonable quality"}
        elif avg_score >= 0.3:
            return {"rating": "Fair", "summary": "Adequate topics but room for improvement"}
        else:
            return {"rating": "Poor", "summary": "Topics need significant improvement in coherence or diversity"}

    def save_metrics(self, results: Dict[str, Any], csv_file: str, json_file: str, summary_output: bool = True):
        """
        Save the calculated metrics to CSV and JSON files.
        
        Args:
            results: Dictionary containing the calculated metrics
            csv_file: Path to save CSV metrics
            json_file: Path to save JSON results
            summary_output: Whether to save detailed summary file
        """
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        
        # Save CSV metrics with proper formatting (matching original RAGTEC format)
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("RAGTEC TOPIC QUALITY ASSESSMENT RESULTS (KEYBERT)\n")
            f.write("=" * 60 + "\n")
            
            # Main metrics section
            f.write(f"Number of Topics,{results['num_topics']}\n")
            
            # Coherence scores
            cv_str = f"{results['cv_coherence']:.3f}" if results['cv_coherence'] is not None else "N/A"
            umass_str = f"{results['umass_coherence']:.3f}" if results['umass_coherence'] is not None else "N/A"
            f.write(f"CV Coherence,{cv_str},Higher is better (0-1 range)\n")
            f.write(f"UMass Coherence,{umass_str},Less negative is better\n")
            
            # Diversity scores
            jaccard_str = f"{results['jaccard_diversity']:.3f}" if results['jaccard_diversity'] is not None else "N/A"
            uwp_str = f"{results['uwp_diversity']:.3f}" if results['uwp_diversity'] is not None else "N/A"
            f.write(f"Jaccard Diversity,{jaccard_str},Higher is better (0-1 range)\n")
            f.write(f"Unique Word Proportion,{uwp_str},Higher is better (0-1 range)\n")
            
            # Statistical metrics
            f.write(f"Avg Keywords per Topic,{results['avg_keywords_per_topic']:.1f},\n")
            f.write(f"Avg Topic Name Length,{results['avg_topic_name_length']:.1f} words,\n")
            
            # Model information
            f.write(f"KeyBERT Model,{results.get('model_name', 'all-MiniLM-L6-v2')},\n")
            f.write(f"N-gram Range,{results.get('ngram_range', '(1,1)')},\n")
            f.write(f"Keywords per Topic,{results.get('top_k_keywords', 10)},\n")
            
            # Individual topic details section
            f.write("\nINDIVIDUAL TOPIC DETAILS\n")
            f.write("Topic ID,Topic Name,Keywords Count,Top Keywords\n")
            
            if hasattr(self, 'topic_keywords') and self.topic_keywords:
                for idx, (topic_name, keywords) in enumerate(self.topic_keywords.items()):
                    keywords_str = ", ".join(keywords[:10]) if keywords else "No keywords"
                    f.write(f"{idx},{topic_name},{len(keywords)},{keywords_str}\n")
        
        print(f"Metrics saved to CSV: {csv_file}")
        
        # Save detailed JSON results
        json_results = {
            'metrics': results,
            'detailed_results': self.results,
            'topic_keywords': self.topic_keywords,
            'topic_distribution': {topic: len(docs) for topic, docs in self.topic_documents.items()},
            'master_topics': self.master_topics
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to JSON: {json_file}")
        
        # Save detailed summary if requested
        if summary_output:
            output_dir = os.path.dirname(csv_file)
            self._save_detailed_summary(results, output_dir)


def run_ragtec_quality_assessment_keybert(extraction_file: str,
                                         classification_file: str,
                                         articles_file: Optional[str] = None,
                                         model_name: str = 'all-MiniLM-L6-v2',
                                         top_k_keywords: int = 10,
                                         ngram_range: Tuple[int, int] = (1, 1)) -> Dict[str, Any]:
    """
    Convenience function to run complete RAGTEC quality assessment with KeyBERT.
    
    Args:
        extraction_file: Path to topic extraction results
        classification_file: Path to classification results
        articles_file: Path to articles file (optional)
        model_name: SentenceTransformer model name for KeyBERT. Popular options:
                   - 'all-MiniLM-L6-v2' (default, fast and lightweight)
                   - 'all-mpnet-base-v2' (higher quality, slower)
                   - 'paraphrase-MiniLM-L6-v2' (good for paraphrases)
                   - 'multi-qa-MiniLM-L6-cos-v1' (optimized for question-answering)
        top_k_keywords: Number of keywords to extract per topic (default: 10)
        ngram_range: Tuple specifying n-gram range for keyword extraction
                    (1,1)=unigrams only, (1,2)=unigrams+bigrams, (2,2)=bigrams only
        
    Returns:
        Dictionary containing calculated metrics
    """
    quality_assessor = RAGTECTopicQualityKeyBert(
        extraction_file=extraction_file,
        classification_file=classification_file,
        articles_file=articles_file,
        model_name=model_name,
        top_k_keywords=top_k_keywords,
        ngram_range=ngram_range
    )
    
    return quality_assessor.run_complete_quality_assessment()
