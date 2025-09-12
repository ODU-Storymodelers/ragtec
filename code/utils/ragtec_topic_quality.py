"""
RAGTEC: Retrieval-Augmented Generation for Topic Extraction and Classification

This module provides the quality assessment component of the RAGTEC framework 
for evaluating topic modeling results from news article analysis.

RAGTEC Quality Assessment Features:
- Comprehensive coherence metrics (CV and UMass)
- Topic diversity measures (Jaccard and Unique Word Proportion)
- Statistical analysis of topic characteristics
- Standardized result storage and export
- Integration with RAG-generated topic structures
"""

import os
import json
import numpy as np
import pandas as pd
import re
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict, Counter
from difflib import SequenceMatcher

# Gensim imports for topic quality metrics
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary


class RAGTECTopicQuality:
    """
    RAGTEC: Retrieval-Augmented Generation for Topic Quality Assessment
    
    This class implements the quality assessment component of the RAGTEC framework 
    which handles comprehensive evaluation of RAG-generated topic modeling results,
    including coherence metrics, diversity measures, and statistical analysis.
    """
    
    def __init__(self, 
                 rag_results_file: str,
                 articles_file: Optional[str] = None,
                 classification_file: Optional[str] = None):
        """
        Initialize the RAGTEC Topic Quality Assessment module.
        
        Args:
            rag_results_file: Path to the RAG topic modeling results JSON file
            articles_file: Path to the formatted articles text file for coherence calculation
            classification_file: Path to the classification results JSON file with document-topic assignments
        """
        self.rag_results_file = rag_results_file
        self.articles_file = articles_file
        self.classification_file = classification_file
        
        # State tracking
        self.rag_data = {}
        self.classification_data = {}
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
        
    def load_rag_topics_data(self) -> Dict[str, Any]:
        """
        Load the JSON file containing RAG topic modeling results.
        
        Returns:
            Dictionary containing RAG topic modeling results
        """
        print(f"Loading RAG topic modeling results from: {self.rag_results_file}")
        
        with open(self.rag_results_file, 'r', encoding='utf-8') as f:
            self.rag_data = json.load(f)
        
        print(f"Loaded RAG topic modeling results from {self.rag_results_file}")
        return self.rag_data
    
    def extract_topics_from_rag_data(self, rag_data: Optional[Dict] = None) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        """
        Extract topics and keywords from RAG topic modeling results.
        
        Args:
            rag_data: RAG data dictionary (uses self.rag_data if None)
            
        Returns:
            Tuple of (topic_keywords, topic_counts)
        """
        if rag_data is None:
            rag_data = self.rag_data
            
        if not rag_data:
            raise ValueError("No RAG data loaded. Call load_rag_topics_data() first.")
        
        topic_keywords = {}
        topic_counts = defaultdict(int)
        
        # The RAG data should have topics in the first (and likely only) entry
        if not rag_data or len(rag_data) == 0:
            raise ValueError("RAG data is empty or invalid format")
        
        # Get the topics from the first entry
        topics_entry = rag_data[0] if isinstance(rag_data, list) else rag_data
        
        if 'topics' not in topics_entry:
            raise ValueError("No 'topics' field found in RAG data")
        
        topics = topics_entry['topics']
        
        for topic_data in topics:
            topic_name = topic_data.get('topic', f"Topic_{len(topic_keywords)}")
            keywords = topic_data.get('keywords', [])
            
            # Ensure keywords are strings and clean them
            cleaned_keywords = [str(kw).strip().lower() for kw in keywords if str(kw).strip()]
            
            topic_keywords[topic_name] = cleaned_keywords
            topic_counts[topic_name] = len(cleaned_keywords)
        
        print(f"Extracted {len(topic_keywords)} topics from RAG results")
        
        # Store in class attributes
        self.topic_keywords = topic_keywords
        self.topic_counts = topic_counts
        
        # Store in results
        self.results['extraction'] = {
            'num_topics': len(topic_keywords),
            'total_keywords': sum(len(keywords) for keywords in topic_keywords.values()),
            'avg_keywords_per_topic': np.mean([len(keywords) for keywords in topic_keywords.values()]) if topic_keywords else 0
        }
        
        return topic_keywords, topic_counts
    
    def load_classification_data(self) -> Dict[str, Any]:
        """
        Load the classification results containing document-topic assignments.
        Uses multi-topic assignments from response.topics field.
        
        Returns:
            Dictionary containing classification results with document-topic assignments
        """
        if not self.classification_file:
            print("No classification file provided. Coherence will be calculated on entire corpus.")
            return {}
        
        if not os.path.exists(self.classification_file):
            print(f"Warning: Classification file not found at {self.classification_file}")
            return {}
        
        print(f"Loading classification results from: {self.classification_file}")
        
        with open(self.classification_file, 'r', encoding='utf-8') as f:
            self.classification_data = json.load(f)
        
        # Extract document-topic assignments using multi-topic approach
        self.document_topic_assignments = {}
        
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
                        # Only include valid topics (exclude no_match and similar)
                        if topic_name and topic_name.lower() not in ['no_match', 'no match', 'none', 'unmatched']:
                            doc_topics.append(topic_name)
                    elif isinstance(topic_item, str):
                        topic_name = topic_item.strip()
                        if topic_name and topic_name.lower() not in ['no_match', 'no match', 'none', 'unmatched']:
                            doc_topics.append(topic_name)
                
                if doc_topics:
                    # Store multiple topics for this document
                    self.document_topic_assignments[idx] = doc_topics
        
        print(f"Loaded {len(self.document_topic_assignments)} valid document-topic assignments (multi-topic)")
        
        # Count excluded documents
        total_docs = len(self.classification_data)
        excluded_docs = total_docs - len(self.document_topic_assignments)
        if excluded_docs > 0:
            print(f"Excluded {excluded_docs} documents with 'no_match' or invalid topic assignments")
        
        return self.classification_data
    
    def load_article_texts(self, separator: str = None) -> List[str]:
        """
        Load article texts for coherence calculation.
        Supports both JSON and text file formats.
        
        Args:
            separator: Separator used between articles (for text files)
            
        Returns:
            List of article content texts
        """
        if not self.articles_file:
            raise ValueError("No articles file provided. Coherence metrics cannot be calculated.")
        
        print(f"Loading article texts from: {self.articles_file}")
        
        article_contents = []
        
        # Check if it's a JSON file
        if self.articles_file.endswith('.json'):
            # Handle JSON format
            with open(self.articles_file, 'r', encoding='utf-8') as f:
                articles_data = json.load(f)
            
            # Extract content from each article
            for article in articles_data:
                if isinstance(article, dict):
                    # Try different possible content fields
                    content = None
                    if 'data' in article and isinstance(article['data'], dict):
                        content = article['data'].get('content')
                    elif 'content' in article:
                        content = article['content']
                    elif 'text' in article:
                        content = article['text']
                    
                    if content and isinstance(content, str) and content.strip():
                        article_contents.append(content.strip())
        
        else:
            # Handle text file format (original implementation)
            if separator is None:
                separator = "\n" + "="*80 + "\n"
                
            with open(self.articles_file, 'r', encoding='utf-8') as f:
                articles_text = f.read()
            
            # Split articles by separator
            articles = [a.strip() for a in articles_text.split(separator) if a.strip()]
            
            # Extract just the content part of each article (after the metadata)
            for article in articles:
                lines = article.split('\n')
                content_lines = []
                
                # Skip metadata lines and extract content
                metadata_ended = False
                for line in lines:
                    line_stripped = line.strip()
                    
                    # Skip empty lines at the beginning
                    if not line_stripped and not metadata_ended:
                        continue
                    
                    # Check if this is a metadata line (contains ':' and is at the beginning)
                    is_metadata = (
                        line_stripped.startswith('Title:') or
                        line_stripped.startswith('Source:') or
                        line_stripped.startswith('Publication Date:') or
                        line_stripped.startswith('URL:') or
                        line_stripped.startswith('Section:') or
                        line_stripped.startswith('Image:') or
                        line_stripped.startswith('Content:')
                    )
                    
                    # If we hit an empty line after metadata, content starts next
                    if not metadata_ended and line_stripped == '' and len(content_lines) == 0:
                        metadata_ended = True
                        continue
                    
                    # If not metadata and we haven't started content yet, this is content
                    if not is_metadata:
                        metadata_ended = True
                        if line_stripped:  # Don't add empty lines at the start of content
                            content_lines.append(line)
                    elif metadata_ended:
                        # We've hit metadata again, so we're done with this article's content
                        break
                
                # Clean up content and add if valid
                if content_lines:
                    content_text = '\n'.join(content_lines).strip()
                    if content_text and len(content_text) > 50:  # Only add substantial content
                        article_contents.append(content_text)
        
        self.article_texts = article_contents
        self.collection_size = len(article_contents)
        
        print(f"Loaded {len(article_contents)} article texts for coherence calculation")
        
        # Store in results
        self.results['articles'] = {
            'total_articles': len(article_contents),
            'avg_article_length': np.mean([len(text) for text in article_contents]) if article_contents else 0
        }
        
        return article_contents
    
    def prepare_data_for_coherence(self) -> Tuple[Optional[Dictionary], Optional[List], Optional[List], Optional[List]]:
        """
        Prepare data structures needed for coherence calculation.
        
        Returns:
            Tuple of (dictionary, corpus, tokenized_texts, topic_word_lists)
        """
        if not self.article_texts:
            raise ValueError("No article texts loaded. Call load_article_texts() first.")
        
        if not self.topic_keywords:
            raise ValueError("No topic keywords extracted. Call extract_topics_from_rag_data() first.")
        
        # Tokenize all documents
        print("Tokenizing texts...")
        tokenized_texts = []
        for text in self.article_texts:
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
            valid_keywords = []
            invalid_keywords = []
            
            # Process keywords
            for keyword in keywords[:10]:  # Top 10 keywords
                if keyword in dictionary.token2id:
                    valid_keywords.append(keyword)
                    keywords_in_dict += 1
                else:
                    # Try individual words from multi-word keywords
                    keyword_parts = keyword.split()
                    keyword_valid = False
                    
                    if len(keyword_parts) > 1:
                        for part in keyword_parts:
                            if part in dictionary.token2id and part not in valid_keywords:
                                valid_keywords.append(part)
                                keywords_in_dict += 1
                                keyword_valid = True
                    
                    if not keyword_valid:
                        invalid_keywords.append(keyword)
                
                total_keywords += 1
            
            # Display validation results
            print(f"Topic {topic_name}:")
            print(f"  - Valid keywords ({len(valid_keywords)}): {', '.join(valid_keywords)}")
            if invalid_keywords:
                print(f"  - Invalid keywords ({len(invalid_keywords)}): {', '.join(invalid_keywords)}")
            
            if valid_keywords:
                topic_word_lists.append(valid_keywords)
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
        Calculate CV and UMass coherence scores for topics.
        Uses topic-specific documents if classification data is available,
        otherwise uses entire corpus.
        
        Returns:
            Tuple of (cv_coherence, umass_coherence)
        """
        if not self.topic_word_lists or len(self.topic_word_lists) < 2:
            print("Warning: Not enough valid topics for coherence calculation.")
            return None, None
        
        if not self.dictionary or len(self.dictionary) == 0 or not self.corpus:
            print("Warning: Dictionary or corpus is empty. Cannot compute coherence.")
            return None, None
        
        # Determine texts to use for coherence calculation
        texts_to_use = self.tokenized_texts
        calculation_method = "entire corpus"
        
        # Use topic-specific documents if classification data is available
        if self.document_topic_assignments and self.classification_data:
            try:
                texts_to_use = self._get_topic_specific_texts()
                calculation_method = "topic-specific documents"
                print(f"Using {calculation_method} for coherence calculation")
            except Exception as e:
                print(f"Warning: Could not use topic-specific texts ({e}). Falling back to entire corpus.")
                texts_to_use = self.tokenized_texts
                calculation_method = "entire corpus (fallback)"
        
        print(f"Coherence calculation method: {calculation_method}")
        
        # Calculate CV coherence
        try:
            print("Calculating CV coherence...")
            print(f"Topics for coherence: {len(self.topic_word_lists)}")
            print(f"Texts for coherence: {len(texts_to_use)}")
            print(f"Dictionary size: {len(self.dictionary)}")
            
            # Validate topic word lists
            valid_topic_lists = []
            for i, topic_words in enumerate(self.topic_word_lists):
                # Check if words exist in dictionary and have sufficient frequency
                valid_words = []
                for word in topic_words:
                    if word in self.dictionary.token2id:
                        word_id = self.dictionary.token2id[word]
                        # Check word frequency in corpus
                        word_freq = sum(1 for doc in self.corpus for token_id, freq in doc if token_id == word_id)
                        if word_freq >= 2:  # Word must appear at least twice
                            valid_words.append(word)
                
                if len(valid_words) >= 3:  # Need at least 3 valid words for coherence
                    valid_topic_lists.append(valid_words[:10])  # Limit to top 10 words
                    print(f"Topic {i}: {len(valid_words)} valid words ({valid_words[:5]}...)")
                else:
                    print(f"Skipping topic {i}: only {len(valid_words)} valid words (need at least 3)")
            
            print(f"Valid topics for coherence: {len(valid_topic_lists)} out of {len(self.topic_word_lists)}")
            
            if len(valid_topic_lists) < 2:
                print("Warning: Not enough valid topics with sufficient keywords for CV coherence.")
                cv_score = None
            else:
                # Additional validation: check for word co-occurrence
                print("Validating word co-occurrence patterns...")
                sufficient_cooccurrence = True
                total_cooccurrences = 0
                
                for topic_idx, topic_words in enumerate(valid_topic_lists):
                    word_pairs = [(topic_words[i], topic_words[j]) 
                                 for i in range(len(topic_words)) 
                                 for j in range(i+1, len(topic_words))]
                    
                    cooccurrence_count = 0
                    for word1, word2 in word_pairs[:10]:  # Check first 10 pairs
                        if word1 in self.dictionary.token2id and word2 in self.dictionary.token2id:
                            id1, id2 = self.dictionary.token2id[word1], self.dictionary.token2id[word2]
                            # Count documents where both words appear
                            cooccur_docs = sum(1 for doc in self.corpus 
                                             if any(token_id == id1 for token_id, _ in doc) and 
                                                any(token_id == id2 for token_id, _ in doc))
                            if cooccur_docs > 0:
                                cooccurrence_count += 1
                    
                    total_cooccurrences += cooccurrence_count
                    print(f"Topic {topic_idx}: {cooccurrence_count} co-occurring word pairs out of {min(10, len(word_pairs))} checked")
                    
                    if cooccurrence_count < 1:  # Relaxed requirement: at least 1 co-occurring pair
                        print(f"Topic with words {topic_words[:3]}... has insufficient co-occurrence")
                        sufficient_cooccurrence = False
                
                print(f"Total co-occurrences found: {total_cooccurrences}")
                print(f"Co-occurrence validation passed: {sufficient_cooccurrence}")
                
                if not sufficient_cooccurrence:
                    print("Warning: Insufficient word co-occurrence for reliable CV coherence calculation.")
                    cv_score = None
                else:
                    # Suppress runtime warnings from Gensim's coherence calculation
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        
                        try:
                            coherence_cv = CoherenceModel(
                                topics=valid_topic_lists, 
                                texts=texts_to_use, 
                                dictionary=self.dictionary, 
                                coherence='c_v',
                                processes=1,
                                window_size=20  # Increase window size for better co-occurrence detection
                            )
                            cv_score = coherence_cv.get_coherence()
                            
                            # Check for invalid values
                            if np.isnan(cv_score) or np.isinf(cv_score) or cv_score is None:
                                print("Warning: CV coherence calculation resulted in invalid value (NaN/Inf/None).")
                                print("This often occurs when topics have insufficient word co-occurrence.")
                                cv_score = None
                            else:
                                print(f"Successfully calculated CV coherence score: {cv_score}")
                        except (ZeroDivisionError, ValueError, FloatingPointError) as math_error:
                            print(f"Warning: Mathematical error in CV coherence calculation: {math_error}")
                            print("This suggests insufficient word relationships in the corpus for coherence measurement.")
                            cv_score = None
                        except Exception as gensim_error:
                            print(f"Warning: Gensim coherence calculation failed: {gensim_error}")
                            print("Trying alternative coherence calculation method...")
                            
                            # Alternative method: simplified coherence calculation
                            try:
                                # Use only the most frequent words and a smaller window
                                simplified_topics = []
                                for topic_words in valid_topic_lists:
                                    # Keep only the top 5 most frequent words
                                    word_frequencies = []
                                    for word in topic_words[:8]:  # Check top 8 words
                                        if word in self.dictionary.token2id:
                                            word_id = self.dictionary.token2id[word]
                                            freq = sum(freq for doc in self.corpus for token_id, freq in doc if token_id == word_id)
                                            word_frequencies.append((word, freq))
                                    
                                    # Sort by frequency and take top 5
                                    word_frequencies.sort(key=lambda x: x[1], reverse=True)
                                    simplified_topic = [word for word, freq in word_frequencies[:5]]
                                    
                                    if len(simplified_topic) >= 3:
                                        simplified_topics.append(simplified_topic)
                                
                                if len(simplified_topics) >= 2:
                                    coherence_cv_alt = CoherenceModel(
                                        topics=simplified_topics, 
                                        texts=texts_to_use, 
                                        dictionary=self.dictionary, 
                                        coherence='c_v',
                                        processes=1,
                                        window_size=10  # Smaller window
                                    )
                                    cv_score = coherence_cv_alt.get_coherence()
                                    
                                    if np.isnan(cv_score) or np.isinf(cv_score) or cv_score is None:
                                        print("Alternative CV coherence calculation also failed.")
                                        cv_score = None
                                    else:
                                        print(f"Successfully calculated CV coherence score using alternative method: {cv_score}")
                                else:
                                    print("Alternative method also insufficient - not enough valid simplified topics.")
                                    cv_score = None
                            except Exception as alt_error:
                                print(f"Alternative coherence calculation also failed: {alt_error}")
                                cv_score = None
                    
        except Exception as e:
            print(f"Error calculating CV coherence: {e}")
            cv_score = None
        
        # Calculate UMass coherence
        try:
            print("Calculating UMass coherence...")
            
            if len(valid_topic_lists) < 2:
                print("Warning: Not enough valid topics with sufficient keywords for UMass coherence.")
                umass_score = None
            else:
                coherence_umass = CoherenceModel(
                    topics=valid_topic_lists, 
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
            'calculation_method': calculation_method
        }
        
        return cv_score, umass_score
    
    def _get_topic_specific_texts(self) -> List[List[str]]:
        """
        Get topic-specific tokenized texts for more accurate coherence calculation.
        Now supports multi-topic document assignments.
        
        Returns:
            List of tokenized texts filtered by topic assignments
        """
        if not self.document_topic_assignments:
            return self.tokenized_texts
        
        # Group documents by topic (multi-topic support)
        topic_documents = defaultdict(list)
        
        for doc_idx, topics in self.document_topic_assignments.items():
            if doc_idx < len(self.tokenized_texts):
                # Handle both single topic (string) and multi-topic (list) assignments
                if isinstance(topics, str):
                    topics = [topics]  # Convert single topic to list
                
                for topic in topics:
                    if topic in self.topic_keywords:
                        topic_documents[topic].append(self.tokenized_texts[doc_idx])
        
        # Combine all topic-specific documents
        all_topic_texts = []
        for topic, docs in topic_documents.items():
            all_topic_texts.extend(docs)
        
        if not all_topic_texts:
            print("Warning: No topic-specific texts found. Using entire corpus.")
            return self.tokenized_texts
        
        print(f"Using {len(all_topic_texts)} topic-specific documents for coherence calculation")
        return all_topic_texts
    
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
                words = keywords[:min(len(keywords), top_n)]
                topic_terms.append(set(words))
            
            if len(topic_terms) < 2:
                return 0.0
            
            jaccard_distances = []
            for i in range(len(topic_terms)):
                for j in range(i+1, len(topic_terms)):
                    intersection = len(topic_terms[i].intersection(topic_terms[j]))
                    union = len(topic_terms[i].union(topic_terms[j]))
                    distance = 1 - (intersection / union if union > 0 else 0)
                    jaccard_distances.append(distance)
            
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
        Calculate additional statistical metrics for RAG topics.
        
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
        
        # Store in results
        self.results['statistical'] = metrics
        
        return metrics
    
    def run_complete_quality_assessment(self) -> Dict[str, Any]:
        """
        Run the complete RAGTEC quality assessment pipeline.
        
        Returns:
            Comprehensive results dictionary with all quality metrics
        """
        print("=" * 60)
        print("STARTING RAGTEC QUALITY ASSESSMENT")
        print("Topic Quality Assessment for RAG-Generated Topics")
        print("=" * 60)
        
        # Step 1: Load RAG topic data
        print("\n1. Loading RAG topic modeling results...")
        rag_data = self.load_rag_topics_data()
        
        # Step 2: Extract topics and keywords
        print("\n2. Extracting topics from RAG data...")
        self.extract_topics_from_rag_data(rag_data)
        
        if not self.topic_keywords:
            raise ValueError("Error: No topics found in RAG data.")
        
        num_topics = len(self.topic_keywords)
        print(f"Found {num_topics} topics for quality assessment")
        
        # Step 2.5: Load classification data if available
        print("\n2.5. Loading classification data...")
        self.load_classification_data()
        
        # Step 3: Load article texts if available for coherence calculation
        coherence_cv_score = None
        coherence_umass_score = None
        
        if self.articles_file:
            print("\n3. Loading article texts for coherence calculation...")
            self.load_article_texts()
            
            print("\n4. Preparing data for coherence metrics...")
            self.prepare_data_for_coherence()
            
            print("\n5. Calculating coherence metrics...")
            coherence_cv_score, coherence_umass_score = self.calculate_coherence_metrics()
        else:
            print("\n3. Skipping coherence calculation (no articles file provided)")
        
        # Step 4: Calculate diversity metrics
        print("\n6. Calculating diversity metrics...")
        uwp_diversity_score = self.calculate_unique_word_proportion(top_n=10)
        jaccard_diversity_score = self.calculate_jaccard_diversity(top_n=10)
        
        # Step 5: Calculate statistical metrics
        print("\n7. Calculating statistical metrics...")
        statistical_metrics = self.calculate_statistical_metrics()
        
        # Step 6: Compile comprehensive results
        print("\n8. Compiling final results...")
        
        # Round scores to 3 decimal places for consistency
        final_results = {
            'num_topics': num_topics,
            'cv_coherence': round(coherence_cv_score, 3) if coherence_cv_score is not None else None,
            'umass_coherence': round(coherence_umass_score, 3) if coherence_umass_score is not None else None,
            'jaccard_diversity': round(jaccard_diversity_score, 3) if jaccard_diversity_score is not None else None,
            'uwp_diversity': round(uwp_diversity_score, 3) if uwp_diversity_score is not None else None,
            'avg_keywords_per_topic': round(statistical_metrics.get('avg_keywords_per_topic', 0), 1),
            'avg_topic_name_length': round(statistical_metrics.get('avg_topic_name_length', 0), 1),
            'min_keywords_per_topic': statistical_metrics.get('min_keywords_per_topic', 0),
            'max_keywords_per_topic': statistical_metrics.get('max_keywords_per_topic', 0),
            'topic_keywords': self.topic_keywords,
            'statistical_metrics': statistical_metrics
        }
        
        # Store in class results
        self.results['final_assessment'] = final_results
        
        print("\n" + "=" * 60)
        print("RAGTEC QUALITY ASSESSMENT COMPLETED")
        print("Quality Assessment Finished")
        print("=" * 60)
        
        return final_results
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calculate all quality metrics for RAG topics.
        
        Returns:
            Dictionary containing all calculated metrics
        """
        return self.run_complete_quality_assessment()
    
    def save_assessment_results(self, 
                              results: Dict[str, Any], 
                              output_csv: str, 
                              output_json: str,
                              output_summary: str = None) -> None:
        """
        Save the calculated quality assessment results to CSV, JSON, and summary text files.
        
        Args:
            results: Dictionary containing all assessment results
            output_csv: Path to save CSV file
            output_json: Path to save JSON file
            output_summary: Path to save detailed summary text file (optional)
        """
        if not results:
            raise ValueError("No results to save.")
        
        # Ensure output directories exist
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        if output_summary:
            os.makedirs(os.path.dirname(output_summary), exist_ok=True)
        
        print(f"Saving quality assessment results...")
        
        # Save metrics to CSV
        with open(output_csv, 'w', encoding='utf-8') as f:
            f.write("RAGTEC TOPIC QUALITY ASSESSMENT RESULTS\n")
            f.write("="*50 + "\n")
            f.write(f"Number of Topics,{results['num_topics']}\n")
            
            # Write coherence scores
            cv_str = f"{results['cv_coherence']:.3f}" if results['cv_coherence'] is not None else "N/A"
            umass_str = f"{results['umass_coherence']:.3f}" if results['umass_coherence'] is not None else "N/A"
            jaccard_str = f"{results['jaccard_diversity']:.3f}" if results['jaccard_diversity'] is not None else "N/A"
            uwp_str = f"{results['uwp_diversity']:.3f}" if results['uwp_diversity'] is not None else "N/A"
            
            f.write(f"CV Coherence,{cv_str},Higher is better (0-1 range)\n")
            f.write(f"UMass Coherence,{umass_str},Less negative is better\n")
            f.write(f"Jaccard Diversity,{jaccard_str},Higher is better (0-1 range)\n")
            f.write(f"Unique Word Proportion,{uwp_str},Higher is better (0-1 range)\n")
            f.write(f"Avg Keywords per Topic,{results['avg_keywords_per_topic']:.1f},\n")
            f.write(f"Avg Topic Name Length,{results['avg_topic_name_length']:.1f} words,\n\n")
            
            # Add individual topic details
            f.write("INDIVIDUAL TOPIC DETAILS\n")
            f.write("Topic ID,Topic Name,Keywords Count,Top Keywords\n")
            
            for i, (topic_name, keywords) in enumerate(results['topic_keywords'].items()):
                keyword_count = len(keywords)
                top_keywords = ", ".join(keywords[:10])
                safe_topic_name = str(topic_name).replace(",", ";")
                f.write(f"{i},{safe_topic_name},{keyword_count},{top_keywords}\n")
        
        # Save as JSON
        assessment_json = {
            "assessment_info": {
                "framework": "RAGTEC_Quality_Assessment",
                "version": "1.0",
                "model_type": "RAG_Topic_Modeling",
                "num_topics": results['num_topics'],
                "timestamp": Path(self.rag_results_file).stat().st_mtime if os.path.exists(self.rag_results_file) else None
            },
            "quality_metrics": {
                "coherence": {
                    "cv_coherence": results['cv_coherence'],
                    "umass_coherence": results['umass_coherence']
                },
                "diversity": {
                    "jaccard_diversity": results['jaccard_diversity'],
                    "uwp_diversity": results['uwp_diversity']
                },
                "statistical": {
                    "avg_keywords_per_topic": results['avg_keywords_per_topic'],
                    "avg_topic_name_length": results['avg_topic_name_length'],
                    "min_keywords_per_topic": results.get('min_keywords_per_topic', 0),
                    "max_keywords_per_topic": results.get('max_keywords_per_topic', 0)
                }
            },
            "topics": [
                {
                    "topic_id": i,
                    "topic_name": topic_name,
                    "keywords_count": len(keywords),
                    "keywords": keywords[:10],
                    "all_keywords": keywords
                }
                for i, (topic_name, keywords) in enumerate(results['topic_keywords'].items())
            ]
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(assessment_json, f, indent=2, ensure_ascii=False)
        
        # Save detailed summary if requested
        if output_summary:
            self._save_detailed_summary(results, output_summary)
        
        print(f"Quality assessment results saved:")
        print(f"  CSV: {output_csv}")
        print(f"  JSON: {output_json}")
        if output_summary:
            print(f"  Summary: {output_summary}")

    def _save_detailed_summary(self, results: Dict[str, Any], output_summary: str) -> None:
        """
        Save detailed execution summary with all diagnostic information.
        
        Args:
            results: Dictionary containing all assessment results
            output_summary: Path to save summary text file
        """
        with open(output_summary, 'w', encoding='utf-8') as f:
            f.write("RAGTEC TOPIC QUALITY ASSESSMENT - DETAILED EXECUTION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic information
            f.write("ASSESSMENT INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Framework: RAGTEC Quality Assessment\n")
            f.write(f"Assessment Type: RAG-Generated Topics with LLM Keywords\n")
            f.write(f"Number of Topics: {results['num_topics']}\n")
            f.write(f"RAG Results File: {self.rag_results_file}\n")
            f.write(f"Classification File: {self.classification_file}\n")
            f.write(f"Articles File: {self.articles_file}\n\n")
            
            # Data loading summary
            f.write("DATA LOADING SUMMARY\n")
            f.write("-" * 40 + "\n")
            if hasattr(self, 'classification_data') and self.classification_data:
                f.write(f"Classification Results: Loaded from {self.classification_file}\n")
                f.write(f"Valid Document-Topic Assignments: {len(self.document_topic_assignments)} (multi-topic)\n")
                total_docs = len(self.classification_data)
                excluded_docs = total_docs - len(self.document_topic_assignments)
                f.write(f"Excluded Documents: {excluded_docs} with 'no_match' or invalid topic assignments\n")
            
            if hasattr(self, 'article_texts') and self.article_texts:
                f.write(f"Article Texts: Loaded {len(self.article_texts)} articles for coherence calculation\n")
            f.write("\n")
            
            # Multi-topic assignment statistics
            if hasattr(self, 'document_topic_assignments') and self.document_topic_assignments:
                total_documents = len(self.document_topic_assignments)
                total_assignments = sum(len(topics) if isinstance(topics, list) else 1 
                                      for topics in self.document_topic_assignments.values())
                avg_topics_per_doc = total_assignments / total_documents if total_documents > 0 else 0
                
                f.write("MULTI-TOPIC ASSIGNMENT STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Documents with valid topic assignments: {total_documents}\n")
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
                    f.write(f"  {doc_count} documents ({percentage:.1f}%) assigned to {topic_count} topic(s)\n")
                f.write("\n")
            
            # Data preparation details
            if hasattr(self, 'results') and 'preparation' in self.results:
                prep_results = self.results['preparation']
                f.write("DATA PREPARATION DETAILS\n")
                f.write("-" * 40 + "\n")
                f.write("Tokenization and Dictionary Building:\n")
                f.write(f"  Original dictionary size: {prep_results.get('original_dict_size', 'N/A')}\n")
                f.write(f"  Filtered dictionary size: {prep_results.get('filtered_dict_size', 'N/A')}\n")
                removed_terms = prep_results.get('original_dict_size', 0) - prep_results.get('filtered_dict_size', 0)
                f.write(f"  Removed terms: {removed_terms}\n")
                f.write(f"  Corpus size: {prep_results.get('corpus_size', 'N/A')} documents\n")
                f.write(f"  Keyword coverage: {prep_results.get('keyword_coverage', 0):.1%}\n\n")
            
            # Topic keyword validation details
            f.write("KEYWORD VALIDATION DETAILS\n")
            f.write("-" * 40 + "\n")
            if hasattr(self, 'topic_keywords') and self.topic_keywords:
                total_keywords = 0
                valid_keywords = 0
                
                for topic_name, keywords in self.topic_keywords.items():
                    f.write(f"Topic {topic_name}:\n")
                    f.write(f"  Keywords ({len(keywords[:10])}): {', '.join(keywords[:10])}\n")
                    total_keywords += len(keywords[:10])
                    
                    # Count valid keywords if dictionary exists
                    if hasattr(self, 'dictionary') and self.dictionary:
                        valid_count = sum(1 for kw in keywords[:10] if kw in self.dictionary.token2id)
                        valid_keywords += valid_count
                        f.write(f"  Valid in dictionary: {valid_count}/{len(keywords[:10])}\n")
                    f.write("\n")
                
                if hasattr(self, 'dictionary') and self.dictionary:
                    f.write(f"Overall Keyword Validation:\n")
                    f.write(f"  Total keywords checked: {total_keywords}\n")
                    f.write(f"  Valid keywords: {valid_keywords}\n")
                    f.write(f"  Validation rate: {valid_keywords/total_keywords:.1%}\n\n")
            
            # Coherence calculation details
            if hasattr(self, 'results') and 'coherence' in self.results:
                coh_results = self.results['coherence']
                f.write("COHERENCE CALCULATION DETAILS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Calculation method: {coh_results.get('calculation_method', 'N/A')}\n")
                f.write(f"Topics used for coherence: {coh_results.get('num_topics_used', 'N/A')}\n")
                
                if hasattr(self, 'document_topic_assignments'):
                    # Calculate topic-specific document count
                    topic_doc_count = 0
                    for topics in self.document_topic_assignments.values():
                        if isinstance(topics, list):
                            topic_doc_count += len(topics)
                        else:
                            topic_doc_count += 1
                    f.write(f"Topic-specific documents used: {topic_doc_count}\n")
                f.write("\n")
            
            # Quality metrics summary
            f.write("QUALITY METRICS SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            # Coherence metrics
            f.write("Coherence Metrics:\n")
            if results['cv_coherence'] is not None:
                f.write(f"  C_V Coherence Score: {results['cv_coherence']:.3f}\n")
                f.write(f"    Interpretation: {'Good' if results['cv_coherence'] > 0.3 else 'Fair' if results['cv_coherence'] > 0.2 else 'Poor'}\n")
            else:
                f.write(f"  C_V Coherence Score: Not calculable\n")
            
            if results['umass_coherence'] is not None:
                f.write(f"  UMass Coherence Score: {results['umass_coherence']:.3f}\n")
                f.write(f"    Interpretation: {'Good' if results['umass_coherence'] > -1 else 'Fair' if results['umass_coherence'] > -2 else 'Poor'}\n")
            else:
                f.write(f"  UMass Coherence Score: Not calculable\n")
            f.write("\n")
            
            # Diversity metrics
            f.write("Diversity Metrics:\n")
            if results['jaccard_diversity'] is not None:
                f.write(f"  Jaccard Diversity Score: {results['jaccard_diversity']:.3f}\n")
                f.write(f"    Interpretation: {'Excellent' if results['jaccard_diversity'] > 0.8 else 'Good' if results['jaccard_diversity'] > 0.6 else 'Fair'}\n")
            else:
                f.write(f"  Jaccard Diversity Score: Not calculable\n")
            
            if results['uwp_diversity'] is not None:
                f.write(f"  Unique Word Proportion Score: {results['uwp_diversity']:.3f}\n")
                f.write(f"    Interpretation: {'Excellent' if results['uwp_diversity'] > 0.8 else 'Good' if results['uwp_diversity'] > 0.6 else 'Fair'}\n")
            else:
                f.write(f"  Unique Word Proportion Score: Not calculable\n")
            f.write("\n")
            
            # Statistical metrics
            f.write("Statistical Metrics:\n")
            f.write(f"  Average Keywords per Topic: {results['avg_keywords_per_topic']:.1f}\n")
            f.write(f"  Average Topic Name Length: {results['avg_topic_name_length']:.1f} words\n")
            if 'statistical_metrics' in results:
                stats = results['statistical_metrics']
                f.write(f"  Min Keywords per Topic: {stats.get('min_keywords_per_topic', 'N/A')}\n")
                f.write(f"  Max Keywords per Topic: {stats.get('max_keywords_per_topic', 'N/A')}\n")
            f.write("\n")
            
            # Topic distribution if available
            if hasattr(self, 'document_topic_assignments') and self.document_topic_assignments:
                f.write("TOPIC DOCUMENT DISTRIBUTION\n")
                f.write("-" * 40 + "\n")
                topic_doc_counts = defaultdict(int)
                for topics in self.document_topic_assignments.values():
                    if isinstance(topics, list):
                        for topic in topics:
                            topic_doc_counts[topic] += 1
                    else:
                        topic_doc_counts[topics] += 1
                
                for topic, count in sorted(topic_doc_counts.items()):
                    f.write(f"  {topic}: {count} documents\n")
                f.write("\n")
            
            # Assessment conclusions
            f.write("ASSESSMENT CONCLUSIONS\n")
            f.write("-" * 40 + "\n")
            f.write("Overall Quality Assessment:\n")
            
            # Calculate overall quality score
            quality_scores = []
            if results['cv_coherence'] is not None:
                quality_scores.append(results['cv_coherence'])
            if results['jaccard_diversity'] is not None:
                quality_scores.append(results['jaccard_diversity'])
            if results['uwp_diversity'] is not None:
                quality_scores.append(results['uwp_diversity'])
            
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                f.write(f"  Average Quality Score: {avg_quality:.3f}\n")
                if avg_quality > 0.7:
                    f.write(f"  Overall Rating: Excellent topic quality\n")
                elif avg_quality > 0.5:
                    f.write(f"  Overall Rating: Good topic quality\n")
                elif avg_quality > 0.3:
                    f.write(f"  Overall Rating: Fair topic quality\n")
                else:
                    f.write(f"  Overall Rating: Poor topic quality - consider revision\n")
            else:
                f.write(f"  Overall Rating: Unable to calculate - insufficient valid metrics\n")
            
            f.write(f"\nAssessment completed successfully with {results['num_topics']} topics analyzed.\n")
    
    def save_metrics(self, results: Dict[str, Any], output_csv: str, output_json: str, output_summary: str = None) -> None:
        """
        Legacy method for backward compatibility.
        
        Args:
            results: Dictionary containing all assessment results
            output_csv: Path to save CSV file
            output_json: Path to save JSON file
            output_summary: Path to save detailed summary text file (optional)
        """
        self.save_assessment_results(results, output_csv, output_json, output_summary)


# Convenience function for simple usage
def run_ragtec_quality_assessment(rag_results_file: str,
                                 articles_file: str = None,
                                 output_csv: str = None,
                                 output_json: str = None,
                                 output_summary: str = None,
                                 save_results: bool = True) -> Dict[str, Any]:
    """
    Convenience function to run RAGTEC quality assessment with file paths.
    
    Args:
        rag_results_file: Path to RAG topic modeling results JSON file
        articles_file: Path to formatted articles text file (optional)
        output_csv: Path to save CSV results (required if save_results=True)
        output_json: Path to save JSON results (required if save_results=True)
        output_summary: Path to save detailed summary text file (optional)
        save_results: Whether to save results to files
        
    Returns:
        Complete quality assessment results dictionary
    """
    # Initialize quality assessment
    quality_assessor = RAGTECTopicQuality(
        rag_results_file=rag_results_file,
        articles_file=articles_file
    )
    
    # Run complete assessment
    results = quality_assessor.run_complete_quality_assessment()
    
    # Save results if requested
    if save_results:
        if not output_csv or not output_json:
            raise ValueError("output_csv and output_json are required when save_results=True")
        quality_assessor.save_assessment_results(results, output_csv, output_json, output_summary)
    
    return results


if __name__ == "__main__":
    """
    Example usage of the RAGTEC Quality Assessment framework.
    """
    # Define file paths
    dataset_name = "mozambique"  # Change this to use different datasets
    rag_results_file = f"../output/{dataset_name}/ragtec/extraction/{dataset_name}_ragtec_topic_results.json"
    articles_file = f"../data/{dataset_name}/{dataset_name}_articles_formatted.txt"
    output_csv = f"../output/{dataset_name}/ragtec/quality/{dataset_name}_metrics_ragtec_topics.csv"
    output_json = f"../output/{dataset_name}/ragtec/quality/{dataset_name}_metrics_ragtec_topics.json"
    output_summary = f"../output/{dataset_name}/ragtec/quality/{dataset_name}_quality_assessment_summary.txt"
    
    print("=" * 60)
    print("RAGTEC QUALITY ASSESSMENT - STANDALONE EXECUTION")
    print("=" * 60)
    
    try:
        # Run quality assessment using convenience function with summary
        results = run_ragtec_quality_assessment(
            rag_results_file=rag_results_file,
            articles_file=articles_file,
            output_csv=output_csv,
            output_json=output_json,
            output_summary=output_summary,
            save_results=True
        )
        
        # Print summary
        print(f"\n=== FINAL QUALITY ASSESSMENT SUMMARY ===")
        print(f"Number of Topics: {results['num_topics']}")
        print(f"CV Coherence: {results['cv_coherence']:.3f}" if results['cv_coherence'] is not None else "CV Coherence: Not calculable")
        print(f"UMass Coherence: {results['umass_coherence']:.3f}" if results['umass_coherence'] is not None else "UMass Coherence: Not calculable")
        print(f"Jaccard Diversity: {results['jaccard_diversity']:.3f}" if results['jaccard_diversity'] is not None else "Jaccard Diversity: Not calculable")
        print(f"UWP Diversity: {results['uwp_diversity']:.3f}" if results['uwp_diversity'] is not None else "UWP Diversity: Not calculable")
        print(f"Avg Keywords per Topic: {results['avg_keywords_per_topic']:.1f}")
        print(f"Avg Topic Name Length: {results['avg_topic_name_length']:.1f} words")
        
    except Exception as e:
        print(f"Error during quality assessment: {e}")
        raise