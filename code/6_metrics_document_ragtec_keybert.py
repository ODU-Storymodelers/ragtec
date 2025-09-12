import json
import os
import sys
import numpy as np
from collections import defaultdict
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import pandas as pd
import re
from difflib import SequenceMatcher

from utils.ragtec_topic_quality_keybert import RAGTECTopicQualityKeyBert, run_ragtec_quality_assessment_keybert
from utils.ragtec_topic_classification import run_ragtec_topic_classification
import dotenv

# Load environment variables
dotenv.load_dotenv()

if __name__ == "__main__":
    # Define file paths
    dataset_name = "burundi"  # Change to your dataset name
    classification_file = f"../output/{dataset_name}/ragtec/classification/{dataset_name}_ragtec_classification_results.json"

    output_dir = f"../output/{dataset_name}/ragtec/quality-keybert"
    output_metrics_file = f"{output_dir}/{dataset_name}_metrics_ragtec_keybert.csv"
    json_output = f"{output_dir}/{dataset_name}_metrics_ragtec_keybert.json"

    # Check if classification file exists
    if not os.path.exists(classification_file):
        print(f"Classification file not found at {classification_file}")
        print("Creating classification results using RAGTEC classification pipeline...")
        
        # Define required paths for classification creation
        rag_results_file = f"../output/{dataset_name}/ragtec/extraction/{dataset_name}_ragtec_topic_results.json"
        articles_gnews_file = f"../data/{dataset_name}/{dataset_name}_articles_gnews.json"
        
        # Create output directory if it doesn't exist
        classification_output_dir = f"../output/{dataset_name}/ragtec/classification"
        os.makedirs(classification_output_dir, exist_ok=True)
        
        # Define required paths
        prompt_path = "../prompts/topic_classification_prompt.txt"
        
        # Run classification
        try:
            classification_results = run_ragtec_topic_classification(
                articles_file=articles_gnews_file,
                topics_file=rag_results_file,
                prompt_path=prompt_path,
                output_dir=classification_output_dir,
                model="gpt-4o-mini",
                save_results=True,
                batch_save_interval=10,
                dataset_name=dataset_name
            )
            
            # Update classification file path to the created file
            classification_file = f"../output/{dataset_name}/ragtec/classification/{dataset_name}_ragtec_classification_results.json"
            print(f"Classification created successfully: {classification_file}")
            
        except Exception as e:
            print(f"Error creating classification: {e}")
            print("Cannot proceed without classification data. Exiting.")
            exit(1)
    else:
        print(f"Using existing classification file: {classification_file}")

    # Run quality assessment using KeyBERT approach
    print("=== CALCULATING RAG TOPIC MODELING QUALITY METRICS WITH KEYBERT ===")
    
    # User-configurable parameters for KeyBERT (as requested by user)
    KEYBERT_MODEL = 'all-MiniLM-L6-v2'  # User parameter: SentenceTransformer model
    KEYWORDS_PER_TOPIC = 20              # User parameter: Number of keywords per topic  
    NGRAM_RANGE = (1, 1)                 # User parameter: (1,1)=unigrams only, (1,2)=unigrams+bigrams
    
    print(f"KeyBERT Configuration:")
    print(f"  Model: {KEYBERT_MODEL}")
    print(f"  Keywords per topic: {KEYWORDS_PER_TOPIC}")
    print(f"  N-gram range: {NGRAM_RANGE}")
    
    # Define required files for KeyBERT quality assessment
    extraction_file = f"../output/{dataset_name}/ragtec/extraction/{dataset_name}_ragtec_topic_results.json"
    
    # Check if extraction file exists
    if not os.path.exists(extraction_file):
        print(f"Extraction file not found at {extraction_file}")
        print("Please run the topic extraction step first.")
        exit(1)
    
    try:
        # Initialize KeyBERT quality assessor
        from utils.ragtec_topic_quality_keybert import RAGTECTopicQualityKeyBert
        
        quality_assessor = RAGTECTopicQualityKeyBert(
            extraction_file=extraction_file,
            classification_file=classification_file,
            model_name=KEYBERT_MODEL,
            top_k_keywords=KEYWORDS_PER_TOPIC,
            ngram_range=NGRAM_RANGE
        )
        
        # Run complete assessment
        results = quality_assessor.run_complete_quality_assessment()
        
        # Save results
        if results:
            quality_assessor.save_metrics(results, output_metrics_file, json_output, summary_output=True)
    except Exception as e:
        print(f"Error calculating quality metrics: {e}")
        exit(1)
    
    if not results:
        print("Error: Could not calculate metrics. Exiting.")
        exit(1)
    
    # Print results with multi-topic statistics
    print(f"\n=== RAG TOPIC MODELING QUALITY METRICS (KEYBERT) ===")
    print(f"Number of Topics: {results['num_topics']}")
    print(f"KeyBERT Model: {results.get('model_name', 'N/A')}")
    print(f"Extraction Method: {results.get('extraction_method', 'KeyBERT')}")
    
    # Add multi-topic statistics
    if hasattr(quality_assessor, 'document_topic_assignments') and quality_assessor.document_topic_assignments:
        total_documents = len(quality_assessor.document_topic_assignments)
        total_assignments = sum(len(topics) if isinstance(topics, list) else 1 
                              for topics in quality_assessor.document_topic_assignments.values())
        avg_topics_per_doc = total_assignments / total_documents if total_documents > 0 else 0
        
        print(f"\n=== MULTI-TOPIC ASSIGNMENT STATISTICS ===")
        print(f"Documents with valid topic assignments: {total_documents}")
        print(f"Total topic assignments: {total_assignments}")
        print(f"Average topics per document: {avg_topics_per_doc:.2f}")
        
        # Count documents by number of assigned topics
        topic_count_distribution = {}
        for topics in quality_assessor.document_topic_assignments.values():
            count = len(topics) if isinstance(topics, list) else 1
            topic_count_distribution[count] = topic_count_distribution.get(count, 0) + 1
        
        print("Topic assignment distribution:")
        for topic_count in sorted(topic_count_distribution.keys()):
            doc_count = topic_count_distribution[topic_count]
            percentage = (doc_count / total_documents) * 100
            print(f"  {doc_count} documents ({percentage:.1f}%) assigned to {topic_count} topic(s)")
    
    print(f"\n=== COHERENCE AND DIVERSITY METRICS ===")
    
    if results['cv_coherence'] is not None:
        print(f"C_V Coherence Score: {results['cv_coherence']:.3f}")
    else:
        print("C_V Coherence Score: Not calculable")
    
    if results['umass_coherence'] is not None:
        print(f"UMass Coherence Score: {results['umass_coherence']:.3f}")
    else:
        print("UMass Coherence Score: Not calculable")
    
    if results['jaccard_diversity'] is not None:
        print(f"Jaccard Diversity Score: {results['jaccard_diversity']:.3f}")
    else:
        print("Jaccard Diversity Score: Not calculable")
    
    if results['uwp_diversity'] is not None:
        print(f"Unique Word Proportion Score: {results['uwp_diversity']:.3f}")
    else:
        print("UWP Diversity Score: Not calculable")
    
    print(f"Average Keywords per Topic: {results['avg_keywords_per_topic']:.1f}")
    print(f"Average Topic Name Length: {results['avg_topic_name_length']:.1f} words")
    
    # Print topic distribution if available
    if 'topic_document_counts' in results:
        print(f"\nTopic Document Distribution:")
        for topic, count in results['topic_document_counts'].items():
            print(f"  {topic}: {count} documents")
    
    # Create output directories if they don't exist (already handled by convenience function)
    os.makedirs(os.path.dirname(output_metrics_file), exist_ok=True)
    os.makedirs(os.path.dirname(json_output), exist_ok=True)
    
    print("\n=== COMPLETED SUCCESSFULLY ===")
    print(f"CSV metrics saved to: {output_metrics_file}")
    print(f"JSON metrics saved to: {json_output}")

