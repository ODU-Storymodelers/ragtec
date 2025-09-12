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

from utils.ragtec_topic_quality import RAGTECTopicQuality
from utils.ragtec_topic_classification import run_ragtec_topic_classification
import dotenv

# Load environment variables
dotenv.load_dotenv()

if __name__ == "__main__":
    # Define file paths
    dataset_name = "mozambique"
    rag_results_file = f"../output/{dataset_name}/ragtec/extraction/{dataset_name}_ragtec_topic_results.json"
    articles_file = f"../data/{dataset_name}/{dataset_name}_articles_formatted.txt"
    classification_file = f"../output/{dataset_name}/ragtec/classification/{dataset_name}_ragtec_classification_results.json"

    output_dir = f"../output/{dataset_name}/ragtec/quality"

    output_metrics_file = f"{output_dir}/{dataset_name}_metrics_ragtec_topics.csv"
    json_output = f"{output_dir}/{dataset_name}_metrics_ragtec_topics.json"

    # Check if classification file exists
    if not os.path.exists(classification_file):
        print(f"Classification file not found at {classification_file}")
        print("Creating classification results using RAGTEC classification pipeline...")
        
        # Create output directory if it doesn't exist
        classification_output_dir = f"../output/{dataset_name}/ragtec/classification"
        os.makedirs(classification_output_dir, exist_ok=True)
        
        # Define required paths
        articles_gnews_file = f"../data/{dataset_name}/{dataset_name}_articles_gnews.json"
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
            
            # Update classification file path to the created file (fixed path)
            classification_file = f"../output/{dataset_name}/ragtec/classification/{dataset_name}_ragtec_classification_results.json"
            print(f"Classification created successfully: {classification_file}")
            
        except Exception as e:
            print(f"Error creating classification: {e}")
            print("Proceeding without classification data...")
            classification_file = None
    else:
        print(f"Using existing classification file: {classification_file}")

    # Create RAGTECTopicQuality instance
    rag_quality = RAGTECTopicQuality(
        rag_results_file=rag_results_file,
        articles_file=articles_file,
        classification_file=classification_file
    )
    
    # Calculate all metrics
    print("=== CALCULATING RAG TOPIC MODELING QUALITY METRICS ===")
    results = rag_quality.calculate_all_metrics()
    
    if not results:
        print("Error: Could not calculate metrics. Exiting.")
        exit(1)
    
    # Print results with multi-topic statistics
    print(f"\n=== RAG TOPIC MODELING QUALITY METRICS ===")
    print(f"Number of Topics: {results['num_topics']}")
    
    # Add multi-topic statistics
    if hasattr(rag_quality, 'document_topic_assignments') and rag_quality.document_topic_assignments:
        total_documents = len(rag_quality.document_topic_assignments)
        total_assignments = sum(len(topics) if isinstance(topics, list) else 1 
                              for topics in rag_quality.document_topic_assignments.values())
        avg_topics_per_doc = total_assignments / total_documents if total_documents > 0 else 0
        
        print(f"\n=== MULTI-TOPIC ASSIGNMENT STATISTICS ===")
        print(f"Documents with valid topic assignments: {total_documents}")
        print(f"Total topic assignments: {total_assignments}")
        print(f"Average topics per document: {avg_topics_per_doc:.2f}")
        
        # Count documents by number of assigned topics
        topic_count_distribution = {}
        for topics in rag_quality.document_topic_assignments.values():
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
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(output_metrics_file), exist_ok=True)
    os.makedirs(os.path.dirname(json_output), exist_ok=True)
    
    # Define summary output file path
    summary_output_file = f"{output_dir}/{dataset_name}_quality_assessment_summary.txt"

    # Save metrics with detailed summary
    rag_quality.save_metrics(results, output_metrics_file, json_output, summary_output_file)
    
    print("\n=== COMPLETED SUCCESSFULLY ===")
    print(f"CSV metrics saved to: {output_metrics_file}")
    print(f"JSON metrics saved to: {json_output}")
    print(f"Detailed summary saved to: {summary_output_file}")

