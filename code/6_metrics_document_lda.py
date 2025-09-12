import json
import os
import numpy as np
from collections import defaultdict
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return []
    # Convert to lowercase and remove special characters
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words and len(w) > 2]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

def load_data(json_file_path):
    """Load the JSON file containing articles with topic assignments"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} articles from {json_file_path}")
    return data

def extract_documents_by_topic(data):
    """Extract documents assigned to each topic and topic keywords"""
    topic_docs = defaultdict(list)
    topic_keywords = {}
    all_content = []
    
    # Process each article
    for article in data:
        if 'content' not in article or not article['content']:
            continue
            
        content = article['content']
        all_content.append(content)
        
        # Find topic assignment and keywords
        if 'topic_info' in article and article['topic_info']:
            topic_id = article['topic_info'].get('topic_id')
            keywords = article['topic_info'].get('keywords', [])
            
            if topic_id is not None:
                # Add document to topic collection
                topic_docs[topic_id].append(content)
                
                # Store keywords for topic if not already stored
                if topic_id not in topic_keywords:
                    topic_keywords[topic_id] = keywords
    
    return topic_docs, topic_keywords, all_content

def calculate_coherence(topic_keywords, all_content):
    """Calculate coherence scores for topics using UMass and CV coherence measures"""
    try:
        # Check if we have any topics with keywords
        if not topic_keywords or all(not keywords for keywords in topic_keywords.values()):
            print("No valid topic keywords found for coherence calculation")
            return None, None
        
        # Process all documents
        processed_docs = [clean_text(doc) for doc in all_content]
        
        # Filter out empty documents
        processed_docs = [doc for doc in processed_docs if doc]
        
        if not processed_docs:
            print("Warning: No valid processed documents for coherence calculation")
            return None, None
        
        # Create dictionary from all texts
        dictionary = Dictionary(processed_docs)
        
        # Filter extreme words - matching LDA script parameters
        dictionary.filter_extremes(no_below=2, no_above=0.8)
        
        # Get topic word lists, ensuring keywords exist
        topic_word_lists = []
        for keywords in topic_keywords.values():
            if keywords:  # Only include non-empty keyword lists
                topic_word_lists.append(keywords[:10])  # Use top 10 keywords for consistency
        
        if not topic_word_lists:
            print("No valid topic keyword lists found")
            return None, None
        
        # Calculate UMass coherence
        try:
            coherence_umass = CoherenceModel(
                topics=topic_word_lists,
                texts=processed_docs,
                dictionary=dictionary,
                coherence='u_mass',
                processes=1
            )
            umass_score = coherence_umass.get_coherence()
            print(f"UMass coherence calculated: {umass_score}")
        except Exception as e:
            print(f"Error calculating UMass coherence: {e}")
            umass_score = None
        
        # Calculate CV coherence
        try:
            coherence_cv = CoherenceModel(
                topics=topic_word_lists,
                texts=processed_docs,
                dictionary=dictionary,
                coherence='c_v',
                processes=1
            )
            cv_score = coherence_cv.get_coherence()
            print(f"CV coherence calculated: {cv_score}")
        except Exception as e:
            print(f"Error calculating CV coherence: {e}")
            cv_score = None
        
        return umass_score, cv_score
    
    except Exception as e:
        print(f"Error in coherence calculation: {e}")
        return None, None

def calculate_jaccard_diversity(topic_keywords, top_n=10):
    """Calculate topic diversity score using Jaccard distance - matching LDA script approach"""
    try:
        if not topic_keywords:
            return None
            
        # Convert to sets of words for each topic
        topic_terms = []
        for keywords in topic_keywords.values():
            if keywords:  # Only include non-empty keyword lists
                words = keywords[:min(len(keywords), top_n)]
                topic_terms.append(set(words))
        
        if len(topic_terms) < 2:
            print("Need at least 2 topics to calculate diversity")
            return None
        
        # Calculate average Jaccard distance between all topic pairs
        jaccard_distances = []
        for i in range(len(topic_terms)):
            for j in range(i+1, len(topic_terms)):
                intersection = len(topic_terms[i].intersection(topic_terms[j]))
                union = len(topic_terms[i].union(topic_terms[j]))
                distance = 1 - (intersection / union if union > 0 else 0)
                jaccard_distances.append(distance)
                
        diversity_score = sum(jaccard_distances) / len(jaccard_distances) if jaccard_distances else 0
        return diversity_score
    
    except Exception as e:
        print(f"Error calculating Jaccard diversity: {e}")
        return None

def calculate_unique_word_proportion(topic_keywords, top_n=10):
    """Compute topic diversity as the proportion of unique words across topics."""
    try:
        if not topic_keywords:
            return None
            
        unique_words = set()
        total_words = 0
        
        for keywords in topic_keywords.values():
            if keywords:  # Only include non-empty keyword lists
                words = keywords[:min(len(keywords), top_n)]
                unique_words.update(words)
                total_words += len(words)
        
        diversity_score = len(unique_words) / total_words if total_words > 0 else 0
        return diversity_score
    
    except Exception as e:
        print(f"Error calculating UWP diversity: {e}")
        return None

if __name__ == "__main__":
    # Define file paths
    dataset_name = "mozambique"
    input_file = f"../output/{dataset_name}/lda/Mozambique_articles_topic_document_lda.json"
    country = os.path.basename(input_file).split("_")[0]
    output_dir = f"../output/{dataset_name}/lda"
    
    output_file = f"{output_dir}/{country}_document_lda_metrics.csv"
    json_output = f"{output_dir}/{country}_document_lda_metrics.json"
    
    # Load data
    data = load_data(input_file)
    
    # Extract documents by topic and get keywords
    topic_docs, topic_keywords, all_content = extract_documents_by_topic(data)
    
    # Calculate metrics
    umass_score, cv_score = calculate_coherence(topic_keywords, all_content)
    jaccard_diversity = calculate_jaccard_diversity(topic_keywords, top_n=10)
    uwp_diversity = calculate_unique_word_proportion(topic_keywords, top_n=10)
    
    # Round scores to 3 decimal places
    umass_score = round(umass_score, 3) if umass_score is not None else "N/A"
    cv_score = round(cv_score, 3) if cv_score is not None else "N/A"
    jaccard_diversity = round(jaccard_diversity, 3) if jaccard_diversity is not None else "N/A"
    uwp_diversity = round(uwp_diversity, 3) if uwp_diversity is not None else "N/A"
    
    print(f"Number of Topics: {len(topic_keywords)}")
    print(f"UMass Coherence Score: {umass_score}")
    print(f"CV Coherence Score: {cv_score}")
    print(f"Jaccard Diversity Score: {jaccard_diversity}")
    print(f"Unique Word Proportion: {uwp_diversity}")
    
    # Save metrics to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        f.write("OVERALL METRICS\n")
        f.write(f"Number of Topics,{len(topic_keywords)}\n")
        f.write(f"UMass Coherence,{umass_score},Closer to 0 is better\n")
        f.write(f"CV Coherence,{cv_score},Higher is better\n")
        f.write(f"Jaccard Diversity,{jaccard_diversity},Higher is better\n")
        f.write(f"Unique Word Proportion,{uwp_diversity},Higher is better\n\n")
        
        # Add individual topic details
        f.write("INDIVIDUAL TOPIC METRICS\n")
        f.write("Topic ID,Topic Name,Document Count,Top Words\n")
        
        # Write each topic's details
        for topic_id, keywords in topic_keywords.items():
            count = len(topic_docs[topic_id])
            name = f"Topic {topic_id}"
            top_words_str = ", ".join(keywords[:10]) if keywords else ""
            
            f.write(f"{topic_id},{name},{count},{top_words_str}\n")
    
    print(f"Saved metrics to {output_file}")
    
    # Also save as JSON for easier programmatic access
    metrics_json = {
        "model_info": {
            "num_topics": len(topic_keywords),
            "umass_coherence": umass_score,
            "cv_coherence": cv_score,
            "jaccard_diversity": jaccard_diversity,
            "uwp_diversity": uwp_diversity
        },
        "topics": [
            {
                "id": topic_id,
                "name": f"Topic {topic_id}",
                "document_count": len(topic_docs[topic_id]),
                "keywords": keywords
            }
            for topic_id, keywords in topic_keywords.items()
        ]
    }
    
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)
    
    print(f"Saved JSON metrics to {json_output}")